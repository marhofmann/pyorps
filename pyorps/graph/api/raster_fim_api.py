"""
Eikonal (fast iterative method) graph API — continuous least-cost routing.

Unlike the discrete backends, this backend solves the *continuous* problem
``|grad T| = c(x)`` directly on the cost raster (GPU block-FIM) and traces
paths by steepest descent on the T field. There is no neighborhood and no
metrication (elongation) bias: costs satisfy ``T <= discrete cost`` on the
same raster, with O(h) discretization error instead of a fixed directional
bias (2.79% worst-case for the default R2 neighborhood).

Semantics that differ from the discrete backends — by design:

- ``steps`` / neighborhood are accepted and IGNORED (the PDE has no
  neighborhood); a debug log notes this.
- The authoritative routing metric is ``T[target]`` (``last_field_costs``).
  PathFinder's edge-based recompute over the rasterized cell path
  re-quantizes the continuous path and will differ slightly (upward) —
  the known dual-metric reporting issue applies here too.
- Continuous polylines (float row/col) are kept on ``last_polylines``
  (source -> target order) for the GUI/reporting; the returned node paths
  are supercover rasterizations for the existing Path machinery.
- Multi-target from one source costs ONE solve (the discrete backends pay
  per pair); multi-source to one target uses the isotropic symmetry
  (field solved from the target).
- DEM / gradient LUTs raise: slope responses are anisotropic — the
  isotropic solver cannot honor them, and silently ignoring them would be
  a silent-correctness bug (see plan 2026-08-06 section 10 for the
  anisotropic increment).

Requirements: NVIDIA GPU with CUDA support (cupy-cuda12x >= 13.0.0).

Usage:
    pf = PathFinder(..., graph_api="raster_fim")
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
from numpy import ndarray

from pyorps.core.exceptions import (
    NoPathFoundError, AlgorithmNotImplementedError, PairwiseError
)
from pyorps.core.types import NodeList, NodePathList
from pyorps.graph.api.graph_api import GraphAPI

try:
    from pyorps.utils.eikonal_gpu import (
        FINITE_LIMIT,
        eikonal_raster_gpu,
        polyline_to_cells,
        trace_paths_gpu,
    )
    from pyorps.utils.traversal_gpu import GPU_AVAILABLE as _gpu_flag
    RASTER_FIM_AVAILABLE = _gpu_flag
except ImportError:
    RASTER_FIM_AVAILABLE = False

logger = logging.getLogger(__name__)

_ACCEPTED_ALGORITHMS = ("fim", "eikonal", "dijkstra")


class RasterFIMAPI(GraphAPI):
    """Graph API backed by the GPU eikonal / block-FIM solver.

    The raster IS the continuous cost field — no graph object, no edge
    list, no neighborhood.
    """

    def __init__(
            self,
            raster_data: ndarray,
            steps: ndarray,
            ignore_max: bool = True,
            dem_data: Optional[ndarray] = None,
            tile: int = 16,
            n_inner: Optional[int] = None,
            eps_rel: float = 1e-6,
            eps_abs: Optional[float] = None,
            disk_init: bool = True,
            max_outer_iterations: Optional[int] = None,
            order: int = 1,
            gradient_luts=None,
            **kwargs,
    ):
        if not RASTER_FIM_AVAILABLE:
            raise ImportError(
                "The raster_fim backend requires CuPy with CUDA support. "
                "Install with: pip install cupy-cuda12x"
            )
        if gradient_luts is not None or dem_data is not None:
            # Slope responses are direction-dependent (anisotropic
            # eikonal); the isotropic solver cannot honor them.
            raise AlgorithmNotImplementedError(
                "anisotropic eikonal (DEM / gradient slope terms — see "
                "docs/superpowers/plans/2026-08-06-eikonal-fim-gpu-"
                "backend.md section 10)",
                graph_library="raster_fim",
            )

        super().__init__(raster_data, steps, ignore_max, dem_data)
        logger.debug(
            "raster_fim ignores the neighborhood (steps/neighborhood_str)"
            " — the eikonal PDE is continuous; changing the neighborhood "
            "changes nothing.")

        self._solver_kwargs = dict(
            tile=tile, n_inner=n_inner, eps_rel=eps_rel, eps_abs=eps_abs,
            disk_init=disk_init,
            max_outer_iterations=max_outer_iterations,
            order=order,
        )
        rows, cols = raster_data.shape[:2]
        self._rows, self._cols = int(rows), int(cols)

        # Raster-direct: no graph construction
        self.edge_construction_time = 0.0
        self.graph_creation_time = 0.0
        self.graph = None

        #: Continuous polylines (float (row, col), source -> target
        #: order) of the last shortest_path call, parallel to the
        #: returned path list (None for unreachable pairs).
        self.last_polylines: List[Optional[np.ndarray]] = []
        #: Authoritative routing metric T[target] per returned path
        #: (np.inf for unreachable pairs).
        self.last_field_costs: List[float] = []
        # Trace field of the most recent solve (GPU tracer input; the
        # first-order field when order=2 — see _solve)
        self._last_host_field: Optional[np.ndarray] = None
        self._last_trace_host: Optional[np.ndarray] = None
        self._last_trace_device = None

    # ------------------------------------------------------------------
    # Solve + trace helpers
    # ------------------------------------------------------------------

    def _solve(self, field_sources: np.ndarray,
               target_index: Optional[int] = None) -> np.ndarray:
        """One eikonal solve; T field for the given field sources.

        The trace field of the most recent solve is kept for the GPU
        tracer (matched by identity in ``_paths_from_field`` — older
        cached host fields simply re-upload inside the tracer). With
        order=2 the trace field is the preserved FIRST-order field:
        refined fields carry the costs but are not descent-connected
        (genuine local minima near cost shocks on rough rasters).
        ``target_index`` enables the exact targeted early exit — only
        passed for single-pair solves whose field is used once.
        """
        t_field, _d_t, (t_trace, d_trace) = eikonal_raster_gpu(
            self.raster_data, field_sources,
            ignore_max=self.ignore_max, return_device=True,
            return_trace_field=True,
            target_index=target_index, **self._solver_kwargs)
        self._last_host_field = t_field
        self._last_trace_host = t_trace
        self._last_trace_device = d_trace
        return t_field

    def _paths_from_field(
            self,
            t_field: np.ndarray,
            field_sources: np.ndarray,
            trace_starts: np.ndarray,
            reverse: bool,
    ) -> List[Optional[NodeList]]:
        """Trace descent paths for ``trace_starts`` on one T field.

        With ``reverse=True`` the field sources are the routing sources
        (polylines are flipped to run source -> target); with False the
        field was solved from the routing target and each traced start is
        a routing source (already source -> target).
        """
        if t_field is self._last_host_field:
            trace_host = self._last_trace_host
            trace_dev = self._last_trace_device
        else:                          # older cached field: re-upload
            trace_host, trace_dev = t_field, None
        polylines = trace_paths_gpu(trace_host, field_sources,
                                    trace_starts, t_device=trace_dev)
        forbidden = t_field >= FINITE_LIMIT
        results: List[Optional[NodeList]] = []
        for poly in polylines:
            if poly is None:
                self.last_polylines.append(None)
                results.append(None)
                continue
            if reverse:
                poly = poly[::-1]
            self.last_polylines.append(poly)
            results.append(polyline_to_cells(
                poly, self._rows, self._cols, forbidden_mask=forbidden))
        return results

    def _field_cost(self, t_field: np.ndarray, cell: int) -> float:
        value = float(t_field.ravel()[int(cell)])
        return value if value < FINITE_LIMIT else float("inf")

    # ------------------------------------------------------------------
    # Source/target case handlers (RasterGPUAPI semantics)
    # ------------------------------------------------------------------

    def _single_to_single(self, source, target):
        """Lean single-pair path (performance plan phase 1): the solve
        skips the full-field D2H; T[target] is a device scalar read,
        tracing runs device-only, and only the forbidden mask (uint8,
        a quarter of the float field) crosses the bus."""
        source, target = int(source), int(target)
        src_arr = np.array([source])
        _t_none, d_t, (_tr_none, d_trace) = eikonal_raster_gpu(
            self.raster_data, src_arr,
            ignore_max=self.ignore_max, return_device=True,
            return_trace_field=True, target_index=target,
            download=False, **self._solver_kwargs)
        cost = float(d_t[target])
        self.last_field_costs.append(
            cost if cost < FINITE_LIMIT else float("inf"))
        poly = trace_paths_gpu(
            None, src_arr, [target], t_device=d_trace,
            shape=(self._rows, self._cols))[0]
        if poly is None:
            self.last_polylines.append(None)
            raise NoPathFoundError(source=source, target=target)
        poly = poly[::-1]                      # source -> target order
        self.last_polylines.append(poly)
        forbidden = (d_trace >= FINITE_LIMIT).get().reshape(
            self._rows, self._cols)
        return polyline_to_cells(poly, self._rows, self._cols,
                                 forbidden_mask=forbidden)

    def _single_to_multi(self, source, targets):
        t_field = self._solve(np.array([int(source)]))
        paths = self._paths_from_field(
            t_field, np.array([int(source)]),
            np.asarray(targets, dtype=np.int64), reverse=True)
        for t in targets:
            self.last_field_costs.append(self._field_cost(t_field, t))
        return [p if p is not None else [] for p in paths]

    def _multi_to_single(self, sources, target):
        # Isotropic costs are symmetric: one field from the target, then
        # trace each source down to it (polylines already run
        # source -> target).
        t_field = self._solve(np.array([int(target)]))
        paths = self._paths_from_field(
            t_field, np.array([int(target)]),
            np.asarray(sources, dtype=np.int64), reverse=False)
        for s in sources:
            self.last_field_costs.append(self._field_cost(t_field, s))
        return [p if p is not None else [] for p in paths]

    def _multi_to_multi(self, sources, targets, pairwise):
        if pairwise and len(sources) != len(targets):
            raise PairwiseError()
        results: List[NodeList] = []
        pairs = (zip(sources, targets) if pairwise
                 else ((s, t) for s in sources for t in targets))
        # One solve per unique source, reused across its pairs
        fields: dict = {}
        for s, t in pairs:
            s = int(s)
            if s not in fields:
                fields[s] = self._solve(np.array([s]))
            t_field = fields[s]
            paths = self._paths_from_field(
                t_field, np.array([s]), np.array([int(t)]), reverse=True)
            self.last_field_costs.append(self._field_cost(t_field, t))
            results.append(paths[0] if paths[0] is not None else [])
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shortest_path(
            self,
            source_indices: Union[int, List[int], ndarray],
            target_indices: Union[int, List[int], ndarray],
            algorithm: str = "fim",
            **kwargs,
    ) -> Union[NodeList, NodePathList]:
        """Continuous least-cost path(s) via the eikonal field.

        Parameters:
            source_indices: Source node index/indices (flat raster cells)
            target_indices: Target node index/indices
            algorithm: "fim" / "eikonal" (also accepts "dijkstra" — the
                PathFinder default — as an alias for the least-cost solve)
            **kwargs: pairwise (bool) for pairwise computation

        Returns:
            Path or list of paths as node index lists. Continuous
            polylines and T[target] costs are kept on ``last_polylines``
            and ``last_field_costs``.
        """
        if algorithm.lower() not in _ACCEPTED_ALGORITHMS:
            raise AlgorithmNotImplementedError(
                algorithm, graph_library="raster_fim")

        self.last_polylines = []
        self.last_field_costs = []

        source_has_len = hasattr(source_indices, "__len__")
        target_has_len = hasattr(target_indices, "__len__")

        if not source_has_len and not target_has_len:
            return self._single_to_single(source_indices, target_indices)
        if not source_has_len and target_has_len:
            return self._single_to_multi(source_indices, target_indices)
        if source_has_len and not target_has_len:
            return self._multi_to_single(source_indices, target_indices)
        return self._multi_to_multi(source_indices, target_indices,
                                    kwargs.get("pairwise", False))

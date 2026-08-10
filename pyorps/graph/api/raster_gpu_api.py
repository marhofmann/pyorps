"""
GPU-accelerated graph API using raster-direct delta-stepping SSSP.

This backend extends GraphAPI directly (like CythonAPI) and operates on the
raster grid without constructing an explicit edge list or graph object.
Neighbors are computed on-the-fly during frontier expansion.

Phase 1b of the real-world performance plan
(``docs/superpowers/plans/2026-08-07-realworld-5km-performance-plan.md``)
turned this class from a *stateless* dispatcher into the owner of one
device-resident :class:`~pyorps.utils.sssp_gpu.GpuSsspSession`:

* **1.2** the raster, the DEM, the step tables, the auto-delta and the
  ``dist``/``pred``/arena buffers are uploaded and allocated **once** per
  API object instead of once per query;
* **1.3/1.4** path queries download only the path chains and the target
  costs (via the on-device predecessor walk) instead of the full
  ``dist``+``pred`` pair, and repair only the predecessor links on those
  chains instead of every vertex in the window;
* **1.5** a k-pair job runs **one solve per distinct source** instead of
  one per pair (361 solves -> 19 on the 19x19 meshed substation matrix);
* **1.6** a k-source -> 1-target query runs **one reversed solve**, not k.

None of that changes an answer: see ``_solve_paths`` and
``_shortest_path_multi_to_single`` for the exactness arguments.

The session holds ~14 B/cell of device memory for as long as this object
lives. That is deliberate and it is the caller's to manage: drop the API
object, or call :meth:`close`, and every device buffer is released (the
CuPy block cache is handed back to the driver too -- this is a shared
6 GB card).

Requirements:
    - NVIDIA GPU with CUDA support
    - cupy-cuda12x >= 13.0.0

Usage:
    pf = PathFinder(..., graph_api="raster_gpu")
"""

from __future__ import annotations

from typing import Union, List, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray

from pyorps.core.exceptions import (
    NoPathFoundError, AlgorithmNotImplementedError, PairwiseError
)
from pyorps.core.types import NodeList, NodePathList
from pyorps.graph.api.graph_api import GraphAPI

# Availability check — only depends on CuPy (no cuGraph/cuDF)
try:
    from pyorps.utils.sssp_gpu import GpuSsspSession, sssp_raster_gpu
    from pyorps.utils.traversal_gpu import GPU_AVAILABLE as _gpu_flag
    RASTER_GPU_AVAILABLE = _gpu_flag
except ImportError:  # pragma: no cover - exercised only without CuPy
    GpuSsspSession = None
    sssp_raster_gpu = None
    RASTER_GPU_AVAILABLE = False

#: ``dist`` values at or above this are the kernel's "unreachable"
#: sentinel. The device already rewrites them to ``inf`` on download; the
#: comparison is kept so a host walk over a raw field behaves the same.
_UNREACHABLE = 1e29


def _walk_predecessors(
        pred: np.ndarray,
        dist: np.ndarray,
        source: int,
        target: int,
        n_nodes: int,
) -> Optional[List[int]]:
    """Host predecessor walk: ``source -> target`` node list, or None.

    The CPU twin of the on-device ``v5_walk_pred`` walk. It is reached
    only on the V4/V3 fallback (no V5 => no session => no device walk)
    and from :meth:`RasterGPUAPI._extract_path`, which tests and external
    callers still use against a downloaded field.

    Returns ``None`` for exactly the cases the device walk reports as
    "no path": target out of range, unreachable target, a broken
    predecessor link, or a cycle.
    """
    source = int(source)
    target = int(target)
    if target < 0 or target >= len(dist):
        return None
    if np.isinf(dist[target]) or dist[target] >= _UNREACHABLE:
        return None

    path: List[int] = []
    current = target
    max_steps = n_nodes
    visited = set()

    while current != source and max_steps > 0:
        path.append(current)
        if current in visited:
            return None
        visited.add(current)

        p = pred[current]
        if p < 0:
            return None
        current = int(p)
        max_steps -= 1

    if current != source:
        return None
    path.append(source)
    path.reverse()
    return path


class RasterGPUAPI(GraphAPI):
    """
    Graph API using raster-direct GPU delta-stepping SSSP.

    No edge list construction, no graph object. The raster IS the implicit
    graph — neighbor costs are computed on-the-fly during frontier expansion.
    """

    #: Whether the kernel's edge weights satisfy ``w(u, v) == w(v, u)``.
    #:
    #: This is what lets a k-source -> 1-target query run as ONE reversed
    #: solve (plan item 1.6). It holds for every edge model the V5/V4
    #: kernels implement today:
    #:
    #: * terrain term ``(raster[u] + raster[v] + sum(intermediates)) *
    #:   cost_factor[s]`` — the endpoint sum is symmetric, the
    #:   Bresenham intermediate *set* of step ``s`` from ``u`` is the set
    #:   of step ``-s`` from ``v`` (``floor(-x) == -ceil(x)`` pairs the
    #:   two entries per rung and reverses the rung order), and
    #:   ``cost_factor[s] == cost_factor[-s]`` since it is
    #:   ``|s| / (2 + n_inter)``;
    #: * gradient term — the kernel bins ``fabsf(dem[v] - dem[u])`` and
    #:   applies the *same* LUT pair in both directions
    #:   (``sssp_gpu.py:828, 1274, 1378``).
    #:
    #: A bare ``dem_data`` without ``gradient_luts`` never reaches the
    #: kernel at all (see ``_gradient_dem``), so it cannot break symmetry
    #: either — that is the over-broad guard item 1.6 removed.
    #:
    #: **A directional slope response** (signed ``dem[v] - dem[u]``, i.e.
    #: uphill priced differently from downhill — the eikonal increment-3
    #: lineage) would make the model a Finsler metric and invalidate all
    #: of the above. Whoever lands it must set this to ``False`` for the
    #: affected configurations;
    #: ``tests/test_graph/test_phase1b_gpu_dispatch.py`` pins both the
    #: symmetry and the ``False`` fallback.
    symmetric_edge_weights: bool = True

    def __init__(
            self,
            raster_data: ndarray,
            steps: ndarray,
            ignore_max: bool = True,
            dem_data: Optional[ndarray] = None,
            delta: Union[float, str] = "auto",
            margin: float = 1.00001,
            gradient_luts=None,
            arena_factor: float = 1.0,
            **kwargs,
    ):
        if not RASTER_GPU_AVAILABLE:
            raise ImportError(
                "Raster GPU backend requires CuPy with CUDA support. "
                "Install with: pip install cupy-cuda12x"
            )

        super().__init__(raster_data, steps, ignore_max, dem_data)

        # Per-edge gradient terms (feasibility plan): only active when the
        # slope-response LUT pair is provided — a bare dem_data keeps the
        # legacy behavior (it merely disables the symmetric-search trick).
        self.gradient_luts = gradient_luts
        if gradient_luts is not None and dem_data is None:
            raise ValueError(
                "gradient_luts require dem_data aligned to the raster")

        self._delta = delta
        self._margin = margin
        self._arena_factor = float(arena_factor)
        self._n_nodes = int(raster_data.shape[0] * raster_data.shape[1])

        # Item 1.6 safety net: reversing a search is only sound when every
        # step has its negation in the set. PathFinder builds raster_gpu
        # neighborhoods with directed=True, which is closed under
        # negation; a hand-passed half step-set is not, and reversing it
        # would silently answer a question about the reverse graph.
        self._reversible_steps = self._steps_closed_under_negation(steps)

        #: Routing cost of each path returned by the last
        #: :meth:`shortest_path` call, in the same order (``inf`` where no
        #: path exists). Free by-product of the O(path) device extraction
        #: — the field value at the target, bit-identical to what a full
        #: ``dist`` download would have carried. Informational: nothing in
        #: the package consumes it yet.
        self.last_path_costs: List[float] = []

        # Device state (item 1.2), built on first query and owned by this
        # object. Lazy so that constructing an API never touches the GPU.
        self._session = None
        self._session_unavailable = False

        # No graph construction — these are reported as zero
        self.edge_construction_time = 0.0
        self.graph_creation_time = 0.0
        self.graph = None

    # ------------------------------------------------------------------
    # Device session lifetime (item 1.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _steps_closed_under_negation(steps) -> bool:
        """True when ``-s`` is in the step set for every step ``s``."""
        try:
            pairs = {(int(a), int(b))
                     for a, b in np.asarray(steps).reshape(-1, 2)}
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return False
        return all((-a, -b) in pairs for a, b in pairs)

    def _get_session(self):
        """The device-resident session, or None if V5 is unavailable.

        Built once, on the first query. A ``RuntimeError`` means this
        machine cannot compile the V5 cooperative kernel; the stateless
        ``sssp_raster_gpu`` entry point then serves the query from V4/V3
        exactly as it did before Phase 1b.
        """
        if self._session is not None:
            return self._session
        if self._session_unavailable or not RASTER_GPU_AVAILABLE:
            return None
        try:
            self._session = GpuSsspSession(
                self.raster_data,
                self.steps,
                ignore_max=bool(self.ignore_max),
                delta=self._delta,
                dem=self.dem_data if self.gradient_luts is not None else None,
                gradient_luts=self.gradient_luts,
                arena_factor=self._arena_factor,
            )
        except RuntimeError:
            self._session_unavailable = True
            return None
        return self._session

    @property
    def device_bytes(self) -> int:
        """Device memory this API currently holds resident (0 when idle)."""
        return 0 if self._session is None else int(self._session.device_bytes)

    def close(self, free_pool: bool = True) -> None:
        """Release every device buffer this API holds. Idempotent.

        ``free_pool=True`` (the default): CuPy would otherwise keep the
        freed blocks in its own pool, where the driver still counts them
        against this process. On a 6 GB card shared with the user's other
        work, giving them back is the point. Pass ``False`` to keep the
        pool warm when another solve on this process is imminent.
        """
        session, self._session = self._session, None
        if session is not None:
            session.close(free_pool=free_pool)

    def invalidate_session(self) -> None:
        """Drop the device session without releasing the CuPy pool.

        The session snapshots the raster (and DEM) at upload time and
        deliberately never re-reads them — re-checking values every query
        is exactly the O(window) pass a session exists to avoid. Call this
        after mutating ``raster_data`` in place; the next query rebuilds
        and re-uploads.
        """
        session, self._session = self._session, None
        if session is not None:
            session.close()

    def __enter__(self) -> "RasterGPUAPI":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self):
        # PathFinder drops its _graph_api on every objective/handler
        # rebuild, so refcounting is the primary release path.
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _run_sssp(
            self,
            source: int,
            target_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SSSP and return the FULL (dist, pred) arrays.

        The cost-surface route: it downloads O(window). Path queries go
        through :meth:`_solve_paths` instead (items 1.3/1.4). Kept because
        callers that want the field want the whole field.
        """
        targets = None
        if target_indices is not None:
            targets = np.asarray(target_indices, dtype=np.int32)

        session = self._get_session()
        if session is not None:
            return session.solve(
                int(source),
                target_indices=targets,
                margin=self._margin,
                return_predecessor=True,
            )

        return sssp_raster_gpu(
            self.raster_data,
            self.steps,
            source_idx=int(source),
            delta=self._delta,
            ignore_max=self.ignore_max,
            target_indices=targets,
            margin=self._margin,
            return_predecessor=True,
            dem=self.dem_data if self.gradient_luts is not None else None,
            gradient_luts=self.gradient_luts,
        )

    def _solve_paths(
            self,
            source: int,
            target_indices,
    ) -> Tuple[List[Optional[List[int]]], np.ndarray]:
        """One solve; the paths to ``target_indices`` and their costs.

        Items 1.3 + 1.4: with a session this runs the on-device
        predecessor walk, so only the chains and ``dist`` at the targets
        cross PCIe, and only the links on those chains are repaired.

        Exactness. Batching several targets into one solve makes the
        kernel run *longer* than a single-target solve would (its early
        exit fires once **all** targets are settled), and never shorter.
        A vertex ``v`` on an optimal path to target ``t`` has
        ``d(v) <= d(t)``, and delta-stepping's exit criterion guarantees
        every vertex with true distance below the current span base is
        final — so every vertex the chain to ``t`` walks through already
        carried its final distance in the single-target solve. The extra
        relaxation therefore cannot move any distance the chain depends
        on, and ``dist[t]`` is identical either way.

        Returns ``(paths, costs)``: ``paths[i]`` is a node list running
        source -> target (both inclusive) or ``None`` where no path
        exists; ``costs[i]`` is ``dist`` at that target (``inf`` when
        unreachable).
        """
        source = int(source)
        targets = np.ascontiguousarray(
            np.asarray(target_indices, dtype=np.int32).ravel())

        session = self._get_session()
        if session is not None:
            chains, costs = session.solve_paths(
                source, targets, margin=self._margin)
            return [None if c is None else c.tolist() for c in chains], costs

        # V4/V3 fallback: full field + host walk (pre-Phase-1b behaviour).
        dist, pred = self._run_sssp(source, targets)
        paths: List[Optional[List[int]]] = []
        costs = np.full(targets.size, np.inf, dtype=np.float32)
        for i, target in enumerate(targets):
            target = int(target)
            paths.append(
                _walk_predecessors(pred, dist, source, target, self._n_nodes))
            if 0 <= target < dist.size:
                value = float(dist[target])
                costs[i] = np.inf if value >= _UNREACHABLE else value
        return paths, costs

    def _paths_for_pairs(
            self,
            pairs: Sequence[Tuple[int, int]],
    ) -> NodePathList:
        """Solve a list of (source, target) pairs, one solve per source.

        Item 1.5, mirroring what ``raster_fim_api`` already does. Pairs
        are grouped by source, each group is solved once with its whole
        target array, and the results are scattered back into the
        caller's original pair order — that ordering is the contract, and
        it is what the shuffled-pair test pins.
        """
        groups: dict = {}
        for index, (source, target) in enumerate(pairs):
            groups.setdefault(int(source), []).append((index, int(target)))

        results: List[Optional[NodeList]] = [None] * len(pairs)
        costs: List[float] = [float("inf")] * len(pairs)
        for source, entries in groups.items():
            targets = np.array([t for _, t in entries], dtype=np.int32)
            paths, group_costs = self._solve_paths(source, targets)
            for (index, _target), path, cost in zip(entries, paths,
                                                    group_costs):
                results[index] = [] if path is None else path
                costs[index] = float(cost)

        self.last_path_costs.extend(costs)
        return results

    def _extract_path(
            self,
            pred: np.ndarray,
            dist: np.ndarray,
            source: int,
            target: int,
    ) -> NodeList:
        """Reconstruct a single path from a downloaded predecessor array.

        Thin wrapper over :func:`_walk_predecessors` that keeps the
        ``NoPathFoundError`` contract. Path queries no longer come
        through here — they use the on-device walk (item 1.3) — but the
        full-field route and the V4/V3 fallback still do.
        """
        path = _walk_predecessors(pred, dist, source, target, self._n_nodes)
        if path is None:
            raise NoPathFoundError(source=int(source), target=int(target))
        return path

    # ------------------------------------------------------------------
    # Source/target case handlers
    # ------------------------------------------------------------------

    def _shortest_path_single_to_single(self, source_indices, target_indices):
        """Single source, single target."""
        source, target = int(source_indices), int(target_indices)
        paths, costs = self._solve_paths(source, [target])
        self.last_path_costs.append(float(costs[0]))
        if paths[0] is None:
            raise NoPathFoundError(source=source, target=target)
        return paths[0]

    def _shortest_path_single_to_multi(self, source_indices, target_indices):
        """Single source, multiple targets — already one solve."""
        paths, costs = self._solve_paths(source_indices, target_indices)
        self.last_path_costs.extend(float(c) for c in costs)
        return [[] if p is None else p for p in paths]

    def _shortest_path_multi_to_single(self, source_indices, target_indices):
        """Multiple sources, single target.

        Item 1.6. Edge weights are symmetric (see
        :attr:`symmetric_edge_weights`), so ONE solve rooted at the target
        with all sources as its targets answers every pair; each returned
        chain is reversed into source -> target order. Previously this ran
        k solves whenever ``dem_data`` was set, even though a bare DEM
        never reaches the kernel and the gradient term is
        ``fabsf``-symmetric.
        """
        target = int(target_indices)
        if self.symmetric_edge_weights and self._reversible_steps:
            paths, costs = self._solve_paths(target, source_indices)
            self.last_path_costs.extend(float(c) for c in costs)
            return [[] if p is None else p[::-1] for p in paths]
        # Asymmetric (or non-reversible step set): one solve per source.
        return self._paths_for_pairs(
            [(int(s), target) for s in source_indices])

    def _shortest_path_multi_to_multi(self, source_indices, target_indices,
                                      pairwise):
        """Multiple sources, multiple targets — one solve per source."""
        if pairwise:
            if len(source_indices) != len(target_indices):
                raise PairwiseError()
            pairs = [(int(s), int(t))
                     for s, t in zip(source_indices, target_indices)]
        else:
            pairs = [(int(s), int(t))
                     for s in source_indices for t in target_indices]
        return self._paths_for_pairs(pairs)

    def shortest_path(
            self,
            source_indices: Union[int, List[int], ndarray],
            target_indices: Union[int, List[int], ndarray],
            algorithm: str = "delta-stepping",
            **kwargs,
    ) -> Union[NodeList, NodePathList]:
        """
        Find shortest path(s) using raster-direct GPU delta-stepping.

        Parameters:
            source_indices: Source node index/indices
            target_indices: Target node index/indices
            algorithm: Only "delta-stepping" and "dijkstra" are accepted
            **kwargs: pairwise (bool) for pairwise computation

        Returns:
            Path or list of paths as node index lists

        Side effect: :attr:`last_path_costs` is replaced with the routing
        cost of each returned path, in the returned order.
        """
        if algorithm.lower() not in ("delta-stepping", "dijkstra"):
            raise AlgorithmNotImplementedError(
                algorithm, graph_library="raster_gpu"
            )

        self.last_path_costs = []

        source_has_len = hasattr(source_indices, "__len__")
        target_has_len = hasattr(target_indices, "__len__")

        if not source_has_len and not target_has_len:
            return self._shortest_path_single_to_single(
                source_indices, target_indices)
        if not source_has_len and target_has_len:
            return self._shortest_path_single_to_multi(
                source_indices, target_indices)
        if source_has_len and not target_has_len:
            return self._shortest_path_multi_to_single(
                source_indices, target_indices)
        return self._shortest_path_multi_to_multi(
            source_indices, target_indices, kwargs.get("pairwise", False))

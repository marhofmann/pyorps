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

Slope (Tier A). With a DEM the solve becomes Riemannian:

    M(x) = c(x)^2 (I + grad_z grad_z^T)

which is EXACTLY — and only — the unconditional 3D-length stretch
``sqrt(1 + (s/100)^2)`` that is pyorps' default ``GradientOptions``.
Travelling over sloped ground covers more real distance than the map
shows; that term is a genuine Riemannian metric and the solver honours it
without directional bias.

**The DEM contract.** ``dem_data`` is held BY REFERENCE — it is never
copied, because a copy costs 576 MB at 144 M cells. Everything derived
from it (the metric ``q`` planes, the grade-limit chord maps, the
legal-chord certificates) is cached. Therefore:

    Editing ``dem_data`` in place after construction invalidates those
    caches, and it is the CALLER's job to say so by calling
    :meth:`RasterFIMAPI.invalidate_dem_caches`.

There is a tripwire — a sampled checksum re-tested before every use of a
DEM-derived cache — but it is a **courtesy, not a guarantee**, and it must
not be relied on. It hashes a bounded ~256 KiB sample, so it catches
global or bulk edits and reliably misses localized ones at scale.
Measured on this implementation, editing the same buffer in place:

===============  ============  ==================  ===================
DEM size         sample step   one off-sample cell  50x50 patch (+5 m)
===============  ============  ==================  ===================
40 K cells       1             n/a (all sampled)    detected
9 M cells        137           **missed**           detected
144 M cells      2197          **missed**           **missed**
===============  ============  ==================  ===================

Pass ``strict_dem_check=True`` to hash the whole array instead. That is
exact at any size and costs a full pass over the DEM before each cached
use — measure it against your solve time before switching it on.

What is refused, and why (each refusal quotes its measured number rather
than saying "unsupported"):

- a configured slope MULTIPLIER curve, and the additive exposure term
  (``w_gradient > 0``): neither is of the form ``sqrt(d^T M d)``. Feeding
  them to the PDE would silently solve the *convexified* problem — which
  prices switchbacks below the objective as posed.
- ``order=2`` together with a DEM: the stage-2 operator is a per-axis
  one-sided second difference with no simplex analogue.
- a DEM without ``cell_size``: the metric needs metres.

``max_gradient_pct`` is NOT put into the metric (an ellipse-intersect-slab
"metric" re-admits 59 % of the forbidden compass). It is enforced OUTSIDE
the solver by a solve -> check -> mask -> re-solve loop that verifies the
returned cell path with ``_dijkstra.pyx``'s own binned arithmetic before
returning it.

**Read this before relying on a grade-limit failure.** That loop never
returns an invalid route, but it is optimal for the MASKED problem, and
the masked problem is much weaker than the rule: the limit forbids
STEPS, the solver can only be told to avoid CELLS, and forbidding a cell
also forbids the contour-following traverse across it that the rule
allows. Measured against the Cython kernel on identical inputs — 300
trials, 40x40, r1 steps on both sides, identical ``GradientLUTs``, limits
{8, 12, 20, 30} %, four terrain families, 4 seeds x 300 trials — the
kernel found a route in 662 cases and this loop returned a verified
route in 342 of them: a **false-negative rate of 320/662 = 48.3 %**
(per seed 40.6 %, 49.7 %, 50.9 %, 51.7 %;
``benchmarks/fuzz_grade_limit_differential.py``). Varying the mask rule
(endpoints-only, escalating, connectivity-preserving) moved that by at
most 3 points, so it is a property of masking cells and not of the
particular rule. A route this backend RETURNS is trustworthy; a failure
is not a verdict on the problem. The discrete backends are the authority
on ``max_gradient_pct``.

Three properties that loop is built to have:

- *exactness*. The check reproduces the kernel expression
  ``slope_bin = <int>(|dh| * <double>bin_factor[d])`` — including the
  float32 storage of ``bin_factor`` — rather than recomputing a percent
  slope in float64. The two disagree on ~1.5e-5 of steps, always at a bin
  boundary, which is precisely where a hard limit lives; a route this
  backend accepts must be one the discrete backends accept.
- *termination*. Masks are monotone and every iteration adds at least one
  cell of a finite set. The loop is additionally capped at
  ``max_mask_iterations`` (default 32) and RAISES at the cap — it never
  returns an unverified route and never silently falls back to the
  strictly smaller ``grade_limit_mode='eager'`` feasible set.
- *infeasibility*. Before the first solve, connected components of the
  LEGAL-CHORD graph decide exactly whether any route with all steps
  inside the limit exists — over the CALLER's own ``steps``, not a
  hardcoded 8-neighbourhood. (PathFinder's default is 'r2', whose knight
  moves cross faces no 8-adjacent route can; certifying with 8 would
  report the default configuration infeasible while a legal r2 route
  exists.) If not, that is reported immediately as a property of the
  problem, at the cost of no solve at all — never as an iteration cap.
  Failure messages distinguish "infeasible" from "the masked relaxation
  gave up although a legal route exists".

Requirements: NVIDIA GPU with CUDA support (cupy-cuda12x >= 13.0.0).

Usage:
    pf = PathFinder(..., graph_api="raster_fim")
"""

from __future__ import annotations

import logging
import math
import zlib
from typing import List, Optional, Tuple, Union

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
        Q_ACUTE_LIMIT,
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

#: Measured recall of the masked relaxation against the discrete kernel.
#: Differential fuzz, 4 seeds x 300 trials, 40x40, r1 steps on BOTH
#: sides, identical GradientLUTs, limits {8, 12, 20, 30} %, terrain =
#: smooth hills / white noise / plane+fault / random walk. The Cython
#: kernel found a route in 662 of the 1200 cases; this backend's loop
#: returned a verified route in 342 of those 662. Per-seed rate 40.6 %,
#: 49.7 %, 50.9 %, 51.7 % — the spread is real, so the pooled figure is
#: the one quoted. Reproduce with
#: ``benchmarks/fuzz_grade_limit_differential.py`` (seeds 0-3).
MASK_FALSE_NEGATIVE_RATE = 320 / 662           # 48.3 %

#: Wording reused by every grade-limit failure and by the docs — the
#: honest statement of what the masked problem is. It is NOT "slightly"
#: conservative: the number below is measured, not estimated.
_MASK_CAVEAT = (
    "The masked relaxation is a SCREEN, not an authority. The limit "
    "forbids STEPS (it is direction-dependent) and the solver can only "
    "be told to avoid CELLS, so a cell is forbidden entirely once a "
    "route through it violated the limit — and with it every legal "
    "contour-following traverse across the same face. Measured "
    "false-negative rate against the Cython kernel on the same problem: "
    "320 of 662 (48.3 %; per-seed 40.6-51.7 %) — differential fuzz, 4 "
    "seeds x 300 trials, 40x40, r1 steps on both sides, identical "
    "GradientLUTs, limits {8, 12, 20, 30} %, four terrain families; see "
    "benchmarks/fuzz_grade_limit_differential.py. So: a route this "
    "backend RETURNS is verified against the discrete rule and is safe "
    "to use, but a FAILURE here says little about the problem — roughly "
    "half of them are routes a discrete backend (raster_gpu / cython) "
    "finds. Treat the discrete backends as the authority on "
    "max_gradient_pct. (A discrete backend also needs "
    "r >= sqrt((s_max/limit)^2 - 1) to make net uphill progress at all.)"
)


def tier_a_report(gradient_luts) -> Tuple[bool, str]:
    """Decide Tier A acceptance from the LUT ARRAYS, not the option names.

    This is the mechanism behind "never silently solving a different
    problem than the one posed", and it is strictly stronger than testing
    ``opts.multiplier is None``: a ``Callable`` multiplier bypasses the
    option check but cannot bypass the arrays it produced. Conversely a
    callable that happens to BE the identity passes — correctly, because
    Tier A is defined by the metric it produces, not by how the user
    spelled it.

    Accepted iff, over the live (non-forbidden) bins, ``mult`` is the pure
    3D stretch ``sqrt(1 + (s/100)^2)`` and ``add`` is identically zero,
    and the infinite bins are exactly the bins above
    ``max_gradient_pct``.

    Returns ``(accepted, reason)``; ``reason`` is "" when accepted.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    mult = np.asarray(gradient_luts.mult, dtype=np.float64)
    add = np.asarray(gradient_luts.add, dtype=np.float64)
    n_bins = int(gradient_luts.n_bins)
    bw = float(gradient_luts.bin_width_pct)
    s = (np.arange(n_bins, dtype=np.float64) + 0.5) * bw
    stretch = np.sqrt(1.0 + (s / 100.0) ** 2)

    inf_bins = ~np.isfinite(mult)
    live = ~inf_bins

    limit = gradient_luts.max_gradient_pct
    expected_inf = (s > limit) if limit is not None else np.zeros_like(
        inf_bins)
    if not np.array_equal(inf_bins, expected_inf):
        return False, (
            "the slope-response LUT forbids a set of bins that is not "
            "exactly {s > max_gradient_pct} — the only source of an "
            "infinite multiplier this backend can interpret is the hard "
            "grade limit."
        )
    if np.any(add[inf_bins] != 0.0):
        return False, "forbidden bins carry a non-zero additive term."

    if not np.allclose(add, 0.0, rtol=0.0, atol=0.0):
        peak = float(np.abs(add).max())
        return False, (
            f"the additive exposure term is active (w_gradient > 0; peak "
            f"|add| = {peak:.6g}). The per-metre feasibility is then "
            f"c*Gamma_mult + Gamma_add, which is NOT of the form "
            f"sqrt(d^T M d) for any metric M — there is no Riemannian "
            f"solver for it. Use a discrete backend (raster_gpu / "
            f"cython) for additive slope exposure."
        )

    if live.any() and not np.allclose(mult[live], stretch[live],
                                      rtol=1e-6, atol=0.0):
        k = int(np.argmax(np.abs(mult[live] - stretch[live])
                          / stretch[live]))
        s_bad = s[live][k]
        got, want = float(mult[live][k]), float(stretch[live][k])
        return False, (
            f"a slope MULTIPLIER curve is configured: at {s_bad:.2f} % "
            f"grade the LUT asks for x{got:.6g} where the pure 3D-length "
            f"stretch is x{want:.6g}. Any non-identity multiplier makes "
            f"the indicatrix NON-CONVEX, and a continuum solver then "
            f"silently solves the convexified problem F** instead — "
            f"which prices switchbacks below the objective as posed "
            f"(benchmarks/benchmark_slope_indicatrix.py measures up to "
            f"+76.8 % for a single two-leg switchback; the decision "
            f"document 2026-08-11-anisotropic-eikonal-increment.md "
            f"reports up to +18 758 % once switchbacks are unbounded). "
            f"Refused rather than approximated. Use a discrete backend "
            f"(raster_gpu / cython) for multiplier models."
        )
    return True, ""


class RasterFIMAPI(GraphAPI):
    """Graph API backed by the GPU eikonal / block-FIM solver.

    The raster IS the continuous cost field — no graph object, no edge
    list, no neighborhood.

    **DEM aliasing.** With a DEM, ``dem_data`` is held by reference — no
    defensive copy (a 144 M-cell float32 DEM is 576 MB). The Tier A metric,
    grade-limit chord maps and legal-chord certificates are cached from that
    array. In-place edits after construction leave those caches stale
    silently unless :meth:`invalidate_dem_caches` is called after each
    intentional update, or a fresh ``RasterFIMAPI`` is built.
    """

    def __init__(
            self,
            raster_data: ndarray,
            steps: ndarray,
            ignore_max: bool = True,
            dem_data: Optional[ndarray] = None,
            tile: Optional[int] = None,
            n_inner: Optional[int] = None,
            eps_rel: float = 1e-6,
            eps_abs: Optional[float] = None,
            disk_init: bool = True,
            max_outer_iterations: Optional[int] = None,
            order: int = 1,
            gradient_luts=None,
            cell_size: Optional[float] = None,
            q_flat_eps: float = 1e-3,
            slope_stencil: str = "central",
            strict_dem_check: bool = False,
            grade_limit_mode: str = "lazy",
            max_mask_iterations: int = 32,
            **kwargs,
    ):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if not RASTER_FIM_AVAILABLE:
            raise ImportError(
                "The raster_fim backend requires CuPy with CUDA support. "
                "Install with: pip install cupy-cuda12x"
            )
        if gradient_luts is not None and dem_data is None:
            raise AlgorithmNotImplementedError(
                "slope-response LUTs without a DEM (nothing to take the "
                "elevation gradient of)",
                graph_library="raster_fim",
            )
        if grade_limit_mode not in ("lazy", "eager"):
            raise ValueError(
                f"grade_limit_mode must be 'lazy' or 'eager', "
                f"got {grade_limit_mode!r}")

        super().__init__(raster_data, steps, ignore_max, dem_data)
        logger.debug(
            "raster_fim ignores the neighborhood (steps/neighborhood_str)"
            " — the eikonal PDE is continuous; changing the neighborhood "
            "changes nothing.")

        rows, cols = raster_data.shape[:2]
        self._rows, self._cols = int(rows), int(cols)

        #: The chords the CALLER routes with. The PDE has no
        #: neighborhood, but the grade-limit certificates are statements
        #: about the discrete routes the caller can take, so they are
        #: taken over this set and never over a hardcoded 8.
        self._cert_offsets = self._offsets_from_steps(steps)

        # --- Tier A decision table (module docstring) ------------------
        self._dem = None
        self._cell_size = None
        self._dem_checksum = None
        self._strict_dem_check = bool(strict_dem_check)
        self._luts = gradient_luts
        self._grade_limit = None
        self._grade_limit_mode = grade_limit_mode
        self._max_mask_iterations = int(max_mask_iterations)
        self._eager_mask: Optional[np.ndarray] = None
        #: Never mask a cell whose removal would cut the last legal
        #: corridor — see :meth:`_drop_disconnecting`. Escalation point
        #: of the aggressive rule; 0 = always aggressive (measured best).
        self._mask_keeps_connectivity = True
        self._mask_escalate_after = 0
        self._mask_block = ""
        #: Consecutive non-improving iterations before the loop calls the
        #: relaxation stalled instead of grinding to the cap. 12 was
        #: measured (differential fuzz, seed 0, 300 trials) to give the
        #: same recall as no cut-off at all — 82/161 false negatives
        #: either way — for half the wall time (20.6 s vs 38.6 s) and 7
        #: cap-outs instead of 50.
        self._mask_patience = 12
        aniso_kwargs = {}

        if dem_data is not None:
            if order != 1:
                raise AlgorithmNotImplementedError(
                    "order=2 together with a DEM — the second-order "
                    "refinement is a per-axis one-sided difference with "
                    "a frozen upwind code byte and has no anisotropic "
                    "simplex analogue (it is also measured to undershoot "
                    "by up to 10 % on piecewise-constant planning "
                    "surfaces). Use order=1 with the DEM",
                    graph_library="raster_fim",
                )
            if cell_size is None:
                raise ValueError(
                    "raster_fim with a DEM needs cell_size (metres per "
                    "cell): the Tier A metric M = c^2 (I + grad_z "
                    "grad_z^T) is built from an elevation gradient in "
                    "metres of rise per metre of run, so without it the "
                    "anisotropy is wrong by a factor of the cell size.")
            cell_size = float(cell_size)
            if not math.isfinite(cell_size) or cell_size <= 0:
                raise ValueError(
                    f"cell_size must be finite and > 0, got {cell_size}")

            s_max_pct = 100.0 * 2.0            # solver default clamp
            if gradient_luts is not None:
                self._check_cell_size(gradient_luts, steps, cell_size)
                self._check_bin_factor(gradient_luts, steps, cell_size)
                ok, reason = tier_a_report(gradient_luts)
                if not ok:
                    raise AlgorithmNotImplementedError(
                        f"this objective on raster_fim — {reason}",
                        graph_library="raster_fim",
                    )
                s_max_pct = (float(gradient_luts.n_bins)
                             * float(gradient_luts.bin_width_pct))
                self._grade_limit = gradient_luts.max_gradient_pct
            q_clamp = s_max_pct / 100.0
            if q_clamp > Q_ACUTE_LIMIT:
                raise AlgorithmNotImplementedError(
                    f"s_max_pct = {s_max_pct:.1f} % (|q| = {q_clamp:.4f}) "
                    f"on raster_fim — beyond the 8-simplex acuteness "
                    f"threshold {Q_ACUTE_LIMIT:.5f} (219.74 %, anisotropy "
                    f"ratio kappa = 1 + sqrt(2) = 2.41421) the stencil "
                    f"converges to a slightly LARGER metric in a wedge of "
                    f"directions (measured over-pricing 0.60 % at |q| = "
                    f"2.5, 3.67 % at |q| = 3.0), which is the very "
                    f"metrication bias this backend exists to remove. "
                    f"Lower s_max_pct (the default 200 % is inside the "
                    f"provably exact regime) or use a discrete backend",
                    graph_library="raster_fim",
                )

            # BY REFERENCE, deliberately: a copy is 576 MB at 144 M cells.
            # ascontiguousarray is a no-op for an already-float32
            # C-contiguous input, so self._dem usually IS the caller's
            # array. See the DEM contract in the module docstring.
            self._dem = np.ascontiguousarray(dem_data, dtype=np.float32)
            if self._dem.shape != (self._rows, self._cols):
                raise ValueError(
                    f"dem_data shape {self._dem.shape} does not match the "
                    f"cost raster {(self._rows, self._cols)}")
            self._strict_dem_check = bool(strict_dem_check)
            self._dem_checksum = self._dem_checksum_of(
                self._dem, self._strict_dem_check)
            self._cell_size = cell_size
            aniso_kwargs = dict(
                dem=self._dem, cell_size=cell_size, q_clamp=q_clamp,
                q_flat_eps=float(q_flat_eps),
                slope_stencil=slope_stencil,
            )
            logger.debug(
                "raster_fim Tier A: anisotropic 3D-length metric active "
                "(cell_size=%.4g m, s_max=%.1f %%, grade limit=%s)",
                cell_size, s_max_pct, self._grade_limit)

        self._solver_kwargs = dict(
            tile=tile, n_inner=n_inner, eps_rel=eps_rel, eps_abs=eps_abs,
            disk_init=disk_init,
            max_outer_iterations=max_outer_iterations,
            order=order,
            **aniso_kwargs,
        )

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
        # The Tier A metric is a property of the DEM, not of any one
        # solve, so it is cached once and EVERY trace under a DEM uses
        # it — see _trace_metric.
        self._metric_device = (None, None)
        self._q_cache = None
        self._chord_cache: dict = {}
        self._edges_built = False
        self._edge_cache = None
        self._labels_built = False
        self._label_cache = None
        #: Grade-limit mask iterations per returned path (empty without a
        #: limit) — the honest per-route cost of the lazy-constraint loop.
        self.last_mask_iterations: List[int] = []

        if self._grade_limit is not None and grade_limit_mode == "eager":
            self._eager_mask = self._steep_cells(self._grade_limit)
            logger.debug(
                "raster_fim grade limit %.3g %% in EAGER mode: %d cells "
                "pre-masked", self._grade_limit, self._eager_mask.size)

    # ------------------------------------------------------------------
    # Tier A helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dem_checksum_of(dem: np.ndarray, strict: bool = False) -> tuple:
        """Fingerprint for spotting in-place DEM edits without copying it.

        BEST EFFORT, NOT A GUARANTEE — see the DEM contract in the module
        docstring. With ``strict=False`` this samples at most ~256 KiB of
        float32 payload, so a 144 M-cell grid pays microseconds instead of
        a full pass. The price is that any subsample misses a localized
        edit: measured on the same buffer, a 50x50-cell patch edit is
        DETECTED at 9 M cells and MISSED at 144 M cells. It catches shape,
        dtype and reallocation changes exactly, and bulk/global edits with
        high probability.

        ``strict=True`` hashes every byte, which is exact at any size and
        costs a full pass. The contract is still that the caller invokes
        ``invalidate_dem_caches()``; this only decides how likely we are to
        notice when they forget.
        """
        flat = np.ascontiguousarray(dem).ravel()
        n = int(flat.size)
        if n == 0:
            return (dem.shape, dem.dtype.str, 0)
        step = 1 if strict else max(1, n // 65536)
        return (dem.shape, dem.dtype.str,
                int(dem.ctypes.data), zlib.adler32(flat[::step].tobytes()))

    def _check_dem_unchanged(self) -> None:
        """Trip if an in-place DEM edit is VISIBLE to the fingerprint.

        Best effort by construction — a sampled fingerprint cannot see a
        localized edit at 144 M cells. Passing this check is therefore not
        evidence the DEM is unchanged; the contract is that the caller
        calls ``invalidate_dem_caches()``.
        """
        if self._dem is None:
            return
        if self._dem_checksum_of(
                self._dem, self._strict_dem_check) == self._dem_checksum:
            return
        raise RuntimeError(
            "raster_fim: dem_data was modified in place after "
            "RasterFIMAPI was constructed. The Tier A metric, "
            "grade-limit chord maps and legal-chord certificates were "
            "built from the old elevations and are now stale. Pass a "
            "fresh array into a new API instance, or call "
            "invalidate_dem_caches() after updating the DEM "
            "intentionally. NOTE: this check samples the DEM unless "
            "strict_dem_check=True, so it catches bulk edits and misses "
            "localized ones at scale — reaching this message means an "
            "edit was caught, never that edits are always caught.")

    def invalidate_dem_caches(self) -> None:
        """Drop every cache derived from ``dem_data``.

        Call this after an intentional in-place DEM edit. Does not copy
        the array — it re-fingerprints the current buffer and forces the
        metric, host ``q`` planes, chord maps and legal-chord graph to be
        rebuilt on the next use.
        """
        if self._dem is None:
            return
        self._dem_checksum = self._dem_checksum_of(
            self._dem, self._strict_dem_check)
        self._metric_device = (None, None)
        self._q_cache = None
        self._chord_cache = {}
        self._edges_built = False
        self._edge_cache = None
        self._labels_built = False
        self._label_cache = None

    #: Above this many cells the legal-chord connected-components
    #: certificate is skipped: it materialises up to 4 edge arrays over
    #: the whole grid. Failures are then reported as UNCERTIFIED rather
    #: than guessed at.
    _MAX_CERTIFY_CELLS = 4_000_000

    #: 8-neighbour offsets. The eikonal tracer's rasterized cell paths
    #: are 8-adjacent (``polyline_to_cells``), so these are the chords
    #: the grade rule has to be projected onto for MASKING — the routes
    #: this backend returns can only ever use them.
    #:
    #: They are NOT the chords the infeasibility certificate may use: a
    #: certificate has to speak about the routes the CALLER can take
    #: (``self._cert_offsets``, built from ``steps``), or it declares
    #: problems infeasible that the caller's own neighborhood solves.
    _OFFSETS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))

    @staticmethod
    def _offsets_from_steps(steps) -> Tuple[Tuple[int, int], ...]:
        """Symmetric closure of the caller's ``steps`` as (dr, dc) pairs.

        The certificate is an undirected reachability statement, so a
        directed step array (``get_neighborhood_steps(..., directed=True)``
        already returns both signs, but nothing guarantees it) is closed
        under negation first. Falls back to the 8-neighbourhood only when
        there are no usable steps at all.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        try:
            arr = np.atleast_2d(np.asarray(steps, dtype=np.int64))
        except (TypeError, ValueError):
            return RasterFIMAPI._OFFSETS8
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            return RasterFIMAPI._OFFSETS8
        out = set()
        for dr, dc in arr.tolist():
            if dr == 0 and dc == 0:
                continue
            out.add((int(dr), int(dc)))
            out.add((-int(dr), -int(dc)))
        return tuple(sorted(out)) if out else RasterFIMAPI._OFFSETS8

    @staticmethod
    def _forward_offsets(offsets) -> Tuple[Tuple[int, int], ...]:
        """One representative per undirected chord direction."""
        return tuple(o for o in offsets
                     if o[0] > 0 or (o[0] == 0 and o[1] > 0))

    @staticmethod
    def _bin_factor_for(length_cells, cell_size: float,
                        bin_inv: float) -> np.ndarray:
        """Reproduce ``GradientLUTs.bin_factor`` for arbitrary step lengths.

        The discrete kernels do NOT compute a slope in percent and then
        divide by the bin width; they multiply ``|Δh|`` by a per-direction
        factor that was *stored as float32*::

            slope_bin = <int>(height_diff * <double>grad_bin_factor[i])

        (``_dijkstra.pyx`` lines 328-332). Recomputing the same quantity
        in float64 as ``floor(100*Δh/(L*cell_size) * bin_inv)`` is the
        same mathematics with a different rounding, and the two disagree
        on roughly 1.5e-5 of steps — always at a bin boundary, which is
        exactly where a hard grade limit lives. Measured on the default
        R1 neighborhood: ``bin_factor`` for a diagonal is float32
        28.284273 where the exact value is 28.2842712…, so the kernel bins
        marginally HIGHER and would forbid steps this backend was calling
        legal.

        So the factor is reconstructed byte-for-byte the way
        ``objective.py`` builds it (float32 ``inv_horiz_m``, widened to
        float64, scaled, narrowed back to float32) and the caller widens
        it once more, exactly as the kernel does. ``_check_bin_factor``
        asserts at construction that this reproduction is bit-identical to
        the array the kernels are actually handed.
        """
        length = np.atleast_1d(np.asarray(length_cells, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = (1.0 / (length * cell_size)).astype(np.float32)
            return (100.0 * inv.astype(np.float64)
                    * bin_inv).astype(np.float32)

    @classmethod
    def _check_bin_factor(cls, luts, steps, cell_size: float) -> None:
        """Tripwire: our reproduction must BE the kernels' own array.

        If ``objective.py`` ever changes how ``bin_factor`` is formed, the
        grade check silently starts answering a slightly different
        question than the discrete backends. Caught at construction, for
        free, rather than as a one-in-a-million accepted-but-illegal step.
        """
        bf = getattr(luts, "bin_factor", None)
        if bf is None:
            return
        arr = np.asarray(steps)
        lengths = np.sqrt(arr[:, 0].astype(np.float64) ** 2
                          + arr[:, 1].astype(np.float64) ** 2)
        ours = cls._bin_factor_for(lengths, cell_size, float(luts.bin_inv))
        theirs = np.asarray(bf, dtype=np.float32)
        if ours.shape != theirs.shape or not np.array_equal(ours, theirs):
            raise ValueError(
                "raster_fim cannot reproduce GradientLUTs.bin_factor "
                "bit-for-bit, so it cannot verify a route against the "
                "discrete grade rule the search kernels apply. The "
                "grade-limit loop is only trustworthy while this "
                "reproduction is exact — refusing rather than checking "
                "with slightly different arithmetic.")

    @staticmethod
    def _check_cell_size(luts, steps, cell_size: float) -> None:
        """Unit-bug tripwire (plan 7.3).

        ``GradientLUTs.inv_horiz_m[k] = 1/(step_len_cells[k]*cell_size)``,
        so the cell size the LUTs were built with is recoverable from
        them. A mismatch means the metric would be formed with the wrong
        horizontal scale — a plausible-looking field with the wrong
        anisotropy at every scale, and the highest-probability silent bug
        in this increment. Caught at construction, for free.
        """
        lengths = np.asarray(getattr(luts, "step_len_cells", None)
                             if getattr(luts, "step_len_cells", None)
                             is not None
                             else np.hypot(np.asarray(steps)[:, 0],
                                           np.asarray(steps)[:, 1]),
                             dtype=np.float64)
        inv = np.asarray(luts.inv_horiz_m, dtype=np.float64)
        if inv.size != lengths.size or not np.all(inv > 0):
            return
        recovered = 1.0 / (inv * lengths)
        if not np.allclose(recovered, cell_size, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"cell_size mismatch: raster_fim was given "
                f"cell_size = {cell_size!r} m, but the gradient LUTs were "
                f"built with {float(np.median(recovered)):.10g} m "
                f"(recovered from inv_horiz_m). The Tier A metric would "
                f"then be scaled wrong at every point. Pass the search "
                f"window's cell size.")

    def _q_host(self) -> Tuple[np.ndarray, np.ndarray]:
        """Host metric planes (cached) — the grade-limit mask rule and
        the host tracer fallback both need |q| per cell."""
        self._check_dem_unchanged()
        if getattr(self, "_q_cache", None) is None:
            from pyorps.utils.eikonal_gpu import metric_from_dem
            q_r, q_c, _bad = metric_from_dem(
                self._dem, self._cell_size,
                q_clamp=self._solver_kwargs["q_clamp"],
                slope_stencil=self._solver_kwargs["slope_stencil"])
            self._q_cache = (q_r, q_c)
        return self._q_cache

    def _chord_maps(self, offsets=None):
        """Per-cell projections of the DISCRETE grade rule itself (cached).

        ``offsets`` selects the chord set. It defaults to the
        8-neighbourhood — the only chords a route RETURNED by this
        backend can contain — and the infeasibility certificate passes
        ``self._cert_offsets`` instead, because an "isolated" verdict is
        a statement about the routes the caller can take.

        Every chord out of every cell is evaluated with the
        kernel's own binned arithmetic — the same expression
        :meth:`grade_violations` uses on a route — giving three planes:

        ``steep``     at least one chord out of the cell is forbidden, so
                      the cell CAN cause a violation on its own and
                      masking it is never speculative;
        ``isolated``  every in-bounds chord is forbidden, so no route can
                      enter or leave the cell at all — genuine
                      infeasibility, detectable before any solve;
        ``max_pct``   the steepest chord out of the cell, in percent, used
                      to break ties when a violating step has no steep
                      endpoint.

        Deriving these from the rule rather than from the central
        difference ``|q|`` is what makes ``grade_limit_mode='eager'``
        provably valid for 8-adjacent routes: if no cell on a route is
        ``steep``, then no unit step of that route is forbidden. The
        central difference cannot promise that — it halves a one-sided
        cliff, so a cell can read flat and still have an illegal chord.
        """
        self._check_dem_unchanged()
        offsets = tuple(self._OFFSETS8 if offsets is None else offsets)
        cache = getattr(self, "_chord_cache", None)
        if cache is None:
            cache = self._chord_cache = {}
        if offsets in cache:
            return cache[offsets]
        rows, cols = self._rows, self._cols
        dem = self._dem.astype(np.float64)
        n_bins = int(self._luts.n_bins)
        mult = np.asarray(self._luts.mult)
        bin_inv = float(self._luts.bin_inv)

        n_valid = np.zeros((rows, cols), dtype=np.int16)
        n_viol = np.zeros((rows, cols), dtype=np.int16)
        max_pct = np.zeros((rows, cols), dtype=np.float64)
        for dr, dc in offsets:
            if abs(dr) >= rows or abs(dc) >= cols:
                continue
            rs = slice(max(0, -dr), rows - max(0, dr))
            rd = slice(max(0, dr), rows - max(0, -dr))
            cs = slice(max(0, -dc), cols - max(0, dc))
            cd = slice(max(0, dc), cols - max(0, -dc))
            hd = np.abs(dem[rd, cd] - dem[rs, cs])
            length = math.sqrt(float(dr) ** 2 + float(dc) ** 2)
            bf = float(self._bin_factor_for(
                length, self._cell_size, bin_inv)[0])
            bins = np.minimum((hd * bf).astype(np.int64), n_bins - 1)
            bad = ~np.isfinite(mult[bins])
            n_valid[rs, cs] += 1
            n_viol[rs, cs] += bad
            np.maximum(max_pct[rs, cs],
                       100.0 * hd / (length * self._cell_size),
                       out=max_pct[rs, cs])
        steep = n_viol > 0
        isolated = (n_valid > 0) & (n_viol == n_valid)
        cache[offsets] = (steep, isolated, max_pct)
        return cache[offsets]

    def _passable(self) -> np.ndarray:
        """The raster's OWN impassability, as the solver applies it."""
        raster = self.raster_data
        if np.issubdtype(raster.dtype, np.floating):
            ok = np.isfinite(raster) & (raster < FINITE_LIMIT)
        elif self.ignore_max:
            ok = raster != np.iinfo(raster.dtype).max
        else:
            ok = np.ones(raster.shape, dtype=bool)
        return ok & np.isfinite(self._dem)

    def _legal_chord_edges(self):
        """Edge arrays ``(a, b)`` of the LEGAL-CHORD graph (cached).

        Vertices are passable cells; ``u — v`` is an edge iff ``u`` and
        ``v`` are neighbours **under the caller's own step set** and the
        discrete rule permits that chord. Every route of the discrete
        backend with no violating step is a walk in this graph and vice
        versa, which is what makes the component test below an exact
        certificate rather than a heuristic.

        Returns None when the raster is too large to certify cheaply or
        scipy is missing; callers must then say the verdict is
        uncertified rather than guess.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        self._check_dem_unchanged()
        if getattr(self, "_edges_built", False):
            return self._edge_cache
        self._edges_built = True
        self._edge_cache = None
        rows, cols = self._rows, self._cols
        if rows * cols > self._MAX_CERTIFY_CELLS:
            return None
        try:
            import scipy.sparse                       # noqa: F401  # pylint: disable=unused-import
        except ImportError:              # pragma: no cover - scipy is a dep
            return None

        dem = self._dem.astype(np.float64)
        ok = self._passable()
        n_bins = int(self._luts.n_bins)
        mult = np.asarray(self._luts.mult)
        bin_inv = float(self._luts.bin_inv)
        flat_id = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)

        src_parts, dst_parts = [], []
        # FORWARD half of the caller's steps: each undirected edge once.
        for dr, dc in self._forward_offsets(self._cert_offsets):
            if abs(dr) >= rows or abs(dc) >= cols:
                continue
            rs = slice(max(0, -dr), rows - max(0, dr))
            rd = slice(max(0, dr), rows - max(0, -dr))
            cs = slice(max(0, -dc), cols - max(0, dc))
            cd = slice(max(0, dc), cols - max(0, -dc))
            hd = np.abs(dem[rd, cd] - dem[rs, cs])
            length = math.sqrt(float(dr) ** 2 + float(dc) ** 2)
            bf = float(self._bin_factor_for(
                length, self._cell_size, bin_inv)[0])
            bins = np.minimum((hd * bf).astype(np.int64), n_bins - 1)
            good = (np.isfinite(mult[bins]) & ok[rs, cs] & ok[rd, cd])
            src_parts.append(flat_id[rs, cs][good])
            dst_parts.append(flat_id[rd, cd][good])

        a = np.concatenate(src_parts) if src_parts else np.empty(0, np.int64)
        b = np.concatenate(dst_parts) if dst_parts else np.empty(0, np.int64)
        self._edge_cache = (a, b)
        return self._edge_cache

    def _components(self, removed: Optional[np.ndarray] = None
                    ) -> Optional[np.ndarray]:
        """Component labels of the legal-chord graph minus ``removed``."""
        edges = self._legal_chord_edges()
        if edges is None:
            return None
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        a, b = edges
        ok = self._passable().ravel().copy()
        if removed is not None and len(removed):
            ok[np.asarray(removed, dtype=np.int64)] = False
            keep = ok[a] & ok[b]
            a, b = a[keep], b[keep]
        n = self._rows * self._cols
        adj = coo_matrix(
            (np.ones(a.size, dtype=np.uint8), (a, b)), shape=(n, n))
        _n_comp, labels = connected_components(
            adj.tocsr(), directed=False, return_labels=True)
        labels = labels.astype(np.int64)
        # Impassable / removed cells are isolated vertices; give them a
        # label no terminal can accidentally match.
        labels[~ok] = -1
        return labels

    def _legal_chord_labels(self) -> Optional[np.ndarray]:
        """Connected components of the LEGAL-CHORD graph (cached).

        Two cells in different components cannot be joined by ANY legal
        route over the caller's neighborhood: an exact infeasibility
        certificate that costs no solve at all.

        It is what separates "the constraint disconnects source from
        target" (report it, immediately) from "the masked relaxation ran
        out of iterations" (a limitation of this backend, say so). The
        certificate is built on ``steps`` as handed to the constructor —
        PathFinder's default is 'r2', whose knight moves cross faces the
        8-neighbourhood cannot, so hard-coding 8 here would declare
        problems infeasible that the DEFAULT configuration solves.

        Returns None when the raster is too large to certify cheaply; the
        caller must then say the verdict is uncertified rather than guess.
        """
        if getattr(self, "_labels_built", False):
            return self._label_cache
        self._labels_built = True
        self._label_cache = self._components(None)
        return self._label_cache

    def _steep_cells(self, limit_pct: Optional[float] = None) -> np.ndarray:
        """Flat indices of cells with at least one forbidden chord.

        ``limit_pct`` is accepted for call-site readability and ignored:
        the limit already lives inside the LUT the chord test consults.
        """
        steep, _isolated, _max_pct = self._chord_maps()
        return np.flatnonzero(steep.ravel())

    def grade_violations(self, cells) -> np.ndarray:
        """Steps of a cell path that the discrete grade rule forbids.

        Deliberately uses the DISCRETE kernel's own binned arithmetic
        (``_dijkstra.pyx`` lines 322-338) over the RETURNED cell path,
        because that is the object the rule is defined on and what
        pyorps' evaluator will re-measure — a route accepted here must be
        the route a discrete re-check accepts. Comparing against the raw
        limit instead would let bin-boundary cases through.

        Returns an int array of step indices ``i`` (step = cells[i] ->
        cells[i+1]) that violate the limit.
        """
        self._check_dem_unchanged()
        arr = np.asarray(cells, dtype=np.int64)
        if arr.size < 2 or self._grade_limit is None:
            return np.empty(0, dtype=np.int64)
        cols = self._cols
        ra, ca = np.divmod(arr[:-1], cols)
        rb, cb = np.divmod(arr[1:], cols)
        d_r = (rb - ra).astype(np.float64)
        d_c = (cb - ca).astype(np.float64)
        # sqrt(dr^2 + dc^2), not hypot: objective.py forms the direction
        # lengths this way and the two are not required to agree in the
        # last ulp.
        len_cells = np.sqrt(d_r * d_r + d_c * d_c)
        flat = self._dem.ravel()
        # <double>grad_dem[b] - <double>grad_dem[a] on a float32 DEM.
        height_diff = np.abs(flat[arr[1:]].astype(np.float64)
                             - flat[arr[:-1]].astype(np.float64))

        luts = self._luts
        if luts is None or getattr(luts, "bin_factor", None) is None:
            # No LUTs to be faithful to: compare against the raw limit.
            with np.errstate(divide="ignore", invalid="ignore"):
                slope_pct = 100.0 * height_diff / (len_cells
                                                   * self._cell_size)
            slope_pct = np.nan_to_num(slope_pct, nan=0.0, posinf=np.inf)
            return np.flatnonzero(slope_pct > self._grade_limit)

        n_bins = int(luts.n_bins)
        mult = np.asarray(luts.mult)
        bin_factor = self._bin_factor_for(
            len_cells, self._cell_size, float(luts.bin_inv))
        with np.errstate(invalid="ignore"):
            # THE kernel expression, in the kernel's own precision:
            #   slope_bin = <int>(height_diff * <double>bin_factor[i])
            #   if slope_bin >= n_bins: slope_bin = n_bins - 1
            raw = height_diff * bin_factor.astype(np.float64)
        finite = np.isfinite(raw)
        # <int>(x) truncates toward zero and height_diff >= 0, so the
        # cast is floor(); the kernel has no lower clamp because it
        # cannot produce a negative bin either.
        bins = np.minimum(np.where(finite, raw, 0.0).astype(np.int64),
                          n_bins - 1)
        violating = ~np.isfinite(mult[bins])
        # A non-finite product means a NaN in the DEM (the kernel's
        # behaviour there is undefined, so refuse the step) or a
        # zero-length step (impossible after dedup, and legal anyway).
        violating |= ~finite & (len_cells > 0)
        violating &= len_cells > 0
        return np.flatnonzero(violating)

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
        self._check_dem_unchanged()
        t_field, _d_t, (t_trace, d_trace), q_dev = eikonal_raster_gpu(
            self.raster_data, field_sources,
            ignore_max=self.ignore_max, return_device=True,
            return_trace_field=True, return_metric=True,
            target_index=target_index,
            forbidden_indices=self._forbidden(),
            **self._solver_kwargs)
        self._last_host_field = t_field
        self._last_trace_host = t_trace
        self._last_trace_device = d_trace
        self._remember_metric(q_dev)
        return t_field

    def _remember_metric(self, q_dev) -> None:
        """Keep the solve's metric planes for later traces on any field.

        ``M(x)`` depends only on the DEM, the cell size and the clamp —
        never on the source — so one copy serves every field this API
        ever solves, and a trace can never end up without it.
        """
        if q_dev is not None and q_dev[0] is not None \
                and self._metric_device[0] is None:
            self._metric_device = (q_dev[0], q_dev[1])

    def _trace_metric(self):
        """The metric EVERY trace under a DEM must descend with.

        The Riemannian descent direction is ``-M^-1 grad T``; tracing a
        field that was solved under ``M`` with plain ``-grad T`` is a
        silent wrong answer, not a degradation (measured on the analytic
        corrugated-ramp geodesic: 0.71 cells off with the metric, 14.33
        cells off without it). So this never returns ``(None, None)``
        while a DEM is configured: if the cached planes are gone it
        rebuilds them from the DEM rather than letting the tracer fall
        back to the isotropic branch.
        """
        if self._dem is None:
            return (None, None)
        self._check_dem_unchanged()
        if self._metric_device[0] is None:
            from pyorps.utils.eikonal_gpu import _device_metric
            d_qr, d_qc, _bad, _diag = _device_metric(
                self._dem, self._cell_size,
                q_clamp=self._solver_kwargs["q_clamp"],
                slope_stencil=self._solver_kwargs["slope_stencil"])
            self._metric_device = (d_qr, d_qc)
        if self._metric_device[0] is None:      # pragma: no cover - guard
            raise RuntimeError(
                "raster_fim: the Tier A metric is unavailable for "
                "tracing a field that was solved WITH it. Refusing to "
                "trace isotropically — that silently descends -grad T "
                "instead of -M^-1 grad T and returns a plausible route "
                "that is not the geodesic.")
        return self._metric_device

    def _forbidden(self, extra=None):
        """Flat indices forced impassable for the next solve."""
        if self._eager_mask is None and extra is None:
            return None
        parts = [p for p in (self._eager_mask, extra) if p is not None]
        if not parts:
            return None
        return np.unique(np.concatenate(
            [np.asarray(p, dtype=np.int64).ravel() for p in parts]))

    def _paths_from_field(
            self,
            t_field: np.ndarray,
            field_sources: np.ndarray,
            trace_starts: np.ndarray,
            reverse: bool,
            trace: Optional[tuple] = None,
    ) -> List[Optional[NodeList]]:
        """Trace descent paths for ``trace_starts`` on one T field.

        With ``reverse=True`` the field sources are the routing sources
        (polylines are flipped to run source -> target); with False the
        field was solved from the routing target and each traced start is
        a routing source (already source -> target).
        """
        if trace is not None:          # caller kept this field's own pair
            trace_host, trace_dev = trace
        elif t_field is self._last_host_field:
            trace_host = self._last_trace_host
            trace_dev = self._last_trace_device
        elif self._last_trace_host is self._last_host_field:
            # order=1: the returned field IS its own trace field, so an
            # older cached one can be re-uploaded safely.
            trace_host, trace_dev = t_field, None
        else:
            # order=2: the returned field is the REFINED one, which is not
            # descent-connected (see _solve). Tracing it instead of the
            # preserved first-order field silently moves the polyline —
            # measured at 1.27 cells. Callers that reuse a field across
            # pairs must keep its trace pair and pass it in.
            raise RuntimeError(
                "cannot trace a cached order=2 field without its "
                "first-order trace field; pass trace=(host, device)")
        # NEVER derived from which field this is: the metric belongs to
        # the DEM, and a field solved under M must be traced under M.
        q_dev = self._trace_metric()
        polylines = trace_paths_gpu(trace_host, field_sources,
                                    trace_starts, t_device=trace_dev,
                                    q_device=q_dev)
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

    def _solve_trace_pair(self, source: int, target: int, mask):
        """One solve + trace for a single pair; returns (cells, poly, cost).

        ``poly`` runs source -> target. Returns ``(None, None, inf)``
        when the target is unreachable under ``mask``.
        """
        self._check_dem_unchanged()
        src_arr = np.array([source])
        _t_none, d_t, (_tr, d_trace), q_dev = eikonal_raster_gpu(
            self.raster_data, src_arr,
            ignore_max=self.ignore_max, return_device=True,
            return_trace_field=True, return_metric=True,
            target_index=target, download=False,
            forbidden_indices=self._forbidden(mask),
            **self._solver_kwargs)
        self._remember_metric(q_dev)
        cost = float(d_t[target])
        if not cost < FINITE_LIMIT:
            return None, None, float("inf")
        poly = trace_paths_gpu(
            None, src_arr, [target], t_device=d_trace, q_device=q_dev,
            shape=(self._rows, self._cols))[0]
        if poly is None:
            return None, None, float("inf")
        poly = poly[::-1]                      # source -> target order
        forbidden = (d_trace >= FINITE_LIMIT).get().reshape(
            self._rows, self._cols)
        cells = polyline_to_cells(poly, self._rows, self._cols,
                                  forbidden_mask=forbidden)
        return cells, poly, cost

    def _single_to_single(self, source, target):
        """Lean single-pair path (performance plan phase 1): the solve
        skips the full-field D2H; T[target] is a device scalar read,
        tracing runs device-only, and only the forbidden mask (uint8,
        a quarter of the float field) crosses the bus.

        With a hard grade limit this becomes the lazy-constraint loop of
        :meth:`_solve_with_grade_limit`."""
        source, target = int(source), int(target)
        if self._grade_limit is not None and self._grade_limit_mode == "lazy":
            return self._solve_with_grade_limit(source, target)
        cells, poly, cost = self._solve_trace_pair(source, target, None)
        self.last_field_costs.append(cost)
        self.last_mask_iterations.append(0)
        if poly is None:
            self.last_polylines.append(None)
            raise NoPathFoundError(source=source, target=target)
        self.last_polylines.append(poly)
        if self._grade_limit is not None:      # eager mode: verify anyway
            self._assert_no_violation(cells, source, target)
        return cells

    # ------------------------------------------------------------------
    # max_gradient_pct: solve -> check -> mask -> re-solve
    # ------------------------------------------------------------------

    def _verify_if_limited(self, cells, source, target) -> None:
        """Verify a route from a NON-looping code path, if a limit is on.

        Only 'eager' mode reaches the multi-target/multi-source handlers
        with a limit active (lazy mode falls back to per-pair loops), and
        eager mode's mask makes every 8-adjacent step legal by
        construction. Checking anyway is what keeps "never return an
        unverified route" a property of the backend rather than of one
        code path.
        """
        if self._grade_limit is None or not len(cells):
            return
        self._assert_no_violation(cells, source, target)

    def _assert_no_violation(self, cells, source, target) -> None:
        """Never return an unverified route (acceptance criterion 11)."""
        bad = self.grade_violations(cells)
        if bad.size:
            raise RuntimeError(
                f"raster_fim returned a route from {source} to {target} "
                f"with {bad.size} step(s) above the grade limit "
                f"{self._grade_limit:.4g} % in grade_limit_mode='eager'. "
                f"This should not happen — eager mode pre-masks every "
                f"cell steeper than the limit. Please report it; "
                f"grade_limit_mode='lazy' verifies each returned route "
                f"and re-solves until it is valid.")

    def _solve_with_grade_limit(self, source: int, target: int):
        """Lazy-constraint enforcement of ``max_gradient_pct``.

        The solver stays a pure Riemannian solver: the hard limit is a
        direction-dependent constraint, which no metric can represent
        without converting it into a soft price (an ellipse-intersect-slab
        "metric" re-admits 59 % of the forbidden compass). So instead:
        solve, trace, verify the returned cell path with the discrete
        kernel's own binned arithmetic, forbid the offending cells,
        re-solve — until the route is valid or the problem is infeasible.

        Masks are monotone (never unmasked) and every iteration adds at
        least one cell from the finite set of cells steeper than the
        limit, so the loop terminates; it is capped at
        ``max_mask_iterations`` and RAISES on the cap rather than
        returning anything unverified and rather than silently switching
        to the strictly smaller 'eager' feasible set.

        Honest statement of what this buys, and what it does not. The
        returned route is never invalid, and it is optimal for the MASKED
        problem. The masked problem is a long way from the rule: masking
        a cell also forbids the contour-following traverse across it that
        the rule allows, and the measured false-negative rate against the
        Cython kernel is 320/662 = 48.3 % (see ``_MASK_CAVEAT`` and
        ``benchmarks/fuzz_grade_limit_differential.py``). Endpoint-only,
        escalating and connectivity-preserving mask rules were all
        measured on the same fuzz and moved recall by at most 3 points —
        the gap is masking CELLS against a rule on STEPS, not the choice
        of rule. So: trust a route this returns, do not trust a failure;
        run the discrete backend before concluding anything.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        self._check_feasible(source, target)
        mask = np.empty(0, dtype=np.int64)
        cells = poly = None
        cost = float("inf")
        last_max_slope = float("nan")
        best_bad, stale = np.inf, 0
        for it in range(self._max_mask_iterations + 1):
            cells, poly, cost = self._solve_trace_pair(source, target, mask)
            if poly is None:
                self.last_polylines.append(None)
                self.last_field_costs.append(float("inf"))
                self.last_mask_iterations.append(it)
                raise NoPathFoundError(
                    source=source, target=target,
                    add_message=(
                        f" Grade limit {self._grade_limit:.4g} %: no "
                        f"route came back at mask iteration {it} with "
                        f"{mask.size} cell(s) masked (the target is "
                        f"unreachable through the unmasked cells, or the "
                        f"tracer found no descent)"
                        + (f"; the last valid route peaked at "
                           f"{last_max_slope:.4g} % grade."
                           if np.isfinite(last_max_slope) else ".")
                        + " " + self._certificate_note(source, target)
                        + " " + _MASK_CAVEAT))
            bad = self.grade_violations(cells)
            if bad.size == 0:
                self.last_polylines.append(poly)
                self.last_field_costs.append(cost)
                self.last_mask_iterations.append(it)
                return cells                     # VERIFIED route
            last_max_slope = self._max_step_slope(cells)
            if bad.size < best_bad:
                best_bad, stale = bad.size, 0
            else:
                stale += 1
            if stale >= self._mask_patience:
                # Masking more cells is no longer reducing the number of
                # illegal steps: the relaxation has hit the cell-vs-step
                # gap, and grinding to the cap would cost one solve per
                # iteration to reach the same verdict.
                self.last_polylines.append(None)
                self.last_field_costs.append(float("inf"))
                self.last_mask_iterations.append(it)
                raise NoPathFoundError(
                    source=source, target=target,
                    add_message=(
                        f" Grade limit {self._grade_limit:.4g} %: "
                        f"{self._mask_patience} consecutive mask "
                        f"iterations failed to reduce the number of "
                        f"illegal steps (best {int(best_bad)}, still "
                        f"{last_max_slope:.4g} % peak grade with "
                        f"{mask.size} cell(s) masked). "
                        + self._certificate_note(source, target)
                        + " " + _MASK_CAVEAT))
            new_mask = self._cells_to_mask(cells, bad, source, target,
                                           iteration=it, current_mask=mask)
            if new_mask.size == 0:
                self.last_polylines.append(None)
                self.last_field_costs.append(float("inf"))
                self.last_mask_iterations.append(it)
                if self._mask_block == "connectivity":
                    raise NoPathFoundError(
                        source=source, target=target,
                        add_message=(
                            f" Grade limit {self._grade_limit:.4g} %: the "
                            f"mask cannot grow without cutting the last "
                            f"legal corridor — every cell the "
                            f"{last_max_slope:.4g} %-peak route would "
                            f"have to give up is one a LEGAL route still "
                            f"needs, because the limit forbids STEPS and "
                            f"the relaxation can only forbid CELLS. "
                            + self._certificate_note(source, target)
                            + " " + _MASK_CAVEAT))
                raise NoPathFoundError(
                    source=source, target=target,
                    add_message=(
                        f" Grade limit {self._grade_limit:.4g} %: the "
                        f"only cells violating it are the source and/or "
                        f"the target themselves ({last_max_slope:.4g} % "
                        f"peak grade), which cannot be masked. The "
                        f"terminals sit on ground steeper than the "
                        f"limit."))
            merged = np.union1d(mask, new_mask)
            if merged.size == mask.size:
                # Masks are monotone and a masked cell cannot reappear on
                # a route, so this is unreachable in practice — but if it
                # ever happens the loop would re-solve the same problem
                # until the cap. Report it instead of spinning.
                self.last_polylines.append(None)
                self.last_field_costs.append(float("inf"))
                self.last_mask_iterations.append(it)
                raise NoPathFoundError(
                    source=source, target=target,
                    add_message=(
                        f" Grade limit {self._grade_limit:.4g} %: the "
                        f"mask stopped growing at {mask.size} cell(s) "
                        f"while the route still peaked at "
                        f"{last_max_slope:.4g} % grade, so no further "
                        f"iteration can change the answer. " +
                        _MASK_CAVEAT))
            mask = merged
        verdict = self._certificate_note(source, target) + " "
        raise RuntimeError(
            f"raster_fim grade-limit loop hit its cap "
            f"(max_mask_iterations = {self._max_mask_iterations}) for "
            f"{source} -> {target} with {mask.size} cell(s) masked; the "
            f"last route still peaked at {last_max_slope:.4g} % grade "
            f"against a limit of {self._grade_limit:.4g} %. {verdict}"
            f"Raising the "
            f"cap costs one solve + trace per extra iteration. The "
            f"deterministic alternative is "
            f"grade_limit_mode='eager', which pre-masks every cell "
            f"steeper than the limit and always finishes in one solve — "
            f"it is chosen explicitly rather than escalated to "
            f"automatically, because it is a strictly smaller feasible "
            f"set. {_MASK_CAVEAT}")

    def _check_feasible(self, source: int, target: int) -> None:
        """Genuine infeasibility, decided BEFORE any solve.

        Two exact tests, in increasing cost:

        1. a terminal every one of whose neighbour chords exceeds the
           limit cannot be left (or entered) by any route at all;
        2. source and target in different components of the legal-chord
           graph cannot be joined by any legal route.

        Both tests are taken over ``self._cert_offsets`` — the CALLER's
        own step set, not a hardcoded 8-neighbourhood. PathFinder's
        default neighborhood is 'r2', whose knight moves (2,1)/(1,2)
        cross faces no 8-adjacent route can; certifying with 8 would
        report the DEFAULT configuration infeasible while a legal r2
        route exists. A certificate has to speak about the routes the
        caller can actually take.

        Both are properties of the problem, not of the relaxation, so
        they are reported as ``NoPathFoundError`` — the constraint made
        the pair disconnected — rather than as a cap or an iteration
        limit. Without them the loop would mask its way outward until
        reachability collapsed (or until the cap), reporting the right
        answer for the wrong reason after up to ``max_mask_iterations``
        solves.
        """
        n_off = len(self._cert_offsets)
        _steep, isolated, max_pct = self._chord_maps(self._cert_offsets)
        iso, peaks = isolated.ravel(), max_pct.ravel()
        for name, cell in (("source", source), ("target", target)):
            if iso[int(cell)]:
                raise NoPathFoundError(
                    source=source, target=target,
                    add_message=(
                        f" Grade limit {self._grade_limit:.4g} %: the "
                        f"{name} cell {int(cell)} is ISOLATED — every one "
                        f"of its {n_off} neighbour chords (the caller's "
                        f"own step set) is at least "
                        f"{float(peaks[int(cell)]):.4g} % grade, so no "
                        f"route can leave it. INFEASIBLE as posed; "
                        f"masking cannot help. " + _MASK_CAVEAT))

        labels = self._legal_chord_labels()
        if labels is None:
            return                       # too large to certify; say so later
        ok = self._passable().ravel()
        s, t = int(source), int(target)
        if ok[s] and ok[t] and labels[s] != labels[t]:
            raise NoPathFoundError(
                source=source, target=target,
                add_message=(
                    f" Grade limit {self._grade_limit:.4g} %: source and "
                    f"target lie in DIFFERENT components of the "
                    f"legal-chord graph over the caller's {n_off}-step "
                    f"neighborhood, so no route between them has every "
                    f"step within the limit. INFEASIBLE as posed — this "
                    f"is an exact certificate over exactly the steps the "
                    f"caller routes with, not a failure of the mask loop, "
                    f"and it cost no solve. " + _MASK_CAVEAT))

    def _certificate_note(self, source: int, target: int) -> str:
        """What the exact certificate says, for a failure message.

        A failure of the masked relaxation and a genuinely infeasible
        problem deserve different words. ``_check_feasible`` has already
        raised for a certified-infeasible pair, so reaching here means
        either "feasible, and the relaxation gave up" or "not certified".
        """
        labels = self._legal_chord_labels()
        if labels is None:
            return (f"Whether a legal route exists at all was NOT "
                    f"certified: the raster exceeds "
                    f"{self._MAX_CERTIFY_CELLS} cells, above which the "
                    f"legal-chord component test is skipped.")
        return (f"A legal route DOES exist over the caller's "
                f"{len(self._cert_offsets)}-step neighborhood — source "
                f"and target share a component of the legal-chord graph "
                f"— so this is a limit of the masked relaxation, not of "
                f"the problem, and it is the expected outcome about half "
                f"the time (48.3 % measured). A discrete backend "
                f"(raster_gpu / cython) solves the direction-dependent "
                f"rule exactly; run it before concluding the pair cannot "
                f"be connected.")

    def _max_step_slope(self, cells) -> float:
        arr = np.asarray(cells, dtype=np.int64)
        if arr.size < 2:
            return float("nan")
        ra, ca = np.divmod(arr[:-1], self._cols)
        rb, cb = np.divmod(arr[1:], self._cols)
        length = np.hypot((rb - ra).astype(np.float64),
                          (cb - ca).astype(np.float64))
        flat = self._dem.ravel().astype(np.float64)
        hd = np.abs(flat[arr[1:]] - flat[arr[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nanmax(100.0 * hd / (length * self._cell_size)))

    def _cells_to_mask(self, cells, bad_steps, source, target,
                       iteration: int = 0, current_mask=None) -> np.ndarray:
        """Which cells a violating route forbids.

        Two groups, and every cell in both of them is one that CAN
        violate the limit on its own (``|q| > limit/100``) — nothing is
        masked speculatively:

        1. the endpoints of each violating step. "Steep" is the chord
           rule itself (:meth:`_chord_maps`), so for an 8-adjacent
           violating step at least one endpoint is always steep by
           construction — the step that violates IS one of that cell's
           chords. The fallback below only fires for the rare
           non-adjacent jump ``polyline_to_cells`` can emit when a
           grazing sample is dropped at a forbidden cell; there the
           endpoint with the steeper local relief is masked so the loop
           still makes progress;
        2. every steep cell ON THE ROUTE JUST TRIED. The plan's minimal
           rule (endpoints only) is what "lazy" suggests, but measured on
           a broad steep region it makes the route slide outward one cell
           per solve: on the Gaussian-hill case it needed 22 iterations
           at a 20 % limit and blew the 32-iteration cap at 12 %. Adding
           the route's own steep cells is still strictly lazy — only
           cells on routes actually attempted are ever masked, and the
           resulting route stays measurably cheaper than eager mode's
           (93.30 vs 95.67 on that case) — and it converges where the
           minimal rule does not. ``_mask_escalate_after`` delays it (0 =
           from the first iteration, the shipped setting); the 300-trial
           differential fuzz measured endpoints-only, escalate-after-4
           and escalate-after-8 within 1.6 points of it, which is how we
           know the rule is not the lever (see ``_MASK_CAVEAT``).

        Terminals are never masked: masking one would produce a confusing
        "unreachable" on the next pass instead of the real reason. Nor is
        any cell whose removal would cut the last legal corridor —
        :meth:`_drop_disconnecting`.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        arr = np.asarray(cells, dtype=np.int64)
        steep, _isolated, max_pct = self._chord_maps()
        is_steep = steep.ravel()
        peaks = max_pct.ravel()
        out = []
        for i in bad_steps.tolist():
            a, b = int(arr[i]), int(arr[i + 1])
            picked = [c for c in (a, b) if is_steep[c]]
            if not picked:
                picked = [a if peaks[a] >= peaks[b] else b]
            out.extend(picked)
        if iteration >= self._mask_escalate_after:
            out.extend(int(c) for c in arr[is_steep[arr]])
        out = [c for c in out if c != source and c != target]
        self._mask_block = "terminals" if not out else ""
        if not out:
            return np.empty(0, dtype=np.int64)
        if self._mask_keeps_connectivity:
            kept = self._drop_disconnecting(out, current_mask, source,
                                            target)
            if not kept:
                self._mask_block = "connectivity"
                return np.empty(0, dtype=np.int64)
            out = kept
        return np.unique(np.asarray(out, dtype=np.int64))

    #: Connectivity probes per mask growth once the whole batch is
    #: rejected. Each probe is one connected-components pass over the
    #: legal-chord graph, so this bounds the filter's cost; beyond it the
    #: loop keeps what it has verified and lets the stall be reported.
    _MAX_CONNECTIVITY_PROBES = 32

    def _drop_disconnecting(self, cand, current_mask, source,
                            target) -> list:
        """Keep only the candidates that leave a legal route possible.

        Masking a cell forbids EVERY chord through it, including the
        contour-following ones the grade rule permits — which is the
        whole gap between this relaxation and the rule (see
        ``_MASK_CAVEAT``). Without this filter the mask can cut the last
        legal corridor and the next solve then reports the target
        unreachable: a failure of the relaxation dressed up as a
        property of the problem. Measured on the 300-trial differential
        fuzz, that self-inflicted verdict was 82 of 87 failures; with the
        filter it is 3. Recall does not change (the routes were not
        findable by masking either way) — the DIAGNOSIS does, and a
        wrong diagnosis is what makes a user distrust a right answer.

        One components pass covers the common case (the whole batch is
        safe). Only when the batch is rejected does it probe candidate by
        candidate, capped at ``_MAX_CONNECTIVITY_PROBES``.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if self._legal_chord_labels() is None:   # uncertified: no filter
            return cand
        base = ([] if current_mask is None
                else np.asarray(current_mask, dtype=np.int64).ravel()
                .tolist())
        seen, ordered = set(base), []
        for c in cand:
            if c not in seen:
                seen.add(c)
                ordered.append(int(c))
        if not ordered:
            return []

        def connected(removed) -> bool:
            lab = self._components(np.asarray(removed, dtype=np.int64))
            return lab is not None and 0 <= lab[source] == lab[target]

        if connected(base + ordered):
            return ordered                 # the whole batch is safe
        keep: list = []
        for c in ordered[:self._MAX_CONNECTIVITY_PROBES]:
            if connected(base + keep + [c]):
                keep.append(c)
        return keep
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    def _pairwise_limited(self, pairs) -> List[NodeList]:
        """Fallback for a hard grade limit: the mask that certifies a
        route is specific to that route, so one field can no longer serve
        several targets. Each pair pays its own solve/check/mask loop —
        the price of never returning an unverified route.
        """
        results: List[NodeList] = []
        for s, t in pairs:
            try:
                results.append(self._solve_with_grade_limit(int(s), int(t)))
            except NoPathFoundError:
                results.append([])
        return results

    def _single_to_multi(self, source, targets):
        if self._grade_limit is not None and self._grade_limit_mode == "lazy":
            return self._pairwise_limited(
                (source, t) for t in targets)
        t_field = self._solve(np.array([int(source)]))
        paths = self._paths_from_field(
            t_field, np.array([int(source)]),
            np.asarray(targets, dtype=np.int64), reverse=True)
        for t in targets:
            self.last_field_costs.append(self._field_cost(t_field, t))
            self.last_mask_iterations.append(0)
        out = [p if p is not None else [] for p in paths]
        for t, p in zip(targets, out):
            self._verify_if_limited(p, source, t)
        return out

    def _multi_to_single(self, sources, target):
        # Costs are symmetric — under Tier A too: M(x) is a per-cell
        # symmetric form, so the length of a curve does not depend on the
        # direction of travel. One field from the target, then trace each
        # source down to it (polylines already run source -> target).
        if self._grade_limit is not None and self._grade_limit_mode == "lazy":
            return self._pairwise_limited(
                (s, target) for s in sources)
        t_field = self._solve(np.array([int(target)]))
        paths = self._paths_from_field(
            t_field, np.array([int(target)]),
            np.asarray(sources, dtype=np.int64), reverse=False)
        for s in sources:
            self.last_field_costs.append(self._field_cost(t_field, s))
            self.last_mask_iterations.append(0)
        out = [p if p is not None else [] for p in paths]
        for s, p in zip(sources, out):
            self._verify_if_limited(p, s, target)
        return out

    def _multi_to_multi(self, sources, targets, pairwise):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if pairwise and len(sources) != len(targets):
            raise PairwiseError()
        pairs = (list(zip(sources, targets)) if pairwise
                 else [(s, t) for s in sources for t in targets])
        if self._grade_limit is not None and self._grade_limit_mode == "lazy":
            return self._pairwise_limited(pairs)
        results: List[NodeList] = []
        # One solve per unique source, reused across its pairs
        fields: dict = {}
        for s, t in pairs:
            s = int(s)
            if s not in fields:
                # Keep the trace pair WITH the field: with order=2 the
                # returned field is the refined one and is not
                # descent-connected, so a later pair on the same source
                # cannot recover it from _last_trace_host.
                fields[s] = (self._solve(np.array([s])),
                             (self._last_trace_host,
                              self._last_trace_device))
            t_field, t_trace = fields[s]
            paths = self._paths_from_field(
                t_field, np.array([s]), np.array([int(t)]), reverse=True,
                trace=t_trace)
            self.last_field_costs.append(self._field_cost(t_field, t))
            self.last_mask_iterations.append(0)
            results.append(paths[0] if paths[0] is not None else [])
            self._verify_if_limited(results[-1], s, t)
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
        self.last_mask_iterations = []

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

"""
Exactness referee: independent re-pricing of returned paths + cross-backend
comparison under a written tolerance policy (plan item 0.5).

The project's brand is exact routing, but exactness is currently *asserted*
from small runs: constrained suites at 16^2, cross-backend parity at
<= 3000^2. Several known hazards only become reachable on the > 5 km target
workload (silent per-thread relaxation-buffer drops, the ignore_max
0-sentinel, the wall-with-gap residue, the recorded 0.04 % Dijkstra-vs-delta
anomaly). This module makes "exact" testable at 25-100 M cells at O(path)
cost instead of O(raster).

Two layers
----------
LAYER 1 -- independent re-pricing (:func:`reprice_path`). Given a returned
path and the raster it was computed on, walk the polyline and recompute its
cost in float64 from first principles::

    intermediates(dr, dc)   as in _raster_context.pyx:33-96
    cost_factor  = sqrt(dr^2 + dc^2) / (2 + n_intermediates)   (:99-124)
    w            = (raster[u] + sum(intermediates) + raster[v]) * cost_factor
    if gradients:                                     (_dijkstra.pyx:299-315)
        b = min(int(|dem[v] - dem[u]| * bin_factor[d]), n_bins - 1)
        w = w * mult[b] + add[b] * step_len_cells[d]   (mult[b] == inf: edge
                                                       forbidden)

and assert (a) the backend's reported cost equals the re-priced cost within
the tolerance policy, (b) every path cell and every intermediate cell is
passable under the raster's exclusion semantics, (c) consecutive cells are
connected by a legal step of the configured neighbourhood.

This is deliberately a *second implementation*: nothing here imports the
compiled kernels to do arithmetic, and the exclusion sentinel is restated
locally rather than imported. A referee that calls the code under test
proves nothing.

LAYER 2 -- cross-backend comparison (:func:`compare_backends`). For a set of
pairs on one raster, run the available backends, re-price every returned
path, and report each backend's excess over the reference
(``dijkstra_2d_cython``). A backend that returns a *cheaper* path than the
exact reference is an ERROR that names the reference as the suspect -- that
is precisely how the recorded 0.04 % anomaly
(``benchmarks/FUSED_KERNEL_FINDINGS.md:87-89``) should have surfaced. The
same rule catches the wall-with-gap class: reference reports no path where
another backend finds one.

TOLERANCE POLICY
----------------
There is no such policy in the repo today; this is it.

*Why one is needed.* The CPU Dijkstra accumulates ``dist`` in float64
(``_dijkstra.pyx``), while delta-stepping (``_delta_stepping.pyx``: ``float
dist``) and both GPU backends accumulate in float32. At 5-10 km and 1 m
resolution a path is 10^4+ hops, so float32 accumulation alone moves the
reported total by a relative amount that is *not* negligible and must not be
confused with suboptimality.

*Derivation.* Recursive summation of H non-negative terms in a format with
unit roundoff u has the standard forward error bound
``|s_hat - s| <= gamma_H * sum|x_i|`` with ``gamma_H = H*u/(1 - H*u) ~ H*u``;
all edge weights are >= 0, so ``sum|x_i| = s`` and the bound is *relative*:
``H*u``. Each edge weight itself costs a few roundings (three adds, one
multiply, plus the gradient multiply-add), covered by
``EDGE_ROUNDING_OPS = 8`` extra units. Hence

    accumulation_bound(H, u) = (H + 8) * u                    [worst case]

    u(float32) = 2^-24 = 5.96e-8  ->  H = 10^4:  6.0e-4  (0.060 %)
    u(float64) = 2^-53 = 1.11e-16 ->  H = 10^4:  1.1e-12

This is the all-roundings-in-one-direction worst case, so it is used
*without* an extra fudge factor for the optimality gate: anything above it
cannot be explained by float32 arithmetic. Self-consistency (reported vs
re-priced cost of the *same* path) is a sanity check on a different failure
mode -- a predecessor chain that does not correspond to the reported
distance -- so it gets ``SELF_CONSISTENCY_SAFETY = 4`` on top, both because
``/fp:fast`` and ``-ffast-math`` are still applied to the kernels (item 0.8:
reassociation invalidates the strict recursive-summation model) and because
it must only fire on real bugs.

*The three-way classification.* For a candidate path re-priced at ``c`` and
the reference at ``c_ref``, ``excess = (c - c_ref) / c_ref``:

===================== =======================================================
``EXACT``             ``|excess| <=`` the float64-precision bound; the two
                      searches agree to within double rounding.
``WITHIN_FLOAT32``    ``|excess| <=`` the float32 accumulation bound of the
                      participating searches. NOT a proof of optimality --
                      an honest "indistinguishable at this hop count".
``SUBOPTIMAL``        ``excess >`` that bound: genuinely worse, ERROR.
``REFERENCE_SUSPECT`` ``excess < -``bound: cheaper than the exact reference,
                      ERROR against the *reference*.
===================== =======================================================

*FIM is a separate accuracy class, never an exactness failure.*
``raster_fim`` solves a continuous approximation of a different problem; its
route costs are measured between **-1.1 % and +3.7 %** of R2 on real
planning rasters (``benchmarks/EIKONAL_FINDINGS.md``), with no certified
worst-case bound on piecewise-constant surfaces. It is therefore compared
against the documented envelope :data:`FIM_ENVELOPE` and reported as
``WITHIN_ENVELOPE`` / ``OUTSIDE_ENVELOPE``. Its reported ``T[target]`` is a
continuous field value, not the discrete cost of the returned supercover
chain, so self-consistency is skipped for it by construction.

Usage
-----
As a library (the other harnesses and the test suite call it)::

    from benchmarks.exactness_referee import EdgeModel, reprice_path
    model = EdgeModel(steps, max_value=65535)
    priced = reprice_path(path, raster, model)
    assert priced.ok and abs(priced.cost - reported) <= ...

As a CLI, on a real raster or on a generated one::

    .venv/Scripts/python.exe benchmarks/exactness_referee.py \
        --raster benchmarks/realworld_data/cost.tif --pairs pairs.json \
        --neighborhood r2 --backends dijkstra delta-stepping raster_gpu \
        --json benchmarks/results/referee.json

    .venv/Scripts/python.exe benchmarks/exactness_referee.py \
        --synthetic random --size 500 --n-pairs 20 --reps 5 \
        --backends dijkstra delta-stepping --num-threads 8

The second form re-tests the recorded 0.04 % Dijkstra-vs-delta anomaly on
its original geometry (random 500^2), with ``--reps`` because the parallel
suboptimality is nondeterministic. Exit code 1 whenever any pair fails, so
it can gate a lever's acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "IMPASSABLE_CELL_COST",
    "U_FLOAT32",
    "U_FLOAT64",
    "EDGE_ROUNDING_OPS",
    "SELF_CONSISTENCY_SAFETY",
    "FIM_ENVELOPE",
    "BACKEND_PRECISION",
    "APPROXIMATE_BACKENDS",
    "EXACT",
    "WITHIN_FLOAT32",
    "SUBOPTIMAL",
    "REFERENCE_SUSPECT",
    "WITHIN_ENVELOPE",
    "OUTSIDE_ENVELOPE",
    "NO_PATH",
    "INVALID_PATH",
    "FAILING_VERDICTS",
    "intermediate_steps",
    "cost_factor",
    "EdgeModel",
    "RepricedPath",
    "reprice_path",
    "accumulation_bound",
    "self_consistency_tolerance",
    "optimality_tolerance",
    "classify_excess",
    "BackendRun",
    "PairVerdict",
    "RefereeReport",
    "run_backend",
    "judge_run",
    "compare_backends",
    "synthetic_raster",
    "random_pairs",
]

# ==================== CONSTANTS ====================

#: Exclusion sentinel. Restated rather than imported from
#: ``pyorps.core.types`` -- the referee must not inherit the value it checks.
IMPASSABLE_CELL_COST = 65535

#: Sentinel for "exclude nothing". Deliberately outside the uint16 domain so no
#: cell value can equal it. ``ignore_max=False`` maps here, NOT to 0 -- mapping
#: it to 0 makes every zero-cost cell impassable, which is the defect plan item
#: 0.7 fixes. Restated, not imported, for the same reason as above.
NO_EXCLUSION_VALUE = 65536

#: Unit roundoff (eps/2) of the two accumulation formats.
U_FLOAT32 = 2.0 ** -24
U_FLOAT64 = 2.0 ** -53

#: Roundings charged to a single edge weight (3 adds, 1 multiply, gradient
#: multiply-add, cost-factor rounding) on top of the per-hop accumulation.
EDGE_ROUNDING_OPS = 8

#: Extra slack for the reported-vs-re-priced check only (see policy above).
SELF_CONSISTENCY_SAFETY = 4.0

#: Measured envelope of raster_fim route costs relative to R2 on real
#: planning rasters (EIKONAL_FINDINGS.md). Includes the sigma=4 cell-noise
#: class (~+2.3 %). An approximate backend is judged against this, never
#: asserted equal.
FIM_ENVELOPE = (-0.011, 0.037)

#: Unit roundoff of each backend's distance accumulation.
BACKEND_PRECISION = {
    "dijkstra": U_FLOAT64,
    "delta-stepping": U_FLOAT32,
    "delta-stepping-circular": U_FLOAT32,
    "raster_gpu": U_FLOAT32,
    "raster_fim": U_FLOAT32,
}

#: Backends solving a different (continuous) problem -- judged against an
#: envelope, and exempt from the self-consistency and step-legality gates.
APPROXIMATE_BACKENDS = frozenset({"raster_fim"})

# Verdicts
EXACT = "EXACT"
WITHIN_FLOAT32 = "WITHIN_FLOAT32"
SUBOPTIMAL = "SUBOPTIMAL"
REFERENCE_SUSPECT = "REFERENCE_SUSPECT"
WITHIN_ENVELOPE = "WITHIN_ENVELOPE"
OUTSIDE_ENVELOPE = "OUTSIDE_ENVELOPE"
NO_PATH = "NO_PATH"
INVALID_PATH = "INVALID_PATH"

FAILING_VERDICTS = frozenset(
    {SUBOPTIMAL, REFERENCE_SUSPECT, OUTSIDE_ENVELOPE, NO_PATH, INVALID_PATH})

#: Violations kept verbatim per path (the count is always complete).
MAX_RECORDED_VIOLATIONS = 32


# ==================== LAYER 1: THE EDGE MODEL ====================

def intermediate_steps(dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    """Intermediate cells of one step, as (ddr, ddc) offsets from ``u``.

    Independent re-implementation of ``_calculate_intermediate_steps_cython``
    (``_raster_context.pyx:33-96``): no intermediates for a cardinal step,
    the two orthogonal components for a single diagonal, and floor/ceil
    samples of the linear interpolant for every longer step.
    """
    dr = int(dr)
    dc = int(dc)
    abs_dr = abs(dr)
    abs_dc = abs(dc)

    if abs_dr + abs_dc <= 1:
        return ()
    if max(abs_dr, abs_dc) == 1:
        return ((dr, 0), (0, dc))

    k = max(abs_dr, abs_dc)
    out: list[tuple[int, int]] = []
    for p in range(1, k):
        dr_k = (p * dr) / k
        dc_k = (p * dc) / k
        floor_pt = (int(math.floor(dr_k)), int(math.floor(dc_k)))
        out.append(floor_pt)
        ceil_pt = (int(math.ceil(dr_k)), int(math.ceil(dc_k)))
        if ceil_pt != floor_pt:
            out.append(ceil_pt)
    return tuple(out)


def cost_factor(dr: int, dc: int, n_intermediates: int,
                precision: str = "float32") -> float:
    """Movement cost factor ``sqrt(dr^2 + dc^2) / (2 + n_intermediates)``.

    ``precision="float32"`` reproduces the value the kernels actually hold
    (``StepData.cost_factor`` is a float32, built by the float32-native
    ``_get_cost_factor_cython_f32``). The GPU host builds the same table in
    float64 and rounds once (``traversal_gpu.prepare_step_lookup_tables``);
    the two differ by at most 1 ulp on ~15 % of directions, which is inside
    :data:`EDGE_ROUNDING_OPS`. ``precision="float64"`` prices the exact real
    model instead.
    """
    n = int(n_intermediates)
    if precision == "float64":
        return math.sqrt(float(dr * dr + dc * dc)) / (2.0 + n)
    if precision != "float32":
        raise ValueError(
            f"precision must be 'float32' or 'float64', got {precision!r}")
    distance = np.sqrt(np.float32(dr * dr + dc * dc), dtype=np.float32)
    return float(distance / np.float32(2.0 + n))


def _lut_arrays(gradient_luts: Any) -> dict[str, Any]:
    """Normalize a GradientLUTs object or an equivalent mapping.

    Only the five fields the kernels read are used: ``mult``, ``add``,
    ``bin_factor``, ``step_len_cells``, ``n_bins`` -- all float32 in the
    kernel (``_dijkstra.pyx:709-720``).
    """
    def get(name):
        if isinstance(gradient_luts, dict):
            if name not in gradient_luts:
                raise ValueError(f"gradient_luts is missing '{name}'")
            return gradient_luts[name]
        if not hasattr(gradient_luts, name):
            raise ValueError(f"gradient_luts is missing '{name}'")
        return getattr(gradient_luts, name)

    mult = np.ascontiguousarray(get("mult"), dtype=np.float32)
    add = np.ascontiguousarray(get("add"), dtype=np.float32)
    bin_factor = np.ascontiguousarray(get("bin_factor"), dtype=np.float32)
    step_len = np.ascontiguousarray(get("step_len_cells"), dtype=np.float32)
    n_bins = int(get("n_bins"))
    if mult.shape[0] != n_bins or add.shape[0] != n_bins:
        raise ValueError("gradient LUT sizes do not match n_bins")
    return dict(mult=mult, add=add, bin_factor=bin_factor,
                step_len=step_len, n_bins=n_bins)


class EdgeModel:
    """The edge model a path is re-priced under.

    Parameters:
        steps: (n_dirs, 2) neighbourhood steps (dr, dc).
        max_value: Exclusion sentinel. ``65535`` mirrors ``ignore_max=True``;
            ``0`` mirrors ``ignore_max=False``, which makes every 0-cost cell
            impassable (``cython_api.py:28`` -- plan item 0.7). The referee
            reproduces whichever semantics the run used and flags the
            0-sentinel situation as a note.
        dem: Optional DEM aligned to the raster; stored as float32 because
            that is what the kernels read.
        gradient_luts: Optional ``GradientLUTs`` (or a mapping with the same
            fields) enabling the per-edge gradient term.
        cost_factor_precision: See :func:`cost_factor`.
    """

    def __init__(self, steps, max_value: int = IMPASSABLE_CELL_COST,
                 dem=None, gradient_luts=None,
                 cost_factor_precision: str = "float32"):
        self.steps = np.asarray(steps, dtype=np.int64).reshape(-1, 2)
        if self.steps.shape[0] == 0:
            raise ValueError("steps must not be empty")
        self.max_value = int(max_value)
        self.cost_factor_precision = cost_factor_precision
        self.dem = (None if dem is None
                    else np.ascontiguousarray(dem, dtype=np.float32))

        self.gradient = None
        if gradient_luts is not None:
            if self.dem is None:
                raise ValueError(
                    "gradient_luts require a dem aligned to the raster")
            self.gradient = _lut_arrays(gradient_luts)
            for name in ("bin_factor", "step_len"):
                if self.gradient[name].shape[0] < self.steps.shape[0]:
                    raise ValueError(
                        f"gradient LUT '{name}' has "
                        f"{self.gradient[name].shape[0]} entries for "
                        f"{self.steps.shape[0]} directions")

        # Direction table keyed by (dr, dc); first occurrence wins, matching
        # the kernels' index order.
        self._edges: dict[tuple[int, int], tuple[int, tuple, float]] = {}
        for i in range(self.steps.shape[0]):
            key = (int(self.steps[i, 0]), int(self.steps[i, 1]))
            if key in self._edges:
                continue
            inter = intermediate_steps(*key)
            self._edges[key] = (
                i, inter, cost_factor(key[0], key[1], len(inter),
                                      cost_factor_precision))

    def is_legal_step(self, dr: int, dc: int) -> bool:
        """Whether (dr, dc) belongs to the configured neighbourhood."""
        return (int(dr), int(dc)) in self._edges

    def edge(self, dr: int, dc: int) -> tuple[int, tuple, float]:
        """(direction index, intermediates, cost factor) for a step.

        Steps outside the neighbourhood get index ``-1`` and are still
        priced by the same formula, so an illegal step is reported *and*
        costed rather than aborting the walk.
        """
        key = (int(dr), int(dc))
        hit = self._edges.get(key)
        if hit is not None:
            return hit
        inter = intermediate_steps(*key)
        return (-1, inter,
                cost_factor(key[0], key[1], len(inter),
                            self.cost_factor_precision))


@dataclass
class RepricedPath:
    """Result of an independent float64 re-pricing of one path."""

    cost: float
    hops: int
    n_cells: int
    length_cells: float
    violations: list[dict] = field(default_factory=list)
    n_violations: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No violations and a finite cost."""
        return self.n_violations == 0 and math.isfinite(self.cost)

    def to_dict(self) -> dict:
        return dict(cost=self.cost, hops=self.hops, n_cells=self.n_cells,
                    length_cells=self.length_cells,
                    n_violations=self.n_violations,
                    violations=self.violations, notes=self.notes)


def _to_rowcol(path, cols: int) -> np.ndarray:
    """Accept flat indices or (row, col) pairs; return an (n, 2) int array."""
    arr = np.asarray(path)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim != 1:
        raise ValueError(
            f"path must be flat indices or (n, 2) row/col pairs, got shape "
            f"{arr.shape}")
    flat = arr.astype(np.int64, copy=False)
    return np.stack([flat // cols, flat % cols], axis=1)


def reprice_path(path, raster: np.ndarray, model: EdgeModel,
                 strict: bool = True) -> RepricedPath:
    """Re-price a returned path in float64, independently of the kernels.

    Parameters:
        path: Flat cell indices (as every backend returns) or (n, 2)
            row/col pairs.
        raster: The uint16 cost raster the path was computed on.
        model: The :class:`EdgeModel` the run used.
        strict: When False, illegal steps and revisited cells are recorded
            as notes instead of violations -- used for the approximate
            backends, whose supercover chains legally contain grazing jumps
            (``eikonal_gpu.polyline_to_cells``).

    Returns:
        :class:`RepricedPath`. ``cost`` is ``inf`` if the walk hits a
        forbidden gradient bin or an impassable cell.
    """
    rows, cols = int(raster.shape[0]), int(raster.shape[1])
    cells = _to_rowcol(path, cols)
    n_cells = int(cells.shape[0])

    violations: list[dict] = []
    n_violations = 0
    notes: list[str] = []

    def violate(kind: str, **info):
        nonlocal n_violations
        n_violations += 1
        if len(violations) < MAX_RECORDED_VIOLATIONS:
            violations.append(dict(kind=kind, **info))

    if n_cells == 0:
        return RepricedPath(cost=math.inf, hops=0, n_cells=0,
                            length_cells=0.0,
                            violations=[dict(kind="empty_path")],
                            n_violations=1)

    if model.max_value == 0 and bool(np.any(raster == 0)):
        notes.append(
            "max_value=0 was passed explicitly: every 0-cost cell is treated "
            "as impassable. ignore_max=False does NOT mean this -- it maps to "
            "NO_EXCLUSION_VALUE.")

    grad = model.gradient
    dem = model.dem
    if grad is not None and (dem.shape[0] != rows or dem.shape[1] != cols):
        raise ValueError(
            f"DEM shape {dem.shape} does not match the raster ({rows}, {cols})")

    # (b) every path cell in bounds and passable
    for i in range(n_cells):
        r, c = int(cells[i, 0]), int(cells[i, 1])
        if r < 0 or r >= rows or c < 0 or c >= cols:
            violate("out_of_bounds", position=i, cell=[r, c])
        elif int(raster[r, c]) == model.max_value:
            violate("impassable_cell", position=i, cell=[r, c])

    if strict:
        seen = set()
        for i in range(n_cells):
            key = (int(cells[i, 0]), int(cells[i, 1]))
            if key in seen:
                violate("revisited_cell", position=i, cell=list(key))
            seen.add(key)

    total = 0.0
    length_cells = 0.0
    forbidden = False

    for i in range(n_cells - 1):
        ur, uc = int(cells[i, 0]), int(cells[i, 1])
        vr, vc = int(cells[i + 1, 0]), int(cells[i + 1, 1])
        dr, dc = vr - ur, vc - uc

        if dr == 0 and dc == 0:
            violate("zero_step", position=i, cell=[ur, uc])
            continue

        d_index, intermediates, factor = model.edge(dr, dc)
        length_cells += math.sqrt(float(dr * dr + dc * dc))

        # (c) legal step of the configured neighbourhood
        if d_index < 0:
            if strict:
                violate("illegal_step", position=i, step=[dr, dc])
            else:
                notes.append(f"step ({dr}, {dc}) at hop {i} is outside the "
                             f"configured neighbourhood")

        if (ur < 0 or ur >= rows or uc < 0 or uc >= cols or
                vr < 0 or vr >= rows or vc < 0 or vc >= cols):
            forbidden = True
            continue

        cell_sum = float(raster[ur, uc]) + float(raster[vr, vc])
        for ddr, ddc in intermediates:
            ir, ic = ur + ddr, uc + ddc
            if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
                violate("intermediate_out_of_bounds", position=i,
                        cell=[ir, ic])
                forbidden = True
                continue
            if int(raster[ir, ic]) == model.max_value:
                violate("impassable_intermediate", position=i, cell=[ir, ic])
                forbidden = True
            cell_sum += float(raster[ir, ic])

        weight = cell_sum * factor

        if grad is not None:
            if d_index < 0:
                notes.append(
                    f"gradient term skipped at hop {i}: no direction index "
                    f"for step ({dr}, {dc})")
            else:
                height_diff = abs(float(dem[vr, vc]) - float(dem[ur, uc]))
                slope_bin = int(height_diff
                                * float(grad["bin_factor"][d_index]))
                if slope_bin >= grad["n_bins"]:
                    slope_bin = grad["n_bins"] - 1
                mult = float(grad["mult"][slope_bin])
                if not math.isfinite(mult):
                    violate("forbidden_gradient", position=i,
                            step=[dr, dc], slope_bin=slope_bin)
                    forbidden = True
                else:
                    weight = (weight * mult
                              + float(grad["add"][slope_bin])
                              * float(grad["step_len"][d_index]))

        total += weight

    return RepricedPath(
        cost=math.inf if forbidden else total,
        hops=n_cells - 1,
        n_cells=n_cells,
        length_cells=length_cells,
        violations=violations,
        n_violations=n_violations,
        notes=notes,
    )


# ==================== TOLERANCE POLICY ====================

def accumulation_bound(hops: int, unit_roundoff: float = U_FLOAT32) -> float:
    """Worst-case relative error of a hop-by-hop distance accumulation.

    ``(hops + EDGE_ROUNDING_OPS) * u`` -- the recursive-summation bound
    ``gamma_H ~ H*u`` for non-negative terms plus the per-edge roundings.
    See the module docstring for the derivation.
    """
    return (int(hops) + EDGE_ROUNDING_OPS) * float(unit_roundoff)


def self_consistency_tolerance(hops: int,
                               unit_roundoff: float = U_FLOAT32) -> float:
    """Tolerance for "reported cost == re-priced cost of the same path"."""
    return SELF_CONSISTENCY_SAFETY * accumulation_bound(hops, unit_roundoff)


def optimality_tolerance(hops_ref: int, hops_cand: int,
                         u_ref: float = U_FLOAT64,
                         u_cand: float = U_FLOAT32) -> float:
    """Largest excess still explainable by the two searches' arithmetic.

    Both paths are re-priced in float64, so the only slack is that each
    search *selected* its path under its own precision and may have settled
    a float32-tied but float64-worse route.
    """
    return (accumulation_bound(hops_ref, u_ref)
            + accumulation_bound(hops_cand, u_cand))


def classify_excess(excess: float, tol_optimality: float,
                    tol_exact: float = 0.0, approximate: bool = False,
                    envelope: tuple[float, float] = FIM_ENVELOPE) -> str:
    """Map a relative excess over the reference onto a verdict."""
    if approximate:
        low, high = envelope
        return WITHIN_ENVELOPE if low <= excess <= high else OUTSIDE_ENVELOPE
    if excess < -tol_optimality:
        return REFERENCE_SUSPECT
    if abs(excess) <= tol_exact:
        return EXACT
    if excess <= tol_optimality:
        return WITHIN_FLOAT32
    return SUBOPTIMAL


# ==================== LAYER 2: BACKEND RUNNERS ====================

@dataclass
class BackendRun:
    """One backend's answer for one pair."""

    backend: str
    source: int
    target: int
    path: list[int] | None
    reported_cost: float | None
    runtime_s: float
    error: str | None = None

    def to_dict(self) -> dict:
        return dict(backend=self.backend, source=self.source,
                    target=self.target,
                    n_cells=0 if self.path is None else len(self.path),
                    reported_cost=self.reported_cost,
                    runtime_s=self.runtime_s, error=self.error)


def _run_dijkstra(raster, steps, source, target, cfg) -> tuple[list, float | None]:
    from pyorps.utils.path_algorithms import dijkstra_2d_cython
    path = dijkstra_2d_cython(
        raster, steps, np.uint32(source), np.uint32(target),
        max_value=cfg["max_value"], dem=cfg["dem"],
        gradient_luts=cfg["gradient_luts"])
    return [int(p) for p in path], None


def _run_delta(raster, steps, source, target, cfg) -> tuple[list, float | None]:
    from pyorps.utils.path_algorithms import delta_stepping_2d_persistent
    path = delta_stepping_2d_persistent(
        raster, steps, np.uint64(source), np.uint64(target),
        delta=cfg.get("delta_cpu", 100.0), max_value=cfg["max_value"],
        num_threads=cfg.get("num_threads", 0),
        margin=cfg.get("margin", 1.00001))
    return [int(p) for p in path], None


def _run_raster_gpu(raster, steps, source, target, cfg) -> tuple[list, float | None]:
    """Calls the SSSP entry point directly so ``dist[target]`` is available
    as the backend's *reported* cost for the self-consistency gate."""
    from pyorps.utils.sssp_gpu import sssp_raster_gpu
    targets = np.array([int(target)], dtype=np.int32)
    dist, pred = sssp_raster_gpu(
        raster, steps, source_idx=int(source), delta=cfg.get("delta", "auto"),
        ignore_max=cfg["ignore_max"], target_indices=targets,
        margin=cfg.get("margin", 1.00001), return_predecessor=True,
        dem=cfg["dem"] if cfg["gradient_luts"] is not None else None,
        gradient_luts=cfg["gradient_luts"])
    d = float(dist[int(target)])
    if not math.isfinite(d) or d >= 1e29:
        return [], math.inf
    path = [int(target)]
    current = int(target)
    limit = int(raster.shape[0]) * int(raster.shape[1])
    while current != int(source):
        p = int(pred[current])
        if p < 0 or len(path) > limit:
            raise RuntimeError(
                f"predecessor chain broken at cell {current} although "
                f"dist[target] = {d} is finite")
        path.append(p)
        current = p
    path.reverse()
    return path, d


def _run_raster_fim(raster, steps, source, target, cfg) -> tuple[list, float | None]:
    from pyorps.graph.api.raster_fim_api import RasterFIMAPI
    api = RasterFIMAPI(raster, steps, ignore_max=cfg["ignore_max"],
                       **cfg.get("fim_kwargs", {}))
    path = api.shortest_path(int(source), int(target))
    reported = (float(api.last_field_costs[0])
                if api.last_field_costs else None)
    return [int(p) for p in path], reported


BACKEND_RUNNERS: dict[str, Callable] = {
    "dijkstra": _run_dijkstra,
    "delta-stepping": _run_delta,
    "raster_gpu": _run_raster_gpu,
    "raster_fim": _run_raster_fim,
}


def run_backend(backend: str, raster: np.ndarray, steps: np.ndarray,
                source: int, target: int, cfg: dict) -> BackendRun:
    """Run one backend on one pair, capturing failures instead of raising."""
    runner = BACKEND_RUNNERS.get(backend)
    if runner is None:
        raise KeyError(
            f"unknown backend {backend!r}; known: "
            f"{sorted(BACKEND_RUNNERS)}")
    t0 = time.perf_counter()
    try:
        path, reported = runner(raster, steps, source, target, cfg)
    except Exception as exc:  # a crashing backend is a result, not an abort
        return BackendRun(backend, int(source), int(target), None, None,
                          time.perf_counter() - t0,
                          f"{type(exc).__name__}: {exc}")
    dt = time.perf_counter() - t0
    if path is None or len(path) == 0:
        return BackendRun(backend, int(source), int(target), None, reported,
                          dt, None)
    return BackendRun(backend, int(source), int(target), list(path), reported,
                      dt, None)


# ==================== LAYER 2: COMPARISON ====================

@dataclass
class PairVerdict:
    """One (pair, backend) judgement against the reference."""

    backend: str
    source: int
    target: int
    verdict: str
    excess_rel: float | None
    tolerance: float
    cost: float | None
    reference_cost: float | None
    hops: int
    reference_hops: int
    self_consistency: str
    reported_cost: float | None
    runtime_s: float
    violations: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    rep: int = 0

    @property
    def failed(self) -> bool:
        return self.verdict in FAILING_VERDICTS or self.error is not None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["failed"] = self.failed
        return d


@dataclass
class RefereeReport:
    """Everything one referee run judged."""

    reference: str
    n_pairs: int
    raster_shape: tuple[int, int]
    neighborhood: str
    verdicts: list[PairVerdict] = field(default_factory=list)
    reference_paths: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[PairVerdict]:
        return [v for v in self.verdicts if v.failed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, dict]:
        """Per-backend aggregate: counts, max and median excess."""
        out: dict[str, dict] = {}
        for backend in dict.fromkeys(v.backend for v in self.verdicts):
            rows = [v for v in self.verdicts if v.backend == backend]
            excesses = [v.excess_rel for v in rows
                        if v.excess_rel is not None]
            counts: dict[str, int] = {}
            for v in rows:
                counts[v.verdict] = counts.get(v.verdict, 0) + 1
            out[backend] = dict(
                n=len(rows),
                verdicts=counts,
                n_failed=sum(1 for v in rows if v.failed),
                max_excess=max(excesses) if excesses else None,
                min_excess=min(excesses) if excesses else None,
                median_excess=(statistics.median(excesses)
                               if excesses else None),
                total_runtime_s=sum(v.runtime_s for v in rows),
            )
        return out

    def to_dict(self) -> dict:
        return dict(reference=self.reference, n_pairs=self.n_pairs,
                    raster_shape=list(self.raster_shape),
                    neighborhood=self.neighborhood,
                    tolerance_policy=dict(
                        u_float32=U_FLOAT32, u_float64=U_FLOAT64,
                        edge_rounding_ops=EDGE_ROUNDING_OPS,
                        self_consistency_safety=SELF_CONSISTENCY_SAFETY,
                        fim_envelope=list(FIM_ENVELOPE)),
                    summary=self.summary(),
                    verdicts=[v.to_dict() for v in self.verdicts],
                    reference_paths=self.reference_paths,
                    notes=self.notes, ok=self.ok)

    def format_table(self) -> str:
        lines = [
            f"exactness referee -- reference: {self.reference}, "
            f"raster {self.raster_shape[0]}x{self.raster_shape[1]}, "
            f"neighborhood {self.neighborhood}, {self.n_pairs} pair(s)",
            f"{'backend':<22}{'src->tgt':>18}{'hops':>8}"
            f"{'excess':>13}{'tol':>11}  verdict",
        ]
        for v in self.verdicts:
            excess = ("      n/a" if v.excess_rel is None
                      else f"{v.excess_rel * 100:+9.5f} %")
            lines.append(
                f"{v.backend:<22}{f'{v.source}->{v.target}':>18}"
                f"{v.hops:>8}{excess:>13}{v.tolerance * 100:>10.5f}%  "
                f"{v.verdict}"
                + (f"  [{v.error}]" if v.error else ""))
        for backend, agg in self.summary().items():
            lines.append(
                f"  {backend:<20} failed={agg['n_failed']}/{agg['n']} "
                f"max_excess="
                + ("n/a" if agg["max_excess"] is None
                   else f"{agg['max_excess'] * 100:+.5f} %"))
        lines.append("OK" if self.ok else
                     f"FAIL -- {len(self.failures)} failing judgement(s)")
        return "\n".join(lines)


def _self_consistency(run: BackendRun, priced: RepricedPath,
                      unit_roundoff: float, approximate: bool) -> str:
    """(a) reported cost == independently re-priced cost of the same path."""
    if approximate:
        return "SKIPPED_APPROXIMATE"
    if run.reported_cost is None:
        return "SKIPPED_NO_REPORTED_COST"
    if not math.isfinite(run.reported_cost) or not math.isfinite(priced.cost):
        return ("OK" if math.isinf(run.reported_cost)
                and math.isinf(priced.cost) else "MISMATCH")
    tol = self_consistency_tolerance(priced.hops, unit_roundoff)
    denom = max(abs(priced.cost), 1e-300)
    return ("OK" if abs(run.reported_cost - priced.cost) / denom <= tol
            else "MISMATCH")


def judge_run(run: BackendRun, reference: RepricedPath | None,
              reference_backend: str, raster: np.ndarray,
              model: EdgeModel) -> PairVerdict:
    """Judge one backend run against an already re-priced reference path."""
    approximate = run.backend in APPROXIMATE_BACKENDS
    u_cand = BACKEND_PRECISION.get(run.backend, U_FLOAT32)
    u_ref = BACKEND_PRECISION.get(reference_backend, U_FLOAT64)
    ref_cost = None if reference is None else reference.cost
    ref_hops = 0 if reference is None else reference.hops

    if run.error is not None:
        return PairVerdict(
            run.backend, run.source, run.target, INVALID_PATH, None, 0.0,
            None, ref_cost, 0, ref_hops, "SKIPPED_NO_PATH",
            run.reported_cost, run.runtime_s, error=run.error)

    if run.path is None:
        # No path where the reference found one is the wall-with-gap class;
        # no path on either side is a legitimate agreement.
        verdict = NO_PATH if (reference is not None
                              and math.isfinite(reference.cost)) else EXACT
        return PairVerdict(
            run.backend, run.source, run.target, verdict, None, 0.0,
            None, ref_cost, 0, ref_hops, "SKIPPED_NO_PATH",
            run.reported_cost, run.runtime_s)

    priced = reprice_path(run.path, raster, model, strict=not approximate)
    consistency = _self_consistency(run, priced, u_cand, approximate)

    tol = optimality_tolerance(ref_hops, priced.hops, u_ref, u_cand)
    tol_exact = optimality_tolerance(ref_hops, priced.hops,
                                     U_FLOAT64, U_FLOAT64)

    if reference is None or not math.isfinite(ref_cost):
        # Nothing to compare against: a finite path where the reference has
        # none indicts the reference (the wall-with-gap class).
        verdict = (REFERENCE_SUSPECT if priced.ok and not approximate
                   else EXACT)
        excess = None
    elif ref_cost == 0.0:
        # Relative excess is undefined; weights are non-negative, so any
        # positive cost against a zero reference is a genuine excess.
        excess = None
        if priced.cost <= 0.0:
            verdict = EXACT
        else:
            verdict = WITHIN_ENVELOPE if approximate else SUBOPTIMAL
    else:
        excess = (priced.cost - ref_cost) / ref_cost
        verdict = classify_excess(excess, tol, tol_exact, approximate)

    if not priced.ok:
        verdict = INVALID_PATH
    elif consistency == "MISMATCH":
        verdict = INVALID_PATH

    return PairVerdict(
        backend=run.backend, source=run.source, target=run.target,
        verdict=verdict, excess_rel=excess, tolerance=tol,
        cost=priced.cost, reference_cost=ref_cost, hops=priced.hops,
        reference_hops=ref_hops, self_consistency=consistency,
        reported_cost=run.reported_cost, runtime_s=run.runtime_s,
        violations=priced.violations, notes=priced.notes)


def compare_backends(raster: np.ndarray, steps: np.ndarray,
                     pairs: Iterable[Sequence[int]],
                     backends: Sequence[str] = ("dijkstra", "delta-stepping"),
                     reference: str = "dijkstra",
                     ignore_max: bool = True,
                     dem=None, gradient_luts=None,
                     neighborhood: str = "custom",
                     reps: int = 1,
                     cfg: dict | None = None) -> RefereeReport:
    """Layer 2: run the backends on every pair and judge them.

    The reference is always re-priced first; every other backend is judged
    against the reference's re-priced (float64) cost, never against its own
    reported one.

    ``reps > 1`` repeats every non-reference backend per pair: parallel
    delta-stepping is nondeterministically suboptimal at >= 2 threads
    (FUSED_KERNEL_FINDINGS.md), so a single run can pass by luck.
    """
    raster = np.asarray(raster)
    steps = np.asarray(steps, dtype=np.int8)
    max_value = IMPASSABLE_CELL_COST if ignore_max else NO_EXCLUSION_VALUE
    model = EdgeModel(steps, max_value=max_value, dem=dem,
                      gradient_luts=gradient_luts)

    run_cfg = dict(max_value=max_value, ignore_max=ignore_max, dem=dem,
                   gradient_luts=gradient_luts)
    run_cfg.update(cfg or {})

    ordered = [reference] + [b for b in backends if b != reference]
    pairs = [(int(s), int(t)) for s, t in pairs]

    report = RefereeReport(
        reference=reference, n_pairs=len(pairs),
        raster_shape=(int(raster.shape[0]), int(raster.shape[1])),
        neighborhood=neighborhood)
    if not ignore_max and bool(np.any(raster == 0)):
        report.notes.append(
            "ignore_max=False with 0-cost cells present: nothing is excluded, "
            "so those cells are free to traverse in every backend "
            "(plan item 0.7)")

    for source, target in pairs:
        ref_run = run_backend(reference, raster, steps, source, target,
                              run_cfg)
        ref_priced = (None if ref_run.path is None
                      else reprice_path(ref_run.path, raster, model))
        report.reference_paths.append(dict(
            source=source, target=target, runtime_s=ref_run.runtime_s,
            error=ref_run.error,
            **({} if ref_priced is None else ref_priced.to_dict())))
        if ref_priced is not None and not ref_priced.ok:
            report.notes.append(
                f"reference path {source}->{target} violates the edge model: "
                f"{ref_priced.violations[:3]}")

        for backend in ordered:
            for rep in range(1 if backend == reference else max(1, reps)):
                run = (ref_run if backend == reference
                       else run_backend(backend, raster, steps, source,
                                        target, run_cfg))
                verdict = judge_run(run, ref_priced, reference, raster, model)
                verdict.rep = rep
                report.verdicts.append(verdict)

    return report


# ==================== SYNTHETIC WORKLOADS ====================

def synthetic_raster(pattern: str, n: int, rng) -> np.ndarray:
    """Rasters that exercise the referee's own failure classes.

    ``random`` / ``heavy_tail`` mirror the delta-stepping and GPU benchmark
    patterns (so the 0.04 % anomaly's original geometry is reproducible);
    ``walls`` exercises the exclusion sentinel and narrow openings;
    ``zero_cost`` exercises 0-cost edges (the sub-delta / re-open protocol
    and the ignore_max 0-sentinel).
    """
    if pattern == "random":
        return rng.integers(1, 200, (n, n), dtype=np.uint16)
    if pattern == "heavy_tail":
        return np.clip(rng.lognormal(2.0, 1.5, (n, n)), 1,
                       5000).astype(np.uint16)
    if pattern == "walls":
        raster = rng.integers(1, 50, (n, n), dtype=np.uint16)
        for col in range(n // 10, n, max(1, n // 10)):
            raster[:, col] = IMPASSABLE_CELL_COST
            gap = int(rng.integers(1, n - 1))
            raster[gap, col] = 1
        return raster
    if pattern == "zero_cost":
        raster = rng.integers(1, 200, (n, n), dtype=np.uint16)
        raster[rng.random((n, n)) < 0.05] = 0
        return raster
    raise ValueError(f"unknown synthetic pattern {pattern!r}")


def random_pairs(raster: np.ndarray, n_pairs: int, rng,
                 max_value: int = IMPASSABLE_CELL_COST
                 ) -> list[tuple[int, int]]:
    """Random distinct passable (source, target) flat-index pairs."""
    passable = np.flatnonzero(raster.ravel() != max_value)
    if passable.size < 2:
        raise ValueError("raster has fewer than two passable cells")
    pairs = []
    while len(pairs) < n_pairs:
        s, t = rng.choice(passable, size=2, replace=False)
        if s != t:
            pairs.append((int(s), int(t)))
    return pairs


# ==================== CLI ====================

def _load_raster(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        import rasterio
        with rasterio.open(p) as src:
            arr = src.read(1)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return np.ascontiguousarray(arr)


def _load_pairs(path: str, cols: int) -> list[tuple[int, int]]:
    """Pairs as ``[[src_idx, tgt_idx], ...]`` or
    ``[{"source": [row, col], "target": [row, col]}, ...]``."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data["pairs"]
    out = []
    for entry in data:
        if isinstance(entry, dict):
            sr, sc = entry["source"]
            tr, tc = entry["target"]
            out.append((int(sr) * cols + int(sc), int(tr) * cols + int(tc)))
        else:
            out.append((int(entry[0]), int(entry[1])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Independent re-pricing + cross-backend exactness gate.")
    ap.add_argument("--raster", default=None,
                    help="cost raster (.tif or .npy, uint16)")
    ap.add_argument("--pairs", default=None,
                    help="JSON pair list (flat indices or row/col)")
    ap.add_argument("--synthetic", default=None,
                    choices=["random", "heavy_tail", "walls", "zero_cost"],
                    help="generate the raster and the pairs instead")
    ap.add_argument("--size", type=int, default=500,
                    help="synthetic raster edge length")
    ap.add_argument("--n-pairs", type=int, default=4,
                    help="synthetic pair count")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--reps", type=int, default=1,
                    help="runs per pair per non-reference backend")
    ap.add_argument("--dem", default=None, help="optional DEM (.tif or .npy)")
    ap.add_argument("--neighborhood", default="r2")
    ap.add_argument("--backends", nargs="+",
                    default=["dijkstra", "delta-stepping"],
                    choices=sorted(BACKEND_RUNNERS))
    ap.add_argument("--reference", default="dijkstra",
                    choices=sorted(BACKEND_RUNNERS))
    ap.add_argument("--num-threads", type=int, default=0,
                    help="delta-stepping threads (0 = auto)")
    ap.add_argument("--delta-cpu", type=float, default=100.0)
    ap.add_argument("--margin", type=float, default=1.00001)
    ap.add_argument("--no-ignore-max", action="store_true",
                    help="ignore_max=False: sentinel becomes 0")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args(argv)

    from pyorps.utils.neighborhood import get_neighborhood_steps

    ignore_max = not args.no_ignore_max
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        raster = synthetic_raster(args.synthetic, args.size, rng)
        pairs = random_pairs(
            raster, args.n_pairs, rng,
            IMPASSABLE_CELL_COST if ignore_max else NO_EXCLUSION_VALUE)
    elif args.raster and args.pairs:
        raster = _load_raster(args.raster)
        pairs = _load_pairs(args.pairs, int(raster.shape[1]))
    else:
        ap.error("pass either --raster with --pairs, or --synthetic")

    dem = None if args.dem is None else _load_raster(args.dem).astype(
        np.float32)
    steps = get_neighborhood_steps(args.neighborhood)

    report = compare_backends(
        raster, steps, pairs, backends=args.backends,
        reference=args.reference, ignore_max=ignore_max,
        dem=dem, neighborhood=args.neighborhood, reps=args.reps,
        cfg=dict(num_threads=args.num_threads, delta_cpu=args.delta_cpu,
                 margin=args.margin))

    print(report.format_table())
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=1))
        print(f"report -> {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

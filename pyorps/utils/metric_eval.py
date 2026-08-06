"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Multi-metric path evaluation (reporting ONLY — never route choice).

One numba walk of an already-optimal path over all metric layers, the DEM
and the category band. Produces honest physical quantities per criterion
(EUR, meters, grade exposure, per-class lengths) plus the achieved
feasibility — the value the search actually minimized — recomputed
quantization-free from the float layers through the SAME slope-response
LUTs the kernels used (no formula drift).

Geometric conventions match edge construction exactly: segments decompose
into {source, intermediates, target} cells (same Bresenham offsets), the
per-meter value of a segment is the mean over those cells, and lengths use
the endpoint-chord 3D basis whenever a DEM is present.

This module is intentionally NEW — never extend ``_traversal_numba.py``,
which is shadowed by a stale compiled ``_traversal`` extension.
"""
from dataclasses import dataclass, field

import numba as nb
import numpy as np

from pyorps.utils.metric_edges import _intermediate_offsets

__all__ = ["evaluate_path_metrics", "PathEvaluation"]


@nb.njit(cache=True)
def _walk_path(path_rows, path_cols, layers, dem, use_dem, category,
               use_category, cell_size, max_category):
    """Single pass over the path segments.

    Returns (metric_totals[K], length_2d, length_3d, grad_exposure,
    grad_max_pct, cat_lengths).
    """
    n_layers = layers.shape[0]
    n_segments = path_rows.shape[0] - 1

    totals = np.zeros(n_layers, dtype=np.float64)
    cat_lengths = np.zeros(max_category + 1, dtype=np.float64)
    length_2d = 0.0
    length_3d = 0.0
    grad_exposure = 0.0
    grad_max = 0.0

    for i in range(n_segments):
        r0, c0 = path_rows[i], path_cols[i]
        r1, c1 = path_rows[i + 1], path_cols[i + 1]
        dr = int(r1 - r0)
        dc = int(c1 - c0)
        if dr == 0 and dc == 0:
            continue

        offsets = _intermediate_offsets(dr, dc)
        n_cells = 2 + offsets.shape[0]

        seg_2d_m = np.sqrt(float(dr * dr + dc * dc)) * cell_size
        if use_dem:
            dh = abs(float(dem[r1, c1]) - float(dem[r0, c0]))
            seg_3d_m = np.sqrt(seg_2d_m * seg_2d_m + dh * dh)
            slope_pct = dh / seg_2d_m * 100.0
        else:
            seg_3d_m = seg_2d_m
            slope_pct = 0.0

        length_2d += seg_2d_m
        length_3d += seg_3d_m
        grad_exposure += slope_pct * seg_3d_m
        if slope_pct > grad_max:
            grad_max = slope_pct

        # Per-layer means over {source, intermediates, target}
        for k in range(n_layers):
            value_sum = layers[k, r0, c0] + layers[k, r1, c1]
            for j in range(offsets.shape[0]):
                value_sum += layers[k, r0 + offsets[j, 0],
                                    c0 + offsets[j, 1]]
            totals[k] += value_sum / n_cells * seg_3d_m

        # Category attribution (per cell share of the segment length),
        # mirroring the legacy per-cell length attribution.
        if use_category:
            share = seg_3d_m / n_cells
            cat_lengths[category[r0, c0]] += share
            cat_lengths[category[r1, c1]] += share
            for j in range(offsets.shape[0]):
                cat_lengths[category[r0 + offsets[j, 0],
                                     c0 + offsets[j, 1]]] += share

    return (totals, length_2d, length_3d, grad_exposure, grad_max,
            cat_lengths)


@nb.njit(cache=True)
def _walk_feasibility(path_rows, path_cols, weighted_surface, dem, use_dem,
                      cell_size, mult_lut, add_lut, bin_inv, n_bins,
                      use_luts):
    """Achieved objective: mean weighted-surface value × 2D length × Γ_mult
    + Γ_add × 2D length, per segment — the kernel formula in user units."""
    n_segments = path_rows.shape[0] - 1
    feasibility = 0.0
    for i in range(n_segments):
        r0, c0 = path_rows[i], path_cols[i]
        r1, c1 = path_rows[i + 1], path_cols[i + 1]
        dr = int(r1 - r0)
        dc = int(c1 - c0)
        if dr == 0 and dc == 0:
            continue
        offsets = _intermediate_offsets(dr, dc)
        n_cells = 2 + offsets.shape[0]
        value_sum = weighted_surface[r0, c0] + weighted_surface[r1, c1]
        for j in range(offsets.shape[0]):
            value_sum += weighted_surface[r0 + offsets[j, 0],
                                          c0 + offsets[j, 1]]
        mean_value = value_sum / n_cells
        seg_2d_m = np.sqrt(float(dr * dr + dc * dc)) * cell_size

        mult = 1.0
        add = 0.0
        if use_luts:
            if use_dem:
                dh = abs(float(dem[r1, c1]) - float(dem[r0, c0]))
                slope_pct = dh / seg_2d_m * 100.0
            else:
                slope_pct = 0.0
            b = int(slope_pct * bin_inv)
            if b >= n_bins:
                b = n_bins - 1
            mult = float(mult_lut[b])
            add = float(add_lut[b])
        feasibility += mean_value * seg_2d_m * mult + add * seg_2d_m
    return feasibility


@dataclass
class PathEvaluation:
    """Honest per-criterion accounting of one already-optimal path."""

    metrics: dict = field(default_factory=dict)   # name -> physical total
    feasibility: float = 0.0                       # achieved objective
    total_length_2d: float = 0.0                   # m
    total_length_3d: float = 0.0                   # m (== 2d without DEM)
    mean_gradient_pct: float = 0.0                 # exposure / 3d length
    max_gradient_pct: float = 0.0                  # steepest step on route
    length_by_class: dict = field(default_factory=dict)  # label -> m


def evaluate_path_metrics(
        path_rows: np.ndarray,
        path_cols: np.ndarray,
        layers: dict,
        objective,
        cell_size: float,
        dem: np.ndarray | None = None,
        category: np.ndarray | None = None,
        category_labels: dict | None = None,
        eval_luts=None,
) -> PathEvaluation:
    """Evaluate a path against every metric layer (reporting only).

    Parameters:
        path_rows / path_cols: Path cell coordinates on the layer grid.
        layers: Mapping name -> float32 2D array (all same shape).
        objective: The :class:`pyorps.core.objective.Objective` that was
            minimized (weights + gradient responses).
        cell_size: Cell size in meters.
        dem: Optional float32 DEM on the same grid (3D basis + gradient).
        category: Optional uint16 feature-class band.
        category_labels: Optional id -> label mapping for the breakdown.
        eval_luts: Optional GradientLUTs built with quant_scale=1 (user
            units). Built on the fly from the objective when omitted and a
            DEM is present.

    Returns:
        :class:`PathEvaluation`. ``metrics`` carries the physical exposure
        of EVERY layer (response-free line integrals over the 3D length),
        plus ``length`` and — with a DEM — ``gradient`` (%·m exposure).
        ``feasibility`` is the achieved search objective in user units:
        with responses configured it intentionally differs from
        ``sum(w_k * metrics[k])`` (the responses shape the search, the
        metrics stay honest).
    """
    path_rows = np.ascontiguousarray(path_rows, dtype=np.int64)
    path_cols = np.ascontiguousarray(path_cols, dtype=np.int64)
    if path_rows.shape[0] != path_cols.shape[0]:
        raise ValueError("path_rows and path_cols must have equal length")

    names = list(layers)
    if not names:
        raise ValueError("evaluate_path_metrics needs at least one layer")
    shape = layers[names[0]].shape
    stack = np.stack([np.ascontiguousarray(layers[n], dtype=np.float32)
                      for n in names])

    use_dem = dem is not None
    dem_arr = (np.ascontiguousarray(dem, dtype=np.float32) if use_dem
               else np.zeros((1, 1), dtype=np.float32))

    use_category = category is not None
    if use_category:
        category_arr = np.ascontiguousarray(category, dtype=np.int64)
        max_category = int(category_arr.max()) if category_arr.size else 0
    else:
        category_arr = np.zeros((1, 1), dtype=np.int64)
        max_category = 0

    # LUTs in user units (quant_scale=1) — the same discretization the
    # search used, so feasibility has no formula drift.
    use_luts = use_dem
    if use_luts and eval_luts is None:
        unit_steps = np.array([[1, 0]], dtype=np.int8)
        eval_luts = objective.build_gradient_luts(
            unit_steps, cell_size=cell_size, quant_scale=1.0)
    if use_luts:
        mult_lut = np.ascontiguousarray(eval_luts.mult, dtype=np.float32)
        add_lut = np.ascontiguousarray(eval_luts.add, dtype=np.float32)
        bin_inv = float(eval_luts.bin_inv)
        n_bins = int(eval_luts.n_bins)
    else:
        mult_lut = np.ones(1, dtype=np.float32)
        add_lut = np.zeros(1, dtype=np.float32)
        bin_inv = 1.0
        n_bins = 1

    (totals, length_2d, length_3d, grad_exposure, grad_max,
     cat_lengths) = _walk_path(
        path_rows, path_cols, stack, dem_arr, use_dem,
        category_arr, use_category, float(cell_size), max_category)

    # Achieved objective: weighted float surface through the LUTs.
    weights = objective.weights
    weighted = np.zeros(shape, dtype=np.float32)
    for name, weight in weights.items():
        if weight > 0 and name in layers:
            weighted += np.float32(weight) * layers[name]
    w_length = weights.get("length", 0.0)
    if w_length > 0:
        weighted += np.float32(w_length)
    feasibility = float(_walk_feasibility(
        path_rows, path_cols, weighted, dem_arr, use_dem,
        float(cell_size), mult_lut, add_lut, bin_inv, n_bins, use_luts))

    metrics = {name: float(t) for name, t in zip(names, totals)}
    metrics["length"] = float(length_3d)
    if use_dem:
        metrics["gradient"] = float(grad_exposure)

    length_by_class = {}
    if use_category:
        labels = category_labels or {}
        for cat_id in range(1, max_category + 1):
            meters = float(cat_lengths[cat_id])
            if meters > 0:
                label = labels.get(cat_id, f"class {cat_id}")
                length_by_class[label] = (
                    length_by_class.get(label, 0.0) + meters)

    return PathEvaluation(
        metrics=metrics,
        feasibility=feasibility,
        total_length_2d=float(length_2d),
        total_length_3d=float(length_3d),
        mean_gradient_pct=(float(grad_exposure) / float(length_3d)
                          if length_3d > 0 else 0.0),
        max_gradient_pct=float(grad_max),
        length_by_class=length_by_class,
    )

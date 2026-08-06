"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Gradient-aware edge construction for the graph-library backends.

Builds float64 edge lists from the combined feasibility weight raster, the
DEM and the slope-response LUT pair (feasibility plan section 3.2)::

    s      = |DEM[v] - DEM[u]| / horiz_m(d)                (chord slope)
    b      = min(int(|Δh| * bin_factor[d]), n_bins - 1)
    w(u→v) = W_sum * cost_factor(d) * Γ_mult[b] + Γ_add[b] * step_len(d)

Γ_mult carries the unconditional 3D length stretch (and any multiplicative
response); Γ_mult[b] == inf marks the edge forbidden (hard grade limit).
There is deliberately NO 65535 clamp on the weights (that was a uint16-ism
of the legacy 3D path) and the slope is meter-correct at any resolution.

This module is intentionally NEW — never extend ``_traversal_numba.py``,
which is shadowed by a stale compiled ``_traversal`` extension.
"""
import numba as nb
import numpy as np
from numba import prange

__all__ = ["construct_edges_gradient", "construct_edges_weights"]


@nb.njit(cache=True)
def _intermediate_offsets(dr, dc):
    """Bresenham-like intermediate cell offsets for a (dr, dc) step.

    Mirrors ``_traversal_numba.intermediate_steps_numba`` /
    ``traversal_gpu._intermediate_steps_cpu``.
    """
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    if abs_dr + abs_dc <= 1:
        return np.zeros((0, 2), dtype=np.int8)

    k = max(abs_dr, abs_dc)
    if k == 1:
        result = np.zeros((2, 2), dtype=np.int8)
        result[0, 0] = dr
        result[1, 1] = dc
        return result

    result = np.zeros((2 * (k - 1), 2), dtype=np.int8)
    for i in range(k - 1):
        dr_k = (i + 1) * dr / k
        dc_k = (i + 1) * dc / k
        idx = i * 2
        result[idx, 0] = np.int8(np.floor(dr_k))
        result[idx, 1] = np.int8(np.floor(dc_k))
        result[idx + 1, 0] = np.int8(np.ceil(dr_k))
        result[idx + 1, 1] = np.int8(np.ceil(dc_k))
    return result


@nb.njit(cache=True, parallel=True, fastmath=True)
def _edges_for_direction(weights, dem, exclude_mask, dr, dc, intermediates,
                         cost_factor, step_len, bin_factor,
                         mult_lut, add_lut, n_bins,
                         from_nodes, to_nodes, edge_costs, valid_flags):
    """Fill per-source-cell edge data for one step direction (parallel)."""
    rows, cols = weights.shape
    r_start = max(0, -dr)
    r_end = min(rows, rows - dr)
    c_start = max(0, -dc)
    c_end = min(cols, cols - dc)
    width = c_end - c_start

    for sr in prange(r_start, r_end):
        for sc in range(c_start, c_end):
            slot = (sr - r_start) * width + (sc - c_start)
            tr = sr + dr
            tc = sc + dc

            if exclude_mask[sr, sc] == 0 or exclude_mask[tr, tc] == 0:
                continue

            weight_sum = 0.0
            blocked = False
            for i in range(intermediates.shape[0]):
                ir = sr + intermediates[i, 0]
                ic = sc + intermediates[i, 1]
                if ir < 0 or ir >= rows or ic < 0 or ic >= cols or \
                        exclude_mask[ir, ic] == 0:
                    blocked = True
                    break
                weight_sum += weights[ir, ic]
            if blocked:
                continue

            weight_sum += weights[sr, sc] + weights[tr, tc]
            edge_cost = weight_sum * cost_factor

            height_diff = abs(dem[tr, tc] - dem[sr, sc])
            slope_bin = int(height_diff * bin_factor)
            if slope_bin >= n_bins:
                slope_bin = n_bins - 1
            mult = mult_lut[slope_bin]
            if not np.isfinite(mult):
                continue  # hard grade limit: edge forbidden
            edge_cost = edge_cost * mult + add_lut[slope_bin] * step_len

            from_nodes[slot] = sr * cols + sc
            to_nodes[slot] = tr * cols + tc
            edge_costs[slot] = edge_cost
            valid_flags[slot] = 1


class _PlainLUTs:
    """Trivial slope-response tables: mult ≡ 1, add ≡ 0 (no gradient)."""

    def __init__(self, steps):
        lengths = np.sqrt((steps[:, 0].astype(np.float64)) ** 2
                          + (steps[:, 1].astype(np.float64)) ** 2)
        self.mult = np.ones(1, dtype=np.float32)
        self.add = np.zeros(1, dtype=np.float32)
        self.bin_factor = np.zeros(len(steps), dtype=np.float32)
        self.step_len_cells = lengths.astype(np.float32)
        self.n_bins = 1


def construct_edges_weights(weights, steps, ignore_max=True):
    """Plain (gradient-free) edge construction from a weight raster.

    The float-precision twin of the legacy ``construct_edges``: accepts
    float32/float64 weights (forbidden = non-finite) as well as uint16
    (forbidden = 65535). Used by the library backends in
    ``weight_precision="float32"`` mode without a DEM.
    """
    steps = np.ascontiguousarray(steps, dtype=np.int8)
    dem = np.zeros(weights.shape, dtype=np.float32)
    return construct_edges_gradient(weights, dem, steps,
                                    _PlainLUTs(steps), ignore_max)


def construct_edges_gradient(weights, dem, steps, gradient_luts,
                             ignore_max=True):
    """Construct gradient-aware graph edges for the library backends.

    Parameters:
        weights: Combined feasibility raster — uint16 (65535 = forbidden)
            or float32/float64 (non-finite = forbidden, lossless mode).
        dem: float32 DEM aligned to ``weights`` (same shape, finite).
        steps: (n_dirs, 2) int8 neighborhood steps.
        gradient_luts: :class:`pyorps.core.objective.GradientLUTs`.
        ignore_max: Treat sentinel cells as forbidden (default True).

    Returns:
        (from_nodes uint32, to_nodes uint32, costs float64) edge arrays.
    """
    if np.issubdtype(np.asarray(weights).dtype, np.floating):
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        exclude_mask = np.isfinite(weights).astype(np.uint8)
    else:
        weights = np.ascontiguousarray(weights, dtype=np.uint16)
        if ignore_max:
            exclude_mask = (weights != np.iinfo(np.uint16).max) \
                .astype(np.uint8)
        else:
            exclude_mask = np.ones_like(weights, dtype=np.uint8)
    dem = np.ascontiguousarray(dem, dtype=np.float32)
    if dem.shape != weights.shape:
        raise ValueError(
            f"DEM shape {dem.shape} does not match weights "
            f"{weights.shape} — align the DEM first.")
    steps = np.ascontiguousarray(steps, dtype=np.int8)

    mult_lut = np.ascontiguousarray(gradient_luts.mult, dtype=np.float32)
    add_lut = np.ascontiguousarray(gradient_luts.add, dtype=np.float32)
    bin_factors = np.asarray(gradient_luts.bin_factor, dtype=np.float64)
    step_lens = np.asarray(gradient_luts.step_len_cells, dtype=np.float64)
    n_bins = int(gradient_luts.n_bins)

    all_from, all_to, all_costs = [], [], []
    for d in range(steps.shape[0]):
        dr = int(steps[d, 0])
        dc = int(steps[d, 1])
        intermediates = _intermediate_offsets(dr, dc)
        cost_factor = float(step_lens[d] / (2.0 + intermediates.shape[0]))

        rows, cols = weights.shape
        height = min(rows, rows - dr) - max(0, -dr)
        width = min(cols, cols - dc) - max(0, -dc)
        if height <= 0 or width <= 0:
            continue
        n_slots = height * width
        from_nodes = np.zeros(n_slots, dtype=np.uint32)
        to_nodes = np.zeros(n_slots, dtype=np.uint32)
        edge_costs = np.zeros(n_slots, dtype=np.float64)
        valid_flags = np.zeros(n_slots, dtype=np.uint8)

        _edges_for_direction(
            weights, dem, exclude_mask, dr, dc, intermediates,
            cost_factor, float(step_lens[d]), float(bin_factors[d]),
            mult_lut, add_lut, n_bins,
            from_nodes, to_nodes, edge_costs, valid_flags,
        )

        keep = valid_flags == 1
        all_from.append(from_nodes[keep])
        all_to.append(to_nodes[keep])
        all_costs.append(edge_costs[keep])

    if not all_from:
        return (np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32),
                np.zeros(0, dtype=np.float64))
    return (np.concatenate(all_from), np.concatenate(all_to),
            np.concatenate(all_costs))

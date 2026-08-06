# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Shared constrained pathfinding infrastructure: structs, state encoding,
and precomputation used by both constrained Dijkstra and delta-stepping.

Extracted from constrained_path_algorithms.pyx.
Contains:
- ValidNeighbor, StateData, CRelaxBuf, LazyState structs
- State packing/unpacking (with and without height class)
- Intermediate path checking (raw pointer version)
- Neighbor precomputation (_build_valid_neighbors, _flatten_valid_neighbors)
- Cache precomputation (_precompute_intermediate_cache, _precompute_gradient_cache)
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t, UINT64_MAX
from libc.math cimport INFINITY, fabsf, sqrtf, expf
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libcpp.vector cimport vector
from cython.parallel cimport prange, threadid

from pyorps.utils._heap cimport (
    StepData, CachedStepData, IntermediatePoint, npy_intp,
)
from pyorps.utils._raster_context cimport (
    check_path_cached, precompute_directions, precompute_cached_steps,
)


# ==================== CONSTANTS ====================

cdef uint8_t FLAG_TOUCHED = 1
cdef uint8_t FLAG_VISITED = 2


# ==================== INTERMEDIATE CHECK (RAW POINTER) ====================

cdef inline int _check_intermediates_ptr(
    const vector[IntermediatePoint]& intermediates,
    int current_row, int current_col,
    const uint8_t* mask_ptr, const uint16_t* raster_ptr,
    int rows, int cols, float* total_cost,
) noexcept nogil:
    """C-pointer version of check_path_cached — no memoryview overhead."""
    cdef float cost = 0.0
    cdef int i, int_row, int_col, cell_idx
    cdef int num = <int>intermediates.size()

    for i in range(num):
        int_row = current_row + intermediates[i].dr
        int_col = current_col + intermediates[i].dc

        if <unsigned int>int_row >= <unsigned int>rows or <unsigned int>int_col >= <unsigned int>cols:
            return 0

        cell_idx = int_row * cols + int_col

        if mask_ptr[cell_idx] == 0:
            return 0

        cost += <float>raster_ptr[cell_idx]

    total_cost[0] = cost
    return 1


# ==================== TOWER AREA COST ====================

cdef inline double _area_terrain_cost(
    int row, int col, int rows, int cols,
    const uint16_t* raster_ptr,
    const float* tower_terrain_ptr,
    const int32_t* offsets,
    int n_offsets,
) noexcept nogil:
    """Sum terrain costs over rotated square pixel offsets.

    offsets is a flat array of (dr, dc) pairs: [dr0, dc0, dr1, dc1, ...].
    Each offset is relative to (row, col). Out-of-bounds pixels are skipped.
    Returns INFINITY if any pixel in the footprint is forbidden (65535).
    """
    cdef double total = 0.0
    cdef int i, nr, nc
    cdef uint16_t val
    for i in range(n_offsets):
        nr = row + offsets[2 * i]
        nc = col + offsets[2 * i + 1]
        if <unsigned int>nr < <unsigned int>rows and <unsigned int>nc < <unsigned int>cols:
            val = raster_ptr[nr * cols + nc]
            if val == 65535:
                return INFINITY
            total += <double>tower_terrain_ptr[val]
    return total


cdef inline double _avg_slope_pct(
    int row, int col, int rows, int cols,
    const float* dem_ptr, float cell_size,
    const int32_t* offsets, int n_offsets,
) noexcept nogil:
    """Compute average slope (%) over the tower footprint pixels.

    For each pixel in the footprint, slope is the max absolute elevation
    difference to its 4-connected neighbors, divided by cell_size.
    """
    cdef double slope_sum = 0.0
    cdef int count = 0
    cdef int i, nr, nc, idx
    cdef float elev, max_diff, diff
    cdef int dr, dc
    for i in range(n_offsets):
        nr = row + offsets[2 * i]
        nc = col + offsets[2 * i + 1]
        if <unsigned int>nr >= <unsigned int>rows or <unsigned int>nc >= <unsigned int>cols:
            continue
        idx = nr * cols + nc
        elev = dem_ptr[idx]
        max_diff = 0.0
        # 4-connected neighbors
        if nr > 0:
            diff = fabsf(dem_ptr[idx - cols] - elev)
            if diff > max_diff:
                max_diff = diff
        if nr < rows - 1:
            diff = fabsf(dem_ptr[idx + cols] - elev)
            if diff > max_diff:
                max_diff = diff
        if nc > 0:
            diff = fabsf(dem_ptr[idx - 1] - elev)
            if diff > max_diff:
                max_diff = diff
        if nc < cols - 1:
            diff = fabsf(dem_ptr[idx + 1] - elev)
            if diff > max_diff:
                max_diff = diff
        slope_sum += <double>(max_diff / cell_size * 100.0)
        count += 1
    if count == 0:
        return 0.0
    return slope_sum / <double>count


cdef inline double _tower_terrain(
    int use_area, int row, int col,
    int d_in, int d_out, int n_dirs,
    int rows, int cols,
    const uint16_t* raster_ptr, const float* tower_terrain_ptr,
    double fallback,
    const int32_t* offsets, const int32_t* starts, const int32_t* counts,
    const float* dem_ptr, float cell_size, float gradient_scale,
) noexcept nogil:
    """Return tower terrain cost with slope-dependent foundation multiplier.

    When exact mode: sum pixel costs over rotated footprint, then multiply
    by exp(gradient_scale * avg_slope_pct / 100).
    When uniform mode: use fallback cost, still apply slope multiplier if
    DEM is available.
    """
    cdef double base_cost
    cdef int pair_idx
    cdef double avg_slope, slope_mult
    cdef float elev, max_d, d_val
    cdef int idx

    if use_area == 0:
        base_cost = fallback
    else:
        pair_idx = d_in * n_dirs + d_out
        base_cost = _area_terrain_cost(
            row, col, rows, cols, raster_ptr, tower_terrain_ptr,
            &offsets[starts[pair_idx] * 2], counts[pair_idx])

    # Apply slope-dependent foundation cost multiplier
    if dem_ptr != NULL and base_cost < INFINITY:
        if use_area != 0:
            pair_idx = d_in * n_dirs + d_out
            avg_slope = _avg_slope_pct(
                row, col, rows, cols, dem_ptr, cell_size,
                &offsets[starts[pair_idx] * 2], counts[pair_idx])
        else:
            # Uniform mode: single-pixel slope at tower center
            avg_slope = 0.0
            if <unsigned int>row < <unsigned int>rows and <unsigned int>col < <unsigned int>cols:
                idx = row * cols + col
                elev = dem_ptr[idx]
                max_d = 0.0
                if row > 0:
                    d_val = fabsf(dem_ptr[idx - cols] - elev)
                    if d_val > max_d: max_d = d_val
                if row < rows - 1:
                    d_val = fabsf(dem_ptr[idx + cols] - elev)
                    if d_val > max_d: max_d = d_val
                if col > 0:
                    d_val = fabsf(dem_ptr[idx - 1] - elev)
                    if d_val > max_d: max_d = d_val
                if col < cols - 1:
                    d_val = fabsf(dem_ptr[idx + 1] - elev)
                    if d_val > max_d: max_d = d_val
                avg_slope = <double>(max_d / cell_size * 100.0)
        if avg_slope > 0.0:
            slope_mult = expf(<float>(gradient_scale * avg_slope / 100.0))
            base_cost = base_cost * slope_mult

    return base_cost


cdef inline double _area_terrain_cost_cell(
    int row, int col, int rows, int cols,
    const float* tower_cost_ptr,
    const int32_t* offsets,
    int n_offsets,
) noexcept nogil:
    """Per-cell variant of _area_terrain_cost (feasibility plan Phase 8).

    Sums a precomputed per-cell tower-cost raster over the footprint
    instead of LUT[raster[px]] — the tower cost then depends on the land
    use (cost layer), not on combined feasibility values. Forbidden cells
    carry INFINITY in the raster.
    """
    cdef double total = 0.0
    cdef int i, nr, nc
    cdef float val
    for i in range(n_offsets):
        nr = row + offsets[2 * i]
        nc = col + offsets[2 * i + 1]
        if <unsigned int>nr < <unsigned int>rows and <unsigned int>nc < <unsigned int>cols:
            val = tower_cost_ptr[nr * cols + nc]
            if val == INFINITY:
                return INFINITY
            total += <double>val
    return total


cdef inline double _tower_terrain_cell(
    int use_area, int row, int col,
    int d_in, int d_out, int n_dirs,
    int rows, int cols,
    const float* tower_cost_ptr,
    double fallback,
    const int32_t* offsets, const int32_t* starts, const int32_t* counts,
    const float* dem_ptr, float cell_size, float gradient_scale,
) noexcept nogil:
    """Per-cell-raster twin of _tower_terrain (identical slope handling)."""
    cdef double base_cost
    cdef int pair_idx
    cdef double avg_slope, slope_mult
    cdef float elev, max_d, d_val
    cdef int idx

    if use_area == 0:
        base_cost = fallback
    else:
        pair_idx = d_in * n_dirs + d_out
        base_cost = _area_terrain_cost_cell(
            row, col, rows, cols, tower_cost_ptr,
            &offsets[starts[pair_idx] * 2], counts[pair_idx])

    # Apply slope-dependent foundation cost multiplier
    if dem_ptr != NULL and base_cost < INFINITY:
        if use_area != 0:
            pair_idx = d_in * n_dirs + d_out
            avg_slope = _avg_slope_pct(
                row, col, rows, cols, dem_ptr, cell_size,
                &offsets[starts[pair_idx] * 2], counts[pair_idx])
        else:
            # Uniform mode: single-pixel slope at tower center
            avg_slope = 0.0
            if <unsigned int>row < <unsigned int>rows and <unsigned int>col < <unsigned int>cols:
                idx = row * cols + col
                elev = dem_ptr[idx]
                max_d = 0.0
                if row > 0:
                    d_val = fabsf(dem_ptr[idx - cols] - elev)
                    if d_val > max_d: max_d = d_val
                if row < rows - 1:
                    d_val = fabsf(dem_ptr[idx + cols] - elev)
                    if d_val > max_d: max_d = d_val
                if col > 0:
                    d_val = fabsf(dem_ptr[idx - 1] - elev)
                    if d_val > max_d: max_d = d_val
                if col < cols - 1:
                    d_val = fabsf(dem_ptr[idx + 1] - elev)
                    if d_val > max_d: max_d = d_val
                avg_slope = <double>(max_d / cell_size * 100.0)
        if avg_slope > 0.0:
            slope_mult = expf(<float>(gradient_scale * avg_slope / 100.0))
            base_cost = base_cost * slope_mult

    return base_cost


# ==================== STATE PACKING ====================

cdef inline uint64_t _pack_state(uint32_t cell, uint8_t direction,
                                  uint16_t span_bin, int n_dirs,
                                  int n_span_bins) noexcept nogil:
    return (<uint64_t>cell) * n_dirs * n_span_bins + direction * n_span_bins + span_bin


cdef inline void _unpack_state(uint64_t state, int n_dirs, int n_span_bins,
                                uint32_t* cell, uint8_t* direction,
                                uint16_t* span_bin) noexcept nogil:
    cdef uint64_t states_per_cell = n_dirs * n_span_bins
    cell[0] = <uint32_t>(state / states_per_cell)
    cdef uint32_t remainder = <uint32_t>(state % states_per_cell)
    direction[0] = <uint8_t>(remainder / n_span_bins)
    span_bin[0] = <uint16_t>(remainder % n_span_bins)


# Python-accessible wrappers for testing
def pack_state(uint32_t cell, uint8_t direction, uint16_t span_bin,
               int n_dirs, int n_span_bins):
    """Pack (cell, direction, span_bin) into a single uint64 state index."""
    return _pack_state(cell, direction, span_bin, n_dirs, n_span_bins)


def unpack_state(uint64_t state, int n_dirs, int n_span_bins):
    """Unpack a state index into (cell, direction, span_bin)."""
    cdef uint32_t cell
    cdef uint8_t direction
    cdef uint16_t span_bin
    _unpack_state(state, n_dirs, n_span_bins, &cell, &direction, &span_bin)
    return int(cell), int(direction), int(span_bin)


# ==================== VARIABLE-HEIGHT STATE PACKING ====================

cdef inline uint64_t _pack_state_h(uint32_t cell, uint8_t direction,
                                    uint16_t span_bin, uint8_t height_class,
                                    int n_dirs, int n_span_bins,
                                    int n_heights) noexcept nogil:
    """Pack (cell, direction, span_bin, height_class) into uint64 state."""
    return ((<uint64_t>cell) * n_dirs * n_span_bins * n_heights +
            <uint64_t>direction * n_span_bins * n_heights +
            <uint64_t>span_bin * n_heights +
            <uint64_t>height_class)


cdef inline void _unpack_state_h(uint64_t state, int n_dirs, int n_span_bins,
                                  int n_heights,
                                  uint32_t* cell, uint8_t* direction,
                                  uint16_t* span_bin,
                                  uint8_t* height_class) noexcept nogil:
    """Unpack uint64 state into (cell, direction, span_bin, height_class)."""
    cdef uint64_t sph = <uint64_t>n_span_bins * n_heights
    cdef uint64_t spc = <uint64_t>n_dirs * sph
    cell[0] = <uint32_t>(state / spc)
    cdef uint64_t rem1 = state % spc
    direction[0] = <uint8_t>(rem1 / sph)
    cdef uint64_t rem2 = rem1 % sph
    span_bin[0] = <uint16_t>(rem2 / n_heights)
    height_class[0] = <uint8_t>(rem2 % n_heights)


def pack_state_h(uint32_t cell, uint8_t direction, uint16_t span_bin,
                 uint8_t height_class, int n_dirs, int n_span_bins,
                 int n_heights):
    """Pack (cell, direction, span_bin, height_class) — Python wrapper."""
    return _pack_state_h(cell, direction, span_bin, height_class,
                         n_dirs, n_span_bins, n_heights)


def unpack_state_h(uint64_t state, int n_dirs, int n_span_bins,
                   int n_heights):
    """Unpack state into (cell, direction, span_bin, height_class)."""
    cdef uint32_t cell
    cdef uint8_t direction
    cdef uint16_t span_bin
    cdef uint8_t height_class
    _unpack_state_h(state, n_dirs, n_span_bins, n_heights,
                    &cell, &direction, &span_bin, &height_class)
    return int(cell), int(direction), int(span_bin), int(height_class)


# ==================== PRECOMPUTATION ====================

cdef vector[vector[ValidNeighbor]] _build_valid_neighbors(
    vector[StepData]& directions,
    float[:, :] angle_cost_view,
    uint8_t[:, :] angle_valid_view,
    float[:] step_dist_view,
    float[:, :] tower_angle_view,
    int n_dirs,
):
    """Build per-direction lists of valid outgoing neighbors."""
    cdef vector[vector[ValidNeighbor]] result
    cdef vector[ValidNeighbor] per_dir
    cdef ValidNeighbor nb
    cdef int d_in, d_out

    result.reserve(n_dirs)
    for d_in in range(n_dirs):
        per_dir = vector[ValidNeighbor]()
        for d_out in range(n_dirs):
            if angle_valid_view[d_in, d_out] != 0:
                nb.d_out = <uint8_t>d_out
                nb.dr = directions[d_out].dr
                nb.dc = directions[d_out].dc
                nb.cost_factor = directions[d_out].cost_factor
                nb.step_distance = step_dist_view[d_out]
                nb.angle_cost = angle_cost_view[d_in, d_out]
                nb.tower_angle_cost = tower_angle_view[d_in, d_out]
                per_dir.push_back(nb)
        result.push_back(per_dir)

    return result


cdef void _flatten_valid_neighbors(
    vector[vector[ValidNeighbor]]& nested,
    int n_dirs,
    vector[ValidNeighbor]& flat_out,
    vector[int]& offsets_out,
):
    """Flatten vector-of-vectors into contiguous array + offset table."""
    cdef int d, k, total = 0
    offsets_out.resize(n_dirs + 1)
    for d in range(n_dirs):
        offsets_out[d] = total
        total += <int>nested[d].size()
    offsets_out[n_dirs] = total
    flat_out.resize(total)
    total = 0
    for d in range(n_dirs):
        for k in range(<int>nested[d].size()):
            flat_out[total] = nested[d][k]
            total += 1


cdef void _precompute_intermediate_cache(
    int total_cells, int n_dirs, int rows, int cols,
    const uint8_t* mask_ptr, const uint16_t* raster_ptr,
    vector[StepData]& directions,
    vector[CachedStepData]& cached_steps,
    uint8_t* icache_status, float* icache_cost,
) noexcept:
    """Precompute all (cell, d_out) intermediate validity in parallel.

    For each destination cell and each incoming direction d_out, compute
    the intermediate path validity and cost. The source cell for each pair
    is: src = dest - step[d_out].

    Status encoding: 0=not computed (should not remain), 1=invalid, 2=valid.
    """
    cdef int pc_cell, pc_d
    cdef int pc_src_row, pc_src_col
    cdef size_t pc_idx
    cdef float pc_cost
    cdef int pc_valid

    for pc_cell in prange(total_cells, nogil=True, schedule='static'):
        if mask_ptr[pc_cell] == 0:
            # Destination cell is impassable — mark all directions invalid.
            # These entries are never accessed (mask check precedes cache check),
            # but marking them avoids leaving status=0.
            for pc_d in range(n_dirs):
                icache_status[<size_t>pc_cell * <size_t>n_dirs + <size_t>pc_d] = 1
            continue

        for pc_d in range(n_dirs):
            pc_idx = <size_t>pc_cell * <size_t>n_dirs + <size_t>pc_d

            if cached_steps[pc_d].intermediate_count == 0:
                # Cardinal step: no intermediates, always valid, cost=0
                icache_status[pc_idx] = 2
                # icache_cost[pc_idx] already 0.0 from calloc
                continue

            # Compute source cell = dest_cell - step[d_out]
            pc_src_row = pc_cell / cols - directions[pc_d].dr
            pc_src_col = pc_cell % cols - directions[pc_d].dc

            if (<unsigned int>pc_src_row >= <unsigned int>rows or
                    <unsigned int>pc_src_col >= <unsigned int>cols):
                icache_status[pc_idx] = 1
            elif mask_ptr[pc_src_row * cols + pc_src_col] == 0:
                icache_status[pc_idx] = 1
            else:
                pc_cost = 0.0
                pc_valid = _check_intermediates_ptr(
                    cached_steps[pc_d].intermediates,
                    pc_src_row, pc_src_col,
                    mask_ptr, raster_ptr, rows, cols,
                    &pc_cost)
                if pc_valid == 0:
                    icache_status[pc_idx] = 1
                else:
                    icache_status[pc_idx] = 2
                    icache_cost[pc_idx] = pc_cost


cdef void _precompute_gradient_cache(
    int total_cells, int n_dirs, int rows, int cols,
    const float* dem_ptr, float cell_size,
    float max_gradient_pct, float gradient_scale,
    vector[StepData]& directions,
    uint8_t* icache_status,
    float* grad_penalty,
) noexcept:
    """Precompute gradient penalties for all (cell, d_out) pairs in parallel.

    Must be called AFTER _precompute_intermediate_cache.
    Only processes entries where icache_status == 2 (valid intermediate path).
    Invalidates entries where gradient exceeds max_gradient_pct.
    """
    cdef int gc, gd
    cdef int g_src_row, g_src_col, g_src_cell
    cdef size_t g_idx
    cdef float g_height_diff, g_horiz_dist, g_slope, g_grad_pct

    for gc in prange(total_cells, nogil=True, schedule='static'):
        for gd in range(n_dirs):
            g_idx = <size_t>gc * <size_t>n_dirs + <size_t>gd

            if icache_status[g_idx] != 2:
                grad_penalty[g_idx] = 1.0
                continue

            # Source cell = dest_cell - step[d_out]
            g_src_row = gc / cols - directions[gd].dr
            g_src_col = gc % cols - directions[gd].dc

            if (<unsigned int>g_src_row >= <unsigned int>rows or
                    <unsigned int>g_src_col >= <unsigned int>cols):
                grad_penalty[g_idx] = 1.0
                continue

            g_src_cell = g_src_row * cols + g_src_col
            g_height_diff = fabsf(dem_ptr[gc] - dem_ptr[g_src_cell])
            g_horiz_dist = sqrtf(<float>(directions[gd].dr * directions[gd].dr +
                                         directions[gd].dc * directions[gd].dc)) * cell_size

            if g_horiz_dist < 1e-6:
                grad_penalty[g_idx] = 1.0
                continue

            g_slope = g_height_diff / g_horiz_dist
            g_grad_pct = g_slope * 100.0

            if g_grad_pct > max_gradient_pct:
                icache_status[g_idx] = 1  # Too steep — impassable
                grad_penalty[g_idx] = 0.0
            else:
                # Exponential penalty: exp(slope^2 * scale)
                grad_penalty[g_idx] = expf(g_slope * g_slope * gradient_scale)


# ==================== CLEARANCE HELPERS ====================

cdef inline int _check_span_clearance(
    int cur_row, int cur_col,
    float span_dist_m,
    uint8_t direction,
    float tower_height, float cond_w, float cond_t, float min_clearance,
    const float* dem_ptr, const float* obstacle_ptr,
    int rows, int cols,
    vector[StepData]& directions,
    vector[CachedStepData]& cached_steps,
    float step_distance,
) noexcept nogil:
    """Check conductor clearance along span ending at (cur_row, cur_col).

    Walks BACKWARD to previous tower, then forward checking catenary sag
    at every raster cell. Parabolic approximation: sag(x) = w*x*(L-x)/(2*T).
    Returns 1 if clearance OK everywhere, 0 if violated.
    """
    cdef int n_steps = <int>(span_dist_m / step_distance + 0.5)
    if n_steps <= 1:
        return 1

    cdef int ta_row = cur_row - n_steps * directions[direction].dr
    cdef int ta_col = cur_col - n_steps * directions[direction].dc

    if (<unsigned int>ta_row >= <unsigned int>rows or
            <unsigned int>ta_col >= <unsigned int>cols):
        return 0

    cdef float span_len = <float>n_steps * step_distance
    cdef float dem_a = dem_ptr[ta_row * cols + ta_col]
    cdef float dem_b = dem_ptr[cur_row * cols + cur_col]
    cdef float attach_a = dem_a + tower_height
    cdef float attach_b = dem_b + tower_height

    # Quick check: if max sag is small relative to tower height, skip full walk
    cdef float max_sag = (cond_w * span_len * span_len) / (8.0 * cond_t)
    if tower_height - max_sag - min_clearance > 50.0:
        return 1

    cdef int walk_row = ta_row
    cdef int walk_col = ta_col
    cdef float x, chord_z, sag_x, cond_z, ground_z, obs_z, clr
    cdef int step, i, int_row, int_col, n_inter

    for step in range(n_steps):
        n_inter = cached_steps[direction].intermediate_count
        for i in range(n_inter):
            int_row = walk_row + cached_steps[direction].intermediates[i].dr
            int_col = walk_col + cached_steps[direction].intermediates[i].dc
            if (<unsigned int>int_row >= <unsigned int>rows or
                    <unsigned int>int_col >= <unsigned int>cols):
                return 0
            x = <float>step * step_distance + step_distance * (<float>(i + 1) / <float>(n_inter + 1))
            chord_z = attach_a + (attach_b - attach_a) * x / span_len
            sag_x = (cond_w * x * (span_len - x)) / (2.0 * cond_t)
            cond_z = chord_z - sag_x
            ground_z = dem_ptr[int_row * cols + int_col]
            obs_z = 0.0
            if obstacle_ptr != NULL:
                obs_z = obstacle_ptr[int_row * cols + int_col]
            clr = cond_z - ground_z - obs_z
            if clr < min_clearance:
                return 0

        walk_row += directions[direction].dr
        walk_col += directions[direction].dc

        if step > 0:
            x = <float>(step + 1) * step_distance
            if (<unsigned int>walk_row >= <unsigned int>rows or
                    <unsigned int>walk_col >= <unsigned int>cols):
                return 0
            chord_z = attach_a + (attach_b - attach_a) * x / span_len
            sag_x = (cond_w * x * (span_len - x)) / (2.0 * cond_t)
            cond_z = chord_z - sag_x
            ground_z = dem_ptr[walk_row * cols + walk_col]
            obs_z = 0.0
            if obstacle_ptr != NULL:
                obs_z = obstacle_ptr[walk_row * cols + walk_col]
            clr = cond_z - ground_z - obs_z
            if clr < min_clearance:
                return 0

    return 1


cdef inline int _check_span_clearance_vh(
    int cur_row, int cur_col,
    float span_dist_m,
    uint8_t direction,
    float height_a, float height_b,
    float cond_w, float cond_t, float min_clearance,
    const float* dem_ptr, const float* obstacle_ptr,
    int rows, int cols,
    vector[StepData]& directions,
    vector[CachedStepData]& cached_steps,
    float step_distance,
) noexcept nogil:
    """Check conductor clearance with variable tower heights (h_A, h_B).

    Tower A is the previous tower (at span start), tower B is at (cur_row, cur_col).
    Walks backward from B to locate A, then checks catenary sag at every cell.
    Returns 1 if clearance OK everywhere, 0 if violated.
    """
    cdef int n_steps = <int>(span_dist_m / step_distance + 0.5)
    if n_steps <= 1:
        return 1

    cdef int ta_row = cur_row - n_steps * directions[direction].dr
    cdef int ta_col = cur_col - n_steps * directions[direction].dc

    if (<unsigned int>ta_row >= <unsigned int>rows or
            <unsigned int>ta_col >= <unsigned int>cols):
        return 0

    cdef float span_len = <float>n_steps * step_distance
    cdef float dem_a = dem_ptr[ta_row * cols + ta_col]
    cdef float dem_b = dem_ptr[cur_row * cols + cur_col]
    cdef float attach_a = dem_a + height_a
    cdef float attach_b = dem_b + height_b

    # Quick check: if tower height alone (ignoring terrain variation)
    # provides ample clearance margin even at max sag, skip full walk.
    cdef float max_sag = (cond_w * span_len * span_len) / (8.0 * cond_t)
    cdef float min_h = height_a if height_a < height_b else height_b
    if min_h - max_sag - min_clearance > 50.0:
        return 1

    cdef int walk_row = ta_row
    cdef int walk_col = ta_col
    cdef float x, chord_z, sag_x, cond_z, ground_z, obs_z, clr
    cdef int step, i, int_row, int_col, n_inter

    for step in range(n_steps):
        n_inter = cached_steps[direction].intermediate_count
        for i in range(n_inter):
            int_row = walk_row + cached_steps[direction].intermediates[i].dr
            int_col = walk_col + cached_steps[direction].intermediates[i].dc
            if (<unsigned int>int_row >= <unsigned int>rows or
                    <unsigned int>int_col >= <unsigned int>cols):
                return 0
            x = <float>step * step_distance + step_distance * (<float>(i + 1) / <float>(n_inter + 1))
            chord_z = attach_a + (attach_b - attach_a) * x / span_len
            sag_x = (cond_w * x * (span_len - x)) / (2.0 * cond_t)
            cond_z = chord_z - sag_x
            ground_z = dem_ptr[int_row * cols + int_col]
            obs_z = 0.0
            if obstacle_ptr != NULL:
                obs_z = obstacle_ptr[int_row * cols + int_col]
            clr = cond_z - ground_z - obs_z
            if clr < min_clearance:
                return 0

        walk_row += directions[direction].dr
        walk_col += directions[direction].dc

        if step > 0:
            x = <float>(step + 1) * step_distance
            if (<unsigned int>walk_row >= <unsigned int>rows or
                    <unsigned int>walk_col >= <unsigned int>cols):
                return 0
            chord_z = attach_a + (attach_b - attach_a) * x / span_len
            sag_x = (cond_w * x * (span_len - x)) / (2.0 * cond_t)
            cond_z = chord_z - sag_x
            ground_z = dem_ptr[walk_row * cols + walk_col]
            obs_z = 0.0
            if obstacle_ptr != NULL:
                obs_z = obstacle_ptr[walk_row * cols + walk_col]
            clr = cond_z - ground_z - obs_z
            if clr < min_clearance:
                return 0

    return 1

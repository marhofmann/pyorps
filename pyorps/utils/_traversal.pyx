"""
Graph construction, node validation, and path analysis utilities.

Cython port of the Numba functions in traversal.py for near-instant import.
Reuses existing Cython building blocks from _heap and _raster_context.

Contains:
- Region bound calculations for graph construction
- Node validation and edge finding (2D and 3D)
- Graph edge construction (construct_edges, construct_edges_3d)
- Path metrics calculation
- Utility functions (segment length, euclidean distances, etc.)
"""

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sqrtf, floor, ceil, fabs, exp, atan, sin, pow, abs
from libc.stdint cimport UINT16_MAX
from libcpp.vector cimport vector
from libcpp cimport bool

from pyorps.utils._heap cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
    IntermediatePoint,
    ravel_index,
)

from pyorps.utils._raster_context cimport (
    _calculate_intermediate_steps_cython,
    _get_cost_factor_cython,
)

cdef double PI = 3.14159265358979323846
cdef uint16_t IMPASSABLE = 65535

# ==================== REGION BOUNDS ====================

cdef void _calculate_source_region_bounds(
    int dr, int rows, int dc, int cols,
    uint32_t* s_rows_start, uint32_t* s_rows_end,
    uint32_t* s_cols_start, uint32_t* s_cols_end) noexcept nogil:

    if dr > 0:
        s_rows_start[0] = 0
        s_rows_end[0] = <uint32_t>(rows - dr)
    elif dr < 0:
        s_rows_start[0] = <uint32_t>(-dr)
        s_rows_end[0] = <uint32_t>rows
    else:
        s_rows_start[0] = 0
        s_rows_end[0] = <uint32_t>rows

    if dc > 0:
        s_cols_start[0] = 0
        s_cols_end[0] = <uint32_t>(cols - dc)
    elif dc < 0:
        s_cols_start[0] = <uint32_t>(-dc)
        s_cols_end[0] = <uint32_t>cols
    else:
        s_cols_start[0] = 0
        s_cols_end[0] = <uint32_t>cols


cdef void _calculate_target_region_bounds(
    int dr, int rows, int dc, int cols,
    uint32_t* t_rows_start, uint32_t* t_rows_end,
    uint32_t* t_cols_start, uint32_t* t_cols_end) noexcept nogil:

    if dr > 0:
        t_rows_start[0] = <uint32_t>dr
        t_rows_end[0] = <uint32_t>rows
    elif dr < 0:
        t_rows_start[0] = 0
        t_rows_end[0] = <uint32_t>(rows + dr)
    else:
        t_rows_start[0] = 0
        t_rows_end[0] = <uint32_t>rows

    if dc > 0:
        t_cols_start[0] = <uint32_t>dc
        t_cols_end[0] = <uint32_t>cols
    elif dc < 0:
        t_cols_start[0] = 0
        t_cols_end[0] = <uint32_t>(cols + dc)
    else:
        t_cols_start[0] = 0
        t_cols_end[0] = <uint32_t>cols


# ==================== SEGMENT LENGTH ====================

cdef double _calculate_segment_length(int abs_dr, int abs_dc) noexcept nogil:
    if abs_dr <= 1 and abs_dc <= 1:
        if abs_dr == 1 and abs_dc == 1:
            return 1.4142135623730951  # sqrt(2)
        return 1.0
    elif (abs_dr == 2 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 2):
        return 2.236067977499789  # sqrt(5)
    elif (abs_dr == 3 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 3):
        return 3.1622776601683795  # sqrt(10)
    elif (abs_dr == 3 and abs_dc == 2) or (abs_dr == 2 and abs_dc == 3):
        return 3.605551275463989  # sqrt(13)
    else:
        return sqrt(<double>(abs_dr * abs_dr + abs_dc * abs_dc))


# ==================== GRADIENT PENALTY ====================

# Gradient mode constants (avoid string comparisons in nogil)
DEF GRAD_EXPONENTIAL = 0
DEF GRAD_POWER = 1
DEF GRAD_ENERGY = 2
DEF GRAD_SIGMOID = 3
DEF GRAD_DEFAULT = 4

cdef int _gradient_mode_from_string(str gradient_function):
    if gradient_function == "exponential":
        return GRAD_EXPONENTIAL
    elif gradient_function == "power":
        return GRAD_POWER
    elif gradient_function == "energy":
        return GRAD_ENERGY
    elif gradient_function == "sigmoid":
        return GRAD_SIGMOID
    else:
        return GRAD_DEFAULT


cdef double _calculate_gradient_penalty(
    double height_diff, double horizontal_dist,
    double edge_length_3d, int gradient_mode) noexcept nogil:

    cdef double slope, penalty, angle_rad, normalized_angle
    cdef double adaptive_exponent, stretch_ratio, gradient_angle
    cdef double gravity_factor, angle_degrees, center, steepness, x, sigmoid_value

    if horizontal_dist < 1e-10:
        return 1000.0

    slope = height_diff / horizontal_dist

    if gradient_mode == GRAD_EXPONENTIAL:
        penalty = exp(slope * slope * 2.0)
        if penalty > 10000.0:
            return 10000.0
        return penalty

    elif gradient_mode == GRAD_POWER:
        angle_rad = atan(slope)
        normalized_angle = fabs(angle_rad) / (PI / 2.0)
        adaptive_exponent = 1.0 + 4.0 * normalized_angle
        penalty = 1.0 + pow(100.0 * normalized_angle, adaptive_exponent)
        if penalty > 10000.0:
            return 10000.0
        return penalty

    elif gradient_mode == GRAD_ENERGY:
        stretch_ratio = edge_length_3d / horizontal_dist
        gradient_angle = atan(slope)
        gravity_factor = sin(gradient_angle)
        penalty = stretch_ratio * exp(3.0 * gravity_factor * gravity_factor)
        if penalty > 10000.0:
            return 10000.0
        return penalty

    elif gradient_mode == GRAD_SIGMOID:
        angle_degrees = atan(slope) * 180.0 / PI
        center = 15.0
        steepness = 0.2
        x = (fabs(angle_degrees) - center) * steepness
        sigmoid_value = 1.0 / (1.0 + exp(-x))
        penalty = 1.0 + 999.0 * sigmoid_value
        return penalty

    else:
        # Default: quadratic
        angle_rad = atan(slope)
        normalized_angle = fabs(angle_rad) / (PI / 2.0)
        penalty = 1.0 + 400.0 * normalized_angle * normalized_angle
        return penalty


# ==================== NODE VALIDATION (2D) ====================

cdef inline int _is_valid_node(
    int sr, int sc, int tr, int tc,
    const uint8_t[:, :] exclude_mask,
    const vector[IntermediatePoint]& intermediates,
    const uint16_t[:, :] raster,
    int rows, int cols,
    double* out_cost) noexcept nogil:
    """Returns 1 if valid, 0 if not. Writes cost to out_cost[0]."""
    cdef int i, ir, ic
    cdef double cost = 0.0
    cdef IntermediatePoint pt

    if sr < 0 or sr >= rows or sc < 0 or sc >= cols:
        return 0
    if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
        return 0
    if exclude_mask[sr, sc] == 0 or exclude_mask[tr, tc] == 0:
        return 0

    for i in range(<int>intermediates.size()):
        pt = intermediates[i]
        ir = sr + pt.dr
        ic = sc + pt.dc
        if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
            return 0
        if exclude_mask[ir, ic] == 0:
            return 0
        cost += <double>raster[ir, ic]

    cost += <double>raster[sr, sc] + <double>raster[tr, tc]
    out_cost[0] = cost
    return 1


# ==================== FIND VALID NODES (2D) ====================

cdef tuple _find_valid_nodes(
    int dr, int dc,
    uint32_t s_rows_start, uint32_t s_rows_end,
    uint32_t s_cols_start, uint32_t s_cols_end,
    const uint8_t[:, :] exclude_mask,
    const uint16_t[:, :] raster,
    const vector[IntermediatePoint]& intermediates,
    int rows, int cols,
    double cost_factor, int max_nodes):

    cdef uint32_t max_valid_nodes = min(
        <uint32_t>((s_rows_end - s_rows_start) * (s_cols_end - s_cols_start)),
        <uint32_t>max_nodes)

    cdef np.ndarray[uint32_t, ndim=1] from_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    cdef np.ndarray[uint32_t, ndim=1] to_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    cdef np.ndarray[float64_t, ndim=1] costs = np.zeros(max_valid_nodes, dtype=np.float64)

    cdef uint32_t valid_count = 0
    cdef int sr, sc, tr, tc
    cdef double cost_temp

    for sr in range(<int>s_rows_start, <int>s_rows_end):
        for sc in range(<int>s_cols_start, <int>s_cols_end):
            tr = sr + dr
            tc = sc + dc

            if _is_valid_node(sr, sc, tr, tc, exclude_mask, intermediates,
                              raster, rows, cols, &cost_temp):
                if valid_count < max_valid_nodes:
                    from_nodes[valid_count] = ravel_index(sr, sc, cols)
                    to_nodes[valid_count] = ravel_index(tr, tc, cols)
                    costs[valid_count] = cost_temp * cost_factor
                    valid_count += 1
                else:
                    break

    return (from_nodes[:valid_count], to_nodes[:valid_count],
            costs[:valid_count], <int>valid_count)


# ==================== FIND VALID NODES (3D) ====================

cdef tuple _find_valid_nodes_3d(
    int dr, int dc,
    uint32_t s_rows_start, uint32_t s_rows_end,
    uint32_t s_cols_start, uint32_t s_cols_end,
    const uint8_t[:, :] exclude_mask,
    const uint16_t[:, :] raster,
    const float32_t[:, :] dem_data,
    const vector[IntermediatePoint]& intermediates,
    int rows, int cols,
    int gradient_mode, int max_nodes):

    cdef uint32_t max_valid_nodes = min(
        <uint32_t>((s_rows_end - s_rows_start) * (s_cols_end - s_cols_start)),
        <uint32_t>max_nodes)

    cdef np.ndarray[uint32_t, ndim=1] from_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    cdef np.ndarray[uint32_t, ndim=1] to_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    cdef np.ndarray[float64_t, ndim=1] costs = np.zeros(max_valid_nodes, dtype=np.float64)

    cdef uint32_t valid_count = 0
    cdef int sr, sc, tr, tc, i, ir, ic
    cdef double base_cost, height_diff, horizontal_dist, edge_length_3d
    cdef double cost_factor_3d, divisor, gradient_penalty, final_cost
    cdef int valid
    cdef IntermediatePoint pt
    cdef int num_intermediates = <int>intermediates.size()

    horizontal_dist = sqrt(<double>(dr * dr + dc * dc))

    for sr in range(<int>s_rows_start, <int>s_rows_end):
        for sc in range(<int>s_cols_start, <int>s_cols_end):
            tr = sr + dr
            tc = sc + dc

            if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
                continue
            if exclude_mask[sr, sc] == 0 or exclude_mask[tr, tc] == 0:
                continue

            valid = 1
            base_cost = 0.0

            for i in range(num_intermediates):
                pt = intermediates[i]
                ir = sr + pt.dr
                ic = sc + pt.dc
                if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
                    valid = 0
                    break
                if exclude_mask[ir, ic] == 0:
                    valid = 0
                    break
                base_cost += <double>raster[ir, ic]

            if not valid:
                continue

            base_cost += <double>raster[sr, sc] + <double>raster[tr, tc]

            height_diff = fabs(<double>dem_data[tr, tc] - <double>dem_data[sr, sc])
            edge_length_3d = sqrt(horizontal_dist * horizontal_dist +
                                  height_diff * height_diff)

            divisor = 2.0 + <double>num_intermediates
            cost_factor_3d = edge_length_3d / divisor

            final_cost = base_cost * cost_factor_3d

            gradient_penalty = _calculate_gradient_penalty(
                height_diff, horizontal_dist, edge_length_3d, gradient_mode)
            final_cost = final_cost * gradient_penalty

            if final_cost > <double>IMPASSABLE:
                final_cost = <double>IMPASSABLE

            if valid_count < max_valid_nodes:
                from_nodes[valid_count] = ravel_index(sr, sc, cols)
                to_nodes[valid_count] = ravel_index(tr, tc, cols)
                costs[valid_count] = final_cost
                valid_count += 1
            else:
                break

    return (from_nodes[:valid_count], to_nodes[:valid_count],
            costs[:valid_count], <int>valid_count)


# ====================================================================
# PUBLIC PYTHON API — these are the functions imported by traversal.py
# ====================================================================

def find_nearest_valid_positions_numba(
        np.ndarray[uint16_t, ndim=2] raster_data,
        np.ndarray[int32_t, ndim=2] invalid_positions,
        int max_value):
    """Find nearest valid position for each invalid position."""
    cdef int rows = raster_data.shape[0]
    cdef int cols = raster_data.shape[1]
    cdef int num_positions = invalid_positions.shape[0]
    cdef np.ndarray[int32_t, ndim=2] corrected = np.empty((num_positions, 2), dtype=np.int32)

    cdef int i, row, col, radius, max_radius, dr, dc
    cdef int new_row, new_col, best_row, best_col
    cdef double min_dist, dist
    cdef int found

    for i in range(num_positions):
        row = invalid_positions[i, 0]
        col = invalid_positions[i, 1]

        min_dist = 1e300
        best_row = row
        best_col = col
        found = 0

        max_radius = rows if rows > cols else cols
        for radius in range(1, max_radius):
            if found:
                break

            for dr in range(-radius, radius + 1):
                new_row = row + dr
                if new_row < 0 or new_row >= rows:
                    continue

                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue

                    new_col = col + dc
                    if new_col < 0 or new_col >= cols:
                        continue

                    if raster_data[new_row, new_col] != max_value:
                        dist = sqrt(<double>(dr * dr + dc * dc))
                        if dist < min_dist:
                            min_dist = dist
                            best_row = new_row
                            best_col = new_col
                            found = 1

        corrected[i, 0] = best_row
        corrected[i, 1] = best_col

    return corrected


def check_max_values(np.ndarray[uint16_t, ndim=2] raster_data,
                     np.ndarray[int32_t, ndim=2] indices_2d,
                     int max_value):
    """Check which positions have the maximum value."""
    cdef int num_positions = indices_2d.shape[0]
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] invalid_mask = np.empty(num_positions, dtype=np.bool_)
    cdef int invalid_count = 0
    cdef int i, row, col, idx

    for i in range(num_positions):
        row = indices_2d[i, 0]
        col = indices_2d[i, 1]
        if raster_data[row, col] == max_value:
            invalid_mask[i] = True
            invalid_count += 1
        else:
            invalid_mask[i] = False

    cdef np.ndarray[int32_t, ndim=2] invalid_indices = np.empty((invalid_count, 2), dtype=np.int32)
    idx = 0
    for i in range(num_positions):
        if invalid_mask[i]:
            invalid_indices[idx, 0] = indices_2d[i, 0]
            invalid_indices[idx, 1] = indices_2d[i, 1]
            idx += 1

    return invalid_count > 0, invalid_mask, invalid_indices


def intermediate_steps_numba(int8_t dr, int8_t dc):
    """Calculate intermediate steps — thin wrapper around Cython implementation."""
    cdef vector[IntermediatePoint] result = _calculate_intermediate_steps_cython(dr, dc)
    cdef int n = <int>result.size()
    cdef np.ndarray[int8_t, ndim=2] out = np.zeros((n, 2), dtype=np.int8)
    cdef int i

    for i in range(n):
        out[i, 0] = result[i].dr
        out[i, 1] = result[i].dc

    return out


def get_cost_factor_numba(int8_t dr, int8_t dc, int intermediates_count):
    """Calculate cost factor — thin wrapper around Cython implementation."""
    return _get_cost_factor_cython(dr, dc, intermediates_count)


def py_ravel_index(int row, int col, int cols):
    """Convert 2D grid coordinates to 1D linear index."""
    return ravel_index(row, col, cols)


def calculate_region_bounds(int8_t dr, int8_t dc, int rows, int cols):
    """Calculate valid source region bounds for graph construction."""
    cdef uint32_t s_rows_start, s_rows_end, s_cols_start, s_cols_end
    _calculate_source_region_bounds(dr, rows, dc, cols,
                                    &s_rows_start, &s_rows_end,
                                    &s_cols_start, &s_cols_end)
    return np.array([s_rows_start, s_rows_end, s_cols_start, s_cols_end],
                    dtype=np.uint32)


def is_valid_node(int sr, int sc, int tr, int tc,
                  np.ndarray[uint8_t, ndim=2] exclude_mask,
                  np.ndarray[int8_t, ndim=2] intermediates_arr,
                  np.ndarray[uint16_t, ndim=2] raster,
                  int rows, int cols,
                  np.ndarray[float64_t, ndim=1] out_cost):
    """Check if a node transition is valid and calculate its traversal cost."""
    cdef vector[IntermediatePoint] intermediates
    cdef IntermediatePoint pt
    cdef int i

    for i in range(intermediates_arr.shape[0]):
        pt.dr = intermediates_arr[i, 0]
        pt.dc = intermediates_arr[i, 1]
        intermediates.push_back(pt)

    cdef double cost_out = 0.0
    cdef int valid = _is_valid_node(sr, sc, tr, tc,
                                     exclude_mask, intermediates,
                                     raster, rows, cols, &cost_out)
    if valid:
        out_cost[0] = cost_out
    return True if valid else False


def find_valid_nodes(int8_t dr, int8_t dc,
                     int s_rows_start, int s_rows_end,
                     int s_cols_start, int s_cols_end,
                     np.ndarray[uint8_t, ndim=2] exclude_mask,
                     np.ndarray[uint16_t, ndim=2] raster,
                     np.ndarray[int8_t, ndim=2] intermediates_arr,
                     int rows, int cols,
                     double cost_factor, int max_nodes):
    """Find all valid node transitions for a given step direction."""
    cdef vector[IntermediatePoint] intermediates
    cdef IntermediatePoint pt
    cdef int i

    for i in range(intermediates_arr.shape[0]):
        pt.dr = intermediates_arr[i, 0]
        pt.dc = intermediates_arr[i, 1]
        intermediates.push_back(pt)

    return _find_valid_nodes(dr, dc,
                             <uint32_t>s_rows_start, <uint32_t>s_rows_end,
                             <uint32_t>s_cols_start, <uint32_t>s_cols_end,
                             exclude_mask, raster, intermediates,
                             rows, cols, cost_factor, max_nodes)


def get_max_number_of_edges(uint32_t n, uint32_t m,
                             np.ndarray[int8_t, ndim=2] steps):
    """Calculate the maximum number of edges for a given raster and neighborhood."""
    cdef uint32_t max_edges = 0
    cdef int step_idx
    for step_idx in range(steps.shape[0]):
        max_edges += (n - <uint32_t>abs(steps[step_idx, 0])) * \
                     (m - <uint32_t>abs(steps[step_idx, 1]))
    return max_edges


def construct_edges(np.ndarray[uint16_t, ndim=2] raster,
                    np.ndarray[int8_t, ndim=2] steps,
                    bint ignore_max=True):
    """Construct graph edges from rasterized geodata."""
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef uint32_t nr_of_edges = get_max_number_of_edges(rows, cols, steps)

    cdef np.ndarray[uint32_t, ndim=1] from_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    cdef np.ndarray[uint32_t, ndim=1] to_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    cdef np.ndarray[float64_t, ndim=1] cost_edges = np.zeros(nr_of_edges, dtype=np.float64)
    cdef uint32_t last_index = 0

    # Create exclude mask
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask
    cdef int i, j
    if ignore_max:
        exclude_mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if raster[i, j] != IMPASSABLE:
                    exclude_mask[i, j] = 1
    else:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)

    cdef int step_idx, dr, dc
    cdef vector[IntermediatePoint] intermediates
    cdef double cost_factor
    cdef uint32_t s_rows_start, s_rows_end, s_cols_start, s_cols_end
    cdef uint32_t remaining, end_idx, valid_count

    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        intermediates = _calculate_intermediate_steps_cython(dr, dc)
        cost_factor = _get_cost_factor_cython(dr, dc, <int>intermediates.size())

        _calculate_source_region_bounds(dr, rows, dc, cols,
                                        &s_rows_start, &s_rows_end,
                                        &s_cols_start, &s_cols_end)

        remaining = nr_of_edges - last_index

        result = _find_valid_nodes(dr, dc,
                                   s_rows_start, s_rows_end,
                                   s_cols_start, s_cols_end,
                                   exclude_mask, raster, intermediates,
                                   rows, cols, cost_factor, <int>remaining)

        from_arr, to_arr, cost_arr, vc = result
        valid_count = <uint32_t>vc
        if valid_count > 0:
            end_idx = last_index + valid_count
            from_nodes_edges[last_index:end_idx] = from_arr
            to_nodes_edges[last_index:end_idx] = to_arr
            cost_edges[last_index:end_idx] = cost_arr
            last_index = end_idx

    return (from_nodes_edges[:last_index], to_nodes_edges[:last_index],
            cost_edges[:last_index])


def calculate_segment_length(int abs_dr, int abs_dc):
    """Calculate the geometric length of a path segment."""
    return _calculate_segment_length(abs_dr, abs_dc)


def calculate_path_metrics_numba(np.ndarray[uint16_t, ndim=2] raster,
                                  np.ndarray[uint32_t, ndim=1] path_indices):
    """Calculate comprehensive metrics for a power line path."""
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_segments = len(path_indices) - 1
    cdef int n_path = len(path_indices)

    # Convert linear indices to 2D coordinates
    cdef np.ndarray[int64_t, ndim=2] path_2d = np.empty((n_path, 2), dtype=np.int64)
    cdef int i
    for i in range(n_path):
        path_2d[i, 0] = path_indices[i] // cols
        path_2d[i, 1] = path_indices[i] % cols

    # Identify unique cost categories
    cdef np.ndarray[uint16_t, ndim=1] categories_array = np.sort(np.unique(raster))
    cdef int num_categories = len(categories_array)

    # Create category-to-index mapping
    cdef uint16_t min_category = categories_array[0]
    cdef uint16_t max_category = categories_array[num_categories - 1]
    cdef int range_size = <int>(max_category - min_category + 1)
    cdef np.ndarray[int32_t, ndim=1] category_to_index = np.full(range_size, -1, dtype=np.int32)

    for i in range(num_categories):
        category_to_index[<int>(categories_array[i] - min_category)] = i

    # Accumulate metrics
    cdef double total_length = 0.0
    cdef np.ndarray[float64_t, ndim=1] lengths_array = np.zeros(num_categories, dtype=np.float64)

    cdef int64_t row, col, next_row, next_col, r, c
    cdef int dr, dc, abs_dr_val, abs_dc_val, j
    cdef double segment_length, cell_length
    cdef uint16_t category
    cdef int map_idx, idx
    cdef vector[IntermediatePoint] intermediates
    cdef IntermediatePoint pt
    cdef int num_cells

    for i in range(n_segments):
        row = path_2d[i, 0]
        col = path_2d[i, 1]
        next_row = path_2d[i + 1, 0]
        next_col = path_2d[i + 1, 1]

        dr = <int>(next_row - row)
        dc = <int>(next_col - col)
        abs_dr_val = abs(dr)
        abs_dc_val = abs(dc)

        segment_length = _calculate_segment_length(abs_dr_val, abs_dc_val)
        total_length += segment_length

        # Get intermediates
        intermediates = _calculate_intermediate_steps_cython(dr, dc)
        num_cells = <int>intermediates.size() + 2  # source + intermediates + target
        cell_length = segment_length / <double>num_cells

        # Source cell
        category = raster[<int>row, <int>col]
        if min_category <= category <= max_category:
            map_idx = <int>(category - min_category)
            idx = category_to_index[map_idx]
            if idx >= 0:
                lengths_array[idx] += cell_length

        # Intermediate cells
        for j in range(<int>intermediates.size()):
            pt = intermediates[j]
            r = row + pt.dr
            c = col + pt.dc
            if 0 <= r < rows and 0 <= c < cols:
                category = raster[<int>r, <int>c]
                if min_category <= category <= max_category:
                    map_idx = <int>(category - min_category)
                    idx = category_to_index[map_idx]
                    if idx >= 0:
                        lengths_array[idx] += cell_length

        # Target cell
        category = raster[<int>next_row, <int>next_col]
        if min_category <= category <= max_category:
            map_idx = <int>(category - min_category)
            idx = category_to_index[map_idx]
            if idx >= 0:
                lengths_array[idx] += cell_length

    return total_length, categories_array, lengths_array


def euclidean_distances_numba(np.ndarray[float64_t, ndim=2] points,
                               np.ndarray[float64_t, ndim=1] target_point):
    """Calculate Euclidean distances from all points to a target point."""
    cdef int n_points = points.shape[0]
    cdef int n_dims = points.shape[1]
    cdef np.ndarray[float64_t, ndim=1] distances = np.empty(n_points, dtype=np.float64)
    cdef int i, j
    cdef double dx, dy, squared_dist, diff

    if n_dims == 2:
        for i in range(n_points):
            dx = points[i, 0] - target_point[0]
            dy = points[i, 1] - target_point[1]
            distances[i] = sqrt(dx * dx + dy * dy)
    else:
        for i in range(n_points):
            squared_dist = 0.0
            for j in range(n_dims):
                diff = points[i, j] - target_point[j]
                squared_dist += diff * diff
            distances[i] = sqrt(squared_dist)

    return distances


def get_outgoing_edges(int node_idx,
                       np.ndarray[uint16_t, ndim=2] raster,
                       np.ndarray[int8_t, ndim=2] steps,
                       int rows, int cols,
                       exclude_mask_in=None):
    """Get outgoing edges from a specific node."""
    cdef int row = node_idx // cols
    cdef int col = node_idx % cols

    cdef int max_edges = steps.shape[0]
    cdef np.ndarray[uint32_t, ndim=1] to_nodes = np.zeros(max_edges, dtype=np.uint32)
    cdef np.ndarray[float64_t, ndim=1] costs_out = np.zeros(max_edges, dtype=np.float64)
    cdef int edge_count = 0

    cdef np.ndarray[uint8_t, ndim=2] exclude_mask
    cdef int i, j
    if exclude_mask_in is None:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if raster[i, j] == IMPASSABLE:
                    exclude_mask[i, j] = 0
    else:
        exclude_mask = exclude_mask_in

    cdef int step_idx, dr, dc, tr, tc, ir, ic
    cdef int valid
    cdef double cost, cost_factor
    cdef vector[IntermediatePoint] intermediates
    cdef IntermediatePoint pt

    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        tr = row + dr
        tc = col + dc

        if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
            continue
        if exclude_mask[tr, tc] == 0:
            continue

        intermediates = _calculate_intermediate_steps_cython(dr, dc)
        valid = 1
        cost = <double>raster[row, col]

        for i in range(<int>intermediates.size()):
            pt = intermediates[i]
            ir = row + pt.dr
            ic = col + pt.dc
            if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
                valid = 0
                break
            if exclude_mask[ir, ic] == 0:
                valid = 0
                break
            cost += <double>raster[ir, ic]

        if not valid:
            continue

        cost += <double>raster[tr, tc]
        cost_factor = _get_cost_factor_cython(dr, dc, <int>intermediates.size())

        to_nodes[edge_count] = <uint32_t>(tr * cols + tc)
        costs_out[edge_count] = cost * cost_factor
        edge_count += 1

    return to_nodes[:edge_count], costs_out[:edge_count]


def calculate_gradient_penalty(double height_diff, double horizontal_dist,
                                double edge_length_3d, str gradient_function):
    """Calculate gradient penalty factor based on terrain slope."""
    cdef int mode = _gradient_mode_from_string(gradient_function)
    return _calculate_gradient_penalty(height_diff, horizontal_dist,
                                       edge_length_3d, mode)


def find_valid_nodes_3d(int8_t dr, int8_t dc,
                         int s_rows_start, int s_rows_end,
                         int s_cols_start, int s_cols_end,
                         np.ndarray[uint8_t, ndim=2] exclude_mask,
                         np.ndarray[uint16_t, ndim=2] raster,
                         np.ndarray[float32_t, ndim=2] dem_data,
                         np.ndarray[int8_t, ndim=2] intermediates_arr,
                         int rows, int cols,
                         str gradient_function,
                         int max_nodes):
    """Find all valid 3D node transitions for a given step direction."""
    cdef vector[IntermediatePoint] intermediates
    cdef IntermediatePoint pt
    cdef int i

    for i in range(intermediates_arr.shape[0]):
        pt.dr = intermediates_arr[i, 0]
        pt.dc = intermediates_arr[i, 1]
        intermediates.push_back(pt)

    cdef int gradient_mode = _gradient_mode_from_string(gradient_function)

    return _find_valid_nodes_3d(dr, dc,
                                <uint32_t>s_rows_start, <uint32_t>s_rows_end,
                                <uint32_t>s_cols_start, <uint32_t>s_cols_end,
                                exclude_mask, raster, dem_data, intermediates,
                                rows, cols, gradient_mode, max_nodes)


def construct_edges_3d(np.ndarray[uint16_t, ndim=2] raster,
                        np.ndarray[float32_t, ndim=2] dem_data,
                        np.ndarray[int8_t, ndim=2] steps,
                        bint ignore_max=True,
                        str gradient_function="squared"):
    """Construct graph edges from rasterized geodata with DEM elevation data."""
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef uint32_t nr_of_edges = get_max_number_of_edges(rows, cols, steps)
    cdef int gradient_mode = _gradient_mode_from_string(gradient_function)

    cdef np.ndarray[uint32_t, ndim=1] from_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    cdef np.ndarray[uint32_t, ndim=1] to_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    cdef np.ndarray[float64_t, ndim=1] cost_edges = np.zeros(nr_of_edges, dtype=np.float64)
    cdef uint32_t last_index = 0

    # Create exclude mask
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask
    cdef int i, j
    if ignore_max:
        exclude_mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if raster[i, j] != IMPASSABLE:
                    exclude_mask[i, j] = 1
    else:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)

    cdef int step_idx, dr, dc
    cdef vector[IntermediatePoint] intermediates
    cdef uint32_t s_rows_start, s_rows_end, s_cols_start, s_cols_end
    cdef uint32_t remaining, end_idx, valid_count

    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        intermediates = _calculate_intermediate_steps_cython(dr, dc)

        _calculate_source_region_bounds(dr, rows, dc, cols,
                                        &s_rows_start, &s_rows_end,
                                        &s_cols_start, &s_cols_end)

        remaining = nr_of_edges - last_index

        result = _find_valid_nodes_3d(dr, dc,
                                      s_rows_start, s_rows_end,
                                      s_cols_start, s_cols_end,
                                      exclude_mask, raster, dem_data,
                                      intermediates, rows, cols,
                                      gradient_mode, <int>remaining)

        from_arr, to_arr, cost_arr, vc = result
        valid_count = <uint32_t>vc
        if valid_count > 0:
            end_idx = last_index + valid_count
            from_nodes_edges[last_index:end_idx] = from_arr
            to_nodes_edges[last_index:end_idx] = to_arr
            cost_edges[last_index:end_idx] = cost_arr
            last_index = end_idx

    return (from_nodes_edges[:last_index], to_nodes_edges[:last_index],
            cost_edges[:last_index])
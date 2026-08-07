"""
Raster geometry, precomputation, and path validation utilities
for high-performance pathfinding operations.

Extracted from path_core.pyx as the second module in the OO refactoring.
Contains:
- Intermediate step calculations for complex movement patterns
- Cost factor computation
- Path validation (cached and uncached variants)
- Direction precomputation
- RasterContext cdef class bundling raster data + precomputed state
- Utility functions: create_exclude_mask, path_cost, path_cost_uint32
"""

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sqrtf, floor, ceil, abs
from libcpp.vector cimport vector
from libcpp cimport bool

from pyorps.utils._heap cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
    StepData, CachedStepData, IntermediatePoint, SystemLimits,
    INF_F32, get_system_limits, ravel_index, unravel_index,
)

# ==================== INTERMEDIATE STEP CALCULATIONS ====================

cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) noexcept nogil:
    """
    Calculate intermediate steps for movement between non-adjacent cells.

    For simple moves (distance <= 1): No intermediate steps needed.
    For single-step diagonals: Add orthogonal components separately.
    For complex moves: Use linear interpolation with floor/ceil sampling.

    Parameters:
        dr: Row displacement
        dc: Column displacement

    Returns:
        Vector of IntermediatePoint structs representing the path steps
    """
    cdef vector[IntermediatePoint] result
    cdef IntermediatePoint point
    cdef int abs_dr = abs(dr)
    cdef int abs_dc = abs(dc)
    cdef int sum_abs = abs_dr + abs_dc
    cdef int k, p
    cdef double dr_k, dc_k, ddr, ddc, dk, dp
    cdef int8_t floor_dr, floor_dc, ceil_dr, ceil_dc

    if sum_abs <= 1:
        # Adjacent or same cell - no intermediate steps required
        pass
    elif max(abs_dr, abs_dc) == 1:
        # Single diagonal step - decompose into orthogonal components
        point.dr = <int8_t>dr
        point.dc = 0
        result.push_back(point)

        point.dr = 0
        point.dc = <int8_t>dc
        result.push_back(point)
    else:
        # Complex movement requiring linear interpolation
        k = max(abs_dr, abs_dc)  # Number of major steps
        ddr = <double>dr
        ddc = <double>dc
        dk = <double>k

        for p in range(1, k):
            dp = <double>p
            # Calculate fractional position along the movement vector
            dr_k = (dp * ddr) / dk
            dc_k = (dp * ddc) / dk

            # Add floor approximation (conservative path)
            floor_dr = <int8_t>floor(dr_k)
            floor_dc = <int8_t>floor(dc_k)
            point.dr = floor_dr
            point.dc = floor_dc
            result.push_back(point)

            # Add ceiling approximation if different (ensures connectivity)
            ceil_dr = <int8_t>ceil(dr_k)
            ceil_dc = <int8_t>ceil(dc_k)
            if floor_dr != ceil_dr or floor_dc != ceil_dc:
                point.dr = ceil_dr
                point.dc = ceil_dc
                result.push_back(point)
    return result


cdef inline double _get_cost_factor_cython(int dr, int dc, int intermediates_count) noexcept nogil:
    """
    Calculate movement cost factor incorporating distance and path complexity.

    Formula: distance / (2.0 + intermediate_steps_count)

    Parameters:
        dr: Row displacement for the movement
        dc: Column displacement for the movement
        intermediates_count: Number of intermediate steps required

    Returns:
        Multiplicative factor to apply to raw cell costs (always positive)
    """
    cdef double distance = sqrt(<double>(dr * dr + dc * dc))
    cdef double divisor = 2.0 + <double>intermediates_count
    return distance / divisor


cdef inline float _get_cost_factor_cython_f32(int dr, int dc, int intermediates_count) noexcept nogil:
    """
    Float32 version of cost factor calculation for memory-efficient operations.
    """
    cdef float distance = sqrtf(<float>(dr * dr + dc * dc))
    cdef float divisor = <float>2.0 + <float>intermediates_count
    return distance / divisor


# ==================== CACHED STEP PRECOMPUTATION ====================

cdef vector[CachedStepData] precompute_cached_steps(np.ndarray[int8_t, ndim=2] steps_arr):
    """
    Precompute and cache intermediate steps for all movement directions.

    Parameters:
        steps_arr: 2D array where each row contains [dr, dc] for one direction

    Returns:
        Vector of CachedStepData containing precomputed intermediate steps
    """
    cdef vector[CachedStepData] cached_steps
    cdef CachedStepData step_cache
    cdef int num_steps, s, dr, dc
    cdef vector[IntermediatePoint] intermediates

    num_steps = <int>steps_arr.shape[0]
    cached_steps.reserve(<size_t>num_steps)

    for s in range(num_steps):
        dr = steps_arr[s, 0]
        dc = steps_arr[s, 1]

        intermediates = _calculate_intermediate_steps_cython(dr, dc)
        step_cache.intermediates = intermediates
        step_cache.intermediate_count = <int>intermediates.size()

        cached_steps.push_back(step_cache)

    return cached_steps


cdef int check_path_cached(const vector[IntermediatePoint]& cached_intermediates,
                          int current_row, int current_col,
                          const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                          int rows, int cols, float* total_cost) except -1 nogil:
    """
    Validate movement path using cached intermediate steps.

    Parameters:
        cached_intermediates: Precomputed intermediate steps for this direction
        current_row: Starting row position
        current_col: Starting column position
        exclude_mask: 2D traversability mask
        raster: 2D cost raster
        rows: Total rows in raster
        cols: Total columns in raster
        total_cost: Output parameter for intermediate costs

    Returns:
        1 if path is valid, 0 if blocked
    """
    cdef float cost = 0.0
    cdef int i, int_row, int_col, num_intermediates
    cdef IntermediatePoint point

    num_intermediates = <int>cached_intermediates.size()

    for i in range(num_intermediates):
        point = cached_intermediates[i]
        int_row = current_row + point.dr
        int_col = current_col + point.dc

        if int_row < 0 or int_row >= rows or int_col < 0 or int_col >= cols:
            return 0

        if exclude_mask[int_row, int_col] == 0:
            return 0

        cost += <float>raster[int_row, int_col]

    total_cost[0] = cost
    return 1


cdef vector[StepData] precompute_directions_optimized(np.ndarray[int8_t, ndim=2] steps_arr,
                                                     const vector[CachedStepData]& cached_steps):
    """
    Create optimized direction data using cached intermediate steps.

    Parameters:
        steps_arr: Raw movement directions
        cached_steps: Precomputed intermediate step data

    Returns:
        Vector of StepData with directions and cost factors
    """
    cdef vector[StepData] directions
    cdef StepData direction
    cdef int s, dr, dc, steps_count

    steps_count = <int>steps_arr.shape[0]
    directions.reserve(<size_t>steps_count)

    for s in range(steps_count):
        dr = steps_arr[s, 0]
        dc = steps_arr[s, 1]

        direction.dr = dr
        direction.dc = dc
        direction.cost_factor = _get_cost_factor_cython_f32(dr, dc, cached_steps[s].intermediate_count)

        directions.push_back(direction)

    return directions


# ==================== PATH VALIDATION ====================

cdef int check_path(int dr, int dc, int current_row, int current_col,
                    const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                    int rows, int cols, double* total_cost) except -1 nogil:
    """
    Validate a movement path and calculate intermediate step costs.

    Parameters:
        dr: Row displacement for the movement
        dc: Column displacement for the movement
        current_row: Starting row position
        current_col: Starting column position
        exclude_mask: 2D mask indicating traversable cells (1=ok, 0=blocked)
        raster: 2D cost raster for calculating traversal costs
        rows: Total number of rows in the raster
        cols: Total number of columns in the raster
        total_cost: Pointer to store the calculated intermediate costs

    Returns:
        1 if path is valid, 0 if path is blocked or out of bounds
    """
    cdef double cost = 0.0
    cdef int i, int_row, int_col

    # Get intermediate steps for this movement
    cdef vector[IntermediatePoint] intermediates = (
        _calculate_intermediate_steps_cython(dr, dc))
    cdef IntermediatePoint point

    # Check each intermediate point along the path
    for i in range(intermediates.size()):
        point = intermediates[i]
        int_row = current_row + point.dr
        int_col = current_col + point.dc

        # Validate bounds and traversability
        if (int_row < 0 or int_row >= rows or
                int_col < 0 or int_col >= cols or
                exclude_mask[int_row, int_col] == 0):
            return 0  # Invalid path

        # Accumulate cost of traversing this intermediate cell
        cost += raster[int_row, int_col]

    # Path is valid - return total intermediate cost
    total_cost[0] = cost
    return 1


cdef vector[StepData] precompute_directions(np.ndarray[int8_t, ndim=2] steps_arr):
    """
    Precompute movement data for all possible directions in the neighborhood.

    Parameters:
        steps_arr: 2D array where each row contains [dr, dc] for one direction

    Returns:
        Vector of StepData structures with precomputed movement information
    """
    cdef vector[StepData] directions
    cdef StepData direction
    cdef int s, dr, dc
    cdef int intermediates_count
    cdef int steps_count = <int>steps_arr.shape[0]

    directions.reserve(steps_count)

    for s in range(steps_count):
        dr = steps_arr[s, 0]
        dc = steps_arr[s, 1]

        # Count intermediate steps for this direction
        intermediates_count = <int>(
            _calculate_intermediate_steps_cython(dr, dc).size())

        # Store precomputed direction data
        direction.dr = dr
        direction.dc = dc
        direction.cost_factor = <float>_get_cost_factor_cython(
            dr, dc, intermediates_count)

        directions.push_back(direction)

    return directions


# ==================== UTILITY FUNCTIONS ====================

# Obstacle sentinel outside the uint16 cost domain: no cell is an obstacle.
# The GPU kernels use the same value and compare in the int domain, because
# casting it back to uint16 truncates to 0 and forbids every 0-cost cell.
NO_EXCLUSION_VALUE = 65536


cpdef np.ndarray[uint8_t, ndim=2] create_exclude_mask(
        np.ndarray[uint16_t, ndim=2] raster_arr, int64_t max_value):
    """
    Create a binary mask identifying traversable cells in the raster.

    Parameters:
        raster_arr: 2D numpy array containing cost values for each cell
        max_value: Value representing obstacles/barriers (typically 65535).
            Any value outside the uint16 domain (see NO_EXCLUSION_VALUE)
            disables exclusion: every cell stays traversable.

    Returns:
        2D numpy array of uint8 values (1=traversable, 0=obstacle)
    """
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef uint16_t[:, :] raster
    cdef uint8_t[:, :] exclude_mask
    cdef uint16_t obstacle
    cdef int i, j

    # Initialize mask with all cells marked as traversable
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr = np.ones((rows, cols),
                                                                dtype=np.uint8)

    # Out-of-domain sentinel: nothing to exclude. Returning the all-traversable
    # mask keeps the per-relaxation mask lookups branch-free.
    if max_value < 0 or max_value > 65535:
        return exclude_mask_arr

    raster = raster_arr
    exclude_mask = exclude_mask_arr
    obstacle = <uint16_t>max_value

    for i in range(rows):
        for j in range(cols):
            if raster[i, j] == obstacle:
                exclude_mask[i, j] = 0  # Mark as obstacle

    return exclude_mask_arr


cpdef double path_cost(np.ndarray[uint64_t, ndim=1] path,
                       np.ndarray[uint16_t, ndim=2] raster_arr, uint64_t cols):
    """
    Calculate the total traversal cost for a given path through the raster.

    Parameters:
        path: 1D array of linear indices representing the path sequence
        raster_arr: 2D cost raster containing per-cell traversal costs
        cols: Number of columns in raster (for index conversion)

    Returns:
        Total cost as sum of individual cell costs along the path
    """
    cdef int i
    cdef uint64_t idx, row, col
    cdef double cost = 0.0
    cdef int path_len = <int> path.shape[0]

    for i in range(path_len):
        idx = path[i]
        row = idx // cols
        col = idx % cols
        cost += <double> raster_arr[row, col]

    return cost


cpdef double path_cost_uint32(np.ndarray[uint32_t, ndim=1] path,
                              np.ndarray[uint16_t, ndim=2] raster_arr, int cols):
    """
    Calculate the total traversal cost for a given path (uint32 indices).

    Parameters:
        path: 1D array of linear indices (uint32) representing the path sequence
        raster_arr: 2D cost raster containing per-cell traversal costs
        cols: Number of columns in raster (int, matching Dijkstra's internal type)

    Returns:
        Total cost as sum of individual cell costs along the path
    """
    cdef int i
    cdef uint32_t idx
    cdef int row, col
    cdef double cost = 0.0
    cdef int path_len = <int> path.shape[0]

    for i in range(path_len):
        idx = path[i]
        row = <int> (idx // cols)
        col = <int> (idx % cols)
        cost += <double> raster_arr[row, col]

    return cost


# ==================== RASTER CONTEXT CLASS ====================

cdef class RasterContext:
    """
    Bundles raster data with precomputed navigation state for pathfinding.

    Holds the cost raster, traversability mask, precomputed directions,
    cached intermediate steps, and system resource limits. Intended to be
    constructed once and passed into solver objects.

    Attributes:
        rows: Number of rows in the raster
        cols: Number of columns in the raster
        total_cells: Total number of cells (rows * cols)
        raster_view: Typed memory view of the uint16 cost raster
        exclude_mask_view: Typed memory view of the uint8 traversability mask
        directions: Precomputed StepData vector with cost factors
        cached_steps: Precomputed CachedStepData with intermediate points
        sys_limits: System resource limits
    """

    def __cinit__(self, object raster_arr, object steps_arr, int64_t max_value=65535):
        self.rows = <int>raster_arr.shape[0]
        self.cols = <int>raster_arr.shape[1]
        self.total_cells = self.rows * self.cols
        cdef object exclude_mask_arr = create_exclude_mask(raster_arr, max_value)
        self.raster_view = raster_arr
        self.exclude_mask_view = exclude_mask_arr
        self.cached_steps = precompute_cached_steps(steps_arr)
        self.directions = precompute_directions_optimized(steps_arr, self.cached_steps)
        self.sys_limits = get_system_limits()

    @property
    def num_directions(self):
        """Number of precomputed movement directions."""
        return <int>self.directions.size()

    def is_traversable(self, int row, int col):
        """Check if a cell is traversable (not blocked by max_value)."""
        return self.exclude_mask_view[row, col] != 0


# ==================== PYTHON TEST WRAPPERS ====================

def py_create_exclude_mask(np.ndarray[uint16_t, ndim=2] raster_arr, int64_t max_value=65535):
    """Python wrapper for create_exclude_mask for testing."""
    return create_exclude_mask(raster_arr, max_value)


def py_path_cost(np.ndarray[uint64_t, ndim=1] path,
                 np.ndarray[uint16_t, ndim=2] raster_arr, uint64_t cols):
    """Python wrapper for path_cost for testing."""
    return path_cost(path, raster_arr, cols)


def py_path_cost_uint32(np.ndarray[uint32_t, ndim=1] path,
                        np.ndarray[uint16_t, ndim=2] raster_arr, int cols):
    """Python wrapper for path_cost_uint32 for testing."""
    return path_cost_uint32(path, raster_arr, cols)

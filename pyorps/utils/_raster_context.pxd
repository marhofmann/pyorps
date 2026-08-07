# _raster_context.pxd
# Raster geometry, precomputation, and path validation utilities.
# Extracted from path_core.pxd as the second module in the OO refactoring.

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

from pyorps.utils._heap cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
    StepData, CachedStepData, IntermediatePoint, SystemLimits,
)

# ==================== FREE FUNCTIONS ====================

# Intermediate step calculation
cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) noexcept nogil

# Cost factor calculation
cdef double _get_cost_factor_cython(int dr, int dc, int intermediates_count) noexcept nogil
cdef float _get_cost_factor_cython_f32(int dr, int dc, int intermediates_count) noexcept nogil

# Path validation
cdef int check_path(int dr, int dc, int current_row, int current_col,
                    const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                    int rows, int cols, double* total_cost) except -1 nogil

cdef int check_path_cached(const vector[IntermediatePoint]& cached_intermediates,
                          int current_row, int current_col,
                          const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                          int rows, int cols, float* total_cost) except -1 nogil

# ==================== PRECOMPUTATION FUNCTIONS ====================

cdef vector[CachedStepData] precompute_cached_steps(np.ndarray[int8_t, ndim=2] steps_arr)
cdef vector[StepData] precompute_directions(np.ndarray[int8_t, ndim=2] steps_arr)
cdef vector[StepData] precompute_directions_optimized(np.ndarray[int8_t, ndim=2] steps_arr,
                                                     const vector[CachedStepData]& cached_steps)

# ==================== RASTER CONTEXT CLASS ====================

cdef class RasterContext:
    cdef readonly int rows, cols, total_cells
    cdef uint16_t[:, :] raster_view
    cdef uint8_t[:, :] exclude_mask_view
    cdef vector[StepData] directions
    cdef vector[CachedStepData] cached_steps
    cdef SystemLimits sys_limits

# ==================== UTILITY FUNCTIONS ====================

cpdef np.ndarray[uint8_t, ndim=2] create_exclude_mask(
        np.ndarray[uint16_t, ndim=2] raster_arr, int64_t max_value)

cpdef double path_cost(np.ndarray[uint64_t, ndim=1] path,
                      np.ndarray[uint16_t, ndim=2] raster_arr, uint64_t cols)

cpdef double path_cost_uint32(np.ndarray[uint32_t, ndim=1] path,
                              np.ndarray[uint16_t, ndim=2] raster_arr, int cols)

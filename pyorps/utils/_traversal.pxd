# _traversal.pxd
# Graph construction and path analysis utilities.
# Cython port of the Numba functions in traversal.py for faster import.

import numpy as np
cimport numpy as np

from pyorps.utils._heap cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
)

# ==================== REGION BOUNDS ====================

cdef void _calculate_source_region_bounds(
    int dr, int rows, int dc, int cols,
    uint32_t* s_rows_start, uint32_t* s_rows_end,
    uint32_t* s_cols_start, uint32_t* s_cols_end) noexcept nogil

cdef void _calculate_target_region_bounds(
    int dr, int rows, int dc, int cols,
    uint32_t* t_rows_start, uint32_t* t_rows_end,
    uint32_t* t_cols_start, uint32_t* t_cols_end) noexcept nogil

# ==================== SEGMENT LENGTH ====================

cdef double _calculate_segment_length(int abs_dr, int abs_dc) noexcept nogil

# ==================== GRADIENT PENALTY ====================

cdef double _calculate_gradient_penalty(
    double height_diff, double horizontal_dist,
    double edge_length_3d, int gradient_mode) noexcept nogil
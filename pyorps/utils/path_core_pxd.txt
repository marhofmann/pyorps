# path_core.pxd

# Import necessary Cython and NumPy types
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, floor, ceil, abs
from libcpp.vector cimport vector
from libcpp cimport bool

# Type definitions for consistent data handling across the module
ctypedef np.int8_t int8_t
ctypedef np.uint8_t uint8_t
ctypedef np.uint16_t uint16_t
ctypedef np.uint32_t uint32_t
ctypedef np.int32_t int32_t
ctypedef np.float64_t float64_t
ctypedef Py_ssize_t npy_intp

# Core data structures for pathfinding operations
cdef struct IntermediatePoint:
    int8_t dr
    int8_t dc

cdef struct StepData:
    int dr
    int dc
    double cost_factor

cdef struct PQNode:
    uint32_t index
    double priority

cdef struct BinaryHeap:
    vector[PQNode] nodes

# Function declarations with exception values
cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) nogil
cdef double _get_cost_factor_cython(int dr, int dc, int intermediates_count) nogil
cdef int heap_init(BinaryHeap* heap) except -1 nogil
cdef bool heap_empty(const BinaryHeap* heap) nogil
cdef PQNode heap_top(const BinaryHeap* heap) nogil
cdef int heap_push(BinaryHeap* heap, uint32_t idx, double priority) except -1 nogil
cdef int heap_pop(BinaryHeap* heap) except -1 nogil
cdef uint32_t ravel_index(int row, int col, int cols) nogil
cdef int unravel_index(uint32_t idx, int cols, npy_intp* row, npy_intp* col) except -1 nogil

# Utility functions with exception values
cdef int check_path(int dr, int dc, int current_row, int current_col,
                    const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                    int rows, int cols, double* total_cost) except -1 nogil
cdef vector[StepData] precompute_directions(np.ndarray[int8_t, ndim=2] steps_arr)

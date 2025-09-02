# path_core.pxd

# Import necessary Cython and NumPy types
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sqrtf, floor, ceil, abs, logf
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport UINT32_MAX, UINT64_MAX

# Type definitions for consistent data handling across the module
ctypedef np.int8_t int8_t
ctypedef np.uint8_t uint8_t
ctypedef np.uint16_t uint16_t
ctypedef np.uint32_t uint32_t
ctypedef np.int32_t int32_t
ctypedef np.int64_t int64_t
ctypedef np.uint64_t uint64_t
ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t
ctypedef Py_ssize_t npy_intp

# Constants
cdef float INF_F32

# Core data structures for pathfinding operations
cdef struct IntermediatePoint:
    int8_t dr
    int8_t dc

cdef struct StepData:
    int dr
    int dc
    #double cost_factor
    float cost_factor

cdef struct CachedStepData:
    vector[IntermediatePoint] intermediates
    int intermediate_count

cdef struct PQNode:
    uint32_t index
    double priority

cdef struct BinaryHeap:
    vector[PQNode] nodes

# System resource management structures
cdef struct SystemLimits:
    uint64_t max_memory_bytes
    uint64_t available_memory_bytes
    uint64_t max_array_size
    uint64_t max_path_length
    uint32_t max_buckets
    int max_iterations
    int num_cores

cdef struct PreprocessingResult:
    float delta           # Calculated or validated delta value
    float margin          # Calculated early termination margin
    float min_cost        # Minimum traversable cell cost
    float second_min_cost # Second minimum cost
    float avg_cost        # Average cost of traversable cells
    float max_cost        # Maximum cost
    int avg_degree        # Average connectivity degree from steps


# Function declarations with exception values
cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) nogil
cdef double _get_cost_factor_cython(int dr, int dc, int intermediates_count) nogil
cdef float _get_cost_factor_cython_f32(int dr, int dc, int intermediates_count) noexcept nogil

# Binary heap operations
cdef int heap_init(BinaryHeap* heap) except -1 nogil
cdef bool heap_empty(const BinaryHeap* heap) nogil
cdef PQNode heap_top(const BinaryHeap* heap) nogil
cdef int heap_push(BinaryHeap* heap, uint32_t idx, double priority) except -1 nogil
cdef int heap_pop(BinaryHeap* heap) except -1 nogil

# Index conversion functions
cdef uint32_t ravel_index(int row, int col, int cols) nogil
cdef int unravel_index(uint32_t idx, int cols, npy_intp* row, npy_intp* col) except -1 nogil

# Path validation and cost calculation
cdef int check_path(int dr, int dc, int current_row, int current_col,
                    const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                    int rows, int cols, double* total_cost) except -1 nogil

cdef int check_path_cached(const vector[IntermediatePoint]& cached_intermediates,
                          int current_row, int current_col,
                          const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                          int rows, int cols, float* total_cost) except -1 nogil

# Precomputation functions
cdef vector[StepData] precompute_directions(np.ndarray[int8_t, ndim=2] steps_arr)
cdef vector[CachedStepData] precompute_cached_steps(np.ndarray[int8_t, ndim=2] steps_arr)
cdef vector[StepData] precompute_directions_optimized(np.ndarray[int8_t, ndim=2] steps_arr,
                                                     const vector[CachedStepData]& cached_steps)

# System resource functions
cdef SystemLimits get_system_limits() except*
cdef uint32_t calculate_initial_bucket_size(uint64_t total_cells, SystemLimits* limits) noexcept nogil
cdef int calculate_thread_buffer_capacity(uint64_t total_cells, int num_threads, SystemLimits* limits) noexcept nogil

# Utility functions
cpdef np.ndarray[uint8_t, ndim=2] create_exclude_mask(np.ndarray[uint16_t, ndim=2] raster_arr, uint16_t max_value)

# Path cost calculation functions (moved from path_algorithms.pxd)
cpdef double path_cost(np.ndarray[uint64_t, ndim=1] path,
                      np.ndarray[uint16_t, ndim=2] raster_arr, uint64_t cols)
cpdef double path_cost_uint32(np.ndarray[uint32_t, ndim=1] path,
                              np.ndarray[uint16_t, ndim=2] raster_arr, int cols)

# Circular buffer utilities
cdef size_t get_circular_index(size_t logical_bucket, size_t buffer_size) noexcept nogil
cdef bint is_bucket_in_window(size_t logical_bucket, size_t window_start, size_t window_size) noexcept nogil
cdef size_t round_up_power_of_two(size_t n) noexcept nogil



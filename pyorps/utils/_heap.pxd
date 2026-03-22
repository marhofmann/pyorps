# _heap.pxd
# Extracted from path_core.pxd — heap data structures, index utilities,
# system resource management, and circular buffer helpers.

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sqrtf, floor, ceil, abs, logf
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport UINT32_MAX, UINT64_MAX

# ==================== TYPE DEFINITIONS ====================
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

# ==================== CONSTANTS ====================
cdef float INF_F32

# ==================== STRUCT DEFINITIONS ====================
cdef struct IntermediatePoint:
    int8_t dr
    int8_t dc

cdef struct StepData:
    int dr
    int dc
    float cost_factor

cdef struct CachedStepData:
    vector[IntermediatePoint] intermediates
    int intermediate_count

cdef struct PQNode:
    uint32_t index
    double priority

cdef struct BinaryHeap:
    vector[PQNode] nodes

cdef struct PQNode64:
    uint64_t index
    double priority

cdef struct BinaryHeap64:
    PQNode64* nodes
    int size
    int capacity

cdef struct SystemLimits:
    uint64_t max_memory_bytes
    uint64_t available_memory_bytes
    uint64_t max_array_size
    uint64_t max_path_length
    uint32_t max_buckets
    int max_iterations
    int num_cores

# ==================== INDEX CONVERSION ====================
cdef uint32_t ravel_index(int row, int col, int cols) nogil
cdef int unravel_index(uint32_t idx, int cols, npy_intp* row, npy_intp* col) except -1 nogil

# ==================== CIRCULAR BUFFER UTILITIES ====================
cdef size_t get_circular_index(size_t logical_bucket, size_t buffer_size) noexcept nogil
cdef bint is_bucket_in_window(size_t logical_bucket, size_t window_start, size_t window_size) noexcept nogil
cdef size_t round_up_power_of_two(size_t n) noexcept nogil

# ==================== SYSTEM RESOURCE FUNCTIONS ====================
cdef SystemLimits get_system_limits() except*
cdef uint32_t calculate_initial_bucket_size(uint64_t total_cells, SystemLimits* limits) noexcept nogil
cdef int calculate_thread_buffer_capacity(uint64_t total_cells, int num_threads, SystemLimits* limits) noexcept nogil

# ==================== BINARY HEAP 32-BIT ====================
cdef int heap_init(BinaryHeap* heap) except -1 nogil
cdef bool heap_empty(const BinaryHeap* heap) nogil
cdef PQNode heap_top(const BinaryHeap* heap) nogil
cdef int heap_push(BinaryHeap* heap, uint32_t idx, double priority) except -1 nogil
cdef int heap_pop(BinaryHeap* heap) except -1 nogil

# ==================== BINARY HEAP 64-BIT ====================
cdef void heap64_init(BinaryHeap64* heap, int capacity) nogil
cdef void heap64_push(BinaryHeap64* heap, uint64_t index, double priority) nogil
cdef PQNode64 heap64_top(BinaryHeap64* heap) nogil
cdef void heap64_pop(BinaryHeap64* heap) nogil
cdef bint heap64_empty(BinaryHeap64* heap) nogil
cdef void heap64_free(BinaryHeap64* heap) nogil

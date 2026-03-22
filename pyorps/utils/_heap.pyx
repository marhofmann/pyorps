"""
Heap data structures, index utilities, system resource management,
and circular buffer helpers for high-performance pathfinding.

Extracted from path_core.pyx as the first module in the OO refactoring.
"""

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sqrtf, floor, ceil, abs, logf
from libcpp.vector cimport vector
from libcpp cimport bool
import psutil
import sys
from libc.stdint cimport UINT32_MAX, UINT64_MAX
from libc.stdlib cimport malloc, realloc, free as c_free

# Define INFINITY for float32
cdef float INF_F32 = 1e38

# ==================== SYSTEM RESOURCE MANAGEMENT ====================

cdef SystemLimits get_system_limits() except*:
    """
    Query system resources and determine safe operating limits.

    Returns:
        SystemLimits structure with memory, array, path, bucket, iteration,
        and core count limits.
    """
    cdef SystemLimits limits
    cdef object mem_info
    cdef uint64_t total_memory, available_memory
    cdef double safety_factor = 0.85
    cdef uint64_t MAX_SAFE_SIZE = <uint64_t>2147483647

    mem_info = psutil.virtual_memory()
    total_memory = <uint64_t>mem_info.total
    available_memory = <uint64_t>mem_info.available

    limits.max_memory_bytes = <uint64_t>(total_memory * safety_factor)
    limits.available_memory_bytes = <uint64_t>(available_memory * safety_factor)

    limits.max_array_size = min(
        limits.available_memory_bytes // 16,
        <uint64_t>sys.maxsize if sys.maxsize > 0 else MAX_SAFE_SIZE,
        UINT64_MAX // 2
    )

    limits.max_path_length = min(
        limits.max_array_size,
        <uint64_t>sys.maxsize if sys.maxsize > 0 else MAX_SAFE_SIZE
    )

    cdef uint64_t bucket_calc = limits.available_memory_bytes // (1024 * sizeof(uint64_t))
    if bucket_calc > UINT32_MAX // 2:
        limits.max_buckets = UINT32_MAX // 2
    else:
        limits.max_buckets = <uint32_t>bucket_calc

    if sys.maxsize < 2147483647:
        limits.max_iterations = sys.maxsize // 2
    else:
        limits.max_iterations = 2147483647

    if limits.max_array_size < <uint64_t>limits.max_iterations:
        limits.max_iterations = <int>min(limits.max_array_size, <uint64_t>2147483647)

    num_cores = psutil.cpu_count(logical=True)
    limits.num_cores = 12 if num_cores > 12 else num_cores

    return limits


cdef uint32_t calculate_initial_bucket_size(uint64_t total_cells, SystemLimits* limits) noexcept nogil:
    """
    Calculate optimal initial bucket size based on problem size and system resources.
    """
    cdef uint32_t size
    cdef uint64_t memory_based_size

    size = <uint32_t>min(total_cells // 100, 100000)

    memory_based_size = limits.available_memory_bytes // (10000 * sizeof(uint64_t))
    if memory_based_size < size:
        size = <uint32_t>memory_based_size

    if size < 1000:
        size = 1000
    if size > limits.max_buckets // 10:
        size = limits.max_buckets // 10

    return size


cdef int calculate_thread_buffer_capacity(uint64_t total_cells, int num_threads, SystemLimits* limits) noexcept nogil:
    """
    Calculate per-thread buffer capacity for parallel edge relaxation.
    """
    cdef uint64_t per_thread_memory
    cdef uint64_t capacity_64
    cdef int capacity
    cdef int MAX_INT = 2147483647
    cdef int dynamic_minimum
    cdef uint64_t cells_per_thread

    if num_threads <= 0:
        num_threads = 1

    cells_per_thread = total_cells // num_threads

    dynamic_minimum = max(
        256,
        min(
            <int>(cells_per_thread // 100),
            65536
        )
    )

    per_thread_memory = limits.available_memory_bytes // (num_threads * 4)

    capacity_64 = min(
        per_thread_memory // 16,
        total_cells // (num_threads * 10),
        <uint64_t>MAX_INT // 2
    )

    if capacity_64 > limits.max_array_size // num_threads:
        capacity_64 = limits.max_array_size // num_threads

    if capacity_64 > <uint64_t>MAX_INT // 2:
        capacity = MAX_INT // 2
    else:
        capacity = <int>capacity_64

    if capacity < dynamic_minimum:
        capacity = dynamic_minimum

    if capacity < 256:
        capacity = 256

    return capacity


# ==================== BINARY HEAP 32-BIT ====================

cdef inline int heap_init(BinaryHeap* heap) except -1 nogil:
    """Initialize an empty binary heap with default capacity of 1000."""
    heap.nodes.clear()
    heap.nodes.reserve(1000)
    return 0


cdef inline bool heap_empty(const BinaryHeap* heap) noexcept nogil:
    """Check if the binary heap is empty."""
    return heap.nodes.size() == 0


cdef inline PQNode heap_top(const BinaryHeap* heap) noexcept nogil:
    """Return the minimum priority node without removing it."""
    return heap.nodes[0]


cdef inline int heap_push(BinaryHeap* heap, uint32_t idx, double priority) except -1 nogil:
    """Insert a node into the heap maintaining min-heap property. O(log n)."""
    cdef PQNode node
    node.index = idx
    node.priority = priority
    heap.nodes.push_back(node)

    cdef npy_intp pos = heap.nodes.size() - 1
    cdef npy_intp parent
    cdef PQNode temp

    while pos > 0:
        parent = (pos - 1) // 2
        if heap.nodes[parent].priority <= heap.nodes[pos].priority:
            break
        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[parent]
        heap.nodes[parent] = temp
        pos = parent

    return 0


cdef inline int heap_pop(BinaryHeap* heap) except -1 nogil:
    """Remove the minimum priority node. O(log n)."""
    if heap.nodes.size() == 0:
        return 1

    if heap.nodes.size() > 1:
        heap.nodes[0] = heap.nodes[heap.nodes.size() - 1]

    heap.nodes.pop_back()

    if heap.nodes.size() <= 1:
        return 0

    cdef npy_intp pos = 0
    cdef npy_intp left, right, smallest
    cdef npy_intp heap_size = heap.nodes.size()
    cdef PQNode temp

    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos

        if (left < heap_size and
                heap.nodes[left].priority < heap.nodes[smallest].priority):
            smallest = left

        if (right < heap_size and
                heap.nodes[right].priority < heap.nodes[smallest].priority):
            smallest = right

        if smallest == pos:
            break

        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[smallest]
        heap.nodes[smallest] = temp
        pos = smallest
    return 0


# ==================== INDEX CONVERSION FUNCTIONS ====================

cdef inline uint32_t ravel_index(int row, int col, int cols) noexcept nogil:
    """Convert 2D (row, col) to 1D linear index (row-major)."""
    return <uint32_t>(<int64_t>row * <int64_t>cols + col)


cdef inline int unravel_index(uint32_t idx, int cols, npy_intp* row, npy_intp* col) except -1 nogil:
    """Convert 1D linear index back to 2D (row, col)."""
    row[0] = idx // cols
    col[0] = idx % cols
    return 0


# ==================== CIRCULAR BUFFER UTILITIES ====================

cdef inline size_t get_circular_index(size_t logical_bucket, size_t buffer_size) noexcept nogil:
    """Map logical bucket index to physical position using bitwise AND (power-of-2 sizes)."""
    return logical_bucket & (buffer_size - 1)


cdef inline bint is_bucket_in_window(size_t logical_bucket, size_t window_start,
                                     size_t window_size) noexcept nogil:
    """Check if a logical bucket index is within the current processing window."""
    return logical_bucket >= window_start and logical_bucket < window_start + window_size


cdef size_t round_up_power_of_two(size_t n) noexcept nogil:
    """Round up to the nearest power of two."""
    if n <= 1:
        return 1

    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    if sizeof(size_t) > 4:
        n |= n >> 32
    n += 1

    return n


# ==================== BINARY HEAP 64-BIT ====================

cdef inline void heap64_init(BinaryHeap64* heap, int capacity) noexcept nogil:
    """Initialize BinaryHeap64 with pre-allocated capacity."""
    heap.nodes = <PQNode64*>malloc(capacity * sizeof(PQNode64))
    heap.size = 0
    heap.capacity = capacity


cdef inline bint heap64_empty(BinaryHeap64* heap) noexcept nogil:
    """Check if heap is empty."""
    return heap.size == 0


cdef inline PQNode64 heap64_top(BinaryHeap64* heap) noexcept nogil:
    """Return the minimum priority node without removing it."""
    return heap.nodes[0]


cdef inline void heap64_push(BinaryHeap64* heap, uint64_t index, double priority) noexcept nogil:
    """Insert a node into the heap, growing if needed."""
    cdef int pos
    cdef int parent
    cdef PQNode64 temp

    if heap.size >= heap.capacity:
        heap.capacity = heap.capacity * 2
        heap.nodes = <PQNode64*>realloc(heap.nodes, heap.capacity * sizeof(PQNode64))

    pos = heap.size
    heap.nodes[pos].index = index
    heap.nodes[pos].priority = priority
    heap.size += 1

    while pos > 0:
        parent = (pos - 1) // 2
        if heap.nodes[parent].priority <= heap.nodes[pos].priority:
            break
        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[parent]
        heap.nodes[parent] = temp
        pos = parent


cdef inline void heap64_pop(BinaryHeap64* heap) noexcept nogil:
    """Remove the minimum priority node from the heap."""
    cdef int pos
    cdef int left, right, smallest
    cdef PQNode64 temp

    if heap.size == 0:
        return

    heap.size -= 1
    if heap.size > 0:
        heap.nodes[0] = heap.nodes[heap.size]

        pos = 0
        while True:
            left = 2 * pos + 1
            right = 2 * pos + 2
            smallest = pos

            if left < heap.size and heap.nodes[left].priority < heap.nodes[smallest].priority:
                smallest = left
            if right < heap.size and heap.nodes[right].priority < heap.nodes[smallest].priority:
                smallest = right

            if smallest == pos:
                break

            temp = heap.nodes[pos]
            heap.nodes[pos] = heap.nodes[smallest]
            heap.nodes[smallest] = temp
            pos = smallest


cdef inline void heap64_free(BinaryHeap64* heap) noexcept nogil:
    """Free heap memory."""
    if heap.nodes != NULL:
        c_free(heap.nodes)
        heap.nodes = NULL
        heap.size = 0
        heap.capacity = 0


# ==================== PYTHON WRAPPERS ====================

cdef class PyBinaryHeap64:
    """Python-accessible wrapper for testing the uint64-indexed binary heap."""
    cdef BinaryHeap64 _heap

    def __cinit__(self):
        heap64_init(&self._heap, 1024)

    def __dealloc__(self):
        heap64_free(&self._heap)

    def push(self, uint64_t index, double priority):
        heap64_push(&self._heap, index, priority)

    def top(self):
        cdef PQNode64 node = heap64_top(&self._heap)
        return (int(node.index), float(node.priority))

    def pop(self):
        heap64_pop(&self._heap)

    def empty(self):
        return heap64_empty(&self._heap) != 0


cdef class PyBinaryHeap32:
    """Python-accessible wrapper for testing the uint32-indexed binary heap."""
    cdef BinaryHeap _heap

    def __cinit__(self):
        heap_init(&self._heap)

    def push(self, uint32_t index, double priority):
        heap_push(&self._heap, index, priority)

    def top(self):
        cdef PQNode node = heap_top(&self._heap)
        return (int(node.index), float(node.priority))

    def pop(self):
        heap_pop(&self._heap)

    def empty(self):
        return heap_empty(&self._heap)

    def size(self):
        return self._heap.nodes.size()


# ==================== PYTHON TEST WRAPPERS ====================

def py_ravel_index(int row, int col, int cols):
    """Python wrapper for ravel_index for testing."""
    return int(ravel_index(row, col, cols))


def py_unravel_index(uint32_t idx, int cols):
    """Python wrapper for unravel_index for testing."""
    cdef npy_intp row, col
    unravel_index(idx, cols, &row, &col)
    return (int(row), int(col))


def py_round_up_power_of_two(size_t n):
    """Python wrapper for round_up_power_of_two for testing."""
    return int(round_up_power_of_two(n))


def py_get_circular_index(size_t logical_bucket, size_t buffer_size):
    """Python wrapper for get_circular_index for testing."""
    return int(get_circular_index(logical_bucket, buffer_size))

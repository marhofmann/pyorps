# path_algorithms.pxd

# Import core data structures and utilities from path_core
from pyorps.utils.path_core cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
    StepData, CachedStepData, BinaryHeap, SystemLimits,
    heap_init, heap_empty, heap_top, heap_push, heap_pop,
    ravel_index, unravel_index, check_path, check_path_cached,
    precompute_directions, precompute_cached_steps,
    precompute_directions_optimized, get_system_limits,
    calculate_initial_bucket_size, calculate_thread_buffer_capacity,
    INF_F32
)

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from openmp cimport omp_lock_t

# Thread-local results structure for parallel delta-stepping
cdef struct ThreadResults:
    uint64_t *vertices
    uint32_t *bucket_indices
    float *distances
    int count
    int capacity

# Internal Dijkstra implementation for reuse across different algorithms
cdef np.ndarray[uint32_t, ndim=1] _dijkstra_2d_cython_internal(
    uint16_t[:, :] raster, uint8_t[:, :] exclude_mask,
    vector[StepData] directions, uint32_t source_idx,
    uint32_t target_idx, int rows, int cols
)

# Dynamic bucket management for delta-stepping
cdef void ensure_bucket_size_dynamic(
    vector[vector[uint64_t]]& buckets, size_t bidx,
    SystemLimits* limits
) noexcept nogil

# Edge relaxation for delta-stepping (correct function name)
cdef void relax_edges_delta_stepping(
    vector[uint64_t]& vertices,
    float* dist,
    int64_t* pred,
    const uint16_t[:, :] raster,
    const uint8_t[:, :] exclude_mask,
    const vector[StepData]& directions,
    const vector[CachedStepData]& cached_steps,
    int rows,
    uint64_t cols,
    float delta,
    bint light_phase_only,
    ThreadResults* thread_results,
    int num_threads,
    omp_lock_t* hash_locks,
    int num_hash_locks,
    uint64_t total_cells,
    uint64_t target_idx,
    SystemLimits* limits
) noexcept nogil

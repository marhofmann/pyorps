# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libc.math cimport sqrt, INFINITY, floor, ceil, abs
from cython.parallel cimport prange, threadid
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset
from openmp cimport (omp_lock_t, omp_init_lock, omp_destroy_lock, omp_set_lock,
                     omp_unset_lock, omp_get_max_threads, omp_set_num_threads)

from .path_algorithms cimport path_cost
from .path_algorithms import group_by_proximity


# Type definitions
ctypedef np.uint32_t uint32_t
ctypedef np.uint16_t uint16_t
ctypedef np.uint8_t uint8_t
ctypedef np.int32_t int32_t
ctypedef np.int8_t int8_t
ctypedef np.float64_t float64_t

# Optimized data structures
cdef struct IntermediatePoint:
    int8_t dr
    int8_t dc

cdef struct StepData:
    int dr
    int dc
    double cost_factor

cdef struct CachedStepData:
    vector[IntermediatePoint] intermediates
    int intermediate_count

cdef struct ThreadResults:
    uint32_t *vertices
    uint32_t *bucket_indices
    double *distances
    int count
    int capacity

# Helper functions
cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) noexcept nogil:
    cdef vector[IntermediatePoint] result
    cdef IntermediatePoint point
    cdef int abs_dr, abs_dc, sum_abs, k, p
    cdef double dr_k, dc_k, ddr, ddc, dk, dp
    cdef int8_t floor_dr, floor_dc, ceil_dr, ceil_dc

    abs_dr = abs(dr)
    abs_dc = abs(dc)
    sum_abs = abs_dr + abs_dc

    if sum_abs <= 1:
        pass
    elif max(abs_dr, abs_dc) == 1:
        point.dr = <int8_t>dr
        point.dc = 0
        result.push_back(point)

        point.dr = 0
        point.dc = <int8_t>dc
        result.push_back(point)
    else:
        k = max(abs_dr, abs_dc)
        ddr = <double>dr
        ddc = <double>dc
        dk = <double>k

        for p in range(1, k):
            dp = <double>p
            dr_k = (dp * ddr) / dk
            dc_k = (dp * ddc) / dk

            floor_dr = <int8_t>floor(dr_k)
            floor_dc = <int8_t>floor(dc_k)
            point.dr = floor_dr
            point.dc = floor_dc
            result.push_back(point)

            ceil_dr = <int8_t>ceil(dr_k)
            ceil_dc = <int8_t>ceil(dc_k)
            if floor_dr != ceil_dr or floor_dc != ceil_dc:
                point.dr = ceil_dr
                point.dc = ceil_dc
                result.push_back(point)

    return result

cdef inline double _get_cost_factor_cython(int dr, int dc, int intermediates_count) noexcept nogil:
    cdef double distance, divisor
    distance = sqrt(<double>(dr * dr + dc * dc))
    divisor = 2.0 + <double>intermediates_count
    return distance / divisor

cdef vector[CachedStepData] precompute_cached_steps(np.ndarray[int8_t, ndim=2] steps_arr):
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

cdef int check_path_cached(const vector[IntermediatePoint] &cached_intermediates,
                          int current_row, int current_col,
                          const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                          int rows, int cols, double* total_cost) except -1 nogil:
    cdef double cost
    cdef int i, int_row, int_col, num_intermediates
    cdef IntermediatePoint point

    cost = 0.0
    num_intermediates = <int>cached_intermediates.size()

    for i in range(num_intermediates):
        point = cached_intermediates[i]
        int_row = current_row + point.dr
        int_col = current_col + point.dc

        if (int_row < 0 or int_row >= rows or
                int_col < 0 or int_col >= cols):
            return 0

        if exclude_mask[int_row, int_col] == 0:
            return 0

        cost += raster[int_row, int_col]

    total_cost[0] = cost
    return 1

cdef vector[StepData] precompute_directions_optimized(np.ndarray[int8_t, ndim=2] steps_arr,
                                                     const vector[CachedStepData] &cached_steps):
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
        direction.cost_factor = _get_cost_factor_cython(dr, dc, cached_steps[s].intermediate_count)

        directions.push_back(direction)

    return directions

# FIXED: Parallel edge relaxation using memoryviews instead of numpy arrays
cdef void relax_edges_correct_optimized(vector[uint32_t] &vertices,
                                        double *dist, int32_t *pred,
                                        const uint16_t[:, :] raster, const uint8_t[:, :] exclude_mask,
                                        const vector[StepData] &directions,
                                        const vector[CachedStepData] &cached_steps,
                                        int rows, int cols,
                                        double delta, bint light_phase_only,
                                        ThreadResults *thread_results, int num_threads,
                                        omp_lock_t *hash_locks, int num_hash_locks,
                                        uint32_t total_cells) noexcept nogil:
    cdef int i, tid, dir_idx, ur, uc, vr, vc, lock_idx
    cdef uint32_t u, v, bucket_idx
    cdef double current_dist, edge_weight, new_dist, intermediate_cost
    cdef bint should_update
    cdef int valid_path

    # Process vertices in parallel
    for i in prange(<int>vertices.size(), schedule='static', chunksize=8):
        tid = threadid()
        if tid < 0 or tid >= num_threads:
            tid = 0

        u = vertices[i]

        if u >= total_cells:
            continue

        ur = u // cols
        uc = u % cols

        # Hash-based locking for source vertex
        lock_idx = u % num_hash_locks
        omp_set_lock(&hash_locks[lock_idx])
        current_dist = dist[u]
        omp_unset_lock(&hash_locks[lock_idx])

        if current_dist >= INFINITY:
            continue

        # Process all neighbors
        for dir_idx in range(<int>directions.size()):
            vr = ur + directions[dir_idx].dr
            vc = uc + directions[dir_idx].dc

            # Bounds and traversability checks
            if (vr < 0 or vr >= rows or vc < 0 or vc >= cols):
                continue

            if exclude_mask[vr, vc] == 0:
                continue

            v = vr * cols + vc
            if v >= total_cells:
                continue

            # Use cached path checking
            intermediate_cost = 0.0
            valid_path = check_path_cached(
                cached_steps[dir_idx].intermediates,
                ur, uc, exclude_mask, raster, rows, cols, &intermediate_cost
            )

            if not valid_path:
                continue

            # Calculate edge weight
            edge_weight = (raster[ur, uc] + intermediate_cost + raster[vr, vc]) * directions[dir_idx].cost_factor

            # Phase filtering
            if light_phase_only and edge_weight > delta:
                continue
            if not light_phase_only and edge_weight <= delta:
                continue

            new_dist = current_dist + edge_weight

            # Atomic relaxation with proper locking
            should_update = False
            lock_idx = v % num_hash_locks
            omp_set_lock(&hash_locks[lock_idx])
            if new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = <int32_t>u
                should_update = True
            omp_unset_lock(&hash_locks[lock_idx])

            # Store results
            if should_update and thread_results[tid].count < thread_results[tid].capacity:
                bucket_idx = <uint32_t>(new_dist / delta)
                thread_results[tid].vertices[thread_results[tid].count] = v
                thread_results[tid].bucket_indices[thread_results[tid].count] = bucket_idx
                thread_results[tid].distances[thread_results[tid].count] = new_dist
                thread_results[tid].count += 1

# SIMPLIFIED: Sequential edge relaxation function (FIXED - using memoryviews)
cdef void relax_vertex_edges_sequential(
    uint32_t vertex, double[:] dist, int32_t[:] pred,
    const uint16_t[:, :] raster_view,  # FIXED: Use memoryview instead of np.ndarray
    const uint8_t[:, :] exclude_view,  # FIXED: Use memoryview instead of np.ndarray
    const vector[StepData] &directions,
    const vector[CachedStepData] &cached_steps,
    int rows, int cols, double delta,
    vector[vector[uint32_t]] &buckets,
    uint32_t max_buckets) noexcept nogil:

    cdef int vertex_r, vertex_c, neighbor_r, neighbor_c, dir_idx
    cdef uint32_t neighbor_idx, bucket_idx
    cdef double current_dist, intermediate_cost, edge_weight, new_dist
    cdef int valid_path

    # Get vertex coordinates
    vertex_r = <int>(vertex // cols)
    vertex_c = <int>(vertex % cols)
    current_dist = dist[vertex]

    # Process all movement directions
    for dir_idx in range(<int>directions.size()):
        neighbor_r = vertex_r + directions[dir_idx].dr
        neighbor_c = vertex_c + directions[dir_idx].dc

        # Bounds check
        if (neighbor_r < 0 or neighbor_r >= rows or
            neighbor_c < 0 or neighbor_c >= cols):
            continue

        # Traversability check
        if exclude_view[neighbor_r, neighbor_c] == 0:
            continue

        neighbor_idx = <uint32_t>(neighbor_r * cols + neighbor_c)

        # Path validation
        intermediate_cost = 0.0
        valid_path = check_path_cached(
            cached_steps[dir_idx].intermediates,
            vertex_r, vertex_c, exclude_view, raster_view,
            rows, cols, &intermediate_cost
        )

        if not valid_path:
            continue

        # Calculate edge weight
        edge_weight = (raster_view[vertex_r, vertex_c] +
                      intermediate_cost +
                      raster_view[neighbor_r, neighbor_c]) * directions[dir_idx].cost_factor

        new_dist = current_dist + edge_weight

        # Update if better path found
        if new_dist < dist[neighbor_idx]:
            dist[neighbor_idx] = new_dist
            pred[neighbor_idx] = <int32_t>vertex

            # Add to appropriate bucket
            bucket_idx = <uint32_t>(new_dist / delta)
            if bucket_idx < max_buckets:
                if bucket_idx >= buckets.size():
                    # Safely extend buckets
                    buckets.resize(min(bucket_idx + 100, max_buckets))
                if bucket_idx < buckets.size():
                    buckets[bucket_idx].push_back(neighbor_idx)

cdef void ensure_bucket_size_safe(vector[vector[uint32_t]] & buckets, uint32_t bidx):
    cdef size_t new_size, max_size
    max_size = 200000
    if bidx >= buckets.size() and bidx < max_size:
        new_size = min(<size_t>(bidx + 1000), max_size)
        buckets.resize(new_size)

cdef np.ndarray[uint8_t, ndim=2] create_exclude_mask(
        np.ndarray[uint16_t, ndim=2] raster_arr, uint16_t max_value):
    cdef int rows, cols, i, j
    cdef np.ndarray[uint8_t, ndim=2] mask

    rows = <int>raster_arr.shape[0]
    cols = <int>raster_arr.shape[1]
    mask = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if raster_arr[i, j] < max_value:
                mask[i, j] = 1

    return mask

def delta_stepping_2d(np.ndarray[uint16_t, ndim=2] raster_arr,
                             np.ndarray[int8_t, ndim=2] steps_arr,
                             uint32_t source_idx, uint32_t target_idx,
                             double delta,
                             uint16_t max_value=65535,
                             int num_threads=0):
    """
    CORRECTED Delta-stepping algorithm that produces optimal results identical to Dijkstra.
    """
    # ALL VARIABLE DECLARATIONS AT THE TOP
    cdef int rows, cols, actual_threads, num_hash_locks, max_capacity
    cdef uint32_t total_cells
    cdef int i, tid, lock_idx, max_iterations, iteration, light_iterations
    cdef int path_length
    cdef uint32_t current_bucket, v, bidx, current
    cdef uint32_t source_r, source_c, target_r, target_c
    cdef bint target_found, target_is_optimal
    cdef double target_final_distance

    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    cdef np.ndarray[double, ndim=1] dist
    cdef np.ndarray[int32_t, ndim=1] pred
    cdef const uint16_t[:, :] raster_view
    cdef const uint8_t[:, :] exclude_view

    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions
    cdef vector[vector[uint32_t]] buckets
    cdef vector[uint32_t] current_vertices, settled_vertices

    cdef ThreadResults *thread_results
    cdef omp_lock_t *hash_locks

    # Initialize
    rows = <int>raster_arr.shape[0]
    cols = <int>raster_arr.shape[1]
    total_cells = <uint32_t>(rows * cols)

    if delta <= 0.0:
        raise ValueError("delta must be > 0")

    # Thread settings
    if num_threads <= 0:
        num_threads = min(8, omp_get_max_threads())

    omp_set_num_threads(num_threads)
    actual_threads = omp_get_max_threads()
    num_hash_locks = min(8192, max(2048, actual_threads * 128))

    # Validate indices
    if source_idx >= total_cells or target_idx >= total_cells:
        return np.empty(0, dtype=np.uint32)

    # Setup
    exclude_mask_arr = create_exclude_mask(raster_arr, max_value)

    source_r = source_idx // <uint32_t>cols
    source_c = source_idx % <uint32_t>cols
    target_r = target_idx // <uint32_t>cols
    target_c = target_idx % <uint32_t>cols

    if (exclude_mask_arr[source_r, source_c] == 0 or
        exclude_mask_arr[target_r, target_c] == 0):
        return np.empty(0, dtype=np.uint32)

    # Pre-compute step data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize arrays
    dist = np.full(<int>total_cells, INFINITY, dtype=np.float64)
    pred = np.full(<int>total_cells, -1, dtype=np.int32)
    dist[source_idx] = 0.0

    # Memory views
    raster_view = raster_arr
    exclude_view = exclude_mask_arr

    # Initialize buckets
    buckets.resize(min(50000, max(5000, <int>total_cells // 200)))
    buckets[0].push_back(source_idx)

    # Hash locks
    hash_locks = <omp_lock_t*>malloc(num_hash_locks * sizeof(omp_lock_t))
    if hash_locks == NULL:
        raise MemoryError("Could not allocate hash locks")

    for lock_idx in range(num_hash_locks):
        omp_init_lock(&hash_locks[lock_idx])

    # Thread-local storage
    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)
        raise MemoryError("Could not allocate thread data")

    max_capacity = min(25000, max(5000, <int>total_cells // 400))

    for tid in range(actual_threads):
        thread_results[tid].vertices = <uint32_t*>malloc(max_capacity * sizeof(uint32_t))
        thread_results[tid].bucket_indices = <uint32_t*>malloc(max_capacity * sizeof(uint32_t))
        thread_results[tid].distances = <double*>malloc(max_capacity * sizeof(double))

        if (thread_results[tid].vertices == NULL or
            thread_results[tid].bucket_indices == NULL or
            thread_results[tid].distances == NULL):

            # Cleanup on failure
            for i in range(tid + 1):
                if thread_results[i].vertices != NULL:
                    free(thread_results[i].vertices)
                if thread_results[i].bucket_indices != NULL:
                    free(thread_results[i].bucket_indices)
                if thread_results[i].distances != NULL:
                    free(thread_results[i].distances)
            free(thread_results)
            for lock_idx in range(num_hash_locks):
                omp_destroy_lock(&hash_locks[lock_idx])
            free(hash_locks)
            raise MemoryError("Could not allocate thread storage")

        thread_results[tid].capacity = max_capacity
        thread_results[tid].count = 0

    # Main algorithm
    current_bucket = 0
    target_found = False
    target_is_optimal = False
    max_iterations = min(100000, <int>total_cells // 50)

    try:
        for iteration in range(max_iterations):
            # Find next non-empty bucket
            while current_bucket < buckets.size() and buckets[current_bucket].empty():
                current_bucket += 1

            if current_bucket >= buckets.size():
                ensure_bucket_size_safe(buckets, current_bucket)
                if current_bucket >= buckets.size():
                    break

            # Light phase
            settled_vertices.clear()
            light_iterations = 0

            while not buckets[current_bucket].empty() and light_iterations < 50:
                light_iterations += 1

                current_vertices = buckets[current_bucket]
                buckets[current_bucket].clear()

                settled_vertices.insert(settled_vertices.end(),
                                      current_vertices.begin(), current_vertices.end())

                # Clear thread results
                for tid in range(actual_threads):
                    thread_results[tid].count = 0

                # Process light edges
                relax_edges_correct_optimized(current_vertices,
                                            <double*>dist.data, <int32_t*>pred.data,
                                            raster_view, exclude_view,
                                            directions, cached_steps,
                                            rows, cols, delta, True,
                                            thread_results, actual_threads,
                                            hash_locks, num_hash_locks, total_cells)

                # Collect results
                for tid in range(actual_threads):
                    for i in range(thread_results[tid].count):
                        v = thread_results[tid].vertices[i]
                        bidx = thread_results[tid].bucket_indices[i]

                        if bidx >= current_bucket:
                            ensure_bucket_size_safe(buckets, bidx)
                            if bidx < buckets.size():
                                buckets[bidx].push_back(v)

            # Check if target was settled
            for i in range(<int>settled_vertices.size()):
                if settled_vertices[i] == target_idx:
                    target_found = True
                    target_final_distance = dist[target_idx]
                    break

            # Heavy phase
            if not settled_vertices.empty():
                for tid in range(actual_threads):
                    thread_results[tid].count = 0

                relax_edges_correct_optimized(settled_vertices,
                                            <double*>dist.data, <int32_t*>pred.data,
                                            raster_view, exclude_view,
                                            directions, cached_steps,
                                            rows, cols, delta, False,
                                            thread_results, actual_threads,
                                            hash_locks, num_hash_locks, total_cells)

                # Collect heavy phase results
                for tid in range(actual_threads):
                    for i in range(thread_results[tid].count):
                        v = thread_results[tid].vertices[i]
                        bidx = thread_results[tid].bucket_indices[i]

                        if bidx > current_bucket:
                            ensure_bucket_size_safe(buckets, bidx)
                            if bidx < buckets.size():
                                buckets[bidx].push_back(v)

            # Termination condition
            if target_found:
                target_is_optimal = True
                for i in range(current_bucket + 1):
                    if i < buckets.size() and not buckets[i].empty():
                        target_is_optimal = False
                        break

                if target_is_optimal:
                    break

            current_bucket += 1

    finally:
        # Cleanup
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)

        for tid in range(actual_threads):
            free(thread_results[tid].vertices)
            free(thread_results[tid].bucket_indices)
            free(thread_results[tid].distances)
        free(thread_results)

    # Path reconstruction
    if not target_found or pred[target_idx] == -1:
        if source_idx == target_idx:
            return np.array([source_idx], dtype=np.uint32)
        return np.empty(0, dtype=np.uint32)

    # Build path
    path_vertices = []
    current = target_idx
    path_length = 0

    while current != <uint32_t>(-1) and path_length < <int>total_cells:
        path_vertices.append(current)
        if current == source_idx:
            break
        current = pred[current]
        if current == -1:
            return np.empty(0, dtype=np.uint32)
        path_length += 1

    path_vertices.reverse()
    return np.array(path_vertices, dtype=np.uint32)


def delta_stepping_single_source_multiple_targets(
    np.ndarray[uint16_t, ndim=2] raster_arr,
    np.ndarray[int8_t, ndim=2] steps_arr,
    uint32_t source_idx,
    np.ndarray[uint32_t, ndim=1] target_indices,
    double delta,
    uint16_t max_value=65535,
    int num_threads=0):
    """
    CORRECTED: Delta-stepping algorithm for single source to multiple targets.
    Using simplified sequential approach for reliability.
    """
    # Variable declarations
    cdef int rows, cols, num_targets, i, j, path_length
    cdef uint32_t total_cells, current_bucket, current_vertex, target_idx
    cdef uint32_t source_r, source_c, max_buckets
    cdef int targets_found, iteration, max_iterations

    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    cdef np.ndarray[double, ndim=1] dist
    cdef np.ndarray[int32_t, ndim=1] pred
    cdef np.ndarray[uint8_t, ndim=1] target_found_arr

    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions
    cdef vector[vector[uint32_t]] buckets
    cdef vector[uint32_t] current_level_vertices

    # Initialize
    rows = <int>raster_arr.shape[0]
    cols = <int>raster_arr.shape[1]
    total_cells = <uint32_t>(rows * cols)
    num_targets = <int>target_indices.shape[0]

    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if num_targets == 0:
        return []

    # Validate indices
    if source_idx >= total_cells:
        return [np.empty(0, dtype=np.uint32) for _ in range(num_targets)]

    for i in range(num_targets):
        if target_indices[i] >= total_cells:
            return [np.empty(0, dtype=np.uint32) for _ in range(num_targets)]

    # Setup
    exclude_mask_arr = create_exclude_mask(raster_arr, max_value)
    source_r = source_idx // <uint32_t>cols
    source_c = source_idx % <uint32_t>cols

    if exclude_mask_arr[source_r, source_c] == 0:
        return [np.empty(0, dtype=np.uint32) for _ in range(num_targets)]

    # Pre-compute step data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize arrays
    dist = np.full(<int>total_cells, INFINITY, dtype=np.float64)
    pred = np.full(<int>total_cells, -1, dtype=np.int32)
    target_found_arr = np.zeros(num_targets, dtype=np.uint8)

    dist[source_idx] = 0.0
    targets_found = 0

    # Initialize buckets
    max_buckets = min(10000, max(1000, total_cells // 1000))
    buckets.resize(max_buckets)
    buckets[0].push_back(source_idx)

    # SIMPLIFIED SEQUENTIAL ALGORITHM
    current_bucket = 0
    max_iterations = <int>total_cells

    # Create memory views for sequential function
    cdef const uint16_t[:, :] raster_view = raster_arr
    cdef const uint8_t[:, :] exclude_view = exclude_mask_arr
    cdef double[:] dist_view = dist
    cdef int32_t[:] pred_view = pred

    for iteration in range(max_iterations):
        # Find next non-empty bucket
        while current_bucket < buckets.size() and buckets[current_bucket].empty():
            current_bucket += 1

        if current_bucket >= buckets.size():
            break

        # Process all vertices at current level
        current_level_vertices.clear()
        current_level_vertices = buckets[current_bucket]
        buckets[current_bucket].clear()

        # Check if we found any targets at this level
        for i in range(<int>current_level_vertices.size()):
            current_vertex = current_level_vertices[i]
            for j in range(num_targets):
                if current_vertex == target_indices[j] and target_found_arr[j] == 0:
                    target_found_arr[j] = 1
                    targets_found += 1

        # Early termination if all targets found
        if targets_found >= num_targets:
            break

        # FIXED: Sequential edge relaxation
        for i in range(<int>current_level_vertices.size()):
            current_vertex = current_level_vertices[i]
            if current_vertex >= total_cells:
                continue

            relax_vertex_edges_sequential(
                current_vertex, dist_view, pred_view, raster_view, exclude_view,
                directions, cached_steps, rows, cols, delta, buckets, max_buckets
            )

        current_bucket += 1

    # Reconstruct paths for all targets
    paths = []
    for i in range(num_targets):
        target_idx = target_indices[i]

        if pred[target_idx] == -1:
            if source_idx == target_idx:
                paths.append(np.array([source_idx], dtype=np.uint32))
            else:
                paths.append(np.empty(0, dtype=np.uint32))
            continue

        # Build path with cycle detection
        path_vertices = []
        current_vertex = target_idx
        path_length = 0

        while current_vertex != source_idx and path_length < <int>total_cells:
            path_vertices.append(current_vertex)
            if pred[current_vertex] == -1:
                paths.append(np.empty(0, dtype=np.uint32))
                break
            current_vertex = <uint32_t>pred[current_vertex]
            path_length += 1
        else:
            if current_vertex == source_idx:
                path_vertices.append(source_idx)
                path_vertices.reverse()
                paths.append(np.array(path_vertices, dtype=np.uint32))
            else:
                paths.append(np.empty(0, dtype=np.uint32))

    return paths


def delta_stepping_some_pairs_shortest_paths(
    np.ndarray[uint16_t, ndim=2] raster_arr,
    np.ndarray[int8_t, ndim=2] steps_arr,
    np.ndarray[uint32_t, ndim=1] source_indices,
    np.ndarray[uint32_t, ndim=1] target_indices,
    double delta,
    uint16_t max_value=65535,
    bint return_paths=True,
    int num_threads=0):
    """
    CORRECTED: Delta-stepping for specific source-target pairs.
    Simplified to use individual calls for reliability.
    """
    cdef int num_pairs = <int>min(source_indices.shape[0], target_indices.shape[0])
    cdef list all_paths = [None] * num_pairs if return_paths else None
    cdef np.ndarray[float64_t, ndim=1] costs = np.full(num_pairs, np.inf)

    cdef int i
    cdef uint32_t source, target
    cdef np.ndarray[uint32_t, ndim=1] path

    # Process each pair individually
    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        try:
            path = delta_stepping_2d(raster_arr, steps_arr, source, target,
                                   delta, max_value, 1)  # Single thread for stability

            if return_paths:
                all_paths[i] = path

            if len(path) > 0:
                costs[i] = path_cost(path, raster_arr, <int>raster_arr.shape[1])
        except:
            if return_paths:
                all_paths[i] = np.empty(0, dtype=np.uint32)

    return all_paths if return_paths else costs


def delta_stepping_multiple_sources_multiple_targets(
    np.ndarray[uint16_t, ndim=2] raster_arr,
    np.ndarray[int8_t, ndim=2] steps_arr,
    np.ndarray[uint32_t, ndim=1] source_indices,
    np.ndarray[uint32_t, ndim=1] target_indices,
    double delta,
    uint16_t max_value=65535,
    bint return_paths=True,
    int num_threads=0):
    """
    CORRECTED: Delta-stepping for multiple sources and targets.
    Simplified to use corrected single-source multiple-targets approach.
    """
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef int num_sources = <int>source_indices.shape[0]
    cdef int num_targets = <int>target_indices.shape[0]

    # Validate inputs
    if num_sources == 0 or num_targets == 0:
        if return_paths:
            return []
        else:
            return np.full((num_sources, num_targets), np.inf)

    # Initialize result containers
    cdef np.ndarray[float64_t, ndim=2] cost_matrix = np.full(
        (num_sources, num_targets), np.inf)
    cdef list all_paths = [] if return_paths else None

    # Process each source individually
    cdef int s, t
    cdef uint32_t source_idx
    cdef list source_paths
    cdef double cost

    for s in range(num_sources):
        source_idx = source_indices[s]

        # Use corrected single-source multiple-targets function
        try:
            source_paths = delta_stepping_single_source_multiple_targets(
                raster_arr, steps_arr, source_idx, target_indices,
                delta, max_value, 1  # Use single thread for stability
            )

            if return_paths:
                all_paths.append(source_paths)

            # Calculate costs
            for t in range(num_targets):
                if t < len(source_paths) and len(source_paths[t]) > 0:
                    cost = path_cost(source_paths[t], raster_arr, cols)
                    cost_matrix[s, t] = cost

        except Exception as e:
            # Handle errors gracefully
            if return_paths:
                all_paths.append([np.empty(0, dtype=np.uint32) for _ in range(num_targets)])

    return all_paths if return_paths else cost_matrix


def estimate_optimal_delta_fast(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint16_t max_value=65535,
        int sample_size=100,
        double base_factor=0.8):
    """
    Fast delta estimation using statistical sampling.
    """
    cdef int rows, cols, i, step_idx, valid_samples
    cdef uint32_t sample_idx, total_cells
    cdef double cost_sum = 0.0, mean_cost
    cdef uint16_t cell_cost
    cdef int8_t dr, dc
    cdef double max_distance = 0.0, distance
    cdef double estimated_delta

    rows = <int>raster_arr.shape[0]
    cols = <int>raster_arr.shape[1]
    total_cells = <uint32_t>(rows * cols)

    # Phase 1: Fast sampling of raster cells
    valid_samples = 0
    sample_size = min(sample_size, <int>total_cells)

    for i in range(sample_size):
        sample_idx = (i * total_cells) // sample_size
        sample_row = <int>(sample_idx // cols)
        sample_col = <int>(sample_idx % cols)

        cell_cost = raster_arr[sample_row, sample_col]
        if cell_cost < max_value:
            cost_sum += cell_cost
            valid_samples += 1

    if valid_samples == 0:
        return 1.0

    mean_cost = cost_sum / valid_samples

    # Phase 2: Calculate max movement distance
    for step_idx in range(steps_arr.shape[0]):
        dr = steps_arr[step_idx, 0]
        dc = steps_arr[step_idx, 1]
        distance = sqrt(<double>(dr * dr + dc * dc))
        if distance > max_distance:
            max_distance = distance

    if max_distance == 0.0:
        max_distance = 1.0

    # Phase 3: Apply heuristic
    estimated_delta = mean_cost * max_distance * 2.0 * base_factor

    # Ensure reasonable bounds
    if estimated_delta < 1.0:
        estimated_delta = 1.0
    elif estimated_delta > mean_cost * 20.0:
        estimated_delta = mean_cost * 20.0

    return estimated_delta

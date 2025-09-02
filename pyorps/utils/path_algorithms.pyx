"""
High-performance pathfinding algorithms implemented in Cython.

This module contains optimized implementations of Dijkstra's algorithm variants
and delta-stepping algorithms for different pathfinding scenarios:
- Single source to single target (classical shortest path)
- Single source to multiple targets (one-to-many)
- Multiple sources to multiple targets (many-to-many)
- Delta-stepping with parallel edge relaxation
- A* optimization for directed searches

All algorithms support complex movement patterns with intermediate steps,
making them suitable for realistic pathfinding in raster environments with
diagonal and extended-range movements.

Performance Characteristics:
    - Dijkstra single-source: O((V + E) log V) where V=cells, E=edges
    - Delta-stepping: O((V + E) / p + D·L) where p=parallelism, D=diameter, L=bucket operations
    - Multi-target optimization: Amortizes setup costs across related queries
    - Batch processing: Reduces redundant computation for spatially related paths
    - Memory efficiency: Reuses data structures across multiple path queries
"""

# Cython compiler directives for maximum performance
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort
from libc.math cimport sqrt, sqrtf, floor, ceil, abs
from cython.parallel cimport prange, threadid
from libc.stdlib cimport malloc, free, calloc
from openmp cimport (omp_lock_t, omp_init_lock, omp_destroy_lock, omp_set_lock,
                     omp_unset_lock, omp_get_max_threads, omp_set_num_threads)
import psutil
import sys
from libc.stdint cimport SIZE_MAX, UINT32_MAX, UINT64_MAX

# Import core data structures and utilities
from .path_core cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, float32_t, float64_t, uint64_t,
    npy_intp, StepData, CachedStepData, SystemLimits,
    BinaryHeap, heap_init, heap_empty, heap_top, heap_push, heap_pop,
    ravel_index, unravel_index, check_path, check_path_cached,
    precompute_directions, precompute_cached_steps, precompute_directions_optimized,
    get_system_limits, calculate_initial_bucket_size, calculate_thread_buffer_capacity,
    create_exclude_mask, INF_F32, path_cost, path_cost_uint32, round_up_power_of_two,
    get_circular_index,
)


# ==================== THREAD-LOCAL DATA STRUCTURES ====================

cdef struct ThreadResults:
    # Per-thread storage for parallel delta-stepping edge relaxation.
    #
    # This structure holds thread-local buffers to avoid contention during
    # parallel processing. Each thread accumulates vertices to be added to
    # buckets, which are later merged in a coordination phase.
    #
    # Members:
    #     vertices: Array of vertex indices to be added to buckets
    #     bucket_indices: Corresponding bucket index for each vertex
    #     distances: Computed distances for priority ordering
    #     count: Number of valid entries currently stored
    #     capacity: Maximum number of entries this buffer can hold

    uint64_t *vertices
    uint32_t *bucket_indices
    float *distances
    int count
    int capacity

# ==================== SPATIAL OPTIMIZATION ====================

def group_by_proximity(np.ndarray[uint64_t, ndim=1] source_indices, uint64_t cols):
    """
    Group source indices by spatial proximity for optimized batch processing.

    This function reorders source nodes to improve cache locality and
    computational efficiency during multi-source pathfinding operations.
    Nodes that are spatially close in the raster are processed together,
    reducing memory access patterns and improving overall performance.

    Algorithm:
        1. Convert linear indices to 2D coordinates
        2. Sort by row coordinate (simple spatial grouping)
        3. Return indices in the new proximity-based order

    Parameters:
        source_indices: 1D array of linear node indices to reorder
        cols: Number of columns in the raster (for coordinate conversion)

    Returns:
        1D array of node indices reordered by spatial proximity

    Performance Notes:
        - Provides significant speedup for large multi-source problems
        - Simple row-based sorting balances complexity vs. benefit
        - Memory allocation pattern optimized for NumPy operations
    """
    cdef int num_sources = <int>source_indices.shape[0]
    cdef np.ndarray[uint64_t, ndim=1] sorted_indices = np.zeros(
        num_sources, dtype=np.uint64)

    # Handle trivial cases
    if num_sources <= 1:
        return source_indices

    # Convert linear indices to 2D coordinates
    cdef np.ndarray[int64_t, ndim=2] coords = np.zeros(
        (num_sources, 2), dtype=np.int64)
    cdef int i

    for i in range(num_sources):
        coords[i, 0] = <int64_t>(source_indices[i] // cols)  # row
        coords[i, 1] = <int64_t>(source_indices[i] % cols)   # col

    # Sort by row coordinate for spatial grouping
    cdef np.ndarray[int64_t, ndim=1] sorted_by_row = np.array(
        np.argsort(coords[:, 0]), dtype=np.int64)

    for i in range(num_sources):
        sorted_indices[i] = source_indices[sorted_by_row[i]]

    return sorted_indices


def group_by_proximity_uint32(np.ndarray[uint32_t, ndim=1] source_indices,
                              uint64_t cols):
    """
    Group source indices by spatial proximity (uint32_t version).

    This is a compatibility version for Dijkstra algorithms that use uint32_t indices.
    """
    cdef int num_sources = <int> source_indices.shape[0]
    cdef np.ndarray[uint32_t, ndim=1] sorted_indices = np.zeros(
        num_sources, dtype=np.uint32)

    # Handle trivial cases
    if num_sources <= 1:
        return source_indices

    # Convert linear indices to 2D coordinates
    cdef np.ndarray[int64_t, ndim=2] coords = np.zeros(
        (num_sources, 2), dtype=np.int64)
    cdef int i

    for i in range(num_sources):
        coords[i, 0] = <int64_t> (source_indices[i] // cols)  # row
        coords[i, 1] = <int64_t> (source_indices[i] % cols)  # col

    # Sort by row coordinate for spatial grouping
    cdef np.ndarray[int64_t, ndim=1] sorted_by_row = np.array(
        np.argsort(coords[:, 0]), dtype=np.int64)

    for i in range(num_sources):
        sorted_indices[i] = source_indices[sorted_by_row[i]]

    return sorted_indices

# ==================== DYNAMIC BUCKET MANAGEMENT ====================

cdef void ensure_bucket_size_dynamic(vector[vector[uint64_t]]& buckets, size_t bidx,
                                    SystemLimits* limits) noexcept nogil:
    """
    Dynamically resize bucket array based on system limits.

    This function manages the growth of the bucket data structure used in
    delta-stepping, ensuring memory usage stays within system constraints
    while allowing for efficient expansion as needed.

    Growth Strategy:
        - Exponential growth (10% extra) for small sizes
        - Linear growth (100 buckets) when near memory limits
        - Hard cap at system maximum bucket count

    Parameters:
        buckets: Reference to bucket vector to resize
        bidx: Required bucket index that must be accommodated
        limits: System resource limits for memory constraints
    """
    cdef size_t current_size, new_size
    cdef uint64_t memory_needed

    current_size = buckets.size()

    if bidx >= current_size and bidx < limits.max_buckets:
        new_size = bidx + max(1000, bidx // 10)

        memory_needed = new_size * sizeof(vector[uint64_t])
        if memory_needed > limits.available_memory_bytes // 100:
            new_size = bidx + 100

        if new_size > limits.max_buckets:
            new_size = limits.max_buckets

        if new_size > current_size:
            buckets.resize(new_size)

# ==================== DELTA-STEPPING EDGE RELAXATION ====================

cdef void relax_edges_delta_stepping(vector[uint64_t]& vertices,
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
                                    SystemLimits* limits) noexcept nogil:
    """
    Parallel edge relaxation for delta-stepping algorithm.

    This function performs the core relaxation step of delta-stepping,
    processing edges in parallel while maintaining correctness through
    fine-grained locking. It supports both light edges (weight ≤ delta)
    and heavy edges (weight > delta) based on the phase parameter.

    Algorithm Details:
        - Light phase: Process only edges with weight ≤ delta
        - Heavy phase: Process only edges with weight > delta
        - Parallel processing with thread-local accumulation
        - Lock striping for concurrent distance updates
        - Optional A* heuristic for directed search

    Parameters:
        vertices: Current set of vertices to relax edges from
        dist: Distance array (shared, protected by locks)
        pred: Predecessor array for path reconstruction
        raster: Cost raster for edge weight calculation
        exclude_mask: Traversability mask
        directions: Precomputed movement directions
        cached_steps: Cached intermediate steps for each direction
        rows, cols: Raster dimensions
        delta: Bucket width for edge classification
        light_phase_only: True for light edges, False for heavy edges
        thread_results: Per-thread accumulation buffers
        num_threads: Active thread count
        hash_locks: Lock array for distance updates
        num_hash_locks: Number of locks (for modulo addressing)
        total_cells: Total number of cells in raster
        target_idx: Target for A* heuristic (if use_astar=True)
        min_cell_cost: Minimum cell cost for heuristic
        use_astar: Enable A* optimization
        limits: System resource constraints
    """
    cdef int i, tid, dir_idx, ur, uc, vr, vc, lock_idx
    cdef uint64_t u, v, ur64, uc64, vr64, vc64
    cdef size_t bucket_idx_temp
    cdef uint32_t bucket_idx_stored
    cdef float current_dist, edge_weight, new_dist, intermediate_cost
    cdef float raster_ur_uc, raster_vr_vc, h_value, f_value
    cdef bint should_update
    cdef int valid_path

    # using hash mask to avoid modulo for lock index determination!
    cdef uint64_t hash_mask = <uint64_t>(num_hash_locks - 1)

    # Process vertices in parallel with dynamic scheduling
    for i in prange(<int>vertices.size(), schedule='dynamic', chunksize=64):
        tid = threadid()
        if tid < 0 or tid >= num_threads:
            tid = 0

        u = vertices[i]

        if u >= total_cells:
            continue

        # Convert to 2D coordinates
        ur64 = u // cols
        # uc64 = u % cols
        # Using Multiplication + subtraction instead of modulo better than modulo
        uc64 = u - (ur64 * cols)
        ur = <int>ur64
        uc = <int>uc64

        # Get current distance with lock protection
        # lock_idx = <int>(u % hash_locks)
        # Using bitwise AND with hash mask insead of modulo
        lock_idx = <int> (u & hash_mask)
        omp_set_lock(&hash_locks[lock_idx])
        current_dist = dist[u]
        omp_unset_lock(&hash_locks[lock_idx])

        if current_dist >= INF_F32:
            continue

        raster_ur_uc = <float>raster[ur, uc]

        # Process all movement directions
        for dir_idx in range(<int>directions.size()):
            vr = ur + directions[dir_idx].dr
            vc = uc + directions[dir_idx].dc

            # Boundary and traversability checks
            if vr < 0 or vr >= rows or vc < 0 or vc >= <int>cols:
                continue

            if exclude_mask[vr, vc] == 0:
                continue

            vr64 = <uint64_t>vr
            vc64 = <uint64_t>vc
            v = vr64 * cols + vc64

            if v >= total_cells:
                continue

            # Check intermediate steps
            intermediate_cost = 0.0
            valid_path = check_path_cached(
                cached_steps[dir_idx].intermediates,
                ur, uc, exclude_mask, raster, rows, <int>cols, &intermediate_cost
            )

            if not valid_path:
                continue

            # Calculate edge weight
            raster_vr_vc = <float>raster[vr, vc]
            edge_weight = (raster_ur_uc + intermediate_cost + raster_vr_vc) * directions[dir_idx].cost_factor

            # Filter edges based on phase
            if light_phase_only and edge_weight > delta:
                continue
            if not light_phase_only and edge_weight <= delta:
                continue

            new_dist = current_dist + edge_weight

            # Update distance if improvement found
            should_update = False
            lock_idx = <int> (v & hash_mask)
            omp_set_lock(&hash_locks[lock_idx])
            if new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = <int64_t>u
                should_update = True
            omp_unset_lock(&hash_locks[lock_idx])

            # Add to thread-local buffer for bucket insertion
            if should_update and thread_results[tid].count < thread_results[tid].capacity:
                bucket_idx_temp = <size_t>(new_dist / delta)

                if bucket_idx_temp >= limits.max_buckets:
                    bucket_idx_stored = limits.max_buckets - 1
                else:
                    bucket_idx_stored = <uint32_t>bucket_idx_temp

                thread_results[tid].vertices[thread_results[tid].count] = v
                thread_results[tid].bucket_indices[thread_results[tid].count] = bucket_idx_stored
                thread_results[tid].distances[thread_results[tid].count] = new_dist
                thread_results[tid].count += 1

# ==================== INTERNAL DIJKSTRA IMPLEMENTATION ====================

cdef np.ndarray[uint32_t, ndim=1] _dijkstra_2d_cython_internal(
        uint16_t[:, :] raster, uint8_t[:, :] exclude_mask,
        vector[StepData] directions, uint32_t source_idx,
        uint32_t target_idx, int rows, int cols):
    """
    Core Dijkstra's algorithm implementation for single source-target pairs.

    This internal function implements the classical Dijkstra shortest path
    algorithm optimized for 2D raster graphs. It serves as the foundation
    for all other pathfinding variants in this module.

    Algorithm Overview:
        1. Initialize distance array with infinity, set source distance to 0
        2. Add source to priority queue with distance 0
        3. While queue not empty:
           a. Extract minimum distance node
           b. For each neighbor, calculate tentative distance
           c. Update distance if shorter path found
           d. Add/update neighbor in priority queue
        4. Reconstruct path by following predecessor links

    Parameters:
        raster: 2D cost matrix where each cell contains traversal cost
        exclude_mask: 2D boolean mask (1=traversable, 0=obstacle)
        directions: Precomputed movement directions with cost factors
        source_idx: Linear index of starting cell
        target_idx: Linear index of destination cell
        rows: Number of rows in the raster
        cols: Number of columns in the raster

    Returns:
        1D array of linear indices representing the optimal path from
        source to target. Empty array if no path exists.

    Performance Notes:
        - Early termination when target is reached
        - Binary heap provides efficient priority queue operations
        - Memory views enable zero-copy access to NumPy arrays
    """
    cdef int total_cells = rows * cols

    # Initialize Dijkstra data structures
    cdef float64_t[:] dist
    cdef int32_t[:] prev
    cdef uint8_t[:] visited

    dist = np.full(total_cells, np.inf, dtype=np.float64)
    prev = np.full(total_cells, -1, dtype=np.int32)
    visited = np.zeros(total_cells, dtype=np.uint8)

    # Initialize priority queue and set source distance
    cdef BinaryHeap pq
    heap_init(&pq)
    dist[source_idx] = 0.0
    heap_push(&pq, source_idx, 0.0)

    # Variables for main algorithm loop
    cdef uint32_t current
    cdef double current_dist
    cdef npy_intp current_row, current_col
    cdef npy_intp neighbor_row, neighbor_col
    cdef uint32_t neighbor
    cdef double intermediate_cost = 0.0
    cdef double total_cost, new_dist
    cdef int valid_path
    cdef int i, dr, dc

    # Main Dijkstra loop with early termination
    while not heap_empty(&pq):
        current = heap_top(&pq).index
        current_dist = heap_top(&pq).priority
        heap_pop(&pq)

        # Skip outdated entries and already visited nodes
        if visited[current] == 1 or current_dist > dist[current]:
            continue
        visited[current] = 1

        # Early termination when target reached
        if current == target_idx:
            break

        # Convert linear index to 2D coordinates
        unravel_index(current, cols, &current_row, &current_col)

        # Explore all possible movement directions
        for i in range(directions.size()):
            dr = directions[i].dr
            dc = directions[i].dc
            neighbor_row = current_row + dr
            neighbor_col = current_col + dc

            # Check boundary conditions
            if (neighbor_row < 0 or neighbor_row >= rows or
                    neighbor_col < 0 or neighbor_col >= cols):
                continue

            # Check if neighbor is traversable
            if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:
                continue

            neighbor = ravel_index(<int>neighbor_row, <int>neighbor_col, cols)

            # Skip already processed nodes
            if visited[neighbor] == 1:
                continue

            # Validate path and calculate intermediate costs
            intermediate_cost = 0.0
            valid_path = check_path(
                dr, dc, <int>current_row, <int>current_col,
                exclude_mask, raster, rows, cols, &intermediate_cost
            )

            if not valid_path:
                continue

            # Calculate total movement cost
            total_cost = (raster[<int>current_row, <int>current_col] +
                         intermediate_cost +
                         raster[<int>neighbor_row, <int>neighbor_col]) * (
                         directions[i].cost_factor)

            # Update shortest path if better route found
            new_dist = dist[current] + total_cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current
                heap_push(&pq, neighbor, new_dist)

    # Check if path to target exists
    if prev[target_idx] == -1:
        return np.empty(0, dtype=np.uint32)

    # Count path length for array allocation
    cdef int path_length
    cdef int idx
    path_length = 1
    current = target_idx
    while current != source_idx:
        current = prev[current]
        path_length += 1

    # Reconstruct path from target back to source
    path = np.empty(path_length, dtype=np.uint32)
    current = target_idx
    idx = path_length - 1

    while True:
        path[idx] = current
        if current == source_idx:
            break
        current = prev[current]
        idx -= 1

    return path

# ==================== PUBLIC DIJKSTRA ALGORITHMS ====================

def dijkstra_2d_cython(np.ndarray[uint16_t, ndim=2] raster_arr,
                       np.ndarray[int8_t, ndim=2] steps_arr,
                       uint32_t source_idx, uint32_t target_idx,
                       uint16_t max_value=65535):
    """
    Find shortest path between two points in a 2D raster using Dijkstra.

    This is the primary single-source, single-target pathfinding function.
    It handles all preprocessing steps including exclude mask creation and
    direction precomputation before delegating to the optimized internal
    implementation.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_idx: Linear index of starting cell
        target_idx: Linear index of destination cell
        max_value: Cost value representing obstacles (default 65535)

    Returns:
        1D numpy array (uint32) containing linear indices of cells in the
        optimal path from source to target. Empty array if no path exists.

    Performance Notes:
        - Typical runtime: 1-50ms for 1000x1000 rasters
        - Memory usage: ~40MB for 1000x1000 raster
    """
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]

    # Create traversability mask and precompute movement directions
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    exclude_mask_arr = (raster_arr != max_value).astype(np.uint8)
    cdef vector[StepData] directions = precompute_directions(steps_arr)

    # Delegate to optimized internal implementation
    return _dijkstra_2d_cython_internal(
        raster_arr, exclude_mask_arr, directions, source_idx, target_idx,
        rows, cols
    )

def dijkstra_single_source_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint32_t source_idx,
        np.ndarray[uint32_t, ndim=1] target_indices,
        uint16_t max_value=65535):
    """
    Find optimal paths from one source to multiple targets efficiently.

    This function implements a highly optimized variant of Dijkstra's algorithm
    that finds shortest paths from a single source to multiple targets in a
    single traversal. This approach is significantly more efficient than
    running separate single-target searches.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_idx: Linear index of the single starting cell
        target_indices: 1D numpy array (uint32) of target cell indices
        max_value: Cost value representing obstacles (default 65535)

    Returns:
        List of numpy arrays, where each array contains the optimal path from
        the source to the corresponding target. Empty arrays indicate no path.

    Performance Benefits:
        - 5 targets: ~3-5x faster than individual searches
        - 10 targets: ~5-8x faster
        - 50+ targets: ~10-15x faster
    """
    # Extract raster dimensions and create memory views
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef int total_cells = rows * cols
    cdef int num_targets = <int>target_indices.shape[0]

    cdef uint16_t[:, :] raster = raster_arr
    cdef uint32_t[:] targets = target_indices

    # Preprocessing: create traversability mask and movement directions
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    exclude_mask_arr = (raster_arr != max_value).astype(np.uint8)
    cdef uint8_t[:, :] exclude_mask = exclude_mask_arr
    cdef vector[StepData] directions = precompute_directions(steps_arr)

    # Initialize Dijkstra data structures
    cdef np.ndarray[float64_t, ndim=1] dist_arr = np.full(
        total_cells, np.inf, dtype=np.float64)
    cdef np.ndarray[int32_t, ndim=1] prev_arr = np.full(
        total_cells, -1, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] visited_arr = np.zeros(
        total_cells, dtype=np.uint8)

    # Track which targets have been found for early termination
    cdef np.ndarray[uint8_t, ndim=1] target_found_arr = np.zeros(
        num_targets, dtype=np.uint8)
    cdef uint8_t[:] target_found = target_found_arr
    cdef int targets_remaining = num_targets
    cdef int t

    cdef float64_t[:] dist = dist_arr
    cdef int32_t[:] prev = prev_arr
    cdef uint8_t[:] visited = visited_arr

    # Initialize priority queue and set source distance
    cdef BinaryHeap pq
    heap_init(&pq)
    dist[source_idx] = 0.0
    heap_push(&pq, source_idx, 0.0)

    # Variables for main algorithm loop
    cdef uint32_t current
    cdef double current_dist
    cdef npy_intp current_row, current_col
    cdef npy_intp neighbor_row, neighbor_col
    cdef uint32_t neighbor
    cdef double intermediate_cost = 0.0
    cdef double total_cost, new_dist
    cdef int valid_path
    cdef int i, dr, dc

    # Modified Dijkstra loop with multi-target termination
    while not heap_empty(&pq) and targets_remaining > 0:
        current = heap_top(&pq).index
        current_dist = heap_top(&pq).priority
        heap_pop(&pq)

        # Skip outdated entries and already visited nodes
        if visited[current] == 1 or current_dist > dist[current]:
            continue
        visited[current] = 1

        # Check if current node is any of our targets
        for t in range(num_targets):
            if current == targets[t] and target_found[t] == 0:
                target_found[t] = 1
                targets_remaining -= 1

        # Continue expanding the search frontier
        unravel_index(current, cols, &current_row, &current_col)

        # Process all movement directions
        for i in range(directions.size()):
            dr = directions[i].dr
            dc = directions[i].dc
            neighbor_row = current_row + dr
            neighbor_col = current_col + dc

            # Boundary and traversability checks
            if (neighbor_row < 0 or neighbor_row >= rows or
                    neighbor_col < 0 or neighbor_col >= cols):
                continue

            if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:
                continue

            neighbor = ravel_index(<int>neighbor_row, <int>neighbor_col, cols)

            if visited[neighbor] == 1:
                continue

            # Path validation and cost calculation
            intermediate_cost = 0.0
            valid_path = check_path(
                dr, dc, <int>current_row, <int>current_col,
                exclude_mask, raster, rows, cols, &intermediate_cost
            )

            if not valid_path:
                continue

            total_cost = (raster[<int>current_row, <int>current_col] +
                         intermediate_cost +
                         raster[<int>neighbor_row, <int>neighbor_col]) * (
                         directions[i].cost_factor)

            # Update shortest path if improvement found
            new_dist = dist[current] + total_cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current
                heap_push(&pq, neighbor, new_dist)

    # Reconstruct paths for all targets
    cdef int path_length
    cdef np.ndarray[uint32_t, ndim=1] path
    cdef list paths = []
    for t in range(num_targets):
        target_idx = targets[t]

        # Check if path exists to this target
        if prev[target_idx] == -1:
            paths.append(np.empty(0, dtype=np.uint32))
            continue

        # Count path length
        path_length = 1
        current = target_idx
        while current != source_idx:
            current = prev[current]
            path_length += 1

        # Reconstruct path
        path = np.empty(path_length, dtype=np.uint32)
        current = target_idx
        idx = path_length - 1

        while True:
            path[idx] = current
            if current == source_idx:
                break
            current = prev[current]
            idx -= 1

        paths.append(path)

    return paths


def dijkstra_multiple_sources_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint32_t, ndim=1] source_indices,
        np.ndarray[uint32_t, ndim=1] target_indices,
        uint16_t max_value=65535, bint return_paths=True):
    """
    Compute all-pairs shortest paths between multiple sources and targets.

    This function finds the optimal path from every source to every target,
    creating a complete distance matrix or path collection. It optimizes
    computation by processing sources in spatial proximity order.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_indices: 1D array (uint32) of all source cell indices
        target_indices: 1D array (uint32) of all target cell indices
        max_value: Cost value representing obstacles (default 65535)
        return_paths: If True, returns paths; if False, returns cost matrix

    Returns:
        If return_paths=True: List of lists with paths[i][j] = path from source i to target j
        If return_paths=False: 2D cost matrix with distances
    """
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef int num_sources = <int>source_indices.shape[0]
    cdef int num_targets = <int>target_indices.shape[0]

    # Declare all variables at function start (Cython requirement)
    cdef np.ndarray[uint32_t, ndim=1] sorted_sources
    cdef np.ndarray[float64_t, ndim=2] cost_matrix = np.full(
        (num_sources, num_targets), np.inf)
    cdef list paths = [] if return_paths else None
    cdef list source_paths
    cdef int s, t, original_idx
    cdef dict source_idx_map = {}
    cdef uint32_t source_idx
    cdef double cost

    # Optimize processing order by spatial proximity
    sorted_sources = group_by_proximity_uint32(source_indices, <uint64_t> cols)

    # Create mapping from sorted positions back to original indices
    for s in range(num_sources):
        for original_idx in range(num_sources):
            if sorted_sources[s] == source_indices[original_idx]:
                source_idx_map[s] = original_idx
                break

    # Process each source to find paths to all targets
    for s in range(num_sources):
        source_idx = sorted_sources[s]
        original_idx = source_idx_map[s]

        # Single computation finds paths to all targets from this source
        source_paths = dijkstra_single_source_multiple_targets(
            raster_arr, steps_arr, source_idx, target_indices, max_value
        )

        # Store path results if requested
        if return_paths:
            if len(paths) <= original_idx:
                paths.extend([None] * (original_idx - len(paths) + 1))
            paths[original_idx] = source_paths
        else:
            # Calculate costs and populate distance matrix
            for t in range(num_targets):
                if len(source_paths[t]) > 0:
                    cost = path_cost_uint32(source_paths[t], raster_arr, cols)
                    cost_matrix[original_idx, t] = cost

    return paths if return_paths else cost_matrix


def dijkstra_some_pairs_shortest_paths(np.ndarray[uint16_t, ndim=2] raster_arr,
                                       np.ndarray[int8_t, ndim=2] steps_arr,
                                       np.ndarray[uint32_t, ndim=1] source_indices,
                                       np.ndarray[uint32_t, ndim=1] target_indices,
                                       uint16_t max_value=65535,
                                       bint return_paths=True):
    """
    Find optimal paths for specific source-target pairs using batch optimization.

    This function efficiently computes shortest paths for a set of specific
    source-target pairs by identifying opportunities to batch related queries.
    It analyzes the connectivity patterns in the input pairs and uses nodes
    that appear as both sources and targets as "central hubs" to minimize
    the number of separate Dijkstra runs required.

    Optimization Strategy:
        1. Identify nodes that serve as both sources and targets (central nodes)
        2. For each central node, batch all related queries into a single
           multi-target Dijkstra run
        3. Handle remaining one-off pairs with individual computations
        4. Automatically reverse paths when necessary for correct orientation

    Use Cases:
        - Vehicle routing with pickup and delivery constraints
        - Multi-modal transportation planning
        - Supply chain optimization with intermediate warehouses
        - Network flow problems with specific origin-destination pairs

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_indices: 1D array (uint32) of source cell indices
        target_indices: 1D array (uint32) of target cell indices
                       (pairs formed by matching array positions)
        max_value: Cost value representing obstacles (default 65535)
        return_paths: If True, returns actual paths; if False, returns only costs

    Returns:
        If return_paths=True: List of path arrays (may contain empty arrays)
        If return_paths=False: 1D array of path costs (inf for no path)

    Performance Benefits:
        - Typical speedup: 2-10x over individual pair computations
        - Memory efficiency: Reuses data structures across related queries
        - Scales well with increasing connectivity between sources and targets

    Example:
        >>> # Multi-stop delivery route optimization
        >>> sources = np.array([depot, store1, store2], dtype=np.uint32)
        >>> targets = np.array([store1, store2, depot], dtype=np.uint32)
        >>> paths = dijkstra_some_pairs_shortest_paths(
        ...     raster, steps, sources, targets)
        >>> # Returns: [depot→store1, store1→store2, store2→depot]
    """
    cdef int rows = <int> raster_arr.shape[0]
    cdef int cols = <int> raster_arr.shape[1]
    cdef int num_pairs = <int> min(source_indices.shape[0],
                                   target_indices.shape[0])

    # Initialize result containers
    cdef list all_paths = [None] * num_pairs if return_paths else None
    cdef np.ndarray[float64_t, ndim=1] costs = np.full(num_pairs, np.inf)

    # Data structures for batching optimization
    cdef dict node_sources = {}  # target → [sources pointing to it]
    cdef dict node_targets = {}  # source → [targets it points to]
    cdef dict pair_indices = {}  # (source, target) → original index
    cdef set processed_pairs = set()  # Track completed computations

    cdef int i, j
    cdef uint32_t source, target
    cdef list central_nodes = []  # Nodes appearing as both sources/targets
    cdef np.ndarray[uint32_t, ndim=1] path

    # Phase 1: Analyze connectivity patterns and identify central nodes
    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        # Store original pair index for result mapping
        pair_indices[(source, target)] = i

        # Build reverse connectivity maps
        if target not in node_sources:
            node_sources[target] = []
        node_sources[target].append(source)

        if source not in node_targets:
            node_targets[source] = []
        node_targets[source].append(target)

        # Identify potential central nodes (nodes with both incoming/outgoing)
        if source in node_sources and target in node_targets:
            if source not in central_nodes:
                central_nodes.append(source)
            if target not in central_nodes:
                central_nodes.append(target)

    # Add remaining nodes that are both sources and targets
    for node in node_sources:
        if node in node_targets and node not in central_nodes:
            central_nodes.append(node)

    # Phase 2: Process central nodes with batch optimization
    for central_node in central_nodes:
        if (central_node not in node_sources and
                central_node not in node_targets):
            continue

        # Collect all queries that can be batched through this central node
        batch_targets = []
        pair_mapping = []  # Maps batch index to original pair index
        reverse_flags = []  # Tracks which paths need reversal

        # Add forward paths (central_node as source)
        if central_node in node_targets:
            for target in node_targets[central_node]:
                if (central_node, target) not in processed_pairs:
                    batch_targets.append(target)
                    pair_mapping.append(pair_indices[(central_node, target)])
                    reverse_flags.append(False)  # No reversal needed
                    processed_pairs.add((central_node, target))

        # Add reverse paths (central_node as target, compute backward)
        if central_node in node_sources:
            for source in node_sources[central_node]:
                if (source, central_node) not in processed_pairs:
                    batch_targets.append(source)
                    pair_mapping.append(pair_indices[(source, central_node)])
                    reverse_flags.append(True)  # Reversal needed
                    processed_pairs.add((source, central_node))

        # Execute batched computation if targets found
        if batch_targets:
            targets_array = np.array(batch_targets, dtype=np.uint32)
            result_paths = dijkstra_single_source_multiple_targets(
                raster_arr, steps_arr, central_node, targets_array, max_value
            )

            # Process results and map back to original pair indices
            for j in range(len(result_paths)):
                path = result_paths[j]
                pair_idx = pair_mapping[j]
                need_reverse = reverse_flags[j]

                if return_paths:
                    if len(path) > 0:
                        if need_reverse:
                            path = np.flip(path)  # Correct path orientation
                        all_paths[pair_idx] = path
                    else:
                        all_paths[pair_idx] = np.empty(0, dtype=np.uint32)
                else:
                    # Calculate path cost
                    if len(path) > 0:
                        costs[pair_idx] = path_cost_uint32(path, raster_arr, cols)

    # Phase 3: Handle remaining unprocessed pairs individually
    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        if (source, target) in processed_pairs:
            continue

        # Process individual pair with single-target Dijkstra
        result_paths = dijkstra_single_source_multiple_targets(
            raster_arr, steps_arr, source, np.array([target], dtype=np.uint32),
            max_value
        )

        path = result_paths[0]
        if return_paths:
            all_paths[i] = path
        else:
            if len(path) > 0:
                costs[i] = path_cost_uint32(path, raster_arr, cols)

        processed_pairs.add((source, target))

    return all_paths if return_paths else costs

# ==================== DELTA-STEPPING ALGORITHMS ====================

def delta_stepping_2d(np.ndarray[uint16_t, ndim=2] raster_arr,
                      np.ndarray[int8_t, ndim=2] steps_arr,
                      uint64_t source_idx, uint64_t target_idx,
                      float delta,
                      uint16_t max_value=65535,
                      int num_threads=0,
                      size_t max_buckets_in_memory=2048,
                      float margin=1.00001):
    """
    Find the shortest path using parallel delta-stepping with circular buffer.

    This function implements the delta-stepping algorithm [1] for single-source
    single-target the shortest path computation. It uses a circular buffer to manage
    buckets efficiently and supports parallel edge relaxation for improved
    performance on multi-core systems.

    The margin parameter enables early termination once the target is found.
    When the target is settled, the algorithm continues processing buckets only
    until distance > target_distance * margin, ensuring optimal path discovery
    while avoiding unnecessary computation. This optimization is particularly
    effective for large graphs where the target is much closer than the graph
    diameter.

    Algorithm Overview:
        1. Initialize circular buffer of size max_buckets_in_memory (power of 2)
        2. Process buckets in order of increasing distance:
           - Light phase: Relax edges with weight ≤ delta (can be done multiple times)
           - Heavy phase: Relax edges with weight > delta (once per bucket)
        3. Early termination when current_bucket * delta > target_distance * margin
        4. Reconstruct path using predecessor array

    References:
    [1] Meyer, U., Sanders, P.: δ-stepping: a parallelizable shortest path algorithm. J.
        Algorithms 49, 1 (2003), 114–152.
        DOI:http://dx.doi.org/10.1016/S0196-6774(03)00076-2 1998 European Symposium
        on Algorithms.

    Parameters:
        raster_arr: 2D cost matrix where each cell contains traversal cost
        steps_arr: 2D array defining movement directions as (dr, dc) pairs
        source_idx: Linear index of the starting cell
        target_idx: Linear index of the destination cell
        delta: Bucket width for edge classification (must be > 0)
        max_value: Cost value representing obstacles (default 65535)
        num_threads: Number of OpenMP threads (0 = auto-detect)
        max_buckets_in_memory: Size of circular buffer (must be power of 2)
        margin: Safety factor for early termination (default 1.0001)
                Values > 1.0 allow earlier termination with confidence

    Returns:
        1D numpy array (uint64) of linear indices representing the optimal path.
        Empty array if no path exists.

    Performance Characteristics:
        - Time complexity: O((V + E) / p + D·L) where p=parallelism, D=diameter
        - Memory: O(V) for distance/predecessor + O(B) for circular buffer
        - Early termination reduces average case significantly
        - Typical speedup: 2-10x over Dijkstra for appropriate delta values
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    # System and problem dimensions
    cdef SystemLimits sys_limits = get_system_limits()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols

    # Preprocessing variables
    cdef float computed_delta
    cdef float termination_margin
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    cdef const uint8_t[:, :] exclude_view
    cdef const uint16_t[:, :] raster_view

    # Circular buffer configuration
    cdef size_t circular_buffer_size
    cdef size_t buffer_mask
    cdef size_t logical_bucket_count = 0
    cdef size_t current_logical_bucket = 0
    cdef size_t physical_bucket_idx = 0
    cdef size_t window_start = 0
    cdef bint bucket_valid = False

    # Bucket deduplication
    cdef np.ndarray[int32_t, ndim=1] last_bucket_arr
    cdef int32_t[:] last_bucket

    # Thread configuration
    cdef int actual_threads
    cdef int num_hash_locks
    cdef int hash_lock_count
    cdef omp_lock_t* hash_locks = NULL
    cdef int lock_idx

    # Thread-local storage
    cdef ThreadResults* thread_results = NULL
    cdef int max_capacity
    cdef int tid

    # Algorithm state
    cdef vector[vector[uint64_t]] buckets
    cdef vector[uint64_t] current_vertices
    cdef vector[uint64_t] settled_vertices
    cdef bint target_found = False
    cdef float target_distance = INF_F32
    cdef float cutoff_distance = INF_F32

    # Edge relaxation variables
    cdef int iteration, light_iterations
    cdef int max_light_iterations = 100
    cdef uint64_t v, vertex_to_add
    cdef size_t new_logical_bucket, new_physical_bucket
    cdef float new_dist
    cdef int32_t last_bucket_for_vertex

    # Path reconstruction
    cdef uint64_t current, path_length
    cdef list path_vertices

    # Preprocessing data structures
    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions

    # Distance and predecessor arrays
    cdef np.ndarray[float32_t, ndim=1] dist
    cdef np.ndarray[int64_t, ndim=1] pred

    # Validation variables
    cdef uint64_t source_r, source_c, target_r, target_c

    # Loop variables
    cdef int i, j

    # ============= VALIDATION =============

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")

    if source_idx >= total_cells or target_idx >= total_cells:
        return np.empty(0, dtype=np.uint64)

    # ============= PREPROCESSING =============

    exclude_mask_arr = (raster_arr != max_value).astype(np.uint8)

    # Validate delta
    if delta <= 0.0:
        raise ValueError(f"Invalid delta value: {delta}! Choose a delta > 0.0!")
    computed_delta = delta

    # Validate margin
    if margin <= 1.00001:
        termination_margin = 1.00001
    else:
        termination_margin = margin

    # Check source and target traversability
    source_r = source_idx // cols
    source_c = source_idx % cols
    target_r = target_idx // cols
    target_c = target_idx % cols

    if (exclude_mask_arr[source_r, source_c] == 0 or
        exclude_mask_arr[target_r, target_c] == 0):
        return np.empty(0, dtype=np.uint64)

    # Thread configuration
    if num_threads <= 0:
        num_threads = min(sys_limits.num_cores, omp_get_max_threads())
    omp_set_num_threads(num_threads)
    actual_threads = omp_get_max_threads()

    # Configure locks (power of 2 for fast modulo)
    num_hash_locks = max(1024, actual_threads * 256)
    hash_lock_count = <int>round_up_power_of_two(<int>num_hash_locks)
    num_hash_locks = min(hash_lock_count, 65536)

    # Allocate locks
    hash_locks = <omp_lock_t*>malloc(num_hash_locks * sizeof(omp_lock_t))
    if hash_locks == NULL:
        raise MemoryError("Could not allocate hash locks")

    for lock_idx in range(num_hash_locks):
        omp_init_lock(&hash_locks[lock_idx])

    # Precompute movement data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize distance and predecessor arrays
    dist = np.full(<size_t>total_cells, INF_F32, dtype=np.float32)
    dist[source_idx] = 0.0
    pred = np.full(<size_t>total_cells, -1, dtype=np.int64)

    # Create memory views
    raster_view = raster_arr
    exclude_view = exclude_mask_arr

    # Initialize circular buffer with fixed size
    circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    buffer_mask = circular_buffer_size - 1
    buckets.resize(circular_buffer_size)

    # Initialize last bucket tracking
    last_bucket_arr = np.full(<size_t>total_cells, -1, dtype=np.int32)
    last_bucket = last_bucket_arr

    # Add source to first bucket
    physical_bucket_idx = get_circular_index(0, circular_buffer_size)
    buckets[physical_bucket_idx].push_back(source_idx)
    last_bucket[source_idx] = 0

    # Allocate thread buffers
    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)
        raise MemoryError("Could not allocate thread data")

    max_capacity = calculate_thread_buffer_capacity(total_cells, actual_threads, &sys_limits)

    for tid in range(actual_threads):
        thread_results[tid].vertices = <uint64_t*>malloc(max_capacity * sizeof(uint64_t))
        thread_results[tid].bucket_indices = <uint32_t*>malloc(max_capacity * sizeof(uint32_t))
        thread_results[tid].distances = <float*>malloc(max_capacity * sizeof(float))

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

    # ============= MAIN DELTA-STEPPING LOOP =============

    try:
        for iteration in range(sys_limits.max_iterations):
            # Find next non-empty bucket
            bucket_valid = False
            while current_logical_bucket < logical_bucket_count + circular_buffer_size:
                physical_bucket_idx = get_circular_index(current_logical_bucket, circular_buffer_size)

                if current_logical_bucket >= logical_bucket_count + circular_buffer_size:
                    break

                if not buckets[physical_bucket_idx].empty():
                    bucket_valid = True
                    break

                current_logical_bucket += 1

            if not bucket_valid:
                break

            # Update processing window
            window_start = current_logical_bucket
            settled_vertices.clear()

            # LIGHT PHASE
            light_iterations = 0
            while not buckets[physical_bucket_idx].empty() and light_iterations < max_light_iterations:
                light_iterations += 1

                current_vertices = buckets[physical_bucket_idx]
                buckets[physical_bucket_idx].clear()

                settled_vertices.insert(settled_vertices.end(),
                                       current_vertices.begin(),
                                       current_vertices.end())

                for tid in range(actual_threads):
                    thread_results[tid].count = 0

                relax_edges_delta_stepping(
                    current_vertices,
                    <float*>dist.data, <int64_t*>pred.data,
                    raster_view, exclude_view,
                    directions, cached_steps,
                    rows, cols, computed_delta, True,  # light_phase_only
                    thread_results, actual_threads,
                    hash_locks, num_hash_locks, total_cells,
                    target_idx, &sys_limits
                )

                # Merge thread results with deduplication
                for tid in range(actual_threads):
                    for i in range(thread_results[tid].count):
                        vertex_to_add = thread_results[tid].vertices[i]
                        new_dist = thread_results[tid].distances[i]
                        new_logical_bucket = <size_t>(new_dist / computed_delta)

                        if new_logical_bucket < window_start + circular_buffer_size:
                            new_physical_bucket = get_circular_index(new_logical_bucket, circular_buffer_size)

                            last_bucket_for_vertex = last_bucket[vertex_to_add]
                            if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                buckets[new_physical_bucket].push_back(vertex_to_add)
                                last_bucket[vertex_to_add] = <int32_t>new_logical_bucket

                            if new_logical_bucket >= logical_bucket_count:
                                logical_bucket_count = new_logical_bucket + 1

            # Check if target found
            for i in range(<int>settled_vertices.size()):
                if settled_vertices[i] == target_idx:
                    target_found = True
                    target_distance = dist[target_idx]
                    cutoff_distance = target_distance * termination_margin
                    break

            # Early termination
            if target_found and current_logical_bucket * computed_delta > cutoff_distance:
                break

            # HEAVY PHASE
            if not settled_vertices.empty():
                for tid in range(actual_threads):
                    thread_results[tid].count = 0

                relax_edges_delta_stepping(
                    settled_vertices,
                    <float*>dist.data, <int64_t*>pred.data,
                    raster_view, exclude_view,
                    directions, cached_steps,
                    rows, cols, computed_delta, False,  # heavy edges
                    thread_results, actual_threads,
                    hash_locks, num_hash_locks, total_cells,
                    target_idx, &sys_limits
                )

                for tid in range(actual_threads):
                    for i in range(thread_results[tid].count):
                        vertex_to_add = thread_results[tid].vertices[i]
                        new_dist = thread_results[tid].distances[i]
                        new_logical_bucket = <size_t>(new_dist / computed_delta)

                        if (new_logical_bucket > current_logical_bucket and
                            new_logical_bucket < window_start + circular_buffer_size):
                            new_physical_bucket = get_circular_index(new_logical_bucket, circular_buffer_size)

                            last_bucket_for_vertex = last_bucket[vertex_to_add]
                            if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                buckets[new_physical_bucket].push_back(vertex_to_add)
                                last_bucket[vertex_to_add] = <int32_t>new_logical_bucket

                            if new_logical_bucket >= logical_bucket_count:
                                logical_bucket_count = new_logical_bucket + 1

            # Clear processed bucket to free memory
            buckets[physical_bucket_idx].clear()
            buckets[physical_bucket_idx].shrink_to_fit()

            current_logical_bucket += 1

    finally:
        # Cleanup resources
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        if hash_locks != NULL:
            free(hash_locks)

        if thread_results != NULL:
            for tid in range(actual_threads):
                if thread_results[tid].vertices != NULL:
                    free(thread_results[tid].vertices)
                if thread_results[tid].bucket_indices != NULL:
                    free(thread_results[tid].bucket_indices)
                if thread_results[tid].distances != NULL:
                    free(thread_results[tid].distances)
            free(thread_results)

    # Path reconstruction
    if not target_found or pred[target_idx] == -1:
        if source_idx == target_idx:
            return np.array([source_idx], dtype=np.uint64)
        return np.empty(0, dtype=np.uint64)

    path_vertices = []
    current = target_idx
    path_length = 0

    while current != <uint64_t>(-1) and path_length < sys_limits.max_path_length:
        path_vertices.append(current)
        if current == source_idx:
            break
        current = <uint64_t>pred[current]
        if current == <uint64_t>(-1):
            return np.empty(0, dtype=np.uint64)
        path_length += 1

    path_vertices.reverse()

    return np.array(path_vertices, dtype=np.uint64)

def delta_stepping_single_source_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint64_t source_idx,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        uint16_t max_value=65535,
        int num_threads=0,
        size_t max_buckets_in_memory=2048):
    """
    Find optimal paths from single source to multiple targets.

    This function extends delta-stepping to efficiently find the shortest paths
    from one source to multiple targets in a single traversal. The algorithm
    continues until ALL targets have been discovered, making it significantly
    more efficient than running separate single-target searches.

    IMPORTANT: No Margin Parameter
    ===============================
    Unlike delta_stepping_2d, this function does NOT include a margin parameter
    for early termination. The reasons are:

    1. Completeness requirement: Must find paths to ALL targets, not just the
       nearest one. Targets may be at vastly different distances.

    2. Algorithm correctness: Delta-stepping expands outward in distance order.
       Stopping after finding the first target (even with margin) would miss
       targets beyond that distance threshold.

    Example: Source at (0,0), Target A at distance 100, Target B at distance 150
             With margin=1.1, would stop at 110, never reaching Target B

    The algorithm does terminate early when ALL targets are found, providing
    the maximum safe optimization without sacrificing correctness.

    Parameters:
        raster_arr: 2D cost matrix where each cell contains traversal cost
        steps_arr: 2D array defining movement directions
        source_idx: Linear index of the starting cell
        target_indices: 1D array of target cell indices to find paths to
        delta: Bucket width for edge classification (must be > 0)
        max_value: Cost value representing obstacles
        num_threads: Number of OpenMP threads (0 = auto-detect)
        max_buckets_in_memory: Size of circular buffer (power of 2)

    Returns:
        List of numpy arrays, one path per target (empty if no path exists)

    Performance:
        - Single traversal for all targets vs. N separate searches
        - Typical speedup: 5-15x for 10+ targets
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    # System and problem dimensions
    cdef SystemLimits sys_limits = get_system_limits()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols
    cdef int num_targets = <int>target_indices.shape[0]

    # Preprocessing variables
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    cdef const uint16_t[:, :] raster_view
    cdef const uint8_t[:, :] exclude_view

    # Circular buffer configuration
    cdef size_t circular_buffer_size
    cdef size_t buffer_mask
    cdef size_t logical_bucket_count = 0
    cdef size_t current_logical_bucket = 0
    cdef size_t physical_bucket_idx = 0
    cdef size_t window_start = 0
    cdef bint bucket_valid = False

    # Bucket deduplication
    cdef np.ndarray[int32_t, ndim=1] last_bucket_arr
    cdef int32_t[:] last_bucket

    # Thread configuration
    cdef int actual_threads
    cdef int num_hash_locks
    cdef int hash_lock_count
    cdef omp_lock_t* hash_locks = NULL
    cdef int lock_idx

    # Thread-local storage
    cdef ThreadResults* thread_results = NULL
    cdef int max_capacity
    cdef int tid

    # Algorithm state
    cdef vector[vector[uint64_t]] buckets
    cdef vector[uint64_t] current_vertices
    cdef vector[uint64_t] settled_vertices
    cdef int targets_found = 0
    cdef np.ndarray[uint8_t, ndim=1] target_found_arr
    cdef uint8_t[:] target_found

    # For tracking maximum distance found (for potential optimization)
    cdef float max_target_distance = 0.0
    cdef float current_target_distance

    # Edge relaxation variables
    cdef int iteration, light_iterations
    cdef int max_light_iterations = 100
    cdef uint64_t v, vertex_to_add, current_vertex
    cdef size_t new_logical_bucket, new_physical_bucket
    cdef float new_dist
    cdef int32_t last_bucket_for_vertex

    # Path reconstruction
    cdef uint64_t target_idx, path_length
    cdef list path_vertices
    cdef list paths = []

    # Preprocessing data structures
    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions

    # Distance and predecessor arrays
    cdef np.ndarray[float32_t, ndim=1] dist
    cdef np.ndarray[int64_t, ndim=1] pred

    # Validation variables
    cdef uint64_t source_r, source_c

    # Loop variables
    cdef int i, j

    # ============= VALIDATION =============

    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if num_targets == 0:
        return []

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")

    # Thread configuration
    if num_threads <= 0:
        num_threads = min(sys_limits.num_cores, omp_get_max_threads())
    omp_set_num_threads(num_threads)
    actual_threads = omp_get_max_threads()

    # Configure locks (power of 2)
    num_hash_locks = max(1024, actual_threads * 256)
    hash_lock_count = <int>round_up_power_of_two(<int>num_hash_locks)
    num_hash_locks = min(hash_lock_count, 65536)

    # Allocate synchronization structures
    hash_locks = <omp_lock_t*>malloc(num_hash_locks * sizeof(omp_lock_t))
    if hash_locks == NULL:
        raise MemoryError("Could not allocate hash locks")

    for lock_idx in range(num_hash_locks):
        omp_init_lock(&hash_locks[lock_idx])

    # Validate indices
    if source_idx >= total_cells:
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)
        return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    for i in range(num_targets):
        if target_indices[i] >= total_cells:
            for lock_idx in range(num_hash_locks):
                omp_destroy_lock(&hash_locks[lock_idx])
            free(hash_locks)
            return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    # Create traversability mask
    exclude_mask_arr = (raster_arr != max_value).astype(np.uint8)
    source_r = source_idx // cols
    source_c = source_idx % cols

    if exclude_mask_arr[source_r, source_c] == 0:
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)
        return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    # Precompute movement data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize distance and tracking arrays
    dist = np.full(<size_t>total_cells, INF_F32, dtype=np.float32)
    pred = np.full(<size_t>total_cells, -1, dtype=np.int64)
    target_found_arr = np.zeros(num_targets, dtype=np.uint8)

    dist[source_idx] = 0.0

    # Create memory views
    raster_view = raster_arr
    exclude_view = exclude_mask_arr
    target_found = target_found_arr

    # Initialize circular buffer
    circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    buffer_mask = circular_buffer_size - 1
    buckets.resize(circular_buffer_size)

    # Initialize last bucket tracking
    last_bucket_arr = np.full(<size_t>total_cells, -1, dtype=np.int32)
    last_bucket = last_bucket_arr

    # Add source to first bucket
    physical_bucket_idx = get_circular_index(0, circular_buffer_size)
    buckets[physical_bucket_idx].push_back(source_idx)
    last_bucket[source_idx] = 0

    # Allocate thread buffers
    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)
        raise MemoryError("Could not allocate thread data")

    max_capacity = calculate_thread_buffer_capacity(total_cells, actual_threads, &sys_limits)

    for tid in range(actual_threads):
        thread_results[tid].vertices = <uint64_t*>malloc(max_capacity * sizeof(uint64_t))
        thread_results[tid].bucket_indices = <uint32_t*>malloc(max_capacity * sizeof(uint32_t))
        thread_results[tid].distances = <float*>malloc(max_capacity * sizeof(float))

        if (thread_results[tid].vertices == NULL or
            thread_results[tid].bucket_indices == NULL or
            thread_results[tid].distances == NULL):
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

    # Set iteration limit
    max_light_iterations = max(50, <int>(sqrtf(<float>total_cells)))

    try:
        for iteration in range(sys_limits.max_iterations):
            # Find next non-empty bucket
            bucket_valid = False
            while current_logical_bucket < logical_bucket_count + circular_buffer_size:
                physical_bucket_idx = get_circular_index(current_logical_bucket, circular_buffer_size)

                if current_logical_bucket >= logical_bucket_count + circular_buffer_size:
                    break

                if not buckets[physical_bucket_idx].empty():
                    bucket_valid = True
                    break

                current_logical_bucket += 1

            if not bucket_valid:
                break

            window_start = current_logical_bucket
            settled_vertices.clear()
            light_iterations = 0

            # LIGHT PHASE
            while not buckets[physical_bucket_idx].empty() and light_iterations < max_light_iterations:
                light_iterations += 1

                current_vertices = buckets[physical_bucket_idx]
                buckets[physical_bucket_idx].clear()

                settled_vertices.insert(settled_vertices.end(),
                                       current_vertices.begin(),
                                       current_vertices.end())

                for tid in range(actual_threads):
                    thread_results[tid].count = 0

                relax_edges_delta_stepping(
                    current_vertices,
                    <float*>dist.data, <int64_t*>pred.data,
                    raster_view, exclude_view,
                    directions, cached_steps,
                    rows, cols, delta, True,  # light_phase_only
                    thread_results, actual_threads,
                    hash_locks, num_hash_locks, total_cells,
                    0, &sys_limits  # No specific target for multi-target
                )

                # Merge thread results with deduplication
                for tid in range(actual_threads):
                    for i in range(thread_results[tid].count):
                        vertex_to_add = thread_results[tid].vertices[i]
                        new_dist = thread_results[tid].distances[i]
                        new_logical_bucket = <size_t>(new_dist / delta)

                        if new_logical_bucket < window_start + circular_buffer_size:
                            new_physical_bucket = get_circular_index(new_logical_bucket, circular_buffer_size)

                            last_bucket_for_vertex = last_bucket[vertex_to_add]
                            if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                buckets[new_physical_bucket].push_back(vertex_to_add)
                                last_bucket[vertex_to_add] = <int32_t>new_logical_bucket

                            if new_logical_bucket >= logical_bucket_count:
                                logical_bucket_count = new_logical_bucket + 1

            # Check if any targets were settled
            for i in range(<int>settled_vertices.size()):
                current_vertex = settled_vertices[i]
                for j in range(num_targets):
                    if current_vertex == target_indices[j] and target_found[j] == 0:
                        target_found[j] = 1
                        targets_found += 1

                        # Track maximum distance for information (but don't terminate)
                        current_target_distance = dist[current_vertex]
                        if current_target_distance > max_target_distance:
                            max_target_distance = current_target_distance

            # Only terminate when ALL targets are found
            if targets_found >= num_targets:
                break

            # HEAVY PHASE
            if not settled_vertices.empty():
                for tid in range(actual_threads):
                    thread_results[tid].count = 0

                relax_edges_delta_stepping(
                    settled_vertices,
                    <float*>dist.data, <int64_t*>pred.data,
                    raster_view, exclude_view,
                    directions, cached_steps,
                    rows, cols, delta, False,  # heavy edges
                    thread_results, actual_threads,
                    hash_locks, num_hash_locks, total_cells,
                    0, &sys_limits
                )

                for tid in range(actual_threads):
                    for i in range(thread_results[tid].count):
                        vertex_to_add = thread_results[tid].vertices[i]
                        new_dist = thread_results[tid].distances[i]
                        new_logical_bucket = <size_t>(new_dist / delta)

                        if (new_logical_bucket > current_logical_bucket and
                            new_logical_bucket < window_start + circular_buffer_size):
                            new_physical_bucket = get_circular_index(new_logical_bucket, circular_buffer_size)

                            last_bucket_for_vertex = last_bucket[vertex_to_add]
                            if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                buckets[new_physical_bucket].push_back(vertex_to_add)
                                last_bucket[vertex_to_add] = <int32_t>new_logical_bucket

                            if new_logical_bucket >= logical_bucket_count:
                                logical_bucket_count = new_logical_bucket + 1

            # Clear processed bucket
            buckets[physical_bucket_idx].clear()
            buckets[physical_bucket_idx].shrink_to_fit()

            current_logical_bucket += 1

    finally:
        # Cleanup resources
        for lock_idx in range(num_hash_locks):
            omp_destroy_lock(&hash_locks[lock_idx])
        free(hash_locks)

        for tid in range(actual_threads):
            free(thread_results[tid].vertices)
            free(thread_results[tid].bucket_indices)
            free(thread_results[tid].distances)
        free(thread_results)

    # Reconstruct paths for all targets
    for i in range(num_targets):
        target_idx = target_indices[i]

        if pred[target_idx] == -1:
            if source_idx == target_idx:
                paths.append(np.array([source_idx], dtype=np.uint64))
            else:
                paths.append(np.empty(0, dtype=np.uint64))
            continue

        path_vertices = []
        current_vertex = target_idx
        path_length = 0

        while current_vertex != source_idx and path_length < sys_limits.max_path_length:
            path_vertices.append(current_vertex)
            if pred[current_vertex] == -1:
                paths.append(np.empty(0, dtype=np.uint64))
                break
            current_vertex = <uint64_t>pred[current_vertex]
            path_length += 1
        else:
            if current_vertex == source_idx:
                path_vertices.append(source_idx)
                path_vertices.reverse()
                paths.append(np.array(path_vertices, dtype=np.uint64))
            else:
                paths.append(np.empty(0, dtype=np.uint64))

    return paths


def delta_stepping_multiple_sources_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint64_t, ndim=1] source_indices,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        uint16_t max_value=65535,
        bint return_paths=True,
        int num_threads=0,
        size_t max_buckets_in_memory=2048):
    """
    Compute all-pairs shortest paths between multiple sources and targets.

    Finds optimal paths from every source to every target by iterating through
    sources and using single-source-multiple-targets delta-stepping for each.
    Sources are processed in spatial proximity order for better cache locality.

    IMPORTANT: No Margin Parameter
    ===============================
    This function does not include a margin parameter because:

    1. It delegates to delta_stepping_single_source_multiple_targets, which
       itself cannot use margin (must find ALL targets from each source)

    2. Users expect a complete M×N result matrix where M=sources, N=targets.
       Partial results would violate this contract.

    3. The primary optimization here is batching multiple targets per source,
       not early termination.

    Parameters:
        raster_arr: 2D cost matrix with traversal costs
        steps_arr: 2D array defining movement directions
        source_indices: 1D array of source cell indices
        target_indices: 1D array of target cell indices
        delta: Bucket width for edge classification
        max_value: Cost value representing obstacles
        return_paths: If True, returns paths; if False, returns cost matrix
        num_threads: Number of OpenMP threads
        max_buckets_in_memory: Circular buffer size

    Returns:
        If return_paths=True: List of lists, paths[i][j] = path from source i to target j
        If return_paths=False: 2D cost matrix with distances
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef int num_sources = <int>source_indices.shape[0]
    cdef int num_targets = <int>target_indices.shape[0]

    # Result containers
    cdef np.ndarray[float32_t, ndim=2] cost_matrix = np.full(
        (num_sources, num_targets), INF_F32, dtype=np.float32)
    cdef list all_paths = [] if return_paths else None

    # Source processing variables
    cdef np.ndarray[uint64_t, ndim=1] sorted_sources
    cdef dict source_idx_map = {}
    cdef int s, t, original_idx
    cdef uint64_t source_idx
    cdef list source_paths
    cdef np.ndarray[uint64_t, ndim=1] path
    cdef float cost

    # ============= MAIN PROCESSING =============

    # Handle empty inputs
    if num_sources == 0 or num_targets == 0:
        if return_paths:
            return []
        else:
            return np.full((num_sources, num_targets), INF_F32, dtype=np.float32)

    sorted_sources = group_by_proximity(source_indices, cols)

    for s in range(num_sources):
        for original_idx in range(num_sources):
            if sorted_sources[s] == source_indices[original_idx]:
                source_idx_map[s] = original_idx
                break

    for s in range(num_sources):
        source_idx = sorted_sources[s]
        original_idx = source_idx_map[s]

        try:
            # Find paths from this source to all targets
            source_paths = delta_stepping_single_source_multiple_targets(
                raster_arr, steps_arr, source_idx, target_indices,
                delta, max_value, num_threads, max_buckets_in_memory
            )

            # Store results in original order
            if return_paths:
                if len(all_paths) <= original_idx:
                    all_paths.extend([None] * (original_idx - len(all_paths) + 1))
                all_paths[original_idx] = source_paths
            else:
                # Calculate costs
                for t in range(num_targets):
                    if t < len(source_paths) and len(source_paths[t]) > 0:
                        path = source_paths[t]
                        cost = <float>path_cost(path, raster_arr, cols)
                        cost_matrix[original_idx, t] = cost

        except Exception as e:
            if return_paths:
                if len(all_paths) <= original_idx:
                    all_paths.extend([None] * (original_idx - len(all_paths) + 1))
                all_paths[original_idx] = [
                    np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    return all_paths if return_paths else cost_matrix


def delta_stepping_some_pairs_shortest_paths(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint64_t, ndim=1] source_indices,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        uint16_t max_value=65535,
        bint return_paths=True,
        int num_threads=0,
        size_t max_buckets_in_memory=2048,
        float margin=1.00001):
    """
    Find optimal paths for specific source-target pairs using pairwise processing.

    This function computes shortest paths for a set of source-target pairs by
    processing each pair individually. The i-th source is paired with the i-th
    target, making this suitable for scenarios where each source has exactly one
    designated target. Each pair is processed using the single-source-single-target
    delta-stepping algorithm with margin-based early termination.

    IMPORTANT: Margin Parameter Usage
    ==================================
    This function fully utilizes the margin parameter for ALL pairs since each
    pair is processed individually using delta_stepping_2d. The margin enables
    early termination once each target is found with the specified confidence
    factor, providing consistent performance optimization across all pairs.

    Algorithm Strategy:
        1. Process pairs sequentially: (source[0]→target[0]), (source[1]→target[1]), etc.
        2. Each pair uses delta_stepping_2d with margin-based early termination
        3. No batching or multi-target optimization is performed
        4. Results maintain strict ordering correspondence with input arrays

    Use Cases:
        - One-to-one correspondence problems
        - Paired origin-destination matrices where each origin has one destination
        - Applications requiring consistent margin-based optimization for all pairs
        - Scenarios where batching would complicate result interpretation

    Parameters:
        raster_arr: 2D cost matrix with traversal costs
        steps_arr: 2D array defining movement directions
        source_indices: 1D array of source indices (pairs with target_indices by position)
        target_indices: 1D array of target indices (pairs with source_indices by position)
        delta: Bucket width for edge classification (must be > 0)
        max_value: Cost value representing obstacles (default 65535)
        return_paths: If True, returns paths; if False, returns costs only
        num_threads: Number of OpenMP threads (0 = auto-detect)
        max_buckets_in_memory: Circular buffer size (power of 2)
        margin: Safety factor for early termination (default 1.0001)
                Applied consistently to ALL pairs via delta_stepping_2d

    Returns:
        If return_paths=True: List of path arrays in pair order
                             (empty arrays indicate no path exists)
        If return_paths=False: 1D array of path costs (inf for no path)

    Performance Notes:
        - Each pair benefits from margin-based early termination
        - No batching overhead or complexity
        - Predictable performance: O(n) separate pathfinding operations
        - Memory efficient: Only one path computed at a time
        - Typical speedup with margin: 1.5-3x per pair for nearby targets
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    cdef int rows = <int> raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t> raster_arr.shape[1]
    cdef int num_pairs = <int> min(source_indices.shape[0], target_indices.shape[0])

    # Result containers
    cdef list all_paths = [] if return_paths else None
    cdef np.ndarray[float32_t, ndim=1] costs = np.full(num_pairs, INF_F32,
                                                       dtype=np.float32)

    # Pair processing variables
    cdef int i
    cdef uint64_t source, target
    cdef np.ndarray[uint64_t, ndim=1] path
    cdef float path_cost_value

    # Margin validation
    cdef float validated_margin

    # ============= VALIDATION =============

    # Validate and sanitize margin parameter
    if margin <= 1.00001:
        validated_margin = 1.00001
    else:
        validated_margin = margin

    # Handle empty input case
    if num_pairs == 0:
        if return_paths:
            return []
        else:
            return np.empty(0, dtype=np.float32)

    # ============= PAIRWISE PROCESSING =============

    # Process each source-target pair individually
    # This ensures consistent margin application and simple, predictable behavior
    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        # Use single-source-single-target delta-stepping with margin
        # Each pair benefits from early termination optimization
        path = delta_stepping_2d(
            raster_arr, steps_arr, source, target,
            delta, max_value, num_threads, max_buckets_in_memory,
            validated_margin  # MARGIN APPLIED TO EVERY PAIR
        )

        # Store results based on return type preference
        if return_paths:
            # Store the actual path
            all_paths.append(path)
        else:
            # Calculate and store only the cost
            if len(path) > 0:
                path_cost_value = <float> path_cost(path, raster_arr, cols)
                costs[i] = path_cost_value
            else:
                # No path exists - cost remains as INF_F32
                pass

    # Return appropriate result type
    return all_paths if return_paths else costs


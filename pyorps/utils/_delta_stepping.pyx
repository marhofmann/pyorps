"""
Delta-stepping algorithms for parallel shortest path computation on raster grids.

Extracted from path_algorithms.pyx as the fourth module in the OO refactoring.
Contains:
- ThreadResults struct: per-thread storage for parallel edge relaxation
- PersistentState struct: shared state for persistent thread pool variants
- group_by_proximity: spatial reordering for batch processing
- ensure_bucket_size_dynamic: dynamic bucket resizing
- relax_edges_delta_stepping: parallel edge relaxation with atomic CAS
- relax_vertex_edges_inline: single-vertex edge relaxation for persistent pools
- Standard delta-stepping variants (2d, single-source-multi-target, etc.)
- Persistent delta-stepping variants (single prange call with manual barriers)
"""

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.math cimport sqrtf, abs
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free, calloc
from openmp cimport omp_get_max_threads, omp_set_num_threads
from time import monotonic as _monotonic


# Import core data structures and utilities from refactored modules
# NOTE: Must be before extern declarations since they use these types
from pyorps.utils._heap cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
    StepData, CachedStepData, SystemLimits, INF_F32,
    get_system_limits, calculate_thread_buffer_capacity,
    round_up_power_of_two, get_circular_index,
)
from pyorps.utils._raster_context cimport (
    RasterContext, check_path_cached,
    precompute_cached_steps, precompute_directions_optimized,
)
from pyorps.utils._raster_context import path_cost, create_exclude_mask


# Atomic CAS helpers for lock-free distance+predecessor updates
cdef extern from "atomic_cas.h" nogil:
    uint64_t pack_dist_pred(float dist, uint32_t pred)
    float unpack_dist(uint64_t packed)
    uint32_t unpack_pred(uint64_t packed)
    int atomic_try_update_dist_pred(volatile uint64_t* dist_pred, uint64_t v_idx,
                                     float new_dist, uint32_t new_pred)
    uint64_t atomic_load_u64(volatile uint64_t* addr)

# Barrier and fetch-add for persistent thread pool
cdef extern from "atomic_cas.h" nogil:
    int atomic_fetch_add_int(volatile int* addr, int val)
    void thread_barrier_wait(volatile int* arrive_count, volatile int* sense,
                             int num_threads, int* local_sense)


# ==================== RELAXATION BUFFER OVERFLOW PROTOCOL ====================
#
# Every parallel relaxation phase commits a distance improvement by CAS and
# then queues the improved vertex in a fixed per-thread buffer. A dropped
# queue entry is a LOST RELAXATION: the improvement is already visible in the
# distance array, but the vertex is never re-relaxed, so the returned path can
# be silently suboptimal. The buffers only fill on frontiers of 10^5+ vertices,
# i.e. at raster sizes the test suite never reaches.
#
# The protocol below is the one proven in _delta_stepping_fused.pyx: a thread
# stops claiming new work once its buffer is within one chunk's worth of
# relaxations of capacity (`guard`), and the work it did not claim rolls over
# into an extra round of the same phase. Invariant:
#
#     a chunk is claimed only while count <= guard - 1, and one chunk adds at
#     most CHUNK_SIZE * out_degree entries, so count stays < capacity.
#
# `drops` is therefore an assertable witness and must stay 0.

_relaxation_stats = {
    "capacity": 0,
    "guard": 0,
    "rollovers": 0,
    "drops": 0,
}

_relaxation_config = {"forced_capacity": 0}


def set_relaxation_buffer_capacity(int capacity):
    """Pin the per-thread relaxation buffer capacity (0 = size from system limits).

    Test hook: rollover only engages once a buffer fills, which on real
    capacities needs frontiers far larger than a unit test can build. A pinned
    capacity is still clamped up to the protocol's safety floor.
    """
    _relaxation_config["forced_capacity"] = int(capacity)


def get_relaxation_stats():
    """Relaxation-buffer bookkeeping of the most recent delta-stepping call."""
    return dict(_relaxation_stats)


# ==================== SYSTEM LIMITS MEMO ====================
#
# get_system_limits() probes psutil.virtual_memory() on every kernel entry.
# What it returns are process-wide sizing hints -- an available-memory guard,
# an iteration cap, a core count -- not per-query facts, so a short
# time-to-live keeps them honest while taking the probe out of multi-pair
# loops. The TTL matters exactly where the cost does: it only expires between
# queries slow enough for a re-probe to be free.

cdef double SYS_LIMITS_TTL_SECONDS = 1.0

cdef SystemLimits _sys_limits_value
cdef double _sys_limits_expiry = 0.0
cdef bint _sys_limits_fresh = False


cdef SystemLimits system_limits_memoized() except *:
    """get_system_limits(), re-probed at most once per TTL."""
    global _sys_limits_value, _sys_limits_expiry, _sys_limits_fresh
    cdef double now = _monotonic()
    if _sys_limits_fresh and now < _sys_limits_expiry:
        return _sys_limits_value
    _sys_limits_value = get_system_limits()
    _sys_limits_expiry = now + SYS_LIMITS_TTL_SECONDS
    _sys_limits_fresh = True
    return _sys_limits_value


def invalidate_system_limits_memo():
    """Force the next kernel entry to re-probe the system (test hook)."""
    global _sys_limits_fresh
    _sys_limits_fresh = False


# ==================== PER-RASTER WORKSPACE ====================
#
# Every kernel entry re-derives the same per-raster facts and re-allocates the
# same O(cells) state before a single edge is relaxed:
#   * the exclude mask, (raster != max_value)
#   * the P1.3 span statistic, the largest traversable cost in the raster
#   * the packed dist+pred array (8 B/cell) and the bucket stamps (4 B/cell)
# At 25 M cells that is ~450 MB of memory traffic per query, and the multi-pair
# and multi-source drivers below pay all of it once per query on a raster that
# is identical across the whole loop.
#
# KEYING -- why a stale workspace cannot be served:
#   A workspace is bound to one raster *object* plus one max_value, and the
#   only code that constructs one is the drivers in this module, which build it
#   and drop it inside a single call. There is deliberately no process-global
#   cache: cost rasters are edited in place (the GUI does exactly that) and no
#   key cheaper than the derivation itself can witness an in-place edit, so a
#   workspace that outlived its call would be unsound. Within one driver call
#   the raster provably cannot change -- the drivers never write to it and the
#   kernels bind it as `const uint16_t[:, :]`. matches() additionally pins the
#   buffer address and the cell count, so an array re-pointed or resized in
#   place is rejected rather than reused. Every top-level entry therefore
#   re-derives the mask and the span statistic from the live raster.

cdef class DeltaWorkspace:
    """Per-raster derivations and state arrays shared by one driver call.

    Not a public API: pass a workspace only to the kernel it was built for, and
    never hold one across a point where the raster could be edited.
    """

    cdef object raster
    cdef object exclude_mask
    cdef object dist_pred
    cdef object last_bucket
    cdef int64_t max_value
    cdef size_t data_ptr
    cdef uint64_t total_cells
    cdef readonly double max_traversable_cost
    cdef readonly bint has_traversable

    def __cinit__(self, np.ndarray raster_arr, int64_t max_value):
        # bool -> uint8 is exactly the 0/1 the kernels read, so the mask is a
        # zero-copy view of the comparison result rather than an astype copy.
        traversable = (raster_arr != max_value)
        self.raster = raster_arr
        self.max_value = max_value
        self.data_ptr = <size_t><void*>raster_arr.data
        self.total_cells = (<uint64_t>raster_arr.shape[0] *
                            <uint64_t>raster_arr.shape[1])
        self.exclude_mask = traversable.view(np.uint8)
        self.dist_pred = None
        self.last_bucket = None
        self.has_traversable = <bint>traversable.any()
        if self.has_traversable:
            # Equal to np.max(raster_arr[traversable]): costs are uint16 >= 0,
            # so the identity element 0 can never win. Avoids materialising the
            # fancy-index copy of every traversable cell.
            self.max_traversable_cost = <double>np.max(raster_arr, initial=0,
                                                       where=traversable)
        else:
            self.max_traversable_cost = 0.0

    cdef bint matches(self, np.ndarray raster_arr, int64_t max_value):
        return (raster_arr is self.raster and
                max_value == self.max_value and
                <size_t><void*>raster_arr.data == self.data_ptr and
                (<uint64_t>raster_arr.shape[0] *
                 <uint64_t>raster_arr.shape[1]) == self.total_cells)

    cdef object take_dist_pred(self, uint64_t init_packed, uint64_t source_idx):
        """Packed dist+pred array with every element re-initialised.

        fill() writes the whole array, so the reset is complete by
        construction: there is no touched-region bookkeeping that could leave a
        settled label from the previous query behind.
        """
        cdef object arr = self.dist_pred
        if arr is None:
            arr = np.empty(<size_t>self.total_cells, dtype=np.uint64)
            self.dist_pred = arr
        arr.fill(init_packed)
        arr[source_idx] = pack_dist_pred(0.0, 0xFFFFFFFF)
        return arr

    cdef object take_last_bucket(self):
        """Bucket stamps with every element reset to -1 (see take_dist_pred)."""
        cdef object arr = self.last_bucket
        if arr is None:
            arr = np.empty(<size_t>self.total_cells, dtype=np.int32)
            self.last_bucket = arr
        arr.fill(-1)
        return arr


cdef inline DeltaWorkspace bind_workspace(DeltaWorkspace workspace,
                                          np.ndarray raster_arr,
                                          int64_t max_value):
    """The caller's workspace if it belongs to this raster, else a fresh one."""
    if workspace is not None and workspace.matches(raster_arr, max_value):
        return workspace
    return DeltaWorkspace(raster_arr, max_value)


# Work-claim granularity of relax_edges_delta_stepping(). The persistent
# kernels declare their own CHUNK_SIZE with the same value.
cdef enum:
    RELAX_CHUNK_SIZE = 64


cdef inline int relax_guard_slack(int out_degree, int chunk_size) noexcept nogil:
    """Buffer headroom one claimed chunk can consume.

    A chunk holds chunk_size vertices, each contributing at most out_degree
    successful relaxations after the guard was last tested.
    """
    return chunk_size * out_degree + 64


cdef inline int size_relax_buffer(uint64_t total_cells, int threads,
                                  int guard_slack, SystemLimits* limits,
                                  int forced) noexcept nogil:
    """Per-thread buffer capacity, never below twice the guard slack."""
    cdef int capacity = calculate_thread_buffer_capacity(
        total_cells, threads, limits)
    if forced > 0 and forced < capacity:
        capacity = forced
    if capacity < 2 * guard_slack:
        capacity = 2 * guard_slack
    return capacity


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
    #     guard: Fill level at which the thread stops claiming new work
    #     overflow: Dropped relaxations; must stay 0 (see protocol note above)

    uint64_t *vertices
    uint32_t *bucket_indices
    float *distances
    int count
    int capacity
    int guard
    int overflow


# Shared claim counter for the chunked relaxation loop. A struct member is
# addressable from inside prange without triggering Cython's reduction-variable
# analysis (same pattern as PersistentState below).
cdef struct RelaxWork:
    int work_idx

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

# ==================== PERSISTENT THREAD POOL PRIMITIVES ====================

cdef inline void relax_vertex_edges_inline(
        uint64_t u,
        int tid,
        uint64_t* dist_pred,
        const uint16_t[:, :] raster,
        const uint8_t[:, :] exclude_mask,
        const vector[StepData]& directions,
        const vector[CachedStepData]& cached_steps,
        int rows,
        uint64_t cols,
        float delta,
        bint light_phase_only,
        ThreadResults* thread_results,
        uint64_t total_cells,
        SystemLimits* limits) noexcept nogil:
    """
    Process edges for a single vertex. Extracted from the prange body of
    relax_edges_delta_stepping() for use in the persistent thread pool.

    This is the same logic as the original function,
    but operates on a single vertex instead of iterating over a vector.
    """
    cdef int dir_idx, ur, uc, vr, vc
    cdef uint64_t v, ur64, uc64, vr64, vc64
    cdef size_t bucket_idx_temp
    cdef uint32_t bucket_idx_stored
    cdef float current_dist, edge_weight, new_dist, intermediate_cost
    cdef float raster_ur_uc, raster_vr_vc
    cdef int should_update
    cdef int valid_path

    if u >= total_cells:
        return

    # Convert to 2D coordinates
    ur64 = u // cols
    uc64 = u - (ur64 * cols)
    ur = <int>ur64
    uc = <int>uc64

    # Atomic load of current distance
    current_dist = unpack_dist(atomic_load_u64(&dist_pred[u]))

    if current_dist >= INF_F32:
        return

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

        # Lock-free CAS update of distance + predecessor
        should_update = atomic_try_update_dist_pred(
            <volatile uint64_t*>dist_pred, v, new_dist, <uint32_t>u
        )

        # Add to thread-local buffer for bucket insertion
        if should_update:
            if thread_results[tid].count < thread_results[tid].capacity:
                bucket_idx_temp = <size_t>(new_dist / delta)

                if bucket_idx_temp >= limits.max_buckets:
                    bucket_idx_stored = limits.max_buckets - 1
                else:
                    bucket_idx_stored = <uint32_t>bucket_idx_temp

                thread_results[tid].vertices[thread_results[tid].count] = v
                thread_results[tid].bucket_indices[thread_results[tid].count] = bucket_idx_stored
                thread_results[tid].distances[thread_results[tid].count] = new_dist
                thread_results[tid].count += 1
            else:
                # Unreachable while the callers honour `guard`; counted so the
                # invariant can be asserted instead of assumed.
                thread_results[tid].overflow += 1


# ==================== DELTA-STEPPING EDGE RELAXATION ====================

cdef int relax_edges_delta_stepping(vector[uint64_t]& vertices,
                                    int start_idx,
                                    uint64_t* dist_pred,
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
                                    uint64_t total_cells,
                                    uint64_t target_idx,
                                    SystemLimits* limits) noexcept nogil:
    """
    Parallel edge relaxation for delta-stepping algorithm.

    Uses lock-free atomic CAS on a packed dist+pred uint64 array
    instead of mutex locks. IEEE 754 positive floats preserve integer
    ordering, so packed comparisons are equivalent to distance comparisons.

    Threads claim CHUNK_SIZE-sized slices of vertices[start_idx:] through an
    atomic counter and stop claiming once their result buffer reaches `guard`,
    so no successful relaxation is ever dropped. Whatever was left unclaimed is
    reported back to the caller, which merges and calls again from there.

    Parameters:
        vertices: Current set of vertices to relax edges from
        start_idx: First vertex to process (rollover resume point)
        dist_pred: Packed distance+predecessor array (lock-free via CAS)
        raster: Cost raster for edge weight calculation
        exclude_mask: Traversability mask
        directions: Precomputed movement directions
        cached_steps: Cached intermediate steps for each direction
        rows, cols: Raster dimensions
        delta: Bucket width for edge classification
        light_phase_only: True for light edges, False for heavy edges
        thread_results: Per-thread accumulation buffers
        num_threads: Active thread count
        total_cells: Total number of cells in raster
        target_idx: Target index (unused, kept for API consistency)
        limits: System resource constraints

    Returns:
        Index of the first vertex that was NOT processed (== vertices.size()
        when the batch completed).
    """
    cdef RelaxWork rw_data
    cdef RelaxWork* rw = &rw_data
    cdef int CHUNK_SIZE = RELAX_CHUNK_SIZE
    cdef int n_vertices = <int>vertices.size()
    cdef int tid, chunk_start, chunk_end, j
    cdef int omp_team = num_threads

    rw.work_idx = start_idx

    for tid in prange(omp_team, schedule='static', num_threads=omp_team):
        while True:
            if thread_results[tid].count >= thread_results[tid].guard:
                break
            chunk_start = atomic_fetch_add_int(
                <volatile int*>&rw.work_idx, CHUNK_SIZE)
            if chunk_start >= n_vertices:
                break
            chunk_end = chunk_start + CHUNK_SIZE
            if chunk_end > n_vertices:
                chunk_end = n_vertices
            for j in range(chunk_start, chunk_end):
                relax_vertex_edges_inline(
                    vertices[j], tid,
                    dist_pred, raster, exclude_mask,
                    directions, cached_steps,
                    rows, cols, delta, light_phase_only,
                    thread_results, total_cells, limits
                )

    if rw.work_idx > n_vertices:
        return n_vertices
    return rw.work_idx


# Shared mutable state struct for persistent thread pool.
# Accessed via pointer inside prange to avoid Cython reduction variable analysis.
cdef struct PersistentState:
    size_t current_logical_bucket
    size_t logical_bucket_count
    size_t physical_bucket_idx
    size_t window_start
    int done_flag
    int barrier_arrive_count
    int barrier_sense
    int work_idx
    int n_vertices
    int light_iterations
    int light_phase_active
    int heavy_phase_active
    int bucket_valid
    int target_found_flag
    float target_distance
    float cutoff_distance
    int targets_found
    float max_target_distance
    int rollovers
    uint64_t* vertices_ptr


# ==================== DELTA-STEPPING ALGORITHMS ====================

def delta_stepping_2d(np.ndarray[uint16_t, ndim=2] raster_arr,
                      np.ndarray[int8_t, ndim=2] steps_arr,
                      uint64_t source_idx, uint64_t target_idx,
                      float delta,
                      int64_t max_value=65535,
                      int num_threads=0,
                      size_t max_buckets_in_memory=2048,
                      float margin=1.00001,
                      DeltaWorkspace workspace=None):
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
           - Light phase: Relax edges with weight <= delta (can be done multiple times)
           - Heavy phase: Relax edges with weight > delta (once per bucket)
        3. Early termination when current_bucket * delta > target_distance * margin
        4. Reconstruct path using predecessor array

    References:
    [1] Meyer, U., Sanders, P.: delta-stepping: a parallelizable shortest path algorithm. J.
        Algorithms 49, 1 (2003), 114-152.
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
        workspace: Optional DeltaWorkspace built for this exact raster object
                and max_value; ignored (and re-derived) if it does not match

    Returns:
        1D numpy array (uint64) of linear indices representing the optimal path.
        Empty array if no path exists.

    Performance Characteristics:
        - Time complexity: O((V + E) / p + D*L) where p=parallelism, D=diameter
        - Memory: O(V) for distance/predecessor + O(B) for circular buffer
        - Early termination reduces average case significantly
        - Typical speedup: 2-10x over Dijkstra for appropriate delta values
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    # System and problem dimensions
    cdef SystemLimits sys_limits = system_limits_memoized()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols

    # Preprocessing variables
    cdef DeltaWorkspace ws
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
    cdef int max_light_iterations = 10000
    cdef uint64_t v, vertex_to_add
    cdef size_t new_logical_bucket, new_physical_bucket
    cdef float new_dist
    cdef int32_t last_bucket_for_vertex

    # Path reconstruction
    cdef uint64_t current, path_length
    cdef uint32_t pred_val
    cdef list path_vertices

    # Preprocessing data structures
    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions

    # Packed distance+predecessor array
    cdef np.ndarray[uint64_t, ndim=1] dist_pred_arr
    cdef uint64_t* dist_pred_ptr
    cdef uint64_t init_packed

    # Validation variables
    cdef uint64_t source_r, source_c, target_r, target_c

    # Loop variables
    cdef int i, j

    # Relaxation buffer overflow protocol
    cdef int out_degree, guard_slack, resume_idx
    cdef int forced_capacity = <int>_relaxation_config["forced_capacity"]
    cdef int64_t rollovers = 0
    cdef int64_t drops = 0

    # ============= VALIDATION =============

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")

    if source_idx >= total_cells or target_idx >= total_cells:
        return np.empty(0, dtype=np.uint64)

    # ============= PREPROCESSING =============

    ws = bind_workspace(workspace, raster_arr, max_value)
    exclude_mask_arr = ws.exclude_mask

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

    # Precompute movement data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize packed distance+predecessor array
    if total_cells > 0xFFFFFFFF:
        raise OverflowError(
            f"Raster has {total_cells} cells, exceeding uint32 predecessor limit (4294967295)")
    init_packed = pack_dist_pred(INF_F32, 0xFFFFFFFF)
    dist_pred_arr = ws.take_dist_pred(init_packed, source_idx)
    dist_pred_ptr = <uint64_t*>dist_pred_arr.data

    # Create memory views
    raster_view = raster_arr
    exclude_view = exclude_mask_arr

    # Initialize circular buffer with fixed size
    circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    buffer_mask = circular_buffer_size - 1
    buckets.resize(circular_buffer_size)

    # P1.3 fix: Validate circular buffer can hold max bucket span
    cdef double _max_step_dist = 0.0
    cdef double _sd, _dr_f, _dc_f, _max_span
    cdef int _si
    for _si in range(steps_arr.shape[0]):
        _dr_f = <double>steps_arr[_si, 0]
        _dc_f = <double>steps_arr[_si, 1]
        _sd = (_dr_f * _dr_f + _dc_f * _dc_f) ** 0.5
        if _sd > _max_step_dist:
            _max_step_dist = _sd
    if ws.has_traversable:
        _max_span = ws.max_traversable_cost * _max_step_dist / computed_delta
        if _max_span >= <double>circular_buffer_size:
            raise ValueError(
                f"Delta-stepping: max edge/delta ratio ({_max_span:.0f}) "
                f"exceeds circular buffer size ({circular_buffer_size}). "
                f"Increase max_buckets_in_memory or delta.")


    # Initialize last bucket tracking
    last_bucket_arr = ws.take_last_bucket()
    last_bucket = last_bucket_arr

    # Add source to first bucket
    physical_bucket_idx = get_circular_index(0, circular_buffer_size)
    buckets[physical_bucket_idx].push_back(source_idx)
    last_bucket[source_idx] = 0

    # Allocate thread buffers
    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        raise MemoryError("Could not allocate thread data")

    out_degree = <int>directions.size()
    guard_slack = relax_guard_slack(out_degree, RELAX_CHUNK_SIZE)
    max_capacity = size_relax_buffer(total_cells, actual_threads, guard_slack,
                                     &sys_limits, forced_capacity)

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
            raise MemoryError("Could not allocate thread storage")

        thread_results[tid].capacity = max_capacity
        thread_results[tid].guard = max_capacity - guard_slack
        thread_results[tid].count = 0
        thread_results[tid].overflow = 0

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
                # Lost-relaxation fix: popped vertices are no longer queued
                # anywhere, so clear their dedup stamp -- a later improvement
                # landing in this same bucket must be able to re-queue them.
                for i in range(<int>current_vertices.size()):
                    last_bucket[current_vertices[i]] = -1

                settled_vertices.insert(settled_vertices.end(),
                                       current_vertices.begin(),
                                       current_vertices.end())

                # Rollover: relax until every vertex of this batch was claimed;
                # a round ends early when a thread buffer reaches its guard.
                resume_idx = 0
                while True:
                    for tid in range(actual_threads):
                        thread_results[tid].count = 0

                    resume_idx = relax_edges_delta_stepping(
                        current_vertices, resume_idx,
                        dist_pred_ptr,
                        raster_view, exclude_view,
                        directions, cached_steps,
                        rows, cols, computed_delta, True,  # light_phase_only
                        thread_results, actual_threads, total_cells,
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

                    if resume_idx >= <int>current_vertices.size():
                        break
                    rollovers += 1

            # Check if target found
            for i in range(<int>settled_vertices.size()):
                if settled_vertices[i] == target_idx:
                    target_found = True
                    target_distance = unpack_dist(dist_pred_ptr[target_idx])
                    cutoff_distance = target_distance * termination_margin
                    break

            # Early termination
            if target_found and current_logical_bucket * computed_delta > cutoff_distance:
                break

            # HEAVY PHASE
            if not settled_vertices.empty():
                resume_idx = 0
                while True:
                    for tid in range(actual_threads):
                        thread_results[tid].count = 0

                    resume_idx = relax_edges_delta_stepping(
                        settled_vertices, resume_idx,
                        dist_pred_ptr,
                        raster_view, exclude_view,
                        directions, cached_steps,
                        rows, cols, computed_delta, False,  # heavy edges
                        thread_results, actual_threads, total_cells,
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

                    if resume_idx >= <int>settled_vertices.size():
                        break
                    rollovers += 1

            # Clear processed bucket to free memory
            buckets[physical_bucket_idx].clear()
            buckets[physical_bucket_idx].shrink_to_fit()

            current_logical_bucket += 1

    finally:
        # Cleanup resources (no locks to destroy)
        if thread_results != NULL:
            for tid in range(actual_threads):
                drops += thread_results[tid].overflow
                if thread_results[tid].vertices != NULL:
                    free(thread_results[tid].vertices)
                if thread_results[tid].bucket_indices != NULL:
                    free(thread_results[tid].bucket_indices)
                if thread_results[tid].distances != NULL:
                    free(thread_results[tid].distances)
            free(thread_results)
        _relaxation_stats["capacity"] = int(max_capacity)
        _relaxation_stats["guard"] = int(max_capacity - guard_slack)
        _relaxation_stats["rollovers"] = int(rollovers)
        _relaxation_stats["drops"] = int(drops)

    # Path reconstruction
    pred_val = unpack_pred(dist_pred_ptr[target_idx])
    if not target_found or pred_val == 0xFFFFFFFF:
        if source_idx == target_idx:
            return np.array([source_idx], dtype=np.uint64)
        return np.empty(0, dtype=np.uint64)

    path_vertices = []
    current = target_idx
    path_length = 0

    while path_length < sys_limits.max_path_length:
        path_vertices.append(current)
        if current == source_idx:
            break
        pred_val = unpack_pred(dist_pred_ptr[current])
        if pred_val == 0xFFFFFFFF:
            return np.empty(0, dtype=np.uint64)
        current = <uint64_t>pred_val
        path_length += 1

    path_vertices.reverse()

    return np.array(path_vertices, dtype=np.uint64)

def delta_stepping_single_source_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint64_t source_idx,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        int64_t max_value=65535,
        int num_threads=0,
        size_t max_buckets_in_memory=2048,
        DeltaWorkspace workspace=None):
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
        workspace: Optional DeltaWorkspace built for this exact raster object
            and max_value; ignored (and re-derived) if it does not match

    Returns:
        List of numpy arrays, one path per target (empty if no path exists)

    Performance:
        - Single traversal for all targets vs. N separate searches
        - Typical speedup: 5-15x for 10+ targets
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    # System and problem dimensions
    cdef SystemLimits sys_limits = system_limits_memoized()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols
    cdef int num_targets = <int>target_indices.shape[0]

    # Preprocessing variables
    cdef DeltaWorkspace ws
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
    cdef int max_light_iterations = 10000
    cdef uint64_t v, vertex_to_add, current_vertex
    cdef size_t new_logical_bucket, new_physical_bucket
    cdef float new_dist
    cdef int32_t last_bucket_for_vertex

    # Path reconstruction
    cdef uint64_t target_idx, path_length
    cdef uint32_t pred_val
    cdef list path_vertices
    cdef list paths = []

    # Preprocessing data structures
    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions

    # Packed distance+predecessor array
    cdef np.ndarray[uint64_t, ndim=1] dist_pred_arr
    cdef uint64_t* dist_pred_ptr
    cdef uint64_t init_packed

    # Validation variables
    cdef uint64_t source_r, source_c

    # Loop variables
    cdef int i, j

    # Relaxation buffer overflow protocol
    cdef int out_degree, guard_slack, resume_idx
    cdef int forced_capacity = <int>_relaxation_config["forced_capacity"]
    cdef int64_t rollovers = 0
    cdef int64_t drops = 0

    # ============= VALIDATION =============

    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if num_targets == 0:
        return []

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")

    if total_cells > 0xFFFFFFFF:
        raise OverflowError(
            f"Raster has {total_cells} cells, exceeding uint32 predecessor limit (4294967295)")

    # Thread configuration
    if num_threads <= 0:
        num_threads = min(sys_limits.num_cores, omp_get_max_threads())
    omp_set_num_threads(num_threads)
    actual_threads = omp_get_max_threads()

    # Validate indices
    if source_idx >= total_cells:
        return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    for i in range(num_targets):
        if target_indices[i] >= total_cells:
            return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    # Create traversability mask
    ws = bind_workspace(workspace, raster_arr, max_value)
    exclude_mask_arr = ws.exclude_mask
    source_r = source_idx // cols
    source_c = source_idx % cols

    if exclude_mask_arr[source_r, source_c] == 0:
        return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    # Precompute movement data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize packed distance+predecessor array
    init_packed = pack_dist_pred(INF_F32, 0xFFFFFFFF)
    dist_pred_arr = ws.take_dist_pred(init_packed, source_idx)
    dist_pred_ptr = <uint64_t*>dist_pred_arr.data
    target_found_arr = np.zeros(num_targets, dtype=np.uint8)

    # Create memory views
    raster_view = raster_arr
    exclude_view = exclude_mask_arr
    target_found = target_found_arr

    # Initialize circular buffer
    circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    buffer_mask = circular_buffer_size - 1
    buckets.resize(circular_buffer_size)

    # P1.3 fix: Validate circular buffer can hold max bucket span
    cdef double _max_step_dist = 0.0
    cdef double _sd, _dr_f, _dc_f, _max_span
    cdef int _si
    for _si in range(steps_arr.shape[0]):
        _dr_f = <double>steps_arr[_si, 0]
        _dc_f = <double>steps_arr[_si, 1]
        _sd = (_dr_f * _dr_f + _dc_f * _dc_f) ** 0.5
        if _sd > _max_step_dist:
            _max_step_dist = _sd
    if ws.has_traversable:
        _max_span = ws.max_traversable_cost * _max_step_dist / delta
        if _max_span >= <double>circular_buffer_size:
            raise ValueError(
                f"Delta-stepping: max edge/delta ratio ({_max_span:.0f}) "
                f"exceeds circular buffer size ({circular_buffer_size}). "
                f"Increase max_buckets_in_memory or delta.")


    # Initialize last bucket tracking
    last_bucket_arr = ws.take_last_bucket()
    last_bucket = last_bucket_arr

    # Add source to first bucket
    physical_bucket_idx = get_circular_index(0, circular_buffer_size)
    buckets[physical_bucket_idx].push_back(source_idx)
    last_bucket[source_idx] = 0

    # Allocate thread buffers
    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        raise MemoryError("Could not allocate thread data")

    out_degree = <int>directions.size()
    guard_slack = relax_guard_slack(out_degree, RELAX_CHUNK_SIZE)
    max_capacity = size_relax_buffer(total_cells, actual_threads, guard_slack,
                                     &sys_limits, forced_capacity)

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
            raise MemoryError("Could not allocate thread storage")

        thread_results[tid].capacity = max_capacity
        thread_results[tid].guard = max_capacity - guard_slack
        thread_results[tid].count = 0
        thread_results[tid].overflow = 0

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
                # Lost-relaxation fix: popped vertices are no longer queued
                # anywhere, so clear their dedup stamp -- a later improvement
                # landing in this same bucket must be able to re-queue them.
                for i in range(<int>current_vertices.size()):
                    last_bucket[current_vertices[i]] = -1

                settled_vertices.insert(settled_vertices.end(),
                                       current_vertices.begin(),
                                       current_vertices.end())

                # Rollover: relax until every vertex of this batch was claimed;
                # a round ends early when a thread buffer reaches its guard.
                resume_idx = 0
                while True:
                    for tid in range(actual_threads):
                        thread_results[tid].count = 0

                    resume_idx = relax_edges_delta_stepping(
                        current_vertices, resume_idx,
                        dist_pred_ptr,
                        raster_view, exclude_view,
                        directions, cached_steps,
                        rows, cols, delta, True,  # light_phase_only
                        thread_results, actual_threads, total_cells,
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

                    if resume_idx >= <int>current_vertices.size():
                        break
                    rollovers += 1

            # Check if any targets were settled
            for i in range(<int>settled_vertices.size()):
                current_vertex = settled_vertices[i]
                for j in range(num_targets):
                    if current_vertex == target_indices[j] and target_found[j] == 0:
                        target_found[j] = 1
                        targets_found += 1

                        # Track maximum distance for information (but don't terminate)
                        current_target_distance = unpack_dist(dist_pred_ptr[current_vertex])
                        if current_target_distance > max_target_distance:
                            max_target_distance = current_target_distance

            # Only terminate when ALL targets are found
            if targets_found >= num_targets:
                break

            # HEAVY PHASE
            if not settled_vertices.empty():
                resume_idx = 0
                while True:
                    for tid in range(actual_threads):
                        thread_results[tid].count = 0

                    resume_idx = relax_edges_delta_stepping(
                        settled_vertices, resume_idx,
                        dist_pred_ptr,
                        raster_view, exclude_view,
                        directions, cached_steps,
                        rows, cols, delta, False,  # heavy edges
                        thread_results, actual_threads, total_cells,
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

                    if resume_idx >= <int>settled_vertices.size():
                        break
                    rollovers += 1

            # Clear processed bucket
            buckets[physical_bucket_idx].clear()
            buckets[physical_bucket_idx].shrink_to_fit()

            current_logical_bucket += 1

    finally:
        # Cleanup resources (no locks to destroy)
        if thread_results != NULL:
            for tid in range(actual_threads):
                drops += thread_results[tid].overflow
                free(thread_results[tid].vertices)
                free(thread_results[tid].bucket_indices)
                free(thread_results[tid].distances)
            free(thread_results)
        _relaxation_stats["capacity"] = int(max_capacity)
        _relaxation_stats["guard"] = int(max_capacity - guard_slack)
        _relaxation_stats["rollovers"] = int(rollovers)
        _relaxation_stats["drops"] = int(drops)

    # Reconstruct paths for all targets
    for i in range(num_targets):
        target_idx = target_indices[i]

        pred_val = unpack_pred(dist_pred_ptr[target_idx])
        if pred_val == 0xFFFFFFFF:
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
            pred_val = unpack_pred(dist_pred_ptr[current_vertex])
            if pred_val == 0xFFFFFFFF:
                paths.append(np.empty(0, dtype=np.uint64))
                break
            current_vertex = <uint64_t>pred_val
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
        int64_t max_value=65535,
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

    2. Users expect a complete M x N result matrix where M=sources, N=targets.
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
    cdef DeltaWorkspace workspace = None

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

    # The raster is the same for every source and nothing here writes to it, so
    # its derivations and state arrays are derived once for the whole loop.
    workspace = DeltaWorkspace(raster_arr, max_value)

    for s in range(num_sources):
        source_idx = sorted_sources[s]
        original_idx = source_idx_map[s]

        try:
            # Find paths from this source to all targets
            source_paths = delta_stepping_single_source_multiple_targets(
                raster_arr, steps_arr, source_idx, target_indices,
                delta, max_value, num_threads, max_buckets_in_memory,
                workspace
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
        int64_t max_value=65535,
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
        1. Process pairs sequentially: (source[0]->target[0]), (source[1]->target[1]), etc.
        2. Each pair uses delta_stepping_2d with margin-based early termination
        3. No batching or multi-target optimization is performed
        4. Results maintain strict ordering correspondence with input arrays

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

    # Per-raster derivations shared by every pair
    cdef DeltaWorkspace workspace = None

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

    # The raster is the same for every pair and nothing here writes to it, so
    # its derivations and state arrays are derived once for the whole loop.
    workspace = DeltaWorkspace(raster_arr, max_value)

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
            validated_margin,  # MARGIN APPLIED TO EVERY PAIR
            workspace
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


# ==================== PERSISTENT DELTA-STEPPING (SINGLE TARGET) ====================

def delta_stepping_2d_persistent(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint64_t source_idx, uint64_t target_idx,
        float delta,
        int64_t max_value=65535,
        int num_threads=0,
        size_t max_buckets_in_memory=2048,
        float margin=1.00001,
        DeltaWorkspace workspace=None):
    """
    Persistent-thread-pool variant of delta_stepping_2d.

    Uses a single prange(num_threads) call for the entire algorithm,
    with sense-reversing barriers for phase synchronization and
    atomic fetch-and-add for dynamic work distribution.

    Eliminates all OpenMP fork/join overhead -- only one thread team
    creation/destruction per algorithm invocation.

    Parameters: same as delta_stepping_2d
    Returns: same as delta_stepping_2d
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    # System and problem dimensions
    cdef SystemLimits sys_limits = system_limits_memoized()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols

    # Preprocessing variables
    cdef DeltaWorkspace ws
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
    cdef int max_light_iterations = 10000
    cdef uint64_t v, vertex_to_add
    cdef size_t new_logical_bucket, new_physical_bucket
    cdef float new_dist
    cdef int32_t last_bucket_for_vertex

    # Path reconstruction
    cdef uint64_t current, path_length
    cdef uint32_t pred_val
    cdef list path_vertices

    # Preprocessing data structures
    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions

    # Packed distance+predecessor array
    cdef np.ndarray[uint64_t, ndim=1] dist_pred_arr
    cdef uint64_t* dist_pred_ptr
    cdef uint64_t init_packed

    # Validation variables
    cdef uint64_t source_r, source_c, target_r, target_c

    # Loop variables
    cdef int i, j

    # Persistent thread pool state (struct avoids Cython reduction variable issues)
    cdef PersistentState ps_data
    cdef PersistentState* ps = &ps_data
    cdef int CHUNK_SIZE = 64
    cdef int chunk_start, chunk_end
    cdef int local_sense

    # Relaxation buffer overflow protocol
    cdef int out_degree, guard_slack
    cdef int forced_capacity = <int>_relaxation_config["forced_capacity"]
    cdef int64_t drops = 0

    # ============= VALIDATION (same as original) =============

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")

    if source_idx >= total_cells or target_idx >= total_cells:
        return np.empty(0, dtype=np.uint64)

    # ============= PREPROCESSING (same as original) =============

    ws = bind_workspace(workspace, raster_arr, max_value)
    exclude_mask_arr = ws.exclude_mask

    if delta <= 0.0:
        raise ValueError(f"Invalid delta value: {delta}! Choose a delta > 0.0!")
    computed_delta = delta

    if margin <= 1.00001:
        termination_margin = 1.00001
    else:
        termination_margin = margin

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

    # Precompute movement data
    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    # Initialize packed distance+predecessor array
    if total_cells > 0xFFFFFFFF:
        raise OverflowError(
            f"Raster has {total_cells} cells, exceeding uint32 predecessor limit (4294967295)")
    init_packed = pack_dist_pred(INF_F32, 0xFFFFFFFF)
    dist_pred_arr = ws.take_dist_pred(init_packed, source_idx)
    dist_pred_ptr = <uint64_t*>dist_pred_arr.data

    # Create memory views
    raster_view = raster_arr
    exclude_view = exclude_mask_arr

    # Initialize circular buffer with fixed size
    circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    buffer_mask = circular_buffer_size - 1
    buckets.resize(circular_buffer_size)

    # P1.3 fix: Validate circular buffer can hold max bucket span
    cdef double _max_step_dist = 0.0
    cdef double _sd, _dr_f, _dc_f, _max_span
    cdef int _si
    for _si in range(steps_arr.shape[0]):
        _dr_f = <double>steps_arr[_si, 0]
        _dc_f = <double>steps_arr[_si, 1]
        _sd = (_dr_f * _dr_f + _dc_f * _dc_f) ** 0.5
        if _sd > _max_step_dist:
            _max_step_dist = _sd
    if ws.has_traversable:
        _max_span = ws.max_traversable_cost * _max_step_dist / computed_delta
        if _max_span >= <double>circular_buffer_size:
            raise ValueError(
                f"Delta-stepping: max edge/delta ratio ({_max_span:.0f}) "
                f"exceeds circular buffer size ({circular_buffer_size}). "
                f"Increase max_buckets_in_memory or delta.")


    # Initialize last bucket tracking
    last_bucket_arr = ws.take_last_bucket()
    last_bucket = last_bucket_arr

    # Add source to first bucket
    physical_bucket_idx = get_circular_index(0, circular_buffer_size)
    buckets[physical_bucket_idx].push_back(source_idx)
    last_bucket[source_idx] = 0

    # Allocate thread buffers
    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        raise MemoryError("Could not allocate thread data")

    out_degree = <int>directions.size()
    guard_slack = relax_guard_slack(out_degree, CHUNK_SIZE)
    max_capacity = size_relax_buffer(total_cells, actual_threads, guard_slack,
                                     &sys_limits, forced_capacity)

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
            raise MemoryError("Could not allocate thread storage")

        thread_results[tid].capacity = max_capacity
        thread_results[tid].guard = max_capacity - guard_slack
        thread_results[tid].count = 0
        thread_results[tid].overflow = 0

    # ============= MAIN PERSISTENT DELTA-STEPPING LOOP =============

    # Initialize persistent state
    ps.current_logical_bucket = 0
    ps.logical_bucket_count = 0
    ps.physical_bucket_idx = 0
    ps.window_start = 0
    ps.done_flag = 0
    ps.barrier_arrive_count = 0
    ps.barrier_sense = 0
    ps.work_idx = 0
    ps.n_vertices = 0
    ps.light_iterations = 0
    ps.light_phase_active = 0
    ps.heavy_phase_active = 0
    ps.bucket_valid = 0
    ps.target_found_flag = 0
    ps.target_distance = INF_F32
    ps.cutoff_distance = INF_F32
    ps.rollovers = 0
    ps.vertices_ptr = NULL

    try:
        if actual_threads <= 1:
            # ---- SINGLE-THREAD FAST PATH (no barriers needed) ----
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

                # LIGHT PHASE
                light_iterations = 0
                while not buckets[physical_bucket_idx].empty() and light_iterations < max_light_iterations:
                    light_iterations += 1
                    current_vertices = buckets[physical_bucket_idx]
                    buckets[physical_bucket_idx].clear()
                    # Lost-relaxation fix (see above): clear dedup stamps
                    for i in range(<int>current_vertices.size()):
                        last_bucket[current_vertices[i]] = -1
                    settled_vertices.insert(settled_vertices.end(),
                                           current_vertices.begin(),
                                           current_vertices.end())

                    # Relax in guard-bounded slices, merging between slices so
                    # the buffer can never overflow.
                    j = 0
                    while j < <int>current_vertices.size():
                        thread_results[0].count = 0
                        while (j < <int>current_vertices.size() and
                               thread_results[0].count < thread_results[0].guard):
                            relax_vertex_edges_inline(
                                current_vertices[j], 0,
                                dist_pred_ptr, raster_view, exclude_view,
                                directions, cached_steps,
                                rows, cols, computed_delta, True,
                                thread_results, total_cells, &sys_limits
                            )
                            j += 1

                        # Merge results
                        for i in range(thread_results[0].count):
                            vertex_to_add = thread_results[0].vertices[i]
                            new_dist = thread_results[0].distances[i]
                            new_logical_bucket = <size_t>(new_dist / computed_delta)

                            if new_logical_bucket < window_start + circular_buffer_size:
                                new_physical_bucket = get_circular_index(new_logical_bucket, circular_buffer_size)
                                last_bucket_for_vertex = last_bucket[vertex_to_add]
                                if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                    buckets[new_physical_bucket].push_back(vertex_to_add)
                                    last_bucket[vertex_to_add] = <int32_t>new_logical_bucket
                                if new_logical_bucket >= logical_bucket_count:
                                    logical_bucket_count = new_logical_bucket + 1

                        if j < <int>current_vertices.size():
                            ps.rollovers += 1

                # Check target
                for i in range(<int>settled_vertices.size()):
                    if settled_vertices[i] == target_idx:
                        target_found = True
                        target_distance = unpack_dist(dist_pred_ptr[target_idx])
                        cutoff_distance = target_distance * termination_margin
                        break

                if target_found and current_logical_bucket * computed_delta > cutoff_distance:
                    break

                # HEAVY PHASE
                if not settled_vertices.empty():
                    j = 0
                    while j < <int>settled_vertices.size():
                        thread_results[0].count = 0
                        while (j < <int>settled_vertices.size() and
                               thread_results[0].count < thread_results[0].guard):
                            relax_vertex_edges_inline(
                                settled_vertices[j], 0,
                                dist_pred_ptr, raster_view, exclude_view,
                                directions, cached_steps,
                                rows, cols, computed_delta, False,
                                thread_results, total_cells, &sys_limits
                            )
                            j += 1

                        for i in range(thread_results[0].count):
                            vertex_to_add = thread_results[0].vertices[i]
                            new_dist = thread_results[0].distances[i]
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

                        if j < <int>settled_vertices.size():
                            ps.rollovers += 1

                buckets[physical_bucket_idx].clear()
                buckets[physical_bucket_idx].shrink_to_fit()
                current_logical_bucket += 1

        else:
            # ---- MULTI-THREAD PERSISTENT LOOP ----
            # All shared mutable scalars accessed through ps pointer
            # to avoid Cython reduction variable analysis in prange.

            for tid in prange(actual_threads, schedule='static', nogil=True,
                              num_threads=actual_threads):
                local_sense = 0

                while True:
                    # ======= PHASE 1: Thread 0 finds next bucket & prepares work =======
                    if tid == 0:
                        ps.work_idx = 0
                        ps.n_vertices = 0
                        ps.done_flag = 0
                        ps.light_phase_active = 0

                        ps.bucket_valid = 0
                        while ps.current_logical_bucket < ps.logical_bucket_count + circular_buffer_size:
                            ps.physical_bucket_idx = get_circular_index(
                                ps.current_logical_bucket, circular_buffer_size)
                            if ps.current_logical_bucket >= ps.logical_bucket_count + circular_buffer_size:
                                break
                            if not buckets[ps.physical_bucket_idx].empty():
                                ps.bucket_valid = 1
                                break
                            ps.current_logical_bucket += 1

                        if not ps.bucket_valid:
                            ps.done_flag = 1
                        else:
                            ps.window_start = ps.current_logical_bucket
                            settled_vertices.clear()
                            ps.light_phase_active = 1

                            current_vertices = buckets[ps.physical_bucket_idx]
                            buckets[ps.physical_bucket_idx].clear()
                            # Lost-relaxation fix: clear dedup stamps
                            for i in range(<int>current_vertices.size()):
                                last_bucket[current_vertices[i]] = -1
                            settled_vertices.insert(settled_vertices.end(),
                                                   current_vertices.begin(),
                                                   current_vertices.end())

                            ps.vertices_ptr = current_vertices.data()
                            ps.n_vertices = <int>current_vertices.size()
                            ps.work_idx = 0
                            ps.light_iterations = 1

                            for i in range(actual_threads):
                                thread_results[i].count = 0

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

                    if ps.done_flag:
                        break

                    # ======= PHASE 2: All threads relax LIGHT edges (may repeat) =======
                    while ps.light_phase_active:
                        while True:
                            if thread_results[tid].count >= thread_results[tid].guard:
                                break
                            chunk_start = atomic_fetch_add_int(<volatile int*>&ps.work_idx, CHUNK_SIZE)
                            if chunk_start >= ps.n_vertices:
                                break
                            chunk_end = chunk_start + CHUNK_SIZE
                            if chunk_end > ps.n_vertices:
                                chunk_end = ps.n_vertices
                            for j in range(chunk_start, chunk_end):
                                relax_vertex_edges_inline(
                                    ps.vertices_ptr[j], tid,
                                    dist_pred_ptr, raster_view, exclude_view,
                                    directions, cached_steps,
                                    rows, cols, computed_delta, True,
                                    thread_results, total_cells, &sys_limits
                                )

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                        if tid == 0:
                            for i in range(actual_threads):
                                for j in range(thread_results[i].count):
                                    vertex_to_add = thread_results[i].vertices[j]
                                    new_dist = thread_results[i].distances[j]
                                    new_logical_bucket = <size_t>(new_dist / computed_delta)

                                    if new_logical_bucket < ps.window_start + circular_buffer_size:
                                        new_physical_bucket = get_circular_index(
                                            new_logical_bucket, circular_buffer_size)
                                        last_bucket_for_vertex = last_bucket[vertex_to_add]
                                        if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                            buckets[new_physical_bucket].push_back(vertex_to_add)
                                            last_bucket[vertex_to_add] = <int32_t>new_logical_bucket
                                        if new_logical_bucket >= ps.logical_bucket_count:
                                            ps.logical_bucket_count = new_logical_bucket + 1

                            if ps.work_idx < ps.n_vertices:
                                # Rollover: some threads stopped claiming at
                                # their buffer guard. Re-issue the unclaimed
                                # tail as another light round -- these vertices
                                # are already in settled_vertices and must not
                                # be re-stamped or re-counted.
                                ps.vertices_ptr = ps.vertices_ptr + ps.work_idx
                                ps.n_vertices = ps.n_vertices - ps.work_idx
                                ps.work_idx = 0
                                ps.rollovers += 1
                                for i in range(actual_threads):
                                    thread_results[i].count = 0
                            elif (not buckets[ps.physical_bucket_idx].empty()
                                    and ps.light_iterations < max_light_iterations):
                                ps.light_iterations += 1
                                current_vertices = buckets[ps.physical_bucket_idx]
                                buckets[ps.physical_bucket_idx].clear()
                                # Lost-relaxation fix: clear dedup stamps
                                for i in range(<int>current_vertices.size()):
                                    last_bucket[current_vertices[i]] = -1
                                settled_vertices.insert(settled_vertices.end(),
                                                       current_vertices.begin(),
                                                       current_vertices.end())

                                ps.vertices_ptr = current_vertices.data()
                                ps.n_vertices = <int>current_vertices.size()
                                ps.work_idx = 0

                                for i in range(actual_threads):
                                    thread_results[i].count = 0
                            else:
                                ps.light_phase_active = 0

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                    # ======= PHASE 3: Check target found (thread 0) =======
                    if tid == 0:
                        for i in range(<int>settled_vertices.size()):
                            if settled_vertices[i] == target_idx:
                                ps.target_found_flag = 1
                                ps.target_distance = unpack_dist(dist_pred_ptr[target_idx])
                                ps.cutoff_distance = ps.target_distance * termination_margin
                                break

                        if ps.target_found_flag and ps.current_logical_bucket * computed_delta > ps.cutoff_distance:
                            ps.done_flag = 1

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

                    if ps.done_flag:
                        break

                    # ======= PHASE 4: All threads relax HEAVY edges =======
                    if tid == 0:
                        if not settled_vertices.empty():
                            ps.vertices_ptr = settled_vertices.data()
                            ps.n_vertices = <int>settled_vertices.size()
                        else:
                            ps.n_vertices = 0
                        ps.work_idx = 0
                        ps.heavy_phase_active = 1
                        for i in range(actual_threads):
                            thread_results[i].count = 0

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

                    while ps.heavy_phase_active:
                        if ps.n_vertices > 0:
                            while True:
                                if thread_results[tid].count >= thread_results[tid].guard:
                                    break
                                chunk_start = atomic_fetch_add_int(<volatile int*>&ps.work_idx, CHUNK_SIZE)
                                if chunk_start >= ps.n_vertices:
                                    break
                                chunk_end = chunk_start + CHUNK_SIZE
                                if chunk_end > ps.n_vertices:
                                    chunk_end = ps.n_vertices
                                for j in range(chunk_start, chunk_end):
                                    relax_vertex_edges_inline(
                                        ps.vertices_ptr[j], tid,
                                        dist_pred_ptr, raster_view, exclude_view,
                                        directions, cached_steps,
                                        rows, cols, computed_delta, False,
                                        thread_results, total_cells, &sys_limits
                                    )

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                        # ======= PHASE 5: Thread 0 merges heavy results =======
                        if tid == 0:
                            for i in range(actual_threads):
                                for j in range(thread_results[i].count):
                                    vertex_to_add = thread_results[i].vertices[j]
                                    new_dist = thread_results[i].distances[j]
                                    new_logical_bucket = <size_t>(new_dist / computed_delta)

                                    if (new_logical_bucket > ps.current_logical_bucket and
                                        new_logical_bucket < ps.window_start + circular_buffer_size):
                                        new_physical_bucket = get_circular_index(
                                            new_logical_bucket, circular_buffer_size)
                                        last_bucket_for_vertex = last_bucket[vertex_to_add]
                                        if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                            buckets[new_physical_bucket].push_back(vertex_to_add)
                                            last_bucket[vertex_to_add] = <int32_t>new_logical_bucket
                                        if new_logical_bucket >= ps.logical_bucket_count:
                                            ps.logical_bucket_count = new_logical_bucket + 1

                            if ps.work_idx < ps.n_vertices:
                                ps.vertices_ptr = ps.vertices_ptr + ps.work_idx
                                ps.n_vertices = ps.n_vertices - ps.work_idx
                                ps.work_idx = 0
                                ps.rollovers += 1
                                for i in range(actual_threads):
                                    thread_results[i].count = 0
                            else:
                                ps.heavy_phase_active = 0

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                    if tid == 0:
                        buckets[ps.physical_bucket_idx].clear()
                        buckets[ps.physical_bucket_idx].shrink_to_fit()
                        ps.current_logical_bucket += 1

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

            # Copy back from struct for path reconstruction
            target_found = <bint>ps.target_found_flag

    finally:
        # Cleanup resources
        if thread_results != NULL:
            for tid in range(actual_threads):
                drops += thread_results[tid].overflow
                if thread_results[tid].vertices != NULL:
                    free(thread_results[tid].vertices)
                if thread_results[tid].bucket_indices != NULL:
                    free(thread_results[tid].bucket_indices)
                if thread_results[tid].distances != NULL:
                    free(thread_results[tid].distances)
            free(thread_results)
        _relaxation_stats["capacity"] = int(max_capacity)
        _relaxation_stats["guard"] = int(max_capacity - guard_slack)
        _relaxation_stats["rollovers"] = int(ps.rollovers)
        _relaxation_stats["drops"] = int(drops)

    # Path reconstruction (same as original)
    pred_val = unpack_pred(dist_pred_ptr[target_idx])
    if not target_found or pred_val == 0xFFFFFFFF:
        if source_idx == target_idx:
            return np.array([source_idx], dtype=np.uint64)
        return np.empty(0, dtype=np.uint64)

    path_vertices = []
    current = target_idx
    path_length = 0

    while path_length < sys_limits.max_path_length:
        path_vertices.append(current)
        if current == source_idx:
            break
        pred_val = unpack_pred(dist_pred_ptr[current])
        if pred_val == 0xFFFFFFFF:
            return np.empty(0, dtype=np.uint64)
        current = <uint64_t>pred_val
        path_length += 1

    path_vertices.reverse()

    return np.array(path_vertices, dtype=np.uint64)


# ==================== PERSISTENT DELTA-STEPPING (MULTI TARGET) ====================

def delta_stepping_single_source_multiple_targets_persistent(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint64_t source_idx,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        int64_t max_value=65535,
        int num_threads=0,
        size_t max_buckets_in_memory=2048,
        DeltaWorkspace workspace=None):
    """
    Persistent-thread-pool variant of delta_stepping_single_source_multiple_targets.

    Same algorithm as the original but uses a single prange call with manual barriers
    instead of repeated prange fork/join.

    Parameters: same as delta_stepping_single_source_multiple_targets
    Returns: same as delta_stepping_single_source_multiple_targets
    """
    # ============= ALL VARIABLE DECLARATIONS AT TOP =============

    cdef SystemLimits sys_limits = system_limits_memoized()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols
    cdef int num_targets = <int>target_indices.shape[0]

    cdef DeltaWorkspace ws
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr
    cdef const uint16_t[:, :] raster_view
    cdef const uint8_t[:, :] exclude_view

    cdef size_t circular_buffer_size
    cdef size_t buffer_mask
    cdef size_t logical_bucket_count = 0
    cdef size_t current_logical_bucket = 0
    cdef size_t physical_bucket_idx = 0
    cdef size_t window_start = 0
    cdef bint bucket_valid = False

    cdef np.ndarray[int32_t, ndim=1] last_bucket_arr
    cdef int32_t[:] last_bucket

    cdef int actual_threads
    cdef ThreadResults* thread_results = NULL
    cdef int max_capacity
    cdef int tid

    cdef vector[vector[uint64_t]] buckets
    cdef vector[uint64_t] current_vertices
    cdef vector[uint64_t] settled_vertices
    cdef int targets_found = 0
    cdef np.ndarray[uint8_t, ndim=1] target_found_arr
    cdef uint8_t[:] target_found

    cdef float max_target_distance = 0.0
    cdef float current_target_distance

    cdef int iteration, light_iterations
    cdef int max_light_iterations = 10000
    cdef uint64_t v, vertex_to_add, current_vertex
    cdef size_t new_logical_bucket, new_physical_bucket
    cdef float new_dist
    cdef int32_t last_bucket_for_vertex

    cdef uint64_t target_idx, path_length
    cdef uint32_t pred_val
    cdef list path_vertices
    cdef list paths = []

    cdef vector[CachedStepData] cached_steps
    cdef vector[StepData] directions

    cdef np.ndarray[uint64_t, ndim=1] dist_pred_arr
    cdef uint64_t* dist_pred_ptr
    cdef uint64_t init_packed

    cdef uint64_t source_r, source_c
    cdef int i, j

    # Persistent thread pool state (struct avoids Cython reduction variable issues)
    cdef PersistentState ps_data
    cdef PersistentState* ps = &ps_data
    cdef int CHUNK_SIZE = 64
    cdef int chunk_start, chunk_end
    cdef int local_sense

    # Relaxation buffer overflow protocol
    cdef int out_degree, guard_slack
    cdef int forced_capacity = <int>_relaxation_config["forced_capacity"]
    cdef int64_t drops = 0

    # ============= VALIDATION =============

    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if num_targets == 0:
        return []

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")

    if total_cells > 0xFFFFFFFF:
        raise OverflowError(
            f"Raster has {total_cells} cells, exceeding uint32 predecessor limit (4294967295)")

    if num_threads <= 0:
        num_threads = min(sys_limits.num_cores, omp_get_max_threads())
    omp_set_num_threads(num_threads)
    actual_threads = omp_get_max_threads()

    if source_idx >= total_cells:
        return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    for i in range(num_targets):
        if target_indices[i] >= total_cells:
            return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    ws = bind_workspace(workspace, raster_arr, max_value)
    exclude_mask_arr = ws.exclude_mask
    source_r = source_idx // cols
    source_c = source_idx % cols

    if exclude_mask_arr[source_r, source_c] == 0:
        return [np.empty(0, dtype=np.uint64) for _ in range(num_targets)]

    cached_steps = precompute_cached_steps(steps_arr)
    directions = precompute_directions_optimized(steps_arr, cached_steps)

    init_packed = pack_dist_pred(INF_F32, 0xFFFFFFFF)
    dist_pred_arr = ws.take_dist_pred(init_packed, source_idx)
    dist_pred_ptr = <uint64_t*>dist_pred_arr.data
    target_found_arr = np.zeros(num_targets, dtype=np.uint8)

    raster_view = raster_arr
    exclude_view = exclude_mask_arr
    target_found = target_found_arr

    circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    buffer_mask = circular_buffer_size - 1
    buckets.resize(circular_buffer_size)

    # P1.3 fix: Validate circular buffer can hold max bucket span
    cdef double _max_step_dist = 0.0
    cdef double _sd, _dr_f, _dc_f, _max_span
    cdef int _si
    for _si in range(steps_arr.shape[0]):
        _dr_f = <double>steps_arr[_si, 0]
        _dc_f = <double>steps_arr[_si, 1]
        _sd = (_dr_f * _dr_f + _dc_f * _dc_f) ** 0.5
        if _sd > _max_step_dist:
            _max_step_dist = _sd
    if ws.has_traversable:
        _max_span = ws.max_traversable_cost * _max_step_dist / delta
        if _max_span >= <double>circular_buffer_size:
            raise ValueError(
                f"Delta-stepping: max edge/delta ratio ({_max_span:.0f}) "
                f"exceeds circular buffer size ({circular_buffer_size}). "
                f"Increase max_buckets_in_memory or delta.")


    last_bucket_arr = ws.take_last_bucket()
    last_bucket = last_bucket_arr

    physical_bucket_idx = get_circular_index(0, circular_buffer_size)
    buckets[physical_bucket_idx].push_back(source_idx)
    last_bucket[source_idx] = 0

    thread_results = <ThreadResults*>calloc(actual_threads, sizeof(ThreadResults))
    if thread_results == NULL:
        raise MemoryError("Could not allocate thread data")

    out_degree = <int>directions.size()
    guard_slack = relax_guard_slack(out_degree, CHUNK_SIZE)
    max_capacity = size_relax_buffer(total_cells, actual_threads, guard_slack,
                                     &sys_limits, forced_capacity)

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
            raise MemoryError("Could not allocate thread storage")

        thread_results[tid].capacity = max_capacity
        thread_results[tid].guard = max_capacity - guard_slack
        thread_results[tid].count = 0
        thread_results[tid].overflow = 0

    max_light_iterations = max(50, <int>(sqrtf(<float>total_cells)))

    # ============= MAIN PERSISTENT LOOP =============

    # Initialize persistent state
    ps.current_logical_bucket = 0
    ps.logical_bucket_count = 0
    ps.physical_bucket_idx = 0
    ps.window_start = 0
    ps.done_flag = 0
    ps.barrier_arrive_count = 0
    ps.barrier_sense = 0
    ps.work_idx = 0
    ps.n_vertices = 0
    ps.light_iterations = 0
    ps.light_phase_active = 0
    ps.heavy_phase_active = 0
    ps.bucket_valid = 0
    ps.target_found_flag = 0
    ps.target_distance = INF_F32
    ps.cutoff_distance = INF_F32
    ps.targets_found = 0
    ps.max_target_distance = 0.0
    ps.rollovers = 0
    ps.vertices_ptr = NULL

    try:
        if actual_threads <= 1:
            # ---- SINGLE-THREAD FAST PATH ----
            for iteration in range(sys_limits.max_iterations):
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

                while not buckets[physical_bucket_idx].empty() and light_iterations < max_light_iterations:
                    light_iterations += 1
                    current_vertices = buckets[physical_bucket_idx]
                    buckets[physical_bucket_idx].clear()
                    # Lost-relaxation fix (see above): clear dedup stamps
                    for i in range(<int>current_vertices.size()):
                        last_bucket[current_vertices[i]] = -1
                    settled_vertices.insert(settled_vertices.end(),
                                           current_vertices.begin(),
                                           current_vertices.end())

                    # Relax in guard-bounded slices, merging between slices so
                    # the buffer can never overflow.
                    j = 0
                    while j < <int>current_vertices.size():
                        thread_results[0].count = 0
                        while (j < <int>current_vertices.size() and
                               thread_results[0].count < thread_results[0].guard):
                            relax_vertex_edges_inline(
                                current_vertices[j], 0,
                                dist_pred_ptr, raster_view, exclude_view,
                                directions, cached_steps,
                                rows, cols, delta, True,
                                thread_results, total_cells, &sys_limits
                            )
                            j += 1

                        for i in range(thread_results[0].count):
                            vertex_to_add = thread_results[0].vertices[i]
                            new_dist = thread_results[0].distances[i]
                            new_logical_bucket = <size_t>(new_dist / delta)

                            if new_logical_bucket < window_start + circular_buffer_size:
                                new_physical_bucket = get_circular_index(new_logical_bucket, circular_buffer_size)
                                last_bucket_for_vertex = last_bucket[vertex_to_add]
                                if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                    buckets[new_physical_bucket].push_back(vertex_to_add)
                                    last_bucket[vertex_to_add] = <int32_t>new_logical_bucket
                                if new_logical_bucket >= logical_bucket_count:
                                    logical_bucket_count = new_logical_bucket + 1

                        if j < <int>current_vertices.size():
                            ps.rollovers += 1

                # Check targets
                for i in range(<int>settled_vertices.size()):
                    current_vertex = settled_vertices[i]
                    for j in range(num_targets):
                        if current_vertex == target_indices[j] and target_found[j] == 0:
                            target_found[j] = 1
                            targets_found += 1
                            current_target_distance = unpack_dist(dist_pred_ptr[current_vertex])
                            if current_target_distance > max_target_distance:
                                max_target_distance = current_target_distance

                if targets_found >= num_targets:
                    break

                # HEAVY PHASE
                if not settled_vertices.empty():
                    j = 0
                    while j < <int>settled_vertices.size():
                        thread_results[0].count = 0
                        while (j < <int>settled_vertices.size() and
                               thread_results[0].count < thread_results[0].guard):
                            relax_vertex_edges_inline(
                                settled_vertices[j], 0,
                                dist_pred_ptr, raster_view, exclude_view,
                                directions, cached_steps,
                                rows, cols, delta, False,
                                thread_results, total_cells, &sys_limits
                            )
                            j += 1

                        for i in range(thread_results[0].count):
                            vertex_to_add = thread_results[0].vertices[i]
                            new_dist = thread_results[0].distances[i]
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

                        if j < <int>settled_vertices.size():
                            ps.rollovers += 1

                buckets[physical_bucket_idx].clear()
                buckets[physical_bucket_idx].shrink_to_fit()
                current_logical_bucket += 1

        else:
            # ---- MULTI-THREAD PERSISTENT LOOP ----
            # All shared mutable scalars accessed through ps pointer.

            for tid in prange(actual_threads, schedule='static', nogil=True,
                              num_threads=actual_threads):
                local_sense = 0

                while True:
                    if tid == 0:
                        ps.work_idx = 0
                        ps.n_vertices = 0
                        ps.done_flag = 0
                        ps.light_phase_active = 0

                        ps.bucket_valid = 0
                        while ps.current_logical_bucket < ps.logical_bucket_count + circular_buffer_size:
                            ps.physical_bucket_idx = get_circular_index(
                                ps.current_logical_bucket, circular_buffer_size)
                            if ps.current_logical_bucket >= ps.logical_bucket_count + circular_buffer_size:
                                break
                            if not buckets[ps.physical_bucket_idx].empty():
                                ps.bucket_valid = 1
                                break
                            ps.current_logical_bucket += 1

                        if not ps.bucket_valid:
                            ps.done_flag = 1
                        else:
                            ps.window_start = ps.current_logical_bucket
                            settled_vertices.clear()
                            ps.light_phase_active = 1

                            current_vertices = buckets[ps.physical_bucket_idx]
                            buckets[ps.physical_bucket_idx].clear()
                            # Lost-relaxation fix: clear dedup stamps
                            for i in range(<int>current_vertices.size()):
                                last_bucket[current_vertices[i]] = -1
                            settled_vertices.insert(settled_vertices.end(),
                                                   current_vertices.begin(),
                                                   current_vertices.end())

                            ps.vertices_ptr = current_vertices.data()
                            ps.n_vertices = <int>current_vertices.size()
                            ps.work_idx = 0
                            ps.light_iterations = 1

                            for i in range(actual_threads):
                                thread_results[i].count = 0

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

                    if ps.done_flag:
                        break

                    # Phase 2: Light edge relaxation (may repeat)
                    while ps.light_phase_active:
                        while True:
                            if thread_results[tid].count >= thread_results[tid].guard:
                                break
                            chunk_start = atomic_fetch_add_int(<volatile int*>&ps.work_idx, CHUNK_SIZE)
                            if chunk_start >= ps.n_vertices:
                                break
                            chunk_end = chunk_start + CHUNK_SIZE
                            if chunk_end > ps.n_vertices:
                                chunk_end = ps.n_vertices
                            for j in range(chunk_start, chunk_end):
                                relax_vertex_edges_inline(
                                    ps.vertices_ptr[j], tid,
                                    dist_pred_ptr, raster_view, exclude_view,
                                    directions, cached_steps,
                                    rows, cols, delta, True,
                                    thread_results, total_cells, &sys_limits
                                )

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                        if tid == 0:
                            for i in range(actual_threads):
                                for j in range(thread_results[i].count):
                                    vertex_to_add = thread_results[i].vertices[j]
                                    new_dist = thread_results[i].distances[j]
                                    new_logical_bucket = <size_t>(new_dist / delta)

                                    if new_logical_bucket < ps.window_start + circular_buffer_size:
                                        new_physical_bucket = get_circular_index(
                                            new_logical_bucket, circular_buffer_size)
                                        last_bucket_for_vertex = last_bucket[vertex_to_add]
                                        if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                            buckets[new_physical_bucket].push_back(vertex_to_add)
                                            last_bucket[vertex_to_add] = <int32_t>new_logical_bucket
                                        if new_logical_bucket >= ps.logical_bucket_count:
                                            ps.logical_bucket_count = new_logical_bucket + 1

                            if ps.work_idx < ps.n_vertices:
                                # Rollover: some threads stopped claiming at
                                # their buffer guard. Re-issue the unclaimed
                                # tail as another light round -- these vertices
                                # are already in settled_vertices and must not
                                # be re-stamped or re-counted.
                                ps.vertices_ptr = ps.vertices_ptr + ps.work_idx
                                ps.n_vertices = ps.n_vertices - ps.work_idx
                                ps.work_idx = 0
                                ps.rollovers += 1
                                for i in range(actual_threads):
                                    thread_results[i].count = 0
                            elif (not buckets[ps.physical_bucket_idx].empty()
                                    and ps.light_iterations < max_light_iterations):
                                ps.light_iterations += 1
                                current_vertices = buckets[ps.physical_bucket_idx]
                                buckets[ps.physical_bucket_idx].clear()
                                # Lost-relaxation fix: clear dedup stamps
                                for i in range(<int>current_vertices.size()):
                                    last_bucket[current_vertices[i]] = -1
                                settled_vertices.insert(settled_vertices.end(),
                                                       current_vertices.begin(),
                                                       current_vertices.end())
                                ps.vertices_ptr = current_vertices.data()
                                ps.n_vertices = <int>current_vertices.size()
                                ps.work_idx = 0
                                for i in range(actual_threads):
                                    thread_results[i].count = 0
                            else:
                                ps.light_phase_active = 0

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                    # Phase 3: Check all targets found
                    if tid == 0:
                        for i in range(<int>settled_vertices.size()):
                            current_vertex = settled_vertices[i]
                            for j in range(num_targets):
                                if current_vertex == target_indices[j] and target_found[j] == 0:
                                    target_found[j] = 1
                                    ps.targets_found += 1
                                    current_target_distance = unpack_dist(
                                        dist_pred_ptr[current_vertex])
                                    if current_target_distance > ps.max_target_distance:
                                        ps.max_target_distance = current_target_distance

                        if ps.targets_found >= num_targets:
                            ps.done_flag = 1

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

                    if ps.done_flag:
                        break

                    # Phase 4: Heavy edge relaxation
                    if tid == 0:
                        if not settled_vertices.empty():
                            ps.vertices_ptr = settled_vertices.data()
                            ps.n_vertices = <int>settled_vertices.size()
                        else:
                            ps.n_vertices = 0
                        ps.work_idx = 0
                        ps.heavy_phase_active = 1
                        for i in range(actual_threads):
                            thread_results[i].count = 0

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

                    while ps.heavy_phase_active:
                        if ps.n_vertices > 0:
                            while True:
                                if thread_results[tid].count >= thread_results[tid].guard:
                                    break
                                chunk_start = atomic_fetch_add_int(<volatile int*>&ps.work_idx, CHUNK_SIZE)
                                if chunk_start >= ps.n_vertices:
                                    break
                                chunk_end = chunk_start + CHUNK_SIZE
                                if chunk_end > ps.n_vertices:
                                    chunk_end = ps.n_vertices
                                for j in range(chunk_start, chunk_end):
                                    relax_vertex_edges_inline(
                                        ps.vertices_ptr[j], tid,
                                        dist_pred_ptr, raster_view, exclude_view,
                                        directions, cached_steps,
                                        rows, cols, delta, False,
                                        thread_results, total_cells, &sys_limits
                                    )

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                        # Phase 5: Heavy merge
                        if tid == 0:
                            for i in range(actual_threads):
                                for j in range(thread_results[i].count):
                                    vertex_to_add = thread_results[i].vertices[j]
                                    new_dist = thread_results[i].distances[j]
                                    new_logical_bucket = <size_t>(new_dist / delta)

                                    if (new_logical_bucket > ps.current_logical_bucket and
                                        new_logical_bucket < ps.window_start + circular_buffer_size):
                                        new_physical_bucket = get_circular_index(
                                            new_logical_bucket, circular_buffer_size)
                                        last_bucket_for_vertex = last_bucket[vertex_to_add]
                                        if last_bucket_for_vertex != <int32_t>new_logical_bucket:
                                            buckets[new_physical_bucket].push_back(vertex_to_add)
                                            last_bucket[vertex_to_add] = <int32_t>new_logical_bucket
                                        if new_logical_bucket >= ps.logical_bucket_count:
                                            ps.logical_bucket_count = new_logical_bucket + 1

                            if ps.work_idx < ps.n_vertices:
                                ps.vertices_ptr = ps.vertices_ptr + ps.work_idx
                                ps.n_vertices = ps.n_vertices - ps.work_idx
                                ps.work_idx = 0
                                ps.rollovers += 1
                                for i in range(actual_threads):
                                    thread_results[i].count = 0
                            else:
                                ps.heavy_phase_active = 0

                        thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                           <volatile int*>&ps.barrier_sense,
                                           actual_threads, &local_sense)

                    if tid == 0:
                        buckets[ps.physical_bucket_idx].clear()
                        buckets[ps.physical_bucket_idx].shrink_to_fit()
                        ps.current_logical_bucket += 1

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                       <volatile int*>&ps.barrier_sense,
                                       actual_threads, &local_sense)

    finally:
        if thread_results != NULL:
            for tid in range(actual_threads):
                drops += thread_results[tid].overflow
                free(thread_results[tid].vertices)
                free(thread_results[tid].bucket_indices)
                free(thread_results[tid].distances)
            free(thread_results)
        _relaxation_stats["capacity"] = int(max_capacity)
        _relaxation_stats["guard"] = int(max_capacity - guard_slack)
        _relaxation_stats["rollovers"] = int(ps.rollovers)
        _relaxation_stats["drops"] = int(drops)

    # Reconstruct paths for all targets (same as original)
    for i in range(num_targets):
        target_idx = target_indices[i]

        pred_val = unpack_pred(dist_pred_ptr[target_idx])
        if pred_val == 0xFFFFFFFF:
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
            pred_val = unpack_pred(dist_pred_ptr[current_vertex])
            if pred_val == 0xFFFFFFFF:
                paths.append(np.empty(0, dtype=np.uint64))
                break
            current_vertex = <uint64_t>pred_val
            path_length += 1
        else:
            if current_vertex == source_idx:
                path_vertices.append(source_idx)
                path_vertices.reverse()
                paths.append(np.array(path_vertices, dtype=np.uint64))
            else:
                paths.append(np.empty(0, dtype=np.uint64))

    return paths


# ==================== PERSISTENT WRAPPERS FOR MULTI-SOURCE ====================

def delta_stepping_multiple_sources_multiple_targets_persistent(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint64_t, ndim=1] source_indices,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        int64_t max_value=65535,
        bint return_paths=True,
        int num_threads=0,
        size_t max_buckets_in_memory=2048):
    """
    Persistent-thread-pool variant of delta_stepping_multiple_sources_multiple_targets.

    Same algorithm but uses persistent thread pool for each source's SSSP.
    """
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef int num_sources = <int>source_indices.shape[0]
    cdef int num_targets = <int>target_indices.shape[0]

    cdef np.ndarray[float32_t, ndim=2] cost_matrix = np.full(
        (num_sources, num_targets), INF_F32, dtype=np.float32)
    cdef list all_paths = [] if return_paths else None

    cdef np.ndarray[uint64_t, ndim=1] sorted_sources
    cdef dict source_idx_map = {}
    cdef int s, t, original_idx
    cdef uint64_t source_idx
    cdef list source_paths
    cdef np.ndarray[uint64_t, ndim=1] path
    cdef float cost
    cdef DeltaWorkspace workspace = None

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

    # The raster is the same for every source and nothing here writes to it, so
    # its derivations and state arrays are derived once for the whole loop.
    workspace = DeltaWorkspace(raster_arr, max_value)

    for s in range(num_sources):
        source_idx = sorted_sources[s]
        original_idx = source_idx_map[s]

        try:
            source_paths = delta_stepping_single_source_multiple_targets_persistent(
                raster_arr, steps_arr, source_idx, target_indices,
                delta, max_value, num_threads, max_buckets_in_memory,
                workspace
            )

            if return_paths:
                if len(all_paths) <= original_idx:
                    all_paths.extend([None] * (original_idx - len(all_paths) + 1))
                all_paths[original_idx] = source_paths
            else:
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


def delta_stepping_some_pairs_shortest_paths_persistent(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint64_t, ndim=1] source_indices,
        np.ndarray[uint64_t, ndim=1] target_indices,
        float delta,
        int64_t max_value=65535,
        bint return_paths=True,
        int num_threads=0,
        size_t max_buckets_in_memory=2048,
        float margin=1.00001):
    """
    Persistent-thread-pool variant of delta_stepping_some_pairs_shortest_paths.

    Same algorithm but uses persistent thread pool for each pair's SSSP.
    """
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef int num_pairs = <int>min(source_indices.shape[0], target_indices.shape[0])

    cdef list all_paths = [] if return_paths else None
    cdef np.ndarray[float32_t, ndim=1] costs = np.full(num_pairs, INF_F32,
                                                       dtype=np.float32)

    cdef int i
    cdef uint64_t source, target
    cdef np.ndarray[uint64_t, ndim=1] path
    cdef float path_cost_value
    cdef float validated_margin
    cdef DeltaWorkspace workspace = None

    if margin <= 1.00001:
        validated_margin = 1.00001
    else:
        validated_margin = margin

    if num_pairs == 0:
        if return_paths:
            return []
        else:
            return np.empty(0, dtype=np.float32)

    # The raster is the same for every pair and nothing here writes to it, so
    # its derivations and state arrays are derived once for the whole loop.
    workspace = DeltaWorkspace(raster_arr, max_value)

    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        path = delta_stepping_2d_persistent(
            raster_arr, steps_arr, source, target,
            delta, max_value, num_threads, max_buckets_in_memory,
            validated_margin,
            workspace
        )

        if return_paths:
            all_paths.append(path)
        else:
            if len(path) > 0:
                path_cost_value = <float>path_cost(path, raster_arr, cols)
                costs[i] = path_cost_value

    return all_paths if return_paths else costs

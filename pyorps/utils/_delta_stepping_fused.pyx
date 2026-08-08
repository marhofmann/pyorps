"""
Fused delta-stepping kernel: bucket fusion + adaptive window (decreasing
effective delta) + VGC-style local-work cap.

Implements three published levers for high-diameter graphs on top of the
persistent-thread-pool delta-stepping architecture from _delta_stepping.pyx:

1. Bucket fusion (Zhang et al., "Optimizing Ordered Graph Algorithms with
   GraphIt", CGO 2020): relaxations that land inside the current bucket
   window are processed immediately from a thread-local stack instead of
   going through a global merge + barrier per light iteration.
2. Decreasing effective delta (Hassan & Arifuzzaman, PD3, HPEC 2024),
   realized as an adaptive super-bucket window: W consecutive base buckets
   are processed as one unit (effective delta = W * delta). W grows while
   the frontier is too small to occupy the thread pool and shrinks once it
   saturates. No re-bucketing is required because the base bucket width
   never changes.
3. Vertical granularity control (Dong, Gu, Sun, Wang, PASGAL, SPAA 2024):
   a cap (fusion_cap) on local-stack pops per thread per phase bounds load
   imbalance from long shortest-path chains; leftovers are re-chunked
   across threads in an extra round.

The baseline kernels in _delta_stepping.pyx are intentionally untouched so
benchmarks compare against the unmodified implementation.

Overflow safety: threads stop grabbing work chunks when their insert or
settled buffers approach capacity; unfinished work rolls over into another
round (both in the light and the heavy phase). No successful relaxation is
ever dropped.
"""

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free, calloc
from cython.parallel cimport prange
from openmp cimport omp_get_max_threads, omp_set_num_threads

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

cdef extern from "atomic_cas.h" nogil:
    uint64_t pack_dist_pred(float dist, uint32_t pred)
    float unpack_dist(uint64_t packed)
    uint32_t unpack_pred(uint64_t packed)
    int atomic_try_update_dist_pred(volatile uint64_t* dist_pred, uint64_t v_idx,
                                     float new_dist, uint32_t new_pred)
    uint64_t atomic_load_u64(volatile uint64_t* addr)
    int atomic_fetch_add_int(volatile int* addr, int val)
    uint8_t atomic_exchange_u8(volatile uint8_t* addr, uint8_t val)
    void thread_barrier_wait(volatile int* arrive_count, volatile int* sense,
                             int num_threads, int* local_sense)


# ==================== PER-THREAD STATE ====================

cdef struct FusedThread:
    # Insert buffer: successful relaxations destined for future buckets
    uint64_t* ins_v
    float* ins_d
    int ins_count
    int ins_cap
    int ins_guard          # stop grabbing new work above this fill level
    # Local fusion stack: in-window continuations (bucket fusion)
    uint64_t* stack
    int top
    int stack_cap
    # Vertices settled in the current window (for the heavy phase)
    uint64_t* settled
    int settled_count
    int settled_cap
    int settled_guard


cdef struct FusedState:
    # Shared mutable scalars, accessed through a pointer inside prange to
    # avoid Cython reduction-variable analysis (same pattern as the
    # persistent kernel in _delta_stepping.pyx).
    size_t window_start
    size_t window_len
    size_t logical_bucket_count
    int window_id
    int done_flag
    int dirty
    int inner_rounds
    int barrier_arrive_count
    int barrier_sense
    int work_idx
    int n_vertices
    int n_heavy
    int target_found_flag
    float cutoff_distance
    uint64_t* vertices_ptr
    uint64_t* heavy_ptr


# ==================== FUSED RELAXATION ====================

cdef inline void _relax_one(
        uint64_t u,
        FusedThread* T,
        uint64_t* dist_pred,
        const uint16_t[:, :] raster,
        const uint8_t[:, :] exclude_mask,
        const vector[StepData]& directions,
        const vector[CachedStepData]& cached_steps,
        int rows,
        uint64_t cols,
        float span_w,
        float we_dist,
        bint heavy_mode,
        int32_t* stamp,
        uint8_t* pending,
        int window_id,
        uint64_t total_cells) noexcept nogil:
    """
    Relax the edges of a single vertex under the current window.

    Light mode: only edges with weight <= span_w; successful updates that
    land inside the window go to the local fusion stack (or the insert
    buffer when the stack is full), all others to the insert buffer.
    Heavy mode: only edges with weight > span_w; every successful update
    goes to the insert buffer (provably lands beyond the window).
    """
    cdef int dir_idx, ur, uc, vr, vc
    cdef uint64_t v, ur64
    cdef float du, edge_weight, new_dist, intermediate_cost
    cdef float raster_ur_uc
    cdef int valid_path

    if u >= total_cells:
        return

    if not heavy_mode:
        # Consumer side of the pending protocol: clear BEFORE loading the
        # distance so any producer that skipped its queue push (because it
        # saw pending==1) is guaranteed to have its CAS visible below.
        atomic_exchange_u8(&pending[u], 0)

    du = unpack_dist(atomic_load_u64(&dist_pred[u]))
    if du >= INF_F32:
        return

    ur64 = u // cols
    ur = <int>ur64
    uc = <int>(u - ur64 * cols)

    if not heavy_mode:
        # Record for the heavy phase (deduplicated per window; the stamp
        # write race is benign -- a duplicate only causes a redundant,
        # CAS-guarded heavy relaxation).
        if stamp[u] != window_id and T.settled_count < T.settled_cap:
            stamp[u] = window_id
            T.settled[T.settled_count] = u
            T.settled_count += 1

    raster_ur_uc = <float>raster[ur, uc]

    for dir_idx in range(<int>directions.size()):
        vr = ur + directions[dir_idx].dr
        vc = uc + directions[dir_idx].dc

        if vr < 0 or vr >= rows or vc < 0 or vc >= <int>cols:
            continue
        if exclude_mask[vr, vc] == 0:
            continue

        intermediate_cost = 0.0
        valid_path = check_path_cached(
            cached_steps[dir_idx].intermediates,
            ur, uc, exclude_mask, raster, rows, <int>cols, &intermediate_cost
        )
        if not valid_path:
            continue

        edge_weight = (raster_ur_uc + intermediate_cost +
                       <float>raster[vr, vc]) * directions[dir_idx].cost_factor

        if heavy_mode:
            if edge_weight <= span_w:
                continue
        else:
            if edge_weight > span_w:
                continue

        v = <uint64_t>vr * cols + <uint64_t>vc
        new_dist = du + edge_weight

        if atomic_try_update_dist_pred(
                <volatile uint64_t*>dist_pred, v, new_dist, <uint32_t>u):
            if (not heavy_mode) and new_dist < we_dist:
                # Producer side: queue only if not already queued -- this
                # coalesces repeated improvements of the same vertex within
                # a window (the queued instance re-reads the fresh distance
                # when processed, so no relaxation is lost).
                if atomic_exchange_u8(&pending[v], 1) == 0:
                    if T.top < T.stack_cap:
                        # Bucket fusion: keep in-window work local
                        T.stack[T.top] = v
                        T.top += 1
                    elif T.ins_count < T.ins_cap:
                        T.ins_v[T.ins_count] = v
                        T.ins_d[T.ins_count] = new_dist
                        T.ins_count += 1
            elif T.ins_count < T.ins_cap:
                T.ins_v[T.ins_count] = v
                T.ins_d[T.ins_count] = new_dist
                T.ins_count += 1
            # else: unreachable by construction -- threads stop grabbing
            # work chunks before the insert buffer can fill (ins_guard)


# ==================== MAIN KERNEL ====================

def delta_stepping_2d_fused(np.ndarray[uint16_t, ndim=2] raster_arr,
                            np.ndarray[int8_t, ndim=2] steps_arr,
                            uint64_t source_idx, uint64_t target_idx,
                            float delta,
                            int64_t max_value=65535,
                            int num_threads=0,
                            size_t max_buckets_in_memory=2048,
                            float margin=1.00001,
                            size_t window_init=8,
                            size_t window_max=64,
                            bint adaptive_window=True,
                            int fusion_cap=1024):
    """
    Single-source single-target delta-stepping with bucket fusion,
    adaptive super-bucket window, and VGC-style local-work cap.

    Parameters beyond delta_stepping_2d_persistent:
        window_init: initial window width in base buckets (1 = classic)
        window_max: adaptive-window upper bound (must leave circular-buffer
                    headroom for the max edge span)
        adaptive_window: grow/shrink the window from frontier size; when
                    False the window stays at window_init
        fusion_cap: max local-stack pops per thread per phase (0 disables
                    bucket fusion entirely)

    Returns a 1D numpy array (uint64) of linear indices, empty if no path.
    """
    # ---- dimensions / limits ----
    cdef SystemLimits sys_limits = get_system_limits()
    cdef int rows = <int>raster_arr.shape[0]
    cdef uint64_t cols = <uint64_t>raster_arr.shape[1]
    cdef uint64_t total_cells = <uint64_t>rows * cols

    if total_cells > sys_limits.max_array_size:
        raise MemoryError(f"Problem size ({total_cells} cells) exceeds system limits")
    if total_cells > 0xFFFFFFFF:
        raise OverflowError(
            f"Raster has {total_cells} cells, exceeding uint32 predecessor limit")
    if delta <= 0.0:
        raise ValueError(f"Invalid delta value: {delta}! Choose a delta > 0.0!")
    if source_idx >= total_cells or target_idx >= total_cells:
        return np.empty(0, dtype=np.uint64)
    if source_idx == target_idx:
        return np.array([source_idx], dtype=np.uint64)

    cdef float termination_margin = margin if margin > 1.00001 else 1.00001

    # ---- masks / traversability ----
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr = \
        (raster_arr != max_value).astype(np.uint8)
    cdef uint64_t source_r = source_idx // cols
    cdef uint64_t source_c = source_idx % cols
    cdef uint64_t target_r = target_idx // cols
    cdef uint64_t target_c = target_idx % cols
    if (exclude_mask_arr[source_r, source_c] == 0 or
            exclude_mask_arr[target_r, target_c] == 0):
        return np.empty(0, dtype=np.uint64)

    # ---- threads ----
    if num_threads <= 0:
        num_threads = min(sys_limits.num_cores, omp_get_max_threads())
    omp_set_num_threads(num_threads)
    cdef int actual_threads = omp_get_max_threads()

    # ---- movement precomputation ----
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps_arr)
    cdef vector[StepData] directions = \
        precompute_directions_optimized(steps_arr, cached_steps)
    cdef int out_degree = <int>directions.size()

    # ---- packed dist+pred ----
    cdef uint64_t init_packed = pack_dist_pred(INF_F32, 0xFFFFFFFF)
    cdef np.ndarray[uint64_t, ndim=1] dist_pred_arr = np.full(
        <size_t>total_cells, init_packed, dtype=np.uint64)
    dist_pred_arr[source_idx] = pack_dist_pred(0.0, 0xFFFFFFFF)
    cdef uint64_t* dist_pred_ptr = <uint64_t*>dist_pred_arr.data

    cdef const uint16_t[:, :] raster_view = raster_arr
    cdef const uint8_t[:, :] exclude_view = exclude_mask_arr

    # ---- circular buckets ----
    cdef size_t circular_buffer_size = round_up_power_of_two(max_buckets_in_memory)
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(circular_buffer_size)

    # Max edge span in buckets (mirrors the P1.3 validation of the baseline,
    # extended by the window width headroom)
    cdef double _max_step_dist = 0.0
    cdef double _sd, _dr_f, _dc_f, _max_cell, _max_span
    cdef int _si
    for _si in range(steps_arr.shape[0]):
        _dr_f = <double>steps_arr[_si, 0]
        _dc_f = <double>steps_arr[_si, 1]
        _sd = (_dr_f * _dr_f + _dc_f * _dc_f) ** 0.5
        if _sd > _max_step_dist:
            _max_step_dist = _sd
    _valid_mask = (exclude_mask_arr == 1)
    cdef size_t max_span_buckets = 0
    if np.any(_valid_mask):
        _max_cell = <double>np.max(raster_arr[_valid_mask])
        # Edge weight upper bound: (cost_u + intermediates + cost_v) * factor
        # conservative: 4 * max_cell * step_dist
        _max_span = 4.0 * _max_cell * _max_step_dist / delta
        max_span_buckets = <size_t>_max_span + 2
    if window_max < 1:
        window_max = 1
    if window_init < 1:
        window_init = 1
    if window_init > window_max:
        window_init = window_max
    if window_max + max_span_buckets + 2 >= circular_buffer_size:
        raise ValueError(
            f"Fused delta-stepping: window_max ({window_max}) + max edge span "
            f"({max_span_buckets} buckets) exceeds circular buffer "
            f"({circular_buffer_size}). Increase max_buckets_in_memory or "
            f"delta, or reduce window_max.")

    # ---- dedup arrays ----
    cdef np.ndarray[int32_t, ndim=1] last_bucket_arr = np.full(
        <size_t>total_cells, -1, dtype=np.int32)
    cdef int32_t[:] last_bucket = last_bucket_arr
    cdef np.ndarray[int32_t, ndim=1] stamp_arr = np.full(
        <size_t>total_cells, -1, dtype=np.int32)
    cdef int32_t* stamp = <int32_t*>stamp_arr.data
    cdef np.ndarray[uint8_t, ndim=1] pending_arr = np.zeros(
        <size_t>total_cells, dtype=np.uint8)
    cdef uint8_t* pending = <uint8_t*>pending_arr.data

    # ---- seed ----
    buckets[get_circular_index(0, circular_buffer_size)].push_back(source_idx)
    last_bucket[source_idx] = 0

    # ---- per-thread buffers ----
    # Guard slack must cover the worst case between two guard checks: one
    # full chunk of relaxations (CHUNK * out_degree successful inserts)
    # plus a complete flush of the fusion stack. The capacity floor makes
    # the silent-drop branch in _relax_one unreachable by construction.
    cdef int stack_cap = 4096 if fusion_cap > 0 else 0
    cdef int stack_alloc = stack_cap if stack_cap > 0 else 1
    cdef int chunk_margin = 64 * out_degree + 64
    cdef int min_cap = stack_alloc + 4 * chunk_margin
    cdef int ins_cap = calculate_thread_buffer_capacity(
        total_cells, actual_threads, &sys_limits)
    if ins_cap < min_cap:
        ins_cap = min_cap
    cdef FusedThread* T = <FusedThread*>calloc(actual_threads, sizeof(FusedThread))
    if T == NULL:
        raise MemoryError("Could not allocate thread state")
    cdef int tid_i
    cdef bint alloc_failed = False
    for tid_i in range(actual_threads):
        T[tid_i].ins_v = <uint64_t*>malloc(ins_cap * sizeof(uint64_t))
        T[tid_i].ins_d = <float*>malloc(ins_cap * sizeof(float))
        T[tid_i].stack = <uint64_t*>malloc(stack_alloc * sizeof(uint64_t))
        T[tid_i].settled = <uint64_t*>malloc(ins_cap * sizeof(uint64_t))
        if (T[tid_i].ins_v == NULL or T[tid_i].ins_d == NULL or
                T[tid_i].stack == NULL or T[tid_i].settled == NULL):
            alloc_failed = True
        T[tid_i].ins_cap = ins_cap
        T[tid_i].ins_guard = ins_cap - stack_alloc - chunk_margin
        T[tid_i].stack_cap = stack_cap
        T[tid_i].settled_cap = ins_cap
        T[tid_i].settled_guard = ins_cap - chunk_margin
        T[tid_i].ins_count = 0
        T[tid_i].top = 0
        T[tid_i].settled_count = 0
    if alloc_failed:
        for tid_i in range(actual_threads):
            if T[tid_i].ins_v != NULL: free(T[tid_i].ins_v)
            if T[tid_i].ins_d != NULL: free(T[tid_i].ins_d)
            if T[tid_i].stack != NULL: free(T[tid_i].stack)
            if T[tid_i].settled != NULL: free(T[tid_i].settled)
        free(T)
        raise MemoryError("Could not allocate thread buffers")

    # ---- shared state ----
    cdef FusedState ps_data
    cdef FusedState* ps = &ps_data
    ps.window_start = 0
    ps.window_len = window_init
    ps.logical_bucket_count = 1
    ps.window_id = 0
    ps.done_flag = 0
    ps.dirty = 0
    ps.inner_rounds = 0
    ps.barrier_arrive_count = 0
    ps.barrier_sense = 0
    ps.work_idx = 0
    ps.n_vertices = 0
    ps.n_heavy = 0
    ps.target_found_flag = 0
    ps.cutoff_distance = INF_F32
    ps.vertices_ptr = NULL
    ps.heavy_ptr = NULL

    cdef vector[uint64_t] work
    cdef vector[uint64_t] heavy_work
    # Window-wide settled list, drained from the per-thread buffers every
    # round by thread 0 so the per-thread guard is per-round (bounded by
    # round work), never per-window (unbounded on low-cost rasters).
    cdef vector[uint64_t] settled_window

    # ---- loop-local declarations (thread-private inside prange) ----
    cdef int tid, local_sense, chunk_start, chunk_end, j, b_i, pops
    cdef int CHUNK = 64
    cdef int max_inner_rounds = <int>min(<uint64_t>total_cells + 100000,
                                         <uint64_t>2000000000)
    cdef size_t b, phys, new_logical, scan_limit, w_len
    cdef uint64_t u, v_add
    cdef float dval, ws_dist, we_dist, span_w, tdist
    cdef size_t settled_total
    cdef int32_t lb
    cdef size_t grow_thr = <size_t>(actual_threads * 64 * 2)
    cdef size_t shrink_thr = <size_t>(actual_threads * 64 * 16)
    cdef bint target_found = False

    try:
        with nogil:
            for tid in prange(actual_threads, schedule='static'):
                local_sense = 0
                while True:
                    # ======= P1: thread 0 prepares the next window =======
                    if tid == 0:
                        ps.dirty = 0
                        ps.inner_rounds = 0
                        scan_limit = ps.logical_bucket_count + circular_buffer_size
                        while ps.window_start < scan_limit:
                            phys = get_circular_index(ps.window_start,
                                                      circular_buffer_size)
                            if not buckets[phys].empty():
                                break
                            ps.window_start = ps.window_start + 1
                        work.clear()
                        if ps.window_start >= scan_limit:
                            ps.done_flag = 1
                        else:
                            w_len = ps.window_len
                            ws_dist = <float>ps.window_start * delta
                            for b in range(ps.window_start,
                                           ps.window_start + w_len):
                                phys = get_circular_index(b, circular_buffer_size)
                                for j in range(<int>buckets[phys].size()):
                                    u = buckets[phys][j]
                                    # skip vertices settled in earlier windows
                                    dval = unpack_dist(
                                        atomic_load_u64(&dist_pred_ptr[u]))
                                    if dval >= ws_dist and dval < INF_F32:
                                        work.push_back(u)
                                buckets[phys].clear()
                            ps.window_id = ps.window_id + 1
                            ps.n_vertices = <int>work.size()
                            ps.vertices_ptr = work.data()
                            ps.work_idx = 0
                            settled_window.clear()
                            for j in range(actual_threads):
                                T[j].ins_count = 0
                                T[j].top = 0
                                T[j].settled_count = 0

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                        <volatile int*>&ps.barrier_sense,
                                        actual_threads, &local_sense)
                    if ps.done_flag:
                        break

                    ws_dist = <float>ps.window_start * delta
                    we_dist = <float>(ps.window_start + ps.window_len) * delta
                    span_w = <float>ps.window_len * delta

                    # ======= P2: light rounds until window fixed point =======
                    while True:
                        pops = 0
                        while True:
                            if (T[tid].ins_count >= T[tid].ins_guard or
                                    T[tid].settled_count >= T[tid].settled_guard):
                                break
                            chunk_start = atomic_fetch_add_int(
                                <volatile int*>&ps.work_idx, CHUNK)
                            if chunk_start >= ps.n_vertices:
                                break
                            chunk_end = chunk_start + CHUNK
                            if chunk_end > ps.n_vertices:
                                chunk_end = ps.n_vertices
                            for j in range(chunk_start, chunk_end):
                                _relax_one(ps.vertices_ptr[j], &T[tid],
                                           dist_pred_ptr, raster_view,
                                           exclude_view, directions,
                                           cached_steps, rows, cols,
                                           span_w, we_dist, False,
                                           stamp, pending, ps.window_id,
                                           total_cells)
                            # bucket fusion: drain the local stack (capped)
                            while (T[tid].top > 0 and pops < fusion_cap and
                                   T[tid].ins_count < T[tid].ins_guard and
                                   T[tid].settled_count < T[tid].settled_guard):
                                T[tid].top = T[tid].top - 1
                                _relax_one(T[tid].stack[T[tid].top], &T[tid],
                                           dist_pred_ptr, raster_view,
                                           exclude_view, directions,
                                           cached_steps, rows, cols,
                                           span_w, we_dist, False,
                                           stamp, pending, ps.window_id,
                                           total_cells)
                                pops = pops + 1
                        # drain what remains within the cap, then flush
                        while (T[tid].top > 0 and pops < fusion_cap and
                               T[tid].ins_count < T[tid].ins_guard and
                               T[tid].settled_count < T[tid].settled_guard):
                            T[tid].top = T[tid].top - 1
                            _relax_one(T[tid].stack[T[tid].top], &T[tid],
                                       dist_pred_ptr, raster_view,
                                       exclude_view, directions,
                                       cached_steps, rows, cols,
                                       span_w, we_dist, False,
                                       stamp, pending, ps.window_id,
                                       total_cells)
                            pops = pops + 1
                        while T[tid].top > 0:
                            T[tid].top = T[tid].top - 1
                            u = T[tid].stack[T[tid].top]
                            if T[tid].ins_count < T[tid].ins_cap:
                                T[tid].ins_v[T[tid].ins_count] = u
                                T[tid].ins_d[T[tid].ins_count] = unpack_dist(
                                    atomic_load_u64(&dist_pred_ptr[u]))
                                T[tid].ins_count = T[tid].ins_count + 1

                        thread_barrier_wait(
                            <volatile int*>&ps.barrier_arrive_count,
                            <volatile int*>&ps.barrier_sense,
                            actual_threads, &local_sense)

                        if tid == 0:
                            ps.dirty = 0
                            ps.inner_rounds = ps.inner_rounds + 1
                            # unfinished work rolls over (buffer-guard exits)
                            b_i = ps.work_idx
                            if b_i < ps.n_vertices:
                                for j in range(b_i, ps.n_vertices):
                                    work[j - b_i] = work[j]
                                work.resize(ps.n_vertices - b_i)
                                ps.dirty = 1
                            else:
                                work.clear()
                            # drain per-thread settled buffers every round
                            for j in range(actual_threads):
                                for b_i in range(T[j].settled_count):
                                    settled_window.push_back(T[j].settled[b_i])
                                T[j].settled_count = 0
                            for j in range(actual_threads):
                                for b_i in range(T[j].ins_count):
                                    v_add = T[j].ins_v[b_i]
                                    dval = T[j].ins_d[b_i]
                                    new_logical = <size_t>(dval / delta)
                                    if new_logical < (ps.window_start +
                                                      ps.window_len):
                                        work.push_back(v_add)
                                        ps.dirty = 1
                                    else:
                                        phys = get_circular_index(
                                            new_logical, circular_buffer_size)
                                        lb = last_bucket[v_add]
                                        if lb != <int32_t>new_logical:
                                            buckets[phys].push_back(v_add)
                                            last_bucket[v_add] = \
                                                <int32_t>new_logical
                                        if new_logical >= \
                                                ps.logical_bucket_count:
                                            ps.logical_bucket_count = \
                                                new_logical + 1
                                T[j].ins_count = 0
                            if ps.inner_rounds >= max_inner_rounds:
                                ps.dirty = 0
                            ps.n_vertices = <int>work.size()
                            ps.vertices_ptr = work.data()
                            ps.work_idx = 0

                        thread_barrier_wait(
                            <volatile int*>&ps.barrier_arrive_count,
                            <volatile int*>&ps.barrier_sense,
                            actual_threads, &local_sense)
                        if not ps.dirty:
                            break

                    # ======= P3: thread 0 checks target / termination =======
                    if tid == 0:
                        if not ps.target_found_flag:
                            tdist = unpack_dist(
                                atomic_load_u64(&dist_pred_ptr[target_idx]))
                            if tdist < we_dist:
                                # window closure => this distance is final
                                ps.target_found_flag = 1
                                ps.cutoff_distance = tdist * termination_margin
                        if (ps.target_found_flag and
                                ws_dist > ps.cutoff_distance):
                            ps.done_flag = 1
                        if not ps.done_flag:
                            # build heavy work list from the window's settled
                            heavy_work.clear()
                            for j in range(<int>settled_window.size()):
                                heavy_work.push_back(settled_window[j])
                            ps.n_heavy = <int>heavy_work.size()
                            ps.heavy_ptr = heavy_work.data()
                            ps.work_idx = 0
                            ps.dirty = 0

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                        <volatile int*>&ps.barrier_sense,
                                        actual_threads, &local_sense)
                    if ps.done_flag:
                        break

                    # ======= P4: heavy rounds (with rollover) =======
                    while True:
                        while True:
                            if T[tid].ins_count >= T[tid].ins_guard:
                                break
                            chunk_start = atomic_fetch_add_int(
                                <volatile int*>&ps.work_idx, CHUNK)
                            if chunk_start >= ps.n_heavy:
                                break
                            chunk_end = chunk_start + CHUNK
                            if chunk_end > ps.n_heavy:
                                chunk_end = ps.n_heavy
                            for j in range(chunk_start, chunk_end):
                                _relax_one(ps.heavy_ptr[j], &T[tid],
                                           dist_pred_ptr, raster_view,
                                           exclude_view, directions,
                                           cached_steps, rows, cols,
                                           span_w, we_dist, True,
                                           stamp, pending, ps.window_id,
                                           total_cells)

                        thread_barrier_wait(
                            <volatile int*>&ps.barrier_arrive_count,
                            <volatile int*>&ps.barrier_sense,
                            actual_threads, &local_sense)

                        if tid == 0:
                            ps.dirty = 0
                            b_i = ps.work_idx
                            if b_i < ps.n_heavy:
                                for j in range(b_i, ps.n_heavy):
                                    heavy_work[j - b_i] = heavy_work[j]
                                heavy_work.resize(ps.n_heavy - b_i)
                                ps.dirty = 1
                            else:
                                heavy_work.clear()
                            for j in range(actual_threads):
                                for b_i in range(T[j].ins_count):
                                    v_add = T[j].ins_v[b_i]
                                    dval = T[j].ins_d[b_i]
                                    new_logical = <size_t>(dval / delta)
                                    if new_logical < (ps.window_start +
                                                      ps.window_len):
                                        # float-boundary safety clamp
                                        new_logical = ps.window_start + \
                                            ps.window_len
                                    phys = get_circular_index(
                                        new_logical, circular_buffer_size)
                                    lb = last_bucket[v_add]
                                    if lb != <int32_t>new_logical:
                                        buckets[phys].push_back(v_add)
                                        last_bucket[v_add] = \
                                            <int32_t>new_logical
                                    if new_logical >= ps.logical_bucket_count:
                                        ps.logical_bucket_count = \
                                            new_logical + 1
                                T[j].ins_count = 0
                            ps.n_heavy = <int>heavy_work.size()
                            ps.heavy_ptr = heavy_work.data()
                            ps.work_idx = 0

                        thread_barrier_wait(
                            <volatile int*>&ps.barrier_arrive_count,
                            <volatile int*>&ps.barrier_sense,
                            actual_threads, &local_sense)
                        if not ps.dirty:
                            break

                    # ======= P5: thread 0 advances window, adapts W =======
                    if tid == 0:
                        settled_total = settled_window.size()
                        ps.window_start = ps.window_start + ps.window_len
                        if adaptive_window:
                            if settled_total < grow_thr:
                                if ps.window_len * 2 <= window_max:
                                    ps.window_len = ps.window_len * 2
                            elif settled_total > shrink_thr:
                                if ps.window_len > 1:
                                    ps.window_len = ps.window_len // 2

                    thread_barrier_wait(<volatile int*>&ps.barrier_arrive_count,
                                        <volatile int*>&ps.barrier_sense,
                                        actual_threads, &local_sense)

        target_found = <bint>ps.target_found_flag
    finally:
        for tid_i in range(actual_threads):
            if T[tid_i].ins_v != NULL: free(T[tid_i].ins_v)
            if T[tid_i].ins_d != NULL: free(T[tid_i].ins_d)
            if T[tid_i].stack != NULL: free(T[tid_i].stack)
            if T[tid_i].settled != NULL: free(T[tid_i].settled)
        free(T)

    # ---- path reconstruction (same contract as the baseline) ----
    cdef uint32_t pred_val = unpack_pred(dist_pred_ptr[target_idx])
    if not target_found or pred_val == 0xFFFFFFFF:
        return np.empty(0, dtype=np.uint64)

    cdef list path_vertices = []
    cdef uint64_t current = target_idx
    cdef uint64_t path_length = 0
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

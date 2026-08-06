# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False

"""Parallel constrained delta-stepping variants.

Extracted from constrained_path_algorithms.pyx.
Contains:
- constrained_delta_stepping_2d          (basic parallel constrained)
- constrained_delta_stepping_clearance_2d (fixed-height clearance)
- constrained_delta_stepping_height_2d   (variable-height clearance)
- _height_sparse                         (compact-dense variable-height)
- constrained_delta_stepping_lazy        (lazy hash-map variant)
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t, UINT64_MAX
from libc.math cimport INFINITY, fabsf, sqrtf, expf
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref
from cython.parallel cimport prange, threadid
from openmp cimport omp_get_max_threads, omp_set_num_threads

from pyorps.utils._heap cimport (
    StepData, CachedStepData, IntermediatePoint, npy_intp,
    BinaryHeap64, PQNode64,
    heap64_init, heap64_push, heap64_pop, heap64_top, heap64_empty, heap64_free,
    SystemLimits, get_system_limits,
)
from pyorps.utils._raster_context cimport (
    precompute_directions, precompute_cached_steps,
)
from pyorps.utils._constrained_context cimport (
    ValidNeighbor, StateData, CRelaxBuf, LazyState,
    FLAG_TOUCHED, FLAG_VISITED,
    _pack_state, _unpack_state, _pack_state_h, _unpack_state_h,
    _check_intermediates_ptr,
    _build_valid_neighbors, _flatten_valid_neighbors,
    _precompute_intermediate_cache, _precompute_gradient_cache,
    _check_span_clearance, _check_span_clearance_vh,
    _tower_terrain,
)


# ===========================================================================
# 1. BASIC PARALLEL CONSTRAINED DELTA-STEPPING
# ===========================================================================

def constrained_delta_stepping_2d(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col,
    int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins,
    float span_bin_size,
    float min_span,
    float max_span,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask=None,
    np.ndarray[np.float32_t, ndim=2] dem_data=None,
    float cell_size=1.0,
    float max_gradient_pct=100.0,
    float gradient_scale=2.0,
    np.ndarray[np.int32_t, ndim=1] area_offsets=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_starts=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_counts=None,
):
    """Parallel constrained delta-stepping with tower placement.

    Uses OpenMP thread-parallel edge relaxation within each bucket phase.
    Thread-local buffers collect results; sequential merge maintains correctness.
    Same result as constrained_dijkstra_2d but leverages multiple cores.
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int spc = n_dirs * n_span_bins
    cdef uint64_t total_states = <uint64_t>total_cells * spc

    raster = np.ascontiguousarray(raster)
    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)
    else:
        exclude_mask = np.ascontiguousarray(exclude_mask)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs

    cdef uint16_t* raster_ptr = &raster_view[0, 0]
    cdef uint8_t* mask_ptr = &mask_view[0, 0]
    cdef float* tower_terrain_ptr = &tower_terrain_view[0]

    # Area cost offsets (exact mode)
    cdef int use_area_cost = 0
    cdef int32_t[:] ao_view, as_view, ac_view
    cdef int32_t* area_offsets_ptr = NULL
    cdef int32_t* area_starts_ptr = NULL
    cdef int32_t* area_counts_ptr = NULL
    if area_offsets is not None:
        use_area_cost = 1
        ao_view = area_offsets
        as_view = area_offset_starts
        ac_view = area_offset_counts
        area_offsets_ptr = &ao_view[0]
        area_starts_ptr = &as_view[0]
        area_counts_ptr = &ac_view[0]

    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs,
    )
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    # Compute delta
    cdef double min_raster_val_d = 1.0
    traversable = raster[raster != 65535]
    if traversable.size > 0:
        min_raster_val_d = <double>max(1, int(traversable.min()))
    cdef double min_cf = 1e9
    cdef int dd
    for dd in range(n_dirs):
        if directions[dd].cost_factor < min_cf:
            min_cf = directions[dd].cost_factor
    cdef double delta = max(1.0, 2.0 * min_raster_val_d * min_cf * (1.0 - 1e-9))

    # Bucket queue
    cdef size_t n_phys_buckets = 65536
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    # AoS state array
    cdef StateData* states = <StateData*>calloc(total_states, sizeof(StateData))

    # Precomputed intermediate cache
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))

    if states == NULL or icache_status == NULL or icache_cost == NULL:
        if states != NULL: free(states)
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate memory for constrained delta-stepping")

    _precompute_intermediate_cache(
        total_cells, n_dirs, rows, cols,
        mask_ptr, raster_ptr, directions, cached_steps,
        icache_status, icache_cost,
    )

    # 3D gradient precomputation (if DEM provided)
    cdef float* grad_penalty_ptr = NULL
    cdef float* dem_ptr_f = NULL
    cdef float[:, :] dem_view
    if dem_data is not None:
        dem_data = np.ascontiguousarray(dem_data)
        dem_view = dem_data
        dem_ptr_f = &dem_view[0, 0]
        grad_penalty_ptr = <float*>malloc(cache_size * sizeof(float))
        if grad_penalty_ptr == NULL:
            free(states); free(icache_status); free(icache_cost)
            raise MemoryError("Failed to allocate gradient cache")
        _precompute_gradient_cache(
            total_cells, n_dirs, rows, cols,
            dem_ptr_f, cell_size, max_gradient_pct, gradient_scale,
            directions, icache_status, grad_penalty_ptr,
        )

    # Thread-local result buffers
    import os as _os
    cdef int num_threads = int(_os.environ.get('OMP_NUM_THREADS', str(_os.cpu_count() or 4)))
    if num_threads < 1:
        num_threads = 1
    cdef int buf_cap = 131072  # 128K entries per thread
    cdef CRelaxBuf* tbufs = <CRelaxBuf*>malloc(num_threads * sizeof(CRelaxBuf))
    cdef int t
    if tbufs == NULL:
        free(states); free(icache_status); free(icache_cost)
        raise MemoryError("Failed to allocate thread buffers")
    for t in range(num_threads):
        tbufs[t].states = <uint64_t*>malloc(buf_cap * sizeof(uint64_t))
        tbufs[t].dists = <double*>malloc(buf_cap * sizeof(double))
        tbufs[t].preds = <int64_t*>malloc(buf_cap * sizeof(int64_t))
        tbufs[t].span_dists = <float*>malloc(buf_cap * sizeof(float))
        tbufs[t].count = 0
        tbufs[t].capacity = buf_cap

    # Initialize source
    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx
    cdef int d
    for d in range(n_dirs):
        state_idx = (<uint64_t>source_cell) * spc + d * n_span_bins
        states[state_idx].touched = 1
        states[state_idx].dist = 0.0
        states[state_idx].pred = -1
        states[state_idx].span_dist = 0.0
        buckets[0].push_back(state_idx)

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY

    cdef size_t current_logical = 0
    cdef size_t max_logical = 0
    cdef size_t phys_idx, new_logical

    # Active states for parallel processing
    cdef vector[uint64_t] active
    cdef vector[uint64_t] batch
    cdef vector[uint64_t] deferred

    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef size_t entry_logical
    cdef size_t bi

    # Parallel loop variables
    cdef int ai, n_active, tid
    cdef uint8_t p_cur_dir
    cdef int p_cur_row, p_cur_col
    cdef double p_cur_span_m, p_cur_dist
    cdef uint16_t p_cur_raster_val
    cdef double p_cur_tower_terrain
    cdef uint64_t p_remainder
    cdef int p_nr, p_nc, p_k, p_n_valid, p_nb_start
    cdef uint32_t p_nb_cell, p_cur_cell
    cdef double p_edge_cost, p_terrain_cost, p_new_dist, p_tower_cost
    cdef double p_new_span_m, p_reset_span_m
    cdef uint16_t p_new_span_bin, p_reset_span_bin
    cdef uint64_t p_new_state, p_base_nb
    cdef float p_icost
    cdef size_t p_cidx
    cdef ValidNeighbor p_nb
    cdef int p_skip, p_sb, p_cnt

    try:
        while current_logical <= max_logical:
            if <double>current_logical * delta > best_target_dist:
                break

            phys_idx = current_logical & bucket_mask

            if buckets[phys_idx].size() == 0:
                current_logical += 1
                continue

            deferred.clear()

            # Process bucket: extract -> filter -> parallel relax -> merge
            while buckets[phys_idx].size() > 0:
                batch.swap(buckets[phys_idx])
                # Livelock fix: the swap put the PREVIOUS (already processed)
                # batch into the bucket -- discard it, or the two vectors
                # oscillate forever once any same-bucket push occurs.
                buckets[phys_idx].clear()

                # Sequential filter: identify active (unvisited, correct bucket) states
                active.clear()
                for bi in range(batch.size()):
                    cur_state = batch[bi]
                    if states[cur_state].visited != 0:
                        continue
                    cur_dist = states[cur_state].dist
                    if cur_dist >= best_target_dist:
                        continue
                    entry_logical = <size_t>(cur_dist / delta)
                    if entry_logical != current_logical:
                        if entry_logical > current_logical:
                            deferred.push_back(cur_state)
                        continue
                    # Mark visited
                    states[cur_state].visited = 1
                    # Check target
                    cur_cell = <uint32_t>(cur_state / spc)
                    if cur_cell == target_cell:
                        if cur_dist < best_target_dist:
                            best_target_dist = cur_dist
                            best_target_state = cur_state
                        continue
                    active.push_back(cur_state)

                n_active = <int>active.size()
                if n_active == 0:
                    break

                # Reset thread buffers
                for t in range(num_threads):
                    tbufs[t].count = 0

                # ---- PARALLEL EDGE RELAXATION ----
                for ai in prange(n_active, nogil=True, schedule='dynamic',
                                 chunksize=64, num_threads=num_threads):
                    tid = threadid()
                    if tid < 0 or tid >= num_threads:
                        tid = 0

                    cur_state = active[ai]
                    p_cur_dist = states[cur_state].dist
                    p_cur_cell = <uint32_t>(cur_state / spc)
                    p_remainder = cur_state - (<uint64_t>p_cur_cell) * spc
                    p_cur_dir = <uint8_t>(p_remainder / n_span_bins)
                    p_cur_row = <int>(p_cur_cell / cols)
                    p_cur_col = <int>(p_cur_cell % cols)
                    p_cur_span_m = <double>states[cur_state].span_dist
                    p_cur_raster_val = raster_ptr[p_cur_cell]
                    p_cur_tower_terrain = <double>tower_terrain_ptr[p_cur_raster_val]

                    p_nb_start = nb_offsets[p_cur_dir]
                    p_n_valid = nb_offsets[p_cur_dir + 1] - p_nb_start

                    for p_k in range(p_n_valid):
                        p_nb = flat_nb[p_nb_start + p_k]
                        p_nr = p_cur_row + p_nb.dr
                        p_nc = p_cur_col + p_nb.dc

                        if (<unsigned int>p_nr >= <unsigned int>rows or
                                <unsigned int>p_nc >= <unsigned int>cols):
                            continue

                        p_nb_cell = <uint32_t>(p_nr * cols + p_nc)
                        if mask_ptr[p_nb_cell] == 0:
                            continue

                        # Pre-visited skip
                        p_base_nb = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins
                        p_skip = 1
                        for p_sb in range(n_span_bins):
                            if (states[p_base_nb + p_sb].visited == 0 or
                                    <double>states[p_base_nb + p_sb].dist > p_cur_dist):
                                p_skip = 0
                                break
                        if p_skip:
                            continue

                        # Intermediate cache lookup
                        p_cidx = <size_t>p_nb_cell * <size_t>n_dirs + <size_t>p_nb.d_out
                        if icache_status[p_cidx] != 2:
                            continue
                        p_icost = icache_cost[p_cidx]

                        p_terrain_cost = (<double>p_cur_raster_val +
                                         <double>p_icost +
                                         <double>raster_ptr[p_nb_cell]) * <double>p_nb.cost_factor
                        if grad_penalty_ptr != NULL:
                            p_terrain_cost = p_terrain_cost * <double>grad_penalty_ptr[p_cidx]
                        p_edge_cost = p_terrain_cost + <double>p_nb.angle_cost
                        p_new_span_m = p_cur_span_m + <double>p_nb.step_distance

                        if p_nb.d_out == p_cur_dir:
                            # Same direction: continue span
                            if p_new_span_m < <double>max_span:
                                p_new_span_bin = <uint16_t>(p_new_span_m / span_bin_size)
                                p_new_state = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins + p_new_span_bin
                                p_new_dist = p_cur_dist + p_edge_cost
                                if (states[p_new_state].visited == 0 or
                                        p_new_dist < <double>states[p_new_state].dist):
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_new_span_m
                                        tbufs[tid].count = p_cnt + 1

                            # Same direction: optional tower
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                p_tower_cost = _tower_terrain(
                                    use_area_cost, p_cur_row, p_cur_col,
                                    p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                    raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                    area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                    dem_ptr_f, cell_size, gradient_scale,
                                ) + <double>p_nb.tower_angle_cost
                                p_reset_span_m = <double>p_nb.step_distance
                                p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                p_new_state = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins + p_reset_span_bin
                                p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                if (states[p_new_state].visited == 0 or
                                        p_new_dist < <double>states[p_new_state].dist):
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                        tbufs[tid].count = p_cnt + 1
                        else:
                            # Direction change: mandatory tower (min_span enforced)
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                p_tower_cost = _tower_terrain(
                                    use_area_cost, p_cur_row, p_cur_col,
                                    p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                    raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                    area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                    dem_ptr_f, cell_size, gradient_scale,
                                ) + <double>p_nb.tower_angle_cost
                                p_reset_span_m = <double>p_nb.step_distance
                                p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                p_new_state = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins + p_reset_span_bin
                                p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                if (states[p_new_state].visited == 0 or
                                        p_new_dist < <double>states[p_new_state].dist):
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                        tbufs[tid].count = p_cnt + 1

                    # ---- SEQUENTIAL MERGE ----
                for t in range(num_threads):
                    for bi in range(tbufs[t].count):
                        p_new_state = tbufs[t].states[bi]
                        p_new_dist = tbufs[t].dists[bi]
                        if (states[p_new_state].visited == 0 or
                                p_new_dist < states[p_new_state].dist):
                            if states[p_new_state].touched == 0 or p_new_dist < states[p_new_state].dist:
                                states[p_new_state].touched = 1
                                # Re-open protocol: a settled state improved
                                # (possible when delta > min edge cost, e.g.
                                # 0-cost cells) returns to the frontier.
                                states[p_new_state].visited = 0
                                states[p_new_state].dist = p_new_dist
                                states[p_new_state].pred = tbufs[t].preds[bi]
                                states[p_new_state].span_dist = tbufs[t].span_dists[bi]
                                new_logical = <size_t>(p_new_dist / delta)
                                buckets[new_logical & bucket_mask].push_back(p_new_state)
                                if new_logical > max_logical:
                                    max_logical = new_logical

            # Re-insert deferred
            for bi in range(deferred.size()):
                cur_state = deferred[bi]
                if states[cur_state].visited == 0:
                    new_logical = <size_t>(states[cur_state].dist / delta)
                    if new_logical > current_logical:
                        buckets[new_logical & bucket_mask].push_back(cur_state)

            current_logical += 1

    finally:
        # Free thread buffers
        if tbufs != NULL:
            for t in range(num_threads):
                if tbufs[t].states != NULL: free(tbufs[t].states)
                if tbufs[t].dists != NULL: free(tbufs[t].dists)
                if tbufs[t].preds != NULL: free(tbufs[t].preds)
                if tbufs[t].span_dists != NULL: free(tbufs[t].span_dists)
            free(tbufs)

    # Path reconstruction
    cdef list path_cells = []
    cdef list tower_cells = []

    if best_target_state == UINT64_MAX:
        free(states); free(icache_status); free(icache_cost)
        if grad_penalty_ptr != NULL: free(grad_penalty_ptr)
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir
    cdef uint16_t prev_span

    cdef list state_chain = []
    while walk_state != UINT64_MAX and states[walk_state].pred != -1:
        state_chain.append(walk_state)
        walk_state = <uint64_t>states[walk_state].pred
    state_chain.append(walk_state)
    state_chain.reverse()

    cdef uint64_t st, prev_state_val
    cdef int k
    for k in range(len(state_chain)):
        st = <uint64_t>state_chain[k]
        _unpack_state(st, n_dirs, n_span_bins, &walk_cell, &walk_dir, &walk_span)
        path_cells.append(<int>walk_cell)

        if k > 0 and n_span_bins > 1:
            prev_state_val = <uint64_t>state_chain[k - 1]
            _unpack_state(prev_state_val, n_dirs, n_span_bins,
                         &prev_cell, &prev_dir, &prev_span)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)

    free(states); free(icache_status); free(icache_cost)
    if grad_penalty_ptr != NULL: free(grad_penalty_ptr)

    return (
        np.array(path_cells, dtype=np.uint32),
        np.array(tower_cells, dtype=np.uint32),
    )


# ===========================================================================
# 2. PARALLEL 3D DELTA-STEPPING WITH CLEARANCE
# ===========================================================================

def constrained_delta_stepping_clearance_2d(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col, int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins, float span_bin_size, float min_span, float max_span,
    np.ndarray[np.float32_t, ndim=2] dem_data,
    float cell_size, float tower_height,
    float conductor_weight_per_m, float conductor_tension, float min_clearance_val,
    np.ndarray[np.float32_t, ndim=2] obstacle_heights=None,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask=None,
    float max_gradient_pct=100.0, float gradient_scale=2.0,
):
    """Parallel 3D delta-stepping with catenary clearance checking.

    At every tower placement, validates conductor sag clearance above
    ground + obstacles along the entire span. Guarantees globally optimal
    feasible path.
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int spc = n_dirs * n_span_bins
    cdef uint64_t total_states = <uint64_t>total_cells * spc

    raster = np.ascontiguousarray(raster)
    dem_data = np.ascontiguousarray(dem_data)
    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)
    else:
        exclude_mask = np.ascontiguousarray(exclude_mask)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs
    cdef float[:, :] dem_view = dem_data

    cdef uint16_t* raster_ptr = &raster_view[0, 0]
    cdef uint8_t* mask_ptr = &mask_view[0, 0]
    cdef float* tower_terrain_ptr = &tower_terrain_view[0]
    cdef float* dem_ptr = &dem_view[0, 0]
    cdef float* obstacle_ptr = NULL
    cdef float[:, :] obs_view
    if obstacle_heights is not None:
        obstacle_heights = np.ascontiguousarray(obstacle_heights)
        obs_view = obstacle_heights
        obstacle_ptr = &obs_view[0, 0]

    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs)
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    cdef double min_raster_val_d = 1.0
    traversable = raster[raster != 65535]
    if traversable.size > 0:
        min_raster_val_d = <double>max(1, int(traversable.min()))
    cdef double min_cf = 1e9
    cdef int dd
    for dd in range(n_dirs):
        if directions[dd].cost_factor < min_cf:
            min_cf = directions[dd].cost_factor
    cdef double delta_val = max(1.0, 2.0 * min_raster_val_d * min_cf * (1.0 - 1e-9))

    cdef size_t n_phys_buckets = 65536
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    cdef StateData* states = <StateData*>calloc(total_states, sizeof(StateData))
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))
    if states == NULL or icache_status == NULL or icache_cost == NULL:
        if states != NULL: free(states)
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate memory")

    _precompute_intermediate_cache(total_cells, n_dirs, rows, cols,
        mask_ptr, raster_ptr, directions, cached_steps, icache_status, icache_cost)

    cdef float* grad_penalty_ptr = <float*>malloc(cache_size * sizeof(float))
    if grad_penalty_ptr == NULL:
        free(states); free(icache_status); free(icache_cost)
        raise MemoryError("Failed to allocate gradient cache")
    _precompute_gradient_cache(total_cells, n_dirs, rows, cols,
        dem_ptr, cell_size, max_gradient_pct, gradient_scale,
        directions, icache_status, grad_penalty_ptr)

    import os as _os
    cdef int num_threads = int(_os.environ.get('OMP_NUM_THREADS', str(_os.cpu_count() or 4)))
    if num_threads < 1:
        num_threads = 1
    cdef int buf_cap = 131072
    cdef CRelaxBuf* tbufs = <CRelaxBuf*>malloc(num_threads * sizeof(CRelaxBuf))
    cdef int t
    if tbufs == NULL:
        free(states); free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        raise MemoryError("Failed to allocate thread buffers")
    for t in range(num_threads):
        tbufs[t].states = <uint64_t*>malloc(buf_cap * sizeof(uint64_t))
        tbufs[t].dists = <double*>malloc(buf_cap * sizeof(double))
        tbufs[t].preds = <int64_t*>malloc(buf_cap * sizeof(int64_t))
        tbufs[t].span_dists = <float*>malloc(buf_cap * sizeof(float))
        tbufs[t].count = 0
        tbufs[t].capacity = buf_cap

    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx
    cdef int d
    for d in range(n_dirs):
        state_idx = (<uint64_t>source_cell) * spc + d * n_span_bins
        states[state_idx].touched = 1
        states[state_idx].dist = 0.0
        states[state_idx].pred = -1
        states[state_idx].span_dist = 0.0
        buckets[0].push_back(state_idx)

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY
    cdef size_t current_logical = 0, max_logical = 0, phys_idx, new_logical
    cdef vector[uint64_t] active, batch, deferred
    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef size_t entry_logical, bi

    cdef int ai, n_active, tid
    cdef uint8_t p_cur_dir
    cdef int p_cur_row, p_cur_col, p_nr, p_nc, p_k, p_n_valid, p_nb_start
    cdef double p_cur_span_m, p_cur_dist, p_cur_tower_terrain
    cdef uint16_t p_cur_raster_val
    cdef uint32_t p_nb_cell, p_cur_cell
    cdef uint64_t p_remainder, p_new_state, p_base_nb
    cdef double p_edge_cost, p_terrain_cost, p_new_dist, p_tower_cost
    cdef double p_new_span_m, p_reset_span_m
    cdef uint16_t p_new_span_bin, p_reset_span_bin
    cdef float p_icost
    cdef size_t p_cidx
    cdef ValidNeighbor p_nb
    cdef int p_skip, p_sb, p_cnt, p_clearance_ok

    try:
        while current_logical <= max_logical:
            if <double>current_logical * delta_val > best_target_dist:
                break
            phys_idx = current_logical & bucket_mask
            if buckets[phys_idx].size() == 0:
                current_logical += 1
                continue
            deferred.clear()
            while buckets[phys_idx].size() > 0:
                batch.swap(buckets[phys_idx])
                # Livelock fix: the swap put the PREVIOUS (already processed)
                # batch into the bucket -- discard it, or the two vectors
                # oscillate forever once any same-bucket push occurs.
                buckets[phys_idx].clear()
                active.clear()
                for bi in range(batch.size()):
                    cur_state = batch[bi]
                    if states[cur_state].visited != 0:
                        continue
                    cur_dist = states[cur_state].dist
                    if cur_dist >= best_target_dist:
                        continue
                    entry_logical = <size_t>(cur_dist / delta_val)
                    if entry_logical != current_logical:
                        if entry_logical > current_logical:
                            deferred.push_back(cur_state)
                        continue
                    states[cur_state].visited = 1
                    cur_cell = <uint32_t>(cur_state / spc)
                    if cur_cell == target_cell:
                        if cur_dist < best_target_dist:
                            best_target_dist = cur_dist
                            best_target_state = cur_state
                        continue
                    active.push_back(cur_state)

                n_active = <int>active.size()
                if n_active == 0:
                    break
                for t in range(num_threads):
                    tbufs[t].count = 0

                for ai in prange(n_active, nogil=True, schedule='dynamic',
                                 chunksize=64, num_threads=num_threads):
                    tid = threadid()
                    if tid < 0 or tid >= num_threads:
                        tid = 0
                    cur_state = active[ai]
                    p_cur_dist = states[cur_state].dist
                    p_cur_cell = <uint32_t>(cur_state / spc)
                    p_remainder = cur_state - (<uint64_t>p_cur_cell) * spc
                    p_cur_dir = <uint8_t>(p_remainder / n_span_bins)
                    p_cur_row = <int>(p_cur_cell / cols)
                    p_cur_col = <int>(p_cur_cell % cols)
                    p_cur_span_m = <double>states[cur_state].span_dist
                    p_cur_raster_val = raster_ptr[p_cur_cell]
                    p_cur_tower_terrain = <double>tower_terrain_ptr[p_cur_raster_val]
                    p_nb_start = nb_offsets[p_cur_dir]
                    p_n_valid = nb_offsets[p_cur_dir + 1] - p_nb_start
                    for p_k in range(p_n_valid):
                        p_nb = flat_nb[p_nb_start + p_k]
                        p_nr = p_cur_row + p_nb.dr
                        p_nc = p_cur_col + p_nb.dc
                        if (<unsigned int>p_nr >= <unsigned int>rows or
                                <unsigned int>p_nc >= <unsigned int>cols):
                            continue
                        p_nb_cell = <uint32_t>(p_nr * cols + p_nc)
                        if mask_ptr[p_nb_cell] == 0:
                            continue
                        p_base_nb = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins
                        p_skip = 1
                        for p_sb in range(n_span_bins):
                            if (states[p_base_nb + p_sb].visited == 0 or
                                    <double>states[p_base_nb + p_sb].dist > p_cur_dist):
                                p_skip = 0
                                break
                        if p_skip:
                            continue
                        p_cidx = <size_t>p_nb_cell * <size_t>n_dirs + <size_t>p_nb.d_out
                        if icache_status[p_cidx] != 2:
                            continue
                        p_icost = icache_cost[p_cidx]
                        p_terrain_cost = (<double>p_cur_raster_val + <double>p_icost +
                                         <double>raster_ptr[p_nb_cell]) * <double>p_nb.cost_factor
                        p_terrain_cost = p_terrain_cost * <double>grad_penalty_ptr[p_cidx]
                        p_edge_cost = p_terrain_cost + <double>p_nb.angle_cost
                        p_new_span_m = p_cur_span_m + <double>p_nb.step_distance

                        if p_nb.d_out == p_cur_dir:
                            if p_new_span_m < <double>max_span:
                                p_new_span_bin = <uint16_t>(p_new_span_m / span_bin_size)
                                p_new_state = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins + p_new_span_bin
                                p_new_dist = p_cur_dist + p_edge_cost
                                if (states[p_new_state].visited == 0 or
                                        p_new_dist < <double>states[p_new_state].dist):
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_new_span_m
                                        tbufs[tid].count = p_cnt + 1
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                p_clearance_ok = _check_span_clearance(
                                    p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                    tower_height, conductor_weight_per_m, conductor_tension,
                                    min_clearance_val, dem_ptr, obstacle_ptr, rows, cols,
                                    directions, cached_steps, step_dist_view[p_cur_dir])
                                if p_clearance_ok:
                                    p_tower_cost = p_cur_tower_terrain + <double>p_nb.tower_angle_cost
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins + p_reset_span_bin
                                    if states[p_new_state].visited == 0:
                                        p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                        p_cnt = tbufs[tid].count
                                        if p_cnt < tbufs[tid].capacity:
                                            tbufs[tid].states[p_cnt] = p_new_state
                                            tbufs[tid].dists[p_cnt] = p_new_dist
                                            tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                            tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                            tbufs[tid].count = p_cnt + 1
                        else:
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                p_clearance_ok = _check_span_clearance(
                                    p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                    tower_height, conductor_weight_per_m, conductor_tension,
                                    min_clearance_val, dem_ptr, obstacle_ptr, rows, cols,
                                    directions, cached_steps, step_dist_view[p_cur_dir])
                                if p_clearance_ok:
                                    p_tower_cost = p_cur_tower_terrain + <double>p_nb.tower_angle_cost
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * n_span_bins + p_reset_span_bin
                                    if states[p_new_state].visited == 0:
                                        p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                        p_cnt = tbufs[tid].count
                                        if p_cnt < tbufs[tid].capacity:
                                            tbufs[tid].states[p_cnt] = p_new_state
                                            tbufs[tid].dists[p_cnt] = p_new_dist
                                            tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                            tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                            tbufs[tid].count = p_cnt + 1

                for t in range(num_threads):
                    for bi in range(tbufs[t].count):
                        p_new_state = tbufs[t].states[bi]
                        p_new_dist = tbufs[t].dists[bi]
                        if (states[p_new_state].visited == 0 or
                                p_new_dist < states[p_new_state].dist):
                            if states[p_new_state].touched == 0 or p_new_dist < states[p_new_state].dist:
                                states[p_new_state].touched = 1
                                # Re-open protocol: a settled state improved
                                # (possible when delta > min edge cost, e.g.
                                # 0-cost cells) returns to the frontier.
                                states[p_new_state].visited = 0
                                states[p_new_state].dist = p_new_dist
                                states[p_new_state].pred = tbufs[t].preds[bi]
                                states[p_new_state].span_dist = tbufs[t].span_dists[bi]
                                new_logical = <size_t>(p_new_dist / delta_val)
                                buckets[new_logical & bucket_mask].push_back(p_new_state)
                                if new_logical > max_logical:
                                    max_logical = new_logical

            for bi in range(deferred.size()):
                cur_state = deferred[bi]
                if states[cur_state].visited == 0:
                    new_logical = <size_t>(states[cur_state].dist / delta_val)
                    if new_logical > current_logical:
                        buckets[new_logical & bucket_mask].push_back(cur_state)
            current_logical += 1

    finally:
        if tbufs != NULL:
            for t in range(num_threads):
                if tbufs[t].states != NULL: free(tbufs[t].states)
                if tbufs[t].dists != NULL: free(tbufs[t].dists)
                if tbufs[t].preds != NULL: free(tbufs[t].preds)
                if tbufs[t].span_dists != NULL: free(tbufs[t].span_dists)
            free(tbufs)

    cdef list path_cells = []
    cdef list tower_cells = []
    if best_target_state == UINT64_MAX:
        free(states); free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir
    cdef uint16_t prev_span
    cdef list state_chain = []
    while walk_state != UINT64_MAX and states[walk_state].pred != -1:
        state_chain.append(walk_state)
        walk_state = <uint64_t>states[walk_state].pred
    state_chain.append(walk_state)
    state_chain.reverse()
    cdef uint64_t st, prev_state_val
    cdef int k
    for k in range(len(state_chain)):
        st = <uint64_t>state_chain[k]
        _unpack_state(st, n_dirs, n_span_bins, &walk_cell, &walk_dir, &walk_span)
        path_cells.append(<int>walk_cell)
        if k > 0 and n_span_bins > 1:
            prev_state_val = <uint64_t>state_chain[k - 1]
            _unpack_state(prev_state_val, n_dirs, n_span_bins,
                         &prev_cell, &prev_dir, &prev_span)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)

    free(states); free(icache_status); free(icache_cost); free(grad_penalty_ptr)
    return (np.array(path_cells, dtype=np.uint32), np.array(tower_cells, dtype=np.uint32))


# ===========================================================================
# 3. VARIABLE-HEIGHT DELTA-STEPPING WITH CLEARANCE
# ===========================================================================

def constrained_delta_stepping_height_2d(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col, int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins, float span_bin_size, float min_span, float max_span,
    np.ndarray[np.float32_t, ndim=2] dem_data,
    float cell_size,
    np.ndarray[np.float32_t, ndim=1] tower_heights,
    np.ndarray[np.float32_t, ndim=1] height_premiums,
    float conductor_weight_per_m, float conductor_tension, float min_clearance_val,
    np.ndarray[np.float32_t, ndim=2] obstacle_heights=None,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask=None,
    float max_gradient_pct=100.0, float gradient_scale=2.0,
    np.ndarray[np.int32_t, ndim=1] area_offsets=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_starts=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_counts=None,
):
    """3D delta-stepping with variable tower heights and clearance.

    State = (cell, direction, span_bin, height_class). At tower placement,
    all candidate heights for the new tower B are explored. Heights are sorted
    descending so clearance failure triggers early exit (shorter towers always
    fail if a taller one does).

    Automatically selects dense arrays when the state space fits in memory,
    otherwise uses sparse hash maps (heap-based Dijkstra).

    Returns:
        3-tuple: (path_cells uint32[], tower_cells uint32[], tower_heights float32[])
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int n_heights = tower_heights.shape[0]
    cdef int spc = n_dirs * n_span_bins * n_heights  # states per cell
    cdef uint64_t total_states = <uint64_t>total_cells * spc

    # Dense limit: ~12 GB with 24-byte StateData
    cdef uint64_t dense_limit = 500000000

    # Area cost offsets (exact mode) — declared early for sparse delegation
    cdef int use_area_cost = 0
    cdef int32_t[:] ao_view, as_view, ac_view
    cdef int32_t* area_offsets_ptr = NULL
    cdef int32_t* area_starts_ptr = NULL
    cdef int32_t* area_counts_ptr = NULL
    if area_offsets is not None:
        use_area_cost = 1
        ao_view = area_offsets
        as_view = area_offset_starts
        ac_view = area_offset_counts
        area_offsets_ptr = &ao_view[0]
        area_starts_ptr = &as_view[0]
        area_counts_ptr = &ac_view[0]

    if total_states > dense_limit:
        return _height_sparse(
            raster, source_row, source_col, target_row, target_col,
            steps, angle_cost_lut, angle_valid_lut, step_distances,
            tower_terrain_costs, tower_angle_costs,
            n_span_bins, span_bin_size, min_span, max_span,
            dem_data, cell_size, tower_heights, height_premiums,
            conductor_weight_per_m, conductor_tension, min_clearance_val,
            obstacle_heights, exclude_mask,
            max_gradient_pct, gradient_scale,
            use_area_cost, area_offsets_ptr, area_starts_ptr, area_counts_ptr,
        )

    raster = np.ascontiguousarray(raster)
    dem_data = np.ascontiguousarray(dem_data)
    tower_heights = np.ascontiguousarray(tower_heights)
    height_premiums = np.ascontiguousarray(height_premiums)
    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)
    else:
        exclude_mask = np.ascontiguousarray(exclude_mask)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs
    cdef float[:, :] dem_view = dem_data
    cdef float[:] th_view = tower_heights
    cdef float[:] hp_view = height_premiums

    cdef uint16_t* raster_ptr = &raster_view[0, 0]
    cdef uint8_t* mask_ptr = &mask_view[0, 0]
    cdef float* tower_terrain_ptr = &tower_terrain_view[0]
    cdef float* dem_ptr = &dem_view[0, 0]
    cdef float* th_ptr = &th_view[0]
    cdef float* hp_ptr = &hp_view[0]
    cdef float* obstacle_ptr = NULL
    cdef float[:, :] obs_view
    if obstacle_heights is not None:
        obstacle_heights = np.ascontiguousarray(obstacle_heights)
        obs_view = obstacle_heights
        obstacle_ptr = &obs_view[0, 0]

    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs)
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    # Compute delta
    cdef double min_raster_val_d = 1.0
    traversable = raster[raster != 65535]
    if traversable.size > 0:
        min_raster_val_d = <double>max(1, int(traversable.min()))
    cdef double min_cf = 1e9
    cdef int dd
    for dd in range(n_dirs):
        if directions[dd].cost_factor < min_cf:
            min_cf = directions[dd].cost_factor
    cdef double delta_val = max(1.0, 2.0 * min_raster_val_d * min_cf * (1.0 - 1e-9))

    # Circular bucket queue
    cdef size_t n_phys_buckets = 65536
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    # AoS state array
    cdef StateData* states = <StateData*>calloc(total_states, sizeof(StateData))

    # Precomputed intermediate + gradient caches
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))
    if states == NULL or icache_status == NULL or icache_cost == NULL:
        if states != NULL: free(states)
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate memory for variable-height delta-stepping")

    _precompute_intermediate_cache(total_cells, n_dirs, rows, cols,
        mask_ptr, raster_ptr, directions, cached_steps, icache_status, icache_cost)

    cdef float* grad_penalty_ptr = <float*>malloc(cache_size * sizeof(float))
    if grad_penalty_ptr == NULL:
        free(states); free(icache_status); free(icache_cost)
        raise MemoryError("Failed to allocate gradient cache")
    _precompute_gradient_cache(total_cells, n_dirs, rows, cols,
        dem_ptr, cell_size, max_gradient_pct, gradient_scale,
        directions, icache_status, grad_penalty_ptr)

    # Thread-local buffers, scaled by n_heights
    import os as _os
    cdef int num_threads = int(_os.environ.get('OMP_NUM_THREADS', str(_os.cpu_count() or 4)))
    if num_threads < 1:
        num_threads = 1
    cdef int buf_cap = 131072 * max(1, n_heights // 4)
    cdef CRelaxBuf* tbufs = <CRelaxBuf*>malloc(num_threads * sizeof(CRelaxBuf))
    cdef int t
    if tbufs == NULL:
        free(states); free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        raise MemoryError("Failed to allocate thread buffers")
    for t in range(num_threads):
        tbufs[t].states = <uint64_t*>malloc(buf_cap * sizeof(uint64_t))
        tbufs[t].dists = <double*>malloc(buf_cap * sizeof(double))
        tbufs[t].preds = <int64_t*>malloc(buf_cap * sizeof(int64_t))
        tbufs[t].span_dists = <float*>malloc(buf_cap * sizeof(float))
        tbufs[t].count = 0
        tbufs[t].capacity = buf_cap

    # Initialize source: all directions x all heights at span_bin=0.
    # Source anchor pays the height premium for its chosen height class.
    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx
    cdef int d, h
    cdef int sph = n_span_bins * n_heights  # span_bins * heights per direction
    cdef double init_dist
    cdef size_t init_logical
    for d in range(n_dirs):
        for h in range(n_heights):
            state_idx = (<uint64_t>source_cell) * spc + d * sph + h
            init_dist = <double>hp_ptr[h]
            states[state_idx].touched = 1
            states[state_idx].dist = init_dist
            states[state_idx].pred = -1
            states[state_idx].span_dist = 0.0
            init_logical = <size_t>(init_dist / delta_val)
            buckets[init_logical & bucket_mask].push_back(state_idx)
            if init_logical > 0:
                if init_logical > <size_t>0:
                    pass  # max_logical updated below

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY
    # max_logical must account for the tallest source premium
    cdef size_t max_init_logical = 0
    if n_heights > 0 and hp_ptr[0] > 0:
        max_init_logical = <size_t>(<double>hp_ptr[0] / delta_val)
    cdef size_t current_logical = 0, max_logical = max_init_logical, phys_idx, new_logical
    cdef vector[uint64_t] active, batch, deferred
    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef size_t entry_logical, bi

    # Parallel loop variables
    cdef int ai, n_active, tid
    cdef uint8_t p_cur_dir, p_cur_hc
    cdef int p_cur_row, p_cur_col, p_nr, p_nc, p_k, p_n_valid, p_nb_start
    cdef double p_cur_span_m, p_cur_dist, p_cur_tower_terrain
    cdef uint16_t p_cur_raster_val
    cdef uint32_t p_nb_cell, p_cur_cell
    cdef uint64_t p_remainder, p_new_state, p_base_nb
    cdef double p_edge_cost, p_terrain_cost, p_new_dist, p_tower_cost
    cdef double p_new_span_m, p_reset_span_m
    cdef uint16_t p_new_span_bin, p_reset_span_bin
    cdef float p_icost
    cdef size_t p_cidx
    cdef ValidNeighbor p_nb
    cdef int p_skip, p_sb, p_cnt, p_clearance_ok, p_hb
    cdef float p_h_a, p_h_b, p_hp_b
    cdef uint64_t p_sph = <uint64_t>sph

    try:
        while current_logical <= max_logical:
            if <double>current_logical * delta_val > best_target_dist:
                break
            phys_idx = current_logical & bucket_mask
            if buckets[phys_idx].size() == 0:
                current_logical += 1
                continue
            deferred.clear()
            while buckets[phys_idx].size() > 0:
                batch.swap(buckets[phys_idx])
                # Livelock fix: the swap put the PREVIOUS (already processed)
                # batch into the bucket -- discard it, or the two vectors
                # oscillate forever once any same-bucket push occurs.
                buckets[phys_idx].clear()
                active.clear()
                for bi in range(batch.size()):
                    cur_state = batch[bi]
                    if states[cur_state].visited != 0:
                        continue
                    cur_dist = states[cur_state].dist
                    if cur_dist >= best_target_dist:
                        continue
                    entry_logical = <size_t>(cur_dist / delta_val)
                    if entry_logical != current_logical:
                        if entry_logical > current_logical:
                            deferred.push_back(cur_state)
                        continue
                    states[cur_state].visited = 1
                    cur_cell = <uint32_t>(cur_state / spc)
                    if cur_cell == target_cell:
                        if cur_dist < best_target_dist:
                            best_target_dist = cur_dist
                            best_target_state = cur_state
                        continue
                    active.push_back(cur_state)

                n_active = <int>active.size()
                if n_active == 0:
                    break
                for t in range(num_threads):
                    tbufs[t].count = 0

                # ---- PARALLEL EDGE RELAXATION ----
                for ai in prange(n_active, nogil=True, schedule='dynamic',
                                 chunksize=64, num_threads=num_threads):
                    tid = threadid()
                    if tid < 0 or tid >= num_threads:
                        tid = 0
                    cur_state = active[ai]
                    p_cur_dist = states[cur_state].dist
                    p_cur_cell = <uint32_t>(cur_state / spc)
                    p_remainder = cur_state - (<uint64_t>p_cur_cell) * spc
                    p_cur_dir = <uint8_t>(p_remainder / p_sph)
                    p_cur_hc = <uint8_t>((p_remainder % p_sph) % n_heights)
                    p_cur_row = <int>(p_cur_cell / cols)
                    p_cur_col = <int>(p_cur_cell % cols)
                    p_cur_span_m = <double>states[cur_state].span_dist
                    p_cur_raster_val = raster_ptr[p_cur_cell]
                    p_cur_tower_terrain = <double>tower_terrain_ptr[p_cur_raster_val]
                    p_h_a = th_ptr[p_cur_hc]  # current tower height (tower A)

                    p_nb_start = nb_offsets[p_cur_dir]
                    p_n_valid = nb_offsets[p_cur_dir + 1] - p_nb_start
                    for p_k in range(p_n_valid):
                        p_nb = flat_nb[p_nb_start + p_k]
                        p_nr = p_cur_row + p_nb.dr
                        p_nc = p_cur_col + p_nb.dc
                        if (<unsigned int>p_nr >= <unsigned int>rows or
                                <unsigned int>p_nc >= <unsigned int>cols):
                            continue
                        p_nb_cell = <uint32_t>(p_nr * cols + p_nc)
                        if mask_ptr[p_nb_cell] == 0:
                            continue

                        # Skip if ALL height states of this (cell, dir) are visited
                        p_base_nb = (<uint64_t>p_nb_cell) * spc + <uint64_t>p_nb.d_out * p_sph
                        p_skip = 1
                        for p_sb in range(n_span_bins * n_heights):
                            if states[p_base_nb + p_sb].visited == 0:
                                p_skip = 0
                                break
                        if p_skip:
                            continue

                        # Intermediate cache lookup
                        p_cidx = <size_t>p_nb_cell * <size_t>n_dirs + <size_t>p_nb.d_out
                        if icache_status[p_cidx] != 2:
                            continue
                        p_icost = icache_cost[p_cidx]

                        p_terrain_cost = (<double>p_cur_raster_val + <double>p_icost +
                                         <double>raster_ptr[p_nb_cell]) * <double>p_nb.cost_factor
                        p_terrain_cost = p_terrain_cost * <double>grad_penalty_ptr[p_cidx]
                        p_edge_cost = p_terrain_cost + <double>p_nb.angle_cost
                        p_new_span_m = p_cur_span_m + <double>p_nb.step_distance

                        if p_nb.d_out == p_cur_dir:
                            # Same direction: continue span (height carries forward)
                            if p_new_span_m < <double>max_span:
                                p_new_span_bin = <uint16_t>(p_new_span_m / span_bin_size)
                                p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                              <uint64_t>p_nb.d_out * p_sph +
                                              <uint64_t>p_new_span_bin * n_heights +
                                              <uint64_t>p_cur_hc)
                                p_new_dist = p_cur_dist + p_edge_cost
                                if (states[p_new_state].visited == 0 or
                                        p_new_dist < <double>states[p_new_state].dist):
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_new_span_m
                                        tbufs[tid].count = p_cnt + 1

                            # Same direction: optional tower with height exploration
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                # Loop over candidate heights for new tower B (descending)
                                for p_hb in range(n_heights):
                                    p_h_b = th_ptr[p_hb]
                                    p_clearance_ok = _check_span_clearance_vh(
                                        p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                        p_h_a, p_h_b,
                                        conductor_weight_per_m, conductor_tension,
                                        min_clearance_val, dem_ptr, obstacle_ptr,
                                        rows, cols, directions, cached_steps,
                                        step_dist_view[p_cur_dir])
                                    if not p_clearance_ok:
                                        break  # heights sorted desc: shorter will also fail
                                    p_hp_b = hp_ptr[p_hb]
                                    p_tower_cost = (_tower_terrain(
                                        use_area_cost, p_cur_row, p_cur_col,
                                        p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                        raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                        area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                        dem_ptr, cell_size, gradient_scale,
                                    ) + <double>p_nb.tower_angle_cost +
                                                   <double>p_hp_b)
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                                  <uint64_t>p_nb.d_out * p_sph +
                                                  <uint64_t>p_reset_span_bin * n_heights +
                                                  <uint64_t>p_hb)
                                    if states[p_new_state].visited == 0:
                                        p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                        p_cnt = tbufs[tid].count
                                        if p_cnt < tbufs[tid].capacity:
                                            tbufs[tid].states[p_cnt] = p_new_state
                                            tbufs[tid].dists[p_cnt] = p_new_dist
                                            tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                            tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                            tbufs[tid].count = p_cnt + 1
                        else:
                            # Direction change: mandatory tower (min_span enforced)
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                for p_hb in range(n_heights):
                                    p_h_b = th_ptr[p_hb]
                                    p_clearance_ok = _check_span_clearance_vh(
                                        p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                        p_h_a, p_h_b,
                                        conductor_weight_per_m, conductor_tension,
                                        min_clearance_val, dem_ptr, obstacle_ptr,
                                        rows, cols, directions, cached_steps,
                                        step_dist_view[p_cur_dir])
                                    if not p_clearance_ok:
                                        break  # height-sorted early exit
                                    p_hp_b = hp_ptr[p_hb]
                                    p_tower_cost = (_tower_terrain(
                                        use_area_cost, p_cur_row, p_cur_col,
                                        p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                        raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                        area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                        dem_ptr, cell_size, gradient_scale,
                                    ) + <double>p_nb.tower_angle_cost +
                                                   <double>p_hp_b)
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                                  <uint64_t>p_nb.d_out * p_sph +
                                                  <uint64_t>p_reset_span_bin * n_heights +
                                                  <uint64_t>p_hb)
                                    if states[p_new_state].visited == 0:
                                        p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                        p_cnt = tbufs[tid].count
                                        if p_cnt < tbufs[tid].capacity:
                                            tbufs[tid].states[p_cnt] = p_new_state
                                            tbufs[tid].dists[p_cnt] = p_new_dist
                                            tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                            tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                            tbufs[tid].count = p_cnt + 1

                # ---- SEQUENTIAL MERGE ----
                for t in range(num_threads):
                    for bi in range(tbufs[t].count):
                        p_new_state = tbufs[t].states[bi]
                        p_new_dist = tbufs[t].dists[bi]
                        if (states[p_new_state].visited == 0 or
                                p_new_dist < states[p_new_state].dist):
                            if states[p_new_state].touched == 0 or p_new_dist < states[p_new_state].dist:
                                states[p_new_state].touched = 1
                                # Re-open protocol: a settled state improved
                                # (possible when delta > min edge cost, e.g.
                                # 0-cost cells) returns to the frontier.
                                states[p_new_state].visited = 0
                                states[p_new_state].dist = p_new_dist
                                states[p_new_state].pred = tbufs[t].preds[bi]
                                states[p_new_state].span_dist = tbufs[t].span_dists[bi]
                                new_logical = <size_t>(p_new_dist / delta_val)
                                buckets[new_logical & bucket_mask].push_back(p_new_state)
                                if new_logical > max_logical:
                                    max_logical = new_logical

            for bi in range(deferred.size()):
                cur_state = deferred[bi]
                if states[cur_state].visited == 0:
                    new_logical = <size_t>(states[cur_state].dist / delta_val)
                    if new_logical > current_logical:
                        buckets[new_logical & bucket_mask].push_back(cur_state)
            current_logical += 1

    finally:
        if tbufs != NULL:
            for t in range(num_threads):
                if tbufs[t].states != NULL: free(tbufs[t].states)
                if tbufs[t].dists != NULL: free(tbufs[t].dists)
                if tbufs[t].preds != NULL: free(tbufs[t].preds)
                if tbufs[t].span_dists != NULL: free(tbufs[t].span_dists)
            free(tbufs)

    # Path reconstruction -- extract (cell, height_class) per tower
    cdef list path_cells = []
    cdef list tower_cells = []
    cdef list tower_height_classes = []
    if best_target_state == UINT64_MAX:
        free(states); free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir, walk_hc
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir, prev_hc
    cdef uint16_t prev_span
    cdef list state_chain = []
    while walk_state != UINT64_MAX and states[walk_state].pred != -1:
        state_chain.append(walk_state)
        walk_state = <uint64_t>states[walk_state].pred
    state_chain.append(walk_state)
    state_chain.reverse()

    cdef uint64_t st, prev_state_val
    cdef int k
    for k in range(len(state_chain)):
        st = <uint64_t>state_chain[k]
        _unpack_state_h(st, n_dirs, n_span_bins, n_heights,
                        &walk_cell, &walk_dir, &walk_span, &walk_hc)
        path_cells.append(<int>walk_cell)

        if k > 0 and n_span_bins > 1:
            prev_state_val = <uint64_t>state_chain[k - 1]
            _unpack_state_h(prev_state_val, n_dirs, n_span_bins, n_heights,
                            &prev_cell, &prev_dir, &prev_span, &prev_hc)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)
                    tower_height_classes.append(<int>prev_hc)

    # Convert height classes to actual heights
    cdef np.ndarray[np.float32_t, ndim=1] tower_h_out = np.empty(
        len(tower_height_classes), dtype=np.float32)
    for k in range(len(tower_height_classes)):
        tower_h_out[k] = tower_heights[tower_height_classes[k]]

    free(states); free(icache_status); free(icache_cost); free(grad_penalty_ptr)
    return (np.array(path_cells, dtype=np.uint32),
            np.array(tower_cells, dtype=np.uint32),
            tower_h_out)


# ===========================================================================
# 4. COMPACT-DENSE VARIABLE-HEIGHT DELTA-STEPPING (SPARSE FALLBACK)
# ===========================================================================

cdef _height_sparse(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col, int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins, float span_bin_size, float min_span, float max_span,
    np.ndarray[np.float32_t, ndim=2] dem_data,
    float cell_size,
    np.ndarray[np.float32_t, ndim=1] tower_heights,
    np.ndarray[np.float32_t, ndim=1] height_premiums,
    float conductor_weight_per_m, float conductor_tension, float min_clearance_val,
    np.ndarray[np.float32_t, ndim=2] obstacle_heights,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask,
    float max_gradient_pct, float gradient_scale,
    int use_area_cost_flag=0,
    const int32_t* area_offsets_arg=NULL,
    const int32_t* area_starts_arg=NULL,
    const int32_t* area_counts_arg=NULL,
):
    """Compact-dense delta-stepping with variable heights and clearance.

    Uses dense float32 dist + uint8 flags arrays (5 bytes/state) for O(1)
    distance lookups and visited checks. Hash maps only for predecessor and
    span_dist (written at tower placement, read only during reconstruction).
    Bucket queue + OpenMP parallel relaxation for speed.
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int n_heights = tower_heights.shape[0]
    cdef int spc = n_dirs * n_span_bins * n_heights
    cdef int sph = n_span_bins * n_heights

    raster = np.ascontiguousarray(raster)
    dem_data = np.ascontiguousarray(dem_data)
    tower_heights = np.ascontiguousarray(tower_heights)
    height_premiums = np.ascontiguousarray(height_premiums)
    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)
    else:
        exclude_mask = np.ascontiguousarray(exclude_mask)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs
    cdef float[:, :] dem_view = dem_data
    cdef float[:] th_view = tower_heights
    cdef float[:] hp_view = height_premiums

    cdef uint16_t* raster_ptr = &raster_view[0, 0]
    cdef uint8_t* mask_ptr = &mask_view[0, 0]
    cdef float* tower_terrain_ptr = &tower_terrain_view[0]
    cdef float* dem_ptr = &dem_view[0, 0]
    cdef float* th_ptr = &th_view[0]
    cdef float* hp_ptr = &hp_view[0]
    cdef float* obstacle_ptr = NULL
    cdef float[:, :] obs_view
    if obstacle_heights is not None:
        obstacle_heights = np.ascontiguousarray(obstacle_heights)
        obs_view = obstacle_heights
        obstacle_ptr = &obs_view[0, 0]

    # Area cost offsets (exact mode) — uses C pointers from caller
    cdef int use_area_cost = use_area_cost_flag
    cdef const int32_t* area_offsets_ptr = area_offsets_arg
    cdef const int32_t* area_starts_ptr = area_starts_arg
    cdef const int32_t* area_counts_ptr = area_counts_arg

    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs)
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    # Per-cell intermediate + gradient caches (independent of height dimension)
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))
    if icache_status == NULL or icache_cost == NULL:
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate intermediate cache")

    _precompute_intermediate_cache(total_cells, n_dirs, rows, cols,
        mask_ptr, raster_ptr, directions, cached_steps, icache_status, icache_cost)

    cdef float* grad_penalty_ptr = <float*>malloc(cache_size * sizeof(float))
    if grad_penalty_ptr == NULL:
        free(icache_status); free(icache_cost)
        raise MemoryError("Failed to allocate gradient cache")
    _precompute_gradient_cache(total_cells, n_dirs, rows, cols,
        dem_ptr, cell_size, max_gradient_pct, gradient_scale,
        directions, icache_status, grad_penalty_ptr)

    # Fully dense arrays: NO hash maps in hot path -> pure O(1) merge.
    # dist(float32) + pred(int64) + span(float32) + flags(uint8) = 17 bytes/state.
    cdef uint64_t total_states = <uint64_t>total_cells * spc
    cdef float* cdist = <float*>malloc(total_states * sizeof(float))
    cdef int64_t* cpred = <int64_t*>malloc(total_states * sizeof(int64_t))
    cdef float* cspan = <float*>calloc(total_states, sizeof(float))
    cdef uint8_t* cflags = <uint8_t*>calloc(total_states, sizeof(uint8_t))

    if cdist == NULL or cpred == NULL or cspan == NULL or cflags == NULL:
        if cdist != NULL: free(cdist)
        if cpred != NULL: free(cpred)
        if cspan != NULL: free(cspan)
        if cflags != NULL: free(cflags)
        free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        raise MemoryError(
            f"Failed to allocate dense arrays ({total_states * 17 / 1e9:.1f} GB). "
            f"Reduce tower_heights or n_span_bins."
        )

    # Compute delta for bucket queue
    cdef double min_raster_val_d = 1.0
    traversable = raster[raster != 65535]
    if traversable.size > 0:
        min_raster_val_d = <double>max(1, int(traversable.min()))
    cdef double min_cf = 1e9
    cdef int dd
    for dd in range(n_dirs):
        if directions[dd].cost_factor < min_cf:
            min_cf = directions[dd].cost_factor
    cdef double delta_val = max(1.0, 2.0 * min_raster_val_d * min_cf * (1.0 - 1e-9))

    # Circular bucket queue
    cdef size_t n_phys_buckets = 65536
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    # Thread-local buffers for parallel relaxation
    import os as _os
    cdef int num_threads = int(_os.environ.get('OMP_NUM_THREADS', str(_os.cpu_count() or 4)))
    if num_threads < 1:
        num_threads = 1
    cdef int buf_cap = 131072 * max(1, n_heights // 4)
    cdef CRelaxBuf* tbufs = <CRelaxBuf*>malloc(num_threads * sizeof(CRelaxBuf))
    cdef int t
    if tbufs == NULL:
        free(cdist); free(cflags)
        free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        raise MemoryError("Failed to allocate thread buffers")
    for t in range(num_threads):
        tbufs[t].states = <uint64_t*>malloc(buf_cap * sizeof(uint64_t))
        tbufs[t].dists = <double*>malloc(buf_cap * sizeof(double))
        tbufs[t].preds = <int64_t*>malloc(buf_cap * sizeof(int64_t))
        tbufs[t].span_dists = <float*>malloc(buf_cap * sizeof(float))
        tbufs[t].count = 0
        tbufs[t].capacity = buf_cap

    # Initialize source: all directions x all heights at span_bin=0
    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx
    cdef int d, h
    cdef float init_dist_f
    cdef size_t init_logical

    for d in range(n_dirs):
        for h in range(n_heights):
            state_idx = (<uint64_t>source_cell) * spc + d * sph + h
            init_dist_f = hp_ptr[h]
            cdist[state_idx] = init_dist_f
            cpred[state_idx] = -1
            cspan[state_idx] = 0.0
            cflags[state_idx] = FLAG_TOUCHED
            init_logical = <size_t>(<double>init_dist_f / delta_val)
            buckets[init_logical & bucket_mask].push_back(state_idx)

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY
    cdef size_t max_init_logical = 0
    if n_heights > 0 and hp_ptr[0] > 0:
        max_init_logical = <size_t>(<double>hp_ptr[0] / delta_val)
    cdef size_t current_logical = 0, max_logical = max_init_logical
    cdef size_t phys_idx, new_logical, entry_logical
    cdef vector[uint64_t] batch, deferred

    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef uint64_t remainder
    cdef uint64_t u_sph = <uint64_t>sph
    cdef size_t bi

    cdef vector[uint64_t] active_states
    cdef vector[double] active_dists
    cdef vector[float] active_spans
    cdef int n_active, ai, tid

    # Parallel loop variables
    cdef uint8_t p_cur_dir, p_cur_hc
    cdef int p_cur_row, p_cur_col, p_nr, p_nc, p_k, p_n_valid, p_nb_start
    cdef double p_cur_span_m, p_cur_dist, p_cur_tower_terrain
    cdef uint32_t p_nb_cell, p_cur_cell
    cdef uint64_t p_new_state, p_remainder
    cdef double p_edge_cost, p_terrain_cost, p_new_dist, p_tower_cost
    cdef double p_new_span_m, p_reset_span_m
    cdef uint16_t p_new_span_bin, p_reset_span_bin, p_cur_raster_val
    cdef float p_icost, p_h_a, p_h_b, p_hp_b
    cdef size_t p_cidx
    cdef ValidNeighbor p_nb
    cdef int p_cnt, p_clearance_ok, p_hb

    try:
        while current_logical <= max_logical:
            if <double>current_logical * delta_val > best_target_dist:
                break
            phys_idx = current_logical & bucket_mask
            if buckets[phys_idx].size() == 0:
                current_logical += 1
                continue
            deferred.clear()

            while buckets[phys_idx].size() > 0:
                batch.swap(buckets[phys_idx])
                # Livelock fix: the swap put the PREVIOUS (already processed)
                # batch into the bucket -- discard it, or the two vectors
                # oscillate forever once any same-bucket push occurs.
                buckets[phys_idx].clear()

                # Sequential filter with O(1) dense array lookups
                active_states.clear()
                active_dists.clear()
                active_spans.clear()
                for bi in range(batch.size()):
                    cur_state = batch[bi]
                    if cflags[cur_state] & FLAG_VISITED:
                        continue
                    if not (cflags[cur_state] & FLAG_TOUCHED):
                        continue
                    cur_dist = <double>cdist[cur_state]
                    if cur_dist >= best_target_dist:
                        continue
                    entry_logical = <size_t>(cur_dist / delta_val)
                    if entry_logical != current_logical:
                        if entry_logical > current_logical:
                            deferred.push_back(cur_state)
                        continue

                    cflags[cur_state] = cflags[cur_state] | FLAG_VISITED
                    cur_cell = <uint32_t>(cur_state / spc)

                    if cur_cell == target_cell:
                        if cur_dist < best_target_dist:
                            best_target_dist = cur_dist
                            best_target_state = cur_state
                        continue

                    active_states.push_back(cur_state)
                    active_dists.push_back(cur_dist)
                    active_spans.push_back(cspan[cur_state])

                n_active = <int>active_states.size()
                if n_active == 0:
                    break
                for t in range(num_threads):
                    tbufs[t].count = 0

                # ---- PARALLEL EDGE RELAXATION ----
                # Reads only from: dense arrays (cdist, cflags), flat_nb,
                # icache, grad_penalty, raster -- all thread-safe.
                for ai in prange(n_active, nogil=True, schedule='dynamic',
                                 chunksize=64, num_threads=num_threads):
                    tid = threadid()
                    if tid < 0 or tid >= num_threads:
                        tid = 0

                    cur_state = active_states[ai]
                    p_cur_dist = active_dists[ai]
                    p_cur_cell = <uint32_t>(cur_state / spc)
                    p_remainder = cur_state - (<uint64_t>p_cur_cell) * spc
                    p_cur_dir = <uint8_t>(p_remainder / u_sph)
                    p_cur_hc = <uint8_t>((p_remainder % u_sph) % n_heights)
                    p_cur_row = <int>(p_cur_cell / cols)
                    p_cur_col = <int>(p_cur_cell % cols)
                    p_cur_span_m = <double>active_spans[ai]
                    p_cur_raster_val = raster_ptr[p_cur_cell]
                    p_cur_tower_terrain = <double>tower_terrain_ptr[p_cur_raster_val]
                    p_h_a = th_ptr[p_cur_hc]

                    p_nb_start = nb_offsets[p_cur_dir]
                    p_n_valid = nb_offsets[p_cur_dir + 1] - p_nb_start
                    for p_k in range(p_n_valid):
                        p_nb = flat_nb[p_nb_start + p_k]
                        p_nr = p_cur_row + p_nb.dr
                        p_nc = p_cur_col + p_nb.dc
                        if (<unsigned int>p_nr >= <unsigned int>rows or
                                <unsigned int>p_nc >= <unsigned int>cols):
                            continue
                        p_nb_cell = <uint32_t>(p_nr * cols + p_nc)
                        if mask_ptr[p_nb_cell] == 0:
                            continue

                        p_cidx = <size_t>p_nb_cell * <size_t>n_dirs + <size_t>p_nb.d_out
                        if icache_status[p_cidx] != 2:
                            continue
                        p_icost = icache_cost[p_cidx]

                        p_terrain_cost = (<double>p_cur_raster_val +
                                         <double>p_icost +
                                         <double>raster_ptr[p_nb_cell]) * <double>p_nb.cost_factor
                        p_terrain_cost = p_terrain_cost * <double>grad_penalty_ptr[p_cidx]
                        p_edge_cost = p_terrain_cost + <double>p_nb.angle_cost
                        p_new_span_m = p_cur_span_m + <double>p_nb.step_distance

                        if p_nb.d_out == p_cur_dir:
                            if p_new_span_m < <double>max_span:
                                p_new_span_bin = <uint16_t>(p_new_span_m / span_bin_size)
                                p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                              <uint64_t>p_nb.d_out * u_sph +
                                              <uint64_t>p_new_span_bin * n_heights +
                                              <uint64_t>p_cur_hc)
                                if not (cflags[p_new_state] & FLAG_VISITED):
                                    p_new_dist = p_cur_dist + p_edge_cost
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_new_span_m
                                        tbufs[tid].count = p_cnt + 1

                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                for p_hb in range(n_heights):
                                    p_h_b = th_ptr[p_hb]
                                    p_clearance_ok = _check_span_clearance_vh(
                                        p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                        p_h_a, p_h_b,
                                        conductor_weight_per_m, conductor_tension,
                                        min_clearance_val, dem_ptr, obstacle_ptr,
                                        rows, cols, directions, cached_steps,
                                        step_dist_view[p_cur_dir])
                                    if not p_clearance_ok:
                                        break
                                    p_hp_b = hp_ptr[p_hb]
                                    p_tower_cost = (_tower_terrain(
                                        use_area_cost, p_cur_row, p_cur_col,
                                        p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                        raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                        area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                        dem_ptr, cell_size, gradient_scale,
                                    ) + <double>p_nb.tower_angle_cost +
                                                   <double>p_hp_b)
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                                  <uint64_t>p_nb.d_out * u_sph +
                                                  <uint64_t>p_reset_span_bin * n_heights +
                                                  <uint64_t>p_hb)
                                    if not (cflags[p_new_state] & FLAG_VISITED):
                                        p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                        p_cnt = tbufs[tid].count
                                        if p_cnt < tbufs[tid].capacity:
                                            tbufs[tid].states[p_cnt] = p_new_state
                                            tbufs[tid].dists[p_cnt] = p_new_dist
                                            tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                            tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                            tbufs[tid].count = p_cnt + 1
                        else:
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                for p_hb in range(n_heights):
                                    p_h_b = th_ptr[p_hb]
                                    p_clearance_ok = _check_span_clearance_vh(
                                        p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                        p_h_a, p_h_b,
                                        conductor_weight_per_m, conductor_tension,
                                        min_clearance_val, dem_ptr, obstacle_ptr,
                                        rows, cols, directions, cached_steps,
                                        step_dist_view[p_cur_dir])
                                    if not p_clearance_ok:
                                        break
                                    p_hp_b = hp_ptr[p_hb]
                                    p_tower_cost = (_tower_terrain(
                                        use_area_cost, p_cur_row, p_cur_col,
                                        p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                        raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                        area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                        dem_ptr, cell_size, gradient_scale,
                                    ) + <double>p_nb.tower_angle_cost +
                                                   <double>p_hp_b)
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                                  <uint64_t>p_nb.d_out * u_sph +
                                                  <uint64_t>p_reset_span_bin * n_heights +
                                                  <uint64_t>p_hb)
                                    if not (cflags[p_new_state] & FLAG_VISITED):
                                        p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                        p_cnt = tbufs[tid].count
                                        if p_cnt < tbufs[tid].capacity:
                                            tbufs[tid].states[p_cnt] = p_new_state
                                            tbufs[tid].dists[p_cnt] = p_new_dist
                                            tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                            tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                            tbufs[tid].count = p_cnt + 1

                # ---- SEQUENTIAL MERGE: O(1) dense array writes ----
                for t in range(num_threads):
                    for bi in range(tbufs[t].count):
                        p_new_state = tbufs[t].states[bi]
                        if cflags[p_new_state] & FLAG_VISITED:
                            continue
                        p_new_dist = tbufs[t].dists[bi]
                        if (not (cflags[p_new_state] & FLAG_TOUCHED) or
                                <float>p_new_dist < cdist[p_new_state]):
                            cflags[p_new_state] = cflags[p_new_state] | FLAG_TOUCHED
                            cdist[p_new_state] = <float>p_new_dist
                            cpred[p_new_state] = tbufs[t].preds[bi]
                            cspan[p_new_state] = tbufs[t].span_dists[bi]
                            new_logical = <size_t>(p_new_dist / delta_val)
                            buckets[new_logical & bucket_mask].push_back(p_new_state)
                            if new_logical > max_logical:
                                max_logical = new_logical

            for bi in range(deferred.size()):
                cur_state = deferred[bi]
                if not (cflags[cur_state] & FLAG_VISITED):
                    new_logical = <size_t>(<double>cdist[cur_state] / delta_val)
                    if new_logical > current_logical:
                        buckets[new_logical & bucket_mask].push_back(cur_state)
            current_logical += 1

    finally:
        if tbufs != NULL:
            for t in range(num_threads):
                if tbufs[t].states != NULL: free(tbufs[t].states)
                if tbufs[t].dists != NULL: free(tbufs[t].dists)
                if tbufs[t].preds != NULL: free(tbufs[t].preds)
                if tbufs[t].span_dists != NULL: free(tbufs[t].span_dists)
            free(tbufs)

    # Path reconstruction
    cdef list path_cells = []
    cdef list tower_cells = []
    cdef list tower_height_classes = []

    if best_target_state == UINT64_MAX:
        free(cdist); free(cpred); free(cspan); free(cflags)
        free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir, walk_hc
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir, prev_hc
    cdef uint16_t prev_span

    cdef list state_chain = []
    while True:
        state_chain.append(walk_state)
        if cpred[walk_state] == -1:
            break
        walk_state = <uint64_t>cpred[walk_state]
    state_chain.reverse()

    cdef uint64_t st, prev_state_val
    cdef int k
    for k in range(len(state_chain)):
        st = <uint64_t>state_chain[k]
        _unpack_state_h(st, n_dirs, n_span_bins, n_heights,
                        &walk_cell, &walk_dir, &walk_span, &walk_hc)
        path_cells.append(<int>walk_cell)

        if k > 0 and n_span_bins > 1:
            prev_state_val = <uint64_t>state_chain[k - 1]
            _unpack_state_h(prev_state_val, n_dirs, n_span_bins, n_heights,
                            &prev_cell, &prev_dir, &prev_span, &prev_hc)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)
                    tower_height_classes.append(<int>prev_hc)

    # Convert height classes to actual heights
    cdef np.ndarray[np.float32_t, ndim=1] tower_h_out = np.empty(
        len(tower_height_classes), dtype=np.float32)
    for k in range(len(tower_height_classes)):
        tower_h_out[k] = tower_heights[tower_height_classes[k]]

    free(cdist); free(cpred); free(cspan); free(cflags)
    free(icache_status); free(icache_cost); free(grad_penalty_ptr)
    return (np.array(path_cells, dtype=np.uint32),
            np.array(tower_cells, dtype=np.uint32),
            tower_h_out)


# ===========================================================================
# 5. LAZY PARALLEL DELTA-STEPPING WITH CLEARANCE
# ===========================================================================

def constrained_delta_stepping_lazy(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col, int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins, float span_bin_size, float min_span, float max_span,
    np.ndarray[np.float32_t, ndim=2] dem_data,
    float cell_size,
    np.ndarray[np.float32_t, ndim=1] tower_heights,
    np.ndarray[np.float32_t, ndim=1] height_premiums,
    float conductor_weight_per_m, float conductor_tension, float min_clearance_val,
    np.ndarray[np.float32_t, ndim=2] obstacle_heights=None,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask=None,
    float max_gradient_pct=100.0, float gradient_scale=2.0,
    np.ndarray[np.int32_t, ndim=1] area_offsets=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_starts=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_counts=None,
):
    """Parallel delta-stepping with lazy hash map state allocation and clearance.

    Memory-efficient alternative to constrained_delta_stepping_height_2d for
    large rasters where the dense state array would exceed available memory.
    Same algorithm, same results, but only allocates states as they are touched.

    Uses unordered_map for state storage (O(visited) memory instead of O(total)).
    Parallel relaxation via prange + thread-local buffers; threads read only from
    contiguous snapshot vectors (no hash map access in parallel section).

    Supports both fixed-height (n_heights=1) and variable-height (n_heights>1).

    Returns:
        3-tuple: (path_cells uint32[], tower_cells uint32[], tower_heights float32[])
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int n_heights = tower_heights.shape[0]
    cdef int spc = n_dirs * n_span_bins * n_heights
    cdef int sph = n_span_bins * n_heights

    raster = np.ascontiguousarray(raster)
    dem_data = np.ascontiguousarray(dem_data)
    tower_heights = np.ascontiguousarray(tower_heights)
    height_premiums = np.ascontiguousarray(height_premiums)
    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)
    else:
        exclude_mask = np.ascontiguousarray(exclude_mask)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs
    cdef float[:, :] dem_view = dem_data
    cdef float[:] th_view = tower_heights
    cdef float[:] hp_view = height_premiums

    cdef uint16_t* raster_ptr = &raster_view[0, 0]
    cdef uint8_t* mask_ptr = &mask_view[0, 0]
    cdef float* tower_terrain_ptr = &tower_terrain_view[0]
    cdef float* dem_ptr = &dem_view[0, 0]
    cdef float* th_ptr = &th_view[0]
    cdef float* hp_ptr = &hp_view[0]
    cdef float* obstacle_ptr = NULL
    cdef float[:, :] obs_view
    if obstacle_heights is not None:
        obstacle_heights = np.ascontiguousarray(obstacle_heights)
        obs_view = obstacle_heights
        obstacle_ptr = &obs_view[0, 0]

    # Area cost offsets (exact mode)
    cdef int use_area_cost = 0
    cdef int32_t[:] ao_view, as_view, ac_view
    cdef int32_t* area_offsets_ptr = NULL
    cdef int32_t* area_starts_ptr = NULL
    cdef int32_t* area_counts_ptr = NULL
    if area_offsets is not None:
        use_area_cost = 1
        ao_view = area_offsets
        as_view = area_offset_starts
        ac_view = area_offset_counts
        area_offsets_ptr = &ao_view[0]
        area_starts_ptr = &as_view[0]
        area_counts_ptr = &ac_view[0]

    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs)
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    # Dense per-(cell, dir) caches -- affordable at O(cells * dirs)
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))
    if icache_status == NULL or icache_cost == NULL:
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate intermediate cache")

    _precompute_intermediate_cache(total_cells, n_dirs, rows, cols,
        mask_ptr, raster_ptr, directions, cached_steps, icache_status, icache_cost)

    cdef float* grad_penalty_ptr = <float*>malloc(cache_size * sizeof(float))
    if grad_penalty_ptr == NULL:
        free(icache_status); free(icache_cost)
        raise MemoryError("Failed to allocate gradient cache")
    _precompute_gradient_cache(total_cells, n_dirs, rows, cols,
        dem_ptr, cell_size, max_gradient_pct, gradient_scale,
        directions, icache_status, grad_penalty_ptr)

    # Compute delta for bucket queue
    cdef double min_raster_val_d = 1.0
    traversable = raster[raster != 65535]
    if traversable.size > 0:
        min_raster_val_d = <double>max(1, int(traversable.min()))
    cdef double min_cf = 1e9
    cdef int dd
    for dd in range(n_dirs):
        if directions[dd].cost_factor < min_cf:
            min_cf = directions[dd].cost_factor
    cdef double delta_val = max(1.0, 2.0 * min_raster_val_d * min_cf * (1.0 - 1e-9))

    # Circular bucket queue
    cdef size_t n_phys_buckets = 65536
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    # LAZY STATE MAP
    cdef unordered_map[uint64_t, LazyState] states
    states.reserve(2000000)

    # Thread-local relaxation buffers
    import os as _os
    cdef int num_threads = int(_os.environ.get('OMP_NUM_THREADS', str(_os.cpu_count() or 4)))
    if num_threads < 1:
        num_threads = 1
    cdef int buf_cap = 131072 * max(1, n_heights // 4)
    cdef CRelaxBuf* tbufs = <CRelaxBuf*>malloc(num_threads * sizeof(CRelaxBuf))
    cdef int t
    if tbufs == NULL:
        free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        raise MemoryError("Failed to allocate thread buffers")
    for t in range(num_threads):
        tbufs[t].states = <uint64_t*>malloc(buf_cap * sizeof(uint64_t))
        tbufs[t].dists = <double*>malloc(buf_cap * sizeof(double))
        tbufs[t].preds = <int64_t*>malloc(buf_cap * sizeof(int64_t))
        tbufs[t].span_dists = <float*>malloc(buf_cap * sizeof(float))
        tbufs[t].count = 0
        tbufs[t].capacity = buf_cap

    # Initialize source: all directions x all heights at span_bin=0
    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx
    cdef int d, h
    cdef uint64_t p_sph = <uint64_t>sph
    cdef LazyState init_ls
    cdef size_t init_logical

    for d in range(n_dirs):
        for h in range(n_heights):
            state_idx = (<uint64_t>source_cell) * spc + d * sph + h
            init_ls.dist = <double>hp_ptr[h]
            init_ls.pred = -1
            init_ls.span_dist = 0.0
            init_ls.visited = 0
            states[state_idx] = init_ls
            init_logical = <size_t>(init_ls.dist / delta_val)
            buckets[init_logical & bucket_mask].push_back(state_idx)

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY
    cdef size_t max_init_logical = 0
    if n_heights > 0 and hp_ptr[0] > 0:
        max_init_logical = <size_t>(<double>hp_ptr[0] / delta_val)
    cdef size_t current_logical = 0, max_logical = max_init_logical
    cdef size_t phys_idx, new_logical, entry_logical
    cdef vector[uint64_t] active, batch, deferred
    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef size_t bi

    # Parallel loop variables
    cdef int ai, n_active, tid
    cdef uint8_t p_cur_dir, p_cur_hc
    cdef int p_cur_row, p_cur_col, p_nr, p_nc, p_k, p_n_valid, p_nb_start
    cdef double p_cur_span_m, p_cur_dist, p_cur_tower_terrain
    cdef uint16_t p_cur_raster_val
    cdef uint32_t p_nb_cell, p_cur_cell
    cdef uint64_t p_remainder, p_new_state
    cdef double p_edge_cost, p_terrain_cost, p_new_dist, p_tower_cost
    cdef double p_new_span_m, p_reset_span_m
    cdef uint16_t p_new_span_bin, p_reset_span_bin
    cdef float p_icost
    cdef size_t p_cidx
    cdef ValidNeighbor p_nb
    cdef int p_cnt, p_clearance_ok, p_hb
    cdef float p_h_a, p_h_b, p_hp_b

    # Snapshot vectors for parallel phase (threads read these, not the map)
    cdef vector[double] active_dists
    cdef vector[float] active_spans
    cdef vector[uint8_t] active_hcs

    # Iterator + temp variables for sequential merge
    cdef unordered_map[uint64_t, LazyState].iterator it, it_nb, it_def, it_walk
    cdef LazyState new_ls

    try:
        while current_logical <= max_logical:
            if <double>current_logical * delta_val > best_target_dist:
                break
            phys_idx = current_logical & bucket_mask
            if buckets[phys_idx].size() == 0:
                current_logical += 1
                continue
            deferred.clear()
            while buckets[phys_idx].size() > 0:
                batch.swap(buckets[phys_idx])
                # Livelock fix: the swap put the PREVIOUS (already processed)
                # batch into the bucket -- discard it, or the two vectors
                # oscillate forever once any same-bucket push occurs.
                buckets[phys_idx].clear()
                active.clear()
                active_dists.clear()
                active_spans.clear()
                active_hcs.clear()
                for bi in range(batch.size()):
                    cur_state = batch[bi]
                    it = states.find(cur_state)
                    if it == states.end():
                        continue
                    if deref(it).second.visited != 0:
                        continue
                    cur_dist = deref(it).second.dist
                    if cur_dist >= best_target_dist:
                        continue
                    entry_logical = <size_t>(cur_dist / delta_val)
                    if entry_logical != current_logical:
                        if entry_logical > current_logical:
                            deferred.push_back(cur_state)
                        continue
                    deref(it).second.visited = 1
                    cur_cell = <uint32_t>(cur_state / spc)
                    if cur_cell == target_cell:
                        if cur_dist < best_target_dist:
                            best_target_dist = cur_dist
                            best_target_state = cur_state
                        continue
                    active.push_back(cur_state)
                    active_dists.push_back(cur_dist)
                    active_spans.push_back(deref(it).second.span_dist)
                    p_remainder = cur_state - (<uint64_t>cur_cell) * spc
                    active_hcs.push_back(<uint8_t>((p_remainder % p_sph) % n_heights))

                n_active = <int>active.size()
                if n_active == 0:
                    break
                for t in range(num_threads):
                    tbufs[t].count = 0

                # ---- PARALLEL EDGE RELAXATION ----
                # Threads read ONLY from: raster_ptr, mask_ptr, icache, grad_penalty,
                # flat_nb, active/active_dists/active_spans/active_hcs vectors.
                # NO hash map access in this section.
                for ai in prange(n_active, nogil=True, schedule='dynamic',
                                 chunksize=64, num_threads=num_threads):
                    tid = threadid()
                    if tid < 0 or tid >= num_threads:
                        tid = 0
                    cur_state = active[ai]
                    p_cur_dist = active_dists[ai]
                    p_cur_cell = <uint32_t>(cur_state / spc)
                    p_remainder = cur_state - (<uint64_t>p_cur_cell) * spc
                    p_cur_dir = <uint8_t>(p_remainder / p_sph)
                    p_cur_hc = active_hcs[ai]
                    p_cur_row = <int>(p_cur_cell / cols)
                    p_cur_col = <int>(p_cur_cell % cols)
                    p_cur_span_m = <double>active_spans[ai]
                    p_cur_raster_val = raster_ptr[p_cur_cell]
                    p_cur_tower_terrain = <double>tower_terrain_ptr[p_cur_raster_val]
                    p_h_a = th_ptr[p_cur_hc]

                    p_nb_start = nb_offsets[p_cur_dir]
                    p_n_valid = nb_offsets[p_cur_dir + 1] - p_nb_start
                    for p_k in range(p_n_valid):
                        p_nb = flat_nb[p_nb_start + p_k]
                        p_nr = p_cur_row + p_nb.dr
                        p_nc = p_cur_col + p_nb.dc
                        if (<unsigned int>p_nr >= <unsigned int>rows or
                                <unsigned int>p_nc >= <unsigned int>cols):
                            continue
                        p_nb_cell = <uint32_t>(p_nr * cols + p_nc)
                        if mask_ptr[p_nb_cell] == 0:
                            continue

                        p_cidx = <size_t>p_nb_cell * <size_t>n_dirs + <size_t>p_nb.d_out
                        if icache_status[p_cidx] != 2:
                            continue
                        p_icost = icache_cost[p_cidx]

                        p_terrain_cost = (<double>p_cur_raster_val + <double>p_icost +
                                         <double>raster_ptr[p_nb_cell]) * <double>p_nb.cost_factor
                        p_terrain_cost = p_terrain_cost * <double>grad_penalty_ptr[p_cidx]
                        p_edge_cost = p_terrain_cost + <double>p_nb.angle_cost
                        p_new_span_m = p_cur_span_m + <double>p_nb.step_distance

                        if p_nb.d_out == p_cur_dir:
                            # Same direction: continue span, carry height forward
                            if p_new_span_m < <double>max_span:
                                p_new_span_bin = <uint16_t>(p_new_span_m / span_bin_size)
                                p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                              <uint64_t>p_nb.d_out * p_sph +
                                              <uint64_t>p_new_span_bin * n_heights +
                                              <uint64_t>p_cur_hc)
                                p_new_dist = p_cur_dist + p_edge_cost
                                p_cnt = tbufs[tid].count
                                if p_cnt < tbufs[tid].capacity:
                                    tbufs[tid].states[p_cnt] = p_new_state
                                    tbufs[tid].dists[p_cnt] = p_new_dist
                                    tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                    tbufs[tid].span_dists[p_cnt] = <float>p_new_span_m
                                    tbufs[tid].count = p_cnt + 1

                            # Same direction: optional tower with height exploration
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                for p_hb in range(n_heights):
                                    p_h_b = th_ptr[p_hb]
                                    p_clearance_ok = _check_span_clearance_vh(
                                        p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                        p_h_a, p_h_b,
                                        conductor_weight_per_m, conductor_tension,
                                        min_clearance_val, dem_ptr, obstacle_ptr,
                                        rows, cols, directions, cached_steps,
                                        step_dist_view[p_cur_dir])
                                    if not p_clearance_ok:
                                        break
                                    p_hp_b = hp_ptr[p_hb]
                                    p_tower_cost = (_tower_terrain(
                                        use_area_cost, p_cur_row, p_cur_col,
                                        p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                        raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                        area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                        dem_ptr, cell_size, gradient_scale,
                                    ) + <double>p_nb.tower_angle_cost +
                                                   <double>p_hp_b)
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                                  <uint64_t>p_nb.d_out * p_sph +
                                                  <uint64_t>p_reset_span_bin * n_heights +
                                                  <uint64_t>p_hb)
                                    p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                        tbufs[tid].count = p_cnt + 1
                        else:
                            # Direction change: mandatory tower
                            if n_span_bins > 1 and p_cur_span_m >= <double>min_span:
                                for p_hb in range(n_heights):
                                    p_h_b = th_ptr[p_hb]
                                    p_clearance_ok = _check_span_clearance_vh(
                                        p_cur_row, p_cur_col, <float>p_cur_span_m, p_cur_dir,
                                        p_h_a, p_h_b,
                                        conductor_weight_per_m, conductor_tension,
                                        min_clearance_val, dem_ptr, obstacle_ptr,
                                        rows, cols, directions, cached_steps,
                                        step_dist_view[p_cur_dir])
                                    if not p_clearance_ok:
                                        break
                                    p_hp_b = hp_ptr[p_hb]
                                    p_tower_cost = (_tower_terrain(
                                        use_area_cost, p_cur_row, p_cur_col,
                                        p_cur_dir, p_nb.d_out, n_dirs, rows, cols,
                                        raster_ptr, tower_terrain_ptr, p_cur_tower_terrain,
                                        area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                        dem_ptr, cell_size, gradient_scale,
                                    ) + <double>p_nb.tower_angle_cost +
                                                   <double>p_hp_b)
                                    p_reset_span_m = <double>p_nb.step_distance
                                    p_reset_span_bin = <uint16_t>(p_reset_span_m / span_bin_size)
                                    p_new_state = ((<uint64_t>p_nb_cell) * spc +
                                                  <uint64_t>p_nb.d_out * p_sph +
                                                  <uint64_t>p_reset_span_bin * n_heights +
                                                  <uint64_t>p_hb)
                                    p_new_dist = p_cur_dist + p_edge_cost + p_tower_cost
                                    p_cnt = tbufs[tid].count
                                    if p_cnt < tbufs[tid].capacity:
                                        tbufs[tid].states[p_cnt] = p_new_state
                                        tbufs[tid].dists[p_cnt] = p_new_dist
                                        tbufs[tid].preds[p_cnt] = <int64_t>cur_state
                                        tbufs[tid].span_dists[p_cnt] = <float>p_reset_span_m
                                        tbufs[tid].count = p_cnt + 1

                # ---- SEQUENTIAL MERGE into hash map ----
                for t in range(num_threads):
                    for bi in range(tbufs[t].count):
                        p_new_state = tbufs[t].states[bi]
                        p_new_dist = tbufs[t].dists[bi]
                        it_nb = states.find(p_new_state)
                        if it_nb != states.end():
                            if deref(it_nb).second.visited != 0:
                                continue
                            if p_new_dist < deref(it_nb).second.dist:
                                deref(it_nb).second.dist = p_new_dist
                                deref(it_nb).second.pred = tbufs[t].preds[bi]
                                deref(it_nb).second.span_dist = tbufs[t].span_dists[bi]
                                new_logical = <size_t>(p_new_dist / delta_val)
                                buckets[new_logical & bucket_mask].push_back(p_new_state)
                                if new_logical > max_logical:
                                    max_logical = new_logical
                        else:
                            new_ls.dist = p_new_dist
                            new_ls.pred = tbufs[t].preds[bi]
                            new_ls.span_dist = tbufs[t].span_dists[bi]
                            new_ls.visited = 0
                            states[p_new_state] = new_ls
                            new_logical = <size_t>(p_new_dist / delta_val)
                            buckets[new_logical & bucket_mask].push_back(p_new_state)
                            if new_logical > max_logical:
                                max_logical = new_logical

            for bi in range(deferred.size()):
                cur_state = deferred[bi]
                it_def = states.find(cur_state)
                if it_def != states.end() and deref(it_def).second.visited == 0:
                    new_logical = <size_t>(deref(it_def).second.dist / delta_val)
                    if new_logical > current_logical:
                        buckets[new_logical & bucket_mask].push_back(cur_state)
            current_logical += 1

    finally:
        if tbufs != NULL:
            for t in range(num_threads):
                if tbufs[t].states != NULL: free(tbufs[t].states)
                if tbufs[t].dists != NULL: free(tbufs[t].dists)
                if tbufs[t].preds != NULL: free(tbufs[t].preds)
                if tbufs[t].span_dists != NULL: free(tbufs[t].span_dists)
            free(tbufs)

    # Path reconstruction via hash map predecessor chain
    cdef list path_cells = []
    cdef list tower_cells = []
    cdef list tower_height_classes = []

    if best_target_state == UINT64_MAX:
        free(icache_status); free(icache_cost); free(grad_penalty_ptr)
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir, walk_hc
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir, prev_hc
    cdef uint16_t prev_span
    cdef list state_chain = []

    while True:
        state_chain.append(walk_state)
        it_walk = states.find(walk_state)
        if it_walk == states.end() or deref(it_walk).second.pred == -1:
            break
        walk_state = <uint64_t>deref(it_walk).second.pred
    state_chain.reverse()

    cdef uint64_t st, prev_state_val
    cdef int k
    for k in range(len(state_chain)):
        st = <uint64_t>state_chain[k]
        _unpack_state_h(st, n_dirs, n_span_bins, n_heights,
                        &walk_cell, &walk_dir, &walk_span, &walk_hc)
        path_cells.append(<int>walk_cell)

        if k > 0 and n_span_bins > 1:
            prev_state_val = <uint64_t>state_chain[k - 1]
            _unpack_state_h(prev_state_val, n_dirs, n_span_bins, n_heights,
                            &prev_cell, &prev_dir, &prev_span, &prev_hc)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)
                    tower_height_classes.append(<int>prev_hc)

    cdef np.ndarray[np.float32_t, ndim=1] tower_h_out = np.empty(
        len(tower_height_classes), dtype=np.float32)
    for k in range(len(tower_height_classes)):
        tower_h_out[k] = tower_heights[tower_height_classes[k]]

    free(icache_status); free(icache_cost); free(grad_penalty_ptr)
    return (np.array(path_cells, dtype=np.uint32),
            np.array(tower_cells, dtype=np.uint32),
            tower_h_out)

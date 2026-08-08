# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False

"""Constrained Dijkstra with extended state (cell, direction, span_bin).

Extracted from constrained_path_algorithms.pyx.
Dense variant uses a bucket queue; sparse variant uses hash maps.
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
from cython.parallel cimport prange

from pyorps.utils._heap cimport (
    StepData, CachedStepData, IntermediatePoint, npy_intp,
    BinaryHeap64, PQNode64,
    heap64_init, heap64_push, heap64_pop, heap64_top, heap64_empty, heap64_free,
)
from pyorps.utils._raster_context cimport (
    precompute_directions, precompute_cached_steps,
)
from pyorps.utils._constrained_context cimport (
    ValidNeighbor, StateData,
    _pack_state, _unpack_state,
    _check_intermediates_ptr,
    _build_valid_neighbors, _flatten_valid_neighbors,
    _precompute_intermediate_cache, _precompute_gradient_cache,
    _tower_terrain, _tower_terrain_cell,
)


def constrained_dijkstra_2d(
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
    double initial_best_dist=INFINITY,
    np.ndarray[np.float32_t, ndim=2] dem_data=None,
    float cell_size=1.0,
    float max_gradient_pct=100.0,
    float gradient_scale=2.0,
    np.ndarray[np.int32_t, ndim=1] area_offsets=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_starts=None,
    np.ndarray[np.int32_t, ndim=1] area_offset_counts=None,
    np.ndarray[np.float32_t, ndim=2] tower_cost_raster=None,
    int force_sparse=0,
    int return_dist=0,
):
    """Find constrained shortest path with tower placement.

    Extended-state Dijkstra with state = (cell, direction, span_bin).
    Exact accumulated span tracked as float to avoid quantization drift.

    Automatically selects dense arrays when state space fits in ~2 GB,
    otherwise uses sparse hash maps.

    Parameters
    ----------
    initial_best_dist : double, optional
        Upper bound on the optimal cost, e.g. from a coarser search.
    dem_data : ndarray (float32, 2D), optional
        Digital Elevation Model aligned with raster. Enables 3D gradient
        penalties and max-gradient constraints.
    cell_size : float, optional
        Raster cell size in meters (for gradient computation).
    max_gradient_pct : float, optional
        Maximum allowed gradient in percent. Steeper edges are impassable.
    gradient_scale : float, optional
        Exponential penalty scale: penalty = exp(slope^2 * scale).
    tower_cost_raster : ndarray (float32, 2D), optional
        Precomputed per-cell tower foundation cost (feasibility plan
        Phase 8). When given, tower terrain costs come from this raster
        (indexed by CELL) instead of tower_terrain_costs[raster_value] —
        required under a feasibility objective, where the search raster
        holds combined weights that no longer identify the land use.
        Forbidden tower sites carry INFINITY.
    force_sparse : int, optional
        Nonzero forces the sparse heap-based implementation regardless of
        state-space size (testing hook: the heap path is exact by
        construction and serves as reference for the bucket path).
    return_dist : int, optional
        Nonzero appends the optimal path cost (float) to the return tuple:
        (path, towers, dist). Unreachable targets report inf.
    """
    if <double>n_span_bins * <double>span_bin_size < <double>max_span - 1e-6:
        raise ValueError(
            f"n_span_bins ({n_span_bins}) * span_bin_size ({span_bin_size}) "
            f"must cover max_span ({max_span}); otherwise span bins alias "
            f"into neighboring states and silently corrupt the search."
        )
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef uint64_t total_states = <uint64_t>total_cells * n_dirs * n_span_bins

    # Set up C pointers from ndarray area cost parameters
    cdef int use_area_cost_flag = 0
    cdef int32_t[:] ao_view, as_view, ac_view
    cdef const int32_t* area_offsets_ptr = NULL
    cdef const int32_t* area_starts_ptr = NULL
    cdef const int32_t* area_counts_ptr = NULL
    if area_offsets is not None:
        use_area_cost_flag = 1
        ao_view = area_offsets
        as_view = area_offset_starts
        ac_view = area_offset_counts
        area_offsets_ptr = &ao_view[0]
        area_starts_ptr = &as_view[0]
        area_counts_ptr = &ac_view[0]

    # Optional per-cell tower-cost raster (feasibility objective mode)
    cdef float[:, :] tcr_view
    cdef const float* tower_cost_ptr = NULL
    if tower_cost_raster is not None:
        tower_cost_raster = np.ascontiguousarray(tower_cost_raster)
        if (tower_cost_raster.shape[0] != rows or
                tower_cost_raster.shape[1] != cols):
            raise ValueError("tower_cost_raster shape must match raster")
        tcr_view = tower_cost_raster
        tower_cost_ptr = &tcr_view[0, 0]

    # Memory budget: StateData(24) per state for dense
    cdef uint64_t dense_limit = 500000000  # ~12 GB

    if total_states <= dense_limit and not force_sparse:
        return _dijkstra_dense(
            raster, source_row, source_col, target_row, target_col,
            steps, angle_cost_lut, angle_valid_lut, step_distances,
            tower_terrain_costs, tower_angle_costs,
            n_span_bins, span_bin_size, min_span, max_span, exclude_mask,
            initial_best_dist, dem_data, cell_size, max_gradient_pct,
            gradient_scale,
            use_area_cost_flag, area_offsets_ptr, area_starts_ptr, area_counts_ptr,
            tower_cost_ptr, return_dist,
        )
    else:
        return _dijkstra_sparse(
            raster, source_row, source_col, target_row, target_col,
            steps, angle_cost_lut, angle_valid_lut, step_distances,
            tower_terrain_costs, tower_angle_costs,
            n_span_bins, span_bin_size, min_span, max_span, exclude_mask,
            initial_best_dist, dem_data, cell_size, max_gradient_pct,
            gradient_scale,
            use_area_cost_flag, area_offsets_ptr, area_starts_ptr, area_counts_ptr,
            tower_cost_ptr, return_dist,
        )


# ==================== DENSE IMPLEMENTATION (BUCKET QUEUE) ====================

cdef _dijkstra_dense(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col,
    int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins, float span_bin_size, float min_span, float max_span,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask,
    double initial_best_dist,
    np.ndarray[np.float32_t, ndim=2] dem_data,
    float cell_size, float max_gradient_pct, float gradient_scale,
    int use_area_cost_flag=0,
    const int32_t* area_offsets_arg=NULL,
    const int32_t* area_starts_arg=NULL,
    const int32_t* area_counts_arg=NULL,
    const float* tower_cost_ptr=NULL,
    int return_dist=0,
):
    """Dense bucket-queue Dijkstra with AoS state layout for cache locality."""
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int spc = n_dirs * n_span_bins  # states_per_cell
    cdef uint64_t total_states = <uint64_t>total_cells * spc

    # Force C-contiguity for safe C pointer access.
    raster = np.ascontiguousarray(raster)
    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)
    else:
        exclude_mask = np.ascontiguousarray(exclude_mask)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    # Typed memoryviews (kept for API compatibility)
    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs

    # Raw C pointers for hot-loop indexing (no memoryview stride overhead)
    cdef uint16_t* raster_ptr = &raster_view[0, 0]
    cdef uint8_t* mask_ptr = &mask_view[0, 0]
    cdef float* tower_terrain_ptr = &tower_terrain_view[0]

    # Area cost offsets (exact mode) — uses C pointers from caller
    cdef int use_area_cost = use_area_cost_flag
    cdef const int32_t* area_offsets_ptr = area_offsets_arg
    cdef const int32_t* area_starts_ptr = area_starts_arg
    cdef const int32_t* area_counts_ptr = area_counts_arg

    # Build valid-neighbor LUT then flatten to contiguous array + offsets
    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs,
    )
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    # Compute delta: minimum possible terrain-only edge cost.
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

    # Circular bucket queue — power-of-2 size for fast modulo.
    cdef size_t n_phys_buckets = 65536  # 2^16
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    # AoS state array: all per-state data in one struct for cache locality.
    cdef StateData* states = <StateData*>calloc(total_states, sizeof(StateData))

    # Per-cell intermediate path cache — precomputed in parallel via OpenMP.
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))

    if states == NULL or icache_status == NULL or icache_cost == NULL:
        if states != NULL: free(states)
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate memory for constrained Dijkstra")

    # Parallel precomputation: fill icache for ALL (cell, d_out) pairs.
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

    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef uint8_t cur_dir
    cdef int cur_row, cur_col
    cdef double cur_span_m
    cdef uint16_t cur_raster_val
    cdef double cur_tower_terrain
    cdef uint64_t remainder

    cdef int nr, nc, k, n_valid, nb_start
    cdef uint32_t nb_cell
    cdef double edge_cost, terrain_cost, new_dist_val, tower_cost
    cdef double new_span_m
    cdef uint16_t new_span_bin
    cdef uint64_t new_state
    cdef float intermediate_cost_f
    cdef uint16_t reset_span_bin
    cdef double reset_span_m
    cdef ValidNeighbor nb

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = initial_best_dist

    cdef size_t current_logical = 0
    cdef size_t max_logical = 0
    cdef size_t phys_idx, new_logical, entry_logical
    cdef vector[uint64_t] batch
    cdef vector[uint64_t] deferred
    cdef size_t bi, bsize, di

    cdef size_t cache_idx
    cdef uint64_t base_nb_state
    cdef int skip_nb, sb

    while current_logical <= max_logical:
        if <double>current_logical * delta > best_target_dist:
            break

        phys_idx = current_logical & bucket_mask
        deferred.clear()

        batch.clear()
        while buckets[phys_idx].size() > 0:
            batch.swap(buckets[phys_idx])
            # Livelock fix: the swap put the PREVIOUS (already processed)
            # batch into the bucket -- discard it, or the two vectors
            # oscillate forever once any same-bucket push occurs.
            buckets[phys_idx].clear()

            bsize = batch.size()
            for bi in range(bsize):
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

                states[cur_state].visited = 1

                cur_cell = <uint32_t>(cur_state / spc)
                remainder = cur_state - (<uint64_t>cur_cell) * spc
                cur_dir = <uint8_t>(remainder / n_span_bins)

                if cur_cell == target_cell:
                    if cur_dist < best_target_dist:
                        best_target_dist = cur_dist
                        best_target_state = cur_state
                    continue

                cur_row = <int>(cur_cell / cols)
                cur_col = <int>(cur_cell % cols)
                cur_span_m = <double>states[cur_state].span_dist
                cur_raster_val = raster_ptr[cur_cell]
                if tower_cost_ptr != NULL:
                    cur_tower_terrain = <double>tower_cost_ptr[cur_cell]
                else:
                    cur_tower_terrain = <double>tower_terrain_ptr[cur_raster_val]

                nb_start = nb_offsets[cur_dir]
                n_valid = nb_offsets[cur_dir + 1] - nb_start
                for k in range(n_valid):
                    nb = flat_nb[nb_start + k]

                    nr = cur_row + nb.dr
                    nc = cur_col + nb.dc

                    if <unsigned int>nr >= <unsigned int>rows or <unsigned int>nc >= <unsigned int>cols:
                        continue

                    nb_cell = <uint32_t>(nr * cols + nc)

                    if mask_ptr[nb_cell] == 0:
                        continue

                    # Pre-visited neighbor skip
                    base_nb_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins
                    skip_nb = 1
                    for sb in range(n_span_bins):
                        if (states[base_nb_state + sb].visited == 0 or
                                <double>states[base_nb_state + sb].dist > cur_dist):
                            skip_nb = 0
                            break
                    if skip_nb:
                        continue

                    # Precomputed intermediate cache — just a lookup, no computation
                    cache_idx = <size_t>nb_cell * <size_t>n_dirs + <size_t>nb.d_out
                    if icache_status[cache_idx] != 2:
                        continue
                    intermediate_cost_f = icache_cost[cache_idx]

                    terrain_cost = (<double>cur_raster_val +
                                   <double>intermediate_cost_f +
                                   <double>raster_ptr[nb_cell]) * <double>nb.cost_factor
                    if grad_penalty_ptr != NULL:
                        terrain_cost = terrain_cost * <double>grad_penalty_ptr[cache_idx]
                    edge_cost = terrain_cost + <double>nb.angle_cost

                    new_span_m = cur_span_m + <double>nb.step_distance

                    if nb.d_out == cur_dir:
                        if new_span_m < <double>max_span:
                            new_span_bin = <uint16_t>(new_span_m / span_bin_size)
                            new_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins + new_span_bin
                            new_dist_val = cur_dist + edge_cost
                            if (states[new_state].visited == 0 or
                                    new_dist_val < states[new_state].dist):
                                if states[new_state].touched == 0 or new_dist_val < states[new_state].dist:
                                    states[new_state].touched = 1
                                    # Re-open protocol (see _constrained_delta)
                                    states[new_state].visited = 0
                                    states[new_state].dist = new_dist_val
                                    states[new_state].pred = <int64_t>cur_state
                                    states[new_state].span_dist = <float>new_span_m
                                    new_logical = <size_t>(new_dist_val / delta)
                                    buckets[new_logical & bucket_mask].push_back(new_state)
                                    if new_logical > max_logical:
                                        max_logical = new_logical

                        if n_span_bins > 1 and cur_span_m >= <double>min_span:
                            if tower_cost_ptr != NULL:
                                tower_cost = _tower_terrain_cell(
                                    use_area_cost, cur_row, cur_col,
                                    cur_dir, nb.d_out, n_dirs, rows, cols,
                                    tower_cost_ptr, cur_tower_terrain,
                                    area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                    dem_ptr_f, cell_size, gradient_scale,
                                ) + <double>nb.tower_angle_cost
                            else:
                                tower_cost = _tower_terrain(
                                    use_area_cost, cur_row, cur_col,
                                    cur_dir, nb.d_out, n_dirs, rows, cols,
                                    raster_ptr, tower_terrain_ptr, cur_tower_terrain,
                                    area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                    dem_ptr_f, cell_size, gradient_scale,
                                ) + <double>nb.tower_angle_cost
                            reset_span_m = <double>nb.step_distance
                            reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                            new_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins + reset_span_bin
                            new_dist_val = cur_dist + edge_cost + tower_cost
                            if (states[new_state].visited == 0 or
                                    new_dist_val < states[new_state].dist):
                                if states[new_state].touched == 0 or new_dist_val < states[new_state].dist:
                                    states[new_state].touched = 1
                                    # Re-open protocol (see _constrained_delta)
                                    states[new_state].visited = 0
                                    states[new_state].dist = new_dist_val
                                    states[new_state].pred = <int64_t>cur_state
                                    states[new_state].span_dist = <float>reset_span_m
                                    new_logical = <size_t>(new_dist_val / delta)
                                    buckets[new_logical & bucket_mask].push_back(new_state)
                                    if new_logical > max_logical:
                                        max_logical = new_logical
                    else:
                        if n_span_bins > 1 and cur_span_m >= <double>min_span:
                            if tower_cost_ptr != NULL:
                                tower_cost = _tower_terrain_cell(
                                    use_area_cost, cur_row, cur_col,
                                    cur_dir, nb.d_out, n_dirs, rows, cols,
                                    tower_cost_ptr, cur_tower_terrain,
                                    area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                    dem_ptr_f, cell_size, gradient_scale,
                                ) + <double>nb.tower_angle_cost
                            else:
                                tower_cost = _tower_terrain(
                                    use_area_cost, cur_row, cur_col,
                                    cur_dir, nb.d_out, n_dirs, rows, cols,
                                    raster_ptr, tower_terrain_ptr, cur_tower_terrain,
                                    area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                                    dem_ptr_f, cell_size, gradient_scale,
                                ) + <double>nb.tower_angle_cost
                            reset_span_m = <double>nb.step_distance
                            reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                            new_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins + reset_span_bin
                            new_dist_val = cur_dist + edge_cost + tower_cost
                            if (states[new_state].visited == 0 or
                                    new_dist_val < states[new_state].dist):
                                if states[new_state].touched == 0 or new_dist_val < states[new_state].dist:
                                    states[new_state].touched = 1
                                    # Re-open protocol (see _constrained_delta)
                                    states[new_state].visited = 0
                                    states[new_state].dist = new_dist_val
                                    states[new_state].pred = <int64_t>cur_state
                                    states[new_state].span_dist = <float>reset_span_m
                                    new_logical = <size_t>(new_dist_val / delta)
                                    buckets[new_logical & bucket_mask].push_back(new_state)
                                    if new_logical > max_logical:
                                        max_logical = new_logical

        for di in range(deferred.size()):
            cur_state = deferred[di]
            if states[cur_state].visited == 0:
                new_logical = <size_t>(states[cur_state].dist / delta)
                if new_logical > current_logical:
                    buckets[new_logical & bucket_mask].push_back(cur_state)

        current_logical += 1

    # Path reconstruction
    cdef list path_cells = []
    cdef list tower_cells = []

    if best_target_state == UINT64_MAX:
        free(states); free(icache_status); free(icache_cost)
        if grad_penalty_ptr != NULL: free(grad_penalty_ptr)
        if return_dist:
            return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32),
                    float(INFINITY))
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

    if return_dist:
        return (
            np.array(path_cells, dtype=np.uint32),
            np.array(tower_cells, dtype=np.uint32),
            float(best_target_dist),
        )
    return (
        np.array(path_cells, dtype=np.uint32),
        np.array(tower_cells, dtype=np.uint32),
    )


# ==================== SPARSE IMPLEMENTATION ====================

cdef _dijkstra_sparse(
    np.ndarray[uint16_t, ndim=2] raster,
    int source_row, int source_col,
    int target_row, int target_col,
    np.ndarray[np.int8_t, ndim=2] steps,
    np.ndarray[np.float32_t, ndim=2] angle_cost_lut,
    np.ndarray[np.uint8_t, ndim=2] angle_valid_lut,
    np.ndarray[np.float32_t, ndim=1] step_distances,
    np.ndarray[np.float32_t, ndim=1] tower_terrain_costs,
    np.ndarray[np.float32_t, ndim=2] tower_angle_costs,
    int n_span_bins, float span_bin_size, float min_span, float max_span,
    np.ndarray[np.uint8_t, ndim=2] exclude_mask,
    double initial_best_dist,
    np.ndarray[np.float32_t, ndim=2] dem_data,
    float cell_size, float max_gradient_pct, float gradient_scale,
    int use_area_cost_flag=0,
    const int32_t* area_offsets_arg=NULL,
    const int32_t* area_starts_arg=NULL,
    const int32_t* area_counts_arg=NULL,
    const float* tower_cost_ptr=NULL,
    int return_dist=0,
):
    """Sparse Dijkstra — for state spaces too large for dense arrays."""
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef int spc = n_dirs * n_span_bins

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

    # Area cost offsets (exact mode) — uses C pointers from caller
    cdef int use_area_cost = use_area_cost_flag
    cdef const int32_t* area_offsets_ptr = area_offsets_arg
    cdef const int32_t* area_starts_ptr = area_starts_arg
    cdef const int32_t* area_counts_ptr = area_counts_arg

    cdef vector[vector[ValidNeighbor]] nested_nb = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs,
    )
    cdef vector[ValidNeighbor] flat_nb
    cdef vector[int] nb_offsets
    _flatten_valid_neighbors(nested_nb, n_dirs, flat_nb, nb_offsets)
    nested_nb.clear()

    # Per-cell intermediate path cache — precomputed in parallel.
    cdef size_t cache_size = <size_t>total_cells * <size_t>n_dirs
    cdef uint8_t* icache_status = <uint8_t*>calloc(cache_size, sizeof(uint8_t))
    cdef float* icache_cost = <float*>calloc(cache_size, sizeof(float))

    if icache_status == NULL or icache_cost == NULL:
        if icache_status != NULL: free(icache_status)
        if icache_cost != NULL: free(icache_cost)
        raise MemoryError("Failed to allocate intermediate cache for sparse Dijkstra")

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
            free(icache_status); free(icache_cost)
            raise MemoryError("Failed to allocate gradient cache")
        _precompute_gradient_cache(
            total_cells, n_dirs, rows, cols,
            dem_ptr_f, cell_size, max_gradient_pct, gradient_scale,
            directions, icache_status, grad_penalty_ptr,
        )

    cdef unordered_map[uint64_t, double] dist_map
    cdef unordered_map[uint64_t, int64_t] pred_map
    cdef unordered_map[uint64_t, float] span_dist_map
    cdef unordered_set[uint64_t] visited_set

    cdef uint64_t reserve_count = 2000000
    if <uint64_t>total_cells < reserve_count:
        reserve_count = <uint64_t>total_cells
    dist_map.reserve(reserve_count)
    pred_map.reserve(reserve_count)
    span_dist_map.reserve(reserve_count)
    visited_set.reserve(reserve_count)

    cdef BinaryHeap64 heap
    heap64_init(&heap, 1048576)

    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx

    cdef int d
    for d in range(n_dirs):
        state_idx = (<uint64_t>source_cell) * spc + d * n_span_bins
        dist_map[state_idx] = 0.0
        pred_map[state_idx] = -1
        span_dist_map[state_idx] = 0.0
        heap64_push(&heap, state_idx, 0.0)

    cdef PQNode64 top_node
    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef uint8_t cur_dir
    cdef int cur_row, cur_col
    cdef double cur_span_m
    cdef uint16_t cur_raster_val
    cdef double cur_tower_terrain
    cdef uint64_t remainder

    cdef int nr, nc, k, n_valid, nb_start
    cdef uint32_t nb_cell
    cdef double edge_cost, terrain_cost, new_dist_val, tower_cost
    cdef double new_span_m
    cdef uint16_t new_span_bin
    cdef uint64_t new_state
    cdef float intermediate_cost_f
    cdef uint16_t reset_span_bin
    cdef double reset_span_m
    cdef ValidNeighbor nb

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = initial_best_dist
    cdef unordered_map[uint64_t, double].iterator dist_it
    cdef unordered_map[uint64_t, float].iterator span_it

    cdef size_t cache_idx

    while not heap64_empty(&heap):
        top_node = heap64_top(&heap)
        cur_state = top_node.index
        cur_dist = top_node.priority
        heap64_pop(&heap)

        if cur_dist > best_target_dist:
            break
        if visited_set.count(cur_state) > 0:
            continue

        dist_it = dist_map.find(cur_state)
        if cur_dist > deref(dist_it).second:
            continue
        visited_set.insert(cur_state)

        cur_cell = <uint32_t>(cur_state / spc)
        remainder = cur_state - (<uint64_t>cur_cell) * spc
        cur_dir = <uint8_t>(remainder / n_span_bins)

        if cur_cell == target_cell:
            if cur_dist < best_target_dist:
                best_target_dist = cur_dist
                best_target_state = cur_state
            continue

        cur_row = <int>(cur_cell / cols)
        cur_col = <int>(cur_cell % cols)

        span_it = span_dist_map.find(cur_state)
        cur_span_m = <double>deref(span_it).second

        cur_raster_val = raster_ptr[cur_cell]
        if tower_cost_ptr != NULL:
            cur_tower_terrain = <double>tower_cost_ptr[cur_cell]
        else:
            cur_tower_terrain = <double>tower_terrain_ptr[cur_raster_val]

        nb_start = nb_offsets[cur_dir]
        n_valid = nb_offsets[cur_dir + 1] - nb_start
        for k in range(n_valid):
            nb = flat_nb[nb_start + k]

            nr = cur_row + nb.dr
            nc = cur_col + nb.dc

            if <unsigned int>nr >= <unsigned int>rows or <unsigned int>nc >= <unsigned int>cols:
                continue

            nb_cell = <uint32_t>(nr * cols + nc)

            if mask_ptr[nb_cell] == 0:
                continue

            # Precomputed intermediate cache — just a lookup
            cache_idx = <size_t>nb_cell * <size_t>n_dirs + <size_t>nb.d_out
            if icache_status[cache_idx] != 2:
                continue
            intermediate_cost_f = icache_cost[cache_idx]

            terrain_cost = (<double>cur_raster_val +
                           <double>intermediate_cost_f +
                           <double>raster_ptr[nb_cell]) * <double>nb.cost_factor
            if grad_penalty_ptr != NULL:
                terrain_cost = terrain_cost * <double>grad_penalty_ptr[cache_idx]
            edge_cost = terrain_cost + <double>nb.angle_cost

            new_span_m = cur_span_m + <double>nb.step_distance

            if nb.d_out == cur_dir:
                if new_span_m < <double>max_span:
                    new_span_bin = <uint16_t>(new_span_m / span_bin_size)
                    new_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins + new_span_bin
                    if visited_set.count(new_state) == 0:
                        new_dist_val = cur_dist + edge_cost
                        dist_it = dist_map.find(new_state)
                        if dist_it == dist_map.end() or new_dist_val < deref(dist_it).second:
                            dist_map[new_state] = new_dist_val
                            pred_map[new_state] = <int64_t>cur_state
                            span_dist_map[new_state] = <float>new_span_m
                            heap64_push(&heap, new_state, new_dist_val)

                if n_span_bins > 1 and cur_span_m >= <double>min_span:
                    if tower_cost_ptr != NULL:
                        tower_cost = _tower_terrain_cell(
                            use_area_cost, cur_row, cur_col,
                            cur_dir, nb.d_out, n_dirs, rows, cols,
                            tower_cost_ptr, cur_tower_terrain,
                            area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                            dem_ptr_f, cell_size, gradient_scale,
                        ) + <double>nb.tower_angle_cost
                    else:
                        tower_cost = _tower_terrain(
                            use_area_cost, cur_row, cur_col,
                            cur_dir, nb.d_out, n_dirs, rows, cols,
                            raster_ptr, tower_terrain_ptr, cur_tower_terrain,
                            area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                            dem_ptr_f, cell_size, gradient_scale,
                        ) + <double>nb.tower_angle_cost
                    reset_span_m = <double>nb.step_distance
                    reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                    new_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins + reset_span_bin
                    if visited_set.count(new_state) == 0:
                        new_dist_val = cur_dist + edge_cost + tower_cost
                        dist_it = dist_map.find(new_state)
                        if dist_it == dist_map.end() or new_dist_val < deref(dist_it).second:
                            dist_map[new_state] = new_dist_val
                            pred_map[new_state] = <int64_t>cur_state
                            span_dist_map[new_state] = <float>reset_span_m
                            heap64_push(&heap, new_state, new_dist_val)
            else:
                if n_span_bins > 1 and cur_span_m >= <double>min_span:
                    if tower_cost_ptr != NULL:
                        tower_cost = _tower_terrain_cell(
                            use_area_cost, cur_row, cur_col,
                            cur_dir, nb.d_out, n_dirs, rows, cols,
                            tower_cost_ptr, cur_tower_terrain,
                            area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                            dem_ptr_f, cell_size, gradient_scale,
                        ) + <double>nb.tower_angle_cost
                    else:
                        tower_cost = _tower_terrain(
                            use_area_cost, cur_row, cur_col,
                            cur_dir, nb.d_out, n_dirs, rows, cols,
                            raster_ptr, tower_terrain_ptr, cur_tower_terrain,
                            area_offsets_ptr, area_starts_ptr, area_counts_ptr,
                            dem_ptr_f, cell_size, gradient_scale,
                        ) + <double>nb.tower_angle_cost
                    reset_span_m = <double>nb.step_distance
                    reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                    new_state = (<uint64_t>nb_cell) * spc + <uint64_t>nb.d_out * n_span_bins + reset_span_bin
                    if visited_set.count(new_state) == 0:
                        new_dist_val = cur_dist + edge_cost + tower_cost
                        dist_it = dist_map.find(new_state)
                        if dist_it == dist_map.end() or new_dist_val < deref(dist_it).second:
                            dist_map[new_state] = new_dist_val
                            pred_map[new_state] = <int64_t>cur_state
                            span_dist_map[new_state] = <float>reset_span_m
                            heap64_push(&heap, new_state, new_dist_val)

    # Path reconstruction
    cdef list path_cells = []
    cdef list tower_cells = []

    if best_target_state == UINT64_MAX:
        heap64_free(&heap)
        free(icache_status); free(icache_cost)
        if grad_penalty_ptr != NULL: free(grad_penalty_ptr)
        if return_dist:
            return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32),
                    float(INFINITY))
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir
    cdef uint16_t prev_span
    cdef unordered_map[uint64_t, int64_t].iterator pred_it

    cdef list state_chain = []
    while True:
        state_chain.append(walk_state)
        pred_it = pred_map.find(walk_state)
        if pred_it == pred_map.end() or deref(pred_it).second == -1:
            break
        walk_state = <uint64_t>deref(pred_it).second
    state_chain.reverse()

    cdef uint64_t st, prev_state_val
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

    heap64_free(&heap)
    free(icache_status); free(icache_cost)
    if grad_penalty_ptr != NULL: free(grad_penalty_ptr)

    if return_dist:
        return (
            np.array(path_cells, dtype=np.uint32),
            np.array(tower_cells, dtype=np.uint32),
            float(best_target_dist),
        )
    return (
        np.array(path_cells, dtype=np.uint32),
        np.array(tower_cells, dtype=np.uint32),
    )

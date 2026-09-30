# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Constrained pathfinding algorithms with extended state (cell, direction, span).

Coupled route + tower placement via extended-state Dijkstra.
State = (cell, direction, span_bin) where span_bin quantizes distance since
last tower. Exact accumulated span is tracked separately (float) to avoid
quantization drift when span_bin_size > step_distance.

Automatically selects dense arrays (fast) when the state space fits in memory,
or sparse hash maps when it doesn't.
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t, UINT64_MAX
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, calloc, free
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref

from pyorps.utils.path_core cimport (
    BinaryHeap64, PQNode64, StepData, CachedStepData,
    heap64_init, heap64_push, heap64_pop, heap64_top, heap64_empty, heap64_free,
    check_path_cached, precompute_directions, precompute_cached_steps,
    npy_intp
)


# Precomputed valid neighbor for a given incoming direction.
cdef struct ValidNeighbor:
    uint8_t d_out
    int dr
    int dc
    float cost_factor
    float step_distance
    float angle_cost
    float tower_angle_cost


cdef inline uint64_t _pack_state(uint32_t cell, uint8_t direction,
                                  uint16_t span_bin, int n_dirs,
                                  int n_span_bins) noexcept nogil:
    return (<uint64_t>cell) * n_dirs * n_span_bins + direction * n_span_bins + span_bin


cdef inline void _unpack_state(uint64_t state, int n_dirs, int n_span_bins,
                                uint32_t* cell, uint8_t* direction,
                                uint16_t* span_bin) noexcept nogil:
    cdef uint64_t states_per_cell = n_dirs * n_span_bins
    cell[0] = <uint32_t>(state / states_per_cell)
    cdef uint32_t remainder = <uint32_t>(state % states_per_cell)
    direction[0] = <uint8_t>(remainder / n_span_bins)
    span_bin[0] = <uint16_t>(remainder % n_span_bins)


# Python-accessible wrappers for testing
def pack_state(uint32_t cell, uint8_t direction, uint16_t span_bin,
               int n_dirs, int n_span_bins):
    """Pack (cell, direction, span_bin) into a single uint64 state index."""
    return _pack_state(cell, direction, span_bin, n_dirs, n_span_bins)


def unpack_state(uint64_t state, int n_dirs, int n_span_bins):
    """Unpack a state index into (cell, direction, span_bin)."""
    cdef uint32_t cell
    cdef uint8_t direction
    cdef uint16_t span_bin
    _unpack_state(state, n_dirs, n_span_bins, &cell, &direction, &span_bin)
    return int(cell), int(direction), int(span_bin)


# ==================== PRECOMPUTATION ====================

cdef vector[vector[ValidNeighbor]] _build_valid_neighbors(
    vector[StepData]& directions,
    float[:, :] angle_cost_view,
    uint8_t[:, :] angle_valid_view,
    float[:] step_dist_view,
    float[:, :] tower_angle_view,
    int n_dirs,
):
    """Build per-direction lists of valid outgoing neighbors."""
    cdef vector[vector[ValidNeighbor]] result
    cdef vector[ValidNeighbor] per_dir
    cdef ValidNeighbor nb
    cdef int d_in, d_out

    result.reserve(n_dirs)
    for d_in in range(n_dirs):
        per_dir = vector[ValidNeighbor]()
        for d_out in range(n_dirs):
            if angle_valid_view[d_in, d_out] != 0:
                nb.d_out = <uint8_t>d_out
                nb.dr = directions[d_out].dr
                nb.dc = directions[d_out].dc
                nb.cost_factor = directions[d_out].cost_factor
                nb.step_distance = step_dist_view[d_out]
                nb.angle_cost = angle_cost_view[d_in, d_out]
                nb.tower_angle_cost = tower_angle_view[d_in, d_out]
                per_dir.push_back(nb)
        result.push_back(per_dir)

    return result


# ==================== CONSTRAINED DIJKSTRA ====================

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
):
    """Find constrained shortest path with tower placement.

    Extended-state Dijkstra with state = (cell, direction, span_bin).
    Exact accumulated span tracked as float to avoid quantization drift.

    Automatically selects dense arrays when state space fits in ~2 GB,
    otherwise uses sparse hash maps.
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef uint64_t total_states = <uint64_t>total_cells * n_dirs * n_span_bins

    # Memory budget: dist(8) + pred(8) + visited(1) + span_dist(4) = 21 bytes/state
    cdef uint64_t dense_limit = 500000000  # ~10.5 GB

    if total_states <= dense_limit:
        return _dijkstra_dense(
            raster, source_row, source_col, target_row, target_col,
            steps, angle_cost_lut, angle_valid_lut, step_distances,
            tower_terrain_costs, tower_angle_costs,
            n_span_bins, span_bin_size, min_span, max_span, exclude_mask,
        )
    else:
        return _dijkstra_sparse(
            raster, source_row, source_col, target_row, target_col,
            steps, angle_cost_lut, angle_valid_lut, step_distances,
            tower_terrain_costs, tower_angle_costs,
            n_span_bins, span_bin_size, min_span, max_span, exclude_mask,
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
):
    """Dense bucket-queue Dijkstra for constrained pathfinding.

    Replaces binary heap (O(log n) per operation) with a circular bucket
    queue (O(1) amortized insert). Delta is auto-computed from the minimum
    terrain edge cost. Tower placement edges jump many buckets ahead.
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols
    cdef uint64_t total_states = <uint64_t>total_cells * n_dirs * n_span_bins

    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs

    cdef vector[vector[ValidNeighbor]] valid_nb_lut = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs,
    )
    # Compute delta: minimum possible terrain-only edge cost.
    # For any step, terrain_cost = (cur + intermediate + nb) * cost_factor.
    # Minimum is 2 * min_raster_val * min_cost_factor (cardinal, no intermediate).
    cdef double min_raster_val_d = 1.0
    traversable = raster[raster != 65535]
    if traversable.size > 0:
        min_raster_val_d = <double>max(1, int(traversable.min()))

    cdef double min_cf = 1e9
    cdef int dd
    for dd in range(n_dirs):
        if directions[dd].cost_factor < min_cf:
            min_cf = directions[dd].cost_factor
    # Shrink delta slightly below the minimum edge cost to guarantee that
    # every edge advances at least one logical bucket, preventing floating-
    # point rounding from cycling entries within the same bucket.
    cdef double delta = max(1.0, 2.0 * min_raster_val_d * min_cf * (1.0 - 1e-9))

    # Circular bucket queue — power-of-2 size for fast modulo.
    # Size must exceed max_single_edge_cost / delta to avoid wrap collisions.
    cdef size_t n_phys_buckets = 65536  # 2^16
    cdef size_t bucket_mask = n_phys_buckets - 1
    cdef vector[vector[uint64_t]] buckets
    buckets.resize(n_phys_buckets)

    # Allocate dense state arrays
    cdef double* dist = <double*>malloc(total_states * sizeof(double))
    cdef int64_t* pred = <int64_t*>malloc(total_states * sizeof(int64_t))
    cdef uint8_t* visited = <uint8_t*>calloc(total_states, sizeof(uint8_t))
    cdef float* span_dist = <float*>malloc(total_states * sizeof(float))

    if dist == NULL or pred == NULL or visited == NULL or span_dist == NULL:
        if dist != NULL: free(dist)
        if pred != NULL: free(pred)
        if visited != NULL: free(visited)
        if span_dist != NULL: free(span_dist)
        raise MemoryError("Failed to allocate memory for constrained Dijkstra")

    cdef uint64_t i
    for i in range(total_states):
        dist[i] = INFINITY
        pred[i] = -1
        span_dist[i] = 0.0
    cdef uint32_t source_cell = <uint32_t>(source_row * cols + source_col)
    cdef uint32_t target_cell = <uint32_t>(target_row * cols + target_col)
    cdef uint64_t state_idx

    cdef int d
    for d in range(n_dirs):
        state_idx = _pack_state(source_cell, <uint8_t>d, 0, n_dirs, n_span_bins)
        dist[state_idx] = 0.0
        buckets[0].push_back(state_idx)

    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef uint8_t cur_dir
    cdef uint16_t cur_span_bin
    cdef int cur_row, cur_col
    cdef double cur_span_m
    cdef uint16_t cur_raster_val
    cdef double cur_tower_terrain

    cdef int nr, nc, k, n_valid
    cdef uint32_t nb_cell
    cdef double edge_cost, terrain_cost, new_dist_val, tower_cost
    cdef double new_span_m
    cdef uint16_t new_span_bin
    cdef uint64_t new_state
    cdef float intermediate_cost_f
    cdef int valid_path
    cdef uint16_t reset_span_bin
    cdef double reset_span_m
    cdef ValidNeighbor nb

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY

    cdef size_t current_logical = 0
    cdef size_t max_logical = 0
    cdef size_t phys_idx, new_logical, entry_logical
    cdef vector[uint64_t] batch
    cdef vector[uint64_t] deferred
    cdef size_t bi, bsize, di

    while current_logical <= max_logical:
        # Early termination: all future buckets have dist > best_target_dist
        if <double>current_logical * delta > best_target_dist:
            break

        phys_idx = current_logical & bucket_mask

        # Collect entries that belong to future logical buckets but
        # collide into the same physical slot (wrap-around). These must
        # be re-inserted after we finish the current bucket.
        deferred.clear()

        # Process current bucket — keep going until empty.
        # Light edges (terrain only) may re-insert into the same bucket.
        batch.clear()  # Prevent stale entries from polluting swap
        while buckets[phys_idx].size() > 0:
            # O(1) swap: extract bucket contents without copying
            batch.swap(buckets[phys_idx])

            bsize = batch.size()
            for bi in range(bsize):
                cur_state = batch[bi]

                if visited[cur_state] != 0:
                    continue

                cur_dist = dist[cur_state]
                entry_logical = <size_t>(cur_dist / delta)

                if entry_logical != current_logical:
                    # Future bucket collision — save for re-insertion
                    if entry_logical > current_logical:
                        deferred.push_back(cur_state)
                    # else: past bucket, truly stale — drop
                    continue

                visited[cur_state] = 1

                _unpack_state(cur_state, n_dirs, n_span_bins,
                              &cur_cell, &cur_dir, &cur_span_bin)

                if cur_cell == target_cell:
                    if cur_dist < best_target_dist:
                        best_target_dist = cur_dist
                        best_target_state = cur_state
                    continue

                cur_row = <int>(cur_cell / cols)
                cur_col = <int>(cur_cell % cols)
                cur_span_m = <double>span_dist[cur_state]
                cur_raster_val = raster_view[cur_row, cur_col]
                cur_tower_terrain = <double>tower_terrain_view[cur_raster_val]

                n_valid = <int>valid_nb_lut[cur_dir].size()
                for k in range(n_valid):
                    nb = valid_nb_lut[cur_dir][k]

                    nr = cur_row + nb.dr
                    nc = cur_col + nb.dc

                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if mask_view[nr, nc] == 0:
                        continue

                    nb_cell = <uint32_t>(nr * cols + nc)

                    intermediate_cost_f = 0.0
                    valid_path = check_path_cached(
                        cached_steps[nb.d_out].intermediates,
                        cur_row, cur_col,
                        mask_view, raster_view, rows, cols,
                        &intermediate_cost_f)
                    if valid_path == 0:
                        continue

                    terrain_cost = (<double>cur_raster_val +
                                   <double>intermediate_cost_f +
                                   <double>raster_view[nr, nc]) * <double>nb.cost_factor
                    edge_cost = terrain_cost + <double>nb.angle_cost

                    new_span_m = cur_span_m + <double>nb.step_distance

                    if nb.d_out == cur_dir:
                        # Same direction: continue span (if under max_span)
                        if new_span_m < <double>max_span:
                            new_span_bin = <uint16_t>(new_span_m / span_bin_size)
                            new_state = _pack_state(nb_cell, nb.d_out, new_span_bin,
                                                    n_dirs, n_span_bins)
                            if visited[new_state] == 0:
                                new_dist_val = cur_dist + edge_cost
                                if new_dist_val < dist[new_state]:
                                    dist[new_state] = new_dist_val
                                    pred[new_state] = <int64_t>cur_state
                                    span_dist[new_state] = <float>new_span_m
                                    new_logical = <size_t>(new_dist_val / delta)
                                    buckets[new_logical & bucket_mask].push_back(new_state)
                                    if new_logical > max_logical:
                                        max_logical = new_logical

                        # Same direction: optionally place tower (span >= min_span)
                        if n_span_bins > 1 and cur_span_m >= <double>min_span:
                            tower_cost = cur_tower_terrain + <double>nb.tower_angle_cost
                            reset_span_m = <double>nb.step_distance
                            reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                            new_state = _pack_state(nb_cell, nb.d_out, reset_span_bin,
                                                    n_dirs, n_span_bins)
                            if visited[new_state] == 0:
                                new_dist_val = cur_dist + edge_cost + tower_cost
                                if new_dist_val < dist[new_state]:
                                    dist[new_state] = new_dist_val
                                    pred[new_state] = <int64_t>cur_state
                                    span_dist[new_state] = <float>reset_span_m
                                    new_logical = <size_t>(new_dist_val / delta)
                                    buckets[new_logical & bucket_mask].push_back(new_state)
                                    if new_logical > max_logical:
                                        max_logical = new_logical
                    else:
                        # Direction change: tower is MANDATORY (overhead lines need
                        # a tower at every turn). Bypass min_span check.
                        if n_span_bins > 1:
                            tower_cost = cur_tower_terrain + <double>nb.tower_angle_cost
                            reset_span_m = <double>nb.step_distance
                            reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                            new_state = _pack_state(nb_cell, nb.d_out, reset_span_bin,
                                                    n_dirs, n_span_bins)
                            if visited[new_state] == 0:
                                new_dist_val = cur_dist + edge_cost + tower_cost
                                if new_dist_val < dist[new_state]:
                                    dist[new_state] = new_dist_val
                                    pred[new_state] = <int64_t>cur_state
                                    span_dist[new_state] = <float>reset_span_m
                                    new_logical = <size_t>(new_dist_val / delta)
                                    buckets[new_logical & bucket_mask].push_back(new_state)
                                    if new_logical > max_logical:
                                        max_logical = new_logical

        # Re-insert deferred entries into their correct physical buckets.
        # These were from future logical buckets that collided with phys_idx.
        for di in range(deferred.size()):
            cur_state = deferred[di]
            if visited[cur_state] == 0:
                new_logical = <size_t>(dist[cur_state] / delta)
                if new_logical > current_logical:
                    buckets[new_logical & bucket_mask].push_back(cur_state)

        current_logical += 1

    # Path reconstruction
    cdef list path_cells = []
    cdef list tower_cells = []

    if best_target_state == UINT64_MAX:
        free(dist)
        free(pred)
        free(visited)
        free(span_dist)
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32))

    cdef uint64_t walk_state = best_target_state
    cdef uint32_t walk_cell
    cdef uint8_t walk_dir
    cdef uint16_t walk_span
    cdef uint32_t prev_cell
    cdef uint8_t prev_dir
    cdef uint16_t prev_span

    cdef list state_chain = []
    while walk_state != UINT64_MAX and pred[walk_state] != -1:
        state_chain.append(walk_state)
        walk_state = <uint64_t>pred[walk_state]
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
            # Tower detected by span_bin drop (span-based tower) OR
            # direction change (mandatory turn tower)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)

    free(dist)
    free(pred)
    free(visited)
    free(span_dist)

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
):
    """Sparse Dijkstra — for state spaces too large for dense arrays.

    Uses hash maps for state storage to avoid allocating flat arrays
    that don't fit in memory.
    """
    cdef int rows = raster.shape[0]
    cdef int cols = raster.shape[1]
    cdef int n_dirs = steps.shape[0]
    cdef int total_cells = rows * cols

    if exclude_mask is None:
        exclude_mask = (raster != 65535).astype(np.uint8)

    cdef vector[StepData] directions = precompute_directions(steps)
    cdef vector[CachedStepData] cached_steps = precompute_cached_steps(steps)

    cdef uint16_t[:, :] raster_view = raster
    cdef uint8_t[:, :] mask_view = exclude_mask
    cdef float[:, :] angle_cost_view = angle_cost_lut
    cdef uint8_t[:, :] angle_valid_view = angle_valid_lut
    cdef float[:] step_dist_view = step_distances
    cdef float[:] tower_terrain_view = tower_terrain_costs
    cdef float[:, :] tower_angle_view = tower_angle_costs

    cdef vector[vector[ValidNeighbor]] valid_nb_lut = _build_valid_neighbors(
        directions, angle_cost_view, angle_valid_view,
        step_dist_view, tower_angle_view, n_dirs,
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
        state_idx = _pack_state(source_cell, <uint8_t>d, 0, n_dirs, n_span_bins)
        dist_map[state_idx] = 0.0
        pred_map[state_idx] = -1
        span_dist_map[state_idx] = 0.0
        heap64_push(&heap, state_idx, 0.0)

    cdef PQNode64 top_node
    cdef uint64_t cur_state
    cdef double cur_dist
    cdef uint32_t cur_cell
    cdef uint8_t cur_dir
    cdef uint16_t cur_span_bin
    cdef int cur_row, cur_col
    cdef double cur_span_m
    cdef uint16_t cur_raster_val
    cdef double cur_tower_terrain

    cdef int nr, nc, k, n_valid
    cdef uint32_t nb_cell
    cdef double edge_cost, terrain_cost, new_dist_val, tower_cost
    cdef double new_span_m
    cdef uint16_t new_span_bin
    cdef uint64_t new_state
    cdef float intermediate_cost_f
    cdef int valid_path
    cdef uint16_t reset_span_bin
    cdef double reset_span_m
    cdef ValidNeighbor nb

    cdef uint64_t best_target_state = UINT64_MAX
    cdef double best_target_dist = INFINITY
    cdef unordered_map[uint64_t, double].iterator dist_it
    cdef unordered_map[uint64_t, float].iterator span_it

    while not heap64_empty(&heap):
        top_node = heap64_top(&heap)
        cur_state = top_node.index
        cur_dist = top_node.priority
        heap64_pop(&heap)

        if cur_dist > best_target_dist:
            break
        if visited_set.count(cur_state) > 0:
            continue

        # Stale entry check
        dist_it = dist_map.find(cur_state)
        if cur_dist > deref(dist_it).second:
            continue
        visited_set.insert(cur_state)

        _unpack_state(cur_state, n_dirs, n_span_bins, &cur_cell, &cur_dir, &cur_span_bin)

        if cur_cell == target_cell:
            if cur_dist < best_target_dist:
                best_target_dist = cur_dist
                best_target_state = cur_state
            continue

        cur_row = <int>(cur_cell / cols)
        cur_col = <int>(cur_cell % cols)

        # Exact accumulated span
        span_it = span_dist_map.find(cur_state)
        cur_span_m = <double>deref(span_it).second

        cur_raster_val = raster_view[cur_row, cur_col]
        cur_tower_terrain = <double>tower_terrain_view[cur_raster_val]

        n_valid = <int>valid_nb_lut[cur_dir].size()
        for k in range(n_valid):
            nb = valid_nb_lut[cur_dir][k]

            nr = cur_row + nb.dr
            nc = cur_col + nb.dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if mask_view[nr, nc] == 0:
                continue

            nb_cell = <uint32_t>(nr * cols + nc)

            intermediate_cost_f = 0.0
            valid_path = check_path_cached(
                cached_steps[nb.d_out].intermediates,
                cur_row, cur_col,
                mask_view, raster_view, rows, cols,
                &intermediate_cost_f)
            if valid_path == 0:
                continue

            terrain_cost = (<double>cur_raster_val +
                           <double>intermediate_cost_f +
                           <double>raster_view[nr, nc]) * <double>nb.cost_factor
            edge_cost = terrain_cost + <double>nb.angle_cost

            new_span_m = cur_span_m + <double>nb.step_distance

            if nb.d_out == cur_dir:
                # Same direction: continue span (if under max_span)
                if new_span_m < <double>max_span:
                    new_span_bin = <uint16_t>(new_span_m / span_bin_size)
                    new_state = _pack_state(nb_cell, nb.d_out, new_span_bin,
                                            n_dirs, n_span_bins)
                    if visited_set.count(new_state) == 0:
                        new_dist_val = cur_dist + edge_cost
                        dist_it = dist_map.find(new_state)
                        if dist_it == dist_map.end() or new_dist_val < deref(dist_it).second:
                            dist_map[new_state] = new_dist_val
                            pred_map[new_state] = <int64_t>cur_state
                            span_dist_map[new_state] = <float>new_span_m
                            heap64_push(&heap, new_state, new_dist_val)

                # Same direction: optionally place tower (span >= min_span)
                if n_span_bins > 1 and cur_span_m >= <double>min_span:
                    tower_cost = cur_tower_terrain + <double>nb.tower_angle_cost
                    reset_span_m = <double>nb.step_distance
                    reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                    new_state = _pack_state(nb_cell, nb.d_out, reset_span_bin,
                                            n_dirs, n_span_bins)
                    if visited_set.count(new_state) == 0:
                        new_dist_val = cur_dist + edge_cost + tower_cost
                        dist_it = dist_map.find(new_state)
                        if dist_it == dist_map.end() or new_dist_val < deref(dist_it).second:
                            dist_map[new_state] = new_dist_val
                            pred_map[new_state] = <int64_t>cur_state
                            span_dist_map[new_state] = <float>reset_span_m
                            heap64_push(&heap, new_state, new_dist_val)
            else:
                # Direction change: tower is MANDATORY. Bypass min_span.
                if n_span_bins > 1:
                    tower_cost = cur_tower_terrain + <double>nb.tower_angle_cost
                    reset_span_m = <double>nb.step_distance
                    reset_span_bin = <uint16_t>(reset_span_m / span_bin_size)
                    new_state = _pack_state(nb_cell, nb.d_out, reset_span_bin,
                                            n_dirs, n_span_bins)
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
            # Tower detected by span_bin drop (span-based tower) OR
            # direction change (mandatory turn tower)
            if (prev_span >= 1 and walk_span < prev_span) or (walk_dir != prev_dir):
                if <int>prev_cell not in tower_cells:
                    tower_cells.append(<int>prev_cell)

    heap64_free(&heap)

    return (
        np.array(path_cells, dtype=np.uint32),
        np.array(tower_cells, dtype=np.uint32),
    )

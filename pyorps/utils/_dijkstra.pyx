"""
Dijkstra solver classes for high-performance pathfinding on raster grids.

Extracted from path_algorithms.pyx as the third module in the OO refactoring.
Contains:
- group_by_proximity_uint32: spatial reordering for batch processing
- DijkstraSolver cdef class: owns dist/prev/visited arrays, provides methods
  for single-pair, single-source-multi-target, multi-source-multi-target,
  and some-pairs shortest path queries.
- Public API wrappers with unchanged signatures for backward compatibility.
"""

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY, fabs
from libcpp.vector cimport vector
from libcpp cimport bool

from pyorps.utils._heap cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t,
    float32_t, float64_t, npy_intp,
    StepData, CachedStepData, SystemLimits,
    BinaryHeap, PQNode, heap_init, heap_empty, heap_top, heap_push, heap_pop,
    ravel_index, unravel_index,
)
from pyorps.utils._raster_context cimport (
    RasterContext, check_path, precompute_directions,
)
from pyorps.utils._raster_context import path_cost_uint32


# ==================== SPATIAL OPTIMIZATION ====================

def group_by_proximity_uint32(np.ndarray[uint32_t, ndim=1] source_indices,
                              uint64_t cols):
    """
    Group source indices by spatial proximity (uint32_t version).

    Reorders source nodes by row coordinate to improve cache locality
    during multi-source pathfinding operations.

    Parameters:
        source_indices: 1D array of linear node indices to reorder
        cols: Number of columns in the raster (for coordinate conversion)

    Returns:
        1D array of node indices reordered by spatial proximity
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


# ==================== DIJKSTRA SOLVER CLASS ====================

cdef class DijkstraSolver:
    """
    Dijkstra shortest path solver that owns dist/prev/visited arrays.

    Constructed with a RasterContext that provides the raster data,
    exclude mask, and precomputed directions. Array allocation is done
    once in __cinit__; arrays are reset between queries via _reset().

    Methods:
        single_pair: single source to single target
        single_source_multi_target: one source to many targets
        multi_source_multi_target: all-pairs via batched one-to-many
        some_pairs: pairwise with central-node batching optimization
    """
    cdef RasterContext ctx
    cdef float64_t[:] dist
    cdef int32_t[:] prev
    cdef uint8_t[:] visited

    # O(1) target membership for single_source_multi_target: is_target[cell]
    # is 1 for every cell of the CURRENT target list. Allocated on the first
    # multi-target query and reused; set and cleared in O(num_targets), never
    # in O(total_cells). Stale marks left behind by an interrupted query are
    # harmless - they only make the linear target scan run for a cell that
    # matches no entry, which changes nothing.
    cdef object target_map_arr
    cdef uint8_t[:] is_target
    cdef bint has_target_map

    # Optional gradient terms (feasibility plan section 3.2):
    #   s-bin  b = min(int(|dem[v]-dem[u]| * bin_factor[d]), n_bins-1)
    #   weight w = terrain * mult_lut[b] + add_lut[b] * step_len[d]
    # mult_lut[b] == inf marks the edge forbidden (hard grade limit).
    cdef bint use_gradient
    cdef float32_t[:, :] grad_dem
    cdef float32_t[:] grad_mult
    cdef float32_t[:] grad_add
    cdef float32_t[:] grad_bin_factor
    cdef float32_t[:] grad_step_len
    cdef int grad_n_bins

    # Live heap for SearchSession resume. single_pair still resets via
    # reset_root, so one-shot queries stay exact and stateless.
    cdef BinaryHeap pq
    cdef bint search_active
    cdef bint multi_root
    cdef uint32_t current_root
    cdef int _expansion_count

    def __cinit__(self, RasterContext ctx):
        self.ctx = ctx
        cdef int n = ctx.total_cells
        self.dist = np.full(n, np.inf, dtype=np.float64)
        self.prev = np.full(n, -1, dtype=np.int32)
        self.visited = np.zeros(n, dtype=np.uint8)
        self.use_gradient = False
        self.grad_n_bins = 0
        self.has_target_map = False
        self.search_active = False
        self.multi_root = False
        self.current_root = 0
        self._expansion_count = 0
        heap_init(&self.pq)

    def set_gradient(self,
                     np.ndarray[float32_t, ndim=2] dem,
                     np.ndarray[float32_t, ndim=1] mult_lut,
                     np.ndarray[float32_t, ndim=1] add_lut,
                     np.ndarray[float32_t, ndim=1] bin_factor,
                     np.ndarray[float32_t, ndim=1] step_len_cells,
                     int n_bins):
        """Enable per-edge gradient terms for all subsequent queries.

        Parameters:
            dem: float32 DEM aligned to the raster grid (same shape).
            mult_lut: (n_bins,) multiplicative slope response Γ_mult
                (3D stretch × penalty; inf beyond the hard grade limit).
            add_lut: (n_bins,) additive slope response Γ_add, pre-scaled
                by the quantization scale.
            bin_factor: (n_dirs,) per-direction |Δh|→bin factor.
            step_len_cells: (n_dirs,) step length in cell units.
            n_bins: Number of LUT bins.
        """
        if (dem.shape[0] != self.ctx.rows or
                dem.shape[1] != self.ctx.cols):
            raise ValueError(
                f"DEM shape ({dem.shape[0]}, {dem.shape[1]}) does not "
                f"match the raster ({self.ctx.rows}, {self.ctx.cols})")
        if mult_lut.shape[0] != n_bins or add_lut.shape[0] != n_bins:
            raise ValueError("LUT sizes do not match n_bins")
        self.grad_dem = np.ascontiguousarray(dem)
        self.grad_mult = np.ascontiguousarray(mult_lut)
        self.grad_add = np.ascontiguousarray(add_lut)
        self.grad_bin_factor = np.ascontiguousarray(bin_factor)
        self.grad_step_len = np.ascontiguousarray(step_len_cells)
        self.grad_n_bins = n_bins
        self.use_gradient = True

    cdef _reset(self):
        """Reset arrays for a new query."""
        cdef int n = self.ctx.total_cells
        self.dist[:] = np.inf
        self.prev[:] = -1
        self.visited[:] = 0

    cdef _ensure_target_map(self):
        """Allocate the zero-filled cell -> target lookup on first use.

        Kept out of __cinit__ so single-pair solvers never pay the extra
        byte per cell.
        """
        if not self.has_target_map:
            self.target_map_arr = np.zeros(self.ctx.total_cells,
                                           dtype=np.uint8)
            self.is_target = self.target_map_arr
            self.has_target_map = True

    cdef np.ndarray[uint32_t, ndim=1] _reconstruct_path(self, uint32_t source, uint32_t target):
        """
        Reconstruct path from prev array.

        Written once, used by all methods. Returns empty array if no path exists.
        """
        if self.prev[target] == -1:
            return np.empty(0, dtype=np.uint32)

        cdef int path_length = 1
        cdef uint32_t current = target
        while current != source:
            current = self.prev[current]
            path_length += 1

        cdef np.ndarray[uint32_t, ndim=1] path = np.empty(path_length, dtype=np.uint32)
        current = target
        cdef int idx = path_length - 1

        while True:
            path[idx] = current
            if current == source:
                break
            current = self.prev[current]
            idx -= 1

        return path

    def reset_root(self, uint32_t source_idx):
        """Wipe labels and seed a new single-source search at ``source_idx``."""
        self._require_arrays()
        if source_idx >= <uint32_t>self.ctx.total_cells:
            raise IndexError(f"root {source_idx} lies outside the raster")
        self._reset()
        heap_init(&self.pq)
        self.dist[source_idx] = 0.0
        heap_push(&self.pq, source_idx, 0.0)
        self.search_active = True
        self.multi_root = False
        self.current_root = source_idx
        self._expansion_count = 0

    def reset_roots(self, cells_arr, labels_arr=None):
        """Wipe labels and seed a multi-root search (plan D1(b)).

        Root ``k`` starts at ``labels[k]`` (CELL units, the units ``dist``
        uses; ``None`` means every label is 0). Roots on excluded cells are
        skipped, as in :class:`MultiSourceSolver`; a cell listed twice keeps
        its lowest label. Afterwards :meth:`search_until`,
        :meth:`settle_all` and :meth:`extract_path` work as after
        :meth:`reset_root`, and a path runs from the root that reached the
        target -- i.e. it is ``min_k labels[k] + d(cells[k], target)``.

        For tests and small windows; the streamed, memory-lean variant is
        :meth:`MultiSourceSolver.solve_stream`.
        """
        self._require_arrays()
        cells_in = np.asarray(cells_arr).ravel()
        if cells_in.size and (np.any(cells_in < 0) or np.any(
                cells_in >= self.ctx.total_cells)):
            raise IndexError("a root lies outside the raster")
        cdef np.ndarray[uint32_t, ndim=1] cells = np.ascontiguousarray(
            cells_in, dtype=np.uint32)
        cdef Py_ssize_t n_roots = cells.shape[0]
        if n_roots == 0:
            raise ValueError("at least one root is required")
        cdef np.ndarray[float64_t, ndim=1] labels
        if labels_arr is None:
            labels = np.zeros(n_roots, dtype=np.float64)
        else:
            labels = np.ascontiguousarray(labels_arr, dtype=np.float64)
        if labels.shape[0] != n_roots:
            raise ValueError(f"{n_roots} roots but {labels.shape[0]} labels")
        if not np.all(np.isfinite(labels)) or np.any(labels < 0):
            raise ValueError("root labels must be finite and >= 0")
        self._reset()
        heap_init(&self.pq)
        cdef uint8_t[:, :] mask = self.ctx.exclude_mask_view
        cdef int cols = self.ctx.cols
        cdef npy_intp r, c
        cdef Py_ssize_t k
        cdef uint32_t cell
        for k in range(n_roots):
            cell = cells[k]
            unravel_index(cell, cols, &r, &c)
            if mask[<int>r, <int>c] == 0:
                continue
            if labels[k] < self.dist[cell]:
                self.dist[cell] = labels[k]
                heap_push(&self.pq, cell, labels[k])
        self.search_active = True
        self.multi_root = True
        self.current_root = cells[0]
        self._expansion_count = 0

    def is_settled(self, uint32_t idx):
        """True if ``idx`` has been dequeued (final Dijkstra label)."""
        return self.search_active and self.visited[idx] == 1

    def peek_dist(self, uint32_t idx):
        """Final distance label at ``idx``, in CELL units.

        ``inf`` when the cell is unreachable, out of range, or simply not
        settled YET. A cell that is only on the heap carries an upper
        bound, not a distance: handing that out would silently price a
        route too high and no caller could tell. Call ``search_until(idx)``
        (or ``settle_all()``) first to turn a tentative label into an
        answer -- that is what makes the number safe, not this method.

        Note the root is unsettled until it is popped, so
        ``reset_root(r); peek_dist(r)`` is ``inf``, not 0.0.
        """
        if not self.search_active:
            return float("inf")
        if idx >= <uint32_t>self.visited.shape[0]:
            return float("inf")
        if self.visited[idx] == 0:
            return float("inf")
        return float(self.dist[idx])

    def peek_dists(self, idx_arr):
        """``peek_dist`` over an index array, without the per-cell call.

        Parameters:
            idx_arr: 1D array-like of uint32 linear cell indices.

        Returns:
            float64 array of the same length; ``inf`` per element under
            exactly the ``peek_dist`` rule.
        """
        cdef np.ndarray[uint32_t, ndim=1] idxs = np.ascontiguousarray(
            idx_arr, dtype=np.uint32)
        cdef Py_ssize_t n = idxs.shape[0]
        cdef np.ndarray[float64_t, ndim=1] out = np.full(
            n, np.inf, dtype=np.float64)
        if not self.search_active:
            return out
        cdef uint32_t total = <uint32_t>self.visited.shape[0]
        cdef float64_t[:] dist = self.dist
        cdef uint8_t[:] visited = self.visited
        cdef float64_t[:] out_view = out
        cdef uint32_t[:] idx_view = idxs
        cdef Py_ssize_t i
        cdef uint32_t idx
        with nogil:
            for i in range(n):
                idx = idx_view[i]
                if idx < total and visited[idx] == 1:
                    out_view[i] = dist[idx]
        return out

    def dist_array(self):
        """Copy of the whole label array, ``inf`` wherever unsettled.

        The ``peek_dist`` rule applied to every cell at once. Only a
        complete field after ``settle_all()``; before that the unsettled
        remainder reads as unreachable.
        """
        if not self.search_active or self.dist.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        out = np.asarray(self.dist).copy()
        out[np.asarray(self.visited) == 0] = np.inf
        return out

    def prev_array(self):
        """Copy of the predecessor array, ``-1`` wherever unsettled.

        The companion to :meth:`dist_array`: together they are everything
        a settled field needs to be stored and reopened without re-running
        the search. Unsettled cells are forced to ``-1`` for the same
        reason ``dist_array`` forces ``inf`` -- a stale predecessor from a
        previous root would otherwise read as a usable parent.
        """
        if not self.search_active or self.prev.shape[0] == 0:
            return np.empty(0, dtype=np.int32)
        out = np.asarray(self.prev).copy()
        out[np.asarray(self.visited) == 0] = -1
        return out

    cdef _require_arrays(self):
        """Refuse to run on the empty arrays ``release()`` leaves behind."""
        if self.dist.shape[0] != self.ctx.total_cells:
            raise RuntimeError(
                "the label arrays were dropped by release(); build a new "
                "solver with make_dijkstra_solver")

    def extract_path(self, uint32_t target_idx):
        """Reconstruct the path from the current root to ``target_idx``.

        After :meth:`reset_roots` the walk ends at whichever root reached
        the target (its predecessor is ``-1``), not at ``current_root``.
        """
        if not self.search_active:
            return np.empty(0, dtype=np.uint32)
        if self.multi_root:
            return self._walk_to_seed(target_idx)
        if target_idx == self.current_root:
            return np.array([self.current_root], dtype=np.uint32)
        return self._reconstruct_path(self.current_root, target_idx)

    cdef np.ndarray[uint32_t, ndim=1] _walk_to_seed(self, uint32_t target):
        """Path from the seed that owns ``target`` to ``target``.

        Walks predecessors until one is ``-1``: with several roots there
        is no single source to stop at, and reading ``prev == -1`` as a
        uint32 cell id would run off the array.
        """
        if (target >= <uint32_t>self.visited.shape[0]
                or self.visited[target] == 0):
            return np.empty(0, dtype=np.uint32)
        cdef int length = 1
        cdef uint32_t walk = target
        while self.prev[walk] != -1:
            walk = <uint32_t>self.prev[walk]
            length += 1
        cdef np.ndarray[uint32_t, ndim=1] out = np.empty(length,
                                                         dtype=np.uint32)
        cdef int idx = length - 1
        walk = target
        while True:
            out[idx] = walk
            if self.prev[walk] == -1:
                break
            walk = <uint32_t>self.prev[walk]
            idx -= 1
        return out

    @property
    def expansion_count(self):
        return self._expansion_count

    def release(self):
        """Drop label arrays so the solver no longer pins the window."""
        self.dist = np.empty(0, dtype=np.float64)
        self.prev = np.empty(0, dtype=np.int32)
        self.visited = np.empty(0, dtype=np.uint8)
        heap_init(&self.pq)
        self.search_active = False
        self._expansion_count = 0

    def memory_bytes(self):
        # int64, as in the constrained solver below: 13 B/cell overflows a
        # C int above ~165 M cells and reports NEGATIVE bytes, which a
        # caller sizing a spill decision would read as "fits easily".
        cdef int64_t n = <int64_t>self.dist.shape[0]
        if n == 0:
            return 0
        return n * (8 + 4 + 1)

    def search_until(self, uint32_t target_idx):
        """Continue the live heap until ``target_idx`` is settled.

        Does not reset. Returns True if the target was settled.
        ``expansion_count`` is the number of nodes newly settled in this
        call (0 when the target was already in the disk).
        """
        self._expansion_count = 0
        if not self.search_active:
            raise RuntimeError("search_until requires reset_root first")
        if self.visited[target_idx] == 1:
            return True
        self._run(<int64_t>target_idx)
        return self.visited[target_idx] == 1

    def settle_all(self):
        """Drain the heap: every reachable cell gets its final label.

        Turns the partial disk a sequence of ``search_until`` calls grew
        into the full field. Work is not duplicated -- a cell is popped
        once for the lifetime of the root -- so settle-on-demand followed
        by ``settle_all`` costs exactly one Dijkstra in total.
        ``expansion_count`` reports what THIS call added.
        """
        self._expansion_count = 0
        if not self.search_active:
            raise RuntimeError("settle_all requires reset_root first")
        self._run(-1)

    cdef int _run(self, int64_t stop_at) except -1:
        """Pop-and-relax loop. ``stop_at < 0`` means 'drain the heap'."""
        cdef bint has_stop = stop_at >= 0
        cdef uint32_t stop_idx = 0
        if has_stop:
            stop_idx = <uint32_t>stop_at

        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        cdef vector[StepData] directions = self.ctx.directions

        cdef float64_t[:] dist = self.dist
        cdef int32_t[:] prev = self.prev
        cdef uint8_t[:] visited = self.visited

        cdef uint32_t current
        cdef double current_dist
        cdef npy_intp current_row, current_col
        cdef npy_intp neighbor_row, neighbor_col
        cdef uint32_t neighbor
        cdef double intermediate_cost = 0.0
        cdef double total_cost, new_dist
        cdef int valid_path
        cdef int i, dr, dc

        cdef bint use_grad = self.use_gradient
        cdef double grad_mult_val, height_diff
        cdef int slope_bin

        while not heap_empty(&self.pq):
            current = heap_top(&self.pq).index
            current_dist = heap_top(&self.pq).priority
            heap_pop(&self.pq)

            if visited[current] == 1 or current_dist > dist[current]:
                continue
            visited[current] = 1
            self._expansion_count += 1

            unravel_index(current, cols, &current_row, &current_col)

            for i in range(directions.size()):
                dr = directions[i].dr
                dc = directions[i].dc
                neighbor_row = current_row + dr
                neighbor_col = current_col + dc

                if (neighbor_row < 0 or neighbor_row >= rows or
                        neighbor_col < 0 or neighbor_col >= cols):
                    continue

                if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:
                    continue

                neighbor = ravel_index(<int>neighbor_row, <int>neighbor_col, cols)

                if visited[neighbor] == 1:
                    continue

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

                if use_grad:
                    height_diff = fabs(
                        <double>self.grad_dem[<int>neighbor_row,
                                              <int>neighbor_col] -
                        <double>self.grad_dem[<int>current_row,
                                              <int>current_col])
                    slope_bin = <int>(height_diff *
                                      <double>self.grad_bin_factor[i])
                    if slope_bin >= self.grad_n_bins:
                        slope_bin = self.grad_n_bins - 1
                    grad_mult_val = <double>self.grad_mult[slope_bin]
                    if grad_mult_val == INFINITY:
                        continue
                    total_cost = (total_cost * grad_mult_val +
                                  <double>self.grad_add[slope_bin] *
                                  <double>self.grad_step_len[i])

                new_dist = dist[current] + total_cost
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heap_push(&self.pq, neighbor, new_dist)

            if has_stop and current == stop_idx:
                # The stop node's OWN label is already final at pop time --
                # that part of Dijkstra's invariant does not need its
                # neighbors relaxed. But a RESUMED search (a later
                # search_until with a different target, or settle_all)
                # continues this same heap, and any node whose true
                # shortest path runs through this node needs the relaxation
                # above to have already happened, or it can be finalized
                # with a stale, too-high distance later. So: relax first,
                # break after -- stopping here only means "don't keep
                # popping BEYOND this node," not "skip this node's edges."
                break

        return 0

    def single_pair(self, uint32_t source_idx, uint32_t target_idx):
        """
        Find shortest path between two points in the raster.

        Wrapper around reset_root / search_until / extract_path so one-shot
        callers (and existing tests) keep today's reset-every-query behaviour.

        Parameters:
            source_idx: Linear index of starting cell
            target_idx: Linear index of destination cell

        Returns:
            1D numpy array (uint32) of linear cell indices forming the
            optimal path. Empty array if no path exists.
        """
        if source_idx == target_idx:
            return np.array([source_idx], dtype=np.uint32)

        self.reset_root(source_idx)
        self.search_until(target_idx)
        return self.extract_path(target_idx)

    def single_source_multi_target(self, uint32_t source_idx, targets_arr):
        """
        Find optimal paths from one source to multiple targets.

        Runs a single Dijkstra traversal from source_idx and terminates
        early once all targets have been settled.

        Parameters:
            source_idx: Linear index of the single starting cell
            targets_arr: 1D numpy array (uint32) of target cell indices

        Returns:
            List of numpy arrays, where each array is the optimal path from
            source to the corresponding target. Empty arrays for unreachable.
        """
        cdef np.ndarray[uint32_t, ndim=1] target_indices = np.asarray(
            targets_arr, dtype=np.uint32)
        cdef int num_targets = <int>target_indices.shape[0]
        cdef uint32_t[:] targets = target_indices

        self._reset()
        self._ensure_target_map()

        cdef uint8_t[:] is_target = self.is_target
        cdef int total_cells = self.ctx.total_cells
        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        cdef vector[StepData] directions = self.ctx.directions

        cdef float64_t[:] dist = self.dist
        cdef int32_t[:] prev = self.prev
        cdef uint8_t[:] visited = self.visited

        # Track which targets have been found for early termination
        cdef np.ndarray[uint8_t, ndim=1] target_found_arr = np.zeros(
            num_targets, dtype=np.uint8)
        cdef uint8_t[:] target_found = target_found_arr
        cdef int targets_remaining = num_targets
        cdef int t

        # Mark the target cells so the settle loop needs one array lookup
        # instead of a scan over the whole target list. Indices outside the
        # raster can never be settled, so they stay unmapped rather than
        # writing past the end of is_target.
        for t in range(num_targets):
            if <uint64_t>targets[t] < <uint64_t>total_cells:
                is_target[targets[t]] = 1

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

        # Gradient locals (hoisted; used only when use_grad is set)
        cdef bint use_grad = self.use_gradient
        cdef double grad_mult_val, height_diff
        cdef int slope_bin

        # Modified Dijkstra loop with multi-target termination
        while not heap_empty(&pq) and targets_remaining > 0:
            current = heap_top(&pq).index
            current_dist = heap_top(&pq).priority
            heap_pop(&pq)

            # Skip outdated entries and already visited nodes
            if visited[current] == 1 or current_dist > dist[current]:
                continue
            visited[current] = 1

            # Check if current node is any of our targets. The list scan runs
            # only for cells that are in the list, and each cell settles at
            # most once, so duplicate target entries are still all marked in
            # the same visit exactly as before.
            if is_target[current] != 0:
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

                # Per-edge gradient terms (chord slope over the step)
                if use_grad:
                    height_diff = fabs(
                        <double>self.grad_dem[<int>neighbor_row,
                                              <int>neighbor_col] -
                        <double>self.grad_dem[<int>current_row,
                                              <int>current_col])
                    slope_bin = <int>(height_diff *
                                      <double>self.grad_bin_factor[i])
                    if slope_bin >= self.grad_n_bins:
                        slope_bin = self.grad_n_bins - 1
                    grad_mult_val = <double>self.grad_mult[slope_bin]
                    if grad_mult_val == INFINITY:
                        continue  # hard grade limit: edge forbidden
                    total_cost = (total_cost * grad_mult_val +
                                  <double>self.grad_add[slope_bin] *
                                  <double>self.grad_step_len[i])

                # Update shortest path if improvement found
                new_dist = dist[current] + total_cost
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heap_push(&pq, neighbor, new_dist)

        # Hand the membership map back clean for the next query
        for t in range(num_targets):
            if <uint64_t>targets[t] < <uint64_t>total_cells:
                is_target[targets[t]] = 0

        # Reconstruct paths for all targets
        cdef uint32_t target_idx
        cdef list paths = []
        for t in range(num_targets):
            target_idx = targets[t]
            paths.append(self._reconstruct_path(source_idx, target_idx))

        return paths

    def multi_source_multi_target(self, sources_arr, targets_arr,
                                  bint return_paths=True):
        """
        Compute all-pairs shortest paths between multiple sources and targets.

        Processes sources in spatial proximity order for better cache locality.
        Delegates to single_source_multi_target per source.

        Parameters:
            sources_arr: 1D array (uint32) of all source cell indices
            targets_arr: 1D array (uint32) of all target cell indices
            return_paths: If True, returns paths; if False, returns cost matrix

        Returns:
            If return_paths=True: List of lists, paths[i][j] = source i to target j
            If return_paths=False: 2D cost matrix with distances
        """
        cdef np.ndarray[uint32_t, ndim=1] source_indices = np.asarray(
            sources_arr, dtype=np.uint32)
        cdef np.ndarray[uint32_t, ndim=1] target_indices = np.asarray(
            targets_arr, dtype=np.uint32)
        cdef np.ndarray[uint16_t, ndim=2] raster_arr = np.asarray(self.ctx.raster_view)

        cdef int cols = self.ctx.cols
        cdef int num_sources = <int>source_indices.shape[0]
        cdef int num_targets = <int>target_indices.shape[0]

        # Declare variables
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
        sorted_sources = group_by_proximity_uint32(
            source_indices, <uint64_t> cols)

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
            source_paths = self.single_source_multi_target(
                source_idx, target_indices
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
                        cost = path_cost_uint32(
                            source_paths[t], raster_arr, cols)
                        cost_matrix[original_idx, t] = cost

        return paths if return_paths else cost_matrix

    def some_pairs(self, sources_arr, targets_arr, bint return_paths=True):
        """
        Find optimal paths for specific source-target pairs using batch optimization.

        Identifies central nodes (appearing as both source and target) and
        batches related queries through them to minimize Dijkstra runs.

        Parameters:
            sources_arr: 1D array (uint32) of source cell indices
            targets_arr: 1D array (uint32) of target cell indices
                        (pairs formed by matching array positions)
            return_paths: If True, returns actual paths; if False, returns costs

        Returns:
            If return_paths=True: List of path arrays (may contain empty arrays)
            If return_paths=False: 1D array of path costs (inf for no path)
        """
        cdef np.ndarray[uint32_t, ndim=1] source_indices = np.asarray(
            sources_arr, dtype=np.uint32)
        cdef np.ndarray[uint32_t, ndim=1] target_indices = np.asarray(
            targets_arr, dtype=np.uint32)
        cdef np.ndarray[uint16_t, ndim=2] raster_arr = np.asarray(self.ctx.raster_view)

        cdef int cols = self.ctx.cols
        cdef int num_pairs = <int> min(source_indices.shape[0],
                                       target_indices.shape[0])

        # Initialize result containers
        cdef list all_paths = [None] * num_pairs if return_paths else None
        cdef np.ndarray[float64_t, ndim=1] costs = np.full(num_pairs, np.inf)

        # Data structures for batching optimization
        cdef dict node_sources = {}  # target -> [sources pointing to it]
        cdef dict node_targets = {}  # source -> [targets it points to]
        cdef dict pair_indices = {}  # (source, target) -> original index
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
                        pair_mapping.append(
                            pair_indices[(central_node, target)])
                        reverse_flags.append(False)  # No reversal needed
                        processed_pairs.add((central_node, target))

            # Add reverse paths (central_node as target, compute backward)
            if central_node in node_sources:
                for source in node_sources[central_node]:
                    if (source, central_node) not in processed_pairs:
                        batch_targets.append(source)
                        pair_mapping.append(
                            pair_indices[(source, central_node)])
                        reverse_flags.append(True)  # Reversal needed
                        processed_pairs.add((source, central_node))

            # Execute batched computation if targets found
            if batch_targets:
                targets_array = np.array(batch_targets, dtype=np.uint32)
                result_paths = self.single_source_multi_target(
                    central_node, targets_array
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
                            all_paths[pair_idx] = np.empty(
                                0, dtype=np.uint32)
                    else:
                        # Calculate path cost
                        if len(path) > 0:
                            costs[pair_idx] = path_cost_uint32(
                                path, raster_arr, cols)

        # Phase 3: Handle remaining unprocessed pairs individually
        for i in range(num_pairs):
            source = source_indices[i]
            target = target_indices[i]

            if (source, target) in processed_pairs:
                continue

            # Process individual pair with single-target Dijkstra
            result_paths = self.single_source_multi_target(
                source, np.array([target], dtype=np.uint32)
            )

            path = result_paths[0]
            if return_paths:
                all_paths[i] = path
            else:
                if len(path) > 0:
                    costs[i] = path_cost_uint32(path, raster_arr, cols)

            processed_pairs.add((source, target))

        return all_paths if return_paths else costs


# ==================== MULTI-SOURCE (CORRIDOR) SOLVER ====================

cdef class MultiSourceSolver:
    """
    Multi-source Dijkstra that labels every cell with the terminal that owns it.

    Sibling of :class:`DijkstraSolver`, sharing the same ``RasterContext`` and
    therefore the same raster, exclude mask, direction table, ``check_path``
    validation and gradient LUTs. Three deltas:

    1. ALL terminals are seeded at ``dist = 0`` before the loop, each carrying
       its own ``region`` label.
    2. The label is propagated wherever ``prev[neighbor]`` is written, so the
       ``prev`` forest becomes a forest of shortest-path trees rooted at the
       terminals and ``region`` is the discrete Voronoi partition induced by
       the routing metric.
    3. There is no early termination: the heap is drained, so every reachable
       cell gets a final label.

    Reusing the context verbatim is what makes the corridor inherit exact
    PYORPS routing semantics: every edge is weighted by the same arithmetic,
    gradient terms included, so a segment recovered from this forest is a
    SHORTEST path between its endpoints and costs exactly what the pairwise
    search would charge for it.

    It need not be the SAME shortest path. The tie-break below is an extra
    rule DijkstraSolver does not have (it keeps whichever equal-cost
    relaxation the heap ran first), so with one terminal the two solvers agree
    on ``dist`` cell for cell but their ``prev`` forests differ wherever the
    optimum is not unique -- on a raster with few cost categories, that is
    most cells. Only the cost is a promise; the geometry is one optimum among
    several, chosen deterministically.

    Determinism
    -----------
    Ties are pervasive on rasters with few cost categories and the binary heap
    has no stable order for equal priorities, so without a rule the partition
    would vary with the order the terminals are passed in. Relaxation therefore
    applies a documented lexicographic rule on
    ``(new_dist, terminal_cell[region[current]], current)``: a strictly shorter
    label always wins, and an equal label wins only when it comes from a
    terminal sitting on a lower CELL index, or from the lower cell index within
    the same terminal.

    The tie-break keys on the terminal's cell, never on its position in the
    argument array. Keying on the position would make the answer depend on the
    order the caller happened to list the terminals in, which is precisely the
    dependency the rule exists to remove.

    With strictly positive step costs that rule makes the whole forest
    permutation-independent: every cell ``u`` that can tie for ``v`` satisfies
    ``dist[u] < dist[v]``, so ``u`` is popped before ``v`` is settled and its
    candidate is always seen. Zero-cost steps (a raster region of value 0)
    break that argument -- ``v`` can settle before a tied ``u`` is popped --
    and are reported by :meth:`has_zero_cost_steps`.
    """
    cdef RasterContext ctx
    cdef float64_t[:] dist
    cdef int32_t[:] prev
    cdef uint8_t[:] visited
    cdef int32_t[:] region

    # Optional gradient terms, identical to DijkstraSolver: the corridor must
    # see the same per-edge weights the pairwise search does, or segment
    # metrics silently diverge from find_route metrics.
    cdef bint use_gradient
    cdef float32_t[:, :] grad_dem
    cdef float32_t[:] grad_mult
    cdef float32_t[:] grad_add
    cdef float32_t[:] grad_bin_factor
    cdef float32_t[:] grad_step_len
    cdef int grad_n_bins

    cdef object _terminals_arr
    # Tie-break key per region: the terminal's CELL index. Held as a view so
    # the relaxation inner loop compares two uint32 loads instead of touching
    # a Python object.
    cdef uint32_t[:] region_key
    cdef bint _solved
    # lean: allocate only dist + visited (9 B/cell). prev and region are then
    # empty until solve_stream(keep_prev/keep_owner=True) asks for them.
    cdef bint lean
    # What the LAST solve recorded: predecessors, owners, and whether its
    # weights were the plain raster ones (boundary_steps needs all three).
    cdef bint _has_prev
    cdef bint _has_owner
    cdef bint _plain_weights

    def __cinit__(self, RasterContext ctx, bint lean=False):
        self.ctx = ctx
        cdef int n = ctx.total_cells
        self.lean = lean
        self.dist = np.full(n, np.inf, dtype=np.float64)
        self.visited = np.zeros(n, dtype=np.uint8)
        if lean:
            self.prev = np.empty(0, dtype=np.int32)
            self.region = np.empty(0, dtype=np.int32)
        else:
            self.prev = np.full(n, -1, dtype=np.int32)
            self.region = np.full(n, -1, dtype=np.int32)
        self.use_gradient = False
        self.grad_n_bins = 0
        self._terminals_arr = np.empty(0, dtype=np.uint32)
        self.region_key = np.empty(0, dtype=np.uint32)
        self._solved = False
        self._has_prev = False
        self._has_owner = False
        self._plain_weights = True

    def set_gradient(self,
                     np.ndarray[float32_t, ndim=2] dem,
                     np.ndarray[float32_t, ndim=1] mult_lut,
                     np.ndarray[float32_t, ndim=1] add_lut,
                     np.ndarray[float32_t, ndim=1] bin_factor,
                     np.ndarray[float32_t, ndim=1] step_len_cells,
                     int n_bins):
        """Enable per-edge gradient terms, exactly as DijkstraSolver does."""
        if (dem.shape[0] != self.ctx.rows or
                dem.shape[1] != self.ctx.cols):
            raise ValueError(
                f"DEM shape ({dem.shape[0]}, {dem.shape[1]}) does not "
                f"match the raster ({self.ctx.rows}, {self.ctx.cols})")
        if mult_lut.shape[0] != n_bins or add_lut.shape[0] != n_bins:
            raise ValueError("LUT sizes do not match n_bins")
        # bin_factor and step_len_cells are indexed by DIRECTION in the
        # relaxation loop, so a short array is an out-of-bounds read with
        # boundscheck off, not an exception.
        if (bin_factor.shape[0] != <Py_ssize_t>self.ctx.directions.size()
                or step_len_cells.shape[0] !=
                <Py_ssize_t>self.ctx.directions.size()):
            raise ValueError(
                f"bin_factor and step_len_cells need one entry per "
                f"direction ({self.ctx.directions.size()}), got "
                f"{bin_factor.shape[0]} and {step_len_cells.shape[0]}")
        self.grad_dem = np.ascontiguousarray(dem)
        self.grad_mult = np.ascontiguousarray(mult_lut)
        self.grad_add = np.ascontiguousarray(add_lut)
        self.grad_bin_factor = np.ascontiguousarray(bin_factor)
        self.grad_step_len = np.ascontiguousarray(step_len_cells)
        self.grad_n_bins = n_bins
        self.use_gradient = True

    cdef _reset(self):
        self.dist[:] = np.inf
        self.prev[:] = -1
        self.visited[:] = 0
        self.region[:] = -1
        self._solved = False

    def has_zero_cost_steps(self):
        """True when a zero-cost step exists, i.e. the raster holds a 0 cell.

        The lexicographic tie-break is permutation-independent only when every
        step costs something; this is the cheap witness for the exception. The
        step weight is ``(raster[u] + intermediates + raster[v]) *
        cost_factor``, so it vanishes exactly when every cell it touches is 0.
        """
        # `bool` is the C++ one here (cimported from libcpp), so build the
        # Python answer explicitly rather than calling it.
        if np.any(np.asarray(self.ctx.raster_view) == 0):
            return True
        return False

    def solve(self, terminals_arr):
        """Seed every terminal and drain the heap.

        Parameters:
            terminals_arr: 1D array-like (uint32) of linear terminal cell
                indices. Position in this array is the region label.

        Returns:
            The number of cells settled.

        Terminals on an excluded cell are skipped (they can never be settled
        and would otherwise seed a region no route can leave); terminals
        sharing a cell keep the LOWEST index, which is the same rule the
        relaxation tie-break applies everywhere else.
        """
        cdef np.ndarray[uint32_t, ndim=1] terminals = np.ascontiguousarray(
            terminals_arr, dtype=np.uint32)
        cdef int num_terminals = <int>terminals.shape[0]
        if num_terminals == 0:
            raise ValueError("At least one terminal is required")
        if self.lean:
            raise RuntimeError(
                "solve() needs the predecessor and region arrays; build the "
                "solver with lean=False")
        cdef uint32_t[:] term_view = terminals

        # release() drops the label arrays but keeps the context, so
        # total_cells still reports the full window while dist/prev/visited
        # are zero-length. Without this the seeding loop writes through empty
        # memoryviews, sets _solved anyway, and boundary_steps() then scans
        # rows x cols over nothing.
        if self.dist.shape[0] != self.ctx.total_cells:
            raise RuntimeError(
                "solve() was called after release(): the label arrays were "
                "dropped. Build a new solver with make_multi_source_solver.")

        # Every crossing is reported once, in the orientation whose current
        # cell holds the smaller region label; that is complete only if the
        # reverse of every step is also in the table. An asymmetric table
        # would silently lose whole terminal pairs rather than misprice them,
        # so refuse it here instead of returning a corridor with holes.
        cdef vector[StepData] dirs_check = self.ctx.directions
        cdef set forward = set()
        cdef Py_ssize_t di
        for di in range(<Py_ssize_t>dirs_check.size()):
            forward.add((dirs_check[di].dr, dirs_check[di].dc))
        if any((-dr_, -dc_) not in forward for dr_, dc_ in forward):
            raise ValueError(
                "the neighbourhood step table is not closed under negation, "
                "so boundary_steps() would drop crossings in one direction. "
                "Use get_neighborhood_steps(..., directed=True).")

        # Validate every seed BEFORE writing any of them, so a bad index
        # cannot leave a half-seeded solver behind.
        cdef int t_check
        for t_check in range(num_terminals):
            if <uint64_t>term_view[t_check] >= <uint64_t>self.ctx.total_cells:
                raise IndexError(
                    f"Terminal index {term_view[t_check]} is outside the "
                    f"raster ({self.ctx.total_cells} cells)")

        self._reset()
        self._terminals_arr = terminals
        self.region_key = terminals

        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef int total_cells = self.ctx.total_cells
        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        cdef vector[StepData] directions = self.ctx.directions

        cdef float64_t[:] dist = self.dist
        cdef int32_t[:] prev = self.prev
        cdef uint8_t[:] visited = self.visited
        cdef int32_t[:] region = self.region

        cdef BinaryHeap pq
        heap_init(&pq)

        cdef int t
        cdef uint32_t seed
        cdef npy_intp seed_row, seed_col
        for t in range(num_terminals):
            seed = term_view[t]
            unravel_index(seed, cols, &seed_row, &seed_col)
            if exclude_mask[<int>seed_row, <int>seed_col] == 0:
                continue
            if region[seed] != -1:
                # Coincident terminal: the lower index already owns the cell.
                continue
            dist[seed] = 0.0
            region[seed] = t
            heap_push(&pq, seed, 0.0)

        cdef uint32_t current
        cdef double current_dist
        cdef npy_intp current_row, current_col
        cdef npy_intp neighbor_row, neighbor_col
        cdef uint32_t neighbor
        cdef double intermediate_cost = 0.0
        cdef double total_cost, new_dist
        cdef int valid_path
        cdef int i, dr, dc
        cdef int settled = 0
        cdef int32_t cand_region, held_region
        cdef int32_t incumbent
        cdef bint improves

        cdef bint use_grad = self.use_gradient
        cdef double grad_mult_val, height_diff
        cdef int slope_bin

        while not heap_empty(&pq):
            current = heap_top(&pq).index
            current_dist = heap_top(&pq).priority
            heap_pop(&pq)

            if visited[current] == 1 or current_dist > dist[current]:
                continue
            visited[current] = 1
            settled += 1

            unravel_index(current, cols, &current_row, &current_col)

            for i in range(directions.size()):
                dr = directions[i].dr
                dc = directions[i].dc
                neighbor_row = current_row + dr
                neighbor_col = current_col + dc

                if (neighbor_row < 0 or neighbor_row >= rows or
                        neighbor_col < 0 or neighbor_col >= cols):
                    continue

                if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:
                    continue

                neighbor = ravel_index(<int>neighbor_row, <int>neighbor_col,
                                       cols)

                if visited[neighbor] == 1:
                    continue

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

                if use_grad:
                    height_diff = fabs(
                        <double>self.grad_dem[<int>neighbor_row,
                                              <int>neighbor_col] -
                        <double>self.grad_dem[<int>current_row,
                                              <int>current_col])
                    slope_bin = <int>(height_diff *
                                      <double>self.grad_bin_factor[i])
                    if slope_bin >= self.grad_n_bins:
                        slope_bin = self.grad_n_bins - 1
                    grad_mult_val = <double>self.grad_mult[slope_bin]
                    if grad_mult_val == INFINITY:
                        continue  # hard grade limit: edge forbidden
                    total_cost = (total_cost * grad_mult_val +
                                  <double>self.grad_add[slope_bin] *
                                  <double>self.grad_step_len[i])

                new_dist = dist[current] + total_cost

                # Lexicographic (new_dist, terminal_cell[region[current]],
                # current). The equality branch is what makes the partition
                # independent of the order the terminals were passed in;
                # without it the winner is whichever relaxation the heap
                # happened to run first.
                improves = False
                if new_dist < dist[neighbor]:
                    improves = True
                elif new_dist == dist[neighbor]:
                    incumbent = prev[neighbor]
                    if incumbent != -1:
                        cand_region = region[current]
                        held_region = region[<uint32_t>incumbent]
                        if cand_region != held_region:
                            improves = (self.region_key[cand_region] <
                                        self.region_key[held_region])
                        else:
                            improves = current < <uint32_t>incumbent

                if improves:
                    dist[neighbor] = new_dist
                    prev[neighbor] = <int32_t>current
                    region[neighbor] = region[current]
                    heap_push(&pq, neighbor, new_dist)

        self._solved = True
        self._has_prev = True
        self._has_owner = True
        self._plain_weights = True
        return settled

    def solve_stream(self, order_arr, labels_arr, double length_rate=0.0,
                     double weight_mult=1.0, no_transit=None,
                     bint keep_prev=False, bint keep_owner=False):
        """Seeded multi-source Dijkstra over a presorted seed stream (plan D1).

        Computes ``dist[v] = min_k labels[k] + d_w(order[k], v)`` for every
        cell, where ``d_w`` is the shortest walk under the step weight

            ``w(step) = weight_mult * w_raster(step) + length_rate * len(step)``

        ``w_raster`` being the unchanged PYORPS step weight (gradient terms
        included) and ``len`` the step's Euclidean length in cells. All
        labels are float64 and in CELL units, like :meth:`solve`: multiply
        by the cell size for ``Path.total_cost`` units. ``length_rate`` is
        in raster-value units per cell of length, so a per-metre rate R in
        the raster's units is passed as ``R`` unchanged.

        Seeds never enter the heap. They are consumed from the sorted
        stream whenever the next seed's label is at most the heap's top, so
        100 M seeds cost 12 B each (the two input arrays) instead of heap
        entries -- the D1 design point for seeding a drain with a whole
        field.

        Parameters:
            order_arr: uint32 cell of seed ``k``.
            labels_arr: float64 label of seed ``k``; finite, >= 0 and
                non-decreasing in ``k``.
            length_rate: Added per cell of step length, >= 0.
            weight_mult: Scales the raster part of every step, >= 0.
            no_transit: Cells no walk may enter or cross (the turbines of
                the collector DW). They are masked for this drain and
                restored afterwards; a seed ON such a cell still expands.
                Seeds are skipped only on cells the raster itself excludes.
            keep_prev: Record the predecessor forest (``-1`` at seeds). Off
                by default (plan D1), so a drain costs dist + visited only.
            keep_owner: Record which seed reached each cell (its ``k``);
                with ``keep_prev`` it also turns on the deterministic
                tie-break of :meth:`solve`, keyed on the seed's cell.
                On a lean solver, arrays a call does not keep are released
                again, so the solver stays at 9 B/cell.

        Returns:
            The number of cells settled.
        """
        order_in = np.asarray(order_arr).ravel()
        if order_in.size and (np.any(order_in < 0) or np.any(
                order_in >= self.ctx.total_cells)):
            raise IndexError("a seed lies outside the raster")
        cdef np.ndarray[uint32_t, ndim=1] order = np.ascontiguousarray(
            order_in, dtype=np.uint32)
        cdef np.ndarray[float64_t, ndim=1] labels = np.ascontiguousarray(
            labels_arr, dtype=np.float64)
        cdef Py_ssize_t n_seeds = order.shape[0]
        if labels.shape[0] != n_seeds:
            raise ValueError(f"{n_seeds} seeds but {labels.shape[0]} labels")
        if n_seeds == 0:
            raise ValueError("at least one seed is required")
        if not np.all(np.isfinite(labels)) or np.any(labels < 0):
            raise ValueError("seed labels must be finite and >= 0")
        if n_seeds > 1 and not np.all(labels[1:] >= labels[:-1]):
            raise ValueError("seed labels must be non-decreasing; sort the "
                             "stream by label first")
        if not (0.0 <= length_rate < INFINITY
                and 0.0 <= weight_mult < INFINITY):
            raise ValueError("length_rate and weight_mult must be finite "
                             "and >= 0")
        if self.dist.shape[0] != self.ctx.total_cells:
            raise RuntimeError(
                "solve_stream() was called after release(); build a new "
                "solver")
        cdef int total_cells = self.ctx.total_cells
        if n_seeds >= 2147483647:
            raise ValueError("more seeds than an int32 owner label can name")

        cdef np.ndarray[uint32_t, ndim=1] nt_cells = np.empty(0,
                                                              dtype=np.uint32)
        cdef np.ndarray[uint8_t, ndim=1] nt_saved = np.empty(0, dtype=np.uint8)
        cdef Py_ssize_t n_nt = 0
        if no_transit is not None:
            nt_in = np.asarray(no_transit).ravel()
            if nt_in.size and (np.any(nt_in < 0) or np.any(
                    nt_in >= total_cells)):
                raise IndexError("a no-transit cell lies outside the raster")
            nt_cells = np.unique(np.ascontiguousarray(nt_in, dtype=np.uint32))
            n_nt = nt_cells.shape[0]
            nt_saved = np.empty(n_nt, dtype=np.uint8)

        cdef bint track_prev = keep_prev
        cdef bint track_owner = keep_owner
        if track_prev and self.prev.shape[0] != total_cells:
            self.prev = np.full(total_cells, -1, dtype=np.int32)
        if track_owner and self.region.shape[0] != total_cells:
            self.region = np.full(total_cells, -1, dtype=np.int32)
        if self.lean and not track_prev:
            self.prev = np.empty(0, dtype=np.int32)
        if self.lean and not track_owner:
            self.region = np.empty(0, dtype=np.int32)
        cdef bint tie_break = track_prev and track_owner

        self.dist[:] = np.inf
        self.visited[:] = 0
        if self.prev.shape[0] == total_cells:
            self.prev[:] = -1
        if self.region.shape[0] == total_cells:
            self.region[:] = -1
        self._solved = False
        self._terminals_arr = order
        self.region_key = order

        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        cdef vector[StepData] directions = self.ctx.directions
        cdef Py_ssize_t n_dirs = <Py_ssize_t>directions.size()
        cdef np.ndarray[float64_t, ndim=1] step_len = np.empty(
            n_dirs, dtype=np.float64)
        cdef Py_ssize_t di
        for di in range(n_dirs):
            step_len[di] = (<double>directions[di].dr * directions[di].dr
                            + <double>directions[di].dc * directions[di].dc
                            ) ** 0.5

        # Seeds are judged on the raster's own mask, BEFORE the no-transit
        # cells are masked: a seed on a turbine must still expand.
        cdef np.ndarray[uint8_t, ndim=1] seed_ok = np.empty(n_seeds,
                                                            dtype=np.uint8)
        cdef npy_intp sr, sc
        cdef Py_ssize_t k
        for k in range(n_seeds):
            unravel_index(order[k], cols, &sr, &sc)
            seed_ok[k] = 1 if exclude_mask[<int>sr, <int>sc] != 0 else 0


        cdef float64_t[:] dist = self.dist
        cdef uint8_t[:] visited = self.visited
        cdef int32_t[:] prev = self.prev
        cdef int32_t[:] region = self.region

        cdef BinaryHeap pq
        heap_init(&pq)
        cdef Py_ssize_t next_seed = 0
        cdef uint32_t current, neighbor, seed_cell
        cdef double current_dist, lab
        cdef npy_intp current_row, current_col, neighbor_row, neighbor_col
        cdef double intermediate_cost, total_cost, new_dist
        cdef int valid_path, i, dr, dc
        cdef int settled = 0
        cdef bint from_seed, improves
        cdef int32_t cand_region, held_region, incumbent
        cdef bint use_grad = self.use_gradient
        cdef double grad_mult_val, height_diff
        cdef int slope_bin
        cdef npy_intp nr, nc

        # Mask the no-transit cells for this drain only.
        for k in range(n_nt):
            unravel_index(nt_cells[k], cols, &nr, &nc)
            nt_saved[k] = exclude_mask[<int>nr, <int>nc]
            exclude_mask[<int>nr, <int>nc] = 0
        try:
            while True:
                from_seed = False
                if next_seed < n_seeds and (
                        heap_empty(&pq)
                        or labels[next_seed] <= heap_top(&pq).priority):
                    from_seed = True
                elif heap_empty(&pq):
                    break
                if from_seed:
                    k = next_seed
                    next_seed += 1
                    if seed_ok[k] == 0:
                        continue
                    seed_cell = order[k]
                    lab = labels[k]
                    if visited[seed_cell] == 1 or lab > dist[seed_cell]:
                        continue
                    # lab <= dist: the seed wins ties against a relaxed,
                    # not yet settled label, so a cell a seed sits on is
                    # always owned by a seed.
                    dist[seed_cell] = lab
                    if track_prev:
                        prev[seed_cell] = -1
                    if track_owner:
                        region[seed_cell] = <int32_t>k
                    current = seed_cell
                    current_dist = lab
                else:
                    current = heap_top(&pq).index
                    current_dist = heap_top(&pq).priority
                    heap_pop(&pq)
                    if visited[current] == 1 or current_dist > dist[current]:
                        continue
                visited[current] = 1
                settled += 1

                unravel_index(current, cols, &current_row, &current_col)
                for i in range(n_dirs):
                    dr = directions[i].dr
                    dc = directions[i].dc
                    neighbor_row = current_row + dr
                    neighbor_col = current_col + dc
                    if (neighbor_row < 0 or neighbor_row >= rows or
                            neighbor_col < 0 or neighbor_col >= cols):
                        continue
                    if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:
                        continue
                    neighbor = ravel_index(<int>neighbor_row,
                                           <int>neighbor_col, cols)
                    if visited[neighbor] == 1:
                        continue
                    intermediate_cost = 0.0
                    valid_path = check_path(
                        dr, dc, <int>current_row, <int>current_col,
                        exclude_mask, raster, rows, cols, &intermediate_cost)
                    if not valid_path:
                        continue
                    total_cost = (raster[<int>current_row, <int>current_col] +
                                  intermediate_cost +
                                  raster[<int>neighbor_row,
                                         <int>neighbor_col]) * (
                                  directions[i].cost_factor)
                    if use_grad:
                        height_diff = fabs(
                            <double>self.grad_dem[<int>neighbor_row,
                                                  <int>neighbor_col] -
                            <double>self.grad_dem[<int>current_row,
                                                  <int>current_col])
                        slope_bin = <int>(height_diff *
                                          <double>self.grad_bin_factor[i])
                        if slope_bin >= self.grad_n_bins:
                            slope_bin = self.grad_n_bins - 1
                        grad_mult_val = <double>self.grad_mult[slope_bin]
                        if grad_mult_val == INFINITY:
                            continue
                        total_cost = (total_cost * grad_mult_val +
                                      <double>self.grad_add[slope_bin] *
                                      <double>self.grad_step_len[i])
                    total_cost = (weight_mult * total_cost
                                  + length_rate * step_len[i])
                    new_dist = dist[current] + total_cost

                    improves = False
                    if new_dist < dist[neighbor]:
                        improves = True
                    elif tie_break and new_dist == dist[neighbor]:
                        incumbent = prev[neighbor]
                        if incumbent != -1:
                            cand_region = region[current]
                            held_region = region[<uint32_t>incumbent]
                            if cand_region != held_region:
                                improves = (self.region_key[cand_region] <
                                            self.region_key[held_region])
                            else:
                                improves = current < <uint32_t>incumbent
                    if improves:
                        dist[neighbor] = new_dist
                        if track_prev:
                            prev[neighbor] = <int32_t>current
                        if track_owner:
                            region[neighbor] = region[current]
                        heap_push(&pq, neighbor, new_dist)
        finally:
            for k in range(n_nt):
                unravel_index(nt_cells[k], cols, &nr, &nc)
                exclude_mask[<int>nr, <int>nc] = nt_saved[k]

        self._solved = True
        self._has_prev = track_prev
        self._has_owner = track_owner
        self._plain_weights = (weight_mult == 1.0 and length_rate == 0.0)
        return settled

    # ---------- accessors ----------

    @property
    def terminals(self):
        """The terminal index array of the last solve (uint32)."""
        return self._terminals_arr

    @property
    def solved(self):
        return self._solved

    def dist_array(self):
        """Final distance labels in CELL units, ``inf`` where unreached."""
        out = np.asarray(self.dist).copy()
        out[np.asarray(self.visited) == 0] = np.inf
        return out

    def prev_array(self):
        """Predecessor forest (int32, -1 at the roots and where unreached)."""
        return np.asarray(self.prev).copy()

    def region_array(self):
        """Terminal label per cell (int32, -1 unreached)."""
        return np.asarray(self.region).copy()

    def visited_array(self):
        """1 where the cell carries a final label."""
        return np.asarray(self.visited).copy()

    def memory_bytes(self):
        # int64: 17 B/cell overflows a C int above ~126 M cells, which a 1 m
        # raster reaches at 11 km x 11 km.
        cdef int64_t n = <int64_t>self.dist.shape[0]
        if n == 0:
            return 0
        return (n * 9 + <int64_t>self.prev.shape[0] * 4
                + <int64_t>self.region.shape[0] * 4)

    def release(self):
        """Drop the label arrays so the solver no longer pins the window."""
        self.dist = np.empty(0, dtype=np.float64)
        self.prev = np.empty(0, dtype=np.int32)
        self.visited = np.empty(0, dtype=np.uint8)
        self.region = np.empty(0, dtype=np.int32)
        self._solved = False

    def path_to_root(self, uint32_t cell):
        """Cells from the terminal that owns ``cell`` down to ``cell``.

        Empty when the cell was never settled. The result is a valid PYORPS
        route: consecutive entries differ by one entry of the direction table.
        Raises when the last solve kept no predecessors (a lean solver, or
        ``solve_stream(keep_prev=False)``): there is no forest to walk.
        """
        if self.visited.shape[0] == 0 or cell >= <uint32_t>self.visited.shape[0]:
            return np.empty(0, dtype=np.uint32)
        if (not self._has_prev
                or self.prev.shape[0] != self.visited.shape[0]):
            raise RuntimeError(
                "the last solve kept no predecessors; call solve() or "
                "solve_stream(keep_prev=True)")
        if self.visited[cell] == 0:
            return np.empty(0, dtype=np.uint32)

        cdef int length = 1
        cdef uint32_t walk = cell
        while self.prev[walk] != -1:
            walk = <uint32_t>self.prev[walk]
            length += 1

        cdef np.ndarray[uint32_t, ndim=1] out = np.empty(length,
                                                         dtype=np.uint32)
        cdef int idx = length - 1
        walk = cell
        while True:
            out[idx] = walk
            if self.prev[walk] == -1:
                break
            walk = <uint32_t>self.prev[walk]
            idx -= 1
        return out

    # ---------- boundary extraction ----------

    def boundary_steps(self):
        """Every settled step whose two ends belong to different terminals.

        One pass over the settled cells and ``ctx.directions``. For a step
        ``u -> v`` with ``region[u] != region[v]``, both settled and
        ``check_path`` valid, the connection cost is

            ``dist[u] + step_cost(u -> v) + dist[v]``

        which is the cost of the concrete route ``terminal(u) ... u -> v ...
        terminal(v)``, priced by the same arithmetic the search used. Junctions
        can therefore only land on step endpoints: no step is ever split and no
        cost apportioned.

        Retaining the cheapest such step per terminal pair is Mehlhorn's
        terminal distance network [1]; the reduction itself is left to the
        caller so that k-cheapest and Pareto variants stay possible.

        Returns:
            dict of parallel numpy arrays with keys ``region_u``, ``region_v``
            (int32 terminal indices), ``cell_u``, ``cell_v`` (uint32 linear
            indices), ``direction`` (int32 index into ``ctx.directions``) and
            ``cost`` (float64, CELL units). Each undirected boundary step is
            reported ONCE, in the orientation with ``region_u < region_v``.

        References:
            [1] Mehlhorn, K.: 'A faster approximation algorithm for the Steiner
                problem in graphs', Inf. Process. Lett., 1988, 27, (3),
                pp. 125-128
        """
        if not self._solved:
            raise RuntimeError("boundary_steps requires solve() first")
        if (not self._has_prev or not self._has_owner
                or self.region.shape[0] != self.dist.shape[0]):
            raise RuntimeError(
                "boundary_steps needs predecessors and owners; call solve() "
                "or solve_stream(keep_prev=True, keep_owner=True)")
        if not self._plain_weights:
            raise RuntimeError(
                "the last solve used weight_mult/length_rate, but "
                "boundary_steps prices crossings with the raster step cost "
                "alone; the two would not add up")

        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        cdef vector[StepData] directions = self.ctx.directions

        cdef float64_t[:] dist = self.dist
        cdef uint8_t[:] visited = self.visited
        cdef int32_t[:] region = self.region

        cdef vector[int] out_ru
        cdef vector[int] out_rv
        cdef vector[unsigned int] out_cu
        cdef vector[unsigned int] out_cv
        cdef vector[int] out_dir
        cdef vector[double] out_cost

        cdef int r, c, i, dr, dc
        cdef uint32_t current, neighbor
        cdef int neighbor_row, neighbor_col
        cdef int32_t reg_u, reg_v
        cdef double intermediate_cost, step_cost
        cdef int valid_path

        cdef bint use_grad = self.use_gradient
        cdef double grad_mult_val, height_diff
        cdef int slope_bin

        for r in range(rows):
            for c in range(cols):
                current = ravel_index(r, c, cols)
                if visited[current] == 0:
                    continue
                reg_u = region[current]
                if reg_u < 0:
                    continue

                for i in range(directions.size()):
                    dr = directions[i].dr
                    dc = directions[i].dc
                    neighbor_row = r + dr
                    neighbor_col = c + dc

                    if (neighbor_row < 0 or neighbor_row >= rows or
                            neighbor_col < 0 or neighbor_col >= cols):
                        continue
                    if exclude_mask[neighbor_row, neighbor_col] == 0:
                        continue

                    neighbor = ravel_index(neighbor_row, neighbor_col, cols)
                    if visited[neighbor] == 0:
                        continue
                    reg_v = region[neighbor]
                    # Report each undirected boundary step once. The step
                    # weight is symmetric (the raster endpoints, the
                    # intermediate set, the cost factor and |dh| all are), so
                    # the reverse orientation carries no extra information.
                    if reg_v <= reg_u:
                        continue

                    intermediate_cost = 0.0
                    valid_path = check_path(
                        dr, dc, r, c, exclude_mask, raster, rows, cols,
                        &intermediate_cost
                    )
                    if not valid_path:
                        continue

                    step_cost = (raster[r, c] + intermediate_cost +
                                 raster[neighbor_row, neighbor_col]) * (
                                 directions[i].cost_factor)

                    if use_grad:
                        height_diff = fabs(
                            <double>self.grad_dem[neighbor_row, neighbor_col] -
                            <double>self.grad_dem[r, c])
                        slope_bin = <int>(height_diff *
                                          <double>self.grad_bin_factor[i])
                        if slope_bin >= self.grad_n_bins:
                            slope_bin = self.grad_n_bins - 1
                        grad_mult_val = <double>self.grad_mult[slope_bin]
                        if grad_mult_val == INFINITY:
                            continue
                        step_cost = (step_cost * grad_mult_val +
                                     <double>self.grad_add[slope_bin] *
                                     <double>self.grad_step_len[i])

                    out_ru.push_back(<int>reg_u)
                    out_rv.push_back(<int>reg_v)
                    out_cu.push_back(<unsigned int>current)
                    out_cv.push_back(<unsigned int>neighbor)
                    out_dir.push_back(i)
                    out_cost.push_back(dist[current] + step_cost +
                                       dist[neighbor])

        cdef Py_ssize_t n = <Py_ssize_t>out_ru.size()
        cdef np.ndarray[int32_t, ndim=1] a_ru = np.empty(n, dtype=np.int32)
        cdef np.ndarray[int32_t, ndim=1] a_rv = np.empty(n, dtype=np.int32)
        cdef np.ndarray[uint32_t, ndim=1] a_cu = np.empty(n, dtype=np.uint32)
        cdef np.ndarray[uint32_t, ndim=1] a_cv = np.empty(n, dtype=np.uint32)
        cdef np.ndarray[int32_t, ndim=1] a_dir = np.empty(n, dtype=np.int32)
        cdef np.ndarray[float64_t, ndim=1] a_cost = np.empty(
            n, dtype=np.float64)
        cdef Py_ssize_t j
        for j in range(n):
            a_ru[j] = <int32_t>out_ru[j]
            a_rv[j] = <int32_t>out_rv[j]
            a_cu[j] = <uint32_t>out_cu[j]
            a_cv[j] = <uint32_t>out_cv[j]
            a_dir[j] = <int32_t>out_dir[j]
            a_cost[j] = out_cost[j]

        # Sorted by (cost, lower cell, upper cell). The scan itself emits rows
        # in an order that depends on which endpoint holds the smaller region
        # LABEL, and a label is the terminal's position in the argument array
        # -- so permuting the terminals reshuffles these rows even though the
        # candidate SET is identical. A caller that keeps "the k cheapest per
        # pair" would then pick a different route among equal-cost candidates,
        # and cost ties are pervasive; ordering on cell indices instead makes
        # the whole graph, not only the partition, independent of the order
        # the terminals were listed in.
        order = np.lexsort((np.maximum(a_cu, a_cv), np.minimum(a_cu, a_cv),
                            a_cost))
        return {
            "region_u": a_ru[order],
            "region_v": a_rv[order],
            "cell_u": a_cu[order],
            "cell_v": a_cv[order],
            "direction": a_dir[order],
            "cost": a_cost[order],
        }

    def price_route(self, cells):
        """Price an arbitrary cell sequence with the kernel's own arithmetic.

        The referee for "does this corridor segment cost what the search
        charged for it". Consecutive cells must differ by an entry of the
        direction table; anything else is a step the kernel could not have
        taken and raises rather than being priced by a made-up rule.

        Parameters:
            cells: 1D array-like of linear cell indices, in route order.

        Returns:
            The route weight in CELL units. ``0.0`` for a route of one cell.

        Raises:
            ValueError: a consecutive pair is not a legal step, or the step is
                blocked (excluded cell, invalid intermediate, grade limit).
        """
        cdef np.ndarray[uint32_t, ndim=1] arr = np.ascontiguousarray(
            cells, dtype=np.uint32)
        cdef Py_ssize_t n = arr.shape[0]
        if n < 2:
            return 0.0
        # Validate the whole route up front, so an out-of-window cell fails
        # loudly here instead of being priced from adjacent heap. This is the
        # documented referee for anything that slices a route apart; pricing a
        # route whose indices came from a LARGER window is the realistic way
        # to reach it.
        cdef object bad = np.flatnonzero(arr >= <uint32_t>self.ctx.total_cells)
        if bad.size:
            raise IndexError(
                f"cells[{int(bad[0])}] = {int(arr[int(bad[0])])} is outside "
                f"the raster ({self.ctx.total_cells} cells); the route was "
                f"probably computed against a different search window")

        cdef vector[StepData] directions = self.ctx.directions
        cdef int n_dirs = <int>directions.size()
        cdef int cols = self.ctx.cols
        cdef uint32_t[:] view = arr

        cdef double total = 0.0
        cdef Py_ssize_t k
        cdef npy_intp r0, c0, r1, c1
        cdef int dr, dc, i, found
        cdef double one

        for k in range(n - 1):
            unravel_index(view[k], cols, &r0, &c0)
            unravel_index(view[k + 1], cols, &r1, &c1)
            dr = <int>(r1 - r0)
            dc = <int>(c1 - c0)
            found = -1
            for i in range(n_dirs):
                if directions[i].dr == dr and directions[i].dc == dc:
                    found = i
                    break
            if found < 0:
                raise ValueError(
                    f"cells[{k}] -> cells[{k + 1}] is the step ({dr}, {dc}), "
                    f"which is not in the neighbourhood step table")
            one = self.step_cost(view[k], found)
            if one == INFINITY:
                raise ValueError(
                    f"cells[{k}] -> cells[{k + 1}] is blocked: excluded cell, "
                    f"invalid intermediate or grade limit")
            total += one
        return total

    def step_cost(self, uint32_t cell_u, int direction_index):
        """Weight of one step, priced by the arithmetic the search uses.

        Exposed so callers reconcile segment costs against the search instead
        of reimplementing the step rule. Returns ``inf`` when the step leaves
        the raster, hits an excluded cell or is forbidden by the grade limit.
        """
        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef vector[StepData] directions = self.ctx.directions
        if direction_index < 0 or direction_index >= <int>directions.size():
            raise IndexError("direction_index out of range")

        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        # The SOURCE cell must be range-checked before it is unravelled. With
        # boundscheck off, an index past the end yields a row past the end,
        # and a step with a negative enough dr pulls the NEIGHBOUR back into
        # range -- so the neighbour check below passes and both memoryviews
        # are read |dr| * cols elements past their buffers. Reachable from
        # price_route_cython whenever a caller prices a route whose indices
        # were computed against a larger window than the raster passed here.
        if cell_u >= <uint32_t>self.ctx.total_cells:
            raise IndexError(
                f"cell index {cell_u} is outside the raster "
                f"({self.ctx.total_cells} cells)")
        cdef npy_intp r, c
        unravel_index(cell_u, cols, &r, &c)

        cdef int dr = directions[direction_index].dr
        cdef int dc = directions[direction_index].dc
        cdef int nr = <int>r + dr
        cdef int nc = <int>c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            return float("inf")
        if exclude_mask[nr, nc] == 0 or exclude_mask[<int>r, <int>c] == 0:
            return float("inf")

        cdef double intermediate_cost = 0.0
        cdef int valid_path = check_path(
            dr, dc, <int>r, <int>c, exclude_mask, raster, rows, cols,
            &intermediate_cost)
        if not valid_path:
            return float("inf")

        cdef double cost = (raster[<int>r, <int>c] + intermediate_cost +
                            raster[nr, nc]) * (
                            directions[direction_index].cost_factor)

        cdef double height_diff, grad_mult_val
        cdef int slope_bin
        if self.use_gradient:
            height_diff = fabs(<double>self.grad_dem[nr, nc] -
                               <double>self.grad_dem[<int>r, <int>c])
            slope_bin = <int>(height_diff *
                              <double>self.grad_bin_factor[direction_index])
            if slope_bin >= self.grad_n_bins:
                slope_bin = self.grad_n_bins - 1
            grad_mult_val = <double>self.grad_mult[slope_bin]
            if grad_mult_val == INFINITY:
                return float("inf")
            cost = (cost * grad_mult_val +
                    <double>self.grad_add[slope_bin] *
                    <double>self.grad_step_len[direction_index])
        return cost

# ==================== PUBLIC API WRAPPERS ====================
# These maintain the exact same signatures as path_algorithms.pyx
# for backward compatibility.

cdef _apply_gradient_kwargs(solver, gradient_luts, dem):
    """Configure a DijkstraSolver from a GradientLUTs object (or no-op)."""
    if gradient_luts is None or dem is None:
        return
    solver.set_gradient(
        np.ascontiguousarray(dem, dtype=np.float32),
        np.ascontiguousarray(gradient_luts.mult, dtype=np.float32),
        np.ascontiguousarray(gradient_luts.add, dtype=np.float32),
        np.ascontiguousarray(gradient_luts.bin_factor, dtype=np.float32),
        np.ascontiguousarray(gradient_luts.step_len_cells, dtype=np.float32),
        int(gradient_luts.n_bins),
    )


def make_dijkstra_solver(np.ndarray[uint16_t, ndim=2] raster_arr,
                         np.ndarray[int8_t, ndim=2] steps_arr,
                         int64_t max_value=65535,
                         dem=None, gradient_luts=None):
    """
    Build a DijkstraSolver bound to one raster and step set.

    Exposed so callers that issue many queries against the same raster pay
    the RasterContext build (an O(cells) exclude-mask scan plus the system
    limits probe) and the dist/prev/visited allocation once instead of once
    per query. Reuse is exact: every query entry point resets those arrays
    before it runs, so a solver carries no state between queries.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        max_value: Cost value representing obstacles (default 65535)
        dem: Optional float32 DEM aligned to raster_arr (same shape)
        gradient_luts: Optional GradientLUTs enabling per-edge gradient terms

    Returns:
        A DijkstraSolver ready for single_pair, single_source_multi_target,
        multi_source_multi_target and some_pairs queries.
    """
    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = DijkstraSolver(ctx)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
    return solver


def make_multi_source_solver(np.ndarray[uint16_t, ndim=2] raster_arr,
                             np.ndarray[int8_t, ndim=2] steps_arr,
                             int64_t max_value=65535,
                             dem=None, gradient_luts=None, bint lean=False):
    """
    Build a MultiSourceSolver bound to one raster and step set.

    Mirrors :func:`make_dijkstra_solver` so a caller can cache the
    RasterContext build and the label allocation exactly as the pairwise
    path does; the corridor solver adds one int32 array per cell on top of
    the 13 B/cell the Dijkstra solver already owns.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        max_value: Cost value representing obstacles (default 65535)
        dem: Optional float32 DEM aligned to raster_arr (same shape)
        gradient_luts: Optional GradientLUTs enabling per-edge gradient terms
        lean: Allocate only the distance and visited arrays (9 B/cell);
            for solve_stream(keep_prev=False, keep_owner=False) drains.

    Returns:
        A MultiSourceSolver ready for solve() / boundary_steps().
    """
    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = MultiSourceSolver(ctx, lean)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
    return solver


def price_route_cython(np.ndarray[uint16_t, ndim=2] raster_arr,
                       np.ndarray[int8_t, ndim=2] steps_arr,
                       cells,
                       int64_t max_value=65535,
                       dem=None, gradient_luts=None):
    """
    Price a cell sequence with the same arithmetic the search uses.

    Public API wrapper around :meth:`MultiSourceSolver.price_route`. Exists so
    that corridor segments, and anything else that slices a route apart, are
    reconciled against the kernel instead of against a reimplementation of the
    step rule.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        cells: 1D array-like of linear cell indices, in route order
        max_value: Cost value representing obstacles (default 65535)
        dem: Optional float32 DEM aligned to raster_arr (same shape)
        gradient_luts: Optional GradientLUTs enabling per-edge gradient terms

    Returns:
        The route weight in CELL units.
    """
    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = MultiSourceSolver(ctx)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
    return solver.price_route(cells)


def dijkstra_2d_cython(np.ndarray[uint16_t, ndim=2] raster_arr,
                       np.ndarray[int8_t, ndim=2] steps_arr,
                       uint32_t source_idx, uint32_t target_idx,
                       int64_t max_value=65535,
                       dem=None, gradient_luts=None):
    """
    Find shortest path between two points in a 2D raster using Dijkstra.

    Public API wrapper that creates a RasterContext and DijkstraSolver
    then delegates to single_pair.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_idx: Linear index of starting cell
        target_idx: Linear index of destination cell
        max_value: Cost value representing obstacles (default 65535)
        dem: Optional float32 DEM aligned to raster_arr (same shape)
        gradient_luts: Optional pyorps.core.objective.GradientLUTs enabling
            per-edge gradient terms (feasibility plan section 3.2)

    Returns:
        1D numpy array (uint32) of linear indices of cells in the
        optimal path from source to target. Empty array if no path exists.
    """
    # Early return if source equals target
    if source_idx == target_idx:
        return np.array([source_idx], dtype=np.uint32)

    solver = make_dijkstra_solver(raster_arr, steps_arr, max_value,
                                  dem, gradient_luts)
    return solver.single_pair(source_idx, target_idx)


def dijkstra_single_source_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        uint32_t source_idx,
        np.ndarray[uint32_t, ndim=1] target_indices,
        int64_t max_value=65535,
        dem=None, gradient_luts=None):
    """
    Find optimal paths from one source to multiple targets efficiently.

    Public API wrapper.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_idx: Linear index of the single starting cell
        target_indices: 1D numpy array (uint32) of target cell indices
        max_value: Cost value representing obstacles (default 65535)
        dem: Optional float32 DEM aligned to raster_arr (same shape)
        gradient_luts: Optional GradientLUTs enabling per-edge gradient terms

    Returns:
        List of numpy arrays, one per target. Empty arrays for unreachable.
    """
    solver = make_dijkstra_solver(raster_arr, steps_arr, max_value,
                                  dem, gradient_luts)
    return solver.single_source_multi_target(source_idx, target_indices)


def dijkstra_multiple_sources_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint32_t, ndim=1] source_indices,
        np.ndarray[uint32_t, ndim=1] target_indices,
        int64_t max_value=65535, bint return_paths=True,
        dem=None, gradient_luts=None):
    """
    Compute all-pairs shortest paths between multiple sources and targets.

    Public API wrapper.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_indices: 1D array (uint32) of all source cell indices
        target_indices: 1D array (uint32) of all target cell indices
        max_value: Cost value representing obstacles (default 65535)
        return_paths: If True, returns paths; if False, returns cost matrix

    Returns:
        If return_paths=True: List of lists, paths[i][j] = path from source i to target j
        If return_paths=False: 2D cost matrix with distances
    """
    solver = make_dijkstra_solver(raster_arr, steps_arr, max_value,
                                  dem, gradient_luts)
    return solver.multi_source_multi_target(
        source_indices, target_indices, return_paths)


def dijkstra_some_pairs_shortest_paths(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint32_t, ndim=1] source_indices,
        np.ndarray[uint32_t, ndim=1] target_indices,
        int64_t max_value=65535,
        bint return_paths=True,
        dem=None, gradient_luts=None):
    """
    Find optimal paths for specific source-target pairs using batch optimization.

    Public API wrapper.

    Parameters:
        raster_arr: 2D numpy array (uint16) containing cell traversal costs
        steps_arr: 2D numpy array (int8) defining movement directions
        source_indices: 1D array (uint32) of source cell indices
        target_indices: 1D array (uint32) of target cell indices
        max_value: Cost value representing obstacles (default 65535)
        return_paths: If True, returns actual paths; if False, returns costs

    Returns:
        If return_paths=True: List of path arrays
        If return_paths=False: 1D array of path costs (inf for no path)
    """
    solver = make_dijkstra_solver(raster_arr, steps_arr, max_value,
                                  dem, gradient_luts)
    return solver.some_pairs(
        source_indices, target_indices, return_paths)

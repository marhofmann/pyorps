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

    def __cinit__(self, RasterContext ctx):
        self.ctx = ctx
        cdef int n = ctx.total_cells
        self.dist = np.full(n, np.inf, dtype=np.float64)
        self.prev = np.full(n, -1, dtype=np.int32)
        self.visited = np.zeros(n, dtype=np.uint8)
        self.use_gradient = False
        self.grad_n_bins = 0

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

    def single_pair(self, uint32_t source_idx, uint32_t target_idx):
        """
        Find shortest path between two points in the raster.

        Logic from _dijkstra_2d_cython_internal. Uses early termination
        when target is reached.

        Parameters:
            source_idx: Linear index of starting cell
            target_idx: Linear index of destination cell

        Returns:
            1D numpy array (uint32) of linear cell indices forming the
            optimal path. Empty array if no path exists.
        """
        # Early return if source equals target
        if source_idx == target_idx:
            return np.array([source_idx], dtype=np.uint32)

        self._reset()

        cdef int rows = self.ctx.rows
        cdef int cols = self.ctx.cols
        cdef uint16_t[:, :] raster = self.ctx.raster_view
        cdef uint8_t[:, :] exclude_mask = self.ctx.exclude_mask_view
        cdef vector[StepData] directions = self.ctx.directions

        cdef float64_t[:] dist = self.dist
        cdef int32_t[:] prev = self.prev
        cdef uint8_t[:] visited = self.visited

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

                # Update shortest path if better route found
                new_dist = dist[current] + total_cost
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heap_push(&pq, neighbor, new_dist)

        # Reconstruct path
        return self._reconstruct_path(source_idx, target_idx)

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

    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = DijkstraSolver(ctx)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
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
    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = DijkstraSolver(ctx)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
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
    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = DijkstraSolver(ctx)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
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
    ctx = RasterContext(raster_arr, steps_arr, max_value)
    solver = DijkstraSolver(ctx)
    _apply_gradient_kwargs(solver, gradient_luts, dem)
    return solver.some_pairs(
        source_indices, target_indices, return_paths)

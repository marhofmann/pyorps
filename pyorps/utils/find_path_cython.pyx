# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, nonecheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_compile_args=["/wd4551"]

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, floor, ceil, abs
from libcpp.vector cimport vector
from libcpp cimport bool

# Import Numpy C-API
np.import_array()

# Type definitions
ctypedef np.int8_t int8_t
ctypedef np.uint8_t uint8_t
ctypedef np.uint16_t uint16_t
ctypedef np.uint32_t uint32_t
ctypedef np.int32_t int32_t
ctypedef np.float64_t float64_t
# Add npy_intp type for NumPy array indexing
ctypedef Py_ssize_t npy_intp

# Structure to hold intermediate steps
cdef struct IntermediatePoint:
    int8_t dr
    int8_t dc

# Structure to hold step data
cdef struct StepData:
    int dr
    int dc
    double cost_factor

# Function to calculate intermediate steps between source and target cells
cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) noexcept nogil:
    # Function implementation from original code
    cdef vector[IntermediatePoint] result
    cdef IntermediatePoint point
    cdef int abs_dr = abs(dr)
    cdef int abs_dc = abs(dc)
    cdef int sum_abs = abs_dr + abs_dc
    cdef int k, p
    cdef double dr_k, dc_k, ddr, ddc, dk, dp  # Changed from float to double
    cdef int8_t floor_dr, floor_dc, ceil_dr, ceil_dc

    if sum_abs <= 1:
        # No intermediate steps needed
        pass
    elif max(abs_dr, abs_dc) == 1:
        # Special case for diagonal cells
        point.dr = <int8_t>dr
        point.dc = 0
        result.push_back(point)

        point.dr = 0
        point.dc = <int8_t>dc
        result.push_back(point)
    else:
        # General case
        k = max(abs_dr, abs_dc)
        ddr = <double>dr
        ddc = <double>dc
        dk = <double>k
        for p in range(1, k):
            dp = <double>p
            # Use double for calculations
            dr_k = (dp * ddr) / dk
            dc_k = (dp * ddc) / dk

            # Add floor values
            floor_dr = <int8_t>floor(dr_k)
            floor_dc = <int8_t>floor(dc_k)
            point.dr = floor_dr
            point.dc = floor_dc
            result.push_back(point)

            # Add ceil values (only if different from floor)
            ceil_dr = <int8_t>ceil(dr_k)
            ceil_dc = <int8_t>ceil(dc_k)
            if floor_dr != ceil_dr or floor_dc != ceil_dc:
                point.dr = ceil_dr
                point.dc = ceil_dc
                result.push_back(point)
    return result

# Calculate cost factor - changed to return double
cdef inline double _get_cost_factor_cython(int dr, int dc, int intermediates_count) noexcept nogil:
    cdef double distance = sqrt(<double>(dr * dr + dc * dc))
    cdef double divisor = 2.0 + <double>intermediates_count
    return distance / divisor

# Node structure for priority queue
cdef struct PQNode:
    uint32_t index
    double priority

# Priority queue implementation
cdef struct BinaryHeap:
    vector[PQNode] nodes

cdef inline int heap_init(BinaryHeap* heap) nogil:
    heap.nodes.clear()
    heap.nodes.reserve(1000)
    return 0

cdef inline bool heap_empty(const BinaryHeap* heap) noexcept nogil:
    return heap.nodes.size() == 0

cdef inline PQNode heap_top(const BinaryHeap* heap) noexcept nogil:
    return heap.nodes[0]

cdef inline int heap_push(BinaryHeap* heap, uint32_t idx, double priority) nogil:
    cdef PQNode node
    node.index = idx
    node.priority = priority
    heap.nodes.push_back(node)

    # Bubble up
    cdef npy_intp pos = heap.nodes.size() - 1  # Changed from size_t to npy_intp
    cdef npy_intp parent  # Changed from size_t to npy_intp
    cdef PQNode temp

    while pos > 0:
        parent = (pos - 1) // 2
        if heap.nodes[parent].priority <= heap.nodes[pos].priority:
            break
        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[parent]
        heap.nodes[parent] = temp
        pos = parent

    return 0

cdef inline int heap_pop(BinaryHeap* heap) nogil:
    if heap.nodes.size() == 0:
        return 1  # Error: empty heap

    if heap.nodes.size() > 1:
        heap.nodes[0] = heap.nodes[heap.nodes.size() - 1]

    heap.nodes.pop_back()

    if heap.nodes.size() <= 1:
        return 0

    # Sift down
    cdef npy_intp pos = 0  # Changed from size_t to npy_intp
    cdef npy_intp left, right, smallest  # Changed from size_t to npy_intp
    cdef npy_intp heap_size = heap.nodes.size()  # Changed from size_t to npy_intp
    cdef PQNode temp

    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos

        if left < heap_size and heap.nodes[left].priority < heap.nodes[smallest].priority:
            smallest = left

        if right < heap_size and heap.nodes[right].priority < heap.nodes[smallest].priority:
            smallest = right

        if smallest == pos:
            break

        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[smallest]
        heap.nodes[smallest] = temp
        pos = smallest

    return 0

# Fast index conversion functions
cdef inline uint32_t ravel_index(int row, int col, int cols) noexcept nogil:
    return <uint32_t>(row * cols + col)

# Changed parameter types to use npy_intp for row and col pointers
cdef inline int unravel_index(uint32_t idx, int cols, npy_intp* row, npy_intp* col) nogil:
    row[0] = idx // cols
    col[0] = idx % cols
    return 0

# Create exclude mask efficiently
def create_exclude_mask(np.ndarray[uint16_t, ndim=2] raster_arr, uint16_t max_value):
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef uint16_t[:, :] raster = raster_arr

    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr = np.ones((rows, cols), dtype=np.uint8)
    cdef uint8_t[:, :] exclude_mask = exclude_mask_arr

    cdef int i, j
    for i in range(rows):
        for j in range(cols):
            if raster[i, j] == max_value:
                exclude_mask[i, j] = 0

    return exclude_mask_arr

# Check intermediate cells and calculate cost
cdef int check_path(int dr, int dc, int current_row, int current_col,
                  const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                  int rows, int cols, double* total_cost) nogil:
    cdef double cost = 0.0
    cdef int i, int_row, int_col

    # Get intermediate steps
    cdef vector[IntermediatePoint] intermediates = _calculate_intermediate_steps_cython(dr, dc)
    cdef IntermediatePoint point

    # Check each intermediate point
    for i in range(intermediates.size()):
        point = intermediates[i]
        int_row = current_row + point.dr
        int_col = current_col + point.dc

        # Check if valid
        if (int_row < 0 or int_row >= rows or
                int_col < 0 or int_col >= cols or
                exclude_mask[int_row, int_col] == 0):
            return 0

        # Add cost
        cost += raster[int_row, int_col]

    # Return the cost
    total_cost[0] = cost
    return 1

# Check intermediate cells and calculate cost using plain arrays (for nogil)
cdef int check_path_nogil(int dr, int dc, int current_row, int current_col,
                          const uint8_t* exclude_mask, const uint16_t* raster,
                          int rows, int cols, double* total_cost, int stride) nogil:
    cdef double cost = 0.0
    cdef int i, int_row, int_col

    # Get intermediate steps
    cdef vector[IntermediatePoint] intermediates = _calculate_intermediate_steps_cython(dr, dc)
    cdef IntermediatePoint point

    # Check each intermediate point
    for i in range(intermediates.size()):
        point = intermediates[i]
        int_row = current_row + point.dr
        int_col = current_col + point.dc

        # Check if valid
        if (int_row < 0 or int_row >= rows or
                int_col < 0 or int_col >= cols or
                exclude_mask[int_row * stride + int_col] == 0):
            return 0

        # Add cost
        cost += raster[int_row * stride + int_col]

    # Return the cost
    total_cost[0] = cost
    return 1

# Precompute directions
cdef vector[StepData] precompute_directions(np.ndarray[int8_t, ndim=2] steps_arr):
    cdef vector[StepData] directions
    cdef StepData direction
    cdef int s, dr, dc
    cdef int intermediates_count
    cdef int steps_count = <int>steps_arr.shape[0]

    directions.reserve(steps_count)

    for s in range(steps_count):
        dr = steps_arr[s, 0]
        dc = steps_arr[s, 1]

        # Count intermediates
        intermediates_count = <int>_calculate_intermediate_steps_cython(dr, dc).size()

        # Store direction data
        direction.dr = dr
        direction.dc = dc
        direction.cost_factor = _get_cost_factor_cython(dr, dc, intermediates_count)

        directions.push_back(direction)

    return directions

# Internal Dijkstra implementation for reuse
cdef np.ndarray[uint32_t, ndim=1] _dijkstra_2d_cython_internal(
    uint16_t[:, :] raster,
    uint8_t[:, :] exclude_mask,
    vector[StepData] directions,
    uint32_t source_idx,
    uint32_t target_idx,
    int rows,
    int cols):

    cdef int total_cells = rows * cols

    # Create numpy arrays
    dist_arr = np.full(total_cells, np.inf, dtype=np.float64)
    prev_arr = np.full(total_cells, -1, dtype=np.int32)
    visited_arr = np.zeros(total_cells, dtype=np.uint8)

    cdef float64_t[:] dist = dist_arr
    cdef int32_t[:] prev = prev_arr
    cdef uint8_t[:] visited = visited_arr

    # Initialize priority queue
    cdef BinaryHeap pq
    heap_init(&pq)

    # Set source distance and add to queue
    dist[source_idx] = 0.0
    heap_push(&pq, source_idx, 0.0)

    # Variables for main loop
    cdef PQNode top_node
    cdef uint32_t current
    cdef double current_dist
    cdef npy_intp current_row, current_col  # Changed from int to npy_intp
    cdef npy_intp neighbor_row, neighbor_col  # Changed from int to npy_intp
    cdef uint32_t neighbor
    cdef double intermediate_cost = 0.0
    cdef double total_cost, new_dist
    cdef int valid_path
    cdef int i, dr, dc

    # Main Dijkstra loop
    while not heap_empty(&pq):
        top_node = heap_top(&pq)
        heap_pop(&pq)

        current = top_node.index
        current_dist = top_node.priority

        # Skip if already processed or better path found
        if visited[current] == 1 or current_dist > dist[current]:
            continue

        # Mark as visited
        visited[current] = 1

        # Early termination
        if current == target_idx:
            break

        # Get coordinates
        unravel_index(current, cols, &current_row, &current_col)

        # Process each direction
        for i in range(directions.size()):
            # Get direction
            dr = directions[i].dr
            dc = directions[i].dc

            # Calculate neighbor
            neighbor_row = current_row + dr
            neighbor_col = current_col + dc

            # Check bounds
            if (neighbor_row < 0 or neighbor_row >= rows or
                    neighbor_col < 0 or neighbor_col >= cols):
                continue

            # Check if excluded
            if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:  # Cast to int for indexing
                continue

            # Calculate neighbor index
            neighbor = ravel_index(<int>neighbor_row, <int>neighbor_col, cols)  # Cast to int for function call

            # Skip if visited
            if visited[neighbor] == 1:
                continue

            # Check intermediate cells
            intermediate_cost = 0.0
            valid_path = check_path(
                dr, dc, <int>current_row, <int>current_col,  # Cast to int for function call
                exclude_mask, raster, rows, cols, &intermediate_cost
            )

            if not valid_path:
                continue

            # Calculate total cost
            total_cost = (raster[<int>current_row, <int>current_col] + intermediate_cost +  # Cast to int for indexing
                        raster[<int>neighbor_row, <int>neighbor_col]) * directions[i].cost_factor  # Cast to int for indexing

            # Update if better path found
            new_dist = dist[current] + total_cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current

                # Add to queue
                heap_push(&pq, neighbor, new_dist)

    # Path reconstruction
    if prev_arr[target_idx] == -1:
        return np.empty(0, dtype=np.uint32)  # No path found

    # Count path length
    path_length = 1
    current = target_idx
    while current != source_idx:
        current = prev_arr[current]
        path_length += 1

    # Create path array
    path = np.empty(path_length, dtype=np.uint32)

    # Fill path
    current = target_idx
    idx = path_length - 1

    while True:
        path[idx] = current
        if current == source_idx:
            break
        current = prev_arr[current]
        idx -= 1

    return path

# Original Dijkstra implementation (kept for reference)
def dijkstra_2d_cython(np.ndarray[uint16_t, ndim=2] raster_arr,
                      np.ndarray[int8_t, ndim=2] steps_arr,
                      uint32_t source_idx,
                      uint32_t target_idx,
                      uint16_t max_value=65535):
    """
    Original Dijkstra implementation (single source to single target)
    """
    # Get dimensions
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]

    # Create exclude mask
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr = create_exclude_mask(raster_arr, max_value)

    # Precompute direction data
    cdef vector[StepData] directions = precompute_directions(steps_arr)

    # Call the internal implementation
    return _dijkstra_2d_cython_internal(
        raster_arr,
        exclude_mask_arr,
        directions,
        source_idx,
        target_idx,
        rows,
        cols
    )

# APPROACH 1: Single source to multiple targets
def dijkstra_single_source_multiple_targets(
    np.ndarray[uint16_t, ndim=2] raster_arr,
    np.ndarray[int8_t, ndim=2] steps_arr,
    uint32_t source_idx,
    np.ndarray[uint32_t, ndim=1] target_indices,
    uint16_t max_value=65535):
    """
    Finds optimal paths from a single source to multiple targets.

    Parameters:
    -----------
    raster_arr : 2D numpy array (uint16)
        Cost raster. Values of max_value are treated as obstacles.
    steps_arr : 2D numpy array (int8)
        Movement directions, e.g., [[0,1], [1,0], ...].
    source_idx : uint32
        Linear index of the starting cell.
    target_indices : 1D numpy array (uint32)
        Linear indices of the target cells.
    max_value : uint16, default=65535
        Value representing obstacles/barriers in the raster.

    Returns:
    --------
    list of numpy arrays
        Each array contains the path from source to the corresponding target.
        Empty arrays indicate no path found.
    """
    # Get dimensions
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef int total_cells = rows * cols
    cdef int steps_count = <int>steps_arr.shape[0]
    cdef int num_targets = <int>target_indices.shape[0]

    # Get memory views
    cdef uint16_t[:, :] raster = raster_arr
    cdef int8_t[:, :] steps = steps_arr
    cdef uint32_t[:] targets = target_indices

    # Create exclude mask
    cdef np.ndarray[uint8_t, ndim=2] exclude_mask_arr = create_exclude_mask(raster_arr, max_value)
    cdef uint8_t[:, :] exclude_mask = exclude_mask_arr

    # Precompute direction data
    cdef vector[StepData] directions = precompute_directions(steps_arr)

    # Initialize Dijkstra arrays
    cdef np.ndarray[float64_t, ndim=1] dist_arr = np.full(total_cells, np.inf, dtype=np.float64)
    cdef np.ndarray[int32_t, ndim=1] prev_arr = np.full(total_cells, -1, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] visited_arr = np.zeros(total_cells, dtype=np.uint8)

    # Track found targets
    cdef np.ndarray[uint8_t, ndim=1] target_found_arr = np.zeros(num_targets, dtype=np.uint8)
    cdef uint8_t[:] target_found = target_found_arr
    cdef int targets_remaining = num_targets
    cdef int t

    cdef float64_t[:] dist = dist_arr
    cdef int32_t[:] prev = prev_arr
    cdef uint8_t[:] visited = visited_arr

    # Initialize priority queue
    cdef BinaryHeap pq
    heap_init(&pq)

    # Set source distance and add to queue
    dist[source_idx] = 0.0
    heap_push(&pq, source_idx, 0.0)

    # Variables for main loop
    cdef PQNode top_node
    cdef uint32_t current
    cdef double current_dist
    cdef npy_intp current_row, current_col  # Changed from int to npy_intp
    cdef npy_intp neighbor_row, neighbor_col  # Changed from int to npy_intp
    cdef uint32_t neighbor
    cdef double intermediate_cost = 0.0
    cdef double total_cost, new_dist
    cdef int valid_path
    cdef int i, dr, dc

    # Main Dijkstra loop - continue until all targets found or queue is empty
    while not heap_empty(&pq) and targets_remaining > 0:
        # Get minimum cost node
        top_node = heap_top(&pq)
        heap_pop(&pq)

        current = top_node.index
        current_dist = top_node.priority

        # Skip if already processed or better path found
        if visited[current] == 1 or current_dist > dist[current]:
            continue

        # Mark as visited
        visited[current] = 1

        # Check if this is a target
        for t in range(num_targets):
            if current == targets[t] and target_found[t] == 0:
                target_found[t] = 1
                targets_remaining -= 1

        # Get coordinates
        unravel_index(current, cols, &current_row, &current_col)

        # Process each direction
        for i in range(directions.size()):
            # Get direction
            dr = directions[i].dr
            dc = directions[i].dc

            # Calculate neighbor
            neighbor_row = current_row + dr
            neighbor_col = current_col + dc

            # Check bounds
            if (neighbor_row < 0 or neighbor_row >= rows or
                    neighbor_col < 0 or neighbor_col >= cols):
                continue

            # Check if excluded
            if exclude_mask[<int>neighbor_row, <int>neighbor_col] == 0:  # Cast to int for indexing
                continue

            # Calculate neighbor index
            neighbor = ravel_index(<int>neighbor_row, <int>neighbor_col, cols)  # Cast to int for function call

            # Skip if visited
            if visited[neighbor] == 1:
                continue

            # Check intermediate cells
            intermediate_cost = 0.0
            valid_path = check_path(
                dr, dc, <int>current_row, <int>current_col,  # Cast to int for function call
                exclude_mask, raster, rows, cols, &intermediate_cost
            )

            if not valid_path:
                continue

            # Calculate total cost
            total_cost = (raster[<int>current_row, <int>current_col] + intermediate_cost +  # Cast to int for indexing
                        raster[<int>neighbor_row, <int>neighbor_col]) * directions[i].cost_factor  # Cast to int for indexing

            # Update if better path found
            new_dist = dist[current] + total_cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current

                # Add to queue
                heap_push(&pq, neighbor, new_dist)

    # Reconstruct paths
    paths = []
    for t in range(num_targets):
        target_idx = targets[t]

        # Check if path exists
        if prev[target_idx] == -1:
            paths.append(np.empty(0, dtype=np.uint32))
            continue

        # Count path length
        path_length = 1
        current = target_idx
        while current != source_idx:
            current = prev[current]
            path_length += 1

        # Create path array
        path = np.empty(path_length, dtype=np.uint32)

        # Fill path
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


cdef double path_cost(np.ndarray[uint32_t, ndim=1] path, np.ndarray[uint16_t, ndim=2] raster_arr, int cols):
    """Calculate the total cost of a path"""
    cdef int i
    cdef npy_intp row, col  # Changed from int to npy_intp
    cdef double cost = 0.0

    for i in range(len(path)):
        row = path[i] // cols
        col = path[i] % cols
        cost += raster_arr[<int>row, <int>col]  # Cast to int for indexing

    return cost


def dijkstra_some_pairs_shortest_paths(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint32_t, ndim=1] source_indices,
        np.ndarray[uint32_t, ndim=1] target_indices,
        uint16_t max_value=65535,
        bint return_paths=True):
    """
    Finds optimal paths between specific source-target pairs by efficiently
    utilizing the single-source multiple-targets implementation.

    This function identifies nodes that appear as both sources and targets and uses
    them as central points for path computation to minimize the number of Dijkstra runs.

    Parameters:
    -----------
    raster_arr : 2D numpy array (uint16)
        Cost raster. Values of max_value are treated as obstacles.
    steps_arr : 2D numpy array (int8)
        Movement directions, e.g., [[0,1], [1,0], ...].
    source_indices : 1D numpy array (uint32)
        Linear indices of the source cells.
    target_indices : 1D numpy array (uint32)
        Linear indices of the target cells.
    max_value : uint16, default=65535
        Value representing obstacles/barriers in the raster.
    return_paths : bool, default=True
        If True, returns the actual paths; if False, returns only the costs.

    Returns:
    --------
    If return_paths=True:
        list of numpy arrays, where each array contains the path from source to target.
        Empty arrays indicate no path found.
    If return_paths=False:
        1D numpy array of costs for each source-target pair.
    """
    # Get dimensions
    cdef int rows = <int> raster_arr.shape[0]
    cdef int cols = <int> raster_arr.shape[1]
    cdef int num_pairs = <int> min(source_indices.shape[0], target_indices.shape[0])

    # Declare variables
    cdef list all_paths = [None] * num_pairs if return_paths else None
    cdef np.ndarray[float64_t, ndim=1] costs = np.full(num_pairs, np.inf)

    # Maps to track source and target connections
    cdef dict node_sources = {}  # For each node, which nodes are sources pointing to it
    cdef dict node_targets = {}  # For each node, which nodes are targets it points to
    cdef dict pair_indices = {}  # Maps (source, target) to original index
    cdef set processed_pairs = set()  # Tracks which pairs have been processed

    cdef int i, j
    cdef uint32_t source, target
    cdef list central_nodes = []  # Nodes that appear as both sources and targets

    # Step 1: Build maps and identify central nodes
    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        # Store pair index for later reference
        pair_indices[(source, target)] = i

        # For each target, record which sources point to it
        if target not in node_sources:
            node_sources[target] = []
        node_sources[target].append(source)

        # For each source, record which targets it points to
        if source not in node_targets:
            node_targets[source] = []
        node_targets[source].append(target)

        # If a node appears as both a source and a target, it's a central node
        if source in node_sources and target in node_targets:
            if source not in central_nodes:
                central_nodes.append(source)
            if target not in central_nodes:
                central_nodes.append(target)

    # Add remaining nodes that are both sources and targets
    for node in node_sources:
        if node in node_targets and node not in central_nodes:
            central_nodes.append(node)

    # Step 2: Process central nodes first
    for central_node in central_nodes:
        # Skip if we've already processed all pairs involving this node
        if central_node not in node_sources and central_node not in node_targets:
            continue

        batch_targets = []
        pair_mapping = []  # Maps result index to original pair index
        reverse_flags = []  # Tracks which paths need to be reversed

        # Add targets for the paths where central_node is the source
        if central_node in node_targets:
            for target in node_targets[central_node]:
                if (central_node, target) not in processed_pairs:
                    batch_targets.append(target)
                    pair_mapping.append(pair_indices[(central_node, target)])
                    reverse_flags.append(False)  # No reversal needed
                    processed_pairs.add((central_node, target))

        # Add sources for the paths where central_node is the target
        if central_node in node_sources:
            for source in node_sources[central_node]:
                if (source, central_node) not in processed_pairs:
                    batch_targets.append(source)
                    pair_mapping.append(pair_indices[(source, central_node)])
                    reverse_flags.append(True)  # Need to reverse this path
                    processed_pairs.add((source, central_node))

        # If we have targets to process
        if batch_targets:
            # Convert to numpy array for dijkstra_single_source_multiple_targets
            targets_array = np.array(batch_targets, dtype=np.uint32)

            # Single Dijkstra run for this central node
            result_paths = dijkstra_single_source_multiple_targets(
                raster_arr, steps_arr, central_node, targets_array, max_value
            )

            # Process results and map back to original pairs
            for j in range(len(result_paths)):
                path = result_paths[j]
                pair_idx = pair_mapping[j]
                need_reverse = reverse_flags[j]

                if return_paths:
                    if len(path) > 0:
                        if need_reverse:
                            # Reverse the path since we computed target->source instead of source->target
                            path = np.flip(path)
                        all_paths[pair_idx] = path
                    else:
                        all_paths[pair_idx] = np.empty(0, dtype=np.uint32)

                if len(path) > 0:
                    costs[pair_idx] = path_cost(path, raster_arr, cols)

    # Step 3: Process remaining source-target pairs
    for i in range(num_pairs):
        source = source_indices[i]
        target = target_indices[i]

        if (source, target) in processed_pairs:
            continue

        # Process this pair normally
        result_paths = dijkstra_single_source_multiple_targets(
            raster_arr, steps_arr, source, np.array([target], dtype=np.uint32), max_value
        )

        path = result_paths[0]
        if return_paths:
            all_paths[i] = path

        if len(path) > 0:
            costs[i] = path_cost(path, raster_arr, cols)

        processed_pairs.add((source, target))

    return all_paths if return_paths else costs


def group_by_proximity(np.ndarray[np.uint32_t, ndim=1] source_indices, int cols):
    """Groups source indices by spatial proximity"""
    cdef int num_sources = <int>source_indices.shape[0]
    cdef np.ndarray[np.uint32_t, ndim=1] sorted_indices = np.zeros(num_sources, dtype=np.uint32)

    # If only one source, just return it
    if num_sources <= 1:
        return source_indices

    # Convert to coordinates
    cdef np.ndarray[np.int32_t, ndim=2] coords = np.zeros((num_sources, 2), dtype=np.int32)
    cdef int i

    for i in range(num_sources):
        coords[i, 0] = <int>(source_indices[i] // cols)
        coords[i, 1] = <int>(source_indices[i] % cols)

    # Sort by row first (simple spatial grouping)
    cdef np.ndarray[np.int32_t, ndim=1] sorted_by_row = np.array(np.argsort(coords[:, 0]), dtype=np.int32)

    for i in range(num_sources):
        sorted_indices[i] = source_indices[sorted_by_row[i]]

    return sorted_indices



def dijkstra_multiple_sources_multiple_targets(
        np.ndarray[uint16_t, ndim=2] raster_arr,
        np.ndarray[int8_t, ndim=2] steps_arr,
        np.ndarray[uint32_t, ndim=1] source_indices,
        np.ndarray[uint32_t, ndim=1] target_indices,
        uint16_t max_value=65535,
        bint return_paths=True):
    # Get dimensions
    cdef int rows = <int>raster_arr.shape[0]
    cdef int cols = <int>raster_arr.shape[1]
    cdef int num_sources = <int>source_indices.shape[0]
    cdef int num_targets = <int>target_indices.shape[0]

    # Declare ALL variables at the beginning
    cdef np.ndarray[uint32_t, ndim=1] sorted_sources
    cdef np.ndarray[float64_t, ndim=2] cost_matrix = np.full((num_sources, num_targets), np.inf)
    cdef list paths = [] if return_paths else None
    cdef list source_paths
    cdef int s, t, original_idx
    cdef dict source_idx_map = {}
    cdef uint32_t source_idx
    cdef double cost

    # Group sources by spatial proximity to maximize computation reuse
    sorted_sources = group_by_proximity(source_indices, cols)

    # Map sorted indices back to original positions
    for s in range(num_sources):
        for original_idx in range(num_sources):
            if sorted_sources[s] == source_indices[original_idx]:
                source_idx_map[s] = original_idx
                break

    # Process sources in spatial proximity order
    for s in range(num_sources):
        source_idx = sorted_sources[s]
        original_idx = source_idx_map[s]

        # Use the already optimized single-source function
        source_paths = dijkstra_single_source_multiple_targets(
            raster_arr, steps_arr, source_idx, target_indices, max_value
        )

        if return_paths:
            paths.append(source_paths)

        # Calculate costs and update matrix
        for t in range(num_targets):
            if len(source_paths[t]) > 0:
                # Calculate path cost
                cost = path_cost(source_paths[t], raster_arr, cols)
                cost_matrix[original_idx, t] = cost

    return paths if return_paths else cost_matrix



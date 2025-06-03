"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""

from typing import Tuple, Union
import numpy as np
import numba as nb

# Define Numba types for clarity and performance optimization
pyint_type = nb.types.intp
int8_type = nb.types.int8
uint16_type = nb.types.uint16
int32_type = nb.types.int32
uint32_type = nb.types.uint32
float64_type = nb.types.float64
boolean_type = nb.types.boolean

# Array types - using contiguous arrays for best performance
int8_2d_array = nb.types.Array(int8_type, 2, 'A')
uint16_2d_array = nb.types.Array(uint16_type, 2, 'A')
uint8_2d_array = nb.types.Array(nb.types.uint8, 2, 'A')
int32_1d_array = nb.types.Array(int32_type, 1, 'A')
uint32_1d_array = nb.types.Array(uint32_type, 1, 'A')
float64_1d_array = nb.types.Array(float64_type, 1, 'A')
uint16_1d_array_c = nb.types.Array(uint16_type, 1, 'C')
float64_1d_array_c = nb.types.Array(float64_type, 1, 'C')


@nb.njit(int8_2d_array(int8_type, int8_type), cache=True, parallel=True,
         fastmath=True)
def intermediate_steps_numba(dr: int8_type, dc: int8_type) -> int8_2d_array:
    """
    Calculate intermediate steps for line traversal using Bresenham-like algorithm.

    This function determines all intermediate grid cells that a line segment passes
    through when moving from one grid cell to another. It's essential for
    calculating edge weights in the graph representation of rasterized geodata.

    Parameters:
        dr (int): Row difference (delta row) between source and target
        dc (int): Column difference (delta column) between source and target

    Returns:
        np.ndarray: Array of intermediate step coordinates as (row, col) pairs

    References:
        [1]
    """
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    sum_abs = abs_dr + abs_dc

    # Handle simple cases first for efficiency
    if sum_abs <= 1:
        return np.zeros((0, 2), dtype=np.int8)

    k = max(abs_dr, abs_dc)
    if k == 1:
        return np.array([[dr, 0], [0, dc]], dtype=np.int8)

    # Pre-allocate result array for complex cases
    result = np.zeros((2 * (k - 1), 2), dtype=np.int8)

    # Calculate intermediate points using linear interpolation
    for i in range(k - 1):
        # Calculate fractional position along the line
        dr_k = (i + 1) * dr / k
        dc_k = (i + 1) * dc / k

        # Store floor and ceil values to capture all traversed cells
        idx = i * 2
        result[idx, 0] = np.int8(np.floor(dr_k))
        result[idx, 1] = np.int8(np.floor(dc_k))
        result[idx + 1, 0] = np.int8(np.ceil(dr_k))
        result[idx + 1, 1] = np.int8(np.ceil(dc_k))

    return result


@nb.njit(float64_type(int8_type, int8_type, pyint_type), cache=True,
         fastmath=True)
def get_cost_factor_numba(dr: int8_type, dc: int8_type, 
                          intermediates_count: pyint_type) -> float64_type:
    """
    Calculate the cost factor for an edge based on its geometric length.

    The cost factor normalizes edge weights by distributing the Euclidean
    distance over all cells involved in the traversal (source, target, and
    intermediates).

    Parameters:
        dr (int): Row difference between source and target
        dc (int): Column difference between source and target
        intermediates_count (int): Number of intermediate cells traversed

    Returns:
        float: Cost factor for edge weight calculation

    References:
        [1]
    """
    # Calculate Euclidean distance using Pythagorean theorem
    distance = np.sqrt(dr * dr + dc * dc)
    # Normalize by total number of cells (source + target + intermediates)
    divisor = 2.0 + intermediates_count
    return distance / divisor


@nb.njit(uint32_type(pyint_type, pyint_type, pyint_type),
         cache=True, fastmath=True)
def ravel_index(row: pyint_type,
                col: pyint_type,
                cols: pyint_type) -> uint32_type:
    """
    Convert 2D grid coordinates to 1D linear index.

    This is a high-performance replacement for np.ravel_multi_index optimized
    for regular grid indexing operations in graph construction.

    Parameters:
        row (int): Row coordinate in the grid
        col (int): Column coordinate in the grid
        cols (int): Total number of columns in the grid

    Returns:
        int: Linear index corresponding to the 2D coordinates
    """
    return uint32_type(row * cols + col)


@nb.njit(uint32_1d_array(int8_type, int8_type, pyint_type, pyint_type),
         cache=True, fastmath=True)
def calculate_region_bounds(dr: int8_type, dc: int8_type,
                            rows: pyint_type, cols: pyint_type) -> uint32_1d_array:
    """
    Calculate valid region bounds for source and target areas in graph construction.

    This function determines the valid coordinate ranges for source and target
    nodes when creating edges with the given step direction, ensuring all
    coordinates remain within grid boundaries.

    Parameters:
        dr (int): Row step direction
        dc (int): Column step direction
        rows (int): Total number of rows in the grid
        cols (int): Total number of columns in the grid

    Returns:
        np.ndarray: Array containing bounds for source and target regions

    References:
        [1]
    """
    # Calculate source region bounds based on step direction
    if dr > 0:
        s_rows_start, s_rows_end = 0, rows - dr
    else:
        s_rows_start, s_rows_end = abs(dr) if dr != 0 else 0, rows

    if dc > 0:
        s_cols_start, s_cols_end = 0, cols - dc
    else:
        s_cols_start, s_cols_end = abs(dc) if dc != 0 else 0, cols

    # Calculate target region bounds based on step direction
    if dr > 0:
        t_rows_start, t_rows_end = dr, rows
    else:
        t_rows_start, t_rows_end = 0, rows + dr if dr != 0 else rows

    if dc > 0:
        t_cols_start, t_cols_end = dc, cols
    else:
        t_cols_start, t_cols_end = 0, cols + dc if dc != 0 else cols

    return np.array([s_rows_start, s_rows_end, s_cols_start, s_cols_end,
                     t_rows_start, t_rows_end, t_cols_start, t_cols_end],
                    dtype=np.uint32)


@nb.njit(boolean_type(
    pyint_type, pyint_type, pyint_type, pyint_type,
    uint8_2d_array, int8_2d_array, uint16_2d_array,
    pyint_type, pyint_type, float64_1d_array), cache=True, fastmath=True)
def is_valid_node(sr: pyint_type, sc: pyint_type, tr: pyint_type, tc: pyint_type,
                  exclude_mask: uint8_2d_array, intermediates: int8_2d_array,
                  raster: uint16_2d_array, rows: uint8_2d_array, cols: uint8_2d_array,
                  out_cost: float64_1d_array) -> boolean_type:
    """
    Check if a node transition is valid and calculate its traversal cost.

    This function validates that a path from source to target coordinates is
    feasible by checking boundary conditions, exclusion masks, and intermediate
    cell validity. It also calculates the total cost for traversing this path.

    Parameters:
        sr (int): Source row coordinate
        sc (int): Source column coordinate
        tr (int): Target row coordinate
        tc (int): Target column coordinate
        exclude_mask (np.ndarray): Binary mask indicating forbidden areas
        intermediates (np.ndarray): Array of intermediate step coordinates
        raster (np.ndarray): Cost raster with terrain/construction costs
        rows (int): Total number of rows in the grid
        cols (int): Total number of columns in the grid
        out_cost (np.ndarray): Output array to store calculated cost

    Returns:
        bool: True if the node transition is valid, False otherwise

    References:
        [1]
    """
    # Check if source or target coordinates are out of bounds
    if (sr < 0 or sr >= rows or sc < 0 or sc >= cols or tr < 0 or tr >= rows or
            tc < 0 or tc >= cols):
        return False

    # Skip if source or target is in forbidden area
    if exclude_mask[sr, sc] == 0 or exclude_mask[tr, tc] == 0:
        return False

    cost = 0.0

    # Check intermediate points and accumulate cost
    for i in range(intermediates.shape[0]):
        ir = sr + intermediates[i, 0]
        ic = sc + intermediates[i, 1]

        # Check if intermediate point is valid
        if (ir < 0 or ir >= rows or ic < 0 or ic >= cols or
                exclude_mask[ir, ic] == 0):
            return False

        # Add intermediate traversal cost
        cost += raster[ir, ic]

    # Add source and target costs to total
    cost += raster[sr, sc] + raster[tr, tc]

    # Store the calculated cost
    out_cost[0] = cost
    return True


@nb.njit(nb.types.Tuple((uint32_1d_array, uint32_1d_array,
                         float64_1d_array, pyint_type))
         (int8_type, int8_type, pyint_type, pyint_type, pyint_type, pyint_type,
          uint8_2d_array, uint16_2d_array, int8_2d_array, pyint_type, pyint_type,
          float64_type, pyint_type),
         cache=True, parallel=True, fastmath=True)
def find_valid_nodes(dr: int8_type, dc: int8_type,
                     s_rows_start: pyint_type, s_rows_end: pyint_type,
                     s_cols_start: pyint_type, s_cols_end: pyint_type,
                     exclude_mask: uint8_2d_array, raster: uint16_2d_array,
                     intermediates: int8_2d_array, rows: pyint_type, cols: pyint_type,
                     cost_factor: float64_type, max_nodes: pyint_type
                     ) -> nb.types.Tuple((uint32_1d_array, uint32_1d_array,
                                          float64_1d_array, pyint_type)):
    """
    Find all valid node transitions for a given step direction within bounds.

    This function systematically searches through a defined region to identify
    all valid edges (node transitions) for a specific step direction,
    calculating their costs and storing them for graph construction.

    Parameters:
        dr (int): Row step direction
        dc (int): Column step direction
        s_rows_start (int): Starting row for source region search
        s_rows_end (int): Ending row for source region search
        s_cols_start (int): Starting column for source region search
        s_cols_end (int): Ending column for source region search
        exclude_mask (np.ndarray): Binary mask indicating forbidden areas
        raster (np.ndarray): Cost raster with terrain/construction costs
        intermediates (np.ndarray): Array of intermediate step coordinates
        rows (int): Total number of rows in the grid
        cols (int): Total number of columns in the grid
        cost_factor (float): Cost normalization factor for this step direction
        max_nodes (int): Maximum number of valid nodes to find

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]: Edge data and count

    References:
        [1]
    """
    # Pre-allocate arrays for maximum possible valid nodes
    max_valid_nodes = min((s_rows_end - s_rows_start) *
                          (s_cols_end - s_cols_start), max_nodes)
    from_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    to_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    costs = np.zeros(max_valid_nodes, dtype=np.float64)

    valid_count = 0
    cost_temp = np.zeros(1, dtype=np.float64)
    dr_int, dc_int = int(dr), int(dc)

    # Systematically search through the defined region
    for sr in range(s_rows_start, s_rows_end):
        for sc in range(s_cols_start, s_cols_end):
            tr = sr + dr_int
            tc = sc + dc_int

            # Check validity and get cost for this node transition
            if is_valid_node(sr, sc, tr, tc, exclude_mask, intermediates,
                              raster, rows, cols, cost_temp):
                if valid_count < max_valid_nodes:  # Prevent array overflow
                    # Store linear indices and normalized cost
                    from_nodes[valid_count] = ravel_index(sr, sc, cols)
                    to_nodes[valid_count] = ravel_index(tr, tc, cols)
                    costs[valid_count] = cost_temp[0] * cost_factor
                    valid_count += 1
                else:
                    break

    # Return only the valid entries
    return (from_nodes[:valid_count], to_nodes[:valid_count],
            costs[:valid_count], valid_count)


@nb.njit(uint32_type(uint32_type, uint32_type, int8_2d_array), fastmath=True,
         cache=True)
def get_max_number_of_edges(n: uint32_type,
                            m: uint32_type,
                            steps: int8_2d_array) -> uint32_type:
    """
    Calculate the maximum number of edges for a given raster shape and neighborhood.

    This function estimates the upper bound on the number of edges that will be
    created when converting a raster into a graph using the specified
    neighborhood steps. This is used for memory pre-allocation optimization.

    Parameters:
        n (int): Number of rows in the raster
        m (int): Number of columns in the raster
        steps (np.ndarray): Array of neighborhood step directions

    Returns:
        int: Maximum possible number of edges

    References:
        [1]
    """
    max_nr_of_edges = 0
    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]
        # Calculate valid region size for this step direction
        max_nr_of_edges = (max_nr_of_edges + (n - uint32_type(abs(dr))) *
                           (m - uint32_type(abs(dc))))
    return max_nr_of_edges


@nb.njit(nb.types.Tuple((uint32_1d_array, uint32_1d_array, float64_1d_array))
         (uint16_2d_array, int8_2d_array, nb.types.boolean),
         parallel=True, cache=True, fastmath=True)
def construct_edges(raster: uint16_2d_array,
                    steps: int8_2d_array,
                    ignore_max: nb.types.boolean = True
                    ) -> nb.types.Tuple((uint32_1d_array,
                                         uint32_1d_array,
                                         float64_1d_array)):
    """
    Construct graph edges from rasterized geodata using specified neighborhood steps.

    This is the main function for converting rasterized cost data into a
    weighted graph representation suitable for least-cost path analysis. It
    processes each step direction in the neighborhood to create edges between
    valid grid cells.

    Parameters:
        raster (np.ndarray): 2D cost raster representing terrain/construction costs
        steps (np.ndarray): Array of neighborhood step directions (Rk neighborhood)
        ignore_max (bool): If True, treats maximum cost values as forbidden areas

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Complete edge list for graph

    References:
        [1]
    """
    rows, cols = raster.shape
    nr_of_edges = get_max_number_of_edges(rows, cols, steps)

    # Pre-allocate result arrays for optimal performance
    from_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    to_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    cost_edges = np.zeros(nr_of_edges, dtype=np.float64)
    last_index = 0

    # Create exclusion mask for forbidden areas
    if ignore_max:
        # Maximum uint16 value represents forbidden areas
        max_cost = np.iinfo(np.uint16).max
        exclude_mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                exclude_mask[i, j] = 1 if raster[i, j] != max_cost else 0
    else:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)

    # Process each step direction in the neighborhood
    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        # Calculate intermediate steps for this direction
        intermediates = intermediate_steps_numba(dr, dc)

        # Determine valid regions for source and target nodes
        bounds = calculate_region_bounds(dr, dc, rows, cols)
        s_rows_start, s_rows_end, s_cols_start, s_cols_end = bounds[:4]

        # Calculate cost normalization factor
        cost_factor = get_cost_factor_numba(dr, dc, intermediates.shape[0])

        # Calculate remaining array capacity
        remaining = nr_of_edges - last_index

        # Find all valid edges for this step direction
        from_nodes, to_nodes, costs, valid_count = find_valid_nodes(
            dr, dc, s_rows_start, s_rows_end, s_cols_start, s_cols_end,
            exclude_mask, raster, intermediates, rows, cols, cost_factor,
            remaining)

        # Add valid edges to result arrays
        if valid_count > 0:
            end_idx = last_index + valid_count
            from_nodes_edges[last_index:end_idx] = from_nodes
            to_nodes_edges[last_index:end_idx] = to_nodes
            cost_edges[last_index:end_idx] = costs
            last_index = end_idx

    # Return arrays trimmed to actual edge count
    return (from_nodes_edges[:last_index], to_nodes_edges[:last_index],
            cost_edges[:last_index])


@nb.njit(cache=True)
def calculate_segment_length(abs_dr: int, abs_dc: int) -> float:
    """
    Calculate the geometric length of a path segment between grid cells.

    This function provides optimized calculations for common step patterns
    and falls back to the Pythagorean theorem for arbitrary steps.

    Parameters:
        abs_dr (int): Absolute row difference
        abs_dc (int): Absolute column difference

    Returns:
        float: Euclidean length of the segment
    """
    # Optimized calculations for common patterns
    if abs_dr <= 1 and abs_dc <= 1:
        # sqrt(2) or 1
        return 1.4142135623730951 if (abs_dr == 1 and abs_dc == 1) else 1.0
    elif (abs_dr == 2 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 2):
        return 2.236067977499789  # sqrt(5)
    elif (abs_dr == 3 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 3):
        return 3.1622776601683795  # sqrt(10)
    elif (abs_dr == 3 and abs_dc == 2) or (abs_dr == 2 and abs_dc == 3):
        return 3.605551275463989  # sqrt(13)
    else:
        # General case using Pythagorean theorem
        return np.sqrt(abs_dr * abs_dr + abs_dc * abs_dc)


@nb.njit(nb.types.Tuple((float64_type, uint16_1d_array_c, float64_1d_array_c))
        (uint16_2d_array, uint32_1d_array),
    fastmath=True, parallel=True)
def calculate_path_metrics_numba(raster:uint16_2d_array,
                                 path_indices: uint32_1d_array
                                 ) -> nb.types.Tuple((float64_type,
                                                      uint16_1d_array_c,
                                                      float64_1d_array_c)):
    """
    Calculate comprehensive metrics for a power line path.

    This function analyzes an optimal path found by the routing algorithm to
    provide detailed statistics about path length, terrain traversed, and cost
    distribution. This information is essential for power line planning and
    cost estimation.

    Parameters:
        raster (np.ndarray): 2D cost raster representing terrain/construction costs
        path_indices (np.ndarray): Array of linear indices representing the path

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Total length, categories, lengths

    References:
        [1]
    """
    # Get raster dimensions for coordinate conversion
    rows, cols = raster.shape
    n_segments = len(path_indices) - 1

    # Convert linear indices to 2D coordinates for path analysis
    path_2d = np.empty((len(path_indices), 2), dtype=np.int64)
    for i in nb.prange(len(path_indices)):
        path_2d[i, 0] = path_indices[i] // cols  # Row coordinate
        path_2d[i, 1] = path_indices[i] % cols   # Column coordinate

    # Identify unique cost categories in the raster
    categories_array = np.sort(np.unique(raster))
    num_categories = len(categories_array)

    # Create efficient mapping from category values to array indices
    min_category = categories_array[0]
    max_category = categories_array[-1]
    range_size = max_category - min_category + 1
    category_to_index = np.full(range_size, -1, dtype=np.int32)

    for i in range(num_categories):
        category_to_index[categories_array[i] - min_category] = i

    # Thread-local storage to avoid race conditions in parallel processing
    num_threads = nb.get_num_threads()
    thread_local_lengths = np.zeros((num_threads, num_categories),
                                    dtype=np.float64)
    thread_local_total_lengths = np.zeros(num_threads, dtype=np.float64)

    # Process each path segment in parallel for performance
    for i in nb.prange(n_segments):
        thread_id = nb.get_thread_id()

        # Get segment endpoints
        row, col = path_2d[i, 0], path_2d[i, 1]
        next_row, next_col = path_2d[i + 1, 0], path_2d[i + 1, 1]

        # Calculate step direction and segment length
        dr = next_row - row
        dc = next_col - col
        abs_dr = abs(dr)
        abs_dc = abs(dc)

        segment_length = calculate_segment_length(abs_dr, abs_dc)
        thread_local_total_lengths[thread_id] += segment_length

        # Get all cells traversed by this segment (including intermediates)
        intermediates = intermediate_steps_numba(np.int8(dr), np.int8(dc))
        all_cells = np.empty((intermediates.shape[0] + 2, 2), dtype=np.int64)

        # Source cell
        all_cells[0, 0] = row
        all_cells[0, 1] = col

        # Intermediate cells
        for j in range(intermediates.shape[0]):
            all_cells[j + 1, 0] = row + intermediates[j, 0]
            all_cells[j + 1, 1] = col + intermediates[j, 1]

        # Target cell
        all_cells[-1, 0] = next_row
        all_cells[-1, 1] = next_col

        # Distribute segment length proportionally among traversed cells
        cell_length = segment_length / all_cells.shape[0]

        # Accumulate length for each terrain category
        for j in range(all_cells.shape[0]):
            r, c = all_cells[j, 0], all_cells[j, 1]
            if 0 <= r < rows and 0 <= c < cols:
                category = raster[r, c]
                if min_category <= category <= max_category:
                    map_idx = category - min_category
                    idx = category_to_index[map_idx]
                    if idx >= 0:
                        thread_local_lengths[thread_id, idx] += cell_length

    # Combine results from all threads
    total_length = 0.0
    for i in range(num_threads):
        total_length += thread_local_total_lengths[i]

    lengths_array = np.zeros(num_categories, dtype=np.float64)
    for i in range(num_threads):
        for j in range(num_categories):
            lengths_array[j] += thread_local_lengths[i, j]

    return total_length, categories_array, lengths_array


@nb.njit(fastmath=True, parallel=True)
def euclidean_distances_numba(raster: np.ndarray,
                              target_point: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distances from all points to a target point.

    This function is optimized for spatial analysis and can be used for various
    distance-based calculations in power line routing applications.

    Parameters:
        raster (np.ndarray): Array of coordinate points
        target_point (np.ndarray): Target point coordinates

    Returns:
        np.ndarray: Array of distances from each point to the target
    """
    n_points = raster.shape[0]
    distances = np.empty(n_points, dtype=np.float64)

    # Optimized implementation for 2D coordinates (most common case)
    if raster.shape[1] == 2:
        for i in nb.prange(n_points):
            dx = raster[i, 0] - target_point[0]
            dy = raster[i, 1] - target_point[1]
            distances[i] = np.sqrt(dx * dx + dy * dy)
    else:
        # General case for any number of dimensions
        n_dims = raster.shape[1]
        for i in nb.prange(n_points):
            squared_dist = 0.0
            for j in range(n_dims):
                diff = raster[i, j] - target_point[j]
                squared_dist += diff * diff
            distances[i] = np.sqrt(squared_dist)

    return distances


@nb.njit(cache=True)
def get_outgoing_edges(node_idx: int, raster: np.ndarray, steps: np.ndarray,
                       rows: int, cols: int,
                       exclude_mask: Union[np.ndarray, None] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get outgoing edges from a specific node for dynamic graph traversal.

    This function calculates outgoing edges on-demand rather than pre-computing
    the entire graph, which can be memory-efficient for large rasters or
    specialized pathfinding algorithms.

    Parameters:
        node_idx (int): Linear index of the source node
        raster (np.ndarray): 2D cost raster
        steps (np.ndarray): Array of neighborhood step directions
        rows (int): Number of rows in the raster
        cols (int): Number of columns in the raster
        exclude_mask (Union[np.ndarray, None]): Optional exclusion mask

    Returns:
        Tuple[np.ndarray, np.ndarray]: Target nodes and edge costs

    References:
        [1]
    """
    # Convert linear index to 2D coordinates
    row = node_idx // cols
    col = node_idx % cols

    # Prepare result arrays for maximum possible edges
    max_edges = steps.shape[0]
    to_nodes = np.zeros(max_edges, dtype=np.uint32)
    costs = np.zeros(max_edges, dtype=np.float64)
    edge_count = 0

    # Create default exclusion mask if not provided
    if exclude_mask is None:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)
        max_cost = np.iinfo(np.uint16).max
        for i in range(rows):
            for j in range(cols):
                if raster[i, j] == max_cost:
                    exclude_mask[i, j] = 0

    # Process each possible step direction
    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        # Calculate target coordinates
        tr = row + dr
        tc = col + dc

        # Check boundary conditions
        if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
            continue

        # Check if target is accessible
        if exclude_mask[tr, tc] == 0:
            continue

        # Get intermediate steps and validate path
        intermediates = intermediate_steps_numba(dr, dc)
        valid = True
        cost = raster[row, col]  # Start with source cost

        # Check all intermediate cells
        for i in range(intermediates.shape[0]):
            ir = row + intermediates[i, 0]
            ic = col + intermediates[i, 1]

            if (ir < 0 or ir >= rows or ic < 0 or ic >= cols or
                    exclude_mask[ir, ic] == 0):
                valid = False
                break

            cost += raster[ir, ic]

        if not valid:
            continue

        # Add target cost and calculate final edge weight
        cost += raster[tr, tc]
        cost_factor = get_cost_factor_numba(dr, dc, intermediates.shape[0])

        # Store edge information
        to_nodes[edge_count] = tr * cols + tc
        costs[edge_count] = cost * cost_factor
        edge_count += 1

    return to_nodes[:edge_count], costs[:edge_count]

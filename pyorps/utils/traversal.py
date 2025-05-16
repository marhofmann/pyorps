import numpy as np
import numba as nb
from numba.typed import Dict

# Define Numba types for clarity
int8_type = nb.types.int8
uint16_type = nb.types.uint16
int32_type = nb.types.int32
uint32_type = nb.types.uint32
float64_type = nb.types.float64
boolean_type = nb.types.boolean

# Array types - using contiguous arrays for best performance
int8_2d_array = nb.types.Array(int8_type, 2, 'A')
uint16_2d_array = nb.types.Array(uint16_type, 2, 'A')
uint8_2d_array = nb.types.Array(nb.types.uint8, 2, 'A')  # For exclusion mask
int32_1d_array = nb.types.Array(int32_type, 1, 'A')
uint32_1d_array = nb.types.Array(uint32_type, 1, 'A')
float64_1d_array = nb.types.Array(float64_type, 1, 'A')


@nb.njit(int8_2d_array(int8_type, int8_type), cache=True, parallel=True, fastmath=True)
def intermediate_steps_numba(dr, dc):
    """JIT-compiled version of intermediate_steps with direct array manipulation"""
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    sum_abs = abs_dr + abs_dc

    # Handle simple cases first for efficiency
    if sum_abs <= 1:
        return np.zeros((0, 2), dtype=np.int8)

    k = max(abs_dr, abs_dc)
    if k == 1:
        return np.array([[dr, 0], [0, dc]], dtype=np.int8)

    # Pre-allocate result array
    result = np.zeros((2 * (k - 1), 2), dtype=np.int8)

    # Manual calculation loop - avoid redundant calculations
    for i in range(k - 1):
        # Calculate fractional position
        dr_k = (i + 1) * dr / k
        dc_k = (i + 1) * dc / k

        # Store floor and ceil values directly
        idx = i * 2
        result[idx, 0] = np.int8(np.floor(dr_k))
        result[idx, 1] = np.int8(np.floor(dc_k))
        result[idx + 1, 0] = np.int8(np.ceil(dr_k))
        result[idx + 1, 1] = np.int8(np.ceil(dc_k))

    return result


@nb.njit(float64_type(int8_type, int8_type, nb.types.intp), cache=True, fastmath=True)
def get_cost_factor_numba(dr, dc, intermediates_count):
    """JIT-compiled version of get_cost_factor"""
    # Use double precision for intermediate calculation
    distance = np.sqrt(dr * dr + dc * dc)
    divisor = 2.0 + intermediates_count
    return distance / divisor


@nb.njit(uint32_type(nb.types.intp, nb.types.intp, nb.types.intp), cache=True, fastmath=True)
def ravel_index(row, col, cols):
    """Convert 2D indices to 1D linear index - much faster than np.ravel_multi_index"""
    return uint32_type(row * cols + col)


@nb.njit(uint32_1d_array(int8_type, int8_type, nb.types.intp, nb.types.intp), cache=True, fastmath=True)
def calculate_region_bounds(dr, dc, rows, cols):
    """Calculate region bounds for source and target regions"""
    # Source region bounds
    if dr > 0:
        s_rows_start, s_rows_end = 0, rows - dr
    else:
        s_rows_start, s_rows_end = abs(dr) if dr != 0 else 0, rows

    if dc > 0:
        s_cols_start, s_cols_end = 0, cols - dc
    else:
        s_cols_start, s_cols_end = abs(dc) if dc != 0 else 0, cols

    # Target region bounds
    if dr > 0:
        t_rows_start, t_rows_end = dr, rows
    else:
        t_rows_start, t_rows_end = 0, rows + dr if dr != 0 else rows

    if dc > 0:
        t_cols_start, t_cols_end = dc, cols
    else:
        t_cols_start, t_cols_end = 0, cols + dc if dc != 0 else cols

    return np.array([s_rows_start, s_rows_end, s_cols_start, s_cols_end,
                            t_rows_start, t_rows_end, t_cols_start, t_cols_end], dtype=np.uint32)


@nb.njit(boolean_type(
    nb.types.intp, nb.types.intp, nb.types.intp, nb.types.intp,
    uint8_2d_array, int8_2d_array, uint16_2d_array,
    nb.types.intp, nb.types.intp, float64_type[:]), cache=True, fastmath=True)
def is_valid_node(sr, sc, tr, tc, exclude_mask, intermediates, raster, rows, cols, out_cost):
    """Check if a node is valid and calculate its cost"""
    # Check if source or target coordinates are out of bounds
    if sr < 0 or sr >= rows or sc < 0 or sc >= cols or tr < 0 or tr >= rows or tc < 0 or tc >= cols:
        return False

    # Skip if source or target is invalid
    if exclude_mask[sr, sc] == 0 or exclude_mask[tr, tc] == 0:
        return False

    cost = 0.0

    # Check intermediate points and calculate cost together
    for i in range(intermediates.shape[0]):
        ir = sr + intermediates[i, 0]
        ic = sc + intermediates[i, 1]

        # Check if intermediate point is valid
        if ir < 0 or ir >= rows or ic < 0 or ic >= cols or exclude_mask[ir, ic] == 0:
            return False

        # Add intermediate cost
        cost += raster[ir, ic]

    # Add source and target costs
    cost += raster[sr, sc] + raster[tr, tc]

    # Store the cost
    out_cost[0] = cost
    return True


@nb.njit(nb.types.Tuple((uint32_1d_array, uint32_1d_array, float64_1d_array, nb.types.intp))
             (int8_type, int8_type, nb.types.intp, nb.types.intp, nb.types.intp, nb.types.intp,
              uint8_2d_array, uint16_2d_array, int8_2d_array,
              nb.types.intp, nb.types.intp, float64_type, nb.types.intp),
         cache=True, parallel=True, fastmath=True)
def find_valid_nodes(dr, dc, s_rows_start, s_rows_end, s_cols_start, s_cols_end,
                     exclude_mask, raster, intermediates, rows, cols, cost_factor, max_nodes):
    """Find all valid nodes for a step direction"""
    # Pre-allocate arrays - using uint32 directly for linear indices
    max_valid_nodes = min((s_rows_end - s_rows_start) * (s_cols_end - s_cols_start), max_nodes)
    from_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    to_nodes = np.zeros(max_valid_nodes, dtype=np.uint32)
    costs = np.zeros(max_valid_nodes, dtype=np.float64)

    valid_count = 0
    cost_temp = np.zeros(1, dtype=np.float64)
    dr_int, dc_int = int(dr), int(dc)
    # Find valid nodes and calculate costs
    for sr in range(s_rows_start, s_rows_end):
        for sc in range(s_cols_start, s_cols_end):
            tr = sr + dr_int
            tc = sc + dc_int

            # Check validity and get cost
            if is_valid_node(sr, sc, tr, tc, exclude_mask, intermediates, raster, rows, cols, cost_temp):
                if valid_count < max_valid_nodes:  # MUST KEEP THIS CHECK
                    # Store linear indices directly - avoid array resizing
                    from_nodes[valid_count] = ravel_index(sr, sc, cols)
                    to_nodes[valid_count] = ravel_index(tr, tc, cols)
                    costs[valid_count] = cost_temp[0] * cost_factor
                    valid_count += 1
                else:
                    break

    # Return only valid entries
    return from_nodes[:valid_count], to_nodes[:valid_count], costs[:valid_count], valid_count


@nb.njit(uint32_type(uint32_type, uint32_type, int8_2d_array), fastmath=True, cache=True)
def get_max_number_of_edges(n, m, steps):
    """
    Returns the maximum number of edges defined by a neighborhood and a given raster shape.
    :param n: The number of rows in the raster.
    :param m: The number of columns in the raster.
    :param steps: The set of steps for a neighborhood.
    :return: The maximum number of edges defined by a neighborhood and a given raster shape.
    """
    max_nr_of_edges = 0
    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]
        max_nr_of_edges = max_nr_of_edges + (n - uint32_type(abs(dr))) * (m - uint32_type(abs(dc)))
    return max_nr_of_edges


@nb.njit(nb.types.Tuple((uint32_1d_array, uint32_1d_array, float64_1d_array))
             (uint16_2d_array, int8_2d_array, nb.types.boolean),
         parallel=True, cache=True, fastmath=True)
def construct_edges(raster, steps, ignore_max=True):
    """Optimized version using Numba with integer indexing and direct linear indices"""
    rows, cols = raster.shape
    nr_of_edges = get_max_number_of_edges(rows, cols, steps)
    # Pre-allocate result arrays
    from_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    to_nodes_edges = np.zeros(nr_of_edges, dtype=np.uint32)
    cost_edges = np.zeros(nr_of_edges, dtype=np.float64)
    last_index = 0

    # Find max cost if needed
    if ignore_max:
        max_cost = np.iinfo(np.uint16).max  # Max uint16 value

        # Create exclusion mask using unsigned char (uint8) - more efficient than boolean
        exclude_mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                exclude_mask[i, j] = 1 if raster[i, j] != max_cost else 0
    else:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)

    # Process each step direction
    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        # Get intermediate steps
        intermediates = intermediate_steps_numba(dr, dc)

        # Calculate region bounds
        bounds = calculate_region_bounds(dr, dc, rows, cols)
        s_rows_start, s_rows_end, s_cols_start, s_cols_end = bounds[:4]

        # Get cost factor
        cost_factor = get_cost_factor_numba(dr, dc, intermediates.shape[0])

        # Calculate remaining capacity
        remaining = nr_of_edges - last_index

        # Find valid nodes - direct use of linear indices
        from_nodes, to_nodes, costs, valid_count = find_valid_nodes(
            dr, dc, s_rows_start, s_rows_end, s_cols_start, s_cols_end,
            exclude_mask, raster, intermediates, rows, cols, cost_factor, remaining)

        # If any valid nodes found, copy to result arrays
        if valid_count > 0:
            # Calculate end index
            end_idx = last_index + valid_count

            # Copy in bulk - much faster than looping
            from_nodes_edges[last_index:end_idx] = from_nodes
            to_nodes_edges[last_index:end_idx] = to_nodes
            cost_edges[last_index:end_idx] = costs

            # Update index
            last_index = end_idx

    # Return slices up to the valid count
    return from_nodes_edges[:last_index], to_nodes_edges[:last_index], cost_edges[:last_index]


@nb.njit(cache=True)
def calculate_segment_length(abs_dr, abs_dc):
    """Calculate the length of a segment based on absolute differences in rows and columns."""
    if abs_dr <= 1 and abs_dc <= 1:
        return 1.4142135623730951 if (abs_dr == 1 and abs_dc == 1) else 1.0
    elif (abs_dr == 2 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 2):
        return 2.236067977499789  # sqrt(5)
    elif (abs_dr == 3 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 3):
        return 3.1622776601683795  # sqrt(10)
    elif (abs_dr == 3 and abs_dc == 2) or (abs_dr == 2 and abs_dc == 3):
        return 3.605551275463989  # sqrt(13)
    else:
        return np.sqrt(abs_dr * abs_dr + abs_dc * abs_dc)


@nb.njit(
    nb.types.Tuple((float64_type,
                    nb.types.Array(uint16_type, 1, 'C'),
                    nb.types.Array(float64_type, 1, 'C')))(
        nb.types.Array(uint16_type, 2, 'A'),
        nb.types.Array(uint32_type, 1, 'A')
    ),
    fastmath=True, parallel=True
)
def calculate_path_metrics_numba(raster, path_indices):
    """Calculate metrics about the path including total length and length through each cost category."""
    # Get raster dimensions
    rows, cols = raster.shape

    # Number of segments in the path
    n_segments = len(path_indices) - 1

    # Convert 1D indices to 2D (row, col)
    path_2d = np.empty((len(path_indices), 2), dtype=np.int64)
    for i in nb.prange(len(path_indices)):
        path_2d[i, 0] = path_indices[i] // cols  # Row
        path_2d[i, 1] = path_indices[i] % cols  # Col

    # Get unique values and sort them
    categories_array = np.sort(np.unique(raster))
    num_categories = len(categories_array)

    # Find min and max categories to create a compact mapping
    min_category = categories_array[0]
    max_category = categories_array[-1]
    range_size = max_category - min_category + 1

    # Create a mapping array that only spans from min_category to max_category
    # Initialize with -1 to indicate invalid categories
    category_to_index = np.full(range_size, -1, dtype=np.int32)

    # Fill the mapping array
    for i in range(num_categories):
        category_to_index[categories_array[i] - min_category] = i

    # Create thread-local arrays to avoid race conditions
    # Each thread will have its own copy of the lengths array
    num_threads = nb.get_num_threads()
    thread_local_lengths = np.zeros((num_threads, num_categories), dtype=np.float64)
    thread_local_total_lengths = np.zeros(num_threads, dtype=np.float64)

    # Process each segment in the path in parallel
    for i in nb.prange(n_segments):
        # Get the thread ID to use thread-local storage
        thread_id = nb.get_thread_id()

        row, col = path_2d[i, 0], path_2d[i, 1]
        next_row, next_col = path_2d[i + 1, 0], path_2d[i + 1, 1]

        # Calculate deltas
        dr = next_row - row
        dc = next_col - col
        abs_dr = abs(dr)
        abs_dc = abs(dc)

        # Calculate segment length using the helper function
        segment_length = calculate_segment_length(abs_dr, abs_dc)
        thread_local_total_lengths[thread_id] += segment_length

        # Get intermediate cells using intermediate_steps_numba
        intermediates = intermediate_steps_numba(np.int8(dr), np.int8(dc))

        # Create a list of all cells including source, target, and intermediates
        all_cells = np.empty((intermediates.shape[0] + 2, 2), dtype=np.int64)

        # Add source cell
        all_cells[0, 0] = row
        all_cells[0, 1] = col

        # Add intermediate cells
        for j in range(intermediates.shape[0]):
            all_cells[j + 1, 0] = row + intermediates[j, 0]
            all_cells[j + 1, 1] = col + intermediates[j, 1]

        # Add target cell
        all_cells[-1, 0] = next_row
        all_cells[-1, 1] = next_col

        # Distribute segment length among cells
        cell_length = segment_length / all_cells.shape[0]

        # Allocate length to categories
        for j in range(all_cells.shape[0]):
            r, c = all_cells[j, 0], all_cells[j, 1]
            if 0 <= r < rows and 0 <= c < cols:
                category = raster[r, c]

                # Only access the mapping array if the category is in range
                if min_category <= category <= max_category:
                    map_idx = category - min_category
                    idx = category_to_index[map_idx]
                    if idx >= 0:  # Verify the index is valid
                        thread_local_lengths[thread_id, idx] += cell_length

    # Combine results from all threads
    total_length = 0.0
    for i in range(num_threads):
        total_length += thread_local_total_lengths[i]

    # Sum up lengths from all threads
    lengths_array = np.zeros(num_categories, dtype=np.float64)
    for i in range(num_threads):
        for j in range(num_categories):
            lengths_array[j] += thread_local_lengths[i, j]

    return total_length, categories_array, lengths_array


@nb.njit(fastmath=True, parallel=True)
def euclidean_distances_numba(raster, target_point):
    """
    Numba-accelerated function to calculate Euclidean distances from all points to a target point.
    """
    n_points = raster.shape[0]
    distances = np.empty(n_points, dtype=np.float64)
    # Specialized case for 2D points (common case)
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
def get_outgoing_edges(node_idx, raster, steps, rows, cols, exclude_mask=None):
    """Get outgoing edges from a specific node only when needed"""
    # Convert linear index to 2D coordinates
    row = node_idx // cols
    col = node_idx % cols

    # Prepare result arrays
    max_edges = steps.shape[0]  # Maximum possible outgoing edges
    to_nodes = np.zeros(max_edges, dtype=np.uint32)
    costs = np.zeros(max_edges, dtype=np.float64)
    edge_count = 0

    # Create exclude mask if not provided
    if exclude_mask is None:
        exclude_mask = np.ones((rows, cols), dtype=np.uint8)
        max_cost = np.iinfo(np.uint16).max
        for i in range(rows):
            for j in range(cols):
                if raster[i, j] == max_cost:
                    exclude_mask[i, j] = 0

    # Process each possible step
    for step_idx in range(steps.shape[0]):
        dr = steps[step_idx, 0]
        dc = steps[step_idx, 1]

        # Calculate target coordinates
        tr = row + dr
        tc = col + dc

        # Check if target is within bounds
        if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
            continue

        # Check if target is valid
        if exclude_mask[tr, tc] == 0:
            continue

        # Get intermediate steps
        intermediates = intermediate_steps_numba(dr, dc)

        # Check if all intermediate cells are valid
        valid = True
        cost = raster[row, col]  # Start with source cost

        for i in range(intermediates.shape[0]):
            ir = row + intermediates[i, 0]
            ic = col + intermediates[i, 1]

            if ir < 0 or ir >= rows or ic < 0 or ic >= cols or exclude_mask[ir, ic] == 0:
                valid = False
                break

            cost += raster[ir, ic]  # Add intermediate cost

        if not valid:
            continue

        # Add target cost
        cost += raster[tr, tc]

        # Calculate cost factor
        cost_factor = get_cost_factor_numba(dr, dc, intermediates.shape[0])

        # Add edge to result arrays
        to_nodes[edge_count] = tr * cols + tc  # Linear index of target
        costs[edge_count] = cost * cost_factor
        edge_count += 1

    return to_nodes[:edge_count], costs[:edge_count]



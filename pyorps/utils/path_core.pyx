"""
Core Cython module for high-performance pathfinding operations.

This module provides optimized data structures and utility functions for 
graph-based pathfinding algorithms, including binary heap implementation,
coordinate transformations, and intermediate step calculations for complex
movement patterns.

The module is designed for maximum performance with:
- nogil sections for true parallelization potential  
- Efficient memory management with C++ vectors
- Optimized algorithms for spatial operations
- Direct memory access patterns for cache efficiency

Performance Notes:
    - All critical path operations are implemented without Python overhead
    - Memory views provide zero-copy access to NumPy arrays
    - C++ vectors offer dynamic sizing with minimal allocation overhead
    - Intermediate step calculations use floating-point precision for accuracy
"""
# At the very top of path_core.pyx and path_algorithms.pyx
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# Cython compiler directives for maximum performance
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, floor, ceil, abs
from libcpp.vector cimport vector
from libcpp cimport bool


# Core data structures for pathfinding operations
cdef struct IntermediatePoint:
    # Represents a single intermediate step in complex movement patterns.
    #
    # Used for calculating sub-pixel accurate paths between non-adjacent cells
    # in raster-based pathfinding. Each point represents a relative displacement
    # from a current position.
    #
    # Members:
    #     dr: Row displacement (-127 to 127 range sufficient for typical steps)
    #     dc: Column displacement (-127 to 127 range sufficient for typical steps)
    int8_t dr
    int8_t dc


cdef struct StepData:
    # Precomputed data for a single movement direction in the pathfinding graph.
    #
    # This structure caches expensive calculations performed once during
    # initialization to avoid repeated computation during pathfinding. The
    # cost_factor incorporates both geometric distance and intermediate step
    # complexity.
    #
    # Members:
    #     dr: Row displacement for this movement direction
    #     dc: Column displacement for this movement direction
    #     cost_factor: Precomputed multiplier incorporating distance and complexity
    int dr
    int dc
    double cost_factor


cdef struct PQNode:
    # Single node in the binary heap priority queue for Dijkstra's algorithm.
    #
    # The priority queue maintains nodes sorted by their path cost, enabling
    # efficient extraction of the minimum cost node during graph traversal.
    #
    # Members:
    #     index: Linear index of the graph node (0 to num_cells-1)
    #     priority: Total path cost to reach this node (lower = higher priority)
    uint32_t index
    double priority


cdef struct BinaryHeap:
    # Binary min-heap implementation for efficient priority queue operations.
    #
    # This heap maintains the frontier of nodes to explore during pathfinding,
    # with O(log n) insertion and extraction times. The heap property ensures
    # the minimum cost node is always at the root.
    #
    # Members:
    #     nodes: Dynamic array storing all heap nodes in heap order
    vector[PQNode] nodes


# Core utility functions for pathfinding algorithms
cdef vector[IntermediatePoint] _calculate_intermediate_steps_cython(int dr, int dc) noexcept nogil:
    """
    Calculate intermediate steps for movement between non-adjacent cells.
    
    
    This function determines all cells that must be traversed when moving
    from one raster cell to another that is not immediately adjacent. The
    algorithm ensures that paths remain connected and accounts for diagonal
    movements that might otherwise skip over obstacle cells.
    
    Algorithm Details:
        - For simple moves (distance <= 1): No intermediate steps needed
        - For single-step diagonals: Add orthogonal components separately  
        - For complex moves: Use linear interpolation with floor/ceil sampling
    
    Parameters (all nogil compatible):
        dr: Row displacement (-∞ to +∞, typically -10 to +10)
        dc: Column displacement (-∞ to +∞, typically -10 to +10)
    
    Returns:
        Vector of IntermediatePoint structs representing the path steps
    
    Performance Notes:
        - Executes entirely without GIL for maximum concurrency potential
        - Uses efficient C++ vector for dynamic result storage
        - Floating-point calculations maintain sub-pixel accuracy
    """
    cdef vector[IntermediatePoint] result
    cdef IntermediatePoint point
    cdef int abs_dr = abs(dr)
    cdef int abs_dc = abs(dc)
    cdef int sum_abs = abs_dr + abs_dc
    cdef int k, p
    cdef double dr_k, dc_k, ddr, ddc, dk, dp
    cdef int8_t floor_dr, floor_dc, ceil_dr, ceil_dc

    if sum_abs <= 1:
        # Adjacent or same cell - no intermediate steps required
        pass
    elif max(abs_dr, abs_dc) == 1:
        # Single diagonal step - decompose into orthogonal components
        # This ensures we check cells along both row and column directions
        point.dr = <int8_t>dr
        point.dc = 0
        result.push_back(point)


        point.dr = 0
        point.dc = <int8_t>dc
        result.push_back(point)
    else:
        # Complex movement requiring linear interpolation
        # Sample points along the line between source and destination
        k = max(abs_dr, abs_dc)  # Number of major steps
        ddr = <double>dr
        ddc = <double>dc
        dk = <double>k

        for p in range(1, k):
            dp = <double>p
            # Calculate fractional position along the movement vector
            dr_k = (dp * ddr) / dk
            dc_k = (dp * ddc) / dk

            # Add floor approximation (conservative path)
            floor_dr = <int8_t>floor(dr_k)
            floor_dc = <int8_t>floor(dc_k)
            point.dr = floor_dr
            point.dc = floor_dc
            result.push_back(point)

            # Add ceiling approximation if different (ensures connectivity)
            ceil_dr = <int8_t>ceil(dr_k)
            ceil_dc = <int8_t>ceil(dc_k)
            if floor_dr != ceil_dr or floor_dc != ceil_dc:
                point.dr = ceil_dr
                point.dc = ceil_dc
                result.push_back(point)
    return result


cdef inline double _get_cost_factor_cython( int dr, int dc, int intermediates_count) noexcept nogil:
    """
    Calculate movement cost factor incorporating distance and path complexity.
    
    
    The cost factor adjusts the base movement cost to account for both the
    geometric distance traveled and the complexity introduced by intermediate
    steps. This ensures that longer, more complex movements are properly
    penalized relative to simpler alternatives.
    
    Formula: distance / (2.0 + intermediate_steps_count)
    - Numerator: Euclidean distance between source and destination
    - Denominator: Base penalty (2.0) plus complexity penalty
    
    Parameters (all nogil compatible):
        dr: Row displacement for the movement
        dc: Column displacement for the movement  
        intermediates_count: Number of intermediate steps required
    
    Returns:
        Multiplicative factor to apply to raw cell costs (always positive)
    
    Performance Notes:
        - Inlined for zero function call overhead
        - Single square root operation for distance calculation
        - Result cached in StepData structure to avoid recalculation
    """
    cdef double distance = sqrt(<double>(dr * dr + dc * dc))
    cdef double divisor = 2.0 + <double>intermediates_count
    return distance / divisor


# Binary heap implementation for efficient priority queue operations
cdef inline int heap_init(BinaryHeap* heap) except -1 nogil:
    """
    Initialize an empty binary heap with reasonable default capacity.
    
    
    This function prepares a heap for use by clearing any existing contents
    and pre-allocating memory for expected usage patterns. The initial
    capacity of 1000 nodes is chosen based on typical pathfinding scenarios.
    
    Parameters:
        heap: Pointer to BinaryHeap structure to initialize
    
    Returns:
        0 on success (error handling may be added in future versions)
    
    Performance Notes:
        - Reserve operation minimizes memory allocations during heap growth
        - Clear operation is O(1) for vectors
        - No memory allocation failures in current implementation
    """
    heap.nodes.clear()
    heap.nodes.reserve(1000)
    return 0


cdef inline bool heap_empty(const BinaryHeap* heap) noexcept nogil:
    """
    Check if the binary heap contains any nodes.
    
    
    Parameters:
        heap: Pointer to BinaryHeap structure to check
    
    Returns:
        True if heap is empty, False if it contains nodes
    
    Performance Notes:
        - O(1) operation using vector's size method
        - Marked noexcept for maximum compiler optimization
    """
    return heap.nodes.size() == 0


cdef inline PQNode heap_top(const BinaryHeap* heap) noexcept nogil:
    """
    Retrieve the minimum priority node without removing it from the heap.
    
    In a min-heap, the root node (index 0) always contains the element
    with the smallest priority value. This operation does not modify
    the heap structure.
    
    Parameters:
        heap: Pointer to BinaryHeap structure (must not be empty)
    
    Returns:
        PQNode with the minimum priority value
    
    Warning:
        Calling this function on an empty heap results in undefined behavior.
        Always check heap_empty() first in production code.
    
    Performance Notes:
        - O(1) operation - simple array access
        - No bounds checking for maximum performance  
    """
    return heap.nodes[0]


cdef inline int heap_push(BinaryHeap* heap, uint32_t idx, double priority) except -1 nogil:
    """
    Insert a new node into the binary heap maintaining heap property.
    
    This function adds a node to the heap and restores the min-heap property
    by bubbling the new element up the tree until it reaches its correct
    position. The heap property ensures parent nodes always have priority
    values less than or equal to their children.
    
    Algorithm:
        1. Add new node at the end of the heap (next available position)
        2. Compare with parent and swap if new node has lower priority  
        3. Repeat until heap property is satisfied or root is reached
    
    Parameters:
        heap: Pointer to BinaryHeap structure to modify
        idx: Graph node index to insert
        priority: Priority value (lower values = higher priority)
    
    Returns:
        0 on success (error handling may be added in future versions)
    
    Time Complexity: O(log n) where n is the number of nodes in heap
    """
    cdef PQNode node
    node.index = idx
    node.priority = priority
    heap.nodes.push_back(node)

    # Bubble up to maintain heap property
    cdef npy_intp pos = heap.nodes.size() - 1
    cdef npy_intp parent
    cdef PQNode temp

    while pos > 0:
        parent = (pos - 1) // 2
        if heap.nodes[parent].priority <= heap.nodes[pos].priority:
            break  # Heap property satisfied

        # Swap with parent
        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[parent]
        heap.nodes[parent] = temp
        pos = parent

    return 0


cdef inline int heap_pop(BinaryHeap* heap) except -1 nogil:
    """
    Remove the minimum priority node from the heap maintaining heap property.
    
    This function removes the root node (minimum priority) and restores the
    heap property by moving the last element to the root and sifting it down
    to its correct position. This is the standard algorithm for heap deletion.
    
    Algorithm:
        1. Replace root with the last element in the heap
        2. Remove the last element (now duplicated at root)
        3. Sift down the new root until heap property is satisfied
    
    Parameters:
        heap: Pointer to BinaryHeap structure to modify
    
    Returns:
        0 on success, 1 if heap was empty
    
    Time Complexity: O(log n) where n is the number of nodes in heap
    
    Performance Notes:
        - Early returns for empty or single-element heaps
        - Efficient sift-down with minimal comparisons
    """
    if heap.nodes.size() == 0:
        return 1  # Error: empty heap

    if heap.nodes.size() > 1:
        # Move last element to root position
        heap.nodes[0] = heap.nodes[heap.nodes.size() - 1]

    heap.nodes.pop_back()

    if heap.nodes.size() <= 1:
        return 0  # No sifting needed for 0 or 1 elements

    # Sift down to restore heap property
    cdef npy_intp pos = 0
    cdef npy_intp left, right, smallest
    cdef npy_intp heap_size = heap.nodes.size()
    cdef PQNode temp

    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos

        # Find the smallest among current node and its children
        if (left < heap_size and
                heap.nodes[left].priority < heap.nodes[smallest].priority):
            smallest = left

        if (right < heap_size and
                heap.nodes[right].priority < heap.nodes[smallest].priority):
            smallest = right

        if smallest == pos:
            break  # Heap property satisfied

        # Swap with smallest child
        temp = heap.nodes[pos]
        heap.nodes[pos] = heap.nodes[smallest]
        heap.nodes[smallest] = temp
        pos = smallest
    return 0


# Index conversion functions for raster-graph mapping
cdef inline uint32_t ravel_index(int row, int col, int cols) noexcept nogil:
    """
    Convert 2D raster coordinates to 1D graph node index.
    
    This function performs the standard row-major order conversion from
    2D array indices to a linear index. This mapping is essential for
    representing raster cells as graph nodes.
    
    Formula: linear_index = row * cols + col
    
    Parameters:
        row: Row index in the raster (0-based)
        col: Column index in the raster (0-based)  
        cols: Total number of columns in the raster
    
    Returns:
        Linear node index suitable for graph algorithms
    
    Performance Notes:
        - Single multiplication and addition operation
        - Inlined for zero function call overhead
        - No bounds checking for maximum performance
    """
    return <uint32_t>(row * cols + col)


cdef inline int unravel_index(uint32_t idx, int cols, npy_intp* row, npy_intp* col) except -1 nogil:
    """
    Convert 1D graph node index back to 2D raster coordinates.
    
    This function performs the inverse of ravel_index, converting a linear
    graph node index back to row and column coordinates in the original raster.
    The results are written to the provided pointer locations.
    
    Formula: 
        row = linear_index // cols
        col = linear_index % cols
    
    Parameters:
        idx: Linear node index from graph algorithms
        cols: Total number of columns in the raster
        row: Pointer to store the calculated row index
        col: Pointer to store the calculated column index
    
    Returns:
        0 on success (error handling may be added in future versions)
    
    Performance Notes:
        - Single division and modulo operation
        - Uses pointer outputs to avoid return value copying
        - Inlined for zero function call overhead
    """
    row[0] = idx // cols
    col[0] = idx % cols
    return 0


cdef int check_path(int dr, int dc, int current_row, int current_col,
                    const uint8_t[:, :] exclude_mask, const uint16_t[:, :] raster,
                    int rows, int cols, double* total_cost) except -1 nogil:
    """
    Validate a movement path and calculate intermediate step costs.
    
    This function checks if a movement from the current position is valid by
    examining all intermediate cells that would be traversed. If any 
    intermediate cell is out of bounds or blocked, the movement is invalid.
    If valid, the total cost of traversing intermediate cells is calculated.
    
    Parameters:
        dr: Row displacement for the movement
        dc: Column displacement for the movement
        current_row: Starting row position  
        current_col: Starting column position
        exclude_mask: 2D mask indicating traversable cells (1=ok, 0=blocked)
        raster: 2D cost raster for calculating traversal costs
        rows: Total number of rows in the raster
        cols: Total number of columns in the raster  
        total_cost: Pointer to store the calculated intermediate costs
    
    Returns:
        1 if path is valid, 0 if path is blocked or out of bounds
    
    Performance Notes:
        - Executes without GIL for maximum concurrency potential
        - Early termination on first invalid cell encountered
        - Direct memory access patterns for cache efficiency
    """
    cdef double cost = 0.0
    cdef int i, int_row, int_col


    # Get intermediate steps for this movement
    cdef vector[IntermediatePoint] intermediates = (
        _calculate_intermediate_steps_cython(dr, dc))
    cdef IntermediatePoint point

    # Check each intermediate point along the path
    for i in range(intermediates.size()):
        point = intermediates[i]
        int_row = current_row + point.dr
        int_col = current_col + point.dc

        # Validate bounds and traversability
        if (int_row < 0 or int_row >= rows or
                int_col < 0 or int_col >= cols or
                exclude_mask[int_row, int_col] == 0):
            return 0  # Invalid path

        # Accumulate cost of traversing this intermediate cell
        cost += raster[int_row, int_col]

    # Path is valid - return total intermediate cost
    total_cost[0] = cost
    return 1


cdef vector[StepData] precompute_directions( np.ndarray[int8_t, ndim=2] steps_arr):
    """
    Precompute movement data for all possible directions in the neighborhood.
    
    This function calculates cost factors and intermediate step counts for
    each movement direction during initialization. Precomputing this data
    avoids expensive recalculation during pathfinding and significantly
    improves runtime performance.
    
    Parameters:
        steps_arr: 2D array where each row contains [dr, dc] for one direction
    
    Returns:
        Vector of StepData structures with precomputed movement information
    
    Performance Notes:
        - Single pass computation during initialization  
        - Results cached for entire pathfinding session
        - Memory pre-allocation with reserve() for efficiency
    """
    cdef vector[StepData] directions
    cdef StepData direction
    cdef int s, dr, dc
    cdef int intermediates_count
    cdef int steps_count = <int>steps_arr.shape[0]

    directions.reserve(steps_count)

    for s in range(steps_count):
        dr = steps_arr[s, 0]
        dc = steps_arr[s, 1]

        # Count intermediate steps for this direction
        intermediates_count = <int>(
            _calculate_intermediate_steps_cython(dr, dc).size())

        # Store precomputed direction data
        direction.dr = dr
        direction.dc = dc
        direction.cost_factor = _get_cost_factor_cython(
            dr, dc, intermediates_count)

        directions.push_back(direction)

    return directions

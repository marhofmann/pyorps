"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland

Backward-compatibility shim — imports from Cython _traversal module.
Falls back to Numba implementations if Cython extension is not available.
"""

try:
    from pyorps.utils._traversal import (
        # Core path functions
        calculate_path_metrics_numba,
        intermediate_steps_numba,

        # Graph construction
        construct_edges,
        construct_edges_3d,
        get_max_number_of_edges,

        # Distance calculations
        euclidean_distances_numba,
        get_cost_factor_numba,

        # Index manipulation
        py_ravel_index as ravel_index,
        calculate_region_bounds,

        # Node validation
        is_valid_node,
        find_valid_nodes,
        find_valid_nodes_3d,

        # Path analysis
        get_outgoing_edges,
        calculate_segment_length,

        # Position correction
        find_nearest_valid_positions_numba,
        check_max_values,

        # Gradient
        calculate_gradient_penalty,
    )

    _CYTHON_TRAVERSAL = True

except ImportError:
    # Fallback: use the Numba implementations
    _CYTHON_TRAVERSAL = False

    from pyorps.utils._traversal_numba import (
        calculate_path_metrics_numba,
        intermediate_steps_numba,
        construct_edges,
        construct_edges_3d,
        get_max_number_of_edges,
        euclidean_distances_numba,
        get_cost_factor_numba,
        ravel_index,
        calculate_region_bounds,
        is_valid_node,
        find_valid_nodes,
        find_valid_nodes_3d,
        get_outgoing_edges,
        calculate_segment_length,
        find_nearest_valid_positions_numba,
        check_max_values,
        calculate_gradient_penalty,
    )
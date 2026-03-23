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
        # Gradient
        calculate_gradient_penalty,
        # Core path functions
        calculate_path_metrics_numba,
        calculate_region_bounds,
        calculate_segment_length,
        check_max_values,
        # Graph construction
        construct_edges,
        construct_edges_3d,
        # Distance calculations
        euclidean_distances_numba,
        # Position correction
        find_nearest_valid_positions_numba,
        find_valid_nodes,
        find_valid_nodes_3d,
        get_cost_factor_numba,
        get_max_number_of_edges,
        # Path analysis
        get_outgoing_edges,
        intermediate_steps_numba,
        # Node validation
        is_valid_node,
    )
    from pyorps.utils._traversal import (
        # Index manipulation
        py_ravel_index as ravel_index,
    )

    _CYTHON_TRAVERSAL = True

except ImportError:
    # Fallback: use the Numba implementations
    _CYTHON_TRAVERSAL = False


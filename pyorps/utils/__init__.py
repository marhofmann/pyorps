"""
Utility functions for geospatial data processing and visualization.

This module provides:
1. Numba-accelerated traversal functions for path calculation and metrics
2. Helper functions for spatial calculations and operations
3. Utilities for working with raster indices and graph construction
4. Cython-optimized pathfinding algorithms (Dijkstra and Delta-stepping)
"""

# Import traversal functions
from .traversal import (
    # Core path functions
    calculate_path_metrics_numba,
    intermediate_steps_numba,

    # Graph construction helpers
    construct_edges,
    get_max_number_of_edges,

    # Distance calculations
    euclidean_distances_numba,
    get_cost_factor_numba,

    # Index manipulation
    ravel_index,
    calculate_region_bounds,

    # Node validation
    is_valid_node,
    find_valid_nodes,

    # Path analysis
    get_outgoing_edges,
    calculate_segment_length
)

# Try to import Cython extensions
try:
    from .path_algorithms import (
        # Dijkstra algorithms
        dijkstra_2d_cython,
        dijkstra_single_source_multiple_targets,
        dijkstra_some_pairs_shortest_paths,
        dijkstra_multiple_sources_multiple_targets,

        # Delta-stepping algorithms
        delta_stepping_2d,
        delta_stepping_single_source_multiple_targets,
        delta_stepping_some_pairs_shortest_paths,
        delta_stepping_multiple_sources_multiple_targets,

        # Utility functions
        group_by_proximity,
    )
    from .path_core import create_exclude_mask, path_cost_uint32, path_cost

    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    print(f"⚠ Cython extensions not available: {e}")

    # Provide informative error functions for all algorithms
    def _cython_not_available(*args, **kwargs):
        raise ImportError(
            "Cython extension 'path_algorithms' not available. "
            "Please install from source or use a pre-compiled wheel."
        )

    # Dijkstra algorithms
    dijkstra_2d_cython = _cython_not_available
    dijkstra_single_source_multiple_targets = _cython_not_available
    dijkstra_some_pairs_shortest_paths = _cython_not_available
    dijkstra_multiple_sources_multiple_targets = _cython_not_available

    # Delta-stepping algorithms
    delta_stepping_2d = _cython_not_available
    delta_stepping_single_source_multiple_targets = _cython_not_available
    delta_stepping_some_pairs_shortest_paths = _cython_not_available
    delta_stepping_multiple_sources_multiple_targets = _cython_not_available

    # Utility functions
    group_by_proximity = _cython_not_available
    path_cost = _cython_not_available
    path_cost_f32 = _cython_not_available
    create_exclude_mask = _cython_not_available

__all__ = [
    # Availability flag
    'CYTHON_AVAILABLE',

    # Dijkstra algorithms
    'dijkstra_2d_cython',
    'dijkstra_single_source_multiple_targets',
    'dijkstra_some_pairs_shortest_paths',
    'dijkstra_multiple_sources_multiple_targets',

    # Delta-stepping algorithms
    'delta_stepping_2d',
    'delta_stepping_single_source_multiple_targets',
    'delta_stepping_some_pairs_shortest_paths',
    'delta_stepping_multiple_sources_multiple_targets',

    # Utility functions from Cython
    'group_by_proximity',
    'path_cost',
    'path_cost_f32',
    'create_exclude_mask',

    # Core path functions from Numba
    "calculate_path_metrics_numba",
    "intermediate_steps_numba",

    # Graph construction helpers from Numba
    "construct_edges",
    "get_max_number_of_edges",

    # Distance calculations from Numba
    "euclidean_distances_numba",
    "get_cost_factor_numba",

    # Index manipulation from Numba
    "ravel_index",
    "calculate_region_bounds",

    # Node validation from Numba
    "is_valid_node",
    "find_valid_nodes",

    # Path analysis from Numba
    "get_outgoing_edges",
    "calculate_segment_length",
]

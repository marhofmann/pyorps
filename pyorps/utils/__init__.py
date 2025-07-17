"""
Utility functions for geospatial data processing and visualization.

This module provides:
1. Numba-accelerated traversal functions for path calculation and metrics
2. Helper functions for spatial calculations and operations
3. Utilities for working with raster indices and graph construction
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
        dijkstra_2d_cython,
        dijkstra_single_source_multiple_targets,
        dijkstra_some_pairs_shortest_paths,
        dijkstra_multiple_sources_multiple_targets,
    )
    from .path_algorithms import create_exclude_mask

    CYTHON_AVAILABLE = True
    print("✓ Cython extensions loaded successfully")
except ImportError as e:
    CYTHON_AVAILABLE = False
    print(f"⚠ Cython extensions not available: {e}")

    # Provide informative error functions
    def dijkstra_2d_cython(*args, **kwargs):
        raise ImportError(
            "Cython extension 'path_algorithms' not available. "
            "Please install from source or use a pre-compiled wheel."
        )

    # Copy for other functions
    dijkstra_single_source_multiple_targets = dijkstra_2d_cython
    dijkstra_some_pairs_shortest_paths = dijkstra_2d_cython
    dijkstra_multiple_sources_multiple_targets = dijkstra_2d_cython
    create_exclude_mask = dijkstra_2d_cython

__all__ = [
    # Cython interface
    'dijkstra_2d_cython',
    'dijkstra_single_source_multiple_targets',
    'dijkstra_some_pairs_shortest_paths',
    'dijkstra_multiple_sources_multiple_targets',
    'create_exclude_mask',
    'CYTHON_AVAILABLE',

    # Core path functions
    "calculate_path_metrics_numba",
    "intermediate_steps_numba",

    # Graph construction helpers
    "construct_edges",
    "get_max_number_of_edges",

    # Distance calculations
    "euclidean_distances_numba",
    "get_cost_factor_numba",

    # Index manipulation
    "ravel_index",
    "calculate_region_bounds",

    # Node validation
    "is_valid_node",
    "find_valid_nodes",

    # Path analysis
    "get_outgoing_edges",
    "calculate_segment_length",

    "dijkstra_2d_cython",
    "dijkstra_multiple_sources_multiple_targets",
    "dijkstra_some_pairs_shortest_paths"
]

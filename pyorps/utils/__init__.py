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

__all__ = [
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
    "calculate_segment_length"
]

"""
Graph operations and routing algorithms for optimal path finding.

This module provides:
1. The main RasterGraph class for creating paths on cost surfaces
2. Path and PathCollection classes for storing and analyzing paths
3. Dynamic loading of graph implementations via get_graph_api_class
"""

# Import main graph class and key function
from .path_finder import PathFinder, get_graph_api_class

# Import Path classes from core (do not re-export from graph.raster_graph)
from ..core.path import Path, PathCollection

# Import exceptions
from ..core.exceptions import NoPathFoundError, AlgorthmNotImplementedError

# Import API base classes
from .api import GraphAPI, GraphLibraryAPI

__all__ = [
    # Main graph class and factory function
    "PathFinder",
    "get_graph_api_class",

    # Path classes
    "Path",
    "PathCollection",

    # API base classes
    "GraphAPI",
    "GraphLibraryAPI",

    # Exceptions
    "NoPathFoundError",
    "AlgorthmNotImplementedError"
]

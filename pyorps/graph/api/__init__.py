"""
Graph API abstractions for different graph libraries and routing algorithms.

This module provides base classes for graph APIs that are implemented
by various backend libraries.
"""

# Base API classes only - specific implementations will be loaded dynamically
from .graph_api import GraphAPI
from .graph_library_api import GraphLibraryAPI

__all__ = [
    # Base API classes
    "GraphAPI",
    "GraphLibraryAPI"
]

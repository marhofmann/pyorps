"""
Graph operations and routing algorithms for optimal path finding.

This module provides:
1. The main RasterGraph class for creating paths on cost surfaces
2. Path and PathCollection classes for storing and analyzing paths
3. Dynamic loading of graph implementations via get_graph_api_class
"""

# Import main graph class and key function
# Import exceptions
from ..core.exceptions import AlgorithmNotImplementedError, NoPathFoundError

# Import Path classes from core (do not re-export from graph.raster_graph)
from ..core.path import Path, PathCollection

# Import API base classes
from .api import GraphAPI, GraphLibraryAPI
from .corridor import (
    cell_sharing_profile,
    corridor_graph_from_routes,
    route_metrics,
    supercover_cells,
)
from .path_finder import PathFinder, get_graph_api_class
from .search_session import (
    FIELD_STORAGE,
    CostField,
    CostFieldSet,
    Leg,
    SavedCostField,
    SearchSession,
    cost_field_provenance,
    fields_fit_in_memory,
    full_window_buffer_m,
)
from .tower_field import (
    TowerField,
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
    check_bounds,
    tower_field_bounds,
    tower_field_from_raster,
)

__all__ = [
    # Main graph class and factory function
    "PathFinder",
    "get_graph_api_class",
    "SearchSession",
    "CostField",
    "CostFieldSet",
    "Leg",
    "SavedCostField",
    "full_window_buffer_m",
    "fields_fit_in_memory",
    "cost_field_provenance",
    "FIELD_STORAGE",

    # Corridor graphs
    "corridor_graph_from_routes",
    "route_metrics",
    "cell_sharing_profile",
    "supercover_cells",

    # Path classes
    "Path",
    "PathCollection",

    # API base classes
    "GraphAPI",
    "GraphLibraryAPI",

    # Exceptions
    "NoPathFoundError",
    "AlgorithmNotImplementedError"
]

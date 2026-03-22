"""Backward-compatibility shim — imports from new OO modules."""
from pyorps.utils._dijkstra import (
    dijkstra_2d_cython,
    dijkstra_single_source_multiple_targets,
    dijkstra_multiple_sources_multiple_targets,
    dijkstra_some_pairs_shortest_paths,
    group_by_proximity_uint32,
)
from pyorps.utils._delta_stepping import (
    group_by_proximity,
    delta_stepping_2d,
    delta_stepping_single_source_multiple_targets,
    delta_stepping_multiple_sources_multiple_targets,
    delta_stepping_some_pairs_shortest_paths,
    delta_stepping_2d_persistent,
    delta_stepping_single_source_multiple_targets_persistent,
    delta_stepping_multiple_sources_multiple_targets_persistent,
    delta_stepping_some_pairs_shortest_paths_persistent,
)

from typing import Union
from numpy import ndarray, array, uint32


from pyorps.core.exceptions import PairwiseError
from .graph_api import GraphAPI
from pyorps.utils.path_algorithms import (dijkstra_2d_cython,
                          dijkstra_multiple_sources_multiple_targets,
                          dijkstra_single_source_multiple_targets,
                          dijkstra_some_pairs_shortest_paths
                          )


class CythonAPI(GraphAPI):
    """
    Graph API implementation that directly uses Cython algorithms on the raster data.
    """

    def shortest_path(self,
                      source_indices: Union[int, list[int], ndarray[int]],
                      target_indices: Union[int, list[int], ndarray[int]],
                      algorithm: str = "dijkstra",
                      pairwise: bool = False) -> list[list[int]]:
        """
        Find shortest/least-cost path(s) directly on raster data using Cython
        implementations.

        Parameters:
            source_indices: Source node indices
            target_indices: Target node indices
            algorithm: Algorithm name (only "dijkstra" supported currently)
            pairwise: Whether to compute paths pairwise (for multiple sources/targets)

        Returns:
            list of path indices for each source-target pair
        """
        # Check if we have multiple sources or targets
        source_list = isinstance(source_indices, (list, ndarray))
        is_source_list = source_list and len(source_indices) > 1
        target_list = isinstance(target_indices, (list, ndarray))
        is_target_list = target_list and len(target_indices) > 1

        # For single source and target
        if not is_source_list and not is_target_list:
            source_idx = source_indices[0] if source_list else source_indices
            target_idx = target_indices[0] if target_list else target_indices

            path_indices = dijkstra_2d_cython(self.raster_data,
                                              self.steps,
                                              source_idx,
                                              target_idx)

            return list(path_indices)

        # Convert to lists if they aren't already
        if isinstance(source_indices, list):
            source_list = source_indices
        else:
            source_list = source_indices.tolist()
        if isinstance(target_indices, list):
            target_list = target_indices
        else:
            target_list = target_indices.tolist()

        # Case: single source, multiple targets
        if not is_source_list:
            source_idx = source_list[0]
            s = array([source_idx], dtype=uint32)
            t = array(target_indices, dtype=uint32)
            paths_nb_list = dijkstra_single_source_multiple_targets(self.raster_data,
                                                                    self.steps,
                                                                    s, t)
            paths = [list(path) for path in paths_nb_list]

        # Case: multiple sources, multiple targets (all pairs)
        # Case: multiple sources, multiple targets (same length -> pairwise)
        else:
            s = array(source_indices, dtype=uint32)
            t = array(target_indices, dtype=uint32)

            if pairwise:
                if len(source_list) != len(target_list):
                    raise PairwiseError()
                paths = dijkstra_some_pairs_shortest_paths(self.raster_data,
                                                           self.steps,
                                                           s, t)
            else:
                paths = dijkstra_multiple_sources_multiple_targets(self.raster_data,
                                                                   self.steps,
                                                                   s, t)
            paths = [list(p) for path in paths for p in path]

        return paths

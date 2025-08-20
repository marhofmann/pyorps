from numpy import array, uint32, uint16

from pyorps.core.exceptions import PairwiseError, AlgorthmNotImplementedError
from pyorps.graph.api.graph_api import *
from pyorps.utils.path_algorithms import (dijkstra_2d_cython,
                                          dijkstra_multiple_sources_multiple_targets,
                                          dijkstra_single_source_multiple_targets,
                                          dijkstra_some_pairs_shortest_paths
                                          )
from pyorps.utils.path_delta import (delta_stepping_2d,
                                     delta_stepping_some_pairs_shortest_paths,
                                     delta_stepping_multiple_sources_multiple_targets,
                                     delta_stepping_single_source_multiple_targets,
                                     estimate_optimal_delta_fast)


class CythonAPI(GraphAPI):
    """
    Graph API implementation that directly uses Cython algorithms on the raster data.
    """

    def shortest_path(
            self,
            source_indices: Union[int, list[int], ndarray[int]],
            target_indices: Union[int, list[int], ndarray[int]],
            algorithm: str = "dijkstra",
            **kwargs
    ) -> Union[NodeList, NodePathList]:
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

        self.max_value = 65535 if self.ignore_max else 65534

        # Check if we have multiple sources or targets
        is_source_list, source_list = self.is_list(source_indices)
        is_target_list, target_list = self.is_list(target_indices)

        # For single source and target
        if not is_source_list and not is_target_list:
            return self._single_source_single_target(source_indices, source_list,
                                                     target_indices, target_list,
                                                     algorithm=algorithm)

        # Case: single source, multiple targets
        if not is_source_list:
            paths = self._single_source_multi_target(source_indices, target_indices,
                                                     algorithm=algorithm)

        # Case: multiple sources, multiple targets (all pairs)
        # Case: multiple sources, multiple targets (same length -> pairwise)
        else:
            paths = self._multi_source_multi_target(source_indices, source_list,
                                                    target_indices, target_list,
                                                    algorithm, kwargs)

        return paths

    def is_list(self, source_indices):
        source_list = isinstance(source_indices, (list, ndarray))
        is_source_list = source_list and len(source_indices) > 1
        return is_source_list, source_list

    def _get_lists(self, source_indices, target_indices):
        # Convert to lists if they aren't already
        if isinstance(source_indices, list):
            source_list = source_indices
        else:
            source_list = source_indices.tolist()
        if isinstance(target_indices, list):
            target_list = target_indices
        else:
            target_list = target_indices.tolist()
        return source_list, target_list

    def _multi_source_multi_target(self, source_indices, source_list, target_indices,
                                   target_list, algorithm, kwargs):
        s = array(source_indices, dtype=uint32)
        t = array(target_indices, dtype=uint32)
        if kwargs.get('pairwise', False):
            if len(source_list) != len(target_list):
                raise PairwiseError()
            if algorithm.lower() == "dijkstra":
                paths = dijkstra_some_pairs_shortest_paths(self.raster_data,
                                                           self.steps,
                                                           s, t,
                                                           max_value=self.max_value)
            elif algorithm.lower() == "delta-stepping":
                func = delta_stepping_some_pairs_shortest_paths
                paths = func(self.raster_data, self.steps, s, t, delta=50,
                             max_value=self.max_value)
            else:
                raise AlgorthmNotImplementedError(algorithm, graph_library="cython")

        else:
            if algorithm.lower() == "dijkstra":
                paths = dijkstra_multiple_sources_multiple_targets(self.raster_data,
                                                                   self.steps,
                                                                   s, t,
                                                                   self.max_value)
            elif algorithm.lower() == "delta-stepping":
                func = delta_stepping_multiple_sources_multiple_targets
                paths = func(self.raster_data, self.steps, s, t, delta=50,
                             max_value=self.max_value)
            else:
                raise AlgorthmNotImplementedError(algorithm, graph_library="cython")

        paths = [list(p) for path in paths for p in path]
        return paths

    def _single_source_multi_target(self, source_idx, target_indices, algorithm):
        s = array([source_idx], dtype=uint32)
        t = array(target_indices, dtype=uint32)
        if algorithm.lower() == "dijkstra":
            paths_nb_list = dijkstra_single_source_multiple_targets(self.raster_data,
                                                                    self.steps,
                                                                    s, t,
                                                                    self.max_value)
        elif algorithm.lower() == "delta-stepping":
            func = delta_stepping_single_source_multiple_targets
            paths_nb_list = func(self.raster_data, self.steps,
                                 s, t, delta=50, max_value=self.max_value)
        else:
            raise AlgorthmNotImplementedError(algorithm, graph_library="cython")

        paths = [list(path) for path in paths_nb_list]
        return paths

    def _single_source_single_target(self, source_indices, source_list, target_indices,
                                     target_list, algorithm):
        source_idx = source_indices[0] if source_list else source_indices
        target_idx = target_indices[0] if target_list else target_indices
        if algorithm.lower() == "dijkstra":
            path_indices = dijkstra_2d_cython(self.raster_data,
                                              self.steps,
                                              source_idx,
                                              target_idx,
                                              max_value=self.max_value)
        elif algorithm.lower() == "delta-stepping":
            path_indices = delta_stepping_2d(self.raster_data, self.steps,
                                             source_idx, target_idx, delta=50,
                                             max_value=self.max_value)
        else:
            raise AlgorthmNotImplementedError(algorithm, graph_library="cython")
        return list(path_indices)


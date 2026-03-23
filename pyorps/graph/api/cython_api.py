
import numpy as np
from numpy import array, astype, ndarray, uint32, uint64

from pyorps.core.exceptions import AlgorithmNotImplementedError, PairwiseError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.graph.api.graph_api import GraphAPI
from pyorps.utils.path_algorithms import (
    delta_stepping_2d_persistent,
    delta_stepping_multiple_sources_multiple_targets_persistent,
    delta_stepping_single_source_multiple_targets_persistent,
    delta_stepping_some_pairs_shortest_paths_persistent,
    dijkstra_2d_cython,
    dijkstra_multiple_sources_multiple_targets,
    dijkstra_single_source_multiple_targets,
    dijkstra_some_pairs_shortest_paths,
)


class CythonAPI(GraphAPI):
    """
    Graph API implementation that directly uses Cython algorithms on raster data.
    """

    def __init__(self, raster_data, steps, ignore_max=True, dem_data=None):
        super().__init__(raster_data, steps, ignore_max)
        self.max_value = IMPASSABLE_CELL_COST if self.ignore_max else 0

    def shortest_path(
            self,
            source_indices: int | list[int] | ndarray,
            target_indices: int | list[int] | ndarray,
            algorithm: str = "dijkstra",
            **kwargs
    ) -> list[int] | list[list[int]]:
        """
        Find shortest/least-cost path(s) using Cython implementations.

        Parameters:
            source_indices: Source node indices
            target_indices: Target node indices
            algorithm: Algorithm name ("dijkstra" or "delta-stepping")
            **kwargs: Additional parameters:
                - pairwise: bool - compute paths pairwise (for multiple sources/targets)
                - delta: float - bucket width for delta-stepping (default 50)
                - use_astar: bool - enable A* heuristic for delta-stepping (default True)
                - num_threads: int - number of threads for delta-stepping (0=auto)
                - min_cell_cost: float - minimum cell cost for A* heuristic

        Returns:
            List of path indices or list of lists for multiple paths
        """
        # Convert inputs to numpy arrays
        sources = self._to_array(source_indices)
        targets = self._to_array(target_indices)

        # Determine the scenario
        is_single_source = len(sources) == 1
        is_single_target = len(targets) == 1
        is_pairwise = kwargs.pop('pairwise', False)  # Remove from kwargs after reading

        # Validate algorithm
        algo = algorithm.lower()
        if algo not in ["dijkstra", "delta-stepping", "delta-stepping-circular"]:
            raise AlgorithmNotImplementedError(algorithm, graph_library="cython")

        # Route to appropriate implementation
        if is_single_source and is_single_target:
            return self._single_path(sources[0], targets[0], algo, **kwargs)
        if is_single_source:
            return self._single_to_multi(sources, targets, algo, **kwargs)
        if is_pairwise:
            return self._pairwise_paths(sources, targets, algo, **kwargs)
        return self._multi_to_multi(sources, targets, algo, **kwargs)

    def _to_array(self, indices):
        """
        Convert input to numpy array, using uint64 if values would overflow uint32.
        """
        UINT32_MAX = np.iinfo(np.uint32).max  # 4294967295

        # Convert to numpy array first
        if isinstance(indices, (int, np.integer)):
            arr = array([indices])
        elif isinstance(indices, list):
            arr = array(indices)
        elif isinstance(indices, ndarray):
            arr = indices if indices.ndim > 0 else array([indices.item()])
        else:
            raise TypeError(f"Unsupported type for indices: {type(indices)}")

        # Choose appropriate dtype based on maximum value
        dtype = uint64 if arr.max() > UINT32_MAX else uint32
        return arr.astype(dtype)

    def _single_path(self, source, target, algo, **kwargs):
        """Single source to single target."""
        if algo == "dijkstra":
            path = dijkstra_2d_cython(
                self.raster_data, self.steps,
                source, target,
                max_value=self.max_value
            )
        else:
            path = delta_stepping_2d_persistent(
                self.raster_data, self.steps,
                uint64(source), uint64(target),
                delta=kwargs.get("delta", 100),
                max_value=self.max_value,
                num_threads=kwargs.get('num_threads', 0),
                margin=kwargs.get('margin', 1.1)
            )
        return list(path)

    def _single_to_multi(self, sources, targets, algo, **kwargs):
        """Single source to multiple targets."""
        if algo == "dijkstra":
            paths = dijkstra_single_source_multiple_targets(
                self.raster_data, self.steps,
                sources[0], targets,
                self.max_value
            )
        else:  # delta-stepping
            # Convert to uint64 for delta-stepping
            paths = delta_stepping_single_source_multiple_targets_persistent(
                self.raster_data, self.steps,
                uint64(sources[0]),
                array(targets, dtype=uint64),
                delta=kwargs.get('delta', 100),
                max_value=self.max_value,
                num_threads=kwargs.get('num_threads', 0),
            )
        return [list(path) for path in paths]

    def _pairwise_paths(self, sources, targets, algo, **kwargs):
        """Pairwise source-target paths."""
        if len(sources) != len(targets):
            raise PairwiseError()

        if algo == "dijkstra":
            paths = dijkstra_some_pairs_shortest_paths(
                self.raster_data, self.steps,
                sources, targets,
                max_value=self.max_value
            )
        else:  # delta-stepping
            # Convert to uint64 for delta-stepping
            paths = delta_stepping_some_pairs_shortest_paths_persistent(
                self.raster_data, self.steps,
                array(sources, dtype=uint64),
                array(targets, dtype=uint64),
                delta=kwargs.get('delta', 100),
                max_value=self.max_value,
            )
        return [list(path) for path in paths]

    def _multi_to_multi(self, sources, targets, algo, **kwargs):
        """Multiple sources to multiple targets (all pairs)."""
        if algo == "dijkstra":
            paths = dijkstra_multiple_sources_multiple_targets(
                self.raster_data, self.steps,
                astype(sources, uint32),
                astype(targets, uint32),
                self.max_value
            )
        else:  # delta-stepping
            # Convert to uint64 for delta-stepping
            paths = delta_stepping_multiple_sources_multiple_targets_persistent(
                self.raster_data, self.steps,
                array(sources, dtype=uint64),
                array(targets, dtype=uint64),
                delta=kwargs.get('delta', 100),
                max_value=self.max_value,
                return_paths=True,
                num_threads=kwargs.get('num_threads', 0)
            )

        # Flatten nested structure for all-pairs results
        return [list(p) for path in paths for p in path]

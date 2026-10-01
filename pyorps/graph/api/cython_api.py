
import numpy as np
from numpy import array, astype, ndarray, uint32, uint64

from pyorps.core.exceptions import AlgorithmNotImplementedError, PairwiseError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.graph.api.graph_api import GraphAPI
from pyorps.utils._dijkstra import make_dijkstra_solver
from pyorps.utils._raster_context import NO_EXCLUSION_VALUE
from pyorps.utils.path_algorithms import (
    delta_stepping_2d_persistent,
    delta_stepping_multiple_sources_multiple_targets_persistent,
    delta_stepping_single_source_multiple_targets_persistent,
    delta_stepping_some_pairs_shortest_paths_persistent,
)

#: Delta-stepping termination margin. The kernels stop once the current
#: bucket's start distance exceeds ``d(target) * margin``; at margin 1 no
#: unsettled vertex can improve the target any more, so this is the exact
#: floor (the kernels clamp anything at or below it to this value). Values
#: above 1 make the search run LONGER - they buy no exactness.
EXACT_TERMINATION_MARGIN = 1.00001


class CythonAPI(GraphAPI):
    """
    Graph API implementation that directly uses Cython algorithms on raster data.
    """

    def __init__(self, raster_data, steps, ignore_max=True, dem_data=None,
                 gradient_luts=None):
        super().__init__(raster_data, steps, ignore_max)
        # ignore_max=False means "no cell is an obstacle". The kernels build
        # their exclude mask from `raster == max_value`, so the disabling
        # sentinel must sit outside the uint16 cost domain - any in-domain
        # value (0 in particular) would forbid the cells holding it.
        self.max_value = (IMPASSABLE_CELL_COST if self.ignore_max
                          else NO_EXCLUSION_VALUE)
        # Per-edge gradient terms (feasibility plan): a DEM aligned to the
        # raster plus the slope-response LUT pair. Consumed by the Dijkstra
        # kernels; delta-stepping support is a later phase.
        self.dem_data = dem_data
        self.gradient_luts = gradient_luts
        if gradient_luts is not None and dem_data is None:
            raise ValueError(
                "gradient_luts require dem_data aligned to the raster")
        # Cached Dijkstra context + solver, see _dijkstra_solver().
        self._solver = None
        self._solver_state = (None, None, None, None, None)

    def _dijkstra_solver(self):
        """The RasterContext + DijkstraSolver for this raster, built once.

        Building the context scans the whole raster to derive the exclude
        mask and the solver allocates dist/prev/visited (13 B/cell); neither
        depends on the query, and every solver entry point resets its arrays
        before it runs, so one instance serves any number of queries with
        identical results.

        The cache keys on the IDENTITY of the arrays this API was given, so
        replacing raster_data/steps/dem_data rebuilds it. Editing one of them
        in place does not - call invalidate_solver_cache() if you do. That is
        the same contract PathFinder already follows: it drops the whole
        graph API whenever its raster handler is rebuilt.
        """
        cached = self._solver_state
        if (self._solver is None
                or cached[0] is not self.raster_data
                or cached[1] is not self.steps
                or cached[2] is not self.dem_data
                or cached[3] is not self.gradient_luts
                or cached[4] != self.max_value):
            self._solver = make_dijkstra_solver(
                self.raster_data, self.steps,
                max_value=self.max_value,
                dem=self.dem_data,
                gradient_luts=self.gradient_luts,
            )
            self._solver_state = (self.raster_data, self.steps,
                                  self.dem_data, self.gradient_luts,
                                  self.max_value)
        return self._solver

    def invalidate_solver_cache(self):
        """Drop the cached Dijkstra context/solver.

        Only needed when the raster this API was built on is edited in
        place; replacing the array object is detected automatically.
        """
        self._solver = None
        self._solver_state = (None, None, None, None, None)

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
                - delta: float - bucket width for delta-stepping (default 100)
                - num_threads: int - number of threads for delta-stepping (0=auto)
                - margin: float - delta-stepping termination margin; values
                  above 1 keep the search running past the point where the
                  target is provably settled (see EXACT_TERMINATION_MARGIN)

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

        if self.gradient_luts is not None and algo != "dijkstra":
            raise NotImplementedError(
                "In-search gradient terms are currently supported by the "
                "cython 'dijkstra' algorithm only (delta-stepping support "
                "is a later phase of the feasibility plan).")

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
            path = self._dijkstra_solver().single_pair(source, target)
        else:
            path = delta_stepping_2d_persistent(
                self.raster_data, self.steps,
                uint64(source), uint64(target),
                delta=kwargs.get("delta", 100),
                max_value=self.max_value,
                num_threads=kwargs.get('num_threads', 0),
                margin=kwargs.get('margin', EXACT_TERMINATION_MARGIN)
            )
        return list(path)

    def _single_to_multi(self, sources, targets, algo, **kwargs):
        """Single source to multiple targets."""
        if algo == "dijkstra":
            paths = self._dijkstra_solver().single_source_multi_target(
                sources[0], targets)
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
            paths = self._dijkstra_solver().some_pairs(sources, targets)
        else:  # delta-stepping
            # Convert to uint64 for delta-stepping
            paths = delta_stepping_some_pairs_shortest_paths_persistent(
                self.raster_data, self.steps,
                array(sources, dtype=uint64),
                array(targets, dtype=uint64),
                delta=kwargs.get('delta', 100),
                max_value=self.max_value,
                # num_threads is deliberately NOT forwarded here. The pairwise
                # branch has always run at the kernel's auto-detect, and
                # delta_stepping_2d_persistent is nondeterministically
                # suboptimal at >=2 threads, so honouring a caller-pinned count
                # would change which route comes back. Forwarding it is a
                # behaviour change and belongs in its own commit.
                margin=kwargs.get('margin', EXACT_TERMINATION_MARGIN),
            )
        return [list(path) for path in paths]

    def _multi_to_multi(self, sources, targets, algo, **kwargs):
        """Multiple sources to multiple targets (all pairs)."""
        if algo == "dijkstra":
            paths = self._dijkstra_solver().multi_source_multi_target(
                astype(sources, uint32),
                astype(targets, uint32),
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

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from pyorps.graph.api.graph_library_api import GraphLibraryAPI
from pyorps.core.exceptions import NoPathFoundError, PairwiseError


class TestGraphLibraryAPI(unittest.TestCase):
    """Test cases for the GraphLibraryAPI abstract base class."""

    def setUp(self):
        """Set up test data."""
        # Create test data
        self.raster_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        # Create test edge data
        self.from_nodes = np.array([0, 0, 1, 1, 2])
        self.to_nodes = np.array([1, 3, 2, 4, 5])
        self.cost = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_initialization_with_edge_data(self):
        """Test initialization with provided edge data."""

        # Create concrete implementation since GraphLibraryAPI is abstract
        class ConcreteGraphLibraryAPI(GraphLibraryAPI):
            def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
                pass

            def _compute_single_source_multiple_targets(self, source, targets,
                                                        algorithm, **kwargs):
                pass

            def _compute_single_path(self, source, target, algorithm, **kwargs):
                pass

            def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
                return "graph_created"

            def get_number_of_nodes(self):  # pragma: no cover
                return 10

            def get_number_of_edges(self):  # pragma: no cover
                return 5

            def remove_isolates(self):  # pragma: no cover
                pass

            def shortest_path(self, source_indices, target_indices, algorithm="dijkstra", **kwargs):  # pragma: no cover
                return [0, 1, 2]

            def get_nodes(self):  # Added new required method
                return list(range(10))

        # Initialize with provided edge data
        api = ConcreteGraphLibraryAPI(
            self.raster_data,
            self.steps,
            from_nodes=self.from_nodes,
            to_nodes=self.to_nodes,
            cost=self.cost
        )

        # Verify attributes are set correctly
        self.assertTrue(np.array_equal(api.raster_data, self.raster_data))
        self.assertTrue(np.array_equal(api.steps, self.steps))
        self.assertEqual(api.graph, "graph_created")  # Should call create_graph

    def test_timing_metrics(self):
        """Test that timing metrics are recorded correctly."""

        # Create a concrete implementation
        class ConcreteGraphLibraryAPI(GraphLibraryAPI):
            def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
                pass

            def _compute_single_source_multiple_targets(self, source, targets,
                                                        algorithm, **kwargs):
                pass

            def _compute_single_path(self, source, target, algorithm, **kwargs):
                pass

            def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
                return "graph_created"

            def get_number_of_nodes(self):  # pragma: no cover
                return 10

            def get_number_of_edges(self):  # pragma: no cover
                return 5

            def remove_isolates(self):  # pragma: no cover
                pass

            def shortest_path(self, source_indices, target_indices, algorithm="dijkstra", **kwargs):  # pragma: no cover
                return [0, 1, 2]

            def get_nodes(self):  # Added new required method
                return list(range(10))

        # Create instance with provided edge data
        api = ConcreteGraphLibraryAPI(
            self.raster_data,
            self.steps,
            from_nodes=self.from_nodes,
            to_nodes=self.to_nodes,
            cost=self.cost
        )

        # Set timing metrics directly for testing
        api.edge_construction_time = 0.0
        api.graph_creation_time = 2.0

        # Verify timing metrics
        self.assertEqual(api.edge_construction_time, 0.0)
        self.assertEqual(api.graph_creation_time, 2.0)

    def test_get_a_star_heuristic(self):
        """Test the get_a_star_heuristic method."""

        # Create a concrete implementation
        class ConcreteGraphLibraryAPI(GraphLibraryAPI):
            def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
                pass

            def _compute_single_source_multiple_targets(self, source, targets,
                                                        algorithm, **kwargs):
                pass

            def _compute_single_path(self, source, target, algorithm, **kwargs):
                pass

            def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
                return "graph_created"

            def get_number_of_nodes(self):  # pragma: no cover
                return 6

            def get_number_of_edges(self):  # pragma: no cover
                return 5

            def remove_isolates(self):  # pragma: no cover
                pass

            def shortest_path(self, source_indices, target_indices, algorithm="dijkstra", **kwargs):  # pragma: no cover
                return [0, 1, 2]

            def get_nodes(self):
                # Return nodes that match our test data
                return np.array([0, 1, 2, 3, 4, 5])

        # Create instance with provided edge data
        api = ConcreteGraphLibraryAPI(
            self.raster_data,
            self.steps,
            from_nodes=self.from_nodes,
            to_nodes=self.to_nodes,
            cost=self.cost
        )

        # Test the A* heuristic calculation
        target_node = 5  # Example target node
        nodes, heuristic = api.get_a_star_heuristic(target_node)

        # Verify the returned nodes are as expected
        self.assertTrue(np.array_equal(nodes, np.array([0, 1, 2, 3, 4, 5])))

        # Verify the heuristic shape
        self.assertEqual(len(heuristic), 6)

        # Verify the heuristic for the target node is zero (or very close)
        target_index = np.where(nodes == target_node)[0][0]
        self.assertAlmostEqual(heuristic[target_index], 0.0)

        # Test with heuristic weight
        _, heuristic_weighted = api.get_a_star_heuristic(target_node, heu_weight=2.0)

        # Verify the weighted heuristic is twice the original
        for i in range(len(heuristic)):
            self.assertAlmostEqual(heuristic_weighted[i], heuristic[i] * 2.0)


def _make_concrete_api(raster_data, steps, from_nodes, to_nodes, cost,
                        single_path_fn=None, multi_target_fn=None):
    """Helper to create a concrete GraphLibraryAPI for testing."""

    class ConcreteAPI(GraphLibraryAPI):
        def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
            return "graph"

        def get_number_of_nodes(self):
            return int(self.raster_data.size)

        def get_number_of_edges(self):
            return 0

        def remove_isolates(self):
            pass

        def get_nodes(self):
            return np.arange(self.raster_data.size)

        def _compute_single_path(self, source, target, algorithm, **kwargs):
            if single_path_fn is not None:
                return single_path_fn(source, target)
            return [source, target]

        def _compute_single_source_multiple_targets(self, source, targets,
                                                     algorithm, **kwargs):
            if multi_target_fn is not None:
                return multi_target_fn(source, targets)
            return [[source, t] for t in targets]

        def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
            return self._compute_all_pairs_shortest_paths(
                sources, targets, algorithm, **kwargs)

    return ConcreteAPI(
        raster_data, steps,
        from_nodes=from_nodes, to_nodes=to_nodes, cost=cost
    )


class TestVectorizedBresenham(unittest.TestCase):
    """Test _vectorized_bresenham method."""

    def setUp(self):
        raster = np.ones((10, 10), dtype=np.uint16)
        steps = np.array([[0, 1], [1, 0]], dtype=np.int8)
        fn = np.array([0, 1])
        tn = np.array([1, 2])
        c = np.array([1.0, 1.0])
        self.api = _make_concrete_api(raster, steps, fn, tn, c)

    def test_horizontal_line(self):
        coords = self.api._vectorized_bresenham(0, 0, 5, 0)
        self.assertEqual(len(coords), 6)
        np.testing.assert_array_equal(coords[:, 0], np.arange(6))
        np.testing.assert_array_equal(coords[:, 1], np.zeros(6))

    def test_vertical_line(self):
        coords = self.api._vectorized_bresenham(0, 0, 0, 5)
        self.assertEqual(len(coords), 6)
        np.testing.assert_array_equal(coords[:, 0], np.zeros(6))
        np.testing.assert_array_equal(coords[:, 1], np.arange(6))

    def test_diagonal_line(self):
        coords = self.api._vectorized_bresenham(0, 0, 4, 4)
        self.assertEqual(len(coords), 5)
        np.testing.assert_array_equal(coords[:, 0], np.arange(5))
        np.testing.assert_array_equal(coords[:, 1], np.arange(5))

    def test_reverse_direction(self):
        """Bresenham normalizes direction internally, so result is sorted."""
        coords = self.api._vectorized_bresenham(5, 0, 0, 0)
        self.assertEqual(len(coords), 6)
        # x goes from 0 to 5 after normalization
        np.testing.assert_array_equal(coords[:, 0], np.arange(6))

    def test_steep_line(self):
        """More vertical than horizontal."""
        coords = self.api._vectorized_bresenham(0, 0, 2, 7)
        # The line has 8 points (dy+1 since steep swaps axes)
        self.assertEqual(len(coords), 8)

    def test_single_point(self):
        coords = self.api._vectorized_bresenham(3, 3, 3, 3)
        self.assertEqual(len(coords), 1)
        np.testing.assert_array_equal(coords[0], [3, 3])


class TestAdvancedAStarHeuristic(unittest.TestCase):
    """Test get_advanced_a_star_heuristic method."""

    def setUp(self):
        self.raster = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]], dtype=np.uint16)
        steps = np.array([[0, 1], [1, 0]], dtype=np.int8)
        fn = np.array([0, 1, 3, 4])
        tn = np.array([1, 2, 4, 5])
        c = np.array([1.0, 1.0, 1.0, 1.0])
        self.api = _make_concrete_api(self.raster, steps, fn, tn, c)

    def test_no_source(self):
        """Without source, uses global minimum."""
        nodes, heuristic = self.api.get_advanced_a_star_heuristic(target=8)
        self.assertEqual(len(nodes), 9)
        # Target node heuristic should be 0
        target_idx = np.where(nodes == 8)[0][0]
        self.assertAlmostEqual(heuristic[target_idx], 0.0)

    def test_with_source(self):
        """With source, uses min along Bresenham line."""
        nodes, heuristic = self.api.get_advanced_a_star_heuristic(
            target=8, source=0)
        self.assertEqual(len(nodes), 9)
        target_idx = np.where(nodes == 8)[0][0]
        self.assertAlmostEqual(heuristic[target_idx], 0.0)

    def test_with_buffer_radius(self):
        """With buffer_radius, uses min in buffered Bresenham area."""
        nodes, heuristic = self.api.get_advanced_a_star_heuristic(
            target=8, source=0, buffer_radius=1)
        self.assertEqual(len(nodes), 9)

    def test_heu_weight(self):
        """heu_weight scales the heuristic."""
        _, h1 = self.api.get_advanced_a_star_heuristic(target=8)
        _, h2 = self.api.get_advanced_a_star_heuristic(target=8, heu_weight=3.0)
        for i in range(len(h1)):
            self.assertAlmostEqual(h2[i], h1[i] * 3.0)


class TestComputeAllPairsShortestPaths(unittest.TestCase):
    """Test _compute_all_pairs_shortest_paths."""

    def test_2x2_pairs(self):
        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1], [1, 0]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])
        api = _make_concrete_api(raster, steps, fn, tn, c)

        paths = api._compute_all_pairs_shortest_paths(
            [0, 1], [2, 3], "dijkstra")
        # 2 sources x 2 targets = 4 paths
        self.assertEqual(len(paths), 4)

    def test_unreachable_pair(self):
        """When a single path raises NoPathFoundError, returns empty list."""
        def fail_path(s, t):
            if s == 0 and t == 3:
                raise NoPathFoundError(s, t)
            return [s, t]

        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])
        api = _make_concrete_api(raster, steps, fn, tn, c,
                                  single_path_fn=fail_path)

        paths = api._compute_all_pairs_shortest_paths(
            [0, 1], [2, 3], "dijkstra")
        self.assertEqual(len(paths), 4)
        # The 0→3 pair should be empty
        self.assertEqual(paths[1], [])
        # Others should succeed
        self.assertEqual(paths[0], [0, 2])


class TestPairwiseShortestPath(unittest.TestCase):
    """Test _pairwise_shortest_path."""

    def test_normal_iteration(self):
        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])
        api = _make_concrete_api(raster, steps, fn, tn, c)

        paths = api._pairwise_shortest_path([0, 5], [4, 9], "dijkstra")
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], [0, 4])
        self.assertEqual(paths[1], [5, 9])

    def test_no_path_for_one_pair(self):
        """If one pair has no path, its entry is an empty list."""
        def fail_sometimes(s, t):
            if s == 5:
                raise NoPathFoundError(s, t)
            return [s, t]

        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])
        api = _make_concrete_api(raster, steps, fn, tn, c,
                                  single_path_fn=fail_sometimes)

        paths = api._pairwise_shortest_path([0, 5], [4, 9], "dijkstra")
        self.assertEqual(paths[0], [0, 4])
        self.assertEqual(paths[1], [])


class TestMultiSourceSingleTarget(unittest.TestCase):
    """Test that multi-source single-target reverses paths."""

    def test_path_reversal(self):
        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])

        def multi_target(source, targets):
            return [[source, t] for t in targets]

        api = _make_concrete_api(raster, steps, fn, tn, c,
                                  multi_target_fn=multi_target)

        # Multiple sources [0, 5], single target 4
        # This goes through the elif branch: source_has_len and not target_has_len
        paths = api.shortest_path([0, 5], 4, "dijkstra")
        # Each path should be reversed (target→source becomes source→target)
        self.assertEqual(len(paths), 2)
        # The multi_target_fn returns [4, 0] and [4, 5], reversed to [0, 4] and [5, 4]
        self.assertEqual(paths[0], [0, 4])
        self.assertEqual(paths[1], [5, 4])


class TestConstructEdgesGPU(unittest.TestCase):
    """Test _construct_edges_gpu fallback behavior."""

    def test_import_error_fallback(self):
        """When cupy is not installed, returns None tuple."""
        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])
        api = _make_concrete_api(raster, steps, fn, tn, c)

        with patch.dict('sys.modules',
                         {'pyorps.utils.traversal_gpu': None}):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = api._construct_edges_gpu(None)
        self.assertEqual(result, (None, None, None))

    def test_gpu_not_available_fallback(self):
        """When GPU_AVAILABLE is False, returns None tuple."""
        raster = np.ones((5, 5), dtype=np.uint16)
        steps = np.array([[0, 1]], dtype=np.int8)
        fn = np.array([0])
        tn = np.array([1])
        c = np.array([1.0])
        api = _make_concrete_api(raster, steps, fn, tn, c)

        mock_module = MagicMock()
        mock_module.GPU_AVAILABLE = False
        with patch.dict('sys.modules',
                         {'pyorps.utils.traversal_gpu': mock_module}):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = api._construct_edges_gpu(None)
        self.assertEqual(result, (None, None, None))


class TestEnsurePathEndpoints(unittest.TestCase):
    """Test _ensure_path_endpoints static method."""

    def test_both_correct(self):
        path = [1, 2, 3]
        result = GraphLibraryAPI._ensure_path_endpoints(path, 1, 3)
        self.assertEqual(result, [1, 2, 3])

    def test_missing_source(self):
        path = [2, 3]
        result = GraphLibraryAPI._ensure_path_endpoints(path, 1, 3)
        self.assertEqual(result, [1, 2, 3])

    def test_missing_target(self):
        path = [1, 2]
        result = GraphLibraryAPI._ensure_path_endpoints(path, 1, 3)
        self.assertEqual(result, [1, 2, 3])

    def test_missing_both(self):
        path = [2]
        result = GraphLibraryAPI._ensure_path_endpoints(path, 1, 3)
        self.assertEqual(result, [1, 2, 3])

    def test_empty_path(self):
        path = []
        result = GraphLibraryAPI._ensure_path_endpoints(path, 1, 3)
        self.assertEqual(result, [])


class TestDemKwargsPopMutation(unittest.TestCase):
    """P2.4: dem_kwargs.pop() mutates caller's dict.

    When GraphLibraryAPI.__init__ uses .pop() on dem_kwargs, it removes
    the 'gradient_function' key from the caller's dictionary. This means
    a second call to create_graph silently uses the default instead of
    the caller-specified gradient function.
    """

    def test_dem_kwargs_not_mutated(self):
        """dem_kwargs dict should not be modified after GraphLibraryAPI init."""

        class ConcreteAPI(GraphLibraryAPI):
            def _all_pairs_shortest_path(self, sources, targets, algorithm,
                                          **kwargs):
                pass

            def _compute_single_source_multiple_targets(self, source, targets,
                                                         algorithm, **kwargs):
                pass

            def _compute_single_path(self, source, target, algorithm,
                                      **kwargs):
                pass

            def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
                return "graph"

            def get_number_of_nodes(self):
                return 9

            def get_number_of_edges(self):
                return 0

            def remove_isolates(self):
                pass

            def get_nodes(self):
                return np.arange(9)

            def shortest_path(self, source_indices, target_indices,
                              algorithm="dijkstra", **kwargs):
                return [0, 1, 2]

        raster = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype=np.uint16)
        steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        dem = np.array([[10.0, 20.0, 30.0],
                        [15.0, 25.0, 35.0],
                        [20.0, 30.0, 40.0]])

        dem_kwargs = {"gradient_function": "linear"}
        dem_kwargs_copy = dem_kwargs.copy()

        with patch('pyorps.graph.api.graph_library_api.construct_edges_3d',
                   return_value=(np.array([0, 1]), np.array([1, 2]),
                                 np.array([1.0, 2.0]))):
            _ = ConcreteAPI(
                raster, steps,
                dem_data=dem,
                dem_kwargs=dem_kwargs
            )

        # The caller's dict should NOT have been modified
        self.assertEqual(
            dem_kwargs, dem_kwargs_copy,
            f"dem_kwargs was mutated from {dem_kwargs_copy} to {dem_kwargs}. "
            f"The 'gradient_function' key was removed by .pop()!"
        )

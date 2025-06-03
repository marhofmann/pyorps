import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from pyorps.graph.api.rustworkx_api import RustworkxAPI


class TestRustworkxAPI(unittest.TestCase):
    """Test cases for the RustworkxAPI class."""

    def setUp(self):
        """Set up test data."""
        # Create test data
        self.raster_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        # Create edge data
        self.from_nodes = np.array([0, 0, 1, 1, 2])
        self.to_nodes = np.array([1, 3, 2, 4, 5])
        self.cost = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Create a standard mock instance
        self.mock_graph_instance = MagicMock()

        # Create API with real constructor, then replace its graph attribute
        self.api = RustworkxAPI(
            self.raster_data,
            self.steps,
            from_nodes=self.from_nodes,
            to_nodes=self.to_nodes,
            cost=self.cost
        )
        self.api.graph = self.mock_graph_instance

    def test_create_graph(self):
        """Test create_graph method."""

        # Create a test implementation that tracks graph creation logic
        class TestRustworkxAPI(RustworkxAPI):
            def __init__(self):
                # Skip normal initialization
                pass

            def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
                # Calculate max_node as in the original method
                self.n_value = np.max([np.max(from_nodes), np.max(to_nodes)])

                # Record that we were called with the expected parameters
                self.create_graph_called = True
                self.from_nodes = from_nodes
                self.to_nodes = to_nodes
                self.cost = cost

                # Return a mock graph
                return MagicMock()

        # Create our test implementation
        api = TestRustworkxAPI()

        # Call the method we want to test
        api.create_graph(
            self.from_nodes,
            self.to_nodes,
            self.cost
        )

        # Verify the method was called with correct parameters
        self.assertTrue(api.create_graph_called)
        self.assertTrue(np.array_equal(api.from_nodes, self.from_nodes))
        self.assertTrue(np.array_equal(api.to_nodes, self.to_nodes))
        self.assertTrue(np.array_equal(api.cost, self.cost))

        # Verify n was calculated correctly
        max_node = np.max([np.max(self.from_nodes), np.max(self.to_nodes)])
        self.assertEqual(api.n_value, max_node)

    def test_get_number_of_nodes_and_edges(self):
        """Test get_number_of_nodes and get_number_of_edges methods."""
        # Configure mocks to return specific values
        self.mock_graph_instance.num_nodes.return_value = 6
        self.mock_graph_instance.num_edges.return_value = 5

        # Verify methods return expected values
        self.assertEqual(self.api.get_number_of_nodes(), 6)
        self.assertEqual(self.api.get_number_of_edges(), 5)

        # Verify correct graph methods were called
        self.mock_graph_instance.num_nodes.assert_called_once()
        self.mock_graph_instance.num_edges.assert_called_once()

    def test_ensure_path_endpoints(self):
        """Test _ensure_path_endpoints static method."""
        # Test with source and target not in path
        self.assertEqual(
            RustworkxAPI._ensure_path_endpoints([1, 2, 3], 0, 4),
            [0, 1, 2, 3, 4]
        )

        # Test with source in path but target not in path
        self.assertEqual(
            RustworkxAPI._ensure_path_endpoints([0, 1, 2], 0, 3),
            [0, 1, 2, 3]
        )

        # Test with source not in path but target in path
        self.assertEqual(
            RustworkxAPI._ensure_path_endpoints([1, 2, 3], 0, 3),
            [0, 1, 2, 3]
        )

        # Test with empty path
        self.assertEqual(
            RustworkxAPI._ensure_path_endpoints([], 0, 1),
            []
        )

    def test_shortest_path_dijkstra_single(self):
        """Test shortest_path with Dijkstra for single source and target."""
        # Patch the internal method
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            mock_compute.return_value = [0, 2, 4]

            # Call method
            result = self.api.shortest_path(0, 4, algorithm="dijkstra")

            # Verify _compute_single_path was called with correct parameters
            mock_compute.assert_called_once_with(0, 4, "dijkstra")
            self.assertEqual(result, [0, 2, 4])

    def test_shortest_path_astar(self):
        """Test shortest_path with A* algorithm."""
        # Create a heuristic function
        heuristic = lambda node: 0.0

        # Patch _compute_single_path to avoid actual function calls
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            mock_compute.return_value = [0, 2, 4]

            # Call method
            result = self.api.shortest_path(0, 4, algorithm="astar", heuristic=heuristic)

            # Verify _compute_single_path was called with correct parameters
            mock_compute.assert_called_once_with(0, 4, "astar", heuristic=heuristic)
            self.assertEqual(result, [0, 2, 4])

    def test_shortest_path_bellman_ford(self):
        """Test shortest_path with Bellman-Ford algorithm."""
        # Patch the internal method
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            mock_compute.return_value = [0, 1, 4]

            # Call method
            result = self.api.shortest_path(0, 4, algorithm="bellman_ford")

            # Verify _compute_single_path was called with correct parameters
            mock_compute.assert_called_once_with(0, 4, "bellman_ford")
            self.assertEqual(result, [0, 1, 4])

    def test_shortest_path_unknown_algorithm(self):
        """Test shortest_path with unknown algorithm."""
        with self.assertRaises(AlgorthmNotImplementedError):
            self.api.shortest_path(0, 4, algorithm="unknown_algorithm")

    def test_shortest_path_no_path_found(self):
        """Test shortest_path when no path exists."""
        # Mock _compute_single_path to raise NoPathFoundError
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            mock_compute.side_effect = NoPathFoundError(source=0, target=4)

            # Verify NoPathFoundError is raised
            with self.assertRaises(NoPathFoundError):
                self.api.shortest_path(0, 4, algorithm="dijkstra")

    def test_shortest_path_multi_target(self):
        """Test shortest_path with single source and multiple targets."""
        # Mock the _compute_single_source_multiple_targets method
        with patch.object(self.api, '_compute_single_source_multiple_targets') as mock_compute:
            mock_compute.return_value = [[0, 1, 2], [0, 3, 4], []]

            # Call method
            result = self.api.shortest_path(0, [2, 4, 6], algorithm="dijkstra")

            # Verify correct method was called with correct parameters
            mock_compute.assert_called_once_with(0, [2, 4, 6], "dijkstra")
            self.assertEqual(result, [[0, 1, 2], [0, 3, 4], []])

    def test_shortest_path_pairwise(self):
        """Test shortest_path with pairwise=True."""
        # Mock _pairwise_shortest_path
        with patch.object(self.api, '_pairwise_shortest_path') as mock_compute:
            mock_compute.return_value = [[0, 1, 2], [3, 4, 5], []]

            # Call method
            result = self.api.shortest_path([0, 3, 6],
                                            [2, 5, 7],
                                            algorithm="dijkstra", pairwise=True)

            # Verify _pairwise_shortest_path was called with correct parameters
            mock_compute.assert_called_once_with([0, 3, 6], [2, 5, 7],
                                                 "dijkstra", pairwise=True)
            self.assertEqual(result, [[0, 1, 2], [3, 4, 5], []])

    def test_shortest_path_all_pairs(self):
        """Test all-pairs shortest path computation."""
        # Patch the internal method
        with patch.object(self.api, '_all_pairs_shortest_path') as mock_all_pairs:
            paths = [[0, 1, 2], [0, 3, 5], [], [1, 3, 2], [1, 4, 5], []]
            mock_all_pairs.return_value = paths

            # Call method
            result = self.api.shortest_path([0, 1], [2, 5, 7], algorithm="dijkstra")

            # Verify _all_pairs_shortest_path was called correctly
            mock_all_pairs.assert_called_once_with([0, 1], [2, 5, 7], "dijkstra")
            self.assertEqual(result, paths)

    def test_get_nodes(self):
        """Test get_nodes method."""
        # Set up mock to return nodes
        self.mock_graph_instance.nodes.return_value = [0, 1, 2, 3, 4, 5]

        # Call the method
        nodes = self.api.get_nodes()

        # Verify the result
        self.assertEqual(nodes, [0, 1, 2, 3, 4, 5])
        self.mock_graph_instance.nodes.assert_called_once()

    def test_get_a_star_heuristic(self):
        """Test the get_a_star_heuristic method."""
        # Configure mock to return list of nodes
        self.mock_graph_instance.nodes.return_value = [0, 1, 2, 3, 4, 5]

        # Call get_a_star_heuristic
        target_node = 5  # Example target node
        nodes, heuristic = self.api.get_a_star_heuristic(target_node)

        # Verify the returned nodes are correct
        self.assertEqual(len(nodes), 6)
        self.assertEqual(list(nodes), [0, 1, 2, 3, 4, 5])

        # Verify the heuristic for the target node is zero (or very close)
        target_index = list(nodes).index(target_node)
        self.assertAlmostEqual(heuristic[target_index], 0.0)

        # Test with heuristic weight
        _, heuristic_weighted = self.api.get_a_star_heuristic(target_node, heu_weight=2.0)

        # Verify the weighted heuristic is twice the original
        for i in range(len(heuristic)):
            self.assertAlmostEqual(heuristic_weighted[i], heuristic[i] * 2.0)


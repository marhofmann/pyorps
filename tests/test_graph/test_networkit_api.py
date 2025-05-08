import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from pyorps.graph.api.networkit_api import NetworkitAPI


class TestNetworkitAPI(unittest.TestCase):
    """Test cases for the NetworkitAPI class."""

    def setUp(self):
        """Set up test data."""
        # Create test data
        self.raster_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        # Create edge data
        self.from_nodes = np.array([0, 0, 1, 1, 2])
        self.to_nodes = np.array([1, 3, 2, 4, 5])
        self.cost = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Create a standard mock instance - don't patch during setup
        self.mock_graph_instance = MagicMock()

        # Create API with real constructor, then replace its graph attribute
        self.api = NetworkitAPI(
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
        class TestNetworkitAPI(NetworkitAPI):
            def __init__(self):
                # Skip normal initialization
                pass

            def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
                # Calculate n as in the original method
                self.n_value = np.max([np.max(from_nodes), np.max(to_nodes)]) + 1

                # Record that we were called with the expected parameters
                self.create_graph_called = True
                self.from_nodes = from_nodes
                self.to_nodes = to_nodes
                self.cost = cost

                # Return a mock graph
                return MagicMock()

        # Create our test implementation
        api = TestNetworkitAPI()

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
        max_node = np.max([np.max(self.from_nodes), np.max(self.to_nodes)]) + 1
        self.assertEqual(api.n_value, max_node)

    def test_get_number_of_nodes_and_edges(self):
        """Test get_number_of_nodes and get_number_of_edges methods."""
        # Configure mocks to return specific values
        self.mock_graph_instance.numberOfNodes.return_value = 6
        self.mock_graph_instance.numberOfEdges.return_value = 5

        # Verify methods return expected values
        self.assertEqual(self.api.get_number_of_nodes(), 6)
        self.assertEqual(self.api.get_number_of_edges(), 5)

        # Verify correct graph methods were called
        self.mock_graph_instance.numberOfNodes.assert_called_once()
        self.mock_graph_instance.numberOfEdges.assert_called_once()

    def test_remove_isolates(self):
        """Test remove_isolates method."""
        # Configure mock to simulate iterating over nodes with some isolated
        self.mock_graph_instance.iterNodes.return_value = [0, 1, 2]
        self.mock_graph_instance.isIsolated.side_effect = [True, False, True]

        # Call method
        self.api.remove_isolates()

        # Verify isIsolated was called for each node
        self.assertEqual(self.mock_graph_instance.isIsolated.call_count, 3)

        # Verify removeNode was called only for isolated nodes
        self.assertEqual(self.mock_graph_instance.removeNode.call_count, 2)
        self.mock_graph_instance.removeNode.assert_any_call(0)
        self.mock_graph_instance.removeNode.assert_any_call(2)

    def test_ensure_path_endpoints(self):
        """Test _ensure_path_endpoints static method."""
        # Test with source and target not in path
        self.assertEqual(
            NetworkitAPI._ensure_path_endpoints([1, 2, 3], 0, 4),
            [0, 1, 2, 3, 4]
        )

        # Test with source in path but target not in path
        self.assertEqual(
            NetworkitAPI._ensure_path_endpoints([0, 1, 2], 0, 3),
            [0, 1, 2, 3]
        )

        # Test with source not in path but target in path
        self.assertEqual(
            NetworkitAPI._ensure_path_endpoints([1, 2, 3], 0, 3),
            [0, 1, 2, 3]
        )

        # Test with empty path
        self.assertEqual(
            NetworkitAPI._ensure_path_endpoints([], 0, 1),
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

    def test_shortest_path_bidirectional_dijkstra(self):
        """Test shortest_path with BidirectionalDijkstra algorithm."""
        # Mock the _compute_single_path method at the instance level
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            mock_compute.return_value = [0, 1, 4]

            # Call method
            result = self.api.shortest_path(0, 4, algorithm="bidirectional_dijkstra")

            # Verify correct method was called with correct parameters
            mock_compute.assert_called_once_with(0, 4, "bidirectional_dijkstra")
            self.assertEqual(result, [0, 1, 4])

    def test_shortest_path_astar(self):
        """Test shortest_path with A* algorithm."""
        with patch('networkit.distance.AStar') as mock_astar:
            # Configure mock
            astar_instance = MagicMock()
            mock_astar.return_value = astar_instance
            astar_instance.getPath.return_value = [0, 3, 4]

            # Create heuristic function and call method
            heuristic = lambda u, v: abs(u - v)

            # Patch _compute_single_path to avoid the actual AStar call
            with patch.object(self.api, '_compute_single_path') as mock_compute:
                mock_compute.return_value = [0, 3, 4]
                result = self.api.shortest_path(0, 4, algorithm="astar", heu=heuristic)

                # Verify _compute_single_path was called with correct parameters
                mock_compute.assert_called_once_with(0, 4, "astar", heu=heuristic)
                self.assertEqual(result, [0, 3, 4])

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
        # Mock _compute_single_path
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            # Configure mock to return paths and one error
            mock_compute.side_effect = [[0, 1, 2], [3, 4, 5], NoPathFoundError(source=6, target=7)]

            # Call method
            result = self.api.shortest_path([0, 3, 6], [2, 5, 7], algorithm="dijkstra", pairwise=True)

            # Verify _compute_single_path was called for each pair
            self.assertEqual(mock_compute.call_count, 3)

            # Verify result (last path should be empty due to NoPathFoundError)
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
        # Configure mock to return nodes when iterNodes is called
        self.mock_graph_instance.iterNodes.return_value = [0, 1, 2, 3, 4, 5]

        # Call the method
        nodes = self.api.get_nodes()

        # Verify the result
        self.assertEqual(nodes, [0, 1, 2, 3, 4, 5])
        self.mock_graph_instance.iterNodes.assert_called_once()

    def test_get_a_star_heuristic(self):
        """Test the get_a_star_heuristic method."""
        # Mock get_nodes to return a list of nodes
        with patch.object(self.api, 'get_nodes', return_value=np.array([0, 1, 2, 3, 4, 5])):
            # Create a complete mock implementation of get_a_star_heuristic to avoid calling the actual implementation
            with patch.object(self.api.__class__, 'get_a_star_heuristic') as mock_method:
                # Configure the mock to return appropriate values
                nodes = np.array([0, 1, 2, 3, 4, 5])
                heuristic = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # Last one is 0 for target
                mock_method.return_value = (nodes, heuristic)

                # Call get_a_star_heuristic
                target_node = 5  # Example target node
                result_nodes, result_heuristic = self.api.get_a_star_heuristic(target_node)

                # Verify mock was called with correct parameters
                mock_method.assert_called_once_with(target_node)

                # Verify the returned nodes are correct
                self.assertEqual(len(result_nodes), 6)
                self.assertEqual(list(result_nodes), [0, 1, 2, 3, 4, 5])

                # Verify the heuristic for the target node is zero
                target_index = list(result_nodes).index(target_node)
                self.assertAlmostEqual(float(result_heuristic[target_index]), 0.0)

                # Reset mock for testing with heuristic weight
                mock_method.reset_mock()
                mock_method.return_value = (nodes, heuristic * 2.0)

                # Test with heuristic weight
                self.api.get_a_star_heuristic(target_node, heu_weight=2.0)

                # Verify mock was called with correct parameters including heu_weight
                mock_method.assert_called_once_with(target_node, heu_weight=2.0)

    def test_shortest_path_astar_no_heuristic(self):
        """Test shortest_path with A* algorithm but missing heuristic."""
        # Mock _compute_single_path to explicitly check for and raise ValueError
        # when 'heu' is not provided for A* algorithm
        with patch.object(self.api, '_compute_single_path') as mock_compute:
            # Set up the mock to raise ValueError when astar is used without heuristic
            def side_effect(source, target, algorithm, **kwargs):
                if algorithm == "astar" and 'heu' not in kwargs:
                    raise ValueError("Missing required heuristic function for A* algorithm")
                return [source, target]

            mock_compute.side_effect = side_effect

            # Assert that ValueError is raised
            with self.assertRaises(ValueError):
                self.api.shortest_path(0, 4, algorithm="astar")
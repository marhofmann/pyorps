import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from pyorps.graph.api.igraph_api import IGraphAPI


class TestIGraphAPI(unittest.TestCase):
    """Test cases for the IGraphAPI class."""

    def setUp(self):
        """Set up test data."""
        # Create test data
        self.raster_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        # Create edge data
        self.from_nodes = np.array([0, 0, 1, 1, 2])
        self.to_nodes = np.array([1, 3, 2, 4, 5])
        self.cost = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Create API with real constructor
        self.api = IGraphAPI(
            self.raster_data,
            self.steps,
            from_nodes=self.from_nodes,
            to_nodes=self.to_nodes,
            cost=self.cost
        )

        # Replace real igraph with a mock
        self.mock_graph_instance = MagicMock()
        self.api.graph = self.mock_graph_instance

    def test_create_graph(self):
        """Test create_graph method."""
        # Mock igraph.Graph constructor
        with patch('igraph.Graph') as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph_cls.return_value = mock_graph

            # Create a test API instance
            IGraphAPI(
                self.raster_data,
                self.steps,
                from_nodes=self.from_nodes,
                to_nodes=self.to_nodes,
                cost=self.cost
            )

            # Verify Graph constructor was called
            mock_graph_cls.assert_called_once()

            # Verify add_edges was called
            mock_graph.add_edges.assert_called_once()

            # Verify weights were set
            self.assertEqual(mock_graph.es.__setitem__.call_count, 1)

    def test_create_graph_without_weights(self):
        """Test create_graph method with no weights."""
        with patch('igraph.Graph') as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph_cls.return_value = mock_graph

            # Create a test API instance with no cost
            IGraphAPI(
                self.raster_data,
                self.steps,
                from_nodes=self.from_nodes,
                to_nodes=self.to_nodes,
                cost=None
            )

            # Verify add_edges was called but weights were not set
            mock_graph.add_edges.assert_called_once()
            self.assertEqual(mock_graph.es.__setitem__.call_count, 0)

    def test_get_number_of_nodes_and_edges(self):
        """Test get_number_of_nodes and get_number_of_edges methods."""
        # Configure mocks to return specific values
        self.mock_graph_instance.vcount.return_value = 6
        self.mock_graph_instance.ecount.return_value = 5

        # Verify methods return expected values
        self.assertEqual(self.api.get_number_of_nodes(), 6)
        self.assertEqual(self.api.get_number_of_edges(), 5)

        # Verify correct graph methods were called
        self.mock_graph_instance.vcount.assert_called_once()
        self.mock_graph_instance.ecount.assert_called_once()

    def test_remove_isolates(self):
        """Test remove_isolates method."""
        # Create mock vertices with various degrees
        mock_vertices = []
        for i in range(5):
            vertex = MagicMock()
            vertex.index = i
            # Vertices 1 and 3 are isolated (degree 0)
            vertex.degree.return_value = 0 if i in [1, 3] else 2
            mock_vertices.append(vertex)

        # Configure mock graph
        self.mock_graph_instance.vs = mock_vertices

        # Call the method
        self.api.remove_isolates()

        # Verify delete_vertices was called for vertices with degree 0
        # Note: We expect them to be deleted in reverse order to avoid reindexing issues
        self.assertEqual(self.mock_graph_instance.delete_vertices.call_count, 2)

    def test_ensure_path_endpoints(self):
        """Test _ensure_path_endpoints static method."""
        # Test with source and target not in path
        self.assertEqual(
            IGraphAPI._ensure_path_endpoints([1, 2, 3], 0, 4),
            [0, 1, 2, 3, 4]
        )

        # Test with source in path but target not in path
        self.assertEqual(
            IGraphAPI._ensure_path_endpoints([0, 1, 2], 0, 3),
            [0, 1, 2, 3]
        )

        # Test with source not in path but target in path
        self.assertEqual(
            IGraphAPI._ensure_path_endpoints([1, 2, 3], 0, 3),
            [0, 1, 2, 3]
        )

    def test_get_nodes(self):
        """Test get_nodes method."""
        # Create mock vertices
        mock_vertices = []
        for i in range(6):
            vertex = MagicMock()
            vertex.index = i
            mock_vertices.append(vertex)

        # Set up the mock to return vertices
        self.mock_graph_instance.vs.return_value = mock_vertices

        # Call the method
        nodes = self.api.get_nodes()

        # Verify we get expected node indices
        self.assertEqual(nodes, [0, 1, 2, 3, 4, 5])

    def test_get_a_star_heuristic(self):
        """Test the get_a_star_heuristic method."""
        # Create mock vertices
        mock_vertices = []
        for i in range(6):  # Create 6 mock vertices with indices 0-5
            vertex = MagicMock()
            vertex.index = i
            mock_vertices.append(vertex)

        # Setup the mock graph to return these vertices
        self.mock_graph_instance.vs.return_value = mock_vertices

        # Mock get_nodes to return node indices
        with patch.object(self.api, 'get_nodes', return_value=[0, 1, 2, 3, 4, 5]):
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
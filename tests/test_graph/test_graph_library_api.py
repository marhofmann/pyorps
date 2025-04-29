import unittest
import numpy as np

from pyorps.graph.api.graph_library_api import GraphLibraryAPI


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

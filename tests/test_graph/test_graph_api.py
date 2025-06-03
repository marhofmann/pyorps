import unittest
import numpy as np

from pyorps.graph.api.graph_api import GraphAPI


class TestGraphAPI(unittest.TestCase):
    """Test cases for the GraphAPI abstract base class."""

    def test_initialization(self):
        """Test the initialization of the GraphAPI."""
        # Create a minimal concrete implementation for testing since GraphAPI is abstract
        class ConcreteGraphAPI(GraphAPI):
            def shortest_path(self, source_indices,
                              target_indices,
                              algorithm="dijkstra",
                              pairwise=False):  # pragma: no cover
                return [[0, 1, 2]]

        # Create test data
        raster_data = np.ones((10, 10), dtype=np.int32)
        steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        # Create instance and verify attributes were set correctly
        graph_api = ConcreteGraphAPI(raster_data, steps)
        self.assertTrue(np.array_equal(graph_api.raster_data, raster_data))
        self.assertTrue(np.array_equal(graph_api.steps, steps))



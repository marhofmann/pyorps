import unittest
import numpy as np
from shapely.geometry import LineString

from pyorps.core.path import Path, PathCollection


class TestPath(unittest.TestCase):
    def setUp(self):
        """Create a sample path for testing."""
        self.path = Path(
            source=1,
            target=2,
            algorithm="dijkstra",
            graph_api="networkx",
            path_indices=np.array([1, 3, 5, 2]),
            path_coords=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            path_geometry=LineString([(0, 0), (1, 1), (2, 2), (3, 3)]),
            euclidean_distance=4.24,
            runtimes={"preprocessing": 0.1, "pathfinding": 0.2},
            path_id=42,
            search_space_buffer_m=1,
            neighborhood="r0",
            total_length=5.0,
            total_cost=10.0,
            length_by_category={1.0: 2.5, 2.0: 2.5},
            length_by_category_percent={1.0: 0.5, 2.0: 0.5}
        )

    def test_path_initialization(self):
        """Test that Path initializes correctly."""
        self.assertEqual(self.path.path_id, 42)
        self.assertEqual(self.path.source, 1)
        self.assertEqual(self.path.target, 2)
        self.assertEqual(self.path.algorithm, "dijkstra")
        self.assertEqual(self.path.graph_api, "networkx")
        self.assertTrue(np.array_equal(self.path.path_indices, np.array([1, 3, 5, 2])))
        self.assertTrue(np.array_equal(self.path.path_coords, np.array([[0, 0], [1, 1], [2, 2], [3, 3]])))
        self.assertEqual(self.path.euclidean_distance, 4.24)
        self.assertEqual(self.path.total_length, 5.0)
        self.assertEqual(self.path.total_cost, 10.0)
        self.assertEqual(self.path.length_by_category, {1.0: 2.5, 2.0: 2.5})

    def test_path_to_geodataframe_dict(self):
        """Test conversion to GeoDataFrame dict."""
        result = self.path.to_geodataframe_dict()

        # Check that all expected keys are present
        expected_keys = [
            'runtime_preprocessing', 'runtime_pathfinding', 'path_id', 'source', 'target',
            'algorithm', 'graph_api', 'geometry', 'path_length', 'path_cost',
            'length_cost_1.0', 'length_cost_2.0', 'percent_cost_1.0', 'percent_cost_2.0'
        ]

        for key in expected_keys:
            self.assertIn(key, result)

        # Check specific values
        self.assertEqual(result['path_id'], 42)
        self.assertEqual(result['source'], "1")
        self.assertEqual(result['target'], "2")
        self.assertEqual(result['algorithm'], "dijkstra")
        self.assertEqual(result['path_length'], 5.0)

    def test_path_string_representation(self):
        """Test string representation of Path."""
        str_repr = str(self.path)
        self.assertIn("Path(id=42", str_repr)
        self.assertIn("length_m=5.00", str_repr)
        self.assertIn("cost=10.00", str_repr)

        repr_str = repr(self.path)
        self.assertIn("Path(id=42", repr_str)
        self.assertIn("length_m=5.00", repr_str)
        self.assertIn("cost=10.00", repr_str)


class TestPathCollection(unittest.TestCase):
    def setUp(self):
        """Set up test paths and collection."""
        self.collection = PathCollection()

        # Create test paths
        self.path1 = Path(
            source=1, target=2, algorithm="dijkstra", graph_api="networkx",
            path_indices=np.array([]), path_coords=np.array([]),
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.0, runtimes={}, path_id=None, search_space_buffer_m=3, neighborhood="R0"
        )

        self.path2 = Path(
            source=2, target=3, algorithm="astar", graph_api="networkx",
            path_indices=np.array([]), path_coords=np.array([]),
            path_geometry=LineString([(1, 1), (2, 2)]),
            euclidean_distance=1.0, runtimes={}, path_id=5, search_space_buffer_m=3, neighborhood="R0"
        )

        # Create another path with explicit ID for replace tests
        self.path3 = Path(
            source=3, target=4, algorithm="bellman-ford", graph_api="networkx",
            path_indices=np.array([]), path_coords=np.array([]),
            path_geometry=LineString([(2, 2), (3, 3)]),
            euclidean_distance=1.0, runtimes={}, path_id=10, search_space_buffer_m=3, neighborhood="R0"
        )

    def test_add_path_default(self):
        """Test adding paths to collection with default behavior (replace=False)."""
        # Add paths
        self.collection.add(self.path1)
        self.collection.add(self.path2)

        # Test automatic ID assignment
        self.assertEqual(self.path1.path_id, 0)
        self.assertEqual(self.path2.path_id, 1)  # Should get a new ID even though it had path_id=5

        # The next available ID should be 2
        self.assertEqual(self.collection._next_id, 2)

        # Test length
        self.assertEqual(len(self.collection), 2)

    def test_add_path_with_replace_true(self):
        """Test adding paths with replace=True."""
        # Add path with no ID and replace=True (should behave like default)
        self.collection.add(self.path1, replace=True)
        self.assertEqual(self.path1.path_id, 0)
        self.assertEqual(self.collection._next_id, 1)

        # Add path with ID and replace=True (should keep the original ID)
        self.collection.add(self.path2, replace=True)
        self.assertEqual(self.path2.path_id, 5)

        # The next available ID should be updated based on the highest ID + 1
        self.assertEqual(self.collection._next_id, 6)

        # Add another path with higher ID and replace=True
        self.collection.add(self.path3, replace=True)
        self.assertEqual(self.path3.path_id, 10)
        self.assertEqual(self.collection._next_id, 11)

        # Test length
        self.assertEqual(len(self.collection), 3)

    def test_add_path_replace_and_update(self):
        """Test replacing an existing path."""
        # Add initial path
        self.collection.add(self.path1)
        self.assertEqual(self.path1.path_id, 0)

        # Create a new path with the same ID to replace the existing one
        path_replacement = Path(
            source=99, target=100, algorithm="modified", graph_api="networkx",
            path_indices=np.array([]), path_coords=np.array([]),
            path_geometry=LineString([(5, 5), (6, 6)]),
            euclidean_distance=2.0, runtimes={}, path_id=0, search_space_buffer_m=2, neighborhood="R0"
        )

        # Add the replacement path with replace=True
        self.collection.add(path_replacement, replace=True)

        # Check that path was replaced
        self.assertEqual(self.collection[0].source, 99)
        self.assertEqual(self.collection[0].target, 100)
        self.assertEqual(len(self.collection), 1)

        # The next_id should still be 1
        self.assertEqual(self.collection._next_id, 1)

    def test_get_path(self):
        """Test retrieving paths from collection."""
        self.collection.add(self.path1)
        self.collection.add(self.path2, replace=True)

        # Get by ID
        self.assertEqual(self.collection.get(path_id=0), self.path1)
        self.assertEqual(self.collection.get(path_id=5), self.path2)

        # Get by source/target
        self.assertEqual(self.collection.get(source=1, target=2), self.path1)
        self.assertEqual(self.collection.get(source=2, target=3), self.path2)

        # Non-existent path
        self.assertIsNone(self.collection.get(path_id=99))
        self.assertIsNone(self.collection.get(source=99, target=99))

    def test_iteration(self):
        """Test iterating through the collection."""
        self.collection.add(self.path1)
        self.collection.add(self.path2, replace=True)

        paths = list(self.collection)
        self.assertEqual(len(paths), 2)
        self.assertIn(self.path1, paths)
        self.assertIn(self.path2, paths)

    def test_getitem(self):
        """Test accessing paths by ID."""
        self.collection.add(self.path1)
        self.collection.add(self.path2, replace=True)

        self.assertEqual(self.collection[0], self.path1)
        self.assertEqual(self.collection[5], self.path2)

    def test_to_geodataframe_records(self):
        """Test conversion to GeoDataFrame records."""
        self.path1.total_length = 1.5
        self.path1.total_cost = 3.0
        self.path1.length_by_category = {1.0: 1.5}
        self.path1.length_by_category_percent = {1.0: 1.0}
        self.collection.add(self.path1)

        records = self.collection.to_geodataframe_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['path_id'], 0)
        self.assertEqual(records[0]['path_length'], 1.5)

    def test_string_representation(self):
        """Test string representation of PathCollection."""
        self.collection.add(self.path1)
        self.collection.add(self.path2, replace=True)

        str_repr = str(self.collection)
        self.assertIn("PathCollection(count=2)", str_repr)

        repr_str = repr(self.collection)
        self.assertIn("PathCollection(paths=[", repr_str)
        self.assertIn("count=2", repr_str)

    def test_all_property(self):
        """Test the all property."""
        self.collection.add(self.path1)
        self.collection.add(self.path2, replace=True)

        all_paths = self.collection.all
        self.assertEqual(len(all_paths), 2)
        self.assertIn(self.path1, all_paths)
        self.assertIn(self.path2, all_paths)


if __name__ == "__main__":
    unittest.main()
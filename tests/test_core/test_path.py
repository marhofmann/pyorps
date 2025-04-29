import unittest
import numpy as np
from shapely.geometry import LineString
import geopandas as gpd

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
        self.assertIn("length=5.00", str_repr)
        self.assertIn("cost=10.00", str_repr)

        repr_str = repr(self.path)
        self.assertIn("Path(path_id=42", repr_str)
        self.assertIn("total_length=5.00", repr_str)
        self.assertIn("total_cost=10.00", repr_str)


class TestPathCollection(unittest.TestCase):
    def setUp(self):
        """Set up test paths and collection."""
        self.collection = PathCollection()

        # Create test paths
        self.path1 = Path(
            source=1, target=2, algorithm="dijkstra", graph_api="networkx",
            path_indices=np.array([]), path_coords=np.array([]),
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.0, runtimes={}, path_id=None
        )

        self.path2 = Path(
            source=2, target=3, algorithm="astar", graph_api="networkx",
            path_indices=np.array([]), path_coords=np.array([]),
            path_geometry=LineString([(1, 1), (2, 2)]),
            euclidean_distance=1.0, runtimes={}, path_id=5
        )

    def test_add_path(self):
        """Test adding paths to collection."""
        # Add paths
        self.collection.add(self.path1)
        self.collection.add(self.path2)

        # Test automatic ID assignment
        self.assertEqual(self.path1.path_id, 0)
        self.assertEqual(self.path2.path_id, 5)

        # Test length
        self.assertEqual(len(self.collection), 2)

    def test_get_path(self):
        """Test retrieving paths from collection."""
        self.collection.add(self.path1)
        self.collection.add(self.path2)

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
        self.collection.add(self.path2)

        paths = list(self.collection)
        self.assertEqual(len(paths), 2)
        self.assertIn(self.path1, paths)
        self.assertIn(self.path2, paths)

    def test_getitem(self):
        """Test accessing paths by ID."""
        self.collection.add(self.path1)
        self.collection.add(self.path2)

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
        self.collection.add(self.path2)

        str_repr = str(self.collection)
        self.assertIn("PathCollection(paths=2)", str_repr)

        repr_str = repr(self.collection)
        self.assertIn("PathCollection(paths=[", repr_str)
        self.assertIn("count=2", repr_str)

    def test_all_property(self):
        """Test the all property."""
        self.collection.add(self.path1)
        self.collection.add(self.path2)

        all_paths = self.collection.all
        self.assertEqual(len(all_paths), 2)
        self.assertIn(self.path1, all_paths)
        self.assertIn(self.path2, all_paths)

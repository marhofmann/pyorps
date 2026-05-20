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

    def test_path_simplification_fields_default_none(self):
        from shapely.geometry import LineString
        from pyorps.core.path import Path
        p = Path(
            source=(0, 0),
            target=(1, 1),
            algorithm="dijkstra",
            graph_api="cython",
            path_indices=[0, 1],
            path_coords=[(0.0, 0.0), (1.0, 1.0)],
            path_geometry=LineString([(0.0, 0.0), (1.0, 1.0)]),
            euclidean_distance=1.4142135,
            runtimes={},
            path_id=0,
            search_space_buffer_m=0.0,
            neighborhood="8",
        )
        assert p.simplification_method is None
        assert p.simplification_tolerance is None
        assert p.original_path_geometry is None

    def test_path_simplification_fields_set(self):
        from shapely.geometry import LineString
        from pyorps.core.path import Path
        orig = LineString([(0, 0), (0.5, 0), (1, 0)])
        simp = LineString([(0, 0), (1, 0)])
        p = Path(
            source=(0, 0), target=(1, 0),
            algorithm="dijkstra", graph_api="cython",
            path_indices=[0, 1], path_coords=[(0.0, 0.0), (1.0, 0.0)],
            path_geometry=simp,
            euclidean_distance=1.0,
            runtimes={}, path_id=0,
            search_space_buffer_m=0.0, neighborhood="8",
            simplification_method="douglas_peucker",
            simplification_tolerance=0.1,
            original_path_geometry=orig,
        )
        assert p.simplification_method == "douglas_peucker"
        assert p.simplification_tolerance == 0.1
        assert p.original_path_geometry.equals(orig)

    def test_to_geodataframe_dict_includes_simplification(self):
        from shapely.geometry import LineString
        from pyorps.core.path import Path
        p = Path(
            source=(0, 0), target=(1, 0),
            algorithm="dijkstra", graph_api="cython",
            path_indices=[0, 1], path_coords=[(0.0, 0.0), (1.0, 0.0)],
            path_geometry=LineString([(0, 0), (1, 0)]),
            euclidean_distance=1.0,
            runtimes={}, path_id=0,
            search_space_buffer_m=0.0, neighborhood="8",
            simplification_method="visvalingam",
            simplification_tolerance=2.5,
        )
        d = p.to_geodataframe_dict()
        assert d["simplification_method"] == "visvalingam"
        assert d["simplification_tolerance"] == 2.5

    def test_analyze_mentions_simplification_when_present(self):
        from shapely.geometry import LineString
        from pyorps.core.path import Path
        p = Path(
            source=(0, 0), target=(1, 0),
            algorithm="dijkstra", graph_api="cython",
            path_indices=[0, 1], path_coords=[(0.0, 0.0), (1.0, 0.0)],
            path_geometry=LineString([(0, 0), (1, 0)]),
            euclidean_distance=1.0,
            runtimes={}, path_id=0,
            search_space_buffer_m=0.0, neighborhood="8",
            simplification_method="visvalingam",
            simplification_tolerance=4.0,
            original_path_geometry=LineString([(0, 0), (0.5, 0.1), (1, 0)]),
            total_length=1.0,
            length_by_category={1.0: 1.0},
            length_by_category_percent={1.0: 100.0},
        )
        output = p.analyze()
        assert "Simplification" in output
        assert "visvalingam" in output
        assert "4.0" in output

    def test_path_str_shows_runtime_not_cost(self):
        """P1.5: __str__ must show runtimes['runtime_total'], not total_cost."""
        path = Path(
            source=(0, 0), target=(1, 1), algorithm="dijkstra",
            graph_api="cython", path_indices=np.array([0, 1]),
            path_coords=np.array([[0, 0], [1, 1]]),
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.41, path_id=0,
            search_space_buffer_m=100, neighborhood="r2",
            runtimes={"runtime_total": 1.23},
            total_length=5.0, total_cost=99.0
        )
        s = str(path)
        self.assertIn("runtime_total=1.23", s)
        self.assertNotIn("runtime_total=99", s)


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
        self.assertIn("count=2)", str_repr)

        repr_str = repr(self.collection)
        self.assertIn("PathCollection(paths=[", repr_str)
        self.assertIn("count=2", repr_str)

    def test_str_shows_path_objects_not_keys(self):
        """P1.6: __str__ and __repr__ must iterate Path objects, not dict keys."""
        self.collection.add(self.path1)
        s = str(self.collection)
        # Should contain Path representation, not bare integer "0"
        self.assertIn("Path(", s)

        r = repr(self.collection)
        self.assertIn("Path(", r)

    def test_all_property(self):
        """Test the all property."""
        self.collection.add(self.path1)
        self.collection.add(self.path2, replace=True)

        all_paths = self.collection.all
        self.assertEqual(len(all_paths), 2)
        self.assertIn(self.path1, all_paths)
        self.assertIn(self.path2, all_paths)


class TestPathEquality(unittest.TestCase):
    """Test Path.__eq__ edge cases."""

    def setUp(self):
        self.base_kwargs = dict(
            algorithm="dijkstra", graph_api="networkx",
            path_indices=np.array([1, 3, 5]),
            path_coords=np.array([[0, 0], [1, 1], [2, 2]]),
            path_geometry=LineString([(0, 0), (1, 1), (2, 2)]),
            euclidean_distance=2.83, runtimes={}, path_id=0,
            search_space_buffer_m=100, neighborhood="r2"
        )

    def test_equal_paths(self):
        """Two paths with identical fields are equal."""
        p1 = Path(source=(1, 2), target=(3, 4), **self.base_kwargs)
        p2 = Path(source=(1, 2), target=(3, 4), **self.base_kwargs)
        self.assertTrue(p1 == p2)

    def test_numpy_array_source_target(self):
        """Equality works when source/target are numpy arrays."""
        p1 = Path(source=np.array([1, 2]), target=np.array([3, 4]),
                   **self.base_kwargs)
        p2 = Path(source=np.array([1, 2]), target=np.array([3, 4]),
                   **self.base_kwargs)
        self.assertTrue(p1 == p2)

    def test_different_indices_not_equal(self):
        """Paths with different path_indices are not equal."""
        p1 = Path(source=(1, 2), target=(3, 4), **self.base_kwargs)
        kwargs2 = {**self.base_kwargs, "path_indices": np.array([1, 99, 5])}
        p2 = Path(source=(1, 2), target=(3, 4), **kwargs2)
        self.assertFalse(p1 == p2)

    def test_different_neighborhood_not_equal(self):
        """Paths with different neighborhoods are not equal."""
        p1 = Path(source=(1, 2), target=(3, 4), **self.base_kwargs)
        kwargs2 = {**self.base_kwargs, "neighborhood": "r0"}
        p2 = Path(source=(1, 2), target=(3, 4), **kwargs2)
        self.assertFalse(p1 == p2)


class TestPathCollectionEquality(unittest.TestCase):
    """Test PathCollection.__eq__ edge cases."""

    def _make_path(self, source, target, indices=None):
        return Path(
            source=source, target=target, algorithm="dijkstra",
            graph_api="networkx",
            path_indices=indices if indices is not None else np.array([0, 1]),
            path_coords=np.array([[0, 0], [1, 1]]),
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.0, runtimes={}, path_id=None,
            search_space_buffer_m=100, neighborhood="r2"
        )

    def test_different_order_equal(self):
        """Collections with same paths in different order are equal."""
        c1 = PathCollection()
        c2 = PathCollection()
        p1 = self._make_path(1, 2)
        p2 = self._make_path(3, 4, np.array([2, 3]))

        c1.add(self._make_path(1, 2))
        c1.add(self._make_path(3, 4, np.array([2, 3])))

        c2.add(self._make_path(3, 4, np.array([2, 3])))
        c2.add(self._make_path(1, 2))

        self.assertTrue(c1 == c2)

    def test_superset_asymmetry(self):
        """A superset collection is equal to subset from subset's perspective but
        the check is 'all of other in self', so superset==subset is True but
        subset==superset would fail if superset has extra paths."""
        c1 = PathCollection()
        c2 = PathCollection()

        c1.add(self._make_path(1, 2))
        c1.add(self._make_path(3, 4, np.array([2, 3])))

        c2.add(self._make_path(1, 2))

        # c1 == c2 checks all of c2.all are in c1 → True
        self.assertTrue(c1 == c2)
        # c2 == c1 checks all of c1.all are in c2 → False (missing path 3→4)
        self.assertFalse(c2 == c1)


class TestPathCollectionStr(unittest.TestCase):
    """Test PathCollection __str__ and __repr__ truncation."""

    def _make_path(self, pid):
        return Path(
            source=pid, target=pid + 1, algorithm="dijkstra",
            graph_api="networkx",
            path_indices=np.array([pid]),
            path_coords=np.array([[0, 0]]),
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.0, runtimes={}, path_id=pid,
            search_space_buffer_m=100, neighborhood="r2"
        )

    def test_empty_collection_str(self):
        c = PathCollection()
        s = str(c)
        self.assertIn("count=0", s)

    def test_small_collection_str(self):
        c = PathCollection()
        c.add(self._make_path(0), replace=True)
        s = str(c)
        self.assertIn("count=1", s)

    def test_large_collection_str_truncation(self):
        """Collections with >5 paths show first 2 and last."""
        c = PathCollection()
        for i in range(7):
            c.add(self._make_path(i), replace=True)
        s = str(c)
        self.assertIn("...", s)
        self.assertIn("count=7", s)

    def test_large_collection_repr_truncation(self):
        c = PathCollection()
        for i in range(7):
            c.add(self._make_path(i), replace=True)
        r = repr(c)
        self.assertIn("...", r)
        self.assertIn("count=7", r)


class TestPathCollectionAddReplace(unittest.TestCase):
    """Test PathCollection.add with replace=True and _next_id tracking."""

    def _make_path(self, pid):
        return Path(
            source=pid, target=pid + 1, algorithm="dijkstra",
            graph_api="networkx",
            path_indices=np.array([pid]),
            path_coords=np.array([[0, 0]]),
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.0, runtimes={}, path_id=pid,
            search_space_buffer_m=100, neighborhood="r2"
        )

    def test_replace_true_keeps_id_and_updates_next_id(self):
        c = PathCollection()
        p = self._make_path(10)
        c.add(p, replace=True)
        self.assertEqual(p.path_id, 10)
        self.assertEqual(c._next_id, 11)

    def test_replace_false_overrides_id(self):
        c = PathCollection()
        p = self._make_path(10)
        c.add(p, replace=False)
        self.assertEqual(p.path_id, 0)
        self.assertEqual(c._next_id, 1)

    def test_replace_existing_path(self):
        c = PathCollection()
        p1 = self._make_path(0)
        c.add(p1)

        p2 = self._make_path(0)
        p2.source = 99
        c.add(p2, replace=True)

        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].source, 99)


class TestPathCostSemantics(unittest.TestCase):
    """P4.11: total_cost (distance-weighted) and total_cell_cost (raw cell sum)."""

    def test_total_cell_cost_field_exists(self):
        """Path should have total_cell_cost field, defaulting to None."""
        path = Path(
            source=(0, 0), target=(1, 1), algorithm="dijkstra",
            graph_api="cython", path_indices=np.array([0, 1]),
            path_coords=[(0, 0), (1, 1)],
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.41, runtimes={}, path_id=0,
            search_space_buffer_m=100, neighborhood="r2"
        )
        self.assertIsNone(path.total_cell_cost)

    def test_total_cell_cost_in_geodataframe_dict(self):
        """total_cell_cost should appear in GeoDataFrame export."""
        path = Path(
            source=(0, 0), target=(1, 1), algorithm="dijkstra",
            graph_api="cython", path_indices=np.array([0, 1]),
            path_coords=[(0, 0), (1, 1)],
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.41, runtimes={}, path_id=0,
            search_space_buffer_m=100, neighborhood="r2",
            total_length=1.41, total_cost=5.0, total_cell_cost=10.0,
            length_by_category={5: 1.0}, length_by_category_percent={5: 100}
        )
        gdf_dict = path.to_geodataframe_dict()
        self.assertEqual(gdf_dict["path_cell_cost"], 10.0)

    def test_total_cost_and_cell_cost_differ(self):
        """total_cost (weighted) and total_cell_cost (raw) should differ."""
        path = Path(
            source=(0, 0), target=(1, 1), algorithm="dijkstra",
            graph_api="cython", path_indices=np.array([0, 1]),
            path_coords=[(0, 0), (1, 1)],
            path_geometry=LineString([(0, 0), (1, 1)]),
            euclidean_distance=1.41, runtimes={}, path_id=0,
            search_space_buffer_m=100, neighborhood="r2",
            total_length=10.0, total_cost=50.0, total_cell_cost=10.0
        )
        self.assertNotEqual(path.total_cost, path.total_cell_cost)


if __name__ == "__main__":
    unittest.main()
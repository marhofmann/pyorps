import unittest
import numpy as np

from pyorps.graph.api.cython_api import CythonAPI
from pyorps.core.exceptions import (
    AlgorithmNotImplementedError, PairwiseError
)


class TestCythonAPIInit(unittest.TestCase):
    """Test CythonAPI initialization."""

    def setUp(self):
        self.raster = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]], dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)

    def test_ignore_max_true(self):
        api = CythonAPI(self.raster, self.steps, ignore_max=True)
        self.assertTrue(api.ignore_max)
        self.assertEqual(api.max_value, 65535)

    def test_ignore_max_false(self):
        from pyorps.utils._raster_context import NO_EXCLUSION_VALUE

        api = CythonAPI(self.raster, self.steps, ignore_max=False)
        self.assertFalse(api.ignore_max)
        # Out-of-uint16-domain sentinel: nothing is excluded. Using 0 here made
        # every zero-cost cell impassable.
        self.assertEqual(api.max_value, NO_EXCLUSION_VALUE)

    def test_ignore_max_default_matches_base_class(self):
        """Regression: CythonAPI default for ignore_max must match GraphAPI (True)."""
        from pyorps.graph.api.graph_api import GraphAPI
        import inspect

        # Get the default value from both classes
        base_sig = inspect.signature(GraphAPI.__init__)
        cython_sig = inspect.signature(CythonAPI.__init__)

        base_default = base_sig.parameters['ignore_max'].default
        cython_default = cython_sig.parameters['ignore_max'].default

        self.assertEqual(
            cython_default, base_default,
            f"CythonAPI ignore_max default ({cython_default}) "
            f"does not match GraphAPI default ({base_default})"
        )

        # Also verify the default is True when constructed without explicit ignore_max
        api = CythonAPI(self.raster, self.steps)
        self.assertTrue(api.ignore_max)
        self.assertEqual(api.max_value, 65535)

    def test_stores_raster_and_steps(self):
        api = CythonAPI(self.raster, self.steps, ignore_max=True)
        np.testing.assert_array_equal(api.raster_data, self.raster)
        np.testing.assert_array_equal(api.steps, self.steps)


class TestCythonAPIToArray(unittest.TestCase):
    """Test CythonAPI._to_array conversion."""

    def setUp(self):
        self.raster = np.ones((5, 5), dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0]], dtype=np.int8)
        self.api = CythonAPI(self.raster, self.steps, ignore_max=True)

    def test_int_input(self):
        result = self.api._to_array(42)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result[0], 42)
        self.assertEqual(result.dtype, np.uint32)

    def test_list_input(self):
        result = self.api._to_array([1, 2, 3])
        self.assertEqual(result.shape, (3,))
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_ndarray_input(self):
        arr = np.array([10, 20], dtype=np.int64)
        result = self.api._to_array(arr)
        self.assertEqual(result.shape, (2,))
        np.testing.assert_array_equal(result, [10, 20])

    def test_0d_array(self):
        arr = np.array(7)
        result = self.api._to_array(arr)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result[0], 7)

    def test_uint64_overflow(self):
        """Values exceeding uint32 max should use uint64."""
        big = np.iinfo(np.uint32).max + 1
        result = self.api._to_array(big)
        self.assertEqual(result.dtype, np.uint64)
        self.assertEqual(result[0], big)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            self.api._to_array("invalid")


class TestCythonAPIShortestPathRouting(unittest.TestCase):
    """Test CythonAPI.shortest_path algorithm validation and routing dispatch."""

    def setUp(self):
        self.raster = np.ones((5, 5), dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
        self.api = CythonAPI(self.raster, self.steps, ignore_max=True)

    def test_invalid_algorithm_raises(self):
        with self.assertRaises(AlgorithmNotImplementedError):
            self.api.shortest_path(0, 24, algorithm="bellman-ford")


class TestCythonAPISinglePath(unittest.TestCase):
    """Test CythonAPI single source-target path computation."""

    def setUp(self):
        # 5x5 uniform cost raster
        self.raster = np.ones((5, 5), dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
        self.api = CythonAPI(self.raster, self.steps, ignore_max=True)

    def test_dijkstra_single_path(self):
        path = self.api.shortest_path(0, 24, algorithm="dijkstra")
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 24)

    def test_delta_stepping_single_path(self):
        path = self.api.shortest_path(0, 24, algorithm="delta-stepping")
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 24)


class TestCythonAPISingleToMulti(unittest.TestCase):
    """Test CythonAPI single source to multiple targets."""

    def setUp(self):
        self.raster = np.ones((5, 5), dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
        self.api = CythonAPI(self.raster, self.steps, ignore_max=True)

    def test_dijkstra_single_to_multi(self):
        paths = self.api.shortest_path(0, [4, 24], algorithm="dijkstra")
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0][0], 0)
        self.assertEqual(paths[0][-1], 4)
        self.assertEqual(paths[1][0], 0)
        self.assertEqual(paths[1][-1], 24)

    def test_delta_stepping_single_to_multi(self):
        paths = self.api.shortest_path(0, [4, 24], algorithm="delta-stepping")
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0][-1], 4)
        self.assertEqual(paths[1][-1], 24)


class TestCythonAPIPairwisePaths(unittest.TestCase):
    """Test CythonAPI pairwise path computation."""

    def setUp(self):
        self.raster = np.ones((5, 5), dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
        self.api = CythonAPI(self.raster, self.steps, ignore_max=True)

    def test_dijkstra_pairwise(self):
        paths = self.api.shortest_path(
            [0, 5], [4, 9], algorithm="dijkstra", pairwise=True)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0][0], 0)
        self.assertEqual(paths[0][-1], 4)
        self.assertEqual(paths[1][0], 5)
        self.assertEqual(paths[1][-1], 9)

    def test_pairwise_length_mismatch_raises(self):
        with self.assertRaises(PairwiseError):
            self.api.shortest_path(
                [0, 5, 10], [4, 9], algorithm="dijkstra", pairwise=True)


class TestCythonAPIMultiToMulti(unittest.TestCase):
    """Test CythonAPI multi source to multi target (all pairs)."""

    def setUp(self):
        self.raster = np.ones((5, 5), dtype=np.uint16)
        self.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
        self.api = CythonAPI(self.raster, self.steps, ignore_max=True)

    def test_dijkstra_multi_to_multi(self):
        paths = self.api.shortest_path(
            [0, 5], [4, 9], algorithm="dijkstra", pairwise=False)
        # 2 sources x 2 targets = 4 paths
        self.assertEqual(len(paths), 4)

    def test_delta_stepping_multi_to_multi(self):
        paths = self.api.shortest_path(
            [0, 5], [4, 9], algorithm="delta-stepping", pairwise=False)
        self.assertEqual(len(paths), 4)


if __name__ == "__main__":
    unittest.main()

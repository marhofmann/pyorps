# test_utils/test_traversal.py
import unittest
import numpy as np

from pyorps.utils.traversal import (
    intermediate_steps_numba, get_cost_factor_numba, ravel_index,
    calculate_region_bounds, calculate_segment_length,
    get_max_number_of_edges, calculate_path_metrics_numba,
    euclidean_distances_numba
)


class TestTraversalBasics(unittest.TestCase):
    """Test basic traversal utility functions."""

    def test_intermediate_steps_numba(self):
        """Test intermediate_steps_numba function."""
        # For a step (2,1) we should get intermediate steps
        result = intermediate_steps_numba(2, 1)

        # Check shape and type
        self.assertEqual(result.dtype, np.int8)
        self.assertEqual(result.shape[1], 2)  # Should be 2D coordinates

        # We should have (2-1)*2 = 2 steps for this case, each with 2 options
        self.assertEqual(result.shape[0], 2)

        # For a step (1,0) there should be no intermediate steps
        result = intermediate_steps_numba(1, 0)
        self.assertEqual(result.shape[0], 0)

    def test_get_cost_factor_numba(self):
        """Test get_cost_factor_numba function."""
        # For a step (1,0) with no intermediates, factor should be distance/2
        result = get_cost_factor_numba(1, 0, 0)
        self.assertAlmostEqual(result, 0.5)

        # For a step (3,4) with 2 intermediates, factor should be distance/4
        result = get_cost_factor_numba(3, 4, 2)
        distance = np.sqrt(3 ** 2 + 4 ** 2)
        self.assertAlmostEqual(result, distance / 4)

    def test_ravel_index(self):
        """Test ravel_index function."""
        # Check conversion from 2D to 1D indices
        result = ravel_index(2, 3, 10)
        self.assertEqual(result, 23)  # 2*10 + 3

        result = ravel_index(0, 0, 5)
        self.assertEqual(result, 0)

        result = ravel_index(5, 6, 7)
        self.assertEqual(result, 41)  # 5*7 + 6

    def test_calculate_region_bounds(self):
        """Test calculate_region_bounds function."""
        # For step (1,1) and grid size 5x5
        bounds = calculate_region_bounds(1, 1, 5, 5)

        # Unpack the 8 values
        (s_rows_start, s_rows_end, s_cols_start, s_cols_end,
         t_rows_start, t_rows_end, t_cols_start, t_cols_end) = bounds

        # Source region should be (0,0) to (4,4)
        self.assertEqual(s_rows_start, 0)
        self.assertEqual(s_rows_end, 4)
        self.assertEqual(s_cols_start, 0)
        self.assertEqual(s_cols_end, 4)

        # Target region should be (1,1) to (5,5)
        self.assertEqual(t_rows_start, 1)
        self.assertEqual(t_rows_end, 5)
        self.assertEqual(t_cols_start, 1)
        self.assertEqual(t_cols_end, 5)

    def test_calculate_segment_length(self):
        """Test calculate_segment_length function."""
        # Straight line case
        length = calculate_segment_length(0, 1)
        self.assertAlmostEqual(length, 1.0)

        # Diagonal case
        length = calculate_segment_length(1, 1)
        self.assertAlmostEqual(length, 1.4142135623730951)  # sqrt(2)

        # Knight's move case
        length = calculate_segment_length(2, 1)
        self.assertAlmostEqual(length, 2.236067977499789)  # sqrt(5)

        # Another special case
        length = calculate_segment_length(3, 1)
        self.assertAlmostEqual(length, 3.1622776601683795)  # sqrt(10)

        # General case
        length = calculate_segment_length(4, 7)
        self.assertAlmostEqual(length, np.sqrt(4 ** 2 + 7 ** 2))


class TestTraversalAdvanced(unittest.TestCase):
    """Test advanced traversal utility functions with full grid setup."""

    def setUp(self):
        """Set up test grid and other necessary structures."""
        # Create a 5x5 cost grid
        self.raster = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint16)

        # Define steps for k=1 neighborhood
        self.steps = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1],  # cardinal
            [1, 1], [-1, 1], [1, -1], [-1, -1]  # diagonal
        ], dtype=np.int8)

    def test_get_max_number_of_edges(self):
        """Test get_max_number_of_edges function."""
        # For a 5x5 grid with k=1 neighborhood
        max_edges = get_max_number_of_edges(5, 5, self.steps)

        # Calculate expected number:
        # - 4 cardinal directions: each can connect (5-1)*5 = 20 cells
        # - 4 diagonal directions: each can connect (5-1)*(5-1) = 16 cells
        expected = 4 * 20 + 4 * 16
        self.assertEqual(max_edges, expected)

    def test_calculate_path_metrics_numba(self):
        """Test calculate_path_metrics_numba function."""
        # Create a simple path through the grid
        path_indices = np.array([0, 6, 12, 18, 24], dtype=np.uint32)  # Diagonal path

        # Calculate path metrics
        total_length, categories, lengths = calculate_path_metrics_numba(self.raster, path_indices)

        # Check total length
        # Path is 4 diagonal segments, each of length sqrt(2)
        expected_length = 4 * np.sqrt(2)
        self.assertAlmostEqual(total_length, expected_length)

        # Check categories detected
        self.assertTrue(np.array_equal(np.sort(categories), np.array([1, 2, 3])))

        # Check total distributed length
        total_distributed = sum(lengths)
        self.assertAlmostEqual(total_distributed, expected_length)

    def test_euclidean_distances_numba(self):
        """Test euclidean_distances_numba function."""
        # Create a set of 2D points
        points = np.array([
            [0, 0],
            [3, 4],
            [1, 1],
            [5, 12]
        ], dtype=np.float64)

        # Calculate distances to target (0,0)
        target = np.array([0, 0], dtype=np.float64)
        distances = euclidean_distances_numba(points, target)

        # Check results
        expected = np.array([0, 5, np.sqrt(2), 13])
        np.testing.assert_almost_equal(distances, expected)

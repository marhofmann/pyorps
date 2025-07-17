# test_utils/test_traversal.py
import unittest
import numpy as np

from pyorps.utils.traversal import (
    intermediate_steps_numba, get_cost_factor_numba, ravel_index,
    calculate_region_bounds, calculate_segment_length,
    get_max_number_of_edges, calculate_path_metrics_numba,
    euclidean_distances_numba, is_valid_node, find_valid_nodes,
    construct_edges, get_outgoing_edges
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


class TestTraversalHelpers(unittest.TestCase):
    """Test helper functions for traversal."""

    def setUp(self):
        """Set up test data for helper function tests."""
        # Create a simple 3x3 raster
        self.raster = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ], dtype=np.uint16)

        # Create exclusion mask (all cells valid)
        self.exclude_mask = np.ones((3, 3), dtype=np.uint8)

        # Simple 4-connected neighborhood
        self.steps = np.array([
            [1, 0],  # Down
            [0, 1],  # Right
            [-1, 0],  # Up
            [0, -1]  # Left
        ], dtype=np.int8)

        self.rows, self.cols = self.raster.shape

    def test_is_valid_node(self):
        """Test is_valid_node function."""
        # No intermediate steps
        intermediates = np.zeros((0, 2), dtype=np.int8)

        # For valid source and target
        out_cost = np.zeros(1, dtype=np.float64)
        result = is_valid_node(0, 0, 1, 1, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertTrue(result)
        self.assertEqual(out_cost[0], self.raster[0, 0] + self.raster[1, 1])  # Source + target

        # For invalid source (out of bounds)
        result = is_valid_node(-1, 0, 1, 1, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertFalse(result)

        # For invalid target (out of bounds)
        result = is_valid_node(0, 0, 3, 3, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertFalse(result)

        # For excluded source
        self.exclude_mask[0, 0] = 0
        result = is_valid_node(0, 0, 1, 1, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertFalse(result)
        self.exclude_mask[0, 0] = 1  # Restore

        # For excluded target
        self.exclude_mask[1, 1] = 0
        result = is_valid_node(0, 0, 1, 1, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertFalse(result)
        self.exclude_mask[1, 1] = 1  # Restore

        # With intermediate steps
        intermediates = np.array([[0, 1], [1, 0]], dtype=np.int8)  # Two intermediates
        result = is_valid_node(0, 0, 1, 1, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertTrue(result)
        # Cost should be source + target + intermediates
        expected_cost = self.raster[0, 0] + self.raster[1, 1] + self.raster[0, 1] + self.raster[1, 0]
        self.assertEqual(out_cost[0], expected_cost)

        # With invalid intermediate
        self.exclude_mask[0, 1] = 0
        result = is_valid_node(0, 0, 1, 1, self.exclude_mask, intermediates,
                               self.raster, self.rows, self.cols, out_cost)
        self.assertFalse(result)
        self.exclude_mask[0, 1] = 1  # Restore

    def test_find_valid_nodes(self):
        """Test find_valid_nodes function."""
        # Test for step (1, 0) - move down
        dr, dc = 1, 0
        intermediates = intermediate_steps_numba(np.int8(dr), np.int8(dc))
        cost_factor = get_cost_factor_numba(np.int8(dr), np.int8(dc), intermediates.shape[0])

        # Source region is top two rows, all columns
        s_rows_start, s_rows_end = 0, 2
        s_cols_start, s_cols_end = 0, 3

        from_nodes, to_nodes, costs, valid_count = find_valid_nodes(
            np.int8(dr), np.int8(dc), s_rows_start, s_rows_end, s_cols_start, s_cols_end,
            self.exclude_mask, self.raster, intermediates, self.rows, self.cols, cost_factor, 100
        )

        # Should find 6 valid nodes (2 rows x 3 cols)
        self.assertEqual(valid_count, 6)

        # Check first node (0,0) -> (1,0)
        self.assertEqual(from_nodes[0], ravel_index(0, 0, self.cols))
        self.assertEqual(to_nodes[0], ravel_index(1, 0, self.cols))
        self.assertAlmostEqual(costs[0], (self.raster[0, 0] + self.raster[1, 0]) * cost_factor)

        # Exclude one cell and test again
        self.exclude_mask[1, 0] = 0
        from_nodes, to_nodes, costs, valid_count = find_valid_nodes(
            np.int8(dr), np.int8(dc), s_rows_start, s_rows_end, s_cols_start, s_cols_end,
            self.exclude_mask, self.raster, intermediates, self.rows, self.cols, cost_factor, 100
        )

        # Should find 4 valid nodes (one excluded)
        self.assertEqual(valid_count, 4)

    def test_construct_edges(self):
        """Test construct_edges function."""
        # Construct edges
        from_nodes, to_nodes, costs = construct_edges(self.raster, self.steps,
                                                      ignore_max=False)

        # Calculate expected number of edges
        # Each step direction should produce (3-1)*3 = 6 edges
        # Total: 4 directions * 6 = 24 edges
        self.assertEqual(len(from_nodes), 24)
        self.assertEqual(len(to_nodes), 24)
        self.assertEqual(len(costs), 24)

        # Test with a cell set to max value (should be ignored)
        raster_with_max = self.raster.copy()
        raster_with_max[0, 0] = np.iinfo(np.uint16).max  # Max uint16

        from_nodes_max, to_nodes_max, costs_max = construct_edges(raster_with_max, self.steps, ignore_max=True)

        # Should have fewer edges since the max cell is excluded
        self.assertLess(len(from_nodes_max), len(from_nodes) - 1)

        # Test ignoring max flag
        from_nodes_no_ignore, to_nodes_no_ignore, costs_no_ignore = construct_edges(
            raster_with_max, self.steps, ignore_max=False
        )
        self.assertEqual(len(from_nodes_no_ignore), len(from_nodes))  # Should be same as original

    def test_get_outgoing_edges(self):
        """Test get_outgoing_edges function."""
        # Get outgoing edges from center cell (1,1)
        center_idx = ravel_index(1, 1, self.cols)
        to_nodes, costs = get_outgoing_edges(center_idx, self.raster, self.steps,
                                             self.rows, self.cols)

        # Should have 4 outgoing edges (up, down, left, right)
        self.assertEqual(len(to_nodes), 4)
        self.assertEqual(len(costs), 4)

        # Check if target nodes are correct (may be in different order)
        expected_targets = [
            ravel_index(2, 1, self.cols),  # Down
            ravel_index(1, 2, self.cols),  # Right
            ravel_index(0, 1, self.cols),  # Up
            ravel_index(1, 0, self.cols)  # Left
        ]

        # Verify all expected targets are present
        for target in expected_targets:
            self.assertIn(target, to_nodes)

        # Test edge cell (0,0) which should have only 2 outgoing edges
        corner_idx = ravel_index(0, 0, self.cols)
        to_nodes, costs = get_outgoing_edges(corner_idx, self.raster, self.steps,
                                             self.rows, self.cols)

        # Should have 2 outgoing edges (down, right)
        self.assertEqual(len(to_nodes), 2)
        expected_corner_targets = [
            ravel_index(1, 0, self.cols),  # Down
            ravel_index(0, 1, self.cols)  # Right
        ]
        for target in expected_corner_targets:
            self.assertIn(target, to_nodes)

        # Test with excluded cell
        exclude_mask = np.ones((self.rows, self.cols), dtype=np.uint8)
        exclude_mask[2, 1] = 0  # Exclude down direction from center

        to_nodes, costs = get_outgoing_edges(center_idx, self.raster, self.steps,
                                             self.rows, self.cols, exclude_mask)

        # Should have 3 outgoing edges (right, up, left)
        self.assertEqual(len(to_nodes), 3)
        self.assertNotIn(ravel_index(2, 1, self.cols), to_nodes)  # Down direction should be excluded


if __name__ == "__main__":
    unittest.main()
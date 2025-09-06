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
        s_rows_start, s_rows_end, s_cols_start, s_cols_end = bounds

        # Source region should be (0,0) to (4,4)
        self.assertEqual(s_rows_start, 0)
        self.assertEqual(s_rows_end, 4)
        self.assertEqual(s_cols_start, 0)
        self.assertEqual(s_cols_end, 4)


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


class TestNewTraversalFunctions(unittest.TestCase):
    """Test the new find_nearest_valid_positions_numba and check_max_values functions."""

    def setUp(self):
        """Set up test data for the new functions."""
        # Create a test raster with some max values (invalid positions)
        self.max_value = 65535  # max uint16
        self.raster = np.array([
            [1, 2, 3, 4, 5],
            [2, self.max_value, self.max_value, 6, 7],
            [3, self.max_value, 8, 9, 10],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9]
        ], dtype=np.uint16)

    def test_find_nearest_valid_positions_basic(self):
        """Test find_nearest_valid_positions_numba with basic invalid positions."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Test with invalid positions at (1,1) and (1,2)
        invalid_positions = np.array([[1, 1], [1, 2]], dtype=np.int32)

        corrected = find_nearest_valid_positions_numba(
            self.raster, invalid_positions, self.max_value
        )

        # Check shape
        self.assertEqual(corrected.shape, (2, 2))

        # For position (1,1), nearest valid should be one of the adjacent cells
        row1, col1 = corrected[0]
        self.assertNotEqual(self.raster[row1, col1], self.max_value)

        # For position (1,2), nearest valid should also be adjacent
        row2, col2 = corrected[1]
        self.assertNotEqual(self.raster[row2, col2], self.max_value)

        # Check that corrected positions are actually closer than original
        dist1 = np.sqrt((row1 - 1) ** 2 + (col1 - 1) ** 2)
        dist2 = np.sqrt((row2 - 1) ** 2 + (col2 - 2) ** 2)
        self.assertLessEqual(dist1, np.sqrt(2))  # Should be at most sqrt(2) away
        self.assertLessEqual(dist2, np.sqrt(2))

    def test_find_nearest_valid_positions_single(self):
        """Test find_nearest_valid_positions_numba with a single invalid position."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Test with single invalid position at (2,1)
        invalid_positions = np.array([[2, 1]], dtype=np.int32)

        corrected = find_nearest_valid_positions_numba(
            self.raster, invalid_positions, self.max_value
        )

        self.assertEqual(corrected.shape, (1, 2))

        # The nearest valid position should be adjacent
        row, col = corrected[0]
        self.assertNotEqual(self.raster[row, col], self.max_value)

        # Distance should be minimal
        dist = np.sqrt((row - 2) ** 2 + (col - 1) ** 2)
        self.assertLessEqual(dist, 1.0)  # Should be at most 1 cell away

    def test_find_nearest_valid_positions_edge_cases(self):
        """Test find_nearest_valid_positions_numba with edge cases."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Create a raster with max values at corners
        corner_raster = np.ones((5, 5), dtype=np.uint16)
        corner_raster[0, 0] = self.max_value
        corner_raster[0, 4] = self.max_value
        corner_raster[4, 0] = self.max_value
        corner_raster[4, 4] = self.max_value

        # Test corner positions
        invalid_positions = np.array([[0, 0], [0, 4], [4, 0], [4, 4]], dtype=np.int32)

        corrected = find_nearest_valid_positions_numba(
            corner_raster, invalid_positions, self.max_value
        )

        self.assertEqual(corrected.shape, (4, 2))

        # All corrected positions should be valid
        for i in range(4):
            row, col = corrected[i]
            self.assertNotEqual(corner_raster[row, col], self.max_value)
            self.assertGreaterEqual(row, 0)
            self.assertLess(row, 5)
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, 5)

    def test_find_nearest_valid_positions_already_valid_behavior(self):
        """Test that find_nearest_valid_positions_numba finds neighbors even for valid positions."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Create a raster with no max values
        valid_raster = np.arange(25, dtype=np.uint16).reshape(5, 5)

        # The function is designed to find nearby valid positions
        # It starts searching from radius 1, not 0, so won't return same position
        positions = np.array([[0, 0], [2, 2], [4, 4]], dtype=np.int32)

        corrected = find_nearest_valid_positions_numba(
            valid_raster, positions, self.max_value
        )

        # The function will return nearby valid positions (not the same positions)
        for i in range(len(positions)):
            orig_row, orig_col = positions[i]
            new_row, new_col = corrected[i]

            # Should be valid (not max_value)
            self.assertNotEqual(valid_raster[new_row, new_col], self.max_value)

            # Should be within bounds
            self.assertGreaterEqual(new_row, 0)
            self.assertLess(new_row, 5)
            self.assertGreaterEqual(new_col, 0)
            self.assertLess(new_col, 5)

            # Should be close to original (at most sqrt(2) away for adjacent cells)
            dist = np.sqrt((new_row - orig_row) ** 2 + (new_col - orig_col) ** 2)
            self.assertLessEqual(dist, np.sqrt(2))

            # Should NOT be the same position (function starts search at radius 1)
            self.assertFalse(new_row == orig_row and new_col == orig_col)

    def test_find_nearest_valid_positions_large_search(self):
        """Test find_nearest_valid_positions_numba with a large invalid region."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Create a raster with a large invalid region in the center
        large_raster = np.ones((10, 10), dtype=np.uint16) * 5
        large_raster[3:7, 3:7] = self.max_value  # 4x4 invalid region

        # Invalid position in the center of the invalid region
        invalid_positions = np.array([[5, 5]], dtype=np.int32)

        corrected = find_nearest_valid_positions_numba(
            large_raster, invalid_positions, self.max_value
        )

        row, col = corrected[0]

        # Should find a valid position
        self.assertNotEqual(large_raster[row, col], self.max_value)

        # Should be on or near the border of the invalid region
        dist = np.sqrt((row - 5) ** 2 + (col - 5) ** 2)
        self.assertGreaterEqual(dist, 2.0)  # At least 2 cells away
        self.assertLessEqual(dist, 3.0)  # At most 3 cells away

    def test_find_nearest_valid_positions_mostly_invalid(self):
        """Test find_nearest_valid_positions_numba when most of raster is invalid."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Create a raster that's mostly invalid
        mostly_invalid = np.full((5, 5), self.max_value, dtype=np.uint16)
        # Only a few valid positions
        mostly_invalid[0, 0] = 1
        mostly_invalid[4, 4] = 2
        mostly_invalid[0, 4] = 3

        # Test from center position
        invalid_positions = np.array([[2, 2]], dtype=np.int32)

        corrected = find_nearest_valid_positions_numba(
            mostly_invalid, invalid_positions, self.max_value
        )

        row, col = corrected[0]
        # Should find one of the three valid positions
        self.assertIn((row, col), [(0, 0), (4, 4), (0, 4)])

    def test_check_max_values_with_invalid(self):
        """Test check_max_values with indices containing max values."""
        from pyorps.utils.traversal import check_max_values

        # Create indices that include some max value positions
        indices_2d = np.array([
            [0, 0],  # Valid (value=1)
            [1, 1],  # Invalid (max_value)
            [1, 2],  # Invalid (max_value)
            [2, 2],  # Valid (value=8)
            [2, 1],  # Invalid (max_value)
            [3, 3],  # Valid (value=7)
        ], dtype=np.int32)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        # Should detect max values
        self.assertTrue(has_max)

        # Check invalid mask
        expected_mask = np.array([False, True, True, False, True, False],
                                 dtype=np.bool_)
        np.testing.assert_array_equal(invalid_mask, expected_mask)

        # Check invalid indices
        self.assertEqual(invalid_indices.shape[0], 3)  # 3 invalid positions
        expected_invalid = np.array([[1, 1], [1, 2], [2, 1]], dtype=np.int32)
        np.testing.assert_array_equal(invalid_indices, expected_invalid)

    def test_check_max_values_no_invalid(self):
        """Test check_max_values with indices containing no max values."""
        from pyorps.utils.traversal import check_max_values

        # Create indices that don't include max value positions
        indices_2d = np.array([
            [0, 0],  # Valid (value=1)
            [0, 3],  # Valid (value=4)
            [2, 2],  # Valid (value=8)
            [3, 3],  # Valid (value=7)
            [4, 4],  # Valid (value=9)
        ], dtype=np.int32)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        # Should not detect max values
        self.assertFalse(has_max)

        # All should be valid
        expected_mask = np.array([False, False, False, False, False], dtype=np.bool_)
        np.testing.assert_array_equal(invalid_mask, expected_mask)

        # No invalid indices
        self.assertEqual(invalid_indices.shape[0], 0)

    def test_check_max_values_all_invalid(self):
        """Test check_max_values when all indices are invalid."""
        from pyorps.utils.traversal import check_max_values

        # Create indices that are all max value positions
        indices_2d = np.array([
            [1, 1],  # Invalid (max_value)
            [1, 2],  # Invalid (max_value)
            [2, 1],  # Invalid (max_value)
        ], dtype=np.int32)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        # Should detect max values
        self.assertTrue(has_max)

        # All should be invalid
        expected_mask = np.array([True, True, True], dtype=np.bool_)
        np.testing.assert_array_equal(invalid_mask, expected_mask)

        # All indices should be in invalid_indices
        self.assertEqual(invalid_indices.shape[0], 3)
        np.testing.assert_array_equal(invalid_indices, indices_2d)

    def test_check_max_values_empty(self):
        """Test check_max_values with empty indices."""
        from pyorps.utils.traversal import check_max_values

        # Empty indices array
        indices_2d = np.array([], dtype=np.int32).reshape(0, 2)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        # Should not detect max values
        self.assertFalse(has_max)

        # Empty mask
        self.assertEqual(invalid_mask.shape[0], 0)

        # Empty invalid indices
        self.assertEqual(invalid_indices.shape[0], 0)

    def test_check_max_values_single_position(self):
        """Test check_max_values with a single position."""
        from pyorps.utils.traversal import check_max_values

        # Test with single valid position
        indices_2d = np.array([[0, 0]], dtype=np.int32)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        self.assertFalse(has_max)
        np.testing.assert_array_equal(invalid_mask, np.array([False], dtype=np.bool_))
        self.assertEqual(invalid_indices.shape[0], 0)

        # Test with single invalid position
        indices_2d = np.array([[1, 1]], dtype=np.int32)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        self.assertTrue(has_max)
        np.testing.assert_array_equal(invalid_mask, np.array([True], dtype=np.bool_))
        self.assertEqual(invalid_indices.shape[0], 1)
        np.testing.assert_array_equal(invalid_indices, indices_2d)

    def test_check_max_values_mixed_order(self):
        """Test check_max_values preserves order of invalid indices."""
        from pyorps.utils.traversal import check_max_values

        # Mix valid and invalid in various order
        indices_2d = np.array([
            [1, 1],  # Invalid
            [0, 0],  # Valid
            [2, 1],  # Invalid
            [3, 3],  # Valid
            [1, 2],  # Invalid
        ], dtype=np.int32)

        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, indices_2d, self.max_value
        )

        self.assertTrue(has_max)

        # Check mask pattern
        expected_mask = np.array([True, False, True, False, True], dtype=np.bool_)
        np.testing.assert_array_equal(invalid_mask, expected_mask)

        # Check that invalid indices are in order
        expected_invalid = np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)
        np.testing.assert_array_equal(invalid_indices, expected_invalid)

    def test_integration_check_and_correct(self):
        """Test integration of check_max_values and find_nearest_valid_positions_numba."""
        from pyorps.utils.traversal import check_max_values, \
            find_nearest_valid_positions_numba

        # Create a path that goes through some invalid cells
        path_indices_2d = np.array([
            [0, 0],  # Start - valid
            [1, 0],  # Valid
            [1, 1],  # Invalid!
            [2, 1],  # Invalid!
            [2, 2],  # Valid
            [3, 3],  # Valid
            [4, 4],  # End - valid
        ], dtype=np.int32)

        # First check for invalid positions
        has_max, invalid_mask, invalid_indices = check_max_values(
            self.raster, path_indices_2d, self.max_value
        )

        self.assertTrue(has_max)
        self.assertEqual(invalid_indices.shape[0], 2)

        # Now correct the invalid positions
        corrected_positions = find_nearest_valid_positions_numba(
            self.raster, invalid_indices, self.max_value
        )

        # Verify all corrected positions are valid
        for i in range(corrected_positions.shape[0]):
            row, col = corrected_positions[i]
            self.assertNotEqual(self.raster[row, col], self.max_value)

        # Create corrected path
        corrected_path = path_indices_2d.copy()
        invalid_idx = 0
        for i in range(len(path_indices_2d)):
            if invalid_mask[i]:
                corrected_path[i] = corrected_positions[invalid_idx]
                invalid_idx += 1

        # Verify entire corrected path is valid
        for i in range(len(corrected_path)):
            row, col = corrected_path[i]
            self.assertNotEqual(self.raster[row, col], self.max_value)

    def test_find_nearest_valid_positions_performance(self):
        """Test find_nearest_valid_positions_numba handles multiple positions efficiently."""
        from pyorps.utils.traversal import find_nearest_valid_positions_numba

        # Create a larger raster with scattered invalid positions
        large_raster = np.random.randint(1, 100, size=(20, 20), dtype=np.uint16)

        # Add some invalid positions
        invalid_positions = []
        for i in range(5, 15, 3):
            for j in range(5, 15, 3):
                large_raster[i, j] = self.max_value
                invalid_positions.append([i, j])

        invalid_positions = np.array(invalid_positions, dtype=np.int32)

        # Correct all invalid positions at once
        corrected = find_nearest_valid_positions_numba(
            large_raster, invalid_positions, self.max_value
        )

        # Verify all corrected positions
        self.assertEqual(corrected.shape[0], len(invalid_positions))
        for i in range(len(corrected)):
            row, col = corrected[i]
            self.assertNotEqual(large_raster[row, col], self.max_value)

            # Check distance is reasonable
            orig_row, orig_col = invalid_positions[i]
            dist = np.sqrt((row - orig_row) ** 2 + (col - orig_col) ** 2)
            self.assertLessEqual(dist, 3.0)  # Should find something within 3 cells

    def test_different_max_values(self):
        """Test both functions with different max values."""
        from pyorps.utils.traversal import check_max_values, \
            find_nearest_valid_positions_numba

        # Test with a different max value
        custom_max = 1000
        custom_raster = np.array([
            [1, 2, 3],
            [4, custom_max, 6],
            [7, 8, 9]
        ], dtype=np.uint16)

        indices_2d = np.array([[1, 1], [0, 0], [2, 2]], dtype=np.int32)

        # Check with custom max
        has_max, invalid_mask, invalid_indices = check_max_values(
            custom_raster, indices_2d, custom_max
        )

        self.assertTrue(has_max)
        expected_mask = np.array([True, False, False], dtype=np.bool_)
        np.testing.assert_array_equal(invalid_mask, expected_mask)

        # Find nearest valid for custom max
        corrected = find_nearest_valid_positions_numba(
            custom_raster, invalid_indices, custom_max
        )

        row, col = corrected[0]
        self.assertNotEqual(custom_raster[row, col], custom_max)
        self.assertLess(custom_raster[row, col], custom_max)


if __name__ == "__main__":
    unittest.main()
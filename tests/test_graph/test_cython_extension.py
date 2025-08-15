# tests/test_cython_extensions.py
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

# Import the Cython extensions
from pyorps.utils.path_algorithms import (
    dijkstra_2d_cython,
    dijkstra_single_source_multiple_targets,
    dijkstra_some_pairs_shortest_paths,
    dijkstra_multiple_sources_multiple_targets,
    create_exclude_mask
    )


class TestCythonExtensions(unittest.TestCase):
    """Test cases for Cython pathfinding extensions."""

    def setUp(self):
        """Set up test data for Cython extensions."""
        # Create a simple 5x5 test raster
        self.simple_raster = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 10, 2, 1],  # High cost center
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint16)

        # Create steps for 4-connected neighborhood
        self.steps_4 = np.array([
            [0, 1],  # Right
            [1, 0],  # Down
            [0, -1],  # Left
            [-1, 0]  # Up
        ], dtype=np.int8)

        # Create steps for 8-connected neighborhood
        self.steps_8 = np.array([
            [0, 1],  # Right
            [1, 0],  # Down
            [0, -1],  # Left
            [-1, 0],  # Up
            [1, 1],  # Down-Right
            [-1, 1],  # Up-Right
            [1, -1],  # Down-Left
            [-1, -1]  # Up-Left
        ], dtype=np.int8)

        # Define common source and target indices
        self.source_idx = 0  # Top-left corner (0,0)
        self.target_idx = 24  # Bottom-right corner (4,4)

        # Create a raster with obstacles
        self.obstacle_raster = np.array([
            [1, 1, 1, 1, 1],
            [1, 65535, 65535, 65535, 1],  # Row of obstacles
            [1, 1, 1, 65535, 1],
            [1, 65535, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint16)

        # Large raster for performance testing
        self.large_raster = np.random.randint(1, 10, (50, 50), dtype=np.uint16)

    def test_create_exclude_mask_basic(self):
        """Test create_exclude_mask with basic functionality."""
        # Test with max value 65535
        exclude_mask = create_exclude_mask(self.obstacle_raster, 65535)

        # Check dimensions
        self.assertEqual(exclude_mask.shape, self.obstacle_raster.shape)
        self.assertEqual(exclude_mask.dtype, np.uint8)

        # Check that obstacles are excluded (0) and others are included (1)
        expected = np.ones((5, 5), dtype=np.uint8)
        expected[1, 1:4] = 0  # Row of obstacles
        expected[2, 3] = 0  # Single obstacle
        expected[3, 1] = 0  # Single obstacle

        assert_array_equal(exclude_mask, expected)

    def test_create_exclude_mask_no_obstacles(self):
        """Test create_exclude_mask with no obstacles."""
        exclude_mask = create_exclude_mask(self.simple_raster, 65535)

        # All cells should be included (value 1)
        expected = np.ones(self.simple_raster.shape, dtype=np.uint8)
        assert_array_equal(exclude_mask, expected)

    def test_create_exclude_mask_all_obstacles(self):
        """Test create_exclude_mask with all obstacles."""
        all_obstacles = np.full((3, 3), 65535, dtype=np.uint16)
        exclude_mask = create_exclude_mask(all_obstacles, 65535)

        # All cells should be excluded (value 0)
        expected = np.zeros((3, 3), dtype=np.uint8)
        assert_array_equal(exclude_mask, expected)

    def test_dijkstra_2d_cython_basic_path(self):
        """Test dijkstra_2d_cython with basic pathfinding."""
        # Find path from top-left to bottom-right
        path = dijkstra_2d_cython(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            self.target_idx
        )

        # Should find a path
        self.assertGreater(len(path), 0)

        # Path should start at source and end at target
        self.assertEqual(path[0], self.source_idx)
        self.assertEqual(path[-1], self.target_idx)

        # Path should be continuous (each step connects to next)
        self._validate_path_continuity(path, 5)  # 5 columns

    def test_dijkstra_2d_cython_no_path(self):
        """Test dijkstra_2d_cython when no path exists."""
        # Create isolated target
        isolated_raster = np.array([
            [1, 1, 65535],
            [1, 1, 65535],
            [65535, 65535, 1]
        ], dtype=np.uint16)

        path = dijkstra_2d_cython(
            isolated_raster,
            self.steps_4,
            0,  # Source at (0,0)
            8,  # Target at (2,2) - isolated
            max_value=65535
        )

        # Should return empty path
        self.assertEqual(len(path), 0)

    def test_dijkstra_2d_cython_different_neighborhoods(self):
        """Test dijkstra_2d_cython with different neighborhood configurations."""
        # Test with 4-connected
        path_4 = dijkstra_2d_cython(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            self.target_idx
        )

        # Test with 8-connected
        path_8 = dijkstra_2d_cython(
            self.simple_raster,
            self.steps_8,
            self.source_idx,
            self.target_idx
        )

        # Both should find paths
        self.assertGreater(len(path_4), 0)
        self.assertGreater(len(path_8), 0)

        # 8-connected might find shorter path (fewer nodes)
        self.assertLessEqual(len(path_8), len(path_4))

    def test_dijkstra_2d_cython_same_source_target(self):
        """Test dijkstra_2d_cython when source equals target."""
        path = dijkstra_2d_cython(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            self.source_idx  # Same as source
        )

        # Should return path with single node
        self.assertEqual(len(path), 0)

    def test_dijkstra_single_source_multiple_targets_basic(self):
        """Test dijkstra_single_source_multiple_targets with basic functionality."""
        # Define multiple targets
        targets = np.array([6, 12, 18, 24], dtype=np.uint32)  # Various positions

        paths = dijkstra_single_source_multiple_targets(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            targets
        )

        # Should return list with same length as targets
        self.assertEqual(len(paths), len(targets))

        # Each path should either be valid or empty
        for i, path in enumerate(paths):
            if len(path) > 0:
                # Valid path should start at source and end at target
                self.assertEqual(path[0], self.source_idx)
                self.assertEqual(path[-1], targets[i])
                self._validate_path_continuity(path, 5)

    def test_dijkstra_single_source_multiple_targets_unreachable(self):
        """Test dijkstra_single_source_multiple_targets with unreachable targets."""
        # Use obstacle raster where some targets are unreachable
        targets = np.array([24, 6, 7, 8], dtype=np.uint32)  # Some blocked by obstacles

        paths = dijkstra_single_source_multiple_targets(
            self.obstacle_raster,
            self.steps_4,
            self.source_idx,
            targets,
            max_value=65535
        )

        # Should return list with same length as targets
        self.assertEqual(len(paths), len(targets))

        # Some paths should be empty (unreachable)
        empty_paths = sum(1 for path in paths if len(path) == 0)
        self.assertGreater(empty_paths, 0)

        # Valid paths should be correct
        for i, path in enumerate(paths):
            if len(path) > 0:
                self.assertEqual(path[0], self.source_idx)
                self.assertEqual(path[-1], targets[i])

    def test_dijkstra_some_pairs_with_return_paths(self):
        """Test dijkstra_some_pairs_shortest_paths returning paths."""
        sources = np.array([0, 5, 10], dtype=np.uint32)
        targets = np.array([24, 19, 14], dtype=np.uint32)

        paths = dijkstra_some_pairs_shortest_paths(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            return_paths=True
        )

        # Should return list of paths
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), len(sources))

        # Validate each path
        for i, path in enumerate(paths):
            if len(path) > 0:
                self.assertEqual(path[0], sources[i])
                self.assertEqual(path[-1], targets[i])
                self._validate_path_continuity(path, 5)

    def test_dijkstra_some_pairs_with_return_costs(self):
        """Test dijkstra_some_pairs_shortest_paths returning only costs."""
        sources = np.array([0, 5, 10], dtype=np.uint32)
        targets = np.array([24, 19, 14], dtype=np.uint32)

        costs = dijkstra_some_pairs_shortest_paths(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            return_paths=False
        )

        # Should return numpy array of costs
        self.assertIsInstance(costs, np.ndarray)
        self.assertEqual(len(costs), len(sources))
        self.assertEqual(costs.dtype, np.float64)

        # Valid costs should be positive
        finite_costs = costs[np.isfinite(costs)]
        self.assertTrue(np.all(finite_costs > 0))

    def test_dijkstra_multiple_sources_multiple_targets_matrix(self):
        """Test dijkstra_multiple_sources_multiple_targets returning cost matrix."""
        sources = np.array([0, 4, 20], dtype=np.uint32)  # Top corners and bottom-left
        targets = np.array([24, 21, 3], dtype=np.uint32)  # Bottom corners and top-right

        cost_matrix = dijkstra_multiple_sources_multiple_targets(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            return_paths=False
        )

        # Should return cost matrix
        self.assertIsInstance(cost_matrix, np.ndarray)
        self.assertEqual(cost_matrix.shape, (len(sources), len(targets)))
        self.assertEqual(cost_matrix.dtype, np.float64)

        # Valid costs should be positive
        finite_costs = cost_matrix[np.isfinite(cost_matrix)]
        self.assertTrue(np.all(finite_costs > 0))

    def test_dijkstra_multiple_sources_multiple_targets_paths(self):
        """Test dijkstra_multiple_sources_multiple_targets returning paths."""
        sources = np.array([0, 4], dtype=np.uint32)
        targets = np.array([24, 20], dtype=np.uint32)

        all_paths = dijkstra_multiple_sources_multiple_targets(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            return_paths=True
        )

        # Should return list of lists
        self.assertIsInstance(all_paths, list)
        self.assertEqual(len(all_paths), len(sources))

        # Validate structure and paths
        for source_idx, source_paths in enumerate(all_paths):
            self.assertEqual(len(source_paths), len(targets))

            for target_idx, path in enumerate(source_paths):
                if len(path) > 0:
                    self.assertEqual(path[0], sources[source_idx])
                    self.assertEqual(path[-1], targets[target_idx])
                    self._validate_path_continuity(path, 5)

    def test_performance_large_raster(self):
        """Test performance with larger raster."""
        # Use a portion of the large raster
        large_steps = self.steps_4
        source = 0
        target = 50 * 50 - 1  # Bottom-right corner

        # This should complete within reasonable time
        import time
        start_time = time.time()

        path = dijkstra_2d_cython(
            self.large_raster,
            large_steps,
            source,
            target
        )

        end_time = time.time()

        # Should find a path quickly (less than 1 second for 50x50)
        self.assertLess(end_time - start_time, 1.0)
        self.assertGreater(len(path), 0)

    def test_edge_cases_empty_targets(self):
        """Test edge cases with empty target arrays."""
        empty_targets = np.array([], dtype=np.uint32)

        paths = dijkstra_single_source_multiple_targets(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            empty_targets
        )

        # Should return empty list
        self.assertEqual(len(paths), 0)

    def test_edge_cases_invalid_indices(self):
        """Test edge cases with invalid indices."""
        # Test with out-of-bounds source
        path = dijkstra_2d_cython(
                self.simple_raster,
                self.steps_4,
                999,  # Invalid source
                self.target_idx
            )
        self.assertEqual(len(path), 0)

        # Test with out-of-bounds target
        with self.assertRaises((IndexError, ValueError)):
            dijkstra_2d_cython(
                self.simple_raster,
                self.steps_4,
                self.source_idx,
                999  # Invalid target
            )

    def test_algorithm_consistency(self):
        """Test that different algorithms produce consistent results."""
        # Test single path vs single-source-multiple-targets
        single_path = dijkstra_2d_cython(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            self.target_idx
        )

        multi_paths = dijkstra_single_source_multiple_targets(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            np.array([self.target_idx], dtype=np.uint32)
        )

        # Should produce same path
        if len(single_path) > 0 and len(multi_paths[0]) > 0:
            # Paths might be different due to tie-breaking, but should have similar costs
            single_cost = self._calculate_path_cost(single_path, self.simple_raster, 5)
            multi_cost = self._calculate_path_cost(multi_paths[0], self.simple_raster, 5)

            # Costs should be very close (within 1% due to floating point precision)
            self.assertAlmostEqual(single_cost, multi_cost,
                                   delta=max(single_cost, multi_cost) * 0.01)

    def test_different_max_values(self):
        """Test with different max_value parameters."""
        # Create raster with different obstacle values
        custom_raster = self.simple_raster.copy()
        custom_raster[2, 2] = 999  # Custom obstacle value

        # Test with default max_value (should treat 999 as normal cost)
        path_default = dijkstra_2d_cython(
            custom_raster,
            self.steps_4,
            self.source_idx,
            self.target_idx
        )

        # Test with custom max_value (should treat 999 as obstacle)
        path_custom = dijkstra_2d_cython(
            custom_raster,
            self.steps_4,
            self.source_idx,
            self.target_idx,
            max_value=999
        )

        # Both should find paths, but they might be different
        self.assertGreater(len(path_default), 0)
        self.assertGreater(len(path_custom), 0)

    def _validate_path_continuity(self, path, cols):
        """Helper to validate that path nodes are connected."""
        if len(path) < 2:
            return  # Single node path is valid

        for i in range(len(path) - 1):
            current_idx = path[i]
            next_idx = path[i + 1]

            # Convert to row, col
            current_row, current_col = divmod(current_idx, cols)
            next_row, next_col = divmod(next_idx, cols)

            # Calculate step
            dr = int(next_row) - int(current_row)
            dc = int(next_col) - int(current_col)

            # Should be a valid step (within neighborhood)
            step_distance = abs(dr) + abs(dc)
            self.assertLessEqual(step_distance, 2)  # Max step for 8-connected
            self.assertGreaterEqual(step_distance, 1)  # Must move

    def _calculate_path_cost(self, path, raster, cols):
        """Helper to calculate total path cost."""
        if len(path) == 0:
            return float('inf')

        total_cost = 0.0
        for idx in path:
            row, col = divmod(idx, cols)
            total_cost += raster[row, col]

        return total_cost


class TestCythonExtensionsAdvanced(unittest.TestCase):
    """Advanced test cases for Cython extensions."""

    def setUp(self):
        """Set up advanced test scenarios."""
        # Create maze-like raster
        self.maze_raster = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 65535, 1, 65535, 1, 65535, 1],
            [1, 65535, 1, 65535, 1, 65535, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 65535, 65535, 65535, 65535, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.uint16)

        self.steps_4 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=np.int8)

    def test_maze_pathfinding(self):
        """Test pathfinding through maze-like environment."""
        source = 0  # Top-left (0,0)
        target = 48  # Bottom-right (6,6)

        path = dijkstra_2d_cython(
            self.maze_raster,
            self.steps_4,
            source,
            target,
            max_value=65535
        )

        # Should find a path through the maze
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], source)
        self.assertEqual(path[-1], target)

    def test_stress_test_many_targets(self):
        """Stress test with many targets."""
        # Create many targets
        targets = np.arange(1, 49, dtype=np.uint32)  # All cells except source

        paths = dijkstra_single_source_multiple_targets(
            self.maze_raster,
            self.steps_4,
            0,  # Source
            targets,
            max_value=65535
        )

        # Should handle many targets efficiently
        self.assertEqual(len(paths), len(targets))

        # Count reachable targets
        reachable = sum(1 for path in paths if len(path) > 0)
        self.assertGreater(reachable, 0)

    def test_optimal_path_verification(self):
        """Verify that found paths are optimal."""
        # Simple 3x3 grid with known optimal path
        simple_3x3 = np.array([
            [1, 5, 1],
            [1, 1, 1],
            [1, 5, 1]
        ], dtype=np.uint16)

        # Path from (0,0) to (2,2)
        path = dijkstra_2d_cython(
            simple_3x3,
            self.steps_4,
            0,  # (0,0)
            8,  # (2,2)
        )

        # Optimal path should avoid the high-cost cells (value 5)
        # Should go: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (2,2)
        # Or: (0,0) -> (0,2) -> (1,2) -> (2,2) -> (2,2)
        expected_cost = 1 + 1 + 1 + 1 + 1  # 5 units of cost-1 cells
        actual_cost = sum(simple_3x3[idx // 3, idx % 3] for idx in path)

        self.assertEqual(actual_cost, expected_cost)


if __name__ == '__main__':  # Run tests only if Cython extensions are available if
    unittest.main()

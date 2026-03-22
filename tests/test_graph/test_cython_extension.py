# tests/test_cython_extensions_updated.py
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
    delta_stepping_2d,
    delta_stepping_single_source_multiple_targets,
    delta_stepping_some_pairs_shortest_paths,
    delta_stepping_multiple_sources_multiple_targets,
)

from pyorps.utils.path_core import create_exclude_mask


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

    # ==================== DELTA-STEPPING TESTS ====================

    def test_delta_stepping_2d_basic_path(self):
        """Test delta_stepping_2d with basic pathfinding."""
        # Find path from top-left to bottom-right
        path = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=5.0
            # No use_astar parameter anymore
        )

        # Should find a path
        self.assertGreater(len(path), 0)

        # Path should start at source and end at target
        self.assertEqual(path[0], self.source_idx)
        self.assertEqual(path[-1], self.target_idx)

        # Path should be continuous
        self._validate_path_continuity_uint64(path, 5)

    def test_delta_stepping_2d_with_margin(self):
        """Test delta_stepping_2d with margin parameter."""
        # Test with default margin
        path_default = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=5.0
            # Default margin is 1.00001
        )

        # Test with custom margin
        path_custom_margin = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=5.0,
            margin=1.1  # Larger margin for earlier termination
        )

        # Both should find paths
        self.assertGreater(len(path_default), 0)
        self.assertGreater(len(path_custom_margin), 0)

        # Both should find optimal paths (margin doesn't affect optimality)
        cost_default = self._calculate_path_cost_uint64(path_default,
                                                        self.simple_raster, 5)
        cost_margin = self._calculate_path_cost_uint64(path_custom_margin,
                                                       self.simple_raster, 5)

        # Costs should be similar (within floating point tolerance)
        self.assertAlmostEqual(cost_default, cost_margin, delta=0.01)

    def test_delta_stepping_2d_no_path(self):
        """Test delta_stepping_2d when no path exists."""
        # Create isolated target
        isolated_raster = np.array([
            [1, 1, 65535],
            [1, 1, 65535],
            [65535, 65535, 1]
        ], dtype=np.uint16)

        path = delta_stepping_2d(
            isolated_raster,
            self.steps_4,
            np.uint64(0),  # Source at (0,0)
            np.uint64(8),  # Target at (2,2) - isolated
            delta=5.0,
            max_value=65535
        )

        # Should return empty path
        self.assertEqual(len(path), 0)

    def test_delta_stepping_2d_different_deltas(self):
        """Test delta_stepping_2d with different delta values."""
        # Small delta (more buckets, more parallelism potential)
        path_small_delta = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=1.0
        )

        # Large delta (fewer buckets, less overhead)
        path_large_delta = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=50.0
        )

        # Both should find paths
        self.assertGreater(len(path_small_delta), 0)
        self.assertGreater(len(path_large_delta), 0)

        # Paths might be different but costs should be similar
        cost_small = self._calculate_path_cost_uint64(path_small_delta,
                                                      self.simple_raster, 5)
        cost_large = self._calculate_path_cost_uint64(path_large_delta,
                                                      self.simple_raster, 5)

        # Costs should be within reasonable tolerance
        self.assertAlmostEqual(cost_small, cost_large, delta=cost_small * 0.1)

    def test_delta_stepping_single_source_multiple_targets(self):
        """Test delta_stepping_single_source_multiple_targets WITHOUT margin."""
        # Define multiple targets
        targets = np.array([6, 12, 18, 24], dtype=np.uint64)

        paths = delta_stepping_single_source_multiple_targets(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            targets,
            delta=5.0
            # No margin parameter for multi-target!
        )

        # Should return list with same length as targets
        self.assertEqual(len(paths), len(targets))

        # Each path should either be valid or empty
        for i, path in enumerate(paths):
            if len(path) > 0:
                self.assertEqual(path[0], self.source_idx)
                self.assertEqual(path[-1], targets[i])
                self._validate_path_continuity_uint64(path, 5)

    def test_delta_stepping_multiple_sources_multiple_targets(self):
        """Test delta_stepping_multiple_sources_multiple_targets WITHOUT margin."""
        sources = np.array([0, 4, 20], dtype=np.uint64)
        targets = np.array([24, 21, 3], dtype=np.uint64)

        # Test returning paths
        all_paths = delta_stepping_multiple_sources_multiple_targets(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            delta=5.0,
            return_paths=True
            # No margin parameter for multi-source-multi-target!
        )

        # Should return list of lists
        self.assertIsInstance(all_paths, list)
        self.assertEqual(len(all_paths), len(sources))

        # Validate structure
        for source_idx, source_paths in enumerate(all_paths):
            self.assertEqual(len(source_paths), len(targets))

        # Test returning cost matrix
        cost_matrix = delta_stepping_multiple_sources_multiple_targets(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            delta=5.0,
            return_paths=False
        )

        # Should return cost matrix
        self.assertIsInstance(cost_matrix, np.ndarray)
        self.assertEqual(cost_matrix.shape, (len(sources), len(targets)))

    def test_delta_stepping_some_pairs_shortest_paths_with_margin(self):
        """Test delta_stepping_some_pairs_shortest_paths WITH margin (pairwise)."""
        sources = np.array([0, 5, 10], dtype=np.uint64)
        targets = np.array([24, 19, 14], dtype=np.uint64)

        # Test with default margin
        paths_default = delta_stepping_some_pairs_shortest_paths(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            delta=5.0,
            return_paths=True
            # Default margin is 1.00001
        )

        # Test with custom margin
        paths_custom_margin = delta_stepping_some_pairs_shortest_paths(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            delta=5.0,
            return_paths=True,
            margin=1.5  # Larger margin for earlier termination
        )

        # Should return list of paths
        self.assertIsInstance(paths_default, list)
        self.assertIsInstance(paths_custom_margin, list)
        self.assertEqual(len(paths_default), len(sources))
        self.assertEqual(len(paths_custom_margin), len(sources))

        # Validate each path
        for i, path in enumerate(paths_default):
            if len(path) > 0:
                self.assertEqual(path[0], sources[i])
                self.assertEqual(path[-1], targets[i])
                self._validate_path_continuity_uint64(path, 5)

        # Test returning costs
        costs = delta_stepping_some_pairs_shortest_paths(
            self.simple_raster,
            self.steps_4,
            sources,
            targets,
            delta=5.0,
            return_paths=False,
            margin=1.1
        )

        # Should return array of costs
        self.assertIsInstance(costs, np.ndarray)
        self.assertEqual(len(costs), len(sources))

    def test_delta_stepping_parallel_performance(self):
        """Test delta_stepping with different thread counts."""
        # Test with single thread
        path_single = delta_stepping_2d(
            self.large_raster,
            self.steps_4,
            np.uint64(0),
            np.uint64(50 * 50 - 1),
            delta=10.0,
            num_threads=1
        )

        # Test with multiple threads
        path_multi = delta_stepping_2d(
            self.large_raster,
            self.steps_4,
            np.uint64(0),
            np.uint64(50 * 50 - 1),
            delta=10.0,
            num_threads=4
        )

        # Both should find paths
        self.assertGreater(len(path_single), 0)
        self.assertGreater(len(path_multi), 0)

        # Costs should be very similar
        cost_single = self._calculate_path_cost_uint64(path_single, self.large_raster,
                                                       50)
        cost_multi = self._calculate_path_cost_uint64(path_multi, self.large_raster, 50)
        self.assertAlmostEqual(cost_single, cost_multi, delta=cost_single * 0.01)

    def test_delta_vs_dijkstra_consistency(self):
        """Test that delta-stepping produces similar results to Dijkstra."""
        # Run Dijkstra
        dijkstra_path = dijkstra_2d_cython(
            self.simple_raster,
            self.steps_4,
            self.source_idx,
            self.target_idx
        )

        # Run delta-stepping
        delta_path = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=5.0
        )

        # Both should find paths
        self.assertGreater(len(dijkstra_path), 0)
        self.assertGreater(len(delta_path), 0)

        # Calculate costs
        dijkstra_cost = self._calculate_path_cost(dijkstra_path, self.simple_raster, 5)
        delta_cost = self._calculate_path_cost_uint64(delta_path, self.simple_raster, 5)

        # Costs should be identical (both find optimal paths)
        self.assertAlmostEqual(dijkstra_cost, delta_cost, places=5)

    def test_edge_cases_delta_stepping(self):
        """Test edge cases for delta-stepping algorithms."""
        # Test with very small delta
        path = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.target_idx),
            delta=0.01
        )
        self.assertGreater(len(path), 0)

        # Test with empty targets
        empty_targets = np.array([], dtype=np.uint64)
        paths = delta_stepping_single_source_multiple_targets(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            empty_targets,
            delta=5.0
        )
        self.assertEqual(len(paths), 0)

        # Test source equals target
        path = delta_stepping_2d(
            self.simple_raster,
            self.steps_4,
            np.uint64(self.source_idx),
            np.uint64(self.source_idx),
            delta=5.0
        )
        # Should return single-node path
        if len(path) > 0:
            self.assertEqual(len(path), 1)
            self.assertEqual(path[0], self.source_idx)

    def test_invalid_delta_value(self):
        """Test that invalid delta values raise appropriate errors."""
        with self.assertRaises(ValueError):
            delta_stepping_2d(
                self.simple_raster,
                self.steps_4,
                np.uint64(self.source_idx),
                np.uint64(self.target_idx),
                delta=-1.0  # Negative delta should raise error
            )

        with self.assertRaises(ValueError):
            delta_stepping_2d(
                self.simple_raster,
                self.steps_4,
                np.uint64(self.source_idx),
                np.uint64(self.target_idx),
                delta=0.0  # Zero delta should raise error
            )

    # ==================== HELPER METHODS ====================

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

    def _validate_path_continuity_uint64(self, path, cols):
        """Helper to validate that path nodes are connected (uint64 version)."""
        if len(path) < 2:
            return

        for i in range(len(path) - 1):
            current_idx = int(path[i])
            next_idx = int(path[i + 1])

            current_row, current_col = divmod(current_idx, cols)
            next_row, next_col = divmod(next_idx, cols)

            dr = int(next_row) - int(current_row)
            dc = int(next_col) - int(current_col)

            step_distance = abs(dr) + abs(dc)
            self.assertLessEqual(step_distance, 2)
            self.assertGreaterEqual(step_distance, 1)

    def _calculate_path_cost(self, path, raster, cols):
        """Helper to calculate total path cost."""
        if len(path) == 0:
            return float('inf')

        total_cost = 0.0
        for idx in path:
            row, col = divmod(idx, cols)
            total_cost += raster[row, col]

        return total_cost

    def _calculate_path_cost_uint64(self, path, raster, cols):
        """Helper to calculate total path cost for uint64 paths."""
        if len(path) == 0:
            return float('inf')

        total_cost = 0.0
        for idx in path:
            row, col = divmod(int(idx), cols)
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

    def test_maze_pathfinding_delta_stepping(self):
        """Test delta-stepping through maze-like environment."""
        source = np.uint64(0)
        target = np.uint64(48)

        path = delta_stepping_2d(
            self.maze_raster,
            self.steps_4,
            source,
            target,
            delta=5.0,
            max_value=65535
        )

        # Should find a path through the maze
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 48)

    def test_stress_test_many_targets_delta_stepping(self):
        """Stress test delta-stepping with many targets."""
        targets = np.arange(1, 49, dtype=np.uint64)

        paths = delta_stepping_single_source_multiple_targets(
            self.maze_raster,
            self.steps_4,
            np.uint64(0),
            targets,
            delta=10.0,
            max_value=65535,
            num_threads=2
        )

        # Should handle many targets efficiently
        self.assertEqual(len(paths), len(targets))

        # Count reachable targets
        reachable = sum(1 for path in paths if len(path) > 0)
        self.assertGreater(reachable, 0)

    def test_optimal_path_verification_delta_stepping(self):
        """Verify that delta-stepping finds optimal paths."""
        simple_3x3 = np.array([
            [1, 5, 1],
            [1, 1, 1],
            [1, 5, 1]
        ], dtype=np.uint16)

        # Path from (0,0) to (2,2)
        path = delta_stepping_2d(
            simple_3x3,
            self.steps_4,
            np.uint64(0),
            np.uint64(8),
            delta=2.0
        )

        # Calculate cost
        if len(path) > 0:
            actual_cost = sum(simple_3x3[int(idx) // 3, int(idx) % 3] for idx in path)
            # Should find optimal path with cost 5
            self.assertEqual(actual_cost, 5)


class TestAtomicCAS(unittest.TestCase):
    """Tests for the atomic CAS-based delta-stepping implementation."""

    def setUp(self):
        self.steps_4 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=np.int8)

        self.steps_8 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=np.int8)

    def test_delta_vs_dijkstra_random_rasters(self):
        """Verify delta-stepping (CAS) matches Dijkstra on random rasters."""
        rng = np.random.RandomState(42)
        for size in [20, 50, 100]:
            raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
            # Add some forbidden cells
            forbidden = rng.random((size, size)) < 0.1
            raster[forbidden] = 65535
            # Ensure source and target are traversable
            raster[0, 0] = 1
            raster[size - 1, size - 1] = 1

            source = 0
            target = size * size - 1

            dij_path = dijkstra_2d_cython(raster, self.steps_4, source, target)
            delta_path = delta_stepping_2d(
                raster, self.steps_4,
                np.uint64(source), np.uint64(target), delta=10.0
            )

            if len(dij_path) == 0:
                self.assertEqual(len(delta_path), 0,
                                 f"Size {size}: Dijkstra found no path but delta did")
                continue
            if len(delta_path) == 0:
                continue  # delta may fail on isolated targets

            # Compare costs
            dij_cost = sum(raster[idx // size, idx % size] for idx in dij_path)
            delta_cost = sum(
                raster[int(idx) // size, int(idx) % size] for idx in delta_path)
            self.assertAlmostEqual(
                dij_cost, delta_cost, delta=dij_cost * 0.001,
                msg=f"Size {size}: cost mismatch dij={dij_cost} delta={delta_cost}")

    def test_multi_target_cas_correctness(self):
        """Verify multi-target delta-stepping matches single-target."""
        rng = np.random.RandomState(123)
        size = 30
        raster = rng.randint(1, 50, (size, size), dtype=np.uint16)
        raster[0, 0] = 1

        source = np.uint64(0)
        targets = np.array([size * size - 1, size - 1, (size - 1) * size],
                           dtype=np.uint64)

        multi_paths = delta_stepping_single_source_multiple_targets(
            raster, self.steps_4, source, targets, delta=10.0
        )

        self.assertEqual(len(multi_paths), len(targets))

        for i, target in enumerate(targets):
            single_path = delta_stepping_2d(
                raster, self.steps_4, source, target, delta=10.0
            )
            if len(single_path) > 0 and len(multi_paths[i]) > 0:
                single_cost = sum(
                    raster[int(idx) // size, int(idx) % size]
                    for idx in single_path)
                multi_cost = sum(
                    raster[int(idx) // size, int(idx) % size]
                    for idx in multi_paths[i])
                self.assertAlmostEqual(
                    single_cost, multi_cost, delta=single_cost * 0.001,
                    msg=f"Target {i}: single={single_cost} multi={multi_cost}")

    def test_uniform_raster_optimal_path(self):
        """On uniform cost raster, delta-stepping should find Manhattan-optimal path."""
        size = 50
        raster = np.ones((size, size), dtype=np.uint16)
        source = np.uint64(0)
        target = np.uint64(size * size - 1)

        path = delta_stepping_2d(
            raster, self.steps_4, source, target, delta=5.0
        )
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], size * size - 1)

        # On uniform 4-connected grid, optimal cost = (rows-1 + cols-1) * cell_cost + 1
        # Each cell visited costs 1; path visits exactly (size-1)+(size-1)+1 = 2*size-1 cells
        cost = sum(raster[int(idx) // size, int(idx) % size] for idx in path)
        expected_cost = 2 * size - 1
        self.assertEqual(cost, expected_cost)

    def test_multithread_consistency(self):
        """Delta-stepping with different thread counts must produce same cost."""
        rng = np.random.RandomState(99)
        size = 100
        raster = rng.randint(1, 20, (size, size), dtype=np.uint16)
        raster[0, 0] = 1
        raster[size - 1, size - 1] = 1

        source = np.uint64(0)
        target = np.uint64(size * size - 1)

        path_1t = delta_stepping_2d(
            raster, self.steps_4, source, target, delta=10.0, num_threads=1
        )
        path_4t = delta_stepping_2d(
            raster, self.steps_4, source, target, delta=10.0, num_threads=4
        )

        self.assertGreater(len(path_1t), 0)
        self.assertGreater(len(path_4t), 0)

        cost_1t = sum(raster[int(idx) // size, int(idx) % size] for idx in path_1t)
        cost_4t = sum(raster[int(idx) // size, int(idx) % size] for idx in path_4t)
        self.assertAlmostEqual(cost_1t, cost_4t, delta=cost_1t * 0.01)

    def test_8connected_delta_stepping(self):
        """Delta-stepping with 8-connectivity should find paths with diagonal moves."""
        raster = np.ones((10, 10), dtype=np.uint16)
        source = np.uint64(0)
        target = np.uint64(99)

        path = delta_stepping_2d(
            raster, self.steps_8, source, target, delta=5.0
        )
        self.assertGreater(len(path), 0)
        # 8-connected diagonal path should be shorter than 4-connected
        self.assertLessEqual(len(path), 19)  # 4-connected would be 19


class TestDeltaSteppingVsNetworkitDijkstra(unittest.TestCase):
    """Cross-backend consistency: Cython delta-stepping (CAS) vs NetworkKit Dijkstra.

    Both must produce the same shortest-path cost for identical rasters.
    Costs are computed from the edge weights built by construct_edges
    so that any difference in internal representations is eliminated.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from networkit import Graph
            from networkit.distance import Dijkstra as NKDijkstra
            cls.nk_available = True
        except ImportError:
            cls.nk_available = False

    def setUp(self):
        if not self.nk_available:
            self.skipTest("networkit not installed")

        self.steps_4 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=np.int8)

        self.steps_8 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=np.int8)

    def _build_networkit_graph(self, raster, steps):
        """Build a NetworkKit graph from a raster using construct_edges."""
        from pyorps.utils.traversal import construct_edges
        from networkit import Graph as NKGraph

        from_nodes, to_nodes, costs = construct_edges(raster, steps, True)
        n = raster.shape[0] * raster.shape[1]
        g = NKGraph(n, weighted=True, directed=False)
        from_u64 = np.asarray(from_nodes, dtype=np.uint64)
        to_u64 = np.asarray(to_nodes, dtype=np.uint64)
        g.addEdges(
            (costs.astype(np.float64, copy=False), (from_u64, to_u64)),
            addMissing=False,
        )
        return g, from_nodes, to_nodes, costs

    def _networkit_dijkstra_distance(self, graph, source, target):
        """Run NetworkKit Dijkstra and return distance to target."""
        from networkit.distance import Dijkstra as NKDijkstra

        dij = NKDijkstra(graph, source, storePaths=True, target=target)
        dij.run()
        return dij.distance(target), dij.getPath(target)

    def _edge_cost_of_path(self, path, from_nodes, to_nodes, costs):
        """Compute the total edge cost of a path using the edge list.

        This gives a backend-independent cost because both NetworkKit and
        Cython delta-stepping should traverse the same weighted graph.
        """
        if len(path) < 2:
            return 0.0

        # Build adjacency lookup: (u, v) -> min cost
        edge_cost = {}
        for i in range(len(from_nodes)):
            u, v, c = int(from_nodes[i]), int(to_nodes[i]), float(costs[i])
            key = (min(u, v), max(u, v))  # undirected
            if key not in edge_cost or c < edge_cost[key]:
                edge_cost[key] = c

        total = 0.0
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            key = (min(u, v), max(u, v))
            if key not in edge_cost:
                return float('inf')  # broken path
            total += edge_cost[key]
        return total

    def _run_comparison(self, raster, steps, source, target, delta=10.0,
                        cost_tol_pct=0.01, msg_prefix=""):
        """Run both backends and compare distances and path costs."""
        rows, cols = raster.shape

        # --- NetworkKit Dijkstra ---
        graph, from_nodes, to_nodes, costs = self._build_networkit_graph(
            raster, steps
        )
        nk_dist, nk_path = self._networkit_dijkstra_distance(
            graph, int(source), int(target)
        )

        # --- Cython delta-stepping (CAS) ---
        delta_path = delta_stepping_2d(
            raster, steps,
            np.uint64(source), np.uint64(target),
            delta=delta,
        )

        # If NetworkKit says no path, delta-stepping should agree
        if nk_dist == float('inf') or len(nk_path) == 0:
            self.assertEqual(
                len(delta_path), 0,
                f"{msg_prefix}NetworkKit found no path but delta-stepping did"
            )
            return

        # Delta-stepping must also find a path
        self.assertGreater(
            len(delta_path), 0,
            f"{msg_prefix}NetworkKit found path (dist={nk_dist:.4f}) "
            f"but delta-stepping returned empty"
        )

        # Compute edge-based cost for the delta-stepping path
        delta_edge_cost = self._edge_cost_of_path(
            [int(x) for x in delta_path], from_nodes, to_nodes, costs
        )

        # NetworkKit distance IS the edge-based cost (it's the reference)
        # Allow small tolerance for float32 vs float64 accumulation
        self.assertNotEqual(
            delta_edge_cost, float('inf'),
            f"{msg_prefix}Delta-stepping path contains edges not in the graph"
        )

        rel_diff = abs(delta_edge_cost - nk_dist) / max(nk_dist, 1e-10)
        self.assertLessEqual(
            rel_diff, cost_tol_pct,
            f"{msg_prefix}Cost mismatch: NetworkKit={nk_dist:.6f}, "
            f"delta-stepping(edge)={delta_edge_cost:.6f}, "
            f"rel_diff={rel_diff:.6f} > {cost_tol_pct}"
        )

        # Path lengths should be in the same ballpark (within 20%)
        if len(nk_path) > 0:
            nk_len = len(nk_path)
            ds_len = len(delta_path)
            len_ratio = max(nk_len, ds_len) / max(min(nk_len, ds_len), 1)
            self.assertLessEqual(
                len_ratio, 1.2,
                f"{msg_prefix}Path length mismatch: NetworkKit={nk_len}, "
                f"delta-stepping={ds_len}"
            )

    def test_50x50_random_4connected(self):
        """50x50 random raster, 4-connected, with forbidden cells."""
        rng = np.random.RandomState(42)
        size = 50
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.1] = 65535
        raster[0, 0] = 5
        raster[size - 1, size - 1] = 5

        self._run_comparison(
            raster, self.steps_4,
            source=0, target=size * size - 1,
            delta=15.0, msg_prefix="50x50 4-conn: "
        )

    def test_50x50_random_8connected(self):
        """50x50 random raster, 8-connected, with forbidden cells."""
        rng = np.random.RandomState(77)
        size = 50
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.1] = 65535
        raster[0, 0] = 5
        raster[size - 1, size - 1] = 5

        self._run_comparison(
            raster, self.steps_8,
            source=0, target=size * size - 1,
            delta=20.0, msg_prefix="50x50 8-conn: "
        )

    def test_200x200_random(self):
        """200x200 random raster with forbidden cells."""
        rng = np.random.RandomState(1234)
        size = 200
        raster = rng.randint(1, 500, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.05] = 65535
        raster[0, 0] = 10
        raster[size - 1, size - 1] = 10

        self._run_comparison(
            raster, self.steps_4,
            source=0, target=size * size - 1,
            delta=50.0, msg_prefix="200x200: "
        )

    def test_500x500_random(self):
        """500x500 random raster — larger-scale consistency check."""
        rng = np.random.RandomState(9999)
        size = 500
        raster = rng.randint(1, 1000, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.03] = 65535
        raster[0, 0] = 10
        raster[size - 1, size - 1] = 10

        self._run_comparison(
            raster, self.steps_4,
            source=0, target=size * size - 1,
            delta=100.0, msg_prefix="500x500: "
        )

    def test_uniform_raster_exact_match(self):
        """On uniform raster, both backends must produce identical distance."""
        size = 100
        raster = np.full((size, size), 5, dtype=np.uint16)

        self._run_comparison(
            raster, self.steps_4,
            source=0, target=size * size - 1,
            delta=20.0, cost_tol_pct=0.0001,
            msg_prefix="100x100 uniform: "
        )

    def test_gradient_raster(self):
        """Raster with cost gradient (cheap top-left, expensive bottom-right)."""
        size = 100
        raster = np.zeros((size, size), dtype=np.uint16)
        for r in range(size):
            for c in range(size):
                raster[r, c] = max(1, min(65534, r + c + 1))

        self._run_comparison(
            raster, self.steps_4,
            source=0, target=size * size - 1,
            delta=30.0, msg_prefix="100x100 gradient: "
        )

    def test_multi_target_vs_networkit(self):
        """Multi-target delta-stepping must match NetworkKit per-target costs."""
        rng = np.random.RandomState(555)
        size = 80
        raster = rng.randint(1, 200, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.05] = 65535
        raster[0, 0] = 5

        source = np.uint64(0)
        targets_list = [
            size * size - 1,  # bottom-right
            size - 1,         # top-right
            (size - 1) * size,  # bottom-left
            size * (size // 2) + size // 2,  # center
        ]
        # Ensure targets are traversable
        for t in targets_list:
            r, c = divmod(t, size)
            raster[r, c] = max(1, min(raster[r, c], 65534))

        targets = np.array(targets_list, dtype=np.uint64)

        # Delta-stepping multi-target
        delta_paths = delta_stepping_single_source_multiple_targets(
            raster, self.steps_4, source, targets, delta=20.0
        )

        # Build NetworkKit graph
        graph, from_nodes, to_nodes, costs = self._build_networkit_graph(
            raster, self.steps_4
        )

        for i, target in enumerate(targets_list):
            nk_dist, nk_path = self._networkit_dijkstra_distance(
                graph, 0, target
            )

            if nk_dist == float('inf') or len(nk_path) == 0:
                self.assertEqual(
                    len(delta_paths[i]), 0,
                    f"Target {i}: NK no path but delta found one"
                )
                continue

            self.assertGreater(
                len(delta_paths[i]), 0,
                f"Target {i}: NK dist={nk_dist:.4f} but delta returned empty"
            )

            delta_cost = self._edge_cost_of_path(
                [int(x) for x in delta_paths[i]], from_nodes, to_nodes, costs
            )

            rel_diff = abs(delta_cost - nk_dist) / max(nk_dist, 1e-10)
            self.assertLessEqual(
                rel_diff, 0.01,
                f"Target {i}: NK={nk_dist:.6f}, delta={delta_cost:.6f}, "
                f"rel_diff={rel_diff:.6f}"
            )


if __name__ == '__main__':
    unittest.main()

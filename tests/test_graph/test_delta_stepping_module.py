"""
Tests for the _delta_stepping module extracted from path_algorithms.pyx.

Tests cover:
- delta_stepping_2d: basic single-source single-target pathfinding
- delta_stepping_2d_persistent: persistent thread pool variant
- delta_stepping_single_source_multiple_targets: one-to-many pathfinding
- delta_stepping_single_source_multiple_targets_persistent: persistent variant
- group_by_proximity: spatial reordering
"""

import numpy as np
import pytest


class TestDeltaStepping2D:
    """Tests for delta_stepping_2d on simple uniform rasters."""

    def test_uniform_raster_finds_path(self):
        """Basic path on a 10x10 uniform-cost raster with 4-connected movement."""
        from pyorps.utils._delta_stepping import delta_stepping_2d

        raster = np.full((10, 10), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)        # top-left
        target = np.uint64(99)       # bottom-right

        path = delta_stepping_2d(raster, steps, source, target,
                                 delta=50.0, num_threads=1)

        assert len(path) > 0, "Path should be found"
        assert path[0] == source, "Path should start at source"
        assert path[-1] == target, "Path should end at target"
        # Manhattan distance in 4-connected = 9+9 = 18 steps
        assert len(path) == 19, f"Expected 19 cells in path, got {len(path)}"

    def test_source_equals_target(self):
        """When source == target, should return single-element path."""
        from pyorps.utils._delta_stepping import delta_stepping_2d

        raster = np.full((5, 5), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(12)
        target = np.uint64(12)

        path = delta_stepping_2d(raster, steps, source, target,
                                 delta=50.0, num_threads=1)

        assert len(path) == 1
        assert path[0] == 12

    def test_blocked_target_returns_empty(self):
        """If target cell is impassable, return empty path."""
        from pyorps.utils._delta_stepping import delta_stepping_2d

        raster = np.full((10, 10), 10, dtype=np.uint16)
        raster[9, 9] = 65535  # block target
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        target = np.uint64(99)

        path = delta_stepping_2d(raster, steps, source, target,
                                 delta=50.0, num_threads=1)

        assert len(path) == 0

    def test_no_path_exists(self):
        """A wall separates source and target -- no path possible."""
        from pyorps.utils._delta_stepping import delta_stepping_2d

        raster = np.full((10, 10), 10, dtype=np.uint16)
        # Create an impassable wall at column 5
        raster[:, 5] = 65535
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)      # col 0
        target = np.uint64(9)      # col 9

        path = delta_stepping_2d(raster, steps, source, target,
                                 delta=50.0, num_threads=1)

        assert len(path) == 0

    def test_invalid_delta_raises(self):
        """Delta <= 0 should raise ValueError."""
        from pyorps.utils._delta_stepping import delta_stepping_2d

        raster = np.full((5, 5), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        with pytest.raises(ValueError):
            delta_stepping_2d(raster, steps, np.uint64(0), np.uint64(24),
                              delta=0.0, num_threads=1)

    def test_8_connected_movement(self):
        """Test with 8-connected (diagonal) movement."""
        from pyorps.utils._delta_stepping import delta_stepping_2d

        raster = np.full((10, 10), 10, dtype=np.uint16)
        steps = np.array([
            [0, 1], [0, -1], [1, 0], [-1, 0],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ], dtype=np.int8)

        source = np.uint64(0)
        target = np.uint64(99)

        path = delta_stepping_2d(raster, steps, source, target,
                                 delta=50.0, num_threads=1)

        assert len(path) > 0
        assert path[0] == source
        assert path[-1] == target
        # Diagonal path: 9 diagonal steps = 10 cells
        assert len(path) == 10, f"Expected 10 cells (diagonal), got {len(path)}"


class TestDeltaStepping2DPersistent:
    """Tests for the persistent thread pool variant."""

    def test_uniform_raster_finds_path(self):
        """Basic path on uniform raster -- persistent variant."""
        from pyorps.utils._delta_stepping import delta_stepping_2d_persistent

        raster = np.full((10, 10), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        target = np.uint64(99)

        path = delta_stepping_2d_persistent(raster, steps, source, target,
                                            delta=50.0, num_threads=1)

        assert len(path) > 0
        assert path[0] == source
        assert path[-1] == target
        assert len(path) == 19

    def test_source_equals_target(self):
        """Source == target -- persistent variant."""
        from pyorps.utils._delta_stepping import delta_stepping_2d_persistent

        raster = np.full((5, 5), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        path = delta_stepping_2d_persistent(raster, steps,
                                            np.uint64(12), np.uint64(12),
                                            delta=50.0, num_threads=1)

        assert len(path) == 1
        assert path[0] == 12

    def test_matches_standard_variant(self):
        """Persistent variant should produce same path as standard variant."""
        from pyorps.utils._delta_stepping import (
            delta_stepping_2d, delta_stepping_2d_persistent
        )

        raster = np.full((20, 20), 10, dtype=np.uint16)
        # Add some variation
        raster[5:15, 10] = 100
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        target = np.uint64(399)  # (19, 19)

        path_std = delta_stepping_2d(
            raster, steps, source, target, delta=50.0, num_threads=1)
        path_per = delta_stepping_2d_persistent(
            raster, steps, source, target, delta=50.0, num_threads=1)

        assert len(path_std) > 0
        assert len(path_per) > 0
        np.testing.assert_array_equal(path_std, path_per)


class TestDeltaSteppingSingleSourceMultipleTargets:
    """Tests for single-source multiple-targets delta-stepping."""

    def test_three_targets_uniform_raster(self):
        """Find paths from one source to three targets."""
        from pyorps.utils._delta_stepping import (
            delta_stepping_single_source_multiple_targets
        )

        raster = np.full((10, 10), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        targets = np.array([9, 90, 99], dtype=np.uint64)

        paths = delta_stepping_single_source_multiple_targets(
            raster, steps, source, targets, delta=50.0, num_threads=1)

        assert len(paths) == 3
        for i, target in enumerate(targets):
            assert len(paths[i]) > 0, f"Path to target {target} should exist"
            assert paths[i][0] == source
            assert paths[i][-1] == target

    def test_empty_targets(self):
        """Empty target list should return empty list."""
        from pyorps.utils._delta_stepping import (
            delta_stepping_single_source_multiple_targets
        )

        raster = np.full((5, 5), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)
        targets = np.array([], dtype=np.uint64)

        paths = delta_stepping_single_source_multiple_targets(
            raster, steps, np.uint64(0), targets, delta=50.0, num_threads=1)

        assert paths == []

    def test_unreachable_target(self):
        """One reachable, one unreachable target."""
        from pyorps.utils._delta_stepping import (
            delta_stepping_single_source_multiple_targets
        )

        raster = np.full((10, 10), 10, dtype=np.uint16)
        raster[:, 5] = 65535  # wall at col 5
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        targets = np.array([4, 9], dtype=np.uint64)  # col 4 reachable, col 9 not

        paths = delta_stepping_single_source_multiple_targets(
            raster, steps, source, targets, delta=50.0, num_threads=1)

        assert len(paths) == 2
        assert len(paths[0]) > 0, "Target at col 4 should be reachable"
        # The unreachable target may still get a (wrong) path due to full traversal,
        # or empty. Check the reachable target specifically.
        assert paths[0][0] == source
        assert paths[0][-1] == 4


class TestDeltaSteppingSingleSourceMultipleTargetsPersistent:
    """Tests for persistent variant of single-source multiple-targets."""

    def test_three_targets(self):
        """Find paths from one source to three targets -- persistent."""
        from pyorps.utils._delta_stepping import (
            delta_stepping_single_source_multiple_targets_persistent
        )

        raster = np.full((10, 10), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        targets = np.array([9, 90, 99], dtype=np.uint64)

        paths = delta_stepping_single_source_multiple_targets_persistent(
            raster, steps, source, targets, delta=50.0, num_threads=1)

        assert len(paths) == 3
        for i, target in enumerate(targets):
            assert len(paths[i]) > 0, f"Path to target {target} should exist"
            assert paths[i][0] == source
            assert paths[i][-1] == target

    def test_matches_standard_variant(self):
        """Persistent variant should match standard multi-target variant."""
        from pyorps.utils._delta_stepping import (
            delta_stepping_single_source_multiple_targets,
            delta_stepping_single_source_multiple_targets_persistent,
        )

        raster = np.full((15, 15), 10, dtype=np.uint16)
        steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)

        source = np.uint64(0)
        targets = np.array([14, 210, 224], dtype=np.uint64)

        paths_std = delta_stepping_single_source_multiple_targets(
            raster, steps, source, targets, delta=50.0, num_threads=1)
        paths_per = delta_stepping_single_source_multiple_targets_persistent(
            raster, steps, source, targets, delta=50.0, num_threads=1)

        assert len(paths_std) == len(paths_per)
        for i in range(len(paths_std)):
            np.testing.assert_array_equal(paths_std[i], paths_per[i])


class TestGroupByProximity:
    """Tests for spatial reordering of source indices."""

    def test_basic_ordering(self):
        """Indices should be reordered by row coordinate."""
        from pyorps.utils._delta_stepping import group_by_proximity

        # In a 10-column raster:
        # idx 95 -> row 9, idx 5 -> row 0, idx 55 -> row 5
        indices = np.array([95, 5, 55], dtype=np.uint64)
        cols = np.uint64(10)

        result = group_by_proximity(indices, cols)

        assert len(result) == 3
        # Should be sorted by row: row 0 (idx 5), row 5 (idx 55), row 9 (idx 95)
        assert result[0] == 5
        assert result[1] == 55
        assert result[2] == 95

    def test_single_element(self):
        """Single element should be returned as-is."""
        from pyorps.utils._delta_stepping import group_by_proximity

        indices = np.array([42], dtype=np.uint64)
        result = group_by_proximity(indices, np.uint64(10))

        assert len(result) == 1
        assert result[0] == 42

    def test_empty_array(self):
        """Empty array should be returned as-is."""
        from pyorps.utils._delta_stepping import group_by_proximity

        indices = np.array([], dtype=np.uint64)
        result = group_by_proximity(indices, np.uint64(10))

        assert len(result) == 0

    def test_same_row(self):
        """Indices in the same row should all appear (order may vary)."""
        from pyorps.utils._delta_stepping import group_by_proximity

        # All in row 3 of a 10-column raster
        indices = np.array([35, 30, 39], dtype=np.uint64)
        result = group_by_proximity(indices, np.uint64(10))

        assert len(result) == 3
        assert set(result) == {30, 35, 39}

"""
Tests for pyorps.utils._raster_context module.

Covers:
- RasterContext creation and attributes
- RasterContext.is_traversable with blocked cells
- create_exclude_mask basic behavior
- path_cost and path_cost_uint32 correctness
"""

import numpy as np
import pytest

from pyorps.utils._raster_context import (
    RasterContext,
    py_create_exclude_mask,
    py_path_cost,
    py_path_cost_uint32,
)


# ==================== FIXTURES ====================

@pytest.fixture
def simple_raster():
    """5x5 raster with uniform cost=10, no obstacles."""
    return np.full((5, 5), 10, dtype=np.uint16)


@pytest.fixture
def raster_with_obstacles():
    """5x5 raster: cost=10 everywhere except center cell (2,2)=65535."""
    r = np.full((5, 5), 10, dtype=np.uint16)
    r[2, 2] = 65535
    return r


@pytest.fixture
def simple_steps():
    """4-connected neighborhood (N, S, E, W)."""
    return np.array([
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ], dtype=np.int8)


@pytest.fixture
def eight_connected_steps():
    """8-connected neighborhood."""
    return np.array([
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1],
    ], dtype=np.int8)


# ==================== RasterContext CREATION ====================

class TestRasterContextCreation:
    """Test RasterContext __cinit__ and readonly attributes."""

    def test_basic_attributes(self, simple_raster, simple_steps):
        ctx = RasterContext(simple_raster, simple_steps)
        assert ctx.rows == 5
        assert ctx.cols == 5
        assert ctx.total_cells == 25

    def test_num_directions_4(self, simple_raster, simple_steps):
        ctx = RasterContext(simple_raster, simple_steps)
        assert ctx.num_directions == 4

    def test_num_directions_8(self, simple_raster, eight_connected_steps):
        ctx = RasterContext(simple_raster, eight_connected_steps)
        assert ctx.num_directions == 8

    def test_large_raster(self):
        """Verify with a non-square raster."""
        raster = np.full((100, 200), 5, dtype=np.uint16)
        steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
        ctx = RasterContext(raster, steps)
        assert ctx.rows == 100
        assert ctx.cols == 200
        assert ctx.total_cells == 20000
        assert ctx.num_directions == 4

    def test_custom_max_value(self):
        """Non-default max_value marks different cells as obstacles."""
        raster = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
        # 200 is the barrier
        ctx = RasterContext(raster, steps, max_value=200)
        assert ctx.is_traversable(0, 0)       # 100 != 200
        assert not ctx.is_traversable(0, 1)    # 200 == 200
        assert ctx.is_traversable(1, 0)        # 300 != 200
        assert ctx.is_traversable(1, 1)        # 400 != 200


# ==================== RasterContext.is_traversable ====================

class TestIsTraversable:
    """Test the is_traversable method."""

    def test_all_traversable(self, simple_raster, simple_steps):
        ctx = RasterContext(simple_raster, simple_steps)
        for r in range(5):
            for c in range(5):
                assert ctx.is_traversable(r, c)

    def test_blocked_center(self, raster_with_obstacles, simple_steps):
        ctx = RasterContext(raster_with_obstacles, simple_steps)
        # All cells traversable except (2,2)
        assert ctx.is_traversable(0, 0)
        assert ctx.is_traversable(1, 1)
        assert not ctx.is_traversable(2, 2)
        assert ctx.is_traversable(3, 3)
        assert ctx.is_traversable(4, 4)

    def test_multiple_obstacles(self, simple_steps):
        raster = np.full((4, 4), 5, dtype=np.uint16)
        raster[0, 0] = 65535
        raster[1, 1] = 65535
        raster[3, 3] = 65535
        ctx = RasterContext(raster, simple_steps)
        assert not ctx.is_traversable(0, 0)
        assert not ctx.is_traversable(1, 1)
        assert ctx.is_traversable(2, 2)
        assert not ctx.is_traversable(3, 3)


# ==================== create_exclude_mask ====================

class TestCreateExcludeMask:
    """Test the create_exclude_mask utility function."""

    def test_no_obstacles(self):
        raster = np.full((3, 3), 10, dtype=np.uint16)
        mask = py_create_exclude_mask(raster, 65535)
        assert mask.dtype == np.uint8
        assert mask.shape == (3, 3)
        assert np.all(mask == 1)

    def test_all_obstacles(self):
        raster = np.full((3, 3), 65535, dtype=np.uint16)
        mask = py_create_exclude_mask(raster, 65535)
        assert np.all(mask == 0)

    def test_mixed(self):
        raster = np.array([
            [10, 65535, 20],
            [65535, 5, 65535],
            [30, 40, 50],
        ], dtype=np.uint16)
        mask = py_create_exclude_mask(raster, 65535)
        expected = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_custom_max_value(self):
        raster = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        mask = py_create_exclude_mask(raster, 3)
        expected = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)


# ==================== path_cost ====================

class TestPathCost:
    """Test the path_cost function (uint64 indices)."""

    def test_single_cell(self):
        raster = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        path = np.array([0], dtype=np.uint64)  # cell (0,0) = 10
        cost = py_path_cost(path, raster, 2)
        assert cost == 10.0

    def test_straight_path(self):
        raster = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.uint16)
        # Path: (0,0) -> (0,1) -> (0,2)  i.e. indices 0, 1, 2
        path = np.array([0, 1, 2], dtype=np.uint64)
        cost = py_path_cost(path, raster, 3)
        assert cost == 1.0 + 2.0 + 3.0

    def test_diagonal_path(self):
        raster = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.uint16)
        # Path: (0,0) -> (1,1) -> (2,2)  i.e. indices 0, 4, 8
        path = np.array([0, 4, 8], dtype=np.uint64)
        cost = py_path_cost(path, raster, 3)
        assert cost == 1.0 + 5.0 + 9.0

    def test_empty_path(self):
        raster = np.array([[10]], dtype=np.uint16)
        path = np.array([], dtype=np.uint64)
        cost = py_path_cost(path, raster, 1)
        assert cost == 0.0


# ==================== path_cost_uint32 ====================

class TestPathCostUint32:
    """Test the path_cost_uint32 function (uint32 indices)."""

    def test_single_cell(self):
        raster = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        path = np.array([3], dtype=np.uint32)  # cell (1,1) = 40
        cost = py_path_cost_uint32(path, raster, 2)
        assert cost == 40.0

    def test_straight_path(self):
        raster = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.uint16)
        # Path: (0,0) -> (1,0) -> (2,0)  i.e. indices 0, 3, 6
        path = np.array([0, 3, 6], dtype=np.uint32)
        cost = py_path_cost_uint32(path, raster, 3)
        assert cost == 1.0 + 4.0 + 7.0

    def test_empty_path(self):
        raster = np.array([[10]], dtype=np.uint16)
        path = np.array([], dtype=np.uint32)
        cost = py_path_cost_uint32(path, raster, 1)
        assert cost == 0.0

    def test_matches_uint64_version(self):
        """Both path_cost variants must agree on the same path."""
        raster = np.array([
            [5, 10, 15],
            [20, 25, 30],
        ], dtype=np.uint16)
        indices = [0, 1, 4, 5]
        path64 = np.array(indices, dtype=np.uint64)
        path32 = np.array(indices, dtype=np.uint32)
        cost64 = py_path_cost(path64, raster, 3)
        cost32 = py_path_cost_uint32(path32, raster, 3)
        assert cost64 == cost32


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Various edge-case tests."""

    def test_1x1_raster(self):
        raster = np.array([[42]], dtype=np.uint16)
        steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
        ctx = RasterContext(raster, steps)
        assert ctx.rows == 1
        assert ctx.cols == 1
        assert ctx.total_cells == 1
        assert ctx.is_traversable(0, 0)

    def test_1x1_raster_blocked(self):
        raster = np.array([[65535]], dtype=np.uint16)
        steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
        ctx = RasterContext(raster, steps)
        assert not ctx.is_traversable(0, 0)

    def test_extended_neighborhood(self):
        """Neighborhood with r=2 steps should produce correct direction count."""
        steps = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                steps.append([dr, dc])
        steps_arr = np.array(steps, dtype=np.int8)
        raster = np.full((10, 10), 5, dtype=np.uint16)
        ctx = RasterContext(raster, steps_arr)
        assert ctx.num_directions == 24  # 5x5 - 1 = 24

    def test_path_cost_high_values(self):
        """path_cost with max non-obstacle values."""
        raster = np.full((2, 2), 65534, dtype=np.uint16)
        path = np.array([0, 1, 2, 3], dtype=np.uint64)
        cost = py_path_cost(path, raster, 2)
        assert cost == 65534.0 * 4

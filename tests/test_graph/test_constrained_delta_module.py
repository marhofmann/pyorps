"""Tests for _constrained_delta module (parallel constrained delta-stepping).

Verifies that all five extracted functions are importable and produce valid
results on small grids.
"""

import numpy as np
import pytest


def _make_inputs():
    """Create minimal inputs for constrained delta-stepping."""
    rows, cols = 10, 10
    raster = np.ones((rows, cols), dtype=np.uint16) * 10
    steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)
    n_dirs = 4
    angle_cost = np.zeros((n_dirs, n_dirs), dtype=np.float32)
    angle_valid = np.ones((n_dirs, n_dirs), dtype=np.uint8)
    step_distances = np.ones(n_dirs, dtype=np.float32) * 10.0
    tower_terrain = np.ones(65536, dtype=np.float32) * 5.0
    tower_angle = np.zeros((n_dirs, n_dirs), dtype=np.float32)
    return (raster, steps, angle_cost, angle_valid, step_distances,
            tower_terrain, tower_angle)


class TestConstrainedDeltaImport:
    """Verify all public functions are importable."""

    def test_import_basic(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_2d
        assert callable(constrained_delta_stepping_2d)

    def test_import_clearance(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_clearance_2d
        assert callable(constrained_delta_stepping_clearance_2d)

    def test_import_height(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_height_2d
        assert callable(constrained_delta_stepping_height_2d)

    def test_import_lazy(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_lazy
        assert callable(constrained_delta_stepping_lazy)


class TestConstrainedDeltaBasic:
    """Test constrained_delta_stepping_2d on small grids."""

    def test_basic_path(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_2d
        args = _make_inputs()
        path, towers = constrained_delta_stepping_2d(
            args[0], 0, 0, 9, 9, args[1],
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
        )
        assert isinstance(path, np.ndarray)
        assert len(path) > 0
        assert path[0] == 0  # source (0,0) -> cell index 0
        assert path[-1] == 99  # target (9,9) -> cell index 99

    def test_no_path_blocked(self):
        """If raster is all impassable, should return empty path."""
        from pyorps.utils._constrained_delta import constrained_delta_stepping_2d
        args = _make_inputs()
        raster = np.ones((10, 10), dtype=np.uint16) * 65535
        # Make source and target traversable but no connecting path
        raster[0, 0] = 10
        raster[9, 9] = 10
        path, towers = constrained_delta_stepping_2d(
            raster, 0, 0, 9, 9, args[1],
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
        )
        assert len(path) == 0

    def test_with_dem(self):
        """Test basic function with DEM gradient data."""
        from pyorps.utils._constrained_delta import constrained_delta_stepping_2d
        args = _make_inputs()
        dem = np.zeros((10, 10), dtype=np.float32)
        path, towers = constrained_delta_stepping_2d(
            args[0], 0, 0, 9, 9, args[1],
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
            dem_data=dem, cell_size=10.0,
        )
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 99


class TestConstrainedDeltaClearance:
    """Test constrained_delta_stepping_clearance_2d."""

    def test_clearance_path(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_clearance_2d
        args = _make_inputs()
        dem = np.zeros((10, 10), dtype=np.float32)
        path, towers = constrained_delta_stepping_clearance_2d(
            args[0], 0, 0, 9, 9, args[1],
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
            dem_data=dem, cell_size=10.0,
            tower_height=50.0,
            conductor_weight_per_m=2.0,
            conductor_tension=50000.0,
            min_clearance_val=8.0,
        )
        assert isinstance(path, np.ndarray)
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 99


class TestConstrainedDeltaHeight:
    """Test constrained_delta_stepping_height_2d."""

    def test_height_path(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_height_2d
        args = _make_inputs()
        dem = np.zeros((10, 10), dtype=np.float32)
        # Heights sorted descending for early exit optimization
        tower_heights = np.array([50.0, 40.0, 30.0], dtype=np.float32)
        height_premiums = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        path, towers, tower_h = constrained_delta_stepping_height_2d(
            args[0], 0, 0, 9, 9, args[1],
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
            dem_data=dem, cell_size=10.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=2.0,
            conductor_tension=50000.0,
            min_clearance_val=8.0,
        )
        assert isinstance(path, np.ndarray)
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 99
        assert isinstance(tower_h, np.ndarray)
        assert tower_h.dtype == np.float32


class TestConstrainedDeltaLazy:
    """Test constrained_delta_stepping_lazy."""

    def test_lazy_path(self):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_lazy
        args = _make_inputs()
        dem = np.zeros((10, 10), dtype=np.float32)
        tower_heights = np.array([50.0], dtype=np.float32)
        height_premiums = np.array([0.0], dtype=np.float32)
        path, towers, tower_h = constrained_delta_stepping_lazy(
            args[0], 0, 0, 9, 9, args[1],
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
            dem_data=dem, cell_size=10.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=2.0,
            conductor_tension=50000.0,
            min_clearance_val=8.0,
        )
        assert isinstance(path, np.ndarray)
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 99
        assert isinstance(tower_h, np.ndarray)

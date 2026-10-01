"""Tests for _constrained_dijkstra module."""
import numpy as np
import pytest


def _make_simple_inputs():
    """Create minimal constrained pathfinding inputs."""
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


class TestConstrainedDijkstraImport:
    def test_import(self):
        from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d
        assert callable(constrained_dijkstra_2d)


class TestConstrainedDijkstraBasic:
    def test_returns_tuple(self):
        from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d
        args = _make_simple_inputs()
        raster, steps = args[0], args[1]
        result = constrained_dijkstra_2d(
            raster, 0, 0, 9, 9, steps,
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        path, towers = result
        assert path.dtype == np.uint32

    def test_path_endpoints(self):
        from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d
        args = _make_simple_inputs()
        raster, steps = args[0], args[1]
        path, towers = constrained_dijkstra_2d(
            raster, 0, 0, 9, 9, steps,
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
        )
        assert len(path) > 0
        # First cell should be source (0,0) = index 0
        assert path[0] == 0
        # Last cell should be target (9,9) = index 99
        assert path[-1] == 99

    def test_no_path_blocked(self):
        from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d
        args = _make_simple_inputs()
        raster, steps = args[0], args[1]
        # Block the entire middle row
        raster[5, :] = 65535
        path, towers = constrained_dijkstra_2d(
            raster, 0, 0, 9, 9, steps,
            angle_cost_lut=args[2], angle_valid_lut=args[3],
            step_distances=args[4], tower_terrain_costs=args[5],
            tower_angle_costs=args[6],
            n_span_bins=10, span_bin_size=20.0,
            min_span=40.0, max_span=200.0,
        )
        assert len(path) == 0

"""
Tests for the fused delta-stepping kernel (_delta_stepping_fused).

Covers:
- API contract parity with delta_stepping_2d (path shape, empty on no-path)
- Cost-optimality vs Dijkstra across raster patterns, neighborhoods,
  thread counts, and all lever combinations (fusion / window / adaptive)
- Parameter validation
"""

import numpy as np
import pytest

from pyorps.utils._delta_stepping_fused import delta_stepping_2d_fused
from pyorps.utils._dijkstra import dijkstra_2d_cython
from pyorps.utils._raster_context import path_cost

STEPS_4 = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)
STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)
STEPS_16 = np.vstack([STEPS_8, np.array([
    [1, 2], [2, 1], [-1, 2], [-2, 1],
    [1, -2], [2, -1], [-1, -2], [-2, -1]
], dtype=np.int8)])

# (fusion_cap, window_init, adaptive) lever combinations
LEVER_COMBOS = [
    pytest.param(0, 1, False, id="no-levers"),
    pytest.param(1024, 1, False, id="fusion-only"),
    pytest.param(0, 8, True, id="window-only"),
    pytest.param(1024, 8, True, id="fusion+window"),
    pytest.param(256, 32, True, id="big-window-small-cap"),
]


def _cost(path, raster):
    return path_cost(np.asarray(path, dtype=np.uint64), raster,
                     raster.shape[1])


def _dijkstra_cost(raster, steps, src, tgt):
    p = dijkstra_2d_cython(raster, steps, np.uint32(src), np.uint32(tgt))
    assert len(p) > 0
    return _cost(p, raster)


class TestContract:
    """API contract parity with delta_stepping_2d."""

    def test_uniform_raster_finds_path(self):
        raster = np.full((10, 10), 10, dtype=np.uint16)
        path = delta_stepping_2d_fused(raster, STEPS_4, np.uint64(0),
                                       np.uint64(99), delta=50.0,
                                       num_threads=1)
        assert len(path) == 19
        assert path[0] == 0
        assert path[-1] == 99

    def test_source_equals_target(self):
        raster = np.full((5, 5), 10, dtype=np.uint16)
        path = delta_stepping_2d_fused(raster, STEPS_4, np.uint64(12),
                                       np.uint64(12), delta=50.0,
                                       num_threads=1)
        assert len(path) == 1 and path[0] == 12

    def test_blocked_target_returns_empty(self):
        raster = np.full((10, 10), 10, dtype=np.uint16)
        raster[9, 9] = 65535
        path = delta_stepping_2d_fused(raster, STEPS_4, np.uint64(0),
                                       np.uint64(99), delta=50.0,
                                       num_threads=1)
        assert len(path) == 0

    def test_no_path_exists(self):
        raster = np.full((10, 10), 10, dtype=np.uint16)
        raster[:, 5] = 65535
        path = delta_stepping_2d_fused(raster, STEPS_4, np.uint64(0),
                                       np.uint64(9), delta=50.0,
                                       num_threads=1)
        assert len(path) == 0

    def test_invalid_delta_raises(self):
        raster = np.full((5, 5), 10, dtype=np.uint16)
        with pytest.raises(ValueError):
            delta_stepping_2d_fused(raster, STEPS_4, np.uint64(0),
                                    np.uint64(24), delta=0.0, num_threads=1)

    def test_window_exceeding_buffer_raises(self):
        raster = np.full((10, 10), 10, dtype=np.uint16)
        with pytest.raises(ValueError):
            delta_stepping_2d_fused(raster, STEPS_4, np.uint64(0),
                                    np.uint64(99), delta=50.0, num_threads=1,
                                    max_buckets_in_memory=32,
                                    window_init=64, window_max=64)

    def test_path_steps_are_valid_moves(self):
        rng = np.random.default_rng(1)
        raster = rng.integers(1, 100, (30, 30), dtype=np.uint16)
        path = delta_stepping_2d_fused(raster, STEPS_8, np.uint64(0),
                                       np.uint64(899), delta=50.0,
                                       num_threads=1)
        moves = {(int(dr), int(dc)) for dr, dc in STEPS_8}
        for a, b in zip(path[:-1], path[1:]):
            dr = int(b // 30) - int(a // 30)
            dc = int(b % 30) - int(a % 30)
            assert (dr, dc) in moves


def _make_rasters(rng, shape):
    rasters = {}
    rasters["uniform"] = np.full(shape, 7, dtype=np.uint16)
    rasters["random"] = rng.integers(1, 200, shape, dtype=np.uint16)
    heavy = np.clip(
        rng.lognormal(2.0, 1.5, shape), 1, 5000).astype(np.uint16)
    rasters["heavy_tail"] = heavy
    obst = rng.integers(1, 50, shape, dtype=np.uint16)
    mask = rng.random(shape) < 0.3
    obst[mask] = 65535
    obst[0, 0] = 1
    obst[-1, -1] = 1
    rasters["obstacles"] = obst
    wall = np.full(shape, 5, dtype=np.uint16)
    wall[:, shape[1] // 2] = 65535
    wall[shape[0] // 2, shape[1] // 2] = 5  # single gap
    rasters["wall_gap"] = wall
    return rasters


class TestOptimality:
    """Cost equality vs Dijkstra across patterns and lever combos."""

    @pytest.mark.parametrize("fusion_cap,window_init,adaptive", LEVER_COMBOS)
    @pytest.mark.parametrize("pattern", ["uniform", "random", "heavy_tail",
                                         "obstacles", "wall_gap"])
    def test_matches_dijkstra_8connected(self, pattern, fusion_cap,
                                         window_init, adaptive):
        rng = np.random.default_rng(7)
        shape = (60, 70)
        raster = _make_rasters(rng, shape)[pattern]
        src, tgt = 0, shape[0] * shape[1] - 1
        c_ref = _dijkstra_cost(raster, STEPS_8, src, tgt)
        path = delta_stepping_2d_fused(
            raster, STEPS_8, np.uint64(src), np.uint64(tgt), delta=100.0,
            num_threads=2, fusion_cap=fusion_cap, window_init=window_init,
            adaptive_window=adaptive)
        assert len(path) > 0
        c = _cost(path, raster)
        assert c == pytest.approx(c_ref, rel=1e-4)

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    def test_thread_invariance(self, num_threads):
        rng = np.random.default_rng(11)
        raster = rng.integers(1, 300, (80, 90), dtype=np.uint16)
        src, tgt = 0, 80 * 90 - 1
        c_ref = _dijkstra_cost(raster, STEPS_8, src, tgt)
        path = delta_stepping_2d_fused(
            raster, STEPS_8, np.uint64(src), np.uint64(tgt), delta=100.0,
            num_threads=num_threads, fusion_cap=1024, window_init=8,
            adaptive_window=True)
        assert _cost(path, raster) == pytest.approx(c_ref, rel=1e-4)

    def test_16_neighborhood(self):
        rng = np.random.default_rng(13)
        raster = rng.integers(1, 150, (50, 55), dtype=np.uint16)
        src, tgt = 0, 50 * 55 - 1
        c_ref = _dijkstra_cost(raster, STEPS_16, src, tgt)
        path = delta_stepping_2d_fused(
            raster, STEPS_16, np.uint64(src), np.uint64(tgt), delta=100.0,
            num_threads=2, fusion_cap=1024, window_init=8,
            adaptive_window=True)
        assert _cost(path, raster) == pytest.approx(c_ref, rel=1e-4)

    def test_small_delta_many_buckets(self):
        """Tiny delta stresses window advancement over many buckets."""
        rng = np.random.default_rng(17)
        raster = rng.integers(1, 40, (40, 40), dtype=np.uint16)
        src, tgt = 0, 40 * 40 - 1
        c_ref = _dijkstra_cost(raster, STEPS_8, src, tgt)
        path = delta_stepping_2d_fused(
            raster, STEPS_8, np.uint64(src), np.uint64(tgt), delta=5.0,
            num_threads=2, fusion_cap=1024, window_init=8,
            adaptive_window=True, max_buckets_in_memory=4096)
        assert _cost(path, raster) == pytest.approx(c_ref, rel=1e-4)

    def test_repeated_runs_stable(self):
        """Costs are stable across repeated parallel runs (race check)."""
        rng = np.random.default_rng(19)
        raster = rng.integers(1, 500, (70, 70), dtype=np.uint16)
        src, tgt = 0, 70 * 70 - 1
        c_ref = _dijkstra_cost(raster, STEPS_8, src, tgt)
        for _ in range(5):
            path = delta_stepping_2d_fused(
                raster, STEPS_8, np.uint64(src), np.uint64(tgt), delta=100.0,
                num_threads=4, fusion_cap=1024, window_init=8,
                adaptive_window=True)
            assert _cost(path, raster) == pytest.approx(c_ref, rel=1e-4)

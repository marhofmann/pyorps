import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import os

# The constrained GPU planners V1-V4 are not production code and the plan
# recommends retiring them (item 3.8: V1 cannot fit 6 GB, V2-managed can
# exhaust system RAM, V3 hangs on real rasters). Measured 2026-08-10: running
# them pins the GPU at 100% and 84 C for minutes without completing a single
# test, and the CUDA context can outlive a killed process, leaving the card
# clocked up and unusable for anything else. Opt in deliberately:
#   PYORPS_RUN_CONSTRAINED_GPU=1 pytest tests/test_graph/test_constrained_gpu_v2.py
RUN_CONSTRAINED_GPU = os.environ.get(
    "PYORPS_RUN_CONSTRAINED_GPU", "").strip().lower() not in ("", "0", "false")

pytestmark = [
    pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available"),
    pytest.mark.skipif(
        not RUN_CONSTRAINED_GPU,
        reason="constrained GPU planners are unvalidated and can wedge the "
               "GPU; set PYORPS_RUN_CONSTRAINED_GPU=1 to run"),
]


class TestStateEncoding:
    """Test 4-field state packing/unpacking for GPU kernel."""

    def test_pack_unpack_roundtrip(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import pack_state, unpack_state
        cell, d, sb, hc = 12345, 7, 3, 2
        n_dirs, n_span_bins, n_heights = 32, 6, 3
        spc = n_dirs * n_span_bins * n_heights
        state = pack_state(cell, d, sb, hc, spc, n_span_bins, n_heights)
        c2, d2, sb2, hc2 = unpack_state(state, spc, n_span_bins, n_heights)
        assert (c2, d2, sb2, hc2) == (cell, d, sb, hc)

    def test_unique_states(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import pack_state
        n_dirs, n_span_bins, n_heights = 8, 6, 3
        spc = n_dirs * n_span_bins * n_heights
        states = set()
        for cell in range(100):
            for d in range(n_dirs):
                for sb in range(n_span_bins):
                    for hc in range(n_heights):
                        s = pack_state(cell, d, sb, hc, spc, n_span_bins, n_heights)
                        assert s not in states, f"Collision at {cell},{d},{sb},{hc}"
                        states.add(s)

    def test_large_cell_index(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import pack_state, unpack_state
        cell = 2_000_000
        n_dirs, n_span_bins, n_heights = 32, 6, 3
        spc = n_dirs * n_span_bins * n_heights
        state = pack_state(cell, 31, 5, 2, spc, n_span_bins, n_heights)
        c, d, sb, hc = unpack_state(state, spc, n_span_bins, n_heights)
        assert (c, d, sb, hc) == (cell, 31, 5, 2)


class TestMemoryBudget:
    """Test memory allocation and budget estimation."""

    def test_budget_computation(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import compute_memory_budget_gb
        gb = compute_memory_budget_gb(
            rows=2000, cols=2000, n_dirs=32, n_span_bins=6, n_heights=3)
        assert 13.0 < gb < 14.0  # ~13.4 GB for 2000x2000 R3

    def test_budget_raises_on_exceed(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import check_memory_fits
        with pytest.raises(MemoryError):
            check_memory_fits(
                rows=4000, cols=4000, n_dirs=48, n_span_bins=6, n_heights=3,
                vram_gb=16.0)


class TestBasicKernel:
    """Test the v2 persistent cooperative constrained kernel."""

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing.

        Returns (raster, params_dict).
        """
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_v2",
            "description": "test profile for v2 kernel",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    def test_uniform_raster_finds_path(self):
        """Source (5,5) to target (45,45) on uniform 50x50 raster."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            **params)

        assert len(path) > 0, "Path should not be empty"
        assert path[0] == 5 * 50 + 5, f"Path start {path[0]} != source {5*50+5}"
        assert path[-1] == 45 * 50 + 45, f"Path end {path[-1]} != target {45*50+45}"
        # Distance ~566m with max_span_tracked=350m -> at least 1 tower
        assert len(towers) >= 1, (
            f"Expected at least 1 tower for ~566m path with max_span=300m, "
            f"got {len(towers)}")
        assert len(heights) == len(towers)

    def test_no_feasible_path_returns_empty(self):
        """Block entire middle band with 65535 so no path exists."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        raster[20:30, :] = 65535

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            **params)

        assert len(path) == 0, f"Expected empty path, got {len(path)} cells"
        assert len(towers) == 0
        assert len(heights) == 0

    def test_source_on_forbidden_raises(self):
        """Source cell = 65535 should raise ValueError."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2)

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10)

        raster[5, 5] = 65535

        with pytest.raises(ValueError, match="Source cell.*forbidden"):
            constrained_sssp_raster_gpu_v2(
                raster=raster,
                source_row=5, source_col=5,
                target_row=45, target_col=45,
                steps=steps,
                **params)

    def test_target_on_forbidden_raises(self):
        """Target cell = 65535 should raise ValueError."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2)

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10)

        raster[45, 45] = 65535

        with pytest.raises(ValueError, match="Target cell.*forbidden"):
            constrained_sssp_raster_gpu_v2(
                raster=raster,
                source_row=5, source_col=5,
                target_row=45, target_col=45,
                steps=steps,
                **params)

    def test_adjacent_cells_no_tower_needed(self):
        """Source and target 1 cell apart should find path with no towers."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=10, cols=10, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=4, source_col=4,
            target_row=5, target_col=5,
            steps=steps,
            **params)

        assert len(path) > 0, "Should find a path between adjacent cells"
        # Distance is ~14.1m (diagonal), well under min_span=50m
        assert len(towers) == 0, (
            f"Expected no towers for 14m path, got {len(towers)}")

    def test_long_straight_path_multiple_towers(self):
        """Long horizontal path should produce multiple towers."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=10, cols=100, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=150.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=0,
            target_row=5, target_col=99,
            steps=steps,
            **params)

        # Path is ~990m horizontal. With max_span_tracked=200m (4 bins * 50m),
        # we need towers roughly every 200m -> ~4-5 towers minimum.
        assert len(path) > 0, "Should find a path"
        assert len(towers) >= 4, (
            f"Expected at least 4 towers for ~990m path with max_span=150m, "
            f"got {len(towers)}")
        assert len(heights) == len(towers)


class TestPathReconstruction:
    """Test CPU-side path reconstruction from tower records."""

    def test_direction_walk_backward_straight(self):
        """Walking backward along a straight horizontal line."""
        from pyorps.utils.constrained_sssp_gpu_v2 import _direction_walk_backward
        from pyorps.utils.neighborhood import get_neighborhood_steps

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)
        cols = 50

        # Find east direction (0, 1)
        east_dir = None
        for i in range(n_dirs):
            if steps[i, 0] == 0 and steps[i, 1] == 1:
                east_dir = i
                break
        assert east_dir is not None, "R1 should have east direction (0,1)"

        target_cell = 5 * cols + 10
        source_cell = 5 * cols + 5
        path = _direction_walk_backward(
            target_cell, east_dir, source_cell, cols, steps, n_dirs)

        assert path[0] == source_cell
        assert path[-1] == target_cell
        assert len(path) == 6  # 5,5 -> 5,6 -> 5,7 -> 5,8 -> 5,9 -> 5,10

    def test_reconstruct_no_towers(self):
        """Reconstruction with 0 tower records gives direct path."""
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            _reconstruct_from_tower_records, pack_state)
        from pyorps.utils.neighborhood import get_neighborhood_steps

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)
        n_span_bins = 7
        n_heights = 1
        spc = n_dirs * n_span_bins * n_heights
        cols = 50

        # Find east direction
        east_dir = None
        for i in range(n_dirs):
            if steps[i, 0] == 0 and steps[i, 1] == 1:
                east_dir = i
                break

        source_cell = 5 * cols + 5
        target_cell = 5 * cols + 8
        best_state = pack_state(target_cell, east_dir, 1, 0,
                                spc, n_span_bins, n_heights)

        tower_dtype = np.dtype([
            ('state', np.int64),
            ('pred_state', np.int64),
            ('span_dist', np.float16),
            ('tower_height', np.float16),
            ('_pad', np.uint8, 4),
        ])
        empty_records = np.zeros(0, dtype=tower_dtype)

        path, towers, heights = _reconstruct_from_tower_records(
            empty_records, 0, best_state, source_cell,
            spc, n_span_bins, n_heights, n_dirs, cols, steps)

        assert len(path) > 0
        assert path[0] == source_cell
        assert path[-1] == target_cell
        assert len(towers) == 0
        assert len(heights) == 0


class TestClearanceAndHeight:
    """Test clearance checking and variable height support (Task 3)."""

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing.

        Returns (raster, params_dict).
        """
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_clearance",
            "description": "test profile for clearance tests",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    def test_flat_dem_no_obstacles_finds_path(self):
        """Flat DEM with no obstacles should find path same as without DEM."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        dem = np.zeros((50, 50), dtype=np.float32)  # flat DEM
        # Tower heights sorted descending; single height of 30m
        tower_heights_arr = np.array([30.0], dtype=np.float32)

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            dem=dem,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=7.0,
            tower_heights=tower_heights_arr,
            **params)

        assert len(path) > 0, "Path should not be empty with flat DEM"
        assert path[0] == 5 * 50 + 5
        assert path[-1] == 45 * 50 + 45
        assert len(towers) >= 1, (
            f"Expected at least 1 tower for ~566m path, got {len(towers)}")
        assert len(heights) == len(towers)

    def test_tall_obstacle_forces_taller_tower(self):
        """20m obstacle in span should force selection of taller tower."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=10, cols=100, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=200.0,
            span_bin_size=50.0)

        dem = np.zeros((10, 100), dtype=np.float32)
        obstacle = np.zeros((10, 100), dtype=np.float32)
        # Place 20m obstacle band across columns 40-60
        obstacle[:, 40:60] = 20.0

        # Heights sorted descending: 42, 34, 25
        # With min_clearance=7:
        #   25m tower: max clearance = 25 - 7 = 18 < 20m obstacle -> fails
        #   34m tower: max clearance = 34 - 7 = 27 >= 20m -> may pass
        #   42m tower: max clearance = 42 - 7 = 35 >= 20m -> should pass
        tower_heights_arr = np.array([42.0, 34.0, 25.0], dtype=np.float32)
        height_premiums_arr = np.array([500.0, 200.0, 0.0], dtype=np.float32)

        path, towers, heights_out = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=0,
            target_row=5, target_col=99,
            steps=steps,
            height_premiums=height_premiums_arr,
            n_heights=3,
            dem=dem,
            obstacle_heights=obstacle,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=7.0,
            tower_heights=tower_heights_arr,
            **params)

        assert len(path) > 0, "Path should exist (taller towers can clear 20m)"
        assert len(towers) >= 1
        # At least one tower near the obstacle zone should have height > 25
        # (the short tower cannot clear 20m obstacles with 7m min clearance)
        assert any(h > 25.0 for h in heights_out), (
            f"Expected at least one tower > 25m to clear 20m obstacle, "
            f"got heights: {heights_out}")

    def test_impassable_obstacle_no_path(self):
        """Obstacle taller than all towers should block the path.

        Uses a long narrow corridor (1 row high) so that the path MUST
        place towers within the obstacle zone. The obstacle is tall enough
        that even the tallest tower cannot clear it, forcing all tower
        placements that span the obstacle zone to fail.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        # Use a 3xN corridor: only row 1 is traversable, others blocked
        rows, cols = 3, 100
        raster, params = self._make_params(
            steps, rows=rows, cols=cols, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=150.0,
            span_bin_size=50.0)
        raster[0, :] = 65535  # block row 0
        raster[2, :] = 65535  # block row 2

        dem = np.zeros((rows, cols), dtype=np.float32)
        obstacle = np.zeros((rows, cols), dtype=np.float32)
        # 50m obstacle wall in the only traversable row (40-60)
        # Path must traverse through this, and max_span=150 forces
        # at least one tower whose span crosses the obstacle zone.
        obstacle[1, 40:60] = 50.0

        # Even tallest tower (42m) - 7m clearance = 35m < 50m obstacle
        tower_heights_arr = np.array([42.0, 34.0, 25.0], dtype=np.float32)
        height_premiums_arr = np.array([500.0, 200.0, 0.0], dtype=np.float32)

        path, towers, heights_out = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=1, source_col=0,
            target_row=1, target_col=99,
            steps=steps,
            height_premiums=height_premiums_arr,
            n_heights=3,
            dem=dem,
            obstacle_heights=obstacle,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=7.0,
            tower_heights=tower_heights_arr,
            **params)

        assert len(path) == 0, (
            f"Expected no path when obstacle (50m) exceeds all tower heights "
            f"in a single-row corridor, got {len(path)} cells")

    def test_gradient_penalty_avoids_steep(self):
        """Steep DEM slope should steer route around or block steep cells."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Create DEM with a very steep ridge across columns 20-30
        # Rise of 100m over 1 cell (10m) = 1000% slope, far exceeds 40% limit
        dem = np.zeros((50, 50), dtype=np.float32)
        dem[:, 25:] = 100.0  # 100m step at col boundary 24->25

        tower_heights_arr = np.array([30.0], dtype=np.float32)

        # With max_gradient_pct=40, the steep edge cells are rejected
        # The wall spans the full height, so no path around exists
        path, towers, heights_out = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            dem=dem,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            max_gradient_pct=40.0,
            gradient_scale=2.0,
            tower_heights=tower_heights_arr,
            **params)

        # The steep wall blocks all east-west traversal, and target is
        # on the other side -> no path
        assert len(path) == 0, (
            f"Expected no path through 1000% gradient wall with 40% limit, "
            f"got {len(path)} cells")


class TestWarpCooperative:
    """Test warp-cooperative tower placement protocol (Task 4).

    The warp-cooperative protocol replaces inline tower placement with
    parallel clearance checking across 32 warp lanes. These tests verify
    that correctness is unchanged and clearance logic still works.
    """

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing."""
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_warp_coop",
            "description": "test profile for warp-cooperative tests",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    def test_correctness_unchanged(self):
        """Warp-cooperative produces same results as before.

        Runs the same test case as test_uniform_raster_finds_path and
        verifies path found, towers placed, same general behavior.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            **params)

        assert len(path) > 0, "Path should not be empty"
        assert path[0] == 5 * 50 + 5, f"Path start {path[0]} != source"
        assert path[-1] == 45 * 50 + 45, f"Path end {path[-1]} != target"
        assert len(towers) >= 1, (
            f"Expected at least 1 tower for ~566m path with max_span=300m, "
            f"got {len(towers)}")
        assert len(heights) == len(towers)

    def test_clearance_still_works(self):
        """Warp-cooperative clearance matches sequential behavior.

        Same scenario as test_tall_obstacle_forces_taller_tower: 20m obstacle
        should force selection of taller tower height class.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=10, cols=100, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=200.0,
            span_bin_size=50.0)

        dem = np.zeros((10, 100), dtype=np.float32)
        obstacle = np.zeros((10, 100), dtype=np.float32)
        obstacle[:, 40:60] = 20.0  # 20m obstacle band

        tower_heights_arr = np.array([42.0, 34.0, 25.0], dtype=np.float32)
        height_premiums_arr = np.array([500.0, 200.0, 0.0], dtype=np.float32)

        path, towers, heights_out = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=0,
            target_row=5, target_col=99,
            steps=steps,
            height_premiums=height_premiums_arr,
            n_heights=3,
            dem=dem,
            obstacle_heights=obstacle,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=7.0,
            tower_heights=tower_heights_arr,
            **params)

        assert len(path) > 0, "Path should exist (taller towers can clear 20m)"
        assert len(towers) >= 1
        # At least one tower should be taller than 25m to clear the obstacle
        assert any(h > 25.0 for h in heights_out), (
            f"Expected at least one tower > 25m to clear 20m obstacle, "
            f"got heights: {heights_out}")

    def test_warp_coop_long_span_clearance(self):
        """Long span clearance is correctly parallelized across warp lanes.

        Uses a scenario with a long span (many cells to check) to verify
        that the parallel clearance walk produces correct results when the
        work is distributed across all 32 lanes.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        # Large cells so span covers many cells (span_cells = span/cell_size)
        raster, params = self._make_params(
            steps, rows=10, cols=200, raster_val=10,
            cell_size=5.0, min_span=50.0, max_span=250.0,
            span_bin_size=50.0)

        dem = np.zeros((10, 200), dtype=np.float32)
        # Place a small obstacle at a specific point in the span
        # This tests that the parallel clearance correctly detects the
        # obstacle even when it falls on a specific lane's check cell
        obstacle = np.zeros((10, 200), dtype=np.float32)
        obstacle[5, 95] = 15.0  # Single cell 15m obstacle

        tower_heights_arr = np.array([25.0, 15.0], dtype=np.float32)
        height_premiums_arr = np.array([200.0, 0.0], dtype=np.float32)

        path, towers, heights_out = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=0,
            target_row=5, target_col=199,
            steps=steps,
            height_premiums=height_premiums_arr,
            n_heights=2,
            dem=dem,
            obstacle_heights=obstacle,
            cell_size=5.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=7.0,
            tower_heights=tower_heights_arr,
            **params)

        assert len(path) > 0, "Path should exist with taller tower option"
        assert len(towers) >= 1

    def test_no_dem_skips_clearance(self):
        """Without DEM, clearance checking is skipped (fast path)."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # No DEM passed -- should work fine, clearance skipped
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            **params)

        assert len(path) > 0, "Path should not be empty without DEM"
        assert path[0] == 5 * 50 + 5
        assert path[-1] == 45 * 50 + 45


class TestAreaCost:
    """Test rotated square footprint area cost (Task 5).

    When area_offsets are provided, tower terrain cost sums all pixels in
    the footprint. Forbidden pixels (65535) reject the tower placement.
    Slope multiplier is averaged over the footprint.
    """

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing."""
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_area",
            "description": "test profile for area cost tests",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    @staticmethod
    def _make_simple_area_offsets(n_dirs, radius=1):
        """Create simple area offsets: a square of (2*radius+1)^2 pixels.

        For simplicity, all direction pairs use the same square offsets
        (no rotation). This tests the mechanism without needing the full
        Cython precomputation.
        """
        offsets_list = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                offsets_list.extend([dr, dc])
        n_offsets_per_pair = (2 * radius + 1) ** 2
        n_pairs = n_dirs * n_dirs

        # All direction pairs use the same set of offsets
        area_offsets = np.array(offsets_list, dtype=np.int32)
        # Tile for all pairs
        area_offsets_full = np.tile(area_offsets, n_pairs)
        area_starts = np.arange(n_pairs, dtype=np.int32) * n_offsets_per_pair
        area_counts = np.full(n_pairs, n_offsets_per_pair, dtype=np.int32)

        return area_offsets_full, area_starts, area_counts

    def test_forbidden_pixel_blocks_tower(self):
        """Tower placement rejected if any footprint pixel is 65535."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)
        # Narrow corridor: only row 1 traversable, rows 0 and 2 blocked
        rows, cols = 3, 100
        raster, params = self._make_params(
            steps, rows=rows, cols=cols, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=150.0,
            span_bin_size=50.0)
        raster[0, :] = 65535
        raster[2, :] = 65535

        # Place a forbidden pixel at column 50 row 1
        # With a radius=1 footprint (3x3), any tower at row 1 near col 50
        # will have its footprint extend to rows 0 and 2 (both 65535).
        # Since all towers must be on row 1 (only traversable), ALL tower
        # placements will have forbidden footprint pixels -> no path.
        area_offsets, area_starts, area_counts = \
            self._make_simple_area_offsets(n_dirs, radius=1)

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=1, source_col=0,
            target_row=1, target_col=99,
            steps=steps,
            area_offsets=area_offsets,
            area_offset_starts=area_starts,
            area_offset_counts=area_counts,
            **params)

        assert len(path) == 0, (
            f"Expected no path when all tower footprints include forbidden "
            f"pixels (rows 0,2 = 65535), got {len(path)} cells")

    def test_area_cost_sums_pixels(self):
        """Area cost sums terrain costs over all footprint pixels.

        With area offsets, an expensive patch in the footprint should
        increase tower cost, while the same patch without area offsets
        (single-pixel) would not affect tower cost if the tower center
        is on cheap terrain.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Run with area offsets (radius=0 = single pixel, same as uniform)
        area_offsets_r0, area_starts_r0, area_counts_r0 = \
            self._make_simple_area_offsets(n_dirs, radius=0)

        path_r0, towers_r0, heights_r0 = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            area_offsets=area_offsets_r0,
            area_offset_starts=area_starts_r0,
            area_offset_counts=area_counts_r0,
            **params)

        assert len(path_r0) > 0, "Should find path with radius=0 area offsets"
        assert path_r0[0] == 5 * 50 + 5
        assert path_r0[-1] == 45 * 50 + 45
        assert len(towers_r0) >= 1

    def test_uniform_mode_no_area_offsets(self):
        """When area_offsets is None, uses single-pixel cost (backward compat).

        This is the same as test_uniform_raster_finds_path but explicitly
        verifies that not passing area offsets still works.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Explicitly no area offsets
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            area_offsets=None,
            area_offset_starts=None,
            area_offset_counts=None,
            **params)

        assert len(path) > 0, "Path should not be empty"
        assert path[0] == 5 * 50 + 5
        assert path[-1] == 45 * 50 + 45
        assert len(towers) >= 1

    def test_slope_multiplier_increases_cost(self):
        """Tower on steep DEM slope has higher cost than flat ground.

        Create DEM with steep section. With gradient_scale > 0 and area
        offsets, the slope multiplier exp(gradient_scale * avg_slope / 100)
        increases tower cost on steep terrain.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Create DEM with gentle slope (not steep enough to block traversal
        # but enough to increase tower cost via slope multiplier)
        dem = np.zeros((50, 50), dtype=np.float32)
        # Linear slope from 0 to 30m over 50 cells = 0.6m/cell = 6% slope
        for c in range(50):
            dem[:, c] = c * 0.6

        tower_heights_arr = np.array([30.0], dtype=np.float32)

        # Run without area offsets (uniform mode)
        path_uniform, _, _ = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            dem=dem,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            gradient_scale=2.0,
            tower_heights=tower_heights_arr,
            **params)

        # Run with area offsets (radius=1, 3x3 footprint)
        area_offsets, area_starts, area_counts = \
            self._make_simple_area_offsets(n_dirs, radius=1)

        path_area, _, _ = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            dem=dem,
            cell_size=10.0,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            gradient_scale=2.0,
            tower_heights=tower_heights_arr,
            area_offsets=area_offsets,
            area_offset_starts=area_starts,
            area_offset_counts=area_counts,
            **params)

        # Both should find a path (slope is gentle enough)
        assert len(path_uniform) > 0, "Uniform mode should find path"
        assert len(path_area) > 0, "Area mode should find path"


class TestIntegration:
    """End-to-end tests wiring GPU v2 through ConstrainedPathFinder."""

    def _make_test_raster(self, shape=(100, 100), value=100, cell_size=1.0):
        """Create a temporary GeoTIFF raster file for testing."""
        import tempfile
        import rasterio
        from rasterio.transform import from_bounds

        tmpfile = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
        width_m = shape[1] * cell_size
        height_m = shape[0] * cell_size
        transform = from_bounds(0, 0, width_m, height_m,
                                shape[1], shape[0])
        data = np.full(shape, value, dtype=np.uint16)

        with rasterio.open(
            tmpfile.name, "w", driver="GTiff",
            height=shape[0], width=shape[1],
            count=1, dtype="uint16",
            crs="EPSG:32632", transform=transform,
        ) as dst:
            dst.write(data, 1)

        return tmpfile.name

    def test_constrained_path_finder_gpu_backend(self):
        """Full end-to-end: ConstrainedPathFinder with raster_gpu backend."""
        import os
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        from pyorps.core.constrained_path import ConstrainedPath

        profile = InfrastructureProfile.from_dict({
            "name": "test_gpu_v2",
            "description": "GPU v2 integration test",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 60.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": 20.0,
            "max_span_m": 40.0,
            "span_bin_size_m": 1.0,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {"max_angle_deg": 10.0,
                                   "base_cost": 500},
                    "angle_tower": {"max_angle_deg": 60.0,
                                    "base_cost": 2000},
                },
            },
        })

        raster_path = self._make_test_raster(shape=(100, 100), value=100)
        try:
            pf = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=profile,
                graph_api="raster_gpu",
                neighborhood_str="r1",
            )
            result = pf.find_route()
            assert isinstance(result, ConstrainedPath)
            assert len(result.path_indices) > 0
            assert result.n_towers > 0
            assert result.total_terrain_cost > 0

            # Tower GeoDataFrame export works
            tower_gdf = result.towers_to_geodataframe()
            assert len(tower_gdf) == result.n_towers

            # Path geometry is valid
            assert result.path_geometry is not None
        finally:
            os.unlink(raster_path)

    def test_gpu_backend_matches_cython_basic(self):
        """GPU v2 and Cython backends produce comparable results."""
        import os
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        profile = InfrastructureProfile.from_dict({
            "name": "test_gpu_vs_cython",
            "description": "GPU vs Cython comparison",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 60.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": 20.0,
            "max_span_m": 40.0,
            "span_bin_size_m": 1.0,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {"max_angle_deg": 10.0,
                                   "base_cost": 500},
                    "angle_tower": {"max_angle_deg": 60.0,
                                    "base_cost": 2000},
                },
            },
        })

        raster_path = self._make_test_raster(shape=(100, 100), value=100)
        try:
            pf_gpu = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=profile,
                graph_api="raster_gpu",
                neighborhood_str="r1",
            )
            result_gpu = pf_gpu.find_route()

            pf_cy = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=profile,
                graph_api="cython",
                neighborhood_str="r1",
            )
            result_cy = pf_cy.find_route()

            # Both should find paths
            assert len(result_gpu.path_indices) > 0
            assert len(result_cy.path_indices) > 0

            # Both should have towers
            assert result_gpu.n_towers > 0
            assert result_cy.n_towers > 0

            # Costs should be in the same ballpark (within 20%)
            cost_ratio = (result_gpu.total_terrain_cost
                          / max(result_cy.total_terrain_cost, 1.0))
            assert 0.8 < cost_ratio < 1.2, (
                f"GPU terrain cost {result_gpu.total_terrain_cost:.0f} vs "
                f"Cython {result_cy.total_terrain_cost:.0f} "
                f"(ratio {cost_ratio:.3f})")
        finally:
            os.unlink(raster_path)

    def test_gpu_fallback_warning(self):
        """GPU backend emits warning and falls back when GPU unavailable."""
        import os
        import warnings
        import unittest.mock as mock
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        from pyorps.core.constrained_path import ConstrainedPath

        profile = InfrastructureProfile.from_dict({
            "name": "test_fallback",
            "description": "Test fallback",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 60.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": 20.0,
            "max_span_m": 40.0,
            "span_bin_size_m": 1.0,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {"max_angle_deg": 10.0,
                                   "base_cost": 500},
                    "angle_tower": {"max_angle_deg": 60.0,
                                    "base_cost": 2000},
                },
            },
        })

        raster_path = self._make_test_raster(shape=(100, 100), value=100)
        try:
            pf = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=profile,
                graph_api="raster_gpu",
                neighborhood_str="r1",
            )

            # Mock the import to simulate GPU unavailability
            original_import = __builtins__.__import__ if hasattr(
                __builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "pyorps.utils.constrained_sssp_gpu_v2":
                    raise ImportError("Mocked: no GPU")
                return original_import(name, *args, **kwargs)

            with mock.patch('builtins.__import__', side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = pf.find_route()
                    # Should have fallen back to Cython
                    assert isinstance(result, ConstrainedPath)
                    assert len(result.path_indices) > 0
                    # Check fallback warning was issued
                    gpu_warnings = [
                        x for x in w
                        if "GPU v2 unavailable" in str(x.message)]
                    assert len(gpu_warnings) > 0
        finally:
            os.unlink(raster_path)


class TestValidation:
    """Phase 1-2 validation: GPU correctness against Cython reference."""

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing.

        Returns (raster, params_dict).
        """
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_validation",
            "description": "test profile for validation tests",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    @pytest.mark.parametrize("size", [30, 50, 100])
    def test_cost_match_vs_cython(self, size):
        """GPU and Cython find paths with similar tower counts on uniform rasters."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_height_2d)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        cell_size = 10.0
        min_span = 50.0
        max_span = 200.0
        span_bin_size = 50.0

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=size, cols=size, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)

        source_row, source_col = 2, 2
        target_row, target_col = size - 3, size - 3

        # DEM and height params (flat DEM, single height)
        dem = np.zeros((size, size), dtype=np.float32)
        tower_heights = np.array([30.0], dtype=np.float32)
        height_premiums = np.array([0.0], dtype=np.float32)

        # --- Run GPU v2 ---
        path_gpu, towers_gpu, heights_gpu = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=steps,
            dem=dem,
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=1,
            **params)

        # --- Run Cython ---
        path_cy, towers_cy, heights_cy = constrained_delta_stepping_height_2d(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=steps,
            angle_cost_lut=params["angle_cost_lut"],
            angle_valid_lut=params["angle_valid_lut"],
            step_distances=params["step_distances"],
            tower_terrain_costs=params["tower_terrain_costs"],
            tower_angle_costs=params["tower_angle_costs"],
            n_span_bins=params["n_span_bins"],
            span_bin_size=params["span_bin_size"],
            min_span=params["min_span"],
            max_span=params["max_span"],
            dem_data=dem,
            cell_size=cell_size,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance_val=0.0,
        )

        # Both should find paths
        assert len(path_gpu) > 0, (
            f"GPU should find a path on {size}x{size} uniform raster")
        assert len(path_cy) > 0, (
            f"Cython should find a path on {size}x{size} uniform raster")

        # Tower counts within +/- 2
        n_towers_gpu = len(towers_gpu)
        n_towers_cy = len(towers_cy)
        assert abs(n_towers_gpu - n_towers_cy) <= 2, (
            f"Tower count mismatch on {size}x{size}: "
            f"GPU={n_towers_gpu}, Cython={n_towers_cy} (diff > 2)")

    def test_height_selection_matches_with_obstacles(self):
        """When obstacles force taller towers, both GPU and Cython select taller."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_height_2d)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        size = 50
        cell_size = 10.0
        min_span = 50.0
        max_span = 200.0
        span_bin_size = 50.0

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=size, cols=size, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)

        # Flat DEM with 25m obstacle band across columns 20-30
        # Obstacle is tall enough that the shortest tower (25m) with
        # min_clearance=10 cannot clear it: 25 - 10 = 15 < 25m obstacle
        dem = np.zeros((size, size), dtype=np.float32)
        obstacle = np.zeros((size, size), dtype=np.float32)
        obstacle[:, 20:30] = 25.0

        # tower_heights=[42, 34, 25] sorted descending, min_clearance=10
        # 25m tower: 25 - 10 = 15 < 25m obstacle -> fails clearance
        # 34m tower: 34 - 10 = 24 < 25m obstacle -> also fails (marginal)
        # 42m tower: 42 - 10 = 32 >= 25m obstacle -> passes
        tower_heights = np.array([42.0, 34.0, 25.0], dtype=np.float32)
        height_premiums = np.array([500.0, 200.0, 0.0], dtype=np.float32)

        # --- Run GPU v2 ---
        path_gpu, towers_gpu, heights_gpu = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            dem=dem,
            obstacle_heights=obstacle,
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=10.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=3,
            **params)

        # --- Run Cython ---
        path_cy, towers_cy, heights_cy = constrained_delta_stepping_height_2d(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            angle_cost_lut=params["angle_cost_lut"],
            angle_valid_lut=params["angle_valid_lut"],
            step_distances=params["step_distances"],
            tower_terrain_costs=params["tower_terrain_costs"],
            tower_angle_costs=params["tower_angle_costs"],
            n_span_bins=params["n_span_bins"],
            span_bin_size=params["span_bin_size"],
            min_span=params["min_span"],
            max_span=params["max_span"],
            dem_data=dem,
            cell_size=cell_size,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance_val=10.0,
            obstacle_heights=obstacle,
        )

        # Both should find paths (the tallest tower can clear the obstacle)
        assert len(path_gpu) > 0, "GPU should find a path with obstacles"
        assert len(path_cy) > 0, "Cython should find a path with obstacles"

        # Both should select at least one tower taller than 25m
        # because 25m tower cannot clear 25m obstacle with 10m clearance
        assert any(h > 25.0 for h in heights_gpu), (
            f"GPU should select at least one tower > 25m, "
            f"got heights: {list(heights_gpu)}")
        assert any(h > 25.0 for h in heights_cy), (
            f"Cython should select at least one tower > 25m, "
            f"got heights: {list(heights_cy)}")

    def test_no_path_agreement(self):
        """Both GPU and Cython agree when no path exists."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_height_2d)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        size = 30
        cell_size = 10.0
        min_span = 50.0
        max_span = 200.0
        span_bin_size = 50.0

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=size, cols=size, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)

        # Block everything — fully impassable raster
        raster[:, :] = 65535
        # Except source and target cells (so they don't raise ValueError)
        raster[2, 2] = 10
        raster[27, 27] = 10

        dem = np.zeros((size, size), dtype=np.float32)
        tower_heights = np.array([30.0], dtype=np.float32)
        height_premiums = np.array([0.0], dtype=np.float32)

        # --- Run GPU v2 ---
        path_gpu, towers_gpu, heights_gpu = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=2, source_col=2,
            target_row=27, target_col=27,
            steps=steps,
            dem=dem,
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=1,
            **params)

        # --- Run Cython ---
        path_cy, towers_cy, heights_cy = constrained_delta_stepping_height_2d(
            raster=raster,
            source_row=2, source_col=2,
            target_row=27, target_col=27,
            steps=steps,
            angle_cost_lut=params["angle_cost_lut"],
            angle_valid_lut=params["angle_valid_lut"],
            step_distances=params["step_distances"],
            tower_terrain_costs=params["tower_terrain_costs"],
            tower_angle_costs=params["tower_angle_costs"],
            n_span_bins=params["n_span_bins"],
            span_bin_size=params["span_bin_size"],
            min_span=params["min_span"],
            max_span=params["max_span"],
            dem_data=dem,
            cell_size=cell_size,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance_val=0.0,
        )

        # Both should return empty paths
        assert len(path_gpu) == 0, (
            f"GPU should find no path on blocked raster, "
            f"got {len(path_gpu)} cells")
        assert len(path_cy) == 0, (
            f"Cython should find no path on blocked raster, "
            f"got {len(path_cy)} cells")

    def test_forbidden_footprint_routes_around(self):
        """GPU routes around cells where tower footprint would be forbidden.

        Creates a wide raster where a band of forbidden cells exists adjacent
        to the direct path. Towers placed near the band would have forbidden
        footprints. The path should route around via cells with valid footprints.
        Both GPU and Cython should find paths that avoid the forbidden band.
        """
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_height_2d)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        cell_size = 10.0
        min_span = 50.0
        max_span = 200.0
        span_bin_size = 50.0

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)

        # 30x50 raster — wide enough for routing around
        rows, cols = 30, 50
        raster, params = self._make_params(
            steps, rows=rows, cols=cols, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)

        # Place forbidden band at rows 13-17 across middle columns
        # Tower footprints near this band overlap forbidden cells
        raster[13:17, 15:35] = 65535

        dem = np.zeros((rows, cols), dtype=np.float32)
        tower_heights = np.array([30.0], dtype=np.float32)
        height_premiums = np.array([0.0], dtype=np.float32)

        # Build 3x3 area offsets (radius=1)
        offsets_list = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                offsets_list.extend([dr, dc])
        n_offsets_per_pair = 9
        n_pairs = n_dirs * n_dirs
        area_offsets = np.tile(
            np.array(offsets_list, dtype=np.int32), n_pairs)
        area_starts = (np.arange(n_pairs, dtype=np.int32)
                       * n_offsets_per_pair)
        area_counts = np.full(n_pairs, n_offsets_per_pair,
                              dtype=np.int32)

        # Source at top-left, target at bottom-right — must route around
        # the forbidden band
        path_gpu, towers_gpu, heights_gpu = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=25, target_col=45,
            steps=steps,
            dem=dem,
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=1,
            area_offsets=area_offsets,
            area_offset_starts=area_starts,
            area_offset_counts=area_counts,
            **params)

        path_cy, towers_cy, heights_cy = constrained_delta_stepping_height_2d(
            raster=raster,
            source_row=5, source_col=5,
            target_row=25, target_col=45,
            steps=steps,
            angle_cost_lut=params["angle_cost_lut"],
            angle_valid_lut=params["angle_valid_lut"],
            step_distances=params["step_distances"],
            tower_terrain_costs=params["tower_terrain_costs"],
            tower_angle_costs=params["tower_angle_costs"],
            n_span_bins=params["n_span_bins"],
            span_bin_size=params["span_bin_size"],
            min_span=params["min_span"],
            max_span=params["max_span"],
            dem_data=dem,
            cell_size=cell_size,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance_val=0.0,
            area_offsets=area_offsets,
            area_offset_starts=area_starts,
            area_offset_counts=area_counts,
        )

        # Both should find paths that route around the forbidden band
        assert len(path_gpu) > 0, (
            "GPU should find path routing around forbidden footprint band")
        assert len(path_cy) > 0, (
            "Cython should find path routing around forbidden footprint band")

        # Verify no tower is placed inside the forbidden band
        for t in towers_gpu:
            t_row, t_col = t // cols, t % cols
            assert not (13 <= t_row <= 16 and 15 <= t_col <= 34), (
                f"GPU tower at ({t_row},{t_col}) is inside forbidden band")
        for t in towers_cy:
            t_row, t_col = t // cols, t % cols
            assert not (13 <= t_row <= 16 and 15 <= t_col <= 34), (
                f"Cython tower at ({t_row},{t_col}) is inside forbidden band")


class TestBenchmarks:
    """Phase 3: performance benchmarks (GPU v2 vs Cython)."""

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing.

        Returns (raster, params_dict).
        """
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_benchmark",
            "description": "test profile for benchmark tests",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    @pytest.mark.parametrize("size", [200, 500])
    def test_speedup_vs_cython(self, size):
        """GPU should be faster than Cython for large rasters."""
        import time
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_height_2d)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        cell_size = 10.0
        min_span = 100.0
        max_span = 400.0
        span_bin_size = 100.0

        steps = get_neighborhood_steps(1, directed=True)

        # Create random raster with some variation (not uniform)
        np.random.seed(42)
        raster = np.random.randint(5, 500, size=(size, size),
                                   dtype=np.uint16)

        n_span_bins = int(max_span / span_bin_size) + 1
        _, params = self._make_params(
            steps, rows=size, cols=size, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)

        source_row, source_col = 5, 5
        target_row, target_col = size - 6, size - 6

        dem = np.zeros((size, size), dtype=np.float32)
        tower_heights = np.array([30.0], dtype=np.float32)
        height_premiums = np.array([0.0], dtype=np.float32)

        common_gpu = dict(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=steps,
            dem=dem,
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=1,
            **params,
        )

        common_cy = dict(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=steps,
            angle_cost_lut=params["angle_cost_lut"],
            angle_valid_lut=params["angle_valid_lut"],
            step_distances=params["step_distances"],
            tower_terrain_costs=params["tower_terrain_costs"],
            tower_angle_costs=params["tower_angle_costs"],
            n_span_bins=params["n_span_bins"],
            span_bin_size=params["span_bin_size"],
            min_span=params["min_span"],
            max_span=params["max_span"],
            dem_data=dem,
            cell_size=cell_size,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance_val=0.0,
        )

        # --- Cython timing ---
        t0 = time.perf_counter()
        path_cy, towers_cy, heights_cy = \
            constrained_delta_stepping_height_2d(**common_cy)
        t_cy = time.perf_counter() - t0

        # --- GPU warm-up (kernel compilation) ---
        _ = constrained_sssp_raster_gpu_v2(**common_gpu)

        # --- GPU measured run ---
        t0 = time.perf_counter()
        path_gpu, towers_gpu, heights_gpu = \
            constrained_sssp_raster_gpu_v2(**common_gpu)
        t_gpu = time.perf_counter() - t0

        speedup = t_cy / max(t_gpu, 1e-6)
        print(f"\n  [{size}x{size}] Cython: {t_cy:.3f}s | "
              f"GPU v2: {t_gpu:.3f}s | Speedup: {speedup:.1f}x")

        # Both should find a path
        assert len(path_gpu) > 0, f"GPU should find path on {size}x{size}"
        assert len(path_cy) > 0, f"Cython should find path on {size}x{size}"

        # GPU should be faster for size >= 200
        assert t_gpu < t_cy, (
            f"GPU ({t_gpu:.3f}s) should be faster than Cython ({t_cy:.3f}s) "
            f"for {size}x{size} raster")

    def test_large_raster_completes(self):
        """1000x1000 raster completes without OOM or hang."""
        import time
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        size = 1000
        cell_size = 10.0
        min_span = 200.0
        max_span = 600.0
        span_bin_size = 200.0

        # Use R1 neighborhood (8 dirs) to keep state space manageable
        steps = get_neighborhood_steps(1, directed=True)

        np.random.seed(123)
        raster = np.random.randint(5, 500, size=(size, size),
                                   dtype=np.uint16)

        _, params = self._make_params(
            steps, rows=size, cols=size, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)

        dem = np.zeros((size, size), dtype=np.float32)
        tower_heights = np.array([30.0], dtype=np.float32)
        height_premiums = np.array([0.0], dtype=np.float32)

        # Warm-up on small raster to compile kernel
        raster_small = np.full((20, 20), 10, dtype=np.uint16)
        _, params_small = self._make_params(
            steps, rows=20, cols=20, raster_val=10,
            cell_size=cell_size, min_span=min_span, max_span=max_span,
            span_bin_size=span_bin_size)
        _ = constrained_sssp_raster_gpu_v2(
            raster=raster_small,
            source_row=2, source_col=2,
            target_row=17, target_col=17,
            steps=steps,
            dem=np.zeros((20, 20), dtype=np.float32),
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=1,
            **params_small)

        # Timed run on 1000x1000
        t0 = time.perf_counter()
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=10, source_col=10,
            target_row=size - 11, target_col=size - 11,
            steps=steps,
            dem=dem,
            cell_size=cell_size,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            tower_heights=tower_heights,
            height_premiums=height_premiums,
            n_heights=1,
            **params)
        elapsed = time.perf_counter() - t0

        print(f"\n  [1000x1000] GPU v2: {elapsed:.3f}s | "
              f"path={len(path)} cells, towers={len(towers)}")

        # Just assert it completes and returns valid results
        assert len(path) > 0, "Should find path on 1000x1000 raster"
        assert len(towers) >= 1, "Should place at least 1 tower"
        assert len(heights) == len(towers)


class TestSparseKernel:
    """Test sparse hash table mode for large rasters that exceed dense VRAM."""

    @staticmethod
    def _make_params(steps, rows=50, cols=50, raster_val=10,
                     cell_size=10.0, min_span=50.0, max_span=300.0,
                     span_bin_size=50.0):
        """Create uniform raster + profile params for testing."""
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1

        config = {
            "name": "test_sparse",
            "description": "test profile for sparse kernel tests",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {
                        "max_angle_deg": 90.0,
                        "base_cost": 1000,
                    },
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        step_distances = profile.compute_step_distances(steps, cell_size)
        tower_terrain_costs = profile.precompute_tower_terrain_costs()
        tower_angle_costs = profile.precompute_tower_angle_costs(steps)

        params = {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": step_distances.astype(np.float32),
            "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
            "tower_angle_costs": tower_angle_costs.astype(np.float32),
            "n_span_bins": n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }
        return raster, params

    def test_sparse_finds_same_path_as_dense(self):
        """Sparse and dense modes produce identical results on small raster."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Run dense (default)
        path_dense, towers_dense, heights_dense = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            sparse=False,
            **params)

        # Run sparse
        path_sparse, towers_sparse, heights_sparse = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            sparse=True,
            **params)

        # Both should find paths
        assert len(path_dense) > 0, "Dense mode should find a path"
        assert len(path_sparse) > 0, "Sparse mode should find a path"

        # Same start and end
        assert path_dense[0] == path_sparse[0], (
            f"Path start mismatch: dense={path_dense[0]}, sparse={path_sparse[0]}")
        assert path_dense[-1] == path_sparse[-1], (
            f"Path end mismatch: dense={path_dense[-1]}, sparse={path_sparse[-1]}")

        # Tower counts should match (same problem, same algorithm)
        assert abs(len(towers_dense) - len(towers_sparse)) <= 1, (
            f"Tower count mismatch: dense={len(towers_dense)}, "
            f"sparse={len(towers_sparse)}")

        # Path lengths should be similar (within 10%)
        len_ratio = len(path_sparse) / max(len(path_dense), 1)
        assert 0.9 < len_ratio < 1.1, (
            f"Path length mismatch: dense={len(path_dense)}, "
            f"sparse={len(path_sparse)}")

    def test_sparse_memory_much_smaller(self):
        """Sparse mode uses much less memory than dense."""
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            compute_memory_budget_gb)

        rows, cols = 1433, 1834  # Real-world R4 dimensions
        n_dirs, n_span_bins, n_heights = 48, 6, 3

        dense_gb = compute_memory_budget_gb(
            rows, cols, n_dirs, n_span_bins, n_heights, sparse=False)
        sparse_gb = compute_memory_budget_gb(
            rows, cols, n_dirs, n_span_bins, n_heights, sparse=True)

        assert sparse_gb < dense_gb, (
            f"Sparse ({sparse_gb:.1f} GB) should use less memory than "
            f"dense ({dense_gb:.1f} GB)")
        # Sparse should use at most 1/3 of dense memory for large state spaces
        assert sparse_gb < dense_gb / 3, (
            f"Sparse ({sparse_gb:.1f} GB) should use much less than "
            f"dense ({dense_gb:.1f} GB)")

    def test_sparse_handles_no_path(self):
        """Sparse mode terminates cleanly when no path exists."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2, _check_v2_available)

        if not _check_v2_available():
            pytest.skip("V2 persistent kernel not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = self._make_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Block entire middle band
        raster[20:30, :] = 65535

        path, towers, heights = constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            sparse=True,
            **params)

        assert len(path) == 0, f"Expected empty path, got {len(path)} cells"
        assert len(towers) == 0
        assert len(heights) == 0

    def test_sparse_fits_large_raster(self):
        """Sparse mode fits R4 on a raster that dense cannot."""
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            compute_memory_budget_gb, check_memory_fits)

        # R4 on 1433x1834 raster: dense needs ~13 GB, sparse ~0.5 GB
        rows, cols = 1433, 1834
        n_dirs, n_span_bins, n_heights = 48, 6, 3

        dense_gb = compute_memory_budget_gb(
            rows, cols, n_dirs, n_span_bins, n_heights, sparse=False)
        sparse_gb = compute_memory_budget_gb(
            rows, cols, n_dirs, n_span_bins, n_heights, sparse=True)

        # Dense should NOT fit in 6 GB
        assert dense_gb > 6.0, (
            f"Dense ({dense_gb:.1f} GB) should exceed 6 GB VRAM")

        # Sparse should fit in 6 GB
        assert sparse_gb < 6.0, (
            f"Sparse ({sparse_gb:.1f} GB) should fit in 6 GB VRAM")

        # Dense should raise MemoryError at 6 GB
        with pytest.raises(MemoryError):
            check_memory_fits(
                rows, cols, n_dirs, n_span_bins, n_heights,
                vram_gb=6.0, sparse=False)

        # Sparse should NOT raise MemoryError at 6 GB
        check_memory_fits(
            rows, cols, n_dirs, n_span_bins, n_heights,
            vram_gb=6.0, sparse=True)  # should not raise

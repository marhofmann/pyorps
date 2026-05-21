import unittest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy not available")
class TestConstrainedGPU(unittest.TestCase):
    """Tests for GPU constrained delta-stepping kernel."""

    def _make_profile_params(self, steps, cell_size=10.0,
                              min_span=50.0, max_span=100.0,
                              span_bin_size=10.0,
                              hard_angle=90.0, angle_scale=100):
        """Create profile parameter dict for GPU tests."""
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        config = {
            "name": "test", "description": "test",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": hard_angle,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": angle_scale},
            "min_span_m": min_span,
            "max_span_m": max_span,
            "span_bin_size_m": span_bin_size,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {"max_angle_deg": 90.0, "base_cost": 1000},
                },
            },
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        return {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": profile.compute_step_distances(
                steps, cell_size).astype(np.float32),
            "tower_terrain_costs": profile.precompute_tower_terrain_costs(
                ).astype(np.float32),
            "tower_angle_costs": profile.precompute_tower_angle_costs(
                steps).astype(np.float32),
            "n_span_bins": profile.n_span_bins,
            "span_bin_size": span_bin_size,
            "min_span": min_span,
            "max_span": max_span,
        }

    def _make_direction_only_params(self, steps, cell_size=10.0,
                                     hard_angle=90.0, angle_scale=100):
        """Create params without span constraints (direction-only mode)."""
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        config = {
            "name": "test", "description": "test",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": hard_angle,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": angle_scale},
        }
        profile = InfrastructureProfile.from_dict(config)
        angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
        n_dirs = len(steps)
        return {
            "angle_cost_lut": angle_cost_lut.astype(np.float32),
            "angle_valid_lut": angle_valid_lut.astype(np.uint8),
            "step_distances": profile.compute_step_distances(
                steps, cell_size).astype(np.float32),
            "tower_terrain_costs": np.zeros(65536, dtype=np.float32),
            "tower_angle_costs": np.zeros(
                (n_dirs, n_dirs), dtype=np.float32),
            "n_span_bins": 1,
            "span_bin_size": 1e6,
            "min_span": 0.0,
            "max_span": 1e6,
        }

    @staticmethod
    def _path_cost(path_arr, raster, ncols):
        """Compute terrain-only cost along a path."""
        total = 0.0
        for i in range(len(path_arr) - 1):
            r1, c1 = int(path_arr[i]) // ncols, int(path_arr[i]) % ncols
            r2, c2 = int(path_arr[i+1]) // ncols, int(path_arr[i+1]) % ncols
            dist = ((r2 - r1)**2 + (c2 - c1)**2) ** 0.5
            avg = (float(raster[r1, c1]) + float(raster[r2, c2])) / 2.0
            total += avg * dist
        return total

    def test_gpu_matches_cython_uniform_raster(self):
        """GPU constrained SSSP produces same path as Cython on uniform."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_path_algorithms import (
            constrained_dijkstra_2d)
        from pyorps.utils.constrained_sssp_gpu import (
            constrained_sssp_raster_gpu)

        steps = get_neighborhood_steps(1, directed=True)
        raster = np.full((50, 50), 100, dtype=np.uint16)
        params = self._make_profile_params(steps, cell_size=10.0,
                                           min_span=50.0, max_span=100.0,
                                           span_bin_size=10.0)

        cy_path, cy_towers = constrained_dijkstra_2d(
            raster=raster, source_row=0, source_col=0,
            target_row=49, target_col=49, steps=steps, **params)
        gpu_path, gpu_towers = constrained_sssp_raster_gpu(
            raster=raster, source_row=0, source_col=0,
            target_row=49, target_col=49, steps=steps, **params)

        cy_cost = self._path_cost(cy_path, raster, 50)
        gpu_cost = self._path_cost(gpu_path, raster, 50)
        self.assertAlmostEqual(cy_cost, gpu_cost, delta=cy_cost * 0.01,
            msg="GPU path cost should be within 1% of Cython")
        self.assertEqual(len(cy_towers), len(gpu_towers))

    def test_gpu_forced_tower_placement(self):
        """Path longer than max_span forces tower placement."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu import (
            constrained_sssp_raster_gpu)

        steps = get_neighborhood_steps(1, directed=True)
        raster = np.full((50, 50), 100, dtype=np.uint16)
        params = self._make_profile_params(steps, cell_size=10.0,
                                           min_span=50.0, max_span=100.0,
                                           span_bin_size=10.0)

        gpu_path, gpu_towers = constrained_sssp_raster_gpu(
            raster=raster, source_row=0, source_col=0,
            target_row=0, target_col=49, steps=steps, **params)
        # Path is 490m (49 cells * 10m), max_span=100m -> at least 4 towers
        self.assertGreaterEqual(len(gpu_towers), 4)

    def test_gpu_matches_cython_nonuniform(self):
        """GPU and Cython produce same cost on random raster."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_path_algorithms import (
            constrained_dijkstra_2d)
        from pyorps.utils.constrained_sssp_gpu import (
            constrained_sssp_raster_gpu)

        steps = get_neighborhood_steps(1, directed=True)
        np.random.seed(42)
        raster = np.random.randint(10, 500, size=(30, 30), dtype=np.uint16)
        params = self._make_profile_params(steps, cell_size=10.0,
                                           min_span=30.0, max_span=80.0,
                                           span_bin_size=10.0)

        cy_path, _ = constrained_dijkstra_2d(
            raster=raster, source_row=0, source_col=0,
            target_row=29, target_col=29, steps=steps, **params)
        gpu_path, _ = constrained_sssp_raster_gpu(
            raster=raster, source_row=0, source_col=0,
            target_row=29, target_col=29, steps=steps, **params)

        self.assertGreater(len(cy_path), 0)
        self.assertGreater(len(gpu_path), 0)
        cy_cost = self._path_cost(cy_path, raster, 30)
        gpu_cost = self._path_cost(gpu_path, raster, 30)
        self.assertAlmostEqual(cy_cost, gpu_cost, delta=cy_cost * 0.01,
            msg="GPU path cost should be within 1% of Cython on nonuniform")

    def test_gpu_angle_rejection(self):
        """Angle penalties make path avoid sharp turns near obstacle."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu import (
            constrained_sssp_raster_gpu)

        # R2 for finer angular resolution
        steps = get_neighborhood_steps(2, directed=True)
        raster = np.full((30, 30), 100, dtype=np.uint16)
        # Partial wall forcing a detour
        raster[0:12, 15] = 65535

        # 45° hard limit with high angle penalty — path must go around
        params = self._make_direction_only_params(
            steps, cell_size=10.0, hard_angle=45.0, angle_scale=500)

        gpu_path, _ = constrained_sssp_raster_gpu(
            raster=raster, source_row=0, source_col=0,
            target_row=0, target_col=29, steps=steps, **params)
        self.assertGreater(len(gpu_path), 0)
        # Path must detour around the wall — check that it doesn't go
        # through the blocked column at col=15 in rows 0-11
        ncols = 30
        for idx in gpu_path:
            r, c = int(idx) // ncols, int(idx) % ncols
            self.assertFalse(
                r < 12 and c == 15,
                f"Path went through wall at ({r},{c})")


if __name__ == "__main__":
    unittest.main()

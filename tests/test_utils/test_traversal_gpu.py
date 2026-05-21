"""
Tests for GPU-accelerated edge construction.

All tests are guarded with skipUnless so they are skipped gracefully
on machines without an NVIDIA GPU or CuPy installed.
"""

import unittest
import numpy as np

from pyorps.utils.traversal_gpu import (
    intermediate_steps_cpu,
    prepare_step_lookup_tables,
    GPU_AVAILABLE,
)

# Import CPU reference implementations for comparison
from pyorps.utils.traversal import (
    construct_edges,
    intermediate_steps_numba,
    get_cost_factor_numba,
)
from pyorps.utils.neighborhood import get_neighborhood_steps


class TestIntermediateStepsCPU(unittest.TestCase):
    """Test the pure-Python intermediate_steps_cpu against Numba reference."""

    def test_simple_steps(self):
        """Adjacent steps should have no intermediates."""
        self.assertEqual(intermediate_steps_cpu(1, 0), [])
        self.assertEqual(intermediate_steps_cpu(0, 1), [])
        self.assertEqual(intermediate_steps_cpu(-1, 0), [])
        self.assertEqual(intermediate_steps_cpu(0, -1), [])

    def test_diagonal_step(self):
        """Diagonal step (1,1) should have two intermediates."""
        result = intermediate_steps_cpu(1, 1)
        self.assertEqual(len(result), 2)
        self.assertIn((1, 0), result)
        self.assertIn((0, 1), result)

    def test_knight_move(self):
        """Knight move (2,1) should produce intermediates."""
        result = intermediate_steps_cpu(2, 1)
        self.assertEqual(len(result), 2)

    def test_matches_numba(self):
        """CPU version must match Numba version for all common steps."""
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                cpu_result = intermediate_steps_cpu(dr, dc)
                numba_result = intermediate_steps_numba(
                    np.int8(dr), np.int8(dc)
                )
                self.assertEqual(
                    len(cpu_result), numba_result.shape[0],
                    f"Mismatch for step ({dr}, {dc}): "
                    f"CPU={len(cpu_result)}, Numba={numba_result.shape[0]}"
                )
                for i, (idr, idc) in enumerate(cpu_result):
                    self.assertEqual(idr, numba_result[i, 0])
                    self.assertEqual(idc, numba_result[i, 1])


class TestPrepareStepLookupTables(unittest.TestCase):
    """Test pre-computation of step lookup tables."""

    def test_r0_neighborhood(self):
        """R0 (4-connectivity) should produce correct tables."""
        steps = get_neighborhood_steps("r0", directed=False)
        steps_arr, inter_lut, n_inter, cost_factors = \
            prepare_step_lookup_tables(steps)

        self.assertEqual(steps_arr.shape[0], len(steps))
        self.assertEqual(steps_arr.shape[1], 2)
        self.assertEqual(n_inter.shape[0], len(steps))
        self.assertEqual(cost_factors.shape[0], len(steps))

        # R0 has no intermediates for any step
        for i in range(len(steps)):
            self.assertEqual(n_inter[i], 0)

    def test_r1_neighborhood(self):
        """R1 (8-connectivity) should produce correct tables."""
        steps = get_neighborhood_steps("r1", directed=False)
        steps_arr, inter_lut, n_inter, cost_factors = \
            prepare_step_lookup_tables(steps)

        self.assertEqual(steps_arr.shape[0], len(steps))

        # Diagonal steps in R1 should have 2 intermediates each
        for i in range(len(steps)):
            dr, dc = int(steps[i, 0]), int(steps[i, 1])
            if abs(dr) == 1 and abs(dc) == 1:
                self.assertEqual(n_inter[i], 2)
            else:
                self.assertEqual(n_inter[i], 0)

    def test_cost_factors_match_numba(self):
        """Cost factors must match the Numba reference (float32 precision)."""
        steps = get_neighborhood_steps("r2", directed=False)
        _, _, n_inter, cost_factors = prepare_step_lookup_tables(steps)

        for i in range(len(steps)):
            dr, dc = np.int8(steps[i, 0]), np.int8(steps[i, 1])
            expected = get_cost_factor_numba(dr, dc, int(n_inter[i]))
            np.testing.assert_allclose(
                cost_factors[i], expected, rtol=5e-4,
                err_msg=f"Cost factor mismatch for step ({dr}, {dc})"
            )


@unittest.skipUnless(GPU_AVAILABLE, "CUDA GPU or CuPy not available")
class TestConstructEdgesGPU2D(unittest.TestCase):
    """Test GPU edge construction against CPU reference (2D)."""

    def _compare_cpu_gpu(self, raster, steps, ignore_max=True, rtol=5e-4,
                         atol=0.1):
        """Helper: run CPU and GPU, sort edges, compare."""
        from pyorps.utils.traversal_gpu import construct_edges_gpu

        # CPU reference
        cpu_from, cpu_to, cpu_cost = construct_edges(raster, steps, ignore_max)

        # GPU
        gpu_from, gpu_to, gpu_cost = construct_edges_gpu(
            raster, steps, ignore_max
        )

        self.assertEqual(len(cpu_from), len(gpu_from),
                         f"Edge count mismatch: CPU={len(cpu_from)}, "
                         f"GPU={len(gpu_from)}")

        if len(cpu_from) == 0:
            return

        # Sort both by (from, to) for comparison
        cpu_order = np.lexsort((cpu_to, cpu_from))
        gpu_order = np.lexsort((gpu_to, gpu_from))

        np.testing.assert_array_equal(
            cpu_from[cpu_order], gpu_from[gpu_order],
            err_msg="from_nodes mismatch"
        )
        np.testing.assert_array_equal(
            cpu_to[cpu_order], gpu_to[gpu_order],
            err_msg="to_nodes mismatch"
        )
        np.testing.assert_allclose(
            cpu_cost[cpu_order], gpu_cost[gpu_order],
            rtol=rtol, atol=atol,
            err_msg="Edge costs mismatch"
        )

    def test_uniform_5x5_r0(self):
        """Uniform 5x5 raster with R0 neighborhood."""
        raster = np.full((5, 5), 100, dtype=np.uint16)
        steps = get_neighborhood_steps("r0", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_uniform_5x5_r1(self):
        """Uniform 5x5 raster with R1 neighborhood."""
        raster = np.full((5, 5), 100, dtype=np.uint16)
        steps = get_neighborhood_steps("r1", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_with_obstacles(self):
        """5x5 raster with forbidden cells (65535)."""
        raster = np.full((5, 5), 100, dtype=np.uint16)
        raster[2, 2] = 65535  # forbidden center
        raster[0, 4] = 65535  # forbidden corner
        steps = get_neighborhood_steps("r1", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_all_forbidden(self):
        """All cells forbidden → 0 edges."""
        raster = np.full((5, 5), 65535, dtype=np.uint16)
        steps = get_neighborhood_steps("r0", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_single_valid_cell(self):
        """Only one valid cell → 0 edges (no neighbors)."""
        raster = np.full((5, 5), 65535, dtype=np.uint16)
        raster[2, 2] = 100
        steps = get_neighborhood_steps("r0", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_ignore_max_false(self):
        """With ignore_max=False, all cells are valid including 65535.

        Note: CPU construct_edges has a uint16 overflow bug when adding
        raster values near 65535 (e.g. 65535 + 200 overflows uint16).
        GPU correctly uses double precision, so we avoid overflow-triggering
        values here and test with moderate values instead.
        """
        raster = np.array([
            [100, 200, 300],
            [400, 500, 600],
            [700, 800, 900],
        ], dtype=np.uint16)
        steps = get_neighborhood_steps("r0", directed=False)
        self._compare_cpu_gpu(raster, steps, ignore_max=False)

    def test_varying_costs(self):
        """Raster with varying costs."""
        raster = np.array([
            [10, 20, 30, 40, 50],
            [60, 70, 80, 90, 100],
            [110, 120, 130, 140, 150],
            [160, 170, 180, 190, 200],
            [210, 220, 230, 240, 250],
        ], dtype=np.uint16)
        steps = get_neighborhood_steps("r1", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_larger_raster_r2(self):
        """20x20 raster with R2 neighborhood."""
        np.random.seed(42)
        raster = np.random.randint(1, 1000, size=(20, 20), dtype=np.uint16)
        # Add some forbidden cells
        raster[5, 5] = 65535
        raster[10, 10] = 65535
        steps = get_neighborhood_steps("r2", directed=False)
        self._compare_cpu_gpu(raster, steps)

    def test_return_cupy(self):
        """Test that return_cupy=True returns CuPy arrays."""
        try:
            import cupy as cp
        except ImportError:
            self.skipTest("CuPy not available")

        from pyorps.utils.traversal_gpu import construct_edges_gpu

        raster = np.full((5, 5), 100, dtype=np.uint16)
        steps = get_neighborhood_steps("r0", directed=False)

        gpu_from, gpu_to, gpu_cost = construct_edges_gpu(
            raster, steps, return_cupy=True
        )

        self.assertIsInstance(gpu_from, cp.ndarray)
        self.assertIsInstance(gpu_to, cp.ndarray)
        self.assertIsInstance(gpu_cost, cp.ndarray)


if __name__ == "__main__":
    unittest.main()

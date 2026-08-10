"""Tests for the V5 asynchronous bucket-queue GPU SSSP kernel.

V5 (open-levers plan section 2) removes grid-wide barriers from the hot
path; exactness rests on atomicMin + re-queue (label-correcting) plus the
span-boundary rescan. These tests pin bit-exactness against the V4
persistent kernel across raster families, connectivities, gradient mode,
float32 rasters, targeted early exit, and predecessor reconstruction.
"""
import numpy as np
import pytest

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU = True
    except Exception:
        GPU = False
except ImportError:
    GPU = False

pytestmark = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

if GPU:
    from pyorps.utils.sssp_gpu import sssp_raster_gpu_v4, sssp_raster_gpu_v5

STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)
STEPS_4 = STEPS_8[:4]


def _assert_bit_exact(d5, ref):
    f_ref = np.isfinite(ref) & (ref < 1e29)
    f_v5 = np.isfinite(d5) & (d5 < 1e29)
    assert np.array_equal(f_ref, f_v5), "reachability mask differs"
    if f_ref.any():
        assert float(np.max(np.abs(d5[f_ref] - ref[f_ref]))) == 0.0


class TestV5Exactness:
    @pytest.mark.parametrize("n,steps", [
        (50, STEPS_4), (50, STEPS_8), (200, STEPS_8), (500, STEPS_8),
    ])
    def test_random_matches_v4(self, n, steps):
        rng = np.random.default_rng(n)
        raster = rng.integers(1, 200, (n, n)).astype(np.uint16)
        ref = sssp_raster_gpu_v4(raster, steps, 0, delta="auto")
        d5 = sssp_raster_gpu_v5(raster, steps, 0, delta="auto")
        _assert_bit_exact(d5, ref)

    def test_uniform_raster(self):
        raster = np.ones((100, 100), dtype=np.uint16)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto")
        _assert_bit_exact(d5, ref)

    def test_heavy_tail(self):
        rng = np.random.default_rng(3)
        raster = np.clip(rng.lognormal(2.0, 1.5, (300, 300)), 1,
                         5000).astype(np.uint16)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto")
        _assert_bit_exact(d5, ref)

    def test_obstacle_wall(self):
        rng = np.random.default_rng(4)
        raster = rng.integers(1, 200, (200, 200)).astype(np.uint16)
        raster[50:150, 100] = 65535
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto",
                                 ignore_max=False)
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                ignore_max=False)
        _assert_bit_exact(d5, ref)

    def test_unreachable_region(self):
        raster = np.ones((60, 60), dtype=np.uint16)
        raster[:, 30] = 65535  # full wall
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto",
                                 ignore_max=False)
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                ignore_max=False)
        _assert_bit_exact(d5, ref)

    @pytest.mark.parametrize("chunk", [32, 128, 1024])
    def test_chunk_sizes(self, chunk):
        rng = np.random.default_rng(5)
        raster = rng.integers(1, 200, (150, 150)).astype(np.uint16)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                chunk=chunk)
        _assert_bit_exact(d5, ref)

    def test_fixed_delta(self):
        rng = np.random.default_rng(6)
        raster = rng.integers(1, 200, (150, 150)).astype(np.uint16)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta=500.0)
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta=500.0)
        _assert_bit_exact(d5, ref)

    def test_float32_raster(self):
        rng = np.random.default_rng(8)
        raster = rng.uniform(0.5, 200.0, (150, 150)).astype(np.float32)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto")
        _assert_bit_exact(d5, ref)


class TestV5Gradient:
    def test_gradient_mode_matches_v4(self):
        from pyorps.core.objective import Objective
        rng = np.random.default_rng(12)
        n = 150
        raster = rng.integers(1, 200, (n, n)).astype(np.uint16)
        x, y = np.meshgrid(np.arange(n, dtype=np.float32),
                           np.arange(n, dtype=np.float32))
        dem = (20.0 * np.sin(x / 15.0) * np.cos(y / 20.0)).astype(np.float32)
        obj = Objective({"cost": 1.0},
                        gradient_options={"multiplier": "exponential",
                                          "max_gradient_pct": 60.0})
        luts = obj.build_gradient_luts(STEPS_8, cell_size=10.0)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto",
                                 dem=dem, gradient_luts=luts)
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                dem=dem, gradient_luts=luts)
        _assert_bit_exact(d5, ref)


class TestV5Targets:
    def test_target_dists_match_full_run(self):
        rng = np.random.default_rng(9)
        raster = rng.integers(1, 200, (300, 300)).astype(np.uint16)
        targets = np.array([300 * 300 - 1, 150 * 300 + 150], dtype=np.int64)
        full = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto")
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                target_indices=targets)
        for t in targets:
            assert d5[t] == full[t]

    def test_early_exit_matches_v4_targets(self):
        rng = np.random.default_rng(10)
        raster = rng.integers(1, 200, (300, 300)).astype(np.uint16)
        targets = np.array([299, 300 * 299], dtype=np.int64)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto",
                                 target_indices=targets)
        d5 = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                target_indices=targets)
        for t in targets:
            assert d5[t] == ref[t]


class TestV5TargetedStress:
    """Regression: two nondeterministic targeted-mode bugs (2026-08-06).

    1. The rendezvous target check reused C5_MIN_DIST as scratch and
       zeroed it right after the barrier that published gm through the
       same word -- slower blocks read gm == 0 and the blocks' barrier
       sequences diverged (~11 % corrupted runs at 50x50).
    2. The hot-path pred store races concurrent better atomicMin winners;
       the v5_repair_pred post-pass must leave every chain consistent.
    Single runs pass by luck; repetition is the test.
    """

    def test_repeated_targeted_runs_deterministic(self):
        rng = np.random.RandomState(42)
        size = 50
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.1] = 65535
        raster[0, 0] = 5
        raster[size - 1, size - 1] = 5
        t = size * size - 1
        tarr = np.array([t], dtype=np.int32)
        ref = sssp_raster_gpu_v5(raster, STEPS_4, 0, delta="auto")[t]
        for i in range(30):
            d, pred = sssp_raster_gpu_v5(
                raster, STEPS_4, 0, delta="auto", target_indices=tarr,
                return_predecessor=True)
            assert d[t] == ref, f"run {i}: dist {d[t]} != {ref}"
            v, hops = t, 0
            while v != 0:
                u = int(pred[v])
                assert u >= 0, f"run {i}: broken pred chain at {v}"
                assert d[u] <= d[v], f"run {i}: non-descending pred chain"
                v = u
                hops += 1
                assert hops <= size * size, f"run {i}: pred cycle"


class TestV5Predecessors:
    def test_pred_chain_reconstructs(self):
        rng = np.random.default_rng(11)
        n = 200
        raster = rng.integers(1, 200, (n, n)).astype(np.uint16)
        d5, pred = sssp_raster_gpu_v5(raster, STEPS_8, 0, delta="auto",
                                      return_predecessor=True)
        target = n * n - 1
        assert np.isfinite(d5[target]) and d5[target] < 1e29
        # Walk the chain: dist must strictly decrease, terminate at source.
        v, hops = target, 0
        while v != 0:
            u = int(pred[v])
            assert u >= 0, f"broken pred chain at {v}"
            assert d5[u] < d5[v]
            v = u
            hops += 1
            assert hops <= n * n, "pred cycle"

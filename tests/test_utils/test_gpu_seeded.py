"""Many-seed GPU field (``GpuSsspSession.solve_seeded``): the float32 bound against the exact CPU kernel.

Plan 2026-10-06-paper-run-pipeline.md section 6: a float32 GPU field may only be used as a lower bound with a
rounding margin; the exact seeded kernel is the reference. Run in its own pytest process (GPU).
"""
import numpy as np
import pytest

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    GPU = True
except Exception:  # noqa: BLE001
    GPU = False

gpu_only = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

STEPS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1],
                  [1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]], dtype=np.int8)
U = 2.0 ** -24


def _raster(seed=0, shape=(70, 80)):
    rng = np.random.default_rng(seed)
    r = rng.integers(20, 400, size=shape).astype(np.uint16)
    r[rng.random(shape) < 0.08] = 65535
    return r


def _seeds(r, n, seed=1):
    rng = np.random.default_rng(seed)
    free = np.flatnonzero(r.ravel() != 65535)
    s = rng.choice(free, size=n, replace=False)
    return s, rng.random(n).astype(np.float32) * 300.0


@gpu_only
def test_seeded_field_is_a_sound_bound_of_the_exact_kernel():
    from pyorps.graph.drain import drain
    from pyorps.utils.sssp_gpu import GpuSsspSession

    r = _raster()
    seeds, d0 = _seeds(r, 12)
    exact = drain(r, STEPS, seeds, d0.astype(np.float64))
    with GpuSsspSession(r, STEPS, arena_factor=1.0) as s:
        got = s.solve_seeded(seeds, d0).reshape(r.shape)
    fin = np.isfinite(exact)
    assert np.array_equal(fin, np.isfinite(got))
    k_max = int(np.ceil(exact[fin].max() / 10.0)) + 2        # every step costs at least 10 here
    lower = got[fin] * (1.0 - k_max * U)
    assert (lower <= exact[fin] * (1 + 1e-12)).all()          # the margin makes it a sound lower bound
    assert np.allclose(got[fin], exact[fin], rtol=1e-4)       # and it is close


@gpu_only
def test_seeded_equals_the_minimum_of_single_source_runs_with_a_grade_table():
    from pyorps.core.objective import GradientOptions, Objective
    from pyorps.utils.sssp_gpu import GpuSsspSession

    r = _raster(seed=3)
    rng = np.random.default_rng(4)
    dem = np.cumsum(np.cumsum(rng.normal(0, 0.3, r.shape), 0), 1).astype(np.float32)
    luts = Objective({"cost": 1.0}, GradientOptions(max_gradient_pct=40.0)).build_gradient_luts(STEPS, 1.0)
    seeds, d0 = _seeds(r, 5, seed=5)
    with GpuSsspSession(r, STEPS, dem=dem, gradient_luts=luts, arena_factor=1.0) as s:
        seeded = s.solve_seeded(seeds, d0)
        singles = np.min([s.solve(int(c)) + o for c, o in zip(seeds, d0)], axis=0)
    fin = np.isfinite(singles)
    assert np.array_equal(fin, np.isfinite(seeded))
    assert np.allclose(seeded[fin], singles[fin], rtol=2e-5)


@gpu_only
def test_seed_validation():
    from pyorps.utils.sssp_gpu import GpuSsspSession

    r = _raster()
    wall = int(np.flatnonzero(r.ravel() == 65535)[0])
    with GpuSsspSession(r, STEPS, arena_factor=1.0) as s:
        with pytest.raises(ValueError, match="impassable"):
            s.solve_seeded([wall])
        with pytest.raises(ValueError, match="outside"):
            s.solve_seeded([r.size])
        with pytest.raises(ValueError, match="non-negative"):
            s.solve_seeded([0], [-1.0])
        free = int(np.flatnonzero(r.ravel() != 65535)[0])
        both = s.solve_seeded([free, free], [5.0, 1.0])
        one = s.solve_seeded([free], [1.0])
        assert np.array_equal(both, one)                       # a repeated seed keeps the smallest label

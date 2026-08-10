"""Tests for the GPU eikonal / block-FIM solver and the path tracer.

Comparison contract (plan section 1): FIM is *supposed* to differ from the
discrete backends — never assert bit-exactness against Dijkstra/V5.
Analytic ground truth is the referee wherever a closed form exists;
otherwise the naive Jacobi solver serves as the oracle for the block-FIM
scheme (same discretization, trivially correct iteration).

Tracer + rasterization tests are pure numpy and run without a GPU.
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

from pyorps.utils.eikonal_gpu import (
    FINITE_LIMIT,
    _disk_init_values,
    _masked_gradient,
    _slowness_field,
    polyline_to_cells,
    trace_path,
    trace_paths,
    trace_paths_gpu,
)

if GPU:
    from pyorps.utils.eikonal_gpu import (
        eikonal_raster_gpu,
        eikonal_raster_gpu_naive,
    )

needs_gpu = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")


def idx(r, c, cols):
    return r * cols + c


def cone_field(rows, cols, sr, sc, c=1.0):
    """Analytic uniform-cost field T = c * r (euclidean)."""
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return (c * np.hypot(rr - sr, cc - sc)).astype(np.float32)


def snell_reference(c1, c2, src_rc, tgt_rc, interface_row):
    """Analytic two-half-plane minimum via 1D minimization over the
    crossing point on the interface line (row = interface_row)."""
    sr, sc = src_rc
    tr, tc = tgt_rc
    x = np.linspace(min(sc, tc) - 50.0, max(sc, tc) + 50.0, 200001)
    leg1 = np.hypot(interface_row - sr, x - sc)
    leg2 = np.hypot(tr - interface_row, tc - x)
    return float(np.min(c1 * leg1 + c2 * leg2))


# ============================================================================
# Phase 0 gates: calibration + point source
# ============================================================================

@needs_gpu
class TestCalibration:
    """The unit-convention contract: uniform axis line -> T = v * L."""

    @pytest.mark.parametrize("solver_kw", [{}, {"tile": 8}, {"tile": 32}])
    def test_uniform_axis_line_fim(self, solver_kw):
        v, rows, cols = 7, 60, 160
        raster = np.full((rows, cols), v, dtype=np.uint16)
        source = idx(30, 20, cols)
        t = eikonal_raster_gpu(raster, source, **solver_kw)
        for length in (10, 50, 100, 130):
            expected = float(v * length)
            got = float(t[30, 20 + length])
            assert got == pytest.approx(expected, rel=1e-5), \
                f"axis calibration broken at L={length}"

    def test_uniform_axis_line_naive(self):
        v, rows, cols = 3, 50, 120
        raster = np.full((rows, cols), v, dtype=np.uint16)
        t = eikonal_raster_gpu_naive(raster, idx(25, 10, cols))
        assert float(t[25, 110]) == pytest.approx(3.0 * 100, rel=1e-5)

    def test_uniform_axis_line_float_raster(self):
        raster = np.full((40, 100), 2.5, dtype=np.float32)
        t = eikonal_raster_gpu(raster, idx(20, 5, 100))
        assert float(t[20, 95]) == pytest.approx(2.5 * 90, rel=1e-5)

    def test_uniform_point_source_error(self):
        """Field error vs T = c*r with the default 3-cell exact disk.

        Axis directions are exact; the L-inf error sits on the diagonals
        near the source — the known first-order singularity pollution
        (O(sqrt(h)) unless the exact disk is fixed in physical units;
        the convergence tests demonstrate clean O(h) with a scaled disk).
        """
        n, v = 201, 10
        raster = np.full((n, n), v, dtype=np.uint16)
        sr = sc = n // 2
        t = eikonal_raster_gpu(raster, idx(sr, sc, n))
        exact = cone_field(n, n, sr, sc, float(v))
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        r_cells = np.hypot(rr - sr, cc - sc)
        probe = r_cells >= 5.0    # outside the frozen init disk
        rel = np.abs(t[probe] - exact[probe]) / exact[probe]
        assert float(rel.max()) < 0.06, \
            f"point-source L-inf rel error {rel.max():.4f} >= 6%"
        assert float(rel.mean()) < 0.015
        # axis rays are reproduced exactly by the upwind scheme
        axis_err = np.abs(t[sr, sc + 5:] - exact[sr, sc + 5:])
        assert float((axis_err / exact[sr, sc + 5:]).max()) < 1e-5


# ============================================================================
# Phase 1 gate: block-FIM vs naive oracle
# ============================================================================

@needs_gpu
class TestFIMvsNaive:
    """Same discretization, different iteration -> near-identical fields.

    Differences are bounded by the eps convergence slack accumulated
    along causal chains (not bit-exact by construction).
    """

    def _compare(self, raster, source, rel_tol=1e-3, **kw):
        naive_kw = {k: v for k, v in kw.items()
                    if k not in ("tile", "n_inner")}
        ref = eikonal_raster_gpu_naive(raster, source, **naive_kw)
        fim = eikonal_raster_gpu(raster, source, **kw)
        f_ref = ref < FINITE_LIMIT
        f_fim = fim < FINITE_LIMIT
        assert np.array_equal(f_ref, f_fim), "reachability mask differs"
        if f_ref.any():
            denom = np.maximum(ref[f_ref], 1e-6)
            rel = np.abs(fim[f_ref] - ref[f_ref]) / denom
            assert float(rel.max()) < rel_tol, \
                f"max rel deviation {rel.max():.2e} vs naive oracle"

    def test_random_500(self):
        rng = np.random.default_rng(42)
        raster = rng.integers(1, 200, (500, 500)).astype(np.uint16)
        self._compare(raster, idx(250, 250, 500))

    def test_random_offcenter_source(self):
        rng = np.random.default_rng(7)
        raster = rng.integers(1, 50, (300, 200)).astype(np.uint16)
        self._compare(raster, idx(10, 190, 200))

    def test_heavy_tail(self):
        rng = np.random.default_rng(3)
        raster = np.clip(rng.lognormal(2.0, 1.5, (300, 300)),
                         1, 5000).astype(np.uint16)
        self._compare(raster, idx(150, 150, 300))

    def test_wall_with_gap(self):
        rng = np.random.default_rng(5)
        raster = rng.integers(1, 100, (200, 200)).astype(np.uint16)
        raster[20:180, 100] = np.iinfo(np.uint16).max
        self._compare(raster, idx(100, 20, 200))

    @pytest.mark.parametrize("tile", [8, 32])
    def test_tile_sizes(self, tile):
        rng = np.random.default_rng(11)
        raster = rng.integers(1, 200, (257, 131)).astype(np.uint16)
        self._compare(raster, idx(128, 65, 131), tile=tile)


# ============================================================================
# Phase 2 gates: exclusions, unreachable, multi-source, sources
# ============================================================================

@needs_gpu
class TestExclusions:
    def test_wall_with_gap_goes_through_gap(self):
        rows = cols = 100
        raster = np.ones((rows, cols), dtype=np.uint16)
        raster[0:80, 50] = np.iinfo(np.uint16).max   # gap at rows 80..99
        source = idx(40, 10, cols)
        target = idx(40, 90, cols)
        t = eikonal_raster_gpu(raster, source)
        assert t[40, 90] < FINITE_LIMIT
        # wall cells never traversed / never reached
        assert (t[0:80, 50] >= FINITE_LIMIT).all()
        # detour through the gap is much longer than the straight line
        assert float(t[40, 90]) > 80.0
        poly = trace_path(t, target, source)
        cells = polyline_to_cells(poly[::-1], rows, cols,
                                  forbidden_mask=t >= FINITE_LIMIT)
        wall = {idx(r, 50, cols) for r in range(0, 80)}
        assert not (set(cells) & wall), "traced path crosses the wall"
        # the path actually dips down through the gap
        assert max(i // cols for i in cells) >= 79

    def test_enclosed_target_unreachable(self):
        raster = np.ones((60, 60), dtype=np.uint16)
        raster[20, 20:41] = np.iinfo(np.uint16).max
        raster[40, 20:41] = np.iinfo(np.uint16).max
        raster[20:41, 20] = np.iinfo(np.uint16).max
        raster[20:41, 40] = np.iinfo(np.uint16).max
        t = eikonal_raster_gpu(raster, idx(5, 5, 60))
        assert t[30, 30] >= FINITE_LIMIT
        assert t[5, 50] < FINITE_LIMIT

    def test_ignore_max_false_wall_passable(self):
        raster = np.ones((40, 40), dtype=np.uint16)
        raster[:, 20] = np.iinfo(np.uint16).max
        t = eikonal_raster_gpu(raster, idx(20, 5, 40), ignore_max=False)
        assert t[20, 35] < FINITE_LIMIT
        # crossing one expensive cell: roughly 29 cheap cells + one 65535
        assert float(t[20, 35]) > 60000.0

    def test_float_raster_inf_excluded(self):
        raster = np.ones((50, 50), dtype=np.float32)
        raster[:40, 25] = np.inf
        t = eikonal_raster_gpu(raster, idx(25, 5, 50))
        assert (t[:40, 25] >= FINITE_LIMIT).all()
        assert t[25, 45] < FINITE_LIMIT
        assert float(t[25, 45]) > 40.0   # forced detour

    def test_source_on_impassable_cell(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        raster[15, 15] = np.iinfo(np.uint16).max
        t = eikonal_raster_gpu(raster, idx(15, 15, 30))
        assert (t >= FINITE_LIMIT).all()

    def test_source_out_of_range_raises(self):
        raster = np.ones((20, 20), dtype=np.uint16)
        with pytest.raises(ValueError):
            eikonal_raster_gpu(raster, 400)


@needs_gpu
class TestMultiSource:
    def test_min_of_single_source_fields(self):
        """Multi-source vs pointwise min of the single-source fields.

        The min of the individual *discrete* solutions is a supersolution
        of the union-boundary system, so tm <= min(t1, t2) holds sharply.
        At shock cells the Godunov quadratic mixes the two wavefronts and
        lands strictly below the min (closer to the continuum truth) —
        the deviation is one-sided and metrication-scale, never 1e-5
        (measured ~0.5% max on this raster; the true oracle equality is
        against the naive multi-source solve).
        """
        rng = np.random.default_rng(9)
        raster = rng.integers(1, 60, (150, 150)).astype(np.uint16)
        s1 = idx(20, 20, 150)
        s2 = idx(120, 130, 150)
        t1 = eikonal_raster_gpu(raster, s1)
        t2 = eikonal_raster_gpu(raster, s2)
        tm = eikonal_raster_gpu(raster, np.array([s1, s2]))
        ref = np.minimum(t1, t2)
        finite = ref < FINITE_LIMIT
        assert np.array_equal(finite, tm < FINITE_LIMIT)
        # supersolution bound: tm <= min of singles (+ float noise)
        overshoot = (tm[finite] - ref[finite]) / np.maximum(ref[finite],
                                                            1.0)
        assert float(overshoot.max()) < 1e-4, \
            f"multi-source exceeds min-of-fields by {overshoot.max():.2e}"
        # one-sided shock mixing stays metrication-scale
        undershoot = (ref[finite] - tm[finite]) / np.maximum(ref[finite],
                                                             1.0)
        assert float(undershoot.max()) < 0.02
        # away from shocks the fields agree tightly (99% of cells)
        assert float(np.quantile(undershoot, 0.99)) < 1e-3
        # oracle: naive multi-source gives the same fixed point
        nm = eikonal_raster_gpu_naive(raster, np.array([s1, s2]))
        rel = (np.abs(tm[finite] - nm[finite])
               / np.maximum(nm[finite], 1.0))
        assert float(rel.max()) < 1e-3

    def test_close_sources_skip_disk_init(self):
        raster = np.full((50, 50), 5, dtype=np.uint16)
        s1 = idx(25, 24, 50)
        s2 = idx(25, 27, 50)   # 3 cells apart < 2*r0 = 6
        seed_idx, seed_val = _disk_init_values(
            np.array([s1, s2]), raster, True)
        assert seed_idx.size == 0
        # solve still works, both sources at 0
        t = eikonal_raster_gpu(raster, np.array([s1, s2]))
        assert t[25, 24] == 0.0 and t[25, 27] == 0.0
        assert t[25, 40] < FINITE_LIMIT

    def test_disk_init_skipped_on_nonuniform_cost(self):
        raster = np.full((50, 50), 5, dtype=np.uint16)
        raster[25, 27] = 200   # breaks local constancy near the source
        seed_idx, _ = _disk_init_values(
            np.array([idx(25, 25, 50)]), raster, True)
        assert seed_idx.size == 0

    def test_disk_init_values_exact(self):
        raster = np.full((30, 30), 4, dtype=np.uint16)
        seed_idx, seed_val = _disk_init_values(
            np.array([idx(15, 15, 30)]), raster, True)
        assert seed_idx.size > 1
        rr, cc = np.divmod(seed_idx, 30)
        expected = 4.0 * np.hypot(rr - 15, cc - 15)
        np.testing.assert_allclose(seed_val, expected, rtol=1e-6)


# ============================================================================
# Analytic suite beyond phase 0 (Snell, radial) + convergence
# ============================================================================

@needs_gpu
class TestAnalytic:
    def test_snell_two_half_planes(self):
        """Refraction: cost error vs the closed-form minimum <= 1%."""
        n = 200
        c1, c2 = 5.0, 15.0
        raster = np.full((n, n), c1, dtype=np.uint16)
        raster[n // 2:, :] = int(c2)
        sr, sc = 30, 40
        tr, tc = 170, 160
        t = eikonal_raster_gpu(raster, idx(sr, sc, n))
        # interface between cell rows 99 and 100 -> continuous y = 99.5
        expected = snell_reference(c1, c2, (sr, sc), (tr, tc), 99.5)
        got = float(t[tr, tc])
        assert got == pytest.approx(expected, rel=0.01), \
            f"Snell cost off by {abs(got - expected) / expected:.4f}"

    def test_snell_path_bends_at_interface(self):
        """The traced path refracts: straighter in the slow medium."""
        n = 200
        raster = np.full((n, n), 5, dtype=np.uint16)
        raster[n // 2:, :] = 15
        source = idx(30, 40, n)
        target = idx(170, 160, n)
        t = eikonal_raster_gpu(raster, source)
        poly = trace_path(t, target, source)   # target -> source order
        assert poly is not None
        rows_ = poly[:, 0]
        cols_ = poly[:, 1]
        # horizontal travel per medium: most sideways motion happens in
        # the fast (cheap) upper medium
        fast = rows_ < 99.5
        slow = ~fast
        dx_fast = np.abs(np.diff(cols_[fast])).sum()
        dx_slow = np.abs(np.diff(cols_[slow])).sum()
        assert dx_fast > 1.5 * dx_slow

    def test_radial_linear_slowness(self):
        """c(r) = a + b*r: radial ray optimal, T(R) = a R + b R^2 / 2."""
        n = 201
        a, b = 10.0, 0.5
        sr = sc = n // 2
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        r_cells = np.hypot(rr - sr, cc - sc)
        raster = (a + b * r_cells).astype(np.float32)
        t = eikonal_raster_gpu(raster, idx(sr, sc, n))
        for radius in (20, 50, 90):
            expected = a * radius + b * radius ** 2 / 2.0
            got = float(t[sr, sc + radius])
            assert got == pytest.approx(expected, rel=0.02), \
                f"radial closed form off at R={radius}"

    def test_grid_convergence_uniform(self):
        """Refining h halves the error: observed order >= ~0.8.

        The exact-init disk is held fixed in PHYSICAL units across the
        refinement (disk_radius scales with n): with an h-scale disk the
        point-source singularity pollutes the diagonals at O(sqrt(h))
        and floors the observed order — the known first-order behavior,
        measured and documented in EIKONAL_FINDINGS.md.
        """
        v = 10
        errors = []
        for n in (101, 201, 401):
            raster = np.full((n, n), v, dtype=np.uint16)
            sr = sc = n // 2
            t = eikonal_raster_gpu(raster, idx(sr, sc, n),
                                   disk_radius=3.0 * n / 101.0)
            exact = cone_field(n, n, sr, sc, float(v))
            rr, cc = np.meshgrid(np.arange(n), np.arange(n),
                                 indexing="ij")
            r_cells = np.hypot(rr - sr, cc - sc)
            probe = r_cells >= 0.08 * n   # same physical region each n
            rel = np.abs(t[probe] - exact[probe]) / exact[probe]
            errors.append(float(rel.max()))
        orders = [np.log2(errors[i] / errors[i + 1])
                  for i in range(len(errors) - 1)]
        # measured: [1.03, 0.98] — float32 does not floor it here
        assert min(orders) >= 0.8, \
            f"convergence order {orders} below ~0.8 (errors: {errors})"

    def test_grid_convergence_snell(self):
        c1, c2 = 5.0, 15.0
        errors = []
        for n in (100, 200, 400):
            raster = np.full((n, n), c1, dtype=np.uint16)
            raster[n // 2:, :] = int(c2)
            src = (int(0.15 * n), int(0.2 * n))
            tgt = (int(0.85 * n), int(0.8 * n))
            t = eikonal_raster_gpu(raster, idx(*src, n),
                                   disk_radius=3.0 * n / 100.0)
            expected = snell_reference(c1, c2, src, tgt, n // 2 - 0.5)
            errors.append(abs(float(t[tgt]) - expected) / expected)
        orders = [np.log2(errors[i] / errors[i + 1])
                  for i in range(len(errors) - 1)]
        # measured: [0.98, 0.98]
        assert min(orders) >= 0.8, \
            f"Snell convergence degraded: orders {orders}, errs {errors}"


@needs_gpu
class TestTracerOnSolvedFields:
    def test_noisy_random_field_traces_to_source(self):
        """Regression: cell-scale noise creates interpolation attractors
        that cycled the tracer against the discrete-hop fallback until
        the step cap. The monotone-T rule must keep it terminating."""
        rng = np.random.default_rng(20260806)
        raster = rng.integers(1, 200, (500, 500)).astype(np.uint16)
        t = eikonal_raster_gpu(raster, 0)
        poly = trace_path(t, 500 * 500 - 1, 0)
        assert poly is not None
        assert tuple(poly[0]) == (499.0, 499.0)
        assert tuple(poly[-1]) == (0.0, 0.0)
        seg = np.diff(poly, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1]).sum()
        assert length < 4 * 499 * np.sqrt(2), "wildly wandering trace"

    def test_heavy_tail_field_traces(self):
        rng = np.random.default_rng(3)
        raster = np.clip(rng.lognormal(2.0, 1.5, (300, 300)),
                         1, 5000).astype(np.uint16)
        t = eikonal_raster_gpu(raster, 0)
        poly = trace_path(t, 300 * 300 - 1, 0)
        assert poly is not None and tuple(poly[-1]) == (0.0, 0.0)


def spiral_raster(n=200, corridor=6, v=5):
    """Rectangular spiral corridor: passable path winds from the border
    to the center — long wound characteristics, the worst case for
    value iteration and for the outer-iteration cap."""
    raster = np.full((n, n), v, dtype=np.uint16)
    wall = np.iinfo(np.uint16).max
    step = 2 * corridor
    lo, hi = corridor, n - corridor
    k = 0
    while hi - lo > 2 * step:
        if k % 4 == 0:      # top wall, gap at right
            raster[lo, lo:hi] = wall
        elif k % 4 == 1:    # right wall, gap at bottom
            raster[lo:hi, hi - 1] = wall
        elif k % 4 == 2:    # bottom wall, gap at left
            raster[hi - 1, lo + step:hi] = wall
            lo += step
        else:               # left wall, gap at top
            raster[lo:hi - step, lo] = wall
            hi -= step
        k += 1
    return raster


@needs_gpu
class TestSpiralCharacteristics:
    """Wound characteristics stress (the paper's 'iterations grow with
    speed-function complexity' caveat taken to its geometric extreme):
    correctness of tile re-activation over many wavefront turns, and
    the default outer-iteration cap staying generous enough."""

    def test_spiral_matches_naive_oracle(self):
        raster = spiral_raster(200)
        source = idx(2, 2, 200)
        fim, passes = eikonal_raster_gpu(raster, source,
                                         return_iterations=True)
        ref = eikonal_raster_gpu_naive(raster, source,
                                       max_iterations=20000)
        f_ref = ref < FINITE_LIMIT
        assert np.array_equal(f_ref, fim < FINITE_LIMIT)
        rel = (np.abs(fim[f_ref] - ref[f_ref])
               / np.maximum(ref[f_ref], 1e-6))
        assert float(rel.max()) < 1e-3
        # the winding path is far longer than the grid diagonal
        assert float(np.nanmax(np.where(f_ref, fim, np.nan))) > 5 * 200
        # default cap (64 * tiles_per_side) was not even close
        assert passes < 64 * ((200 + 15) // 16)

    def test_spiral_trace_reaches_source(self):
        raster = spiral_raster(160, corridor=8)
        source = idx(2, 2, 160)
        t = eikonal_raster_gpu(raster, source)
        finite = t < FINITE_LIMIT
        target = int(np.argmax(np.where(finite, t, -1.0)))
        poly = trace_path(t, target, source)
        assert poly is not None
        assert tuple(poly[-1]) == (2.0, 2.0)


@needs_gpu
class TestUpdateStats:
    def test_redundancy_metric(self):
        rng = np.random.default_rng(6)
        raster = rng.integers(1, 200, (300, 300)).astype(np.uint16)
        t, stats = eikonal_raster_gpu(raster, idx(150, 150, 300),
                                      return_stats=True)
        assert t[0, 0] < FINITE_LIMIT
        assert stats["outer_passes"] > 0
        assert stats["tile_sweeps"] >= stats["outer_passes"]
        # settle checks end sweeps early: actual iterations are counted
        # (at most n_inner = 32 per sweep, at least the first check
        # window of 8)
        assert stats["cell_updates"] <= (stats["tile_sweeps"] * 16 * 16
                                         * 32)
        assert stats["cell_updates"] >= (stats["tile_sweeps"] * 16 * 16
                                         * 8)
        assert stats["updates_per_cell"] > 1.0

    def test_stats_on_unreachable(self):
        raster = np.ones((40, 40), dtype=np.uint16)
        raster[10, 10] = np.iinfo(np.uint16).max
        t, stats = eikonal_raster_gpu(raster, idx(10, 10, 40),
                                      return_stats=True)
        assert (t >= FINITE_LIMIT).all()
        assert stats["tile_sweeps"] == 0


@needs_gpu
class TestSolverGuards:
    def test_outer_iteration_cap_raises(self):
        rng = np.random.default_rng(1)
        raster = rng.integers(1, 200, (200, 200)).astype(np.uint16)
        with pytest.raises(RuntimeError, match="outer-iteration cap"):
            eikonal_raster_gpu(raster, 0, max_outer_iterations=2)

    def test_naive_iteration_cap_raises(self):
        raster = np.ones((100, 100), dtype=np.uint16)
        with pytest.raises(RuntimeError, match="iteration cap"):
            eikonal_raster_gpu_naive(raster, 0, max_iterations=5)

    def test_return_iterations(self):
        raster = np.ones((64, 64), dtype=np.uint16)
        t, n_outer = eikonal_raster_gpu(raster, idx(32, 32, 64),
                                        return_iterations=True)
        assert n_outer > 0
        assert t[0, 0] < FINITE_LIMIT

    def test_negative_float_cost_rejected(self):
        raster = np.full((20, 20), -1.0, dtype=np.float32)
        with pytest.raises(ValueError, match="negative"):
            eikonal_raster_gpu(raster, 0)


# ============================================================================
# Tracer robustness (pure numpy — no GPU required)
# ============================================================================

class TestTracer:
    def test_straight_descent_uniform_cone(self):
        t = cone_field(100, 100, 50, 50)
        poly = trace_path(t, idx(50, 90, 100), idx(50, 50, 100))
        assert poly is not None
        assert tuple(poly[0]) == (50.0, 90.0)
        assert tuple(poly[-1]) == (50.0, 50.0)
        # straight line: no point strays far from row 50
        assert np.abs(poly[:, 0] - 50.0).max() < 1.5
        # path length close to euclidean
        seg = np.diff(poly, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1]).sum()
        assert length < 41.0 * 1.05

    def test_diagonal_descent(self):
        t = cone_field(120, 120, 20, 20)
        poly = trace_path(t, idx(100, 100, 120), idx(20, 20, 120))
        seg = np.diff(poly, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1]).sum()
        assert length == pytest.approx(80 * np.sqrt(2), rel=0.05)

    def test_unreachable_target_returns_none(self):
        t = cone_field(50, 50, 25, 25)
        t[40, 40] = 1e30
        assert trace_path(t, idx(40, 40, 50), idx(25, 25, 50)) is None

    def test_plateau_fallback(self):
        """A flat T region (zero-cost plaza) is crossed via BFS escape."""
        t = cone_field(80, 80, 40, 10)
        # plaza: constant T over a block right of the cone
        plaza_t = float(t[40, 50])
        t[30:51, 50:71] = plaza_t
        poly = trace_path(t, idx(40, 70, 80), idx(40, 10, 80))
        assert poly is not None
        assert tuple(poly[-1]) == (40.0, 10.0)

    def test_shock_ridge_between_two_sources(self):
        """Equidistant ridge: oscillation fallback must resolve it."""
        rows = cols = 100
        s1 = idx(50, 20, cols)
        s2 = idx(50, 80, cols)
        t = np.minimum(cone_field(rows, cols, 50, 20),
                       cone_field(rows, cols, 50, 80))
        poly = trace_path(t, idx(5, 50, cols), np.array([s1, s2]))
        assert poly is not None
        end = tuple(poly[-1])
        assert end in ((50.0, 20.0), (50.0, 80.0))

    def test_max_steps_cap_raises(self):
        t = cone_field(200, 200, 100, 100)
        with pytest.raises(RuntimeError, match="max_steps"):
            trace_path(t, idx(100, 195, 200), idx(100, 100, 200),
                       max_steps=10)

    def test_broken_field_local_minimum_raises(self):
        """A spurious local minimum away from any source must be loud."""
        t = cone_field(60, 60, 10, 10)   # fake attractor at (10, 10)
        with pytest.raises(RuntimeError):
            # true source elsewhere; descent lands in the fake minimum
            trace_path(t, idx(50, 50, 60), idx(55, 5, 60))

    def test_target_is_source(self):
        t = cone_field(30, 30, 15, 15)
        poly = trace_path(t, idx(15, 15, 30), idx(15, 15, 30))
        assert poly is not None and len(poly) >= 1
        assert tuple(poly[0]) == (15.0, 15.0)

    def test_trace_paths_multiple_targets(self):
        t = cone_field(100, 100, 50, 50)
        polys = trace_paths(t, idx(50, 50, 100),
                            [idx(10, 50, 100), idx(90, 90, 100)])
        assert len(polys) == 2
        assert all(p is not None for p in polys)

    def test_masked_gradient_ignores_walls(self):
        t = cone_field(50, 50, 25, 5)
        t[:, 30] = 1e30
        gr, gc = _masked_gradient(t)
        assert np.isnan(gr[10, 30]) and np.isnan(gc[10, 30])
        # neighbors of the wall use one-sided differences: finite
        assert np.isfinite(gc[10, 29]) and np.isfinite(gc[10, 31])


class TestPolylineToCells:
    def test_straight_segment(self):
        poly = np.array([[5.0, 2.0], [5.0, 9.0]])
        cells = polyline_to_cells(poly, 10, 12)
        assert cells[0] == 5 * 12 + 2
        assert cells[-1] == 5 * 12 + 9
        assert cells == [5 * 12 + c for c in range(2, 10)]

    def test_diagonal_connected(self):
        poly = np.array([[0.0, 0.0], [7.0, 7.0]])
        cells = polyline_to_cells(poly, 8, 8)
        rc = [(i // 8, i % 8) for i in cells]
        for (r0, c0), (r1, c1) in zip(rc, rc[1:]):
            assert max(abs(r1 - r0), abs(c1 - c0)) == 1, "gap in path"
        assert cells[0] == 0 and cells[-1] == 63

    def test_no_duplicates_on_slow_curve(self):
        ts = np.linspace(0, np.pi / 2, 200)
        poly = np.stack([10 + 8 * np.sin(ts), 10 + 8 * np.cos(ts)]).T
        cells = polyline_to_cells(poly, 30, 30)
        for a, b in zip(cells, cells[1:]):
            assert a != b
        for a, b in zip(cells, cells[2:]):
            assert a != b, "immediate backtrack survived"

    def test_empty(self):
        assert polyline_to_cells(np.empty((0, 2)), 5, 5) == []


@needs_gpu
class TestDeviceTracer:
    """trace_paths_gpu is a drop-in for trace_paths (increment 2).

    Individual float-marginal direction decisions may differ between
    the kernel and the host tracer; both must satisfy the same
    invariants (reaches the source, monotone T, passable cells).
    """

    def _path_cost(self, t, poly):
        """Interpolated T at the polyline start (authoritative cost)."""
        return float(t[int(poly[0][0]), int(poly[0][1])])

    def test_parity_random_field(self):
        rng = np.random.default_rng(7)
        raster = rng.integers(1, 200, (300, 300)).astype(np.uint16)
        t, d_t = eikonal_raster_gpu(raster, 0, return_device=True)
        tgt = 300 * 300 - 1
        host = trace_paths(t, 0, [tgt])[0]
        dev = trace_paths_gpu(t, 0, [tgt], t_device=d_t)[0]
        assert dev is not None and host is not None
        assert dev.dtype == np.float64 and dev.shape[1] == 2
        assert tuple(dev[0]) == (299.0, 299.0)
        assert tuple(dev[-1]) == (0.0, 0.0)
        # equivalent length (not bit-identical decisions)
        len_h = np.hypot(*np.diff(host, axis=0).T).sum()
        len_d = np.hypot(*np.diff(dev, axis=0).T).sum()
        assert abs(len_h - len_d) <= 0.05 * len_h + 2.0

    def test_monotone_t_along_device_trace(self):
        rng = np.random.default_rng(11)
        raster = rng.integers(1, 100, (200, 200)).astype(np.uint16)
        t, d_t = eikonal_raster_gpu(raster, 0, return_device=True)
        dev = trace_paths_gpu(t, 0, [200 * 200 - 1], t_device=d_t)[0]
        vals = t[np.clip(dev[:, 0].round().astype(int), 0, 199),
                 np.clip(dev[:, 1].round().astype(int), 0, 199)]
        # rounded-cell T may wobble below cell scale; no big uphill jumps
        assert (np.diff(vals) < np.median(raster) * 2.0).all()

    def test_wall_gap(self):
        raster = np.ones((100, 100), dtype=np.uint16)
        raster[0:80, 50] = np.iinfo(np.uint16).max
        src = idx(60, 10, 100)
        t, d_t = eikonal_raster_gpu(raster, src, return_device=True)
        dev = trace_paths_gpu(t, src, [idx(60, 90, 100)],
                              t_device=d_t)[0]
        assert dev is not None
        cells = polyline_to_cells(dev, 100, 100,
                                  forbidden_mask=t >= FINITE_LIMIT)
        assert all(raster.ravel()[c] != 65535 for c in cells)
        assert cells[-1] == src

    def test_unreachable_gives_none(self):
        raster = np.ones((60, 60), dtype=np.uint16)
        raster[20:40, 20:40] = np.iinfo(np.uint16).max
        raster[25:35, 25:35] = 1
        t = eikonal_raster_gpu(raster, 0)
        assert trace_paths_gpu(t, 0, [idx(30, 30, 60)])[0] is None

    def test_multi_source_stops_at_nearest(self):
        raster = np.full((120, 120), 7, dtype=np.uint16)
        sources = np.array([idx(10, 10, 120), idx(110, 110, 120)])
        t, d_t = eikonal_raster_gpu(raster, sources, return_device=True)
        dev = trace_paths_gpu(t, sources, [idx(100, 100, 120)],
                              t_device=d_t)[0]
        assert dev is not None
        assert tuple(dev[-1]) == (110.0, 110.0)   # nearer source wins

    def test_multi_target_one_call(self):
        rng = np.random.default_rng(5)
        raster = rng.integers(1, 50, (150, 150)).astype(np.uint16)
        t, d_t = eikonal_raster_gpu(raster, 0, return_device=True)
        targets = [idx(140, 140, 150), idx(10, 140, 150),
                   idx(140, 10, 150), idx(75, 75, 150)]
        polys = trace_paths_gpu(t, 0, targets, t_device=d_t)
        assert all(p is not None for p in polys)
        assert all(tuple(p[-1]) == (0.0, 0.0) for p in polys)

    def test_plateau_falls_back_to_host(self):
        """A flat-T region has no strictly-lower neighbor: the kernel
        flags host fallback, whose BFS crosses the plateau."""
        t = cone_field(60, 60, 5, 5)
        t[20:40, 20:40] = t[30, 30]          # flat plateau in the field
        tgt = idx(30, 30, 60)
        host = trace_path(t, tgt, idx(5, 5, 60))
        dev = trace_paths_gpu(t, idx(5, 5, 60), [tgt])[0]
        assert dev is not None and host is not None
        assert tuple(dev[-1]) == (5.0, 5.0)
        np.testing.assert_allclose(dev, host)   # same fallback code path

    def test_broken_field_raises_like_host(self):
        """Local minimum away from any source: host raises loudly; the
        wrapper's fallback must surface the same error."""
        t = cone_field(50, 50, 25, 25)
        far_src = idx(0, 0, 50)               # not the field's minimum
        with pytest.raises(RuntimeError, match="no admissible descent"):
            trace_paths_gpu(t, far_src, [idx(40, 40, 50)])

    def test_batching_matches_single_batch(self, monkeypatch):
        import pyorps.utils.eikonal_gpu as eik
        rng = np.random.default_rng(9)
        raster = rng.integers(1, 50, (80, 80)).astype(np.uint16)
        t, d_t = eikonal_raster_gpu(raster, 0, return_device=True)
        targets = [idx(70, 70, 80), idx(5, 70, 80), idx(70, 5, 80)]
        full = trace_paths_gpu(t, 0, targets, t_device=d_t)
        per_target = (20 * (80 + 80) + 2) * 2 * 8
        monkeypatch.setattr(eik, "_TRACE_BATCH_BYTES", per_target)
        batched = eik.trace_paths_gpu(t, 0, targets, t_device=d_t)
        for a, b in zip(full, batched):
            np.testing.assert_array_equal(a, b)


@needs_gpu
class TestTargetedEarlyExit:
    """target_index= stops the solve early with the solver's ordinary
    eps guarantee (increment 2): T[target] and every cell at
    T <= T[target] match the full solve to within the convergence
    tolerance. (Strict equality is not defined even between two full
    solves — the chaotic sweep leaves nondeterministic sub-eps slack.)
    """

    #: eps_rel default is 1e-6; sub-eps slack differences stay below a
    #: few eps.
    RTOL = 5e-6

    def test_target_value_exact_random(self):
        rng = np.random.default_rng(13)
        raster = rng.integers(1, 200, (400, 400)).astype(np.uint16)
        src = 0
        tgt = idx(200, 200, 400)     # mid-grid: exit has room to save
        t_full, n_full = eikonal_raster_gpu(raster, src,
                                            return_iterations=True)
        t_ee, n_ee = eikonal_raster_gpu(raster, src, target_index=tgt,
                                        return_iterations=True)
        np.testing.assert_allclose(t_ee.ravel()[tgt],
                                   t_full.ravel()[tgt], rtol=self.RTOL)
        assert n_ee <= n_full
        # the whole level-set region is final (to eps tolerance)
        level = t_full.ravel()[tgt]
        region = t_full <= level
        np.testing.assert_allclose(t_ee[region], t_full[region],
                                   rtol=self.RTOL, atol=1e-3)

    def test_traced_path_matches_full_solve(self):
        rng = np.random.default_rng(17)
        raster = rng.integers(1, 100, (300, 300)).astype(np.uint16)
        src, tgt = 0, idx(150, 150, 300)
        t_full = eikonal_raster_gpu(raster, src)
        t_ee, d_ee = eikonal_raster_gpu(raster, src, target_index=tgt,
                                        return_device=True)
        p_full = trace_paths(t_full, src, [tgt])[0]
        p_ee = trace_paths_gpu(t_ee, src, [tgt], t_device=d_ee)[0]
        assert p_ee is not None
        assert tuple(p_ee[-1]) == (0.0, 0.0)
        len_f = np.hypot(*np.diff(p_full, axis=0).T).sum()
        len_e = np.hypot(*np.diff(p_ee, axis=0).T).sum()
        assert abs(len_f - len_e) <= 0.05 * len_f + 2.0

    def test_corner_target_still_exact(self):
        """Corner-to-corner: exit fires at the very end (no savings) —
        must stay exact regardless."""
        rng = np.random.default_rng(19)
        raster = rng.integers(1, 50, (200, 200)).astype(np.uint16)
        tgt = 200 * 200 - 1
        t_full = eikonal_raster_gpu(raster, 0)
        t_ee = eikonal_raster_gpu(raster, 0, target_index=tgt)
        np.testing.assert_allclose(t_ee.ravel()[tgt],
                                   t_full.ravel()[tgt], rtol=self.RTOL)

    def test_unreachable_target_solves_fully(self):
        raster = np.ones((80, 80), dtype=np.uint16)
        raster[30:50, 30:50] = np.iinfo(np.uint16).max
        raster[35:45, 35:45] = 1
        tgt = idx(40, 40, 80)
        t = eikonal_raster_gpu(raster, 0, target_index=tgt)
        assert t.ravel()[tgt] >= FINITE_LIMIT       # never arrives
        # rest of the field is the ordinary full solve
        t_full = eikonal_raster_gpu(raster, 0)
        np.testing.assert_array_equal(t, t_full)

    def test_invalid_target_raises(self):
        raster = np.ones((50, 50), dtype=np.uint16)
        with pytest.raises(ValueError, match="target_index"):
            eikonal_raster_gpu(raster, 0, target_index=50 * 50)

    def test_spiral_early_exit_exact(self):
        """Wound characteristics: the target sits mid-spiral; the wave
        passes it long before the solve finishes."""
        raster = spiral_raster(200)
        src = idx(2, 2, 200)
        tgt = idx(3, 100, 200)      # early on the outer corridor
        t_full = eikonal_raster_gpu(raster, src)
        t_ee, n_ee = eikonal_raster_gpu(raster, src, target_index=tgt,
                                        return_iterations=True)
        _, n_full = eikonal_raster_gpu(raster, src,
                                       return_iterations=True)
        np.testing.assert_allclose(t_ee.ravel()[tgt],
                                   t_full.ravel()[tgt], rtol=self.RTOL)
        assert n_ee < n_full        # genuinely exits early here


@needs_gpu
class TestSecondOrder:
    """order=2 refinement (increment 2): second-order one-sided upwind
    with frozen structure, causality safeguard and stall degrade."""

    def _cone(self, n, v=10):
        sr = sc = n // 2
        raster = np.full((n, n), v, dtype=np.uint16)
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        exact = v * np.hypot(rr - sr, cc - sc)
        return raster, sr * n + sc, exact, rr, cc, sr, sc

    def test_cone_much_more_accurate(self):
        n = 401
        raster, src, exact, rr, cc, sr, sc = self._cone(n)
        probe = np.hypot(rr - sr, cc - sc) >= 10
        t1 = eikonal_raster_gpu(raster, src)
        t2 = eikonal_raster_gpu(raster, src, order=2)
        e1 = (np.abs(t1[probe] - exact[probe]) / exact[probe]).mean()
        e2 = (np.abs(t2[probe] - exact[probe]) / exact[probe]).mean()
        assert e2 < e1 / 4          # measured: ~9x better
        assert e2 < 0.003           # mean < 0.3% (measured 0.07%)

    def test_observed_order_two(self):
        """Grid convergence with physically scaled disk: observed
        order ~1.9-2.1 (measured 1.86-2.08; first order gives ~1)."""
        v = 10
        errs = []
        for n in (101, 201, 401):
            raster, src, exact, rr, cc, sr, sc = self._cone(n, v)
            r_phys = np.hypot(rr - sr, cc - sc) / (n / 101.0)
            probe = r_phys >= 10
            t = eikonal_raster_gpu(raster, src, order=2,
                                   disk_radius=3.0 * n / 101.0)
            errs.append(
                (np.abs(t[probe] - exact[probe]) / exact[probe]).mean())
        rate1 = np.log2(errs[0] / errs[1])
        rate2 = np.log2(errs[1] / errs[2])
        assert rate1 > 1.5 and rate2 > 1.5

    def test_snell_improves(self):
        n, c1, c2 = 400, 5, 15
        raster = np.full((n, n), c1, dtype=np.uint16)
        raster[n // 2:, :] = c2
        src = (int(0.15 * n), int(0.2 * n))
        tgt = (int(0.85 * n), int(0.8 * n))
        expected = snell_reference(c1, c2, src, tgt, n // 2 - 0.5)
        t1 = eikonal_raster_gpu(raster, src[0] * n + src[1])
        t2 = eikonal_raster_gpu(raster, src[0] * n + src[1], order=2)
        e1 = abs(float(t1[tgt]) - expected) / expected
        e2 = abs(float(t2[tgt]) - expected) / expected
        assert e2 < e1
        assert float(t2[tgt]) > expected * 0.999   # no gross undershoot

    def test_exclusions_respected(self):
        raster = np.ones((100, 100), dtype=np.uint16)
        raster[0:80, 50] = np.iinfo(np.uint16).max
        t = eikonal_raster_gpu(raster, idx(60, 10, 100), order=2)
        assert t[10, 90] < FINITE_LIMIT            # around the gap
        assert t[0, 50] >= FINITE_LIMIT            # wall stays walled

    def test_noise_field_safe_and_terminates(self):
        """Cell-scale noise: the smoothness test keeps most cells first
        order; the causality safeguard + stall degrade guarantee a
        bounded, finite, non-negative field."""
        rng = np.random.default_rng(0)
        raster = rng.integers(1, 200, (500, 500)).astype(np.uint16)
        t1 = eikonal_raster_gpu(raster, 0)
        t2 = eikonal_raster_gpu(raster, 0, order=2)
        assert np.isfinite(t2[t2 < FINITE_LIMIT]).all()
        assert (t2 >= 0).all()
        both = (t1 < FINITE_LIMIT) & (t2 < FINITE_LIMIT)
        shift = (np.abs(t2[both] - t1[both])
                 / np.maximum(t1[both], 1.0))
        assert shift.max() < 0.10   # bounded correction (measured 5.6%)

    def test_trace_field_contract(self):
        """Refined fields are for costs; tracing consumes the preserved
        first-order field (descent-connected by construction — refined
        fields develop genuine local minima near cost shocks on rough
        rasters, measured on the real planning surface). Smooth raster:
        the roughness guard must NOT fire here."""
        n = 200
        base = np.linspace(10, 60, n, dtype=np.float64)
        raster = (base[None, :] + base[:, None]).astype(np.uint16)
        t2, d2, (t_tr, d_tr) = eikonal_raster_gpu(
            raster, 0, order=2, return_device=True,
            return_trace_field=True)
        assert t_tr is not t2                     # distinct o1 field
        poly = trace_paths_gpu(t_tr, 0, [n * n - 1],
                               t_device=d_tr)[0]
        assert poly is not None
        assert tuple(poly[-1]) == (0.0, 0.0)
        # order=1: trace field IS the solved field, no copies
        t1, d1, (t1_tr, d1_tr) = eikonal_raster_gpu(
            raster, 0, return_device=True, return_trace_field=True)
        assert t1_tr is t1 and d1_tr is d1

    def test_rough_surface_guard_returns_first_order(self):
        """Cell-scale noise: the roughness guard skips the refinement
        (measured to undershoot up to 10% outside its domain) with a
        warning; the returned field is the first-order solution."""
        rng = np.random.default_rng(0)
        raster = rng.integers(1, 200, (300, 300)).astype(np.uint16)
        t1 = eikonal_raster_gpu(raster, 0)
        with pytest.warns(RuntimeWarning, match="validated domain"):
            t2 = eikonal_raster_gpu(raster, 0, order=2)
        # identical up to chaotic sub-eps slack
        np.testing.assert_allclose(t2[t2 < FINITE_LIMIT],
                                   t1[t1 < FINITE_LIMIT], rtol=5e-6,
                                   atol=1e-3)

    def test_obstacle_field_guard_fires(self):
        """Dense obstacles shape the solution with diffraction shocks
        (the real-raster failure class): forbidden fraction > 2%
        triggers the guard even though the passable cost is uniform."""
        rng = np.random.default_rng(2)
        raster = np.full((300, 300), 10, dtype=np.uint16)
        for _ in range(40):
            r0, c0 = rng.integers(10, 260, 2)
            raster[r0:r0 + 15, c0:c0 + 15] = np.iinfo(np.uint16).max
        with pytest.warns(RuntimeWarning, match="validated domain"):
            eikonal_raster_gpu(raster, 0, order=2)

    def test_smooth_surface_guard_does_not_fire(self):
        import warnings as _w
        n = 201
        raster = np.full((n, n), 10, dtype=np.uint16)
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            t2 = eikonal_raster_gpu(raster, (n // 2) * n + n // 2,
                                    order=2)
        assert t2[0, 0] < FINITE_LIMIT

    def test_invalid_order_raises(self):
        raster = np.ones((50, 50), dtype=np.uint16)
        with pytest.raises(ValueError, match="order"):
            eikonal_raster_gpu(raster, 0, order=3)

    def test_multi_source_order2(self):
        raster = np.full((150, 150), 7, dtype=np.uint16)
        sources = np.array([idx(10, 10, 150), idx(140, 140, 150)])
        t = eikonal_raster_gpu(raster, sources, order=2)
        assert float(t[10, 10]) == 0.0
        assert float(t[140, 140]) == 0.0
        assert (t < FINITE_LIMIT).all()

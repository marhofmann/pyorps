"""Tests for the Tier A anisotropic (3D-length) GPU eikonal solver.

CALIBRATION CONTRACT — read before trusting a green run.

The discrete slope-aware kernels are the ground truth only in the limit
of an INFINITELY WIDE neighborhood. Metrication error does not vanish
under grid refinement at fixed R; it is a fixed directional bias, which
is the entire reason this backend exists:

    lim_{R -> inf} discrete_R  =  continuum  =  FIM + O(h)
    lim_{h -> 0}   discrete_R  =  F_R  !=  continuum

So validation here is DIRECTIONAL (``T_FIM <= discrete_R + tol``) with
analytic truth as the referee wherever a closed form exists. Never copy a
discrete expectation into an assertion — a slope-aware discrete cost
*looks* authoritative and is not.

Analytic cases used:
  * constant-slope plane — M is constant, so geodesics are straight and
    ``T(x) = c * sqrt(|dx|^2 + (q.dx)^2)`` exactly (the 3D Euclidean
    length on the surface).
  * corrugated ramp ``z = f(col)`` — a cylinder, hence isometric to the
    plane by unrolling: with ``s(x) = int_0^x sqrt(1 + f'^2)``,
    ``T = c * sqrt((s(x)-s0)^2 + (y-y0)^2)`` exactly and geodesics are
    straight in ``(s, y)``, i.e. CURVED in the raster frame. This is the
    case that separates a correct tracer from a plausible one.
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

from pyorps.utils.eikonal_gpu import (   # noqa: E402
    FINITE_LIMIT,
    Q_ACUTE_LIMIT,
    metric_from_dem,
    trace_paths,
)

if GPU:
    from pyorps.utils.eikonal_gpu import (   # noqa: E402
        eikonal_raster_gpu,
        eikonal_raster_gpu_naive,
        trace_paths_gpu,
    )

needs_gpu = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

CELL = 10.0                      # metres per cell


# ---------------------------------------------------------------------------
# analytic fixtures
# ---------------------------------------------------------------------------

def plane_dem(rows, cols, grade, azimuth_deg=45.0, cell=CELL):
    """z of a constant-slope plane with |grad z| = grade."""
    a = np.deg2rad(azimuth_deg)
    qr, qc = grade * np.cos(a), grade * np.sin(a)
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dem = (qr * rr * cell + qc * cc * cell).astype(np.float32)
    return dem, np.array([qr, qc])


def plane_exact(rows, cols, sr, sc, q, c=1.0):
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dr = (rr - sr).astype(np.float64)
    dc = (cc - sc).astype(np.float64)
    rise = q[0] * dr + q[1] * dc
    return c * np.sqrt(dr * dr + dc * dc + rise * rise)


def ramp(cols, peak_grade, period=60.0, cell=CELL):
    """z = A sin(k x) as a function of the COLUMN only, plus its
    unrolled arc length in cell units."""
    k = 2.0 * np.pi / period
    amp_cells = peak_grade / (k / cell)          # so max |f'| = peak_grade
    x = np.arange(cols)
    f = amp_cells * cell * np.sin(k * x) / cell
    f = amp_cells * np.sin(k * x)                # metres
    fp = amp_cells * k * np.cos(k * x) / cell    # rise per metre of run
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(1.0 + fp[:-1] ** 2))])
    return f.astype(np.float32), fp, s


# ---------------------------------------------------------------------------
# metric assembly (no GPU needed)
# ---------------------------------------------------------------------------

class TestMetricAssembly:
    def test_plane_gradient_is_exact(self):
        for grade in (0.1, 0.5, 1.0, 2.0):
            dem, q = plane_dem(40, 40, grade)
            q_r, q_c, bad = metric_from_dem(dem, CELL)
            assert not bad.any()
            core = (slice(1, -1), slice(1, -1))
            assert np.allclose(q_r[core], q[0], atol=1e-5)
            assert np.allclose(q_c[core], q[1], atol=1e-5)

    def test_cell_size_enters_as_rise_per_metre(self):
        """The unit tripwire: doubling the cell size HALVES the gradient
        for the same elevation array."""
        dem, _ = plane_dem(30, 30, 1.0, cell=CELL)
        a_r, a_c, _ = metric_from_dem(dem, CELL)
        b_r, b_c, _ = metric_from_dem(dem, 2 * CELL)
        core = (slice(1, -1), slice(1, -1))
        assert np.allclose(b_r[core], a_r[core] / 2.0, atol=1e-6)
        assert np.allclose(b_c[core], a_c[core] / 2.0, atol=1e-6)

    def test_clamp_preserves_direction(self):
        dem, _ = plane_dem(30, 30, 5.0, azimuth_deg=30.0)
        q_r, q_c, _ = metric_from_dem(dem, CELL, q_clamp=2.0)
        core = (slice(1, -1), slice(1, -1))
        mag = np.hypot(q_r[core], q_c[core])
        assert np.allclose(mag, 2.0, atol=1e-5)
        ratio = q_c[core] / q_r[core]
        assert np.allclose(ratio, np.tan(np.deg2rad(30.0)), rtol=1e-4)

    def test_q_clamp_above_acuteness_threshold_raises(self):
        dem, _ = plane_dem(20, 20, 0.1)
        with pytest.raises(ValueError, match="acuteness"):
            metric_from_dem(dem, CELL, q_clamp=Q_ACUTE_LIMIT + 0.01)
        metric_from_dem(dem, CELL, q_clamp=Q_ACUTE_LIMIT - 1e-4)

    def test_nodata_cell_is_flagged_and_neighbours_degrade(self):
        dem = np.zeros((10, 10), dtype=np.float32)
        dem[5, 5] = np.nan
        q_r, q_c, bad = metric_from_dem(dem, CELL)
        assert bad[5, 5] and bad.sum() == 1
        assert np.isfinite(q_r).all() and np.isfinite(q_c).all()

    def test_horn_stencil_matches_central_on_a_plane(self):
        dem, q = plane_dem(30, 30, 0.8, azimuth_deg=20.0)
        q_r, q_c, _ = metric_from_dem(dem, CELL, slope_stencil="horn")
        core = (slice(1, -1), slice(1, -1))
        assert np.allclose(q_r[core], q[0], atol=1e-5)
        assert np.allclose(q_c[core], q[1], atol=1e-5)


class TestSimplexExactnessOnLinearFields:
    """Reproduces the stencil-error table in-repo (plan section 0 /
    appendix A.4) with a pure-numpy twin of the device update."""

    OFF = np.array([(1, 0), (1, 1), (0, 1), (-1, 1),
                    (-1, 0), (-1, -1), (0, -1), (1, -1)], dtype=float)

    @classmethod
    def update(cls, tn, c, q, pairs):
        off = cls.OFF
        len2 = (off ** 2).sum(1)
        qe = off @ q
        best = np.inf
        for k in range(8):
            if tn[k] < 1e29:
                best = min(best, tn[k] + c * np.sqrt(len2[k] + qe[k] ** 2))
        for i, j in pairs:
            t1, t2 = tn[i], tn[j]
            if t1 >= 1e29 or t2 >= 1e29:
                continue
            m11 = len2[i] + qe[i] ** 2
            m22 = len2[j] + qe[j] ** 2
            m12 = off[i] @ off[j] + qe[i] * qe[j]
            ss = m11 - 2 * m12 + m22
            det = m11 * m22 - m12 * m12
            d = t1 - t2
            disc = c * c * ss - d * d
            if disc < 0:
                continue
            t = ((m22 - m12) * t1 + (m11 - m12) * t2
                 + np.sqrt(det * disc)) / ss
            a, b = t - t1, t - t2
            if a < 0 or b < 0:
                continue
            if m22 * a - m12 * b < 0 or m11 * b - m12 * a < 0:
                continue
            best = min(best, t)
        return best

    @staticmethod
    def _pairs(n):
        if n == 8:
            return [(k, (k + 1) % 8) for k in range(8)]
        return [(0, 2), (2, 4), (4, 6), (6, 0)]     # axis quadrants

    def worst(self, grade, n_simplex, azis=19, dirs=91):
        worst = 0.0
        for phi in np.linspace(0, np.pi / 2, azis):
            q = grade * np.array([np.cos(phi), np.sin(phi)])
            d_inv = np.linalg.inv(np.eye(2) + np.outer(q, q))
            for th in np.linspace(0, 2 * np.pi, dirs):
                u = np.array([np.cos(th), np.sin(th)])
                p = u / np.sqrt(u @ d_inv @ u)
                tn = self.OFF @ p
                got = self.update(tn, 1.0, q, self._pairs(n_simplex))
                worst = max(worst, abs(got) / max(np.abs(tn).max(), 1e-12))
        return worst

    @pytest.mark.parametrize("grade", [0.1, 0.5, 1.0, 2.0, 2.1973])
    def test_eight_simplex_is_exact_inside_the_acute_regime(self, grade):
        assert self.worst(grade, 8) < 1e-12

    @pytest.mark.parametrize("grade", [0.5, 1.0, 2.0])
    def test_four_simplex_is_biased(self, grade):
        """The stencil the parent plan proposed: metric-obtuse for every
        non-axis-aligned slope, hence a fixed directional over-estimate
        of the same order as the whole accuracy claim."""
        assert self.worst(grade, 4) > 1e-3

    def test_acuteness_threshold_matches_the_derived_value(self):
        first = None
        for k in np.arange(0.0, 4.0, 0.01):
            phis = np.linspace(0, np.pi / 2, 721)
            qr, qc = k * np.cos(phis), k * np.sin(phis)
            if np.any(np.abs(qr * qc)
                      > 1 + np.minimum(qr ** 2, qc ** 2) + 1e-12):
                first = k
                break
        assert first is not None
        assert abs(first - Q_ACUTE_LIMIT) <= 0.01 + 1e-9


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

#: Largest square side for which the block-FIM solve was measured
#: BIT-reproducible run to run. The solver is a chaotic-relaxation
#: (block Gauss-Seidel) scheme with an atomically maintained active
#: list, so the order in which tiles settle depends on how the blocks
#: happen to interleave. That order is stable while the grid is small
#: enough that the resident blocks cover it, and stops being stable once
#: it is not.
#:
#: MEASURED on this GPU (RTX PRO 500 Blackwell), uint16 costs in
#: [1, 60), single source. Two different comparisons, two different
#: boundaries — the tighter one is what the constant holds:
#:
#:   run to run, same call (12 repeats x 2 rasters per size)
#:       <= 192   identical        208, 256, 512   DIFFER (max |dT| 2.4e-3)
#:
#:   no-DEM vs flat-DEM, same tiling (12 seeds x {random, barrier})
#:       <= 160   24/24 identical
#:          176   23/24
#:          192   13/24
#:
#: The bit-identity assertions below are therefore sound ONLY inside
#: this regime, and :func:`assert_bitwise_regime` refuses to let a
#: future edit enlarge the raster without noticing. Above it, compare
#: with a tolerance (every other test in this file does) — the
#: nondeterminism is a scheduling artefact of the relaxation, not a
#: correctness bug, and it is bounded by the convergence epsilon
#: (largest disagreement seen anywhere above: 2.4e-3 on fields of
#: O(10^3)).
BITWISE_DETERMINISTIC_MAX_DIM = 160


def assert_bitwise_regime(shape):
    """Guard a bit-identity assertion against being run out of regime."""
    assert max(shape) <= BITWISE_DETERMINISTIC_MAX_DIM, (
        f"bit-identity of the block-FIM solve is only measured up to "
        f"{BITWISE_DETERMINISTIC_MAX_DIM} px per side; this raster is "
        f"{shape} and the assertion would be testing the block "
        f"scheduler, not the operator. Compare with a tolerance instead.")


@needs_gpu
class TestIsotropicRegression:
    def test_the_bitwise_regime_is_where_we_think_it_is(self):
        """The guard needs its own guard: if the solve stopped being
        reproducible at 192 the two assertions below would go flaky
        instead of failing honestly."""
        rng = np.random.default_rng(7)
        n = BITWISE_DETERMINISTIC_MAX_DIM
        raster = rng.integers(1, 60, size=(n, n)).astype(np.uint16)
        ref = eikonal_raster_gpu(raster, [0])
        for _ in range(4):
            other = eikonal_raster_gpu(raster, [0])
            assert np.array_equal(ref.view(np.uint32),
                                  other.view(np.uint32))

    def test_no_dem_takes_the_untouched_isotropic_path(self):
        """A1 — the regression bar, structurally.

        With ``dem=None`` the anisotropic work is not merely equivalent,
        it is not reachable: a different kernel, with the same tuning
        defaults as before this increment. Bit-identity of the isotropic
        output is therefore a property of the code path, and the whole of
        ``test_eikonal_gpu.py`` (with its exact analytic expectations) is
        the regression suite for it."""
        from pyorps.utils import eikonal_gpu as E
        assert E.TILE_DEFAULT_ISO == 16
        assert E._default_n_inner(16, aniso=False) == 32
        # the isotropic sweep source contains no anisotropic machinery
        assert "aniso" not in E._FIM_SWEEP_KERNEL
        assert "q_r" not in E._FIM_SWEEP_KERNEL
        rng = np.random.default_rng(7)
        raster = rng.integers(1, 60, size=(96, 128)).astype(np.uint16)
        raster[40:60, 30] = 65535
        # bit-identity is only claimed inside the measured regime
        assert_bitwise_regime(raster.shape)
        a = eikonal_raster_gpu(raster, [0])
        b = eikonal_raster_gpu(raster, [0])
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))

    def test_flat_dem_reproduces_the_isotropic_field_exactly(self):
        """A flat DEM leaves every cell on the isotropic fast path, so at
        the SAME tiling the field must be BIT-identical to the no-DEM
        solve — the fast path really is the old operator, not a
        near-miss. (The anisotropic default tiling differs, so the
        comparison pins it.)

        SCOPE — what was actually measured, since "bit-identical"
        invites over-reading:

        * a UNIFORM elevation (exactly constant). A DEM that is merely
          flat-ish puts cells on the anisotropic branch, where the
          8-simplex operator is a genuinely different discretisation —
          see ``test_flat_dem_forced_anisotropic_agrees_closely``,
          which can only claim 2 %.
        * a raster inside ``BITWISE_DETERMINISTIC_MAX_DIM``. On a
          UNIFORM COST raster the identity held at every size tried (up
          to 256^2); on random / barrier cost rasters it held 24/24 at
          <= 160 px, 23/24 at 176 and only 13/24 at 192, because the
          block scheduler stops being reproducible there. So the claim
          is: same operator, same schedule — not "the anisotropic
          solver reproduces the isotropic one at any size"."""
        rng = np.random.default_rng(11)
        raster = rng.integers(1, 60, size=(96, 128)).astype(np.uint16)
        assert_bitwise_regime(raster.shape)
        dem = np.full(raster.shape, 231.0, dtype=np.float32)
        iso = eikonal_raster_gpu(raster, [0], tile=16, n_inner=32)
        flat = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL,
                                  tile=16, n_inner=32)
        assert np.array_equal(iso.view(np.uint32), flat.view(np.uint32))

    def test_flat_dem_forced_anisotropic_agrees_closely(self):
        """A2 — with q_flat_eps=0 every cell takes the 8-simplex path
        even though q == 0. The two schemes are genuinely DIFFERENT
        operators (the 8-simplex one is not the 4-point Godunov at q=0),
        so this is a discretisation-level agreement, not an identity —
        and both must bracket the analytic cone."""
        raster = np.ones((81, 81), dtype=np.uint16)
        dem = np.zeros(raster.shape, dtype=np.float32)
        iso = eikonal_raster_gpu(raster, [40 * 81 + 40])
        forced = eikonal_raster_gpu(raster, [40 * 81 + 40], dem=dem,
                                    cell_size=CELL, q_flat_eps=0.0)
        rr, cc = np.meshgrid(np.arange(81), np.arange(81), indexing="ij")
        exact = np.hypot(rr - 40, cc - 40)
        far = exact > 8
        assert np.abs(forced - iso)[far].max() / exact[far].max() < 0.02
        # both are honest discretisations of the same continuum answer
        for field in (iso, forced):
            rel = np.abs(field - exact)[far] / exact[far]
            assert rel.max() < 0.05


@needs_gpu
class TestAnalyticPlane:
    @pytest.mark.parametrize("grade", [0.1, 0.5, 1.0, 2.0])
    def test_constant_slope_plane_matches_3d_length(self, grade):
        n = 121
        raster = np.ones((n, n), dtype=np.uint16)
        dem, q = plane_dem(n, n, grade)
        t = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL)
        exact = plane_exact(n, n, 0, 0, q)
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        far = np.hypot(rr, cc) > 10
        rel = np.abs(t - exact)[far] / exact[far]
        assert rel.mean() < 0.01
        assert rel.max() < 0.04

    def test_downslope_direction_is_exact(self):
        """Along the plane's own steepest line the axis/diagonal edge
        candidate represents the geodesic exactly, so the only error left
        is the source singularity."""
        n = 121
        raster = np.ones((n, n), dtype=np.uint16)
        dem, q = plane_dem(n, n, 1.0, azimuth_deg=45.0)
        t = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL)
        exact = plane_exact(n, n, 0, 0, q)
        k = n - 1
        assert t[k, k] == pytest.approx(exact[k, k], rel=1e-4)

    def test_stretch_is_the_default_gradient_option(self):
        """Tier A must price EXACTLY sqrt(1 + (s/100)^2) and nothing
        else: the axis traverse of a plane sloping along that axis."""
        n = 101
        raster = np.ones((n, n), dtype=np.uint16)
        for grade in (0.25, 1.0, 2.0):
            rr, _ = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            dem = (grade * rr * CELL).astype(np.float32)
            t = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL)
            expect = (n - 1) * np.sqrt(1.0 + grade ** 2)
            assert t[n - 1, 0] == pytest.approx(expect, rel=2e-4)


@needs_gpu
class TestCorrugatedRamp:
    """Non-constant metric with a closed-form field AND a closed-form
    geodesic — the primary anisotropic analytic case."""

    @pytest.mark.parametrize("peak", [0.1, 0.5, 1.0, 2.0])
    def test_field_matches_the_unrolled_cylinder(self, peak):
        n = 161
        raster = np.ones((n, n), dtype=np.uint16)
        f, fp, s = ramp(n, peak)
        dem = np.tile(f, (n, 1)).astype(np.float32)
        sr, sc = n // 2, 10
        t = eikonal_raster_gpu(raster, [sr * n + sc], dem=dem,
                               cell_size=CELL)
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        exact = np.sqrt((s[cc] - s[sc]) ** 2 + (rr - sr) ** 2)
        far = np.hypot(rr - sr, cc - sc) > 8
        rel = np.abs(t - exact)[far] / exact[far]
        assert rel.mean() < 0.01
        assert rel.max() < 0.05

    def test_grid_convergence_order(self):
        """First-order scheme on a fixed continuous terrain: refining the
        grid must reduce the error at observed order >= 0.8."""
        errs = []
        for n in (81, 161, 321):
            scale = (n - 1) / 80.0
            cell = CELL / scale
            raster = np.ones((n, n), dtype=np.uint16)
            f, fp, s = ramp(n, 1.0, period=60.0 * scale, cell=cell)
            dem = np.tile(f, (n, 1)).astype(np.float32)
            sr, sc = n // 2, int(10 * scale)
            t = eikonal_raster_gpu(raster, [sr * n + sc], dem=dem,
                                   cell_size=cell,
                                   disk_radius=3.0 * scale)
            rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            exact = np.sqrt((s[cc] - s[sc]) ** 2 + (rr - sr) ** 2)
            far = np.hypot(rr - sr, cc - sc) > 10 * scale
            errs.append(float(np.mean(np.abs(t - exact)[far]
                                      / exact[far])))
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        assert min(orders) >= 0.8, (errs, orders)


@needs_gpu
class TestAgainstTheNaiveOracle:
    """The tiled kernel has no published convergence proof; the naive
    full-grid Jacobi iteration is trivially correct and is the guard."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_metric_field(self, seed):
        rng = np.random.default_rng(seed)
        n = 61
        raster = rng.integers(1, 40, size=(n, n)).astype(np.uint16)
        dem = (rng.normal(size=(n, n)).cumsum(0).cumsum(1) * 0.5
               ).astype(np.float32)
        q_r, q_c, _ = metric_from_dem(dem, CELL)
        assert np.hypot(q_r, q_c).max() <= 2.0 + 1e-6
        a = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL)
        b = eikonal_raster_gpu_naive(raster, [0], dem=dem, cell_size=CELL)
        fin = (a < FINITE_LIMIT) & (b < FINITE_LIMIT)
        scale = max(float(b[fin].max()), 1e-9)
        assert np.abs(a[fin] - b[fin]).max() / scale < 1e-5

    def test_with_barriers(self):
        n = 61
        raster = np.full((n, n), 3, dtype=np.uint16)
        raster[10:50, 20] = 65535
        raster[10:50, 40] = 65535
        rng = np.random.default_rng(5)
        dem = (rng.normal(size=(n, n)).cumsum(1) * 2.0).astype(np.float32)
        a = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL)
        b = eikonal_raster_gpu_naive(raster, [0], dem=dem, cell_size=CELL)
        fin = (a < FINITE_LIMIT) & (b < FINITE_LIMIT)
        assert np.array_equal(a >= FINITE_LIMIT, b >= FINITE_LIMIT)
        assert np.abs(a[fin] - b[fin]).max() / max(
            float(b[fin].max()), 1e-9) < 1e-5


@needs_gpu
class TestCornerHalo:
    def test_route_across_a_tile_corner(self):
        """R2 regression: the isotropic sweep loads EDGE halo only, so
        the four corner slots of the shared tile were never written. The
        8-simplex stencil reads them. A raster whose only viable route
        runs diagonally across tile corners would otherwise pick up stale
        values from the previous tile of the same block — sometimes small
        (silently under-priced)."""
        tile = 16
        n = 4 * tile
        raster = np.full((n, n), 65535, dtype=np.uint16)
        for i in range(n):                       # a 1-cell diagonal
            raster[i, i] = 1
            if i + 1 < n:
                raster[i, i + 1] = 1
        dem = np.tile((np.arange(n) * 2.0), (n, 1)).astype(np.float32)
        t = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL,
                               tile=tile)
        naive = eikonal_raster_gpu_naive(raster, [0], dem=dem,
                                         cell_size=CELL)
        end = (n - 1) * n + (n - 1)
        assert t.ravel()[end] < FINITE_LIMIT
        assert t.ravel()[end] == pytest.approx(
            naive.ravel()[end], rel=1e-5)

    def test_diagonal_only_costs_more_than_the_corner_shortcut(self):
        """A cheap corner cell must actually be usable — proves the
        corner slot carries a real value, not just a large one."""
        tile = 8
        n = 2 * tile
        raster = np.full((n, n), 200, dtype=np.uint16)
        raster[tile - 1, tile - 1] = 1
        raster[tile, tile] = 1
        dem = np.zeros((n, n), dtype=np.float32)
        t_hi = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL,
                                  q_flat_eps=0.0, tile=tile)
        naive = eikonal_raster_gpu_naive(raster, [0], dem=dem,
                                         cell_size=CELL, q_flat_eps=0.0)
        assert np.abs(t_hi - naive).max() / naive.max() < 1e-5


@needs_gpu
class TestSolverGuards:
    def test_order_2_with_dem_raises(self):
        raster = np.ones((32, 32), dtype=np.uint16)
        dem = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="order=2"):
            eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL,
                               order=2)

    def test_dem_without_cell_size_raises(self):
        raster = np.ones((32, 32), dtype=np.uint16)
        dem = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="cell_size"):
            eikonal_raster_gpu(raster, [0], dem=dem)

    def test_shape_mismatch_raises(self):
        raster = np.ones((32, 32), dtype=np.uint16)
        with pytest.raises(ValueError, match="does not match"):
            eikonal_raster_gpu(raster, [0],
                               dem=np.zeros((16, 16), dtype=np.float32),
                               cell_size=CELL)

    def test_forbidden_indices_block_the_route(self):
        n = 41
        raster = np.ones((n, n), dtype=np.uint16)
        dem = np.zeros((n, n), dtype=np.float32)
        wall = [r * n + n // 2 for r in range(n)]
        t = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL,
                               forbidden_indices=wall)
        assert t[0, n - 1] >= FINITE_LIMIT
        assert t[0, n // 2 - 1] < FINITE_LIMIT

    def test_dem_nodata_cells_become_impassable(self):
        n = 41
        raster = np.ones((n, n), dtype=np.uint16)
        dem = np.zeros((n, n), dtype=np.float32)
        dem[:, n // 2] = np.nan
        t = eikonal_raster_gpu(raster, [0], dem=dem, cell_size=CELL)
        assert t[0, n - 1] >= FINITE_LIMIT

    def test_stats_report_the_anisotropic_flag(self):
        raster = np.ones((40, 40), dtype=np.uint16)
        dem, _ = plane_dem(40, 40, 0.5)
        _t, stats = eikonal_raster_gpu(raster, [0], dem=dem,
                                       cell_size=CELL, return_stats=True)
        assert stats["anisotropic"] is True
        assert stats["n_nodata"] == 0


# ---------------------------------------------------------------------------
# tracer
# ---------------------------------------------------------------------------

@needs_gpu
class TestAnisotropicTracer:
    """Getting the tracer wrong produces a CORRECT T field and wrong
    polylines — a silent failure. Every case here therefore ships with
    the control below."""

    @staticmethod
    def _ramp_case(peak=1.0, n=161):
        raster = np.ones((n, n), dtype=np.uint16)
        f, fp, s = ramp(n, peak)
        dem = np.tile(f, (n, 1)).astype(np.float32)
        src = 20 * n + 10
        tgt = 140 * n + 150
        t, d_t, (qr, qc) = eikonal_raster_gpu(
            raster, [src], dem=dem, cell_size=CELL,
            return_device=True, return_metric=True)
        return raster, dem, s, src, tgt, t, d_t, (qr, qc), n

    @staticmethod
    def _unrolled_deviation(poly, s, src, tgt, n):
        ps = np.interp(poly[:, 1], np.arange(n), s)
        py = poly[:, 0]
        s0, y0 = s[src % n], float(src // n)
        s1, y1 = s[tgt % n], float(tgt // n)
        num = np.abs((s1 - s0) * (y0 - py) - (s0 - ps) * (y1 - y0))
        return float((num / np.hypot(s1 - s0, y1 - y0)).max())

    def test_geodesic_is_straight_in_the_unrolled_frame(self):
        _r, _d, s, src, tgt, t, _dt, q, n = self._ramp_case()
        q_h = (q[0].get().reshape(n, n), q[1].get().reshape(n, n))
        poly = trace_paths(t, [src], [tgt], q_fields=q_h)[0]
        assert self._unrolled_deviation(poly, s, src, tgt, n) < 1.5

    def test_the_isotropic_tracer_FAILS_the_same_case(self):
        """THE CONTROL (mandatory). A tracer test the old code also
        passes proves nothing."""
        _r, _d, s, src, tgt, t, _dt, _q, n = self._ramp_case()
        poly = trace_paths(t, [src], [tgt], q_fields=None)[0]
        dev = self._unrolled_deviation(poly, s, src, tgt, n)
        assert dev > 5.0, (
            f"the isotropic -grad T tracer deviated only {dev:.2f} cells "
            f"— the case no longer separates the two tracers")

    def test_device_tracer_matches_the_host_tracer(self):
        _r, _d, s, src, tgt, t, d_t, q, n = self._ramp_case()
        q_h = (q[0].get().reshape(n, n), q[1].get().reshape(n, n))
        gpu = trace_paths_gpu(t, [src], [tgt], t_device=d_t,
                              q_device=q)[0]
        host = trace_paths(t, [src], [tgt], q_fields=q_h)[0]
        k = min(len(gpu), len(host))
        assert np.abs(gpu[:k] - host[:k]).max() < 1e-3

    def test_integrated_cost_matches_the_field_value(self):
        """The general-purpose bias detector: works where no analytic
        answer exists, so this is the check to run on a real DEM."""
        rng = np.random.default_rng(3)
        n = 121
        raster = np.ones((n, n), dtype=np.uint16)
        dem = np.zeros((n, n), dtype=np.float32)
        for _ in range(6):                       # smooth random terrain
            r0, c0 = rng.integers(0, n, 2)
            rr, cc = np.meshgrid(np.arange(n), np.arange(n),
                                 indexing="ij")
            dem += (40.0 * rng.normal()
                    * np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2)
                             / (2 * 25.0 ** 2))).astype(np.float32)
        src, tgt = 5 * n + 5, 115 * n + 115
        t, d_t, q = eikonal_raster_gpu(
            raster, [src], dem=dem, cell_size=CELL,
            return_device=True, return_metric=True)
        q_h = (q[0].get().reshape(n, n), q[1].get().reshape(n, n))
        poly = trace_paths(t, [src], [tgt], q_fields=q_h)[0]
        seg = poly[1:] - poly[:-1]
        mid = 0.5 * (poly[1:] + poly[:-1])
        ri = np.clip(np.round(mid[:, 0]).astype(int), 0, n - 1)
        ci = np.clip(np.round(mid[:, 1]).astype(int), 0, n - 1)
        rise = q_h[0][ri, ci] * seg[:, 0] + q_h[1][ri, ci] * seg[:, 1]
        length = np.sqrt((seg ** 2).sum(1) + rise ** 2)
        integrated = float(length.sum())
        assert integrated == pytest.approx(float(t.ravel()[tgt]), rel=0.02)

    def test_flat_dem_traces_identically_to_the_isotropic_tracer(self):
        n = 81
        raster = np.ones((n, n), dtype=np.uint16)
        dem = np.zeros((n, n), dtype=np.float32)
        src, tgt = 0, (n - 1) * n + (n - 1)
        t, d_t, q = eikonal_raster_gpu(
            raster, [src], dem=dem, cell_size=CELL,
            return_device=True, return_metric=True)
        a = trace_paths_gpu(t, [src], [tgt], t_device=d_t, q_device=q)[0]
        b = trace_paths_gpu(t, [src], [tgt], t_device=d_t)[0]
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 1e-9


@needs_gpu
class TestRouteChangesWithSlope:
    def test_3d_stretch_moves_the_route_onto_the_flat_ground(self):
        """Direction-of-effect: the 2D-optimal route runs straight across
        a broad corrugated band; the 3D route detours into the flatter
        ground below it, is LONGER in 2D and CHEAPER in 3D length.

        Note the geometry has to earn the detour. The steepness clamp
        caps the crossing penalty at sqrt(1 + q_clamp^2) = 2.236x, so a
        narrow ridge is always cheaper to climb than to walk around —
        which is why this uses a wide band, not a single ridge."""
        n = 121
        raster = np.ones((n, n), dtype=np.uint16)
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        k = 2.0 * np.pi / 20.0
        amp = 2.0 / (k / CELL)              # peak grade 2.0
        band = np.where((cc >= 20) & (cc <= 100),
                        amp * np.sin(k * cc), 0.0)
        taper = np.clip((40.0 - rr) / 28.0, 0.0, 1.0)
        dem = (taper * band).astype(np.float32)
        src, tgt = 10 * n + 10, 10 * n + 110

        poly_flat = trace_paths(
            eikonal_raster_gpu(raster, [src]), [src], [tgt])[0]
        t, _d, q = eikonal_raster_gpu(
            raster, [src], dem=dem, cell_size=CELL,
            return_device=True, return_metric=True)
        q_h = (q[0].get().reshape(n, n), q[1].get().reshape(n, n))
        poly_dem = trace_paths(t, [src], [tgt], q_fields=q_h)[0]

        def length2d(poly):
            seg = poly[1:] - poly[:-1]
            return float(np.hypot(seg[:, 0], seg[:, 1]).sum())

        def cost3d(poly):
            seg = poly[1:] - poly[:-1]
            mid = 0.5 * (poly[1:] + poly[:-1])
            ri = np.clip(np.round(mid[:, 0]).astype(int), 0, n - 1)
            ci = np.clip(np.round(mid[:, 1]).astype(int), 0, n - 1)
            rise = q_h[0][ri, ci] * seg[:, 0] + q_h[1][ri, ci] * seg[:, 1]
            return float(np.sqrt((seg ** 2).sum(1) + rise ** 2).sum())

        assert poly_flat[:, 0].max() < 11.0          # 2D route: straight
        assert poly_dem[:, 0].max() > 25.0           # 3D route: detours
        assert length2d(poly_dem) > length2d(poly_flat) * 1.05
        assert cost3d(poly_dem) < cost3d(poly_flat) * 0.95
        # the detour is what the field priced, not tracer drift
        assert cost3d(poly_dem) == pytest.approx(
            float(t.ravel()[tgt]), rel=0.03)

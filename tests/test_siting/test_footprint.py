"""Footprint screening, and the two things a screen must not do.

A screen narrows the field; it does not decide. So the tests here pin
the properties a caller relies on -- the kernel's area is the same at
every rotation, a blocked cell is never reported feasible, and a
placement that hangs off the edge of the grid is UNKNOWN rather than
free -- and leave the winner to :func:`verify_exact`.
"""

import math

import numpy as np
import pytest

from pyorps.siting import (
    Footprint,
    rotated_kernel,
    screen_footprints,
    verify_exact,
)

pytest.importorskip("scipy")


@pytest.fixture
def affine():
    from affine import Affine
    return Affine(5.0, 0.0, 400000.0, 0.0, -5.0, 5500000.0)


class TestRotatedKernel:
    @pytest.mark.parametrize("theta", [0.0, 15.0, 37.5, 90.0, 143.0])
    def test_area_is_renormalised(self, theta):
        """The integer rasterisation swings ~4 %; renormalising removes it."""
        k = rotated_kernel(theta, 40.0, 20.0, 5.0)
        assert k.sum() == pytest.approx(40.0 * 20.0 / 25.0, rel=1e-5)

    def test_unrotated_kernel_covers_the_right_cells(self):
        """An odd cell count lands on cell boundaries, so coverage is
        exactly whole cells and the extent is unambiguous."""
        k = rotated_kernel(0.0, 45.0, 25.0, 5.0, supersample=8)
        rows, cols = np.nonzero(k > 0.5)
        assert np.ptp(rows) + 1 == 9       # 45 m / 5 m
        assert np.ptp(cols) + 1 == 5       # 25 m / 5 m

    def test_square_is_rotation_invariant_in_area(self):
        areas = [rotated_kernel(t, 30.0, 30.0, 5.0).sum()
                 for t in (0.0, 22.5, 45.0)]
        assert max(areas) - min(areas) < 1e-3

    def test_supersample_one_is_the_integer_pixel_set(self):
        """One sample per cell means a cell is either in or out."""
        k = rotated_kernel(0.0, 20.0, 10.0, 5.0, supersample=1)
        assert np.unique(k[k > 0]).size == 1
        assert k.sum() == pytest.approx(20.0 * 10.0 / 25.0, rel=1e-5)


class TestScreen:
    def _layers(self, n=90):
        rng = np.random.default_rng(4)
        cost = (rng.random((n, n)).astype(np.float32) * 50 + 10)
        blocked = np.zeros((n, n), np.float32)
        blocked[50:65, 20:40] = 1.0
        return cost, blocked

    def test_blocked_area_is_infeasible(self, affine):
        cost, blocked = self._layers()
        fp = Footprint(40.0, 20.0, rotation_step_deg=30.0)
        s = screen_footprints(cost, blocked, footprint=fp, transform=affine,
                              resolution_m=5.0)
        assert not s.feasible[55:60, 25:35].any()
        assert np.isnan(s.build_cost_eur[57, 30])

    def test_edges_are_unknown_not_free(self, affine):
        """The FFT pads with zeros, so an overhanging footprint reads
        as both cheap and unblocked. Those cells must not compete."""
        cost, blocked = self._layers()
        fp = Footprint(40.0, 20.0, rotation_step_deg=45.0)
        s = screen_footprints(cost, blocked, footprint=fp, transform=affine,
                              resolution_m=5.0)
        margin = int(math.ceil(math.hypot(4.0, 2.0))) + 1
        assert not s.feasible[:margin, :].any()
        assert not s.feasible[:, -margin:].any()
        assert s.feasible[margin:-margin, margin:-margin].any()

    def test_cost_tracks_the_underlying_layer(self, affine):
        """A uniform cost layer prices every feasible cell identically."""
        n = 60
        cost = np.full((n, n), 3.0, np.float32)
        blocked = np.zeros((n, n), np.float32)
        fp = Footprint(20.0, 20.0, rotation_step_deg=90.0)
        s = screen_footprints(cost, blocked, footprint=fp, transform=affine,
                              resolution_m=5.0)
        vals = s.build_cost_eur[s.feasible]
        assert vals.size
        assert np.allclose(vals, 3.0 * 400.0 / 25.0, rtol=1e-4)

    def test_best_theta_is_one_of_the_swept_angles(self, affine):
        cost, blocked = self._layers()
        fp = Footprint(40.0, 20.0, rotation_step_deg=30.0)
        s = screen_footprints(cost, blocked, footprint=fp, transform=affine,
                              resolution_m=5.0)
        assert set(np.unique(s.best_theta_deg[s.feasible])).issubset(
            set(fp.thetas.astype(np.float32)))

    def test_shapes_must_match(self, affine):
        with pytest.raises(ValueError, match="must match"):
            screen_footprints(np.ones((10, 10)), np.ones((10, 11)),
                              footprint=Footprint(10.0, 10.0),
                              resolution_m=1.0)


class TestVerifyExact:
    def test_coverage_mode_reproduces_the_screen(self, affine):
        """Same quantity, computed exactly instead of through an FFT."""
        n = 60
        rng = np.random.default_rng(9)
        cost = (rng.random((n, n)).astype(np.float32) * 40 + 5)
        blocked = np.zeros((n, n), np.float32)
        fp = Footprint(20.0, 20.0, rotation_step_deg=90.0)
        s = screen_footprints(cost, blocked, footprint=fp, transform=affine,
                              resolution_m=5.0)
        rows, cols = np.nonzero(s.feasible)
        pairs = np.column_stack([rows[:6], cols[:6]])
        got = verify_exact(cost, blocked, s, pairs, mode="coverage",
                           supersample=4)
        assert np.allclose(got, s.build_cost_eur[rows[:6], cols[:6]],
                           rtol=1e-4)

    def test_pixel_mode_is_a_different_quantity(self, affine):
        """It prices the cells you have to BUY, and it is larger.

        Measured on a uniform layer: 75 against 48 for a 20 x 20 m
        footprint at 5 m, because the rectangle covers 16 cells' worth
        of area but touches 25 cells.
        """
        n = 60
        cost = np.full((n, n), 3.0, np.float32)
        blocked = np.zeros((n, n), np.float32)
        fp = Footprint(20.0, 20.0, rotation_step_deg=90.0)
        s = screen_footprints(cost, blocked, footprint=fp, transform=affine,
                              resolution_m=5.0)
        rows, cols = np.nonzero(s.feasible)
        pairs = np.column_stack([rows[:4], cols[:4]])
        pixels = verify_exact(cost, blocked, s, pairs, mode="pixels")
        coverage = verify_exact(cost, blocked, s, pairs, mode="coverage")
        assert np.allclose(pixels, 75.0)
        assert np.allclose(coverage, 48.0, rtol=1e-4)
        assert np.all(pixels >= coverage)

    def test_unknown_mode_refused(self, affine):
        cost = np.full((30, 30), 1.0, np.float32)
        s = screen_footprints(cost, np.zeros((30, 30), np.float32),
                              footprint=Footprint(10.0, 10.0,
                                                  rotation_step_deg=90.0),
                              transform=affine, resolution_m=5.0)
        with pytest.raises(ValueError, match="pixels"):
            verify_exact(cost, np.zeros((30, 30)), s, [(15, 15)],
                         mode="fractional")

    def test_rejects_a_placement_that_leaves_the_grid(self, affine):
        cost = np.full((40, 40), 1.0)
        blocked = np.zeros((40, 40))
        fp = Footprint(40.0, 20.0, rotation_step_deg=90.0)
        s = screen_footprints(cost.astype(np.float32),
                              blocked.astype(np.float32), footprint=fp,
                              transform=affine, resolution_m=5.0)
        assert not np.isfinite(verify_exact(cost, blocked, s, [(0, 0)])[0])

    def test_rejects_an_overlap_the_screen_let_through(self, affine):
        """The exact pixel set is the arbiter, and it must be able to
        disagree with the supersampled one."""
        cost = np.full((40, 40), 1.0)
        blocked = np.zeros((40, 40))
        fp = Footprint(20.0, 20.0, rotation_step_deg=90.0)
        s = screen_footprints(cost.astype(np.float32),
                              blocked.astype(np.float32), footprint=fp,
                              transform=affine, resolution_m=5.0)
        blocked[20, 20] = 1.0            # introduced AFTER the screen
        assert not np.isfinite(verify_exact(cost, blocked, s, [(20, 20)])[0])


class TestFootprintValidation:
    def test_sides_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            Footprint(0.0, 10.0)

    def test_thetas_and_diagonal(self):
        fp = Footprint(40.0, 30.0, rotation_step_deg=45.0)
        assert list(fp.thetas) == [0.0, 45.0, 90.0, 135.0]
        assert fp.half_diagonal_m == pytest.approx(25.0)
        assert fp.area_m2 == 1200.0

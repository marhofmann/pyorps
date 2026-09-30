"""The exact site field (plan rev. 5, D3) against direct pixel sums.

Plan D3's acceptance test: for every anchor and rotation the FFT value
equals ``verify_exact(mode="pixels")`` on the same integer-cent raster, to
the cent, including non-integer EUR prices and the 1 200 EUR/m^2 forest
class. Both FFT methods (base-256 limbs, per-class indicators) must agree
with each other and with a direct pixel sum everywhere, borders and
forbidden pixels included.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from pyorps.siting.footprint import Footprint, ScreenResult, verify_exact
from pyorps.siting.site_field import (
    INFEASIBLE,
    footprint_mask,
    site_field,
    site_values,
)

CELL = 1.0


def _raster(rng, rows=41, cols=47, n_classes=6):
    classes = rng.integers(0, n_classes, size=(rows, cols))
    # EUR per pixel with cents, and a forest class at 1 200 EUR
    class_cents = rng.integers(0, 150_000, size=n_classes)
    class_cents[0] = 120_000
    class_cents[1] = 12_345
    forbidden = rng.random((rows, cols)) < 0.03
    return classes, class_cents, class_cents[classes], forbidden


class TestAgainstDirectSums:
    @pytest.mark.parametrize("seed", [1, 2])
    @pytest.mark.parametrize("method", ["limbs", "classes"])
    def test_every_anchor_and_rotation(self, seed, method):
        rng = np.random.default_rng(seed)
        classes, cc, price, forb = _raster(rng)
        thetas = [0.0, 15.0, 30.0, 45.0, 90.0, 135.0]
        L, W = 9.0, 5.0
        field = site_field(price, forb, length_m=L, width_m=W, cell_m=CELL,
                           thetas=thetas, method=method, classes=classes,
                           class_cents=cc, block=16, keep_rotations=True)
        rows, cols = price.shape
        anchors = np.array([(r, c) for r in range(rows) for c in range(cols)])
        for ri, t in enumerate(thetas):
            direct, _ = site_values(price, forb, anchors, t, length_m=L,
                                    width_m=W, cell_m=CELL)
            np.testing.assert_array_equal(
                field.per_rotation[ri].ravel(), direct)
        # the best rotation and its value
        stack = field.per_rotation.reshape(len(thetas), -1)
        np.testing.assert_array_equal(field.cents.ravel(), stack.min(0))
        feas = field.cents.ravel() != INFEASIBLE
        first = np.argmax(stack == stack.min(0), axis=0)
        np.testing.assert_array_equal(field.rotation.ravel()[feas],
                                      first[feas])
        assert np.all(field.rotation.ravel()[~feas] == -1)

    def test_verify_exact_agrees_to_the_cent(self):
        """Plan D3: D3.value(s, r) == verify_exact(pixels) with best_theta = r."""
        rng = np.random.default_rng(3)
        classes, cc, price, forb = _raster(rng, rows=50, cols=60)
        fp = Footprint(12.0, 7.0)
        thetas = fp.thetas[::3]
        field = site_field(price, forb, length_m=fp.length_m,
                           width_m=fp.width_m, cell_m=CELL, thetas=thetas,
                           keep_rotations=True, block=24)
        anchors = np.array([(r, c) for r in range(0, 50, 3)
                            for c in range(0, 60, 4)])
        for ri, t in enumerate(thetas):
            screen = ScreenResult(
                build_cost_eur=np.zeros(price.shape),
                best_theta_deg=np.full(price.shape, t, dtype=np.float32),
                feasible=np.ones(price.shape, dtype=bool), transform=None,
                resolution_m=CELL, footprint=fp)
            ref = verify_exact(price.astype(np.float64),
                               forb.astype(np.float64), screen, anchors)
            got = field.per_rotation[ri][anchors[:, 0], anchors[:, 1]]
            inf = ~np.isfinite(ref)
            assert np.all(got[inf] == INFEASIBLE)
            np.testing.assert_array_equal(got[~inf], ref[~inf].astype(np.int64))

    def test_class_counts_at_the_argmin(self):
        rng = np.random.default_rng(4)
        classes, cc, price, forb = _raster(rng)
        field = site_field(price, forb, length_m=8.0, width_m=6.0,
                           cell_m=CELL, thetas=[0.0, 45.0])
        r, c = np.argwhere(field.feasible)[0]
        t = field.thetas[field.rotation[r, c]]
        cents, counts = site_values(price, forb, [(r, c)], t, length_m=8.0,
                                    width_m=6.0, cell_m=CELL, classes=classes,
                                    n_classes=len(cc))
        assert cents[0] == field.cents[r, c]
        assert int(counts[0] @ cc) == int(field.cents[r, c])
        assert counts[0].sum() == footprint_mask(t, 8.0, 6.0, CELL).sum()


class TestExactness:
    def test_large_prices_stay_exact(self):
        """Prices near 2^31 cents: a single float64 FFT of the raw cents
        would drift; the base-256 limbs keep every value exact."""
        rng = np.random.default_rng(5)
        price = rng.integers(2**30, 2**31, size=(30, 30)).astype(np.int64)
        forb = np.zeros_like(price, dtype=bool)
        field = site_field(price, forb, length_m=10.0, width_m=10.0,
                           cell_m=CELL, thetas=[0.0, 30.0],
                           keep_rotations=True)
        anchors = np.argwhere(field.per_rotation[1] != INFEASIBLE)
        direct, _ = site_values(price, forb, anchors, 30.0, length_m=10.0,
                                width_m=10.0, cell_m=CELL)
        np.testing.assert_array_equal(
            field.per_rotation[1][anchors[:, 0], anchors[:, 1]], direct)

    def test_border_and_forbidden_are_infeasible(self):
        price = np.full((20, 20), 100, dtype=np.int64)
        forb = np.zeros((20, 20), dtype=bool)
        forb[10, 10] = True
        field = site_field(price, forb, length_m=4.0, width_m=4.0,
                           cell_m=CELL, thetas=[0.0])
        assert field.cents[0, 0] == INFEASIBLE          # leaves the grid
        assert field.cents[10, 10] == INFEASIBLE        # covers a forbidden
        assert field.cents[5, 5] == 100 * footprint_mask(
            0.0, 4.0, 4.0, CELL).sum()
        assert np.isinf(field.eur()[0, 0])

    def test_refusals(self):
        price = np.ones((5, 5), dtype=np.int64)
        forb = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="non-negative"):
            site_field(-price, forb, length_m=2, width_m=2, cell_m=1,
                       thetas=[0.0])
        with pytest.raises(ValueError, match="classes"):
            site_field(price, forb, length_m=2, width_m=2, cell_m=1,
                       thetas=[0.0], method="classes")

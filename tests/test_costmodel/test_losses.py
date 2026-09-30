"""Loss energy from joint power categories (plan rev. 5, section 2.1, C11).

Plan test 16: the category square sums ``W(X)`` against a direct
hour-by-hour sum on a small synthetic record. Also the properties the
engines rely on: the mean-only table never overstates, the covariance
correction makes it exact, per-turbine tables are what keep simultaneity
(check K2), and the annuity and price conventions are what the plan says.
"""
import itertools
from pathlib import Path

import numpy as np
import pytest

from pyorps.costmodel import (
    LossCategories,
    annuity_factor,
    category_square_sums,
    loss_coef,
    loss_price_eur_per_mwh,
)

TABLE = Path(__file__).resolve().parents[2] / "case_studies" / \
    "runkel_free_siting" / "data" / "loss_categories.npz"


def _record(rng, hours=500, n=4):
    """Correlated per-turbine output: a common wind factor plus wakes."""
    wind = rng.uniform(0.0, 1.0, size=hours) ** 1.5
    wake = rng.uniform(0.8, 1.0, size=(hours, n))
    return np.clip(6.8 * wind[:, None] * wake, 0.0, 6.8)


def _direct(P, Q=None):
    """W(X) by brute force over the hours and the subsets."""
    n = P.shape[1]
    Q = np.zeros_like(P) if Q is None else Q
    scale = 8760.0 / P.shape[0]
    out = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        idx = [t for t in range(n) if mask >> t & 1]
        s = P[:, idx].sum(1) + 1j * Q[:, idx].sum(1)
        out[mask] = scale * np.sum(np.abs(s) ** 2)
    return out


class TestSquareSums:
    def test_hourly_categories_equal_the_direct_sum(self):
        rng = np.random.default_rng(1)
        P = _record(rng)
        Q = 0.3 * P
        h = np.full(P.shape[0], 8760.0 / P.shape[0])
        np.testing.assert_allclose(category_square_sums(h, P, Q),
                                   _direct(P, Q), rtol=1e-12)

    def test_binned_means_never_overstate_and_covariance_repairs(self):
        rng = np.random.default_rng(2)
        P = _record(rng, hours=800)
        exact = _direct(P)
        bins = np.floor(P.mean(1) / 0.5).astype(int)      # a width setting
        cats = np.unique(bins)
        h = np.array([np.sum(bins == c) for c in cats]) * 8760.0 / len(P)
        means = np.array([P[bins == c].mean(0) for c in cats])
        w_cat = category_square_sums(h, means)
        assert np.all(w_cat[1:] <= exact[1:] * (1 + 1e-12))
        within = sum(h[i] * np.cov(P[bins == c].T, bias=True)
                     for i, c in enumerate(cats) if np.sum(bins == c) > 1)
        n = P.shape[1]
        member = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
        w_var = w_cat + np.einsum("si,ij,sj->s", member, within, member)
        np.testing.assert_allclose(w_var[1:], exact[1:], rtol=1e-10)

    def test_separate_turbine_histograms_undercount(self):
        """Check K2: the sum of per-turbine square sums misses the cross
        terms, i.e. the turbines peaking together."""
        rng = np.random.default_rng(3)
        P = _record(rng)
        w = _direct(P)
        full = (1 << P.shape[1]) - 1
        singles = sum(w[1 << t] for t in range(P.shape[1]))
        assert singles < 0.5 * w[full]


class TestValuation:
    def test_annuity_factor_headline(self):
        assert annuity_factor(25, 0.04) == pytest.approx(15.622080, abs=1e-6)
        assert annuity_factor(20, 0.04, growth=0.04) == 20.0
        assert annuity_factor(25, 0.04, growth=-0.02) < annuity_factor(25, 0.04)

    def test_loss_coef_units(self):
        """1 ohm/m, W = 1 MVA^2 h/a at 1 kV loses 1 MWh/a per metre."""
        c = loss_coef(price_eur_per_mwh=60.0, u_kv=1.0)
        assert c == pytest.approx(60.0 * annuity_factor(25, 0.04))
        assert loss_coef(price_eur_per_mwh=60.0, u_kv=20.0) == \
            pytest.approx(c / 400.0)

    def test_price_conventions(self):
        assert loss_price_eur_per_mwh(price=60.0, nu=0.08) == pytest.approx(55.2)
        assert loss_price_eur_per_mwh(price=67.37, nu=0.08,
                                      convention="clipped") == 67.37
        with pytest.raises(ValueError):
            loss_price_eur_per_mwh(price=60.0, nu=1.2)


@pytest.mark.skipif(not TABLE.exists(), reason="case-study table not present")
class TestRunkelTable:
    def test_table_is_consistent_and_exact(self):
        tab = LossCategories.load(TABLE)
        assert tab.n == 7
        assert tab.sha256 == ("c62b4266d1adb48804b889aa5c185d91"
                              "d8185d7bee054f38e47438369c220681")
        exact = np.array(tab.loss_weight("exact"))
        var = np.array(tab.loss_weight("var"))
        cat = np.array(tab.loss_weight("cat"))
        np.testing.assert_allclose(var[1:], exact[1:], rtol=1e-12)
        assert np.all(cat[1:] <= exact[1:])
        # theta of the coherent comparison: inside the plan's 1866-1956 h
        assert 1866.0 <= tab.theta_hours(7 * 6.8) <= 1956.0
        assert tab.meta["curve"] == "pywake"

    def test_w_min_by_size_is_monotone(self):
        tab = LossCategories.load(TABLE)
        w = np.array(tab.loss_weight("exact"))
        by_size = [min(w[m] for m in range(1, 128) if bin(m).count("1") == k)
                   for k in range(1, 8)]
        assert all(a < b for a, b in itertools.pairwise(by_size))

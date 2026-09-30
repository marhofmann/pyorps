"""The tree certificate and the ellipse window (plan section 3.5) against
the collector engine on the whole raster."""
import random

import numpy as np
import pytest

from pyorps.certify import certify_tree_field, ellipse_window
from pyorps.collector.bounds import mu_star
from pyorps.collector.raster import RasterCollector
from pyorps.utils.neighborhood import get_neighborhood_steps

from test_collector.test_engine_vs_oracle_v2 import _model


def _instance(seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    R, C = 9, 11
    vals = nrng.integers(1, 6, size=(R, C)).astype(np.uint16)
    if seed % 2:
        vals[nrng.random((R, C)) < 0.06] = 65535
    steps = np.asarray(get_neighborhood_steps(rng.choice([1, 2]),
                                              directed=True), dtype=np.int8)
    inside = np.zeros((R, C), dtype=bool)
    r0, c0 = int(nrng.integers(0, 3)), int(nrng.integers(0, 3))
    inside[r0:r0 + 6, c0:c0 + 7] = True
    free = np.flatnonzero((inside & (vals != 65535)).ravel())
    n = rng.choice([2, 3])
    turb = [int(x) for x in nrng.choice(free, size=n, replace=False)]
    model = _model(rng, n)
    return vals, steps, inside, turb, model, rng.choice([1.0, 2.0]), \
        rng.choice([1.0, 1.2])


@pytest.mark.parametrize("seed", range(6))
def test_tree_certificate_is_sound_and_exact_where_it_says(seed):
    """A 12 x 12 window with room around the turbines: the certificate
    must mark cells exact (9-30 % here), and be right about them."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    R = C = 16
    vals = nrng.integers(1, 6, size=(R, C)).astype(np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    inside = np.zeros((R, C), dtype=bool)
    inside[2:14, 2:14] = True
    turb = [7 * C + 7, 8 * C + 9, 9 * C + 6][:rng.choice([2, 3])]
    model = _model(rng, len(turb))
    full = RasterCollector(vals, steps, 1.0, turb, model, engine="B",
                           drain_engine="python").run().ravel()
    win = vals.copy()
    win[~inside] = 65535
    mv_w = RasterCollector(win, steps, 1.0, turb, model, engine="B",
                           drain_engine="python").run().ravel()
    mu_min = min(mu_star(model)[1:])
    if not np.isfinite(mu_min):
        pytest.skip("no feasible label in this draw")
    cert = certify_tree_field(vals, steps, inside, turb, mv_w,
                              mu_min=mu_min, engine="python")
    lower, exact = cert.lower.ravel(), cert.exact.ravel()
    fin = inside.ravel() & np.isfinite(full)
    assert np.all(lower[fin] <= full[fin] * (1 + 1e-9) + 1e-9)
    np.testing.assert_allclose(mv_w[exact], full[exact], rtol=1e-9)
    if seed in (0, 1, 3):
        assert cert.exact_fraction > 0.05


def test_tree_certificate_on_tight_windows_is_still_sound():
    for seed in range(8):
        vals, steps, inside, turb, model, cell, mult = _instance(seed)
        full = RasterCollector(vals, steps, cell, turb, model, engine="B",
                               trench_mult=mult,
                               drain_engine="python").run().ravel()
        win = vals.copy()
        win[~inside] = 65535
        mv_w = RasterCollector(win, steps, cell, turb, model, engine="B",
                               trench_mult=mult,
                               drain_engine="python").run().ravel()
        mu_min = min(mu_star(model)[1:])
        if not np.isfinite(mu_min):
            continue
        cert = certify_tree_field(vals, steps, inside, turb, mv_w,
                                  mu_min=mu_min, trench_mult=mult,
                                  cell_m=cell, engine="python")
        fin = inside.ravel() & np.isfinite(full)
        assert np.all(cert.lower.ravel()[fin]
                      <= full[fin] * (1 + 1e-9) + 1e-9)
        ex = cert.exact.ravel()
        np.testing.assert_allclose(mv_w[ex], full[ex], rtol=1e-9)


def test_tree_certificate_needs_every_turbine_inside():
    vals, steps, inside, turb, model, cell, mult = _instance(0)
    inside2 = inside.copy()
    inside2.ravel()[turb[0]] = False
    with pytest.raises(ValueError, match="every turbine"):
        certify_tree_field(vals, steps, inside2, turb,
                           np.zeros(vals.size), mu_min=1.0)


@pytest.mark.parametrize("seed", range(6))
def test_competitive_designs_stay_inside_the_ellipse(seed):
    vals, steps, _inside, turb, model, cell, mult = _instance(seed)
    rc = RasterCollector(vals, steps, cell, turb, model, engine="B",
                         trench_mult=mult, drain_engine="python",
                         keep_trace=True)
    mv = rc.run().ravel()
    fin = np.flatnonzero(np.isfinite(mv))
    if fin.size == 0:
        pytest.skip("infeasible draw")
    nrng = np.random.default_rng(100 + seed)
    budget = np.full(vals.size, np.nan)
    roots = nrng.choice(fin, size=min(6, fin.size), replace=False)
    budget[roots] = mv[roots] * nrng.uniform(1.0, 1.3, size=roots.size)
    mu_min = min(mu_star(model)[1:])
    W = ellipse_window(vals, steps, turb, budget, mu_min=mu_min,
                       trench_mult=mult, cell_m=cell,
                       engine="python").ravel()
    for g in roots:
        design = rc.design(int(g))
        assert all(W[c] for c in design.tnodes), (g, design.tnodes)
    # a tight budget at one root: the ellipse is smaller than the raster
    # and still holds that root's optimal design
    g = int(fin[np.argmin(mv[fin])])
    tight = np.full(vals.size, np.nan)
    tight[g] = mv[g]
    W1 = ellipse_window(vals, steps, turb, tight, mu_min=mu_min,
                        trench_mult=mult, cell_m=cell,
                        engine="python").ravel()
    assert all(W1[c] for c in rc.design(g).tnodes)
    assert W1.mean() < 1.0

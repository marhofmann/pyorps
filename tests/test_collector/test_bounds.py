"""The rho-hat DW Stage-2 bound (plan D4a) never exceeds engine B.

Measured 2026-09-24 (seed 21, 60 instances, 2-4 turbines, stations,
parallel cables, bay limits): 0 violations over 327 roots, mean gap 6.5 %,
in line with the appendix's 5.8 % on design A's instances.
"""
import math
import random

import numpy as np
import pytest

from pyorps.collector import CollectorGraph, solve_collector
from pyorps.collector.bounds import rho_hat, rho_hat_lower_bound

from .test_engine_vs_oracle_v2 import _grid, _model


@pytest.mark.parametrize("seed,count", [
    (21, 12), pytest.param(22, 60, marks=pytest.mark.slow)])
def test_bound_is_admissible_in_the_field(seed, count):
    rng = random.Random(seed)
    roots = 0
    for _ in range(count):
        R, C = rng.choice([(2, 3), (3, 3), (3, 4)])
        N, E = _grid(R, C, rng, rng.random() < 0.5)
        n = rng.choice([2, 3, 4])
        turb = rng.sample(range(N), n)
        model = _model(rng, n)
        G = CollectorGraph(N, E)
        mv = solve_collector(G, turb, model, engine="B").mv
        lb = rho_hat_lower_bound(G, turb, model)
        fin = np.isfinite(mv)
        assert np.all(lb[fin] <= mv[fin] + 1e-9 * np.maximum(1.0, mv[fin]))
        assert np.all(np.isinf(lb[list(turb)]))
        roots += int(fin.sum())
    assert roots > 20


def test_rho_hat_takes_the_cheapest_partition():
    """Two turbines that one cable cannot carry: rho_hat is the two-cable
    rate with sigma, and nb_min counts conductors."""
    rng = random.Random(5)
    model = _model(rng, 2)
    rho, nb = rho_hat(model)
    for S in (1, 2, 3):
        assert rho[S] <= min(
            model.rate([(S, opt)]) for opt in model.options)
    assert nb[3] >= 1
    assert rho[0] == 0.0 and not math.isnan(rho[3])


@pytest.mark.parametrize("seed,count", [
    (11, 5), pytest.param(12, 20, marks=pytest.mark.slow)])
def test_raster_bound_equals_the_graph_bound_and_is_admissible(seed, count):
    """Plan D10: the raster port (one drain per subset) equals the graph
    bound on the same window and never exceeds the raster engine B.
    Measured 2026-09-24: 418 roots, equal to 4e-16, 0 violations, mean gap
    2.9 %."""
    from pyorps.collector.bounds import rho_hat_lower_bound_raster
    from pyorps.collector.raster import RasterCollector, graph_from_raster
    from pyorps.utils.neighborhood import get_neighborhood_steps

    rng = random.Random(seed)
    roots = 0
    for it in range(count):
        R, C = rng.choice([(6, 7), (7, 8)])
        nrng = np.random.default_rng(100 * seed + it)
        vals = nrng.integers(1, 6, size=(R, C)).astype(np.uint16)
        if it % 2:
            vals[nrng.random((R, C)) < 0.08] = 65535
        steps = np.asarray(get_neighborhood_steps(rng.choice([1, 2]),
                                                  directed=True), dtype=np.int8)
        free = np.flatnonzero(vals.ravel() != 65535)
        n = rng.choice([2, 3])
        turb = [int(x) for x in nrng.choice(free, size=n, replace=False)]
        model = _model(rng, n)
        cell = rng.choice([1.0, 2.5])
        mult = rng.choice([1.0, 1.2])
        G = graph_from_raster(vals, steps, cell, turb, trench_mult=mult)
        lbg = np.where(vals.ravel() == 65535, np.inf,
                       rho_hat_lower_bound(G, turb, model))
        lbr = rho_hat_lower_bound_raster(vals, steps, cell, turb, model,
                                         trench_mult=mult,
                                         drain_engine="python").ravel()
        np.testing.assert_array_equal(np.isfinite(lbr), np.isfinite(lbg))
        fin = np.isfinite(lbg)
        np.testing.assert_allclose(lbr[fin], lbg[fin], rtol=1e-12)
        mv = RasterCollector(vals, steps, cell, turb, model, engine="B",
                             trench_mult=mult,
                             drain_engine="python").run().ravel()
        ok = np.isfinite(mv)
        assert np.all(lbr[ok] <= mv[ok] * (1 + 1e-9) + 1e-9)
        roots += int(ok.sum())
    assert roots > 50

"""Two engines, one exact answer (plan section 3.2): A's gap window, B
pruned against the budget, every root exact or certified above budget.

Measured 2026-09-24: 224 exact roots, 0 mismatches against the unpruned
engine B, the cheapest total found in 15 of 15 windows, gap window 34 % of
the cells on average.
"""
import random

import numpy as np
import pytest

from pyorps.collector.design import reprice
from pyorps.collector.exact import exact_collector_field
from pyorps.collector.raster import RasterCollector
from pyorps.collector.raster_pricer import RasterStepPricer
from pyorps.utils.neighborhood import get_neighborhood_steps

from .test_engine_vs_oracle_v2 import _model


@pytest.mark.parametrize("seed,count", [
    (41, 6), pytest.param(42, 24, marks=pytest.mark.slow)])
def test_every_root_is_exact_or_certified_above_budget(seed, count):
    rng = random.Random(seed)
    exact_roots = 0
    for it in range(count):
        R, C = rng.choice([(6, 7), (7, 8)])
        nrng = np.random.default_rng(1000 * seed + it)
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
        root_cost = (nrng.uniform(0, 5, size=vals.size) if it % 2 == 0
                     else np.zeros(vals.size))
        full = RasterCollector(vals, steps, cell, turb, model, engine="B",
                               trench_mult=mult,
                               drain_engine="python").run().ravel()
        total = full + root_cost
        fin = np.isfinite(total)
        if not fin.any():
            continue
        budget = float(np.quantile(total[fin], rng.choice([0.1, 0.4])))
        res = exact_collector_field(vals, steps, cell, turb, model,
                                    budget=budget, root_cost=root_cost,
                                    trench_mult=mult, drain_engine="python")
        want = fin & (total <= budget)
        ex = res.exact.ravel()
        np.testing.assert_array_equal(ex, want)
        np.testing.assert_allclose(res.mv.ravel()[ex], full[ex], rtol=1e-9)
        # every candidate root is either exact or certified above budget
        cand = (vals.ravel() != 65535)
        cand[turb] = False
        assert np.all(ex | res.above_budget.ravel() | ~cand)
        assert not np.any(ex & res.above_budget.ravel())
        g = res.argmin
        assert res.total.ravel()[g] == pytest.approx(total[fin].min())
        cost, _ = reprice(res.engine_b.design(g),
                          RasterStepPricer(vals, steps, cell, turb,
                                           trench_mult=mult), turb, model)
        assert cost == pytest.approx(res.mv.ravel()[g], rel=1e-9)
        exact_roots += int(ex.sum())
    assert exact_roots > 10


def test_negative_eps_is_refused():
    vals = np.full((3, 3), 2, dtype=np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    with pytest.raises(ValueError, match="safe sign"):
        exact_collector_field(vals, steps, 1.0, [0, 8],
                              _model(random.Random(1), 2), budget=10.0,
                              eps=-1.0, drain_engine="python")

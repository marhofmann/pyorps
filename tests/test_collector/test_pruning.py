"""Certificate pruning of the raster engine (plan section 3.2, item 6).

A label value is dropped where ``F + Out_|S| > budget``; ``Out`` is the
aggregated completion bound (appendix Part IV, design B) with merges at
turbine cells, which carries the transposed turbine rule (verifier V-04).

The pruning check: on random windows, both engines, with and without a
per-cell root cost, every root inside the budget keeps its exact value and
no root outside reads below its true value. Measured 2026-09-24: 464
in-budget roots, 0 mismatches, 72 % of label cells dropped.
"""
import math
import random

import numpy as np
import pytest

from pyorps.collector import CableType, CollectorModel
from pyorps.collector.raster import RasterCollector
from pyorps.utils.neighborhood import get_neighborhood_steps

from .test_engine_vs_oracle_v2 import _model


@pytest.mark.parametrize("seed,count", [
    (31, 6), pytest.param(32, 24, marks=pytest.mark.slow)])
def test_pruning_keeps_every_root_inside_the_budget(seed, count):
    rng = random.Random(seed)
    in_budget = pruned = 0
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
        root_cost = (nrng.uniform(0, 5, size=vals.size) if it % 3 == 0
                     else None)
        for engine in ("A", "B"):
            full = RasterCollector(vals, steps, cell, turb, model,
                                   engine=engine, trench_mult=mult,
                                   drain_engine="python").run().ravel()
            total = full + (0.0 if root_cost is None else root_cost)
            fin = np.isfinite(total)
            if not fin.any():
                continue
            budget = float(np.quantile(total[fin],
                                       rng.choice([0.05, 0.3, 0.7])))
            rc = RasterCollector(vals, steps, cell, turb, model,
                                 engine=engine, trench_mult=mult,
                                 drain_engine="python", budget=budget,
                                 root_cost=root_cost)
            got = rc.run().ravel()
            inside = fin & (total <= budget)
            np.testing.assert_allclose(got[inside], full[inside], rtol=1e-9)
            outside = fin & ~inside
            assert np.all(got[outside] >= full[outside] - 1e-9)
            in_budget += int(inside.sum())
            pruned += rc.stats["cells_pruned"]
    assert in_budget > 20 and pruned > 0


def _chain_model():
    return CollectorModel.identical_turbines(
        2, current_one_a=1.0, types=(CableType("a", 5.0, 1.0, 0.0),),
        loss_coef=1.0, derating=(1.0, 1.0, 1.0, 1.0), sigma_eur_per_m=0.0,
        m_max=4, p_max=1, switchgear_a=math.inf, ring_panels=3,
        turbine_panel_eur=0.0, station_building_eur=5.0,
        station_panel_eur=0.5, allow_stations=True, bay_eur=0.0,
        bay_a=math.inf)


def test_a_chain_through_a_turbine_survives_a_tight_budget():
    """Verifier V-04: t2 - x - t1 - y - g on a line. The only design is the
    chain t2 -> t1 -> g; a completion bound without the transposed turbine
    rule would read inf at x and cut it."""
    vals = np.full((1, 5), 3, dtype=np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    model = _chain_model()
    full = RasterCollector(vals, steps, 1.0, [0, 2], model, engine="B",
                           drain_engine="python").run().ravel()
    assert np.isfinite(full[4])
    got = RasterCollector(vals, steps, 1.0, [0, 2], model, engine="B",
                          drain_engine="python",
                          budget=float(full[4])).run().ravel()
    assert got[4] == pytest.approx(full[4])


def test_a_pruned_design_still_traces_and_reprices():
    from pyorps.collector.design import reprice
    from pyorps.collector.raster_pricer import RasterStepPricer

    rng = random.Random(5)
    vals = np.random.default_rng(5).integers(1, 6, size=(7, 8)) \
        .astype(np.uint16)
    steps = np.asarray(get_neighborhood_steps(2, directed=True),
                       dtype=np.int8)
    turb = [3, 20, 50]
    model = _model(rng, 3)
    full = RasterCollector(vals, steps, 1.0, turb, model, engine="B",
                           drain_engine="python").run().ravel()
    fin = np.isfinite(full)
    if not fin.any():
        pytest.skip("infeasible draw")
    budget = float(np.quantile(full[fin], 0.2))
    rc = RasterCollector(vals, steps, 1.0, turb, model, engine="B",
                         drain_engine="python", keep_trace=True,
                         budget=budget)
    got = rc.run().ravel()
    pricer = RasterStepPricer(vals, steps, 1.0, turb)
    for g in np.flatnonzero(fin & (full <= budget)):
        cost, _ = reprice(rc.design(int(g)), pricer, turb, model)
        assert cost == pytest.approx(got[g], rel=1e-9)


def test_negative_eps_is_refused():
    vals = np.full((1, 5), 3, dtype=np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    with pytest.raises(ValueError, match="safe sign"):
        RasterCollector(vals, steps, 1.0, [0, 2], _chain_model(),
                        engine="B", drain_engine="python", budget=10.0,
                        prune_eps=-1.0).run()

"""The raster-window collector engine equals the reference engine.

``RasterCollector`` runs the same recursion as ``solve_collector`` but in
the plan's raster layout (one seeded drain per growable label, labels in
potential order, station states expanded by one exact step). On the graph
:func:`graph_from_raster` builds from the same raster -- the kernel's step
weights, turbines no-transit, exclusions dropped -- both give the same
field. Measured 2026-09-24: 692 roots, engines A and B, 0 mismatches.
"""
import math
import random

import numpy as np
import pytest

from pyorps.collector import solve_collector
from pyorps.collector.raster import RasterCollector, graph_from_raster
from pyorps.utils.neighborhood import get_neighborhood_steps

from .test_engine_vs_oracle_v2 import _model

INF = math.inf


@pytest.mark.parametrize("seed,count", [
    (3, 4), pytest.param(4, 16, marks=pytest.mark.slow)])
def test_raster_engine_equals_the_reference(seed, count):
    rng = random.Random(seed)
    roots = 0
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
        G = graph_from_raster(vals, steps, cell, turb, trench_mult=mult)
        for engine in ("A", "B"):
            ref = solve_collector(G, turb, model, engine=engine).mv
            ref = np.where(vals.ravel() == 65535, INF, ref)
            ras = RasterCollector(vals, steps, cell, turb, model,
                                  engine=engine, trench_mult=mult,
                                  drain_engine="python").run().ravel()
            fin = np.isfinite(ref)
            np.testing.assert_array_equal(np.isfinite(ras), fin)
            np.testing.assert_allclose(ras[fin], ref[fin], rtol=1e-9)
            roots += int(fin.sum())
    assert roots > 50


def test_graph_from_raster_drops_steps_through_turbines_and_exclusions():
    vals = np.full((3, 3), 2, dtype=np.uint16)
    vals[0, 2] = 65535
    steps = np.asarray(get_neighborhood_steps(1, directed=True), dtype=np.int8)
    G = graph_from_raster(vals, steps, 1.0, [4])      # turbine at the centre
    pairs = {frozenset((u, w)) for u, w, _c, _l in G.edges}
    assert all(2 not in pair for pair in pairs)       # excluded cell
    # 0-4 is a real diagonal edge INTO the turbine (an arrival)
    assert frozenset((0, 4)) in pairs
    # 1-3 is a diagonal whose intermediate cells are 0 and 4 (the turbine)
    assert frozenset((1, 3)) not in pairs


@pytest.mark.parametrize("seed,count", [
    (7, 6), pytest.param(8, 24, marks=pytest.mark.slow)])
def test_traced_designs_reprice_to_the_field(seed, count):
    """Plan D7: every design traced out of the raster engine re-prices, on
    the kernel's own ``price_route`` and an independent length, to the
    engine's value at its root. Measured 2026-09-24: 534 roots, worst
    relative gap 4e-16, 42 with stations, 110 with parallel cables."""
    from pyorps.collector.design import reprice
    from pyorps.collector.raster_pricer import RasterStepPricer

    rng = random.Random(seed)
    roots = stations = parallel = 0
    for it in range(count):
        R, C = rng.choice([(6, 7), (7, 8)])
        nrng = np.random.default_rng(500 + 100 * seed + it)
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
        rc = RasterCollector(vals, steps, cell, turb, model, engine="B",
                             trench_mult=mult, drain_engine="python",
                             keep_trace=True)
        mv = rc.run().ravel()
        pricer = RasterStepPricer(vals, steps, cell, turb, trench_mult=mult)
        for g in np.flatnonzero(np.isfinite(mv)):
            design = rc.design(int(g))
            cost, _ = reprice(design, pricer, turb, model)
            assert cost == pytest.approx(mv[g], rel=1e-9, abs=1e-9)
            roots += 1
            stations += bool(design.junctions)
            parallel += any(s.option[1] > 1 for s in design.systems)
    assert roots > 100 and stations > 0 and parallel > 0


def test_trace_needs_engine_b_and_keep_trace():
    vals = np.full((4, 4), 2, dtype=np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True), dtype=np.int8)
    model = _model(random.Random(1), 2)
    with pytest.raises(ValueError, match="engine B"):
        RasterCollector(vals, steps, 1.0, [0, 15], model, engine="A",
                        keep_trace=True, drain_engine="python").run()
    rc = RasterCollector(vals, steps, 1.0, [0, 15], model, engine="B",
                         drain_engine="python")
    rc.run()
    with pytest.raises(RuntimeError, match="keep_trace"):
        rc.design(5)


def test_step_pricer_refuses_illegal_steps():
    from pyorps.collector.design import DesignError
    from pyorps.collector.raster_pricer import RasterStepPricer

    vals = np.full((5, 5), 3, dtype=np.uint16)
    vals[0, 4] = 65535
    steps = np.asarray(get_neighborhood_steps(2, directed=True), dtype=np.int8)
    pr = RasterStepPricer(vals, steps, 2.0, [7], trench_mult=1.5)
    cost, length = pr.price_step(None, 0, 1)            # one cell east
    assert length == 2.0 and cost == pytest.approx(1.5 * 2.0 * 3.0)
    with pytest.raises(DesignError, match="not a step"):
        pr.price_step(None, 0, 4)                       # 4 cells east
    # the r2 step (+1, +2) from (1,1) to (2,3) passes over (1,2) and (2,2);
    # (1,2) is cell 7, the turbine
    with pytest.raises(DesignError, match="turbine"):
        pr.price_step(None, 6, 13)
    with pytest.raises(DesignError, match="refuses"):
        pr.price_step(None, 3, 4)                       # into the excluded

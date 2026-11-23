"""The canonical node order (lemma L3) removes states, not designs.

User decision D2, 2026-10-05: engine B applies the node operations at a
Steiner cell in one topological order (no free-block station, no back-fill of
a down by a later child). The rule must leave every value, and every traced
design, unchanged. Here it is compared with the full interleaving
(``node_order="full"``) on random instances, in the reference engine and on
a raster, and against the independent oracles through the default engine
tests, which now run on the canonical order.
"""
import math
import random

import numpy as np
import pytest

from pyorps.collector import CollectorGraph, reprice, solve_collector
from pyorps.collector.raster import RasterCollector
from pyorps.collector.raster_pricer import RasterStepPricer
from pyorps.utils.neighborhood import get_neighborhood_steps

from .test_engine_vs_oracle_v2 import _grid, _model

INF = math.inf


def _same(a, b):
    fin = np.isfinite(a)
    np.testing.assert_array_equal(fin, np.isfinite(b))
    np.testing.assert_allclose(a[fin], b[fin], rtol=1e-12, atol=1e-12)


def _instances(seed, count, n_choices):
    rng = random.Random(seed)
    for _ in range(count):
        R, C = rng.choice([(2, 3), (3, 3), (2, 4), (3, 4)])
        N, E = _grid(R, C, rng, rng.random() < 0.4)
        n = rng.choice(n_choices)
        if n + 1 > N:
            continue
        cells = rng.sample(range(N), n + 1)
        yield N, E, cells[:n], cells[n], _model(rng, n)


@pytest.mark.parametrize("seed,n_choices,count", [
    (21, (2, 3), 14), (22, (4,), 5)])
def test_canonical_equals_full_in_the_reference_engine(seed, n_choices, count):
    fewer = 0
    checked = 0
    for N, E, turb, g, model in _instances(seed, count, n_choices):
        G = CollectorGraph(N, E)
        for root in (None, g):
            full = solve_collector(G, turb, model, engine="B", root=root,
                                   node_order="full")
            can = solve_collector(G, turb, model, engine="B", root=root,
                                  node_order="canonical")
            _same(can.mv, full.mv)
            for k in ("labels", "growable", "merge_pairs", "junction_arcs"):
                assert can.stats[k] <= full.stats[k], k
            fewer += (can.stats["merge_pairs"] + can.stats["junction_arcs"]
                      < full.stats["merge_pairs"] + full.stats["junction_arcs"])
            if root is not None and np.isfinite(can.mv[g]):
                cost, _ = reprice(can.design(g), G, turb, model,
                                  root_transit=False)
                assert cost == pytest.approx(can.mv[g], rel=1e-12, abs=1e-12)
            checked += 1
    assert checked >= 8 and fewer >= 1      # the rule does remove work


def test_engine_a_ignores_the_node_order():
    for N, E, turb, _g, model in _instances(23, 5, (2, 3)):
        G = CollectorGraph(N, E)
        if not model.interchangeable():
            model_a = model
        else:
            model_a = model
        a1 = solve_collector(G, turb, model_a, engine="A",
                             node_order="full").mv
        a2 = solve_collector(G, turb, model_a, engine="A",
                             node_order="canonical").mv
        _same(a1, a2)
        b = solve_collector(G, turb, model, engine="B").mv
        fin = np.isfinite(a1)
        assert np.all(a1[fin] <= b[fin] * (1 + 1e-9) + 1e-9)   # A <= B


def test_canonical_equals_full_on_a_raster():
    rng = random.Random(31)
    roots = 0
    for it in range(4):
        R, C = rng.choice([(6, 7), (7, 8)])
        nrng = np.random.default_rng(3100 + it)
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
        out = {}
        for order in ("full", "canonical"):
            rc = RasterCollector(vals, steps, cell, turb, model, engine="B",
                                 drain_engine="python", keep_trace=True,
                                 node_order=order)
            out[order] = (rc.run().ravel(), rc)
        _same(out["canonical"][0], out["full"][0])
        assert (out["canonical"][1].stats["merge_pairs"]
                <= out["full"][1].stats["merge_pairs"])
        mv = out["canonical"][0]
        pricer = RasterStepPricer(vals, steps, cell, turb)
        for g in np.flatnonzero(np.isfinite(mv))[:6]:
            design = out["canonical"][1].design(int(g))
            cost, _ = reprice(design, pricer, turb, model)
            assert cost == pytest.approx(mv[g], rel=1e-9, abs=1e-9)
            roots += 1
    assert roots >= 10


def test_a_bad_node_order_is_refused():
    N, E, turb, _g, model = next(_instances(24, 1, (2,)))
    with pytest.raises(ValueError, match="node_order"):
        solve_collector(CollectorGraph(N, E), turb, model, engine="B",
                        node_order="sorted")
    vals = np.full((4, 4), 2, dtype=np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True), dtype=np.int8)
    with pytest.raises(ValueError, match="node_order"):
        RasterCollector(vals, steps, 1.0, [0, 15], model, engine="B",
                        drain_engine="python", node_order="x").run()

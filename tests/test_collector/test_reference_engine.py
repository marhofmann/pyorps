"""The reference collector engine against the ported oracles.

``pyorps.collector.reference`` is a new implementation, written after the
oracles in ``tests/test_collector/oracles`` were ported and passed on their
own (plan section 13). It must reproduce them wherever the models coincide:

* with one cable per connection (``p_max = 1``) and no station building, a
  station costs ``panel * degree``, which is design A's non-additive
  junction price with ``J_T = 3 panel``, ``J_K = 4 panel`` for up to three
  turbines -- so engine B must equal design A's TYPED DP (which tries every
  cable type, i.e. it also checks the candidate-size rule);
* the per-section conductor variant must equal design A's untyped DP;
* the field over every root (collectors crossing the UW cell) is never
  above the single-root value;
* engine A (count tokens, type free per section) is a lower bound on
  engine B at every root, and count tokens with one option per cable equal
  set tokens exactly when the turbines are interchangeable;
* every traced design re-prices to its DP value in the independent
  re-pricer.
"""
import math
import random

import numpy as np
import pytest

from pyorps.collector import (
    CableType,
    CollectorGraph,
    CollectorModel,
    DesignError,
    reprice,
    solve_collector,
)

from .oracles import a_dp, a_gen
from .oracles.a_model import Inst

INF = math.inf
TOL = 1e-9


def _same(a, b):
    return (a == INF and b == INF) or abs(a - b) <= TOL * max(1.0, abs(a))


def _ours(inst, panel, *, p_max=1, building=0.0, switchgear=None):
    """Design A's instance in our model (identical current = power)."""
    n = inst.k
    types = tuple(CableType(f"t{i}", cap, a, b)
                  for i, (cap, a, b) in enumerate(inst.types))
    powers = [inst.power(m) if m else 0.0 for m in range(1 << n)]
    model = CollectorModel(
        n=n, types=types, current_a=tuple(powers),
        loss_weight=tuple(p * p for p in powers), loss_coef=1.0,
        derating=tuple(inst.derate[1:]), sigma_eur_per_m=inst.sigma,
        m_max=inst.mmax, p_max=p_max,
        switchgear_a=float(inst.kmax) if switchgear is None else switchgear,
        ring_panels=3, turbine_panel_eur=inst.panel,
        station_building_eur=building, station_panel_eur=panel,
        bay_eur=inst.bay)
    return model, CollectorGraph(inst.n, list(inst.edges))


def _instances(seed, count):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        kind = ("grid", "rand", "spur")[i % 3]
        inst = (a_gen.spur_inst(rng) if kind == "spur"
                else a_gen.rand_inst(rng, kind=kind))
        out.append((inst, rng.choice([0.0, 0.5, 1.0, 2.0]), rng))
    return out


class TestAgainstDesignA:
    @pytest.mark.parametrize("seed,count", [
        (5, 5), pytest.param(6, 12, marks=pytest.mark.slow)])
    def test_engine_b_equals_design_a_typed(self, seed, count):
        feasible = 0
        for inst, s, _rng in _instances(seed, count):
            inst.JT, inst.JK, inst.dmax = 3 * s, 4 * s, 5
            va = a_dp.ShareDP(inst, typed=True).solve()
            model, G = _ours(inst, s)
            res = solve_collector(G, inst.turb, model, engine="B",
                                  root=inst.root)
            vo = res.mv[inst.root]
            assert _same(va, vo), (va, vo)
            if vo < INF:
                feasible += 1
                cost, _ = reprice(res.design(inst.root), G, inst.turb,
                                  model, root_transit=False)
                assert math.isclose(cost, vo, rel_tol=1e-12, abs_tol=1e-12)
        assert feasible >= count // 2

    @pytest.mark.parametrize("seed", [7])
    def test_per_section_conductor_equals_design_a_untyped(self, seed):
        for inst, s, _rng in _instances(seed, 12):
            inst.JT, inst.JK, inst.dmax = 3 * s, 4 * s, 5
            va = a_dp.ShareDP(inst).solve()
            model, G = _ours(inst, s)
            vo = solve_collector(G, inst.turb, model, tokens="set",
                                 conductor="per_section",
                                 root=inst.root).mv[inst.root]
            assert _same(va, vo), (va, vo)

    def test_heterogeneous_turbines_equal_design_a(self):
        """Different turbine powers (user decision T5): set tokens are
        exact. The switchgear limit is lifted because design A counts
        turbines where our model limits current."""
        rng = random.Random(9)
        for _ in range(4):
            inst = a_gen.rand_inst(rng, kind="grid")
            inst.P = [rng.choice([0.5, 0.8, 1.0, 1.3, 2.0])
                      for _ in range(inst.k)]
            inst._pow = {}
            inst.kmax = inst.k
            s = rng.choice([0.0, 1.0])
            inst.JT, inst.JK, inst.dmax = 3 * s, 4 * s, 5
            va = a_dp.ShareDP(inst, typed=True).solve()
            model, G = _ours(inst, s, switchgear=INF)
            vo = solve_collector(G, inst.turb, model, engine="B",
                                 root=inst.root).mv[inst.root]
            assert _same(va, vo), (va, vo)

    def test_kiosk_with_a_station_building(self):
        """The verifier's kiosk case, priced as a station (building 2, panel
        1 per connected cable). Its optimum is a four-input kiosk at ``a``
        fed from below AND from above (t4 comes down from b), output up:
        design A prices that degree-5 junction at J_K = 6 and gets 119.1;
        our station costs 2 + 5 panels = 7, so the same design costs 120.1.
        The alternative kiosk at b would cost 121.2 here."""
        E = [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (2, 3, 1.0, 1.0),
             (2, 4, 1.0, 1.0), (2, 5, 1.0, 1.0), (6, 1, 1.0, 1.0),
             (7, 0, 1.0, 1.0), (7, 6, 3.0, 1.0)]
        inst = Inst(n=8, edges=E, turb=[3, 4, 5, 6], root=0, JT=5.0, JK=6.0,
                    dmax=5, mmax=99, types=[(10.0, 1.0, 0.0)],
                    derate=[1.0] * 9, sigma=0.1, bay=100.0, panel=0.0,
                    kmax=1)
        model, G = _ours(inst, 1.0, building=2.0)
        res = solve_collector(G, inst.turb, model, engine="B", root=0)
        assert res.mv[0] == pytest.approx(120.1, abs=1e-9)
        design = res.design(0)
        cost, parts = reprice(design, G, inst.turb, model,
                              root_transit=False)
        assert cost == pytest.approx(120.1, abs=1e-9)
        assert parts["stations"] == pytest.approx(7.0, abs=1e-12)
        (junction,) = design.junctions
        assert G.n_nodes and design.tnodes[junction.tnode] == 2    # at a
        assert len(junction.inputs) == 4


class TestEngineRelations:
    @pytest.mark.parametrize("seed", [17, 18])
    def test_field_bounds_and_symmetry(self, seed):
        roots = 0
        for inst, s, rng in _instances(seed, 12):
            model, G = _ours(inst, s, p_max=rng.choice([1, 2]),
                             building=rng.choice([0.0, 1.5]))
            fB = solve_collector(G, inst.turb, model, engine="B")
            fA = solve_collector(G, inst.turb, model, engine="A").mv
            fC = solve_collector(G, inst.turb, model, tokens="count",
                                 conductor="per_cable").mv
            for g in range(inst.n):
                if g in inst.turb:
                    assert fB.mv[g] == INF
                    continue
                roots += 1
                # engine A is admissible
                assert fA[g] <= fB.mv[g] + TOL
                # count tokens are an exact symmetry reduction here
                assert _same(fC[g], fB.mv[g])
                single = solve_collector(G, inst.turb, model, engine="B",
                                         root=g).mv[g]
                assert fB.mv[g] <= single + TOL
                if fB.mv[g] < INF:
                    cost, _ = reprice(fB.design(g), G, inst.turb, model)
                    assert math.isclose(cost, fB.mv[g], rel_tol=1e-12,
                                        abs_tol=1e-12)
        assert roots > 40

    @pytest.mark.parametrize("seed,count", [
        (24, 3), pytest.param(23, 8, marks=pytest.mark.slow)])
    def test_candidate_rule_is_exact_with_parallel_cables(self, seed, count):
        for inst, s, rng in _instances(seed, count):
            model, G = _ours(inst, s, p_max=2,
                             building=rng.choice([0.0, 1.5]))
            ruled = solve_collector(G, inst.turb, model, engine="B").mv
            every = solve_collector(G, inst.turb, model, engine="B",
                                    candidate_rule=False).mv
            fin = np.isfinite(every)
            np.testing.assert_array_equal(np.isfinite(ruled), fin)
            assert np.all(np.abs(ruled[fin] - every[fin])
                          <= TOL * np.maximum(1.0, every[fin]))


class TestStationAndParallelRules:
    """Small hand-built cases for the 2026-09-24 rules."""

    @staticmethod
    def _line(n_turb, c=1.0):
        """t0 - x - g with the other turbines hanging off x."""
        # nodes: 0..n_turb-1 turbines, n_turb = x, n_turb + 1 = g
        x, g = n_turb, n_turb + 1
        edges = [(i, x, c, 1.0) for i in range(n_turb)]
        edges.append((x, g, 10.0 * c, 10.0))
        return CollectorGraph(n_turb + 2, edges), list(range(n_turb)), x, g

    def test_a_station_pays_off_only_when_it_saves_cable(self):
        """Two turbines behind a long shared run: two separate cables in
        one trench against one station and a single cable. The station is
        built exactly when it is cheaper."""
        G, A, x, g = self._line(2)
        types = (CableType("small", 1.0, 5.0, 0.0),
                 CableType("big", 2.0, 7.0, 0.0))
        # switchgear_a = 1 turbine: no loop-in chain through a turbine, so
        # the choice is two cables to the UW or one station at x.
        base = dict(n=2, types=types, current_one_a=1.0, switchgear_a=1.0,
                    m_max=4, bay_eur=0.0)
        cheap = CollectorModel.identical_turbines(
            station_building_eur=10.0, station_panel_eur=1.0, **base)
        dear = CollectorModel.identical_turbines(
            station_building_eur=40.0, station_panel_eur=1.0, **base)
        v_cheap = solve_collector(G, A, cheap, root=g)
        v_dear = solve_collector(G, A, dear, root=g)
        # spurs: 2 x (trench 1 + small cable 5) = 12; trunk trench 10.
        # no station: two small cables on the 10 m trunk, 2 x 50 -> 122
        # station at x: building + 3 panels + one big cable 70 -> 95 + b
        assert v_cheap.mv[g] == pytest.approx(12 + 10 + 70 + 10 + 3)
        assert v_dear.mv[g] == pytest.approx(12 + 10 + 100)
        assert v_cheap.design(g).summary()["stations"] == 1
        assert v_dear.design(g).summary()["stations"] == 0

    def test_two_sections_pay_the_building_once(self):
        """Four turbines, one station node x, two outgoing cables (each at
        most two turbines, e.g. a feeder-bay limit): the station has two
        busbar sections and pays ONE building."""
        G, A, x, g = self._line(4)
        types = (CableType("c", 2.0, 5.0, 0.0),)
        model = CollectorModel.identical_turbines(
            4, types=types, current_one_a=1.0, m_max=4,
            station_building_eur=50.0, station_panel_eur=1.0,
            bay_eur=0.0, bay_a=2.0)
        res = solve_collector(G, A, model, root=g)
        d = res.design(g)
        cost, parts = reprice(d, G, A, model)
        assert cost == pytest.approx(res.mv[g])
        s = d.summary()
        if s["junctions"] == 2:
            assert s["stations"] == 1
            assert parts["stations"] == pytest.approx(50.0 + 6 * 1.0)

    def test_parallel_cables_count_panels_and_trench_cables(self):
        """A parallel pair is two cables: two bays, two turbine panels,
        and two toward m_max and the derating."""
        G = CollectorGraph(2, [(0, 1, 1.0, 10.0)])
        types = (CableType("c", 1.0, 3.0, 0.0),)
        model = CollectorModel.identical_turbines(
            1, types=types, current_one_a=1.5, m_max=4, p_max=2,
            turbine_panel_eur=2.0, bay_eur=4.0)
        res = solve_collector(G, [0], model, root=1)
        # one cable cannot carry 1.5 A, a pair can: trench 1 + 2 x 3 x 10
        # + sigma 0 + 2 turbine panels x 2 + 2 bays x 4
        assert res.mv[1] == pytest.approx(1.0 + 60.0 + 4.0 + 8.0)
        d = res.design(1)
        assert d.systems[0].option == (0, 2)
        # m_max = 1 forbids the pair outright
        tight = CollectorModel.identical_turbines(
            1, types=types, current_one_a=1.5, m_max=1, p_max=2)
        assert solve_collector(G, [0], tight, root=1).mv[1] == INF

    def test_bay_current_is_per_conductor(self):
        G = CollectorGraph(2, [(0, 1, 1.0, 1.0)])
        types = (CableType("c", 10.0, 1.0, 0.0),)
        one = CollectorModel.identical_turbines(
            1, types=types, current_one_a=3.0, p_max=1, bay_a=2.0)
        two = CollectorModel.identical_turbines(
            1, types=types, current_one_a=3.0, p_max=2, bay_a=2.0)
        assert solve_collector(G, [0], one, root=1).mv[1] == INF
        assert solve_collector(G, [0], two, root=1).mv[1] < INF

    def test_no_stations_means_no_junctions(self):
        G, A, x, g = self._line(3)
        types = (CableType("c", 3.0, 5.0, 0.0),)
        model = CollectorModel.identical_turbines(
            3, types=types, current_one_a=1.0, m_max=4,
            allow_stations=False)
        res = solve_collector(G, A, model, root=g)
        assert res.design(g).summary()["junctions"] == 0


class TestRepricerRefusals:
    def test_a_tampered_option_is_caught(self):
        G = CollectorGraph(2, [(0, 1, 1.0, 10.0)])
        types = (CableType("c", 1.0, 3.0, 0.0),)
        model = CollectorModel.identical_turbines(
            1, types=types, current_one_a=1.5, m_max=4, p_max=2)
        res = solve_collector(G, [0], model, root=1)
        d = res.design(1)
        d.systems[0].option = (0, 1)        # one cable cannot carry 1.5 A
        with pytest.raises(DesignError, match="infeasible"):
            reprice(d, G, [0], model)

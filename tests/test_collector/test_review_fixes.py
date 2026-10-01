"""Regression tests for the read-only review of the collector (2026-09-24)."""
import itertools
import math
import random

import pytest

from pyorps.collector import (
    CableType,
    CollectorGraph,
    CollectorModel,
    Design,
    DesignError,
    System,
    reprice,
    solve_collector,
)
from pyorps.collector import reference as ref

from .test_engine_vs_oracle_v2 import _grid, _model


def _one_turbine_model(**kw):
    base = dict(types=(CableType("a", 5.0, 1.0, 0.0),), loss_coef=1.0,
                derating=(1.0, 1.0, 1.0, 1.0), sigma_eur_per_m=0.0,
                m_max=4, p_max=1, switchgear_a=math.inf, ring_panels=3,
                turbine_panel_eur=0.0, station_building_eur=0.0,
                station_panel_eur=0.0, allow_stations=False, bay_eur=0.0,
                bay_a=math.inf)
    base.update(kw)
    return CollectorModel.identical_turbines(1, current_one_a=1.0, **base)


def _path(L):
    return CollectorGraph(L + 1, [(i, i + 1, 1.0, 1.0) for i in range(L)])


def test_traceback_of_a_long_trench_needs_no_deep_recursion():
    """Finding 1: design() recursed once per trench step and failed near
    1 000 steps."""
    L = 2000
    G = _path(L)
    model = _one_turbine_model()
    res = solve_collector(G, [0], model, engine="B", root=L)
    assert res.mv[L] == pytest.approx(2 * L)
    cost, _ = reprice(res.design(L), G, [0], model, root_transit=False)
    assert cost == pytest.approx(2 * L)


def test_reprice_accepts_a_deep_design_numbered_leaf_first():
    """Finding 3: reprice's depth walk was recursive."""
    L = 1500
    G = _path(L)
    model = _one_turbine_model()
    # trench node k sits on graph node k; node L is the root
    design = Design(tnodes=list(range(L + 1)),
                    tedges=[(k, k + 1, k) for k in range(L)],
                    root_tnode=L, turbine_tnode={0: 0},
                    systems=[System(mask=1, option=(0, 1),
                                    source=("turbine", 0),
                                    sink=("root", 0))],
                    junctions=[])
    cost, _ = reprice(design, G, [0], model)
    assert cost == pytest.approx(2 * L)


def test_merge_prefilter_is_exact(monkeypatch):
    """Finding 2: the needs index must never drop a pair merge() accepts."""
    rng = random.Random(3)
    for _ in range(8):
        R, C = rng.choice([(2, 3), (3, 3)])
        N, E = _grid(R, C, rng, rng.random() < 0.5)
        n = rng.choice([2, 3])
        turb = rng.sample(range(N), n)
        model = _model(rng, n)
        G = CollectorGraph(N, E)
        fast = solve_collector(G, turb, model, engine="B").mv
        with monkeypatch.context() as m:
            m.setattr(ref._SetAlgebra, "merge_pairs",
                      lambda self, a, b, s1, s2: itertools.product(a, b))
            slow = solve_collector(G, turb, model, engine="B").mv
        assert list(fast) == list(slow)


def test_count_tokens_per_cable_refused_for_mixed_turbines():
    """Finding 4: that switch combination can land ABOVE engine B."""
    types = (CableType("small", 1.0, 1.0, 0.0),
             CableType("big", 3.0, 5.0, 0.0))
    mixed = CollectorModel(
        n=2, types=types, current_a=(0.0, 2.0, 0.5, 2.5),
        loss_weight=(0.0, 0.0, 0.0, 0.0), loss_coef=1.0,
        derating=(1.0, 1.0), sigma_eur_per_m=0.0, m_max=2, p_max=1,
        switchgear_a=math.inf, ring_panels=3, turbine_panel_eur=0.0,
        station_building_eur=0.0, station_panel_eur=0.0,
        allow_stations=False, bay_eur=0.0, bay_a=math.inf)
    G = CollectorGraph(3, [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0)])
    with pytest.raises(ValueError, match="neither exact nor a relaxation"):
        solve_collector(G, [0, 2], mixed, tokens="count",
                        conductor="per_cable")
    a = solve_collector(G, [0, 2], mixed, engine="A").mv[1]
    b = solve_collector(G, [0, 2], mixed, engine="B").mv[1]
    assert a <= b
    same = CollectorModel.identical_turbines(
        2, current_one_a=1.0, types=types, loss_coef=1.0, derating=(1.0, 1.0),
        sigma_eur_per_m=0.0, m_max=2, p_max=1, switchgear_a=math.inf,
        ring_panels=3, turbine_panel_eur=0.0, station_building_eur=0.0,
        station_panel_eur=0.0, allow_stations=False, bay_eur=0.0,
        bay_a=math.inf)
    solve_collector(G, [0, 2], same, tokens="count", conductor="per_cable")


def test_reprice_checks_options_against_the_catalogue():
    """Finding 5."""
    G = _path(3)
    model = _one_turbine_model(p_max=2)
    res = solve_collector(G, [0], model, engine="B", root=3)
    for bad in ((0, 3), (5, 1), (-1, 1)):
        design = res.design(3)
        design.systems[0].option = bad
        with pytest.raises(DesignError, match="catalogue"):
            reprice(design, G, [0], model, root_transit=False)


def test_negative_inputs_are_refused():
    """Finding 6: a negative resistance broke A <= B."""
    with pytest.raises(ValueError, match="resistance"):
        CableType("x", 1.0, 1.0, -0.5)
    with pytest.raises(ValueError, match="cost"):
        CableType("x", 1.0, -1.0, 0.1)
    with pytest.raises(ValueError, match="ampacity"):
        CableType("x", 0.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="current_a"):
        CollectorModel(
            n=1, types=(CableType("a", 5.0, 1.0, 0.0),),
            current_a=(0.0, -1.0), loss_weight=(0.0, 1.0), loss_coef=1.0,
            derating=(1.0,), sigma_eur_per_m=0.0, m_max=4, p_max=1,
            switchgear_a=math.inf, ring_panels=3, turbine_panel_eur=0.0,
            station_building_eur=0.0, station_panel_eur=0.0,
            allow_stations=False, bay_eur=0.0, bay_a=math.inf)
    with pytest.raises(ValueError, match="loss_weight"):
        CollectorModel(
            n=1, types=(CableType("a", 5.0, 1.0, 0.0),),
            current_a=(0.0, 1.0), loss_weight=(0.0, -3.0), loss_coef=1.0,
            derating=(1.0,), sigma_eur_per_m=0.0, m_max=4, p_max=1,
            switchgear_a=math.inf, ring_panels=3, turbine_panel_eur=0.0,
            station_building_eur=0.0, station_panel_eur=0.0,
            allow_stations=False, bay_eur=0.0, bay_a=math.inf)

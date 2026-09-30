"""The plant around the collector: NAP, HV leg, UW transformer with OLTC.

* the plant tree's load flow equals pandapower (line, tap-changing
  transformer with iron losses as ``trafo_model="pi"``, collector cables,
  static generators) at every tap, <= 1e-9 p.u.;
* the plant controller meets the reactive target at the NAP; a target the
  turbines cannot reach is a reported deficit (compensation needed);
* the tap changer rests only where the MV busbar is within its band; a
  range that cannot reach the band is a violation;
* the no-load reactive limit (Syna: 0.02 P_b,inst over-excited) asks for a
  reactor behind a 110 kV cable, not behind an overhead line;
* Syna's no-FRT criterion (1.08 U_NS) binds when the setpoint is high;
* the operators' rules come from ``voltage_2026.yaml`` as the TABs state
  them.
"""
import dataclasses
import json
import math
import random

import numpy as np
import pytest

from pyorps.collector import CollectorGraph, solve_collector
from pyorps.collector.voltage import (
    Branch,
    ElectricalTree,
    load_flow,
    voltage_model_from_yaml,
)
from pyorps.collector.voltage_plant import (
    GridRules,
    HVLeg,
    NapCorner,
    PlantModel,
    UWTransformer,
    _solve_tap,
    check_plant,
    check_plant_tree,
    grid_rules_from_yaml,
    plant_model_from_yaml,
    plant_tree,
)

from .test_engine_vs_oracle_v2 import _grid, _model
from .test_voltage import YAML, _vmodel_for

pp = pytest.importorskip("pandapower")
OMEGA = 2 * math.pi * 50
OHL = dict(r_ohm_per_km=0.1188, x_ohm_per_km=0.39, c_nf_per_km=9.0)
CABLE = dict(r_ohm_per_km=0.06, x_ohm_per_km=0.144, c_nf_per_km=144.0)


def _collector(u_kv=20.0):
    """Seven turbines on three feeders, km-scale 240/630 mm2 cables."""
    vm = voltage_model_from_yaml(YAML, u_kv=u_kv, cable_mm2=[240, 630],
                                 n_turbines=7)
    c240, c630 = vm.cables
    lab = [("root", 0)] + [("turbine", t) for t in range(7)]
    spec = [(1, 0, c630, 2400, 2), (2, 1, c240, 900, 1), (3, 2, c240, 700, 1),
            (4, 0, c630, 1800, 1), (5, 4, c240, 800, 1), (6, 0, c240, 1500, 1),
            (7, 6, c240, 600, 1)]
    br = [Branch(ch, pa, c.r_ohm_per_m * L / p, c.x_ohm_per_m * L / p,
                 OMEGA * c.c_f_per_m * L * p) for ch, pa, c, L, p in spec]
    return vm, ElectricalTree(u_kv, lab, br, {t: 1 + t for t in range(7)})


def _plant(vm, hv=None, **kw):
    return dataclasses.replace(plant_model_from_yaml(YAML, vm, hv_leg=hv),
                               **kw)


@pytest.mark.parametrize("tap", [-9, -3, 0, 5, 9])
@pytest.mark.parametrize("leg", ["ohl", "cable"])
def test_plant_tree_equals_pandapower(tap, leg):
    vm, tree = _collector()
    hv = HVLeg.from_type(**(OHL if leg == "ohl" else CABLE), length_km=4.0)
    plant = _plant(vm, hv)
    tr = plant.transformer
    pt = plant_tree(tree, plant, tap)
    s = [0j] * pt.n_nodes
    for t in range(7):
        s[pt.turbine_node[t]] = complex(6.8, 2.1)
    u_nap = 1.03
    lf = load_flow(pt, s, u_nap * 110 / tr.ratio(tap) / 20.0)
    net = pp.create_empty_network()
    b_nap, b_hv = pp.create_bus(net, 110.0), pp.create_bus(net, 110.0)
    mv = [pp.create_bus(net, 20.0) for _ in range(8)]
    pp.create_ext_grid(net, b_nap, vm_pu=u_nap)
    k = OHL if leg == "ohl" else CABLE
    pp.create_line_from_parameters(net, b_nap, b_hv, 4.0, k["r_ohm_per_km"],
                                   k["x_ohm_per_km"], k["c_nf_per_km"], 1.0)
    pp.create_transformer_from_parameters(
        net, b_hv, mv[0], sn_mva=tr.s_r_mva, vn_hv_kv=tr.u_hv_kv,
        vn_lv_kv=tr.u_mv_kv, vk_percent=100 * tr.uk,
        vkr_percent=100 * tr.vkr, pfe_kw=tr.pfe_kw, i0_percent=100 * tr.i0,
        tap_side="hv", tap_neutral=0, tap_min=tr.tap_min,
        tap_max=tr.tap_max, tap_step_percent=tr.tap_step_pct, tap_pos=tap,
        tap_changer_type="Ratio")
    for b in tree.branches:
        pp.create_line_from_parameters(net, mv[b.child], mv[b.parent], 1.0,
                                       b.r_ohm, b.x_ohm, b.b_s / OMEGA * 1e9,
                                       1.0)
    for t in range(7):
        pp.create_sgen(net, mv[1 + t], p_mw=6.8, q_mvar=2.1)
    pp.runpp(net, trafo_model="pi", tolerance_mva=1e-11)
    ours = [abs(lf.v_pu[k]) for k in range(2, pt.n_nodes)]
    np.testing.assert_allclose(ours, net.res_bus.vm_pu.values[2:],
                               atol=1e-9, rtol=0)
    assert lf.s_root_mva.imag == pytest.approx(
        -net.res_ext_grid.q_mvar.values[0], abs=1e-6)


def test_the_controller_meets_the_nap_target():
    # (0.41 over-excited behind 5 km of line is NOT reachable with the
    # turbines' 0.54 Q/P: the UW transformer and the line consume the rest;
    # see test_a_target_the_turbines_cannot_reach_is_a_deficit)
    vm, tree = _collector()
    plant = _plant(vm, HVLeg.from_type(**OHL, length_km=5.0))
    for q in (0.35, 0.2, 0.0, -0.33):
        corner = NapCorner("c", 1.0, q)
        for tap in (-4, 0, 4):
            r = _solve_tap(plant_tree(tree, plant, tap), plant, vm, corner,
                           tap)
            assert r.deficit == 0.0
            assert r.q_nap == pytest.approx(q * plant.p_b_inst, abs=1e-6)
            assert -plant.q_under_p <= r.kappa <= plant.q_over_p


def test_a_target_the_turbines_cannot_reach_is_a_deficit():
    vm, tree = _collector()
    plant = _plant(vm, HVLeg.from_type(**OHL, length_km=20.0),
                   q_over_p=0.30)
    rules = GridRules("t", (NapCorner("Q", 1.0, 0.41,
                                      checks=frozenset({"band"})),))
    chk = check_plant_tree(tree, vm, plant, rules)
    assert not chk.passed and not chk.reactive_ok
    assert chk.q_deficit_mvar > 1.0
    assert any("compensation" in v for v in chk.reactive)
    r = chk.corners["Q"]
    assert all(k == pytest.approx(0.30) for k in r.kappa.values())


def test_the_tap_changer_rests_only_inside_its_band():
    vm, tree = _collector()
    plant = _plant(vm)
    rules = GridRules("t", (NapCorner("A", 0.95, 0.2),
                            NapCorner("B", 1.08, -0.2)))
    chk = check_plant_tree(tree, vm, plant, rules)
    lo = plant.u_ms_pu * (1 - plant.deadband)
    hi = plant.u_ms_pu * (1 + plant.deadband)
    for c in chk.corners.values():
        assert c.taps and not c.oltc_exhausted
        assert all(lo <= v <= hi for v in c.v_mv_pu.values())
    # a short range cannot reach the band at an extreme NAP voltage
    short = dataclasses.replace(
        plant, transformer=dataclasses.replace(plant.transformer, tap_min=-1,
                                               tap_max=1))
    rules2 = GridRules("t", (NapCorner("C", 0.88, 0.0),))
    chk2 = check_plant_tree(tree, vm, short, rules2)
    assert chk2.corners["C"].oltc_exhausted
    assert any("range exhausted" in v for v in chk2.violations)


def test_no_load_reactive_power_needs_a_reactor_behind_a_cable():
    vm, tree = _collector()
    rules = grid_rules_from_yaml(YAML, "syna")
    only_n0 = dataclasses.replace(
        rules, corners=tuple(c for c in rules.corners if c.name == "N0"))
    cable = check_plant_tree(tree, vm,
                             _plant(vm, HVLeg.from_type(**CABLE,
                                                        length_km=5.0)),
                             only_n0)
    ohl = check_plant_tree(tree, vm,
                           _plant(vm, HVLeg.from_type(**OHL, length_km=5.0)),
                           only_n0)
    allowance = 0.02 * 47.6
    # ~0.55 Mvar/km of 110 kV cable charging, plus the MV cables
    assert cable.no_load_q_mvar > 5 * 0.5
    assert cable.reactor_mvar == pytest.approx(
        cable.no_load_q_mvar - allowance, rel=1e-9)
    assert any("shunt reactor" in v for v in cable.reactive)
    assert cable.voltage_ok and not cable.reactive_ok
    assert ohl.no_load_q_mvar < allowance
    assert ohl.reactor_mvar == 0.0


def test_syna_frt_criterion_binds():
    vm, tree = _collector()
    rules = grid_rules_from_yaml(YAML, "syna")
    s3 = dataclasses.replace(
        rules, corners=tuple(c for c in rules.corners if c.name == "S3"))
    ok = check_plant_tree(tree, vm, _plant(vm), s3)
    top = ok.corners["S3"].max_ns
    assert ok.passed and 1.0 < top < 1.08
    # U_NS = U_MS / ue scales with the setpoint, so a higher setpoint moves
    # U_LV / U_NS little
    high = check_plant_tree(tree, vm, _plant(vm, u_ms_pu=1.04), s3)
    assert high.corners["S3"].max_ns == pytest.approx(top, rel=0.02)
    # a criterion below what every turbine tap reaches fails the check
    best = min(ok.corners["S3"].max_ns, top)
    tight = dataclasses.replace(s3, frt_ns=(0.92, best - 0.05))
    bad = check_plant_tree(tree, vm, _plant(vm), tight)
    assert not bad.passed
    assert any("FRT" in v for v in bad.violations)


def test_operator_rules_from_the_yaml():
    syna = grid_rules_from_yaml(YAML, "syna")
    names = [c.name for c in syna.corners]
    assert names == ["Qover_lo", "Qover_hi", "Qunder_lo", "Qunder_hi",
                     "S1a", "S1b", "S1c", "S1d", "S3", "N0"]
    c = {c.name: c for c in syna.corners}
    assert c["Qover_lo"].q_nap == 0.41 and c["Qunder_hi"].q_nap == -0.33
    assert c["Qover_lo"].u_nap_pu == pytest.approx(96 / 110)
    assert c["Qunder_hi"].u_nap_pu == pytest.approx(123 / 110)
    assert c["S1a"].u_nap_pu == 0.90 and c["S1a"].q_nap == 0.33
    assert c["S1d"].q_nap == -0.33 and "no_trip" in c["S1d"].checks
    assert c["S3"].u_nap_pu == 1.05 and "no_frt" in c["S3"].checks
    assert c["N0"].p_frac == 0.0 and c["N0"].q_nap is None
    assert syna.no_load_q == (0.05, 0.02)
    assert syna.trip_ns == (0.80, 1.25) and syna.mv_upper_ms == 1.10
    ava = grid_rules_from_yaml(YAML, "avacon")
    ca = {c.name: c for c in ava.corners}
    assert ca["Qover_lo"].q_nap == 0.48 and ca["Qunder_lo"].q_nap == -0.41
    assert "no_trip" in ca["Qover_hi"].checks and "S3" not in ca
    assert ava.no_load_q == (0.05, 0.0)
    assert len(syna.meta["sha256"]) == 64


def test_the_whole_check_on_a_traced_design_is_recorded():
    rng = random.Random(3)
    N, E = _grid(3, 4, rng, True)
    turb = [0, 5, 11]
    model = _model(rng, 3)
    G = CollectorGraph(N, E)
    res = solve_collector(G, turb, model, engine="B")
    g = int(np.flatnonzero(np.isfinite(res.mv))[0])
    vm = _vmodel_for(model, random.Random(1))
    plant = PlantModel(UWTransformer(63, 115, 21, 0.125, 0.0032, 22, 0.0004),
                       (2.0, 3.4, 6.8), 0.54, 0.484)
    chk = check_plant(res.design(g), G, turb, model, vm, plant,
                      grid_rules_from_yaml(YAML, "syna"))
    rec = json.loads(json.dumps(chk.as_record()))
    assert rec["rules"] == "syna" and set(rec["corners"]) >= {"S3", "N0"}
    assert isinstance(chk.passed, bool)

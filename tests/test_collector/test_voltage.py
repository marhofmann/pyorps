"""The collector voltage check C5 (plan ``2026-09-25-voltage-limit-c5.md``).

* the backward--forward sweep equals the two-bus closed form and
  pandapower's Newton--Raphson on random radial networks with cable
  charging (<= 1e-7 p.u.);
* the electrical tree of a traced design (graph and raster engines) has the
  re-pricer's lengths, and its load flow equals pandapower's on the same
  network, parallel cables and zero-length systems included;
* the tap logic: a far turbine that passes only on a higher tap, a turbine
  whose high and low corners cannot share one tap, the common-tap policy,
  the equipment rating;
* the model built from ``voltage_2026.yaml`` carries the file's values.
"""
import math
import random
from pathlib import Path

import numpy as np
import pytest

from pyorps.collector import CollectorGraph, reprice, solve_collector
from pyorps.collector.design import layout
from pyorps.collector.voltage import (
    Branch,
    CableElectrical,
    Corner,
    ElectricalTree,
    LoadFlowError,
    TurbineTransformer,
    VoltageModel,
    check_voltage,
    electrical_tree,
    load_flow,
    standard_corners,
    voltage_model_from_yaml,
)

from .test_engine_vs_oracle_v2 import _grid, _model

pp = pytest.importorskip("pandapower")

YAML = (Path(__file__).resolve().parents[2] / "case_studies"
        / "runkel_free_siting" / "config" / "voltage_2026.yaml")
OMEGA = 2 * math.pi * 50


def _two_bus_closed_form(r, x, p, q, v0):
    """|V_s| of a PQ injection behind z = r + jx from a slack v0 (p.u.)."""
    a = r * p + x * q
    b = x * p - r * q
    k = 2 * a + v0 ** 2
    return math.sqrt((k + math.sqrt(k * k - 4 * (a * a + b * b))) / 2)


def _pandapower_voltages(tree, s_mva, v0):
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, tree.u_kv) for _ in range(tree.n_nodes)]
    pp.create_ext_grid(net, buses[0], vm_pu=v0)
    for b in tree.branches:
        if b.r_ohm == 0 and b.x_ohm == 0:
            assert b.b_s == 0
            pp.create_switch(net, buses[b.child], buses[b.parent], et="b",
                             closed=True)
            continue
        # one 1 km line carrying the branch's totals
        pp.create_line_from_parameters(
            net, buses[b.child], buses[b.parent], length_km=1.0,
            r_ohm_per_km=b.r_ohm, x_ohm_per_km=b.x_ohm,
            c_nf_per_km=b.b_s / OMEGA * 1e9, max_i_ka=10.0)
    for k, s in enumerate(s_mva):
        if k and s != 0:
            pp.create_sgen(net, buses[k], p_mw=s.real, q_mvar=s.imag)
    pp.runpp(net, tolerance_mva=1e-11, max_iteration=50)
    return net.res_bus.vm_pu.values


def _random_tree(rng, n_nodes, u_kv):
    labels = [("root", 0)] + [("junction", k) for k in range(1, n_nodes)]
    branches = []
    for k in range(1, n_nodes):
        par = rng.randrange(k)
        km = rng.uniform(0.1, 4.0)
        branches.append(Branch(k, par, r_ohm=km * rng.uniform(0.06, 0.27),
                               x_ohm=km * rng.uniform(0.09, 0.13),
                               b_s=OMEGA * km * rng.uniform(0.2, 0.45) * 1e-6))
    return ElectricalTree(u_kv=u_kv, labels=labels, branches=branches)


# ------------------------------------------------------------ load flow


@pytest.mark.parametrize("r,x,p,q,v0", [
    (0.001, 0.002, 6.8, 3.672, 1.0), (0.003, 0.001, 6.8, -3.29, 0.99),
    (0.0005, 0.0009, 0.0, 2.0, 1.02), (0.002, 0.004, 20.0, 0.0, 1.0107)])
def test_two_bus_equals_the_closed_form(r, x, p, q, v0):
    zb = 20.0 ** 2
    tree = ElectricalTree(20.0, [("root", 0), ("turbine", 0)],
                          [Branch(1, 0, r * zb, x * zb)], {0: 1})
    lf = load_flow(tree, [0, complex(p, q)], v0)
    assert abs(lf.v_pu[1]) == pytest.approx(
        _two_bus_closed_form(r, x, p, q, v0), abs=1e-11)
    # the losses are the series element's z |I|^2
    i2 = abs(lf.i_ka[0] * 1e3 / (1e3 / (math.sqrt(3) * 20.0))) ** 2
    assert lf.losses_mva == pytest.approx(complex(r, x) * i2, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_sweep_equals_pandapower_on_random_trees(seed):
    rng = random.Random(seed)
    for _ in range(5):
        u = rng.choice([20.0, 30.0, 33.0])
        tree = _random_tree(rng, rng.randint(2, 14), u)
        s = [0j] + [complex(rng.choice([0.0, 6.8, 4.1]),
                            rng.uniform(-3.3, 3.7))
                    for _ in range(tree.n_nodes - 1)]
        v0 = rng.uniform(0.98, 1.03)
        lf = load_flow(tree, s, v0)
        ref = _pandapower_voltages(tree, s, v0)
        np.testing.assert_allclose([abs(v) for v in lf.v_pu], ref,
                                   atol=1e-7, rtol=0)


def test_zero_length_branch_and_divergence():
    tree = ElectricalTree(20.0, [("root", 0), ("junction", 0),
                                 ("turbine", 0)],
                          [Branch(1, 0, 0.0, 0.0, 0.0),
                           Branch(2, 1, 1.0, 2.0, 1e-5)], {0: 2})
    lf = load_flow(tree, [0, 0, 6 + 2j], 1.0)
    assert lf.v_pu[1] == lf.v_pu[0]
    ref = _pandapower_voltages(tree, [0, 0, 6 + 2j], 1.0)
    np.testing.assert_allclose([abs(v) for v in lf.v_pu], ref, atol=1e-7)
    # an impossible transfer: no convergence, never a number
    far = ElectricalTree(20.0, [("root", 0), ("turbine", 0)],
                         [Branch(1, 0, 400.0, 400.0)], {0: 1})
    with pytest.raises(LoadFlowError):
        load_flow(far, [0, -5 - 5j], 1.0, max_iter=100)


def test_tree_refuses_bad_networks():
    with pytest.raises(ValueError, match="two out-branches"):
        ElectricalTree(20.0, [("root", 0), ("junction", 0)],
                       [Branch(1, 0, 1, 1), Branch(1, 0, 1, 1)])
    with pytest.raises(ValueError, match="cycle|connected"):
        ElectricalTree(20.0, [("root", 0), ("junction", 0), ("junction", 1)],
                       [Branch(1, 2, 1, 1), Branch(2, 1, 1, 1)])


# ------------------------------------------------- traced designs


def _vmodel_for(model, rng, u_kv=20.0, *, per_m=1.0):
    """Synthetic cables whose impedance makes the tiny test graphs (edge
    lengths ~1-3 m) behave like km-long collectors."""
    cables = tuple(CableElectrical(rng.uniform(0.06, 0.27) * per_m,
                                   rng.uniform(0.09, 0.13) * per_m,
                                   rng.uniform(0.2, 0.45) * 1e-6 * per_m,
                                   name=f"c{i}")
                   for i in range(len(model.types)))
    p = [rng.choice([2.0, 3.4, 6.8]) for _ in range(model.n)]
    return VoltageModel(
        u_kv=u_kv, cables=cables,
        corners=standard_corners(p, q_over_p=0.54, q_under_p=0.484),
        taps_kv=(20.0, 20.5, 21.0, 21.5, 22.0), u_m_kv=24.0,
        transformer=TurbineTransformer(7.8, 0.09, 80.0))


def _design_cases(seed, count):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        R, C = rng.choice([(3, 3), (3, 4), (4, 4)])
        N, E = _grid(R, C, rng, rng.random() < 0.4)
        n = rng.choice([2, 3])
        turb = rng.sample(range(N), n)
        model = _model(rng, n)
        G = CollectorGraph(N, E)
        res = solve_collector(G, turb, model, engine="B")
        for g in np.flatnonzero(np.isfinite(res.mv)):
            out.append((G, turb, model, res.design(int(g)), rng))
    return out


def test_traced_designs_load_flow_equals_pandapower():
    parallel = stations = 0
    cases = _design_cases(21, 8)
    assert len(cases) > 20
    for G, turb, model, design, rng in cases[::3]:
        vm = _vmodel_for(model, rng)
        tree = electrical_tree(design, G, turb, model, vm)
        lay = layout(design, G, turb, model)
        # branch lengths are the re-pricer's routes
        for b in tree.branches:
            assert b.length_m == pytest.approx(lay.length_m(b.system))
            ti, p = design.systems[b.system].option
            assert b.r_ohm == pytest.approx(vm.cables[ti].r_ohm_per_m
                                            * b.length_m / p)
            assert b.b_s == pytest.approx(OMEGA * vm.cables[ti].c_f_per_m
                                          * b.length_m * p)
        for c in vm.corners:
            s = [0j] * tree.n_nodes
            for t in range(model.n):
                s[tree.turbine_node[t]] = complex(c.p_mw[t], c.q_mvar[t])
            lf = load_flow(tree, s, c.bus_pu)
            ref = _pandapower_voltages(tree, s, c.bus_pu)
            np.testing.assert_allclose([abs(v) for v in lf.v_pu], ref,
                                       atol=1e-7, rtol=0)
        chk = check_voltage(design, G, turb, model, vm)
        assert set(chk.corners) == {"H1", "L1", "H0", "N1"}
        parallel += any(s.option[1] > 1 for s in design.systems)
        stations += bool(design.junctions)
    assert parallel and stations


def test_parallel_cables_equal_pandapower_parallel_lines():
    """One system of p = 2 cables = pandapower's ``parallel = 2`` line."""
    cab = CableElectrical(0.16e-3, 0.113e-3, 0.31e-9)
    L, p = 3500.0, 2
    tree = ElectricalTree(20.0, [("root", 0), ("turbine", 0)],
                          [Branch(1, 0, cab.r_ohm_per_m * L / p,
                                  cab.x_ohm_per_m * L / p,
                                  OMEGA * cab.c_f_per_m * L * p)], {0: 1})
    lf = load_flow(tree, [0, 13.6 + 7.3j], 1.0107)
    net = pp.create_empty_network()
    b0, b1 = pp.create_bus(net, 20.0), pp.create_bus(net, 20.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0107)
    pp.create_line_from_parameters(net, b1, b0, L / 1e3, 0.16, 0.113, 310.0,
                                   1.0, parallel=p)
    pp.create_sgen(net, b1, p_mw=13.6, q_mvar=7.3)
    pp.runpp(net, tolerance_mva=1e-11)
    assert abs(lf.v_pu[1]) == pytest.approx(net.res_bus.vm_pu[b1], abs=1e-9)


def test_raster_designs_check_on_the_pricer_lengths():
    from pyorps.collector.raster import RasterCollector
    from pyorps.collector.raster_pricer import RasterStepPricer
    from pyorps.utils.neighborhood import get_neighborhood_steps

    rng = random.Random(5)
    done = 0
    for it in range(3):
        nrng = np.random.default_rng(40 + it)
        vals = nrng.integers(1, 6, size=(6, 7)).astype(np.uint16)
        steps = np.asarray(get_neighborhood_steps(rng.choice([1, 2]),
                                                  directed=True),
                           dtype=np.int8)
        n = 2
        turb = [int(x) for x in nrng.choice(vals.size, size=n,
                                            replace=False)]
        model = _model(rng, n)
        cell = 2.5
        rc = RasterCollector(vals, steps, cell, turb, model, engine="B",
                             drain_engine="python", keep_trace=True)
        mv = rc.run().ravel()
        pricer = RasterStepPricer(vals, steps, cell, turb)
        vm = _vmodel_for(model, rng, per_m=0.5)
        for g in np.flatnonzero(np.isfinite(mv))[::7]:
            design = rc.design(int(g))
            reprice(design, pricer, turb, model)
            tree = electrical_tree(design, pricer, turb, model, vm)
            total = sum(b.length_m for b in tree.branches)
            lay = layout(design, pricer, turb, model)
            assert total == pytest.approx(sum(lay.length_m(s) for s in
                                              range(len(design.systems))))
            chk = check_voltage(design, pricer, turb, model, vm)
            assert isinstance(chk.passed, bool)
            done += 1
    assert done >= 5


# ------------------------------------------------------------- taps


def _one_turbine(r_ohm, x_ohm, corners, taps=(20.0, 20.5, 21.0, 21.5, 22.0),
                 transformer=None, u_m=24.0):
    """A turbine behind one cable, as a two-node design on a 2-node graph."""
    from pyorps.collector import CableType, CollectorModel, Design, System

    model = CollectorModel(n=1, types=(CableType("c", 1e9, 0.0, 0.0),),
                           current_a=(0.0, 1.0), loss_weight=(0.0, 1.0))
    G = CollectorGraph(2, [(0, 1, 0.0, 1000.0)])
    design = Design(tnodes=[1, 0], tedges=[(1, 0, 0)], root_tnode=0,
                    turbine_tnode={0: 1},
                    systems=[System(1, (0, 1), ("turbine", 0), ("root", 0))],
                    junctions=[])
    vm = VoltageModel(u_kv=20.0,
                      cables=(CableElectrical(r_ohm / 1000, x_ohm / 1000,
                                              0.0),),
                      corners=corners, taps_kv=taps, u_m_kv=u_m,
                      transformer=transformer)
    return design, G, [0], model, vm


def _v_at(r_ohm, x_ohm, p, q, v0):
    zb = 400.0
    return _two_bus_closed_form(r_ohm / zb, x_ohm / zb, p, q, v0)


def test_a_far_turbine_passes_only_on_a_higher_tap():
    corners = (Corner("H1", (6.8,), (3.672,), 1.0107),
               Corner("L1", (6.8,), (-3.29,), 0.9893))
    r, x = 6.0, 6.0                       # ~ 1.12 p.u. at H1
    vh = _v_at(r, x, 6.8, 3.672, 1.0107)
    vl = _v_at(r, x, 6.8, -3.29, 0.9893)
    assert 1.10 < vh < 1.10 * 22 / 20 and vl * 20 / 22 > 0.9
    chk = check_voltage(*_one_turbine(r, x, corners))
    assert chk.passed
    assert 20.0 not in chk.feasible_taps[0]
    assert min(chk.feasible_taps[0]) == min(
        t for t in (20.0, 20.5, 21.0, 21.5, 22.0) if vh * 20 / t <= 1.10)
    assert chk.corners["H1"].v_lv[0][0] == pytest.approx(vh, abs=1e-9)


def test_high_and_low_corners_that_cannot_share_a_tap():
    # a reactive cable: H1 needs a tap >= ~20.9 kV, L1 one <= ~20.5 kV
    r, x = 1.0, 9.0
    hi_c = Corner("H1", (6.8,), (3.672,), 1.05)
    lo_c = Corner("L1", (6.8,), (-3.29,), 0.98)
    vh = _v_at(r, x, 6.8, 3.672, 1.05)
    vl = _v_at(r, x, 6.8, -3.29, 0.98)
    need_hi = [t for t in (20.0, 20.5, 21.0, 21.5, 22.0)
               if vh * 20 / t <= 1.10]
    need_lo = [t for t in (20.0, 20.5, 21.0, 21.5, 22.0)
               if vl * 20 / t >= 0.90]
    assert need_hi and need_lo and not set(need_hi) & set(need_lo)
    assert check_voltage(*_one_turbine(r, x, (hi_c,))).passed
    assert check_voltage(*_one_turbine(r, x, (lo_c,))).passed
    both = check_voltage(*_one_turbine(r, x, (hi_c, lo_c)))
    assert not both.passed
    assert both.feasible_taps[0] == ()
    assert both.margin_pu[0] < 0
    assert any("no tap" in v for v in both.violations)


def test_common_tap_policy_and_equipment_rating():
    from pyorps.collector import CableType, CollectorModel, Design, System

    # two turbines on their own cables: one near (needs a low tap at L1),
    # one far (needs a high tap at H1)
    model = CollectorModel(n=2, types=(CableType("a", 1e9, 0.0, 0.0),
                                       CableType("b", 1e9, 0.0, 0.0)),
                           current_a=(0, 1, 1, 2), loss_weight=(0, 1, 1, 4))
    G = CollectorGraph(3, [(1, 0, 0.0, 1000.0), (2, 0, 0.0, 1000.0)])
    design = Design(tnodes=[0, 1, 2], tedges=[(1, 0, 0), (2, 0, 1)],
                    root_tnode=0, turbine_tnode={0: 1, 1: 2},
                    systems=[System(1, (0, 1), ("turbine", 0), ("root", 0)),
                             System(2, (1, 1), ("turbine", 1), ("root", 0))],
                    junctions=[])
    corners = (Corner("H1", (6.8, 6.8), (3.672, 3.672), 1.0107),
               Corner("L1", (6.8, 6.8), (-3.29, -3.29), 0.92))
    vm = VoltageModel(u_kv=20.0,
                      cables=(CableElectrical(1e-5, 1e-5, 0.0),
                              CableElectrical(5e-3, 5e-3, 0.0)),
                      corners=corners,
                      taps_kv=(20.0, 20.5, 21.0, 21.5, 22.0), u_m_kv=24.0)
    per = check_voltage(design, G, [1, 2], model, vm)
    common = check_voltage(design, G, [1, 2], model, vm, tap_policy="common")
    assert per.passed and not common.passed
    assert per.tap_kv[0] < per.tap_kv[1]
    # the far turbine's MV terminal (~22.8 kV at H1) is above a 22 kV rating
    import dataclasses
    tight = dataclasses.replace(vm, u_m_kv=22.0)
    low = check_voltage(design, G, [1, 2], model, tight)
    assert not low.passed and low.um_margin_kv < 0
    assert any("exceeds U_m" in v for v in low.violations)


def test_transformer_adds_its_drop_on_the_lv_side():
    tr = TurbineTransformer(7.8, 0.09, 80.0)
    corners = (Corner("H1", (6.8,), (3.672,), 1.0107),)
    design, G, turb, model, vm = _one_turbine(1e-6, 1e-6, corners,
                                              transformer=tr)
    chk = check_voltage(design, G, turb, model, vm)
    v_mv = chk.corners["H1"].v_pu[1]
    zt = tr.z_pu * 20.0 ** 2 / 7.8 / 400.0          # at the 20 kV tap
    s = complex(6.8, 3.672)
    expect = abs(v_mv + zt * (s / v_mv).conjugate())
    assert chk.corners["H1"].v_lv[0][0] == pytest.approx(expect, rel=1e-12)
    # about 5 % above the MV terminal at full over-excited output
    assert 0.045 < chk.corners["H1"].v_lv[0][0] / v_mv - 1 < 0.055


# ------------------------------------------------------------- data


def test_model_from_the_yaml_carries_the_file_values():
    vm = voltage_model_from_yaml(YAML, u_kv=20, cable_mm2=[240, 630],
                                 n_turbines=7)
    assert vm.n == 7 and vm.u_m_kv == 24.0
    assert vm.taps_kv == (20.0, 20.5, 21.0, 21.5, 22.0)
    c240 = vm.cables[0]
    assert c240.r_ohm_per_m == pytest.approx(0.160e-3)
    assert c240.x_ohm_per_m == pytest.approx(OMEGA * 0.36e-6)
    assert c240.c_f_per_m == pytest.approx(0.31e-9)
    h1 = {c.name: c for c in vm.corners}["H1"]
    assert h1.p_mw[0] == pytest.approx(6.8)
    assert h1.q_mvar[0] == pytest.approx(0.540 * 6.8)
    assert h1.bus_pu == pytest.approx(1.0107)
    assert vm.transformer.z_pu.real == pytest.approx(80.0 / 7800.0)
    assert abs(vm.transformer.z_pu) == pytest.approx(0.09)
    assert vm.lv_band == pytest.approx((0.90, 1.10))
    assert len(vm.meta["sha256"]) == 64
    vm33 = voltage_model_from_yaml(YAML, u_kv=33, cable_mm2=[240],
                                   n_turbines=2)
    assert vm33.taps_kv[-1] == 33.0 and vm33.u_m_kv == 36.0
    assert vm33.meta["provisional"] == (vm33.meta["cable_class"] == "kv30")


def test_missing_33kv_cables_fall_back_and_flag(tmp_path):
    import yaml

    data = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    data["cables"].pop("kv33", None)
    f = tmp_path / "v.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    vm = voltage_model_from_yaml(f, u_kv=33, cable_mm2=[240], n_turbines=1)
    assert vm.meta["provisional"] and vm.meta["cable_class"] == "kv30"
    with pytest.raises(ValueError, match="kv33"):
        voltage_model_from_yaml(f, u_kv=33, cable_mm2=[240], n_turbines=1,
                                cable_class="kv33")

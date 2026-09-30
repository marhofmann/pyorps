"""Batched voltage checks (``pyorps.collector.voltage_batch``).

* a :class:`DesignBatch` built from traced designs holds exactly the
  branches of their electrical trees (graph and raster engines);
* every backend -- the compiled and the numpy sweep, Newton--Raphson on the
  inflated network and both power-grid-model styles -- gives the scalar
  sweep's voltages (<= 1e-9 p.u.), zero-length branches and parallel cables
  included;
* the batched check gives the single-design check's pass flags, taps and
  margins;
* bad batches (a cycle, a turbine not connected, a branch off the tree) are
  refused, and a design that does not converge is flagged, not passed.

The power-grid-model tests skip when the package is not installed.
"""
import math
import random

import numpy as np
import pytest

from pyorps.collector import CollectorGraph, solve_collector
from pyorps.collector.voltage import (
    Branch,
    ElectricalTree,
    check_tree,
    check_voltage,
    electrical_tree,
    load_flow,
    voltage_model_from_yaml,
)
from pyorps.collector.voltage_batch import (
    BACKENDS,
    DesignBatch,
    batch_voltages,
    check_batch,
    design_batch,
)

from .test_engine_vs_oracle_v2 import _grid, _model
from .test_voltage import YAML, _vmodel_for

OMEGA = 2 * math.pi * 50
HAS_PGM = True
try:
    import power_grid_model  # noqa: F401
except ImportError:
    HAS_PGM = False


def _random_trees(rng, B, n, J, vm, *, zero_p=0.03):
    """Random radial collectors: turbines and up to J junction slots hung
    one by one onto earlier nodes, km-scale cables of the model's types,
    p in {1, 2}, some zero-length branches."""
    trees = []
    for _ in range(B):
        j = rng.randint(0, J)
        labels = ([("root", 0)] + [("turbine", t) for t in range(n)]
                  + [("junction", k) for k in range(j)])
        order = list(range(1, len(labels)))
        rng.shuffle(order)
        placed, br = [0], []
        for node in order:
            par = rng.choice(placed)
            placed.append(node)
            if rng.random() < zero_p:
                br.append(Branch(node, par, 0.0, 0.0, 0.0))
                continue
            L = rng.uniform(200, 3000)
            c = rng.choice(vm.cables)
            p = rng.choice([1, 2])
            br.append(Branch(node, par, c.r_ohm_per_m * L / p,
                             c.x_ohm_per_m * L / p,
                             OMEGA * c.c_f_per_m * L * p))
        trees.append(ElectricalTree(vm.u_kv, labels, br,
                                    {t: 1 + t for t in range(n)}))
    return trees


def _scalar(trees, vm, n_slots, n):
    ref = np.full((len(vm.corners), len(trees), n_slots), np.nan,
                  dtype=complex)
    base = {"root": 0, "turbine": 1, "junction": 1 + n}
    for i, tr in enumerate(trees):
        slot = [base[k] + j for k, j in tr.labels]
        for c, cor in enumerate(vm.corners):
            s = [0j] * tr.n_nodes
            for t in range(n):
                s[tr.turbine_node[t]] = complex(cor.p_mw[t], cor.q_mvar[t])
            v = load_flow(tr, s, cor.bus_pu).v_pu
            for k in range(tr.n_nodes):
                ref[c, i, slot[k]] = v[k]
    return ref


@pytest.fixture(scope="module")
def farm():
    vm = voltage_model_from_yaml(YAML, u_kv=20, cable_mm2=[150, 240, 400,
                                                           630],
                                 n_turbines=7)
    trees = _random_trees(random.Random(9), 120, 7, 4, vm, zero_p=0.05)
    batch = DesignBatch.from_trees(trees, n=7)
    return vm, trees, batch, _scalar(trees, vm, batch.n_slots, 7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_equals_the_scalar_sweep(farm, backend):
    if backend.startswith("pgm") and not HAS_PGM:
        pytest.skip("power-grid-model not installed")
    vm, _trees, batch, ref = farm
    lf = batch_voltages(batch, vm, backend=backend)
    assert lf.converged.all()
    diff = np.abs(np.where(lf.used[None], lf.v_pu - ref, 0))
    assert np.nanmax(diff) <= 1e-9
    # slots a design does not use carry the busbar voltage, never NaN
    assert np.isfinite(lf.v_pu).all()


@pytest.mark.parametrize("backend", [b for b in BACKENDS
                                     if not b.startswith("pgm")]
                         + [pytest.param(b, marks=pytest.mark.skipif(
                             not HAS_PGM, reason="no power-grid-model"))
                            for b in BACKENDS if b.startswith("pgm")])
def test_batched_check_equals_the_single_design_check(farm, backend):
    vm, trees, batch, _ref = farm
    chk = check_batch(batch, vm, backend=backend)
    for i, tr in enumerate(trees):
        one = check_tree(tr, vm)
        assert bool(chk.passed[i]) == one.passed
        for t in range(7):
            assert chk.tap_kv[i, t] == one.tap_kv[t]
            assert chk.margin_pu[i, t] == pytest.approx(one.margin_pu[t],
                                                        abs=1e-9)
        assert chk.um_margin_kv[i] == pytest.approx(one.um_margin_kv,
                                                    abs=1e-7)
    rec = chk.record(0)
    assert rec["backend"] == backend and len(rec["tap_kv"]) == 7


def test_common_tap_policy_matches(farm):
    vm, trees, batch, _ref = farm
    chk = check_batch(batch, vm, tap_policy="common")
    for i, tr in enumerate(trees[:40]):
        one = check_tree(tr, vm, tap_policy="common")
        assert bool(chk.passed[i]) == one.passed
        assert len(set(chk.tap_kv[i])) == 1


def test_batch_from_traced_designs_holds_their_trees():
    rng = random.Random(21)
    designs, graphs, turbs, models = [], [], [], []
    for _ in range(6):
        N, E = _grid(3, 4, rng, rng.random() < 0.4)
        turb = rng.sample(range(N), 3)
        model = _model(rng, 3)
        G = CollectorGraph(N, E)
        res = solve_collector(G, turb, model, engine="B")
        for g in np.flatnonzero(np.isfinite(res.mv))[::4]:
            designs.append(res.design(int(g)))
            graphs.append(G)
            turbs.append(turb)
            models.append(model)
    assert len(designs) > 10
    for d, G, turb, model in zip(designs, graphs, turbs, models):
        vm = _vmodel_for(model, random.Random(3))
        batch = design_batch([d], G, turb, model, vm)
        tree = electrical_tree(d, G, turb, model, vm)
        assert batch.active.sum() == len(tree.branches)
        chk = check_batch(batch, vm)
        one = check_voltage(d, G, turb, model, vm)
        assert bool(chk.passed[0]) == one.passed
        np.testing.assert_allclose(chk.margin_pu[0],
                                   [one.margin_pu[t] for t in range(3)],
                                   atol=1e-9)


def test_array_round_trip_and_subset(farm):
    vm, _trees, batch, _ref = farm
    again = DesignBatch.from_arrays(batch.to_arrays(), n=7,
                                    n_slots=batch.n_slots, u_kv=20.0)
    np.testing.assert_array_equal(again.active, batch.active)
    sub = batch.subset([3, 5])
    assert len(sub) == 2 and sub.keys == [3, 5]
    lf = batch_voltages(sub, vm)
    full = batch_voltages(batch, vm)
    np.testing.assert_allclose(lf.v_pu, full.v_pu[:, [3, 5]], atol=1e-15)


def _one_design(edges, n=2, S=4):
    """A 1-design batch from ``[(u, v, r, x)]`` slot pairs."""
    E = S * (S - 1) // 2
    b = DesignBatch(n=n, n_slots=S, u_kv=20.0,
                    active=np.zeros((1, E), bool), r_ohm=np.zeros((1, E)),
                    x_ohm=np.zeros((1, E)), b_s=np.zeros((1, E)))
    for u, v, r, x in edges:
        e = b.edge_index[u, v]
        b.active[0, e] = True
        b.r_ohm[0, e], b.x_ohm[0, e] = r, x
    return b


def test_bad_batches_are_refused():
    vm = voltage_model_from_yaml(YAML, u_kv=20, cable_mm2=[240],
                                 n_turbines=2)
    cycle = _one_design([(0, 1, 1, 1), (1, 2, 1, 1), (0, 2, 1, 1)])
    with pytest.raises(ValueError, match="cycle"):
        batch_voltages(cycle, vm)
    alone = _one_design([(0, 1, 1, 1)])
    with pytest.raises(ValueError, match="turbine is not connected"):
        batch_voltages(alone, vm)
    off = _one_design([(0, 1, 1, 1), (0, 2, 1, 1), (2, 1, 0, 0)],
                      n=2, S=4)
    with pytest.raises(ValueError, match="cycle|connected"):
        batch_voltages(off, vm)
    with pytest.raises(ValueError, match="shape"):
        DesignBatch(n=2, n_slots=4, u_kv=20.0, active=np.zeros((1, 6), bool),
                    r_ohm=np.zeros((1, 5)), x_ohm=np.zeros((1, 6)),
                    b_s=np.zeros((1, 6)))


@pytest.mark.parametrize("backend", ["numba", "numpy", "inflated_nr"])
def test_a_design_that_does_not_converge_is_flagged(backend):
    vm = voltage_model_from_yaml(YAML, u_kv=20, cable_mm2=[240],
                                 n_turbines=2)
    ok = _one_design([(0, 1, 1.0, 1.0), (0, 2, 1.0, 1.0)])
    bad = _one_design([(0, 1, 1.0, 1.0), (0, 2, 400.0, 400.0)])
    both = DesignBatch(n=2, n_slots=4, u_kv=20.0,
                       active=np.vstack([ok.active, bad.active]),
                       r_ohm=np.vstack([ok.r_ohm, bad.r_ohm]),
                       x_ohm=np.vstack([ok.x_ohm, bad.x_ohm]),
                       b_s=np.vstack([ok.b_s, bad.b_s]))
    chk = check_batch(both, vm, backend=backend, max_iter=60)
    assert chk.converged.tolist() == [True, False]
    assert chk.passed[1] == np.False_
    assert np.isfinite(chk.margin_pu[0]).all()

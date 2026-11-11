"""Feasible incumbents for bound-and-prune (math.md M15, theorem T-F)."""
import dataclasses
import math

import numpy as np
import pytest

from pyorps.certify.feasible_incumbent import leg_length_interval_km, length_grid_km, make_feasible
from pyorps.collector.voltage_plant import (
    GridRules,
    HVLeg,
    NapCorner,
    PlantModel,
    UWTransformer,
    check_plant_tree,
)
from tests.test_collector.test_voltage_rv import _field, _vmodel


def _plant_for(L_km: float) -> PlantModel:
    leg = HVLeg.from_type(r_ohm_per_km=0.06, x_ohm_per_km=0.144, c_nf_per_km=144.0, length_km=L_km)
    return PlantModel(UWTransformer(63, 115, 21, 0.125, 0.0032, 22, 4e-4), (6.8, 6.8), 0.54, 0.484, hv_leg=leg)


@pytest.fixture(scope="module")
def farm():
    fld, pricer, turb, model = _field(5)
    vm = _vmodel(2)
    d0 = fld.engine_b.design(fld.argmin)
    base = GridRules("t", (NapCorner("S3", 1.05, 0.33, checks=frozenset({"no_frt"})),), frt_ns=(0.5, 2.0))
    from pyorps.collector.design import layout  # noqa: F401  (design layout is used inside make_feasible)
    from pyorps.collector.voltage import electrical_tree
    tree = electrical_tree(d0, pricer, turb, model, vm)
    top = check_plant_tree(tree, vm, _plant_for(5.0), base).corners["S3"].max_ns
    return fld, pricer, turb, model, vm, d0, base, top


GRID = np.array([2.0, 5.0, 8.0])


def test_a_passing_design_is_unchanged_and_costs_nothing_extra(farm):
    _fld, pricer, turb, model, vm, d0, base, _top = farm
    res = make_feasible(d0, pricer, turb, model, vm, base, _plant_for, GRID)
    assert res.feasible and res.extra_eur == 0.0 and res.design is d0
    assert res.record["rounds"] == 0 and res.record["grid_km"] == [2.0, 8.0, 3]


def test_a_failing_design_is_repaired_and_then_passes_at_every_length(farm):
    _fld, pricer, turb, model, vm, d0, base, top = farm
    # fails at some leg lengths only if the top varies
    rules = dataclasses.replace(base, frt_ns=(0.5, top + 0.02))
    rules = dataclasses.replace(base, frt_ns=(0.5, top - 1e-3))
    res = make_feasible(d0, pricer, turb, model, vm, rules, _plant_for, GRID)
    assert res.feasible and res.extra_eur >= 0.0 and res.record["repair_steps"]
    from pyorps.collector.voltage import electrical_tree
    tree = electrical_tree(res.design, pricer, turb, model, vm)
    assert all(check_plant_tree(tree, vm, _plant_for(L), rules).voltage_ok for L in GRID)


def test_a_limit_no_cable_can_meet_gives_infinite_extra_cost(farm):
    _fld, pricer, turb, model, vm, d0, base, _top = farm
    rules = dataclasses.replace(base, frt_ns=(0.0, 0.0))
    res = make_feasible(d0, pricer, turb, model, vm, rules, _plant_for, GRID)
    assert not res.feasible and math.isinf(res.extra_eur) and res.design is None
    assert res.record["violations"]


def test_repair_off_reports_the_failure_without_changing_the_design(farm):
    _fld, pricer, turb, model, vm, d0, base, top = farm
    rules = dataclasses.replace(base, frt_ns=(0.5, top - 1e-3))
    res = make_feasible(d0, pricer, turb, model, vm, rules, _plant_for, GRID, repair=False)
    assert not res.feasible and math.isinf(res.extra_eur) and res.record["rounds"] == 0


def test_leg_interval_brackets_every_route_and_the_grid_has_both_ends():
    taps = np.array([[0.0, 0.0], [10_000.0, 0.0]])
    lo, hi = leg_length_interval_km((3_000.0, 4_000.0), taps, 7.0e6, 700.0)
    assert lo == pytest.approx(5.0) and hi == pytest.approx(10.0)
    g = length_grid_km(lo, hi, 0.025)
    assert g[0] == lo and g[-1] == hi and np.max(np.diff(g)) <= 0.025 + 1e-12
    # a tap farther than the budget allows cannot make the interval empty
    lo2, hi2 = leg_length_interval_km((0.0, 0.0), np.array([[8_000.0, 0.0]]), 1.0e6, 1000.0)
    assert hi2 >= lo2

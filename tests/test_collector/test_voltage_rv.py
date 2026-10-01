"""Repair and relax-and-verify with the voltage limit (plan C5, 4.3 / 3).

* repair turns a failing traced design into one that passes, by upgrades
  that only lower impedance; the repaired design re-prices and never costs
  less than the (optimal) original; a limit no cable can meet is reported,
  not papered over;
* ``certify_voltage`` on a synthetic raster window: the unconstrained field
  is a lower bound per site; a binding limit that the argmin meets closes
  the certificate; an argmin that fails leaves a bracket that is never
  reported without an escalation; an escalation that closes it is accepted.
"""
import dataclasses
import math

import numpy as np
import pytest

from pyorps.collector import CableType, CollectorModel
from pyorps.collector.design import reprice
from pyorps.collector.exact import exact_collector_field
from pyorps.collector.raster import RasterCollector
from pyorps.collector.raster_pricer import RasterStepPricer
from pyorps.collector.voltage import (
    CableElectrical,
    Corner,
    VoltageModel,
    check_voltage,
)
from pyorps.collector.voltage_rv import certify_voltage, repair_design
from pyorps.utils.neighborhood import get_neighborhood_steps

INF = math.inf


def _model(n):
    """Three cable types: thin and cheap, medium, thick and dear; losses
    priced, so the unconstrained optimum uses thin cables on long runs."""
    types = (CableType("thin", 3.0, 0.3, 0.3),
             CableType("mid", 3.0, 0.6, 0.12),
             CableType("thick", 3.0, 1.2, 0.05))
    cur = tuple(float(bin(m).count("1")) for m in range(1 << n))
    return CollectorModel(n=n, types=types, current_a=cur,
                          loss_weight=tuple(c * c for c in cur),
                          loss_coef=0.05, derating=(1.0, 0.9, 0.8, 0.8),
                          sigma_eur_per_m=0.05, m_max=4, p_max=2,
                          station_building_eur=2.0, station_panel_eur=0.2,
                          bay_eur=0.5)


def _vmodel(n, *, bus=1.04, taps=(20.0,), u_m=24.0):
    """Impedances scaled so that a few tens of metres behave like km."""
    cables = (CableElectrical(0.30, 0.20, 0.0, "thin"),
              CableElectrical(0.12, 0.12, 0.0, "mid"),
              CableElectrical(0.05, 0.08, 0.0, "thick"))
    p = tuple(6.8 for _ in range(n))
    q = tuple(3.672 for _ in range(n))
    return VoltageModel(u_kv=20.0, cables=cables,
                        corners=(Corner("H1", p, q, bus),),
                        taps_kv=taps, u_m_kv=u_m)


def _window(seed, n=2, shape=(6, 8)):
    nrng = np.random.default_rng(seed)
    vals = nrng.integers(1, 4, size=shape).astype(np.uint16)
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    turb = [int(x) for x in nrng.choice(vals.size, size=n, replace=False)]
    return vals, steps, 2.5, turb


def _traced(seed, n=2):
    vals, steps, cell, turb = _window(seed, n)
    model = _model(n)
    rc = RasterCollector(vals, steps, cell, turb, model, engine="B",
                         drain_engine="python", keep_trace=True)
    mv = rc.run().ravel()
    pricer = RasterStepPricer(vals, steps, cell, turb)
    return rc, mv, pricer, turb, model


def _limit_for(designs, pricer, turb, model, n, quantile):
    """A band upper limit that a given share of the designs violate."""
    vm = _vmodel(n)
    top = []
    for d in designs:
        chk = check_voltage(d, pricer, turb, model, vm)
        top.append(max(c.max_v_pu for c in chk.corners.values()))
    return float(np.quantile(top, quantile))


def test_repair_passes_by_upgrades_and_never_costs_less():
    rc, mv, pricer, turb, model = _traced(3)
    roots = np.flatnonzero(np.isfinite(mv))[::3]
    designs = [rc.design(int(g)) for g in roots]
    hi = _limit_for(designs, pricer, turb, model, 2, 0.5)
    vm = dataclasses.replace(_vmodel(2), lv_band=(0.5, hi))
    repaired = failed = 0
    for g, d in zip(roots, designs):
        if check_voltage(d, pricer, turb, model, vm).passed:
            continue
        rep = repair_design(d, pricer, turb, model, vm)
        if rep.design is None:
            failed += 1
            continue
        repaired += 1
        assert rep.check.passed
        assert check_voltage(rep.design, pricer, turb, model, vm).passed
        cost, _ = reprice(rep.design, pricer, turb, model)
        assert cost == pytest.approx(rep.cost, rel=1e-12)
        assert rep.cost >= mv[g] * (1 - 1e-9)       # the original is optimal
        for _sid, old, new in rep.steps:
            z = [math.hypot(c.r_ohm_per_m, c.x_ohm_per_m) for c in vm.cables]
            assert z[new[0]] / new[1] < z[old[0]] / old[1]
    assert repaired > 0


def test_a_limit_no_cable_can_meet_is_reported():
    rc, mv, pricer, turb, model = _traced(3)
    g = int(np.flatnonzero(np.isfinite(mv))[0])
    d = rc.design(g)
    vm = _vmodel(2, bus=1.12)                       # the busbar alone > 1.1
    rep = repair_design(d, pricer, turb, model, vm)
    assert rep.design is None and not rep.passed
    assert rep.cost == INF


def _field(seed, n=2):
    vals, steps, cell, turb = _window(seed, n)
    model = _model(n)
    full = RasterCollector(vals, steps, cell, turb, model, engine="B",
                           drain_engine="python").run().ravel()
    budget = float(np.nanmax(np.where(np.isfinite(full), full, np.nan)))
    fld = exact_collector_field(vals, steps, cell, turb, model,
                                budget=budget, drain_engine="python")
    pricer = RasterStepPricer(vals, steps, cell, turb)
    return fld, pricer, turb, model


def _top(fld, g, pricer, turb, model):
    chk = check_voltage(fld.engine_b.design(int(g)), pricer, turb, model,
                        _vmodel(2))
    return max(c.max_v_pu for c in chk.corners.values())


def test_certificate_closes_when_the_argmin_passes():
    fld, pricer, turb, model = _field(5)
    g0 = fld.argmin
    tops = {int(g): _top(fld, g, pricer, turb, model)
            for g in np.flatnonzero(fld.exact.ravel())[::5]}
    hi = _top(fld, g0, pricer, turb, model) + 1e-6
    assert any(t > hi for t in tops.values())    # the limit binds elsewhere
    vm = dataclasses.replace(_vmodel(2), lv_band=(0.5, hi))
    cert = certify_voltage(fld, pricer, turb, model, vm)
    assert cert.rv.optimal
    assert cert.argmin == fld.argmin
    assert cert.rv.z_ub == pytest.approx(float(fld.total.ravel()[fld.argmin]))
    assert check_voltage(cert.design, pricer, turb, model, vm).passed
    assert cert.record["sites"][str(cert.argmin)]["passed"]
    assert cert.record["voltage_model"]["u_kv"] == 20.0


def test_a_failing_argmin_is_a_bracket_until_escalated():
    fld, pricer, turb, model = _field(5)
    g0 = fld.argmin
    d0 = fld.engine_b.design(g0)
    top = max(c.max_v_pu for c in
              check_voltage(d0, pricer, turb, model, _vmodel(2)).corners
              .values())
    vm = dataclasses.replace(_vmodel(2), lv_band=(0.5, top - 1e-4))
    assert not check_voltage(d0, pricer, turb, model, vm).passed
    with pytest.raises(RuntimeError, match="bracket"):
        certify_voltage(fld, pricer, turb, model, vm)
    seen = []

    def escalate(key, bracket):
        seen.append((key, bracket))
        return bracket[1], bracket[1]          # stub: the repair is optimal
    cert = certify_voltage(fld, pricer, turb, model, vm, escalate=escalate)
    assert seen and seen[0][0] == g0
    assert cert.rv.optimal
    assert check_voltage(cert.design, pricer, turb, model, vm).passed
    rec = cert.record["sites"][str(g0)]
    assert rec["passed"] is False and rec["repaired"] is True
    assert rec["repaired_cost"] >= float(fld.total.ravel()[g0]) * (1 - 1e-9)


def test_repair_and_certificate_with_the_plant_rules():
    """The whole plant at the NAP decides: a no-FRT limit just below the
    argmin's worst turbine fails it, thicker cables repair it, and the
    certificate records the plant and the rules."""
    from pyorps.collector.voltage_plant import (
        GridRules,
        NapCorner,
        PlantModel,
        UWTransformer,
        check_plant,
    )

    fld, pricer, turb, model = _field(5)
    vm = _vmodel(2)
    plant = PlantModel(UWTransformer(63, 115, 21, 0.125, 0.0032, 22, 4e-4),
                       (6.8, 6.8), 0.54, 0.484)
    base = GridRules("t", (NapCorner("S3", 1.05, 0.33,
                                     checks=frozenset({"no_frt"})),),
                     frt_ns=(0.5, 2.0))
    g0 = fld.argmin
    d0 = fld.engine_b.design(g0)
    top = check_plant(d0, pricer, turb, model, vm, plant, base) \
        .corners["S3"].max_ns
    rules = dataclasses.replace(base, frt_ns=(0.5, top - 1e-3))
    first = check_plant(d0, pricer, turb, model, vm, plant, rules)
    assert not first.voltage_ok
    rep = repair_design(d0, pricer, turb, model, vm, plant=plant,
                        rules=rules)
    assert rep.passed and rep.steps
    assert check_plant(rep.design, pricer, turb, model, vm, plant,
                       rules).voltage_ok
    assert rep.cost >= float(fld.mv.ravel()[g0]) * (1 - 1e-9)
    cert = certify_voltage(fld, pricer, turb, model, vm, plant=plant,
                           rules=rules,
                           escalate=lambda k, b: (b[1], b[1]))
    assert cert.rv.optimal
    assert cert.record["plant"]["p_b_inst_mw"] == pytest.approx(13.6)
    assert cert.record["rules"]["corners"][0]["name"] == "S3"
    assert cert.record["sites"][str(g0)]["repaired"] is True
    with pytest.raises(ValueError, match="both plant and rules"):
        repair_design(d0, pricer, turb, model, vm, plant=plant)

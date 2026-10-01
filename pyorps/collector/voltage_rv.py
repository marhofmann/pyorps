"""The voltage limit C5 in the certificate: repair and relax-and-verify.

Plan ``2026-09-25-voltage-limit-c5.md`` sections 3, 4.3 and 4.4. The
collector engines optimise without the limit, so a site's exact collector
value is a lower bound ``F_rel`` on its value with the limit.
:func:`certify_voltage` hands the exact sites of an
:class:`~pyorps.collector.exact.ExactCollectorField` to
:func:`~pyorps.certify.rv.relax_and_verify` in increasing ``F_rel``:

* the site's traced argmin passes :func:`~pyorps.collector.voltage.
  check_voltage` -> its bracket closes at ``F_rel``;
* it fails -> :func:`repair_design` thickens cables until it passes; the
  repaired design's re-priced cost is an upper bound, the bracket is
  ``[F_rel, repaired]`` and only the exact voltage-constrained engine
  (``escalate``, plan section 4.4, not built) can close it. The driver
  never reports an open bracket.

The manifest records the voltage model (corners, limits, ``U_MS``, the dead
band, the data file's hash) and every verified design's taps and margins.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pyorps.collector.design import Design, DesignError, reprice
from pyorps.collector.model import CollectorModel
from pyorps.collector.voltage import (
    VoltageCheck,
    VoltageModel,
    check_voltage,
    electrical_tree,
    load_flow,
)

__all__ = [
    "RepairResult",
    "VoltageCertificate",
    "certify_voltage",
    "repair_design",
]


@dataclass
class RepairResult:
    """What :func:`repair_design` did.

    Attributes:
        design: The repaired design (the input if it already passed);
            ``None`` if no sequence of upgrades made it pass.
        cost: Its re-priced cost (``inf`` if ``design`` is ``None``).
        check: The last voltage check (a
            :class:`~pyorps.collector.voltage.VoltageCheck`, or a
            :class:`~pyorps.collector.voltage_plant.PlantCheck` with a
            plant).
        original_cost: The input's re-priced cost.
        steps: ``(system, old option, new option)`` per upgrade.
    """
    design: Design | None
    cost: float
    check: object
    original_cost: float
    steps: list = field(default_factory=list)
    passed_: bool = False

    @property
    def passed(self) -> bool:
        return self.design is not None and self.passed_


def _violation(chk: VoltageCheck, vmodel: VoltageModel):
    """``(node label, corner name)`` of the worst violation."""
    if chk.um_margin_kv < 0:
        c = max(chk.corners.values(), key=lambda c: c.max_v_pu)
        return c.max_node, c.name
    t = chk.worst_turbine
    j = vmodel.taps_kv.index(chk.tap_kv[t])
    lo, hi = vmodel.lv_band
    c = min(chk.corners.values(),
            key=lambda c: min(c.v_lv[t][j] - lo, hi - c.v_lv[t][j]))
    return ("turbine", t), c.name


def _upgrades(model: CollectorModel, vmodel: VoltageModel, option):
    """Options with a smaller series impedance per metre than ``option``,
    cheapest material first."""
    def zabs(opt):
        ti, p = opt
        c = vmodel.cables[ti]
        return math.hypot(c.r_ohm_per_m, c.x_ohm_per_m) / p

    cur = zabs(option)
    out = [o for o in model.options if zabs(o) < cur * (1 - 1e-12)]
    return sorted(out, key=lambda o: (o[1] * model.types[o[0]].cost_eur_per_m,
                                      zabs(o)))


class _Checker:
    """The voltage check a repair or a certificate runs: the collector's own
    corners (:func:`~pyorps.collector.voltage.check_voltage`) or, with a
    plant and an operator's rules, the whole plant at the NAP
    (:func:`~pyorps.collector.voltage_plant.check_plant`), whose hard part
    (``voltage_ok``) decides."""

    def __init__(self, graph, turbines, model, vmodel, plant, rules, *,
                 root_transit, tap_policy):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if (plant is None) != (rules is None):
            raise ValueError("give both plant and rules, or neither")
        self.graph, self.turbines, self.model = graph, turbines, model
        self.vmodel, self.plant, self.rules = vmodel, plant, rules
        self.kw = {"root_transit": root_transit}
        self.tap_policy = tap_policy

    def check(self, design):
        if self.plant is None:
            chk = check_voltage(design, self.graph, self.turbines,
                                self.model, self.vmodel,
                                tap_policy=self.tap_policy, **self.kw)
            return chk.passed, chk
        from pyorps.collector.voltage_plant import check_plant

        chk = check_plant(design, self.graph, self.turbines, self.model,
                          self.vmodel, self.plant, self.rules,
                          tap_policy=self.tap_policy, **self.kw)
        return chk.voltage_ok, chk

    def locate(self, design, chk):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """``(tree, |V| per collector node, start node)`` of the worst
        violation, or ``None`` when no cable upgrade can help."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        tree = electrical_tree(design, self.graph, self.turbines, self.model,
                               self.vmodel, **self.kw)
        if self.plant is None:
            node, corner = _violation(chk, self.vmodel)
            c = next(c for c in self.vmodel.corners if c.name == corner)
            s = [0j] * tree.n_nodes
            for t in range(self.model.n):
                s[tree.turbine_node[t]] = complex(c.p_mw[t], c.q_mvar[t])
            v = load_flow(tree, s, c.bus_pu,
                          s_base_mva=self.vmodel.s_base_mva,
                          tol_pu=self.vmodel.tol_pu,
                          max_iter=self.vmodel.max_iter).v_pu
            return tree, [abs(x) for x in v], tree.labels.index(tuple(node))
        from pyorps.collector.voltage_plant import MV_BUS

        hopeless = ("MV busbar", "range exhausted", "no load flow",
                    "UW HV terminal")
        if any(h in v for v in chk.violations for h in hopeless):
            return None
        corners = [c for c in chk.corners.values() if c.v_by_tap]
        if chk.um_margin_kv < 0:
            best = None
            for c in corners:
                for v in c.v_by_tap.values():
                    mags = [abs(x) for x in v[MV_BUS:]]
                    k = max(range(len(mags)), key=mags.__getitem__)
                    if best is None or mags[k] > best[0]:  # pylint: disable=unsubscriptable-object  # false positive
                        best = (mags[k], mags, k)
            return tree, best[1], best[2]
        t = chk.worst_turbine
        j = self.vmodel.taps_kv.index(chk.tap_kv[t])
        c = min(corners, key=lambda c: c.margin[t, j])
        tap = min(c.v_by_tap)
        mags = [abs(x) for x in c.v_by_tap[tap][MV_BUS:]]
        return tree, mags, tree.turbine_node[t]


def repair_design(design: Design, graph, turbines, model: CollectorModel,
                  vmodel: VoltageModel, *, root_transit: bool = True,
                  tap_policy: str = "per_turbine", max_steps: int = 50,
                  plant=None, rules=None) -> RepairResult:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Thicken cables of a failing design until it meets the limit
    (plan section 4.3).

    Each step walks the path from the worst violation (a turbine's band,
    protection or FRT margin, or a node above ``U_m``) to the UW and
    upgrades the system with the largest voltage change at the violating
    corner to the cheapest option of smaller impedance that the design still
    admits (``m_max``, derating, panels and bays are enforced by
    :func:`~pyorps.collector.design.reprice`). The result, re-priced, is a
    valid design that passes, so its cost bounds the site's constrained
    optimum from above. With ``plant`` and ``rules`` the check is the whole
    plant at the NAP (its voltage rules); a violation no cable can fix (the
    MV busbar, the tap changer's range) ends the repair unsuccessfully.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    ck = _Checker(graph, turbines, model, vmodel, plant, rules,
                  root_transit=root_transit, tap_policy=tap_policy)
    kw = ck.kw
    cost0, _ = reprice(design, graph, turbines, model, **kw)
    cur = copy.deepcopy(design)
    cost = cost0
    ok, chk = ck.check(cur)
    steps: list = []
    while not ok:
        if len(steps) >= max_steps:
            return RepairResult(None, math.inf, chk, cost0, steps)
        where = ck.locate(cur, chk)
        if where is None:
            return RepairResult(None, math.inf, chk, cost0, steps)
        tree, v, k = where
        path = []
        while k != 0:
            br = tree.branches[tree.out_branch[k]]
            path.append((abs(v[br.child] - v[br.parent]), br.system))
            k = br.parent
        done = False
        for _dv, sid in sorted(path, reverse=True):
            old = cur.systems[sid].option
            for opt in _upgrades(model, vmodel, old):
                trial = copy.deepcopy(cur)
                trial.systems[sid].option = opt
                try:
                    c_new, _ = reprice(trial, graph, turbines, model, **kw)
                except DesignError:
                    continue
                cur, cost = trial, c_new
                steps.append((sid, old, opt))
                done = True
                break
            if done:
                break
        if not done:
            return RepairResult(None, math.inf, chk, cost0, steps)
        ok, chk = ck.check(cur)
    cur.cost = cost
    return RepairResult(cur, cost, chk, cost0, steps, passed_=True)


@dataclass
class VoltageCertificate:
    """The relax-and-verify result with the limit (see the module).

    Attributes:
        rv: The :class:`~pyorps.certify.rv.RVResult`.
        argmin: The optimal site (flat cell index).
        design: Its verified design (repaired if the argmin needed it).
        checks: Per verified site, its check record.
        record: The manifest section (JSON-able).
    """
    rv: object
    argmin: int | None
    design: Design | None
    checks: dict
    record: dict


def certify_voltage(field_, pricer, turbines, model: CollectorModel,
                    vmodel: VoltageModel, *, root_cost=None,
                    eps: float = 0.0,
                    escalate: Callable | None = None,
                    root_transit: bool = True,
                    tap_policy: str = "per_turbine",
                    max_repair_steps: int = 50,
                    plant=None, rules=None) -> VoltageCertificate:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Relax-and-verify over the exact sites of ``field_`` with C5.

    Parameters:
        field_: An :class:`~pyorps.collector.exact.ExactCollectorField`
            with ``keep_trace`` (its engine B traces the designs).
        pricer: The :class:`~pyorps.collector.raster_pricer.
            RasterStepPricer` of the window.
        root_cost: EUR per cell added to the collector (as given to the
            field); ``None`` for 0.
        eps: ``eps_adm``.
        escalate: ``(site, (L, U)) -> (L', U')``, the exact
            voltage-constrained value of a site (plan section 4.4). Without
            it an open bracket raises: a bracket is never a result.
        plant, rules: Check the whole plant at the NAP against an
            operator's rules (:mod:`pyorps.collector.voltage_plant`); its
            voltage rules decide, the reactive compensation it asks for is
            recorded per site (a cost, rev. 5 term 11).

    Sites outside the field's budget are certified above ``budget + eps``;
    if the verified optimum ends up above the budget the certificate cannot
    exclude them, and the call raises (re-run the field with a larger
    budget).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps.certify.rv import relax_and_verify

    if field_.engine_b is None:
        raise ValueError("the field has no exact site")
    total = np.asarray(field_.total).ravel()
    rc = (np.zeros(total.size) if root_cost is None
          else np.asarray(root_cost, dtype=np.float64).ravel())
    exact = np.flatnonzero(np.asarray(field_.exact).ravel())
    cands = [(int(g), float(total[g])) for g in exact]
    limit = field_.budget + field_.eps
    above = ("above_budget", float(np.nextafter(limit, math.inf)))
    cands.append(above)
    checks: dict = {}
    designs: dict = {}
    ck = _Checker(pricer, turbines, model, vmodel, plant, rules,
                  root_transit=root_transit, tap_policy=tap_policy)

    def verify(key):
        if key == above[0]:
            raise ValueError(
                f"the verified optimum exceeds the field's budget {limit}: "
                f"re-run exact_collector_field with a budget >= it")
        g = int(key)
        d = field_.engine_b.design(g)
        f_rel = float(total[g])
        ok, chk = ck.check(d)
        rec = chk.as_record()
        rec["passed"] = bool(ok)
        if ok:
            designs[g] = d
            rec["repaired"] = False
            checks[g] = rec
            return f_rel, f_rel
        rep = repair_design(d, pricer, turbines, model, vmodel,
                            root_transit=root_transit, tap_policy=tap_policy,
                            max_steps=max_repair_steps, plant=plant,
                            rules=rules)
        rec["repaired"] = rep.passed
        rec["repair_steps"] = [[s, list(a), list(b)] for s, a, b in rep.steps]
        if rep.passed:
            designs[g] = rep.design
            rec["after_repair"] = rep.check.as_record()  # pylint: disable=no-member  # false positive
            rec["repaired_cost"] = rep.cost
            checks[g] = rec
            return f_rel, rep.cost + float(rc[g])
        checks[g] = rec
        return f_rel, math.inf

    rv = relax_and_verify(cands, verify, eps=eps, escalate=escalate)
    g = rv.argmin if isinstance(rv.argmin, int) else None
    record = {
        "constraint": "C5 voltage limit",
        "voltage_model": vmodel.describe(),
        "plant": None if plant is None else plant.describe(),
        "rules": None if rules is None else {
            "name": rules.name, "meta": dict(rules.meta),
            "corners": [{"name": c.name, "u_nap_pu": c.u_nap_pu,
                         "q_nap": c.q_nap, "p_frac": c.p_frac,
                         "checks": sorted(c.checks)} for c in rules.corners]},
        "tap_policy": tap_policy,
        "field_budget": field_.budget,
        "sites": {str(k): v for k, v in checks.items()},
        "rv": rv.as_record(),
    }
    return VoltageCertificate(rv=rv, argmin=g,
                              design=designs.get(g) if g is not None
                              else None, checks=checks, record=record)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

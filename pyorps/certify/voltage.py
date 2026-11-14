"""The voltage limit C5 in the certificate: relax-and-verify.

Plan ``2026-09-25-voltage-limit-c5.md`` sections 3 and 4.4. The collector
engines optimise without the limit, so a site's exact collector value is a
lower bound ``F_rel`` on its value with the limit.
:func:`certify_voltage` hands the exact sites of an
:class:`~pyorps.collector.exact.ExactCollectorField` to
:func:`~pyorps.certify.rv.relax_and_verify` in increasing ``F_rel``:

* the site's traced argmin passes :func:`~pyorps.collector.voltage.
  check_voltage` -> its bracket closes at ``F_rel``;
* it fails -> :func:`~pyorps.collector.voltage_rv.repair_design` thickens
  cables until it passes; the repaired design's re-priced cost is an upper
  bound, the bracket is ``[F_rel, repaired]`` and only the exact
  voltage-constrained engine (``escalate``, plan section 4.4, not built) can
  close it. The driver never reports an open bracket.

The manifest records the voltage model (corners, limits, ``U_MS``, the dead
band, the data file's hash) and every verified design's taps and margins.

The repair and the voltage check it runs stay in
:mod:`pyorps.collector.voltage_rv`; the certificate lives here because it
needs :mod:`pyorps.certify.rv` and the collector must not import it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyorps.certify.rv import relax_and_verify
from pyorps.collector.design import Design
from pyorps.collector.model import CollectorModel
from pyorps.collector.voltage import VoltageModel
from pyorps.collector.voltage_rv import _Checker, repair_design
from pyorps.core.exceptions import InvalidInputError

__all__ = [
    "VoltageCertificate",
    "certify_voltage",
]


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
    if field_.engine_b is None:
        raise InvalidInputError("the field has no exact site")
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
            raise InvalidInputError(
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

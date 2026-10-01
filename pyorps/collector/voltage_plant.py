"""The voltage limit at the connection point: the whole plant (C5 with C6).

The grid code states its requirements at the 110 kV connection point (NAP):
a reactive range (Syna: variant 2, 0.41 over- / 0.33 under-excited Q/P_b,inst,
TAB Hochspannung 2026 p. 14) at a NAP voltage, plus the operator's planning
checks (Syna TAB, Anhang K). To check them, the collector of a traced
design is extended into the plant:

    NAP (slack) --HV leg-- UW HV --[ideal ratio n(tap)]--Z_T-- MV busbar
                                                            -- collector

Everything is referred to the MV side at the tap in use: the NAP voltage
``U_NAP / n``, the HV leg ``Z / n^2`` and ``B n^2``; the UW transformer's
short-circuit impedance is constant in ohms on the MV side (taps on the HV
winding), its magnetising branch a shunt at the HV terminal. The result is
a radial tree, solved by :func:`~pyorps.collector.voltage.load_flow`.

Per operating corner (a NAP voltage, a reactive target at the NAP, a turbine
output), for every tap of the on-load tap changer:

* the plant controller sets one reactive share ``kappa`` (Mvar per MW rated)
  for every turbine such that the NAP reactive power meets the target
  (false position on ``kappa`` within the turbines' capability; a target
  outside it is reported as a deficit: compensation needed);
* the tap changer holds the MV busbar within ``U_MS (1 +/- B)``: the taps
  whose busbar voltage lies in that band are the ones the controller may
  rest on, and each of them must pass; if none does, the range is exhausted
  and the tap nearest to the band is used; that is a violation (Avacon's
  TAB: the transformer must regulate the whole HV voltage band).

Checks per corner (the operator's rules, :class:`GridRules`):

* ``band``: the turbine's own LV band (950 V +/- 10 %) at its transformer
  tap, as in :func:`~pyorps.collector.voltage.evaluate_limits`;
* ``no_trip``: the turbine protection, ``0.80 < U_LV / U_NS < 1.25``
  (E.7), with ``U_NS = U_MS / ue`` -- the turbine's LV voltage referred to
  the MV side, divided by ``U_MS``, so independent of the turbine's tap;
* ``no_frt``: ``U_LV / U_NS < 1.08`` (and ``> 0.92``) -- no switch into
  fault-ride-through mode in undisturbed operation (Syna Anhang K);
* ``mv_upper``: the MV busbar below ``1.10 U_MS`` (E.7, U> at the MV side);
* ``no_load_q``: all turbines idle, the NAP reactive power within
  ``[-0.05, +0.02] P_b,inst`` (Syna Anhang K);
* always: ``U_m`` at every MV node and the reactive target met.

The voltage rules are hard (:attr:`PlantCheck.voltage_ok`); a reactive
target the turbines cannot meet, or a no-load reactive power above the
allowance, asks for compensation equipment -- a cost (rev. 5 term 11), kept
apart in :attr:`PlantCheck.reactive`.

Every turbine needs ONE transformer tap that passes every applicable check
at every corner and every admissible tap-changer position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyorps.collector.voltage import (
    Branch,
    ElectricalTree,
    LoadFlowError,
    VoltageModel,
    decide_taps,
    electrical_tree,
    load_flow,
)

__all__ = [
    "GridRules",
    "HVLeg",
    "NapCorner",
    "PlantCheck",
    "PlantCornerResult",
    "PlantModel",
    "UWTransformer",
    "check_plant",
    "check_plant_tree",
    "grid_rules_from_yaml",
    "plant_model_from_yaml",
    "plant_tree",
]

CHECKS = frozenset({"band", "no_trip", "no_frt", "mv_upper", "no_load_q"})


# ------------------------------------------------------------ equipment


@dataclass(frozen=True)
class UWTransformer:
    """The substation's HV/MV transformer with an on-load tap changer on
    the HV winding.

    ``uk``, ``vkr`` and ``i0`` are fractions of the rating; ``pfe_kw`` the
    iron losses. The short-circuit impedance is constant in ohms on the MV
    side; the tap changes the ratio ``u_hv (1 + tap step) / u_mv``.
    """
    s_r_mva: float
    u_hv_kv: float
    u_mv_kv: float
    uk: float
    vkr: float = 0.0
    pfe_kw: float = 0.0
    i0: float = 0.0
    tap_step_pct: float = 16.0 / 9.0
    tap_min: int = -9
    tap_max: int = 9

    def __post_init__(self):
        if not (self.s_r_mva > 0 and self.u_hv_kv > 0 and self.u_mv_kv > 0):
            raise ValueError("ratings must be > 0")
        if not 0 <= self.vkr < self.uk < 1:
            raise ValueError("need 0 <= vkr < uk < 1")
        if self.tap_min > 0 or self.tap_max < 0:
            raise ValueError("the tap range must contain the middle tap 0")

    @property
    def taps(self) -> range:
        return range(self.tap_min, self.tap_max + 1)

    def ratio(self, tap: int) -> float:
        """``U_HV(tap) / U_MV`` (kV per kV)."""
        return self.u_hv_kv * (1.0 + tap * self.tap_step_pct / 100.0) \
            / self.u_mv_kv

    @property
    def z_ohm_mv(self) -> complex:
        zb = self.u_mv_kv ** 2 / self.s_r_mva
        return complex(self.vkr, math.sqrt(self.uk ** 2 - self.vkr ** 2)) * zb

    @property
    def y_mag_s_mv(self) -> complex:
        """Magnetising admittance referred to the MV side (absorbing)."""
        g = self.pfe_kw / 1e3 / self.u_mv_kv ** 2
        y = self.i0 * self.s_r_mva / self.u_mv_kv ** 2
        b = math.sqrt(max(y * y - g * g, 0.0))
        return complex(g, -b)


@dataclass(frozen=True)
class HVLeg:
    """The customer's 110 kV connection from the UW to the NAP (totals)."""
    r_ohm: float = 0.0
    x_ohm: float = 0.0
    b_s: float = 0.0
    name: str = "co-located"

    @classmethod
    def from_type(cls, *, r_ohm_per_km: float, x_ohm_per_km: float,
                  c_nf_per_km: float, length_km: float, f_hz: float = 50.0,
                  name: str = "") -> HVLeg:
        L = float(length_km)
        return cls(r_ohm_per_km * L, x_ohm_per_km * L,
                   2 * math.pi * f_hz * c_nf_per_km * 1e-9 * L,
                   name or f"{L:g} km")


@dataclass(frozen=True)
class NapCorner:
    """An operating corner at the NAP.

    ``u_nap_pu``: NAP voltage over its nominal; ``q_nap``: reactive target
    over ``P_b,inst`` (positive = over-excited, delivered to the grid) or
    ``None`` for turbines at zero reactive power (the no-load case);
    ``p_frac``: turbine output over rating; ``checks``: a subset of
    :data:`CHECKS`.
    """
    name: str
    u_nap_pu: float
    q_nap: float | None
    p_frac: float = 1.0
    checks: frozenset = frozenset({"band", "mv_upper"})

    def __post_init__(self):
        bad = set(self.checks) - CHECKS
        if bad:
            raise ValueError(f"unknown checks {sorted(bad)}")
        if not (self.u_nap_pu > 0 and 0 <= self.p_frac <= 1):
            raise ValueError("need u_nap_pu > 0 and 0 <= p_frac <= 1")


@dataclass(frozen=True)
class GridRules:
    """One operator's requirements (see the module docstring)."""
    name: str
    corners: tuple[NapCorner, ...]
    trip_ns: tuple[float, float] = (0.80, 1.25)
    frt_ns: tuple[float, float] = (0.92, 1.08)
    mv_upper_ms: float = 1.10
    no_load_q: tuple[float, float] = (0.05, 0.02)      # (under, over)
    meta: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class PlantModel:
    """The plant around a collector.

    ``p_rated_mw`` per turbine; ``q_over_p`` / ``q_under_p`` the turbines'
    reactive capability (Mvar per MW rated, at the MV terminal);
    ``u_ms_pu`` the tap changer's setpoint (p.u. of the collector nominal)
    and ``deadband`` its band; ``p_b_inst_mw`` defaults to the sum of the
    ratings.
    """
    transformer: UWTransformer
    p_rated_mw: tuple[float, ...]
    q_over_p: float
    q_under_p: float
    hv_leg: HVLeg = HVLeg()
    u_nap_kv: float = 110.0
    u_ms_pu: float = 1.0
    deadband: float = 0.0107
    p_b_inst_mw: float | None = None
    tol_q_mvar: float = 1e-7
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def p_b_inst(self) -> float:
        return (float(sum(self.p_rated_mw)) if self.p_b_inst_mw is None
                else float(self.p_b_inst_mw))

    def describe(self) -> dict:
        t, h = self.transformer, self.hv_leg
        return {"transformer": {k: getattr(t, k) for k in (
                    "s_r_mva", "u_hv_kv", "u_mv_kv", "uk", "vkr", "pfe_kw",
                    "i0", "tap_step_pct", "tap_min", "tap_max")},
                "hv_leg": {"r_ohm": h.r_ohm, "x_ohm": h.x_ohm,
                           "b_s": h.b_s, "name": h.name},
                "p_rated_mw": list(self.p_rated_mw),
                "q_over_p": self.q_over_p, "q_under_p": self.q_under_p,
                "u_nap_kv": self.u_nap_kv, "u_ms_pu": self.u_ms_pu,
                "deadband": self.deadband, "p_b_inst_mw": self.p_b_inst,
                "meta": dict(self.meta)}


# ----------------------------------------------------------- the tree

NAP, UW_HV, MV_BUS = 0, 1, 2


def plant_tree(tree: ElectricalTree, plant: PlantModel, tap: int
               ) -> ElectricalTree:
    """The collector ``tree`` inside the plant at tap ``tap`` (module
    docstring). Node 0 is the NAP, 1 the UW HV terminal, 2 the MV busbar
    (the collector's root); collector node ``k`` becomes ``k + 2``."""
    tr = plant.transformer
    n = tr.ratio(tap)
    hv = plant.hv_leg
    zt = tr.z_ohm_mv
    off = MV_BUS
    branches = [Branch(UW_HV, NAP, hv.r_ohm / n ** 2, hv.x_ohm / n ** 2,
                       hv.b_s * n ** 2),
                Branch(MV_BUS, UW_HV, zt.real, zt.imag, 0.0)]
    for b in tree.branches:
        branches.append(Branch(b.child + off, b.parent + off, b.r_ohm,
                               b.x_ohm, b.b_s, system=b.system,
                               length_m=b.length_m, option=b.option))
    half = 0.5 * tr.y_mag_s_mv                 # pi model: half at each end
    shunts = {UW_HV: half, MV_BUS: half}
    for k, y in tree.shunts.items():
        shunts[k + off] = shunts.get(k + off, 0j) + y
    labels = [("nap", 0), ("uw_hv", 0)] + list(tree.labels)
    return ElectricalTree(u_kv=tree.u_kv, labels=labels, branches=branches,
                          turbine_node={t: k + off for t, k in
                                        tree.turbine_node.items()},
                          shunts=shunts)


# -------------------------------------------------------------- solving


@dataclass
class _TapRun:
    tap: int
    kappa: float
    deficit: float
    q_nap: float
    v: list
    v_mv: float
    error: str = ""


def _solve_tap(ptree, plant, vmodel, corner, tap, kappa0=None) -> _TapRun:  # pylint: disable=unused-argument
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """The controller's reactive share at one tap (false position)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    n = plant.transformer.ratio(tap)
    v_root = corner.u_nap_pu * plant.u_nap_kv / n / ptree.u_kv
    p = [corner.p_frac * pr for pr in plant.p_rated_mw]
    tn = [ptree.turbine_node[t] for t in range(len(p))]

    def run(kappa):
        s = [0j] * ptree.n_nodes
        for t, k in enumerate(tn):
            s[k] = complex(p[t], kappa * plant.p_rated_mw[t])
        lf = load_flow(ptree, s, v_root, s_base_mva=vmodel.s_base_mva,
                       tol_pu=vmodel.tol_pu, max_iter=vmodel.max_iter)
        return lf.s_root_mva.imag, lf

    try:
        if corner.q_nap is None or corner.p_frac == 0:
            q, lf = run(0.0)
            kappa, deficit = 0.0, 0.0
        else:
            target = corner.q_nap * plant.p_b_inst
            a, b = -plant.q_under_p, plant.q_over_p
            qa, lfa = run(a)
            qb, lfb = run(b)
            if qb < target - plant.tol_q_mvar:
                kappa, q, lf, deficit = b, qb, lfb, target - qb
            elif qa > target + plant.tol_q_mvar:
                kappa, q, lf, deficit = a, qa, lfa, target - qa
            else:
                fa, fb = qa - target, qb - target
                kappa, q, lf = (a, qa, lfa) if abs(fa) < abs(fb) else \
                    (b, qb, lfb)
                side = 0
                for _ in range(100):
                    if abs(q - target) <= plant.tol_q_mvar or b - a < 1e-13:
                        break
                    c = b - fb * (b - a) / (fb - fa)       # false position
                    qc, lfc = run(c)
                    fc = qc - target
                    kappa, q, lf = c, qc, lfc
                    if fc * fb > 0:
                        b, fb = c, fc
                        if side == -1:
                            fa *= 0.5                      # Illinois
                        side = -1
                    else:
                        a, fa = c, fc
                        if side == 1:
                            fb *= 0.5
                        side = 1
                deficit = 0.0
    except LoadFlowError as exc:
        return _TapRun(tap, math.nan, math.nan, math.nan, [], math.nan,
                       error=str(exc))
    return _TapRun(tap, kappa, deficit, q, lf.v_pu, abs(lf.v_pu[MV_BUS]))
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite


# -------------------------------------------------------------- results


@dataclass
class PlantCornerResult:
    """One corner of :func:`check_plant_tree`.

    ``taps``: the tap-changer positions checked (those the controller may
    rest on); ``oltc_exhausted``: no position reaches the band;
    ``kappa``: the controller's reactive share per checked tap (Mvar per
    MW rated); ``q_nap_mvar``, ``deficit_mvar`` likewise (a deficit > 0
    means the turbines cannot deliver the target: over-excited
    compensation needed; < 0: under-excited); ``v_mv_pu``: the busbar
    voltage per tap; ``max_ns`` / ``min_ns``: the extreme turbine
    ``U_LV / U_NS`` at the chosen turbine-transformer taps; ``max_hv_kv``:
    the UW HV terminal (information only: the grid itself runs up to 123 kV
    continuously and 127 kV for 30 min, so HV equipment is specified for
    that range whatever the collector does).
    """
    name: str
    u_nap_pu: float
    q_target_mvar: float | None
    taps: list[int]
    oltc_exhausted: bool
    kappa: dict
    q_nap_mvar: dict
    deficit_mvar: dict
    v_mv_pu: dict
    max_v_pu: float
    max_ns: float
    min_ns: float
    errors: list
    max_hv_kv: float = math.nan
    margin: np.ndarray | None = field(default=None, repr=False)
    v_by_tap: dict = field(default_factory=dict, repr=False)


@dataclass
class PlantCheck:
    """What :func:`check_plant_tree` found (see the module docstring).

    ``voltage_ok``: every voltage rule holds (turbine band, protection, FRT,
    MV busbar, ``U_m``, the tap changer's range) -- the hard part of C5.
    ``reactive_ok``: the reactive target at the NAP is met and the no-load
    reactive power is within the allowance; where not, ``reactive`` says
    what compensation is needed (``q_deficit_mvar`` at full output,
    ``reactor_mvar`` at no load) -- a cost of the design (rev. 5 term 11),
    not an infeasibility. ``passed`` = both.
    """
    passed: bool
    rules: str
    feasible_taps: dict
    tap_kv: dict
    margin_pu: dict
    um_margin_kv: float
    q_deficit_mvar: float
    no_load_q_mvar: float | None
    reactor_mvar: float
    corners: dict
    violations: list
    plant: dict
    voltage_ok: bool = True
    reactive_ok: bool = True
    reactive: list = field(default_factory=list)
    worst_turbine: int = -1

    def as_record(self) -> dict:
        return {
            "passed": self.passed, "rules": self.rules,
            "voltage_ok": self.voltage_ok, "reactive_ok": self.reactive_ok,
            "reactive": list(self.reactive),
            "worst_turbine": self.worst_turbine,
            "tap_kv": {str(k): v for k, v in self.tap_kv.items()},
            "feasible_taps": {str(k): list(v)
                              for k, v in self.feasible_taps.items()},
            "margin_pu": {str(k): v for k, v in self.margin_pu.items()},
            "um_margin_kv": self.um_margin_kv,
            "q_deficit_mvar": self.q_deficit_mvar,
            "no_load_q_mvar": self.no_load_q_mvar,
            "reactor_mvar": self.reactor_mvar,
            "corners": {c.name: {"u_nap_pu": c.u_nap_pu,
                                 "q_target_mvar": c.q_target_mvar,
                                 "taps": c.taps,
                                 "oltc_exhausted": c.oltc_exhausted,
                                 "kappa": {str(k): v for k, v in
                                           c.kappa.items()},
                                 "deficit_mvar": {str(k): v for k, v in
                                                  c.deficit_mvar.items()},
                                 "v_mv_pu": {str(k): v for k, v in
                                             c.v_mv_pu.items()},
                                 "max_v_pu": c.max_v_pu,
                                 "max_ns": c.max_ns, "min_ns": c.min_ns,
                                 "max_hv_kv": c.max_hv_kv,
                                 "errors": c.errors}
                        for c in self.corners.values()},
            "violations": list(self.violations),
            "plant": self.plant,
        }


# --------------------------------------------------------------- check


def check_plant_tree(tree: ElectricalTree, vmodel: VoltageModel,
                     plant: PlantModel, rules: GridRules, *,
                     tap_policy: str = "per_turbine") -> PlantCheck:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Check one collector inside its plant against an operator's rules."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    n = vmodel.n
    if len(plant.p_rated_mw) != n:
        raise ValueError("the plant and the voltage model differ in the "
                         "number of turbines")
    taps = np.asarray(vmodel.taps_kv, dtype=np.float64)
    K = len(taps)
    tr_t = vmodel.transformer
    zt_k = (np.zeros(K, dtype=np.complex128) if tr_t is None else
            np.array([tr_t.z_ohm_mv(t) for t in taps]) / vmodel.z_base_ohm)
    lo, hi = vmodel.lv_band
    u_ms = plant.u_ms_pu
    band_lo, band_hi = u_ms * (1 - plant.deadband), u_ms * (1 + plant.deadband)
    margin = np.full((n, K), np.inf)
    um_margin = math.inf
    q_def = 0.0
    no_load = None
    reactor = 0.0
    violations: list[str] = []
    reactive: list[str] = []
    results: dict[str, PlantCornerResult] = {}
    ns_by_corner: dict = {}
    ptrees = {tap: plant_tree(tree, plant, tap)
              for tap in plant.transformer.taps}
    tn = [ptrees[0].turbine_node[t] for t in range(n)]
    mv_nodes = list(range(MV_BUS, ptrees[0].n_nodes))
    for corner in rules.corners:
        runs = [_solve_tap(ptrees[tap], plant, vmodel, corner, tap)
                for tap in plant.transformer.taps]
        ok = [r for r in runs if not r.error]
        errors = [f"tap {r.tap}: {r.error}" for r in runs if r.error]
        inside = [r for r in ok if band_lo <= r.v_mv <= band_hi]
        exhausted = not inside
        if exhausted and ok:
            d = [max(band_lo - r.v_mv, r.v_mv - band_hi) for r in ok]
            best = min(d)
            inside = [r for r, di in zip(ok, d) if di <= best + 1e-12]
        if not inside:
            violations.append(f"corner {corner.name}: no load flow "
                              f"converged at any tap")
            continue
        if exhausted:
            violations.append(
                f"corner {corner.name}: the tap changer cannot hold the MV "
                f"busbar within U_MS (1 +/- {plant.deadband}) (range "
                f"exhausted; checked at tap {[r.tap for r in inside]})")
        target = (None if corner.q_nap is None
                  else corner.q_nap * plant.p_b_inst)
        max_v = 0.0
        max_hv = 0.0
        cm = np.full((n, K), np.inf)
        ns_hi = np.full((n, K), -np.inf)
        ns_lo = np.full((n, K), np.inf)
        s_t = np.array([complex(corner.p_frac * plant.p_rated_mw[t],
                                0.0) for t in range(n)])
        for r in inside:
            v = np.asarray(r.v, dtype=np.complex128)
            mags = np.abs(v[mv_nodes])
            max_v = max(max_v, float(mags.max()))
            um_margin = min(um_margin,
                            vmodel.u_m_kv - float(mags.max()) * vmodel.u_kv)
            max_hv = max(max_hv, abs(v[UW_HV])
                         * plant.transformer.ratio(r.tap) * vmodel.u_kv)
            if "mv_upper" in corner.checks and \
                    r.v_mv > rules.mv_upper_ms * u_ms:
                violations.append(
                    f"corner {corner.name} tap {r.tap}: MV busbar "
                    f"{r.v_mv:.4f} p.u. > {rules.mv_upper_ms} U_MS")
            if target is not None and abs(r.deficit) > plant.tol_q_mvar:
                q_def = max(q_def, abs(r.deficit))
                side = "over" if r.deficit > 0 else "under"
                reactive.append(
                    f"corner {corner.name} tap {r.tap}: the turbines cannot "
                    f"meet {target:+.2f} Mvar at the NAP ({side}-excited "
                    f"compensation of {abs(r.deficit):.2f} Mvar needed)")
            if "no_load_q" in corner.checks:
                lo_q = -rules.no_load_q[0] * plant.p_b_inst
                hi_q = rules.no_load_q[1] * plant.p_b_inst
                no_load = (r.q_nap if no_load is None
                           else max(no_load, r.q_nap, key=abs))
                if r.q_nap > hi_q + plant.tol_q_mvar:
                    reactor = max(reactor, r.q_nap - hi_q)
                    reactive.append(
                        f"corner {corner.name} tap {r.tap}: no-load NAP "
                        f"reactive power {r.q_nap:+.3f} Mvar exceeds "
                        f"+{hi_q:.3f} Mvar (over-excited): a shunt reactor "
                        f"of {r.q_nap - hi_q:.3f} Mvar is needed")
                if r.q_nap < lo_q - plant.tol_q_mvar:
                    reactive.append(
                        f"corner {corner.name} tap {r.tap}: no-load NAP "
                        f"reactive power {r.q_nap:+.3f} Mvar below "
                        f"{lo_q:.3f} Mvar (under-excited)")
            # per turbine and turbine tap
            vt = v[tn]
            st = s_t + 1j * r.kappa * np.asarray(plant.p_rated_mw)
            s_pu = st / vmodel.s_base_mva
            with np.errstate(divide="ignore", invalid="ignore"):
                cur = np.where(s_pu == 0, 0.0, np.conj(s_pu / vt))
            v_lv_mv = np.abs(vt[:, None] + zt_k[None, :] * cur[:, None])
            if "band" in corner.checks:
                v_lv = v_lv_mv * (vmodel.u_kv / taps)[None, :]
                cm = np.minimum(cm, np.minimum(v_lv - lo, hi - v_lv))
            ns = v_lv_mv / u_ms                              # U_LV / U_NS
            ns_hi = np.maximum(ns_hi, ns)
            ns_lo = np.minimum(ns_lo, ns)
            if "no_trip" in corner.checks:
                a, b = rules.trip_ns
                cm = np.minimum(cm, np.minimum(ns - a, b - ns))
            if "no_frt" in corner.checks:
                a, b = rules.frt_ns
                cm = np.minimum(cm, np.minimum(ns - a, b - ns))
        margin = np.minimum(margin, cm)
        results[corner.name] = PlantCornerResult(
            name=corner.name, u_nap_pu=corner.u_nap_pu,
            q_target_mvar=target, taps=[r.tap for r in inside],
            oltc_exhausted=exhausted,
            kappa={r.tap: r.kappa for r in inside},
            q_nap_mvar={r.tap: r.q_nap for r in inside},
            deficit_mvar={r.tap: r.deficit for r in inside},
            v_mv_pu={r.tap: r.v_mv for r in inside},
            max_v_pu=max_v, max_ns=math.nan, min_ns=math.nan, errors=errors,
            max_hv_kv=max_hv,
            margin=cm, v_by_tap={r.tap: r.v for r in inside})
        ns_by_corner[corner.name] = (ns_hi, ns_lo)
    if um_margin < 0:
        violations.append(f"an MV node exceeds U_m = {vmodel.u_m_kv} kV")
    lim = decide_taps(margin[None], np.array([um_margin]),
                      tap_policy=tap_policy)
    tlist = [float(t) for t in taps]
    feasible = {t: tuple(tlist[j] for j in range(K) if lim.feasible[0, t, j])
                for t in range(n)}
    tap_kv = {t: tlist[int(lim.choice[0, t])] for t in range(n)}
    marg = {t: float(lim.margin[0, t]) for t in range(n)}
    pick = np.asarray(lim.choice[0])
    for name, (hi_a, lo_a) in ns_by_corner.items():
        results[name].max_ns = float(hi_a[np.arange(n), pick].max())
        results[name].min_ns = float(lo_a[np.arange(n), pick].min())
    for t in range(n):
        if marg[t] < 0:
            violations.append(
                f"turbine {t}: no transformer tap meets the band and the "
                f"protection / FRT limits at every corner (best "
                f"{tap_kv[t]} kV, margin {marg[t]:.4f})")
    voltage_ok = not violations
    reactive_ok = not reactive
    worst = min(range(n), key=lambda t: marg[t]) if n else -1
    return PlantCheck(passed=voltage_ok and reactive_ok, rules=rules.name,
                      feasible_taps=feasible, tap_kv=tap_kv, margin_pu=marg,
                      um_margin_kv=um_margin, q_deficit_mvar=q_def,
                      no_load_q_mvar=no_load, reactor_mvar=reactor,
                      corners=results, violations=violations,
                      plant=plant.describe(), voltage_ok=voltage_ok,
                      reactive_ok=reactive_ok, reactive=reactive,
                      worst_turbine=worst)


def check_plant(design, graph, turbines, model, vmodel: VoltageModel,
                plant: PlantModel, rules: GridRules, *,
                root_transit: bool = True,
                tap_policy: str = "per_turbine") -> PlantCheck:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """:func:`check_plant_tree` on a traced design's electrical tree."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    tree = electrical_tree(design, graph, turbines, model, vmodel,
                           root_transit=root_transit)
    return check_plant_tree(tree, vmodel, plant, rules,
                            tap_policy=tap_policy)


# ----------------------------------------------------------------- YAML


def grid_rules_from_yaml(path, operator: str) -> GridRules:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """The rules of ``"syna"`` or ``"avacon"`` from ``voltage_2026.yaml``.

    Syna (TAB Hochspannung 2026): variant 2 at the ends of the NAP voltage
    ranges in which the full reactive power is due (over-excited 96-120 kV,
    under-excited 103-123 kV, continuous), the planning checks of Anhang K
    (stability 1: no trip; stability 3: no FRT entry) and the no-load
    reactive limit. Avacon (NT-10-32): its TAB fixes no variant, so the
    envelope of variants 1-3 (0.48 over-, 0.41 under-excited) at the same
    voltages, no trip there, and the rev. 5 plan's no-load reading of
    VDE-AR-N 4120.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import yaml

    from pyorps.io.provenance import sha256_file

    path = Path(path)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    gc = d["grid_code"]
    op = gc["operators"][operator]
    v4120 = gc["vde_ar_n_4120"]
    var = v4120["q_variants_at_pcc"]
    env = v4120["q_voltage_envelope_kv"]
    cont = v4120["pcc_voltage_range_kv"]["continuous"]["value"]
    un = 110.0
    o_lo, o_hi = (x / un for x in env["full_over_excited"]["value"])
    u_lo = env["full_under_excited"]["value"][0] / un
    u_hi = min(env["full_under_excited"]["value"][1], cont[1]) / un
    steady = {"band", "mv_upper"}
    if operator == "syna":
        vv = var[f"variant_{op['q_variant']['value']}"]
        over, under = vv["over_excited"], vv["under_excited"]
        steady_checks = frozenset(steady)
    else:
        over = max(var[k]["over_excited"] for k in var if
                   k.startswith("variant_"))
        under = max(var[k]["under_excited"] for k in var if
                    k.startswith("variant_"))
        steady_checks = frozenset(steady | {"no_trip"})
    corners = [
        NapCorner("Qover_lo", o_lo, over, 1.0, steady_checks),
        NapCorner("Qover_hi", o_hi, over, 1.0, steady_checks),
        NapCorner("Qunder_lo", u_lo, -under, 1.0, steady_checks),
        NapCorner("Qunder_hi", u_hi, -under, 1.0, steady_checks),
    ]
    prot = op.get("protection_e7", {}).get("turbines", {}).get("value")
    trip = (0.80, 1.25)
    if prot:
        trip = (float(prot["U<"][0]), float(prot["U>>"][0]))
    frt = (0.92, 1.08)
    if operator == "syna":
        pc = op["planning_checks"]
        for i, (u, q) in enumerate(pc["stability_1"]["value"]):
            corners.append(NapCorner(f"S1{'abcd'[i]}", float(u), float(q),
                                     1.0, frozenset({"no_trip", "band",
                                                     "mv_upper"})))
        for u, q in pc["stability_3_busbar"]["value"]:
            corners.append(NapCorner("S3", float(u), float(q), 1.0,
                                     frozenset({"no_frt", "band",
                                                "mv_upper"})))
    nl = op["no_load_q_limit"]["value"]
    corners.append(NapCorner("N0", 1.0, None, 0.0,
                             frozenset({"no_load_q", "band"})))
    mv = op.get("protection_e7", {}).get("mv_side", {}).get("value")
    mv_upper = float(mv["U>"][0]) if mv else 1.10
    return GridRules(
        name=operator, corners=tuple(corners), trip_ns=trip, frt_ns=frt,
        mv_upper_ms=mv_upper,
        no_load_q=(float(nl["under_excited"]), float(nl["over_excited"])),
        meta={"source_file": path.name, "sha256": sha256_file(path),
              "q_over": over, "q_under": under})


def plant_model_from_yaml(path, vmodel: VoltageModel, *,
                          hv_leg: HVLeg | None = None,
                          transformer: UWTransformer | None = None,
                          u_ms_pu: float | None = None,
                          deadband: float | None = None) -> PlantModel:
    """The plant around ``vmodel`` with the file's default UW transformer
    (``substation_transformer.plant_default``) and turbine capability."""
    import yaml

    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if transformer is None:
        t = d["substation_transformer"]["plant_default"]["value"]
        transformer = UWTransformer(
            s_r_mva=float(t["s_r_mva"]), u_hv_kv=float(t["u_hv_kv"]),
            u_mv_kv=float(t["u_mv_over_un"]) * vmodel.u_kv,
            uk=float(t["uk"]), vkr=float(t["vkr"]),
            pfe_kw=float(t["pfe_kw"]), i0=float(t["i0"]),
            tap_step_pct=float(t["tap_step_pct"]),
            tap_min=int(t["tap_min"]), tap_max=int(t["tap_max"]))
    p_rated = tuple(max(c.p_mw[t] for c in vmodel.corners)
                    for t in range(vmodel.n))
    meta = vmodel.meta
    return PlantModel(
        transformer=transformer, p_rated_mw=p_rated,
        q_over_p=float(meta.get("q_over_p", 0.540)),
        q_under_p=float(meta.get("q_under_p", 0.484)),
        hv_leg=hv_leg or HVLeg(),
        u_ms_pu=float(meta.get("u_ms_pu", 1.0) if u_ms_pu is None
                      else u_ms_pu),
        deadband=float(meta.get("deadband", 0.0107) if deadband is None
                       else deadband))

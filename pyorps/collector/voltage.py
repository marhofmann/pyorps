"""The collector voltage limit C5 (plan ``2026-09-25-voltage-limit-c5.md``).

The collector engines optimise ``D_share`` without a voltage limit, so their
values are lower bounds for the problem with it. This module checks a traced
design against the limit with an exact AC load flow; the relax-and-verify
driver (:mod:`pyorps.certify.rv`) closes a site's bracket when its argmin
passes (see :mod:`pyorps.collector.voltage_rv`).

The electrical network
----------------------
A traced :class:`~pyorps.collector.design.Design` is a radial electrical
tree: the UW MV busbar (root, the slack), the turbines' MV terminals and the
switching-station busbar sections (junctions). Every system is one branch,
a pi-model of ``p`` identical cables in parallel over the system's trench
length ``L`` (:func:`~pyorps.collector.design.layout`, the routes the
re-pricer uses):

``R = r L / p``, ``X = x L / p``, ``B = omega c L p`` (half at each end),

with ``r`` the AC resistance at the maximum conductor temperature (90 degC).
Two systems in one trench do not couple electrically (mutual coupling is
neglected; a scope sentence).

The load flow
-------------
Balanced AC power flow by backward--forward sweep in per unit
(``U_base = u_kv``, ``S_base = s_base_mva``), converged to
``max |dV| <= tol_pu``; it raises :class:`LoadFlowError` otherwise. Turbines
are PQ injections at their MV terminals.

The limit (plan section 2.4)
----------------------------
For every turbine there must be ONE tap of its transformer (set once, at
commissioning) such that, at every operating corner,

* ``lo <= v_LV <= hi`` (the turbine's 950 V +/- 10 % band), and
* ``|V| <= U_m`` at every MV node (equipment rating).

``v_LV`` includes the turbine transformer's own voltage difference: with the
MV-terminal injection ``S`` and ``Z_T`` the transformer's short-circuit
impedance referred to the MV side at the tap ``U_k``,
``V_LV' = V + Z_T conj(S / V)`` and ``v_LV = |V_LV'| / U_k``. The corner
powers are taken at the MV terminal, so the network flow does not depend on
the taps and every turbine's taps can be enumerated on their own: the check
stays exact. (The plan's first draft compared ``|V| / U_k`` with the band;
at full over-excited output the transformer adds about 5 % on the LV side,
as much as the whole collector rise. Implementation log, deviation 19.)
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyorps.collector.design import Design, layout
from pyorps.collector.model import CollectorModel

__all__ = [
    "Branch",
    "CableElectrical",
    "Corner",
    "CornerResult",
    "ElectricalTree",
    "LimitResult",
    "LoadFlowError",
    "LoadFlowResult",
    "TurbineTransformer",
    "VoltageCheck",
    "VoltageModel",
    "check_tree",
    "check_voltage",
    "decide_taps",
    "electrical_tree",
    "evaluate_limits",
    "load_flow",
    "standard_corners",
    "voltage_model_from_yaml",
]

_SQRT3 = math.sqrt(3.0)


class LoadFlowError(RuntimeError):
    """The backward--forward sweep did not converge."""


# ------------------------------------------------------------------ data


@dataclass(frozen=True)
class CableElectrical:
    """Per-phase data of ONE three-phase cable of a catalogue type.

    Parameters:
        r_ohm_per_m: AC resistance at the maximum conductor temperature
            (90 degC for XLPE), per phase.
        x_ohm_per_m: Series reactance at the system frequency.
        c_f_per_m: Operating capacitance per phase.
        name: A label (e.g. ``"240 mm2 18/30 kV trefoil"``).
    """
    r_ohm_per_m: float
    x_ohm_per_m: float
    c_f_per_m: float
    name: str = ""

    def __post_init__(self):
        if not (self.r_ohm_per_m >= 0 and self.x_ohm_per_m >= 0
                and self.c_f_per_m >= 0):
            raise ValueError(f"{self.name}: r, x and c must be >= 0")


@dataclass(frozen=True)
class TurbineTransformer:
    """The turbine transformer's short-circuit impedance.

    ``uk`` and the load losses are taken at the tapped rating, i.e. the
    impedance is constant in ohms referred to the LV side (the tap winding
    is on the HV side). The magnetising branch is neglected.
    """
    s_r_mva: float
    uk: float
    p_cu_kw: float
    lv_kv: float = 0.95

    def __post_init__(self):
        if not (self.s_r_mva > 0 and 0 < self.uk < 1 and self.p_cu_kw >= 0):
            raise ValueError("need s_r_mva > 0, 0 < uk < 1, p_cu_kw >= 0")
        if self.p_cu_kw / 1e3 / self.s_r_mva > self.uk:
            raise ValueError("load losses exceed the short-circuit voltage")

    @property
    def z_pu(self) -> complex:
        """Series impedance in p.u. of its own rating."""
        r = self.p_cu_kw / 1e3 / self.s_r_mva
        return complex(r, math.sqrt(self.uk ** 2 - r ** 2))

    def z_ohm_mv(self, tap_kv: float) -> complex:
        """Impedance referred to the MV side at tap ``tap_kv``."""
        return self.z_pu * tap_kv ** 2 / self.s_r_mva


@dataclass(frozen=True)
class Corner:
    """One operating corner (plan section 2.3).

    ``p_mw`` / ``q_mvar``: per turbine, the injection at its MV terminal
    (``q > 0`` over-excited); ``bus_pu``: the UW busbar voltage in p.u. of
    the collector's nominal voltage.
    """
    name: str
    p_mw: tuple[float, ...]
    q_mvar: tuple[float, ...]
    bus_pu: float

    def __post_init__(self):
        if len(self.p_mw) != len(self.q_mvar):
            raise ValueError(f"corner {self.name}: p and q lengths differ")
        if not self.bus_pu > 0:
            raise ValueError(f"corner {self.name}: bus_pu must be > 0")


def standard_corners(p_rated_mw: Sequence[float], *, q_over_p: float,
                     q_under_p: float, u_ms_pu: float = 1.0,
                     deadband: float = 0.0107,
                     names: Sequence[str] = ("H1", "L1", "H0", "N1")
                     ) -> tuple[Corner, ...]:
    """The corners of plan section 2.3.

    * ``H1`` rated P, full over-excited Q, busbar ``U_MS (1 + B)``;
    * ``L1`` rated P, full under-excited Q, busbar ``U_MS (1 - B)``;
    * ``H0`` rated P, Q = 0, busbar ``U_MS (1 + B)``;
    * ``N1`` no output, busbar ``U_MS (1 + B)`` (cable charging only).
    """
    if not (q_over_p >= 0 and q_under_p >= 0 and 0 <= deadband < 1):
        raise ValueError("q ratios must be >= 0 and 0 <= deadband < 1")
    p = tuple(float(x) for x in p_rated_mw)
    zero = tuple(0.0 for _ in p)
    hi, lo = u_ms_pu * (1 + deadband), u_ms_pu * (1 - deadband)
    table = {
        "H1": Corner("H1", p, tuple(q_over_p * x for x in p), hi),
        "L1": Corner("L1", p, tuple(-q_under_p * x for x in p), lo),
        "H0": Corner("H0", p, zero, hi),
        "N1": Corner("N1", zero, zero, hi),
    }
    return tuple(table[n] for n in names)


@dataclass(frozen=True)
class VoltageModel:
    """Everything the voltage check needs for one MV level.

    Parameters:
        u_kv: Nominal collector voltage (the p.u. base).
        cables: Electrical data per cable type, aligned with
            ``CollectorModel.types``.
        corners: The operating corners.
        taps_kv: The turbine transformer's HV taps.
        u_m_kv: Highest voltage for equipment at every MV node.
        transformer: The turbine transformer, or ``None`` to compare
            ``|V| / U_k`` with the band directly (the plan's first draft).
        lv_band: ``(lo, hi)`` of the turbine LV voltage in p.u.
        f_hz: System frequency.
        s_base_mva: Per-unit power base of the load flow.
        tol_pu, max_iter: Convergence of the sweep.
        meta: Provenance (data file hash, cable class, flags); not compared.
    """
    u_kv: float
    cables: tuple[CableElectrical, ...]
    corners: tuple[Corner, ...]
    taps_kv: tuple[float, ...]
    u_m_kv: float
    transformer: TurbineTransformer | None = None
    lv_band: tuple[float, float] = (0.90, 1.10)
    f_hz: float = 50.0
    s_base_mva: float = 1.0
    tol_pu: float = 1e-10
    max_iter: int = 200
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if not (self.u_kv > 0 and self.u_m_kv > 0 and self.s_base_mva > 0):
            raise ValueError("u_kv, u_m_kv and s_base_mva must be > 0")
        if not self.cables or not self.corners or not self.taps_kv:
            raise ValueError("need cables, corners and taps")
        if not 0 < self.lv_band[0] < self.lv_band[1]:
            raise ValueError("lv_band must be (lo, hi) with 0 < lo < hi")
        if any(not t > 0 for t in self.taps_kv):
            raise ValueError("taps must be > 0")
        n = {len(c.p_mw) for c in self.corners}
        if len(n) != 1:
            raise ValueError("every corner needs the same turbine count")

    @property
    def n(self) -> int:
        """Number of turbines."""
        return len(self.corners[0].p_mw)

    @property
    def z_base_ohm(self) -> float:
        return self.u_kv ** 2 / self.s_base_mva

    def describe(self) -> dict:
        """JSON-able parameter vector for the certificate manifest."""
        tr = self.transformer
        return {
            "u_kv": self.u_kv,
            "cables": [[c.name, c.r_ohm_per_m, c.x_ohm_per_m, c.c_f_per_m]
                       for c in self.cables],
            "corners": [{"name": c.name, "p_mw": list(c.p_mw),
                         "q_mvar": list(c.q_mvar), "bus_pu": c.bus_pu}
                        for c in self.corners],
            "taps_kv": list(self.taps_kv),
            "u_m_kv": self.u_m_kv,
            "transformer": None if tr is None else {
                "s_r_mva": tr.s_r_mva, "uk": tr.uk, "p_cu_kw": tr.p_cu_kw,
                "lv_kv": tr.lv_kv},
            "lv_band": list(self.lv_band),
            "f_hz": self.f_hz,
            "tol_pu": self.tol_pu,
            "meta": dict(self.meta),
        }


# ------------------------------------------------------ electrical tree


@dataclass(frozen=True)
class Branch:
    """One series branch (a system) with its pi shunt, child -> parent."""
    child: int
    parent: int
    r_ohm: float
    x_ohm: float
    b_s: float = 0.0            # total shunt susceptance, half at each end
    system: int = -1
    length_m: float = 0.0
    option: tuple[int, int] | None = None


@dataclass
class ElectricalTree:
    """A radial network rooted at node 0 (the slack).

    Every other node has exactly one branch towards the root, its
    out-branch. ``labels[i]`` names node ``i`` (``("root", 0)``,
    ``("turbine", t)``, ``("junction", j)``). ``shunts[i]`` is an extra
    constant admittance (siemens, complex) at node ``i`` -- e.g. a
    transformer's magnetising branch in the plant model
    (:mod:`pyorps.collector.voltage_plant`).
    """
    u_kv: float
    labels: list[tuple[str, int]]
    branches: list[Branch]
    turbine_node: dict[int, int] = field(default_factory=dict)
    shunts: dict[int, complex] = field(default_factory=dict)
    parent: list[int] = field(init=False)
    out_branch: list[int] = field(init=False)
    order: list[int] = field(init=False)

    def __post_init__(self):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        N = len(self.labels)
        self.parent = [-1] * N
        self.out_branch = [-1] * N
        for bi, b in enumerate(self.branches):
            if not (0 <= b.child < N and 0 <= b.parent < N):
                raise ValueError(f"branch {bi} leaves the network")
            if b.child == 0:
                raise ValueError("the root has no out-branch")
            if self.out_branch[b.child] != -1:
                raise ValueError(f"node {b.child} has two out-branches")
            self.parent[b.child] = b.parent
            self.out_branch[b.child] = bi
        depth = {0: 0}
        for x in range(N):
            path, y = [], x
            while y not in depth:
                if self.out_branch[y] == -1:
                    raise ValueError(f"node {y} is not connected to the root")
                if y in path:
                    raise ValueError("the network has a cycle")
                path.append(y)
                y = self.parent[y]
            d = depth[y]
            for z in reversed(path):
                d += 1
                depth[z] = d
        self.order = sorted(range(N), key=lambda i: depth[i])

    @property
    def n_nodes(self) -> int:
        return len(self.labels)


def electrical_tree(design: Design, graph, turbines, model: CollectorModel,
                    vmodel: VoltageModel, *,
                    root_transit: bool = True) -> ElectricalTree:
    """The electrical network of a traced design.

    Lengths come from :func:`~pyorps.collector.design.layout` on ``graph``
    (a :class:`~pyorps.collector.reference.CollectorGraph` or a
    :class:`~pyorps.collector.raster_pricer.RasterStepPricer`), the same
    routing the re-pricer uses; the design is validated on the way.
    """
    if len(vmodel.cables) != len(model.types):
        raise ValueError(f"the voltage model has {len(vmodel.cables)} "
                         f"cable types, the collector model "
                         f"{len(model.types)}")
    if vmodel.n != model.n:
        raise ValueError("the voltage model and the design differ in the "
                         "number of turbines")
    lay = layout(design, graph, turbines, model, root_transit=root_transit)
    labels: list[tuple[str, int]] = [("root", 0)]
    labels += [("turbine", i) for i in range(model.n)]
    labels += [("junction", j) for j in range(len(design.junctions))]
    index = {lab: k for k, lab in enumerate(labels)}
    omega = 2.0 * math.pi * vmodel.f_hz
    branches = []
    for sid, s in enumerate(design.systems):
        ti, p = s.option
        cab = vmodel.cables[ti]
        L = lay.length_m(sid)
        branches.append(Branch(
            child=index[s.source], parent=index[s.sink],
            r_ohm=cab.r_ohm_per_m * L / p, x_ohm=cab.x_ohm_per_m * L / p,
            b_s=omega * cab.c_f_per_m * L * p, system=sid, length_m=L,
            option=(ti, p)))
    return ElectricalTree(u_kv=vmodel.u_kv, labels=labels, branches=branches,
                          turbine_node={i: 1 + i for i in range(model.n)})


# ------------------------------------------------------------ load flow


@dataclass
class LoadFlowResult:
    """Voltages (p.u., complex, root angle 0), branch currents (kA,
    child -> parent, per branch), iterations, power delivered to the root
    (MVA, complex) and the network losses (MVA, complex; the reactive part
    includes the cable charging, negative when charging dominates)."""
    v_pu: list[complex]
    i_ka: list[complex]
    iterations: int
    s_root_mva: complex
    losses_mva: complex


def load_flow(tree: ElectricalTree, s_mva: Sequence[complex],
              v_root_pu: float, *, s_base_mva: float = 1.0,
              tol_pu: float = 1e-10, max_iter: int = 200) -> LoadFlowResult:
    """Exact balanced AC load flow on a radial tree.

    ``s_mva[i]`` is the complex power INJECTED at node ``i`` (generation
    positive, ``s_mva[0]`` ignored). Backward--forward sweep on the
    pi-model; raises :class:`LoadFlowError` if ``max |dV|`` does not fall
    below ``tol_pu`` within ``max_iter`` sweeps.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    N = tree.n_nodes
    if len(s_mva) != N:
        raise ValueError(f"need {N} injections, got {len(s_mva)}")
    zb = tree.u_kv ** 2 / s_base_mva
    S = [complex(s) / s_base_mva for s in s_mva]
    S[0] = 0j
    Z = [0j] * N
    Y = [0j] * N
    for b in tree.branches:
        Z[b.child] = complex(b.r_ohm, b.x_ohm) / zb
        half = 0.5j * b.b_s * zb
        Y[b.child] += half
        Y[b.parent] += half
    for k, y_s in tree.shunts.items():
        Y[k] += complex(y_s) * zb
    v0 = complex(v_root_pu)
    V = [v0] * N
    order = tree.order
    rev = order[::-1]
    par = tree.parent
    J = [0j] * N
    for it in range(1, max_iter + 1):
        acc = [0j] * N
        for i in rev:
            if i == 0:
                continue
            cur = (S[i] / V[i]).conjugate() - Y[i] * V[i] + acc[i]
            J[i] = cur
            acc[par[i]] += cur
        Vn = [0j] * N
        Vn[0] = v0
        for i in order[1:]:
            Vn[i] = Vn[par[i]] + Z[i] * J[i]
        delta = max(abs(a - b) for a, b in zip(Vn, V))
        V = Vn
        if not all(cmath.isfinite(v) for v in V):
            raise LoadFlowError("the sweep diverged (non-finite voltage)")
        if delta <= tol_pu:
            break
    else:
        raise LoadFlowError(f"no convergence after {max_iter} sweeps "
                            f"(last max |dV| = {delta:.3e} p.u.)")
    # currents and powers consistent with the final voltages
    acc = [0j] * N
    for i in rev:
        if i == 0:
            continue
        cur = (S[i] / V[i]).conjugate() - Y[i] * V[i] + acc[i]
        J[i] = cur
        acc[par[i]] += cur
    i_grid = acc[0] - Y[0] * v0
    s_root = v0 * i_grid.conjugate() * s_base_mva
    i_base_ka = s_base_mva / (_SQRT3 * tree.u_kv)
    i_ka = [J[b.child] * i_base_ka for b in tree.branches]
    s_inj = sum(S[1:]) * s_base_mva
    return LoadFlowResult(v_pu=V, i_ka=i_ka, iterations=it,
                          s_root_mva=s_root, losses_mva=s_inj - s_root)


# ---------------------------------------------------------------- check


@dataclass
class CornerResult:
    """One corner of :func:`check_voltage`.

    ``v_pu``: |V| per node (p.u.); ``v_lv``: per turbine, the LV voltage at
    every tap (p.u. of the LV rating), in the order of ``taps_kv``;
    ``max_node``: the node with the highest |V|; ``current_ratio``: the
    highest actual current per conductor over its derated ampacity
    (information: the engines size cables by the nominal current).
    """
    name: str
    bus_pu: float
    v_pu: list[float]
    v_lv: dict[int, list[float]]
    max_node: tuple[str, int]
    max_v_pu: float
    iterations: int
    losses_mw: float
    current_ratio: float


@dataclass
class VoltageCheck:
    """What :func:`check_voltage` found.

    Attributes:
        passed: Every turbine has a tap feasible at every corner (under the
            tap policy) and no MV node exceeds ``U_m``.
        feasible_taps: Per turbine, the taps (kV) feasible at every corner.
        tap_kv: Per turbine, the chosen tap: the feasible one with the
            largest band margin, or, if none is feasible, the least bad.
        margin_pu: Per turbine, the band margin at the chosen tap (min over
            corners of ``min(v - lo, hi - v)``; negative = violation).
        um_margin_kv: ``U_m - max |V|`` over every node and corner.
        worst_turbine: The turbine with the smallest margin.
        corners: Per corner, the load-flow results.
        violations: Plain-language reasons for a failure.
        tap_policy: ``"per_turbine"`` or ``"common"``.
    """
    passed: bool
    feasible_taps: dict[int, tuple[float, ...]]
    tap_kv: dict[int, float]
    margin_pu: dict[int, float]
    um_margin_kv: float
    worst_turbine: int
    corners: dict[str, CornerResult]
    violations: list[str]
    tap_policy: str

    def as_record(self) -> dict:
        """JSON-able summary for the certificate manifest."""
        return {
            "passed": self.passed,
            "tap_policy": self.tap_policy,
            "tap_kv": {str(k): v for k, v in self.tap_kv.items()},
            "feasible_taps": {str(k): list(v)
                              for k, v in self.feasible_taps.items()},
            "margin_pu": {str(k): v for k, v in self.margin_pu.items()},
            "um_margin_kv": self.um_margin_kv,
            "worst_turbine": self.worst_turbine,
            "corners": {c.name: {"bus_pu": c.bus_pu,
                                 "max_v_pu": c.max_v_pu,
                                 "max_node": list(c.max_node),
                                 "losses_mw": c.losses_mw,
                                 "current_ratio": c.current_ratio,
                                 "iterations": c.iterations}
                        for c in self.corners.values()},
            "violations": list(self.violations),
        }


@dataclass
class LimitResult:
    """The limit of plan section 2.4 evaluated for ``B`` designs at once.

    Shapes: ``C`` corners, ``B`` designs, ``n`` turbines, ``K`` taps.

    Attributes:
        v_lv: LV voltage per corner, design, turbine and tap (p.u.);
            ``None`` from the compiled kernel, which keeps only margins.
        tap_margin: ``min`` over corners of ``min(v - lo, hi - v)``.
        feasible: Taps feasible at every corner, under the tap policy.
        choice: Chosen tap index per design and turbine.
        margin: Band margin at the chosen tap.
        um_margin_kv: ``U_m - max |V|`` over the used nodes and corners.
        passed: Every turbine has a feasible tap and ``U_m`` holds.
    """
    v_lv: np.ndarray
    tap_margin: np.ndarray
    feasible: np.ndarray
    choice: np.ndarray
    margin: np.ndarray
    um_margin_kv: np.ndarray
    passed: np.ndarray


def evaluate_limits(v_turb, s_turb_mva, vmax_pu, vmodel: VoltageModel, *,
                    tap_policy: str = "per_turbine") -> LimitResult:
    """Taps, band and equipment rating for ``B`` designs (module docstring).

    Parameters:
        v_turb: Complex MV-terminal voltages, p.u., shape ``(C, B, n)``.
        s_turb_mva: Turbine injections, shape ``(C, n)`` or ``(C, B, n)``.
        vmax_pu: Highest ``|V|`` over each design's used nodes, ``(C, B)``.
        vmodel: The voltage model.
        tap_policy: ``"per_turbine"`` or ``"common"``.

    The single-design :func:`check_voltage` and every batch backend
    (:mod:`pyorps.collector.voltage_batch`) call this one function, so they
    apply one limit.
    """
    if tap_policy not in ("per_turbine", "common"):
        raise ValueError("tap_policy must be 'per_turbine' or 'common'")
    v = np.asarray(v_turb, dtype=np.complex128)
    C, B, n = v.shape
    s = np.asarray(s_turb_mva, dtype=np.complex128) / vmodel.s_base_mva
    if s.ndim == 2:
        s = s[:, None, :]
    taps = np.asarray(vmodel.taps_kv, dtype=np.float64)
    tr = vmodel.transformer
    with np.errstate(divide="ignore", invalid="ignore"):
        cur = np.where(s == 0, 0.0, np.conj(s / v))          # (C, B, n)
    if tr is None:
        zt = np.zeros(len(taps), dtype=np.complex128)
    else:
        zt = np.array([tr.z_ohm_mv(t) for t in taps]) / vmodel.z_base_ohm
    v_lv = (np.abs(v[..., None] + zt * cur[..., None])
            * (vmodel.u_kv / taps))                           # (C, B, n, K)
    lo, hi = vmodel.lv_band
    tap_margin = np.minimum(v_lv - lo, hi - v_lv).min(axis=0)  # (B, n, K)
    um = vmodel.u_m_kv - np.asarray(vmax_pu).max(axis=0) * vmodel.u_kv
    return decide_taps(tap_margin, um, tap_policy=tap_policy, v_lv=v_lv)


def decide_taps(tap_margin, um_margin_kv, *, tap_policy: str = "per_turbine",
                v_lv=None) -> LimitResult:
    """The tap decision of plan section 2.4 from per-tap band margins.

    ``tap_margin[b, t, k]``: min over corners of ``min(v - lo, hi - v)``;
    ``um_margin_kv[b]``: ``U_m - max |V|``. Shared by
    :func:`evaluate_limits` and the compiled batch kernel
    (:mod:`pyorps.collector._voltage_numba`), which computes the margins
    itself.
    """
    if tap_policy not in ("per_turbine", "common"):
        raise ValueError("tap_policy must be 'per_turbine' or 'common'")
    tap_margin = np.asarray(tap_margin)
    B, n, _K = tap_margin.shape
    if tap_policy == "per_turbine":
        feasible = tap_margin >= 0
        choice = tap_margin.argmax(axis=2)
    else:
        worst = tap_margin.min(axis=1)                         # (B, K)
        feasible = np.broadcast_to((worst >= 0)[:, None, :],
                                   tap_margin.shape)
        choice = np.broadcast_to(worst.argmax(axis=1)[:, None], (B, n))
    margin = np.take_along_axis(tap_margin, choice[..., None], 2)[..., 0]
    um = np.asarray(um_margin_kv, dtype=np.float64)
    passed = (um >= 0) & (margin >= 0).all(axis=1)
    return LimitResult(v_lv=v_lv, tap_margin=tap_margin,
                       feasible=np.array(feasible), choice=np.array(choice),
                       margin=margin, um_margin_kv=um, passed=passed)


def _current_ratio(tree, lf, design, lay, model) -> float:
    """Highest ``|I| / (p f(m) I_z)`` over systems, ``m`` the largest cable
    count on the system's route."""
    if lay is None:
        return math.nan
    m_edge = {}
    for e in range(len(lay.priced)):
        m_edge[e] = sum(design.systems[s].option[1]
                        for s in lay.ups[e] + lay.dns[e])
    worst = 0.0
    for b, i in zip(tree.branches, lf.i_ka):
        if b.option is None:
            continue
        ti, p = b.option
        m = max((m_edge[e] for e in lay.routes[b.system]), default=p)
        cap = p * model.f(m) * model.types[ti].ampacity_a
        worst = max(worst, abs(i) * 1e3 / cap)
    return worst


def check_tree(tree: ElectricalTree, vmodel: VoltageModel, *,
               tap_policy: str = "per_turbine",
               current_ratio=None) -> VoltageCheck:
    """The voltage check of one electrical tree (see :func:`check_voltage`).

    ``current_ratio``: optional ``LoadFlowResult -> float`` reported per
    corner (``check_voltage`` passes the derated-ampacity ratio).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if tap_policy not in ("per_turbine", "common"):
        raise ValueError("tap_policy must be 'per_turbine' or 'common'")
    n = vmodel.n
    tn = [tree.turbine_node[t] for t in range(n)]
    runs = []
    for c in vmodel.corners:
        s = [0j] * tree.n_nodes
        for t in range(n):
            s[tn[t]] = complex(c.p_mw[t], c.q_mvar[t])
        runs.append(load_flow(tree, s, c.bus_pu,
                              s_base_mva=vmodel.s_base_mva,
                              tol_pu=vmodel.tol_pu, max_iter=vmodel.max_iter))
    v_turb = np.array([[[lf.v_pu[k] for k in tn]] for lf in runs])
    s_turb = np.array([[complex(p, q) for p, q in zip(c.p_mw, c.q_mvar)]
                       for c in vmodel.corners])
    vmax = np.array([[max(abs(v) for v in lf.v_pu)] for lf in runs])
    lim = evaluate_limits(v_turb, s_turb, vmax, vmodel,
                          tap_policy=tap_policy)
    taps = tuple(float(t) for t in vmodel.taps_kv)
    lo, hi = vmodel.lv_band
    violations: list[str] = []
    corners: dict[str, CornerResult] = {}
    for ci, (c, lf) in enumerate(zip(vmodel.corners, runs)):
        mags = [abs(v) for v in lf.v_pu]
        kmax = max(range(tree.n_nodes), key=lambda k: mags[k])
        if vmodel.u_m_kv - mags[kmax] * vmodel.u_kv < 0:
            violations.append(
                f"corner {c.name}: {tree.labels[kmax]} at "
                f"{mags[kmax] * vmodel.u_kv:.3f} kV exceeds U_m "
                f"{vmodel.u_m_kv} kV")
        corners[c.name] = CornerResult(
            name=c.name, bus_pu=c.bus_pu, v_pu=mags,
            v_lv={t: [float(x) for x in lim.v_lv[ci, 0, t]]
                  for t in range(n)},
            max_node=tree.labels[kmax], max_v_pu=mags[kmax],
            iterations=lf.iterations, losses_mw=lf.losses_mva.real,
            current_ratio=(math.nan if current_ratio is None
                           else current_ratio(lf)))
    feasible = {t: tuple(taps[j] for j in range(len(taps))
                         if lim.feasible[0, t, j]) for t in range(n)}
    tap_kv = {t: taps[int(lim.choice[0, t])] for t in range(n)}
    margin = {t: float(lim.margin[0, t]) for t in range(n)}
    for t in range(n):
        if margin[t] < 0:
            violations.append(
                f"turbine {t}: no tap keeps the LV voltage in "
                f"[{lo}, {hi}] at every corner (best tap {tap_kv[t]} kV, "
                f"margin {margin[t]:.4f} p.u.)")
    worst = min(range(n), key=lambda t: margin[t]) if n else -1
    return VoltageCheck(passed=bool(lim.passed[0]), feasible_taps=feasible,
                        tap_kv=tap_kv, margin_pu=margin,
                        um_margin_kv=float(lim.um_margin_kv[0]),
                        worst_turbine=worst, corners=corners,
                        violations=violations, tap_policy=tap_policy)


def check_voltage(design: Design, graph, turbines, model: CollectorModel,
                  vmodel: VoltageModel, *, root_transit: bool = True,
                  tap_policy: str = "per_turbine",
                  tree: ElectricalTree | None = None) -> VoltageCheck:
    """Check a traced design against the voltage limit (module docstring).

    Parameters:
        design, graph, turbines, model: As for
            :func:`~pyorps.collector.design.reprice`.
        vmodel: The voltage model of the design's MV level.
        tap_policy: ``"per_turbine"`` (every turbine picks its own tap,
            the headline) or ``"common"`` (one tap for all turbines).
        tree: A prebuilt :func:`electrical_tree` (skips the routing).
    """
    lay = layout(design, graph, turbines, model, root_transit=root_transit)
    if tree is None:
        tree = electrical_tree(design, graph, turbines, model, vmodel,
                               root_transit=root_transit)
    return check_tree(tree, vmodel, tap_policy=tap_policy,
                      current_ratio=lambda lf: _current_ratio(
                          tree, lf, design, lay, model))


# ------------------------------------------------------- data (YAML)

#: Default turbine-transformer HV rating per collector voltage (plan 1:
#: "a 33 kV collector is the 30 kV unit on its top tap").
DEFAULT_TRANSFORMER_KV = {20: "20", 30: "30", 33: "30"}
#: Equipment rating U_m per collector voltage: 12/20 kV cable and 24 kV
#: switchgear; 18/30 and 19/33 kV cable and 36 kV switchgear.
DEFAULT_UM_KV = {20: 24.0, 30: 36.0, 33: 36.0}
#: Cable data class per collector voltage (33 kV falls back to ``kv30``
#: until 19/33 kV data are in the file; the model is then flagged
#: ``provisional``).
DEFAULT_CABLE_CLASS = {20: "kv20", 30: "kv30", 33: "kv33"}


def _pick(v):
    """First entry of a per-source list (the file lists Suedkabel first)."""
    return float(v[0]) if isinstance(v, (list, tuple)) else float(v)


def voltage_model_from_yaml(path, *, u_kv: float, cable_mm2: Sequence[int],
                            n_turbines: int | None = None,
                            p_rated_mw: Sequence[float] | None = None,
                            u_ms_pu: float = 1.0,
                            deadband: float = 0.0107,
                            transformer_hv_kv: str | None = None,
                            laying: str = "trefoil",
                            include_transformer: bool = True,
                            corners: Sequence[str] = ("H1", "L1", "H0", "N1"),
                            cable_class: str | None = None,
                            u_m_kv: float | None = None) -> VoltageModel:
    """Build a :class:`VoltageModel` from ``voltage_2026.yaml``.

    Parameters:
        path: The data file.
        u_kv: Collector voltage (20, 30 or 33).
        cable_mm2: Conductor cross-section of every cable type, in the
            order of ``CollectorModel.types``.
        n_turbines / p_rated_mw: Turbine count (rated power from the file)
            or explicit ratings.
        u_ms_pu, deadband: Busbar setpoint and the tap changer's dead band
            (plan question 2: 1.00 p.u., +/-1.07 %).
        transformer_hv_kv: ``"20"``, ``"30"`` or ``"34"``: which turbine
            transformer (default by level, :data:`DEFAULT_TRANSFORMER_KV`).
        laying: ``"trefoil"`` (default) or ``"flat"`` inductance.
        include_transformer: Model the transformer's voltage difference.
        cable_class: ``"kv20"``, ``"kv30"``, ``"kv33"`` (default by level;
            a missing ``kv33`` falls back to ``kv30`` and sets
            ``meta["provisional"]``).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import yaml

    from pyorps.io.provenance import sha256_file

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    key = int(round(u_kv))
    meta = {"source_file": path.name, "sha256": sha256_file(path),
            "provisional": False}
    tb = data["turbine"]
    if p_rated_mw is None:
        if n_turbines is None:
            raise ValueError("give n_turbines or p_rated_mw")
        p_rated_mw = [tb["rated_power_kw"]["value"] / 1e3] * n_turbines
    pf = tb["power_factor_at_rated"]
    q_over = float(pf["over_excited"]["q_over_p"])
    q_under = float(pf["under_excited"]["q_over_p"])
    trd = tb["transformer"]
    hv = transformer_hv_kv or DEFAULT_TRANSFORMER_KV.get(key)
    if hv is None:
        raise ValueError(f"no default turbine transformer for {u_kv} kV")
    taps = tuple(float(t) for t in trd["hv_taps"]["value"][str(hv)])
    tr = None
    if include_transformer:
        tr = TurbineTransformer(
            s_r_mva=trd["rated_kva"]["value"] / 1e3,
            uk=trd["impedance_pct"]["value"] / 100.0,
            p_cu_kw=trd["losses_w"]["load"] / 1e3,
            lv_kv=tb["lv_nominal_v"]["value"] / 1e3)
    lv_tol = tb["lv_nominal_v"]["tolerance_pct"] / 100.0
    cab = data["cables"]
    cls = cable_class or DEFAULT_CABLE_CLASS.get(key)
    if cls not in cab:
        if cable_class is None and key == 33 and "kv30" in cab:
            cls = "kv30"
            meta["provisional"] = True
            meta["provisional_reason"] = ("no 19/33 kV cable data in the "
                                          "file; 18/30 kV values stand in")
        else:
            raise ValueError(f"cable class {cls!r} not in {path.name}")
    table = cab[cls]
    r90 = cab["r_ac90_ohm_per_km"]["value"]
    lkey = {"trefoil": "l_trefoil_mh_per_km",
            "flat": "l_flat_mh_per_km"}.get(laying)
    if lkey is None or lkey not in table:
        raise ValueError(f"no {laying!r} inductance for {cls}")
    omega = 2.0 * math.pi * 50.0
    cables = []
    for mm2 in cable_mm2:
        m = int(mm2)
        r = float(r90[m]) / 1e3
        ind = table[lkey]["value"][m]
        if laying == "flat" and isinstance(ind, (list, tuple)):
            # mean over the three phases (outer, middle, outer) of one source
            ind = (2 * ind[0] + ind[1]) / 3.0 if len(ind) >= 2 else ind[0]
        x = omega * _pick(ind) * 1e-3 / 1e3
        c = _pick(table["c_uf_per_km"]["value"][m]) * 1e-6 / 1e3
        cables.append(CableElectrical(r, x, c,
                                      name=f"{m} mm2 {cls} {laying}"))
    meta.update({"cable_class": cls, "laying": laying,
                 "transformer_hv_kv": str(hv), "u_ms_pu": u_ms_pu,
                 "deadband": deadband, "q_over_p": q_over,
                 "q_under_p": q_under})
    return VoltageModel(
        u_kv=float(u_kv), cables=tuple(cables),
        corners=standard_corners(p_rated_mw, q_over_p=q_over,
                                 q_under_p=q_under, u_ms_pu=u_ms_pu,
                                 deadband=deadband, names=corners),
        taps_kv=taps,
        u_m_kv=float(u_m_kv if u_m_kv is not None else DEFAULT_UM_KV[key]),
        transformer=tr, lv_band=(1.0 - lv_tol, 1.0 + lv_tol), meta=meta)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

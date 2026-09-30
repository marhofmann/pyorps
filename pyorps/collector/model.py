"""The trench-sharing MV collector model ``D_share`` (plan rev. 5, section 3.2).

This module states the model once, as data, so that the DP engine
(:mod:`pyorps.collector.reference`), the independent re-pricer
(:mod:`pyorps.collector.design`) and the test oracles all price the same
thing. It follows plan revision 5 plus the user's decisions of 2026-09-24
(``docs/superpowers/plans/2026-09-24-rev5-implementation-log.md``).

Instance
--------
* An undirected graph: nodes ``0..N-1`` (raster cells), edges
  ``(u, w, c, l)`` with trench cost ``c >= 0`` (EUR, already capitalised)
  and length ``l > 0`` (m).
* Turbines ``A = [a_0, ..., a_{n-1}]``, distinct nodes. Turbine nodes are
  NO-TRANSIT: no trench passes through one.
* The UW root ``g`` may be any non-turbine node; one run prices every
  ``g`` at once. Collector trenches may cross ``g`` (plan MATH-05).

Cables
------
A *system* (one electrical connection between two nodes of the network)
carries the turbine set ``X`` below it and uses one cable option
``kappa = (type, p)``: ``p`` identical parallel cables of one type
(user decision K1c, 2026-09-24). A system keeps its option between its
two end nodes (user decision T2). For a system carrying ``X`` in a trench
step that holds ``m`` cables in total (the sum of ``p`` over every system
crossing the step, in either direction):

* feasible iff ``m <= m_max`` and ``p * f(m) * I_z(type) >= I(X)``,
  ``f`` the group-derating factor, non-increasing in ``m``;
* cost per metre ``rho(X, kappa) = p * cost(type)
  + loss_coef * (r(type) / p) * W(X)``,
  ``W(X)`` the loss weight of the turbine set (the category square sum of
  plan section 2.1, MVA^2 h per year) and ``loss_coef`` the capitalised
  value per ohm and unit of W, i.e. ``Lambda / U^2`` in EUR / (ohm MVA^2 h);
* the step pays its trench once plus ``sigma`` per additional cable:
  ``c(e) + [sum over systems of rho + sigma * (m - 1)] * l(e)``.

Nodes
-----
* **Turbine** ``t``: one out-system carrying ``{t}`` plus everything that
  flows in; in-systems either arrive from below on their own trench edges
  or come down its trench from above. The out-system's current must stay
  within the turbine switchgear rating, ``I(X_out) <= switchgear_a``, and
  the connected conductors (the sum of ``p`` over the out-system and every
  in-system) must not exceed ``ring_panels``. Each connected conductor
  costs ``turbine_panel_eur``.
* **Switching station** (user decision K1, 2026-09-24): a trench-tree node
  off the turbines that hosts one or more junctions (busbar sections). A
  junction has at least two input systems and one output system carrying
  their union; the output may use any option. The station costs
  ``station_building_eur`` ONCE, plus ``station_panel_eur`` per conductor
  connected to any of its junctions (inputs and output of each). Stations
  are optional: without one, cables do not join (there are no plain
  joints) and simply share trenches.
* **Root** ``g``: every system that ends at ``g`` arrives on a trench edge
  into ``g`` and pays ``bay_eur`` per conductor; each conductor's current
  ``I(X) / p`` must stay within ``bay_a``.

Scope sentences (printed with every result, plan section 2.1): the trench
network is a tree; parallel trenches in one cell are separate trenches and
do not derate each other; a system keeps one option between nodes; no
turbine's power crosses one trench twice in the same direction (R-once).

The loss weight and the currents are SET functions, so turbines may differ
in rating and in production (user decision T5). The fast engine A replaces
them by their minimum over sets of equal size, which is admissible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


__all__ = [
    "INF",
    "CableType",
    "CollectorModel",
    "all_submasks",
    "bits",
    "popcount",
    "set_partitions",
]

INF = math.inf


def popcount(mask: int) -> int:
    """Number of turbines in a bitmask."""
    return bin(mask).count("1")


@dataclass(frozen=True)
class CableType:
    """One cable type of the catalogue.

    Parameters:
        name: A label, e.g. ``"NA2XS(F)2Y 1x240 RM/25 18/30 kV"``.
        ampacity_a: Continuous current rating at the reference laying
            condition, before group derating (A).
        cost_eur_per_m: Capitalised cost of ONE three-phase cable of this
            type per metre of trench: material, laying, dielectric losses,
            O&M and end-of-life, everything except the trench and the load
            losses.
        r_ohm_per_m: AC resistance per phase per metre at the loss
            temperature (plan: 43.3 degC).
    """
    name: str
    ampacity_a: float
    cost_eur_per_m: float
    r_ohm_per_m: float

    def __post_init__(self):
        # Negative prices or resistances break both the engines' bound
        # (A <= B) and Dijkstra's non-negative weights (review 2026-09-24).
        if not self.ampacity_a > 0:
            raise ValueError(f"{self.name}: ampacity_a must be > 0")
        if not (self.cost_eur_per_m >= 0 and self.r_ohm_per_m >= 0):
            raise ValueError(f"{self.name}: cost and resistance must be >= 0")


@dataclass(frozen=True)
class CollectorModel:
    """One MV level of ``D_share``; see the module docstring.

    ``current_a`` and ``loss_weight`` are indexed by the turbine bitmask
    (length ``2**n``, entry 0 unused). Build them with
    :meth:`identical_turbines` or :meth:`from_turbines`.
    """
    n: int
    types: tuple[CableType, ...]
    current_a: tuple[float, ...]
    loss_weight: tuple[float, ...]
    loss_coef: float = 0.0
    derating: tuple[float, ...] = (1.0,)
    sigma_eur_per_m: float = 0.0
    m_max: int | None = 4
    p_max: int = 1
    switchgear_a: float = INF
    ring_panels: int = 3
    turbine_panel_eur: float = 0.0
    station_building_eur: float = 0.0
    station_panel_eur: float = 0.0
    allow_stations: bool = True
    bay_eur: float = 0.0
    bay_a: float = INF
    _options: tuple[tuple[int, int], ...] = field(init=False, repr=False,
                                                   compare=False)

    def __post_init__(self):
        full = 1 << self.n
        if len(self.current_a) != full or len(self.loss_weight) != full:
            raise ValueError(
                f"current_a and loss_weight need 2**n = {full} entries "
                f"(indexed by turbine bitmask)")
        if not self.types:
            raise ValueError("the catalogue is empty")
        if any(not c >= 0 for c in self.current_a):
            raise ValueError("current_a must be >= 0 (and not NaN)")
        if any(not w >= 0 for w in self.loss_weight):
            raise ValueError("loss_weight must be >= 0 (and not NaN)")
        if any(not f > 0 for f in self.derating):
            raise ValueError("derating factors must be > 0")
        if self.p_max < 1:
            raise ValueError("p_max must be >= 1")
        if not self.derating or self.derating[0] != 1.0:
            raise ValueError("derating[0] is f(1) and must be 1.0")
        if any(b > a for a, b in zip(self.derating, self.derating[1:])):
            raise ValueError("derating must be non-increasing in m")
        if self.m_max is not None and self.m_max < 1:
            raise ValueError("m_max must be >= 1 or None")
        for name in ("sigma_eur_per_m", "turbine_panel_eur",
                     "station_building_eur", "station_panel_eur", "bay_eur",
                     "loss_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        opts = tuple((ti, p) for p in range(1, self.p_max + 1)
                     for ti in range(len(self.types)))
        object.__setattr__(self, "_options", opts)

    # ------------------------------------------------------ constructors

    @classmethod
    def from_turbines(cls, *, rated_mw: Sequence[float], u_kv: float,
                      gamma: float = 1.10,
                      loss_weight: Callable[[int], float] | Sequence[float],
                      **kw) -> CollectorModel:
        """Currents from ratings: ``I(X) = gamma * sum P_r / (sqrt(3) U)``.

        ``loss_weight`` is a function of the bitmask or a ``2**n``
        sequence.
        """
        n = len(rated_mw)
        full = 1 << n
        cur = [0.0] * full
        for mask in range(1, full):
            p = sum(rated_mw[i] for i in range(n) if mask >> i & 1)
            cur[mask] = gamma * p * 1e6 / (math.sqrt(3.0) * u_kv * 1e3)
        if callable(loss_weight):
            w = [0.0] + [float(loss_weight(m)) for m in range(1, full)]
        else:
            w = [float(x) for x in loss_weight]
        return cls(n=n, current_a=tuple(cur), loss_weight=tuple(w), **kw)

    @classmethod
    def identical_turbines(cls, n: int, *, current_one_a: float,
                           loss_one: float = 1.0, **kw) -> CollectorModel:
        """Interchangeable turbines: ``I(X) = |X| I_1``, ``W(X) = |X|^2 W_1``
        (all turbines at the same output at the same time)."""
        full = 1 << n
        cur = tuple(popcount(m) * current_one_a for m in range(full))
        w = tuple(popcount(m) ** 2 * loss_one for m in range(full))
        return cls(n=n, current_a=cur, loss_weight=w, **kw)

    # ------------------------------------------------------------ basics

    @property
    def full(self) -> int:
        """Bitmask of all turbines."""
        return (1 << self.n) - 1

    @property
    def options(self) -> tuple[tuple[int, int], ...]:
        """Every cable option ``(type index, p)``."""
        return self._options

    def f(self, m: int) -> float:
        """Group-derating factor for ``m`` cables in one trench."""
        return self.derating[min(m, len(self.derating)) - 1]

    def m_cap(self) -> int:
        """Largest cable count that can ever be feasible in one trench."""
        if self.m_max is not None:
            return self.m_max
        return len(self.derating) + self.n * self.p_max

    def feasible(self, mask: int, option: tuple[int, int], m: int) -> bool:
        """Can a system carrying ``mask`` on ``option`` share a trench of
        ``m`` cables?"""
        ti, p = option
        if self.m_max is not None and m > self.m_max:
            return False
        return (p * self.f(m) * self.types[ti].ampacity_a
                >= self.current_a[mask] * (1.0 - 1e-12))

    def rho(self, mask: int, option: tuple[int, int]) -> float:
        """Per-metre cost of a system, excluding trench and sigma."""
        ti, p = option
        t = self.types[ti]
        return (p * t.cost_eur_per_m
                + self.loss_coef * (t.r_ohm_per_m / p) * self.loss_weight[mask])

    def rate(self, systems: Sequence[tuple[int, tuple[int, int]]]) -> float:
        """Per-metre cost of a trench step carrying ``systems``.

        ``systems`` are ``(mask, option)`` pairs, both directions together.
        Returns ``inf`` when the step is infeasible. The trench cost
        ``c(e)`` is NOT included.
        """
        if not systems:
            return 0.0
        m = sum(opt[1] for _mask, opt in systems)
        if self.m_max is not None and m > self.m_max:
            return INF
        tot = self.sigma_eur_per_m * (m - 1)
        for mask, opt in systems:
            if not self.feasible(mask, opt, m):
                return INF
            tot += self.rho(mask, opt)
        return tot

    def turbine_ok(self, out_mask: int, conductors: int) -> bool:
        """Switchgear current and ring-panel limit at a turbine."""
        return (self.current_a[out_mask] <= self.switchgear_a * (1 + 1e-12)
                and conductors <= self.ring_panels)

    def bay_ok(self, mask: int, option: tuple[int, int]) -> bool:
        """Can a system on ``option`` end at the UW (bay current)?"""
        return self.current_a[mask] / option[1] <= self.bay_a * (1 + 1e-12)

    # ----------------------------------------------- the candidate rule

    def candidate_options(self, mask: int) -> tuple[tuple[int, int], ...]:
        """Options a NEW system carrying ``mask`` needs to try.

        Plan section 3.2, per ``p`` (implementation log section 2.2): for a
        system with ``p`` parallel cables, the largest cable count on its
        route being ``m``, the cheapest type feasible at ``m`` is feasible
        on every step of the route (``f`` is non-increasing) and costs no
        more, and keeping ``p`` leaves every trench's count unchanged. So
        only ``kappa*_p(X, m)`` for ``m = p .. m_cap`` can be optimal.
        """
        out = []
        for p in range(1, self.p_max + 1):
            for m in range(p, self.m_cap() + 1):
                best, arg = INF, None
                for ti in range(len(self.types)):
                    opt = (ti, p)
                    if not self.feasible(mask, opt, m):
                        continue
                    c = self.rho(mask, opt)
                    if arg is None or c < best - 1e-12 * max(1.0, abs(best)):
                        best, arg = c, opt
                if arg is not None and arg not in out:
                    out.append(arg)
        return tuple(out)

    # ------------------------------------------- engine-A relaxations

    def min_over_size(self, values: Sequence[float]) -> tuple[float, ...]:
        """``v_min(k) = min over |X| = k of v(X)``, k = 0..n."""
        out = [INF] * (self.n + 1)
        out[0] = 0.0
        for mask in range(1, 1 << self.n):
            k = popcount(mask)
            out[k] = min(out[k], values[mask])
        return tuple(out)

    def interchangeable(self, rel: float = 1e-12) -> bool:
        """True when current and loss weight depend only on ``|X|`` -- the
        case where count tokens are an exact symmetry reduction."""
        imin = self.min_over_size(self.current_a)
        wmin = self.min_over_size(self.loss_weight)
        for mask in range(1, 1 << self.n):
            k = popcount(mask)
            for v, lo in ((self.current_a[mask], imin[k]),
                          (self.loss_weight[mask], wmin[k])):
                if abs(v - lo) > rel * max(1.0, abs(v)):
                    return False
        return True

    def describe(self) -> dict:
        """JSON-able parameter vector, for provenance records."""
        return {
            "n": self.n,
            "types": [[t.name, t.ampacity_a, t.cost_eur_per_m, t.r_ohm_per_m]
                      for t in self.types],
            "current_a": list(self.current_a),
            "loss_weight": list(self.loss_weight),
            "loss_coef": self.loss_coef,
            "derating": list(self.derating),
            "sigma_eur_per_m": self.sigma_eur_per_m,
            "m_max": self.m_max,
            "p_max": self.p_max,
            "switchgear_a": self.switchgear_a,
            "ring_panels": self.ring_panels,
            "turbine_panel_eur": self.turbine_panel_eur,
            "station_building_eur": self.station_building_eur,
            "station_panel_eur": self.station_panel_eur,
            "allow_stations": self.allow_stations,
            "bay_eur": self.bay_eur,
            "bay_a": self.bay_a,
        }


def all_submasks(mask: int):
    """Every submask of ``mask``, including 0 and ``mask``."""
    s = mask
    while True:
        yield s
        if s == 0:
            return
        s = (s - 1) & mask


def bits(mask: int):
    """Indices of the set bits of ``mask``, ascending."""
    i = 0
    while mask:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1


def set_partitions(items):
    """Every partition of a list into non-empty blocks."""
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


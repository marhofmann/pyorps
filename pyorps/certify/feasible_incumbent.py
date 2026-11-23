"""Feasible incumbents for bound-and-prune (paper math.md M15, theorem T-F).

``Z_UB`` of the prune is only sound if it is the cost of a FEASIBLE design: the plant must meet the voltage rules at the
point of connection (``check_plant_tree``: turbine band, no-trip, no-FRT, MV busbar, tap-changer range). The collector
optimum of engine B ignores those rules, so a priced incumbent is checked here, repaired by thicker cables if needed
(``repair_design``) and discarded (cost ``inf``) if no repair meets the rules.

The 110 kV leg length ``L`` is not known exactly for a pad whose leg was priced by a field. It lies in a closed interval
(``leg_length_interval_km``): at least the straight-line distance to the nearest connection point or tower, at most the
leg bill over the cheapest per-metre weight. The plant is checked at every length of a grid over that interval, so the
verdict holds for the unknown real route. Between grid points the checked quantities change by far less than the grid
step times their slope; the report gives the grid step, and the claim between grid points is [A], not proven.

Reactive shortfalls are priced, not failures (decision D7), and the price is outside F_model until the owner supplies
it: the record lists them, the verdict ignores them.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from pyorps.collector.design import reprice
from pyorps.collector.voltage import electrical_tree
from pyorps.collector.voltage_plant import check_plant_tree
from pyorps.collector.voltage_rv import repair_design

__all__ = ["FeasibleResult", "leg_length_interval_km", "length_grid_km", "make_feasible"]


@dataclass
class FeasibleResult:
    """Verdict on one priced design.

    Attributes:
        feasible: The (possibly repaired) design meets the voltage rules at every grid length.
        extra_eur: Repair cost added to the design's price (0 if it passed, ``inf`` if no repair works).
        design: The design that passed (``None`` if none).
        record: JSON-able facts: grid, rounds, steps, violations, reactive figures.
    """
    feasible: bool
    extra_eur: float
    design: object | None
    record: dict = field(default_factory=dict)


def leg_length_interval_km(pad_xy: Sequence[float], tap_xy: np.ndarray, hv_cost_ub_eur: float,
                           min_weight_eur_per_m: float) -> tuple[float, float]:
    """``[L_lo, L_hi]`` in km of the 110 kV route from the pad to a tap.

    ``L_lo`` is the distance to the nearest tap (every route is at least as long as the straight line);
    ``L_hi = hv_cost_ub / min_weight`` (the bill of a route is its length times at least the cheapest weight, and the
    cheapest route costs at most ``hv_cost_ub``). Both ends are admissible bounds of the length of the priced route.
    """
    tap = np.asarray(tap_xy, dtype=float).reshape(-1, 2)
    lo = float(np.min(np.hypot(tap[:, 0] - pad_xy[0], tap[:, 1] - pad_xy[1]))) / 1e3
    hi = float(hv_cost_ub_eur) / float(min_weight_eur_per_m) / 1e3
    return lo, max(hi, lo)


def length_grid_km(lo: float, hi: float, step_km: float = 0.025) -> np.ndarray:
    """Grid over ``[lo, hi]`` including both ends, spacing at most ``step_km``."""
    n = max(2, int(math.ceil((hi - lo) / step_km)) + 1)
    return np.linspace(lo, hi, n)


def _worst(records: list[dict]) -> dict:
    """Reactive figures over the grid (priced, outside F_model); ``None`` where the rules have no such corner."""
    def vals(key):
        return [r[key] for r in records if r.get(key) is not None]

    q, nl, rx = vals("q_deficit_mvar"), vals("no_load_q_mvar"), vals("reactor_mvar")
    return {
        "q_deficit_mvar_max": max(q) if q else None,
        "no_load_q_mvar_range": [min(nl), max(nl)] if nl else None,
        "reactor_mvar_max": max(rx) if rx else None,
    }


@dataclass(frozen=True)
class _GridContext:
    """The fixed inputs of one ``make_feasible`` call (everything except the design that changes between rounds)."""
    graph: object
    turbines: object
    model: object
    vmodel: object
    rules: object
    plant_for_leg: Callable[[float], object]
    grid_km: np.ndarray
    root_transit: bool
    tap_policy: str


def _grid_facts(ctx: _GridContext) -> list:
    """``[first, last, count]`` of the checked grid, as reported in every record."""
    grid_km = ctx.grid_km
    return [float(grid_km[0]), float(grid_km[-1]), len(grid_km)]


def _check_grid(ctx: _GridContext, cur) -> list:
    """The plant check of ``cur`` at every grid length."""
    tree = electrical_tree(cur, ctx.graph, ctx.turbines, ctx.model, ctx.vmodel, root_transit=ctx.root_transit)
    return [check_plant_tree(tree, ctx.vmodel, ctx.plant_for_leg(float(L)), ctx.rules, tap_policy=ctx.tap_policy)
            for L in ctx.grid_km]


def _passed_result(ctx: _GridContext, cur, cost0: float, rounds: int, steps: list, recs: list[dict]) -> FeasibleResult:
    """The verdict for a design that passed at every grid length."""
    cost, _ = reprice(cur, ctx.graph, ctx.turbines, ctx.model, root_transit=ctx.root_transit)
    return FeasibleResult(True, float(cost - cost0), cur, {
        "grid_km": _grid_facts(ctx),
        "rounds": rounds, "repair_steps": steps,
        "price_before_eur": float(cost0), "price_after_eur": float(cost), **_worst(recs)})


def _failed_result(ctx: _GridContext, rounds: int, steps: list, failing: list[int], recs: list[dict],
                   repair_note: str | None = None) -> FeasibleResult:
    """The verdict for a design that fails and is not (or cannot be) repaired."""
    first = failing[0]
    record = {
        "grid_km": _grid_facts(ctx),
        "rounds": rounds, "repair_steps": steps,
        "failing_lengths": len(failing), "first_failing_km": float(ctx.grid_km[first]),
        "violations": list(recs[first]["violations"][:6])}
    if repair_note is not None:
        record["repair"] = repair_note
    record.update(_worst(recs))
    return FeasibleResult(False, math.inf, None, record)


def make_feasible(design, graph, turbines, model, vmodel, rules, plant_for_leg: Callable[[float], object],
                  grid_km: np.ndarray, *, root_transit: bool = True, tap_policy: str = "per_turbine",
                  repair: bool = True, max_rounds: int = 6) -> FeasibleResult:
    """Check ``design`` at every leg length of ``grid_km``; repair it if any length fails.

    The repair is run at the first failing length and the whole grid is checked again, up to ``max_rounds`` times. A
    design that cannot be repaired gets ``extra_eur = inf`` and must never serve as ``Z_UB``. The returned extra cost is
    the re-priced cost of the final design minus that of the input.
    """
    # lizard forgives: public signature (12 parameters) is part of the API and pinned by the callers and tests
    ctx = _GridContext(graph, turbines, model, vmodel, rules, plant_for_leg, grid_km, root_transit, tap_policy)
    cost0, _ = reprice(design, graph, turbines, model, root_transit=root_transit)
    cur = design
    rounds, steps = 0, []
    while True:
        checks = _check_grid(ctx, cur)
        failing = [i for i, c in enumerate(checks) if not c.voltage_ok]
        recs = [c.as_record() for c in checks]
        if not failing:
            return _passed_result(ctx, cur, cost0, rounds, steps, recs)
        if not repair or rounds >= max_rounds:
            return _failed_result(ctx, rounds, steps, failing, recs)
        rep = repair_design(cur, graph, turbines, model, vmodel, root_transit=root_transit, tap_policy=tap_policy,
                            plant=plant_for_leg(float(grid_km[failing[0]])), rules=rules)
        rounds += 1
        if not rep.passed:
            return _failed_result(ctx, rounds, steps, failing, recs, "no sequence of upgrades passes")
        steps += [list(map(str, s)) for s in rep.steps]
        cur = rep.design

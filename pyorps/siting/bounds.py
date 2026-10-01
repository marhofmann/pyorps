"""Screening bounds, with their preconditions made executable.

Implements sections 3.2, 3.3 and 3.4 of
``docs/superpowers/plans/2026-09-20-generalized-free-siting-facility-chains.md``
(revision 3). Every function here states which direction it errs in,
because the plan's own review found two bounds pointing the wrong way
and a third whose preconditions were silently assumed.

The cheap overhead screen
-------------------------

An overhead line between two points costs at least what the terrain
under it costs, and it needs at least
``ceil(distance / max_span) - 1`` intermediate towers. So an ordinary
unconstrained :class:`~pyorps.graph.search_session.CostField` plus a
tower count is a valid lower bound on the constrained cost, and it is
free -- the field is already there.

The algebra is sound: ``_terrain_eur`` is term-for-term identical to the
field's edge sum, so the "trapezoidal versus category-length" worry that
an earlier revision raised was unfounded. But the tightening leaves zero
slack, and **three live conditions flip it**, which is why
:func:`assert_overhead_bound_preconditions` refuses rather than warns:

1. ``ignore_max_cost=False`` is a supported setting. Both searches then
   traverse ``65535`` cells, but the field CHARGES 65535 per metre where
   ``_terrain_eur(ignore=(IMPASSABLE,))`` charges zero -- so the "lower"
   bound sails far above the thing it bounds.
2. The neighbourhoods must nest: ``terrain(R) >= d_unconstrained``
   needs ``steps_constrained`` to be a SUBSET of ``steps_field``. An r1
   field overstates an r2 route by ``(sqrt(2) + 1) / sqrt(5) = 1.0797``,
   which breaks the bound at every price level. And the 110 kV profile
   is unusable with r1 anyway: 45 deg spacing exceeds its 40 deg hard
   angle limit.
3. There are two constrained costs and they differ by >= 560 kEUR. The
   kernel does not charge the two terminal towers; the report does. A
   bound has to name which one it bounds, so ``quantity`` is required.

For a tighter bound that actually solves the span and spacing problem
rather than counting towers, use
:func:`~pyorps.graph.tower_field.tower_field_bounds`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "OverheadScreenBound",
    "assert_overhead_bound_preconditions",
    "collection_cost_upper_bound",
    "lipschitz_stride_gap",
    "local_lipschitz_gap",
    "min_tower_cost_eur",
    "overhead_screen_lower_bound",
]


def min_tower_cost_eur(profile) -> float:
    """Cheapest any single intermediate tower can be, in EUR.

    Computed from the profile's own LUTs rather than from a remembered
    number, because the remembered number was wrong: the plan's review
    found 30 000 quoted where the true floor is 65 000, the difference
    being the ``350 EUR/m^2 x 100 m^2`` terrain floor that
    ``precompute_tower_terrain_costs`` already applies and the old
    figure discarded. Taking the minimum over the whole 65536-entry LUT
    also settles the "check ``terrain_cost_map`` has no lower key"
    caveat, since ``np.interp`` clamps outside the supplied keys and the
    LUT is what the kernel actually reads.

    Excluded cells are skipped: a tower cannot stand on one, so its
    price is not a floor on anything.
    """
    terrain = np.asarray(profile.precompute_tower_terrain_costs(),
                         dtype=np.float64)
    floor_terrain = float(terrain[:65535].min())
    types = profile.tower_cost_params.get("angle_types", {}) or {}
    floor_type = min((float(t["base_cost"]) for t in types.values()),
                     default=0.0)
    return floor_terrain + floor_type


def assert_overhead_bound_preconditions(*, ignore_max_cost: bool,
                                        field_steps, constrained_steps,
                                        quantity: str) -> dict:
    """Refuse an overhead screen whose preconditions do not hold.

    Raises:
        ValueError: any of the three conditions in this module's
            docstring fails. Each message says which one and why it
            flips the inequality, because "the bound was violated" three
            months later is not a debuggable symptom.

    Returns:
        A record of what was checked, to store beside the bound.
    """
    if quantity not in ("kernel", "reported"):
        raise ValueError(
            f"quantity must be 'kernel' or 'reported', got {quantity!r}. "
            f"The two differ by the two terminal towers -- >= 560 kEUR on "
            f"the shipped 110 kV profile -- so a bound that does not name "
            f"one bounds nothing in particular.")
    if not ignore_max_cost:
        raise ValueError(
            "this bound needs ignore_max_cost=True. With exclusion "
            "disabled both searches cross IMPASSABLE_CELL_COST cells, but "
            "the field CHARGES 65535 per metre there while _terrain_eur "
            "drops that category from its sum -- so the 'lower' bound "
            "lands far above the cost it is supposed to bound.")
    fs = {(int(a), int(b))
          for a, b in np.asarray(field_steps).reshape(-1, 2)[:, :2]}
    cs = {(int(a), int(b))
          for a, b in np.asarray(constrained_steps).reshape(-1, 2)[:, :2]}
    if not cs <= fs:
        missing = sorted(cs - fs)[:6]
        raise ValueError(
            f"the constrained search uses {len(cs - fs)} step(s) the field "
            f"does not, e.g. {missing}. terrain(route) >= d_unconstrained "
            f"needs the field's neighbourhood to CONTAIN the constrained "
            f"one; an r1 field overstates an r2 route by "
            f"(sqrt(2)+1)/sqrt(5) = 1.0797 and the bound fails at every "
            f"price level.")
    return {"ignore_max_cost": True, "quantity": quantity,
            "field_steps": len(fs), "constrained_steps": len(cs),
            "steps_nested": True}


class OverheadScreenBound:
    """A lower bound on the constrained cost, with its provenance."""

    def __init__(self, values, *, terrain_eur, tower_eur, n_towers,
                 quantity, checks):
        self.values = np.asarray(values, dtype=np.float64)
        self.terrain_eur = np.asarray(terrain_eur, dtype=np.float64)
        self.tower_eur = np.asarray(tower_eur, dtype=np.float64)
        self.n_towers = np.asarray(n_towers, dtype=np.int64)
        self.quantity = str(quantity)
        self.checks = dict(checks)

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        finite = np.isfinite(self.values)
        return (f"OverheadScreenBound({self.values.size} candidates, "
                f"bounds the {self.quantity} cost, "
                f"{int(finite.sum())} reachable)")


def overhead_screen_lower_bound(terrain_eur, distance_m, *, profile,
                                quantity: str, ignore_max_cost: bool,
                                field_steps, constrained_steps,
                                min_tower: float | None = None
                                ) -> OverheadScreenBound:
    """``terrain + (ceil(d / max_span) - 1) * min_tower`` [+ terminals].

    A valid lower bound on what an overhead line from the field's origin
    to each candidate can cost, given that the conductor's ground track
    cannot be cheaper than the unconstrained least-cost path and that it
    needs at least that many intermediate towers.

    ``ceil(d / max_span) - 1`` is sound and TIGHT -- it leaves no slack
    at all -- which is exactly why the preconditions are asserted rather
    than assumed. See :func:`assert_overhead_bound_preconditions`.

    Parameters:
        terrain_eur: Unconstrained field cost per candidate, in EUR.
            Under a quantised field this must be the LOWER end of the
            interval; understating here only loosens the bound, which is
            the safe direction for this particular inequality.
        distance_m: Straight-line distance origin to candidate. Using
            the EUCLIDEAN distance keeps the tower count a bound; the
            routed length would be longer and would overstate it.
        profile: Supplies ``max_span_m`` and the tower cost floor.
        quantity: ``"kernel"`` (terminals not charged) or ``"reported"``
            (they are, at ``terminal_tower_cost`` each).
        ignore_max_cost: The finder's setting, checked not trusted.
        field_steps, constrained_steps: The two neighbourhoods, checked
            for nesting.
        min_tower: Override for the tower floor; defaults to
            :func:`min_tower_cost_eur`.
    """
    checks = assert_overhead_bound_preconditions(
        ignore_max_cost=ignore_max_cost, field_steps=field_steps,
        constrained_steps=constrained_steps, quantity=quantity)
    if not profile.has_span_constraints:
        raise ValueError(
            f"profile {profile.name!r} has no max_span_m, so it implies no "
            f"minimum tower count and this bound degenerates to the "
            f"terrain term alone")

    terrain = np.asarray(terrain_eur, dtype=np.float64)
    dist = np.asarray(distance_m, dtype=np.float64)
    floor = min_tower_cost_eur(profile) if min_tower is None else float(
        min_tower)
    checks["min_tower_eur"] = floor

    n = np.maximum(0, np.ceil(dist / float(profile.max_span_m)) - 1.0)
    n = np.where(np.isfinite(dist), n, 0.0).astype(np.int64)
    towers = n.astype(np.float64) * floor
    if quantity == "reported":
        towers = towers + 2.0 * float(profile.terminal_tower_cost)
        checks["terminal_tower_eur"] = float(profile.terminal_tower_cost)
    return OverheadScreenBound(terrain + towers, terrain_eur=terrain,
                               tower_eur=towers, n_towers=n,
                               quantity=quantity, checks=checks)


# ======================================================================
# the stride
# ======================================================================

def lipschitz_stride_gap(stride_m: float, max_value_eur_per_m: float, *,
                         diagonal: bool = True) -> float:
    """How much a stride lattice can miss, in EUR. Never assume it is zero.

    Evaluating candidates every ``stride_m`` metres certifies the
    LATTICE, not the space: nothing bounds the positions in between.
    That gap is what BSSS / BTST geometric branch and bound exists to
    close (Hansen et al. 1985; Drezner & Suzuki, Oper. Res. 52(1),
    2004), and a cost field admits the simplest version of it, because
    it is 1-Lipschitz in its own metric::

        d(a', b) >= d(a, b) - L * ||a - a'||

    with ``L`` the most a single metre of travel can cost. Moving the
    facility by at most half a lattice diagonal therefore changes its
    field cost by at most ``L * stride * sqrt(2) / 2``.

    ``L`` taken as the global maximum traversable cell value is valid
    but loose; :func:`local_lipschitz_gap` takes it over the
    neighbourhood each candidate could move within, which is the version
    worth reporting.

    Parameters:
        stride_m: Lattice spacing.
        max_value_eur_per_m: Largest traversable cell value, i.e. the
            most one metre of travel can cost. Do NOT pass the exclusion
            sentinel; an excluded cell is not traversable, so it bounds
            nothing.
        diagonal: Reach to the furthest point of a square lattice cell
            (half the diagonal). ``False`` uses half the stride, which
            is only right for a 1-D sweep.

    Returns:
        The largest amount by which the lattice optimum can exceed the
        continuous one, in EUR.
    """
    reach = stride_m * (math.sqrt(2.0) / 2.0 if diagonal else 0.5)
    return float(max_value_eur_per_m) * float(reach)


def local_lipschitz_gap(values, rows, cols, *, stride_m: float,
                        resolution_m: float, impassable: float = 65535.0,
                        diagonal: bool = True) -> np.ndarray:
    """:func:`lipschitz_stride_gap` with ``L`` taken locally per candidate.

    The global maximum cell value is a valid Lipschitz constant and
    usually a useless one -- one motorway crossing sets it for the whole
    study. This takes the maximum over the cells a candidate could
    actually move into, which is the neighbourhood the stride can hide.

    Excluded cells are skipped: a facility cannot move onto one, so its
    value is not part of the constant.
    """
    values = np.asarray(values, dtype=np.float64)
    reach_px = int(math.ceil(
        (stride_m * (math.sqrt(2.0) / 2.0 if diagonal else 0.5))
        / resolution_m))
    traversable = np.where(values >= impassable, -np.inf, values)
    rr = np.asarray(rows, dtype=np.int64)
    cc = np.asarray(cols, dtype=np.int64)
    out = np.zeros(rr.shape, dtype=np.float64)
    h, w = values.shape
    for i in range(rr.size):
        r0, r1 = max(0, rr[i] - reach_px), min(h, rr[i] + reach_px + 1)
        c0, c1 = max(0, cc[i] - reach_px), min(w, cc[i] + reach_px + 1)
        block = traversable[r0:r1, c0:c1]
        local = float(block.max()) if block.size else 0.0
        out[i] = max(0.0, local) * (
            stride_m * (math.sqrt(2.0) / 2.0 if diagonal else 0.5))
    return out


# ======================================================================
# collection cost -- the inequality that runs the other way
# ======================================================================

def collection_cost_upper_bound(distances) -> np.ndarray:
    """Sum of radial connections: an UPPER bound on collection cost.

    Named for the direction it goes in, because the plan's revision 2
    used it as a screening LOWER bound and it is not one. Shared
    trenching can only REDUCE the cost of collecting several sources to
    one point, so the radial sum sits above the true (Steiner) optimum
    and pruning with it prunes too much.

    A valid screening LOWER bound needs a Steiner lower bound.
    ``SMT >= MST / 2`` is correct and the weakest standard form; where a
    MILP solver is already in the loop the bidirected-cut LP relaxation
    is available at no modelling cost, with Wong's dual ascent
    (Math. Prog. 28:271-287, 1984) as the combinatorial fallback.
    PYORPS deliberately ships neither: selection among candidates is the
    third-party MILP's job, and a half-implemented Steiner bound here
    would only invite someone to prune with it.

    Parameters:
        distances: ``(n_sources, n_candidates)`` costs.

    Returns:
        ``(n_candidates,)`` column sums.
    """
    arr = np.asarray(distances, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"expected (n_sources, n_candidates), got {arr.shape}")
    return arr.sum(axis=0)

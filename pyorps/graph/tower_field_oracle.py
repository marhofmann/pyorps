"""Phase 0: reconcile the tower field with the constrained kernel.

The plan puts this first and says it blocks everything, for a reason that
is about DEFINITIONS rather than bugs. Three quantities in this codebase
all answer to "the cost of that overhead line" and none of them is the
same number:

* what ``_constrained_dijkstra.pyx`` minimises -- terrain plus interior
  towers, with **neither terminal tower charged**, because the source is
  seeded at ``dist = 0`` with no tower and the search stops the instant
  the target cell settles;
* what ``ConstrainedPath`` REPORTS -- the same line plus two terminal
  towers, ``>= 560 kEUR`` more on the shipped 110 kV profile, whose own
  footer says so;
* what ``_terrain_eur`` prices -- category length times category value,
  with ``IMPASSABLE_CELL_COST`` dropped from the sum, where the kernel
  charges it.

:class:`~pyorps.graph.tower_field.TowerFieldModel` names which of those a
field computes, and this module is the harness that proves it. Three
entry points, in increasing order of how much of PYORPS they involve:

:func:`score_tower_chain`
    The definition made executable. Given tower cells in order, price
    them under a model -- independently of the solver, so an agreement
    between them means something.
:func:`compare_with_kernel`
    Run ``constrained_dijkstra_2d`` itself and compare its objective to
    the field at the same target. In the matched regime (lattice factor
    1, the kernel's own step set, tier 1, zero angle premium) these
    agree to float64 round-off, and any disagreement is a real defect in
    one of the two.
:func:`compare_with_find_route`
    Run the full ``ConstrainedPathFinder.find_route`` and score the
    route it returns. This is the end-to-end check, and the one that
    sees the discretisation: the field is restricted to straight spans
    between lattice cells, the router is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pyorps.graph.tower_field import (
    TowerFieldModel,
    TowerLattice,
    TowerFieldSolver,
    intermediate_offsets,
)

__all__ = [
    "ChainScore",
    "OracleComparison",
    "score_tower_chain",
    "kernel_objective",
    "compare_with_kernel",
    "compare_with_find_route",
]


@dataclass(frozen=True)
class ChainScore:
    """What a given tower sequence costs under a stated model."""

    total: float
    terrain: float
    towers: float
    n_towers: int
    spans_m: tuple[float, ...]
    violations: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        """True when the chain breaks none of the model's own rules."""
        return not self.violations


def score_tower_chain(cells, *, values, tower_cost, lattice: TowerLattice,
                      model: TowerFieldModel, angles=None) -> ChainScore:
    """Price an explicit tower sequence under ``model``. The definition, run.

    Parameters:
        cells: Tower positions in order, ``(row, col)`` on the lattice,
            source first.
        values: Lattice cost values, EUR per metre.
        tower_cost: Node cost per lattice cell (interior towers).
        lattice: Supplies the step geometry and the admissible directions.
        model: Spans, terminal-tower treatment and span integral.
        angles: An
            :class:`~pyorps.graph.tower_field.AngleTables`, required to
            score a tier-2 chain: each interior tower then also pays
            ``premium[incoming, outgoing]``, and a deflection the hard
            limit forbids is reported as a violation rather than priced.
            Omit it for tier 1, where ``tower_cost`` already carries the
            cheapest tower type.

    Returns:
        A :class:`ChainScore`. ``violations`` lists every rule the chain
        breaks -- a span outside the admissible range, two towers not
        collinear with any lattice direction, a turn past the hard limit
        -- rather than raising, so a comparison can report WHY an oracle
        route is outside the field's feasible set instead of merely
        disagreeing with it.
    """
    values = np.asarray(values, dtype=np.float64)
    tower_cost = np.asarray(tower_cost, dtype=np.float64)
    pts = [(int(r), int(c)) for r, c in cells]
    if len(pts) < 2:
        node = (model.terminal_tower_cost
                if model.charge_terminal_towers else 0.0)
        return ChainScore(node, 0.0, node, len(pts), (), ())

    index_of = {(int(p), int(q)): i
                for i, (p, q) in enumerate(lattice.directions.tolist())}
    terrain = 0.0
    towers = 0.0
    spans: list[float] = []
    steps: list[tuple[int, int] | None] = []
    bad: list[str] = []

    terminal = (model.terminal_tower_cost
                if model.charge_terminal_towers else 0.0)
    towers += 2.0 * terminal
    for r, c in pts[1:-1]:
        towers += float(tower_cost[r, c])

    for i in range(len(pts) - 1):
        (r0, c0), (r1, c1) = pts[i], pts[i + 1]
        dr, dc = r1 - r0, c1 - c0
        g = math.gcd(abs(dr), abs(dc))
        if g == 0:
            bad.append(f"span {i}: two towers on the same cell {pts[i]}")
            spans.append(0.0)
            steps.append(None)
            continue
        step = (dr // g, dc // g)
        steps.append(step)
        if step not in index_of:
            bad.append(
                f"span {i}: {pts[i]} -> {pts[i + 1]} is not collinear with "
                f"any lattice direction (primitive step {step})")
        length = lattice.step_length_m((dr, dc))
        spans.append(length)
        is_last = i == len(pts) - 2
        lo = (model.effective_last_span_min_m if is_last
              else model.min_span_m)
        if length < lo - 1e-9:
            bad.append(f"span {i}: {length:.2f} m is below {lo:g} m")
        over = (length > model.max_span_m + 1e-9
                if model.max_span_inclusive
                else length >= model.max_span_m - 1e-9)
        if over:
            bad.append(
                f"span {i}: {length:.2f} m reaches max_span_m "
                f"{model.max_span_m:g} m")
        terrain += _span_terrain(values, (r0, c0), step, g,
                                 lattice.step_length_parts(step), model)

    if angles is not None:
        for i in range(1, len(pts) - 1):
            a, b = steps[i - 1], steps[i]
            if a is None or b is None or a not in index_of \
                    or b not in index_of:
                continue
            ia, ib = index_of[a], index_of[b]
            if not angles.valid[ia, ib]:
                bad.append(
                    f"tower {i} at {pts[i]}: the turn {a} -> {b} exceeds "
                    f"the profile's hard angle limit")
                continue
            towers += float(angles.premium[ia, ib])

    return ChainScore(terrain + towers, terrain, towers, len(pts),
                      tuple(spans), tuple(bad))


def _span_terrain(values, start, step, n_steps, length_parts, model) -> float:
    """Terrain cost of one straight span, by the model's own convention.

    ``length_parts`` is :meth:`TowerLattice.step_length_parts`, kept
    split so a square lattice's arithmetic is unchanged.
    """
    p, q = step
    hyp, sigma = length_parts
    total = 0.0
    if model.span_integral == "midpoint":
        for j in range(1, n_steps + 1):
            total += values[start[0] + j * p, start[1] + j * q]
        return total * hyp * sigma
    if model.span_integral == "trapezoid":
        for j in range(1, n_steps + 1):
            a = values[start[0] + (j - 1) * p, start[1] + (j - 1) * q]
            b = values[start[0] + j * p, start[1] + j * q]
            total += 0.5 * (a + b)
        return total * hyp * sigma
    inter = intermediate_offsets(p, q)
    factor = hyp / (2.0 + len(inter))
    for j in range(1, n_steps + 1):
        ur, uc = start[0] + (j - 1) * p, start[1] + (j - 1) * q
        acc = values[ur, uc] + values[ur + p, uc + q]
        for orow, ocol in inter:
            acc += values[ur + orow, uc + ocol]
        total += acc * factor
    return total * sigma


# ======================================================================
# the kernel itself
# ======================================================================

def kernel_objective(raster, source, target, *, directions, sigma_m,
                     min_span_m, max_span_m, tower_value_lut,
                     angle_cost=None, angle_valid=None,
                     tower_angle_cost=None, force_sparse: int = 0):
    """Run ``constrained_dijkstra_2d`` and return ``(path, towers, dist)``.

    A thin, explicit wrapper so a test states every LUT it feeds the
    kernel rather than inheriting one from a profile. ``dist`` is the
    KERNEL's objective: terrain plus interior towers, terminals not
    charged.

    Parameters:
        raster: ``uint16`` cost values; ``65535`` is excluded, which is
            also how the kernel builds its default mask.
        source, target: ``(row, col)``.
        directions: ``(K, 2)`` steps, which must be the same set the
            tower field uses or the two solve different problems.
        sigma_m: Cell size in metres.
        tower_value_lut: ``(65536,)`` tower cost by raster value.
        angle_cost, angle_valid, tower_angle_cost: Angle LUTs; default
            to zero cost, everything allowed, zero tower type cost --
            the tier-1 regime.
    """
    from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d

    raster = np.ascontiguousarray(raster, dtype=np.uint16)
    steps = np.ascontiguousarray(directions, dtype=np.int8)
    n = steps.shape[0]
    if angle_cost is None:
        angle_cost = np.zeros((n, n), dtype=np.float32)
    if angle_valid is None:
        angle_valid = np.ones((n, n), dtype=np.uint8)
    if tower_angle_cost is None:
        tower_angle_cost = np.zeros((n, n), dtype=np.float32)
    step_distances = (np.linalg.norm(steps.astype(np.float64), axis=1)
                      * sigma_m).astype(np.float32)
    # The kernel deduplicates states on span BINS but enforces the span
    # on the exact float, so the bin size only has to be fine enough not
    # to merge distinct spans; min_span is what ConstrainedPathFinder
    # itself uses (constrained_path_finder.py:130-136).
    bin_size = float(min_span_m) if min_span_m > 0 else float(max_span_m)
    n_bins = max(2, int(math.ceil(max_span_m / bin_size)))
    return constrained_dijkstra_2d(
        raster, int(source[0]), int(source[1]),
        int(target[0]), int(target[1]), steps,
        np.ascontiguousarray(angle_cost, dtype=np.float32),
        np.ascontiguousarray(angle_valid, dtype=np.uint8),
        step_distances,
        np.ascontiguousarray(tower_value_lut, dtype=np.float32),
        np.ascontiguousarray(tower_angle_cost, dtype=np.float32),
        n_bins, bin_size, float(min_span_m), float(max_span_m),
        cell_size=float(sigma_m), force_sparse=force_sparse, return_dist=1,
    )


@dataclass(frozen=True)
class OracleComparison:
    """One target, priced both ways."""

    target: tuple[int, int]
    oracle_cost: float
    field_cost: float
    oracle_towers: int          #: INTERIOR towers, terminals excluded
    field_towers: int           #: same convention, for a like comparison
    note: str = ""

    @property
    def delta(self) -> float:
        """``field - oracle``. Negative means the field found it cheaper."""
        return self.field_cost - self.oracle_cost

    @property
    def relative(self) -> float:
        if not math.isfinite(self.oracle_cost) or self.oracle_cost == 0:
            return float("nan")
        return self.delta / abs(self.oracle_cost)

    def __repr__(self) -> str:
        return (f"OracleComparison(target={self.target}, "
                f"oracle={self.oracle_cost:,.2f}, "
                f"field={self.field_cost:,.2f}, "
                f"delta={self.delta:+,.4f})")


def compare_with_kernel(raster, *, source, targets, tower_value_lut,
                        sigma_m, min_span_m, max_span_m, directions,
                        model: TowerFieldModel | None = None,
                        angles=None, force_sparse: int = 0,
                        return_field: bool = False):
    """Kernel objective vs tower field, target by target.

    The comparison is only meaningful when the two solve the same
    problem, so this pins the matched regime and lets the caller vary
    only what the plan says may vary:

    * the tower lattice is the raster itself (``factor = 1``);
    * the field's directions are the kernel's step set;
    * ``span_integral="pyorps"``, terminals uncharged, no minimum on the
      last span -- :meth:`TowerFieldModel.matching_kernel`'s defaults.

    In that regime the two agree to float64 round-off, with or without
    angle premiums: pass ``angles`` (an
    :class:`~pyorps.graph.tower_field.AngleTables`) and the harness
    feeds the kernel the same two LUTs it splits into a tier-2 premium,
    since the kernel charges the turn penalty on the edge that leaves a
    tower and the tower-type cost at that same tower.

    Outside the matched regime the delta is the DISCRETISATION, and
    should be reported as such rather than asserted away.

    Tower counts are compared as INTERIOR towers on both sides: the
    kernel's ``towers`` array excludes the two ends, so the field's
    sequence has its terminals dropped before counting.
    """
    raster = np.ascontiguousarray(raster, dtype=np.uint16)
    dirs = np.asarray(directions, dtype=np.int64).reshape(-1, 2)
    lut = np.asarray(tower_value_lut, dtype=np.float64)

    if model is None:
        model = TowerFieldModel(
            min_span_m=float(min_span_m), max_span_m=float(max_span_m),
            max_span_inclusive=False, last_span_min_m=0.0,
            charge_terminal_towers=False, span_integral="pyorps",
            angle_tier=2 if angles is not None else 1)

    lattice = TowerLattice(cell_size_m=float(sigma_m), factor=1,
                           directions=dirs)
    blocked = raster >= 65535
    values = np.where(blocked, 0.0, raster.astype(np.float64))
    tower_cost = lut[raster.astype(np.int64)]
    if model.angle_tier == 1 and angles is not None:
        usable = angles.premium[angles.valid & np.isfinite(angles.premium)]
        tower_cost = tower_cost + (float(usable.min()) if usable.size else 0.0)
    tower_cost = np.where(blocked, np.inf, tower_cost)

    field = TowerFieldSolver(
        values=values, tower_cost=tower_cost, lattice=lattice, model=model,
        blocked=blocked, angles=angles).solve(source, record_pred=True)

    # The kernel wants the two halves of the premium separately: the
    # turn penalty rides on the edge, the tower type on the tower.
    if angles is None:
        k_angle = k_tower_angle = k_valid = None
    else:
        n = dirs.shape[0]
        k_valid = angles.valid.astype(np.uint8)
        k_angle = np.zeros((n, n), dtype=np.float32)
        k_tower_angle = np.where(np.isfinite(angles.premium),
                                 angles.premium, 0.0).astype(np.float32)

    out: list[OracleComparison] = []
    for tgt in targets:
        tgt = (int(tgt[0]), int(tgt[1]))
        path, towers, dist = kernel_objective(
            raster, source, tgt, directions=dirs, sigma_m=sigma_m,
            min_span_m=min_span_m, max_span_m=max_span_m,
            tower_value_lut=lut, angle_cost=k_angle, angle_valid=k_valid,
            tower_angle_cost=k_tower_angle, force_sparse=force_sparse)
        seq = field.tower_sequence(*tgt) if field.has_paths else []
        out.append(OracleComparison(
            target=tgt, oracle_cost=float(dist),
            field_cost=float(field.arrival[tgt]),
            oracle_towers=int(len(towers)),
            field_towers=max(0, len(seq) - 2),
            note="kernel found no route" if len(path) == 0 else ""))
    return (out, field) if return_field else out


def compare_with_find_route(finder, targets: Sequence, *,
                            model: TowerFieldModel | None = None,
                            directions=None, factor: int = 1,
                            angle_tier: int | None = None,
                            **solver_kwargs) -> dict[str, Any]:
    """End-to-end: ``ConstrainedPathFinder.find_route`` vs the tower field.

    Runs the router for real, then scores the route it returns with
    :func:`score_tower_chain` under the SAME model the field used, so
    the two numbers are commensurable by construction. Three things come
    out of it, and all three matter:

    ``comparisons``
        Per target: the scored oracle route against the field value. The
        field is a minimum over a restricted feasible set, so
        ``field <= score(oracle route)`` whenever the oracle route is
        inside that set; a positive delta means the router found
        something the lattice cannot express.
    ``infeasible``
        Oracle routes whose towers are NOT collinear with a lattice
        direction, or whose spans fall outside the model. These are the
        discretisation, named rather than averaged away.
    ``reported_vs_kernel``
        The router's own reported cost next to the scored one, so the
        terminal-tower difference is visible instead of implicit.
    """
    from pyorps.graph.tower_field import (
        angle_tables_from_profile, tower_field_from_raster)

    profile = finder.profile
    raster = np.asarray(finder.raster_handler.data[0])
    window_transform = finder.raster_handler.window_transform
    # Both pixel sizes, not `abs(a)` twice: they differ on a real raster.
    cell_x = float(abs(window_transform.a))
    cell_y = float(abs(window_transform.e))
    if directions is None:
        directions = np.asarray(finder.steps)[:, :2].astype(np.int64)
    if model is None:
        model = TowerFieldModel.matching_kernel(profile)
    if angle_tier is not None:
        model = TowerFieldModel(**{**model.__dict__,
                                   "angle_tier": angle_tier})

    lattice = TowerLattice(cell_size_x_m=cell_x, cell_size_y_m=cell_y,
                           factor=factor, directions=np.asarray(directions))
    angles = (angle_tables_from_profile(profile, lattice)
              if model.angle_tier == 2 else None)

    src_rc = finder.raster_handler.coords_to_indices([finder.source_coords])
    source_cell = (int(src_rc[0][0]) // factor, int(src_rc[0][1]) // factor)

    field = tower_field_from_raster(
        raster, cell_size_x_m=cell_x, cell_size_y_m=cell_y,
        source_cell=source_cell,
        profile=profile, model=model, factor=factor, directions=directions,
        angles=angles, transform=window_transform,
        **solver_kwargs)

    blocked = raster >= 65535
    values = np.where(blocked, 0.0, raster.astype(np.float64))
    lut = profile.precompute_tower_terrain_costs()
    tower_cost = lut[raster.astype(np.int64)]

    comparisons: list[OracleComparison] = []
    infeasible: list[dict[str, Any]] = []
    reported: list[dict[str, Any]] = []
    ncols = raster.shape[1]
    for tgt in targets:
        path = finder.find_route(target=tgt)
        cells = [(int(t.cell_index) // ncols, int(t.cell_index) % ncols)
                 for t in path.towers]
        score = score_tower_chain(cells, values=values,
                                  tower_cost=tower_cost, lattice=lattice,
                                  model=model)
        tgt_rc = finder.raster_handler.coords_to_indices([tgt])
        cell = (int(tgt_rc[0][0]) // factor, int(tgt_rc[0][1]) // factor)
        comparisons.append(OracleComparison(
            target=cell, oracle_cost=score.total,
            field_cost=float(field.arrival[cell]),
            oracle_towers=score.n_towers,
            field_towers=(field.n_towers_at(*cell) if field.has_paths
                          else -1),
            note="; ".join(score.violations)))
        if score.violations:
            infeasible.append({"target": tgt, "cell": cell,
                               "violations": score.violations})
        reported.append({
            "target": tgt,
            "reported_total": float(path.total_terrain_cost
                                    + path.total_tower_cost),
            "reported_terrain": float(path.total_terrain_cost),
            "reported_towers": float(path.total_tower_cost),
            "scored_under_model": score.total,
        })
    return {"field": field, "comparisons": comparisons,
            "infeasible": infeasible, "reported_vs_kernel": reported,
            "model": model.describe()}

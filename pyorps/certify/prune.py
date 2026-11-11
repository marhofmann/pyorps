"""Bound-and-prune over positions (plan 2026-09-23, sections 3.1 and 3.7; paper math.md, M4).

The paper's global-optimisation strategy: a lower bound ``LB(s, b)`` is built for every position ``s`` of every design
branch ``b`` from the exact or windowed fields of the separable cost terms; a position is dropped when
``LB - eps_adm > Z_UB``; the kept positions are priced exactly; the run is **optimal** when every dropped position
satisfies ``LB - eps_adm > Z*_E`` (the strict corollary of the separability theorem). There is no LP, no cut and no
solver tolerance: the tightness of ``LB`` changes the number of kept positions, never the result.

This module is the part of that procedure that does not depend on the data: the prune rule, the incumbent loop (F0),
the exact evaluation of the kept set and the optimality check. The fields themselves come from the term builders.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = ["BoundAndPruneResult", "branch_lower_bound", "bound_and_prune", "prune_mask"]


def branch_lower_bound(constant: float, terms) -> np.ndarray:
    """``LB(., b) = K(b) + sum_k LB_k`` for same-shaped term bounds (M4, eq. 4.1).

    Infinite entries (an infeasible term) stay infinite; NaN is refused because it would silently survive a comparison.
    """
    arrs = [np.asarray(t, dtype=np.float64) for t in terms]
    if not arrs:
        raise ValueError("at least one term bound is needed")
    if any(np.isnan(a).any() for a in arrs):
        raise ValueError("a term bound contains NaN")
    if any((a < 0).any() for a in arrs):
        raise ValueError("every term bound must be non-negative (the admissibility argument needs it)")
    out = np.full(arrs[0].shape, float(constant), dtype=np.float64)
    for a in arrs:
        out += a
    return out


def prune_mask(lb: np.ndarray, z_ub: float, eps_adm: float) -> np.ndarray:
    """True where the position is KEPT: not ``LB - eps_adm > Z_UB`` (M4, eq. 4.3)."""
    if eps_adm < 0 or math.isnan(z_ub) or math.isnan(eps_adm):
        raise ValueError("eps_adm must be >= 0 and Z_UB a number")
    lb = np.asarray(lb, dtype=np.float64)
    if np.isnan(lb).any():
        raise ValueError("lower bound contains NaN")
    return ~(lb - eps_adm > z_ub)


@dataclass(frozen=True)
class BoundAndPruneResult:
    """Outcome of one run.

    Attributes:
        z_star: ``Z*_E``, the exact optimum over the kept positions (``inf`` if none is feasible).
        argmin: ``(branch, flat index)`` of the best design; ties are in ``ties``.
        ties: Every evaluated design within ``eps_adm`` of ``z_star`` (the co-optimal set, strict corollary).
        z_lb: ``min(Z*_E - eps_E, min over dropped positions of LB - eps_adm)``.
        optimal: ``z_lb >= z_ub_final - eps_adm`` (proven optimal within the margin).
        strict: True when every dropped position has ``LB - eps_adm > Z*_E``, so co-optimal designs outside E are
            impossible.
        kept: Number of kept positions per branch.
        total: Number of positions per branch.
        z_ub_history: ``Z_UB`` before the F0 passes and after each.
    """

    z_star: float
    argmin: tuple | None
    ties: tuple
    z_lb: float
    optimal: bool
    strict: bool
    kept: Mapping
    total: Mapping
    z_ub_history: tuple = field(default_factory=tuple)


def bound_and_prune(lb: Mapping, exact: Callable[[object, np.ndarray], np.ndarray], *, z_ub: float, eps_adm: float,
                    eps_E: float = 0.0, f0_per_branch: int = 20, f0_passes: int = 2) -> BoundAndPruneResult:
    """Run the procedure of M4 on bound fields ``lb[b]`` (any shape) and an exact evaluator.

    Args:
        lb: Branch id -> admissible lower-bound field (``LB(s, b)``, already including ``K(b)``).
        exact: ``exact(b, flat_indices) -> exact total cost`` at those positions (float64, ``inf`` if infeasible). It is
            called on the F0 candidates and on the kept set only.
        z_ub: Incumbent cost (a complete, feasible, re-priced design).
        eps_adm: Admissibility margin (all numerical understatements of the bounds).
        eps_E: Storage understatement on the argmin chain.
        f0_per_branch: Positions per branch evaluated exactly to lower ``Z_UB`` (plan phase F0: 20).
        f0_passes: How often the F0 loop is run (plan: once, then once more).

    Raises:
        AssertionError: if a bound is not admissible at an evaluated position (``LB - eps_adm > exact``): the run must
            stop, the plan's phase-G acceptance test.
    """
    flat = {b: np.asarray(v, dtype=np.float64).ravel() for b, v in lb.items()}
    cache: dict = {}

    def ev(b, idx):
        idx = np.asarray(idx, dtype=np.int64)
        need = [i for i in idx.tolist() if (b, i) not in cache]
        if need:
            vals = np.asarray(exact(b, np.asarray(need, dtype=np.int64)), dtype=np.float64)
            for i, v in zip(need, vals.tolist()):
                if flat[b][i] - eps_adm > v:
                    raise AssertionError(f"lower bound not admissible at branch {b!r}, position {i}: "
                                         f"LB - eps_adm = {flat[b][i] - eps_adm} > exact {v}")
                cache[(b, i)] = v
        return np.asarray([cache[(b, i)] for i in idx.tolist()])

    z = float(z_ub)
    history = [z]
    for _ in range(max(0, f0_passes)):
        for b, v in flat.items():
            keep = np.flatnonzero(prune_mask(v, z, eps_adm))
            if keep.size == 0:
                continue
            first = keep[np.argsort(v[keep], kind="stable")[:f0_per_branch]]
            vals = ev(b, first)
            if vals.size:
                z = min(z, float(vals.min()))
        history.append(z)

    kept, total, dropped_min = {}, {}, math.inf
    best, argmin = math.inf, None
    evaluated = []
    for b, v in flat.items():
        mask = prune_mask(v, z, eps_adm)
        idx = np.flatnonzero(mask)
        kept[b], total[b] = int(idx.size), int(v.size)
        if (~mask).any():
            dropped_min = min(dropped_min, float(v[~mask].min()) - eps_adm)
        if idx.size:
            vals = ev(b, idx)
            evaluated.extend((float(x), b, int(i)) for x, i in zip(vals, idx))
            j = int(np.argmin(vals))
            if vals[j] < best:
                best, argmin = float(vals[j]), (b, int(idx[j]))
    z_star = min(best, z) if best < math.inf else best
    ties = tuple(sorted((b, i) for x, b, i in evaluated if x <= best + eps_adm)) if best < math.inf else ()
    z_lb = min(best - eps_E, dropped_min)
    strict = dropped_min > best
    optimal = bool(best < math.inf and z_lb >= min(best, z) - eps_adm)
    return BoundAndPruneResult(z_star=z_star, argmin=argmin, ties=ties, z_lb=z_lb, optimal=optimal, strict=strict,
                               kept=kept, total=total, z_ub_history=tuple(history))

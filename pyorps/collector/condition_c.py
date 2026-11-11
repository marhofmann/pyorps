"""Condition (C) of the trench-sharing collector (plan rev. 5, section 2.1 (i)).

The collector's trenches form a tree. That is exact whenever sigma plus the
largest group-derating premium is at most the cheapest trench rate: then
merging two trenches that carry separate cable systems never costs more than
keeping them apart, so a cycle can never beat a tree. This module computes the
left-hand side from a :class:`~pyorps.collector.model.CollectorModel` and
reports the margin, so every result can print it (user decision T4).

The left-hand side is the worst super-additivity of the per-metre rate,
``delta = max [ r(a + b) - r(a) - r(b) ]`` over two disjoint groups ``a`` and
``b`` of systems with ``|a| + |b| <= m``, ``r`` the rate of one trench with
every system on its cheapest feasible option ``(type, p)`` for the trench's
conductor count (sigma included, the convention of the test oracle
``tests/test_collector/oracles/adapters.py:delta_max``). Parallel cables are
part of the options: without them a big system could not share a derated trench
and the check would read ``inf`` for a model that is fine. It is ``inf`` only
when merging leaves a system with no feasible option.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyorps.collector.model import INF, CollectorModel, bits, set_partitions
from pyorps.core.exceptions import InvalidInputError

__all__ = ["ConditionC", "condition_c"]


@dataclass(frozen=True)
class ConditionC:
    """The result of the check.

    Attributes:
        delta: sigma plus the worst derating premium, EUR per metre (inf when
            merging can make a system infeasible).
        min_trench_rate: The cheapest trench rate the check was given, EUR/m.
        margin: ``min_trench_rate - delta``.
        holds: ``delta <= min_trench_rate``.
        groups: Number of (a, b) merges examined.
    """
    delta: float
    min_trench_rate: float
    margin: float
    holds: bool
    groups: int

    def __str__(self) -> str:
        state = "holds" if self.holds else "FAILS"
        text = (f"condition (C) {state}: sigma + worst derating premium = "
                f"{self.delta:.6g} EUR/m against the cheapest trench rate "
                f"{self.min_trench_rate:.6g} EUR/m (margin "
                f"{self.margin:.6g} EUR/m, {self.groups} merges checked)")
        if not self.holds:
            text += ("; the result is exact for the tree model only, and a "
                     "strict one-trench-per-cell re-price is due (plan 2.1 (i))")
        return text

    def as_record(self) -> dict:
        """JSON-able record for the result manifest."""
        return {"delta": self.delta, "min_trench_rate": self.min_trench_rate,
                "margin": self.margin, "holds": self.holds,
                "groups": self.groups}


def _rate(model: CollectorModel, masks: list[int], cache: dict) -> float:
    """Per-metre rate of one trench carrying the systems ``masks``, every
    system on its cheapest feasible ``(type, p)`` for the trench's conductor
    count ``m = sum p``; ``inf`` when some system has no feasible option."""
    key = tuple(sorted(masks))
    if key in cache:
        return cache[key]
    k = len(masks)
    m_hi = min(model.m_cap(), k * model.p_max)
    best = INF
    for m in range(k, m_hi + 1):
        dp = {0: 0.0}                      # conductors used -> cheapest cost
        for mask in masks:
            per_p = {}
            for p in range(1, model.p_max + 1):
                costs = [model.rho(mask, (ti, p))
                         for ti in range(len(model.types))
                         if model.feasible(mask, (ti, p), m)]
                if costs:
                    per_p[p] = min(costs)
            nxt: dict[int, float] = {}
            for used, cost in dp.items():
                for p, rho in per_p.items():
                    if used + p <= m and cost + rho < nxt.get(used + p, INF):
                        nxt[used + p] = cost + rho
            dp = nxt
            if not dp:
                break
        if m in dp:
            best = min(best, dp[m] + model.sigma_eur_per_m * (m - 1))
    cache[key] = best
    return best


def condition_c(model: CollectorModel, min_trench_rate: float) -> ConditionC:
    """Evaluate condition (C) for ``model`` against ``min_trench_rate`` (the
    smallest ``c(e) / l(e)`` of the trench graph, EUR per metre)."""
    if not min_trench_rate >= 0:
        raise InvalidInputError("min_trench_rate must be >= 0")
    cap = min(model.m_cap(), model.n)       # a system carries >= 1 turbine
    worst, groups, cache = 0.0, 0, {}
    for union in range(1, 1 << model.n):
        for blocks in set_partitions(list(bits(union))):
            k = len(blocks)
            if k < 2 or k > cap:
                continue
            masks = [sum(1 << i for i in b) for b in blocks]
            both = _rate(model, masks, cache)
            for sel in range((1 << (k - 1)) - 1):
                a = [masks[0]] + [masks[j + 1] for j in range(k - 1)
                                  if sel >> j & 1]
                b = [masks[j + 1] for j in range(k - 1) if not sel >> j & 1]
                groups += 1
                ra, rb = _rate(model, a, cache), _rate(model, b, cache)
                if ra == INF or rb == INF:
                    continue
                if both == INF:
                    worst = INF
                    continue
                worst = max(worst, both - ra - rb)
    return ConditionC(delta=worst, min_trench_rate=float(min_trench_rate),
                      margin=float(min_trench_rate) - worst,
                      holds=worst <= min_trench_rate * (1 + 1e-12),
                      groups=groups)

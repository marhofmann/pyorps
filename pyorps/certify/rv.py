"""Relax-and-verify, the consent lattice and the epsilon manifest
(plan rev. 5, sections 3.6 and 3.7; D7).

**Relax-and-verify (T-RV).** Write the objective as ``F = F_rel + Delta``
with ``Delta >= 0``. Candidates ``(s, b)`` are enumerated in increasing
``F_rel``; each is verified to a bracket ``[L_i, U_i]`` (``L_i = U_i`` when
the exact treatment closes it). The driver stops once the next ``F_rel``
is strictly greater than ``U* + eps_adm`` -- so ties are enumerated -- and
the result is **optimal** iff ``Z_LB >= Z_UB - eps_adm``. A bracket that
stays open is an intermediate state: it is handed to ``escalate`` (the
exact escalations of the plan's table) until it closes, and the driver
refuses to report an open bracket as a result.

**Consent lattice** (one-off permitting costs ``C_k`` per procedure class
``k`` and a delay value ``V(S)``):
``F* = min over S subset K of [F_S + sum_{k in S} C_k + V(S)]``, where
``F_S`` is the optimum with the classes outside ``S`` masked. ``F_S`` only
grows as classes are masked, so ``F_all + sum C_k + V(S)`` -- tightened by
any already evaluated superset -- bounds every ``S`` from below, and ``S``
is skipped once that bound is strictly above the best value found plus
``eps``. Only the surviving ``S`` need masked re-runs.

**Epsilon manifest.** Every numerical allowance that enters ``eps_adm``
(storage understatement, the float32 step factor, integer scaling of the
access weights, ...) is a named, sourced, non-negative entry; the total
is what the prune rule and the optimality test use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations

__all__ = [
    "ConsentResult",
    "EpsilonManifest",
    "RVResult",
    "consent_lattice",
    "relax_and_verify",
]


# ------------------------------------------------------------ epsilon


@dataclass
class EpsilonManifest:
    """Named, sourced numerical allowances (plan section 3.7).

    ``adm`` entries add up to ``eps_adm`` (the prune rule and the
    optimality test); ``E`` entries to ``eps_E`` (the storage
    understatement on the argmin chain).
    """
    entries: list = field(default_factory=list)

    def add(self, name: str, value: float, *, kind: str = "adm",
            source: str = "") -> EpsilonManifest:
        if kind not in ("adm", "E"):
            raise ValueError("kind must be 'adm' or 'E'")
        value = float(value)
        if not (value >= 0.0 and math.isfinite(value)):
            raise ValueError(f"epsilon entry {name!r} must be finite and "
                             f">= 0, got {value}")
        if any(e["name"] == name and e["kind"] == kind for e in self.entries):
            raise ValueError(f"epsilon entry {name!r} ({kind}) is already "
                             f"recorded")
        self.entries.append({"name": name, "kind": kind, "value": value,
                             "source": source})
        return self

    def total(self, kind: str = "adm") -> float:
        return float(sum(e["value"] for e in self.entries
                         if e["kind"] == kind))

    @property
    def eps_adm(self) -> float:
        return self.total("adm")

    @property
    def eps_E(self) -> float:
        return self.total("E")

    def as_record(self) -> dict:
        """JSON-able form for the certificate manifest."""
        return {"entries": [dict(e) for e in self.entries],
                "eps_adm": self.eps_adm, "eps_E": self.eps_E}


# ------------------------------------------------------ relax-and-verify


@dataclass
class RVResult:
    """What :func:`relax_and_verify` proved.

    Attributes:
        z_ub: The best verified upper end ``U*`` (an incumbent's cost).
        z_lb: ``min(`` lower ends of the enumerated candidates, the
            relaxed value of the first one not enumerated ``)``.
        optimal: ``z_lb >= z_ub - eps``.
        argmin: The key that set ``z_ub``.
        ties: Every enumerated key with ``U_i <= z_ub + eps``.
        enumerated: ``[(key, F_rel, L, U)]`` in enumeration order.
        next_relaxed: The first ``F_rel`` not enumerated (``inf`` if
            none was left).
        log: One line per step, for the manifest.
    """
    z_ub: float
    z_lb: float
    optimal: bool
    argmin: Hashable | None
    ties: list
    enumerated: list
    next_relaxed: float
    eps: float
    log: list = field(default_factory=list)

    def as_record(self) -> dict:
        rec = asdict(self)
        rec["argmin"] = repr(self.argmin)
        rec["ties"] = [repr(k) for k in self.ties]
        rec["enumerated"] = [[repr(k), f, lo, up]
                             for k, f, lo, up in self.enumerated]
        return rec


def relax_and_verify(candidates: Iterable[tuple[Hashable, float]],
                     verify: Callable[[Hashable], tuple[float, float]], *,
                     eps: float = 0.0,
                     escalate: Callable[[Hashable, tuple[float, float]],
                                        tuple[float, float]] | None = None,
                     z_ub: float = math.inf,
                     max_escalations: int = 8) -> RVResult:
    """The T-RV driver of plan section 3.6.

    Parameters:
        candidates: ``(key, F_rel)`` with ``F_rel`` a LOWER bound on the
            candidate's exact value.
        verify: ``key -> (L, U)``, the exact treatment at fixed ``key``.
        eps: ``eps_adm`` (>= 0).
        escalate: ``(key, (L, U)) -> (L', U')`` with a strictly tighter or
            closed bracket; called while ``U - L > eps``. Without it an open
            bracket raises: a bracket is never a result.
        z_ub: An incumbent's verified cost to start from.
        max_escalations: Per candidate, before giving up with an error.

    Raises:
        ValueError: a bracket is inconsistent (``L > U``, or ``L`` below
            the candidate's own ``F_rel`` by more than ``eps``).
        RuntimeError: a bracket stays open.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if eps < 0:
        raise ValueError("eps must be >= 0")
    order = sorted(candidates, key=lambda kv: kv[1])
    best_key = None
    z_best = float(z_ub)
    lows: list[float] = []
    enumerated = []
    log = []
    next_rel = math.inf
    for i, (key, f_rel) in enumerate(order):  # pylint: disable=unused-variable
        f_rel = float(f_rel)
        if f_rel > z_best + eps:
            next_rel = f_rel
            log.append(f"stop: next F_rel {f_rel!r} > U* {z_best!r} + eps")
            break
        lo, up = (float(x) for x in verify(key))
        steps = 0
        while up - lo > eps:
            if escalate is None:
                raise RuntimeError(
                    f"candidate {key!r} stays a bracket [{lo}, {up}]; a "
                    f"bracket is never a result -- pass escalate=")
            if steps >= max_escalations:
                raise RuntimeError(
                    f"candidate {key!r} still [{lo}, {up}] after "
                    f"{steps} escalations")
            lo2, up2 = (float(x) for x in escalate(key, (lo, up)))
            if lo2 < lo - eps or up2 > up + eps:
                raise ValueError(f"escalation of {key!r} widened the "
                                 f"bracket: [{lo}, {up}] -> [{lo2}, {up2}]")
            lo, up = max(lo, lo2), min(up, up2)
            steps += 1
        if lo > up + eps:
            raise ValueError(f"candidate {key!r}: L {lo} > U {up}")
        if lo < f_rel - eps:
            raise ValueError(f"candidate {key!r}: L {lo} below its relaxed "
                             f"value {f_rel}: F_rel is not a lower bound")
        enumerated.append((key, f_rel, lo, up))
        lows.append(lo)
        log.append(f"{key!r}: F_rel {f_rel!r} -> [{lo!r}, {up!r}]"
                   + (f" after {steps} escalation(s)" if steps else ""))
        if up < z_best:
            z_best, best_key = up, key
    z_lb = min(lows + [next_rel]) if (lows or next_rel < math.inf) \
        else math.inf
    ties = [k for k, _f, _lo, up in enumerated if up <= z_best + eps]
    return RVResult(z_ub=z_best, z_lb=z_lb,
                    optimal=z_lb >= z_best - eps, argmin=best_key,
                    ties=ties, enumerated=enumerated, next_relaxed=next_rel,
                    eps=float(eps), log=log)


# ------------------------------------------------------- consent lattice


@dataclass
class ConsentResult:
    """What :func:`consent_lattice` proved.

    Attributes:
        value: ``F*``.
        classes: The consented class set ``S`` that attains it.
        ties: Every evaluated ``S`` within ``eps`` of ``F*``.
        evaluated: ``{S: F_S}`` for the masked runs actually made.
        skipped: ``{S: lower bound}`` for the sets never run.
    """
    value: float
    classes: frozenset
    ties: list
    evaluated: dict
    skipped: dict
    eps: float


def consent_lattice(classes: Sequence[Hashable], one_off: dict,
                    masked_value: Callable[[frozenset], float], *,
                    delay: Callable[[frozenset], float] | None = None,
                    eps: float = 0.0) -> ConsentResult:
    """``F* = min over S of F_S + sum C_k + V(S)`` (plan section 3.6).

    Parameters:
        classes: The procedure classes ``K``.
        one_off: ``C_k >= 0`` per class.
        masked_value: ``S -> F_S``, the optimum with every class outside
            ``S`` forbidden (``inf`` when nothing feasible remains). Called
            once with ``S = K`` (``F_all``) and then only for sets whose
            bound survives.
        delay: ``V(S) >= 0``, non-decreasing in ``S`` (``None``: 0).
        eps: ``eps_adm``; sets are skipped only when their bound exceeds
            the best value by MORE than ``eps``, so ties are evaluated.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if eps < 0:
        raise ValueError("eps must be >= 0")
    K = tuple(classes)
    if len(set(K)) != len(K):
        raise ValueError("classes must be distinct")
    for k in K:
        if not one_off.get(k, 0.0) >= 0:
            raise ValueError(f"one-off cost of {k!r} must be >= 0")
    V = delay or (lambda _S: 0.0)
    subsets = [frozenset(c) for r in range(len(K) + 1)
               for c in combinations(K, r)]

    def extra_cost(S):
        v = float(V(S))
        if v < 0:
            raise ValueError(f"V({set(S)}) = {v} < 0")
        return sum(float(one_off.get(k, 0.0)) for k in S) + v

    full = frozenset(K)
    evaluated = {full: float(masked_value(full))}
    f_all = evaluated[full]
    best = f_all + extra_cost(full)
    best_S = full
    skipped = {}

    def bound(S):
        lb = f_all
        for T, fT in evaluated.items():
            if S <= T and fT > lb:
                lb = fT
        return lb + extra_cost(S)

    # smallest bound first, so the incumbent tightens early
    for S in sorted(subsets, key=lambda S: (f_all + extra_cost(S), len(S))):
        if S in evaluated:
            continue
        lb = bound(S)
        if lb > best + eps:
            skipped[S] = lb
            continue
        fS = float(masked_value(S))
        if fS < f_all - 1e-9 * max(1.0, abs(f_all)):
            raise ValueError(f"F_S {fS} < F_all {f_all} for S = {set(S)}: "
                             f"masking cannot make the optimum cheaper")
        evaluated[S] = fS
        total = fS + extra_cost(S)
        if total < best:
            best, best_S = total, S
    ties = [S for S, fS in evaluated.items()
            if fS + extra_cost(S) <= best + eps]
    return ConsentResult(value=best, classes=best_S, ties=ties,
                         evaluated=evaluated, skipped=skipped,
                         eps=float(eps))
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

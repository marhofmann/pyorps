"""Reference cut-signature Dreyfus--Wagner for ``D_share`` (plan rev. 5, 3.2).

A clear, slow, pure-Python implementation of the exact trench-sharing
collector engine, for graphs of tens to a few thousand nodes. It is the
executable specification the raster-scale engine is tested against, and
it is itself tested against the independent oracles in
``tests/test_collector/``.

What it computes
----------------
``MV(g)`` for EVERY non-turbine node ``g`` at once: the cheapest
``D_share`` collector (see :mod:`pyorps.collector.model`) rooted at ``g``.
One run gives the whole field because collector trenches may cross the UW
cell (plan MATH-05, Remark R of the appendix).

The recursion (appendix Part IV, "Corrected algorithm")
------------------------------------------------------
A label ``L = (S, U, D)`` is the cut of the electrical network by one
trench-tree edge: ``S`` the turbines below it, ``U`` the systems crossing
it towards the root, ``D`` those crossing it away from the root. Labels
are processed by increasing ``|S|``; within one ``S`` a product Dijkstra
over (label, station bit, node) applies

* **grow** -- extend the trench by one graph edge (never into a turbine),
  cost ``c(e) + R(L) l(e)``;
* **turbine** -- the seed at a turbine's own node: children arrive by one
  step with ``D`` empty, in-systems from above become down tokens;
* **merge** -- two partial trees meet at a node: an up token of one side
  that equals a down token of the other is one cable passing the node;
* **Ja / Jb** -- a switching-station junction (output up / output down).

Stations (user decision K1, 2026-09-24) pay their building ONCE per
trench-tree node however many junctions (busbar sections) it hosts: node
states carry a station bit ``st``, a junction pays the building only when
``st = 0`` and sets it, a grow clears it, and two states may merge only if
at most one has it (implementation log section 2.1).

Token modes
-----------
``tokens="set"``: a token carries its turbine set -- exact for turbines
that differ in rating or production (engine B, user decision T5).
``tokens="count"``: a token carries only the number of turbines -- an
exact symmetry reduction when turbines are interchangeable. For turbines
that differ it uses the minimum current and loss weight over sets of that
size, which is a relaxation only together with ``conductor="per_section"``
(engine A): with one option per cable the options come from one
representative set and the result can land above engine B (review
2026-09-24: 12.0 against B's 8.0), so that combination is refused unless
the turbines are interchangeable.

``conductor="per_cable"``: every system keeps one option ``(type, p)``
between its end nodes (user decision T2), restricted by the exact
candidate-size rule (:meth:`CollectorModel.candidate_options`).
``conductor="per_section"``: the type is chosen per trench section
(``p`` stays per system) -- a relaxation, engine A.

Engine A = ``("count", "per_section")`` is an admissible lower bound on
engine B = ``("set", "per_cable")`` at every root.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pyorps.collector.model import (
    INF,
    CollectorModel,
    all_submasks,
    bits,
    popcount,
    set_partitions,
)

__all__ = [
    "CollectorGraph",
    "CollectorResult",
    "solve_collector",
]

_REL = 1e-12


@dataclass
class CollectorGraph:
    """An undirected graph with trench costs and lengths.

    ``edges[i] = (u, w, trench_eur, length_m)``. Node ids are ``0..n_nodes-1``.
    """
    n_nodes: int
    edges: list[tuple[int, int, float, float]]
    adj: list[list[tuple[int, int]]] = field(init=False, repr=False)

    def __post_init__(self):
        self.adj = [[] for _ in range(self.n_nodes)]
        for ei, (u, w, c, l) in enumerate(self.edges):
            if not (0 <= u < self.n_nodes and 0 <= w < self.n_nodes):
                raise ValueError(f"edge {ei} ({u}, {w}) leaves the graph")
            if u == w:
                raise ValueError(f"edge {ei} is a loop")
            if c < 0 or not l > 0:
                raise ValueError(f"edge {ei} needs c >= 0 and l > 0")
            self.adj[u].append((w, ei))
            self.adj[w].append((u, ei))


# ------------------------------------------------------------------ tokens
#
# A token is a tuple (key, p, ti):
#   key  turbine bitmask (set mode) or turbine count (count mode),
#   p    parallel cables,
#   ti   cable type index, or -1 when the type is free per section.
# A label is (S, U, D) with U, D sorted tuples of tokens.


class _Algebra:
    """Token rules shared by both modes; subclasses fill the differences."""

    def __init__(self, model: CollectorModel, per_cable: bool,
                 candidate_rule: bool = True):
        self.m = model
        self.per_cable = per_cable
        self.candidate_rule = candidate_rule
        self.full = model.full
        self._cost_cache: dict = {}
        self._rate_cache: dict = {}
        self._new_cache: dict = {}
        cap = model.m_cap()
        self.node_cap = (2 * model.n - 1) * model.p_max + cap

    # -- per key quantities (overridden in count mode)
    def current(self, key):
        return self.m.current_a[key]

    def weight(self, key):
        return self.m.loss_weight[key]

    def rep_mask(self, key):
        return key

    # -- costs
    def sys_cost(self, tok, mcount):
        ck = (tok, mcount)
        hit = self._cost_cache.get(ck)
        if hit is not None:
            return hit
        key, p, ti = tok
        model = self.m
        cur = self.current(key)
        w = self.weight(key)
        f = model.f(mcount)
        best = INF
        cand = range(len(model.types)) if ti < 0 else (ti,)
        for t in cand:
            typ = model.types[t]
            if p * f * typ.ampacity_a < cur * (1.0 - _REL):
                continue
            c = (p * typ.cost_eur_per_m
                 + model.loss_coef * (typ.r_ohm_per_m / p) * w)
            best = min(best, c)
        self._cost_cache[ck] = best
        return best

    def rate(self, U, D):
        key = (U, D)
        hit = self._rate_cache.get(key)
        if hit is not None:
            return hit
        toks = U + D
        mcount = sum(t[1] for t in toks)
        if not toks:
            r = 0.0
        elif self.m.m_max is not None and mcount > self.m.m_max:
            r = INF
        else:
            r = self.m.sigma_eur_per_m * (mcount - 1)
            for t in toks:
                c = self.sys_cost(t, mcount)
                if c == INF:
                    r = INF
                    break
                r += c
        self._rate_cache[key] = r
        return r

    def new_tokens(self, key):
        """Tokens a NEW system carrying ``key`` may use."""
        hit = self._new_cache.get(key)
        if hit is not None:
            return hit
        if self.per_cable:
            opts = (self.m.candidate_options(self.rep_mask(key))
                    if self.candidate_rule else self.m.options)
            out = tuple((key, p, ti) for (ti, p) in opts)
        else:
            out = tuple((key, p, -1) for p in range(1, self.m.p_max + 1))
        self._new_cache[key] = out
        return out

    def cables(self, toks):
        return sum(t[1] for t in toks)

    def bay_ok(self, tok):
        key, p, _ti = tok
        return self.current(key) / p <= self.m.bay_a * (1 + _REL)

    def turbine_ok(self, out_key, conductors):
        return (self.current(out_key) <= self.m.switchgear_a * (1 + _REL)
                and conductors <= self.m.ring_panels)

    def merge_pairs(self, L1s, L2s, S1, S2):  # pylint: disable=unused-argument  # interface shared with the set algebra
        """Every pair of labels :meth:`merge` could accept (count form:
        all of them)."""
        return itertools.product(L1s, L2s)


class _SetAlgebra(_Algebra):
    """Tokens carry turbine sets: exact for heterogeneous turbines."""

    mode = "set"

    @staticmethod
    def union(toks):
        u = 0
        for t in toks:
            u |= t[0]
        return u

    def valid(self, S, U, D, *, node):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """A's rules R1-R4 (R5 unless ``node``)."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        acc = 0
        for t in U:
            if t[0] & acc:
                return False
            acc |= t[0]
            if not node and not t[0] & S:
                return False
        accd = 0
        for t in D:
            if t[0] & accd or t[0] & S:
                return False
            accd |= t[0]
            if not any((t[0] & u[0]) == t[0] for u in U):
                return False
        return acc == (S | accd)

    def growable(self, L):
        S, U, D = L
        if not U or (S == self.full and D):
            return False
        if not self.valid(S, U, D, node=False):
            return False
        return self.rate(U, D) < INF

    def merge_pairs(self, L1s, L2s, S1, S2):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """The pairs of labels (sets ``S1`` and ``S2``) worth a
        :meth:`merge` -- an exact pre-filter, not a heuristic.

        A down token of one side that carries a turbine of the other side
        must be matched by an equal up token there, or it would stay a
        down token over the merged set (rule R2) and :meth:`merge` would
        reject the pair. So every other label is indexed by those "needs"
        and only pairs whose needs are met are tried (review 2026-09-24:
        without it n = 5 made 2.8 M merge calls per level, 0.17 % useful).
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        groups: dict = {}
        ups2 = []
        for j, b2 in enumerate(L2s):
            need2 = frozenset(d for d in b2[2] if d[0] & S1)
            groups.setdefault(need2, []).append(j)
            ups2.append(frozenset(b2[1]))
        for b1 in L1s:
            need1 = frozenset(d for d in b1[2] if d[0] & S2)
            U1 = b1[1]
            for r in range(len(U1) + 1):
                for T in itertools.combinations(U1, r):
                    for j in groups.get(frozenset(T), ()):
                        if need1 <= ups2[j]:
                            yield b1, L2s[j]

    def merge(self, L1, L2):
        """The merged label, as a list of zero or one labels. In set form
        an up token equal to a down token of the other side IS that cable
        passing the node, so the matching is forced."""
        S1, U1, D1 = L1
        S2, U2, D2 = L2
        u1, d1, u2, d2 = set(U1), set(D1), set(U2), set(D2)
        m12 = u1 & d2
        m21 = u2 & d1
        U = (u1 - m12) | (u2 - m21)
        D = (d1 - m21) | (d2 - m12)
        if len(U) != len(u1 - m12) + len(u2 - m21):
            return []
        if len(D) != len(d1 - m21) + len(d2 - m12):
            return []
        S = S1 | S2
        U, D = tuple(sorted(U)), tuple(sorted(D))
        if not self.valid(S, U, D, node=True):
            return []
        if self.cables(U + D) > self.node_cap:
            return []
        return [(S, U, D)]

    def junctions(self, L):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """(L2, conductors connected) for every Ja/Jb at one node."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        S, U, D = L
        out = []
        occupied = S | self.union(D)
        avail = self.full & ~occupied
        Ul = list(U)
        above = _disjoint_collections(avail)
        # Ja: inputs = some ups + new downs from above, output up.
        for r in range(len(Ul) + 1):
            for Mb in itertools.combinations(Ul, r):
                for Ma_masks in above:
                    if r + len(Ma_masks) < 2:
                        continue
                    sig = self.union(Mb)
                    for c in Ma_masks:
                        sig |= c
                    for Ma in itertools.product(*[self.new_tokens(c)
                                                  for c in Ma_masks]):
                        for o in self.new_tokens(sig):
                            U2 = tuple(sorted([t for t in Ul if t not in Mb]
                                              + [o]))
                            D2 = tuple(sorted(D + Ma))
                            if (self.cables(U2 + D2) > self.node_cap
                                    or not self.valid(S, U2, D2, node=True)):
                                continue
                            pan = (self.cables(Mb) + self.cables(Ma)
                                   + o[1])
                            out.append(((S, U2, D2), pan))
        # Jb: a down token splits into >= 2 inputs from above.
        for C in D:
            atoms = [1 << i for i in bits(C[0])]
            for part in set_partitions(atoms):
                if len(part) < 2:
                    continue
                masks = [sum(b) for b in part]
                for toks in itertools.product(*[self.new_tokens(mk)
                                                for mk in masks]):
                    D2 = tuple(sorted([t for t in D if t != C] + list(toks)))
                    if (self.cables(U + D2) > self.node_cap
                            or not self.valid(S, U, D2, node=True)):
                        continue
                    out.append(((S, U, D2), C[1] + self.cables(toks)))
        return out

    def down_collections(self, avail):
        """Token tuples for in-systems that come down from above."""
        res = []
        for masks in _disjoint_collections(avail):
            for toks in itertools.product(*[self.new_tokens(c)
                                            for c in masks]):
                res.append(tuple(sorted(toks)))
        return res

    def key_of_turbine(self, i):
        return 1 << i

    def child_key_union(self, keys):
        u = 0
        for k in keys:
            u |= k
        return u


class _CountAlgebra(_Algebra):
    """Tokens carry turbine counts: engine A."""

    mode = "count"

    def __init__(self, model, per_cable, candidate_rule=True):
        super().__init__(model, per_cable, candidate_rule)
        self._imin = model.min_over_size(model.current_a)
        self._wmin = model.min_over_size(model.loss_weight)
        self._rep = {}
        for mask in range(1, model.full + 1):
            self._rep.setdefault(popcount(mask), mask)

    def current(self, key):
        return self._imin[key]

    def weight(self, key):
        return self._wmin[key]

    def rep_mask(self, key):
        return self._rep[key]

    def valid(self, S, U, D, *, node):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        s = popcount(S)
        su = sum(t[0] for t in U)
        sd = sum(t[0] for t in D)
        if su - sd != s:
            return False
        if sd > self.m.n - s:
            return False
        if not node and (not U or len(U) > s):
            return False
        return all(1 <= t[0] <= self.m.n for t in U + D)

    def growable(self, L):
        S, U, D = L
        if not U or (S == self.full and D):
            return False
        if not self.valid(S, U, D, node=False):
            return False
        return self.rate(U, D) < INF

    def merge(self, L1, L2):
        """Every merged label. In count form an up token of one side equal
        to a down token of the other side MAY be the same cable passing the
        node, or a different one, so every matching is enumerated."""
        S = L1[0] | L2[0]
        opts12 = _matchings(L1[1], L2[2])
        opts21 = _matchings(L2[1], L1[2])
        out = []
        for (u1, d2) in opts12:
            for (u2, d1) in opts21:
                U = tuple(sorted(u1 + u2))
                D = tuple(sorted(d1 + d2))
                if self.cables(U + D) > self.node_cap:
                    continue
                if not self.valid(S, U, D, node=True):
                    continue
                out.append((S, U, D))
        return out

    def junctions(self, L):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        S, U, D = L
        out = []
        n = self.m.n
        room = n - popcount(S) - sum(t[0] for t in D)
        Ul = list(U)
        # Ja: sub-multiset of ups + new downs from above, output up.
        idx = list(range(len(Ul)))
        seen = set()
        for r in range(len(Ul) + 1):
            for pick in itertools.combinations(idx, r):
                Mb = tuple(sorted(Ul[i] for i in pick))
                rest = tuple(sorted(Ul[i] for i in idx if i not in pick))
                if (Mb, rest) in seen:
                    continue
                seen.add((Mb, rest))
                for downs in _count_multisets(room):
                    if r + len(downs) < 2:
                        continue
                    total = sum(t[0] for t in Mb) + sum(downs)
                    if total > n:
                        continue
                    for Ma in itertools.product(*[self.new_tokens(d)
                                                  for d in downs]):
                        for o in self.new_tokens(total):
                            U2 = tuple(sorted(rest + (o,)))
                            D2 = tuple(sorted(D + Ma))
                            if (self.cables(U2 + D2) > self.node_cap
                                    or not self.valid(S, U2, D2, node=True)):
                                continue
                            pan = self.cables(Mb) + self.cables(Ma) + o[1]
                            out.append(((S, U2, D2), pan))
        # Jb: split a down token.
        for C in set(D):
            for parts in _count_partitions(C[0]):
                if len(parts) < 2:
                    continue
                for toks in itertools.product(*[self.new_tokens(k)
                                                for k in parts]):
                    D2 = list(D)
                    D2.remove(C)
                    D2 = tuple(sorted(D2 + list(toks)))
                    if (self.cables(U + D2) > self.node_cap
                            or not self.valid(S, U, D2, node=True)):
                        continue
                    out.append(((S, U, D2), C[1] + self.cables(toks)))
        return out

    def down_collections(self, avail):
        room = popcount(avail)
        res = []
        for downs in _count_multisets(room):
            for toks in itertools.product(*[self.new_tokens(d)
                                            for d in downs]):
                res.append(tuple(sorted(toks)))
        return res

    def key_of_turbine(self, i):  # pylint: disable=unused-argument  # interface shared with the set algebra
        return 1

    def child_key_union(self, keys):
        return sum(keys)


def _matchings(ups, downs):
    """Every way to cancel equal tokens of ``ups`` (one side's up-systems)
    against ``downs`` (the other side's down-systems): a list of the
    remaining ``(ups, downs)`` tuples, including "match nothing"."""
    common = sorted(set(ups) & set(downs))
    if not common:
        return [(tuple(ups), tuple(downs))]
    ranges = [range(min(ups.count(t), downs.count(t)) + 1) for t in common]
    out = []
    for take in itertools.product(*ranges):
        u, d = list(ups), list(downs)
        for t, k in zip(common, take):
            for _ in range(k):
                u.remove(t)
                d.remove(t)
        out.append((tuple(u), tuple(d)))
    return out


def _disjoint_collections(mask: int) -> list[tuple[int, ...]]:
    """Every collection of pairwise-disjoint non-empty submasks of ``mask``,
    including the empty collection."""
    out: list[tuple[int, ...]] = []

    def rec(rem, acc):
        if rem == 0:
            out.append(tuple(acc))
            return
        b = rem & -rem
        rec(rem ^ b, acc)
        rest = rem ^ b
        for sub in all_submasks(rest):
            c = sub | b
            rec(rem & ~c, acc + [c])

    rec(mask, [])
    return out


def _count_multisets(room: int) -> list[tuple[int, ...]]:
    """Multisets of positive counts with sum <= room (incl. the empty one)."""
    out: list[tuple[int, ...]] = [()]

    def rec(rem, mx, cur):
        for k in range(min(rem, mx), 0, -1):
            nxt = cur + [k]
            out.append(tuple(sorted(nxt)))
            rec(rem - k, k, nxt)

    rec(room, room, [])
    return out


def _count_partitions(total: int) -> list[tuple[int, ...]]:
    """Integer partitions of ``total``."""
    out = []

    def rec(rem, mx, cur):
        if rem == 0:
            out.append(tuple(sorted(cur)))
            return
        for k in range(min(rem, mx), 0, -1):
            rec(rem - k, k, cur + [k])

    rec(total, total, [])
    return out


# ------------------------------------------------------------------ result


@dataclass
class CollectorResult:
    """``MV(g)`` for every node, plus what traceback needs.

    ``mv[g]`` is ``inf`` at turbine nodes and where no feasible collector
    exists. :meth:`design` rebuilds the argmin design at one root (set
    mode only).
    """
    mv: np.ndarray
    engine: str
    stats: dict
    _solver: _Solver | None = field(default=None, repr=False)

    def design(self, g: int):
        """The optimal design rooted at ``g`` (see :mod:`.design`)."""
        if self._solver is None:
            raise RuntimeError("solve with keep_trace=True to trace designs")
        return self._solver.trace(int(g))


def solve_collector(graph: CollectorGraph, turbines: Sequence[int],
                    model: CollectorModel, *, engine: str = "B",
                    tokens: str | None = None, conductor: str | None = None,
                    root: int | None = None, candidate_rule: bool = True,
                    keep_trace: bool = True) -> CollectorResult:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """``MV(g)`` for every node ``g`` of ``graph``.

    Parameters:
        graph: The routing graph.
        turbines: Turbine nodes; turbine ``i`` is bit ``i`` of every mask.
        model: The ``D_share`` model of one MV level.
        engine: ``"B"`` (exact, set tokens, one option per cable) or
            ``"A"`` (fast bound, count tokens, type free per section).
            ``tokens`` / ``conductor`` override the two switches.
        root: Solve for this one UW node only, with the UW cell closed to
            every other trench (plan's native model). ``None`` (default)
            gives the field over every node, collectors allowed to cross
            the UW cell (MATH-05).
        candidate_rule: Restrict a new system's options to the exact
            candidate set (default). ``False`` tries every option -- only
            for testing the rule.
        keep_trace: Keep predecessor records so :meth:`CollectorResult.design`
            works (set mode only).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if engine not in ("A", "B"):
        raise ValueError("engine must be 'A' or 'B'")
    tokens = tokens or ("set" if engine == "B" else "count")
    conductor = conductor or ("per_cable" if engine == "B" else "per_section")
    if tokens not in ("set", "count") or conductor not in ("per_cable",
                                                           "per_section"):
        raise ValueError("tokens must be set|count, conductor "
                         "per_cable|per_section")
    if (tokens == "count" and conductor == "per_cable"
            and not model.interchangeable()):
        raise ValueError(
            "count tokens with one option per cable are neither exact nor a "
            "relaxation for turbines that differ; use engine A "
            "(count, per_section) for a bound or engine B for the value")
    turbines = [int(t) for t in turbines]
    if len(turbines) != model.n or len(set(turbines)) != model.n:
        raise ValueError(f"need {model.n} distinct turbine nodes")
    algebra_cls = _SetAlgebra if tokens == "set" else _CountAlgebra
    alg = algebra_cls(model, conductor == "per_cable", candidate_rule)
    solver = _Solver(graph, turbines, model, alg,
                     keep_trace=keep_trace and tokens == "set", root=root)
    mv = solver.run()
    return CollectorResult(mv=mv, engine=f"{tokens}/{conductor}",
                           stats=solver.stats,
                           _solver=solver if solver.keep_trace else None)


# ------------------------------------------------------------------ solver


class _Solver:
    def __init__(self, graph, turbines, model, alg, keep_trace, root=None):
        self.G = graph
        self.A = turbines
        self.model = model
        self.alg = alg
        self.keep_trace = keep_trace
        N = graph.n_nodes
        self.N = N
        self.is_turb = np.zeros(N, dtype=bool)
        self.is_turb[turbines] = True
        self.root = None if root is None else int(root)
        if self.root is not None and self.is_turb[self.root]:
            raise ValueError("the root cannot be a turbine node")
        # no_transit: nodes a grow may not enter and where no merge or
        # junction happens -- the turbines, plus the root in single-root mode
        self.no_transit = self.is_turb.copy()
        if self.root is not None:
            self.no_transit[self.root] = True
        self.steiner = np.flatnonzero(~self.no_transit)
        self.F: dict = {}            # label -> [F0, F1] arrays over nodes
        self.Aarr: dict = {}         # label (D empty, growable) -> array
        self.aarg: dict = {}         # label -> {node: (x, ei)}
        self.pred: dict = {}         # (label, st, v) -> record
        self.by_S: dict = {}         # S -> [labels]
        self.stats = {"labels": 0, "growable": 0, "merge_pairs": 0,
                      "junction_arcs": 0, "pops": 0}

    # -------------------------------------------------------------- run
    def run(self) -> np.ndarray:
        model = self.model
        order = sorted(range(1, model.full + 1), key=lambda s: (popcount(s), s))
        for S in order:
            self._level(S)
        return self._root()

    def _level(self, S):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        alg, N = self.alg, self.N
        dist: dict = {}
        heap: list = []

        def arr(L):
            a = dist.get(L)
            if a is None:
                a = [np.full(N, INF), np.full(N, INF)]
                dist[L] = a
            return a

        def relax(L, st, v, d, rec):
            a = arr(L)[st]
            if d < a[v]:
                a[v] = d
                if self.keep_trace:
                    self.pred[(L, st, v)] = rec
                heapq.heappush(heap, (d, L, st, v))

        # ---- turbine seeds
        for i in bits(S):
            for L, cost, rec in self._turbine_seeds(S, i):
                relax(L, 0, self.A[i], cost, rec)
        # ---- merges at non-turbine nodes, from finished smaller levels
        low = S & -S
        rest = S ^ low
        stein = self.steiner
        for sub in all_submasks(rest):
            S1 = sub | low
            if S1 == S:
                continue
            S2 = S ^ S1
            for b1, b2 in alg.merge_pairs(self.by_S.get(S1, ()),
                                          self.by_S.get(S2, ()), S1, S2):
                merged = alg.merge(b1, b2)
                if not merged:
                    continue
                F1, F2 = self.F[b1], self.F[b2]
                for st1, st2 in ((0, 0), (1, 0), (0, 1)):
                    vals = F1[st1][stein] + F2[st2][stein]
                    ok = np.flatnonzero(np.isfinite(vals))
                    if not ok.size:
                        continue
                    st = st1 | st2
                    for L in merged:
                        self.stats["merge_pairs"] += 1
                        for j in ok:
                            v = int(stein[j])
                            relax(L, st, v, float(vals[j]),
                                  ("merge", (b1, st1), (b2, st2)))
        # ---- product Dijkstra over (label, st, node)
        junc_cache: dict = {}
        grow_ok: dict = {}
        model = self.model
        while heap:
            d, L, st, v = heapq.heappop(heap)
            if d > dist[L][st][v]:
                continue
            self.stats["pops"] += 1
            g_ok = grow_ok.get(L)
            if g_ok is None:
                g_ok = alg.growable(L)
                grow_ok[L] = g_ok
            if g_ok:
                r = alg.rate(L[1], L[2])
                for (w, ei) in self.G.adj[v]:
                    if self.no_transit[w]:
                        continue
                    _, _, c, ln = self.G.edges[ei]
                    relax(L, 0, w, d + c + r * ln, ("grow", v, st, ei))
            if self.no_transit[v] or not model.allow_stations:
                continue
            ops = junc_cache.get(L)
            if ops is None:
                ops = alg.junctions(L)
                junc_cache[L] = ops
                self.stats["junction_arcs"] += len(ops)
            for L2, pan in ops:
                cost = (model.station_panel_eur * pan
                        + (model.station_building_eur if st == 0 else 0.0))
                relax(L2, 1, v, d + cost, ("junc", (L, st), pan))
        # ---- finish the level
        labels = []
        for L, a in dist.items():
            if not (np.isfinite(a[0]).any() or np.isfinite(a[1]).any()):
                continue
            labels.append(L)
            self.F[L] = a
            self.stats["labels"] += 1
            if grow_ok.get(L, alg.growable(L)):
                self.stats["growable"] += 1
                if not L[2]:
                    self._arrival(L, a)
        self.by_S[S] = labels
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    def _arrival(self, L, a):
        """Aarr(L, y) for every node y: one last step into y."""
        r = self.alg.rate(L[1], L[2])
        best = np.full(self.N, INF)
        arg = {}
        Fmin = np.minimum(a[0], a[1])
        for y in range(self.N):
            for (x, ei) in self.G.adj[y]:
                fx = Fmin[x]
                if fx == INF:
                    continue
                _, _, c, ln = self.G.edges[ei]
                val = fx + c + r * ln
                if val < best[y]:
                    best[y] = val
                    arg[y] = (x, ei)
        self.Aarr[L] = best
        self.aarg[L] = arg

    def _turbine_seeds(self, S, i):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """(label, cost, trace record) for turbine i at its node."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        alg, model = self.alg, self.model
        t = self.A[i]
        Y = S & ~(1 << i)
        # Children: partitions of Y into groups, each arriving on its own
        # trench edge with a label (Yg, U, ()) -- keep the cheapest per
        # resulting in-token tuple.
        children: dict = {}
        if Y == 0:
            children[()] = (0.0, ())
        else:
            ybits = [1 << j for j in bits(Y)]
            for part in set_partitions(ybits):
                groups = [sum(b) for b in part]
                opts = []
                for Yg in groups:
                    lst = []
                    for L in self.by_S.get(Yg, ()):
                        if L[2] or L not in self.Aarr:
                            continue
                        a = self.Aarr[L][t]
                        if a < INF:
                            lst.append((L, a))
                    if not lst:
                        break
                    opts.append(lst)
                else:
                    for combo in itertools.product(*opts):
                        ins = tuple(sorted(tok for (L, _a) in combo
                                           for tok in L[1]))
                        if alg.cables(ins) > model.ring_panels - 1:
                            continue
                        cost = sum(a for (_L, a) in combo)
                        prev = children.get(ins)
                        if prev is None or cost < prev[0]:
                            children[ins] = (cost, tuple(L for (L, _a) in combo))
        if not children:
            return []
        out = []
        avail = model.full & ~S
        downs_all = alg.down_collections(avail)
        own = alg.key_of_turbine(i)
        for ins, (cost_in, kids) in children.items():
            in_cables = alg.cables(ins)
            in_key = alg.child_key_union(tok[0] for tok in ins)
            for downs in downs_all:
                cond_in = in_cables + alg.cables(downs)
                if cond_in > model.ring_panels - 1:
                    continue
                out_key = own + in_key + sum(tok[0] for tok in downs) \
                    if alg.mode == "count" else (own | in_key
                                                  | alg.union(downs))
                for o in alg.new_tokens(out_key):
                    cond = cond_in + o[1]
                    if not alg.turbine_ok(out_key, cond):
                        continue
                    L = (S, (o,), downs)
                    if not alg.growable(L):
                        continue
                    cost = cost_in + model.turbine_panel_eur * cond
                    out.append((L, cost, ("turb", i, kids, ins, downs, o)))
        return out

    def _root(self):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """MV(g) = cheapest partition of A into feeders arriving at g."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        # pylint: disable=attribute-defined-outside-init  # result state set by the solve
        model, alg = self.model, self.alg
        N = self.N
        best = {}
        self.best_arg = {}
        self.root_labels = []
        for L, arrv in self.Aarr.items():
            S, U, D = L
            if D or not all(alg.bay_ok(t) for t in U):
                continue
            lid = len(self.root_labels)
            self.root_labels.append(L)
            val = arrv + model.bay_eur * alg.cables(U)
            cur = best.get(S)
            if cur is None:
                best[S] = val.copy()
                arg = np.full(N, -1, dtype=np.int64)
                arg[np.isfinite(val)] = lid
                self.best_arg[S] = arg
            else:
                better = val < cur
                cur[better] = val[better]
                self.best_arg[S][better] = lid
        H = {0: np.zeros(N)}
        self.H_arg = {}
        for X in range(1, model.full + 1):
            low = X & -X
            h = np.full(N, INF)
            harg = np.zeros(N, dtype=np.int64)
            for sub in all_submasks(X ^ low):
                Y = sub | low
                b = best.get(Y)
                if b is None:
                    continue
                val = b + H[X ^ Y]
                better = val < h
                h[better] = val[better]
                harg[better] = Y
            H[X] = h
            self.H_arg[X] = harg
        mv = H[model.full].copy()
        mv[self.is_turb] = INF
        if self.root is not None:
            keep = mv[self.root]
            mv[:] = INF
            mv[self.root] = keep
        self.mv = mv
        return mv

    # ------------------------------------------------------------ trace
    def trace(self, g: int):
        from pyorps.collector.design import Design, Junction, System

        if not np.isfinite(self.mv[g]):
            raise ValueError(f"no feasible collector at root {g}")
        tnodes: list[int] = []
        tedges: list[tuple[int, int, int]] = []
        systems: list[System] = []
        junctions: list[Junction] = []
        turbine_tnode: dict[int, int] = {}
        # token -> system id, one per (token, open context); set mode makes
        # tokens unique within one design, so a dict suffices.
        sys_of: dict = {}

        def system_id(tok):
            sid = sys_of.get(tok)
            if sid is None:
                sid = len(systems)
                systems.append(System(mask=tok[0], option=(tok[2], tok[1])))
                sys_of[tok] = sid
            return sid

        def new_node(cell):
            tnodes.append(cell)
            return len(tnodes) - 1

        def rooted(L, st, v, top=None):
            # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
            first = top = new_node(v) if top is None else top
            # grow chains iterate: a trench path of thousands of steps must
            # not cost thousands of stack frames (review 2026-09-24)
            while True:
                rec = self.pred.get((L, st, v))
                if rec is None:
                    raise RuntimeError(f"no trace record for {(L, st, v)}")
                if rec[0] != "grow":
                    break
                _, u, st_u, ei = rec
                child = new_node(u)
                tedges.append((child, top, ei))
                top, st, v = child, st_u, u
            kind = rec[0]
            if kind == "turb":
                _, i, kids, ins, downs, o = rec
                turbine_tnode[i] = top
                systems[system_id(o)].source = ("turbine", i)
                for tok in ins + downs:
                    systems[system_id(tok)].sink = ("turbine", i)
                for kid in kids:
                    x, ei = self.aarg[kid][v]
                    st_k = 0 if self.F[kid][0][x] <= self.F[kid][1][x] else 1
                    c_top = rooted(kid, st_k, x)
                    tedges.append((c_top, top, ei))
            elif kind == "merge":
                _, (b1, s1), (b2, s2) = rec
                rooted(b1, s1, v, top)
                rooted(b2, s2, v, top)
            elif kind == "junc":
                _, (Lf, stf), _pan = rec
                rooted(Lf, stf, v, top)
                _S, Uf, Df = Lf
                _S, Ut, Dt = L
                jid = len(junctions)
                if set(Ut) != set(Uf):          # Ja: output goes up
                    out_tok = next(iter(set(Ut) - set(Uf)))
                    ins = list(set(Uf) - set(Ut)) + list(set(Dt) - set(Df))
                else:                           # Jb: output goes down
                    out_tok = next(iter(set(Df) - set(Dt)))
                    ins = list(set(Dt) - set(Df))
                junctions.append(Junction(tnode=top, inputs=[], output=-1))
                oid = system_id(out_tok)
                systems[oid].source = ("junction", jid)
                junctions[jid].output = oid
                for tok in ins:
                    sid = system_id(tok)
                    systems[sid].sink = ("junction", jid)
                    junctions[jid].inputs.append(sid)
            else:
                raise RuntimeError(f"unknown trace record {kind!r}")
            return first

        root_node = new_node(g)
        X = self.model.full
        while X:
            Y = int(self.H_arg[X][g])
            L = self.root_labels[int(self.best_arg[Y][g])]
            x, ei = self.aarg[L][g]
            st_x = 0 if self.F[L][0][x] <= self.F[L][1][x] else 1
            c_top = rooted(L, st_x, x)
            tedges.append((c_top, root_node, ei))
            for tok in L[1]:
                systems[system_id(tok)].sink = ("root", 0)
            X ^= Y
        return Design(tnodes=tnodes, tedges=tedges, root_tnode=root_node,
                      turbine_tnode=turbine_tnode, systems=systems,
                      junctions=junctions, cost=float(self.mv[g]))

# Frozen port of the 2026-09-23 prototype `design_a/share_model.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""D_share (trench-tree model M2): instance, bundle rates, cut labels.

A *label* is the cut of the electrical arborescence H by one trench-tree edge:
    (S, U, D)
    S : bitmask of turbines inside the trench subtree below the edge
    U : frozenset of bitmasks, the up-systems (flow toward the root)
    D : frozenset of bitmasks, the down-systems (flow away from the root)
Structural rules (R-once + canonical form):
    U pairwise disjoint, D pairwise disjoint, union(U) == S | union(D),
    D-sets disjoint from S, every D-set inside one U-block, every U-block meets S.
Rate of a trench step carrying label L: R(L) * length + c(e), with
    R(L) = sum_p min_type{a + b P_p^2 : cap*f(m) >= P_p} + sigma*(m-1),  m = |U|+|D|.
Variant 'typed': systems carry a fixed conductor type (constant along the arc).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

INF = float("inf")


def popcount(x: int) -> int:
    return bin(x).count("1")


def bits(x: int):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def union(masks) -> int:
    u = 0
    for m in masks:
        u |= m
    return u


@dataclass
class Inst:
    n: int
    edges: list            # (u, w, c, l)
    turb: list             # turbine cells
    root: int
    sigma: float = 0.1
    JT: float = 2.0
    JK: float = 3.0
    dmax: int = 5          # max junction degree (inputs + 1)
    bay: float = 1.0
    panel: float = 0.1
    kmax: int = 3          # switchgear through-current limit (turbines on a turbine's out-arc)
    mmax: int = 3          # max 3-phase systems per trench
    types: list = field(default_factory=lambda: [(1.2, 0.3, 0.2), (2.5, 0.5, 0.08), (3.6, 0.8, 0.04)])
    derate: list = field(default_factory=lambda: [1.0, 1.0, 0.85, 0.75, 0.70, 0.66, 0.62, 0.6])
    P: list = None         # turbine powers (units)

    def __post_init__(self):
        self.k = len(self.turb)
        if self.P is None:
            self.P = [1.0] * self.k
        self.adj = [[] for _ in range(self.n)]
        for ei, (u, w, c, l) in enumerate(self.edges):
            self.adj[u].append((w, ei))
            self.adj[w].append((u, ei))
        self.cell2turb = {t: i for i, t in enumerate(self.turb)}
        self.full = (1 << self.k) - 1
        self._pow = {}

    def power(self, mask: int) -> float:
        p = self._pow.get(mask)
        if p is None:
            p = sum(self.P[i] for i in bits(mask))
            self._pow[mask] = p
        return p

    def f(self, m: int) -> float:
        return self.derate[min(m, len(self.derate) - 1)]

    # ---- per-system cost (free conductor choice per step) ----
    def sys_cost(self, mask: int, m: int, typ=None) -> float:
        P = self.power(mask)
        best = INF
        cand = range(len(self.types)) if typ is None else (typ,)
        for ti in cand:
            cap, a, b = self.types[ti]
            if cap * self.f(m) >= P - 1e-12:
                best = min(best, a + b * P * P)
        return best

    def rate(self, systems) -> float:
        """systems: iterable of masks (free type) or (mask, type) tuples (typed)."""
        systems = list(systems)
        m = len(systems)
        if m == 0:
            return 0.0
        if m > self.mmax:
            return INF
        tot = 0.0
        for s in systems:
            if isinstance(s, tuple):
                c = self.sys_cost(s[0], m, s[1])
            else:
                c = self.sys_cost(s, m)
            if c == INF:
                return INF
            tot += c
        return tot + self.sigma * (m - 1)

    def junction_cost(self, deg: int) -> float:
        if deg < 3 or deg > self.dmax:
            return INF
        return self.JT if deg == 3 else self.JK


# --------------------------------------------------------------------------
# combinatorics helpers
# --------------------------------------------------------------------------
def submasks(mask: int):
    s = mask
    while True:
        yield s
        if s == 0:
            break
        s = (s - 1) & mask


def disjoint_collections(mask: int, max_sets: int = 99):
    """All collections (frozensets) of pairwise-disjoint non-empty submasks of mask."""
    out = []

    def rec(rem, acc):
        if rem == 0:
            out.append(frozenset(acc))
            return
        b = rem & -rem
        rec(rem ^ b, acc)
        if len(acc) >= max_sets:
            return
        rest = rem ^ b
        for sub in submasks(rest):
            C = sub | b
            rec(rem & ~C, acc + [C])

    rec(mask, [])
    return out


def set_partitions(items):
    """All partitions of a list into non-empty blocks (lists)."""
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


def valid_label(S, U, D, allow_free_blocks=False) -> bool:
    """allow_free_blocks: node-only states may carry an up-block with no turbine of S
    (the output of a junction fed only from above, consumed at the same node)."""
    Ul, Dl = list(U), list(D)
    acc = 0
    for B in Ul:
        if B & acc:
            return False
        acc |= B
        if not (B & S) and not allow_free_blocks:
            return False
    accd = 0
    for C in Dl:
        if C & accd:
            return False
        accd |= C
        if C & S:
            return False
        if not any((C & B) == C for B in Ul):
            return False
    return acc == (S | accd)


def enumerate_labels(inst: Inst, allow_down=True, typed=False, node_labels=True):
    """All structurally valid labels.  Returns dict label->rate.
    node_labels=True keeps labels whose rate is INF (m > mmax or capacity): they are
    legal *node* states (between a merge and a junction) but can never grow."""
    k, full = inst.k, inst.full
    labs = {}
    ntypes = len(inst.types)
    for S in range(1, full + 1):
        comp = full ^ S
        Dcols = disjoint_collections(comp) if allow_down else [frozenset()]
        for D in Dcols:
            if not node_labels and len(D) + 1 > inst.mmax:
                continue
            atoms_s = [1 << i for i in bits(S)]
            atoms_d = list(D)
            atoms = atoms_s + atoms_d
            for part in set_partitions(atoms):
                blocks = []
                free = False
                for blk in part:
                    if not any((a & S) for a in blk):
                        free = True
                    blocks.append(union(blk))
                if free and not node_labels:
                    continue
                m = len(blocks) + len(D)
                if not node_labels and m > inst.mmax:
                    continue
                U = frozenset(blocks)
                if not typed:
                    r = INF if free else inst.rate(list(U) + list(D))
                    if r < INF or node_labels:
                        labs[(S, U, frozenset(D))] = r
                else:
                    # typed: every system carries a conductor type
                    sysl = list(U) + list(D)
                    nU = len(U)
                    import itertools
                    for tv in itertools.product(range(ntypes), repeat=len(sysl)):
                        Ut = frozenset((sysl[i], tv[i]) for i in range(nU))
                        Dt = frozenset((sysl[i], tv[i]) for i in range(nU, len(sysl)))
                        r = INF if free else inst.rate([(sysl[i], tv[i]) for i in range(len(sysl))])
                        if r < INF or node_labels:
                            labs[(S, Ut, Dt)] = r
    return labs

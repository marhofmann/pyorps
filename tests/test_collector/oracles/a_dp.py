# Frozen port of the 2026-09-23 prototype `design_a/share_dp.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""Cut-state Dreyfus-Wagner for D_share (trench-tree model M2), plus the relaxed LB.

Labels are (S, U, D) with system tokens (mask, or (mask, type) when typed).
Levels are processed by increasing |S|.  Within a level one product Dijkstra over
(label, cell) handles grows (weight c(e) + R(L) l(e)) and junction transitions
(unary label changes at a non-turbine, non-root cell, cost J(deg)).
"""
from __future__ import annotations

import heapq
import itertools
from functools import lru_cache

from .a_model import INF, Inst, bits, disjoint_collections, enumerate_labels, popcount, set_partitions, union, valid_label


def tm(tok):
    return tok[0] if isinstance(tok, tuple) else tok


def label_ok(S, U, D):
    return valid_label(S, [tm(t) for t in U], [tm(t) for t in D], allow_free_blocks=True)


class ShareDP:
    def __init__(self, inst: Inst, allow_down=True, typed=False, allow_junction=True, mmax_override=None, root_transit=False):
        self.root_transit = root_transit
        self.I = inst
        self.typed = typed
        self.allow_junction = allow_junction
        self.labs = enumerate_labels(inst, allow_down=allow_down, typed=typed)
        if mmax_override is not None:
            self.labs = {L: r for L, r in self.labs.items() if len(L[1]) + len(L[2]) <= mmax_override}
        self.byS = {}
        for L in self.labs:
            self.byS.setdefault(L[0], []).append(L)
        self.ntypes = len(inst.types)
        self._merge_cache = {}
        self.seedarg, self.pred, self.aarg = {}, {}, {}
        self.stats = {"labels": len(self.labs), "merge_pairs": 0, "junction_trans": 0}

    # ---------------------------------------------------------------
    def new_tokens(self, mask):
        return [(mask, t) for t in range(self.ntypes)] if self.typed else [mask]

    def merge(self, b1, b2):
        key = (b1, b2)
        if key in self._merge_cache:
            return self._merge_cache[key]
        S1, U1, D1 = b1
        S2, U2, D2 = b2
        S = S1 | S2
        m12 = U1 & D2
        m21 = U2 & D1
        U = (U1 - m12) | (U2 - m21)
        D = (D1 - m21) | (D2 - m12)
        res = None
        if len(U) == len(U1 - m12) + len(U2 - m21) and len(D) == len(D1 - m21) + len(D2 - m12):
            if label_ok(S, U, D):
                L = (S, frozenset(U), frozenset(D))
                if L in self.labs:
                    res = L
        self._merge_cache[key] = res
        return res

    def junction_transitions(self, L):
        """Unary junction ops on label L at a non-turbine cell.  Returns [(L2, cost)]."""
        I = self.I
        S, U, D = L
        out = []
        if not self.allow_junction:
            return out
        full = I.full
        occupied = S | union(tm(t) for t in D)
        avail = full & ~occupied
        Ul = list(U)
        # (a) inputs = M_below (>=1 up-systems) + M_above (new down arcs), output goes up
        for r in range(0, len(Ul) + 1):
            for Mb in itertools.combinations(Ul, r):
                for Ma_masks in disjoint_collections(avail):
                    nin = r + len(Ma_masks)
                    if nin < 2:
                        continue
                    cost = I.junction_cost(nin + 1)
                    if cost == INF:
                        continue
                    sig = union(tm(t) for t in Mb) | union(Ma_masks)
                    # tokens for new down arcs (each gets any type if typed)
                    Ma_tok_choices = [self.new_tokens(C) for C in Ma_masks]
                    for Ma_toks in itertools.product(*Ma_tok_choices) if Ma_masks else [()]:
                        for out_tok in self.new_tokens(sig):
                            U2 = (U - set(Mb)) | {out_tok}
                            D2 = D | set(Ma_toks)
                            L2 = (S, frozenset(U2), frozenset(D2))
                            if L2 in self.labs and label_ok(*L2):
                                out.append((L2, cost))
        # (b) inputs all from above, output goes down into the subtree (matches a D-set)
        for C in D:
            Cm = tm(C)
            atoms = [1 << i for i in bits(Cm)]
            for part in set_partitions(atoms):
                if len(part) < 2:
                    continue
                cost = I.junction_cost(len(part) + 1)
                if cost == INF:
                    continue
                masks = [union(b) for b in part]
                for toks in itertools.product(*[self.new_tokens(mk) for mk in masks]):
                    D2 = (D - {C}) | set(toks)
                    if len(D2) != len(D) - 1 + len(toks):
                        continue
                    L2 = (S, U, frozenset(D2))
                    if L2 in self.labs and label_ok(*L2):
                        out.append((L2, cost))
        return out

    def turbine_leaf_options(self, i):
        """Child-arc collections M (<=2 panels) for turbine i: list of token tuples."""
        I = self.I
        others = I.full ^ (1 << i)
        res = []
        for M in disjoint_collections(others, max_sets=2):
            if len(M) > 2:
                continue
            if popcount(union(M)) + 1 > I.kmax:
                continue
            Ml = sorted(M)
            for toks in itertools.product(*[self.new_tokens(C) for C in Ml]) if Ml else [()]:
                res.append(tuple(toks))
        return res

    # ---------------------------------------------------------------
    def solve(self):
        I = self.I
        n = I.n
        is_turb = [False] * n
        for t in I.turb:
            is_turb[t] = True
        steiner = [v for v in range(n) if not is_turb[v] and (v != I.root or self.root_transit)]
        term_cells = list(I.turb) + [I.root]
        F = {}
        Aarr = {}
        leaf_opts = {i: self.turbine_leaf_options(i) for i in range(I.k)}
        order = sorted(self.byS.keys(), key=lambda s: (popcount(s), s))
        for S in order:
            labs = self.byS[S]
            idx = {L: j for j, L in enumerate(labs)}
            seeds = [[INF] * n for _ in labs]
            # --- turbine seeds
            for i in bits(S):
                t = I.turb[i]
                rest = S ^ (1 << i)
                for Mt in leaf_opts[i]:
                    Mm = [tm(x) for x in Mt]
                    inside = [x for x in Mt if (tm(x) & rest) == tm(x)]
                    outside = [x for x in Mt if (tm(x) & S) == 0]
                    if len(inside) + len(outside) != len(Mt):
                        continue
                    if union(tm(x) for x in inside) != rest:
                        continue
                    sig = (1 << i) | union(Mm)
                    base = I.panel * (1 + len(Mt))
                    best_grp = []
                    if inside:
                        best_ch = INF
                        for grouping in set_partitions(inside):
                            tot = 0.0
                            for grp in grouping:
                                child = (union(tm(x) for x in grp), frozenset(grp), frozenset())
                                a = Aarr.get(child, {}).get(t, INF)
                                tot += a
                                if tot == INF:
                                    break
                            if tot < best_ch:
                                best_ch, best_grp = tot, grouping
                    else:
                        best_ch = 0.0
                    if best_ch == INF:
                        continue
                    for out_tok in self.new_tokens(sig):
                        L = (S, frozenset([out_tok]), frozenset(outside))
                        j = idx.get(L)
                        if j is None:
                            continue
                        val = base + best_ch
                        if val < seeds[j][t]:
                            seeds[j][t] = val
                            self.seedarg[(L, t)] = ("turb", i, Mt, best_grp)
            # --- bundle merges at Steiner cells
            low = S & -S
            rest = S ^ low
            for sub in _submasks(rest):
                S1 = sub | low
                if S1 == S:
                    continue
                S2 = S ^ S1
                for b1 in self.byS.get(S1, []):
                    f1 = F[b1]
                    for b2 in self.byS.get(S2, []):
                        L = self.merge(b1, b2)
                        if L is None:
                            continue
                        j = idx.get(L)
                        if j is None:
                            continue
                        self.stats["merge_pairs"] += 1
                        f2 = F[b2]
                        row = seeds[j]
                        for v in steiner:
                            sv = f1[v] + f2[v]
                            if sv < row[v]:
                                row[v] = sv
                                self.seedarg[(L, v)] = ("merge", b1, b2)
            # --- product Dijkstra with junction transitions
            trans = [[(idx[L2], c) for (L2, c) in self.junction_transitions(L) if L2 in idx] for L in labs]
            pred = [[None] * n for _ in labs]
            self.stats["junction_trans"] += sum(len(x) for x in trans)
            rates = [self.labs[L] for L in labs]
            dist = [row[:] for row in seeds]
            heap = [(dist[j][v], j, v) for j in range(len(labs)) for v in range(n) if dist[j][v] < INF]
            heapq.heapify(heap)
            while heap:
                d, j, v = heapq.heappop(heap)
                if d > dist[j][v]:
                    continue
                if v == I.root and not self.root_transit:
                    continue
                r = rates[j]
                for (w, ei) in I.adj[v]:
                    if is_turb[w] or (w == I.root and not self.root_transit):
                        continue
                    _, _, c, l = I.edges[ei]
                    nd = d + c + r * l
                    if nd < dist[j][w]:
                        dist[j][w] = nd
                        pred[j][w] = ("grow", v, ei)
                        heapq.heappush(heap, (nd, j, w))
                if not is_turb[v] and (v != I.root or self.root_transit):
                    for (j2, cost) in trans[j]:
                        nd = d + cost
                        if nd < dist[j2][v]:
                            dist[j2][v] = nd
                            pred[j2][v] = ("junc", labs[j], cost)
                            heapq.heappush(heap, (nd, j2, v))
            for j, L in enumerate(labs):
                F[L] = dist[j]
                self.pred[L] = pred[j]
                r = rates[j]
                arr = {}
                aarg = {}
                for y in term_cells:
                    best = INF
                    for (x, ei) in I.adj[y]:
                        if x == I.root and not self.root_transit:
                            continue
                        if x == y:
                            continue
                        _, _, c, l = I.edges[ei]
                        val = dist[j][x] + c + r * l
                        if val < best:
                            best = val
                            aarg[y] = (x, ei)
                    arr[y] = best
                Aarr[L] = arr
                self.aarg[L] = aarg
        # --- root assembly
        bestS = {}
        self.rootlab = {}
        for L in self.labs:
            S, U, D = L
            if D:
                continue
            val = Aarr.get(L, {}).get(I.root, INF) + I.bay * len(U)
            if val < bestS.get(S, INF):
                bestS[S] = val
                self.rootlab[S] = L
        H = {0: 0.0}
        self.Harg = {}
        for X in range(1, I.full + 1):
            low = X & -X
            rest = X ^ low
            best = INF
            for sub in _submasks(rest):
                Y = sub | low
                b = bestS.get(Y, INF)
                if b < INF:
                    v = b + H[X ^ Y]
                    if v < best:
                        best = v
                        self.Harg[X] = Y
            H[X] = best
        self.F, self.Aarr = F, Aarr
        return H[I.full]


def _submasks(mask):
    s = mask
    while True:
        yield s
        if s == 0:
            break
        s = (s - 1) & mask


# ---------------------------------------------------------------------------
# Relaxed lower bound: subset-only DW with rho_hat(S) = min_L R(L), zero J,
# bays >= nb_min(S), panels >= 1 per turbine, no-transit kept.
# ---------------------------------------------------------------------------
def relaxed_lb(inst: Inst, labs=None):
    I = inst
    if labs is None:
        labs = enumerate_labels(I, allow_down=True)
    rho_hat, nb_min = {}, {}
    for (S, U, D), r in labs.items():
        if r == INF:
            continue
        rho_hat[S] = min(rho_hat.get(S, INF), r)
        if not D:
            nb_min[S] = min(nb_min.get(S, 10 ** 9), len(U))
    n = I.n
    is_turb = [False] * n
    for t in I.turb:
        is_turb[t] = True
    steiner = [v for v in range(n) if not is_turb[v] and v != I.root]
    F, Aarr = {}, {}
    for S in sorted(range(1, I.full + 1), key=lambda s: (popcount(s), s)):
        seed = [INF] * n
        for i in bits(S):
            t = I.turb[i]
            rest = S ^ (1 << i)
            # G_t(rest) = min over partitions of rest into arriving child sets
            G = {0: 0.0}
            for X in sorted([x for x in _submasks(rest)], key=popcount):
                if X == 0:
                    continue
                lowx = X & -X
                best = INF
                for sub in _submasks(X ^ lowx):
                    Y = sub | lowx
                    a = Aarr.get(Y, {}).get(t, INF)
                    if a < INF and G.get(X ^ Y, INF) < INF:
                        best = min(best, a + G[X ^ Y])
                G[X] = best
            seed[t] = min(seed[t], G[rest])
        low = S & -S
        for sub in _submasks(S ^ low):
            S1 = sub | low
            if S1 == S:
                continue
            f1, f2 = F[S1], F[S ^ S1]
            for v in steiner:
                if f1[v] + f2[v] < seed[v]:
                    seed[v] = f1[v] + f2[v]
        r = rho_hat.get(S, INF)
        dist = seed[:]
        heap = [(dist[v], v) for v in range(n) if dist[v] < INF]
        heapq.heapify(heap)
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v] or v == I.root or r == INF:
                continue
            for (w, ei) in I.adj[v]:
                if is_turb[w] or w == I.root:
                    continue
                _, _, c, l = I.edges[ei]
                nd = d + c + r * l
                if nd < dist[w]:
                    dist[w] = nd
                    heapq.heappush(heap, (nd, w))
        F[S] = dist
        arr = {}
        for y in list(I.turb) + [I.root]:
            best = INF
            for (x, ei) in I.adj[y]:
                if x == I.root or r == INF:
                    continue
                _, _, c, l = I.edges[ei]
                best = min(best, dist[x] + c + r * l)
            arr[y] = best
        Aarr[S] = arr
    H = {0: 0.0}
    for X in range(1, I.full + 1):
        low = X & -X
        best = INF
        for sub in _submasks(X ^ low):
            Y = sub | low
            a = Aarr[Y][I.root]
            if a < INF and Y in nb_min:
                best = min(best, a + I.bay * nb_min[Y] + H[X ^ Y])
        H[X] = best
    return H[I.full] + I.panel * I.k

# Frozen port of the 2026-09-23 prototype `design_a/share_brute.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""Exhaustive oracle for D_share = M2 (trench-tree model), independent of the DP.

M2 design = abstract trench tree T (rooted at g) embedded in G (edges -> steps,
nodes -> cells, not necessarily injective) + electrical arborescence H whose arcs run
along T-paths.  Enumerated as: every subtree T of the expansion G^(K) (K copies of each
non-terminal cell; terminal cells single) that contains all terminals and has only
terminal leaves, times every H (turbines + junction slots) placed on T.
Also provides the physical network model M1 cost of an explicit walk design.
"""
from __future__ import annotations

import itertools
from collections import deque

from .a_model import INF, Inst, bits, popcount, union


def expand(inst: Inst, K: int):
    term = set(inst.turb) | {inst.root}
    nodes, copies = [], {}
    for v in range(inst.n):
        kk = 1 if v in term else K
        copies[v] = []
        for c in range(kk):
            copies[v].append(len(nodes))
            nodes.append((v, c))
    E = []
    for ei, (u, w, c, l) in enumerate(inst.edges):
        for a in copies[u]:
            for b in copies[w]:
                E.append((a, b, c, l, ei))
    adj = [[] for _ in nodes]
    for fi, (a, b, *_rest) in enumerate(E):
        adj[a].append((b, fi))
        adj[b].append((a, fi))
    return nodes, E, adj, copies


def enum_trees(nN, E, adj, root, required_mask, max_trees=5_000_000):
    out = []

    def rec(nm, tedges, cand, excl):
        if len(out) >= max_trees:
            return
        if not cand:
            if nm & required_mask == required_mask:
                out.append(tedges)
            return
        e = cand[0]
        rest = cand[1:]
        rec(nm, tedges, rest, excl | (1 << e))
        a, b = E[e][0], E[e][1]
        new = b if (nm >> a) & 1 else a
        nm2 = nm | (1 << new)
        rest2 = [f for f in rest if not (((nm2 >> E[f][0]) & 1) and ((nm2 >> E[f][1]) & 1))]
        add = [f for (x, f) in adj[new] if not (excl >> f) & 1 and not (nm2 >> x) & 1]
        rec(nm2, tedges + (e,), rest2 + add, excl)

    rec(1 << root, (), [f for (x, f) in adj[root]], 0)
    return out


def h_templates(k, kmax, dmax, jmax=None):
    """Abstract electrical arborescences: turbines 0..k-1, junction slots k..k+j-1, root -1."""
    if jmax is None:
        jmax = k - 1
    res = []
    for j in range(jmax + 1):
        N = k + j
        choices = [[-1] + [x for x in range(N) if x != a] for a in range(N)]
        for par in itertools.product(*choices):
            ok = True
            for a in range(N):
                seen, x = 0, a
                while x != -1:
                    if (seen >> x) & 1:
                        ok = False
                        break
                    seen |= 1 << x
                    x = par[x]
                if not ok:
                    break
            if not ok:
                continue
            nch = [0] * N
            for a in range(N):
                if par[a] != -1:
                    nch[par[a]] += 1
            if any(nch[a] > 2 for a in range(k)):
                continue
            if any(nch[a] < 2 or nch[a] > dmax - 1 for a in range(k, N)):
                continue
            down = [0] * N
            for a in range(k):
                x = a
                while x != -1:
                    down[x] |= 1 << a
                    x = par[x]
            if any(popcount(down[a]) > kmax for a in range(k)):
                continue
            # canonical order of junction slots: by downstream mask (distinct for distinct junctions)
            dm = [down[a] for a in range(k, N)]
            if any(dm[i] >= dm[i + 1] for i in range(len(dm) - 1)):
                continue
            res.append((j, par, down, nch))
    return res


class BruteM2:
    def __init__(self, inst: Inst, K=2, jmax=None, typed=False, max_trees=3_000_000, ronce=True):
        self.ronce = ronce
        self.max_trees = max_trees
        self.I = inst
        self.K = K
        self.typed = typed
        self.nodes, self.E, self.adj, self.copies = expand(inst, K)
        self.templates = h_templates(inst.k, inst.kmax, inst.dmax, jmax)
        self.rcache = {}

    def rate(self, systems):
        key = tuple(sorted(systems))
        r = self.rcache.get(key)
        if r is None:
            r = self.I.rate(list(key))
            self.rcache[key] = r
        return r

    def solve(self, return_design=False):
        I = self.I
        nodes, E, adj = self.nodes, self.E, self.adj
        root = self.copies[I.root][0]
        tnode = [self.copies[t][0] for t in I.turb]
        term_nodes = set(tnode) | {root}
        req = (1 << root)
        for x in tnode:
            req |= 1 << x
        trees = enum_trees(len(nodes), E, adj, root, req, max_trees=self.max_trees)
        self.truncated = len(trees) >= self.max_trees
        best, best_design = INF, None
        ntrees = 0
        for tedges in trees:
            # leaves must be terminals
            deg = {}
            for e in tedges:
                a, b = E[e][0], E[e][1]
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
            if any(d == 1 and x not in term_nodes for x, d in deg.items()):
                continue
            # symmetry: copy c of a cell used only if copy c-1 used
            used = set(deg)
            skip = False
            for x in used:
                cell, c = nodes[x]
                if c > 0 and self.copies[cell][c - 1] not in used:
                    skip = True
                    break
            if skip:
                continue
            ntrees += 1
            val, des = self.eval_tree(tedges, root, tnode, term_nodes)
            if val < best:
                best, best_design = val, (tedges, des)
        self.ntrees = ntrees
        if return_design:
            return best, best_design
        return best

    def eval_tree(self, tedges, root, tnode, term_nodes):
        I, E = self.I, self.E
        tadj = {}
        for idx, e in enumerate(tedges):
            a, b = E[e][0], E[e][1]
            tadj.setdefault(a, []).append((b, idx))
            tadj.setdefault(b, []).append((a, idx))
        parent, pedge, depth = {root: None}, {root: None}, {root: 0}
        dq = deque([root])
        while dq:
            x = dq.popleft()
            for (y, idx) in tadj.get(x, []):
                if y not in parent:
                    parent[y], pedge[y], depth[y] = x, idx, depth[x] + 1
                    dq.append(y)
        cands = [x for x in parent if x not in term_nodes]

        pcache = {}

        def path(a, b):
            key = (a, b)
            if key in pcache:
                return pcache[key]
            up, dn = [], []
            x, y = a, b
            na, nb = [a], [b]
            while depth[x] > depth[y]:
                up.append(pedge[x]); x = parent[x]; na.append(x)
            while depth[y] > depth[x]:
                dn.append(pedge[y]); y = parent[y]; nb.append(y)
            while x != y:
                up.append(pedge[x]); x = parent[x]; na.append(x)
                dn.append(pedge[y]); y = parent[y]; nb.append(y)
            seq = na + nb[-2::-1]
            interior = seq[1:-1]
            bad = any(z in term_nodes for z in interior)
            res = None if bad else ([(e, +1) for e in up] + [(e, -1) for e in reversed(dn)])
            pcache[key] = res
            return res

        nte = len(tedges)
        best, best_des = INF, None
        ntypes = len(I.types)
        for (j, par, down, nch) in self.templates:
            for mp in itertools.product(cands, repeat=j):
                pos = list(tnode) + list(mp)
                arcs = []
                ok = True
                for a in range(len(pos)):
                    p = par[a]
                    dst = root if p == -1 else pos[p]
                    pth = path(pos[a], dst)
                    if pth is None:
                        ok = False
                        break
                    arcs.append((down[a], pth))
                if not ok:
                    continue
                node_cost = 0.0
                for a in range(I.k):
                    node_cost += I.panel * (1 + nch[a])
                for a in range(I.k, len(pos)):
                    node_cost += I.junction_cost(nch[a] + 1)
                node_cost += I.bay * sum(1 for a in range(len(pos)) if par[a] == -1)
                if node_cost == INF:
                    continue
                type_iter = itertools.product(range(ntypes), repeat=len(arcs)) if self.typed else [None]
                for tv in type_iter:
                    upm = [[] for _ in range(nte)]
                    dnm = [[] for _ in range(nte)]
                    for ai, (mask, pth) in enumerate(arcs):
                        tok = mask if tv is None else (mask, tv[ai])
                        for (e, d) in pth:
                            (upm if d > 0 else dnm)[e].append(tok)
                    tot = node_cost
                    for e in range(nte):
                        um, dm = upm[e], dnm[e]
                        if not self.ronce:
                            r = self.rate(um + dm)
                            if r == INF:
                                tot = INF
                                break
                            _, _, c, l, _ = E[tedges[e]]
                            tot += c + r * l
                            if tot >= best:
                                break
                            continue
                        # R-once: per direction pairwise disjoint
                        acc = 0
                        for t in um:
                            mk = t if tv is None else t[0]
                            if acc & mk:
                                tot = INF
                                break
                            acc |= mk
                        if tot == INF:
                            break
                        acc = 0
                        for t in dm:
                            mk = t if tv is None else t[0]
                            if acc & mk:
                                tot = INF
                                break
                            acc |= mk
                        if tot == INF:
                            break
                        r = self.rate(um + dm)
                        if r == INF:
                            tot = INF
                            break
                        _, _, c, l, _ = E[tedges[e]]
                        tot += c + r * l
                        if tot >= best:
                            break
                    if tot < best:
                        best, best_des = tot, (j, par, tuple(mp), tv)
        return best, best_des


def m1_cost(inst: Inst, arcs, node_cost):
    """Physical network model M1: arcs = [(mask, [cells...])]; trench once per used step,
    all systems on a step share it (derated jointly)."""
    step = {}
    for mask, cells in arcs:
        for a, b in zip(cells[:-1], cells[1:]):
            key = (min(a, b), max(a, b))
            step.setdefault(key, []).append(mask)
    emap = {}
    for (u, w, c, l) in inst.edges:
        emap[(min(u, w), max(u, w))] = (c, l)
    tot = node_cost
    for key, masks in step.items():
        c, l = emap[key]
        tot += c + inst.rate(masks) * l
    return tot

# Frozen port of the 2026-09-23 prototype `design_b/csdw.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""Cut-signature Dreyfus-Wagner (CS-DW) for the shared-trench MV collector, on a graph.

States (X, sig): X = turbine subset below a trench edge, sig = (U, D) flow multisets.
Per X, one *layered* Dijkstra over (sig, cell): spatial steps only in drainable layers,
zero-length inter-layer arcs = local operations (turn / joints) at field cells.
Seeds: binary merges (multiset union) at field cells, and the turbine rule at turbine cells.
Root: MV(g) = min over (U, ()) of L(A, (U, ()), g) + bay * |U|.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .b_model import INF, unary_ops, union


class Graph:
    def __init__(self, N, edges):
        self.N = N
        self.nbrs = [[] for _ in range(N)]
        for u, v, c, l in edges:
            self.nbrs[u].append((v, c, l))
            self.nbrs[v].append((u, c, l))


def king_graph(R, C, cell, diag=True):
    """8-connected grid; step trench cost = mean endpoint cost x length (pyorps-like)."""
    idx = lambda r, c: r * C + c
    edges = []
    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if diag else [])
    for r in range(R):
        for c in range(C):
            for dr, dc in steps:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < R and 0 <= c2 < C:
                    l = math.hypot(dr, dc)
                    cc = 0.5 * (cell[r, c] + cell[r2, c2]) * l
                    edges.append((idx(r, c), idx(r2, c2), cc, l))
    return Graph(R * C, edges)


def _submasks_with_low(X):
    low = X & -X
    rest = X ^ low
    sub = rest
    while True:
        X1 = sub | low
        if X1 != X:
            yield X1
        if sub == 0:
            break
        sub = (sub - 1) & rest


def _down_multisets(max_len, max_flow):
    out = [()]
    for L in range(1, max_len + 1):
        for comb in itertools.combinations_with_replacement(range(1, max_flow + 1), L):
            out.append(tuple(sorted(comb)))
    return out


def csdw(G, A, P, with_down=True, stats=None, r1=False, outk=None, bmax=None):
    N, n = G.N, len(A)
    isT = np.zeros(N, bool)
    isT[list(A)] = True
    full = (1 << n) - 1
    L = {}
    order = sorted(range(1, full + 1), key=lambda x: bin(x).count("1"))
    wstep = 1.0 + P.omega
    n_drained = 0
    n_layers = 0
    n_merge_pairs = 0
    for X in order:
        s = bin(X).count("1")
        seeds = {}

        def put(sig, arr):
            if sig in seeds:
                np.minimum(seeds[sig], arr, out=seeds[sig])
            else:
                seeds[sig] = arr.copy()

        # binary merges at field cells
        if s >= 2:
            for X1 in _submasks_with_low(X):
                X2 = X ^ X1
                for s1, a1 in L[X1].items():
                    for s2, a2 in L[X2].items():
                        sig = union(s1, s2)
                        if not P.transient_ok(sig, s):
                            continue
                        n_merge_pairs += 1
                        val = a1 + a2
                        val[isT] = INF
                        if bmax is not None:
                            val[val + outk[s] > bmax] = INF
                        if np.isfinite(val).any():
                            put(sig, val)
        # turbine rule
        for i in range(n):
            if not (X >> i) & 1:
                continue
            t = A[i]
            Y = X & ~(1 << i)
            outs = []
            maxd = max(P.k_max - 1, 0)
            if Y == 0:
                for Dp in (_down_multisets(P.max_in, maxd) if with_down else [()]):
                    k_out = 1 + sum(Dp)
                    if k_out > P.k_max:
                        continue
                    outs.append((((k_out,), Dp), P.panel * len(Dp)))
            else:
                def arr(Ym, sig):
                    if sig not in L[Ym] or not P.valid(sig):
                        return INF
                    lab = L[Ym][sig]
                    mu = P.mu(sig)
                    best = INF
                    for x, c, l in G.nbrs[t]:
                        if lab[x] < INF:
                            best = min(best, lab[x] + wstep * c + mu * l)
                    return best
                for sig in list(L[Y].keys()):
                    U, D = sig
                    if D or len(U) > P.max_in or not P.valid(sig):
                        continue
                    a = arr(Y, sig)
                    if a == INF:
                        continue
                    for Dp in (_down_multisets(P.max_in - len(U), maxd) if with_down else [()]):
                        k_out = 1 + sum(U) + sum(Dp)
                        if k_out > P.k_max:
                            continue
                        outs.append((((k_out,), Dp), a + P.panel * (len(U) + len(Dp))))
                if P.max_in >= 2:
                    for Y1 in _submasks_with_low(Y):
                        Y2 = Y ^ Y1
                        for s1 in L[Y1]:
                            if s1[1] or len(s1[0]) != 1:
                                continue
                            a1 = arr(Y1, s1)
                            if a1 == INF:
                                continue
                            for s2 in L[Y2]:
                                if s2[1] or len(s2[0]) != 1:
                                    continue
                                k_out = 1 + s1[0][0] + s2[0][0]
                                if k_out > P.k_max:
                                    continue
                                a2 = arr(Y2, s2)
                                if a2 == INF:
                                    continue
                                outs.append((((k_out,), ()), a1 + a2 + 2 * P.panel))
            for sig, cost in outs:
                if not P.valid(sig):
                    continue
                v = np.full(N, INF)
                v[t] = cost
                put(sig, v)
        # layered drain for X
        dist = {sig: a for sig, a in seeds.items()}
        heap = []
        for sig, a in dist.items():
            for v in np.nonzero(np.isfinite(a))[0]:
                heap.append((a[v], sig, int(v)))
        heapq.heapify(heap)
        settled = 0
        settled_dr = 0
        ok_s = outk[s] if bmax is not None else None
        while heap:
            d, sig, v = heapq.heappop(heap)
            if d > dist[sig][v]:
                continue
            if ok_s is not None and d + ok_s[v] > bmax:
                dist[sig][v] = INF
                continue
            settled += 1
            if P.valid(sig):
                settled_dr += 1
            if P.valid(sig) and not (r1 and X == full and sig[1]):
                mu = P.mu(sig)
                lab = dist[sig]
                for u, c, l in G.nbrs[v]:
                    if isT[u]:
                        continue
                    nd = d + wstep * c + mu * l
                    if nd < lab[u]:
                        lab[u] = nd
                        heapq.heappush(heap, (nd, sig, u))
            if not isT[v]:
                for sig2, nj in unary_ops(sig, P.n, with_down):
                    if not P.transient_ok(sig2, s):
                        continue
                    nd = d + nj * P.J
                    lab2 = dist.get(sig2)
                    if lab2 is None:
                        lab2 = np.full(N, INF)
                        dist[sig2] = lab2
                    if nd < lab2[v]:
                        lab2[v] = nd
                        heapq.heappush(heap, (nd, sig2, v))
        L[X] = {sig: a for sig, a in dist.items() if np.isfinite(a).any()}
        if stats is not None:
            stats.setdefault("settled", {})[X] = settled
            stats.setdefault("settled_drainable", {})[X] = settled_dr
            stats.setdefault("finite", {})[X] = {sig: int(np.isfinite(a).sum()) for sig, a in L[X].items()}
        n_layers += len(L[X])
        n_drained += sum(1 for sig in L[X] if P.valid(sig))
    MV = np.full(N, INF)
    for sig, a in L[full].items():
        if sig[1] == () and max(sig[0]) <= P.k_bay:
            np.minimum(MV, a + P.bay * len(sig[0]), out=MV)
    MV[isT] = INF
    if stats is not None:
        stats.update(drained_layers=n_drained, all_layers=n_layers, merge_pairs=n_merge_pairs)
    return MV, L


def mu_star(P, kmax_flow=None, m_cap=None):
    """mu*(k) = min per-metre system cost over drainable signatures with net up-flow k."""
    n = P.n
    m_cap = m_cap or (2 * n - 1)
    best = {k: INF for k in range(1, n + 1)}
    for mU in range(1, m_cap + 1):
        for U in itertools.combinations_with_replacement(range(1, n + 1), mU):
            for mD in range(0, m_cap - mU + 1):
                for D in itertools.combinations_with_replacement(range(1, n + 1), mD):
                    k = sum(U) - sum(D)
                    if 1 <= k <= n:
                        mu = P.mu((tuple(U), tuple(D)))
                        if mu < best[k]:
                            best[k] = mu
                if P.m_max is not None and mU + mD >= P.m_max:
                    break
    return best


def relaxed_dw(G, A, P, bay_term="corrected"):
    """Admissible LB for D_share: flow-dependent DW over subsets with weight
    (1+omega)c + mu*(|X|) l; transit and merges allowed everywhere (relaxations);
    root: at least ceil(n / kcap1) systems end at the root, each paying a bay.

    PORT: bay_term="corrected" (default) is the verifier's fix V-03,
    bay*ceil(n/k_bay) + min(bay, J)*(ceil(n/kcap1) - ceil(n/k_bay))^+ ; the
    prototype's bay*ceil(n/min(kcap1, k_bay)) ("naive") is inadmissible when a
    joint on the UW cell may merge feeders (lb_bay: 41.0 > optimum 23.0) and is
    kept only so that regression stays reproducible."""
    N, n = G.N, len(A)
    ms = mu_star(P)
    full = (1 << n) - 1
    order = sorted(range(1, full + 1), key=lambda x: bin(x).count("1"))
    I = {}
    for X in order:
        s = bin(X).count("1")
        seed = np.full(N, INF)
        if s == 1:
            seed[A[X.bit_length() - 1]] = 0.0
        else:
            for X1 in _submasks_with_low(X):
                np.minimum(seed, I[X1] + I[X ^ X1], out=seed)
        mu = ms[s]
        dist = seed.copy()
        if mu < INF:
            h = [(dist[v], v) for v in np.nonzero(np.isfinite(dist))[0]]
            heapq.heapify(h)
            while h:
                d, v = heapq.heappop(h)
                if d > dist[v]:
                    continue
                for u, c, l in G.nbrs[v]:
                    nd = d + (1.0 + P.omega) * c + mu * l
                    if nd < dist[u]:
                        dist[u] = nd
                        heapq.heappush(h, (nd, u))
        I[X] = dist
    kcap1 = 0
    while kcap1 + 1 <= n and P.rho(kcap1 + 1, 1) < INF:
        kcap1 += 1
    if bay_term == "naive":
        nb = -(-n // max(min(kcap1, P.k_bay), 1))
        bays = P.bay * nb
    elif bay_term == "corrected":
        nb_bay = -(-n // max(P.k_bay, 1))
        nb_cap = -(-n // max(kcap1, 1))
        bays = P.bay * nb_bay + min(P.bay, P.J) * max(nb_cap - nb_bay, 0)
    else:
        raise ValueError(f"bay_term must be 'corrected' or 'naive', got {bay_term!r}")
    lb = I[full] + bays
    lb[list(A)] = INF
    return lb, ms


def aggregated_bounds(G, A, P, ms):
    """Consistent admissible completion bounds Out_k(v), k = 1..n (root allowed anywhere).
    In_j: aggregated relaxed inside (disjointness ignored); Out_n = 0; Out_k seeded by
    min_j In_j + Out_{k+j}, drained with the relaxed weight of net flow k."""
    N, n = G.N, len(A)

    def drain(seed, mu):
        dist = seed.copy()
        if mu == INF:
            return dist
        h = [(dist[v], v) for v in np.nonzero(np.isfinite(dist))[0]]
        heapq.heapify(h)
        while h:
            d, v = heapq.heappop(h)
            if d > dist[v]:
                continue
            for u, c, l in G.nbrs[v]:
                nd = d + (1.0 + P.omega) * c + mu * l
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(h, (nd, u))
        return dist
    In = {}
    s0 = np.full(N, INF)
    s0[list(A)] = 0.0
    In[1] = drain(s0, ms[1])
    for j in range(2, n + 1):
        seed = np.full(N, INF)
        for a in range(1, j // 2 + 1):
            np.minimum(seed, In[a] + In[j - a], out=seed)
        In[j] = drain(seed, ms[j])
    Out = {n: np.zeros(N)}
    for k in range(n - 1, 0, -1):
        seed = np.full(N, INF)
        for j in range(1, n - k + 1):
            np.minimum(seed, In[j] + Out[k + j], out=seed)
        Out[k] = drain(seed, ms[k])
    return Out, In

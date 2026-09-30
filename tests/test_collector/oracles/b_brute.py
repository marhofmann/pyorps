# Frozen port of the 2026-09-23 prototype `design_b/brute.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""Brute-force oracle for D_share on a small graph.

Independent of the CS-DW recursion: it enumerates
  * every electrical design E (radial tree on turbines + q joints + root, flows, k_max, panels),
  * every abstract trench topology tau (labelled tree on turbines, joints, root and up to
    n_leaves-2 pure Steiner nodes of degree >= 3),
routes every system along its unique tau-path (no transit through a turbine or the root node),
prices every tau-edge by its own system multiset, and embeds tau exactly with a fixed-topology
tree DP over all-pairs no-transit distances (placements of joints/Steiner nodes free, including
co-location). One tree DP gives the value for every root cell g at once.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from .b_model import INF


def all_pairs(G, mu, omega, isT):
    N = G.N
    D = np.full((N, N), INF)
    w1 = 1.0 + omega
    for src in range(N):
        dist = [INF] * N
        dist[src] = 0.0
        h = [(0.0, src)]
        while h:
            d, v = heapq.heappop(h)
            if d > dist[v]:
                continue
            if isT[v] and v != src:
                continue            # no transit through a turbine cell
            for u, c, l in G.nbrs[v]:
                nd = d + w1 * c + mu * l
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(h, (nd, u))
        D[src] = dist
    return D


def electrical_designs(n, q, P):
    R = n + q
    nodes = list(range(n + q))
    out = []
    for par in itertools.product(range(n + q + 1), repeat=n + q):
        if any(par[x] == x for x in nodes):
            continue
        ok = True
        for x in nodes:           # acyclic, reaches root
            seen = set()
            y = x
            while y != R:
                if y in seen:
                    ok = False
                    break
                seen.add(y)
                y = par[y]
            if not ok:
                break
        if not ok:
            continue
        ch = {x: [y for y in nodes if par[y] == x] for x in nodes + [R]}
        if any(len(ch[j]) < 2 for j in range(n, n + q)):
            continue
        if any(len(ch[t]) > P.max_in for t in range(n)):
            continue
        flow = {}

        def fl(x):
            if x in flow:
                return flow[x]
            v = (1 if x < n else 0) + sum(fl(y) for y in ch[x])
            flow[x] = v
            return v
        for x in nodes:
            fl(x)
        if any(flow[t] > P.k_max for t in range(n)):
            continue
        if any(flow[y] > P.k_bay for y in ch[R]):
            continue
        cost = P.J * sum(len(ch[j]) - 1 for j in range(n, n + q)) \
            + P.bay * len(ch[R]) + P.panel * sum(len(ch[t]) for t in range(n))
        out.append((par, flow, cost))
    return out


def prufer_trees(Nt):
    if Nt == 2:
        yield [(0, 1)]
        return
    for seq in itertools.product(range(Nt), repeat=Nt - 2):
        deg = [1] * Nt
        for x in seq:
            deg[x] += 1
        edges = []
        deg2 = deg[:]
        for x in seq:
            leaf = min(i for i in range(Nt) if deg2[i] == 1)
            edges.append((leaf, x))
            deg2[leaf] -= 1
            deg2[x] -= 1
        u, v = [i for i in range(Nt) if deg2[i] == 1]
        edges.append((u, v))
        yield edges


_TOPO_CACHE = {}


def topologies(n, q, sS, P):
    key0 = (n, q, sS, P.max_in)
    if key0 in _TOPO_CACHE:
        return _TOPO_CACHE[key0]
    res = _topologies(n, q, sS, P)
    _TOPO_CACHE[key0] = res
    return res


def _topologies(n, q, sS, P):
    Nt = n + q + 1 + sS
    steiner = list(range(n + q + 1, Nt))
    seen = set()
    res = []
    for edges in prufer_trees(Nt):
        deg = [0] * Nt
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        if any(deg[s] < 3 for s in steiner):
            continue
        if any(deg[t] > P.max_in + 1 for t in range(n)):
            continue
        if any(deg[j] < 2 for j in range(n, n + q)):
            continue
        keys = []
        for perm in itertools.permutations(steiner):
            mp = {s: p for s, p in zip(steiner, perm)}
            f = lambda x: mp.get(x, x)
            keys.append(tuple(sorted(tuple(sorted((f(a), f(b)))) for a, b in edges)))
        key = min(keys)
        if key in seen:
            continue
        seen.add(key)
        res.append(edges)
    return res


def brute(G, A, P, q_max=1, dist_cache=None):
    N, n = G.N, len(A)
    isT = np.zeros(N, bool)
    isT[list(A)] = True
    if dist_cache is None:
        dist_cache = {}
    Zinv = np.full((N, N), INF)
    np.fill_diagonal(Zinv, 0.0)

    def Dmat(flows):
        mu = P.mu_flows(flows)
        if mu == INF:
            return Zinv                      # only a zero-length (co-located) edge
        if mu not in dist_cache:
            dist_cache[mu] = all_pairs(G, mu, P.omega, isT)
        return dist_cache[mu]

    best = np.full(N, INF)
    n_pairs = 0
    for q in range(0, q_max + 1):
        designs = electrical_designs(n, q, P)
        if not designs:
            continue
        R = n + q
        for sS in range(0, max(0, n + 1 - 2) + 1):
            for edges in topologies(n, q, sS, P):
                Nt = n + q + 1 + sS
                adj = [[] for _ in range(Nt)]
                for a, b in edges:
                    adj[a].append(b)
                    adj[b].append(a)
                parent = [-1] * Nt
                depth = [0] * Nt
                orderl = [R]
                parent[R] = R
                for x in orderl:
                    for y in adj[x]:
                        if parent[y] == -1:
                            parent[y] = x
                            depth[y] = depth[x] + 1
                            orderl.append(y)

                def path_edges(a, b):
                    ea, eb = [], []
                    inter = []
                    while depth[a] > depth[b]:
                        ea.append((a, parent[a])); a = parent[a]; inter.append(a)
                    while depth[b] > depth[a]:
                        eb.append((b, parent[b])); b = parent[b]; inter.append(b)
                    while a != b:
                        ea.append((a, parent[a])); eb.append((b, parent[b]))
                        a = parent[a]; b = parent[b]
                        inter.append(a); inter.append(b)
                    return ea + eb, inter
                for par, flow, ncost in designs:
                    ef = {}
                    ok = True
                    for x in range(n + q):
                        p = par[x]
                        pe, inter = path_edges(x, p)
                        # interior tau-nodes (exclude endpoints)
                        for y in inter:
                            if y == x or y == p:
                                continue
                            if y < n or y == R:
                                ok = False
                                break
                        if not ok:
                            break
                        for (a, b) in pe:
                            key = a if parent[a] == b else b   # child endpoint names the edge
                            ef.setdefault(key, []).append(flow[x])
                    if not ok:
                        continue
                    if any(y != R and y not in ef for y in range(Nt)):
                        continue                      # tau-edge without a system: dominated
                    n_pairs += 1
                    C = [None] * Nt
                    for x in reversed(orderl):
                        if x < n:
                            base = np.full(N, INF)
                            base[A[x]] = 0.0
                        else:
                            base = np.where(isT, INF, 0.0)
                        vec = base
                        for y in adj[x]:
                            if parent[y] == x and y != x:
                                W = Dmat(tuple(sorted(ef[y])))
                                vec = vec + (W + C[y][None, :]).min(axis=1)
                        C[x] = vec
                    np.minimum(best, C[R] + ncost, out=best)
    best[isT] = INF
    return best, n_pairs

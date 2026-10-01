# Frozen port of the 2026-09-23 prototype `design_b/strict_milp.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative and the
# command-line driver dropped; the MILP itself is unchanged. Do not import from
# pyorps here -- the oracles must stay independent of the engine they check.
"""D_strict (physical model: one trench per used raster step, all systems on a step derate each
other, arbitrary walks, cyclic unions allowed) as a MILP (HiGHS via scipy), no field joints.
Compared with CS-DW (tree-of-trenches model D, co-location allowed) at a fixed root g.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .b_model import INF


def strict_value(G, A, g, P, derate=True):
    n = len(A)
    N = G.N
    und = []
    for v in range(N):
        for u, c, l in G.nbrs[v]:
            if v < u:
                und.append((v, u, c, l))
    E = len(und)
    arcs = [(i, j, k) for i in range(n) for j in list(range(n)) + ["g"] if j != i
            for k in range(1, P.k_max + 1)]
    cell = lambda j: g if j == "g" else A[j]
    ms = range(1, (P.m_max or 2 * n) + 1)
    # variable layout
    idx = {}

    def var(key):
        if key not in idx:
            idx[key] = len(idx)
        return idx[key]
    for a in arcs:
        var(("z",) + a)
        for e in range(E):
            var(("x",) + a + (e, 0))
            var(("x",) + a + (e, 1))
    for e in range(E):
        var(("y", e))
        for m in ms:
            var(("w", e, m))
    for a in arcs:
        for e in range(E):
            for m in ms:
                if m >= 2:
                    var(("xi",) + a + (e, m))
    nv = len(idx)
    cost = np.zeros(nv)
    lb = np.zeros(nv)
    ub = np.ones(nv)
    integ = np.ones(nv)
    rows = []   # (coef dict, lo, hi)
    for (i, j, k) in arcs:
        if j == "g":
            cost[idx[("z", i, j, k)]] += P.bay
        else:
            cost[idx[("z", i, j, k)]] += P.panel
    for e, (v, u, c, l) in enumerate(und):
        cost[idx[("y", e)]] = (1 + P.omega) * c
        for (i, j, k) in arcs:
            r1 = P.rho(k, 1)
            for d in (0, 1):
                cost[idx[("x", i, j, k, e, d)]] = (r1 + P.sigma) * l if r1 < INF else 0.0
                if r1 == INF:
                    ub[idx[("x", i, j, k, e, d)]] = 0
            for m in ms:
                if m >= 2:
                    rm = P.rho(k, m) if derate else P.rho(k, 1)
                    key = ("xi", i, j, k, e, m)
                    if rm == INF:
                        # x + w <= 1
                        rows.append(({idx[("x", i, j, k, e, 0)]: 1, idx[("x", i, j, k, e, 1)]: 1,
                                      idx[("w", e, m)]: 1}, -np.inf, 1))
                        ub[idx[key]] = 0
                    else:
                        cost[idx[key]] = (rm - r1) * l
                        integ[idx[key]] = 0
                        rows.append(({idx[key]: 1, idx[("x", i, j, k, e, 0)]: -1,
                                      idx[("x", i, j, k, e, 1)]: -1, idx[("w", e, m)]: -1},
                                     -1, np.inf))
    # electrical design
    for i in range(n):
        rows.append(({idx[("z", i, j, k)]: 1 for (ii, j, k) in arcs if ii == i}, 1, 1))
        d = {}
        for (ii, j, k) in arcs:
            if ii == i:
                d[idx[("z", ii, j, k)]] = d.get(idx[("z", ii, j, k)], 0) + k
            if j == i:
                d[idx[("z", ii, j, k)]] = d.get(idx[("z", ii, j, k)], 0) - k
        rows.append((d, 1, 1))
        rows.append(({idx[("z", ii, j, k)]: 1 for (ii, j, k) in arcs if j == i}, -np.inf, P.max_in))
    # routing: path per arc, no transit through turbine cells other than endpoints
    Aset = set(A)
    for (i, j, k) in arcs:
        s, t = A[i], cell(j)
        for v in range(N):
            d = {}
            for e, (a, b, c, l) in enumerate(und):
                if a == v:
                    d[idx[("x", i, j, k, e, 0)]] = d.get(idx[("x", i, j, k, e, 0)], 0) + 1
                    d[idx[("x", i, j, k, e, 1)]] = d.get(idx[("x", i, j, k, e, 1)], 0) - 1
                if b == v:
                    d[idx[("x", i, j, k, e, 1)]] = d.get(idx[("x", i, j, k, e, 1)], 0) + 1
                    d[idx[("x", i, j, k, e, 0)]] = d.get(idx[("x", i, j, k, e, 0)], 0) - 1
            z = idx[("z", i, j, k)]
            if v == s:
                d[z] = d.get(z, 0) - 1
            if v == t:
                d[z] = d.get(z, 0) + 1
            rows.append((d, 0, 0))
            if v in Aset and v not in (s, t):
                # no flow through turbine cell v
                inflow = {}
                for e, (a, b, c, l) in enumerate(und):
                    if b == v:
                        inflow[idx[("x", i, j, k, e, 0)]] = 1
                    if a == v:
                        inflow[idx[("x", i, j, k, e, 1)]] = 1
                rows.append((inflow, 0, 0))
    # trench and system count per step
    for e in range(E):
        allx = {}
        for (i, j, k) in arcs:
            for dd in (0, 1):
                xx = idx[("x", i, j, k, e, dd)]
                rows.append(({idx[("y", e)]: 1, xx: -1}, 0, np.inf))
                allx[xx] = 1
        # sum x = sum_m m w_m ; sum_m w_m <= y
        d = dict(allx)
        for m in ms:
            d[idx[("w", e, m)]] = d.get(idx[("w", e, m)], 0) - m
        rows.append((d, 0, 0))
        rows.append(({idx[("w", e, m)]: 1 for m in ms}, -np.inf, 1))
    Am = lil_matrix((len(rows), nv))
    lo = np.empty(len(rows))
    hi = np.empty(len(rows))
    for r, (d, a, b) in enumerate(rows):
        for kk, vv in d.items():
            Am[r, kk] = vv
        lo[r] = a
        hi[r] = b
    res = milp(cost, constraints=LinearConstraint(Am.tocsr(), lo, hi), integrality=integ,
               bounds=Bounds(lb, ub), options=dict(time_limit=120, mip_rel_gap=1e-9))
    return (res.fun if res.status == 0 else None), res.status

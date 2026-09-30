# Frozen port of the 2026-09-23 verifier helpers `verifier/xcheck.py`
# (convention adapters, graph builder, instance generator), `verifier/vs_strict.py`
# (its instance generator) and `verifier/cond_c.py` (delta_max), plan rev. 5,
# section 3.2. The command-line drivers are dropped; the functions are unchanged.
"""Put design A's and design B's DPs into one common model.

The two designs differ in conventions only, and every cross-check needs the
same adapters, so they live here once:

* sigma is charged per ADDITIONAL system in A (``BParamsA`` gives B that
  convention) and per system in B (``AInstB`` gives A that convention);
* A's derating list is 0-based with a dummy ``derate[0]``, B's and the
  strict oracle's are 1-based;
* A's non-additive junction price is made additive-equivalent with
  ``JT = J``, ``JK = 2J``, ``dmax = 4``;
* A's root-transit field includes one panel per turbine that B does not.
"""
import itertools
import math

import numpy as np

from . import a_dp as adp
from . import a_model as amodel
from . import b_csdw as bcsdw
from . import b_model as bmodel
from .v_strict import Model

INF = float("inf")


class BParamsA(bmodel.Params):
    """Design B's Params with sigma charged per ADDITIONAL system (A's convention)."""

    def mu_flows(self, flows):
        flows = tuple(sorted(flows))
        if flows in self._mu:
            return self._mu[flows]
        m = len(flows)
        if m == 0 or (self.m_max is not None and m > self.m_max):
            val = INF
        else:
            val = self.sigma * (m - 1)
            for k in flows:
                r = self.rho(k, m)
                if r == INF:
                    val = INF
                    break
                val += r
        self._mu[flows] = val
        return val


class AInstB(amodel.Inst):
    """Design A's Inst with sigma charged per system (B's convention)."""

    def rate(self, systems):
        systems = list(systems)
        m = len(systems)
        if m == 0:
            return 0.0
        if m > self.mmax:
            return INF
        tot = 0.0
        for s in systems:
            c = self.sys_cost(s[0], m, s[1]) if isinstance(s, tuple) else self.sys_cost(s, m)
            if c == INF:
                return INF
            tot += c
        return tot + self.sigma * m


def make_graph(R, C, cell, diag):
    idx = lambda r, c: r * C + c
    edges = []
    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if diag else [])
    for r in range(R):
        for c in range(C):
            for dr, dc in steps:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < R and 0 <= c2 < C:
                    l = math.hypot(dr, dc)
                    cc = 0.5 * (cell[r][c] + cell[r2][c2]) * l
                    edges.append((idx(r, c), idx(r2, c2), round(cc, 6), l))
    return R * C, edges


def run_A_field(N, edges, turb, prm, sigma_conv):
    cls = amodel.Inst if sigma_conv == "A" else AInstB
    root = [v for v in range(N) if v not in turb][0]
    I = cls(n=N, edges=edges, turb=list(turb), root=root, sigma=prm["sigma"], JT=prm["J"], JK=2 * prm["J"],
            dmax=4, bay=prm["bay"], panel=prm["panel"], kmax=prm["kmax"],
            mmax=prm["mmax"] if prm["mmax"] else 99, types=prm["types"], derate=[1.0] + list(prm["derate"]))
    dp = adp.ShareDP(I, root_transit=True)
    dp.solve()
    full = I.full
    mv = np.full(N, INF)
    for L, arr in dp.F.items():
        S, U, D = L
        if S != full or D:
            continue
        a = np.array(arr) + prm["bay"] * len(U)
        np.minimum(mv, a, out=mv)
    mv[list(turb)] = INF
    return mv - prm["panel"] * len(turb), dp


def run_B_field(N, edges, turb, prm, sigma_conv):
    cls = BParamsA if sigma_conv == "A" else bmodel.Params
    P = cls(len(turb), prm["types"], I1=1.0, derate=tuple(prm["derate"]), lossc=1.0, sigma=prm["sigma"],
            omega=0.0, J=prm["J"], bay=prm["bay"], panel=prm["panel"], k_max=prm["kmax"],
            m_max=prm["mmax"], m_cap_T=3 * len(turb), max_in=2, k_bay=len(turb))
    G = bcsdw.Graph(N, edges)
    mv, L = bcsdw.csdw(G, list(turb), P, with_down=True)
    return mv, P, G


TYPESETS = [
    [(1.2, 0.3, 0.2), (2.5, 0.5, 0.08), (3.6, 0.8, 0.04)],
    [(1.1, 0.5, 0.3), (2.2, 0.9, 0.1), (4.5, 1.6, 0.03)],
    [(2.2, 0.4, 0.15), (3.3, 0.7, 0.05)],
    [(1.5, 1.5, 0.5), (3.5, 2.5, 0.2)],
    [(1.2, 2.0, 1.0), (2.2, 3.0, 0.5), (3.3, 4.5, 0.3)],
]


def rand_case(rng, n_turb):
    """xcheck.py's generator: grids with optional bottleneck column."""
    R, C = rng.choice([(3, 3), (2, 4), (3, 4), (2, 5)])
    diag = rng.random() < 0.6
    lo, hi = rng.choice([(1.0, 1.0), (0.3, 1.0), (1.0, 3.0), (0.2, 4.0), (0.5, 0.6)])
    cell = [[round(rng.uniform(lo, hi), 3) for _ in range(C)] for _ in range(R)]
    # optional bottleneck column/rows with high cost to provoke sharing trade-offs
    if rng.random() < 0.3:
        c0 = rng.randrange(C)
        for r in range(R):
            if rng.random() < 0.7:
                cell[r][c0] = round(cell[r][c0] * rng.choice([3.0, 6.0]), 3)
    N, edges = make_graph(R, C, cell, diag)
    turb = rng.sample(range(N), n_turb)
    f2 = rng.choice([1.0, 0.9, 0.8, 0.6, 0.5])
    f3 = f2 * rng.choice([1.0, 0.9, 0.8, 0.6])
    f4 = f3 * rng.choice([1.0, 0.9])
    cmin = min(min(r) for r in cell)
    sig = rng.choice([0.0, 0.05, 0.2, 0.6, 1.0 * cmin, 0.98 * cmin, 1.02 * cmin, 1.5 * cmin])
    prm = dict(types=rng.choice(TYPESETS), derate=[1.0, f2, f3, f4, f4, f4, f4, f4],
               sigma=round(sig, 6), J=rng.choice([0.0, 0.5, 2.0, 5.0, 1e6]), bay=rng.choice([0.0, 0.5, 2.0, 6.0]),
               panel=rng.choice([0.0, 0.1, 0.3]), kmax=rng.choice([2, 3, 3, 4]), mmax=rng.choice([None, 1, 2, 3]))
    return dict(R=R, C=C, diag=diag, cell=cell, N=N, edges=edges, turb=turb, prm=prm)


def rand_case_strict(rng, n_turb, nc0):
    """vs_strict.py's generator. nc0=True: no derating, unbounded m, sigma <= min c/l."""
    R, C = rng.choice([(3, 3), (2, 4), (2, 3), (3, 3)])
    diag = rng.random() < 0.25
    lo, hi = rng.choice([(1.0, 1.0), (0.3, 1.0), (1.0, 3.0), (0.2, 4.0)])
    cell = [[round(rng.uniform(lo, hi), 3) for _ in range(C)] for _ in range(R)]
    if rng.random() < 0.4:
        c0 = rng.randrange(C)
        for r in range(R):
            if rng.random() < 0.7:
                cell[r][c0] = round(cell[r][c0] * rng.choice([3.0, 8.0]), 3)
    N, edges = make_graph(R, C, cell, diag)
    turb = rng.sample(range(N), n_turb)
    minrate = min(c / l for (_, _, c, l) in edges)
    if nc0:
        derate = [1.0] * 8
        mmax = None
        sig = rng.choice([0.0, 0.05, 0.3, 0.9, 1.0]) * minrate
    else:
        f2 = rng.choice([1.0, 0.9, 0.8, 0.6, 0.5])
        f3 = f2 * rng.choice([1.0, 0.9, 0.8, 0.6])
        derate = [1.0, f2, f3, f3, f3, f3, f3, f3]
        mmax = rng.choice([None, 1, 2, 3])
        sig = rng.choice([0.0, 0.05, 0.3, 1.0, 1.5, 3.0]) * minrate
    prm = dict(types=rng.choice(TYPESETS), derate=derate, sigma=round(sig, 6),
               J=rng.choice([0.0, 0.5, 2.0, 5.0, 1e6]), bay=rng.choice([0.0, 0.5, 2.0, 6.0]),
               panel=rng.choice([0.0, 0.1, 0.3]), kmax=rng.choice([2, 3, 3]), mmax=mmax)
    return dict(R=R, C=C, diag=diag, cell=cell, N=N, edges=edges, turb=turb, prm=prm, minrate=minrate)


def delta_max(types, derate, n, sigma, mcap):
    """cond_c.py: worst super-additivity rate(a+b) - rate(a) - rate(b), sigma
    included (A's convention); identical turbines only."""
    M = Model(1, [], [], types, derate, sigma, 0, 0, 0, n, None)
    flows = range(1, n + 1)
    worst = 0.0
    for m1 in range(1, mcap):
        for m2 in range(1, mcap - m1 + 1):
            for a in itertools.combinations_with_replacement(flows, m1):
                for b in itertools.combinations_with_replacement(flows, m2):
                    r12 = M.rate(a + b)
                    r1, r2 = M.rate(a), M.rate(b)
                    if r1 == INF or r2 == INF:
                        continue
                    if r12 == INF:
                        return INF
                    worst = max(worst, r12 - r1 - r2)   # includes sigma (A's convention)
    return worst

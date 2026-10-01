# Frozen port of the 2026-09-23 prototype `design_b/model.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""D_share cost model and cut-signature operations (shared by the DP and the oracle).

A *signature* is sig = (U, D): two sorted tuples of positive integer flows.
U = systems carrying power OUT of the subtree below a trench edge (towards the root side),
D = systems carrying power INTO the subtree (they terminate at a turbine or a joint below).
With identical turbines the flow of a system is the number of turbines electrically
downstream of it, so every rate depends on flows only.
"""
from __future__ import annotations

import math
from functools import lru_cache

INF = float("inf")


class Params:
    def __init__(self, n, types, I1=1.0, derate=(1.0, 0.85, 0.75, 0.7, 0.65), lossc=0.3,
                 sigma=0.4, omega=0.0, J=4.0, bay=6.0, panel=0.3, k_max=2, m_max=None,
                 m_cap_T=None, max_in=2, k_bay=None):
        self.n = n                      # number of turbines
        self.types = list(types)        # (ampacity, cost per m, R per m)
        self.I1 = I1                    # current of one turbine (with gamma)
        self.derate = list(derate)      # f(m), m = 1..; last value extends
        self.lossc = lossc              # capitalised loss factor
        self.sigma = sigma              # per-system overhead per metre
        self.omega = omega              # trench capex multiplier (1+omega)
        self.J = J                      # price of one binary joint (additive pricing)
        self.bay = bay                  # one bay per system ending at the UW
        self.panel = panel              # price per in-system ring panel at a turbine
        self.k_max = k_max              # turbine switchgear through-flow limit
        self.m_max = m_max              # max systems per trench edge (None = unbounded)
        self.m_cap_T = m_cap_T if m_cap_T is not None else 3 * n   # transient cap
        self.max_in = max_in            # in-systems per turbine (ring panels - 1)
        self.k_bay = k_bay if k_bay is not None else n   # UW feeder-bay rating (turbines)
        self._rho = {}
        self._mu = {}

    def f(self, m):
        return self.derate[min(m, len(self.derate)) - 1]

    def rho(self, k, m):
        key = (k, m)
        if key in self._rho:
            return self._rho[key]
        best = INF
        I = k * self.I1
        for amp, cost, R in self.types:
            if amp * self.f(m) + 1e-12 >= I:
                best = min(best, cost + self.lossc * R * I * I)
        self._rho[key] = best
        return best

    def mu_flows(self, flows):
        """Per-metre system cost of a trench edge carrying these flows (direction-free)."""
        flows = tuple(sorted(flows))
        if flows in self._mu:
            return self._mu[flows]
        m = len(flows)
        if m == 0 or (self.m_max is not None and m > self.m_max):
            val = INF
        else:
            val = 0.0
            for k in flows:
                r = self.rho(k, m)
                if r == INF:
                    val = INF
                    break
                val += r + self.sigma
        self._mu[flows] = val
        return val

    def mu(self, sig):
        U, D = sig
        return self.mu_flows(U + D)

    def valid(self, sig):
        """Drainable: may traverse a positive-length trench edge."""
        return len(sig[0]) >= 1 and self.mu(sig) < INF

    def transient_ok(self, sig, s):
        U, D = sig
        if len(U) < 1 or len(U) > s:
            return False
        if len(U) + len(D) > self.m_cap_T:
            return False
        if any(k > self.n for k in U) or any(k > self.n for k in D):
            return False
        return True


def _rm(t, x):
    l = list(t)
    l.remove(x)
    return tuple(l)


def _add(t, *xs):
    return tuple(sorted(t + tuple(xs)))


@lru_cache(maxsize=None)
def unary_ops(sig, n, with_down=True):
    """Local operations at a field cell. Returns tuple of (sig2, n_joints).

    T   : up(k) + down(k)          -> nothing          (system turns at the cell), 0 joints
    J2  : up(a) + up(b)            -> up(a+b)          (joint, output goes up),     1 joint
    JdT : up(k) + down(k+d)        -> down(d)          (joint, one in from above,
                                                         output goes down),          1 joint
    DD  : down(a+b)                -> down(a) + down(b) (joint, two ins from above,
                                                         output goes down),          1 joint
    """
    U, D = sig
    out = []
    uset = sorted(set(U))
    dset = sorted(set(D))
    if with_down:
        for u in uset:
            if u in D:
                out.append(((_rm(U, u), _rm(D, u)), 0))
    for i, a in enumerate(uset):
        for b in uset[i:]:
            if a == b and U.count(a) < 2:
                continue
            if a + b > n:
                continue
            U2 = _rm(_rm(U, a), b)
            out.append(((_add(U2, a + b), D), 1))
    if with_down:
        for u in uset:
            for d in dset:
                if d > u:
                    out.append(((_rm(U, u), _add(_rm(D, d), d - u)), 1))
        for d in dset:
            for a in range(1, d // 2 + 1):
                out.append(((U, _add(_rm(D, d), a, d - a)), 1))
    return tuple(out)


def union(s1, s2):
    return (tuple(sorted(s1[0] + s2[0])), tuple(sorted(s1[1] + s2[1])))


def partitions_multiset(total, max_part, max_len):
    """All multisets (sorted tuples) of positive ints summing to total."""
    res = []

    def rec(rem, mx, cur):
        if rem == 0:
            res.append(tuple(sorted(cur)))
            return
        if len(cur) >= max_len:
            return
        for p in range(min(rem, mx), 0, -1):
            rec(rem - p, p, cur + [p])

    rec(total, max_part, [])
    return res


def default_params(**kw):
    types = [(1.2, 2.0, 1.0), (2.2, 3.0, 0.5), (3.3, 4.5, 0.3)]
    n = kw.pop("n", 3)
    return Params(n, types, **kw)

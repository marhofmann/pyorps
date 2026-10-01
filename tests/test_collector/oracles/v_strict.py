# Frozen port of the 2026-09-23 prototype `verifier/strict_oracle.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""Independent physical-network oracle (my own; shares no code with designs A or B).

D_strict: every electrical system (3-phase circuit) is a SIMPLE walk in G between its end
cells; a raster step used by >= 1 system carries exactly ONE trench, paid once, and all
systems on it share it (group derating by the count on that step, m_max on that step).
Cycles in the trench union are allowed.  Also reports the best design whose trench union
is a forest (every such design is a D_share design at equal cost).

Cost (sigma per ADDITIONAL system, design A's convention):
  sum_steps [ c(e) + ( sum_p rho(k_p, m) + sigma*(m-1) ) * l(e) ]
  + J*(d_in-1) per joint + bay per system ending at the root + panel per turbine in-system.
Electrical rules: identical or heterogeneous turbine powers, turbine out-power <= kmax,
<= 2 in-systems per turbine, joints >= 2 inputs, radial to the root.
root_transit: walks may pass the root cell; joint_at_root: a joint may sit on the root cell.
"""
from __future__ import annotations

import itertools
import math

INF = float("inf")


class Model:
    def __init__(self, N, edges, turb, types, derate, sigma, J, bay, panel, kmax, mmax,
                 P=None, root_transit=True, joint_at_root=True, qmax=1, jfun=None):
        self.N, self.edges, self.turb = N, edges, list(turb)
        self.n = len(turb)
        self.types, self.derate = types, derate
        self.sigma, self.J, self.bay, self.panel = sigma, J, bay, panel
        self.kmax, self.mmax = kmax, mmax
        self.P = P if P is not None else [1.0] * self.n
        self.root_transit, self.joint_at_root, self.qmax = root_transit, joint_at_root, qmax
        self.jfun = jfun if jfun is not None else (lambda d_in: J * (d_in - 1))
        self.adj = [[] for _ in range(N)]
        self.emap = {}
        for (u, w, c, l) in edges:
            self.adj[u].append(w)
            self.adj[w].append(u)
            self.emap[(min(u, w), max(u, w))] = (c, l)
        self._rho = {}
        self._rate = {}

    def f(self, m):
        return self.derate[min(m, len(self.derate)) - 1]

    def rho(self, p, m):
        key = (p, m)
        if key not in self._rho:
            best = INF
            for cap, a, b in self.types:
                if cap * self.f(m) >= p - 1e-12:
                    best = min(best, a + b * p * p)
            self._rho[key] = best
        return self._rho[key]

    def rate(self, pw):
        key = tuple(sorted(pw))
        if key in self._rate:
            return self._rate[key]
        m = len(key)
        if self.mmax is not None and m > self.mmax:
            r = INF
        else:
            r = self.sigma * (m - 1)
            for p in key:
                x = self.rho(p, m)
                if x == INF:
                    r = INF
                    break
                r += x
        self._rate[key] = r
        return r

    # ---------------------------------------------------------------- walks
    def simple_paths(self, s, t, root, maxlen=None):
        """Simple walks s->t; interior avoids turbine cells (and the root unless root_transit)."""
        if s == t:
            return [[s]]
        bad = set(self.turb)
        if not self.root_transit:
            bad.add(root)
        out = []
        maxlen = maxlen or self.N

        def rec(v, path, seen):
            if len(path) > maxlen + 1:
                return
            for w in self.adj[v]:
                if w == t:
                    out.append(path + [t])
                    continue
                if w in seen or w in bad:
                    continue
                seen.add(w)
                rec(w, path + [w], seen)
                seen.discard(w)
        rec(s, [s], {s})
        return out

    # ------------------------------------------------------ electrical designs
    def designs(self):
        n = self.n
        res = []
        for q in range(0, self.qmax + 1):
            nodes = n + q
            R = nodes
            for par in itertools.product(range(nodes + 1), repeat=nodes):
                if any(par[x] == x for x in range(nodes)):
                    continue
                ok = True
                for x in range(nodes):
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
                ch = {x: [y for y in range(nodes) if par[y] == x] for x in range(nodes + 1)}
                if any(len(ch[j]) < 2 for j in range(n, nodes)):
                    continue
                if any(len(ch[t]) > 2 for t in range(n)):
                    continue
                # canonical joint order
                pw = {}

                def power(x):
                    if x in pw:
                        return pw[x]
                    v = (self.P[x] if x < n else 0.0) + sum(power(y) for y in ch[x])
                    pw[x] = v
                    return v
                sets = {}

                def tset(x):
                    if x in sets:
                        return sets[x]
                    s = ((1 << x) if x < n else 0)
                    for y in ch[x]:
                        s |= tset(y)
                    sets[x] = s
                    return s
                for x in range(nodes):
                    power(x); tset(x)
                if any(bin(tset(t)).count("1") > self.kmax for t in range(n)):
                    continue
                if q >= 2 and any(tset(n + i) >= tset(n + i + 1) for i in range(q - 1)):
                    continue
                ncost = sum(self.jfun(len(ch[j])) for j in range(n, nodes)) \
                    + self.bay * len(ch[R]) + self.panel * sum(len(ch[t]) for t in range(n))
                res.append((q, par, dict(pw), ncost))
        return res

    # ----------------------------------------------------------------- solve
    def solve(self, root, maxlen=None, want_design=False):
        n = self.n
        best = [INF, INF]            # [strict, forest]
        bestdes = [None, None]
        pcache = {}

        def paths(a, b):
            k = (a, b)
            if k not in pcache:
                pl = self.simple_paths(a, b, root, maxlen)
                pcache[k] = pl
            return pcache[k]
        cand_j = [v for v in range(self.N) if v not in self.turb and (v != root or self.joint_at_root)]
        for (q, par, pw, ncost) in self.designs():
            if ncost >= best[1]:
                continue
            nodes = n + q
            for place in itertools.product(cand_j, repeat=q):
                cell = list(self.turb) + list(place)
                arcs = []
                ok = True
                for x in range(nodes):
                    a = cell[x]
                    b = root if par[x] == nodes else cell[par[x]]
                    if a == b and par[x] != nodes and x < n:
                        ok = False       # a turbine cannot be co-located with its parent
                        break
                    pl = paths(a, b)
                    if not pl:
                        ok = False
                        break
                    arcs.append((pw[x], pl))
                if not ok:
                    continue
                # order arcs: fewest path options first
                arcs.sort(key=lambda t: len(t[1]))
                load = {}

                def rec(i, tot, chosen):
                    if tot >= best[1]:
                        return
                    if i == len(arcs):
                        forest = self._is_forest(load)
                        if tot < best[0]:
                            best[0] = tot
                            bestdes[0] = list(chosen)
                        if forest and tot < best[1]:
                            best[1] = tot
                            bestdes[1] = list(chosen)
                        return
                    p, pl = arcs[i]
                    for path in pl:
                        delta = 0.0
                        touched = []
                        feas = True
                        for u, w in zip(path[:-1], path[1:]):
                            key = (min(u, w), max(u, w))
                            c, l = self.emap[key]
                            old = load.get(key)
                            if old is None:
                                newl = (p,)
                                r = self.rate(newl)
                                if r == INF:
                                    feas = False
                                    break
                                delta += c + r * l
                            else:
                                r0 = self.rate(old)
                                newl = old + (p,)
                                r = self.rate(newl)
                                if r == INF:
                                    feas = False
                                    break
                                delta += (r - r0) * l
                            touched.append((key, old))
                            load[key] = newl
                        if feas and tot + delta < best[1]:
                            chosen.append(path)
                            rec(i + 1, tot + delta, chosen)
                            chosen.pop()
                        for key, old in reversed(touched):
                            if old is None:
                                del load[key]
                            else:
                                load[key] = old
                rec(0, ncost, [])
        if want_design:
            return best, bestdes
        return best

    @staticmethod
    def _is_forest(load):
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        for (u, w) in load:
            ru, rw = find(u), find(w)
            if ru == rw:
                return False
            parent[ru] = rw
        return True

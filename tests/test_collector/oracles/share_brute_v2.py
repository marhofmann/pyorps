# Independent exhaustive oracle for D_share under the 2026-09-24 rules
# (optional stations with the building paid once per trench node, panels
# per conductor, parallel cables, bays per conductor). Written on
# 2026-09-24 by a separate agent from the specification in
# pyorps/collector/model.py alone, without reading the DP engine, and
# validated by it against the prototype design-A DP (366 roots with
# stations, max deviation 1.4e-14) and hand cases. Do not import the DP
# engine here.
"""Exhaustive oracle ("brute force") for the trench-sharing MV collector model.

The model is ``D_share`` as stated in the module docstring of
:mod:`pyorps.collector.model` (plan rev. 5, section 3.2, with the user
decisions K1, K1b and K1c of 2026-09-24: optional stations whose building is
paid once per station, one switch panel per conductor, parallel cables).

Independence. This module is written from that docstring alone. From
:mod:`pyorps.collector` it imports only :mod:`pyorps.collector.model`, and of
the model it uses only the pricing primitives ``rate``, ``feasible``,
``rho``, ``turbine_ok`` and ``bay_ok`` (plus the parameters they read), so it
prices exactly what the engine prices. It never touches the DP engine.

What is enumerated, for one root cell g
---------------------------------------
* A trench tree T: a finite tree rooted at the root T-node, its nodes mapped
  to cells and its edges to graph edges, not necessarily injectively. T is
  built in the K-copy expansion of the graph: one T-node per turbine cell, the
  root T-node at g, K ordinary copies of every other cell and, with
  ``root_transit=True``, K - 1 ordinary copies of g so that other trenches may
  cross g. Every copy of u is joined to every copy of w for each graph edge
  (u, w), so parallel trenches on one graph edge and several T-nodes on one
  cell (at most K) are representable. Every tree whose leaves are terminals is
  generated exactly once: paths are attached in turbine order and the copies of
  a cell are numbered by first appearance.
* An electrical arborescence H: every turbine has exactly one out-system;
  0 .. n-1 junction slots each have >= 2 inputs and one output; H is radial
  into the root. Templates are canonical (junction slots sorted by their
  downstream turbine set). Junctions sit on ordinary T-nodes (never on a
  turbine or the root T-node, but possibly on an ordinary copy of g); several
  junctions may share one T-node (one station with several busbar sections).
* Systems: every H-arc runs along its unique T-path, never through a turbine
  T-node or the root T-node, and carries one option (type, p) of
  ``model.options``.

Pricing and constraints (literal, per the model docstring)
---------------------------------------------------------
* T-edge on graph edge e: ``c(e) + model.rate(systems crossing it, both
  directions) * l(e)`` (``rate`` is inf when the step is infeasible);
* turbine: ``model.turbine_ok(out_mask, conductors)`` with conductors = the sum
  of p over its out-system and every in-system; cost
  ``turbine_panel_eur * conductors``;
* station = a T-node hosting >= 1 junction: ``station_building_eur`` once plus
  ``station_panel_eur`` per conductor of every input and of the output of each
  of its junctions;
* root: every system ending there must pass ``model.bay_ok`` and pays
  ``bay_eur * p``;
* R-once: on every T-edge, the systems crossing it in one direction have
  pairwise disjoint turbine sets; every T-edge carries at least one system.

Reductions that provably do not change the minimum
--------------------------------------------------
1. No non-terminal leaves in T. A leaf edge that carries nothing is excluded by
   the rule above. A junction on a non-terminal leaf x has all its inputs and
   its output on the leaf edge (p, x); p cannot be a turbine or the root (no
   system could feed it), so the junction moves to p: every system loses the
   step (p, x), the edge disappears, m only falls on the remaining steps (so
   every step stays feasible and ``rate`` does not rise), and a junction that
   now meets another junction on p merges with it (the building is paid at
   most once and the internal link's panels disappear).
2. No zero-length H-arcs: a junction feeding a junction on the same T-node
   merges with it at no extra cost (reason as in 1).
3. Given the p of every system, the cable count m of every step is fixed and
   the edge cost separates by system, so each system takes the cheapest type
   that is feasible on every step of its route. ``type_search="joint"``
   enumerates every type vector instead (slow; used to check this reduction).
   Every candidate that could improve the incumbent is re-priced literally with
   ``model.rate`` before it is accepted, and the two prices must agree.
4. Branch and bound with an admissible bound: for every T-edge its trench cost
   plus its length times the cheapest ``rho`` of any single system, plus one
   panel per turbine and one bay. Used only when every ``rho`` is >= 0 (the
   model already requires the other costs to be >= 0); ``prune=False``
   switches it off.

Truncation is explicit: when ``max_trees`` trees have been priced the search
stops, the third return value is True and a RuntimeWarning is issued; the cost
returned is then only an upper bound on the K-limited optimum.

A design that needs more than K T-nodes on one cell (e.g. three parallel
trenches through one cell) is outside the enumerated space, so an exact DP
with unlimited copies may be LOWER than this oracle at K = 2; re-run with a
larger K to confirm.
"""

from __future__ import annotations

import itertools
import time
import warnings

from pyorps.collector.model import INF, CollectorModel

__all__ = [
    "ORDINARY",
    "ROOT",
    "TURBINE",
    "brute_share",
    "expand",
    "h_templates",
    "reprice_design",
]

TURBINE, ROOT, ORDINARY = "turbine", "root", "ordinary"
_REL = 1e-9          # relative tolerance for "cannot improve" and self-checks


class _Stop(Exception):
    """Unwinds the tree enumeration when ``max_trees`` is reached."""


# --------------------------------------------------------------- expansion

def expand(N, edges, turbines, g, K=2, root_transit=True):
    """The K-copy expansion of the instance graph.

    Returns ``(nodes, adj, root, tnodes, prev_copy)``:
    ``nodes[x] = (cell, kind, copy)`` with kind TURBINE, ROOT or ORDINARY;
    ``adj[x]`` = list of ``(y, edge index)``, cheapest trench first;
    ``root`` = the root T-node; ``tnodes[i]`` = turbine i's T-node;
    ``prev_copy[x]`` = the ordinary copy numbered one below ``x`` on the same
    cell (None for copy 0 and for terminals).
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not 0 <= g < N:
        raise ValueError("root cell out of range")
    tidx = {}
    for i, t in enumerate(turbines):
        if not 0 <= t < N or t in tidx:
            raise ValueError("turbine cells must be distinct nodes of the graph")
        tidx[t] = i
    if g in tidx:
        raise ValueError("the root cell g must not be a turbine")
    nodes, prev_copy = [], []
    bycell = [[] for _ in range(N)]
    tnodes = [None] * len(turbines)
    root = None
    for v in range(N):
        if v in tidx:
            tnodes[tidx[v]] = len(nodes)
            bycell[v].append(len(nodes))
            nodes.append((v, TURBINE, 0))
            prev_copy.append(None)
            continue
        if v == g:
            root = len(nodes)
            bycell[v].append(root)
            nodes.append((v, ROOT, 0))
            prev_copy.append(None)
            k_ord = K - 1 if root_transit else 0
        else:
            k_ord = K
        last = None
        for c in range(k_ord):
            x = len(nodes)
            bycell[v].append(x)
            nodes.append((v, ORDINARY, c))
            prev_copy.append(last)
            last = x
    adj = [[] for _ in nodes]
    for ei, (u, w, c, l) in enumerate(edges):
        if u == w:
            raise ValueError(f"edge {ei} is a self-loop")
        if c < 0 or not l > 0:
            raise ValueError(f"edge {ei}: need c >= 0 and l > 0")
        for a in bycell[u]:
            for b in bycell[w]:
                adj[a].append((b, ei))
                adj[b].append((a, ei))
    for lst in adj:
        lst.sort(key=lambda yb: (edges[yb[1]][2], yb[1], yb[0]))
    return nodes, adj, root, tnodes, prev_copy


# ------------------------------------------------------ electrical templates

def _acyclic(par):
    m = len(par)
    for a in range(m):
        x, steps = a, 0
        while x != -1:
            x = par[x]
            steps += 1
            if steps > m:
                return False
    return True


def h_templates(model, allow_stations=None):
    """Every canonical electrical arborescence on n turbines and j junctions.

    Items ``(j, par, down, kids)``: nodes 0..n-1 are the turbines, n..n+j-1
    the junction slots; ``par[a]`` is the H-parent of a (-1 = the root);
    ``down[a]`` the turbine bitmask carried by the out-system of a;
    ``kids[a]`` the in-systems of a. Junctions have >= 2 inputs (so j <= n-1),
    slots are sorted by ``down`` (distinct junctions carry distinct sets), and
    a template is dropped only when a turbine fails ``turbine_ok`` even with
    one conductor per system.
    """
    n = model.n
    if allow_stations is None:
        allow_stations = model.allow_stations
    jmax = max(n - 1, 0) if allow_stations else 0
    out = []
    for j in range(jmax + 1):
        m = n + j
        choices = [[-1] + [b for b in range(m) if b != a] for a in range(m)]
        for par in itertools.product(*choices):
            if not _acyclic(par):
                continue
            kids = [[] for _ in range(m)]
            for a, b in enumerate(par):
                if b >= 0:
                    kids[b].append(a)
            if any(len(kids[a]) < 2 for a in range(n, m)):
                continue
            down = [0] * m
            for t in range(n):
                x = t
                while x != -1:
                    down[x] |= 1 << t
                    x = par[x]
            if any(down[a] >= down[a + 1] for a in range(n, m - 1)):
                continue
            if any(not model.turbine_ok(down[t], 1 + len(kids[t]))
                   for t in range(n)):
                continue
            out.append((j, tuple(par), tuple(down),
                        tuple(tuple(k) for k in kids)))
    return out


# ------------------------------------------------------------ trench trees

def _enumerate_trees(adj, root, tnodes, prev_copy, edge_lb, threshold,
                     max_trees, visit):
    """Call ``visit(parent, lb)`` once per trench tree whose leaves are
    terminals (up to renumbering the copies of a cell).

    ``parent[x] = (parent T-node, graph edge index)``; ``lb`` is the sum of
    ``edge_lb`` over the tree. The path of turbine i is attached to the tree
    spanned by the root and turbines 0..i-1; its interior may pass through
    turbine T-nodes not yet in the tree. Ordinary copy c of a cell may enter
    only after copy c-1. Partial trees whose ``lb`` exceeds ``threshold()``
    are cut. Returns ``(trees visited, truncated)``.
    """
    nn = len(adj)
    in_tree = [False] * nn
    in_tree[root] = True
    parent = {}
    count = [0]
    nt = len(tnodes)

    def attach(i, lb):
        if i == nt:
            if count[0] >= max_trees:
                raise _Stop
            count[0] += 1
            visit(parent, lb)
            return
        t = tnodes[i]
        if in_tree[t]:
            attach(i + 1, lb)
            return
        path, pe = [t], []
        on_path = [False] * nn
        on_path[t] = True

        def dfs(x, lbp):
            for (y, ei) in adj[x]:
                nlb = lbp + edge_lb[ei]
                if nlb > threshold():
                    continue
                if in_tree[y]:
                    last = len(path) - 1
                    for k, z in enumerate(path):
                        parent[z] = ((path[k + 1], pe[k]) if k < last
                                     else (y, ei))
                        in_tree[z] = True
                    attach(i + 1, nlb)
                    for z in path:
                        parent.pop(z)
                        in_tree[z] = False
                elif not on_path[y]:
                    pc = prev_copy[y]
                    if pc is not None and not (in_tree[pc] or on_path[pc]):
                        continue
                    path.append(y)
                    pe.append(ei)
                    on_path[y] = True
                    dfs(y, nlb)
                    path.pop()
                    pe.pop()
                    on_path[y] = False

        dfs(t, lb)

    try:
        attach(0, 0.0)
    except _Stop:
        return count[0], True
    return count[0], False


# ------------------------------------------------------------------ search

def brute_share(N, edges, turbines, g, model, K=2, root_transit=True,
                max_trees=1_000_000, return_design=False, *,
                type_search="separable", prune=True, stats=None):
    """Minimum cost of a D_share design rooted at cell ``g``, by enumeration.

    Parameters:
        N: number of graph nodes (cells) ``0..N-1``.
        edges: ``[(u, w, c, l)]``, trench cost ``c >= 0``, length ``l > 0``.
        turbines: turbine cells; turbine i is bit i of the model's masks.
        g: the root (UW) cell, not a turbine.
        model: a :class:`pyorps.collector.model.CollectorModel` with
            ``model.n == len(turbines)``.
        K: T-nodes allowed per non-terminal cell (the root cell gets K - 1
            ordinary copies besides the root T-node when ``root_transit``).
        root_transit: whether other trenches may cross g.
        max_trees: stop after this many trench trees (then the third return
            value is True, a RuntimeWarning is issued and the cost is only an
            upper bound).
        return_design: also return the argmin as a readable dict.
        type_search: ``"separable"`` (exact, fast) or ``"joint"`` (every
            type vector; checks the separable choice).
        prune: use the admissible branch-and-bound cut.
        stats: optional dict, filled with counters.

    Returns:
        ``(cost, design or None, truncated)``; cost is ``inf`` when no design
        exists in the enumerated space.
    """
    if not isinstance(model, CollectorModel):
        raise TypeError("model must be a CollectorModel")
    n = model.n
    if len(turbines) != n:
        raise ValueError("len(turbines) must equal model.n")
    if type_search not in ("separable", "joint"):
        raise ValueError("type_search must be 'separable' or 'joint'")
    t0 = time.perf_counter()
    nodes, adj, root, tnodes, prev_copy = expand(N, edges, turbines, g, K,
                                                 root_transit)
    templates = h_templates(model)
    ntypes = len(model.types)
    pmax = model.p_max
    mmax = model.m_max
    sigma = model.sigma_eur_per_m
    q = model.turbine_panel_eur
    bay = model.bay_eur
    building = model.station_building_eur
    spanel = model.station_panel_eur

    rho_floor = min(model.rho(mask, opt) for mask in range(1, 1 << n)
                    for opt in model.options)
    use_bb = prune and rho_floor >= 0.0
    if use_bb:
        edge_lb = [c + l * rho_floor for (_u, _w, c, l) in edges]
        node_lb = n * q + bay
    else:
        edge_lb = [0.0] * len(edges)
        node_lb = 0.0

    best = {"cost": INF, "arg": None}
    counters = {"trees": 0, "placements": 0, "priced": 0, "literal": 0}

    def threshold():
        b = best["cost"]
        if not use_bb or b == INF:
            return INF
        return b + _REL * max(1.0, abs(b)) - node_lb

    feas_cache = {}

    def feasible(mask, opt, m):
        key = (mask, opt, m)
        r = feas_cache.get(key)
        if r is None:
            r = model.feasible(mask, opt, m)
            feas_cache[key] = r
        return r

    def literal_price(parent, tn, on_edge, pos, par, down, kids, opts):
        trench = 0.0
        for x in tn:
            systems = [(down[a], opts[a]) for a, _d in on_edge[x]]
            r = model.rate(systems)
            if r == INF:
                return INF
            _u, _w, c, l = edges[parent[x][1]]
            trench += c + r * l
        tcost = 0.0
        for t in range(n):
            cond = opts[t][1] + sum(opts[k][1] for k in kids[t])
            if not model.turbine_ok(down[t], cond):
                return INF
            tcost += q * cond
        scost = 0.0
        hosts = {}
        for a in range(n, len(pos)):
            hosts.setdefault(pos[a], []).append(a)
        for js in hosts.values():
            cond = sum(opts[a][1] + sum(opts[k][1] for k in kids[a])
                       for a in js)
            scost += building + spanel * cond
        bcost = 0.0
        for a in range(len(pos)):
            if par[a] < 0:
                if not model.bay_ok(down[a], opts[a]):
                    return INF
                bcost += bay * opts[a][1]
        return trench + tcost + scost + bcost

    def offer(value, parent, tn, on_edge, pos, par, down, kids, opts, paths):
        if value < best["cost"]:
            best["cost"] = value
            best["arg"] = (dict(parent), list(tn), dict(on_edge), list(pos),
                           par, down, kids, list(opts), list(paths))

    def evaluate(parent, tn, pos, par, down, kids, paths):
        m_nodes = len(pos)
        on_edge = {x: [] for x in tn}
        for a, pth in enumerate(paths):
            for (x, d) in pth:
                on_edge[x].append((a, d))
        for x in tn:
            lst = on_edge[x]
            if not lst:
                return                      # an empty trench edge
            upm = dnm = 0
            for a, d in lst:
                mk = down[a]
                if d > 0:
                    if upm & mk:
                        return              # R-once, towards the root
                    upm |= mk
                else:
                    if dnm & mk:
                        return              # R-once, away from the root
                    dnm |= mk
        counters["placements"] += 1
        root_arcs = [a for a in range(m_nodes) if par[a] < 0]
        hosts = {}
        for a in range(n, m_nodes):
            hosts.setdefault(pos[a], []).append(a)
        plen = [sum(edges[parent[x][1]][3] for x, _d in pth) for pth in paths]
        base_c = sum(edges[parent[x][1]][2] for x in tn)
        for pv in itertools.product(range(1, pmax + 1), repeat=m_nodes):
            mcount = {x: sum(pv[a] for a, _d in on_edge[x]) for x in tn}
            if mmax is not None and any(v > mmax for v in mcount.values()):
                continue
            ok = True
            node = 0.0
            for t in range(n):
                cond = pv[t] + sum(pv[k] for k in kids[t])
                if not model.turbine_ok(down[t], cond):
                    ok = False
                    break
                node += q * cond
            if not ok:
                continue
            for a in root_arcs:
                if not model.bay_ok(down[a], (0, pv[a])):
                    ok = False
                    break
                node += bay * pv[a]
            if not ok:
                continue
            for js in hosts.values():
                node += building + spanel * sum(
                    pv[a] + sum(pv[k] for k in kids[a]) for a in js)
            counters["priced"] += 1
            if type_search == "joint":
                for tv in itertools.product(range(ntypes), repeat=m_nodes):
                    opts = [(tv[a], pv[a]) for a in range(m_nodes)]
                    val = literal_price(parent, tn, on_edge, pos, par, down,
                                        kids, opts)
                    counters["literal"] += 1
                    offer(val, parent, tn, on_edge, pos, par, down, kids,
                          opts, paths)
                continue
            opts = []
            arc_cost = 0.0
            for a in range(m_nodes):
                mk, p = down[a], pv[a]
                bt, br = None, INF
                for ti in range(ntypes):
                    opt = (ti, p)
                    if all(feasible(mk, opt, mcount[x]) for x, _d in paths[a]):
                        r = model.rho(mk, opt)
                        if r < br:
                            bt, br = ti, r
                if bt is None:
                    ok = False
                    break
                opts.append((bt, p))
                arc_cost += br * plen[a]
            if not ok:
                continue
            fast = (node + base_c + arc_cost
                    + sum(sigma * (mcount[x] - 1) * edges[parent[x][1]][3]
                          for x in tn))
            b = best["cost"]
            if b != INF and fast > b + _REL * max(1.0, abs(b)):
                continue
            val = literal_price(parent, tn, on_edge, pos, par, down, kids,
                                opts)
            counters["literal"] += 1
            if abs(val - fast) > _REL * max(1.0, abs(val)):
                raise AssertionError(
                    f"separable price {fast!r} != literal price {val!r}")
            offer(val, parent, tn, on_edge, pos, par, down, kids, opts, paths)

    def visit(parent, lb):
        if lb > threshold():
            return
        tn = list(parent)
        depth = {root: 0}
        for x0 in tn:
            stack, x = [], x0
            while x not in depth:
                stack.append(x)
                x = parent[x][0]
            d = depth[x]
            while stack:
                d += 1
                depth[stack.pop()] = d
        cands = [x for x in tn if nodes[x][1] == ORDINARY]
        pcache = {}

        def tpath(a, b):
            """T-edges from a to b as (child T-node, +1 up / -1 down), or
            None when the path passes through a turbine or the root."""
            key = (a, b)
            if key in pcache:
                return pcache[key]
            ups, dns = [], []
            x, y = a, b
            while depth[x] > depth[y]:
                ups.append(x)
                x = parent[x][0]
            while depth[y] > depth[x]:
                dns.append(y)
                y = parent[y][0]
            while x != y:
                ups.append(x)
                x = parent[x][0]
                dns.append(y)
                y = parent[y][0]
            interior = ups[1:] + dns[1:] + ([x] if x != a and x != b else [])
            if any(nodes[z][1] != ORDINARY for z in interior):
                res = None
            else:
                res = tuple([(z, 1) for z in ups]
                            + [(z, -1) for z in reversed(dns)])
            pcache[key] = res
            return res

        for (j, par, down, kids) in templates:
            for pl in itertools.product(cands, repeat=j):
                pos = list(tnodes) + list(pl)
                paths = []
                for a in range(len(pos)):
                    dst = root if par[a] < 0 else pos[par[a]]
                    if pos[a] == dst:
                        break               # zero-length arc (reduction 2)
                    pth = tpath(pos[a], dst)
                    if pth is None:
                        break
                    paths.append(pth)
                else:
                    evaluate(parent, tn, pos, par, down, kids, paths)

    ntrees, truncated = _enumerate_trees(adj, root, tnodes, prev_copy,
                                         edge_lb, threshold, max_trees, visit)
    counters["trees"] = ntrees
    if stats is not None:
        stats.update(counters)
        stats["templates"] = len(templates)
        stats["expanded_nodes"] = len(nodes)
        stats["seconds"] = time.perf_counter() - t0
        stats["pruning"] = use_bb
        stats["truncated"] = truncated
    if truncated:
        warnings.warn(
            f"brute_share stopped after max_trees={max_trees} trench trees; "
            f"the cost {best['cost']!r} is only an upper bound",
            RuntimeWarning, stacklevel=2)
    design = None
    if return_design and best["arg"] is not None:
        design = _readable(best["cost"], best["arg"], nodes, edges, model, g,
                           K, root_transit)
    return best["cost"], design, truncated


# ---------------------------------------------------------------- readable

def _readable(cost, arg, nodes, edges, model, g, K, root_transit):
    parent, tn, on_edge, pos, par, down, kids, opts, paths = arg
    n = model.n

    def name(a):
        if a < 0:
            return f"root@{g}"
        if a < n:
            return f"turbine{a}@{nodes[pos[a]][0]}"
        return f"junction{a - n}@{nodes[pos[a]][0]}(T{pos[a]})"

    def tnode(x):
        cell, kind, copy = nodes[x]
        return {"tnode": x, "cell": cell, "kind": kind, "copy": copy}

    trench = []
    for x in tn:
        px, ei = parent[x]
        _u, _w, c, l = edges[ei]
        systems = [{"arc": a, "dir": "up" if d > 0 else "down",
                    "mask": down[a], "option": tuple(opts[a])}
                   for a, d in on_edge[x]]
        r = model.rate([(s["mask"], s["option"]) for s in systems])
        trench.append({"child": tnode(x), "parent": tnode(px), "edge": ei,
                       "c": c, "l": l,
                       "m": sum(s["option"][1] for s in systems),
                       "systems": systems, "rate": r, "cost": c + r * l})
    arcs = []
    for a in range(len(pos)):
        cells = [nodes[pos[a]][0]]
        for x, d in paths[a]:
            cells.append(nodes[parent[x][0]][0] if d > 0 else nodes[x][0])
        arcs.append({"arc": a, "from": name(a), "to": name(par[a]),
                     "mask": down[a], "option": tuple(opts[a]),
                     "cells": cells})
    turbs = []
    for t in range(n):
        cond = opts[t][1] + sum(opts[k][1] for k in kids[t])
        turbs.append({"turbine": t, "cell": nodes[pos[t]][0],
                      "out_mask": down[t], "conductors": cond,
                      "cost": model.turbine_panel_eur * cond})
    hosts = {}
    for a in range(n, len(pos)):
        hosts.setdefault(pos[a], []).append(a)
    stations = []
    for x, js in hosts.items():
        cond = sum(opts[a][1] + sum(opts[k][1] for k in kids[a]) for a in js)
        stations.append({
            "tnode": x, "cell": nodes[x][0],
            "junctions": [{"junction": a - n,
                           "inputs": [name(k) for k in kids[a]],
                           "input_p": [opts[k][1] for k in kids[a]],
                           "output_to": name(par[a]),
                           "output_p": opts[a][1], "mask": down[a]}
                          for a in js],
            "conductors": cond,
            "cost": (model.station_building_eur
                     + model.station_panel_eur * cond)})
    feeders = [{"arc": a, "mask": down[a], "option": tuple(opts[a]),
                "cost": model.bay_eur * opts[a][1]}
               for a in range(len(pos)) if par[a] < 0]
    parts = {"trench_and_cables": sum(e["cost"] for e in trench),
             "turbines": sum(t["cost"] for t in turbs),
             "stations": sum(s["cost"] for s in stations),
             "bays": sum(f["cost"] for f in feeders)}
    return {"cost": cost, "root_cell": g, "K": K,
            "root_transit": root_transit, "trench_edges": trench,
            "arcs": arcs, "turbines": turbs, "stations": stations,
            "feeders": feeders, "parts": parts}


def reprice_design(design, model):
    """Re-price a readable design from its own records (a self-check).

    Re-checks R-once, the station and node rules and returns the total, or
    ``inf`` if a rule fails.
    """
    tot = 0.0
    for e in design["trench_edges"]:
        if not e["systems"]:
            return INF
        up = dn = 0
        for s in e["systems"]:
            if s["dir"] == "up":
                if up & s["mask"]:
                    return INF
                up |= s["mask"]
            else:
                if dn & s["mask"]:
                    return INF
                dn |= s["mask"]
        r = model.rate([(s["mask"], tuple(s["option"]))
                        for s in e["systems"]])
        tot += e["c"] + r * e["l"]
    for t in design["turbines"]:
        if not model.turbine_ok(t["out_mask"], t["conductors"]):
            return INF
        tot += model.turbine_panel_eur * t["conductors"]
    for s in design["stations"]:
        cond = sum(sum(j["input_p"]) + j["output_p"] for j in s["junctions"])
        if cond != s["conductors"] or any(len(j["inputs"]) < 2
                                          for j in s["junctions"]):
            return INF
        tot += model.station_building_eur + model.station_panel_eur * cond
    for f in design["feeders"]:
        if not model.bay_ok(f["mask"], tuple(f["option"])):
            return INF
        tot += model.bay_eur * f["option"][1]
    return tot

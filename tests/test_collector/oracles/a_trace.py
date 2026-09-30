# Frozen port of the 2026-09-23 prototype `design_a/share_trace.py` (plan rev. 5, section 3.2).
# Ported for the D4 oracle tests: imports made package-relative, nothing else
# changed unless a comment says "PORT:". Do not import from pyorps here -- the
# oracles must stay independent of the engine they check.
"""Traceback of the cut-state DW and an independent M2 re-pricer of the witness."""
from __future__ import annotations

from .a_model import INF, popcount, union
from .a_dp import tm


def traceback(dp):
    I = dp.I
    Tn, Te, En = [], [], []

    def new_node(cell):
        Tn.append(cell)
        return len(Tn) - 1

    def rooted(L, v, top=None):
        if top is None:
            top = new_node(v)
        pr = dp.pred[L][v]
        if pr is None:
            sa = dp.seedarg[(L, v)]
            if sa[0] == "turb":
                _, i, Mt, grouping = sa
                out_tok = next(iter(L[1]))
                En.append(("turb", top, out_tok, list(Mt), i))
                for grp in grouping:
                    child = (union(tm(x) for x in grp), frozenset(grp), frozenset())
                    x, ei = dp.aarg[child][v]
                    c_top = rooted(child, x)
                    Te.append((c_top, top, ei, child))
            else:
                _, b1, b2 = sa
                rooted(b1, v, top)
                rooted(b2, v, top)
        elif pr[0] == "grow":
            _, u, ei = pr
            c_top = rooted(L, u)
            Te.append((c_top, top, ei, L))
        else:
            _, Lf, cost = pr
            rooted(Lf, v, top)
            _, Uf, Df = Lf
            _, Ut, Dt = L
            if Ut != Uf:
                Mb, out, Ma = Uf - Ut, Ut - Uf, Dt - Df
                En.append(("junc", top, next(iter(out)), list(Mb) + list(Ma), None))
            else:
                C, toks = Df - Dt, Dt - Df
                En.append(("junc", top, next(iter(C)), list(toks), None))
        return top

    root_node = new_node(I.root)
    X = I.full
    while X:
        Y = dp.Harg[X]
        L = dp.rootlab[Y]
        x, ei = dp.aarg[L][I.root]
        c_top = rooted(L, x)
        Te.append((c_top, root_node, ei, L))
        X ^= Y
    return Tn, Te, En, root_node


def reprice(I, Tn, Te, En, root_node, check_labels=True):
    """Independent M2 evaluation.  Returns (cost, info) or raises AssertionError."""
    parent, pedge = {root_node: None}, {root_node: None}
    for idx, (c, p, ei, L) in enumerate(Te):
        assert c not in parent or c == root_node and False, "T-node with two parents"
        parent[c], pedge[c] = p, idx
    depth = {}

    def dep(x):
        if x in depth:
            return depth[x]
        d = 0 if parent[x] is None else dep(parent[x]) + 1
        depth[x] = d
        return d

    for x in range(len(Tn)):
        assert x in parent, "disconnected T-node"
        dep(x)
    # step validity
    for (c, p, ei, L) in Te:
        u, w, _, _ = I.edges[ei]
        assert {Tn[c], Tn[p]} == {u, w}, "T-edge not on its raster step"
    turb_nodes = {}
    for (kind, x, out, ins, i) in En:
        if kind == "turb":
            assert Tn[x] == I.turb[i]
            assert i not in turb_nodes, "turbine twice"
            turb_nodes[i] = x
    assert len(turb_nodes) == I.k, "turbine missing"
    special = set(turb_nodes.values()) | {root_node}
    for x, cell in enumerate(Tn):
        if (cell in I.turb or cell == I.root) and x not in special:
            raise AssertionError("second T-node on a turbine/root cell (transit)")
    # electrical arcs
    consumer = {}
    for ni, (kind, x, out, ins, i) in enumerate(En):
        if kind == "junc":
            assert Tn[x] not in I.turb and Tn[x] != I.root
        for tok in ins:
            assert tok not in consumer, "arc consumed twice"
            consumer[tok] = ni
    arcs = []
    for ni, (kind, x, out, ins, i) in enumerate(En):
        dst = En[consumer[out]][1] if out in consumer else root_node
        arcs.append((out, x, dst))
    # every input must be produced by some node
    produced = {a[0] for a in arcs}
    for tok in consumer:
        assert tok in produced, "input arc never produced"
    # downstream sets consistent: out mask == own turbine | inputs
    for (kind, x, out, ins, i) in En:
        own = (1 << i) if kind == "turb" else 0
        assert tm(out) == own | union(tm(t) for t in ins), "flow conservation"
    ups = {e: [] for e in range(len(Te))}
    dns = {e: [] for e in range(len(Te))}
    bays = 0
    for (tok, a, b) in arcs:
        if b == root_node:
            bays += 1
        x, y = a, b
        na, nb, up, dn = [a], [b], [], []
        while depth[x] > depth[y]:
            up.append(pedge[x]); x = parent[x]; na.append(x)
        while depth[y] > depth[x]:
            dn.append(pedge[y]); y = parent[y]; nb.append(y)
        while x != y:
            up.append(pedge[x]); x = parent[x]; na.append(x)
            dn.append(pedge[y]); y = parent[y]; nb.append(y)
        seq = na + nb[-2::-1]
        for z in seq[1:-1]:
            assert z not in special, "arc transits a turbine or the root"
        for e in up:
            ups[e].append(tok)
        for e in dn:
            dns[e].append(tok)
    cost = 0.0
    max_nodes_per_cell = {}
    for x, cell in enumerate(Tn):
        max_nodes_per_cell[cell] = max_nodes_per_cell.get(cell, 0) + 1
    steps_used = {}
    for e, (c, p, ei, L) in enumerate(Te):
        for lst in (ups[e], dns[e]):
            acc = 0
            for t in lst:
                assert not (acc & tm(t)), "R-once violated"
                acc |= tm(t)
        if check_labels:
            assert frozenset(ups[e]) == L[1] and frozenset(dns[e]) == L[2], ("label mismatch", e, L, ups[e], dns[e])
        r = I.rate(ups[e] + dns[e])
        assert r < INF, "infeasible bundle"
        _, _, cc, l = I.edges[ei]
        cost += cc + r * l
        steps_used[ei] = steps_used.get(ei, 0) + 1
    for (kind, x, out, ins, i) in En:
        if kind == "turb":
            assert len(ins) <= 2 and popcount(tm(out)) <= I.kmax, "switchgear"
            cost += I.panel * (1 + len(ins))
        else:
            cost += I.junction_cost(len(ins) + 1)
    cost += I.bay * bays
    info = dict(parallel_steps=sum(1 for v in steps_used.values() if v > 1),
                max_tnodes_per_cell=max(max_nodes_per_cell.values()),
                n_junctions=sum(1 for e in En if e[0] == "junc"),
                n_down_edges=sum(1 for e in range(len(Te)) if dns[e]))
    return cost, info

# Frozen port of the 2026-09-23 prototypes `design_a/gen.py` and
# `design_a/gen_spur.py` (plan rev. 5, section 3.2): design A's random instance
# generators. Merged into one module and made package-relative; the generators
# are unchanged. The global `random` fallback is removed: every caller passes a
# seeded `random.Random`.
"""Random D_share instances for design A's DP and enumerator."""
import math

from .a_model import Inst


def grid(rows, cols, diag_prob=0.0, rng=None):
    idx = lambda r, c: r*cols + c
    E = []
    for r in range(rows):
        for c in range(cols):
            if c+1 < cols: E.append((idx(r,c), idx(r,c+1), 1.0))
            if r+1 < rows: E.append((idx(r,c), idx(r+1,c), 1.0))
            if r+1 < rows and c+1 < cols and rng.random() < diag_prob: E.append((idx(r,c), idx(r+1,c+1), math.sqrt(2)))
            if r+1 < rows and c-1 >= 0 and rng.random() < diag_prob: E.append((idx(r,c), idx(r+1,c-1), math.sqrt(2)))
    return rows*cols, E


def randgraph(n, extra, rng):
    E = []
    for v in range(1, n):
        u = rng.randrange(v)
        E.append((u, v, rng.choice([1.0, 1.0, 1.5, 2.0])))
    have = {(min(a,b),max(a,b)) for a,b,_ in E}
    tries = 0
    while extra > 0 and tries < 200:
        tries += 1
        a, b = rng.sample(range(n), 2)
        key = (min(a,b), max(a,b))
        if key in have: continue
        have.add(key); E.append((a, b, rng.choice([1.0, 1.5, 2.0]))); extra -= 1
    return n, E


def rand_params(rng):
    f2 = rng.choice([1.0, 0.9, 0.8, 0.6])
    f3 = f2 * rng.choice([1.0, 0.9, 0.8])
    f4 = f3 * rng.choice([1.0, 0.9])
    types = rng.choice([
        [(1.2, 0.3, 0.2), (2.5, 0.5, 0.08), (3.6, 0.8, 0.04)],
        [(1.1, 0.5, 0.3), (2.2, 0.9, 0.1), (4.5, 1.6, 0.03)],
        [(2.2, 0.4, 0.15), (3.3, 0.7, 0.05)],
        [(1.5, 1.5, 0.5), (3.5, 2.5, 0.2)],
    ])
    JT = rng.choice([0.0, 0.5, 2.0, 5.0, 20.0])
    return dict(sigma=rng.choice([0.0, 0.05, 0.2, 0.6]), JT=JT, JK=JT + rng.choice([0.0, 1.0, 3.0]),
                dmax=rng.choice([4, 5]), bay=rng.choice([0.0, 0.5, 2.0]), panel=rng.choice([0.0, 0.1]),
                kmax=rng.choice([2, 3, 3]), mmax=rng.choice([1, 2, 3, 3, 4]), types=types,
                derate=[1.0, 1.0, f2, f3, f4, f4, f4, f4])


def rand_inst(rng, kind="grid", k=3):
    if kind == "grid":
        rows, cols = rng.choice([(2,3), (3,3), (2,4)])
        n, E = grid(rows, cols, diag_prob=rng.choice([0.0, 0.2]), rng=rng)
    else:
        n, E = randgraph(rng.choice([6, 7, 8]), rng.choice([1, 2, 3]), rng)
    cmin, cmax = rng.choice([(1.0, 1.0), (1.0, 3.0), (0.3, 1.0), (2.0, 4.0)])
    edges = [(u, w, round(rng.uniform(cmin, cmax) * l, 3), l) for (u, w, l) in E]
    cells = rng.sample(range(n), k + 1)
    p = rand_params(rng)
    return Inst(n=n, edges=edges, turb=cells[:k], root=cells[k], **p)


def spur_inst(rng, k=3):
    # root(0) - a(1) - b(2) [- c] trunk; turbines on spurs (optionally through a spur node)
    cells = 3
    trunk = [1, 2]
    E = [(0, 1, rng.choice([1.0, 2.0])), (1, 2, rng.choice([1.0, 1.5, 2.0]))]
    turb = []
    nst = 2
    for i in range(k):
        att = rng.choice(trunk)
        if nst < 4 and rng.random() < 0.6:
            x = cells; cells += 1; nst += 1
            E.append((att, x, rng.choice([0.5, 1.0])))
            t = cells; cells += 1
            E.append((x, t, rng.choice([0.5, 1.0])))
        else:
            t = cells; cells += 1
            E.append((att, t, rng.choice([0.5, 1.0, 1.5])))
        turb.append(t)
    # optional chord between two non-root cells
    if rng.random() < 0.5:
        a, b = rng.sample(range(1, cells), 2)
        if (a, b) not in [(u, w) for u, w, _ in E] and (b, a) not in [(u, w) for u, w, _ in E]:
            E.append((a, b, rng.choice([1.0, 2.0])))
    cmin, cmax = rng.choice([(2.0, 4.0), (1.0, 3.0), (3.0, 6.0)])
    edges = [(u, w, round(rng.uniform(cmin, cmax) * l, 3), l) for (u, w, l) in E]
    p = rand_params(rng)
    p["JT"] = rng.choice([2.0, 5.0, 20.0]); p["JK"] = p["JT"] + 1.0
    p["mmax"] = rng.choice([2, 3, 4])
    return Inst(n=cells, edges=edges, turb=turb, root=0, **p)

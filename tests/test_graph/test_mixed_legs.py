"""Mixed cable / overhead legs (plan rev. 5, D6) against a brute force.

The brute force is a shortest path over transition points with three
states -- in cable at x, a line STARTING at p (after paying ``C_tr``), a
line ENDED at q -- priced by single-source cable drains and single-source
tower fields. It allows any number of switches, and a line may only start
after a switch (ending and restarting overhead without switching is not an
operation of D). Measured 2026-09-24: equal to 1e-9 on 6 one-band windows,
mixed routes 3-5x cheaper than cable only.
"""
import heapq

import numpy as np
import pytest

from pyorps.certify.windows import drain
from pyorps.graph.mixed_legs import mixed_leg_fixpoint
from pyorps.graph.tower_field import (
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
)
from pyorps.utils.directional import primitive_directions
from pyorps.utils.neighborhood import get_neighborhood_steps

H = W = 22
CELL = 10.0


def _world(seed, bands):
    rng = np.random.default_rng(seed)
    cab = rng.integers(20, 40, size=(H, W)).astype(np.uint16)
    for r0, r1 in bands:                       # bands too dear for cable
        cab[r0:r1, :] = rng.integers(300, 400, size=(r1 - r0, W))
    ov = rng.integers(1, 10, size=(H, W)).astype(np.float64)
    tower = rng.integers(300, 600, size=(H, W)).astype(np.float64)
    if len(bands) == 2:
        # no cheap tower between the bands (70 m, beyond the 50 m span):
        # the best line comes down to cable in between and goes up again
        tower[bands[0][1]:bands[1][0], :] = 1e6
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    lattice = TowerLattice(cell_size_m=CELL, factor=1,
                           directions=primitive_directions(1))
    model = TowerFieldModel(min_span_m=10.0, max_span_m=50.0,
                            charge_terminal_towers=False)
    solver = TowerFieldSolver(values=ov, tower_cost=tower, lattice=lattice,
                              model=model)
    return rng, cab, steps, solver


def _brute(cab, steps, solver, src, tgt, T, c_tr, overhead_source=None):
    pts = [src] + T + [tgt]
    dc = {p: drain(cab, steps, [p], [0.0], engine="python").ravel() * CELL
          for p in pts}
    do = {p: solver.solve((p // W, p % W)).arrival.ravel() for p in T}
    dist = {("C", src): 0.0}
    pq = [(0.0, "C", src)]
    if overhead_source is not None:
        start = overhead_source[0] * W + overhead_source[1]
        do[start] = solver.solve(overhead_source).arrival.ravel()
        dist[("E", start)] = 0.0
        pq.append((0.0, "E", start))
    while pq:
        d, kind, p = heapq.heappop(pq)
        if d > dist.get((kind, p), np.inf):
            continue
        if kind == "C":
            nbrs = [("C", q, dc[p][q]) for q in pts]
            if p in T:
                nbrs.append(("E", p, c_tr))
        elif kind == "E":
            nbrs = [("O", q, do[p][q]) for q in T]
        else:
            nbrs = [("C", p, c_tr)]
        for k2, q, w in nbrs:
            nd = d + w
            if np.isfinite(nd) and nd < dist.get((k2, q), np.inf):
                dist[(k2, q)] = nd
                heapq.heappush(pq, (nd, k2, q))
    return dist.get(("C", tgt), np.inf), dc[src][tgt]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("bands", [[(8, 14)], [(4, 7), (14, 17)]])
def test_fixpoint_equals_the_brute_force(seed, bands):
    rng, cab, steps, solver = _world(seed, bands)
    src, tgt = 1 * W + 11, 20 * W + 10
    T = rng.choice(H * W, size=24, replace=False)
    T = np.unique(np.concatenate([T, [3 * W + 11, 8 * W + 10, 13 * W + 10,
                                      18 * W + 10]]))
    T = [int(t) for t in T if t not in (src, tgt)]
    c_tr = float(rng.choice([500.0, 2000.0]))
    res = mixed_leg_fixpoint(cab, steps, CELL, src, solver,
                             np.stack([T, T], axis=1), c_tr=c_tr,
                             drain_engine="python")
    want, cable_only = _brute(cab, steps, solver, src, tgt, T, c_tr)
    assert res.value_at(tgt) == pytest.approx(want, rel=1e-9)
    assert want < cable_only
    if len(bands) == 2:
        # two bands: at least two overhead segments, found in round >= 2
        assert res.rounds >= 3


def test_starting_overhead_at_the_source_pays_no_switch():
    rng, cab, steps, solver = _world(7, [(8, 14)])
    src, tgt = 1 * W + 11, 20 * W + 10
    T = [int(t) for t in rng.choice(H * W, size=20, replace=False)
         if t not in (src, tgt)]
    res = mixed_leg_fixpoint(cab, steps, CELL, src, solver,
                             np.stack([T, T], axis=1), c_tr=800.0,
                             overhead_source=(1, 11), drain_engine="python")
    want, _ = _brute(cab, steps, solver, src, tgt, T, 800.0,
                     overhead_source=(1, 11))
    assert res.value_at(tgt) == pytest.approx(want, rel=1e-9)


def test_bad_arguments():
    _rng, cab, steps, solver = _world(1, [])
    with pytest.raises(ValueError, match="c_tr"):
        mixed_leg_fixpoint(cab, steps, CELL, 0, solver, [[1, 1]], c_tr=-1.0)
    with pytest.raises(ValueError, match="lattice"):
        mixed_leg_fixpoint(cab, steps, CELL, 0, solver, [[H * W, 1]],
                           c_tr=1.0)

"""Every copy of the PYORPS step-cost rule agrees (plan rev. 5, D8).

The step weight is ``(v[u] + sum of intermediate cells + v[v]) *
hypot(dr, dc) / (2 + n_intermediates)``. At least nine copies exist
(``_dijkstra`` search and ``step_cost``, ``price_route_cython``,
``_traversal.construct_edges``, the tower field's ``_step_cost``, the
benchmark referee, ...), in two precisions: the Dijkstra kernels hold the
factor as float32, ``construct_edges`` and the tower field use float64.
The certificate prices every field with the SAME rule, so this checks,
for r0-r3 on a raster with exclusions:

* the kernel's ``step_cost`` and ``price_route_cython`` equal the float32
  reference bit for bit;
* ``construct_edges`` and the tower field equal the float64 reference to
  1e-12;
* the two precisions differ by at most 3e-8 relative per step (the
  documented float32 rounding of the factor, ``(3, 1)`` being the worst).
"""
import math

import numpy as np
import pytest

from pyorps.graph.tower_field import (
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
    intermediate_offsets,
)
from pyorps.utils.neighborhood import get_neighborhood_steps

dj = pytest.importorskip("pyorps.utils._dijkstra")
tr = pytest.importorskip("pyorps.utils._traversal")

ROWS, COLS = 13, 15


def _raster(seed=1):
    rng = np.random.default_rng(seed)
    r = rng.integers(125, 2000, size=(ROWS, COLS)).astype(np.uint16)
    r[4, 3:9] = 65535
    r[9, 10] = 65535
    return r


def _factor(dr, dc, n, precision):
    if precision == "float64":
        return math.hypot(dr, dc) / (2.0 + n)
    d = np.sqrt(np.float32(dr * dr + dc * dc), dtype=np.float32)
    return float(d / np.float32(2.0 + n))


def _reference(raster, steps, precision):
    """{(u, v): weight} for every usable step, the rule written out once."""
    out = {}
    for (dr, dc) in steps:
        dr, dc = int(dr), int(dc)
        inter = intermediate_offsets(dr, dc)
        fac = _factor(dr, dc, len(inter), precision)
        for r in range(ROWS):
            for c in range(COLS):
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < ROWS and 0 <= c2 < COLS):
                    continue
                cells = [(r, c), (r2, c2)] + [(r + a, c + b) for a, b in inter]
                if any(not (0 <= a < ROWS and 0 <= b < COLS) for a, b in cells):
                    continue
                if any(raster[a, b] == 65535 for a, b in cells):
                    continue
                s = sum(float(raster[a, b]) for a, b in cells)
                out[(r * COLS + c, r2 * COLS + c2)] = s * fac
    return out


@pytest.mark.parametrize("k", [0, 1, 2, 3])
class TestParity:
    def test_kernel_step_cost_is_the_float32_rule(self, k):
        raster = _raster()
        steps = np.asarray(get_neighborhood_steps(k, directed=True),
                           dtype=np.int8)
        ref = _reference(raster, steps[:, :2], "float32")
        solver = dj.make_multi_source_solver(raster, steps)
        for d, (dr, dc) in enumerate(steps[:, :2]):
            for r in range(ROWS):
                for c in range(COLS):
                    u = r * COLS + c
                    v = (r + int(dr)) * COLS + (c + int(dc))
                    w = solver.step_cost(u, d)
                    want = ref.get((u, v))
                    if want is None:
                        assert not math.isfinite(w)
                    else:
                        assert w == want

    def test_price_route_is_the_float32_rule(self, k):
        raster = _raster(2)
        steps = np.asarray(get_neighborhood_steps(k, directed=True),
                           dtype=np.int8)
        ref = _reference(raster, steps[:, :2], "float32")
        for (u, v), want in list(ref.items())[::7]:
            got = dj.price_route_cython(raster, steps,
                                        np.array([u, v], dtype=np.uint32))
            assert float(got) == want

    def test_construct_edges_is_the_float64_rule(self, k):
        raster = _raster(3)
        steps = np.asarray(get_neighborhood_steps(k, directed=True),
                           dtype=np.int8)
        ref64 = _reference(raster, steps[:, :2], "float64")
        ref32 = _reference(raster, steps[:, :2], "float32")
        frm, to, cost = tr.construct_edges(raster, steps, True)
        got = {(int(a), int(b)): float(w) for a, b, w in zip(frm, to, cost)}
        assert set(got) == set(ref64)
        for key, w in got.items():
            assert w == pytest.approx(ref64[key], rel=1e-12)
            assert abs(ref32[key] - ref64[key]) <= 3e-8 * ref64[key]

    def test_tower_field_step_is_the_float64_rule(self, k):
        raster = _raster(4).astype(np.float64)
        raster[raster == 65535] = 0.0            # tower fields block, not price
        steps = np.asarray(get_neighborhood_steps(k, directed=True),
                           dtype=np.int8)[:, :2]
        lattice = TowerLattice(cell_size_m=1.0, factor=1,
                               directions=np.asarray(steps, dtype=np.int64))
        model = TowerFieldModel(min_span_m=1.0, max_span_m=10.0,
                                charge_terminal_towers=False)
        solver = TowerFieldSolver(values=raster,
                                  tower_cost=np.full(raster.shape, 1.0),
                                  lattice=lattice, model=model)
        for dr, dc in steps:
            arr = solver._step_cost(int(dr), int(dc))
            inter = intermediate_offsets(int(dr), int(dc))
            fac = _factor(int(dr), int(dc), len(inter), "float64")
            for r in range(ROWS):
                for c in range(COLS):
                    r0, c0 = r - int(dr), c - int(dc)
                    cells = [(r0, c0), (r, c)] + [(r0 + a, c0 + b)
                                                  for a, b in inter]
                    if any(not (0 <= a < ROWS and 0 <= b < COLS)
                           for a, b in cells):
                        continue
                    want = sum(raster[a, b] for a, b in cells) * fac
                    assert arr[r, c] == pytest.approx(want, rel=1e-12)

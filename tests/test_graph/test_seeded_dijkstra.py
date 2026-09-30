"""Seeded multi-source Dijkstra (plan rev. 5, Phase D1).

``MultiSourceSolver.solve_stream`` computes
``dist[v] = min_k labels[k] + d_w(order[k], v)`` with the step weight
``weight_mult * w_raster + length_rate * len``, never entering a
no-transit cell, from a presorted seed stream that never enters the heap.
``DijkstraSolver.reset_roots`` is the heap-seeded variant for tests and
small windows. The plan's oracles:

* all-zero seeds equal ``MultiSourceSolver.solve``;
* a single seed equals ``DijkstraSolver``;
* random seeds, labels and no-transit cells match an independent
  Bellman--Ford built from the kernel's own ``step_cost``;
* ``length_rate`` adds exactly ``rate * length`` per step.

Runs against ``pyorps.utils._dijkstra``; until that extension is rebuilt
from the current ``.pyx`` it lacks ``solve_stream`` and these tests skip.
``PYORPS_D1_MODULE`` names another build of the same source to test instead.
"""
import importlib
import math
import os

import numpy as np
import pytest

from pyorps.graph.tower_field import intermediate_offsets
from pyorps.utils.neighborhood import get_neighborhood_steps

_MOD = os.environ.get("PYORPS_D1_MODULE", "pyorps.utils._dijkstra")
dj = pytest.importorskip(_MOD)
if not hasattr(dj.MultiSourceSolver, "solve_stream"):
    pytest.skip(f"{_MOD} predates solve_stream; run "
                f"`python setup.py build_ext --inplace`",
                allow_module_level=True)

from pyorps.utils._raster_context import RasterContext  # noqa: E402

INF = math.inf


def _steps(k=2):
    return np.asarray(get_neighborhood_steps(k, directed=True),
                      dtype=np.int8)


def _raster(rng, rows=14, cols=17, lo=125, hi=900, walls=0):
    r = rng.integers(lo, hi, size=(rows, cols)).astype(np.uint16)
    for _ in range(walls):
        rr = int(rng.integers(0, rows))
        c0 = int(rng.integers(0, cols - 4))
        r[rr, c0:c0 + 4] = 65535
    return r


def _bellman_ford(raster, steps, seeds, labels, *, length_rate=0.0,
                  weight_mult=1.0, no_transit=()):
    """Independent label-correcting oracle over the same step weights."""
    rows, cols = raster.shape
    solver = dj.make_multi_source_solver(raster, steps)
    nt = set(int(c) for c in no_transit)
    dist = np.full(rows * cols, INF)
    for c, lab in zip(seeds, labels):
        r, q = divmod(int(c), cols)
        if raster[r, q] == 65535:
            continue
        dist[c] = min(dist[c], float(lab))
    edges = []
    for u in range(rows * cols):
        ur, uc = divmod(u, cols)
        for d, (dr, dc) in enumerate(steps[:, :2]):
            vr, vc = ur + int(dr), uc + int(dc)
            if not (0 <= vr < rows and 0 <= vc < cols):
                continue
            v = vr * cols + vc
            if v in nt:
                continue
            if any(((ur + a) * cols + (uc + b)) in nt
                   for a, b in intermediate_offsets(int(dr), int(dc))):
                continue
            w = solver.step_cost(u, d)
            if not math.isfinite(w):
                continue
            edges.append((u, v, weight_mult * w
                          + length_rate * math.hypot(int(dr), int(dc))))
    for _ in range(rows * cols):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    # a no-transit cell is reachable only as a seed
    for c in nt:
        if c not in set(int(s) for s in seeds):
            dist[c] = INF
    return dist


def _stream(cells, labels):
    order = np.argsort(np.asarray(labels), kind="stable")
    return (np.asarray(cells, dtype=np.uint32)[order],
            np.asarray(labels, dtype=np.float64)[order])


class TestAgainstExistingSolvers:
    def test_zero_seeds_equal_multi_source_solve(self):
        rng = np.random.default_rng(1)
        raster = _raster(rng, walls=3)
        steps = _steps(2)
        cells = rng.choice(raster.size, size=6, replace=False)
        a = dj.make_multi_source_solver(raster, steps)
        a.solve(cells)
        b = dj.make_multi_source_solver(raster, steps)
        b.solve_stream(np.sort(cells), np.zeros(6))
        np.testing.assert_array_equal(a.dist_array(), b.dist_array())

    def test_single_seed_equals_dijkstra(self):
        rng = np.random.default_rng(2)
        raster = _raster(rng, walls=2)
        steps = _steps(3)
        root = int(rng.integers(0, raster.size))
        while raster.ravel()[root] == 65535:
            root = int(rng.integers(0, raster.size))
        ref = dj.make_dijkstra_solver(raster, steps)
        ref.reset_root(root)
        ref.settle_all()
        ms = dj.make_multi_source_solver(raster, steps)
        ms.solve_stream([root], [0.0])
        np.testing.assert_array_equal(ref.dist_array(), ms.dist_array())


class TestAgainstBellmanFord:
    @pytest.mark.parametrize("seed", [3, 4, 5])
    @pytest.mark.parametrize("k", [1, 2])
    def test_labelled_seeds_and_no_transit(self, seed, k):
        rng = np.random.default_rng(seed)
        raster = _raster(rng, rows=11, cols=13, walls=2)
        steps = _steps(k)
        n = raster.size
        cells = rng.choice(n, size=5, replace=False)
        labels = rng.uniform(0.0, 3000.0, size=5)
        free = np.setdiff1d(np.arange(n), cells)
        nt = rng.choice(free, size=12, replace=False)
        nt = np.concatenate([nt, cells[:1]])        # one seed is no-transit
        rate = float(rng.choice([0.0, 37.5]))
        mult = float(rng.choice([1.0, 0.5, 1.25]))
        oracle = _bellman_ford(raster, steps, cells, labels,
                               length_rate=rate, weight_mult=mult,
                               no_transit=nt)
        order, labs = _stream(cells, labels)
        ms = dj.make_multi_source_solver(raster, steps)
        ms.solve_stream(order, labs, length_rate=rate, weight_mult=mult,
                        no_transit=nt)
        got = ms.dist_array()
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(oracle))
        fin = np.isfinite(oracle)
        np.testing.assert_allclose(got[fin], oracle[fin], rtol=1e-12)

    def test_no_transit_mask_is_restored(self):
        rng = np.random.default_rng(6)
        raster = _raster(rng)
        steps = _steps(2)
        ms = dj.make_multi_source_solver(raster, steps)
        before = ms.dist_array().copy()
        ms.solve_stream([0], [0.0], no_transit=[5, 6, 7, 30])
        ms.solve_stream([0], [0.0])
        plain = dj.make_multi_source_solver(raster, steps)
        plain.solve_stream([0], [0.0])
        np.testing.assert_array_equal(ms.dist_array(), plain.dist_array())
        assert before.shape == plain.dist_array().shape

    def test_reset_roots_matches_the_stream(self):
        rng = np.random.default_rng(7)
        raster = _raster(rng, walls=2)
        steps = _steps(2)
        cells = rng.choice(raster.size, size=7, replace=False)
        labels = rng.uniform(0.0, 5000.0, size=7)
        order, labs = _stream(cells, labels)
        ms = dj.make_multi_source_solver(raster, steps)
        ms.solve_stream(order, labs, keep_prev=True, keep_owner=True)
        ds = dj.make_dijkstra_solver(raster, steps)
        ds.reset_roots(cells, labels)
        ds.settle_all()
        np.testing.assert_array_equal(ds.dist_array(), ms.dist_array())
        # a multi-root path ends at the root that owns the target
        target = int(np.flatnonzero(np.isfinite(ms.dist_array()))[-1])
        walk = ds.extract_path(target)
        assert walk[-1] == target
        start = int(walk[0])
        assert start in set(int(c) for c in cells)
        assert ms.region_array()[target] == int(
            np.flatnonzero(order == start)[0])


class TestLengthRateAndLean:
    def test_length_rate_adds_rate_times_length(self):
        """On a uniform raster the shortest walk is the same with or without
        a rate, so the labels differ by exactly rate * walk length."""
        raster = np.full((9, 9), 200, dtype=np.uint16)
        steps = _steps(1)
        base = dj.make_multi_source_solver(raster, steps)
        base.solve_stream([40], [0.0])
        rated = dj.make_multi_source_solver(raster, steps)
        rated.solve_stream([40], [0.0], length_rate=10.0)
        # walk length in cells for the 8-neighbourhood: max(|dr|,|dc|)
        # steps, of which min(|dr|,|dc|) are diagonal
        rows, cols = np.divmod(np.arange(81), 9)
        dr, dc = np.abs(rows - 4), np.abs(cols - 4)
        length = (np.minimum(dr, dc) * math.sqrt(2.0)
                  + (np.maximum(dr, dc) - np.minimum(dr, dc)))
        np.testing.assert_allclose(rated.dist_array() - base.dist_array(),
                                   10.0 * length, rtol=1e-12, atol=1e-9)

    def test_lean_solver_needs_no_predecessors(self):
        rng = np.random.default_rng(8)
        raster = _raster(rng)
        steps = _steps(2)
        lean = dj.make_multi_source_solver(raster, steps, lean=True)
        assert lean.memory_bytes() == raster.size * 9
        lean.solve_stream([3, 50], [0.0, 10.0], keep_prev=False,
                          keep_owner=False)
        full = dj.make_multi_source_solver(raster, steps)
        full.solve_stream([3, 50], [0.0, 10.0])
        np.testing.assert_array_equal(lean.dist_array(), full.dist_array())
        with pytest.raises(RuntimeError, match="lean"):
            lean.solve(np.array([3], dtype=np.uint32))

    def test_bad_streams_are_refused(self):
        raster = np.full((5, 5), 100, dtype=np.uint16)
        ms = dj.make_multi_source_solver(raster, _steps(1))
        with pytest.raises(ValueError, match="non-decreasing"):
            ms.solve_stream([1, 2], [5.0, 1.0])
        with pytest.raises(ValueError, match="finite"):
            ms.solve_stream([1], [math.inf])
        with pytest.raises(IndexError):
            ms.solve_stream([999], [0.0])
        with pytest.raises(ValueError, match="labels"):
            ms.solve_stream([1, 2], [0.0])

    def test_seed_on_an_excluded_cell_is_skipped(self):
        raster = np.full((5, 5), 100, dtype=np.uint16)
        raster[2, 2] = 65535
        ms = dj.make_multi_source_solver(raster, _steps(1))
        ms.solve_stream([12, 0], [0.0, 0.0])
        d = ms.dist_array()
        assert d[12] == INF and d[0] == 0.0


class TestGuards:
    """The D1 review (2026-09-24) found reads and writes past empty arrays
    after lean or keep-less solves and after release(); each now raises."""

    def test_path_to_root_needs_predecessors(self):
        raster = np.full((6, 6), 100, dtype=np.uint16)
        lean = dj.make_multi_source_solver(raster, _steps(1), lean=True)
        lean.solve_stream([0], [0.0])
        with pytest.raises(RuntimeError, match="predecessors"):
            lean.path_to_root(7)
        lean.solve_stream([0], [0.0], keep_prev=True)
        assert lean.path_to_root(7)[0] == 0

    def test_boundary_steps_refuses_weighted_or_keepless_solves(self):
        raster = np.full((6, 6), 100, dtype=np.uint16)
        ms = dj.make_multi_source_solver(raster, _steps(1))
        ms.solve_stream([0, 35], [0.0, 0.0])
        with pytest.raises(RuntimeError, match="predecessors and owners"):
            ms.boundary_steps()
        ms.solve_stream([0, 35], [0.0, 0.0], weight_mult=2.0, keep_prev=True,
                        keep_owner=True)
        with pytest.raises(RuntimeError, match="weight_mult"):
            ms.boundary_steps()
        ms.solve_stream([0, 35], [0.0, 0.0], keep_prev=True, keep_owner=True)
        assert ms.boundary_steps()

    def test_released_solvers_refuse_new_roots(self):
        raster = np.full((6, 6), 100, dtype=np.uint16)
        ds = dj.make_dijkstra_solver(raster, _steps(1))
        ds.release()
        with pytest.raises(RuntimeError, match="release"):
            ds.reset_roots([0], [0.0])
        with pytest.raises(RuntimeError, match="release"):
            ds.reset_root(0)

    def test_a_bad_no_transit_leaves_the_last_result_intact(self):
        raster = np.full((6, 6), 100, dtype=np.uint16)
        ms = dj.make_multi_source_solver(raster, _steps(1))
        ms.solve_stream([0], [0.0])
        before = ms.dist_array()
        with pytest.raises(IndexError):
            ms.solve_stream([3], [0.0], no_transit=[10 ** 6])
        np.testing.assert_array_equal(ms.dist_array(), before)

    def test_infinite_rates_and_wrapping_seeds_are_refused(self):
        raster = np.full((6, 6), 100, dtype=np.uint16)
        ms = dj.make_multi_source_solver(raster, _steps(1))
        with pytest.raises(ValueError, match="finite"):
            ms.solve_stream([0], [0.0], length_rate=math.inf)
        with pytest.raises(ValueError, match="finite"):
            ms.solve_stream([0], [0.0], weight_mult=math.inf)
        with pytest.raises(IndexError):
            ms.solve_stream(np.array([2 ** 32 + 3], dtype=np.int64), [0.0])
        with pytest.raises(IndexError):
            ms.solve_stream(np.array([-1], dtype=np.int64), [0.0])

    def test_lean_keeps_its_nine_bytes(self):
        raster = np.full((20, 30), 100, dtype=np.uint16)
        lean = dj.make_multi_source_solver(raster, _steps(1), lean=True)
        lean.solve_stream([5], [0.0], keep_prev=True, keep_owner=True)
        assert lean.memory_bytes() == raster.size * 17
        lean.solve_stream([5], [0.0])
        assert lean.memory_bytes() == raster.size * 9

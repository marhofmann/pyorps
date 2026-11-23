"""The Cython Dijkstra's predecessor walks are bounded.

``DijkstraSolver._reconstruct_path`` (walk to the root), ``DijkstraSolver.
_walk_to_seed`` and ``MultiSourceSolver.path_to_root`` (walk to the first cell
without predecessor) all go through ``_walk_prev`` in ``_dijkstra.pyx``. A
simple path visits each cell at most once, so a chain still running after
``len(prev)`` cells is a cycle and raises ``PyorpsRuntimeError`` instead of
looping forever. The search cannot produce such a field, and the solvers'
``prev`` arrays are not writable from Python, so the walk is tested through
the test hook ``_walk_prev_path`` on hand-made arrays; the last two tests pin
that the hook and the solvers' own walks return the same paths.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyorps.utils._dijkstra import _walk_prev_path, make_dijkstra_solver

from pyorps.core.exceptions import InvalidInputError, PyorpsError, PyorpsRuntimeError

NONE = -1


@pytest.mark.parametrize("stop", [5, NONE], ids=["to_root", "to_seed"])
def test_walk_raises_on_a_two_cycle(stop):
    # 4 -> 3 -> 2 -> 1 -> 0 -> 1 -> ...; root 5 is never reached.
    prev = [1, 0, 1, 2, 3, NONE]
    with pytest.raises(PyorpsRuntimeError, match="cycle") as info:
        _walk_prev_path(prev, 4, stop)
    assert isinstance(info.value, PyorpsError)
    assert "within 6 cells" in str(info.value)


@pytest.mark.parametrize("stop", [0, NONE], ids=["to_root", "to_seed"])
def test_walk_raises_on_a_self_loop(stop):
    prev = [NONE, 0, 2, 2]          # cell 2 is its own predecessor
    with pytest.raises(PyorpsRuntimeError, match="cycle"):
        _walk_prev_path(prev, 3, stop)


def test_cycle_through_the_root_ends_at_the_root_but_not_at_a_seed():
    # prev: 0 -> 2 -> 1 -> 0 -> ...: walking to root 1 stops there; no seed.
    prev = [2, 0, 1]
    assert _walk_prev_path(prev, 0, 1).tolist() == [1, 2, 0]
    with pytest.raises(PyorpsRuntimeError, match="cycle"):
        _walk_prev_path(prev, 0, NONE)


def test_walk_returns_paths_and_reports_no_path():
    prev = [NONE, 0, 1, NONE, 3]
    assert _walk_prev_path(prev, 2, 0).tolist() == [0, 1, 2]
    assert _walk_prev_path(prev, 2, NONE).tolist() == [0, 1, 2]
    assert _walk_prev_path(prev, 0, 0).tolist() == [0]
    assert _walk_prev_path(prev, 0, NONE).tolist() == [0]
    assert _walk_prev_path(prev, 4, NONE).tolist() == [3, 4]   # its own seed
    assert _walk_prev_path(prev, 4, 0).size == 0               # never meets root 0


@pytest.mark.parametrize("stop", [0, NONE], ids=["to_root", "to_seed"])
def test_a_simple_path_through_every_cell_is_not_a_cycle(stop):
    n = 9
    prev = [NONE] + list(range(n - 1))
    path = _walk_prev_path(prev, n - 1, stop)
    assert path.dtype == np.uint32
    assert path.tolist() == list(range(n))


@pytest.mark.parametrize("prev,start,stop", [
    ([NONE, 0], 2, NONE),           # start outside
    ([NONE, 0], 1, 2),              # stop outside
    ([NONE, 5], 1, 0),              # predecessor outside
    ([NONE, -2], 1, 0),             # negative id other than -1
    ([], 0, NONE),
])
def test_walk_hook_rejects_out_of_range_input(prev, start, stop):
    with pytest.raises(InvalidInputError):
        _walk_prev_path(prev, start, stop)


def _solver_case():
    rng = np.random.default_rng(3)
    raster = rng.choice(np.array([0, 1, 1, 2, 4], dtype=np.uint16), size=(23, 29))
    raster[rng.random(raster.shape) < 0.06] = 65535
    raster[0, 0] = raster[22, 28] = raster[11, 3] = 1
    steps = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1],
                      [1, 0], [1, 1]], dtype=np.int8)
    return raster, steps


def test_hook_matches_the_single_root_walk():
    raster, steps = _solver_case()
    solver = make_dijkstra_solver(raster, steps, max_value=65535)
    root = 0
    solver.reset_root(root)
    solver.settle_all()
    prev = solver.prev_array()
    reached = 0
    for cell in range(raster.size):
        got = _walk_prev_path(prev, cell, root)
        want = solver.extract_path(cell)
        assert np.array_equal(got, want), cell
        reached += got.size > 0
    assert reached > raster.size // 2


def test_hook_matches_the_multi_root_walk():
    raster, steps = _solver_case()
    solver = make_dijkstra_solver(raster, steps, max_value=65535)
    seeds = np.array([0, raster.size - 1, 11 * 29 + 3], dtype=np.uint32)
    solver.reset_roots(seeds)
    solver.settle_all()
    prev = solver.prev_array()
    dist = solver.dist_array()
    for cell in range(raster.size):
        want = solver.extract_path(cell)
        if not np.isfinite(dist[cell]):
            assert want.size == 0
            continue
        got = _walk_prev_path(prev, cell, NONE)
        assert np.array_equal(got, want), cell
        assert int(got[0]) in set(seeds.tolist())

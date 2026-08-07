"""Plan item 0.6 -- the per-thread relaxation buffers must never drop work.

Both parallel kernels used to commit a distance improvement by CAS and then
skip the enqueue when the thread's result buffer was full, leaving the
improvement in the distance array with nobody to re-relax it: a silently
suboptimal route. The kernels now stop claiming work before the buffer can
fill and re-issue whatever stayed unclaimed.

These tests need `python setup.py build_ext --inplace` first -- they use
bookkeeping (`get_relaxation_stats`) and a capacity hook
(`set_relaxation_buffer_capacity`) that only exist in the rebuilt extension.
"""

import numpy as np
import pytest

from pyorps.utils._delta_stepping import (
    delta_stepping_2d,
    delta_stepping_2d_persistent,
    delta_stepping_single_source_multiple_targets,
    delta_stepping_single_source_multiple_targets_persistent,
    get_relaxation_stats,
    set_relaxation_buffer_capacity,
)
from pyorps.utils._dijkstra import dijkstra_2d_cython

# The referee re-prices a path in float64 from the raster alone, independently
# of the compiled kernels (benchmarks/exactness_referee.py, plan item 0.5); its
# own arithmetic is pinned by tests/test_exactness_referee.py.
from benchmarks.exactness_referee import EdgeModel, reprice_path

STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
], dtype=np.int8)

# One claimed chunk is 64 vertices, each committing at most out_degree
# successful relaxations before the guard is tested again.
CHUNK = 64
GUARD_SLACK = CHUNK * len(STEPS_8) + 64
FLOOR_CAPACITY = 2 * GUARD_SLACK

# Bucket width well below the spike-cell edge weights, so the heavy phase
# gets real work too, and well above the plateau edge weights, so a single
# bucket holds a frontier of thousands of vertices.
DELTA = 40.0
# The guard is capacity - GUARD_SLACK = 576 successful relaxations per thread
# per light iteration, and the frontier is split across threads, so the grid has
# to be large enough that every parametrised thread count still crosses it.
# Measured rollovers at pinned capacity (single-target / multi-target):
#   grid 300 -> 153 / 55 /  0 at 1 / 2 / 4 threads
#   grid 600 -> 790 / 332 / 101
# 300 silently stopped exercising the rollover path at 4 threads.
GRID = 600
SOURCE = (GRID // 2) * GRID + GRID // 2


@pytest.fixture(autouse=True)
def _reset_capacity_hook():
    """The hook is process-global; never leak it into other tests."""
    set_relaxation_buffer_capacity(0)
    yield
    set_relaxation_buffer_capacity(0)


def _pressure_raster(n=GRID, seed=12345):
    """Low-cost plateau with texture and sparse expensive cells.

    Uniform plateaus hide lost relaxations (every route costs the same); the
    1..3 texture turns a dropped improvement into a cost delta. The 2 % spike
    cells produce edges above DELTA so the heavy phase is exercised as well.
    """
    rng = np.random.default_rng(seed)
    raster = rng.integers(1, 4, size=(n, n)).astype(np.uint16)
    spikes = rng.random((n, n)) < 0.02
    raster[spikes] = rng.integers(150, 250,
                                  size=int(spikes.sum())).astype(np.uint16)
    return raster


def _cost(path, raster):
    """Cost under the objective the solvers actually minimise.

    NOT ``_raster_context.path_cost``: that sums the raw cell values along the
    path, ignoring the geometric cost factor and the intermediate cells, so it
    ranks two genuinely tied optima differently -- a diagonal-heavy route
    crosses fewer cells for the same distance. Comparing against a Dijkstra
    reference under that metric reports a 270-vs-271 "suboptimality" for paths
    the objective prices identically at 350.4249.
    """
    assert len(path) > 0, "no path returned"
    priced = reprice_path([int(i) for i in path], raster, EdgeModel(STEPS_8))
    assert priced.ok, f"illegal path: {priced.violations}"
    return priced.cost


def _reference_cost(raster, source, target):
    path = dijkstra_2d_cython(raster, STEPS_8, np.uint32(source),
                              np.uint32(target))
    return _cost(path, raster)


class TestCapacityFloor:
    """The pinned capacity is a test hook, not a way to break the invariant."""

    def test_pinned_capacity_is_clamped_to_the_protocol_floor(self):
        raster = np.full((32, 32), 5, dtype=np.uint16)
        set_relaxation_buffer_capacity(1)

        delta_stepping_2d(raster, STEPS_8, np.uint64(0), np.uint64(32 * 32 - 1),
                          delta=50.0, num_threads=2)

        stats = get_relaxation_stats()
        assert stats["capacity"] == FLOOR_CAPACITY
        assert stats["guard"] == FLOOR_CAPACITY - GUARD_SLACK
        assert stats["guard"] > 0

    def test_small_problem_never_rolls_over_or_drops(self):
        raster = np.full((32, 32), 5, dtype=np.uint16)

        delta_stepping_2d(raster, STEPS_8, np.uint64(0), np.uint64(32 * 32 - 1),
                          delta=50.0, num_threads=2)

        stats = get_relaxation_stats()
        assert stats["rollovers"] == 0
        assert stats["drops"] == 0


class TestForcedRollover:
    """A pinned capacity puts the frontier over the guard on a test-sized raster."""

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    def test_delta_stepping_2d_stays_optimal(self, num_threads):
        raster = _pressure_raster()
        target = raster.size - 1
        expected = _reference_cost(raster, SOURCE, target)

        set_relaxation_buffer_capacity(1)
        path = delta_stepping_2d(raster, STEPS_8, np.uint64(SOURCE),
                                 np.uint64(target), delta=DELTA,
                                 num_threads=num_threads)
        stats = get_relaxation_stats()

        assert stats["drops"] == 0
        assert stats["rollovers"] > 0, "test did not reach the overflow path"
        assert _cost(path, raster) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    def test_persistent_kernel_stays_optimal(self, num_threads):
        raster = _pressure_raster()
        target = raster.size - 1
        expected = _reference_cost(raster, SOURCE, target)

        set_relaxation_buffer_capacity(1)
        path = delta_stepping_2d_persistent(
            raster, STEPS_8, np.uint64(SOURCE), np.uint64(target),
            delta=DELTA, num_threads=num_threads)
        stats = get_relaxation_stats()

        assert stats["drops"] == 0
        assert stats["rollovers"] > 0, "test did not reach the overflow path"
        assert _cost(path, raster) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("num_threads", [1, 4])
    def test_multi_target_stays_optimal(self, num_threads):
        raster = _pressure_raster()
        targets = np.array([raster.size - 1, GRID - 1, GRID * (GRID - 1)],
                           dtype=np.uint64)
        expected = [_reference_cost(raster, SOURCE, int(t)) for t in targets]

        set_relaxation_buffer_capacity(1)
        paths = delta_stepping_single_source_multiple_targets(
            raster, STEPS_8, np.uint64(SOURCE), targets, delta=DELTA,
            num_threads=num_threads)
        stats = get_relaxation_stats()

        assert stats["drops"] == 0
        assert stats["rollovers"] > 0, "test did not reach the overflow path"
        for got, want in zip(paths, expected):
            assert _cost(got, raster) == pytest.approx(want, rel=1e-6)

    @pytest.mark.parametrize("num_threads", [1, 4])
    def test_multi_target_persistent_stays_optimal(self, num_threads):
        raster = _pressure_raster()
        targets = np.array([raster.size - 1, GRID - 1, GRID * (GRID - 1)],
                           dtype=np.uint64)
        expected = [_reference_cost(raster, SOURCE, int(t)) for t in targets]

        set_relaxation_buffer_capacity(1)
        paths = delta_stepping_single_source_multiple_targets_persistent(
            raster, STEPS_8, np.uint64(SOURCE), targets, delta=DELTA,
            num_threads=num_threads)
        stats = get_relaxation_stats()

        assert stats["drops"] == 0
        assert stats["rollovers"] > 0, "test did not reach the overflow path"
        for got, want in zip(paths, expected):
            assert _cost(got, raster) == pytest.approx(want, rel=1e-6)

    def test_obstacles_do_not_change_the_answer(self):
        raster = _pressure_raster(n=200, seed=7)
        raster[60:140, 50] = 65535
        raster[60:140, 150] = 65535
        source, target = 0, raster.size - 1
        expected = _reference_cost(raster, source, target)

        set_relaxation_buffer_capacity(1)
        path = delta_stepping_2d_persistent(
            raster, STEPS_8, np.uint64(source), np.uint64(target),
            delta=DELTA, num_threads=4)

        assert get_relaxation_stats()["drops"] == 0
        assert _cost(path, raster) == pytest.approx(expected, rel=1e-6)


class TestNaturalCapacity:
    """Same geometry at the capacity the kernels size for themselves."""

    @pytest.mark.parametrize("num_threads", [1, 4])
    def test_optimal_without_the_hook(self, num_threads):
        raster = _pressure_raster()
        target = raster.size - 1
        expected = _reference_cost(raster, SOURCE, target)

        path = delta_stepping_2d_persistent(
            raster, STEPS_8, np.uint64(SOURCE), np.uint64(target),
            delta=DELTA, num_threads=num_threads)

        assert get_relaxation_stats()["drops"] == 0
        assert _cost(path, raster) == pytest.approx(expected, rel=1e-6)


def _constrained_inputs(rows=14, cols=14):
    raster = np.full((rows, cols), 10, dtype=np.uint16)
    steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],
                      [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
    n_dirs = steps.shape[0]
    return dict(
        raster=raster,
        steps=steps,
        angle_cost_lut=np.zeros((n_dirs, n_dirs), dtype=np.float32),
        angle_valid_lut=np.ones((n_dirs, n_dirs), dtype=np.uint8),
        step_distances=np.full(n_dirs, 10.0, dtype=np.float32),
        tower_terrain_costs=np.full(65536, 5.0, dtype=np.float32),
        tower_angle_costs=np.zeros((n_dirs, n_dirs), dtype=np.float32),
    )


class TestConstrainedRelaxationRounds:
    """The constrained kernel's guarded claim must not change its answer.

    Its buffers are 131 072 entries per thread and there is no capacity hook,
    so this is a parity regression for the restructured claim loop, not a
    forced-overflow test: rollover there is only reachable on frontiers far
    beyond what a unit test can build.
    """

    @pytest.mark.parametrize("threads", ["1", "4"])
    def test_parallel_matches_sequential(self, threads, monkeypatch):
        from pyorps.utils._constrained_delta import constrained_delta_stepping_2d
        from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d

        cfg = _constrained_inputs()
        common = dict(
            angle_cost_lut=cfg["angle_cost_lut"],
            angle_valid_lut=cfg["angle_valid_lut"],
            step_distances=cfg["step_distances"],
            tower_terrain_costs=cfg["tower_terrain_costs"],
            tower_angle_costs=cfg["tower_angle_costs"],
            n_span_bins=8, span_bin_size=20.0,
            min_span=40.0, max_span=160.0,
            return_dist=1,
        )

        ref = constrained_dijkstra_2d(
            cfg["raster"], 0, 0, 13, 13, cfg["steps"], **common)

        monkeypatch.setenv("OMP_NUM_THREADS", threads)
        got = constrained_delta_stepping_2d(
            cfg["raster"], 0, 0, 13, 13, cfg["steps"], **common)

        assert got[2] == pytest.approx(ref[2], rel=1e-9)
        assert got[0][0] == ref[0][0]
        assert got[0][-1] == ref[0][-1]

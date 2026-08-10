"""Plan item 1.9 -- the reusable per-raster delta-stepping workspace.

The kernels used to re-derive the exclude mask and the P1.3 span statistic and
to re-allocate the 8 B/cell dist+pred and 4 B/cell bucket-stamp arrays on every
entry; the multi-pair and multi-source drivers paid all of it once per query on
an identical raster. They now derive it once per driver call and hand it down.

What has to stay true, and is asserted here:
  * a query answered from a shared workspace returns exactly what the same
    query returns on its own -- the state arrays are reset completely between
    queries, so nothing from the previous search can leak into the next one;
  * a raster edited in place between top-level calls is never served from a
    stale derivation -- neither its costs, nor its exclusion set, nor the
    max-cost statistic the P1.3 circular-buffer guard is built on;
  * the P1.3 guard still raises on a buffer too small for the raster's span.

These tests need `python setup.py build_ext --inplace` first: the workspace and
the extra `workspace` parameter only exist in the rebuilt extension.
"""

import numpy as np
import pytest

from pyorps.utils._delta_stepping import (
    DeltaWorkspace,
    delta_stepping_2d,
    delta_stepping_2d_persistent,
    delta_stepping_multiple_sources_multiple_targets,
    delta_stepping_multiple_sources_multiple_targets_persistent,
    delta_stepping_single_source_multiple_targets,
    delta_stepping_single_source_multiple_targets_persistent,
    delta_stepping_some_pairs_shortest_paths,
    delta_stepping_some_pairs_shortest_paths_persistent,
    invalidate_system_limits_memo,
)

STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
], dtype=np.int8)

IMPASSABLE = 65535

# One thread everywhere: the parallel merge order is not pinned, so exact index
# comparisons between a driver query and the same query on its own are only
# meaningful on the deterministic single-thread path. What is under test is the
# workspace, which is thread-count independent.
THREADS = 1
DELTA = 20.0


def idx(row, col, n):
    return int(row) * n + int(col)


def textured_raster(n=48, seed=4242):
    """Cheap plateau with per-cell texture and a scatter of expensive cells.

    Uniform costs hide reset bugs (every route is tied), so the 1..4 texture
    turns any leaked label from a previous query into a different index chain.
    """
    rng = np.random.default_rng(seed)
    raster = rng.integers(1, 5, size=(n, n)).astype(np.uint16)
    spikes = rng.random((n, n)) < 0.05
    raster[spikes] = rng.integers(60, 120, size=int(spikes.sum())).astype(np.uint16)
    return raster


def island_raster(n=40):
    """Traversable cell sealed behind a two-cell-thick impassable ring.

    Two cells thick so no 8-neighbour step and no R2-style step can jump the
    wall: the target stays traversable (so the kernels do not bail out on the
    cheap source/target check) but unreachable.
    """
    raster = np.full((n, n), 5, dtype=np.uint16)
    raster[10:18, 10:18] = IMPASSABLE
    raster[12:16, 12:16] = 5
    return raster, idx(14, 14, n)


def pair_list(n):
    return [
        (idx(0, 0, n), idx(n - 1, n - 1, n)),
        (idx(n - 1, 0, n), idx(0, n - 1, n)),
        (idx(n // 2, 0, n), idx(n // 2, n - 1, n)),
        (idx(3, 3, n), idx(n - 4, n // 3, n)),
        (idx(0, n - 1, n), idx(n - 1, 0, n)),
    ]


def solo_2d(raster, source, target, persistent):
    kernel = delta_stepping_2d_persistent if persistent else delta_stepping_2d
    return kernel(raster, STEPS_8, np.uint64(source), np.uint64(target),
                  delta=DELTA, num_threads=THREADS, margin=1.00001)


def driver_pairs(raster, pairs, persistent):
    kernel = (delta_stepping_some_pairs_shortest_paths_persistent if persistent
              else delta_stepping_some_pairs_shortest_paths)
    return kernel(raster, STEPS_8,
                  np.array([p[0] for p in pairs], dtype=np.uint64),
                  np.array([p[1] for p in pairs], dtype=np.uint64),
                  delta=DELTA, num_threads=THREADS)


def solo_multi_target(raster, source, targets, persistent):
    kernel = (delta_stepping_single_source_multiple_targets_persistent
              if persistent else delta_stepping_single_source_multiple_targets)
    return kernel(raster, STEPS_8, np.uint64(source),
                  np.array(targets, dtype=np.uint64),
                  delta=DELTA, num_threads=THREADS)


def driver_multi_source(raster, sources, targets, persistent):
    kernel = (delta_stepping_multiple_sources_multiple_targets_persistent
              if persistent else
              delta_stepping_multiple_sources_multiple_targets)
    return kernel(raster, STEPS_8,
                  np.array(sources, dtype=np.uint64),
                  np.array(targets, dtype=np.uint64),
                  delta=DELTA, num_threads=THREADS)


class TestSharedWorkspaceIsExact:
    """A shared workspace may not change a single returned index."""

    @pytest.mark.parametrize("persistent", [False, True])
    def test_pair_driver_matches_isolated_calls(self, persistent):
        n = 48
        raster = textured_raster(n)
        pairs = pair_list(n)

        got = driver_pairs(raster, pairs, persistent)

        assert len(got) == len(pairs)
        for (source, target), path in zip(pairs, got):
            want = solo_2d(raster, source, target, persistent)
            assert len(want) > 0, "fixture pair is unreachable"
            assert list(path) == list(want)

    @pytest.mark.parametrize("persistent", [False, True])
    def test_multi_source_driver_matches_isolated_calls(self, persistent):
        n = 40
        raster = textured_raster(n, seed=99)
        sources = [idx(0, 0, n), idx(n - 1, n - 1, n), idx(n // 2, 2, n)]
        targets = [idx(n - 1, 0, n), idx(0, n - 1, n), idx(n // 3, n - 2, n)]

        got = driver_multi_source(raster, sources, targets, persistent)

        assert len(got) == len(sources)
        for source, source_paths in zip(sources, got):
            want = solo_multi_target(raster, source, targets, persistent)
            assert len(source_paths) == len(want)
            for path, expected in zip(source_paths, want):
                assert len(expected) > 0, "fixture target is unreachable"
                assert list(path) == list(expected)

    @pytest.mark.parametrize("persistent", [False, True])
    def test_repeated_calls_on_one_raster_are_identical(self, persistent):
        n = 40
        raster = textured_raster(n, seed=7)
        source, target = idx(1, 1, n), idx(n - 2, n - 2, n)

        first = solo_2d(raster, source, target, persistent)
        for _ in range(3):
            assert list(solo_2d(raster, source, target, persistent)) == list(first)


class TestStateIsFullyResetBetweenQueries:
    """A partial reset would let one query answer the next one."""

    @pytest.mark.parametrize("persistent", [False, True])
    def test_unreachable_pair_after_reachable_pair(self, persistent):
        n = 40
        raster, island = island_raster(n)
        outside_source = idx(1, 1, n)
        outside_target = idx(n - 2, n - 2, n)
        pairs = [(outside_source, outside_target), (outside_source, island)]

        got = driver_pairs(raster, pairs, persistent)

        assert len(got[0]) > 0
        assert len(got[1]) == 0

    @pytest.mark.parametrize("persistent", [False, True])
    def test_reachable_pair_after_unreachable_pair(self, persistent):
        n = 40
        raster, island = island_raster(n)
        outside_source = idx(1, 1, n)
        outside_target = idx(n - 2, n - 2, n)
        pairs = [(outside_source, island), (outside_source, outside_target)]

        got = driver_pairs(raster, pairs, persistent)

        assert len(got[0]) == 0
        assert list(got[1]) == list(
            solo_2d(raster, outside_source, outside_target, persistent))

    @pytest.mark.parametrize("persistent", [False, True])
    def test_unreachable_target_in_multi_source_driver(self, persistent):
        """The sealed target stays empty for every source in the loop."""
        n = 40
        raster, island = island_raster(n)
        sources = [idx(1, 1, n), idx(n - 2, 1, n)]
        targets = [idx(n - 2, n - 2, n), island]

        got = driver_multi_source(raster, sources, targets, persistent)

        for source, source_paths in zip(sources, got):
            want = solo_multi_target(raster, source, targets, persistent)
            assert len(source_paths[0]) > 0
            assert len(source_paths[1]) == 0
            for path, expected in zip(source_paths, want):
                assert list(path) == list(expected)


class TestMutatedRasterIsNeverServedStale:
    """The trap: a raster edited in place between calls, same array object."""

    @pytest.mark.parametrize("persistent", [False, True])
    def test_cost_edit_changes_the_route(self, persistent):
        n = 40
        raster = np.full((n, n), 100, dtype=np.uint16)
        raster[:n - 4, n // 2] = IMPASSABLE  # wall with a gap at the bottom
        source, target = idx(2, 2, n), idx(2, n - 3, n)

        before = solo_2d(raster, source, target, persistent)
        assert len(before) > 0

        raster[2, n // 2] = 1  # in place: open a cheap door next to the source

        after = solo_2d(raster, source, target, persistent)
        assert len(after) > 0
        assert list(after) != list(before), "stale exclude mask / costs"
        assert list(after) == list(
            solo_2d(raster.copy(), source, target, persistent))

    @pytest.mark.parametrize("persistent", [False, True])
    def test_exclusion_edit_blocks_the_route(self, persistent):
        n = 32
        raster = np.full((n, n), 10, dtype=np.uint16)
        source, target = idx(1, 1, n), idx(n - 2, n - 2, n)

        assert len(solo_2d(raster, source, target, persistent)) > 0

        raster[n - 4:, :] = IMPASSABLE  # in place: seal the target off
        raster[n - 2, n - 2] = 10  # ... but keep the target itself traversable

        assert len(solo_2d(raster, source, target, persistent)) == 0

    @pytest.mark.parametrize("persistent", [False, True])
    def test_cost_edit_is_caught_by_the_p13_guard(self, persistent):
        """The sharpest stale-cache case: the guard's own input is edited.

        A stale max-cost statistic would let a raster whose edge/delta span
        overflows the circular buffer straight into the search, which silently
        corrupts it. The guard must fire on the very next call.
        """
        n = 24
        raster = np.full((n, n), 5, dtype=np.uint16)
        source, target = 0, n * n - 1

        kernel = delta_stepping_2d_persistent if persistent else delta_stepping_2d
        assert len(kernel(raster, STEPS_8, np.uint64(source), np.uint64(target),
                          delta=5.0, num_threads=THREADS,
                          max_buckets_in_memory=16)) > 0

        raster[n // 2, n // 2] = 400  # in place: 400*sqrt(2)/5 = 113 buckets

        with pytest.raises(ValueError, match="exceeds circular buffer size"):
            kernel(raster, STEPS_8, np.uint64(source), np.uint64(target),
                   delta=5.0, num_threads=THREADS, max_buckets_in_memory=16)

    @pytest.mark.parametrize("persistent", [False, True])
    def test_cost_edit_is_caught_by_the_p13_guard_in_a_driver(self, persistent):
        n = 24
        raster = np.full((n, n), 5, dtype=np.uint16)
        sources = np.array([0], dtype=np.uint64)
        targets = np.array([n * n - 1], dtype=np.uint64)
        kernel = (delta_stepping_some_pairs_shortest_paths_persistent
                  if persistent else delta_stepping_some_pairs_shortest_paths)

        first = kernel(raster, STEPS_8, sources, targets, delta=5.0,
                       num_threads=THREADS, max_buckets_in_memory=16)
        assert len(first[0]) > 0

        raster[n // 2, n // 2] = 400

        with pytest.raises(ValueError, match="exceeds circular buffer size"):
            kernel(raster, STEPS_8, sources, targets, delta=5.0,
                   num_threads=THREADS, max_buckets_in_memory=16)


class TestSpanGuard:
    """P1.3 must still refuse a circular buffer too small for the raster."""

    def _call(self, name, raster, n):
        source, target = np.uint64(0), np.uint64(n * n - 1)
        targets = np.array([n * n - 1], dtype=np.uint64)
        if name == "2d":
            return delta_stepping_2d(
                raster, STEPS_8, source, target, delta=5.0,
                num_threads=THREADS, max_buckets_in_memory=16)
        if name == "2d_persistent":
            return delta_stepping_2d_persistent(
                raster, STEPS_8, source, target, delta=5.0,
                num_threads=THREADS, max_buckets_in_memory=16)
        if name == "multi_target":
            return delta_stepping_single_source_multiple_targets(
                raster, STEPS_8, source, targets, delta=5.0,
                num_threads=THREADS, max_buckets_in_memory=16)
        return delta_stepping_single_source_multiple_targets_persistent(
            raster, STEPS_8, source, targets, delta=5.0,
            num_threads=THREADS, max_buckets_in_memory=16)

    KERNELS = ["2d", "2d_persistent", "multi_target", "multi_target_persistent"]

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_undersized_buffer_raises(self, kernel):
        n = 24
        raster = np.full((n, n), 5, dtype=np.uint16)
        raster[n // 2, n // 2] = 400

        with pytest.raises(ValueError, match="exceeds circular buffer size"):
            self._call(kernel, raster, n)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_adequate_buffer_does_not_raise(self, kernel):
        n = 24
        raster = np.full((n, n), 5, dtype=np.uint16)

        result = self._call(kernel, raster, n)

        if kernel.startswith("multi_target"):
            assert all(len(path) > 0 for path in result)
        else:
            assert len(result) > 0

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_fully_impassable_raster_reports_no_path(self, kernel):
        """No traversable cell: no statistic to test the guard against."""
        n = 12
        raster = np.full((n, n), IMPASSABLE, dtype=np.uint16)

        result = self._call(kernel, raster, n)

        if kernel.startswith("multi_target"):
            assert all(len(path) == 0 for path in result)
        else:
            assert len(result) == 0


class TestWorkspaceBinding:
    """A workspace that does not belong to the raster must be ignored."""

    @pytest.mark.parametrize("persistent", [False, True])
    def test_foreign_workspace_is_rejected(self, persistent):
        n = 32
        raster = textured_raster(n, seed=5)
        other = textured_raster(n, seed=6)
        source, target = idx(0, 0, n), idx(n - 1, n - 1, n)
        kernel = delta_stepping_2d_persistent if persistent else delta_stepping_2d

        expected = solo_2d(raster, source, target, persistent)
        got = kernel(raster, STEPS_8, np.uint64(source), np.uint64(target),
                     delta=DELTA, num_threads=THREADS, margin=1.00001,
                     workspace=DeltaWorkspace(other, 65535))

        assert list(got) == list(expected)

    @pytest.mark.parametrize("persistent", [False, True])
    def test_workspace_with_a_different_max_value_is_rejected(self, persistent):
        """max_value is part of the key: it decides the exclusion set."""
        n = 28
        raster = np.full((n, n), 10, dtype=np.uint16)
        raster[n // 2, :n - 3] = IMPASSABLE
        source, target = idx(0, 0, n), idx(n - 1, n - 1, n)
        kernel = delta_stepping_2d_persistent if persistent else delta_stepping_2d

        expected = solo_2d(raster, source, target, persistent)
        # Built with exclusion disabled -- every cell traversable, so reusing it
        # would let the route walk straight through the wall.
        got = kernel(raster, STEPS_8, np.uint64(source), np.uint64(target),
                     delta=DELTA, num_threads=THREADS, margin=1.00001,
                     workspace=DeltaWorkspace(raster, 65536))

        assert list(got) == list(expected)

    def test_workspace_reports_the_derivations_it_caches(self):
        raster = np.array([[3, 7, IMPASSABLE], [1, IMPASSABLE, 4]],
                          dtype=np.uint16)

        workspace = DeltaWorkspace(raster, 65535)

        assert workspace.has_traversable
        assert workspace.max_traversable_cost == 7.0

    def test_workspace_of_a_fully_impassable_raster(self):
        raster = np.full((4, 4), IMPASSABLE, dtype=np.uint16)

        workspace = DeltaWorkspace(raster, 65535)

        assert not workspace.has_traversable


class TestSystemLimitsMemo:
    """The memoised SystemLimits may not change any answer."""

    def test_results_are_identical_across_a_memo_refresh(self):
        n = 32
        raster = textured_raster(n, seed=11)
        source, target = idx(0, 0, n), idx(n - 1, n - 1, n)

        first = solo_2d(raster, source, target, True)
        invalidate_system_limits_memo()
        second = solo_2d(raster, source, target, True)

        assert list(second) == list(first)

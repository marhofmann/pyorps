"""Phase 1a items 1.7 / 1.8: cython context reuse and the exact margin.

These are pure performance items, so every test here is an EXACTNESS test:

* 1.7a  CythonAPI caches one RasterContext + DijkstraSolver per raster.
        A solver that leaked dist/prev/visited state between queries would
        return a different route on the second query than on the first —
        so every reused-instance result is compared against a result from a
        freshly constructed API.
* 1.7b  ``single_source_multi_target`` swaps the per-settled-node scan over
        the target list for an O(1) membership lookup. Multi-target results
        must equal the single-pair results for the same targets, duplicates
        and unreachable targets included.
* 1.8   The delta-stepping termination margin drops from 1.1 to the exact
        clamp floor 1.00001. Margin > 1 makes the search run LONGER, so the
        routes must be bit-identical.

Requires a rebuild of the cython extensions (``python setup.py build_ext
--inplace``) — ``_dijkstra.pyx`` changed.
"""
import unittest

import numpy as np

from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.graph.api.cython_api import EXACT_TERMINATION_MARGIN, CythonAPI
from pyorps.utils._raster_context import py_path_cost_uint32
from pyorps.utils.neighborhood import get_neighborhood_steps

ROWS, COLS = 60, 60


def rough_raster(seed=20260807):
    """Costly, obstacle-riddled terrain: many near-equal competing routes."""
    rng = np.random.default_rng(seed)
    raster = rng.integers(1, 250, size=(ROWS, COLS)).astype(np.uint16)
    # Three walls with off-centre gaps — forces long detours and makes the
    # settled order (and therefore any stale-state bug) visible in the route.
    raster[15, :50] = IMPASSABLE_CELL_COST
    raster[30, 10:] = IMPASSABLE_CELL_COST
    raster[45, :40] = IMPASSABLE_CELL_COST
    return raster


def smooth_raster():
    """Near-uniform terrain: ties everywhere, tie-breaking must not drift."""
    row = np.arange(COLS, dtype=np.uint16) % 3 + 10
    return np.tile(row, (ROWS, 1)).astype(np.uint16)


def walled_off_cell(raster):
    """Index of a cell fully enclosed by obstacles (unreachable target)."""
    raster[50:53, 50:53] = IMPASSABLE_CELL_COST
    raster[51, 51] = 7
    return 51 * COLS + 51


def as_indices(path):
    return [int(i) for i in path]


def cell_cost(path, raster):
    """Sum of cell costs along a path (inf for an empty/no-path result)."""
    arr = np.asarray(path, dtype=np.uint32)
    if arr.size == 0:
        return float("inf")
    return float(py_path_cost_uint32(arr, raster, raster.shape[1]))


def api(raster, steps, **kwargs):
    return CythonAPI(raster, steps, ignore_max=True, **kwargs)


PAIRS = [
    (0, ROWS * COLS - 1),
    (5 * COLS + 3, 55 * COLS + 50),
    (59 * COLS + 0, 0 * COLS + 59),
    (20 * COLS + 40, 40 * COLS + 5),
    (10 * COLS + 10, 12 * COLS + 12),
]


class TestSolverReuseSinglePair(unittest.TestCase):
    """1.7a: repeated queries on one API == queries on fresh APIs."""

    def setUp(self):
        self.raster = rough_raster()
        self.steps = get_neighborhood_steps("r2")

    def test_repeated_single_pair_matches_fresh_instances(self):
        reused = api(self.raster, self.steps)
        for source, target in PAIRS:
            with self.subTest(source=source, target=target):
                got = reused.shortest_path(source, target,
                                           algorithm="dijkstra")
                expected = api(self.raster, self.steps).shortest_path(
                    source, target, algorithm="dijkstra")
                self.assertEqual(as_indices(got), as_indices(expected))
                self.assertEqual(cell_cost(got, self.raster),
                                 cell_cost(expected, self.raster))

    def test_same_query_twice_is_stable(self):
        """The classic stale-solver symptom: query 2 differs from query 1."""
        reused = api(self.raster, self.steps)
        source, target = PAIRS[0]
        first = as_indices(reused.shortest_path(source, target,
                                                algorithm="dijkstra"))
        for _ in range(4):
            self.assertEqual(
                as_indices(reused.shortest_path(source, target,
                                                algorithm="dijkstra")),
                first)

    def test_interleaved_query_shapes_match_fresh_instances(self):
        """Mixing query kinds is what actually pollutes a shared solver."""
        reused = api(self.raster, self.steps)
        source, target = PAIRS[1]
        targets = [t for _, t in PAIRS]

        reused.shortest_path(source, targets, algorithm="dijkstra")
        reused.shortest_path([s for s, _ in PAIRS], [t for _, t in PAIRS],
                             algorithm="dijkstra", pairwise=True)
        reused.shortest_path([s for s, _ in PAIRS[:2]],
                             [t for _, t in PAIRS[:2]],
                             algorithm="dijkstra", pairwise=False)
        got = reused.shortest_path(source, target, algorithm="dijkstra")

        expected = api(self.raster, self.steps).shortest_path(
            source, target, algorithm="dijkstra")
        self.assertEqual(as_indices(got), as_indices(expected))

    def test_source_equals_target_between_real_queries(self):
        reused = api(self.raster, self.steps)
        source, target = PAIRS[0]
        first = as_indices(reused.shortest_path(source, target,
                                                algorithm="dijkstra"))
        self.assertEqual(
            as_indices(reused.shortest_path(source, source,
                                            algorithm="dijkstra")),
            [source])
        self.assertEqual(
            as_indices(reused.shortest_path(source, target,
                                            algorithm="dijkstra")),
            first)


class TestSolverReuseMultiQuery(unittest.TestCase):
    """1.7a for the multi-target / pairwise / all-pairs entry points."""

    def setUp(self):
        self.raster = rough_raster()
        self.steps = get_neighborhood_steps("r2")
        self.sources = [s for s, _ in PAIRS]
        self.targets = [t for _, t in PAIRS]

    def _compare(self, call):
        reused = api(self.raster, self.steps)
        first = call(reused)
        second = call(reused)
        fresh = call(api(self.raster, self.steps))
        self.assertEqual([as_indices(p) for p in first],
                         [as_indices(p) for p in second])
        self.assertEqual([as_indices(p) for p in first],
                         [as_indices(p) for p in fresh])

    def test_single_to_multi(self):
        self._compare(lambda a: a.shortest_path(
            self.sources[0], self.targets, algorithm="dijkstra"))

    def test_pairwise(self):
        self._compare(lambda a: a.shortest_path(
            self.sources, self.targets, algorithm="dijkstra", pairwise=True))

    def test_multi_to_multi(self):
        self._compare(lambda a: a.shortest_path(
            self.sources[:3], self.targets[:3], algorithm="dijkstra",
            pairwise=False))


class TestSolverCacheLifecycle(unittest.TestCase):
    """The cache must actually cache, and must notice replaced inputs."""

    def setUp(self):
        self.raster = rough_raster()
        self.steps = get_neighborhood_steps("r1")
        self.api = api(self.raster, self.steps)

    def test_solver_instance_is_reused(self):
        first = self.api._dijkstra_solver()
        self.api.shortest_path(*PAIRS[0], algorithm="dijkstra")
        self.assertIs(self.api._dijkstra_solver(), first)

    def test_invalidate_rebuilds(self):
        first = self.api._dijkstra_solver()
        self.api.invalidate_solver_cache()
        self.assertIsNot(self.api._dijkstra_solver(), first)

    def test_replacing_raster_rebuilds(self):
        first = self.api._dijkstra_solver()
        self.api.raster_data = rough_raster(seed=7)
        self.assertIsNot(self.api._dijkstra_solver(), first)

    def test_rebuilt_solver_matches_new_raster(self):
        """A replaced raster must be routed on, not the cached one."""
        source, target = PAIRS[0]
        self.api.shortest_path(source, target, algorithm="dijkstra")
        other = rough_raster(seed=7)
        self.api.raster_data = other
        got = self.api.shortest_path(source, target, algorithm="dijkstra")
        expected = api(other, self.steps).shortest_path(
            source, target, algorithm="dijkstra")
        self.assertEqual(as_indices(got), as_indices(expected))


class TestMultiTargetMembership(unittest.TestCase):
    """1.7b: the O(1) target lookup must not change any returned path."""

    def setUp(self):
        self.raster = rough_raster()
        self.steps = get_neighborhood_steps("r2")
        self.api = api(self.raster, self.steps)

    def test_matches_single_pair_per_target(self):
        source = PAIRS[0][0]
        targets = [t for _, t in PAIRS]
        paths = self.api.shortest_path(source, targets, algorithm="dijkstra")
        for target, path in zip(targets, paths):
            with self.subTest(target=target):
                single = api(self.raster, self.steps).shortest_path(
                    source, target, algorithm="dijkstra")
                self.assertEqual(as_indices(path), as_indices(single))
                self.assertEqual(cell_cost(path, self.raster),
                                 cell_cost(single, self.raster))

    def test_many_targets_match_single_pair(self):
        """Enough targets that the old linear scan dominated the settle loop."""
        source = 0
        targets = [r * COLS + c
                   for r in range(20, 59, 4)
                   for c in range(5, 59, 6)]
        self.assertGreater(len(targets), 60)
        paths = self.api.shortest_path(source, targets, algorithm="dijkstra")
        self.assertEqual(len(paths), len(targets))
        for target, path in zip(targets, paths):
            with self.subTest(target=target):
                single = api(self.raster, self.steps).shortest_path(
                    source, target, algorithm="dijkstra")
                self.assertEqual(as_indices(path), as_indices(single))

    def test_duplicate_targets_are_all_served(self):
        source = PAIRS[0][0]
        a, b = PAIRS[1][1], PAIRS[2][1]
        paths = self.api.shortest_path(source, [a, b, a, a],
                                       algorithm="dijkstra")
        self.assertEqual(len(paths), 4)
        self.assertEqual(as_indices(paths[0]), as_indices(paths[2]))
        self.assertEqual(as_indices(paths[0]), as_indices(paths[3]))
        self.assertEqual(as_indices(paths[0][-1:]), [a])
        self.assertEqual(as_indices(paths[1][-1:]), [b])

    def test_source_as_its_own_target(self):
        """The source settles first and marks itself found — unchanged.

        Reconstruction still returns an empty array for it (prev[source]
        is -1); the point of the test is that the OTHER target is served
        exactly as a fresh instance serves it.
        """
        source, target = PAIRS[0]
        paths = self.api.shortest_path(source, [source, target],
                                       algorithm="dijkstra")
        self.assertEqual(len(paths[0]), 0)
        fresh = api(self.raster, self.steps).shortest_path(
            source, target, algorithm="dijkstra")
        self.assertEqual(as_indices(paths[1]), as_indices(fresh))

    def test_unreachable_target_stays_empty_and_leaves_no_residue(self):
        raster = rough_raster()
        walled = walled_off_cell(raster)
        one = api(raster, self.steps)
        reachable = PAIRS[0][1]
        paths = one.shortest_path(0, [reachable, walled],
                                  algorithm="dijkstra")
        self.assertEqual(len(paths[1]), 0)
        # The membership map must be clean afterwards: the next query has to
        # agree with a fresh instance.
        # A one-element target list dispatches to the single-pair path
        # (`is_single_target = len(targets) == 1`), so these come back flat,
        # not as a list of paths. That predates item 1.7.
        again = one.shortest_path(0, [reachable], algorithm="dijkstra")
        fresh = api(raster, self.steps).shortest_path(
            0, [reachable], algorithm="dijkstra")
        self.assertEqual(as_indices(again), as_indices(fresh))
        self.assertEqual(as_indices(paths[0]), as_indices(fresh))


class TestExactMargin(unittest.TestCase):
    """1.8: margin 1.00001 must return exactly what margin 1.1 returned.

    num_threads=1 throughout — the parallel delta-stepping kernel is a
    documented source of nondeterminism at >= 2 threads, which would mask
    (or fake) a margin regression.
    """

    def test_constant_is_the_clamp_floor(self):
        self.assertEqual(EXACT_TERMINATION_MARGIN, 1.00001)

    def test_single_pair_routes_unchanged(self):
        for raster in (rough_raster(), smooth_raster()):
            for neighborhood in ("r1", "r2"):
                steps = get_neighborhood_steps(neighborhood)
                a = api(raster, steps)
                for source, target in PAIRS:
                    with self.subTest(neighborhood=neighborhood,
                                      source=source, target=target):
                        loose = a.shortest_path(
                            source, target, algorithm="delta-stepping",
                            num_threads=1, margin=1.1)
                        tight = a.shortest_path(
                            source, target, algorithm="delta-stepping",
                            num_threads=1)
                        self.assertEqual(as_indices(tight), as_indices(loose))
                        self.assertEqual(cell_cost(tight, raster),
                                         cell_cost(loose, raster))

    def test_default_equals_explicit_floor(self):
        raster = rough_raster()
        steps = get_neighborhood_steps("r2")
        a = api(raster, steps)
        source, target = PAIRS[1]
        default = a.shortest_path(source, target, algorithm="delta-stepping",
                                  num_threads=1)
        explicit = a.shortest_path(source, target, algorithm="delta-stepping",
                                   num_threads=1,
                                   margin=EXACT_TERMINATION_MARGIN)
        self.assertEqual(as_indices(default), as_indices(explicit))

    def test_endpoints_still_connected(self):
        """The tightened margin must not truncate a route."""
        raster = rough_raster()
        steps = get_neighborhood_steps("r2")
        a = api(raster, steps)
        for source, target in PAIRS:
            with self.subTest(source=source, target=target):
                path = as_indices(a.shortest_path(
                    source, target, algorithm="delta-stepping",
                    num_threads=1))
                self.assertGreater(len(path), 1)
                self.assertEqual(path[0], source)
                self.assertEqual(path[-1], target)

    def test_pairwise_forwards_margin_and_threads(self):
        """The pairwise entry point forwarded neither before item 1.8."""
        raster = rough_raster()
        steps = get_neighborhood_steps("r1")
        a = api(raster, steps)
        sources = [s for s, _ in PAIRS]
        targets = [t for _, t in PAIRS]
        default = a.shortest_path(sources, targets,
                                  algorithm="delta-stepping", pairwise=True,
                                  num_threads=1)
        loose = a.shortest_path(sources, targets,
                                algorithm="delta-stepping", pairwise=True,
                                num_threads=1, margin=1.1)
        self.assertEqual([as_indices(p) for p in default],
                         [as_indices(p) for p in loose])


class TestGradientSolverReuse(unittest.TestCase):
    """The cached solver must keep its gradient configuration.

    Terrain: a uniform cost raster plus a 30 m plateau spanning rows 0-49 of
    columns 28-31. With a 50 % hard grade limit, every edge onto or off that
    plateau is forbidden (mult == inf), so a gradient-aware route MUST dip
    below row 49 to get across while a flat route runs straight. That makes
    "the cache silently dropped the LUTs" a visible route difference.
    """

    SOURCE = 25 * COLS + 5
    TARGET = 25 * COLS + 55

    def setUp(self):
        from pyorps.core.objective import Objective

        self.raster = np.full((ROWS, COLS), 100, dtype=np.uint16)
        self.steps = get_neighborhood_steps("r1")
        self.dem = np.zeros((ROWS, COLS), dtype=np.float32)
        self.dem[0:50, 28:32] = 30.0
        self.luts = Objective(
            {"length": 1.0},
            gradient_options={"max_gradient_pct": 50.0},
        ).build_gradient_luts(self.steps, 1.0)

    def _gradient_api(self):
        return CythonAPI(self.raster, self.steps, ignore_max=True,
                         dem_data=self.dem, gradient_luts=self.luts)

    @staticmethod
    def _max_row(path):
        return max(int(i) // COLS for i in path)

    def test_repeated_gradient_queries_match_fresh_instances(self):
        reused = self._gradient_api()
        for source, target in PAIRS[:3] + [(self.SOURCE, self.TARGET)]:
            with self.subTest(source=source, target=target):
                got = reused.shortest_path(source, target,
                                           algorithm="dijkstra")
                fresh = self._gradient_api().shortest_path(
                    source, target, algorithm="dijkstra")
                self.assertEqual(as_indices(got), as_indices(fresh))

    def test_gradient_terms_survive_repeated_queries(self):
        reused = self._gradient_api()
        flat = api(self.raster, self.steps).shortest_path(
            self.SOURCE, self.TARGET, algorithm="dijkstra")
        self.assertLess(self._max_row(flat), 50)
        for _ in range(3):
            path = reused.shortest_path(self.SOURCE, self.TARGET,
                                        algorithm="dijkstra")
            self.assertEqual(as_indices(path)[0], self.SOURCE)
            self.assertEqual(as_indices(path)[-1], self.TARGET)
            self.assertGreaterEqual(self._max_row(path), 50)


if __name__ == "__main__":
    unittest.main()

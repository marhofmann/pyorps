"""Phase 1b (API half): device-session ownership, O(path) extraction,
same-source dedup, and the relaxed reversed-single-solve guard.

Plan: ``docs/superpowers/plans/2026-08-07-realworld-5km-performance-plan.md``
items 1.2/1.3/1.4 (API half), 1.5 and 1.6. The kernel half lives in
``tests/test_utils/test_phase1b_gpu_session.py``.

Every item here is a **performance** item, so nearly every test is an
equivalence test against the behaviour it replaces:

* a reused API (one device session, one upload, path-local repair) must
  return what a *fresh API per query* returned before;
* a deduped pair batch must return what a per-pair loop returned, in the
  **caller's** order, for a deliberately shuffled pair list;
* the single reversed solve must return what k per-source solves
  returned, both with and without gradient LUTs.

Two properties make route *identity* (not merely route cost) a legitimate
assertion here, which it is not in general:

1. ``dist`` is bit-deterministic. Every candidate the async hot path can
   offer for a vertex is ``dist[u] + w`` with ``dist[u] >= d(u)``, and
   float addition is monotone, so the ``atomicMin`` result is the same
   value in every run regardless of the race order. (Measured: V5's field
   is bit-identical to V4's.)
2. ``pred`` is **not** deterministic -- the repair keeps an already-valid
   raced entry -- so wherever a route is compared across two independent
   solves it runs on :func:`corridor_raster`, whose passable cells form a
   simple path. There the optimum is unique, so *any* valid predecessor
   chain is the same chain. Random rasters are used only where costs are
   compared.

Costs are compared exactly (``assert_array_equal``) for same-direction
solves. The *reversed* solve is compared with a tolerance: it accumulates
the identical per-edge weights in the opposite order, and float32
addition is not associative.

.. note::
   The GPU classes must be run by the parent on the machine's GPU; this
   agent never executes CUDA. The CPU-only classes
   (``TestPairOrdering``, ``TestHostWalk``, ``TestStepSetGuard``) need no
   device and pin the reassembly logic that item 1.5 is most likely to
   get wrong.
"""

import numpy as np
import pytest

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU = True
    except Exception:                                 # pragma: no cover
        GPU = False
except ImportError:                                   # pragma: no cover
    GPU = False

from pyorps.core.exceptions import NoPathFoundError, PairwiseError
from pyorps.core.objective import Objective
from pyorps.graph.api.raster_gpu_api import RasterGPUAPI, _walk_predecessors
from pyorps.utils.traversal_gpu import prepare_step_lookup_tables

gpu_only = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

STEPS_4 = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)
STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
], dtype=np.int8)
#: R2 -- the production accuracy default (eight knight moves on top).
STEPS_16 = np.array(
    [list(s) for s in STEPS_8] + [
        [1, 2], [2, 1], [-1, 2], [-2, 1],
        [1, -2], [2, -1], [-1, -2], [-2, -1],
    ], dtype=np.int8)
#: Deliberately NOT closed under negation (item 1.6's safety net).
STEPS_HALF = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int8)

IMPASSABLE = 65535


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------

def corridor_cells(n, spacing=3):
    """Ordered cells of a width-1 serpentine corridor on an n x n grid.

    Legs sit on rows ``0, spacing, 2*spacing, ...`` and are joined at
    alternating ends. Legs are >= 3 rows apart, so no two legs touch even
    diagonally: with everything else impassable the corridor is a **simple
    path**, and the optimal route between any two of its cells is unique
    for R1, R2 and R3 alike (every off-corridor shortcut needs an
    impassable intermediate, which the kernels reject).
    """
    cells = []
    legs = list(range(0, n, spacing))
    for i, row in enumerate(legs):
        cols = range(n) if i % 2 == 0 else range(n - 1, -1, -1)
        cells.extend((row, col) for col in cols)
        if i + 1 < len(legs):
            end_col = n - 1 if i % 2 == 0 else 0
            cells.extend((r, end_col) for r in range(row + 1, legs[i + 1]))
    return cells


def corridor_raster(n=25, cheap=10, spacing=3):
    """(raster, ordered flat indices of the corridor)."""
    raster = np.full((n, n), IMPASSABLE, dtype=np.uint16)
    cells = corridor_cells(n, spacing)
    for row, col in cells:
        raster[row, col] = cheap
    order = np.array([r * n + c for r, c in cells], dtype=np.int64)
    return raster, order


def random_raster(n=48, seed=17, low=1, high=1000):
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=(n, n), dtype=np.uint16)


def tilted_dem(n, slope_pct=25.0):
    """A tilted plane; finite everywhere (the kernel rejects non-finite)."""
    rise = slope_pct / 100.0
    return (np.arange(n, dtype=np.float32)[:, None] * rise
            * np.ones((1, n), dtype=np.float32))


def gradient_luts_for(steps, weight=20.0):
    return Objective({"cost": 1.0, "gradient": weight}).build_gradient_luts(
        np.asarray(steps), cell_size=1.0, quant_scale=1.0)


def route_cost(path, raster, steps):
    """Re-walk a returned route in float64 (the exactness referee).

    Reproduces the kernel's edge model exactly:
    ``(raster[u] + raster[v] + sum(intermediates)) * cost_factor[step]``.
    Gradient-free, so only used on gradient-free configurations.
    """
    steps_i8, inter_lut, n_inter, cost_factors = prepare_step_lookup_tables(
        np.asarray(steps))
    cols = raster.shape[1]
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(steps_i8)}
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        ur, uc = divmod(int(u), cols)
        vr, vc = divmod(int(v), cols)
        s = index[(vr - ur, vc - uc)]
        acc = float(raster[ur, uc]) + float(raster[vr, vc])
        for k in range(int(n_inter[s])):
            dr, dc = inter_lut[s, k]
            acc += float(raster[ur + int(dr), uc + int(dc)])
        total += acc * float(cost_factors[s])
    return total


class CountingAPI(RasterGPUAPI):
    """RasterGPUAPI that records every kernel solve it dispatches.

    One entry per ``_solve_paths`` call = one SSSP launch. That is the
    quantity items 1.5 and 1.6 exist to reduce, so it is asserted
    directly rather than inferred from wall-clock.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solve_log = []

    def _solve_paths(self, source, target_indices):
        self.solve_log.append((
            int(source),
            np.asarray(target_indices, dtype=np.int32).ravel().tolist(),
        ))
        return super()._solve_paths(source, target_indices)


# =====================================================================
# CPU-only: the reassembly logic (no device needed)
# =====================================================================

class TestHostWalk:
    """`_walk_predecessors` -- the host twin of the on-device walk."""

    def test_walks_source_to_target_inclusive(self):
        dist = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        pred = np.array([-1, 0, 1, 2], dtype=np.int32)
        assert _walk_predecessors(pred, dist, 0, 3, 4) == [0, 1, 2, 3]

    def test_source_equals_target(self):
        dist = np.zeros(4, dtype=np.float32)
        pred = np.full(4, -1, dtype=np.int32)
        assert _walk_predecessors(pred, dist, 2, 2, 4) == [2]

    @pytest.mark.parametrize("target", [-1, 4, 99])
    def test_out_of_range_target_is_no_path(self, target):
        dist = np.zeros(4, dtype=np.float32)
        pred = np.full(4, -1, dtype=np.int32)
        assert _walk_predecessors(pred, dist, 0, target, 4) is None

    def test_unreachable_target_is_no_path(self):
        dist = np.array([0.0, np.inf, 1e30, 2.0], dtype=np.float32)
        pred = np.array([-1, -1, -1, 0], dtype=np.int32)
        assert _walk_predecessors(pred, dist, 0, 1, 4) is None
        assert _walk_predecessors(pred, dist, 0, 2, 4) is None

    def test_broken_link_is_no_path(self):
        dist = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        pred = np.array([-1, -1, 1, 2], dtype=np.int32)
        assert _walk_predecessors(pred, dist, 0, 3, 4) is None

    def test_cycle_is_no_path(self):
        dist = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        pred = np.array([-1, 2, 1, 2], dtype=np.int32)
        assert _walk_predecessors(pred, dist, 0, 3, 4) is None


class TestStepSetGuard:
    """Item 1.6 needs a step set closed under negation to reverse."""

    @pytest.mark.parametrize("steps", [STEPS_4, STEPS_8, STEPS_16])
    def test_standard_neighborhoods_are_reversible(self, steps):
        assert RasterGPUAPI._steps_closed_under_negation(steps)

    def test_half_step_set_is_not_reversible(self):
        assert not RasterGPUAPI._steps_closed_under_negation(STEPS_HALF)


def _ordering_harness():
    """A RasterGPUAPI with `_solve_paths` stubbed out -- no GPU involved.

    Bypasses ``__init__`` (which requires CuPy) on purpose: what is under
    test is the pair grouping and the scatter back into caller order, and
    that logic is device-independent.
    """
    api = object.__new__(RasterGPUAPI)
    api.last_path_costs = []
    api.solve_log = []
    api.symmetric_edge_weights = True
    api._reversible_steps = True

    def fake_solve_paths(source, target_indices):
        targets = np.asarray(target_indices, dtype=np.int32).ravel()
        api.solve_log.append((int(source), targets.tolist()))
        paths = [None if int(t) < 0 else [int(source), int(t)]
                 for t in targets]
        costs = np.array([np.inf if int(t) < 0 else 1000 * int(source) + int(t)
                          for t in targets], dtype=np.float32)
        return paths, costs

    api._solve_paths = fake_solve_paths
    return api


class TestPairOrdering:
    """Item 1.5: dedup must not disturb the caller's pair order."""

    #: Sources repeat, targets repeat, and the order is deliberately
    #: interleaved -- a group-major implementation that forgets to
    #: scatter back would pass a sorted list and fail this one.
    SHUFFLED = [(3, 7), (1, 5), (3, 9), (1, 7), (2, 5),
                (3, 5), (1, 9), (2, 9), (2, 7), (3, 7)]

    def test_results_follow_caller_order(self):
        api = _ordering_harness()
        result = api._paths_for_pairs(self.SHUFFLED)
        assert result == [[s, t] for s, t in self.SHUFFLED]

    def test_costs_follow_caller_order(self):
        api = _ordering_harness()
        api._paths_for_pairs(self.SHUFFLED)
        assert api.last_path_costs == [
            float(1000 * s + t) for s, t in self.SHUFFLED]

    def test_one_solve_per_distinct_source(self):
        api = _ordering_harness()
        api._paths_for_pairs(self.SHUFFLED)
        assert [s for s, _ in api.solve_log] == [3, 1, 2]

    def test_group_targets_keep_first_appearance_order(self):
        api = _ordering_harness()
        api._paths_for_pairs(self.SHUFFLED)
        grouped = dict(api.solve_log)
        assert grouped[3] == [7, 9, 5, 7]     # duplicates preserved
        assert grouped[1] == [5, 7, 9]
        assert grouped[2] == [5, 9, 7]

    def test_no_path_becomes_empty_list_without_disturbing_neighbours(self):
        api = _ordering_harness()
        pairs = [(1, 5), (1, -1), (2, 7)]
        result = api._paths_for_pairs(pairs)
        assert result == [[1, 5], [], [2, 7]]
        assert api.last_path_costs[1] == float("inf")

    def test_empty_pair_list_solves_nothing(self):
        api = _ordering_harness()
        assert api._paths_for_pairs([]) == []
        assert api.solve_log == []

    def test_multi_to_multi_product_order(self):
        api = _ordering_harness()
        result = api._shortest_path_multi_to_multi([4, 5], [8, 9], False)
        assert result == [[4, 8], [4, 9], [5, 8], [5, 9]]
        assert [s for s, _ in api.solve_log] == [4, 5]

    def test_multi_to_multi_pairwise_length_mismatch(self):
        api = _ordering_harness()
        with pytest.raises(PairwiseError):
            api._shortest_path_multi_to_multi([1, 2], [3], True)

    def test_multi_to_single_reverses_one_solve(self):
        api = _ordering_harness()
        result = api._shortest_path_multi_to_single([4, 5, 4], 9)
        # one solve, rooted at the TARGET, sources as its targets
        assert api.solve_log == [(9, [4, 5, 4])]
        # each chain flipped back into source -> target order
        assert result == [[4, 9], [5, 9], [4, 9]]

    def test_multi_to_single_falls_back_when_asymmetric(self):
        api = _ordering_harness()
        api.symmetric_edge_weights = False
        result = api._shortest_path_multi_to_single([4, 5], 9)
        assert [s for s, _ in api.solve_log] == [4, 5]
        assert result == [[4, 9], [5, 9]]

    def test_multi_to_single_falls_back_on_half_step_set(self):
        api = _ordering_harness()
        api._reversible_steps = False
        api._shortest_path_multi_to_single([4, 5], 9)
        assert [s for s, _ in api.solve_log] == [4, 5]


# =====================================================================
# GPU: equivalence
# =====================================================================

@gpu_only
class TestSessionLifetime:
    """Item 1.2 -- one device session per API, released on drop."""

    def test_session_is_lazy(self):
        raster, _order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            assert api._session is None
            assert api.device_bytes == 0
        finally:
            api.close()

    def test_session_is_built_once_and_reused(self):
        raster, order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            api.shortest_path(int(order[0]), int(order[10]))
            session = api._session
            assert session is not None
            assert api.device_bytes > 0
            api.shortest_path(int(order[3]), int(order[20]))
            assert api._session is session          # no re-upload
        finally:
            api.close()

    def test_close_releases_and_is_idempotent(self):
        raster, order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        api.shortest_path(int(order[0]), int(order[10]))
        assert api.device_bytes > 0
        api.close()
        assert api._session is None
        assert api.device_bytes == 0
        api.close()                                  # idempotent
        assert api.device_bytes == 0

    def test_query_after_close_rebuilds(self):
        raster, order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        first = api.shortest_path(int(order[0]), int(order[12]))
        api.close()
        second = api.shortest_path(int(order[0]), int(order[12]))
        api.close()
        assert first == second

    def test_context_manager_closes(self):
        raster, order = corridor_raster()
        with RasterGPUAPI(raster, STEPS_8) as api:
            api.shortest_path(int(order[0]), int(order[8]))
            assert api.device_bytes > 0
        assert api.device_bytes == 0

    def test_invalidate_session_forces_a_fresh_upload(self):
        raster, order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            api.shortest_path(int(order[0]), int(order[8]))
            first = api._session
            api.invalidate_session()
            assert api._session is None
            api.shortest_path(int(order[0]), int(order[8]))
            assert api._session is not None and api._session is not first
        finally:
            api.close()

    def test_resident_footprint_is_proportional_to_the_raster(self):
        """Guards against a per-query allocation leak or an O(n^2) buffer.

        Budget: raster 2 + dist 4 + pred 4 + arena 4 = 14 B/cell, plus the
        arena's 4096-per-ring floor (512 KiB) and the small step tables.
        """
        raster, order = corridor_raster(n=25)
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            for i in range(5):
                api.shortest_path(int(order[0]), int(order[5 + i]))
            budget = 14 * raster.size + 32 * 4096 * 4 + (1 << 16)
            assert 0 < api.device_bytes <= budget
        finally:
            api.close()


@gpu_only
class TestSessionEquivalence:
    """Items 1.2/1.3/1.4 -- the reused, O(path) route answers the same."""

    def test_reused_api_matches_fresh_api_per_query(self):
        raster, order = corridor_raster()
        pairs = [(int(order[0]), int(order[15])),
                 (int(order[4]), int(order[30])),
                 (int(order[30]), int(order[4])),
                 (int(order[9]), int(order[9]))]

        reused = RasterGPUAPI(raster, STEPS_8)
        try:
            for source, target in pairs:
                fresh = RasterGPUAPI(raster, STEPS_8)
                try:
                    expected = fresh.shortest_path(source, target)
                    expected_cost = list(fresh.last_path_costs)
                finally:
                    fresh.close()
                got = reused.shortest_path(source, target)
                assert got == expected
                np.testing.assert_array_equal(
                    reused.last_path_costs, expected_cost)
        finally:
            reused.close()

    def test_o_path_route_matches_the_full_field_route(self):
        """The device walk + path-local repair (1.3/1.4) against the
        full dist/pred download + host walk it replaces."""
        raster, order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            source = int(order[2])
            targets = [int(order[i]) for i in (7, 18, 33, 2)]
            lean = api.shortest_path(source, targets)
            lean_costs = list(api.last_path_costs)

            dist, pred = api._run_sssp(source, np.asarray(targets,
                                                          dtype=np.int32))
            for i, target in enumerate(targets):
                assert lean[i] == api._extract_path(pred, dist, source,
                                                    target)
                assert lean_costs[i] == pytest.approx(float(dist[target]))
        finally:
            api.close()

    def test_costs_are_bit_identical_on_a_random_raster(self):
        """Route identity is not asserted here (ties exist); the field is
        bit-deterministic, so the costs are."""
        raster = random_raster(n=48)
        n = raster.shape[0]
        pairs = [(0, n * n - 1), (n * n - 1, 0), (n + 1, n * n - n - 2),
                 (5 * n + 5, 40 * n + 40)]
        reused = RasterGPUAPI(raster, STEPS_16)
        try:
            for source, target in pairs:
                fresh = RasterGPUAPI(raster, STEPS_16)
                try:
                    fresh.shortest_path(source, target)
                    expected = np.asarray(fresh.last_path_costs)
                finally:
                    fresh.close()
                reused.shortest_path(source, target)
                np.testing.assert_array_equal(
                    np.asarray(reused.last_path_costs), expected)
        finally:
            reused.close()

    def test_returned_cost_is_the_route_rewalked_in_float64(self):
        raster, order = corridor_raster()
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            path = api.shortest_path(int(order[1]), int(order[26]))
            reported = api.last_path_costs[0]
            assert reported == pytest.approx(
                route_cost(path, raster, STEPS_8), rel=1e-5)
        finally:
            api.close()

    def test_unreachable_single_pair_still_raises(self):
        raster, order = corridor_raster()
        blocked = int(np.flatnonzero(raster.ravel() == IMPASSABLE)[0])
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            with pytest.raises(NoPathFoundError):
                api.shortest_path(int(order[0]), blocked)
        finally:
            api.close()

    def test_unreachable_target_in_a_batch_is_an_empty_list(self):
        raster, order = corridor_raster()
        blocked = int(np.flatnonzero(raster.ravel() == IMPASSABLE)[0])
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            paths = api.shortest_path(
                int(order[0]), [int(order[6]), blocked, int(order[11])])
            assert paths[1] == []
            assert paths[0][0] == int(order[0])
            assert paths[2][-1] == int(order[11])
            assert np.isinf(api.last_path_costs[1])
        finally:
            api.close()


# =====================================================================
# GPU: item 1.5 -- same-source dedup and target batching
# =====================================================================

@gpu_only
class TestSameSourceDedup:

    #: Sources and targets both repeat and the pairs are interleaved.
    def _pairs(self, order):
        idx = [(0, 20), (6, 33), (0, 33), (12, 20), (6, 20),
               (0, 6), (12, 33), (6, 6), (12, 12), (0, 20)]
        return [(int(order[a]), int(order[b])) for a, b in idx]

    def test_pairwise_matches_the_per_pair_loop_for_a_shuffled_list(self):
        raster, order = corridor_raster()
        pairs = self._pairs(order)
        sources = [s for s, _ in pairs]
        targets = [t for _, t in pairs]

        loop = RasterGPUAPI(raster, STEPS_8)
        try:
            expected = [loop.shortest_path(s, t) for s, t in pairs]
        finally:
            loop.close()

        batched = CountingAPI(raster, STEPS_8)
        try:
            got = batched.shortest_path(sources, targets, pairwise=True)
            assert got == expected
            # one solve per DISTINCT source, in first-appearance order
            assert [s for s, _ in batched.solve_log] == list(
                dict.fromkeys(sources))
        finally:
            batched.close()

    def test_pairwise_costs_match_the_per_pair_loop(self):
        raster, order = corridor_raster()
        pairs = self._pairs(order)
        loop = RasterGPUAPI(raster, STEPS_8)
        try:
            expected = []
            for source, target in pairs:
                loop.shortest_path(source, target)
                expected.append(loop.last_path_costs[0])
        finally:
            loop.close()

        batched = RasterGPUAPI(raster, STEPS_8)
        try:
            batched.shortest_path([s for s, _ in pairs],
                                  [t for _, t in pairs], pairwise=True)
            np.testing.assert_array_equal(
                np.asarray(batched.last_path_costs), np.asarray(expected))
        finally:
            batched.close()

    def test_all_pairs_matrix_order_and_solve_count(self):
        """The dominant workload: an s x t matrix over shared sources."""
        raster, order = corridor_raster()
        sources = [int(order[i]) for i in (0, 10, 25)]
        targets = [int(order[i]) for i in (5, 18, 31, 40)]

        loop = RasterGPUAPI(raster, STEPS_8)
        try:
            expected = [loop.shortest_path(s, t)
                        for s in sources for t in targets]
        finally:
            loop.close()

        batched = CountingAPI(raster, STEPS_8)
        try:
            got = batched.shortest_path(sources, targets)
            assert got == expected                       # product order
            assert len(batched.solve_log) == len(sources)
            for i, source in enumerate(sources):
                assert batched.solve_log[i][0] == source
                assert batched.solve_log[i][1] == targets
        finally:
            batched.close()

    def test_matrix_endpoints_land_on_the_right_pair(self):
        raster, order = corridor_raster()
        sources = [int(order[i]) for i in (2, 14)]
        targets = [int(order[i]) for i in (8, 22, 36)]
        api = RasterGPUAPI(raster, STEPS_8)
        try:
            paths = api.shortest_path(sources, targets)
            expected = [(s, t) for s in sources for t in targets]
            for path, (source, target) in zip(paths, expected):
                assert path[0] == source and path[-1] == target
        finally:
            api.close()

    def test_single_source_multi_target_is_one_solve(self):
        raster, order = corridor_raster()
        api = CountingAPI(raster, STEPS_8)
        try:
            api.shortest_path(int(order[0]),
                              [int(order[i]) for i in (9, 21, 34)])
            assert len(api.solve_log) == 1
        finally:
            api.close()


# =====================================================================
# GPU: item 1.6 -- the reversed single solve
# =====================================================================

@gpu_only
class TestReversedSingleSolve:

    def test_multi_to_single_is_one_solve_without_a_dem(self):
        raster, order = corridor_raster()
        sources = [int(order[i]) for i in (3, 17, 29, 3)]
        target = int(order[41])

        loop = RasterGPUAPI(raster, STEPS_8)
        try:
            expected = [loop.shortest_path(s, target) for s in sources]
        finally:
            loop.close()

        api = CountingAPI(raster, STEPS_8)
        try:
            got = api.shortest_path(sources, target)
            assert got == expected
            assert len(api.solve_log) == 1
            assert api.solve_log[0][0] == target      # rooted at target
        finally:
            api.close()

    def test_multi_to_single_is_one_solve_with_a_bare_dem(self):
        """The guard item 1.6 removed: dem_data without gradient_luts
        never reaches the kernel, so it cannot make anything asymmetric."""
        raster, order = corridor_raster()
        dem = tilted_dem(raster.shape[0])
        sources = [int(order[i]) for i in (2, 16, 28)]
        target = int(order[39])

        loop = RasterGPUAPI(raster, STEPS_8, dem_data=dem)
        try:
            expected = [loop.shortest_path(s, target) for s in sources]
        finally:
            loop.close()

        api = CountingAPI(raster, STEPS_8, dem_data=dem)
        try:
            got = api.shortest_path(sources, target)
            assert got == expected
            assert len(api.solve_log) == 1
        finally:
            api.close()

    def test_multi_to_single_is_one_solve_with_gradient_luts(self):
        raster, order = corridor_raster()
        dem = tilted_dem(raster.shape[0])
        luts = gradient_luts_for(STEPS_8)
        sources = [int(order[i]) for i in (1, 15, 27)]
        target = int(order[38])

        loop = RasterGPUAPI(raster, STEPS_8, dem_data=dem,
                            gradient_luts=luts)
        try:
            expected = []
            expected_costs = []
            for source in sources:
                expected.append(loop.shortest_path(source, target))
                expected_costs.append(loop.last_path_costs[0])
        finally:
            loop.close()

        api = CountingAPI(raster, STEPS_8, dem_data=dem, gradient_luts=luts)
        try:
            got = api.shortest_path(sources, target)
            assert got == expected
            # Reversed accumulation: identical per-edge weights summed in
            # the opposite order, so float32 rounding is the only slack.
            np.testing.assert_allclose(
                np.asarray(api.last_path_costs),
                np.asarray(expected_costs), rtol=1e-5)
            assert len(api.solve_log) == 1
        finally:
            api.close()

    def test_edge_weights_are_direction_symmetric_with_gradient_luts(self):
        """THE assumption behind item 1.6.

        The kernel bins ``fabsf(dem[v] - dem[u])`` and applies the same
        LUT pair both ways, so ``d(s -> t) == d(t -> s)``. A **directional**
        slope response (signed height difference -- the eikonal
        increment-3 lineage) would break this: whoever lands one must set
        ``RasterGPUAPI.symmetric_edge_weights = False`` for the affected
        configurations, and this test is what will fail first if they
        do not.
        """
        raster = random_raster(n=40, seed=5)
        n = raster.shape[0]
        dem = tilted_dem(n, slope_pct=35.0)
        # a DEM with structure in BOTH axes, so an uphill-vs-downhill
        # asymmetry could not cancel out by symmetry of the plane
        rng = np.random.default_rng(11)
        dem = (dem + rng.random((n, n)).astype(np.float32) * 3.0).astype(
            np.float32)
        luts = gradient_luts_for(STEPS_16)
        pairs = [(0, n * n - 1), (n + 3, 30 * n + 21), (7 * n, 33 * n + 39)]

        api = RasterGPUAPI(raster, STEPS_16, dem_data=dem,
                           gradient_luts=luts)
        try:
            for source, target in pairs:
                _paths, forward = api._solve_paths(source, [target])
                _paths, backward = api._solve_paths(target, [source])
                assert np.isfinite(forward[0])
                np.testing.assert_allclose(forward, backward, rtol=1e-5)
        finally:
            api.close()

    def test_asymmetric_flag_falls_back_to_per_source_solves(self):
        """The escape hatch a directional response would use."""
        raster, order = corridor_raster()
        sources = [int(order[i]) for i in (4, 19, 31)]
        target = int(order[42])

        symmetric = RasterGPUAPI(raster, STEPS_8)
        try:
            expected = symmetric.shortest_path(sources, target)
        finally:
            symmetric.close()

        api = CountingAPI(raster, STEPS_8)
        api.symmetric_edge_weights = False
        try:
            got = api.shortest_path(sources, target)
            assert got == expected
            assert [s for s, _ in api.solve_log] == sources
        finally:
            api.close()

    def test_half_step_set_is_not_reversed(self):
        """A step set that is not closed under negation makes the raster a
        genuinely directed graph; reversing it would answer a question
        about the reverse graph."""
        raster, order = corridor_raster()
        api = CountingAPI(raster, STEPS_HALF)
        try:
            assert api._reversible_steps is False
            sources = [int(order[0]), int(order[1])]
            api.shortest_path(sources, int(order[20]))
            assert [s for s, _ in api.solve_log] == sources
        finally:
            api.close()

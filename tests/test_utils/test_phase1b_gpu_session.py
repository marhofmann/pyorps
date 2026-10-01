"""Phase 1b (GPU): device-resident session, O(path) extraction, path-local
predecessor repair, right-sized ring arena.

Plan: ``docs/superpowers/plans/2026-08-07-realworld-5km-performance-plan.md``
items 1.2 (kernel/launcher half), 1.3, 1.4 and 1.11.

Every item here is a *performance* item, so almost every test is an
equivalence test. Three properties are pinned:

1. **Distances do not move.** A session solve, a smaller arena and a
   spilling ring push must leave the distance field bit-identical to the
   V4 persistent kernel -- the same reference ``test_sssp_gpu_v5.py``
   uses.
2. **The device path is the host path.** ``extract_paths`` must return
   exactly what walking the downloaded ``pred`` returns, and the
   path-local repair must produce exactly the chain the full-raster
   ``v5_repair_pred`` pass would have produced from the same raw
   predecessor field.
3. **Route costs are the field.** Re-walking a returned path over the
   raster must reproduce ``dist[target]``.

Route *identity* among equal-cost optima is deliberately never asserted
across two separate solves: the V5 hot path is asynchronous, so the raw
``pred`` it leaves differs run to run and a tie can be broken either way.
Where routes are compared they are compared against the *same* device
state, which makes the comparison exact.

.. warning::
   ``TestArenaOverflow`` runs rasters with 0-cost plateaus far larger
   than one ring. That class is the regression test for the livelock the
   spilling ring push removes -- on the pre-1.11 kernel the rendezvous
   rescans the same oversubscribed bucket forever and the launch never
   returns. Run that class in its own pytest process; if it hangs, the
   spill is broken and the GPU needs resetting.
"""
import numpy as np
import pytest

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU = True
    except Exception:
        GPU = False
except ImportError:
    GPU = False

# Import-safe without CuPy: sssp_gpu guards its cupy import.
from pyorps.utils.sssp_gpu import (
    _v5_arena_cap, _V5_N_BUCKETS, _V5_DEFAULT_ARENA_FACTOR,
)

if GPU:
    from pyorps.utils.sssp_gpu import (
        GpuSsspSession, sssp_raster_gpu_paths, sssp_raster_gpu_v4,
        sssp_raster_gpu_v5, _check_v5_available,
    )
    from pyorps.utils.traversal_gpu import prepare_step_lookup_tables

gpu_only = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

STEPS_4 = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)
STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
], dtype=np.int8)
# R2: the production accuracy default adds the eight knight moves.
STEPS_16 = np.array(
    [list(s) for s in STEPS_8] + [
        [1, 2], [2, 1], [-1, 2], [-2, 1],
        [1, -2], [2, -1], [-1, -2], [-2, -1],
    ], dtype=np.int8)


# ---------------------------------------------------------------------
# host references
# ---------------------------------------------------------------------

def _host_walk(pred, dist, source, target):
    """Reconstruct one path on the host, exactly as RasterGPUAPI does.

    Returns an int32 array source -> target, or None when no path exists.
    """
    n = int(dist.size)
    source, target = int(source), int(target)
    if target < 0 or target >= n:
        return None
    if not np.isfinite(dist[target]) or dist[target] >= 1e29:
        return None
    path, seen, current = [], set(), target
    while current != source:
        path.append(current)
        if current in seen or len(path) > n:
            return None
        seen.add(current)
        p = int(pred[current])
        if p < 0:
            return None
        current = p
    path.append(source)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _path_cost(raster, steps, path):
    """Re-price a path over the raster in float64 (exactness referee)."""
    _, ilut, n_inter, cost_factors = prepare_step_lookup_tables(
        np.asarray(steps))
    cols = raster.shape[1]
    lookup = {(int(steps[s][0]), int(steps[s][1])): s
              for s in range(len(steps))}
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        ur, uc = divmod(int(u), cols)
        vr, vc = divmod(int(v), cols)
        s = lookup[(vr - ur, vc - uc)]
        acc = float(raster[ur, uc]) + float(raster[vr, vc])
        for k in range(int(n_inter[s])):
            acc += float(raster[ur + int(ilut[s, k, 0]),
                                uc + int(ilut[s, k, 1])])
        total += acc * float(cost_factors[s])
    return total


def _assert_bit_exact(got, ref):
    finite_ref = np.isfinite(ref) & (ref < 1e29)
    finite_got = np.isfinite(got) & (got < 1e29)
    assert np.array_equal(finite_ref, finite_got), "reachability mask differs"
    if finite_ref.any():
        assert float(np.max(np.abs(got[finite_ref] - ref[finite_ref]))) == 0.0


def _random_raster(n, seed, lo=1, hi=200):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, (n, n)).astype(np.uint16)


# ---------------------------------------------------------------------
# item 1.11 -- arena sizing (pure host arithmetic, runs without a GPU)
# ---------------------------------------------------------------------

class TestArenaSizing:
    """The sizing rule the no-drop guarantee rests on.

    ``v5_ring_push`` spills a full bucket into the next ring, so the
    binding capacity is the whole ring set, not one bucket. The
    span-boundary refill pushes at most one entry per pixel, so
    ``32 * cap >= n_pixels`` makes a refill drop impossible -- and a
    refill drop is exactly what the rewind cannot make progress against
    (it rescans the same span base and re-pushes the same oversubscribed
    bucket).
    """

    @pytest.mark.parametrize("n", [
        1, 1000, 4096, 131_071, 131_072, 131_073, 250_000, 1_000_000,
        25_000_000, 100_000_000,
    ])
    def test_ring_set_holds_one_entry_per_pixel(self, n):
        assert _v5_arena_cap(n) * _V5_N_BUCKETS >= n, (
            "the refill could drop a vertex, which the rewind cannot "
            "make progress against")

    @pytest.mark.parametrize("n", [1, 1000, 65_536])
    def test_small_rasters_keep_the_4096_floor(self, n):
        """Below 32*4096/factor pixels the floor, not the factor, decides."""
        assert _v5_arena_cap(n) == 4096

    @pytest.mark.parametrize("n", [1_000_000, 25_000_000, 100_000_000])
    def test_bytes_per_cell_matches_the_default_factor(self, n):
        """Default is 2.0 = 8 B/cell (V5 total 18), measured 2026-08-10.

        The old sizing was 3.0 (12 B/cell). A sweep over 2000^2 and 3000^2
        on random / heavy-tail / plateau rasters put 2.0 at 0.97-1.01x of
        3.0 - free within noise - while 1.0 cost up to 5.4% on heavy-tail.
        See benchmarks/benchmark_arena_factor.py.
        """
        arena_bytes = _v5_arena_cap(n) * _V5_N_BUCKETS * 4
        assert arena_bytes / n == pytest.approx(
            4.0 * _V5_DEFAULT_ARENA_FACTOR, rel=1e-3)

    def test_factor_scales_capacity(self):
        n = 10_000_000
        assert _v5_arena_cap(n, 2.0) == pytest.approx(
            2 * _v5_arena_cap(n, 1.0), rel=1e-6)

    def test_rejects_factor_below_one(self):
        """<1.0 lets the span-boundary refill drop, which livelocks."""
        for bad in (0.0, 0.5, 0.999):
            with pytest.raises(ValueError, match=r">= 1\.0"):
                _v5_arena_cap(1_000_000, bad)


# ---------------------------------------------------------------------
# guard: V5 must still compile
# ---------------------------------------------------------------------

@gpu_only
def test_v5_kernel_still_compiles():
    """A CUDA compile error would silently demote every solve to V4.

    ``_check_v5_available`` swallows the exception by design, so without
    this the whole suite would still pass -- on the slower kernel.
    """
    assert _check_v5_available(), (
        "the V5 async kernel no longer compiles; every solve has silently "
        "fallen back to V4")


# ---------------------------------------------------------------------
# item 1.2 -- device-resident session
# ---------------------------------------------------------------------

@gpu_only
class TestSessionEquivalence:
    @pytest.mark.parametrize("n,steps", [
        (50, STEPS_4), (120, STEPS_8), (200, STEPS_8), (150, STEPS_16),
    ])
    def test_session_matches_v4(self, n, steps):
        raster = _random_raster(n, n)
        ref = sssp_raster_gpu_v4(raster, steps, 0, delta="auto")
        with GpuSsspSession(raster, steps) as session:
            got = session.solve(0)
        _assert_bit_exact(got, ref)

    def test_repeated_solves_on_one_session_match_fresh_solves(self):
        """Buffer reuse must leave no residue between queries."""
        raster = _random_raster(160, 7)
        sources = [0, 160 * 80 + 80, 160 * 160 - 1, 5]
        fresh = [sssp_raster_gpu_v4(raster, STEPS_8, s, delta="auto")
                 for s in sources]
        with GpuSsspSession(raster, STEPS_8) as session:
            # twice in order, then reversed: any leftover arena, control
            # or reservation state would surface here.
            for _ in range(2):
                for s, ref in zip(sources, fresh):
                    _assert_bit_exact(session.solve(s), ref)
            for s, ref in reversed(list(zip(sources, fresh))):
                _assert_bit_exact(session.solve(s), ref)

    def test_session_and_sessionless_entry_points_agree(self):
        raster = _random_raster(180, 11)
        targets = np.array([180 * 180 - 1, 180 * 90 + 90], dtype=np.int32)
        d_ref, p_ref = sssp_raster_gpu_v5(
            raster, STEPS_8, 0, target_indices=targets,
            return_predecessor=True)
        with GpuSsspSession(raster, STEPS_8) as session:
            d_ses, p_ses = sssp_raster_gpu_v5(
                raster, STEPS_8, 0, target_indices=targets,
                return_predecessor=True, session=session)
        _assert_bit_exact(d_ses, d_ref)
        for t in targets:
            a = _host_walk(p_ref, d_ref, 0, int(t))
            b = _host_walk(p_ses, d_ses, 0, int(t))
            assert a is not None and b is not None
            # tie-breaking may differ between two asynchronous solves;
            # the cost may not.
            assert _path_cost(raster, STEPS_8, a) == pytest.approx(
                _path_cost(raster, STEPS_8, b), rel=1e-5)

    def test_predecessors_do_not_survive_a_no_predecessor_query(self):
        """A pred-less solve must not leave an extractable stale chain."""
        raster = _random_raster(100, 21)
        target = 100 * 100 - 1
        with GpuSsspSession(raster, STEPS_8) as session:
            session.solve(0, return_predecessor=True)
            session.solve(50, return_predecessor=False)
            with pytest.raises(RuntimeError):
                session.extract_paths(0, [target])

    def test_reports_and_frees_its_device_memory(self):
        n = 300
        cells = n * n
        raster = _random_raster(n, 3)
        session = GpuSsspSession(raster, STEPS_8)
        session.solve(0, return_predecessor=True)
        # raster 2 + dist 4 + pred 4 + arena (4 B/cell above the 4096
        # per-ring floor), plus the small step tables and control words.
        expected = (2 + 4 + 4) * cells + 4 * _V5_N_BUCKETS * _v5_arena_cap(
            cells)
        assert expected <= session.device_bytes <= expected + (1 << 16)
        session.close()
        assert session.closed
        assert session.device_bytes == 0
        with pytest.raises(RuntimeError):
            session.solve(0)
        session.close()  # idempotent

    def test_rejects_a_raster_of_a_different_shape(self):
        raster = _random_raster(60, 4)
        with GpuSsspSession(raster, STEPS_8) as session:
            with pytest.raises(ValueError):
                sssp_raster_gpu_v5(
                    _random_raster(61, 4), STEPS_8, 0, session=session)
            with pytest.raises(ValueError):
                sssp_raster_gpu_v5(
                    raster.astype(np.float32), STEPS_8, 0, session=session)

    def test_blocked_source_returns_the_empty_field(self):
        raster = _random_raster(40, 5)
        raster[0, 0] = 65535
        with GpuSsspSession(raster, STEPS_8) as session:
            dist = session.solve(0)
            assert np.all(np.isinf(dist))
            paths, costs = session.solve_paths(0, [40 * 40 - 1])
        assert paths == [None]
        assert np.isinf(costs[0])

    def test_float32_raster_session(self):
        rng = np.random.default_rng(31)
        n = 140
        raster = rng.uniform(0.5, 200.0, (n, n)).astype(np.float32)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        with GpuSsspSession(raster, STEPS_8) as session:
            _assert_bit_exact(session.solve(0), ref)
            paths, costs = session.solve_paths(0, [n * n - 1])
        assert paths[0] is not None
        assert costs[0] == pytest.approx(ref[n * n - 1], rel=1e-6)

    def test_gradient_session(self):
        from pyorps.core.objective import Objective
        rng = np.random.default_rng(12)
        n = 130
        raster = rng.integers(1, 200, (n, n)).astype(np.uint16)
        x, y = np.meshgrid(np.arange(n, dtype=np.float32),
                           np.arange(n, dtype=np.float32))
        dem = (20.0 * np.sin(x / 15.0) * np.cos(y / 20.0)).astype(np.float32)
        obj = Objective({"cost": 1.0},
                        gradient_options={"multiplier": "exponential",
                                          "max_gradient_pct": 60.0})
        luts = obj.build_gradient_luts(STEPS_8, cell_size=10.0)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto",
                                 dem=dem, gradient_luts=luts)
        with GpuSsspSession(raster, STEPS_8, dem=dem,
                            gradient_luts=luts) as session:
            _assert_bit_exact(session.solve(0), ref)
            paths, costs = session.solve_paths(0, [n * n - 1])
        assert paths[0] is not None
        assert costs[0] == pytest.approx(ref[n * n - 1], rel=1e-6)


# ---------------------------------------------------------------------
# item 1.3 -- on-device path extraction
# ---------------------------------------------------------------------

@gpu_only
class TestDevicePathExtraction:
    @pytest.mark.parametrize("n,steps", [
        (60, STEPS_4), (140, STEPS_8), (110, STEPS_16),
    ])
    def test_device_walk_equals_host_walk(self, n, steps):
        """Same device state, so this comparison is exact, not statistical."""
        raster = _random_raster(n, n + 1)
        targets = np.array(
            [n * n - 1, n * (n // 2) + n // 2, n - 1, n * (n - 1)],
            dtype=np.int32)
        with GpuSsspSession(raster, steps) as session:
            dist, pred = session.solve(
                0, target_indices=targets, return_predecessor=True,
                full_repair=True)
            # repair=False: the full pass already ran, so the device walk
            # must follow exactly the pred array that was downloaded.
            paths, costs = session.extract_paths(0, targets, repair=False)
        for i, t in enumerate(targets):
            expected = _host_walk(pred, dist, 0, int(t))
            assert expected is not None
            assert np.array_equal(paths[i], expected), f"target {t}"
            assert costs[i] == dist[int(t)]

    def test_paths_reprice_to_the_distance_field(self):
        n = 150
        raster = _random_raster(n, 77)
        targets = np.array([n * n - 1, n * 40 + 120], dtype=np.int32)
        with GpuSsspSession(raster, STEPS_8) as session:
            paths, costs = session.solve_paths(0, targets)
        for path, cost in zip(paths, costs):
            assert path is not None
            assert int(path[0]) == 0
            assert _path_cost(raster, STEPS_8, path) == pytest.approx(
                float(cost), rel=1e-4)

    def test_unreachable_and_out_of_range_targets(self):
        raster = np.ones((60, 60), dtype=np.uint16)
        raster[:, 30] = 65535  # impassable column (ignore_max=True)
        targets = np.array([0, 60 * 10 + 5, 60 * 10 + 50, 60 * 60 + 5, -1],
                           dtype=np.int32)
        with GpuSsspSession(raster, STEPS_8) as session:
            paths, costs = session.solve_paths(0, targets)
        assert np.array_equal(paths[0], np.array([0], dtype=np.int32))
        assert costs[0] == 0.0
        assert paths[1] is not None
        assert paths[2] is None and np.isinf(costs[2])   # behind the wall
        assert paths[3] is None and np.isinf(costs[3])   # out of range
        assert paths[4] is None and np.isinf(costs[4])   # negative

    def test_module_level_paths_entry_point(self):
        n = 120
        raster = _random_raster(n, 91)
        targets = np.array([n * n - 1, n * 60 + 60], dtype=np.int32)
        dist = sssp_raster_gpu_v5(raster, STEPS_8, 0,
                                  target_indices=targets)
        paths, costs = sssp_raster_gpu_paths(raster, STEPS_8, 0, targets)
        for i, t in enumerate(targets):
            assert paths[i] is not None
            assert int(paths[i][0]) == 0
            assert int(paths[i][-1]) == int(t)
            assert costs[i] == pytest.approx(dist[int(t)], rel=1e-6)

    def test_no_targets(self):
        raster = _random_raster(40, 2)
        with GpuSsspSession(raster, STEPS_8) as session:
            paths, costs = session.solve_paths(0, [])
        assert paths == [] and costs.size == 0

    def test_many_targets_one_solve(self):
        n = 120
        raster = _random_raster(n, 55)
        rng = np.random.default_rng(56)
        targets = rng.choice(n * n, size=200, replace=False).astype(np.int32)
        with GpuSsspSession(raster, STEPS_8) as session:
            dist, pred = session.solve(
                0, target_indices=targets, return_predecessor=True)
            paths, costs = session.extract_paths(0, targets, repair=False)
        for i, t in enumerate(targets):
            expected = _host_walk(pred, dist, 0, int(t))
            if expected is None:
                assert paths[i] is None
            else:
                assert np.array_equal(paths[i], expected)


# ---------------------------------------------------------------------
# item 1.4 -- path-local predecessor repair
# ---------------------------------------------------------------------

@gpu_only
class TestPathLocalRepair:
    @pytest.mark.parametrize("n,steps,seed", [
        (100, STEPS_4, 1), (140, STEPS_8, 2), (110, STEPS_16, 3),
    ])
    def test_path_local_repair_equals_full_repair(self, n, steps, seed):
        """Both repairs are applied to the *same* raw predecessor field.

        The raw field is snapshotted before the path-local pass and
        restored before the full pass, so any difference is a difference
        between the two repairs and not asynchronous tie-breaking.
        """
        raster = _random_raster(n, seed)
        targets = np.array(
            [n * n - 1, n * (n // 2) + n // 2, n - 1], dtype=np.int32)
        with GpuSsspSession(raster, steps) as session:
            dist, raw_pred = session.solve(
                0, target_indices=targets, return_predecessor=True,
                full_repair=False)
            local_paths, local_costs = session.extract_paths(
                0, targets, repair=True)
            # restore the raw field, then run the full-raster pass on it
            session._d_pred.set(raw_pred)
            session._run_full_repair(0)
            full_pred = session._d_pred.get()
        for i, t in enumerate(targets):
            expected = _host_walk(full_pred, dist, 0, int(t))
            assert expected is not None, f"full repair lost target {t}"
            assert np.array_equal(local_paths[i], expected), (
                f"path-local repair diverged from the full repair at {t}")
            assert local_costs[i] == dist[int(t)]

    def test_repair_is_idempotent(self):
        """Repairing a chain twice must not move it."""
        n = 130
        raster = _random_raster(n, 8)
        targets = np.array([n * n - 1], dtype=np.int32)
        with GpuSsspSession(raster, STEPS_8) as session:
            session.solve(0, target_indices=targets,
                          return_predecessor=True, full_repair=False)
            once, _ = session.extract_paths(0, targets, repair=True)
            twice, _ = session.extract_paths(0, targets, repair=True)
            no_repair, _ = session.extract_paths(0, targets, repair=False)
        assert np.array_equal(once[0], twice[0])
        assert np.array_equal(once[0], no_repair[0])

    def test_plateau_chain_is_a_valid_walk(self):
        """0-cost plateaus are where raced pred entries actually bite."""
        n = 90
        raster = _random_raster(n, 9, lo=1, hi=50)
        raster[20:60, 20:60] = 0
        target = n * n - 1
        with GpuSsspSession(raster, STEPS_8) as session:
            paths, costs = session.solve_paths(0, [target])
        path = paths[0]
        assert path is not None
        assert int(path[0]) == 0 and int(path[-1]) == target
        assert len(set(path.tolist())) == len(path), "repeated cell in path"
        assert _path_cost(raster, STEPS_8, path) == pytest.approx(
            float(costs[0]), rel=1e-4)


# ---------------------------------------------------------------------
# item 1.11 -- overflow / spill behaviour on the device
# ---------------------------------------------------------------------

@gpu_only
@gpu_only
class TestV4TruncationIsLoud:
    """V4 must not answer with a partial field.

    V4's light phase runs at most ``max_light_iterations * window``
    iterations per bucket. A 0-cost region puts every edge at weight 0, so
    the whole raster stays in one bucket and the cap is reached with the
    frontier still non-empty. Before this guard V4 returned whatever it had
    settled and left the rest at ``inf`` - indistinguishable from genuinely
    unreachable. Measured on 700x700 all-zero: 160,801 settled (401^2, i.e.
    exactly the 400 rings the cap allows) and 329,199 cells silently wrong.

    V4 is only the fallback when V5 will not compile, but a fallback that
    lies is worse than one that refuses.
    """

    def test_zero_cost_raster_raises_instead_of_truncating(self):
        raster = np.zeros((700, 700), dtype=np.uint16)
        with pytest.raises(RuntimeError, match="abandoned part of the frontier"):
            sssp_raster_gpu_v4(raster, STEPS_8, 0)

    def test_v5_solves_what_v4_refuses(self):
        """The same input V4 cannot finish, V5 answers correctly."""
        raster = np.zeros((700, 700), dtype=np.uint16)
        with GpuSsspSession(raster, STEPS_8) as session:
            dist = np.asarray(session.solve(0))
        assert np.isfinite(dist).all(), "every cell is reachable at cost 0"
        assert float(dist.max()) == 0.0

    def test_ordinary_raster_is_unaffected(self):
        """The guard must not fire on a raster that drains normally."""
        raster = _random_raster(600, 3, lo=1, hi=900)
        ref = np.asarray(sssp_raster_gpu_v4(raster, STEPS_8, 0))
        with GpuSsspSession(raster, STEPS_8) as session:
            _assert_bit_exact(np.asarray(session.solve(0)), ref)


class TestArenaOverflow:
    """Rasters whose single delta-bucket dwarfs one ring.

    .. warning::
       On the pre-1.11 kernel (per-bucket capacity, no spill) these
       rasters livelock: the refill drops, the rewind rescans the same
       span base, and the launch never returns. They are the regression
       test for that. If one hangs, the spilling ring push is broken --
       do not "wait a bit longer".

    A V4 mismatch here is worth investigating on the V4 side as well:
    V4's frontier buffers are fixed at 8*n entries with no self-heal.
    """

    @pytest.mark.parametrize("size,plateau", [
        (120, (10, 110, 10, 110)),   # 10 000 equal-dist cells, one bucket
        (200, (40, 160, 40, 160)),   # 14 400
    ])
    def test_zero_cost_plateau_matches_v4(self, size, plateau):
        r0, r1, c0, c1 = plateau
        raster = _random_raster(size, size, lo=1, hi=300)
        raster[r0:r1, c0:c1] = 0
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        with GpuSsspSession(raster, STEPS_8) as session:
            got = session.solve(0)
        _assert_bit_exact(got, ref)

    def test_all_zero_raster(self):
        """Every reachable cell shares one distance and one bucket.

        The answer needs no reference implementation: every edge weighs
        zero, so every cell is at distance zero.
        """
        n = 100
        raster = np.zeros((n, n), dtype=np.uint16)
        with GpuSsspSession(raster, STEPS_8) as session:
            got = session.solve(0)
            paths, costs = session.solve_paths(0, [n * n - 1])
        assert np.all(got == 0.0)
        assert paths[0] is not None
        assert costs[0] == 0.0

    @pytest.mark.parametrize("factor", [1.0, 1.5, 4.0])
    def test_arena_factor_does_not_move_the_field(self, factor):
        """More headroom changes only how often the rewind fires.

        n must stay above the 4096-per-ring floor or this proves nothing:
        at n=160 (25,600 px) every factor yields max(4096, 800..3200) =
        4096, i.e. a byte-identical arena, and the test compares a
        configuration against itself. At n=512 (262,144 px) the caps are
        8,192 / 12,288 / 32,768 - genuinely different arenas.
        """
        n = 512
        assert _v5_arena_cap(n * n, factor) > 4096, (
            "arena factors collapsed to the floor; this test would be a no-op")
        raster = _random_raster(n, 41)
        raster[200:380, 60:460] = 0          # plateau: heavy spilling
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        with GpuSsspSession(raster, STEPS_8,
                            arena_factor=factor) as session:
            _assert_bit_exact(session.solve(0), ref)

    def test_shrunk_arena_matches_the_old_sizing(self):
        """The direct regression for item 1.11: 1.0 vs the old 3n/32."""
        n = 512
        raster = _random_raster(n, 77, lo=1, hi=4000)   # heavy tail
        with GpuSsspSession(raster, STEPS_8, arena_factor=1.0) as small:
            got = small.solve(0)
        with GpuSsspSession(raster, STEPS_8, arena_factor=3.0) as old:
            ref = old.solve(0)
        _assert_bit_exact(got, ref)

    def test_arena_factor_below_one_is_rejected(self):
        """Sub-1.0 breaks the refill-cannot-drop guarantee."""
        with pytest.raises(ValueError, match=r"arena_factor must be >= 1\.0"):
            GpuSsspSession(_random_raster(64, 1), STEPS_8, arena_factor=0.5)

    def test_plateau_with_targets_and_paths(self):
        n = 140
        raster = _random_raster(n, 42, lo=1, hi=400)
        raster[30:110, 30:110] = 0
        targets = np.array([n * n - 1, n * 70 + 70], dtype=np.int32)
        ref = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto")
        with GpuSsspSession(raster, STEPS_8) as session:
            paths, costs = session.solve_paths(0, targets)
        for i, t in enumerate(targets):
            assert paths[i] is not None
            assert costs[i] == pytest.approx(ref[int(t)], rel=1e-6)
            assert _path_cost(raster, STEPS_8, paths[i]) == pytest.approx(
                float(costs[i]), rel=1e-4)

"""Tower fields: the recurrence, the bounds, and the reconstruction.

``docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md``,
phases 2 to 5. The oracle comparison against the constrained kernel
lives next door in ``test_tower_field_oracle.py`` -- that one is Phase
0 and it is the one that decides whether any of this means anything.

Here the reference is an explicit Bellman-Ford over tower chains that
uses no prefix sum, no sliding window and no run length: it enumerates
every admissible ``(direction, span)`` for every cell. If the fast form
and that form disagree, the optimisation is wrong -- which is exactly
how the corner-cutting defect showed up (a diagonal span slipping
between two excluded cells, which PYORPS' own step model forbids).
"""

import math

import numpy as np
import pytest

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.tower_field import (
    AngleTables,
    ClearanceModel,
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
    angle_tables_from_profile,
    check_bounds,
    clearance_from_profile,
    coarsen,
    cost_factor,
    intermediate_offsets,
    tower_field_bounds,
    tower_field_from_raster,
)
from pyorps.utils.directional import primitive_directions

PROFILE_PATH = "profiles/overhead_line_110kv.yaml"


@pytest.fixture(scope="module")
def profile():
    return InfrastructureProfile.load(PROFILE_PATH)


# ----------------------------------------------------------------------
# the explicit reference
# ----------------------------------------------------------------------

def _brute_force(values, tower, dirs, sigma, model, src, blocked=None):
    """Tower chains by enumeration: no prefix, no window, no run length."""
    h, w = values.shape
    blocked = (np.zeros(values.shape, bool) if blocked is None
               else np.asarray(blocked, bool))

    def step_cost(p, q, r, c):
        inter = intermediate_offsets(p, q)
        acc = values[r - p, c - q] + values[r, c]
        for orow, ocol in inter:
            acc += values[r - p + orow, c - q + ocol]
        return acc * cost_factor(p, q) * sigma

    node = tower.copy()
    node[src] = (model.terminal_tower_cost
                 if model.charge_terminal_towers else 0.0)
    T = np.full(values.shape, np.inf)
    T[src] = 0.0
    for _ in range(400):
        new = T.copy()
        for r in range(h):
            for c in range(w):
                best = new[r, c]
                for p, q in dirs:
                    ds = math.hypot(p, q) * sigma
                    for m in range(1, int(model.max_span_m / ds) + 2):
                        length = m * ds
                        if (length > model.max_span_m + 1e-9
                                or (not model.max_span_inclusive
                                    and length >= model.max_span_m - 1e-9)):
                            break
                        yr, yc = r - m * p, c - m * q
                        if not (0 <= yr < h and 0 <= yc < w):
                            break
                        if length < model.min_span_m - 1e-9:
                            continue
                        chain = T[yr, yc] + node[yr, yc]
                        if not np.isfinite(chain):
                            continue
                        if any(blocked[r - j * p, c - j * q]
                               for j in range(m + 1)):
                            continue
                        if any(blocked[r - (j + 1) * p + orow,
                                       c - (j + 1) * q + ocol]
                               for j in range(m)
                               for orow, ocol in intermediate_offsets(p, q)):
                            continue
                        total = chain + sum(
                            step_cost(p, q, r - j * p, c - j * q)
                            for j in range(m))
                        best = min(best, total)
                new[r, c] = best
        if np.array_equal(np.nan_to_num(new, posinf=-1.0),
                          np.nan_to_num(T, posinf=-1.0)):
            return T
        T = new
    raise AssertionError("the reference did not converge")


class TestAgainstBruteForce:
    CASES = [
        ((11, 11), 10.0, 20.0, 45.0, 1),
        ((9, 13), 5.0, 10.0, 26.0, 2),
        ((12, 12), 10.0, 0.0, 35.0, 2),
    ]

    @pytest.mark.parametrize("shape,sigma,lo,hi,dmax", CASES)
    def test_tier1_matches(self, shape, sigma, lo, hi, dmax):
        rng = np.random.default_rng(7)
        values = rng.integers(1, 40, size=shape).astype(np.float64)
        tower = rng.integers(100, 400, size=shape).astype(np.float64)
        blocked = rng.random(shape) < 0.08
        src = (shape[0] // 2, 1)
        blocked[src] = False
        tower[blocked] = np.inf

        dirs = primitive_directions(dmax)
        lattice = TowerLattice(cell_size_m=sigma, factor=1, directions=dirs)
        model = TowerFieldModel(min_span_m=lo, max_span_m=hi,
                                charge_terminal_towers=False)
        field = TowerFieldSolver(values=values, tower_cost=tower,
                                 lattice=lattice, model=model,
                                 blocked=blocked).solve(src)
        want = _brute_force(values, tower, dirs.tolist(), sigma, model, src,
                            blocked)
        assert np.array_equal(np.isfinite(field.arrival), np.isfinite(want))
        fin = np.isfinite(want)
        assert np.allclose(field.arrival[fin], want[fin], rtol=0, atol=1e-8)

    def test_forbidden_crossing_respects_intermediates(self):
        """A diagonal span may not slip between two excluded cells.

        PYORPS' step model samples the supercover of a step, and the
        constrained kernel refuses a step whose intermediates are
        excluded. Testing only the cells ON the ray lets a diagonal
        through a diagonal gap, which undercut the kernel by 25-380 EUR
        before this was fixed.
        """
        values = np.ones((9, 9))
        tower = np.full((9, 9), 10.0)
        blocked = np.zeros((9, 9), bool)
        blocked[4, 5] = blocked[5, 4] = True      # a diagonal pinhole at (4,4)->(5,5)
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(1))
        model = TowerFieldModel(min_span_m=0.0, max_span_m=20.0,
                                charge_terminal_towers=False)
        solver = TowerFieldSolver(values=values, tower_cost=tower,
                                  lattice=lattice, model=model,
                                  blocked=blocked)
        run = solver._tables[
            [i for i, d in enumerate(lattice.directions.tolist())
             if tuple(d) == (1, 1)][0]]["run"]
        assert run[5, 5] == 0, "the pinhole diagonal must not be a usable step"


class TestLayeredConvergence:
    def test_sweeps_track_extent_not_cell_count(self):
        """The layered-DAG property, which is why there is no queue.

        Every edge adds exactly one tower, so the sweep count is the
        maximum tower count on an optimal line -- it grows with the
        EXTENT of the window and not with how finely that window is
        sampled.
        """
        sweeps = {}
        for n in (24, 48, 96):
            rng = np.random.default_rng(1)
            values = rng.integers(1, 20, size=(n, n)).astype(np.float64)
            lattice = TowerLattice(cell_size_m=10.0, factor=1,
                                   directions=primitive_directions(1))
            model = TowerFieldModel(min_span_m=50.0, max_span_m=200.0,
                                    charge_terminal_towers=False)
            field = TowerFieldSolver(
                values=values, tower_cost=np.full((n, n), 1000.0),
                lattice=lattice, model=model).solve((n // 2, 0),
                                                    record_pred=False)
            sweeps[n * 10] = field.sweeps
        extents = sorted(sweeps)
        counts = [sweeps[e] for e in extents]
        assert counts == sorted(counts)
        # 4x the cells (24 -> 48) must not cost 4x the sweeps
        assert counts[-1] < 4 * counts[0]

    def test_refining_the_lattice_does_not_add_sweeps(self):
        """Same extent, finer sampling: the sweep count is about the same."""
        counts = []
        for factor, n in ((2, 40), (1, 80)):
            rng = np.random.default_rng(2)
            values = rng.integers(1, 20, size=(80, 80)).astype(np.float64)
            field = tower_field_from_raster(
                values.astype(np.uint16), cell_size_m=5.0,
                source_cell=(n // 2, 0), factor=factor,
                directions=primitive_directions(1),
                model=TowerFieldModel(min_span_m=50.0, max_span_m=150.0,
                                      charge_terminal_towers=False),
                tower_cost=np.full((n, n), 1000.0), impassable=None,
                record_pred=False)
            counts.append(field.sweeps)
        assert abs(counts[0] - counts[1]) <= 2, counts


class TestTiers:
    def test_tier1_is_below_tier2(self, profile):
        """Tier 1 drops a non-negative premium, so it cannot exceed tier 2."""
        rng = np.random.default_rng(2)
        raster = rng.integers(1, 300, size=(80, 80)).astype(np.uint16)
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(40, 2),
                      factor=1, directions=primitive_directions(2),
                      record_pred=False)
        t1 = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, angle_tier=1), **common)
        t2 = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, angle_tier=2), **common)
        both = np.isfinite(t1.arrival) & np.isfinite(t2.arrival)
        assert both.sum() > 1000
        assert np.all(t2.arrival[both] >= t1.arrival[both] - 1e-9)

    def test_window_angle_mode_never_undercuts_exact(self, profile):
        """The K x n_classes trick pools directions, so it can only raise."""
        rng = np.random.default_rng(3)
        raster = rng.integers(1, 300, size=(70, 70)).astype(np.uint16)
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(35, 2),
                      factor=1, directions=primitive_directions(2),
                      record_pred=False)
        exact = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, angle_tier=2, angle_mode="exact"), **common)
        window = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, angle_tier=2, angle_mode="window"), **common)
        both = np.isfinite(exact.arrival) & np.isfinite(window.arrival)
        assert np.all(window.arrival[both] >= exact.arrival[both] - 1e-9)
        assert window.meta["angle_classes"] is not None

    def test_window_mode_refuses_a_predecessor_plane(self, profile):
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(2))
        solver = TowerFieldSolver(
            values=np.ones((20, 20)), tower_cost=np.full((20, 20), 1e4),
            lattice=lattice,
            model=TowerFieldModel.matching_kernel(
                profile, angle_tier=2, angle_mode="window"),
            angles=angle_tables_from_profile(profile, lattice))
        with pytest.raises(ValueError, match="cannot name"):
            solver.solve((10, 1), record_pred=True)


class TestBounds:
    def test_lower_never_exceeds_upper(self, profile):
        rng = np.random.default_rng(4)
        raster = rng.integers(1, 300, size=(70, 70)).astype(np.uint16)
        raster[20:50, 35] = 65535
        lower, upper = tower_field_bounds(
            raster, cell_size_m=10.0, profile=profile, source_cell=(35, 2),
            factor=1, directions=primitive_directions(2))
        report = check_bounds(lower, upper)
        assert report["cells_compared"] > 500
        assert report["gap_min_eur"] >= -1e-9
        assert lower.meta["bound"].startswith("lower")
        assert upper.meta["bound"].startswith("upper")

    def test_a_violated_bound_is_caught(self, profile):
        """The invariant has to actually fire, or it is decoration."""
        rng = np.random.default_rng(5)
        raster = rng.integers(1, 200, size=(40, 40)).astype(np.uint16)
        lower, upper = tower_field_bounds(
            raster, cell_size_m=10.0, profile=profile, source_cell=(20, 2),
            factor=1, directions=primitive_directions(1))
        finite = np.flatnonzero(np.isfinite(lower.arrival.ravel()))
        lower.arrival.ravel()[finite[0]] = np.inf
        upper.arrival.ravel()[finite[0]] = 1.0
        with pytest.raises(AssertionError, match="no relaxation can do"):
            check_bounds(lower, upper)

    def test_restricting_the_feasible_set_raises_cost(self, profile):
        """A smaller direction set cannot produce a cheaper line.

        This is the inequality behind the plan's bound table, tested on
        the one restriction that is exactly comparable: r1's eight
        directions are a SUBSET of r2's sixteen, on the same lattice,
        with the same cost model, so every r1 chain is also an r2 chain.

        A coarser SIGMA is the same kind of restriction in principle,
        but not a clean test of it: pooling changes the cost model as
        well as the feasible set, and a mean-pooled span integral can
        come out below a finely sampled one on the same terrain.
        """
        rng = np.random.default_rng(6)
        raster = rng.integers(20, 300, size=(70, 70)).astype(np.uint16)
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(35, 2),
                      factor=1, record_pred=False,
                      model=TowerFieldModel.matching_kernel(profile))
        wide = tower_field_from_raster(
            raster, directions=primitive_directions(2), **common)
        narrow = tower_field_from_raster(
            raster, directions=primitive_directions(1), **common)
        both = np.isfinite(wide.arrival) & np.isfinite(narrow.arrival)
        assert both.sum() > 500
        assert np.all(narrow.arrival[both] >= wide.arrival[both] - 1e-9)

    def test_min_pooling_is_below_max_pooling(self, profile):
        """The other half of the bound table: how a block collapses."""
        rng = np.random.default_rng(16)
        raster = rng.integers(1, 300, size=(80, 80)).astype(np.uint16)
        common = dict(cell_size_m=5.0, profile=profile, source_cell=(20, 1),
                      factor=2, directions=primitive_directions(2),
                      model=TowerFieldModel.matching_kernel(profile),
                      record_pred=False)
        low = tower_field_from_raster(raster, pooling="min", **common)
        high = tower_field_from_raster(raster, pooling="max", **common)
        both = np.isfinite(low.arrival) & np.isfinite(high.arrival)
        assert np.all(high.arrival[both] >= low.arrival[both] - 1e-9)


class TestClearanceAndCrossings:
    def test_clearance_only_adds_cost(self, profile):
        rng = np.random.default_rng(8)
        raster = rng.integers(1, 200, size=(60, 60)).astype(np.uint16)
        dem = rng.random((60, 60)) * 40.0
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(30, 2),
                      factor=1, directions=primitive_directions(2),
                      model=TowerFieldModel.matching_kernel(profile),
                      record_pred=False)
        flat = tower_field_from_raster(raster, **common)
        hilly = tower_field_from_raster(raster, dem=dem, **common)
        both = np.isfinite(flat.arrival) & np.isfinite(hilly.arrival)
        assert np.all(hilly.arrival[both] >= flat.arrival[both] - 1e-9)
        # an unreachable cell under clearance was reachable without it
        assert np.isfinite(hilly.arrival).sum() <= np.isfinite(
            flat.arrival).sum()

    def test_both_ends_charging_dominates_arriving(self, profile):
        rng = np.random.default_rng(9)
        raster = rng.integers(1, 200, size=(50, 50)).astype(np.uint16)
        dem = rng.random((50, 50)) * 40.0
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(25, 2),
                      factor=1, directions=primitive_directions(2),
                      dem=dem, record_pred=False)
        one = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, clearance_charge="arriving"), **common)
        two = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, clearance_charge="both_ends"), **common)
        both = np.isfinite(one.arrival) & np.isfinite(two.arrival)
        assert np.all(two.arrival[both] >= one.arrival[both] - 1e-9)

    def test_blocked_fraction_controls_a_coarse_lattice(self, profile):
        """At a coarse sigma, "any excluded cell blocks" blocks everything.

        A 20 m lattice cell covers 400 raster cells, so scattered
        exclusions make a target unreachable that the router walks
        straight past -- measured on the CIRED wind farm, where the
        cable reached PCC1 and the overhead field called it unreachable.
        The share is the knob; 0.0 stays the conservative default.
        """
        rng = np.random.default_rng(14)
        raster = rng.integers(1, 200, size=(200, 200)).astype(np.uint16)
        # 2 % scattered exclusions: at factor 20 almost every lattice
        # cell then contains at least one.
        raster[rng.random((200, 200)) < 0.02] = 65535
        # The source's own lattice cell must be clear under BOTH rules,
        # or the strict run refuses to start instead of demonstrating
        # anything. Lattice cell (5, 1) at factor 20 is raster rows
        # 100-119, columns 20-39.
        raster[100:120, 20:40] = 50
        common = dict(cell_size_m=1.0, profile=profile, source_cell=(5, 1),
                      factor=20, directions=primitive_directions(2),
                      model=TowerFieldModel.matching_kernel(profile),
                      record_pred=False)
        strict = tower_field_from_raster(raster, blocked_fraction=0.0,
                                         **common)
        relaxed = tower_field_from_raster(raster, blocked_fraction=0.2,
                                          **common)
        assert strict.reachable.sum() < relaxed.reachable.sum()
        both = strict.reachable & relaxed.reachable
        # Relaxing only ADDS spans, so it can never cost more.
        assert np.all(relaxed.arrival[both] <= strict.arrival[both] + 1e-9)

    def test_unblock_lets_a_line_terminate_on_an_excluded_cell(self,
                                                               profile):
        """A line ENDS on something already built, not on greenfield.

        A PCC inside a substation compound sits on excluded ground, so
        no new tower may stand there -- but the line terminates on the
        existing gantry. Without the exemption the target reads as
        unreachable while a cable routes straight to it (measured on
        the CIRED wind farm: PCC0's lattice cell is 80 % excluded).
        """
        from affine import Affine

        rng = np.random.default_rng(15)
        raster = rng.integers(1, 200, size=(120, 120)).astype(np.uint16)
        # The compound lines up with the lattice, so the target's own
        # cell is fully excluded while its approach is clear. A target
        # buried in the MIDDLE of a multi-cell exclusion stays
        # unreachable however it is priced -- no span can arrive.
        raster[100:120, 100:120] = 65535
        transform = Affine(1.0, 0, 5e5, 0, -1.0, 5.6e6)
        common = dict(cell_size_m=1.0, profile=profile, source_cell=(5, 5),
                      factor=10, directions=primitive_directions(2),
                      transform=transform, record_pred=False,
                      model=TowerFieldModel.as_reported(profile))
        target = (5e5 + 105.5, 5.6e6 - 105.5)    # the compound's near cell

        shut = tower_field_from_raster(raster, **common)
        assert not np.isfinite(shut.costs_to([target])[0])

        opened = tower_field_from_raster(raster, unblock_xy=[target],
                                         **common)
        got = opened.costs_to([target])[0]
        assert np.isfinite(got), "the terminal exemption did not apply"
        # Only the named cell is freed; the rest of the compound stays out.
        deep = (5e5 + 115.5, 5.6e6 - 115.5)
        assert not np.isfinite(opened.costs_to([deep])[0])

    def test_spans_may_cross_what_towers_may_not_stand_on(self, profile):
        """Two different questions, and the default conflates them.

        The exclusion layers behind a cost raster forbid GROUND WORKS;
        a conductor passing overhead is a separate consent. Measured on
        the CIRED wind farm, PCC0 sits in a compound whose whole 3x3
        lattice neighbourhood is >= 80 % excluded, so conflating them
        leaves it with no overhead cost at all.
        """
        raster = np.full((160, 160), 20, dtype=np.uint16)
        raster[:, 70:95] = 65535            # a wide excluded band
        common = dict(cell_size_m=1.0, profile=profile, source_cell=(8, 1),
                      factor=10, directions=primitive_directions(2),
                      record_pred=False,
                      model=TowerFieldModel.as_reported(profile))
        far = (8, 14)                        # the other side of the band

        shut = tower_field_from_raster(raster, **common)
        assert not np.isfinite(shut.arrival[far])

        over = tower_field_from_raster(raster, spans_cross_exclusions=True,
                                       **common)
        assert np.isfinite(over.arrival[far])
        # The exclusion still stops a tower STANDING in the band.
        assert not np.isfinite(over.arrival[8, 8])

    def test_unblock_cells_needs_no_transform(self, profile):
        raster = np.full((80, 80), 20, dtype=np.uint16)
        raster[70:80, 70:80] = 65535
        field = tower_field_from_raster(
            raster, cell_size_m=1.0, profile=profile, source_cell=(1, 1),
            factor=10, directions=primitive_directions(2),
            unblock_cells=[(7, 7)], record_pred=False,
            model=TowerFieldModel.as_reported(profile))
        assert np.isfinite(field.arrival[7, 7])

    def test_unblock_xy_without_a_transform_is_refused(self, profile):
        with pytest.raises(ValueError, match="georeferenced"):
            tower_field_from_raster(
                np.full((60, 60), 20, dtype=np.uint16), cell_size_m=1.0,
                profile=profile, source_cell=(1, 1), factor=10,
                directions=primitive_directions(1),
                unblock_xy=[(0.0, 0.0)], record_pred=False)

    def test_conservative_crossings_never_cheaper_than_exact(self, profile):
        rng = np.random.default_rng(10)
        raster = rng.integers(1, 200, size=(60, 60)).astype(np.uint16)
        raster[15:45, 30] = 65535
        raster[30, 30] = 40
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(30, 2),
                      factor=1, directions=primitive_directions(2),
                      record_pred=False)
        exact = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, forbidden_mode="exact"), **common)
        cons = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, forbidden_mode="conservative"), **common)
        off = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(
                profile, forbidden_mode="off"), **common)
        for tighter, looser in ((off, exact), (exact, cons)):
            both = np.isfinite(tighter.arrival) & np.isfinite(looser.arrival)
            assert np.all(looser.arrival[both] >= tighter.arrival[both] - 1e-9)


class TestReconstruction:
    def test_tower_sequence_reproduces_the_field_value(self, profile):
        from pyorps.graph.tower_field_oracle import score_tower_chain

        rng = np.random.default_rng(11)
        raster = rng.integers(1, 200, size=(60, 60)).astype(np.uint16)
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(2))
        model = TowerFieldModel.matching_kernel(profile)
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, model=model,
            source_cell=(30, 2), factor=1,
            directions=primitive_directions(2))
        lut = profile.precompute_tower_terrain_costs()
        values = raster.astype(np.float64)
        tables = angle_tables_from_profile(profile, lattice)
        usable = tables.premium[tables.valid & np.isfinite(tables.premium)]
        tower_cost = lut[raster.astype(np.int64)] + float(usable.min())

        for target in [(30, 50), (12, 44), (55, 33)]:
            if not np.isfinite(field.arrival[target]):
                continue
            seq = field.tower_sequence(*target)
            assert seq[0].row, seq[0].col == field.source
            assert (seq[-1].row, seq[-1].col) == target
            score = score_tower_chain(
                [(t.row, t.col) for t in seq], values=values,
                tower_cost=tower_cost, lattice=lattice, model=model)
            assert score.violations == (), score.violations
            assert score.total == pytest.approx(
                float(field.arrival[target]), rel=1e-9)

    def test_tier2_sequence_reproduces_its_own_value(self, profile):
        """The layered predecessor walk, which is easy to get subtly wrong.

        Tier 2 keeps one plane per ARRIVING direction, plus a separate
        plane for the last span into the queried point, and the walk has
        to hand the incoming direction from one to the next. Scoring the
        reconstructed chain with the same angle table the field charged
        is what proves it did.
        """
        from pyorps.graph.tower_field_oracle import score_tower_chain

        rng = np.random.default_rng(17)
        raster = rng.integers(1, 200, size=(70, 70)).astype(np.uint16)
        dirs = primitive_directions(2)
        lattice = TowerLattice(cell_size_m=10.0, factor=1, directions=dirs)
        angles = angle_tables_from_profile(profile, lattice)
        model = TowerFieldModel.matching_kernel(profile, angle_tier=2)
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, model=model,
            source_cell=(35, 2), factor=1, directions=dirs, angles=angles)

        lut = profile.precompute_tower_terrain_costs()
        tower_cost = lut[raster.astype(np.int64)]
        checked = 0
        for target in [(35, 60), (10, 50), (60, 45), (20, 20)]:
            if not np.isfinite(field.arrival[target]):
                continue
            seq = field.tower_sequence(*target)
            if len(seq) < 3:
                continue
            checked += 1
            score = score_tower_chain(
                [(t.row, t.col) for t in seq],
                values=raster.astype(np.float64), tower_cost=tower_cost,
                lattice=lattice, model=model, angles=angles)
            assert score.violations == (), (target, score.violations)
            assert score.total == pytest.approx(
                float(field.arrival[target]), rel=1e-9)
        assert checked >= 2

    def test_spans_stay_inside_the_model(self, profile):
        rng = np.random.default_rng(12)
        raster = rng.integers(1, 200, size=(70, 70)).astype(np.uint16)
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, source_cell=(35, 2),
            factor=1, directions=primitive_directions(2))
        seq = field.tower_sequence(35, 60)
        spans = [t.span_to_next_m for t in seq[:-1]]
        assert spans, "expected a multi-tower line"
        assert all(s < profile.max_span_m + 1e-9 for s in spans)
        # every span but the last must also clear min_span
        assert all(s >= profile.min_span_m - 1e-9 for s in spans[:-1])
        assert field.length_at(35, 60) == pytest.approx(sum(spans))

    def test_geometry_needs_a_transform(self, profile):
        raster = np.full((40, 40), 10, dtype=np.uint16)
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, source_cell=(20, 1),
            factor=1, directions=primitive_directions(1))
        assert field.route_geometry(20, 30) is None      # no transform
        from affine import Affine
        geo = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, source_cell=(20, 1),
            factor=1, directions=primitive_directions(1),
            transform=Affine(10, 0, 5e5, 0, -10, 5.6e6))
        line = geo.route_geometry(20, 30)
        assert line is not None
        assert line.length == pytest.approx(geo.length_at(20, 30))

    def test_no_predecessor_plane_is_an_explicit_refusal(self, profile):
        raster = np.full((30, 30), 10, dtype=np.uint16)
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, source_cell=(15, 1),
            factor=1, directions=primitive_directions(1), record_pred=False)
        assert not field.has_paths
        with pytest.raises(NotImplementedError, match="record_pred"):
            field.tower_sequence(15, 25)


class TestAnisotropicPixels:
    """Pixels are not square, and one scalar cannot stand for two sizes.

    ``mod2_raster_wp.tiff`` reports "1 m resolution" everywhere and
    carries ``transform.a = 1.0000017516`` against
    ``transform.e = -0.9999783186``. Taking ``abs(transform.a)`` for both
    -- which is what every caller used to pass as ``cell_size_m`` --
    moved a real 110 kV route's reconstructed cost by up to 0.045 %,
    about 1 500 EUR, and did so with BOTH signs.
    """

    SX, SY = 2.0, 3.0            # far apart on purpose, so it is visible

    @staticmethod
    def _transform(sx, sy):
        from affine import Affine
        return Affine(sx, 0.0, 5e5, 0.0, -sy, 5.6e6)

    def test_step_length_mixes_the_two_pixel_sizes(self):
        lat = TowerLattice(cell_size_x_m=self.SX, cell_size_y_m=self.SY,
                           factor=2, directions=primitive_directions(2))
        assert (lat.sigma_x_m, lat.sigma_y_m) == (2 * self.SX, 2 * self.SY)
        for p, q in ((1, 0), (0, 1), (1, 1), (2, -1)):
            assert lat.step_length_m((p, q)) == pytest.approx(
                math.hypot(p * lat.sigma_y_m, q * lat.sigma_x_m))
        # rows carry the y size, so a row step is NOT sigma_x
        assert lat.step_length_m((1, 0)) == pytest.approx(lat.sigma_y_m)

    def test_square_lattice_keeps_the_scalar_form_bit_for_bit(self):
        lat = TowerLattice(cell_size_m=10.0, factor=2,
                           directions=primitive_directions(3))
        assert lat.is_square and lat.sigma_m == 20.0
        for d in lat.directions.tolist():
            assert lat.step_length_m(d) == math.hypot(d[0], d[1]) * 20.0

    def test_sigma_m_refuses_to_collapse(self):
        lat = TowerLattice(cell_size_x_m=self.SX, cell_size_y_m=self.SY)
        with pytest.raises(ValueError, match="no single sigma_m"):
            lat.sigma_m
        # and there is no scalar cell size either, rather than a guess
        assert lat.cell_size_m is None

    def test_cell_sizes_come_as_a_pair(self):
        with pytest.raises(ValueError, match="come as a pair"):
            TowerLattice(cell_size_x_m=2.0)
        with pytest.raises(ValueError, match="contradicts"):
            TowerLattice(cell_size_m=2.0, cell_size_x_m=2.0,
                         cell_size_y_m=3.0)
        with pytest.raises(ValueError, match="square grid"):
            TowerLattice()

    def test_transform_supplies_both_pixel_sizes(self):
        n = 20
        field = tower_field_from_raster(
            np.full((n, n), 10, dtype=np.uint16),
            source_cell=(0, 0), transform=self._transform(self.SX, self.SY),
            directions=primitive_directions(1), impassable=None,
            tower_cost=np.zeros((n, n)), record_pred=False,
            model=TowerFieldModel(min_span_m=0.0, max_span_m=20.0,
                                  charge_terminal_towers=False))
        assert field.lattice.cell_size_x_m == self.SX
        assert field.lattice.cell_size_y_m == self.SY
        # no scalar sigma in the metadata either, for the same reason
        assert field.meta["sigma_m"] is None
        assert field.meta["sigma_y_m"] == self.SY

    def test_angle_tables_use_the_physical_deflection(self, profile):
        """An anisotropic step does not point along ``atan2(q, p)``.

        The hard angle limit and the tower TYPE are both decided on the
        deflection, so scaling the axes differently moves which pairs are
        admissible -- it is not a relabelling.
        """
        dirs = primitive_directions(2)
        square = angle_tables_from_profile(
            profile, TowerLattice(cell_size_m=self.SX, directions=dirs))
        skewed = angle_tables_from_profile(
            profile, TowerLattice(cell_size_x_m=self.SX,
                                  cell_size_y_m=self.SY, directions=dirs))
        assert not np.array_equal(square.premium, skewed.premium)
        assert not np.array_equal(square.valid, skewed.valid)

    def test_eur_cost_follows_the_true_step_length(self):
        """The regression this pins: the old isotropic collapse, priced.

        Uniform terrain and ONE direction, so the whole chain is
        arithmetic -- ``value x total length + tower_cost x interior
        towers``, with the length from ``hypot(p * sigma_y, q * sigma_x)``
        and the tower count from how many such steps fit under
        ``max_span_m``. Both numbers move, and in opposite directions.
        """
        n, k = 20, 12
        value, tower, max_span = 10.0, 1000.0, 20.0
        common = dict(
            source_cell=(0, 0), directions=np.array([[1, 1]]),
            tower_cost=np.full((n, n), tower), impassable=None,
            record_pred=False,
            model=TowerFieldModel(min_span_m=0.0, max_span_m=max_span,
                                  charge_terminal_towers=False))
        raster = np.full((n, n), int(value), dtype=np.uint16)

        correct = tower_field_from_raster(
            raster, transform=self._transform(self.SX, self.SY), **common)
        # what the library did before: abs(transform.a) for both axes
        collapsed = tower_field_from_raster(
            raster, cell_size_m=self.SX, **common)

        def expected(sx, sy):
            ds = math.hypot(1 * sy, 1 * sx)          # one (1, 1) step
            m_hi = math.ceil(max_span / ds) - 1      # max_span is exclusive
            return value * k * ds + tower * (math.ceil(k / m_hi) - 1)

        assert correct.arrival[k, k] == pytest.approx(
            expected(self.SX, self.SY))
        assert collapsed.arrival[k, k] == pytest.approx(
            expected(self.SX, self.SX))
        assert correct.arrival[k, k] != pytest.approx(collapsed.arrival[k, k])

    def test_the_real_rasters_two_sizes_are_not_interchangeable(self):
        """The CIRED 2026 cost surface, whose header claims 1 m both ways.

        The error is DIRECTION-DEPENDENT -- 2.3e-5 down a column against
        0 along a row -- so it does not cancel over a route and does not
        look like a scale factor anyone would notice.
        """
        lat = TowerLattice(cell_size_x_m=1.0000017516431916,
                           cell_size_y_m=0.9999783186197908, factor=20,
                           directions=primitive_directions(2))
        rel = {}
        for p, q in ((1, 0), (0, 1), (1, 1), (2, 1)):
            true_m = lat.step_length_m((p, q))
            collapsed_m = math.hypot(p, q) * lat.sigma_x_m
            rel[(p, q)] = (collapsed_m - true_m) / true_m
        assert rel[(0, 1)] == 0.0                 # a column step IS sigma_x
        assert rel[(1, 0)] == pytest.approx(2.343e-5, rel=1e-3)
        assert rel[(1, 1)] == pytest.approx(1.172e-5, rel=1e-3)
        assert rel[(2, 1)] == pytest.approx(1.875e-5, rel=1e-3)


class TestModelAndInputs:
    def test_describe_names_every_choice(self, profile):
        kernel = TowerFieldModel.matching_kernel(profile)
        report = TowerFieldModel.as_reported(profile)
        assert "NOT charged" in kernel.describe()
        assert "280,000" in report.describe()
        assert kernel.effective_last_span_min_m == 0.0
        assert report.charge_terminal_towers

    def test_terminal_towers_are_the_only_difference(self, profile):
        rng = np.random.default_rng(13)
        raster = rng.integers(1, 200, size=(50, 50)).astype(np.uint16)
        common = dict(cell_size_m=10.0, profile=profile, source_cell=(25, 2),
                      factor=1, directions=primitive_directions(2),
                      record_pred=False)
        k = tower_field_from_raster(
            raster, model=TowerFieldModel.matching_kernel(profile), **common)
        r = tower_field_from_raster(
            raster, model=TowerFieldModel.as_reported(profile), **common)
        both = np.isfinite(k.arrival) & np.isfinite(r.arrival)
        # The source is one END, not two, so it carries one terminal
        # tower and not the pair -- exclude it from the identity.
        both[k.source] = False
        delta = r.arrival[both] - k.arrival[both]
        assert np.allclose(delta, 2 * profile.terminal_tower_cost)
        assert delta.mean() >= 560_000      # the plan's number
        assert (r.arrival[k.source] - k.arrival[k.source]
                == pytest.approx(profile.terminal_tower_cost))

    def test_infinite_values_are_refused_with_a_reason(self):
        values = np.ones((10, 10))
        values[3, 3] = np.inf
        with pytest.raises(ValueError, match="finite everywhere"):
            TowerFieldSolver(
                values=values, tower_cost=np.ones((10, 10)),
                lattice=TowerLattice(cell_size_m=10.0),
                model=TowerFieldModel(min_span_m=10.0, max_span_m=50.0))

    def test_no_admissible_span_is_refused_with_a_reason(self):
        with pytest.raises(ValueError, match="no direction admits a span"):
            TowerFieldSolver(
                values=np.ones((10, 10)), tower_cost=np.ones((10, 10)),
                lattice=TowerLattice(cell_size_m=100.0,
                                     directions=primitive_directions(1)),
                model=TowerFieldModel(min_span_m=10.0, max_span_m=50.0))

    def test_blocked_source_is_refused(self):
        blocked = np.zeros((10, 10), bool)
        blocked[5, 5] = True
        solver = TowerFieldSolver(
            values=np.ones((10, 10)), tower_cost=np.ones((10, 10)),
            lattice=TowerLattice(cell_size_m=10.0,
                                 directions=primitive_directions(1)),
            model=TowerFieldModel(min_span_m=10.0, max_span_m=50.0),
            blocked=blocked)
        with pytest.raises(ValueError, match="blocked"):
            solver.solve((5, 5))

    def test_non_primitive_direction_is_refused(self):
        with pytest.raises(ValueError, match="not primitive"):
            TowerLattice(cell_size_m=10.0, directions=np.array([[2, 2]]))

    @pytest.mark.parametrize("how,expected", [
        ("mean", 2.5), ("min", 1.0), ("max", 4.0), ("sample", 4.0)])
    def test_coarsen(self, how, expected):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert coarsen(a, 2, how)[0, 0] == expected

    def test_coarsen_drops_partial_blocks(self):
        assert coarsen(np.ones((5, 5)), 2).shape == (2, 2)

    def test_step_geometry_matches_pyorps(self):
        assert intermediate_offsets(0, 1) == []
        assert intermediate_offsets(1, 1) == [(1, 0), (0, 1)]
        assert cost_factor(0, 1) == pytest.approx(0.5)
        assert cost_factor(1, 1) == pytest.approx(math.sqrt(2) / 4)
        assert cost_factor(1, 2) == pytest.approx(
            math.sqrt(5) / (2 + len(intermediate_offsets(1, 2))))


class TestProfileAdapters:
    def test_angle_tables_are_non_negative_and_limited(self, profile):
        lattice = TowerLattice(cell_size_m=10.0,
                               directions=primitive_directions(2))
        tables = angle_tables_from_profile(profile, lattice)
        assert tables.n_directions == 16
        assert np.all(tables.premium[tables.valid] >= 0)
        assert not tables.valid.all(), "a 40 deg hard limit must forbid some"
        assert np.allclose(tables.straight_premium(), 30000.0)

    def test_negative_premium_is_refused(self):
        with pytest.raises(ValueError, match="negative angle premium"):
            AngleTables(premium=np.array([[-1.0, 0.0], [0.0, 0.0]]),
                        valid=np.ones((2, 2), bool))

    def test_clearance_from_profile(self, profile):
        model = clearance_from_profile(profile)
        assert isinstance(model, ClearanceModel)
        assert model.heights_m == (25.0, 34.0, 42.0)
        assert model.height_premium[0] == 0.0
        assert model.sag_m(300.0) == pytest.approx(
            10.0 * 300 ** 2 / (8 * 20000.0))
        prem = model.premium_for(np.array([10.0, 30.0, 50.0]))
        assert prem[0] == 0.0 and np.isinf(prem[2])

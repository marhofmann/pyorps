"""Phase 0: the tower field against the constrained kernel itself.

``docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md`` puts
this first and says it blocks everything, because until the field and
``ConstrainedPathFinder`` agree on a DEFINED quantity, nothing
downstream means anything.

The reconciliation, term by term against ``_constrained_dijkstra.pyx``:

* the source is seeded in every direction at ``dist = 0`` with
  ``span = 0`` and no tower, and the loop stops the moment the target
  CELL settles in any state -- so neither terminal tower is charged and
  the last span carries no minimum length;
* a tower is placed only where ``cur_span >= min_span`` and a span
  continues only while ``new_span < max_span``, so an interior span
  lies in ``[min_span, max_span)``;
* terrain accrues as
  ``(value[u] + intermediates + value[v]) * cost_factor * cell_size``
  per step, which is ``span_integral="pyorps"``;
* the turn penalty rides on the edge that leaves a tower and the
  tower-type cost sits at that same tower, so together they are one
  premium per direction change -- which is what a tier-2 field charges.

In that matched regime the two agree to the KERNEL's precision, about
1e-8 relative. The residual is not a modelling difference: the field
accumulates in float64 (the plan's section 4.1 requires it -- float32
at 1.3e7 EUR overstates half of all cells by up to 0.50 EUR), while
``_raster_context.pyx`` computes each step's cost factor with
``_get_cost_factor_cython_f32``, so ``sqrt(2)/4`` reaches the kernel's
accumulator already rounded to float32. Against an explicit float64
reference the same solver agrees to 1e-12
(``test_tower_field.py::TestAgainstBruteForce``), which is how the two
statements are told apart. Anything above 1e-7 relative is a defect,
and this file is what says which side it is on.
"""

import numpy as np
import pytest

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.tower_field import (
    TowerFieldModel,
    TowerLattice,
    angle_tables_from_profile,
)
from pyorps.graph.tower_field_oracle import (
    ChainScore,
    compare_with_kernel,
    kernel_objective,
    score_tower_chain,
)
from pyorps.utils.directional import primitive_directions

pytest.importorskip("pyorps.utils._constrained_dijkstra",
                    reason="needs the built Cython constrained kernel")

PROFILE_PATH = "profiles/overhead_line_110kv.yaml"
TOWER_LUT = 100.0 + np.arange(65536, dtype=np.float64) * 2.0


@pytest.fixture(scope="module")
def profile():
    return InfrastructureProfile.load(PROFILE_PATH)


def _raster(seed=11, n=21, wall=False):
    rng = np.random.default_rng(seed)
    r = rng.integers(1, 40, size=(n, n)).astype(np.uint16)
    if wall:
        r[n // 3:2 * n // 3 + 2, n // 2] = 65535
        r[n // 2, n // 2] = 12          # one gap, so a route still exists
    return r


class TestTier1MatchesTheKernel:
    """The headline: same problem, same number."""

    @pytest.mark.parametrize("lo,hi", [(20.0, 45.0), (10.0, 35.0),
                                       (30.0, 80.0)])
    def test_open_terrain_r1(self, lo, hi):
        raster = _raster()
        out = compare_with_kernel(
            raster, source=(10, 1),
            targets=[(10, 19), (3, 17), (18, 14), (10, 10), (0, 20)],
            tower_value_lut=TOWER_LUT, sigma_m=10.0, min_span_m=lo,
            max_span_m=hi, directions=primitive_directions(1))
        for c in out:
            assert np.isfinite(c.oracle_cost), c
            assert abs(c.relative) < 1e-7, c
            # Tower COUNTS are not part of the agreement: distinct
            # chains can tie on cost (measured: 6 towers vs 7 at
            # 3 499.62 EUR either way), and which one a solver returns
            # is a tie-break, not a result. The cost is the result.
            assert c.field_towers >= 0, c

    @pytest.mark.parametrize("dmax", [1, 2])
    def test_with_an_exclusion_wall(self, dmax):
        """The case that caught the corner-cutting defect.

        Before the fix the field undercut the kernel by 25-380 EUR
        wherever a route passed the wall, because a diagonal span was
        allowed to slip between two excluded cells that PYORPS' own
        step model forbids.
        """
        raster = _raster(seed=5, n=23, wall=True)
        out = compare_with_kernel(
            raster, source=(11, 1),
            targets=[(11, 21), (4, 20), (20, 18), (2, 2)],
            tower_value_lut=TOWER_LUT, sigma_m=10.0, min_span_m=20.0,
            max_span_m=45.0, directions=primitive_directions(dmax))
        assert any(np.isfinite(c.oracle_cost) for c in out)
        for c in out:
            if not np.isfinite(c.oracle_cost):
                assert not np.isfinite(c.field_cost), c
                continue
            assert abs(c.relative) < 1e-7, c

    def test_the_heap_reference_agrees_too(self):
        """``force_sparse=1`` pops in true priority order, exact by build."""
        raster = _raster(seed=3, n=17)
        out = compare_with_kernel(
            raster, source=(8, 1), targets=[(8, 15), (2, 13)],
            tower_value_lut=TOWER_LUT, sigma_m=10.0, min_span_m=20.0,
            max_span_m=50.0, directions=primitive_directions(1),
            force_sparse=1)
        for c in out:
            assert abs(c.relative) < 1e-7, c


class TestTier2MatchesTheKernel:
    def test_real_110kv_angle_tables(self, profile):
        raster = _raster(seed=5, n=23, wall=True)
        dirs = primitive_directions(2)
        lattice = TowerLattice(cell_size_m=10.0, factor=1, directions=dirs)
        angles = angle_tables_from_profile(profile, lattice)
        out = compare_with_kernel(
            raster, source=(11, 1), targets=[(4, 20), (20, 18), (2, 2)],
            tower_value_lut=TOWER_LUT, sigma_m=10.0, min_span_m=50.0,
            max_span_m=300.0, directions=dirs, angles=angles)
        assert any(np.isfinite(c.oracle_cost) for c in out)
        for c in out:
            if not np.isfinite(c.oracle_cost):
                continue
            assert abs(c.relative) < 1e-7, c

    def test_the_hard_angle_limit_forbids_the_same_turns(self, profile):
        """A route the 40 deg limit rules out is unreachable on both sides."""
        raster = np.full((15, 15), 10, dtype=np.uint16)
        dirs = primitive_directions(1)          # 45 deg spacing > 40 deg
        lattice = TowerLattice(cell_size_m=10.0, factor=1, directions=dirs)
        angles = angle_tables_from_profile(profile, lattice)
        assert not (angles.valid & ~np.eye(8, dtype=bool)).any(), (
            "with r1 the profile forbids every turn -- that is the point")
        out = compare_with_kernel(
            raster, source=(7, 1), targets=[(3, 12)],
            tower_value_lut=TOWER_LUT, sigma_m=10.0, min_span_m=50.0,
            max_span_m=200.0, directions=dirs, angles=angles)
        assert not np.isfinite(out[0].oracle_cost)
        assert not np.isfinite(out[0].field_cost)


class TestScoreTowerChain:
    """The definition, executable and independent of the solver."""

    def test_reproduces_a_hand_computed_chain(self):
        values = np.full((11, 11), 2.0)
        tower = np.full((11, 11), 1000.0)
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(1))
        model = TowerFieldModel(min_span_m=20.0, max_span_m=60.0,
                                charge_terminal_towers=False)
        chain = [(5, 0), (5, 3), (5, 8)]
        score = score_tower_chain(chain, values=values, tower_cost=tower,
                                  lattice=lattice, model=model)
        assert score.feasible
        assert score.spans_m == (30.0, 50.0)
        assert score.towers == 1000.0            # one interior tower
        assert score.terrain == pytest.approx(2.0 * 80.0)
        assert score.total == pytest.approx(1160.0)

    def test_terminal_towers_show_up_when_charged(self):
        values = np.ones((9, 9))
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(1))
        chain = [(4, 0), (4, 4)]
        kernel = score_tower_chain(
            chain, values=values, tower_cost=np.full((9, 9), 7.0),
            lattice=lattice,
            model=TowerFieldModel(min_span_m=20.0, max_span_m=60.0,
                                  charge_terminal_towers=False))
        reported = score_tower_chain(
            chain, values=values, tower_cost=np.full((9, 9), 7.0),
            lattice=lattice,
            model=TowerFieldModel(min_span_m=20.0, max_span_m=60.0,
                                  charge_terminal_towers=True,
                                  terminal_tower_cost=280_000.0))
        assert reported.total - kernel.total == pytest.approx(560_000.0)

    def test_violations_are_named_not_raised(self):
        values = np.ones((9, 9))
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(1))
        model = TowerFieldModel(min_span_m=50.0, max_span_m=100.0,
                                charge_terminal_towers=False)
        score = score_tower_chain([(4, 0), (4, 1)], values=values,
                                  tower_cost=np.full((9, 9), 1.0),
                                  lattice=lattice, model=model)
        assert isinstance(score, ChainScore)
        assert not score.feasible
        assert "below 50 m" in " ".join(score.violations)

    def test_a_non_lattice_direction_is_named(self):
        values = np.ones((13, 13))
        lattice = TowerLattice(cell_size_m=10.0, factor=1,
                               directions=primitive_directions(1))
        model = TowerFieldModel(min_span_m=10.0, max_span_m=200.0,
                                charge_terminal_towers=False)
        score = score_tower_chain([(0, 0), (2, 5)], values=values,
                                  tower_cost=np.ones((13, 13)),
                                  lattice=lattice, model=model)
        assert "not collinear" in " ".join(score.violations)

    def test_scores_a_field_route_back_to_its_own_value(self, profile):
        """The round trip: field -> towers -> score -> the same number."""
        from pyorps.graph.tower_field import tower_field_from_raster

        raster = _raster(seed=21, n=25)
        dirs = primitive_directions(2)
        lattice = TowerLattice(cell_size_m=10.0, factor=1, directions=dirs)
        model = TowerFieldModel(min_span_m=20.0, max_span_m=60.0,
                                max_span_inclusive=False,
                                last_span_min_m=0.0,
                                charge_terminal_towers=False)
        lut = TOWER_LUT[raster.astype(np.int64)]
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, source_cell=(12, 1), model=model,
            factor=1, directions=dirs, tower_cost=lut, impassable=None)
        for target in [(12, 22), (4, 18), (20, 20)]:
            if not np.isfinite(field.arrival[target]):
                continue
            seq = field.tower_sequence(*target)
            score = score_tower_chain(
                [(t.row, t.col) for t in seq],
                values=raster.astype(np.float64), tower_cost=lut,
                lattice=lattice, model=model)
            assert score.violations == (), (target, score.violations)
            assert score.total == pytest.approx(
                float(field.arrival[target]), rel=1e-9)


class TestKernelWrapper:
    def test_returns_path_towers_and_distance(self):
        raster = _raster(seed=2, n=15)
        path, towers, dist = kernel_objective(
            raster, (7, 1), (7, 13), directions=primitive_directions(1),
            sigma_m=10.0, min_span_m=20.0, max_span_m=50.0,
            tower_value_lut=TOWER_LUT)
        assert len(path) > 2
        assert np.isfinite(dist) and dist > 0
        assert 0 <= len(towers) <= len(path)

    def test_unreachable_target_reports_inf(self):
        raster = np.full((15, 15), 10, dtype=np.uint16)
        raster[:, 7] = 65535                  # a wall with no gap
        path, _towers, dist = kernel_objective(
            raster, (7, 1), (7, 13), directions=primitive_directions(1),
            sigma_m=10.0, min_span_m=20.0, max_span_m=50.0,
            tower_value_lut=TOWER_LUT)
        assert len(path) == 0
        assert not np.isfinite(dist)

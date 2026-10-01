"""Screening bounds: the direction, and the preconditions that flip it.

Revision 3 of the free-siting plan found two bounds pointing the wrong
way and a third whose preconditions were assumed rather than checked.
These tests exist so that cannot happen quietly again: every refusal in
:func:`~pyorps.siting.bounds.assert_overhead_bound_preconditions` has a
test that makes it fire.
"""

import numpy as np
import pytest

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.siting import (
    assert_overhead_bound_preconditions,
    collection_cost_upper_bound,
    lipschitz_stride_gap,
    local_lipschitz_gap,
    min_tower_cost_eur,
    overhead_screen_lower_bound,
)
from pyorps.utils.directional import primitive_directions


@pytest.fixture(scope="module")
def profile():
    return InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")


R1 = primitive_directions(1)
R2 = primitive_directions(2)


class TestMinTowerCost:
    def test_includes_the_terrain_floor(self, profile):
        """65 000, not 30 000.

        The old figure counted only the cheapest tower TYPE and
        discarded the ``350 EUR/m^2 x 100 m^2`` foundation floor that
        ``precompute_tower_terrain_costs`` already applies.
        """
        assert min_tower_cost_eur(profile) == pytest.approx(65_000.0)

    def test_is_read_from_the_lut_not_the_yaml(self, profile):
        """Taking the LUT minimum settles the 'no lower key' caveat.

        ``np.interp`` clamps outside the supplied keys, so a map that
        starts above zero still prices low raster values at its first
        entry -- and the LUT is what the kernel actually reads.
        """
        lut = profile.precompute_tower_terrain_costs()
        types = profile.tower_cost_params["angle_types"]
        cheapest_type = min(t["base_cost"] for t in types.values())
        assert min_tower_cost_eur(profile) == pytest.approx(
            float(lut[:65535].min()) + cheapest_type)


class TestPreconditions:
    def test_accepts_a_nested_neighbourhood(self):
        rec = assert_overhead_bound_preconditions(
            ignore_max_cost=True, field_steps=R2, constrained_steps=R1,
            quantity="kernel")
        assert rec["steps_nested"] and rec["quantity"] == "kernel"

    def test_refuses_ignore_max_cost_false(self):
        with pytest.raises(ValueError, match="ignore_max_cost=True"):
            assert_overhead_bound_preconditions(
                ignore_max_cost=False, field_steps=R2, constrained_steps=R1,
                quantity="kernel")

    def test_refuses_a_coarser_field_neighbourhood(self):
        """An r1 field overstates an r2 route by 1.0797 and breaks it."""
        with pytest.raises(ValueError, match="1.0797"):
            assert_overhead_bound_preconditions(
                ignore_max_cost=True, field_steps=R1, constrained_steps=R2,
                quantity="kernel")

    def test_refuses_an_unnamed_quantity(self):
        with pytest.raises(ValueError, match="560 kEUR"):
            assert_overhead_bound_preconditions(
                ignore_max_cost=True, field_steps=R2, constrained_steps=R1,
                quantity="the cost")


class TestOverheadScreen:
    def test_tower_count_and_terminals(self, profile):
        terrain = np.array([1e5, 2e5, 3e5])
        dist = np.array([100.0, 900.0, 3000.0])
        kernel = overhead_screen_lower_bound(
            terrain, dist, profile=profile, quantity="kernel",
            ignore_max_cost=True, field_steps=R2, constrained_steps=R1)
        # max_span_m is 300: ceil(d/300) - 1 intermediate towers
        assert list(kernel.n_towers) == [0, 2, 9]
        assert np.allclose(kernel.values,
                           terrain + kernel.n_towers * 65_000.0)

        reported = overhead_screen_lower_bound(
            terrain, dist, profile=profile, quantity="reported",
            ignore_max_cost=True, field_steps=R2, constrained_steps=R1)
        assert np.allclose(reported.values - kernel.values, 560_000.0)

    def test_it_really_is_below_the_constrained_cost(self, profile):
        """Against the kernel itself, not against an argument."""
        pytest.importorskip("pyorps.utils._constrained_dijkstra")
        from pyorps.graph.tower_field_oracle import kernel_objective

        rng = np.random.default_rng(3)
        raster = rng.integers(1, 40, size=(31, 31)).astype(np.uint16)
        lut = np.zeros(65536, dtype=np.float64)
        lut[:] = min_tower_cost_eur(profile)
        source, targets = (15, 1), [(15, 29), (4, 24), (27, 20)]
        for tgt in targets:
            _path, _towers, dist = kernel_objective(
                raster, source, tgt, directions=R1, sigma_m=10.0,
                min_span_m=50.0, max_span_m=300.0, tower_value_lut=lut)
            if not np.isfinite(dist):
                continue
            euclid = np.hypot(tgt[0] - source[0], tgt[1] - source[1]) * 10.0
            # terrain floor: the cheapest possible EUR/m over the route
            terrain_floor = float(raster.min()) * euclid
            bound = overhead_screen_lower_bound(
                np.array([terrain_floor]), np.array([euclid]),
                profile=profile, quantity="kernel", ignore_max_cost=True,
                field_steps=R1, constrained_steps=R1,
                min_tower=float(lut[0]))
            assert bound.values[0] <= dist + 1e-6, (tgt, bound.values, dist)

    def test_profile_without_spans_is_refused(self):
        flat = InfrastructureProfile(
            name="cable", description="no towers",
            soft_angle_limit_deg=180.0, hard_angle_limit_deg=180.0,
            angle_cost_function="linear")
        with pytest.raises(ValueError, match="no max_span_m"):
            overhead_screen_lower_bound(
                np.array([1.0]), np.array([1.0]), profile=flat,
                quantity="kernel", ignore_max_cost=True,
                field_steps=R2, constrained_steps=R1)


class TestStride:
    def test_gap_scales_with_stride_and_value(self):
        assert lipschitz_stride_gap(5.0, 300.0) == pytest.approx(
            300.0 * 5.0 * np.sqrt(2) / 2)
        assert lipschitz_stride_gap(10.0, 300.0) == pytest.approx(
            2 * lipschitz_stride_gap(5.0, 300.0))
        assert lipschitz_stride_gap(5.0, 300.0, diagonal=False) < \
            lipschitz_stride_gap(5.0, 300.0)

    def test_a_one_cell_stride_still_has_a_gap(self):
        """The stride bound is never zero, which is the point of it."""
        assert lipschitz_stride_gap(1.0, 300.0) > 0

    def test_local_constant_is_at_most_the_global_one(self):
        rng = np.random.default_rng(5)
        values = rng.integers(1, 200, size=(60, 60)).astype(np.float64)
        rows = np.array([10, 30, 50])
        cols = np.array([10, 30, 50])
        local = local_lipschitz_gap(values, rows, cols, stride_m=25.0,
                                    resolution_m=5.0)
        glob = lipschitz_stride_gap(25.0, float(values.max()))
        assert np.all(local <= glob + 1e-9)
        assert np.all(local > 0)

    def test_excluded_cells_do_not_set_the_constant(self):
        """A facility cannot move onto an excluded cell, so 65535 is not
        a Lipschitz constant for anything."""
        values = np.full((40, 40), 10.0)
        values[20, 20] = 65535.0
        gap = local_lipschitz_gap(values, np.array([20]), np.array([20]),
                                  stride_m=10.0, resolution_m=5.0)
        assert gap[0] == pytest.approx(
            lipschitz_stride_gap(10.0, 10.0))


class TestCollectionCost:
    def test_is_named_an_upper_bound(self):
        """Shared trenching only reduces cost, so the radial sum is above
        the optimum and pruning with it prunes too much."""
        d = np.array([[10.0, 20.0], [30.0, 40.0], [1.0, 2.0]])
        assert np.array_equal(collection_cost_upper_bound(d),
                              np.array([41.0, 62.0]))
        assert "UPPER bound" in collection_cost_upper_bound.__doc__

    def test_shape_is_checked(self):
        with pytest.raises(ValueError, match="n_sources"):
            collection_cost_upper_bound(np.array([1.0, 2.0]))

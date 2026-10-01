"""The export contract: what a MILP reads, and what it can rely on.

Section 5 of the tower-field plan. Both ends of every cost are present
and ``cost_lb <= cost_ub`` is checked before anything is written -- a
single number would force the reader to guess whether it may prune with
it, and the plan's review found that guess made wrongly in both
directions.
"""

import numpy as np
import pytest

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.tower_field import tower_field_bounds
from pyorps.siting import CandidateSet, LinkCosts, SitingExport
from pyorps.siting.export import export_tower_fields
from pyorps.utils.directional import primitive_directions


@pytest.fixture(scope="module")
def profile():
    return InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")


@pytest.fixture
def affine():
    from affine import Affine
    return Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5500000.0)


@pytest.fixture
def candidates(affine):
    rows = np.array([10, 20, 30, 40])
    cols = np.array([12, 24, 36, 48])
    xs = affine.c + (cols + 0.5) * affine.a
    ys = affine.f + (rows + 0.5) * affine.e
    return CandidateSet(rows=rows, cols=cols, xs=xs, ys=ys,
                        transform=affine, crs="EPSG:25832", stride_m=10.0,
                        build_cost_eur=np.array([1e5, 2e5, np.nan, 4e5]),
                        best_theta_deg=np.array([0.0, 30.0, 60.0, 90.0]))


class TestLinkCosts:
    def test_key_and_violations(self):
        link = LinkCosts("PCC0", "overhead",
                         cost_lb=np.array([1.0, 2.0, np.inf]),
                         cost_ub=np.array([3.0, 4.0, np.inf]))
        assert link.key == "PCC0::overhead"
        assert link.violations() == 0
        bad = LinkCosts("PCC0", "cable", cost_lb=np.array([5.0]),
                        cost_ub=np.array([1.0]))
        assert bad.violations() == 1

    def test_unknown_link_type_refused(self):
        with pytest.raises(ValueError, match="link_type"):
            LinkCosts("t", "microwave", cost_lb=np.zeros(1),
                      cost_ub=np.zeros(1))

    def test_towers_are_meaningless_on_a_cable(self):
        with pytest.raises(ValueError, match="cable"):
            LinkCosts("t", "cable", cost_lb=np.zeros(1), cost_ub=np.zeros(1),
                      n_towers=np.zeros(1))


class TestSitingExport:
    def test_check_catches_an_inverted_bound(self, candidates):
        exp = SitingExport(candidates)
        exp.add_link(LinkCosts("PCC0", "cable",
                               cost_lb=np.array([5.0, 1.0, 1.0, 1.0]),
                               cost_ub=np.array([1.0, 2.0, 3.0, 4.0])))
        with pytest.raises(AssertionError, match="cost_lb exceeds cost_ub"):
            exp.check()

    def test_check_catches_reachable_only_above(self, candidates):
        exp = SitingExport(candidates)
        exp.add_link(LinkCosts("PCC0", "cable",
                               cost_lb=np.array([np.inf, 1.0, 1.0, 1.0]),
                               cost_ub=np.array([9.0, 2.0, 3.0, 4.0])))
        with pytest.raises(AssertionError, match="no relaxation can do"):
            exp.check()

    def test_duplicate_block_refused(self, candidates):
        exp = SitingExport(candidates)
        link = LinkCosts("PCC0", "cable", cost_lb=np.zeros(4),
                         cost_ub=np.ones(4))
        exp.add_link(link)
        with pytest.raises(ValueError, match="added twice"):
            exp.add_link(LinkCosts("PCC0", "cable", cost_lb=np.zeros(4),
                                   cost_ub=np.ones(4)))

    def test_length_mismatch_refused(self, candidates):
        exp = SitingExport(candidates)
        with pytest.raises(ValueError, match="expected 4 costs"):
            exp.add_link(LinkCosts("PCC0", "cable", cost_lb=np.zeros(3),
                                   cost_ub=np.ones(3)))

    def test_npz_round_trip(self, candidates, tmp_path):
        import json

        exp = SitingExport(candidates)
        exp.add_link(LinkCosts(
            "PCC0", "overhead", cost_lb=np.array([1.0, 2.0, 3.0, 4.0]),
            cost_ub=np.array([1.5, 2.5, 3.5, 4.5]),
            n_towers=np.array([2, 3, 4, 5]),
            length_m=np.array([100.0, 200.0, 300.0, 400.0])))
        exp.add_link(LinkCosts(
            "MV7", "cable", cost_lb=np.zeros(4), cost_ub=np.ones(4)))
        dest = exp.write_npz(tmp_path / "cands")
        assert dest.suffix == ".npz"

        data = np.load(dest, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        assert meta["n_candidates"] == 4
        assert meta["links"] == ["MV7::cable", "PCC0::overhead"]
        assert data["PCC0::overhead::n_towers"].dtype == np.uint16
        assert data["PCC0::overhead::length_m"].dtype == np.float32
        assert np.array_equal(data["candidate_id"], np.arange(4))
        # a candidate the screen could not place is not buildable
        assert list(data["buildable"]) == [True, True, False, True]
        assert "check" in meta

    def test_dataframe_has_one_row_per_candidate(self, candidates):
        pytest.importorskip("pandas")
        exp = SitingExport(candidates)
        exp.add_link(LinkCosts("PCC0", "cable", cost_lb=np.zeros(4),
                               cost_ub=np.ones(4)))
        df = exp.to_dataframe()
        assert len(df) == 4
        assert "PCC0::cable::cost_lb" in df.columns


class TestExportTowerFields:
    def test_end_to_end_from_bound_pairs(self, profile, candidates, affine,
                                         tmp_path):
        rng = np.random.default_rng(2)
        raster = rng.integers(1, 200, size=(60, 60)).astype(np.uint16)
        lower, upper = tower_field_bounds(
            raster, cell_size_m=10.0, profile=profile, source_cell=(30, 2),
            factor=1, directions=primitive_directions(2), transform=affine,
            record_pred=True)
        exp = export_tower_fields(candidates, {"PCC0": (lower, upper)},
                                  with_geometry=True)
        report = exp.check()
        assert "PCC0::overhead" in report
        link = exp.links["PCC0::overhead"]
        assert link.n_towers is not None and link.length_m is not None
        reach = np.isfinite(link.cost_ub)
        assert np.all(link.n_towers[reach] >= 2)     # two terminals at least
        assert np.all(link.length_m[reach] > 0)
        assert "tower-chain cost" in link.meta["lower_model"]
        dest = exp.write_npz(tmp_path / "out.npz")
        assert dest.exists()

    def test_a_single_field_serves_both_ends(self, profile, candidates,
                                             affine):
        """Allowed, and only correct when the field is exact -- so the
        metadata has to say which model produced each end."""
        rng = np.random.default_rng(3)
        raster = rng.integers(1, 200, size=(50, 50)).astype(np.uint16)
        from pyorps.graph.tower_field import tower_field_from_raster
        field = tower_field_from_raster(
            raster, cell_size_m=10.0, profile=profile, source_cell=(25, 2),
            factor=1, directions=primitive_directions(2), transform=affine,
            record_pred=False)
        exp = export_tower_fields(candidates, {"PCC0": field})
        link = exp.links["PCC0::overhead"]
        assert np.array_equal(link.cost_lb, link.cost_ub)
        assert link.meta["lower_model"] == link.meta["upper_model"]

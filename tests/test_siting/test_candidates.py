"""Candidate lattices, and the cell-selection rule pinned in place.

The substation study's ``_sample_field`` used ``np.round`` where every
library path uses rasterio's ``rowcol`` -- which floors. Candidates sit
at pixel centres, so the two rules disagree for essentially all of them,
and the study's winning margin was 28.14 EUR. That is why the migration
of that driver onto this code cannot be bit-for-bit reproducible, and
why the RULE is what gets pinned here rather than any particular
outcome.
"""

import numpy as np
import pytest

from pyorps.siting import (
    CandidateSet,
    Footprint,
    candidate_lattice,
    cell_xy,
    sample_field,
    screen_footprints,
)

pytest.importorskip("scipy")


@pytest.fixture
def affine():
    from affine import Affine
    return Affine(5.0, 0.0, 400000.0, 0.0, -5.0, 5500000.0)


@pytest.fixture
def screen(affine):
    rng = np.random.default_rng(4)
    n = 100
    cost = rng.random((n, n)).astype(np.float32) * 50 + 10
    blocked = np.zeros((n, n), np.float32)
    blocked[60:80, 20:50] = 1.0
    return screen_footprints(cost, blocked,
                             footprint=Footprint(40.0, 20.0,
                                                 rotation_step_deg=30.0),
                             transform=affine, resolution_m=5.0)


class TestCellRule:
    def test_floor_is_the_default_and_matches_rowcol(self, affine):
        rasterio = pytest.importorskip("rasterio")
        field = np.arange(100 * 100, dtype=np.float64).reshape(100, 100)
        rows = np.array([3, 17, 42])
        cols = np.array([5, 11, 88])
        xs, ys = cell_xy(affine, rows, cols)
        got = sample_field(field, affine, xs, ys)
        assert np.array_equal(got, field[rows, cols])
        rr, cc = rasterio.transform.rowcol(affine, xs, ys)
        assert np.array_equal(np.asarray(rr), rows)
        assert np.array_equal(np.asarray(cc), cols)

    def test_round_disagrees_at_pixel_centres(self, affine):
        """The study's rule, kept reachable and kept labelled."""
        field = np.arange(100 * 100, dtype=np.float64).reshape(100, 100)
        xs, ys = cell_xy(affine, np.array([3, 17]), np.array([5, 11]))
        floor = sample_field(field, affine, xs, ys, rule="floor")
        rounded = sample_field(field, affine, xs, ys, rule="round")
        assert not np.array_equal(floor, rounded)

    def test_unknown_rule_refused(self, affine):
        with pytest.raises(ValueError, match="floor"):
            sample_field(np.ones((4, 4)), affine, [0.0], [0.0], rule="ceil")

    def test_outside_is_inf_not_clamped(self, affine):
        field = np.ones((10, 10))
        far = sample_field(field, affine, [affine.c - 1000.0], [affine.f])
        assert np.isinf(far[0])
        clipped = sample_field(field, affine, [affine.c - 1000.0],
                               [affine.f], clip=True)
        assert clipped[0] == 1.0


class TestLattice:
    def test_stride_controls_the_count(self, screen):
        few = candidate_lattice(screen, stride_m=50.0)
        many = candidate_lattice(screen, stride_m=10.0)
        assert 0 < len(few) < len(many)
        assert few.stride_m == 50.0

    def test_every_candidate_is_feasible(self, screen):
        cands = candidate_lattice(screen, stride_m=25.0)
        assert len(cands)
        assert screen.feasible[cands.rows, cands.cols].all()
        assert np.isfinite(cands.build_cost_eur).all()

    def test_erosion_keeps_the_footprint_inside(self, screen):
        """A candidate centre must not place the rectangle partly outside
        the region the screen verified."""
        eroded = candidate_lattice(screen, stride_m=15.0, erode=True)
        raw = candidate_lattice(screen, stride_m=15.0, erode=False)
        assert len(eroded) < len(raw)
        px = round(screen.footprint.half_diagonal_m / screen.resolution_m)
        for r, c in zip(eroded.rows, eroded.cols):
            block = screen.feasible[max(0, r - px):r + px + 1,
                                    max(0, c - px):c + px + 1]
            assert block.all()

    def test_coordinates_round_trip_to_their_own_cells(self, screen):
        cands = candidate_lattice(screen, stride_m=20.0)
        field = np.arange(screen.shape[0] * screen.shape[1],
                          dtype=np.float64).reshape(screen.shape)
        got = sample_field(field, screen.transform, cands.xs, cands.ys)
        assert np.array_equal(got, field[cands.rows, cands.cols])

    def test_metadata_records_the_rule(self, screen):
        cands = candidate_lattice(screen, stride_m=20.0)
        assert cands.meta["cell_rule"] == "floor"
        assert cands.meta["eroded"] is True
        assert cands.meta["erosion_m"] == pytest.approx(
            screen.footprint.half_diagonal_m)


class TestCandidateSet:
    def test_ids_points_and_subset(self, screen):
        cands = candidate_lattice(screen, stride_m=25.0)
        assert np.array_equal(cands.ids, np.arange(len(cands)))
        assert cands.points.shape == (len(cands), 2)
        keep = cands.build_cost_eur < np.median(cands.build_cost_eur)
        half = cands.subset(keep)
        assert len(half) == int(keep.sum())
        assert np.array_equal(half.rows, cands.rows[keep])
        assert np.array_equal(half.build_cost_eur,
                              cands.build_cost_eur[keep])

    def test_empty_set_is_representable(self):
        empty = CandidateSet(rows=np.array([], np.int64),
                             cols=np.array([], np.int64),
                             xs=np.array([]), ys=np.array([]))
        assert len(empty) == 0
        assert empty.points.shape == (0, 2)

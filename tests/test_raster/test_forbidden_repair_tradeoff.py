"""The sub-cell forbidden default, and the two ways detection used to be blind.

Three properties are pinned here, in the order they hang off each other:

1. THE DEFAULT CHANGES NOTHING. ``rasterize()`` and ``rasterize_metrics()``
   default to ``all_touched=False, widen_thin_forbidden=False``, and the burned
   raster is BIT-IDENTICAL to a plain GDAL burn - the array the code produced
   before any of the thinness work existed. The default may WARN; it may not
   move a cell. The reason is that neither repair is safe: at a fixed cell size
   a sub-cell BARRIER and a sub-cell GAP are the same geometry seen from
   opposite sides, so every repair that makes barriers survive also closes
   gates. Swept over 8 gate widths x 8 sub-pixel offsets under four gate-width
   sets (``sweep_forbidden_repair_tradeoff.py`` next to this file reproduces
   the table), counting only the configurations the plain rule leaves passable:
   on a FAT 2.0 m wall C seals 0 % under every set and D 17-100 %; on a THIN
   0.4 m wall C seals 38-100 % and D 31-100 %, i.e. they are indistinguishable
   there. A bare percentage is a property of the gate widths swept, not of the
   repair, so only that structure is asserted anywhere.

2. DETECTION SEES THE HARM A REPAIR CAUSES, AND ONLY THE REAL HARM. Option A
   inspects the REPAIRED burn, which is defect-free by construction, so it can
   only ever confirm that a repair worked. Measured: zero warnings on a fixture
   where widening sealed a 1.20 m gate and produced ``NoPathFoundError``. The
   free-space connectivity check is the missing half - and it has to compare
   against a REAL plain burn, because ``fill_value`` defaults to
   ``IMPASSABLE_CELL_COST`` and makes NODATA byte-identical to forbidden
   (``TestNodataIsNotFreeSpace``).

3. PARTIAL-VANISH RECALL DOES NOT DEPEND ON SUB-PIXEL LUCK. A thin part used to
   count as present when ANY cell of its all-touched footprint carried a
   forbidden burn - including a cell burned by a completely different part of
   the same feature. The same nodata conflation applies to its second step.
"""
import warnings

import geopandas as gpd
import numpy as np
import pytest
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import box

from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.raster.thinness import (
    SealedOpeningError,
    SealedOpeningWarning,
    ThinForbiddenFeatureWarning,
    detect_forbidden_burn_defects,
    detect_repair_seals,
)

RES = 1.0

# EPSG:25832 magnitudes: the transform coefficients are ~1e6, so a burn that
# is only "nearly" identical shows up here and not on a toy grid at the origin.
X0, Y0 = 350000.0, 5600000.0
N = 64
BBOX = box(X0, Y0, X0 + N * RES, Y0 + N * RES)
CRS = "EPSG:25832"

FREE, CHEAP, PRICEY = 10, 40, 120
COSTS = {'category': {'free': FREE, 'cheap': CHEAP, 'pricey': PRICEY,
                      'barrier': IMPASSABLE_CELL_COST}}


def _rasterizer(features):
    gdf = gpd.GeoDataFrame({'category': [c for _, c in features]},
                           geometry=[g for g, _ in features], crs=CRS)
    return GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)


def burn(features, **kwargs):
    """Rasterize ``[(geometry, category), ...]`` through the real API."""
    rasterizer = _rasterizer(features)
    rasterizer.rasterize(resolution_in_m=RES, bounding_box=BBOX, **kwargs)
    return rasterizer.raster


def plain_gdal_burn(features):
    """The reference: rasterio's own scan conversion, no pyorps in the way.

    Deliberately NOT ``rasterize(widen_thin_forbidden=False)`` - that would
    only prove the default equals another pyorps code path. This reproduces
    the contract the burn has always had (one pass over the ascending-cost
    sequence, so the most expensive feature wins an overlap) with rasterio
    called directly.
    """
    values = [COSTS['category'][c] for _, c in features]
    order = np.argsort(np.asarray(values), kind='stable')
    transform = from_bounds(*BBOX.bounds, N, N)
    return rio_rasterize(
        ((features[i][0], values[i]) for i in order),
        out_shape=(N, N),
        fill=IMPASSABLE_CELL_COST,
        dtype="uint16",
        transform=transform,
    )


BACKGROUND = (box(X0, Y0, X0 + N, Y0 + N), 'free')

#: Ordinary classes, a FAT barrier and a sub-cell one, so that the default has
#: something to warn about while it must still not move a single cell.
MIXED = [
    BACKGROUND,
    (box(X0 + 5, Y0 + 5, X0 + 25, Y0 + 25), 'cheap'),
    (box(X0 + 20, Y0 + 20, X0 + 45, Y0 + 40), 'pricey'),
    (box(X0 + 50, Y0 + 2, X0 + 54, Y0 + 62), 'barrier'),        # fat
    (box(X0 + 10.05, Y0 + 30, X0 + 10.45, Y0 + 60), 'barrier'),  # 0.4 m
]


class TestTheDefaultChangesNothing:
    """The bar everything else hangs off: the default is a plain GDAL burn."""

    def test_the_default_is_bit_identical_to_a_plain_burn(self):
        with warnings.catch_warnings():
            # The fixture carries a sub-cell barrier on purpose, so option A
            # is EXPECTED to fire. Warning is exactly what a default may do.
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            default = burn(MIXED)
        reference = plain_gdal_burn(MIXED)
        assert default.shape == reference.shape
        assert np.array_equal(default, reference), (
            f"the default moved {int((default != reference).sum())} cells; it "
            f"may warn, it may not change the raster")

    def test_the_default_really_had_something_to_warn_about(self):
        """Guard: without a defect the identity above would be vacuous."""
        with pytest.warns(ThinForbiddenFeatureWarning):
            burn(MIXED)

    def test_the_old_default_was_not_bit_identical(self):
        """Why this test exists: widening WAS the default and it moved cells."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widened = burn(MIXED, widen_thin_forbidden=True)
        assert not np.array_equal(widened, plain_gdal_burn(MIXED)), (
            "widening no longer changes this fixture, so it no longer shows "
            "what the default had to be rescued from")

    def test_the_metric_stack_default_is_the_plain_burn_too(self):
        def stack(**kwargs):
            rasterizer = _rasterizer(MIXED)
            return rasterizer.rasterize_metrics(resolution_in_m=RES, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            default = stack()
            explicit = stack(all_touched=False, widen_thin_forbidden=False,
                             on_thin_features="ignore")
        assert np.array_equal(default.forbidden_mask,
                              explicit.forbidden_mask)
        assert np.array_equal(default['cost'], explicit['cost'])

    def test_no_repair_means_no_seal_warning(self):
        """With nothing repaired there is nothing that could have sealed."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", SealedOpeningWarning)
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            burn(MIXED)


# ==========================================================================
# 2 - the seal check
# ==========================================================================

MID = Y0 + N / 2.0


def _thin_wall_with_gate(thickness=0.4, gate_m=1.20):
    """A forbidden wall spanning the extent, with a legitimate opening.

    At 0.4 m the wall is thinner than a cell, so option C widens it - by
    ``(sqrt(2) - 0.4) / 2`` on every side, which advances both lips of the gate
    by half a metre and closes a 1.20 m opening the raster could perfectly well
    have represented.
    """
    x0, x1 = X0 + 30.0, X0 + 30.0 + thickness
    half = gate_m / 2.0
    return box(x0, Y0, x1, MID - half).union(box(x0, MID + half, x1, Y0 + N))


def _left_right_connected(grid):
    """Is any left-edge cell 4-connected to the right edge through free space?

    4-, not 8-, because pyorps admits a diagonal step only when BOTH flanking
    cells are passable - so every move it can make is also a 4-path.
    """
    from scipy import ndimage
    free = grid != IMPASSABLE_CELL_COST
    labels, _ = ndimage.label(free, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    left = set(labels[:, 0].tolist()) - {0}
    right = set(labels[:, -1].tolist()) - {0}
    return bool(left & right)


class TestRepairSealsAreDetected:
    """DEFECT: option A inspects the repaired burn and is blind by design."""

    WALL = [BACKGROUND, (_thin_wall_with_gate(), 'barrier')]

    def test_the_plain_rule_leaves_the_gate_open(self):
        """Guard: the gate must be walkable before the repair, or nothing is
        being sealed and the tests below prove nothing."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            grid = burn(self.WALL)
        assert _left_right_connected(grid)

    def test_widening_the_wall_seals_the_gate_and_warns(self):
        with pytest.warns(SealedOpeningWarning, match="SEALED free space"):
            grid = burn(self.WALL, widen_thin_forbidden=True)
        assert not _left_right_connected(grid), (
            "the fixture no longer seals, so the warning above is not about "
            "the harm it was written for")

    def test_option_a_alone_stays_silent_on_it(self):
        """The blindness itself: the repaired burn has no defect to find.

        Kept as a test because it is the reason the connectivity check exists -
        if option A ever started catching this, the check could be revisited.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            warnings.simplefilter("ignore", SealedOpeningWarning)
            burn(self.WALL, widen_thin_forbidden=True)

    def test_the_overlay_seals_it_too_and_warns(self):
        with pytest.warns(SealedOpeningWarning):
            grid = burn(self.WALL, all_touched=True)
        assert not _left_right_connected(grid)

    def test_both_repairs_at_once_still_warn(self):
        """The true positive on the branch that cannot use the snapshot."""
        with pytest.warns(SealedOpeningWarning, match="SEALED free space"):
            grid = burn(self.WALL, all_touched=True,
                        widen_thin_forbidden=True)
        assert not _left_right_connected(grid)

    def test_raise_mode_turns_the_seal_into_an_error(self):
        with pytest.raises(SealedOpeningError):
            burn(self.WALL, widen_thin_forbidden=True,
                 on_thin_features="raise")

    def test_ignore_mode_silences_it(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            burn(self.WALL, widen_thin_forbidden=True,
                 on_thin_features="ignore")

    def test_a_repair_that_only_shrinks_free_space_does_not_warn(self):
        """The false positive the test had to be chosen to avoid.

        A 0.4 m stub in open ground is widened to sqrt(2) m: free space loses
        cells and one region gets smaller, but it stays ONE region. Only a
        SPLIT is a seal.
        """
        stub = box(X0 + 20.05, Y0 + 20.0, X0 + 20.45, Y0 + 30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SealedOpeningWarning)
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            grid = burn([BACKGROUND, (stub, 'barrier')],
                        widen_thin_forbidden=True)
        assert (grid == IMPASSABLE_CELL_COST).any(), (
            "the stub burned nothing, so no cell was lost and the test is "
            "vacuous")
        assert _left_right_connected(grid)

    def test_the_report_counts_regions_not_cells(self):
        """The public helper, straight on a finished raster."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rasterizer = _rasterizer(self.WALL)
            rasterizer.rasterize(resolution_in_m=RES, bounding_box=BBOX,
                                 widen_thin_forbidden=True)
        transform = from_bounds(*BBOX.bounds, N, N)
        report = detect_repair_seals(
            rasterizer.raster, [_thin_wall_with_gate()],
            [_thin_wall_with_gate().buffer(0.51)], (N, N), transform,
            other_geometries=[BACKGROUND[0]])
        assert report.n_split_regions == 1
        assert report.n_free_regions_before == 1
        assert report.n_free_regions_after == 2
        assert report.cells_lost > 0
        assert report.example_cell is not None
        assert not report.ok
        assert "SEALED free space" in report.summary()

    def test_the_plain_burn_must_be_stated_not_guessed(self):
        """Without evidence of the plain burn the check refuses to run.

        The alternative is what it used to do - infer the plain burn from the
        finished raster - and that inference is unsound (see
        ``TestNodataIsNotFreeSpace``). Guessing silently is worse than raising.
        """
        transform = from_bounds(*BBOX.bounds, N, N)
        with pytest.raises(ValueError, match="needs the PLAIN burn"):
            detect_repair_seals(
                np.full((N, N), IMPASSABLE_CELL_COST, dtype="uint16"),
                [_thin_wall_with_gate()],
                [_thin_wall_with_gate().buffer(0.51)], (N, N), transform)


# ==========================================================================
# 2b - the false positive: NODATA is not free space
# ==========================================================================

#: Two 'free' parcels with a 1 m strip between them that NO feature covers, and
#: a 0.4 m forbidden bar lying across that strip.
#:
#: With the default ``fill_value=IMPASSABLE_CELL_COST`` the strip burns 65535 -
#: the very value a forbidden feature burns - so in the finished raster a
#: NODATA cell and a forbidden cell are the same number. The plain burn has TWO
#: free regions and widening the bar cannot join or split them: it eats a few
#: cells off each parcel and leaves both connected.
#:
#: The bar is placed so that widening covers cell (33, 32), which is nodata. The
#: old reconstruction
#:
#:     other = impassable_after & ~forbidden_after
#:     passable_before = ~(other | forbidden_before)
#:
#: drops that cell from ``other`` (the widened footprint covers it) and finds it
#: absent from ``forbidden_before`` (the 0.4 m bar burns nothing), so it
#: reconstructs it as PASSABLE BEFORE - a phantom bridge across the strip. One
#: before-region, two after-regions, one reported "split" that never happened.
NODATA_STRIP = [
    (box(X0, Y0, X0 + 32.0, Y0 + 64.0), 'free'),          # cols 0..31
    (box(X0 + 33.0, Y0, X0 + 64.0, Y0 + 64.0), 'free'),   # cols 33..63
    (box(X0 + 30.0, Y0 + 30.05, X0 + 36.0, Y0 + 30.45), 'barrier'),  # 0.4 m
]


def _free_regions(grid):
    from scipy import ndimage
    _, n = ndimage.label(grid != IMPASSABLE_CELL_COST, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    return n


class TestNodataIsNotFreeSpace:
    """A repair that touches NODATA must not be reported as a seal."""

    def test_the_fixture_really_has_nodata_next_to_the_repair(self):
        """Guard: column 32 is covered by no feature at all, and the widened
        bar reaches across it. Without both, the fixture proves nothing."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = burn(NODATA_STRIP)
            widened = burn(NODATA_STRIP, widen_thin_forbidden=True)
        assert (plain[:, 32] == IMPASSABLE_CELL_COST).all(), "no nodata strip"
        assert plain[33, 31] != IMPASSABLE_CELL_COST
        assert (widened[33, 29:37] == IMPASSABLE_CELL_COST).all(), (
            "the widened bar no longer spans the strip")

    def test_the_plain_burn_already_had_two_free_regions(self):
        """Guard: the two parcels were NEVER connected, so nothing can seal."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = burn(NODATA_STRIP)
        assert _free_regions(plain) == 2

    def test_the_repair_splits_nothing(self):
        """Ground truth: two regions before, two after, both still whole."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widened = burn(NODATA_STRIP, widen_thin_forbidden=True)
        assert _free_regions(widened) == 2

    def test_the_detector_stays_silent(self):
        """THE REGRESSION. This warned before the plain burn was made real."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", SealedOpeningWarning)
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            burn(NODATA_STRIP, widen_thin_forbidden=True)

    def test_the_overlay_is_silent_on_it_too(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SealedOpeningWarning)
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            burn(NODATA_STRIP, all_touched=True)

    def test_both_repairs_at_once_are_silent_on_it(self):
        """The awkward branch. With C on as well, the raster the overlay paints
        over is ALREADY widened, so the pre-overlay snapshot is not the plain
        burn and the check must fall through to the re-burn instead."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", SealedOpeningWarning)
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            burn(NODATA_STRIP, all_touched=True, widen_thin_forbidden=True)

    def test_the_report_agrees_with_the_ground_truth(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rasterizer = _rasterizer(NODATA_STRIP)
            rasterizer.rasterize(resolution_in_m=RES, bounding_box=BBOX,
                                 widen_thin_forbidden=True)
        transform = from_bounds(*BBOX.bounds, N, N)
        report = detect_repair_seals(
            rasterizer.raster, [NODATA_STRIP[2][0]],
            [NODATA_STRIP[2][0].buffer(0.51)], (N, N), transform,
            other_geometries=[g for g, c in NODATA_STRIP if c != 'barrier'])
        assert report.n_free_regions_before == 2
        assert report.n_free_regions_after == 2
        assert report.n_split_regions == 0
        assert report.ok

    def test_the_old_reconstruction_did_fire_on_it(self):
        """Pins the MECHANISM, so the shortcut cannot quietly come back.

        Reproduces the withdrawn expression verbatim on this fixture and
        asserts it invents the bridge. If a future change makes this stop
        firing, the fixture no longer guards what it was written for.
        """
        from pyorps.raster.thinness import (
            _burn_forbidden_id_band,
            _split_free_regions,
            widen_thin_features,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widened = burn(NODATA_STRIP, widen_thin_forbidden=True)
        transform = from_bounds(*BBOX.bounds, N, N)
        plain_geom = np.array([NODATA_STRIP[2][0]], dtype=object)
        repaired_geom = widen_thin_features(plain_geom, RES)

        forbidden_after = _burn_forbidden_id_band(
            repaired_geom, (N, N), transform) != 0
        forbidden_before = _burn_forbidden_id_band(
            plain_geom, (N, N), transform) != 0
        impassable_after = widened == IMPASSABLE_CELL_COST
        other = impassable_after & ~forbidden_after
        reconstructed = ~(other | forbidden_before)

        n_before, n_after, split, _ = _split_free_regions(
            reconstructed, ~impassable_after)
        assert n_before == 1, (
            "the reconstruction no longer bridges the nodata strip, so this "
            "fixture no longer reproduces the false positive")
        assert n_after == 2
        assert split.size == 1


# ==========================================================================
# 3 - partial-vanish recall
# ==========================================================================

#: A toy grid at the origin for the recall fixtures: the arm's cell arithmetic
#: is the point of the fixture and reads better without a 5.6e6 offset.
N64 = 64
TRANSFORM64 = from_origin(0.0, float(N64), RES, RES)


def _fat_base_with_long_thin_arm():
    """A fat base whose burn OVERLAPS the arm's all-touched footprint.

    The base is 2 m x 40.6 m, so it burns rows 23..63 of columns 30 and 31.
    The arm is 0.4 m x 23.4 m: it covers no cell centre of its own (the centres
    of column 30 sit at x = 30.5, outside [30.55, 30.95]), and its all-touched
    footprint is 24 cells - rows 0..23 of column 30. Row 23 is the one cell the
    BASE already burned, and under the old rule that single cell of incidental
    overlap excused the whole arm.
    """
    return box(30.0, 0.0, 32.0, 40.6).union(box(30.55, 40.6, 30.95, 64.0))


def _report(geometries):
    return detect_forbidden_burn_defects(geometries, (N64, N64), TRANSFORM64,
                                         resolution_in_m=RES)


class TestPartialVanishRecall:

    def test_the_fixture_is_the_measured_one(self):
        """Guard: 24 touched cells, exactly one of them burned by the base."""
        from pyorps.raster.thinness import _burn_alone, _burn_forbidden_id_band

        base = box(30.0, 0.0, 32.0, 40.6)
        arm = box(30.55, 40.6, 30.95, 64.0)
        assert arm.bounds[3] - arm.bounds[1] == pytest.approx(23.4)
        window = (0, N64, 0, N64)
        touched = _burn_alone(arm, window, TRANSFORM64, all_touched=True)
        own = _burn_alone(arm, window, TRANSFORM64, all_touched=False)
        burned = _burn_forbidden_id_band([base], (N64, N64), TRANSFORM64) != 0
        assert int(touched.sum()) == 24
        assert int(own.sum()) == 0, "the arm must burn nothing of its own"
        assert int((touched & burned).sum()) == 1, (
            "the incidental overlap with the base is what the old rule fell "
            "for; without it this fixture guards nothing")

    def test_the_arm_is_reported(self):
        report = _report([_fat_base_with_long_thin_arm()])
        assert list(report.partially_vanished) == [0]
        assert not report.ok
        assert "burned only in part" in report.summary()

    def test_the_older_signals_stay_silent(self):
        """It burns cells and they form one block - only the part test sees it."""
        report = _report([_fat_base_with_long_thin_arm()])
        assert report.vanished.size == 0
        assert report.fragmented.size == 0

    def test_forbidden_over_forbidden_is_not_a_defect(self):
        """The same arm, painted over by a LATER forbidden feature.

        Every cell the arm reaches is impassable anyway, so the barrier has no
        hole and reporting one would be crying wolf on ordinary overlapping
        data. Note this is the discriminating case, not the trivial one: the
        arm still burns nothing of its own.
        """
        report = _report([_fat_base_with_long_thin_arm(),
                          box(28.0, 40.0, 34.0, 64.0)])
        assert list(report.partially_vanished) == []
        assert list(report.vanished) == []
        assert report.ok, report.summary()

    def test_a_part_that_burned_its_own_cells_is_not_reported(self):
        """Shifting the arm onto the cell centres makes it present again."""
        shifted = box(30.0, 0.0, 32.0, 40.6).union(
            box(30.3, 40.6, 30.7, 64.0))  # covers x = 30.5
        report = _report([shifted])
        assert list(report.partially_vanished) == []

    def test_an_arm_that_runs_through_nodata_is_not_a_hole(self):
        """The same nodata/forbidden conflation, on the other detector.

        Step 2 asks how much of the part lies in cells that NO forbidden
        feature burned - and with ``fill_value=IMPASSABLE_CELL_COST`` those
        cells include every cell no feature covers at all. A barrier cannot be
        holed where nothing can walk, so the finished raster's passable mask
        settles it.
        """
        geometries = [_fat_base_with_long_thin_arm()]
        assert list(_report(geometries).partially_vanished) == [0], (
            "without the mask this fixture must still be reported, or the "
            "test below is vacuous")

        passable = np.ones((N64, N64), dtype=bool)
        passable[:, 29:33] = False   # the arm's corridor carries no data
        report = detect_forbidden_burn_defects(
            geometries, (N64, N64), TRANSFORM64, resolution_in_m=RES,
            passable=passable)
        assert list(report.partially_vanished) == []
        assert report.ok, report.summary()


if __name__ == "__main__":
    pytest.main([__file__])

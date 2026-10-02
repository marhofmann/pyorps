"""Forbidden features must actually block the router.

Under GDAL's default rule a cell burns only if its CENTRE falls inside the
polygon. For a feature narrower than one cell that is a coin toss decided by
sub-pixel alignment, and it fails in two distinct ways:

* VANISHING - a 0.8 m barrier on a 1 m grid burns 60 cells at centre x=30.1 but
  ZERO at x=30.0 or x=30.9. The exclusion is simply absent from the cost
  surface, silently, and shifting the study-area origin by 10 cm flips it.
* FRAGMENTED - a thin DIAGONAL barrier burns a one-cell-per-row staircase whose
  cells touch only at their corners. pyorps' OWN routers do not slip through
  that gap (a diagonal step is rejected when either flanking cell is
  impassable - ``pyorps/utils/_raster_context.pyx``), but the burn no longer
  represents a continuous barrier, and ``eight_connected`` below - a
  deliberately naive 8-connected flood fill, kept precisely as the contrast -
  walks straight across it.

Neither raises, warns, or leaves any trace in the result: the returned route is
genuinely optimal for the raster that was burned, the raster just does not
represent the world. That makes these tests the only line of defence.

Four fixes are exercised here, all through the real API:

* C ``widen_thin_forbidden=True`` (OPT-IN): forbidden features that are
  ACTUALLY narrower than sqrt(2) cells are buffered up to that width before
  the burn, and nothing else is touched.
* D ``all_touched=True`` (OPT-IN): forbidden features get a second ALL_TOUCHED
  pass painted over the ordinary burn, so a forbidden feature occupies every
  cell it intersects. Every non-forbidden cell keeps GDAL's default rule
  bit-for-bit - see ``test_ordinary_costs_are_untouched``.
* A ``on_thin_features`` detection, active whenever ``all_touched`` is off -
  i.e. also under the DEFAULT, which is the only setting that changes no cell.
* E ``suggest_resolution``, which turns the defect into a cell size.

WHY NEITHER C NOR D IS THE DEFAULT
----------------------------------
Both repair a vanishing barrier, and both SEAL legitimate sub-cell OPENINGS
while doing it - a sub-cell barrier and a sub-cell gap are the same geometry
seen from opposite sides. D is the worse of the two because it fattens every
forbidden feature rather than only the thin ones: measured through this
module's own ``burn()``, a 2 m forbidden wall carrying a 1.20 m slit keeps 4
open cells under the plain rule and under C, and 0 under D. But C is not safe
either - in the sweep reproduced by ``sweep_forbidden_repair_tradeoff.py`` it
sealed 38-100 % of the gates in a 0.4 m wall, depending only on which gate
widths were swept, which is the same band the overlay reaches there. So the DEFAULT is detection only, and the
raster it produces is bit-identical to a plain GDAL burn. See
``TestOpeningsSurvive`` here and ``TestTheDefaultChangesNothing`` there.
"""
import warnings

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.raster.thinness import (
    MIN_FORBIDDEN_WIDTH_CELLS,
    ForbiddenBurnRoutingAssessment,
    ForbiddenBurnSeverity,
    SealedOpeningWarning,
    ThinForbiddenFeatureError,
    ThinForbiddenFeatureWarning,
    detect_forbidden_burn_defects,
    is_thin,
    min_feature_width,
    recommended_geometry_buffer_m,
    safe_forbidden_width_m,
    suggest_resolution,
    widen_thin_features,
)

RES = 1.0
N = 60
EXTENT = N * RES
TRANSFORM = from_origin(0.0, EXTENT, RES, RES)
BBOX = box(0.0, 0.0, EXTENT, EXTENT)
CRS = "EPSG:32632"

FREE = 10
CHEAP = 40
PRICEY = 120

COSTS = {'category': {'free': FREE, 'cheap': CHEAP, 'pricey': PRICEY,
                      'barrier': IMPASSABLE_CELL_COST}}


def burn(features, **kwargs):
    """Rasterize ``[(geometry, category), ...]`` through the real API.

    A fresh GeoRasterizer per call, so the class-band cache can never carry
    state between the variants under comparison.
    """
    gdf = gpd.GeoDataFrame({'category': [c for _, c in features]},
                           geometry=[g for g, _ in features], crs=CRS)
    rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)
    rasterizer.rasterize(resolution_in_m=RES, bounding_box=BBOX, **kwargs)
    return rasterizer.raster


def eight_connected(grid, start, goal):
    """Can a NAIVE 8-connected walker reach *goal* from *start*?

    Deliberately naive: it admits a diagonal whenever both ENDPOINTS are free
    and never looks at the two flanking cells. pyorps' own relaxation does look
    at them, which is why this helper crosses a corner-touching staircase and
    the router does not. Keep it that way - it is the reference that makes the
    difference visible, not a model of pyorps.
    """
    free = grid != IMPASSABLE_CELL_COST
    assert free[start] and free[goal], "a probe point landed on the barrier"
    seen = np.zeros_like(free)
    seen[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < N and 0 <= nc < N and free[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return False


def burn_defects(geometries, all_touched=False):
    """Option A's per-feature verdict, straight from the public helper."""
    return detect_forbidden_burn_defects(geometries, (N, N), TRANSFORM,
                                         resolution_in_m=RES,
                                         all_touched=all_touched)


BACKGROUND = (box(0, 0, EXTENT, EXTENT), 'free')

#: Centres chosen to straddle the pixel-centre grid: x.0 and x.9 cover no cell
#: centre at all under the default rule, x.5 lands exactly on one.
ALIGNMENTS = (30.0, 30.1, 30.25, 30.5, 30.75, 30.9, 31.0)


def _vertical(cx, width=0.8):
    return box(cx - width / 2, 0.0, cx + width / 2, EXTENT)


def _diagonal(width=0.3):
    """Corner to corner, overshooting both ends so the grid is truly sealed.

    A diagonal that stops short of the edges lets the router walk AROUND it,
    which would make this test pass for the wrong reason.
    """
    return LineString([(-5.0, -5.0), (EXTENT + 5.0, EXTENT + 5.0)]).buffer(
        width / 2, cap_style=2)


#: The two repairs, as keyword sets. Both are OPT-IN, and each is spelled out
#: with the other OFF - otherwise every "D" assertion below would silently be
#: about C and D together.
FIXES = {'C (opt-in, widen thin)': {'widen_thin_forbidden': True,
                                    'all_touched': False},
         'D (opt-in, all_touched)': {'all_touched': True,
                                     'widen_thin_forbidden': False}}

#: Neither repair, i.e. GDAL's bare pixel-centre rule - and, since the repairs
#: became opt-in, exactly what ``rasterize()`` does with no arguments at all.
PLAIN = {'all_touched': False, 'widen_thin_forbidden': False}


def repaired(features, **kwargs):
    """``burn()`` with the seal warning silenced.

    A repair that makes a sub-cell barrier block necessarily disconnects the
    free space that used to run through it, so SealedOpeningWarning fires on
    every fixture in this module whose barrier spans the grid. That warning is
    the subject of ``test_forbidden_repair_tradeoff``; here it is noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SealedOpeningWarning)
        return burn(features, **kwargs)


class TestSubCellBarriersBlock:
    """Both repairs, C by default and D on request."""

    @pytest.mark.parametrize("fix", FIXES.values(), ids=list(FIXES))
    @pytest.mark.parametrize("cx", ALIGNMENTS)
    def test_a_thin_barrier_blocks_at_every_alignment(self, cx, fix):
        """0.8 m wide on a 1 m grid: must block wherever it happens to sit."""
        grid = repaired([BACKGROUND, (_vertical(cx), 'barrier')], **fix)
        assert (grid == IMPASSABLE_CELL_COST).any(), (
            f"barrier at x={cx} burned no cells at all")
        assert not eight_connected(grid, (0, 0), (0, N - 1)), (
            f"barrier at x={cx} is permeable: a route crosses it")

    @pytest.mark.parametrize("cx", ALIGNMENTS)
    def test_the_overlaid_barrier_has_no_defects(self, cx):
        """Option A, pointed at the overlay's own footprint, finds nothing.

        Structural counterpart of the routing test above: neither vanished nor
        fragmented, at every alignment.
        """
        geometry = _vertical(cx).intersection(BBOX)
        assert burn_defects([geometry], all_touched=True).ok

    def test_the_default_rule_alone_would_have_missed_some(self):
        """Guard: proves the parametrised test above is not vacuous.

        If GDAL's default rule ever stopped dropping sub-cell features this
        would fail, and the fix could be revisited.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            vanished = [
                cx for cx in ALIGNMENTS
                if (burn([BACKGROUND, (_vertical(cx), 'barrier')],
                         **PLAIN) == IMPASSABLE_CELL_COST).sum() == 0]
        assert vanished, ("no alignment dropped the barrier, so these tests no "
                          "longer exercise the defect they were written for")
        assert len(vanished) == 3, (
            f"expected the measured 3 of 7 alignments to vanish, got "
            f"{vanished}")


class TestThinDiagonalBarriers:
    @pytest.mark.parametrize("fix", FIXES.values(), ids=list(FIXES))
    def test_a_diagonal_barrier_is_not_permeable(self, fix):
        grid = repaired([BACKGROUND, (_diagonal(), 'barrier')], **fix)
        assert not eight_connected(grid, (2, 2), (N - 3, N - 3)), (
            "even a naive 8-connected walker crossed a sealed diagonal "
            "barrier")

    def test_a_diagonal_barrier_is_4_connected(self):
        """The structural property behind the test above.

        Asserted through option A's own per-feature check: a feature whose
        burned cells split into more 4-connected components than the geometry
        has parts is exactly a corner-touching staircase.
        """
        assert burn_defects([_diagonal().intersection(BBOX)],
                            all_touched=True).ok

    def test_the_default_rule_leaves_a_staircase(self):
        """Guard: the plain burn really is permeable, so the fix is load-bearing."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            plain = burn([BACKGROUND, (_diagonal(), 'barrier')], **PLAIN)
        assert eight_connected(plain, (2, 2), (N - 3, N - 3))
        report = burn_defects([_diagonal().intersection(BBOX)])
        assert report.fragmented.size == 1, (
            "the plain burn no longer fragments a 0.3 m diagonal")


class TestOrdinaryCostsAreUnaffected:
    """The bit-identity contract: only forbidden cells may change."""

    MIXED = [BACKGROUND, (box(5, 5, 25, 25), 'cheap'),
             (box(20, 20, 45, 40), 'pricey')]

    def _plain(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            return burn(self.MIXED + [(_diagonal(), 'barrier')], **PLAIN)

    @pytest.mark.parametrize("fix", FIXES.values(), ids=list(FIXES))
    def test_ordinary_costs_are_untouched(self, fix):
        """Every cell that is not forbidden in the repaired raster is legacy.

        Both repairs only ever ADD impassable cells - forbidden features are
        painted last, being the most expensive class - so a cell that comes out
        passable must still carry the value the bare pixel-centre burn gave it.
        """
        plain = self._plain()
        fixed = repaired(self.MIXED + [(_diagonal(), 'barrier')], **fix)

        passable = fixed != IMPASSABLE_CELL_COST
        assert np.array_equal(plain[passable], fixed[passable]), (
            "the forbidden repair changed a cell that is not forbidden")
        assert np.all(fixed[plain == IMPASSABLE_CELL_COST]
                      == IMPASSABLE_CELL_COST), (
            "the repair UNBLOCKED a cell the plain burn had forbidden")

    def test_every_changed_cell_touches_the_barrier(self):
        """The overlay grows the forbidden zone by at most one cell."""
        plain = self._plain()
        fixed = repaired(self.MIXED + [(_diagonal(), 'barrier')],
                         **FIXES['D (opt-in, all_touched)'])
        added = np.argwhere(plain != fixed)
        barrier = _diagonal()
        for row, col in added:
            cell = box(col * RES, EXTENT - (row + 1) * RES,
                       (col + 1) * RES, EXTENT - row * RES)
            assert barrier.intersects(cell), (
                f"cell {(row, col)} was painted forbidden without touching "
                f"the barrier")

    def test_no_forbidden_features_means_an_identical_raster(self):
        mixed = [BACKGROUND, (box(5, 5, 25, 25), 'cheap')]
        assert np.array_equal(burn(mixed, **PLAIN),
                              burn(mixed, all_touched=True))
        assert np.array_equal(burn(mixed, **PLAIN), burn(mixed))
        assert np.array_equal(burn(mixed, **PLAIN),
                              burn(mixed, widen_thin_forbidden=True))


class TestDetection:
    """Option A: the safety net for every burn that is not fattened by D."""

    def test_it_warns_when_nothing_repairs_the_burn(self):
        with pytest.warns(ThinForbiddenFeatureWarning, match="burned NO cells"):
            burn([BACKGROUND, (_vertical(30.0), 'barrier')], **PLAIN)

    def test_defect_warning_leads_with_routing_assessment(self):
        with pytest.warns(ThinForbiddenFeatureWarning) as caught:
            burn([BACKGROUND, (_vertical(30.0), 'barrier')], **PLAIN)
        message = str(next(w.message for w in caught
                           if issubclass(w.category, ThinForbiddenFeatureWarning)))
        assert "Severity: CRITICAL" in message, message
        assert "Practical takeaways:" in message, message
        assert "NOT RELIABLE" in message, message

    def test_the_warning_names_a_usable_resolution(self):
        """Option E feeds A's message, so the warning is actionable."""
        with pytest.warns(ThinForbiddenFeatureWarning) as caught:
            burn([BACKGROUND, (_vertical(30.0), 'barrier')], **PLAIN)
        message = str(next(w.message for w in caught
                           if issubclass(w.category, ThinForbiddenFeatureWarning)))
        assert "0.8" in message, message
        assert "cell size of" in message, message
        # BOTH repairs are named, and each carries its measured sealing
        # rate, so nobody enables one without meeting the trade first.
        assert "widen_thin_forbidden=True" in message, message
        assert "all_touched=True" in message, message
        assert "OPENINGS" in message, message
        assert "widen_thin_forbidden=True sealed 38-100 %" in message, message
        assert "all_touched=True 31-100 %" in message, message

    def test_it_emits_routing_assessment_when_there_is_nothing_to_report(self):
        """A fat barrier burns intact — assessment is informational, not a defect."""
        fat = box(28.0, 0.0, 32.0, EXTENT)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            with pytest.warns(ForbiddenBurnRoutingAssessment,
                              match="SUITABLE for route planning"):
                grid = burn([BACKGROUND, (fat, 'barrier')], **PLAIN)
        assert not eight_connected(grid, (0, 0), (0, N - 1))

    def test_it_warns_under_the_default(self):
        """The default repairs nothing, so A is the only thing standing."""
        with pytest.warns(ThinForbiddenFeatureWarning):
            burn([BACKGROUND, (_vertical(30.0), 'barrier')])

    def test_it_is_silent_once_widening_is_switched_on(self):
        """C repaired the geometry, so A has nothing left to find."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            warnings.simplefilter("ignore", SealedOpeningWarning)
            burn([BACKGROUND, (_vertical(30.0), 'barrier')],
                 widen_thin_forbidden=True)

    def test_it_is_silent_when_the_overlay_is_on(self):
        """D already fixed it; paying for detection as well would be waste."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            burn([BACKGROUND, (_vertical(30.0), 'barrier')],
                 all_touched=True)

    def test_raise_mode(self):
        with pytest.raises(ThinForbiddenFeatureError):
            burn([BACKGROUND, (_vertical(30.0), 'barrier')],
                 on_thin_features="raise", **PLAIN)

    def test_ignore_mode(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            grid = burn([BACKGROUND, (_vertical(30.0), 'barrier')],
                        on_thin_features="ignore", **PLAIN)
        assert not (grid == IMPASSABLE_CELL_COST).any(), (
            "x=30.0 is the alignment that vanishes; ignore mode must not "
            "silently repair it")

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="on_thin_features"):
            burn([BACKGROUND, (_vertical(30.0), 'barrier')],
                 on_thin_features="shout", **PLAIN)

    def test_detection_ignores_the_background_fill(self):
        """``raster == IMPASSABLE_CELL_COST`` is NOT the forbidden mask.

        ``fill_value`` defaults to the same 65535, so a detector wired to the
        cost raster would report the outside-every-feature background as one
        enormous corner-touching barrier. The public helper takes GEOMETRIES.
        """
        report = burn_defects([box(28.0, 0.0, 32.0, EXTENT)])
        assert report.ok
        assert report.n_forbidden == 1


class TestWidening:
    """Option C: repair the geometry instead of the raster."""

    @pytest.mark.parametrize("cx", ALIGNMENTS)
    def test_widening_alone_fixes_every_alignment(self, cx):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            warnings.simplefilter("ignore", SealedOpeningWarning)
            grid = burn([BACKGROUND, (_vertical(cx), 'barrier')],
                        all_touched=False, widen_thin_forbidden=True)
        assert (grid == IMPASSABLE_CELL_COST).any()
        assert not eight_connected(grid, (0, 0), (0, N - 1))

    def test_a_widened_diagonal_is_4_connected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            warnings.simplefilter("ignore", SealedOpeningWarning)
            grid = burn([BACKGROUND, (_diagonal(), 'barrier')],
                        all_touched=False, widen_thin_forbidden=True)
        assert not eight_connected(grid, (2, 2), (N - 3, N - 3))

    @pytest.mark.parametrize("cx", ALIGNMENTS)
    def test_a_widened_feature_is_no_longer_at_risk(self, cx):
        """The repair must not leave its own output flagged.

        A strip of width EXACTLY sqrt(2)*res erodes to a degenerate line, so
        ``is_thin`` reads the widened result as still-thin (measured: 125 of
        400 at 1 m, 343 of 400 at 5 m). Under the new default that would put
        every repaired feature on the report's at-risk list and buy a
        per-feature re-burn for each - cry wolf on data already fixed. The
        margin inside ``widen_thin_features`` is what clears it.
        """
        [widened] = widen_thin_features([_vertical(cx)], RES)
        assert not is_thin([widened], RES)[0]
        assert burn_defects([widened.intersection(BBOX)]).at_risk.size == 0

    def test_widening_leaves_fat_features_alone(self):
        fat = box(28.0, 0.0, 32.0, EXTENT)
        [widened] = widen_thin_features([fat], RES)
        assert widened is fat or widened.equals(fat)

    def test_widening_does_not_move_the_frame(self):
        """The frame must come from the UNWIDENED geometry.

        A forbidden feature on the outer edge, buffered, would otherwise
        enlarge total_bounds, shift the origin and change every cell - a worse
        silent failure than the one being fixed.
        """
        edge = box(-0.1, 0.0, 0.3, EXTENT)  # touches the western boundary
        features = [BACKGROUND, (edge, 'barrier')]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            plain = burn(features, **PLAIN)
            widened = burn(features, all_touched=False,
                           widen_thin_forbidden=True)
        assert plain.shape == widened.shape
        ordinary = (plain != IMPASSABLE_CELL_COST) & \
                   (widened != IMPASSABLE_CELL_COST)
        assert np.array_equal(plain[ordinary], widened[ordinary])

    def test_only_forbidden_features_are_widened(self):
        """A thin ORDINARY feature keeps the pixel-centre rule."""
        sliver = box(10.0, 0.0, 10.3, EXTENT)   # 0.3 m, ordinary cost
        features = [BACKGROUND, (sliver, 'cheap'),
                    (_vertical(30.0), 'barrier')]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            plain = burn(features, **PLAIN)
            widened = burn(features, all_touched=False,
                           widen_thin_forbidden=True)
        assert (plain == CHEAP).sum() == (widened == CHEAP).sum()


class TestThinPredicate:
    """The sqrt(2)-cell rule the other options are built on."""

    def test_the_constant_is_root_two_not_a_half(self):
        assert MIN_FORBIDDEN_WIDTH_CELLS == pytest.approx(2.0 ** 0.5)
        assert safe_forbidden_width_m(2.5) == pytest.approx(3.5355, abs=1e-4)

    def test_a_half_cell_buffer_is_not_enough(self):
        """Measured: +res/2 still vanishes for a 45-degree sliver."""
        sliver = LineString([(20, 20), (40, 40)]).buffer(0.15, cap_style=2)
        assert is_thin([sliver.buffer(RES / 2)], RES)[0]
        assert not is_thin([sliver.buffer(RES / MIN_FORBIDDEN_WIDTH_CELLS)],
                           RES)[0]

    def test_it_is_a_per_point_test_not_a_global_one(self):
        """A block with a thin spike is thin; global measures call it fat.

        Negative-buffer emptiness, 2*area/perimeter and the oriented envelope
        all average the spike away against 900 m2 of block.
        """
        spiked = box(0, 0, 30, 30).union(box(14.8, 30, 15.2, 40))
        assert is_thin([spiked], RES)[0]
        assert not is_thin([box(0, 0, 30, 30)], RES)[0]

    def test_a_tab_on_a_wide_body_is_not_thin(self):
        """The predicate measures INSCRIBED width, not feature width.

        A 0.4 m tab flush against a 30 m block is covered by a disk centred
        inside the block, so the feature burns intact and must not be flagged.
        Anything else would make the widening option chase harmless cadastral
        detail.
        """
        tab = box(0, 0, 30, 30).union(box(30, 10, 30.4, 20))
        assert not is_thin([tab], RES)[0]

    def test_min_feature_width_measures_below_the_cap(self):
        assert min_feature_width(_vertical(30.0, width=0.8), RES) == \
               pytest.approx(0.8, abs=0.01)
        assert min_feature_width(_diagonal(width=0.3), RES) == \
               pytest.approx(0.3, abs=0.01)

    def test_min_feature_width_is_a_gate_not_a_caliper(self):
        """None means 'at least the safe width', never 'unknown'."""
        assert min_feature_width(box(0, 0, 30, 30), RES) is None


class TestResolutionAdvice:
    """Option E."""

    def test_it_names_the_narrowest_feature(self):
        advice = suggest_resolution(
            [box(0, 0, 30, 30), _vertical(30.0, width=0.8),
             _diagonal(width=0.3)],
            RES, (0.0, 0.0, EXTENT, EXTENT))
        assert advice.narrowest_index == 2
        assert advice.narrowest_width_m == pytest.approx(0.3, abs=0.01)
        assert advice.n_thinner_than_safe_width == 2
        assert advice.n_thinner_than_a_cell == 2
        assert advice.safe_resolution_in_m == pytest.approx(
            advice.narrowest_width_m / MIN_FORBIDDEN_WIDTH_CELLS)

    def test_the_cell_count_comes_with_the_resolution(self):
        """A resolution alone is not advice - it can be unaffordable."""
        advice = suggest_resolution([_diagonal(width=0.05)], RES,
                                    (0.0, 0.0, 6000.0, 6000.0))
        assert advice.cells_at_safe_resolution > 1e9
        assert "cells over the same extent" in advice.summary()

    def test_nothing_thin_means_nothing_to_say(self):
        advice = suggest_resolution([box(0, 0, 30, 30)], RES)
        assert advice.narrowest_width_m is None
        assert advice.summary() == ""


class TestMetricStackOverlay:
    """rasterize_metrics carries the same guarantee."""

    @staticmethod
    def _stack(**kwargs):
        gdf = gpd.GeoDataFrame(
            {'category': ['free', 'barrier']},
            geometry=[box(0, 0, EXTENT, EXTENT), _vertical(30.0)], crs=CRS)
        rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)
        return rasterizer.rasterize_metrics(resolution_in_m=RES, **kwargs)

    @pytest.mark.parametrize("fix", FIXES.values(), ids=list(FIXES))
    def test_the_cost_band_keeps_the_barrier(self, fix):
        stack = self._stack(**fix)
        # add_layer folds the 65535 sentinel into forbidden_mask and stores 0
        # in the band itself, so the mask is where the barrier now lives.
        assert stack.forbidden_mask is not None
        assert stack.forbidden_mask.any(), (
            "the sub-cell barrier vanished from the metric stack")
        assert stack.forbidden_mask.sum() >= N, (
            "the barrier spans the full height, so it must claim a whole "
            "column of cells")

    def test_it_warns_with_every_repair_off(self):
        with pytest.warns(ThinForbiddenFeatureWarning):
            self._stack(**PLAIN)


# ==========================================================================
# Regression: the three defects adversarial verification found
# ==========================================================================

def _wall_with_slit(gap_m=1.20, x0=30.0, x1=32.0):
    """A 2 m forbidden wall carrying a legitimate horizontal opening.

    The wall itself is FAT - four times the safe width - so option C leaves it
    exactly as digitized. Only a rule that fattens forbidden features
    indiscriminately can close the gap.
    """
    half = gap_m / 2.0
    mid = EXTENT / 2.0
    return box(x0, 0.0, x1, mid - half).union(box(x0, mid + half, x1, EXTENT))


def _open_cells_in_the_wall(grid, x0=30, x1=32):
    """How many cells of the wall's own columns are still passable."""
    return int((grid[:, x0:x1] != IMPASSABLE_CELL_COST).sum())


class TestOpeningsSurvive:
    """DEFECT 1: ``all_touched=True`` seals legitimate sub-cell openings.

    The slit is 1.20 m - wider than a cell, so the raster CAN represent it,
    and it is the kind of feature (a gate, a culvert, a gap between parcels)
    that decides whether a study area is routable at all. D closes it anyway,
    because it fattens the fat wall on both sides of the gap; C does not,
    because the wall is not thin and is therefore never touched.
    """

    def test_the_slit_stays_open_under_the_default(self):
        grid = burn([BACKGROUND, (_wall_with_slit(), 'barrier')])
        assert _open_cells_in_the_wall(grid) == 4, (
            "the default sealed a 1.20 m opening")
        assert eight_connected(grid, (0, 0), (0, N - 1)), (
            "the 1.20 m gate is no longer walkable")

    def test_the_overlay_seals_it(self):
        """The measurement that decides the default; not an endorsement."""
        grid = burn([BACKGROUND, (_wall_with_slit(), 'barrier')],
                    **FIXES['D (opt-in, all_touched)'])
        assert _open_cells_in_the_wall(grid) == 0
        assert not eight_connected(grid, (0, 0), (0, N - 1))

    def test_widening_matches_the_plain_rule_here(self):
        """C is a no-op on a fat wall: the raster is the legacy one.

        This is what makes C the better of the two repairs, and it is the
        0 % in the fat-wall column of the sweep - the one figure that came out
        the same under every parameterization. It is not what makes it safe:
        on a 0.4 m wall C sealed 38-100 % of the gates, no better than the
        overlay's 31-100 % (see ``sweep_forbidden_repair_tradeoff.py``).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            plain = burn([BACKGROUND, (_wall_with_slit(), 'barrier')],
                         **PLAIN)
        assert np.array_equal(
            plain, burn([BACKGROUND, (_wall_with_slit(), 'barrier')],
                        widen_thin_forbidden=True))


#: A taller grid, so the 24 m arm of the fat-base-plus-thin-arm fixture fits.
N64 = 64
TRANSFORM64 = from_origin(0.0, float(N64), RES, RES)


def _fat_base_with_thin_arm():
    """ONE forbidden polygon: a 2 m base plus a 0.4 m arm on top of it.

    The base burns 2 solid columns for 40 rows; the arm burns NOTHING, so the
    barrier column reads "40 of 64 impassable, 24 passable" - a barrier with a
    hole in it, which is exactly what option A exists to catch.
    """
    return box(30.0, 0.0, 32.0, 40.0).union(
        box(30.55, 40.0, 30.95, float(N64)))


class TestPartialVanishIsDetected:
    """DEFECT 2: a PARTIAL vanish used to be invisible.

    The feature as a whole burns cells, so ``vanished`` is empty; the cells
    form one 4-connected block, so ``fragmented`` is empty. Only asking
    whether the burned footprint faithfully represents the GEOMETRY finds it.
    """

    def _report(self):
        return detect_forbidden_burn_defects(
            [_fat_base_with_thin_arm()], (N64, N64), TRANSFORM64,
            resolution_in_m=RES)

    def test_the_fixture_really_has_a_hole(self):
        """Guard: the arm must burn nothing, or the test is vacuous."""
        gdf = gpd.GeoDataFrame(
            {'category': ['free', 'barrier']},
            geometry=[box(0, 0, float(N64), float(N64)),
                      _fat_base_with_thin_arm()], crs=CRS)
        rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)
        with warnings.catch_warnings():
            # The point of this fixture IS the defect; A is expected to fire.
            warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
            rasterizer.rasterize(
                resolution_in_m=RES,
                bounding_box=box(0, 0, float(N64), float(N64)), **PLAIN)
        column = rasterizer.raster[:, 30]
        assert int((column == IMPASSABLE_CELL_COST).sum()) == 40
        assert int((column != IMPASSABLE_CELL_COST).sum()) == 24

    def test_the_older_signals_stay_silent(self):
        """Which is why a third one was needed, not why this is fine."""
        report = self._report()
        assert report.vanished.size == 0
        assert report.fragmented.size == 0

    def test_the_hole_is_reported(self):
        report = self._report()
        assert list(report.partially_vanished) == [0]
        assert not report.ok
        assert "burned only in part" in report.summary()

    def test_a_fat_feature_is_not_flagged(self):
        """The counter-example: no arm, no report."""
        report = detect_forbidden_burn_defects(
            [box(30.0, 0.0, 32.0, 40.0)], (N64, N64), TRANSFORM64,
            resolution_in_m=RES)
        assert report.ok
        assert report.partially_vanished.size == 0
        assert report.at_risk.size == 0

    def test_a_thin_feature_that_burned_fine_is_at_risk_not_defective(self):
        """Geometry-thinness predicts risk; only the burn confirms a defect.

        A detector that failed a clean burn on geometry alone would fire on
        165 of 1168 features in the measured 6000x6000 sweep and be switched
        off - so the at-risk list is reported but does not clear ``ok``.
        """
        report = burn_defects([_vertical(30.1).intersection(BBOX)])
        assert report.ok
        assert list(report.at_risk) == [0]

    def test_widening_repairs_the_hole(self):
        """The default fix reaches the arm, because the arm is what is thin."""
        gdf = gpd.GeoDataFrame(
            {'category': ['free', 'barrier']},
            geometry=[box(0, 0, float(N64), float(N64)),
                      _fat_base_with_thin_arm()], crs=CRS)
        rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            # Repairing this barrier necessarily disconnects the free space
            # that ran through its hole - that is the trade, reported by
            # SealedOpeningWarning and asserted on in the sibling module.
            warnings.simplefilter("ignore", SealedOpeningWarning)
            rasterizer.rasterize(resolution_in_m=RES,
                                 bounding_box=box(0, 0, float(N64),
                                                  float(N64)),
                                 widen_thin_forbidden=True)
        impassable = rasterizer.raster == IMPASSABLE_CELL_COST
        assert impassable[:, 28:34].any(axis=1).all(), (
            "the widened barrier still leaves a row open")


class TestOverpaintedForbiddenIsNotVanished:
    """DEFECT 3: a false positive on the detector's own test data.

    The id band is a replace burn, so a forbidden feature painted over by a
    LATER forbidden feature disappears from it. Its cells are still
    impassable - nothing is wrong - and reporting it teaches the reader to
    ignore the warning, which costs exactly as much as missing a real defect.
    """

    #: Both fat, both well inside the grid; the second swallows the first.
    COVERED = box(10.0, 10.0, 20.0, 20.0)
    COVERING = box(5.0, 5.0, 25.0, 25.0)

    def test_the_covered_feature_is_not_reported(self):
        report = burn_defects([self.COVERED, self.COVERING])
        assert list(report.vanished) == []
        assert report.ok, report.summary()

    def test_the_id_band_really_does_lose_it(self):
        """Guard: the false positive's mechanism is still present."""
        from pyorps.raster.thinness import (
            _burn_forbidden_id_band,
            _vanished_ids,
        )
        band = _burn_forbidden_id_band([self.COVERED, self.COVERING],
                                       (N, N), TRANSFORM)
        assert list(_vanished_ids(band, 2)) == [0], (
            "the covered feature no longer disappears from the id band, so "
            "this test no longer guards anything")

    def test_a_genuinely_absent_feature_is_still_reported(self):
        """The confirmation must not disarm the check it filters."""
        report = burn_defects([_vertical(30.0).intersection(BBOX),
                               self.COVERING])
        assert list(report.vanished) == [0]

    def test_rasterize_stays_silent_on_overlapping_forbidden_features(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            with pytest.warns(ForbiddenBurnRoutingAssessment):
                grid = burn([BACKGROUND, (self.COVERED, 'barrier'),
                             (self.COVERING, 'barrier')], **PLAIN)
        assert (grid == IMPASSABLE_CELL_COST).any()


class TestRoutingAssessment:
    """Severity verdict and practical takeaways for route planning."""

    def test_clean_burn_assesses_as_none(self):
        fat = box(28.0, 0.0, 32.0, EXTENT)
        report = burn_defects([fat])
        verdict = report.assess()
        assert verdict.severity is ForbiddenBurnSeverity.NONE
        assert verdict.routing_suitable is True
        assert "SUITABLE" in verdict.headline
        assert report.assessment_message().startswith(
            "Forbidden-feature burn audit")

    def test_vanished_assesses_as_critical(self):
        report = burn_defects([_vertical(30.0)])
        verdict = report.assess()
        assert verdict.severity is ForbiddenBurnSeverity.CRITICAL
        assert verdict.routing_suitable is False
        assert any("zero cells" in t for t in verdict.takeaways)

    def test_at_risk_only_assesses_as_low(self):
        """Thin but intact at this alignment → LOW, still suitable."""
        thin = box(29.9, 0.0, 30.9, EXTENT)  # 1 m wide at 1 m cells
        report = burn_defects([thin])
        assert report.ok
        assert report.at_risk.size > 0
        verdict = report.assess()
        assert verdict.severity is ForbiddenBurnSeverity.LOW
        assert verdict.routing_suitable is True

    def test_a_geometry_buffer_clears_sub_cell_vanishing(self):
        """ALKIS-style fix: one-cell outward buffer before the burn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ThinForbiddenFeatureWarning)
            with pytest.warns(ForbiddenBurnRoutingAssessment,
                              match="SUITABLE for route planning"):
                burn([BACKGROUND, (_vertical(30.0), 'barrier')],
                     geometry_buffer_m=1.0, **PLAIN)

    def test_recommended_buffer_matches_one_cell(self):
        assert recommended_geometry_buffer_m(1.0) == 1.0
        assert recommended_geometry_buffer_m(2.0) == 2.0

    def test_rasterizer_stores_last_report(self):
        fat = box(28.0, 0.0, 32.0, EXTENT)
        features = [BACKGROUND, (fat, 'barrier')]
        gdf = gpd.GeoDataFrame({'category': [c for _, c in features]},
                               geometry=[g for g, _ in features], crs=CRS)
        rz = GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ForbiddenBurnRoutingAssessment)
            rz.rasterize(resolution_in_m=RES, bounding_box=BBOX, **PLAIN)
        assert rz.last_forbidden_burn_report is not None
        assert rz.last_forbidden_burn_report.assess().severity is (
            ForbiddenBurnSeverity.NONE)


if __name__ == "__main__":
    pytest.main([__file__])

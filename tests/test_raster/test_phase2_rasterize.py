"""Equivalence tests for the Phase 2 rasterization rewrites.

Performance plan ``docs/superpowers/plans/2026-08-07-realworld-5km-performance-plan.md``
items 2.4 (single index burn + LUT gathers), 2.5 (class-id band + LUT re-cost)
and 2.7 (one sorted overlay burn, dropped bbox pre-burn).

All three replace *many* scan conversions of one geometry sequence with *one*.
The only thing that makes that legal is the paint order: rasterio burns with
``merge_alg=MergeAlg.replace``, so the feature owning a cell is the last one in
the sequence covering it — a choice that does not depend on the value burned.
Every test here therefore builds the LEGACY result explicitly and asserts the
new one is bit-identical to it, rather than asserting properties of the new
code. The reference implementations below are deliberate copies of the code
that was removed; they must not be refactored to call the production helpers.
"""
import unittest

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.core.metric_stack import MetricStack
from pyorps.io.geo_dataset import InMemoryRasterDataset, InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer

CRS = "EPSG:25832"
FILL = 65535
EXTENT = 48.0


def overlapping_features(seed=20260807, count=140, extent=EXTENT):
    """A deterministic soup of overlapping boxes across five land classes.

    The boxes are placed so class overlaps are dense (every class meets every
    other one) while a strip along the top edge stays uncovered, which is what
    exercises the fill values.
    """
    rng = np.random.default_rng(seed)
    classes = ["meadow", "field", "forest", "water", "built"]
    geometries, labels = [], []
    for index in range(count):
        x = float(rng.uniform(0.0, extent - 12.0))
        y = float(rng.uniform(0.0, extent - 18.0))
        width = float(rng.uniform(3.0, 13.0))
        height = float(rng.uniform(3.0, 13.0))
        geometries.append(box(x, y, x + width, y + height))
        labels.append(classes[index % len(classes)])
    # Two anchors pin the extent so every cost table sees the same grid.
    geometries += [box(0.0, 0.0, 0.5, 0.5), box(extent - 0.5, extent - 0.5,
                                                extent, extent)]
    labels += ["meadow", "meadow"]
    return gpd.GeoDataFrame({"landuse": labels, "geometry": geometries},
                            crs=CRS)


def make_rasterizer(gdf, cost_dict):
    return GeoRasterizer(InMemoryVectorDataset(gdf.copy(), crs=CRS),
                         cost_dict)


# --------------------------------------------------------------------------
# Legacy reference implementations (verbatim behaviour of the removed code)
# --------------------------------------------------------------------------

def legacy_prepared(gdf, cost_dict, fill_value=FILL, dtype="uint16"):
    """The frame the legacy burn saw: costs applied, filled, rounded, sorted."""
    data = gdf.copy()
    CostAssumptions(cost_dict).apply_to_geodataframe(data)
    if data["cost"].isna().any():
        data["cost"] = data["cost"].fillna(fill_value)
    data["cost"] = data["cost"].round().astype(dtype)
    return data.sort_values(by="cost", ascending=True)


def legacy_grid(rasterizer, data, resolution_in_m=1.0, bounding_box=None):
    """out_shape/transform exactly as :meth:`rasterize` derives them."""
    if bounding_box is None:
        out_shape = rasterizer._calculate_out_shape_from_geodataframe(
            data, resolution_in_m)
        transform = from_bounds(*data.total_bounds, *out_shape[::-1])
    else:
        out_shape = rasterizer._calculate_out_shape_from_bounding_box(
            bounding_box, resolution_in_m)
        transform = from_bounds(*bounding_box.bounds, *out_shape[::-1])
    return out_shape, transform


def legacy_cost_burn(data, out_shape, transform, fill_value=FILL,
                     dtype="uint16"):
    """The removed single-pass burn of the un-bounded branch."""
    shapes = ((geom, value) for geom, value
              in zip(data["geometry"], data["cost"]))
    return rasterize(shapes, out_shape=out_shape, fill=fill_value,
                     dtype=dtype, transform=transform)


def legacy_bbox_burn(data, bounding_box, out_shape, transform,
                     fill_value=FILL, dtype="uint16"):
    """The removed bounding-box branch: bbox pre-burn + one pass per value."""
    raster = rasterize([(bounding_box, fill_value)], out_shape=out_shape,
                       fill=fill_value, dtype=dtype, transform=transform)
    for unique_value in sorted(data["cost"].unique()):
        value_geoms = data.loc[data["cost"] == unique_value]
        shapes = ((geom, value) for geom, value
                  in zip(value_geoms["geometry"], value_geoms["cost"]))
        rasterize(shapes, out_shape=out_shape, fill=fill_value, out=raster,
                  dtype=dtype, transform=transform)
    return raster


def legacy_apply_cost_groups(raster, transform, gdf, ignore_value, multiply):
    """The removed per-unique-value loop of modify_raster_from_dataset."""
    raster = raster.copy()
    for unique_value in gdf["cost"].unique():
        value_geoms = gdf.loc[gdf["cost"] == unique_value]
        if value_geoms.empty:
            continue
        mask_array = geometry_mask(value_geoms["geometry"].values,
                                   transform=transform, invert=True,
                                   out_shape=raster.shape)
        if ignore_value is None:
            ignore_value_mask = np.ones_like(raster, dtype=bool)
        else:
            ignore_value_mask = raster != ignore_value
        mask = mask_array & ignore_value_mask
        if multiply:
            raster[mask] = np.clip(
                raster[mask].astype(np.uint32) * np.uint32(unique_value),
                0, np.iinfo(np.uint16).max).astype(raster.dtype)
        else:
            raster[mask] = unique_value
    return raster


def assert_bit_identical(test, produced, reference, what):
    """Byte equality, not np.allclose — the contract is bit-identical."""
    test.assertEqual(produced.dtype, reference.dtype, f"{what}: dtype")
    test.assertEqual(produced.shape, reference.shape, f"{what}: shape")
    if produced.tobytes() != reference.tobytes():
        differing = int(np.count_nonzero(produced != reference))
        test.fail(f"{what}: not bit-identical — {differing} of "
                  f"{produced.size} cells differ")


# ==========================================================================
# Item 2.4 — one index burn, K+2 gathers
# ==========================================================================

METRIC_ASSUMPTIONS = {
    "landuse": {
        "meadow": {"cost": 10.0, "landscape": 0.25, "permit": 3.5},
        "field": {"cost": 40.0, "landscape": 0.75, "permit": 1.0},
        "forest": {"cost": 300.0, "landscape": 2.5, "permit": 0.125},
        "built": {"cost": 900.0, "landscape": 1.5, "permit": 12.25},
        "water": 65535,  # scalar leaf: forbidden in every metric
    }
}


class TestIndexBurnEquivalence(unittest.TestCase):
    """The K+2 separate full-extent passes collapse into one, exactly."""

    def setUp(self):
        self.gdf = overlapping_features()
        self.rasterizer = make_rasterizer(self.gdf, METRIC_ASSUMPTIONS)
        self.stack = self.rasterizer.rasterize_metrics(resolution_in_m=1.0)
        self.reference = self._legacy_stack()

    def _legacy_stack(self):
        """rasterize_metrics as it was: one rasterize pass per band.

        Everything downstream of the burns (fills, sort order, add_layer,
        attach_category) is left identical, so any difference between the
        two stacks can only come from the burn/gather change itself.
        """
        rasterizer = make_rasterizer(self.gdf, METRIC_ASSUMPTIONS)
        manager = rasterizer.cost_manager
        data = rasterizer.base_data
        manager.apply_to_geodataframe(data)
        self.metric_names = manager.metric_names
        data["cost"] = data["cost"].fillna(FILL)
        for name in self.metric_names:
            if name != "cost":
                data[name] = data[name].fillna(0.0)
        data = data.sort_values(by="cost", ascending=True)

        out_shape = rasterizer._calculate_out_shape_from_geodataframe(
            data, 1.0)
        transform = from_bounds(*data.total_bounds, *out_shape[::-1])
        self.out_shape = out_shape
        self.transform = transform
        self.data = data

        stack = MetricStack(transform, rasterizer.crs)
        stack.add_layer("cost", rasterize(
            zip(data["geometry"], data["cost"].astype(np.float32)),
            out_shape=out_shape, fill=float(FILL), dtype="float32",
            transform=transform))
        for name in self.metric_names:
            if name == "cost":
                continue
            stack.add_layer(name, rasterize(
                zip(data["geometry"], data[name].astype(np.float32)),
                out_shape=out_shape, fill=0.0, dtype="float32",
                transform=transform))
        ids, labels = rasterizer._build_category_ids(data)
        stack.attach_category(rasterize(
            zip(data["geometry"], ids), out_shape=out_shape, fill=0,
            dtype="uint16", transform=transform), labels)
        return stack

    def test_grid_is_unchanged(self):
        self.assertEqual(self.stack.shape, tuple(self.out_shape))
        self.assertEqual(self.stack.transform, self.transform)
        self.assertEqual(self.stack.layer_names, self.reference.layer_names)

    def test_every_band_bit_identical(self):
        self.assertGreater(len(self.metric_names), 2)  # K+2 really is > 2
        for name in self.metric_names:
            with self.subTest(band=name):
                assert_bit_identical(self, self.stack[name],
                                     self.reference[name], f"band {name}")

    def test_category_band_bit_identical(self):
        assert_bit_identical(self, self.stack.category,
                             self.reference.category, "category band")
        self.assertEqual(self.stack.category_labels,
                         self.reference.category_labels)

    def test_forbidden_mask_bit_identical(self):
        assert_bit_identical(self, self.stack.forbidden_mask,
                             self.reference.forbidden_mask, "forbidden mask")

    def test_uncovered_cells_keep_the_per_band_fill(self):
        """Outside every feature: cost 65535, metrics 0.0, category 0.

        The 65535 cost fill is what add_layer turns into the forbidden mask
        (storing 0.0 in its place), so both halves are asserted.
        """
        covered = rasterize(
            ((geom, 1) for geom in self.data["geometry"]),
            out_shape=self.out_shape, fill=0, dtype="uint8",
            transform=self.transform).astype(bool)
        outside = ~covered
        self.assertGreater(int(outside.sum()), 0, "test grid is fully covered")
        self.assertTrue(np.all(self.stack.category[outside] == 0))
        self.assertTrue(np.all(self.stack.forbidden_mask[outside]))
        for name in self.metric_names:
            self.assertTrue(np.all(self.stack[name][outside] == 0.0), name)

        # And the raw burn really did put 65535 / 0.0 there.
        raw_cost = rasterize(
            zip(self.data["geometry"], self.data["cost"].astype(np.float32)),
            out_shape=self.out_shape, fill=float(FILL), dtype="float32",
            transform=self.transform)
        self.assertTrue(np.all(raw_cost[outside] == float(FILL)))

    def test_winner_per_cell_agrees_across_all_bands(self):
        """The alignment invariant the whole feasibility pipeline rests on."""
        views = self.rasterizer.cost_manager.metric_assumptions
        cost_view, landscape_view = views["cost"], views["landscape"]
        landscape_by_cost = {float(cost_view[key]): float(landscape_view[key])
                             for key in cost_view}
        self.assertEqual(len(landscape_by_cost), len(cost_view))  # unique

        raw_cost = rasterize(
            zip(self.data["geometry"], self.data["cost"].astype(np.float32)),
            out_shape=self.out_shape, fill=float(FILL), dtype="float32",
            transform=self.transform)
        inside = raw_cost != float(FILL)
        self.assertGreater(int(inside.sum()), 0)
        expected = np.array([landscape_by_cost[float(value)]
                             for value in raw_cost[inside]], dtype=np.float32)
        self.assertTrue(np.array_equal(self.stack["landscape"][inside],
                                       expected))


# ==========================================================================
# Item 2.5 — class-id band + LUT re-cost
# ==========================================================================

#: Baseline ranking meadow < field < forest < built < water(=forbidden).
COSTS_BASE = {"landuse": {"meadow": 10, "field": 40, "forest": 300,
                          "built": 900, "water": 65535}}
#: Same ranking, different magnitudes -> the burned band stays valid.
COSTS_RESCALED = {"landuse": {"meadow": 12, "field": 55, "forest": 310,
                              "built": 1200, "water": 65535}}
#: forest and built swap rank -> the burned band MUST be thrown away.
COSTS_SWAPPED = {"landuse": {"meadow": 10, "field": 40, "forest": 900,
                             "built": 300, "water": 65535}}
#: meadow and field tie; used to prove ties are painted in ranked order.
COSTS_TIED = {"landuse": {"meadow": 25, "field": 25, "forest": 300,
                          "built": 900, "water": 65535}}
#: ... and then separate again without changing anyone's rank.
COSTS_UNTIED = {"landuse": {"meadow": 25, "field": 90, "forest": 300,
                            "built": 900, "water": 65535}}


class TestClassBandLutRecost(unittest.TestCase):
    """A cost-table edit must give exactly what a full re-burn gives."""

    def setUp(self):
        self.gdf = overlapping_features()

    def reference_raster(self, cost_dict, resolution_in_m=1.0):
        """Legacy full re-burn for this cost table."""
        rasterizer = make_rasterizer(self.gdf, cost_dict)
        data = legacy_prepared(self.gdf, cost_dict)
        out_shape, transform = legacy_grid(rasterizer, data, resolution_in_m)
        return legacy_cost_burn(data, out_shape, transform)

    def test_first_burn_matches_legacy(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0)
        assert_bit_identical(self, rasterizer.raster,
                             self.reference_raster(COSTS_BASE), "first burn")
        self.assertIsNotNone(rasterizer._class_band_cache)

    def test_recost_with_unchanged_order_reuses_the_band(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0)
        burned = rasterizer._class_band_cache["band"]

        rasterizer.cost_manager = CostAssumptions(COSTS_RESCALED)
        rasterizer.rasterize(resolution_in_m=1.0)

        self.assertIs(rasterizer._class_band_cache["band"], burned,
                      "the class band was re-burned although the cost order "
                      "did not change")
        assert_bit_identical(self, rasterizer.raster,
                             self.reference_raster(COSTS_RESCALED),
                             "LUT re-cost")

    def test_cost_order_change_invalidates_the_cache(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0)
        burned = rasterizer._class_band_cache["band"]
        order = rasterizer._class_band_cache["order"].copy()
        codes_len = len(rasterizer._class_band_cache["codes"])
        self.assertGreater(codes_len, 0)

        rasterizer.cost_manager = CostAssumptions(COSTS_SWAPPED)
        rasterizer.rasterize(resolution_in_m=1.0)

        self.assertIsNot(rasterizer._class_band_cache["band"], burned,
                         "a rank swap did NOT invalidate the class band")
        expected = self.reference_raster(COSTS_SWAPPED)
        assert_bit_identical(self, rasterizer.raster, expected,
                             "re-burn after rank swap")
        # The swap really does change the answer, so the test has teeth.
        self.assertFalse(np.array_equal(expected,
                                        self.reference_raster(COSTS_BASE)))
        self.assertEqual(len(order), len(rasterizer._class_band_cache["order"]))

    def test_stale_band_would_have_been_wrong(self):
        """Proves the invalidation rule is load-bearing, not decorative.

        Gathers the swapped cost table through the band burned for the base
        table — i.e. what would ship if the rule only checked for changed
        cost VALUES instead of a changed cost ORDER.
        """
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0)
        cache = rasterizer._class_band_cache
        band, order, codes = cache["band"], cache["order"], cache["codes"]

        swapped = legacy_prepared(self.gdf, COSTS_SWAPPED)
        # Class values in the SAME code order the cache uses (base row order).
        by_class = np.zeros(len(order), dtype=np.uint16)
        base_order_costs = swapped.sort_index()["cost"].to_numpy()
        by_class[codes] = base_order_costs
        stale = rasterizer._gather_band(band, by_class[order], FILL,
                                        np.uint16)

        correct = self.reference_raster(COSTS_SWAPPED)
        self.assertGreater(int((stale != correct).sum()), 0,
                           "the stale-band scenario is not adversarial here")

    def test_ties_are_painted_in_ranked_order(self):
        """Two classes may tie today and separate tomorrow.

        Painting by raw cost value would let the tie be resolved arbitrarily;
        the band burns class RANKS, so a later re-cost that separates the two
        without reordering them still resolves overlaps correctly.
        """
        rasterizer = make_rasterizer(self.gdf, COSTS_TIED)
        rasterizer.rasterize(resolution_in_m=1.0)
        burned = rasterizer._class_band_cache["band"]

        rasterizer.cost_manager = CostAssumptions(COSTS_UNTIED)
        rasterizer.rasterize(resolution_in_m=1.0)

        self.assertIs(rasterizer._class_band_cache["band"], burned)
        assert_bit_identical(self, rasterizer.raster,
                             self.reference_raster(COSTS_UNTIED),
                             "re-cost that separates a tie")

    def test_use_class_cache_false_matches_and_clears(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0, use_class_cache=False)
        self.assertIsNone(rasterizer._class_band_cache)
        assert_bit_identical(self, rasterizer.raster,
                             self.reference_raster(COSTS_BASE),
                             "burn with the cache disabled")

    def test_invalidate_class_cache_forces_a_reburn(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0)
        burned = rasterizer._class_band_cache["band"]
        rasterizer.invalidate_class_cache()
        rasterizer.rasterize(resolution_in_m=1.0)
        self.assertIsNot(rasterizer._class_band_cache["band"], burned)
        assert_bit_identical(self, rasterizer.raster,
                             self.reference_raster(COSTS_BASE),
                             "burn after explicit invalidation")

    def test_resolution_change_rebuilds_the_band(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0)
        rasterizer.rasterize(resolution_in_m=2.0)
        assert_bit_identical(self, rasterizer.raster,
                             self.reference_raster(COSTS_BASE, 2.0),
                             "burn at a second resolution")

    def test_geometry_buffer_matches_legacy(self):
        """Buffering moved before the sort; the burn must not notice."""
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0, geometry_buffer_m=1.5)

        data = legacy_prepared(self.gdf, COSTS_BASE)
        buffered = data.copy()
        buffered["geometry"] = buffered.buffer(1.5)
        out_shape, transform = legacy_grid(rasterizer, buffered, 1.0)
        assert_bit_identical(self, rasterizer.raster,
                             legacy_cost_burn(buffered, out_shape, transform),
                             "buffered burn")


class TestBoundingBoxBranch(unittest.TestCase):
    """Item 2.7b: the dropped bbox pre-burn changed nothing."""

    def setUp(self):
        self.gdf = overlapping_features()
        self.bbox = box(-4.0, -4.0, EXTENT + 6.0, EXTENT + 2.0)

    def _legacy(self, cost_dict):
        rasterizer = make_rasterizer(self.gdf, cost_dict)
        data = legacy_prepared(self.gdf, cost_dict)
        out_shape, transform = legacy_grid(rasterizer, data, 1.0, self.bbox)
        return legacy_bbox_burn(data, self.bbox, out_shape, transform)

    def test_bbox_burn_matches_legacy_pre_burn_plus_loop(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0, bounding_box=self.bbox)
        assert_bit_identical(self, rasterizer.raster, self._legacy(COSTS_BASE),
                             "bounding-box burn")

    def test_bbox_recost_matches_legacy(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0, bounding_box=self.bbox)
        rasterizer.cost_manager = CostAssumptions(COSTS_SWAPPED)
        rasterizer.rasterize(resolution_in_m=1.0, bounding_box=self.bbox)
        assert_bit_identical(self, rasterizer.raster,
                             self._legacy(COSTS_SWAPPED),
                             "bounding-box re-cost")

    def test_area_outside_every_feature_is_the_fill_value(self):
        rasterizer = make_rasterizer(self.gdf, COSTS_BASE)
        rasterizer.rasterize(resolution_in_m=1.0, bounding_box=self.bbox)
        # The bbox reaches 6 m past the data on the right; that column band
        # can only hold the fill value.
        self.assertTrue(np.all(rasterizer.raster[:, -4:] == FILL))


# ==========================================================================
# Item 2.7a — one sorted overlay burn instead of a loop per unique value
# ==========================================================================

class TestOverlayCostGroups(unittest.TestCase):
    """modify_raster_from_dataset's per-value loop, collapsed exactly."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.transform = from_bounds(0, 0, 40, 40, 40, 40)
        base = rng.integers(1, 400, size=(40, 40)).astype(np.uint16)
        # A pre-existing forbidden patch: must stay untouched with the
        # default ignore_value.
        base[2:9, 2:9] = FILL
        self.base = base

        geometries, costs = [], []
        # Deliberately NOT sorted, with the ignore value appearing in the
        # middle so the freeze-then-skip behaviour is exercised.
        values = [7, FILL, 120, 7, FILL, 33]
        for index in range(36):
            x = float(rng.uniform(0, 30))
            y = float(rng.uniform(0, 30))
            geometries.append(box(x, y, x + float(rng.uniform(3, 11)),
                                  y + float(rng.uniform(3, 11))))
            costs.append(values[index % len(values)])
        self.overlay = gpd.GeoDataFrame({"cost": costs, "geometry": geometries},
                                        crs=CRS)

    def make_rasterizer(self):
        dataset = InMemoryRasterDataset(self.base.copy(), CRS, self.transform)
        rasterizer = GeoRasterizer(dataset, COSTS_BASE)
        rasterizer.transform = self.transform
        return rasterizer

    def test_replace_mode_matches_legacy(self):
        for ignore_value in (FILL, None, 7, 120):
            with self.subTest(ignore_value=ignore_value):
                rasterizer = self.make_rasterizer()
                rasterizer._apply_cost_groups(self.overlay, ignore_value,
                                              multiply=False)
                expected = legacy_apply_cost_groups(
                    self.base, self.transform, self.overlay, ignore_value,
                    multiply=False)
                assert_bit_identical(self, rasterizer.raster, expected,
                                     f"replace, ignore_value={ignore_value}")
                self.assertGreater(int((expected != self.base).sum()), 0)

    def test_multiply_mode_matches_legacy(self):
        for ignore_value in (FILL, None):
            with self.subTest(ignore_value=ignore_value):
                rasterizer = self.make_rasterizer()
                rasterizer._apply_cost_groups(self.overlay, ignore_value,
                                              multiply=True)
                expected = legacy_apply_cost_groups(
                    self.base, self.transform, self.overlay, ignore_value,
                    multiply=True)
                assert_bit_identical(self, rasterizer.raster, expected,
                                     f"multiply, ignore_value={ignore_value}")

    def test_forbidden_cells_are_never_modified(self):
        rasterizer = self.make_rasterizer()
        rasterizer._apply_cost_groups(self.overlay, FILL, multiply=False)
        self.assertTrue(np.all(rasterizer.raster[2:9, 2:9] == FILL))

    def test_single_group_matches_legacy(self):
        single = self.overlay.loc[self.overlay["cost"] == 120]
        self.assertFalse(single.empty)
        for multiply in (False, True):
            with self.subTest(multiply=multiply):
                rasterizer = self.make_rasterizer()
                rasterizer._apply_cost_groups(single, FILL, multiply=multiply)
                expected = legacy_apply_cost_groups(
                    self.base, self.transform, single, FILL, multiply)
                assert_bit_identical(self, rasterizer.raster, expected,
                                     f"single group, multiply={multiply}")

    def test_modify_raster_with_geodataframe_matches_legacy(self):
        for ignore_value in (FILL, None):
            with self.subTest(ignore_value=ignore_value):
                rasterizer = self.make_rasterizer()
                rasterizer.modify_raster_with_geodataframe(
                    self.overlay, value=55, ignore_value=ignore_value)
                mask = geometry_mask(self.overlay["geometry"].values,
                                     transform=self.transform, invert=True,
                                     out_shape=self.base.shape)
                expected = self.base.copy()
                if ignore_value is not None:
                    mask = mask & (expected != ignore_value)
                expected[mask] = 55
                assert_bit_identical(self, rasterizer.raster, expected,
                                     f"geodataframe, ignore={ignore_value}")


if __name__ == "__main__":
    unittest.main()

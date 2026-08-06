"""Tests for MetricStack: alignment, forbidden semantics, DEM handling,
joint windowing, masking, GeoTIFF round trip, legacy aliasing.

Phase 2 of docs/superpowers/plans/2026-08-04-feasibility-multi-objective-routing.md
"""
import os
import tempfile
import unittest
import warnings

import numpy as np
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from pyorps.core.exceptions import MetricStackError
from pyorps.core.metric_stack import FORBIDDEN_VALUE, MetricStack

TRANSFORM = from_origin(1000.0, 2000.0, 5.0, 5.0)  # 5 m cells
CRS = "EPSG:25832"


def make_stack(shape=(4, 6)):
    stack = MetricStack(TRANSFORM, CRS)
    cost = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    stack.add_layer("cost", cost)
    return stack


class TestStackBasics(unittest.TestCase):
    def test_construction_and_properties(self):
        stack = make_stack()
        self.assertEqual(stack.shape, (4, 6))
        self.assertEqual(stack.cell_size, 5.0)
        self.assertEqual(stack.layer_names, ["cost"])
        self.assertFalse(stack.is_legacy_alias)
        self.assertFalse(bool(stack.forbidden_mask.any()))

    def test_alignment_enforced(self):
        stack = make_stack((4, 6))
        with self.assertRaises(MetricStackError):
            stack.add_layer("landscape", np.zeros((4, 5), dtype=np.float32))
        with self.assertRaises(MetricStackError):
            stack.add_layer("landscape", np.zeros((4, 6, 1)))
        with self.assertRaises(MetricStackError):
            stack.attach_category(np.zeros((3, 6), dtype=np.uint16))

    def test_layer_name_validation(self):
        stack = make_stack()
        with self.assertRaises(MetricStackError):
            stack.add_layer("cost", np.zeros((4, 6)))       # duplicate
        with self.assertRaises(MetricStackError):
            stack.add_layer("length", np.zeros((4, 6)))     # reserved
        with self.assertRaises(MetricStackError):
            stack.add_layer("gradient", np.zeros((4, 6)))   # reserved
        with self.assertRaises(MetricStackError):
            stack.add_layer("__dem__", np.zeros((4, 6)))    # band name
        with self.assertRaises(MetricStackError):
            stack.add_layer("", np.zeros((4, 6)))

    def test_negative_values_rejected(self):
        stack = make_stack()
        bad = np.zeros((4, 6), dtype=np.float32)
        bad[1, 1] = -3.0
        with self.assertRaises(MetricStackError):
            stack.add_layer("landscape", bad)

    def test_missing_layer_error_lists_available(self):
        stack = make_stack()
        with self.assertRaises(MetricStackError) as ctx:
            _ = stack["landscape"]
        self.assertIn("cost", str(ctx.exception))
        stack.ensure_layers(["cost"])
        with self.assertRaises(MetricStackError):
            stack.ensure_layers(["cost", "permit"])


class TestForbiddenSemantics(unittest.TestCase):
    def test_forbidden_union_across_layers(self):
        stack = MetricStack(TRANSFORM, CRS)
        cost = np.ones((3, 3), dtype=np.float32)
        cost[0, 0] = FORBIDDEN_VALUE          # forbidden via sentinel
        stack.add_layer("cost", cost)
        landscape = np.zeros((3, 3), dtype=np.float32)
        landscape[1, 1] = np.nan              # forbidden via NaN
        landscape[2, 2] = np.inf              # forbidden via inf
        stack.add_layer("landscape", landscape)

        self.assertTrue(stack.forbidden_mask[0, 0])
        self.assertTrue(stack.forbidden_mask[1, 1])
        self.assertTrue(stack.forbidden_mask[2, 2])
        self.assertEqual(int(stack.forbidden_mask.sum()), 3)
        # layer arrays stay finite; forbidden cells hold 0
        self.assertEqual(stack["cost"][0, 0], 0.0)
        self.assertTrue(np.all(np.isfinite(stack["landscape"])))

    def test_mask_outside(self):
        stack = make_stack((4, 6))
        # geometry covering the left half of the grid (3 columns of 5 m)
        left_half = box(1000.0, 2000.0 - 20.0, 1000.0 + 15.0, 2000.0)
        stack.mask_outside(left_half)
        self.assertFalse(stack.forbidden_mask[:, :3].any())
        self.assertTrue(stack.forbidden_mask[:, 3:].all())


class TestDem(unittest.TestCase):
    def test_attach_same_shape(self):
        stack = make_stack((4, 6))
        dem = np.full((4, 6), 210.0, dtype=np.float32)
        stack.attach_dem(dem)
        np.testing.assert_array_equal(stack.dem, dem)

    def test_attach_resamples_other_shape(self):
        stack = make_stack((4, 6))
        dem = np.linspace(100.0, 200.0, 8 * 12,
                          dtype=np.float32).reshape(8, 12)
        stack.attach_dem(dem)
        self.assertEqual(stack.dem.shape, (4, 6))
        self.assertTrue(np.isfinite(stack.dem).all())
        with self.assertRaises(MetricStackError):
            stack.attach_dem(dem, resample=False)

    def test_nan_dem_cells_become_forbidden_and_filled(self):
        stack = make_stack((4, 6))
        dem = np.full((4, 6), 300.0, dtype=np.float32)
        dem[2, 2] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stack.attach_dem(dem)
        self.assertTrue(any("non-finite" in str(w.message) for w in caught))
        self.assertTrue(stack.forbidden_mask[2, 2])
        self.assertTrue(np.isfinite(stack.dem).all())
        self.assertAlmostEqual(float(stack.dem[2, 2]), 300.0, places=3)

    def test_dem_requires_defined_grid(self):
        stack = MetricStack(TRANSFORM, CRS)
        with self.assertRaises(MetricStackError):
            stack.attach_dem(np.zeros((4, 6), dtype=np.float32))


class TestWindowing(unittest.TestCase):
    def test_window_slices_every_band(self):
        stack = make_stack((6, 8))
        stack.add_layer("landscape",
                        np.random.default_rng(7).random((6, 8))
                        .astype(np.float32))
        stack.attach_category(
            np.arange(48, dtype=np.uint16).reshape(6, 8) % 5,
            labels={0: "none", 1: "a", 2: "b", 3: "c", 4: "d"})
        stack.attach_dem(np.full((6, 8), 100.0, dtype=np.float32))

        sub = stack.window(Window(2, 1, 4, 3))  # col_off, row_off, w, h
        self.assertEqual(sub.shape, (3, 4))
        np.testing.assert_array_equal(sub["cost"],
                                      stack["cost"][1:4, 2:6])
        np.testing.assert_array_equal(sub["landscape"],
                                      stack["landscape"][1:4, 2:6])
        np.testing.assert_array_equal(sub.category,
                                      stack.category[1:4, 2:6])
        np.testing.assert_array_equal(sub.dem, stack.dem[1:4, 2:6])
        np.testing.assert_array_equal(sub.forbidden_mask,
                                      stack.forbidden_mask[1:4, 2:6])
        self.assertEqual(sub.category_labels, stack.category_labels)
        # transform shifted by the window offset (2 cols, 1 row, 5 m cells)
        self.assertAlmostEqual(sub.transform.c,
                               stack.transform.c + 2 * 5.0)
        self.assertAlmostEqual(sub.transform.f,
                               stack.transform.f - 1 * 5.0)
        self.assertEqual(sub.cell_size, stack.cell_size)


class TestRoundTrip(unittest.TestCase):
    def test_save_load_roundtrip(self):
        stack = make_stack((5, 7))
        landscape = np.random.default_rng(3).random((5, 7)) \
            .astype(np.float32)
        stack.add_layer("landscape", landscape)
        cost = stack["cost"]
        stack.attach_category(
            (np.arange(35, dtype=np.uint16) % 4).reshape(5, 7),
            labels={1: "Forest", 2: "Water > River", 3: "Urban"})
        stack.attach_dem(
            np.linspace(100, 150, 35, dtype=np.float32).reshape(5, 7))
        stack.forbidden_mask[0, 0] = True

        with tempfile.NamedTemporaryFile(suffix=".tiff",
                                         delete=False) as tmp:
            path = tmp.name
        try:
            stack.save(path)
            loaded = MetricStack.load(path)
            self.assertEqual(loaded.layer_names, ["cost", "landscape"])
            np.testing.assert_array_equal(loaded["cost"], cost)
            np.testing.assert_array_equal(loaded["landscape"], landscape)
            np.testing.assert_array_equal(loaded.forbidden_mask,
                                          stack.forbidden_mask)
            np.testing.assert_array_equal(loaded.category, stack.category)
            self.assertEqual(loaded.category_labels, stack.category_labels)
            np.testing.assert_array_equal(loaded.dem, stack.dem)
            self.assertEqual(loaded.cell_size, stack.cell_size)
            for attr in ("a", "e", "c", "f"):
                self.assertAlmostEqual(getattr(loaded.transform, attr),
                                       getattr(stack.transform, attr))
        finally:
            os.unlink(path)

    def test_load_rejects_foreign_tiff(self):
        import rasterio
        with tempfile.NamedTemporaryFile(suffix=".tiff",
                                         delete=False) as tmp:
            path = tmp.name
        try:
            with rasterio.open(path, "w", driver="GTiff", height=2, width=2,
                               count=1, dtype="float32", crs=CRS,
                               transform=TRANSFORM) as dst:
                dst.write(np.zeros((2, 2), dtype=np.float32), 1)
            with self.assertRaises(MetricStackError):
                MetricStack.load(path)
        finally:
            os.unlink(path)


class TestLegacyAlias(unittest.TestCase):
    def test_zero_copy_until_access(self):
        raster = np.full((4, 6), 130, dtype=np.uint16)
        raster[0, 0] = 65535
        stack = MetricStack.from_single_raster(raster, TRANSFORM, CRS)
        self.assertTrue(stack.is_legacy_alias)
        # the original object, untouched — the pinning guarantee
        self.assertIs(stack.legacy_raster, raster)
        self.assertEqual(stack.shape, (4, 6))

        # first layer access materializes the float view
        cost = stack["cost"]
        self.assertFalse(stack.is_legacy_alias)
        self.assertEqual(cost.dtype, np.float32)
        self.assertEqual(cost[1, 1], 130.0)
        self.assertEqual(cost[0, 0], 0.0)  # forbidden cells hold 0
        self.assertTrue(stack.forbidden_mask[0, 0])
        self.assertEqual(int(stack.forbidden_mask.sum()), 1)
        # the aliased raster is still available and untouched
        self.assertEqual(raster[0, 0], 65535)
        self.assertEqual(raster[1, 1], 130)

    def test_rejects_non_2d(self):
        with self.assertRaises(MetricStackError):
            MetricStack.from_single_raster(
                np.zeros((1, 4, 6), dtype=np.uint16), TRANSFORM, CRS)


if __name__ == "__main__":
    unittest.main()

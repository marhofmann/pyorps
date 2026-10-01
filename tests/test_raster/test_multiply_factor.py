"""The multiply-mode cost factor must survive as a fraction.

``modify_raster_with_geodataframe(multiply=True)`` and its bulk sibling
``_apply_cost_groups_multiply`` both used to widen the factor to ``uint32``
alongside the raster::

    raster[mask].astype(np.uint32) * np.uint32(value)

The widening was only ever needed so the PRODUCT could not overflow uint16.
Applying it to the factor truncated it, and cost factors are routinely
fractional: ``create_raster_for_windfarm_layout.py`` multiplies by 1.05, 1.15,
1.2, 1.25, 1.3, 1.5 and 1.8 for drinking-water zones, soil classes, landscape
protection and Natura 2000. Every one of those became ``np.uint32(x) == 1``,
i.e. a silent no-op, and 2.5 became 2 -- so a raster burned today did not
reproduce one burned when the code was correct.

The existing equivalence suites could not catch this: they pass integer costs
(2 and 5), for which the truncating and non-truncating forms agree exactly.
These tests pin the fractional case in both code paths, and pin the integer
case too so the fix cannot regress the values those suites already assert.
"""
import unittest

import geopandas as gpd
import numpy as np
from rasterio.transform import from_bounds
from shapely.geometry import box

from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer

CRS = "EPSG:25832"
EXTENT = 16.0


def _rasterizer(base_cost=100):
    """A uniform raster of ``base_cost`` over a 16 m square, at 1 m cells."""
    gdf = gpd.GeoDataFrame(
        {"landuse": ["field"], "geometry": [box(0, 0, EXTENT, EXTENT)]},
        crs=CRS,
    )
    dataset = InMemoryVectorDataset(gdf)
    assumptions = CostAssumptions({"landuse": {"field": base_cost}})
    rasterizer = GeoRasterizer(dataset, assumptions)
    rasterizer.rasterize(resolution_in_m=1.0)
    return rasterizer


def _overlay(cost):
    """A half-width strip carrying ``cost``, to be applied as a factor."""
    return gpd.GeoDataFrame(
        {"cost": [cost], "geometry": [box(0, 0, EXTENT / 2, EXTENT)]},
        crs=CRS,
    )


class TestFractionalFactorSurvives(unittest.TestCase):
    """A fractional factor must scale the cost, not vanish."""

    def test_single_geodataframe_path_applies_a_fractional_factor(self):
        rasterizer = _rasterizer(base_cost=100)
        rasterizer.modify_raster_with_geodataframe(
            _overlay(1.15), value=1.15, multiply=True)
        touched = rasterizer.raster[:, : int(EXTENT / 2)]
        untouched = rasterizer.raster[:, int(EXTENT / 2):]
        self.assertTrue(np.all(touched == 115),
                        f"expected 115, got {np.unique(touched)}")
        self.assertTrue(np.all(untouched == 100))

    def test_bulk_cost_group_path_applies_a_fractional_factor(self):
        rasterizer = _rasterizer(base_cost=100)
        rasterizer._apply_cost_groups(_overlay(1.15), ignore_value=None,
                                      multiply=True)
        touched = rasterizer.raster[:, : int(EXTENT / 2)]
        self.assertTrue(np.all(touched == 115),
                        f"expected 115, got {np.unique(touched)}")

    def test_every_factor_the_case_study_uses_is_not_a_no_op(self):
        """The exact factor list from create_raster_for_windfarm_layout.py."""
        for factor in (1.05, 1.15, 1.2, 1.25, 1.3, 1.5, 1.8):
            with self.subTest(factor=factor):
                rasterizer = _rasterizer(base_cost=200)
                rasterizer.modify_raster_with_geodataframe(
                    _overlay(factor), value=factor, multiply=True)
                touched = rasterizer.raster[:, : int(EXTENT / 2)]
                self.assertEqual(int(touched[0, 0]), round(200 * factor))
                self.assertNotEqual(int(touched[0, 0]), 200,
                                    "factor was truncated to a no-op")

    def test_a_factor_below_one_reduces_the_cost(self):
        rasterizer = _rasterizer(base_cost=400)
        rasterizer.modify_raster_with_geodataframe(
            _overlay(0.5), value=0.5, multiply=True)
        touched = rasterizer.raster[:, : int(EXTENT / 2)]
        # Truncation made this 0, which is worse than a no-op: it turns
        # expensive ground into free ground.
        self.assertTrue(np.all(touched == 200),
                        f"expected 200, got {np.unique(touched)}")


class TestIntegerFactorUnchanged(unittest.TestCase):
    """The corrected path must agree with the old one wherever it was right."""

    def test_integer_factors_are_bit_identical_to_the_truncating_form(self):
        for factor in (1, 2, 3, 5, 100):
            with self.subTest(factor=factor):
                rasterizer = _rasterizer(base_cost=137)
                before = rasterizer.raster.copy()
                rasterizer.modify_raster_with_geodataframe(
                    _overlay(factor), value=factor, multiply=True)
                touched = rasterizer.raster[:, : int(EXTENT / 2)]
                legacy = np.clip(
                    before[:, : int(EXTENT / 2)].astype(np.uint32)
                    * np.uint32(factor),
                    0, np.iinfo(np.uint16).max).astype(before.dtype)
                np.testing.assert_array_equal(touched, legacy)

    def test_the_product_still_clips_instead_of_overflowing(self):
        rasterizer = _rasterizer(base_cost=60000)
        rasterizer.modify_raster_with_geodataframe(
            _overlay(2.0), value=2.0, multiply=True)
        touched = rasterizer.raster[:, : int(EXTENT / 2)]
        self.assertTrue(np.all(touched == np.iinfo(np.uint16).max))


if __name__ == "__main__":
    unittest.main()

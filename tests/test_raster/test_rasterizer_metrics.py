"""Tests for GeoRasterizer.rasterize_metrics (multi-band metric stacks).

The crucial property: every band is rasterized from the SAME sorted geometry
sequence, so the winning feature on overlaps is identical in every band —
cost, metrics, and category can never disagree about which feature a cell
belongs to.
"""
import unittest

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer

CRS = "EPSG:25832"

MULTI_ASSUMPTIONS = {
    "landuse": {
        "cheap": {"cost": 10.0, "landscape": 0.9},
        "expensive": {"cost": 500.0, "landscape": 0.1},
        "blocked": 65535,
    }
}


def make_rasterizer(assumptions=None):
    """Two overlapping squares plus a forbidden one, on a 10x30 m extent."""
    geometries = [
        Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]),     # cheap (left 2/3)
        Polygon([(10, 0), (30, 0), (30, 10), (10, 10)]),   # expensive (right)
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),         # blocked corner
    ]
    gdf = gpd.GeoDataFrame({
        "landuse": ["cheap", "expensive", "blocked"],
        "geometry": geometries,
    }, geometry="geometry", crs=CRS)
    dataset = InMemoryVectorDataset(gdf, crs=CRS)
    return GeoRasterizer(dataset, assumptions or MULTI_ASSUMPTIONS)


class TestRasterizeMetrics(unittest.TestCase):
    def setUp(self):
        self.rasterizer = make_rasterizer()
        self.stack = self.rasterizer.rasterize_metrics(resolution_in_m=1.0)

    def test_stack_layout(self):
        self.assertEqual(self.stack.layer_names, ["cost", "landscape"])
        self.assertEqual(self.stack.shape, (10, 30))
        self.assertEqual(self.stack.cell_size, 1.0)
        self.assertIsNotNone(self.stack.category)
        self.assertIs(self.rasterizer.metric_stack, self.stack)

    def test_float_metrics_not_rounded(self):
        landscape = self.stack["landscape"]
        self.assertIn(0.9, np.unique(landscape))
        self.assertIn(0.1, np.unique(landscape))

    def test_overlap_winner_consistent_across_bands(self):
        """In the overlap zone the expensive feature wins in EVERY band."""
        cost = self.stack["cost"]
        landscape = self.stack["landscape"]
        category = self.stack.category
        labels = self.stack.category_labels

        # overlap zone: x in (10, 20), pick an interior cell (row 5, col 15)
        self.assertEqual(cost[5, 15], 500.0)
        self.assertEqual(landscape[5, 15], 0.1)
        self.assertEqual(labels[int(category[5, 15])], "expensive")

        # cheap-only zone (col 7, outside the blocked corner rows)
        self.assertEqual(cost[2, 7], 10.0)
        self.assertEqual(landscape[2, 7], 0.9)
        self.assertEqual(labels[int(category[2, 7])], "cheap")

    def test_alignment_property_everywhere(self):
        """cost==500 <=> landscape==0.1 <=> category=='expensive' etc."""
        cost = self.stack["cost"]
        landscape = self.stack["landscape"]
        forbidden = self.stack.forbidden_mask
        valid = ~forbidden
        np.testing.assert_array_equal(
            (cost[valid] == 500.0), (landscape[valid] == 0.1))
        np.testing.assert_array_equal(
            (cost[valid] == 10.0), (landscape[valid] == 0.9))

    def test_forbidden_class_and_fill(self):
        forbidden = self.stack.forbidden_mask
        # blocked corner: rows 5..10 (y in 0..5) x cols 0..5 => bottom rows
        self.assertTrue(forbidden[7, 2])
        # cells inside features and not blocked are traversable
        self.assertFalse(forbidden[2, 7])
        self.assertFalse(forbidden[5, 25])
        # forbidden cells hold 0 in the layers (finite invariant)
        self.assertEqual(self.stack["cost"][7, 2], 0.0)

    def test_category_labels_complete(self):
        labels = set(self.stack.category_labels.values())
        self.assertEqual(labels, {"cheap", "expensive", "blocked"})

    def test_without_category_band(self):
        rasterizer = make_rasterizer()
        stack = rasterizer.rasterize_metrics(resolution_in_m=1.0,
                                             include_category=False)
        self.assertIsNone(stack.category)

    def test_weight_metric_flows_into_stack(self):
        rasterizer = make_rasterizer({
            "landuse": {
                "cheap": {"cost": 10.0, "factor": 3.0},
                "expensive": {"cost": 500.0},
                "blocked": 65535,
            }
        })
        stack = rasterizer.rasterize_metrics(resolution_in_m=1.0)
        self.assertIn("weight", stack.layer_names)
        self.assertEqual(stack["weight"][2, 7], 30.0)     # 10 * 3
        self.assertEqual(stack["weight"][5, 25], 500.0)

    def test_legacy_rasterize_still_works_with_multi_metric(self):
        """The legacy single-band path keeps producing the uint16 raster."""
        rasterizer = make_rasterizer()
        dataset = rasterizer.rasterize(resolution_in_m=1.0)
        self.assertEqual(dataset.data.dtype, np.uint16)
        self.assertEqual(rasterizer.raster[5, 25], 500)
        self.assertEqual(rasterizer.raster[7, 2], 65535)


class TestRasterizeMetricsScalarAssumptions(unittest.TestCase):
    def test_scalar_assumptions_produce_cost_only_stack(self):
        rasterizer = make_rasterizer({
            "landuse": {"cheap": 10, "expensive": 500, "blocked": 65535}
        })
        stack = rasterizer.rasterize_metrics(resolution_in_m=1.0)
        self.assertEqual(stack.layer_names, ["cost"])
        self.assertEqual(stack["cost"][5, 25], 500.0)
        self.assertTrue(stack.forbidden_mask[7, 2])


if __name__ == "__main__":
    unittest.main()

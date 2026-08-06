"""Phase 9 tests: the lossless float32 weight-precision mode.

The raison d'etre test: two equal-length corridors whose per-meter costs
differ by 2 units on a 1e6 base — indistinguishable after uint16
quantization (the combine diagnostics warn), but the float32 mode routes
through the truly cheaper one.
"""
import unittest
import warnings

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from pyorps import PathFinder
from pyorps.core.metric_stack import MetricStack
from pyorps.core.objective import Objective

CRS = "EPSG:25832"


def corridor_gdf():
    """Two equal-length corridors (rows split by a barrier) + end caps."""
    polys = {
        "cap": [Polygon([(0, 0), (3, 0), (3, 12), (0, 12)]),
                Polygon([(27, 0), (30, 0), (30, 12), (27, 12)])],
        "corridor_a": [Polygon([(3, 7), (27, 7), (27, 12), (3, 12)])],
        "corridor_b": [Polygon([(3, 0), (27, 0), (27, 5), (3, 5)])],
        "barrier": [Polygon([(3, 5), (27, 5), (27, 7), (3, 7)])],
    }
    records = [(k, g) for k, geoms in polys.items() for g in geoms]
    return gpd.GeoDataFrame({"landuse": [r[0] for r in records],
                             "geometry": [r[1] for r in records]}, crs=CRS)


ASSUMPTIONS = {"landuse": {
    "cap": {"cost": 1_000_000.0},
    "corridor_a": {"cost": 1_000_003.0},   # 2/m more expensive than B
    "corridor_b": {"cost": 1_000_001.0},
    "barrier": 65535,
}}


def make_finder(precision, graph_api="networkx", objective=None):
    return PathFinder(
        dataset_source=corridor_gdf(),
        source_coords=(1.5, 6.0), target_coords=(28.5, 6.0),
        search_space_buffer_m=50,
        graph_api=graph_api,
        cost_assumptions=ASSUMPTIONS,
        objective=objective or {"cost": 1.0},
        weight_precision=precision,
        resolution_in_m=1.0,
    )


def route_landuse(finder, path):
    """Which corridor did the route use? Returns set of class labels."""
    sub = finder.metric_stack.window(finder.raster_handler.window)
    cols = finder.raster_handler.data[0].shape[1]
    idx = np.array(path.path_indices)
    labels = sub.category_labels
    cats = sub.category[idx // cols, idx % cols]
    return {labels[int(c)] for c in cats if int(c) in labels}


class TestCombineFloatPath(unittest.TestCase):
    def test_quantize_false_is_lossless(self):
        stack = MetricStack(from_origin(0, 4, 1, 1), CRS)
        cost = np.array([[1e6, 1e6 + 1], [1e6 + 3, 65535]],
                        dtype=np.float32)
        stack.add_layer("cost", cost)
        result = stack.combine(Objective.cheapest(), quantize=False)
        self.assertEqual(result.weights.dtype, np.float32)
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.resolution, 0.0)
        self.assertEqual(result.weights[0, 0], 1e6)
        self.assertEqual(result.weights[0, 1], 1e6 + 1)  # NOT collapsed
        self.assertTrue(np.isinf(result.weights[1, 1]))  # forbidden

    def test_uint16_collapses_the_same_surface(self):
        stack = MetricStack(from_origin(0, 4, 1, 1), CRS)
        cost = np.array([[1e6, 1e6 + 1], [1e6 + 3, 100.0]],
                        dtype=np.float32)
        stack.add_layer("cost", cost)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = stack.combine(Objective.cheapest())
        self.assertEqual(result.weights[0, 0], result.weights[0, 1])


class TestResolutionRescue(unittest.TestCase):
    def test_float_mode_finds_the_truly_cheaper_corridor(self):
        """uint16 cannot tell the corridors apart; float32 can."""
        f_float = make_finder("float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # collapse warning is uint16's
            path = f_float.find_route()
        used = route_landuse(f_float, path)
        self.assertIn("corridor_b", used)
        self.assertNotIn("corridor_a", used)
        # honest metrics still work in float mode
        self.assertIsNotNone(path.metrics)
        self.assertIsNotNone(path.total_length)
        self.assertGreater(path.feasibility, 0)
        # legacy uint16-only fields are absent (documented)
        self.assertIsNone(path.total_cost)

    def test_uint16_mode_warns_about_collapse(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f_int = make_finder("uint16", graph_api="cython")  # combine runs here
        self.assertTrue(any("collapsed" in str(w.message) for w in caught))
        f_int.find_route()
        # proof of collapse: both corridors share one quantized value
        sub = f_int.metric_stack.window(f_int.raster_handler.window)
        W = f_int.raster_handler.data[0]
        labels = {v: k for k, v in sub.category_labels.items()}
        cat = sub.category
        w_a = np.unique(W[(cat == labels["corridor_a"])
                          & np.isfinite(sub["cost"])])
        w_b = np.unique(W[cat == labels["corridor_b"]])
        np.testing.assert_array_equal(w_a, w_b)

    def test_float_mode_on_gpu(self):
        pytest.importorskip("cupy")
        f_gpu = make_finder("float32", graph_api="raster_gpu")
        path = f_gpu.find_route()
        used = route_landuse(f_gpu, path)
        self.assertIn("corridor_b", used)
        self.assertNotIn("corridor_a", used)


class TestFloatGradient(unittest.TestCase):
    def test_float_plus_gradient_contour(self):
        """float precision composes with the gradient machinery."""
        N = 41
        dem = (np.arange(N, dtype=np.float32)[:, None] * 0.6
               * np.ones((1, N), dtype=np.float32))
        finder = PathFinder(
            dataset_source=np.full((N, N), 100, dtype=np.uint16),
            crs=CRS, transform=from_origin(0.0, float(N), 1.0, 1.0),
            source_coords=(2.5, 20.5), target_coords=(38.5, 20.5),
            search_space_buffer_m=200, graph_api="networkx",
            dem=dem, objective={"gradient": 50.0, "length": 0.01},
            weight_precision="float32")
        path = finder.find_route()
        cols = finder.raster_handler.data[0].shape[1]
        rows = np.unique(np.array(path.path_indices) // cols)
        self.assertEqual(len(rows), 1)  # stays on the contour


class TestFloatGuards(unittest.TestCase):
    def test_cython_rejected(self):
        with self.assertRaises(NotImplementedError):
            make_finder("float32", graph_api="cython")

    def test_requires_objective(self):
        with self.assertRaises(ValueError):
            PathFinder(
                dataset_source=np.full((10, 10), 5, dtype=np.uint16),
                crs=CRS, transform=from_origin(0, 10, 1, 1),
                source_coords=(1.5, 5.5), target_coords=(8.5, 5.5),
                search_space_buffer_m=50, graph_api="networkx",
                weight_precision="float32")

    def test_requires_explicit_buffer(self):
        with self.assertRaises(ValueError):
            PathFinder(
                dataset_source=np.full((10, 10), 5, dtype=np.uint16),
                crs=CRS, transform=from_origin(0, 10, 1, 1),
                source_coords=(1.5, 5.5), target_coords=(8.5, 5.5),
                search_space_buffer_m=None, graph_api="networkx",
                objective={"cost": 1.0}, weight_precision="float32")

    def test_invalid_precision_string(self):
        with self.assertRaises(ValueError):
            make_finder("float64")

    def test_forbidden_endpoint_snapped(self):
        finder = make_finder("float32")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = finder.find_route(source=(15.0, 6.0),  # on the barrier
                                     target=(28.5, 6.0))
        self.assertGreater(len(path.path_indices), 0)
        self.assertTrue(any("corrected" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()

"""Phase 6 tests: the multi-metric path evaluator (reporting only).

Covers hand-computed totals, the feasibility identity/parity, category
breakdowns, and the end-to-end honest reporting through PathFinder.
"""
import unittest

import numpy as np
from rasterio.transform import from_origin

from pyorps import PathFinder
from pyorps.core.objective import Objective
from pyorps.utils.metric_edges import construct_edges_gradient
from pyorps.utils.metric_eval import evaluate_path_metrics
from pyorps.utils.neighborhood import get_neighborhood_steps

CRS = "EPSG:25832"


class TestEvaluatorUnits(unittest.TestCase):
    def test_hand_computed_straight_path(self):
        """3-cell straight path, 1 m cells, no DEM: totals by hand."""
        cost = np.array([[10.0, 20.0, 40.0]], dtype=np.float32)
        landscape = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
        rows = np.array([0, 0, 0])
        cols = np.array([0, 1, 2])
        obj = Objective({"cost": 1.0, "landscape": 100.0})
        ev = evaluate_path_metrics(
            rows, cols, {"cost": cost, "landscape": landscape}, obj,
            cell_size=1.0)
        # segment means: (10+20)/2=15, (20+40)/2=30 => cost total 45
        self.assertAlmostEqual(ev.metrics["cost"], 45.0, places=5)
        self.assertAlmostEqual(ev.metrics["landscape"], 1.0, places=5)
        self.assertAlmostEqual(ev.metrics["length"], 2.0, places=5)
        self.assertAlmostEqual(ev.total_length_2d, 2.0, places=5)
        self.assertEqual(ev.max_gradient_pct, 0.0)

    def test_feasibility_identity_without_responses(self):
        """No DEM, no responses: feasibility == sum(w_k * M_k) exactly."""
        rng = np.random.default_rng(9)
        cost = rng.random((6, 8)).astype(np.float32) * 100
        noise = rng.random((6, 8)).astype(np.float32)
        rows = np.array([2, 2, 3, 4, 4])
        cols = np.array([1, 2, 3, 4, 5])
        obj = Objective({"cost": 2.0, "noise": 50.0, "length": 7.0})
        ev = evaluate_path_metrics(
            rows, cols, {"cost": cost, "noise": noise}, obj, cell_size=1.0)
        expected = (2.0 * ev.metrics["cost"] + 50.0 * ev.metrics["noise"]
                    + 7.0 * ev.metrics["length"])
        self.assertAlmostEqual(ev.feasibility, expected, places=3)

    def test_dem_lengths_and_gradient(self):
        """Known height steps: 3D lengths and grade statistics by hand."""
        cost = np.full((1, 3), 10.0, dtype=np.float32)
        dem = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)  # 100% then 0%
        rows = np.array([0, 0, 0])
        cols = np.array([0, 1, 2])
        obj = Objective({"cost": 1.0})
        ev = evaluate_path_metrics(
            rows, cols, {"cost": cost}, obj, cell_size=1.0, dem=dem)
        self.assertAlmostEqual(ev.total_length_2d, 2.0, places=5)
        self.assertAlmostEqual(ev.total_length_3d,
                               np.sqrt(2.0) + 1.0, places=5)
        self.assertAlmostEqual(ev.max_gradient_pct, 100.0, places=3)
        # exposure = 100% * sqrt(2) m + 0% * 1 m
        self.assertAlmostEqual(ev.metrics["gradient"],
                               100.0 * np.sqrt(2.0), places=3)
        # metrics use the true 3D basis
        self.assertAlmostEqual(ev.metrics["cost"],
                               10.0 * (np.sqrt(2.0) + 1.0), places=3)

    def test_intermediate_cells_in_segment_mean(self):
        """A knight step averages over source + 2 intermediates + target."""
        cost = np.zeros((3, 3), dtype=np.float32)
        cost[0, 0] = 10.0   # source
        cost[2, 1] = 20.0   # target
        cost[1, 0] = 100.0  # intermediate (floor)
        cost[1, 1] = 200.0  # intermediate (ceil)
        rows = np.array([0, 2])
        cols = np.array([0, 1])
        obj = Objective({"cost": 1.0})
        ev = evaluate_path_metrics(rows, cols, {"cost": cost}, obj,
                                   cell_size=1.0)
        mean = (10.0 + 20.0 + 100.0 + 200.0) / 4.0
        self.assertAlmostEqual(ev.metrics["cost"],
                               mean * np.sqrt(5.0), places=3)

    def test_category_breakdown(self):
        cost = np.full((1, 4), 5.0, dtype=np.float32)
        category = np.array([[1, 1, 2, 2]], dtype=np.uint16)
        rows = np.array([0, 0, 0, 0])
        cols = np.array([0, 1, 2, 3])
        obj = Objective({"cost": 1.0})
        ev = evaluate_path_metrics(
            rows, cols, {"cost": cost}, obj, cell_size=1.0,
            category=category, category_labels={1: "Forest", 2: "Field"})
        self.assertAlmostEqual(sum(ev.length_by_class.values()),
                               ev.total_length_3d, places=5)
        self.assertAlmostEqual(ev.length_by_class["Forest"], 1.5, places=5)
        self.assertAlmostEqual(ev.length_by_class["Field"], 1.5, places=5)

    def test_feasibility_matches_search_weights(self):
        """feasibility == the summed gradient edge weights (user units)."""
        N = 21
        steps = get_neighborhood_steps("r1", directed=True)
        W = np.full((N, N), 100, dtype=np.uint16)
        dem = (np.arange(N, dtype=np.float32)[:, None] * 0.5
               * np.ones((1, N), dtype=np.float32))
        obj = Objective({"cost": 1.0, "gradient": 10.0},
                        gradient_options={"multiplier": "exponential"})
        luts = obj.build_gradient_luts(steps, cell_size=1.0, quant_scale=1.0)
        from_n, to_n, costs = construct_edges_gradient(W, dem, steps, luts)
        weight_of = {(int(u), int(v)): float(c)
                     for u, v, c in zip(from_n, to_n, costs)}

        # an arbitrary monotone staircase path
        rows = np.array([2, 3, 4, 5, 5, 6])
        cols = np.array([2, 3, 3, 4, 5, 6])
        idx = rows * N + cols
        search_weight = sum(weight_of[(int(a), int(b))]
                            for a, b in zip(idx[:-1], idx[1:]))

        ev = evaluate_path_metrics(
            rows, cols, {"cost": W.astype(np.float32)}, obj,
            cell_size=1.0, dem=dem)
        self.assertAlmostEqual(ev.feasibility / search_weight, 1.0,
                               places=4)


class TestPathFinderReporting(unittest.TestCase):
    def _corridor(self, objective):
        import geopandas as gpd
        from shapely.geometry import Polygon
        assumptions = {"landuse": {
            "neutral": {"cost": 50.0, "landscape": 0.0},
            "protected_cheap": {"cost": 10.0, "landscape": 1.0},
            "open_expensive": {"cost": 200.0, "landscape": 0.0},
            "barrier": 65535,
        }}
        polys = {
            "neutral": [Polygon([(0, 0), (3, 0), (3, 12), (0, 12)]),
                        Polygon([(27, 0), (30, 0), (30, 12), (27, 12)])],
            "protected_cheap": [Polygon([(3, 6), (27, 6), (27, 12), (3, 12)])],
            "open_expensive": [Polygon([(3, 0), (27, 0), (27, 4), (3, 4)])],
            "barrier": [Polygon([(3, 4), (27, 4), (27, 6), (3, 6)])],
        }
        records = [(k, g) for k, geoms in polys.items() for g in geoms]
        gdf = gpd.GeoDataFrame({"landuse": [r[0] for r in records],
                                "geometry": [r[1] for r in records]},
                               crs=CRS)
        finder = PathFinder(
            dataset_source=gdf, source_coords=(1.5, 6.0),
            target_coords=(28.5, 6.0), search_space_buffer_m=50,
            graph_api="cython", cost_assumptions=assumptions,
            objective=objective, resolution_in_m=1.0)
        return finder, finder.find_route()

    def test_all_metrics_reported_for_both_variants(self):
        _, cheap = self._corridor({"cost": 1.0})
        _, protective = self._corridor({"cost": 1.0, "landscape": 1000.0})
        for path in (cheap, protective):
            self.assertIn("cost", path.metrics)
            self.assertIn("landscape", path.metrics)
            self.assertIn("length", path.metrics)
            self.assertIsNotNone(path.feasibility)
            self.assertGreater(path.feasibility, 0)
        # honest trade-off: the protective route pays more EUR for less
        # landscape exposure
        self.assertGreater(protective.metrics["cost"],
                           cheap.metrics["cost"])
        self.assertLess(protective.metrics["landscape"],
                        cheap.metrics["landscape"])
        self.assertGreater(cheap.metrics["landscape"], 0.0)
        self.assertEqual(protective.metrics["landscape"], 0.0)

    def test_class_breakdown_and_analyze(self):
        _, path = self._corridor({"cost": 1.0})
        self.assertTrue(path.length_by_class)
        self.assertIn("protected_cheap", path.length_by_class)
        self.assertAlmostEqual(sum(path.length_by_class.values()),
                               path.total_length_3d, places=3)
        report = path.analyze()
        self.assertIn("Feasibility", report)
        self.assertIn("landscape", report)
        self.assertIn("Feature-class breakdown", report)

    def test_geodataframe_columns(self):
        finder, _ = self._corridor({"cost": 1.0})
        gdf = finder.create_path_geodataframe()
        for column in ("feasibility", "metric_cost", "metric_landscape",
                       "metric_length", "objective_weights"):
            self.assertIn(column, gdf.columns)

    def test_3d_length_reported_with_dem(self):
        """Objective + DEM: total_length is the 3D length (2D retained)."""
        N = 41
        dem = np.zeros((N, N), dtype=np.float32)
        dem[:, 15:25] = 8.0  # plateau crossing the route
        finder = PathFinder(
            dataset_source=np.full((N, N), 100, dtype=np.uint16),
            crs=CRS, transform=from_origin(0.0, float(N), 1.0, 1.0),
            source_coords=(2.5, 20.5), target_coords=(38.5, 20.5),
            search_space_buffer_m=200, graph_api="cython",
            dem=dem, objective={"cost": 1.0})
        path = finder.find_route()
        self.assertIsNotNone(path.total_length_3d)
        self.assertGreater(path.total_length_3d, path.total_length_2d)
        self.assertAlmostEqual(path.total_length, path.total_length_3d,
                               places=5)
        self.assertGreater(path.max_gradient_pct, 0.0)
        self.assertIn("gradient", path.metrics)

    def test_legacy_mode_untouched(self):
        """Without an objective none of the new fields appear."""
        finder = PathFinder(
            dataset_source=np.full((20, 20), 50, dtype=np.uint16),
            crs=CRS, transform=from_origin(0.0, 20.0, 1.0, 1.0),
            source_coords=(2.5, 10.5), target_coords=(17.5, 10.5),
            search_space_buffer_m=100, graph_api="cython")
        path = finder.find_route()
        self.assertIsNone(path.metrics)
        self.assertIsNone(path.feasibility)
        self.assertIsNone(path.length_by_class)


if __name__ == "__main__":
    unittest.main()

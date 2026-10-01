"""Phase 8 tests: feasibility objectives on constrained routing.

The gate: without an objective everything stays bit-identical (legacy LUT
path); with the default objective the per-cell tower raster must reproduce
the LUT results exactly.
"""
import unittest
import warnings

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

CRS = "EPSG:25832"
N = 30
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)

PROFILE = dict(
    name="test_ohl", description="test overhead line",
    soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
    angle_cost_function="linear", angle_cost_params={"scale": 10.0},
    min_span_m=3.0, max_span_m=8.0, span_bin_size_m=1.0,
    tower_cost_params={
        "terrain_cost_map": {0: 100, 100: 500, 500: 2000},
        "angle_types": {
            "suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
            "angle": {"max_angle_deg": 90.0, "base_cost": 200.0},
        },
    },
)


def make_finder(objective=None, raster=None, **kwargs):
    kwargs.setdefault("graph_api", "cython")
    return ConstrainedPathFinder(
        dataset_source=(raster if raster is not None
                        else np.full((N, N), 100, dtype=np.uint16)),
        crs=CRS, transform=TRANSFORM,
        source_coords=(3.5, 15.5), target_coords=(26.5, 15.5),
        search_space_buffer_m=100,
        profile=dict(PROFILE),
        objective=objective,
        **kwargs,
    )


class TestLegacyEquivalence(unittest.TestCase):
    def test_default_objective_bit_identical(self):
        """objective={'cost': 1.0} (per-cell tower raster) reproduces the
        legacy value-keyed LUT routing exactly."""
        legacy = make_finder().find_route()
        objective = make_finder(objective={"cost": 1.0}).find_route()
        self.assertEqual(list(legacy.path_indices),
                         list(objective.path_indices))
        self.assertEqual([t.cell_index for t in legacy.towers],
                         [t.cell_index for t in objective.towers])
        self.assertEqual(legacy.n_towers, objective.n_towers)

    def test_objective_mode_reports_metrics(self):
        path = make_finder(objective={"cost": 1.0}).find_route()
        self.assertIsNotNone(path.metrics)
        self.assertIn("cost", path.metrics)
        self.assertIsNotNone(path.feasibility)

    def test_legacy_mode_reports_no_metrics(self):
        path = make_finder().find_route()
        self.assertIsNone(path.metrics)
        self.assertIsNone(path.feasibility)


class TestReservedWeights(unittest.TestCase):
    def test_smoothness_zero_clears_angle_lut(self):
        finder = make_finder(objective={"cost": 1.0, "smoothness": 0.0})
        finite = np.isfinite(finder._angle_cost_lut)
        self.assertTrue(np.all(finder._angle_cost_lut[finite] == 0.0))
        # hard limits survive as inf
        self.assertTrue(np.isinf(finder._angle_cost_lut[~finite]).all())
        finder.find_route()  # still routes

    def test_tower_weight_scales_luts(self):
        base = make_finder(objective={"cost": 1.0})
        heavy = make_finder(objective={"cost": 1.0, "tower": 3.0})
        np.testing.assert_allclose(
            heavy._tower_terrain_costs, base._tower_terrain_costs * 3.0)
        np.testing.assert_allclose(
            heavy._tower_angle_costs, base._tower_angle_costs * 3.0)

    def test_tower_weight_never_increases_tower_count(self):
        cheap_towers = make_finder(
            objective={"cost": 1.0, "tower": 0.01}).find_route().n_towers
        pricey_towers = make_finder(
            objective={"cost": 1.0, "tower": 5.0}).find_route().n_towers
        self.assertLessEqual(pricey_towers, cheap_towers)


class TestMultiMetricConstrained(unittest.TestCase):
    def _finder(self, objective):
        assumptions = {"landuse": {
            "neutral": {"cost": 50.0, "landscape": 0.0},
            "protected_cheap": {"cost": 10.0, "landscape": 1.0},
            "open_expensive": {"cost": 200.0, "landscape": 0.0},
            "barrier": 65535,
        }}
        polys = {
            "neutral": [Polygon([(0, 0), (3, 0), (3, 24), (0, 24)]),
                        Polygon([(27, 0), (30, 0), (30, 24), (27, 24)])],
            "protected_cheap": [
                Polygon([(3, 12), (27, 12), (27, 24), (3, 24)])],
            "open_expensive": [
                Polygon([(3, 0), (27, 0), (27, 8), (3, 8)])],
            "barrier": [Polygon([(3, 8), (27, 8), (27, 12), (3, 12)])],
        }
        records = [(k, g) for k, geoms in polys.items() for g in geoms]
        gdf = gpd.GeoDataFrame({"landuse": [r[0] for r in records],
                                "geometry": [r[1] for r in records]},
                               crs=CRS)
        return ConstrainedPathFinder(
            dataset_source=gdf,
            source_coords=(1.5, 12.0), target_coords=(28.5, 12.0),
            search_space_buffer_m=50,
            profile=dict(PROFILE), graph_api="cython",
            cost_assumptions=assumptions, objective=objective,
            resolution_in_m=1.0,
        )

    def test_landscape_weight_flips_constrained_route(self):
        cheap = self._finder({"cost": 1.0}).find_route()
        protective = self._finder(
            {"cost": 1.0, "landscape": 5000.0}).find_route()
        self.assertGreater(cheap.metrics["landscape"], 0.0)
        self.assertEqual(protective.metrics["landscape"], 0.0)
        self.assertGreater(protective.metrics["cost"],
                           cheap.metrics["cost"])
        # both are proper constrained results with towers
        self.assertGreater(cheap.n_towers, 0)
        self.assertGreater(protective.n_towers, 0)
        self.assertTrue(cheap.length_by_class)


class TestGuards(unittest.TestCase):
    def test_gradient_objective_rejected(self):
        # without a DEM the parent's gradient-requires-DEM check fires
        with self.assertRaises(ValueError):
            make_finder(objective={"cost": 1.0, "gradient": 5.0})
        # with a DEM, the constrained-specific guard fires
        dem = np.zeros((N, N), dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            make_finder(objective={"cost": 1.0, "gradient": 5.0}, dem=dem)

    def test_non_cython_backend_rejected(self):
        with self.assertRaises(NotImplementedError):
            make_finder(objective={"cost": 1.0}, graph_api="cython_parallel")

    def test_clearance_plus_objective_rejected(self):
        profile = dict(PROFILE)
        profile.update(tower_height_m=25.0, conductor_weight_per_m=15.0,
                       conductor_tension_n=25000.0, min_clearance_m=6.0)
        with self.assertRaises(NotImplementedError):
            ConstrainedPathFinder(
                dataset_source=np.full((N, N), 100, dtype=np.uint16),
                crs=CRS, transform=TRANSFORM,
                source_coords=(3.5, 15.5), target_coords=(26.5, 15.5),
                search_space_buffer_m=100, profile=profile,
                graph_api="cython", objective={"cost": 1.0})

    def test_dem_plus_objective_warns(self):
        dem = np.zeros((N, N), dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_finder(objective={"cost": 1.0}, dem=dem)
        self.assertTrue(any("profile's gradient model" in str(w.message)
                            for w in caught))

    def test_legacy_without_objective_unaffected_by_guards(self):
        path = make_finder(graph_api="cython").find_route()
        self.assertGreater(len(path.path_indices), 0)


if __name__ == "__main__":
    unittest.main()

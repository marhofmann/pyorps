"""End-to-end tests for PathFinder with a feasibility objective (Phase 3).

Covers: byte-identical legacy equivalence under the default objective,
shortest-path vs cheapest-path divergence, the cheap-but-protected corridor
scenario (inexpressible before this feature), set_objective rerouting, and
the Phase-3 guards.
"""
import unittest

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from pyorps import PathFinder
from pyorps.core.objective import Objective

CRS = "EPSG:25832"
TRANSFORM = from_origin(0.0, 30.0, 1.0, 1.0)  # 1 m cells, 30 rows tall


def detour_raster():
    """Uniform cost 10 with an expensive (but passable) vertical wall.

    The wall (cost 500) has a cheap gap at the top, far from the straight
    line between source and target: the cost-optimal route detours through
    the gap, the shortest route crosses the wall.
    """
    raster = np.full((30, 40), 10, dtype=np.uint16)
    raster[:, 19:21] = 500       # vertical wall, columns 19-20
    raster[0:2, 19:21] = 10      # gap at the top rows
    return raster


SOURCE = (5.5, 15.5)   # row 14, col 5
TARGET = (35.5, 15.5)  # row 14, col 35


def make_finder(objective=None, raster=None, **kwargs):
    return PathFinder(
        dataset_source=raster if raster is not None else detour_raster(),
        crs=CRS,
        transform=TRANSFORM,
        source_coords=SOURCE,
        target_coords=TARGET,
        search_space_buffer_m=100,
        graph_api="cython",
        objective=objective,
        **kwargs,
    )


class TestLegacyEquivalence(unittest.TestCase):
    def test_default_objective_reproduces_legacy_route(self):
        legacy = make_finder(objective=None).find_route()
        explicit = make_finder(objective={"cost": 1.0}).find_route()
        self.assertEqual(list(legacy.path_indices),
                         list(explicit.path_indices))
        self.assertEqual(legacy.total_cost, explicit.total_cost)

    def test_passthrough_and_spec(self):
        finder = make_finder(objective={"cost": 1.0})
        path = finder.find_route()
        self.assertTrue(finder._combine_result.legacy_passthrough)
        self.assertIsNotNone(path.objective_spec)
        self.assertEqual(path.objective_spec["weights"], {"cost": 1.0})
        self.assertTrue(path.objective_spec["legacy_passthrough"])

    def test_no_objective_has_no_spec(self):
        path = make_finder().find_route()
        self.assertIsNone(path.objective_spec)


class TestObjectiveSemantics(unittest.TestCase):
    def test_shortest_vs_cheapest(self):
        cheapest = make_finder(objective={"cost": 1.0}).find_route()
        shortest = make_finder(objective={"length": 1.0}).find_route()

        # cost-optimal detours through the gap => longer but cheaper cells
        self.assertGreater(cheapest.total_length, shortest.total_length)
        self.assertNotEqual(list(cheapest.path_indices),
                            list(shortest.path_indices))
        # the shortest route is essentially the straight line
        self.assertAlmostEqual(shortest.total_length,
                               shortest.euclidean_distance, delta=1.0)

    def test_set_objective_reroutes_same_finder(self):
        finder = make_finder(objective={"cost": 1.0})
        cheapest = finder.find_route()
        finder.set_objective(Objective.shortest())
        shortest = finder.find_route()
        self.assertNotEqual(list(cheapest.path_indices),
                            list(shortest.path_indices))
        self.assertGreater(cheapest.total_length, shortest.total_length)
        # both results retained in the collection with distinct ids
        self.assertEqual(len(finder.paths), 2)

    def test_per_call_objective_override(self):
        finder = make_finder(objective={"cost": 1.0})
        shortest = finder.find_route(objective={"length": 1.0})
        self.assertAlmostEqual(shortest.total_length,
                               shortest.euclidean_distance, delta=1.0)


class TestVectorMetricPipeline(unittest.TestCase):
    """Two corridors: cheap-but-protected vs expensive-but-unprotected."""

    def setUp(self):
        # 30 m x 12 m extent: corridor A (rows 0-5), barrier (6-7, forbidden),
        # corridor B (8-11). Source/target sit in a shared neutral column at
        # both ends spanning the full height.
        self.assumptions = {
            "landuse": {
                "neutral": {"cost": 50.0, "landscape": 0.0},
                "protected_cheap": {"cost": 10.0, "landscape": 1.0},
                "open_expensive": {"cost": 200.0, "landscape": 0.0},
                "barrier": 65535,
            }
        }
        polys = {
            "neutral": [Polygon([(0, 0), (3, 0), (3, 12), (0, 12)]),
                        Polygon([(27, 0), (30, 0), (30, 12), (27, 12)])],
            "protected_cheap": [Polygon([(3, 6), (27, 6), (27, 12), (3, 12)])],
            "open_expensive": [Polygon([(3, 0), (27, 0), (27, 4), (3, 4)])],
            "barrier": [Polygon([(3, 4), (27, 4), (27, 6), (3, 6)])],
        }
        records = [(name, geom) for name, geoms in polys.items()
                   for geom in geoms]
        self.gdf = gpd.GeoDataFrame(
            {"landuse": [r[0] for r in records],
             "geometry": [r[1] for r in records]}, crs=CRS)
        self.source = (1.5, 6.0)
        self.target = (28.5, 6.0)

    def _finder(self, objective):
        return PathFinder(
            dataset_source=self.gdf,
            source_coords=self.source,
            target_coords=self.target,
            search_space_buffer_m=50,
            graph_api="cython",
            cost_assumptions=self.assumptions,
            objective=objective,
            resolution_in_m=1.0,
        )

    def _corridor_of(self, finder, path):
        """Classify the route by its landscape exposure along the stack."""
        stack = finder.metric_stack
        handler = finder.raster_handler
        cols = handler.data[0].shape[1]
        window = handler.window
        rows_idx = np.array(path.path_indices) // cols + int(window.row_off)
        cols_idx = np.array(path.path_indices) % cols + int(window.col_off)
        return float(stack["landscape"][rows_idx, cols_idx].sum())

    def test_cheapest_takes_protected_corridor(self):
        finder = self._finder({"cost": 1.0})
        path = finder.find_route()
        self.assertGreater(self._corridor_of(finder, path), 10.0)

    def test_landscape_weight_flips_the_route(self):
        finder = self._finder({"cost": 1.0, "landscape": 1000.0})
        path = finder.find_route()
        self.assertEqual(self._corridor_of(finder, path), 0.0)

    def test_multi_metric_without_objective_stays_legacy(self):
        finder = PathFinder(
            dataset_source=self.gdf,
            source_coords=self.source,
            target_coords=self.target,
            search_space_buffer_m=50,
            graph_api="cython",
            cost_assumptions=self.assumptions,
            resolution_in_m=1.0,
        )
        path = finder.find_route()
        self.assertIsNone(finder.metric_stack)
        self.assertIsNone(path.objective_spec)


class TestPhase3Guards(unittest.TestCase):
    def test_gradient_terms_require_dem(self):
        """Since Phase 4, gradient terms are supported on CPU backends —
        but they still require a DEM (no dem= here => ValueError)."""
        with self.assertRaises(ValueError):
            make_finder(objective={"cost": 1.0, "gradient": 10.0})
        with self.assertRaises(ValueError):
            make_finder(objective={"cost": 1.0},
                        gradient_options={"multiplier": "exponential"})
        with self.assertRaises(ValueError):
            make_finder(objective={"cost": 1.0},
                        gradient_options={"max_gradient_pct": 30.0})

    def test_gradient_options_require_objective(self):
        with self.assertRaises(ValueError):
            make_finder(gradient_options={"bin_width_pct": 0.5})

    def test_set_objective_requires_metric_pipeline(self):
        finder = make_finder()  # no objective at construction
        with self.assertRaises(ValueError):
            finder.set_objective({"length": 1.0})

    def test_modifiers_not_yet_supported(self):
        with self.assertRaises(NotImplementedError):
            make_finder(objective={"cost": 1.0},
                        datasets_to_modify=[{"input_data": "x.gpkg"}])


if __name__ == "__main__":
    unittest.main()

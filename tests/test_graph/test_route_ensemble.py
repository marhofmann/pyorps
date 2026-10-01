"""Phase 7 tests: route ensembles, Pareto filtering, compare_optimal."""
import unittest

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from pyorps import PathFinder
from pyorps.core.ensemble import EnsembleError, RouteEnsemble
from pyorps.core.objective import Objective
from pyorps.core.path import Path

CRS = "EPSG:25832"


def fake_path(name_id, metrics, indices):
    """Minimal Path carrying only what the ensemble needs."""
    return Path(
        source=(0.0, 0.0), target=(1.0, 1.0), algorithm="dijkstra",
        graph_api="cython", path_indices=list(indices),
        path_coords=[(0.0, 0.0), (1.0, 1.0)],
        path_geometry=LineString([(0, 0), (1, 1)]),
        euclidean_distance=1.0, runtimes={"shortest_path": 0.01},
        path_id=name_id, search_space_buffer_m=0.0, neighborhood="r1",
        metrics=metrics, feasibility=1.0,
        objective_spec={"weights": {"cost": 1.0}},
    )


class TestRouteEnsembleUnit(unittest.TestCase):
    def test_add_and_lookup(self):
        ens = RouteEnsemble()
        ens.add("a", fake_path(0, {"cost": 1.0}, [1, 2]))
        self.assertIn("a", ens)
        self.assertEqual(len(ens), 1)
        with self.assertRaises(EnsembleError):
            ens.add("a", fake_path(1, {"cost": 2.0}, [1, 3]))
        with self.assertRaises(EnsembleError):
            _ = ens["missing"]

    def test_dataframe_and_duplicate_marking(self):
        ens = RouteEnsemble()
        ens.add("a", fake_path(0, {"cost": 10.0, "landscape": 5.0}, [1, 2]))
        ens.add("b", fake_path(1, {"cost": 20.0, "landscape": 0.0}, [1, 3]))
        ens.add("c", fake_path(2, {"cost": 10.0, "landscape": 5.0}, [1, 2]))
        table = ens.to_dataframe()
        self.assertEqual(list(table.index), ["a", "b", "c"])
        self.assertEqual(table.loc["a", "cost"], 10.0)
        self.assertEqual(table.loc["a", "same_route_as"], "")
        self.assertEqual(table.loc["c", "same_route_as"], "a")

    def test_pareto_front(self):
        ens = RouteEnsemble()
        ens.add("cheap", fake_path(0, {"cost": 10.0, "landscape": 5.0},
                                   [1, 2]))
        ens.add("green", fake_path(1, {"cost": 20.0, "landscape": 0.0},
                                   [1, 3]))
        ens.add("bad", fake_path(2, {"cost": 25.0, "landscape": 5.0},
                                 [1, 4]))     # dominated by both
        ens.add("twin", fake_path(3, {"cost": 10.0, "landscape": 5.0},
                                  [1, 5]))    # exact metric tie with cheap
        front = ens.pareto_front(["cost", "landscape"])
        self.assertEqual(front.names, ["cheap", "green"])
        with self.assertRaises(EnsembleError):
            ens.pareto_front(["cost", "duration"])   # missing metric
        with self.assertRaises(EnsembleError):
            ens.pareto_front([])


class TestFindRouteEnsemble(unittest.TestCase):
    def _finder(self):
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
        return PathFinder(
            dataset_source=gdf, source_coords=(1.5, 6.0),
            target_coords=(28.5, 6.0), search_space_buffer_m=50,
            graph_api="cython", cost_assumptions=assumptions,
            objective={"cost": 1.0}, resolution_in_m=1.0)

    def test_ensemble_run_and_pareto(self):
        finder = self._finder()
        ensemble = finder.find_route_ensemble({
            "cheapest": Objective.cheapest(),
            "shortest": Objective.shortest(),
            "protective": {"cost": 1.0, "landscape": 1000.0},
        })
        self.assertEqual(ensemble.names,
                         ["cheapest", "shortest", "protective"])
        table = ensemble.to_dataframe()
        self.assertIn("cost", table.columns)
        self.assertIn("landscape", table.columns)
        self.assertEqual(table.loc["protective", "landscape"], 0.0)
        self.assertGreater(table.loc["cheapest", "landscape"], 0.0)
        self.assertLess(table.loc["cheapest", "cost"],
                        table.loc["protective", "cost"])

        front = ensemble.pareto_front(["cost", "landscape"])
        self.assertIn("cheapest", front.names)
        self.assertIn("protective", front.names)

        # the active objective is restored afterwards
        self.assertEqual(finder.objective.weights, {"cost": 1.0})
        # all variant paths are retained in the collection
        self.assertEqual(len(finder.paths), 3)

    def test_ensemble_with_list_and_kwarg_names(self):
        finder = self._finder()
        ensemble = finder.find_route_ensemble(
            [{"cost": 1.0}, {"length": 1.0}])
        self.assertEqual(ensemble.names, ["variant_0", "variant_1"])

    def test_ensemble_requires_metric_pipeline(self):
        finder = PathFinder(
            dataset_source=np.full((20, 20), 50, dtype=np.uint16),
            crs=CRS, transform=from_origin(0.0, 20.0, 1.0, 1.0),
            source_coords=(2.5, 10.5), target_coords=(17.5, 10.5),
            search_space_buffer_m=100, graph_api="cython")
        with self.assertRaises(ValueError):
            finder.find_route_ensemble({"a": {"cost": 1.0}})

    def test_ensemble_rejects_multi_target(self):
        finder = self._finder()
        with self.assertRaises(EnsembleError):
            finder.find_route_ensemble(
                {"a": {"cost": 1.0}},
                target=[(28.5, 6.0), (28.5, 8.0)])
        # objective restored even after the failure
        self.assertEqual(finder.objective.weights, {"cost": 1.0})

    def test_compare_optimal_deltas(self):
        finder = self._finder()
        finder.set_objective({"cost": 1.0, "landscape": 1000.0})
        table = finder.compare_optimal(("cost",))
        self.assertIn("current", table.index)
        self.assertIn("cost-optimal", table.index)
        delta_row = "delta current - cost-optimal"
        self.assertIn(delta_row, table.index)
        # the protective policy pays more EUR ...
        self.assertGreater(table.loc[delta_row, "cost"], 0.0)
        # ... and buys less landscape exposure
        self.assertLess(table.loc[delta_row, "landscape"], 0.0)


if __name__ == "__main__":
    unittest.main()

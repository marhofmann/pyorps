"""Phase 0 items 0.10 / 0.11 / 0.4a of the 2026-08-07 performance plan.

Covers the library-backend size guard, the honest report of the missing GPU
edge builder, the Path.total_cost_basis disclosure and the runtime phase
attribution (graph construction must not be charged to "shortest_path").
"""

import unittest
import warnings
from time import sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from rasterio.transform import from_origin

from pyorps.graph.api import graph_library_api
from pyorps.graph.api.graph_library_api import (
    GraphLibraryAPI,
    check_library_backend_size,
)
from pyorps.graph.path_finder import RUNTIME_PHASES, PathFinder
from pyorps.io.geo_dataset import RasterDataset


class _ConcreteLibraryAPI(GraphLibraryAPI):
    """Minimal concrete GraphLibraryAPI for guard/dispatch tests."""

    def create_graph(self, from_nodes, to_nodes, cost=None, **kwargs):
        return "graph"

    def get_number_of_nodes(self):
        return 0

    def get_number_of_edges(self):
        return 0

    def remove_isolates(self):
        pass

    def get_nodes(self):
        return []

    def _compute_single_path(self, source, target, algorithm, **kwargs):
        return []

    def _compute_single_source_multiple_targets(self, source, targets,
                                                algorithm, **kwargs):
        return []

    def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
        return []


def _make_api(**kwargs):
    raster = np.ones((5, 5), dtype=np.uint16)
    steps = np.array([[0, 1], [1, 0]], dtype=np.int8)
    return _ConcreteLibraryAPI(
        raster, steps,
        from_nodes=np.array([0], dtype=np.uint32),
        to_nodes=np.array([1], dtype=np.uint32),
        cost=np.array([1.0]),
        **kwargs)


class TestLibraryBackendSizeGuard(unittest.TestCase):
    """0.10 (b): refuse edge-list backends above MAX_LIBRARY_BACKEND_CELLS."""

    def test_below_limit_passes(self):
        check_library_backend_size(1_000_000, 8, "NetworkitAPI")

    def test_above_limit_raises_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            check_library_backend_size(25_000_000, 8, "NetworkitAPI")
        message = str(ctx.exception)
        self.assertIn("NetworkitAPI", message)
        # names the raster-direct alternatives
        self.assertIn("cython", message)
        self.assertIn("raster_gpu", message)
        self.assertIn("raster_fim", message)
        # names the float32 trap and both override mechanisms
        self.assertIn("float32", message)
        self.assertIn("allow_large_raster", message)
        self.assertIn("MAX_LIBRARY_BACKEND_CELLS", message)

    def test_instance_override(self):
        check_library_backend_size(25_000_000, 8, "NetworkitAPI",
                                   allow_large_raster=True)

    def test_module_flag_override(self):
        with patch.object(graph_library_api, "MAX_LIBRARY_BACKEND_CELLS",
                          None):
            check_library_backend_size(10 ** 9, 8, "NetworkitAPI")

    def test_guard_is_wired_into_the_api_constructor(self):
        with patch.object(graph_library_api, "MAX_LIBRARY_BACKEND_CELLS", 8):
            with self.assertRaises(ValueError):
                _make_api()

    def test_guard_bypassed_by_kwarg_in_constructor(self):
        with patch.object(graph_library_api, "MAX_LIBRARY_BACKEND_CELLS", 8):
            api = _make_api(allow_large_raster=True)
        self.assertEqual(api.graph, "graph")


class TestGpuEdgeConstructionIsNotImplemented(unittest.TestCase):
    """0.10 (a): the GPU edge builder does not exist — say so honestly."""

    def test_traversal_gpu_has_no_edge_builder(self):
        import pyorps.utils.traversal_gpu as traversal_gpu
        self.assertFalse(hasattr(traversal_gpu, "construct_edges_gpu"),
                         "traversal_gpu gained an edge builder — remove the "
                         "not-implemented branch in _construct_edges_gpu")

    def test_warns_not_implemented_and_falls_back(self):
        api = _make_api()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = api._construct_edges_gpu(None)
        self.assertEqual(result, (None, None, None))
        messages = " ".join(str(w.message) for w in caught)
        self.assertIn("not implemented", messages)
        self.assertNotIn("cupy not installed", messages)


class TestTotalCostBasis(unittest.TestCase):
    """0.11: total_cost keeps its formula; the mismatch becomes visible."""

    @staticmethod
    def _finder(graph_api_name="cython", dem=None, objective=None,
                combine_result=None):
        finder = PathFinder.__new__(PathFinder)
        finder.graph_api_name = graph_api_name
        finder.dem_raster_handler = dem
        finder.objective = objective
        finder._combine_result = combine_result
        finder._total_cost_basis_warned = False
        return finder

    def test_plain_2d_has_no_divergence(self):
        basis, divergences = self._finder()._total_cost_basis()
        self.assertIn("2D length", basis)
        self.assertEqual(divergences, [])

    def test_dem_run_reports_stretch_and_response(self):
        _, divergences = self._finder(dem=MagicMock())._total_cost_basis()
        self.assertEqual(len(divergences), 1)
        self.assertIn("3D length stretch", divergences[0])
        # without an objective the reported length is 2D as well
        self.assertIn("total_length", divergences[0])

    def test_objective_run_reports_quantized_units(self):
        finder = self._finder(objective=MagicMock(),
                              combine_result=SimpleNamespace(scale=2.5))
        _, divergences = finder._total_cost_basis()
        self.assertEqual(len(divergences), 1)
        self.assertIn("2.5", divergences[0])
        self.assertIn("not EUR", divergences[0])

    def test_raster_fim_reports_continuous_mismatch(self):
        _, divergences = self._finder("raster_fim")._total_cost_basis()
        self.assertIn("T[target]", divergences[0])

    def test_report_sets_attribute_and_warns_once(self):
        finder = self._finder(dem=MagicMock())
        first, second = SimpleNamespace(), SimpleNamespace()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finder._report_total_cost_basis(first, computed=True)
            finder._report_total_cost_basis(second, computed=True)
        self.assertEqual(len(caught), 1)
        self.assertIn("2D length", first.total_cost_basis)
        self.assertEqual(first.total_cost_basis, second.total_cost_basis)

    def test_no_warning_without_divergence(self):
        finder = self._finder()
        path = SimpleNamespace()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finder._report_total_cost_basis(path, computed=True)
        self.assertEqual(len(caught), 0)

    def test_float32_raster_reports_not_computed(self):
        finder = self._finder()
        path = SimpleNamespace()
        finder._report_total_cost_basis(path, computed=False)
        self.assertIn("not computed", path.total_cost_basis)


class TestRuntimeAccounting(unittest.TestCase):
    """0.4a: graph construction is its own phase; "total" is a real total."""

    def setUp(self):
        self.source_coords = (500000, 5600000)
        self.target_coords = (500100, 5600100)
        self.raster_data = np.ones((1, 100, 100), dtype=np.uint16)

        self.dataset = MagicMock(spec=RasterDataset)
        self.dataset.data = self.raster_data
        self.dataset.transform = from_origin(500000, 5600000, 1, 1)
        self.dataset.crs = "EPSG:32632"

    def _build_finder(self, mock_get_api, mock_handler_cls, mock_init_geo,
                      build_seconds=0.05):
        mock_init_geo.return_value = self.dataset
        handler = MagicMock()
        handler.data = self.raster_data
        handler.coords_to_indices.return_value = np.array([[0, 0], [10, 10]])
        handler.indices_to_coords.return_value = np.array([
            [500000.0, 5600000.0],
            [500001.0, 5600001.0],
            [500002.0, 5600002.0],
        ])
        mock_handler_cls.return_value = handler

        def slow_constructor(*args, **kwargs):
            sleep(build_seconds)
            api = MagicMock()
            api.edge_construction_time = build_seconds * 0.8
            api.graph_creation_time = build_seconds * 0.2
            api.shortest_path.return_value = [0, 101, 202]
            return api

        mock_get_api.return_value = MagicMock(side_effect=slow_constructor)
        return PathFinder(self.dataset, self.source_coords,
                          self.target_coords, graph_api="networkit")

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_graph_build_is_not_charged_to_shortest_path(
            self, mock_get_api, mock_handler_cls, mock_init_geo):
        finder = self._build_finder(mock_get_api, mock_handler_cls,
                                    mock_init_geo)
        with patch.object(PathFinder, "calculate_path_metrics"):
            finder.find_route()

        runtimes = finder.runtimes
        self.assertGreaterEqual(runtimes["graph_build"], 0.04)
        self.assertLess(runtimes["shortest_path"], runtimes["graph_build"])

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_total_is_the_sum_of_all_phases(
            self, mock_get_api, mock_handler_cls, mock_init_geo):
        finder = self._build_finder(mock_get_api, mock_handler_cls,
                                    mock_init_geo)
        with patch.object(PathFinder, "calculate_path_metrics"):
            path = finder.find_route()

        runtimes = finder.runtimes
        for phase in RUNTIME_PHASES:
            self.assertIn(phase, runtimes)
        self.assertAlmostEqual(
            runtimes["total"],
            sum(runtimes[phase] for phase in RUNTIME_PHASES), places=9)
        # the Path carries the completed accounting, not a pre-metrics snapshot
        self.assertEqual(path.runtimes["total"], runtimes["total"])
        self.assertIn("path_metrics", path.runtimes)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_reused_graph_costs_the_second_route_nothing(
            self, mock_get_api, mock_handler_cls, mock_init_geo):
        finder = self._build_finder(mock_get_api, mock_handler_cls,
                                    mock_init_geo)
        with patch.object(PathFinder, "calculate_path_metrics"):
            finder.find_route()
            first_total = finder.runtimes["total"]
            finder.find_route()

        self.assertEqual(finder.runtimes["graph_build"], 0.0)
        self.assertLess(finder.runtimes["total"], first_total)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_metrics_disabled_zeroes_the_phase(
            self, mock_get_api, mock_handler_cls, mock_init_geo):
        finder = self._build_finder(mock_get_api, mock_handler_cls,
                                    mock_init_geo)
        with patch.object(PathFinder, "calculate_path_metrics"):
            finder.find_route(calculate_metrics=False)
        self.assertEqual(finder.runtimes["path_metrics"], 0.0)


if __name__ == "__main__":
    unittest.main()

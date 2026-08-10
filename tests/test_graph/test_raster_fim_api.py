"""Tests for RasterFIMAPI and the PathFinder "raster_fim" backend.

The eikonal backend is *supposed* to differ from the discrete backends:
assertions compare against analytic truth or check the one-sided bound
T <= discrete cost, never bit-exactness (plan section 1).
"""
import numpy as np
import pytest

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU = True
    except Exception:
        GPU = False
except ImportError:
    GPU = False

pytestmark = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

from pyorps.core.exceptions import (           # noqa: E402
    AlgorithmNotImplementedError, NoPathFoundError)

if GPU:
    from pyorps.graph.api.raster_fim_api import RasterFIMAPI

STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)


def idx(r, c, cols):
    return r * cols + c


def make_api(raster, **kw):
    return RasterFIMAPI(raster, STEPS_8, **kw)


class TestConstruction:
    def test_basic(self):
        raster = np.ones((50, 50), dtype=np.uint16)
        api = make_api(raster)
        assert api.graph is None
        assert api.edge_construction_time == 0.0
        assert api.graph_creation_time == 0.0

    def test_dem_raises(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        dem = np.zeros((30, 30), dtype=np.float32)
        with pytest.raises(AlgorithmNotImplementedError):
            make_api(raster, dem_data=dem)

    def test_gradient_luts_raise(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        with pytest.raises(AlgorithmNotImplementedError):
            make_api(raster, gradient_luts=object())

    def test_extra_kwargs_swallowed(self):
        """PathFinder passes use_gpu/dem_kwargs to non-cython backends."""
        raster = np.ones((30, 30), dtype=np.uint16)
        api = make_api(raster, use_gpu=True, dem_kwargs=None)
        assert api is not None


class TestShortestPath:
    def test_single_to_single(self):
        raster = np.ones((60, 100), dtype=np.uint16)
        api = make_api(raster)
        path = api.shortest_path(idx(30, 10, 100), idx(30, 90, 100))
        assert path[0] == idx(30, 10, 100)
        assert path[-1] == idx(30, 90, 100)
        # straight axis line: cells stay on row 30
        assert all(i // 100 == 30 for i in path)
        assert api.last_field_costs[0] == pytest.approx(80.0, rel=1e-5)
        assert len(api.last_polylines) == 1
        poly = api.last_polylines[0]
        assert tuple(poly[0]) == (30.0, 10.0)
        assert tuple(poly[-1]) == (30.0, 90.0)

    def test_algorithm_aliases(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        api = make_api(raster)
        for alg in ("fim", "eikonal", "dijkstra", "FIM"):
            path = api.shortest_path(idx(5, 5, 30), idx(25, 25, 30),
                                     algorithm=alg)
            assert len(path) > 0

    def test_unknown_algorithm_raises(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        api = make_api(raster)
        with pytest.raises(AlgorithmNotImplementedError):
            api.shortest_path(0, 5, algorithm="astar")

    def test_no_path_raises(self):
        raster = np.ones((40, 40), dtype=np.uint16)
        raster[:, 20] = np.iinfo(np.uint16).max   # full wall
        api = make_api(raster)
        with pytest.raises(NoPathFoundError):
            api.shortest_path(idx(20, 5, 40), idx(20, 35, 40))

    def test_single_to_multi_one_solve(self):
        raster = np.ones((80, 80), dtype=np.uint16)
        api = make_api(raster)
        targets = [idx(10, 70, 80), idx(70, 70, 80), idx(70, 10, 80)]
        paths = api.shortest_path(idx(10, 10, 80), targets)
        assert len(paths) == 3
        for p, t in zip(paths, targets):
            assert p[0] == idx(10, 10, 80)
            assert p[-1] == t
        assert len(api.last_field_costs) == 3

    def test_single_to_multi_unreachable_gives_empty(self):
        raster = np.ones((40, 40), dtype=np.uint16)
        raster[10, 28:32] = np.iinfo(np.uint16).max
        raster[14, 28:32] = np.iinfo(np.uint16).max
        raster[10:15, 28] = np.iinfo(np.uint16).max
        raster[10:15, 31] = np.iinfo(np.uint16).max
        api = make_api(raster)
        enclosed = idx(12, 30, 40)
        paths = api.shortest_path(idx(20, 5, 40),
                                  [idx(20, 35, 40), enclosed])
        assert len(paths[0]) > 0
        assert paths[1] == []
        assert api.last_field_costs[1] == float("inf")
        assert api.last_polylines[1] is None

    def test_multi_to_single_symmetric(self):
        raster = np.ones((80, 80), dtype=np.uint16)
        api = make_api(raster)
        sources = [idx(10, 10, 80), idx(70, 20, 80)]
        paths = api.shortest_path(sources, idx(40, 70, 80))
        assert len(paths) == 2
        for p, s in zip(paths, sources):
            assert p[0] == s
            assert p[-1] == idx(40, 70, 80)

    def test_multi_to_multi_pairwise(self):
        raster = np.ones((60, 60), dtype=np.uint16)
        api = make_api(raster)
        sources = [idx(5, 5, 60), idx(50, 5, 60)]
        targets = [idx(5, 50, 60), idx(50, 50, 60)]
        paths = api.shortest_path(sources, targets, pairwise=True)
        assert len(paths) == 2
        assert paths[0][0] == sources[0] and paths[0][-1] == targets[0]
        assert paths[1][0] == sources[1] and paths[1][-1] == targets[1]

    def test_multi_to_multi_all_pairs(self):
        raster = np.ones((60, 60), dtype=np.uint16)
        api = make_api(raster)
        sources = [idx(5, 5, 60), idx(50, 5, 60)]
        targets = [idx(5, 50, 60), idx(50, 50, 60)]
        paths = api.shortest_path(sources, targets, pairwise=False)
        assert len(paths) == 4    # source-major order
        assert paths[0][0] == sources[0] and paths[0][-1] == targets[0]
        assert paths[1][0] == sources[0] and paths[1][-1] == targets[1]
        assert paths[2][0] == sources[1] and paths[2][-1] == targets[0]
        assert paths[3][0] == sources[1] and paths[3][-1] == targets[1]

    def test_pairwise_length_mismatch(self):
        from pyorps.core.exceptions import PairwiseError
        raster = np.ones((30, 30), dtype=np.uint16)
        api = make_api(raster)
        with pytest.raises(PairwiseError):
            api.shortest_path([0, 5], [10], pairwise=True)

    def test_cost_beats_discrete_backend(self):
        """T[target] < discrete cost for an off-lattice direction.

        At ~22 degrees (near the 8-neighborhood's worst case, +8.2%
        elongation) the discrete backend must zigzag; the eikonal field
        carries only its O(h) discretization error. On exactly
        representable directions (axes, diagonals) the discrete cost is
        exact and FIM sits marginally above it — that is expected and NOT
        tested as a violation.
        """
        from pyorps.utils.sssp_gpu import sssp_raster_gpu
        n = 200
        raster = np.full((n, n), 10, dtype=np.uint16)
        s = idx(10, 10, n)
        t = idx(80, 180, n)     # direction atan(70/170) ~ 22.4 deg
        api = make_api(raster)
        api.shortest_path(s, t)
        fim_cost = api.last_field_costs[0]
        dist = sssp_raster_gpu(raster, STEPS_8, s,
                               target_indices=np.array([t],
                                                       dtype=np.int32))
        disc = float(dist[t])
        exact = 10.0 * float(np.hypot(70, 170))
        assert fim_cost < disc, \
            f"FIM {fim_cost} not below zigzagging discrete {disc}"
        assert fim_cost == pytest.approx(exact, rel=0.02)
        # discrete pays the expected elongation on this direction
        assert disc == pytest.approx(
            10.0 * (70 * np.sqrt(2) + 100), rel=1e-3)

    def test_path_cells_adjacent_and_passable(self):
        rng = np.random.default_rng(21)
        raster = rng.integers(1, 100, (150, 150)).astype(np.uint16)
        raster[30:120, 75] = np.iinfo(np.uint16).max
        api = make_api(raster)
        path = api.shortest_path(idx(75, 10, 150), idx(75, 140, 150))
        wall = {idx(r, 75, 150) for r in range(30, 120)}
        assert not (set(path) & wall)
        rc = [(i // 150, i % 150) for i in path]
        for (r0, c0), (r1, c1) in zip(rc, rc[1:]):
            assert max(abs(r1 - r0), abs(c1 - c0)) <= 2, \
                "path jumped more than a corner-graze allows"


class TestPathFinderIntegration:
    def _make_finder(self, raster, src, tgt, **kw):
        from pyorps.graph.path_finder import PathFinder
        from rasterio.transform import from_origin
        rows = raster.shape[0]
        return PathFinder(
            dataset_source=raster,
            crs="EPSG:32632",
            transform=from_origin(0.0, float(rows), 1.0, 1.0),
            source_coords=src,
            target_coords=tgt,
            search_space_buffer_m=500,
            graph_api="raster_fim",
            **kw,
        )

    def test_factory_registration(self):
        from pyorps.graph.path_finder import get_graph_api_class
        assert get_graph_api_class("raster_fim") is RasterFIMAPI

    def test_end_to_end_route(self):
        rng = np.random.default_rng(7)
        raster = rng.integers(1, 50, (100, 100)).astype(np.uint16)
        finder = self._make_finder(raster, (10.5, 50.5), (89.5, 50.5))
        path = finder.find_route()
        assert path is not None
        assert len(path.path_coords) > 2
        assert path.graph_api == "raster_fim"
        # continuous polyline retained on the API object
        assert len(finder.graph_api.last_polylines) == 1
        assert finder.graph_api.last_field_costs[0] > 0

    def test_end_to_end_route_around_wall(self):
        raster = np.ones((100, 100), dtype=np.uint16)
        raster[0:80, 50] = np.iinfo(np.uint16).max
        finder = self._make_finder(raster, (10.5, 60.5), (90.5, 60.5))
        path = finder.find_route()
        assert len(path.path_coords) > 100  # forced detour via the gap


class TestMetricStackPipeline:
    """Plan section 5.3: MetricStack works unchanged on raster_fim.

    The FIM solver consumes the combined scalar raster exactly like the
    discrete backends — the objective steers the route and the honest
    per-criterion metrics report from the float layers. Fixture mirrors
    TestVectorMetricPipeline (test_path_finder_objective.py): corridor
    "protected_cheap" (cheap, landscape exposure 1/m) vs
    "open_expensive" (20x the cost, no exposure), hard barrier between.
    """

    ASSUMPTIONS = {
        "landuse": {
            "neutral": {"cost": 50.0, "landscape": 0.0},
            "protected_cheap": {"cost": 10.0, "landscape": 1.0},
            "open_expensive": {"cost": 200.0, "landscape": 0.0},
            "barrier": 65535,
        }
    }

    def _finder(self, objective, **kw):
        import geopandas as gpd
        from shapely.geometry import Polygon
        from pyorps.graph.path_finder import PathFinder
        polys = {
            "neutral": [Polygon([(0, 0), (3, 0), (3, 12), (0, 12)]),
                        Polygon([(27, 0), (30, 0), (30, 12), (27, 12)])],
            "protected_cheap": [
                Polygon([(3, 6), (27, 6), (27, 12), (3, 12)])],
            "open_expensive": [
                Polygon([(3, 0), (27, 0), (27, 4), (3, 4)])],
            "barrier": [Polygon([(3, 4), (27, 4), (27, 6), (3, 6)])],
        }
        records = [(name, geom) for name, geoms in polys.items()
                   for geom in geoms]
        gdf = gpd.GeoDataFrame(
            {"landuse": [r[0] for r in records],
             "geometry": [r[1] for r in records]}, crs="EPSG:25832")
        return PathFinder(
            dataset_source=gdf,
            source_coords=(1.5, 6.0),
            target_coords=(28.5, 6.0),
            search_space_buffer_m=50,
            graph_api="raster_fim",
            cost_assumptions=self.ASSUMPTIONS,
            objective=objective,
            resolution_in_m=1.0,
            **kw,
        )

    def _exposure(self, finder, path):
        stack = finder.metric_stack
        handler = finder.raster_handler
        cols = handler.data[0].shape[1]
        window = handler.window
        rows_idx = (np.array(path.path_indices) // cols
                    + int(window.row_off))
        cols_idx = (np.array(path.path_indices) % cols
                    + int(window.col_off))
        return float(stack["landscape"][rows_idx, cols_idx].sum())

    def test_cheapest_takes_protected_corridor(self):
        finder = self._finder({"cost": 1.0})
        path = finder.find_route()
        assert self._exposure(finder, path) > 10.0

    def test_landscape_weight_flips_the_route(self):
        finder = self._finder({"cost": 1.0, "landscape": 1000.0})
        path = finder.find_route()
        assert self._exposure(finder, path) == 0.0

    def test_honest_metrics_reported(self):
        finder = self._finder({"cost": 1.0, "landscape": 1000.0})
        path = finder.find_route()
        assert path.objective_spec is not None
        assert path.objective_spec["weights"]["landscape"] == 1000.0
        assert path.metrics is not None
        assert path.metrics["landscape"] == 0.0
        assert path.metrics["cost"] > 0.0
        assert path.feasibility > 0.0

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

    def test_dem_without_cell_size_raises(self):
        """Tier A is accepted, but never without metres: the metric is
        built from rise per metre of run."""
        raster = np.ones((30, 30), dtype=np.uint16)
        dem = np.zeros((30, 30), dtype=np.float32)
        with pytest.raises(ValueError, match="cell_size"):
            make_api(raster, dem_data=dem)

    def test_dem_with_cell_size_accepted(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        dem = np.zeros((30, 30), dtype=np.float32)
        api = make_api(raster, dem_data=dem, cell_size=10.0)
        assert api is not None

    def test_gradient_luts_without_dem_raise(self):
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


# ===========================================================================
# Tier A: slope-aware (3D-length) routing
# ===========================================================================

CELL = 10.0


def _luts(objective, cell_size=CELL, steps=STEPS_8):
    return objective.build_gradient_luts(steps, cell_size)


def _plane(rows, cols, grade, cell=CELL):
    rr, _cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return (grade * rr * cell).astype(np.float32)


class TestTierAAcceptance:
    """Acceptance is decided from the LUT ARRAYS, not the option names —
    the only check a Callable multiplier cannot bypass."""

    def test_default_gradient_options_are_tier_a(self):
        from pyorps.core.objective import Objective
        from pyorps.graph.api.raster_fim_api import tier_a_report
        ok, reason = tier_a_report(_luts(Objective({"cost": 1.0})))
        assert ok and reason == ""

    def test_identity_callable_multiplier_is_accepted(self):
        """Tier A is defined by the metric produced, not by how the user
        spelled it: a callable that happens to BE the identity passes."""
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.raster_fim_api import tier_a_report
        obj = Objective({"cost": 1.0}, GradientOptions(
            multiplier=lambda s: np.ones_like(s)))
        ok, reason = tier_a_report(_luts(obj))
        assert ok, reason

    def test_non_identity_callable_multiplier_is_refused(self):
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.raster_fim_api import tier_a_report
        obj = Objective({"cost": 1.0}, GradientOptions(
            multiplier=lambda s: 1.0 + 0.02 * s))
        ok, reason = tier_a_report(_luts(obj))
        assert not ok
        assert "MULTIPLIER" in reason and "switchback" in reason
        assert "76.8" in reason        # our own measured figure

    def test_named_multiplier_model_is_refused(self):
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.raster_fim_api import tier_a_report
        obj = Objective({"cost": 1.0},
                        GradientOptions(multiplier="power"))
        ok, reason = tier_a_report(_luts(obj))
        assert not ok and "NON-CONVEX" in reason

    def test_additive_exposure_term_is_refused(self):
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.raster_fim_api import tier_a_report
        obj = Objective({"cost": 1.0, "gradient": 2.0},
                        GradientOptions(additive="linear"))
        ok, reason = tier_a_report(_luts(obj))
        assert not ok
        assert "additive" in reason and "sqrt(d^T M d)" in reason

    def test_grade_limit_alone_stays_tier_a(self):
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.raster_fim_api import tier_a_report
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=15.0))
        ok, reason = tier_a_report(_luts(obj))
        assert ok, reason


class TestTierARefusals:
    """Every refusal names its specific reason and cites its measured
    justification — never a bare 'not supported'."""

    def _api(self, **kw):
        raster = np.ones((40, 40), dtype=np.uint16)
        dem = _plane(40, 40, 0.2)
        base = dict(dem_data=dem, cell_size=CELL)
        base.update(kw)
        return make_api(raster, **base)

    def test_multiplier_refusal_message(self):
        from pyorps.core.objective import Objective, GradientOptions
        obj = Objective({"cost": 1.0},
                        GradientOptions(multiplier="power"))
        with pytest.raises(AlgorithmNotImplementedError) as e:
            self._api(gradient_luts=_luts(obj))
        assert "switchback" in str(e.value)
        assert "benchmark_slope_indicatrix" in str(e.value)

    def test_additive_refusal_message(self):
        from pyorps.core.objective import Objective, GradientOptions
        obj = Objective({"cost": 1.0, "gradient": 3.0},
                        GradientOptions(additive="linear"))
        with pytest.raises(AlgorithmNotImplementedError) as e:
            self._api(gradient_luts=_luts(obj))
        assert "sqrt(d^T M d)" in str(e.value)

    def test_order_2_with_dem_refusal_message(self):
        with pytest.raises(AlgorithmNotImplementedError) as e:
            self._api(order=2)
        assert "order=2" in str(e.value)
        assert "simplex analogue" in str(e.value)

    def test_s_max_above_acuteness_threshold_refusal_message(self):
        from pyorps.core.objective import Objective, GradientOptions
        obj = Objective({"cost": 1.0},
                        GradientOptions(s_max_pct=400.0))
        with pytest.raises(AlgorithmNotImplementedError) as e:
            self._api(gradient_luts=_luts(obj))
        assert "acuteness" in str(e.value)
        assert "2.41421" in str(e.value)

    def test_cell_size_cross_check_catches_the_unit_bug(self):
        """The highest-probability silent bug in the increment: q formed
        per cell instead of per metre. inv_horiz_m makes it free to
        catch at construction."""
        from pyorps.core.objective import Objective
        luts = _luts(Objective({"cost": 1.0}), cell_size=CELL)
        with pytest.raises(ValueError) as e:
            self._api(gradient_luts=luts, cell_size=1.0)
        assert "cell_size mismatch" in str(e.value)
        assert "inv_horiz_m" in str(e.value)


class TestTierASolve:
    def test_axis_traverse_prices_the_pure_3d_stretch(self):
        """The whole of Tier A, in one number: a plane sloping along the
        traverse costs exactly sqrt(1 + (s/100)^2) per cell of run."""
        from pyorps.core.objective import Objective
        n = 81
        raster = np.ones((n, n), dtype=np.uint16)
        for grade in (0.25, 1.0, 2.0):
            api = make_api(raster, dem_data=_plane(n, n, grade),
                           cell_size=CELL,
                           gradient_luts=_luts(Objective({"cost": 1.0})))
            api.shortest_path(idx(0, 0, n), idx(n - 1, 0, n))
            expect = (n - 1) * np.sqrt(1.0 + grade ** 2)
            assert api.last_field_costs[0] == pytest.approx(expect,
                                                            rel=2e-3)

    def test_flat_dem_matches_the_no_dem_backend(self):
        rng = np.random.default_rng(4)
        n = 90
        raster = rng.integers(1, 40, (n, n)).astype(np.uint16)
        dem = np.full((n, n), 412.0, dtype=np.float32)
        a = make_api(raster)
        b = make_api(raster, dem_data=dem, cell_size=CELL)
        s, t = idx(5, 5, n), idx(80, 80, n)
        a.shortest_path(s, t)
        b.shortest_path(s, t)
        assert b.last_field_costs[0] == pytest.approx(
            a.last_field_costs[0], rel=1e-6)

    def test_multi_target_still_costs_one_solve(self):
        n = 81
        raster = np.ones((n, n), dtype=np.uint16)
        api = make_api(raster, dem_data=_plane(n, n, 0.5),
                       cell_size=CELL)
        paths = api.shortest_path(idx(0, 0, n),
                                  [idx(n - 1, 0, n), idx(0, n - 1, n)])
        assert len(paths) == 2
        assert all(len(p) > 2 for p in paths)
        # the down-slope traverse is dearer than the level one
        assert api.last_field_costs[0] > api.last_field_costs[1]


class TestGradeLimitLoop:
    """max_gradient_pct is enforced OUTSIDE the solver. No test here may
    accept an unverified route."""

    @staticmethod
    def _terrain(n=81):
        """A Gaussian hill sitting on the direct line, with flat ground
        all around it — so a legal detour exists and the unconstrained
        route still prefers to climb over."""
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        hill = 200.0 * np.exp(
            -(((rr - 40) ** 2 + (cc - 40) ** 2) / (2 * 10.0 ** 2)))
        return hill.astype(np.float32)

    def _api(self, limit, mode="lazy", n=81, **kw):
        from pyorps.core.objective import Objective, GradientOptions
        raster = np.ones((n, n), dtype=np.uint16)
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=limit))
        return make_api(raster, dem_data=self._terrain(n),
                        cell_size=CELL, gradient_luts=_luts(obj),
                        grade_limit_mode=mode, **kw), n

    def test_returned_route_is_verified_against_the_discrete_rule(self):
        api, n = self._api(limit=12.0)
        path = api.shortest_path(idx(40, 5, n), idx(40, 75, n))
        assert len(path) > 2
        assert api.grade_violations(path).size == 0

    def test_the_loop_actually_had_to_mask(self):
        """A limit the unconstrained route violates must cost > 0 mask
        iterations — otherwise the test proves nothing about the loop."""
        api, n = self._api(limit=20.0)
        raster = np.ones((n, n), dtype=np.uint16)
        free = make_api(raster, dem_data=self._terrain(n),
                        cell_size=CELL)          # no grade limit at all
        s, t = idx(40, 5, n), idx(40, 75, n)
        free_path = free.shortest_path(s, t)
        # the unconstrained route really does violate the limit
        assert api.grade_violations(free_path).size > 0
        path = api.shortest_path(s, t)
        assert api.last_mask_iterations[0] >= 1
        assert api.grade_violations(path).size == 0
        # ... and the constrained route is dearer than the free one
        assert api.last_field_costs[0] > free.last_field_costs[0]

    def test_lazy_beats_eager_on_cost(self):
        """The point of paying for the loop: the smallest mask that
        certifies the route leaves a cheaper route than pre-masking every
        steep cell."""
        s, t = idx(40, 5, 81), idx(40, 75, 81)
        lazy, _n = self._api(limit=12.0)
        eager, _n = self._api(limit=12.0, mode="eager")
        lazy.shortest_path(s, t)
        eager.shortest_path(s, t)
        assert lazy.last_field_costs[0] < eager.last_field_costs[0]

    def test_eager_mode_needs_no_iterations_and_is_still_valid(self):
        api, n = self._api(limit=12.0, mode="eager")
        path = api.shortest_path(idx(40, 5, n), idx(40, 75, n))
        assert api.last_mask_iterations[0] == 0
        assert api.grade_violations(path).size == 0

    def test_infeasible_case_reports_the_conservatism(self):
        """A limit no route can meet must raise, and the message must say
        exactly what the masked problem is — not oversell it."""
        n = 61
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        # a steep collar right around the target: no approach is legal
        dem = (60.0 * np.clip(12.0 - np.hypot(rr - 30, cc - 45), 0.0, 12.0)
               ).astype(np.float32)
        from pyorps.core.objective import Objective, GradientOptions
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=10.0))
        api = make_api(np.ones((n, n), dtype=np.uint16), dem_data=dem,
                       cell_size=CELL, gradient_luts=_luts(obj))
        with pytest.raises(NoPathFoundError) as e:
            api.shortest_path(idx(30, 5, n), idx(30, 45, n))
        msg = str(e.value)
        assert "Grade limit" in msg
        # the caveat must be stated as the MEASURED screen it is
        assert "SCREEN, not an authority" in msg
        assert "48.3 %" in msg
        assert "slightly" not in msg.lower()

    def test_cap_raises_rather_than_escalating_to_eager(self):
        """Silently switching to the strictly smaller 'eager' feasible
        set on cap-out would be the same class of failure this whole
        increment exists to prevent."""
        api, n = self._api(limit=12.0, max_mask_iterations=1)
        with pytest.raises(RuntimeError) as e:
            api.shortest_path(idx(40, 5, n), idx(40, 75, n))
        msg = str(e.value)
        assert "cap" in msg
        assert "eager" in msg
        assert "chosen explicitly" in msg

    def test_no_limit_means_no_mask_machinery(self):
        n = 81
        raster = np.ones((n, n), dtype=np.uint16)
        api = make_api(raster, dem_data=self._terrain(n), cell_size=CELL)
        api.shortest_path(idx(40, 5, n), idx(40, 75, n))
        assert api.last_mask_iterations == [0]
        assert api.grade_violations([0, 1]).size == 0


class TestGradeCheckMatchesTheDiscreteKernel:
    """The whole loop is worthless if "FIM says valid" and "the Cython
    kernel says valid" can disagree — the route would be accepted here
    and rejected by the backend the rule is defined on.

    ``_dijkstra.pyx`` (lines 328-332) does NOT compute a percent slope: it
    multiplies |dh| by a per-direction factor that was STORED AS FLOAT32.
    """

    #: A diagonal |dh| (metres) at cell_size 10 m whose true slope is
    #: 5.000000 % — dead on a bin boundary. float32 ``bin_factor`` is
    #: 28.284273 where the exact factor is 28.2842712…, so the kernel bins
    #: it UP into the forbidden bin while a float64 recompute of
    #: ``floor(100*dh/(L*cell)*bin_inv)`` bins it DOWN into a legal one.
    H_ON_THE_BOUNDARY = np.float32(0.7071067690849304)
    H_CLEARLY_LEGAL = np.float32(0.60)
    LIMIT = 5.0

    def _corridor(self, height):
        """3x3 where the ONLY candidate step is the (0,0)->(1,1) diagonal:
        every other cell sits 1000 m up, so any detour is wildly illegal.
        """
        raster = np.ones((3, 3), dtype=np.uint16)
        dem = np.full((3, 3), 1000.0, dtype=np.float32)
        dem[0, 0] = 0.0
        dem[1, 1] = height
        return raster, dem

    def _both_verdicts(self, height):
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.cython_api import CythonAPI
        raster, dem = self._corridor(height)
        luts = Objective(
            {"cost": 1.0},
            GradientOptions(max_gradient_pct=self.LIMIT)
        ).build_gradient_luts(STEPS_8, CELL)
        fim = make_api(raster, dem_data=dem, cell_size=CELL,
                       gradient_luts=luts)
        cython = CythonAPI(raster, STEPS_8, dem_data=dem,
                           gradient_luts=luts)
        try:
            cython_route = list(cython.shortest_path(0, 4))
        except Exception:
            cython_route = []
        return fim.grade_violations([0, 4]).size, bool(cython_route)

    def test_boundary_step_forbidden_by_both(self):
        n_viol, cython_found = self._both_verdicts(self.H_ON_THE_BOUNDARY)
        assert n_viol == 1, "the check must reject what the kernel rejects"
        assert not cython_found

    def test_legal_step_accepted_by_both(self):
        n_viol, cython_found = self._both_verdicts(self.H_CLEARLY_LEGAL)
        assert n_viol == 0
        assert cython_found

    def test_the_naive_float64_recompute_would_have_disagreed(self):
        """Pins WHY the check is written the way it is. If someone
        'simplifies' grade_violations back to a percent-slope compare,
        this is the case that silently starts accepting illegal routes.
        """
        from pyorps.core.objective import Objective, GradientOptions
        luts = Objective(
            {"cost": 1.0},
            GradientOptions(max_gradient_pct=self.LIMIT)
        ).build_gradient_luts(STEPS_8, CELL)
        h = float(self.H_ON_THE_BOUNDARY)
        length = np.sqrt(2.0)
        naive_bin = int(np.floor(100.0 * h / (length * CELL) * luts.bin_inv))
        kernel_bin = int(h * np.float64(luts.bin_factor[4]))
        assert kernel_bin == naive_bin + 1
        assert not np.isfinite(luts.mult[kernel_bin])   # kernel: forbidden
        assert np.isfinite(luts.mult[naive_bin])        # naive: allowed

    def test_bin_factor_reproduction_is_bit_exact(self):
        """The check reconstructs bin_factor instead of indexing it, so a
        path step of any length can be verified. That is only sound while
        the reconstruction IS the array."""
        from pyorps.core.objective import Objective
        luts = _luts(Objective({"cost": 1.0}))
        lengths = np.sqrt(STEPS_8[:, 0].astype(np.float64) ** 2
                          + STEPS_8[:, 1].astype(np.float64) ** 2)
        ours = RasterFIMAPI._bin_factor_for(lengths, CELL,
                                            float(luts.bin_inv))
        assert np.array_equal(ours, np.asarray(luts.bin_factor,
                                               dtype=np.float32))

    def test_construction_refuses_luts_it_cannot_reproduce(self):
        """The tripwire: if objective.py ever forms bin_factor
        differently, refuse rather than verify with other arithmetic."""
        import copy
        from pyorps.core.objective import Objective
        luts = copy.copy(_luts(Objective({"cost": 1.0})))
        luts.bin_factor = np.asarray(luts.bin_factor,
                                     dtype=np.float32) * np.float32(1.01)
        with pytest.raises(ValueError, match="bit-for-bit"):
            make_api(np.ones((20, 20), dtype=np.uint16),
                     dem_data=np.zeros((20, 20), dtype=np.float32),
                     cell_size=CELL, gradient_luts=luts)


class TestGradeLimitRidgeAndPass:
    """A ridge that cannot be crossed, with one gentle pass in it."""

    N = 81
    PASS_ROW = 20
    CREST_COL = 40

    @classmethod
    def _dem(cls, height=120.0, notch=0.92):
        n = cls.N
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        profile = np.exp(-((cc - float(cls.CREST_COL)) ** 2) / (2 * 4.0 ** 2))
        amplitude = height * (
            1.0 - notch * np.exp(-((rr - float(cls.PASS_ROW)) ** 2)
                                 / (2 * 6.0 ** 2)))
        return (amplitude * profile).astype(np.float32)

    def _api(self, limit, **kw):
        from pyorps.core.objective import Objective, GradientOptions
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=limit))
        return make_api(np.ones((self.N, self.N), dtype=np.uint16),
                        dem_data=self._dem(), cell_size=CELL,
                        gradient_luts=_luts(obj), **kw)

    def test_the_terrain_is_what_the_test_claims(self):
        """Guard the fixture itself: the crest really is impassable at the
        limit under test and the pass really is passable."""
        dem = self._dem().astype(np.float64)
        crest = 100.0 * np.abs(np.diff(dem[70])).max() / CELL
        gap = 100.0 * np.abs(np.diff(dem[self.PASS_ROW])).max() / CELL
        assert crest > 100.0
        assert gap < 15.0

    def test_route_goes_through_the_pass(self):
        api = self._api(limit=15.0)
        source = idx(self.CREST_COL, 5, self.N)
        target = idx(self.CREST_COL, 75, self.N)
        path = api.shortest_path(source, target)
        assert api.grade_violations(path).size == 0
        crossing_rows = sorted({int(c) // self.N for c in path
                                if int(c) % self.N == self.CREST_COL})
        assert crossing_rows == [self.PASS_ROW]
        # and it really was the constraint that moved it there
        free = make_api(np.ones((self.N, self.N), dtype=np.uint16),
                        dem_data=self._dem(), cell_size=CELL)
        free_path = free.shortest_path(source, target)
        free_rows = sorted({int(c) // self.N for c in free_path
                            if int(c) % self.N == self.CREST_COL})
        assert free_rows != [self.PASS_ROW]
        assert api.grade_violations(free_path).size > 0

    def test_iterations_stay_inside_the_documented_bound(self):
        api = self._api(limit=15.0)
        api.shortest_path(idx(self.CREST_COL, 5, self.N),
                          idx(self.CREST_COL, 75, self.N))
        assert 0 < api.last_mask_iterations[0] <= api._max_mask_iterations

    def test_pass_too_steep_is_certified_infeasible_with_no_solve(self):
        """A limit below the pass grade disconnects the two sides. That is
        a property of the PROBLEM, so it must be reported as such — before
        any solve, not as an iteration cap."""
        api = self._api(limit=8.0)
        with pytest.raises(NoPathFoundError) as excinfo:
            api.shortest_path(idx(self.CREST_COL, 5, self.N),
                              idx(self.CREST_COL, 75, self.N))
        msg = str(excinfo.value)
        assert "INFEASIBLE" in msg
        assert "legal-chord graph" in msg
        assert "cost no solve" in msg
        assert api.last_mask_iterations == []      # no solve was attempted


class TestGradeLimitLoopIterates:
    """Proof that the loop is a loop: first solve invalid, second valid."""

    N = 41

    @staticmethod
    def _bump_dem(n, height=1.5):
        """One raised cell on the straight line. Cheap enough that the
        unconstrained field routes straight over it (cost x1.0056), steep
        enough that both of its steps break a 10 % limit (15 %)."""
        dem = np.zeros((n, n), dtype=np.float32)
        dem[n // 2, n // 2] = height
        return dem

    def _apis(self, limit=10.0, **kw):
        from pyorps.core.objective import Objective, GradientOptions
        n = self.N
        raster = np.ones((n, n), dtype=np.uint16)
        dem = self._bump_dem(n)
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=limit))
        limited = make_api(raster, dem_data=dem, cell_size=CELL,
                           gradient_luts=_luts(obj), **kw)
        free = make_api(raster, dem_data=dem, cell_size=CELL)
        return limited, free, n

    def test_exactly_one_mask_iteration(self):
        limited, free, n = self._apis()
        source, target = idx(n // 2, 5, n), idx(n // 2, n - 6, n)
        free_path = free.shortest_path(source, target)
        # 1st solve: the unconstrained answer, and it IS invalid
        assert limited.grade_violations(free_path).size == 2
        path = limited.shortest_path(source, target)
        # 2nd solve: valid
        assert limited.last_mask_iterations[0] == 1
        assert limited.grade_violations(path).size == 0
        assert limited.last_field_costs[0] > free.last_field_costs[0]

    def test_a_cap_of_zero_proves_the_first_solve_was_the_invalid_one(self):
        """With no iterations allowed the loop must FAIL rather than
        return the route it just measured as illegal."""
        limited, _free, n = self._apis(max_mask_iterations=0)
        with pytest.raises(RuntimeError, match="cap"):
            limited.shortest_path(idx(n // 2, 5, n),
                                  idx(n // 2, n - 6, n))

    def test_cap_message_says_a_legal_route_exists(self):
        """Cap-out and genuine infeasibility are different failures and
        must not read the same."""
        limited, _free, n = self._apis(max_mask_iterations=0)
        with pytest.raises(RuntimeError) as excinfo:
            limited.shortest_path(idx(n // 2, 5, n),
                                  idx(n // 2, n - 6, n))
        # the certificate speaks about the CALLER's steps, so the wording
        # names the neighborhood rather than hardcoding "8-adjacent"
        msg = str(excinfo.value)
        assert "A legal route DOES exist" in msg
        assert "8-step neighborhood" in msg


class TestGradeLimitInfeasibility:
    """Infeasible must mean reported, promptly — never an invalid route,
    never an unbounded loop."""

    N = 61

    def _api(self, dem, limit=10.0, **kw):
        from pyorps.core.objective import Objective, GradientOptions
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=limit))
        return make_api(np.ones((self.N, self.N), dtype=np.uint16),
                        dem_data=dem, cell_size=CELL,
                        gradient_luts=_luts(obj), **kw)

    def test_every_route_between_the_pair_exceeds_the_limit(self):
        """A tilted plane at 300 %: the only legal chords are the two
        anti-diagonals that stay on one contour, so the legal-chord graph
        splits into contour lines and no route joins two of them. Reported
        with zero solves."""
        n = self.N
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dem = (30.0 * (rr + cc)).astype(np.float32)
        api = self._api(dem)
        with pytest.raises(NoPathFoundError) as excinfo:
            api.shortest_path(idx(30, 5, n), idx(30, 55, n))
        msg = str(excinfo.value)
        assert "DIFFERENT components" in msg
        assert "INFEASIBLE" in msg
        assert api.last_mask_iterations == []

    def test_a_terminal_on_a_spike_is_reported_as_isolated(self):
        """Every chord out of the source is illegal — the cheapest
        infeasibility there is, and it must not cost a solve either."""
        n = self.N
        dem = np.zeros((n, n), dtype=np.float32)
        dem[30, 5] = 50.0
        api = self._api(dem)
        with pytest.raises(NoPathFoundError) as excinfo:
            api.shortest_path(idx(30, 5, n), idx(30, 55, n))
        msg = str(excinfo.value)
        assert "ISOLATED" in msg
        assert "INFEASIBLE" in msg
        assert api.last_mask_iterations == []

    def test_target_behind_an_uncrossable_collar_is_reported(self):
        """The terminals themselves are fine; it is the terrain between
        them that disconnects the pair."""
        n = self.N
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        ring = np.abs(np.hypot(rr - 30.0, cc - 45.0) - 8.0)
        dem = (200.0 * np.clip(1.0 - ring, 0.0, 1.0)).astype(np.float32)
        api = self._api(dem)
        with pytest.raises(NoPathFoundError) as excinfo:
            api.shortest_path(idx(30, 5, n), idx(30, 45, n))
        msg = str(excinfo.value)
        assert "Grade limit" in msg
        assert "INFEASIBLE" in msg or "CONSERVATIVE RELAXATION" in msg
        # bounded: whatever happened, it did not run past the cap
        iters = api.last_mask_iterations
        assert iters == [] or iters[0] <= api._max_mask_iterations

    def test_infeasible_never_returns_a_path(self):
        """The failure mode that must never happen: an invalid route
        returned instead of an exception."""
        n = self.N
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dem = (30.0 * (rr + cc)).astype(np.float32)
        api = self._api(dem)
        for source, target in (((30, 5), (30, 55)), ((5, 5), (55, 55))):
            with pytest.raises(NoPathFoundError):
                api.shortest_path(idx(*source, n), idx(*target, n))

    def test_multi_target_infeasible_pairs_come_back_empty(self):
        """The list API cannot raise per pair, so an infeasible pair must
        degrade to [] — not to an unverified route."""
        n = self.N
        rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dem = np.zeros((n, n), dtype=np.float32)
        dem[:, 30:] = (30.0 * (rr + cc))[:, 30:]        # far side is a cliff
        api = self._api(dem)
        paths = api.shortest_path(idx(30, 5, n),
                                 [idx(30, 20, n), idx(30, 55, n)])
        assert len(paths) == 2
        assert len(paths[0]) > 2 and api.grade_violations(paths[0]).size == 0
        assert paths[1] == []


class TestGradeLimitAndExclusions:
    """The mask must COMPOSE with the raster's own impassable cells, not
    replace them."""

    N = 61

    def _fixture(self, curb_row=46):
        from pyorps.core.objective import Objective, GradientOptions
        n = self.N
        raster = np.ones((n, n), dtype=np.uint16)
        raster[0:46, 30] = 65535           # wall, gap at rows 46..60
        dem = np.zeros((n, n), dtype=np.float32)
        dem[curb_row, 30] = 1.5            # a curb inside the only gap
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=10.0))
        api = make_api(raster, dem_data=dem, cell_size=CELL,
                       gradient_luts=_luts(obj))
        return api, raster, n

    def test_route_respects_both_the_wall_and_the_limit(self):
        api, raster, n = self._fixture()
        path = api.shortest_path(idx(20, 5, n), idx(20, 55, n))
        wall = set(np.flatnonzero(raster.ravel() == 65535).tolist())
        assert not wall & set(int(c) for c in path)
        assert api.grade_violations(path).size == 0
        assert api.last_mask_iterations[0] == 1        # the curb was masked
        crossings = sorted({int(c) // n for c in path if int(c) % n == 30})
        assert crossings == [47]                       # around the curb

    def test_masked_cells_do_not_leak_into_later_calls(self):
        """Masks are per route by design (that is why a limited multi-
        target query falls back to per-pair loops). A second query must
        not inherit the first one's mask."""
        api, _raster, n = self._fixture()
        api.shortest_path(idx(20, 5, n), idx(20, 55, n))
        assert api._eager_mask is None
        assert api._forbidden() is None

    def test_eager_mask_composes_with_the_wall(self):
        from pyorps.core.objective import Objective, GradientOptions
        n = self.N
        raster = np.ones((n, n), dtype=np.uint16)
        raster[0:46, 30] = 65535
        dem = np.zeros((n, n), dtype=np.float32)
        dem[46, 30] = 1.5
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=10.0))
        api = make_api(raster, dem_data=dem, cell_size=CELL,
                       gradient_luts=_luts(obj), grade_limit_mode="eager")
        path = api.shortest_path(idx(20, 5, n), idx(20, 55, n))
        wall = set(np.flatnonzero(raster.ravel() == 65535).tolist())
        assert not wall & set(int(c) for c in path)
        assert api.grade_violations(path).size == 0
        assert api.last_mask_iterations[0] == 0
        assert idx(46, 30, n) in set(api._eager_mask.tolist())


class TestChordRuleProjection:
    """The cell-level 'steep' test is the discrete rule itself, not the
    central-difference |q| — which is what makes eager mode sound."""

    def test_a_one_sided_cliff_is_steep_even_though_q_reads_flat(self):
        from pyorps.core.objective import Objective, GradientOptions
        n = 21
        dem = np.zeros((n, n), dtype=np.float32)
        dem[:, 10:] = 3.0                  # a 30 % step, flat either side
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=10.0))
        api = make_api(np.ones((n, n), dtype=np.uint16), dem_data=dem,
                       cell_size=CELL, gradient_luts=_luts(obj))
        steep, _isolated, _peak = api._chord_maps()
        assert steep[10, 9] and steep[10, 10]           # both cliff lips
        assert not steep[10, 2]                         # flat ground is not
        q_r, q_c = api._q_host()
        # the central difference halves the step and misses cell 9 at a
        # 10 % limit — 15 % read where the real chord is 30 %
        assert np.hypot(q_r[10, 9], q_c[10, 9]) < 0.30

    def test_eager_mode_route_has_no_illegal_step(self):
        """If no cell on an 8-adjacent route is steep, no step of that
        route is forbidden. Checked on the cliff the |q| test misses."""
        from pyorps.core.objective import Objective, GradientOptions
        n = 41
        dem = np.zeros((n, n), dtype=np.float32)
        dem[15:25, 20] = 3.0
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=10.0))
        api = make_api(np.ones((n, n), dtype=np.uint16), dem_data=dem,
                       cell_size=CELL, gradient_luts=_luts(obj),
                       grade_limit_mode="eager")
        path = api.shortest_path(idx(20, 5, n), idx(20, 35, n))
        assert len(path) > 2
        assert api.grade_violations(path).size == 0


class TestTierAPathFinderIntegration:
    def test_end_to_end_with_dem(self):
        from pyorps.graph.path_finder import PathFinder
        from rasterio.transform import from_origin
        rng = np.random.default_rng(9)
        n = 100
        raster = rng.integers(1, 50, (n, n)).astype(np.uint16)
        _rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dem = (30.0 * np.sin(cc / 8.0)).astype(np.float32)
        transform = from_origin(0.0, float(n), 1.0, 1.0)
        finder = PathFinder(
            dataset_source=raster,
            crs="EPSG:32632",
            transform=transform,
            source_coords=(10.5, 50.5),
            target_coords=(89.5, 50.5),
            search_space_buffer_m=500,
            graph_api="raster_fim",
            dem=dem,
            dem_kwargs={"crs": "EPSG:32632", "transform": transform},
        )
        path = finder.find_route()
        assert path is not None
        api = finder.graph_api
        # cell_size reached the backend and Tier A is live
        assert api._cell_size == pytest.approx(1.0)
        assert api._dem is not None
        assert api.last_field_costs[0] > 0


# ===========================================================================
# Adversarial regressions (2026-08-11 review)
# ===========================================================================

def _ramp(cols, peak_grade, period=60.0, cell=CELL):
    """Corrugated ramp ``z = A sin(k x)``, plus its unrolled arc length.

    The surface is a cylinder, hence isometric to the plane: the geodesic
    is a straight line in ``(s, y)`` and therefore CURVED in the raster
    frame. That is what separates a tracer that honours the metric from
    one that merely looks plausible.
    """
    k = 2.0 * np.pi / period
    amp = peak_grade / (k / cell)
    x = np.arange(cols)
    f = amp * np.sin(k * x)
    fp = amp * k * np.cos(k * x) / cell
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(1.0 + fp[:-1] ** 2))])
    return f.astype(np.float32), s


def _unrolled_deviation(poly, s, src, tgt, n):
    """Max distance of a polyline from the analytic geodesic, in cells."""
    ps = np.interp(poly[:, 1], np.arange(n), s)
    py = poly[:, 0]
    s0, y0 = s[src % n], float(src // n)
    s1, y1 = s[tgt % n], float(tgt // n)
    num = np.abs((s1 - s0) * (y0 - py) - (s0 - ps) * (y1 - y0))
    return float((num / np.hypot(s1 - s0, y1 - y0)).max())


class TestCachedFieldKeepsItsMetric:
    """CRITICAL 1. A field solved under the Tier A metric M must be
    traced with ``-M^-1 grad T``. Tracing it with plain ``-grad T``
    produces a plausible polyline that is not the geodesic — a silent
    wrong answer, not a degradation.

    The path that regressed is the CACHED one: ``_multi_to_multi`` keeps
    one field per unique source, so a pairwise call whose source repeats
    after another source traces a field that is no longer the most
    recent solve. The assertion is therefore geometric (distance from
    the analytic geodesic), because an isotropic fallback still returns
    a well-formed path and would pass any structural check.
    """

    N = 161

    def _run(self):
        f, s = _ramp(self.N, 1.0)
        dem = np.tile(f, (self.N, 1)).astype(np.float32)
        raster = np.ones((self.N, self.N), dtype=np.uint16)
        api = make_api(raster, dem_data=dem, cell_size=CELL)
        src_a, src_b = 20 * self.N + 10, 100 * self.N + 12
        tgt_a, tgt_b = 140 * self.N + 150, 30 * self.N + 150
        api.shortest_path([src_a, src_b, src_a], [tgt_a, tgt_b, tgt_a],
                          pairwise=True)
        return api, s, src_a, tgt_a

    def test_the_cached_pair_really_is_the_cached_code_path(self):
        """Guard the guard: if the field for pair 2 stopped being a
        cached one, the test below would pass for the wrong reason."""
        api, _s, _src, _tgt = self._run()
        assert len(api.last_polylines) == 3
        assert all(p is not None for p in api.last_polylines)

    def test_cached_trace_follows_the_analytic_geodesic(self):
        api, s, src, tgt = self._run()
        fresh = _unrolled_deviation(api.last_polylines[0], s, src, tgt,
                                    self.N)
        cached = _unrolled_deviation(api.last_polylines[2], s, src, tgt,
                                     self.N)
        # measured: 0.71 cells with the metric, 14.33 cells without it
        assert fresh < 2.0, fresh
        assert cached < 2.0, (
            f"the cached-field trace deviated {cached:.2f} cells from "
            f"the analytic geodesic while the freshly solved one "
            f"deviated {fresh:.2f} — the cached path descended -grad T "
            f"instead of -M^-1 grad T")

    def test_the_metric_is_never_dropped_for_a_trace(self):
        api, _s, _src, _tgt = self._run()
        q_dev = api._trace_metric()
        assert q_dev[0] is not None and q_dev[1] is not None

    def test_isotropic_backend_still_traces_isotropically(self):
        """The dem=None path must stay untouched: no metric, no
        anisotropic tracer branch."""
        api = make_api(np.ones((40, 40), dtype=np.uint16))
        assert api._trace_metric() == (None, None)


class TestCertificateUsesTheCallersSteps:
    """CRITICAL 2. The exact infeasibility certificates enumerated the
    8-neighbourhood while PathFinder's DEFAULT neighborhood is 'r2'.
    A single 2 m elevation step at 10 m cells is illegal for every
    8-chord at a 10 % limit (axis 20 %, diagonal 14.1 %) and legal for
    every knight move (8.9 %), so an 8-based certificate calls this
    infeasible while the caller's own kernel routes across it.
    """

    N = 24
    LIMIT = 10.0
    RISE = 2.0
    WALL_COL = 12

    def _steps_r2(self):
        from pyorps.utils.neighborhood import get_neighborhood_steps
        return get_neighborhood_steps("r2", directed=True)

    def _case(self, steps):
        from pyorps.core.objective import Objective, GradientOptions
        n = self.N
        dem = np.zeros((n, n), dtype=np.float32)
        dem[:, self.WALL_COL:] = self.RISE
        raster = np.ones((n, n), dtype=np.uint16)
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=self.LIMIT))
        luts = obj.build_gradient_luts(steps, CELL)
        api = RasterFIMAPI(raster, steps, dem_data=dem, cell_size=CELL,
                           gradient_luts=luts)
        return api, luts, dem, raster

    def test_the_wall_is_the_case_it_claims_to_be(self):
        """8-chords all illegal, knight moves all legal — otherwise the
        regression below proves nothing."""
        for length, legal in ((1.0, False), (np.sqrt(2.0), False),
                              (np.sqrt(5.0), True)):
            slope = 100.0 * self.RISE / (length * CELL)
            assert bool(slope <= self.LIMIT) is legal, (length, slope)

    def test_r2_pair_is_not_declared_infeasible(self):
        n = self.N
        api, _l, _d, _r = self._case(self._steps_r2())
        s = idx(12, self.WALL_COL - 3, n)
        t = idx(12, self.WALL_COL + 3, n)
        # the certificate is what regressed: it must not fire at all
        api._check_feasible(s, t)                      # must not raise
        assert len(api._cert_offsets) == 16

    def test_a_discrete_r2_backend_really_does_cross_the_wall(self):
        """The differential half: the pair the certificate used to call
        INFEASIBLE is one the caller's own kernel routes."""
        from pyorps.graph.api.cython_api import CythonAPI
        n = self.N
        steps = self._steps_r2()
        _api, luts, dem, raster = self._case(steps)
        cy = CythonAPI(raster, steps, dem_data=dem, gradient_luts=luts)
        route = list(cy.shortest_path(idx(12, self.WALL_COL - 3, n),
                                      idx(12, self.WALL_COL + 3, n)))
        assert len(route) > 1

    def test_an_8_neighborhood_caller_still_gets_the_certificate(self):
        """Same terrain, 8 steps: now the pair IS infeasible and the
        exact certificate must still say so before any solve."""
        n = self.N
        api, _l, _d, _r = self._case(STEPS_8)
        assert len(api._cert_offsets) == 8
        with pytest.raises(NoPathFoundError, match="DIFFERENT components"):
            api._check_feasible(idx(12, self.WALL_COL - 3, n),
                                idx(12, self.WALL_COL + 3, n))

    def test_certificate_message_names_the_callers_neighborhood(self):
        n = self.N
        api, _l, _d, _r = self._case(STEPS_8)
        with pytest.raises(NoPathFoundError) as e:
            api._check_feasible(idx(12, self.WALL_COL - 3, n),
                                idx(12, self.WALL_COL + 3, n))
        assert "8-step neighborhood" in str(e.value)


class TestMaskCaveatIsMeasuredNotSoothing:
    """MAJOR. The masked relaxation misses about half the routes the
    Cython kernel finds (320/662 measured). Every place that describes it
    must carry that number instead of the word "slightly"."""

    def test_the_caveat_states_the_measured_rate(self):
        from pyorps.graph.api.raster_fim_api import (
            _MASK_CAVEAT, MASK_FALSE_NEGATIVE_RATE)
        assert 0.40 < MASK_FALSE_NEGATIVE_RATE < 0.55
        assert "48.3 %" in _MASK_CAVEAT
        assert "320 of 662" in _MASK_CAVEAT
        assert "slightly" not in _MASK_CAVEAT.lower()
        # and it must name who the authority is
        assert "cython" in _MASK_CAVEAT.lower()

    def test_module_and_loop_docs_carry_it_too(self):
        import pyorps.graph.api.raster_fim_api as mod
        for text in (mod.__doc__,
                     mod.RasterFIMAPI._solve_with_grade_limit.__doc__):
            assert "48.3 %" in text
            assert "slightly conservative" not in text

    def test_a_known_false_negative_is_reported_as_a_relaxation_limit(self):
        """The honesty regression: on a case the Cython kernel solves,
        the loop failure must say a legal route exists and point at the
        discrete backend — never "INFEASIBLE as posed"."""
        from pyorps.core.objective import Objective, GradientOptions
        from pyorps.graph.api.cython_api import CythonAPI
        n = 40
        rng = np.random.default_rng(0)          # a measured FN case
        dem = (rng.normal(size=(n, n)).cumsum(0).cumsum(1)
               * 0.25).astype(np.float32)
        raster = np.ones((n, n), dtype=np.uint16)
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=12.0))
        luts = obj.build_gradient_luts(STEPS_8, CELL)
        s, t = idx(3, 3, n), idx(36, 36, n)
        cy = CythonAPI(raster, STEPS_8, dem_data=dem, gradient_luts=luts)
        assert list(cy.shortest_path(s, t))       # the kernel solves it
        api = make_api(raster, dem_data=dem, cell_size=CELL,
                       gradient_luts=luts)
        try:
            path = api.shortest_path(s, t)
        except (NoPathFoundError, RuntimeError) as exc:
            msg = str(exc)
            assert "INFEASIBLE as posed" not in msg
            assert "A legal route DOES exist" in msg
            assert "cython" in msg
            assert "48.3 %" in msg
        else:
            # finding it is fine too — it must then be verified
            assert api.grade_violations(path).size == 0

    def test_the_mask_never_cuts_the_last_legal_corridor(self):
        """The connectivity filter: no candidate may disconnect source
        from target in the legal-chord graph, so a lost-reachability
        report can no longer be self-inflicted."""
        from pyorps.core.objective import Objective, GradientOptions
        n = 24
        dem = np.zeros((n, n), dtype=np.float32)
        dem[:, 12:] = 2.0
        dem[11, :] = 0.0                       # the ONLY legal corridor
        raster = np.ones((n, n), dtype=np.uint16)
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=10.0))
        api = make_api(raster, dem_data=dem, cell_size=CELL,
                       gradient_luts=_luts(obj))
        s, t = idx(11, 2, n), idx(11, 21, n)
        cand = [idx(11, c, n) for c in range(3, 21)]
        kept = api._drop_disconnecting(cand, None, s, t)
        # masking the whole corridor would disconnect it; the filter
        # must refuse at least part of the batch
        assert len(kept) < len(cand)
        labels = api._components(np.asarray(kept, dtype=np.int64))
        assert labels[s] == labels[t] >= 0


class TestDemCacheContract:
    """``dem_data`` is aliased, not copied — stale caches must not survive
    silent in-place edits."""

    def test_dem_is_aliased_not_copied(self):
        raster = np.ones((20, 20), dtype=np.uint16)
        dem = np.zeros((20, 20), dtype=np.float32)
        api = make_api(raster, dem_data=dem, cell_size=CELL)
        assert api._dem is dem

    def test_in_place_edit_raises_on_next_use(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        dem = np.zeros((30, 30), dtype=np.float32)
        api = make_api(raster, dem_data=dem, cell_size=CELL)
        api.shortest_path(idx(5, 5, 30), idx(25, 25, 30))
        dem[15, 15] = 100.0
        with pytest.raises(RuntimeError, match="modified in place"):
            api.shortest_path(idx(5, 5, 30), idx(25, 25, 30))

    def test_invalidate_dem_caches_allows_routing_after_edit(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        dem = np.zeros((30, 30), dtype=np.float32)
        api = make_api(raster, dem_data=dem, cell_size=CELL)
        api.shortest_path(idx(5, 5, 30), idx(25, 25, 30))
        dem[15, 15] = 100.0
        api.invalidate_dem_caches()
        path = api.shortest_path(idx(5, 5, 30), idx(25, 25, 30))
        assert len(path) >= 2

    def test_invalidate_clears_metric_cache(self):
        raster = np.ones((30, 30), dtype=np.uint16)
        dem = np.zeros((30, 30), dtype=np.float32)
        api = make_api(raster, dem_data=dem, cell_size=CELL)
        api.shortest_path(idx(5, 5, 30), idx(25, 25, 30))
        assert api._metric_device[0] is not None
        api.invalidate_dem_caches()
        assert api._metric_device == (None, None)
        assert api._q_cache is None
        assert api._chord_cache == {}
        assert not api._edges_built

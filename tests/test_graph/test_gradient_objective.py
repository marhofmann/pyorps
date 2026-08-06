"""Phase 4 tests: per-edge gradient terms in the CPU search.

The contour test is the v2/v3 discriminator: on a uniformly TILTED PLANE
every cell has identical |∇DEM|, so no per-cell layer can prefer
contour-following — only the in-kernel edge slope can.
"""
import math
import unittest

import numpy as np
from rasterio.transform import from_origin

from pyorps import PathFinder
from pyorps.core.objective import Objective
from pyorps.utils.metric_edges import construct_edges_gradient
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.utils.path_algorithms import dijkstra_2d_cython

CRS = "EPSG:25832"
N = 41  # grid size
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)  # 1 m cells


def tilted_plane_dem(slope_pct=40.0):
    """Height rises with row: contours run along rows (east-west)."""
    rise_per_cell = slope_pct / 100.0
    return (np.arange(N, dtype=np.float32)[:, None] * rise_per_cell *
            np.ones((1, N), dtype=np.float32))


def uniform_raster(value=100):
    return np.full((N, N), value, dtype=np.uint16)


def finder(objective, dem, raster=None, gradient_options=None, **kwargs):
    return PathFinder(
        dataset_source=raster if raster is not None else uniform_raster(),
        crs=CRS,
        transform=TRANSFORM,
        source_coords=(2.5, N - 0.5 - 20),   # row 20, col 2
        target_coords=(N - 2.5, N - 0.5 - 20),
        search_space_buffer_m=200,
        graph_api=kwargs.pop("graph_api", "cython"),
        dem=dem,
        objective=objective,
        gradient_options=gradient_options,
        **kwargs,
    )


class TestContourBehavior(unittest.TestCase):
    """Source/target sit on the same contour of a tilted plane."""

    def test_same_contour_route_stays_level(self):
        f = finder({"gradient": 50.0, "length": 0.01}, tilted_plane_dem())
        path = f.find_route()
        rows = np.unique(np.array(path.path_indices)
                         // f.raster_handler.data[0].shape[1])
        self.assertEqual(len(rows), 1)  # never leaves the contour

    def test_diagonal_demand_hugs_contour(self):
        """Corner-to-corner on the plane: gradient weighting changes the
        route away from the straight diagonal."""
        f_plain = finder({"cost": 1.0}, tilted_plane_dem())
        plain = f_plain.find_route(
            source=(2.5, N - 2.5), target=(N - 2.5, 2.5))
        f_grad = finder({"gradient": 50.0, "length": 0.01},
                        tilted_plane_dem())
        grad = f_grad.find_route(
            source=(2.5, N - 2.5), target=(N - 2.5, 2.5))
        self.assertNotEqual(list(plain.path_indices),
                            list(grad.path_indices))
        self.assertGreater(grad.total_length, plain.total_length)

    def test_per_cell_layer_cannot_do_this(self):
        """On the tilted plane |∇DEM| is constant — documents WHY the
        gradient must be an edge metric."""
        dem = tilted_plane_dem()
        dy, dx = np.gradient(dem, 1.0)
        cell_slope = np.sqrt(dx ** 2 + dy ** 2)
        interior = cell_slope[1:-1, 1:-1]
        self.assertAlmostEqual(float(interior.min()),
                               float(interior.max()), places=5)


class TestHardGradeLimit(unittest.TestCase):
    def test_steep_wall_blocks_without_detour_option(self):
        """A ridge steeper than the limit must force a route around it."""
        dem = np.zeros((N, N), dtype=np.float32)
        # ridge along column 20, rows 5..N: 3 m high => steps onto it are
        # 300% slope; rows 0..4 stay flat (the pass).
        dem[5:, 19:22] = 3.0
        f = finder({"cost": 1.0}, dem,
                   gradient_options={"max_gradient_pct": 100.0})
        path = f.find_route()
        rows = np.array(path.path_indices) // f.raster_handler.data[0].shape[1]
        self.assertLess(int(rows.min()), 5)  # detours through the pass

        # without the limit the route crosses the ridge directly
        f2 = finder({"cost": 1.0}, dem)
        path2 = f2.find_route()
        rows2 = np.array(path2.path_indices) \
            // f2.raster_handler.data[0].shape[1]
        self.assertGreaterEqual(int(rows2.min()), 5)


class TestStretchGeometry(unittest.TestCase):
    def test_stretch_applies_without_gradient_terms(self):
        """Objective + DEM => 3D stretch always (plan section 0).

        A single sharp spike sits exactly on the straight line. The
        sidestep around it costs ~0.8 extra flat cells; crossing the
        spike costs ~2.5 stretch-equivalent cells — with NO gradient
        response configured, only the unconditional stretch can make the
        route sidestep."""
        dem = np.zeros((N, N), dtype=np.float32)
        dem[20, 20] = 25.0  # spike on the straight row-20 line
        f = finder({"cost": 1.0}, dem)
        path = f.find_route()
        cols = f.raster_handler.data[0].shape[1]
        idx = np.array(path.path_indices)
        spike_cell = 20 * cols + 20
        self.assertNotIn(spike_cell, idx.tolist())  # sidesteps the spike

        # identical setup WITHOUT dem: straight through the spike cell
        f2 = finder({"cost": 1.0}, None)
        idx2 = np.array(f2.find_route().path_indices)
        self.assertIn(20 * f2.raster_handler.data[0].shape[1] + 20,
                      idx2.tolist())


class TestBackendParity(unittest.TestCase):
    def test_cython_vs_networkx_same_objective(self):
        """Both backends consume the same LUTs => same achieved objective
        (routes may tie-break differently, so compare path weights)."""
        dem = tilted_plane_dem(60.0)
        objective = {"cost": 1.0, "gradient": 30.0}
        f_cy = finder(objective, dem)
        p_cy = f_cy.find_route(source=(2.5, N - 2.5),
                               target=(N - 2.5, 2.5))
        f_nx = finder(objective, dem, graph_api="networkx")
        p_nx = f_nx.find_route(source=(2.5, N - 2.5),
                               target=(N - 2.5, 2.5))

        # rebuild the exact edge weights (buffer covers the full grid, so
        # window == full grid and the original dem applies unchanged)
        steps = get_neighborhood_steps("r2", directed=True)
        obj = f_cy.objective
        luts = obj.build_gradient_luts(steps, cell_size=1.0,
                                       quant_scale=1.0)
        W = f_cy.raster_handler.data[0]
        from_n, to_n, costs = construct_edges_gradient(W, dem, steps, luts)
        weight_of = {(int(u), int(v)): float(c)
                     for u, v, c in zip(from_n, to_n, costs)}

        def path_weight(p):
            total = 0.0
            idx = list(p.path_indices)
            for u, v in zip(idx[:-1], idx[1:]):
                total += weight_of[(int(u), int(v))]
            return total

        w_cy = path_weight(p_cy)
        w_nx = path_weight(p_nx)
        self.assertAlmostEqual(w_cy, w_nx, delta=1e-6 * max(w_cy, 1.0))

    def test_kernel_vs_metric_edges_weights(self):
        """The Cython kernel and metric_edges produce identical edge
        weights for the same inputs (spot-checked via known edges)."""
        steps = get_neighborhood_steps("r1", directed=True)
        W = np.full((9, 9), 50, dtype=np.uint16)
        dem = tilted_plane_dem()[:9, :9].copy()
        obj = Objective({"cost": 1.0, "gradient": 10.0},
                        gradient_options={"multiplier": "exponential"})
        luts = obj.build_gradient_luts(steps, cell_size=1.0,
                                       quant_scale=0.5)
        from_n, to_n, costs = construct_edges_gradient(
            W, dem, steps, luts)
        # reference: recompute one straight-down edge by hand, using the
        # ACTUAL float32 height difference (bin boundaries are sensitive
        # to float32 rounding — 5*0.4 - 4*0.4 != 0.4 exactly)
        u, v = 4 * 9 + 4, 5 * 9 + 4  # step (1, 0): ~0.4 m, ~40% slope
        edge = costs[(from_n == u) & (to_n == v)]
        self.assertEqual(len(edge), 1)
        d = next(i for i in range(len(steps))
                 if steps[i, 0] == 1 and steps[i, 1] == 0)
        dh = float(np.float32(dem[5, 4]) - np.float32(dem[4, 4]))
        s_bin = min(int(abs(dh) * float(luts.bin_factor[d])),
                    luts.n_bins - 1)
        expected = (50.0 + 50.0) * (float(luts.step_len_cells[d]) / 2.0) \
            * float(luts.mult[s_bin]) \
            + float(luts.add[s_bin]) * float(luts.step_len_cells[d])
        self.assertAlmostEqual(float(edge[0]), float(expected), places=3)


class TestGuardsAndPinning(unittest.TestCase):
    def test_gradient_without_dem_rejected(self):
        with self.assertRaises(ValueError):
            finder({"gradient": 1.0, "length": 0.01}, None)

    def test_unsupported_backend_rejected(self):
        """raster_gpu is supported since Phase 5 — cugraph is not."""
        with self.assertRaises(NotImplementedError):
            finder({"gradient": 1.0, "length": 0.01},
                   tilted_plane_dem(), graph_api="cugraph")

    def test_delta_stepping_with_gradient_rejected(self):
        f = finder({"gradient": 1.0, "length": 0.01}, tilted_plane_dem())
        with self.assertRaises(NotImplementedError):
            f.find_route(algorithm="delta-stepping")

    def test_no_dem_objective_unchanged(self):
        """Objective without DEM: no gradient machinery activates."""
        f = finder({"cost": 1.0}, None)
        path = f.find_route()
        self.assertAlmostEqual(path.total_length, path.euclidean_distance,
                               delta=1.0)

    def test_kernel_null_path_pinned(self):
        """dijkstra_2d_cython without gradient kwargs is byte-identical."""
        steps = get_neighborhood_steps("r2", directed=True)
        rng = np.random.default_rng(5)
        W = (rng.integers(1, 500, size=(30, 30))).astype(np.uint16)
        a = dijkstra_2d_cython(W, steps, np.uint32(0), np.uint32(899))
        b = dijkstra_2d_cython(W, steps, np.uint32(0), np.uint32(899),
                               dem=None, gradient_luts=None)
        np.testing.assert_array_equal(a, b)


class TestMetricLayers(unittest.TestCase):
    def _vector_finder(self, metric_layers, objective):
        import geopandas as gpd
        from shapely.geometry import Polygon
        gdf = gpd.GeoDataFrame(
            {"landuse": ["field"],
             "geometry": [Polygon([(0, 0), (N, 0), (N, N), (0, N)])]},
            crs=CRS)
        return PathFinder(
            dataset_source=gdf,
            source_coords=(2.5, N / 2),
            target_coords=(N - 2.5, N / 2),
            search_space_buffer_m=200,
            graph_api="cython",
            cost_assumptions={"landuse": {"field": 100}},
            dem=tilted_plane_dem(),
            transform=TRANSFORM,
            objective=objective,
            metric_layers=metric_layers,
            resolution_in_m=1.0,
        )

    def test_terrain_slope_derived_layer(self):
        f = self._vector_finder(
            {"terrain_slope": {"derive": "slope_from_dem"}},
            {"cost": 1.0, "terrain_slope": 0.0},
        )
        f.find_route()
        self.assertIn("terrain_slope", f.metric_stack.layer_names)
        slope = f.metric_stack["terrain_slope"]
        interior = slope[2:-2, 2:-2]
        self.assertAlmostEqual(float(np.median(interior)), 40.0, delta=2.0)

    def test_prebuilt_array_layer_and_hard_max(self):
        noise = np.zeros((N, N), dtype=np.float32)
        noise[:, 20] = 9.0
        noise[:4, 20] = 0.0  # quiet gap at the top rows
        f = self._vector_finder(
            {"noise": {"source": noise, "hard_max": 5.0}},
            {"cost": 1.0, "noise": 1.0},
        )
        path = f.find_route()
        self.assertIn("noise", f.metric_stack.layer_names)
        # hard_max forbids the loud part of column 20, keeps the gap
        self.assertTrue(f.metric_stack.forbidden_mask[4:, 20].all())
        self.assertFalse(f.metric_stack.forbidden_mask[:4, 20].any())
        # the route detours through the quiet gap
        cols = f.raster_handler.data[0].shape[1]
        rows_on_20 = [int(i) // cols for i in path.path_indices
                      if int(i) % cols == 20]
        self.assertTrue(all(r < 4 for r in rows_on_20))

    def test_metric_layers_require_objective(self):
        with self.assertRaises(ValueError):
            PathFinder(
                dataset_source=uniform_raster(),
                crs=CRS, transform=TRANSFORM,
                source_coords=(2.5, 20.5), target_coords=(38.5, 20.5),
                search_space_buffer_m=200,
                metric_layers={"noise": np.zeros((N, N))},
            )


if __name__ == "__main__":
    unittest.main()

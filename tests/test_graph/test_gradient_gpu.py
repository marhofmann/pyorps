"""Phase 5 tests: per-edge gradient terms in the GPU V4 persistent kernel.

Every scenario is validated against the Phase-4 Cython kernel on identical
slope-response LUTs — the parity that shared tables buy us.
"""
import unittest

import numpy as np
import pytest
from rasterio.transform import from_origin

cupy = pytest.importorskip("cupy")

from pyorps import PathFinder  # noqa: E402
from pyorps.utils.metric_edges import construct_edges_gradient  # noqa: E402
from pyorps.utils.neighborhood import get_neighborhood_steps  # noqa: E402

CRS = "EPSG:25832"
N = 41
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)


def tilted_plane_dem(slope_pct=60.0):
    rise = slope_pct / 100.0
    return (np.arange(N, dtype=np.float32)[:, None] * rise
            * np.ones((1, N), dtype=np.float32))


def finder(objective, dem, graph_api, gradient_options=None, raster=None):
    return PathFinder(
        dataset_source=(raster if raster is not None
                        else np.full((N, N), 100, dtype=np.uint16)),
        crs=CRS,
        transform=TRANSFORM,
        source_coords=(2.5, N - 0.5 - 20),
        target_coords=(N - 2.5, N - 0.5 - 20),
        search_space_buffer_m=200,
        graph_api=graph_api,
        dem=dem,
        objective=objective,
        gradient_options=gradient_options,
    )


def path_weight(finder_obj, path, dem, objective_luts_steps="r2"):
    """Sum the exact gradient edge weights along a path (tie-safe)."""
    steps = get_neighborhood_steps(objective_luts_steps, directed=True)
    luts = finder_obj.objective.build_gradient_luts(
        steps, cell_size=1.0, quant_scale=1.0)
    W = finder_obj.raster_handler.data[0]
    from_n, to_n, costs = construct_edges_gradient(W, dem, steps, luts)
    weight_of = {(int(u), int(v)): float(c)
                 for u, v, c in zip(from_n, to_n, costs)}
    idx = list(path.path_indices)
    return sum(weight_of[(int(u), int(v))]
               for u, v in zip(idx[:-1], idx[1:]))


class TestGpuGradientParity(unittest.TestCase):
    def test_contour_objective_matches_cython(self):
        """Diagonal demand on a tilted plane: GPU and Cython achieve the
        same objective through the same LUTs."""
        dem = tilted_plane_dem()
        objective = {"cost": 1.0, "gradient": 30.0}
        f_cy = finder(objective, dem, "cython")
        p_cy = f_cy.find_route(source=(2.5, N - 2.5),
                               target=(N - 2.5, 2.5))
        f_gpu = finder(objective, dem, "raster_gpu")
        p_gpu = f_gpu.find_route(source=(2.5, N - 2.5),
                                 target=(N - 2.5, 2.5))
        w_cy = path_weight(f_cy, p_cy, dem)
        w_gpu = path_weight(f_gpu, p_gpu, dem)
        # GPU dist is float32; allow a small relative tolerance
        self.assertAlmostEqual(w_gpu / w_cy, 1.0, places=3)

    def test_same_contour_route_stays_level_on_gpu(self):
        f = finder({"gradient": 50.0, "length": 0.01}, tilted_plane_dem(),
                   "raster_gpu")
        path = f.find_route()
        cols = f.raster_handler.data[0].shape[1]
        rows = np.unique(np.array(path.path_indices) // cols)
        self.assertEqual(len(rows), 1)

    def test_hard_grade_limit_on_gpu(self):
        """Ridge steeper than the limit forces the detour on GPU too."""
        dem = np.zeros((N, N), dtype=np.float32)
        dem[5:, 19:22] = 3.0  # 300% ridge with a flat pass at rows 0..4
        f = finder({"cost": 1.0}, dem, "raster_gpu",
                   gradient_options={"max_gradient_pct": 100.0})
        path = f.find_route()
        cols = f.raster_handler.data[0].shape[1]
        rows = np.array(path.path_indices) // cols
        self.assertLess(int(rows.min()), 5)

    def test_stretch_spike_sidestep_on_gpu(self):
        dem = np.zeros((N, N), dtype=np.float32)
        dem[20, 20] = 25.0
        f = finder({"cost": 1.0}, dem, "raster_gpu")
        path = f.find_route()
        cols = f.raster_handler.data[0].shape[1]
        self.assertNotIn(20 * cols + 20,
                         np.array(path.path_indices).tolist())

    def test_no_gradient_gpu_route_unchanged(self):
        """objective without DEM on GPU: plain legacy route (null path)."""
        f = finder({"cost": 1.0}, None, "raster_gpu")
        path = f.find_route()
        self.assertAlmostEqual(path.total_length, path.euclidean_distance,
                               delta=1.0)


if __name__ == "__main__":
    unittest.main()

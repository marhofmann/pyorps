"""Path.total_length is a GROUND-TRUTH length in CRS units, not a cell count.

``calculate_path_metrics_numba`` is deliberately a pure cell-space kernel: it
never sees a transform and can only count steps (1.0 orthogonal, sqrt(2)
diagonal). The conversion to metres therefore has to happen at the boundary —
in ``PathFinder.calculate_path_metrics`` and in the GUI's
``evaluate_route_cost`` — and it has to use the one cell-size convention the
rest of pyorps uses, ``|window_transform.a|`` (pyorps assumes square cells).

The invariant pinned here: the SAME physical route, rasterized at different
resolutions, reports the SAME length. This was invisible for a long time
because virtually every fixture in the suite is a 1 m grid, where a cell count
and a metre count are numerically identical.
"""
import unittest

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString

from pyorps import PathFinder

# The GUI parity check below needs pyorps.gui, which imports dash — an
# optional extra ([gui]/[viz]) that is in neither the `dev` nor the `full`
# install. An unguarded import here would turn a missing extra into a
# COLLECTION error for the whole tests/test_graph directory, so only the one
# test that needs it is skipped.
try:
    from pyorps.gui.services.cost import evaluate_route_cost
except ImportError:                              # pragma: no cover
    evaluate_route_cost = None

CRS = "EPSG:25832"

#: A 400 m x 400 m uniform field. Both endpoints sit on cell boundaries at
#: every resolution below, so the discretized route is exactly the straight
#: 360 m connection — no rounding slack to hide a scaling error in.
EXTENT_M = 400.0
SOURCE = (20.0, 200.0)
TARGET = (380.0, 200.0)
TRUE_LENGTH_M = 360.0
CELL_COST = 10

#: Includes non-integer (2.5) and coarse (5.0) cells: a scaling bug that
#: happened to cancel on power-of-two grids would survive a 1/2/4 sweep.
RESOLUTIONS = (1.0, 2.0, 2.5, 4.0, 5.0)

#: A 45-degree route across the same field. The orthogonal route above only
#: exercises unit steps; this one is all sqrt(2) diagonals, which is where a
#: fix that scaled the step COUNT instead of the step LENGTH would show up.
DIAG_SOURCE = (20.0, 20.0)
DIAG_TARGET = (380.0, 380.0)
DIAG_TRUE_LENGTH_M = 360.0 * 2 ** 0.5          # 509.1169 m


def make_finder(resolution: float, source=SOURCE, target=TARGET,
                buffer_m: float = 50) -> PathFinder:
    """A finder over a uniform cost surface at *resolution* metre cells.

    Uniform cost makes the straight connection strictly optimal: a diagonal
    step costs sqrt(2) x an orthogonal one, so the solver has no tie to
    break and the route is the same physical line at every resolution.
    """
    n = int(EXTENT_M / resolution)
    raster = np.full((n, n), CELL_COST, dtype=np.uint16)
    return PathFinder(
        dataset_source=raster,
        crs=CRS,
        transform=from_origin(0.0, EXTENT_M, resolution, resolution),
        source_coords=source,
        target_coords=target,
        search_space_buffer_m=buffer_m,
        graph_api="cython",
    )


class TestTotalLengthIsMetres(unittest.TestCase):
    def test_length_is_resolution_invariant(self):
        """360 m of ground truth reads as ~360 m at 1 m, 2 m and 4 m cells."""
        for resolution in RESOLUTIONS:
            with self.subTest(resolution=resolution):
                path = make_finder(resolution).find_route()
                # Tolerance is one cell: the endpoints snap to cell centres.
                self.assertAlmostEqual(path.total_length, TRUE_LENGTH_M,
                                       delta=resolution)
                # euclidean_distance is built from world coordinates and was
                # always in metres — the detour factor only means something
                # once total_length shares those units.
                self.assertAlmostEqual(path.euclidean_distance, TRUE_LENGTH_M,
                                       delta=1e-6)
                self.assertAlmostEqual(
                    path.total_length / path.euclidean_distance, 1.0,
                    delta=resolution / TRUE_LENGTH_M)

    def test_diagonal_steps_scale_too(self):
        """An all-diagonal route: sqrt(2) x cell_size, not sqrt(2) x 1.

        Scaling a step COUNT rather than a step LENGTH would still pass the
        orthogonal test above, because there 1 step == 1 cell_size. Here
        every step is sqrt(2) cells, so the two differ.
        """
        for resolution in RESOLUTIONS:
            with self.subTest(resolution=resolution):
                path = make_finder(resolution, DIAG_SOURCE, DIAG_TARGET,
                                   buffer_m=80).find_route()
                self.assertAlmostEqual(path.total_length,
                                       DIAG_TRUE_LENGTH_M,
                                       delta=2 * resolution)
                self.assertAlmostEqual(path.euclidean_distance,
                                       DIAG_TRUE_LENGTH_M, delta=1e-6)

    def test_category_lengths_and_cost_scale_with_the_total(self):
        """length_by_category is metres too, so total_cost is EUR/m x m."""
        for resolution in RESOLUTIONS:
            with self.subTest(resolution=resolution):
                path = make_finder(resolution).find_route()
                self.assertAlmostEqual(sum(path.length_by_category.values()),
                                       path.total_length, places=6)
                # One category only, so cost = 10 EUR/m x 360 m.
                self.assertAlmostEqual(path.total_cost,
                                       CELL_COST * TRUE_LENGTH_M,
                                       delta=CELL_COST * resolution)
                # Percentages are ratios and must stay unaffected.
                self.assertAlmostEqual(
                    path.length_by_category_percent[CELL_COST], 100.0,
                    places=6)

    @unittest.skipIf(evaluate_route_cost is None,
                     "pyorps.gui needs the optional [gui] extra (dash)")
    def test_gui_route_cost_agrees_at_every_resolution(self):
        """The GUI's edit-cost metric must not drift from PathFinder's.

        Both sites call the same cell-space kernel, so a fix applied to only
        one of them shows up here (and nowhere else — the existing parity
        checks in tests/test_webviz and tests/test_gui all run at 1 m, where
        cells and metres coincide).
        """
        line = LineString([SOURCE, TARGET])
        for resolution in RESOLUTIONS:
            with self.subTest(resolution=resolution):
                finder = make_finder(resolution)
                path = finder.find_route()
                cost = evaluate_route_cost(line, finder.raster_handler)
                self.assertAlmostEqual(cost.total_length, path.total_length,
                                       delta=resolution)
                self.assertAlmostEqual(cost.total_length,
                                       cost.geodesic_length_m,
                                       delta=resolution)
                self.assertAlmostEqual(cost.total_cost, path.total_cost,
                                       delta=CELL_COST * resolution)


if __name__ == "__main__":
    unittest.main()

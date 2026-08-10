"""Phase 2 corridor-first data path: items 2.2 and 2.6.

The contract these tests exist to defend is exactness, not speed:

* item 2.2 — the fetch/read/burn stages are restricted to the search
  corridor, and the raster the solver ends up with must be BIT-IDENTICAL to
  the one it saw when every stage ran over the whole data extent. The tests
  below demonstrate that by building both rasters and comparing them cell by
  cell, rather than asserting it from the design;
* item 2.6 — the objective is scalarized on the search window instead of the
  full stack. That one is deliberately NOT bit-identical: the uint16
  quantization scale is derived from the windowed maximum. The tests pin down
  both halves of that statement — identical when the maximum lies inside the
  window, and strictly finer (never coarser) when it does not, with the
  float-based per-metric reporting unchanged either way.

Everything here runs on the CPU cython backend and on rasters of a few tens
of thousands of cells.
"""
import inspect
import shutil
import tempfile
import unittest
from os.path import join

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from pyorps.core.metric_stack import MetricStack
from pyorps.graph.path_finder import PathFinder
from pyorps.raster.handler import RasterHandler

CRS = "EPSG:25832"

#: The burn extent: 400 m x 400 m, 2 m cells -> a 200 x 200 raster whose
#: pixel size comes out exactly 2.0 (round bounds), i.e. the case in which
#: the corridor grid reproduces the full grid bit-for-bit.
EXTENT = (0.0, 0.0, 400.0, 400.0)
RESOLUTION = 2.0
FULL_SHAPE = (200, 200)

#: A horizontal pair 360 m apart; with a 30 m buffer the corridor is a
#: 60 m tall band -> 30 of the 200 raster rows, 15 % of the burn.
SOURCE = (20.0, 200.0)
TARGET = (380.0, 200.0)
BUFFER_M = 30.0


def landuse_gdf(with_far_expensive_class: bool = False) -> gpd.GeoDataFrame:
    """Background + a barrier with a gap, both inside the corridor band.

    ``with_far_expensive_class`` adds a class that occurs ONLY outside the
    corridor. It does not change any cell the search can reach; it only moves
    the GLOBAL maximum of the objective, which is what makes the item 2.6
    quantization consequence observable.
    """
    records = [
        ("field", box(*EXTENT)),
        # Vertical barrier at x in [200, 210] with a gap at y in [215, 225].
        ("barrier", Polygon([(200, 100), (210, 100), (210, 215), (200, 215)])),
        ("barrier", Polygon([(200, 225), (210, 225), (210, 300), (200, 300)])),
        # A more expensive but passable band across the corridor.
        ("forest", box(280.0, 180.0, 320.0, 220.0)),
    ]
    if with_far_expensive_class:
        # y >= 350 is 120 m above the corridor's upper edge (y = 230).
        records.append(("quarry", box(0.0, 350.0, 400.0, 400.0)))
    return gpd.GeoDataFrame(
        {"landuse": [name for name, _ in records],
         "geometry": [geom for _, geom in records]},
        crs=CRS)


COST_ASSUMPTIONS = {
    "landuse": {
        "field": 10,
        "forest": 40,
        "quarry": 3000,
        "barrier": 65535,
    }
}

METRIC_ASSUMPTIONS = {
    "landuse": {
        "field": {"cost": 10.0, "landscape": 0.0},
        "forest": {"cost": 40.0, "landscape": 5.0},
        "quarry": {"cost": 3000.0, "landscape": 0.0},
        "barrier": {"cost": 65535.0, "landscape": 0.0},
    }
}


def window_tuple(window) -> tuple[int, int, int, int]:
    return (int(window.col_off), int(window.row_off),
            int(window.width), int(window.height))


def vector_finder(corridor_first: bool, buffer_m=BUFFER_M,
                  objective=None, far_class: bool = False) -> PathFinder:
    """A PathFinder over freshly built vector input.

    A fresh GeoDataFrame per finder: GeoRasterizer writes the derived cost
    column into the frame it is handed, so sharing one would couple the two
    sides of every comparison.
    """
    return PathFinder(
        dataset_source=landuse_gdf(far_class),
        source_coords=SOURCE,
        target_coords=TARGET,
        search_space_buffer_m=buffer_m,
        graph_api="cython",
        cost_assumptions=(METRIC_ASSUMPTIONS if objective is not None
                          else COST_ASSUMPTIONS),
        objective=objective,
        corridor_first=corridor_first,
        resolution_in_m=RESOLUTION,
    )


class TestCorridorGeometry(unittest.TestCase):
    """The corridor PathFinder derives must be the one the handler uses."""

    def test_matches_the_handler_buffer_geometry(self):
        finder = vector_finder(corridor_first=True)
        self.assertTrue(
            finder.raster_handler.buffer_geometry.equals(
                finder.corridor_geometry()),
            "PathFinder.corridor_geometry has drifted away from the "
            "geometry RasterHandler buffers; every corridor-first stage "
            "would then act on a different area than the search.")

    def test_bounds_helper_agrees_with_the_geometry(self):
        finder = vector_finder(corridor_first=True)
        self.assertEqual(finder.corridor_bounds(),
                         finder.corridor_geometry().bounds)

    def test_no_corridor_without_a_positive_buffer(self):
        finder = vector_finder(corridor_first=True)
        self.assertIsNone(finder.corridor_geometry(0))
        self.assertIsNone(finder.corridor_bounds(0))


class TestVectorBurnCorridor(unittest.TestCase):
    """Item 2.2: burn the corridor, not the data extent."""

    @classmethod
    def setUpClass(cls):
        cls.full = vector_finder(corridor_first=False)
        cls.corridor = vector_finder(corridor_first=True)
        cls.full_path = cls.full.find_route()
        cls.corridor_path = cls.corridor.find_route()

    def test_the_corridor_burn_actually_engaged(self):
        """Guard: without this the identity assertions below are vacuous."""
        self.assertEqual(self.full.geo_rasterizer.raster.shape, FULL_SHAPE)
        burned = self.corridor.geo_rasterizer.raster
        self.assertLess(burned.size, self.full.geo_rasterizer.raster.size)
        # 30 of 200 rows: the corridor band, full width.
        self.assertEqual(burned.shape, (30, 200))
        self.assertLess(burned.size / self.full.geo_rasterizer.raster.size,
                        0.25)

    def test_search_raster_is_bit_identical(self):
        """What the solver sees, cell for cell, from both burns."""
        full_data = self.full.raster_handler.data
        corridor_data = self.corridor.raster_handler.data
        self.assertEqual(full_data.shape, corridor_data.shape)
        self.assertEqual(full_data.dtype, corridor_data.dtype)
        self.assertTrue(np.array_equal(full_data, corridor_data),
                        "the corridor burn changed cell values inside the "
                        "search window")

    def test_window_transform_is_the_same_grid(self):
        full_t = self.full.raster_handler.window_transform
        corridor_t = self.corridor.raster_handler.window_transform
        for attribute in ("a", "b", "c", "d", "e", "f"):
            self.assertAlmostEqual(
                getattr(full_t, attribute), getattr(corridor_t, attribute),
                delta=PathFinder.CORRIDOR_GRID_TOLERANCE_M,
                msg=f"window transform component {attribute!r} moved")

    def test_route_and_cost_are_identical(self):
        self.assertEqual(list(self.full_path.path_indices),
                         list(self.corridor_path.path_indices))
        self.assertTrue(np.allclose(self.full_path.path_coords,
                                    self.corridor_path.path_coords))
        self.assertEqual(self.full_path.total_cost,
                         self.corridor_path.total_cost)
        self.assertEqual(self.full_path.total_length,
                         self.corridor_path.total_length)

    def test_route_actually_detours_through_the_gap(self):
        """Guard: a straight line would not exercise the burned barrier."""
        self.assertGreater(self.corridor_path.total_length,
                           self.corridor_path.euclidean_distance + 5.0)

    def test_certificate_refuses_a_user_supplied_bounding_box(self):
        finder = vector_finder(corridor_first=True)
        kwargs = {"resolution_in_m": RESOLUTION,
                  "bounding_box": box(*EXTENT)}
        self.assertIs(finder._with_corridor_bounding_box(kwargs), kwargs)

    def test_certificate_refuses_a_preprocessing_hook(self):
        finder = vector_finder(corridor_first=True)
        self.assertIsNone(finder._certified_corridor_bounding_box(
            resolution_in_m=RESOLUTION,
            preprocessing_function=lambda data: None))


class TestNoneBufferFallback(unittest.TestCase):
    """Item 2.2: with no explicit buffer, nothing may shrink.

    ``RasterHandler.estimate_buffer_width`` samples the raster it is handed
    at indices clamped against that raster's SHAPE, so a smaller raster can
    yield a different estimate — and a smaller estimate would shrink the
    searched area below what the caller would have got. The corridor-first
    steps therefore stand down entirely.
    """

    @classmethod
    def setUpClass(cls):
        cls.reference = vector_finder(corridor_first=False, buffer_m=None)
        cls.corridor = vector_finder(corridor_first=True, buffer_m=None)
        cls.reference_path = cls.reference.find_route()
        cls.corridor_path = cls.corridor.find_route()

    def test_the_burn_was_not_shrunk(self):
        self.assertEqual(self.corridor.geo_rasterizer.raster.shape,
                         FULL_SHAPE)

    def test_certificate_returns_none(self):
        self.assertIsNone(
            self.corridor._certified_corridor_bounding_box(
                resolution_in_m=RESOLUTION))

    def test_estimated_buffer_and_searched_area_are_unchanged(self):
        self.assertEqual(self.corridor.search_space_buffer_m,
                         self.reference.search_space_buffer_m)
        self.assertEqual(self.corridor.raster_handler.data.shape,
                         self.reference.raster_handler.data.shape)
        self.assertEqual(list(self.corridor_path.path_indices),
                         list(self.reference_path.path_indices))
        self.assertEqual(self.corridor_path.total_cost,
                         self.reference_path.total_cost)

    def test_fallback_is_the_estimator_clamp_maximum(self):
        """4000 m is the widest buffer the estimator can ever return."""
        signature = inspect.signature(RasterHandler.estimate_buffer_width)
        self.assertGreaterEqual(
            PathFinder.CORRIDOR_FALLBACK_BUFFER_M,
            signature.parameters["max_buffer"].default)

    def test_fallback_corridor_contains_the_estimated_one(self):
        estimated = self.corridor.corridor_geometry(
            self.corridor.search_space_buffer_m)
        fallback = self.corridor.corridor_geometry(
            PathFinder.CORRIDOR_FALLBACK_BUFFER_M)
        self.assertTrue(fallback.contains(estimated))


class TestWindowedRasterRead(unittest.TestCase):
    """Item 2.2: a raster/DEM file is read window-first, not whole."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="pyorps_phase2_")
        cls.path = join(cls.directory, "cost.tif")
        rng = np.random.default_rng(20260807)
        data = rng.integers(1, 60, size=(1, *FULL_SHAPE), dtype=np.uint16)
        data[0, 40:160, 100:103] = 65535          # barrier
        data[0, 88:93, 100:103] = 5               # gap through it
        cls.data = data
        with rasterio.open(
                cls.path, "w", driver="GTiff",
                height=FULL_SHAPE[0], width=FULL_SHAPE[1], count=1,
                dtype="uint16", crs=CRS,
                transform=from_origin(EXTENT[0], EXTENT[3],
                                      RESOLUTION, RESOLUTION)) as destination:
            destination.write(data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def _finder(self, corridor_first, buffer_m=BUFFER_M):
        return PathFinder(
            dataset_source=self.path,
            source_coords=SOURCE,
            target_coords=TARGET,
            search_space_buffer_m=buffer_m,
            graph_api="cython",
            corridor_first=corridor_first,
        )

    def test_only_the_window_is_read(self):
        finder = self._finder(corridor_first=True)
        self.assertTrue(finder.raster_handler.windowed_source_read)
        self.assertIsNone(finder.dataset.data,
                          "the whole raster was pulled into memory anyway")

    def test_full_read_when_switched_off(self):
        finder = self._finder(corridor_first=False)
        self.assertFalse(finder.raster_handler.windowed_source_read)
        self.assertIsNotNone(finder.dataset.data)

    def test_window_and_values_are_identical(self):
        windowed = self._finder(corridor_first=True)
        full = self._finder(corridor_first=False)
        self.assertEqual(window_tuple(windowed.raster_handler.window),
                         window_tuple(full.raster_handler.window))
        self.assertTrue(np.array_equal(windowed.raster_handler.data,
                                       full.raster_handler.data))

    def test_route_is_identical(self):
        windowed = self._finder(corridor_first=True).find_route()
        full = self._finder(corridor_first=False).find_route()
        self.assertEqual(list(windowed.path_indices), list(full.path_indices))
        self.assertEqual(windowed.total_cost, full.total_cost)

    def test_none_buffer_still_reads_the_whole_file(self):
        finder = self._finder(corridor_first=True, buffer_m=None)
        self.assertFalse(finder.raster_handler.windowed_source_read)
        self.assertIsNotNone(finder.dataset.data)


class TestWindowFirstCombine(unittest.TestCase):
    """Item 2.6: scalarize the window, not the extent."""

    def test_stack_is_narrowed_to_the_search_window(self):
        finder = vector_finder(corridor_first=True, objective={"cost": 1.0,
                                                              "landscape": 2.0})
        handler = finder.raster_handler
        self.assertEqual(finder.metric_stack.shape, (30, 200))
        self.assertEqual(int(handler.window.row_off), 0)
        self.assertEqual(int(handler.window.col_off), 0)
        # The invariant the metric evaluator and ConstrainedPathFinder rely
        # on: the handler window indexes into metric_stack.
        self.assertEqual(handler.data.shape[1:], finder.metric_stack.shape)

    def test_not_narrowed_without_an_explicit_buffer(self):
        finder = vector_finder(corridor_first=True, buffer_m=None,
                               objective={"cost": 1.0, "landscape": 2.0})
        self.assertEqual(finder.metric_stack.shape, FULL_SHAPE)

    def test_identical_when_the_maximum_lies_inside_the_window(self):
        """No far class: windowed max == global max => same scale, same bits."""
        objective = {"cost": 1.0, "landscape": 200.0}
        full = vector_finder(corridor_first=False, objective=objective)
        corridor = vector_finder(corridor_first=True, objective=objective)
        self.assertEqual(full.metric_stack.shape, FULL_SHAPE)
        self.assertEqual(corridor.metric_stack.shape, (30, 200))
        self.assertEqual(full._combine_result.scale,
                         corridor._combine_result.scale)
        self.assertTrue(np.array_equal(full.raster_handler.data,
                                       corridor.raster_handler.data))
        full_path = full.find_route()
        corridor_path = corridor.find_route()
        self.assertEqual(list(full_path.path_indices),
                         list(corridor_path.path_indices))
        self.assertEqual(full_path.total_cost, corridor_path.total_cost)

    def test_scale_is_finer_when_the_maximum_lies_outside(self):
        """The documented semantic consequence, pinned down in both directions.

        The quarry class exists only 120 m above the corridor. It cannot be
        reached, but it used to set the quantization scale for the whole
        raster. Windowing first drops it, so the same objective is resolved
        on a FINER grid — never a coarser one — and the float-based metric
        reporting is unaffected.
        """
        objective = {"cost": 1.0, "landscape": 200.0}
        full = vector_finder(corridor_first=False, objective=objective,
                             far_class=True)
        corridor = vector_finder(corridor_first=True, objective=objective,
                                 far_class=True)
        self.assertGreater(corridor._combine_result.scale,
                           full._combine_result.scale)
        self.assertLess(corridor._combine_result.resolution,
                        full._combine_result.resolution)

        full_path = full.find_route()
        corridor_path = corridor.find_route()
        self.assertTrue(np.allclose(full_path.path_coords,
                                    corridor_path.path_coords))
        # Reported metrics come from the unquantized float layers.
        for name, value in full_path.metrics.items():
            self.assertAlmostEqual(value, corridor_path.metrics[name],
                                   places=6, msg=f"metric {name!r} moved")
        # ... and the scale in force is recorded on the result.
        self.assertEqual(corridor_path.objective_spec["quantization_scale"],
                         corridor._combine_result.scale)

    def test_legacy_alias_stack_keeps_its_zero_copy_passthrough(self):
        """A single uint16 raster must not be materialized into layers."""
        raster = np.full(FULL_SHAPE, 10, dtype=np.uint16)
        finder = PathFinder(
            dataset_source=raster,
            crs=CRS,
            transform=from_origin(EXTENT[0], EXTENT[3],
                                  RESOLUTION, RESOLUTION),
            source_coords=SOURCE,
            target_coords=TARGET,
            search_space_buffer_m=BUFFER_M,
            graph_api="cython",
            objective={"cost": 1.0},
            corridor_first=True,
        )
        self.assertTrue(finder._combine_result.legacy_passthrough)
        self.assertEqual(finder.metric_stack.shape, FULL_SHAPE)


class TestAddLayerCopy(unittest.TestCase):
    """Item 2.6 follow-up: add_layer no longer copies unconditionally."""

    def _stack(self):
        stack = MetricStack(from_origin(0.0, 10.0, 1.0, 1.0), CRS)
        stack.add_layer("cost", np.full((10, 10), 5.0, dtype=np.float32))
        return stack

    def test_default_protects_the_callers_array(self):
        stack = self._stack()
        values = np.full((10, 10), 3.0, dtype=np.float32)
        values[0, 0] = np.inf                    # forbidden -> written to 0
        stack.add_layer("landscape", values)
        self.assertTrue(np.isinf(values[0, 0]),
                        "add_layer wrote into the caller's array")
        self.assertIsNot(stack["landscape"], values)

    def test_copy_false_adopts_the_array(self):
        stack = self._stack()
        values = np.full((10, 10), 3.0, dtype=np.float32)
        stack.add_layer("landscape", values, copy=False)
        self.assertIs(stack["landscape"], values)

    def test_conversion_still_never_aliases(self):
        stack = self._stack()
        values = np.full((10, 10), 3, dtype=np.uint16)
        stack.add_layer("landscape", values)
        self.assertFalse(np.may_share_memory(stack["landscape"], values))

    def test_values_are_unchanged_by_the_copy_policy(self):
        copied = self._stack()
        adopted = self._stack()
        base = np.arange(100, dtype=np.float32).reshape(10, 10)
        copied.add_layer("landscape", base.copy())
        adopted.add_layer("landscape", base.copy(), copy=False)
        self.assertTrue(np.array_equal(copied["landscape"],
                                       adopted["landscape"]))
        self.assertTrue(np.array_equal(copied.forbidden_mask,
                                       adopted.forbidden_mask))


if __name__ == "__main__":
    unittest.main()

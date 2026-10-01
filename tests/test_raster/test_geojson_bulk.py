"""Bit-identity of the bulk shapely -> GeoJSON path (pyorps/raster/_geojson.py).

Every test here compares FULL arrays with ``np.array_equal``: pyorps burns
land-use polygons into a uint16 cost surface and then runs shortest paths on
it, so a single changed cell can change the optimal route.
"""

import unittest
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)

from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster import rasterizer as rasterizer_module
from pyorps.raster._geojson import bulk_geojson, geojson_shapes
from pyorps.raster.rasterizer import GeoRasterizer

# A plain 1 m grid for the synthetic fixtures, and a second transform at real
# EPSG:25832 magnitudes (easting ~3.2e7) because the half-pixel tie behaviour
# of a scan converter is sensitive to coordinate magnitude -- toy 0..100
# coordinates would not exercise the float64 round trip that matters.
TRANSFORM = from_origin(0, 100, 1, 1)
SHAPE = (100, 100)
UTM_TRANSFORM = from_origin(32_400_000.0, 5_600_000.0, 1, 1)


def _square(x, y, size=10):
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def _burn(geometries, values, transform=TRANSFORM, shape=SHAPE):
    """Burn ``(geom, value)`` pairs exactly as the production sites do."""
    return rasterize(
        ((geom, int(value)) for geom, value in zip(geometries, values)),
        out_shape=shape,
        fill=0,
        dtype="uint16",
        transform=transform,
    )


def _mask(geometries, transform=TRANSFORM, shape=SHAPE):
    return geometry_mask(geometries, transform=transform, invert=True,
                         out_shape=shape)


class BulkGeoJsonParityTests(unittest.TestCase):
    """The burned raster must not change when the bulk path is used."""

    def assert_burn_identical(self, geometries, values=None,
                              transform=TRANSFORM, shape=SHAPE):
        geometries = list(geometries)
        if values is None:
            values = range(1, len(geometries) + 1)
        values = list(values)
        converted = geojson_shapes(np.asarray(geometries, dtype=object))
        legacy = _burn(geometries, values, transform, shape)
        bulk = _burn(converted, values, transform, shape)
        self.assertTrue(np.array_equal(legacy, bulk))
        return legacy

    def test_mixed_polygon_multipolygon_heavy_overlap(self):
        """Painting order is load-bearing; the bulk path must preserve it."""
        geometries = []
        for index in range(60):
            offset = index * 0.7
            if index % 3 == 0:
                geometries.append(MultiPolygon([
                    _square(offset, offset, 25),
                    _square(offset + 30, offset + 10, 20),
                ]))
            else:
                geometries.append(_square(offset, 100 - offset - 25, 25))

        forward = self.assert_burn_identical(geometries)

        # Guard the guard: if order did NOT matter this test would pass
        # vacuously, so prove that reversing the sequence changes the raster.
        reversed_geoms = geometries[::-1]
        reversed_burn = _burn(reversed_geoms, range(1, len(geometries) + 1))
        self.assertFalse(np.array_equal(forward, reversed_burn))
        # ...and that the bulk path reproduces the REVERSED order too.
        self.assertTrue(np.array_equal(
            reversed_burn,
            _burn(geojson_shapes(np.asarray(reversed_geoms, dtype=object)),
                  range(1, len(geometries) + 1)),
        ))

    def test_holed_and_multipart_holed_polygons(self):
        """Interior rings must stay interior rings through the round trip."""
        holed = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)],
                        [[(10, 10), (30, 10), (30, 30), (10, 30)]])
        multi_holed = MultiPolygon([
            (((50, 50), (90, 50), (90, 90), (50, 90)),
             [((60, 60), (80, 60), (80, 80), (60, 80))]),
            (((10, 60), (30, 60), (30, 80), (10, 80)),
             [((15, 65), (25, 65), (25, 75), (15, 75))]),
        ])
        burned = self.assert_burn_identical([holed, multi_holed, _square(35, 35)])
        # The hole must really be a hole, or the fixture proves nothing.
        self.assertEqual(burned[75, 20], 0)

    def test_invalid_bowtie_polygon_is_not_gated_out(self):
        """rasterio burns self-intersecting rings; so must the bulk path."""
        bowtie = Polygon([(10, 10), (30, 30), (10, 30), (30, 10)])
        self.assertFalse(bowtie.is_valid)
        burned = self.assert_burn_identical([bowtie, _square(60, 60)])
        self.assertTrue(burned.any())

    def test_nan_coordinate_polygon(self):
        """A NaN vertex burns zero cells today; it must still burn zero."""
        nan_poly = Polygon([(10, 10), (np.nan, 30), (30, 30), (30, 10)])
        self.assert_burn_identical([nan_poly])
        self.assert_burn_identical([_square(50, 50), nan_poly, _square(55, 55)])

    def test_real_world_utm_magnitudes(self):
        """Parity at EPSG:25832 easting magnitudes, not toy coordinates."""
        origin_x, origin_y = 32_400_000.0, 5_600_000.0
        geometries = []
        for index in range(40):
            x = origin_x + index * 1.37
            y = origin_y - 100 + index * 0.91
            geometries.append(_square(x, y, 23.5))
            if index % 4 == 0:
                geometries.append(MultiPolygon([
                    _square(x + 5.25, y + 3.75, 11.5),
                    _square(x + 40.5, y + 12.25, 9.75),
                ]))
        burned = self.assert_burn_identical(geometries, transform=UTM_TRANSFORM)
        self.assertTrue(burned.any())

    def test_three_dimensional_polygons(self):
        """z is dropped on the bulk path; rasterio only ever reads x/y."""
        flat = _square(10, 10, 30)
        with_z = Polygon([(20, 20, 5), (60, 20, 5), (60, 60, 7), (20, 60, 7)])
        self.assert_burn_identical([flat, with_z])
        self.assertNotIn(3, np.shape(
            bulk_geojson([with_z])[0]["coordinates"][0][0]))

    def test_geometry_mask_bare_geometries(self):
        """geometry_mask takes BARE geometries, not (geom, value) tuples."""
        geometries = [
            _square(5, 5, 30),
            Polygon([(40, 40), (90, 40), (90, 90), (40, 90)],
                    [[(55, 55), (75, 55), (75, 75), (55, 75)]]),
            MultiPolygon([_square(10, 70, 15), _square(70, 10, 15)]),
        ]
        array = np.asarray(geometries, dtype=object)
        self.assertTrue(np.array_equal(_mask(geometries),
                                       _mask(geojson_shapes(array))))

    def test_single_geometry_and_one_cell_raster(self):
        self.assert_burn_identical([_square(20, 20, 40)])
        self.assert_burn_identical([_square(20, 20, 40)],
                                   transform=from_origin(0, 100, 100, 100),
                                   shape=(1, 1))


class BulkGeoJsonEdgeCaseTests(unittest.TestCase):
    """Cases the fast path must either ride along with, or refuse."""

    def test_none_geometry_rides_along_positionally(self):
        """None is common in WFS output; gating on it would kill the win."""
        geometries = [_square(10, 10), None, _square(30, 30)]
        converted = bulk_geojson(geometries)
        self.assertIsNotNone(converted)
        self.assertEqual(len(converted), 3)
        self.assertEqual(converted[1]["coordinates"], [])
        with warnings.catch_warnings(record=True) as legacy_warnings:
            warnings.simplefilter("always")
            legacy = _burn(geometries, [1, 2, 3])
        with warnings.catch_warnings(record=True) as bulk_warnings:
            warnings.simplefilter("always")
            bulk = _burn(converted, [1, 2, 3])
        self.assertTrue(np.array_equal(legacy, bulk))
        # Same skip COUNT (the message text differs; nothing depends on it).
        self.assertEqual(len(legacy_warnings), len(bulk_warnings))

    def test_empty_polygon_rides_along(self):
        geometries = [_square(10, 10), Polygon(), _square(30, 30)]
        converted = bulk_geojson(geometries)
        self.assertIsNotNone(converted)
        self.assertTrue(np.array_equal(_burn(geometries, [1, 2, 3]),
                                       _burn(converted, [1, 2, 3])))

    def test_mask_with_none_geometry(self):
        geometries = [_square(10, 10, 30), None, _square(50, 50, 20)]
        self.assertTrue(np.array_equal(_mask(geometries),
                                       _mask(geojson_shapes(geometries))))

    def test_fallback_returns_none(self):
        """Anything unrepresentable must fall back, never raise."""
        square = _square(10, 10)
        unsupported = {
            "all missing": [None, None],
            "empty": [],
            "linestring": [square, LineString([(0, 0), (50, 50)])],
            "point": [square, Point(20, 20)],
            "collection": [GeometryCollection([square])],
            "float nan": np.asarray([square, float("nan")], dtype=object),
            "generator": (geom for geom in [square]),
        }
        for label, geometries in unsupported.items():
            with self.subTest(label):
                self.assertIsNone(bulk_geojson(geometries))

    def test_fallback_passes_geometries_through_untouched(self):
        """geojson_shapes must hand rasterio the ORIGINAL objects on fallback."""
        geometries = [_square(10, 10), LineString([(0, 0), (50, 50)])]
        self.assertIs(geojson_shapes(geometries), geometries)
        # rasterio burns LineStrings, so the fallback must not drop them.
        self.assertTrue(np.array_equal(
            _burn(geometries, [1, 2]),
            _burn(geojson_shapes(geometries), [1, 2])))

    def test_accepts_geoseries_and_geometry_array(self):
        series = gpd.GeoSeries([_square(10, 10), MultiPolygon([_square(40, 40)])])
        for container in (series, series.values, series.to_numpy()):
            with self.subTest(type(container).__name__):
                converted = bulk_geojson(container)
                self.assertIsNotNone(converted)
                self.assertTrue(np.array_equal(
                    _burn(list(series), [1, 2]), _burn(converted, [1, 2])))


class RasterizerEndToEndParityTests(unittest.TestCase):
    """Parity through GeoRasterizer, with the bulk path disabled as control."""

    def setUp(self):
        rows = []
        geometries = []
        for index in range(40):
            offset = index * 1.9
            if index % 5 == 0:
                geom = MultiPolygon([_square(offset, offset, 22),
                                     _square(offset + 25, offset + 8, 14)])
            elif index % 7 == 0:
                geom = Polygon(
                    [(offset, 60), (offset + 30, 60), (offset + 30, 95),
                     (offset, 95)],
                    [[(offset + 8, 68), (offset + 22, 68),
                      (offset + 22, 86), (offset + 8, 86)]])
            else:
                geom = _square(offset, 100 - offset - 20, 20)
            geometries.append(geom)
            rows.append({'category': 'a' if index % 2 else 'b'})
        self.gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geometries,
                                    crs="EPSG:25832")
        self.dataset = InMemoryVectorDataset(self.gdf, crs="EPSG:25832")
        self.cost_assumptions = CostAssumptions({'category': {'a': 7, 'b': 13}})

    def _rasterize(self, bulk):
        """Run a full rasterize() with the bulk path on or off."""
        original = rasterizer_module.geojson_shapes
        if not bulk:
            rasterizer_module.geojson_shapes = lambda geometries: geometries
        try:
            rasterizer = GeoRasterizer(self.dataset, self.cost_assumptions)
            # use_class_cache=False: with the cache on, a second call returns
            # lut[class_band] and skips the scan conversion entirely, so the
            # comparison would not touch the code under test at all.
            rasterizer.rasterize(resolution_in_m=1.0, use_class_cache=False)
            return rasterizer.raster
        finally:
            rasterizer_module.geojson_shapes = original

    def _cost_groups(self, bulk, multiply):
        original = rasterizer_module.geojson_shapes
        if not bulk:
            rasterizer_module.geojson_shapes = lambda geometries: geometries
        try:
            rasterizer = GeoRasterizer(self.dataset, self.cost_assumptions)
            rasterizer.rasterize(resolution_in_m=1.0, use_class_cache=False)
            overlay = self.gdf.copy()
            overlay['cost'] = [2 if i % 3 else 5 for i in range(len(overlay))]
            rasterizer._apply_cost_groups(overlay, ignore_value=None,
                                          multiply=multiply)
            return rasterizer.raster
        finally:
            rasterizer_module.geojson_shapes = original

    def test_rasterize_is_bit_identical(self):
        legacy = self._rasterize(bulk=False)
        bulk = self._rasterize(bulk=True)
        self.assertTrue(np.array_equal(legacy, bulk))
        self.assertGreater(len(np.unique(legacy)), 1)

    def test_rasterize_metrics_index_band_is_bit_identical(self):
        original = rasterizer_module.geojson_shapes
        rasterizer_module.geojson_shapes = lambda geometries: geometries
        try:
            legacy = GeoRasterizer(self.dataset, self.cost_assumptions)
            legacy_stack = legacy.rasterize_metrics(resolution_in_m=1.0)
        finally:
            rasterizer_module.geojson_shapes = original
        bulk = GeoRasterizer(self.dataset, self.cost_assumptions)
        bulk_stack = bulk.rasterize_metrics(resolution_in_m=1.0)
        self.assertEqual(legacy_stack.layer_names, bulk_stack.layer_names)
        self.assertTrue(legacy_stack.layer_names)
        for name in legacy_stack.layer_names:
            with self.subTest(name):
                # equal_nan: forbidden cells legitimately carry NaN in some
                # bands, and NaN != NaN would fail a plain array_equal.
                self.assertTrue(np.array_equal(legacy_stack[name],
                                               bulk_stack[name],
                                               equal_nan=True))
        self.assertTrue(np.array_equal(legacy_stack.forbidden_mask,
                                       bulk_stack.forbidden_mask))
        self.assertTrue(np.array_equal(legacy_stack.category,
                                       bulk_stack.category))

    def test_apply_cost_groups_replace_is_bit_identical(self):
        self.assertTrue(np.array_equal(self._cost_groups(False, False),
                                       self._cost_groups(True, False)))

    def test_apply_cost_groups_multiply_is_bit_identical(self):
        self.assertTrue(np.array_equal(self._cost_groups(False, True),
                                       self._cost_groups(True, True)))

    def test_modify_raster_with_geodataframe_is_bit_identical(self):
        results = []
        original = rasterizer_module.geojson_shapes
        for bulk in (False, True):
            if not bulk:
                rasterizer_module.geojson_shapes = lambda geometries: geometries
            try:
                rasterizer = GeoRasterizer(self.dataset, self.cost_assumptions)
                rasterizer.rasterize(resolution_in_m=1.0, use_class_cache=False)
                rasterizer.modify_raster_with_geodataframe(self.gdf, value=3)
                results.append(rasterizer.raster.copy())
            finally:
                rasterizer_module.geojson_shapes = original
        self.assertTrue(np.array_equal(*results))


class BulkGeoJsonScaleTests(unittest.TestCase):
    """One scale check. Deliberately NOT a timing assertion.

    Measured on the development machine through ``_burn_index_band``:
    302,500 parcels into 36 M cells took 10.43 s via shapely's
    ``__geo_interface__`` vs 2.88 s via the bulk path (3.62x); 80,089 holed
    and multipart geometries 3.69 s vs 1.02 s (3.63x); the same 302,500
    parcels enlarged to cover every cell 8.86 s vs 3.46 s (2.56x, scan
    conversion dominating). All bit-identical. Wall time is NOT asserted
    here because it is machine and load dependent; only the raster is.
    """

    def test_bit_identical_at_scale_with_mixed_and_missing_geometries(self):
        rng = np.random.default_rng(20260811)
        size = 20_000
        xs = rng.uniform(0, 990, size)
        ys = rng.uniform(0, 990, size)
        geometries = [_square(x, y, 12) for x, y in zip(xs, ys)]
        # Seed the batch with exactly the mix real ALKIS/WFS data delivers.
        geometries[123] = None
        geometries[456] = Polygon()
        geometries[789] = MultiPolygon([_square(100, 100, 20),
                                        _square(400, 400, 20)])
        values = rng.integers(1, 500, size)
        shape = (1000, 1000)
        transform = from_origin(0, 1000, 1, 1)
        converted = geojson_shapes(np.asarray(geometries, dtype=object))
        self.assertNotIsInstance(converted[0], Polygon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = _burn(geometries, values, transform, shape)
            bulk = _burn(converted, values, transform, shape)
        self.assertTrue(np.array_equal(legacy, bulk))


if __name__ == "__main__":
    unittest.main()

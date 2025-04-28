import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import Affine


class GeoTestCase(unittest.TestCase):
    """Base test class with temporary geo files and utility methods."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory that persists for all tests in the class."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Create empty vector test files
        cls.vector_files = {}
        for ext in [".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml"]:
            file_path = cls.temp_path / f"test{ext}"
            # For SHP files, we need to create a valid shapefile
            if ext == ".shp":
                cls._create_test_shapefile(file_path)
            # For other formats, create valid GeoJSON
            elif ext in [".geojson", ".json"]:
                cls._create_test_geojson(file_path)
            # For remaining formats, just create empty files
            else:
                file_path.touch()
            cls.vector_files[ext] = str(file_path)

        # Create empty raster test files
        cls.raster_files = {}
        for ext in [".tif", ".tiff", ".jp2", ".img", ".bil", ".dem"]:
            file_path = cls.temp_path / f"test{ext}"
            if ext in [".tif", ".tiff"]:
                cls._create_test_raster(file_path)
            else:
                file_path.touch()
            cls.raster_files[ext] = str(file_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory after all tests."""
        cls.temp_dir.cleanup()

    @classmethod
    def _create_test_shapefile(cls, path):
        """Create a minimal valid shapefile."""
        points = [Point(0, 0), Point(1, 1)]
        data = {'id': [1, 2], 'name': ['Point A', 'Point B']}
        gdf = gpd.GeoDataFrame(data, geometry=points, crs="EPSG:4326")
        gdf.to_file(path)

    @classmethod
    def _create_test_geojson(cls, path):
        """Create a minimal valid GeoJSON file."""
        points = [Point(0, 0), Point(1, 1)]
        data = {'id': [1, 2], 'name': ['Point A', 'Point B']}
        gdf = gpd.GeoDataFrame(data, geometry=points, crs="EPSG:4326")
        gdf.to_file(path, driver="GeoJSON")

    @classmethod
    def _create_test_raster(cls, path):
        """Create a minimal valid GeoTIFF."""
        test_array = np.zeros((10, 10), dtype=np.float32)
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)

        with rasterio.open(
                path,
                'w',
                driver='GTiff',
                height=test_array.shape[0],
                width=test_array.shape[1],
                count=1,
                dtype=test_array.dtype,
                crs='+proj=latlong',
                transform=transform
        ) as dst:
            dst.write(test_array, 1)

    def assert_gdf_equal(self, gdf1, gdf2, check_crs=True):
        """Assert that two GeoDataFrames are equal."""
        # Check types
        self.assertIsInstance(gdf1, gpd.GeoDataFrame)
        self.assertIsInstance(gdf2, gpd.GeoDataFrame)

        # Check lengths
        self.assertEqual(len(gdf1), len(gdf2))

        # Check CRS if requested
        if check_crs:
            self.assertEqual(gdf1.crs, gdf2.crs)

        # Check geometries
        for i in range(len(gdf1)):
            self.assertTrue(
                gdf1.geometry.iloc[i].equals(gdf2.geometry.iloc[i]),
                f"Geometry at index {i} is not equal"
            )

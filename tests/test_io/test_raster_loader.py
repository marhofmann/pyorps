import unittest
import tempfile
import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from pyorps.io.raster_loader import (
    read_raster_files,
    combine_rasters,
    create_combined_raster_dataset,
    find_raster_files,
    validate_raster_compatibility,
)
from pyorps.io.geo_dataset import InMemoryRasterDataset


def _create_test_tiff(path, width=10, height=10, crs="EPSG:32632",
                      transform=None, dtype="uint16", nodata=None,
                      value=1):
    """Create a minimal GeoTIFF for testing."""
    if transform is None:
        transform = from_origin(500000, 5600000, 1, 1)
    data = np.full((1, height, width), value, dtype=dtype)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata
    ) as dst:
        dst.write(data)
    return data


class TestReadRasterFiles(unittest.TestCase):
    """Test read_raster_files function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_normal_read(self):
        """Read a single valid GeoTIFF."""
        path = os.path.join(self.temp_dir, "test.tif")
        _create_test_tiff(path)

        datasets, mem_files = read_raster_files(path)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].width, 10)
        for ds in datasets:
            ds.close()

    def test_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_raster_files("/nonexistent/path/file.tif")

    def test_no_crs_no_default_raises(self):
        """File without CRS and no default_crs raises ValueError."""
        path = os.path.join(self.temp_dir, "no_crs.tif")
        transform = from_origin(0, 10, 1, 1)
        data = np.ones((1, 5, 5), dtype=np.uint16)
        with rasterio.open(
            path, "w", driver="GTiff",
            height=5, width=5, count=1,
            dtype="uint16", crs=None, transform=transform
        ) as dst:
            dst.write(data)

        with self.assertRaises(ValueError):
            read_raster_files(path)

    def test_non_tiff_warning(self):
        """Non-TIFF extension triggers a warning."""
        path = os.path.join(self.temp_dir, "test.png")
        _create_test_tiff(path)  # Still valid rasterio file

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            datasets, mem_files = read_raster_files(path)
            self.assertTrue(any("may not be a GeoTIFF" in str(warning.message)
                                for warning in w))
        for ds in datasets:
            ds.close()

    def test_crs_assignment(self):
        """File without CRS gets default_crs assigned."""
        path = os.path.join(self.temp_dir, "no_crs.tif")
        transform = from_origin(0, 10, 1, 1)
        data = np.ones((1, 5, 5), dtype=np.uint16)
        with rasterio.open(
            path, "w", driver="GTiff",
            height=5, width=5, count=1,
            dtype="uint16", crs=None, transform=transform
        ) as dst:
            dst.write(data)

        datasets, mem_files = read_raster_files(path, default_crs="EPSG:32632")
        self.assertEqual(len(datasets), 1)
        self.assertIsNotNone(datasets[0].crs)
        self.assertEqual(str(datasets[0].crs), str(CRS.from_string("EPSG:32632")))
        for ds in datasets:
            ds.close()
        for mf in mem_files:
            mf.close()

    def test_reprojection(self):
        """Raster is reprojected to target_crs."""
        path = os.path.join(self.temp_dir, "utm.tif")
        _create_test_tiff(path, crs="EPSG:32632")

        datasets, mem_files = read_raster_files(path, target_crs="EPSG:4326")
        self.assertEqual(len(datasets), 1)
        self.assertEqual(str(datasets[0].crs), str(CRS.from_string("EPSG:4326")))
        for ds in datasets:
            ds.close()
        for mf in mem_files:
            mf.close()

    def test_multiple_files(self):
        """Read multiple files at once."""
        paths = []
        for i in range(3):
            path = os.path.join(self.temp_dir, f"test_{i}.tif")
            _create_test_tiff(path, value=i + 1)
            paths.append(path)

        datasets, mem_files = read_raster_files(paths)
        self.assertEqual(len(datasets), 3)
        for ds in datasets:
            ds.close()


class TestCombineRasters(unittest.TestCase):
    """Test combine_rasters function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _open_test_rasters(self, n=2):
        """Create and open n test rasters."""
        paths = []
        for i in range(n):
            path = os.path.join(self.temp_dir, f"r_{i}.tif")
            _create_test_tiff(path, value=i + 1)
            paths.append(path)
        return [rasterio.open(p) for p in paths]

    def test_empty_list(self):
        """Empty list raises ValueError."""
        with self.assertRaises(ValueError):
            combine_rasters([])

    def test_crs_mismatch(self):
        """Different CRS raises ValueError."""
        p1 = os.path.join(self.temp_dir, "a.tif")
        p2 = os.path.join(self.temp_dir, "b.tif")
        _create_test_tiff(p1, crs="EPSG:32632")
        _create_test_tiff(p2, crs="EPSG:4326",
                          transform=from_origin(9.0, 50.0, 0.001, 0.001))

        ds1 = rasterio.open(p1)
        ds2 = rasterio.open(p2)
        with self.assertRaises(ValueError):
            combine_rasters([ds1, ds2])

    def test_tuple_api(self):
        """Accepts (datasets, memory_files) tuple."""
        datasets = self._open_test_rasters(2)
        merged, profile = combine_rasters((datasets, []))
        self.assertIsNotNone(merged)
        self.assertIn("crs", profile)

    def test_list_api(self):
        """Accepts plain list of datasets."""
        datasets = self._open_test_rasters(2)
        merged, profile = combine_rasters(datasets)
        self.assertIsNotNone(merged)
        self.assertEqual(profile["count"], 1)

    def test_save_path(self):
        """Saves combined raster when save_path provided."""
        datasets = self._open_test_rasters(2)
        save_path = os.path.join(self.temp_dir, "combined.tif")
        merged, profile = combine_rasters(datasets, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_auto_dtype_nodata(self):
        """dtype and nodata are inferred from input datasets."""
        p1 = os.path.join(self.temp_dir, "nd.tif")
        _create_test_tiff(p1, nodata=0)
        datasets = [rasterio.open(p1)]
        merged, profile = combine_rasters(datasets)
        self.assertEqual(profile["nodata"], 0)


class TestCreateCombinedRasterDataset(unittest.TestCase):
    """Test create_combined_raster_dataset integration function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_returns_in_memory_raster_dataset(self):
        """Returns an InMemoryRasterDataset."""
        path = os.path.join(self.temp_dir, "test.tif")
        _create_test_tiff(path)

        result = create_combined_raster_dataset(path)
        self.assertIsInstance(result, InMemoryRasterDataset)
        self.assertIsNotNone(result.data)
        self.assertIsNotNone(result.crs)
        self.assertIsNotNone(result.transform)


class TestFindRasterFiles(unittest.TestCase):
    """Test find_raster_files function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Create some test files
        for name in ["a.tif", "b.tiff", "c.txt", "d.shp"]:
            open(os.path.join(self.temp_dir, name), "w").close()
        # Create a subdirectory with a raster
        sub_dir = os.path.join(self.temp_dir, "sub")
        os.makedirs(sub_dir)
        open(os.path.join(sub_dir, "e.tif"), "w").close()
        open(os.path.join(sub_dir, "temp_f.tif"), "w").close()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_nonexistent_dir(self):
        """Non-existent directory raises ValueError."""
        with self.assertRaises(ValueError):
            find_raster_files("/nonexistent/dir")

    def test_file_path_raises(self):
        """Passing a file instead of directory raises ValueError."""
        path = os.path.join(self.temp_dir, "a.tif")
        with self.assertRaises(ValueError):
            find_raster_files(path)

    def test_recursive(self):
        """Recursive search finds files in subdirectories."""
        results = find_raster_files(self.temp_dir, recursive=True,
                                     extensions={".tif", ".tiff"})
        names = {r.name for r in results}
        self.assertIn("a.tif", names)
        self.assertIn("b.tiff", names)
        self.assertIn("e.tif", names)
        self.assertNotIn("c.txt", names)

    def test_non_recursive(self):
        """Non-recursive search only finds top-level files."""
        results = find_raster_files(self.temp_dir, recursive=False,
                                     extensions={".tif"})
        names = {r.name for r in results}
        self.assertIn("a.tif", names)
        self.assertNotIn("e.tif", names)

    def test_exclude_patterns(self):
        """Exclude patterns filter out matching files."""
        results = find_raster_files(self.temp_dir, recursive=True,
                                     extensions={".tif"},
                                     exclude_patterns=["*temp*"])
        names = {r.name for r in results}
        self.assertNotIn("temp_f.tif", names)
        self.assertIn("a.tif", names)

    def test_return_absolute(self):
        """return_absolute=True returns absolute paths."""
        results = find_raster_files(self.temp_dir, recursive=False,
                                     extensions={".tif"},
                                     return_absolute=True)
        for r in results:
            self.assertTrue(r.is_absolute())

    def test_return_absolute_false_skips_resolve(self):
        """return_absolute=False does not call .absolute() on paths."""
        results_abs = find_raster_files(self.temp_dir, recursive=False,
                                         extensions={".tif"},
                                         return_absolute=True)
        results_raw = find_raster_files(self.temp_dir, recursive=False,
                                         extensions={".tif"},
                                         return_absolute=False)
        # Both sets should find the same files
        self.assertEqual(len(results_abs), len(results_raw))
        # With return_absolute=True, paths are explicitly .absolute()
        for r in results_abs:
            self.assertEqual(r, r.absolute())
        # With return_absolute=False, paths are returned as-is from glob
        self.assertTrue(len(results_raw) > 0)


class TestValidateRasterCompatibility(unittest.TestCase):
    """Test validate_raster_compatibility function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_empty_list(self):
        """Empty file list returns not compatible."""
        result = validate_raster_compatibility([], verbose=False)
        self.assertFalse(result["compatible"])

    def test_compatible_rasters(self):
        """Identical rasters are compatible."""
        paths = []
        for i in range(2):
            p = os.path.join(self.temp_dir, f"compat_{i}.tif")
            _create_test_tiff(p)
            paths.append(p)

        result = validate_raster_compatibility(paths, verbose=False)
        self.assertTrue(result["compatible"])
        self.assertEqual(result["summary"]["total_files"], 2)

    def test_crs_mismatch(self):
        """Different CRS detected as incompatible."""
        p1 = os.path.join(self.temp_dir, "a.tif")
        p2 = os.path.join(self.temp_dir, "b.tif")
        _create_test_tiff(p1, crs="EPSG:32632")
        _create_test_tiff(p2, crs="EPSG:4326",
                          transform=from_origin(9.0, 50.0, 0.001, 0.001))

        result = validate_raster_compatibility([p1, p2], verbose=False)
        self.assertFalse(result["compatible"])
        self.assertTrue(any("Multiple CRS" in issue for issue in result["issues"]))

    def test_resolution_mismatch(self):
        """Different resolutions detected."""
        p1 = os.path.join(self.temp_dir, "a.tif")
        p2 = os.path.join(self.temp_dir, "b.tif")
        _create_test_tiff(p1, transform=from_origin(500000, 5600000, 1, 1))
        _create_test_tiff(p2, transform=from_origin(500000, 5600000, 5, 5))

        result = validate_raster_compatibility([p1, p2],
                                                check_resolution=True,
                                                verbose=False)
        self.assertFalse(result["compatible"])
        self.assertTrue(any("Multiple resolutions" in issue
                            for issue in result["issues"]))

    def test_band_mismatch(self):
        """Different band counts detected when check_bands=True."""
        p1 = os.path.join(self.temp_dir, "a.tif")
        p2 = os.path.join(self.temp_dir, "b.tif")
        _create_test_tiff(p1)

        # Create 2-band raster
        transform = from_origin(500000, 5600000, 1, 1)
        data = np.ones((2, 10, 10), dtype=np.uint16)
        with rasterio.open(
            p2, "w", driver="GTiff",
            height=10, width=10, count=2,
            dtype="uint16", crs="EPSG:32632", transform=transform
        ) as dst:
            dst.write(data)

        result = validate_raster_compatibility([p1, p2],
                                                check_bands=True,
                                                verbose=False)
        self.assertFalse(result["compatible"])
        self.assertTrue(any("Multiple band counts" in issue
                            for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()

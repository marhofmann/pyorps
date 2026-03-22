"""
Tests for security hardening fixes (P5.1 through P5.5).

P5.1: WFS URL validation (SSRF prevention)
P5.2: Raster size validation before Cython entry
P5.3: uint16 overflow in raster multiplication
P5.4: GPU raster sidecar validation
P5.5: WFS filter_params sanitization
"""
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.transform import from_origin
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# P5.1 & P5.5: WFS URL validation + filter_params sanitization
# ---------------------------------------------------------------------------
from pyorps.io.vector_loader import (
    _validate_wfs_url,
    _sanitize_filter_params,
    load_from_wfs,
)

# ---------------------------------------------------------------------------
# P5.2: Raster size validation
# ---------------------------------------------------------------------------
from pyorps.graph.path_finder import PathFinder, MAX_SAFE_CELLS

# ---------------------------------------------------------------------------
# P5.3: uint16 overflow in raster multiplication
# ---------------------------------------------------------------------------
from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.io.geo_dataset import InMemoryRasterDataset

# ---------------------------------------------------------------------------
# P5.4: GPU raster sidecar validation
# ---------------------------------------------------------------------------
try:
    from pyorps.io.gpu_raster import load_gpu_raster
    HAS_GPU_RASTER = True
except ImportError:
    HAS_GPU_RASTER = False


class TestWfsUrlValidation(unittest.TestCase):
    """P5.1: Validate WFS URLs to prevent SSRF."""

    # --- Valid URLs ---

    def test_http_url_accepted(self):
        """Standard http URL should pass validation."""
        _validate_wfs_url("http://example.com/wfs")

    def test_https_url_accepted(self):
        """Standard https URL should pass validation."""
        _validate_wfs_url("https://example.com/wfs")

    def test_public_ip_accepted(self):
        """A public IP address should pass validation."""
        _validate_wfs_url("https://93.184.216.34/wfs")

    def test_hostname_with_port_accepted(self):
        """URL with a port number should pass validation."""
        _validate_wfs_url("https://example.com:8080/wfs")

    # --- Invalid schemes ---

    def test_file_scheme_rejected(self):
        """file:// scheme should be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_wfs_url("file:///etc/passwd")
        self.assertIn("scheme", str(ctx.exception).lower())

    def test_ftp_scheme_rejected(self):
        """ftp:// scheme should be rejected."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("ftp://example.com/wfs")

    def test_empty_scheme_rejected(self):
        """URL without scheme should be rejected."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("example.com/wfs")

    def test_javascript_scheme_rejected(self):
        """javascript: scheme should be rejected."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("javascript:alert(1)")

    def test_data_scheme_rejected(self):
        """data: scheme should be rejected."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("data:text/html,<h1>Hello</h1>")

    # --- Private IP blocking ---

    def test_localhost_blocked(self):
        """127.0.0.1 should be blocked."""
        with self.assertRaises(ValueError) as ctx:
            _validate_wfs_url("http://127.0.0.1/wfs")
        self.assertIn("private", str(ctx.exception).lower())

    def test_10_x_blocked(self):
        """10.0.0.1 (private) should be blocked."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("http://10.0.0.1/wfs")

    def test_172_16_blocked(self):
        """172.16.0.1 (private) should be blocked."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("http://172.16.0.1/wfs")

    def test_192_168_blocked(self):
        """192.168.1.1 (private) should be blocked."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("http://192.168.1.1/wfs")

    def test_ipv6_loopback_blocked(self):
        """IPv6 loopback ::1 should be blocked."""
        with self.assertRaises(ValueError):
            _validate_wfs_url("http://[::1]/wfs")

    def test_private_ip_allowed_when_block_private_false(self):
        """Private IPs should be allowed when block_private=False."""
        # Should NOT raise
        _validate_wfs_url("http://192.168.1.1/wfs", block_private=False)

    def test_no_hostname_rejected(self):
        """URL with no hostname should be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_wfs_url("http:///path")
        self.assertIn("hostname", str(ctx.exception).lower())


class TestFilterParamsSanitization(unittest.TestCase):
    """P5.5: WFS filter_params sanitization."""

    def test_none_passes_through(self):
        """None filter_params should return None."""
        self.assertIsNone(_sanitize_filter_params(None))

    def test_empty_dict_passes_through(self):
        """Empty dict should return empty dict (falsy)."""
        result = _sanitize_filter_params({})
        self.assertFalse(result)

    def test_allowed_params_pass(self):
        """Recognised filter params should pass through unchanged."""
        params = {"CQL_FILTER": "foo='bar'", "MAXFEATURES": "100"}
        result = _sanitize_filter_params(params)
        self.assertEqual(result, params)

    def test_reserved_key_service_rejected(self):
        """SERVICE key should be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _sanitize_filter_params({"SERVICE": "WMS"})
        self.assertIn("reserved", str(ctx.exception).lower())

    def test_reserved_key_request_rejected(self):
        """REQUEST key should be rejected."""
        with self.assertRaises(ValueError):
            _sanitize_filter_params({"REQUEST": "GetMap"})

    def test_reserved_key_version_rejected(self):
        """VERSION key should be rejected."""
        with self.assertRaises(ValueError):
            _sanitize_filter_params({"VERSION": "1.0.0"})

    def test_reserved_key_case_insensitive(self):
        """Reserved key detection should be case-insensitive."""
        with self.assertRaises(ValueError):
            _sanitize_filter_params({"service": "WFS"})
        with self.assertRaises(ValueError):
            _sanitize_filter_params({"Service": "WFS"})

    def test_unrecognised_param_warns_but_passes(self):
        """Unrecognised parameter should log a warning but pass through."""
        with self.assertLogs("pyorps.io.vector_loader", level="WARNING") as cm:
            result = _sanitize_filter_params({"CUSTOM_PARAM": "value"})
        self.assertIn("CUSTOM_PARAM", cm.output[0])
        self.assertEqual(result["CUSTOM_PARAM"], "value")

    def test_mixed_allowed_and_unrecognised(self):
        """Mix of allowed and unrecognised params."""
        with self.assertLogs("pyorps.io.vector_loader", level="WARNING"):
            result = _sanitize_filter_params({
                "CQL_FILTER": "x=1",
                "MY_CUSTOM": "val"
            })
        self.assertEqual(result["CQL_FILTER"], "x=1")
        self.assertEqual(result["MY_CUSTOM"], "val")


class TestWfsEntryPointValidation(unittest.TestCase):
    """P5.1 + P5.5: Validation wired into load_from_wfs."""

    def test_load_from_wfs_rejects_file_scheme(self):
        """load_from_wfs should reject file:// URLs before any network call."""
        with self.assertRaises(ValueError):
            load_from_wfs("file:///etc/passwd", "some_layer")

    def test_load_from_wfs_rejects_reserved_filter_param(self):
        """load_from_wfs should reject reserved filter params."""
        with self.assertRaises(ValueError):
            load_from_wfs(
                "https://example.com/wfs",
                "some_layer",
                filter_params={"SERVICE": "WMS"}
            )


class TestRasterSizeValidation(unittest.TestCase):
    """P5.2: Validate raster size before Cython entry to prevent uint32 overflow."""

    def test_max_safe_cells_constant(self):
        """MAX_SAFE_CELLS should equal 2**32 - 1."""
        self.assertEqual(MAX_SAFE_CELLS, 2**32 - 1)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_oversized_raster_raises(self, mock_get_api, mock_rh_cls,
                                     mock_init_geo):
        """A raster exceeding MAX_SAFE_CELLS should raise ValueError."""
        # Create a mock raster dataset
        mock_dataset = MagicMock()
        mock_dataset.crs = "EPSG:32632"
        mock_init_geo.return_value = mock_dataset

        # Create a mock raster handler whose data has an oversized shape
        mock_rh = MagicMock()
        # Shape: band=1, rows=70000, cols=70000 => 4.9 billion cells > uint32
        oversized = np.empty((1,), dtype=object)
        oversized_band = MagicMock()
        oversized_band.shape = (70000, 70000)
        mock_rh.data.__getitem__ = MagicMock(return_value=oversized_band)
        mock_rh.search_space_buffer_m = 1000
        mock_rh_cls.return_value = mock_rh

        # Build PathFinder
        pf = PathFinder.__new__(PathFinder)
        pf.source_coords = (500000, 5600000)
        pf.target_coords = (500100, 5600100)
        pf.graph_api_name = "cython"
        pf.raster_handler = mock_rh
        pf.dem_raster_handler = None
        pf.dem_kwargs = None
        pf.ignore_max_cost = False
        pf.use_gpu = False
        pf.runtimes = {}
        pf.steps = [(0, 1, 1.0)]
        pf._graph_api = None

        with self.assertRaises(ValueError) as ctx:
            pf.create_graph()

        self.assertIn("exceeds", str(ctx.exception))

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_normal_raster_passes(self, mock_get_api, mock_rh_cls,
                                  mock_init_geo):
        """A reasonably sized raster should not raise."""
        mock_dataset = MagicMock()
        mock_dataset.crs = "EPSG:32632"
        mock_init_geo.return_value = mock_dataset

        mock_rh = MagicMock()
        normal_band = np.ones((100, 100), dtype=np.uint16)
        mock_rh.data.__getitem__ = MagicMock(return_value=normal_band)
        mock_rh.search_space_buffer_m = 1000
        mock_rh_cls.return_value = mock_rh

        mock_api_instance = MagicMock()
        mock_api_instance.edge_construction_time = 0.0
        mock_api_instance.graph_creation_time = 0.0
        mock_api_instance.graph = MagicMock()
        mock_get_api.return_value = MagicMock(return_value=mock_api_instance)

        pf = PathFinder.__new__(PathFinder)
        pf.source_coords = (500000, 5600000)
        pf.target_coords = (500100, 5600100)
        pf.graph_api_name = "cython"
        pf.raster_handler = mock_rh
        pf.dem_raster_handler = None
        pf.dem_kwargs = None
        pf.ignore_max_cost = False
        pf.use_gpu = False
        pf.runtimes = {}
        pf.steps = [(0, 1, 1.0)]
        pf._graph_api = None

        # Should not raise
        pf.create_graph()


class TestUint16OverflowProtection(unittest.TestCase):
    """P5.3: uint16 multiplication should clip instead of wrapping."""

    def setUp(self):
        """Create a rasterizer with known uint16 raster values."""
        self.transform = from_origin(500000, 5600000, 10, 10)
        self.crs = "EPSG:32632"

    def _make_rasterizer(self, raster_value):
        """Create a GeoRasterizer with uniform raster of given value."""
        raster_data = np.full((10, 10), raster_value, dtype=np.uint16)
        raster_dataset = InMemoryRasterDataset(
            raster_data, self.crs, self.transform
        )
        cost_assumptions = {'category': {'road': 10}}
        return GeoRasterizer(raster_dataset, cost_assumptions)

    def test_multiply_clips_instead_of_wrapping(self):
        """Multiplying 60000 * 2 should clip to 65535, not wrap to 54464."""
        rasterizer = self._make_rasterizer(60000)

        # Create a GeoDataFrame covering the entire raster
        geom = Polygon([
            (500000, 5599900), (500000, 5600000),
            (500100, 5600000), (500100, 5599900)
        ])
        gdf = gpd.GeoDataFrame(
            {'val': [1]}, geometry=[geom], crs=self.crs
        )

        result = rasterizer.modify_raster_with_geodataframe(
            gdf, value=2, ignore_value=None, multiply=True
        )

        # 60000 * 2 = 120000 > 65535, should be clipped
        max_val = np.iinfo(np.uint16).max  # 65535
        # All cells that were inside the polygon should be clipped to 65535
        # (The geometry_mask may not cover every single cell perfectly due to
        # rasterization, but at minimum the cells that were multiplied
        # should be clipped.)
        self.assertTrue(np.all(result <= max_val))
        # Verify that at least some cells got the clipped value
        # (not the wrapped value 54464 = 120000 - 65536)
        wrapped_value = (60000 * 2) % 65536  # 54464
        # If ANY cell has the wrapped value, the fix failed
        self.assertFalse(
            np.any(result == wrapped_value),
            f"Found wrapped value {wrapped_value}; clip did not work"
        )

    def test_multiply_within_range_unchanged(self):
        """Multiplying 100 * 3 = 300 should remain 300."""
        rasterizer = self._make_rasterizer(100)

        geom = Polygon([
            (500000, 5599900), (500000, 5600000),
            (500100, 5600000), (500100, 5599900)
        ])
        gdf = gpd.GeoDataFrame(
            {'val': [1]}, geometry=[geom], crs=self.crs
        )

        result = rasterizer.modify_raster_with_geodataframe(
            gdf, value=3, ignore_value=None, multiply=True
        )

        # Some cells should have the multiplied value (300)
        self.assertTrue(np.any(result == 300))

    def test_multiply_by_one_unchanged(self):
        """Multiplying by 1 should leave values unchanged."""
        rasterizer = self._make_rasterizer(500)

        geom = Polygon([
            (500000, 5599900), (500000, 5600000),
            (500100, 5600000), (500100, 5599900)
        ])
        gdf = gpd.GeoDataFrame(
            {'val': [1]}, geometry=[geom], crs=self.crs
        )

        result = rasterizer.modify_raster_with_geodataframe(
            gdf, value=1, ignore_value=None, multiply=True
        )

        self.assertTrue(np.all(result == 500))


@unittest.skipUnless(HAS_GPU_RASTER, "gpu_raster not available")
class TestGpuRasterSidecarValidation(unittest.TestCase):
    """P5.4: Validate GPU raster sidecar metadata against binary file."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_gpur_files(self, shape, dtype, binary_data=None):
        """Create matching .gpur and .gpur.json files."""
        gpur_path = self.temp_path / "test.gpur"
        sidecar_path = self.temp_path / "test.gpur.json"

        if binary_data is None:
            binary_data = np.zeros(shape, dtype=dtype)
        binary_data.tofile(str(gpur_path))

        metadata = {
            "shape": list(shape),
            "dtype": str(np.dtype(dtype)),
            "crs": "EPSG:32632",
            "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5600000.0],
        }
        with open(sidecar_path, "w") as f:
            json.dump(metadata, f)

        return str(gpur_path)

    def test_valid_file_loads_successfully(self):
        """A valid .gpur file with matching metadata should load fine."""
        path = self._create_gpur_files((100, 100), "uint16")
        raster, metadata = load_gpu_raster(path, device=False)
        self.assertEqual(raster.shape, (100, 100))
        self.assertEqual(raster.dtype, np.uint16)

    def test_size_mismatch_raises(self):
        """Binary file size mismatch should raise ValueError."""
        gpur_path = self.temp_path / "test.gpur"
        sidecar_path = self.temp_path / "test.gpur.json"

        # Write 100 bytes of data
        np.zeros(50, dtype=np.uint16).tofile(str(gpur_path))

        # But claim shape is (100, 100) = 20000 bytes
        metadata = {
            "shape": [100, 100],
            "dtype": "uint16",
            "crs": "EPSG:32632",
            "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5600000.0],
        }
        with open(sidecar_path, "w") as f:
            json.dump(metadata, f)

        with self.assertRaises(ValueError) as ctx:
            load_gpu_raster(str(gpur_path), device=False)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_disallowed_dtype_raises(self):
        """A dtype not in the allowlist should raise ValueError."""
        gpur_path = self.temp_path / "test.gpur"
        sidecar_path = self.temp_path / "test.gpur.json"

        # Write valid binary data
        np.zeros((10, 10), dtype=np.int32).tofile(str(gpur_path))

        # But use disallowed dtype "int32"
        metadata = {
            "shape": [10, 10],
            "dtype": "int32",
            "crs": "EPSG:32632",
            "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5600000.0],
        }
        with open(sidecar_path, "w") as f:
            json.dump(metadata, f)

        with self.assertRaises(ValueError) as ctx:
            load_gpu_raster(str(gpur_path), device=False)
        self.assertIn("unsupported dtype", str(ctx.exception).lower())

    def test_allowed_dtypes(self):
        """uint16, float32, and float64 should all be accepted."""
        for dtype_str in ("uint16", "float32", "float64"):
            path = self._create_gpur_files((10, 10), dtype_str)
            raster, metadata = load_gpu_raster(path, device=False)
            self.assertEqual(raster.dtype, np.dtype(dtype_str))

    def test_missing_sidecar_raises(self):
        """Missing sidecar file should raise FileNotFoundError."""
        gpur_path = self.temp_path / "orphan.gpur"
        np.zeros((10, 10), dtype=np.uint16).tofile(str(gpur_path))

        with self.assertRaises(FileNotFoundError):
            load_gpu_raster(str(gpur_path), device=False)


if __name__ == "__main__":
    unittest.main()

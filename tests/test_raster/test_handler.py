import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile
from shapely.geometry import Polygon
from rasterio.windows import Window
from rasterio.transform import from_origin

from pyorps.io.geo_dataset import InMemoryRasterDataset
from pyorps.raster.handler import RasterHandler, create_test_tiff


class TestRasterHandler(unittest.TestCase):
    """Test cases for the RasterHandler class."""

    def setUp(self):
        """Set up test data."""
        # No super().setUp() call needed when directly using unittest.TestCase

        # Create test raster data
        self.raster_data = np.ones((1, 10, 10), dtype=np.uint16)
        self.transform = from_origin(500000, 5600000, 10, 10)
        self.crs = "EPSG:32632"

        # Create test raster dataset
        self.raster_dataset = InMemoryRasterDataset(
            self.raster_data,
            self.crs,
            self.transform
        )

        # Set up test coordinates
        self.source_coords = (500050, 5599950)  # Center of pixel (0, 5)
        self.target_coords = (500150, 5599850)  # Center of pixel (5, 15)

        # Set up test multi-coordinates
        self.source_coords_list = [(500050, 5599950), (500070, 5599930)]
        self.target_coords_list = [(500150, 5599850), (500170, 5599830)]

    def test_initialization_single_coords(self):
        """Test RasterHandler initialization with single coordinates."""
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Check that internal attributes were set correctly
        self.assertIsNotNone(handler.buffer_geometry)
        self.assertIsNotNone(handler.window)
        self.assertIsNotNone(handler.window_transform)
        self.assertIsNotNone(handler.data)

        # Check that data is correctly shaped
        self.assertEqual(len(handler.data.shape), 3)  # (bands, height, width)

    def test_initialization_multi_coords(self):
        """Test RasterHandler initialization with multiple coordinates."""
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords_list,
            self.target_coords_list,
            search_space_buffer_m=100
        )

        # Check that internal attributes were set correctly
        self.assertIsNotNone(handler.buffer_geometry)
        self.assertIsNotNone(handler.window)
        self.assertIsNotNone(handler.window_transform)
        self.assertIsNotNone(handler.data)

    def test_transform_coords(self):
        """Test the _transform_coords method."""
        # Create handler
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Test with same CRS (no transformation needed)
        input_coord = (500000, 5600000)
        result = handler._transform_coords(input_coord, self.crs, self.crs)
        self.assertEqual(result, input_coord)

        # Test with list of coordinates and same CRS
        input_coords = [(500000, 5600000), (500010, 5600010)]
        result = handler._transform_coords(input_coords, self.crs, self.crs)
        self.assertEqual(result, input_coords)

    @patch('pyproj.Transformer.from_crs')
    def test_transform_coords_with_mocked_transformer(self, mock_from_crs):
        """Test _transform_coords with a mocked transformer."""
        # Set up mock
        mock_transformer = MagicMock()
        mock_transformer.transform.side_effect = lambda x, y: (x + 10, y + 10)
        mock_from_crs.return_value = mock_transformer

        # Create handler
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Test with single coordinate
        input_coord = (5000, 6000)
        input_crs = "EPSG:25832"
        result = handler._transform_coords(input_coord, input_crs, self.crs)

        self.assertEqual(result, (5010, 6010))
        mock_transformer.transform.assert_called_once_with(5000, 6000)

        # Test with list of coordinates
        mock_transformer.transform.reset_mock()
        mock_transformer.transform.side_effect = lambda x, y: (x + 10, y + 10)

        input_coords = [(5000, 6000), (7000, 8000)]
        result = handler._transform_coords(input_coords, input_crs, self.crs)

        self.assertEqual(result, [(5010, 6010), (7010, 8010)])
        self.assertEqual(mock_transformer.transform.call_count, 2)

    def test_coords_to_indices(self):
        """Test the coords_to_indices method."""
        # Create handler with a simple window offset
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Force specific window offset for testing
        handler.window = Window(col_off=2, row_off=3, width=5, height=5)

        # Mock the transform function to return predictable values
        with patch('pyorps.raster.handler.rowcol') as mock_rowcol:
            mock_rowcol.return_value = ([5, 6], [7, 8])  # Rows, Cols for two points

            # Test with list of coordinate tuples
            coords = [(500050, 5599950), (500070, 5599930)]
            indices = handler.coords_to_indices(coords)

            # Update expected values to match actual behavior
            expected = np.array([[2, 5], [3, 6]])
            np.testing.assert_array_equal(indices, expected)

    def test_indices_to_coords(self):
        """Test the indices_to_coords method with mocked transform_xy.

        The method should pass window-local indices to transform_xy using
        the window_transform, and return the coordinates unchanged (no
        additional pixel correction).
        """
        # Create handler with a simple window offset
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Force specific window offset for testing
        handler.window = Window(col_off=2, row_off=3, width=5, height=5)

        # Mock the transform function to return predictable values
        with patch('pyorps.raster.handler.transform_xy') as mock_xy:
            mock_xy.return_value = ([500050, 500070], [5599950, 5599930])

            # Test with list of index tuples
            indices = [(0, 0), (1, 1)]
            coords = handler.indices_to_coords(indices)

            # The method should return transform_xy output directly
            # (no pixel-width/height correction applied)
            expected = np.array([[500050, 5599950], [500070, 5599930]])
            np.testing.assert_array_almost_equal(coords, expected)

    def test_apply_geometry_mask(self):
        """Test the apply_geometry_mask method."""
        # Create a handler
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Create a simple mask
        polygon = Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)])

        # Mock rasterize to return a specific mask
        with patch('pyorps.raster.handler.rasterize') as mock_rasterize:
            # Create a mask with 1s in the left half, 0s in the right
            mask = np.zeros((handler.window.height, handler.window.width),
                            dtype=np.uint8)
            mask[:, :mask.shape[1] // 2] = 1
            mock_rasterize.return_value = mask

            # Apply the mask
            result = handler.apply_geometry_mask(polygon)

            # Get max value for the data type
            max_val = np.iinfo(handler.data.dtype).max

            for b in range(handler.data.shape[0]):
                # The original values inside the mask should be preserved (which are 1s)
                np.testing.assert_array_equal(
                    result[b, :, :mask.shape[1] // 2],  # Left half (inside mask)
                    np.ones_like(result[b, :, :mask.shape[1] // 2])  # Expected: 1s
                )

                # Outside the mask should be set to max value
                np.testing.assert_array_equal(
                    result[b, :, mask.shape[1] // 2:],  # Right half (outside mask)
                    np.full_like(result[b, :, mask.shape[1] // 2:], max_val)
                    # Expected: max_val
                )

    def test_estimate_optimal_buffer_width(self):
        """Test the estimate_optimal_buffer_width method."""
        # Create a raster with some "obstacles"
        raster_data = np.ones((1, 20, 20), dtype=np.uint16)

        # Add some high cost areas (obstacles)
        raster_data[0, 5:15, 10:12] = 65535  # Vertical obstacle

        raster_dataset = InMemoryRasterDataset(
            raster_data,
            self.crs,
            self.transform
        )

        # Create handler
        handler = RasterHandler(
            raster_dataset,
            (500050, 5599950),  # Left of obstacle
            (500250, 5599950),  # Right of obstacle
            search_space_buffer_m=100
        )

        # Test buffer estimation - should be higher due to obstacle
        buffer_width = handler.estimate_buffer_width(
            (500050, 5599950),
            (500250, 5599950)
        )

        # Buffer should be at least the min_buffer
        self.assertGreaterEqual(buffer_width, 200)

    def test_save_section_as_raster(self):
        """Test the save_section_as_raster method."""
        # Create a handler
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100
        )

        # Use a temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock rasterio.open to avoid actual file operations
            with patch('pyorps.raster.handler.rio_open') as mock_open:
                # Set up mock for context manager
                mock_dataset = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dataset

                # Save the raster
                handler.save_section_as_raster(tmp_path)

                # Check that open was called with correct parameters
                mock_open.assert_called_once()
                args, kwargs = mock_open.call_args

                # Verify that the correct path was used
                self.assertEqual(args[0], tmp_path)

                # Check write mode is the second positional argument
                self.assertEqual(args[1], 'w')

                # Check that correct metadata was passed
                self.assertEqual(kwargs['driver'], 'GTiff')
                self.assertEqual(kwargs['height'], handler.window.height)
                self.assertEqual(kwargs['width'], handler.window.width)
                self.assertEqual(kwargs['count'], handler.raster_dataset.count)
                self.assertEqual(kwargs['dtype'], handler.data.dtype)
                self.assertEqual(kwargs['crs'], handler.raster_dataset.crs)
                self.assertEqual(kwargs['transform'], handler.window_transform)

                # Check that write was called with the data
                mock_dataset.write.assert_called_once_with(handler.data)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_create_test_tiff(self):
        """Test the create_test_tiff helper function."""
        # Use a temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock rasterio.open to avoid actual file operations
            with patch('pyorps.raster.handler.rio_open') as mock_open:
                # Set up mock for context manager
                mock_dataset = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dataset

                # Create random pattern test TIFF
                create_test_tiff(tmp_path, pattern="random")

                # Check that open was called with correct parameters
                mock_open.assert_called_once()
                args, kwargs = mock_open.call_args

                # Verify that the correct path was used
                self.assertEqual(args[0], tmp_path)

                # Check that it used expected defaults
                self.assertEqual(kwargs['width'], 100)
                self.assertEqual(kwargs['height'], 100)
                self.assertEqual(kwargs['count'], 1)
                self.assertEqual(kwargs['crs'], "EPSG:32632")

                # Check that write was called
                mock_dataset.write.assert_called_once()

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_create_test_tiff_patterns(self):
        """Test create_test_tiff with different patterns."""
        patterns = ["random", "gradient", "checkerboard"]

        for pattern in patterns:
            # Use a temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Mock rasterio.open to avoid actual file operations
                with patch('pyorps.raster.handler.rio_open') as mock_open:
                    # Set up mock for context manager
                    mock_dataset = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_dataset

                    # Create test TIFF with the pattern
                    data = create_test_tiff(tmp_path, width=50, height=40, bands=2, pattern=pattern)

                    # Check data shape
                    self.assertEqual(data.shape, (2, 40, 50))

                    # Check that open was called with correct parameters
                    mock_open.assert_called_once()
                    args, kwargs = mock_open.call_args

                    # Verify correct dimensions
                    self.assertEqual(kwargs['width'], 50)
                    self.assertEqual(kwargs['height'], 40)
                    self.assertEqual(kwargs['count'], 2)

            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def test_initialization_with_input_crs(self):
        """Test initialization with different input CRS."""
        # Mock the _transform_coords method
        with patch.object(RasterHandler, '_transform_coords') as mock_transform:
            # Set up the mock to convert coordinates
            mock_transform.side_effect = lambda coords, src_crs, dst_crs: (
                (500050, 5599950) if isinstance(coords, tuple) else
                [(500050, 5599950), (500070, 5599930)]
            )

            # Initialize handler with different CRS
            RasterHandler(
                self.raster_dataset,
                (4.33, 52.12),  # WGS84 coordinates
                (4.34, 52.13),  # WGS84 coordinates
                search_space_buffer_m=100,
                input_crs="EPSG:4326"  # WGS84
            )

            # Check that transform was called with correct parameters
            mock_transform.assert_any_call((4.33, 52.12), "EPSG:4326", self.crs)
            mock_transform.assert_any_call((4.34, 52.13), "EPSG:4326", self.crs)

    def test_initialization_with_bands_filter(self):
        """Test initialization with bands filter."""
        # Create a multi-band test raster
        raster_data = np.ones((3, 10, 10), dtype=np.uint16)
        raster_data[0] *= 10  # Band 1 has value 10
        raster_data[1] *= 20  # Band 2 has value 20
        raster_data[2] *= 30  # Band 3 has value 30

        raster_dataset = InMemoryRasterDataset(
            raster_data,
            self.crs,
            self.transform
        )

        # Verify basics
        self.assertEqual(raster_dataset.count, 3)

        # Create handler with mask applied to specific bands
        handler = RasterHandler(
            raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100,
            apply_mask=True,
            bands=[1, 3]  # Apply mask to bands 1 and 3 (1-indexed)
        )

        # Mock apply_geometry_mask to check which bands it's called with
        with patch.object(handler, 'apply_geometry_mask') as mock_apply:
            # Force a new mask application
            handler._init_from_metadata(
                self.source_coords,
                self.target_coords,
                100,
                None,
                True,
                None,
                [1, 3]  # Apply to bands 1 and 3
            )

            # Check that apply_geometry_mask was called with correct parameters
            mock_apply.assert_called_once()
            args, kwargs = mock_apply.call_args

            # Should be called with our specified bands as third positional argument
            self.assertEqual(args[2], [1, 3])

    def test_apply_geometry_mask_with_specified_bands(self):
        """Test apply_geometry_mask with specified bands parameter."""
        # Create a multi-band test raster
        raster_data = np.ones((3, 10, 10), dtype=np.uint16)
        raster_data[0] *= 10  # Band 1 has value 10
        raster_data[1] *= 20  # Band 2 has value 20
        raster_data[2] *= 30  # Band 3 has value 30

        raster_dataset = InMemoryRasterDataset(
            raster_data,
            self.crs,
            self.transform
        )

        # Create handler
        handler = RasterHandler(
            raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100,
            apply_mask=False  # Don't apply mask automatically
        )

        # Create a simple polygon mask
        polygon = Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)])

        # Mock rasterize to return a specific mask
        with patch('pyorps.raster.handler.rasterize') as mock_rasterize:
            # Create a mask with 1s in the left half, 0s in the right
            mask = np.ones((handler.window.height, handler.window.width),
                           dtype=np.uint8)
            mask[:, mask.shape[1] // 2:] = 0  # Right half is masked out
            mock_rasterize.return_value = mask

            # Apply the mask only to bands 0 and 2 (already 0-indexed for test)
            bands = [1, 3]  # will be converted to 0-indexed inside method
            result = handler.apply_geometry_mask(polygon, outside_value=999,
                                                 bands=bands)

            # Check band 0 (values inside mask stay at 10, outside set to 999)
            expected_band0 = np.full_like(result[0], 10)  # Start with all 10s
            expected_band0[:, mask.shape[1] // 2:] = 999  # Right half should be 999
            np.testing.assert_array_equal(result[0], expected_band0)

    def test_initialization_with_outside_value(self):
        """Test initialization with custom outside value for masks."""
        # Create handler with custom outside value
        handler = RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=100,
            outside_value=42  # Custom outside value
        )

        # Mock apply_geometry_mask to check the outside value
        with patch.object(handler, 'apply_geometry_mask') as mock_apply:
            # Force a new mask application
            handler._init_from_metadata(
                self.source_coords,
                self.target_coords,
                100,
                None,
                True,
                42,  # Custom outside value
                None
            )

            # Check that apply_geometry_mask was called with the custom outside value
            # as the second positional argument
            mock_apply.assert_called_once()
            args, kwargs = mock_apply.call_args

            self.assertEqual(args[1], 42)

    def test_initialization_error_handling(self):
        """Test proper error handling during initialization."""
        # Create invalid raster dataset (not a RasterDataset instance)
        invalid_dataset = "not_a_raster_dataset"

        # Should raise AttributeError
        with self.assertRaises(AttributeError):
            RasterHandler(
                invalid_dataset,
                self.source_coords,
                self.target_coords
            )


class TestMaxDistancePair(unittest.TestCase):
    """Test RasterHandler.max_distance_pair static method."""

    def test_single_pair(self):
        """Single source, single target."""
        pair, dist = RasterHandler.max_distance_pair((0, 0), (3, 4))
        self.assertEqual(pair, ((0, 0), (3, 4)))
        self.assertAlmostEqual(dist, 5.0)

    def test_multiple_pairs(self):
        """List of sources and targets, picks max distance pair."""
        sources = [(0, 0), (1, 1)]
        targets = [(10, 0), (0, 10)]
        pair, dist = RasterHandler.max_distance_pair(sources, targets)
        # Max distance should be from (0,0) to (10,0) = 10 or (0,0) to (0,10) = 10
        # or (1,1) to (0,10) ≈ 9.06 or (1,1) to (10,0) ≈ 9.06
        self.assertAlmostEqual(dist, 10.0)

    def test_empty_input(self):
        """Empty lists return None."""
        result = RasterHandler.max_distance_pair([], [(1, 2)])
        self.assertIsNone(result)


class TestCoordinateRoundTrip(unittest.TestCase):
    """Test coordinate conversion correctness WITHOUT mocks.

    These tests verify that indices_to_coords and coords_to_indices produce
    geometrically correct results by using known transforms and computing
    expected pixel center coordinates analytically.
    """

    def setUp(self):
        """Set up a raster with a known transform."""
        # 100x100 raster, 10m pixel size, origin at (500000, 5600000)
        self.raster_data = np.ones((1, 100, 100), dtype=np.uint16)
        self.transform = from_origin(500000, 5600000, 10, 10)
        self.crs = "EPSG:32632"
        self.raster_dataset = InMemoryRasterDataset(
            self.raster_data, self.crs, self.transform
        )
        # Source and target well inside the raster
        self.source_coords = (500200, 5599800)
        self.target_coords = (500700, 5599300)

    def _make_handler(self):
        return RasterHandler(
            self.raster_dataset,
            self.source_coords,
            self.target_coords,
            search_space_buffer_m=200
        )

    def test_indices_to_coords_pixel_center_accuracy(self):
        """indices_to_coords must return pixel centers within 0.01m tolerance.

        For a 10m pixel raster with origin (500000, 5600000):
        - Pixel (row=0, col=0) of the FULL raster has center (500005, 5599995)
        - Pixel (row=r, col=c) has center (500000 + (c+0.5)*10, 5600000 - (r+0.5)*10)

        When using a window, the handler works with local indices that map to
        a sub-region of the full raster. The local pixel (0,0) maps to the
        full-raster pixel at (window.row_off, window.col_off).
        """
        handler = self._make_handler()

        # Pick a few local pixel indices
        local_indices = [(0, 0), (5, 3), (10, 10)]

        coords = handler.indices_to_coords(local_indices)

        for i, (local_row, local_col) in enumerate(local_indices):
            # Compute expected pixel center in the full raster CRS
            full_row = local_row + handler.window.row_off
            full_col = local_col + handler.window.col_off
            expected_x = 500000 + (full_col + 0.5) * 10
            expected_y = 5600000 - (full_row + 0.5) * 10

            actual_x, actual_y = coords[i]
            self.assertAlmostEqual(
                actual_x, expected_x, places=2,
                msg=f"X mismatch for local pixel ({local_row},{local_col}): "
                    f"expected {expected_x}, got {actual_x}"
            )
            self.assertAlmostEqual(
                actual_y, expected_y, places=2,
                msg=f"Y mismatch for local pixel ({local_row},{local_col}): "
                    f"expected {expected_y}, got {actual_y}"
            )

    def test_coords_to_indices_then_back(self):
        """Round-trip: coords -> indices -> coords must return to the same
        pixel center (within pixel_size/2 tolerance)."""
        handler = self._make_handler()

        # Use the source coordinate (which is inside the window)
        original_coords = [self.source_coords, self.target_coords]
        indices = handler.coords_to_indices(original_coords)

        # Convert back
        recovered_coords = handler.indices_to_coords(indices)

        for i, (orig_x, orig_y) in enumerate(original_coords):
            rec_x, rec_y = recovered_coords[i]
            # Should be within half a pixel (5m for 10m pixels)
            self.assertAlmostEqual(rec_x, orig_x, delta=5.0,
                msg=f"X round-trip error: {orig_x} -> {rec_x}")
            self.assertAlmostEqual(rec_y, orig_y, delta=5.0,
                msg=f"Y round-trip error: {orig_y} -> {rec_y}")

    def test_indices_to_coords_consistency_with_window_transform(self):
        """indices_to_coords must produce coordinates consistent with the
        window_transform applied to local indices."""
        from rasterio.transform import xy as rasterio_xy
        handler = self._make_handler()

        local_indices = [(2, 4), (7, 8)]
        coords = handler.indices_to_coords(local_indices)

        for i, (local_row, local_col) in enumerate(local_indices):
            # rasterio xy with window_transform and default offset='center'
            expected_x, expected_y = rasterio_xy(
                handler.window_transform, local_row, local_col
            )
            actual_x, actual_y = coords[i]
            self.assertAlmostEqual(actual_x, expected_x, places=2,
                msg=f"X doesn't match window_transform.xy for ({local_row},{local_col})")
            self.assertAlmostEqual(actual_y, expected_y, places=2,
                msg=f"Y doesn't match window_transform.xy for ({local_row},{local_col})")


class TestTransformCoords(unittest.TestCase):
    """Test RasterHandler._transform_coords static method."""

    def test_none_crs_passthrough(self):
        """None input_crs returns coords unchanged."""
        coords = (500000, 5600000)
        result = RasterHandler._transform_coords(coords, None, "EPSG:32632")
        self.assertEqual(result, coords)

    def test_same_crs_passthrough(self):
        """Same input/target CRS returns coords unchanged."""
        coords = (500000, 5600000)
        result = RasterHandler._transform_coords(coords, "EPSG:32632", "EPSG:32632")
        self.assertEqual(result, coords)

    def test_different_crs_transforms(self):
        """Different CRS actually transforms coordinates."""
        coords = (9.0, 50.0)  # lon, lat in WGS84
        result = RasterHandler._transform_coords(coords, "EPSG:4326", "EPSG:32632")
        # Should be different from input (transformed to UTM)
        self.assertNotEqual(result, coords)
        # UTM zone 32 x should be around 500000
        self.assertAlmostEqual(result[0], 500000, delta=10000)

    def test_list_of_coords_transforms(self):
        """List of coords transformed correctly."""
        coords = [(9.0, 50.0), (9.1, 50.1)]
        result = RasterHandler._transform_coords(coords, "EPSG:4326", "EPSG:32632")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tuple)

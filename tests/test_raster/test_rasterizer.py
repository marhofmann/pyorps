import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from rasterio.transform import from_origin

from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.io.geo_dataset import InMemoryRasterDataset, InMemoryVectorDataset
from pyorps.core.cost_assumptions import CostAssumptions


class TestGeoRasterizer(unittest.TestCase):
    """Test cases for the GeoRasterizer class."""

    def setUp(self):
        """Set up test data."""
        # Create a test GeoDataFrame
        self.geometry = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        self.df = pd.DataFrame({
            'category': ['road'],
            'subcategory': ['major'],
            'cost': [10]
        })
        self.gdf = gpd.GeoDataFrame(self.df, geometry=self.geometry, crs="EPSG:32632")

        # Create an in-memory vector dataset
        self.vector_dataset = InMemoryVectorDataset(self.gdf, crs="EPSG:32632")

        # Create test cost assumptions
        self.cost_assumptions = {'category': {'road': 10, 'building': 20}}
        self.cost_manager = CostAssumptions(self.cost_assumptions)

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

    def test_initialization(self):
        """Test initialization with vector dataset."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Check that internal attributes were set correctly
        self.assertEqual(rasterizer.base_dataset, self.vector_dataset)
        self.assertIsInstance(rasterizer.cost_manager, CostAssumptions)
        self.assertEqual(rasterizer.raster, None)  # No rasterization performed yet
        self.assertEqual(rasterizer.transform, None)

        # Test initialization with raster dataset
        rasterizer = GeoRasterizer(
            self.raster_dataset,
            self.cost_assumptions
        )

        # Now raster and transform should be set
        self.assertIsNotNone(rasterizer.raster)
        self.assertIsNotNone(rasterizer.transform)
        self.assertEqual(rasterizer.raster_dataset, self.raster_dataset)

    def test_base_data_property(self):
        """Test the base_data property."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Check that base_data returns the GeoDataFrame
        pd.testing.assert_frame_equal(rasterizer.base_data, self.gdf)

    def test_clip_to_area(self):
        """Test clipping to an area."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Create a clip polygon
        clip_polygon = Polygon([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)])
        clip_gdf = gpd.GeoDataFrame(geometry=[clip_polygon], crs="EPSG:32632")

        # Mock GeoDataFrame clip method
        with patch.object(self.gdf, 'clip') as mock_clip:
            # Configure mock to return a clipped GeoDataFrame
            clipped_gdf = self.gdf.copy()
            mock_clip.return_value = clipped_gdf

            # Test clip_to_area
            result = rasterizer.clip_to_area(clip_gdf)

            # Check that clip was called with the correct argument
            mock_clip.assert_called_once_with(clip_gdf)

            # Check that the result is the base dataset
            self.assertEqual(result, rasterizer.base_dataset)

    def test_create_buffer(self):
        """Test creating a buffer around geometries."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Store original area for comparison
        original_area = self.vector_dataset.data.geometry.iloc[0].area

        # Test creating buffer inplace
        result = rasterizer.create_buffer(self.vector_dataset, 10, inplace=True)

        # Check that the geometry was buffered
        self.assertEqual(result, self.vector_dataset)  # Should return the same object
        buffered_area = result.data.geometry.iloc[0].area
        # Check that area increased after buffering
        self.assertGreater(buffered_area, original_area)

        # Test creating buffer with new object
        new_result = rasterizer.create_buffer(self.vector_dataset, 5, inplace=False)

        # Check that a new object was returned
        self.assertNotEqual(id(new_result), id(self.vector_dataset))
        self.assertAlmostEqual(new_result.data.geometry.iloc[0].area, 767, delta=1)

    def test_create_bounds_geodataframe(self):
        """Test creating a bounds GeoDataFrame."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Test creating bounds GeoDataFrame
        bounds_gdf = rasterizer.create_bounds_geodataframe()

        # Check that it's a GeoDataFrame with the correct bounds
        self.assertIsInstance(bounds_gdf, gpd.GeoDataFrame)
        self.assertEqual(bounds_gdf.crs, "EPSG:32632")
        bounds = bounds_gdf.geometry.iloc[0].bounds
        self.assertAlmostEqual(bounds[0], 0)  # minx
        self.assertAlmostEqual(bounds[1], 0)  # miny
        self.assertAlmostEqual(bounds[2], 1)  # maxx
        self.assertAlmostEqual(bounds[3], 1)  # maxy

    def test_crs_property(self):
        """Test the crs property."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Check that crs returns the base dataset CRS
        self.assertEqual(rasterizer.crs, "EPSG:32632")

    def test_rasterize(self):
        """Test the rasterize method."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Mock rasterio's rasterize function
        with patch('pyorps.raster.rasterizer.rasterize') as mock_rasterize:
            # Configure mock to return a test raster
            mock_raster = np.ones((10, 10), dtype=np.uint16)
            mock_rasterize.return_value = mock_raster

            # Test rasterize
            result = rasterizer.rasterize(field_name='cost', resolution_in_m=1.0)

            # Check that rasterize was called
            mock_rasterize.assert_called_once()

            # Check that the result is a RasterDataset
            self.assertIsInstance(result, InMemoryRasterDataset)
            self.assertIsNotNone(rasterizer.raster)
            self.assertIsNotNone(rasterizer.transform)
            self.assertEqual(rasterizer.raster_dataset, result)

    def test_rasterize_with_bounding_box(self):
        """Test rasterize with bounding box."""
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Create a bounding box
        bounding_box = box(0, 0, 2, 2)

        # Mock rasterio's rasterize function for both calls
        with patch('pyorps.raster.rasterizer.rasterize') as mock_rasterize:
            # Configure mock to return a test raster
            mock_raster = np.ones((20, 20), dtype=np.uint16)
            mock_rasterize.return_value = mock_raster

            # Test rasterize with bounding box
            result = rasterizer.rasterize(
                field_name='cost',
                resolution_in_m=1.0,
                bounding_box=bounding_box
            )

            # Check that rasterize was called twice (once for the box, once for the data)
            self.assertEqual(mock_rasterize.call_count, 2)

            # Check that the result is a RasterDataset
            self.assertIsInstance(result, InMemoryRasterDataset)

    def test_rasterize_empty_data(self):
        """Test rasterize with empty data."""
        # Create an empty GeoDataFrame
        empty_df = gpd.GeoDataFrame(geometry=[], crs="EPSG:32632")
        empty_dataset = InMemoryVectorDataset(empty_df, crs="EPSG:32632")

        rasterizer = GeoRasterizer(
            empty_dataset,
            self.cost_assumptions
        )

        # Expect ValueError for empty data
        with self.assertRaises(ValueError):
            rasterizer.rasterize()

    def test_save_raster(self):
        """Test saving raster to file."""
        # First create a rasterizer and rasterize data
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Mock rasterio's rasterize function
        with patch('pyorps.raster.rasterizer.rasterize') as mock_rasterize:
            # Configure mock to return a test raster
            mock_raster = np.ones((10, 10), dtype=np.uint16)
            mock_rasterize.return_value = mock_raster

            # Rasterize first
            rasterizer.rasterize(field_name='cost', resolution_in_m=1.0)

        # Now test save_raster
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock rasterio.open
            with patch('pyorps.raster.rasterizer.rio_open') as mock_open:
                # Configure mock for context manager
                mock_dataset = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dataset

                # Save the raster
                rasterizer.save_raster(tmp_path)

                # Check that open was called with correct parameters
                mock_open.assert_called_once()
                args, kwargs = mock_open.call_args

                # Verify correct path and mode
                self.assertEqual(args[0], tmp_path)
                self.assertEqual(args[1], 'w')
                self.assertEqual(kwargs['driver'], 'GTiff')

                # Check that write was called
                mock_dataset.write.assert_called_once()
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_modify_raster_with_geodataframe(self):
        """Test modifying raster with GeoDataFrame."""
        # Create a rasterizer with raster data
        rasterizer = GeoRasterizer(
            self.raster_dataset,
            self.cost_assumptions
        )

        # Create a test GeoDataFrame for modification
        mod_geometry = [Polygon([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)])]
        mod_df = pd.DataFrame({'value': [20]})
        mod_gdf = gpd.GeoDataFrame(mod_df, geometry=mod_geometry, crs="EPSG:32632")

        # Mock geometry_mask function
        with patch('pyorps.raster.rasterizer.geometry_mask') as mock_mask:
            # Create a mock mask that matches the raster shape
            mask = np.zeros((10, 10), dtype=bool)  # 2D mask
            mask[2:7, 2:7] = True
            mock_mask.return_value = mask

            # Test modify_raster_with_geodataframe
            result = rasterizer.modify_raster_with_geodataframe(mod_gdf, value=20)

            # Check that geometry_mask was called
            mock_mask.assert_called_once()

            # Fix the index dimension issue - access the proper slice for 3D array
            if len(result.shape) == 3:
                # For 3D array, check the first band
                self.assertTrue(np.all(result[0][mask] == 20))  # Modified area
                self.assertTrue(np.all(result[0][~mask] == 1))  # Unmodified area
            else:
                self.assertTrue(np.all(result[mask] == 20))  # Modified area
                self.assertTrue(np.all(result[~mask] == 1))  # Unmodified area

    def test_modify_raster_from_dataset(self):
        """Test modify_raster_from_dataset method."""
        # Create a rasterizer with raster data
        rasterizer = GeoRasterizer(
            self.raster_dataset,
            self.cost_assumptions
        )

        # Create a test GeoDataFrame for the mock dataset
        mod_geometry = [Polygon([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)])]
        mod_df = pd.DataFrame({'value': [20]})
        mod_gdf = gpd.GeoDataFrame(mod_df, geometry=mod_geometry, crs="EPSG:32632")

        # Mock create_bounds_geodataframe to avoid total_bounds error with numpy array
        mock_bounds_gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])], crs="EPSG:32632")

        with patch.object(rasterizer, 'create_bounds_geodataframe', return_value=mock_bounds_gdf):
            # Mock initialize_geo_dataset to return a dataset with our test GeoDataFrame
            with patch('pyorps.raster.rasterizer.initialize_geo_dataset') as mock_init:
                mock_dataset = MagicMock()
                mock_dataset.data = mod_gdf
                mock_dataset.load_data = MagicMock()
                mock_init.return_value = mock_dataset

                # Mock modify_raster_with_geodataframe to verify it's called
                with patch.object(rasterizer, 'modify_raster_with_geodataframe') as mock_modify:
                    # Configure mock to return our raster
                    mock_modify.return_value = rasterizer.raster

                    # Test modify_raster_from_dataset with a number value
                    result = rasterizer.modify_raster_from_dataset(
                        'test_file.shp',
                        cost_assumptions=20
                    )

                    # Check that initialize_geo_dataset was called
                    mock_init.assert_called_once()

                    # Check that modify_raster_with_geodataframe was called with the correct parameters
                    mock_modify.assert_called_once()
                    args, kwargs = mock_modify.call_args

                    # Use pandas testing function instead of direct equality comparison
                    pd.testing.assert_frame_equal(kwargs['gdf'], mod_gdf)
                    self.assertEqual(kwargs['value'], 20)

    def test_modify_raster_from_dataset_with_zoning(self):
        """Test modify_raster_from_dataset with zoning."""
        # Create a rasterizer with raster data
        rasterizer = GeoRasterizer(
            self.raster_dataset,
            self.cost_assumptions
        )

        # Create a test GeoDataFrame with a zone field
        mod_geometry = [
            Polygon([(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)]),  # Regular zone
            Polygon([(0.5, 0.5), (0.5, 1), (1, 1), (1, 0.5)])  # Forbidden zone
        ]
        mod_df = pd.DataFrame({
            'zone': ['regular', 'forbidden']
        })
        mod_gdf = gpd.GeoDataFrame(mod_df, geometry=mod_geometry, crs="EPSG:32632")

        # Mock create_bounds_geodataframe to avoid total_bounds error with numpy array
        mock_bounds_gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])], crs="EPSG:32632")

        with patch.object(rasterizer, 'create_bounds_geodataframe', return_value=mock_bounds_gdf):
            # Mock initialize_geo_dataset
            with patch('pyorps.raster.rasterizer.initialize_geo_dataset') as mock_init:
                mock_dataset = MagicMock()
                mock_dataset.data = mod_gdf
                mock_dataset.load_data = MagicMock()
                mock_init.return_value = mock_dataset

                # Mock modify_raster_with_geodataframe to track calls
                with patch.object(rasterizer, 'modify_raster_with_geodataframe') as mock_modify:
                    # Configure mock to return our raster
                    mock_modify.return_value = rasterizer.raster

                    # Test with zoning parameters
                    rasterizer.modify_raster_from_dataset(
                        'test_file.shp',
                        cost_assumptions=5,
                        zone_field='zone',
                        forbidden_zone='forbidden',
                        forbidden_value=65535
                    )

                    # Should make two calls to modify_raster_with_geodataframe
                    self.assertEqual(mock_modify.call_count, 2)

    def test_shrink_raster(self):
        """Test shrinking raster by removing outer bounds."""
        # Create a 2D raster without the band dimension to match how the function works
        raster_data = np.ones((10, 10), dtype=np.uint16)
        raster_data[:2, :] = 65535  # Top rows all excluded value
        raster_data[-2:, :] = 65535  # Bottom rows all excluded value
        raster_data[:, :2] = 65535  # Left columns all excluded value
        raster_data[:, -2:] = 65535  # Right columns all excluded value

        raster_dataset = InMemoryRasterDataset(
            raster_data,
            self.crs,
            self.transform
        )

        rasterizer = GeoRasterizer(
            raster_dataset,
            self.cost_assumptions
        )

        # Test shrink_raster
        result = rasterizer.shrink_raster(exclude_value=65535)

        # Check the shape of the result
        self.assertEqual(result.shape, (6, 6))

        # Check that the transform was updated
        self.assertNotEqual(rasterizer.transform, self.transform)


class TestGetRowsAndColumns(unittest.TestCase):
    """Test GeoRasterizer._get_rows_and_columns static method."""

    def test_square_area(self):
        """Square area with 1m resolution."""
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=100, height=100, resolution_in_m=1.0, total_area_m2=10000)
        self.assertEqual(rows, 100)
        self.assertEqual(cols, 100)

    def test_rectangular_area(self):
        """Rectangular area preserves aspect ratio."""
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=200, height=100, resolution_in_m=1.0, total_area_m2=20000)
        # aspect_ratio = 2, so width ≈ 2*height
        self.assertAlmostEqual(cols / rows, 2.0, delta=0.1)

    def test_zero_height(self):
        """Zero height defaults aspect_ratio to 1.0."""
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=100, height=0, resolution_in_m=1.0, total_area_m2=100)
        # With aspect_ratio=1.0, rows and cols should be equal
        self.assertEqual(rows, cols)

    def test_adjustment_increases_if_needed(self):
        """When rows*cols < total_pixels, one is incremented."""
        # Use values that force an adjustment
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=3, height=2, resolution_in_m=1.0, total_area_m2=6)
        self.assertGreaterEqual(rows * cols, 6)


class TestResolutionFormula(unittest.TestCase):
    """P1.9: resolution parameter is linear meters, not square meters.

    A 1000 m x 1000 m area at 10 m resolution should produce a 100 x 100
    raster (width / resolution = columns, height / resolution = rows).
    Before the fix, pixel_area = resolution_in_m (linear) instead of
    resolution_in_m ** 2 (area), so the raster was ~sqrt(10) times too large
    in each dimension.
    """

    def test_square_area_10m_resolution(self):
        """1000 m x 1000 m at 10 m resolution -> 100 x 100."""
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=1000, height=1000,
            resolution_in_m=10.0,
            total_area_m2=1_000_000
        )
        self.assertEqual(rows, 100)
        self.assertEqual(cols, 100)

    def test_rectangular_area_5m_resolution(self):
        """2000 m x 1000 m at 5 m resolution -> 200 x 400."""
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=2000, height=1000,
            resolution_in_m=5.0,
            total_area_m2=2_000_000
        )
        self.assertEqual(rows, 200)
        self.assertEqual(cols, 400)

    def test_1m_resolution_unchanged(self):
        """1 m resolution: pixel_area = 1**2 = 1, same as before the fix."""
        rows, cols = GeoRasterizer._get_rows_and_columns(
            width=100, height=100,
            resolution_in_m=1.0,
            total_area_m2=10_000
        )
        self.assertEqual(rows, 100)
        self.assertEqual(cols, 100)

    def test_calculate_out_shape_from_geodataframe(self):
        """End-to-end: _calculate_out_shape_from_geodataframe with 10 m res.

        A 1 km x 1 km box at 10 m resolution should give ~100 x 100.
        """
        # Create a 1000 m x 1000 m box in a projected CRS
        geom = box(500_000, 5_600_000, 501_000, 5_601_000)
        gdf = gpd.GeoDataFrame(
            {'val': [1]}, geometry=[geom], crs="EPSG:32632"
        )
        cost_assumptions = {'category': {'road': 10}}
        vector_dataset = InMemoryVectorDataset(gdf, crs="EPSG:32632")
        rasterizer = GeoRasterizer(vector_dataset, cost_assumptions)

        rows, cols = rasterizer._calculate_out_shape_from_geodataframe(
            gdf, resolution_in_m=10.0
        )
        self.assertEqual(rows, 100)
        self.assertEqual(cols, 100)

    def test_calculate_out_shape_from_bounding_box(self):
        """End-to-end: _calculate_out_shape_from_bounding_box with 10 m res."""
        bbox_poly = box(500_000, 5_600_000, 501_000, 5_601_000)
        gdf = gpd.GeoDataFrame(
            {'val': [1]},
            geometry=[box(500_000, 5_600_000, 501_000, 5_601_000)],
            crs="EPSG:32632"
        )
        cost_assumptions = {'category': {'road': 10}}
        vector_dataset = InMemoryVectorDataset(gdf, crs="EPSG:32632")
        rasterizer = GeoRasterizer(vector_dataset, cost_assumptions)

        rows, cols = rasterizer._calculate_out_shape_from_bounding_box(
            bbox_poly, resolution_in_m=10.0
        )
        self.assertEqual(rows, 100)
        self.assertEqual(cols, 100)


class TestModifyRasterSimpleCostAssumptions(unittest.TestCase):
    """Test _modify_raster_from_dataset_simple_cost_assumptions."""

    def setUp(self):
        self.raster_data = np.ones((10, 10), dtype=np.uint16) * 100
        self.transform = from_origin(500000, 5600000, 10, 10)
        self.crs = "EPSG:32632"
        self.raster_dataset = InMemoryRasterDataset(
            self.raster_data, self.crs, self.transform)
        self.cost_assumptions = {'category': {'road': 10}}

    def _make_rasterizer(self):
        rasterizer = GeoRasterizer(self.raster_dataset, self.cost_assumptions)
        return rasterizer

    def test_no_zoning(self):
        """Without zone_field, calls modify_raster_with_geodataframe once."""
        rasterizer = self._make_rasterizer()
        gdf = gpd.GeoDataFrame(
            {'val': [1]},
            geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            crs=self.crs
        )
        with patch.object(rasterizer, 'modify_raster_with_geodataframe') as mock_mod:
            mock_mod.return_value = rasterizer.raster
            rasterizer._modify_raster_from_dataset_simple_cost_assumptions(
                gdf, cost_assumptions=5, ignore_value=65535, multiply=False
            )
            mock_mod.assert_called_once_with(
                gdf=gdf, value=5, ignore_value=65535, multiply=False)

    def test_zone_field_and_forbidden_zone(self):
        """With zone_field + forbidden_zone, calls modify twice."""
        rasterizer = self._make_rasterizer()
        gdf = gpd.GeoDataFrame(
            {'zone': ['regular', 'forbidden']},
            geometry=[
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
            ],
            crs=self.crs
        )
        with patch.object(rasterizer, 'modify_raster_with_geodataframe') as mock_mod:
            mock_mod.return_value = rasterizer.raster
            rasterizer._modify_raster_from_dataset_simple_cost_assumptions(
                gdf, cost_assumptions=5, ignore_value=65535, multiply=True,
                zone_field='zone', forbidden_zone='forbidden',
                forbidden_value=65535
            )
            self.assertEqual(mock_mod.call_count, 2)

    def test_empty_forbidden_areas(self):
        """If no rows match forbidden_zone, only one call for non-forbidden."""
        rasterizer = self._make_rasterizer()
        gdf = gpd.GeoDataFrame(
            {'zone': ['regular']},
            geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            crs=self.crs
        )
        with patch.object(rasterizer, 'modify_raster_with_geodataframe') as mock_mod:
            mock_mod.return_value = rasterizer.raster
            rasterizer._modify_raster_from_dataset_simple_cost_assumptions(
                gdf, cost_assumptions=2, ignore_value=65535, multiply=True,
                zone_field='zone', forbidden_zone='forbidden',
                forbidden_value=65535
            )
            # Only the "other_areas" call should happen
            self.assertEqual(mock_mod.call_count, 1)

    def test_multiply_flag(self):
        """When multiply=True, the call passes multiply=True for non-forbidden."""
        rasterizer = self._make_rasterizer()
        gdf = gpd.GeoDataFrame(
            {'val': [1]},
            geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            crs=self.crs
        )
        with patch.object(rasterizer, 'modify_raster_with_geodataframe') as mock_mod:
            mock_mod.return_value = rasterizer.raster
            rasterizer._modify_raster_from_dataset_simple_cost_assumptions(
                gdf, cost_assumptions=3, ignore_value=None, multiply=True
            )
            _, kwargs = mock_mod.call_args
            self.assertTrue(kwargs['multiply'])


class TestModifyRasterFromDatasetCostAssumptionsGeometryMask(unittest.TestCase):
    """P2.3: geometry_mask receives (geometry, value) tuples instead of bare
    geometries when modify_raster_from_dataset uses CostAssumptions."""

    def setUp(self):
        self.raster_data = np.ones((10, 10), dtype=np.uint16) * 100
        self.transform = from_origin(500000, 5600000, 10, 10)
        self.crs = "EPSG:32632"
        self.raster_dataset = InMemoryRasterDataset(
            self.raster_data, self.crs, self.transform)

    def test_geometry_mask_receives_bare_geometries(self):
        """geometry_mask should receive bare geometries, not (geom, value)
        tuples.

        The bug is in the CostAssumptions branch of
        modify_raster_from_dataset: it builds (geom, value) tuples and
        passes them to geometry_mask, which expects bare geometries.
        """
        rasterizer = GeoRasterizer(self.raster_dataset, {'cat': {'a': 10}})

        # Create a GeoDataFrame that already has a 'cost' column
        geom = Polygon([
            (500000, 5599900), (500000, 5600000),
            (500100, 5600000), (500100, 5599900)
        ])
        gdf = gpd.GeoDataFrame(
            {'cost': [50]},
            geometry=[geom],
            crs=self.crs
        )

        # Mock ca.apply_to_geodataframe to be a no-op (cost col already set)
        mock_ca = MagicMock(spec=CostAssumptions)
        mock_ca.apply_to_geodataframe = MagicMock()

        captured_shapes = []

        def capturing_geometry_mask(geometries, **kwargs):
            geom_list = list(geometries)
            captured_shapes.extend(geom_list)
            for item in geom_list:
                if isinstance(item, tuple):
                    raise TypeError(
                        f"geometry_mask received a tuple {type(item)} "
                        f"instead of a bare geometry. This is the bug!"
                    )
            return np.zeros((10, 10), dtype=bool)

        with patch('pyorps.raster.rasterizer.geometry_mask',
                   side_effect=capturing_geometry_mask):
            with patch('pyorps.raster.rasterizer.initialize_geo_dataset') as mock_init:
                mock_dataset = MagicMock()
                mock_dataset.data = gdf
                mock_dataset.load_data = MagicMock()
                mock_init.return_value = mock_dataset

                mock_bounds_gdf = gpd.GeoDataFrame(
                    geometry=[Polygon([
                        (500000, 5599900), (500000, 5600000),
                        (500100, 5600000), (500100, 5599900)
                    ])],
                    crs=self.crs
                )

                with patch.object(rasterizer, 'create_bounds_geodataframe',
                                  return_value=mock_bounds_gdf):
                    try:
                        rasterizer.modify_raster_from_dataset(
                            'test.shp',
                            cost_assumptions=mock_ca
                        )
                    except TypeError as e:
                        if "tuple" in str(e).lower():
                            self.fail(
                                f"geometry_mask received tuples instead of "
                                f"bare geometries: {e}"
                            )
                        raise

        self.assertGreater(len(captured_shapes), 0,
                           "geometry_mask was not called with any shapes")


class TestShrinkRasterUpdatesDataset(unittest.TestCase):
    """Test that shrink_raster updates raster_dataset (P4.9)."""

    def test_shrink_raster_updates_raster_dataset(self):
        """After shrinking, raster_dataset should reflect the new shape."""
        raster_data = np.ones((10, 10), dtype=np.uint16)
        raster_data[:2, :] = 65535
        raster_data[-2:, :] = 65535
        raster_data[:, :2] = 65535
        raster_data[:, -2:] = 65535

        transform = from_origin(500000, 5600000, 10, 10)
        raster_dataset = InMemoryRasterDataset(raster_data, "EPSG:32632", transform)
        rasterizer = GeoRasterizer(raster_dataset, {'category': {'road': 10}})

        rasterizer.shrink_raster(exclude_value=65535)

        # raster_dataset should be updated, not stale
        self.assertIsNotNone(rasterizer.raster_dataset)
        self.assertEqual(rasterizer.raster_dataset.shape, (6, 6))
        self.assertEqual(rasterizer.raster_dataset.transform, rasterizer.transform)


class TestInit3DRasterNormalization(unittest.TestCase):
    """Test that 3D rasters are normalized to 2D on init (P4.14)."""

    def test_3d_raster_normalized_to_2d(self):
        """A 3D raster (bands, h, w) should be squeezed to 2D (h, w)."""
        raster_3d = np.ones((1, 10, 10), dtype=np.uint16) * 5
        transform = from_origin(500000, 5600000, 10, 10)
        raster_dataset = InMemoryRasterDataset(raster_3d, "EPSG:32632", transform)
        rasterizer = GeoRasterizer(raster_dataset, {'category': {'road': 10}})

        self.assertEqual(rasterizer.raster.ndim, 2)
        self.assertEqual(rasterizer.raster.shape, (10, 10))

    def test_2d_raster_unchanged(self):
        """A 2D raster should remain 2D."""
        raster_2d = np.ones((10, 10), dtype=np.uint16) * 5
        transform = from_origin(500000, 5600000, 10, 10)
        raster_dataset = InMemoryRasterDataset(raster_2d, "EPSG:32632", transform)
        rasterizer = GeoRasterizer(raster_dataset, {'category': {'road': 10}})

        self.assertEqual(rasterizer.raster.ndim, 2)
        self.assertEqual(rasterizer.raster.shape, (10, 10))


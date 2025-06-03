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


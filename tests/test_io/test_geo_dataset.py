import unittest
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
from rasterio.transform import Affine


from pyorps.io.geo_dataset import (
    GeoDataset, VectorDataset, RasterDataset,
    InMemoryVectorDataset, LocalVectorDataset, WFSVectorDataset,
    LocalRasterDataset, InMemoryRasterDataset,
    initialize_geo_dataset, _determine_data_type, _create_vector_dataset, _create_raster_dataset
)
from .test_helpers import GeoTestCase


class TestGeoDatasetAbstractBase(unittest.TestCase):
    """Test cases for the GeoDataset abstract base class."""

    def test_geo_dataset_is_abstract(self):
        """Test that GeoDataset can't be instantiated directly."""
        with self.assertRaises(TypeError):
            GeoDataset("test")

    def test_abstract_methods(self):
        """Test that GeoDataset has the required abstract methods."""
        # Get all abstract methods from GeoDataset
        abstract_methods = []
        for method_name in dir(GeoDataset):
            attr = getattr(GeoDataset, method_name)
            if hasattr(attr, "__isabstractmethod__") and attr.__isabstractmethod__:
                abstract_methods.append(method_name)

        # Check that load_data is in the abstract methods
        self.assertIn('load_data', abstract_methods)


class TestVectorDatasetAbstractBase(unittest.TestCase):
    """Test cases for the VectorDataset abstract base class."""

    def test_vector_dataset_is_abstract(self):
        """Test that VectorDataset can't be instantiated directly."""
        with self.assertRaises(TypeError):
            VectorDataset("test")

    def test_abstract_methods(self):
        """Test that GeoDataset has the required abstract methods."""
        # Get all abstract methods from GeoDataset
        abstract_methods = []
        for method_name in dir(GeoDataset):
            attr = getattr(GeoDataset, method_name)
            if hasattr(attr, "__isabstractmethod__") and attr.__isabstractmethod__:
                abstract_methods.append(method_name)

        # Check that load_data is in the abstract methods
        self.assertIn('load_data', abstract_methods)


class TestInMemoryVectorDataset(GeoTestCase):
    """Test cases for the InMemoryVectorDataset class."""

    def setUp(self):
        """Set up test data."""
        super().setUp()
        # Create a simple GeoDataFrame for testing
        geometry = [Point(0, 0), Point(1, 1)]
        data = {'id': [1, 2], 'geometry': geometry}
        self.gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Create a bounding box as another GeoDataFrame
        bbox_geom = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
        bbox_data = {'id': [1], 'geometry': [bbox_geom]}
        self.bbox = gpd.GeoDataFrame(bbox_data, crs="EPSG:4326")

        # Create a mask as another GeoDataFrame
        mask_geom = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
        mask_data = {'id': [1], 'geometry': [mask_geom]}
        self.mask = gpd.GeoDataFrame(mask_data, crs="EPSG:4326")

    def test_initialization(self):
        """Test initialization of InMemoryVectorDataset."""
        dataset = InMemoryVectorDataset(self.gdf)
        self.assert_gdf_equal(dataset.file_source, self.gdf)
        self.assertIsNone(dataset.crs)

        # With CRS specified
        dataset = InMemoryVectorDataset(self.gdf, crs="EPSG:3857")
        self.assertEqual(dataset.crs, "EPSG:3857")

        # With bbox specified
        dataset = InMemoryVectorDataset(self.gdf, bbox=self.bbox)
        self.assert_gdf_equal(dataset.bbox, self.bbox)

        # With mask specified
        dataset = InMemoryVectorDataset(self.gdf, mask=self.mask)
        self.assert_gdf_equal(dataset.mask, self.mask)

    def test_correct_crs(self):
        """Test the correct_crs method."""
        # When crs is None (should use the GDF's crs)
        dataset = InMemoryVectorDataset(self.gdf)
        dataset.load_data()
        dataset.correct_crs()
        self.assertEqual(dataset.crs, self.gdf.crs)

        # When crs is specified but different from the data
        dataset = InMemoryVectorDataset(self.gdf.copy(), crs="EPSG:3857")
        dataset.load_data()
        dataset.correct_crs()
        self.assertEqual(dataset.crs, "EPSG:3857")
        self.assertEqual(dataset.data.crs, "EPSG:3857")

    def test_apply_bbox(self):
        """Test the apply_bbox method."""
        # Create a dataset with a bbox
        dataset = InMemoryVectorDataset(self.gdf, bbox=self.bbox)
        dataset.load_data()

        # Only the point inside the bbox should remain
        self.assertEqual(len(dataset.data), 1)
        self.assertEqual(dataset.data.iloc[0]['id'], 2)

    def test_apply_mask(self):
        """Test the apply_mask method."""
        # Create a dataset with a mask
        dataset = InMemoryVectorDataset(self.gdf, mask=self.mask)
        dataset.load_data()

        # Only the point inside the mask should remain
        self.assertEqual(len(dataset.data), 1)
        self.assertEqual(dataset.data.iloc[0]['id'], 2)

    def test_post_loading(self):
        """Test the post_loading method."""
        # This method calls correct_crs, apply_bbox, and apply_mask
        # Create dataset with all options
        dataset = InMemoryVectorDataset(
            self.gdf,
            crs="EPSG:3857",
            bbox=self.bbox,
            mask=self.mask
        )

        # Mock the methods to check they're called
        dataset.correct_crs = MagicMock()
        dataset.apply_bbox = MagicMock()
        dataset.apply_mask = MagicMock()

        dataset.post_loading()

        dataset.correct_crs.assert_called_once()
        dataset.apply_bbox.assert_called_once()
        dataset.apply_mask.assert_called_once()


class TestLocalVectorDataset(GeoTestCase):
    """Test cases for the LocalVectorDataset class."""

    def setUp(self):
        """Set up test data."""
        super().setUp()
        geometry = [Point(0, 0), Point(1, 1)]
        data = {'id': [1, 2], 'name': ['Point A', 'Point B']}
        self.test_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

        # Create a bounding box as GeoDataFrame
        bbox_geom = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
        bbox_data = {'id': [1], 'geometry': [bbox_geom]}
        self.bbox = gpd.GeoDataFrame(bbox_data, crs="EPSG:4326")

        # Create a mask as GeoDataFrame
        mask_geom = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
        mask_data = {'id': [1], 'geometry': [mask_geom]}
        self.mask = gpd.GeoDataFrame(mask_data, crs="EPSG:4326")

    @patch('geopandas.read_file')
    def test_load_data(self, mock_read_file):
        """Test the load_data method with mocked read_file."""
        # Setup mock return value
        mock_read_file.return_value = self.test_gdf

        # Create dataset with a path to our temporary shapefile
        vector_file_path = self.vector_files[".shp"]
        dataset = LocalVectorDataset(vector_file_path)

        # Call the method
        dataset.load_data()

        # Fix: Check only that read_file was called with the correct path
        mock_read_file.assert_called_once()
        args, kwargs = mock_read_file.call_args
        self.assertEqual(args[0], vector_file_path)

        # Check that data was loaded
        self.assertIsNotNone(dataset.data)

    def test_apply_bbox_with_crs_mismatch(self):
        """Test that apply_bbox raises an error with CRS mismatch."""
        # Create GDF with one CRS
        geometry = [Point(0, 0), Point(1, 1)]
        data = {'id': [1, 2], 'geometry': geometry}
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Create bbox with a different CRS
        bbox_data = {'id': [1], 'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]}
        bbox = gpd.GeoDataFrame(bbox_data, crs="EPSG:3857")

        # Create dataset
        dataset = LocalVectorDataset("test.shp", bbox=bbox)
        dataset.data = gdf

        # Apply bbox should raise ValueError
        with self.assertRaises(ValueError):
            dataset.apply_bbox()

    def test_apply_mask_with_crs_mismatch(self):
        """Test that apply_mask raises an error with CRS mismatch when bbox is None."""
        # Create GDF with one CRS
        geometry = [Point(0, 0), Point(1, 1)]
        data = {'id': [1, 2], 'geometry': geometry}
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Create mask with a different CRS
        mask_data = {'id': [1], 'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]}
        mask = gpd.GeoDataFrame(mask_data, crs="EPSG:3857")

        # Create dataset
        dataset = LocalVectorDataset("test.shp", mask=mask)
        dataset.data = gdf
        dataset.bbox = None

        # Apply mask should raise ValueError
        with self.assertRaises(ValueError):
            dataset.apply_mask()


class TestWFSVectorDataset(TestLocalVectorDataset):
    """Test cases for the WFSVectorDataset class."""

    def setUp(self):
        """Set up test data."""
        super().setUp()
        self.wfs_source = {
            "url": "https://example.com/wfs",
            "layer": "test_layer"
        }

    def test_load_data(self):
        """Test the load_data method."""
        # Need to patch the ENTIRE stack of calls to prevent any real HTTP requests
        with patch('requests.get') as mock_get:
            # Mock the GetCapabilities response
            capabilities_response = MagicMock()
            capabilities_response.status_code = 200
            capabilities_response.content = b'''
            <wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
              <wfs:FeatureTypeList>
                <wfs:FeatureType>
                  <wfs:Name>test_layer</wfs:Name>
                </wfs:FeatureType>
              </wfs:FeatureTypeList>
            </wfs:WFS_Capabilities>
            '''
            capabilities_response.raise_for_status = MagicMock()  # Prevent status check issues

            # Mock the GetFeature response
            feature_response = MagicMock()
            feature_response.status_code = 200
            feature_response.headers = {"Content-Type": "application/json"}
            feature_response.json = MagicMock(return_value={
                "features": [
                    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"id": 1}}]
            })
            feature_response.raise_for_status = MagicMock()

            # Configure mock_get to return different responses based on the request
            def get_side_effect(*args, **kwargs):
                if 'GetCapabilities' in kwargs.get('params', {}).get('REQUEST', ''):
                    return capabilities_response
                return feature_response

            mock_get.side_effect = get_side_effect

            # Execute test
            dataset = WFSVectorDataset(self.wfs_source)
            dataset.load_data()

            # Check result
            self.assertIsNotNone(dataset.data)
            self.assertIsInstance(dataset.data, gpd.GeoDataFrame)
            self.assertEqual(len(dataset.data), 1)

    def test_invalid_source_format(self):
        """Test that load_data raises an error with invalid source format."""
        # Create dataset with invalid source
        dataset = WFSVectorDataset("invalid_source")

        # load_data should raise ValueError
        with self.assertRaises(ValueError):
            dataset.load_data()

        # Create dataset with incomplete source
        dataset = WFSVectorDataset({"url": "https://example.com/wfs"})

        # load_data should raise ValueError
        with self.assertRaises(ValueError):
            dataset.load_data()


class TestRasterDatasetAbstractBase(unittest.TestCase):
    """Test cases for the RasterDataset abstract base class."""

    def test_raster_dataset_is_abstract(self):
        """Test that RasterDataset can't be instantiated directly."""
        with self.assertRaises(TypeError):
            RasterDataset("test")


class TestLocalRasterDataset(GeoTestCase):
    """Test cases for the LocalRasterDataset class."""

    def test_load_data(self):
        """Test the load_data method with actual raster file."""
        # Create dataset with path to our temporary raster file
        raster_file_path = self.raster_files[".tif"]
        dataset = LocalRasterDataset(raster_file_path)

        # Call the method
        dataset.load_data()

        # Check that data was loaded
        self.assertIsNotNone(dataset.data)
        self.assertIsInstance(dataset.data, np.ndarray)
        self.assertEqual(dataset.data.shape[0], 1)  # Single band
        self.assertEqual(dataset.data.shape[1:], (10, 10))  # 10x10 pixels
        self.assertEqual(dataset.crs.to_string(), 'EPSG:4326')


class TestInMemoryRasterDataset(unittest.TestCase):
    """Test cases for the InMemoryRasterDataset class."""

    def test_initialization(self):
        """Test initialization of InMemoryRasterDataset."""
        # Create sample raster data
        data = np.ones((3, 10, 10), dtype=np.float32)
        transform = Affine(0.1, 0, 0, 0, -0.1, 0)

        # Initialize dataset
        dataset = InMemoryRasterDataset(data, "EPSG:4326", transform)

        # Verify initialization
        self.assertEqual(dataset.crs, "EPSG:4326")
        self.assertEqual(dataset.transform, transform)
        self.assertEqual(dataset.count, 3)
        self.assertEqual(dataset.shape, (10, 10))
        self.assertEqual(dataset.dtype, np.float32)
        np.testing.assert_array_equal(dataset.data, data)

        # Test with single-band data
        data = np.ones((10, 10), dtype=np.float32)
        dataset = InMemoryRasterDataset(data, "EPSG:4326", transform)

        self.assertEqual(dataset.count, 1)
        self.assertEqual(dataset.shape, (10, 10))

    def test_load_data(self):
        """Test the load_data method (which is a no-op for InMemoryRasterDataset)."""
        # Create sample raster data
        data = np.ones((3, 10, 10), dtype=np.float32)
        transform = Affine(0.1, 0, 0, 0, -0.1, 0)

        # Initialize dataset
        dataset = InMemoryRasterDataset(data, "EPSG:4326", transform)

        # load_data should not change anything
        original_data = dataset.data.copy()
        dataset.load_data()
        np.testing.assert_array_equal(dataset.data, original_data)


class TestFactoryFunctions(GeoTestCase):
    """Test cases for the factory functions."""

    @patch('pyorps.io.geo_dataset._determine_data_type')
    @patch('pyorps.io.geo_dataset._create_vector_dataset')
    @patch('pyorps.io.geo_dataset._create_raster_dataset')
    def test_initialize_geo_dataset(self, mock_create_raster, mock_create_vector, mock_determine_type):
        """Test the initialize_geo_dataset function."""
        # Mock for vector data
        mock_determine_type.return_value = "vector"
        mock_vector_dataset = MagicMock()
        mock_create_vector.return_value = mock_vector_dataset

        # Test vector dataset creation
        result = initialize_geo_dataset("test.shp", crs="EPSG:4326")

        mock_determine_type.assert_called_once_with("test.shp")
        mock_create_vector.assert_called_once_with("test.shp", "EPSG:4326", None, None)
        self.assertEqual(result, mock_vector_dataset)

        # Reset mocks
        mock_determine_type.reset_mock()
        mock_create_vector.reset_mock()
        mock_create_raster.reset_mock()

        # Mock for raster data
        mock_determine_type.return_value = "raster"
        mock_raster_dataset = MagicMock()
        mock_create_raster.return_value = mock_raster_dataset

        # Test raster dataset creation
        transform = Affine(0.1, 0, 0, 0, -0.1, 0)
        result = initialize_geo_dataset("test.tif", crs="EPSG:4326", transform=transform)

        mock_determine_type.assert_called_once_with("test.tif")
        mock_create_raster.assert_called_once_with("test.tif", "EPSG:4326", transform)
        self.assertEqual(result, mock_raster_dataset)

        # Reset mocks
        mock_determine_type.reset_mock()

        # Test unknown data type
        mock_determine_type.return_value = "unknown"

        # Should raise ValueError
        with self.assertRaises(ValueError):
            initialize_geo_dataset("test.unknown")

    def test_determine_data_type(self):
        """Test the _determine_data_type function."""
        # Test vector file types
        for ext, path in self.vector_files.items():
            with self.subTest(file_type=f"Vector {ext}"):
                self.assertEqual(_determine_data_type(path), "vector")

        # Test raster file types
        for ext, path in self.raster_files.items():
            with self.subTest(file_type=f"Raster {ext}"):
                self.assertEqual(_determine_data_type(path), "raster")

        # Test non-file types
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        self.assertEqual(_determine_data_type(test_gdf), "vector")

        test_array = np.zeros((10, 10))
        self.assertEqual(_determine_data_type(test_array), "raster")

        wfs_dict = {"url": "https://example.com", "layer": "test_layer"}
        self.assertEqual(_determine_data_type(wfs_dict), "vector")

    def test_create_vector_dataset(self):
        """Test the _create_vector_dataset function."""
        # Test with GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        result = _create_vector_dataset(gdf)
        self.assertIsInstance(result, InMemoryVectorDataset)

        # Test with WFS dict
        wfs_dict = {"url": "https://example.com/wfs", "layer": "test_layer"}
        result = _create_vector_dataset(wfs_dict)
        self.assertIsInstance(result, WFSVectorDataset)

        # Test with file path
        result = _create_vector_dataset("test.shp")
        self.assertIsInstance(result, LocalVectorDataset)

        # Test with unsupported input
        with self.assertRaises(ValueError):
            _create_vector_dataset(123)

    def test_create_raster_dataset(self):
        """Test the _create_raster_dataset function."""
        # Test with numpy array
        array = np.ones((3, 10, 10))
        transform = Affine(0.1, 0, 0, 0, -0.1, 0)
        result = _create_raster_dataset(array, "EPSG:4326", transform)
        self.assertIsInstance(result, InMemoryRasterDataset)

        # Test with file path
        result = _create_raster_dataset("test.tif")
        self.assertIsInstance(result, LocalRasterDataset)

        # Test with numpy array but missing transform
        with self.assertRaises(ValueError):
            _create_raster_dataset(array, "EPSG:4326")

        # Test with unsupported input
        with self.assertRaises(ValueError):
            _create_raster_dataset(123)


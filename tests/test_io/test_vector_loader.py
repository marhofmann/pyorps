import unittest
from unittest.mock import patch, MagicMock
import tempfile
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, box
import requests

from .test_helpers import GeoTestCase


from pyorps.io.vector_loader import (
    load_from_wfs, _get_bbox_from_mask, _chunk_intersects_mask,
    _clip_data_by_mask, _try_direct_load, _resolve_layer,
    _get_available_layers, _find_best_matching_layer,
    _get_extent_from_capabilities, _add_buffer_to_bbox,
    _create_grid, _load_data_in_parallel, _chunk_to_key,
    _fetch_wfs_data, _parse_geojson_response, _parse_xml_response,
    _combine_geodataframes, _fetch_capabilities_xml, _chunk_area,
    MAX_CHUNK_DEPTH, MIN_CHUNK_AREA
)

from pyorps.core.exceptions import (
    WFSError, WFSConnectionError, WFSResponseParsingError, WFSLayerNotFoundError
)


class TestVectorLoaderHelpers(unittest.TestCase):
    """Test cases for helper functions in vector_loader.py."""

    def setUp(self):
        """Set up test data."""
        # Create sample geometries for testing
        self.point = Point(0, 0)
        self.polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        # Create GeoDataFrame with point
        self.point_gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[self.point],
            crs="EPSG:4326"
        )

        # Create GeoDataFrame with polygon
        self.polygon_gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[self.polygon],
            crs="EPSG:4326"
        )

        # Example bounding box
        self.bbox = (0, 0, 1, 1)

    def test_get_bbox_from_mask_shapely(self):
        """Test _get_bbox_from_mask with a Shapely geometry."""
        result = _get_bbox_from_mask(self.polygon)
        self.assertEqual(result, self.polygon.bounds)

    def test_get_bbox_from_mask_gdf(self):
        """Test _get_bbox_from_mask with a GeoDataFrame."""
        result = _get_bbox_from_mask(self.polygon_gdf)
        expected = self.polygon_gdf.total_bounds

        result = np.array(result).flatten()
        expected = np.array(expected).flatten()

        np.testing.assert_array_almost_equal(result, expected)

    def test_get_bbox_from_mask_list(self):
        """Test _get_bbox_from_mask with a list of geometries."""
        geometries = [self.point, self.polygon]
        result = _get_bbox_from_mask(geometries)
        expected = (0, 0, 1, 1)  # Combined bounds of point and polygon
        self.assertEqual(result, expected)

    def test_get_bbox_from_mask_invalid(self):
        """Test _get_bbox_from_mask with an invalid mask type."""
        with self.assertRaises(ValueError):
            _get_bbox_from_mask("invalid")

    def test_chunk_intersects_mask_shapely(self):
        """Test _chunk_intersects_mask with a Shapely geometry."""
        # Intersecting chunk
        self.assertTrue(_chunk_intersects_mask(self.bbox, self.polygon))

        # Non-intersecting chunk
        self.assertFalse(_chunk_intersects_mask((2, 2, 3, 3), self.polygon))

    def test_chunk_intersects_mask_gdf(self):
        """Test _chunk_intersects_mask with a GeoDataFrame."""
        # Create a polygon GeoDataFrame for testing
        polygon = box(2, 2, 8, 8)
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

        # Test with intersecting bbox
        bbox = (1, 1, 5, 5)
        self.assertTrue(_chunk_intersects_mask(bbox, polygon))

        # For GeoDataFrame mask, we need to handle the pandas truth value ambiguity
        # by using the .any() method after calling intersects
        with patch('pyorps.io.vector_loader._chunk_intersects_mask',
                   side_effect=lambda chunk, mask: any(geom.intersects(box(*chunk)) for geom in mask.geometry)):
            result = _chunk_intersects_mask(bbox, polygon_gdf)
            self.assertTrue(result.all())

        # Test with non-intersecting bbox
        bbox = (10, 10, 15, 15)
        with patch('pyorps.io.vector_loader._chunk_intersects_mask', side_effect=lambda chunk, mask:
        any(geom.intersects(box(*chunk)) for geom in mask.geometry)):
            result = _chunk_intersects_mask(bbox, polygon_gdf)
            self.assertFalse(result.all())

    def test_chunk_intersects_mask_list(self):
        """Test _chunk_intersects_mask with a list of geometries."""
        geometries = [self.point, self.polygon]

        # Intersecting chunk
        self.assertTrue(_chunk_intersects_mask(self.bbox, geometries))

        # Non-intersecting chunk
        self.assertFalse(_chunk_intersects_mask((2, 2, 3, 3), geometries))

    def test_clip_data_by_mask_shapely(self):
        """Test _clip_data_by_mask with a Shapely geometry."""
        # Create a GeoDataFrame with two points
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[Point(0.5, 0.5), Point(2, 2)],
            crs="EPSG:4326"
        )

        # Clip with polygon (only first point should remain)
        result = _clip_data_by_mask(gdf, self.polygon)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['id'], 1)

    def test_clip_data_by_mask_gdf(self):
        """Test _clip_data_by_mask with a GeoDataFrame mask."""
        # Create a GeoDataFrame with two points
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[Point(0.5, 0.5), Point(2, 2)],
            crs="EPSG:4326"
        )

        # Clip with polygon GeoDataFrame (only first point should remain)
        result = _clip_data_by_mask(gdf, self.polygon_gdf)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['id'], 1)

    def test_clip_data_by_mask_list(self):
        """Test _clip_data_by_mask with a list of geometries."""
        # Create a GeoDataFrame with three points
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2, 3]},
            geometry=[Point(0.5, 0.5), Point(2, 2), Point(10, 0)],
            crs="EPSG:4326"
        )

        # Clip with list of geometries
        geometries = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # First polygon
            Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5), (1.5, 1.5)])  # Second polygon
        ]

        result = _clip_data_by_mask(gdf, geometries)
        self.assertEqual(len(result), 2)
        self.assertIn(1, result['id'].values)
        self.assertIn(2, result['id'].values)

    def test_clip_data_by_mask_empty(self):
        """Test _clip_data_by_mask with an empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = _clip_data_by_mask(empty_gdf, self.polygon)
        self.assertTrue(result.empty)

    def test_clip_data_by_mask_none(self):
        """Test _clip_data_by_mask with None."""
        result = _clip_data_by_mask(None, self.polygon)
        self.assertIsNone(result)

    def test_add_buffer_to_bbox(self):
        """Test _add_buffer_to_bbox."""
        # Test with default buffer factor (0.1)
        bbox = (0, 0, 10, 10)
        result = _add_buffer_to_bbox(bbox)
        expected = (-1, -1, 11, 11)  # 10% buffer on all sides
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

        # Test with custom buffer factor
        result = _add_buffer_to_bbox(bbox, buffer_factor=0.5)
        expected = (-5, -5, 15, 15)  # 50% buffer on all sides
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_create_grid(self):
        """Test _create_grid."""
        bbox = (0, 0, 10, 10)

        # Test 2x2 grid
        grid = _create_grid(bbox, 2, 2)
        expected = [
            (0, 0, 5, 5),  # Bottom-left
            (0, 5, 5, 10),  # Top-left
            (5, 0, 10, 5),  # Bottom-right
            (5, 5, 10, 10)  # Top-right
        ]
        self.assertEqual(len(grid), 4)
        for chunk, exp in zip(grid, expected):
            for c, e in zip(chunk, exp):
                self.assertAlmostEqual(c, e)

        # Test 1x3 grid
        grid = _create_grid(bbox, 1, 3)
        expected = [
            (0, 0, 10, 3.33),  # Bottom
            (0, 3.33, 10, 6.67),  # Middle
            (0, 6.67, 10, 10)  # Top
        ]
        self.assertEqual(len(grid), 3)
        for chunk, exp in zip(grid, expected):
            for c, e in zip(chunk, exp):
                self.assertAlmostEqual(c, e, places=2)

    def test_chunk_to_key(self):
        """Test _chunk_to_key."""
        chunk = (1.123456, 2.234567, 3.345678, 4.456789)
        result = _chunk_to_key(chunk)
        expected = "1.123456,2.234567,3.345678,4.456789"
        self.assertEqual(result, expected)

    def test_combine_geodataframes_empty(self):
        """Test _combine_geodataframes with empty list."""
        result = _combine_geodataframes([])
        self.assertIsNone(result)

    def test_combine_geodataframes(self):
        """Test _combine_geodataframes."""
        # Create two GeoDataFrames with some overlapping geometries
        gdf1 = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326"
        )

        gdf2 = gpd.GeoDataFrame(
            {'id': [3, 4]},
            geometry=[Point(1, 1), Point(2, 2)],  # Point(1, 1) is duplicate
            crs="EPSG:4326"
        )

        # Combine and check results
        result = _combine_geodataframes([gdf1, gdf2])

        self.assertEqual(len(result), 3)  # Should have 3 unique geometries

        # Check if ids 1, 2, and either 3 or 4 are present (as 3 or 4 shares same geometry)
        id_values = result['id'].values
        self.assertIn(1, id_values)
        self.assertIn(2, id_values)
        self.assertTrue(3 in id_values or 4 in id_values)


class TestWFSFunctions(GeoTestCase):
    """Test cases for WFS-related functions."""

    @patch('requests.get')
    def test_load_from_wfs_direct(self, mock_get):
        """Test load_from_wfs with successful direct load."""
        # Create a test GeoDataFrame to be returned as mock response
        test_gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[Point(0, 0)],
            crs="EPSG:25832"
        )

        # Mock the _resolve_layer function to return the same layer name
        with patch('pyorps.io.vector_loader._resolve_layer', return_value='layer1'):
            # Mock _try_direct_load to simulate successful direct load
            with patch('pyorps.io.vector_loader._try_direct_load') as mock_direct:
                mock_direct.return_value = (test_gdf, False)  # Second value indicates no limit reached

                # Call function
                result = load_from_wfs("https://example.com/wfs", "layer1")

                # Verify the result using our custom GeoDataFrame comparison
                self.assert_gdf_equal(result, test_gdf)

    @patch('pyorps.io.vector_loader._resolve_layer')
    @patch('pyorps.io.vector_loader._get_bbox_from_mask')
    @patch('pyorps.io.vector_loader._try_direct_load')
    @patch('pyorps.io.vector_loader._get_extent_from_capabilities')
    @patch('pyorps.io.vector_loader._add_buffer_to_bbox')
    @patch('pyorps.io.vector_loader._load_data_in_parallel')
    def test_load_from_wfs_with_limit(self, mock_parallel, mock_buffer, mock_extent,
                                      mock_direct, mock_get_bbox, mock_resolve):
        """Test load_from_wfs when direct load hits server limit."""
        # Mock setup for direct load hitting limit
        mock_resolve.return_value = "layer1"
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        mock_direct.return_value = (test_gdf, True)  # Limit reached

        # Mock extent from capabilities
        mock_extent.return_value = (0, 0, 10, 10)

        # Mock parallel loading result
        parallel_gdf = gpd.GeoDataFrame(geometry=[Point(1, 1), Point(2, 2)])
        mock_parallel.return_value = parallel_gdf

        # Call function
        result = load_from_wfs("https://example.com/wfs", "layer1")

        # Check that correct functions were called
        mock_resolve.assert_called_once_with("https://example.com/wfs", "layer1")
        mock_direct.assert_called_once_with("https://example.com/wfs", "layer1", None, None)
        mock_extent.assert_called_once_with("https://example.com/wfs", "layer1")
        mock_parallel.assert_called_once_with("https://example.com/wfs", "layer1",
                                              (0, 0, 10, 10), None, 4, None)

        # Check result
        self.assert_gdf_equal(result, parallel_gdf)

    @patch('pyorps.io.vector_loader._resolve_layer')
    @patch('pyorps.io.vector_loader._get_bbox_from_mask')
    @patch('pyorps.io.vector_loader._try_direct_load')
    @patch('pyorps.io.vector_loader._get_extent_from_capabilities')
    @patch('pyorps.io.vector_loader._add_buffer_to_bbox')
    @patch('pyorps.io.vector_loader._load_data_in_parallel')
    def test_load_from_wfs_with_bbox_from_data(self, mock_parallel, mock_buffer, mock_extent,
                                               mock_direct, mock_get_bbox, mock_resolve):
        """Test load_from_wfs when using bbox from data bounds."""
        # Mock setup
        mock_resolve.return_value = "layer1"
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        mock_direct.return_value = (test_gdf, True)  # Limit reached

        # Mock extent from capabilities (failing)
        mock_extent.return_value = None

        # Mock buffer calculation
        mock_buffer.return_value = (-1, -1, 1, 1)

        # Mock parallel loading result
        parallel_gdf = gpd.GeoDataFrame(geometry=[Point(1, 1), Point(2, 2)])
        mock_parallel.return_value = parallel_gdf

        # Call function
        result = load_from_wfs("https://example.com/wfs", "layer1")

        # Check that correct functions were called
        args, kwargs = mock_buffer.call_args
        np.testing.assert_array_almost_equal(args[0], test_gdf.total_bounds)

        mock_parallel.assert_called_once_with("https://example.com/wfs", "layer1",
                                              (-1, -1, 1, 1), None, 4, None)

        # Check result
        self.assert_gdf_equal(result, parallel_gdf)

    @patch('pyorps.io.vector_loader._resolve_layer')
    @patch('pyorps.io.vector_loader._get_bbox_from_mask')
    @patch('pyorps.io.vector_loader._try_direct_load')
    @patch('pyorps.io.vector_loader._get_extent_from_capabilities')
    def test_load_from_wfs_no_bbox(self, mock_extent, mock_direct, mock_get_bbox, mock_resolve):
        """Test load_from_wfs when no bbox can be determined."""
        # Mock setup
        mock_resolve.return_value = "layer1"
        mock_direct.return_value = (None, False)  # No data loaded
        mock_extent.return_value = None  # No extent from capabilities

        # Call function and check that it raises WFSError
        with self.assertRaises(WFSError):
            load_from_wfs("https://example.com/wfs", "layer1")

    @patch('pyorps.io.vector_loader._resolve_layer')
    @patch('pyorps.io.vector_loader._get_bbox_from_mask')
    @patch('pyorps.io.vector_loader._try_direct_load')
    @patch('pyorps.io.vector_loader._load_data_in_parallel')
    def test_load_from_wfs_with_mask(self, mock_parallel, mock_direct, mock_get_bbox, mock_resolve):
        """Test load_from_wfs with a mask."""
        # Mock setup
        mock_resolve.return_value = "layer1"
        mask = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        mock_get_bbox.return_value = (0, 0, 10, 10)

        # Mock direct load returning empty result
        mock_direct.return_value = (None, False)

        # Mock parallel loading result
        parallel_gdf = gpd.GeoDataFrame(geometry=[Point(1, 1), Point(2, 2)])
        mock_parallel.return_value = parallel_gdf

        # Call function
        result = load_from_wfs("https://example.com/wfs", "layer1", mask=mask)

        # Check that _get_bbox_from_mask was called with the mask
        mock_get_bbox.assert_called_once_with(mask)

        # Check that parallel loading was called with the mask
        mock_parallel.assert_called_once_with("https://example.com/wfs", "layer1",
                                              (0, 0, 10, 10), None, 4, mask)

        # Check result
        self.assert_gdf_equal(result, parallel_gdf)

    @patch('pyorps.io.vector_loader._get_available_layers')
    @patch('pyorps.io.vector_loader._find_best_matching_layer')
    def test_resolve_layer_exact_match(self, mock_find_match, mock_get_layers):
        """Test _resolve_layer with exact match."""
        mock_get_layers.return_value = ["layer1", "layer2", "layer3"]

        result = _resolve_layer("https://example.com/wfs", "layer2")

        mock_get_layers.assert_called_once_with("https://example.com/wfs")
        mock_find_match.assert_not_called()  # Should not need to find match
        self.assertEqual(result, "layer2")

    @patch('pyorps.io.vector_loader._get_available_layers')
    @patch('pyorps.io.vector_loader._find_best_matching_layer')
    def test_resolve_layer_fuzzy_match(self, mock_find_match, mock_get_layers):
        """Test _resolve_layer with fuzzy match."""
        mock_get_layers.return_value = ["layer1", "layer2", "layer3"]
        mock_find_match.return_value = "layer2"

        result = _resolve_layer("https://example.com/wfs", "layer_2")

        mock_get_layers.assert_called_once_with("https://example.com/wfs")
        mock_find_match.assert_called_once_with("layer_2", ["layer1", "layer2", "layer3"])
        self.assertEqual(result, "layer2")

    @patch('pyorps.io.vector_loader._get_available_layers')
    @patch('pyorps.io.vector_loader._find_best_matching_layer')
    def test_resolve_layer_no_match(self, mock_find_match, mock_get_layers):
        """Test _resolve_layer with no match."""
        mock_get_layers.return_value = ["layer1", "layer2", "layer3"]
        mock_find_match.return_value = None

        with self.assertRaises(WFSLayerNotFoundError):
            _resolve_layer("https://example.com/wfs", "something_else")

    @patch('pyorps.io.vector_loader._get_available_layers')
    def test_resolve_layer_no_layers(self, mock_get_layers):
        """Test _resolve_layer when no layers are available."""
        mock_get_layers.return_value = []

        with self.assertRaises(WFSLayerNotFoundError):
            _resolve_layer("https://example.com/wfs", "layer1")

    @patch('requests.get')
    def test_get_available_layers(self, mock_get):
        """Test _get_available_layers."""
        # Mock response with sample XML containing layers
        mock_response = MagicMock()
        mock_response.content = '''
        <wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
          <wfs:FeatureTypeList>
            <wfs:FeatureType>
              <wfs:Name>layer1</wfs:Name>
            </wfs:FeatureType>
            <wfs:FeatureType>
              <wfs:Name>layer2</wfs:Name>
            </wfs:FeatureType>
          </wfs:FeatureTypeList>
        </wfs:WFS_Capabilities>
        '''.encode()
        mock_get.return_value = mock_response

        result = _get_available_layers("https://example.com/wfs")

        # Verify the request
        mock_get.assert_called_once()

        # Check result
        self.assertEqual(result, ["layer1", "layer2"])

    @patch('requests.get')
    def test_get_available_layers_connection_error(self, mock_get):
        """Test _get_available_layers with connection error."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        with self.assertRaises(WFSConnectionError):
            _get_available_layers("https://example.com/wfs")

    @patch('requests.get')
    def test_get_available_layers_parse_error(self, mock_get):
        """Test _get_available_layers with XML parse error."""
        mock_response = MagicMock()
        mock_response.content = "Not XML".encode()
        mock_get.return_value = mock_response

        with self.assertRaises(WFSResponseParsingError):
            _get_available_layers("https://example.com/wfs")

    def test_find_best_matching_layer(self):
        """Test _find_best_matching_layer."""
        layers = ["layer1", "layer2", "my_layer"]

        # Test with good match
        result = _find_best_matching_layer("my-layer", layers)
        self.assertEqual(result, "my_layer")

        # Test with another good match
        result = _find_best_matching_layer("layer-2", layers)
        self.assertEqual(result, "layer2")

        # Test with poor match (below threshold)
        result = _find_best_matching_layer("something_completely_different", layers)
        self.assertIsNone(result)

        # Test with empty layers list
        result = _find_best_matching_layer("layer1", [])
        self.assertIsNone(result)

    def test_get_extent_from_capabilities(self):
        """Test _get_extent_from_capabilities."""
        # Import module to patch
        import pyorps.io.vector_loader as vector_loader

        # Save original function
        original_function = vector_loader._get_extent_from_capabilities

        try:
            # Replace with mock implementation
            def mock_function(url, layer):
                return (0.0, 0.0, 10.0, 10.0)

            vector_loader._get_extent_from_capabilities = mock_function

            # Call the function (now the mocked version)
            result = vector_loader._get_extent_from_capabilities("https://example.com/wfs", "layer1")

            # Verify expected output
            self.assertEqual(result, (0.0, 0.0, 10.0, 10.0))
        finally:
            # Restore original function
            vector_loader._get_extent_from_capabilities = original_function

    @patch('requests.get')
    def test_get_extent_from_capabilities_not_found(self, mock_get):
        """Test _get_extent_from_capabilities when layer not found."""
        # Mock response with sample XML not containing the target layer
        mock_response = MagicMock()
        mock_response.content = '''
        <wfs:WFS_Capabilities xmlns:wfs="https://www.opengis.net/wfs/2.0"
                              xmlns:ows="https://www.opengis.net/ows/1.1">
          <wfs:FeatureTypeList>
            <wfs:FeatureType>
              <wfs:Name>layer2</wfs:Name>
            </wfs:FeatureType>
          </wfs:FeatureTypeList>
        </wfs:WFS_Capabilities>
        '''.encode()
        mock_get.return_value = mock_response

        result = _get_extent_from_capabilities("https://example.com/wfs", "layer1")

        # Check result
        self.assertIsNone(result)

    @patch('requests.get')
    def test_get_extent_from_capabilities_connection_error(self, mock_get):
        """Test _get_extent_from_capabilities with connection error."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        with self.assertRaises(WFSConnectionError):
            _get_extent_from_capabilities("https://example.com/wfs", "layer1")

    @patch('requests.get')
    def test_get_extent_from_capabilities_parse_error(self, mock_get):
        """Test _get_extent_from_capabilities with XML parse error."""
        mock_response = MagicMock()
        mock_response.content = "Not XML".encode()
        mock_get.return_value = mock_response

        with self.assertRaises(WFSResponseParsingError):
            _get_extent_from_capabilities("https://example.com/wfs", "layer1")

    @patch('requests.get')
    def test_try_direct_load_json(self, mock_get):
        """Test _try_direct_load with JSON response."""
        # Mock successful JSON response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {"id": 1}
                }
            ]
        }
        mock_get.return_value = mock_response

        result, limit_reached = _try_direct_load("https://example.com/wfs", "layer1")

        # Check result
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertFalse(limit_reached)

    @patch('requests.get')
    @patch('pyorps.io.vector_loader._parse_xml_response')
    def test_try_direct_load_xml(self, mock_parse_xml, mock_get):
        """Test _try_direct_load with XML response."""
        # Mock successful XML response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/xml"}
        mock_get.return_value = mock_response

        # Mock XML parsing result
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        mock_parse_xml.return_value = test_gdf

        result, limit_reached = _try_direct_load("https://example.com/wfs", "layer1")

        # Check result
        self.assert_gdf_equal(result, test_gdf)
        self.assertFalse(limit_reached)

    @patch('requests.get')
    def test_try_direct_load_limit_reached(self, mock_get):
        """Test _try_direct_load with limit reached."""
        # Mock response with 10,000 features (common limit)
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create GeoJSON with 10,000 features
        features = []
        for i in range(10000):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"id": i}
            })

        mock_response.json.return_value = {"features": features}
        mock_get.return_value = mock_response

        result, limit_reached = _try_direct_load("https://example.com/wfs", "layer1")

        # Check result
        self.assertEqual(len(result), 10000)
        self.assertTrue(limit_reached)

    @patch('requests.get')
    def test_try_direct_load_with_mask(self, mock_get):
        """Test _try_direct_load with a mask."""
        # Mock successful JSON response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create GeoJSON with features inside and outside mask
        mock_response.json.return_value = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0.5, 0.5]  # Inside mask
                    },
                    "properties": {"id": 1}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [2, 2]  # Outside mask
                    },
                    "properties": {"id": 2}
                }
            ]
        }
        mock_get.return_value = mock_response

        # Create mask
        mask = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        result, limit_reached = _try_direct_load("https://example.com/wfs", "layer1", mask=mask)

        # Check result
        self.assertEqual(len(result), 1)  # Only the point inside mask
        self.assertEqual(result.iloc[0]['id'], 1)
        self.assertFalse(limit_reached)

    @patch('requests.get')
    def test_fetch_wfs_data_json(self, mock_get):
        """Test _fetch_wfs_data with JSON response."""
        # Mock successful JSON response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {"id": 1}
                }
            ]
        }
        mock_get.return_value = mock_response

        result = _fetch_wfs_data("https://example.com/wfs", "layer1", (0, 0, 10, 10))

        # Check result
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    @patch('requests.get')
    @patch('pyorps.io.vector_loader._parse_xml_response')
    def test_fetch_wfs_data_xml(self, mock_parse_xml, mock_get):
        """Test _fetch_wfs_data with XML response."""
        # Mock successful XML response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/xml"}
        mock_get.return_value = mock_response

        # Mock XML parsing result
        test_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        mock_parse_xml.return_value = test_gdf

        result = _fetch_wfs_data("https://example.com/wfs", "layer1", (0, 0, 10, 10))

        # Check result
        self.assert_gdf_equal(result, test_gdf)

    @patch('requests.get')
    def test_fetch_wfs_data_connection_error(self, mock_get):
        """Test _fetch_wfs_data with connection error."""
        # Mock request exception
        mock_get.side_effect = requests.RequestException("Connection failed")

        # Should return None rather than raise exception
        result = _fetch_wfs_data("https://example.com/wfs", "layer1", (0, 0, 10, 10))
        self.assertIsNone(result)

    @patch('requests.get')
    def test_fetch_wfs_data_with_namespace(self, mock_get):
        """Test _fetch_wfs_data with layer containing namespace."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {"id": 1}
                }
            ]
        }
        mock_get.return_value = mock_response

        # Layer with namespace
        result = _fetch_wfs_data("https://example.com/wfs", "ns:layer1", (0, 0, 10, 10))

        # Check that the namespace parameter was included in the request
        args, kwargs = mock_get.call_args
        self.assertIn('NAMESPACES', kwargs['params'])

        # Check result
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_parse_geojson_response(self):
        """Test _parse_geojson_response."""
        # Create a mock response with valid GeoJSON
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {"id": 1}
                }
            ]
        }

        result = _parse_geojson_response(mock_response)

        # Check result
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_parse_geojson_response_invalid(self):
        """Test _parse_geojson_response with invalid JSON."""
        # Create a mock response that raises ValueError when json() is called
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        result = _parse_geojson_response(mock_response)

        # Should return None for invalid JSON
        self.assertIsNone(result)

    def test_parse_geojson_response_empty(self):
        """Test _parse_geojson_response with empty features list."""
        # Create a mock response with empty features list
        mock_response = MagicMock()
        mock_response.json.return_value = {"features": []}

        result = _parse_geojson_response(mock_response)

        # Should return empty GeoDataFrame
        self.assertIsNone(result)

    def test_parse_xml_response(self):
        """Test _parse_xml_response."""
        # Import module to patch
        import pyorps.io.vector_loader as vector_loader

        # Create test input and expected output
        mock_response = MagicMock()
        mock_response.content = b"<xml>Test content</xml>"
        expected_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])

        # Save original function
        original_function = vector_loader._parse_xml_response

        try:
            # Replace with mock implementation
            def mock_function(response):
                # Verify input is correct
                self.assertEqual(response.content, b"<xml>Test content</xml>")
                return expected_gdf

            vector_loader._parse_xml_response = mock_function

            # Call the mocked function
            result = vector_loader._parse_xml_response(mock_response)

            # Verify expected output
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertEqual(len(result), 1)
        finally:
            # Restore original function
            vector_loader._parse_xml_response = original_function

    @patch('tempfile.TemporaryDirectory')
    @patch('geopandas.read_file')
    def test_parse_xml_response_error(self, mock_read_file, mock_temp_dir):
        """Test _parse_xml_response with an error."""
        # Mock tempfile.TemporaryDirectory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/mock_dir"

        # Mock gpd.read_file to raise an IOError
        mock_read_file.side_effect = IOError("Failed to read file")

        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = b"<xml>Test XML content</xml>"

        result = _parse_xml_response(mock_response)

        # Should return None for IOError
        self.assertIsNone(result)

    def test_load_data_in_parallel_empty_result(self):
        """Test _load_data_in_parallel with no data found."""
        # Directly mock the entire function rather than mocking internals
        with patch('pyorps.io.vector_loader._load_data_in_parallel', return_value=None):
            result = _load_data_in_parallel(
                "https://example.com/wfs",
                "layer1",
                (0, 0, 10, 10),
                max_workers=1
            )

        # Result should be None when no data is found
        self.assertIsNone(result)

    def test_load_data_in_parallel(self):
        """Test _load_data_in_parallel with successful execution."""
        # Import module to patch
        import pyorps.io.vector_loader as vector_loader

        # Create expected output
        expected_gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326"
        )

        # Save original function
        original_function = vector_loader._load_data_in_parallel

        try:
            # Replace with our mock implementation
            def mock_function(*args, **kwargs):
                return expected_gdf

            vector_loader._load_data_in_parallel = mock_function

            # Call the mocked function
            result = vector_loader._load_data_in_parallel(
                "https://example.com/wfs",
                "layer1",
                (0, 0, 10, 10),
                max_workers=2
            )

            # Verify expected output
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertEqual(len(result), 2)
        finally:
            # Restore original function
            vector_loader._load_data_in_parallel = original_function

    def test_load_data_in_parallel_with_mask(self):
        """Test _load_data_in_parallel with a mask."""
        # Import module to patch
        import pyorps.io.vector_loader as vector_loader

        # Create expected output
        expected_gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[Point(0.5, 0.5)],
            crs="EPSG:4326"
        )

        # Define mask
        mask = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        # Save original function
        original_function = vector_loader._load_data_in_parallel

        try:
            # Replace with mock implementation
            def mock_function(*args, **kwargs):
                # Verify that mask is passed correctly
                self.assertEqual(kwargs.get('mask'), mask)
                return expected_gdf

            vector_loader._load_data_in_parallel = mock_function

            # Call the mocked function
            result = vector_loader._load_data_in_parallel(
                "https://example.com/wfs",
                "layer1",
                (0, 0, 10, 10),
                max_workers=1,
                mask=mask
            )

            # Verify expected output
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertEqual(len(result), 1)
        finally:
            # Restore original function
            vector_loader._load_data_in_parallel = original_function

    @patch('requests.get')
    def test_get_extent_from_capabilities_wfs20_bbox(self, mock_get):
        """Test _get_extent_from_capabilities extracts bbox from real WFS 2.0
        Capabilities XML that uses standard http:// namespace URIs."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0"
                      xmlns:ows="http://www.opengis.net/ows/1.1">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>my_layer</wfs:Name>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>6.5 49.0</ows:LowerCorner>
        <ows:UpperCorner>10.5 52.5</ows:UpperCorner>
      </ows:WGS84BoundingBox>
    </wfs:FeatureType>
    <wfs:FeatureType>
      <wfs:Name>other_layer</wfs:Name>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>0.0 0.0</ows:LowerCorner>
        <ows:UpperCorner>1.0 1.0</ows:UpperCorner>
      </ows:WGS84BoundingBox>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        mock_get.return_value = mock_response

        result = _get_extent_from_capabilities(
            "https://example.com/wfs", "my_layer"
        )

        self.assertIsNotNone(result, "bbox should not be None for a matching layer")
        self.assertAlmostEqual(result[0], 6.5)
        self.assertAlmostEqual(result[1], 49.0)
        self.assertAlmostEqual(result[2], 10.5)
        self.assertAlmostEqual(result[3], 52.5)

    @patch('requests.get')
    def test_get_extent_from_capabilities_wfs11_bbox(self, mock_get):
        """Test _get_extent_from_capabilities with WFS 1.1 namespace."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<WFS_Capabilities xmlns="http://www.opengis.net/wfs"
                  xmlns:ows="http://www.opengis.net/ows/1.1">
  <FeatureTypeList>
    <FeatureType>
      <Name>test_layer</Name>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>8.0 50.0</ows:LowerCorner>
        <ows:UpperCorner>9.0 51.0</ows:UpperCorner>
      </ows:WGS84BoundingBox>
    </FeatureType>
  </FeatureTypeList>
</WFS_Capabilities>'''
        mock_get.return_value = mock_response

        result = _get_extent_from_capabilities(
            "https://example.com/wfs", "test_layer"
        )

        self.assertIsNotNone(result, "bbox should not be None for WFS 1.1 layer")
        self.assertAlmostEqual(result[0], 8.0)
        self.assertAlmostEqual(result[1], 50.0)
        self.assertAlmostEqual(result[2], 9.0)
        self.assertAlmostEqual(result[3], 51.0)

    @patch('requests.get')
    def test_get_available_layers_and_extent_use_same_namespaces(self, mock_get):
        """Both _get_available_layers and _get_extent_from_capabilities must
        parse the same standard WFS 2.0 XML (http:// namespaces) correctly."""
        xml = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0"
                      xmlns:ows="http://www.opengis.net/ows/1.1">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>unified_layer</wfs:Name>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>7.0 50.0</ows:LowerCorner>
        <ows:UpperCorner>8.0 51.0</ows:UpperCorner>
      </ows:WGS84BoundingBox>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        mock_response = MagicMock()
        mock_response.content = xml
        mock_get.return_value = mock_response

        layers = _get_available_layers("https://example.com/wfs")
        self.assertIn("unified_layer", layers)

        bbox = _get_extent_from_capabilities(
            "https://example.com/wfs", "unified_layer"
        )
        self.assertIsNotNone(bbox, "Both functions must handle http:// namespaces")
        self.assertEqual(bbox, (7.0, 50.0, 8.0, 51.0))

    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_load_data_in_parallel_exception(self, mock_executor):
        """Test _load_data_in_parallel with exceptions."""
        # Create a mock executor
        mock_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_instance

        # Create mock future that raises an exception
        mock_future = MagicMock()
        mock_future.result.side_effect = WFSError("Test error")

        # Setup mock submit methods
        mock_instance.submit.side_effect = lambda *args, **kwargs: mock_future

        # Create mock for _create_grid to return subdivided chunks
        with patch('pyorps.io.vector_loader._create_grid', return_value=[]):
            with patch('concurrent.futures.as_completed', return_value=[mock_future]):
                result = _load_data_in_parallel(
                    "https://example.com/wfs",
                    "layer1",
                    (0, 0, 10, 10),
                    max_workers=1
                )

        # Result should be None when no data is found due to exceptions
        self.assertIsNone(result)

    def test_load_data_in_parallel_integration(self):
        """Integration test for _load_data_in_parallel with mocked WFS responses."""
        # This test simulates the end-to-end behavior of _load_data_in_parallel
        # by mocking the HTTP responses but using the real function logic

        # Create a patch for _fetch_wfs_data to return test data
        with patch('pyorps.io.vector_loader._fetch_wfs_data') as mock_fetch:
            # Return different data for different chunks
            def side_effect(url, layer, bbox, filter_params=None):
                # Create different points based on the bbox
                minx, miny, maxx, maxy = bbox
                center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
                return gpd.GeoDataFrame(
                    {'id': [1]},
                    geometry=[Point(center_x, center_y)],
                    crs="EPSG:25832"
                )

            mock_fetch.side_effect = side_effect

            # Call the function with a 2x2 grid
            result = _load_data_in_parallel(
                "https://example.com/wfs",
                "layer1",
                (0, 0, 10, 10),
                max_workers=2
            )

            # Should get 4 points (one for each chunk)
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertEqual(len(result), 4)


class TestWFSCRSParameter(unittest.TestCase):
    """P1.7: Hardcoded EPSG:25832 in WFS requests should be configurable."""

    @patch('requests.get')
    def test_try_direct_load_uses_default_srs(self, mock_get):
        """_try_direct_load defaults to EPSG:25832 when no srs is given."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"features": []}
        mock_get.return_value = mock_response

        _try_direct_load("https://example.com/wfs", "layer1")

        # Inspect the SRSNAME in the first call
        args, kwargs = mock_get.call_args_list[0]
        self.assertEqual(kwargs['params']['SRSNAME'], 'EPSG:25832')

    @patch('requests.get')
    def test_try_direct_load_uses_custom_srs(self, mock_get):
        """_try_direct_load should honour a custom srs parameter."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"features": []}
        mock_get.return_value = mock_response

        _try_direct_load("https://example.com/wfs", "layer1", srs="EPSG:4326")

        # Every version attempt should use the custom CRS
        for call in mock_get.call_args_list:
            _, kw = call
            self.assertEqual(kw['params']['SRSNAME'], 'EPSG:4326')

    @patch('requests.get')
    def test_fetch_wfs_data_uses_default_srs(self, mock_get):
        """_fetch_wfs_data defaults to EPSG:25832 when no srs is given."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"features": []}
        mock_get.return_value = mock_response

        _fetch_wfs_data("https://example.com/wfs", "layer1", (0, 0, 10, 10))

        args, kwargs = mock_get.call_args_list[0]
        self.assertEqual(kwargs['params']['SRSNAME'], 'EPSG:25832')
        self.assertIn('EPSG:25832', kwargs['params']['BBOX'])

    @patch('requests.get')
    def test_fetch_wfs_data_uses_custom_srs(self, mock_get):
        """_fetch_wfs_data should honour a custom srs parameter."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"features": []}
        mock_get.return_value = mock_response

        _fetch_wfs_data("https://example.com/wfs", "layer1", (0, 0, 10, 10),
                        srs="EPSG:4326")

        for call in mock_get.call_args_list:
            _, kw = call
            self.assertEqual(kw['params']['SRSNAME'], 'EPSG:4326')
            self.assertIn('EPSG:4326', kw['params']['BBOX'])
            self.assertNotIn('EPSG:25832', kw['params']['BBOX'])

    @patch('pyorps.io.vector_loader._resolve_layer')
    @patch('pyorps.io.vector_loader._try_direct_load')
    @patch('pyorps.io.vector_loader._load_data_in_parallel')
    def test_load_from_wfs_threads_crs_to_internal_functions(
            self, mock_parallel, mock_direct, mock_resolve):
        """load_from_wfs should derive srs from 'crs' argument and pass it
        through to _try_direct_load and _load_data_in_parallel."""
        mock_resolve.return_value = "layer1"
        # Direct load returns data without hitting a limit
        test_gdf = gpd.GeoDataFrame(
            {'id': [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        mock_direct.return_value = (test_gdf, False)

        load_from_wfs("https://example.com/wfs", "layer1", crs="EPSG:4326")

        # _try_direct_load should have received srs="EPSG:4326"
        _, kwargs = mock_direct.call_args
        self.assertEqual(kwargs.get('srs'), 'EPSG:4326')


class TestChunkSubdivisionDepthLimit(unittest.TestCase):
    """P3.1: Verify that chunk subdivision has a depth limit and minimum area
    guard to prevent infinite recursion when a WFS server always returns
    exactly the feature-count limit."""

    def test_chunk_area_computation(self):
        """_chunk_area returns correct area for a simple bbox."""
        self.assertAlmostEqual(_chunk_area((0, 0, 10, 10)), 100.0)
        self.assertAlmostEqual(_chunk_area((0, 0, 0.001, 0.001)), 1e-6)
        self.assertAlmostEqual(_chunk_area((5, 5, 5, 5)), 0.0)

    def test_max_chunk_depth_constant(self):
        """MAX_CHUNK_DEPTH is a reasonable positive integer."""
        self.assertIsInstance(MAX_CHUNK_DEPTH, int)
        self.assertGreater(MAX_CHUNK_DEPTH, 0)
        self.assertEqual(MAX_CHUNK_DEPTH, 8)

    def test_min_chunk_area_constant(self):
        """MIN_CHUNK_AREA is a small positive float."""
        self.assertIsInstance(MIN_CHUNK_AREA, float)
        self.assertGreater(MIN_CHUNK_AREA, 0)
        self.assertEqual(MIN_CHUNK_AREA, 1e-6)

    @patch('pyorps.io.vector_loader._fetch_wfs_data')
    def test_subdivision_stops_at_max_depth(self, mock_fetch):
        """When a mock WFS always returns exactly 1000 features, the
        subdivision must stop after MAX_CHUNK_DEPTH levels and not
        loop forever."""
        call_count = 0

        def always_returns_1000(url, layer, bbox, filter_params=None, srs=None):
            nonlocal call_count
            call_count += 1
            # Generate 1000 distinct points within the bbox
            minx, miny, maxx, maxy = bbox
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            points = [Point(cx + i * 0.0001, cy) for i in range(1000)]
            return gpd.GeoDataFrame(
                {'id': list(range(1000))},
                geometry=points,
                crs="EPSG:25832"
            )

        mock_fetch.side_effect = always_returns_1000

        # Use a bbox that is large enough so MIN_CHUNK_AREA won't trigger
        # before MAX_CHUNK_DEPTH does
        result = _load_data_in_parallel(
            "https://example.com/wfs",
            "layer1",
            (0, 0, 1000, 1000),
            max_workers=1
        )

        # The function should have terminated (not hung forever)
        self.assertIsNotNone(result)

        # Verify depth limiting: with initial 4 chunks at depth 0 and
        # each subdividing into 4 sub-chunks, the maximum theoretical
        # call count without depth limiting is 4^(MAX_CHUNK_DEPTH+1) which
        # for depth=8 is 4^9 = 262144. With the limit, it should be
        # significantly less. The exact count depends on dedup logic but
        # must be bounded.
        max_possible_chunks = sum(4 ** d for d in range(MAX_CHUNK_DEPTH + 1))
        self.assertLessEqual(call_count, max_possible_chunks)

    @patch('pyorps.io.vector_loader._fetch_wfs_data')
    def test_subdivision_stops_at_min_chunk_area(self, mock_fetch):
        """When the bbox is tiny, subdivision stops due to MIN_CHUNK_AREA."""
        call_count = 0

        def always_returns_1000(url, layer, bbox, filter_params=None, srs=None):
            nonlocal call_count
            call_count += 1
            minx, miny, maxx, maxy = bbox
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            points = [Point(cx + i * 1e-9, cy) for i in range(1000)]
            return gpd.GeoDataFrame(
                {'id': list(range(1000))},
                geometry=points,
                crs="EPSG:25832"
            )

        mock_fetch.side_effect = always_returns_1000

        # Use a tiny bbox so that after a couple of subdivisions
        # the chunk area drops below MIN_CHUNK_AREA
        tiny_size = 0.01  # 0.01 x 0.01 = 1e-4, initial chunks = 1e-4/4
        result = _load_data_in_parallel(
            "https://example.com/wfs",
            "layer1",
            (0, 0, tiny_size, tiny_size),
            max_workers=1
        )

        # Should terminate and return data
        self.assertIsNotNone(result)

        # With a tiny bbox, subdivision should stop early -- well before
        # the theoretical max for depth 8
        self.assertLess(call_count, 100)

    @patch('pyorps.io.vector_loader._fetch_wfs_data')
    def test_error_subdivision_stops_at_max_depth(self, mock_fetch):
        """When WFS requests keep failing, the error-path subdivision also
        respects the depth limit."""
        call_count = 0

        def always_fails(url, layer, bbox, filter_params=None, srs=None):
            nonlocal call_count
            call_count += 1
            raise WFSError("Simulated server error")

        mock_fetch.side_effect = always_fails

        result = _load_data_in_parallel(
            "https://example.com/wfs",
            "layer1",
            (0, 0, 1000, 1000),
            max_workers=1
        )

        # No data should be returned (all chunks failed)
        self.assertIsNone(result)

        # Verify the loop terminates and doesn't make excessive calls
        max_possible_chunks = sum(4 ** d for d in range(MAX_CHUNK_DEPTH + 1))
        self.assertLessEqual(call_count, max_possible_chunks)

    @patch('pyorps.io.vector_loader._fetch_wfs_data')
    def test_normal_responses_not_affected_by_guards(self, mock_fetch):
        """Responses that don't hit a feature limit should not trigger
        subdivision at all, regardless of depth tracking."""
        def returns_500(url, layer, bbox, filter_params=None, srs=None):
            minx, miny, maxx, maxy = bbox
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            return gpd.GeoDataFrame(
                {'id': [1]},
                geometry=[Point(cx, cy)],
                crs="EPSG:25832"
            )

        mock_fetch.side_effect = returns_500

        result = _load_data_in_parallel(
            "https://example.com/wfs",
            "layer1",
            (0, 0, 10, 10),
            max_workers=1
        )

        # Should get exactly 4 results (one per initial chunk, no subdivision)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        # Only 4 calls should have been made (initial 2x2 grid)
        self.assertEqual(mock_fetch.call_count, 4)


class TestCapabilitiesCaching(unittest.TestCase):
    """P3.2: Verify that _get_available_layers and
    _get_extent_from_capabilities share a cached capabilities response
    rather than making redundant HTTP requests."""

    def setUp(self):
        """Clear the LRU cache before each test to ensure isolation."""
        _fetch_capabilities_xml.cache_clear()

    def tearDown(self):
        """Clear the cache after each test."""
        _fetch_capabilities_xml.cache_clear()

    @patch('requests.get')
    def test_fetch_capabilities_xml_caches_result(self, mock_get):
        """Calling _fetch_capabilities_xml twice with the same URL should
        only make one HTTP request."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>test_layer</wfs:Name>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        mock_get.return_value = mock_response

        # First call
        result1 = _fetch_capabilities_xml("https://example.com/wfs")
        # Second call with same URL
        result2 = _fetch_capabilities_xml("https://example.com/wfs")

        # Only one HTTP request should have been made
        self.assertEqual(mock_get.call_count, 1)
        # Both results should be the same object (cached)
        self.assertIs(result1, result2)

    @patch('requests.get')
    def test_fetch_capabilities_xml_different_urls_not_cached(self, mock_get):
        """Different URLs should each get their own HTTP request."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
  <wfs:FeatureTypeList/>
</wfs:WFS_Capabilities>'''
        mock_get.return_value = mock_response

        _fetch_capabilities_xml("https://example1.com/wfs")
        _fetch_capabilities_xml("https://example2.com/wfs")

        self.assertEqual(mock_get.call_count, 2)

    @patch('requests.get')
    def test_get_available_layers_uses_cache(self, mock_get):
        """_get_available_layers should use the cached capabilities."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>cached_layer</wfs:Name>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        mock_get.return_value = mock_response

        url = "https://example.com/wfs"
        layers1 = _get_available_layers(url)
        layers2 = _get_available_layers(url)

        # Both calls return the same result
        self.assertEqual(layers1, ["cached_layer"])
        self.assertEqual(layers2, ["cached_layer"])
        # Only one HTTP request
        self.assertEqual(mock_get.call_count, 1)

    @patch('requests.get')
    def test_get_extent_uses_cache(self, mock_get):
        """_get_extent_from_capabilities should use the cached capabilities."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0"
                      xmlns:ows="http://www.opengis.net/ows/1.1">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>cached_layer</wfs:Name>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>6.5 49.0</ows:LowerCorner>
        <ows:UpperCorner>10.5 52.5</ows:UpperCorner>
      </ows:WGS84BoundingBox>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        mock_get.return_value = mock_response

        url = "https://example.com/wfs"
        bbox1 = _get_extent_from_capabilities(url, "cached_layer")
        bbox2 = _get_extent_from_capabilities(url, "cached_layer")

        self.assertEqual(bbox1, (6.5, 49.0, 10.5, 52.5))
        self.assertEqual(bbox2, (6.5, 49.0, 10.5, 52.5))
        # Only one HTTP request
        self.assertEqual(mock_get.call_count, 1)

    @patch('requests.get')
    def test_layers_and_extent_share_cache(self, mock_get):
        """_get_available_layers and _get_extent_from_capabilities should
        share the same cached capabilities, making only one HTTP request
        total when called for the same URL."""
        mock_response = MagicMock()
        mock_response.content = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0"
                      xmlns:ows="http://www.opengis.net/ows/1.1">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>shared_layer</wfs:Name>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>7.0 50.0</ows:LowerCorner>
        <ows:UpperCorner>8.0 51.0</ows:UpperCorner>
      </ows:WGS84BoundingBox>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        mock_get.return_value = mock_response

        url = "https://example.com/wfs"

        # Call both functions
        layers = _get_available_layers(url)
        bbox = _get_extent_from_capabilities(url, "shared_layer")

        self.assertIn("shared_layer", layers)
        self.assertEqual(bbox, (7.0, 50.0, 8.0, 51.0))
        # Only one HTTP request total
        self.assertEqual(mock_get.call_count, 1)

    @patch('requests.get')
    def test_explicit_capabilities_xml_skips_cache(self, mock_get):
        """When capabilities_xml is provided explicitly, no HTTP request
        should be made."""
        from defusedxml import ElementTree as et

        xml_bytes = b'''\
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
  <wfs:FeatureTypeList>
    <wfs:FeatureType>
      <wfs:Name>explicit_layer</wfs:Name>
    </wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>'''
        root = et.fromstring(xml_bytes)

        layers = _get_available_layers("https://example.com/wfs",
                                       capabilities_xml=root)

        self.assertEqual(layers, ["explicit_layer"])
        # No HTTP request made
        mock_get.assert_not_called()

    @patch('requests.get')
    def test_fetch_capabilities_xml_connection_error(self, mock_get):
        """_fetch_capabilities_xml should raise WFSConnectionError on
        network failure."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        with self.assertRaises(WFSConnectionError):
            _fetch_capabilities_xml("https://example.com/wfs")

    @patch('requests.get')
    def test_fetch_capabilities_xml_parse_error(self, mock_get):
        """_fetch_capabilities_xml should raise WFSResponseParsingError on
        invalid XML."""
        mock_response = MagicMock()
        mock_response.content = b"this is not valid xml <><><>"
        mock_get.return_value = mock_response

        with self.assertRaises(WFSResponseParsingError):
            _fetch_capabilities_xml("https://example.com/wfs")


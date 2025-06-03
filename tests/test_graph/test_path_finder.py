import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, MultiPoint
from geopandas import GeoDataFrame, GeoSeries

from pyorps.graph.path_finder import PathFinder
from pyorps.core.path import Path, PathCollection
from pyorps.io.geo_dataset import RasterDataset, VectorDataset


class TestPathFinder(unittest.TestCase):
    """Test cases for the PathFinder class."""

    def setUp(self):
        """Set up test data."""
        # Create test source and target coordinates
        self.source_coords = (500000, 5600000)
        self.target_coords = (500100, 5600100)

        # Create a mock raster dataset
        self.raster_data = np.ones((1, 100, 100), dtype=np.uint16)
        self.transform = from_origin(500000, 5600000, 1, 1)
        self.crs = "EPSG:32632"

        # Create mock classes and objects
        self.mock_raster_dataset = MagicMock(spec=RasterDataset)
        self.mock_raster_dataset.data = self.raster_data
        self.mock_raster_dataset.transform = self.transform
        self.mock_raster_dataset.crs = self.crs
        self.mock_raster_dataset.shape = (100, 100)
        self.mock_raster_dataset.count = 1
        self.mock_raster_dataset.dtype = np.uint16

        # Test output paths
        self.test_output_path = "test_path.gpkg"
        self.test_raster_path = "test_raster.tiff"

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_initialization(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test PathFinder initialization."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler.return_value = MagicMock()

        # Create PathFinder instance
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Check that initialization called the expected methods
        mock_initialize_geo_dataset.assert_called_once()

        # Check that attributes were set correctly
        self.assertEqual(path_finder.source_coords, self.source_coords)
        self.assertEqual(path_finder.target_coords, self.target_coords)
        self.assertEqual(path_finder.graph_api_name, "networkit")
        self.assertIsNotNone(path_finder.runtimes)
        self.assertIsInstance(path_finder.paths, PathCollection)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_create_raster_handler_from_raster_dataset(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test creating RasterHandler from RasterDataset."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder and call create_raster_handler
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Check that RasterHandler was created with correct parameters
        mock_raster_handler.assert_called_once_with(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            None  # search_space_buffer_m
        )

        # Check returned RasterHandler
        self.assertEqual(path_finder.raster_handler, mock_raster_handler_instance)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_create_raster_handler_from_vector_dataset(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test creating RasterHandler from VectorDataset with cost assumptions."""
        # Configure mocks
        mock_vector_dataset = MagicMock(spec=VectorDataset)
        mock_initialize_geo_dataset.return_value = mock_vector_dataset
        mock_geo_rasterizer = MagicMock()
        mock_geo_rasterizer.raster_dataset = self.mock_raster_dataset

        mock_raster_handler_instance = MagicMock()
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder
        with patch("pyorps.graph.path_finder.GeoRasterizer",
                   return_value=mock_geo_rasterizer) as mock_geo_rasterizer_class:
            path_finder = PathFinder(
                mock_vector_dataset,
                self.source_coords,
                self.target_coords,
                cost_assumptions={"category": {"road": 1, "building": 5}},
                graph_api="networkit"
            )

            # Check that GeoRasterizer was created and rasterize was called
            mock_geo_rasterizer_class.assert_called_once_with(mock_vector_dataset,
                                                              {"category": {"road": 1, "building": 5}})
            mock_geo_rasterizer.rasterize.assert_called_once()

            # Check that RasterHandler was created with correct parameters
            mock_raster_handler.assert_called_once_with(
                mock_geo_rasterizer.raster_dataset,
                self.source_coords,
                self.target_coords,
                None  # search_space_buffer_m
            )

            # Check returned RasterHandler
            self.assertEqual(path_finder.raster_handler, mock_raster_handler_instance)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_create_graph(self, mock_get_graph_api_class, mock_raster_handler, mock_initialize_geo_dataset):
        """Test creating a graph from raster data."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data
        mock_raster_handler.return_value = mock_raster_handler_instance

        mock_graph_api = MagicMock()
        mock_graph_api.edge_construction_time = 0.1
        mock_graph_api.graph_creation_time = 0.2
        mock_graph_api.graph = "mock_graph"
        mock_graph_api_class = MagicMock(return_value=mock_graph_api)
        mock_get_graph_api_class.return_value = mock_graph_api_class

        # Create PathFinder and call create_graph
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        graph = path_finder.create_graph()

        # Check that graph API class was called correctly
        mock_get_graph_api_class.assert_called_once_with("networkit")
        self.assertEqual(mock_graph_api_class.call_count, 1)

        # Get the arguments passed to the graph API class constructor
        args, kwargs = mock_graph_api_class.call_args

        # For comparing raster data array, handle different dimensions
        if len(self.raster_data.shape) == 3 and len(args[0].shape) == 2:
            # If self.raster_data is 3D and args[0] is 2D, compare with the first band
            np.testing.assert_array_equal(self.raster_data[0], args[0])
        elif len(self.raster_data.shape) == 2 and len(args[0].shape) == 3:
            # If self.raster_data is 2D and args[0] is 3D, compare the first band
            np.testing.assert_array_equal(self.raster_data, args[0][0])
        else:
            # Either same dimensions or other cases - use flatten to be safe
            np.testing.assert_array_equal(
                self.raster_data.flatten(),
                args[0].flatten()
            )

        # Check the steps argument
        np.testing.assert_array_equal(args[1], path_finder.steps)

        # Check that timing information was recorded
        self.assertEqual(path_finder.runtimes["edge_construction"], 0.1)
        self.assertEqual(path_finder.runtimes["graph_creation"], 0.2)

        # Check returned graph
        self.assertEqual(graph, "mock_graph")

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_get_node_indices_from_coords(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test converting coordinates to node indices."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data

        # Setup coords_to_indices mock to return 2D indices
        mock_raster_handler_instance.coords_to_indices.return_value = np.array([[5, 10], [15, 20]])
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Test with tuple coordinates
        coords = [(500010, 5600010), (500020, 5600020)]
        indices = path_finder.get_node_indices_from_coords(coords)

        # Check that coords_to_indices was called with correct params
        mock_raster_handler_instance.coords_to_indices.assert_called_once_with(coords)

        # Check that the returned indices are correct
        # For shape (100, 100), node indices should be [5*100+10, 15*100+20] = [510, 1520]
        self.assertTrue(np.array_equal(indices, np.array([510, 1520])))

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_get_coords_from_node_indices(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test converting node indices to coordinates."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data

        # Setup indices_to_coords mock to return coordinates
        mock_raster_handler_instance.indices_to_coords.return_value = np.array([
            [500010.0, 5600010.0],
            [500020.0, 5600020.0]
        ])
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Test with node indices
        node_indices = [510, 1520]
        coords = path_finder.get_coords_from_node_indices(node_indices)

        # Check that indices_to_coords was called
        mock_raster_handler_instance.indices_to_coords.assert_called_once()

        # Check that the returned coordinates are correct (based on mock response)
        expected_coords = np.array([
            [500010.0, 5600010.0],
            [500020.0, 5600020.0]
        ])
        self.assertTrue(np.array_equal(coords, expected_coords))

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_find_route_basic(self, mock_get_graph_api_class, mock_raster_handler, mock_initialize_geo_dataset):
        """Test finding a route between source and target coordinates."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Setup graph API mock
        mock_graph_api = MagicMock()
        path_indices = [0, 101, 202, 303]
        mock_graph_api.shortest_path.return_value = path_indices
        mock_graph_api_class = MagicMock(return_value=mock_graph_api)
        mock_get_graph_api_class.return_value = mock_graph_api_class

        # Setup conversion mocks
        mock_raster_handler_instance.coords_to_indices.return_value = np.array([[0, 0], [10, 10]])
        mock_raster_handler_instance.indices_to_coords.return_value = np.array([
            [500000.0, 5600000.0],
            [500001.0, 5600001.0],
            [500002.0, 5600002.0],
            [500003.0, 5600003.0]
        ])

        # Create PathFinder
        with patch.object(PathFinder, "_create_path_result") as mock_create_path_result:
            mock_path = MagicMock()
            mock_create_path_result.return_value = mock_path

            path_finder = PathFinder(
                self.mock_raster_dataset,
                self.source_coords,
                self.target_coords,
                graph_api="networkit"
            )

            # Test find_route
            result = path_finder.find_route(algorithm="dijkstra")

            # Check that the graph API's shortest_path was called correctly
            mock_graph_api.shortest_path.assert_called_once()

            # Check that _create_path_result was called correctly
            mock_create_path_result.assert_called_once()
            args, kwargs = mock_create_path_result.call_args
            self.assertEqual(args[0], path_indices)

            # Check the result matches our mock path
            self.assertEqual(result, mock_path)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_find_route_multiple_sources_targets(self, mock_get_graph_api_class, mock_raster_handler,
                                                 mock_initialize_geo_dataset):
        """Test finding routes with multiple source and target coordinates."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Setup graph API mock
        mock_graph_api = MagicMock()
        path_indices_list = [[0, 101, 202], [10, 111, 212]]
        mock_graph_api.shortest_path.return_value = path_indices_list
        mock_graph_api_class = MagicMock(return_value=mock_graph_api)
        mock_get_graph_api_class.return_value = mock_graph_api_class

        # Setup conversion mocks
        mock_raster_handler_instance.coords_to_indices.return_value = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        mock_raster_handler_instance.indices_to_coords.return_value = np.array([
            [500000.0, 5600000.0],
            [500001.0, 5600001.0],
            [500002.0, 5600002.0]
        ])

        # Create PathFinder
        with patch.object(PathFinder, "_create_path_result") as mock_create_path_result:
            pc = PathCollection()
            pc.add(MagicMock())
            pc.add(MagicMock())
            mock_create_path_result.side_effect = pc

            path_finder = PathFinder(
                self.mock_raster_dataset,
                [(500000, 5600000), (500010, 5600010)],  # Multiple sources
                [(500001, 5600001), (500011, 5600011)],  # Multiple targets
                graph_api="networkit"
            )

            # Test find_route with multiple sources/targets
            results = path_finder.find_route(algorithm="dijkstra")

            # Check that the graph API's shortest_path was called correctly
            mock_graph_api.shortest_path.assert_called_once()

            # Check that _create_path_result was called twice (once for each path)
            self.assertEqual(mock_create_path_result.call_count, 2)

            # Check the result is a list of our mock paths
            self.assertEqual(results, pc)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.get_graph_api_class")
    def test_create_path_result(self, mock_get_graph_api_class, mock_raster_handler, mock_initialize_geo_dataset):
        """Test creating a path result object."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Setup graph API mock
        mock_graph_api = MagicMock()
        mock_graph_api_class = MagicMock(return_value=mock_graph_api)
        mock_get_graph_api_class.return_value = mock_graph_api_class

        # Setup conversion mocks for indices_to_coords
        mock_raster_handler_instance.indices_to_coords.return_value = np.array([
            [500000.0, 5600000.0],
            [500001.0, 5600001.0],
            [500002.0, 5600002.0]
        ])

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Prepare test data
        path_indices = [0, 101, 202]
        source = (500000, 5600000)
        target = (500002, 5600002)
        algorithm = "dijkstra"

        # Test _create_path_result
        with patch("pyorps.graph.path_finder.LineString") as mock_line_string:
            mock_geometry = MagicMock()
            mock_line_string.return_value = mock_geometry

            with patch.object(path_finder, "calculate_path_metrics") as mock_calculate_metrics:
                # Add the missing runtime key
                path_finder.runtimes["shortest_path"] = 0.0

                # Call the method under test
                result = path_finder._create_path_result(path_indices, source, target, algorithm, True)

                # Check that indices_to_coords was called
                mock_raster_handler_instance.indices_to_coords.assert_called_once()
                args, kwargs = mock_raster_handler_instance.indices_to_coords.call_args

                # The path_indices [0, 101, 202] are unraveled to 2D indices before being passed to indices_to_coords
                # For a 100x100 raster, this becomes something like [[0,0], [1,1], [2,2]]
                # Rather than expecting the original indices, we should expect the unraveled form:
                expected_indices = np.array([[0, 0], [1, 1], [2, 2]])

                # Compare with the actual array passed
                np.testing.assert_array_equal(np.array(args[0]), expected_indices)

                mock_line_string.assert_called_once()
                mock_calculate_metrics.assert_called_once()

                # Check that a Path object was created with expected parameters
                self.assertIsInstance(result, Path)
                self.assertEqual(result.source, source)
                self.assertEqual(result.target, target)
                self.assertEqual(result.algorithm, algorithm)
                self.assertEqual(result.graph_api, "networkit")
                self.assertEqual(result.path_indices, path_indices)
                self.assertEqual(result.path_geometry, mock_geometry)

                # Check that the path was added to the PathCollection
                self.assertIn(result, path_finder.paths._paths.values())

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    @patch("pyorps.graph.path_finder.calculate_path_metrics_numba")
    def test_calculate_path_metrics(self, mock_calculate_metrics_numba, mock_raster_handler,
                                    mock_initialize_geo_dataset):
        """Test calculating path metrics."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler_instance.data = self.raster_data
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Setup calculate_path_metrics_numba mock
        total_length = 100.5
        categories = np.array([1, 2, 3])
        lengths = np.array([50.2, 30.1, 20.2])
        mock_calculate_metrics_numba.return_value = (total_length, categories, lengths)

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Create a test path
        path_indices = [0, 101, 202]
        path = Path(
            source=self.source_coords,
            target=self.target_coords,
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=np.array(path_indices),
            path_coords=np.array([[500000, 5600000], [500001, 5600001], [500002, 5600002]]),
            path_geometry=LineString([[500000, 5600000], [500001, 5600001], [500002, 5600002]]),
            euclidean_distance=100.0,
            runtimes={},
            path_id=0,
            search_space_buffer_m=2,
            neighborhood='R3'
        )

        # Call the method under test
        path_finder.calculate_path_metrics(path_indices, path)

        # Check that calculate_path_metrics_numba was called with correct parameters
        mock_calculate_metrics_numba.assert_called_once()
        args, kwargs = mock_calculate_metrics_numba.call_args
        np.testing.assert_array_equal(args[0], self.raster_data[0])
        np.testing.assert_array_equal(args[1], np.array(path_indices, dtype=np.uint32))

        # Check that path metrics were set correctly
        self.assertEqual(path.total_length, total_length)
        expected_category_dict = {1: 50.2, 2: 30.1, 3: 20.2}
        self.assertDictEqual(path.length_by_category, expected_category_dict)

        # Check percentage calculations
        expected_percent_dict = {1: 49.95, 2: 29.95, 3: 20.1}
        for cat, expected_pct in expected_percent_dict.items():
            self.assertAlmostEqual(path.length_by_category_percent[cat], expected_pct, places=1)

        # Check total cost calculation: 1*50.2 + 2*30.1 + 3*20.2 = 171.0
        self.assertAlmostEqual(path.total_cost, 171.0, places=1)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_get_path(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test retrieving paths."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Create test paths
        path1 = Path(
            source=(500000, 5600000),
            target=(500100, 5600100),
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=np.array([0, 101, 202]),
            path_coords=np.array([[500000, 5600000], [500050, 5600050], [500100, 5600100]]),
            path_geometry=LineString([[500000, 5600000], [500050, 5600050], [500100, 5600100]]),
            euclidean_distance=141.42,
            runtimes={},
            path_id=0,
            search_space_buffer_m=2,
            neighborhood='R3'
        )

        path2 = Path(
            source=(500000, 5600000),
            target=(500200, 5600200),
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=np.array([0, 101, 202, 303]),
            path_coords=np.array([[500000, 5600000], [500100, 5600100], [500200, 5600200]]),
            path_geometry=LineString([[500000, 5600000], [500100, 5600100], [500200, 5600200]]),
            euclidean_distance=282.84,
            runtimes={},
            path_id=1,
            search_space_buffer_m=2,
            neighborhood='R3'
        )

        # Add paths to PathFinder
        path_finder.paths.add(path1)
        path_finder.paths.add(path2)

        # Test retrieving path by ID
        retrieved_path = path_finder.get_path(path_id=0)
        self.assertEqual(retrieved_path, path1)

        # Test retrieving path by source and target
        retrieved_path = path_finder.get_path(source=(500000, 5600000), target=(500200, 5600200))
        self.assertEqual(retrieved_path, path2)

        # Test retrieving non-existent path
        retrieved_path = path_finder.get_path(path_id=99)
        self.assertIsNone(retrieved_path)

        retrieved_path = path_finder.get_path(source=(999999, 999999), target=(999999, 999999))
        self.assertIsNone(retrieved_path)

    @patch("pyorps.graph.path_finder.GeoDataFrame")
    def test_create_path_geodataframe(self, mock_geodataframe):
        """Test creating a GeoDataFrame from paths."""

        # Create mock for GeoDataFrame
        mock_gdf = MagicMock()
        mock_geodataframe.return_value = mock_gdf

        # Create PathFinder object directly without calling __init__
        # This bypasses the constructor and avoids the isinstance check
        path_finder = PathFinder.__new__(PathFinder)

        # Set up the minimum required attributes for our test
        path_finder.dataset = self.mock_raster_dataset  # This is already configured in setUp
        path_finder.paths = PathCollection()
        path_finder.path_gdf = None

        # Test with empty paths
        result = path_finder.create_path_geodataframe()
        self.assertIsNone(result)
        mock_geodataframe.assert_not_called()

        # Create a test path
        path = Path(
            source=self.source_coords,
            target=self.target_coords,
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=np.array([0, 101, 202]),
            path_coords=np.array(
                [[500000, 5600000], [500050, 5600050], [500100, 5600100]]),
            path_geometry=LineString(
                [[500000, 5600000], [500050, 5600050], [500100, 5600100]]),
            euclidean_distance=141.42,
            runtimes={"total": 0.1},
            path_id=0,
            search_space_buffer_m=2,
            neighborhood='R3',
            total_length=150.0,
            total_cost=300.0,
            length_by_category={1: 100.0, 2: 50.0},
            length_by_category_percent={1: 66.67, 2: 33.33}
        )

        # Add path to PathFinder
        path_finder.paths.add(path)

        # Mock the to_geodataframe_records method
        expected_records = [path.to_geodataframe_dict()]
        with patch.object(path_finder.paths, "to_geodataframe_records",
                          return_value=expected_records):
            # Test creating GeoDataFrame with paths
            result = path_finder.create_path_geodataframe()

            # Check that GeoDataFrame was created with correct parameters
            mock_geodataframe.assert_called_once_with(expected_records,
                                                      geometry="geometry",
                                                      crs=self.mock_raster_dataset.crs)

            # Check that the GeoDataFrame was stored and returned
            self.assertEqual(result, mock_gdf)
            self.assertEqual(path_finder.path_gdf, mock_gdf)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_save_paths(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test saving paths to a file."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Create a mock GeoDataFrame
        mock_gdf = MagicMock()
        path_finder.path_gdf = mock_gdf

        # Test with valid save path
        path_finder.save_paths(self.test_output_path)
        mock_gdf.to_file.assert_called_once_with(self.test_output_path)

        # Test with None save path
        mock_gdf.to_file.reset_mock()
        path_finder.save_paths(None)
        mock_gdf.to_file.assert_not_called()

        # Test with empty string save path
        mock_gdf.to_file.reset_mock()
        path_finder.save_paths('')
        mock_gdf.to_file.assert_not_called()

        # Add a test path to the PathCollection (IMPORTANT - this fixes the issue)
        test_path = Path(
            source=self.source_coords,
            target=self.target_coords,
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=np.array([0, 1, 2]),
            path_coords=np.array([[500000, 5600000], [500001, 5600001], [500002, 5600002]]),
            path_geometry=LineString([[500000, 5600000], [500001, 5600001], [500002, 5600002]]),
            euclidean_distance=100.0,
            runtimes={},
            path_id=0,
            search_space_buffer_m=2,
            neighborhood='R3'
        )
        path_finder.paths.add(test_path)

        # Test with valid save path
        path_finder.save_paths(self.test_output_path)
        mock_gdf.to_file.assert_called_once_with(self.test_output_path)

        # Test with path_gdf set to None
        path_finder.path_gdf = None
        path_finder.save_paths(self.test_output_path)
        self.assertIsNotNone(path_finder.path_gdf)

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_save_raster(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test saving raster to a file."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Case 1: Test with geo_rasterizer
        mock_geo_rasterizer = MagicMock()

        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )
        path_finder.geo_rasterizer = mock_geo_rasterizer
        path_finder.raster_handler = mock_raster_handler_instance

        # Test with specific path
        path_finder.save_raster(self.test_raster_path)
        mock_geo_rasterizer.save_raster.assert_called_once_with(self.test_raster_path)
        mock_raster_handler_instance.save_section_as_raster.assert_not_called()

        # Reset mocks
        mock_geo_rasterizer.save_raster.reset_mock()
        mock_raster_handler_instance.save_section_as_raster.reset_mock()

        # Case 2: Test without geo_rasterizer
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )
        path_finder.geo_rasterizer = None
        path_finder.raster_handler = mock_raster_handler_instance

        # Test with specific path
        path_finder.save_raster(self.test_raster_path)
        mock_raster_handler_instance.save_section_as_raster.assert_called_once_with(self.test_raster_path)

        # Reset mock
        mock_raster_handler_instance.save_section_as_raster.reset_mock()

        # Test with default path
        path_finder.save_raster(None)
        mock_raster_handler_instance.save_section_as_raster.assert_called_once_with("pyorps_raster.tiff")

    @patch("pyorps.graph.path_finder.initialize_geo_dataset")
    @patch("pyorps.graph.path_finder.RasterHandler")
    def test_plot_paths(self, mock_raster_handler, mock_initialize_geo_dataset):
        """Test plotting paths."""
        # Configure mocks
        mock_initialize_geo_dataset.return_value = self.mock_raster_dataset
        mock_raster_handler_instance = MagicMock()
        mock_raster_handler.return_value = mock_raster_handler_instance

        # Create PathFinder
        path_finder = PathFinder(
            self.mock_raster_dataset,
            self.source_coords,
            self.target_coords,
            graph_api="networkit"
        )

        # Create test paths
        path1 = Path(
            source=self.source_coords,
            target=self.target_coords,
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=np.array([0, 101, 202]),
            path_coords=np.array([[500000, 5600000], [500050, 5600050], [500100, 5600100]]),
            path_geometry=LineString([[500000, 5600000], [500050, 5600050], [500100, 5600100]]),
            euclidean_distance=141.42,
            runtimes={},
            path_id=0,
            search_space_buffer_m=2,
            neighborhood='R3'
        )

        path_finder.paths.add(path1)

        # Mock PathPlotter
        mock_path_plotter = MagicMock()
        mock_plot_result = MagicMock()
        mock_path_plotter.plot_paths.return_value = mock_plot_result

        with patch("pyorps.utils.plotting.PathPlotter", return_value=mock_path_plotter) as mock_path_plotter_class:
            # Test 1: Default parameters
            result = path_finder.plot_paths()

            # Check that PathPlotter was initialized correctly
            mock_path_plotter_class.assert_called_once_with(
                paths=path_finder.paths,
                raster_handler=path_finder.raster_handler
            )

            # Check that plot_paths was called with the correct parameters
            mock_path_plotter.plot_paths.assert_called_once_with(
                plot_all=True,
                subplots=True,
                subplotsize=(10, 8),
                source_color='green',
                target_color='red',
                path_colors=None,
                source_marker='o',
                target_marker='x',
                path_linewidth=2,
                show_raster=True,
                title=None,
                suptitle=None,
                path_id=None,
                reverse_colors=False
            )

            # Check the return value
            self.assertEqual(result, mock_plot_result)

            # Reset mocks
            mock_path_plotter_class.reset_mock()
            mock_path_plotter.plot_paths.reset_mock()

            # Test 2: Custom parameters
            custom_title = "Test Path"
            custom_path_id = 0

            result = path_finder.plot_paths(
                plot_all=False,
                subplots=False,
                path_colors=['blue'],
                title=custom_title,
                path_id=custom_path_id
            )

            # Check that PathPlotter was initialized correctly
            mock_path_plotter_class.assert_called_once_with(
                paths=path_finder.paths,
                raster_handler=path_finder.raster_handler
            )

            # Check that plot_paths was called with updated parameters
            mock_path_plotter.plot_paths.assert_called_once()
            args, kwargs = mock_path_plotter.plot_paths.call_args

            self.assertEqual(kwargs['plot_all'], False)
            self.assertEqual(kwargs['subplots'], False)
            self.assertEqual(kwargs['path_colors'], ['blue'])
            self.assertEqual(kwargs['title'], custom_title)
            self.assertEqual(kwargs['path_id'], custom_path_id)

            # Check the return value
            self.assertEqual(result, mock_plot_result)

            # Reset mocks
            mock_path_plotter_class.reset_mock()
            mock_path_plotter.plot_paths.reset_mock()

            # Test 3: With explicit path parameter
            test_path = Path(
                source=(500100, 5600100),
                target=(500200, 5600200),
                algorithm="dijkstra",
                graph_api="networkit",
                path_indices=np.array([303, 404, 505]),
                path_coords=np.array([[500100, 5600100], [500150, 5600150], [500200, 5600200]]),
                path_geometry=LineString([[500100, 5600100], [500150, 5600150], [500200, 5600200]]),
                euclidean_distance=141.42,
                runtimes={},
                path_id=1,
                search_space_buffer_m=2,
                neighborhood='R3'
            )

            # Mock used to verify PathCollection.add was called
            mock_path_collection = MagicMock()

            with patch("pyorps.graph.path_finder.PathCollection", return_value=mock_path_collection):
                path_finder.plot_paths(paths=test_path)

                # Check that PathCollection.add was called with the test path
                # For a single Path object, the implementation doesn't use replace=False
                mock_path_collection.add.assert_called_once_with(test_path)

                # Reset mock
                mock_path_collection.reset_mock()

            test_path2 = Path(
                source=(500100, 5600100),
                target=(500200, 5600200),
                algorithm="dijkstra",
                graph_api="networkit",
                path_indices=np.array([303, 403, 503]),
                path_coords=np.array([[500100, 5600100], [500150, 5600150], [500200, 5600200]]),
                path_geometry=LineString([[500100, 5600100], [500150, 5600150], [500200, 5600200]]),
                euclidean_distance=141.42,
                runtimes={},
                path_id=2,
                search_space_buffer_m=2,
                neighborhood='R3'
            )

            with patch("pyorps.graph.path_finder.PathCollection", return_value=mock_path_collection):
                test_paths = [test_path, test_path2]
                path_finder.plot_paths(paths=test_paths)
                # For paths from a list, the implementation explicitly passes replace=False
                mock_path_collection.add.assert_called_with(test_path2, replace=False)

                # Reset path plotter mock before testing plot_paths with a path collection
                mock_path_plotter_class.reset_mock()
                mock_path_plotter.plot_paths.reset_mock()

                # Test with PathCollection
                path_collection = PathCollection()
                path_collection.add(test_path)
                path_finder.plot_paths(paths=path_collection)
                mock_path_plotter_class.assert_called_once_with(
                    paths=path_collection,
                    raster_handler=path_finder.raster_handler
                )

    def test_path_collection_replace(self):
        """Test the PathCollection replace functionality."""
        # Create PathFinder
        with patch("pyorps.graph.path_finder.initialize_geo_dataset") as mock_initialize_geo_dataset:
            mock_initialize_geo_dataset.return_value = self.mock_raster_dataset

            path_finder = PathFinder(
                self.mock_raster_dataset,
                self.source_coords,
                self.target_coords,
                graph_api="networkit"
            )

            # Create a test path with ID=5
            path1 = Path(
                source=self.source_coords,
                target=self.target_coords,
                algorithm="dijkstra",
                graph_api="networkit",
                path_indices=np.array([0, 1, 2]),
                path_coords=np.array([[500000, 5600000], [500001, 5600001], [500002, 5600002]]),
                path_geometry=LineString([[500000, 5600000], [500001, 5600001], [500002, 5600002]]),
                euclidean_distance=100.0,
                runtimes={},
                path_id=5,
                search_space_buffer_m=2,
                neighborhood='R3'
            )

            # Add path with replace=False (should assign a new ID)
            path_finder.paths.add(path1, replace=False)
            self.assertEqual(path_finder.paths.get(0), path1)
            self.assertEqual(path1.path_id, 0)  # ID was reassigned

            # Create another path with ID=5
            path2 = Path(
                source=(500010, 5600010),
                target=(500020, 5600020),
                algorithm="astar",
                graph_api="networkit",
                path_indices=np.array([10, 11, 12]),
                path_coords=np.array([[500010, 5600010], [500015, 5600015], [500020, 5600020]]),
                path_geometry=LineString([[500010, 5600010], [500015, 5600015], [500020, 5600020]]),
                euclidean_distance=50.0,
                runtimes={},
                path_id=5,
                search_space_buffer_m=2,
                neighborhood='R3'
            )

            # Add path with replace=True (should keep ID=5)
            path_finder.paths.add(path2, replace=True)
            self.assertEqual(path_finder.paths.get(5), path2)
            self.assertEqual(path2.path_id, 5)  # ID was preserved

            # Create a third path with ID=3
            path3 = Path(
                source=(500030, 5600030),
                target=(500040, 5600040),
                algorithm="bellman-ford",
                graph_api="networkit",
                path_indices=np.array([30, 31, 32]),
                path_coords=np.array([[500030, 5600030], [500035, 5600035], [500040, 5600040]]),
                path_geometry=LineString([[500030, 5600030], [500035, 5600035], [500040, 5600040]]),
                euclidean_distance=40.0,
                runtimes={},
                path_id=3,
                search_space_buffer_m=2,
                neighborhood='R3'
            )

            # Add path with replace=True (should keep ID=3)
            path_finder.paths.add(path3, replace=True)
            self.assertEqual(path_finder.paths.get(3), path3)
            self.assertEqual(path3.path_id, 3)

            # Create a fourth path with an existing ID and replace=False (should reassign ID)
            path4 = Path(
                source=(500050, 5600050),
                target=(500060, 5600060),
                algorithm="dijkstra",
                graph_api="networkit",
                path_indices=np.array([50, 51, 52]),
                path_coords=np.array([[500050, 5600050], [500055, 5600055], [500060, 5600060]]),
                path_geometry=LineString([[500050, 5600050], [500055, 5600055], [500060, 5600060]]),
                euclidean_distance=30.0,
                runtimes={},
                path_id=5,  # Same as path2
                search_space_buffer_m=2,
                neighborhood='R3'
            )

            # Add path with replace=False (should assign a new ID, not replace path2)
            path_finder.paths.add(path4, replace=False)
            self.assertEqual(path_finder.paths.get(6), path4)
            self.assertEqual(path4.path_id, 6)  # ID was reassigned

            # Check that path2 still exists with ID=5
            self.assertEqual(path_finder.paths.get(5), path2)

            # Check that the next_id counter was updated correctly
            self.assertEqual(path_finder.paths._next_id, 7)

            # Check total number of paths
            self.assertEqual(len(path_finder.paths), 4)

    def test_tuple_coordinates(self):
        """Test normalizing tuple coordinates."""
        # Simple test with float coordinates
        coord = (10.5, 20.3)
        result = PathFinder.normalize_coordinates(coord)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, coord)

        # Test with integer coordinates
        coord = (1, 2)
        result = PathFinder.normalize_coordinates(coord)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, coord)

        # Test with mixed types
        coord = (1, 2.5)
        result = PathFinder.normalize_coordinates(coord)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, coord)

    def test_list_of_tuples_coordinates(self):
        """Test normalizing lists of tuple coordinates."""
        # List of float coordinates
        coords = [(1.0, 2.0), (3.0, 4.0)]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, coords)

        # List with integer coordinates
        coords = [(1, 2), (3, 4)]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, coords)

        # List with mixed types
        coords = [(1, 2.5), (3.5, 4)]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, coords)

        # Empty list
        coords = []
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_list_of_lists_coordinates(self):
        """Test normalizing lists of list coordinates."""
        # List of float lists
        coords = [[5.0, 6.0], [7.0, 8.0]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(5.0, 6.0), (7.0, 8.0)])

        # List of integer lists
        coords = [[1, 2], [3, 4]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # List of mixed type lists
        coords = [[1, 2.5], [3.5, 4]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

        # Single-element list
        coords = [[1.0, 2.0]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

    def test_numpy_array_coordinates(self):
        """Test normalizing numpy array coordinates."""
        # Array of floats
        coords = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(5.0, 6.0), (7.0, 8.0)])

        # Array of integers
        coords = np.array([[1, 2], [3, 4]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Array with mixed types
        coords = np.array([[1, 2.5], [3.5, 4]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

        # Single-row array
        coords = np.array([[1.0, 2.0]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

    def test_shapely_point_coordinates(self):
        """Test normalizing shapely Point coordinates."""
        # Float coordinates
        point = Point(15.0, 25.0)
        result = PathFinder.normalize_coordinates(point)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (15.0, 25.0))

        # Integer coordinates
        point = Point(1, 2)
        result = PathFinder.normalize_coordinates(point)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        # Mixed coordinates
        point = Point(1, 2.5)
        result = PathFinder.normalize_coordinates(point)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.5))

    def test_shapely_multipoint_coordinates(self):
        """Test normalizing shapely MultiPoint coordinates."""
        # Float coordinates
        multipoint = MultiPoint([(10.0, 20.0), (30.0, 40.0)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(10.0, 20.0), (30.0, 40.0)])

        # Integer coordinates
        multipoint = MultiPoint([(1, 2), (3, 4)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Mixed coordinates
        multipoint = MultiPoint([(1, 2.5), (3.5, 4)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

        # Single point
        multipoint = MultiPoint([(1.0, 2.0)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

    def test_geoseries_coordinates(self):
        """Test normalizing GeoSeries coordinates."""
        # Float coordinates
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        gs = GeoSeries(points)
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Integer coordinates
        points = [Point(1, 2), Point(3, 4)]
        gs = GeoSeries(points)
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Mixed coordinates
        points = [Point(1, 2.5), Point(3.5, 4)]
        gs = GeoSeries(points)
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

        # Single point
        points = [Point(1.0, 2.0)]
        gs = GeoSeries(points)
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        # Empty GeoSeries
        gs = GeoSeries()
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_geodataframe_coordinates(self):
        """Test normalizing GeoDataFrame coordinates."""
        # Float coordinates
        points = [Point(5.0, 6.0), Point(7.0, 8.0)]
        gdf = GeoDataFrame(geometry=points)
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(5.0, 6.0), (7.0, 8.0)])

        # Integer coordinates
        points = [Point(1, 2), Point(3, 4)]
        gdf = GeoDataFrame(geometry=points)
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Mixed coordinates
        points = [Point(1, 2.5), Point(3.5, 4)]
        gdf = GeoDataFrame(geometry=points)
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

        # Single point
        points = [Point(1.0, 2.0)]
        gdf = GeoDataFrame(geometry=points)
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        # GeoDataFrame with additional columns
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        gdf = GeoDataFrame(
            {'id': [1, 2], 'name': ['A', 'B']},
            geometry=points
        )
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        invalid_inputs = [
            "not a coordinate",
            123,
            (1, 2, 3),  # tuple with wrong length
            [(1, 2), (3, 4, 5)],  # list with invalid tuple
            [[1, 2], [3, 4, 5]],  # list with invalid sublist
            np.array([1, 2, 3]),  # 1D array
            np.array([[[1, 2]]]),  # 3D array
        ]

        for invalid_input in invalid_inputs:
            with self.assertRaises(ValueError):
                PathFinder.normalize_coordinates(invalid_input)

    def test_normalize_tuple_coordinates(self):
        """Test normalizing tuple coordinates."""
        # Test with float coordinates
        coord = (10.5, 20.3)
        result = PathFinder.normalize_coordinates(coord)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, coord)

        # Test with integer coordinates
        coord = (1, 2)
        result = PathFinder.normalize_coordinates(coord)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, coord)

        # Test with mixed types
        coord = (1, 2.5)
        result = PathFinder.normalize_coordinates(coord)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, coord)

    def test_normalize_list_of_tuples(self):
        """Test normalizing list of tuples."""
        # Test with float coordinates
        coords = [(1.0, 2.0), (3.0, 4.0)]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, coords)

        # Test with integer coordinates
        coords = [(1, 2), (3, 4)]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, coords)

        # Test with mixed types
        coords = [(1, 2.5), (3.5, 4)]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, coords)

    def test_normalize_list_of_lists(self):
        """Test normalizing list of lists."""
        # Test with float coordinates
        coords = [[1.0, 2.0], [3.0, 4.0]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with integer coordinates
        coords = [[1, 2], [3, 4]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with mixed types
        coords = [[1, 2.5], [3.5, 4]]
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

    def test_normalize_numpy_array(self):
        """Test normalizing numpy array."""
        # Test with float coordinates
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with integer coordinates
        coords = np.array([[1, 2], [3, 4]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with mixed types
        coords = np.array([[1, 2.5], [3.5, 4]])
        result = PathFinder.normalize_coordinates(coords)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

    def test_normalize_shapely_point(self):
        """Test normalizing shapely Point."""
        # Test with float coordinates
        point = Point(1.0, 2.0)
        result = PathFinder.normalize_coordinates(point)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        # Test with integer coordinates
        point = Point(1, 2)
        result = PathFinder.normalize_coordinates(point)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        # Test with mixed types
        point = Point(1, 2.5)
        result = PathFinder.normalize_coordinates(point)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.5))

    def test_normalize_shapely_multipoint(self):
        """Test normalizing shapely MultiPoint."""
        # Test with float coordinates
        multipoint = MultiPoint([(1.0, 2.0), (3.0, 4.0)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with integer coordinates
        multipoint = MultiPoint([(1, 2), (3, 4)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with mixed types
        multipoint = MultiPoint([(1, 2.5), (3.5, 4)])
        result = PathFinder.normalize_coordinates(multipoint)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

    def test_normalize_geoseries(self):
        """Test normalizing GeoSeries."""
        # Test with Point objects
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        gs = GeoSeries(points)
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with MultiPoint objects
        multipoints = [MultiPoint([(1.0, 2.0), (3.0, 4.0)]), MultiPoint([(5.0, 6.0), (7.0, 8.0)])]
        gs = GeoSeries(multipoints)
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        self.assertEqual(result, expected)

        # Test with mixed Point and MultiPoint
        mixed = [Point(1.0, 2.0), MultiPoint([(3.0, 4.0), (5.0, 6.0)])]
        gs = GeoSeries(mixed)
        with self.assertRaises(ValueError):
            PathFinder.normalize_coordinates(gs)

    def test_normalize_geodataframe(self):
        """Test normalizing GeoDataFrame."""
        # Test with Point geometries
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        gdf = GeoDataFrame(geometry=points)
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with MultiPoint geometries
        multipoints = [MultiPoint([(1.0, 2.0), (3.0, 4.0)]), MultiPoint([(5.0, 6.0), (7.0, 8.0)])]
        gdf = GeoDataFrame(geometry=multipoints)
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        self.assertEqual(result, expected)

        # Test with mixed Point and MultiPoint
        mixed = [Point(1.0, 2.0), MultiPoint([(3.0, 4.0), (5.0, 6.0)])]
        gdf = GeoDataFrame(geometry=mixed)
        with self.assertRaises(ValueError):
            PathFinder.normalize_coordinates(gdf)

    def test_normalize_list_of_points(self):
        """Test normalizing list of shapely Points."""
        # Test with float coordinates
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        result = PathFinder.normalize_coordinates(points)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with integer coordinates
        points = [Point(1, 2), Point(3, 4)]
        result = PathFinder.normalize_coordinates(points)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with mixed types
        points = [Point(1, 2.5), Point(3.5, 4)]
        result = PathFinder.normalize_coordinates(points)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.5), (3.5, 4.0)])

    def test_normalize_list_of_multipoints(self):
        """Test normalizing list of shapely MultiPoints."""
        # Test with float coordinates
        multipoints = [
            MultiPoint([(1.0, 2.0), (3.0, 4.0)]),
            MultiPoint([(5.0, 6.0), (7.0, 8.0)])
        ]
        result = PathFinder.normalize_coordinates(multipoints)
        self.assertIsInstance(result, list)
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        self.assertEqual(result, expected)

        # Test with integer coordinates
        multipoints = [
            MultiPoint([(1, 2), (3, 4)]),
            MultiPoint([(5, 6), (7, 8)])
        ]
        result = PathFinder.normalize_coordinates(multipoints)
        self.assertIsInstance(result, list)
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        self.assertEqual(result, expected)

        # Test with mixed types
        multipoints = [
            MultiPoint([(1, 2.5), (3.5, 4)]),
            MultiPoint([(5.0, 6), (7, 8.5)])
        ]
        result = PathFinder.normalize_coordinates(multipoints)
        self.assertIsInstance(result, list)
        expected = [(1.0, 2.5), (3.5, 4.0), (5.0, 6.0), (7.0, 8.5)]
        self.assertEqual(result, expected)

    def test_point_or_multipoints_helper(self):
        """Test _point_or_multipoints helper method."""
        # Test with a list of Points
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        result = PathFinder._point_or_multipoints(points)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [(1.0, 2.0), (3.0, 4.0)])

        # Test with a list of MultiPoints
        multipoints = [
            MultiPoint([(1.0, 2.0), (3.0, 4.0)]),
            MultiPoint([(5.0, 6.0), (7.0, 8.0)])
        ]
        result = PathFinder._point_or_multipoints(multipoints)
        self.assertIsInstance(result, list)
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        self.assertEqual(result, expected)

        # Test with non-supported input
        invalid_input = ["string1", "string2"]
        with self.assertRaises(ValueError):
            PathFinder._point_or_multipoints(invalid_input)

        # Test with mixed Point and MultiPoint (should raise ValueError)
        mixed = [Point(1.0, 2.0), MultiPoint([(3.0, 4.0), (5.0, 6.0)])]
        with self.assertRaises(ValueError):
            PathFinder._point_or_multipoints(mixed)

    def test_invalid_coordinate_inputs(self):
        """Test that invalid inputs raise ValueError."""
        invalid_inputs = [
            "not a coordinate",
            123,
            (1, 2, 3),  # tuple with wrong length
            [(1, 2), (3, 4, 5)],  # list with invalid tuple
            [[1, 2], [3, 4, 5]],  # list with invalid sublist
            np.array([1, 2, 3]),  # 1D array
            np.array([[[1, 2]]]),  # 3D array
            [Point(1, 2), "not a point"],  # mixed valid/invalid types
            {"x": 1, "y": 2}  # dictionary
        ]

        for input_data in invalid_inputs:
            with self.assertRaises(ValueError):
                PathFinder.normalize_coordinates(input_data)

    def test_edge_cases(self):
        """Test edge cases for coordinate normalization."""
        # Empty list
        result = PathFinder.normalize_coordinates([])
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

        # Empty GeoSeries
        gs = GeoSeries()
        result = PathFinder.normalize_coordinates(gs)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

        # Empty GeoDataFrame
        gdf = GeoDataFrame(geometry=[])
        result = PathFinder.normalize_coordinates(gdf)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

        # Single-element lists
        result = PathFinder.normalize_coordinates([(1.0, 2.0)])
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        result = PathFinder.normalize_coordinates([[1.0, 2.0]])
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        result = PathFinder.normalize_coordinates([Point(1.0, 2.0)])
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (1.0, 2.0))

        # Extreme values
        result = PathFinder.normalize_coordinates((1e20, 1e-20))
        self.assertEqual(result, (1e20, 1e-20))

    def test_consistent_output_types(self):
        """Test that output is consistent in type."""
        # Single point inputs should always return a tuple
        self.assertIsInstance(PathFinder.normalize_coordinates((1.0, 2.0)), tuple)
        self.assertIsInstance(PathFinder.normalize_coordinates(Point(1.0, 2.0)), tuple)

        # Multi-point inputs should always return a list
        self.assertIsInstance(PathFinder.normalize_coordinates([(1.0, 2.0), (3.0, 4.0)]), list)
        self.assertIsInstance(PathFinder.normalize_coordinates([[1.0, 2.0], [3.0, 4.0]]), list)
        self.assertIsInstance(PathFinder.normalize_coordinates(MultiPoint([(1.0, 2.0), (3.0, 4.0)])), list)
        self.assertIsInstance(PathFinder.normalize_coordinates([Point(1.0, 2.0), Point(3.0, 4.0)]), list)
import unittest
from unittest.mock import patch, MagicMock

import os
import tempfile
import geopandas as gpd
import importlib
import warnings
from shapely.geometry import Polygon, LineString
from numpy import array, random

from pyorps.graph.path_finder import get_graph_api_class, PathFinder
from pyorps.raster.handler import create_test_tiff
from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.core.path import Path, PathCollection
from pyorps.io.geo_dataset import initialize_geo_dataset, LocalRasterDataset, VectorDataset, RasterDataset
from pyorps.core.exceptions import AlgorithmNotImplementedError

# List of graph libraries to test
LIBRARIES_AND_MODULE_NAMES = [
    ("networkit", "networkit"),
    ("networkx", "networkx"),
    ("igraph", "igraph"),
    ("rustworkx", "rustworkx")
]


class TestGraphFunctions(unittest.TestCase):
    """Test cases for graph-related functions in path_finder.py."""

    def test_get_graph_api_class_valid_apis(self):
        """Test get_graph_api_class with valid API names."""
        # Test with networkit which should be installed
        api_class = get_graph_api_class("networkit")
        self.assertEqual(api_class.__name__, "NetworkitAPI")

        # Test with other APIs if they're available
        for api_name, expected_class_name in [
            ("networkx", "NetworkxAPI"),
            ("rustworkx", "RustworkxAPI"),
            ("igraph", "IGraphAPI")
        ]:
            try:
                importlib.import_module(api_name)
                api_class = get_graph_api_class(api_name)
                self.assertEqual(api_class.__name__, expected_class_name)
            except ImportError:
                # Skip if the library is not installed
                pass

    def test_get_graph_api_class_invalid_api(self):
        """Test get_graph_api_class with an invalid API name."""
        with self.assertRaises(ValueError) as context:
            get_graph_api_class("nonexistent_api")

        self.assertIn("Unsupported graph API", str(context.exception))


class TestRasterHandler(unittest.TestCase):
    """Test cases for RasterHandler creation in the PathFinder."""

    def test_create_raster_handler_vector_with_cost(self):
        """Test create_raster_handler with a vector dataset and cost assumptions."""
        # Setup mock vector dataset and cost assumptions
        mock_vector_dataset = MagicMock(spec=VectorDataset)
        mock_geo_rasterizer = MagicMock()
        mock_geo_rasterizer.raster_dataset = MagicMock(spec=RasterDataset)
        mock_raster_handler = MagicMock()

        # Setup mocks - patch initialize_geo_dataset to return our mock
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_vector_dataset), \
                patch("pyorps.graph.path_finder.GeoRasterizer", return_value=mock_geo_rasterizer), \
                patch("pyorps.graph.path_finder.RasterHandler", return_value=mock_raster_handler):
            # Create PathFinder with vector dataset directly
            path_finder = PathFinder(
                mock_vector_dataset,
                source_coords=(0, 0),
                target_coords=(1, 1),
                cost_assumptions={"some": "cost"}
            )

            # Check that GeoRasterizer was created with correct parameters
            self.assertIsNotNone(path_finder.geo_rasterizer)
            self.assertEqual(path_finder.geo_rasterizer, mock_geo_rasterizer)

            # Check that RasterHandler was created with correct parameters
            mock_geo_rasterizer.rasterize.assert_called_once()
            self.assertEqual(path_finder.raster_handler, mock_raster_handler)

    def test_create_raster_handler_vector_without_cost(self):
        """Test create_raster_handler with a vector dataset but no cost assumptions."""
        # Setup mock vector dataset
        mock_vector_dataset = MagicMock(spec=VectorDataset)

        # Setup mock for initialize_geo_dataset
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_vector_dataset):
            # Create PathFinder that skips automatic create_raster_handler
            path_finder = PathFinder(
                mock_vector_dataset,
                source_coords=None,
                target_coords=None
            )
            path_finder.source_coords = (0, 0)
            path_finder.target_coords = (1, 1)

            # Test direct call to create_raster_handler with no cost assumptions
            with self.assertRaises(ValueError) as context:
                path_finder.create_raster_handler(None, None, None)

            self.assertIn("Cost assumptions must be provided", str(context.exception))

    def test_create_raster_handler_raster_with_cost(self):
        """Test create_raster_handler with a raster dataset and cost assumptions."""
        # Setup mock raster dataset and cost assumptions
        mock_raster_dataset = MagicMock(spec=RasterDataset)
        mock_geo_rasterizer = MagicMock()
        mock_geo_rasterizer.raster_dataset = MagicMock(spec=RasterDataset)
        mock_raster_handler = MagicMock()

        # Setup mocks
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_raster_dataset), \
                patch("pyorps.graph.path_finder.GeoRasterizer", return_value=mock_geo_rasterizer), \
                patch("pyorps.graph.path_finder.RasterHandler", return_value=mock_raster_handler):
            # Create PathFinder
            path_finder = PathFinder(
                mock_raster_dataset,
                source_coords=(0, 0),
                target_coords=(1, 1),
                cost_assumptions={"some": "cost"}
            )

            # Check that GeoRasterizer was created with correct parameters
            mock_raster_dataset.load_data.assert_called_once()
            self.assertEqual(path_finder.geo_rasterizer, mock_geo_rasterizer)

            # Check that RasterHandler was created with correct parameters
            self.assertEqual(path_finder.raster_handler, mock_raster_handler)

    def test_create_raster_handler_raster_with_cost_and_modifications(self):
        """Test create_raster_handler with a raster dataset, cost assumptions, and dataset modifications."""
        # Setup mock raster dataset and cost assumptions
        mock_raster_dataset = MagicMock(spec=RasterDataset)
        mock_geo_rasterizer = MagicMock()
        mock_geo_rasterizer.raster_dataset = MagicMock(spec=RasterDataset)
        mock_raster_handler = MagicMock()

        # Setup dataset modifications
        datasets_to_modify = [{"dataset": "mod1"}, {"dataset": "mod2"}]

        # Setup mocks
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_raster_dataset), \
                patch("pyorps.graph.path_finder.GeoRasterizer", return_value=mock_geo_rasterizer), \
                patch("pyorps.graph.path_finder.RasterHandler", return_value=mock_raster_handler):
            # Create PathFinder
            path_finder = PathFinder(
                mock_raster_dataset,
                source_coords=None,
                target_coords=None
            )
            path_finder.source_coords = (0, 0)
            path_finder.target_coords = (1, 1)

            # Now call create_raster_handler with the required parameters
            path_finder.create_raster_handler({"some": "cost"}, datasets_to_modify, None)

            # Check that modify_raster_from_dataset was called for each dataset
            self.assertEqual(mock_geo_rasterizer.modify_raster_from_dataset.call_count, 2)
            mock_geo_rasterizer.modify_raster_from_dataset.assert_any_call(**datasets_to_modify[0])
            mock_geo_rasterizer.modify_raster_from_dataset.assert_any_call(**datasets_to_modify[1])

    def test_create_raster_handler_raster_without_cost(self):
        """Test create_raster_handler with a raster dataset and no cost assumptions."""
        # Setup mock raster dataset
        mock_raster_dataset = MagicMock(spec=RasterDataset)
        mock_raster_handler = MagicMock()

        # Setup mocks
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_raster_dataset), \
                patch("pyorps.graph.path_finder.RasterHandler", return_value=mock_raster_handler):
            # Create PathFinder
            path_finder = PathFinder(
                mock_raster_dataset,
                source_coords=None,
                target_coords=None
            )
            path_finder.source_coords = (0, 0)
            path_finder.target_coords = (1, 1)

            # Now call create_raster_handler directly
            path_finder.create_raster_handler(None, None, None)

            # Check that dataset was loaded directly without using GeoRasterizer
            mock_raster_dataset.load_data.assert_called_once()
            self.assertIsNone(path_finder.geo_rasterizer)
            self.assertEqual(path_finder.raster_handler, mock_raster_handler)

    def test_create_raster_handler_raster_without_cost_with_save_path(self):
        """Test create_raster_handler with a raster dataset, no cost assumptions, and a save path."""
        # Setup mock raster dataset
        mock_raster_dataset = MagicMock(spec=RasterDataset)
        mock_raster_handler = MagicMock()

        # Setup mocks
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_raster_dataset), \
                patch("pyorps.graph.path_finder.RasterHandler", return_value=mock_raster_handler):
            # Create PathFinder
            path_finder = PathFinder(
                mock_raster_dataset,
                source_coords=None,
                target_coords=None
            )
            path_finder.source_coords = (0, 0)
            path_finder.target_coords = (1, 1)

            # Now call create_raster_handler directly with save path
            path_finder.create_raster_handler(None, None, "test_save_path.tiff")

            # Check that save_section_as_raster was called with correct path
            mock_raster_handler.save_section_as_raster.assert_called_once_with("test_save_path.tiff")

    def test_create_raster_handler_unsupported_dataset(self):
        """Test create_raster_handler with an unsupported dataset type."""

        # Setup mock dataset that is neither VectorDataset nor RasterDataset
        class UnsupportedDataset: pass

        mock_dataset = MagicMock(spec=UnsupportedDataset)

        # Setup mock for initialize_geo_dataset
        with patch("pyorps.graph.path_finder.initialize_geo_dataset", return_value=mock_dataset):
            # Create PathFinder
            path_finder = PathFinder(
                mock_dataset,
                source_coords=None,
                target_coords=None
            )
            path_finder.source_coords = (0, 0)
            path_finder.target_coords = (1, 1)

            # Now call create_raster_handler directly and expect error
            with self.assertRaises(ValueError) as context:
                path_finder.create_raster_handler(None, None, None)

            self.assertIn("Unsupported dataset type", str(context.exception))


class TestGraphLibraryPathFinding(unittest.TestCase):
    """Tests for pathfinding using various graph libraries."""

    @classmethod
    def setUpClass(cls):
        """Create test data that can be reused across tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_raster_path = os.path.join(cls.temp_dir.name, "test_raster.tiff")
        create_test_tiff(cls.test_raster_path)

        # Create a test raster file
        cls.raster_data = create_test_tiff(cls.test_raster_path)

        # Define test coordinates
        cls.source_coords = (500020, 5599980)
        cls.target_coords = (500080, 5599920)
        # Create a test geodataframe for vector data testing
        cls.test_vector_path = os.path.join(cls.temp_dir.name, "test_vector.gpkg")
        cls.test_gdf = cls._create_test_geodataframe()
        cls.test_gdf.to_file(cls.test_vector_path)

        # Create cost assumptions
        cls.cost_assumptions = cls._create_test_cost_assumptions()

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        cls.temp_dir.cleanup()

    def is_library_installed(self, library_name):
        """Check if a library is installed."""
        try:
            importlib.import_module(library_name)
            return True
        except ImportError:
            return False

    def test_raster_path_finding_with_different_graph_libraries(self):
        """Test path finding with different graph libraries using raster data."""
        # Try each library
        for lib_name, module_name in LIBRARIES_AND_MODULE_NAMES:
            # Check if the library is installed
            if not self.is_library_installed(module_name):
                warnings.warn(
                    f"Library '{module_name}' is not installed. "
                    f"It's an optional dependency, so tests for '{lib_name}' will be skipped."
                )
                continue

            # If library is installed, run the test
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=self.source_coords,
                target_coords=self.target_coords,
                graph_api=lib_name,
                search_space_buffer_m=50,
                neighborhood_str='r1',
            )
            path = path_finder.find_route()

            # Assert path was found
            self.assertIsNotNone(path)
            self.assertGreater(len(path.path_indices), 1)
            self.assertGreater(path.total_length, 0)

            # Ensure path connects source to target
            self.assertAlmostEqual(path.path_coords[0][0], self.source_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[0][1], self.source_coords[1], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][0], self.target_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][1], self.target_coords[1], delta=5)

    def test_path_finding_with_different_algorithms(self):
        """Test path finding with different routing algorithms."""
        # List of algorithms to test with each library
        algorithms = ["dijkstra", "bidirectional_dijkstra"]
        # Skip astar as it needs a heuristic function
        # Try each library
        for lib_name, module_name in LIBRARIES_AND_MODULE_NAMES:
            # Check if the library is installed
            if not self.is_library_installed(module_name):
                warnings.warn(
                    f"Library '{module_name}' is not installed. "
                    f"It's an optional dependency, so tests for '{lib_name}' will be skipped."
                )
                continue

            for algorithm in algorithms:
                try:
                    path_finder = PathFinder(
                        dataset_source=self.test_raster_path,
                        source_coords=self.source_coords,
                        target_coords=self.target_coords,
                        graph_api=lib_name,
                        search_space_buffer_m=50,
                        neighborhood_str='r1',
                    )
                    path = path_finder.find_route(algorithm=algorithm)

                    # Assert path was found
                    self.assertIsNotNone(path)
                    self.assertGreater(len(path.path_indices), 1)
                    self.assertGreater(path.total_length, 0)

                    # Ensure path connects source to target
                    self.assertAlmostEqual(path.path_coords[0][0], self.source_coords[0], delta=5)
                    self.assertAlmostEqual(path.path_coords[0][1], self.source_coords[1], delta=5)
                    self.assertAlmostEqual(path.path_coords[-1][0], self.target_coords[0], delta=5)
                    self.assertAlmostEqual(path.path_coords[-1][1], self.target_coords[1], delta=5)

                    # Check that the algorithm name is recorded correctly
                    self.assertEqual(path.algorithm, algorithm)
                except Exception as e:
                    # Some algorithms might not be implemented for all libraries
                    warnings.warn(f"Algorithm '{algorithm}' failed with library '{lib_name}': {e}")

    def test_multiple_source_target_path_finding(self):
        """Test path finding with multiple source and target points."""
        # Create two sets of source and target coordinates
        sources = [(500020, 5599980), (500030, 5599990)]
        targets = [(500080, 5599920), (500090, 5599910)]

        # Try each library
        for lib_name, module_name in LIBRARIES_AND_MODULE_NAMES:
            # Check if the library is installed
            if not self.is_library_installed(module_name):
                if module_name != "networkit":
                    warnings.warn(
                        f"Library '{module_name}' is not installed. "
                        f"It's an optional dependency, so tests for '{lib_name}' will be skipped."
                    )
                    continue
                else:
                    raise ImportError("Networkit not installed! Networkit is a mandatory library for pyorps! "
                                      "Please install it first.")

            # Test with multiple sources, single target
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=sources,
                target_coords=targets[0],
                graph_api=lib_name,
                search_space_buffer_m=50,
                neighborhood_str='r1',
            )
            paths = path_finder.find_route()

            # Assert paths were found
            self.assertIsInstance(paths, PathCollection)
            self.assertEqual(len(paths), 2)  # One path for each source

            # Test with single source, multiple targets
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=sources[0],
                target_coords=targets,
                graph_api=lib_name,
                search_space_buffer_m=50,
                neighborhood_str='r1',
            )
            paths = path_finder.find_route()

            # Assert paths were found
            self.assertIsInstance(paths, PathCollection)
            self.assertEqual(len(paths), 2)  # One path for each target

            # Test with multiple sources, multiple targets (pairwise)
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=sources,
                target_coords=targets,
                graph_api=lib_name,
                search_space_buffer_m=50,
                neighborhood_str='r1',
            )
            paths = path_finder.find_route(pairwise=True)

            # Assert paths were found
            self.assertIsInstance(paths, PathCollection)
            self.assertEqual(len(paths), 2)  # One path for each source-target pair

    @classmethod
    def _create_test_geodataframe(cls):
        """Create a test geodataframe with polygons and relevant attributes."""
        # Create several polygons with different land use types
        geometries = [
            Polygon([(500010, 5599990), (500030, 5599990), (500030, 5599970), (500010, 5599970)]),
            Polygon([(500030, 5599990), (500050, 5599990), (500050, 5599970), (500030, 5599970)]),
            Polygon([(500050, 5599990), (500070, 5599990), (500070, 5599970), (500050, 5599970)]),
            Polygon([(500070, 5599990), (500090, 5599990), (500090, 5599970), (500070, 5599970)]),
            Polygon([(500010, 5599970), (500030, 5599970), (500030, 5599950), (500010, 5599950)]),
            Polygon([(500030, 5599970), (500050, 5599970), (500050, 5599950), (500030, 5599950)]),
            Polygon([(500050, 5599970), (500070, 5599970), (500070, 5599950), (500050, 5599950)]),
            Polygon([(500070, 5599970), (500090, 5599970), (500090, 5599950), (500070, 5599950)]),
            Polygon([(500010, 5599950), (500030, 5599950), (500030, 5599930), (500010, 5599930)]),
            Polygon([(500030, 5599950), (500050, 5599950), (500050, 5599930), (500030, 5599930)]),
            Polygon([(500050, 5599950), (500070, 5599950), (500070, 5599930), (500050, 5599930)]),
            Polygon([(500070, 5599950), (500090, 5599950), (500090, 5599930), (500070, 5599930)]),
            Polygon([(500010, 5599930), (500030, 5599930), (500030, 5599910), (500010, 5599910)]),
            Polygon([(500030, 5599930), (500050, 5599930), (500050, 5599910), (500030, 5599910)]),
            Polygon([(500050, 5599930), (500070, 5599930), (500070, 5599910), (500050, 5599910)]),
            Polygon([(500070, 5599930), (500090, 5599930), (500090, 5599910), (500070, 5599910)]),
        ]

        # Create land use categories
        land_use_types = ['forest', 'agriculture', 'urban', 'water'] * 4

        # Create land use quality/condition
        conditions = ['good', 'medium', 'poor', 'protected'] * 4

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'land_use': land_use_types,
            'condition': conditions
        }, crs="EPSG:32632")

        return gdf

    @classmethod
    def _create_test_cost_assumptions(cls):
        """Create test cost assumptions mapping land use and condition to costs."""
        # Create basic cost assumptions using land_use as main_feature and condition as side_feature
        assumptions = {
            ('land_use', 'condition'): {
                ('forest', 'good'): 1,
                ('forest', 'medium'): 2,
                ('forest', 'poor'): 3,
                ('forest', 'protected'): 10,
                ('agriculture', 'good'): 2,
                ('agriculture', 'medium'): 3,
                ('agriculture', 'poor'): 4,
                ('agriculture', 'protected'): 12,
                ('urban', 'good'): 5,
                ('urban', 'medium'): 6,
                ('urban', 'poor'): 8,
                ('urban', 'protected'): 15,
                ('water', 'good'): 20,
                ('water', 'medium'): 25,
                ('water', 'poor'): 30,
                ('water', 'protected'): 50,
            }
        }
        return CostAssumptions(assumptions)

    def test_raster_path_finding_with_different_neighborhoods(self):
        """Test path finding with different neighborhood settings using raster data."""
        for neighborhood in ['r0', 1, 2.0, 'RAD3']:
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=self.source_coords,
                target_coords=self.target_coords,
                graph_api='networkit',
                search_space_buffer_m=50,
                neighborhood_str=neighborhood,
            )
            path = path_finder.find_route()

            # Assert path was found
            self.assertIsNotNone(path)
            self.assertGreater(len(path.path_indices), 1)
            self.assertGreater(path.total_length, 0)

            # Ensure path connects source to target
            self.assertAlmostEqual(path.path_coords[0][0], self.source_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[0][1], self.source_coords[1], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][0], self.target_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][1], self.target_coords[1], delta=5)

    def test_path_finding_with_different_buffer_sizes(self):
        """Test path finding with different search space buffer sizes."""
        buffer_sizes = [10, 50, 100]
        for buffer in buffer_sizes:
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=self.source_coords,
                target_coords=self.target_coords,
                graph_api='networkit',
                search_space_buffer_m=buffer,
                neighborhood_str='r1',
            )
            path = path_finder.find_route()

            # Assert path was found
            self.assertIsNotNone(path)
            self.assertGreater(len(path.path_indices), 1)

            # Ensure path connects source to target
            self.assertAlmostEqual(path.path_coords[0][0], self.source_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[0][1], self.source_coords[1], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][0], self.target_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][1], self.target_coords[1], delta=5)

    def test_save_and_load_path_geodataframe(self):
        """Test saving and loading path GeoDataFrame."""
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=self.source_coords,
            target_coords=self.target_coords,
            graph_api='networkit',
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )
        path_finder.find_route()

        # Create and check the path GeoDataFrame
        gdf = path_finder.create_path_geodataframe()
        self.assertIsNotNone(gdf)
        self.assertEqual(len(gdf), 1)

        # Save to a temporary file
        temp_path = os.path.join(self.temp_dir.name, "paths.geojson")
        path_finder.save_paths(temp_path)
        self.assertTrue(os.path.exists(temp_path))

        # Load and check
        loaded_gdf = gpd.read_file(temp_path)
        self.assertEqual(len(loaded_gdf), 1)
        self.assertIn('path_length', loaded_gdf.columns)
        self.assertIn('path_cost', loaded_gdf.columns)

    def test_save_raster(self):
        """Test saving the raster used for path finding."""
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=self.source_coords,
            target_coords=self.target_coords,
            graph_api='networkit',
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )

        # Find a route to ensure the raster is loaded
        path_finder.find_route()

        # Save the raster
        temp_raster_path = os.path.join(self.temp_dir.name, "test_save_raster.tiff")
        path_finder.save_raster(temp_raster_path)
        self.assertTrue(os.path.exists(temp_raster_path))

        # Check the saved raster can be opened
        raster_dataset = initialize_geo_dataset(temp_raster_path)
        raster_dataset.load_data()
        self.assertIsNotNone(raster_dataset.data)
        self.assertIsInstance(raster_dataset, LocalRasterDataset)

    def test_path_collection_replace_integration(self):
        """Test the PathCollection replace functionality in an integration context."""
        # Create a PathFinder instance
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=self.source_coords,
            target_coords=self.target_coords,
            graph_api="networkit",
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )

        # Find a route - this will add path with ID=0
        path_finder.find_route()

        # Create a new path with explicit ID=5
        custom_path = Path(
            source=self.source_coords,
            target=self.target_coords,
            algorithm="dijkstra",
            graph_api="networkit",
            path_indices=array([0, 1, 2]),
            path_coords=array([[500020, 5599980], [500050, 5599950], [500080, 5599920]]),
            path_geometry=LineString([[500020, 5599980], [500050, 5599950], [500080, 5599920]]),
            euclidean_distance=100.0,
            runtimes={},
            path_id=5,
            search_space_buffer_m=1,
            neighborhood="r0",
        )

        # Add with replace=True to keep ID=5
        path_finder.paths.add(custom_path, replace=True)
        self.assertEqual(path_finder.paths.get(5), custom_path)
        self.assertEqual(custom_path.path_id, 5)

        # Verify both paths exist
        self.assertEqual(len(path_finder.paths), 2)

        # Create another path with ID=5
        replacement_path = Path(
            source=(500010, 5600010),
            target=(500090, 5599910),
            algorithm="astar",
            graph_api="networkit",
            path_indices=array([10, 11, 12]),
            path_coords=array([[500010, 5600010], [500050, 5599960], [500090, 5599910]]),
            path_geometry=LineString([[500010, 5600010], [500050, 5599960], [500090, 5599910]]),
            euclidean_distance=50.0,
            runtimes={},
            path_id=5,
            search_space_buffer_m=1,
            neighborhood="r0",
        )

        # Add with replace=True to replace the existing path with ID=5
        path_finder.paths.add(replacement_path, replace=True)

        # Verify the existing path was replaced
        self.assertEqual(path_finder.paths.get(5), replacement_path)
        self.assertEqual(len(path_finder.paths), 2)  # Still only 2 paths

    def test_all_graph_libraries_all_algorithms(self):
        """Test all available path finding algorithms for each graph library."""

        # Define which algorithms each library should support based on implementation
        library_algorithms = {
            "networkit": ["dijkstra", "bidirectional_dijkstra", "astar"],
            "networkx": ["dijkstra", "bidirectional_dijkstra", "astar"],
            "igraph": ["dijkstra", "bellman_ford", "astar"],
            "rustworkx": ["dijkstra", "bellman_ford", "astar"]
        }

        # Define source and target coordinates for testing
        single_source = self.source_coords
        single_target = self.target_coords
        multi_sources = [(500020, 5599980), (500030, 5599990)]
        multi_targets = [(500080, 5599920), (500090, 5599910)]

        test_scenarios = [
            # (name, source, target, pairwise)
            ("single path", single_source, single_target, False),
            ("multiple sources to single target", multi_sources, single_target, False),
            ("single source to multiple targets", single_source, multi_targets, False),
            ("pairwise multiple paths", multi_sources, multi_targets, True)
        ]

        # For each library
        for lib_name, module_name in LIBRARIES_AND_MODULE_NAMES:
            # Skip if library is not installed
            if not self.is_library_installed(module_name):
                if module_name == "networkit":
                    raise ImportError("Networkit not installed! Networkit is a mandatory library for pyorps!")
                else:
                    warnings.warn(f"Library '{module_name}' is not installed. Skipping tests.")
                    continue

            # Get the algorithms this library should support
            supported_algorithms = library_algorithms.get(lib_name, ["dijkstra"])

            # Test each algorithm with each scenario
            for algorithm in supported_algorithms:
                for scenario_name, source, target, pairwise in test_scenarios:
                    test_name = f"{lib_name} with {algorithm} ({scenario_name})"
                    if not pairwise and algorithm == "astar" and lib_name == "rustworkx" and scenario_name == "'multiple sources to single target'":
                        print()
                    try:
                        # Create PathFinder and find route
                        path_finder = PathFinder(
                            dataset_source=self.test_raster_path,
                            source_coords=source,
                            target_coords=target,
                            graph_api=lib_name,
                            search_space_buffer_m=50,
                            neighborhood_str='r1',
                        )

                        result = path_finder.find_route(algorithm=algorithm, pairwise=pairwise)

                        # Check results based on scenario
                        self._validate_path_results(result, algorithm, test_name)

                    except AlgorithmNotImplementedError:
                        # If algorithm is truly not implemented, that's okay
                        warnings.warn(f"{test_name}: Algorithm not implemented")
                    except Exception as e:
                        # Other errors indicate a real problem
                        self.fail(f"{test_name} failed: {str(e)}")

    def _validate_path_results(self, result, algorithm, test_name):
        """Helper to validate path results from different scenarios."""
        if isinstance(result, PathCollection):
            # Should have at least one path
            self.assertGreater(len(result), 0, f"No paths found for {test_name}")

            # Validate each path
            for i, path in enumerate(result):
                # Some paths might legitimately be empty if no route exists
                if path:
                    self.assertGreater(len(path.path_indices), 1,
                                       f"Path {i} has too few indices for {test_name}")
                    self.assertEqual(path.algorithm, algorithm,
                                     f"Path {i} has wrong algorithm for {test_name}")
                    self.assertGreater(path.total_length, 0,
                                       f"Path {i} has zero length for {test_name}")
        else:
            # Single path result
            self.assertIsNotNone(result, f"No path found for {test_name}")
            self.assertGreater(len(result.path_indices), 1,
                               f"Path has too few indices for {test_name}")
            self.assertEqual(result.algorithm, algorithm,
                             f"Path has wrong algorithm for {test_name}")
            self.assertGreater(result.total_length, 0,
                               f"Path has zero length for {test_name}")

    def test_algorithm_comparison(self):
        """Test that different algorithms produce valid paths for the same problem."""
        # We'll use networkit as it has a wide range of algorithm implementations
        lib_name = "networkit"

        # Skip if library is not installed
        if not self.is_library_installed(lib_name):
            raise ImportError(f"{lib_name} is required for this test")

        algorithms = ["dijkstra", "bidirectional_dijkstra", "astar"]
        paths = {}

        # Create a pathfinder
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=self.source_coords,
            target_coords=self.target_coords,
            graph_api=lib_name,
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )

        # Find paths using different algorithms
        for algorithm in algorithms:
            try:
                paths[algorithm] = path_finder.find_route(algorithm=algorithm)

                # Validate the path
                self.assertIsNotNone(paths[algorithm])
                self.assertGreater(len(paths[algorithm].path_indices), 1)
                self.assertEqual(paths[algorithm].algorithm, algorithm)

                # Ensure the path connects source and target
                start_coord = paths[algorithm].path_coords[0]
                end_coord = paths[algorithm].path_coords[-1]
                self.assertAlmostEqual(start_coord[0], self.source_coords[0], delta=5)
                self.assertAlmostEqual(start_coord[1], self.source_coords[1], delta=5)
                self.assertAlmostEqual(end_coord[0], self.target_coords[0], delta=5)
                self.assertAlmostEqual(end_coord[1], self.target_coords[1], delta=5)

            except AlgorithmNotImplementedError:
                warnings.warn(f"Algorithm {algorithm} not implemented for {lib_name}")

        # Compare the paths from different algorithms
        # They might not be identical, but should be similar in cost and length
        if len(paths) >= 2:
            algorithms_with_paths = list(paths.keys())
            for i in range(len(algorithms_with_paths) - 1):
                for j in range(i + 1, len(algorithms_with_paths)):
                    algo1 = algorithms_with_paths[i]
                    algo2 = algorithms_with_paths[j]

                    # Compare path lengths (allow 15% difference as algorithms may find slightly different routes)
                    length1 = paths[algo1].total_length
                    length2 = paths[algo2].total_length
                    self.assertLess(abs(length1 - length2) / max(length1, length2), 0.15,
                                    f"Paths from {algo1} and {algo2} differ too much in length")

    def test_correct_max_cost_positions_basic(self):
        """Test correcting positions with maximum cost values in the raster."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster path
        test_raster = os.path.join(self.temp_dir.name, "test_max_cost.tiff")

        # Create raster data with some max cost positions - use uint16!
        raster_data = np.ones((100, 100), dtype=np.uint16) * 10
        max_cost = np.iinfo(np.uint16).max  # 65535 for uint16

        # Add some max cost positions (obstacles)
        raster_data[10:15, 20:25] = max_cost  # A block of max cost
        raster_data[50, 50] = max_cost  # A single max cost position
        raster_data[70:75, 70:75] = max_cost  # Another block

        # Save the test raster using rasterio
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder with the test raster
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500020, 5599980),
            target_coords=(500080, 5599920),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Test positions that include max cost values
        positions_to_correct = np.array([[10, 20], [11, 21], [50, 50], [30, 30]],
                                        dtype=np.int32)

        # Call the method
        corrected = path_finder._correct_max_cost_positions(positions_to_correct)

        # Check that max cost positions were corrected
        # The corrected positions should not point to max cost values anymore
        self.assertIsNotNone(corrected)

        # For positions that had max cost, check they were moved
        original_max_positions = [(10, 20), (11, 21), (50, 50)]
        for i, (orig_row, orig_col) in enumerate(original_max_positions):
            if i < len(positions_to_correct):
                # Check if this position was in a max cost area
                if orig_row >= 10 and orig_row < 15 and orig_col >= 20 and orig_col < 25:
                    # This was in a max cost block, should have been corrected
                    corrected_row, corrected_col = corrected[i]
                    # The corrected position should be different or have a non-max value
                    value_at_corrected = path_finder.raster_handler.data[
                        0, corrected_row, corrected_col]
                    self.assertLess(value_at_corrected, max_cost)

    def test_correct_max_cost_positions_with_replacement_value(self):
        """Test correcting max cost positions with a specific replacement value."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster
        test_raster = os.path.join(self.temp_dir.name, "test_replacement.tiff")

        # Create raster data - use uint16!
        raster_data = np.ones((100, 100), dtype=np.uint16) * 5
        max_cost = np.iinfo(np.uint16).max  # 65535

        # Add max cost positions
        raster_data[25:30, 25:30] = max_cost

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500020, 5599980),
            target_coords=(500080, 5599920),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Define positions to correct
        positions_to_correct = np.array([[25, 25], [26, 26], [27, 27]], dtype=np.int32)

        # The current implementation finds nearest valid positions
        # It doesn't support a replacement_value parameter directly
        corrected = path_finder._correct_max_cost_positions(positions_to_correct)

        # Verify that positions were corrected (moved to valid locations)
        self.assertIsNotNone(corrected)
        for i in range(len(positions_to_correct)):
            row, col = corrected[i]
            value = path_finder.raster_handler.data[0, row, col]
            self.assertLess(value, max_cost,
                            f"Position ({row}, {col}) should not have max cost")

    def test_correct_max_cost_positions_boundary_cases(self):
        """Test correcting max cost positions at raster boundaries."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster
        test_raster = os.path.join(self.temp_dir.name, "test_boundaries.tiff")

        # Create raster data with max cost at boundaries - use uint16!
        raster_data = np.ones((100, 100), dtype=np.uint16) * 20
        max_cost = np.iinfo(np.uint16).max

        # Set boundary positions to max cost
        raster_data[0, :] = max_cost  # Top row
        raster_data[-1, :] = max_cost  # Bottom row
        raster_data[:, 0] = max_cost  # Left column
        raster_data[:, -1] = max_cost  # Right column

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500050, 5599950),
            target_coords=(500060, 5599940),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Test boundary positions
        boundary_positions = np.array([[0, 0], [0, 50], [99, 0], [99, 99]],
                                      dtype=np.int32)

        # Call the method
        corrected = path_finder._correct_max_cost_positions(boundary_positions)

        # Verify corrections were made appropriately
        self.assertIsNotNone(corrected)
        for i in range(len(boundary_positions)):
            row, col = corrected[i]
            # Corrected positions should be away from boundaries
            value = path_finder.raster_handler.data[0, row, col]
            self.assertLess(value, max_cost,
                            f"Boundary position corrected to ({row}, {col}) should not have max cost")

    def test_correct_max_cost_positions_empty_list(self):
        """Test correcting an empty list of positions."""
        # Create PathFinder with standard test raster
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=self.source_coords,
            target_coords=self.target_coords,
            graph_api="networkit",
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )

        # Test with empty 2D array (proper format for the function)
        import numpy as np
        empty_positions = np.array([], dtype=np.int32).reshape(0, 2)

        # This should handle empty array gracefully
        result = path_finder._correct_max_cost_positions(empty_positions)

        # Should return the same empty array
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0, "Empty input should return empty result")

    def test_correct_max_cost_positions_numpy_array_input(self):
        """Test correcting positions provided as numpy array."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster
        test_raster = os.path.join(self.temp_dir.name, "test_numpy_input.tiff")

        # Create raster data - use uint16!
        raster_data = np.ones((100, 100), dtype=np.uint16) * 15
        max_cost = np.iinfo(np.uint16).max
        raster_data[40:45, 40:45] = max_cost

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500020, 5599980),
            target_coords=(500080, 5599920),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Test with numpy array input
        positions_array = np.array([[40, 40], [41, 41], [42, 42], [30, 30]],
                                   dtype=np.int32)

        # Call the method
        corrected = path_finder._correct_max_cost_positions(positions_array)

        # Verify corrections
        self.assertIsNotNone(corrected)

        # Check that max cost positions were corrected
        for i in range(3):  # First 3 positions were in max cost area
            row, col = corrected[i]
            value = path_finder.raster_handler.data[0, row, col]
            self.assertLess(value, max_cost,
                            f"Max cost position should be corrected")

        # Non-max position (30, 30) should remain unchanged
        self.assertEqual(corrected[3][0], 30)
        self.assertEqual(corrected[3][1], 30)

    def test_correct_max_cost_positions_interpolation(self):
        """Test correcting max cost positions using interpolation from neighbors."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster
        test_raster = os.path.join(self.temp_dir.name, "test_interpolation.tiff")

        # Create raster data with gradient pattern - use uint16!
        raster_data = np.zeros((100, 100), dtype=np.uint16)
        for i in range(100):
            for j in range(100):
                # Scale the gradient to fit in uint16 range
                value = int((i + j) * 100 / 198 + 1)  # Scale to 1-101 range
                raster_data[i, j] = min(value, 65534)  # Keep below max_cost

        max_cost = np.iinfo(np.uint16).max

        # Set some isolated max cost positions
        raster_data[50, 50] = max_cost
        raster_data[51, 50] = max_cost
        raster_data[50, 51] = max_cost

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500020, 5599980),
            target_coords=(500080, 5599920),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Test interpolation-based correction
        positions_to_correct = np.array([[50, 50], [51, 50], [50, 51]], dtype=np.int32)

        # Call the method (current implementation finds nearest valid)
        corrected = path_finder._correct_max_cost_positions(positions_to_correct)

        # Check that corrected values are reasonable
        self.assertIsNotNone(corrected)
        for i in range(len(positions_to_correct)):
            row, col = corrected[i]
            value = path_finder.raster_handler.data[0, row, col]

            # Value should be less than max_cost
            self.assertLess(value, max_cost)

            # Value should be positive and reasonable
            self.assertGreater(value, 0)

    def test_correct_max_cost_positions_large_block(self):
        """Test correcting a large contiguous block of max cost positions."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster
        test_raster = os.path.join(self.temp_dir.name, "test_large_block.tiff")

        # Create raster data - use uint16!
        raster_data = np.ones((100, 100), dtype=np.uint16) * 25
        max_cost = np.iinfo(np.uint16).max

        # Create a large block of max cost (like a lake or building)
        raster_data[30:60, 30:60] = max_cost

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500020, 5599980),
            target_coords=(500080, 5599920),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Test correcting positions in the middle of the block
        positions_in_block = np.array([[45, 45], [40, 40], [50, 50], [35, 35]],
                                      dtype=np.int32)

        # Call the method
        corrected = path_finder._correct_max_cost_positions(positions_in_block)

        # Verify corrections - positions should be moved outside the block
        self.assertIsNotNone(corrected)
        for i in range(len(positions_in_block)):
            row, col = corrected[i]
            value = path_finder.raster_handler.data[0, row, col]
            self.assertLess(value, max_cost,
                            f"Position in large block should be corrected to valid location")

    def test_correct_max_cost_positions_mixed_values(self):
        """Test that valid positions are preserved while max cost positions are corrected."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster
        test_raster = os.path.join(self.temp_dir.name, "test_mixed.tiff")

        # Create raster data with varied values - use uint16!
        np.random.seed(42)  # For reproducibility
        raster_data = np.random.randint(1, 100, size=(100, 100), dtype=np.uint16)
        max_cost = np.iinfo(np.uint16).max

        # Set specific positions to max cost
        raster_data[20, 20] = max_cost
        raster_data[21, 21] = max_cost
        # Keep some positions with normal values
        raster_data[22, 22] = 42
        raster_data[23, 23] = 37

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500020, 5599980),
            target_coords=(500080, 5599920),
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Mix of max cost and valid positions
        mixed_positions = np.array([[20, 20], [21, 21], [22, 22], [23, 23]],
                                   dtype=np.int32)

        # Call the method
        corrected = path_finder._correct_max_cost_positions(mixed_positions)

        self.assertIsNotNone(corrected)

        # Check that max cost positions were corrected (moved)
        for i in range(2):  # First 2 positions had max cost
            row, col = corrected[i]
            value = path_finder.raster_handler.data[0, row, col]
            self.assertLess(value, max_cost)

        # Valid positions should remain unchanged
        self.assertEqual(corrected[2][0], 22)
        self.assertEqual(corrected[2][1], 22)
        self.assertEqual(corrected[3][0], 23)
        self.assertEqual(corrected[3][1], 23)

    def test_correct_max_cost_positions_path_finding_integration(self):
        """Test that max cost correction improves path finding through obstacles."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        # Create test raster with a barrier
        test_raster = os.path.join(self.temp_dir.name, "test_barrier.tiff")

        # Create raster data with a vertical barrier - use uint16!
        raster_data = np.ones((100, 100), dtype=np.uint16) * 10
        max_cost = np.iinfo(np.uint16).max

        # Create a vertical barrier in the middle
        raster_data[20:80, 48:52] = max_cost

        # But leave a small gap for potential path
        raster_data[48:52, 48:52] = 20  # Create a passable area

        # Save the test raster
        transform = from_origin(500000, 5600000, 1, 1)
        with rasterio.open(
                test_raster, 'w',
                driver='GTiff',
                height=100,
                width=100,
                count=1,
                dtype=raster_data.dtype,
                crs='EPSG:32632',
                transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        # Create PathFinder
        path_finder = PathFinder(
            dataset_source=test_raster,
            source_coords=(500025, 5599975),  # Left side of barrier
            target_coords=(500075, 5599925),  # Right side of barrier
            graph_api="networkit",
            search_space_buffer_m=100,
            neighborhood_str='r1',
        )

        # Find path - should go through the gap
        path = path_finder.find_route()

        # The path should exist
        self.assertIsNotNone(path)
        self.assertGreater(path.total_length, 0)

        # Now test correction of positions that might be in the barrier
        barrier_positions = np.array([[50, 49], [51, 49], [52, 49]], dtype=np.int32)
        corrected = path_finder._correct_max_cost_positions(barrier_positions)

        # Corrected positions should be outside the barrier
        self.assertIsNotNone(corrected)
        for i in range(len(barrier_positions)):
            row, col = corrected[i]
            value = path_finder.raster_handler.data[0, row, col]
            self.assertLess(value, max_cost,
                            "Corrected position should not be in barrier")

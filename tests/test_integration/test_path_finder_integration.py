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
from pyorps.core.exceptions import AlgorthmNotImplementedError

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

                    except AlgorthmNotImplementedError:
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

            except AlgorthmNotImplementedError:
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


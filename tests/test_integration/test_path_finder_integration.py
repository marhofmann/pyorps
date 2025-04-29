import unittest
import os
import tempfile
import geopandas as gpd
from shapely.geometry import Polygon

from pyorps import PathFinder
from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.raster.handler import create_test_tiff
from pyorps.io.geo_dataset import initialize_geo_dataset, LocalRasterDataset


class TestPathFinderIntegration(unittest.TestCase):
    """Integration tests for PathFinder class."""

    @classmethod
    def setUpClass(cls):
        """Create test data that can be reused across tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_raster_path = os.path.join(cls.temp_dir.name, "test_raster.tiff")

        # Create a test raster file
        cls.raster_data = create_test_tiff(
            cls.test_raster_path,
            width=100,
            height=100,
            pattern="gradient"
        )

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

    def test_raster_path_finding_with_different_graph_libraries(self):
        """Test path finding with different graph libraries using raster data."""
        for lib in ["networkit", "networkit", "igraph", "rustworkx"]:
            try:
                path_finder = PathFinder(
                    dataset_source=self.test_raster_path,
                    source_coords=self.source_coords,
                    target_coords=self.target_coords,
                    graph_api=lib,
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
            except ImportError:
                # Skip test if library not installed
                print(f"Skipping test for {lib} - library not installed")

    def test_vector_path_finding(self):
        """Test path finding using vector data with cost assumptions."""
        path_finder = PathFinder(
            dataset_source=self.test_vector_path,
            source_coords=self.source_coords,
            target_coords=self.target_coords,
            graph_api='networkit',
            search_space_buffer_m=50,
            neighborhood_str='r1',
            cost_assumptions=self.cost_assumptions
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

        # Check if path metrics are calculated
        self.assertIsNotNone(path.total_cost)
        self.assertIsNotNone(path.length_by_category)
        self.assertIsNotNone(path.length_by_category_percent)

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

    def test_path_finding_with_different_algorithms(self):
        """Test path finding with different routing algorithms."""
        algorithms = ["dijkstra", "bidirectional_dijkstra"]  # Skip astar as it needs a heuristic function
        for algorithm in algorithms:
            path_finder = PathFinder(
                dataset_source=self.test_raster_path,
                source_coords=self.source_coords,
                target_coords=self.target_coords,
                graph_api='networkit',
                search_space_buffer_m=50,
                neighborhood_str='r1',
            )
            path = path_finder.find_route(algorithm=algorithm)

            # Assert path was found
            self.assertIsNotNone(path)
            self.assertGreater(len(path.path_indices), 1)

            # Ensure path connects source to target
            self.assertAlmostEqual(path.path_coords[0][0], self.source_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[0][1], self.source_coords[1], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][0], self.target_coords[0], delta=5)
            self.assertAlmostEqual(path.path_coords[-1][1], self.target_coords[1], delta=5)

            # Check that the algorithm name is recorded correctly
            self.assertEqual(path.algorithm, algorithm)

    def test_multiple_source_target_path_finding(self):
        """Test path finding with multiple source and target points."""
        # Create two sets of source and target coordinates
        sources = [(500020, 5599980), (500030, 5599990)]
        targets = [(500080, 5599920), (500090, 5599910)]

        # Test with multiple sources, single target
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=sources,
            target_coords=targets[0],
            graph_api='networkit',
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )
        paths = path_finder.find_route()

        # Assert we got a list of paths
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 2)  # One path for each source

        # Test with single source, multiple targets
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=sources[0],
            target_coords=targets,
            graph_api='networkit',
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )
        paths = path_finder.find_route()

        # Assert we got a list of paths
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 2)  # One path for each target

        # Test with multiple sources, multiple targets (pairwise)
        path_finder = PathFinder(
            dataset_source=self.test_raster_path,
            source_coords=sources,
            target_coords=targets,
            graph_api='networkit',
            search_space_buffer_m=50,
            neighborhood_str='r1',
        )
        paths = path_finder.find_route(pairwise=True)

        # Assert we got a list of paths
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 2)  # One path for each source-target pair

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
        path = path_finder.find_route()

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

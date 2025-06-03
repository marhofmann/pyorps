import unittest
import os
import tempfile
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import random
from typing import Tuple, Optional

from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.io.geo_dataset import InMemoryVectorDataset, InMemoryRasterDataset
from pyorps.core.cost_assumptions import CostAssumptions


def create_random_polygon_dataset(
        num_polygons: int = 50,
        x_range: Tuple[float, float] = (0, 1000),
        y_range: Tuple[float, float] = (0, 1000),
        crs: str = "EPSG:32632",
        min_vertices: int = 3,
        max_vertices: int = 8,
        min_size: float = 20,
        max_size: float = 100,
        seed: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Create a random polygon GeoDataFrame with realistic features.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create empty lists to store data
    polygons = []
    cat_values = []
    subcat_values = []
    cost_values = []
    ids = []

    # Define categories and subcategories
    categories = ['forest', 'agriculture', 'water', 'barren']
    subcategories = {
        'forest': ['deciduous', 'coniferous', 'mixed'],
        'agriculture': ['cropland', 'pasture', 'orchard'],
        'water': ['river', 'lake', 'wetland'],
        'barren': ['rock', 'sand', 'gravel']
    }

    # Define costs for each category (simple model)
    costs = {
        'forest': 50,
        'agriculture': 20,
        'water': 100,
        'barren': 10
    }

    # Generate random polygons
    for i in range(num_polygons):
        # Choose a center point
        center_x = random.uniform(x_range[0] + max_size, x_range[1] - max_size)
        center_y = random.uniform(y_range[0] + max_size, y_range[1] - max_size)
        center = Point(center_x, center_y)

        # Determine size and number of vertices
        size = random.uniform(min_size, max_size)
        num_vertices = random.randint(min_vertices, max_vertices)

        # Generate vertices around the center
        angles = sorted(np.random.uniform(0, 2 * np.pi, num_vertices))
        radii = np.random.uniform(0.5 * size, size, num_vertices)

        # Create polygon vertices
        vertices = [(center.x + radii[i] * np.cos(angles[i]),
                     center.y + radii[i] * np.sin(angles[i]))
                    for i in range(num_vertices)]

        # Create polygon
        poly = Polygon(vertices)
        if poly.is_valid:
            # Randomly choose a category and subcategory
            category = random.choice(categories)
            subcat = random.choice(subcategories.get(category, ['default']))

            # Add polygon and associated data
            polygons.append(poly)
            cat_values.append(category)
            subcat_values.append(subcat)
            cost_values.append(costs[category])  # Assign cost directly
            ids.append(i)

    # Create GeoDataFrame
    data = {
        'id': ids,
        'category': cat_values,
        'subcategory': subcat_values,
        'cost': cost_values,  # Include cost column directly
        'geometry': polygons
    }

    gdf = gpd.GeoDataFrame(data, crs=crs)
    return gdf


class TestGeoRasterizerIntegration(unittest.TestCase):
    """Integration test for the GeoRasterizer class with random polygon data."""

    def setUp(self):
        """Set up test data with random polygons."""
        # Create random polygon dataset
        self.random_gdf = create_random_polygon_dataset(
            num_polygons=30,
            x_range=(0, 500),
            y_range=(0, 500),
            seed=42  # For reproducibility
        )

        # Create vector dataset
        self.vector_dataset = InMemoryVectorDataset(self.random_gdf, crs="EPSG:32632")

        # Create simple cost assumptions for testing
        self.cost_dict = {
            'forest': 50,
            'agriculture': 20,
            'water': 100,
            'barren': 10
        }
        self.cost_assumptions = CostAssumptions({'category': self.cost_dict})

        # Create a simple raster for testing
        self.raster_data = np.ones((10, 10), dtype=np.uint16)  # 2D array
        from rasterio.transform import from_bounds
        minx, miny, maxx, maxy = self.random_gdf.total_bounds
        self.transform = from_bounds(minx, miny, maxx, maxy, 10, 10)

        self.raster_dataset = InMemoryRasterDataset(
            self.raster_data,
            "EPSG:32632",
            self.transform
        )

    def test_rasterize_with_cost_field(self):
        """Test rasterizing using the cost field."""
        # Initialize rasterizer with vector data
        rasterizer = GeoRasterizer(
            self.vector_dataset,
            self.cost_assumptions
        )

        # Test rasterization using the existing cost field
        raster_result = rasterizer.rasterize(
            field_name='cost',
            resolution_in_m=5.0,
            fill_value=65535
        )

        # Check raster metadata
        self.assertIsNotNone(raster_result)
        self.assertIsNotNone(rasterizer.raster)
        self.assertIsNotNone(rasterizer.transform)

        # Check raster dimensions
        self.assertEqual(len(rasterizer.raster.shape), 2)  # Should be 2D

        # Verify the raster has expected values
        self.assertTrue(10 in np.unique(rasterizer.raster) or
                        20 in np.unique(rasterizer.raster) or
                        50 in np.unique(rasterizer.raster) or
                        100 in np.unique(rasterizer.raster))

        # Save and verify file output
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            rasterizer.save_raster(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_buffer_and_modify_raster(self):
        """Test creating buffers and modifying the raster."""
        # Initialize rasterizer with the raster dataset
        rasterizer = GeoRasterizer(
            self.raster_dataset,
            self.cost_assumptions
        )

        # Select a subset of data for modification
        if 'forest' in self.random_gdf['category'].values:
            forest_data = self.random_gdf[self.random_gdf['category'] == 'forest']
        else:
            # Use first category if forest doesn't exist
            first_cat = self.random_gdf['category'].iloc[0]
            forest_data = self.random_gdf[self.random_gdf['category'] == first_cat]

        if not forest_data.empty:
            # Create buffered version of the data
            buffered_data = rasterizer.create_buffer(forest_data, 10, inplace=False)

            # Check that buffer increased the area
            self.assertGreater(buffered_data.geometry.area.sum(), forest_data.geometry.area.sum())

            # Modify the raster directly with GeoDataFrame
            modified_raster = rasterizer.modify_raster_with_geodataframe(
                buffered_data,  # Use GeoDataFrame directly
                value=200,  # High cost value
                multiply=False  # Set to this value
            )

            # Check that the raster was modified
            self.assertIsNotNone(modified_raster)

    def test_shrink_raster(self):
        """Test shrinking raster by removing outer bounds."""
        # Create a raster with excluded values around the edges
        raster_data = np.ones((10, 10), dtype=np.uint16)  # 2D array

        # Set edges to exclude value
        exclude_value = 65535
        raster_data[:2, :] = exclude_value  # Top rows
        raster_data[-2:, :] = exclude_value  # Bottom rows
        raster_data[:, :2] = exclude_value  # Left columns
        raster_data[:, -2:] = exclude_value  # Right columns

        # Create a raster dataset
        raster_dataset = InMemoryRasterDataset(
            raster_data,
            "EPSG:32632",
            self.transform
        )

        # Create rasterizer with this dataset
        rasterizer = GeoRasterizer(
            raster_dataset,
            self.cost_assumptions
        )

        # Test shrinking the raster
        result = rasterizer.shrink_raster(exclude_value=exclude_value)

        # Check the result is smaller than original
        self.assertEqual(result.shape, (6, 6))

    def test_modify_with_gdf_zoning(self):
        """Test modifying raster with zoning in GeoDataFrames."""
        # Initialize rasterizer with the raster dataset
        rasterizer = GeoRasterizer(
            self.raster_dataset,
            self.cost_assumptions
        )

        # Create zoned data
        zoning_gdf = self.random_gdf.copy()
        zoning_gdf['zone'] = 'regular'

        # Mark water areas as forbidden if they exist
        if 'water' in zoning_gdf['category'].values:
            water_mask = zoning_gdf['category'] == 'water'
            if water_mask.any():
                zoning_gdf.loc[water_mask, 'zone'] = 'forbidden'

        # Process regular zones
        regular_zones = zoning_gdf[zoning_gdf['zone'] == 'regular']
        if not regular_zones.empty:
            rasterizer.modify_raster_with_geodataframe(
                regular_zones,
                value=5,
                multiply=True  # Multiply existing values by 5
            )

        # Process forbidden zones
        forbidden_zones = zoning_gdf[zoning_gdf['zone'] == 'forbidden']
        if not forbidden_zones.empty:
            rasterizer.modify_raster_with_geodataframe(
                forbidden_zones,
                value=65535  # Set to forbidden value
            )

            # Check that forbidden values were set
            self.assertTrue(np.any(rasterizer.raster == 65535))
        else:
            # If no forbidden zones, just check regular zones were modified
            self.assertTrue(np.any(rasterizer.raster == 5))


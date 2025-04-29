import unittest
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

from pyorps.core.types import (
    InputDataType, CostAssumptionsType, BboxType, GeometryMaskType
)
from pyorps.core.cost_assumptions import CostAssumptions


class TestTypes(unittest.TestCase):
    def test_input_data_type_annotations(self):
        """Test that different types can be assigned to InputDataType variables."""
        # This is mainly a type hint test, but we can verify some basic functionality

        # String path
        path_str: InputDataType = "path/to/file.geojson"
        self.assertEqual(path_str, "path/to/file.geojson")

        # Dictionary
        data_dict: InputDataType = {"key": "value"}
        self.assertEqual(data_dict["key"], "value")

        # GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        gdf_data: InputDataType = gdf
        self.assertIsInstance(gdf_data, gpd.GeoDataFrame)

        # NumPy array
        arr = np.array([1, 2, 3])
        array_data: InputDataType = arr
        self.assertTrue(np.array_equal(array_data, [1, 2, 3]))

    def test_cost_assumptions_type_annotations(self):
        """Test that different types can be assigned to CostAssumptionsType variables."""
        # String path
        path_str: CostAssumptionsType = "path/to/costs.json"
        self.assertEqual(path_str, "path/to/costs.json")

        # Dictionary
        costs_dict: CostAssumptionsType = {"feature": {"value": 1.0}}
        self.assertEqual(costs_dict["feature"]["value"], 1.0)

        # CostAssumptions instance
        ca = CostAssumptions({"feature": {"value": 1.0}})
        ca_instance: CostAssumptionsType = ca
        self.assertIsInstance(ca_instance, CostAssumptions)

    def test_bbox_type_annotations(self):
        """Test that different types can be assigned to BboxType variables."""
        # Polygon
        polygon = Polygon.from_bounds(0, 0, 1, 1)
        poly_bbox: BboxType = polygon
        self.assertIsInstance(poly_bbox, Polygon)

        # GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[Polygon.from_bounds(0, 0, 1, 1)])
        gdf_bbox: BboxType = gdf
        self.assertIsInstance(gdf_bbox, gpd.GeoDataFrame)

        # Tuple
        bbox_tuple: BboxType = (0, 0, 1, 1)
        self.assertEqual(bbox_tuple, (0, 0, 1, 1))


import unittest
import numpy as np
from shapely.geometry import Polygon, Point, MultiPoint
import geopandas as gpd

from pyorps.core.types import (
    InputDataType, CostAssumptionsType, BboxType, GeometryMaskType, CoordinateTuple, CoordinateList, CoordinateInput, NormalizedCoordinate
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

    def test_geometry_mask_type_annotations(self):
        """Test that different types can be assigned to GeometryMaskType variables."""
        # Polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygon_mask: GeometryMaskType = polygon
        self.assertIsInstance(polygon_mask, Polygon)
        self.assertTrue(polygon_mask.is_valid)

        # GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        gdf_mask: GeometryMaskType = gdf
        self.assertIsInstance(gdf_mask, gpd.GeoDataFrame)
        self.assertEqual(len(gdf_mask), 1)

        # Tuple (representing bounds: minx, miny, maxx, maxy)
        bounds_tuple: GeometryMaskType = (0, 0, 1, 1)
        self.assertEqual(bounds_tuple, (0, 0, 1, 1))

        # Tuple (representing polygon coordinates)
        coords_tuple: GeometryMaskType = ((0, 0), (1, 0), (1, 1), (0, 1))
        self.assertEqual(len(coords_tuple), 4)
        self.assertEqual(coords_tuple[0], (0, 0))

    def test_coord_type(self):
        """Test that Coord type works correctly with coordinates."""
        # Create a coordinate tuple that should match Coord type
        coord: CoordinateTuple = (10.5, 20.3)
        self.assertEqual(len(coord), 2)
        self.assertEqual(coord[0], 10.5)
        self.assertEqual(coord[1], 20.3)

        # Integer coordinates should work (implicit conversion to float)
        coord_int: CoordinateTuple = (1, 2)
        self.assertEqual(coord_int, (1, 2))

        # Mixed types should work
        coord_mixed: CoordinateTuple = (1, 2.5)
        self.assertEqual(coord_mixed, (1, 2.5))

    def test_coordlist_type(self):
        """Test that CoordList type works correctly with lists of coordinates."""
        # Create a list of coordinate tuples that should match CoordList type
        coords: CoordinateList = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        self.assertEqual(len(coords), 3)
        self.assertEqual(coords[0], (1.0, 2.0))
        self.assertEqual(coords[1], (3.0, 4.0))
        self.assertEqual(coords[2], (5.0, 6.0))

        # Integer coordinates should work
        coords_int: CoordinateList = [(1, 2), (3, 4)]
        self.assertEqual(coords_int, [(1, 2), (3, 4)])

        # Empty list should work
        coords_empty: CoordinateList = []
        self.assertEqual(len(coords_empty), 0)

    def test_coordinate_input_types(self):
        """Test that various types can be assigned to CoordinateInput variables."""
        # Single coordinate tuple
        coord_input: CoordinateInput = (10.5, 20.3)
        self.assertEqual(coord_input, (10.5, 20.3))

        # List of coordinate tuples
        coord_input = [(1.0, 2.0), (3.0, 4.0)]
        self.assertEqual(coord_input, [(1.0, 2.0), (3.0, 4.0)])

        # List of coordinate lists
        coord_input = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(coord_input, [[5.0, 6.0], [7.0, 8.0]])

        # NumPy array
        coord_input = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(coord_input, np.array([[1.0, 2.0], [3.0, 4.0]]))

        # Shapely Point
        coord_input = Point(10.5, 20.3)
        self.assertEqual((coord_input.x, coord_input.y), (10.5, 20.3))

        # Shapely MultiPoint
        coord_input = MultiPoint([(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(len(coord_input.geoms), 2)

        # GeoSeries
        points = [Point(1.0, 2.0), Point(3.0, 4.0)]
        coord_input = gpd.GeoSeries(points)
        self.assertEqual(len(coord_input), 2)

        # GeoDataFrame
        coord_input = gpd.GeoDataFrame(geometry=points)
        self.assertEqual(len(coord_input), 2)

    def test_coordinate_output_types(self):
        """Test that both possible output types work with CoordinateOutput."""
        # Single coordinate tuple
        coord_output: NormalizedCoordinate = (10.5, 20.3)
        self.assertEqual(coord_output, (10.5, 20.3))

        # List of coordinate tuples
        coord_output = [(1.0, 2.0), (3.0, 4.0)]
        self.assertEqual(coord_output, [(1.0, 2.0), (3.0, 4.0)])

    def test_list_tuple_float_type(self):
        """Test that list[tuple[float]] works in CoordinateInput."""
        # List of tuples with float values
        coords: CoordinateInput = [(1.0, 2.0), (3.0, 4.0)]
        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[0], (1.0, 2.0))
        self.assertEqual(coords[1], (3.0, 4.0))

        # Mixed integer and float values
        coords: CoordinateInput = [(1, 2.5), (3.0, 4)]
        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[0], (1, 2.5))
        self.assertEqual(coords[1], (3.0, 4))

    def test_list_point_type(self):
        """Test that list[Point] works in CoordinateInput."""
        # List of Point objects
        point1 = Point(1.0, 2.0)
        point2 = Point(3.0, 4.0)
        points: CoordinateInput = [point1, point2]

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].x, 1.0)
        self.assertEqual(points[0].y, 2.0)
        self.assertEqual(points[1].x, 3.0)
        self.assertEqual(points[1].y, 4.0)

        # Mixed integer and float coordinates
        point3 = Point(5, 6.5)
        points: CoordinateInput = [point3]
        self.assertEqual(points[0].x, 5.0)
        self.assertEqual(points[0].y, 6.5)

    def test_list_multipoint_type(self):
        """Test that list[MultiPoint] works in CoordinateInput."""
        # List of MultiPoint objects
        mp1 = MultiPoint([(1.0, 2.0), (3.0, 4.0)])
        mp2 = MultiPoint([(5.0, 6.0), (7.0, 8.0)])
        multipoints: CoordinateInput = [mp1, mp2]

        self.assertEqual(len(multipoints), 2)
        self.assertEqual(len(multipoints[0].geoms), 2)
        self.assertEqual(multipoints[0].geoms[0].x, 1.0)
        self.assertEqual(multipoints[0].geoms[0].y, 2.0)
        self.assertEqual(multipoints[0].geoms[1].x, 3.0)
        self.assertEqual(multipoints[0].geoms[1].y, 4.0)
        self.assertEqual(len(multipoints[1].geoms), 2)

        # Mixed integer and float coordinates
        mp3 = MultiPoint([(1, 2), (3, 4.5)])
        multipoints: CoordinateInput = [mp3]
        self.assertEqual(len(multipoints[0].geoms), 2)
        self.assertEqual(multipoints[0].geoms[0].x, 1.0)
        self.assertEqual(multipoints[0].geoms[0].y, 2.0)
        self.assertEqual(multipoints[0].geoms[1].x, 3.0)
        self.assertEqual(multipoints[0].geoms[1].y, 4.5)

    def test_geometry_mask_type_comprehensive(self):
        """Test that GeometryMaskType handles all expected input types."""
        # Simple polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mask: GeometryMaskType = polygon
        self.assertIsInstance(mask, Polygon)
        self.assertTrue(mask.is_valid)
        self.assertEqual(mask.area, 1.0)

        # Complex polygon with a hole
        polygon_with_hole = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],  # Exterior
            [[(3, 3), (7, 3), (7, 7), (3, 7)]]  # Interior (hole)
        )
        mask: GeometryMaskType = polygon_with_hole
        self.assertIsInstance(mask, Polygon)
        self.assertTrue(mask.is_valid)
        self.assertEqual(mask.area, 100 - 16)  # Total area minus hole

        # GeoDataFrame with a single polygon
        gdf = gpd.GeoDataFrame(geometry=[polygon])
        mask: GeometryMaskType = gdf
        self.assertIsInstance(mask, gpd.GeoDataFrame)
        self.assertEqual(len(mask), 1)
        self.assertEqual(mask.geometry.iloc[0].area, 1.0)

        # GeoDataFrame with multiple polygons
        gdf = gpd.GeoDataFrame(geometry=[polygon, polygon_with_hole])
        mask: GeometryMaskType = gdf
        self.assertIsInstance(mask, gpd.GeoDataFrame)
        self.assertEqual(len(mask), 2)
        self.assertEqual(mask.geometry.iloc[0].area, 1.0)
        self.assertEqual(mask.geometry.iloc[1].area, 84.0)

        # Simple bounds tuple
        bounds: GeometryMaskType = (0, 0, 10, 10)
        self.assertIsInstance(bounds, tuple)
        self.assertEqual(len(bounds), 4)
        self.assertEqual(bounds, (0, 0, 10, 10))

        # Polygon coordinates tuple
        coords_tuple: GeometryMaskType = ((0, 0), (1, 0), (1, 1), (0, 1))
        self.assertIsInstance(coords_tuple, tuple)
        self.assertEqual(len(coords_tuple), 4)
        self.assertEqual(coords_tuple[0], (0, 0))
        self.assertEqual(coords_tuple[-1], (0, 1))

        # Nested tuple (e.g., for multipolygon representation)
        complex_tuple: GeometryMaskType = (((0, 0), (10, 0), (10, 10), (0, 10)), ((2, 2), (8, 2), (8, 8), (2, 8)))
        self.assertIsInstance(complex_tuple, tuple)
        self.assertEqual(len(complex_tuple), 2)
        self.assertEqual(len(complex_tuple[0]), 4)
        self.assertEqual(complex_tuple[0][0], (0, 0))


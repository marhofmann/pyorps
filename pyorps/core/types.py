from typing import Union, TypeAlias

from shapely.geometry import Polygon, Point, MultiPoint
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries

from .cost_assumptions import CostAssumptions

# Input for geodata sources
InputDataType: TypeAlias = Union[
    # File path to a local file (vector or raster)
    str,
    # Dictionary containing a request for a geodata server (needs at least an 'url')
    dict,
    # GeoDataframe containing vector data to be rasterized
    GeoDataFrame,
    # GeoSeries containing geometries for Vectorization
    GeoSeries,
    # Numpy array containing raster data
    ndarray
]

CostAssumptionsType: TypeAlias = Union[
    # Dictionary containing attribute - cost pairs (or a nested dictionaries)
    dict,
    # File path to a local file (.csv, .xlsx, .xls, .json)
    str,
    # CostAssumptions object
    CostAssumptions
]

BboxType: TypeAlias = Union[
    # Rectangle as a Polygon
    Polygon,
    # GeoDataFrame containing a rectangle as a Polygon
    GeoDataFrame,
    # GeoSeries containing a rectangle as a Polygon
    GeoSeries,
    # Tuple defining (x-min, y-min, x-max, y-max)
    tuple[float, float, float, float]
]

GeometryMaskType: TypeAlias = Union[
    # Polygon as a mask (does not have to be a rectangle)
    Polygon,
    # GeoDataFrame containing one of multiple Polygons
    GeoDataFrame,
    # Tuple defining (x-min, y-min, x-max, y-max)
    tuple
]

Coord = tuple[float, float]             # Float pair of coordinates

CoordList = list[Coord]                 # List of float pairs of coordinates

CoordinateInput: TypeAlias = Union[
    # Float pair of coordinates
    Coord,
    # List of float pairs of coordinates
    CoordList,
    # List of float pairs of coordinates
    list[list[float]],
    # List of float pairs of coordinates
    list[tuple[float]],
    # List of float pairs of coordinates
    list[Point],
    # List of multiple float pairs of coordinates
    list[MultiPoint],
    # Array of float pairs of coordinates
    ndarray,
    # Shapely Point with pair of coordinates
    Point,
    # MultiPoint for multiple float pairs of coordinates
    MultiPoint,
    # GeoSeries with Point or MultiPoint
    GeoSeries,
    # GeoDataFrame with Point or MultiPoint
    GeoDataFrame
]

CoordinateOutput: TypeAlias = Union[
    # Uniform handling of single Point (float pair of coordinates)
    Coord,
    # Uniform handling of multiple Points (multiple float pairs of coordinates)
    CoordList
]

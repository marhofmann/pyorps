from typing import Union, TypeAlias

from shapely.geometry import Polygon, Point, MultiPoint
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries

from .cost_assumptions import CostAssumptions

# Input for geodata sources
InputDataType: TypeAlias = Union[
    str,                                # File path to a local file (vector or raster)
    dict,                               # Dictionary containing a request for a geodata server (needs at least an 'url')
    GeoDataFrame,                       # GeoDataframe containing vector data to be rasterized
    GeoSeries,                          # GeoSeries containing geometries for Vectorization
    ndarray                             # Numpy array containing raster data
]

CostAssumptionsType: TypeAlias = Union[
    dict,                               # Dictionary containing attribute - cost pairs (or a nested dictionaries)
    str,                                # File path to a local file (.csv, .xlsx, .xls, .json)
    CostAssumptions                     # CostAssumptions object
]

BboxType: TypeAlias = Union[
    Polygon,                            # Rectangle as a Polygon
    GeoDataFrame,                       # GeoDataFrame containing a rectangle as a Polygon
    GeoSeries,                          # GeoSeries containing a rectangle as a Polygon
    tuple[float, float, float, float]   # Tuple defining (x-min, y-min, x-max, y-max)
]

GeometryMaskType: TypeAlias = Union[
    Polygon,                            # Polygon as a mask (does not have to be a rectangle)
    GeoDataFrame,                       # GeoDataFrame containing one of multiple Polygons
    tuple                               # Tuple defining (x-min, y-min, x-max, y-max)
]

Coord = tuple[float, float]             # Float pair of coordinates

CoordList = list[Coord]                 # List of float pairs of coordinates

CoordinateInput: TypeAlias = Union[
    Coord,                              # Float pair of coordinates
    CoordList,                          # List of float pairs of coordinates
    list[list[float]],                  # List of float pairs of coordinates
    list[tuple[float]],                 # List of float pairs of coordinates
    list[Point],                        # List of float pairs of coordinates
    list[MultiPoint],                   # List of multiple float pairs of coordinates
    ndarray,                            # Array of float pairs of coordinates
    Point,                              # Shapely Point with pair of coordinates
    MultiPoint,                         # MultiPoint for multiple float pairs of coordinates
    GeoSeries,                          # GeoSeries with Point or MultiPoint
    GeoDataFrame                        # GeoDataFrame with Point or MultiPoint
]

CoordinateOutput: TypeAlias = Union[
    Coord,                              # Uniform handling of single Point (float pair of coordinates)
    CoordList                           # Uniform handling of multiple Points (multiple float pairs of coordinates)
]

from typing import Union, TypeAlias

from shapely.geometry import Polygon, Point, MultiPoint
from numpy import ndarray, int32, int64, uint32, uint64
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

# A pair of float coordinates
CoordinateTuple: TypeAlias = Union[tuple[float, float], list[float]]

# List of pairs of float coordinates
CoordinateList: TypeAlias = list[CoordinateTuple]

CoordinateInput: TypeAlias = Union[
    # Float a pair of coordinates
    CoordinateTuple,
    # List of float pairs of coordinates
    CoordinateList,
    # List of float pairs of coordinates
    list[Point],
    # List of multiple float pairs of coordinates
    list[MultiPoint],
    # Array of float pairs of coordinates
    ndarray,
    # Shapely Point with a pair of coordinates
    Point,
    # MultiPoint for multiple float pairs of coordinates
    MultiPoint,
    # GeoSeries with Point or MultiPoint
    GeoSeries,
    # GeoDataFrame with Point or MultiPoint
    GeoDataFrame
]

NormalizedCoordinate: TypeAlias = Union[
    # Uniform handling of single Point (a pair of float coordinates)
    CoordinateTuple,
    # Uniform handling of multiple Points (pairs of multiple float coordinates)
    CoordinateList
]

CoordinatePath: TypeAlias = Union[
    # A path of Coordinates
    list[CoordinateTuple],
    ndarray,
]

# A node in the graph
Node: TypeAlias = Union[int, int32, int64, uint32, uint64]

# A list of node indexes
NodeList: TypeAlias = Union[
    list[Node],
    ndarray[int],
]

SourceTargetType: TypeAlias = Union[
    # A single Node (int)
    Node,
    # or a List of Nodes list[int]
    NodeList,
]

# A list of multiple NodeList type objects
NodePathList: TypeAlias = list[NodeList]


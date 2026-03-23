"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
from typing import TypeAlias

from geopandas import GeoDataFrame, GeoSeries
from numpy import int32, int64, ndarray, uint32, uint64
from shapely.geometry import MultiPoint, Point, Polygon

from .cost_assumptions import CostAssumptions

# Input for geodata sources
InputDataType: TypeAlias = str | dict | GeoDataFrame | GeoSeries | ndarray

CostAssumptionsType: TypeAlias = dict | str | CostAssumptions

BboxType: TypeAlias = Polygon | GeoDataFrame | GeoSeries | tuple[float, float, float, float]

GeometryMaskType: TypeAlias = Polygon | GeoDataFrame | tuple

# A pair of float coordinates
CoordinateTuple: TypeAlias = tuple[float, float] | list[float]

# List of pairs of float coordinates
CoordinateList: TypeAlias = list[CoordinateTuple]

CoordinateInput: TypeAlias = CoordinateTuple | CoordinateList | list[Point] | list[MultiPoint] | ndarray | Point | MultiPoint | GeoSeries | GeoDataFrame

NormalizedCoordinate: TypeAlias = CoordinateTuple | CoordinateList

CoordinatePath: TypeAlias = list[CoordinateTuple] | ndarray

# A node in the graph
Node: TypeAlias = int | int32 | int64 | uint32 | uint64

# A list of node indexes
NodeList: TypeAlias = list[Node] | ndarray[int]

SourceTargetType: TypeAlias = Node | NodeList

# A list of multiple NodeList type objects
NodePathList: TypeAlias = list[NodeList]

# Maximum cost value for uint16 rasters — cells with this value are impassable
IMPASSABLE_CELL_COST: int = 65535


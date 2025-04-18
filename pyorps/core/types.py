from typing import Union, TypeAlias
from shapely.geometry import Polygon
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries

from .cost_assumptions import CostAssumptions

InputDataType: TypeAlias = Union[str, dict, GeoDataFrame, GeoSeries, ndarray]
CostAssumptionsType: TypeAlias = Union[dict, str, CostAssumptions]
BboxType: TypeAlias = Union[Polygon, GeoDataFrame, GeoSeries, tuple[float, float, float, float]]
GeometryMaskType: TypeAlias = Union[Polygon, GeoDataFrame, tuple]

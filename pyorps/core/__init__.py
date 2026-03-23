"""Core types and base classes for geospatial data processing."""

from .cost_assumptions import (
                               CostAssumptions,
                               detect_feature_columns,
                               get_zero_cost_assumptions,
                               save_empty_cost_assumptions,
)
from .exceptions import (
                               AlgorithmNotImplementedError,
                               ColumnAnalysisError,
                               # Cost assumption exceptions
                               CostAssumptionsError,
                               FeatureColumnError,
                               FileLoadError,
                               FormatError,
                               InvalidSourceError,
                               NoPathFoundError,
                               NoSuitableColumnsError,
                               PairwiseError,
                               # Base exception
                               PyorpsError,
                               # Graph API exceptions
                               RasterShapeError,
                               WFSConnectionError,
                               # WFS exceptions
                               WFSError,
                               WFSLayerNotFoundError,
                               WFSResponseParsingError,
)
from .path import Path, PathCollection
from .types import (
                               IMPASSABLE_CELL_COST,
                               BboxType,
                               CoordinateInput,
                               CoordinateList,
                               CoordinateTuple,
                               CostAssumptionsType,
                               GeometryMaskType,
                               InputDataType,
                               NormalizedCoordinate,
)

__all__ = [
    # Cost assumptions
    "CostAssumptions", "get_zero_cost_assumptions", "detect_feature_columns", "save_empty_cost_assumptions",

    # Types
    "InputDataType", "CostAssumptionsType", "BboxType", "GeometryMaskType",
    "CoordinateTuple", "CoordinateList", "CoordinateInput",
    "NormalizedCoordinate", "IMPASSABLE_CELL_COST",

    # Path classes
    "Path", "PathCollection",

    # Exceptions - Base
    "PyorpsError",

    # Exceptions - Cost assumptions
    "CostAssumptionsError", "FileLoadError", "InvalidSourceError", "FormatError",
    "FeatureColumnError", "NoSuitableColumnsError", "ColumnAnalysisError",

    # Exceptions - WFS
    "WFSError", "WFSConnectionError", "WFSResponseParsingError", "WFSLayerNotFoundError",

    # Exceptions - Graph API
    "RasterShapeError", "NoPathFoundError", "AlgorithmNotImplementedError",
    "PairwiseError"
]

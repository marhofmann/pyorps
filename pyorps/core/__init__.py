"""Core types and base classes for geospatial data processing."""

from .cost_assumptions import (
                               CostAssumptions,
                               detect_feature_columns,
                               get_zero_cost_assumptions,
                               save_empty_cost_assumptions,
)
from .ensemble import EnsembleError, RouteEnsemble
from .exceptions import (
                               AlgorithmNotImplementedError,
                               ColumnAnalysisError,
                               ConsistencyCheckError,
                               # Cost assumption exceptions
                               CostAssumptionsError,
                               CrsReprojectionWarning,
                               FeatureColumnError,
                               FeatureNotImplementedError,
                               FileLoadError,
                               FormatError,
                               InputFileNotFoundError,
                               InsufficientMemoryError,
                               # Replacements for builtin exceptions
                               InvalidInputError,
                               InvalidSourceError,
                               InvalidTypeError,
                               # Metric stack exceptions
                               MetricStackError,
                               MissingDependencyError,
                               MissingKeyError,
                               NoPathFoundError,
                               NoSuitableColumnsError,
                               NumericalPrecisionError,
                               # Objective exceptions
                               ObjectiveError,
                               PairwiseError,
                               # Base exception
                               PyorpsError,
                               PyorpsRuntimeError,
                               # Warning root
                               PyorpsWarning,
                               # Graph API exceptions
                               RasterShapeError,
                               WFSConnectionError,
                               # WFS exceptions
                               WFSError,
                               WFSLayerNotFoundError,
                               WFSRequestRejectedError,
                               WFSResponseParsingError,
                               WFSServerError,
)
from .metric_stack import CombineResult, MetricStack
from .objective import GradientLUTs, GradientOptions, Objective
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
from .warning_types import AlgorithmFallbackWarning

__all__ = [
    # Cost assumptions
    "CostAssumptions", "get_zero_cost_assumptions", "detect_feature_columns", "save_empty_cost_assumptions",

    # Feasibility objective
    "Objective", "GradientOptions", "GradientLUTs", "ObjectiveError",

    # Metric stack
    "MetricStack", "MetricStackError", "CombineResult",

    # Route ensembles
    "RouteEnsemble", "EnsembleError",

    # Types
    "InputDataType", "CostAssumptionsType", "BboxType", "GeometryMaskType",
    "CoordinateTuple", "CoordinateList", "CoordinateInput",
    "NormalizedCoordinate", "IMPASSABLE_CELL_COST",

    # Path classes
    "Path", "PathCollection",

    # Exceptions - Base (catch this for anything the library reports)
    "PyorpsError",

    # Exceptions - Cost assumptions
    "CostAssumptionsError", "FileLoadError", "InvalidSourceError", "FormatError",
    "FeatureColumnError", "NoSuitableColumnsError", "ColumnAnalysisError",

    # Exceptions - WFS
    "WFSError", "WFSConnectionError", "WFSResponseParsingError", "WFSLayerNotFoundError",
    "WFSServerError", "WFSRequestRejectedError",

    # Exceptions - Graph API
    "RasterShapeError", "NoPathFoundError", "AlgorithmNotImplementedError",
    "PairwiseError",

    # Exceptions - replacements for builtin exceptions (each also derives from its builtin)
    "InvalidInputError", "InvalidTypeError", "MissingKeyError",
    "MissingDependencyError", "FeatureNotImplementedError",
    "PyorpsRuntimeError", "InsufficientMemoryError", "InputFileNotFoundError",
    "ConsistencyCheckError", "NumericalPrecisionError",

    # Warning root
    "PyorpsWarning", "CrsReprojectionWarning", "AlgorithmFallbackWarning",
]

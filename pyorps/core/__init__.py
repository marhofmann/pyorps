"""Core types and base classes for geospatial data processing."""

from .cost_assumptions import CostAssumptions
from .types import (
    InputDataType, CostAssumptionsType, BboxType, GeometryMaskType
)
from .path import Path, PathCollection
from .exceptions import (
    # Cost assumption exceptions
    CostAssumptionsError, FileLoadError, InvalidSourceError, FormatError,
    FeatureColumnError, NoSuitableColumnsError, ColumnAnalysisError,
    # WFS exceptions
    WFSError, WFSConnectionError, WFSResponseParsingError, WFSLayerNotFoundError,
    # Graph API exceptions
    RasterShapeError, NoPathFoundError, AlgorthmNotImplementedError
)

__all__ = [
    # Cost assumptions
    "CostAssumptions",

    # Types
    "InputDataType", "CostAssumptionsType", "BboxType", "GeometryMaskType",

    # Path classes
    "Path", "PathCollection",

    # Exceptions - Cost assumptions
    "CostAssumptionsError", "FileLoadError", "InvalidSourceError", "FormatError",
    "FeatureColumnError", "NoSuitableColumnsError", "ColumnAnalysisError",

    # Exceptions - WFS
    "WFSError", "WFSConnectionError", "WFSResponseParsingError", "WFSLayerNotFoundError",

    # Exceptions - Graph API
    "RasterShapeError", "NoPathFoundError", "AlgorthmNotImplementedError"
]

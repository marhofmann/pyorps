"""
Exceptions for CostAssumptions
"""


class CostAssumptionsError(Exception):
    """
    Base exception for CostAssumptions class.
    """
    pass


class FileLoadError(CostAssumptionsError):
    """
    Exception raised when loading files fails.
    """
    pass


class InvalidSourceError(CostAssumptionsError):
    """
    Exception raised when the provided source is invalid.
    """
    pass


class FormatError(CostAssumptionsError):
    """
    Exception raised when data format is invalid.
    """
    pass


class FeatureColumnError(Exception):
    """
    Base exception for feature column detection errors
    """
    pass


class NoSuitableColumnsError(FeatureColumnError):
    """
    Exception raised when no suitable columns are found
    """
    pass


class ColumnAnalysisError(FeatureColumnError):
    """
    Exception raised when column analysis fails
    """
    pass


"""
Exceptions for vector_loader
"""


class WFSError(Exception):
    """
    Base exception for WFS-related errors.
    """
    pass


class WFSConnectionError(WFSError):
    """
    Exception raised for connection issues with WFS services.
    """
    pass


class WFSResponseParsingError(WFSError):
    """
    Exception raised when parsing WFS responses fails.
    """
    pass


class WFSLayerNotFoundError(WFSError):
    """
    Exception raised when a requested layer cannot be found.
    """
    pass


"""
Exceptions for graph library API
"""


class RasterShapeError(Exception):
    """
    Custom exception if the raster shape is not supported
    """
    def __init__(self, raster_shape: tuple[int, ...]) -> None:
        message = f"Raster shape of {raster_shape} not supported! Only 2D (n, m) or 3D (n, m, 2) supported!"
        super().__init__(message)


class NoPathFoundError(Exception):
    """
    Custom exception if no path can be found in the graph for source and target
    """
    def __init__(self, source: int, target: int) -> None:
        message = f"No path found from {source} to {target}! Choose different source and target or increase buffer!"
        super().__init__(message)


class AlgorthmNotImplementedError(Exception):
    """
    Custom exception if a specific algorithm is not implemented in the API or the graph library
    """
    def __init__(self, algorithm: str, graph_library: str) -> None:
        message = f"Algorithm {algorithm} for {graph_library} not supported!"
        super().__init__(message)

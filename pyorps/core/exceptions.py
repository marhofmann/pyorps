"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland

Exceptions of pyorps.

Catch ``PyorpsError`` for anything the library reports. Every exception class
pyorps defines derives from it. The classes that replace a builtin exception
(``ValueError``, ``TypeError``, ``KeyError``, ``ImportError``,
``NotImplementedError``, ``RuntimeError``, ``MemoryError``,
``FileNotFoundError``) also derive from that builtin, so code that catches
``ValueError`` keeps working.

Hierarchy (a name in brackets is the builtin the class also derives from)::

    PyorpsError
    +-- CostAssumptionsError
    |   +-- FileLoadError
    |   +-- InvalidSourceError
    |   +-- FormatError
    +-- ObjectiveError
    +-- MetricStackError
    +-- FeatureColumnError
    |   +-- NoSuitableColumnsError
    |   +-- ColumnAnalysisError
    +-- WFSError
    |   +-- WFSConnectionError
    |   +-- WFSResponseParsingError
    |   +-- WFSLayerNotFoundError
    |   +-- WFSServerError
    |   +-- WFSRequestRejectedError
    +-- RasterShapeError
    +-- NoPathFoundError
    +-- AlgorithmNotImplementedError
    +-- PairwiseError
    +-- EnsembleError                      (pyorps.core.ensemble)
    +-- InvalidInputError                  [ValueError]
    |   +-- ThinForbiddenFeatureError      (pyorps.raster.thinness)
    |   +-- SealedOpeningError             (pyorps.raster.thinness)
    |   +-- ProvenanceMismatch             (pyorps.io.provenance)
    +-- InvalidTypeError                   [TypeError]
    +-- MissingKeyError                    [KeyError]
    +-- MissingDependencyError             [ImportError]
    +-- FeatureNotImplementedError         [NotImplementedError]
    +-- PyorpsRuntimeError                 [RuntimeError]
    +-- InsufficientMemoryError            [MemoryError]
    +-- InputFileNotFoundError             [FileNotFoundError]

    PyorpsWarning                          [UserWarning]
    +-- ThinForbiddenFeatureWarning        (pyorps.raster.thinness)
    +-- ForbiddenBurnRoutingAssessment     (pyorps.raster.thinness)
    +-- SealedOpeningWarning               (pyorps.raster.thinness)
    +-- CrsReprojectionWarning             (pyorps.core.warning_types)
"""

from .warning_types import CrsReprojectionWarning, PyorpsWarning

__all__ = [
    # Root
    "PyorpsError",
    # Cost assumptions, objective, metric stack, feature columns
    "CostAssumptionsError", "FileLoadError", "InvalidSourceError", "FormatError",
    "ObjectiveError", "MetricStackError",
    "FeatureColumnError", "NoSuitableColumnsError", "ColumnAnalysisError",
    # WFS
    "WFSError", "WFSConnectionError", "WFSResponseParsingError",
    "WFSLayerNotFoundError", "WFSServerError", "WFSRequestRejectedError",
    # Graph API
    "RasterShapeError", "NoPathFoundError", "AlgorithmNotImplementedError",
    "PairwiseError",
    # Replacements for builtin exceptions (each also derives from its builtin)
    "InvalidInputError", "InvalidTypeError", "MissingKeyError",
    "MissingDependencyError", "FeatureNotImplementedError",
    "PyorpsRuntimeError", "InsufficientMemoryError", "InputFileNotFoundError",
    # Warnings
    "PyorpsWarning", "CrsReprojectionWarning",
]


class PyorpsError(Exception):
    """Base exception for all pyorps errors."""


class CostAssumptionsError(PyorpsError):
    """
    Base exception for CostAssumptions class.
    """


class FileLoadError(CostAssumptionsError):
    """
    Exception raised when loading files fails.
    """


class InvalidSourceError(CostAssumptionsError):
    """
    Exception raised when the provided source is invalid.
    """


class FormatError(CostAssumptionsError):
    """
    Exception raised when data format is invalid.
    """


class ObjectiveError(PyorpsError):
    """
    Exception raised for invalid feasibility-objective configuration
    (weights, gradient responses, LUT construction).
    """


class MetricStackError(PyorpsError):
    """
    Exception raised for metric-stack violations (band misalignment,
    missing layers, invalid values, I/O format problems).
    """


class FeatureColumnError(PyorpsError):
    """
    Base exception for feature column detection errors
    """


class NoSuitableColumnsError(FeatureColumnError):
    """
    Exception raised when no suitable columns are found
    """


class ColumnAnalysisError(FeatureColumnError):
    """
    Exception raised when column analysis fails
    """


# Exceptions for vector_loader


class WFSError(PyorpsError):
    """
    Base exception for WFS-related errors.
    """


class WFSConnectionError(WFSError):
    """
    Exception raised for connection issues with WFS services.
    """


class WFSResponseParsingError(WFSError):
    """
    Exception raised when parsing WFS responses fails.
    """


class WFSLayerNotFoundError(WFSError):
    """
    Exception raised when a requested layer cannot be found.
    """


class WFSServerError(WFSError):
    """
    Exception raised when the server accepted a request but failed to
    process it (e.g. an OGC exception report with ``InternalServerError``).
    Retrying a smaller request may succeed.
    """


class WFSRequestRejectedError(WFSError):
    """
    Exception raised when the server rejected a request as invalid (e.g.
    ``InvalidParameterValue``). Splitting the request cannot help.
    """


# Exceptions for graph library API


class RasterShapeError(PyorpsError):
    """
    Custom exception if the raster shape is not supported
    """
    def __init__(self, raster_shape: tuple[int, ...]) -> None:
        self.raster_shape = raster_shape
        message = (f"Raster shape of {raster_shape} not supported! "
                   f"Only 2D (n, m) or 3D (n, m, 2) supported!")
        super().__init__(message)

    def __reduce__(self):
        # Exception pickles as cls(*args) with args == (message,), which this
        # constructor does not accept; rebuild from the real arguments.
        return (type(self), (self.raster_shape,), self.__dict__ or None)


class NoPathFoundError(PyorpsError):
    """
    Custom exception if no path can be found in the graph for source and target
    """
    def __init__(self, source: int, target: int, add_message: str = '') -> None:
        self.source = source
        self.target = target
        self.add_message = add_message
        message = (f"No path found from {source} to {target}! Choose different "
                   f"source and target or increase buffer!")
        message = message + add_message
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (self.source, self.target, self.add_message),
                self.__dict__ or None)


class AlgorithmNotImplementedError(PyorpsError):
    """
    Custom exception if a specific algorithm is not implemented in the API or the graph
    library
    """
    def __init__(self, algorithm: str, graph_library: str) -> None:
        self.algorithm = algorithm
        self.graph_library = graph_library
        message = f"Algorithm {algorithm} for {graph_library} not supported!"
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (self.algorithm, self.graph_library),
                self.__dict__ or None)


class PairwiseError(PyorpsError):
    """
    Custom exception if pairwise computation fails
    """
    def __init__(self) -> None:
        message = ("Pairwise computation failed! Source and target lists must have "
                   "the same length for pairwise computation!")
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (), self.__dict__ or None)


# Errors that replace a builtin exception. Each derives from PyorpsError AND
# from the builtin it stands for, so ``except PyorpsError`` and the old
# ``except ValueError`` (and so on) both catch it. They take the message as the
# only required argument, exactly like the builtin.


class InvalidInputError(PyorpsError, ValueError):
    """
    An argument has a value the library cannot work with (out of range, unknown
    name, wrong shape, inconsistent combination). Also a ``ValueError``.
    """


class InvalidTypeError(PyorpsError, TypeError):
    """
    An argument has the wrong type, or a keyword argument is not accepted. Also a
    ``TypeError``.
    """


class MissingKeyError(PyorpsError, KeyError):
    """
    A required key or label is not there (a config dict without a required key,
    an unknown route key or terminal label). Also a ``KeyError``, so ``str()``
    shows the quoted message like any ``KeyError``.
    """


class MissingDependencyError(PyorpsError, ImportError):
    """
    An optional package (a pyorps extra, or a GPU or compiled component) is not
    installed or cannot be imported. Also an ``ImportError``.

    ``extra`` names the pip extra that provides it (``pip install
    "pyorps[extra]"``), or is ``None`` when no extra applies. ``name`` and
    ``path`` are the ``ImportError`` attributes.
    """

    def __init__(self, *args, extra: str | None = None, name: str | None = None,
                 path: str | None = None) -> None:
        super().__init__(*args, name=name, path=path)
        self.extra = extra


class FeatureNotImplementedError(PyorpsError, NotImplementedError):
    """
    The requested combination of settings is not supported by the chosen backend
    or in this version. Also a ``NotImplementedError``. Use
    ``AlgorithmNotImplementedError`` when the missing piece is a named algorithm
    of a named graph library.
    """


class PyorpsRuntimeError(PyorpsError, RuntimeError):
    """
    An operation failed at run time although its inputs were valid (use after
    ``close()``, a solver that did not converge, a GPU failure). Also a
    ``RuntimeError``.
    """


class InsufficientMemoryError(PyorpsError, MemoryError):
    """
    An operation needs more host or GPU memory than is available or allowed.
    Also a ``MemoryError``.
    """


class InputFileNotFoundError(PyorpsError, FileNotFoundError):
    """
    A file the caller named (raster, profile, saved field) does not exist. Also a
    ``FileNotFoundError``.
    """

---
title: "Exceptions"
summary: "The PyorpsError hierarchy and when each exception is raised."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.core.exceptions"
api:
  - pyorps.PyorpsError
  - pyorps.NoPathFoundError
  - pyorps.RasterShapeError
  - pyorps.AlgorithmNotImplementedError
  - pyorps.PairwiseError
  - pyorps.CostAssumptionsError
  - pyorps.WFSError
  - pyorps.InvalidInputError
  - pyorps.InvalidTypeError
  - pyorps.MissingKeyError
  - pyorps.MissingDependencyError
  - pyorps.FeatureNotImplementedError
  - pyorps.PyorpsRuntimeError
  - pyorps.InsufficientMemoryError
  - pyorps.InputFileNotFoundError
  - pyorps.ConsistencyCheckError
  - pyorps.NumericalPrecisionError
  - pyorps.PyorpsWarning
---
# Exceptions

All PYORPS errors derive from `PyorpsError`, so `except PyorpsError` catches every library error.

Catch `PyorpsError` for anything the library reports. The classes that replace a builtin exception (`ValueError`, `TypeError`, `KeyError`, `ImportError`, `NotImplementedError`, `RuntimeError`, `MemoryError`, `FileNotFoundError`) also derive from that builtin, so code that catches `ValueError` keeps working. Every class pickles, so an error survives a trip between processes.

(exceptions-tree)=
## Hierarchy

| Exception | Raised when |
|---|---|
| `PyorpsError` | Base class of all errors below. |
| `NoPathFoundError` | No path can be found in the graph between source and target. |
| `RasterShapeError` | The raster shape is not supported. |
| `AlgorithmNotImplementedError` | The requested algorithm is not implemented in the chosen backend or graph library, or the backend is a stub. |
| `PairwiseError` | Pairwise computation fails; source and target lists must have the same length. |
| `CostAssumptionsError` | A cost assumptions file or object is invalid (subclasses `FileLoadError`, `InvalidSourceError`, `FormatError`). |
| `WFSError` | A WFS request fails (subclasses `WFSConnectionError`, `WFSResponseParsingError`, `WFSLayerNotFoundError`, `WFSServerError`, `WFSRequestRejectedError`). |
| `ObjectiveError`, `MetricStackError` | An `Objective` or `MetricStack` is inconsistent. |
| `FeatureColumnError` | Automatic feature-column detection fails (subclasses `NoSuitableColumnsError`, `ColumnAnalysisError`). |
| `EnsembleError` | A route-ensemble operation is invalid (`pyorps.core.ensemble`). |
| `InvalidInputError` | An argument has an unusable value: out of range, unknown name, wrong shape. Also a `ValueError`. Subclasses: `ThinForbiddenFeatureError`, `SealedOpeningError`, `ProvenanceMismatch`. |
| `InvalidTypeError` | An argument has the wrong type, or a keyword is misspelt. Also a `TypeError`. |
| `MissingKeyError` | A required key or label is missing (a `simplify` config without `method`, an unknown route key). Also a `KeyError`. |
| `MissingDependencyError` | An optional package is not installed; `.extra` names the pip extra. Also an `ImportError`. |
| `FeatureNotImplementedError` | The backend does not support this combination of settings. Also a `NotImplementedError`. |
| `PyorpsRuntimeError` | A valid call failed at run time: use after `close()`, a solver that did not converge, a GPU failure. Also a `RuntimeError`. |
| `InsufficientMemoryError` | The operation needs more host or GPU memory than is available or allowed. Also a `MemoryError`. |
| `InputFileNotFoundError` | A file the caller named (raster, profile, saved field) does not exist. Also a `FileNotFoundError`. |
| `ConsistencyCheckError` | A result failed a check that must hold if the inputs are what they claim to be, for example a lower bound above an upper bound. Also an `AssertionError`. |
| `NumericalPrecisionError` | A computation cannot be carried out exactly enough in the number format used, for example an FFT convolution too large for exact float64. Also an `ArithmeticError`. |
| `PyorpsWarning` | Root of the warning classes (`ThinForbiddenFeatureWarning`, `SealedOpeningWarning`, `ForbiddenBurnRoutingAssessment`, `CrsReprojectionWarning`). Derives from `UserWarning`. |

The research packages add their own classes: `DesignError` (`pyorps.collector`, an `InvalidInputError`), `LoadFlowError` (`pyorps.collector.voltage`, a `PyorpsRuntimeError`) and `ManifestError` (`pyorps.siting`, an `InvalidInputError`).

The thin-feature tools add their own warning and error classes, all rooted in `PyorpsWarning` and `InvalidInputError`; see {doc}`thin_forbidden_features`.

```python
from pyorps import PathFinder, NoPathFoundError

try:
    path = finder.find_route()
except NoPathFoundError:
    ...  # enlarge search_space_buffer_m or check for barriers
```

Every exception class and `PyorpsWarning` is exported at the top level (`from pyorps import PyorpsError, InvalidInputError, ...`) and from `pyorps.core`. The full tree is in the docstring of `pyorps.core.exceptions`.

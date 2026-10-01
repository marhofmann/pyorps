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
---
# Exceptions

All PYORPS errors derive from `PyorpsError`, so `except PyorpsError` catches every library error.

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
| `WFSError` | A WFS request fails (subclasses `WFSConnectionError`, `WFSResponseParsingError`, `WFSLayerNotFoundError`). |
| `ObjectiveError`, `MetricStackError` | An `Objective` or `MetricStack` is inconsistent. |
| `FeatureColumnError` | Automatic feature-column detection fails (subclasses `NoSuitableColumnsError`, `ColumnAnalysisError`). |

The thin-feature tools add their own warning and error classes; see {doc}`thin_forbidden_features`.

```python
from pyorps import PathFinder, NoPathFoundError

try:
    path = finder.find_route()
except NoPathFoundError:
    ...  # enlarge search_space_buffer_m or check for barriers
```

`PyorpsError`, `NoPathFoundError`, `RasterShapeError`, `AlgorithmNotImplementedError`, `PairwiseError`, `CostAssumptionsError` and `WFSError` are exported at the top level; the other classes live in `pyorps.core.exceptions`.

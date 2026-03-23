"""PYORPS - Python for Optimal Routes in Power Systems."""

__version__ = "0.3.1"

# Suppress third-party deprecation warnings triggered during import
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)
del _warnings

# Import key components for easy access
from .core.cost_assumptions import (
    CostAssumptions,
    detect_feature_columns,
    get_zero_cost_assumptions,
    save_empty_cost_assumptions,
)
from .core.exceptions import (
    AlgorithmNotImplementedError,
    CostAssumptionsError,
    NoPathFoundError,
    PairwiseError,
    PyorpsError,
    RasterShapeError,
    WFSError,
)
from .core.path import (  # Fixed: import from core.path instead of graph
    Path,
    PathCollection,
)
from .graph.path_finder import PathFinder
from .io.geo_dataset import (
    GeoDataset,
    InMemoryRasterDataset,
    InMemoryVectorDataset,
    LocalRasterDataset,
    LocalVectorDataset,
    RasterDataset,
    VectorDataset,
    WFSVectorDataset,
    initialize_geo_dataset,
)
from .raster.rasterizer import GeoRasterizer

__all__ = [
    # Core dataset classes
    "GeoDataset", "VectorDataset", "RasterDataset",
    "InMemoryVectorDataset", "LocalVectorDataset",
    "WFSVectorDataset", "LocalRasterDataset",
    "InMemoryRasterDataset", "initialize_geo_dataset",

    # Rasterization
    "GeoRasterizer",

    # Graph and routing
    "PathFinder", "Path", "PathCollection",

    # Cost assumptions
    "CostAssumptions", "get_zero_cost_assumptions", "detect_feature_columns",
    "save_empty_cost_assumptions",

    # Exceptions
    "PyorpsError", "NoPathFoundError", "RasterShapeError",
    "AlgorithmNotImplementedError", "PairwiseError",
    "CostAssumptionsError", "WFSError",
]

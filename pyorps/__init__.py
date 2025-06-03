"""PYORPS - Python for Optimal Routes in Power Systems."""

__version__ = "0.1.0"

# Import key components for easy access
from .io.geo_dataset import (
    GeoDataset, VectorDataset, RasterDataset,
    InMemoryVectorDataset, LocalVectorDataset,
    WFSVectorDataset, LocalRasterDataset,
    InMemoryRasterDataset, initialize_geo_dataset
)
from .raster.rasterizer import GeoRasterizer
from .graph.path_finder import PathFinder
from .core.path import Path, PathCollection  # Fixed: import from core.path instead of graph
from .core.cost_assumptions import (CostAssumptions, get_zero_cost_assumptions, detect_feature_columns,
                                    save_empty_cost_assumptions)

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
    "CostAssumptions", "get_zero_cost_assumptions", "detect_feature_columns", "save_empty_cost_assumptions",
]

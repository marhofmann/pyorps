"""Python Optimal Routing on Path Surfaces (PYORPS) package."""

__version__ = "0.1.0"

# Import key components for easy access
from .io.geo_dataset import (
    GeoDataset, VectorDataset, RasterDataset,
    InMemoryVectorDataset, LocalVectorDataset,
    WFSVectorDataset, LocalRasterDataset,
    InMemoryRasterDataset, get_geo_dataset
)
from .raster.rasterizer import GeoRasterizer, simply_rasterize
from .graph.raster_graph import RasterGraph
from .core.path import Path, PathCollection  # Fixed: import from core.path instead of graph
from .core.cost_assumptions import CostAssumptions

__all__ = [
    # Core dataset classes
    "GeoDataset", "VectorDataset", "RasterDataset",
    "InMemoryVectorDataset", "LocalVectorDataset",
    "WFSVectorDataset", "LocalRasterDataset",
    "InMemoryRasterDataset", "get_geo_dataset",

    # Rasterization
    "GeoRasterizer", "simply_rasterize",

    # Graph and routing
    "RasterGraph", "Path", "PathCollection",

    # Cost assumptions
    "CostAssumptions",
]

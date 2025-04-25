"""
Input/output operations for geospatial data.

This module provides:
1. Base classes for working with vector and raster geospatial data
2. Dataset implementations for local and in-memory data sources
3. Web Feature Service (WFS) data loading capabilities
4. Factory functions to create appropriate dataset instances
"""

# Core dataset classes
from .geo_dataset import (
    # Base classes
    GeoDataset, VectorDataset, RasterDataset,

    # Vector dataset implementations
    InMemoryVectorDataset, LocalVectorDataset, WFSVectorDataset,

    # Raster dataset implementations
    LocalRasterDataset, InMemoryRasterDataset,

    # Factory function
    initialize_geo_dataset
)

# Data loading functions
from .vector_loader import load_from_wfs

# Exception classes
from ..core.exceptions import (
    WFSError, WFSConnectionError,
    WFSResponseParsingError, WFSLayerNotFoundError
)

__all__ = [
    # Core dataset classes
    "GeoDataset", "VectorDataset", "RasterDataset",

    # Vector dataset implementations
    "InMemoryVectorDataset", "LocalVectorDataset", "WFSVectorDataset",

    # Raster dataset implementations
    "LocalRasterDataset", "InMemoryRasterDataset",

    # Factory function
    "initialize_geo_dataset",

    # Data loading functions
    "load_from_wfs",

    # Exception classes
    "WFSError", "WFSConnectionError",
    "WFSResponseParsingError", "WFSLayerNotFoundError"
]

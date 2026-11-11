"""
Input/output operations for geospatial data.

This module provides:
1. Base classes for working with vector and raster geospatial data
2. Dataset implementations for local and in-memory data sources
3. Web Feature Service (WFS) data loading capabilities
4. Factory functions to create appropriate dataset instances
"""

# Core dataset classes
# Exception classes
from ..core.exceptions import (
    WFSConnectionError,
    WFSError,
    WFSLayerNotFoundError,
    WFSRequestRejectedError,
    WFSResponseParsingError,
    WFSServerError,
)
from .geo_dataset import (
    # Base classes
    GeoDataset,
    InMemoryRasterDataset,
    # Vector dataset implementations
    InMemoryVectorDataset,
    # Raster dataset implementations
    LocalRasterDataset,
    LocalVectorDataset,
    RasterDataset,
    VectorDataset,
    WFSVectorDataset,
    # Factory function
    initialize_geo_dataset,
)
from .provenance import ProvenanceMismatch

# Data loading functions
from .raster_loader import (
    combine_rasters,
    create_combined_raster_dataset,
    find_raster_files,
    read_raster_files,
    validate_raster_compatibility,
)
from .vector_loader import load_from_wfs

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
    "read_raster_files", "combine_rasters", "create_combined_raster_dataset",
    "find_raster_files", "validate_raster_compatibility",

    # Exception classes
    "WFSError", "WFSConnectionError",
    "WFSResponseParsingError", "WFSLayerNotFoundError",
    "WFSServerError", "WFSRequestRejectedError",
    "ProvenanceMismatch",
]

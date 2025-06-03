"""
Raster data processing functionality for geospatial analysis.

This module provides:
1. Classes for handling and manipulating raster datasets
2. Rasterization tools for converting vector data to rasters
3. Cost surface generation capabilities
4. Utility functions for creating test data and processing rasters
"""

# Raster handling and processing
from .handler import RasterHandler

# Rasterization functionality
from .rasterizer import GeoRasterizer

__all__ = [
    # Raster handling
    "RasterHandler",

    # Rasterization
    "GeoRasterizer",
]

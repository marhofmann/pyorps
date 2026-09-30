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

# Sub-cell forbidden features: detection, widening, resolution guidance
from .thinness import (
    MIN_FORBIDDEN_WIDTH_CELLS,
    ForbiddenBurnAssessment,
    ForbiddenBurnReport,
    ForbiddenBurnRoutingAssessment,
    ForbiddenBurnSeverity,
    RepairSealReport,
    ResolutionAdvice,
    SealedOpeningError,
    SealedOpeningWarning,
    ThinForbiddenFeatureError,
    ThinForbiddenFeatureWarning,
    detect_forbidden_burn_defects,
    detect_repair_seals,
    is_thin,
    min_feature_width,
    recommended_geometry_buffer_m,
    safe_forbidden_width_m,
    suggest_resolution,
    thin_parts,
    widen_thin_features,
)

__all__ = [
    # Raster handling
    "RasterHandler",

    # Rasterization
    "GeoRasterizer",

    # Sub-cell forbidden features
    "MIN_FORBIDDEN_WIDTH_CELLS", "safe_forbidden_width_m",
    "recommended_geometry_buffer_m",
    "is_thin", "min_feature_width", "thin_parts", "widen_thin_features",
    "detect_forbidden_burn_defects", "ForbiddenBurnReport",
    "ForbiddenBurnAssessment", "ForbiddenBurnSeverity",
    "ForbiddenBurnRoutingAssessment",
    "suggest_resolution", "ResolutionAdvice",
    "ThinForbiddenFeatureWarning", "ThinForbiddenFeatureError",
    "detect_repair_seals", "RepairSealReport",
    "SealedOpeningWarning", "SealedOpeningError",
]

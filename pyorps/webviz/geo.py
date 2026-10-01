"""DEPRECATED shim: re-exports :mod:`pyorps.gui.services.geo` (Section 15)."""
from pyorps.gui.services.geo import *  # noqa: F401,F403
from pyorps.gui.services.geo import (  # noqa: F401  # pylint: disable=unused-import
    WGS84,
    _make_json_safe,
    as_feature_collection,
    gdf_to_wgs84_geojson,
    geojson_feature_geometry,
    geometry_to_crs,
    linestring_to_wgs84_feature,
    point_wgs84_to_crs,
    transformer_to_crs,
    wgs84_bounds,
    wgs84_linestring_to_crs,
)

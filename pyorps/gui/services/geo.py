"""
PYORPS GUI: coordinate/geometry helpers.

Leaflet works in WGS84 (EPSG:4326) lat/lon, while pyorps routing happens in the
raster's (usually projected) CRS. Everything that crosses the map boundary is
converted here so the rest of the app can stay CRS-agnostic.

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import LineString, mapping, shape
from shapely.ops import transform as shapely_transform

WGS84 = "EPSG:4326"


@lru_cache(maxsize=64)
def crs_transformer(source_crs: str, target_crs: str) -> Transformer:
    """Cached ``Transformer`` (always lon/lat). Building one costs ~0.8 ms, so
    render paths that reproject per-route/per-frame must reuse them."""
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _make_json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/shapely scalars to plain JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    # numpy scalars expose .item(); leave native types untouched.
    item = getattr(obj, "item", None)
    if callable(item) and obj.__class__.__module__ == "numpy":
        return obj.item()
    return obj


#: display-coordinate precision: 6 decimals ≈ 0.11 m at German latitudes —
#: far below routing precision needs (routing always uses the projected gdf;
#: this only trims the WGS84 *display* payload by about a third).
COORD_DECIMALS = 6


def _fast_wgs84_geojson(wgs: gpd.GeoDataFrame) -> dict:
    """Vectorized GeoJSON-dict builder (perf plan 1.1/1.2).

    ``gdf.to_json`` walks ``__geo_interface__`` per row (measured 721 ms for
    20k polygons); ``shapely.to_geojson`` writes the same geometries in C
    (121 ms) and orjson round-trips the attribute records to JSON-safe
    values (numpy scalars, Timestamps → ISO strings, NaN → null) at C speed.
    Feature ids are positional (str(i)) — matching the ``__row``/iloc logic
    used by the attribute table and feature highlighting.
    """
    import orjson
    import numpy as np
    import shapely

    geom_col = wgs.geometry.name
    geoms = shapely.transform(wgs.geometry.values,
                              lambda a: np.round(a, COORD_DECIMALS))
    geom_strs = shapely.to_geojson(geoms)
    # one C-speed parse of ALL geometries beats 20k tiny loads
    joined = "[" + ",".join((g if g is not None else "null")
                            for g in geom_strs) + "]"
    geom_dicts = orjson.loads(joined)  # pylint: disable=no-member  # orjson is a C extension
    records = wgs.drop(columns=[geom_col]).to_dict("records")
    option = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS  # pylint: disable=no-member
    safe = orjson.loads(orjson.dumps(records, default=str, option=option))  # pylint: disable=no-member
    features = [
        {"id": str(i), "type": "Feature", "properties": props,
         "geometry": geom}
        for i, (props, geom) in enumerate(zip(safe, geom_dicts))
    ]
    return {"type": "FeatureCollection", "features": features}


def gdf_to_wgs84_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Reproject a GeoDataFrame to WGS84 and return a plain GeoJSON dict.

    The input CRS must be set on the GeoDataFrame. All attribute columns
    survive as feature properties (dash-leaflet reads them back on click).
    Uses the vectorized shapely/orjson writer when orjson is available
    (~3x less CPU, 6-decimal display coordinates); falls back to the classic
    ``to_json`` path otherwise.
    """
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS. Set one (e.g. gdf.set_crs(...)) so it can "
            "be reprojected to WGS84 for the map."
        )
    wgs = gdf.to_crs(WGS84)
    try:
        return _fast_wgs84_geojson(wgs)
    except ImportError:
        # default=str coerces otherwise non-serializable property values
        # (pandas Timestamps, numpy types, etc. — common in WFS/ALKIS data).
        return json.loads(wgs.to_json(default=str))


def transformer_to_crs(target_crs: Any) -> Transformer:
    """Return a WGS84 -> target_crs transformer (always lon/lat order)."""
    return crs_transformer(WGS84, str(target_crs))


def wgs84_linestring_to_crs(coords_lonlat: list[list[float]],
                            target_crs: Any) -> LineString:
    """Convert a list of [lon, lat] pairs (from Leaflet) into a projected LineString.

    Leaflet/GeoJSON stores coordinates as [lon, lat]; the returned LineString is
    in ``target_crs`` (x, y) so it can be sampled against the cost raster.
    """
    tf = transformer_to_crs(target_crs)
    projected = [tf.transform(lon, lat) for lon, lat in coords_lonlat]
    return LineString(projected)


def geometry_to_crs(geom, source_crs: Any, target_crs: Any):
    """Reproject a single shapely geometry between two CRSs."""
    if str(source_crs) == str(target_crs):
        return geom
    tf = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_transform(lambda x, y, z=None: tf.transform(x, y), geom)


def geojson_feature_geometry(feature: dict):
    """Return the shapely geometry of a GeoJSON Feature or bare geometry dict."""
    if feature.get("type") == "Feature":
        return shape(feature["geometry"])
    return shape(feature)


def wgs84_bounds(geojson: dict) -> list[list[float]] | None:
    """Compute [[south, west], [north, east]] bounds of a WGS84 GeoJSON dict.

    Returns None for empty collections so callers can skip fit-bounds.
    """
    features = geojson.get("features", [])
    if not features:
        return None
    gdf = gpd.GeoDataFrame.from_features(features, crs=WGS84)
    if gdf.empty:
        return None
    minx, miny, maxx, maxy = gdf.total_bounds
    return [[float(miny), float(minx)], [float(maxy), float(maxx)]]


def point_wgs84_to_crs(lat: float, lon: float, target_crs: Any) -> tuple[float, float]:
    """Convert a single Leaflet (lat, lon) click into projected (x, y)."""
    tf = transformer_to_crs(target_crs)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


def as_feature_collection(geojson: dict) -> dict:
    """Normalize any GeoJSON dict into a FeatureCollection with safe properties."""
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
    elif geojson.get("type") == "Feature":
        features = [geojson]
    else:  # bare geometry
        features = [{"type": "Feature", "geometry": geojson, "properties": {}}]
    safe = []
    for feat in features:
        safe.append({
            "type": "Feature",
            "geometry": feat.get("geometry"),
            "properties": _make_json_safe(feat.get("properties") or {}),
        })
    return {"type": "FeatureCollection", "features": safe}


def linestring_to_wgs84_feature(line: LineString, source_crs: Any,
                                properties: dict | None = None) -> dict:
    """Build a WGS84 GeoJSON Feature from a projected LineString."""
    wgs_line = geometry_to_crs(line, source_crs, WGS84)
    return {
        "type": "Feature",
        "geometry": mapping(wgs_line),
        "properties": _make_json_safe(properties or {}),
    }

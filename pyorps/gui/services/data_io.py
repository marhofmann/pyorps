"""
PYORPS GUI: data loading — local files and WFS, clipped to the study area.

All loaders return a GeoDataFrame in the *project CRS*; WGS84 conversion
happens only at the map boundary (C10). WFS goes through pyorps'
``initialize_geo_dataset`` so its typed exceptions flow into the error
registry unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import shape

from .geo import geometry_to_crs

RASTER_EXTS = {".tif", ".tiff", ".jp2", ".img", ".bil", ".dem"}


def study_area_polygon(study_area_geojson: dict | None, target_crs: Any):
    """The drawn study area (WGS84 GeoJSON geometry/feature) in target_crs.

    Returns a shapely polygon or None when no area is drawn.
    """
    if not study_area_geojson:
        return None
    geometry = study_area_geojson.get("geometry", study_area_geojson)
    polygon = shape(geometry)
    return geometry_to_crs(polygon, "EPSG:4326", target_crs)


def study_area_bounds(study_area_geojson: dict | None,
                      target_crs: Any) -> tuple | None:
    """(minx, miny, maxx, maxy) of the study area in target_crs, or None."""
    polygon = study_area_polygon(study_area_geojson, target_crs)
    return tuple(polygon.bounds) if polygon is not None else None


def _use_arrow() -> bool:
    """pyogrio's Arrow fast path (documented 2-4x) needs pyarrow installed."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def load_local_vector(path: str | Path, layer: str | None = None,
                      mask=None, target_crs: Any = None) -> gpd.GeoDataFrame:
    """Read a local vector file, optionally masked, into the project CRS.

    Unmasked reads take pyogrio's ``use_arrow=True`` fast path (2-4x per the
    geopandas docs); masked/bbox reads stay on the classic path — pyogrio
    documents a GDAL filter bug for some drivers under Arrow.
    """
    kwargs: dict[str, Any] = {}
    if layer:
        kwargs["layer"] = layer
    if mask is not None:
        kwargs["mask"] = mask
    elif _use_arrow():
        kwargs["use_arrow"] = True
    gdf = gpd.read_file(str(path), **kwargs)
    if gdf.crs is None:
        if target_crs is None:
            raise ValueError(
                f"'{path}' carries no CRS and no project CRS is set.")
        gdf = gdf.set_crs(target_crs)
    elif target_crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf


def load_wfs_vector(url: str, layer: str, bbox=None, mask=None,
                    target_crs: Any = None) -> gpd.GeoDataFrame:
    """Load a WFS layer via pyorps (typed WFS errors surface unchanged)."""
    from pyorps import initialize_geo_dataset

    from pyorps.core.exceptions import WFSResponseParsingError

    if mask is not None and not hasattr(mask, "total_bounds"):
        # pyorps' WFSVectorDataset derives its request bbox from
        # mask.total_bounds and CRS-checks mask.crs — a bare shapely
        # geometry has neither, so wrap it into a GeoSeries.
        mask = gpd.GeoSeries([mask], crs=target_crs)
    dataset = initialize_geo_dataset({"url": url, "layer": layer},
                                     bbox=bbox, mask=mask, crs=None)
    requested = bbox
    if requested is None and mask is not None:
        requested = tuple(round(v, 1) for v in mask.total_bounds)
    where = (f"requested bbox {requested} in {target_crs or 'the data CRS'}"
             if requested is not None else "no bbox (full extent)")
    try:
        dataset.load_data()
    except AttributeError as exc:
        # pyorps' post_loading crashes on data=None ("'NoneType' object has
        # no attribute 'crs'") when the WFS answered with no features —
        # wrong layer name, or the layer doesn't cover the study area.
        if "NoneType" in str(exc):
            raise WFSResponseParsingError(
                f"The WFS returned no data for layer '{layer}' "
                f"({where}). The layer may not cover your study area "
                "(e.g. a Hessen layer with a study area outside Hessen), "
                "or the layer name is wrong.") from exc
        raise
    gdf = dataset.data
    if gdf is None or gdf.empty:
        raise WFSResponseParsingError(
            f"The WFS returned no features for layer '{layer}' "
            f"({where}) — check the layer name and that it covers the "
            "study area.")
    if target_crs is not None and gdf.crs is not None \
            and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf


def clip_to_area(gdf: gpd.GeoDataFrame, polygon) -> gpd.GeoDataFrame:
    """Clip to the study area polygon (already in the gdf's CRS)."""
    if polygon is None or gdf.empty:
        return gdf
    return gpd.clip(gdf, polygon)


def merge_gdfs(gdfs: list[gpd.GeoDataFrame],
               target_crs: Any = None) -> gpd.GeoDataFrame:
    """Merge several vector datasets into ONE GeoDataFrame (cross-state base).

    Built for merging the per-state ALKIS land-use layers of a cross-border
    project: every input is reprojected to ``target_crs`` (default: the first
    input's CRS), attribute columns are aligned by name (union — a column
    missing in one state simply stays empty there), and the rows are stacked.
    The merged frame then feeds ONE cost table and ONE rasterization step.
    """
    import pandas as pd

    gdfs = [g for g in (gdfs or []) if g is not None and not g.empty]
    if len(gdfs) < 2:
        raise ValueError("Merging needs at least two non-empty vector layers.")
    crs = target_crs or gdfs[0].crs
    if crs is None:
        raise ValueError("The first layer has no CRS and no target CRS is "
                         "set — cannot merge.")
    aligned = []
    for gdf in gdfs:
        if gdf.crs is None:
            raise ValueError("A layer without CRS cannot be merged — reload "
                             "it with a project CRS set.")
        if str(gdf.crs) != str(crs):
            gdf = gdf.to_crs(crs)
        aligned.append(gdf)
    geom_name = aligned[0].geometry.name
    normalized = []
    for gdf in aligned:
        if gdf.geometry.name != geom_name:
            gdf = gdf.rename_geometry(geom_name)
        normalized.append(gdf)
    merged = pd.concat(normalized, ignore_index=True, sort=False)
    return gpd.GeoDataFrame(merged, geometry=geom_name, crs=crs)


def dataset_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Small facts for the dataset list (feature count, columns, CRS)."""
    columns = [c for c in gdf.columns if c != gdf.geometry.name]
    return {"n_features": int(len(gdf)), "columns": columns,
            "crs": str(gdf.crs)}

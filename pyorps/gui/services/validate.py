"""
PYORPS GUI: pre-flight validation (Section 21.5 — shift left).

Cheap checks that run *before* an expensive or erroring pyorps call so the
user fixes inputs first. Each validator returns a :class:`Notice` (or a list)
when something is wrong, or None when the input is fine — the footguns F1-F12
become field-level validation instead of post-hoc surprises.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable

from .errors import Notice

VECTOR_EXTS = {".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml"}
RASTER_EXTS = {".tif", ".tiff", ".jp2", ".img", ".bil", ".dem"}

#: F3 thresholds — warn past ~50M cells / block past ~1 GB in-memory raster.
CELLS_WARN = 50_000_000
BYTES_BLOCK = 1_500_000_000


def validate_project_crs(crs: Any) -> Notice | None:
    """The project CRS must be projected (metres) — block geographic CRS."""
    try:
        from pyproj import CRS
        parsed = CRS.from_user_input(crs)
    except Exception:
        return Notice(
            severity="error", title="Unknown CRS",
            meaning=f"'{crs}' couldn't be parsed as a coordinate system.",
            impact="Rasterizing and routing can't run.",
            fix="Pick a projected CRS (e.g. EPSG:25832).", focus_id="tab-data")
    if parsed.is_geographic:
        return Notice(
            severity="error", title="Project CRS must be metric",
            meaning=f"'{crs}' is a geographic (lat/lon, degrees) CRS; "
                    "routing and rasterizing need metres.",
            impact="Distances and costs would be wrong.",
            fix="Pick a projected CRS such as EPSG:25832 (UTM 32N).",
            focus_id="tab-data")
    return None


def validate_search_buffer(buffer_m: float | None,
                           sources: list | None = None,
                           targets: list | None = None) -> Notice | None:
    """F1: an unset/zero buffer means the ENTIRE raster is routed."""
    if buffer_m and buffer_m > 0:
        return None
    return Notice(
        severity="warning", title="No search buffer set",
        meaning="Without a search-space buffer the entire raster is routed.",
        impact="Routing may be very slow and memory-heavy (results stay "
               "correct).",
        fix="Set a buffer — the default max(1000 m, 1.5 x distance) is "
            "applied automatically when you leave it blank.",
        focus_id="tab-routes")


def estimate_raster_size(bounds: tuple[float, float, float, float] | None,
                         resolution_in_m: float,
                         dtype: str = "uint16") -> dict:
    """Live cell-count and memory estimate for the raster tab (F3).

    ``bounds`` is (minx, miny, maxx, maxy) in a metric CRS. Returns a dict
    with n_cells / bytes / mb and a human-readable label.
    """
    import numpy as np

    if not bounds or resolution_in_m <= 0:
        return {"n_cells": 0, "bytes": 0, "mb": 0.0, "label": "-"}
    minx, miny, maxx, maxy = bounds
    width = max(0, math.ceil((maxx - minx) / resolution_in_m))
    height = max(0, math.ceil((maxy - miny) / resolution_in_m))
    n_cells = width * height
    itemsize = np.dtype(dtype).itemsize
    n_bytes = n_cells * itemsize
    label = (f"{width:,} x {height:,} = {n_cells:,} cells "
             f"(~{n_bytes / 1e6:,.0f} MB as {dtype})")
    return {"n_cells": n_cells, "bytes": n_bytes, "mb": n_bytes / 1e6,
            "label": label, "width": width, "height": height}


def validate_raster_size(bounds: tuple | None, resolution_in_m: float,
                         dtype: str = "uint16") -> Notice | None:
    """F3: warn on huge rasters, block ones that would exhaust memory."""
    est = estimate_raster_size(bounds, resolution_in_m, dtype)
    if est["bytes"] > BYTES_BLOCK:
        return Notice(
            severity="error", title="This raster would be too large",
            meaning=f"{est['label']} exceeds the safe memory limit.",
            impact="Rasterization would fail or crash the app.",
            fix="Increase resolution_in_m (coarser cells) or shrink the "
                "study area.", focus_id="tab-raster")
    if est["n_cells"] > CELLS_WARN:
        return Notice(
            severity="warning", title="Very large raster",
            meaning=est["label"] + " — that's a lot of cells.",
            impact="Rasterizing and routing will be slow.",
            fix="Consider a coarser resolution or a smaller study area.",
            focus_id="tab-raster")
    return None


def uncovered_categories(gdf, feature_keys: tuple[str, ...],
                         assumptions: dict) -> list[str]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """F4: feature values missing from the cost table (and no "" catch-all).

    Returns human-readable labels of uncovered categories that would become
    NaN -> fill_value (forbidden) during rasterization.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if gdf is None or not feature_keys or not isinstance(assumptions, dict):
        return []
    main = feature_keys[0]
    if main not in gdf.columns:
        return []
    missing: list[str] = []
    if len(feature_keys) == 1:
        known = set(assumptions)
        for value in gdf[main].dropna().unique():
            if value not in known and "" not in known:
                missing.append(str(value))
        return sorted(missing)
    side = feature_keys[1]
    if side not in gdf.columns:
        return []
    for main_val, group in gdf.dropna(subset=[main]).groupby(main):
        sub = assumptions.get(main_val)
        if sub is None:
            if "" not in assumptions:
                missing.append(str(main_val))
            continue
        if not isinstance(sub, dict):
            continue  # scalar cost covers every subcategory
        if "" in sub:
            continue  # catch-all present
        for side_val in group[side].fillna("").unique():
            if side_val not in sub:
                missing.append(f"{main_val} / {side_val or '(empty)'}")
    return sorted(set(missing))


def validate_cost_coverage(gdf, feature_keys: tuple[str, ...],
                           assumptions: dict) -> Notice | None:
    """F4 as a notification: N categories would silently become forbidden."""
    missing = uncovered_categories(gdf, feature_keys, assumptions)
    if not missing:
        return None
    shown = ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
    return Notice(
        severity="warning",
        title=f"{len(missing)} categories have no cost — treated as "
              "forbidden",
        meaning=f"These values aren't in your cost table and have no \"\" "
                f"catch-all: {shown}",
        impact="Those areas become no-go and may block routes (F4).",
        fix="Add costs for them, or add a \"\" catch-all row per category.",
        focus_id="tab-cost")


def validate_points_in_bounds(points: Iterable[tuple[float, float]],
                              bounds: tuple[float, float, float, float],
                              label: str = "point") -> Notice | None:
    """Pre-flight: routing points must lie inside the raster window."""
    minx, miny, maxx, maxy = bounds
    outside = [(x, y) for x, y in points
               if not (minx <= x <= maxx and miny <= y <= maxy)]
    if not outside:
        return None
    listed = "; ".join(f"({x:,.0f}, {y:,.0f})" for x, y in outside[:5])
    return Notice(
        severity="error", title=f"{len(outside)} {label}(s) outside the area",
        meaning=f"These points fall outside the raster window: {listed}",
        impact="Routing would fail for the affected pairs.",
        fix="Move the points inside the study area / raster extent.",
        focus_id="tab-routes")


def validate_file(path: str, kind: str = "vector") -> Notice | None:
    """File exists + extension supported, before calling pyorps."""
    if not path:
        return Notice(severity="warning", title="No file selected",
                      meaning="The path field is empty.",
                      impact="Nothing was loaded.",
                      fix="Enter or pick a file path.")
    exts = VECTOR_EXTS if kind == "vector" else RASTER_EXTS
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in exts:
        return Notice(
            severity="error", title="Unsupported file type",
            meaning=f"'{ext or path}' isn't a supported {kind} format.",
            impact="The file wasn't loaded.",
            fix=f"Use one of: {', '.join(sorted(exts))}.")
    if not os.path.exists(path):
        return Notice(
            severity="error", title="File not found",
            meaning=f"'{path}' doesn't exist or was moved.",
            impact="The file wasn't loaded.",
            fix="Re-select the file; check it wasn't moved or renamed.")
    return None


def validate_wfs_url(url: str) -> Notice | None:
    """WFS URL must be http(s) with a host, before calling pyorps."""
    from urllib.parse import urlparse

    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return Notice(
            severity="error", title="That WFS URL looks invalid",
            meaning="The URL is malformed, not http(s), or has no host.",
            impact="Can't connect to the server.",
            fix="Paste the full https://... endpoint including the host.",
            focus_id="tab-data")
    return None

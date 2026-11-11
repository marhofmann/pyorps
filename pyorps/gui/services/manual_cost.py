"""
PYORPS GUI: custom drawn cost layer (Feature 5).

A user draws polygons on the map (the study-area draw tool) and snapshots them
into a *vector layer with a per-polygon ``cost`` column*. That layer is a
first-class citizen: it appears in the Layers panel, its costs are editable in
the 📋 table (see callbacks/layers.py), and it can drive the cost raster either
as a base dataset or as an override modifier — per polygon, using each polygon's
own ``cost`` value.

The drawn shapes arrive as WGS84 GeoJSON (Leaflet's CRS); we reproject them into
the project CRS so they line up with the routing raster.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd

from .geo import WGS84, geojson_feature_geometry

#: meta flag marking a vector layer as a manual/drawn cost layer.
MANUAL_META_KEY = "manual_cost"
COST_COLUMN = "cost"
#: stable id of THE editable custom-cost-polygon layer (task 33) — draw/edit/
#: delete on the map all update this one layer.
MANUAL_LAYER_ID = "manual-cost-layer"


def layer_from_drawn(features: list[dict], project_crs: Any, *,
                     default_cost: float, mode: str = "override",
                     name: str = "Manual costs") -> gpd.GeoDataFrame:
    """Build a project-CRS GeoDataFrame from drawn WGS84 polygon features.

    Each drawn polygon becomes one row with a ``name`` and a numeric ``cost``
    (seeded to ``default_cost``, editable later). ``mode`` ("override" | "base")
    is remembered on every row so the rasterizer knows how to apply it.
    """
    polys = []
    for feat in features or []:
        geom = geojson_feature_geometry(feat)
        if geom is not None and not geom.is_empty and geom.geom_type in (
                "Polygon", "MultiPolygon"):
            polys.append(geom)
    if not polys:
        raise ValueError(
            "No polygons drawn. Use the map's polygon/rectangle draw tool to "
            "outline the area(s), then click 'Create from drawn shapes'.")
    cost = float(default_cost)
    gdf = gpd.GeoDataFrame(
        {"name": [f"{name} {i + 1}" for i in range(len(polys))],
         COST_COLUMN: [cost] * len(polys),
         "mode": [mode] * len(polys)},
        geometry=polys, crs=WGS84)
    return gdf.to_crs(project_crs)


def _match_previous_rows(polys, prev_rows, prev_wkts) -> list[dict]:
    """One previous grid row (or ``{}``) per polygon in ``polys``.

    Rows are matched to polygons by the polygon's WGS84 WKT first:
    ``prev_wkts[j]`` is the shape that grid row ``j`` described (a row's own
    ``__row`` field says which, else its position). Polygons left over fall
    back to the row at their index, unless a geometric match already used it.
    """
    by_wkt: dict[str, list[int]] = {}
    for pos, row in enumerate(prev_rows):
        index = row.get("__row")
        index = index if isinstance(index, int) and not isinstance(index, bool)             else pos
        if 0 <= index < len(prev_wkts):
            by_wkt.setdefault(prev_wkts[index], []).append(pos)
    taken: set[int] = set()
    matched: list[int | None] = []
    for poly in polys:
        candidates = by_wkt.get(poly.wkt)
        if candidates:
            pos = candidates.pop(0)
            taken.add(pos)
            matched.append(pos)
        else:
            matched.append(None)
    result: list[dict] = []
    for i, pos in enumerate(matched):
        if pos is None and i < len(prev_rows) and i not in taken:
            pos = i                            # edited / new shape: by index
        result.append(prev_rows[pos] if pos is not None else {})
    return result


def sync_manual_layer(state, features, *, project_crs, default_cost, mode,
                      name, prev_rows=None):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Build/update THE editable custom-cost-polygon layer from drawn shapes.

    Called live as the user draws / edits vertices / deletes polygons (task 33).
    Per-polygon ``name``/``cost`` are carried over from ``prev_rows`` by
    GEOMETRY: the layer remembers the WGS84 WKT of every polygon in
    ``meta["wgs84_wkts"]`` (aligned with the grid rows), so a polygon deleted in
    the middle of the list no longer shifts the others' names and costs onto
    the wrong shapes. A shape without a geometric match (new, or a vertex was
    just moved) falls back to the previous row at its own index, if no other
    shape already took that row; anything else gets ``default_cost``. Empties
    remove the layer. Returns ``(layer_or_None, grid_rows)`` where each grid
    row is ``{"__row", "name", "cost"}``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from .geo import gdf_to_wgs84_geojson

    prev = prev_rows or []
    # the study area shares the draw tool; and polygons removed from the list
    # stay removed even though their drawn shape lingers — never treat either
    # as a cost polygon (task 62)
    exclude = (getattr(state, "study_area_geoms", None) or set()) | (
        getattr(state, "manual_excluded_geoms", None) or set())
    polys = []
    for feat in features or []:
        geom = geojson_feature_geometry(feat)
        if geom is None or geom.is_empty or geom.geom_type not in (
                "Polygon", "MultiPolygon"):
            continue
        if geom.wkt in exclude:            # this is a study-area shape (task 62)
            continue
        polys.append(geom)
    if not polys:
        state.remove_layer(MANUAL_LAYER_ID)
        return None, []

    existing = state.get(MANUAL_LAYER_ID)
    previous = _match_previous_rows(
        polys, prev, ((existing.meta or {}).get("wgs84_wkts")
                      if existing is not None else None) or [])
    names, costs = [], []
    for i in range(len(polys)):
        row = previous[i]
        names.append(str(row.get("name") or f"{name} {i + 1}"))
        raw = row.get("cost")
        costs.append(float(raw) if raw not in (None, "") else float(
            default_cost))
    gdf = gpd.GeoDataFrame(
        {"name": names, COST_COLUMN: costs, "mode": [mode] * len(polys)},
        geometry=polys, crs=WGS84).to_crs(project_crs)
    geojson = gdf_to_wgs84_geojson(gdf)
    meta = {MANUAL_META_KEY: True, "mode": mode, "cost_column": COST_COLUMN,
            "summary": {"n_features": len(gdf)},
            # WGS84 WKT per kept polygon, aligned with the grid rows — lets the
            # "remove selected" action exclude the exact drawn shape (task 62)
            "wgs84_wkts": [p.wkt for p in polys]}
    layer = state.get(MANUAL_LAYER_ID)
    if layer is None:
        layer = state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs,
                                geojson=geojson, layer_id=MANUAL_LAYER_ID,
                                meta=meta, style={"color": "#ff00ff"})
    else:
        layer.gdf, layer.crs, layer.geojson, layer.name = (gdf, gdf.crs,
                                                           geojson, name)
        layer.meta.update(meta)
    rows = [{"__row": i, "name": names[i], "cost": costs[i]}
            for i in range(len(polys))]
    return layer, rows


def override_specs_from_layer(gdf: gpd.GeoDataFrame, *, buffer_m: float = 0.0,
                              name: str = "Manual costs", ignore_value=None):
    """One override :class:`ModifierSpec` per distinct ``cost`` in the layer.

    Each polygon carries its own cost, so we group by the ``cost`` column and
    emit an override modifier per value (mirrors the legacy per-value expansion
    in callbacks/raster._build_modifiers). Rows with a forbidden/zero cost are
    kept — a user may legitimately want to stamp 0 or 65535. ``ignore_value=None``
    means manual polygons always win (even over forbidden base cells).
    """
    from .rasterize import ModifierSpec

    if COST_COLUMN not in gdf.columns:
        raise ValueError(
            f"Manual cost layer '{name}' has no '{COST_COLUMN}' column.")
    specs = []
    for cost_value, group in gdf.groupby(COST_COLUMN):
        specs.append(ModifierSpec(
            gdf=group, mode="override", factor=float(cost_value),
            buffer_m=buffer_m, name=name, ignore_value=ignore_value,
            condition=f"{COST_COLUMN} = {cost_value:g}"))
    return specs

"""
PYORPS GUI: cost-raster combination table (Feature 2).

Every distinct cost value in a built cost raster is produced by a *combination*
of a base land-use class and the modifier zones that overlap it — e.g.

    ave_nutzung: Wald / Nadelholz                         -> 405
    ave_nutzung: Wald / Nadelholz + TWS_HQS_TK25: ZONE==I  -> 40500  (x100)

Rather than reverse-engineering the raster pixels (ambiguous — two combinations
can share a cost), we reconstruct the mapping **geometrically**, exactly the way
the rasterizer builds the raster: dissolve the base dataset by its feature
columns (one region per class), then split those regions against each modifier
zone (intersection = the modified region, difference = the untouched region),
carrying the full provenance and applying the modifier's cost arithmetic. The
result is a set of disjoint regions, each tagged with its complete
layer->feature->value combination and final cost — which is also what lets a
click on a raster cell select the exact combination (point-in-polygon, see
:func:`locate`).

The cost arithmetic mirrors ``GeoRasterizer.modify_raster_from_dataset``:
multiply clips to the forbidden sentinel, override replaces, and a modifier
whose ``ignore_value`` equals the sentinel never touches an already-forbidden
region (so a forbidden base class stays forbidden).
"""
from __future__ import annotations

import geopandas as gpd
from shapely.ops import unary_union

from .. import presets

FORBIDDEN = presets.FORBIDDEN


# ------------------------------------------------------------ base cost lookup
def _combo_cost(assumptions: dict, feature_keys, values) -> int:
    """Base cost for one (main[, side]) combination, honouring "" catch-alls."""
    main_val = "" if values[0] is None else str(values[0])
    if len(feature_keys) == 1:
        cost = assumptions.get(main_val, assumptions.get("", FORBIDDEN))
    else:
        sub = assumptions.get(main_val)
        if isinstance(sub, dict):
            side_val = "" if values[1] is None else str(values[1])
            cost = sub.get(side_val, sub.get("", FORBIDDEN))
        else:
            cost = sub if sub is not None else FORBIDDEN
    try:
        return int(round(float(cost)))
    except (TypeError, ValueError):
        return FORBIDDEN


def _apply_modifier(cost: int, spec) -> tuple[int, bool]:
    """(new_cost, changed?) for a modifier acting on a region of ``cost``."""
    if spec.ignore_value is not None and cost == int(spec.ignore_value):
        return cost, False                       # e.g. forbidden stays forbidden
    if spec.mode == "multiply":
        return min(int(round(cost * spec.factor)), FORBIDDEN), True
    return min(int(round(spec.factor)), FORBIDDEN), True     # override


# --------------------------------------------------------------- table builder
def ensure_combination_gdf(layer):
    """Materialize a raster layer's combination table on FIRST use (lazy).

    ``run_rasterize`` stores the build inputs in ``meta["combination_inputs"]``
    instead of building inline (the geometric overlay can take seconds on a
    big cost model — perf plan phase 4), so the rasterize callback returns
    sooner; the table is built the first time the 📋 table, the legend or a
    map-click needs it and cached in ``meta["combination_gdf"]`` as before.
    Returns the table or None (never raises — a broken build degrades to
    "no combination table", same as pre-lazy behaviour).
    """
    meta = layer.meta or {}
    combo = meta.get("combination_gdf")
    if combo is not None:
        return combo
    inputs = meta.get("combination_inputs")
    if not inputs:
        return None
    try:
        combo = build_combination_table(
            inputs["base_gdf"], inputs["feature_keys"],
            inputs["assumptions"],
            base_layer_name=inputs.get("base_layer_name", "base"),
            modifiers=inputs.get("modifiers"),
            base_crs=inputs.get("base_crs"),
            bounding_polygon=inputs.get("bounding_polygon"))
    except Exception:      # degrade to "no table" — never crash a viewer
        meta.pop("combination_inputs", None)
        return None
    meta.pop("combination_inputs", None)
    if combo is not None and not combo.empty:
        meta["combination_gdf"] = combo
        return combo
    return None


def build_combination_table(base_gdf, feature_keys, assumptions, *,
                            base_layer_name: str = "base",
                            modifiers=None, base_crs=None,
                            bounding_polygon=None) -> gpd.GeoDataFrame:
    """Disjoint regions with full provenance + final cost (see module docstring).

    Returns a GeoDataFrame in ``base_crs`` with columns ``base``, one per
    feature key, ``modifiers`` (human text), ``cost`` and ``__row``.
    """
    feature_keys = tuple(feature_keys)
    base_crs = base_crs or base_gdf.crs
    key_cols = list(feature_keys)

    diss = base_gdf[key_cols + [base_gdf.geometry.name]].copy()
    for col in key_cols:
        diss[col] = diss[col].fillna("").astype(str)
    diss = diss.dissolve(by=key_cols).reset_index()

    pieces: list[dict] = []
    for _, row in diss.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        values = tuple(row[c] for c in key_cols)
        pieces.append({
            "base": base_layer_name,
            **{c: values[i] for i, c in enumerate(key_cols)},
            "cost": _combo_cost(assumptions, feature_keys, values),
            "modifiers": "", "geometry": geom})

    for spec in modifiers or []:
        if getattr(spec, "is_empty", False) or spec.gdf is None:
            continue
        mod = spec.gdf
        if mod.crs is not None and str(mod.crs) != str(base_crs):
            mod = mod.to_crs(base_crs)
        buffered = (mod.geometry.buffer(spec.buffer_m)
                    if spec.buffer_m else mod.geometry)
        mod_geom = unary_union(list(buffered))
        if mod_geom.is_empty:
            continue
        if not mod_geom.is_valid:
            mod_geom = mod_geom.buffer(0)
        label = f"{spec.name}: {spec.condition}"
        label += (f" x{spec.factor:g}" if spec.mode == "multiply"
                  else f" ={spec.factor:g}")

        split: list[dict] = []
        for piece in pieces:
            new_cost, changed = _apply_modifier(piece["cost"], spec)
            if not changed:
                split.append(piece)
                continue
            geom = piece["geometry"]
            inter = geom.intersection(mod_geom)
            diff = geom.difference(mod_geom)
            if not inter.is_empty:
                modified = dict(piece)
                modified["cost"] = new_cost
                modified["modifiers"] = (f"{piece['modifiers']} + {label}").strip(
                    " +") if piece["modifiers"] else label
                modified["geometry"] = inter
                split.append(modified)
            if not diff.is_empty:
                keep = dict(piece)
                keep["geometry"] = diff
                split.append(keep)
            if inter.is_empty and diff.is_empty:
                split.append(piece)
        pieces = split

    # optional clip to the raster's extent
    if bounding_polygon is not None:
        clipped = []
        for piece in pieces:
            geom = piece["geometry"].intersection(bounding_polygon)
            if not geom.is_empty:
                piece = dict(piece, geometry=geom)
                clipped.append(piece)
        pieces = clipped

    # merge geometry of identical combinations, then number the rows
    merged: dict[tuple, dict] = {}
    for piece in pieces:
        if piece["geometry"].is_empty:
            continue
        sig = (piece["base"], *(piece[c] for c in key_cols),
               piece["modifiers"], piece["cost"])
        if sig in merged:
            merged[sig]["geometry"] = unary_union(
                [merged[sig]["geometry"], piece["geometry"]])
        else:
            merged[sig] = dict(piece)

    records = list(merged.values())
    if not records:
        return gpd.GeoDataFrame(
            columns=["base", *key_cols, "modifiers", "cost", "__row",
                     "geometry"], geometry="geometry", crs=base_crs)
    # sort by cost FIRST, then number rows — so ``__row`` equals the displayed
    # position (which drives the grid's getRowId + scrollTo/page-jump). If it
    # were assigned pre-sort, clicking a cell would jump to the wrong page.
    gdf = gpd.GeoDataFrame(records, geometry="geometry",
                           crs=base_crs).sort_values("cost").reset_index(
        drop=True)
    gdf["__row"] = range(len(gdf))
    return gdf


# ------------------------------------------------------------ grid + selection
def cost_color_map(combo_gdf, colormap="viridis", vmin=None, vmax=None) -> dict:
    """{cost -> hex} matching the tile colours (forbidden -> None)."""
    from .tiles import cost_colors

    if combo_gdf is None or combo_gdf.empty:
        return {}
    costs = [int(c) for c in combo_gdf["cost"].tolist()]
    passable = [c for c in costs if c < FORBIDDEN]
    lo = vmin if vmin is not None else (min(passable) if passable else 0)
    hi = vmax if vmax is not None else (max(passable) if passable else 1)
    return cost_colors(colormap, set(costs), lo, hi)


def legend_items(combo_gdf, colormap="viridis", vmin=None, vmax=None):
    """``[(cost, hex_or_None, label)]`` for the colour legend, cost order.

    ``label`` is the layer→feature[→modifier] combination, e.g.
    ``ave_nutzung: Wald / Nadelholz  + TWS_HQS_TK25: ZONE == Schutzzone I``.
    """
    if combo_gdf is None or combo_gdf.empty:
        return []
    colors = cost_color_map(combo_gdf, colormap, vmin, vmax)
    key_cols = [c for c in combo_gdf.columns
                if c not in ("base", "modifiers", "cost", "__row",
                             combo_gdf.geometry.name)]
    items = []
    for _, row in combo_gdf.iterrows():
        cost = int(row["cost"])
        combo_txt = " / ".join(str(row[k]) for k in key_cols if row[k])
        label = f"{row['base']}: {combo_txt}" if combo_txt else str(row["base"])
        if row["modifiers"]:
            label += f"  + {row['modifiers']}"
        items.append((cost, colors.get(cost), label))
    return items


def grid_payload(combo_gdf: gpd.GeoDataFrame, layer_name: str, *,
                 colormap: str = "viridis", vmin=None, vmax=None):
    """(columnDefs, rowData, title) for the combination table offcanvas.

    Each row is colour-coded (a swatch column) with the exact colour the tile
    renderer paints that cost, so the table reads like a legend for the map.
    Uses dash-ag-grid ``styleConditions`` (data-driven, safe) rather than a raw
    function cellStyle (which can freeze rendering — see project memory).
    """
    key_cols = [c for c in combo_gdf.columns
                if c not in ("base", "modifiers", "cost", "__row",
                             combo_gdf.geometry.name)]
    colors = cost_color_map(combo_gdf, colormap, vmin, vmax)
    distinct = sorted({c for c in colors.values() if c})
    swatch_style = {"styleConditions": [
        {"condition": f"params.data._color == '{c}'",
         "style": {"backgroundColor": c}} for c in distinct]}
    column_defs = [{"field": "_swatch", "headerName": "", "maxWidth": 34,
                    "minWidth": 34, "pinned": "left", "sortable": False,
                    "filter": False, "cellStyle": swatch_style},
                   {"field": "__row", "headerName": "#", "maxWidth": 60,
                    "pinned": "left", "filter": False, "sortable": False},
                   {"field": "cost", "headerName": "cost", "maxWidth": 100,
                    "type": "numericColumn", "pinned": "left",
                    "cellClassRules": {"text-danger fw-bold":
                                       f"params.value >= {FORBIDDEN}"}},
                   {"field": "base", "headerName": "layer"}]
    column_defs += [{"field": str(c), "headerName": str(c)} for c in key_cols]
    column_defs.append({"field": "modifiers", "headerName": "modifiers",
                        "flex": 2})
    row_data = []
    for i in range(len(combo_gdf)):
        row = combo_gdf.iloc[i]
        cost = int(row["cost"])
        record = {"__row": int(row["__row"]), "cost": cost, "_swatch": "",
                  "_color": colors.get(cost), "base": row["base"],
                  "modifiers": row["modifiers"]}
        for c in key_cols:
            record[str(c)] = None if row[c] is None else str(row[c])
        row_data.append(record)
    n_costs = combo_gdf["cost"].nunique() if len(combo_gdf) else 0
    title = (f"{layer_name} — {len(combo_gdf)} combination(s), {n_costs} "
             "distinct cost value(s), colour-coded to the map. Click a raster "
             "cell to select its combination.")
    return column_defs, row_data, title


def locate(combo_gdf: gpd.GeoDataFrame, x: float, y: float):
    """``__row`` of the combination region containing point (x, y), or None."""
    from shapely.geometry import Point

    if combo_gdf is None or combo_gdf.empty:
        return None
    point = Point(x, y)
    hits = combo_gdf[combo_gdf.geometry.contains(point)]
    if hits.empty:
        hits = combo_gdf[combo_gdf.geometry.intersects(point.buffer(1e-6))]
    if hits.empty:
        return None
    return int(hits.iloc[0]["__row"])

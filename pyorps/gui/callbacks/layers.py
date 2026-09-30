"""
PYORPS GUI callbacks: the layer system (Section 7).

The ``layer-host`` LayerGroup — a DIRECT child of the Map (hard rule C1) — is
re-rendered from ``ProjectState`` whenever the small ``layers-view`` store
changes. The Layers tab edits visibility/name in an AG Grid, reorders with
explicit ▲/▼ buttons (guaranteed to round-trip, unlike browser drag), and
removes layers. LayersControl is never involved for dynamic layers.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..layout import (RASTER_TILE_ZINDEX, ROUTE_STYLE, STUDY_AREA_STYLE,
                      TABLE_PAGE_SIZE, VECTOR_STYLE)

_EMPTY_FC = {"type": "FeatureCollection", "features": []}

#: synthetic Layers-grid row id for a whole route group ("group::<label>")
GROUP_ROW_PREFIX = "group::"
UNGROUPED_LABEL = "(ungrouped)"


def _group_members(state, label: str) -> list:
    """The route layers of a grid group row ('(ungrouped)' = no group)."""
    return state.routes_in_group("" if label == UNGROUPED_LABEL else label)


def _expand_selection(state, selected_rows) -> list[str]:
    """Grid selection -> concrete layer ids (group rows -> their members)."""
    ids_out: list[str] = []
    for row in selected_rows or []:
        rid = row.get("id") or ""
        if rid.startswith(GROUP_ROW_PREFIX):
            label = rid[len(GROUP_ROW_PREFIX):]
            ids_out.extend(m.id for m in _group_members(state, label))
        elif rid:
            ids_out.append(rid)
    return ids_out


def layers_grid_rows(state) -> list[dict]:
    """Layers-grid rows, top = foreground. Route layers are COLLAPSED into
    one row per group (the individual routes live in the Routes tree below);
    the group row sits where its topmost member paints."""
    rows: list[dict] = []
    seen_groups: set[str] = set()
    for view_row in reversed(state.layers_view()):
        layer = state.get(view_row.get("id"))
        if layer is None or layer.kind != "route":
            rows.append(view_row)
            continue
        label = (layer.meta or {}).get("group") or UNGROUPED_LABEL
        if label in seen_groups:
            continue
        seen_groups.add(label)
        members = _group_members(state, label)
        rows.append({
            "id": f"{GROUP_ROW_PREFIX}{label}", "name": label,
            "kind": f"routes ({len(members)})",
            "visible": any(m.visible for m in members),
            "z": view_row.get("z"),
        })
    return rows


def _table_page(row_index: int) -> int:
    """0-based page holding ``row_index`` in the fixed-page-size table (F2)."""
    return int(row_index) // TABLE_PAGE_SIZE

#: vector layers with more features than this are served as ``dl.GeoJSON(url=…)``
#: (fetched once, never re-serialized into render_host responses); smaller ones
#: stay inline where an extra HTTP round-trip would cost more than it saves.
URL_GEOJSON_THRESHOLD = 400

_KIND_STYLE = {
    "study_area": STUDY_AREA_STYLE,
    "vector": VECTOR_STYLE,
    "route": ROUTE_STYLE,
}


def _json_safe(value):
    """Coerce a GeoDataFrame cell to something AG-Grid can render (Feature 4)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


#: cap on rows shipped to the attribute grid — the browser paginates anyway and
#: multi-thousand-row payloads make the offcanvas sluggish to open. The first
#: MAX_TABLE_ROWS are shown (with a note); highlighting still uses gdf indices.
MAX_TABLE_ROWS = 5000


def _geobuf_available() -> bool:
    """True when the geobuf encoder is importable (perf plan 1.3)."""
    try:
        import geobuf  # noqa: F401
        return True
    except ImportError:
        return False


def _rows_from_gdf(gdf, cols, limit=None):
    """Fast AG-Grid rows from a GeoDataFrame (vectorized ``to_dict``, ~12x
    faster than a per-row ``iloc`` loop) with a stable ``__row`` index."""
    frame = gdf.iloc[:limit] if limit is not None else gdf
    records = frame[cols].to_dict("records") if cols else [{}] * len(frame)
    return [{"__row": i, **{str(k): _json_safe(v) for k, v in rec.items()}}
            for i, rec in enumerate(records)]


def _attr_table_payload(layer):
    """(columnDefs, rowData, title) for a vector layer's full table (F4).

    A drawn/manual cost layer (Feature 5) gets an editable ``cost`` column so
    the user can set per-polygon costs right in the table; every other layer is
    read-only. ``__row`` is the positional index, matching ``_feature_highlight``
    and the map-click → row-select path (Features 1/3).
    """
    from ..services.manual_cost import COST_COLUMN, MANUAL_META_KEY

    gdf = layer.gdf
    geom_col = gdf.geometry.name
    cols = [c for c in gdf.columns if c != geom_col]
    manual = bool((layer.meta or {}).get(MANUAL_META_KEY))
    column_defs = [{"field": "__row", "headerName": "#", "maxWidth": 70,
                    "pinned": "left", "filter": False, "sortable": False}]
    for c in cols:
        col_def = {"field": str(c), "headerName": str(c)}
        if manual and c in (COST_COLUMN, "name"):
            col_def["editable"] = True
            if c == COST_COLUMN:
                col_def["type"] = "numericColumn"
        column_defs.append(col_def)
    capped = len(gdf) > MAX_TABLE_ROWS
    row_data = _rows_from_gdf(gdf, cols, MAX_TABLE_ROWS if capped else None)
    hint = ("Edit the cost / name cells to set per-polygon values. "
            if manual else "")
    shown = (f"first {MAX_TABLE_ROWS:,} of {len(gdf):,}" if capped
             else f"{len(gdf):,}")
    title = (f"{layer.name} — {shown} feature(s), {len(cols)} column(s). "
             f"{hint}Click a row to highlight it on the map.")
    return column_defs, row_data, title


def _raster_style(layer):
    """(colormap, vmin, vmax) for a raster layer — drives the table colouring."""
    meta = layer.meta or {}
    colormap = meta.get("colormap") or (
        layer.tile.colormap if layer.tile is not None else "viridis")
    return colormap, meta.get("vmin"), meta.get("vmax")


def _combination_table_payload(layer):
    """(columnDefs, rowData, title) for a raster's cost-combination table (F2).

    Returns None when the raster layer has no stored combination table (loaded
    or combined rasters have none — only rasters built from the Cost tab do).
    """
    from ..services import combinations

    combo = combinations.ensure_combination_gdf(layer)   # lazy build (perf)
    if combo is None or getattr(combo, "empty", True):
        return None
    colormap, vmin, vmax = _raster_style(layer)
    return combinations.grid_payload(combo, layer.name, colormap=colormap,
                                     vmin=vmin, vmax=vmax)


def _feature_highlight(layer, row_index: int):
    """WGS84 FeatureCollection with just ``row_index``'s feature (highlight).

    ``layer.geojson`` preserves the GeoDataFrame row order (gdf.to_json), so a
    table/gdf row index maps directly to the cached display feature.
    """
    if layer is None or not layer.geojson:
        return None
    features = layer.geojson.get("features") or []
    if 0 <= row_index < len(features):
        return {"type": "FeatureCollection", "features": [features[row_index]]}
    return None


def _feature_row_index(layer, feature):
    """Positional index of a clicked GeoJSON feature within ``layer`` (or None).

    ``gdf.to_json`` tags every feature with an ``id`` (the row index) and keeps
    row order, so we match the clicked feature by ``id`` first and fall back to
    geometry equality. The returned position lines up with ``_attr_table_payload``
    ``__row`` and ``_feature_highlight`` (Features 1/3).
    """
    if layer is None or not layer.geojson or not isinstance(feature, dict):
        return None
    features = layer.geojson.get("features") or []
    fid = feature.get("id")
    if fid is not None:
        for i, feat in enumerate(features):
            if feat.get("id") == fid:
                return i
    geom = feature.get("geometry")
    if geom is not None:
        for i, feat in enumerate(features):
            if feat.get("geometry") == geom:
                return i
    return None


def render_layer(layer):
    """One Layer -> its dash-leaflet component (None when hidden)."""
    if not layer.visible:
        return None
    if layer.kind == "raster":
        if layer.tile is None:
            return None
        return dl.TileLayer(
            id={"type": ids.TYPE_RASTER_TILE, "id": layer.id},
            url=layer.tile.tile_url, opacity=layer.opacity, maxZoom=22,
            zIndex=RASTER_TILE_ZINDEX)
    if layer.kind == "wms":
        wms = (layer.meta or {}).get("wms") or {}
        kwargs = {}
        if wms.get("bounds"):                     # clip to the study area
            kwargs["bounds"] = wms["bounds"]
        return dl.WMSTileLayer(
            id={"type": ids.TYPE_RASTER_TILE, "id": layer.id},
            url=wms.get("base_url", ""), layers=wms.get("layers", ""),
            format=wms.get("format", "image/png"),
            transparent=wms.get("transparent", True),
            version=wms.get("version", "1.3.0"), opacity=layer.opacity,
            maxZoom=22, **kwargs)
    style = dict(_KIND_STYLE.get(layer.kind, VECTOR_STYLE))
    style.update(layer.style or {})
    geojson = layer.geojson or _EMPTY_FC
    common = {"id": {"type": ids.TYPE_LAYER_GEOJSON, "id": layer.id},
              "options": {"style": style},
              "hoverStyle": {"weight": 4, "color": "#ffff00"}}
    # large vector layers: fetch ONCE via URL instead of shipping the data in
    # every render_host response (perf). Preferred wire format is geobuf —
    # binary protobuf, a fraction of the JSON size and far cheaper for the
    # browser to parse (dash-leaflet decodes it natively); GeoJSON URL is the
    # fallback when the geobuf encoder is missing. Routes/study-area/small
    # layers stay inline — an HTTP round-trip would cost more than it saves.
    if (layer.kind == "vector"
            and len(geojson.get("features") or []) > URL_GEOJSON_THRESHOLD):
        if _geobuf_available():
            return dl.GeoJSON(url=f"/_gb/{layer.id}?v={layer.geojson_rev}",
                              format="geobuf", **common)
        return dl.GeoJSON(url=f"/_gj/{layer.id}?v={layer.geojson_rev}", **common)
    return dl.GeoJSON(data=geojson, **common)


def route_control_markers(layer) -> list:
    """Source/target/waypoint markers for a visible route layer.

    Colours: source green, target red, waypoints yellow — so existing routes
    show their control points on the map, not only the active one.
    """
    from ..services.geo import crs_transformer
    from .interaction import point_marker

    points = (layer.meta or {}).get("control_points") or []
    if len(points) < 2 or layer.crs is None:
        return []
    names = (layer.meta or {}).get("waypoint_names") or []
    tf = crs_transformer(str(layer.crs), "EPSG:4326")
    latlngs = []
    for x, y in points:
        lon, lat = tf.transform(x, y)
        latlngs.append((lat, lon))
    markers = [point_marker("source", *latlngs[0], 0)]
    for i, ll in enumerate(latlngs[1:-1]):
        markers.append(point_marker("waypoint", *ll, i,
                                    names[i] if i < len(names) else ""))
    markers.append(point_marker("target", *latlngs[-1], 0))
    return markers


def render_layer_host(state) -> list:
    """All visible layers in paint order (lowest z first)."""
    children = []
    for layer in state.ordered_layers():
        component = render_layer(layer)
        if component is not None:
            children.append(component)
            if layer.kind == "route":
                children.extend(route_control_markers(layer))
    return children


def host_signature(state) -> tuple:
    """Fingerprint of everything the layer host paints (perf audit #1).

    Cheap (ids + object identities, no serialization). ``layers-view`` is
    written by ~20 callbacks — many of which don't change what's on the map
    (route selection, opacity/colormap handled elsewhere, …); comparing this
    signature lets ``render_host`` skip re-shipping every layer's GeoJSON when
    nothing it renders actually changed. Object *identities* (``id(...)``) work
    as change detectors because geojson / control-points are reassigned (never
    mutated in place) when they change, and routes are immutable (edits spawn
    new layers)."""
    signature = []
    for layer in state.ordered_layers():
        if not layer.visible:
            signature.append((layer.id, False))
            continue
        head = (layer.id, True, layer.z, layer.kind)
        if layer.kind in ("raster", "wms"):
            signature.append(head + (
                round(float(layer.opacity), 4),
                layer.tile.tile_url if layer.tile is not None else None,
                id(layer.meta.get("wms")) if layer.kind == "wms" else None))
        else:
            meta = layer.meta or {}
            signature.append(head + (
                id(layer.geojson), repr(layer.style),
                id(meta.get("control_points")), id(meta.get("waypoint_names")),
                str(layer.crs) if layer.kind == "route" else None))
    return tuple(signature)


def route_tree_component(state):
    """Hierarchical view of routes: all routes → groups → individual routes.

    Rendered as nested <details> (safe, no fragile AG-Grid tree/grouping — see
    project memory on rowGroup freezes). Each route is a click-to-select button.
    """
    routes = state.layers_of_kind("route")
    if not routes:
        return html.Small("No routes yet — plan one in the Routes tab.",
                          className="text-muted")
    groups: dict[str, list] = {}
    order: list[str] = []
    for layer in routes:
        label = (layer.meta or {}).get("group") or "(ungrouped)"
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(layer)

    group_nodes = []
    for label in order:
        members = groups[label]
        items = []
        for layer in members:
            active = layer.id == state.active_route_id
            items.append(html.Div(
                dbc.Button(
                    ("● " if active else "○ ") + layer.name,
                    id={"type": ids.TYPE_ROUTE_TREE_ITEM, "id": layer.id},
                    size="sm", color="link",
                    className="p-0 text-decoration-none"
                    + (" fw-bold" if active else "")),
                className="ms-4"))
        group_nodes.append(html.Details([
            html.Summary(f"{label} ({len(members)})",
                         className="fw-semibold"),
            html.Div(items),
        ], open=True, className="ms-3"))
    return html.Details([
        html.Summary(f"Routes ({len(routes)})", className="fw-bold"),
        html.Div(group_nodes),
    ], open=True)


def register(app, state) -> None:
    # ------------------------------------------------ map render (C1 obeyed)
    @app.callback(Output(ids.LAYER_HOST, "children"),
                  Input(ids.LAYERS_VIEW, "data"))
    def render_host(_layers_view):
        # skip the (potentially multi-MB) rebuild when nothing painted changed
        signature = host_signature(state)
        if signature == getattr(state, "_host_sig", None):
            raise PreventUpdate
        state._host_sig = signature
        return render_layer_host(state)

    # ------------------------------------------------ layers tab grid mirror
    @app.callback(Output(ids.LAYERS_GRID, "rowData"),
                  Input(ids.LAYERS_VIEW, "data"))
    def sync_grid(_layers_view):
        # top row = foreground; route groups collapse into single rows
        return layers_grid_rows(state)

    # ------------------------------------------------ edits: name/visibility
    @app.callback(Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
                  Input(ids.LAYERS_GRID, "cellValueChanged"),
                  prevent_initial_call=True)
    def apply_cell_edit(events):
        if not events:
            raise PreventUpdate
        for event in events if isinstance(events, list) else [events]:
            data = event.get("data") or {}
            layer_id = data.get("id") or ""
            column = (event.get("colId")
                      or event.get("column", {}).get("colId"))
            if layer_id.startswith(GROUP_ROW_PREFIX):
                # a route-group row: rename the group / toggle all members
                label = layer_id[len(GROUP_ROW_PREFIX):]
                members = _group_members(state, label)
                if column == "name":
                    new_label = str(data.get("name") or "").strip()
                    for member in members:
                        member.meta["group"] = new_label or None
                elif column == "visible":
                    for member in members:
                        state.set_visible(member.id,
                                          bool(data.get("visible")))
                continue
            if not layer_id or state.get(layer_id) is None:
                continue
            if column == "name":
                state.rename(layer_id, data.get("name"))
            elif column == "visible":
                state.set_visible(layer_id, bool(data.get("visible")))
        return state.layers_view()

    # -------------------------- reorder: ▲ / ▼ buttons (multi-select aware)
    @app.callback(Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
                  Input("layer-up-btn", "n_clicks"),
                  Input("layer-down-btn", "n_clicks"),
                  State(ids.LAYERS_GRID, "selectedRows"),
                  prevent_initial_call=True)
    def move_layer(_up, _down, selected):
        if not selected:
            raise PreventUpdate
        order = [ly.id for ly in state.ordered_layers()]
        move_set = {i for i in _expand_selection(state, selected)
                    if i in order}
        if not move_set:
            raise PreventUpdate
        # every selected layer moves ONE step, keeping the selection's
        # relative order; blocks stop at the edges ("Up" = higher z)
        changed = False
        if ctx.triggered_id == "layer-up-btn":
            for idx in range(len(order) - 2, -1, -1):
                if order[idx] in move_set and order[idx + 1] not in move_set:
                    order[idx], order[idx + 1] = order[idx + 1], order[idx]
                    changed = True
        else:
            for idx in range(1, len(order)):
                if order[idx] in move_set and order[idx - 1] not in move_set:
                    order[idx - 1], order[idx] = order[idx], order[idx - 1]
                    changed = True
        if not changed:
            raise PreventUpdate
        state.reorder(order)
        return state.layers_view()

    # ------------------------------------------------------------ remove
    @app.callback(Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
                  Output(ids.SELECTED_FEATURE_LAYER, "data",
                         allow_duplicate=True),
                  Input(ids.LAYER_REMOVE_BTN, "n_clicks"),
                  State(ids.LAYERS_GRID, "selectedRows"),
                  prevent_initial_call=True)
    def remove_layer(n_clicks, selected):
        if not n_clicks or not selected:
            raise PreventUpdate
        for layer_id in _expand_selection(state, selected):
            state.remove_layer(layer_id)
        # drop any feature highlight (its layer may be gone)
        return state.layers_view(), _EMPTY_FC

    # -------------------------- Feature 4: open a layer's full attribute table
    @app.callback(
        Output(ids.LAYER_TABLE_OFFCANVAS, "is_open"),
        Output(ids.LAYER_TABLE_GRID, "columnDefs"),
        Output(ids.LAYER_TABLE_GRID, "rowData"),
        Output(ids.LAYER_TABLE_TITLE, "children"),
        Input(ids.LAYER_VIEW_BTN, "n_clicks"),
        State(ids.LAYERS_GRID, "selectedRows"),
        prevent_initial_call=True)
    def open_layer_table(n_clicks, selected):
        if not n_clicks or not selected:
            raise PreventUpdate
        layer = state.get(selected[0].get("id"))
        if layer is not None and layer.kind == "raster":
            payload = _combination_table_payload(layer)
            if payload is not None:
                return (True, *payload)
            return (True, [], [], f"{layer.name}: no combination table "
                    "(build the raster from the Cost tab to get one).")
        if layer is None or layer.gdf is None or layer.gdf.empty:
            return (True, [], [],
                    "This layer has no attribute table (rasters have none).")
        return (True, *_attr_table_payload(layer))

    # ------------------ Feature 5: edit per-polygon cost in the layer table
    @app.callback(
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.LAYER_TABLE_GRID, "cellValueChanged"),
        State(ids.LAYERS_GRID, "selectedRows"),
        prevent_initial_call=True)
    def edit_layer_table_cell(events, layer_selection):
        from ..services.manual_cost import COST_COLUMN, MANUAL_META_KEY

        if not events or not layer_selection:
            raise PreventUpdate
        layer = state.get(layer_selection[0].get("id"))
        if layer is None or not (layer.meta or {}).get(MANUAL_META_KEY):
            raise PreventUpdate
        changed = False
        for event in events if isinstance(events, list) else [events]:
            data = event.get("data") or {}
            row = data.get("__row")
            column = (event.get("colId")
                      or event.get("column", {}).get("colId"))
            if row is None or column not in (COST_COLUMN, "name"):
                continue
            if not (0 <= int(row) < len(layer.gdf)):
                continue
            value = data.get(column)
            if column == COST_COLUMN:
                from ..services import cost_model
                value = cost_model.coerce_factor(value)
            layer.gdf.iat[int(row), layer.gdf.columns.get_loc(column)] = value
            changed = True
        if not changed:
            raise PreventUpdate
        return no_update

    # ------------------ Feature 4/3: table row -> highlight the feature
    @app.callback(
        Output(ids.SELECTED_FEATURE_LAYER, "data", allow_duplicate=True),
        Output(ids.SELECTED_FEATURE, "data", allow_duplicate=True),
        Input(ids.LAYER_TABLE_GRID, "selectedRows"),
        State(ids.LAYERS_GRID, "selectedRows"),
        prevent_initial_call=True)
    def highlight_table_row(table_selection, layer_selection):
        if not table_selection or not layer_selection:
            raise PreventUpdate
        layer = state.get(layer_selection[0].get("id"))
        row_index = table_selection[0].get("__row")
        highlight = (_feature_highlight(layer, int(row_index))
                     if layer is not None and row_index is not None else None)
        if highlight is None:
            raise PreventUpdate
        return highlight, {"layer_id": layer.id, "row": int(row_index),
                           "props": table_selection[0]}

    # -------- Feature 1: click a feature of the selected layer -> its table row
    @app.callback(
        Output(ids.LAYER_TABLE_OFFCANVAS, "is_open", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "columnDefs", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "rowData", allow_duplicate=True),
        Output(ids.LAYER_TABLE_TITLE, "children", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "selectedRows"),
        Output(ids.LAYER_TABLE_GRID, "paginationGoTo"),
        Output(ids.LAYER_TABLE_GRID, "scrollTo"),
        Input({"type": ids.TYPE_LAYER_GEOJSON, "id": ALL}, "clickData"),
        State(ids.LAYERS_GRID, "selectedRows"),
        prevent_initial_call=True)
    def select_clicked_feature_row(_click_datas, layer_selection):
        trigger = ctx.triggered_id
        if not trigger or not layer_selection:
            raise PreventUpdate
        clicked_id = trigger.get("id")
        if clicked_id != layer_selection[0].get("id"):
            raise PreventUpdate           # only the *selected* layer (F1)
        feature = next((item["value"] for item in ctx.triggered
                        if item.get("value")), None)
        layer = state.get(clicked_id)
        if layer is None or layer.gdf is None or not feature:
            raise PreventUpdate
        row_index = _feature_row_index(layer, feature)
        if row_index is None:
            raise PreventUpdate
        column_defs, row_data, title = _attr_table_payload(layer)
        selected = [r for r in row_data if r["__row"] == row_index]
        return (True, column_defs, row_data, title, selected,
                _table_page(row_index), {"rowIndex": row_index})

    # ---- Feature 2: click a raster cell -> select its cost-combination row
    @app.callback(
        Output(ids.LAYER_TABLE_OFFCANVAS, "is_open", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "columnDefs", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "rowData", allow_duplicate=True),
        Output(ids.LAYER_TABLE_TITLE, "children", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "selectedRows", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "paginationGoTo", allow_duplicate=True),
        Output(ids.LAYER_TABLE_GRID, "scrollTo", allow_duplicate=True),
        Input(ids.MAP, "clickData"),
        State(ids.LAYERS_GRID, "selectedRows"),
        State(ids.UI_STATE, "data"),
        prevent_initial_call=True)
    def select_raster_combination(click_data, layer_selection, ui_state):
        from ..services import combinations, geo

        # only when a raster with a combination table is the selected layer and
        # no build/edit click-mode is armed (dispatch_click owns those)
        if (ui_state or {}).get("click_mode", "off") != "off":
            raise PreventUpdate
        if not click_data or not click_data.get("latlng") or not layer_selection:
            raise PreventUpdate
        layer = state.get(layer_selection[0].get("id"))
        combo = (combinations.ensure_combination_gdf(layer)   # lazy build
                 if layer is not None and layer.kind == "raster" else None)
        if layer is None or layer.kind != "raster" or combo is None \
                or layer.crs is None:
            raise PreventUpdate
        lat = float(click_data["latlng"]["lat"])
        lng = float(click_data["latlng"]["lng"])
        x, y = geo.point_wgs84_to_crs(lat, lng, layer.crs)
        row_index = combinations.locate(combo, x, y)
        if row_index is None:
            raise PreventUpdate
        colormap, vmin, vmax = _raster_style(layer)
        column_defs, row_data, title = combinations.grid_payload(
            combo, layer.name, colormap=colormap, vmin=vmin, vmax=vmax)
        selected = [r for r in row_data if r["__row"] == row_index]
        return (True, column_defs, row_data, title, selected,
                _table_page(row_index), {"rowIndex": row_index})

    # -------------------------------- background map: source / opacity / z-order
    @app.callback(
        Output(ids.BASEMAP_TILE, "url"),
        Output(ids.BASEMAP_TILE, "attribution"),
        Output(ids.BASEMAP_TILE, "opacity"),
        Output(ids.BASEMAP_TILE, "zIndex"),
        Input(ids.BASEMAP_SELECT, "value"),
        Input(ids.BASEMAP_OPACITY, "value"),
        Input(ids.BASEMAP_ZORDER, "value"))
    def update_basemap(name, opacity, zorder):
        from ..layout import (BASEMAP_BY_NAME, BASEMAP_Z_ABOVE,
                              BASEMAP_Z_BELOW, BLANK_TILE)

        basemap = BASEMAP_BY_NAME.get(name or "")
        url = basemap["url"] if basemap else BLANK_TILE
        attribution = basemap["attribution"] if basemap else ""
        z = BASEMAP_Z_ABOVE if zorder == "above" else BASEMAP_Z_BELOW
        return url, attribution, float(opacity if opacity is not None else 1.0), z

    # (the attribute-table panel is resized by dragging its top edge —
    # assets/table-resize.js; the old percent slider is gone)

    # ---------------------------------- hierarchical route tree (route → groups)
    @app.callback(Output(ids.ROUTE_TREE, "children"),
                  Input(ids.LAYERS_VIEW, "data"))
    def render_route_tree(_view):
        return route_tree_component(state)

    @app.callback(
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Input({"type": ids.TYPE_ROUTE_TREE_ITEM, "id": ALL}, "n_clicks"),
        prevent_initial_call=True)
    def select_route_from_tree(clicks):
        trigger = ctx.triggered_id
        if not trigger or not any(clicks or []):
            raise PreventUpdate
        route_id = trigger.get("id")
        if state.get(route_id) is None:
            raise PreventUpdate
        return route_id

    # ------------------- clicking a route ON THE MAP selects it (round 11)
    @app.callback(
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Input({"type": ids.TYPE_LAYER_GEOJSON, "id": ALL}, "clickData"),
        prevent_initial_call=True)
    def select_route_on_map(_click_datas):
        trigger = ctx.triggered_id
        if not trigger:
            raise PreventUpdate
        if not any(item.get("value") for item in ctx.triggered):
            raise PreventUpdate            # mount-fire, not a real click
        layer = state.get(trigger.get("id"))
        if layer is None or layer.kind != "route":
            raise PreventUpdate
        if layer.id == state.active_route_id:
            raise PreventUpdate
        return layer.id

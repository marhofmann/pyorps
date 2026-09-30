"""
PYORPS GUI callbacks: study area (C4) + data loading (R2, R4).

The EditControl's ``geojson`` is the single source of drawn shapes; the last
drawn polygon becomes the study area (WGS84 in state, projected for
clipping). Local/WFS loads run through ``errors.guard`` so every failure and
warning lands in the notification stack (C15), and loaded data is clipped to
the study area when requested.
"""
from __future__ import annotations

import os

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import data_io, geo, validate
from ..services.errors import Notice, guard, success

_STUDY_AREA_LAYER_ID = "study-area"


def _expand_bounds(bounds, *, frac: float = 0.02, floor: float = 300.0):
    """Grow a (minx, miny, maxx, maxy) bbox outward so EVERY cell/tile that
    intersects the study area is fetched (WCS SUBSET drops centre-outside edge
    cells; a clipped WMS bounds hides edge tiles). Margin = max(frac*span, floor)
    per side, in the bbox's own units."""
    minx, miny, maxx, maxy = bounds
    mx = max(frac * abs(maxx - minx), floor)
    my = max(frac * abs(maxy - miny), floor)
    return (minx - mx, miny - my, maxx + mx, maxy + my)


def _study_area_info(state) -> str:
    if not state.study_area:
        return "No study area drawn."
    bounds = data_io.study_area_bounds(state.study_area, state.project_crs)
    if bounds is None:
        return "No study area drawn."
    minx, miny, maxx, maxy = bounds
    return (f"Area {abs(maxx - minx) / 1000:.1f} x "
            f"{abs(maxy - miny) / 1000:.1f} km "
            f"({state.project_crs}) — bbox ({minx:,.0f}, {miny:,.0f}, "
            f"{maxx:,.0f}, {maxy:,.0f})")


def _dataset_list(state) -> list:

    from .. import ids as _ids

    items = []
    for layer in state.layers_of_kind("vector"):
        summary = layer.meta.get("summary", {})
        label = (f"{layer.name} — {summary.get('n_features', '?')} "
                 f"features, cols: "
                 f"{', '.join(summary.get('columns', [])[:6])}")
        refresh = ""
        if layer.meta.get("source"):
            refresh = dbc.Button(
                "↻", size="sm", color="link", className="p-0 ms-1",
                title="Reload this dataset from its source",
                id={"type": _ids.TYPE_DATASET_REFRESH, "index": layer.id})
        items.append(html.Li([label, refresh]))
    if not items:
        return [html.Small("No vector datasets loaded.",
                           className="text-muted")]
    return [html.Ul(items, className="small mb-0")]


def _add_vector_layer(state, gdf, name: str, source: dict | None = None):
    """Register a loaded GeoDataFrame as a vector layer (deduplicated).

    A dataset from the same source (same WFS url+layer / same file+layer)
    is loaded only once: loading it again UPDATES the existing layer in
    place (this is also what the ↻ refresh button does). Returns
    ``(layer, refreshed)``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    geojson = geo.gdf_to_wgs84_geojson(gdf)
    summary = data_io.dataset_summary(gdf)
    if source:
        key = {k: v for k, v in source.items() if k != "clip"}
        for layer in state.layers_of_kind("vector"):
            existing = {k: v for k, v in
                        (layer.meta.get("source") or {}).items()
                        if k != "clip"}
            if existing == key:
                layer.gdf = gdf
                layer.crs = gdf.crs
                layer.geojson = geojson
                layer.meta["summary"] = summary
                layer.meta["source"] = source
                return layer, True
    layer = state.add_layer(
        name, "vector", gdf=gdf, crs=gdf.crs, geojson=geojson,
        meta={"summary": summary, "source": source or {}})
    return layer, False


def _load_from_source(state, source: dict, notices: list):
    """(Re)load a dataset from its recorded source dict."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    clip = source.get("clip", True)
    mask = (data_io.study_area_polygon(state.study_area, state.project_crs)
            if clip else None)
    if source.get("kind") == "osm":
        from ..services import osm as osm_svc

        west, south, east, north = source["bbox"]
        gdf, notices = guard(
            osm_svc.load_osm_features, (south, west, north, east),
            source["filters"], notices=notices)
        if gdf is not None:
            gdf = gdf.to_crs(state.project_crs)
        return gdf, notices
    if source.get("kind") == "merge":
        members = [state.get(lid) for lid in source.get("ids", [])]
        gdfs = [ly.gdf for ly in members
                if ly is not None and ly.gdf is not None]
        gdf, notices = guard(data_io.merge_gdfs, gdfs, state.project_crs,
                             notices=notices)
        return gdf, notices
    if source.get("kind") == "wfs":
        gdf, notices = guard(
            data_io.load_wfs_vector, source["url"], source["layer"],
            mask=mask, target_crs=state.project_crs, notices=notices)
    else:
        gdf, notices = guard(
            data_io.load_local_vector, source["path"],
            layer=source.get("layer") or None, mask=mask,
            target_crs=state.project_crs, notices=notices)
    if gdf is not None and clip and mask is not None:
        gdf, notices = guard(data_io.clip_to_area, gdf, mask,
                             notices=notices)
    return gdf, notices


def register(app, state) -> None:
    # --------------------- draw dispatch: study area OR custom cost polygons
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.STUDY_AREA_INFO, "children"),
        Output(ids.MANUAL_GRID, "rowData", allow_duplicate=True),
        Output(ids.MANUAL_STATUS, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.DRAW_CONTROL, "geojson"),
        State(ids.DRAW_TARGET, "data"),
        State(ids.MANUAL_NAME, "value"),
        State(ids.MANUAL_COST, "value"),
        State(ids.MANUAL_MODE, "value"),
        State(ids.MANUAL_GRID, "rowData"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    def on_draw(drawn, target, m_name, m_cost, m_mode, m_rows, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        features = (drawn or {}).get("features") or []
        if target == "cost":
            # live-sync THE editable custom-cost-polygon layer (task 33)
            from ..services import manual_cost

            _layer, rows = manual_cost.sync_manual_layer(
                state, features, project_crs=state.project_crs,
                default_cost=float(m_cost if m_cost is not None else 65535),
                mode=m_mode or "override",
                name=(m_name or "Manual costs").strip() or "Manual costs",
                prev_rows=m_rows)
            status = (f"{len(rows)} cost polygon(s) — edit name/cost below, "
                      "move vertices / delete with the map tools."
                      if rows else "Draw cost polygons on the map.")
            return state.layers_view(), no_update, rows, status, no_update
        # study area (default target)
        if not features:
            if state.study_area is not None:          # deleted -> clear area
                state.study_area = None
                state.remove_layer(_STUDY_AREA_LAYER_ID)
                return (state.layers_view(), _study_area_info(state),
                        no_update, no_update, notices)
            raise PreventUpdate
        feature = features[-1]
        state.study_area = feature
        # remember this shape so the shared draw tool never turns it into a
        # cost polygon (task 62)
        try:
            from shapely.geometry import shape as _shape
            state.study_area_geoms.add(
                _shape(feature.get("geometry", feature)).wkt)
        except Exception:  # pragma: no cover - defensive  # nosec B110
            pass
        state.remove_layer(_STUDY_AREA_LAYER_ID)
        state.add_layer(
            "Study area", "study_area", layer_id=_STUDY_AREA_LAYER_ID,
            geojson={"type": "FeatureCollection", "features": [feature]})
        notices.append(success(
            "Study area set",
            meaning="New data loads and rasterization are clipped to it."))
        return (state.layers_view(), _study_area_info(state), no_update,
                no_update, notices)

    # draw target follows the active tab: shapes drawn while the Cost tab is
    # open become cost polygons, anywhere else they set the study area (the
    # explicit radio was removed — task: no manual toggle to forget)
    @app.callback(Output(ids.DRAW_TARGET, "data"),
                  Input(ids.TABS, "active_tab"),
                  prevent_initial_call=True)
    def set_draw_target(active_tab):
        return "cost" if active_tab == ids.TAB_COST else "area"

    # show only the parameters of the selected vector source (local | wfs)
    @app.callback(Output(ids.DATA_LOCAL_PANEL, "style"),
                  Output(ids.DATA_WFS_PANEL, "style"),
                  Input(ids.DATA_SOURCE_TYPE, "value"))
    def toggle_source_panels(source):
        hidden = {"display": "none"}
        if source == "wfs":
            return hidden, {}
        return {}, hidden

    # ---------------- merge several vector layers into one (cross-state base)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.DATASET_LIST, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.MERGE_BTN, "n_clicks"),
        State(ids.MERGE_SELECT, "value"),
        State(ids.MERGE_NAME, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def merge_layers(n_clicks, layer_ids, name, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        members = [state.get(lid) for lid in (layer_ids or [])]
        members = [ly for ly in members
                   if ly is not None and ly.gdf is not None]
        if len(members) < 2:
            notices.append(Notice(
                severity="warning", title="Pick at least two vector layers",
                meaning="Merging stacks several loaded vector layers (e.g. "
                        "per-state ALKIS land use) into one.",
                impact="Nothing was merged.",
                fix="Select two or more layers in the merge list.",
                focus_id=ids.TAB_DATA,
                focus_control=ids.MERGE_SELECT).to_dict())
            return no_update, no_update, notices
        gdf, notices = guard(data_io.merge_gdfs, [ly.gdf for ly in members],
                             state.project_crs, notices=notices)
        if gdf is None:
            return no_update, no_update, notices
        display = (name or "").strip() or \
            f"Merged: {' + '.join(ly.name for ly in members)}"[:80]
        source = {"kind": "merge",
                  "ids": [ly.id for ly in members], "clip": False}
        layer, refreshed = _add_vector_layer(state, gdf, display,
                                             source=source)
        notices.append(success(
            f"{'Re-merged' if refreshed else 'Merged'} {len(members)} "
            f"layer(s) → '{layer.name}'",
            meaning=f"{len(gdf)} features in {state.project_crs}. Use it as "
                    "the Cost-tab base dataset — one cost table, one "
                    "rasterization for the whole cross-state project."))
        return state.layers_view(), _dataset_list(state), notices

    # -------------------- categorized OSM menu (column -> values, additive)
    @app.callback(Output(ids.OSM_VALUES, "options"),
                  Output(ids.OSM_VALUES, "value"),
                  Input(ids.OSM_KEY, "value"))
    def osm_values_for_key(key):
        from ..services.osm import OSM_KEY_VALUES

        values = OSM_KEY_VALUES.get(key or "", [])
        return [{"label": v, "value": v} for v in values], []

    @app.callback(
        Output(ids.OSM_SELECTIONS, "data"),
        Input(ids.OSM_ADD_BTN, "n_clicks"),
        State(ids.OSM_KEY, "value"),
        State(ids.OSM_VALUES, "value"),
        State(ids.OSM_SELECTIONS, "data"),
        prevent_initial_call=True)
    def osm_add_selection(n_clicks, key, values, selections):
        if not n_clicks or not key:
            raise PreventUpdate
        selections = list(selections or [])
        selections.append({"key": key, "values": list(values or [])})
        return selections

    @app.callback(
        Output(ids.OSM_SELECTIONS, "data", allow_duplicate=True),
        Input({"type": ids.TYPE_OSM_SEL_REMOVE, "index": ALL}, "n_clicks"),
        State(ids.OSM_SELECTIONS, "data"),
        prevent_initial_call=True)
    def osm_remove_selection(clicks, selections):
        trigger = ctx.triggered_id
        if not trigger or not any(clicks or []):
            raise PreventUpdate
        index = int(trigger.get("index"))
        selections = list(selections or [])
        if not 0 <= index < len(selections):
            raise PreventUpdate
        selections.pop(index)
        return selections

    @app.callback(Output(ids.OSM_SELECTION_LIST, "children"),
                  Input(ids.OSM_SELECTIONS, "data"))
    def osm_render_selections(selections):
        from ..services.osm import selection_label

        if not selections:
            return html.Small("No combinations added yet.",
                              className="text-muted")
        chips = []
        for i, sel in enumerate(selections):
            chips.append(html.Div([
                html.Span(selection_label(sel), className="me-1"),
                dbc.Button("×", id={"type": ids.TYPE_OSM_SEL_REMOVE,
                                    "index": i},
                           size="sm", color="link",
                           className="p-0 text-danger",
                           title="Remove this combination"),
            ], className="d-inline-block border rounded px-1 me-1 mb-1"))
        return chips

    # ------------------------------------------------------------ clear area
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.STUDY_AREA_INFO, "children", allow_duplicate=True),
        Output(ids.DRAW_CONTROL, "editToolbar"),
        Input(ids.STUDY_AREA_CLEAR, "n_clicks"),
        prevent_initial_call=True)
    def clear_area(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        state.study_area = None
        state.remove_layer(_STUDY_AREA_LAYER_ID)
        toolbar = {"mode": "remove", "action": "clear all",
                   "n_clicks": n_clicks}
        return state.layers_view(), _study_area_info(state), toolbar

    # ---------------------------------------------------------- project CRS
    @app.callback(
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.PROJECT_CRS, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def set_project_crs(crs, notices):
        notices = list(notices or [])
        if not crs:
            raise PreventUpdate
        problem = validate.validate_project_crs(crs)
        if problem is not None:
            notices.append(problem.to_dict())
            return notices
        state.project_crs = crs
        return notices

    # ----------------------------------------------------------- local load
    # LOCAL_LOAD_STATUS sits inside a dcc.Loading, so its pending update
    # shows a spinner next to the button until the load fully finishes.
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.DATASET_LIST, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.LOCAL_LOAD_STATUS, "children"),
        Input(ids.LOCAL_LOAD_BTN, "n_clicks"),
        State(ids.LOCAL_PATH, "value"),
        State(ids.LOCAL_LAYER, "value"),
        State(ids.CLIP_TO_AREA, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def load_local(n_clicks, path, sub_layer, clip, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        ext = os.path.splitext(str(path or ""))[1].lower()
        kind = "raster" if ext in data_io.RASTER_EXTS else "vector"
        problem = validate.validate_file(path, kind)
        if problem is not None:
            notices.append(problem.to_dict())
            return no_update, no_update, notices, ""
        if kind == "raster":
            from .raster import add_raster_layer
            layer, notices = add_raster_layer(state, path, notices)
            if layer is None:
                return no_update, no_update, notices, ""
            return state.layers_view(), _dataset_list(state), notices, ""

        source = {"kind": "local", "path": str(path),
                  "layer": sub_layer or "", "clip": bool(clip)}
        gdf, notices = _load_from_source(state, source, notices)
        if gdf is None:
            return no_update, no_update, notices, ""
        name = os.path.basename(str(path))
        layer, refreshed = _add_vector_layer(state, gdf, name,
                                             source=source)
        notices.append(success(
            f"{'Refreshed' if refreshed else 'Loaded'} {layer.name}",
            meaning=f"{len(gdf)} features."))
        return state.layers_view(), _dataset_list(state), notices, ""

    # ------------------------------------------------------------ WFS preset
    @app.callback(
        Output(ids.WFS_URL, "value"),
        Output(ids.WFS_LAYER, "value"),
        Input(ids.WFS_PRESET, "value"),
        prevent_initial_call=True)
    def apply_preset(preset):
        if not preset:
            raise PreventUpdate
        url, layer = preset.split("|", 1)
        return url, layer

    # ------------------------ viewport + category preset filtering (Germany)
    @app.callback(
        Output(ids.WFS_PRESET, "options"),
        Input(ids.MAP, "bounds"),
        Input(ids.WFS_CATEGORY, "value"),
        Input(ids.WFS_IN_VIEW_ONLY, "value"))
    def filter_presets(bounds, category, in_view_only):
        """Only list servers covering the current map view (+ category)."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..layout import wfs_option
        from ..presets import MAP_SERVICES, services_in_view

        # panning must not recompute/reship the list when the filter is off
        if ctx.triggered_id == ids.MAP and not in_view_only:
            raise PreventUpdate
        services = [s for s in MAP_SERVICES if s["service"] == "wfs"]
        if category:
            services = [s for s in services if s["category"] == category]
        view_bbox = None
        if in_view_only and bounds:
            (south, west), (north, east) = bounds
            view_bbox = [west, south, east, north]
        shown = services_in_view(view_bbox, services)
        return [{"label": "Custom…", "value": ""}] + \
            [wfs_option(s) for s in shown]

    # ------------------------ discover a server's feature types (GetCapabilities)
    @app.callback(
        Output(ids.WFS_LAYER_SELECT, "options"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.WFS_CAPS_BTN, "n_clicks"),
        State(ids.WFS_URL, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def list_feature_types(n_clicks, url, notices):
        from ..services import catalog

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        if not url:
            notices.append(Notice(
                severity="warning", title="Enter a WFS URL first",
                meaning="Feature types are read from the server's "
                        "GetCapabilities.",
                impact="Nothing was listed.",
                fix="Pick a preset or type a WFS URL, then 'List layers'.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, notices
        result, notices = guard(catalog.wfs_feature_types, url,
                                notices=notices)
        if result is None:
            return no_update, notices
        options = [{"label": f"{t['title']} ({t['name']})", "value": t["name"]}
                   for t in result]
        notices.append(success(
            f"{len(result)} feature type(s) found",
            meaning="Pick the land-use / target layer from the dropdown."))
        return options, notices

    @app.callback(
        Output(ids.WFS_LAYER, "value", allow_duplicate=True),
        Input(ids.WFS_LAYER_SELECT, "value"),
        prevent_initial_call=True)
    def pick_feature_type(value):
        if not value:
            raise PreventUpdate
        return value

    def _add_wms_overlay(preset, notices):
        from ..services import catalog
        from ..presets import find_service

        _service, url, layer = preset.split("|", 2)
        svc = find_service(url, layer)
        name = svc["name"] if svc else f"WMS {layer}"
        wms = catalog.wms_tile_url(url, layer)
        area = data_io.study_area_bounds(state.study_area, "EPSG:4326")
        clipped = ""
        if area is not None:                       # cover the whole study area
            # expand outward (degrees) so edge tiles intersecting the area load
            minx, miny, maxx, maxy = _expand_bounds(area, floor=0.0005)
            wms["bounds"] = [[miny, minx], [maxy, maxx]]
            clipped = " (covering the study area)"
        state.add_layer(name, "wms", crs="EPSG:3857",
                        meta={"wms": wms,
                              "attribution": svc.get("attribution", "")
                              if svc else ""})
        notices.append(success(
            f"Overlay '{name}' added{clipped}",
            meaning="A live WMS tile overlay; set its opacity in the Raster "
                    "tab, or remove it in the Layers tab."))
        return notices

    def _load_dem_for_area(preset, notices):
        """Download the DGM DEM for the study area. Returns (view, notices)."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..services import catalog
        from ..presets import DEFAULT_DEM, find_service

        if state.study_area is None:
            notices.append(Notice(
                severity="warning", title="Draw a study area first",
                meaning="The DEM is downloaded for the drawn area (a full "
                        "state would be far too large).",
                impact="No DEM was loaded.",
                fix="Draw a rectangle/polygon on the map, then retry.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, notices
        url, coverage = DEFAULT_DEM["url"], DEFAULT_DEM["coverage"]
        dem_crs, axis = DEFAULT_DEM["crs"], tuple(DEFAULT_DEM["axis"])
        svc = None
        if preset:
            service, purl, player = preset.split("|", 2)
            if service == "wcs":
                url, coverage = purl, player or coverage
                svc = find_service(purl, player)
                if svc:
                    dem_crs = svc.get("crs", dem_crs)
                    axis = tuple(svc.get("axis", axis))
        # the WCS SUBSET is in the coverage's native CRS (DGM200 = EPSG:25832);
        # expand outward (metres) so every DEM cell intersecting the study area
        # is returned (the SUBSET otherwise drops centre-outside edge cells)
        bounds = data_io.study_area_bounds(state.study_area, dem_crs)
        if bounds is None:
            return no_update, notices
        # the 1 m state models can get huge fast — pre-check the request size
        # against the service's documented per-request limit / a sanity cap
        res = float((svc or {}).get("res") or 200)
        margin = 300.0 if res >= 25 else max(10.0, 5 * res)
        bounds = _expand_bounds(bounds, floor=margin)
        n_px = (abs(bounds[2] - bounds[0]) / res) * \
            (abs(bounds[3] - bounds[1]) / res)
        max_px = (svc or {}).get("max_px")
        if max_px and n_px > max_px:
            side_km = (max_px ** 0.5) * res / 1000.0
            notices.append(Notice(
                severity="error", title="Study area too large for this "
                "elevation service",
                meaning=f"The request would be ~{n_px / 1e6:,.0f} Mpx at "
                        f"{res:g} m, but the server caps one request at "
                        f"{max_px / 1e6:,.0f} Mpx (≈{side_km:.1f} km × "
                        f"{side_km:.1f} km).",
                impact="No DEM/DSM was downloaded.",
                fix="Draw a smaller study area, or pick a coarser service "
                    "(e.g. the nationwide DGM200).",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, notices
        if n_px > 100e6:                     # ~400 MB float32 — warn, proceed
            notices.append(Notice(
                severity="warning", title="Large elevation download",
                meaning=f"~{n_px / 1e6:,.0f} Mpx at {res:g} m — roughly "
                        f"{n_px * 4 / 1e6:,.0f} MB.",
                impact="The download and tiling may take a while.",
                fix="Consider a smaller study area if this stalls.",
                focus_id=ids.TAB_DATA).to_dict())
        result, notices = guard(
            catalog.load_dem_raster, url, coverage, bounds, axis_labels=axis,
            work_dir=state.work_dir, notices=notices)
        if result is None:
            return no_update, notices
        from .raster import add_raster_layer
        is_dsm = "DSM" in (svc or {}).get("category", "")
        name = (svc["name"].split("(")[0].strip() if svc
                else "DEM (DGM)")
        layer, notices = add_raster_layer(
            state, result, notices,
            name=name if svc else "DEM (DGM)",
            colormap="terrain" if not is_dsm else "gist_earth")
        if layer is None:
            return no_update, notices
        return {"fit_bounds": layer.tile.bounds, "seq": 1}, notices

    # ---------- overlays & DEM: WMS -> live overlay, WCS -> DEM for area (43)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.OVERLAY_LOAD_BTN, "n_clicks"),
        State(ids.OVERLAY_PRESET, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def load_overlay(n_clicks, preset, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        if not preset:
            notices.append(Notice(
                severity="warning", title="Pick a service",
                meaning="Choose a WMS overlay or the DGM DEM from the dropdown.",
                impact="Nothing was added.",
                fix="Select a service, then 'Add to map'.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, no_update, notices
        service = preset.split("|", 1)[0]
        if service == "wms":
            notices = _add_wms_overlay(preset, notices)
            return state.layers_view(), no_update, notices
        view, notices = _load_dem_for_area(preset, notices)   # wcs/DEM
        return (state.layers_view() if view is not no_update else no_update,
                view, notices)

    # explicit "Load DEM for area" button (uses the default DGM if none picked)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.DEM_LOAD_BTN, "n_clicks"),
        State(ids.OVERLAY_PRESET, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def load_dem(n_clicks, preset, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        view, notices = _load_dem_for_area(preset, notices)
        return (state.layers_view() if view is not no_update else no_update,
                view, notices)

    # ---------------------------- OpenStreetMap features (Overpass, edit base)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.DATASET_LIST, "children", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.OSM_LOAD_BTN, "n_clicks"),
        State(ids.OSM_PRESET, "value"),
        State(ids.OSM_TAGS, "value"),
        State(ids.NOTICES, "data"),
        State(ids.OSM_SELECTIONS, "data"),
        prevent_initial_call=True)
    def load_osm(n_clicks, preset, tags, notices, selections=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..services import osm

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        # the categorized menu wins; presets / raw tags are the fallback
        filters = osm.filters_from_selections(selections or [])
        label = "; ".join(osm.selection_label(s) for s in (selections or []))
        if not filters:
            filters = osm.custom_filters(tags) if tags else \
                osm.preset_filters(preset or "")
            label = ""
        if not filters:
            notices.append(Notice(
                severity="warning", title="Pick an OSM feature type",
                meaning="Add a column/value combination above, or choose a "
                        "preset / tag (e.g. landuse=forest).",
                impact="Nothing was loaded.",
                fix="Pick a feature column and value, then '+ Add "
                    "combination'.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, no_update, no_update, notices
        bounds = data_io.study_area_bounds(state.study_area, "EPSG:4326")
        if bounds is None:
            notices.append(Notice(
                severity="warning", title="Draw a study area first",
                meaning="OSM queries are bounded to the drawn area so they stay "
                        "small and fast.",
                impact="Nothing was loaded.",
                fix="Draw a rectangle/polygon on the map, then retry.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, no_update, no_update, notices
        west, south, east, north = bounds
        label = (label or tags or preset or "OSM").strip()
        result, notices = guard(
            osm.load_osm_features, (south, west, north, east), filters,
            notices=notices)
        if result is None:
            return no_update, no_update, no_update, notices
        # OSM is WGS84; reproject into the routing CRS so it rasterizes in metres
        gdf = result.to_crs(state.project_crs)
        source = {"kind": "osm", "filters": filters, "bbox": list(bounds),
                  "clip": False}
        layer, refreshed = _add_vector_layer(state, gdf, f"OSM: {label}",
                                             source=source)
        notices.append(success(
            f"Loaded {len(gdf)} OSM feature(s) — {label}",
            meaning="An editable vector layer; pick it as a Cost-model dataset, "
                    "edit it in the 📋 table, or route/select on it."))
        bounds_ll = geo.wgs84_bounds(layer.geojson)
        view = ({"fit_bounds": bounds_ll, "seq": n_clicks} if bounds_ll
                else no_update)
        return state.layers_view(), _dataset_list(state), view, notices

    # -------------------------------------------------------------- WFS load
    # WFS_LOAD_STATUS sits inside a dcc.Loading → spinner during the download
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.DATASET_LIST, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.WFS_LOAD_STATUS, "children"),
        Input(ids.WFS_LOAD_BTN, "n_clicks"),
        State(ids.WFS_URL, "value"),
        State(ids.WFS_LAYER, "value"),
        State(ids.CLIP_TO_AREA, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def load_wfs(n_clicks, url, layer, clip, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        problem = validate.validate_wfs_url(url)
        if problem is not None:
            notices.append(problem.to_dict())
            return no_update, no_update, notices, ""
        if not layer:
            notices.append(Notice(
                severity="warning", title="No WFS layer name",
                meaning="A WFS request needs a layer (feature type) name.",
                impact="Nothing was loaded.",
                fix="Enter the layer name or pick a preset.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, no_update, notices, ""
        if state.study_area is None and clip:
            notices.append(Notice(
                severity="warning", title="Draw a study area first",
                meaning="WFS layers can be huge; loads are clipped to the "
                        "drawn study area.",
                impact="Nothing was loaded.",
                fix="Draw a rectangle/polygon on the map, then retry.",
                focus_id=ids.TAB_DATA).to_dict())
            return no_update, no_update, notices, ""
        source = {"kind": "wfs", "url": str(url), "layer": str(layer),
                  "clip": bool(clip)}
        gdf, notices = _load_from_source(state, source, notices)
        if gdf is None:
            return no_update, no_update, notices, ""
        added, refreshed = _add_vector_layer(state, gdf, layer,
                                             source=source)
        notices.append(success(
            f"{'Refreshed' if refreshed else 'Loaded'} WFS layer "
            f"{added.name}",
            meaning=f"{len(gdf)} features."))
        return state.layers_view(), _dataset_list(state), notices, ""

    # ------------------------------------------------------ dataset refresh
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.DATASET_LIST, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input({"type": ids.TYPE_DATASET_REFRESH, "index": ALL}, "n_clicks"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def refresh_dataset(n_clicks, notices):

        notices = list(notices or [])
        trigger = ctx.triggered_id
        if not trigger or not any(n_clicks or []):
            raise PreventUpdate
        layer = state.get(trigger["index"])
        if layer is None or not layer.meta.get("source"):
            raise PreventUpdate
        source = layer.meta["source"]
        gdf, notices = _load_from_source(state, source, notices)
        if gdf is None:
            return no_update, no_update, notices
        _add_vector_layer(state, gdf, layer.name, source=source)
        notices.append(success(f"Refreshed {layer.name}",
                               meaning=f"{len(gdf)} features."))
        return state.layers_view(), _dataset_list(state), notices

    # ------------------------------------------------- map fit-bounds bridge
    @app.callback(Output(ids.MAP, "viewport"),
                  Input(ids.MAP_VIEW, "data"),
                  prevent_initial_call=True)
    def apply_viewport(view):
        if not view or not view.get("fit_bounds"):
            raise PreventUpdate
        return {"bounds": view["fit_bounds"],
                "transition": "flyToBounds"}
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

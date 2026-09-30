"""
PYORPS webviz: Dash app construction and interactive callbacks.

Three interactive surfaces share the map:

* **Attributes** — click a route/shape/marker to inspect its properties.
* **Route builder** — drop sources/targets/waypoints (any cardinality), pick
  algorithm + CPU/GPU, and run a fresh PathFinder to add routes.
* **Route editor** — pick an existing route and drag its source/target/waypoint
  markers (or add waypoints); pyorps recomputes the least-cost path through the
  ordered control points. The active route dashes while it recomputes.

A single map-click dispatcher routes clicks to the builder (priority) or the
editor depending on the active mode. State lives on the ``RouteViewer`` captured
by closure — appropriate for a local, single-user desktop tool.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import geopandas as gpd
from dash import (ALL, Dash, Input, Output, State, callback_context, dash_table,
                  html, no_update)
from pyproj import Transformer
from shapely.geometry import LineString

from pyorps.core.exceptions import NoPathFoundError

from . import cost, geo
from .builder import run_routing
from .layout import DASH_PATTERN, build_layout
from .reroute import route_through_points

WGS84 = "EPSG:4326"
DEFAULT_BUILD_BUFFER_M = 500.0


def build_app(viewer) -> Dash:
    """Create the Dash app for a RouteViewer and register its callbacks."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
               title=viewer.title, suppress_callback_exceptions=True)
    app.layout = build_layout(viewer)
    _register_callbacks(app, viewer)
    return app


# --------------------------------------------------------------------- helpers
def _target_crs(viewer):
    if viewer.route_crs is not None:
        return viewer.route_crs
    if viewer.cost_handler is not None:
        return viewer.cost_handler.raster_dataset.crs
    if viewer.raster_layers:
        # fall back to the served raster's CRS
        import rasterio
        with rasterio.open(viewer.raster_layers[0].source_path) as ds:
            return ds.crs
    return WGS84


def _cost_view(rc) -> html.Div:
    rows = [
        ("Length", f"{rc.total_length:,.1f} m"),
        ("Total cost", f"{rc.total_cost:,.0f}"),
        ("Cell cost (raw)", f"{rc.total_cell_cost:,.0f}"),
        ("Geometric length", f"{rc.geodesic_length_m:,.1f} m"),
    ]
    items = [html.Tr([html.Td(k, className="text-muted pe-3"),
                      html.Td(v, className="fw-bold")]) for k, v in rows]
    children = [html.Table(html.Tbody(items), className="table table-sm mb-1")]
    if rc.crosses_forbidden:
        children.append(dbc.Alert(
            f"⚠ Route crosses {rc.n_forbidden_cells} excluded cell(s).",
            color="danger", className="py-1 px-2 mb-1 small"))
    if rc.length_by_category:
        cat_rows = [{"cost": f"{c:g}", "length_m": f"{v:,.1f}"}
                    for c, v in sorted(rc.length_by_category.items())]
        children.append(dash_table.DataTable(
            data=cat_rows,
            columns=[{"name": "cost/m", "id": "cost"},
                     {"name": "length [m]", "id": "length_m"}],
            style_cell={"fontSize": "12px", "padding": "2px 6px"},
            style_header={"fontWeight": "bold"}, style_as_list_view=True))
    return html.Div(children)


def _attr_view(feature: dict) -> html.Div:
    props = (feature or {}).get("properties") or {}
    if not props:
        return html.Small("No attributes on this feature.", className="text-muted")
    rows = [{"field": str(k), "value": str(v)} for k, v in props.items()]
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": "field", "id": "field"}, {"name": "value", "id": "value"}],
        style_cell={"fontSize": "12px", "padding": "2px 6px", "textAlign": "left",
                    "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "bold"}, style_as_list_view=True, page_size=15)


def _div_icon(color: str, size: int = 16) -> dict:
    return {
        "html": (f"<div style='background:{color};width:{size}px;"
                 f"height:{size}px;border:2px solid white;border-radius:50%;"
                 "box-shadow:0 0 3px rgba(0,0,0,.6)'></div>"),
        "className": "pyorps-ctrl-marker",
        "iconSize": [size, size], "iconAnchor": [size // 2, size // 2],
    }


def _control_markers(source, target, waypoints) -> list:
    """Colored DivMarkers marking the active route's control points.

    Non-draggable on purpose: dash-leaflet does not sync a dragged marker's
    position back to the server, so moves are driven by map clicks instead.
    """
    markers = [
        dl.DivMarker(position=source, iconOptions=_div_icon("#2ca02c", 18),
                     children=dl.Tooltip("Source")),
        dl.DivMarker(position=target, iconOptions=_div_icon("#d62728", 18),
                     children=dl.Tooltip("Target")),
    ]
    for i, wp in enumerate(waypoints):
        markers.append(dl.DivMarker(
            position=wp, iconOptions=_div_icon("#1f77b4", 14),
            children=dl.Tooltip(f"Waypoint {i + 1}")))
    return markers


def _builder_markers(store) -> list:
    """Non-draggable markers showing points placed while building a routing."""
    markers = []
    for role, color, size in (("sources", "#2ca02c", 16),
                              ("targets", "#d62728", 16),
                              ("waypoints", "#1f77b4", 12)):
        for i, pt in enumerate(store.get(role) or []):
            markers.append(dl.DivMarker(
                position=pt, iconOptions=_div_icon(color, size),
                children=dl.Tooltip(f"{role[:-1].title()} {i + 1}")))
    return markers


def _builder_rows(store) -> list:
    rows = []
    for role, kind in (("sources", "source"), ("targets", "target"),
                       ("waypoints", "waypoint")):
        for pt in store.get(role) or []:
            rows.append({"kind": kind, "lat": round(pt[0], 6), "lng": round(pt[1], 6)})
    return rows


def _rows_to_builder(rows) -> dict:
    out = {"sources": [], "targets": [], "waypoints": []}
    kmap = {"source": "sources", "target": "targets", "waypoint": "waypoints"}
    for r in rows or []:
        key = kmap.get(r.get("kind"))
        if key:
            out[key].append([float(r["lat"]), float(r["lng"])])
    return out


# ------------------------------------------------------------------ callbacks
def _register_callbacks(app: Dash, viewer) -> None:
    target_crs = _target_crs(viewer)
    to_crs = Transformer.from_crs(WGS84, target_crs, always_xy=True)
    to_wgs = Transformer.from_crs(target_crs, WGS84, always_xy=True)

    def crs_to_latlng(xy):
        lon, lat = to_wgs.transform(xy[0], xy[1])
        return [float(lat), float(lon)]

    def latlng_to_crs(latlng):
        x, y = to_crs.transform(latlng[1], latlng[0])
        return (float(x), float(y))

    # -- raster opacity ------------------------------------------------------
    if viewer.raster_layers:
        @app.callback(Output({"type": "raster-tile", "index": ALL}, "opacity"),
                      Input("raster-opacity", "value"))
        def _set_opacity(value):
            return [value] * len(viewer.raster_layers)

    # -- attribute inspector -------------------------------------------------
    attr_inputs = [Input("route-layer", "clickData"),
                   Input("endpoints-layer", "clickData")]
    if viewer.shape_layers:
        attr_inputs.append(Input({"type": "shape-layer", "index": ALL}, "clickData"))

    @app.callback(Output("attr-panel", "children"), *attr_inputs,
                  prevent_initial_call=True)
    def _show_attrs(*_values):
        trig = callback_context.triggered
        if not trig:
            return no_update
        feature = trig[0]["value"]
        if isinstance(feature, list):
            feature = next((f for f in feature if f), None)
        if not feature:
            return no_update
        return _attr_view(feature)

    # -- unified map click: builder (priority) or editor ---------------------
    @app.callback(
        Output("control-store", "data", allow_duplicate=True),
        Output("builder-store", "data", allow_duplicate=True),
        Input("map", "clickData"),
        State("build-mode", "value"), State("edit-mode", "value"),
        State("control-store", "data"), State("builder-store", "data"),
        prevent_initial_call=True,
    )
    def _map_click(click, build_mode, edit_mode, cstore, bstore):
        latlng = (click or {}).get("latlng") or {}
        lat, lng = latlng.get("lat"), latlng.get("lng")
        if lat is None or lng is None:
            return no_update, no_update
        if build_mode in ("source", "target", "waypoint"):
            key = {"source": "sources", "target": "targets",
                   "waypoint": "waypoints"}[build_mode]
            nb = {k: list((bstore or {}).get(k) or [])
                  for k in ("sources", "targets", "waypoints")}
            nb[key].append([lat, lng])
            return no_update, nb
        if edit_mode in ("source", "target", "add") and cstore \
                and cstore.get("active") is not None:
            nc = {**cstore, "waypoints": list(cstore.get("waypoints") or [])}
            if edit_mode == "source":
                nc["source"] = [lat, lng]
            elif edit_mode == "target":
                nc["target"] = [lat, lng]
            else:
                nc["waypoints"].append([lat, lng])
            return nc, no_update
        return no_update, no_update

    _register_builder(app, viewer, latlng_to_crs)
    _register_editor(app, viewer, target_crs, crs_to_latlng, latlng_to_crs)


def _register_builder(app: Dash, viewer, latlng_to_crs) -> None:
    # -- render builder markers + points table -------------------------------
    @app.callback(Output("builder-markers", "children"),
                  Output("builder-points-table", "data"),
                  Input("builder-store", "data"), prevent_initial_call=True)
    def _render_builder(store):
        return _builder_markers(store or {}), _builder_rows(store or {})

    # -- delete builder points (row_deletable) -------------------------------
    @app.callback(Output("builder-store", "data", allow_duplicate=True),
                  Input("builder-points-table", "data"),
                  State("builder-store", "data"), prevent_initial_call=True)
    def _builder_table_edit(rows, store):
        rebuilt = _rows_to_builder(rows)
        if rebuilt == store:
            return no_update
        return rebuilt

    # -- clear all builder points --------------------------------------------
    @app.callback(Output("builder-store", "data", allow_duplicate=True),
                  Input("clear-build-btn", "n_clicks"), prevent_initial_call=True)
    def _clear_builder(_n):
        return {"sources": [], "targets": [], "waypoints": []}

    # -- run the routing -----------------------------------------------------
    @app.callback(
        Output("route-layer", "data", allow_duplicate=True),
        Output("edit-route-select", "options", allow_duplicate=True),
        Output("builder-store", "data", allow_duplicate=True),
        Output("build-status", "children"),
        Input("run-routing-btn", "n_clicks"),
        State("builder-store", "data"), State("build-algorithm", "value"),
        State("build-hardware", "value"), State("build-pairwise", "value"),
        prevent_initial_call=True,
    )
    def _run(_n, store, algorithm, hardware, pairwise):
        if not viewer.raster_layers:
            return no_update, no_update, no_update, "No cost raster loaded."
        sources = store.get("sources") or []
        targets = store.get("targets") or []
        if not sources or not targets:
            return (no_update, no_update, no_update,
                    "Place at least one source and one target, then Run.")
        s_crs = [latlng_to_crs(p) for p in sources]
        t_crs = [latlng_to_crs(p) for p in targets]
        w_crs = [latlng_to_crs(p) for p in (store.get("waypoints") or [])]
        raster_path = viewer.raster_layers[0].source_path
        try:
            finder, built = run_routing(
                raster_path, sources=s_crs, targets=t_crs, waypoints=w_crs,
                algorithm=algorithm, hardware=hardware, pairwise=bool(pairwise),
                search_buffer_m=DEFAULT_BUILD_BUFFER_M)
        except Exception as exc:  # surface backend/GPU/availability errors
            return no_update, no_update, no_update, f"⚠ Routing failed: {exc}"
        if not built:
            return (no_update, no_update, no_update,
                    "No routes found — points may be outside the raster or on "
                    "excluded cells.")
        props = [{"kind": "built", "algorithm": algorithm, "hardware": hardware,
                  "total_length": round(b.cost.total_length, 1),
                  "total_cost": round(b.cost.total_cost, 1)} for b in built]
        viewer.add_route_geometries([b.line for b in built], props)
        if viewer.path_finder is None:
            viewer.path_finder = finder
            viewer.cost_handler = finder.raster_handler
        options = [{"label": f"Route {i}", "value": i}
                   for i in range(len(viewer.route_controls))]
        status = (f"Added {len(built)} route(s) via {algorithm} on "
                  f"{hardware.upper()}.")
        return (viewer.route_geojson, options,
                {"sources": [], "targets": [], "waypoints": []}, status)


def _register_editor(app: Dash, viewer, target_crs, crs_to_latlng,
                     latlng_to_crs) -> None:
    # -- select active route -------------------------------------------------
    @app.callback(Output("control-store", "data", allow_duplicate=True),
                  Input("edit-route-select", "value"), prevent_initial_call=True)
    def _select_route(idx):
        if idx is None or idx >= len(viewer.route_controls):
            return {"active": None, "source": None, "target": None, "waypoints": []}
        ctrl = viewer.route_controls[idx]
        return {"active": idx, "source": crs_to_latlng(ctrl["source"]),
                "target": crs_to_latlng(ctrl["target"]), "waypoints": []}

    # -- waypoint table reorder / delete -------------------------------------
    @app.callback(Output("control-store", "data", allow_duplicate=True),
                  Input("waypoint-table", "data"), State("control-store", "data"),
                  prevent_initial_call=True)
    def _table_edit(rows, store):
        if not store or store.get("active") is None:
            return no_update
        try:
            ordered = sorted(rows or [], key=lambda r: float(r.get("order", 0)))
        except (TypeError, ValueError):
            ordered = rows or []
        new_wps = [[float(r["lat"]), float(r["lng"])] for r in ordered]
        if new_wps == list(store.get("waypoints") or []):
            return no_update
        return {**store, "waypoints": new_wps}

    # -- clear waypoints -----------------------------------------------------
    @app.callback(Output("control-store", "data", allow_duplicate=True),
                  Input("clear-waypoints-btn", "n_clicks"),
                  State("control-store", "data"), prevent_initial_call=True)
    def _clear(_n, store):
        if not store or store.get("active") is None or not store.get("waypoints"):
            return no_update
        return {**store, "waypoints": []}

    # -- render: recompute route through control points ----------------------
    @app.callback(
        Output("active-route", "positions"),
        Output("active-route", "dashArray"),
        Output("control-markers", "children"),
        Output("waypoint-table", "data"),
        Output("cost-stats", "children"),
        Output("edit-status", "children"),
        Input("control-store", "data"), prevent_initial_call=True,
    )
    def _render(store):
        active = (store or {}).get("active")
        source, target = (store or {}).get("source"), (store or {}).get("target")
        if active is None or not source or not target:
            return [], None, [], [], html.Small(
                "Select a route to edit.", className="text-muted"), ""
        waypoints = store.get("waypoints") or []
        markers = _control_markers(source, target, waypoints)
        table = [{"order": i + 1, "lat": round(w[0], 6), "lng": round(w[1], 6)}
                 for i, w in enumerate(waypoints)]
        if viewer.path_finder is None:
            return ([source, *waypoints, target], DASH_PATTERN, markers, table,
                    no_update, "Editing needs a PathFinder (recompute unavailable).")
        pts_crs = [latlng_to_crs(p) for p in [source, *waypoints, target]]
        try:
            line, rc = route_through_points(viewer.path_finder, pts_crs)
        except NoPathFoundError:
            return ([source, *waypoints, target], DASH_PATTERN, markers, table,
                    no_update, "⚠ No path — a control point is outside the "
                    "search area or on an excluded cell.")
        wgs_line = geo.geometry_to_crs(line, target_crs, WGS84)
        positions = [[lat, lng] for lng, lat in wgs_line.coords]
        return (positions, None, markers, table, _cost_view(rc),
                f"Route {active}: {len(waypoints)} waypoint(s), recomputed.")

    # -- client-side: dash the active route while an edit is in flight -------
    app.clientside_callback(
        f"function(_c, _t, _s) {{ return '{DASH_PATTERN}'; }}",
        Output("active-route", "dashArray", allow_duplicate=True),
        Input("map", "clickData"), Input("waypoint-table", "data"),
        Input("edit-route-select", "value"), prevent_initial_call=True,
    )

    # -- export the active (edited) route ------------------------------------
    @app.callback(Output("save-status", "children"), Input("save-btn", "n_clicks"),
                  State("active-route", "positions"), State("save-path", "value"),
                  prevent_initial_call=True)
    def _save(_n, positions, path):
        if not positions or len(positions) < 2:
            return "Nothing to save — select and edit a route first."
        if not path:
            return "Enter an output path (.gpkg / .shp / .geojson)."
        try:
            line = LineString([(lng, lat) for lat, lng in positions])
            gdf = gpd.GeoDataFrame({"kind": ["edited"]}, geometry=[line], crs=WGS84)
            if target_crs and str(target_crs) != WGS84:
                gdf = gdf.to_crs(target_crs)
            gdf.to_file(path)
        except Exception as exc:
            return f"⚠ Save failed: {exc}"
        return f"Saved edited route to {path}"

"""
PYORPS webviz: Dash layout construction.

Two-pane UI — a full-height Leaflet map (basemaps, raster overlay, routes,
shapes, plus the active-route editing layer) and a control/inspection sidebar.

Editing uses a *control-point* model: pick a route, then move its source/target
or add & reorder waypoints; pyorps recomputes the least-cost path through the
ordered control points. The active route is drawn as a dedicated polyline that
dashes while it is being recomputed. All wiring lives in ``app.py``.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import dash_table, dcc, html

# Freely usable XYZ basemaps (attribution required). Google is intentionally
# omitted: its raw tile endpoints violate the Maps ToS. To add Google, use the
# paid Maps Tiles API with a key and insert a BaseLayer here.
BASEMAPS = [
    {"name": "OpenStreetMap",
     "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
     "attribution": "&copy; OpenStreetMap contributors", "default": True},
    {"name": "Esri World Imagery",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Topo",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Carto Light",
     "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
     "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
     "default": False},
    {"name": "Carto Dark",
     "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
     "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
     "default": False},
]

ROUTE_STYLE = {"color": "#e6194b", "weight": 3, "opacity": 0.75}
ROUTE_HOVER = {"weight": 6, "color": "#ffe119"}
ACTIVE_COLOR = "#ffe119"
DASH_PATTERN = "8 8"


def _base_layers() -> list:
    return [
        dl.BaseLayer(
            dl.TileLayer(url=bm["url"], attribution=bm["attribution"], maxZoom=20),
            name=bm["name"], checked=bm["default"])
        for bm in BASEMAPS
    ]


def _raster_overlays(viewer) -> list:
    return [
        dl.Overlay(
            dl.TileLayer(url=layer.tile_url, opacity=0.7,
                         id={"type": "raster-tile", "index": i}, maxZoom=22),
            name=f"\U0001F5FA {layer.name}", checked=True)
        for i, layer in enumerate(viewer.raster_layers)
    ]


def _shape_overlays(viewer) -> list:
    return [
        dl.Overlay(
            dl.GeoJSON(
                data=shape.geojson, id={"type": "shape-layer", "index": i},
                options=dict(style=dict(color=shape.color, weight=1,
                                        fillOpacity=0.25)),
                hoverStyle=dict(weight=3, color="#ffff00")),
            name=f"\U0001F4D0 {shape.name}", checked=True)
        for i, shape in enumerate(viewer.shape_layers)
    ]


def _route_layers(viewer) -> list:
    # Direct map children (NOT inside LayersControl): a GeoJSON nested in a
    # LayersControl Overlay does not reliably re-render when its `data` changes
    # via callback, which hid newly built/edited routes. As direct children their
    # `data` updates apply immediately. Always present (possibly empty).
    empty = {"type": "FeatureCollection", "features": []}
    return [
        dl.GeoJSON(data=viewer.route_geojson or empty, id="route-layer",
                   options=dict(style=ROUTE_STYLE), hoverStyle=ROUTE_HOVER,
                   zoomToBounds=viewer.route_geojson is not None),
        dl.GeoJSON(data=viewer.endpoints_geojson or empty, id="endpoints-layer"),
    ]


def _initial_view(viewer):
    if viewer.raster_layers:
        (s, w), (n, e) = viewer.raster_layers[0].bounds
        return [(s + n) / 2, (w + e) / 2], 14
    return [51.0, 10.0], 6


def _map(viewer):
    center, zoom = _initial_view(viewer)
    return dl.Map(
        id="map", center=center, zoom=zoom,
        style={"height": "100vh", "width": "100%"},
        children=[
            dl.LayersControl(
                id="layers-control", position="topright",
                children=_base_layers() + _raster_overlays(viewer)
                + _shape_overlays(viewer)),
            # Routes + endpoints as direct children so callback data updates render.
            *_route_layers(viewer),
            # Active-route editing layer: the selected route + its control points.
            dl.Polyline(id="active-route", positions=[], color=ACTIVE_COLOR,
                        weight=5, opacity=0.95),
            dl.LayerGroup(id="control-markers", children=[]),
            # Route-builder: markers for points placed while defining a routing.
            dl.LayerGroup(id="builder-markers", children=[]),
            dl.ScaleControl(position="bottomleft"),
        ],
    )


def _edit_card(viewer):
    can_edit = viewer.path_finder is not None and bool(viewer.route_controls)
    route_options = [{"label": f"Route {i}", "value": i}
                     for i in range(len(viewer.route_controls))]
    hint = ("Pick a route, choose what a map click does, then click the map: "
            "move the source/target or add a waypoint. Reorder waypoints by "
            "editing the # column; delete with the row's ✗. The route recomputes "
            "through the ordered points." if can_edit else
            "Attach a PathFinder with routes to enable editing.")
    return dbc.Card(dbc.CardBody([
        html.H6("Edit route", className="card-title"),
        dbc.Label("Active route", html_for="edit-route-select", className="small"),
        dcc.Dropdown(id="edit-route-select", options=route_options, value=None,
                     placeholder="Select a route to edit…",
                     disabled=not can_edit, clearable=True),
        dbc.Label("Map click moves…", className="small mt-2"),
        dbc.RadioItems(
            id="edit-mode",
            options=[
                {"label": " Off", "value": "off"},
                {"label": " Source", "value": "source"},
                {"label": " Target", "value": "target"},
                {"label": " + Waypoint", "value": "add"},
            ],
            value="off", inline=True, className="small"),
        html.Div([
            dbc.Label("Waypoints (edit # to reorder)", className="small mt-2"),
            dash_table.DataTable(
                id="waypoint-table",
                columns=[
                    {"name": "#", "id": "order", "type": "numeric",
                     "editable": True},
                    {"name": "lat", "id": "lat", "editable": False},
                    {"name": "lng", "id": "lng", "editable": False},
                ],
                data=[], row_deletable=True,
                style_cell={"fontSize": "12px", "padding": "2px 6px"},
                style_header={"fontWeight": "bold"},
                style_as_list_view=True),
            dbc.Button("Clear waypoints", id="clear-waypoints-btn", size="sm",
                       color="secondary", className="mt-2"),
        ]),
        html.Div(id="edit-status", className="mt-2 small text-info"),
    ]), className="mb-2")


def _build_card(viewer):
    can_build = bool(viewer.raster_layers)
    hint = ("Set a mode, click the map to drop source(s), target(s) and "
            "waypoints, then Run. Any number of sources/targets is allowed "
            "(single→single, single→multi, multi→multi)."
            if can_build else "Load a cost raster to build routings.")
    return dbc.Card(dbc.CardBody([
        html.H6("New routing", className="card-title"),
        dbc.RadioItems(
            id="build-mode",
            options=[
                {"label": " Off", "value": "off"},
                {"label": " + Source", "value": "source"},
                {"label": " + Target", "value": "target"},
                {"label": " + Waypoint", "value": "waypoint"},
            ],
            value="off", inline=True, className="small"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Algorithm", className="small mt-2"),
                dcc.Dropdown(
                    id="build-algorithm", clearable=False,
                    options=[
                        {"label": "Dijkstra", "value": "dijkstra"},
                        {"label": "Bidirectional Dijkstra",
                         "value": "bidirectional_dijkstra"},
                        {"label": "Delta-stepping (fastest)",
                         "value": "delta-stepping"},
                    ], value="delta-stepping"),
            ], width=7),
            dbc.Col([
                dbc.Label("Hardware", className="small mt-2"),
                dbc.RadioItems(
                    id="build-hardware",
                    options=[{"label": " CPU", "value": "cpu"},
                             {"label": " GPU", "value": "gpu"}],
                    value="cpu", className="small"),
            ], width=5),
        ]),
        dbc.Checkbox(id="build-pairwise", label="Pairwise (source i → target i)",
                     value=False, className="small mt-1"),
        dash_table.DataTable(
            id="builder-points-table",
            columns=[{"name": "kind", "id": "kind"}, {"name": "lat", "id": "lat"},
                     {"name": "lng", "id": "lng"}],
            data=[], row_deletable=True,
            style_cell={"fontSize": "12px", "padding": "2px 6px"},
            style_header={"fontWeight": "bold"}, style_as_list_view=True),
        html.Div([
            dbc.Button("Run routing", id="run-routing-btn", size="sm",
                       color="primary", className="mt-2 me-2",
                       disabled=not can_build),
            dbc.Button("Clear", id="clear-build-btn", size="sm",
                       color="secondary", className="mt-2"),
        ]),
        html.Div(id="build-status", className="mt-2 small text-info"),
        html.Small(hint, className="text-muted"),
    ]), className="mb-2")


def _sidebar(viewer):
    has_raster = bool(viewer.raster_layers)
    raster_card = dbc.Card(dbc.CardBody([
        html.H6("Cost raster", className="card-title"),
        html.Div(
            [html.Small(f"range {viewer.raster_layers[0].vmin:g} – "
                        f"{viewer.raster_layers[0].vmax:g}")]
            if has_raster else [html.Small("no raster loaded")],
            className="text-muted"),
        html.Label("Overlay opacity", className="mt-2"),
        dcc.Slider(id="raster-opacity", min=0, max=1, step=0.05, value=0.7,
                   marks={0: "0", 0.5: "0.5", 1: "1"}),
    ]), className="mb-2")

    cost_card = dbc.Card(dbc.CardBody([
        html.H6("Route cost", className="card-title"),
        html.Div(id="cost-stats", children=html.Small(
            "Select and edit a route to see its length and cost.",
            className="text-muted")),
    ]), className="mb-2")

    attr_card = dbc.Card(dbc.CardBody([
        html.H6("Attributes", className="card-title"),
        html.Div(id="attr-panel", children=html.Small(
            "Click a route or shape to inspect its attributes.",
            className="text-muted")),
    ]), className="mb-2")

    export_card = dbc.Card(dbc.CardBody([
        html.H6("Export edited route", className="card-title"),
        dbc.Input(id="save-path", type="text", size="sm",
                  placeholder="e.g. C:/out/edited_route.gpkg", className="mb-2"),
        dbc.Button("Save", id="save-btn", size="sm", color="success"),
        html.Div(id="save-status", className="mt-2 small text-success"),
    ]))

    return html.Div(
        [html.H4("PYORPS Route Viewer", className="mt-2"), html.Hr(),
         raster_card, _build_card(viewer), _edit_card(viewer), cost_card,
         attr_card, export_card],
        style={"height": "100vh", "overflowY": "auto", "padding": "0.75rem"})


def build_layout(viewer):
    """Assemble the full app layout for a RouteViewer."""
    stores = [
        dcc.Store(id="route-store", data=viewer.route_geojson),
        # Control points of the active route, in WGS84 (map) coordinates.
        dcc.Store(id="control-store",
                  data={"active": None, "source": None, "target": None,
                        "waypoints": []}),
        # Points placed while building a new routing (WGS84).
        dcc.Store(id="builder-store",
                  data={"sources": [], "targets": [], "waypoints": []}),
    ]
    return dbc.Container(
        stores + [
            dbc.Row(
                [dbc.Col(_map(viewer), md=8, lg=9, className="p-0"),
                 dbc.Col(_sidebar(viewer), md=4, lg=3, className="p-0")],
                className="g-0"),
        ],
        fluid=True, className="p-0")

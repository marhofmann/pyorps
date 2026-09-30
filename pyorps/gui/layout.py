"""
PYORPS GUI: layout construction (pure components, no callbacks).

Left: the Leaflet map. The background map is a SINGLE ``basemap-tile`` layer
(direct Map child) whose source, opacity and z-order are picked in the
map-corner 🗺 control; every dynamic layer is rendered into the ``layer-host``
LayerGroup which
is a DIRECT child of the Map (hard rule C1 — layers nested in
LayersControl/Overlay do not re-render on callback updates). Drawing uses an
EditControl in its own FeatureGroup (C4). Right: the tabbed sidebar. All
component ids come from :mod:`pyorps.gui.ids`.
"""
from __future__ import annotations

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import dcc, html

from . import ids
from .services.tiles import COLORMAPS

# Freely usable XYZ basemaps (attribution required), ordered by provider
# (``group``) so related maps sit together in the map-corner switcher. URL
# templates follow the classic QGIS/xyzservices connection list.
_OSM_ATTR = "&copy; OpenStreetMap contributors"
_CARTO_ATTR = "&copy; OpenStreetMap contributors &copy; CARTO"
_BKG_ATTR = "Map data: &copy; dl-de/by-2-0, &copy; BKG"
BASEMAPS = [
    {"name": "OpenStreetMap", "group": "OpenStreetMap",
     "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
     "attribution": _OSM_ATTR, "default": True},
    {"name": "OpenStreetMap DE", "group": "OpenStreetMap",
     "url": "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
     "attribution": _OSM_ATTR, "default": False},
    {"name": "OpenTopoMap", "group": "OpenStreetMap",
     "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
     "attribution": _OSM_ATTR + ", SRTM | &copy; OpenTopoMap (CC-BY-SA)",
     "default": False},
    {"name": "CyclOSM", "group": "OpenStreetMap",
     "url": "https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/"
            "{z}/{x}/{y}.png",
     "attribution": "CyclOSM | " + _OSM_ATTR, "default": False},
    {"name": "BaseMapDE Color", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/"
            "de_basemapde_web_raster_farbe/default/GLOBAL_WEBMERCATOR/"
            "{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "BaseMapDE Grey", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/"
            "de_basemapde_web_raster_grau/default/GLOBAL_WEBMERCATOR/"
            "{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "TopPlusOpen Color", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_topplus_open/tile/1.0.0/"
            "web/default/WEBMERCATOR/{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "TopPlusOpen Grey", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_topplus_open/tile/1.0.0/"
            "web_grau/default/WEBMERCATOR/{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "Esri World Imagery", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Topo", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Street Map", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Gray Canvas", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri NatGeo World Map", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri &mdash; National Geographic",
     "default": False},
    {"name": "Esri World Shaded Relief", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Carto Light", "group": "Carto",
     "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
     "attribution": _CARTO_ATTR, "default": False},
    {"name": "Carto Dark", "group": "Carto",
     "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
     "attribution": _CARTO_ATTR, "default": False},
    {"name": "Carto Voyager", "group": "Carto",
     "url": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/"
            "{z}/{x}/{y}.png",
     "attribution": _CARTO_ATTR, "default": False},
    {"name": "Google Maps", "group": "Google",
     "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
     "attribution": "&copy; Google", "default": False},
    {"name": "Google Satellite", "group": "Google",
     "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
     "attribution": "&copy; Google", "default": False},
    {"name": "Google Hybrid", "group": "Google",
     "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
     "attribution": "&copy; Google", "default": False},
    {"name": "Google Terrain", "group": "Google",
     "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
     "attribution": "&copy; Google", "default": False},
]

#: 1x1 transparent png — the "None" basemap tile (same URL for every tile)
BLANK_TILE = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAA"
              "fFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

BASEMAP_BY_NAME = {bm["name"]: bm for bm in BASEMAPS}
DEFAULT_BASEMAP = "OpenStreetMap"
#: raster overlays paint at this leaflet zIndex; the basemap sits below (1) by
#: default and can be raised above them (500) from the Layers tab.
RASTER_TILE_ZINDEX = 250
BASEMAP_Z_BELOW = 1
BASEMAP_Z_ABOVE = 500

STUDY_AREA_STYLE = {"color": "#3388ff", "weight": 2, "fill": False,
                    "fillOpacity": 0, "dashArray": "6 6"}
#: leaflet-draw rubber-band style — VISIBLE while drawing so every click and
#: mouse move previews the shape's edges live. The CREATED layer is restyled
#: to invisible by assets/draw-preview.js the moment drawing finishes, so only
#: the styled study-area / cost-polygon host layer remains (no duplicate
#: filled "window" — the round-7 problem that used to force weight/opacity 0
#: here, which also killed the live preview).
DRAW_SHAPE_OPTIONS = {"color": "#3388ff", "weight": 2, "opacity": 0.9,
                      "dashArray": "6 6", "fill": True, "fillOpacity": 0.06}
VECTOR_STYLE = {"color": "#2b8cbe", "weight": 1, "fillOpacity": 0.25}
ROUTE_STYLE = {"color": "#e6194b", "weight": 3, "opacity": 0.85}
ACTIVE_COLOR = "#ffe119"
DASH_PATTERN = "8 8"
#: highlight for a single selected feature (Feature 3)
SELECTED_FEATURE_STYLE = {"color": "#ff00ff", "weight": 4, "fillColor":
                          "#ff00ff", "fillOpacity": 0.25}
_EMPTY_FC = {"type": "FeatureCollection", "features": []}

GRID_STYLE = {"height": "260px", "width": "100%"}
SMALL_GRID_STYLE = {"height": "170px", "width": "100%"}
TALL_GRID_STYLE = {"height": "62vh", "width": "100%"}   # Layers list — see all

_DEFAULT_UI_STATE = {"active_tab": ids.TAB_DATA, "click_mode": "off",
                     "active_route_id": None}
_DEFAULT_ROUTE_DRAFT = {"sources": [], "targets": [], "waypoints": []}


# --------------------------------------------------------------------- stores
def _stores() -> list:
    return [
        dcc.Store(id=ids.UI_STATE, data=dict(_DEFAULT_UI_STATE)),
        dcc.Store(id=ids.LAYERS_VIEW, data=[]),
        dcc.Store(id=ids.MAP_VIEW, data={}),
        dcc.Store(id=ids.NOTICES, data=[]),
        dcc.Store(id=ids.ROUTE_DRAFT, data=dict(_DEFAULT_ROUTE_DRAFT)),
        dcc.Store(id=ids.EDIT_REQUEST, data={}),
        dcc.Store(id=ids.COST_GRID_STATE, data={}),
        dcc.Store(id=ids.RECOMPUTE_FLASH, data=0),
        dcc.Store(id=ids.DRAG_EVENT, data={}),
        dcc.Store(id=ids.FOCUS_FLASH, data={}),
        dcc.Store(id=ids.SELECTED_FEATURE, data={}),
        dcc.Store(id=ids.DRAW_TARGET, data="area"),
        dcc.Store(id=ids.THEME_STORE, data=None),
        dcc.Store(id=ids.DIRTY_STORE, data=False),
        dcc.Store(id=ids.OSM_SELECTIONS, data=[]),
        dcc.Store(id=ids.PPB_CONDS, data=[]),
        # polls the background routing job while one is running
        dcc.Interval(id=ids.ROUTING_POLL, interval=600, disabled=True),
    ]


def _card(title: str, children: list, *, intro: str | None = None,
          className: str = "") -> html.Div:
    """One titled sidebar section card — the building block of every tab."""
    head: list = [html.H6(title, className="gui-card-title")]
    if intro:
        head.append(html.Small(intro, className="gui-card-intro"))
    return html.Div(head + children,
                    className=f"gui-card {className}".strip())


def _collapsible_card(title: str, children: list, *,
                      intro: str | None = None, open: bool = False,
                      className: str = "") -> html.Details:
    """A `_card` whose body (intro + controls) hides behind a native
    <details>/<summary> toggle. Closed by default: only the title row with a
    ▾ caret shows; the caret flips to ▴ while open (pure CSS, no callback).
    """
    body: list = []
    if intro:
        body.append(html.Small(intro, className="gui-card-intro"))
    return html.Details([
        html.Summary([
            html.H6(title, className="gui-card-title"),
            html.Span(className="gi gi-chevron card-caret"),
        ], title=f"Show / hide '{title}'"),
        html.Div(body + children, className="gui-card-body"),
    ], className=f"gui-card gui-card-collapsible {className}".strip(),
        open=open)


def _browse(target_id: str) -> dbc.Button:
    """The 📁 button opening the native file/folder picker for an input."""
    return dbc.Button("📁", id=f"browse-{target_id}", size="sm",
                      color="secondary", outline=True,
                      title="Browse…")


# ------------------------------------------------------------------------ map
def build_map() -> dl.Map:
    default = BASEMAP_BY_NAME[DEFAULT_BASEMAP]
    return dl.Map(
        id=ids.MAP, center=[50.6, 9.0], zoom=7, trackViewport=True,
        style={"height": "100vh", "width": "100%"},
        children=[
            # the single background tile — its source/opacity/z-order are set
            # from the Layers tab; below overlays by default (zIndex 1)
            dl.TileLayer(id=ids.BASEMAP_TILE, url=default["url"],
                         attribution=default["attribution"], maxZoom=20,
                         zIndex=BASEMAP_Z_BELOW),
            # THE dynamic layer host: direct Map child (C1)
            dl.LayerGroup(id=ids.LAYER_HOST, children=[]),
            # study-area drawing (C4) — its own FeatureGroup, direct child;
            # the rubber-band preview is visible (DRAW_SHAPE_OPTIONS) and the
            # finished shape is hidden by assets/draw-preview.js so it never
            # doubles the styled study-area / cost-polygon host layer
            dl.FeatureGroup([
                dl.EditControl(
                    id=ids.DRAW_CONTROL, position="topleft",
                    draw={"rectangle": {"shapeOptions": DRAW_SHAPE_OPTIONS},
                          "polygon": {"shapeOptions": DRAW_SHAPE_OPTIONS},
                          "polyline": False, "circle": False,
                          "circlemarker": False, "marker": False},
                    # edit=move vertices, remove=delete shapes (custom cost
                    # polygons are drawn/edited/deleted here — task 33)
                    edit={"edit": True, "remove": True}),
            ]),
            # active-route editing layer + markers, direct children
            # thick yellow halo so the selected route stands out clearly
            dl.Polyline(id=ids.ACTIVE_ROUTE, positions=[],
                        color=ACTIVE_COLOR, weight=11, opacity=0.85),
            dl.LayerGroup(id=ids.CONTROL_MARKERS, children=[]),
            dl.LayerGroup(id=ids.BUILDER_MARKERS, children=[]),
            # single-selected-feature highlight, on top (Feature 3)
            dl.GeoJSON(id=ids.SELECTED_FEATURE_LAYER, data=_EMPTY_FC,
                       options={"style": SELECTED_FEATURE_STYLE}),
            dl.ScaleControl(position="bottomleft"),
        ],
    )


def _basemap_control() -> html.Details:
    """Map-corner background-map switcher (the classic layers icon, top-right).

    A native <details> panel: the summary is the icon button, the panel lists
    every basemap grouped by provider plus opacity / z-order. Closes on any
    outside click (assets/basemap-control.js).
    """
    body: list = [html.Div("Background map", className="gui-card-title")]
    options = [{"label": bm["name"], "value": bm["name"]} for bm in BASEMAPS]
    options.append({"label": "None (no basemap)", "value": ""})
    body.append(dbc.RadioItems(id=ids.BASEMAP_SELECT, options=options,
                               value=DEFAULT_BASEMAP,
                               className="small basemap-list"))
    body += [
        dbc.Label("Opacity", className="small"),
        dcc.Slider(id=ids.BASEMAP_OPACITY, min=0, max=1, step=0.05,
                   value=1.0, marks={0: "0", 1: "1"}),
        dbc.Label("Z-order", className="small"),
        dbc.Select(id=ids.BASEMAP_ZORDER, value="below", size="sm",
                   options=[
                       {"label": "below overlays", "value": "below"},
                       {"label": "above raster overlays", "value": "above"}]),
    ]
    return html.Details([
        html.Summary(html.Span(className="gi gi-layers"),
                     title="Choose the background map"),
        html.Div(body, className="basemap-panel"),
    ], className="basemap-control")


def _map_pane() -> html.Div:
    return html.Div([
        build_map(),
        dbc.Badge("mode: off", id=ids.MODE_BADGE, color="secondary",
                  className="mode-badge"),
        _basemap_control(),
        html.Div(id=ids.NOTICE_STACK, className="notice-stack"),
        # drawer handle: collapses/expands the sidebar (callbacks/theme.py)
        html.Button(html.Span(className="gi gi-chevron"),
                    id=ids.SIDEBAR_TOGGLE, className="sidebar-handle",
                    title="Collapse / expand the sidebar",
                    **{"aria-label": "Collapse or expand the sidebar"}),
        # drag strip on the map/sidebar boundary: resizes the sidebar
        # (assets/sidebar-resize.js writes --sidebar-w; chevron still folds)
        html.Div(className="sidebar-resizer",
                 title="Drag to resize the sidebar"),
    ], style={"position": "relative"})


# ------------------------------------------------------------------- data tab
def wfs_option(svc: dict) -> dict:
    """A WFS-preset dropdown option: '[state] name' -> 'url|layer'."""
    from .presets import STATE_NAME
    state = svc.get("state", "DE")
    tag = state if state == "DE" else STATE_NAME.get(state, state)
    return {"label": f"[{tag}] {svc['name']}",
            "value": f'{svc["url"]}|{svc.get("layer", "")}'}


def overlay_option(svc: dict) -> dict:
    """An overlay-preset option (WMS/WCS): label -> 'service|url|layer'."""
    return {"label": f"{svc['name']}",
            "value": f'{svc["service"]}|{svc["url"]}|{svc.get("layer", "")}'}


def _data_tab() -> html.Div:
    from .presets import MAP_SERVICES, services_in_view
    wfs_services = [s for s in MAP_SERVICES if s["service"] == "wfs"]
    overlay_services = [s for s in MAP_SERVICES if s["service"] in ("wms",
                                                                    "wcs")]
    categories = sorted({s["category"] for s in wfs_services})
    presets_dd = [{"label": "Custom…", "value": ""}]
    presets_dd += [wfs_option(s) for s in services_in_view(None, wfs_services)]
    category_dd = [{"label": "All categories", "value": ""}] + \
        [{"label": c, "value": c} for c in categories]
    overlay_dd = [overlay_option(s) for s in overlay_services]
    return html.Div([
        _card("Study area", [
            html.Div(id=ids.STUDY_AREA_INFO, className="small text-info",
                     children="No study area drawn."),
            html.Div([
                dbc.Button("Clear area", id=ids.STUDY_AREA_CLEAR, size="sm",
                           color="secondary"),
            ]),
            dbc.InputGroup([
                dbc.InputGroupText("Project CRS", className="small"),
                dbc.Input(id=ids.PROJECT_CRS, value="EPSG:25832", size="sm",
                          debounce=True),
            ], size="sm"),
        ], intro="Draw a rectangle or polygon with the map's draw tool; "
                 "loaded data is clipped to it."),
        _card("Add vector data", [
            dbc.RadioItems(
                id=ids.DATA_SOURCE_TYPE, inline=True, value="local",
                options=[{"label": " Local file", "value": "local"},
                         {"label": " WFS server", "value": "wfs"}],
                className="small"),
            dbc.Checkbox(id=ids.CLIP_TO_AREA, value=True,
                         label="Clip to study area", className="small"),
            # only the panel matching the selected source shows
            # (data.toggle_source_panels); local is the default
            html.Div([
                dbc.InputGroup([
                    dbc.Input(id=ids.LOCAL_PATH, placeholder="Path to .shp / "
                              ".geojson / .gpkg / .gml / .kml (or .tif "
                              "raster)", size="sm"),
                    _browse(ids.LOCAL_PATH),
                ], size="sm", className="mb-1"),
                dbc.Input(id=ids.LOCAL_LAYER, placeholder="Sub-layer "
                          "(GPKG, optional)", size="sm", className="mb-1"),
                html.Div([
                    dbc.Button("Load file", id=ids.LOCAL_LOAD_BTN, size="sm",
                               color="primary"),
                    dcc.Loading(html.Span(id=ids.LOCAL_LOAD_STATUS),
                                type="dot", parent_className="load-inline"),
                ], className="d-flex align-items-center gap-2"),
            ], id=ids.DATA_LOCAL_PANEL),
            html.Div([
                html.Small("Free ALKIS land-use, terrain (DEM), protection "
                           "and topographic services for all of Germany. "
                           "Filter by category and — with 'only in view' on "
                           "— only the servers covering the current map area "
                           "are listed.", className="text-muted"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id=ids.WFS_CATEGORY,
                                         options=category_dd, value="",
                                         clearable=False,
                                         className="small"), width=7),
                    dbc.Col(dbc.Checkbox(id=ids.WFS_IN_VIEW_ONLY, value=True,
                                         label="only in view",
                                         className="small"), width=5),
                ], className="g-1 mb-1"),
                dcc.Dropdown(id=ids.WFS_PRESET, options=presets_dd, value="",
                             placeholder="Pick a dataset / server…",
                             className="small mb-1"),
                dbc.Input(id=ids.WFS_URL, placeholder="WFS URL (https://…)",
                          size="sm", className="mb-1"),
                dbc.InputGroup([
                    dbc.Input(id=ids.WFS_LAYER,
                              placeholder="Layer / feature type", size="sm"),
                    dbc.Button("List layers", id=ids.WFS_CAPS_BTN, size="sm",
                               color="secondary", outline=True,
                               title="Query GetCapabilities for this "
                                     "server's feature types"),
                ], size="sm", className="mb-1"),
                dcc.Dropdown(id=ids.WFS_LAYER_SELECT, options=[], value=None,
                             placeholder="…or pick a discovered feature type",
                             className="small mb-1"),
                html.Div([
                    dbc.Button("Load from WFS", id=ids.WFS_LOAD_BTN,
                               size="sm", color="primary"),
                    dcc.Loading(html.Span(id=ids.WFS_LOAD_STATUS),
                                type="dot", parent_className="load-inline"),
                ], className="d-flex align-items-center gap-2"),
            ], id=ids.DATA_WFS_PANEL, style={"display": "none"}),
        ]),
        _collapsible_card("Overlays & terrain (WMS / DEM)", [
            dcc.Dropdown(id=ids.OVERLAY_PRESET, options=overlay_dd,
                         value=None,
                         placeholder="Pick a WMS overlay or the DGM DEM…",
                         className="small"),
            dbc.ButtonGroup([
                dbc.Button("Add to map", id=ids.OVERLAY_LOAD_BTN, size="sm",
                           color="primary", outline=True),
                dbc.Button("Load DEM for area", id=ids.DEM_LOAD_BTN,
                           size="sm", color="primary", outline=True),
            ]),
        ], intro="Nationwide BKG basemaps and the DGM terrain model. "
                 "'Add to map' does the right thing: a WMS becomes a live "
                 "overlay (clipped to the study area); the DGM DEM downloads "
                 "as an elevation raster for the study area (usable for "
                 "slope-aware / overhead-line routing)."),
        _osm_section(),
        _card("Datasets", [
            dcc.Loading(html.Div(id=ids.DATASET_LIST, className="small"),
                        type="dot"),
        ]),
        _collapsible_card("Merge vector layers", [
            dcc.Dropdown(id=ids.MERGE_SELECT, options=[], multi=True,
                         placeholder="Pick two or more vector layers…",
                         className="small"),
            dbc.InputGroup([
                dbc.InputGroupText("Name", className="small"),
                dbc.Input(id=ids.MERGE_NAME, size="sm",
                          placeholder="Merged land use"),
            ], size="sm"),
            html.Div(dbc.Button("Merge layers", id=ids.MERGE_BTN, size="sm",
                                color="primary", outline=True)),
        ], intro="Combine several loaded vector layers (e.g. the ALKIS "
                 "land-use layers of neighbouring states in a cross-state "
                 "project) into ONE new layer: rows are stacked, attribute "
                 "columns aligned by name, everything reprojected to the "
                 "project CRS. The merged layer then takes ONE cost table "
                 "and ONE rasterization step."),
        _collapsible_card("Project", [
            dbc.InputGroup([
                dbc.Input(id=ids.PROJECT_SAVE_PATH, size="sm",
                          placeholder="Save project to folder…"),
                _browse(ids.PROJECT_SAVE_PATH),
            ], size="sm"),
            dbc.Label("Include in save", className="small"),
            dbc.Checklist(
                id=ids.SAVE_INCLUDE, inline=True, className="small",
                value=["raster", "vector", "cost_table"],
                options=[
                    {"label": " raster files", "value": "raster"},
                    {"label": " vector layers", "value": "vector"},
                    {"label": " cost tables & modifiers",
                     "value": "cost_table"}]),
            html.Small("Routes and the study area are ALWAYS saved with "
                       "the project.", className="text-muted"),
            dbc.ButtonGroup([
                dbc.Button("Save project", id=ids.PROJECT_SAVE_BTN, size="sm",
                           color="primary"),
                dbc.Button("New project", id=ids.PROJECT_NEW_BTN, size="sm",
                           color="danger", outline=True),
            ]),
            dbc.InputGroup([
                dbc.Input(id=ids.PROJECT_OPEN_PATH, size="sm",
                          placeholder="Open project… (project.json)"),
                _browse(ids.PROJECT_OPEN_PATH),
            ], size="sm"),
            html.Div(dbc.Button("Open project", id=ids.PROJECT_OPEN_BTN,
                                size="sm", color="primary", outline=True)),
            html.Div(id=ids.PROJECT_STATUS, className="small text-info"),
        ]),
    ], className="gui-tab-body")


def _osm_section() -> html.Details:
    """OpenStreetMap features via Overpass — an editable vector / cost source.

    Guided, no-OSM-knowledge-needed menu: pick a feature column (tag key),
    then the values present in that column, '+ Add' the combination — repeat
    for as many column/value combinations as needed, then load them all at
    once. Presets and a raw tag field remain as shortcuts underneath.
    """
    from .services.osm import OSM_FEATURE_PRESETS, OSM_KEY_VALUES
    options = [{"label": p["name"], "value": p["name"]}
               for p in OSM_FEATURE_PRESETS]
    key_options = [{"label": k, "value": k} for k in OSM_KEY_VALUES]
    return _collapsible_card("OpenStreetMap features", [
        dbc.Label("1. Feature column (OSM tag key)", className="small"),
        dcc.Dropdown(id=ids.OSM_KEY, options=key_options, value=None,
                     placeholder="e.g. landuse, highway, power…",
                     className="small"),
        dbc.Label("2. Values in that column (empty = any)",
                  className="small"),
        dcc.Dropdown(id=ids.OSM_VALUES, options=[], value=[], multi=True,
                     placeholder="pick one or more values…",
                     className="small"),
        html.Div(dbc.Button("+ Add combination", id=ids.OSM_ADD_BTN,
                            size="sm", color="secondary", outline=True)),
        html.Div(id=ids.OSM_SELECTION_LIST, className="small my-1"),
        html.Div(dbc.Button("Load OSM features", id=ids.OSM_LOAD_BTN,
                            size="sm", color="primary", outline=True)),
        dbc.Accordion([dbc.AccordionItem([
            dcc.Dropdown(id=ids.OSM_PRESET, options=options, value=None,
                         placeholder="Pick an OSM feature preset…",
                         className="small mb-1"),
            dbc.InputGroup([
                dbc.InputGroupText("or tag", className="small"),
                dbc.Input(id=ids.OSM_TAGS, size="sm",
                          placeholder="e.g. landuse=forest, highway"),
            ], size="sm"),
            html.Small("Used only while the list above is empty.",
                       className="text-muted"),
        ], title="Presets & raw tags (advanced)")], start_collapsed=True),
    ], intro="Pull OSM vector features (land use, roads, water, power "
             "lines, buildings…) for the study area into an editable "
             "layer — usable as a cost-model base or routing input. "
             "Pick a feature column, then its values, and '+ Add' — "
             "several combinations load together. Draw a study area "
             "first (queries are bounded to it).")


# ------------------------------------------------------------------- cost tab
def _cost_tab() -> html.Div:
    return html.Div([
        _card("Cost model", [
            dbc.Label("Base dataset", className="small"),
            dcc.Dropdown(id=ids.COST_DATASET, options=[], placeholder="Pick a "
                         "loaded vector layer…", className="small"),
            dbc.Label("Feature column(s) — ordered; first = main",
                      className="small"),
            dcc.Dropdown(id=ids.COST_FEATURE_KEYS, options=[], multi=True,
                         placeholder="e.g. nutzart, bez", className="small"),
            html.Div(id=ids.COST_FEATURE_INFO),
            html.Div(dbc.Button("Seed cost table", id=ids.COST_SEED_BTN,
                                size="sm", color="primary",
                                title="Enabled once a base dataset and "
                                      "feature column(s) are picked")),
            dag.AgGrid(
                id=ids.COST_GRID, columnDefs=[], rowData=[],
                defaultColDef={"editable": True, "resizable": True,
                               "sortable": True},
                dashGridOptions={"rowSelection": "multiple",
                                 "stopEditingWhenCellsLoseFocus": True},
                columnSize="sizeToFit", style=GRID_STYLE),
            dbc.ButtonGroup([
                dbc.Button("+ Row", id=ids.COST_ADD_ROW_BTN, size="sm",
                           color="secondary"),
                dbc.Button("− Selected", id=ids.COST_DEL_ROW_BTN, size="sm",
                           color="secondary"),
            ]),
            html.Div(id=ids.COST_COVERAGE_INFO,
                     className="small text-warning"),
            dbc.InputGroup([
                dbc.Input(id=ids.COST_IMPORT_PATH, size="sm",
                          placeholder="Import cost table (.csv/.json/.xlsx)"),
                _browse(ids.COST_IMPORT_PATH),
                dbc.Button("Import", id=ids.COST_IMPORT_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
            dbc.InputGroup([
                dbc.Input(id=ids.COST_EXPORT_PATH, size="sm",
                          placeholder="Export cost table to…"),
                _browse(ids.COST_EXPORT_PATH),
                dbc.Button("Export", id=ids.COST_EXPORT_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
        ]),
        _collapsible_card("Modifier conditions (applied in order — F12)", [
        dag.AgGrid(
            id=ids.MODIFIER_GRID, rowData=[],
            columnDefs=[
                {"field": "dataset", "headerName": "Dataset",
                 "editable": True,
                 "cellEditor": "agSelectCellEditor",
                 "cellEditorParams": {"values": []}},
                {"field": "column", "headerName": "Column", "editable": True,
                 "cellEditor": "agSelectCellEditor",
                 "cellEditorParams": {"values": []}, "maxWidth": 140},
                {"field": "operator", "headerName": "Op", "editable": True,
                 "cellEditor": "agSelectCellEditor",
                 "cellEditorParams": {"values": [
                     "all", "==", "!=", "<", "<=", ">", ">=", "in",
                     "is-empty"]}, "maxWidth": 90},
                {"field": "value", "headerName": "Value", "editable": True,
                 "maxWidth": 120},
                {"field": "mode", "headerName": "Mode", "editable": True,
                 "cellEditor": "agSelectCellEditor",
                 "cellEditorParams": {"values": ["multiply", "override",
                                                 "per-feature"]},
                 "maxWidth": 105},
                {"field": "factor", "headerName": "Factor/cost",
                 "editable": True, "maxWidth": 110},
                {"field": "buffer_m", "headerName": "Buf m",
                 "editable": True, "maxWidth": 85},
            ],
            defaultColDef={"resizable": True},
            dashGridOptions={"rowSelection": "single",
                             "stopEditingWhenCellsLoseFocus": True},
            columnSize="sizeToFit", style=SMALL_GRID_STYLE),
        dbc.ButtonGroup([
            dbc.Button("+ Modifier", id=ids.MODIFIER_ADD_BTN, size="sm",
                       color="secondary"),
            dbc.Button("− Selected", id=ids.MODIFIER_DEL_BTN, size="sm",
                       color="secondary"),
        ]),
        ], intro="Each row is one condition: pick a Dataset, a feature "
                 "Column, an Operator (== ≠ < ≤ > ≥ in is-empty, or 'all' "
                 "for the whole dataset) and a Value; then Mode "
                 "(Multiply = factor, Override = replace, Per-feature = use "
                 "each polygon's own value from the Column) and Factor/cost "
                 "(e.g. 1.25, or 65535 to forbid). Example: water zones — "
                 "one row each: ZONE == 1 → 100, ZONE == 2 → 2, …. For a "
                 "drawn cost layer: Column = cost, Mode = per-feature."),
        _collapsible_card("Preprocessing", [
            dcc.Dropdown(
                id=ids.PREPROC_SELECT, className="small", value="none",
                options=[{"label": "None", "value": "none"},
                         {"label": "Street type + buffer (A/B/L)",
                          "value": "street_buffer"}]),
            dbc.Row([
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText("A", className="small"),
                    dbc.Input(id=ids.PREPROC_BUF_A, type="number", value=10,
                              size="sm")], size="sm"), width=4),
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText("B", className="small"),
                    dbc.Input(id=ids.PREPROC_BUF_B, type="number", value=4,
                              size="sm")], size="sm"), width=4),
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText("L", className="small"),
                    dbc.Input(id=ids.PREPROC_BUF_L, type="number", value=2,
                              size="sm")], size="sm"), width=4),
            ], className="g-1"),
            html.Div("Custom preprocessing steps", className="gui-subhead"),
            html.Small("Ordered edits to the base dataset before rasterizing. "
                       "Each row selects rows by a condition (or a condition "
                       "GROUP built below) then acts: buffer = grow geometry "
                       "by Arg metres; set = write Arg into the Target "
                       "column; keep/drop = filter. Example: buffer all "
                       "features where nutzart == Straßenverkehr by 8 m.",
                       className="gui-card-intro"),
            _preproc_builder(),
            dag.AgGrid(
                id=ids.PREPROC_GRID, rowData=[],
                columnDefs=[
                    {"field": "condition", "headerName": "Condition → action",
                     "editable": False, "minWidth": 180, "flex": 1},
                    {"field": "dataset", "headerName": "Dataset",
                     "editable": True,
                     "cellEditor": "agSelectCellEditor",
                     "cellEditorParams": {"values": []}, "maxWidth": 130},
                    {"field": "op", "headerName": "Op", "editable": True,
                     "cellEditor": "agSelectCellEditor",
                     "cellEditorParams": {"values": ["buffer", "set", "keep",
                                                     "drop"]}, "maxWidth": 90},
                    {"field": "column", "headerName": "Column",
                     "editable": True,
                     "cellEditor": "agSelectCellEditor",
                     "cellEditorParams": {"values": []}, "maxWidth": 130},
                    {"field": "operator", "headerName": "Op2",
                     "editable": True,
                     "cellEditor": "agSelectCellEditor",
                     "cellEditorParams": {"values": [
                         "all", "==", "!=", "<", "<=", ">", ">=", "in",
                         "is-empty"]}, "maxWidth": 85},
                    {"field": "value", "headerName": "Value", "editable": True,
                     "maxWidth": 110},
                    {"field": "target", "headerName": "Target col",
                     "editable": True, "maxWidth": 110},
                    {"field": "arg", "headerName": "Arg (m / new)",
                     "editable": True, "maxWidth": 110},
                ],
                defaultColDef={"resizable": True},
                dashGridOptions={"rowSelection": "single",
                                 "stopEditingWhenCellsLoseFocus": True},
                columnSize="sizeToFit", style=SMALL_GRID_STYLE),
            dbc.ButtonGroup([
                dbc.Button("+ Step", id=ids.PREPROC_ADD_BTN, size="sm",
                           color="secondary"),
                dbc.Button("− Selected", id=ids.PREPROC_DEL_BTN, size="sm",
                           color="secondary"),
            ]),
        ]),
        _manual_cost_section(),
    ], className="gui-tab-body")


def _preproc_builder() -> html.Div:
    """The condition-group step builder (round 11).

    '+ Add condition' collects (column, operator, value) triples; the group
    operator (& / |) combines them; the live preview shows the python-like
    mask, e.g. (("nutzart" == "Wald") & ("bez" == "Nadelholz")) -> buffer=2m.
    'Add step' appends the finished step to the steps grid below. Column and
    value are searchable dropdowns fed from the picked dataset's actual data.
    """
    return html.Div([
        dbc.Label("Build a step from conditions", className="small fw-bold"),
        dcc.Dropdown(id=ids.PPB_DATASET, options=[], value=None,
                     placeholder="Dataset the step applies to…",
                     className="small mb-1"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id=ids.PPB_COLUMN, options=[], value=None,
                                 placeholder="Column…", className="small"),
                    width=5),
            dbc.Col(dbc.Select(
                id=ids.PPB_OPERATOR, size="sm", value="==",
                options=[{"label": o, "value": o} for o in
                         ("==", "!=", "<", "<=", ">", ">=", "in",
                          "is-empty")]), width=3),
            dbc.Col(dcc.Dropdown(id=ids.PPB_VALUE, options=[], value=None,
                                 placeholder="Value…", className="small"),
                    width=4),
        ], className="g-1"),
        html.Div(dbc.Button("+ Add condition", id=ids.PPB_ADD_COND_BTN,
                            size="sm", color="secondary", outline=True,
                            className="my-1")),
        html.Div(id=ids.PPB_COND_LIST, className="small"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Combine", className="small"),
                dbc.Select(id=ids.PPB_COMBINE, size="sm", value="&",
                           options=[
                               {"label": "& (all must match)", "value": "&"},
                               {"label": "| (any matches)", "value": "|"}]),
            ], width=6),
            dbc.Col([
                dbc.Label("Action", className="small"),
                dbc.Select(id=ids.PPB_OP, size="sm", value="buffer",
                           options=[{"label": o, "value": o} for o in
                                    ("buffer", "set", "keep", "drop")]),
            ], width=6),
        ], className="g-1"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Target", className="small"),
                dbc.Input(id=ids.PPB_TARGET, size="sm",
                          placeholder="column (set)"),
            ], size="sm"), width=6),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Arg", className="small"),
                dbc.Input(id=ids.PPB_ARG, size="sm", value="2",
                          placeholder="m / value"),
            ], size="sm"), width=6),
        ], className="g-1 mt-1"),
        html.Div(id=ids.PPB_PREVIEW, className="small text-info my-1",
                 style={"fontFamily": "monospace"}),
        html.Div(dbc.Button("Add step ↓", id=ids.PPB_ADD_STEP_BTN, size="sm",
                            color="primary", outline=True)),
        html.Hr(className="my-2"),
    ])


def _manual_cost_section() -> html.Details:
    """Draw + edit custom cost polygons, per-polygon cost, in the Cost tab (33).

    No draw-target radio anymore: shapes drawn while the COST tab is open
    become cost polygons; drawn anywhere else they set the study area
    (data.set_draw_target keys the shared draw tool off the active tab).
    """
    return _collapsible_card("Custom cost polygons", [
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Name", className="small"),
                dbc.Input(id=ids.MANUAL_NAME, size="sm", value="Manual costs")],
                size="sm"), width=7),
            dbc.Col(dbc.Select(
                id=ids.MANUAL_MODE, size="sm", value="override",
                options=[{"label": "override", "value": "override"},
                         {"label": "base", "value": "base"}]), width=5),
        ], className="g-1"),
        dbc.InputGroup([
            dbc.InputGroupText("Default cost", className="small"),
            dbc.Input(id=ids.MANUAL_COST, type="number", value=65535,
                      size="sm"),
        ], size="sm"),
        dag.AgGrid(
            id=ids.MANUAL_GRID, rowData=[],
            columnDefs=[
                {"field": "__row", "headerName": "#", "maxWidth": 50,
                 "editable": False},
                {"field": "name", "headerName": "name", "editable": True,
                 "flex": 2},
                {"field": "cost", "headerName": "cost", "editable": True,
                 "type": "numericColumn", "maxWidth": 110},
            ],
            getRowId="params.data.__row",
            defaultColDef={"resizable": True},
            dashGridOptions={"rowSelection": "multiple",
                             "stopEditingWhenCellsLoseFocus": True},
            columnSize="sizeToFit", style=SMALL_GRID_STYLE),
        dbc.ButtonGroup([
            dbc.Button("Snapshot drawn shapes", id=ids.MANUAL_CREATE_BTN,
                       size="sm", color="secondary", outline=True,
                       title="Also build the layer from the currently drawn "
                             "shapes"),
            dbc.Button("− Remove selected", id=ids.MANUAL_DEL_BTN, size="sm",
                       color="secondary", outline=True,
                       title="Drop the selected polygon(s) from the cost "
                             "layer (move/add vertices with the map's edit "
                             "tool)"),
        ]),
        html.Div(id=ids.MANUAL_STATUS, className="small text-info"),
    ], intro="While the Cost tab is open, rectangles / free-form polygons "
             "drawn with the map's ✏️ tool become cost polygons (on any "
             "other tab they set the study area). Move vertices with the "
             "edit tool and delete with the 🗑 tool — the layer updates "
             "live. Name each polygon and set its cost in the table. Use "
             "it as a base dataset, or a per-feature override modifier "
             "(Column = cost, Mode = per-feature).")


# ----------------------------------------------------------------- raster tab
def _raster_tab() -> html.Div:
    return html.Div([
        _card("Rasterization", [
            dbc.Label("Dataset", className="small"),
            dcc.Dropdown(id=ids.RASTERIZE_DATASET, options=[], value="",
                         clearable=False, className="small"),
            dbc.Label("Cost table", className="small"),
            dcc.Dropdown(id=ids.RASTERIZE_TABLE, options=[], value="current",
                         clearable=False, className="small"),
            html.Small("Only cost tables whose feature columns exist in the "
                       "picked dataset are offered — a table can never be "
                       "applied to a dataset it doesn't belong to.",
                       className="text-muted"),
            dbc.InputGroup([
                dbc.InputGroupText("Resolution m", className="small"),
                dbc.Input(id=ids.RES_M, type="number", value=1.0, min=0.1,
                          step=0.1, size="sm"),
            ], size="sm"),
            html.Div(id=ids.RASTER_SIZE_INFO, className="small text-info",
                     children="-"),
            dbc.Accordion([dbc.AccordionItem([
                dbc.InputGroup([
                    dbc.InputGroupText("fill_value", className="small"),
                    dbc.Input(id=ids.FILL_VALUE, type="number", value=65535,
                              size="sm"),
                ], size="sm", className="mb-1"),
                html.Small("Unmapped categories become this value — 65535 = "
                           "forbidden (F11).", className="text-muted"),
                dbc.InputGroup([
                    dbc.InputGroupText("dtype", className="small"),
                    dbc.Select(id=ids.RASTER_DTYPE, value="uint16", options=[
                        {"label": v, "value": v}
                        for v in ("uint16", "uint32", "float32")]),
                ], size="sm", className="mb-1"),
                dbc.InputGroup([
                    dbc.InputGroupText("Geometry buffer m",
                                       className="small"),
                    # dbc.Input does NOT accept `title` (dbc 2.x drops it from
                    # the allowed props; Button and Offcanvas still take one,
                    # which is why this was the only site that broke). A
                    # dbc.Tooltip on the same target is the supported way to
                    # get hover text, and it renders on focus too.
                    dbc.Input(id=ids.GEOM_BUFFER, type="number", value=1,
                              size="sm"),
                    dbc.Tooltip(
                        "Outward buffer on every feature before burn (m). "
                        "1 m at 1 m cells clears most ALKIS sub-cell parcel "
                        "defects.",
                        target=ids.GEOM_BUFFER),
                ], size="sm", className="mb-1"),
            ], title="Advanced")], start_collapsed=True),
            dbc.InputGroup([
                dbc.Input(id=ids.RASTER_SAVE_PATH, size="sm",
                          placeholder="Save raster as… (blank = work dir, "
                                      "F2)"),
                _browse(ids.RASTER_SAVE_PATH),
            ], size="sm"),
            dbc.Button("Build cost raster", id=ids.RASTERIZE_BTN, size="sm",
                       color="primary", className="gui-cta",
                       title="Enabled once a cost table is seeded in the "
                             "Cost tab"),
            dcc.Loading(html.Pre(id=ids.RASTERIZE_LOG,
                                 className="small bg-light p-1",
                                 style={"maxHeight": "140px",
                                        "overflowY": "auto"}),
                        type="dot"),
        ]),
        _card("Load existing raster", [
            dbc.InputGroup([
                dbc.Input(id=ids.RASTER_LOAD_PATH, size="sm",
                          placeholder="Path to .tif"),
                _browse(ids.RASTER_LOAD_PATH),
                dbc.Button("Load", id=ids.RASTER_LOAD_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
        ]),
        _collapsible_card("Display & legend", [
            dbc.Label("Raster overlay opacity", className="small"),
            dcc.Slider(id=ids.RASTER_OPACITY, min=0, max=1, step=0.05,
                       value=0.7, marks={0: "0", 0.5: "0.5", 1: "1"}),
            dbc.Label("Colormap", className="small"),
            dbc.Select(id=ids.RASTER_COLORMAP, value="viridis",
                       options=[{"label": c, "value": c} for c in COLORMAPS],
                       size="sm"),
            html.Small("Applies to every raster layer; re-colours instantly "
                       "(F3). Select a built cost raster in the Layers tab "
                       "for its cost ↔ colour ↔ combination legend below.",
                       className="text-muted"),
            # QGIS-style graduated classification of the SELECTED raster
            html.Div("Graduated rendering", className="gui-subhead"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Render", className="small"),
                    dbc.Select(id=ids.GRAD_MODE, size="sm",
                               value="continuous", options=[
                                   {"label": "continuous (default)",
                                    "value": "continuous"},
                                   {"label": "equal interval",
                                    "value": "equal"},
                                   {"label": "quantile (equal count)",
                                    "value": "quantile"},
                                   {"label": "natural breaks (Jenks)",
                                    "value": "jenks"},
                                   {"label": "logarithmic", "value": "log"},
                                   {"label": "unique values",
                                    "value": "unique"}]),
                ], width=7),
                dbc.Col([
                    dbc.Label("Classes", className="small"),
                    dbc.Input(id=ids.GRAD_CLASSES, type="number", value=5,
                              min=2, max=32, step=1, size="sm"),
                ], width=5),
            ], className="g-1"),
            html.Div(dbc.Button("Classify", id=ids.GRAD_APPLY_BTN, size="sm",
                                color="primary", outline=True,
                                title="Classify the raster selected in the "
                                      "Layers tab (or the only raster) with "
                                      "the chosen method")),
            html.Div(id=ids.GRAD_STATUS, className="small text-info"),
            html.Small("Pick a classification and press Classify — the "
                       "legend below gains one editable colour per class "
                       "('continuous' restores the smooth ramp).",
                       className="text-muted"),
            html.Div(id=ids.COLORMAP_LEGEND,
                     style={"maxHeight": "260px", "overflowY": "auto"}),
        ]),
        _raster_algebra_section(),
    ], className="gui-tab-body")


def _raster_algebra_section() -> html.Details:
    """Combine several cost rasters into one (Feature 6)."""
    return _collapsible_card("Combine cost rasters", [
        dcc.Dropdown(id=ids.RASTER_COMBINE_SELECT, options=[], multi=True,
                     placeholder="Select raster layers to combine…",
                     className="small"),
        dbc.InputGroup([
            dbc.InputGroupText("Operation", className="small"),
            dbc.Select(id=ids.RASTER_COMBINE_OP, value="add", options=[
                {"label": "add (sum costs)", "value": "add"},
                {"label": "multiply", "value": "multiply"},
                {"label": "min (cheapest wins)", "value": "min"},
                {"label": "max (most expensive wins)", "value": "max"},
                {"label": "overlay (top valid over below)", "value": "overlay"},
                {"label": "merge / mosaic (fill gaps)", "value": "merge"},
            ], size="sm"),
        ], size="sm"),
        html.Div(dbc.Button("Combine rasters", id=ids.RASTER_COMBINE_BTN,
                            size="sm", color="primary", outline=True,
                            title="Enabled once two or more raster layers "
                                  "are loaded")),
        dcc.Loading(html.Div(id=ids.RASTER_COMBINE_STATUS,
                             className="small text-info"), type="dot"),
    ], intro="Pick two or more raster layers and an operation. Inputs "
             "with a different grid/CRS are resampled onto the FIRST "
             "selected raster's grid; forbidden (65535) cells are kept "
             "forbidden.")


# ----------------------------------------------------------------- routes tab
def _routes_tab() -> html.Div:
    return html.Div([
        _card("New routing", [
            dbc.Label("Route on raster", className="small"),
            dcc.Dropdown(id=ids.ROUTE_RASTER, options=[], className="small",
                         placeholder="Pick a raster layer…"),
            dbc.Label("Map click…", className="small"),
            dbc.RadioItems(
                id=ids.BUILD_MODE, inline=True, value="off",
                options=[{"label": " Off", "value": "off"},
                         {"label": " Source", "value": "source"},
                         {"label": " Target", "value": "target"},
                         {"label": " +Waypoint", "value": "waypoint"},
                         {"label": " Move waypoint", "value": "move-waypoint"}],
                className="small"),
            dbc.Switch(id=ids.EDIT_ENABLE, value=False, className="small",
                       label="Edit the selected route (move/add source, "
                             "target & waypoints update it in place)"),
            html.Small("Edit off → clicks build a NEW route. Edit on → "
                       "clicks update the route selected below (Source/"
                       "Target move it; +Waypoint adds one; Move waypoint "
                       "relocates the nearest waypoint). Editing replaces "
                       "the route.", className="text-muted"),
            dbc.Label("Control points — ONE list for the new route or the "
                      "selected route", className="small"),
            html.Small("Rows are editable (typed coordinates in the project "
                       "CRS) and deletable; order = routing order. With a "
                       "route selected, 'Apply points' recomputes it as a "
                       "new variant.", className="text-muted"),
            dag.AgGrid(
                id=ids.BUILD_POINTS_TABLE, rowData=[],
                columnDefs=[
                    {"field": "kind", "maxWidth": 100, "editable": True,
                     "cellEditor": "agSelectCellEditor",
                     "cellEditorParams": {"values": ["source", "waypoint",
                                                     "target"]}},
                    {"field": "x", "editable": True, "valueFormatter":
                        {"function": "d3.format(',.1f')(params.value)"}},
                    {"field": "y", "editable": True, "valueFormatter":
                        {"function": "d3.format(',.1f')(params.value)"}},
                    {"field": "name", "headerName": "name (waypoint)",
                     "editable": True, "maxWidth": 130},
                    # euclidean metres from the previous control point
                    # (source→waypoint→…→target legs)
                    {"field": "dist", "headerName": "dist m",
                     "editable": False, "maxWidth": 95, "valueFormatter":
                        {"function": "params.value == null ? '' : "
                                     "d3.format(',.0f')(params.value)"}},
                ],
                defaultColDef={"resizable": True},
                dashGridOptions={"rowSelection": "multiple",
                                 "stopEditingWhenCellsLoseFocus": True},
                # responsive: this grid initializes inside a hidden tab (zero
                # width) and the sidebar is drag-resizable — a one-shot
                # sizeToFit would leave columns overflowed/virtualized out
                columnSize="responsiveSizeToFit", style=SMALL_GRID_STYLE),
            dbc.ButtonGroup([
                dbc.Button("− Remove selected", id=ids.POINTS_REMOVE_BTN,
                           size="sm", color="secondary"),
                dbc.Button("Apply points", id=ids.POINTS_APPLY_BTN, size="sm",
                           color="primary", outline=True),
                dbc.Button("Clear all", id=ids.BUILD_CLEAR_BTN, size="sm",
                           color="secondary", outline=True),
            ]),
            # the main CTA lives here (always visible) — the collapsible
            # sections below only tune HOW the routing runs
            html.Div([
                dbc.Button("Run routing", id=ids.RUN_ROUTING_BTN, size="sm",
                           color="primary", className="gui-cta",
                           title="Enabled once a cost raster exists"),
                dbc.Button("⏹ Stop", id=ids.ROUTING_STOP_BTN, size="sm",
                           color="danger", style={"display": "none"},
                           title="Interrupt the running routing job — it "
                                 "stops after the segment in flight; "
                                 "completed routes are kept"),
            ], className="d-flex align-items-center gap-2"),
            dcc.Loading(html.Div(id=ids.ROUTING_STATUS,
                                 className="small text-info"),
                        type="circle"),
        ]),
        _collapsible_card("Algorithm & search", [
        dbc.Row([
            dbc.Col([
                dbc.Label("Algorithm", className="small"),
                dcc.Dropdown(id=ids.ALGORITHM, clearable=False,
                             value="delta-stepping", className="small",
                             options=[]),
            ], width=7),
            dbc.Col([
                dbc.Label("Hardware", className="small"),
                dbc.RadioItems(
                    id=ids.HARDWARE, value="cpu", className="small",
                    options=[{"label": " CPU", "value": "cpu"},
                             {"label": " GPU (fastest, opt-in)",
                              "value": "gpu"}]),
            ], width=5),
        ], className="g-1"),
        dbc.Label("Neighborhood", className="small mt-1"),
        dcc.Dropdown(id=ids.NEIGHBORHOOD, clearable=False, value="r2",
                     className="small",
                     options=[
                         {"label": "r0 — 4 directions (staircase)",
                          "value": "r0"},
                         {"label": "r1 — 8 directions", "value": "r1"},
                         {"label": "r2 — 16 directions (recommended)",
                          "value": "r2"},
                         {"label": "r3 — 32 directions (heavy)",
                          "value": "r3"}]),
        dbc.InputGroup([
            dbc.InputGroupText("Search buffer m", className="small"),
            dbc.Input(id=ids.SEARCH_BUFFER, type="number", placeholder="auto",
                      size="sm"),
        ], size="sm", className="mt-1"),
        html.Div(id=ids.SEARCH_BUFFER_INFO, className="small text-info mb-1",
                 children="blank = max(1000 m, 1.5 x distance) — never the "
                          "whole raster (F1)"),
        dbc.Checkbox(id=ids.IGNORE_MAX_COST, value=True, className="small",
                     label="Snap source/target off forbidden cells "
                           "(shift shown, F7)"),
        dbc.Checkbox(id=ids.PAIRWISE, value=False, className="small",
                     label="Pairwise (source i → target i)"),
        dbc.Row([
            dbc.Col(dbc.Checkbox(id=ids.SIMPLIFY, value=False,
                                 label="Simplify for display/export (F5)",
                                 className="small"), width=8),
            dbc.Col(dbc.Input(id=ids.SIMPLIFY_TOL, type="number", value=1.0,
                              min=0.1, size="sm"), width=4),
        ], className="g-1"),
        dbc.Accordion([dbc.AccordionItem([
            dbc.InputGroup([
                dbc.InputGroupText("delta", className="small"),
                dbc.Input(id="route-delta", type="number", value=100,
                          size="sm")], size="sm", className="mb-1"),
            dbc.InputGroup([
                dbc.InputGroupText("threads", className="small"),
                dbc.Input(id="route-num-threads", type="number", value=0,
                          size="sm")], size="sm", className="mb-1"),
            dbc.Checkbox(id="route-use-astar", value=False,
                         label="A* heuristic (off per project rule, F8)",
                         className="small"),
        ], title="Advanced")], start_collapsed=True),
        ]),
        _collapsible_card("Constrained routing", [
            dbc.Switch(id=ids.OHL_ENABLE, value=False, className="small",
                       label="Constrained routing (overhead line / towers)"),
            html.Small("When on, the extra tower/profile options appear below "
                       "and 'Run routing' does constrained routing on the "
                       "SAME source/target/waypoint points — each waypoint "
                       "forces a tower.", className="text-muted"),
            dbc.Collapse(_ohl_options(), id=ids.OHL_COLLAPSE, is_open=False),
        ]),
        _collapsible_card("Computed routes", [
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id=ids.EDIT_ROUTE_SELECT, options=[], value=None,
                placeholder="Select the active route…",
                className="small"), width=9),
            dbc.Col(dbc.Button("New", id=ids.NEW_ROUTE_BTN, size="sm",
                               color="secondary", outline=True,
                               title="Deselect — the points list goes back "
                                     "to the new-route draft"), width=3),
        ], className="g-1 my-1"),
        # editable routes list, sorted (clustered) by group (Feature 2)
        dag.AgGrid(
            id=ids.ROUTES_GRID, rowData=[],
            columnDefs=[
                {"field": "group", "editable": True, "flex": 1,
                 "sort": "asc"},
                {"field": "name", "editable": True, "flex": 2},
                {"field": "color", "headerName": "col", "editable": True,
                 "maxWidth": 95},
                {"field": "dash", "editable": True, "maxWidth": 95,
                 "cellEditor": "agSelectCellEditor",
                 "cellEditorParams": {"values": ["solid", "dashed", "dotted",
                                                 "dash-dot"]}},
                {"field": "visible", "editable": True, "maxWidth": 80,
                 "cellRenderer": "agCheckboxCellRenderer",
                 "cellEditor": "agCheckboxCellEditor"},
                {"field": "wp", "headerName": "wp", "editable": False,
                 "maxWidth": 55},
                {"field": "cost", "headerName": "cost", "editable": False,
                 "type": "numericColumn", "maxWidth": 90},
                {"field": "length", "headerName": "len m", "editable": False,
                 "type": "numericColumn", "maxWidth": 90},
                {"field": "from", "headerName": "source", "editable": False,
                 "minWidth": 120},
                {"field": "to", "headerName": "target", "editable": False,
                 "minWidth": 120},
                {"field": "id", "hide": True},
            ],
            getRowId="params.data.id",
            defaultColDef={"resizable": True, "sortable": True},
            dashGridOptions={"rowSelection": "multiple", "animateRows": True,
                             "stopEditingWhenCellsLoseFocus": True},
            columnSize="sizeToFit", style=SMALL_GRID_STYLE),
        html.Small("Edit name / group / colour (hex) / line-style / "
                   "visibility inline. Retype the group to move a route, or "
                   "select route(s) and use 'Move to group'.",
                   className="text-muted"),
        dbc.InputGroup([
            dbc.InputGroupText("Move to group", className="small"),
            dbc.Input(id=ids.ROUTE_MOVE_GROUP, size="sm",
                      placeholder="group name (blank = ungrouped)"),
            dbc.Button("Move sel.", id=ids.ROUTE_MOVE_BTN, size="sm",
                       color="secondary"),
        ], size="sm"),
        # group-level actions
        dbc.Label("Group actions", className="small"),
        dcc.Dropdown(id=ids.GROUP_SELECT, options=[], className="small",
                     placeholder="Pick a group…"),
        dbc.InputGroup([
            dbc.Input(id=ids.GROUP_RENAME, size="sm",
                      placeholder="rename group to…"),
            dbc.Button("Rename", id=ids.GROUP_RENAME_BTN, size="sm",
                       color="secondary"),
        ], size="sm"),
        dbc.Row([
            dbc.Col(dbc.Input(id=ids.GROUP_COLOR, type="color",
                              value="#e6194b", size="sm",
                              style={"height": "31px"}), width=3),
            dbc.Col(dbc.Select(id=ids.GROUP_DASH, size="sm", value="solid",
                               options=[{"label": d, "value": d} for d in
                                        ("solid", "dashed", "dotted",
                                         "dash-dot")]), width=4),
            dbc.Col(dbc.Button("Restyle", id=ids.GROUP_RESTYLE_BTN, size="sm",
                               color="secondary", outline=True), width=5),
        ], className="g-1"),
        dbc.ButtonGroup([
            dbc.Button("▲ Group up", id=ids.GROUP_UP_BTN, size="sm",
                       color="secondary", outline=True),
            dbc.Button("▼ Group down", id=ids.GROUP_DOWN_BTN, size="sm",
                       color="secondary", outline=True),
            dbc.Button("Delete group", id=ids.GROUP_DELETE_BTN, size="sm",
                       color="danger", outline=True),
        ]),
        html.Div(dbc.Button(
            "↩ Group points → New routing", id=ids.READD_POINTS_BTN,
            size="sm", color="secondary", outline=True,
            title="Copy the selected group's sources, targets and waypoints "
                  "back into the New-routing points list (they are removed "
                  "from it automatically when a run finishes)")),
        dcc.Loading(html.Div(id=ids.EDIT_COST_READOUT, className="small"),
                    type="dot"),
        html.Div(dbc.Button("Discard selected variant",
                            id=ids.DELETE_VARIANT_BTN, size="sm",
                            color="danger", outline=True)),
        ], intro="New routes appear here right after the run and are "
                 "selected automatically. Editing a route (move source/"
                 "target, add/move waypoints) REPLACES it; use 'Run routing' "
                 "or 'New' to create additional routes."),
        _collapsible_card("Edit selected route", [
            dbc.Row([
                dbc.Col(dbc.Checkbox(id=ids.AUTO_REFRESH, value=True,
                                     label="Auto-refresh on each edit",
                                     className="small"), width=7),
                dbc.Col(dbc.Button("↻ Refresh edited",
                                   id=ids.REFRESH_ROUTES_BTN,
                                   size="sm", color="primary", outline=True),
                        width=5),
            ], className="g-1"),
            html.Small("Auto-refresh off → map-click edits are STAGED (the "
                       "route dashes); press 'Refresh edited' to recompute "
                       "only the edited route(s).", className="text-muted"),
            dbc.Label("Simplify displayed/exported line (F5)",
                      className="small"),
            dbc.InputGroup([
                dbc.InputGroupText("tolerance m", className="small"),
                dbc.Input(id=ids.EDIT_SIMPLIFY_TOL, type="number", min=0,
                          step=0.5, placeholder="0 = off", size="sm"),
                dbc.Button("Apply", id=ids.EDIT_SIMPLIFY_BTN, size="sm",
                           color="primary", outline=True),
            ], size="sm"),
            html.Small("Costs always come from the full routed line; only "
                       "the drawn/exported geometry is simplified.",
                       className="text-muted"),
        ], intro="Turn on 'Edit the selected route' above, then click the "
                 "map (Source/Target/+Waypoint/Move waypoint), or edit/"
                 "delete rows in the control-points list and press "
                 "'Apply points'."),
        _card("Export / load routes", [
            dbc.InputGroup([
                dbc.Input(id=ids.EXPORT_ROUTE_PATH, size="sm",
                          placeholder="Export selected to "
                                      ".geojson/.gpkg/.shp/.csv"),
                _browse(ids.EXPORT_ROUTE_PATH),
                dbc.Button("Export", id=ids.EXPORT_ROUTE_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
            html.Div(id=ids.EXPORT_STATUS, className="small text-success"),
            dbc.InputGroup([
                dbc.Input(id=ids.ROUTES_LOAD_PATH, size="sm",
                          placeholder="Load routes (.geojson/.gpkg/.shp)"),
                _browse(ids.ROUTES_LOAD_PATH),
                dbc.Button("Load", id=ids.ROUTES_LOAD_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
        ]),
    ], className="gui-tab-body")


def _ohl_options() -> list:
    """Constrained (overhead-line) routing parameters — revealed only when the
    'Constrained routing' toggle is on (task 46). Same source/target/waypoint
    points as normal routing drive it; each waypoint forces a tower."""
    return [
        html.Small("Tower-aware routing from an infrastructure profile "
                   "(angles, spans, tower costs). Edit the profile (YAML).",
                   className="text-muted"),
        dcc.Dropdown(id=ids.OHL_PROFILE_PRESET, options=[],
                     placeholder="Load a shipped profile…",
                     className="small my-1"),
        dbc.Textarea(id=ids.OHL_PROFILE_TEXT, size="sm",
                     style={"height": "160px", "fontFamily": "monospace",
                            "fontSize": "11px"}),
        dbc.InputGroup([
            dbc.Input(id=ids.OHL_PROFILE_PATH, size="sm",
                      placeholder="Profile file (.yaml/.json)"),
            _browse(ids.OHL_PROFILE_PATH),
            dbc.Button("Load", id=ids.OHL_PROFILE_LOAD_BTN, size="sm",
                       color="primary", outline=True),
            dbc.Button("Save", id=ids.OHL_PROFILE_SAVE_BTN, size="sm",
                       color="success", outline=True),
        ], size="sm", className="my-1"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Backend", className="small"),
                dbc.Select(id=ids.OHL_BACKEND, value="cython", options=[
                    {"label": "cython (stable)", "value": "cython"},
                    {"label": "cython_parallel (experimental)",
                     "value": "cython_parallel"},
                    {"label": "raster_gpu (experimental)",
                     "value": "raster_gpu"},
                    {"label": "raster_gpu_v4 (experimental)",
                     "value": "raster_gpu_v4"},
                ], size="sm"),
            ], width=7),
            dbc.Col(dbc.Checkbox(
                id=ids.OHL_EXPERIMENTAL, value=False,
                label="allow experimental backends (F10)",
                className="small mt-4"), width=5),
        ], className="g-1"),
        dbc.Input(id=ids.OHL_DEM, size="sm", className="mt-1",
                  placeholder="DEM path (slope, optional)"),
        dbc.Input(id=ids.OHL_DSM, size="sm", className="mt-1",
                  placeholder="DSM path (clearance, optional)"),
        html.Div(id=ids.OHL_STATUS, className="small mt-1"),
        html.Small("Terrain cost readout is 2-D (F9) — compare routes by "
                   "length + tower cost too.", className="text-muted"),
    ]


# ----------------------------------------------------------------- layers tab
def _layers_tab() -> html.Div:
    # the background-map picker lives on the map itself (_basemap_control)
    return html.Div([
        _card("Layers", [
            dag.AgGrid(
                id=ids.LAYERS_GRID, rowData=[],
                columnDefs=[
                    {"field": "name", "editable": True},
                    {"field": "kind", "editable": False, "maxWidth": 100},
                    {"field": "visible", "editable": True, "maxWidth": 100,
                     "cellRenderer": "agCheckboxCellRenderer",
                     "cellEditor": "agCheckboxCellEditor"},
                ],
                getRowId="params.data.id",
                defaultColDef={"resizable": True},
                # multiple + ctrl/cmd-click: move or remove several layers at
                # once; suppressRowDeselection keeps a plain re-click from
                # toggling the selection off
                dashGridOptions={"rowSelection": "multiple",
                                 "suppressRowDeselection": True,
                                 "stopEditingWhenCellsLoseFocus": True,
                                 "animateRows": True},
                columnSize="sizeToFit",
                style={"height": "40vh", "width": "100%"}),
            dbc.ButtonGroup([
                dbc.Button("▲ Up", id="layer-up-btn", size="sm",
                           color="secondary"),
                dbc.Button("▼ Down", id="layer-down-btn", size="sm",
                           color="secondary"),
                dbc.Button("📋 Table", id=ids.LAYER_VIEW_BTN, size="sm",
                           color="secondary", outline=True,
                           title="Open the selected layer's full attribute "
                                 "table"),
                dbc.Button("Remove", id=ids.LAYER_REMOVE_BTN, size="sm",
                           color="danger", outline=True),
            ]),
            html.Small("Ctrl+click selects several layers — ▲▼ and Remove "
                       "act on all of them. Click a feature on the map to "
                       "highlight it, or open the layer's full table.",
                       className="text-muted d-block"),
        ], intro="Top row = foreground (painted last), bottom row = "
                 "background. Route groups appear as ONE row here; the "
                 "individual routes are listed in 'Routes' below."),
        _card("Routes", [
            html.Div(id=ids.ROUTE_TREE, className="small route-tree"),
        ], intro="Routes grouped by run/group — expand to see individual "
                 "routes; click one to select it for editing."),
    ], className="gui-tab-body")


# ------------------------------------------------------------------ attrs tab
def _attrs_tab() -> html.Div:
    return html.Div([
        _card("Attributes", [
            html.Div(id=ids.ATTR_PANEL, children=html.Small(
                "Click a feature on the map to inspect its attributes.",
                className="text-muted")),
        ]),
    ], className="gui-tab-body")


# --------------------------------------------------------------- workflow guide
#: ordered steps shown in the guide (title, one-line hint); a callback ticks
#: each one off as the project progresses so users always know what to do next.
WORKFLOW_STEPS = [
    ("Draw a study area", "Data tab → draw a rectangle/polygon on the map."),
    ("Load data", "Data tab → local file, WFS, OSM or DEM."),
    ("Define costs", "Cost tab → pick dataset + feature column(s) → Seed."),
    ("Build cost raster", "Raster tab → Build cost raster."),
    ("Plan routes", "Routes tab → place source/target → Run routing."),
]


def _workflow_guide() -> html.Div:
    """Placeholder — filled by the gating callback (sync_workflow_guide)."""
    return html.Div(id=ids.WORKFLOW_GUIDE)


# -------------------------------------------------------------------- sidebar
def _header() -> html.Div:
    """Brand lockup + the app-level actions (log viewer, theme toggle)."""
    return html.Div([
        html.Div([
            html.Div(html.Span(className="gi gi-pin"), className="brand-mark"),
            html.Div([
                # keep the contiguous text "PYORPS GUI" (E2E anchor)
                html.Div([html.Span("PYORPS", className="brand-strong"),
                          " GUI"], className="brand-title"),
                html.Div("Optimal route planning", className="brand-sub"),
            ]),
        ], className="brand-lockup"),
        html.Div([
            dbc.Button([html.Span(className="gi gi-log"), "Log"],
                       id=ids.LOG_VIEW_BTN, size="sm", color="link",
                       title="Open the full warning/error log"),
            dbc.Button([html.Span(className="gi gi-sun"),
                        html.Span(className="gi gi-moon")],
                       id=ids.THEME_TOGGLE, size="sm", color="link",
                       title="Switch between light and dark theme"),
        ], className="header-actions"),
    ], className="gui-header")


def _sidebar() -> html.Div:
    return html.Div([
        _header(),
        _workflow_guide(),
        dbc.Tabs([
            dbc.Tab(_data_tab(), label="Data", tab_id=ids.TAB_DATA),
            dbc.Tab(_cost_tab(), label="Cost", tab_id=ids.TAB_COST),
            dbc.Tab(_raster_tab(), label="Raster", tab_id=ids.TAB_RASTER),
            dbc.Tab(_routes_tab(), label="Routes", tab_id=ids.TAB_ROUTES),
            dbc.Tab(_layers_tab(), label="Layers", tab_id=ids.TAB_LAYERS),
            dbc.Tab(_attrs_tab(), label="Attrs", tab_id=ids.TAB_ATTRS),
        ], id=ids.TABS, active_tab=ids.TAB_DATA),
    ], className="gui-sidebar")


def _log_offcanvas() -> dbc.Offcanvas:
    """The persistent warning/error log, readable at any time."""
    return dbc.Offcanvas(
        [
            html.Div([
                dbc.Button("↻ Refresh", id=ids.LOG_REFRESH_BTN, size="sm",
                           color="secondary", outline=True, className="mb-1"),
                html.Div(id=ids.LOG_PATH_INFO,
                         className="small text-muted mb-1"),
            ]),
            html.Pre(id=ids.LOG_CONTENT, className="small bg-light p-2",
                     style={"whiteSpace": "pre-wrap", "maxHeight": "80vh",
                            "overflowY": "auto"}),
        ],
        id=ids.LOG_OFFCANVAS, is_open=False, placement="end",
        scrollable=True, title="Full log (warnings & errors)",
        style={"width": "560px"})


def build_layout(state) -> dbc.Container:
    """Assemble the full app layout (blank map — R1)."""
    # width= (not md=) so the columns NEVER stack: with a 100vh map, a
    # stacked sidebar would sit invisibly below the fold on narrow windows.
    return dbc.Container(
        _stores() + [
            dbc.Row([
                dbc.Col(_map_pane(), width=8, className="p-0 gui-map-col"),
                dbc.Col(_sidebar(), width=4, className="p-0 gui-sidebar-col"),
            ], id=ids.APP_SHELL, className="g-0 flex-nowrap gui-shell"),
            _layer_table_offcanvas(),
            _log_offcanvas(),
            _routing_confirm_modal(),
        ],
        fluid=True, className="p-0")


def _routing_confirm_modal() -> dbc.Modal:
    """Heavy-run gate: prognosed runtime/memory/storage + mitigation tips;
    the run only starts after 'Accept & run' (task: never process long/many
    routes without an up-front warning)."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("This routing run will take a while"),
                        close_button=False),
        dbc.ModalBody(id=ids.ROUTING_CONFIRM_BODY),
        dbc.ModalFooter([
            dbc.Button("Cancel", id=ids.ROUTING_CONFIRM_CANCEL, size="sm",
                       color="secondary", outline=True),
            dbc.Button("Accept & run", id=ids.ROUTING_CONFIRM_OK, size="sm",
                       color="primary"),
        ]),
    ], id=ids.ROUTING_CONFIRM_MODAL, is_open=False, backdrop="static",
        keyboard=False, centered=True)


#: fixed page size for the attribute / combination table — a known page size
#: makes the page-jump deterministic (page = row_index // TABLE_PAGE_SIZE),
#: which paginationAutoPageSize (height-derived, unknown server-side) did not.
TABLE_PAGE_SIZE = 50


def _layer_table_offcanvas() -> dbc.Offcanvas:
    """Bottom panel showing a layer's full attribute table (Excel-like).

    It spans only the MAP column (width 66%), never the sidebar, so every
    general-pane control stays reachable with the table open; a height slider
    shrinks/grows it to reveal more of the map or the sidebar below it.
    """
    return dbc.Offcanvas(
        [
            # drag strip on the panel's top edge — free (px) height resize,
            # like the sidebar width (assets/table-resize.js)
            html.Div(className="table-resizer",
                     title="Drag to resize the table"),
            html.Div([
                html.Div(id=ids.LAYER_TABLE_TITLE,
                         className="small text-muted"),
            ], className="d-flex justify-content-between align-items-center "
                         "gap-2 mb-1"),
            dag.AgGrid(
                id=ids.LAYER_TABLE_GRID, rowData=[], columnDefs=[],
                getRowId="params.data.__row",
                defaultColDef={"resizable": True, "sortable": True,
                               "filter": True, "minWidth": 90},
                dashGridOptions={"rowSelection": "single",
                                 "pagination": True,
                                 "paginationPageSize": TABLE_PAGE_SIZE},
                columnSize="autoSize",
                style={"height": "30vh", "width": "100%"}),
        ],
        id=ids.LAYER_TABLE_OFFCANVAS, is_open=False, placement="bottom",
        scrollable=True, backdrop=False, title="Layer attributes",
        # left 66% only (the map column) → the sidebar is never covered
        style={"height": "40vh", "width": "66%", "left": 0, "right": "auto"})

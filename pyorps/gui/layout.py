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

import logging
from collections.abc import Callable
from typing import Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import dcc, html

from . import ids
from .sections import corridor as corridor_section
from .sections import cost_field as cost_field_section
from .sections import objective as objective_section
from .sections import profile_form as profile_form_section
from .services import jobs as jobs_service
from .services import rasterize, routing
from .services.tiles import COLORMAPS

_LOG = logging.getLogger(__name__)

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
     "url": "https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/" +
            "{z}/{x}/{y}.png",
     "attribution": "CyclOSM | " + _OSM_ATTR, "default": False},
    {"name": "BaseMapDE Color", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/" +
            "de_basemapde_web_raster_farbe/default/GLOBAL_WEBMERCATOR/" +
            "{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "BaseMapDE Grey", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/" +
            "de_basemapde_web_raster_grau/default/GLOBAL_WEBMERCATOR/" +
            "{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "TopPlusOpen Color", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_topplus_open/tile/1.0.0/" +
            "web/default/WEBMERCATOR/{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "TopPlusOpen Grey", "group": "BKG (Germany)",
     "url": "https://sgx.geodatenzentrum.de/wmts_topplus_open/tile/1.0.0/" +
            "web_grau/default/WEBMERCATOR/{z}/{y}/{x}.png",
     "attribution": _BKG_ATTR, "default": False},
    {"name": "Esri World Imagery", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/" +
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Topo", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/" +
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Street Map", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/" +
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri World Gray Canvas", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/" +
            "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Esri NatGeo World Map", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/" +
            "NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri &mdash; National Geographic",
     "default": False},
    {"name": "Esri World Shaded Relief", "group": "Esri",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/" +
            "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles &copy; Esri", "default": False},
    {"name": "Carto Light", "group": "Carto",
     "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
     "attribution": _CARTO_ATTR, "default": False},
    {"name": "Carto Dark", "group": "Carto",
     "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
     "attribution": _CARTO_ATTR, "default": False},
    {"name": "Carto Voyager", "group": "Carto",
     "url": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/" +
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

#: class of the grids whose columns ``assets/grid-fit.js`` fits to the grid width the first time
#: the grid is shown (the grids use it instead of dash-ag-grid's ``columnSize``). ``columnSize``
#: fits at once, also in a hidden tab or a closed card where the grid has no width: AG Grid then
#: warns (#29) or squeezes the columns and does not fit again when the card opens (round 15).
FIT_COLUMNS = "gui-grid-fit"
#: the same, and again whenever the grid's width changes (was ``columnSize="responsiveSizeToFit"``)
FIT_COLUMNS_RESPONSIVE = "gui-grid-fit gui-grid-fit-responsive"


def _row_selection(multi: bool, *, keep: bool = False) -> dict:
    """The ``rowSelection`` option of a grid, in the AG Grid 32.2+ object form.

    The string forms ("single" / "multiple") and ``suppressRowDeselection`` are
    deprecated since AG Grid 32.2 and warn in the browser console. The object
    form turns a checkbox column and click selection on or off by itself, so
    both are spelled out to keep the behaviour the grids always had: a click
    selects a row, ctrl/cmd-click adds one (``multi``), no checkbox column.
    ``keep=True`` replaces ``suppressRowDeselection``: a click only selects, a
    plain re-click or ctrl-click never deselects.
    """
    options: dict = {
        "mode": "multiRow" if multi else "singleRow",
        "checkboxes": False,
        "enableClickSelection": "enableSelection" if keep else True,
    }
    if multi:
        options["headerCheckbox"] = False
    return options


def section_grid(grid_id: str, column_defs: list, *, multi: bool = False,
                 editable: bool = True, height: str = "160px",
                 options: dict | None = None, responsive: bool = False):
    """A small editable AG Grid for a section module (``pyorps.gui.sections``).

    Grid options are built in this module only, so that every grid of the app
    gets the selection options of :func:`_row_selection` (a test checks that no
    other module builds grid options). ``options`` adds further AG Grid options;
    ``rowSelection`` cannot be overridden through it. ``responsive=True`` fits
    the columns again whenever the grid's width changes (:data:`FIT_COLUMNS_RESPONSIVE`).
    """
    grid_options = {"stopEditingWhenCellsLoseFocus": True, **(options or {}),
                    "rowSelection": _row_selection(multi)}
    return dag.AgGrid(
        id=grid_id, columnDefs=column_defs, rowData=[],
        defaultColDef={"editable": editable, "resizable": True,
                       "sortable": False},
        dashGridOptions=grid_options,
        className=FIT_COLUMNS_RESPONSIVE if responsive else FIT_COLUMNS,
        style={**GRID_STYLE, "height": height})

#: a status spinner appears only after this long, so quick calls do not flicker
_SPINNER_DELAY_MS = 300

#: status line under the overlay / DEM buttons (no constant in ids.py yet)
OVERLAY_STATUS = "overlay-status"

_DEFAULT_UI_STATE = {"active_tab": ids.TAB_DATA, "click_mode": "off",
                     "active_route_id": None}
_DEFAULT_ROUTE_DRAFT = {"sources": [], "targets": [], "waypoints": []}
#: poll interval of the rasterize / constrained / cost-field / corridor jobs
_JOB_POLL_MS = 800


# --------------------------------------------------------------------- stores
def _stores() -> list:
    return [
        dcc.Store(id=ids.UI_STATE, data=dict(_DEFAULT_UI_STATE)),
        dcc.Store(id=ids.LAYERS_VIEW, data=[]),
        dcc.Store(id=ids.MAP_VIEW, data={}),
        dcc.Store(id=ids.NOTICES, data=[]),
        dcc.Store(id=ids.RASTERIZE_BUSY, data=False),
        dcc.Store(id=ids.RUN_ROUTING_BUSY, data=False),
        dcc.Store(id=ids.RUN_PLAIN_BUSY, data=False),
        dcc.Store(id=ids.RUN_OHL_BUSY, data=False),
        dcc.Store(id=ids.ROUTING_SETTINGS, data={}),
        dcc.Store(id=ids.RASTER_COMBINE_BUSY, data=False),
        dcc.Store(id=ids.ROUTE_DRAFT, data=dict(_DEFAULT_ROUTE_DRAFT)),
        dcc.Store(id=ids.EDIT_REQUEST, data={}),
        dcc.Store(id=ids.COST_GRID_STATE, data={}),
        dcc.Store(id=ids.FOCUS_FLASH, data={}),
        dcc.Store(id=ids.SELECTED_FEATURE, data={}),
        dcc.Store(id=ids.DRAW_TARGET, data="area"),
        dcc.Store(id=ids.THEME_STORE, data=None),
        dcc.Store(id=ids.DIRTY_STORE, data=False),
        dcc.Store(id=ids.OSM_SELECTIONS, data=[]),
        dcc.Store(id=ids.PPB_CONDS, data=[]),
        dcc.Store(id=ids.RASTER_SETTINGS, data={}),
        # the action waiting for the answer of the confirm modal, and one
        # counter per destructive action that fires it (callbacks/confirm.py)
        dcc.Store(id=ids.CONFIRM_PENDING, data=None),
        *[dcc.Store(id=go, data=None) for go in ids.GO_STORES],
        # round 14: mirrors of the newer cards (objective, cost field,
        # corridor graph, constrained profile form) and the last loaded profile
        dcc.Store(id=ids.OBJ_SETTINGS, data={}),
        dcc.Store(id=ids.CF_SETTINGS, data={}),
        dcc.Store(id=ids.COR_SETTINGS, data={}),
        dcc.Store(id=ids.OHL_SETTINGS, data={}),
        dcc.Store(id=ids.OHLF_BASE, data={}),
        # polls the background routing job while one is running
        dcc.Interval(id=ids.ROUTING_POLL, interval=600, disabled=True),
        # the polls of the other background jobs, off until a job runs
        *[dcc.Interval(id=poll, interval=_JOB_POLL_MS, disabled=True)
          for poll in (ids.RASTERIZE_POLL, ids.OHL_POLL, ids.CF_POLL,
                       ids.COR_POLL)],
        # one per poll: the token of the job outcome the page was handed last
        # (callbacks/job_poll.py; the next tick sends it back as the receipt)
        *[dcc.Store(id=ack, data=None) for ack in ids.JOB_ACK_STORES],
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
                      intro: str | None = None, open: bool = False,  # pylint: disable=redefined-builtin
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


def _lbl(text: str, target: str, *, className: str = "small") -> dbc.Label:
    """A visible label bound to its control: ``<label for=target>``.

    Clicking it focuses the field and a screen reader reads it with the field.
    """
    return dbc.Label(text, html_for=target, className=className)


def _addon(text: str, target: str) -> dbc.InputGroupText:
    """The caption of an input-group field (unit in brackets), bound to it."""
    return dbc.InputGroupText(
        html.Label(text, htmlFor=target, className="mb-0"), className="small")


def _group(text: str, key: str, control, *, className: str = "small") -> html.Div:
    """A control that has no ``for=`` target - a slider (its handle is not a
    labelable element), a radio list or a checklist - under its visible caption.
    The wrapper is a ``role="group"`` named by that caption."""
    label_id = f"{key}-label"
    return html.Div(
        [html.Div(text, id=label_id, className=className), control],
        role="group", **{"aria-labelledby": label_id})


def _help(children, text: str, **kwargs) -> html.Div:
    """``children`` with a hover/focus help text: dbc inputs take no ``title``,
    so the text sits on a wrapper div (same pattern as the routing settings)."""
    return html.Div(children, title=text, **kwargs)


def _browse(target_id: str, what: str = "a file or folder") -> html.Button:
    """The 📁 button opening the native file/folder picker for an input.

    A plain ``html.Button`` (dbc.Button takes no ``aria-label``): the glyph
    alone would be read as "file folder", so the button names what it picks.
    """
    return html.Button(
        "📁", id=f"browse-{target_id}", type="button",
        className="btn btn-outline-secondary btn-sm", title=f"Browse for {what}…",
        **{"aria-label": f"Browse for {what}"})


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
    options = [{"label": bm["name"], "value": bm["name"]} for bm in BASEMAPS]
    options.append({"label": "None (no basemap)", "value": ""})
    body: list = [_group(
        "Background map", ids.BASEMAP_SELECT,
        dbc.RadioItems(id=ids.BASEMAP_SELECT, options=options,
                       value=DEFAULT_BASEMAP, className="small basemap-list"),
        className="gui-card-title")]
    body += [
        _group("Opacity", ids.BASEMAP_OPACITY,
               dcc.Slider(id=ids.BASEMAP_OPACITY, min=0, max=1, step=0.05,
                          value=1.0, marks={0: "0", 1: "1"})),
        _lbl("Z-order", ids.BASEMAP_ZORDER),
        dbc.Select(id=ids.BASEMAP_ZORDER, value="below", size="sm",
                   options=[
                       {"label": "below overlays", "value": "below"},
                       {"label": "above raster overlays", "value": "above"}]),
    ]
    return html.Details([
        html.Summary(html.Span(className="gi gi-layers"),
                     title="Choose the background map",
                     **{"aria-label": "Choose the background map"}),
        html.Div(body, className="basemap-panel"),
    ], className="basemap-control")


def _map_pane() -> html.Div:
    return html.Div([
        build_map(),
        dbc.Badge("mode: off", id=ids.MODE_BADGE, color="secondary",
                  className="mode-badge"),
        _basemap_control(),
        html.Div(id=ids.NOTICE_STACK, className="notice-stack", role="status",
                 **{"aria-live": "polite"}),
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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
                _addon("Project CRS", ids.PROJECT_CRS),
                dbc.Input(id=ids.PROJECT_CRS, value="EPSG:25832", size="sm",
                          debounce=True),
            ], size="sm"),
        ], intro="Draw a rectangle or polygon with the map's draw tool; " +
                 "loaded data is clipped to it."),
        _card("Add vector data", [
            _group("Add data from", ids.DATA_SOURCE_TYPE, dbc.RadioItems(
                id=ids.DATA_SOURCE_TYPE, inline=True, value="local",
                options=[{"label": " Local file", "value": "local"},
                         {"label": " WFS server", "value": "wfs"}],
                className="small")),
            dbc.Checkbox(id=ids.CLIP_TO_AREA, value=True,
                         label="Clip to study area", className="small"),
            # only the panel matching the selected source shows
            # (data.toggle_source_panels); local is the default
            html.Div([
                _lbl("File path", ids.LOCAL_PATH),
                dbc.InputGroup([
                    dbc.Input(id=ids.LOCAL_PATH, placeholder="Path to .shp / " +
                              ".geojson / .gpkg / .gml / .kml (or .tif " +
                              "raster)", size="sm"),
                    _browse(ids.LOCAL_PATH, "a vector or raster file"),
                ], size="sm", className="mb-1"),
                _lbl("Sub-layer (GPKG, optional)", ids.LOCAL_LAYER),
                dbc.Input(id=ids.LOCAL_LAYER, placeholder="Sub-layer " +
                          "(GPKG, optional)", size="sm", className="mb-1"),
                html.Div([
                    dbc.Button("Load file", id=ids.LOCAL_LOAD_BTN, size="sm",
                               color="primary"),
                    dcc.Loading(html.Span(id=ids.LOCAL_LOAD_STATUS),
                                type="dot", parent_className="load-inline"),
                ], className="d-flex align-items-center gap-2"),
            ], id=ids.DATA_LOCAL_PANEL),
            html.Div([
                html.Small("Free ALKIS land-use, terrain (DEM), protection " +
                           "and topographic services for all of Germany. " +
                           "Filter by category and — with 'only in view' on " +
                           "— only the servers covering the current map area " +
                           "are listed.", className="text-muted"),
                _lbl("Category", ids.WFS_CATEGORY),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id=ids.WFS_CATEGORY,
                                         options=category_dd, value="",
                                         clearable=False,
                                         className="small"), width=7),
                    dbc.Col(dbc.Checkbox(id=ids.WFS_IN_VIEW_ONLY, value=True,
                                         label="only in view",
                                         className="small"), width=5),
                ], className="g-1 mb-1"),
                _lbl("Dataset / server", ids.WFS_PRESET),
                dcc.Dropdown(id=ids.WFS_PRESET, options=presets_dd, value="",
                             placeholder="Pick a dataset / server…",
                             className="small mb-1"),
                _lbl("WFS URL", ids.WFS_URL),
                dbc.Input(id=ids.WFS_URL, placeholder="WFS URL (https://…)",
                          size="sm", className="mb-1"),
                _lbl("Layer / feature type", ids.WFS_LAYER),
                dbc.InputGroup([
                    dbc.Input(id=ids.WFS_LAYER,
                              placeholder="Layer / feature type", size="sm"),
                    dbc.Button("List layers", id=ids.WFS_CAPS_BTN, size="sm",
                               color="secondary", outline=True,
                               title="Query GetCapabilities for this " +
                                     "server's feature types"),
                ], size="sm", className="mb-1"),
                _lbl("Discovered feature types", ids.WFS_LAYER_SELECT),
                dcc.Dropdown(id=ids.WFS_LAYER_SELECT, options=[], value=None,
                             placeholder="…or pick a discovered feature type",
                             className="small mb-1"),
                html.Div([
                    dbc.Button("Load from WFS", id=ids.WFS_LOAD_BTN,
                               size="sm", color="primary"),
                    # an explicit choice, never a second click: without a
                    # study area the server would send every feature
                    _help(dbc.Checkbox(
                        id=ids.WFS_WHOLE_LAYER, value=False, className="small",
                        label="Load the whole layer (no study area)"),
                        "Downloads EVERY feature of the layer and ignores the "
                        "study area. A state-wide layer can be millions of "
                        "features and gigabytes of memory, so leave this off "
                        "unless you really need all of it. Off = only the "
                        "study area is loaded (draw one first).",
                        className="wfs-whole-layer"),
                    dcc.Loading(html.Span(id=ids.WFS_LOAD_STATUS),
                                type="dot", parent_className="load-inline"),
                ], className="d-flex align-items-center flex-wrap gap-2"),
            ], id=ids.DATA_WFS_PANEL, style={"display": "none"}),
        ]),
        _collapsible_card("Overlays & terrain (WMS / DEM)", [
            _lbl("Overlay or terrain service", ids.OVERLAY_PRESET),
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
            # status line of the overlay / DEM loads: the spinner target for
            # callbacks/data.py (it writes its `children`; empty until then)
            dcc.Loading(html.Div(id=OVERLAY_STATUS,
                                 className="small text-info"),
                        type="dot", delay_show=_SPINNER_DELAY_MS),
        ], intro="Nationwide BKG basemaps and the DGM terrain model. " +
                 "'Add to map' does the right thing: a WMS becomes a live " +
                 "overlay (clipped to the study area); the DGM DEM downloads " +
                 "as an elevation raster for the study area (usable for " +
                 "slope-aware / overhead-line routing)."),
        _osm_section(),
        _card("Datasets", [
            dcc.Loading(html.Div(id=ids.DATASET_LIST, className="small"),
                        type="dot"),
        ]),
        _collapsible_card("Merge vector layers", [
            _lbl("Layers to merge", ids.MERGE_SELECT),
            dcc.Dropdown(id=ids.MERGE_SELECT, options=[], multi=True,
                         placeholder="Pick two or more vector layers…",
                         className="small"),
            dbc.InputGroup([
                _addon("Name", ids.MERGE_NAME),
                dbc.Input(id=ids.MERGE_NAME, size="sm",
                          placeholder="Merged land use"),
            ], size="sm"),
            html.Div(dbc.Button("Merge layers", id=ids.MERGE_BTN, size="sm",
                                color="primary", outline=True)),
        ], intro="Combine several loaded vector layers (e.g. the ALKIS " +
                 "land-use layers of neighbouring states in a cross-state " +
                 "project) into ONE new layer: rows are stacked, attribute " +
                 "columns aligned by name, everything reprojected to the " +
                 "project CRS. The merged layer then takes ONE cost table " +
                 "and ONE rasterization step."),
        _collapsible_card("Project", [
            _lbl("Save project to", ids.PROJECT_SAVE_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.PROJECT_SAVE_PATH, size="sm",
                          placeholder="Save project to folder…"),
                _browse(ids.PROJECT_SAVE_PATH, "the project save folder"),
            ], size="sm"),
            _group("Include in save", ids.SAVE_INCLUDE, dbc.Checklist(
                id=ids.SAVE_INCLUDE, inline=True, className="small",
                value=["raster", "vector", "cost_table"],
                options=[
                    {"label": " raster files", "value": "raster"},
                    {"label": " vector layers", "value": "vector"},
                    {"label": " cost tables & modifiers",
                     "value": "cost_table"}])),
            html.Small("Routes and the study area are ALWAYS saved with " +
                       "the project.", className="text-muted"),
            dbc.ButtonGroup([
                dbc.Button("Save project", id=ids.PROJECT_SAVE_BTN, size="sm",
                           color="primary"),
                dbc.Button("New project", id=ids.PROJECT_NEW_BTN, size="sm",
                           color="danger", outline=True),
            ]),
            _lbl("Open project file", ids.PROJECT_OPEN_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.PROJECT_OPEN_PATH, size="sm",
                          placeholder="Open project… (project.json)"),
                _browse(ids.PROJECT_OPEN_PATH, "a project file"),
            ], size="sm"),
            html.Div(dbc.Button("Open project", id=ids.PROJECT_OPEN_BTN,
                                size="sm", color="primary", outline=True)),
            # Save / Open write this line (they copy rasters and GPKG files)
            dcc.Loading(html.Div(id=ids.PROJECT_STATUS,
                                 className="small text-info", role="status",
                                 **{"aria-live": "polite"}),
                        type="dot", delay_show=_SPINNER_DELAY_MS),
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
        _lbl("1. Feature column (OSM tag key)", ids.OSM_KEY),
        dcc.Dropdown(id=ids.OSM_KEY, options=key_options, value=None,
                     placeholder="e.g. landuse, highway, power…",
                     className="small"),
        _lbl("2. Values in that column (empty = any)", ids.OSM_VALUES),
        dcc.Dropdown(id=ids.OSM_VALUES, options=[], value=[], multi=True,
                     placeholder="pick one or more values…",
                     className="small"),
        html.Div(dbc.Button("+ Add combination", id=ids.OSM_ADD_BTN,
                            size="sm", color="secondary", outline=True)),
        html.Div(id=ids.OSM_SELECTION_LIST, className="small my-1"),
        html.Div(dbc.Button("Load OSM features", id=ids.OSM_LOAD_BTN,
                            size="sm", color="primary", outline=True)),
        dbc.Accordion([dbc.AccordionItem([
            _lbl("OSM feature preset", ids.OSM_PRESET),
            dcc.Dropdown(id=ids.OSM_PRESET, options=options, value=None,
                         placeholder="Pick an OSM feature preset…",
                         className="small mb-1"),
            dbc.InputGroup([
                _addon("or tag", ids.OSM_TAGS),
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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    return html.Div([
        _card("Cost model", [
            _lbl("Base dataset", ids.COST_DATASET),
            dcc.Dropdown(id=ids.COST_DATASET, options=[], placeholder="Pick a " +
                         "loaded vector layer…", className="small"),
            _lbl("Feature column(s) — ordered; first = main",
                 ids.COST_FEATURE_KEYS),
            dcc.Dropdown(id=ids.COST_FEATURE_KEYS, options=[], multi=True,
                         placeholder="e.g. nutzart, bez", className="small"),
            html.Div(id=ids.COST_FEATURE_INFO),
            html.Div(dbc.Button("Seed cost table", id=ids.COST_SEED_BTN,
                                size="sm", color="primary",
                                title="Enabled once a base dataset and " +
                                      "feature column(s) are picked")),
            dag.AgGrid(
                id=ids.COST_GRID, columnDefs=[], rowData=[],
                defaultColDef={"editable": True, "resizable": True,
                               "sortable": True},
                dashGridOptions={"rowSelection": _row_selection(True),
                                 "stopEditingWhenCellsLoseFocus": True},
                className=FIT_COLUMNS, style=GRID_STYLE),
            dbc.ButtonGroup([
                dbc.Button("+ Row", id=ids.COST_ADD_ROW_BTN, size="sm",
                           color="secondary"),
                dbc.Button("− Selected", id=ids.COST_DEL_ROW_BTN, size="sm",
                           color="secondary"),
            ]),
            # Seed writes the coverage text: a spinner while it computes
            dcc.Loading(html.Div(id=ids.COST_COVERAGE_INFO,
                                 className="small text-warning"),
                        type="dot", delay_show=_SPINNER_DELAY_MS),
            _lbl("Import a cost table from", ids.COST_IMPORT_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.COST_IMPORT_PATH, size="sm",
                          placeholder="Import cost table (.csv/.json/.xlsx)"),
                _browse(ids.COST_IMPORT_PATH, "a cost table file"),
                dbc.Button("Import", id=ids.COST_IMPORT_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
            _lbl("Export the cost table to", ids.COST_EXPORT_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.COST_EXPORT_PATH, size="sm",
                          placeholder="Export cost table to…"),
                _browse(ids.COST_EXPORT_PATH, "the cost table export file"),
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
                {"field": "buffer_m", "headerName": "Buffer (m)",
                 "editable": True, "maxWidth": 95,
                 "headerTooltip": "Grow the matching features by this many "
                                  "metres before burning (0 or more)."},
                # how the row meets cells that are already forbidden, and the
                # library's zone rule (ignore_value / forbidden_value /
                # zone_field / forbidden_zone of modify_raster_from_dataset)
                {"field": "keep_forbidden", "headerName": "Keep forbidden",
                 "editable": True, "minWidth": 90, "maxWidth": 110,
                 "cellRenderer": "agCheckboxCellRenderer",
                 "cellEditor": "agCheckboxCellEditor",
                 "headerTooltip": "On (default): cells that are already "
                                  "forbidden (65535) stay forbidden. Off: "
                                  "this row may change them too."},
                {"field": "forbidden_value", "headerName": "Forbid as",
                 "editable": True, "minWidth": 80, "maxWidth": 100,
                 "headerTooltip": "Cost written for the zone value below "
                                  "(a whole number, 0 or more; 65535 = "
                                  "forbidden)."},
                {"field": "zone_column", "headerName": "Zone column",
                 "editable": True, "minWidth": 90, "maxWidth": 120,
                 "headerTooltip": "Optional: a column of the dataset. "
                                  "Features whose value in it equals the "
                                  "zone value are burnt as 'Forbid as' "
                                  "instead of Factor/cost."},
                {"field": "forbidden_zone", "headerName": "Zone value",
                 "editable": True, "minWidth": 80, "maxWidth": 110,
                 "headerTooltip": "The value in the zone column that marks "
                                  "the forbidden zone (not 0 or empty)."},
            ],
            defaultColDef={"resizable": True},
            dashGridOptions={"rowSelection": _row_selection(False),
                             "stopEditingWhenCellsLoseFocus": True},
            className=FIT_COLUMNS, style=SMALL_GRID_STYLE),
        dbc.ButtonGroup([
            dbc.Button("+ Modifier", id=ids.MODIFIER_ADD_BTN, size="sm",
                       color="secondary"),
            dbc.Button("− Selected", id=ids.MODIFIER_DEL_BTN, size="sm",
                       color="secondary"),
        ]),
        ], intro="Each row is one condition: pick a Dataset, a feature " +
                 "Column, an Operator (== ≠ < ≤ > ≥ in is-empty, or 'all' " +
                 "for the whole dataset) and a Value; then Mode " +
                 "(Multiply = factor, Override = replace, Per-feature = use " +
                 "each polygon's own value from the Column) and Factor/cost " +
                 "(e.g. 1.25, or 65535 to forbid). Example: water zones — " +
                 "one row each: ZONE == 1 → 100, ZONE == 2 → 2, …. For a " +
                 "drawn cost layer: Column = cost, Mode = per-feature."),
        _collapsible_card("Preprocessing", [
            _lbl("Preprocessing", ids.PREPROC_SELECT),
            dcc.Dropdown(
                id=ids.PREPROC_SELECT, className="small", value="none",
                options=[{"label": "None", "value": "none"},
                         {"label": "Street type + buffer (A/B/L)",
                          "value": "street_buffer"}]),
            _help(dbc.Row([
                dbc.Col(dbc.InputGroup([
                    _addon("A (m)", ids.PREPROC_BUF_A),
                    dbc.Input(id=ids.PREPROC_BUF_A, type="number", value=10,
                              min=0, step=0.5, size="sm")], size="sm"),
                    width=4),
                dbc.Col(dbc.InputGroup([
                    _addon("B (m)", ids.PREPROC_BUF_B),
                    dbc.Input(id=ids.PREPROC_BUF_B, type="number", value=4,
                              min=0, step=0.5, size="sm")], size="sm"),
                    width=4),
                dbc.Col(dbc.InputGroup([
                    _addon("L (m)", ids.PREPROC_BUF_L),
                    dbc.Input(id=ids.PREPROC_BUF_L, type="number", value=2,
                              min=0, step=0.5, size="sm")], size="sm"),
                    width=4),
            ], className="g-1"),
                "Street buffers in metres for the street-type preset: A, B "
                "and L are the buffer widths of the three street classes. "
                "Each is 0 or more."),
            html.Div("Custom preprocessing steps", className="gui-subhead"),
            html.Small("Ordered edits to the base dataset before rasterizing. " +
                       "Each row selects rows by a condition (or a condition " +
                       "GROUP built below) then acts: buffer = grow geometry " +
                       "by Arg metres; set = write Arg into the Target " +
                       "column; keep/drop = filter. Example: buffer all " +
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
                dashGridOptions={"rowSelection": _row_selection(False),
                                 "stopEditingWhenCellsLoseFocus": True},
                className=FIT_COLUMNS, style=SMALL_GRID_STYLE),
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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """The condition-group step builder (round 11).

    '+ Add condition' collects (column, operator, value) triples; the group
    operator (& / |) combines them; the live preview shows the python-like
    mask, e.g. (("nutzart" == "Wald") & ("bez" == "Nadelholz")) -> buffer=2m.
    'Add step' appends the finished step to the steps grid below. Column and
    value are searchable dropdowns fed from the picked dataset's actual data.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    return html.Div([
        _lbl("Build a step from conditions", ids.PPB_DATASET,
             className="small fw-bold"),
        dcc.Dropdown(id=ids.PPB_DATASET, options=[], value=None,
                     placeholder="Dataset the step applies to…",
                     className="small mb-1"),
        dbc.Row([
            dbc.Col([_lbl("Column", ids.PPB_COLUMN),
                     dcc.Dropdown(id=ids.PPB_COLUMN, options=[], value=None,
                                  placeholder="Column…", className="small")],
                    width=5),
            dbc.Col([_lbl("Operator", ids.PPB_OPERATOR),
                     dbc.Select(
                         id=ids.PPB_OPERATOR, size="sm", value="==",
                         options=[{"label": o, "value": o} for o in
                                  ("==", "!=", "<", "<=", ">", ">=", "in",
                                   "is-empty")])], width=3),
            dbc.Col([_lbl("Value", ids.PPB_VALUE),
                     dcc.Dropdown(id=ids.PPB_VALUE, options=[], value=None,
                                  placeholder="Value…", className="small")],
                    width=4),
        ], className="g-1"),
        html.Div(dbc.Button("+ Add condition", id=ids.PPB_ADD_COND_BTN,
                            size="sm", color="secondary", outline=True,
                            className="my-1")),
        html.Div(id=ids.PPB_COND_LIST, className="small"),
        dbc.Row([
            dbc.Col([
                _lbl("Combine", ids.PPB_COMBINE),
                dbc.Select(id=ids.PPB_COMBINE, size="sm", value="&",
                           options=[
                               {"label": "& (all must match)", "value": "&"},
                               {"label": "| (any matches)", "value": "|"}]),
            ], width=6),
            dbc.Col([
                _lbl("Action", ids.PPB_OP),
                dbc.Select(id=ids.PPB_OP, size="sm", value="buffer",
                           options=[{"label": o, "value": o} for o in
                                    ("buffer", "set", "keep", "drop")]),
            ], width=6),
        ], className="g-1"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                _addon("Target", ids.PPB_TARGET),
                dbc.Input(id=ids.PPB_TARGET, size="sm",
                          placeholder="column (set)"),
            ], size="sm"), width=6),
            dbc.Col(dbc.InputGroup([
                _addon("Arg (m or value)", ids.PPB_ARG),
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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Draw + edit custom cost polygons, per-polygon cost, in the Cost tab (33).

    No draw-target radio anymore: shapes drawn while the COST tab is open
    become cost polygons; drawn anywhere else they set the study area
    (data.set_draw_target keys the shared draw tool off the active tab).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    return _collapsible_card("Custom cost polygons", [
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                _addon("Name", ids.MANUAL_NAME),
                dbc.Input(id=ids.MANUAL_NAME, size="sm", value="Manual costs")],
                size="sm"), width=7),
            dbc.Col([_lbl("Mode", ids.MANUAL_MODE, className="small visually-hidden"),
                     dbc.Select(
                         id=ids.MANUAL_MODE, size="sm", value="override",
                         options=[{"label": "override", "value": "override"},
                                  {"label": "base", "value": "base"}])],
                    width=5),
        ], className="g-1"),
        _help(dbc.InputGroup([
            _addon("Default cost", ids.MANUAL_COST),
            dbc.Input(id=ids.MANUAL_COST, type="number", value=65535,
                      min=0, step=1, size="sm"),
        ], size="sm"),
            "The cost a new polygon starts with (0 or more, whole numbers). "
            "65535 marks the area forbidden. Edit each polygon's own cost "
            "in the table below."),
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
            dashGridOptions={"rowSelection": _row_selection(True),
                             "stopEditingWhenCellsLoseFocus": True},
            className=FIT_COLUMNS, style=SMALL_GRID_STYLE),
        dbc.ButtonGroup([
            dbc.Button("Snapshot drawn shapes", id=ids.MANUAL_CREATE_BTN,
                       size="sm", color="secondary", outline=True,
                       title="Also build the layer from the currently drawn " +
                             "shapes"),
            dbc.Button("− Remove selected", id=ids.MANUAL_DEL_BTN, size="sm",
                       color="secondary", outline=True,
                       title="Drop the selected polygon(s) from the cost " +
                             "layer (move/add vertices with the map's edit " +
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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    return html.Div([
        _card("Rasterization", [
            _lbl("Dataset", ids.RASTERIZE_DATASET),
            dcc.Dropdown(id=ids.RASTERIZE_DATASET, options=[], value="",
                         clearable=False, className="small"),
            _lbl("Cost table", ids.RASTERIZE_TABLE),
            dcc.Dropdown(id=ids.RASTERIZE_TABLE, options=[], value="current",
                         clearable=False, className="small"),
            html.Small("Only cost tables whose feature columns exist in the " +
                       "picked dataset are offered — a table can never be " +
                       "applied to a dataset it doesn't belong to.",
                       className="text-muted"),
            _help(dbc.InputGroup([
                _addon("Resolution (m)", ids.RES_M),
                dbc.Input(id=ids.RES_M, type="number", value=1.0, min=0.1,
                          step=0.1, size="sm"),
            ], size="sm"),
                "Cell size of the cost raster in metres (above 0; the field "
                "takes 0.1 or more). 1 m suits parcel data; coarser cells "
                "build faster and use less memory."),
            html.Div(id=ids.RASTER_SIZE_INFO, className="small text-info",
                     children="-"),
            dbc.Accordion([dbc.AccordionItem([
                _help(dbc.InputGroup([
                    _addon("Fill value (cost)", ids.FILL_VALUE),
                    dbc.Input(id=ids.FILL_VALUE, type="number", value=65535,
                              min=0, max=65535, step=1, size="sm"),
                ], size="sm", className="mb-1"),
                    "Cost of cells no feature covers and of categories with "
                    "no cost: a whole number from 0 to 65535 for uint16 (the "
                    "range follows the data type below). 65535 = forbidden, "
                    "0 = free."),
                html.Small("Unmapped categories become this value — 65535 = " +
                           "forbidden (F11).", className="text-muted"),
                dbc.InputGroup([
                    _addon("Data type", ids.RASTER_DTYPE),
                    dbc.Select(id=ids.RASTER_DTYPE, value="uint16", options=[
                        {"label": v, "value": v}
                        for v in ("uint16", "uint32", "float32")]),
                ], size="sm", className="mb-1"),
                # dbc.Input takes no ``title``: the help sits on the wrapper
                _help(dbc.InputGroup([
                    _addon("Geometry buffer (m)", ids.GEOM_BUFFER),
                    dbc.Input(id=ids.GEOM_BUFFER, type="number", value=1,
                              min=0, step=0.5, size="sm"),
                ], size="sm", className="mb-1"),
                    "Outward buffer on every feature before it is burnt, in "
                    "metres (0 or more). 1 m at 1 m cells clears most ALKIS "
                    "sub-cell parcel defects. The GUI starts at 1 m; the "
                    "library's own default is 0 m."),
                html.Div("Thin features", className="gui-subhead"),
                _help(dbc.Switch(
                    id=ids.WIDEN_THIN, value=False, className="small",
                    label="Widen thin forbidden features"),
                    "Off (default). On: a forbidden feature narrower than "
                    "1.4 cells is widened to that width before it is burnt, "
                    "so a thin wall cannot vanish between cell centres. It "
                    "can also close a real gap in a thin wall; a finer "
                    "resolution is the only exact fix."),
                _help(dbc.Switch(
                    id=ids.ALL_TOUCHED, value=False, className="small",
                    label="Burn forbidden features into every touched cell"),
                    "Off (default): a cell is forbidden only if its centre "
                    "lies inside a forbidden feature. On: also every cell a "
                    "forbidden feature merely touches, so a forbidden zone "
                    "grows by up to one cell. This also seals real openings "
                    "narrower than a cell (a gate, a gap between parcels)."),
                _help(html.Div([
                    _lbl("If a thin feature is lost", ids.ON_THIN),
                    dbc.Select(id=ids.ON_THIN, value="warn", size="sm",
                               options=[
                                   {"label": "warn (default)", "value": "warn"},
                                   {"label": "raise (stop the build)",
                                    "value": "raise"},
                                   {"label": "ignore", "value": "ignore"}]),
                ], className="mb-1"),
                    "What happens when the burn does not represent the data "
                    "faithfully (a forbidden feature dropped, split or holed, "
                    "or a repair that sealed open space): warn shows a "
                    "notice, raise stops the build with an error, ignore "
                    "skips the check."),
                html.Div("Extent", className="gui-subhead"),
                _help(dbc.Switch(
                    id=ids.CROP_BORDER, value=False, className="small",
                    label="Crop the outer forbidden border"),
                    "Off (default). On: after the build, rows and columns at "
                    "the edge that hold only forbidden cells (65535) are cut "
                    "off, so the raster is no larger than the passable area."),
            ], title="Advanced")], start_collapsed=True),
            _lbl("Save raster as", ids.RASTER_SAVE_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.RASTER_SAVE_PATH, size="sm",
                          placeholder="Save raster as… (blank = work dir, " +
                                      "F2)"),
                _browse(ids.RASTER_SAVE_PATH, "the raster save file"),
            ], size="sm"),
            html.Div([
                dbc.Button("Build cost raster", id=ids.RASTERIZE_BTN,
                           size="sm", color="primary", className="gui-cta",
                           title="Enabled once a cost table is seeded in " +
                                 "the Cost tab"),
                # shown only while a heavy build runs in the background
                dbc.Button("⏹ Stop", id=ids.RASTERIZE_STOP_BTN, size="sm",
                           color="danger", style={"display": "none"},
                           title="Interrupt the running build - it stops " +
                                 "at the next stage; no raster is kept"),
            ], className="d-flex align-items-center gap-2"),
            dcc.Loading(html.Pre(id=ids.RASTERIZE_LOG,
                                 className="small bg-light p-1",
                                 style={"maxHeight": "140px",
                                        "overflowY": "auto"}),
                        type="dot"),
        ]),
        _card("Load existing raster", [
            _lbl("Raster file (.tif)", ids.RASTER_LOAD_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.RASTER_LOAD_PATH, size="sm",
                          placeholder="Path to .tif"),
                _browse(ids.RASTER_LOAD_PATH, "a raster file"),
                dbc.Button("Load", id=ids.RASTER_LOAD_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
        ]),
        _collapsible_card("Display & legend", [
            _group(
                "Raster overlay opacity", ids.RASTER_OPACITY,
                dcc.Slider(id=ids.RASTER_OPACITY, min=0, max=1, step=0.05,
                           value=0.7, marks={0: "0", 0.5: "0.5", 1: "1"})),
            _lbl("Colormap", ids.RASTER_COLORMAP),
            dbc.Select(id=ids.RASTER_COLORMAP, value="viridis",
                       options=[{"label": c, "value": c} for c in COLORMAPS],
                       size="sm"),
            html.Small("Applies to every raster layer; re-colours instantly " +
                       "(F3). Select a built cost raster in the Layers tab " +
                       "for its cost ↔ colour ↔ combination legend below.",
                       className="text-muted"),
            # QGIS-style graduated classification of the SELECTED raster
            html.Div("Graduated rendering", className="gui-subhead"),
            dbc.Row([
                dbc.Col([
                    _lbl("Render", ids.GRAD_MODE),
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
                    _lbl("Classes (2 to 32)", ids.GRAD_CLASSES),
                    dbc.Input(id=ids.GRAD_CLASSES, type="number", value=5,
                              min=2, max=32, step=1, size="sm"),
                ], width=5),
            ], className="g-1"),
            html.Div(dbc.Button("Classify", id=ids.GRAD_APPLY_BTN, size="sm",
                                color="primary", outline=True,
                                title="Classify the raster selected in the " +
                                      "Layers tab (or the only raster) with " +
                                      "the chosen method")),
            # Classify reads the whole band: a spinner while it works
            dcc.Loading(html.Div(id=ids.GRAD_STATUS,
                                 className="small text-info"),
                        type="dot", delay_show=_SPINNER_DELAY_MS),
            html.Small("Pick a classification and press Classify — the " +
                       "legend below gains one editable colour per class " +
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
        _lbl("Rasters to combine", ids.RASTER_COMBINE_SELECT),
        dcc.Dropdown(id=ids.RASTER_COMBINE_SELECT, options=[], multi=True,
                     placeholder="Select raster layers to combine…",
                     className="small"),
        dbc.InputGroup([
            _addon("Operation", ids.RASTER_COMBINE_OP),
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
                            title="Enabled once two or more raster layers " +
                                  "are loaded")),
        dcc.Loading(html.Div(id=ids.RASTER_COMBINE_STATUS,
                             className="small text-info"), type="dot"),
    ], intro="Pick two or more raster layers and an operation. Inputs "
             "with a different grid/CRS are resampled onto the FIRST "
             "selected raster's grid; forbidden (65535) cells are kept "
             "forbidden.")


# ----------------------------------------------------------------- routes tab
def _routes_tab() -> html.Div:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    return html.Div([
        _card("New routing", [
            _lbl("Route on raster", ids.ROUTE_RASTER),
            dcc.Dropdown(id=ids.ROUTE_RASTER, options=[], className="small",
                         placeholder="Pick a raster layer…"),
            _group("Map click…", ids.BUILD_MODE, dbc.RadioItems(
                id=ids.BUILD_MODE, inline=True, value="off",
                options=[{"label": " Off", "value": "off"},
                         {"label": " Source", "value": "source"},
                         {"label": " Target", "value": "target"},
                         {"label": " +Waypoint", "value": "waypoint"},
                         {"label": " Move waypoint", "value": "move-waypoint"}],
                className="small")),
            dbc.Switch(id=ids.EDIT_ENABLE, value=False, className="small",
                       label="Edit the selected route (move/add source, " +
                             "target & waypoints update it in place)"),
            html.Small("Edit off → clicks build a NEW route. Edit on → " +
                       "clicks update the route selected below (Source/" +
                       "Target move it; +Waypoint adds one; Move waypoint " +
                       "relocates the nearest waypoint). Editing replaces " +
                       "the route.", className="text-muted"),
            html.Div("Control points — ONE list for the new route or the " +
                     "selected route", className="small"),
            html.Small("Rows are editable (typed coordinates in the project " +
                       "CRS) and deletable; order = routing order. With a " +
                       "route selected, 'Apply points' recomputes it; the " +
                       "new route replaces it ('Undo last edit' brings the " +
                       "old one back).", className="text-muted"),
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
                        {"function": "params.value == null ? '' : " +
                                     "d3.format(',.0f')(params.value)"}},
                ],
                defaultColDef={"resizable": True},
                dashGridOptions={"rowSelection": _row_selection(True),
                                 "stopEditingWhenCellsLoseFocus": True},
                # responsive: the sidebar is drag-resizable — a one-shot
                # fit would leave columns overflowed/virtualized out
                className=FIT_COLUMNS_RESPONSIVE, style=SMALL_GRID_STYLE),
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
                           title="Interrupt the running routing job — it " +
                                 "stops after the segment in flight; " +
                                 "completed routes are kept"),
            ], className="d-flex align-items-center gap-2"),
            dcc.Loading(html.Div(id=ids.ROUTING_STATUS,
                                 className="small text-info", role="status",
                                 **{"aria-live": "polite"}),
                        type="circle"),
        ]),
        _collapsible_card("Algorithm & search", [
        dbc.Row([
            dbc.Col(html.Div([
                _lbl("Backend", ids.GRAPH_API),
                dcc.Dropdown(
                    id=ids.GRAPH_API, clearable=False, className="small",
                    value=routing.DEFAULT_BACKEND,
                    options=routing.backend_options()),
            ], title="Which pyorps graph backend runs the search. cython is "
                     "the default. A backend whose package is not installed "
                     "is greyed out and names the install command; the GPU "
                     "backends need CuPy and a CUDA GPU."), width=5),
            dbc.Col(html.Div([
                _lbl("Algorithm", ids.ALGORITHM),
                dcc.Dropdown(id=ids.ALGORITHM, clearable=False,
                             value=routing.default_algorithm(
                                 routing.DEFAULT_BACKEND),
                             className="small",
                             options=[{"label": a.replace("_", " "),
                                       "value": a} for a in
                                      routing.backend_algorithms(
                                          routing.DEFAULT_BACKEND)]),
            ], title="Only the algorithms the chosen backend really runs are "
                     "listed. Changing the backend resets this to the "
                     "backend's default."), width=7),
        ], className="g-1"),
        html.Div(dbc.InputGroup([
            _addon("Neighbourhood k", ids.NEIGHBORHOOD),
            dbc.Input(id=ids.NEIGHBORHOOD, type="number", value=2, min=0,
                      max=127, step=1, size="sm"),
        ], size="sm", className="mt-1"),
            title="How many neighbours each raster cell connects to: k = 0 "
                  "gives 4 directions, 1 gives 8, 2 gives 16 (recommended), "
                  "3 gives 32. A larger k turns the route less angular but "
                  "makes the search much heavier. Whole number from 0 to 127."),
        html.Div(dbc.InputGroup([
            _addon("Search buffer (m)", ids.SEARCH_BUFFER),
            dbc.Input(id=ids.SEARCH_BUFFER, type="number", placeholder="auto",
                      min=0, step=10, size="sm"),
        ], size="sm", className="mt-1"),
            title="How far around the control points the search may look, in "
                  "metres. Empty or 0 = automatic: the larger of 1000 m and "
                  "1.5 times the point distance."),
        html.Div(dbc.Switch(
            id=ids.FULL_WINDOW, value=False, className="small",
            label="Use the whole raster as search window"),
            title="Search the entire raster instead of a window around the "
                  "points. The route can then never be cut off by a small "
                  "buffer, but a large raster makes the run slow and "
                  "memory-hungry. The buffer above is not used while this is "
                  "on."),
        html.Div(id=ids.SEARCH_BUFFER_INFO, className="small text-info mb-1",
                 children="blank = max(1000 m, 1.5 x distance); or switch "
                          "on 'whole raster'"),
        # ``ignore_max_cost=True`` is the library's switch that makes cells at
        # the maximum cost (65535, forbidden) hard barriers; it also moves a
        # source/target that sits on one to the nearest routable cell.
        _help(dbc.Checkbox(
            id=ids.IGNORE_MAX_COST, value=True, className="small",
            label="Forbidden cells are barriers (route around them)"),
            "On (default): cells at the maximum cost (65535, forbidden) can "
            "never be crossed, and a source or target standing on one is "
            "moved to the nearest routable cell (the shift is reported). "
            "Off: forbidden cells may be crossed, each metre costing 65535, "
            "and points are not moved."),
        dbc.Checkbox(id=ids.PAIRWISE, value=False, className="small",
                     label="Pairwise (source i → target i)"),
        _help(dbc.Row([
            dbc.Col(dbc.Checkbox(id=ids.SIMPLIFY, value=False,
                                 label="Simplify for display/export (F5)",
                                 className="small"), width=8),
            dbc.Col([
                _lbl("Simplify tolerance (m)", ids.SIMPLIFY_TOL,
                     className="small visually-hidden"),
                dbc.Input(id=ids.SIMPLIFY_TOL, type="number", value=1.0,
                          min=0, step=0.1, size="sm")], width=4),
        ], className="g-1"),
            "Simplify tolerance in metres (0 or more). Only the drawn and "
            "exported line is simplified; costs stay those of the full "
            "route."),
        dbc.Accordion([dbc.AccordionItem([
            html.Div(dbc.InputGroup([
                _addon("delta (cost units)", ids.DELTA),
                dbc.Input(id=ids.DELTA, type="number", value=100,
                          min=0, step=10, size="sm")],
                size="sm", className="mb-1"),
                id=ids.DELTA_WRAP, title=routing.DELTA_HELP),
            html.Div(dbc.InputGroup([
                _addon("threads (0 = auto)", ids.NUM_THREADS),
                dbc.Input(id=ids.NUM_THREADS, type="number", value=0,
                          min=0, step=1, size="sm")],
                size="sm", className="mb-1"),
                id=ids.THREADS_WRAP, title=routing.THREADS_HELP),
            html.Div(dbc.InputGroup([
                _addon("margin (× distance)", ids.MARGIN),
                dbc.Input(id=ids.MARGIN, type="number", placeholder="default",
                          min=0, step=0.1, size="sm")],
                size="sm", className="mb-1"),
                id=ids.MARGIN_WRAP, title=routing.MARGIN_HELP),
        ], title="Advanced")], start_collapsed=True),
        ]),
        # objective & terrain card (closed by default; owned by sections/objective)
        *objective_section.section(),
        _collapsible_card("Constrained routing", [
            dbc.Switch(id=ids.OHL_ENABLE, value=False, className="small",
                       label="Constrained routing (overhead line / towers)"),
            html.Small("When on, the extra tower/profile options appear below " +
                       "and 'Run routing' does constrained routing on the " +
                       "SAME source/target/waypoint points — each waypoint " +
                       "forces a tower.", className="text-muted"),
            dbc.Collapse(_ohl_options(), id=ids.OHL_COLLAPSE, is_open=False),
        ]),
        # cost-field and corridor-graph cards (closed by default; Q3: they live
        # in the Routes tab, owned by sections/cost_field and sections/corridor)
        *cost_field_section.section(),
        *corridor_section.section(),
        _collapsible_card("Computed routes", [
        dbc.Row([
            dbc.Col([
                _lbl("Active route", ids.EDIT_ROUTE_SELECT),
                dcc.Dropdown(
                    id=ids.EDIT_ROUTE_SELECT, options=[], value=None,
                    placeholder="Select the active route…",
                    className="small")], width=9),
            dbc.Col(dbc.Button("New", id=ids.NEW_ROUTE_BTN, size="sm",
                               color="secondary", outline=True,
                               title="Deselect — the points list goes back " +
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
            dashGridOptions={"rowSelection": _row_selection(True),
                             "animateRows": True,
                             "stopEditingWhenCellsLoseFocus": True},
            className=FIT_COLUMNS, style=SMALL_GRID_STYLE),
        html.Small("Edit name / group / colour (hex) / line-style / " +
                   "visibility inline. Retype the group to move a route, or " +
                   "select route(s) and use 'Move to group'.",
                   className="text-muted"),
        dbc.InputGroup([
            _addon("Move to group", ids.ROUTE_MOVE_GROUP),
            dbc.Input(id=ids.ROUTE_MOVE_GROUP, size="sm",
                      placeholder="group name (blank = ungrouped)"),
            dbc.Button("Move sel.", id=ids.ROUTE_MOVE_BTN, size="sm",
                       color="secondary"),
        ], size="sm"),
        # group-level actions
        _lbl("Group actions", ids.GROUP_SELECT),
        dcc.Dropdown(id=ids.GROUP_SELECT, options=[], className="small",
                     placeholder="Pick a group…"),
        _lbl("New group name", ids.GROUP_RENAME, className="small visually-hidden"),
        dbc.InputGroup([
            dbc.Input(id=ids.GROUP_RENAME, size="sm",
                      placeholder="rename group to…"),
            dbc.Button("Rename", id=ids.GROUP_RENAME_BTN, size="sm",
                       color="secondary"),
        ], size="sm"),
        dbc.Row([
            dbc.Col([
                _lbl("Group colour", ids.GROUP_COLOR,
                     className="small visually-hidden"),
                dbc.Input(id=ids.GROUP_COLOR, type="color",
                          value="#e6194b", size="sm",
                          style={"height": "31px"})], width=3),
            dbc.Col([
                _lbl("Group line style", ids.GROUP_DASH,
                     className="small visually-hidden"),
                dbc.Select(id=ids.GROUP_DASH, size="sm", value="solid",
                           options=[{"label": d, "value": d} for d in
                                    ("solid", "dashed", "dotted",
                                     "dash-dot")])], width=4),
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
            title="Copy the selected group's sources, targets and waypoints " +
                  "back into the New-routing points list (they are removed " +
                  "from it automatically when a run finishes)")),
        dcc.Loading(html.Div(id=ids.EDIT_COST_READOUT, className="small"),
                    type="dot"),
        html.Div(dbc.Button("Discard selected variant",
                            id=ids.DELETE_VARIANT_BTN, size="sm",
                            color="danger", outline=True)),
        ], intro="New routes appear here right after the run and are " +
                 "selected automatically. Editing a route (move source/" +
                 "target, add/move waypoints) REPLACES it; use 'Run routing' " +
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
            html.Small("Auto-refresh off → map-click edits are STAGED (the " +
                       "route dashes); press 'Refresh edited' to recompute " +
                       "only the edited route(s).", className="text-muted"),
            # a plain html.Button: dbc.Button takes no aria-label
            html.Div(html.Button(
                "↶ Undo last edit", id=ids.EDIT_UNDO_BTN, type="button",
                disabled=True, className="btn btn-outline-secondary btn-sm",
                title="Bring back the route as it was before the last edit. "
                      "One step only.",
                **{"aria-label": "Undo last edit: bring back the route as it "
                                 "was before the last edit. One step only."})),
            html.Div("Simplify displayed/exported line (F5)",
                     className="small"),
            dbc.InputGroup([
                _addon("tolerance (m)", ids.EDIT_SIMPLIFY_TOL),
                dbc.Input(id=ids.EDIT_SIMPLIFY_TOL, type="number", min=0,
                          step=0.5, placeholder="0 = off", size="sm"),
                dbc.Button("Apply", id=ids.EDIT_SIMPLIFY_BTN, size="sm",
                           color="primary", outline=True),
            ], size="sm"),
            html.Small("Costs always come from the full routed line; only " +
                       "the drawn/exported geometry is simplified.",
                       className="text-muted"),
        ], intro="Turn on 'Edit the selected route' above, then click the " +
                 "map (Source/Target/+Waypoint/Move waypoint), or edit/" +
                 "delete rows in the control-points list and press " +
                 "'Apply points'."),
        _card("Export / load routes", [
            _lbl("Export the selected route to", ids.EXPORT_ROUTE_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.EXPORT_ROUTE_PATH, size="sm",
                          placeholder="Export selected to " +
                                      ".geojson/.gpkg/.shp/.csv"),
                _browse(ids.EXPORT_ROUTE_PATH, "the route export file"),
                dbc.Button("Export", id=ids.EXPORT_ROUTE_BTN, size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
            dcc.Loading(html.Div(id=ids.EXPORT_STATUS,
                                 className="small text-success"),
                        type="dot", delay_show=_SPINNER_DELAY_MS),
            _lbl("Load routes from", ids.ROUTES_LOAD_PATH),
            dbc.InputGroup([
                dbc.Input(id=ids.ROUTES_LOAD_PATH, size="sm",
                          placeholder="Load routes (.geojson/.gpkg/.shp)"),
                _browse(ids.ROUTES_LOAD_PATH, "a route file"),
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
        html.Small("Tower-aware routing from an infrastructure profile " +
                   "(angles, spans, tower costs). Fill in the profile form " +
                   "below; the raw YAML text is kept under “Raw YAML " +
                   "(advanced)”.",
                   className="text-muted"),
        _lbl("Shipped profile", ids.OHL_PROFILE_PRESET),
        dcc.Dropdown(id=ids.OHL_PROFILE_PRESET, options=[],
                     placeholder="Load a shipped profile…",
                     className="small my-1"),
        _lbl("Profile file", ids.OHL_PROFILE_PATH),
        dbc.InputGroup([
            dbc.Input(id=ids.OHL_PROFILE_PATH, size="sm",
                      placeholder="Profile file (.yaml/.json)"),
            _browse(ids.OHL_PROFILE_PATH, "a profile file"),
            dbc.Button("Load", id=ids.OHL_PROFILE_LOAD_BTN, size="sm",
                       color="primary", outline=True),
            dbc.Button("Save", id=ids.OHL_PROFILE_SAVE_BTN, size="sm",
                       color="success", outline=True),
        ], size="sm", className="my-1"),
        # the profile form (P6) and, below it, the raw YAML text in a
        # "Raw YAML (advanced)" block (sections/profile_form)
        *profile_form_section.section(),
        *profile_form_section.yaml_block(),
        dbc.Row([
            dbc.Col([
                _lbl("Backend", ids.OHL_BACKEND),
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
        _lbl("DEM file (slope, optional)", ids.OHL_DEM),
        dbc.Input(id=ids.OHL_DEM, size="sm", className="mt-1",
                  placeholder="DEM path (slope, optional)"),
        _lbl("DSM file (clearance, optional)", ids.OHL_DSM),
        dbc.Input(id=ids.OHL_DSM, size="sm", className="mt-1",
                  placeholder="DSM path (clearance, optional)"),
        # shown only while a heavy constrained run works in the background
        dbc.Button("⏹ Stop", id=ids.OHL_STOP_BTN, size="sm", color="danger",
                   className="mt-1", style={"display": "none"},
                   title="Interrupt the running constrained job - it stops " +
                         "after the segment in flight; an unfinished " +
                         "overhead line is discarded"),
        html.Div(id=ids.OHL_STATUS, className="small mt-1"),
        html.Small("Terrain cost readout is 2-D (F9) — compare routes by " +
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
                # once; keep=True (the old suppressRowDeselection) stops a
                # plain re-click from toggling the selection off
                dashGridOptions={"rowSelection": _row_selection(True, keep=True),
                                 "stopEditingWhenCellsLoseFocus": True,
                                 "animateRows": True},
                className=FIT_COLUMNS,
                style={"height": "40vh", "width": "100%"}),
            dbc.ButtonGroup([
                dbc.Button("▲ Up", id="layer-up-btn", size="sm",
                           color="secondary"),
                dbc.Button("▼ Down", id="layer-down-btn", size="sm",
                           color="secondary"),
                dbc.Button("📋 Table", id=ids.LAYER_VIEW_BTN, size="sm",
                           color="secondary", outline=True,
                           title="Open the selected layer's full attribute " +
                                 "table"),
                dbc.Button("Remove", id=ids.LAYER_REMOVE_BTN, size="sm",
                           color="danger", outline=True),
            ]),
            html.Small("Ctrl+click selects several layers — ▲▼ and Remove " +
                       "act on all of them. Click a feature on the map to " +
                       "highlight it, or open the layer's full table.",
                       className="text-muted d-block"),
        ], intro="Top row = foreground (painted last), bottom row = " +
                 "background. Route groups appear as ONE row here; the " +
                 "individual routes are listed in 'Routes' below."),
        _card("Routes", [
            html.Div(id=ids.ROUTE_TREE, className="small route-tree"),
        ], intro="Routes grouped by run/group — expand to see individual " +
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
            # icon-only: a plain html.Button, dbc.Button takes no aria-label
            html.Button([html.Span(className="gi gi-sun"),
                         html.Span(className="gi gi-moon")],
                        id=ids.THEME_TOGGLE, type="button",
                        className="btn btn-link btn-sm",
                        title="Switch between light and dark theme",
                        **{"aria-label": "Switch between light and dark theme"}),
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
    """Assemble the full app layout.

    A blank project gives the blank map (R1). When the server-side ``state``
    already holds a project (the page was reloaded, or a second tab was
    opened) every store and component whose content is DERIVED from that
    state starts filled, so the browser shows what the server holds
    (:func:`_state_prefill`).
    """
    # width= (not md=) so the columns NEVER stack: with a 100vh map, a
    # stacked sidebar would sit invisibly below the fold on narrow windows.
    layout = dbc.Container(
        _stores() + [
            dbc.Row([
                dbc.Col(_map_pane(), width=8, className="p-0 gui-map-col"),
                dbc.Col(_sidebar(), width=4, className="p-0 gui-sidebar-col"),
            ], id=ids.APP_SHELL, className="g-0 flex-nowrap gui-shell"),
            _layer_table_offcanvas(),
            _log_offcanvas(),
            _routing_confirm_modal(),
            _confirm_modal(),
        ],
        fluid=True, className="p-0")
    _apply_prefill(layout, _state_prefill(state))
    return layout


# ------------------------------------------------- server state -> first paint
_Put = Callable[[str, str, Any], None]


def _crs_text(crs: Any) -> str:
    """The project CRS as the text the CRS field shows (``EPSG:25832``)."""
    if isinstance(crs, str):
        return crs
    to_string = getattr(crs, "to_string", None)
    return str(to_string() if callable(to_string) else crs)


def _prefill_layers(state, put: _Put) -> None:
    """Layer list, project status, CRS: the stores everything else follows."""
    put(ids.PROJECT_CRS, "value", _crs_text(state.project_crs))
    view = state.layers_view()
    if view:
        put(ids.LAYERS_VIEW, "data", view)
    project_path = getattr(state, "project_path", None)
    if project_path:                    # opened (or saved) from a file: name it
        put(ids.PROJECT_STATUS, "children", f"opened: {project_path}")
    elif view:
        put(ids.PROJECT_STATUS, "children",
            f"session restored: {len(view)} layer(s) are still loaded on "
            "the server")


def _prefill_data_tab(state, put: _Put) -> None:
    """Study-area text and the dataset list (read from the same helpers the
    callbacks use, so a reload and a live update cannot disagree)."""
    from .callbacks.data import _dataset_list, _study_area_info

    if state.study_area:
        put(ids.STUDY_AREA_INFO, "children", _study_area_info(state))
    if state.layers_of_kind("vector"):
        put(ids.DATASET_LIST, "children", _dataset_list(state))


def _prefill_routes(state, put: _Put) -> None:
    """Route dropdown and the active route's map pieces (all values are
    computed first, so a failure cannot leave a half-restored selection)."""
    routes = state.layers_of_kind("route")
    if not routes:
        return
    options = [{"label": ly.name, "value": ly.id} for ly in routes]
    active = state.get(state.active_route_id) if state.active_route_id else None
    if active is None or active.kind != "route":
        put(ids.EDIT_ROUTE_SELECT, "options", options)
        return
    from .callbacks.edit import control_markers, cost_readout, points_rows, route_positions

    pieces = {
        (ids.ACTIVE_ROUTE, "positions"): route_positions(active),
        (ids.CONTROL_MARKERS, "children"): control_markers(active),
        (ids.BUILD_POINTS_TABLE, "rowData"): points_rows(active),
        (ids.EDIT_COST_READOUT, "children"): cost_readout(active),
    }
    put(ids.EDIT_ROUTE_SELECT, "options", options)
    put(ids.EDIT_ROUTE_SELECT, "value", active.id)
    put(ids.UI_STATE, "data", {**_DEFAULT_UI_STATE, "active_route_id": active.id})
    tolerance = (active.meta or {}).get("simplify_tol")
    if tolerance is not None:
        put(ids.EDIT_SIMPLIFY_TOL, "value", tolerance)
    for (component_id, prop), value in pieces.items():
        put(component_id, prop, value)


#: routing setting key -> id of the control that shows it (its value)
ROUTING_SETTING_CONTROLS: dict[str, str] = {
    "graph_api": ids.GRAPH_API,
    "algorithm": ids.ALGORITHM,
    "neighborhood_k": ids.NEIGHBORHOOD,
    "search_buffer_m": ids.SEARCH_BUFFER,
    "full_window": ids.FULL_WINDOW,
    "ignore_max_cost": ids.IGNORE_MAX_COST,
    "pairwise": ids.PAIRWISE,
    "simplify": ids.SIMPLIFY,
    "simplify_tol": ids.SIMPLIFY_TOL,
    "delta": ids.DELTA,
    "num_threads": ids.NUM_THREADS,
    "margin": ids.MARGIN,
}


def _prefill_routing_settings(state, put: _Put) -> None:
    """The routing controls take the settings the server holds (a reload, or a
    project that was opened); a blank project leaves the layout's defaults."""
    settings = routing.normalise_routing_settings(
        getattr(state, "routing_settings", None))
    if not settings:
        return
    for key, control in ROUTING_SETTING_CONTROLS.items():
        put(control, "value", settings[key])
    put(ids.ALGORITHM, "options",
        [{"label": a.replace("_", " "), "value": a}
         for a in routing.backend_algorithms(settings["graph_api"])])
    put(ids.ROUTING_SETTINGS, "data", settings)


#: rasterize setting key -> id of the control that shows it (its value)
RASTER_SETTING_CONTROLS: dict[str, str] = {
    "resolution_m": ids.RES_M,
    "fill_value": ids.FILL_VALUE,
    "dtype": ids.RASTER_DTYPE,
    "geometry_buffer_m": ids.GEOM_BUFFER,
    "widen_thin_forbidden": ids.WIDEN_THIN,
    "all_touched": ids.ALL_TOUCHED,
    "on_thin_features": ids.ON_THIN,
    "crop_border": ids.CROP_BORDER,
}


def _prefill_raster_settings(state, put: _Put) -> None:
    """The rasterize controls take the settings the server holds (a reload, or
    a project that was opened); a blank project leaves the layout's defaults."""
    settings = rasterize.normalise_raster_settings(
        getattr(state, "raster_settings", None))
    if not settings:
        return
    for key, control in RASTER_SETTING_CONTROLS.items():
        put(control, "value", settings[key])
    put(ids.RASTER_SETTINGS, "data", settings)


def _prefill_cost_table(state, put: _Put) -> None:
    """The cost table of an opened project (``--project``, "Open project").

    The cost editor is client-side, so without this a reload shows an empty
    table while the project is loaded; with it the page shows the table as it
    was loaded (not later browser edits). A project without a table, and a blank
    one, leave the layout's empty grids. The values come from the helper that
    "Open project" sends to the browser with, so the two cannot disagree.
    """
    from .callbacks.costmodel import _coverage_text
    from .callbacks.io import cost_table_view

    table = getattr(state, "loaded_cost_table", None)
    if not table:
        return
    view = cost_table_view(table)
    try:
        coverage = _coverage_text(state, view["dataset_id"], view["keys"],
                                  view["rows"])
    except Exception:  # pylint: disable=broad-exception-caught  # the coverage line is a hint; the table must still show
        _LOG.warning("page prefill: cost coverage text failed", exc_info=True)
        coverage = ""
    put(ids.COST_GRID, "columnDefs", view["column_defs"])
    put(ids.COST_GRID, "rowData", view["rows"])
    put(ids.COST_GRID_STATE, "data", view["grid_state"])
    put(ids.MODIFIER_GRID, "rowData", view["modifiers"])
    put(ids.COST_COVERAGE_INFO, "children", coverage or "")


def _prefill_routing_job(state, put: _Put) -> None:
    """A heavy routing job that outlived the page: poll it again and show Stop.

    Without this the poll interval stays off and Stop stays hidden while the
    server still holds the job, so "Run routing" would refuse forever.
    """
    job = state.routing_job
    if job is None:
        return
    put(ids.ROUTING_POLL, "disabled", False)          # also collects a finished job
    if _handing_over(job):
        put(ids.ROUTING_STATUS, "children", jobs_service.finalizing_text(jobs_service.ROUTING_LABEL))
    elif getattr(job, "status", "running") == "running":
        put(ids.ROUTING_STOP_BTN, "style", {})
        put(ids.ROUTING_STATUS, "children", job.progress_text())


def _handing_over(job: Any) -> bool:
    """The ended job's result is being added to the project right now (a claimed or finalizing handover)."""
    handover = getattr(job, "handover", None)
    return handover is not None and getattr(handover, "phase", None) != jobs_service.READY


#: the section modules of round 14; each has ``prefill(state, put)``
_SECTION_MODULES = (objective_section, cost_field_section, corridor_section,
                    profile_form_section)


def _section_prefill_step(module) -> Callable[[Any, _Put], None]:
    """One section's ``prefill`` as a step of its own, so a failing section is
    logged under its name and skipped without taking the others down. The
    module's ``prefill`` is looked up when the step runs, not now."""
    def step(state, put: _Put) -> None:
        module.prefill(state, put)

    step.__name__ = f"{module.__name__.rsplit('.', maxsplit=1)[-1]}_prefill"
    return step


def _prefill_sections(state, put: _Put) -> None:
    """The cards of round 14 (objective, cost field, corridor, constrained
    profile form) put their saved settings into their controls."""
    for module in _SECTION_MODULES:
        staged = _run_prefill_step(_section_prefill_step(module), state)
        for component_id, props in staged.items():
            for prop, value in props.items():
                put(component_id, prop, value)


#: background job kind -> (poll id, stop button id or None, status id)
JOB_CONTROLS: dict[str, tuple[str, str | None, str]] = {
    "rasterize": (ids.RASTERIZE_POLL, ids.RASTERIZE_STOP_BTN, ids.RASTERIZE_LOG),
    "constrained": (ids.OHL_POLL, ids.OHL_STOP_BTN, ids.OHL_STATUS),
    "cost_field": (ids.CF_POLL, ids.CF_STOP_BTN, ids.CF_STATUS),
    "corridor": (ids.COR_POLL, None, ids.COR_STATUS),
}


#: background job kind -> (module under ``pyorps.gui``, function) that words the status line of a RUNNING job of
#: that kind. Each kind says honestly where its Stop takes effect (after the segment in flight, at a stage
#: boundary, between fixed points, or that it cannot be stopped); the job service's own line is generic. The
#: modules are imported when needed (``callbacks`` import this module, so a top-level import would be a cycle).
JOB_STATUS_TEXT: dict[str, tuple[str, str]] = {
    "rasterize": ("callbacks.raster", "rasterize_status_text"),
    "constrained": ("services.constrained", "status_text"),
    "cost_field": ("callbacks.cost_field", "status_text"),
    "corridor": ("services.corridor", "status_text"),
}


def _job_status_text(kind: str, job: Any) -> str:
    """The status line a running job of ``kind`` shows: its own wording, else the job's ``progress_text()``."""
    target = JOB_STATUS_TEXT.get(kind)
    if target is not None:
        import importlib

        try:
            function = getattr(importlib.import_module(f"{__package__}.{target[0]}"), target[1])
            return function(job)
        except Exception:  # pylint: disable=broad-exception-caught  # a status line must never take the page down
            _LOG.warning("status text of the %s job failed; showing the generic line", kind, exc_info=True)
    return job.progress_text()


def _prefill_jobs(state, put: _Put) -> None:
    """Background jobs that outlived the page: poll them again and show Stop.

    Same reason as :func:`_prefill_routing_job`: without it the poll stays off
    while the server still holds the job, so the next start would be refused
    for ever. A job is read duck-typed: ``status`` ("running" when missing),
    ``progress_text()`` and ``can_stop`` (True when missing).
    """
    for kind, job in dict(getattr(state, "jobs", None) or {}).items():
        controls = JOB_CONTROLS.get(kind)
        if controls is None or job is None:
            continue
        poll, stop, status = controls
        put(poll, "disabled", False)                  # also collects a finished job
        if _handing_over(job):                         # the result is being added: say so, Stop stays hidden
            put(status, "children", jobs_service.finalizing_text(getattr(job, "label", "") or kind))
            continue
        if getattr(job, "status", "running") != "running":
            continue
        if stop is not None and getattr(job, "can_stop", True):
            put(stop, "style", {})
        put(status, "children", _job_status_text(kind, job))


def _state_prefill(state) -> dict[str, dict[str, Any]]:
    """``{component id: {property: value}}`` for everything derived from ``state``.

    A fresh, empty state yields only the values the layout already has, so a
    blank project is exactly the blank layout. A step that fails is skipped
    (and logged): a broken project must not take the whole page down.
    """
    values: dict[str, dict[str, Any]] = {}
    if state is None:
        return values
    for step in (_prefill_layers, _prefill_data_tab, _prefill_routes,
                 _prefill_routing_settings, _prefill_raster_settings,
                 _prefill_cost_table, _prefill_routing_job,
                 _prefill_sections, _prefill_jobs):
        staged = _run_prefill_step(step, state)
        for component_id, props in staged.items():
            values.setdefault(component_id, {}).update(props)
    return values


def _run_prefill_step(step: Callable[[Any, _Put], None],
                      state) -> dict[str, dict[str, Any]]:
    """What one prefill step wants to set; nothing if the step fails."""
    staged: dict[str, dict[str, Any]] = {}

    def put(component_id: str, prop: str, value: Any) -> None:
        staged.setdefault(component_id, {})[prop] = value

    try:
        step(state, put)
    except Exception:  # pylint: disable=broad-exception-caught  # a reload must still show the page
        _LOG.warning("page prefill step %s failed", step.__name__,
                     exc_info=True)
        return {}
    return staged


def _apply_prefill(layout, values: dict[str, dict[str, Any]]) -> None:
    """Assign the prefilled properties to the components that carry the ids."""
    if not values:
        return
    found: set[str] = set()
    for component in layout._traverse():  # pylint: disable=protected-access  # Dash's own tree walk
        component_id = getattr(component, "id", None)
        if isinstance(component_id, str) and component_id in values:
            found.add(component_id)
            for prop, value in values[component_id].items():
                setattr(component, prop, value)
    for missing in sorted(set(values) - found):
        _LOG.warning("page prefill: no component with id %r", missing)


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


def _confirm_modal() -> dbc.Modal:
    """The one reusable "are you sure?" box (callbacks/confirm.py fills it).

    Same look as the heavy-run gate: a static backdrop, no Escape, an explicit
    Cancel and a confirm button whose label names the consequence.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id=ids.CONFIRM_TITLE),
                        close_button=False),
        dbc.ModalBody(id=ids.CONFIRM_BODY),
        dbc.ModalFooter([
            dbc.Button("Cancel", id=ids.CONFIRM_CANCEL, size="sm",
                       color="secondary", outline=True),
            dbc.Button("Confirm", id=ids.CONFIRM_OK, size="sm",
                       color="danger"),
        ]),
    ], id=ids.CONFIRM_MODAL, is_open=False, backdrop="static",
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
            ], className="d-flex justify-content-between align-items-center " +
                         "gap-2 mb-1"),
            dag.AgGrid(
                id=ids.LAYER_TABLE_GRID, rowData=[], columnDefs=[],
                getRowId="params.data.__row",
                defaultColDef={"resizable": True, "sortable": True,
                               "filter": True, "minWidth": 90},
                dashGridOptions={"rowSelection": _row_selection(False),
                                 "pagination": True,
                                 "paginationPageSize": TABLE_PAGE_SIZE},
                columnSize="autoSize",
                style={"height": "30vh", "width": "100%"}),
        ],
        id=ids.LAYER_TABLE_OFFCANVAS, is_open=False, placement="bottom",
        scrollable=True, backdrop=False, title="Layer attributes",
        # left 66% only (the map column) → the sidebar is never covered
        style={"height": "40vh", "width": "66%", "left": 0, "right": "auto"})

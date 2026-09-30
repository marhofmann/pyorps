"""
PYORPS GUI callbacks: the single map-click dispatcher + mode badge (8.1/8.2).

``click_mode`` is ONE value in the ui-state store — build and edit modes are
mutually exclusive; the badge over the map always shows what a click will do.
Map clicks enter the app in exactly one place (C3: ``Map.clickData``) and are
routed by mode: build modes append to the route draft, edit modes emit an
edit-request consumed by the edit callbacks. Attribute clicks on features are
separate (attrs.py) and unaffected.
"""
from __future__ import annotations

import math

import dash_leaflet as dl
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import geo

_BADGE = {
    "off": ("mode: off", "secondary"),
    "build:source": ("click adds SOURCE", "success"),
    "build:target": ("click adds TARGET", "danger"),
    "build:waypoint": ("click adds WAYPOINT", "warning"),
    "edit:source": ("click MOVES SOURCE", "success"),
    "edit:target": ("click MOVES TARGET", "danger"),
    "edit:waypoint": ("click ADDS WAYPOINT", "warning"),
    "edit:move-waypoint": ("click MOVES NEAREST WAYPOINT", "warning"),
}

#: source = green, target = red, waypoints = yellow (user requirement)
_KIND_COLOR = {"source": "#2ca02c", "target": "#d62728",
               "waypoint": "#f1c40f"}


def routing_crs(state, raster_layer_id: str | None):
    """CRS clicks are projected into: the routing raster's, else project."""
    layer = state.get(raster_layer_id) if raster_layer_id else None
    if layer is not None and layer.crs is not None:
        return layer.crs
    return state.project_crs


def point_marker(kind: str, lat: float, lng: float, index: int,
                 name: str = ""):
    """A control-point marker. A named waypoint shows its name permanently
    beside the symbol (Feature 1); unnamed points get a hover tooltip."""
    color = _KIND_COLOR.get(kind, "#555555")
    label = name.strip() if isinstance(name, str) else ""
    return dl.DivMarker(
        position=[lat, lng],
        iconOptions={
            "html": f'<div class="gui-point-marker" '
                    f'style="background:{color}"></div>',
            "className": "", "iconSize": [14, 14], "iconAnchor": [7, 7]},
        children=dl.Tooltip(label or f"{kind} {index + 1}",
                            permanent=bool(label), direction="right"))


def draft_markers(draft: dict) -> list:
    markers = []
    for kind in ("source", "target", "waypoint"):
        for i, p in enumerate(draft.get(f"{kind}s", []) or []):
            markers.append(point_marker(kind, p["lat"], p["lng"], i,
                                        p.get("name", "")))
    return markers


def leg_distances(rows: list[dict]) -> list[dict]:
    """Annotate rows with 'dist' = euclidean metres from the previous point.

    x/y are already in the metric routing CRS, so the leg distance between
    consecutive control points (source→waypoint→…→target) is a plain hypot.
    Rows without coordinates get None and break the chain.
    """
    prev = None
    for row in rows:
        x, y = row.get("x"), row.get("y")
        if x in (None, "") or y in (None, ""):
            row["dist"] = None
            prev = None
            continue
        point = (float(x), float(y))
        row["dist"] = (None if prev is None
                       else round(math.hypot(point[0] - prev[0],
                                             point[1] - prev[1]), 1))
        prev = point
    return rows


def draft_rows(draft: dict) -> list[dict]:
    # chain order (source → waypoints → target) so each row's 'dist' is the
    # euclidean length of the leg arriving at that point
    rows = []
    for kind in ("source", "waypoint", "target"):
        for p in draft.get(f"{kind}s", []) or []:
            rows.append({"kind": kind, "x": p["x"], "y": p["y"],
                         "name": p.get("name", "")})
    return leg_distances(rows)


def register(app, state) -> None:
    # ------------------------------------------------ unified mode selection
    # ONE mode radio; the "Edit selected route" knob decides whether a click
    # BUILDS the new-route draft or EDITS the selected route (task 55). The
    # ui-state ``click_mode`` keeps the "build:X" / "edit:X" scheme the click
    # dispatcher and edit callbacks already understand.
    @app.callback(
        Output(ids.UI_STATE, "data", allow_duplicate=True),
        Output(ids.MODE_BADGE, "children"),
        Output(ids.MODE_BADGE, "color"),
        Input(ids.BUILD_MODE, "value"),
        Input(ids.EDIT_ENABLE, "value"),
        State(ids.UI_STATE, "data"),
        prevent_initial_call=True)
    def set_mode(build_value, edit_enable, ui_state):
        ui_state = dict(ui_state or {})
        action = build_value or "off"
        if action == "off":
            mode = "off"
        elif edit_enable:
            mode = f"edit:{action}"
        elif action == "move-waypoint":
            mode = "off"          # move-waypoint only applies when editing
        else:
            mode = f"build:{action}"
        ui_state["click_mode"] = mode
        label, color = _BADGE.get(mode, ("mode: off", "secondary"))
        return ui_state, label, color

    # --------------------------------------------- THE map-click entry point
    @app.callback(
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Output(ids.EDIT_REQUEST, "data"),
        Input(ids.MAP, "clickData"),
        State(ids.UI_STATE, "data"),
        State(ids.ROUTE_DRAFT, "data"),
        State(ids.ROUTE_RASTER, "value"),
        prevent_initial_call=True)
    def dispatch_click(click_data, ui_state, draft, raster_layer_id):
        if not click_data or not click_data.get("latlng"):
            raise PreventUpdate
        mode = (ui_state or {}).get("click_mode", "off")
        if mode == "off":
            raise PreventUpdate
        lat = float(click_data["latlng"]["lat"])
        lng = float(click_data["latlng"]["lng"])
        crs = routing_crs(state, raster_layer_id)
        x, y = geo.point_wgs84_to_crs(lat, lng, crs)
        point = {"lat": lat, "lng": lng, "x": x, "y": y, "name": ""}

        family, _, action = mode.partition(":")
        if family == "build":
            draft = {k: list(v) for k, v in (draft or {}).items()}
            draft.setdefault("sources", [])
            draft.setdefault("targets", [])
            draft.setdefault("waypoints", [])
            draft[f"{action}s"].append(point)
            return draft, no_update
        if family == "edit":
            seq = (click_data.get("n_clicks") or 0)
            return no_update, {"action": action, **point, "seq": seq}
        raise PreventUpdate

    # ------------------------------------------- draft -> markers + table view
    @app.callback(
        Output(ids.BUILDER_MARKERS, "children"),
        Output(ids.BUILD_POINTS_TABLE, "rowData"),
        Input(ids.ROUTE_DRAFT, "data"))
    def render_draft(draft):
        draft = draft or {}
        if state.active_route_id:
            # a route is selected: the ONE points grid shows ITS control
            # points (edit.py owns it); only refresh the draft markers
            return draft_markers(draft), no_update
        return draft_markers(draft), draft_rows(draft)

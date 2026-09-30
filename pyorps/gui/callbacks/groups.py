"""
PYORPS GUI callbacks: the editable routes grid + route-group management
(Feature 2, replaces the read-only lineage tree and the separate
"Name, group & style" fields).

A route is a ``Layer(kind="route")`` and a group is a free-string
``meta["group"]``. The grid lists every route grouped by that string; name,
colour, line-style and visibility are edited inline (cellValueChanged). Routes
move between groups by dragging (managed row-drag, synced from virtualRowData)
or, reliably, by selecting one/many and pressing "Move to group". Whole groups
can be renamed, restyled, reordered (to front/back) or deleted.
"""
from __future__ import annotations

from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import routing
from ..services.errors import Notice, success
from .edit import DASH_STYLES, _route_options, dash_style_name

DEFAULT_COLOR = "#e6194b"


def _fmt_point(point) -> str:
    """A control point as a compact 'x, y' label for the routes grid."""
    if not point:
        return ""
    return f"{float(point[0]):,.0f}, {float(point[1]):,.0f}"


def _route_row(layer) -> dict:
    style = layer.style or {}
    meta = layer.meta or {}
    metrics = meta.get("metrics") or {}
    points = meta.get("control_points") or []
    cost = metrics.get("total_cost")
    length = metrics.get("total_length_m")
    return {
        "id": layer.id,
        "name": layer.name,
        "group": meta.get("group") or "",
        "color": style.get("color", DEFAULT_COLOR),
        "dash": dash_style_name(style.get("dashArray")),
        "visible": bool(layer.visible),
        "wp": max(0, len(points) - 2),
        "cost": int(round(float(cost))) if cost is not None else None,
        "length": int(round(float(length))) if length is not None else None,
        "from": _fmt_point(points[0]) if points else "",
        "to": _fmt_point(points[-1]) if points else "",
    }


def _routes_rows(state) -> list[dict]:
    return [_route_row(ly) for ly in state.layers_of_kind("route")]


def _group_options(state) -> list[dict]:
    return [{"label": g, "value": g} for g in state.route_groups()]


def _set_dash(style: dict, dash_name: str) -> dict:
    style = dict(style or {})
    dash_array = DASH_STYLES.get(dash_name or "solid")
    if dash_array is None:
        style.pop("dashArray", None)
    else:
        style["dashArray"] = dash_array
    return style


def register(app, state) -> None:
    # --------------------------------------------- grid mirror + group options
    @app.callback(Output(ids.ROUTES_GRID, "rowData"),
                  Output(ids.GROUP_SELECT, "options"),
                  Input(ids.LAYERS_VIEW, "data"))
    def sync_routes_grid(_view):
        return _routes_rows(state), _group_options(state)

    # ------------------------------------- inline edits (name/colour/dash/vis)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Input(ids.ROUTES_GRID, "cellValueChanged"),
        prevent_initial_call=True)
    def apply_route_cell_edit(events):
        if not events:
            raise PreventUpdate
        changed = False
        for event in (events if isinstance(events, list) else [events]):
            data = event.get("data") or {}
            layer = state.get(data.get("id"))
            if layer is None or layer.kind != "route":
                continue
            column = (event.get("colId")
                      or (event.get("column") or {}).get("colId"))
            if column == "name":
                state.rename(layer.id, data.get("name"))
                routing.refresh_route_display(layer)   # geojson carries name
            elif column == "visible":
                state.set_visible(layer.id, bool(data.get("visible")))
            elif column == "color":
                layer.style = {**(layer.style or {}),
                               "color": data.get("color") or DEFAULT_COLOR}
            elif column == "dash":
                layer.style = _set_dash(layer.style, data.get("dash"))
            elif column == "group":
                layer.meta["group"] = str(data.get("group") or "").strip() or None
            else:
                continue
            changed = True
        if not changed:
            raise PreventUpdate
        return state.layers_view(), _route_options(state)

    # ----------------------------- clicking a grid row selects the active route
    @app.callback(Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
                  Input(ids.ROUTES_GRID, "selectedRows"),
                  prevent_initial_call=True)
    def grid_selects_active(selected):
        if not selected:
            raise PreventUpdate
        route_id = selected[0].get("id")
        if not route_id or state.get(route_id) is None:
            raise PreventUpdate
        if route_id == state.active_route_id:
            raise PreventUpdate
        return route_id

    # --------------------- drag a route onto another group (managed row-drag)
    @app.callback(Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
                  Input(ids.ROUTES_GRID, "virtualRowData"),
                  prevent_initial_call=True)
    def sync_group_from_drag(virtual_rows):
        changed = False
        for row in virtual_rows or []:
            layer = state.get(row.get("id"))
            if layer is None or layer.kind != "route":
                continue
            new_group = str(row.get("group") or "").strip() or None
            if new_group != ((layer.meta or {}).get("group") or None):
                layer.meta["group"] = new_group
                changed = True
        if not changed:
            raise PreventUpdate
        return state.layers_view()

    # ------------------------------ move selected route(s) to a group (button)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.ROUTE_MOVE_BTN, "n_clicks"),
        State(ids.ROUTES_GRID, "selectedRows"),
        State(ids.ROUTE_MOVE_GROUP, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def move_to_group(n_clicks, selected, target, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        routes = [state.get(r.get("id")) for r in (selected or [])]
        routes = [r for r in routes if r is not None and r.kind == "route"]
        if not routes:
            notices.append(Notice(
                severity="warning", title="Select route(s) to move",
                meaning="Pick one or more routes in the grid first.",
                impact="Nothing was moved.",
                fix="Select route row(s), then Move to group.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return no_update, notices
        group = str(target or "").strip() or None
        for route in routes:
            route.meta["group"] = group
        notices.append(success(
            f"Moved {len(routes)} route(s) to "
            f"'{group or 'ungrouped'}'"))
        return state.layers_view(), notices

    # ------------------------------------------------ group: rename all members
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GROUP_RENAME_BTN, "n_clicks"),
        State(ids.GROUP_SELECT, "value"),
        State(ids.GROUP_RENAME, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def rename_group(n_clicks, group, new_name, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        members = state.routes_in_group(group) if group else []
        if not members:
            notices.append(_pick_group_notice())
            return no_update, notices
        new_group = str(new_name or "").strip() or None
        for route in members:
            route.meta["group"] = new_group
        notices.append(success(
            f"Renamed group to '{new_group or 'ungrouped'}' "
            f"({len(members)} route(s))"))
        return state.layers_view(), notices

    # ----------------------------------------------- group: restyle all members
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GROUP_RESTYLE_BTN, "n_clicks"),
        State(ids.GROUP_SELECT, "value"),
        State(ids.GROUP_COLOR, "value"),
        State(ids.GROUP_DASH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def restyle_group(n_clicks, group, color, dash, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        members = state.routes_in_group(group) if group else []
        if not members:
            notices.append(_pick_group_notice())
            return no_update, notices
        for route in members:
            style = _set_dash(route.style, dash)
            style["color"] = color or DEFAULT_COLOR
            route.style = style
        notices.append(success(
            f"Restyled {len(members)} route(s) in '{group}'"))
        return state.layers_view(), notices

    # ----------------------------------------------- group: reorder (front/back)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Input(ids.GROUP_UP_BTN, "n_clicks"),
        Input(ids.GROUP_DOWN_BTN, "n_clicks"),
        State(ids.GROUP_SELECT, "value"),
        prevent_initial_call=True)
    def reorder_group(_up, _down, group):
        members = {m.id for m in (state.routes_in_group(group) if group else [])}
        if not members:
            raise PreventUpdate
        ordered = [ly.id for ly in state.ordered_layers()]
        rest = [i for i in ordered if i not in members]
        block = [i for i in ordered if i in members]
        # "up" = towards the front (drawn on top = higher z)
        new_order = (rest + block if ctx.triggered_id == ids.GROUP_UP_BTN
                     else block + rest)
        state.reorder(new_order)
        return state.layers_view()

    # -------------- re-add a finished group's points to the New-routing list
    @app.callback(
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.READD_POINTS_BTN, "n_clicks"),
        State(ids.GROUP_SELECT, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def readd_group_points(n_clicks, group, notices):
        """Points are consumed from the New-routing list when a run finishes;
        this brings a group's sources/targets/waypoints back into the draft
        (deduplicated) so the group can be re-routed or extended."""
        from ..services.geo import crs_transformer

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        members = state.routes_in_group(group) if group else []
        if not members:
            notices.append(_pick_group_notice())
            return no_update, notices
        draft = {"sources": [], "targets": [], "waypoints": []}
        seen: set[tuple] = set()

        def _push(kind, x, y, crs, name=""):
            key = (kind, round(float(x), 3), round(float(y), 3))
            if key in seen:
                return
            seen.add(key)
            tf = crs_transformer(str(crs), "EPSG:4326")
            lon, lat = tf.transform(float(x), float(y))
            draft[f"{kind}s"].append({"lat": lat, "lng": lon,
                                      "x": float(x), "y": float(y),
                                      "name": str(name or "")})

        for route in members:
            points = (route.meta or {}).get("control_points") or []
            if len(points) < 2 or route.crs is None:
                continue
            names = (route.meta or {}).get("waypoint_names") or []
            _push("source", *points[0], route.crs)
            for i, p in enumerate(points[1:-1]):
                _push("waypoint", *p, route.crs,
                      names[i] if i < len(names) else "")
            _push("target", *points[-1], route.crs)
        n_points = sum(len(v) for v in draft.values())
        if not n_points:
            notices.append(Notice(
                severity="warning", title="No control points in this group",
                meaning="The group's routes carry no source/target points "
                        "(imported routes may not).",
                impact="The New-routing list is unchanged.",
                fix="Pick a group of computed routes.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return no_update, notices
        notices.append(success(
            f"{n_points} point(s) of '{group}' re-added to New routing",
            meaning="Deselect the active route ('New') to see them in the "
                    "points list, then Run routing."))
        return draft, notices

    # ----------------------------------------------- group: delete all members
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GROUP_DELETE_BTN, "n_clicks"),
        State(ids.GROUP_SELECT, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def delete_group(n_clicks, group, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        members = state.routes_in_group(group) if group else []
        if not members:
            notices.append(_pick_group_notice())
            return no_update, no_update, no_update, notices
        active_removed = any(m.id == state.active_route_id for m in members)
        for route in list(members):
            state.remove_layer(route.id)
        notices.append(success(
            f"Deleted group '{group}' ({len(members)} route(s))"))
        new_active = None if active_removed else no_update
        return (state.layers_view(), _route_options(state), new_active,
                notices)


def _pick_group_notice() -> dict:
    return Notice(
        severity="warning", title="Pick a group first",
        meaning="Group actions apply to all routes in the chosen group.",
        impact="Nothing was changed.",
        fix="Select a group from the dropdown.",
        focus_id=ids.TAB_ROUTES, focus_control=ids.GROUP_SELECT).to_dict()

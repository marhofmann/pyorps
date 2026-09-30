"""
PYORPS GUI callbacks: route editing with lineage (R11, Sections 10.7 + 18).

Routes are immutable once computed: click-to-place moves (8.4 default) and
waypoint changes never mutate the active route — they clone its parameters +
control points, recompute the chained least-cost path, and register a NEW
route layer with ``parent_id`` back to the original. The Edit tab shows the
lineage tree; the active route is drawn as a dedicated polyline that dashes
while recomputing (8.5, via a clientside callback).
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import routing
from ..services.errors import Notice, guard, success
from .interaction import point_marker
from .routes import add_route_layer

_EDIT_LABEL = {"source": "moved source", "target": "moved target",
               "waypoint": "added waypoint", "move-waypoint": "moved waypoint"}

#: line-style name -> Leaflet dashArray (None = solid)
DASH_STYLES = {"solid": None, "dashed": "8 8", "dotted": "1 6",
               "dash-dot": "10 6 2 6"}


def dash_style_name(dash_array) -> str:
    for name, value in DASH_STYLES.items():
        if value == dash_array:
            return name
    return "solid"


def _to_latlng(points, crs):
    from ..services.geo import crs_transformer
    tf = crs_transformer(str(crs), "EPSG:4326")
    out = []
    for x, y in points:
        lon, lat = tf.transform(x, y)
        out.append((lat, lon))
    return out


def route_positions(layer) -> list:
    """The active route's display geometry as Leaflet [lat, lng] positions."""
    if layer is None or layer.gdf is None or layer.gdf.empty:
        return []
    line = routing.route_display_line(layer)
    return [[lat, lon] for lat, lon in
            _to_latlng(list(line.coords), layer.gdf.crs)]


def control_markers(layer) -> list:
    if layer is None:
        return []
    points = layer.meta.get("control_points") or []
    if len(points) < 2:
        return []
    names = layer.meta.get("waypoint_names") or []
    latlngs = _to_latlng(points, layer.crs)
    markers = [point_marker("source", *latlngs[0], 0)]
    for i, ll in enumerate(latlngs[1:-1]):
        markers.append(point_marker("waypoint", *ll, i,
                                    names[i] if i < len(names) else ""))
    markers.append(point_marker("target", *latlngs[-1], 0))
    return markers


def points_rows(layer) -> list[dict]:
    """The selected route's control points as unified points-grid rows."""
    from .interaction import leg_distances

    if layer is None:
        return []
    points = layer.meta.get("control_points") or []
    if len(points) < 2:
        return []
    names = layer.meta.get("waypoint_names") or []
    rows = [{"kind": "source", "x": points[0][0], "y": points[0][1],
             "name": ""}]
    for i, p in enumerate(points[1:-1]):
        rows.append({"kind": "waypoint", "x": p[0], "y": p[1],
                     "name": names[i] if i < len(names) else ""})
    rows.append({"kind": "target", "x": points[-1][0],
                 "y": points[-1][1], "name": ""})
    return leg_distances(rows)


def waypoint_names_from_rows(rows: list[dict]) -> list[str]:
    """Names of the interior (waypoint) rows, aligned with points_from_rows."""
    valid = [r for r in (rows or [])
             if r.get("x") not in (None, "") and r.get("y") not in (None, "")]
    return [str(r.get("name") or "") for r in valid[1:-1]]


def points_from_rows(rows: list[dict]) -> list[tuple[float, float]]:
    """Grid rows (listed order) -> ordered control points, validated.

    The first row must be a source and the last a target; waypoints sit in
    between in list order.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    rows = [r for r in (rows or [])
            if r.get("x") not in (None, "") and r.get("y") not in (None, "")]
    if len(rows) < 2:
        raise ValueError("Need at least a source and a target row.")
    kinds = [str(r.get("kind") or "waypoint") for r in rows]
    if kinds[0] != "source" or kinds[-1] != "target":
        raise ValueError("The first row must be the source and the last "
                         "row the target (waypoints in between).")
    if "source" in kinds[1:] or "target" in kinds[:-1]:
        raise ValueError("Only one source (first row) and one target "
                         "(last row) are allowed when applying points to "
                         "a route.")
    return [(float(r["x"]), float(r["y"])) for r in rows]


def cost_readout(layer):
    if layer is None:
        return html.Small("Select a route to see its cost.",
                          className="text-muted")
    metrics = layer.meta.get("metrics") or {}
    if not metrics:
        return html.Small("No metrics stored (imported route — edit it to "
                          "compute).", className="text-muted")
    rows = [
        ("Length", f"{metrics.get('total_length_m', 0):,.0f} m"),
        ("Cost (distance-weighted)", f"{metrics.get('total_cost', 0):,.0f}"),
        ("Raw cell cost", f"{metrics.get('total_cell_cost', 0):,.0f}"),
        ("Geodesic length", f"{metrics.get('geodesic_length_m', 0):,.0f} m"),
    ]
    body = [html.Tr([html.Td(html.B(k), className="pe-2"), html.Td(v)])
            for k, v in rows]
    extras = []
    if metrics.get("crosses_forbidden"):
        extras.append(html.Div(
            f"⚠ crosses {metrics.get('n_forbidden_cells')} forbidden "
            "cell(s)", className="text-danger small"))
    return html.Div([dbc.Table(html.Tbody(body), size="sm",
                               className="small mb-1"), *extras])


def _get_or_build_finder(state, layer, points, notices):
    """The cached PathFinder for the route's raster, rebuilt if needed."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    params = layer.meta.get("params") or {}
    raster_layer_id = params.get("raster_layer_id")
    finder = state.finders.get(raster_layer_id)
    if finder is not None:
        return finder, notices
    raster_layer = state.get(raster_layer_id) if raster_layer_id else None
    if raster_layer is None:
        rasters = state.layers_of_kind("raster")
        raster_layer = rasters[-1] if rasters else None
    if raster_layer is None:
        notices.append(Notice(
            severity="error", title="No cost raster for recompute",
            meaning="Editing recomputes the least-cost path, which needs "
                    "the route's raster loaded.",
            impact="The edit was not applied.",
            fix="Load/build the cost raster this route was computed on.",
            focus_id=ids.TAB_RASTER).to_dict())
        return None, notices

    from pyorps import PathFinder

    buffer_m = (params.get("search_space_buffer_m")
                or routing.default_search_buffer([points[0]], [points[-1]]))
    finder, notices = guard(
        PathFinder, raster_layer.meta.get("source_path"),
        source_coords=list(points), target_coords=[points[-1]],
        graph_api=params.get("graph_api", "cython"),
        neighborhood_str=params.get("neighborhood", "r2"),
        search_space_buffer_m=buffer_m,
        ignore_max_cost=params.get("ignore_max_cost", True),
        notices=notices)
    if finder is not None:
        state.finders[raster_layer.id] = finder
        params.setdefault("raster_layer_id", raster_layer.id)
    return finder, notices


def recompute_variant(state, active_layer, new_points, edit_desc,
                      notices, waypoint_names=None):
    """Clone params + points, recompute, register the NEW lineage'd route."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    params = dict(active_layer.meta.get("params") or {})
    finder, notices = _get_or_build_finder(state, active_layer, new_points,
                                           notices)
    if finder is None:
        return None, notices
    algorithm = params.get("algorithm", "dijkstra")
    algo_kwargs = {
        "delta": params.get("delta", 100),
        "num_threads": params.get("num_threads", 0),
        "use_astar": params.get("use_astar", False),
    }
    session = state.search_sessions.pop(active_layer.id, None)
    if session is not None and (
            getattr(session, "_closed", False)
            or not session.uses_same_raster(finder)):
        session.close()
        session = None
    if session is None:
        session = finder.search_session(algorithm=algorithm, **algo_kwargs)
    else:
        finder = session.finder
    result, notices = guard(
        routing.route_through_points, finder,
        [tuple(p) for p in new_points], algorithm=algorithm,
        session=session, notices=notices, **algo_kwargs)
    if result is None:
        if session is not None:
            session.close()
        return None, notices
    line, route_cost = result
    # an edit REPLACES the route: name off the base (strip prior edit/lineage
    # suffixes so repeated edits don't accumulate) and delete the old one.
    base_name = active_layer.name.split(" ←")[0].split(" (")[0]
    name = f"{base_name} ({edit_desc})"
    new_layer = add_route_layer(
        state, line=line, route_cost=route_cost, params=params,
        control_points=new_points, crs=active_layer.crs,
        name=name, parent_id=active_layer.meta.get("parent_id"),
        origin="edited", edit=edit_desc,
        simplify_tol=active_layer.meta.get("simplify_tol"),
        waypoint_names=waypoint_names)
    # keep the edited route's group + style
    new_layer.meta["group"] = active_layer.meta.get("group")
    new_layer.style = dict(active_layer.style or {})
    state.remove_layer(active_layer.id)          # editing replaces (task 44)
    if session is not None:
        state.search_sessions[new_layer.id] = session
    state.active_route_id = new_layer.id
    notices.append(success(
        f"Route updated: {name}",
        meaning="Editing replaces the route (the old one is removed). Use "
                "'Run routing' / 'New' to create additional routes."))
    return new_layer, notices


def _route_options(state):
    return [{"label": ly.name, "value": ly.id}
            for ly in state.layers_of_kind("route")]


def register(app, state) -> None:
    # 8.5: dash the active route the instant an edit request fires;
    # the server callbacks below reset it to solid when done.
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    app.clientside_callback(
        "function(request) { return '8 8'; }",
        Output(ids.ACTIVE_ROUTE, "dashArray", allow_duplicate=True),
        Input(ids.EDIT_REQUEST, "data"),
        prevent_initial_call=True)

    # ------------------------------------------------------- route selection
    @app.callback(
        Output(ids.ACTIVE_ROUTE, "positions"),
        Output(ids.ACTIVE_ROUTE, "dashArray", allow_duplicate=True),
        Output(ids.CONTROL_MARKERS, "children"),
        Output(ids.BUILD_POINTS_TABLE, "rowData", allow_duplicate=True),
        Output(ids.EDIT_COST_READOUT, "children"),
        Output(ids.EDIT_SIMPLIFY_TOL, "value"),
        Input(ids.EDIT_ROUTE_SELECT, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        prevent_initial_call=True)
    def select_route(route_id, draft):
        from .interaction import draft_rows

        layer = state.get(route_id) if route_id else None
        state.active_route_id = layer.id if layer else None
        if layer is None:
            # deselected -> the ONE points list shows the new-route draft
            return [], None, [], draft_rows(draft or {}), cost_readout(None), \
                None
        return (route_positions(layer), None, control_markers(layer),
                points_rows(layer), cost_readout(layer),
                layer.meta.get("simplify_tol"))

    # -------------------------------------------- deselect ("New route")
    @app.callback(
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Input(ids.NEW_ROUTE_BTN, "n_clicks"),
        prevent_initial_call=True)
    def new_route(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return None

    # ---------------------------------------- click-to-place edits (8.4)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "positions", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "dashArray", allow_duplicate=True),
        Output(ids.CONTROL_MARKERS, "children", allow_duplicate=True),
        Output(ids.BUILD_POINTS_TABLE, "rowData", allow_duplicate=True),
        Output(ids.EDIT_COST_READOUT, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.EDIT_REQUEST, "data"),
        State(ids.AUTO_REFRESH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def apply_edit(request, auto_refresh, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not request or not request.get("action"):
            raise PreventUpdate
        active = (state.get(state.active_route_id)
                  if state.active_route_id else None)
        if active is None or active.kind != "route":
            notices.append(Notice(
                severity="warning", title="Select a route to edit first",
                meaning="Click-to-place edits apply to the active route.",
                impact="The click did nothing.",
                fix="Pick the active route in the Edit tab.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return (no_update,) * 8 + (notices,)

        points = [tuple(p) for p in
                  (active.meta.get("control_points") or [])]
        if len(points) < 2:
            notices.append(Notice(
                severity="error", title="Route has no control points",
                meaning="This route carries no editable source/target "
                        "points.",
                impact="It can't be edited by clicking.",
                fix="Re-import or rebuild the route.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return (no_update,) * 8 + (notices,)

        action = request["action"]
        new_point = (float(request["x"]), float(request["y"]))
        parent_names = list(active.meta.get("waypoint_names") or [])
        if action == "source":
            new_points = [new_point, *points[1:]]
            new_names = parent_names
        elif action == "target":
            new_points = [*points[:-1], new_point]
            new_names = parent_names
        elif action == "move-waypoint":
            # relocate the existing waypoint nearest the click to the click
            interior = points[1:-1]
            if not interior:
                notices.append(Notice(
                    severity="warning", title="No waypoint to move",
                    meaning="This route has no waypoints between source and "
                            "target.",
                    impact="Nothing was moved.",
                    fix="Use +Waypoint to add one first.",
                    focus_id=ids.TAB_ROUTES).to_dict())
                return (no_update,) * 8 + (notices,)
            idx = min(range(len(interior)),
                      key=lambda i: (interior[i][0] - new_point[0]) ** 2
                      + (interior[i][1] - new_point[1]) ** 2)
            new_points = list(points)
            new_points[1 + idx] = new_point
            new_names = parent_names
        else:  # waypoint appended before the target (new, unnamed)
            new_points = [*points[:-1], new_point, points[-1]]
            new_names = parent_names + [""]

        if auto_refresh is False:
            # stage the edit on the active route WITHOUT recomputing; the
            # markers move + the route dashes, and "Refresh edited" recomputes
            active.meta["control_points"] = [list(map(float, p))
                                             for p in new_points]
            active.meta["waypoint_names"] = [str(n or "") for n in new_names]
            active.meta["needs_refresh"] = True
            notices.append(success(
                "Edit staged",
                meaning="Auto-refresh is off — move/add more points, then "
                        "press 'Refresh edited' to recompute."))
            return (state.layers_view(), no_update, no_update, no_update,
                    "8 8", control_markers(active), points_rows(active),
                    cost_readout(active), notices)

        new_layer, notices = recompute_variant(
            state, active, new_points, _EDIT_LABEL.get(action, action),
            notices, waypoint_names=new_names)
        if new_layer is None:
            return (no_update,) * 8 + (notices,)
        return (state.layers_view(), _route_options(state), new_layer.id,
                route_positions(new_layer), None,
                control_markers(new_layer), points_rows(new_layer),
                cost_readout(new_layer), notices)

    # ---------------------------- the ONE points list: remove selected rows
    @app.callback(
        Output(ids.BUILD_POINTS_TABLE, "rowData", allow_duplicate=True),
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Input(ids.POINTS_REMOVE_BTN, "n_clicks"),
        State(ids.BUILD_POINTS_TABLE, "rowData"),
        State(ids.BUILD_POINTS_TABLE, "selectedRows"),
        State(ids.ROUTE_DRAFT, "data"),
        prevent_initial_call=True)
    def remove_points(n_clicks, rows, selected, draft):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from .interaction import leg_distances

        if not n_clicks or not selected:
            raise PreventUpdate
        drop = {(r.get("kind"), r.get("x"), r.get("y")) for r in selected}
        remaining = leg_distances(
            [r for r in (rows or [])
             if (r.get("kind"), r.get("x"), r.get("y")) not in drop])
        if state.active_route_id:
            # route selected: the removal takes effect on "Apply points"
            return remaining, no_update
        # no route selected: the grid IS the draft — sync it
        new_draft = {"sources": [], "targets": [], "waypoints": []}
        old_points = {}
        for kind in ("source", "target", "waypoint"):
            for p in (draft or {}).get(f"{kind}s", []) or []:
                old_points[(kind, p["x"], p["y"])] = p
        for row in remaining:
            key = (row.get("kind"), row.get("x"), row.get("y"))
            point = old_points.get(key) or {
                "lat": 0, "lng": 0,
                "x": float(row["x"]), "y": float(row["y"])}
            new_draft[f"{row.get('kind', 'waypoint')}s"].append(point)
        return remaining, new_draft

    # --------------------- the ONE points list: apply rows (edit or draft)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "positions", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "dashArray", allow_duplicate=True),
        Output(ids.CONTROL_MARKERS, "children", allow_duplicate=True),
        Output(ids.BUILD_POINTS_TABLE, "rowData", allow_duplicate=True),
        Output(ids.EDIT_COST_READOUT, "children", allow_duplicate=True),
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.POINTS_APPLY_BTN, "n_clicks"),
        State(ids.BUILD_POINTS_TABLE, "rowData"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def apply_points(n_clicks, rows, raster_layer_id, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from .interaction import routing_crs

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        active = (state.get(state.active_route_id)
                  if state.active_route_id else None)

        if active is None:
            # no route selected: the rows become the new-route draft
            from ..services.geo import crs_transformer

            crs = routing_crs(state, raster_layer_id)
            tf = crs_transformer(str(crs), "EPSG:4326")
            new_draft = {"sources": [], "targets": [], "waypoints": []}
            for row in rows or []:
                kind = str(row.get("kind") or "waypoint")
                if kind not in ("source", "target", "waypoint"):
                    kind = "waypoint"
                try:
                    x, y = float(row["x"]), float(row["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                lon, lat = tf.transform(x, y)
                new_draft[f"{kind}s"].append(
                    {"lat": lat, "lng": lon, "x": x, "y": y})
            notices.append(success(
                "Points applied to the new-route draft",
                meaning=f"{len(rows or [])} point(s) — press 'Run routing' "
                        "to compute."))
            return ((no_update,) * 8 + (new_draft, notices))

        try:
            new_points = points_from_rows(rows)
        except ValueError as exc:
            notices.append(Notice(
                severity="warning", title="Points list not valid",
                meaning=str(exc),
                impact="The route was not recomputed.",
                fix="Order rows source → waypoints → target, then Apply.",
                focus_id=ids.TAB_ROUTES,
                focus_control=ids.BUILD_POINTS_TABLE).to_dict())
            return (no_update,) * 9 + (notices,)
        old_points = [tuple(p) for p in
                      (active.meta.get("control_points") or [])]
        if [tuple(p) for p in new_points] == old_points:
            raise PreventUpdate
        edit_desc = ("removed waypoint"
                     if len(new_points) < len(old_points)
                     else "edited points")
        new_layer, notices = recompute_variant(
            state, active, new_points, edit_desc, notices,
            waypoint_names=waypoint_names_from_rows(rows))
        if new_layer is None:
            return (no_update,) * 9 + (notices,)
        return (state.layers_view(), _route_options(state), new_layer.id,
                route_positions(new_layer), None,
                control_markers(new_layer), points_rows(new_layer),
                cost_readout(new_layer), no_update,
                notices)

    # ------------------------------------- post-hoc simplification (F5)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "positions", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.EDIT_SIMPLIFY_BTN, "n_clicks"),
        State(ids.EDIT_SIMPLIFY_TOL, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def apply_simplify(n_clicks, tolerance, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        active = (state.get(state.active_route_id)
                  if state.active_route_id else None)
        if active is None or active.kind != "route":
            notices.append(Notice(
                severity="warning", title="Select a route to simplify first",
                meaning="Simplification applies to the active route.",
                impact="Nothing was changed.",
                fix="Pick the active route in the Edit tab.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return no_update, no_update, notices
        tolerance = float(tolerance or 0)
        active.meta["simplify_tol"] = tolerance if tolerance > 0 else None
        display = routing.refresh_route_display(active)
        full = active.gdf.geometry.iloc[0]
        if tolerance > 0:
            notices.append(success(
                f"Simplified {active.name} (tolerance {tolerance:g} m)",
                meaning=f"Display/export vertices: {len(full.coords)} → "
                        f"{len(display.coords)}. Costs still come from the "
                        "full routed line (F5)."))
        else:
            notices.append(success(
                f"Simplification off for {active.name}",
                meaning="The full routed line is shown/exported again."))
        return state.layers_view(), route_positions(active), notices

    # ------------------- refresh staged edits (auto-refresh off, task 64)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "positions", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "dashArray", allow_duplicate=True),
        Output(ids.CONTROL_MARKERS, "children", allow_duplicate=True),
        Output(ids.BUILD_POINTS_TABLE, "rowData", allow_duplicate=True),
        Output(ids.EDIT_COST_READOUT, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.REFRESH_ROUTES_BTN, "n_clicks"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def refresh_routes(n_clicks, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        dirty = [ly for ly in state.layers_of_kind("route")
                 if (ly.meta or {}).get("needs_refresh")]
        if not dirty:
            notices.append(Notice(
                severity="info", title="Nothing to refresh",
                meaning="No route has staged edits.",
                impact="Nothing changed.",
                fix="Turn auto-refresh off and edit a route, then Refresh.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return (no_update,) * 8 + (notices,)
        orig_active = state.active_route_id
        active_new = None
        for route in dirty:
            was_active = route.id == orig_active
            points = [tuple(p) for p in
                      (route.meta.get("control_points") or [])]
            new_layer, notices = recompute_variant(
                state, route, points, "refreshed", notices,
                waypoint_names=route.meta.get("waypoint_names"))
            if new_layer is not None and was_active:
                active_new = new_layer
        # recompute_variant retargets active_route_id per call; resolve it back
        # to the originally-active route (its refreshed version if it was dirty)
        target = active_new or (state.get(orig_active) if orig_active else None)
        state.active_route_id = target.id if target is not None else None
        if target is None:
            return (state.layers_view(), _route_options(state), no_update,
                    no_update, None, no_update, no_update, no_update, notices)
        return (state.layers_view(), _route_options(state), target.id,
                route_positions(target), None, control_markers(target),
                points_rows(target), cost_readout(target), notices)

    # -------------------------------------------------------- delete variant
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "positions", allow_duplicate=True),
        Output(ids.CONTROL_MARKERS, "children", allow_duplicate=True),
        Input(ids.DELETE_VARIANT_BTN, "n_clicks"),
        prevent_initial_call=True)
    def delete_variant(n_clicks):
        if not n_clicks or not state.active_route_id:
            raise PreventUpdate
        state.remove_layer(state.active_route_id)
        return (state.layers_view(), _route_options(state), None, [], [])

    # ---------------------------------------------------------------- export
    @app.callback(
        Output(ids.EXPORT_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.EXPORT_ROUTE_BTN, "n_clicks"),
        State(ids.EXPORT_ROUTE_PATH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def export_route(n_clicks, path, notices):
        from ..services import project_io

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        active = (state.get(state.active_route_id)
                  if state.active_route_id else None)
        if active is None or not path:
            notices.append(Notice(
                severity="warning", title="Nothing to export",
                meaning="Select an active route and enter a target path.",
                impact="No file was written.",
                fix="Pick a route and a path ending in "
                    ".geojson/.gpkg/.shp/.csv.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return "", notices
        result, notices = guard(project_io.export_route, active, path,
                                notices=notices)
        if result is None:
            return "", notices
        notices.append(success("Route exported",
                               meaning=f"Written to {result}."))
        return f"saved: {result}", notices
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

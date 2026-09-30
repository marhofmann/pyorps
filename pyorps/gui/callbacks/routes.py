"""
PYORPS GUI callbacks: the route builder (R7/R8/R9, Sections 10.6 + 17.6).

Waypoints are chained ``source -> w1 -> ... -> target`` per pair (C13); the
backend/algorithm matrix is enforced (C14); the search buffer is never blank
(F1); simplify runs GUI-side with costs from the un-simplified line (F5); A*
stays off by default (F8). Every built route is an immutable layer carrying
its parameters and control points (Section 18 lineage starts here).
"""
from __future__ import annotations

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate
from shapely.geometry import LineString

import geopandas as gpd

from .. import ids
from ..services import data_io, geo, routing, validate
from ..services.errors import Notice, guard, success

_EMPTY_DRAFT = {"sources": [], "targets": [], "waypoints": []}

#: mitigation tips shown in the heavy-run confirm dialog
_MITIGATION_TIPS = (
    ("Search buffer", "set a smaller explicit 'Search buffer m' (Algorithm "
                      "& search) — the window around the points is the main "
                      "cost driver"),
    ("Resolution", "rebuild the cost raster at a coarser resolution "
                   "(Raster tab, e.g. 2-5 m instead of 1 m)"),
    ("Neighborhood", "drop from r3/r2 to r1 — halves/quarters the edges "
                     "(slightly more angular routes)"),
    ("Hardware", "switch Hardware to GPU (fastest for large windows, "
                 "opt-in)"),
    ("Fewer pairs", "route fewer source/target combinations at once, or "
                    "use 'Pairwise' instead of the full cross product"),
    ("Study area", "draw a smaller study area and rebuild the raster"),
)


def _estimate_body(est: dict) -> list:
    """The confirm-modal body: prognosis numbers + how to shrink them."""
    import dash_bootstrap_components as dbc
    from dash import html

    minutes = est["est_seconds"] / 60.0
    runtime = (f"~{est['est_seconds']:,.0f} s" if minutes < 2
               else f"~{minutes:,.1f} min")
    facts = html.Ul([
        html.Li([html.B("Prognosed runtime: "), f"{runtime} ",
                 html.Small(f"({est['algorithm']} on {est['graph_api']}, " +
                            f"rough estimate)", className="text-muted")]),
        html.Li([html.B("Prognosed memory: "),
                 f"~{est['est_memory_mb']:,.0f} MB peak (graph + search " +
                 "arrays)"]),
        html.Li([html.B("Prognosed temp storage: "),
                 f"~{est['est_storage_mb']:,.0f} MB (search-window " +
                 "raster)"]),
        html.Li([html.B("Workload: "),
                 f"{est['n_pairs']} route pair(s), {est['n_segments']} " +
                 f"segment(s), ~{est['total_cells']:,.0f} cells at " +
                 f"{est['resolution_m']:g} m ({est['neighborhood']}, search " +
                 f"buffer {est['buffer_m']:,.0f} m)"]),
    ], className="mb-2 small")
    tips = html.Ul([html.Li([html.B(f"{title}: "), text])
                    for title, text in _MITIGATION_TIPS],
                   className="mb-0 small")
    return [
        html.P("Accept waiting, or cancel and reduce the effort first. " +
               "A running job can be interrupted any time with ⏹ Stop.",
               className="small"),
        facts,
        dbc.Alert([html.B("How to make it lighter:"), tips],
                  color="info", className="py-2 px-3 mb-0"),
    ]


def finalize_routing(state, finder, built, failed, *, meta: dict,
                     notices: list, cancelled: bool = False):
    """Turn a finished routing computation into layers/notices/status.

    Shared by the synchronous path and the background-job poller — the job
    thread never touches state; all mutations happen here on the Dash side.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    raster_layer_id = meta.get("raster_layer_id")
    raster_layer = state.get(raster_layer_id)
    if finder is not None and raster_layer_id:
        state.finders[raster_layer_id] = finder

    group_label = (f"Run {next(state.run_counter)} "
                   f"({meta.get('n_sources', 1)}x{meta.get('n_targets', 1)})")
    for route in built:
        layer = add_route_layer(
            state, line=route.line, route_cost=route.cost,
            params={**route.params, "raster_layer_id": raster_layer_id},
            control_points=route.control_points,
            crs=(raster_layer.crs if raster_layer is not None
                 else state.project_crs) or state.project_crs,
            simplify_tol=meta.get("simplify_tol"),
            waypoint_names=meta.get("wp_names") or [])
        layer.meta["group"] = group_label
        if getattr(route, "session", None) is not None:
            state.search_sessions[layer.id] = route.session
    for src_pt, tgt_pt, message in failed:
        notices.append(Notice(
            severity="error", title="No route could be found",
            meaning=f"Pair ({src_pt[0]:,.0f}, {src_pt[1]:,.0f}) → "
                    f"({tgt_pt[0]:,.0f}, {tgt_pt[1]:,.0f}) failed.",
            impact="This pair produced no route (others may have).",
            fix="Increase the search buffer or move the failing point.",
            details=message, focus_id=ids.TAB_ROUTES).to_dict())
    if cancelled:
        notices.append(Notice(
            severity="warning", title="Routing stopped by you",
            meaning=f"{len(built)} completed route(s) were kept; the "
                    "remaining pairs were not computed.",
            impact="The run is incomplete.",
            fix="Run again (completed pairs re-route quickly) or reduce "
                "the effort first.").to_dict())
    elif built:
        graph_api = built[0].params.get("graph_api", "?")
        notices.append(success(
            f"{len(built)} route(s) computed",
            meaning=f"{meta.get('algorithm')} on {graph_api}, chained "
                    f"through {meta.get('n_waypoints', 0)} waypoint(s)."))
    status = (f"{len(built)} route(s), {len(failed)} failed"
              if failed else f"{len(built)} route(s) built")
    if cancelled:
        status = f"stopped — kept {len(built)} route(s)"
    route_options = [{"label": ly.name, "value": ly.id}
                     for ly in state.layers_of_kind("route")]
    newest = (state.layers_of_kind("route")[-1].id
              if built and state.layers_of_kind("route") else no_update)
    if built:
        state.active_route_id = newest
    # a fully successful run CONSUMES its points: they leave the New-routing
    # list (they live on as the group's control points and can be re-added
    # with '↩ Group points → New routing'). Failed/stopped runs keep the
    # draft so the user can retry the missing pairs.
    draft = (dict(_EMPTY_DRAFT) if built and not failed and not cancelled
             else no_update)
    return state.layers_view(), status, route_options, newest, notices, draft


def add_route_layer(state, *, line: LineString, route_cost, params: dict,
                    control_points: list, crs, name: str | None = None,
                    parent_id: str | None = None, origin: str = "built",
                    edit: str = "", simplify_tol: float | None = None,
                    waypoint_names: list | None = None):
    """Register one immutable route layer with lineage metadata (Section 18).

    The layer's gdf always holds the FULL routed line (metrics come from it,
    F5); ``simplify_tol`` only drives the display/export geometry and can be
    changed later from the Edit tab (post-hoc simplification).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    metrics = {
        "total_length_m": route_cost.total_length,
        "total_cost": route_cost.total_cost,
        "total_cell_cost": route_cost.total_cell_cost,
        "geodesic_length_m": route_cost.geodesic_length_m,
        "n_forbidden_cells": route_cost.n_forbidden_cells,
        "crosses_forbidden": route_cost.crosses_forbidden,
    }
    name = name or state.next_route_name()
    gdf = gpd.GeoDataFrame(
        [{"name": name, **metrics}], geometry=[line], crs=crs)
    layer = state.add_layer(
        name, "route", gdf=gdf, crs=crs,
        meta={
            "route_id": None,  # filled below with the layer id
            "parent_id": parent_id, "origin": origin, "edit": edit,
            "params": dict(params or {}),
            "control_points": [list(map(float, p))
                               for p in control_points],
            "waypoint_names": [str(n or "") for n in (waypoint_names or [])],
            "metrics": metrics,
            "simplify_tol": simplify_tol,
        })
    layer.meta["route_id"] = layer.id
    routing.refresh_route_display(layer)
    return layer


def register(app, state) -> None:
    # ------------------------------------ algorithm options follow hardware
    @app.callback(
        Output(ids.ALGORITHM, "options"),
        Output(ids.ALGORITHM, "value"),
        Input(ids.HARDWARE, "value"))
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    def algorithms_for_hardware(hardware):
        algorithms = routing.valid_algorithms(hardware or "cpu")
        options = [{"label": a.replace("_", " "), "value": a}
                   for a in algorithms]
        default = ("delta-stepping" if "delta-stepping" in algorithms
                   else algorithms[0])
        return options, default

    # ----------------------------------------------- search-buffer readout
    @app.callback(
        Output(ids.SEARCH_BUFFER_INFO, "children"),
        Input(ids.SEARCH_BUFFER, "value"),
        Input(ids.ROUTE_DRAFT, "data"))
    def buffer_info(buffer_m, draft):
        draft = draft or {}
        sources = [(p["x"], p["y"]) for p in draft.get("sources", [])]
        targets = [(p["x"], p["y"]) for p in draft.get("targets", [])]
        if buffer_m and float(buffer_m) > 0:
            return f"routing window: points buffered by {buffer_m} m"
        if sources and targets:
            auto = routing.default_search_buffer(sources, targets)
            return (f"blank → auto {auto:,.0f} m "
                    "(max(1000, 1.5 x distance) — F1)")
        return ("blank = max(1000 m, 1.5 x distance) — never the whole "
                "raster (F1)")

    # ------------------------------------------------------- clear the draft
    @app.callback(
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Input(ids.BUILD_CLEAR_BTN, "n_clicks"),
        prevent_initial_call=True)
    def clear_draft(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return dict(_EMPTY_DRAFT)

    # -------------------------------------------------------------- RUN
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.ROUTING_STATUS, "children"),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.ROUTING_CONFIRM_MODAL, "is_open"),
        Output(ids.ROUTING_CONFIRM_BODY, "children"),
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Input(ids.RUN_ROUTING_BTN, "n_clicks"),
        State(ids.OHL_ENABLE, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.ALGORITHM, "value"),
        State(ids.HARDWARE, "value"),
        State(ids.NEIGHBORHOOD, "value"),
        State(ids.SEARCH_BUFFER, "value"),
        State(ids.IGNORE_MAX_COST, "value"),
        State(ids.PAIRWISE, "value"),
        State(ids.SIMPLIFY, "value"),
        State(ids.SIMPLIFY_TOL, "value"),
        State("route-delta", "value"),
        State("route-num-threads", "value"),
        State("route-use-astar", "value"),
        State(ids.BUILD_POINTS_TABLE, "rowData"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def run(n_clicks, constrained_on, draft, raster_layer_id, algorithm,
            hardware, neighborhood, search_buffer, ignore_max_cost, pairwise,
            simplify, simplify_tol, delta, num_threads, use_astar,
            points_rows_data, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks or constrained_on:
            raise PreventUpdate          # constrained runner handles it

        if state.routing_job is not None:
            return (no_update, "a routing job is already running — press "
                    "⏹ Stop first", no_update, no_update, notices,
                    False, no_update, no_update)

        raster_layer = state.get(raster_layer_id) if raster_layer_id else None
        if raster_layer is None or raster_layer.kind != "raster":
            notices.append(Notice(
                severity="warning", title="Pick a cost raster first",
                meaning="Routing needs a loaded/built raster layer.",
                impact="No route was computed.",
                fix="Raster tab → build or load a raster, then select it "
                    "under 'Route on raster'.",
                focus_id=ids.TAB_RASTER,
                focus_control=ids.ROUTE_RASTER).to_dict())
            return (no_update, "no raster", no_update, no_update, notices,
                    False, no_update, no_update)

        draft = draft or {}
        sources = [(p["x"], p["y"]) for p in draft.get("sources", [])]
        targets = [(p["x"], p["y"]) for p in draft.get("targets", [])]
        waypoints = [(p["x"], p["y"]) for p in draft.get("waypoints", [])]
        if not sources or not targets:
            notices.append(Notice(
                severity="warning", title="Set a source and a target first",
                meaning="One or both routing points are missing.",
                impact="There is nothing to route yet.",
                fix="Use the +Source / +Target click modes on the map.",
                focus_id=ids.TAB_ROUTES,
                focus_control=ids.BUILD_MODE).to_dict())
            return (no_update, "missing points", no_update, no_update,
                    notices, False, no_update, no_update)

        raster_path = raster_layer.meta.get("source_path")
        # pre-flight: all points inside the raster bounds
        import rasterio
        with rasterio.open(raster_path) as src:
            bounds = (src.bounds.left, src.bounds.bottom,
                      src.bounds.right, src.bounds.top)
        problem = validate.validate_points_in_bounds(
            [*sources, *targets, *waypoints], bounds, "routing point")
        if problem is not None:
            notices.append(problem.to_dict())
            return (no_update, "points outside raster", no_update,
                    no_update, notices, False, no_update, no_update)

        run_kwargs = dict(
            sources=sources, targets=targets, waypoints=waypoints,
            algorithm=algorithm or "delta-stepping",
            hardware=hardware or "cpu",
            neighborhood=neighborhood or "r2",
            pairwise=bool(pairwise),
            search_buffer_m=(float(search_buffer)
                             if search_buffer else None),
            ignore_max_cost=bool(ignore_max_cost),
            delta=float(delta or 100), num_threads=int(num_threads or 0),
            use_astar=bool(use_astar))
        meta = {
            "raster_layer_id": raster_layer_id,
            "algorithm": run_kwargs["algorithm"],
            "simplify_tol": float(simplify_tol or 1.0) if simplify else None,
            "wp_names": [str(r.get("name", ""))
                         for r in (points_rows_data or [])
                         if r.get("kind") == "waypoint"],
            "n_sources": len(sources), "n_targets": len(targets),
            "n_waypoints": len(waypoints),
        }

        # ------------- heavy-run gate: warn + require "accept" up front
        try:
            est = routing.estimate_routing(
                raster_path, sources=sources, targets=targets,
                waypoints=waypoints, algorithm=run_kwargs["algorithm"],
                hardware=run_kwargs["hardware"],
                neighborhood=run_kwargs["neighborhood"],
                pairwise=run_kwargs["pairwise"],
                search_buffer_m=run_kwargs["search_buffer_m"])
        except Exception:                       # estimate is best-effort
            est = None
        if est is not None and routing.needs_confirmation(est):
            state.pending_routing = {"raster_path": raster_path,
                                     "kwargs": run_kwargs, "meta": meta,
                                     "estimate": est}
            return (no_update, "waiting for your go-ahead…", no_update,
                    no_update, notices, True, _estimate_body(est),
                    no_update)

        # ------------- light run: synchronous, exactly as before
        result, notices = guard(
            routing.run_routing, raster_path, **run_kwargs, notices=notices)
        if result is None:
            return (no_update, "routing failed", no_update, no_update,
                    notices, False, no_update, no_update)
        finder, built, failed = result
        view, status, options, newest, notices, draft = finalize_routing(
            state, finder, built, failed, meta=meta, notices=notices)
        return (view, status, options, newest, notices, False, no_update,
                draft)

    # --------------------------- heavy-run confirmation + background job
    @app.callback(
        Output(ids.ROUTING_CONFIRM_MODAL, "is_open", allow_duplicate=True),
        Output(ids.ROUTING_STATUS, "children", allow_duplicate=True),
        Output(ids.ROUTING_POLL, "disabled"),
        Output(ids.ROUTING_STOP_BTN, "style"),
        Input(ids.ROUTING_CONFIRM_OK, "n_clicks"),
        prevent_initial_call=True)
    def accept_heavy_run(n_clicks):
        """'Accept & run': start the confirmed run in a background thread —
        the poll interval reports progress and the Stop button can end it."""
        from ..services import routing_job

        if not n_clicks or state.pending_routing is None:
            raise PreventUpdate
        pending, state.pending_routing = state.pending_routing, None
        job = routing_job.start_routing_job(state, pending)
        return False, job.progress_text(), False, {}

    @app.callback(
        Output(ids.ROUTING_CONFIRM_MODAL, "is_open", allow_duplicate=True),
        Output(ids.ROUTING_STATUS, "children", allow_duplicate=True),
        Input(ids.ROUTING_CONFIRM_CANCEL, "n_clicks"),
        prevent_initial_call=True)
    def cancel_heavy_run(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        state.pending_routing = None
        return False, "run cancelled — reduce the effort and try again"

    @app.callback(
        Output(ids.ROUTING_STATUS, "children", allow_duplicate=True),
        Input(ids.ROUTING_STOP_BTN, "n_clicks"),
        prevent_initial_call=True)
    def stop_routing(n_clicks):
        """The user interrupts the running job: cooperative cancel — the
        segment in flight finishes, everything after it is skipped."""
        job = state.routing_job
        if not n_clicks or job is None:
            raise PreventUpdate
        job.cancel.set()
        return "stopping — finishing the segment in flight…"

    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.ROUTING_STATUS, "children", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.ROUTING_POLL, "disabled", allow_duplicate=True),
        Output(ids.ROUTING_STOP_BTN, "style", allow_duplicate=True),
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Input(ids.ROUTING_POLL, "n_intervals"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def poll_routing_job(_n, notices):
        """Progress readout while the job runs; layer creation + notices on
        completion (the worker thread never mutates state itself)."""
        hide = {"display": "none"}
        job = state.routing_job
        if job is None:                     # stale tick — switch polling off
            return (no_update, no_update, no_update, no_update, no_update,
                    True, hide, no_update)
        if job.status == "running":
            return (no_update, job.progress_text(), no_update, no_update,
                    no_update, no_update, no_update, no_update)
        notices = list(notices or [])
        state.routing_job = None
        if job.status == "failed":
            from ..services import errors
            notices.append(errors.translate_exception(job.error).to_dict())
            return (no_update, "routing failed", no_update, no_update,
                    notices, True, hide, no_update)
        finder, built, failed = job.result
        view, status, options, newest, notices, draft = finalize_routing(
            state, finder, built, failed, meta=job.params["meta"],
            notices=notices, cancelled=(job.status == "cancelled"))
        return view, status, options, newest, notices, True, hide, draft

    # ------------------------------------------------- load existing routes
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.ROUTES_LOAD_BTN, "n_clicks"),
        State(ids.ROUTES_LOAD_PATH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def load_routes(n_clicks, path, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        problem = validate.validate_file(path, "vector")
        if problem is not None:
            notices.append(problem.to_dict())
            return no_update, no_update, notices
        gdf, notices = guard(data_io.load_local_vector, path,
                             target_crs=state.project_crs, notices=notices)
        if gdf is None:
            return no_update, no_update, notices
        n_added = 0
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.geom_type not in ("LineString",
                                                      "MultiLineString"):
                continue
            if geom.geom_type == "MultiLineString":
                geom = max(geom.geoms, key=lambda g: g.length)
            name = str(row.get("name") or "") or state.next_route_name()
            props = {k: v for k, v in row.items() if k != "geometry"}
            feature = geo.linestring_to_wgs84_feature(
                geom, gdf.crs, properties={"name": name, **props})
            control_points = [list(geom.coords[0]), list(geom.coords[-1])]
            layer = state.add_layer(
                name, "route",
                gdf=gpd.GeoDataFrame([{"name": name, **props}],
                                     geometry=[geom], crs=gdf.crs),
                crs=gdf.crs,
                geojson={"type": "FeatureCollection",
                         "features": [feature]},
                meta={"route_id": None, "parent_id": None,
                      "origin": "imported", "edit": "",
                      "params": {}, "control_points": control_points,
                      "metrics": {}})
            layer.meta["route_id"] = layer.id
            n_added += 1
        notices.append(success(f"Loaded {n_added} route(s)",
                               meaning="Imported routes are editable; "
                                       "editing spawns new variants."))
        route_options = [{"label": ly.name, "value": ly.id}
                         for ly in state.layers_of_kind("route")]
        return state.layers_view(), route_options, notices
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

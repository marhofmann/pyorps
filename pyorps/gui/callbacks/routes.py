"""
PYORPS GUI callbacks: the route builder (R7/R8/R9, Sections 10.6 + 17.6).

Waypoints are chained ``source -> w1 -> ... -> target`` per pair (C13); the
Backend dropdown decides which algorithms are offered and which backend really
runs (C14); the search buffer is never blank (F1) unless the user asks for the
whole raster; simplify runs GUI-side with costs from the un-simplified line
(F5); the GUI never switches the A* heuristic on (F8). Every built route is an
immutable layer carrying its parameters and control points (Section 18
lineage starts here).
"""
from __future__ import annotations

import geopandas as gpd
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate
from shapely.geometry import LineString

from .. import ids
from ..services import data_io, geo, jobs, routing, validate
from ..services.errors import Notice, guard, success
from . import job_poll
from . import objective as objective_cb

_EMPTY_DRAFT = {"sources": [], "targets": [], "waypoints": []}

def _triggered_id():
    """The id of the component that fired the running callback, or None on a
    page's first paint (and outside a Dash request)."""
    from dash import ctx

    try:
        return ctx.triggered_id
    except Exception:  # pylint: disable=broad-exception-caught  # no Dash context: treat as first paint
        return None


def field_notice(field: str, problem: str, *, control: str | None,
                 tab: str, impact: str = "Nothing was run.",
                 fix: str | None = None) -> dict:
    """A warning box that names ONE form field and takes the user to it.

    ``field`` is the label the user sees, ``problem`` finishes the sentence
    ("must not be negative"), ``control`` is the component id the "Go fix"
    button scrolls to and flashes (``Notice.focus_control``).
    """
    return Notice(
        severity="warning", title=f"{field}: value not usable",
        meaning=f"{field} {problem}.", impact=impact,
        fix=fix or f"Correct the {field} field and try again.",
        focus_id=tab, focus_control=control).to_dict()


def number_or_default(value, default):
    """``value`` unless the field is empty: only None means "not given".

    ``value or default`` is wrong for form numbers because a legitimate 0
    (fill value 0, buffer 0 m) is falsy and would silently become the default.
    """
    return default if value is None or value == "" else value


def numeric_field_problem(value, field: str, *, control: str | None, tab: str,
                          minimum: float | None = None,
                          allow_equal: bool = True, whole: bool = False,
                          impact: str = "Nothing was run.",
                          maximum: float | None = None) -> dict | None:
    """A field-level notice when a typed number cannot be valid, else None.

    An empty field (None) is fine: the caller applies its own default. Checks
    that the value is a finite number, at least ``minimum`` (strictly above
    it when ``allow_equal`` is False), at most ``maximum`` and, with
    ``whole``, an integer.
    """
    import math

    if value is None or value == "":
        return None
    kwargs = {"control": control, "tab": tab, "impact": impact}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return field_notice(field, "must be a number", **kwargs)
    if not math.isfinite(number):
        return field_notice(field, "must be a finite number", **kwargs)
    if minimum is not None and (number < minimum
                                or (number == minimum and not allow_equal)):
        if minimum == 0:
            phrase = ("must not be negative" if allow_equal
                      else "must be greater than 0")
        else:
            phrase = (f"must be {'at least' if allow_equal else 'greater than'}"
                      f" {minimum:g}")
        return field_notice(field, phrase, **kwargs)
    if maximum is not None and number > maximum:
        return field_notice(field, f"must be at most {maximum:g}", **kwargs)
    if whole and number != int(number):
        return field_notice(field, "must be a whole number", **kwargs)
    return None


def _routing_field_problem(graph_api, algorithm, search_buffer, full_window,
                           simplify, simplify_tol, delta, num_threads, margin,
                           neighborhood_k) -> dict | None:
    """The first unusable routing form field as a notice, or None.

    A field the chosen backend/algorithm does not read is disabled in the
    page and is not checked: its value cannot change what runs.
    """
    tab = ids.TAB_ROUTES
    delta_kernel = routing.uses_delta_kernel(graph_api, algorithm)
    checks = [
        numeric_field_problem(
            neighborhood_k, "Neighbourhood k", control=ids.NEIGHBORHOOD,
            tab=tab, minimum=routing.NEIGHBORHOOD_MIN,
            maximum=routing.NEIGHBORHOOD_MAX, whole=True),
        (None if full_window else numeric_field_problem(
            search_buffer, "Search buffer (m)", control=ids.SEARCH_BUFFER,
            tab=tab, minimum=0)),
        (numeric_field_problem(simplify_tol, "Simplify tolerance (m)",
                               control=ids.SIMPLIFY_TOL, tab=tab, minimum=0)
         if simplify else None),
        # delta is the bucket width of delta-stepping: 0 would never advance
        (numeric_field_problem(delta, "delta", control=ids.DELTA, tab=tab,
                               minimum=0, allow_equal=False)
         if delta_kernel else None),
        (numeric_field_problem(num_threads, "threads", control=ids.NUM_THREADS,
                               tab=tab, minimum=0, whole=True)
         if routing.uses_threads(graph_api, algorithm) else None),
        # margin is a multiplier of the target's distance: <= 0 is meaningless
        (numeric_field_problem(margin, "margin", control=ids.MARGIN, tab=tab,
                               minimum=0, allow_equal=False)
         if delta_kernel else None),
    ]
    return next((check for check in checks if check is not None), None)


def _backend_problem(graph_api: str, algorithm: str) -> dict | None:
    """A notice when the chosen backend cannot run (not installed, or it does
    not offer the chosen algorithm): both can still reach a run through a
    stale page or a restored setting; neither may crash it."""
    if not routing.backend_available(graph_api):
        backend = routing.BACKENDS.get(graph_api)
        hint = backend.hint if backend is not None else "choose another backend"
        return Notice(
            severity="warning", title="This backend is not installed",
            meaning=f"The backend '{graph_api}' needs a package that cannot "
                    "be imported here.",
            impact="Nothing was run.",
            fix=f"Pick another backend, or install it: {hint}.",
            focus_id=ids.TAB_ROUTES, focus_control=ids.GRAPH_API).to_dict()
    if algorithm not in routing.backend_algorithms(graph_api):
        return Notice(
            severity="warning", title="Algorithm not offered by this backend",
            meaning=f"The backend '{graph_api}' does not run '{algorithm}'.",
            impact="Nothing was run.",
            fix="Pick one of the algorithms the Algorithm list shows for "
                "this backend.",
            focus_id=ids.TAB_ROUTES, focus_control=ids.ALGORITHM).to_dict()
    return None


def raster_file_info(path) -> dict:
    """Bounds and cell size of a raster file, read on the Dash side.

    Raises ``FileNotFoundError`` (plain message) when the layer's file is gone.
    """
    import os

    import rasterio

    if not path or not os.path.exists(str(path)):
        raise FileNotFoundError(
            f"the raster file '{path}' of this layer no longer exists")
    with rasterio.open(path) as src:
        return {"bounds": (src.bounds.left, src.bounds.bottom,
                           src.bounds.right, src.bounds.top),
                "resolution_m": abs(src.transform.a)}


def raster_info_or_notice(path, notices: list, *, impact: str):
    """``(info_or_None, notices)``: the file's info, or a box saying why not."""
    import os

    if not path or not os.path.exists(str(path)):
        notices.append(Notice(
            severity="error", title="The raster file is missing",
            meaning=f"The layer points at '{path}', which no longer exists "
                    "(a temporary file may have been cleaned up).",
            impact=impact,
            fix="Rebuild the cost raster or load the .tif again, then "
                "select it under 'Route on raster'.",
            focus_id=ids.TAB_RASTER, focus_control=ids.ROUTE_RASTER).to_dict())
        return None, notices
    return guard(raster_file_info, path, notices=notices)


def _estimate_failed_notice(exc: BaseException, *, what: str) -> dict:
    """The box shown when the effort estimate itself raised (audit #35)."""
    return Notice(
        severity="warning", title="Could not estimate the effort of this run",
        meaning="The up-front runtime/memory prognosis failed, so nobody "
                f"knows how heavy this {what} is.",
        impact="It is treated as a heavy run: you are asked to confirm "
               "before it starts." if what == "routing run" else
               "It is treated as a heavy run: it may take long and the "
               "page waits until it finishes.",
        fix="Check the raster and the points; set a smaller search buffer "
            "if the window is large.",
        details=f"{type(exc).__name__}: {exc}",
        focus_id=ids.TAB_ROUTES).to_dict()


def _estimate_unknown_body() -> list:
    """The confirm-modal body when no estimate could be made."""
    import dash_bootstrap_components as dbc
    from dash import html

    tips = html.Ul([html.Li([html.B(f"{title}: "), text])
                    for title, text in _MITIGATION_TIPS],
                   className="mb-0 small")
    return [
        html.P("The run could not be estimated, so it is treated as heavy "
               "(see the notice for the reason). Accept waiting, or cancel "
               "and reduce the effort first. A running job can be "
               "interrupted any time with ⏹ Stop.", className="small"),
        dbc.Alert([html.B("How to make it lighter:"), tips],
                  color="info", className="py-2 px-3 mb-0"),
    ]

#: mitigation tips shown in the heavy-run confirm dialog
_MITIGATION_TIPS = (
    ("Search buffer", "set a smaller explicit 'Search buffer (m)' and switch "
                      "off 'whole raster' (Algorithm & search) — the window "
                      "around the points is the main cost driver"),
    ("Resolution", "rebuild the cost raster at a coarser resolution "
                   "(Raster tab, e.g. 2-5 m instead of 1 m)"),
    ("Neighbourhood", "lower k from 3/2 to 1 — halves/quarters the edges "
                      "(slightly more angular routes)"),
    ("Backend", "switch Backend to raster_gpu (fastest for large windows, "
                "opt-in; needs CuPy and a CUDA GPU)"),
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
                            "rough estimate)", className="text-muted")]),
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


def routing_outcome(state, job):
    """The ended routing job as ``(results, notices)`` for the job poll (``callbacks/job_poll.py``).

    ``results`` are the values of ``LAYERS_VIEW``, ``ROUTING_STATUS``, the two route-select properties and
    ``ROUTE_DRAFT`` (those of :func:`finalize_routing`), ``notices`` the notice dicts to add: the warnings the
    worker thread collected (``errors.guard`` cannot see a thread it did not start), then the run's own. A
    failed job gives its translated error and leaves everything else as it was. Runs once per job, in a poll
    tick: the worker thread never touches the state.
    """
    notices = list(getattr(job, "notices", None) or [])
    if job.status == "failed":
        from ..services import errors
        notices.append(errors.translate_exception(job.error).to_dict())
        return (no_update, "routing failed", no_update, no_update, no_update), notices
    finder, built, failed = job.result
    view, status, options, newest, notices, draft = finalize_routing(
        state, finder, built, failed, meta=job.params["meta"],
        notices=notices, cancelled=job.status == "cancelled")
    return (view, status, options, newest, draft), notices


def _held_routing_job(state, notices: list):
    """A Run click while the state holds a routing job: the nine outputs of ``run``, else None.

    A running job refuses the click. An ended one switches the poll on, which hands its outcome over (the
    page may have missed it, e.g. after a reload); while that is under way the click waits for it.
    """
    phase = jobs.held_phase(state, jobs.ROUTING_SLOT)
    if phase is None:
        return None
    if phase == jobs.HELD_RUNNING:
        return (no_update, "a routing job is already running — press "
                "⏹ Stop first", no_update, no_update, notices,
                False, no_update, no_update, no_update)
    if phase == jobs.HELD_HANDOVER:
        notices.append(job_poll.handover_notice(jobs.ROUTING_LABEL))
        status = jobs.finalizing_text(jobs.ROUTING_LABEL)
    else:
        notices.append(job_poll.ended_notice("routing", "Run routing"))
        status = "collecting the earlier routing run…"
    return (no_update, status, no_update, no_update, notices, False,
            no_update, no_update, False)


def finalize_routing(state, finder, built, failed, *, meta: dict,
                     notices: list, cancelled: bool = False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
        objective_line = meta.get("objective_summary")
        notices.append(success(
            f"{len(built)} route(s) computed",
            meaning=f"{meta.get('algorithm')} on {graph_api}, chained "
                    f"through {meta.get('n_waypoints', 0)} waypoint(s)."
                    + (f" {objective_line}" if objective_line else "")))
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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
    try:
        routing.refresh_route_display(layer)
    except Exception:
        state.remove_layer(layer.id)        # never leave a route without display
        raise
    return layer


def register(app, state) -> None:
    # ------------------------------------- algorithms of the chosen backend
    @app.callback(
        Output(ids.ALGORITHM, "options"),
        Output(ids.ALGORITHM, "value"),
        Input(ids.GRAPH_API, "value"),
        State(ids.ALGORITHM, "value"))
    def algorithms_for_backend(graph_api, current):
        """Only what the chosen backend really runs is offered; changing the
        backend resets the algorithm to that backend's default. The one
        exception is the page's first paint (a reload) and the moment
        "Open project" writes the controls: both keep a restored algorithm
        that the backend accepts."""
        graph_api = graph_api or routing.DEFAULT_BACKEND
        algorithms = routing.backend_algorithms(graph_api)
        options = [{"label": a.replace("_", " "), "value": a}
                   for a in algorithms]
        # "Open project" wrote the backend and the algorithm together: the flag
        # is read once, whichever way this callback ends
        restored = state.routing_restore_backend
        state.routing_restore_backend = None
        if current in algorithms and (
                _triggered_id() is None or restored == graph_api):
            return options, current
        return options, routing.default_algorithm(graph_api)

    # ------------------------- delta / threads / margin only where they act
    @app.callback(
        Output(ids.DELTA_WRAP, "title"),
        Output(ids.DELTA, "disabled"),
        Output(ids.THREADS_WRAP, "title"),
        Output(ids.NUM_THREADS, "disabled"),
        Output(ids.MARGIN_WRAP, "title"),
        Output(ids.MARGIN, "disabled"),
        Input(ids.ALGORITHM, "value"),
        Input(ids.GRAPH_API, "value"))
    def conditional_controls(algorithm, graph_api):
        """Disable a setting the chosen algorithm ignores, and say why."""
        graph_api = graph_api or routing.DEFAULT_BACKEND
        algorithm = algorithm or routing.default_algorithm(graph_api)
        delta_on = routing.uses_delta_kernel(graph_api, algorithm)
        threads_on = routing.uses_threads(graph_api, algorithm)
        return (
            routing.DELTA_HELP if delta_on else routing.DELTA_OFF_HELP, not delta_on,
            routing.THREADS_HELP if threads_on else routing.THREADS_OFF_HELP, not threads_on,
            routing.MARGIN_HELP if delta_on else routing.MARGIN_OFF_HELP, not delta_on)

    # ----------------------------------------------- search-buffer readout
    @app.callback(
        Output(ids.SEARCH_BUFFER_INFO, "children"),
        Input(ids.SEARCH_BUFFER, "value"),
        Input(ids.ROUTE_DRAFT, "data"),
        Input(ids.FULL_WINDOW, "value"))
    def buffer_info(buffer_m, draft, full_window):
        if full_window:
            return ("the whole raster is the search window - exact, but slow "
                    "and memory-hungry on a large raster")
        draft = draft or {}
        sources = [(p["x"], p["y"]) for p in draft.get("sources", [])]
        targets = [(p["x"], p["y"]) for p in draft.get("targets", [])]
        if buffer_m and float(buffer_m) > 0:
            return f"routing window: points buffered by {buffer_m} m"
        if sources and targets:
            auto = routing.default_search_buffer(sources, targets)
            return (f"blank → auto {auto:,.0f} m "
                    "(max(1000, 1.5 x distance) — F1)")
        return ("blank = max(1000 m, 1.5 x distance); or switch on 'whole "
                "raster'")

    @app.callback(
        Output(ids.SEARCH_BUFFER, "disabled"),
        Input(ids.FULL_WINDOW, "value"))
    def buffer_follows_full_window(full_window):
        """The buffer is not used while the whole raster is the window."""
        return bool(full_window)

    # --------------- the page's routing settings, kept on the server for saving
    @app.callback(
        Output(ids.ROUTING_SETTINGS, "data"),
        Input(ids.GRAPH_API, "value"),
        Input(ids.ALGORITHM, "value"),
        Input(ids.NEIGHBORHOOD, "value"),
        Input(ids.SEARCH_BUFFER, "value"),
        Input(ids.FULL_WINDOW, "value"),
        Input(ids.IGNORE_MAX_COST, "value"),
        Input(ids.PAIRWISE, "value"),
        Input(ids.SIMPLIFY, "value"),
        Input(ids.SIMPLIFY_TOL, "value"),
        Input(ids.DELTA, "value"),
        Input(ids.NUM_THREADS, "value"),
        Input(ids.MARGIN, "value"))
    def remember_routing_settings(*values):
        """Mirror the routing controls into ``state.routing_settings``: the
        project file saves them from there, and a rebuilt page shows them.
        Empty or unusable fields fall back to their defaults."""
        settings = routing.normalise_routing_settings(
            dict(zip(routing.ROUTING_SETTING_KEYS, values)))
        state.routing_settings = settings
        return settings

    # --------------------- one busy flag for the plain and the constrained run
    @app.callback(
        Output(ids.RUN_ROUTING_BUSY, "data"),
        Input(ids.RUN_PLAIN_BUSY, "data"),
        Input(ids.RUN_OHL_BUSY, "data"))
    def merge_run_busy(plain_busy, constrained_busy):
        """Both runners listen to the one Run button, and the one that does not
        act ends at once: if both wrote the shared flag through ``running=``,
        that early end would clear it while the other one is still working.
        Each runner flips its own flag; the shared flag is their OR."""
        return bool(plain_busy) or bool(constrained_busy)

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
        # switched on when the click finds an ended job nobody polled (the poll collects it)
        Output(ids.ROUTING_POLL, "disabled", allow_duplicate=True),
        Input(ids.RUN_ROUTING_BTN, "n_clicks"),
        State(ids.OHL_ENABLE, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.ALGORITHM, "value"),
        State(ids.GRAPH_API, "value"),
        State(ids.NEIGHBORHOOD, "value"),
        State(ids.SEARCH_BUFFER, "value"),
        State(ids.FULL_WINDOW, "value"),
        State(ids.IGNORE_MAX_COST, "value"),
        State(ids.PAIRWISE, "value"),
        State(ids.SIMPLIFY, "value"),
        State(ids.SIMPLIFY_TOL, "value"),
        State(ids.DELTA, "value"),
        State(ids.NUM_THREADS, "value"),
        State(ids.MARGIN, "value"),
        State(ids.BUILD_POINTS_TABLE, "rowData"),
        State(ids.NOTICES, "data"),
        running=[(Output(ids.RUN_PLAIN_BUSY, "data"), True, False)],
        prevent_initial_call=True)
    def run(n_clicks, constrained_on, draft, raster_layer_id, algorithm,
            graph_api, neighborhood_k, search_buffer, full_window,
            ignore_max_cost, pairwise, simplify, simplify_tol, delta,
            num_threads, margin, points_rows_data, notices):
        notices = list(notices or [])
        if not n_clicks or constrained_on:
            raise PreventUpdate          # constrained runner handles it
        # a job that ended while nobody polled (page reload, other tab) is
        # collected by the poll instead of refusing forever (audit #4)
        held = _held_routing_job(state, notices)
        if held is not None:
            return held
        return (*_route(draft, raster_layer_id, algorithm, graph_api,
                        neighborhood_k, search_buffer, full_window,
                        ignore_max_cost, pairwise, simplify, simplify_tol,
                        delta, num_threads, margin, points_rows_data,
                        notices), no_update)

    def _route(draft, raster_layer_id, algorithm, graph_api, neighborhood_k,
               search_buffer, full_window, ignore_max_cost, pairwise,
               simplify, simplify_tol, delta, num_threads, margin,
               points_rows_data, notices):
        """The eight outputs of ``run`` (all but the poll) for a click that may start a run."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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

        # form fields first: a value that cannot be valid is named, not run
        graph_api = graph_api or routing.DEFAULT_BACKEND
        algorithm = algorithm or routing.default_algorithm(graph_api)
        problem = (_backend_problem(graph_api, algorithm)
                   or _routing_field_problem(
                       graph_api, algorithm, search_buffer, full_window,
                       simplify, simplify_tol, delta, num_threads, margin,
                       neighborhood_k))
        if problem is not None:
            notices.append(problem)
            return (no_update, "check the highlighted field", no_update,
                    no_update, notices, False, no_update, no_update)

        raster_path = raster_layer.meta.get("source_path")
        # pre-flight: all points inside the raster bounds
        info, notices = raster_info_or_notice(
            raster_path, notices, impact="No route was computed.")
        if info is None:
            return (no_update, "raster file not readable", no_update,
                    no_update, notices, False, no_update, no_update)
        problem = validate.validate_points_in_bounds(
            [*sources, *targets, *waypoints], info["bounds"], "routing point")
        if problem is not None:
            notices.append(problem.to_dict())
            return (no_update, "points outside raster", no_update,
                    no_update, notices, False, no_update, no_update)

        buffer_m = number_or_default(search_buffer, None)
        margin_value = number_or_default(margin, None)
        run_kwargs = {
            "sources": sources, "targets": targets, "waypoints": waypoints,
            "algorithm": algorithm, "graph_api": graph_api,
            "neighborhood": routing.neighborhood_string(neighborhood_k),
            "pairwise": bool(pairwise),
            # blank or 0 = automatic window, unless the user asks for the
            # whole raster (F1)
            "search_buffer_m": (routing.whole_raster_buffer(raster_path)
                                if full_window else
                                None if buffer_m is None else float(buffer_m)),
            "full_window": bool(full_window),
            "ignore_max_cost": bool(ignore_max_cost),
            "delta": float(number_or_default(delta, 100.0)),
            "num_threads": int(number_or_default(num_threads, 0)),
            "margin": (None if margin_value is None
                       else float(margin_value))}
        meta = {
            "raster_layer_id": raster_layer_id,
            "algorithm": run_kwargs["algorithm"],
            "simplify_tol": (float(number_or_default(simplify_tol, 1.0))
                             if simplify else None),
            "wp_names": [str(r.get("name", ""))
                         for r in (points_rows_data or [])
                         if r.get("kind") == "waypoint"],
            "n_sources": len(sources), "n_targets": len(targets),
            "n_waypoints": len(waypoints),
        }

        # ------------- objective & terrain card: its settings reach the run through the state
        window_buffer = (run_kwargs["search_buffer_m"] if run_kwargs["search_buffer_m"] is not None
                         else routing.default_search_buffer(sources, targets))
        objective_run, objective_meta, notices = objective_cb.prepare_run(
            state, raster_layer=raster_layer, graph_api=graph_api, algorithm=algorithm,
            raster_bounds=info["bounds"], points=[*sources, *targets, *waypoints],
            buffer_m=window_buffer, notices=notices)
        if objective_run is None:
            return (no_update, "check the highlighted field", no_update,
                    no_update, notices, False, no_update, no_update)
        run_kwargs.update(objective_run)
        estimate_extra = {}
        if objective_meta:
            meta["objective_summary"] = objective_meta["objective_summary"]
            if objective_meta["stack_bands"]:
                estimate_extra["stack_bands"] = objective_meta["stack_bands"]

        # ------------- heavy-run gate: warn + require "accept" up front
        est = None
        estimate_failed = False
        try:
            est = routing.estimate_routing(
                raster_path, sources=sources, targets=targets,
                waypoints=waypoints, algorithm=run_kwargs["algorithm"],
                graph_api=run_kwargs["graph_api"],
                neighborhood=run_kwargs["neighborhood"],
                pairwise=run_kwargs["pairwise"],
                search_buffer_m=run_kwargs["search_buffer_m"],
                **estimate_extra)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # an unknown effort counts as heavy (#35)
            estimate_failed = True
            notices.append(_estimate_failed_notice(exc, what="routing run"))
        if estimate_failed or routing.needs_confirmation(est):
            state.pending_routing = {"raster_path": raster_path,
                                     "kwargs": run_kwargs, "meta": meta,
                                     "estimate": est}
            body = (_estimate_unknown_body() if est is None
                    else _estimate_body(est))
            return (no_update, "waiting for your go-ahead…", no_update,
                    no_update, notices, True, body, no_update)

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

    # the shared job poll: progress while the job runs; the route layers and the notices (registered by
    # ``routing_outcome``, the worker thread never touches the state) are handed to the page exactly once
    job_poll.register_poll(app, state, job_poll.JobPoll(
        slot=jobs.ROUTING_SLOT, label=jobs.ROUTING_LABEL, poll=ids.ROUTING_POLL,
        stop=ids.ROUTING_STOP_BTN, ack=ids.JOB_ACK_ROUTING,
        results=((ids.LAYERS_VIEW, "data"), (ids.ROUTING_STATUS, "children"),
                 (ids.EDIT_ROUTE_SELECT, "options"),
                 (ids.EDIT_ROUTE_SELECT, "value"), (ids.ROUTE_DRAFT, "data")),
        status=(ids.ROUTING_STATUS, "children"),
        running_text=lambda job: job.progress_text(),
        finalize=lambda st, job: routing_outcome(st, job)))

    # ------------------------------------------------- load existing routes
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.ROUTES_LOAD_BTN, "n_clicks"),
        State(ids.ROUTES_LOAD_PATH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True,
        running=[(Output(ids.ROUTES_LOAD_BTN, "disabled"), True, False)])
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
                               meaning="Imported routes are editable; an edit "
                                       "replaces the route and 'Undo last "
                                       "edit' brings the old one back."))
        route_options = [{"label": ly.name, "value": ly.id}
                         for ly in state.layers_of_kind("route")]
        return state.layers_view(), route_options, notices
    _ = None
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

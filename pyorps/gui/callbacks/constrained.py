"""
PYORPS GUI callbacks: constrained overhead-line routing (R13, Section 20).

The profile is edited as YAML (seeded from the shipped 110/220/380 kV /
rural-road presets), validated live before the run (21.5), and experimental
backends are gated behind an explicit flag (F10). Results land as a route
layer plus a towers layer (rotated-footprint polygons), with the 2-D
terrain-cost caveat (F9) always visible.

A heavy run (``constrained.is_heavy``; also a run whose effort could not be
estimated) goes to a background job (round 14, gap 7): the worker only
computes a snapshot of the run; the shared job poll (``callbacks/job_poll.py``)
registers the route through the same ``finalize_constrained`` as a synchronous
run and hands it to the page exactly once, and
Stop ends the run after the segment in flight (no route is added). A light
run stays inside ``run_constrained``, exactly as before.

Round 16 (R3-B): the objective weights of the "Objective weights" block are read from
``state.gui_settings["constrained"]["objective"]`` (``objective_settings``; the run gained no ``State``). A weight
that is not usable, or a combination the library refuses (another backend than cython, the clearance check),
is an error notice that focuses the field and starts nothing. Weights at their defaults pass nothing to the
library; others go to it through ``services/constrained`` in the synchronous and the background path alike, and
are stored with the route (``params["objective_weights"]``) and named in its status and success notice.

Round 17 (P5): the connection heights of the "Line ends" group are read the same way
(``ends_settings``; the run gained no ``State`` either), checked together with the end-span rule of the profile
(``constrained_ends.run_problems``: a value the library refuses, a backend that cannot run them, an exact search
that would not fit) before anything starts, and passed to the library through ``services/constrained`` in the
synchronous and the background path alike. They are stored with the route (``params["connection_heights"]``). What
the library reports about the line ends (the rule, the heights, a removed end tower, span and search notes, height
premiums) is named in the status, the success notice and, for the notes, a notice of its own; a ``NoPathFoundError``
of a restricted search ends as the notice of ``services/errors`` that names the rule or the heights.
"""
from __future__ import annotations

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..sections import profile_form as form_section
from ..services import constrained, constrained_ends, constrained_objective, geo, jobs, routing
from ..services import profile_form as profile_service
from ..services.errors import Notice, cap_notices, guard, success
from . import job_poll
from .routes import (
    add_route_layer,
    number_or_default,
    numeric_field_problem,
    raster_info_or_notice,
)


class _NoPath:
    """Marker for "the run finished but found no path" (as opposed to a crash); ``info`` names the segment."""

    def __init__(self, info: dict | None = None):
        self.info = info


#: the Stop button of the constrained job is hidden unless a job runs
STOP_HIDDEN = job_poll.STOP_HIDDEN
#: where the state keeps the constrained job
_CONSTRAINED_SLOT = jobs.kind_slot(constrained.JOB_KIND)
#: what the two background-job outputs of ``run_constrained`` (poll, Stop button) hold when a run does not
#: touch the job. ``RUN_OHL_BUSY`` stays a ``running=`` target of that callback alone: a click during a
#: background run is not blocked by it but reaches the job service's refusal notice.
_NO_JOB = (no_update, no_update)
_UNCHANGED = (no_update, no_update, no_update, no_update)


def no_route_notice(info: dict | None = None) -> dict:
    """The box for a finished run whose segment found no feasible tower-by-tower path.

    ``info`` (``{"segment": k, "of": n}``, what ``constrained.run_constrained_path`` leaves in its ``outcome``)
    names the segment of a chain of waypoints that found none; without it, or for a run of one segment, the
    meaning is the general sentence.
    """
    return Notice(
        severity="error", title="No route could be found",
        meaning=constrained.no_route_segment_text(info) or "A constrained segment found no feasible tower-by-tower "
                                                           "path.",
        impact="No overhead-line route was produced.",
        fix="Loosen the profile (angles/spans), enlarge the search buffer, "
            "or move the points.", focus_id=ids.TAB_ROUTES).to_dict()


def estimate_failed_notice(exc: BaseException) -> dict:
    """The box shown when the effort estimate itself raised (audit #35): the run counts as heavy."""
    return Notice(
        severity="warning", title="Could not estimate the effort of this run",
        meaning="The up-front runtime/memory prognosis failed, so nobody "
                "knows how heavy this constrained run is.",
        impact="It is treated as a heavy run: it goes to a background job "
               "and the page stays usable. " + constrained.STOP_NOTE,
        fix="Check the raster and the points; set a smaller search buffer "
            "if the window is large.",
        details=f"{type(exc).__name__}: {exc}",
        focus_id=ids.TAB_ROUTES).to_dict()


def heavy_run_notice(est: dict) -> dict:
    """The prognosis of a heavy run, as the notice that says it runs in the background."""
    return Notice(
        severity="warning", title="Heavy constrained run ahead",
        meaning=f"~{est['total_cells']:,.0f} window cells at "
                f"{est['resolution_m']:g} m; the tower state "
                "multiplies plain-routing effort "
                f"(≥ {est['est_seconds']:,.0f} s, "
                f"≥ {est['est_memory_mb']:,.0f} MB expected).",
        impact="It runs in the background, so the page stays usable; the "
               "route is added when the job has finished. "
               + constrained.STOP_NOTE,
        fix="Watch the status line under the profile; press Stop to end the "
            "run. A smaller search buffer, a coarser raster resolution, a "
            "lower neighborhood (r1) or a GPU backend reduce the effort."
    ).to_dict()


def stopped_notice() -> dict:
    """The box for a run that Stop ended: no route, nothing changed."""
    return Notice(
        severity="info", title="Constrained routing stopped",
        meaning="Stop was pressed; the run ended after the segment in "
                "flight.",
        impact="No overhead line was added (a partial tower chain is not a "
               "route) and the project is unchanged.",
        fix="Press Run routing again to start over.").to_dict()


#: why the library took an end tower out, by the rule in force (a default run says "remove")
_REMOVED_WHY = {
    "remove": "the search had placed it closer to a line end than the shortest span, and the rule removes such a "
              "tower afterwards, so the end span became longer",
    "remove_if_valid": "the search had placed it closer to a line end than the shortest span, and the line stays "
                       "valid without it",
}
_HEIGHT_WORDS = {"source_connection_height_m": "source", "target_connection_height_m": "target"}


def _short_rule(rule: str) -> str:
    """The option text of a rule without its bracket: ``"Keep it"``."""
    return constrained_ends.rule_label(rule).split(" (", 1)[0]


def _heights_text(summary: dict) -> str:
    """``"source 25 m, target 12 m"`` for the connection heights a route reports; "" when none."""
    return ", ".join(f"{_HEIGHT_WORDS[key]} {constrained_objective.format_number(summary[key])} m"
                     for key in constrained_ends.HEIGHT_KEYS if key in summary)


def end_status_text(summary: dict) -> str:
    """What the library reports about the line ends, as the parts of the status line (``"; ..."`` each).

    Only where there is something to say: the rule (when not the default), the connection heights, the removed end
    towers. A classic run whose end tower the library took out says so too (the honest state).
    """
    parts = []
    rule = summary.get("terminal_span_rule")
    if rule and rule != constrained_ends.DEFAULT_RULE:
        parts.append(f"end-span rule: {_short_rule(rule)}")
    heights = _heights_text(summary)
    if heights:
        parts.append(f"connection heights: {heights}")
    removed = summary.get("removed_towers")
    if removed:
        count = int(removed["count"])
        parts.append(f"{count} end tower{'' if count == 1 else 's'} removed")
    return "".join(f"; {part}" for part in parts)


def end_facts(summary: dict) -> list[str]:
    """The sentences of the success notice about the line ends (empty for a run that has nothing to report)."""
    out = []
    rule = summary.get("terminal_span_rule")
    if rule and rule != constrained_ends.DEFAULT_RULE:
        out.append(f"End-span rule: {constrained_ends.rule_label(rule)}.")
    heights = _heights_text(summary)
    if heights:
        out.append(f"Connection heights: {heights} (metres above the ground at the end point).")
    removed = summary.get("removed_towers")
    if removed:
        count = int(removed["count"])
        why = _REMOVED_WHY.get(rule or constrained_ends.DEFAULT_RULE,
                               "the end-span rule took it out after the search")
        out.append(f"Removed end tower{'' if count == 1 else 's'}: {count} (cost {float(removed['cost']):,.0f}, not "
                   f"part of the route's cost). Why: {why}.")
    if summary.get("height_premiums") is not None:
        out.append(f"Height premiums: {float(summary['height_premiums']):,.0f} (the part of the tower cost charged "
                   "for taller towers, already inside the tower cost).")
    return out


def end_notes_notice(summary: dict) -> dict | None:
    """The notice with the span notes and the search notes of the library, or None when it has none."""
    notes = [*summary.get("span_notes", ()), *summary.get("search_notes", ())]
    if not notes:
        return None
    return Notice(
        severity="info", title="Line ends: notes on this route",
        meaning=" ".join(notes),
        impact="The route is the one found; the notes say where an end span differs from the spans of the profile "
               "or how the search ran.",
        fix="Choose another end-span rule or set the connection heights in the Line ends group of the profile form "
            "if you want another result.", focus_id=ids.TAB_ROUTES,
        focus_control=ids.OHLE_BLOCK).to_dict()


def finalize_constrained(state, result: dict, notices: list[dict], *,
                         config: dict, backend: str, neighborhood: str,
                         raster_layer_id: str, control_points: list,
                         wp_names: list, objective_weights: dict | None = None,
                         connection_heights: dict | None = None):
    """What happens after a constrained run found a route: layers, summary, log, notices.

    The one path for the synchronous run (inside ``run_constrained``) and a background run (the
    poll callback). ``result`` is what ``constrained.run_constrained_path`` returned; ``config`` is
    the profile that was run, ``control_points`` the source, the waypoints and the target.
    ``objective_weights`` are the weights the run used (``constrained_objective.used_weights``; None
    or empty for classic routing): they are stored with the route and named in the status and the
    success notice. ``connection_heights`` are the heights the run was given
    (``constrained_ends.used_ends``; None or empty when none was set): stored with the route in its params and
    summary. What the library reports about the line ends (``end_status_text``, ``end_facts``,
    ``end_notes_notice``) is named in the status and the notices. Returns the five values of ``LAYERS_VIEW``,
    ``OHL_STATUS``, the two route-select properties and ``NOTICES``.
    """
    from ..services.cost import RouteCost

    summary = result["summary"]
    line = result["line"]
    route_cost = RouteCost(
        total_length=float(result["total_length"]),
        total_cost=float(result["total_cost"]),
        total_cell_cost=float(result["total_cell_cost"]),
        geodesic_length_m=float(line.length))
    crs = result["crs"]
    n_waypoints = len(control_points) - 2
    params = {"constrained": True, "profile": config, "backend": backend,
              "raster_layer_id": raster_layer_id,
              "neighborhood": neighborhood}
    weights_text = constrained_objective.weights_text(objective_weights)
    if objective_weights:               # a classic run keeps the params it always had
        params["objective_weights"] = dict(objective_weights)
        summary = {**summary, "objective_weights": dict(objective_weights)}
    if connection_heights:              # the heights the run was given (the summary also has the reported ones)
        params["connection_heights"] = dict(connection_heights)
        summary = {**summary, "connection_heights": dict(connection_heights)}
    route_layer = add_route_layer(
        state, line=line, route_cost=route_cost, params=params,
        control_points=control_points, crs=crs,
        name=f"OHL {state.next_route_name()}", origin="built",
        waypoint_names=wp_names)
    route_layer.meta["constrained_summary"] = summary

    if result["towers"] is not None and not result["towers"].empty:
        state.add_layer(
            f"{route_layer.name} towers", "vector", gdf=result["towers"],
            crs=crs, geojson=geo.gdf_to_wgs84_geojson(result["towers"]),
            style={"color": "#7b3294", "weight": 2, "fillOpacity": 0.5},
            meta={"towers_for": route_layer.id})

    facts = end_facts(summary)
    notices.append(success(
        f"Overhead line routed: {summary['n_towers']} towers"
        + (f" ({n_waypoints} forced at waypoints)" if n_waypoints else ""),
        meaning=(f"Tower cost {summary['total_tower_cost']:,.0f}, angle "
                 f"penalties {summary['total_angle_penalty_cost']:,.0f}, "
                 f"max turn {summary['max_turn_angle_deg']:.1f} deg."
                 + (f" Objective weights used: {weights_text}." if weights_text else "")
                 + "".join(f" {fact}" for fact in facts))))
    notices.append(Notice(
        severity="info", title="Terrain cost shown is 2-D",
        meaning="The optimizer used 3-D penalties but reports the plain "
                "2-D terrain cost (known limitation).",
        impact="The terrain-cost number understates the optimized cost "
               "(F9).",
        fix="Treat it as indicative; compare routes by length + tower "
            "cost too.").to_dict())
    notes_notice = end_notes_notice(summary)
    if notes_notice is not None:
        notices.append(notes_notice)
    status = (f"OHL: {summary['n_towers']} towers, spans "
              f"{summary['spans_min_max_avg'][0]:.0f}-"
              f"{summary['spans_min_max_avg'][1]:.0f} m "
              f"(avg {summary['spans_min_max_avg'][2]:.0f})"
              + end_status_text(summary)
              + (f"; weights: {weights_text}" if weights_text else ""))
    route_options = [{"label": ly.name, "value": ly.id}
                     for ly in state.layers_of_kind("route")]
    state.active_route_id = route_layer.id
    return (state.layers_view(), status, route_options, route_layer.id,
            notices)


def constrained_outcome(state, job):
    """The ended constrained job as ``(results, notices)`` for the job poll (``callbacks/job_poll.py``).

    ``results`` are the values of ``LAYERS_VIEW``, ``OHL_STATUS`` and the two route-select properties,
    ``notices`` the notice dicts to add. Done: the shared :func:`finalize_constrained` adds the route and
    its towers (or the "no route" notice when a segment found no path). Stopped: a notice, nothing changes
    in the project. Failed: the translated error as a notice. Runs once per job, in a poll tick.
    """
    notices: list[dict] = []
    if job.status == jobs.FAILED:
        failed = job.error_notice()
        if failed is not None:
            notices.append(failed)
        return (no_update, job.progress_text(), no_update, no_update), notices
    if job.status == jobs.CANCELLED:
        notices.append(stopped_notice())
        return (no_update, f"{job.progress_text()} No route was added.", no_update, no_update), notices
    outcome = job.result or {}
    notices.extend(outcome.get("notices") or [])
    if outcome.get("result") is None:
        notices.append(no_route_notice(outcome.get("no_route")))
        return (no_update, f"{job.progress_text()} No route was found.", no_update, no_update), notices
    view, status, options, newest, notices = finalize_constrained(
        state, outcome["result"], notices, **job.params["finalize"])
    return (view, status, options, newest), notices


def _start_background_run(state, snapshot: dict, finalize: dict,
                          est: dict | None, notices: list[dict]):
    """Start the heavy run as a background job; the seven outputs of ``run_constrained``.

    The worker gets a snapshot of the run inputs, the poll callback the ``finalize`` keywords of
    :func:`finalize_constrained`. A refused start (another job runs) answers with the job
    service's notice and changes nothing.
    """
    refusal = jobs.refusal_notice(state, constrained.JOB_KIND)
    if refusal is not None:
        notices.append(refusal)
        return (*_UNCHANGED, cap_notices(notices), *_NO_JOB)
    job = jobs.start_job(
        state, constrained.JOB_KIND, constrained.background_run,
        label=constrained.JOB_LABEL,
        params={"run": snapshot, "finalize": finalize})
    if job is None:             # another start won the race between the check and here
        notices.append(jobs.refusal_notice(state, constrained.JOB_KIND)
                       or jobs.busy_notice(jobs.busy_label(state)
                                           or constrained.JOB_LABEL))
        return (*_UNCHANGED, cap_notices(notices), *_NO_JOB)
    if est is not None:         # without a prognosis the failed-estimate notice already said it
        notices.append(heavy_run_notice(est))
    return (no_update, constrained.status_text(job), no_update, no_update,
            cap_notices(notices), False, {})


def _held_job_outputs(state, notices: list[dict]):
    """A Run click while the state holds a constrained job: the seven outputs of ``run_constrained``.

    While the job runs the click is refused with the job service's notice. While its outcome is being
    handed to the page the click is refused too and the poll stays on. A job that has ended but was never
    polled (e.g. the page was reloaded) switches the poll on, which hands its outcome over, instead of
    refusing every later click. None when the state holds no constrained job.
    """
    phase = jobs.held_phase(state, _CONSTRAINED_SLOT)
    if phase is None:
        return None
    if phase == jobs.HELD_RUNNING:
        notices.append(jobs.refusal_notice(state, constrained.JOB_KIND)
                       or jobs.busy_notice(constrained.JOB_LABEL))
        return (*_UNCHANGED, cap_notices(notices), *_NO_JOB)
    if phase == jobs.HELD_HANDOVER:
        notices.append(job_poll.handover_notice(constrained.JOB_LABEL))
    else:
        notices.append(job_poll.ended_notice("constrained", "Run routing"))
    return (*_UNCHANGED, cap_notices(notices), False, STOP_HIDDEN)


def _run_or_no_path(*args, **kwargs):
    """``run_constrained_path`` with its "no path" answer (None) made explicit.

    ``guard`` returns None for a crash AND for a plain None result, and a crash
    already carries its own notice; the marker tells the two cases apart so a
    failure is never also blamed on the profile.
    """
    info: dict = {}
    result = constrained.run_constrained_path(*args, **kwargs, outcome=info)
    return _NoPath(info.get("no_route")) if result is None else result


def objective_settings(state) -> dict:
    """The weights of the "Objective weights" block as the project state holds them (defaults when none).

    The run reads them here instead of through a new ``State`` (hard rule of the run callback): the block's
    mirror keeps ``state.gui_settings["constrained"]["objective"]`` current.
    """
    held = (getattr(state, "gui_settings", None) or {}).get("constrained")
    return constrained_objective.normalise(held.get("objective") if isinstance(held, dict) else None)


def ends_settings(state) -> dict:
    """The connection heights of the "Line ends" group as the project state holds them (none set when absent).

    Read here instead of through a new ``State`` (hard rule of the run callback): the group's mirror keeps
    ``state.gui_settings["constrained"]["ends"]`` current. The end-span rule is not here: it is a field of the
    profile text the run parses.
    """
    held = (getattr(state, "gui_settings", None) or {}).get("constrained")
    return constrained_ends.normalise(held.get("ends") if isinstance(held, dict) else None)


def _parse_profile_text(text: str) -> dict:
    import yaml

    config = yaml.safe_load(text or "")
    if not isinstance(config, dict) or not config:
        raise ValueError("The profile editor is empty or not a YAML/JSON "
                         "mapping.")
    return config


def register(app, state) -> None:
    @app.callback(Output(ids.OHL_COLLAPSE, "is_open"),
                  Output(ids.OHL_PROFILE_PRESET, "options"),
                  Input(ids.OHL_ENABLE, "value"))
    def toggle(enabled):
        options = [{"label": name, "value": path}
                   for name, path in constrained.shipped_profiles().items()]
        return bool(enabled), options

    @app.callback(Output(ids.OHL_PROFILE_TEXT, "value"),
                  Input(ids.OHL_PROFILE_PRESET, "value"),
                  prevent_initial_call=True)
    def load_preset(path):
        if not path:
            raise PreventUpdate
        from pathlib import Path

        return Path(path).read_text(encoding="utf-8")

    @app.callback(Output(ids.OHL_PROFILE_TEXT, "value",
                         allow_duplicate=True),
                  Output(ids.NOTICES, "data", allow_duplicate=True),
                  Input(ids.OHL_PROFILE_LOAD_BTN, "n_clicks"),
                  State(ids.OHL_PROFILE_PATH, "value"),
                  State(ids.NOTICES, "data"),
                  prevent_initial_call=True)
    def load_profile_file(n_clicks, path, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        config, notices = guard(constrained.load_profile_dict, path,
                                notices=notices)
        if config is None:
            return no_update, notices
        import yaml

        return yaml.safe_dump(config, sort_keys=False,
                              allow_unicode=True), notices

    @app.callback(Output(ids.NOTICES, "data", allow_duplicate=True),
                  Input(ids.GO_SAVE_PROFILE, "data"),
                  State(ids.OHL_PROFILE_PATH, "value"),
                  State(ids.OHL_PROFILE_TEXT, "value"),
                  State(ids.NOTICES, "data"),
                  prevent_initial_call=True)
    def save_profile_file(go, path, text, notices):
        from ..services import project_io

        notices = list(notices or [])
        if not go:
            raise PreventUpdate
        config, notices = guard(_parse_profile_text, text, notices=notices)
        if config is None:
            return notices
        problem = constrained.validate_profile(config)
        if problem is not None:
            notices.append(problem.to_dict())
            return notices
        result, notices = guard(project_io.save_profile, config, path,
                                notices=notices)
        if result is not None:
            notices.append(success("Profile saved",
                                   meaning=f"Written to {result}."))
        return notices

    # constrained routing shares the main Run button; the toggle decides which
    # runner acts, and waypoints become forced towers (task 46)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.OHL_STATUS, "children"),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.OHL_POLL, "disabled", allow_duplicate=True),
        Output(ids.OHL_STOP_BTN, "style", allow_duplicate=True),
        Input(ids.RUN_ROUTING_BTN, "n_clicks"),
        State(ids.OHL_ENABLE, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.OHL_PROFILE_TEXT, "value"),
        State(ids.OHL_BACKEND, "value"),
        State(ids.OHL_EXPERIMENTAL, "value"),
        State(ids.NEIGHBORHOOD, "value"),
        State(ids.SEARCH_BUFFER, "value"),
        State(ids.OHL_DEM, "value"),
        State(ids.OHL_DSM, "value"),
        State(ids.BUILD_POINTS_TABLE, "rowData"),
        State(ids.NOTICES, "data"),
        running=[(Output(ids.RUN_OHL_BUSY, "data"), True, False)],
        prevent_initial_call=True)
    def run_constrained(n_clicks, constrained_on, draft, raster_layer_id,
                        profile_text, backend, experimental, neighborhood_k,
                        search_buffer, dem, dsm, points_rows, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not n_clicks or not constrained_on:
            raise PreventUpdate          # the unconstrained runner handles it
        held = _held_job_outputs(state, notices)
        if held is not None:           # one run at a time; an ended one nobody polled is collected now
            return held

        stop = (no_update, no_update, no_update, no_update)
        raster_layer = state.get(raster_layer_id) if raster_layer_id else None
        if raster_layer is None or raster_layer.kind != "raster":
            notices.append(Notice(
                severity="warning", title="Pick a cost raster first",
                meaning="Constrained routing needs a raster layer.",
                impact="No route was computed.",
                fix="Build/load a raster and select it under 'Route on "
                    "raster'.", focus_id=ids.TAB_RASTER).to_dict())
            return (*stop, notices, *_NO_JOB)
        draft = draft or {}
        sources = [(p["x"], p["y"]) for p in draft.get("sources", [])]
        targets = [(p["x"], p["y"]) for p in draft.get("targets", [])]
        waypoints = [(p["x"], p["y"]) for p in draft.get("waypoints", [])]
        if not sources or not targets:
            notices.append(Notice(
                severity="warning", title="Set a source and a target first",
                meaning="Constrained routing uses the same source/target/"
                        "waypoint points; each waypoint forces a tower.",
                impact="There is nothing to route yet.",
                fix="Place points with the +Source / +Target / +Waypoint "
                    "click modes.", focus_id=ids.TAB_ROUTES).to_dict())
            return (*stop, notices, *_NO_JOB)
        points = [sources[0], *waypoints, targets[0]]

        config, notices = guard(_parse_profile_text, profile_text,
                                notices=notices)
        if config is None:
            return (*stop, notices, *_NO_JOB)
        problem = constrained.validate_profile(config)     # 21.5 shift-left
        if problem is not None:
            notices.append(problem.to_dict())
            return (*stop, notices, *_NO_JOB)
        # the form's limits and cross-field rules, checked again at run time
        # (the YAML text can be edited past the form): the field is named
        field_problem = profile_service.problem_notice(
            profile_service.profile_problems(config),
            controls=form_section.CONTROLS, tab=ids.TAB_ROUTES,
            impact="No overhead-line route was computed.")
        if field_problem is not None:
            notices.append({**field_problem, "severity": "error"})
            return (*stop, notices, *_NO_JOB)
        problem = numeric_field_problem(
            neighborhood_k, "Neighbourhood k", control=ids.NEIGHBORHOOD,
            tab=ids.TAB_ROUTES, minimum=routing.NEIGHBORHOOD_MIN,
            maximum=routing.NEIGHBORHOOD_MAX, whole=True
        ) or numeric_field_problem(
            search_buffer, "Search buffer (m)", control=ids.SEARCH_BUFFER,
            tab=ids.TAB_ROUTES, minimum=0)
        if problem is not None:
            notices.append(problem)
            return (*stop, notices, *_NO_JOB)
        neighborhood = routing.neighborhood_string(neighborhood_k)
        buffer_m = number_or_default(search_buffer, None)
        buffer_m = (float(buffer_m)
                    if buffer_m is not None and float(buffer_m) > 0 else None)
        info, notices = raster_info_or_notice(
            raster_layer.meta.get("source_path"), notices,
            impact="No overhead-line route was computed.")
        if info is None:
            return (*stop, notices, *_NO_JOB)
        resolution_m = info["resolution_m"]
        perf = constrained.validate_span_bin_vs_resolution(config,
                                                           resolution_m)
        if perf is not None:
            notices.append(perf.to_dict())
        backend, gate_notice = constrained.resolve_backend(
            backend, bool(experimental))                   # F10
        if gate_notice is not None:
            notices.append(gate_notice.to_dict())
        # the objective weights of the "Objective weights" block (from the project state, never a new State):
        # a value or a combination the library refuses is a field notice and nothing starts
        weights = objective_settings(state)
        weight_problems = constrained_objective.problems(
            weights, backend=backend,
            clearance_on=profile_service.clearance_state(config) == "on")
        if weight_problems:
            notices.append(constrained_objective.problem_notice(
                weight_problems, controls=form_section.OBJECTIVE_CONTROLS, tab=ids.TAB_ROUTES))
            return (*stop, notices, *_NO_JOB)
        objective = constrained_objective.library_objective(weights)       # None: classic routing
        objective_kwargs = {"objective": objective} if objective else {}

        # the line ends (round 17): the connection heights come from the project state (never a new State), the
        # end-span rule from the profile just parsed. Whatever the library would refuse, and an exact search that
        # would not fit the window, is a field notice and nothing starts - in the background path too.
        raster_path = raster_layer.meta.get("source_path")
        ends = ends_settings(state)
        end_problems = constrained_ends.run_problems(
            ends, raster_source=raster_path, points=points, backend=backend, profile=config, dem=dem,
            neighborhood=neighborhood, search_buffer_m=buffer_m)
        if end_problems:
            notices.append(constrained_ends.problem_notice(
                end_problems, controls=form_section.ends_controls(ends, config, dem), tab=ids.TAB_ROUTES))
            return (*stop, notices, *_NO_JOB)
        ends_kwargs = {} if constrained_ends.is_default(ends) else {"ends": ends}
        surface_notes = constrained_ends.surface_model_notes(config, dem, dsm)
        if surface_notes:           # the library ignores such a DSM without a word
            notices.append(Notice(
                severity="warning", title="Surface model (DSM) has no effect", meaning=" ".join(surface_notes),
                impact="The run goes on without it; the route is the one without a DSM.",
                fix="Give the run a terrain model (DEM) and a profile with the clearance settings, or clear the DSM.",
                focus_id=ids.TAB_ROUTES, focus_control=ids.OHL_DSM).to_dict())

        # up-front effort estimate (constrained runs are the heaviest; the
        # extended tower state multiplies the plain-routing estimate): a heavy
        # run - or one of unknown effort (#35) - goes to a background job
        est = None
        try:
            est = routing.estimate_routing(
                raster_layer.meta.get("source_path"), sources=[sources[0]],
                targets=[targets[0]], waypoints=waypoints,
                neighborhood=neighborhood, search_buffer_m=buffer_m)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # an unknown effort counts as heavy (#35)
            notices.append(estimate_failed_notice(exc))
        heavy = constrained.is_heavy(est)

        if len(sources) > 1 or len(targets) > 1:
            # one pair per run: say which points were left out, never drop
            # them silently
            notices.append(Notice(
                severity="warning",
                title="Only the first source and target are routed",
                meaning="Constrained routing runs ONE source-target pair. "
                        f"{len(sources) - 1} more source(s) and "
                        f"{len(targets) - 1} more target(s) were left out.",
                impact="Only the route from the first source to the first "
                       "target was computed.",
                fix="Run constrained routing once per pair, with a single "
                    "source and target in the points list each time.",
                focus_id=ids.TAB_ROUTES).to_dict())

        wp_names = [str(r.get("name", "")) for r in (points_rows or [])
                    if r.get("kind") == "waypoint"]
        finalize = {
            "config": config, "backend": backend, "neighborhood": neighborhood,
            "raster_layer_id": raster_layer.id, "wp_names": wp_names,
            "control_points": ([list(sources[0])] + [list(w) for w in waypoints]
                               + [list(targets[0])])}
        if objective:               # a classic run's finalize keywords are what they always were
            finalize["objective_weights"] = constrained_objective.used_weights(weights)
        if ends_kwargs:
            finalize["connection_heights"] = constrained_ends.used_ends(ends)
        if heavy:
            snapshot = constrained.snapshot_run_inputs(
                raster_path, points=points, profile=config, backend=backend,
                neighborhood=neighborhood, search_buffer_m=buffer_m,
                dem=dem, dsm=dsm, **objective_kwargs, **ends_kwargs)
            return _start_background_run(state, snapshot, finalize, est,
                                         notices)

        # light run: synchronous, exactly as before
        result, notices = guard(
            _run_or_no_path,
            raster_path, points=points,
            profile=config, backend=backend, neighborhood=neighborhood,
            search_buffer_m=buffer_m, dem=dem, dsm=dsm, **objective_kwargs, **ends_kwargs,
            notices=notices)
        if result is None:
            return (*stop, notices, *_NO_JOB)       # crashed: guard already explained
        if isinstance(result, _NoPath):
            notices.append(no_route_notice(result.info))
            return (*stop, notices, *_NO_JOB)
        return (*finalize_constrained(state, result, notices, **finalize),
                *_NO_JOB)

    # ------------------------------------------------ the background job
    # the shared job poll: the worker thread never touches the project; the route, its towers and the
    # notices are registered by ``constrained_outcome`` (the same ``finalize_constrained`` as a synchronous
    # run) and handed to the page exactly once
    job_poll.register_poll(app, state, job_poll.JobPoll(
        slot=_CONSTRAINED_SLOT, label=constrained.JOB_LABEL,
        poll=ids.OHL_POLL, stop=ids.OHL_STOP_BTN, ack=ids.JOB_ACK_CONSTRAINED,
        results=((ids.LAYERS_VIEW, "data"), (ids.OHL_STATUS, "children"),
                 (ids.EDIT_ROUTE_SELECT, "options"), (ids.EDIT_ROUTE_SELECT, "value")),
        status=(ids.OHL_STATUS, "children"),
        running_text=lambda job: constrained.status_text(job),
        finalize=lambda st, job: constrained_outcome(st, job)))

    @app.callback(
        Output(ids.OHL_STATUS, "children", allow_duplicate=True),
        Input(ids.OHL_STOP_BTN, "n_clicks"),
        prevent_initial_call=True)
    def stop_constrained(n_clicks):
        """Ask the running job to stop; it ends after the segment in flight and adds no route."""
        if not n_clicks:
            raise PreventUpdate
        outcome = jobs.stop_job(state, constrained.JOB_KIND)
        return (f"{outcome.text} {constrained.STOP_NOTE}" if outcome.accepted
                else outcome.text)

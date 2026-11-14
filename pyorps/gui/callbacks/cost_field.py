"""PYORPS GUI callbacks: the cost-field (siting) card (gap 3, package P11).

The click on "Compute cost field" checks everything through the cost-field service (``plan_run``) and hands the
run to the background-job service (``services/jobs``); the poll callback is the only place that changes the
``ProjectState`` for it. The worker thread only computes:

* ``compute_cost_field`` (the click): refuses while any job runs, collects an ended job nobody took, names a
  problem (status line and a notice that takes the user to the field), else starts the job.
* the job poll (``callbacks/job_poll.py``, ``poll_cost_field``): the progress while the job runs; once it ended
  the field becomes a raster layer (``cost_field_outcome`` -> ``finalize_cost_field``), the status line and the
  best-cell readout are written, handed to the page exactly once, and the poll goes off. A failed job is a notice,
  a stopped one is a notice and changes nothing.
* ``stop_cost_field``: Stop acts BETWEEN fixed points (the library has no cancel hook inside one search).
* ``update_run_button``: the button follows the Backend field and the raster dropdown and says why it is off.
* ``remember_cost_field_settings`` / ``restore_cost_field_settings``: the controls are mirrored into
  ``state.gui_settings["cost_field"]`` (the project file saves them from there) and put back after Open / New.

The click reads the route draft from its store (the draft lives in the browser) and the routing settings
(backend, neighbourhood, search buffer, whole-raster and forbidden-cells switches) from ``state.routing_settings``.

The objective and terrain card applies exactly as it does to a plain route: ``state.gui_settings["objective"]``
is checked with the helpers of ``callbacks/objective`` (:func:`check_objective`: a missing layer, a DEM in
another CRS or not covering the fixed points, a backend or kernel without terrain support refuse the click with a
notice that focuses the control) and translated by :func:`_finder_options` into the ``PathFinder`` keywords of
``plan_run``. With classic settings that is None and the field is the plain one. The settings used travel with
the job (``job.params["finalize"]["objective"]``) and are stored in the layer's ``meta["cost_field"]``.
"""
from __future__ import annotations

from pathlib import Path

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..sections import cost_field as section
from ..services import cost_field as service
from ..services import jobs, project_io, routing
from ..services import objective as objective_service
from ..services.errors import Notice, cap_notices, guard
from . import job_poll
from . import objective as objective_cb

#: the kind of the background job (the key in ``state.jobs`` and in ``layout.JOB_CONTROLS``)
JOB_KIND = "cost_field"
JOB_LABEL = "Cost field"
#: the Stop button is hidden unless a job runs
STOP_HIDDEN = job_poll.STOP_HIDDEN
#: where the state keeps the job
_SLOT = jobs.kind_slot(JOB_KIND)
#: what a stop is promised to do, for the status line and the Stop answer
STOP_NOTE = ("Stop acts between fixed points: the fixed point being solved always finishes (seconds to minutes on "
             "a large window), and a stopped run adds no layer.")
#: a refused run's status line shows at most this many messages
_PROBLEM_LINES = 3

_NO_VIEW = (no_update, no_update)        # LAYERS_VIEW, MAP_VIEW


# --------------------------------------------------------------------------- the objective of the page
#: what a refused objective check says about the run (the notices of ``callbacks/objective`` take it as the impact)
OBJECTIVE_IMPACT = "No cost field was computed."


def _finder_options(state) -> dict | None:
    """The extra ``PathFinder`` keywords of a run: what a plain route would use now, else None.

    Classic objective settings give None (the field is computed on the plain cost raster, exactly as before).
    Otherwise it is the ``objective_kwargs`` of ``services.objective.run_kwargs``: ``objective`` (the weights),
    ``weight_precision`` and, when set, ``dem``, ``gradient_options`` and ``metric_layers``. The settings must
    have passed :func:`check_objective` first (a layer that is not loaded raises ``ValueError`` here).
    """
    settings = objective_cb.read_settings(state)
    if objective_service.is_classic(settings):
        return None
    run = objective_service.run_kwargs(
        settings, layer_path=lambda layer_id: objective_cb.layer_path(state, layer_id),
        graph_api=routing_inputs(state)["graph_api"])
    return dict(run["objective_kwargs"])


def objective_info(state) -> dict | None:
    """What the layer records about the objective: ``{"spec", "summary"}``, None for classic settings.

    ``spec`` is the normalised settings (layer ids, no paths), ``summary`` the one line of the card
    (``objective.summary``), so a later look at the layer shows which objective produced it.
    """
    settings = objective_cb.read_settings(state)
    if objective_service.is_classic(settings):
        return None
    return {"spec": objective_service.objective_spec(settings),
            "summary": objective_service.summary(settings, routing_inputs(state)["graph_api"],
                                                 layer_names=objective_cb.layer_names(state))}


def check_objective(state, raster_layer_id, raster_path, origins: list, raw: dict, notices: list):
    """``(status_or_None, notices)``: the checks a route run makes for the objective, before a job starts.

    None means the run may go on (classic settings, or an objective that passed). Otherwise ``status`` is the
    line for the status component and a notice that names and focuses the control (the Objective & terrain card,
    the Backend field or this card's Kernel field) was appended. The rules are those of the route run
    (``objective_cb.prepare_run``); the backend and kernel rules are the field service's own
    (``service.check_backend``), so the words and the focused control are the ones of a cost field.
    """
    settings = objective_cb.read_settings(state)
    if objective_service.is_classic(settings):
        return None, notices
    inputs = routing_inputs(state)
    algorithm = raw.get("algorithm")
    backend = service.check_backend(
        inputs["graph_api"], algorithm if algorithm in service.ALGORITHM_CHOICES else "auto",
        has_dem=bool(settings["dem_layer_id"]))
    if backend:
        notices.append(problem_notice(backend))
        return problems_text(backend), notices
    try:
        bounds = service.raster_info(raster_path)["bounds"]
    except Exception:  # pylint: disable=broad-exception-caught  # an unreadable file is named by plan_run
        bounds = None
    try:
        buffer_m = service.resolve_buffer_m(raster_path, origins, inputs["search_buffer_m"],
                                            full_window=inputs["full_window"])
    except Exception:  # pylint: disable=broad-exception-caught  # an unusable buffer is named by plan_run
        buffer_m = 0.0
    run, _meta, notices = objective_cb.prepare_run(
        state, raster_layer=state.get(raster_layer_id), graph_api=inputs["graph_api"], algorithm="",
        raster_bounds=bounds, points=origins, buffer_m=buffer_m, notices=notices, impact=OBJECTIVE_IMPACT)
    if run is None:
        return str(notices[-1].get("meaning") or "The objective settings cannot be used."), notices
    return None, notices


# --------------------------------------------------------------------------- what the click reads
def routing_inputs(state) -> dict:
    """The routing settings the field reuses, as the keyword arguments of ``plan_run``.

    They come from ``state.routing_settings`` (mirrored from the controls of "Algorithm & search"); a page that
    has not mirrored anything yet gives the page defaults. Unusable values are kept as typed, so ``plan_run``
    names them.
    """
    settings = (routing.normalise_routing_settings(getattr(state, "routing_settings", None))
                or dict(routing.ROUTING_SETTING_DEFAULTS))
    return {"graph_api": settings["graph_api"], "neighborhood": settings["neighborhood_k"],
            "ignore_max_cost": settings["ignore_max_cost"], "search_buffer_m": settings["search_buffer_m"],
            "full_window": settings["full_window"]}


def _raster_path(state, raster_layer_id) -> str | None:
    """The file of the picked raster layer, or None when no raster layer is picked."""
    layer = state.get(raster_layer_id) if raster_layer_id else None
    if layer is None or layer.kind != "raster":
        return None
    return (layer.meta or {}).get("source_path") or None


# --------------------------------------------------------------------------- texts and notices
def problems_text(problems: list, limit: int = _PROBLEM_LINES) -> str:
    """The first problems of a refused run, one message per line, for the status line."""
    lines = [message for _key, message in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"... and {len(problems) - limit} more.")
    return "\n".join(lines)


def problem_notice(problems: list, *, picking: bool = False) -> dict:
    """The notice of a refused run: it names the first problem and takes the user to the field to change.

    ``picking`` is the case "no raster is picked": the dropdown sits in the Routes tab. Any other cost-raster
    problem (the file is gone, the cell type does not fit) is fixed in the Raster tab. The controls of the
    other problem keys are in the Routes tab (:data:`section.CONTROLS`).
    """
    first = problems[0][0]
    if first == "raster" and not picking:
        controls = {key: control for key, control in section.CONTROLS.items() if key != "raster"}
        return service.problem_notice(problems, controls=controls, tab=ids.TAB_RASTER)
    return service.problem_notice(problems, controls=section.CONTROLS, tab=ids.TAB_ROUTES)


def stopped_notice() -> dict:
    """The box for a run that Stop ended: no layer, nothing changed."""
    return Notice(
        severity="info", title="Cost field stopped",
        meaning="Stop was pressed; the run ended between two fixed points.",
        impact="No cost field was computed and the project is unchanged.",
        fix="Press Compute cost field again to start over.").to_dict()


def status_text(job) -> str:
    """The status line of a running job: its progress, and that a stop is waiting."""
    text = job.progress_text()
    if job.cancel.is_set() and job.status == jobs.RUNNING:
        text += " (stop requested: the run ends when the fixed point in flight is finished)"
    return text


# --------------------------------------------------------------------------- collecting a job
def _discard_result_file(state, job) -> None:
    """Remove the file of a stopped run that still returned a result (Stop came during the write).

    Only a file inside the project's work folder, and only the one this run wrote.
    """
    result = job.result if isinstance(job.result, dict) else {}
    path = result.get("path")
    if not path:
        return
    try:
        target = Path(path).resolve()
        if Path(state.work_dir).resolve() in target.parents and target.is_file():
            target.unlink()
    except (OSError, ValueError):                  # a leftover temporary file is not worth a failure
        pass


def cost_field_outcome(state, job):
    """The ended cost-field job as ``(results, notices)`` for the job poll (``callbacks/job_poll.py``).

    ``results`` are the values of ``(layers_view, map_view, status, best_info)``, ``notices`` the notice
    dicts to add. Done: the field becomes a raster layer (``finalize_cost_field``); the status line gets the
    summary and the readout the cheapest cell. Stopped: a notice, nothing changes in the project. Failed: the
    notice of the error. Runs once per job, in a poll tick.
    """
    notices: list[dict] = []
    if job.status == jobs.FAILED:
        failed = service.job_error_notice(job.error)
        if failed is not None:
            notices.append(failed)
        return (*_NO_VIEW, job.progress_text(), no_update), notices
    if job.status == jobs.CANCELLED:
        _discard_result_file(state, job)
        notices.append(stopped_notice())
        return (*_NO_VIEW, f"{job.progress_text()} No cost field was computed.", no_update), notices
    result = job.result
    if not isinstance(result, dict):
        notices.append(Notice(
            severity="error", title="The cost field run returned nothing",
            meaning="The background run ended without a result.", impact="No cost field layer was added.",
            fix="Press Compute cost field again.").to_dict())
        return (*_NO_VIEW, job.progress_text(), no_update), notices
    finalize = job.params.get("finalize") or {}
    objective = finalize.get("objective")
    if objective:                                  # the layer and the status line say which objective produced it
        result = {**result, "objective_spec": objective.get("spec"), "objective_summary": objective.get("summary")}
    layer, notices = service.finalize_cost_field(
        state, result, notices, raster_layer_id=finalize.get("raster_layer_id"))
    if layer is None:                              # the file could not be served: the notice says why
        return (*_NO_VIEW, f"{job.progress_text()} The field was computed but could not be shown.",
                no_update), notices
    view = {"fit_bounds": layer.tile.bounds, "seq": finalize.get("seq") or 1}
    return (state.layers_view(), view, service.summary_text(result), service.best_text(result)), notices


# --------------------------------------------------------------------------- checking a click
def plan_click(state, raster_layer_id, draft, raw: dict, notices: list[dict]):
    """``(plan_or_None, status, notices)``: the checked run of a Compute click, or why there is none.

    A refused run puts its first problems into ``status`` and a notice that names the field into ``notices``.
    The fixed points come from the route draft (``raw["origins"]`` says which), the routing settings from the
    state, the objective's checks from :func:`check_objective` and its keywords from :func:`_finder_options`.
    """
    raster_path = _raster_path(state, raster_layer_id)
    if raster_path is None:
        problems = [service.Problem(
            "raster", "Cost raster (raster): no raster layer is picked. Allowed: a cost raster picked under "
                      "'Route on raster' (build or load one in the Raster tab).")]
        notices.append(problem_notice(problems, picking=True))
        return None, problems_text(problems), notices
    which = raw["origins"] if raw["origins"] in service.ORIGIN_CHOICES else "sources"
    origins = service.origins_from_draft(draft, which)
    refused, notices = check_objective(state, raster_layer_id, raster_path, origins, raw, notices)
    if refused is not None:
        return None, refused, notices
    plan, notices = guard(
        service.plan_run, raster_path, raw, origins, **routing_inputs(state),
        out_path=service.new_out_path(state.work_dir), study_area=state.study_area,
        finder_options=_finder_options(state), notices=notices)
    if plan is None:                               # plan_run itself failed: guard wrote the notice
        return None, "The cost field could not be checked (see the notice).", notices
    if plan.problems:
        notices.append(problem_notice(plan.problems))
        return None, problems_text(plan.problems), notices
    return plan, "", notices


# --------------------------------------------------------------------------- registration
def register(app, state) -> None:  # pylint: disable=too-many-statements
    """Register the cost-field callbacks."""
    setting_states = [State(section.SETTING_IDS[key], "value") for key in service.COST_FIELD_KEYS]

    # ------------------------------------------------ the Compute click
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.CF_STATUS, "children"),
        Output(ids.CF_BEST_INFO, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.CF_POLL, "disabled"),
        Output(ids.CF_STOP_BTN, "style"),
        Input(ids.CF_RUN_BTN, "n_clicks"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        *setting_states,
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def compute_cost_field(n_clicks, raster_layer_id, draft, *values):
        """Check everything, then start the background job."""
        notices = list(values[-1] or [])
        if not n_clicks:
            raise PreventUpdate

        phase = jobs.held_phase(state, _SLOT)
        if phase in (jobs.HELD_ENDED, jobs.HELD_HANDOVER):
            # an ended job nobody polled (e.g. after a reload): the poll hands its outcome over;
            # while that is under way the click waits for it as well
            notices.append(job_poll.ended_notice("cost field", "Compute cost field") if phase == jobs.HELD_ENDED
                           else job_poll.handover_notice(JOB_LABEL))
            return (*_NO_VIEW, no_update, no_update, cap_notices(notices), False, STOP_HIDDEN)
        refusal = jobs.refusal_notice(state, JOB_KIND)
        if refusal is not None:
            notices.append(refusal)
            return (*_NO_VIEW, no_update, no_update, cap_notices(notices), no_update, no_update)

        plan, status, notices = plan_click(
            state, raster_layer_id, draft, dict(zip(service.COST_FIELD_KEYS, values[:-1])), notices)
        if plan is None:
            return (*_NO_VIEW, status, no_update, cap_notices(notices), no_update, no_update)
        finalize = {"raster_layer_id": raster_layer_id, "seq": n_clicks}
        objective = objective_info(state)
        if objective is not None:
            finalize["objective"] = objective
        job = jobs.start_job(
            state, JOB_KIND, service.job_work, label=JOB_LABEL,
            params={"compute": plan.kwargs, "finalize": finalize})
        if job is None:                            # another start won the race between the check and here
            notices.append(jobs.refusal_notice(state, JOB_KIND)
                           or jobs.busy_notice(jobs.busy_label(state) or JOB_LABEL))
            return (*_NO_VIEW, no_update, no_update, cap_notices(notices), no_update, no_update)
        notices.extend(plan.notices)
        return (*_NO_VIEW, status_text(job), "", cap_notices(notices), False, {} if job.can_stop else STOP_HIDDEN)

    # ------------------------------------------------ the poll
    # the shared job poll: the worker thread never touches the project; the layer and the notices are
    # registered by ``cost_field_outcome`` and handed to the page exactly once
    job_poll.register_poll(app, state, job_poll.JobPoll(
        slot=_SLOT, label=JOB_LABEL, poll=ids.CF_POLL, stop=ids.CF_STOP_BTN, ack=ids.JOB_ACK_COST_FIELD,
        results=((ids.LAYERS_VIEW, "data"), (ids.MAP_VIEW, "data"), (ids.CF_STATUS, "children"),
                 (ids.CF_BEST_INFO, "children")),
        status=(ids.CF_STATUS, "children"),
        running_text=lambda job: status_text(job),
        finalize=lambda st, job: cost_field_outcome(st, job)))

    # ------------------------------------------------ Stop
    @app.callback(
        Output(ids.CF_STATUS, "children", allow_duplicate=True),
        Input(ids.CF_STOP_BTN, "n_clicks"),
        prevent_initial_call=True)
    def stop_cost_field(n_clicks):
        """Ask the running job to stop; it ends after the fixed point in flight and adds no layer."""
        if not n_clicks:
            raise PreventUpdate
        outcome = jobs.stop_job(state, JOB_KIND)
        return f"{outcome.text} {STOP_NOTE}" if outcome.accepted else outcome.text

    # ------------------------------------------------ the Compute button follows the Backend field and the raster
    @app.callback(
        Output(section.RUN_WRAP, "title"),
        Output(ids.CF_RUN_BTN, "disabled"),
        Input(ids.GRAPH_API, "value"),
        Input(ids.ROUTE_RASTER, "value"),
        Input(ids.LAYERS_VIEW, "data"))
    def update_run_button(graph_api, raster_layer_id, _view):
        """Enabled while the Backend can compute a field and a raster is picked; the hover text says why not."""
        ready, text = section.readiness(graph_api, _raster_path(state, raster_layer_id) is not None)
        return text, not ready

    # ------------------------------------------------ the settings, kept on the server for saving
    @app.callback(
        Output(ids.CF_SETTINGS, "data"),
        *[Input(section.SETTING_IDS[key], "value") for key in service.COST_FIELD_KEYS])
    def remember_cost_field_settings(*values):
        """Mirror the controls into ``state.gui_settings["cost_field"]``: the project file saves them from there.

        A blank or unusable value falls back to its default here (this copy is for restoring); the Compute click
        reads the controls themselves and names a value that cannot be used.
        """
        raw = dict(zip(service.COST_FIELD_KEYS, values))
        settings = project_io.normalise_gui_settings(
            {"cost_field": service.normalise_cost_field_settings(raw)}).get("cost_field", {})
        state.gui_settings["cost_field"] = settings
        return settings

    # ------------------------------------------------ Open / New project
    @app.callback(
        *[Output(component_id, prop) for component_id, prop in section.FORM_TARGETS],
        Output(ids.CF_STATUS, "children", allow_duplicate=True),
        Output(ids.CF_BEST_INFO, "children", allow_duplicate=True),
        Input(ids.PROJECT_STATUS, "children"),
        prevent_initial_call=True)
    def restore_cost_field_settings(project_status):
        """The saved settings come back after Open; New (and a project without them) puts the defaults back.

        The status line and the readout belonged to the project that was replaced and are cleared.
        """
        if str(project_status or "").lower().startswith("saved"):
            raise PreventUpdate                    # a save changes nothing the page does not hold
        saved = (state.gui_settings or {}).get("cost_field")
        return (*section.form_values(saved), "", "")

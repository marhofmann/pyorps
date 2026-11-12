"""PYORPS GUI callbacks: shared-trench corridor-graph card (gap 4, package P12).

The card builds the corridor graph through the corridor service (``services/corridor``) and the background-job
service (``services/jobs``). The callbacks stay thin; every decision lives in the services.

* ``build_corridor_graph``: the Build click. ``service.prepare_run`` checks the settings, the Backend field and the
  points against the raster; a problem becomes a notice that focuses the field (``section.CONTROLS``) and the first
  lines of the status. Otherwise the job starts with ``can_stop=False`` and the status says that it cannot be stopped.
* the job poll (``callbacks/job_poll.py``, ``poll_corridor``). While the job runs the status shows its progress;
  when it has ended :func:`corridor_outcome` finalises it (``service.finalize_corridor`` adds the three map layers
  and keeps the graph in ``state.corridor_graphs``), the summary and the notices are handed to the page exactly once
  and the poll goes off. This is the only place that changes the ``ProjectState`` for the job; the worker thread
  only computes.
* ``save_corridor_graph``: "Save graph" writes the latest graph as a GeoPackage. An existing file is never
  replaced silently: the Save click is the ``save_corridor`` gate of ``callbacks/confirm.py`` (it asks "Replace the
  existing corridor GeoPackage?" or goes ahead), and this callback listens on the gate's GO store
  (``ids.GO_SAVE_CORRIDOR``), so it only runs once the question is answered or was not needed.
* ``remember_corridor_settings`` mirrors the controls into ``state.gui_settings["corridor"]`` (the project file
  saves them from there), ``restore_corridor_settings`` puts them back after Open / New project, and
  ``update_run_button`` enables the Build button only while a run can start.
* ``browse_corridor_file`` is the 📁 button of the save path.
"""
from __future__ import annotations

from typing import Any

from dash import Input, Output, Patch, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..sections import corridor as section
from ..services import corridor as service
from ..services import dialogs, jobs, project_io
from ..services.errors import Notice, cap_notices, guard
from . import browse, job_poll

#: cap of the reason shown in the status line of a failed run (the notice holds the rest)
_REASON_CHARS = 300
#: where the state keeps the corridor job
_SLOT = jobs.kind_slot(service.JOB_KIND)


# --------------------------------------------------------------------------- the graphs
def latest_graph(state) -> tuple[str | None, Any]:
    """``(segments layer id, graph)`` of the newest corridor run whose layer still exists, else ``(None, None)``.

    The graph of a layer the user removed is ignored: the card then says there is nothing to save.
    """
    best = None
    for layer_id, graph in list(state.corridor_graphs.items()):
        layer = state.layers.get(layer_id)
        if layer is None:
            continue
        run = int((layer.meta or {}).get("corridor_run") or 0)
        if best is None or run >= best[0]:
            best = (run, layer_id, graph)
    return (best[1], best[2]) if best else (None, None)


def _layer_names(state, layer_ids: list[str]) -> str:
    layers = [state.get(layer_id) for layer_id in layer_ids]
    return ", ".join(f"'{layer.name}'" for layer in layers if layer is not None)


# --------------------------------------------------------------------------- the finished job
def _failed_status(job) -> str:
    """The status line of a failed run: when it failed and, shortly, why."""
    reason = (str(job.error or "").strip().splitlines() or [""])[0][:_REASON_CHARS]
    text = job.progress_text()
    return f"{text} {reason}" if reason else text


def corridor_outcome(state, job):
    """The ended corridor job as ``((layers view, status), notices)`` for the job poll (``callbacks/job_poll.py``).

    Done: ``service.finalize_corridor`` adds the layers and keeps the graph. Failed: the translated error as a
    notice. Stopped (only a project that was replaced can do that: the job cannot be stopped from the card): a
    notice, the project is left alone. Runs once per job, in a poll tick.
    """
    notices: list[dict] = []
    if job.status == jobs.FAILED:
        failed = job.error_notice()
        if failed is not None:
            notices.append(failed)
        return (no_update, _failed_status(job)), notices
    if job.status == jobs.CANCELLED:
        notices.append(Notice(
            severity="info", title="The corridor graph run was dropped",
            meaning="The run ended because the project was replaced while it was running.",
            impact="No corridor graph was kept and the project is unchanged.",
            fix="Press Build corridor graph again to start over.").to_dict())
        return (no_update, f"{job.progress_text()} No corridor graph was kept."), notices
    out, notices = guard(service.finalize_corridor, state, job.result, job.params, notices=notices)
    if out is None:
        return (no_update, "The corridor graph was built, but its layers could not be added (see the notice)."), notices
    notices = [*notices, *out.notices]
    if not out.layer_ids:                    # no pair of points could be connected: the notice says why
        reason = next((n.get("meaning") for n in out.notices if n.get("severity") == "warning"), "")
        return (no_update, f"{out.status_text} {reason}".strip()), notices
    status = f"{out.status_text}\nMap layers added: {_layer_names(state, out.layer_ids)}."
    return (state.layers_view(), status), notices


# --------------------------------------------------------------------------- the save
def _save(state, path, notices: list[dict], *, overwrite: bool) -> tuple[str, list[dict]]:
    """Write the latest graph; ``(status text, notices)``. A refusal is a notice that focuses the field."""
    layer_id, graph = latest_graph(state)
    problems = service.save_problems(graph, path, overwrite=overwrite)
    if problems:
        notices.append(service.problem_notice(problems, controls=section.CONTROLS, impact="Nothing was written."))
        return service.problems_text(problems), cap_notices(notices)
    written, notices = guard(service.save_corridor, graph, path, overwrite=overwrite, notices=notices)
    if written is None:
        return "The corridor graph was not saved (see the notice).", notices
    sections = len(getattr(graph, "segments", ()) or ())
    layer = state.get(layer_id)
    of_layer = f" of '{layer.name}'" if layer is not None else ""
    notices.append(Notice(
        severity="success", title="Corridor graph saved",
        meaning=f"The graph{of_layer} with {sections} trench section{'' if sections == 1 else 's'} was written to "
                f"{written}.",
        impact=f"The GeoPackage holds the layers {service.SEGMENTS_LAYER} and {service.NODES_LAYER}.").to_dict())
    return (f"Saved the corridor graph{of_layer} ({sections} trench section{'' if sections == 1 else 's'}) to "
            f"{written}."), cap_notices(notices)


# --------------------------------------------------------------------------- registration
def register(app, state) -> None:  # pylint: disable=too-many-statements
    """Register the corridor-card callbacks."""
    setting_inputs = [State(section.SETTING_IDS[key], "value") for key in service.SETTING_KEYS]

    # ------------------------------------------------ the Build click
    @app.callback(
        Output(ids.COR_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Output(ids.COR_POLL, "disabled"),
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Input(ids.COR_RUN_BTN, "n_clicks"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        *setting_inputs,
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def build_corridor_graph(n_clicks, raster_layer_id, draft, *values):
        """Check everything, then start the background job (it cannot be stopped)."""
        notices = list(values[-1] or [])
        if not n_clicks:
            raise PreventUpdate
        raw = dict(zip(service.SETTING_KEYS, values[:-1]))

        phase = jobs.held_phase(state, _SLOT)
        if phase in (jobs.HELD_ENDED, jobs.HELD_HANDOVER):
            # an ended job nobody polled (e.g. after a reload): the poll hands its outcome over;
            # while that is under way the click waits for it as well
            notices.append(job_poll.ended_notice("corridor graph", "Build corridor graph") if phase == jobs.HELD_ENDED
                           else job_poll.handover_notice(service.JOB_LABEL))
            return no_update, cap_notices(notices), False, no_update
        refusal = jobs.refusal_notice(state, service.JOB_KIND)
        if refusal is not None:
            notices.append(refusal)
            return no_update, cap_notices(notices), no_update, no_update

        layer = state.get(raster_layer_id) if raster_layer_id else None
        raster_path = (layer.meta or {}).get("source_path") if layer is not None and layer.kind == "raster" else None
        plan, notices = guard(service.prepare_run, raster_path, draft, raw, state.routing_settings, notices=notices)
        if plan is None:
            return "The run could not be checked (see the notice).", notices, no_update, no_update
        if plan.problems:
            notices.append(service.problem_notice(plan.problems, controls=section.CONTROLS))
            return service.problems_text(plan.problems), cap_notices(notices), no_update, no_update

        job = jobs.start_job(state, service.JOB_KIND, service.run_job, label=service.JOB_LABEL,
                             params={**plan.params, "raster_layer_id": raster_layer_id}, can_stop=False)
        if job is None:                      # another start won the race between the check and here
            notices.append(jobs.refusal_notice(state, service.JOB_KIND)
                           or jobs.busy_notice(jobs.busy_label(state) or service.JOB_LABEL))
            return no_update, cap_notices(notices), no_update, no_update
        notices.extend(plan.notices)
        return service.status_text(job), (cap_notices(notices) if plan.notices else no_update), False, no_update

    # ------------------------------------------------ the poll
    # the shared job poll (the job cannot be stopped: no Stop button); ``corridor_outcome`` adds the layers
    # and keeps the graph, and the outcome is handed to the page exactly once
    job_poll.register_poll(app, state, job_poll.JobPoll(
        slot=_SLOT, label=service.JOB_LABEL, poll=ids.COR_POLL, ack=ids.JOB_ACK_CORRIDOR,
        results=((ids.LAYERS_VIEW, "data"), (ids.COR_STATUS, "children")),
        status=(ids.COR_STATUS, "children"),
        running_text=lambda job: service.status_text(job),
        finalize=lambda st, job: corridor_outcome(st, job)))

    # ------------------------------------------------ save: after the gate (the question was answered or not needed)
    @app.callback(
        Output(ids.COR_STATUS, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GO_SAVE_CORRIDOR, "data"),
        State(ids.COR_SAVE_PATH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def save_corridor_graph(go, path, notices):
        """Write the latest graph. The gate has asked about an existing file; here it is replaced."""
        if not go:
            raise PreventUpdate
        return _save(state, path, list(notices or []), overwrite=True)

    # ------------------------------------------------ the 📁 button of the save path
    @app.callback(
        Output(ids.COR_SAVE_PATH, "value"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(browse.browse_button_id(ids.COR_SAVE_PATH), "n_clicks"),
        prevent_initial_call=True)
    def browse_corridor_file(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        path, error = dialogs.ask_save("routes", title="Save the corridor graph as", defaultextension=".gpkg")
        if path:
            return path, no_update
        if error:                            # no dialog could be shown
            append = Patch()
            append.append(browse.dialog_unavailable_notice(error).to_dict())
            return no_update, append
        return no_update, no_update          # cancelled: stay silent

    # ------------------------------------------------ the Build button follows the Backend field and the draft
    @app.callback(
        Output(section.RUN_WRAP, "title"),
        Output(ids.COR_RUN_BTN, "disabled"),
        Input(ids.GRAPH_API, "value"),
        Input(ids.ROUTE_DRAFT, "data"),
        Input(ids.COR_TERMINALS, "value"))
    def update_run_button(graph_api, draft, terminals):
        """Enabled while the Backend is cython and the draft gives two points; the hover text says why not."""
        ready, text = section.readiness(graph_api, draft, terminals)
        return text, not ready

    # ------------------------------------------------ the settings, kept on the server for saving
    @app.callback(
        Output(ids.COR_SETTINGS, "data"),
        *[Input(section.SETTING_IDS[key], "value") for key in service.SETTING_KEYS])
    def remember_corridor_settings(*values):
        """Mirror the controls into ``state.gui_settings["corridor"]``: the project file saves them from there.

        A blank or unusable field falls back to its default; a usable number outside its range is kept, so the
        Build click can name it.
        """
        raw = dict(zip(service.SETTING_KEYS, values))
        settings = project_io.normalise_gui_settings(
            {"corridor": service.normalise_corridor_settings(raw)}).get("corridor", {})
        state.gui_settings["corridor"] = settings
        return settings

    # ------------------------------------------------ Open / New project
    @app.callback(
        *[Output(component_id, prop) for component_id, prop in section.FORM_TARGETS],
        Output(ids.COR_STATUS, "children", allow_duplicate=True),
        Input(ids.PROJECT_STATUS, "children"),
        prevent_initial_call=True)
    def restore_corridor_settings(project_status):
        """The saved settings come back after Open; New (and a project without them) puts the defaults back."""
        if str(project_status or "").lower().startswith("saved"):
            raise PreventUpdate              # a save changes nothing the page does not hold
        saved = (state.gui_settings or {}).get("corridor")
        return (*section.form_values(saved), "")

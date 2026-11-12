"""PYORPS GUI callbacks: background rasterize job (gap 7).

A heavy rasterize run is started by ``callbacks/raster.py`` (``run_rasterize``); this module holds the
poll and the Stop button of that job. The poll is the shared job poll of ``callbacks/job_poll.py``: it is
the only place that changes the ``ProjectState`` for the job, through ``raster.rasterize_outcome``, which
finalises through the same ``raster.finalize_rasterize`` as a synchronous run, and it hands the outcome to
the page exactly once (round 15). The worker thread only computes.
"""
from __future__ import annotations

from dash import Input, Output
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import jobs
from ..services import rasterize as rasterize_service
from . import job_poll, raster

#: the poll of the rasterize job
POLL = job_poll.JobPoll(
    slot=jobs.kind_slot(rasterize_service.JOB_KIND), label=rasterize_service.JOB_LABEL,
    poll=ids.RASTERIZE_POLL, stop=ids.RASTERIZE_STOP_BTN, ack=ids.JOB_ACK_RASTERIZE,
    results=((ids.LAYERS_VIEW, "data"), (ids.MAP_VIEW, "data"), (ids.RASTERIZE_LOG, "children")),
    status=(ids.RASTERIZE_LOG, "children"),
    # looked up when called, so a test can replace them on the module
    running_text=lambda job: raster.rasterize_status_text(job),
    finalize=lambda state, job: raster.rasterize_outcome(state, job))


def register(app, state) -> None:
    job_poll.register_poll(app, state, POLL)

    @app.callback(
        Output(ids.RASTERIZE_LOG, "children", allow_duplicate=True),
        Input(ids.RASTERIZE_STOP_BTN, "n_clicks"),
        prevent_initial_call=True)
    def stop_rasterize(n_clicks):
        """Ask the running build to stop; it ends at its next stage boundary (nothing is built)."""
        if not n_clicks:
            raise PreventUpdate
        outcome = jobs.stop_job(state, rasterize_service.JOB_KIND)
        return f"{outcome.text} {raster.STOP_NOTE}" if outcome.accepted else outcome.text

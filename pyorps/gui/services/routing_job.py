"""
Background routing jobs (accept-waiting flow + Stop button).

Small runs stay synchronous inside their callback. A HEAVY run (see
``routing.needs_confirmation``) is confirmed by the user first and then
executes here, in a daemon thread, so the UI stays responsive and a Stop
button can interrupt it: the cancel event is checked before every pair and
between waypoint legs — the segment in flight finishes, completed routes are
kept.

Thread-safety contract: the worker thread ONLY computes (PathFinder +
geometries) and writes to its own ``RoutingJob``; every ``ProjectState``
mutation (adding route layers, notices) happens later in the polling
callback on the Dash side.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import routing


@dataclass
class RoutingJob:
    """One in-flight (or finished) background routing run."""

    params: dict[str, Any]
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    status: str = "running"          # running | done | cancelled | failed
    done_pairs: int = 0
    total_pairs: int = 0
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    result: tuple | None = None      # (finder, [BuiltRoute], failed)
    error: Exception | None = None

    @property
    def elapsed_s(self) -> float:
        end = self.finished if self.finished is not None else time.monotonic()
        return end - self.started

    def progress_text(self) -> str:
        base = f"routing… {self.done_pairs}/{self.total_pairs or '?'} pair(s)"
        return f"{base}, {self.elapsed_s:,.0f} s elapsed — Stop aborts " \
               "after the segment in flight"


def start_routing_job(state, params: dict[str, Any]) -> RoutingJob:
    """Launch ``routing.run_routing(**params['kwargs'])`` in a daemon thread.

    ``params`` carries everything the polling callback needs to finalize:
    ``kwargs`` (the run_routing arguments), ``raster_layer_id``,
    ``simplify_tol``.
    """
    job = RoutingJob(params=params)

    def _progress(done: int, total: int) -> None:
        job.done_pairs, job.total_pairs = done, total

    def _work() -> None:
        try:
            job.result = routing.run_routing(
                params["raster_path"], cancel=job.cancel,
                progress=_progress, **params["kwargs"])
            job.status = "cancelled" if job.cancel.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - reported via notices later  # pylint: disable=broad-exception-caught
            job.error = exc
            job.status = "failed"
        finally:
            job.finished = time.monotonic()

    job.thread = threading.Thread(target=_work, daemon=True,
                                  name="pyorps-routing-job")
    job.thread.start()
    state.routing_job = job
    return job

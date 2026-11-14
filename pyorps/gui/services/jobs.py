"""
Generic background jobs for the GUI (round 14, package P1).

``routing_job`` (one job, one kind) showed the pattern; this module generalises
it for the rasterize, constrained, cost-field and corridor runs. A job lives in
``state.jobs[kind]`` while it runs and until the page has confirmed its
outcome (:func:`poll_tick`, below).

Thread-safety contract (the same as for the routing job):

* the worker thread ONLY computes. ``work(job)`` gets its own
  :class:`BackgroundJob` and may call ``job.report(...)`` and
  ``job.raise_if_cancelled()``; it must not touch ``ProjectState``, layers or
  Dash objects. Everything the finishing step needs goes in ``job.params``
  (read) and comes back through the return value (``job.result``);
* every ``ProjectState`` change (layers, notices, status text) happens on the
  Dash side, in the finalize step of a poll tick (:func:`poll_tick`);
* only one background job of any kind runs at a time (CPU headroom for the
  user's other work): :func:`busy_label` also sees ``state.routing_job``.

A worker that raises is never lost: the exception and its traceback text are
kept on the job (``error`` / ``traceback_text``), written to the log, and
``job.error_notice()`` turns them into a notice through
``services.errors.translate_exception``.

Handing an ended job to the page (round 15, :func:`poll_tick`)
--------------------------------------------------------------
The browser keeps only the answer of the NEWEST call of a callback: when a poll
interval fires again while the previous tick is still being served, Dash drops
the older tick's answer. A tick that ran the slow finalize step (adding a large
raster layer takes seconds) therefore lost its outputs, and the next tick found
no job and switched the poll off. The handover makes delivery independent of
which tick's answer survives:

1. the first tick that sees the ended job only CLAIMS it (cheap): the status
   line says "adding the result to the project", Stop is hidden, and the job
   counts as busy, so no other job starts;
2. the next tick runs the finalize step exactly once (the claim is moved to
   "finalizing" under the lock; every other tick meanwhile answers "finalizing")
   and keeps its outputs on the server (:class:`Handover`);
3. every tick from then on answers with those outputs and the handover's token,
   which the page writes into a one-shot store; the first tick that carries the
   token back proves the page has the outputs, drops the job and switches the
   poll off.

So the outputs reach the page however long the finalize step takes and however
many ticks fire meanwhile, also after a page reload (the job stays in its slot
until the token comes back, and the layout's prefill switches the poll on again).
The job stays in its slot until then; :func:`take_finished` is the older,
one-step way to take an ended job and is not used by the page any more.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from . import errors

_LOG = logging.getLogger(__name__)

#: ``BackgroundJob.status`` values; a job ends in exactly one of the last three
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"
FINISHED = (DONE, CANCELLED, FAILED)

#: what :func:`busy_label` calls the plain-routing job of ``services.routing_job``
ROUTING_LABEL = "Routing"

_clock = time.monotonic        # one place to replace in tests
_LOCK = threading.Lock()       # makes "check busy, register, start" one step


class JobCancelled(Exception):
    """Raised inside ``work(job)`` when the user's Stop was seen: the job ends "cancelled"."""


def run_collecting_warnings(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> tuple[Any, list[dict]]:
    """``(function(*args, **kwargs), notices of the warnings it raised)``; an exception of ``function`` propagates.

    When ``function`` raises, the exception is re-raised as it is, with the notices collected up to then on its
    ``collected_notices`` attribute (when the exception takes attributes), so a caller that wants the warnings
    of a failed run can still show them.

    A worker thread is not the thread of ``errors.guard`` in a Dash callback, so the library's warnings
    (an ``AlgorithmFallbackWarning``, quantisation warnings, thin-feature warnings, ...) would be lost; the worker
    collects them itself with this helper and returns the notice dicts inside its result.

    ``errors.guard`` uses ``warnings.catch_warnings``, which changes process-wide state: two threads inside it
    at the same moment can leave a stale filter behind. The risk is small (the other side is a short ``guard``
    call on a Dash thread, and one job runs at a time) and the only other way to show the user a library
    warning of a long run is not to show it.
    """
    outcome: dict[str, Any] = {}

    def call() -> None:
        try:
            outcome["value"] = function(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # re-raised below
            outcome["error"] = exc

    _, notices = errors.guard(call)
    if "error" in outcome:
        error = outcome["error"]
        try:
            error.collected_notices = notices
        except AttributeError:               # an exception type without a __dict__ keeps no notices
            pass
        raise error
    return outcome["value"], notices


class StopRequest(NamedTuple):
    """The answer to a stop request: was it accepted, and what to tell the user."""

    accepted: bool
    text: str


@dataclass(eq=False)
class BackgroundJob:
    """One in-flight (or finished) background run.

    The scaffold (``layout._prefill_jobs``, ``ProjectState.clear``) reads it duck-typed:
    ``cancel`` (a ``threading.Event``), ``status``, ``progress_text()``, ``can_stop``.
    """

    kind: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    can_stop: bool = True
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    status: str = RUNNING            # running | done | cancelled | failed (written last by the worker)
    done: int = 0
    total: int = 0
    stage: str = ""
    started: float = field(default_factory=lambda: _clock())
    finished: float | None = None
    result: Any = None               # what work(job) returned (kept for "cancelled" too, when it returned)
    error: BaseException | None = None
    traceback_text: str = ""
    handover: Handover | None = None  # set once a poll tick has claimed the ended job (see poll_tick)

    @property
    def elapsed_s(self) -> float:
        end = self.finished if self.finished is not None else _clock()
        return end - self.started

    def report(self, done: int | None = None, total: int | None = None, stage: str | None = None) -> None:
        """Progress from the worker; an argument left out keeps its old value.

        Fits a service's ``progress=`` callback of either shape: ``progress(done, total)`` or
        ``progress(done, total, stage)``.
        """
        if done is not None:
            self.done = int(done)
        if total is not None:
            self.total = int(total)
        if stage is not None:
            self.stage = str(stage)

    def raise_if_cancelled(self) -> None:
        """Cancel point for ``work``: raises :class:`JobCancelled` when Stop was requested."""
        if self.cancel.is_set():
            raise JobCancelled(f"{self.label} was stopped")

    def request_stop(self) -> StopRequest:
        """Ask the job to stop at its next cancel point; says so when it cannot."""
        if self.status != RUNNING:
            return StopRequest(False, f"{self.label} has already ended.")
        if not self.can_stop:
            return StopRequest(False, f"{self.label} cannot be stopped; it ends by itself when it is done.")
        self.cancel.set()
        return StopRequest(True, f"Stop requested — {self.label} ends after the current step.")

    def progress_text(self) -> str:
        """The status line: "<label>... 2/5 <stage>, 14 s elapsed — Stop ends it after the current step"."""
        elapsed = f"{self.elapsed_s:,.0f} s"
        ended = {DONE: "finished after", CANCELLED: "was stopped after", FAILED: "failed after"}
        if self.status in ended:
            return f"{self.label} {ended[self.status]} {elapsed}."
        if self.total:
            count = f"{self.done}/{self.total}"
        else:
            count = str(self.done) if self.done else ""
        detail = " ".join(part for part in (count, self.stage) if part)
        text = f"{self.label}... " + (f"{detail}, " if detail else "") + f"{elapsed} elapsed"
        if self.can_stop:
            text += " — Stop ends it after the current step"
        return text

    def error_notice(self) -> dict | None:
        """The notice dict for a failed job (None when it did not fail)."""
        if self.error is None:
            return None
        return errors.translate_exception(self.error).to_dict()


# ------------------------------------------------------------------ the worker
def _record_failure(job: BackgroundJob, exc: BaseException) -> None:
    """Keep the exception and its traceback text, log it, and free what its frames hold."""
    job.error = exc
    try:
        job.traceback_text = "".join(traceback.format_exception(exc))
        _LOG.error("Background job %r failed:\n%s", job.kind, job.traceback_text)
        # the traceback keeps every frame (and so every big raster array of the run) alive until the
        # poll callback collects the job; the text above is all the notice needs
        traceback.clear_frames(exc.__traceback__)
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        _LOG.exception("Could not record the failure of background job %r", job.kind)


def _run(job: BackgroundJob, work: Callable[[BackgroundJob], Any]) -> None:
    """Thread body: run ``work(job)``, always end with a status, never raise."""
    outcome = FAILED
    try:
        job.result = work(job)
        outcome = CANCELLED if job.cancel.is_set() else DONE
    except JobCancelled:
        outcome = CANCELLED
    except BaseException as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        _record_failure(job, exc)           # even SystemExit: a silent dead thread would stay "running"
    finally:
        job.finished = _clock()
        job.status = outcome                # last, so a poll that sees it also sees result / error


# ------------------------------------------------------------ state-side helpers
def _status(job: Any) -> str:
    return getattr(job, "status", RUNNING)


def _label(kind: str, job: Any) -> str:
    return getattr(job, "label", "") or kind


def _busy(job: Any) -> bool:
    """A job keeps the machine busy while it runs and while its result is being added to the project."""
    if _status(job) == RUNNING:
        return True
    handover = getattr(job, "handover", None)
    return handover is not None and handover.phase != READY


def busy_label(state: Any) -> str | None:
    """The label of a busy job (``state.routing_job`` included), or None when nothing is busy.

    Busy means running, or ended with its result being added to the project right now (a claimed or
    finalizing :class:`Handover`): no other job may start then.
    """
    routing = getattr(state, "routing_job", None)
    if routing is not None and _busy(routing):
        return ROUTING_LABEL
    for kind, job in dict(getattr(state, "jobs", None) or {}).items():
        if job is not None and _busy(job):
            return _label(kind, job)
    return None


def busy_notice(label: str) -> dict:
    """The notice dict for a start that was refused because ``label`` is still running."""
    return errors.Notice(
        severity="warning", title=f"{label} is still running",
        meaning="Only one background job runs at a time, so the machine keeps room for your other work.",
        impact="Nothing new was started.",
        fix="Wait until it has finished, or press its Stop button, then start again.").to_dict()


def _blocker(state: Any, kind: str) -> tuple[str, str] | None:
    """Why a job of ``kind`` may not start now: ("busy", label) or ("uncollected", label), else None."""
    label = busy_label(state)
    if label is not None:
        return "busy", label
    held = state.jobs.get(kind)      # nothing runs, so a job held here has ended and waits to be collected
    if held is not None:
        return "uncollected", _label(kind, held)
    return None


def refusal_notice(state: Any, kind: str) -> dict | None:
    """The notice dict explaining why :func:`start_job` would refuse ``kind`` now; None if it would start.

    Two reasons: a job (any kind, or the routing job) is running, or the same kind still holds a finished job
    that nobody has taken with :func:`take_finished` yet (starting would drop its result or error).
    """
    blocker = _blocker(state, kind)
    if blocker is None:
        return None
    reason, label = blocker
    if reason == "busy":
        return busy_notice(label)
    return uncollected_notice(label)


def uncollected_notice(label: str) -> dict:
    """The notice dict for a start that was refused because the last run of ``label`` waits to be shown."""
    return errors.Notice(
        severity="info", title="The last run has not been collected yet",
        meaning=f"{label} has ended; its result is still waiting to be shown.",
        impact="Nothing new was started.",
        fix="Wait a moment (the page collects it by itself), then start again.").to_dict()


def start_job(state: Any, kind: str, work: Callable[[BackgroundJob], Any], *, label: str,
              params: dict[str, Any] | None = None, can_stop: bool = True) -> BackgroundJob | None:
    """Run ``work(job)`` in a daemon thread named ``pyorps-<kind>-job``; the job is ``state.jobs[kind]``.

    Returns None, raising nothing, when the start is refused (see :func:`refusal_notice`): then no thread was
    started and ``state.jobs`` is unchanged. Callers normally check :func:`refusal_notice` (or
    :func:`busy_label`) first to show the reason; the check here is the same one, made under a lock.
    ``params`` is copied into ``job.params``. Whatever ``work`` returns becomes ``job.result``.
    """
    job = BackgroundJob(kind=kind, label=label, params=dict(params or {}), can_stop=bool(can_stop))
    thread = threading.Thread(target=_run, args=(job, work), daemon=True, name=f"pyorps-{kind}-job")
    job.thread = thread
    with _LOCK:
        if _blocker(state, kind) is not None:
            return None
        state.jobs[kind] = job
        try:
            thread.start()
        except BaseException:
            state.jobs.pop(kind, None)
            raise
    return job


def take_finished(state: Any, kind: str) -> BackgroundJob | None:
    """Remove and return ``state.jobs[kind]`` when it has ended; None while it still runs or when absent."""
    with _LOCK:
        job = state.jobs.get(kind)
        if job is None or _status(job) == RUNNING:
            return None
        del state.jobs[kind]
        return job


def stop_job(state: Any, kind: str) -> StopRequest:
    """Stop request for the job of ``kind``: accepted, or the reason it was not (nothing runs, cannot stop)."""
    job = state.jobs.get(kind)
    if job is None:
        return StopRequest(False, "Nothing is running to stop.")
    return job.request_stop()


# ------------------------------------------------------------- handing an ended job to the page
#: :class:`Handover` phases: claimed by a tick, being finalized by a tick, outputs waiting for the page
CLAIMED = "claimed"
FINALIZING = "finalizing"
READY = "ready"

#: :class:`Tick` steps: what one poll tick has to tell the page
TICK_IDLE = "idle"                   # no job (or the page has confirmed the outputs): switch the poll off
TICK_RUNNING = "running"             # the job runs: show its progress
TICK_FINALIZING = "finalizing"       # the result is being added to the project: say so, hide Stop
TICK_DELIVER = "deliver"             # the outputs are ready: hand them (again) to the page with the token

#: the words the status line uses while a result is being added (tests and the layout look for them)
FINALIZING_NOTE = "adding the result to the project"


def finalizing_text(label: str) -> str:
    """The status line while an ended job's result is being added to the project (Stop is hidden then)."""
    return f"{label} has ended; {FINALIZING_NOTE}... The page updates by itself."


@dataclass(eq=False)
class Handover:
    """The outcome of one ended job on its way to the page (see the module docstring).

    ``results`` holds the finalize step's values for the kind's result outputs (None: leave them all
    unchanged), ``notices`` the notice dicts it added; the poll tick appends those to the page's current
    notices when it delivers, so a notice the page got in the meantime is never overwritten. ``token`` is what
    the page writes into the kind's one-shot store and sends back to confirm it has the outputs.
    """

    phase: str = CLAIMED
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    results: tuple | None = None
    notices: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class JobSlot:
    """Where the state keeps the job of one kind: ``state.jobs[kind]``, or a state attribute (the routing job)."""

    kind: str
    attribute: str | None = None

    def get(self, state: Any) -> Any:
        """The job held for this kind, or None."""
        if self.attribute is not None:
            return getattr(state, self.attribute, None)
        return (getattr(state, "jobs", None) or {}).get(self.kind)

    def drop(self, state: Any, job: Any) -> None:
        """Forget ``job`` (caller holds the lock); a newer job in the slot is left alone."""
        if self.get(state) is not job:
            return
        if self.attribute is not None:
            setattr(state, self.attribute, None)
        else:
            state.jobs.pop(self.kind, None)


#: the slot of the plain-routing job of ``services.routing_job``
ROUTING_SLOT = JobSlot("routing", attribute="routing_job")


def kind_slot(kind: str) -> JobSlot:
    """The slot of the job of ``kind`` started with :func:`start_job`."""
    return JobSlot(kind)


class Tick(NamedTuple):
    """What one poll tick tells the page: a ``TICK_*`` step, the job (running / finalizing) and the handover."""

    step: str
    job: Any = None
    handover: Handover | None = None


#: what :func:`held_phase` reports for the job a start click finds in its slot
HELD_RUNNING = "running"             # the job runs: refuse the click
HELD_HANDOVER = "handover"           # the page is already being handed the outcome: refuse, keep polling
HELD_ENDED = "ended"                 # ended and never claimed (nobody polled it): switch the poll on to collect it


def held_phase(state: Any, slot: JobSlot) -> str | None:
    """What a start click finds in ``slot``: a ``HELD_*`` value, or None when the slot is empty."""
    with _LOCK:
        job = slot.get(state)
        if job is None:
            return None
        if getattr(job, "handover", None) is not None:
            return HELD_HANDOVER
        return HELD_RUNNING if _status(job) == RUNNING else HELD_ENDED


def _finalize_once(job: Any, finalize: Callable[[Any], tuple[tuple | None, list[dict]]]
                   ) -> tuple[tuple | None, list[dict]]:
    """``finalize(job)``; an exception becomes a notice, so a broken finalize step still ends the handover."""
    try:
        results, notices = finalize(job)
        return results, list(notices or [])
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # reported as a notice
        _LOG.exception("Adding the result of background job %r to the project failed",
                       getattr(job, "kind", "routing"))
        return None, [errors.translate_exception(exc).to_dict()]


def poll_tick(state: Any, slot: JobSlot, ack: Any,
              finalize: Callable[[Any], tuple[tuple | None, list[dict]]]) -> Tick:
    """One poll tick of the job in ``slot``; the shared logic of every job poll (module docstring).

    ``ack`` is the token the page holds in the kind's one-shot store (None when it holds none);
    ``finalize(job)`` turns the ended job into ``(results, notices)``: the values of the kind's result outputs
    (None: leave them unchanged) and the notice dicts to add. It runs exactly once per job, in the calling
    thread (a Dash callback, never the worker thread), and never twice at the same time: the move from
    "claimed" to "finalizing" happens under the lock, and only the tick that made it calls ``finalize``.
    Several ticks may run at once (waitress serves them on several threads); each gets an answer that is
    right to show at that moment.
    """
    with _LOCK:
        job = slot.get(state)
        if job is None:
            return Tick(TICK_IDLE)
        handover = getattr(job, "handover", None)
        if handover is None:
            if _status(job) == RUNNING:
                return Tick(TICK_RUNNING, job)
            job.handover = Handover()               # claim it; the next tick adds the result
            return Tick(TICK_FINALIZING, job, job.handover)
        if handover.phase == READY:
            if ack is not None and ack == handover.token:
                slot.drop(state, job)               # the page has the outputs: done
                return Tick(TICK_IDLE)
            return Tick(TICK_DELIVER, job, handover)
        if handover.phase == FINALIZING:
            return Tick(TICK_FINALIZING, job, handover)
        handover.phase = FINALIZING                 # claimed: this tick (alone) adds the result
    results, notices = _finalize_once(job, finalize)
    with _LOCK:
        handover.results, handover.notices = results, notices
        handover.phase = READY
    return Tick(TICK_DELIVER, job, handover)

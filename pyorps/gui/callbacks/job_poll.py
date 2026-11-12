"""PYORPS GUI callbacks: the one poll callback of every background job (round 15).

Each job kind (rasterize, constrained routing, cost field, corridor graph, plain routing) describes its poll
once as a :class:`JobPoll` and registers it with :func:`register_poll`; the tick logic is
``services.jobs.poll_tick`` (see its module docstring for why: the browser keeps only the answer of the newest
call of a callback, so the finalize step's outputs are kept on the server until the page confirms them).

A poll callback has the kind's result outputs, then ``NOTICES``, the poll's ``disabled``, the Stop button's
``style`` (when the kind has one) and the kind's one-shot store; its Input is the poll's ``n_intervals``, its
States are ``NOTICES`` and that store. What one tick answers:

* nothing held (or the page has just confirmed the outputs): the poll goes off, Stop is hidden;
* the job runs: the status line shows its progress;
* the job has ended and its result is being added to the project: the status line says so, Stop is hidden
  (the poll stays on);
* the outputs are ready: the result outputs, the new notices appended to the page's current ones, Stop hidden
  and the handover's token in the store (the poll stays on; the next tick sends the token back).

The start callbacks never collect a job themselves any more: when a click finds an ended job nobody polled,
it switches the poll on (:func:`ended_notice` says so) and the poll hands the outcome over like any other.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dash import Input, Output, State, no_update

from .. import ids
from ..services import jobs
from ..services.errors import Notice, cap_notices

#: the style of a hidden Stop button
STOP_HIDDEN = {"display": "none"}


@dataclass(frozen=True)
class JobPoll:
    """The poll of one job kind.

    ``results`` are the ``(component id, property)`` pairs the finalize step writes, in the order of the values
    ``finalize(state, job)`` returns (with the list of notice dicts it added); ``status`` is the one of them
    that shows the progress and the "adding the result" line. ``running_text(job)`` words the progress of a
    running job (each kind says where its Stop takes effect).
    """

    slot: jobs.JobSlot
    label: str
    poll: str
    ack: str
    results: tuple[tuple[str, str], ...]
    status: tuple[str, str]
    running_text: Callable[[Any], str]
    finalize: Callable[[Any, Any], tuple[tuple | None, list[dict]]]
    stop: str | None = None

    def outputs(self) -> list[Output]:
        """The poll callback's Outputs, in the order :func:`answer` returns the values."""
        outputs = [Output(component, prop, allow_duplicate=True) for component, prop in self.results]
        outputs += [Output(ids.NOTICES, "data", allow_duplicate=True),
                    Output(self.poll, "disabled", allow_duplicate=True)]
        if self.stop is not None:
            outputs.append(Output(self.stop, "style", allow_duplicate=True))
        outputs.append(Output(self.ack, "data"))
        return outputs


def answer(spec: JobPoll, state: Any, tick: jobs.Tick, notices: list | None) -> tuple:
    """The values of :meth:`JobPoll.outputs` for one tick."""
    values: list[Any] = [no_update] * len(spec.results)
    status_at = spec.results.index(spec.status)
    new_notices: Any = no_update
    poll: Any = no_update
    stop: Any = no_update
    ack: Any = no_update
    if tick.step == jobs.TICK_IDLE:
        poll, stop = True, STOP_HIDDEN
    elif tick.step == jobs.TICK_RUNNING:
        values[status_at] = spec.running_text(tick.job)
    elif tick.step == jobs.TICK_FINALIZING:
        values[status_at] = jobs.finalizing_text(spec.label)
        stop = STOP_HIDDEN
    else:                                            # TICK_DELIVER
        handover = tick.handover
        if handover.results is not None:
            values = list(handover.results)
            for index, (component, _prop) in enumerate(spec.results):
                # the layer list as it is NOW (the user may have changed layers since the finalize step)
                if component == ids.LAYERS_VIEW and values[index] is not no_update:
                    values[index] = state.layers_view()
        if handover.notices:
            new_notices = cap_notices([*(notices or []), *handover.notices])
        stop, ack = STOP_HIDDEN, handover.token
    tail = [new_notices, poll] + ([stop] if spec.stop is not None else []) + [ack]
    return (*values, *tail)


def register_poll(app, state, spec: JobPoll) -> Callable:
    """Register the poll callback of ``spec``'s kind; returns the callback function."""

    def finalize(job):
        return spec.finalize(state, job)

    @app.callback(*spec.outputs(), Input(spec.poll, "n_intervals"), State(ids.NOTICES, "data"),
                  State(spec.ack, "data"), prevent_initial_call=True)
    def poll_job(_n, notices, ack):
        """Progress while the job runs; its outcome, handed over exactly once, after it has ended."""
        return answer(spec, state, jobs.poll_tick(state, spec.slot, ack, finalize), notices)

    poll_job.__name__ = f"poll_{spec.slot.kind}"
    return poll_job


def ended_notice(what: str, again: str) -> dict:
    """The notice of a start click that found an ended job nobody polled: the poll now collects it.

    ``what`` names the run ("rasterize", "cost field" ...), ``again`` the button ("Build cost raster").
    """
    return Notice(
        severity="info", title=f"An earlier {what} run was collected",
        meaning=f"A background {what} run ended while this page was not watching it; its outcome is shown in "
                "a moment.",
        impact="This click did not start a new run.",
        fix=f"Press {again} again for another run.").to_dict()


def handover_notice(label: str) -> dict:
    """The notice of a start click while the page is being handed the outcome of the last run of ``label``."""
    return jobs.uncollected_notice(label)

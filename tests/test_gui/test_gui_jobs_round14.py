"""Round 14, P1 - the generic background-job service (``services/jobs.py``).

Every test that starts a worker either lets it return at once or holds it on an Event it releases itself
(the ``gate`` fixture opens at the end, so no thread outlives its test). Threads are joined with a timeout.
"""
from __future__ import annotations

import logging
import threading
import types
import weakref

import pytest

from pyorps.core import exceptions as pex
from pyorps.gui import ids, layout
from pyorps.gui.services import jobs, routing_job

JOIN_S = 5


@pytest.fixture()
def gate():
    """An Event a worker can wait on; always set at the end so a failing test leaves no blocked thread."""
    event = threading.Event()
    yield event
    event.set()


def _hold(gate, value="held"):
    """A worker that blocks until the test opens ``gate``, then returns ``value``."""
    def work(_job):
        assert gate.wait(JOIN_S), "the test never opened the gate"
        return value
    return work


def _start(state, work, kind="rasterize", label="Rasterizing", **kwargs):
    job = jobs.start_job(state, kind, work, label=label, **kwargs)
    assert job is not None
    return job


def _finish(job):
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive(), "the worker did not end"
    return job


# ============================================================ the three ways a job ends
def test_a_worker_that_returns_ends_done_with_its_result(state):
    job = _finish(_start(state, lambda _j: {"cells": 7}, params={"layer": "a"}))
    assert job.status == "done" and job.result == {"cells": 7}
    assert job.error is None and job.error_notice() is None
    assert job.finished is not None and job.elapsed_s >= 0
    assert (job.kind, job.label, job.params) == ("rasterize", "Rasterizing", {"layer": "a"})
    assert state.jobs["rasterize"] is job          # kept until the poll side takes it


def test_the_thread_is_a_named_daemon_thread(state, gate):
    job = _start(state, _hold(gate), kind="cost_field", label="Cost field")
    assert job.thread.daemon is True and job.thread.name == "pyorps-cost_field-job"
    gate.set()
    _finish(job)


def test_the_worker_gets_its_own_job_and_reports_progress_on_it(state):
    seen = []

    def work(job):
        seen.append(job)
        job.report(2, 5, "burn")
        job.report(done=3)                      # keeps total and stage
        return None

    job = _finish(_start(state, work))
    assert seen == [job]
    assert (job.done, job.total, job.stage) == (3, 5, "burn")


def test_report_fits_a_two_argument_progress_callback(state):
    def work(job):
        for index in range(1, 4):
            job.report(index, 3)               # progress(done, total), as the services call it

    job = _finish(_start(state, work))
    assert (job.done, job.total, job.stage) == (3, 3, "")


def test_a_worker_that_raises_job_cancelled_ends_cancelled(state):
    def work(_job):
        raise jobs.JobCancelled("stopped")

    job = _finish(_start(state, work))
    assert job.status == "cancelled" and job.result is None
    assert job.error is None and job.error_notice() is None


def test_stop_reaches_the_worker_through_the_cancel_event(state):
    entered = threading.Event()

    def work(job):
        entered.set()
        assert job.cancel.wait(JOIN_S), "stop never arrived"
        job.raise_if_cancelled()
        return "not reached"

    job = _start(state, work)
    assert entered.wait(JOIN_S)
    assert job.request_stop() == (True, "Stop requested — Rasterizing ends after the current step.")
    _finish(job)
    assert job.status == "cancelled" and job.result is None


def test_a_worker_that_returns_after_stop_ends_cancelled_and_keeps_its_result(state):
    def work(job):
        assert job.cancel.wait(JOIN_S)
        return "partial"

    job = _start(state, work)
    job.cancel.set()
    _finish(job)
    assert job.status == "cancelled" and job.result == "partial"


def test_raise_if_cancelled_does_nothing_until_stop(state, gate):
    job = _start(state, _hold(gate))
    job.raise_if_cancelled()
    job.cancel.set()
    with pytest.raises(jobs.JobCancelled):
        job.raise_if_cancelled()
    gate.set()
    _finish(job)


# ============================================================ a failing worker is never lost
def _explode(_job):
    raise ValueError("boom in the worker")


def test_a_failing_worker_keeps_the_error_and_its_traceback(state, monkeypatch):
    unhandled = []
    monkeypatch.setattr(threading, "excepthook", unhandled.append)
    job = _finish(_start(state, _explode))
    assert unhandled == []                       # captured, the thread did not die with a raw traceback
    assert job.status == "failed" and job.result is None
    assert isinstance(job.error, ValueError)
    assert "Traceback (most recent call last)" in job.traceback_text
    assert "in _explode" in job.traceback_text and "ValueError: boom in the worker" in job.traceback_text


def test_a_failure_becomes_a_notice_through_the_error_registry(state):
    notice = _finish(_start(state, _explode)).error_notice()
    assert notice["severity"] == "error" and notice["title"] == "Something went wrong"
    assert "boom in the worker" in notice["details"]


def test_a_pyorps_error_gets_its_own_notice(state):
    def work(_job):
        raise pex.NoPathFoundError(1, 2)

    notice = _finish(_start(state, work)).error_notice()
    assert notice["title"] == "No route could be found"


def test_even_a_system_exit_ends_the_job_failed(state):
    def work(_job):
        raise SystemExit(3)

    job = _finish(_start(state, work))
    assert job.status == "failed" and isinstance(job.error, SystemExit)


def test_a_failure_is_written_to_the_log(state, caplog):
    with caplog.at_level(logging.ERROR, logger=jobs.__name__):
        _finish(_start(state, _explode))
    assert "Background job 'rasterize' failed" in caplog.text and "boom in the worker" in caplog.text


def test_a_failed_job_does_not_keep_the_workers_local_data_alive(state):
    class Big:
        pass

    probe = []

    def work(_job):
        big = Big()
        probe.append(weakref.ref(big))
        raise RuntimeError("x")

    _finish(_start(state, work))
    assert probe[0]() is None


# ============================================================ one job at a time
def test_a_second_start_of_the_same_kind_is_refused_while_one_runs(state, gate):
    first = _start(state, _hold(gate))
    assert jobs.busy_label(state) == "Rasterizing"
    second = jobs.start_job(state, "rasterize", _hold(gate), label="Rasterizing")
    assert second is None
    assert state.jobs == {"rasterize": first}
    notice = jobs.refusal_notice(state, "rasterize")
    assert notice["severity"] == "warning" and "Rasterizing is still running" in notice["title"]
    assert notice == {**jobs.busy_notice("Rasterizing"), "id": notice["id"]}
    gate.set()
    _finish(first)


def test_any_running_job_blocks_a_start_of_another_kind(state, gate):
    first = _start(state, _hold(gate), kind="cost_field", label="Cost field")
    assert jobs.start_job(state, "corridor", _hold(gate), label="Corridor graph") is None
    assert list(state.jobs) == ["cost_field"]
    assert "Cost field is still running" in jobs.refusal_notice(state, "corridor")["title"]
    gate.set()
    _finish(first)
    assert jobs.busy_label(state) is None        # ended: nothing runs any more


def test_the_routing_job_counts_as_busy(state):
    state.routing_job = routing_job.RoutingJob(params={})      # status "running"
    assert jobs.busy_label(state) == "Routing"
    assert jobs.start_job(state, "rasterize", lambda _j: None, label="Rasterizing") is None
    assert "Routing is still running" in jobs.refusal_notice(state, "rasterize")["title"]
    state.routing_job.status = "done"
    assert jobs.busy_label(state) is None
    assert jobs.refusal_notice(state, "rasterize") is None
    _finish(_start(state, lambda _j: None))


def test_busy_label_copes_with_a_state_that_has_no_jobs():
    assert jobs.busy_label(types.SimpleNamespace()) is None
    assert jobs.busy_label(types.SimpleNamespace(routing_job=None, jobs={"x": None})) is None


def test_a_finished_job_nobody_collected_blocks_its_kind_so_its_outcome_is_not_lost(state):
    old = _finish(_start(state, _explode))
    assert jobs.busy_label(state) is None
    assert jobs.start_job(state, "rasterize", lambda _j: None, label="Rasterizing") is None
    assert state.jobs["rasterize"] is old and old.error is not None
    notice = jobs.refusal_notice(state, "rasterize")
    assert notice["severity"] == "info" and "not been collected" in notice["title"]
    assert jobs.take_finished(state, "rasterize") is old
    assert jobs.refusal_notice(state, "rasterize") is None
    _finish(_start(state, lambda _j: None))      # the slot is free again


def test_an_uncollected_job_of_another_kind_does_not_block(state):
    _finish(_start(state, lambda _j: None, kind="corridor", label="Corridor graph"))
    other = _finish(_start(state, lambda _j: "ok"))
    assert other.status == "done" and set(state.jobs) == {"corridor", "rasterize"}


def test_simultaneous_starts_admit_exactly_one(state, gate):
    barrier = threading.Barrier(8)
    started = []

    def contender(index):
        barrier.wait(JOIN_S)
        job = jobs.start_job(state, f"kind{index % 3}", _hold(gate), label=f"Job {index}")
        if job is not None:
            started.append(job)

    threads = [threading.Thread(target=contender, args=(i,), daemon=True) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
    assert len(started) == 1 and len(state.jobs) == 1
    gate.set()
    _finish(started[0])


def test_a_thread_that_cannot_start_leaves_no_job_behind(state, monkeypatch):
    def refuse(_self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        jobs.start_job(state, "rasterize", lambda _j: None, label="Rasterizing")
    assert state.jobs == {}


# ============================================================ take_finished
def test_take_finished_pops_only_finished_jobs(state, gate):
    running = _start(state, _hold(gate), kind="rasterize")
    assert jobs.take_finished(state, "rasterize") is None and state.jobs["rasterize"] is running
    assert jobs.take_finished(state, "corridor") is None          # absent
    gate.set()
    _finish(running)
    assert jobs.take_finished(state, "rasterize") is running
    assert state.jobs == {} and jobs.take_finished(state, "rasterize") is None


@pytest.mark.parametrize("work, status", [
    (lambda _j: 1, "done"),
    (_explode, "failed"),
    (lambda _j: (_ for _ in ()).throw(jobs.JobCancelled()), "cancelled"),
])
def test_take_finished_returns_every_kind_of_ending(state, work, status):
    _finish(_start(state, work))
    assert jobs.take_finished(state, "rasterize").status == status


# ============================================================ progress text and stop
def _manual_job(monkeypatch, *, can_stop=True, label="Rasterizing"):
    monkeypatch.setattr(jobs, "_clock", lambda: 114.0)
    job = jobs.BackgroundJob(kind="rasterize", label=label, can_stop=can_stop)
    job.started = 100.0
    return job


def test_progress_text_with_stop(monkeypatch):
    job = _manual_job(monkeypatch)
    job.report(2, 5, "burn")
    assert job.progress_text() == "Rasterizing... 2/5 burn, 14 s elapsed — Stop ends it after the current step"


def test_progress_text_without_stop(monkeypatch):
    job = _manual_job(monkeypatch, can_stop=False, label="Corridor graph")
    job.report(2, 5, "burn")
    assert job.progress_text() == "Corridor graph... 2/5 burn, 14 s elapsed"


@pytest.mark.parametrize("done, total, stage, expected", [
    (0, 0, "", "Rasterizing... 14 s elapsed — Stop ends it after the current step"),
    (0, 0, "crop", "Rasterizing... crop, 14 s elapsed — Stop ends it after the current step"),
    (3, 0, "", "Rasterizing... 3, 14 s elapsed — Stop ends it after the current step"),
    (0, 4, "", "Rasterizing... 0/4, 14 s elapsed — Stop ends it after the current step"),
])
def test_progress_text_leaves_out_what_is_unknown(monkeypatch, done, total, stage, expected):
    job = _manual_job(monkeypatch)
    job.report(done, total, stage)
    assert job.progress_text() == expected


def test_progress_text_of_an_ended_job_says_how_it_ended(monkeypatch):
    job = _manual_job(monkeypatch)
    job.finished = 107.0
    for status, text in (("done", "finished after"), ("cancelled", "was stopped after"), ("failed", "failed after")):
        job.status = status
        assert job.progress_text() == f"Rasterizing {text} 7 s."


def test_stop_on_a_job_that_cannot_stop_says_so(state, gate):
    job = _start(state, _hold(gate), kind="corridor", label="Corridor graph", can_stop=False)
    for outcome in (job.request_stop(), jobs.stop_job(state, "corridor")):
        assert outcome.accepted is False
        assert outcome.text == "Corridor graph cannot be stopped; it ends by itself when it is done."
    assert not job.cancel.is_set()
    gate.set()
    _finish(job)


def test_stop_job_reaches_a_running_job_and_reports_the_other_cases(state, gate):
    assert jobs.stop_job(state, "rasterize") == (False, "Nothing is running to stop.")
    job = _start(state, _hold(gate))
    assert jobs.stop_job(state, "rasterize").accepted is True and job.cancel.is_set()
    gate.set()
    _finish(job)
    assert jobs.stop_job(state, "rasterize") == (False, "Rasterizing has already ended.")


# ============================================================ the contract with the rest of the GUI
def test_the_worker_never_touches_the_project_state(gate):
    touched = []

    class SpyState(types.SimpleNamespace):
        def __getattribute__(self, name):
            if threading.current_thread() is not threading.main_thread():
                touched.append(name)
            return super().__getattribute__(name)

    spy = SpyState(jobs={}, routing_job=None)
    job = jobs.start_job(spy, "rasterize", _hold(gate), label="Rasterizing")
    gate.set()
    _finish(job)
    assert job.status == "done" and touched == []


def test_state_clear_stops_a_real_job_and_forgets_it(state, gate):
    job = _start(state, _hold(gate))
    state.clear()
    assert job.cancel.is_set() and state.jobs == {}
    gate.set()
    _finish(job)
    assert jobs.busy_label(state) is None
    _finish(_start(state, lambda _j: None))      # a new job can start in the cleared state


def _put_recorder():
    puts: list[tuple] = []
    return puts, lambda component_id, prop, value: puts.append((component_id, prop, value))


def test_the_page_prefill_reads_a_real_job(state):
    state.jobs["rasterize"] = job = jobs.BackgroundJob(kind="rasterize", label="Rasterizing")
    job.report(1, 4, "burn")
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    assert puts == [(ids.RASTERIZE_POLL, "disabled", False), (ids.RASTERIZE_STOP_BTN, "style", {}),
                    (ids.RASTERIZE_LOG, "children", job.progress_text())]

    job.can_stop = False
    puts.clear()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    assert puts == [(ids.RASTERIZE_POLL, "disabled", False), (ids.RASTERIZE_LOG, "children", job.progress_text())]

    job.status = "done"
    puts.clear()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    assert puts == [(ids.RASTERIZE_POLL, "disabled", False)]


# ------------------------------------------------------------ tidy-up: one warning-collecting helper
def test_run_collecting_warnings_returns_the_value_and_the_notices_of_the_warnings():
    import warnings

    def work(a, b=0):
        warnings.warn("No search_space_buffer_m set - using full raster", UserWarning)
        return a + b

    value, notices = jobs.run_collecting_warnings(work, 1, b=2)
    assert value == 3
    assert [n["title"][:20] for n in notices] == ["No search buffer set"]


def test_run_collecting_warnings_lets_an_exception_of_the_function_propagate():
    def work():
        raise pex.NoPathFoundError(1, 2)

    with pytest.raises(pex.NoPathFoundError):
        jobs.run_collecting_warnings(work)


def test_run_collecting_warnings_leaves_the_warnings_of_a_failed_run_on_the_exception():
    import warnings

    def work():
        warnings.warn("something odd happened first", UserWarning)
        raise pex.NoPathFoundError(1, 2)

    with pytest.raises(pex.NoPathFoundError) as caught:
        jobs.run_collecting_warnings(work)
    assert [n["details"] for n in caught.value.collected_notices] == ["something odd happened first"]


def test_run_collecting_warnings_takes_a_keyword_called_function():
    value, notices = jobs.run_collecting_warnings(lambda **kw: kw["function"], function="x")
    assert (value, notices) == ("x", [])


def test_no_service_keeps_a_private_copy_of_the_warning_collector():
    from pathlib import Path

    from pyorps.gui import services
    folder = Path(services.__file__).parent
    for path in folder.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "_run_collecting_warnings" not in text, path.name

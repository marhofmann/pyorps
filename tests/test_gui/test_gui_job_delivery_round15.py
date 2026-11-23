"""Round 15 - the outcome of a background job reaches the page exactly once (``jobs.poll_tick``, ``job_poll``).

The browser keeps only the answer of the newest call of a callback. Before round 15 the poll tick that found an
ended job also ran the slow finalize step (adding a large raster layer takes seconds); the interval fired again
meanwhile, the browser dropped the slow tick's answer, and the newer tick found no job and switched the poll off:
the layer was in the project but never on the page. These tests drive every job kind's poll headlessly, with the
finalize step held on an Event while further ticks arrive from other threads (as waitress serves them), and check:

* the finalize step runs once, in a Dash thread, never twice at the same time;
* every tick while it runs says "adding the result" and hides Stop;
* the outputs are answered (again) until the page sends the token back, then the poll goes off for good;
* failed, stopped and empty jobs, and a finalize step that raises, are reported exactly once as well;
* a page reload during the handover gets the outcome too, and nothing else starts meanwhile.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from conftest import invoke, poll_job

from pyorps.gui import ids
from pyorps.gui.callbacks import constrained as constrained_cb
from pyorps.gui.callbacks import corridor as corridor_cb
from pyorps.gui.callbacks import cost_field as cost_field_cb
from pyorps.gui.callbacks import job_poll
from pyorps.gui.callbacks import raster as raster_cb
from pyorps.gui.callbacks import routes as routes_cb
from pyorps.gui.layout import _state_prefill, build_layout
from pyorps.gui.services import constrained, jobs, routing_job
from pyorps.gui.services import corridor as corridor_service
from pyorps.gui.services import cost_field as cost_field_service

JOIN_S = 20
HIDDEN = {"display": "none"}
EMPTY_FC = {"type": "FeatureCollection", "features": []}


# ------------------------------------------------------------------------- the five kinds
KINDS = {
    # kind: (ack store, poll, stop button or None, status id, label, outcome module + name, number of results)
    "rasterize": (ids.JOB_ACK_RASTERIZE, ids.RASTERIZE_POLL, ids.RASTERIZE_STOP_BTN, ids.RASTERIZE_LOG,
                  "Rasterizing", raster_cb, "rasterize_outcome", 3),
    "constrained": (ids.JOB_ACK_CONSTRAINED, ids.OHL_POLL, ids.OHL_STOP_BTN, ids.OHL_STATUS,
                    constrained.JOB_LABEL, constrained_cb, "constrained_outcome", 4),
    "cost_field": (ids.JOB_ACK_COST_FIELD, ids.CF_POLL, ids.CF_STOP_BTN, ids.CF_STATUS,
                   cost_field_cb.JOB_LABEL, cost_field_cb, "cost_field_outcome", 4),
    "corridor": (ids.JOB_ACK_CORRIDOR, ids.COR_POLL, None, ids.COR_STATUS,
                 corridor_service.JOB_LABEL, corridor_cb, "corridor_outcome", 2),
    "routing": (ids.JOB_ACK_ROUTING, ids.ROUTING_POLL, ids.ROUTING_STOP_BTN, ids.ROUTING_STATUS,
                jobs.ROUTING_LABEL, routes_cb, "routing_outcome", 5),
}


def _spec(kind):
    ack, poll, stop, status, label, module, name, n_results = KINDS[kind]
    return SimpleNamespace(kind=kind, ack=ack, poll=poll, stop=stop, status=status, label=label, module=module,
                           name=name, n_results=n_results,
                           slot=jobs.ROUTING_SLOT if kind == "routing" else jobs.kind_slot(kind))


def _ended_job(state, kind, *, status=jobs.DONE, result=None, error=None, params=None):
    """Put an ended job of ``kind`` into its slot, as the worker leaves it."""
    if kind == "routing":
        job = routing_job.RoutingJob(params=params or {"meta": {}})
        job.status, job.result, job.error = status, result, error
        state.routing_job = job
        return job
    job = jobs.BackgroundJob(kind=kind, label=_spec(kind).label, params=params or {})
    job.status, job.result, job.error = status, result, error
    job.finished = job.started
    state.jobs[kind] = job
    return job


def _tick(app, spec, ack=None, notices=None, n=1):
    return invoke(app, f"{spec.ack}.data", n, list(notices or []), ack, triggered=[f"{spec.poll}.n_intervals"])


def _says_finalizing(resp, spec) -> bool:
    return jobs.FINALIZING_NOTE in str(resp.get(spec.status, {}).get("children", ""))


class HeldFinalize:
    """A finalize step that waits on an Event, counts its calls and changes the project once per call."""

    def __init__(self, state, spec):
        self.state, self.spec = state, spec
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls: list[str] = []
        self.running = 0
        self.most_at_once = 0
        self._lock = threading.Lock()

    def __call__(self, state, job):
        with self._lock:
            self.calls.append(threading.current_thread().name)
            self.running += 1
            self.most_at_once = max(self.most_at_once, self.running)
        self.entered.set()
        try:
            assert self.release.wait(JOIN_S), "the test never released the finalize step"
            state.add_layer(f"result of {self.spec.kind}", "vector", geojson=EMPTY_FC)
            values = [f"{self.spec.kind} value {i}" for i in range(self.spec.n_results)]
            values[0] = state.layers_view()              # every kind's first result is the layer list
            values[_status_index(self.spec)] = f"{self.spec.kind} finished"
            notice = {"id": f"done-{self.spec.kind}", "severity": "success", "title": f"{self.spec.kind} done"}
            return tuple(values), [notice]
        finally:
            with self._lock:
                self.running -= 1


def _status_index(spec) -> int:
    return 1 if spec.kind in ("constrained", "corridor", "routing") else 2


@pytest.fixture()
def held(request, monkeypatch, state):
    spec = _spec(request.param)
    fake = HeldFinalize(state, spec)
    monkeypatch.setattr(spec.module, spec.name, fake)
    yield spec, fake
    fake.release.set()


def _in_thread(function, name):
    out: list = []
    thread = threading.Thread(target=lambda: out.append(function()), name=name, daemon=True)
    thread.start()
    return thread, out


# ========================================================== the race: a slow finalize and more ticks
@pytest.mark.parametrize("held", list(KINDS), indirect=True)
def test_a_slow_finalize_is_run_once_and_its_outcome_reaches_the_page_once(app, state, held):
    spec, fake = held
    job = _ended_job(state, spec.kind)
    page_notices = [{"id": "older", "severity": "info", "title": "an older notice"}]

    claim = _tick(app, spec, notices=page_notices)               # the tick that sees the ended job only claims it
    assert _says_finalizing(claim, spec) and spec.ack not in claim and fake.calls == []
    assert spec.poll not in claim and ids.NOTICES not in claim
    if spec.stop:
        assert claim[spec.stop] == {"style": HIDDEN}

    slow, slow_out = _in_thread(lambda: _tick(app, spec, notices=page_notices, n=2), "waitress-slow-tick")
    assert fake.entered.wait(JOIN_S)
    # the interval fires again (several times) while the finalize step runs, served on other threads
    interim = [_in_thread(lambda k=k: _tick(app, spec, notices=page_notices, n=3 + k), f"waitress-{k}")
               for k in range(4)]
    for thread, _out in interim:
        thread.join(JOIN_S)
    for _thread, out in interim:
        resp = out[0]
        assert _says_finalizing(resp, spec) and spec.ack not in resp and spec.poll not in resp
        assert ids.LAYERS_VIEW not in resp and ids.NOTICES not in resp
        if spec.stop:
            assert resp[spec.stop] == {"style": HIDDEN}
    assert jobs.busy_label(state) == spec.label                  # nothing else may start meanwhile

    fake.release.set()
    slow.join(JOIN_S)
    assert not slow.is_alive()
    first = slow_out[0]
    token = first[spec.ack]["data"]
    assert token and first[spec.status]["children"] == f"{spec.kind} finished"

    # the browser dropped that slow answer (a newer tick was sent): the next tick hands the same outcome over
    again = _tick(app, spec, notices=page_notices, n=8)
    assert again[spec.ack]["data"] == token and again[spec.status] == first[spec.status]
    assert again[ids.LAYERS_VIEW]["data"] == state.layers_view()
    assert [n["id"] for n in again[ids.NOTICES]["data"]] == ["older", f"done-{spec.kind}"]
    assert spec.poll not in again                                # the poll stays on for the receipt
    # the page applied it; the next tick carries the token back: poll off, Stop hidden, nothing else
    confirm = _tick(app, spec, ack=token, notices=again[ids.NOTICES]["data"], n=9)
    expected = {spec.poll: {"disabled": True}, **({spec.stop: {"style": HIDDEN}} if spec.stop else {})}
    assert confirm == expected
    assert _tick(app, spec, ack=token, n=10) == expected         # a later tick switches nothing on again
    assert _tick(app, spec, n=11) == expected
    assert fake.calls == ["waitress-slow-tick"] and fake.most_at_once == 1     # once, in a Dash thread
    assert [ly.name for ly in state.layers.values()].count(f"result of {spec.kind}") == 1
    assert spec.slot.get(state) is None and jobs.busy_label(state) is None
    assert job.handover.phase == jobs.READY


@pytest.mark.parametrize("held", ["rasterize", "cost_field"], indirect=True)
def test_ticks_that_race_to_claim_and_finalize_still_finalize_once(app, state, held):
    spec, fake = held
    _ended_job(state, spec.kind)
    start = threading.Barrier(6)

    def racer():
        start.wait(JOIN_S)
        return _tick(app, spec)

    racers = [_in_thread(racer, f"racer-{k}") for k in range(6)]
    assert fake.entered.wait(JOIN_S)
    fake.release.set()
    for thread, _out in racers:
        thread.join(JOIN_S)
    answers = [out[0] for _thread, out in racers]
    assert len(fake.calls) == 1 and fake.most_at_once == 1
    # each answer either says "adding the result" (claimed or finalizing at that moment) or hands over the
    # one outcome (the finalizing tick, and every tick that came after the finalize step had ended)
    tokens = {a[spec.ack]["data"] for a in answers if spec.ack in a}
    assert len(tokens) == 1
    assert all(_says_finalizing(a, spec) for a in answers if spec.ack not in a)
    assert poll_job(app, spec.ack)[spec.poll] == {"disabled": True} and len(fake.calls) == 1


@pytest.mark.parametrize("held", ["rasterize", "routing"], indirect=True)
def test_a_wrong_token_is_no_receipt_the_outcome_is_handed_over_again(app, state, held):
    spec, fake = held
    fake.release.set()
    _ended_job(state, spec.kind)
    _tick(app, spec)
    token = _tick(app, spec)[spec.ack]["data"]
    again = _tick(app, spec, ack="an older token")
    assert again[spec.ack]["data"] == token and spec.slot.get(state) is not None
    assert _tick(app, spec, ack=token)[spec.poll] == {"disabled": True} and spec.slot.get(state) is None


# ====================================================================== a page reload during the handover
@pytest.mark.parametrize("held", ["rasterize", "cost_field", "corridor", "routing"], indirect=True)
def test_a_reload_while_the_result_is_added_polls_again_and_gets_the_outcome(app, state, held):
    spec, fake = held
    _ended_job(state, spec.kind)
    _tick(app, spec)
    slow, slow_out = _in_thread(lambda: _tick(app, spec), "waitress-slow-tick")
    assert fake.entered.wait(JOIN_S)

    page = _state_prefill(state)                                  # the reloaded page while the finalize step runs
    assert page[spec.poll] == {"disabled": False}
    assert jobs.FINALIZING_NOTE in page[spec.status]["children"]
    if spec.stop:
        assert spec.stop not in page                              # Stop stays hidden (the layout default)

    fake.release.set()
    slow.join(JOIN_S)
    assert spec.ack in slow_out[0]                                # the old page's answer, lost with that page

    page = _state_prefill(state)                                  # reloaded after the finalize step
    assert page[spec.poll] == {"disabled": False}
    resp = poll_job(app, spec.ack)                                # the new page holds no token yet
    assert resp[spec.status]["children"] == f"{spec.kind} finished"
    assert [n["id"] for n in resp[ids.NOTICES]["data"]] == [f"done-{spec.kind}"]
    assert resp[spec.poll] == {"disabled": True} and spec.slot.get(state) is None
    assert len(fake.calls) == 1
    assert spec.poll not in _state_prefill(state)                 # nothing left to poll after a further reload


def test_a_prefill_of_a_handover_waiting_for_its_receipt_only_switches_the_poll_on(state):
    job = _ended_job(state, "rasterize")
    job.handover = jobs.Handover(phase=jobs.READY, results=None)
    page = _state_prefill(state)
    assert page[ids.RASTERIZE_POLL] == {"disabled": False}
    assert ids.RASTERIZE_STOP_BTN not in page and ids.RASTERIZE_LOG not in page


# ============================================================ failed, stopped, empty: reported once
def _routing_none():
    return (None, [], [])


OUTCOMES = [
    # kind, status, result, error, the text that must be reported once (None: one error notice)
    ("rasterize", jobs.FAILED, None, ValueError("Unknown preprocessor 'x'."), "Unknown preprocessor"),
    ("rasterize", jobs.CANCELLED, None, None, "Rasterizing stopped"),
    ("rasterize", jobs.DONE, None, None, None),                         # empty: the finalize step raises
    ("constrained", jobs.FAILED, None, RuntimeError("segment broke"), "segment broke"),
    ("constrained", jobs.CANCELLED, None, None, "Constrained routing stopped"),
    ("constrained", jobs.DONE, {"result": None, "notices": []}, None, "No route could be found"),
    ("cost_field", jobs.FAILED, None, RuntimeError("field broke"), "field broke"),
    ("cost_field", jobs.CANCELLED, None, None, "Cost field stopped"),
    ("cost_field", jobs.DONE, None, None, "The cost field run returned nothing"),
    ("corridor", jobs.FAILED, None, RuntimeError("graph broke"), "graph broke"),
    ("corridor", jobs.CANCELLED, None, None, "The corridor graph run was dropped"),
    ("routing", jobs.FAILED, None, RuntimeError("routing broke"), "routing broke"),
    ("routing", jobs.CANCELLED, "none", None, "Routing stopped by you"),
]


def _reports(note: dict, text: str | None) -> bool:
    return note.get("severity") == "error" if text is None else text in str(note)


@pytest.mark.parametrize(("kind", "status", "result", "error", "text"), OUTCOMES)
def test_a_failed_stopped_or_empty_job_is_reported_exactly_once(app, state, kind, status, result, error, text):
    spec = _spec(kind)
    _ended_job(state, kind, status=status, result=_routing_none() if result == "none" else result, error=error)
    notices: list[dict] = []
    reported = 0
    ack = None
    for n in range(1, 9):                                         # many ticks, as the page sends them
        resp = _tick(app, spec, ack=ack, notices=notices, n=n)
        if spec.ack in resp:
            if ack is None:                                       # the page applies the first one it gets
                ack = resp[spec.ack]["data"]
                notices = resp.get(ids.NOTICES, {}).get("data", notices)
                reported += sum(_reports(note, text) for note in notices)
        elif resp.get(spec.poll) == {"disabled": True}:
            break
    assert reported == 1 and spec.slot.get(state) is None
    assert sum(_reports(note, text) for note in notices) == 1
    assert _tick(app, spec, ack=ack, notices=notices)[spec.poll] == {"disabled": True}


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_finalize_step_that_raises_is_one_notice_and_ends_the_handover(app, state, kind, monkeypatch):
    spec = _spec(kind)

    def boom(_state, _job):
        raise RuntimeError("the layer registry broke")

    monkeypatch.setattr(spec.module, spec.name, boom)
    _ended_job(state, kind)
    resp = poll_job(app, spec.ack)
    assert [n["severity"] for n in resp[ids.NOTICES]["data"]] == ["error"]
    assert "the layer registry broke" in str(resp[ids.NOTICES]["data"])
    assert ids.LAYERS_VIEW not in resp and resp[spec.poll] == {"disabled": True}
    assert spec.slot.get(state) is None and jobs.busy_label(state) is None


def test_a_project_cleared_during_the_finalize_step_leaves_nothing_to_poll(app, state, monkeypatch):
    spec = _spec("rasterize")
    fake = HeldFinalize(state, spec)
    monkeypatch.setattr(raster_cb, "rasterize_outcome", fake)
    try:
        _ended_job(state, "rasterize")
        _tick(app, spec)
        slow, _out = _in_thread(lambda: _tick(app, spec), "waitress-slow-tick")
        assert fake.entered.wait(JOIN_S)
        state.clear()
        assert jobs.busy_label(state) is None
    finally:
        fake.release.set()
    slow.join(JOIN_S)
    assert _tick(app, spec) == {ids.RASTERIZE_POLL: {"disabled": True}, ids.RASTERIZE_STOP_BTN: {"style": HIDDEN}}
    assert len(fake.calls) == 1


# ======================================================================= busy while the result is added
def test_while_the_result_is_added_no_other_job_starts_and_after_it_only_the_same_kind_waits(app, state,
                                                                                               monkeypatch):
    spec = _spec("cost_field")
    fake = HeldFinalize(state, spec)
    monkeypatch.setattr(cost_field_cb, "cost_field_outcome", fake)
    try:
        _ended_job(state, "cost_field")
        assert jobs.busy_label(state) is None                    # ended and not claimed: as before round 15
        _tick(app, spec)
        assert jobs.busy_label(state) == "Cost field"
        assert jobs.start_job(state, "corridor", lambda _j: None, label="Corridor graph") is None
        assert jobs.refusal_notice(state, "corridor")["title"] == "Cost field is still running"
        slow, _out = _in_thread(lambda: _tick(app, spec), "waitress-slow-tick")
        assert fake.entered.wait(JOIN_S)
        assert jobs.busy_label(state) == "Cost field"
    finally:
        fake.release.set()
    slow.join(JOIN_S)
    assert jobs.busy_label(state) is None                         # ready: the project has the result
    assert jobs.refusal_notice(state, "cost_field")["title"] == "The last run has not been collected yet"
    other = jobs.start_job(state, "corridor", lambda _j: None, label="Corridor graph")
    assert other is not None
    other.thread.join(JOIN_S)


def test_the_status_line_while_the_result_is_added_names_the_job():
    text = jobs.finalizing_text("Rasterizing")
    assert text.startswith("Rasterizing has ended") and jobs.FINALIZING_NOTE in text


# ====================================================================== a start click during the handover
def _clicks(state):
    """Each kind's start click with inputs that never get past the held-job check."""
    return {
        "rasterize": ((f"{ids.RASTERIZE_LOG}.children", ids.GO_RASTERIZE),
                      (1, [], {}, [], "none", 10, 4, 2, [], 1.0, 65535, "uint16", 1, None, "viridis", [], "",
                       "current", False, False, "warn", False), f"{ids.GO_RASTERIZE}.data"),
        "constrained": ((f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN),
                        (1, True, {}, None, "", "cython", False, 8, 100, None, None, [], []),
                        f"{ids.RUN_ROUTING_BTN}.n_clicks"),
        "cost_field": ((f"{ids.CF_STATUS}.children", ids.CF_RUN_BTN),
                       (1, None, None, *[None] * len(cost_field_service.COST_FIELD_KEYS), []),
                       f"{ids.CF_RUN_BTN}.n_clicks"),
        "corridor": ((f"{ids.COR_STATUS}.children", ids.COR_RUN_BTN),
                     (1, None, None, *[None] * len(corridor_service.SETTING_KEYS), []),
                     f"{ids.COR_RUN_BTN}.n_clicks"),
        "routing": ((f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
                    (1, False, {}, None, "dijkstra", None, 1, 80, False, True, False, False, 1.0, 100, 0, None, [],
                     []), f"{ids.RUN_ROUTING_BTN}.n_clicks"),
    }


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("phase", [jobs.CLAIMED, jobs.READY, None])
def test_a_start_click_while_a_job_ended_switches_the_poll_on_and_starts_nothing(app, state, kind, phase):
    spec = _spec(kind)
    job = _ended_job(state, kind)
    if phase is not None:
        job.handover = jobs.Handover(phase=phase)
    fragments, args, trigger = _clicks(state)[kind]
    resp = invoke(app, fragments, *args, triggered=[trigger])
    titles = [n["title"] for n in resp[ids.NOTICES]["data"]]
    if phase is None:
        assert any(t.startswith("An earlier") and t.endswith("was collected") for t in titles), titles
    else:
        assert titles[-1] == "The last run has not been collected yet"
    assert resp[spec.poll] == {"disabled": False}
    assert spec.slot.get(state) is job and ids.LAYERS_VIEW not in resp     # nothing was collected or started


def test_a_start_click_while_the_job_runs_is_refused_and_leaves_the_poll_alone(app, state):
    job = _ended_job(state, "cost_field", status=jobs.RUNNING)
    fragments, args, trigger = _clicks(state)["cost_field"]
    resp = invoke(app, fragments, *args, triggered=[trigger])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Cost field is still running"
    assert ids.CF_POLL not in resp and state.jobs == {"cost_field": job}


# =========================================================================== the page wiring
def test_each_poll_has_its_own_one_shot_store_in_the_page(state):
    page = build_layout(state)
    found = [c for c in page._traverse() if getattr(c, "id", None) in ids.JOB_ACK_STORES]  # noqa: SLF001
    assert sorted(c.id for c in found) == sorted(ids.JOB_ACK_STORES) and len(set(ids.JOB_ACK_STORES)) == 5
    assert all(c.data is None for c in found)
    assert all(any(child is c for child in page.children) for c in found)      # direct children, like the polls


def test_the_new_ids_swallow_no_older_id():
    others = [v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str) and v not in ids.JOB_ACK_STORES]
    assert {new: [o for o in others if o in new or new in o] for new in ids.JOB_ACK_STORES} == {
        new: [] for new in ids.JOB_ACK_STORES}


@pytest.mark.parametrize("kind", list(KINDS))
def test_the_poll_reads_and_alone_writes_its_store(app, kind):
    spec = _spec(kind)
    writers = [s for s in app._callback_list if f"{spec.ack}.data" in str(s["output"])]      # noqa: SLF001
    assert len(writers) == 1
    poll = writers[0]
    assert [i["id"] for i in poll["inputs"]] == [spec.poll]
    assert [s["id"] for s in poll["state"]] == [ids.NOTICES, spec.ack]
    assert "running" not in poll or not poll["running"]


def test_the_three_run_callbacks_got_no_new_input_or_state(app):
    def spec_of(output, trigger):
        found = [s for s in app._callback_list                                              # noqa: SLF001
                 if output in str(s["output"]) and trigger in str(s["inputs"])]
        assert len(found) == 1
        return found[0]

    plain = spec_of(f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN)
    assert [i["id"] for i in plain["inputs"]] == [ids.RUN_ROUTING_BTN] and len(plain["state"]) == 17
    assert f"{ids.ROUTING_POLL}.disabled" in str(plain["output"])          # the one new Output
    assert plain["running"]["running"] == {f"{ids.RUN_PLAIN_BUSY}.data": True}


def test_answer_of_an_idle_tick_without_a_stop_button():
    spec = job_poll.JobPoll(slot=jobs.kind_slot("x"), label="X", poll="p", ack="a", results=(("s", "children"),),
                            status=("s", "children"), running_text=str, finalize=lambda _s, _j: ((), []))
    assert job_poll.answer(spec, None, jobs.Tick(jobs.TICK_IDLE), []) == (job_poll.no_update, job_poll.no_update,
                                                                          True, job_poll.no_update)


# ================================================================= stopping the shared tile server
class _FakeUvicorn:
    def __init__(self, tasks):
        self.server_state = SimpleNamespace(tasks=tasks)
        self.force_exit = False
        self.should_exit = False


class _FakeServerThread:
    """Stands in for ``server_thread.ServerThread``: ends once uvicorn was told to exit (unless ``stuck``)."""

    def __init__(self, uvicorn=None, *, stuck=False):
        self.srv = SimpleNamespace(server=uvicorn)
        self.stuck = stuck
        self.joined_with: list = []
        self.shutdowns = 0
        self._lts_initialized = True

    def is_alive(self):
        server = self.srv.server
        return self.stuck or server is None or not server.should_exit

    def join(self, timeout=None):
        self.joined_with.append(timeout)

    def shutdown(self):
        self.shutdowns += 1


def test_the_tile_server_stop_lets_requests_in_flight_finish_then_stops_without_waiting_for_idle_connections():
    from pyorps.gui import state as state_module

    tasks = {"a tile render"}
    uvicorn = _FakeUvicorn(tasks)
    server = _FakeServerThread(uvicorn)
    timer = threading.Timer(0.3, tasks.clear)
    timer.start()
    seen_tasks_when_stopped = []
    original = type(uvicorn).__setattr__

    def spy(self, name, value):
        if name == "force_exit" and value:
            seen_tasks_when_stopped.append(set(tasks))
        original(self, name, value)

    type(uvicorn).__setattr__ = spy
    try:
        state_module._stop_server_thread(server, wait_s=JOIN_S)
    finally:
        type(uvicorn).__setattr__ = original
        timer.cancel()
    assert seen_tasks_when_stopped == [set()]                       # the render finished first
    assert uvicorn.force_exit and uvicorn.should_exit
    assert server.joined_with and all(t is not None for t in server.joined_with)   # never an unbounded join
    assert server._lts_initialized


def test_the_tile_server_stop_gives_up_waiting_and_never_joins_for_ever():
    from pyorps.gui import state as state_module

    uvicorn = _FakeUvicorn({"a request that never ends"})
    server = _FakeServerThread(uvicorn, stuck=True)
    state_module._stop_server_thread(server, wait_s=0.2)
    assert uvicorn.force_exit and uvicorn.should_exit and server.joined_with == [state_module._STOP_SERVER_JOIN_S]
    assert server._lts_initialized is False                        # its __del__ will not join again


def test_a_wsgi_tile_server_is_stopped_with_its_own_shutdown():
    from pyorps.gui import state as state_module

    server = _FakeServerThread(None)
    state_module._stop_server_thread(server, wait_s=0)
    assert server.shutdowns == 1


def test_stop_server_takes_the_live_default_server_out_of_the_manager(monkeypatch):
    server_thread = pytest.importorskip("server_thread")
    from pyorps.gui.state import TileManager

    live = dict(server_thread.ServerManager._LIVE_SERVERS)
    uvicorn = _FakeUvicorn(set())
    fake = _FakeServerThread(uvicorn)
    monkeypatch.setattr(server_thread.ServerManager, "_LIVE_SERVERS", {**live, "default": fake})
    TileManager.stop_server()
    assert "default" not in server_thread.ServerManager._LIVE_SERVERS and uvicorn.should_exit

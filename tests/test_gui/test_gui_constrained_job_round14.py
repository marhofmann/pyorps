"""Round 14, P10 - heavy constrained runs go to a background job (Stop + poll); small runs stay as they are.

Headless, in the style of ``test_gui_raster_job_round14.py``. A run is made "heavy" by patching the named
criterion ``constrained.is_heavy`` (never by a large raster). A test that needs the worker to wait holds it on
an Event it opens itself (the worker thread is told apart by its name), and every thread is joined with a
timeout, so no thread outlives its test. The library has no cancel point inside one search, so Stop is tested
between the segments of a run with a waypoint.
"""
from __future__ import annotations

import threading
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from conftest import invoke, poll_job
from dash.exceptions import PreventUpdate
from shapely.geometry import Point

from pyorps.gui import ids
from pyorps.gui.callbacks import constrained as constrained_cb
from pyorps.gui.services import constrained, jobs, routing

CRS = "EPSG:32632"
JOIN_S = 30
WORKER = "pyorps-constrained-job"
RUN = (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
POLL = (f"{ids.OHL_POLL}.disabled", ids.OHL_POLL)
STOP = (f"{ids.OHL_STATUS}.children", ids.OHL_STOP_BTN)
HIDDEN = {"display": "none"}
HEAVY_TITLE = "Heavy constrained run ahead"

MINI_PROFILE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 20, "max_span_m": 40, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 200}, "terrain_interpolation": "linear",
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 100},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300}}}}


# ------------------------------------------------------------------ helpers
@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory):
    """A 100 x 100 m uniform raster at 1 m (the constrained tests' small reference: a run takes ~0.1 s)."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path_factory.mktemp("p10_tiny") / "uniform.tif"
    with rasterio.open(str(path), "w", driver="GTiff", height=100, width=100, count=1, dtype="uint16", crs=CRS,
                       transform=from_bounds(0, 0, 100, 100, 100, 100)) as dst:
        dst.write(np.full((100, 100), 100, dtype=np.uint16), 1)
    return str(path)


def _add_raster(state, path):
    """A raster layer without a tile server (the run only needs its source path)."""
    return state.add_layer("cost", "raster", crs=CRS, meta={"source_path": path})


@pytest.fixture()
def ohl_layer(state, tiny_raster):
    return _add_raster(state, tiny_raster)


def _point(x, y):
    return {"lat": 0, "lng": 0, "x": x, "y": y}


def _draft(*, waypoints=()):
    return {"sources": [_point(10.0, 10.0)], "targets": [_point(90.0, 90.0)],
            "waypoints": [_point(x, y) for x, y in waypoints]}


TWO_SEGMENTS = _draft(waypoints=[(50.0, 50.0)])


def _run(app, raster_id, draft=None, *, profile=None, notices=None, k=1, buffer=100):
    text = yaml.safe_dump(MINI_PROFILE) if profile is None else profile
    return invoke(app, RUN, 1, True, draft or _draft(), raster_id, text, "cython", False, k, buffer, None, None, [],
                  list(notices or []), triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _poll(app, notices=None):
    """The poll as the page runs it: one answer while the job runs; the outcome, handed over and confirmed,
    once it has ended (round 15: ``conftest.poll_job``)."""
    return poll_job(app, ids.JOB_ACK_CONSTRAINED, notices)


def _stop(app):
    return invoke(app, STOP, 1, triggered=[f"{ids.OHL_STOP_BTN}.n_clicks"])


def _titles(resp) -> list[str]:
    return [n["title"] for n in resp[ids.NOTICES]["data"]]


def _same_notice(got: dict, want: dict) -> bool:
    """Two notice dicts that differ only by their random id."""
    return {k: v for k, v in got.items() if k != "id"} == {k: v for k, v in want.items() if k != "id"}


def _join(job):
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive(), "the worker did not end"
    return job


@pytest.fixture()
def heavy(monkeypatch):
    """Every constrained run counts as heavy (the criterion, patched by name)."""
    monkeypatch.setattr(constrained, "is_heavy", lambda _est: True)


class Gate:
    """Holds the worker thread inside one segment until the test opens the gate."""

    def __init__(self, at=1):
        self.at = at                         # the 1-based segment the worker waits in
        self.calls: list[tuple] = []         # (source, target) of every segment that started
        self.reached = threading.Event()
        self.open = threading.Event()

    def wait_here(self):
        if threading.current_thread().name != WORKER or len(self.calls) != self.at:
            return                           # the Dash thread must never block
        self.reached.set()
        assert self.open.wait(JOIN_S), "the test never opened the gate"


@pytest.fixture()
def gate():
    held = Gate()
    yield held
    held.open.set()                          # a failing test leaves no blocked thread behind


def _hold(monkeypatch, held, *, at=1):
    """Every real segment is recorded in ``held.calls``; the worker waits inside segment number ``at``."""
    held.at = at
    original = constrained.run_constrained

    def segment(*args, **kwargs):
        held.calls.append((kwargs["source"], kwargs["target"]))
        held.wait_here()
        return original(*args, **kwargs)

    monkeypatch.setattr(constrained, "run_constrained", segment)
    return held


def _routes(state):
    return state.layers_of_kind("route")


def _towers(state):
    return [ly for ly in state.layers_of_kind("vector") if ly.meta.get("towers_for")]


def _spec_of(app, output_fragment: str, *input_fragments: str) -> dict:
    found = [s for s in app._callback_list                                          # noqa: SLF001
             if output_fragment in str(s["output"]) and all(f in str(s["inputs"]) + str(s["state"])
                                                            for f in input_fragments)]
    assert len(found) == 1, (output_fragment, len(found))
    return found[0]


# ======================================================== the criterion and the page wiring
def test_the_criterion_is_the_plain_routing_one_read_when_asked(monkeypatch):
    light = {"est_seconds": 1.0, "est_memory_mb": 10.0, "n_pairs": 1, "total_cells": 1000}
    assert constrained.is_heavy(light) is False
    assert constrained.is_heavy({**light, "est_seconds": routing.CONFIRM_SECONDS + 1}) is True
    assert constrained.is_heavy({**light, "est_memory_mb": routing.CONFIRM_MEMORY_MB + 1}) is True
    assert constrained.is_heavy({**light, "total_cells": routing.CONFIRM_TOTAL_CELLS + 1}) is True
    seen: list[dict] = []
    monkeypatch.setattr(routing, "needs_confirmation", lambda est: seen.append(est) or True)
    assert constrained.is_heavy(light) is True and seen == [light]


def test_an_unknown_effort_counts_as_heavy():
    assert constrained.is_heavy(None) is True


def test_the_job_kind_is_the_one_the_page_knows():
    from pyorps.gui.layout import JOB_CONTROLS

    assert constrained.JOB_KIND in JOB_CONTROLS
    assert JOB_CONTROLS[constrained.JOB_KIND] == (ids.OHL_POLL, ids.OHL_STOP_BTN, ids.OHL_STATUS)


def test_run_constrained_gets_two_outputs_and_no_new_input_or_state(app):
    """Rule 8: eleven test files call the callback positionally, so only Outputs may be added."""
    spec = _spec_of(app, f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
    assert [i["id"] for i in spec["inputs"]] == [ids.RUN_ROUTING_BTN]
    assert len(spec["state"]) == 12 and spec["state"][-1]["id"] == ids.NOTICES
    outputs = str(spec["output"])
    assert f"{ids.OHL_POLL}.disabled" in outputs and f"{ids.OHL_STOP_BTN}.style" in outputs
    assert f"{ids.RUN_OHL_BUSY}.data" not in outputs                 # still only a running= target
    assert spec["running"]["running"] == {f"{ids.RUN_OHL_BUSY}.data": True}


def test_the_busy_store_is_still_flipped_by_running_only(app):
    spec = _spec_of(app, f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
    assert spec["running"]["runningOff"] == {f"{ids.RUN_OHL_BUSY}.data": False}
    others = [s for s in app._callback_list if f"{ids.RUN_OHL_BUSY}.data" in str(s["output"])]   # noqa: SLF001
    assert others == []


def test_the_poll_and_stop_callbacks_exist_once(app):
    poll = _spec_of(app, f"{ids.OHL_POLL}.disabled", ids.OHL_POLL)
    assert [i["id"] for i in poll["inputs"]] == [ids.OHL_POLL]
    assert [s["id"] for s in poll["state"]] == [ids.NOTICES, ids.JOB_ACK_CONSTRAINED]
    stop = _spec_of(app, f"{ids.OHL_STATUS}.children", ids.OHL_STOP_BTN)
    assert [i["id"] for i in stop["inputs"]] == [ids.OHL_STOP_BTN] and not stop["state"]


# ==================================================================== the service: segments
def _segment_spy(monkeypatch, *, after=None):
    """Records every real segment; ``after(n)`` runs once segment number ``n`` has finished."""
    calls: list[tuple] = []
    original = constrained.run_constrained

    def segment(*args, **kwargs):
        calls.append((kwargs["source"], kwargs["target"]))
        out = original(*args, **kwargs)
        if after is not None:
            after(len(calls))
        return out

    monkeypatch.setattr(constrained, "run_constrained", segment)
    return calls


def _path(raster, *, points=((10.0, 10.0), (50.0, 50.0), (90.0, 90.0)), **kwargs):
    return constrained.run_constrained_path(raster, points=points, profile=MINI_PROFILE, backend="cython",
                                            neighborhood="r1", search_buffer_m=100, **kwargs)


def test_a_path_without_cancel_or_progress_is_the_same_as_with_an_unset_event(tiny_raster):
    plain = _path(tiny_raster)
    seen: list[tuple] = []
    watched = _path(tiny_raster, cancel=threading.Event(), progress=lambda *a: seen.append(a))
    assert plain["line"].equals(watched["line"]) and plain["summary"] == watched["summary"]
    assert plain["total_cost"] == watched["total_cost"] and plain["towers"].equals(watched["towers"])


def test_progress_is_reported_when_each_segment_starts(tiny_raster):
    seen: list[tuple] = []
    _path(tiny_raster, progress=lambda *a: seen.append(a))
    assert seen == [(0, 2, "segments done"), (1, 2, "segments done")]


def test_progress_fits_the_jobs_report_and_reads_as_a_segment_count(tiny_raster):
    job = jobs.BackgroundJob(kind="constrained", label="Constrained routing")
    _path(tiny_raster, progress=job.report)
    assert (job.done, job.total, job.stage) == (1, 2, "segments done")
    assert "1/2 segments done" in job.progress_text()


def test_stop_before_the_first_segment_runs_no_search(tiny_raster, monkeypatch):
    calls = _segment_spy(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(jobs.JobCancelled):
        _path(tiny_raster, cancel=cancel)
    assert calls == []


def test_stop_during_a_segment_ends_the_run_before_the_next_one(tiny_raster, monkeypatch):
    cancel = threading.Event()
    calls = _segment_spy(monkeypatch, after=lambda n: cancel.set() if n == 1 else None)
    with pytest.raises(jobs.JobCancelled):
        _path(tiny_raster, cancel=cancel)
    assert len(calls) == 1                       # the segment in flight finished, the next one never started


def test_stop_during_the_last_segment_still_returns_no_route(tiny_raster, monkeypatch):
    cancel = threading.Event()
    calls = _segment_spy(monkeypatch, after=lambda n: cancel.set() if n == 2 else None)
    with pytest.raises(jobs.JobCancelled):
        _path(tiny_raster, cancel=cancel)
    assert len(calls) == 2                       # nothing is stitched from a stopped run


def test_a_segment_without_a_path_still_answers_none_in_a_watched_run(tiny_raster, monkeypatch):
    class NoPath:
        path_geometry = None

    monkeypatch.setattr(constrained, "run_constrained", lambda *a, **k: (NoPath(), None, CRS))
    assert _path(tiny_raster, cancel=threading.Event(), progress=lambda *a: None) is None


# ================================================================= the service: snapshot, worker
def test_the_snapshot_does_not_change_when_the_project_data_does(tiny_raster):
    profile = {"a": {"b": [1, 2]}, "name": "p"}
    points = [[1, 2], [3, 4]]
    snap = constrained.snapshot_run_inputs(
        tiny_raster, points=points, profile=profile, backend="cython", neighborhood="r2",
        search_buffer_m=50.0, dem=None, dsm="dsm.tif")
    profile["a"]["b"].append(3)
    points.append([5, 6])
    assert snap["profile"] == {"a": {"b": [1, 2]}, "name": "p"}
    assert snap["points"] == [(1.0, 2.0), (3.0, 4.0)]
    assert snap["raster_source"] == tiny_raster and snap["dsm"] == "dsm.tif" and snap["dem"] is None


def _job_for(snapshot):
    return jobs.BackgroundJob(kind=constrained.JOB_KIND, label=constrained.JOB_LABEL, params={"run": snapshot})


def _snapshot(raster="r.tif"):
    return constrained.snapshot_run_inputs(
        raster, points=[(0, 0), (9, 9)], profile={"name": "p"}, backend="cython", neighborhood="r1",
        search_buffer_m=None, dem=None, dsm=None)


def test_the_worker_passes_cancel_and_progress_and_returns_the_result_and_the_warnings(monkeypatch):
    seen: dict = {}
    found = {"line": "a route"}

    def fake(source, **kwargs):
        seen.update(source=source, **kwargs)
        warnings.warn("a segment fell back to another search", UserWarning, stacklevel=1)
        return found

    monkeypatch.setattr(constrained, "run_constrained_path", fake)
    job = _job_for(_snapshot())
    out = constrained.background_run(job)
    assert out["result"] is found
    assert seen["source"] == "r.tif" and seen["points"] == [(0.0, 0.0), (9.0, 9.0)] and seen["neighborhood"] == "r1"
    assert seen["cancel"] is job.cancel and seen["progress"] == job.report
    notice = out["notices"][0]
    assert notice["title"] == "A step reported a warning" and "fell back" in notice["details"]


def test_the_worker_lets_errors_and_stops_through(monkeypatch):
    def broken(*_a, **_k):
        raise ValueError("bad profile value")

    monkeypatch.setattr(constrained, "run_constrained_path", broken)
    with pytest.raises(ValueError, match="bad profile value"):
        constrained.background_run(_job_for(_snapshot()))

    def stopped(*_a, **_k):
        raise jobs.JobCancelled("stopped")

    monkeypatch.setattr(constrained, "run_constrained_path", stopped)
    with pytest.raises(jobs.JobCancelled):
        constrained.background_run(_job_for(_snapshot()))


def test_the_status_line_names_the_segment_as_the_step_where_stop_acts():
    job = jobs.BackgroundJob(kind="constrained", label="Constrained routing", done=1, total=3, stage="segments done")
    text = constrained.status_text(job)
    assert text.startswith("Constrained routing... 1/3 segments done") and "after the segment in flight" in text
    assert "current step" not in text and "stop requested" not in text
    job.cancel.set()
    assert "stop requested" in constrained.status_text(job) and "adds no route" in constrained.status_text(job)
    job.status = jobs.DONE
    job.finished = job.started + 1
    assert constrained.status_text(job) == job.progress_text()       # an ended job: the plain line


# ===================================================== a small run stays exactly as it was
def test_a_light_run_is_synchronous_and_calls_the_search_without_cancel_or_progress(app, state, ohl_layer, monkeypatch):
    calls: list[dict] = []
    real = constrained.run_constrained_path

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(constrained, "run_constrained_path", spy)
    resp = _run(app, ohl_layer.id)
    assert len(calls) == 1 and "cancel" not in calls[0] and "progress" not in calls[0]
    assert state.jobs == {} and len(_routes(state)) == 1 and len(_towers(state)) == 1
    assert ids.OHL_POLL not in resp and ids.OHL_STOP_BTN not in resp           # no_update: untouched
    assert "towers" in resp[ids.OHL_STATUS]["children"]
    assert resp[ids.EDIT_ROUTE_SELECT]["value"] == _routes(state)[0].id == state.active_route_id
    assert HEAVY_TITLE not in _titles(resp) and "Terrain cost shown is 2-D" in _titles(resp)


def test_a_light_run_is_not_blocked_by_a_job_of_another_kind(app, state, ohl_layer):
    release = threading.Event()
    other = jobs.start_job(state, "corridor", lambda _j: release.wait(JOIN_S), label="Corridor graph", can_stop=False)
    try:
        resp = _run(app, ohl_layer.id)
        assert state.jobs == {"corridor": other} and len(_routes(state)) == 1
        assert "Corridor graph is still running" not in _titles(resp)
    finally:
        release.set()
        _join(other)


# ============================================================ a heavy run goes to a job
def test_a_heavy_run_starts_a_job_and_changes_nothing_in_the_project_yet(app, state, ohl_layer, heavy, gate,
                                                                          monkeypatch):
    _hold(monkeypatch, gate)
    before = (state.layers_view(), dict(state.layers))
    resp = _run(app, ohl_layer.id, TWO_SEGMENTS)
    job = state.jobs[constrained.JOB_KIND]
    assert (job.kind, job.label, job.can_stop, job.status) == ("constrained", "Constrained routing", True, "running")
    assert job.thread.name == WORKER and job.thread.daemon
    assert resp[ids.OHL_POLL]["disabled"] is False and resp[ids.OHL_STOP_BTN]["style"] == {}
    assert ids.LAYERS_VIEW not in resp and ids.EDIT_ROUTE_SELECT not in resp
    status = resp[ids.OHL_STATUS]["children"]
    assert status.startswith("Constrained routing...") and "after the segment in flight" in status
    assert HEAVY_TITLE in _titles(resp)
    notice = next(n for n in resp[ids.NOTICES]["data"] if n["title"] == HEAVY_TITLE)
    assert "background" in notice["impact"] and "segment in flight" in notice["impact"]
    assert "UI blocks" not in str(notice)
    assert _routes(state) == [] and _towers(state) == [] and (state.layers_view(), dict(state.layers)) == before
    gate.open.set()
    _join(job)


def test_the_heavy_notice_says_where_stop_acts_and_that_a_run_without_waypoints_is_one_segment(app, state, ohl_layer,
                                                                                               heavy):
    resp = _run(app, ohl_layer.id)
    notice = next(n for n in resp[ids.NOTICES]["data"] if n["title"] == HEAVY_TITLE)
    assert "single segment" in notice["impact"] and "adds no route" in notice["impact"]
    _join(state.jobs[constrained.JOB_KIND])


def test_the_worker_never_touches_the_project(app, state, ohl_layer, heavy, gate, monkeypatch):
    """The raster layer is deleted while the job runs; the file is the snapshot's source, the route is still found."""
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id, TWO_SEGMENTS)
    job = state.jobs[constrained.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    state.remove_layer(ohl_layer.id)
    assert state.get(ohl_layer.id) is None
    gate.open.set()
    _join(job)
    assert job.status == "done" and job.result["result"] is not None
    assert _routes(state) == []                                  # nothing is registered before the poll


def _project_digest(state):
    """What a finished run left in the project, free of random ids and of the (state-specific) layer id."""
    route = _routes(state)[0]
    towers = next(ly for ly in _towers(state) if ly.meta["towers_for"] == route.id)
    params = {k: v for k, v in route.meta["params"].items() if k != "raster_layer_id"}
    return {
        "route": (route.name, route.gdf.geometry.to_wkt().tolist(), route.meta["control_points"],
                  route.meta["metrics"], route.meta["constrained_summary"], params, route.meta["waypoint_names"]),
        "towers": (towers.name, towers.gdf.geometry.to_wkt().tolist(), list(towers.gdf.columns)),
        "active": state.active_route_id == route.id}


def test_the_poll_finalises_the_same_route_and_towers_as_a_synchronous_run(app, state, ohl_layer, monkeypatch,
                                                                            tmp_path):
    from pyorps.gui.app import build_app

    sync_state = type(state)(work_dir=tmp_path / "sync")
    try:
        sync_app = build_app(sync_state)
        sync_layer = _add_raster(sync_state, ohl_layer.meta["source_path"])
        sync_resp = _run(sync_app, sync_layer.id, TWO_SEGMENTS)                 # the reference: below the criterion
        assert sync_state.jobs == {}

        monkeypatch.setattr(constrained, "is_heavy", lambda _est: True)         # the run under test: a job
        start = _run(app, ohl_layer.id, TWO_SEGMENTS)
        job = _join(state.jobs[constrained.JOB_KIND])
        assert _routes(state) == []                                              # nothing until the poll has run
        resp = _poll(app, start[ids.NOTICES]["data"])
        assert job.status == "done" and state.jobs == {}
        assert _project_digest(state) == _project_digest(sync_state)
        # the poll answers like the synchronous run: layers, status, route select, notices
        assert resp[ids.LAYERS_VIEW]["data"] == state.layers_view()
        assert resp[ids.OHL_STATUS]["children"] == sync_resp[ids.OHL_STATUS]["children"]
        assert resp[ids.EDIT_ROUTE_SELECT]["options"] == [{"label": "OHL Route 1", "value": _routes(state)[0].id}]
        assert resp[ids.EDIT_ROUTE_SELECT]["value"] == _routes(state)[0].id
        assert resp[ids.OHL_POLL]["disabled"] is True and resp[ids.OHL_STOP_BTN]["style"] == HIDDEN
        after_start = [t for t in _titles(resp) if t != HEAVY_TITLE]
        assert after_start == _titles(sync_resp)                                 # same notices, same order
        assert HEAVY_TITLE in _titles(resp)                                      # the notice of the start is kept
    finally:
        sync_state.shutdown()


def test_a_heavy_run_with_a_waypoint_forces_a_tower_there(app, state, ohl_layer, heavy):
    start = _run(app, ohl_layer.id, TWO_SEGMENTS)
    _join(state.jobs[constrained.JOB_KIND])
    resp = _poll(app, start[ids.NOTICES]["data"])
    route = _routes(state)[0]
    assert len(route.meta["control_points"]) == 3 and route.meta["params"]["constrained"] is True
    assert route.meta["params"]["raster_layer_id"] == ohl_layer.id and route.meta["params"]["backend"] == "cython"
    towers = _towers(state)[0]
    assert towers.gdf.distance(Point(50.0, 50.0)).min() < 5.0
    assert any("1 forced at waypoints" in t for t in _titles(resp))


def test_the_job_reports_its_segments_and_the_poll_shows_them(app, state, ohl_layer, heavy, gate, monkeypatch):
    _hold(monkeypatch, gate, at=2)                    # segment 1 is done, the worker waits inside segment 2
    _run(app, ohl_layer.id, TWO_SEGMENTS)
    job = state.jobs[constrained.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    resp = _poll(app)
    status = resp[ids.OHL_STATUS]["children"]
    assert "1/2 segments done" in status and "after the segment in flight" in status
    assert set(resp) == {ids.OHL_STATUS}                                  # only the status line changes
    assert gate.calls == [((10.0, 10.0), (50.0, 50.0)), ((50.0, 50.0), (90.0, 90.0))]
    gate.open.set()
    _join(job)


# ================================================================================ Stop
def test_stop_between_segments_adds_no_route_changes_nothing_and_says_so(app, state, ohl_layer, heavy, gate,
                                                                          monkeypatch):
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id, TWO_SEGMENTS)
    job = state.jobs[constrained.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    answer = _stop(app)[ids.OHL_STATUS]["children"]
    assert answer.startswith("Stop requested") and "segment in flight" in answer and "adds no route" in answer
    assert "cannot be interrupted" in answer and job.cancel.is_set()
    running = _poll(app)[ids.OHL_STATUS]["children"]                      # the status line keeps saying it
    assert "stop requested" in running and "segment in flight" in running
    before = (state.layers_view(), dict(state.layers))
    gate.open.set()
    _join(job)
    assert job.status == "cancelled" and len(gate.calls) == 1             # the second segment never started
    resp = _poll(app)
    assert "Constrained routing stopped" in _titles(resp)
    status = resp[ids.OHL_STATUS]["children"]
    assert "stopped after" in status and "No route was added" in status
    assert resp[ids.OHL_POLL]["disabled"] is True and resp[ids.OHL_STOP_BTN]["style"] == HIDDEN
    assert ids.LAYERS_VIEW not in resp and ids.EDIT_ROUTE_SELECT not in resp
    assert (state.layers_view(), dict(state.layers)) == before and state.jobs == {}
    assert _routes(state) == [] and _towers(state) == [] and state.active_route_id is None


def test_stop_during_the_only_segment_discards_its_result(app, state, ohl_layer, heavy, gate, monkeypatch):
    """A run without waypoints is one segment: Stop cannot interrupt it, but the finished line is not added."""
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    assert _stop(app)[ids.OHL_STATUS]["children"].startswith("Stop requested")
    gate.open.set()
    _join(job)
    assert job.status == "cancelled"
    resp = _poll(app)
    assert "Constrained routing stopped" in _titles(resp) and _routes(state) == [] and _towers(state) == []


def test_a_stop_that_arrives_after_the_search_ended_still_changes_nothing(app, state, ohl_layer, heavy, monkeypatch):
    """The worker returns normally with Stop set (it came during the last segment): the run still counts as stopped."""
    def late_work(job):
        job.cancel.set()
        return {"result": {"line": "never used"}, "notices": []}

    monkeypatch.setattr(constrained, "background_run", late_work)
    _run(app, ohl_layer.id)
    _join(state.jobs[constrained.JOB_KIND])
    resp = _poll(app)
    assert "Constrained routing stopped" in _titles(resp) and _routes(state) == []


def test_the_stop_button_answers_when_nothing_runs_and_ignores_a_blank_click(app):
    assert _stop(app)[ids.OHL_STATUS]["children"] == "Nothing is running to stop."
    with pytest.raises(PreventUpdate):
        invoke(app, STOP, None, triggered=[f"{ids.OHL_STOP_BTN}.n_clicks"])


# ================================================================ failure, no path, warnings
def test_a_failed_job_ends_as_a_translated_notice_and_changes_nothing(app, state, ohl_layer, heavy, monkeypatch):
    def broken(*_args, **_kwargs):
        raise ValueError("bad profile value")

    monkeypatch.setattr(constrained, "run_constrained_path", broken)
    _run(app, ohl_layer.id)
    job = _join(state.jobs[constrained.JOB_KIND])
    assert job.status == "failed"
    resp = _poll(app)
    last = resp[ids.NOTICES]["data"][-1]
    assert last["severity"] == "error" and "bad profile value" in str(last)
    assert resp[ids.OHL_STATUS]["children"].startswith("Constrained routing failed after")
    assert resp[ids.OHL_POLL]["disabled"] is True and resp[ids.OHL_STOP_BTN]["style"] == HIDDEN
    assert ids.LAYERS_VIEW not in resp and state.jobs == {} and _routes(state) == []


def test_a_job_whose_segment_finds_no_path_ends_as_the_no_route_notice(app, state, ohl_layer, heavy, monkeypatch):
    monkeypatch.setattr(constrained, "run_constrained_path", lambda *_a, **_k: None)
    _run(app, ohl_layer.id)
    _join(state.jobs[constrained.JOB_KIND])
    resp = _poll(app)
    assert _titles(resp)[-1] == "No route could be found"
    assert _same_notice(resp[ids.NOTICES]["data"][-1], constrained_cb.no_route_notice())
    assert "No route was found" in resp[ids.OHL_STATUS]["children"]
    assert resp[ids.OHL_POLL]["disabled"] is True and state.jobs == {} and _routes(state) == []


def test_the_warnings_of_a_heavy_run_reach_the_user_through_the_result(app, state, ohl_layer, heavy, monkeypatch):
    def warns(*_args, **_kwargs):
        warnings.warn("a backend fell back to cython", UserWarning, stacklevel=1)
        return None

    monkeypatch.setattr(constrained, "run_constrained_path", warns)
    _run(app, ohl_layer.id)
    _join(state.jobs[constrained.JOB_KIND])
    resp = _poll(app)
    flagged = [n for n in resp[ids.NOTICES]["data"] if n["title"] == "A step reported a warning"]
    assert len(flagged) == 1 and "fell back to cython" in flagged[0]["details"]


def test_a_failed_estimate_counts_as_heavy_and_goes_to_a_job(app, state, ohl_layer, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("estimate exploded")

    monkeypatch.setattr(routing, "estimate_routing", boom)
    monkeypatch.setattr(constrained, "run_constrained_path", lambda *_a, **_k: None)
    resp = _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    notice = next(n for n in resp[ids.NOTICES]["data"] if n["title"] == "Could not estimate the effort of this run")
    assert notice["severity"] == "warning" and "estimate exploded" in notice["details"]
    assert "background" in notice["impact"] and "waits until it finishes" not in notice["impact"]
    assert HEAVY_TITLE not in _titles(resp)                   # without a prognosis there are no numbers to show
    assert resp[ids.OHL_POLL]["disabled"] is False
    _join(job)


# =============================================================== run-time profile validation
def test_a_profile_limit_problem_is_an_error_notice_naming_the_field_and_no_run(app, state, ohl_layer, heavy,
                                                                                monkeypatch):
    calls: list = []
    monkeypatch.setattr(constrained, "run_constrained_path", lambda *a, **k: calls.append(k))
    bad = yaml.safe_dump({**MINI_PROFILE, "hard_angle_limit_deg": 200.0})      # past the limit of 180 deg
    resp = _run(app, ohl_layer.id, profile=bad)
    notice = resp[ids.NOTICES]["data"][-1]
    assert notice["severity"] == "error" and "hard_angle_limit_deg" in notice["meaning"]
    assert notice["focus_control"] == ids.OHLF_IDS["hard_angle_limit_deg"] and notice["focus_id"] == ids.TAB_ROUTES
    assert "No overhead-line route was computed" in notice["impact"]
    assert calls == [] and state.jobs == {} and _routes(state) == []          # neither a job nor a synchronous run
    assert ids.OHL_POLL not in resp and ids.OHL_STOP_BTN not in resp


def test_the_older_profile_rules_still_answer_first(app, state, ohl_layer):
    bad = yaml.safe_dump({**MINI_PROFILE, "soft_angle_limit_deg": 30, "hard_angle_limit_deg": 10})
    resp = _run(app, ohl_layer.id, profile=bad)
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Profile setting invalid" and _routes(state) == []


def test_a_valid_profile_passes_the_new_check_without_a_notice(app, state, ohl_layer):
    resp = _run(app, ohl_layer.id)
    assert all(n["severity"] != "error" for n in resp[ids.NOTICES]["data"]) and len(_routes(state)) == 1


# =============================================================================== busy
def test_a_second_click_while_the_job_runs_gets_the_busy_notice_and_starts_nothing(app, state, ohl_layer, heavy,
                                                                                    gate, monkeypatch):
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id, TWO_SEGMENTS)
    job = state.jobs[constrained.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    for draft in (None, TWO_SEGMENTS):
        resp = _run(app, ohl_layer.id, draft)
        assert _same_notice(resp[ids.NOTICES]["data"][-1], jobs.busy_notice("Constrained routing"))
        assert ids.LAYERS_VIEW not in resp and ids.OHL_POLL not in resp and ids.OHL_STOP_BTN not in resp
        assert state.jobs == {constrained.JOB_KIND: job}
    gate.open.set()
    _join(job)


def test_the_busy_click_is_refused_before_anything_is_validated(app, state, ohl_layer, heavy, gate, monkeypatch):
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    resp = _run(app, "no-such-layer")
    assert [n["title"] for n in resp[ids.NOTICES]["data"]][-1] == "Constrained routing is still running"
    gate.open.set()
    _join(job)


def test_a_heavy_run_is_refused_while_a_job_of_another_kind_runs(app, state, ohl_layer, heavy):
    release = threading.Event()
    other = jobs.start_job(state, "corridor", lambda _j: release.wait(JOIN_S), label="Corridor graph", can_stop=False)
    try:
        resp = _run(app, ohl_layer.id)
        assert _same_notice(resp[ids.NOTICES]["data"][-1], jobs.busy_notice("Corridor graph"))
        assert constrained.JOB_KIND not in state.jobs and ids.OHL_POLL not in resp
        assert HEAVY_TITLE not in _titles(resp)               # nothing was started, so nothing runs in the background
    finally:
        release.set()
        _join(other)


def test_a_heavy_run_is_refused_while_plain_routing_runs(app, state, ohl_layer, heavy):
    state.routing_job = SimpleNamespace(status="running")
    resp = _run(app, ohl_layer.id)
    assert _same_notice(resp[ids.NOTICES]["data"][-1], jobs.busy_notice(jobs.ROUTING_LABEL))
    assert state.jobs == {} and ids.OHL_POLL not in resp


def test_an_ended_job_nobody_polled_is_collected_by_the_next_click(app, state, ohl_layer, heavy):
    _run(app, ohl_layer.id)
    job = _join(state.jobs[constrained.JOB_KIND])
    assert job.status == "done" and _routes(state) == []                  # the page never polled it
    resp = _run(app, ohl_layer.id)
    assert "An earlier constrained run was collected" in _titles(resp)
    # round 15: the click switches the poll on and the poll hands the outcome over (no new run was started)
    assert resp[ids.OHL_POLL]["disabled"] is False and resp[ids.OHL_STOP_BTN]["style"] == HIDDEN
    assert state.jobs == {constrained.JOB_KIND: job} and _routes(state) == []
    resp = _poll(app, resp[ids.NOTICES]["data"])
    assert resp[ids.OHL_POLL]["disabled"] is True and resp[ids.OHL_STOP_BTN]["style"] == HIDDEN
    assert len(_routes(state)) == 1 and state.jobs == {}                  # collected
    assert "towers" in resp[ids.OHL_STATUS]["children"]
    assert resp[ids.EDIT_ROUTE_SELECT]["value"] == _routes(state)[0].id


def test_an_ended_stopped_job_nobody_polled_is_collected_as_stopped(app, state, ohl_layer, heavy, monkeypatch):
    monkeypatch.setattr(constrained, "background_run", lambda job: job.cancel.set())
    _run(app, ohl_layer.id)
    _join(state.jobs[constrained.JOB_KIND])
    resp = _run(app, ohl_layer.id)
    assert resp[ids.OHL_POLL]["disabled"] is False                        # round 15: the poll collects it
    resp = _poll(app, resp[ids.NOTICES]["data"])
    assert "Constrained routing stopped" in _titles(resp) and _routes(state) == [] and state.jobs == {}


# ============================================================================ the poll
def test_a_poll_tick_without_a_job_switches_the_poll_off(app, state):
    resp = _poll(app)
    assert resp == {ids.OHL_POLL: {"disabled": True}, ids.OHL_STOP_BTN: {"style": HIDDEN}}
    assert state.jobs == {}


def test_a_poll_tick_after_the_project_was_cleared_switches_the_poll_off(app, state, ohl_layer, heavy, gate,
                                                                         monkeypatch):
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    state.clear()
    assert job.cancel.is_set() and state.jobs == {}
    assert _poll(app) == {ids.OHL_POLL: {"disabled": True}, ids.OHL_STOP_BTN: {"style": HIDDEN}}
    gate.open.set()
    _join(job)
    assert _routes(state) == []


def test_a_poll_tick_leaves_the_notices_alone_while_the_job_runs(app, state, ohl_layer, heavy, gate, monkeypatch):
    _hold(monkeypatch, gate)
    _run(app, ohl_layer.id)
    assert gate.reached.wait(JOIN_S)
    resp = _poll(app, [{"title": "kept"}])
    assert ids.NOTICES not in resp and ids.OHL_POLL not in resp
    gate.open.set()
    _join(state.jobs[constrained.JOB_KIND])

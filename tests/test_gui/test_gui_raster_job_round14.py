"""Round 14, P7 - heavy rasterize runs go to a background job (Stop + poll); small runs stay as they are.

Headless, in the style of ``test_gui_raster_settings_round13.py``. A run is made "heavy" by patching the
threshold name ``rasterize.RASTERIZE_BACKGROUND_CELLS`` to a tiny number, never by building a large raster.
A test that needs the worker to wait holds it on an Event it opens itself (the worker thread is told apart by
its name), and every thread is joined with a timeout, so no thread outlives its test.
"""
from __future__ import annotations

import sys
import threading
import warnings
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from conftest import invoke, poll_job
from dash.exceptions import PreventUpdate
from shapely.geometry import box

from pyorps import GeoRasterizer
from pyorps.gui import ids
from pyorps.gui.callbacks import raster as raster_cb
from pyorps.gui.services import cost_model, geo, jobs, rasterize

CRS = "EPSG:25832"
JOIN_S = 10
WORKER = "pyorps-rasterize-job"
BUILD = (f"{ids.RASTERIZE_LOG}.children", ids.GO_RASTERIZE)
POLL = (f"{ids.RASTERIZE_POLL}.disabled", ids.RASTERIZE_POLL)
STOP = (f"{ids.RASTERIZE_LOG}.children", ids.RASTERIZE_STOP_BTN)
HIDDEN = {"display": "none"}
KEYS = ("nutzart", "bez")


# ------------------------------------------------------------------ helpers
def _landuse_gdf():
    return gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Wald", "Weg", "Sumpf"], "bez": ["Nadelholz", "Laubholz", "", ""]},
        geometry=[box(0, 0, 50, 50), box(50, 0, 100, 50), box(0, 50, 50, 100), box(50, 50, 100, 100)], crs=CRS)


def _add_vector(state, name, gdf):
    return state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs, geojson=geo.gdf_to_wgs84_geojson(gdf), meta={})


def _inputs_for(state):
    """A loaded vector layer + a seeded cost table, ready for "Build"."""
    layer = _add_vector(state, "landuse", _landuse_gdf())
    rows = cost_model.grid_rows_from_assumptions(cost_model.seed_assumptions(layer.gdf, KEYS), KEYS)
    return layer, rows, {"dataset_id": layer.id, "feature_keys": list(KEYS)}


@pytest.fixture()
def inputs(state):
    return _inputs_for(state)


@pytest.fixture()
def heavy(monkeypatch):
    """Every build with more than a handful of cells counts as heavy (the threshold, patched by name)."""
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKGROUND_CELLS", 100)


@pytest.fixture()
def no_tiles(monkeypatch):
    """Skip serving the built raster (a tile server per test would be slow): the build itself still runs."""
    monkeypatch.setattr(raster_cb, "add_raster_layer", lambda state, source, notices, **kw: (None, notices))


def _build(app, inputs, *, modifiers=None, save_path=None, resolution=1.0, notices=None, go=1):
    _layer, rows, grid_state = inputs
    return invoke(
        app, BUILD,
        go, rows, grid_state, modifiers or [], "none", 10, 4, 2, [], resolution, 65535, "uint16", 1,
        save_path, "viridis", list(notices or []), "", "current", False, False, "warn", False,
        triggered=[f"{ids.GO_RASTERIZE}.data"])


def _poll(app, notices=None):
    """The poll as the page runs it: one answer while the job runs; the outcome, handed over and confirmed,
    once it has ended (round 15: ``conftest.poll_job``)."""
    return poll_job(app, ids.JOB_ACK_RASTERIZE, notices)


def _stop(app):
    return invoke(app, STOP, 1, triggered=[f"{ids.RASTERIZE_STOP_BTN}.n_clicks"])


def _titles(resp) -> list[str]:
    return [n["title"] for n in resp[ids.NOTICES]["data"]]


def _same_notice(got: dict, want: dict) -> bool:
    """Two notice dicts that differ only by their random id."""
    return {k: v for k, v in got.items() if k != "id"} == {k: v for k, v in want.items() if k != "id"}


def _join(job):
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive(), "the worker did not end"
    return job


def _rasters_in(state) -> list:
    return sorted(state.work_dir.glob("*.tif")) if state.work_dir.exists() else []


class Gate:
    """Holds the build's worker thread at one point until the test opens the gate."""

    def __init__(self):
        self.reached = threading.Event()
        self.open = threading.Event()

    def wait_here(self):
        if threading.current_thread().name != WORKER:       # the Dash thread must never block
            return
        self.reached.set()
        assert self.open.wait(JOIN_S), "the test never opened the gate"


@pytest.fixture()
def gate():
    held = Gate()
    yield held
    held.open.set()                  # a failing test leaves no blocked thread behind


@pytest.fixture()
def hold_before_the_burn(monkeypatch, gate):
    """The worker waits before its first stage check (the digest of the build is its first step)."""
    original = rasterize._config_target

    def target(**kwargs):
        gate.wait_here()
        return original(**kwargs)

    monkeypatch.setattr(rasterize, "_config_target", target)
    return gate


@pytest.fixture()
def hold_after_the_burn(monkeypatch, gate):
    """The worker waits right after the base layer is burnt."""
    original = GeoRasterizer.rasterize

    def burn(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        gate.wait_here()
        return result

    monkeypatch.setattr(GeoRasterizer, "rasterize", burn)
    return gate


def _build_kwargs(tmp_path, **overrides):
    gdf = _landuse_gdf()
    kwargs = {"base_gdf": gdf, "assumptions": {"Wald": 100, "Weg": 300, "Sumpf": 400}, "feature_keys": ("nutzart",),
              "resolution_in_m": 1.0, "work_dir": tmp_path / "out", "use_cache": False}
    kwargs.update(overrides)
    return kwargs


def _modifier():
    return rasterize.ModifierSpec(gdf=_landuse_gdf().iloc[:1], mode="multiply", factor=2.0, name="m", condition="c")


# ======================================================== the threshold and the page wiring
def test_the_threshold_decides_by_name_and_is_read_when_asked(monkeypatch):
    limit = rasterize.RASTERIZE_BACKGROUND_CELLS
    assert rasterize.is_heavy(limit + 1) and not rasterize.is_heavy(limit)
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKGROUND_CELLS", 5)
    assert rasterize.is_heavy(6) and not rasterize.is_heavy(5)


def test_the_job_kind_is_the_one_the_page_knows():
    from pyorps.gui.layout import JOB_CONTROLS

    assert rasterize.JOB_KIND in JOB_CONTROLS
    assert JOB_CONTROLS[rasterize.JOB_KIND][:2] == (ids.RASTERIZE_POLL, ids.RASTERIZE_STOP_BTN)


def _spec_of(app, output_fragment: str, *input_fragments: str) -> dict:
    found = [s for s in app._callback_list                                          # noqa: SLF001
             if output_fragment in str(s["output"]) and all(f in str(s["inputs"]) + str(s["state"])
                                                            for f in input_fragments)]
    assert len(found) == 1, (output_fragment, len(found))
    return found[0]


def test_run_rasterize_gets_two_outputs_and_no_new_input_or_state(app):
    """Rule 8: eleven test files call the callback positionally, so only Outputs may be added."""
    spec = _spec_of(app, f"{ids.RASTERIZE_LOG}.children", ids.GO_RASTERIZE)
    assert [i["id"] for i in spec["inputs"]] == [ids.GO_RASTERIZE]
    assert len(spec["state"]) == 21
    outputs = str(spec["output"])
    assert f"{ids.RASTERIZE_POLL}.disabled" in outputs and f"{ids.RASTERIZE_STOP_BTN}.style" in outputs
    assert f"{ids.RASTERIZE_BUSY}.data" not in outputs               # still only a running= target
    assert spec["running"]["running"] == {f"{ids.RASTERIZE_BUSY}.data": True}


def test_the_poll_and_stop_callbacks_exist_once(app):
    poll = _spec_of(app, f"{ids.RASTERIZE_POLL}.disabled", ids.RASTERIZE_POLL)
    assert [i["id"] for i in poll["inputs"]] == [ids.RASTERIZE_POLL]
    assert [s["id"] for s in poll["state"]] == [ids.NOTICES, ids.JOB_ACK_RASTERIZE]
    stop = _spec_of(app, f"{ids.RASTERIZE_LOG}.children", ids.RASTERIZE_STOP_BTN)
    assert [i["id"] for i in stop["inputs"]] == [ids.RASTERIZE_STOP_BTN] and not stop["state"]


# ==================================================================== the service: stages
def test_a_build_without_cancel_or_progress_builds_as_always(tmp_path):
    path, log = rasterize.build_cost_raster(**_build_kwargs(tmp_path), modifiers=[_modifier()])
    assert path.endswith(".tif") and any("saved" in line for line in log)
    with rasterio.open(path) as src:
        assert src.read(1).shape == (100, 100)


def test_progress_is_reported_when_each_stage_starts(tmp_path):
    seen: list[tuple] = []
    rasterize.build_cost_raster(**_build_kwargs(tmp_path), modifiers=[_modifier()], crop_border=True,
                                progress=lambda *args: seen.append(args))
    assert [(done, total) for done, total, _stage in seen] == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert [stage.split(" ")[0] for *_x, stage in seen] == ["burning", "modifier", "cropping", "saving"]


def test_an_empty_modifier_is_not_a_stage(tmp_path):
    empty = rasterize.ModifierSpec(gdf=_landuse_gdf().iloc[:0], name="none")
    seen: list[tuple] = []
    rasterize.build_cost_raster(**_build_kwargs(tmp_path), modifiers=[empty], progress=lambda *a: seen.append(a))
    assert [(done, total) for done, total, _stage in seen] == [(0, 2), (1, 2)]


@pytest.mark.parametrize("stop_during, spied", [
    ("burning", {"modify_raster_from_dataset", "shrink_raster", "save_raster"}),
    ("modifier", {"shrink_raster", "save_raster"}),
    ("cropping", {"save_raster"}),
])
def test_stop_ends_the_build_at_the_next_stage_boundary_and_saves_nothing(tmp_path, monkeypatch, stop_during, spied):
    called: list[str] = []
    for name in ("modify_raster_from_dataset", "shrink_raster", "save_raster"):
        original = getattr(GeoRasterizer, name)
        monkeypatch.setattr(GeoRasterizer, name,
                            lambda self, *a, _n=name, _o=original, **k: (called.append(_n), _o(self, *a, **k))[1])
    cancel = threading.Event()

    def progress(_done, _total, stage):
        if stage.startswith(stop_during):
            cancel.set()                          # Stop pressed while this stage starts: it still finishes

    kwargs = _build_kwargs(tmp_path)
    with pytest.raises(jobs.JobCancelled):
        rasterize.build_cost_raster(**kwargs, modifiers=[_modifier()], crop_border=True, cancel=cancel,
                                    progress=progress)
    assert not spied & set(called)
    assert not list((tmp_path / "out").glob("*.tif"))


def test_stop_before_the_base_burn_never_touches_the_rasterizer(tmp_path, monkeypatch):
    burnt: list = []
    monkeypatch.setattr(GeoRasterizer, "rasterize", lambda self, *a, **k: burnt.append(1))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(jobs.JobCancelled):
        rasterize.build_cost_raster(**_build_kwargs(tmp_path), cancel=cancel)
    assert not burnt and not list((tmp_path / "out").glob("*.tif"))


def test_a_failed_save_leaves_no_half_written_cache_file(tmp_path, monkeypatch):
    def broken_save(self, save_path):
        with open(save_path, "wb") as handle:
            handle.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(GeoRasterizer, "save_raster", broken_save)
    kwargs = _build_kwargs(tmp_path, use_cache=True)
    with pytest.raises(OSError, match="disk full"):
        rasterize.build_cost_raster(**kwargs)
    assert not list((tmp_path / "out").glob("*.tif"))        # an unfinished cost_<hash>.tif would be a "cache hit"


def test_a_failed_save_keeps_a_file_that_was_there_before(tmp_path, monkeypatch):
    target = tmp_path / "mine.tif"
    target.write_bytes(b"old")

    def broken_save(self, save_path):
        raise OSError("disk full")

    monkeypatch.setattr(GeoRasterizer, "save_raster", broken_save)
    with pytest.raises(OSError):
        rasterize.build_cost_raster(**_build_kwargs(tmp_path, save_path=str(target)))
    assert target.read_bytes() == b"old"


def test_cached_raster_path_follows_the_build(tmp_path):
    kwargs = _build_kwargs(tmp_path, use_cache=True)
    assert rasterize.cached_raster_path(**kwargs) is None
    path, _log = rasterize.build_cost_raster(**kwargs)
    assert rasterize.cached_raster_path(**kwargs) == path
    assert rasterize.cached_raster_path(**{**kwargs, "resolution_in_m": 2.0}) is None
    assert rasterize.cached_raster_path(**{**kwargs, "save_path": str(tmp_path / "x.tif")}) is None
    assert rasterize.cached_raster_path(**{**kwargs, "use_cache": False}) is None


# ============================================================== the service: the worker's inputs
def test_the_snapshot_does_not_change_when_the_project_data_does(tmp_path):
    kwargs = _build_kwargs(tmp_path, modifiers=[_modifier()], preprocessor_steps=[{"op": "buffer"}])
    snap = rasterize.snapshot_build_inputs(kwargs)
    kwargs["base_gdf"].loc[0, "nutzart"] = "Changed"
    kwargs["modifiers"][0].gdf.loc[0, "nutzart"] = "Changed"
    kwargs["assumptions"]["Wald"] = 1
    kwargs["preprocessor_steps"][0]["op"] = "other"
    assert snap["base_gdf"].loc[0, "nutzart"] == "Wald" and snap["modifiers"][0].gdf.loc[0, "nutzart"] == "Wald"
    assert snap["assumptions"]["Wald"] == 100 and snap["preprocessor_steps"] == [{"op": "buffer"}]
    assert snap["base_gdf"] is not kwargs["base_gdf"] and snap["work_dir"] == kwargs["work_dir"]


def _fake_job(build, cancel=None):
    return SimpleNamespace(params={"build": build}, cancel=cancel or threading.Event(), report=lambda *a: None)


def test_the_worker_returns_the_path_the_log_and_the_warnings_it_raised(monkeypatch):
    def fake_build(**_kwargs):
        warnings.warn("a thin feature was widened", UserWarning, stacklevel=1)
        return "built.tif", ["saved built.tif"]

    monkeypatch.setattr(rasterize, "build_cost_raster", fake_build)
    result = rasterize.background_build(_fake_job({}))
    assert (result["path"], result["log"]) == ("built.tif", ["saved built.tif"])
    assert [n["title"] for n in result["notices"]] == ["A step reported a warning"]
    assert "thin feature" in result["notices"][0]["details"]


def test_the_worker_passes_cancel_and_progress_and_lets_errors_through(monkeypatch):
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        raise ValueError("boom")

    monkeypatch.setattr(rasterize, "build_cost_raster", fake_build)
    job = _fake_job({"resolution_in_m": 3.0})
    with pytest.raises(ValueError, match="boom"):
        rasterize.background_build(job)
    assert seen["resolution_in_m"] == 3.0 and seen["cancel"] is job.cancel and seen["progress"] is job.report


# ===================================================== a small run stays exactly as it was
def test_a_small_run_is_synchronous_and_calls_the_build_without_cancel_or_progress(app, state, inputs, monkeypatch):
    calls: list[dict] = []
    real = rasterize.build_cost_raster

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(rasterize, "build_cost_raster", spy)
    resp = _build(app, inputs)
    assert len(calls) == 1 and "cancel" not in calls[0] and "progress" not in calls[0]
    assert state.jobs == {} and len(state.layers_of_kind("raster")) == 1
    assert ids.RASTERIZE_POLL not in resp and ids.RASTERIZE_STOP_BTN not in resp     # no_update: untouched
    assert "saved" in resp[ids.RASTERIZE_LOG]["children"]
    assert resp[ids.LAYERS_VIEW]["data"] and resp[ids.MAP_VIEW]["data"]["seq"] == 1


def test_a_run_below_the_threshold_stays_synchronous_even_when_a_job_of_another_kind_runs(app, state, inputs, gate):
    other = jobs.start_job(state, "corridor", lambda _j: gate.open.wait(JOIN_S), label="Corridor graph", can_stop=False)
    assert other is not None
    resp = _build(app, inputs)
    assert state.jobs == {"corridor": other} and len(state.layers_of_kind("raster")) == 1
    assert "Corridor graph is still running" not in _titles(resp)
    gate.open.set()
    _join(other)


# ============================================================ a heavy run goes to a job
def test_a_heavy_run_starts_a_job_and_changes_nothing_in_the_project_yet(app, state, inputs, heavy, no_tiles, gate,
                                                                          monkeypatch):
    monkeypatch.setattr(rasterize, "_config_target", _held_target(gate))
    resp = _build(app, inputs)
    job = state.jobs[rasterize.JOB_KIND]
    assert (job.kind, job.label, job.can_stop, job.status) == ("rasterize", "Rasterizing", True, "running")
    assert job.thread.name == WORKER and job.thread.daemon
    assert resp[ids.RASTERIZE_POLL]["disabled"] is False and resp[ids.RASTERIZE_STOP_BTN]["style"] == {}
    assert ids.LAYERS_VIEW not in resp and ids.MAP_VIEW not in resp
    assert resp[ids.RASTERIZE_LOG]["children"].startswith("Rasterizing...")
    assert "Stop ends it" in resp[ids.RASTERIZE_LOG]["children"]
    assert "Rasterizing runs in the background" in _titles(resp)
    notice = next(n for n in resp[ids.NOTICES]["data"] if n["title"] == "Rasterizing runs in the background")
    assert "stage" in notice["impact"]                          # the notice says where Stop takes effect
    assert state.layers_of_kind("raster") == [] and _rasters_in(state) == []
    gate.open.set()
    _join(job)


def _held_target(gate):
    original = rasterize._config_target

    def target(**kwargs):
        gate.wait_here()
        return original(**kwargs)

    return target


def test_the_worker_never_touches_the_project(app, state, inputs, heavy, no_tiles, gate, monkeypatch, tmp_path):
    """The user edits and deletes the layer while the job runs; the raster is still the one of the data at the start."""
    layer, rows, _grid = inputs
    expected_path, _log = rasterize.build_cost_raster(
        base_gdf=_landuse_gdf(), assumptions=cost_model.assumptions_from_grid_rows(rows, KEYS), feature_keys=KEYS,
        resolution_in_m=1.0, fill_value=65535, dtype="uint16", geometry_buffer_m=1.0, preprocessor="none",
        work_dir=tmp_path / "reference")
    monkeypatch.setattr(rasterize, "_config_target", _held_target(gate))
    _build(app, inputs)
    job = state.jobs[rasterize.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    layer.gdf.loc[:, "nutzart"] = "Sumpf"                             # edit in place ...
    state.remove_layer(layer.id)                                      # ... and delete the layer
    gate.open.set()
    _join(job)
    assert job.status == "done"
    built = _read(job.result["path"])
    assert np.array_equal(built, _read(expected_path)) and (built != 65535).any()


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def test_the_poll_finalises_the_same_layer_and_meta_as_a_synchronous_run(app, state, inputs, monkeypatch, tmp_path):
    from pyorps.gui.app import build_app

    sync_state = type(state)(work_dir=tmp_path / "sync")
    try:
        sync_app = build_app(sync_state)
        sync_inputs = _inputs_for(sync_state)
        monkeypatch.setattr(rasterize, "RASTERIZE_BACKGROUND_CELLS", sys.maxsize)       # the reference: synchronous
        sync_resp = _build(sync_app, sync_inputs)
        assert sync_state.jobs == {}
        expected = sync_state.layers_of_kind("raster")[0]

        monkeypatch.setattr(rasterize, "RASTERIZE_BACKGROUND_CELLS", 100)               # the run under test: a job
        start = _build(app, inputs)
        job = _join(state.jobs[rasterize.JOB_KIND])
        assert state.layers_of_kind("raster") == []                  # nothing until the poll has run
        resp = _poll(app, start[ids.NOTICES]["data"])
        layers = state.layers_of_kind("raster")
        assert len(layers) == 1 and state.jobs == {} and job.status == "done"
        got = layers[0]
        assert got.name == expected.name == "Cost raster (1 m)"
        assert got.meta["rasterize_config"] == {**expected.meta["rasterize_config"], "dataset_id": inputs[0].id}
        got_combo, want_combo = got.meta["combination_inputs"], expected.meta["combination_inputs"]
        assert set(got_combo) == set(want_combo)
        assert got_combo["feature_keys"] == want_combo["feature_keys"] == KEYS
        assert got_combo["assumptions"] == want_combo["assumptions"]
        assert got_combo["base_layer_name"] == want_combo["base_layer_name"] == "landuse"
        assert got_combo["base_gdf"].equals(want_combo["base_gdf"])
        assert np.array_equal(_read(got.meta["source_path"]), _read(expected.meta["source_path"]))
        # the poll answers like the synchronous run: layers, fit the map, the log, the notices
        assert resp[ids.LAYERS_VIEW]["data"] == state.layers_view()
        assert resp[ids.MAP_VIEW]["data"] == {"fit_bounds": got.tile.bounds, "seq": 1}
        log = resp[ids.RASTERIZE_LOG]["children"].replace(str(state.work_dir), "<w>")
        assert log == sync_resp[ids.RASTERIZE_LOG]["children"].replace(str(sync_state.work_dir), "<w>")
        assert resp[ids.RASTERIZE_POLL]["disabled"] is True and resp[ids.RASTERIZE_STOP_BTN]["style"] == HIDDEN
        assert f"Raster layer '{got.name}' added" in _titles(resp)
        assert "Rasterizing runs in the background" in _titles(resp)         # the notice of the start is kept
    finally:
        sync_state.shutdown()


def test_the_job_reports_its_stages_and_the_poll_shows_them(app, state, inputs, heavy, no_tiles, hold_after_the_burn):
    modifier = {"dataset": "landuse", "mode": "multiply", "column": "nutzart", "operator": "==", "value": "Wald",
                "factor": 2, "buffer_m": 0}
    _build(app, inputs, modifiers=[modifier])
    job = state.jobs[rasterize.JOB_KIND]
    assert hold_after_the_burn.reached.wait(JOIN_S)
    resp = _poll(app)
    text = resp[ids.RASTERIZE_LOG]["children"]
    assert "0/3" in text and "burning the base layer" in text and "Stop ends it" in text
    assert set(resp) == {ids.RASTERIZE_LOG}                            # a running tick changes nothing else
    hold_after_the_burn.open.set()
    _join(job)
    assert (job.done, job.total) == (2, 3) and job.stage == "saving the raster"


# ================================================================================ Stop
@pytest.mark.parametrize("held", ["hold_before_the_burn", "hold_after_the_burn"])
def test_stop_builds_no_raster_changes_nothing_and_says_so(held, app, state, inputs, heavy, no_tiles, request):
    gate = request.getfixturevalue(held)
    _build(app, inputs)
    job = state.jobs[rasterize.JOB_KIND]
    assert gate.reached.wait(JOIN_S)
    answer = _stop(app)[ids.RASTERIZE_LOG]["children"]
    assert answer.startswith("Stop requested") and "stage" in answer and "builds nothing" in answer
    assert job.cancel.is_set()
    running = _poll(app)[ids.RASTERIZE_LOG]["children"]                 # the status line keeps saying it
    assert "stop requested" in running
    before = (state.layers_view(), dict(state.layers))
    gate.open.set()
    _join(job)
    assert job.status == "cancelled"
    resp = _poll(app)
    assert "Rasterizing stopped" in _titles(resp) and "No raster was built" in resp[ids.RASTERIZE_LOG]["children"]
    assert "stopped after" in resp[ids.RASTERIZE_LOG]["children"]
    assert resp[ids.RASTERIZE_POLL]["disabled"] is True and resp[ids.RASTERIZE_STOP_BTN]["style"] == HIDDEN
    assert ids.LAYERS_VIEW not in resp and ids.MAP_VIEW not in resp
    assert (state.layers_view(), dict(state.layers)) == before and state.jobs == {}
    assert state.layers_of_kind("raster") == [] and _rasters_in(state) == []      # no half-written raster either


def test_a_stop_that_arrives_after_the_last_stage_still_changes_nothing(app, state, inputs, heavy, no_tiles,
                                                                         monkeypatch):
    """The worker returns normally with Stop set (it came during the final save): the run still counts as stopped."""
    def late_work(job):
        job.cancel.set()
        return {"path": "late.tif", "log": ["saved late.tif"], "notices": []}

    monkeypatch.setattr(rasterize, "background_build", late_work)
    _build(app, inputs)
    _join(state.jobs[rasterize.JOB_KIND])
    resp = _poll(app)
    assert "Rasterizing stopped" in _titles(resp) and state.layers_of_kind("raster") == []


def test_the_stop_button_answers_when_nothing_runs_and_ignores_a_blank_click(app):
    assert _stop(app)[ids.RASTERIZE_LOG]["children"] == "Nothing is running to stop."
    with pytest.raises(PreventUpdate):
        invoke(app, STOP, None, triggered=[f"{ids.RASTERIZE_STOP_BTN}.n_clicks"])


# ============================================================================= failure
def test_a_failed_job_ends_as_a_translated_notice_and_changes_nothing(app, state, inputs, heavy, no_tiles, monkeypatch):
    def broken(**_kwargs):
        raise ValueError("Unknown preprocessor 'x'.")

    monkeypatch.setattr(rasterize, "build_cost_raster", broken)
    _build(app, inputs)
    job = _join(state.jobs[rasterize.JOB_KIND])
    assert job.status == "failed"
    resp = _poll(app)
    last = resp[ids.NOTICES]["data"][-1]
    assert last["severity"] == "error" and "Unknown preprocessor" in str(last)
    assert resp[ids.RASTERIZE_LOG]["children"].startswith("Rasterizing failed after")
    assert resp[ids.RASTERIZE_POLL]["disabled"] is True and resp[ids.RASTERIZE_STOP_BTN]["style"] == HIDDEN
    assert ids.LAYERS_VIEW not in resp and state.jobs == {} and state.layers_of_kind("raster") == []


def test_the_warnings_of_a_heavy_build_reach_the_user_through_the_result(app, state, inputs, heavy, no_tiles,
                                                                           monkeypatch):
    def warns(**_kwargs):
        warnings.warn("a forbidden feature was dropped", UserWarning, stacklevel=1)
        return str(state.work_dir / "x.tif"), ["saved"]

    monkeypatch.setattr(rasterize, "build_cost_raster", warns)
    _build(app, inputs)
    _join(state.jobs[rasterize.JOB_KIND])
    resp = _poll(app)
    flagged = [n for n in resp[ids.NOTICES]["data"] if n["title"] == "A step reported a warning"]
    assert len(flagged) == 1 and "forbidden feature was dropped" in flagged[0]["details"]


# =============================================================================== busy
def test_a_second_build_while_one_runs_gets_the_busy_notice_and_starts_nothing(app, state, inputs, heavy, no_tiles,
                                                                               hold_before_the_burn):
    _build(app, inputs)
    job = state.jobs[rasterize.JOB_KIND]
    for resolution in (1.0, 0.5):                                       # small or heavy, the click is refused
        resp = _build(app, inputs, resolution=resolution)
        assert _titles(resp)[-1] == "Rasterizing is still running"
        assert ids.LAYERS_VIEW not in resp and ids.RASTERIZE_POLL not in resp
        assert state.jobs == {rasterize.JOB_KIND: job}
    hold_before_the_burn.open.set()
    _join(job)


def test_the_busy_refusal_is_the_job_services_own_notice(app, state, inputs, heavy, no_tiles, hold_before_the_burn):
    _build(app, inputs)
    resp = _build(app, inputs)
    assert _same_notice(resp[ids.NOTICES]["data"][-1], jobs.busy_notice("Rasterizing"))
    hold_before_the_burn.open.set()
    _join(state.jobs[rasterize.JOB_KIND])


def test_a_heavy_run_is_refused_while_a_job_of_another_kind_runs(app, state, inputs, heavy, no_tiles, gate):
    other = jobs.start_job(state, "constrained", lambda _j: gate.open.wait(JOIN_S), label="Constrained routing")
    resp = _build(app, inputs)
    assert _same_notice(resp[ids.NOTICES]["data"][-1], jobs.busy_notice("Constrained routing"))
    assert rasterize.JOB_KIND not in state.jobs and ids.RASTERIZE_POLL not in resp
    gate.open.set()
    _join(other)


def test_a_heavy_run_is_refused_while_plain_routing_runs(app, state, inputs, heavy, no_tiles):
    state.routing_job = SimpleNamespace(status="running")
    resp = _build(app, inputs)
    assert _same_notice(resp[ids.NOTICES]["data"][-1], jobs.busy_notice(jobs.ROUTING_LABEL))
    assert state.jobs == {}


def test_an_ended_job_nobody_polled_is_collected_by_the_next_build_click(app, state, inputs, heavy, no_tiles):
    _build(app, inputs)
    job = _join(state.jobs[rasterize.JOB_KIND])
    assert job.status == "done"                                         # the page never polled it
    resp = _build(app, inputs)
    assert "An earlier rasterize run was collected" in _titles(resp)
    # round 15: the click switches the poll on and the poll hands the outcome over (no new run was started)
    assert resp[ids.RASTERIZE_POLL]["disabled"] is False and resp[ids.RASTERIZE_STOP_BTN]["style"] == HIDDEN
    assert state.jobs == {rasterize.JOB_KIND: job} and ids.LAYERS_VIEW not in resp
    resp = _poll(app, resp[ids.NOTICES]["data"])
    assert resp[ids.RASTERIZE_POLL]["disabled"] is True and resp[ids.RASTERIZE_STOP_BTN]["style"] == HIDDEN
    assert "saved" in resp[ids.RASTERIZE_LOG]["children"] and state.jobs == {}


def test_the_busy_store_is_still_flipped_by_running_only(app):
    spec = _spec_of(app, f"{ids.RASTERIZE_LOG}.children", ids.GO_RASTERIZE)
    assert spec["running"]["runningOff"] == {f"{ids.RASTERIZE_BUSY}.data": False}
    others = [s for s in app._callback_list if f"{ids.RASTERIZE_BUSY}.data" in str(s["output"])]   # noqa: SLF001
    assert others == []


# ============================================================================ the poll
def test_a_poll_tick_without_a_job_switches_the_poll_off(app, state):
    resp = _poll(app)
    assert resp == {ids.RASTERIZE_POLL: {"disabled": True}, ids.RASTERIZE_STOP_BTN: {"style": HIDDEN}}
    assert state.jobs == {}


def test_a_poll_tick_after_the_project_was_cleared_switches_the_poll_off(app, state, inputs, heavy, no_tiles,
                                                                         hold_before_the_burn):
    _build(app, inputs)
    job = state.jobs[rasterize.JOB_KIND]
    state.clear()
    assert job.cancel.is_set() and state.jobs == {}
    assert _poll(app)[ids.RASTERIZE_POLL] == {"disabled": True}
    hold_before_the_burn.open.set()
    _join(job)


# ==================================================================== the cache stays synchronous
def test_a_cache_hit_stays_synchronous_even_for_a_heavy_run(app, state, inputs, heavy, no_tiles):
    _build(app, inputs)
    first = _join(state.jobs[rasterize.JOB_KIND])
    _poll(app)
    assert state.jobs == {} and len(_rasters_in(state)) == 1 and first.status == "done"
    resp = _build(app, inputs)                                          # same inputs, the file is there
    assert state.jobs == {} and ids.RASTERIZE_POLL not in resp
    assert "cache hit" in resp[ids.RASTERIZE_LOG]["children"]


def test_an_explicit_save_path_is_always_rebuilt_so_a_heavy_run_still_uses_a_job(app, state, inputs, heavy, no_tiles,
                                                                                  tmp_path):
    target = tmp_path / "chosen" / "cost.tif"
    _build(app, inputs, save_path=str(target))
    _join(state.jobs[rasterize.JOB_KIND])
    _poll(app)
    assert target.exists()
    _build(app, inputs, save_path=str(target))
    assert rasterize.JOB_KIND in state.jobs                             # no cache for an explicit path
    _join(state.jobs[rasterize.JOB_KIND])


def test_a_problem_before_the_build_is_answered_on_the_spot_not_in_a_job(app, state, heavy, no_tiles):
    resp = invoke(app, BUILD, 1, [], {}, [], "none", 10, 4, 2, [], 1.0, 65535, "uint16", 0, None, "viridis", [], "",
                  "current", False, False, "warn", False, triggered=[f"{ids.GO_RASTERIZE}.data"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Seed a cost table first"
    assert state.jobs == {}

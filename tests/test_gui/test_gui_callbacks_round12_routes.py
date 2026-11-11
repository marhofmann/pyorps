"""Round 12, routes/raster/layers callbacks: the audited robustness defects.

Headless callback tests (the harness in ``conftest.py`` drives the callbacks
exactly as Dash would). Findings: #1 reload prefill, #4 job reaping, #11
``running=``, #16 non-numeric cost, #17 guarded service calls, #18 route
dropdown / study area follow the layers, #19 one notice per failed constrained
run, #21 table click rules, #22 zero is a value, #35 failed estimate = heavy.
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from conftest import SOURCE, TARGET, TEST_CRS, invoke
from dash.exceptions import PreventUpdate
from shapely.geometry import LineString, box

from pyorps.core import exceptions as pex
from pyorps.gui import ids
from pyorps.gui.callbacks import layers as layers_cb
from pyorps.gui.callbacks.routes import add_route_layer
from pyorps.gui.layout import OVERLAY_STATUS, build_layout
from pyorps.gui.services import combinations, constrained, cost_model, geo, rasterize, routing, routing_job
from pyorps.gui.services.cost import RouteCost

RUN_WIRING = (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN)
OHL_WIRING = (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
RASTERIZE_WIRING = (f"{ids.RASTERIZE_LOG}.children", ids.GO_RASTERIZE)
CRS = "EPSG:25832"


# ------------------------------------------------------------------ helpers
def _components(layout) -> dict:
    """``{string id: component}`` of a built layout."""
    return {c.id: c for c in layout._traverse()                  # noqa: SLF001
            if isinstance(getattr(c, "id", None), str)}


def _draft(source=SOURCE, target=TARGET):
    return {"sources": [{"lat": 0, "lng": 0, "x": source[0], "y": source[1]}],
            "targets": [{"lat": 0, "lng": 0, "x": target[0], "y": target[1]}],
            "waypoints": []}


def _new_notices(resp) -> list[dict]:
    return resp[ids.NOTICES]["data"]


def _run_args(raster_id, *, search_buffer=80, delta=100, threads=0,
              algorithm="dijkstra", notices=None, draft=None):
    return (1, False, draft or _draft(), raster_id, algorithm, None, 1,
            search_buffer, False, True, False, False, 1.0, delta, threads,
            None, [], list(notices or []))


def _run(app, raster_id, **kwargs):
    return invoke(app, RUN_WIRING, *_run_args(raster_id, **kwargs),
                  triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


@pytest.fixture()
def raster_layer(app, state, raster_path):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    return state.layers_of_kind("raster")[0]


@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory):
    """A 100 x 100 m uniform raster (the constrained tests' small reference)."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path_factory.mktemp("r12_tiny") / "uniform.tif"
    with rasterio.open(
            str(path), "w", driver="GTiff", height=100, width=100, count=1,
            dtype="uint16", crs=TEST_CRS,
            transform=from_bounds(0, 0, 100, 100, 100, 100)) as dst:
        dst.write(np.full((100, 100), 100, dtype=np.uint16), 1)
    return str(path)


def _landuse_gdf():
    return gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Wald", "Weg", "Sumpf"],
         "bez": ["Nadelholz", "Laubholz", "", ""]},
        geometry=[box(0, 0, 50, 50), box(50, 0, 100, 50), box(0, 50, 50, 100),
                  box(50, 50, 100, 100)], crs=CRS)


def _add_vector(state, name, gdf):
    return state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs,
                           geojson=geo.gdf_to_wgs84_geojson(gdf))


def _route(state, name="Route 1", *, crs=TEST_CRS):
    return add_route_layer(
        state, line=LineString([SOURCE, TARGET]),
        route_cost=RouteCost(total_length=226.0, total_cost=300.0,
                             total_cell_cost=200.0, geodesic_length_m=226.0),
        params={}, control_points=[list(SOURCE), list(TARGET)], crs=crs,
        name=name)


# ================================================== #1 reload prefill (layout)
def test_layout_from_a_state_with_layers_fills_the_layers_store(state):
    _add_vector(state, "Landuse", _landuse_gdf())
    state.project_crs = "EPSG:25833"
    components = _components(build_layout(state))
    assert components[ids.LAYERS_VIEW].data == state.layers_view()
    assert [row["name"] for row in components[ids.LAYERS_VIEW].data] == ["Landuse"]
    assert components[ids.PROJECT_CRS].value == "EPSG:25833"
    assert "1 layer" in components[ids.PROJECT_STATUS].children
    # dataset list (the data tab) is written from the same helper as live loads
    assert "Landuse" in json.dumps(
        components[ids.DATASET_LIST].children, default=str)


def test_layout_shows_study_area_route_dropdown_and_the_active_route(state):
    state.study_area = {"type": "Polygon", "coordinates": [[
        [9.0, 50.0], [9.1, 50.0], [9.1, 50.1], [9.0, 50.0]]]}
    state.add_layer("Study area", "study_area", layer_id="study-area",
                    geojson={"type": "FeatureCollection", "features": []})
    route = _route(state)
    state.active_route_id = route.id
    components = _components(build_layout(state))
    assert components[ids.EDIT_ROUTE_SELECT].options == [
        {"label": "Route 1", "value": route.id}]
    assert components[ids.EDIT_ROUTE_SELECT].value == route.id
    assert len(components[ids.ACTIVE_ROUTE].positions) >= 2      # highlight
    assert len(components[ids.CONTROL_MARKERS].children) == 2    # src + tgt
    assert [r["kind"] for r in components[ids.BUILD_POINTS_TABLE].rowData] \
        == ["source", "target"]
    assert components[ids.UI_STATE].data["active_route_id"] == route.id
    assert components[ids.STUDY_AREA_INFO].children != "No study area drawn."


def test_layout_of_a_fresh_state_is_exactly_the_blank_layout(state, monkeypatch):
    from dash._utils import to_json

    import pyorps.gui.layout as layout_module

    with_state = to_json(build_layout(state))
    monkeypatch.setattr(layout_module, "_state_prefill", lambda _state: {})
    assert with_state == to_json(build_layout(state))


def test_a_failing_prefill_step_does_not_take_the_page_down(state, monkeypatch):
    import pyorps.gui.layout as layout_module

    _add_vector(state, "Landuse", _landuse_gdf())

    def broken(_state, _put):
        raise RuntimeError("boom")

    monkeypatch.setattr(layout_module, "_prefill_data_tab", broken)
    # the other steps still run: layers are in the store, the page is built
    components = _components(build_layout(state))
    assert [r["name"] for r in components[ids.LAYERS_VIEW].data] == ["Landuse"]


def test_page_load_serves_the_project_and_the_first_render_ships_the_map(app, state):
    """The real path: /_dash-layout (page load), then render_host."""
    layer = _add_vector(state, "Landuse", _landuse_gdf())
    wiring = ("layer-host.children", ids.LAYERS_VIEW)
    first = invoke(app, wiring, [])                      # the old page
    assert first[ids.LAYER_HOST]["children"]
    with pytest.raises(PreventUpdate):
        invoke(app, wiring, [])                          # unchanged: skipped

    response = app.server.test_client().get("/_dash-layout")     # F5
    assert response.status_code == 200

    def find(node, wanted):
        if isinstance(node, dict):
            if (node.get("props") or {}).get("id") == wanted:
                return node
            for value in node.values():
                hit = find(value, wanted)
                if hit is not None:
                    return hit
        elif isinstance(node, list):
            for value in node:
                hit = find(value, wanted)
                if hit is not None:
                    return hit
        return None

    served = json.loads(response.get_data(as_text=True))
    store = find(served, ids.LAYERS_VIEW)
    assert [row["id"] for row in store["props"]["data"]] == [layer.id]
    again = invoke(app, wiring, store["props"]["data"])  # first call, new page
    assert again[ids.LAYER_HOST]["children"] == first[ids.LAYER_HOST]["children"]


# ============================================ #4 a job that outlived the page
def test_layout_with_a_running_job_enables_the_poll_and_shows_stop(state):
    blank = _components(build_layout(state))
    assert blank[ids.ROUTING_POLL].disabled is True
    assert blank[ids.ROUTING_STOP_BTN].style == {"display": "none"}

    job = routing_job.RoutingJob(params={})
    job.total_pairs = 3
    state.routing_job = job
    components = _components(build_layout(state))
    assert components[ids.ROUTING_POLL].disabled is False
    assert components[ids.ROUTING_STOP_BTN].style == {}
    assert "0/3" in components[ids.ROUTING_STATUS].children


def test_layout_with_a_finished_job_polls_once_more_but_hides_stop(state):
    job = routing_job.RoutingJob(params={})
    job.status = "done"
    state.routing_job = job
    components = _components(build_layout(state))
    assert components[ids.ROUTING_POLL].disabled is False   # collects it
    assert components[ids.ROUTING_STOP_BTN].style == {"display": "none"}


def test_run_still_refuses_while_a_job_really_runs(app, state, raster_layer):
    state.routing_job = routing_job.RoutingJob(params={})
    resp = _run(app, raster_layer.id)
    assert "press ⏹ Stop first" in resp[ids.ROUTING_STATUS]["children"]
    assert state.routing_job is not None


def test_run_reaps_a_job_that_failed_while_nobody_polled(app, state):
    job = routing_job.RoutingJob(params={"meta": {}})
    job.status, job.error = "failed", RuntimeError("boom")
    state.routing_job = job
    resp = _run(app, None)
    assert state.routing_job is None                          # not "forever"
    titles = [n["title"] for n in _new_notices(resp)]
    assert "An earlier routing run was collected" in titles
    assert "Something went wrong" in titles                  # the job's error
    assert resp[ids.ROUTING_STATUS]["children"] == "routing failed"
    assert resp[ids.ROUTING_CONFIRM_MODAL]["is_open"] is False


def test_run_reaps_a_finished_job_and_adds_its_routes(app, state, raster_layer,
                                                      raster_path):
    params = {
        "raster_path": raster_path,
        "kwargs": dict(sources=[SOURCE], targets=[TARGET], waypoints=[],
                       algorithm="delta-stepping", graph_api="cython",
                       neighborhood="r1", pairwise=False,
                       search_buffer_m=100.0, ignore_max_cost=True,
                       delta=100.0, num_threads=0),
        "meta": {"raster_layer_id": raster_layer.id,
                 "algorithm": "delta-stepping", "simplify_tol": None,
                 "wp_names": [], "n_sources": 1, "n_targets": 1,
                 "n_waypoints": 0},
    }
    job = routing_job.start_routing_job(state, params)
    job.thread.join(timeout=180)
    assert job.status == "done"
    resp = _run(app, raster_layer.id)
    assert state.routing_job is None
    assert len(state.layers_of_kind("route")) == 1
    assert "1 route(s) built" in resp[ids.ROUTING_STATUS]["children"]


# ================================================ #22 zero is a value, not blank
@pytest.fixture()
def raster_inputs(state):
    """A loaded vector layer + a seeded cost table, ready for "Build"."""
    layer = _add_vector(state, "landuse", _landuse_gdf())
    keys = ("nutzart", "bez")
    rows = cost_model.grid_rows_from_assumptions(
        cost_model.seed_assumptions(layer.gdf, keys), keys)
    return layer, rows, {"dataset_id": layer.id, "feature_keys": list(keys)}


@pytest.fixture()
def build_calls(monkeypatch):
    """Replace the rasterizer: record the kwargs, build nothing."""
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(rasterize, "build_cost_raster", fake_build)
    return calls


def _rasterize(app, rows, grid_state, *, preproc="none", buffers=(10, 4, 2),
               resolution=1.0, fill_value=65535, dtype="uint16",
               geom_buffer=0):
    return invoke(
        app, RASTERIZE_WIRING,
        1, rows, grid_state, [], preproc, *buffers, [], resolution, fill_value,
        dtype, geom_buffer, None, "viridis", [], "", "current",
        False, False, "warn", False,
        triggered=[f"{ids.GO_RASTERIZE}.data"])


def test_fill_value_zero_reaches_the_rasterizer_as_zero(app, raster_inputs,
                                                         build_calls):
    _layer, rows, grid_state = raster_inputs
    _rasterize(app, rows, grid_state, fill_value=0)
    assert len(build_calls) == 1
    assert build_calls[0]["fill_value"] == 0           # was 65535 = forbidden
    assert isinstance(build_calls[0]["fill_value"], int)


def test_zero_buffers_stay_zero_and_a_blank_field_gets_its_default(
        app, raster_inputs, build_calls):
    _layer, rows, grid_state = raster_inputs
    _rasterize(app, rows, grid_state, preproc="street_buffer",
               buffers=(0, None, 0), geom_buffer=0)
    params = build_calls[0]["preprocessor_params"]
    assert params == {"buffer_a": 0.0, "buffer_b": 4.0, "buffer_l": 0.0}
    assert build_calls[0]["geometry_buffer_m"] == 0.0


def test_blank_fill_value_still_means_forbidden(app, raster_inputs, build_calls):
    _layer, rows, grid_state = raster_inputs
    _rasterize(app, rows, grid_state, fill_value=None)
    assert build_calls[0]["fill_value"] == 65535


@pytest.mark.parametrize("field,kwargs,control", [
    ("Geometry buffer", {"geom_buffer": -5}, ids.GEOM_BUFFER),
    ("Resolution", {"resolution": 0}, ids.RES_M),
    ("Resolution", {"resolution": -2}, ids.RES_M),
    ("Fill value", {"fill_value": 70000}, ids.FILL_VALUE),
    ("Fill value", {"fill_value": -1}, ids.FILL_VALUE),
    ("Fill value", {"fill_value": 1.5}, ids.FILL_VALUE),
    ("Street buffer B", {"preproc": "street_buffer",
                         "buffers": (10, -4, 2)}, ids.PREPROC_BUF_B),
])
def test_unusable_raster_fields_are_named_and_nothing_is_built(
        app, raster_inputs, build_calls, field, kwargs, control):
    _layer, rows, grid_state = raster_inputs
    resp = _rasterize(app, rows, grid_state, **kwargs)
    assert build_calls == []                                   # no service call
    notice = _new_notices(resp)[-1]
    assert field in notice["title"] and notice["focus_control"] == control
    assert notice["focus_id"] == ids.TAB_RASTER


def test_fill_value_must_fit_the_dtype(app, raster_inputs, build_calls):
    _layer, rows, grid_state = raster_inputs
    _rasterize(app, rows, grid_state, fill_value=70000, dtype="uint32")
    assert build_calls[0]["fill_value"] == 70000      # fits uint32, no notice


def test_a_non_numeric_cost_cell_is_a_notice_not_a_raw_error(
        app, raster_inputs, build_calls):
    _layer, rows, grid_state = raster_inputs
    rows = [dict(r) for r in rows]
    rows[0]["cost"] = "abc"                                     # #16
    resp = _rasterize(app, rows, grid_state)
    assert build_calls == []
    notice = _new_notices(resp)[-1]
    assert notice["title"] == "A cost is not a number"
    assert "abc" in notice["meaning"]


@pytest.fixture()
def run_calls(monkeypatch):
    calls: list[dict] = []

    def fake_run(_path, **kwargs):
        calls.append(kwargs)
        return None, [], []

    monkeypatch.setattr(routing, "run_routing", fake_run)
    return calls


@pytest.mark.parametrize("kwargs,field,control", [
    ({"search_buffer": -5}, "Search buffer", ids.SEARCH_BUFFER),
    ({"delta": 0, "algorithm": "delta-stepping"}, "delta", "route-delta"),
    ({"delta": -3, "algorithm": "delta-stepping"}, "delta", "route-delta"),
    # threads only act in the CPU delta-stepping kernel; elsewhere the field is
    # disabled in the page and not checked
    ({"threads": -1, "algorithm": "delta-stepping"}, "threads",
     "route-num-threads"),
    ({"threads": 1.5, "algorithm": "delta-stepping"}, "threads",
     "route-num-threads"),
])
def test_unusable_routing_fields_are_named_and_nothing_runs(
        app, raster_layer, run_calls, kwargs, field, control):
    resp = _run(app, raster_layer.id, **kwargs)
    assert run_calls == []                                     # no service call
    notice = _new_notices(resp)[-1]
    assert field in notice["title"] and notice["focus_control"] == control
    assert notice["focus_id"] == ids.TAB_ROUTES


def test_zero_threads_and_blank_buffer_pass_through(app, raster_layer,
                                                    run_calls):
    _run(app, raster_layer.id, search_buffer=None, threads=0, delta=250)
    assert run_calls[0]["num_threads"] == 0
    assert run_calls[0]["search_buffer_m"] is None
    assert run_calls[0]["delta"] == 250.0


# ===================================== #35 a failed estimate counts as heavy
def test_a_failing_estimate_leads_to_the_confirmation_modal(
        app, state, raster_layer, run_calls, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("estimate exploded")

    monkeypatch.setattr(routing, "estimate_routing", boom)
    resp = _run(app, raster_layer.id)
    assert resp[ids.ROUTING_CONFIRM_MODAL]["is_open"] is True      # modal path
    assert run_calls == []                       # nothing started synchronously
    assert state.pending_routing is not None
    notice = _new_notices(resp)[-1]
    assert notice["severity"] == "warning"
    assert "estimate" in notice["title"].lower()
    assert "estimate exploded" in notice["details"]


# ===================================== #19 / #17 / #35 constrained routing
MINI_PROFILE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0,
    "angle_cost_function": "linear", "angle_cost_params": {"scale": 10.0},
    "min_span_m": 20, "max_span_m": 40, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {
        "terrain_cost_map": {"0": 100, "500": 200},
        "terrain_interpolation": "linear",
        "angle_types": {
            "suspension": {"max_angle_deg": 20.0, "base_cost": 100},
            "strain": {"max_angle_deg": 60.0, "base_cost": 300}}},
}


@pytest.fixture()
def ohl_layer(app, state, tiny_raster):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, tiny_raster,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    return state.layers_of_kind("raster")[0]


def _ohl(app, raster_id, *, search_buffer=100):
    import yaml

    draft = _draft((10.0, 10.0), (90.0, 90.0))
    return invoke(
        app, OHL_WIRING, 1, True, draft, raster_id,
        yaml.safe_dump(MINI_PROFILE), "cython", False, 1, search_buffer,
        None, None, [], [], triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


@pytest.mark.parametrize("effect,titles", [
    (MemoryError("no ram"), ["Out of memory"]),
    (pex.NoPathFoundError(3, 9), ["No route could be found"]),
    (None, ["No route could be found"]),            # a run without a path
])
def test_a_failed_constrained_run_gives_exactly_one_notice(
        app, ohl_layer, monkeypatch, effect, titles):
    def fake(*_args, **_kwargs):
        if effect is not None:
            raise effect
        return None

    monkeypatch.setattr(constrained, "run_constrained_path", fake)
    resp = _ohl(app, ohl_layer.id)
    assert [n["title"] for n in _new_notices(resp)] == titles


def test_a_crash_is_not_blamed_on_the_profile(app, ohl_layer, monkeypatch):
    def crash(*_args, **_kwargs):
        raise ValueError("bad profile value")

    monkeypatch.setattr(constrained, "run_constrained_path", crash)
    resp = _ohl(app, ohl_layer.id)
    notices = _new_notices(resp)
    assert len(notices) == 1
    assert "loosen" not in json.dumps(notices).lower()


def test_constrained_estimate_failure_is_announced(app, ohl_layer, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("estimate exploded")

    monkeypatch.setattr(routing, "estimate_routing", boom)
    monkeypatch.setattr(constrained, "run_constrained_path",
                        lambda *a, **k: None)
    resp = _ohl(app, ohl_layer.id)
    titles = [n["title"] for n in _new_notices(resp)]
    assert "Could not estimate the effort of this run" in titles


def test_constrained_negative_buffer_is_named(app, ohl_layer, monkeypatch):
    calls = []
    monkeypatch.setattr(constrained, "run_constrained_path",
                        lambda *a, **k: calls.append(k))
    resp = _ohl(app, ohl_layer.id, search_buffer=-1)
    assert calls == []
    assert _new_notices(resp)[-1]["focus_control"] == ids.SEARCH_BUFFER


def test_missing_raster_file_is_a_notice_for_both_runners(
        app, state, raster_layer, tmp_path):
    raster_layer.meta["source_path"] = str(tmp_path / "gone.tif")
    resp = _run(app, raster_layer.id)
    assert _new_notices(resp)[-1]["title"] == "The raster file is missing"
    resp = _ohl(app, raster_layer.id)
    assert _new_notices(resp)[-1]["title"] == "The raster file is missing"


def test_a_route_that_cannot_be_displayed_is_not_left_behind(state, monkeypatch):
    def broken(_layer):
        raise RuntimeError("no display line")

    monkeypatch.setattr(routing, "refresh_route_display", broken)
    with pytest.raises(RuntimeError):
        _route(state)
    assert state.layers_of_kind("route") == []


# =================================================== #17 edit / groups guard
def test_renaming_a_route_whose_display_fails_gives_a_notice(
        app, state, monkeypatch):
    route = _route(state)

    def broken(_layer):
        raise RuntimeError("no display line")

    monkeypatch.setattr(routing, "refresh_route_display", broken)
    resp = invoke(
        app, ("layers-view.data", "routes-grid", "cellValueChanged"),
        {"data": {"id": route.id, "name": "Renamed"}, "colId": "name"}, [],
        triggered=[f"{ids.ROUTES_GRID}.cellValueChanged"])
    assert _new_notices(resp)[-1]["title"] == "Something went wrong"


def test_simplify_failure_restores_the_old_tolerance(app, state, monkeypatch):
    route = _route(state)
    state.active_route_id = route.id
    route.meta["simplify_tol"] = 2.0

    def broken(_layer):
        raise RuntimeError("no display line")

    monkeypatch.setattr(routing, "refresh_route_display", broken)
    resp = invoke(app, ("layers-view.data", ids.EDIT_SIMPLIFY_BTN), 1, 5.0, [],
                  triggered=[f"{ids.EDIT_SIMPLIFY_BTN}.n_clicks"])
    assert route.meta["simplify_tol"] == 2.0
    assert _new_notices(resp)[-1]["title"] == "Something went wrong"


def test_negative_simplify_tolerance_is_named(app, state):
    route = _route(state)
    state.active_route_id = route.id
    resp = invoke(app, ("layers-view.data", ids.EDIT_SIMPLIFY_BTN), 1, -1, [],
                  triggered=[f"{ids.EDIT_SIMPLIFY_BTN}.n_clicks"])
    assert _new_notices(resp)[-1]["focus_control"] == ids.EDIT_SIMPLIFY_TOL


# ====================== #18 the study area and the route dropdown follow layers
def test_removing_the_study_area_layer_clears_the_study_area(app, state):
    state.study_area = {"type": "Polygon", "coordinates": []}
    state.add_layer("Study area", "study_area", layer_id="study-area",
                    geojson={"type": "FeatureCollection", "features": []})
    resp = invoke(app, ("layers-view.data", ids.GO_REMOVE_LAYERS), 1,
                  [{"id": "study-area"}],
                  triggered=[f"{ids.GO_REMOVE_LAYERS}.data"])
    assert state.study_area is None
    assert resp[ids.STUDY_AREA_INFO]["children"] == "No study area drawn."


def test_removing_another_layer_keeps_the_study_area(app, state):
    state.study_area = {"type": "Polygon", "coordinates": []}
    layer = _add_vector(state, "Landuse", _landuse_gdf())
    resp = invoke(app, ("layers-view.data", ids.GO_REMOVE_LAYERS), 1,
                  [{"id": layer.id}],
                  triggered=[f"{ids.GO_REMOVE_LAYERS}.data"])
    assert state.study_area is not None
    assert ids.STUDY_AREA_INFO not in resp


def _sync_route_options(app, state, selected):
    return invoke(app, ("active-route.positions", ids.LAYERS_VIEW),
                  state.layers_view(), selected,
                  triggered=[f"{ids.LAYERS_VIEW}.data"])


def test_removing_the_selected_route_clears_dropdown_highlight_and_markers(
        app, state):
    keep = _route(state, "Keep")
    gone = _route(state, "Gone")
    state.active_route_id = gone.id
    state.remove_layer(gone.id)
    resp = _sync_route_options(app, state, gone.id)
    select = resp[ids.EDIT_ROUTE_SELECT]
    assert select["options"] == [{"label": "Keep", "value": keep.id}]
    assert select["value"] is None
    assert resp[ids.ACTIVE_ROUTE]["positions"] == []
    assert resp[ids.CONTROL_MARKERS]["children"] == []
    assert state.active_route_id is None


def test_the_dropdown_keeps_a_route_that_still_exists(app, state):
    keep = _route(state, "Keep")
    other = _route(state, "Other")
    state.active_route_id = keep.id
    state.remove_layer(other.id)
    resp = _sync_route_options(app, state, keep.id)
    assert resp[ids.EDIT_ROUTE_SELECT]["options"] == [
        {"label": "Keep", "value": keep.id}]
    assert "value" not in resp[ids.EDIT_ROUTE_SELECT]            # untouched
    assert ids.ACTIVE_ROUTE not in resp
    assert state.active_route_id == keep.id


def test_a_layer_view_change_without_a_selection_only_refreshes_options(
        app, state):
    route = _route(state)
    resp = _sync_route_options(app, state, None)
    assert resp[ids.EDIT_ROUTE_SELECT]["options"][0]["value"] == route.id
    assert "value" not in resp[ids.EDIT_ROUTE_SELECT]


# ======================================================== #21 table clicks
def _click_key(app):
    keys = [k for k, e in app.callback_map.items()
            if "layer-table-grid.paginationGoTo" in k
            and any('"type":"layer"' in str(i.get("id"))
                    for i in e.get("inputs", []))]
    assert len(keys) == 1, keys
    return keys[0]


def test_clicking_a_feature_beyond_the_row_cap_explains_itself(
        app, state, monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"name": list("abcde")},
        geometry=[box(i, 0, i + 1, 1) for i in range(5)], crs=CRS)
    layer = _add_vector(state, "Five", gdf)
    monkeypatch.setattr(layers_cb, "MAX_TABLE_ROWS", 2)
    feat = layer.geojson["features"][4]
    resp = invoke(
        app, _click_key(app), [None], [{"id": layer.id}], [],
        triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON, "id": layer.id},
                    "property": "clickData", "value": feat}])
    notice = _new_notices(resp)[-1]
    assert "beyond the first 2 rows" in notice["meaning"]
    assert "Row 4" in notice["meaning"]
    assert resp[ids.LAYER_TABLE_OFFCANVAS]["is_open"] is True
    assert resp[ids.LAYER_TABLE_GRID]["selectedRows"] == []
    assert len(resp[ids.LAYER_TABLE_GRID]["rowData"]) == 2


def test_clicking_a_feature_inside_the_cap_adds_no_notice(app, state):
    layer = _add_vector(state, "Landuse", _landuse_gdf())
    feat = layer.geojson["features"][2]
    resp = invoke(
        app, _click_key(app), [None], [{"id": layer.id}], [],
        triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON, "id": layer.id},
                    "property": "clickData", "value": feat}])
    assert ids.NOTICES not in resp
    assert resp[ids.LAYER_TABLE_GRID]["selectedRows"][0]["__row"] == 2


def _raster_with_inputs(state):
    """A raster whose combination table is NOT built yet (lazy inputs)."""
    lat, lng = 50.5, 9.0
    x, y = geo.point_wgs84_to_crs(lat, lng, CRS)
    base = gpd.GeoDataFrame(
        {"nutzart": ["Wald"], "bez": ["Nadelholz"]},
        geometry=[box(x - 50, y - 50, x + 50, y + 50)], crs=CRS)
    layer = state.add_layer(
        "Cost raster", "raster", crs=CRS,
        meta={"combination_inputs": {
            "base_gdf": base, "feature_keys": ("nutzart", "bez"),
            "assumptions": {"Wald": {"Nadelholz": 405, "": 405}},
            "base_layer_name": "ave", "base_crs": CRS}})
    return layer, {"latlng": {"lat": lat, "lng": lng}}


def _raster_click_key(app):
    keys = [k for k, e in app.callback_map.items()
            if "layer-table-grid.paginationGoTo" in k
            and any(i.get("id") == ids.MAP for i in e.get("inputs", []))]
    assert len(keys) == 1, keys
    return keys[0]


def _raster_click(app, layer, click, *, panel_open, draw_action=None,
                  click_mode="off", notices=()):
    return invoke(app, _raster_click_key(app), click, [{"id": layer.id}],
                  {"click_mode": click_mode}, panel_open, draw_action,
                  list(notices), triggered=[f"{ids.MAP}.clickData"])


def test_a_map_click_never_starts_the_combination_build(app, state, monkeypatch):
    layer, click = _raster_with_inputs(state)

    def must_not_build(_layer):
        raise AssertionError("a plain map click started the heavy build")

    monkeypatch.setattr(combinations, "ensure_combination_gdf", must_not_build)
    with pytest.raises(PreventUpdate):
        _raster_click(app, layer, click, panel_open=False)


def test_with_the_panel_open_the_click_looks_the_row_up(app, state):
    layer, click = _raster_with_inputs(state)
    resp = _raster_click(app, layer, click, panel_open=True)
    assert resp[ids.LAYER_TABLE_OFFCANVAS]["is_open"] is True
    assert int(resp[ids.LAYER_TABLE_GRID]["selectedRows"][0]["cost"]) == 405


def test_an_already_built_table_is_looked_up_without_the_panel(app, state):
    layer, click = _raster_with_inputs(state)
    assert combinations.ensure_combination_gdf(layer) is not None   # built
    resp = _raster_click(app, layer, click, panel_open=False)
    assert resp[ids.LAYER_TABLE_GRID]["selectedRows"]


@pytest.mark.parametrize("draw_type", ["draw:drawStart", "draw:editStart",
                                       "draw:deleteStart"])
def test_a_click_while_drawing_never_opens_the_table(app, state, draw_type):
    layer, click = _raster_with_inputs(state)
    with pytest.raises(PreventUpdate):
        _raster_click(app, layer, click, panel_open=True,
                      draw_action={"type": draw_type, "layer_type": "polygon",
                                   "n_actions": 3})


def test_after_the_drawing_has_stopped_the_click_works_again(app, state):
    layer, click = _raster_with_inputs(state)
    resp = _raster_click(app, layer, click, panel_open=True,
                         draw_action={"type": "draw:drawStop",
                                      "layer_type": "polygon", "n_actions": 4})
    assert resp[ids.LAYER_TABLE_OFFCANVAS]["is_open"] is True


def test_a_route_click_mode_blocks_the_lookup(app, state):
    layer, click = _raster_with_inputs(state)
    with pytest.raises(PreventUpdate):
        _raster_click(app, layer, click, panel_open=True,
                      click_mode="build:source")


# ================================ #17/#20 a failed combination build is shown
def _failing_raster(state):
    layer = state.add_layer(
        "Cost raster", "raster", crs=CRS,
        meta={"combination_inputs": {
            "base_gdf": None, "feature_keys": ("nutzart",),
            "assumptions": {}, "base_layer_name": "x", "base_crs": CRS}})
    return layer


def test_open_table_shows_a_failed_combination_build_as_a_notice(app, state):
    layer = _failing_raster(state)
    resp = invoke(app, (f"{ids.LAYER_TABLE_GRID}.rowData", ids.LAYER_VIEW_BTN),
                  1, [{"id": layer.id}], [],
                  triggered=[f"{ids.LAYER_VIEW_BTN}.n_clicks"])
    assert layer.meta["combination_error"]
    notice = _new_notices(resp)[-1]
    assert notice["title"].startswith("The cost-combination table could not")
    assert layer.meta["combination_error"] in notice["details"]
    assert "combination_inputs" in layer.meta                    # retry allowed


def test_a_failed_build_is_not_stacked_again_by_every_click(app, state):
    layer = _failing_raster(state)
    click = {"latlng": {"lat": 50.5, "lng": 9.0}}
    first = _raster_click(app, layer, click, panel_open=True)
    with pytest.raises(PreventUpdate):                  # nothing added twice
        _raster_click(app, layer, click, panel_open=True,
                      notices=_new_notices(first))


def test_the_legend_says_why_it_is_missing(app, state):
    layer = _failing_raster(state)
    resp = invoke(app, (f"{ids.COLORMAP_LEGEND}.children", ids.LAYERS_GRID),
                  [{"id": layer.id}], "viridis", "",
                  triggered=[f"{ids.LAYERS_GRID}.selectedRows"])
    assert "could not be built" in json.dumps(resp[ids.COLORMAP_LEGEND])


def test_graduated_classes_out_of_range_are_named(app, state):
    resp = invoke(
        app, (f"{ids.GRAD_STATUS}.children", ids.GRAD_APPLY_BTN),
        1, "equal", 1, [], [], [],
        triggered=[f"{ids.GRAD_APPLY_BTN}.n_clicks"])
    assert _new_notices(resp)[-1]["focus_control"] == ids.GRAD_CLASSES


# ===================================== #11 heavy buttons are disabled while busy
def _dependencies(app) -> list[dict]:
    response = app.server.test_client().get("/_dash-dependencies")
    assert response.status_code == 200
    return json.loads(response.get_data(as_text=True))


def _outputs_of(entry: dict) -> set[str]:
    """``{"<id>.<prop>"}`` written by a registered callback (suffix removed)."""
    names = entry["output"].strip(".").split("...")
    return {name.split("@")[0] for name in names}


@pytest.mark.parametrize("button", [
    ids.RASTER_LOAD_BTN, ids.GRAD_APPLY_BTN, ids.POINTS_APPLY_BTN,
    ids.EDIT_SIMPLIFY_BTN, ids.REFRESH_ROUTES_BTN, ids.EXPORT_ROUTE_BTN,
    ids.LAYER_VIEW_BTN, ids.ROUTES_LOAD_BTN])
def test_heavy_buttons_are_disabled_while_their_callback_runs(app, button):
    """Dash 4 keeps ``running=`` in ``app._callback_list[i]["running"]`` (what
    ``/_dash-dependencies`` serves to the browser), NOT in ``callback_map``."""
    wanted = f"{button}.disabled"
    entries = [e for e in _dependencies(app)
               if (e.get("running") or {}).get("running", {}).get(wanted)
               is True]
    assert len(entries) == 1, f"{button}: {len(entries)} callbacks"
    assert entries[0]["running"]["runningOff"] == {wanted: False}
    # the same entry is on the in-process list (the source of that endpoint)
    assert any(c.get("running") == entries[0]["running"]
               for c in app._callback_list)                    # noqa: SLF001


def test_no_running_target_is_also_written_by_a_callback(app):
    """``running`` sets the property back to a fixed value when the callback
    ends: it must not share a property with a normal output (the workflow
    gating owns Seed / Build / Run / Combine ``disabled``)."""
    entries = _dependencies(app)
    targets = {name for e in entries
               for name in (e.get("running") or {}).get("running", {})}
    written = {name for e in entries for name in _outputs_of(e)}
    assert targets
    # Known and reported, owned by callbacks/costmodel.py: Seed's `running=`
    # also targets cost-seed-btn.disabled, which the workflow gating writes.
    assert (targets & written) <= {f"{ids.COST_SEED_BTN}.disabled"},         sorted(targets & written)


# ============================================== #11 status areas with spinner
def test_status_areas_without_a_spinner_are_wrapped_in_loading(state):
    import dash.dcc as dcc

    layout = build_layout(state)
    wrapped: dict[str, object] = {}
    for component in layout._traverse():                       # noqa: SLF001
        if isinstance(component, dcc.Loading):
            for inner in component._traverse():                # noqa: SLF001
                if isinstance(getattr(inner, "id", None), str):
                    wrapped[inner.id] = component
    for status in (ids.PROJECT_STATUS, ids.COST_COVERAGE_INFO,
                   ids.EXPORT_STATUS, ids.GRAD_STATUS, OVERLAY_STATUS):
        assert status in wrapped, status
        assert wrapped[status].delay_show        # quick calls must not flicker


def test_overlay_status_line_exists_for_the_data_callbacks(state):
    assert OVERLAY_STATUS in _components(build_layout(state))


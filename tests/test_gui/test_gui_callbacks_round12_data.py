"""Round 12 (T5 robustness audit) - the data / io / cost-model callbacks.

One test group per audited finding (#7 reset on New/Open, #8 study area vs cost
polygons, #11 running= on heavy buttons, #16 non-numeric cost cells, #17
unguarded service calls, #29 full-extent WFS loads, #30 project CRS change).
Everything runs headless through the callback harness; the network is mocked.
"""
import json

import geopandas as gpd
import pytest
from conftest import invoke
from dash.exceptions import PreventUpdate
from shapely.geometry import Point, Polygon

from pyorps.gui import ids
from pyorps.gui.callbacks import data as data_cb
from pyorps.gui.callbacks import io as io_cb
from pyorps.gui.services import cost_model, data_io, geo, manual_cost, project_io

AREA = {"type": "Feature", "properties": {"type": "rectangle"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [8.6, 50.0], [8.8, 50.0], [8.8, 50.2], [8.6, 50.2], [8.6, 50.0]]]}}
COST_A = {"type": "Feature", "properties": {},
          "geometry": {"type": "Polygon", "coordinates": [[
              [9.0, 50.0], [9.1, 50.0], [9.1, 50.1], [9.0, 50.1], [9.0, 50.0]]]}}
COST_B = {"type": "Feature", "properties": {},
          "geometry": {"type": "Polygon", "coordinates": [[
              [9.2, 50.0], [9.3, 50.0], [9.3, 50.1], [9.2, 50.1], [9.2, 50.0]]]}}


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def _vector(state, name="landuse"):
    gdf = gpd.GeoDataFrame(
        {"use": ["forest", "road"]},
        geometry=[Polygon([(500000, 5530000), (500100, 5530000),
                           (500100, 5530100)]),
                  Polygon([(500200, 5530200), (500300, 5530200),
                           (500300, 5530300)])],
        crs="EPSG:25832")
    return state.add_layer(
        name, "vector", gdf=gdf, crs=gdf.crs,
        geojson=geo.gdf_to_wgs84_geojson(gdf),
        meta={"summary": data_io.dataset_summary(gdf), "source": {}})


def _draw(app, drawn, target, rows=None, notices=None, cost=100):
    return invoke(app, ("study-area-info.children", ids.DRAW_CONTROL),
                  drawn, target, "Zone", cost, "override", rows or [],
                  notices or [], triggered=[f"{ids.DRAW_CONTROL}.geojson"])


def _titles(resp):
    return [n["title"] for n in resp[ids.NOTICES]["data"]]


def _outputs(entry):
    out = entry["output"]
    out = out if isinstance(out, (list, tuple)) else [out]
    return [(o.component_id, o.component_property.split("@")[0]) for o in out]


# ============================================================ #7 reset helper
def _find_io(app, button):
    from conftest import find_callback

    return find_callback(app, "project-status.children", button)[1]


def test_new_and_open_share_the_reset_outputs(app):
    """Both callbacks write exactly RESET_TARGETS (+ the notices store)."""
    expected = list(io_cb.RESET_TARGETS) + [(ids.NOTICES, "data")]
    assert _outputs(_find_io(app, ids.GO_NEW_PROJECT)) == expected
    assert _outputs(_find_io(app, ids.GO_OPEN_PROJECT)) == expected
    covered = {cid for cid, _ in io_cb.RESET_TARGETS}
    for cid in (ids.COST_GRID, ids.COST_GRID_STATE, ids.DATASET_LIST,
                ids.STUDY_AREA_INFO, ids.DRAW_CONTROL, ids.MANUAL_GRID,
                ids.PREPROC_GRID, ids.OSM_SELECTIONS, ids.RASTERIZE_LOG,
                ids.EDIT_ROUTE_SELECT, ids.ROUTE_DRAFT, ids.ACTIVE_ROUTE,
                ids.CONTROL_MARKERS):
        assert cid in covered


def test_new_project_empties_every_client_value(app, state):
    _vector(state)
    state.study_area = AREA
    state.project_crs = "EPSG:32632"
    resp = invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1,
                  [], triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    got = {t: resp[t[0]][t[1]] for t in io_cb.RESET_TARGETS}
    assert got[(ids.LAYERS_VIEW, "data")] == []
    assert got[(ids.PROJECT_STATUS, "children")] == "new project"
    assert got[(ids.PROJECT_CRS, "value")] == "EPSG:25832"
    assert got[(ids.COST_GRID, "columnDefs")] == []
    assert got[(ids.COST_GRID, "rowData")] == []
    assert got[(ids.COST_GRID_STATE, "data")] == {}
    assert got[(ids.COST_COVERAGE_INFO, "children")] == ""
    assert got[(ids.COST_DATASET, "value")] is None
    assert got[(ids.COST_FEATURE_KEYS, "value")] == []
    for target in ((ids.MODIFIER_GRID, "rowData"),
                   (ids.PREPROC_GRID, "rowData"), (ids.PPB_CONDS, "data"),
                   (ids.MANUAL_GRID, "rowData"),
                   (ids.OSM_SELECTIONS, "data"),
                   (ids.EDIT_ROUTE_SELECT, "options"),
                   (ids.ACTIVE_ROUTE, "positions"),
                   (ids.CONTROL_MARKERS, "children")):
        assert got[target] == [], target
    assert "No vector datasets" in json.dumps(
        got[(ids.DATASET_LIST, "children")])
    assert got[(ids.STUDY_AREA_INFO, "children")] == "No study area drawn."
    assert got[(ids.DRAW_CONTROL, "editToolbar")]["action"] == "clear all"
    assert got[(ids.RASTERIZE_LOG, "children")] == ""
    assert got[(ids.EDIT_ROUTE_SELECT, "value")] is None
    assert got[(ids.ROUTE_DRAFT, "data")] == {
        "sources": [], "targets": [], "waypoints": []}
    assert got[(ids.SELECTED_FEATURE, "data")] == {}
    assert got[(ids.SELECTED_FEATURE_LAYER, "data")]["features"] == []
    assert state.layers == {} and state.study_area is None


def test_open_project_fills_from_state_and_clears_the_rest(app, state,
                                                           tmp_path):
    layer = _vector(state)
    state.study_area = AREA
    state.add_layer("Route 1", "route", geojson=_fc(
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString",
                      "coordinates": [[8.6, 50.0], [8.7, 50.1]]}}))
    rows = [{"use": "forest", "cost": 10, "forbidden": False},
            {"use": "road", "cost": 5, "forbidden": False}]
    path = project_io.save_project(
        state, tmp_path / "p",
        cost_table={"feature_keys": ["use"], "dataset_id": layer.id,
                    "rows": rows,
                    "modifiers": [{"dataset": "landuse", "factor": "2"}]})
    state.active_route_id = "stale-route-id"
    resp = invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1,
                  path, [], triggered=[f"{ids.GO_OPEN_PROJECT}.data"])
    got = {t: resp[t[0]][t[1]] for t in io_cb.RESET_TARGETS}
    # filled from the loaded project
    assert [ly["name"] for ly in got[(ids.LAYERS_VIEW, "data")]] == [
        "landuse", "Route 1"]
    assert got[(ids.PROJECT_STATUS, "children")] == f"opened: {path}"
    assert got[(ids.COST_GRID, "rowData")] == rows
    assert [c["field"] for c in got[(ids.COST_GRID, "columnDefs")]] == [
        "use", "cost", "forbidden"]
    assert got[(ids.COST_GRID_STATE, "data")] == {
        "dataset_id": layer.id, "feature_keys": ["use"]}
    assert got[(ids.COST_COVERAGE_INFO, "children")].startswith("✓")
    assert got[(ids.MODIFIER_GRID, "rowData")][0]["factor"] == "2"
    assert "landuse" in json.dumps(got[(ids.DATASET_LIST, "children")])
    assert "km" in got[(ids.STUDY_AREA_INFO, "children")]
    assert [o["label"] for o in got[(ids.EDIT_ROUTE_SELECT, "options")]] == [
        "Route 1"]
    # cleared: it belonged to the project that was replaced
    assert got[(ids.EDIT_ROUTE_SELECT, "value")] is None
    assert state.active_route_id is None
    assert got[(ids.ROUTE_DRAFT, "data")]["sources"] == []
    assert got[(ids.DRAW_CONTROL, "editToolbar")]["action"] == "clear all"
    for target in ((ids.PREPROC_GRID, "rowData"), (ids.PPB_CONDS, "data"),
                   (ids.OSM_SELECTIONS, "data"), (ids.MANUAL_GRID, "rowData"),
                   (ids.ACTIVE_ROUTE, "positions"),
                   (ids.CONTROL_MARKERS, "children")):
        assert got[target] == [], target
    assert got[(ids.RASTERIZE_LOG, "children")] == ""
    assert got[(ids.SELECTED_FEATURE_LAYER, "data")]["features"] == []


def test_open_keeps_its_study_area_when_the_draw_control_echoes_empty(
        app, state, tmp_path):
    """Clearing the draw control on Open makes it report "no shapes"; that
    echo must not delete the study area the project just loaded."""
    state.study_area = AREA
    path = project_io.save_project(state, tmp_path / "p")
    invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1, path,
           [], triggered=[f"{ids.GO_OPEN_PROJECT}.data"])
    assert state.study_area == AREA
    with pytest.raises(PreventUpdate):
        _draw(app, _fc(), "area")
    assert state.study_area == AREA


def test_status_text_mentions_skipped_layers(app, state, tmp_path):
    # save: a raster layer whose file is gone cannot be written
    state.add_layer("gone raster", "raster",
                    meta={"source_path": str(tmp_path / "nope.tif")})
    target = tmp_path / "saved"
    resp = invoke(app, (f"{ids.PROJECT_STATUS}.children",
                        ids.GO_SAVE_PROJECT),
                  1, str(target), ["raster", "vector", "cost_table"], [],
                  {}, [], [], triggered=[f"{ids.GO_SAVE_PROJECT}.data"])
    assert "1 layer(s) left out" in resp[ids.PROJECT_STATUS]["children"]
    assert "gone raster" in resp[ids.PROJECT_STATUS]["children"]
    # open: an entry that has nothing to restore is reported the same way
    manifest = tmp_path / "old" / "project.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"version": 1, "layers": [
        {"id": "r1", "name": "lost raster", "kind": "raster"}]}),
        encoding="utf-8")
    resp = invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1,
                  str(manifest), [],
                  triggered=[f"{ids.GO_OPEN_PROJECT}.data"])
    status = resp[ids.PROJECT_STATUS]["children"]
    assert status.startswith("opened:") and "1 layer(s) left out" in status
    assert "lost raster" in status


# ====================================================== #8 study area vs cost
def test_editing_with_cost_polygons_does_not_turn_one_into_the_study_area(
        app, state):
    _draw(app, _fc(AREA), "area")
    assert state.study_area == AREA
    both = _fc(AREA, COST_A, COST_B)
    resp = _draw(app, both, "cost")
    rows = resp[ids.MANUAL_GRID]["rowData"]
    assert len(rows) == 2
    # a draw event on the Data tab (an edit or a delete): the shared control
    # still holds both cost polygons, the LAST feature is a cost polygon
    _draw(app, both, "area", rows)
    assert state.study_area == AREA
    assert len(state.get(manual_cost.MANUAL_LAYER_ID).gdf) == 2


def test_deleting_the_study_area_shape_clears_it_even_with_cost_polygons(
        app, state):
    _draw(app, _fc(AREA), "area")
    resp = _draw(app, _fc(AREA, COST_A, COST_B), "cost")
    rows = resp[ids.MANUAL_GRID]["rowData"]
    # the area rectangle was deleted: only the cost polygons are left
    resp = _draw(app, _fc(COST_A, COST_B), "area", rows)
    assert state.study_area is None
    assert state.get("study-area") is None
    assert resp[ids.STUDY_AREA_INFO]["children"] == "No study area drawn."


def test_removed_cost_polygon_shape_never_becomes_the_study_area(app, state):
    resp = _draw(app, _fc(COST_A, COST_B), "cost")
    rows = resp[ids.MANUAL_GRID]["rowData"]
    state.manual_excluded_geoms.add(
        manual_cost.geojson_feature_geometry(COST_B).wkt)
    with pytest.raises(PreventUpdate):
        _draw(app, _fc(COST_B), "area", rows)
    assert state.study_area is None


# =========================================================== #11 running= bar
HEAVY_BUTTONS = [
    ids.WFS_LOAD_BTN, ids.LOCAL_LOAD_BTN, ids.OSM_LOAD_BTN, ids.DEM_LOAD_BTN,
    ids.OVERLAY_LOAD_BTN, ids.WFS_CAPS_BTN, ids.MERGE_BTN,
    ids.PROJECT_SAVE_BTN, ids.PROJECT_OPEN_BTN, ids.COST_SEED_BTN,
    ids.COST_IMPORT_BTN, ids.COST_EXPORT_BTN]

# Save, Open and the cost export ask first (confirm gate): their callback
# listens on the GO store the gate writes, not on the button
GATED_INPUT = {ids.PROJECT_SAVE_BTN: (ids.GO_SAVE_PROJECT, "data"),
               ids.PROJECT_OPEN_BTN: (ids.GO_OPEN_PROJECT, "data"),
               ids.COST_EXPORT_BTN: (ids.GO_EXPORT_COSTS, "data")}


@pytest.mark.parametrize("button", HEAVY_BUTTONS)
def test_heavy_buttons_are_disabled_while_their_callback_runs(app, button):
    """Dash 4 keeps ``running=`` in ``app._callback_list`` (the callback spec:
    ``spec["running"] == {"running": {...}, "runningOff": {...}}``), not in
    ``app.callback_map``."""
    specs = [s for s in app._callback_list
             if ((s.get("running") or {}).get("running") or {}).get(
                 f"{button}.disabled") is True]
    assert len(specs) == 1, f"{button}: {len(specs)} callbacks with running="
    spec = specs[0]
    assert spec["running"]["runningOff"][f"{button}.disabled"] is False
    component, prop = GATED_INPUT.get(button, (button, "n_clicks"))
    assert {"id": component, "property": prop} in spec["inputs"]


# ====================================================== #16 non-numeric costs
def test_coerce_cost_accepts_decimal_comma_and_rejects_the_rest():
    assert cost_model._coerce_cost("1,5") == 1.5
    assert cost_model._coerce_cost(" 405 ") == 405
    assert cost_model._coerce_cost("") == 0
    for bad in ("abc", "1,2,3", "nan", "12 m"):
        with pytest.raises(cost_model.InvalidCostValueError):
            cost_model._coerce_cost(bad)


def test_grid_rows_with_text_cost_name_the_row_and_the_text():
    rows = [{"use": "forest", "cost": "5", "forbidden": False},
            {"use": "road", "cost": "abc", "forbidden": False},
            {"use": "water", "cost": "x", "forbidden": True}]   # ignored
    problems = cost_model.invalid_cost_cells(rows, ("use",))
    assert problems == [{"row": 2, "feature": "road", "text": "abc"}]
    with pytest.raises(cost_model.InvalidCostValueError, match="row 2.*abc"):
        cost_model.assumptions_from_grid_rows(rows, ("use",))
    notice = cost_model.cost_problem_notice(rows, ("use",))
    assert notice["severity"] == "warning" and "abc" in notice["meaning"]
    assert cost_model.cost_problem_notice(
        [{"use": "a", "cost": "1,5", "forbidden": False}], ("use",)) is None
    # ... and a decimal comma is a number
    assert cost_model.assumptions_from_grid_rows(
        [{"use": "a", "cost": "1,5", "forbidden": False}], ("use",)) == {
            "a": 1.5}


def test_coverage_reports_a_text_cost_as_a_notice_not_an_error(app, state):
    layer = _vector(state)
    grid_state = {"dataset_id": layer.id, "feature_keys": ["use"]}
    rows = [{"use": "forest", "cost": "10", "forbidden": False},
            {"use": "road", "cost": "1,2,3", "forbidden": False}]
    resp = invoke(app, ("cost-coverage-info.children", ids.COST_GRID),
                  None, rows, grid_state, [],
                  triggered=[f"{ids.COST_GRID}.cellValueChanged"])
    notice = resp[ids.NOTICES]["data"][-1]
    assert notice["title"] == "A cost is not a number"
    assert "1,2,3" in notice["meaning"] and "row 2" in notice["meaning"]
    assert "not numbers" in resp[ids.COST_COVERAGE_INFO]["children"]
    # the same bad cell is not announced again on the next edit
    again = invoke(app, ("cost-coverage-info.children", ids.COST_GRID),
                   None, rows, grid_state, resp[ids.NOTICES]["data"],
                   triggered=[f"{ids.COST_GRID}.cellValueChanged"])
    assert ids.NOTICES not in again
    # a decimal comma is fine: coverage runs, no notice
    rows[1]["cost"] = "1,5"
    ok = invoke(app, ("cost-coverage-info.children", ids.COST_GRID),
                None, rows, grid_state, [],
                triggered=[f"{ids.COST_GRID}.cellValueChanged"])
    assert ok[ids.COST_COVERAGE_INFO]["children"].startswith("✓")
    assert ids.NOTICES not in ok


# ====================================================== #17 unguarded calls
def test_draw_with_a_text_cost_cell_gives_a_notice(app, state):
    rows = [{"__row": 0, "name": "Zone 1", "cost": "abc"}]
    resp = _draw(app, _fc(COST_A), "cost", rows)
    notice = resp[ids.NOTICES]["data"][-1]
    assert notice["title"] == "A cost is not a number"
    assert "Zone 1" in notice["meaning"] and "abc" in notice["meaning"]


def test_draw_failure_in_the_cost_branch_is_a_notice(app, state,
                                                     monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sync broke")

    monkeypatch.setattr(manual_cost, "sync_manual_layer", boom)
    resp = _draw(app, _fc(COST_A), "cost")
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Something went wrong"


def test_on_draw_caps_the_notice_history(app, state):
    old = [{"severity": "info", "title": f"n{i}", "id": str(i)}
           for i in range(40)]
    resp = _draw(app, _fc(AREA), "area", notices=old)
    assert len(resp[ids.NOTICES]["data"]) == 20
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Study area set"


def test_manual_grid_edit_with_text_cost_is_a_notice_and_changes_nothing(
        app, state):
    resp = _draw(app, _fc(COST_A, COST_B), "cost")
    layer = state.get(manual_cost.MANUAL_LAYER_ID)
    before = layer.gdf["cost"].tolist()
    out = invoke(app, ("layers-view.data", ids.MANUAL_GRID,
                       "cellValueChanged"),
                 {"data": {"__row": 1, "cost": "banana"}, "colId": "cost"},
                 [], triggered=[f"{ids.MANUAL_GRID}.cellValueChanged"])
    assert out[ids.NOTICES]["data"][-1]["title"] == "A cost is not a number"
    assert layer.gdf["cost"].tolist() == before
    assert resp[ids.MANUAL_GRID]["rowData"]


def test_remove_manual_selected_with_text_cost_is_a_notice(app, state):
    resp = _draw(app, _fc(COST_A, COST_B), "cost")
    rows = resp[ids.MANUAL_GRID]["rowData"]
    rows[1]["cost"] = "abc"
    out = invoke(app, ("layers-view.data", ids.MANUAL_DEL_BTN), 1,
                 _fc(COST_A, COST_B), rows, [rows[0]], "Zone", 42,
                 "override", [],
                 triggered=[f"{ids.MANUAL_DEL_BTN}.n_clicks"])
    assert out[ids.NOTICES]["data"][-1]["title"] == "A cost is not a number"
    assert len(state.get(manual_cost.MANUAL_LAYER_ID).gdf) == 2
    assert not state.manual_excluded_geoms


def test_local_load_failure_while_registering_the_layer_is_a_notice(
        app, state, tmp_path, monkeypatch):
    path = tmp_path / "data.geojson"
    gpd.GeoDataFrame({"use": ["a"]}, geometry=[Point(500000, 5530000)],
                     crs="EPSG:25832").to_file(path, driver="GeoJSON")

    def boom(gdf):
        raise RuntimeError("cannot build display data")

    monkeypatch.setattr(data_cb.geo, "gdf_to_wgs84_geojson", boom)
    resp = invoke(app, ("dataset-list.children", ids.LOCAL_LOAD_BTN), 1,
                  str(path), None, False, [],
                  triggered=[f"{ids.LOCAL_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Something went wrong"
    assert state.layers_of_kind("vector") == []


def test_osm_reprojection_failure_is_a_notice(app, state, monkeypatch):
    from pyorps.gui.services import osm

    # a frame without a CRS cannot be reprojected into the project CRS
    monkeypatch.setattr(
        osm, "load_osm_features",
        lambda *a, **k: gpd.GeoDataFrame({"landuse": ["forest"]},
                                         geometry=[Point(8.1, 50.1)]))
    state.study_area = AREA
    resp = invoke(app, ("layers-view.data", ids.OSM_LOAD_BTN),
                  1, "Land use (landuse=*)", "", [], [],
                  triggered=[f"{ids.OSM_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Something went wrong"
    assert state.layers_of_kind("vector") == []


# ================================================= #29 whole-layer WFS loads
def _wfs_gdf():
    return gpd.GeoDataFrame({"use": ["a"]},
                            geometry=[Point(500000, 5530000)],
                            crs="EPSG:25832")


def _wfs_click(app, clip, notices=None, whole=False, layer="lyr"):
    return invoke(app, ("dataset-list.children", ids.WFS_LOAD_BTN), 1,
                  "https://x.test/wfs", layer, clip, notices or [], whole,
                  triggered=[f"{ids.WFS_LOAD_BTN}.n_clicks"])


def test_wfs_without_clip_is_refused_and_a_repeat_changes_nothing(
        app, state, monkeypatch):
    """No second-click confirmation any more: the full download is the
    explicit 'Load the whole layer' checkbox."""
    calls = []

    def fake(url, layer, **kwargs):
        calls.append(kwargs.get("mask"))
        return _wfs_gdf()

    monkeypatch.setattr(data_cb.data_io, "load_wfs_vector", fake)
    for _ in range(2):
        resp = _wfs_click(app, False)
        assert resp[ids.NOTICES]["data"][-1]["title"] == \
            "This would download the whole layer"
    assert calls == [] and state.layers_of_kind("vector") == []


def test_wfs_whole_layer_checkbox_loads_every_feature(app, state, monkeypatch):
    calls = []

    def fake(url, layer, **kwargs):
        calls.append(kwargs.get("mask"))
        return _wfs_gdf()

    monkeypatch.setattr(data_cb.data_io, "load_wfs_vector", fake)
    resp = _wfs_click(app, False, whole=True)
    assert resp[ids.NOTICES]["data"][-1]["title"].startswith("Loaded")
    assert calls == [None]
    assert len(state.layers_of_kind("vector")) == 1


def test_wfs_with_clip_and_an_area_loads_at_once(app, state, monkeypatch):
    seen = []
    monkeypatch.setattr(
        data_cb.data_io, "load_wfs_vector",
        lambda url, layer, **k: seen.append(k.get("mask")) or _wfs_gdf())
    state.study_area = AREA
    resp = _wfs_click(app, True)
    assert resp[ids.NOTICES]["data"][-1]["title"].startswith("Loaded")
    assert seen and seen[0] is not None


def test_wfs_with_clip_and_no_area_still_asks_for_an_area(app, state):
    resp = _wfs_click(app, True)
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Draw a study area first"


def test_crs_change_is_refused_when_layers_exist(app, state):
    _vector(state)
    resp = invoke(app, ("notices.data", ids.PROJECT_CRS), "EPSG:32632", [],
                  triggered=[f"{ids.PROJECT_CRS}.value"])
    notice = resp[ids.NOTICES]["data"][-1]
    assert notice["title"] == "Project CRS not changed"
    assert "1 layer(s)" in notice["meaning"] and "EPSG:25832" in \
        notice["meaning"]
    assert state.project_crs == "EPSG:25832"
    assert resp[ids.PROJECT_CRS]["value"] == "EPSG:25832"      # field reverts


def test_crs_change_without_layers_is_accepted_and_marks_dirty(app, state):
    assert state.dirty is False
    resp = invoke(app, ("notices.data", ids.PROJECT_CRS), "EPSG:32632", [],
                  triggered=[f"{ids.PROJECT_CRS}.value"])
    assert state.project_crs == "EPSG:32632"
    assert state.dirty is True
    assert resp[ids.NOTICES]["data"][-1]["severity"] == "success"
    assert resp[ids.DIRTY_STORE]["data"] is True


def test_crs_echo_of_the_current_value_does_nothing(app, state):
    _vector(state)
    with pytest.raises(PreventUpdate):
        invoke(app, ("notices.data", ids.PROJECT_CRS), "EPSG:25832", [],
               triggered=[f"{ids.PROJECT_CRS}.value"])

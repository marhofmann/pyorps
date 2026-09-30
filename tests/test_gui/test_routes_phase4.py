"""Phase 4: click dispatcher (8.1/8.2), route builder (C13/C14, F1/F5/F8)."""
import json

import pytest
from pyproj import Transformer

from pyorps.gui import ids

from conftest import SOURCE, TARGET, TEST_CRS, invoke


def _to_wgs84(x, y):
    tf = Transformer.from_crs(TEST_CRS, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(x, y)
    return lat, lon


@pytest.fixture()
def raster_layer(app, state, raster_path):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    return state.layers_of_kind("raster")[0]


def _draft(points_by_kind):
    tf = Transformer.from_crs(TEST_CRS, "EPSG:4326", always_xy=True)
    draft = {"sources": [], "targets": [], "waypoints": []}
    for kind, pts in points_by_kind.items():
        for x, y in pts:
            lon, lat = tf.transform(x, y)
            draft[f"{kind}s"].append(
                {"lat": lat, "lng": lon, "x": x, "y": y})
    return draft


# --------------------------------------------------------------- mode select
def test_mode_exclusive_and_badge(app):
    # edit knob OFF -> the mode radio builds the new-route draft (task 55)
    resp = invoke(app, ("ui-state.data", ids.BUILD_MODE),
                  "source", False, {"click_mode": "off"},
                  triggered=[f"{ids.BUILD_MODE}.value"])
    assert resp[ids.UI_STATE]["data"]["click_mode"] == "build:source"
    assert "SOURCE" in resp[ids.MODE_BADGE]["children"]
    # edit knob ON -> the SAME radio edits the selected route
    resp = invoke(app, ("ui-state.data", ids.BUILD_MODE),
                  "target", True, {"click_mode": "build:source"},
                  triggered=[f"{ids.EDIT_ENABLE}.value"])
    assert resp[ids.UI_STATE]["data"]["click_mode"] == "edit:target"
    assert "MOVES TARGET" in resp[ids.MODE_BADGE]["children"]
    # move-waypoint only means something while editing
    resp = invoke(app, ("ui-state.data", ids.BUILD_MODE),
                  "move-waypoint", False, {"click_mode": "off"},
                  triggered=[f"{ids.BUILD_MODE}.value"])
    assert resp[ids.UI_STATE]["data"]["click_mode"] == "off"
    resp = invoke(app, ("ui-state.data", ids.BUILD_MODE),
                  "move-waypoint", True, {"click_mode": "off"},
                  triggered=[f"{ids.EDIT_ENABLE}.value"])
    assert resp[ids.UI_STATE]["data"]["click_mode"] == "edit:move-waypoint"


# ---------------------------------------------------------------- dispatcher
def test_click_dispatch_build_mode(app, state, raster_layer):
    lat, lon = _to_wgs84(*SOURCE)
    click = {"latlng": {"lat": lat, "lng": lon}, "n_clicks": 1}
    resp = invoke(app, ("route-draft.data", "clickData"),
                  click, {"click_mode": "build:source"},
                  {"sources": [], "targets": [], "waypoints": []},
                  raster_layer.id,
                  triggered=[f"{ids.MAP}.clickData"])
    draft = resp[ids.ROUTE_DRAFT]["data"]
    assert len(draft["sources"]) == 1
    # projected back into the raster CRS within a metre
    assert abs(draft["sources"][0]["x"] - SOURCE[0]) < 1
    assert abs(draft["sources"][0]["y"] - SOURCE[1]) < 1


def test_click_dispatch_edit_mode_emits_request(app, state, raster_layer):
    lat, lon = _to_wgs84(*TARGET)
    click = {"latlng": {"lat": lat, "lng": lon}, "n_clicks": 3}
    resp = invoke(app, ("route-draft.data", "clickData"),
                  click, {"click_mode": "edit:waypoint"},
                  {"sources": [], "targets": [], "waypoints": []},
                  raster_layer.id,
                  triggered=[f"{ids.MAP}.clickData"])
    request = resp[ids.EDIT_REQUEST]["data"]
    assert request["action"] == "waypoint"
    assert request["seq"] == 3
    assert abs(request["x"] - TARGET[0]) < 1


def test_draft_renders_markers_and_rows(app):
    import math

    draft = _draft({"source": [SOURCE], "target": [TARGET],
                    "waypoint": [(500100.0, 5599950.0)]})
    resp = invoke(app, (f"{ids.BUILDER_MARKERS}.children",), draft)
    rows = resp[ids.BUILD_POINTS_TABLE]["rowData"]
    # chain order so the 'dist' column shows the leg arriving at each point
    assert [r["kind"] for r in rows] == ["source", "waypoint", "target"]
    assert rows[0]["dist"] is None
    expected = math.hypot(rows[1]["x"] - rows[0]["x"],
                          rows[1]["y"] - rows[0]["y"])
    assert abs(rows[1]["dist"] - expected) < 0.11
    assert len(resp[ids.BUILDER_MARKERS]["children"]) == 3


# -------------------------------------------------------------- run routing
def test_run_routing_creates_route_layer(app, state, raster_layer):
    draft = _draft({"source": [SOURCE], "target": [TARGET],
                    "waypoint": [(500100.0, 5599950.0)]})
    resp = invoke(
        app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
        1, False, draft, raster_layer.id, "dijkstra", "cpu", "r1",
        80, True, False, False, 1.0, 100, 0, False, [], [],
        triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    assert "1 route(s) built" in resp[ids.ROUTING_STATUS]["children"]
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    meta = routes[0].meta
    assert meta["origin"] == "built"
    assert meta["params"]["use_astar"] is False          # F8
    assert len(meta["control_points"]) == 3              # chained waypoint
    assert meta["metrics"]["total_cost"] > 0
    # the finder is cached for editing
    assert raster_layer.id in state.finders
    # edit dropdown options refreshed
    options = resp[ids.EDIT_ROUTE_SELECT]["options"]
    assert options[0]["value"] == routes[0].id


def test_waypoint_names_stored_and_rendered(app, state, raster_layer):
    from pyorps.gui.callbacks.interaction import point_marker
    from pyorps.gui.callbacks.layers import route_control_markers

    draft = _draft({"source": [SOURCE], "target": [TARGET],
                    "waypoint": [(500100.0, 5599950.0)]})
    table_rows = [
        {"kind": "source", "x": SOURCE[0], "y": SOURCE[1], "name": ""},
        {"kind": "target", "x": TARGET[0], "y": TARGET[1], "name": ""},
        {"kind": "waypoint", "x": 500100.0, "y": 5599950.0, "name": "Bridge"},
    ]
    invoke(app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
           1, False, draft, raster_layer.id, "dijkstra", "cpu", "r1",
           80, True, False, False, 1.0, 100, 0, False, table_rows, [],
           triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    route = state.layers_of_kind("route")[0]
    assert route.meta["waypoint_names"] == ["Bridge"]
    # a named waypoint marker shows its name permanently; unnamed points don't
    named = point_marker("waypoint", 50.5, 9.0, 0, "Bridge")
    assert named.children.children == "Bridge" and named.children.permanent
    plain = point_marker("waypoint", 50.5, 9.0, 0, "")
    assert plain.children.children == "waypoint 1"
    assert not plain.children.permanent
    assert any(m.children.children == "Bridge"
               for m in route_control_markers(route))


def test_run_routing_without_raster_warns(app, state):
    resp = invoke(
        app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
        1, False, _draft({"source": [SOURCE], "target": [TARGET]}), None,
        "dijkstra", "cpu", "r1", None, True, False, False, 1.0,
        100, 0, False, [], [],
        triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Pick a cost raster first"


def test_run_routing_point_outside_raster(app, state, raster_layer):
    draft = _draft({"source": [SOURCE], "target": [(999999.0, 999999.0)]})
    resp = invoke(
        app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
        1, False, draft, raster_layer.id, "dijkstra", "cpu", "r1",
        80, True, False, False, 1.0, 100, 0, False, [], [],
        triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    assert "outside" in resp[ids.NOTICES]["data"][-1]["title"]


def test_simplify_keeps_metrics_from_full_line(app, state, raster_layer):
    draft = _draft({"source": [SOURCE], "target": [TARGET]})
    invoke(app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
           1, False, draft, raster_layer.id, "dijkstra", "cpu", "r1",
           80, True, False, True, 5.0, 100, 0, False, [], [],
           triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    route = state.layers_of_kind("route")[-1]
    assert route.meta["simplify_tol"] == 5.0
    # display geometry simplified, but metrics from the un-simplified line
    assert route.meta["metrics"]["total_cost"] > 0
    n_display = len(route.gdf.geometry.iloc[0].coords)
    assert n_display >= 2


# ---------------------------------------------------------- hardware filter
def test_algorithm_options_follow_hardware(app):
    resp = invoke(app, (f"{ids.ALGORITHM}.options",), "gpu",
                  triggered=[f"{ids.HARDWARE}.value"])
    values = [o["value"] for o in resp[ids.ALGORITHM]["options"]]
    assert "bidirectional_dijkstra" not in values
    assert resp[ids.ALGORITHM]["value"] == "delta-stepping"


def test_buffer_info_shows_auto_default(app):
    draft = _draft({"source": [SOURCE], "target": [TARGET]})
    resp = invoke(app, (f"{ids.SEARCH_BUFFER_INFO}.children",), None, draft)
    assert "auto" in resp[ids.SEARCH_BUFFER_INFO]["children"]
    resp = invoke(app, (f"{ids.SEARCH_BUFFER_INFO}.children",), 500, draft)
    assert "500" in resp[ids.SEARCH_BUFFER_INFO]["children"]


# ------------------------------------------------------------- load routes
def test_load_routes_from_file(app, state, tmp_path, finder):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        finder.paths.to_geodataframe_records(), geometry="geometry",
        crs=finder.dataset.crs)
    path = tmp_path / "routes.gpkg"
    gdf.to_file(path, driver="GPKG")
    resp = invoke(app, ("layers-view.data", ids.ROUTES_LOAD_BTN),
                  1, str(path), [],
                  triggered=[f"{ids.ROUTES_LOAD_BTN}.n_clicks"])
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    meta = routes[0].meta
    assert meta["origin"] == "imported"
    assert len(meta["control_points"]) == 2   # endpoints derived

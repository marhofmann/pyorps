"""Phase 5: route editing spawns lineage'd variants (R11, Section 18)."""
import pytest

from pyorps.gui import ids

from conftest import SOURCE, TARGET, invoke
from test_routes_phase4 import _draft


@pytest.fixture()
def built_route(app, state, raster_path):
    """A raster layer + one built route, exactly as Phase 4 produces them."""
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    draft = _draft({"source": [SOURCE], "target": [TARGET]})
    invoke(app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
           1, False, draft, raster_layer.id, "dijkstra", "cpu", "r1",
           80, True, False, False, 1.0, 100, 0, False, [], [],
           triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    return state.layers_of_kind("route")[0]


_EMPTY_DRAFT = {"sources": [], "targets": [], "waypoints": []}


def _select(app, state, route_id):
    return invoke(app, (f"{ids.ACTIVE_ROUTE}.positions",
                        ids.EDIT_ROUTE_SELECT), route_id, _EMPTY_DRAFT,
                  triggered=[f"{ids.EDIT_ROUTE_SELECT}.value"])


def test_select_route_renders_editing_layer(app, state, built_route):
    resp = _select(app, state, built_route.id)
    assert state.active_route_id == built_route.id
    positions = resp[ids.ACTIVE_ROUTE]["positions"]
    assert len(positions) >= 2
    # source + target markers
    assert len(resp[ids.CONTROL_MARKERS]["children"]) == 2
    # the ONE points list shows the route's control points
    rows = resp[ids.BUILD_POINTS_TABLE]["rowData"]
    assert [r["kind"] for r in rows] == ["source", "target"]
    import json
    assert "Length" in json.dumps(resp[ids.EDIT_COST_READOUT]["children"])
    # the editable routes grid lists this route with its run group + name
    from pyorps.gui.callbacks.groups import _route_row
    row = _route_row(built_route)
    assert row["name"] == built_route.name and "Run 1" in row["group"]


def test_move_target_spawns_variant(app, state, built_route):
    _select(app, state, built_route.id)
    new_target = (500150.0, 5599900.0)
    request = {"action": "target", "x": new_target[0], "y": new_target[1],
               "lat": 0, "lng": 0, "seq": 1}
    resp = invoke(app, ("layers-view.data", "edit-request"), request, True, [],
                  triggered=[f"{ids.EDIT_REQUEST}.data"])
    routes = state.layers_of_kind("route")
    assert len(routes) == 1                       # task 44: editing replaces
    edited = routes[0]
    assert edited.meta["origin"] == "edited"
    assert edited.meta["edit"] == "moved target"
    assert edited.meta["control_points"][-1] == list(new_target)
    # the old route is gone (replaced, not preserved)
    assert state.get(built_route.id) is None
    # the edited route became active + selected
    assert state.active_route_id == edited.id
    assert resp[ids.EDIT_ROUTE_SELECT]["value"] == edited.id
    # the routes grid lists only the edited route
    from pyorps.gui.callbacks.groups import _routes_rows
    grid_ids = {r["id"] for r in _routes_rows(state)}
    assert grid_ids == {edited.id}


def test_route_group_management(app, state, built_route):
    """Feature 2: inline edits, move-to-group, group rename/restyle/delete."""
    from pyorps.gui.callbacks.groups import _route_row
    rid = built_route.id

    # inline colour edit via the grid (cellValueChanged)
    invoke(app, ("layers-view.data", "routes-grid", "cellValueChanged"),
           {"data": {"id": rid, "color": "#00ff00"}, "colId": "color"},
           triggered=[f"{ids.ROUTES_GRID}.cellValueChanged"])
    assert state.get(rid).style["color"] == "#00ff00"

    # move the route to a new group (multi-select + button; here one route)
    invoke(app, ("layers-view.data", "route-move-btn"),
           1, [{"id": rid}], "Feeders", [],
           triggered=[f"{ids.ROUTE_MOVE_BTN}.n_clicks"])
    assert state.get(rid).meta["group"] == "Feeders"
    assert "Feeders" in state.route_groups()
    assert _route_row(state.get(rid))["group"] == "Feeders"

    # rename the whole group
    invoke(app, ("layers-view.data", "group-rename-btn"),
           1, "Feeders", "MV Feeders", [],
           triggered=[f"{ids.GROUP_RENAME_BTN}.n_clicks"])
    assert state.get(rid).meta["group"] == "MV Feeders"

    # restyle the whole group (dashed)
    invoke(app, ("layers-view.data", "group-restyle-btn"),
           1, "MV Feeders", "#123456", "dashed", [],
           triggered=[f"{ids.GROUP_RESTYLE_BTN}.n_clicks"])
    assert state.get(rid).style["dashArray"] == "8 8"

    # delete the whole group -> the route is gone
    invoke(app, ("layers-view.data", "group-delete-btn"),
           1, "MV Feeders", [],
           triggered=[f"{ids.GROUP_DELETE_BTN}.n_clicks"])
    assert state.get(rid) is None


def test_add_waypoint_then_delete_via_points_list(app, state, built_route):
    _select(app, state, built_route.id)
    request = {"action": "waypoint", "x": 500100.0, "y": 5599950.0,
               "lat": 0, "lng": 0, "seq": 2}
    resp = invoke(app, ("layers-view.data", "edit-request"), request, True, [],
                  triggered=[f"{ids.EDIT_REQUEST}.data"])
    variant = state.get(state.active_route_id)
    assert len(variant.meta["control_points"]) == 3
    rows = resp[ids.BUILD_POINTS_TABLE]["rowData"]
    assert [r["kind"] for r in rows] == ["source", "waypoint", "target"]

    # delete the waypoint row, then Apply -> a new 2-point variant
    waypoint_row = rows[1]
    resp = invoke(app, (f"{ids.BUILD_POINTS_TABLE}.rowData",
                        ids.POINTS_REMOVE_BTN),
                  1, rows, [waypoint_row], _EMPTY_DRAFT,
                  triggered=[f"{ids.POINTS_REMOVE_BTN}.n_clicks"])
    remaining = resp[ids.BUILD_POINTS_TABLE]["rowData"]
    assert [r["kind"] for r in remaining] == ["source", "target"]

    resp = invoke(app, ("layers-view.data", ids.POINTS_APPLY_BTN),
                  1, remaining, None, [],
                  triggered=[f"{ids.POINTS_APPLY_BTN}.n_clicks"])
    applied = state.get(state.active_route_id)
    assert applied.meta["edit"] == "removed waypoint"
    assert len(applied.meta["control_points"]) == 2
    assert len(state.layers_of_kind("route")) == 1   # task 44: edits replace


def test_move_waypoint_relocates_nearest(app, state, built_route):
    """Task 55: 'Move waypoint' relocates the nearest existing waypoint."""
    _select(app, state, built_route.id)
    # first add a waypoint
    add = {"action": "waypoint", "x": 500100.0, "y": 5599950.0,
           "lat": 0, "lng": 0, "seq": 1}
    invoke(app, ("layers-view.data", "edit-request"), add, True, [],
           triggered=[f"{ids.EDIT_REQUEST}.data"])
    assert len(state.get(state.active_route_id).meta["control_points"]) == 3

    # move it: nearest waypoint jumps to the clicked point (still 3 points)
    move = {"action": "move-waypoint", "x": 500110.0, "y": 5599955.0,
            "lat": 0, "lng": 0, "seq": 2}
    invoke(app, ("layers-view.data", "edit-request"), move, True, [],
           triggered=[f"{ids.EDIT_REQUEST}.data"])
    moved = state.get(state.active_route_id)
    assert moved.meta["edit"] == "moved waypoint"
    assert len(moved.meta["control_points"]) == 3
    waypoint = moved.meta["control_points"][1]
    assert abs(waypoint[0] - 500110.0) < 1 and abs(waypoint[1] - 5599955.0) < 1
    assert len(state.layers_of_kind("route")) == 1   # editing replaces


def test_move_waypoint_without_waypoint_warns(app, state, built_route):
    _select(app, state, built_route.id)           # a 2-point route, no waypoint
    move = {"action": "move-waypoint", "x": 500100.0, "y": 5599950.0,
            "lat": 0, "lng": 0, "seq": 1}
    resp = invoke(app, ("layers-view.data", "edit-request"), move, True, [],
                  triggered=[f"{ids.EDIT_REQUEST}.data"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "No waypoint to move"
    assert len(state.layers_of_kind("route")) == 1


def test_edit_staged_when_auto_refresh_off(app, state, built_route):
    """Task 64: with auto-refresh off, a map-click edit is STAGED (no
    recompute); the Refresh button then recomputes only the edited route."""
    _select(app, state, built_route.id)
    request = {"action": "target", "x": 500150.0, "y": 5599900.0,
               "lat": 0, "lng": 0, "seq": 1}
    resp = invoke(app, ("layers-view.data", "edit-request"), request, False,
                  [], triggered=[f"{ids.EDIT_REQUEST}.data"])
    # staged: same route (not replaced), flagged, dashed, no new layer
    routes = state.layers_of_kind("route")
    assert len(routes) == 1 and routes[0].id == built_route.id
    assert built_route.meta.get("needs_refresh") is True
    assert built_route.meta["control_points"][-1] == [500150.0, 5599900.0]
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Edit staged"
    assert resp[ids.ACTIVE_ROUTE]["dashArray"] == "8 8"

    # Refresh recomputes the staged route (replaces it, clears the flag)
    invoke(app, ("layers-view.data", ids.REFRESH_ROUTES_BTN), 1, [],
           triggered=[f"{ids.REFRESH_ROUTES_BTN}.n_clicks"])
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    refreshed = routes[0]
    assert refreshed.meta["edit"] == "refreshed"
    assert not refreshed.meta.get("needs_refresh")
    assert refreshed.meta["control_points"][-1] == [500150.0, 5599900.0]


def test_refresh_with_nothing_staged_informs(app, state, built_route):
    resp = invoke(app, ("layers-view.data", ids.REFRESH_ROUTES_BTN), 1, [],
                  triggered=[f"{ids.REFRESH_ROUTES_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Nothing to refresh"


def test_apply_points_validation(app, state, built_route):
    _select(app, state, built_route.id)
    bad_rows = [{"kind": "waypoint", "x": 1.0, "y": 2.0},
                {"kind": "target", "x": 3.0, "y": 4.0}]
    resp = invoke(app, ("layers-view.data", ids.POINTS_APPLY_BTN),
                  1, bad_rows, None, [],
                  triggered=[f"{ids.POINTS_APPLY_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Points list not valid"
    assert len(state.layers_of_kind("route")) == 1   # nothing recomputed


def test_apply_points_to_draft_when_deselected(app, state, built_route):
    _select(app, state, None)                        # deselect
    rows = [{"kind": "source", "x": 500020.0, "y": 5599980.0},
            {"kind": "target", "x": 500180.0, "y": 5599820.0}]
    resp = invoke(app, ("layers-view.data", ids.POINTS_APPLY_BTN),
                  1, rows, None, [],
                  triggered=[f"{ids.POINTS_APPLY_BTN}.n_clicks"])
    draft = resp[ids.ROUTE_DRAFT]["data"]
    assert len(draft["sources"]) == 1 and len(draft["targets"]) == 1
    # typed coordinates got WGS84 lat/lng for the map markers
    assert 50 < draft["sources"][0]["lat"] < 51


def test_edit_without_active_route_warns(app, state, built_route):
    state.active_route_id = None
    request = {"action": "target", "x": 1.0, "y": 2.0, "lat": 0, "lng": 0,
               "seq": 9}
    resp = invoke(app, ("layers-view.data", "edit-request"), request, True, [],
                  triggered=[f"{ids.EDIT_REQUEST}.data"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Select a route to edit first"


def test_delete_variant(app, state, built_route):
    _select(app, state, built_route.id)
    resp = invoke(app, ("layers-view.data", ids.DELETE_VARIANT_BTN), 1,
                  triggered=[f"{ids.DELETE_VARIANT_BTN}.n_clicks"])
    assert state.layers_of_kind("route") == []
    assert state.active_route_id is None


def test_posthoc_simplify_and_export(app, state, built_route, tmp_path):
    """Simplification is selectable AFTER planning; costs stay full-line."""
    import geopandas as gpd

    _select(app, state, built_route.id)
    full_coords = len(built_route.gdf.geometry.iloc[0].coords)
    cost_before = built_route.meta["metrics"]["total_cost"]

    resp = invoke(app, ("layers-view.data", ids.EDIT_SIMPLIFY_BTN),
                  1, 10.0, [],
                  triggered=[f"{ids.EDIT_SIMPLIFY_BTN}.n_clicks"])
    assert built_route.meta["simplify_tol"] == 10.0
    # gdf still holds the FULL line; only the display geojson is simplified
    assert len(built_route.gdf.geometry.iloc[0].coords) == full_coords
    display = built_route.geojson["features"][0]["geometry"]["coordinates"]
    assert len(display) < full_coords
    assert len(resp[ids.ACTIVE_ROUTE]["positions"]) == len(display)
    assert built_route.meta["metrics"]["total_cost"] == cost_before

    # export honors the tolerance
    target = tmp_path / "simplified.geojson"
    invoke(app, (f"{ids.EXPORT_STATUS}.children",), 1, str(target), [],
           triggered=[f"{ids.EXPORT_ROUTE_BTN}.n_clicks"])
    back = gpd.read_file(target)
    assert len(back.geometry.iloc[0].coords) < full_coords

    # tolerance 0 switches it off again
    invoke(app, ("layers-view.data", ids.EDIT_SIMPLIFY_BTN), 2, 0, [],
           triggered=[f"{ids.EDIT_SIMPLIFY_BTN}.n_clicks"])
    assert built_route.meta["simplify_tol"] is None
    display = built_route.geojson["features"][0]["geometry"]["coordinates"]
    assert len(display) == full_coords


def test_route_markers_in_layer_host(app, state, built_route):
    """Existing routes show green/red/yellow control-point markers (map)."""
    import json

    resp = invoke(app, ("layer-host.children",), state.layers_view())
    rendered = json.dumps(resp[ids.LAYER_HOST]["children"])
    assert "#2ca02c" in rendered      # source green
    assert "#d62728" in rendered      # target red
    # add a waypoint variant -> yellow marker appears
    _select(app, state, built_route.id)
    request = {"action": "waypoint", "x": 500100.0, "y": 5599950.0,
               "lat": 0, "lng": 0, "seq": 5}
    invoke(app, ("layers-view.data", "edit-request"), request, True, [],
           triggered=[f"{ids.EDIT_REQUEST}.data"])
    resp = invoke(app, ("layer-host.children",), state.layers_view())
    rendered = json.dumps(resp[ids.LAYER_HOST]["children"])
    assert "#f1c40f" in rendered      # waypoint yellow


def test_export_route(app, state, built_route, tmp_path):
    import geopandas as gpd

    _select(app, state, built_route.id)
    target = tmp_path / "route.geojson"
    resp = invoke(app, (f"{ids.EXPORT_STATUS}.children",), 1, str(target),
                  [], triggered=[f"{ids.EXPORT_ROUTE_BTN}.n_clicks"])
    assert "saved" in resp[ids.EXPORT_STATUS]["children"]
    back = gpd.read_file(target)
    assert back.iloc[0]["origin"] == "built"
    assert back.iloc[0]["route_id"] == built_route.id

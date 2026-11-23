"""Round 13: "Undo last edit" for routes (replace + one undo step).

An edit REPLACES the route; the replaced route layer is kept in memory as the one-step undo of the new
one (``state.route_undo``). Headless tests: the harness of ``conftest.py`` drives the callbacks as Dash
would. Nothing of the undo reaches the project file.
"""
from __future__ import annotations

import json

import pytest
from conftest import SOURCE, TARGET, TEST_CRS, invoke
from shapely.geometry import LineString
from test_routes_phase4 import _draft

from pyorps.gui import ids
from pyorps.gui.callbacks.routes import add_route_layer
from pyorps.gui.services import project_io, routing
from pyorps.gui.services.cost import RouteCost
from pyorps.gui.services.errors import NothingToUndoError

UNDO_WIRING = ("layers-view.data", ids.EDIT_UNDO_BTN)
EDIT_WIRING = ("layers-view.data", "edit-request")
UNDONE_TITLE = "Brought back the route as it was before the last edit"
_EMPTY_DRAFT = {"sources": [], "targets": [], "waypoints": []}


# ------------------------------------------------------------------ helpers
@pytest.fixture()
def route(app, state, raster_path):
    """A raster layer + one built route, exactly as the Routes tab produces them."""
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path, "viridis", [],
           triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    draft = _draft({"source": [SOURCE], "target": [TARGET]})
    invoke(app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
           1, False, draft, raster_layer.id, "dijkstra",
           None, 1, 80, False, True, False, False, 1.0, 100, 0, None, [], [],
           triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    built = state.layers_of_kind("route")[0]
    select(app, built.id)
    return built


def select(app, route_id):
    return invoke(app, (f"{ids.ACTIVE_ROUTE}.positions", ids.EDIT_ROUTE_SELECT, ids.ROUTE_DRAFT),
                  route_id, _EMPTY_DRAFT, triggered=[f"{ids.EDIT_ROUTE_SELECT}.value"])


def move(app, state, action="target", xy=(500150.0, 5599900.0), auto_refresh=True):
    """One click-to-place edit of the active route; returns the callback response."""
    request = {"action": action, "x": xy[0], "y": xy[1], "lat": 0, "lng": 0, "seq": 1}
    return invoke(app, EDIT_WIRING, request, auto_refresh, [], triggered=[f"{ids.EDIT_REQUEST}.data"])


def undo(app):
    return invoke(app, UNDO_WIRING, 1, [], triggered=[f"{ids.EDIT_UNDO_BTN}.n_clicks"])


def undo_disabled(app, state, route_id) -> bool:
    resp = invoke(app, f"{ids.EDIT_UNDO_BTN}.disabled", state.layers_view(), route_id)
    return resp[ids.EDIT_UNDO_BTN]["disabled"]


def the_route(state):
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    return routes[0]


def second_route(state, raster_layer_id):
    """A synthetic route B (never edited), added after the built one."""
    return add_route_layer(
        state, line=LineString([SOURCE, TARGET]),
        route_cost=RouteCost(total_length=226.0, total_cost=300.0, total_cell_cost=200.0, geodesic_length_m=226.0),
        params={"algorithm": "dijkstra", "graph_api": "cython", "neighborhood": "r2", "search_space_buffer_m": 80,
                "ignore_max_cost": True, "raster_layer_id": raster_layer_id},
        control_points=[list(SOURCE), list(TARGET)], crs=TEST_CRS, name="Route B")


def snapshot_of(layer):
    """What undo must give back, taken by value before an edit."""
    return {"id": layer.id, "name": layer.name, "line": layer.gdf.geometry.iloc[0].wkt,
            "points": [list(p) for p in layer.meta["control_points"]], "metrics": dict(layer.meta["metrics"]),
            "params": dict(layer.meta["params"])}


def assert_is(layer, before):
    assert layer.id == before["id"]
    assert layer.name == before["name"]
    assert layer.gdf.geometry.iloc[0].wkt == before["line"]
    assert layer.meta["control_points"] == before["points"]
    assert layer.meta["metrics"] == before["metrics"]
    assert layer.meta["params"] == before["params"]


# --------------------------------------------------------------------- state
def test_edit_keeps_the_replaced_route_as_snapshot(app, state, route):
    original = route
    move(app, state)
    edited = the_route(state)
    assert edited.id != original.id
    assert state.route_undo == {edited.id: original}
    assert state.get(original.id) is None              # the snapshot is not a layer of the project
    # only the Layer (line + metadata) is kept: no search session or finder of the replaced route
    assert original.id not in state.search_sessions
    assert original.id not in state.finders


def test_second_edit_keeps_only_the_latest_previous(app, state, route):
    move(app, state, xy=(500150.0, 5599900.0))
    first = the_route(state)
    first_state = snapshot_of(first)
    move(app, state, xy=(500160.0, 5599880.0))
    second = the_route(state)
    assert list(state.route_undo) == [second.id]       # the older step is gone
    assert state.route_undo[second.id] is first
    undo(app)
    restored = the_route(state)
    assert_is(restored, {**first_state, "id": first.id})
    assert state.route_undo == {}                      # one step only, no redo


def test_edit_after_undo_works_and_snapshots_again(app, state, route):
    move(app, state)
    undo(app)
    restored = the_route(state)
    move(app, state, xy=(500140.0, 5599910.0))
    edited = the_route(state)
    assert state.route_undo == {edited.id: restored}


# ------------------------------------------------------------------- service
def test_undo_restores_the_route_as_it_was(app, state, route):
    before = snapshot_of(route)
    route.style = {"color": "#00ff00"}
    route.meta["group"] = "Feeders"
    move(app, state)
    edited = the_route(state)
    assert edited.meta["control_points"] != before["points"]
    # the user restyles / regroups / hides the EDITED route; the restored one takes that place
    edited.style = {"color": "#123456", "dashArray": "8 8"}
    edited.meta["group"] = "Moved"
    edited.visible = False

    resp = undo(app)

    restored = the_route(state)
    assert_is(restored, before)                         # its OWN line, points, params, metrics, name
    assert state.get(edited.id) is None                 # the edited route is gone
    assert restored.style == {"color": "#123456", "dashArray": "8 8"}
    assert restored.meta["group"] == "Moved"
    assert restored.visible is False
    assert state.active_route_id == restored.id
    assert state.route_undo == {}
    # the callback hands the restored route to the page
    assert resp[ids.EDIT_ROUTE_SELECT]["value"] == restored.id
    assert resp[ids.EDIT_ROUTE_SELECT]["options"] == [{"label": restored.name, "value": restored.id}]
    assert resp[ids.NOTICES]["data"][-1]["title"] == UNDONE_TITLE
    assert len(resp[ids.CONTROL_MARKERS]["children"]) == 2
    rows = resp[ids.BUILD_POINTS_TABLE]["rowData"]
    assert [[r["x"], r["y"]] for r in rows] == before["points"]
    assert len(resp[ids.ACTIVE_ROUTE]["positions"]) >= 2
    assert "Length" in json.dumps(resp[ids.EDIT_COST_READOUT]["children"])
    assert resp[ids.LAYERS_VIEW]["data"] == state.layers_view()


def test_undo_keeps_the_place_in_the_layer_order(app, state, route, raster_path):
    raster_id = state.layers_of_kind("raster")[0].id
    other = second_route(state, raster_id)
    move(app, state)                                    # the edited route is added on top of the route band
    edited = state.get(state.active_route_id)
    order_edited = [ly.id for ly in state.ordered_layers()]
    assert order_edited[-1] == edited.id and other.id in order_edited
    undo(app)
    restored = state.get(state.active_route_id)
    order_after = [ly.id for ly in state.ordered_layers()]
    assert order_after == [restored.id if lid == edited.id else lid for lid in order_edited]
    assert restored.z == edited.z
    assert order_after == [raster_id, other.id, restored.id]     # where the edited route stood


def test_undo_without_snapshot_is_a_notice_not_an_exception(app, state, route):
    resp = undo(app)
    notice = resp[ids.NOTICES]["data"][-1]
    assert notice["title"] == "Nothing to undo" and notice["severity"] == "info"
    assert ids.EDIT_ROUTE_SELECT not in resp            # nothing else was touched
    assert the_route(state).id == route.id
    with pytest.raises(NothingToUndoError):
        routing.undo_last_edit(state, route.id)
    with pytest.raises(NothingToUndoError):
        routing.undo_last_edit(state, None)
    with pytest.raises(NothingToUndoError):
        routing.undo_last_edit(state, "no-such-layer")


def test_undo_with_nothing_selected_is_a_notice(app, state, route):
    move(app, state)
    state.active_route_id = None
    resp = undo(app)
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Nothing to undo"
    assert len(state.route_undo) == 1                   # the snapshot is untouched


# -------------------------------------------------------------------- button
def test_button_is_in_the_edit_card_with_its_labels(app):
    from dash import html

    found = [c for c in app.layout._traverse() if getattr(c, "id", None) == ids.EDIT_UNDO_BTN]
    assert len(found) == 1 and isinstance(found[0], html.Button)
    button = found[0]
    assert button.disabled is True                       # nothing to undo when the page opens
    assert button.title == "Bring back the route as it was before the last edit. One step only."
    assert "Undo last edit" in button.children
    assert button.to_plotly_json()["props"]["aria-label"].startswith("Undo last edit")


def test_button_state_before_edit_after_edit_and_after_undo(app, state, route):
    assert undo_disabled(app, state, route.id) is True   # before an edit
    assert undo_disabled(app, state, None) is True       # no route selected
    move(app, state)
    edited = the_route(state)
    assert undo_disabled(app, state, edited.id) is False  # after an edit
    undo(app)
    assert undo_disabled(app, state, the_route(state).id) is True   # after an undo: no redo


def test_an_edit_of_route_a_does_not_enable_undo_for_route_b(app, state, route):
    other = second_route(state, state.layers_of_kind("raster")[0].id)
    move(app, state)                                     # edits the active route A
    edited = state.get(state.active_route_id)
    assert edited.id != other.id
    assert undo_disabled(app, state, edited.id) is False
    assert undo_disabled(app, state, other.id) is True
    select(app, other.id)
    resp = undo(app)                                     # B is active: nothing to undo for it
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Nothing to undo"
    assert state.route_undo.keys() == {edited.id}


def test_overhead_line_route_cannot_be_edited_so_undo_stays_off(app, state, route):
    route.meta["params"]["constrained"] = True
    resp = move(app, state)
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Overhead-line routes cannot be edited here"
    assert state.route_undo == {}
    assert undo_disabled(app, state, route.id) is True


# ----------------------------------------------------------- snapshot clean-up
def test_snapshot_dropped_when_the_route_is_discarded(app, state, route):
    move(app, state)
    assert len(state.route_undo) == 1
    invoke(app, ("layers-view.data", ids.GO_DELETE_VARIANT), 1, triggered=[f"{ids.GO_DELETE_VARIANT}.data"])
    assert state.layers_of_kind("route") == [] and state.route_undo == {}


def test_snapshot_dropped_when_the_group_is_deleted(app, state, route):
    move(app, state)
    group = the_route(state).meta["group"]
    assert group and len(state.route_undo) == 1
    invoke(app, ("layers-view.data", ids.GO_DELETE_GROUP), 1, group, [],
           triggered=[f"{ids.GO_DELETE_GROUP}.data"])
    assert state.layers_of_kind("route") == [] and state.route_undo == {}


def test_snapshot_dropped_when_the_layer_is_removed_in_the_layers_tab(app, state, route):
    move(app, state)
    edited = the_route(state)
    invoke(app, ("layers-view.data", ids.GO_REMOVE_LAYERS), 1, [{"id": edited.id}],
           triggered=[f"{ids.GO_REMOVE_LAYERS}.data"])
    assert state.get(edited.id) is None and state.route_undo == {}


def test_snapshot_dropped_on_new_project(app, state, route):
    move(app, state)
    assert len(state.route_undo) == 1
    invoke(app, ("layers-view.data", ids.GO_NEW_PROJECT), 1, [], triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    assert state.layers == {} and state.route_undo == {}


def test_snapshot_dropped_on_open_project(app, state, route, tmp_path):
    manifest = project_io.manifest_path(tmp_path / "proj")
    project_io.save_project(state, manifest)
    move(app, state)
    assert len(state.route_undo) == 1
    invoke(app, ("layers-view.data", ids.GO_OPEN_PROJECT), 1, str(manifest), [],
           triggered=[f"{ids.GO_OPEN_PROJECT}.data"])
    assert state.route_undo == {}
    assert len(state.layers_of_kind("route")) == 1


def test_nothing_about_undo_is_saved_in_the_project_file(app, state, route, tmp_path):
    move(app, state)                                     # a snapshot exists
    move(app, state, xy=(500140.0, 5599910.0), auto_refresh=False)   # and a staged edit on top of it
    manifest = project_io.manifest_path(tmp_path / "proj")
    project_io.save_project(state, manifest)
    text = manifest.read_text(encoding="utf-8").lower()
    assert "undo" not in text and "snapshot" not in text and "staged_from" not in text
    for gpkg in (tmp_path / "proj" / "assets").glob("*.gpkg"):
        assert b"undo" not in gpkg.read_bytes().lower()


# ------------------------------------------------------------- staged edits
def test_staged_edit_snapshots_when_it_is_applied_not_when_staged(app, state, route):
    before = snapshot_of(route)
    move(app, state, auto_refresh=False)                 # staged: same layer, no recompute
    assert the_route(state).id == route.id
    assert state.route_undo == {}
    assert undo_disabled(app, state, route.id) is True
    assert route.meta["control_points"] != before["points"]      # the markers moved

    move(app, state, xy=(500140.0, 5599910.0), auto_refresh=False)   # a second staging keeps the FIRST original
    assert state.route_undo == {}

    invoke(app, ("layers-view.data", ids.REFRESH_ROUTES_BTN), 1, [], triggered=[f"{ids.REFRESH_ROUTES_BTN}.n_clicks"])
    refreshed = the_route(state)
    assert refreshed.id != route.id
    assert state.route_undo == {refreshed.id: route}     # taken now, at apply time
    assert undo_disabled(app, state, refreshed.id) is False
    # the snapshot is the route as it was BEFORE the staging, without staging marks
    assert route.meta["control_points"] == before["points"]
    assert "needs_refresh" not in route.meta and "staged_from" not in route.meta

    undo(app)
    assert_is(the_route(state), before)

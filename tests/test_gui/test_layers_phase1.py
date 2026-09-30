"""Phase 1: layer host render (C1), Layers tab, attribute inspector."""
import json

import pytest

from pyorps.gui import ids
from pyorps.gui.callbacks.layers import render_layer, render_layer_host

from conftest import invoke

FC = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [9, 50]},
     "properties": {"name": "spot", "cost": 42}},
]}


@pytest.fixture()
def three_layers(state):
    a = state.add_layer("Area", "study_area", geojson=FC)
    b = state.add_layer("Data", "vector", geojson=FC)
    c = state.add_layer("Route 1", "route", geojson=FC)
    return a, b, c


# ------------------------------------------------------------- pure renderers
def test_render_layer_kinds(state, three_layers):
    a, b, c = three_layers
    comp = render_layer(b)
    assert comp.id == {"type": ids.TYPE_LAYER_GEOJSON, "id": b.id}
    assert comp.data == FC
    # study area gets its dashed style
    area = render_layer(a)
    assert area.options["style"]["dashArray"]


def test_render_layer_hidden_is_none(state, three_layers):
    a, *_ = three_layers
    state.set_visible(a.id, False)
    assert render_layer(a) is None


def test_render_layer_host_order(state, three_layers):
    a, b, c = three_layers
    host = render_layer_host(state)
    assert [comp.id["id"] for comp in host] == [a.id, b.id, c.id]
    state.reorder([c.id, b.id, a.id])
    host = render_layer_host(state)
    assert [comp.id["id"] for comp in host] == [c.id, b.id, a.id]


# --------------------------------------------------------- headless callbacks
def test_render_host_callback(app, state, three_layers):
    resp = invoke(app, ("layer-host.children",), state.layers_view())
    children = resp[ids.LAYER_HOST]["children"]
    assert len(children) == 3


def test_grid_mirror_reversed(app, state, three_layers):
    a, b, c = three_layers
    resp = invoke(app, (f"{ids.LAYERS_GRID}.rowData",), state.layers_view())
    rows = resp[ids.LAYERS_GRID]["rowData"]
    # panel shows topmost (highest z) first; route layers collapse into ONE
    # row per group (round 11) — c has no group -> "(ungrouped)"
    assert [r["id"] for r in rows] == ["group::(ungrouped)", b.id, a.id]
    assert rows[0]["kind"] == "routes (1)"


def test_cell_edit_visibility_and_name(app, state, three_layers):
    a, b, c = three_layers
    event = [{"colId": "visible",
              "data": {"id": b.id, "name": "Data", "visible": False}}]
    resp = invoke(app, ("layers-view.data", ids.LAYERS_GRID, "cellValueChanged"), event,
                  triggered=[f"{ids.LAYERS_GRID}.cellValueChanged"])
    assert state.get(b.id).visible is False
    event = [{"colId": "name",
              "data": {"id": b.id, "name": "Renamed", "visible": False}}]
    invoke(app, ("layers-view.data", ids.LAYERS_GRID, "cellValueChanged"), event,
           triggered=[f"{ids.LAYERS_GRID}.cellValueChanged"])
    assert state.get(b.id).name == "Renamed"


def test_move_layer_up_down(app, state, three_layers):
    a, b, c = three_layers
    # move A up (drawn later): [B, A, C]
    resp = invoke(app, ("layers-view.data", "layer-up-btn"),
                  1, None, [{"id": a.id}],
                  triggered=["layer-up-btn.n_clicks"])
    assert [v["id"] for v in resp[ids.LAYERS_VIEW]["data"]] == \
        [b.id, a.id, c.id]
    # move A back down
    resp = invoke(app, ("layers-view.data", "layer-up-btn"),
                  None, 1, [{"id": a.id}],
                  triggered=["layer-down-btn.n_clicks"])
    assert [v["id"] for v in resp[ids.LAYERS_VIEW]["data"]] == \
        [a.id, b.id, c.id]


def test_remove_layer_callback(app, state, three_layers):
    a, b, c = three_layers
    resp = invoke(app, ("layers-view.data", ids.LAYER_REMOVE_BTN),
                  1, [{"id": c.id}],
                  triggered=[f"{ids.LAYER_REMOVE_BTN}.n_clicks"])
    assert state.get(c.id) is None
    assert [v["id"] for v in resp[ids.LAYERS_VIEW]["data"]] == [a.id, b.id]


def test_attribute_inspector(app, state, three_layers):
    _, b, _ = three_layers
    feature = FC["features"][0]
    resp = invoke(
        app, (f"{ids.ATTR_PANEL}.children",), [feature], None,
        triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON, "id": b.id},
                    "property": "clickData", "value": feature}])
    rendered = json.dumps(resp[ids.ATTR_PANEL])
    assert "spot" in rendered and "42" in rendered
    assert "Data" in rendered  # layer name as title


def test_feature_select_highlights_when_layer_selected(app, state,
                                                       three_layers):
    """Feature 3: clicking a feature of the *selected* layer highlights it."""
    _, b, _ = three_layers
    feature = FC["features"][0]
    resp = invoke(
        app, (f"{ids.ATTR_PANEL}.children",), [feature], [{"id": b.id}],
        triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON, "id": b.id},
                    "property": "clickData", "value": feature}])
    highlight = resp[ids.SELECTED_FEATURE_LAYER]["data"]
    assert highlight["features"] == [feature]
    assert resp[ids.SELECTED_FEATURE]["data"]["layer_id"] == b.id

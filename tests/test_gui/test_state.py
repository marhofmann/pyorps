"""Unit tests: ProjectState layer CRUD, ordering, views."""
import pytest

from pyorps.gui.state import ProjectState


def test_add_and_order(state):
    a = state.add_layer("A", "vector")
    b = state.add_layer("B", "raster")
    c = state.add_layer("C", "route")
    assert [ly.id for ly in state.ordered_layers()] == [a.id, b.id, c.id]
    assert [v["name"] for v in state.layers_view()] == ["A", "B", "C"]

    state.reorder([c.id, a.id, b.id])
    assert [v["name"] for v in state.layers_view()] == ["C", "A", "B"]


def test_visibility_rename_remove(state):
    a = state.add_layer("A", "vector")
    state.set_visible(a.id, False)
    assert state.get(a.id).visible is False
    state.rename(a.id, "Renamed")
    assert state.get(a.id).name == "Renamed"
    state.remove_layer(a.id)
    assert state.get(a.id) is None
    # removing twice is a no-op
    state.remove_layer(a.id)


def test_unknown_kind_rejected(state):
    with pytest.raises(ValueError, match="kind"):
        state.add_layer("X", "sandwich")


def test_layers_of_kind_and_route_names(state):
    state.add_layer("A", "vector")
    state.add_layer("B", "raster")
    assert [ly.name for ly in state.layers_of_kind("raster")] == ["B"]
    assert state.next_route_name() == "Route 1"
    assert state.next_route_name() == "Route 2"


def test_clear_resets_everything(state):
    state.add_layer("A", "vector")
    state.study_area = {"type": "Polygon", "coordinates": []}
    state.active_route_id = "x"
    state.clear()
    assert state.layers == {}
    assert state.study_area is None
    assert state.active_route_id is None
    assert state.search_sessions == {}


def test_remove_layer_closes_search_session(state):
    class _Session:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    layer = state.add_layer("R", "route")
    session = _Session()
    state.search_sessions[layer.id] = session
    state.remove_layer(layer.id)
    assert session.closed
    assert layer.id not in state.search_sessions

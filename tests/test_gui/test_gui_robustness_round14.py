"""Round 14 robustness (headless): tile clients are released, grids use the AG Grid 32.2+ selection options.

Fix 1: a raster layer's ``localtileserver`` client holds the ``.tif`` open (its reader) until it is closed. The
``TileManager`` of the project state owns every client and releases each exactly once when its layer is removed or
replaced, when the project is cleared / replaced, and at ``state.shutdown()``. A release that fails never breaks the
user's action: it is written to the logbook and handed to the callbacks that can show a notification box.

Fix 3: the grids pass ``rowSelection`` as an object (the string forms and ``suppressRowDeselection`` are deprecated
since AG Grid 32.2), spelled out so the behaviour of every grid is the one it always had.
"""
from __future__ import annotations

import math
import os
import shutil
from types import SimpleNamespace

import pytest
from conftest import invoke

from pyorps.gui import ids
from pyorps.gui.services import logbook, project_io
from pyorps.gui.state import ProjectState, TileManager

# --------------------------------------------------------------------- helpers


class _Reader:
    def __init__(self, fail: bool = False) -> None:
        self.closed = 0
        self.fail = fail

    def close(self) -> None:
        self.closed += 1
        if self.fail:
            raise OSError("reader refuses to close")


class _Client:
    """Stand-in for a ``TileClient``: counts the calls the manager makes."""

    def __init__(self, *, fail_shutdown: bool = False, fail_close: bool = False) -> None:
        self.shutdowns = 0
        self.fail_shutdown = fail_shutdown
        self.reader = _Reader(fail=fail_close)
        self.filename = "some_raster.tif"

    def shutdown(self) -> None:
        self.shutdowns += 1
        if self.fail_shutdown:
            raise RuntimeError("server refuses to stop")


def _fake_layer(state: ProjectState, client: _Client, layer_id: str | None = None):
    return state.add_layer("raster", "raster", tile=SimpleNamespace(tile_client=client), layer_id=layer_id)


@pytest.fixture(autouse=True)
def _no_stale_failures():
    TileManager.pop_notices()
    yield
    TileManager.pop_notices()


def _released(client: _Client) -> bool:
    return client.shutdowns == 1 and client.reader.closed == 1


# ------------------------------------------------- fix 1: who releases a client
def test_removing_a_layer_releases_its_client_once(state):
    client = _Client()
    layer = _fake_layer(state, client)
    assert client.shutdowns == 0
    state.remove_layer(layer.id)
    assert _released(client)
    state.remove_layer(layer.id)                     # a second remove has nothing left to release
    state.shutdown()
    assert _released(client)


def test_a_layer_replaced_under_the_same_id_releases_the_old_client(state):
    first, second = _Client(), _Client()
    _fake_layer(state, first, layer_id="same")
    _fake_layer(state, second, layer_id="same")
    assert _released(first)
    assert second.shutdowns == 0
    state.remove_layer("same")
    assert _released(second)


def test_the_same_client_registered_again_is_not_released(state):
    client = _Client()
    layer = _fake_layer(state, client, layer_id="one")
    assert state.tiles.register("one", layer.tile) is None
    assert client.shutdowns == 0


def test_a_replacement_without_a_tile_releases_the_old_client(state):
    client = _Client()
    _fake_layer(state, client, layer_id="reused")
    state.add_layer("now a vector", "vector", geojson={"type": "FeatureCollection", "features": []},
                    layer_id="reused")
    assert _released(client)


def test_clear_releases_every_client(state):
    clients = [_Client(), _Client(), _Client()]
    for client in clients:
        _fake_layer(state, client)
    state.clear()
    assert all(_released(c) for c in clients)
    assert not state.layers


def test_shutdown_releases_every_client_once(state):
    clients = [_Client(), _Client()]
    for client in clients:
        _fake_layer(state, client)
    state.shutdown()
    state.shutdown()
    assert all(_released(c) for c in clients)


def test_a_staged_but_never_adopted_layer_is_released(tmp_path):
    client = _Client()
    project_io._release_staged([SimpleNamespace(tile=SimpleNamespace(tile_client=client))])
    assert _released(client)


# ------------------------------------------- fix 1: a failing release is not silent
def test_a_failing_release_never_breaks_the_action_and_is_logged(state):
    bad = _Client(fail_shutdown=True, fail_close=True)
    good = _Client()
    bad_layer = _fake_layer(state, bad)
    good_layer = _fake_layer(state, good)
    state.remove_layer(bad_layer.id)                 # does not raise
    assert state.get(bad_layer.id) is None
    state.remove_layer(good_layer.id)                # the next release still happens
    assert _released(good)
    assert bad.shutdowns == 1 and bad.reader.closed == 1    # both steps were tried

    notices = TileManager.pop_notices()
    assert len(notices) == 2                         # one per failed step
    assert all(n["severity"] == "warning" for n in notices)
    text = " ".join(n["meaning"] for n in notices)
    assert "stop the tile server" in text and "close the raster reader" in text
    assert "some_raster.tif" in text
    assert TileManager.pop_notices() == []           # handed over once

    log = logbook.read_log()
    assert "could not be released cleanly" in log
    assert "reader refuses to close" in log or "server refuses to stop" in log


def test_a_failing_release_during_clear_is_reported(state):
    _fake_layer(state, _Client(fail_close=True))
    state.clear()
    assert [n["severity"] for n in TileManager.pop_notices()] == ["warning"]


def test_new_project_shows_a_notice_for_a_failed_release(app, state):
    _fake_layer(state, _Client(fail_shutdown=True))
    resp = invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1, [],
                  triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    titles = [n["title"] for n in resp[ids.NOTICES]["data"]]
    assert "A raster layer's tile server could not be released cleanly" in titles
    assert "New project" in titles
    assert not state.layers


def test_new_project_without_a_failure_shows_no_release_notice(app, state):
    _fake_layer(state, _Client())
    resp = invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1, [],
                  triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    assert [n["title"] for n in resp[ids.NOTICES]["data"]] == ["New project"]


def test_a_failure_of_an_earlier_action_is_not_shown_in_a_later_one(app, state):
    layer = _fake_layer(state, _Client(fail_shutdown=True))
    state.remove_layer(layer.id)                     # logged; nobody drained it
    resp = invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1, [],
                  triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    assert [n["title"] for n in resp[ids.NOTICES]["data"]] == ["New project"]


def test_stop_server_without_a_live_server_is_silent():
    pytest.importorskip("server_thread")
    TileManager.stop_server()
    TileManager.stop_server()                        # twice: nothing to stop, nothing raised
    assert TileManager.pop_notices() == []


# --------------------------------------- fix 1: the real thing (file lock, server)
def _centre_tile_url(layer) -> str:
    (s, w), (n, e) = layer.tile.bounds
    lat, lon = (s + n) / 2, (w + e) / 2
    z = 15
    nn = 2 ** z
    x = int((lon + 180) / 360 * nn)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * nn)
    return layer.tile.tile_url.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))


def _raster_copy(raster_path, tmp_path, name: str) -> str:
    target = tmp_path / name
    shutil.copy(raster_path, target)
    return str(target)


def test_the_raster_file_is_free_after_its_layer_is_removed(state, raster_path, tmp_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    tif = _raster_copy(raster_path, tmp_path, "removed.tif")
    layer, _ = add_raster_layer(state, tif, [])
    assert layer is not None
    state.remove_layer(layer.id)
    os.remove(tif)                                   # PermissionError on Windows while a client holds it
    assert not os.path.exists(tif)


def test_the_raster_file_can_be_overwritten_after_clear(state, raster_path, tmp_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    tif = _raster_copy(raster_path, tmp_path, "cleared.tif")
    layer, _ = add_raster_layer(state, tif, [])
    assert layer is not None
    state.clear()
    replacement = _raster_copy(raster_path, tmp_path, "replacement.tif")
    os.replace(replacement, tif)                     # overwrite in place
    assert os.path.exists(tif)


def test_the_raster_file_is_free_after_shutdown(state, raster_path, tmp_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    tif = _raster_copy(raster_path, tmp_path, "shutdown.tif")
    layer, _ = add_raster_layer(state, tif, [])
    assert layer is not None
    state.shutdown()
    os.remove(tif)


def test_a_live_layer_keeps_its_file_and_tiles_after_another_layer_is_removed(state, raster_path, tmp_path):
    requests = pytest.importorskip("requests")
    from pyorps.gui.callbacks.raster import add_raster_layer

    first_tif = _raster_copy(raster_path, tmp_path, "first.tif")
    second_tif = _raster_copy(raster_path, tmp_path, "second.tif")
    first, _ = add_raster_layer(state, first_tif, [])
    second, _ = add_raster_layer(state, second_tif, [])
    state.remove_layer(first.id)
    os.remove(first_tif)
    resp = requests.get(_centre_tile_url(second), timeout=30)
    assert resp.status_code == 200                   # the shared tile server still serves the other layer
    assert resp.headers["content-type"].startswith("image/")


# ----------------------------------------------- fix 3: AG Grid selection options
#: grid id -> (multi-select, a click never deselects)
_EXPECTED_SELECTION = {
    ids.COST_GRID: (True, False),
    ids.MODIFIER_GRID: (False, False),
    ids.PREPROC_GRID: (False, False),
    ids.MANUAL_GRID: (True, False),
    ids.BUILD_POINTS_TABLE: (True, False),
    ids.ROUTES_GRID: (True, False),
    ids.LAYERS_GRID: (True, True),
    ids.LAYER_TABLE_GRID: (False, False),
    ids.OHLF_ANGLE_GRID: (True, False),
    ids.OHLF_TERRAIN_GRID: (True, False),
    ids.OHLF_TYPES_GRID: (True, False),
    ids.OBJ_LAYERS_GRID: (True, False),
}


@pytest.fixture(scope="module")
def grids():
    from pyorps.gui.layout import build_layout

    layout = build_layout(ProjectState())
    found = {}
    for component in layout._traverse():
        options = getattr(component, "dashGridOptions", None)
        if options is not None and getattr(component, "id", None):
            found[component.id] = options
    return found


def test_every_grid_is_covered(grids):
    assert set(grids) == set(_EXPECTED_SELECTION)


@pytest.mark.parametrize("grid_id", sorted(_EXPECTED_SELECTION))
def test_grid_selection_uses_the_object_form(grids, grid_id):
    multi, keep = _EXPECTED_SELECTION[grid_id]
    options = grids[grid_id]
    selection = options["rowSelection"]
    assert isinstance(selection, dict), "the string forms of rowSelection are deprecated since AG Grid 32.2"
    assert selection["mode"] == ("multiRow" if multi else "singleRow")
    assert selection["checkboxes"] is False          # the old option showed no checkbox column
    assert ("headerCheckbox" in selection) == multi
    if multi:
        assert selection["headerCheckbox"] is False
    # a click selects a row (the old default); keep = the old suppressRowDeselection
    assert selection["enableClickSelection"] == ("enableSelection" if keep else True)
    assert "suppressRowDeselection" not in options
    assert "suppressRowClickSelection" not in options


def test_no_grid_options_are_built_elsewhere():
    """The grids are all defined in layout.py: a new one must be added to ``_EXPECTED_SELECTION``."""
    import pathlib

    gui = pathlib.Path(__file__).resolve().parents[2] / "pyorps" / "gui"
    offenders = [p.name for p in gui.rglob("*.py")
                 if p.name != "layout.py" and "dashGridOptions" in p.read_text("utf-8")]
    assert offenders == []

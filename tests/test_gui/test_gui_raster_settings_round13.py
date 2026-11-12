"""Round 13: the rasterize settings, the confirm gates, limits and labels of the GUI.

Headless callback tests in the style of ``test_gui_routing_settings_round13.py``: the harness of
``conftest.py`` drives the callbacks as Dash would. Every new setting has a test that the value set in the
page arrives at the library call (a spy at ``GeoRasterizer.rasterize`` / ``shrink_raster``), one for its
default and one for an invalid value (a field notice, no exception). The destructive buttons ask first
(one in-page modal), the layout is walked for labels, and a project round-trips the new settings.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import dash_bootstrap_components as dbc
import geopandas as gpd
import pytest
from conftest import SOURCE, TARGET, TEST_CRS, invoke
from dash import dcc, html
from dash.exceptions import PreventUpdate
from shapely.geometry import LineString, box

from pyorps import GeoRasterizer
from pyorps.gui import ids
from pyorps.gui.callbacks import confirm as confirm_cb
from pyorps.gui.callbacks import io as io_cb
from pyorps.gui.callbacks import raster as raster_cb
from pyorps.gui.callbacks.routes import add_route_layer
from pyorps.gui.layout import build_layout
from pyorps.gui.services import cost_model, geo, project_io, rasterize, routing
from pyorps.gui.services.cost import RouteCost

CRS = "EPSG:25832"


# ------------------------------------------------------------------ helpers
def _notices(resp) -> list[dict]:
    return resp[ids.NOTICES]["data"]


def _children(component):
    """The child components of a Dash component (a list, one component or text)."""
    children = getattr(component, "children", None)
    if children is None:
        return []
    return children if isinstance(children, (list, tuple)) else [children]


def _walk(component, parents=()):
    """``(component, parents)`` for every component below ``component``, depth first."""
    yield component, parents
    for child in _children(component):
        if hasattr(child, "_traverse"):
            yield from _walk(child, (*parents, component))


def _text(component) -> str:
    """The visible text of a component tree."""
    if isinstance(component, str):
        return component
    if not hasattr(component, "_traverse"):
        return ""
    return " ".join(_text(child) for child in _children(component))


def _by_id(layout) -> dict:
    return {c.id: c for c, _ in _walk(layout) if isinstance(getattr(c, "id", None), str)}


def _parent_of(layout, wanted_id):
    for component, parents in _walk(layout):
        if getattr(component, "id", None) == wanted_id:
            return parents[-1]
    raise AssertionError(wanted_id)


def _help_of(layout, wanted_id) -> str:
    """The hover help (``title``) of the nearest ancestor of a control that carries one."""
    for component, parents in _walk(layout):
        if getattr(component, "id", None) == wanted_id:
            for parent in reversed(parents):
                if getattr(parent, "title", None):
                    return parent.title
    raise AssertionError(f"no help text above {wanted_id}")


def _spy(monkeypatch, owner, name: str) -> list[tuple]:
    """Record every call of ``owner.name`` (class or module attribute) and run it as before."""
    calls: list[tuple] = []
    original = getattr(owner, name)

    if isinstance(owner, type):
        def wrapper(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original(self, *args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


def _landuse_gdf():
    return gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Wald", "Weg", "Sumpf"], "bez": ["Nadelholz", "Laubholz", "", ""]},
        geometry=[box(0, 0, 50, 50), box(50, 0, 100, 50), box(0, 50, 50, 100), box(50, 50, 100, 100)], crs=CRS)


def _add_vector(state, name, gdf, meta=None):
    return state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs, geojson=geo.gdf_to_wgs84_geojson(gdf),
                           meta=meta or {})


def _route(state, name="Route 1", group=None):
    layer = add_route_layer(
        state, line=LineString([SOURCE, TARGET]),
        route_cost=RouteCost(total_length=226.0, total_cost=300.0, total_cell_cost=200.0, geodesic_length_m=226.0),
        params={}, control_points=[list(SOURCE), list(TARGET)], crs=TEST_CRS, name=name)
    if group:
        layer.meta["group"] = group
    return layer


@pytest.fixture()
def layout(state):
    return build_layout(state)


@pytest.fixture()
def inputs(state):
    """A loaded vector layer + a seeded cost table, ready for "Build"."""
    layer = _add_vector(state, "landuse", _landuse_gdf())
    keys = ("nutzart", "bez")
    rows = cost_model.grid_rows_from_assumptions(cost_model.seed_assumptions(layer.gdf, keys), keys)
    return layer, rows, {"dataset_id": layer.id, "feature_keys": list(keys)}


@pytest.fixture()
def no_tiles(monkeypatch):
    """Skip serving the built raster (a tile server per test would be slow): the build itself still runs."""
    monkeypatch.setattr(raster_cb, "add_raster_layer", lambda state, source, notices, **kw: (None, notices))


@pytest.fixture()
def fake_build(monkeypatch):
    """Replace the service: the keywords the callback hands to ``build_cost_raster``."""
    calls: list[dict] = []

    def build(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(rasterize, "build_cost_raster", build)
    return calls


def _build(app, inputs, *, modifiers=None, preproc="none", resolution=1.0, fill_value=65535, dtype="uint16",
           geom_buffer=1, save_path=None, widen=False, touched=False, on_thin="warn", crop=False):
    _layer, rows, grid_state = inputs
    return invoke(
        app, (f"{ids.RASTERIZE_LOG}.children", ids.GO_RASTERIZE),
        1, rows, grid_state, modifiers or [], preproc, 10, 4, 2, [], resolution, fill_value, dtype, geom_buffer,
        save_path, "viridis", [], "", "current", widen, touched, on_thin, crop,
        triggered=[f"{ids.GO_RASTERIZE}.data"])


# ======================================================== A: the thin-feature burn controls
def test_the_thin_feature_controls_are_in_the_rasterization_advanced_section(layout):
    by_id = _by_id(layout)
    for control, kind in ((ids.WIDEN_THIN, dbc.Switch), (ids.ALL_TOUCHED, dbc.Switch), (ids.ON_THIN, dbc.Select),
                          (ids.CROP_BORDER, dbc.Switch)):
        assert isinstance(by_id[control], kind), control
    items = [c for c, _ in _walk(layout) if isinstance(c, dbc.AccordionItem) and c.title == "Advanced"]
    advanced = next(a for a in items if any(getattr(c, "id", None) == ids.FILL_VALUE for c, _ in _walk(a)))
    inside = {getattr(c, "id", None) for c, _ in _walk(advanced)}
    assert {ids.WIDEN_THIN, ids.ALL_TOUCHED, ids.ON_THIN, ids.CROP_BORDER} <= inside


def test_the_controls_start_on_the_librarys_defaults(layout):
    """The library's own defaults are preselected (the geometry buffer is the one documented exception)."""
    parameters = inspect.signature(GeoRasterizer.rasterize).parameters
    by_id = _by_id(layout)
    assert by_id[ids.WIDEN_THIN].value is parameters["widen_thin_forbidden"].default is False
    assert by_id[ids.ALL_TOUCHED].value is parameters["all_touched"].default is False
    assert by_id[ids.ON_THIN].value == parameters["on_thin_features"].default == "warn"
    assert [o["value"] for o in by_id[ids.ON_THIN].options] == list(rasterize.THIN_POLICIES)
    assert by_id[ids.RASTER_DTYPE].value == parameters["dtype"].default
    assert by_id[ids.FILL_VALUE].value == parameters["fill_value"].default
    assert by_id[ids.RES_M].value == parameters["resolution_in_m"].default
    assert by_id[ids.CROP_BORDER].value is False
    assert rasterize.RASTER_SETTING_DEFAULTS["geometry_buffer_m"] == 1 != parameters["geometry_buffer_m"].default


@pytest.mark.parametrize("widen,touched,on_thin", [
    (True, False, "warn"), (False, True, "warn"), (False, False, "raise"), (False, False, "ignore"),
    (True, True, "ignore")])
def test_the_gui_values_reach_georasterizer_rasterize(app, state, inputs, no_tiles, monkeypatch, widen, touched,
                                                      on_thin):
    calls = _spy(monkeypatch, GeoRasterizer, "rasterize")
    _build(app, inputs, widen=widen, touched=touched, on_thin=on_thin)
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert (kwargs["widen_thin_forbidden"], kwargs["all_touched"], kwargs["on_thin_features"]) == (
        widen, touched, on_thin)


def test_the_default_values_reach_georasterizer_as_the_librarys_defaults(app, state, inputs, no_tiles, monkeypatch):
    calls = _spy(monkeypatch, GeoRasterizer, "rasterize")
    _build(app, inputs)
    kwargs = calls[0][1]
    assert (kwargs["widen_thin_forbidden"], kwargs["all_touched"], kwargs["on_thin_features"]) == (False, False, "warn")


def test_blank_switches_and_policy_mean_the_defaults(app, state, inputs, fake_build):
    _build(app, inputs, widen=None, touched=None, on_thin=None, crop=None)
    call = fake_build[0]
    assert (call["widen_thin_forbidden"], call["all_touched"], call["on_thin_features"], call["crop_border"]) == (
        False, False, "warn", False)


def test_an_unusable_thin_policy_is_a_field_notice_and_builds_nothing(app, state, inputs, fake_build):
    resp = _build(app, inputs, on_thin="shout")
    assert fake_build == []
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.ON_THIN and notice["focus_id"] == ids.TAB_RASTER
    assert "On thin features" in notice["title"] and "warn, raise, ignore" in notice["meaning"]


def test_an_unusable_data_type_is_a_field_notice_and_builds_nothing(app, state, inputs, fake_build):
    notice = _notices(_build(app, inputs, dtype="int8"))[-1]
    assert fake_build == [] and notice["focus_control"] == ids.RASTER_DTYPE


def test_the_thin_settings_are_in_the_layers_rasterize_config(app, state, inputs, monkeypatch):
    seen = {}
    real = raster_cb.add_raster_layer

    def keep(state_, source, notices, **kwargs):
        layer, notices = real(state_, source, notices, **kwargs)
        seen["layer"] = layer
        return layer, notices

    monkeypatch.setattr(raster_cb, "add_raster_layer", keep)
    _build(app, inputs, widen=True, touched=True, on_thin="ignore", crop=True)
    config = seen["layer"].meta["rasterize_config"]
    assert (config["widen_thin_forbidden"], config["all_touched"], config["on_thin_features"],
            config["crop_border"]) == (True, True, "ignore", True)


# ----------------------------------------------------------------- the cache key
def _hash_args(**override):
    args = dict(feature_keys=("nutzart",), assumptions={"Wald": 100}, resolution_in_m=1.0, fill_value=65535,
                dtype="uint16", geometry_buffer_m=1.0, bounds=(0, 0, 100, 100), preprocessor=None,
                preprocessor_params={}, modifier_meta=[], n_features=4)
    args.update(override)
    return args


@pytest.mark.parametrize("override", [
    {"widen_thin_forbidden": True}, {"all_touched": True}, {"on_thin_features": "raise"},
    {"on_thin_features": "ignore"}, {"crop_border": True}])
def test_each_new_setting_changes_the_cache_key(override):
    assert rasterize.config_hash(**_hash_args()) != rasterize.config_hash(**_hash_args(**override))
    assert rasterize.config_hash(**_hash_args()) == rasterize.config_hash(**_hash_args(on_thin_features="warn"))


def _service_args(**override):
    args = dict(base_gdf=_landuse_gdf(), assumptions={"Wald": {"Nadelholz": 100, "Laubholz": 120, "": 150},
                                                       "Weg": {"": 50}, "Sumpf": {"": 65535}},
                feature_keys=("nutzart", "bez"), geometry_buffer_m=0.0)
    args.update(override)
    return args


def test_a_changed_setting_does_not_return_the_cached_raster(tmp_path, monkeypatch):
    calls = _spy(monkeypatch, GeoRasterizer, "rasterize")
    first, log = rasterize.build_cost_raster(work_dir=tmp_path, **_service_args())
    again, log_again = rasterize.build_cost_raster(work_dir=tmp_path, **_service_args())
    assert again == first and any("cache hit" in line for line in log_again) and len(calls) == 1
    for override in ({"all_touched": True}, {"widen_thin_forbidden": True}, {"on_thin_features": "ignore"},
                     {"crop_border": True}):
        path, log_changed = rasterize.build_cost_raster(work_dir=tmp_path, **_service_args(**override))
        assert path != first, override
        assert not any("cache hit" in line for line in log_changed), override
    assert len(calls) == 5                                 # every changed setting rebuilt


# ======================================================== B: the crop switch
def test_the_crop_switch_calls_shrink_raster_after_the_burn(app, state, inputs, no_tiles, monkeypatch):
    shrink = _spy(monkeypatch, GeoRasterizer, "shrink_raster")
    _build(app, inputs, crop=True)
    assert len(shrink) == 1 and shrink[0][0] == (65535,)


def test_without_the_crop_switch_shrink_raster_is_not_called(app, state, inputs, no_tiles, monkeypatch):
    shrink = _spy(monkeypatch, GeoRasterizer, "shrink_raster")
    _build(app, inputs)
    assert shrink == []


def test_the_crop_removes_the_forbidden_border_and_only_that(tmp_path):
    import rasterio
    from shapely.geometry import box as _box

    study_area = _box(-50, -50, 150, 150)        # 200 x 200 m: a 50 m forbidden border around the data
    kwargs = _service_args(bounding_polygon=study_area, geometry_buffer_m=0.0)
    plain, _log = rasterize.build_cost_raster(work_dir=tmp_path, **kwargs)
    cropped, log = rasterize.build_cost_raster(work_dir=tmp_path, crop_border=True, **kwargs)
    with rasterio.open(plain) as src:
        assert (src.height, src.width) == (200, 200)
    with rasterio.open(cropped) as src:
        assert src.height <= 100 + 2 and src.width <= 100 + 2          # the border is gone
        assert src.read(1).min() < 65535                                # passable cells are still there
        assert src.bounds.left >= -1 and src.bounds.right <= 101       # still georeferenced on the data
    assert any("cropped" in line for line in log)


def test_a_raster_without_a_forbidden_border_is_left_alone(tmp_path):
    import rasterio

    cropped, log = rasterize.build_cost_raster(
        work_dir=tmp_path, crop_border=True,
        **_service_args(assumptions={"Wald": {"": 100, "Nadelholz": 100, "Laubholz": 100}, "Weg": {"": 50},
                                     "Sumpf": {"": 70}}))
    with rasterio.open(cropped) as src:
        assert (src.height, src.width) == (100, 100)
    assert any("no outer row or column is fully forbidden" in line for line in log)


# ======================================================== D: zero is a value, labels tell the truth
def test_fill_value_zero_reaches_georasterizer_as_zero(app, state, inputs, no_tiles, monkeypatch):
    calls = _spy(monkeypatch, GeoRasterizer, "rasterize")
    _build(app, inputs, fill_value=0, geom_buffer=0)
    kwargs = calls[0][1]
    assert kwargs["fill_value"] == 0 and kwargs["geometry_buffer_m"] == 0


def test_ignore_max_cost_says_what_the_library_does(layout):
    """``ignore_max_cost=True`` makes maximum-cost cells barriers (and snaps points off them)."""
    checkbox = _by_id(layout)[ids.IGNORE_MAX_COST]
    assert "barrier" in checkbox.label.lower() and "snap" not in checkbox.label.lower()
    help_text = _help_of(layout, ids.IGNORE_MAX_COST)
    assert "65535" in help_text and "never be crossed" in help_text and "Off:" in help_text
    assert checkbox.value is True            # the library's default (True) is preselected


def test_the_geometry_buffer_help_names_the_gui_default(layout):
    help_text = _help_of(layout, ids.GEOM_BUFFER)
    assert "GUI starts at 1 m" in help_text and "library's own default is 0 m" in help_text
    assert _by_id(layout)[ids.GEOM_BUFFER].value == 1


# ======================================================== E: confirm before destroying
def _gate(app, button, *states):
    return invoke(app, (f"{_go_of(button)}.data", button), 1, *states, triggered=[f"{button}.n_clicks"])


def _go_of(button):
    return next(g.go for g in confirm_cb.GATES if g.button == button)


def _asked(resp, go):
    """The gate opened the modal (and did not fire the action)."""
    assert resp[ids.CONFIRM_MODAL]["is_open"] is True and go not in resp
    return resp


def _went_ahead(resp, go):
    """The gate fired the action at once: no modal."""
    assert resp[go]["data"] and ids.CONFIRM_MODAL not in resp
    return resp


def _confirm(app, pending):
    # (the heavy-run modal's ids contain ours as a substring: key on the pending store, which only we write)
    return invoke(app, (f"{ids.CONFIRM_PENDING}.data", ids.CONFIRM_OK), 1, pending)


def _cancel(app):
    return invoke(app, (f"{ids.CONFIRM_PENDING}.data", ids.CONFIRM_CANCEL), 1)


def test_there_is_one_modal_and_no_browser_dialog(layout):
    modals = [c for c, _ in _walk(layout) if isinstance(c, dbc.Modal)]
    assert {m.id for m in modals} == {ids.ROUTING_CONFIRM_MODAL, ids.CONFIRM_MODAL}
    assert not [c for c, _ in _walk(layout) if isinstance(c, dcc.ConfirmDialog)]
    pending = _by_id(layout)[ids.CONFIRM_PENDING]
    assert isinstance(pending, dcc.Store) and pending.data is None
    for source in Path(confirm_cb.__file__).parents[1].rglob("*"):
        if source.suffix in {".py", ".js"}:
            text = source.read_text(encoding="utf-8", errors="ignore")
            assert "ConfirmDialog(" not in text and "window.confirm(" not in text, source
            assert "alert(" not in text or source.suffix == ".py", source


def test_every_destructive_button_has_a_gate_and_its_action_listens_on_the_go_store(app):
    assert len(confirm_cb.GATES) == len(ids.GO_STORES) == 11
    inputs_of = {}
    for spec in app._callback_list:                                           # noqa: SLF001
        for item in spec["inputs"]:
            inputs_of.setdefault(item["id"], []).append(item["property"])
    for gate in confirm_cb.GATES:
        assert inputs_of[gate.button].count("n_clicks") == 1, gate.action     # the gate, and only the gate
        assert inputs_of[gate.go] == ["data"], gate.action                   # the action behind it


def test_new_project_asks_only_when_work_would_be_lost(app, state):
    _went_ahead(_gate(app, ids.PROJECT_NEW_BTN), ids.GO_NEW_PROJECT)          # an empty project loses nothing
    _add_vector(state, "landuse", _landuse_gdf())
    assert state.dirty
    resp = _asked(_gate(app, ids.PROJECT_NEW_BTN), ids.GO_NEW_PROJECT)
    assert resp[ids.CONFIRM_PENDING]["data"] == {"action": "new_project"}
    assert "1 vector" in json.dumps(resp[ids.CONFIRM_BODY]["children"])
    assert resp[ids.CONFIRM_OK]["children"] == "Discard and start new"
    state.dirty = False                                                       # saved: nothing to lose
    _went_ahead(_gate(app, ids.PROJECT_NEW_BTN), ids.GO_NEW_PROJECT)


def test_an_emptied_project_loses_nothing(app, state):
    layer = _add_vector(state, "landuse", _landuse_gdf())
    state.remove_layer(layer.id)
    assert state.dirty and not state.layers
    _went_ahead(_gate(app, ids.PROJECT_NEW_BTN), ids.GO_NEW_PROJECT)


def test_new_project_cancel_changes_nothing_and_confirm_clears(app, state):
    _add_vector(state, "landuse", _landuse_gdf())
    _asked(_gate(app, ids.PROJECT_NEW_BTN), ids.GO_NEW_PROJECT)
    resp = _cancel(app)
    assert resp[ids.CONFIRM_MODAL]["is_open"] is False and resp[ids.CONFIRM_PENDING]["data"] is None
    assert len(state.layers) == 1 and state.dirty                             # untouched
    _asked(_gate(app, ids.PROJECT_NEW_BTN), ids.GO_NEW_PROJECT)
    resp = _confirm(app, {"action": "new_project"})
    assert resp[ids.CONFIRM_MODAL]["is_open"] is False and resp[ids.GO_NEW_PROJECT]["data"]
    assert len(state.layers) == 1                                             # the gate alone clears nothing
    invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), resp[ids.GO_NEW_PROJECT]["data"], [])
    assert state.layers == {}


def test_open_project_asks_only_with_unsaved_changes_and_a_path(app, state, tmp_path):
    path = project_io.save_project(state, tmp_path / "p")
    _went_ahead(_gate(app, ids.PROJECT_OPEN_BTN, path), ids.GO_OPEN_PROJECT)
    _add_vector(state, "landuse", _landuse_gdf())
    resp = _asked(_gate(app, ids.PROJECT_OPEN_BTN, path), ids.GO_OPEN_PROJECT)
    assert "p" in json.dumps(resp[ids.CONFIRM_BODY]["children"]) and resp[ids.CONFIRM_PENDING]["data"] == {
        "action": "open_project"}
    _went_ahead(_gate(app, ids.PROJECT_OPEN_BTN, ""), ids.GO_OPEN_PROJECT)    # no path: the action names it


def test_open_project_cancel_keeps_the_open_project_and_confirm_replaces_it(app, state, tmp_path):
    path = project_io.save_project(state, tmp_path / "p")                     # an empty project on disk
    _add_vector(state, "landuse", _landuse_gdf())
    _asked(_gate(app, ids.PROJECT_OPEN_BTN, path), ids.GO_OPEN_PROJECT)
    _cancel(app)
    assert len(state.layers) == 1
    _asked(_gate(app, ids.PROJECT_OPEN_BTN, path), ids.GO_OPEN_PROJECT)
    go = _confirm(app, {"action": "open_project"})[ids.GO_OPEN_PROJECT]["data"]
    invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), go, path, [])
    assert state.layers == {}


def test_removing_layers_names_them_and_asks(app, state):
    keep = _add_vector(state, "Keep", _landuse_gdf())
    gone = _add_vector(state, "Gone", _landuse_gdf())
    resp = _asked(_gate(app, ids.LAYER_REMOVE_BTN, [{"id": gone.id}]), ids.GO_REMOVE_LAYERS)
    body = json.dumps(resp[ids.CONFIRM_BODY]["children"])
    assert "'Gone'" in body and "'Keep'" not in body and "1 layer will be removed" in body
    assert resp[ids.CONFIRM_OK]["children"] == "Remove 1 layer"
    _cancel(app)
    assert {keep.id, gone.id} <= set(state.layers)
    go = _confirm(app, {"action": "remove_layers"})[ids.GO_REMOVE_LAYERS]["data"]
    invoke(app, ("layers-view.data", ids.GO_REMOVE_LAYERS), go, [{"id": gone.id}])
    assert gone.id not in state.layers and keep.id in state.layers


def test_removing_with_nothing_selected_asks_nothing_and_does_nothing(app, state):
    _add_vector(state, "Keep", _landuse_gdf())
    for selected in ([], None):
        with pytest.raises(PreventUpdate):
            _gate(app, ids.LAYER_REMOVE_BTN, selected)
    assert len(state.layers) == 1


def test_deleting_a_group_asks_with_the_route_count(app, state):
    first = _route(state, "Route 1", group="Feeders")
    _route(state, "Route 2", group="Feeders")
    resp = _asked(_gate(app, ids.GROUP_DELETE_BTN, "Feeders"), ids.GO_DELETE_GROUP)
    assert "2 routes" in json.dumps(resp[ids.CONFIRM_BODY]["children"])
    _cancel(app)
    assert state.get(first.id) is not None and len(state.layers_of_kind("route")) == 2
    go = _confirm(app, {"action": "delete_group"})[ids.GO_DELETE_GROUP]["data"]
    invoke(app, ("layers-view.data", ids.GO_DELETE_GROUP), go, "Feeders", [])
    assert state.layers_of_kind("route") == []


def test_deleting_no_group_goes_to_the_actions_own_notice(app, state):
    _went_ahead(_gate(app, ids.GROUP_DELETE_BTN, None), ids.GO_DELETE_GROUP)
    _went_ahead(_gate(app, ids.GROUP_DELETE_BTN, "Nothing here"), ids.GO_DELETE_GROUP)


def test_discarding_a_route_asks_and_cancel_keeps_it(app, state):
    route = _route(state, "Route 7")
    state.active_route_id = route.id
    resp = _asked(_gate(app, ids.DELETE_VARIANT_BTN), ids.GO_DELETE_VARIANT)
    assert "'Route 7'" in json.dumps(resp[ids.CONFIRM_BODY]["children"])
    _cancel(app)
    assert state.get(route.id) is not None
    go = _confirm(app, {"action": "delete_variant"})[ids.GO_DELETE_VARIANT]["data"]
    invoke(app, ("layers-view.data", ids.GO_DELETE_VARIANT), go)
    assert state.get(route.id) is None


def test_discarding_without_an_active_route_does_nothing(app, state):
    with pytest.raises(PreventUpdate):
        _gate(app, ids.DELETE_VARIANT_BTN)


def _file(tmp_path, name="existing.txt"):
    path = tmp_path / name
    path.write_text("old", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("button,states_for", [
    (ids.PROJECT_SAVE_BTN, lambda tmp: (str(tmp / "proj"),)),
    (ids.RASTERIZE_BTN, lambda tmp: (_file(tmp, "r.tif"),)),
    (ids.COST_EXPORT_BTN, lambda tmp: (_file(tmp, "c.csv"),)),
    (ids.OHL_PROFILE_SAVE_BTN, lambda tmp: (_file(tmp, "p.yaml"),))])
def test_overwriting_an_existing_file_asks_and_a_new_name_does_not(app, state, tmp_path, button, states_for):
    go = _go_of(button)
    existing = states_for(tmp_path)
    if button == ids.PROJECT_SAVE_BTN:
        (tmp_path / "proj").mkdir()
        (tmp_path / "proj" / "project.json").write_text("{}", encoding="utf-8")
    resp = _asked(_gate(app, button, *existing), go)
    assert "already exists" in json.dumps(resp[ids.CONFIRM_BODY]["children"])
    assert resp[ids.CONFIRM_OK]["children"] == "Replace it"
    new_name = str(tmp_path / "brand-new" / "file.out")
    _went_ahead(_gate(app, button, new_name), go)                    # a new name loses nothing
    _went_ahead(_gate(app, button, ""), go)                          # blank: the action names the problem
    _went_ahead(_gate(app, button, None), go)


def test_exporting_a_route_over_an_existing_file_asks_cancel_leaves_the_file_confirm_replaces_it(app, state,
                                                                                                tmp_path):
    route = _route(state)
    state.active_route_id = route.id
    target = tmp_path / "route.geojson"
    target.write_text("old", encoding="utf-8")
    resp = _asked(_gate(app, ids.EXPORT_ROUTE_BTN, str(target)), ids.GO_EXPORT_ROUTE)
    assert "route.geojson" in json.dumps(resp[ids.CONFIRM_BODY]["children"])
    _cancel(app)
    assert target.read_text(encoding="utf-8") == "old"
    go = _confirm(app, {"action": "export_route"})[ids.GO_EXPORT_ROUTE]["data"]
    invoke(app, (f"{ids.EXPORT_STATUS}.children", ids.GO_EXPORT_ROUTE), go, str(target), [])
    assert target.read_text(encoding="utf-8") != "old"
    _went_ahead(_gate(app, ids.EXPORT_ROUTE_BTN, str(tmp_path / "other.geojson")), ids.GO_EXPORT_ROUTE)


def test_exporting_without_a_route_goes_to_the_actions_own_notice(app, state, tmp_path):
    _went_ahead(_gate(app, ids.EXPORT_ROUTE_BTN, _file(tmp_path)), ids.GO_EXPORT_ROUTE)


def test_saving_a_project_over_an_existing_one_cancel_and_confirm(app, state, tmp_path):
    target = tmp_path / "session"
    project_io.save_project(state, target)
    manifest = target / "project.json"
    manifest.write_text('{"marker": 1}', encoding="utf-8")
    _asked(_gate(app, ids.PROJECT_SAVE_BTN, str(target)), ids.GO_SAVE_PROJECT)
    _cancel(app)
    assert manifest.read_text(encoding="utf-8") == '{"marker": 1}'
    go = _confirm(app, {"action": "save_project"})[ids.GO_SAVE_PROJECT]["data"]
    invoke(app, (f"{ids.PROJECT_STATUS}.children", ids.GO_SAVE_PROJECT), go, str(target),
           ["raster", "vector", "cost_table"], [], {}, [], [])
    assert "marker" not in manifest.read_text(encoding="utf-8")


def test_the_confirm_button_of_a_forgotten_action_does_nothing(app):
    with pytest.raises(PreventUpdate):
        _confirm(app, None)
    with pytest.raises(PreventUpdate):
        _confirm(app, {"action": "launch_missiles"})


def test_every_action_callback_ignores_an_empty_go_store(app):
    """The stores start empty: nothing may run before the first answer."""
    for gate in confirm_cb.GATES:
        keys = [k for k, entry in app.callback_map.items()
                if any(i["id"] == gate.go for i in entry.get("inputs", []))
                and not any(i["id"] == gate.button for i in entry.get("inputs", []))]
        assert len(keys) == 1, gate.action
        assert app.callback_map[keys[0]].get("prevent_initial_call", True) is not False


# ======================================================== I: the whole-layer checkbox
def test_the_whole_layer_checkbox_sits_next_to_the_load_button_and_warns_about_the_size(layout):
    box_ = _by_id(layout)[ids.WFS_WHOLE_LAYER]
    assert isinstance(box_, dbc.Checkbox) and box_.value is False and "whole layer" in box_.label
    row = _parent_of(layout, ids.WFS_LOAD_BTN)
    assert ids.WFS_WHOLE_LAYER in {getattr(c, "id", None) for c, _ in _walk(row)}
    help_text = _help_of(layout, ids.WFS_WHOLE_LAYER)
    assert "EVERY feature" in help_text and "gigabytes" in help_text


def test_no_second_click_behaviour_is_left_in_the_code():
    from pyorps.gui.callbacks import data as data_cb

    source = inspect.getsource(data_cb)
    assert "_confirm_full_extent" not in source and "_full_extent_ok" not in source
    assert not hasattr(data_cb, "_confirm_full_extent")


def _wfs(app, clip, whole, url="https://x.test/wfs", layer="lyr"):
    return invoke(app, ("dataset-list.children", ids.WFS_LOAD_BTN), 1, url, layer, clip, [], whole,
                  triggered=[f"{ids.WFS_LOAD_BTN}.n_clicks"])


def test_the_checkbox_loads_every_feature_without_a_mask_even_with_a_study_area(app, state, monkeypatch):
    from pyorps.gui.callbacks import data as data_cb

    masks = []
    monkeypatch.setattr(data_cb.data_io, "load_wfs_vector",
                        lambda url, layer, **k: masks.append(k.get("mask")) or _landuse_gdf())
    state.study_area = {"type": "Polygon", "coordinates": [[[8.0, 50.0], [8.1, 50.0], [8.1, 50.1], [8.0, 50.0]]]}
    resp = _wfs(app, True, True)
    assert masks == [None] and _notices(resp)[-1]["title"].startswith("Loaded")
    source = state.layers_of_kind("vector")[0].meta["source"]
    assert source["whole_layer"] is True and source["clip"] is False


def test_without_the_checkbox_a_missing_study_area_is_a_notice_that_points_at_it(app, state, monkeypatch):
    from pyorps.gui.callbacks import data as data_cb

    monkeypatch.setattr(data_cb.data_io, "load_wfs_vector", lambda *a, **k: pytest.fail("must not load"))
    notice = _notices(_wfs(app, True, False))[-1]
    assert notice["title"] == "Draw a study area first" and notice["focus_control"] == ids.WFS_WHOLE_LAYER
    notice = _notices(_wfs(app, False, False))[-1]                    # clip off and no checkbox: still refused
    assert notice["title"] == "This would download the whole layer"
    assert "Load the whole layer" in notice["fix"] and notice["focus_control"] == ids.WFS_WHOLE_LAYER
    assert state.layers_of_kind("vector") == []


def test_refreshing_keeps_a_whole_layer_whole_and_a_clipped_layer_needs_its_area(app, state, monkeypatch):
    from pyorps.gui.callbacks import data as data_cb

    calls = []
    monkeypatch.setattr(data_cb.data_io, "load_wfs_vector",
                        lambda url, layer, **k: calls.append(k.get("mask")) or _landuse_gdf())

    def refresh(layer):
        return invoke(app, ("dataset-list.children", "dataset-refresh"), [1], [],
                      triggered=[{"id": {"type": ids.TYPE_DATASET_REFRESH, "index": layer.id},
                                  "property": "n_clicks", "value": 1}])

    whole = _add_vector(state, "whole", _landuse_gdf(), meta={"source": {
        "kind": "wfs", "url": "https://x.test/wfs", "layer": "w", "clip": False, "whole_layer": True}})
    assert _notices(refresh(whole))[-1]["title"].startswith("Refreshed") and calls == [None]
    clipped = _add_vector(state, "clipped", _landuse_gdf(), meta={"source": {
        "kind": "wfs", "url": "https://x.test/wfs", "layer": "c", "clip": True, "whole_layer": False}})
    notice = _notices(refresh(clipped))[-1]
    assert notice["title"] == "This would download the whole layer" and calls == [None]   # nothing new was loaded


# ======================================================== K: Open and New refresh the controls
ROUTING_TARGETS = [(ids.GRAPH_API, "value"), (ids.ALGORITHM, "options"), (ids.ALGORITHM, "value"),
                   (ids.NEIGHBORHOOD, "value"), (ids.SEARCH_BUFFER, "value"), (ids.FULL_WINDOW, "value"),
                   (ids.IGNORE_MAX_COST, "value"), (ids.PAIRWISE, "value"), (ids.SIMPLIFY, "value"),
                   (ids.SIMPLIFY_TOL, "value"), (ids.DELTA, "value"), (ids.NUM_THREADS, "value"),
                   (ids.MARGIN, "value")]
RASTER_TARGETS = [(ids.RES_M, "value"), (ids.FILL_VALUE, "value"), (ids.RASTER_DTYPE, "value"),
                  (ids.GEOM_BUFFER, "value"), (ids.WIDEN_THIN, "value"), (ids.ALL_TOUCHED, "value"),
                  (ids.ON_THIN, "value"), (ids.CROP_BORDER, "value")]


@pytest.mark.parametrize("target", ROUTING_TARGETS + RASTER_TARGETS + [(ids.WFS_WHOLE_LAYER, "value")])
def test_new_and_open_write_every_control_of_the_routing_and_raster_settings(app, target):
    assert target in io_cb.RESET_TARGETS
    for button in (ids.GO_NEW_PROJECT, ids.GO_OPEN_PROJECT):
        entry = next(e for e in app.callback_map.values()
                     if any(i["id"] == button for i in e.get("inputs", [])))
        written = [(o.component_id, o.component_property.split("@")[0]) for o in entry["output"]]
        assert target in written, button


def _non_default_state(state):
    state.routing_settings = {
        "graph_api": "networkx", "algorithm": "astar", "neighborhood_k": 3, "search_buffer_m": 250,
        "full_window": True, "ignore_max_cost": False, "pairwise": True, "simplify": True, "simplify_tol": 2.5,
        "delta": 50, "num_threads": 2, "margin": 1.5}
    state.raster_settings = {
        "resolution_m": 2.5, "fill_value": 0, "dtype": "uint32", "geometry_buffer_m": 0, "widen_thin_forbidden": True,
        "all_touched": True, "on_thin_features": "raise", "crop_border": True}


def test_open_refreshes_the_routing_and_raster_controls_from_the_opened_project(app, state, tmp_path):
    _non_default_state(state)
    path = project_io.save_project(state, tmp_path / "p")
    state.clear()                                               # the page still shows the old values
    resp = invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1, path, [])
    got = {t: resp[t[0]][t[1]] for t in io_cb.RESET_TARGETS}
    assert got[(ids.GRAPH_API, "value")] == "networkx" and got[(ids.ALGORITHM, "value")] == "astar"
    assert [o["value"] for o in got[(ids.ALGORITHM, "options")]] == routing.backend_algorithms("networkx")
    assert got[(ids.NEIGHBORHOOD, "value")] == 3 and got[(ids.SEARCH_BUFFER, "value")] == 250
    assert got[(ids.FULL_WINDOW, "value")] is True and got[(ids.IGNORE_MAX_COST, "value")] is False
    assert got[(ids.PAIRWISE, "value")] is True and got[(ids.SIMPLIFY, "value")] is True
    assert got[(ids.SIMPLIFY_TOL, "value")] == 2.5 and got[(ids.DELTA, "value")] == 50
    assert got[(ids.NUM_THREADS, "value")] == 2 and got[(ids.MARGIN, "value")] == 1.5
    assert got[(ids.RES_M, "value")] == 2.5 and got[(ids.FILL_VALUE, "value")] == 0
    assert got[(ids.RASTER_DTYPE, "value")] == "uint32" and got[(ids.GEOM_BUFFER, "value")] == 0
    assert got[(ids.WIDEN_THIN, "value")] is True and got[(ids.ALL_TOUCHED, "value")] is True
    assert got[(ids.ON_THIN, "value")] == "raise" and got[(ids.CROP_BORDER, "value")] is True
    assert got[(ids.WFS_WHOLE_LAYER, "value")] is False


def test_open_of_an_older_project_puts_the_controls_back_to_their_defaults(app, state, tmp_path):
    state.routing_settings = {}
    manifest = tmp_path / "old" / "project.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"version": 1, "layers": []}), encoding="utf-8")       # no settings at all
    _non_default_state(state)                                  # what the page showed before
    resp = invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1, str(manifest), [])
    got = {t: resp[t[0]][t[1]] for t in io_cb.RESET_TARGETS}
    assert got[(ids.GRAPH_API, "value")] == routing.DEFAULT_BACKEND
    assert got[(ids.ALGORITHM, "value")] == routing.default_algorithm(routing.DEFAULT_BACKEND)
    assert got[(ids.NEIGHBORHOOD, "value")] == routing.ROUTING_SETTING_DEFAULTS["neighborhood_k"]
    assert got[(ids.SEARCH_BUFFER, "value")] is None and got[(ids.MARGIN, "value")] is None
    assert got[(ids.RES_M, "value")] == 1.0 and got[(ids.FILL_VALUE, "value")] == 65535
    assert got[(ids.GEOM_BUFFER, "value")] == 1 and got[(ids.ON_THIN, "value")] == "warn"
    assert got[(ids.WIDEN_THIN, "value")] is False and got[(ids.CROP_BORDER, "value")] is False


def test_new_project_puts_the_controls_back_to_their_defaults(app, state):
    _non_default_state(state)
    resp = invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1, [])
    got = {t: resp[t[0]][t[1]] for t in io_cb.RESET_TARGETS}
    assert got[(ids.GRAPH_API, "value")] == routing.DEFAULT_BACKEND and got[(ids.FULL_WINDOW, "value")] is False
    assert got[(ids.IGNORE_MAX_COST, "value")] is True and got[(ids.DELTA, "value")] == 100
    assert got[(ids.ON_THIN, "value")] == "warn" and got[(ids.ALL_TOUCHED, "value")] is False
    assert state.routing_settings == {} and state.raster_settings == {}


def test_the_opened_algorithm_survives_the_backend_change_once(app, state, tmp_path):
    """Writing the backend makes the algorithm list callback run: it must keep the opened algorithm, and only once."""
    _non_default_state(state)
    path = project_io.save_project(state, tmp_path / "p")
    invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1, path, [])
    assert state.routing_restore_backend == "networkx"
    kept = invoke(app, (f"{ids.ALGORITHM}.options", ids.GRAPH_API), "networkx", "astar",
                  triggered=[f"{ids.GRAPH_API}.value"])
    assert kept[ids.ALGORITHM]["value"] == "astar" and state.routing_restore_backend is None
    # a later pick by the user resets the algorithm as before
    reset = invoke(app, (f"{ids.ALGORITHM}.options", ids.GRAPH_API), "networkx", "astar",
                   triggered=[f"{ids.GRAPH_API}.value"])
    assert reset[ids.ALGORITHM]["value"] == routing.default_algorithm("networkx")


def test_a_stale_restore_flag_does_not_hold_a_different_backend(app, state):
    state.routing_restore_backend = "networkx"
    resp = invoke(app, (f"{ids.ALGORITHM}.options", ids.GRAPH_API), "cython", "astar",
                  triggered=[f"{ids.GRAPH_API}.value"])
    assert resp[ids.ALGORITHM]["value"] == routing.default_algorithm("cython")
    assert state.routing_restore_backend is None


# ------------------------------------------------ settings round trip and old projects
def test_the_raster_controls_are_mirrored_into_the_state_and_saved_with_the_project(app, state, tmp_path):
    resp = invoke(app, (f"{ids.RASTER_SETTINGS}.data",), 2.0, 0, "float32", 0.5, True, True, "ignore", True)
    expected = {"resolution_m": 2.0, "fill_value": 0, "dtype": "float32", "geometry_buffer_m": 0.5,
                "widen_thin_forbidden": True, "all_touched": True, "on_thin_features": "ignore",
                "crop_border": True}
    assert resp[ids.RASTER_SETTINGS]["data"] == expected and state.raster_settings == expected
    path = project_io.save_project(state, tmp_path / "p")
    assert json.loads(Path(path).read_text(encoding="utf-8"))["raster_settings"] == expected
    state.clear()
    assert state.raster_settings == {}
    project_io.load_project(state, path)
    assert state.raster_settings == expected
    shown = _by_id(build_layout(state))                         # a page built again shows the saved values
    assert shown[ids.RES_M].value == 2.0 and shown[ids.FILL_VALUE].value == 0
    assert shown[ids.RASTER_DTYPE].value == "float32" and shown[ids.GEOM_BUFFER].value == 0.5
    assert shown[ids.WIDEN_THIN].value is True and shown[ids.ALL_TOUCHED].value is True
    assert shown[ids.ON_THIN].value == "ignore" and shown[ids.CROP_BORDER].value is True


def test_blank_raster_fields_are_saved_as_their_defaults(app, state):
    resp = invoke(app, (f"{ids.RASTER_SETTINGS}.data",), None, None, None, None, None, None, None, None)
    assert resp[ids.RASTER_SETTINGS]["data"] == rasterize.RASTER_SETTING_DEFAULTS


def test_an_old_project_without_raster_settings_loads_with_defaults(state, tmp_path):
    manifest = tmp_path / "old" / "project.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"version": 1, "layers": []}), encoding="utf-8")
    project_io.load_project(state, str(manifest))
    assert state.raster_settings == {}
    shown = _by_id(build_layout(state))
    assert shown[ids.RES_M].value == 1.0 and shown[ids.GEOM_BUFFER].value == 1
    assert shown[ids.ON_THIN].value == "warn" and shown[ids.WIDEN_THIN].value is False


@pytest.mark.parametrize("raw,key,expected", [
    ({"on_thin_features": "shout"}, "on_thin_features", "warn"),
    ({"dtype": "int8"}, "dtype", "uint16"),
    ({"crop_border": "yes"}, "crop_border", False),
    ({"resolution_m": float("nan")}, "resolution_m", 1.0),
    ({"fill_value": "0"}, "fill_value", 65535),
    ({"fill_value": 0}, "fill_value", 0)])
def test_unusable_saved_raster_settings_fall_back_to_the_defaults(raw, key, expected):
    assert rasterize.normalise_raster_settings(raw)[key] == expected
    assert rasterize.normalise_raster_settings(None) == {} == rasterize.normalise_raster_settings({})


# ======================================================== C: modifier grid columns
def _modifier_row(**over):
    row = {"dataset": "mods", "column": "", "operator": "all", "value": "", "mode": "multiply", "factor": "2",
           "buffer_m": 0}
    row.update(over)
    return row


@pytest.fixture()
def mods(state):
    gdf = gpd.GeoDataFrame({"zone": [1, 2, 2], "tag": ["a", "b", "c"]},
                           geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)], crs=CRS)
    return _add_vector(state, "mods", gdf)


def test_the_modifier_grid_has_the_new_columns_and_new_rows_carry_their_defaults(layout, app):
    grid = _by_id(layout)[ids.MODIFIER_GRID]
    fields = [c["field"] for c in grid.columnDefs]
    assert {"keep_forbidden", "forbidden_value", "zone_column", "forbidden_zone"} <= set(fields)
    keep = next(c for c in grid.columnDefs if c["field"] == "keep_forbidden")
    assert keep["cellRenderer"] == "agCheckboxCellRenderer"           # a switch, not text
    resp = invoke(app, (f"{ids.MODIFIER_GRID}.rowData", ids.MODIFIER_ADD_BTN), 1, [],
                  triggered=[f"{ids.MODIFIER_ADD_BTN}.n_clicks"])
    row = resp[ids.MODIFIER_GRID]["rowData"][0]
    assert (row["keep_forbidden"], row["forbidden_value"], row["zone_column"], row["forbidden_zone"]) == (
        True, 65535, "", "")


def test_an_old_modifier_row_means_what_it_always_meant(state, mods):
    (spec,) = raster_cb._build_modifiers(state, [_modifier_row()])            # none of the four columns
    assert spec.ignore_value == 65535 and spec.forbidden_value == 65535
    assert spec.zone_field is None and spec.forbidden_zone is None


def test_the_keep_forbidden_switch_is_the_ignore_value(state, mods):
    (on,) = raster_cb._build_modifiers(state, [_modifier_row(keep_forbidden=True)])
    (off,) = raster_cb._build_modifiers(state, [_modifier_row(keep_forbidden=False)])
    assert on.ignore_value == 65535 and off.ignore_value is None


def test_the_modifier_columns_reach_modify_raster_from_dataset(app, state, inputs, mods, no_tiles, monkeypatch):
    calls = _spy(monkeypatch, GeoRasterizer, "modify_raster_from_dataset")
    row = _modifier_row(keep_forbidden=False, forbidden_value=60000, zone_column="zone", forbidden_zone="2")
    _build(app, inputs, modifiers=[row])
    kwargs = calls[0][1]
    assert kwargs["ignore_value"] is None and kwargs["forbidden_value"] == 60000
    assert kwargs["zone_field"] == "zone" and kwargs["forbidden_zone"] == 2.0     # numeric column: a number


def test_the_default_modifier_columns_reach_the_library_as_its_defaults(app, state, inputs, mods, no_tiles,
                                                                       monkeypatch):
    parameters = inspect.signature(GeoRasterizer.modify_raster_from_dataset).parameters   # before the spy wraps it
    calls = _spy(monkeypatch, GeoRasterizer, "modify_raster_from_dataset")
    _build(app, inputs, modifiers=[_modifier_row()])
    kwargs = calls[0][1]
    assert kwargs["ignore_value"] == parameters["ignore_value"].default
    assert kwargs["forbidden_value"] == parameters["forbidden_value"].default
    assert kwargs["zone_field"] is parameters["zone_field"].default is None
    assert kwargs["forbidden_zone"] is parameters["forbidden_zone"].default is None


def test_a_text_zone_value_is_compared_as_text(state, mods):
    (spec,) = raster_cb._build_modifiers(state, [_modifier_row(zone_column="tag", forbidden_zone=" b ")])
    assert spec.zone_field == "tag" and spec.forbidden_zone == "b"


@pytest.mark.parametrize("over,fragment", [
    ({"forbidden_value": "abc"}, "must be a number"),
    ({"forbidden_value": -1}, "whole number from 0 up"),
    ({"forbidden_value": 1.5}, "whole number from 0 up"),
    ({"forbidden_value": 70000}, "does not fit the raster type uint16"),
    ({"zone_column": "zone"}, "needs both a zone column and a zone value"),
    ({"forbidden_zone": "2"}, "needs both a zone column and a zone value"),
    ({"zone_column": "nope", "forbidden_zone": "2"}, "not an attribute"),
    ({"zone_column": "zone", "forbidden_zone": "two"}, "is numeric"),
    ({"zone_column": "zone", "forbidden_zone": "0"}, "not supported")])
def test_an_unusable_modifier_column_is_a_clear_error_not_a_crash(state, mods, over, fragment):
    with pytest.raises(ValueError, match=fragment):
        raster_cb._build_modifiers(state, [_modifier_row(**over)], "uint16")


def test_an_unusable_modifier_column_becomes_a_notice_and_builds_nothing(app, state, inputs, mods, fake_build):
    resp = _build(app, inputs, modifiers=[_modifier_row(forbidden_value=-5)])
    assert fake_build == []
    assert _notices(resp)[-1]["title"] == "Modifier configuration invalid"


def test_a_large_forbid_as_fits_a_wider_raster_type(state, mods):
    (spec,) = raster_cb._build_modifiers(state, [_modifier_row(forbidden_value=70000)], "uint32")
    assert spec.forbidden_value == 70000


def test_the_modifier_columns_change_the_cache_key(tmp_path):
    base = _service_args()
    spec = rasterize.ModifierSpec(gdf=_landuse_gdf().iloc[:1], factor=2.0)
    first, _ = rasterize.build_cost_raster(work_dir=tmp_path, modifiers=[spec], **base)
    for change in ({"ignore_value": None}, {"forbidden_value": 60000}):
        other, _ = rasterize.build_cost_raster(
            work_dir=tmp_path, modifiers=[rasterize.ModifierSpec(gdf=_landuse_gdf().iloc[:1], factor=2.0, **change)],
            **base)
        assert other != first, change


def test_modifier_columns_round_trip_with_the_project_and_old_rows_load(app, state, tmp_path):
    new = _modifier_row(keep_forbidden=False, forbidden_value=60000, zone_column="zone", forbidden_zone="2")
    old = _modifier_row()
    path = project_io.save_project(state, tmp_path / "p", cost_table={"feature_keys": [], "rows": [],
                                                                      "modifiers": [new, old]})
    resp = invoke(app, ("project-status.children", ids.GO_OPEN_PROJECT), 1, path, [])
    assert resp[ids.MODIFIER_GRID]["rowData"] == [new, old]


def test_the_combination_table_mirrors_the_zone_rule():
    from pyorps.gui.services import combinations

    gdf = gpd.GeoDataFrame({"zone": [1, 2]}, geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=CRS)
    spec = rasterize.ModifierSpec(gdf=gdf, factor=3.0, zone_field="zone", forbidden_zone=2, forbidden_value=60000)
    parts = list(combinations._expand_zones([spec]))                          # noqa: SLF001
    assert [(len(p.gdf), p.mode, p.factor) for p in parts] == [(1, "multiply", 3.0), (1, "override", 60000.0)]
    plain = rasterize.ModifierSpec(gdf=gdf, factor=3.0)
    assert list(combinations._expand_zones([plain])) == [plain]               # noqa: SLF001


# ======================================================== F: limits and units
def _label_texts(layout) -> dict[str, str]:
    texts = {}
    for component, _parents in _walk(layout):
        target = (getattr(component, "html_for", None) if isinstance(component, dbc.Label)
                  else getattr(component, "htmlFor", None) if isinstance(component, html.Label) else None)
        if target:
            texts[target] = _text(component).strip()
    return texts


LIMITS = [
    # control, minimum, maximum, step, unit text expected in the label
    (ids.RES_M, 0.1, None, 0.1, "(m)"),
    (ids.FILL_VALUE, 0, 65535, 1, "cost"),
    (ids.GEOM_BUFFER, 0, None, 0.5, "(m)"),
    (ids.PREPROC_BUF_A, 0, None, 0.5, "(m)"),
    (ids.PREPROC_BUF_B, 0, None, 0.5, "(m)"),
    (ids.PREPROC_BUF_L, 0, None, 0.5, "(m)"),
    (ids.MANUAL_COST, 0, None, 1, "cost"),
    (ids.GRAD_CLASSES, 2, 32, 1, "2 to 32"),
    (ids.NEIGHBORHOOD, 0, 127, 1, "k"),
    (ids.SEARCH_BUFFER, 0, None, 10, "(m)"),
    (ids.SIMPLIFY_TOL, 0, None, 0.1, "(m)"),
    (ids.DELTA, 0, None, 10, "cost"),
    (ids.NUM_THREADS, 0, None, 1, "auto"),
    (ids.MARGIN, 0, None, 0.1, "distance"),
    (ids.EDIT_SIMPLIFY_TOL, 0, None, 0.5, "(m)")]


from pyorps.gui.services import profile_form as _profile_form_service

# the number inputs of the constrained profile form: limits and unit come from the service's specs
LIMITS += [(ids.OHLF_IDS[key], spec.minimum, spec.maximum, 1 if spec.whole else "any",
            f"({spec.unit})" if spec.unit else spec.label)
           for key, spec in _profile_form_service.FIELD_SPECS.items() if spec.kind == "number"]


from pyorps.gui.services import corridor as _corridor_service

# the number inputs of the corridor card: limits and unit come from the service's specs
_COR_NUMBER_IDS = {"k_per_pair": ids.COR_K, "min_shared_length_m": ids.COR_MIN_SHARED,
                   "max_pair_cost": ids.COR_MAX_PAIR_COST}
LIMITS += [(_COR_NUMBER_IDS[key], spec.minimum, spec.maximum, 1 if spec.whole else "any",
            f"({spec.unit})" if spec.unit else spec.label)
           for key, spec in _corridor_service.FIELD_SPECS.items() if spec.kind == "number"]


from pyorps.gui.services import cost_field as _cost_field_service

# the number input of the cost-field card: its limit comes from the service (0 = automatic)
LIMITS += [(ids.CF_THREADS, 0, _cost_field_service.max_threads(), 1, "0 = automatic")]


from pyorps.gui.sections import objective as _objective_section
from pyorps.gui.services import objective as _objective_service

# the number inputs of the objective card: limits and unit come from the service's specs (unit "-" = none, so the
# label itself is what is checked); the slope range is at least one table step wide, so its minimum is that step
LIMITS += [(_objective_section.NUMBER_IDS[key],
            _objective_service.FIELD_SPECS["gradient.bin_width_pct"].minimum if key == "gradient.s_max_pct"
            else spec.minimum,
            spec.maximum, "any", f"({spec.unit})" if spec.unit and spec.unit != "-" else spec.label)
           for key, spec in _objective_service.FIELD_SPECS.items() if key in _objective_section.NUMBER_IDS]


@pytest.mark.parametrize("control,minimum,maximum,step,unit", LIMITS)
def test_numeric_inputs_state_their_limits_and_their_unit(layout, control, minimum, maximum, step, unit):
    component = _by_id(layout)[control]
    assert isinstance(component, dbc.Input) and getattr(component, "type", None) == "number"
    assert component.min == minimum and component.step == step
    assert getattr(component, "max", None) == maximum
    assert unit in _label_texts(layout)[control], _label_texts(layout)[control]


def test_every_number_input_of_the_four_tabs_has_a_minimum_and_a_unit_or_a_reason(layout):
    numbers = [c.id for c, _ in _walk(layout) if isinstance(c, dbc.Input) and getattr(c, "type", None) == "number"]
    assert set(numbers) <= {row[0] for row in LIMITS}, set(numbers) - {row[0] for row in LIMITS}


@pytest.mark.parametrize("dtype,expected", [("uint16", (0, 65535)), ("uint32", (0, 4294967295)),
                                            ("float32", (None, None)), (None, (0, 65535))])
def test_the_fill_value_range_follows_the_raster_type(app, dtype, expected):
    resp = invoke(app, (f"{ids.FILL_VALUE}.min",), dtype)
    assert (resp[ids.FILL_VALUE]["min"], resp[ids.FILL_VALUE]["max"]) == expected


@pytest.mark.parametrize("over,control,fragment", [
    ({"resolution": 0}, ids.RES_M, "Resolution (m)"),
    ({"resolution": -3}, ids.RES_M, "greater than 0"),
    ({"geom_buffer": -0.5}, ids.GEOM_BUFFER, "Geometry buffer (m)"),
    ({"fill_value": -1}, ids.FILL_VALUE, "Fill value"),
    ({"fill_value": 65536}, ids.FILL_VALUE, "does not fit the raster type uint16"),
    ({"preproc": "street_buffer"}, None, None)])
def test_an_out_of_range_value_is_a_field_notice_not_a_crash_or_a_clamp(app, state, inputs, fake_build, over,
                                                                        control, fragment):
    if control is None:                                          # a valid street preset builds
        _build(app, inputs, **over)
        assert len(fake_build) == 1
        return
    resp = _build(app, inputs, **over)
    assert fake_build == []                                      # not clamped, not run
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == control and notice["focus_id"] == ids.TAB_RASTER
    assert fragment in notice["title"] + notice["meaning"]


def test_a_negative_default_cost_is_a_field_notice_and_adds_no_polygon(app, state):
    resp = invoke(app, ("study-area-info.children", ids.DRAW_CONTROL),
                  {"features": [{"type": "Feature", "properties": {}, "geometry": {
                      "type": "Polygon", "coordinates": [[[8.6, 50.0], [8.7, 50.0], [8.7, 50.1], [8.6, 50.0]]]}}]},
                  "cost", "Zone", -5, "override", [], [], triggered=[f"{ids.DRAW_CONTROL}.geojson"])
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.MANUAL_COST and "Default cost" in notice["title"]
    assert state.layers_of_kind("vector") == []


# ======================================================== H: loading feedback
def test_project_status_and_cost_coverage_have_a_delayed_spinner(layout):
    wrapped = {}
    for component, _parents in _walk(layout):
        if isinstance(component, dcc.Loading):
            for inner, _ in _walk(component):
                if isinstance(getattr(inner, "id", None), str):
                    wrapped[inner.id] = component
    for status in (ids.PROJECT_STATUS, ids.COST_COVERAGE_INFO):
        assert wrapped[status].delay_show == 300, status


def test_nothing_a_poll_writes_gets_a_new_spinner(app, layout):
    """The interval poll only writes the routing status, which keeps its pre-existing wrapper; the two new
    spinners wrap components that no ``dcc.Interval`` callback writes."""
    polled = set()
    for spec in app._callback_list:                                           # noqa: SLF001
        if any(item["id"] == ids.ROUTING_POLL for item in spec["inputs"]):
            polled |= {out.split("@")[0].lstrip(".").split(".")[0] for out in str(spec["output"]).split("...")}
    assert ids.PROJECT_STATUS not in polled and ids.COST_COVERAGE_INFO not in polled


# ======================================================== G: labels for assistive technology
LABELLED_BY_FOR = (dbc.Input, dbc.Textarea, dbc.Select, dcc.Dropdown)
LABELLED_BY_OWN_LABEL = (dbc.Checkbox, dbc.Switch)
LABELLED_BY_OPTIONS = (dbc.RadioItems, dbc.Checklist)
LABELLED_BY_GROUP = (dcc.Slider,)


def _label_targets(layout) -> set[str]:
    """Ids that a visible ``<label for=...>`` points at."""
    targets = set()
    for component, _parents in _walk(layout):
        target = (getattr(component, "html_for", None) if isinstance(component, dbc.Label)
                  else getattr(component, "htmlFor", None) if isinstance(component, html.Label) else None)
        if target and _text(component).strip():
            targets.add(target)
    return targets


def _in_named_group(parents) -> bool:
    """Whether a ``role="group"`` ancestor is named by a caption (``aria-labelledby``)."""
    return any(getattr(p, "role", None) == "group" and getattr(p, "aria-labelledby", None) for p in parents)


def label_problems(layout) -> list[str]:
    """Every input without a label and every icon-only button without ``aria-label``."""
    targets = _label_targets(layout)
    captions = {getattr(c, "id", None) for c, _ in _walk(layout)}
    found = []
    for component, parents in _walk(layout):
        name = type(component).__name__
        cid = getattr(component, "id", None)
        if isinstance(component, LABELLED_BY_FOR):
            if cid is None or cid not in targets:
                found.append(f"{name} {cid!r}: no <label for>")
        elif isinstance(component, LABELLED_BY_OWN_LABEL):
            if not (getattr(component, "label", None) or cid in targets):
                found.append(f"{name} {cid!r}: no label")
        elif isinstance(component, LABELLED_BY_OPTIONS):
            if not component.options or not all(o.get("label") for o in component.options):
                found.append(f"{name} {cid!r}: an option without a label")
            if not _in_named_group(parents):
                found.append(f"{name} {cid!r}: the list has no named group")
        elif isinstance(component, LABELLED_BY_GROUP):
            if not _in_named_group(parents):
                found.append(f"{name} {cid!r}: no named group")
        elif isinstance(component, (html.Button, dbc.Button, html.Summary)):
            icon_only = not re.search(r"[A-Za-z]", _text(component))
            if icon_only and not getattr(component, "aria-label", None):
                found.append(f"{name} {cid!r}: icon-only without aria-label")
        if getattr(component, "role", None) == "group":
            named = getattr(component, "aria-labelledby", None)
            if named and named not in captions:
                found.append(f"group {cid!r}: names a caption {named!r} that does not exist")
    return found


def test_every_input_has_a_label_and_every_icon_button_a_name(layout):
    assert label_problems(layout) == []


def test_the_label_walk_finds_a_missing_label_and_a_nameless_icon_button():
    broken = html.Div([
        dbc.Input(id="a"), dbc.Label("B", html_for="b"), dbc.Input(id="b"),
        dbc.Button("📁", id="browse"), html.Button("📁", id="ok", **{"aria-label": "Browse"}),
        dcc.Slider(id="s", min=0, max=1)])
    problems = label_problems(broken)
    assert any("'a'" in p and "no <label for>" in p for p in problems)
    assert not any("'b'" in p for p in problems)
    assert any("'browse'" in p for p in problems)
    assert not any("'ok'" in p for p in problems)
    assert any("'s'" in p for p in problems)


def test_a_label_without_text_binds_nothing():
    broken = html.Div([dbc.Label("", html_for="a"), dbc.Input(id="a")])
    assert label_problems(broken)


def test_every_browse_button_names_what_it_picks(layout):
    browse = [c for c, _ in _walk(layout) if str(getattr(c, "id", "")).startswith("browse-")]
    assert len(browse) >= 10
    labels = [getattr(b, "aria-label") for b in browse]
    assert all(label.startswith("Browse for ") for label in labels) and len(set(labels)) == len(labels)
    assert all(b.title for b in browse)


def test_the_theme_toggle_and_the_basemap_summary_are_named(layout):
    toggle = _by_id(layout)[ids.THEME_TOGGLE]
    assert getattr(toggle, "aria-label", None) and toggle.title
    summary = next(c for c, _ in _walk(layout) if isinstance(c, html.Summary) and not _text(c).strip())
    assert getattr(summary, "aria-label", None)


def test_changing_status_regions_are_live_regions(layout):
    by_id = _by_id(layout)
    for region in (ids.ROUTING_STATUS, ids.PROJECT_STATUS, ids.NOTICE_STACK):
        assert getattr(by_id[region], "role", None) == "status", region
        assert getattr(by_id[region], "aria-live", None) == "polite", region

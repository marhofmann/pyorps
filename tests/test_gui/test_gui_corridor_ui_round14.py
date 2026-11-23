"""Round 14, P12: the "Corridor graph" card of the Routes tab.

Headless callback tests (the harness of ``conftest.py`` drives the callbacks as Dash would) plus layout checks. The
rasters are tiny GeoTIFFs whose open cells form a one-cell-wide network (the rest is forbidden, 65535): with the
4-neighbour step table (``r0``, ``neighborhood_k`` 0) the route between two points is unique. Components are
addressed as "<id>.<prop>": several ids contain each other. Every job thread is joined with a timeout.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from conftest import invoke, poll_job
from dash import html
from dash.exceptions import PreventUpdate

from pyorps.core.exceptions import InvalidInputError
from pyorps.gui import ids
from pyorps.gui.app import build_app
from pyorps.gui.callbacks import corridor as callbacks
from pyorps.gui.layout import build_layout
from pyorps.gui.sections import corridor as section
from pyorps.gui.services import corridor as service
from pyorps.gui.services import dialogs, jobs, project_io
from pyorps.gui.state import ProjectState

BUILD = (f"{ids.COR_STATUS}.children", ids.COR_RUN_BTN)
POLL = (f"{ids.COR_POLL}.disabled", ids.COR_POLL)
SAVE_GATE = (f"{ids.GO_SAVE_CORRIDOR}.data", ids.COR_SAVE_BTN)      # the click: the `save_corridor` gate of confirm.py
SAVE = (f"{ids.COR_STATUS}.children", ids.GO_SAVE_CORRIDOR)         # the save itself, after the gate's GO store fired
CONFIRM = (f"{ids.CONFIRM_PENDING}.data", ids.CONFIRM_OK)           # the modal's Confirm click (key: pending store)
MIRROR = (f"{ids.COR_SETTINGS}.data",)
RESTORE = (f"{ids.COR_TERMINALS}.value", ids.PROJECT_STATUS)
READY = (f"{section.RUN_WRAP}.title",)
BROWSE = (f"{ids.COR_SAVE_PATH}.value", f"browse-{ids.COR_SAVE_PATH}")

CELL = 2.0
X0, Y0 = 442000.0, 5587000.0
CRS = "EPSG:25832"
WALL, OPEN = 65535, 100
Y_A, Y_B, Y_C = (10, 0), (0, 20), (10, 39)           # a trunk along row 10 and a branch up column 20
JOIN_TIMEOUT_S = 60.0


# ------------------------------------------------------------------ helpers
def xy(row: int, col: int) -> tuple[float, float]:
    return X0 + (col + 0.5) * CELL, Y0 - (row + 0.5) * CELL


def _write_raster(path: Path, array: np.ndarray) -> str:
    transform = Affine(CELL, 0.0, X0, 0.0, -CELL, Y0)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=str(array.dtype), crs=CRS, transform=transform) as dataset:
        dataset.write(array, 1)
    return str(path)


def _draft(sources=(), waypoints=(), targets=()):
    def points(cells):
        return [{"lat": 0.0, "lng": 0.0, "x": xy(*c)[0], "y": xy(*c)[1], "name": ""} for c in cells]
    return {"sources": points(sources), "waypoints": points(waypoints), "targets": points(targets)}


Y_DRAFT = _draft(sources=[Y_A], waypoints=[Y_B], targets=[Y_C])


@pytest.fixture(scope="module")
def y_net(tmp_path_factory) -> str:
    array = np.full((16, 44), WALL, dtype=np.uint16)
    array[10, 0:40] = OPEN
    array[0:10, 20] = OPEN
    return _write_raster(tmp_path_factory.mktemp("corridor_ui_y") / "y.tif", array)


@pytest.fixture()
def page(state):
    return build_layout(state)


@pytest.fixture()
def layer(state, y_net):
    """The cost raster layer the run reads (only its file path matters to the card)."""
    state.routing_settings = {"neighborhood_k": 0}
    return state.add_layer("Cost raster", "raster", meta={"source_path": y_net})


def _components(page):
    return list(page._traverse())  # noqa: SLF001


def _by_id(page, component_id):
    found = [c for c in _components(page) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id}: {len(found)} components"
    return found[0]


def _label_of(page, component_id):
    return next(c for c in _components(page)
                if isinstance(c, dbc.Label) and getattr(c, "html_for", None) == component_id)


def _wrapper_of(page, component):
    """The ``html.Div`` that holds ``component`` as its child or one of its children (the hover-text wrapper)."""
    def holds(div):
        children = div.children if isinstance(div.children, list) else [div.children]
        return any(child is component for child in children)

    return next(c for c in _components(page) if isinstance(c, html.Div) and holds(c))


def _text(component) -> str:
    return " ".join(part.children for part in component._traverse()  # noqa: SLF001
                    if isinstance(getattr(part, "children", None), str))


def _settings(**over) -> dict:
    return {**service.CORRIDOR_DEFAULTS, **over}


def _build(app, layer_, draft=Y_DRAFT, settings=None, notices=None, clicks=1):
    s = _settings(**(settings or {}))
    return invoke(app, BUILD, clicks, layer_.id if layer_ is not None else None, draft, s["terminals"],
                  s["k_per_pair"], s["min_shared_length_m"], s["max_pair_cost"], s["validate_pair_costs"],
                  list(notices or []))


def _poll(app, notices=None):
    """The poll as the page runs it: one answer while the job runs; the outcome, handed over and confirmed,
    once it has ended (round 15: ``conftest.poll_job``)."""
    return poll_job(app, ids.JOB_ACK_CORRIDOR, notices)


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _join(state) -> jobs.BackgroundJob:
    job = state.jobs[service.JOB_KIND]
    job.thread.join(timeout=JOIN_TIMEOUT_S)
    assert not job.thread.is_alive(), "the corridor job did not end"
    return job


def _run_to_the_end(app, state, layer_, **kwargs):
    """Start a run, wait for the thread, let the poll finish it: ``(start response, job, poll response)``."""
    start = _build(app, layer_, **kwargs)
    assert start[ids.COR_POLL]["disabled"] is False, start
    job = _join(state)
    return start, job, _poll(app)


def _names(state) -> list[str]:
    return [lay.name for lay in state.ordered_layers()]


# ================================================================== the layout
def test_the_card_is_collapsible_closed_by_default_and_titled(page):
    card = _by_id(page, ids.COR_CARD)
    assert isinstance(card, html.Details) and card.open is False
    assert card.children[0].children[0].children == "Corridor graph"
    assert "gui-card-collapsible" in card.className


def test_the_card_sits_in_the_routes_tab_after_the_constrained_card(page):
    routes = next(c for c in _components(page) if isinstance(c, html.Div)
                  and isinstance(c.children, list) and any(getattr(ch, "id", None) == ids.COR_CARD
                                                           for ch in c.children))
    titles = [_text(ch.children[0]) if isinstance(ch, html.Details) else "" for ch in routes.children]
    position = routes.children.index(_by_id(page, ids.COR_CARD))
    assert titles.index("Constrained routing") < position < titles.index("Computed routes")
    assert any(getattr(part, "id", None) == ids.ROUTE_RASTER for part in routes._traverse())  # noqa: SLF001


def test_every_control_and_part_exists_exactly_once(page):
    for component_id in (ids.COR_CARD, ids.COR_TERMINALS, ids.COR_K, ids.COR_MIN_SHARED, ids.COR_MAX_PAIR_COST,
                         ids.COR_VALIDATE, ids.COR_RUN_BTN, ids.COR_STATUS, ids.COR_SAVE_PATH, ids.COR_SAVE_BTN,
                         ids.COR_POLL, ids.COR_SETTINGS, section.RUN_WRAP, f"browse-{ids.COR_SAVE_PATH}"):
        _by_id(page, component_id)
    assert _by_id(page, ids.COR_POLL).disabled is True and _by_id(page, ids.COR_POLL).interval == 800
    assert _by_id(page, ids.COR_SETTINGS).data == {}


def test_the_setting_ids_follow_the_service_keys():
    assert tuple(section.SETTING_IDS) == service.SETTING_KEYS
    assert tuple(section.TERMINAL_LABELS) == service.TERMINAL_CHOICES
    assert tuple(target for target, _ in section.FORM_TARGETS) == tuple(section.SETTING_IDS.values())


@pytest.mark.parametrize("key", ["k_per_pair", "min_shared_length_m", "max_pair_cost"])
def test_each_number_input_comes_from_its_spec(page, key):
    spec = service.FIELD_SPECS[key]
    control = _by_id(page, section.SETTING_IDS[key])
    assert isinstance(control, dbc.Input) and control.type == "number" and control.debounce is True
    assert (control.min, getattr(control, "max", None)) == (spec.minimum, spec.maximum)
    assert control.step == (1 if spec.whole else "any")
    label = _label_of(page, section.SETTING_IDS[key])
    assert label.children == (f"{spec.label} ({spec.unit})" if spec.unit else spec.label)
    wrapper = _wrapper_of(page, control)
    assert spec.help in wrapper.title and service.describe_range(spec) in wrapper.title
    assert getattr(control, "value", None) == spec.default


def test_every_unit_is_in_a_label_and_a_blank_cost_limit_says_what_blank_means(page):
    for key, spec in service.FIELD_SPECS.items():
        if spec.unit:
            assert f"({spec.unit})" in _label_of(page, section.SETTING_IDS[key]).children
    spec = service.FIELD_SPECS["max_pair_cost"]
    assert spec.blank_means in _by_id(page, ids.COR_MAX_PAIR_COST).placeholder


def test_the_terminal_radio_comes_from_the_spec(page):
    spec = service.FIELD_SPECS["terminals"]
    radios = _by_id(page, ids.COR_TERMINALS)
    assert isinstance(radios, dbc.RadioItems) and radios.value == spec.default
    assert [o["value"] for o in radios.options] == list(spec.options)
    assert [o["label"].strip() for o in radios.options] == ["All control points", "Sources and targets only"]
    group = next(c for c in _components(page) if getattr(c, "role", None) == "group"
                 and any(part is radios for part in c._traverse()))  # noqa: SLF001
    assert getattr(group, "aria-labelledby") == f"{ids.COR_TERMINALS}-label"
    assert _text(group).startswith(spec.label)
    assert spec.help in _wrapper_of(page, group).title


def test_the_switch_comes_from_the_spec(page):
    spec = service.FIELD_SPECS["validate_pair_costs"]
    switch = _by_id(page, ids.COR_VALIDATE)
    assert isinstance(switch, dbc.Switch) and switch.value is False and switch.label == spec.label
    assert spec.help in _wrapper_of(page, switch).title


def test_the_help_text_says_what_the_graph_is_built_on_and_that_it_must_be_rebuilt(page):
    text = _text(_by_id(page, ids.COR_CARD))
    for fragment in ("plain cost raster", "cython backend only", "no DEM", "no objective weights",
                     "cannot be stopped", "not stored in the project file", "build the graph again"):
        assert fragment in text, fragment
    assert "reloaded" in text


def test_the_build_button_starts_disabled_with_a_reason_and_has_no_stop_button(page):
    button = _by_id(page, ids.COR_RUN_BTN)
    assert button.children == "Build corridor graph" and button.disabled is True
    assert "at least 2 points" in _wrapper_of(page, button).title
    assert not [c for c in _components(page) if str(getattr(c, "id", "")).startswith("cor-") and "stop" in c.id]


def test_the_status_line_is_a_polite_live_region(page):
    status = _by_id(page, ids.COR_STATUS)
    assert status.role == "status" and getattr(status, "aria-live") == "polite"


def test_the_save_row_names_the_file_type_and_the_browse_button_what_it_picks(page):
    assert ".gpkg" in _label_of(page, ids.COR_SAVE_PATH).children
    browse = _by_id(page, f"browse-{ids.COR_SAVE_PATH}")
    assert getattr(browse, "aria-label").startswith("Browse for ") and browse.title
    assert _by_id(page, ids.COR_SAVE_BTN).children == "Save graph"


# ================================================================== the Build button follows the page
@pytest.mark.parametrize(("graph_api", "draft", "terminals", "ready", "fragment"), [
    ("cython", Y_DRAFT, "all", True, "3 points"),
    ("cython", Y_DRAFT, "sources_targets", True, "2 points"),
    ("cython", _draft(sources=[Y_A]), "all", False, "at least 2 points, but the route draft gives 1"),
    ("cython", _draft(sources=[Y_A], waypoints=[Y_B]), "sources_targets", False, "gives 1"),
    ("cython", None, "all", False, "gives 0"),
    ("networkx", Y_DRAFT, "all", False, "cython backend only"),
    (None, Y_DRAFT, "all", True, "3 points"),                      # a blank Backend field means the default
])
def test_the_build_button_is_enabled_only_for_cython_and_two_points(app, graph_api, draft, terminals, ready,
                                                                    fragment):
    resp = invoke(app, READY, graph_api, draft, terminals)
    assert resp[ids.COR_RUN_BTN]["disabled"] is (not ready)
    assert fragment in resp[section.RUN_WRAP]["title"]


def test_a_draft_that_is_not_a_mapping_leaves_the_button_disabled(app):
    resp = invoke(app, READY, "cython", ["not", "a", "draft"], "all")
    assert resp[ids.COR_RUN_BTN]["disabled"] is True


# ================================================================== the Build click: problems
def _problem(resp):
    (notice,) = _notices(resp)
    return notice


def test_a_click_without_a_count_does_nothing(app, layer):
    with pytest.raises(PreventUpdate):
        _build(app, layer, clicks=None)


def test_no_raster_is_a_notice_that_focuses_the_raster_dropdown(app, state):
    resp = _build(app, None)
    notice = _problem(resp)
    assert notice["focus_control"] == ids.ROUTE_RASTER and notice["focus_id"] == ids.TAB_ROUTES
    assert notice["severity"] == "warning" and "No corridor graph was built" in notice["impact"]
    assert "No cost raster is selected" in resp[ids.COR_STATUS]["children"]
    assert state.jobs == {} and ids.COR_POLL not in resp


def test_a_layer_that_is_not_a_raster_counts_as_no_raster(app, state):
    vector = state.add_layer("Points", "vector")
    resp = _build(app, vector)
    assert _problem(resp)["focus_control"] == ids.ROUTE_RASTER and state.jobs == {}


def test_a_missing_raster_file_is_named(app, state, tmp_path):
    gone = state.add_layer("Gone", "raster", meta={"source_path": str(tmp_path / "gone.tif")})
    state.routing_settings = {"neighborhood_k": 0}
    resp = _build(app, gone)
    assert _problem(resp)["focus_control"] == ids.ROUTE_RASTER
    assert "cannot be read" in resp[ids.COR_STATUS]["children"] and state.jobs == {}


@pytest.mark.parametrize(("draft", "fragment"), [
    (_draft(sources=[Y_A]), "at least 2 points"),
    (_draft(sources=[Y_A], targets=[(5, 5)]), "forbidden cells"),
    (_draft(sources=[Y_A], targets=[Y_A]), "duplicates"),
    ({"sources": [{"x": 1.0, "y": 2.0}], "targets": [{"x": 3.0, "y": 4.0}], "waypoints": []}, "outside"),
])
def test_points_that_cannot_be_connected_focus_the_points_table(app, state, layer, draft, fragment):
    resp = _build(app, layer, draft=draft)
    notice = _problem(resp)
    assert notice["focus_control"] == ids.BUILD_POINTS_TABLE and notice["focus_id"] == ids.TAB_ROUTES
    assert fragment in resp[ids.COR_STATUS]["children"] and state.jobs == {}


def test_waypoints_count_only_with_all_control_points(app, state, layer):
    draft = _draft(sources=[Y_A], waypoints=[Y_B])
    resp = _build(app, layer, draft=draft, settings={"terminals": "sources_targets"})
    assert _problem(resp)["focus_control"] == ids.BUILD_POINTS_TABLE and state.jobs == {}


def test_another_backend_focuses_the_backend_field(app, state, layer):
    state.routing_settings = {"graph_api": "networkx", "neighborhood_k": 0}
    resp = _build(app, layer)
    notice = _problem(resp)
    assert notice["focus_control"] == ids.GRAPH_API and "cython" in notice["fix"]
    assert "cython backend only" in resp[ids.COR_STATUS]["children"] and state.jobs == {}


def test_an_unusable_neighbourhood_focuses_the_neighbourhood_field(app, state, layer):
    state.routing_settings = {"neighborhood_k": 999}
    resp = _build(app, layer)
    assert _problem(resp)["focus_control"] == ids.NEIGHBORHOOD and state.jobs == {}


@pytest.mark.parametrize(("key", "value", "control"), [
    ("k_per_pair", 0, ids.COR_K), ("k_per_pair", 6, ids.COR_K), ("k_per_pair", 2.5, ids.COR_K),
    ("k_per_pair", None, ids.COR_K),
    ("min_shared_length_m", -0.5, ids.COR_MIN_SHARED), ("min_shared_length_m", 10000.5, ids.COR_MIN_SHARED),
    ("min_shared_length_m", None, ids.COR_MIN_SHARED),
    ("max_pair_cost", 0, ids.COR_MAX_PAIR_COST), ("max_pair_cost", -1, ids.COR_MAX_PAIR_COST),
    ("terminals", "everything", ids.COR_TERMINALS),
    ("validate_pair_costs", "yes", ids.COR_VALIDATE),
])
def test_each_limit_of_the_specs_is_checked_again_by_the_click(app, state, layer, key, value, control):
    resp = _build(app, layer, settings={key: value})
    notice = _problem(resp)
    assert notice["focus_control"] == control and notice["focus_id"] == ids.TAB_ROUTES
    assert service.label_for(key) in notice["title"] and service.label_for(key) in resp[ids.COR_STATUS]["children"]
    assert state.jobs == {} and ids.COR_POLL not in resp


def test_the_limits_of_the_specs_are_the_ones_the_click_enforces():
    """The edges of every number spec are accepted, the first values beyond them are refused."""
    for key in ("k_per_pair", "min_shared_length_m"):
        spec = service.FIELD_SPECS[key]
        assert service.corridor_problems({key: spec.minimum}) == []
        assert service.corridor_problems({key: spec.maximum}) == []
        assert service.corridor_problems({key: spec.minimum - 1}) and service.corridor_problems({key: spec.maximum + 1})
    assert service.corridor_problems({"max_pair_cost": 0.001}) == []


def test_the_first_three_problems_are_in_the_status_line(app, state, layer):
    resp = _build(app, layer, settings={"k_per_pair": 9, "min_shared_length_m": -4, "max_pair_cost": 0})
    status = resp[ids.COR_STATUS]["children"]
    assert status.count("\n") == 2 and "k_per_pair" in status and "min_shared_length_m" in status
    assert len(_notices(resp)) == 1                        # one notice: the first problem; the status lists more


# ================================================================== the run and the poll
def test_a_run_starts_a_job_without_stop_and_the_status_says_so(app, state, layer):
    start = _build(app, layer)
    job = _join(state)
    try:
        assert start[ids.COR_POLL]["disabled"] is False
        assert "cannot be stopped" in start[ids.COR_STATUS]["children"]
        assert start[ids.COR_STATUS]["children"].startswith("Corridor graph")
        assert job.can_stop is False and job.kind == service.JOB_KIND
        assert ids.NOTICES not in start                                            # nothing to add: history untouched
        assert job.params["raster_layer_id"] == layer.id and job.params["settings"] == _settings()
        assert job.params["neighborhood"] == "r0" and len(job.params["terminals"]) == 3
    finally:
        jobs.take_finished(state, service.JOB_KIND)


def test_the_poll_adds_the_layers_the_summary_and_the_notices_and_switches_off(app, state, layer):
    start, job, resp = _run_to_the_end(app, state, layer)
    assert job.status == jobs.DONE and state.jobs == {}
    assert resp[ids.COR_POLL]["disabled"] is True
    summary = job.result.summary
    assert summary["n_terminals"] == 3 and summary["n_shared_sections"] == 1
    names = [lay["name"] for lay in resp[ids.LAYERS_VIEW]["data"]]
    assert {"Corridor segments", "Corridor shared runs", "Corridor nodes"} <= set(names)
    status = resp[ids.COR_STATUS]["children"]
    assert summary["text"] in status and "Map layers added: 'Corridor segments', 'Corridor shared runs'" in status
    shown = _notices(resp)
    assert shown[-1]["severity"] == "success" and "Corridor graph built" in shown[-1]["title"]
    segments = next(lay for lay in state.layers.values() if lay.meta.get("corridor_role") == "segments")
    assert state.corridor_graphs[segments.id] is job.result.graph
    assert segments.meta["raster_layer_id"] == layer.id and segments.kind == "vector"
    assert segments.meta["corridor_settings"] == _settings()


def test_the_layers_carry_the_map_geometry_and_the_graph_attributes(app, state, layer):
    _run_to_the_end(app, state, layer)
    segments = next(lay for lay in state.layers.values() if lay.meta.get("corridor_role") == "segments")
    assert segments.geojson["features"] and segments.gdf.crs.to_epsg() == 25832
    for column in ("length_m", "construction_cost", "use_count", "shared"):
        assert column in segments.gdf.columns
    shared = next(lay for lay in state.layers.values() if lay.meta.get("corridor_role") == "shared")
    assert set(shared.gdf["shared"]) == {1}


def test_the_poll_shows_the_progress_while_the_job_runs(app, state, layer, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    real = service.run_job

    def slow(job):
        job.report(2, 4, "building the shared trenches (cannot be stopped; you can keep working)")
        entered.set()
        assert release.wait(JOIN_TIMEOUT_S)
        return real(job)

    monkeypatch.setattr(service, "run_job", slow)
    _build(app, layer)
    try:
        assert entered.wait(JOIN_TIMEOUT_S)
        resp = _poll(app, [{"id": "old"}])
        status = resp[ids.COR_STATUS]["children"]
        assert "2/4" in status and "cannot be stopped" in status and status.count("cannot be stopped") == 1
        assert ids.NOTICES not in resp and ids.LAYERS_VIEW not in resp and ids.COR_POLL not in resp
        assert service.JOB_KIND in state.jobs                                     # still there, not collected
    finally:
        release.set()
        _join(state)
    done = _poll(app)
    assert done[ids.COR_POLL]["disabled"] is True and "Corridor segments" in _names(state)


def test_a_poll_without_a_job_switches_off_and_touches_nothing_else(app, state):
    resp = _poll(app)
    assert resp == {ids.COR_POLL: {"disabled": True}}


def test_a_click_while_the_job_runs_gets_the_busy_notice_and_keeps_the_status(app, state, layer, monkeypatch):
    release = threading.Event()
    real = service.run_job
    monkeypatch.setattr(service, "run_job", lambda job: (release.wait(JOIN_TIMEOUT_S), real(job))[1])
    _build(app, layer)
    try:
        resp = _build(app, layer, notices=[])
        (notice,) = _notices(resp)
        assert notice["title"] == "Corridor graph is still running"
        assert ids.COR_STATUS not in resp and ids.COR_POLL not in resp and len(state.jobs) == 1
    finally:
        release.set()
        _join(state)
        jobs.take_finished(state, service.JOB_KIND)


def test_another_running_job_refuses_the_start(app, state, layer):
    other = SimpleNamespace(status=jobs.RUNNING, label="Cost field", cancel=threading.Event(), can_stop=True,
                            progress_text=lambda: "Cost field... 1 s elapsed")
    state.jobs["cost_field"] = other
    resp = _build(app, layer)
    assert _problem(resp)["title"] == "Cost field is still running"
    assert service.JOB_KIND not in state.jobs and ids.COR_POLL not in resp


def test_the_routing_job_refuses_the_start(app, state, layer):
    state.routing_job = SimpleNamespace(status=jobs.RUNNING, cancel=threading.Event())
    resp = _build(app, layer)
    assert _problem(resp)["title"] == "Routing is still running" and state.jobs == {}


def test_an_ended_job_nobody_collected_is_collected_by_the_next_click(app, state, layer):
    _build(app, layer)
    job = _join(state)
    resp = _build(app, layer)                                        # the page did not poll in between
    assert [n["title"] for n in _notices(resp)][0] == "An earlier corridor graph run was collected"
    assert "did not start a new run" in _notices(resp)[0]["impact"]
    # round 15: the click switches the poll on and the poll hands the outcome over
    assert resp[ids.COR_POLL]["disabled"] is False and state.jobs == {service.JOB_KIND: job}
    resp = _poll(app, _notices(resp))
    assert resp[ids.COR_POLL]["disabled"] is True and state.jobs == {}
    assert job.result.summary["text"] in resp[ids.COR_STATUS]["children"] and "Corridor segments" in _names(state)


def test_a_second_run_adds_numbered_layers_and_keeps_both_graphs(app, state, layer):
    _run_to_the_end(app, state, layer)
    _run_to_the_end(app, state, layer)
    assert {"Corridor segments", "Corridor segments (2)", "Corridor nodes (2)"} <= set(_names(state))
    assert len(state.corridor_graphs) == 2


def test_a_heavy_window_adds_its_warning_to_the_start(app, state, layer, monkeypatch):
    monkeypatch.setattr(service, "HEAVY_WINDOW_CELLS", 1)
    start = _build(app, layer)
    try:
        _join(state)
        warning = next(n for n in _notices(start) if n["title"] == "This corridor graph covers a large area")
        assert warning["severity"] == "warning" and "cannot be stopped" in warning["impact"]
    finally:
        jobs.take_finished(state, service.JOB_KIND)


def test_a_run_that_connects_no_pair_is_a_notice_and_leaves_the_project_alone(app, state, layer):
    _, job, resp = _run_to_the_end(app, state, layer, settings={"max_pair_cost": 1.0})
    assert job.status == jobs.DONE and resp[ids.COR_POLL]["disabled"] is True
    assert ids.LAYERS_VIEW not in resp and state.corridor_graphs == {}
    assert [n["title"] for n in _notices(resp)] == ["No corridors were built"]
    assert resp[ids.COR_STATUS]["children"].startswith("No corridors were built.")
    assert "costs more than the limit" in resp[ids.COR_STATUS]["children"]
    assert "Corridor segments" not in _names(state)


def test_a_failed_job_ends_as_a_translated_notice(app, state, layer, monkeypatch):
    def broken(_job):
        raise InvalidInputError("The corridor graph cannot be built: the raster went away.")

    monkeypatch.setattr(service, "run_job", broken)
    _build(app, layer)
    _join(state)
    resp = _poll(app)
    assert resp[ids.COR_POLL]["disabled"] is True and state.jobs == {}
    (notice,) = _notices(resp)
    assert notice["severity"] in {"error", "warning"} and "raster went away" in str(notice)
    status = resp[ids.COR_STATUS]["children"]
    assert status.startswith("Corridor graph failed after") and "raster went away" in status
    assert ids.LAYERS_VIEW not in resp and state.corridor_graphs == {}


def test_a_dropped_job_is_an_info_notice_and_changes_nothing(app, state, layer):
    _build(app, layer)
    job = _join(state)
    job.status = jobs.CANCELLED                                       # what a replaced project does to a job
    resp = _poll(app)
    (notice,) = _notices(resp)
    assert notice["severity"] == "info" and "No corridor graph was kept" in resp[ids.COR_STATUS]["children"]
    assert ids.LAYERS_VIEW not in resp and state.corridor_graphs == {} and resp[ids.COR_POLL]["disabled"] is True


def test_a_finalize_that_raises_is_a_notice_not_a_traceback(app, state, layer, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("layer registry exploded")

    monkeypatch.setattr(service, "finalize_corridor", boom)
    _build(app, layer)
    _join(state)
    resp = _poll(app)
    assert "could not be added" in resp[ids.COR_STATUS]["children"] and _notices(resp)
    assert resp[ids.COR_POLL]["disabled"] is True and state.jobs == {}


# ================================================================== save
def _built(app, state, layer_):
    _run_to_the_end(app, state, layer_)


def _gate(app, path):
    """The Save click: the gate either asks (modal outputs) or fires the GO store (``resp[ids.GO_SAVE_CORRIDOR]``)."""
    return invoke(app, SAVE_GATE, 1, None if path is None else str(path), triggered=[f"{ids.COR_SAVE_BTN}.n_clicks"])


def _save_after_go(app, path, go, notices=None):
    return invoke(app, SAVE, go, None if path is None else str(path), list(notices or []))


def _save(app, path, notices=None):
    """Click Save and, when the gate lets it through, run the save; a gate that asked is returned as it is."""
    gate = _gate(app, path)
    if ids.CONFIRM_MODAL in gate:
        return gate
    return _save_after_go(app, path, gate[ids.GO_SAVE_CORRIDOR]["data"], notices)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_save_without_a_graph_is_refused_and_says_to_build_first(app, state, tmp_path):
    target = tmp_path / "corridor.gpkg"
    resp = _save(app, target)
    (notice,) = _notices(resp)
    assert notice["focus_control"] == ids.COR_SAVE_BTN and notice["impact"] == "Nothing was written."
    assert "Build one first" in resp[ids.COR_STATUS]["children"] and not target.exists()
    assert ids.CONFIRM_MODAL not in resp


def test_save_with_a_blank_or_wrong_path_focuses_the_path_field(app, state, layer, tmp_path):
    _built(app, state, layer)
    (tmp_path / "folder.gpkg").mkdir()
    cases = ((None, "No file name"), ("  ", "No file name"), (str(tmp_path / "x.shp"), "not a GeoPackage"),
             (str(tmp_path / "folder.gpkg"), "is a folder"))
    for path, fragment in cases:
        resp = _save(app, path)
        (notice,) = _notices(resp)
        assert notice["focus_control"] == ids.COR_SAVE_PATH and fragment in resp[ids.COR_STATUS]["children"]
        assert ids.CONFIRM_MODAL not in resp
    assert not [p for p in tmp_path.glob("*.gpkg") if p.is_file()]


def test_save_writes_a_geopackage_and_reports_where_it_went(app, state, layer, tmp_path):
    _built(app, state, layer)
    target = tmp_path / "out" / "corridor.gpkg"                       # the folder does not exist yet
    resp = _save(app, target, notices=[{"id": "old", "severity": "info", "title": "kept"}])
    assert target.is_file()
    status = resp[ids.COR_STATUS]["children"]
    assert str(target) in status and "Corridor segments" in status
    assert resp[ids.NOTICES]["data"][0]["id"] == "old"
    done = resp[ids.NOTICES]["data"][-1]
    assert done["severity"] == "success" and str(target) in done["meaning"]
    assert service.SEGMENTS_LAYER in done["impact"] and ids.CONFIRM_MODAL not in resp
    _, graph = callbacks.latest_graph(state)
    written = gpd.read_file(target, layer=service.SEGMENTS_LAYER)
    assert len(written) == len(graph.segments) and set(written["shared"]) == {0, 1}
    assert len(gpd.read_file(target, layer=service.NODES_LAYER)) == len(graph.nodes_to_geodataframe())
    assert not list(target.parent.glob("*.tmp.gpkg"))


def test_save_writes_the_graph_of_the_newest_run(app, state, layer, tmp_path):
    _built(app, state, layer)
    _built(app, state, layer)
    layer_id, graph = callbacks.latest_graph(state)
    assert state.layers[layer_id].name == "Corridor segments (2)"
    resp = _save(app, tmp_path / "second.gpkg")
    assert "Corridor segments (2)" in resp[ids.COR_STATUS]["children"]


def test_the_graph_of_a_removed_layer_is_not_saved(app, state, layer, tmp_path):
    _built(app, state, layer)
    layer_id, _ = callbacks.latest_graph(state)
    state.remove_layer(layer_id)
    assert callbacks.latest_graph(state) == (None, None)
    resp = _save(app, tmp_path / "gone.gpkg")
    assert "Build one first" in resp[ids.COR_STATUS]["children"] and not (tmp_path / "gone.gpkg").exists()


def test_an_existing_file_is_asked_about_and_left_alone_until_the_answer(app, state, layer, tmp_path):
    _built(app, state, layer)
    target = tmp_path / "corridor.gpkg"
    target.write_bytes(b"precious old content")
    before = _sha(target)
    resp = _gate(app, target)
    assert resp[ids.CONFIRM_MODAL]["is_open"] is True and ids.GO_SAVE_CORRIDOR not in resp
    assert resp[ids.CONFIRM_TITLE]["children"] == "Replace the existing corridor GeoPackage?"
    assert resp[ids.CONFIRM_OK]["children"] == "Replace it"
    body = str(resp[ids.CONFIRM_BODY]["children"])
    assert "corridor.gpkg" in body and "already exists" in body
    assert "the old content cannot be brought back from here" in body
    assert resp[ids.CONFIRM_PENDING]["data"] == {"action": "save_corridor"}
    assert ids.COR_STATUS not in resp and ids.NOTICES not in resp
    assert _sha(target) == before


def test_confirming_fires_the_save_which_replaces_the_file_and_reports_the_path(app, state, layer, tmp_path):
    _built(app, state, layer)
    target = tmp_path / "corridor.gpkg"
    target.write_bytes(b"old")
    asked = _gate(app, target)
    assert asked[ids.CONFIRM_PENDING]["data"] == {"action": "save_corridor"} and target.read_bytes() == b"old"
    answer = invoke(app, CONFIRM, 1, asked[ids.CONFIRM_PENDING]["data"])
    assert answer[ids.CONFIRM_MODAL]["is_open"] is False and answer[ids.CONFIRM_PENDING]["data"] is None
    assert answer[ids.GO_SAVE_CORRIDOR]["data"]
    assert target.read_bytes() == b"old"                              # the answer alone writes nothing
    resp = _save_after_go(app, target, answer[ids.GO_SAVE_CORRIDOR]["data"])
    assert str(target) in resp[ids.COR_STATUS]["children"]
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Corridor graph saved"
    assert len(gpd.read_file(target, layer=service.SEGMENTS_LAYER)) > 0


def test_cancelling_the_question_leaves_the_file_alone(app, state, layer, tmp_path):
    _built(app, state, layer)
    target = tmp_path / "corridor.gpkg"
    target.write_bytes(b"precious old content")
    assert _gate(app, target)[ids.CONFIRM_MODAL]["is_open"] is True
    cancelled = invoke(app, (f"{ids.CONFIRM_PENDING}.data", ids.CONFIRM_CANCEL), 1)
    assert cancelled[ids.CONFIRM_MODAL]["is_open"] is False and cancelled[ids.CONFIRM_PENDING]["data"] is None
    assert target.read_bytes() == b"precious old content"


def test_a_new_file_name_goes_ahead_without_a_question(app, state, layer, tmp_path):
    _built(app, state, layer)
    resp = _gate(app, tmp_path / "fresh.gpkg")
    assert resp[ids.GO_SAVE_CORRIDOR]["data"] and ids.CONFIRM_MODAL not in resp


def test_an_existing_file_without_a_graph_is_not_asked_about_and_stays_untouched(app, state, tmp_path):
    target = tmp_path / "corridor.gpkg"
    target.write_bytes(b"old")
    resp = _gate(app, target)                                         # the real problem is "no graph": no question
    assert resp[ids.GO_SAVE_CORRIDOR]["data"] and ids.CONFIRM_MODAL not in resp
    done = _save_after_go(app, target, resp[ids.GO_SAVE_CORRIDOR]["data"])
    assert target.read_bytes() == b"old" and "Build one first" in done[ids.COR_STATUS]["children"]


def test_the_card_has_no_listener_of_its_own_on_the_confirm_modal(app):
    """The corridor save is a gate in ``confirm.GATES`` now: one callback answers the Confirm click."""
    from pyorps.gui.callbacks import confirm
    assert "save_corridor" in {gate.action for gate in confirm.GATES}
    answering = [key for key, entry in app.callback_map.items()
                 if any(item["id"] == ids.CONFIRM_OK for item in entry.get("inputs", []))]
    assert len(answering) == 1
    save_listeners = [entry for entry in app.callback_map.values()
                      if any(item["id"] == ids.COR_SAVE_BTN for item in entry.get("inputs", []))]
    assert len(save_listeners) == 1                                   # the gate, and nothing else
    assert [item["id"] for item in save_listeners[0]["inputs"]] == [ids.COR_SAVE_BTN]


def test_a_save_without_a_go_count_does_nothing(app, state):
    with pytest.raises(PreventUpdate):
        invoke(app, SAVE, None, "x.gpkg", [])


def test_a_write_that_fails_ends_as_a_notice(app, state, layer, tmp_path, monkeypatch):
    _built(app, state, layer)
    _, graph = callbacks.latest_graph(state)

    def refuse(_path):
        raise PermissionError("the folder is read-only")

    monkeypatch.setattr(graph, "save", refuse)
    resp = _save(app, tmp_path / "locked.gpkg")
    assert "was not saved" in resp[ids.COR_STATUS]["children"] and _notices(resp)
    assert not (tmp_path / "locked.gpkg").exists()


# ================================================================== the browse button
def _browse(app, clicks=1):
    return invoke(app, BROWSE, clicks, outputs_list=[{"id": ids.COR_SAVE_PATH, "property": "value"},
                                                      {"id": ids.NOTICES, "property": "data"}])


def test_browse_fills_the_save_path(app, monkeypatch):
    seen = {}

    def chosen(kind, title="", defaultextension=""):
        seen.update(kind=kind, extension=defaultextension)
        return dialogs.DialogResult.chosen("C:/out/corridor.gpkg")

    monkeypatch.setattr(dialogs, "ask_save", chosen)
    resp = _browse(app)
    assert resp[ids.COR_SAVE_PATH]["value"] == "C:/out/corridor.gpkg" and ids.NOTICES not in resp
    assert seen == {"kind": "routes", "extension": ".gpkg"}


def test_browse_cancel_is_silent_and_unavailable_is_a_notice(app, monkeypatch):
    monkeypatch.setattr(dialogs, "ask_save", lambda *a, **k: dialogs.DialogResult.cancelled())
    resp = _browse(app)
    assert ids.COR_SAVE_PATH not in resp and ids.NOTICES not in resp
    monkeypatch.setattr(dialogs, "ask_save", lambda *a, **k: dialogs.DialogResult.unavailable("no display"))
    resp = _browse(app)
    assert ids.COR_SAVE_PATH not in resp
    (operation,) = resp[ids.NOTICES]["data"]["operations"]
    assert operation["params"]["value"]["title"].startswith("The file dialog")
    with pytest.raises(PreventUpdate):
        _browse(app, clicks=None)


# ================================================================== settings: mirror, restore, prefill
def test_the_settings_are_mirrored_into_the_state_and_the_store(app, state):
    resp = invoke(app, MIRROR, "sources_targets", 3, 25.5, 4000.0, True)
    saved = {"terminals": "sources_targets", "k_per_pair": 3, "min_shared_length_m": 25.5,
             "max_pair_cost": 4000.0, "validate_pair_costs": True}
    assert state.gui_settings["corridor"] == saved and resp[ids.COR_SETTINGS]["data"] == saved


def test_blank_fields_fall_back_to_the_defaults_in_the_mirror(app, state):
    invoke(app, MIRROR, "all", None, None, None, False)
    assert state.gui_settings["corridor"] == service.CORRIDOR_DEFAULTS


def test_a_value_outside_its_range_is_kept_so_the_click_can_name_it(app, state):
    invoke(app, MIRROR, "all", 9, -4, 0, False)
    assert state.gui_settings["corridor"]["k_per_pair"] == 9
    assert service.corridor_problems(state.gui_settings["corridor"])


def test_the_mirror_changes_the_dict_in_place_and_leaves_other_settings_alone(app, state):
    state.gui_settings["objective"] = {"keep": 1}
    held = state.gui_settings
    invoke(app, MIRROR, "all", 1, 0, None, False)
    assert state.gui_settings is held and held["objective"] == {"keep": 1} and "corridor" in held


def _restore(app, status="opened: p"):
    return invoke(app, RESTORE, status, triggered=[f"{ids.PROJECT_STATUS}.children"])


def test_open_project_puts_the_saved_settings_back_and_clears_the_status(app, state):
    state.gui_settings = {"corridor": {"terminals": "sources_targets", "k_per_pair": 2, "min_shared_length_m": 15.0,
                                       "max_pair_cost": 900.0, "validate_pair_costs": True}}
    resp = _restore(app)
    assert resp[ids.COR_TERMINALS]["value"] == "sources_targets" and resp[ids.COR_K]["value"] == 2
    assert resp[ids.COR_MIN_SHARED]["value"] == 15.0 and resp[ids.COR_MAX_PAIR_COST]["value"] == 900.0
    assert resp[ids.COR_VALIDATE]["value"] is True and resp[ids.COR_STATUS]["children"] == ""


@pytest.mark.parametrize("settings", [{}, {"objective": {"x": 1}}, {"corridor": {}}])
def test_new_project_and_an_older_project_put_the_defaults_back(app, state, settings):
    state.gui_settings = settings                                      # "{}" = New, or a project without the card
    resp = _restore(app, status="new project")
    assert (resp[ids.COR_TERMINALS]["value"], resp[ids.COR_K]["value"], resp[ids.COR_MIN_SHARED]["value"],
            resp[ids.COR_MAX_PAIR_COST]["value"], resp[ids.COR_VALIDATE]["value"]) == (
        "all", 1, 0.0, None, False)


def test_saving_a_project_does_not_rewrite_the_card(app, state):
    state.gui_settings = {"corridor": {"k_per_pair": 4}}
    with pytest.raises(PreventUpdate):
        _restore(app, status="saved: C:/p/project.json")


def test_a_blank_project_prefills_nothing(state):
    puts: list = []
    section.prefill(state, lambda *args: puts.append(args))
    assert puts == []
    state.gui_settings = {"corridor": {}}
    section.prefill(state, lambda *args: puts.append(args))
    assert puts == []


def test_prefill_puts_every_control_and_the_store(state):
    state.gui_settings = {"corridor": {"terminals": "sources_targets", "k_per_pair": 2, "max_pair_cost": 700.0}}
    puts: dict = {}
    section.prefill(state, lambda cid, prop, value: puts.__setitem__((cid, prop), value))
    assert puts[(ids.COR_TERMINALS, "value")] == "sources_targets" and puts[(ids.COR_K, "value")] == 2
    assert puts[(ids.COR_MAX_PAIR_COST, "value")] == 700.0 and puts[(ids.COR_MIN_SHARED, "value")] == 0.0
    assert puts[(ids.COR_VALIDATE, "value")] is False
    assert puts[(ids.COR_SETTINGS, "data")]["k_per_pair"] == 2 and len(puts) == 6


def test_a_reloaded_page_shows_the_saved_settings(state):
    state.gui_settings = {"corridor": {"terminals": "sources_targets", "k_per_pair": 4, "min_shared_length_m": 12.5,
                                       "max_pair_cost": 321.0, "validate_pair_costs": True}}
    page = build_layout(state)
    assert _by_id(page, ids.COR_TERMINALS).value == "sources_targets" and _by_id(page, ids.COR_K).value == 4
    assert _by_id(page, ids.COR_MIN_SHARED).value == 12.5 and _by_id(page, ids.COR_MAX_PAIR_COST).value == 321.0
    assert _by_id(page, ids.COR_VALIDATE).value is True
    assert _by_id(page, ids.COR_SETTINGS).data == state.gui_settings["corridor"]
    assert _by_id(page, ids.COR_CARD).open is False


def test_a_fresh_page_has_no_status_and_its_poll_is_off(state):
    page = build_layout(state)
    assert getattr(_by_id(page, ids.COR_STATUS), "children", None) is None
    assert _by_id(page, ids.COR_POLL).disabled is True


def test_a_running_job_survives_a_page_reload_with_its_poll_and_status(state):
    release = threading.Event()
    job = jobs.start_job(state, service.JOB_KIND, lambda j: release.wait(JOIN_TIMEOUT_S), label=service.JOB_LABEL,
                         can_stop=False)
    try:
        page = build_layout(state)
        assert _by_id(page, ids.COR_POLL).disabled is False
        assert _by_id(page, ids.COR_STATUS).children.startswith("Corridor graph...")
    finally:
        release.set()
        job.thread.join(timeout=JOIN_TIMEOUT_S)


def test_the_settings_survive_save_and_open(app, state, tmp_path):
    invoke(app, MIRROR, "sources_targets", 2, 5.0, 1000.0, True)
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings["corridor"] == state.gui_settings["corridor"]
        page = build_layout(other)
        assert _by_id(page, ids.COR_TERMINALS).value == "sources_targets"
        assert _by_id(page, ids.COR_MAX_PAIR_COST).value == 1000.0
    finally:
        other.shutdown()


def test_the_corridor_layers_are_saved_with_the_project_but_not_the_graph(app, state, layer, tmp_path):
    _built(app, state, layer)
    state.remove_layer(layer.id)                                      # a raster without tiles is not under test here
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert "Corridor segments" in _names(other) and other.corridor_graphs == {}
        resp = invoke(build_app(other), SAVE, 1, str(tmp_path / "again.gpkg"), [])
        assert "Build one first" in resp[ids.COR_STATUS]["children"] and not (tmp_path / "again.gpkg").exists()
        assert "Saved the corridor graph" in _save(app, tmp_path / "again.gpkg")[ids.COR_STATUS]["children"]
    finally:
        other.shutdown()


# ================================================================== no callback loop
def _prop(component_id, prop) -> tuple[str, str]:
    return str(component_id).split("@", maxsplit=1)[0], prop


def _graph(app):
    """Callback key -> (outputs, inputs) as ``(id, prop)`` pairs; States are not triggers."""
    graph = {}
    for key, entry in app.callback_map.items():
        outputs = entry["output"] if isinstance(entry["output"], (list, tuple)) else [entry["output"]]
        graph[key] = ({_prop(o.component_id, o.component_property) for o in outputs},
                      {_prop(i["id"], i["property"]) for i in entry.get("inputs", []) if isinstance(i["id"], str)})
    return graph


def _reachable(graph, start):
    seen, todo = set(), [start]
    while todo:
        key = todo.pop()
        for other, (_, inputs) in graph.items():
            if graph[key][0] & inputs and other not in seen:
                seen.add(other)
                todo.append(other)
    return seen


def test_no_corridor_callback_can_reach_itself_through_other_callbacks(app):
    graph = _graph(app)
    mine = [key for key, (outputs, inputs) in graph.items()
            if any(cid.startswith("cor-") or cid == ids.COR_POLL for cid, _ in outputs | inputs)]
    assert len(mine) == 8
    for key in mine:
        assert key not in _reachable(graph, key), key


def test_the_controls_are_only_ever_written_by_the_restore_callback(app):
    graph = _graph(app)
    for cid in section.SETTING_IDS.values():
        writers = [key for key, (outputs, _) in graph.items() if (cid, "value") in outputs]
        assert len(writers) == 1 and (ids.PROJECT_STATUS, "children") in graph[writers[0]][1], cid


def test_the_job_has_no_stop_callback_and_the_scaffold_knows_it():
    from pyorps.gui.layout import JOB_CONTROLS

    assert JOB_CONTROLS["corridor"] == (ids.COR_POLL, None, ids.COR_STATUS)


def test_the_poll_is_the_only_collector_besides_the_build_click(app):
    taking = [key for key, entry in app.callback_map.items()
              if any(i["id"] == ids.COR_POLL for i in entry.get("inputs", []))]
    assert len(taking) == 1


def test_every_problem_key_of_the_service_has_a_control_in_the_page(page):
    keys = {*service.SETTING_KEYS, "points", "raster", "graph_api", "neighborhood_k", "settings", "graph", "save_path"}
    assert set(section.CONTROLS) == keys
    for control in section.CONTROLS.values():
        _by_id(page, control)

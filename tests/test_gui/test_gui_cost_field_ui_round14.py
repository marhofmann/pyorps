"""Round 14, P11 - the "Cost field" card of the Routes tab (``sections/cost_field.py``, ``callbacks/cost_field.py``).

Headless callback tests (the harness of ``conftest.py`` drives the callbacks as Dash would) plus layout checks, on a
tiny synthetic raster. Components are addressed as "<id>.<prop>": several ids contain each other. A test that needs
the worker to wait holds it on an Event it opens itself (the compute step is replaced by a fake that waits), and every
job thread is joined with a timeout, so no thread outlives its test. The numbers of the field itself are pinned by
``test_gui_cost_field_service_round14.py``; here the wiring is: the click, the checks and their notices, the job, the
poll, Stop, the settings and the page.
"""
from __future__ import annotations

import json
import threading

import dash_bootstrap_components as dbc
import numpy as np
import pytest
import rasterio
from conftest import invoke, poll_job
from dash import html
from dash.exceptions import PreventUpdate
from rasterio.transform import from_origin

from pyorps.gui import ids
from pyorps.gui.callbacks import cost_field as cf_cb
from pyorps.gui.layout import _state_prefill, build_layout
from pyorps.gui.sections import cost_field as section
from pyorps.gui.services import cost_field as service
from pyorps.gui.services import jobs, project_io, routing
from pyorps.gui.state import ProjectState

CRS = "EPSG:32632"
X0, Y0 = 500000.0, 5000000.0
SIZE = 20
FORBIDDEN = 65535
JOIN_S = 60
KIND = "cost_field"
HIDDEN = {"display": "none"}

RUN = (f"{ids.CF_STATUS}.children", ids.CF_RUN_BTN)
POLL = (f"{ids.CF_STATUS}.children", ids.CF_POLL)
STOP = (f"{ids.CF_STATUS}.children", ids.CF_STOP_BTN)
BUTTON = (f"{ids.CF_RUN_BTN}.disabled",)
MIRROR = (f"{ids.CF_SETTINGS}.data",)
RESTORE = (f"{ids.CF_ORIGINS}.value", ids.PROJECT_STATUS)


# ------------------------------------------------------------------ helpers
def _write(path, data: np.ndarray) -> str:
    rows, cols = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1, dtype=str(data.dtype), crs=CRS,
                       transform=from_origin(X0, Y0 + rows, 1, 1)) as dst:
        dst.write(data, 1)
    return str(path)


@pytest.fixture(scope="module")
def uniform_path(tmp_path_factory):
    """20 x 20 cells of 1 m, every cell costs 10."""
    return _write(tmp_path_factory.mktemp("p11_uniform") / "uniform.tif", np.full((SIZE, SIZE), 10, dtype=np.uint16))


@pytest.fixture(scope="module")
def forbidden_path(tmp_path_factory):
    data = np.full((SIZE, SIZE), 10, dtype=np.uint16)
    data[5, 5] = FORBIDDEN
    return _write(tmp_path_factory.mktemp("p11_forbidden") / "forbidden.tif", data)


@pytest.fixture(scope="module")
def float_path(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("p11_float") / "float.tif", np.full((SIZE, SIZE), 10, dtype=np.float32))


def _raster_layer(state, path):
    """A raster layer without a tile server (a run only needs its source path)."""
    return state.add_layer("cost", "raster", crs=CRS, meta={"source_path": path})


@pytest.fixture()
def layer(state, uniform_path):
    return _raster_layer(state, uniform_path)


def _pt(row: int, col: int) -> dict:
    """The point at the centre of cell (row, col). lat/lng are the placeholders the points list writes."""
    return {"lat": 0, "lng": 0, "x": X0 + col + 0.5, "y": Y0 + SIZE - row - 0.5}


#: two sources in one row, ten cells apart: the cheapest cell of the sum is worth exactly 10 cells x 10
TWO_SOURCES = {"sources": [_pt(10, 2), _pt(10, 12)], "targets": [], "waypoints": []}
ONE_SOURCE_EACH = {"sources": [_pt(10, 2)], "targets": [_pt(10, 12)], "waypoints": [_pt(4, 8)]}


def _click(app, raster_id, draft, *, origins="sources", combine="sum", algorithm="dijkstra", threads=0, notices=None,
           clicks=1):
    return invoke(app, RUN, clicks, raster_id, draft, origins, combine, algorithm, threads, list(notices or []),
                  triggered=[f"{ids.CF_RUN_BTN}.n_clicks"])


def _poll(app, notices=None):
    """The poll as the page runs it: one answer while the job runs; the outcome, handed over and confirmed,
    once it has ended (round 15: ``conftest.poll_job``)."""
    return poll_job(app, ids.JOB_ACK_COST_FIELD, notices)


def _stop(app):
    return invoke(app, STOP, 1, triggered=[f"{ids.CF_STOP_BTN}.n_clicks"])


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _titles(resp) -> list[str]:
    return [n["title"] for n in _notices(resp)]


def _join(state):
    """The running job of the card, joined with a timeout (it stays registered until a poll takes it)."""
    job = state.jobs[KIND]
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive(), "the worker did not end"
    return job


def _cost_field_layers(state):
    return [ly for ly in state.layers_of_kind("raster") if "cost_field" in (ly.meta or {})]


def _as_json(value):
    return json.loads(json.dumps(value))


class Gate:
    """Holds the worker thread in the (fake) compute step until the test opens the gate."""

    def __init__(self):
        self.reached = threading.Event()
        self.open = threading.Event()
        self.calls: list[dict] = []


@pytest.fixture()
def gate():
    held = Gate()
    yield held
    held.open.set()                          # a failing test leaves no blocked thread behind


def _fake_compute(monkeypatch, held, *, error=None, keep_result=False):
    """Replace the compute step: it reports one step, waits at the gate, then raises, stops or returns nothing.

    ``keep_result`` returns the real result of a run (written file included) although Stop was pressed meanwhile,
    which is what a Stop that arrives during the write of the file does.
    """
    original = service.compute_cost_field

    def fake(*, cancel=None, progress=None, **kwargs):
        held.calls.append(kwargs)
        progress(0, 3, "point 1 of 2: settling the whole field (Stop acts between points)")
        held.reached.set()
        assert held.open.wait(JOIN_S), "the test never opened the gate"
        if error is not None:
            raise error
        if keep_result:
            return original(**kwargs)
        if cancel is not None and cancel.is_set():
            raise jobs.JobCancelled("stopped")
        return original(**kwargs)

    monkeypatch.setattr(service, "compute_cost_field", fake)
    return held


def _walk(component, parents=()):
    yield component, parents
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            if hasattr(child, "_traverse"):
                yield from _walk(child, (*parents, component))
    elif hasattr(children, "_traverse"):
        yield from _walk(children, (*parents, component))


def _text(component) -> str:
    """All plain text inside a component, and the hover texts of its wrappers."""
    pieces = []
    for part in component._traverse():  # noqa: SLF001
        child = getattr(part, "children", None)
        if isinstance(child, str):
            pieces.append(child)
    return " ".join(pieces)


@pytest.fixture()
def page(state):
    return build_layout(state)


def _by_id(page, component_id):
    found = [c for c in page._traverse() if getattr(c, "id", None) == component_id]  # noqa: SLF001
    assert len(found) == 1, f"{component_id}: {len(found)} components"
    return found[0]


def _wrapper_of(page, control):
    """The ``_help`` div that carries the hover text of a control."""
    return next(c for c, _p in _walk(page) if isinstance(c, html.Div) and getattr(c, "title", None)
                and isinstance(c.children, list)
                and any(hasattr(child, "_traverse") and _has_id(child, control) for child in c.children))


def _has_id(component, component_id) -> bool:
    """Is ``component`` or anything below it the one with this id? (``_traverse`` yields only the descendants.)"""
    return getattr(component, "id", None) == component_id or any(
        getattr(c, "id", None) == component_id for c in component._traverse())  # noqa: SLF001


# ================================================================== the layout
def test_the_card_is_collapsible_closed_and_sits_after_constrained_routing(page):
    card = _by_id(page, ids.CF_CARD)
    assert isinstance(card, html.Details) and card.open is False
    assert "gui-card-collapsible" in card.className
    assert _text(card.children[0]).strip() == "Cost field"
    tab = next(parent[-1] for c, parent in _walk(page) if c is card)
    siblings = [c for c in tab.children if hasattr(c, "_traverse")]
    constrained = next(i for i, c in enumerate(siblings) if _has_id(c, ids.OHL_COLLAPSE))
    assert siblings[constrained + 1] is card


def test_every_control_of_the_card_is_inside_it_exactly_once(page):
    card = _by_id(page, ids.CF_CARD)
    for component_id in (ids.CF_ORIGINS, ids.CF_COMBINE, ids.CF_ALGORITHM, ids.CF_THREADS, ids.CF_RUN_BTN,
                         ids.CF_STOP_BTN, ids.CF_STATUS, ids.CF_BEST_INFO, section.RUN_WRAP):
        assert _has_id(card, component_id), component_id
        _by_id(page, component_id)                      # exactly once in the whole page
    assert _by_id(page, ids.CF_POLL).disabled is True and _by_id(page, ids.CF_SETTINGS).data == {}


def test_the_option_lists_and_labels_come_from_the_service_constants(page):
    assert _by_id(page, ids.CF_ORIGINS).options == service.ORIGIN_OPTIONS
    assert _by_id(page, ids.CF_COMBINE).options == service.COMBINE_OPTIONS
    kernel = _by_id(page, ids.CF_ALGORITHM)
    assert isinstance(kernel, dbc.Select) and kernel.options == service.ALGORITHM_OPTIONS
    assert [o["value"] for o in service.ALGORITHM_OPTIONS] == ["auto", "dijkstra"]
    texts = _text(_by_id(page, ids.CF_CARD))
    for key in ("origins", "combine", "algorithm"):
        assert service.label_for(key) in texts


def test_a_fresh_card_shows_the_service_defaults(page):
    assert _by_id(page, ids.CF_ORIGINS).value == service.COST_FIELD_DEFAULTS["origins"]
    assert _by_id(page, ids.CF_COMBINE).value == service.COST_FIELD_DEFAULTS["combine"]
    assert _by_id(page, ids.CF_ALGORITHM).value == service.COST_FIELD_DEFAULTS["algorithm"]
    assert _by_id(page, ids.CF_THREADS).value == service.COST_FIELD_DEFAULTS["threads"]


def test_the_threads_input_states_its_limits_and_says_zero_is_automatic(page):
    threads = _by_id(page, ids.CF_THREADS)
    assert isinstance(threads, dbc.Input) and threads.type == "number"
    assert (threads.min, threads.max, threads.step) == (0, service.max_threads(), 1)
    label = next(c for c, _p in _walk(page) if isinstance(c, dbc.Label) and c.html_for == ids.CF_THREADS)
    assert label.children == "Threads (0 = automatic)"
    tip = _wrapper_of(page, ids.CF_THREADS).title
    assert f"from 0 to {service.max_threads()} threads" in tip and "0 = automatic" in tip


def test_the_stop_button_is_hidden_until_a_job_runs(page):
    stop = _by_id(page, ids.CF_STOP_BTN)
    assert stop.style == HIDDEN and stop.color == "danger" and "Stop" in stop.children and stop.title


def test_the_compute_button_starts_off_with_its_reason_on_the_wrapper(page):
    button = _by_id(page, ids.CF_RUN_BTN)
    wrapper = _by_id(page, section.RUN_WRAP)
    assert button.children == "Compute cost field" and button.disabled is True
    assert "Pick a cost raster" in wrapper.title


def test_the_help_text_explains_the_field_in_plain_words(page):
    card = _by_id(page, ids.CF_CARD)
    text = _text(card)
    assert "Sum over the points" in text and "Lowest of the points" in text
    assert "Cells that cannot be reached, and forbidden cells, are shown as no data" in text
    assert service.UNIT in text and service.UNIT_NOTE in text
    assert "plain cost raster" in text and "objective and terrain card is not applied" in text
    assert "Stop acts between fixed points" in text and "adds no layer" in text
    for backend in service.FIELD_BACKENDS:
        assert backend in text


# ================================================================== the Compute button follows the page
def _button(app, graph_api, raster_id, view=None):
    resp = invoke(app, BUTTON, graph_api, raster_id, view or [])
    return resp[section.RUN_WRAP]["title"], resp[ids.CF_RUN_BTN]["disabled"]


def test_the_button_is_enabled_with_a_field_backend_and_a_raster(app, layer):
    title, disabled = _button(app, "cython", layer.id)
    assert disabled is False and "Compute the cost field" in title


@pytest.mark.parametrize("backend", ["networkx", "igraph", "rustworkx", "networkit"])
def test_a_backend_that_cannot_compute_a_field_disables_the_button_and_says_why(app, layer, backend):
    title, disabled = _button(app, backend, layer.id)
    assert disabled is True
    assert "cannot compute a cost field" in title and "cython" in title


def test_no_raster_disables_the_button_and_says_what_to_do(app, state):
    title, disabled = _button(app, "cython", None)
    assert disabled is True and "Pick a cost raster under 'Route on raster'" in title
    title, disabled = _button(app, "cython", "no-such-layer")
    assert disabled is True and "Pick a cost raster" in title


def test_a_picked_layer_that_is_no_raster_does_not_enable_the_button(app, state):
    vector = state.add_layer("area", "vector")
    assert _button(app, "cython", vector.id)[1] is True


def test_a_blank_backend_counts_as_the_default(app, layer):
    assert _button(app, None, layer.id)[1] is False


# ================================================================== run -> job -> poll -> layer
def test_a_click_starts_the_job_and_the_poll_adds_the_layer_and_the_readout(app, state, layer):
    resp = _click(app, layer.id, TWO_SOURCES, clicks=7)
    assert resp[ids.CF_POLL]["disabled"] is False and resp[ids.CF_STOP_BTN]["style"] == {}
    assert resp[ids.CF_BEST_INFO]["children"] == ""
    assert resp[ids.CF_STATUS]["children"].startswith("Cost field")       # "...", or "finished" on a fast machine
    assert ids.LAYERS_VIEW not in resp and ids.MAP_VIEW not in resp
    job = _join(state)
    assert job.status == jobs.DONE and job.params["finalize"] == {"raster_layer_id": layer.id, "seq": 7}

    resp = _poll(app)
    assert resp[ids.CF_POLL]["disabled"] is True and resp[ids.CF_STOP_BTN]["style"] == HIDDEN
    assert state.jobs == {}
    (field,) = _cost_field_layers(state)
    assert field.name == "Cost field (sum of 2 point(s))"
    assert field.meta["cost_field"]["raster_layer_id"] == layer.id
    assert field.meta["cost_field"]["best_cost"] == pytest.approx(100.0)       # 10 cells x cost 10, row-wise
    assert resp[ids.LAYERS_VIEW]["data"] == _as_json(state.layers_view())
    assert resp[ids.MAP_VIEW]["data"] == _as_json({"fit_bounds": field.tile.bounds, "seq": 7})
    assert resp[ids.CF_STATUS]["children"] == service.summary_text(job.result)
    assert resp[ids.CF_BEST_INFO]["children"] == service.best_text(job.result)
    assert resp[ids.CF_BEST_INFO]["children"].startswith("Cheapest cell: (")
    assert "cost units" in resp[ids.CF_BEST_INFO]["children"]
    assert _titles(resp)[-1] == f"Cost field layer '{field.name}' added"
    assert resp[ids.NOTICES]["data"][-1]["severity"] == "success"

    with rasterio.open(job.result["path"]) as src:                              # the layer's file is the field
        values = src.read(1)
    assert values.shape == (SIZE, SIZE)
    assert values[values != service.NODATA].min() == pytest.approx(100.0)


def test_all_control_points_use_the_targets_and_waypoints_too(app, state, layer):
    _click(app, layer.id, ONE_SOURCE_EACH, origins="all")
    _join(state)
    _poll(app)
    (field,) = _cost_field_layers(state)
    assert len(field.meta["cost_field"]["origins"]) == 3
    assert field.name == "Cost field (sum of 3 point(s))"


def test_the_sources_setting_leaves_the_targets_out(app, state, layer):
    _click(app, layer.id, ONE_SOURCE_EACH, origins="sources", combine="min")
    _join(state)
    resp = _poll(app)
    (field,) = _cost_field_layers(state)
    assert len(field.meta["cost_field"]["origins"]) == 1 and field.meta["cost_field"]["combine"] == "min"
    assert "No cheapest cell is named" in resp[ids.CF_BEST_INFO]["children"]


def test_the_default_kernel_and_threads_run_too(app, state, layer):
    _click(app, layer.id, TWO_SOURCES, algorithm="auto", threads=1)
    job = _join(state)
    assert job.status == jobs.DONE, job.error
    _poll(app)
    assert len(_cost_field_layers(state)) == 1


def test_the_points_of_the_draft_are_used_as_stored_not_rebuilt_from_lat_lng(app, state, layer, monkeypatch):
    seen = []
    real = service.plan_run

    def spy(raster_path, raw, origins, **kwargs):
        seen.append(origins)
        return real(raster_path, raw, origins, **kwargs)

    monkeypatch.setattr(service, "plan_run", spy)
    _click(app, layer.id, TWO_SOURCES)
    _join(state)
    _poll(app)
    assert seen == [[(X0 + 2.5, Y0 + SIZE - 10.5), (X0 + 12.5, Y0 + SIZE - 10.5)]]


def test_the_routing_settings_and_the_study_area_reach_the_check_from_the_state(app, state, layer, monkeypatch):
    state.routing_settings = {**routing.ROUTING_SETTING_DEFAULTS, "neighborhood_k": 1, "search_buffer_m": 7,
                              "full_window": True, "ignore_max_cost": False}
    state.study_area = {"type": "Polygon", "coordinates": [[[8.0, 50.0], [8.1, 50.0], [8.1, 50.1], [8.0, 50.0]]]}
    seen = []

    def spy(raster_path, raw, origins, **kwargs):
        seen.append((raster_path, raw, origins, kwargs))
        return service.RunPlan(problems=[service.Problem("origins", "stop here")])

    monkeypatch.setattr(service, "plan_run", spy)
    _click(app, layer.id, TWO_SOURCES, origins="all", combine="min", algorithm="dijkstra", threads=2)
    raster_path, raw, _origins, kwargs = seen[0]
    assert raster_path == layer.meta["source_path"]
    assert raw == {"origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 2}
    assert (kwargs["graph_api"], kwargs["neighborhood"], kwargs["search_buffer_m"]) == ("cython", 1, 7)
    assert (kwargs["full_window"], kwargs["ignore_max_cost"]) == (True, False)
    assert kwargs["study_area"] == state.study_area
    assert str(kwargs["out_path"]).startswith(str(state.work_dir)) and str(kwargs["out_path"]).endswith(".tif")


def test_a_page_that_never_mirrored_its_routing_settings_gets_the_page_defaults(state):
    assert state.routing_settings == {}
    got = cf_cb.routing_inputs(state)
    assert got == {"graph_api": "cython", "neighborhood": routing.DEFAULT_NEIGHBORHOOD_K, "ignore_max_cost": True,
                   "search_buffer_m": None, "full_window": False}


# ------------------------------------------------------------------ the hook for the objective package
def test_the_objective_hook_returns_none_and_the_objective_settings_are_not_interpreted(app, state, layer,
                                                                                         monkeypatch):
    state.gui_settings["objective"] = {"weights": {"cost": 2.0, "length": 1.0}, "dem": "some-dem"}
    assert cf_cb._finder_options(state) is None                                    # noqa: SLF001
    seen = []

    def spy(*_args, **kwargs):
        seen.append(kwargs["finder_options"])
        return service.RunPlan(problems=[service.Problem("origins", "stop here")])

    monkeypatch.setattr(service, "plan_run", spy)
    _click(app, layer.id, TWO_SOURCES)
    assert seen == [None]


def test_what_the_hook_returns_is_what_the_check_gets(app, state, layer, monkeypatch):
    seen = []

    def spy(*_args, **kwargs):
        seen.append(kwargs["finder_options"])
        return service.RunPlan(problems=[service.Problem("origins", "stop here")])

    monkeypatch.setattr(service, "plan_run", spy)
    monkeypatch.setattr(cf_cb, "_finder_options", lambda _state: {"dem": "a-dem.tif"})
    _click(app, layer.id, TWO_SOURCES)
    assert seen == [{"dem": "a-dem.tif"}]


# ------------------------------------------------------------------ the check names every problem
def _refused(resp, state):
    """A refused click: a status line and a notice, no job, the poll and Stop untouched."""
    assert state.jobs == {}
    assert ids.CF_POLL not in resp and ids.CF_STOP_BTN not in resp
    assert ids.LAYERS_VIEW not in resp and ids.MAP_VIEW not in resp
    return _notices(resp)[-1]


def test_no_raster_picked_is_a_notice_that_focuses_the_dropdown(app, state):
    resp = _click(app, None, TWO_SOURCES)
    notice = _refused(resp, state)
    assert (notice["focus_id"], notice["focus_control"]) == (ids.TAB_ROUTES, ids.ROUTE_RASTER)
    assert notice["title"] == "Cost raster: value not usable"
    assert "no raster layer is picked" in resp[ids.CF_STATUS]["children"]


def test_a_picked_layer_that_is_no_raster_is_the_same_notice(app, state):
    resp = _click(app, state.add_layer("area", "vector").id, TWO_SOURCES)
    assert _refused(resp, state)["focus_control"] == ids.ROUTE_RASTER


def test_no_fixed_point_is_a_notice_that_focuses_the_fixed_points_field(app, state, layer):
    resp = _click(app, layer.id, {"sources": [], "targets": [_pt(3, 3)], "waypoints": []})
    notice = _refused(resp, state)
    assert (notice["focus_id"], notice["focus_control"]) == (ids.TAB_ROUTES, ids.CF_ORIGINS)
    assert "none given" in resp[ids.CF_STATUS]["children"]
    assert resp[ids.CF_STATUS]["children"] == notice["meaning"]


def test_a_point_outside_the_raster_is_named(app, state, layer):
    outside = {"sources": [_pt(3, 3), {"x": X0 - 500.0, "y": Y0, "lat": 0, "lng": 0}], "targets": [],
               "waypoints": []}
    resp = _click(app, layer.id, outside)
    notice = _refused(resp, state)
    assert notice["focus_control"] == ids.CF_ORIGINS and "outside the raster" in resp[ids.CF_STATUS]["children"]


def test_a_point_on_a_forbidden_cell_is_named(app, state, forbidden_path):
    resp = _click(app, _raster_layer(state, forbidden_path).id,
                  {"sources": [_pt(5, 5), _pt(10, 10)], "targets": [], "waypoints": []})
    notice = _refused(resp, state)
    assert notice["focus_control"] == ids.CF_ORIGINS and "forbidden cell" in resp[ids.CF_STATUS]["children"]


def test_with_the_barrier_switch_off_a_point_on_a_forbidden_cell_is_allowed(app, state, forbidden_path):
    state.routing_settings = {**routing.ROUTING_SETTING_DEFAULTS, "ignore_max_cost": False}
    _click(app, _raster_layer(state, forbidden_path).id,
           {"sources": [_pt(5, 5), _pt(10, 10)], "targets": [], "waypoints": []})
    job = _join(state)
    assert job.status == jobs.DONE, job.error
    jobs.take_finished(state, KIND)


@pytest.mark.parametrize("override,control,fragment", [
    ({"combine": "max"}, ids.CF_COMBINE, "Combine"),
    ({"algorithm": "fast"}, ids.CF_ALGORITHM, "Kernel"),
    ({"threads": 10 ** 6}, ids.CF_THREADS, "outside the range"),
    ({"threads": 1.5}, ids.CF_THREADS, "not a whole number"),
    ({"threads": -1}, ids.CF_THREADS, "outside the range"),
    ({"origins": "nothing"}, ids.CF_ORIGINS, "Fixed points")])
def test_a_card_setting_that_cannot_be_used_is_a_field_notice(app, state, layer, override, control, fragment):
    resp = _click(app, layer.id, TWO_SOURCES, **override)
    notice = _refused(resp, state)
    assert notice["focus_control"] == control and notice["focus_id"] == ids.TAB_ROUTES
    assert fragment in notice["title"] + notice["meaning"]
    assert resp[ids.CF_STATUS]["children"].splitlines()[0] == notice["meaning"]


@pytest.mark.parametrize("settings,control,fragment", [
    ({"graph_api": "networkx"}, ids.GRAPH_API, "cannot compute a cost field"),
    ({"neighborhood_k": 500}, ids.NEIGHBORHOOD, "Neighbourhood k"),
    ({"neighborhood_k": -1}, ids.NEIGHBORHOOD, "Neighbourhood k"),
    ({"search_buffer_m": -5}, ids.SEARCH_BUFFER, "Search buffer")])
def test_a_routing_setting_that_cannot_be_used_points_at_the_routing_control(app, state, layer, settings, control,
                                                                              fragment):
    state.routing_settings = {**routing.ROUTING_SETTING_DEFAULTS, **settings}
    resp = _click(app, layer.id, TWO_SOURCES)
    notice = _refused(resp, state)
    assert (notice["focus_id"], notice["focus_control"]) == (ids.TAB_ROUTES, control)
    assert fragment in notice["meaning"] and fragment in resp[ids.CF_STATUS]["children"]


def test_a_raster_with_the_wrong_cell_type_sends_the_user_to_the_raster_tab(app, state, float_path):
    resp = _click(app, _raster_layer(state, float_path).id, TWO_SOURCES)
    notice = _refused(resp, state)
    assert (notice["focus_id"], notice["focus_control"]) == (ids.TAB_RASTER, None)
    assert "float32" in resp[ids.CF_STATUS]["children"] and "uint16" in resp[ids.CF_STATUS]["children"]


def test_a_raster_file_that_is_gone_sends_the_user_to_the_raster_tab(app, state, tmp_path):
    missing = _raster_layer(state, str(tmp_path / "gone.tif"))
    resp = _click(app, missing.id, TWO_SOURCES)
    notice = _refused(resp, state)
    assert (notice["focus_id"], notice["focus_control"]) == (ids.TAB_RASTER, None)
    assert "no longer exists" in resp[ids.CF_STATUS]["children"]


def test_a_run_that_would_not_fit_into_memory_is_an_error_that_points_at_the_search_buffer(app, state, layer,
                                                                                            monkeypatch):
    monkeypatch.setattr(service, "MEMORY_LIMIT_MB", 0.0001)
    resp = _click(app, layer.id, TWO_SOURCES)
    notice = _refused(resp, state)
    assert notice["severity"] == "error" and notice["title"] == "The cost field would not fit into memory"
    assert notice["focus_control"] == ids.SEARCH_BUFFER and resp[ids.CF_STATUS]["children"].startswith("Memory")


def test_more_fixed_points_than_the_limit_is_a_notice(app, state, layer, monkeypatch):
    monkeypatch.setattr(service, "MAX_ORIGINS", 1)
    resp = _click(app, layer.id, TWO_SOURCES)
    notice = _refused(resp, state)
    assert notice["focus_control"] == ids.CF_ORIGINS and "Allowed: 1 to 1 points" in notice["meaning"]


def test_several_problems_show_the_first_three_in_the_status_and_the_rest_in_the_details(app, state, layer):
    state.routing_settings = {**routing.ROUTING_SETTING_DEFAULTS, "graph_api": "networkx", "neighborhood_k": 500,
                              "search_buffer_m": -5}
    resp = _click(app, layer.id, {"sources": [], "targets": [], "waypoints": []}, combine="max", threads=-4)
    notice = _refused(resp, state)
    assert len(resp[ids.CF_STATUS]["children"].splitlines()) == 4              # three messages and "... and N more."
    assert resp[ids.CF_STATUS]["children"].splitlines()[-1].startswith("... and ")
    assert notice["details"]


def test_problems_text_does_not_cut_a_short_list():
    problems = [("origins", "one"), ("combine", "two")]
    assert cf_cb.problems_text(problems) == "one\ntwo"
    longer = problems + [("threads", "three"), ("raster", "four")]
    assert cf_cb.problems_text(longer) == "one\ntwo\nthree\n... and 1 more."


def test_a_blank_threads_field_means_automatic_and_runs(app, state, layer):
    resp = _click(app, layer.id, TWO_SOURCES, threads=None)
    assert resp[ids.CF_POLL]["disabled"] is False
    assert _join(state).status == jobs.DONE
    jobs.take_finished(state, KIND)


def test_a_soft_memory_warning_is_shown_when_the_job_starts(app, state, layer, monkeypatch):
    monkeypatch.setattr(service, "MEMORY_WARN_MB", 0.0001)
    resp = _click(app, layer.id, TWO_SOURCES, notices=[{"title": "earlier", "severity": "info"}])
    assert resp[ids.CF_POLL]["disabled"] is False
    assert _titles(resp) == ["earlier", "This cost field needs a lot of memory"]
    _join(state)
    jobs.take_finished(state, KIND)


# ================================================================== the busy rules
def test_a_second_click_while_the_job_runs_gets_the_job_services_refusal(app, state, layer, gate, monkeypatch):
    _fake_compute(monkeypatch, gate)
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    job = state.jobs[KIND]
    resp = _click(app, layer.id, TWO_SOURCES)
    notice = _notices(resp)[-1]
    assert {k: v for k, v in notice.items() if k != "id"} == {
        k: v for k, v in jobs.busy_notice("Cost field").items() if k != "id"}
    assert notice["title"] == "Cost field is still running"
    assert state.jobs[KIND] is job and len(gate.calls) == 1                    # no second run
    assert ids.CF_POLL not in resp and ids.CF_STATUS not in resp               # the running job's page is untouched
    gate.open.set()
    _join(state)
    _poll(app)


def test_a_click_while_another_job_runs_names_that_job(app, state, layer, gate):
    release = threading.Event()

    def hold(job):
        gate.reached.set()
        release.wait(JOIN_S)
        return None

    other = jobs.start_job(state, "rasterize", hold, label="Rasterizing")
    try:
        assert gate.reached.wait(JOIN_S)
        resp = _click(app, layer.id, TWO_SOURCES)
        assert _notices(resp)[-1]["title"] == "Rasterizing is still running"
        assert KIND not in state.jobs
    finally:
        release.set()
        other.thread.join(JOIN_S)
        jobs.take_finished(state, "rasterize")


def test_a_start_that_loses_the_race_shows_the_refusal(app, state, layer, monkeypatch):
    monkeypatch.setattr(jobs, "start_job", lambda *_a, **_k: None)
    resp = _click(app, layer.id, TWO_SOURCES)
    assert _notices(resp)[-1]["title"] == "Cost field is still running" and KIND not in state.jobs
    assert ids.CF_POLL not in resp


def test_an_ended_job_nobody_collected_is_collected_by_the_next_click(app, state, layer):
    _click(app, layer.id, TWO_SOURCES)
    job = _join(state)                                      # ended, and no poll took it
    resp = _click(app, layer.id, TWO_SOURCES, clicks=2)
    assert "An earlier cost field run was collected" in _titles(resp)
    # round 15: the click switches the poll on and the poll hands the outcome over (no second run)
    assert resp[ids.CF_POLL]["disabled"] is False and resp[ids.CF_STOP_BTN]["style"] == HIDDEN
    assert state.jobs == {KIND: job} and _cost_field_layers(state) == []
    resp = _poll(app, resp[ids.NOTICES]["data"])
    assert resp[ids.CF_POLL]["disabled"] is True and resp[ids.CF_STOP_BTN]["style"] == HIDDEN
    assert resp[ids.CF_STATUS]["children"] == service.summary_text(job.result)
    assert resp[ids.CF_BEST_INFO]["children"] == service.best_text(job.result)
    assert state.jobs == {} and len(_cost_field_layers(state)) == 1            # collected, and no second run
    resp = _click(app, layer.id, TWO_SOURCES, clicks=3)                         # now a new run can start
    assert resp[ids.CF_POLL]["disabled"] is False
    _join(state)
    jobs.take_finished(state, KIND)


def test_a_click_without_a_press_does_nothing(app, layer):
    with pytest.raises(PreventUpdate):
        invoke(app, RUN, None, layer.id, TWO_SOURCES, "sources", "sum", "auto", 0, [],
               triggered=[f"{ids.CF_RUN_BTN}.n_clicks"])


# ================================================================== progress, Stop, failure
def test_progress_shows_while_the_job_runs_and_changes_nothing_else(app, state, layer, gate, monkeypatch):
    _fake_compute(monkeypatch, gate)
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    resp = _poll(app)
    assert "point 1 of 2" in resp[ids.CF_STATUS]["children"] and "Stop acts between points" in resp[
        ids.CF_STATUS]["children"]
    for untouched in (ids.LAYERS_VIEW, ids.MAP_VIEW, ids.CF_BEST_INFO, ids.NOTICES, ids.CF_POLL, ids.CF_STOP_BTN):
        assert untouched not in resp, untouched
    gate.open.set()
    _join(state)
    _poll(app)


def test_stop_asks_for_a_stop_and_says_it_acts_between_fixed_points(app, state, layer, gate, monkeypatch):
    _fake_compute(monkeypatch, gate)
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    text = _stop(app)[ids.CF_STATUS]["children"]
    assert text.startswith("Stop requested") and "Stop acts between fixed points" in text
    assert "adds no layer" in text
    assert state.jobs[KIND].cancel.is_set()
    assert "stop requested" in _poll(app)[ids.CF_STATUS]["children"]
    gate.open.set()
    _join(state)
    _poll(app)


def test_a_stopped_job_adds_no_layer_and_changes_nothing(app, state, layer, gate, monkeypatch):
    _fake_compute(monkeypatch, gate)
    before = state.layers_view()
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    _stop(app)
    gate.open.set()
    job = _join(state)
    assert job.status == jobs.CANCELLED
    resp = _poll(app)
    assert _titles(resp) == ["Cost field stopped"] and resp[ids.NOTICES]["data"][-1]["severity"] == "info"
    assert resp[ids.CF_POLL]["disabled"] is True and resp[ids.CF_STOP_BTN]["style"] == HIDDEN
    assert "No cost field was computed" in resp[ids.CF_STATUS]["children"]
    assert ids.LAYERS_VIEW not in resp and ids.MAP_VIEW not in resp and ids.CF_BEST_INFO not in resp
    assert state.layers_view() == before and _cost_field_layers(state) == [] and state.jobs == {}


def test_a_stop_that_came_while_the_file_was_written_still_adds_nothing_and_leaves_no_file(app, state, layer, gate,
                                                                                            monkeypatch):
    _fake_compute(monkeypatch, gate, keep_result=True)
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    _stop(app)
    gate.open.set()
    job = _join(state)
    assert job.status == jobs.CANCELLED and job.result["path"]
    written = job.result["path"]
    resp = _poll(app)
    assert _titles(resp) == ["Cost field stopped"] and _cost_field_layers(state) == []
    import os

    assert not os.path.exists(written)


def test_stop_with_nothing_running_says_so(app, state):
    assert _stop(app)[ids.CF_STATUS]["children"] == "Nothing is running to stop."


def test_stop_without_a_press_does_nothing(app):
    with pytest.raises(PreventUpdate):
        invoke(app, STOP, None, triggered=[f"{ids.CF_STOP_BTN}.n_clicks"])


def test_a_refusal_inside_the_job_reaches_the_poll_as_a_field_notice(app, state, layer, gate, monkeypatch):
    refusal = service.CostFieldRefused([service.Problem("origins", "Fixed points (origins): point 2 moved.")])
    _fake_compute(monkeypatch, gate, error=refusal)
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    gate.open.set()
    assert _join(state).status == jobs.FAILED
    resp = _poll(app)
    notice = _notices(resp)[-1]
    assert notice["title"] == "Fixed points: value not usable"
    assert resp[ids.CF_STATUS]["children"].startswith("Cost field failed after")
    assert resp[ids.CF_POLL]["disabled"] is True and resp[ids.CF_STOP_BTN]["style"] == HIDDEN
    assert ids.LAYERS_VIEW not in resp and state.jobs == {} and _cost_field_layers(state) == []


def test_a_crash_inside_the_job_is_an_error_notice_and_the_next_run_can_start(app, state, layer, gate, monkeypatch):
    _fake_compute(monkeypatch, gate, error=RuntimeError("the kernel fell over"))
    _click(app, layer.id, TWO_SOURCES)
    assert gate.reached.wait(JOIN_S)
    gate.open.set()
    _join(state)
    resp = _poll(app)
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and "the kernel fell over" in (notice["details"] or "")
    assert state.jobs == {}
    monkeypatch.undo()
    resp = _click(app, layer.id, TWO_SOURCES, clicks=2)
    assert resp[ids.CF_POLL]["disabled"] is False
    _join(state)
    jobs.take_finished(state, KIND)


def test_a_file_that_cannot_be_served_is_an_error_notice_and_no_layer(app, state, layer, monkeypatch):
    from pyorps.gui.services import tiles

    def boom(*_a, **_k):
        raise ValueError("no tile server today")

    monkeypatch.setattr(tiles, "build_tile_layer", boom)
    _click(app, layer.id, TWO_SOURCES)
    _join(state)
    resp = _poll(app)
    assert resp[ids.NOTICES]["data"][-1]["severity"] == "error"
    assert "could not be shown" in resp[ids.CF_STATUS]["children"]
    assert ids.LAYERS_VIEW not in resp and _cost_field_layers(state) == []
    assert resp[ids.CF_POLL]["disabled"] is True


def test_a_tick_without_a_job_switches_the_poll_off_and_hides_stop(app, state):
    resp = _poll(app)
    assert resp == {ids.CF_POLL: {"disabled": True}, ids.CF_STOP_BTN: {"style": HIDDEN}}


# ================================================================== the page and the server hold the same settings
def test_the_mirror_keeps_the_four_settings_on_the_server_and_in_the_store(app, state):
    resp = invoke(app, MIRROR, "all", "min", "dijkstra", 3)
    expected = {"origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 3}
    assert resp[ids.CF_SETTINGS]["data"] == expected and state.gui_settings["cost_field"] == expected


@pytest.mark.parametrize("threads", [None, 10 ** 6, -2, 1.5, "x"])
def test_the_mirror_falls_back_to_the_default_for_an_unusable_value(app, state, threads):
    resp = invoke(app, MIRROR, "sources", "sum", "auto", threads)
    assert resp[ids.CF_SETTINGS]["data"]["threads"] == 0 and state.gui_settings["cost_field"]["threads"] == 0


def test_the_mirror_is_json_safe_and_survives_a_project_round_trip(app, state, tmp_path):
    invoke(app, MIRROR, "all", "min", "dijkstra", 2)
    saved = project_io.save_project(state, tmp_path / "proj")
    assert json.loads(open(saved, encoding="utf-8").read())["gui_settings"]["cost_field"] == {
        "origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 2}
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, saved)
        page = build_layout(other)
        assert (_by_id(page, ids.CF_ORIGINS).value, _by_id(page, ids.CF_COMBINE).value,
                _by_id(page, ids.CF_ALGORITHM).value, _by_id(page, ids.CF_THREADS).value) == (
            "all", "min", "dijkstra", 2)
        assert _by_id(page, ids.CF_SETTINGS).data == {"origins": "all", "combine": "min", "algorithm": "dijkstra",
                                                      "threads": 2}
    finally:
        other.shutdown()


def test_a_blank_project_prefills_nothing(state):
    assert {cid for cid in _state_prefill(state)} & {ids.CF_ORIGINS, ids.CF_COMBINE, ids.CF_ALGORITHM,
                                                     ids.CF_THREADS, ids.CF_SETTINGS} == set()


def test_the_saved_settings_fill_the_controls_on_a_page_load(state):
    state.gui_settings["cost_field"] = {"origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 1}
    values = _state_prefill(state)
    assert values[ids.CF_ORIGINS] == {"value": "all"} and values[ids.CF_COMBINE] == {"value": "min"}
    assert values[ids.CF_ALGORITHM] == {"value": "dijkstra"} and values[ids.CF_THREADS] == {"value": 1}
    assert values[ids.CF_SETTINGS] == {"data": state.gui_settings["cost_field"]}


def test_a_saved_value_that_is_no_longer_usable_prefills_its_default(state):
    state.gui_settings["cost_field"] = {"origins": "all", "combine": "nonsense", "algorithm": 7, "threads": 10 ** 6}
    page = build_layout(state)
    assert _by_id(page, ids.CF_ORIGINS).value == "all"
    assert _by_id(page, ids.CF_COMBINE).value == service.COST_FIELD_DEFAULTS["combine"]
    assert _by_id(page, ids.CF_ALGORITHM).value == service.COST_FIELD_DEFAULTS["algorithm"]
    assert _by_id(page, ids.CF_THREADS).value == service.COST_FIELD_DEFAULTS["threads"]


def _restore(app, status):
    return invoke(app, RESTORE, status, triggered=[f"{ids.PROJECT_STATUS}.children"])


def test_open_project_puts_the_saved_settings_back_and_clears_the_readouts(app, state):
    state.gui_settings["cost_field"] = {"origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 2}
    resp = _restore(app, "opened: C:/somewhere/project.json")
    assert (resp[ids.CF_ORIGINS]["value"], resp[ids.CF_COMBINE]["value"], resp[ids.CF_ALGORITHM]["value"],
            resp[ids.CF_THREADS]["value"]) == ("all", "min", "dijkstra", 2)
    assert resp[ids.CF_STATUS]["children"] == "" and resp[ids.CF_BEST_INFO]["children"] == ""


def test_new_project_and_an_old_project_put_the_defaults_back(app, state):
    resp = _restore(app, "new project")
    assert (resp[ids.CF_ORIGINS]["value"], resp[ids.CF_COMBINE]["value"], resp[ids.CF_ALGORITHM]["value"],
            resp[ids.CF_THREADS]["value"]) == tuple(service.COST_FIELD_DEFAULTS.values())


def test_a_save_changes_nothing_on_the_page(app):
    with pytest.raises(PreventUpdate):
        _restore(app, "saved: C:/somewhere/project.json")


# ================================================================== the module's own pieces
def test_the_job_kind_is_the_one_the_page_knows():
    from pyorps.gui.layout import JOB_CONTROLS

    assert cf_cb.JOB_KIND == KIND and JOB_CONTROLS[KIND] == (ids.CF_POLL, ids.CF_STOP_BTN, ids.CF_STATUS)


def test_the_setting_ids_follow_the_service_keys():
    assert tuple(section.SETTING_IDS) == service.COST_FIELD_KEYS
    assert [t[0] for t in section.FORM_TARGETS] == [section.SETTING_IDS[k] for k in service.COST_FIELD_KEYS]
    assert section.CONTROLS["graph_api"] == ids.GRAPH_API and section.CONTROLS["raster"] == ids.ROUTE_RASTER


def test_every_problem_key_of_the_service_has_a_control():
    keys = set(service.FIELD_LABELS)
    assert keys <= set(section.CONTROLS), keys - set(section.CONTROLS)


def test_form_values_of_nothing_are_the_defaults_and_of_a_dict_are_normalised():
    assert section.form_values(None) == tuple(service.COST_FIELD_DEFAULTS.values())
    assert section.form_values({"origins": "all", "threads": 2, "combine": "bad"}) == ("all", "sum", "auto", 2)


def test_the_page_shows_a_running_job_after_a_reload(state):
    job = jobs.BackgroundJob(kind=KIND, label="Cost field")
    state.jobs[KIND] = job
    values = _state_prefill(state)
    assert values[ids.CF_POLL] == {"disabled": False} and values[ids.CF_STOP_BTN] == {"style": {}}
    assert values[ids.CF_STATUS]["children"].startswith("Cost field...")


def test_the_card_has_one_number_input_and_it_has_a_label(page):
    card = _by_id(page, ids.CF_CARD)
    numbers = [c for c, _p in _walk(card) if isinstance(c, dbc.Input) and getattr(c, "type", None) == "number"]
    assert [c.id for c in numbers] == [ids.CF_THREADS]
    assert any(isinstance(c, dbc.Label) and c.html_for == ids.CF_THREADS for c, _p in _walk(card))

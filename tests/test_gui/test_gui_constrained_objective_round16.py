"""Round 16, R3-B: the objective weights of constrained (overhead-line) routing in the GUI.

Headless, tiny synthetic rasters, the cython constrained backend. Four layers are tested:

* the pure service ``services/constrained_objective`` (specs, normalising, problems, what reaches the library),
  and that the library itself accepts and refuses what the service says;
* the run (``callbacks/constrained.run_constrained``): the weights come from the project state, defaults pass
  nothing to the library, a weight that is not usable or a refused combination starts nothing and focuses the
  field, the synchronous and the background path hand the library the same objective, and the weights are stored
  with the route;
* the block in the profile form (layout, mirror, status line, disabled states, reset, restore, first paint);
* the project file (save and open).

A run is made "heavy" by patching the named criterion ``constrained.is_heavy`` (never by a large raster).
"""
from __future__ import annotations

import json
import math

import dash_bootstrap_components as dbc
import numpy as np
import pytest
import yaml
from conftest import invoke, poll_job
from dash import dcc, html
from dash.exceptions import PreventUpdate
from rasterio.transform import from_bounds, from_origin

from pyorps.core.exceptions import InvalidInputError, ObjectiveError, PyorpsError
from pyorps.core.objective import RESERVED_LAYERS
from pyorps.gui import ids
from pyorps.gui.layout import build_layout
from pyorps.gui.sections import profile_form as form_section
from pyorps.gui.services import constrained, project_io
from pyorps.gui.services import constrained_objective as co
from pyorps.gui.services import profile_form as profile_service
from pyorps.gui.state import ProjectState

CRS = "EPSG:32632"
JOIN_S = 30
RUN = (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
SMOOTH, TOWER = ids.OHLO_IDS["smoothness"], ids.OHLO_IDS["tower"]
MIRROR = (f"{ids.OHLO_SETTINGS}.data",)
STATUS = (f"{ids.OHLO_STATUS}.children",)
RESET = (f"{SMOOTH}.value", ids.OHLO_RESET_BTN)
RESTORE = (f"{SMOOTH}.value", ids.PROJECT_STATUS)
PROFILE_MIRROR = (f"{ids.OHL_SETTINGS}.data",)

MINI_PROFILE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 20, "max_span_m": 40, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 200}, "terrain_interpolation": "linear",
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 100},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300}}}}
CLEARANCE = {"tower_height_m": 30.0, "conductor_weight_per_m": 15.0, "conductor_tension_n": 25000.0,
             "min_clearance_m": 6.0}
POINTS = [(10.0, 10.0), (90.0, 90.0)]
CHANGED = {"smoothness": 0.5, "tower": 2.0}


# ------------------------------------------------------------------ helpers
def _write_raster(path, array):
    import rasterio

    with rasterio.open(str(path), "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype="uint16", crs=CRS,
                       transform=from_bounds(0, 0, array.shape[1], array.shape[0], array.shape[1],
                                             array.shape[0])) as dst:
        dst.write(array, 1)
    return str(path)


@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory):
    """A 100 x 100 m uniform raster at 1 m (a run takes about a tenth of a second)."""
    return _write_raster(tmp_path_factory.mktemp("r16_tiny") / "uniform.tif", np.full((100, 100), 100, np.uint16))


@pytest.fixture()
def ohl_layer(state, tiny_raster):
    return state.add_layer("cost", "raster", crs=CRS, meta={"source_path": tiny_raster})


def _point(x, y):
    return {"lat": 0, "lng": 0, "x": x, "y": y}


DRAFT = {"sources": [_point(*POINTS[0])], "targets": [_point(*POINTS[1])], "waypoints": []}


def _run(app, raster_id, *, profile=None, backend="cython", experimental=False, dem=None, notices=None, k=1):
    text = yaml.safe_dump(MINI_PROFILE if profile is None else profile)
    return invoke(app, RUN, 1, True, DRAFT, raster_id, text, backend, experimental, k, 100, dem, None, [],
                  list(notices or []), triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _weights(state, **weights):
    """The project state as the block's mirror leaves it when the user typed ``weights``."""
    state.gui_settings["constrained"] = {"backend": "cython", "experimental": False,
                                         "objective": {**co.DEFAULTS, **weights}}


def _routes(state):
    return state.layers_of_kind("route")


def _towers(state):
    return [ly for ly in state.layers_of_kind("vector") if ly.meta.get("towers_for")]


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _titles(resp) -> list[str]:
    return [n["title"] for n in _notices(resp)]


@pytest.fixture()
def library_calls(monkeypatch):
    """The keyword arguments of every ``ConstrainedPathFinder`` the GUI builds (the real class still runs)."""
    import pyorps.graph.constrained_path_finder as library

    calls: list[dict] = []
    real = library.ConstrainedPathFinder

    class Spy(real):
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(library, "ConstrainedPathFinder", Spy)
    return calls


@pytest.fixture()
def heavy(monkeypatch):
    """Every constrained run counts as heavy (the criterion, patched by name)."""
    monkeypatch.setattr(constrained, "is_heavy", lambda _est: True)


def _components(page):
    return list(page._traverse())  # noqa: SLF001


def _by_id(page, component_id):
    found = [c for c in _components(page) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id}: {len(found)} components"
    return found[0]


@pytest.fixture()
def page(state):
    return build_layout(state)


# ====================================================================== the spec table
def test_the_weights_are_reserved_names_of_the_library_objective():
    assert co.WEIGHT_KEYS == ("smoothness", "tower") == ids.OHLO_KEYS
    assert set(co.WEIGHT_KEYS) <= set(RESERVED_LAYERS)
    assert tuple(co.FIELD_SPECS) == co.WEIGHT_KEYS and tuple(co.DEFAULTS) == co.WEIGHT_KEYS


@pytest.mark.parametrize("key", co.WEIGHT_KEYS)
def test_each_spec_has_a_label_a_unit_a_range_a_default_and_a_help_text(key):
    spec = co.FIELD_SPECS[key]
    assert spec.key == key and spec.label and spec.unit and spec.help
    assert (spec.minimum, spec.maximum, spec.allow_equal, spec.default) == (0.0, 1000.0, True, 1.0)
    assert co.describe_range(spec) == f">= 0 and <= 1,000 ({spec.unit})"
    assert co.label_with_unit(spec) == f"{spec.label} ({spec.unit})"
    assert co.LABELS[key] == f"{spec.label} ({key})"


def test_the_problem_keys_are_the_weights_and_the_two_combinations():
    assert co.PROBLEM_KEYS == ("smoothness", "tower", "backend", "clearance")


# ====================================================================== normalising
@pytest.mark.parametrize("raw", [None, {}, [], "text", 3, {"other": 5}, {"smoothness": None, "tower": ""}])
def test_nothing_or_blank_gives_the_defaults(raw):
    assert co.normalise(raw) == {"smoothness": 1.0, "tower": 1.0}
    assert co.is_default(raw) and co.library_objective(raw) is None and co.used_weights(raw) == {}


def test_numbers_and_numeric_text_become_floats_and_other_text_is_kept_for_the_problem():
    assert co.normalise({"smoothness": 2, "tower": " 0.5 "}) == {"smoothness": 2.0, "tower": 0.5}
    assert co.normalise({"smoothness": "abc"})["smoothness"] == "abc"
    assert co.normalise({"smoothness": True})["smoothness"] is True
    assert co.normalise({"smoothness": [1]})["smoothness"] == "[1]"
    assert math.isnan(co.normalise({"smoothness": float("nan")})["smoothness"])


def test_normalising_is_idempotent_json_safe_and_drops_unknown_keys():
    once = co.normalise({"smoothness": "0.25", "tower": 3, "gradient": 9, "cost": 4})
    assert once == {"smoothness": 0.25, "tower": 3.0}
    assert co.normalise(once) == once and json.loads(json.dumps(once)) == once


def test_input_values_show_a_number_as_typed_and_anything_else_blank():
    assert co.input_values({"smoothness": 5000, "tower": "abc"}) == {"smoothness": 5000.0, "tower": None}
    assert co.input_values(None) == {"smoothness": 1.0, "tower": 1.0}


# ====================================================================== problems: values
@pytest.mark.parametrize("key", co.WEIGHT_KEYS)
@pytest.mark.parametrize(("value", "word"), [(-0.001, "too low"), (-5, "too low"), (1000.001, "too high"),
                                             (1e6, "too high")])
def test_a_weight_outside_its_range_is_named_with_value_range_and_unit(key, value, word):
    found = co.problems({key: value})
    assert [k for k, _ in found] == [key]
    spec, message = co.FIELD_SPECS[key], found[0][1]
    assert spec.label in message and f"({key})" in message and word in message
    assert co.describe_range(spec) in message and spec.unit in message
    assert co.format_number(value) in message


@pytest.mark.parametrize("key", co.WEIGHT_KEYS)
@pytest.mark.parametrize("value", [0, 0.0, 1000, 1000.0, 1, 0.001, 999.999])
def test_the_limits_themselves_are_allowed(key, value):
    assert co.problems({key: value}) == []


@pytest.mark.parametrize("key", co.WEIGHT_KEYS)
@pytest.mark.parametrize(("value", "fragment"), [("abc", "is not a number"), (float("nan"), "not a finite number"),
                                                 (float("inf"), "not a finite number"), (True, "not a number"),
                                                 ([1], "not a number")])
def test_a_value_that_is_no_number_is_named(key, value, fragment):
    found = co.problems({key: value})
    assert [k for k, _ in found] == [key] and fragment in found[0][1]
    assert "Allowed:" in found[0][1]


def test_both_weights_can_be_wrong_at_once_in_form_order():
    found = co.problems({"smoothness": -1, "tower": 5000}, backend="cython_parallel", clearance_on=True)
    assert [k for k, _ in found] == ["smoothness", "tower"]          # the combination rules need usable numbers


# ====================================================================== problems: combinations
def test_defaults_are_never_refused_whatever_the_backend_or_the_profile():
    for backend in ("cython", "cython_parallel", "raster_gpu", "raster_gpu_v4"):
        assert co.problems({}, backend=backend, clearance_on=True) == []


@pytest.mark.parametrize("backend", ["cython_parallel", "raster_gpu", "raster_gpu_v4"])
def test_a_changed_weight_on_another_backend_is_a_backend_problem(backend):
    found = co.problems({"tower": 2.0}, backend=backend)
    assert [k for k, _ in found] == ["backend"]
    message = found[0][1]
    assert backend in message and "tower cost weight 2" in message and "cython" in message
    assert "Reset weights" in message


def test_a_changed_weight_with_the_clearance_check_is_a_clearance_problem():
    found = co.problems({"smoothness": 0.0}, clearance_on=True)
    assert [k for k, _ in found] == ["clearance"]
    message = found[0][1]
    assert "turn penalty weight 0" in message and "clearance" in message.lower() and "Reset weights" in message


def test_both_combinations_are_reported_together():
    assert [k for k, _ in co.problems({"tower": 3}, backend="raster_gpu", clearance_on=True)] == [
        "backend", "clearance"]


def test_the_backend_default_is_cython_and_a_blank_backend_counts_as_cython():
    assert co.problems({"tower": 3}) == [] == co.problems({"tower": 3}, backend=None)
    assert co.support(backend=None) == co.Support(True, "")


def test_a_terrain_model_never_makes_a_problem_only_a_note():
    assert co.problems({"tower": 3}) == [] == co.problems({"tower": 3}, has_dem=True)
    assert any("terrain model" in note for note in co.notes({"tower": 3}, has_dem=True))
    assert co.notes({}, has_dem=True) == []                        # nothing to say about weights at their default


def test_the_tower_weight_note_names_a_profile_without_spans():
    assert any("no towers" in note for note in co.notes({"tower": 3}, spans_on=False))
    assert co.notes({"tower": 3}, spans_on=True) == [] == co.notes({"tower": 3}, spans_on=None)
    assert co.notes({"smoothness": 3}, spans_on=False) == []       # the turn weight still works without towers


# ====================================================================== support and view
def test_support_says_why_the_weights_cannot_apply():
    assert co.support(backend="cython", clearance_on=False) == co.Support(True, "")
    off = co.support(backend="raster_gpu")
    assert not off.enabled and "raster_gpu" in off.reason and "cython" in off.reason
    cleared = co.support(clearance_on=True)
    assert not cleared.enabled and cleared.reason == co.CLEARANCE_REASON
    assert co.support(backend="raster_gpu", clearance_on=True).reason == off.reason     # the backend first


def test_the_view_is_the_summary_or_the_first_problems_and_carries_the_disabled_state():
    plain = co.view({})
    assert plain == co.View(co.summary({}), False, True, "")
    changed = co.view(CHANGED, has_dem=True, spans_on=False)
    assert changed.enabled and not changed.problem
    assert co.summary(CHANGED) in changed.text and "terrain model" in changed.text and "no towers" in changed.text
    refused = co.view(CHANGED, backend="raster_gpu")
    assert refused.problem and not refused.enabled and "raster_gpu" in refused.text and refused.reason
    idle = co.view({}, backend="raster_gpu")                        # defaults on a backend that cannot take them
    assert not idle.problem and not idle.enabled and idle.text == idle.reason
    wrong = co.view({"tower": -1})
    assert wrong.problem and wrong.enabled and "Tower cost weight" in wrong.text


def test_the_summary_names_the_changed_weights_only():
    assert co.summary({"smoothness": 0.5}).partition(". The")[0] == "Weights: turn penalty weight 0.5"
    assert co.summary({"tower": 3}).partition(". The")[0] == "Weights: tower cost weight 3"
    assert "Both weights are 1" in co.summary({})


def test_a_weights_text_names_every_weight_a_route_used():
    assert co.weights_text({"smoothness": 0.5, "tower": 2.0}) == "turn penalty weight 0.5, tower cost weight 2"
    assert co.weights_text({}) == "" == co.weights_text(None)


# ====================================================================== what reaches the library
def test_the_library_objective_is_none_for_defaults_and_carries_cost_one_and_both_weights_otherwise():
    assert co.library_objective({}) is None
    assert co.library_objective({"smoothness": 1, "tower": 1.0}) is None
    assert co.library_objective({"tower": 2}) == {"cost": 1.0, "smoothness": 1.0, "tower": 2.0}
    assert co.library_objective(CHANGED) == {"cost": 1.0, "smoothness": 0.5, "tower": 2.0}
    assert co.used_weights(CHANGED) == CHANGED
    assert co.library_objective({"smoothness": 0, "tower": 0}) == {"cost": 1.0, "smoothness": 0.0, "tower": 0.0}


@pytest.mark.parametrize("raw", [{"smoothness": -1}, {"tower": 1e9}, {"tower": "x"}, {"smoothness": float("nan")}])
def test_the_library_objective_refuses_a_value_that_is_a_problem_with_a_project_error(raw):
    with pytest.raises(InvalidInputError) as caught:
        co.library_objective(raw)
    assert isinstance(caught.value, PyorpsError)


# ====================================================================== the notice of a problem
@pytest.mark.parametrize(("key", "control"), [("smoothness", SMOOTH), ("tower", TOWER),
                                              ("backend", ids.OHL_BACKEND),
                                              ("clearance", ids.OHLF_IDS["min_clearance_m"])])
def test_each_problem_key_has_a_control_and_a_notice_that_focuses_it(key, control):
    assert form_section.OBJECTIVE_CONTROLS[key] == control
    notice = co.problem_notice([(key, "a message")], controls=form_section.OBJECTIVE_CONTROLS, tab=ids.TAB_ROUTES)
    assert notice["severity"] == "error" and notice["meaning"] == "a message"
    assert notice["focus_control"] == control and notice["focus_id"] == ids.TAB_ROUTES
    assert notice["title"] == f"{co.LABELS[key]}: value not usable"
    assert "Reset weights" in notice["fix"] and "overhead-line" in notice["impact"]


def test_no_problems_make_no_notice_and_the_text_counts_the_rest():
    assert co.problem_notice([]) is None
    found = [("smoothness", "a"), ("tower", "b"), ("backend", "c"), ("clearance", "d")]
    assert co.problems_text(found, limit=2).splitlines() == ["a", "b", "... and 2 more."]


# ====================================================================== the library agrees
def _finder(objective=None, **kwargs):
    from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

    n = 30
    profile = {"name": "p", "description": "d", "soft_angle_limit_deg": 5.0, "hard_angle_limit_deg": 90.0,
               "angle_cost_function": "linear", "angle_cost_params": {"scale": 10.0}, "min_span_m": 3.0,
               "max_span_m": 8.0, "span_bin_size_m": 1.0,
               "tower_cost_params": {"terrain_cost_map": {0: 100, 100: 500, 500: 2000},
                                     "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                                                     "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}}}
    profile.update(kwargs.pop("profile_extra", {}))
    kwargs.setdefault("graph_api", "cython")
    return ConstrainedPathFinder(
        dataset_source=np.full((n, n), 100, dtype=np.uint16), crs=CRS, transform=from_origin(0.0, float(n), 1.0, 1.0),
        source_coords=(3.5, 15.5), target_coords=(26.5, 15.5), search_space_buffer_m=100, profile=profile,
        objective=objective, **kwargs)


@pytest.mark.parametrize("raw", [{"smoothness": 0.0, "tower": 0.0}, {"smoothness": 1000.0, "tower": 1000.0},
                                 {"smoothness": 0.5, "tower": 2.0}, {"tower": 1000.0}])
def test_the_library_takes_every_weight_the_service_allows_and_still_routes(raw):
    assert co.problems(raw) == []
    finder = _finder(co.library_objective(raw))
    path = finder.find_route()
    assert path.n_towers > 0 and finder.objective.weights == co.library_objective(raw)
    assert path.metrics is not None and path.feasibility is not None      # objective mode reports its metrics


def test_the_library_refuses_what_the_service_refuses():
    objective = co.library_objective(CHANGED)
    assert [k for k, _ in co.problems(CHANGED, backend="cython_parallel")] == ["backend"]
    with pytest.raises(NotImplementedError):
        _finder(objective, graph_api="cython_parallel")
    clearance = {**CLEARANCE}
    assert [k for k, _ in co.problems(CHANGED, clearance_on=True)] == ["clearance"]
    with pytest.raises(NotImplementedError):
        _finder(objective, profile_extra=clearance)
    for bad in ({"smoothness": -1.0}, {"tower": float("nan")}):
        assert co.problems(bad)
        with pytest.raises(ObjectiveError):
            _finder({"cost": 1.0, **bad})


def test_the_library_does_not_refuse_a_terrain_model_it_only_warns():
    dem = np.zeros((30, 30), dtype=np.float32)
    with pytest.warns(UserWarning, match="gradient model"):
        _finder(co.library_objective(CHANGED), dem=dem)
    assert co.problems(CHANGED) == []


def test_the_clearance_state_of_the_profile_form_is_the_library_rule():
    for extra, on in (({}, False), (CLEARANCE, True), ({"tower_height_m": 30.0}, False)):
        profile = {**MINI_PROFILE, **extra}
        assert (profile_service.clearance_state(profile) == "on") is on
    assert form_section.profile_facts({**MINI_PROFILE, **CLEARANCE}) == {"clearance_on": True, "spans_on": True}
    assert form_section.profile_facts({}) == {"clearance_on": False, "spans_on": None}
    assert form_section.profile_facts({"min_span_m": 5}) == {"clearance_on": False, "spans_on": False}


# ====================================================================== the run: defaults pass nothing
@pytest.mark.parametrize("held", [None, {}, {"smoothness": 1.0, "tower": 1.0}, {"smoothness": None, "tower": ""}])
def test_default_weights_pass_no_objective_to_the_library_and_leave_the_route_params_alone(
        app, state, ohl_layer, library_calls, held):
    if held is not None:
        state.gui_settings["constrained"] = {"objective": held}
    resp = _run(app, ohl_layer.id)
    assert len(library_calls) == 1 and "objective" not in library_calls[0]
    route = _routes(state)[0]
    assert "objective_weights" not in route.meta["params"]
    assert "objective_weights" not in route.meta["constrained_summary"]
    assert "weights" not in resp[ids.OHL_STATUS]["children"]
    assert "Objective weights used" not in str(_notices(resp))


def test_a_default_run_is_the_same_route_as_the_library_without_an_objective(app, state, ohl_layer, tiny_raster):
    _run(app, ohl_layer.id)
    plain = constrained.run_constrained_path(tiny_raster, points=POINTS, profile=MINI_PROFILE, backend="cython",
                                             neighborhood="r1", search_buffer_m=100)         # the page's k = 1
    assert _routes(state)[0].gdf.geometry.iloc[0].equals(plain["line"])


# ====================================================================== the run: weights reach the library
def test_changed_weights_reach_the_library_unchanged_in_a_synchronous_run(app, state, ohl_layer, library_calls):
    _weights(state, **CHANGED)
    resp = _run(app, ohl_layer.id)
    assert state.jobs == {} and ids.OHL_POLL not in resp
    assert library_calls[0]["objective"] == {"cost": 1.0, "smoothness": 0.5, "tower": 2.0}
    assert library_calls[0]["graph_api"] == "cython"
    assert len(_routes(state)) == 1 and len(_towers(state)) == 1


def test_changed_weights_reach_the_library_unchanged_in_a_background_job(app, state, ohl_layer, library_calls,
                                                                         heavy):
    _weights(state, **CHANGED)
    start = _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    assert job.params["run"]["objective"] == {"cost": 1.0, "smoothness": 0.5, "tower": 2.0}
    assert job.params["finalize"]["objective_weights"] == CHANGED
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive()
    poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    assert library_calls[0]["objective"] == {"cost": 1.0, "smoothness": 0.5, "tower": 2.0}
    assert _routes(state)[0].meta["params"]["objective_weights"] == CHANGED


def test_the_job_keeps_the_weights_of_its_start_when_the_page_changes_them_afterwards(app, state, ohl_layer, heavy):
    _weights(state, **CHANGED)
    start = _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    state.gui_settings["constrained"]["objective"]["tower"] = 7.0           # the user edits while it runs
    job.thread.join(JOIN_S)
    poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    assert _routes(state)[0].meta["params"]["objective_weights"] == CHANGED


def test_a_default_background_run_has_no_objective_in_its_snapshot_or_finalize(app, state, ohl_layer, heavy):
    _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    assert "objective" not in job.params["run"] and "objective_weights" not in job.params["finalize"]
    job.thread.join(JOIN_S)


def test_the_service_snapshot_copies_the_weights_and_leaves_a_classic_snapshot_unchanged():
    weights = {"cost": 1.0, "smoothness": 0.5, "tower": 2.0}
    kwargs = {"points": POINTS, "profile": {"name": "p"}, "backend": "cython", "neighborhood": "r1",
              "search_buffer_m": None, "dem": None, "dsm": None}
    snap = constrained.snapshot_run_inputs("r.tif", objective=weights, **kwargs)
    weights["tower"] = 99.0
    assert snap["objective"] == {"cost": 1.0, "smoothness": 0.5, "tower": 2.0}
    classic = constrained.snapshot_run_inputs("r.tif", **kwargs)
    assert "objective" not in classic
    assert "objective" not in constrained.snapshot_run_inputs("r.tif", objective=None, **kwargs)
    assert "objective" not in constrained.snapshot_run_inputs("r.tif", objective={}, **kwargs)


# ====================================================================== a weight changes the route
@pytest.fixture(scope="module")
def zigzag_raster(tmp_path_factory):
    """Dear ground (cost 300) with a cheap winding corridor (cost 20): bends and towers are the price of the
    cheap ground, so a weight on either changes the route."""
    n = 100
    array = np.full((n, n), 300, dtype=np.uint16)
    rows, cols = np.mgrid[0:n, 0:n]
    x, y = cols + 0.5, n - (rows + 0.5)
    corridor = [(10, 10), (30, 40), (60, 20), (80, 60), (90, 90)]
    for (x0, y0), (x1, y1) in zip(corridor[:-1], corridor[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        array[np.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= 4] = 20
    return _write_raster(tmp_path_factory.mktemp("r16_zigzag") / "zigzag.tif", array)


def _path(raster, objective=None):
    return constrained.run_constrained_path(raster, points=POINTS, profile=MINI_PROFILE, backend="cython",
                                            neighborhood="r2", search_buffer_m=100,
                                            **({"objective": objective} if objective else {}))


def _turning(line) -> float:
    """The sum of the heading changes along a line, in degrees (how much it bends)."""
    steps = np.diff(np.asarray(line.coords), axis=0)
    headings = np.degrees(np.arctan2(steps[:, 1], steps[:, 0]))
    return float(np.abs((np.diff(headings) + 180.0) % 360.0 - 180.0).sum())


def test_a_cheap_turn_weight_follows_the_winding_corridor_and_a_dear_one_runs_straight(zigzag_raster):
    free = _path(zigzag_raster, co.library_objective({"smoothness": 0.0}))
    classic = _path(zigzag_raster)
    stiff = _path(zigzag_raster, co.library_objective({"smoothness": 1000.0}))
    assert not free["line"].equals(stiff["line"]) and not classic["line"].equals(stiff["line"])
    assert _turning(free["line"]) > _turning(stiff["line"])


def test_a_dear_tower_weight_changes_the_route_and_never_adds_towers(zigzag_raster):
    free = _path(zigzag_raster, co.library_objective({"tower": 0.0}))
    dear = _path(zigzag_raster, co.library_objective({"tower": 1000.0}))
    assert not free["line"].equals(dear["line"])
    assert dear["summary"]["n_towers"] <= free["summary"]["n_towers"]


def test_the_all_ones_objective_is_the_classic_route(zigzag_raster):
    """The library's own gate: weights of 1 are the profile (the GUI sends nothing for them)."""
    classic = _path(zigzag_raster)
    ones = _path(zigzag_raster, {"cost": 1.0, "smoothness": 1.0, "tower": 1.0})
    assert classic["line"].equals(ones["line"]) and classic["summary"]["n_towers"] == ones["summary"]["n_towers"]


def test_the_run_of_the_page_gives_the_weighted_route_not_the_classic_one(app, state, zigzag_raster):
    layer = state.add_layer("cost", "raster", crs=CRS, meta={"source_path": zigzag_raster})
    _run(app, layer.id, k=2)
    _weights(state, smoothness=1000.0)
    _run(app, layer.id, k=2)
    classic, stiff = _routes(state)
    assert not classic.gdf.geometry.iloc[0].equals(stiff.gdf.geometry.iloc[0])
    assert stiff.meta["params"]["objective_weights"] == {"smoothness": 1000.0, "tower": 1.0}
    assert "objective_weights" not in classic.meta["params"]


# ====================================================================== what is stored and said
def test_the_weights_are_stored_with_the_route_and_named_in_status_and_notice(app, state, ohl_layer):
    _weights(state, **CHANGED)
    resp = _run(app, ohl_layer.id)
    route = _routes(state)[0]
    assert route.meta["params"]["objective_weights"] == CHANGED
    assert route.meta["params"]["constrained"] is True and "profile" in route.meta["params"]
    assert route.meta["constrained_summary"]["objective_weights"] == CHANGED
    status = resp[ids.OHL_STATUS]["children"]
    assert status.startswith("OHL: ") and status.endswith("; weights: turn penalty weight 0.5, tower cost weight 2")
    success = next(n for n in _notices(resp) if n["title"].startswith("Overhead line routed"))
    assert "Objective weights used: turn penalty weight 0.5, tower cost weight 2." in success["meaning"]


def test_the_stored_weights_do_not_change_when_the_state_is_changed_later(app, state, ohl_layer):
    _weights(state, **CHANGED)
    _run(app, ohl_layer.id)
    state.gui_settings["constrained"]["objective"]["tower"] = 5.0
    assert _routes(state)[0].meta["params"]["objective_weights"] == CHANGED


def test_a_route_with_weights_survives_save_and_open(app, state, ohl_layer, tmp_path):
    _weights(state, **CHANGED)
    _run(app, ohl_layer.id)
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings["constrained"]["objective"] == {"smoothness": 0.5, "tower": 2.0}
        route = other.layers_of_kind("route")[0]
        assert route.meta["params"]["objective_weights"] == CHANGED
        assert route.meta["constrained_summary"]["objective_weights"] == CHANGED
    finally:
        other.shutdown()


# ====================================================================== the run: problems start nothing
def _nothing_started(state, calls):
    assert calls == [] and state.jobs == {} and _routes(state) == [] and _towers(state) == []


@pytest.mark.parametrize(("key", "value", "control", "fragment"), [
    ("smoothness", 5000.0, SMOOTH, "too high"), ("smoothness", -1.0, SMOOTH, "too low"),
    ("tower", 1000.5, TOWER, "too high"), ("tower", "abc", TOWER, "not a number")])
def test_a_weight_that_is_not_allowed_is_an_error_notice_that_focuses_the_field_and_starts_nothing(
        app, state, ohl_layer, library_calls, key, value, control, fragment):
    state.gui_settings["constrained"] = {"objective": {key: value}}
    resp = _run(app, ohl_layer.id)
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and notice["focus_control"] == control
    assert notice["focus_id"] == ids.TAB_ROUTES and fragment in notice["meaning"]
    assert co.FIELD_SPECS[key].label in notice["title"] and "No overhead-line route was computed" in notice["impact"]
    _nothing_started(state, library_calls)
    assert ids.OHL_POLL not in resp and ids.OHL_STOP_BTN not in resp


def test_a_weight_that_is_not_allowed_starts_no_background_job_either(app, state, ohl_layer, library_calls, heavy):
    _weights(state, tower=5000.0)
    resp = _run(app, ohl_layer.id)
    assert _notices(resp)[-1]["focus_control"] == TOWER
    _nothing_started(state, library_calls)


def test_a_changed_weight_on_an_experimental_backend_is_refused_and_focuses_the_backend(app, state, ohl_layer,
                                                                                         library_calls):
    _weights(state, tower=2.0)
    resp = _run(app, ohl_layer.id, backend="cython_parallel", experimental=True)
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and notice["focus_control"] == ids.OHL_BACKEND
    assert "cython_parallel" in notice["meaning"] and "tower cost weight 2" in notice["meaning"]
    _nothing_started(state, library_calls)


def test_a_backend_that_the_run_resets_to_cython_takes_the_weights(app, state, ohl_layer, library_calls):
    """Without the experimental flag the run uses cython (the gate), so the weights are fine."""
    _weights(state, tower=2.0)
    resp = _run(app, ohl_layer.id, backend="cython_parallel", experimental=False)
    assert "Backend reset to cython" in _titles(resp)
    assert library_calls[0]["graph_api"] == "cython" and library_calls[0]["objective"]["tower"] == 2.0
    assert len(_routes(state)) == 1


def test_changed_weights_with_the_clearance_check_are_refused_and_focus_a_clearance_field(app, state, ohl_layer,
                                                                                           library_calls):
    _weights(state, smoothness=0.0)
    resp = _run(app, ohl_layer.id, profile={**MINI_PROFILE, **CLEARANCE})
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and notice["focus_control"] == ids.OHLF_IDS["min_clearance_m"]
    assert "Clearance check" in notice["title"] and "turn penalty weight 0" in notice["meaning"]
    _nothing_started(state, library_calls)


def test_default_weights_run_with_the_clearance_check_as_before(app, state, ohl_layer, library_calls):
    """A profile with clearance and no weights is not touched by this block (it needs a DSM at the library)."""
    resp = _run(app, ohl_layer.id, profile={**MINI_PROFILE, **CLEARANCE})
    assert "Clearance check: value not usable" not in _titles(resp)
    assert all("objective" not in call for call in library_calls)


def test_a_partial_clearance_setting_does_not_refuse_the_weights(app, state, ohl_layer, library_calls):
    _weights(state, tower=2.0)
    resp = _run(app, ohl_layer.id, profile={**MINI_PROFILE, "tower_height_m": 30.0})
    assert "Clearance check: value not usable" not in _titles(resp)
    assert library_calls[0]["objective"]["tower"] == 2.0


def test_a_terrain_model_only_makes_the_library_warn_and_the_run_goes_on(app, state, ohl_layer, library_calls,
                                                                         tmp_path):
    dem = _write_raster(tmp_path / "dem.tif", np.zeros((100, 100), np.uint16))
    _weights(state, tower=2.0)
    resp = _run(app, ohl_layer.id, dem=dem)
    assert "Backend: value not usable" not in _titles(resp)
    assert library_calls and library_calls[0]["objective"]["tower"] == 2.0
    assert any("gradient model" in json.dumps(n) for n in _notices(resp))


def test_the_run_callback_gained_no_state_and_no_input(app):
    spec = next(s for s in app._callback_list  # noqa: SLF001
                if f"{ids.OHL_STATUS}.children" in str(s["output"]) and ids.RUN_ROUTING_BTN in str(s["inputs"]))
    assert [i["id"] for i in spec["inputs"]] == [ids.RUN_ROUTING_BTN]
    assert len(spec["state"]) == 12


# ====================================================================== ids
def test_the_new_ids_have_the_prefix_and_neither_contain_nor_sit_inside_an_older_id():
    mine = [*ids.OHLO_IDS.values(), ids.OHLO_BLOCK, ids.OHLO_FIELDS, ids.OHLO_STATUS, ids.OHLO_RESET_BTN,
            ids.OHLO_SETTINGS]
    assert all(value.startswith("ohlo-") for value in mine) and len(set(mine)) == len(mine) == 7
    everything = {k: v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str)}
    older = [v for k, v in everything.items() if v not in mine]
    assert [(m, o) for m in mine for o in older if o in m or m in o] == []
    assert tuple(ids.OHLO_IDS) == ids.OHLO_KEYS and ids.OHLO_IDS == {"smoothness": "ohlo-w-smoothness",
                                                                    "tower": "ohlo-w-tower"}


# ====================================================================== the block in the layout
def test_the_block_sits_inside_the_profile_form_as_a_collapsed_details(page):
    block = _by_id(page, ids.OHLO_BLOCK)
    assert isinstance(block, html.Details) and block.open is False
    form = next(c for c in _components(page) if getattr(c, "className", "") == "ohlf-form my-1")
    assert any(part is block for part in form._traverse())  # noqa: SLF001
    summary = block.children[0]
    assert isinstance(summary, html.Summary) and summary.children == "Objective weights (advanced)"


@pytest.mark.parametrize("key", co.WEIGHT_KEYS)
def test_each_weight_input_is_built_from_its_spec(page, key):
    spec = co.FIELD_SPECS[key]
    control = _by_id(page, ids.OHLO_IDS[key])
    assert isinstance(control, dbc.Input) and control.type == "number" and control.debounce is True
    assert (control.min, control.max, control.step, control.value) == (spec.minimum, spec.maximum, "any",
                                                                      spec.default)
    label = next(c for c in _components(page)
                 if isinstance(c, dbc.Label) and getattr(c, "html_for", None) == ids.OHLO_IDS[key])
    assert label.children == f"{spec.label} ({spec.unit})"
    wrapper = next(c for c in _components(page) if isinstance(c, html.Div) and isinstance(c.children, list)
                   and label in c.children and control in c.children)
    assert spec.help in wrapper.title and co.describe_range(spec) in wrapper.title


def test_the_block_has_one_status_line_one_fieldset_one_reset_button_and_one_store(page):
    status = _by_id(page, ids.OHLO_STATUS)
    assert status.children == co.summary({}) and status.className == form_section.STATUS_CLASS_OK
    fieldset = _by_id(page, ids.OHLO_FIELDS)
    assert isinstance(fieldset, html.Fieldset) and fieldset.disabled is False
    assert getattr(fieldset, "title", None) is None
    assert [c.id for c in fieldset._traverse() if getattr(c, "id", None) in ids.OHLO_IDS.values()] == [  # noqa: SLF001
        SMOOTH, TOWER]
    assert _by_id(page, ids.OHLO_RESET_BTN).children == "Reset weights"
    store = _by_id(page, ids.OHLO_SETTINGS)
    assert isinstance(store, dcc.Store) and store.data == {"smoothness": 1.0, "tower": 1.0}


# ====================================================================== the mirror
def test_the_weights_are_mirrored_into_the_state_and_the_store(app, state):
    resp = invoke(app, MIRROR, 0.5, 2)
    assert state.gui_settings["constrained"]["objective"] == {"smoothness": 0.5, "tower": 2.0}
    assert resp[ids.OHLO_SETTINGS]["data"] == {"smoothness": 0.5, "tower": 2.0}


def test_defaults_leave_no_objective_key_and_create_no_constrained_settings(app, state):
    invoke(app, MIRROR, None, None)
    assert "constrained" not in state.gui_settings
    invoke(app, MIRROR, 1, 1.0)
    assert state.gui_settings == {}


def test_going_back_to_the_defaults_removes_the_key_and_keeps_the_other_settings(app, state):
    state.gui_settings["constrained"] = {"backend": "raster_gpu", "experimental": True, "dem": None, "dsm": None}
    invoke(app, MIRROR, 3, 1)
    assert state.gui_settings["constrained"]["objective"] == {"smoothness": 3.0, "tower": 1.0}
    assert state.gui_settings["constrained"]["backend"] == "raster_gpu"
    invoke(app, MIRROR, 1, 1)
    assert state.gui_settings["constrained"] == {"backend": "raster_gpu", "experimental": True, "dem": None,
                                                 "dsm": None}


def test_a_value_that_is_no_number_is_kept_for_the_run_to_name(app, state):
    invoke(app, MIRROR, 1, "x")
    saved = state.gui_settings["constrained"]["objective"]
    assert saved == {"smoothness": 1.0, "tower": "x"}
    assert [key for key, _ in co.problems(saved)] == ["tower"]


def test_the_profile_mirror_keeps_the_weights_when_it_rewrites_the_constrained_settings(app, state):
    invoke(app, MIRROR, 0.5, 2)
    text = yaml.safe_dump(MINI_PROFILE)
    resp = invoke(app, PROFILE_MIRROR, "raster_gpu", True, "dem.tif", None, text)
    saved = state.gui_settings["constrained"]
    assert saved["objective"] == {"smoothness": 0.5, "tower": 2.0} and saved["backend"] == "raster_gpu"
    assert saved["profile"]["min_span_m"] == 20 and resp[ids.OHL_SETTINGS]["data"] == saved


def test_the_profile_mirror_without_weights_has_no_objective_key(app, state):
    invoke(app, PROFILE_MIRROR, "cython", False, None, None, "")
    assert state.gui_settings["constrained"] == {"backend": "cython", "experimental": False,
                                                 "dem": None, "dsm": None}


# ====================================================================== the status line and the disabled states
def _status(app, weights=None, *, backend="cython", experimental=False, dem=None, base=None):
    resp = invoke(app, STATUS, co.normalise(weights), backend, experimental, dem, base or {})
    return (resp[ids.OHLO_STATUS]["children"], resp[ids.OHLO_STATUS]["className"],
            resp[ids.OHLO_FIELDS]["disabled"], resp[ids.OHLO_FIELDS]["title"])


def test_the_status_line_shows_the_summary_and_the_inputs_are_enabled(app):
    text, css, disabled, title = _status(app)
    assert text == co.summary({}) and css == form_section.STATUS_CLASS_OK and disabled is False and title is None
    text, css, disabled, _ = _status(app, CHANGED)
    assert text.startswith("Weights: turn penalty weight 0.5, tower cost weight 2") and disabled is False


def test_a_weight_that_is_not_allowed_is_named_in_the_status_line_in_the_words_of_the_run(app, state, ohl_layer):
    text, css, disabled, _ = _status(app, {"tower": 5000})
    assert css == form_section.STATUS_CLASS_PROBLEM and disabled is False
    state.gui_settings["constrained"] = {"objective": {"tower": 5000}}
    run_notice = _notices(_run(app, ohl_layer.id))[-1]
    assert text == run_notice["meaning"]


def test_the_inputs_are_disabled_with_the_reason_on_a_backend_that_cannot_take_the_weights(app):
    text, css, disabled, title = _status(app, backend="raster_gpu", experimental=True)
    assert disabled is True and text == title and "raster_gpu" in title and "Reset weights" in title
    assert css == form_section.STATUS_CLASS_OK                          # defaults: information, not an error
    text, css, disabled, title = _status(app, CHANGED, backend="raster_gpu", experimental=True)
    assert disabled is True and css == form_section.STATUS_CLASS_PROBLEM and "raster_gpu" in text and title


def test_an_experimental_backend_without_the_flag_counts_as_cython_like_the_run_does(app):
    text, css, disabled, title = _status(app, CHANGED, backend="raster_gpu", experimental=False)
    assert disabled is False and title is None and css == form_section.STATUS_CLASS_OK


def test_the_inputs_are_disabled_with_the_reason_while_the_clearance_check_is_on(app):
    base = {**MINI_PROFILE, **CLEARANCE}
    text, css, disabled, title = _status(app, CHANGED, base=base)
    assert disabled is True and title == co.CLEARANCE_REASON and css == form_section.STATUS_CLASS_PROBLEM
    assert "Clearance check" in text
    _, _, disabled, title = _status(app, base={**MINI_PROFILE, "tower_height_m": 30.0})
    assert disabled is False and title is None


def test_a_terrain_model_and_a_profile_without_spans_are_noted_not_refused(app):
    text, css, disabled, _ = _status(app, {"tower": 2}, dem="dem.tif", base={"min_span_m": 5})
    assert css == form_section.STATUS_CLASS_OK and disabled is False
    assert "terrain model" in text and "no towers" in text
    plain, _, _, _ = _status(app, {"tower": 2}, dem="  ", base={})
    assert "terrain model" not in plain and "no towers" not in plain


def test_the_status_callback_listens_to_weights_backend_flag_dem_and_the_profile(app):
    _key, entry = next((k, e) for k, e in app.callback_map.items() if f"{ids.OHLO_STATUS}.children" in k)
    assert [i["id"] for i in entry["inputs"]] == [ids.OHLO_SETTINGS, ids.OHL_BACKEND, ids.OHL_EXPERIMENTAL,
                                                  ids.OHL_DEM, ids.OHLF_BASE]


# ====================================================================== reset
def test_reset_puts_every_weight_back_to_its_default(app):
    resp = invoke(app, RESET, 1, triggered=[f"{ids.OHLO_RESET_BTN}.n_clicks"])
    assert resp[SMOOTH]["value"] == 1.0 and resp[TOWER]["value"] == 1.0


def test_reset_without_a_click_changes_nothing(app):
    with pytest.raises(PreventUpdate):
        invoke(app, RESET, None, triggered=[f"{ids.OHLO_RESET_BTN}.n_clicks"])


# ====================================================================== restore after Open and New
def _restore(app, current=(1.0, 1.0), status="opened: p"):
    return invoke(app, RESTORE, status, *current, triggered=[f"{ids.PROJECT_STATUS}.children"])


def test_open_project_puts_the_saved_weights_back_and_opens_the_block(app, state):
    state.gui_settings = {"constrained": {"objective": {"smoothness": 0.5, "tower": 2.0}}}
    resp = _restore(app, (1.0, 1.0))
    assert resp[SMOOTH]["value"] == 0.5 and resp[TOWER]["value"] == 2.0 and resp[ids.OHLO_BLOCK]["open"] is True


def test_new_project_puts_the_defaults_back_and_closes_the_block(app, state):
    state.gui_settings = {}
    resp = _restore(app, (0.5, 2.0), status="new project")
    assert resp[SMOOTH]["value"] == 1.0 and resp[TOWER]["value"] == 1.0 and resp[ids.OHLO_BLOCK]["open"] is False


def test_a_project_saved_before_the_block_existed_opens_with_defaults(app, state):
    state.gui_settings = {"constrained": {"backend": "cython"}}
    assert _restore(app, (3.0, 3.0))[SMOOTH]["value"] == 1.0


def test_a_page_that_already_shows_the_project_is_left_alone(app, state):
    state.gui_settings = {"constrained": {"objective": {"smoothness": 0.5, "tower": 2.0}}}
    with pytest.raises(PreventUpdate):
        _restore(app, (0.5, 2.0))
    state.gui_settings = {}
    with pytest.raises(PreventUpdate):
        _restore(app, (None, None))                 # blank inputs show the defaults


def test_saving_a_project_does_not_rewrite_the_inputs(app, state):
    state.gui_settings = {"constrained": {"objective": {"smoothness": 0.5, "tower": 2.0}}}
    with pytest.raises(PreventUpdate):
        _restore(app, (1.0, 1.0), status="saved: C:/p/project.json")


def test_a_saved_value_that_is_no_number_comes_back_as_a_blank_input(app, state):
    state.gui_settings = {"constrained": {"objective": {"smoothness": "abc", "tower": 2.0}}}
    resp = _restore(app, (1.0, 1.0))
    assert resp[SMOOTH]["value"] is None and resp[TOWER]["value"] == 2.0


# ====================================================================== first paint and the project file
def test_a_blank_project_prefills_nothing_for_the_block(state):
    puts: list = []
    form_section.prefill_objective({}, lambda *args: puts.append(args))
    form_section.prefill_objective({"objective": {}}, lambda *args: puts.append(args))
    form_section.prefill(state, lambda *args: puts.append(args))
    assert puts == []


def test_a_saved_project_without_weights_keeps_the_block_as_the_layout_has_it(state):
    state.gui_settings = {"constrained": {"backend": "cython", "experimental": False}}
    page = build_layout(state)
    assert _by_id(page, SMOOTH).value == 1.0 and _by_id(page, ids.OHLO_BLOCK).open is False


def test_a_reloaded_page_shows_the_saved_weights_status_and_an_open_block(state):
    state.gui_settings = {"constrained": {"objective": {"smoothness": 0.5, "tower": 2.0}, "backend": "cython",
                                          "profile": {**MINI_PROFILE}}}
    page = build_layout(state)
    assert _by_id(page, SMOOTH).value == 0.5 and _by_id(page, TOWER).value == 2.0
    assert _by_id(page, ids.OHLO_SETTINGS).data == {"smoothness": 0.5, "tower": 2.0}
    assert _by_id(page, ids.OHLO_BLOCK).open is True
    assert _by_id(page, ids.OHLO_STATUS).children.startswith("Weights: turn penalty weight 0.5")
    assert _by_id(page, ids.OHLO_FIELDS).disabled is False
    assert _by_id(page, ids.OHL_SETTINGS).data == state.gui_settings["constrained"]


def test_a_reloaded_page_disables_saved_weights_that_the_saved_backend_cannot_take(state):
    state.gui_settings = {"constrained": {"objective": {"tower": 3.0}, "backend": "raster_gpu_v4",
                                          "experimental": True}}
    page = build_layout(state)
    fieldset = _by_id(page, ids.OHLO_FIELDS)
    assert fieldset.disabled is True and "raster_gpu_v4" in fieldset.title
    assert _by_id(page, ids.OHLO_STATUS).className == form_section.STATUS_CLASS_PROBLEM


def test_a_reloaded_page_disables_saved_weights_while_the_saved_profile_has_the_clearance_check(state):
    state.gui_settings = {"constrained": {"objective": {"tower": 3.0}, "profile": {**MINI_PROFILE, **CLEARANCE}}}
    page = build_layout(state)
    assert _by_id(page, ids.OHLO_FIELDS).title == co.CLEARANCE_REASON


def test_the_weights_survive_save_and_open_and_show_on_the_new_page(app, state, tmp_path):
    invoke(app, MIRROR, 0.5, 2)
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings["constrained"]["objective"] == {"smoothness": 0.5, "tower": 2.0}
        other.gui_settings = json.loads(json.dumps(other.gui_settings))
        other_page = build_layout(other)
        assert _by_id(other_page, SMOOTH).value == 0.5 and _by_id(other_page, TOWER).value == 2.0
    finally:
        other.shutdown()


def test_a_project_without_weights_is_saved_without_an_objective_key(app, state, tmp_path):
    invoke(app, PROFILE_MIRROR, "cython", False, None, None, yaml.safe_dump(MINI_PROFILE))
    invoke(app, MIRROR, None, None)
    path = project_io.save_project(state, tmp_path / "proj")
    manifest = json.loads(project_io.manifest_path(path).read_text(encoding="utf-8"))
    assert "objective" not in manifest["gui_settings"]["constrained"]

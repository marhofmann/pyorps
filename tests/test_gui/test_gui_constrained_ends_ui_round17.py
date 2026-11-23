"""Round 17, P5: the line ends of constrained (overhead-line) routing on the page.

Headless, tiny synthetic rasters, the cython constrained backend. Five layers are tested:

* the end-span rule as the 24th field of the profile form (spec, select, form edits) and the corrected help of the
  profile's report-only end height;
* the "Line ends" group (layout, ids, mirror, status lines, disabled states, clear, restore, first paint, project
  file); the rule itself is a profile field, the two connection heights are run settings kept in
  ``state.gui_settings["constrained"]["ends"]``;
* the run (``callbacks/constrained.run_constrained``): the heights come from the project state, a classic run passes
  nothing new, a value or combination the library refuses (and an exact search that would not fit) starts nothing and
  focuses the field - in the synchronous and the background path alike, the heights reach the library from both;
* the result: heights stored with the route, the rule, the heights, a removed end tower, span and search notes named in
  the status, the success notice and a notes notice, a classic run with a removed end tower included;
* a ``NoPathFoundError`` of a restricted search ends as the notice that names the rule or the heights.

A run is made "heavy" by patching the named criterion ``constrained.is_heavy`` (never by a large raster). The
constrained kernels are not deterministic with several threads, so every test runs with one.
"""
from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import numpy as np
import pytest
import yaml
from conftest import invoke, poll_job
from dash import dcc, html
from dash.exceptions import PreventUpdate
from rasterio.transform import from_bounds

from pyorps.core.infrastructure_profile import TERMINAL_SPAN_RULES
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
from pyorps.gui import ids
from pyorps.gui.callbacks import constrained as run_callbacks
from pyorps.gui.layout import build_layout
from pyorps.gui.sections import profile_form as form_section
from pyorps.gui.services import constrained, project_io
from pyorps.gui.services import constrained_ends as ce
from pyorps.gui.services import profile_form as profile_service
from pyorps.gui.state import ProjectState

CRS = "EPSG:32632"
JOIN_S = 60
SRC, TGT = ce.HEIGHT_KEYS
SRC_ID, TGT_ID = ids.OHLE_IDS[SRC], ids.OHLE_IDS[TGT]
RULE_ID = ids.OHLF_IDS["terminal_span_rule"]
RUN = (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
MIRROR = (f"{ids.OHLE_SETTINGS}.data",)
STATUS = (f"{ids.OHLE_STATUS}.children",)
RULE_STATUS = (f"{ids.OHLE_RULE_STATUS}.children",)
CLEAR = (f"{SRC_ID}.value", ids.OHLE_CLEAR_BTN)
RESTORE = (f"{SRC_ID}.value", ids.PROJECT_STATUS)
PROFILE_MIRROR = (f"{ids.OHL_SETTINGS}.data",)
EDIT = (f"{ids.OHLF_BASE}.data", ids.OHLF_IDS["name"])

BASE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 6, "max_span_m": 20, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 200}, "terrain_interpolation": "linear",
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 100},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300}}}}
CLEARANCE = {"tower_height_m": 30.0, "conductor_weight_per_m": 15.0, "conductor_tension_n": 25000.0,
             "min_clearance_m": 6.0}
WITH_CLEARANCE = {**BASE, **CLEARANCE}
ENFORCE = {**BASE, "terminal_span_rule": "enforce"}
POINTS = [(8.0, 8.0), (52.0, 52.0)]
#: a one-row corridor: the default rule takes out the end tower that stands 2 m from the target
CORRIDOR_PROFILE = {
    "soft_angle_limit_deg": 5.0, "hard_angle_limit_deg": 90.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 3.0, "max_span_m": 8.0, "span_bin_size_m": 1.0,
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 100},
                          "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                                          "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}}}
CORRIDOR_POINTS = [(0.5, 1.5), (23.5, 1.5)]
FORBIDDEN = 65535


# ------------------------------------------------------------------ helpers
@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # the constrained kernels read OMP_NUM_THREADS per call: one thread keeps ties deterministic (and the test
    # independent of the number of cores)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def _write_raster(path, array, dtype="uint16"):
    import rasterio

    rows, cols = array.shape
    with rasterio.open(str(path), "w", driver="GTiff", height=rows, width=cols, count=1, dtype=dtype, crs=CRS,
                       transform=from_bounds(0, 0, cols, rows, cols, rows)) as dst:
        dst.write(array.astype(dtype), 1)
    return str(path)


@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory):
    """A 60 x 60 m uniform cost raster at 1 m."""
    return _write_raster(tmp_path_factory.mktemp("p5") / "uniform.tif", np.full((60, 60), 100, np.uint16))


@pytest.fixture(scope="module")
def dem_file(tmp_path_factory):
    """Flat terrain on the grid of :func:`tiny_raster`."""
    return _write_raster(tmp_path_factory.mktemp("p5dem") / "dem.tif", np.zeros((60, 60), np.float32), "float32")


@pytest.fixture(scope="module")
def wall_raster(tmp_path_factory):
    """The uniform raster with a forbidden wall across it: no route exists between the two corners."""
    array = np.full((60, 60), 100, np.uint16)
    array[:, 30] = FORBIDDEN
    return _write_raster(tmp_path_factory.mktemp("p5wall") / "wall.tif", array)


@pytest.fixture(scope="module")
def corridor_raster(tmp_path_factory):
    array = np.full((3, 24), FORBIDDEN, np.uint16)
    array[1, :] = 100
    return _write_raster(tmp_path_factory.mktemp("p5corridor") / "corridor.tif", array)


def _layer(state, path):
    return state.add_layer("cost", "raster", crs=CRS, meta={"source_path": path})


@pytest.fixture()
def ohl_layer(state, tiny_raster):
    return _layer(state, tiny_raster)


def _point(x, y):
    return {"lat": 0, "lng": 0, "x": x, "y": y}


def _draft(points=POINTS, waypoints=()):
    return {"sources": [_point(*points[0])], "targets": [_point(*points[1])],
            "waypoints": [_point(*w) for w in waypoints]}


def _run(app, raster_id, *, profile=None, backend="cython", experimental=False, dem=None, points=POINTS,
         waypoints=(), notices=None, k=1):
    text = yaml.safe_dump(BASE if profile is None else profile)
    return invoke(app, RUN, 1, True, _draft(points, waypoints), raster_id, text, backend, experimental, k, 100, dem,
                  None, [], list(notices or []), triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _heights(state, **heights):
    """The project state as the group's mirror leaves it when the user typed ``heights`` (by key prefix)."""
    named = {SRC if key == "source" else TGT: value for key, value in heights.items()}
    state.gui_settings["constrained"] = {"backend": "cython", "experimental": False,
                                         "ends": {**ce.DEFAULTS, **named}}


def _routes(state):
    return state.layers_of_kind("route")


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _titles(resp) -> list[str]:
    return [n["title"] for n in _notices(resp)]


def _notice(resp, title_start):
    return next(n for n in _notices(resp) if n["title"].startswith(title_start))


@pytest.fixture()
def route_calls(monkeypatch):
    """The keywords of every ``find_route`` call of the finders the GUI builds (the real class still runs)."""
    import pyorps.graph.constrained_path_finder as library

    calls: list[dict] = []
    real = library.ConstrainedPathFinder

    class Spy(real):
        def find_route(self, *args, **kwargs):
            calls.append(dict(kwargs))
            return super().find_route(*args, **kwargs)

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


def _nothing_started(state, calls):
    assert calls == [] and state.jobs == {} and _routes(state) == []


def _edit_args(profile, edits=None, text=""):
    """The positional arguments of ``form_to_yaml``: the field values, the grids, the states."""
    values, angle, terrain, types = profile_service.form_values(profile)
    values = {**values, **(edits or {})}
    return (*(values[key] for key in profile_service.FORM_KEYS), angle, terrain, types, None, None, None, text,
            profile, "", [], None, None, None)


# ====================================================================== the rule: spec and select
def test_the_end_span_rule_is_the_24th_form_key_and_one_select_spec_with_the_four_library_values():
    assert len(ids.OHLF_KEYS) == 24 and ids.OHLF_KEYS[-1] == "terminal_span_rule"
    assert profile_service.FORM_KEYS == ids.OHLF_KEYS
    spec = profile_service.FIELD_SPECS["terminal_span_rule"]
    assert spec.kind == "select" and spec.options == TERMINAL_SPAN_RULES and spec.default == ce.DEFAULT_RULE
    assert spec.path == ("terminal_span_rule",) and spec.required is False and spec.label == "End-span rule"
    assert ids.OHLF_IDS["terminal_span_rule"] == "ohlf-terminal-span-rule"


def test_the_help_of_the_rule_says_what_an_end_tower_is_and_what_never_placing_one_needs():
    help_text = profile_service.FIELD_SPECS["terminal_span_rule"].help
    assert "end tower" in help_text and "closer to the line end than the shortest span" in help_text
    assert "Never place one" in help_text and "exact search" in help_text and "CPU backend" in help_text
    assert "span limits" in help_text


def test_the_help_of_the_end_height_of_the_profile_says_it_is_only_reported():
    help_text = profile_service.FIELD_SPECS["terminal_tower_height_m"].help
    assert help_text.startswith("Reported only") and "Line ends" in help_text and "connection heights" in help_text
    assert "does not use it" in help_text


def test_the_select_shows_the_labels_of_the_service_and_the_values_of_the_library(page):
    control = _by_id(page, RULE_ID)
    assert isinstance(control, dbc.Select) and control.value == "remove"
    assert control.options == ce.rule_options()
    assert [o["value"] for o in control.options] == list(TERMINAL_SPAN_RULES)
    assert [o["label"] for o in control.options] == [ce.RULE_LABELS[v].label for v in TERMINAL_SPAN_RULES]
    assert "Never place one" in control.options[-1]["label"]
    label = next(c for c in _components(page) if isinstance(c, dbc.Label) and getattr(c, "html_for", None) == RULE_ID)
    assert label.children == "End-span rule"
    wrapper = next(c for c in _components(page) if isinstance(c, html.Div) and isinstance(c.children, list)
                   and label in c.children and control in c.children)
    spec = profile_service.FIELD_SPECS["terminal_span_rule"]
    assert spec.help in wrapper.title and profile_service.describe_range(spec) in wrapper.title


def test_a_chosen_rule_is_written_into_the_yaml_text_and_the_base_store(app):
    text = yaml.safe_dump(BASE, sort_keys=False)
    resp = invoke(app, EDIT, *_edit_args(BASE, {"terminal_span_rule": "keep"}, text))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["terminal_span_rule"] == "keep" and written["min_span_m"] == 6
    assert resp[ids.OHLF_BASE]["data"] == written
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_OK


def test_the_default_rule_adds_no_key_and_a_stored_rule_survives_other_edits(app):
    text = yaml.safe_dump(BASE, sort_keys=False)
    resp = invoke(app, EDIT, *_edit_args(BASE, {"description": "x"}, text))
    assert "terminal_span_rule" not in yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    kept = {**BASE, "terminal_span_rule": "enforce"}
    resp = invoke(app, EDIT, *_edit_args(kept, {"description": "x"}, yaml.safe_dump(kept, sort_keys=False)))
    assert yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])["terminal_span_rule"] == "enforce"


def test_a_value_that_is_none_of_the_four_rules_is_a_problem_of_the_form():
    found = profile_service.profile_problems({**BASE, "terminal_span_rule": "sideways"})
    assert [key for key, _ in found] == ["terminal_span_rule"]
    message = found[0][1]
    assert "'sideways'" in message and all(rule in message for rule in TERMINAL_SPAN_RULES)


def test_a_shipped_profile_fills_the_rule_select_with_the_default(app):
    name = next(iter(constrained.shipped_profiles()))
    resp = invoke(app, (f"{ids.OHLF_BASE}.data", ids.OHL_PROFILE_LOAD_BTN, ids.OHLF_FROM_YAML_BTN),
                  constrained.shipped_profiles()[name], None, None, "", "", [],
                  triggered=[f"{ids.OHL_PROFILE_PRESET}.value"])
    assert resp[RULE_ID]["value"] == "remove"


# ====================================================================== ids
def test_the_new_ids_have_the_prefix_and_neither_contain_nor_sit_inside_an_older_id():
    mine = [*ids.OHLE_IDS.values(), ids.OHLE_BLOCK, ids.OHLE_FIELDS, ids.OHLE_STATUS, ids.OHLE_RULE_STATUS,
            ids.OHLE_CLEAR_BTN, ids.OHLE_SETTINGS]
    assert all(value.startswith("ohle-") for value in mine) and len(set(mine)) == len(mine) == 8
    everything = {k: v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str)}
    older = [v for k, v in everything.items() if v not in mine]
    assert [(m, o) for m in mine for o in older if o in m or m in o] == []
    assert ids.OHLE_KEYS == ce.HEIGHT_KEYS
    assert ids.OHLE_IDS == {"source_connection_height_m": "ohle-source-connection-height-m",
                            "target_connection_height_m": "ohle-target-connection-height-m"}


# ====================================================================== the group in the layout
def test_the_group_sits_last_in_the_profile_form_as_a_collapsed_details_with_the_rule_select_inside(page):
    block = _by_id(page, ids.OHLE_BLOCK)
    assert isinstance(block, html.Details) and block.open is False
    form = next(c for c in _components(page) if getattr(c, "className", "") == "ohlf-form my-1")
    assert any(part is block for part in form._traverse())  # noqa: SLF001
    assert form.children[-1] is block                         # after every other group of the form
    summary = block.children[0]
    assert isinstance(summary, html.Summary) and summary.children == "Line ends"
    assert any(part is _by_id(page, RULE_ID) for part in block._traverse())  # noqa: SLF001


@pytest.mark.parametrize("key", ce.HEIGHT_KEYS)
def test_each_height_input_is_built_from_its_spec(page, key):
    spec = ce.FIELD_SPECS[key]
    control = _by_id(page, ids.OHLE_IDS[key])
    assert isinstance(control, dbc.Input) and control.type == "number" and control.debounce is True
    assert (control.min, control.max, control.step, control.value) == (spec.minimum, spec.maximum, "any", None)
    label = next(c for c in _components(page)
                 if isinstance(c, dbc.Label) and getattr(c, "html_for", None) == ids.OHLE_IDS[key])
    assert label.children == f"{spec.label} ({spec.unit})"
    wrapper = next(c for c in _components(page) if isinstance(c, html.Div) and isinstance(c.children, list)
                   and label in c.children and control in c.children)
    assert spec.help in wrapper.title and "> 0 and <= 300 (m)" in wrapper.title and "blank" in wrapper.title


def test_the_group_has_two_status_lines_one_fieldset_one_clear_button_and_one_store(page):
    status = _by_id(page, ids.OHLE_STATUS)
    assert status.className in (form_section.STATUS_CLASS_OK, form_section.STATUS_CLASS_PROBLEM)
    fieldset = _by_id(page, ids.OHLE_FIELDS)
    assert isinstance(fieldset, html.Fieldset)
    assert [c.id for c in fieldset._traverse() if getattr(c, "id", None) in ids.OHLE_IDS.values()] == [  # noqa: SLF001
        SRC_ID, TGT_ID]
    clear = _by_id(page, ids.OHLE_CLEAR_BTN)
    assert clear.children == "Clear heights"
    assert not any(part is clear for part in fieldset._traverse())   # noqa: SLF001  # the way out stays enabled
    store = _by_id(page, ids.OHLE_SETTINGS)
    assert isinstance(store, dcc.Store) and store.data == {SRC: None, TGT: None}
    assert _by_id(page, ids.OHLE_RULE_STATUS) is not None


def test_a_page_without_a_profile_says_why_nothing_can_act_yet(page):
    """No profile means no span limits: the first paint is what the status callbacks answer for an empty profile."""
    fieldset = _by_id(page, ids.OHLE_FIELDS)
    reason = ce.support(backend="cython", profile={}).reason
    assert fieldset.disabled is True and fieldset.title == reason
    assert _by_id(page, ids.OHLE_STATUS).children.startswith(reason)
    assert _by_id(page, RULE_ID).disabled is True
    assert _by_id(page, ids.OHLE_RULE_STATUS).children == ce.rule_support({}).reason


def test_the_spans_group_points_at_the_line_ends_group(page):
    texts = [c.children for c in _components(page) if isinstance(c, html.Small) and isinstance(c.children, str)]
    assert any("Line ends" in text and "end-span rule" in text for text in texts)


# ====================================================================== the mirror
def test_the_heights_are_mirrored_into_the_state_and_the_store(app, state):
    resp = invoke(app, MIRROR, 25, "12.5")
    assert state.gui_settings["constrained"]["ends"] == {SRC: 25.0, TGT: 12.5}
    assert resp[ids.OHLE_SETTINGS]["data"] == {SRC: 25.0, TGT: 12.5}


def test_no_height_leaves_no_ends_key_and_creates_no_constrained_settings(app, state):
    invoke(app, MIRROR, None, None)
    assert "constrained" not in state.gui_settings
    invoke(app, MIRROR, "", "  ")
    assert state.gui_settings == {}


def test_clearing_the_heights_removes_the_key_and_keeps_the_other_settings(app, state):
    state.gui_settings["constrained"] = {"backend": "raster_gpu", "experimental": True, "dem": None, "dsm": None}
    invoke(app, MIRROR, 30, None)
    assert state.gui_settings["constrained"]["ends"] == {SRC: 30.0, TGT: None}
    assert state.gui_settings["constrained"]["backend"] == "raster_gpu"
    invoke(app, MIRROR, None, None)
    assert state.gui_settings["constrained"] == {"backend": "raster_gpu", "experimental": True, "dem": None,
                                                 "dsm": None}


def test_a_value_that_is_no_number_is_kept_for_the_run_to_name(app, state):
    invoke(app, MIRROR, None, "tall")
    saved = state.gui_settings["constrained"]["ends"]
    assert saved == {SRC: None, TGT: "tall"}
    assert [key for key, _ in ce.problems(saved)] == [TGT]


def test_the_profile_mirror_keeps_the_heights_when_it_rewrites_the_constrained_settings(app, state):
    invoke(app, MIRROR, 25, 12)
    text = yaml.safe_dump({**BASE, "terminal_span_rule": "keep"})
    resp = invoke(app, PROFILE_MIRROR, "raster_gpu", True, "dem.tif", None, text)
    saved = state.gui_settings["constrained"]
    assert saved["ends"] == {SRC: 25.0, TGT: 12.0} and saved["backend"] == "raster_gpu"
    assert saved["profile"]["terminal_span_rule"] == "keep" and resp[ids.OHL_SETTINGS]["data"] == saved


def test_the_profile_mirror_without_heights_has_no_ends_key(app, state):
    invoke(app, PROFILE_MIRROR, "cython", False, None, None, "")
    assert state.gui_settings["constrained"] == {"backend": "cython", "experimental": False, "dem": None,
                                                 "dsm": None}


def test_the_heights_and_the_weights_do_not_disturb_each_other(app, state):
    invoke(app, (f"{ids.OHLO_SETTINGS}.data",), 0.5, 2)
    invoke(app, MIRROR, 25, None)
    invoke(app, PROFILE_MIRROR, "cython", False, None, None, "")
    saved = state.gui_settings["constrained"]
    assert saved["objective"] == {"smoothness": 0.5, "tower": 2.0} and saved["ends"] == {SRC: 25.0, TGT: None}


# ====================================================================== the status line of the heights
def _ends_status(app, heights=None, *, backend="cython", experimental=False, dem=None, base=None):
    resp = invoke(app, STATUS, ce.normalise(heights), backend, experimental, dem, BASE if base is None else base)
    return (resp[ids.OHLE_STATUS]["children"], resp[ids.OHLE_STATUS]["className"],
            resp[ids.OHLE_FIELDS]["disabled"], resp[ids.OHLE_FIELDS]["title"])


def test_the_status_line_shows_the_summary_and_the_inputs_are_enabled(app):
    text, css, disabled, title = _ends_status(app)
    assert text == ce.summary({}) and css == form_section.STATUS_CLASS_OK and disabled is False and title is None
    text, css, disabled, _ = _ends_status(app, {SRC: 25})
    assert text.startswith(ce.summary({SRC: 25})) and css == form_section.STATUS_CLASS_OK and disabled is False


def test_a_height_that_cannot_act_is_reported_only_and_the_status_line_says_the_route_does_not_change(app):
    text, css, disabled, _ = _ends_status(app, {SRC: 25}, dem=None)
    assert "reported only, the route does not change" in text and "no terrain model (DEM)" in text
    assert css == form_section.STATUS_CLASS_OK and disabled is False
    text, _, _, _ = _ends_status(app, {SRC: 25}, dem="dem.tif", base=WITH_CLEARANCE)
    assert "reported only" not in text and "exact span bins" in text
    plain, _, _, _ = _ends_status(app, {SRC: 25}, dem="  ", base=WITH_CLEARANCE)       # a blank DEM is none
    assert "reported only" in plain


def test_a_height_that_is_not_allowed_is_named_in_the_status_line_in_the_words_of_the_run(app, state, ohl_layer):
    text, css, disabled, _ = _ends_status(app, {TGT: 500})
    assert css == form_section.STATUS_CLASS_PROBLEM and disabled is False and "too high" in text
    state.gui_settings["constrained"] = {"ends": {TGT: 500}}
    run_notice = _notices(_run(app, ohl_layer.id))[-1]
    assert text == run_notice["meaning"]


def test_the_inputs_are_disabled_with_the_reason_while_the_profile_places_no_towers(app):
    base = {"min_span_m": 5}
    text, css, disabled, title = _ends_status(app, base=base)
    reason = ce.support(backend="cython", profile=base).reason
    assert disabled is True and title == reason and text.startswith(reason) and css == form_section.STATUS_CLASS_OK
    assert _ends_status(app, base={})[2] is True                       # no profile at all: no span limits


def test_the_inputs_are_disabled_with_the_reason_on_a_gpu_backend(app):
    text, css, disabled, title = _ends_status(app, backend="raster_gpu", experimental=True)
    assert disabled is True and text.startswith(title) and "raster_gpu" in title and "cython" in title
    assert css == form_section.STATUS_CLASS_OK                          # nothing is set: information, not an error
    text, css, disabled, _ = _ends_status(app, {SRC: 25}, backend="raster_gpu", experimental=True,
                                          dem="dem.tif", base=WITH_CLEARANCE)
    assert disabled is True and css == form_section.STATUS_CLASS_PROBLEM and "raster_gpu" in text


def test_an_experimental_backend_without_the_flag_counts_as_cython_like_the_run_does(app):
    _, css, disabled, title = _ends_status(app, {SRC: 25}, backend="raster_gpu", experimental=False,
                                           dem="dem.tif", base=WITH_CLEARANCE)
    assert disabled is False and title is None and css == form_section.STATUS_CLASS_OK


def test_the_status_callback_listens_to_heights_backend_flag_dem_and_the_profile(app):
    _key, entry = next((k, e) for k, e in app.callback_map.items() if f"{ids.OHLE_STATUS}.children" in k)
    assert [i["id"] for i in entry["inputs"]] == [ids.OHLE_SETTINGS, ids.OHL_BACKEND, ids.OHL_EXPERIMENTAL,
                                                  ids.OHL_DEM, ids.OHLF_BASE]


# ====================================================================== the line under the rule select
def _rule_status(app, rule="remove", *, backend="cython", experimental=False, base=None):
    resp = invoke(app, RULE_STATUS, rule, backend, experimental, BASE if base is None else base)
    return (resp[ids.OHLE_RULE_STATUS]["children"], resp[ids.OHLE_RULE_STATUS]["className"],
            resp[RULE_ID]["disabled"])


@pytest.mark.parametrize("rule", TERMINAL_SPAN_RULES)
def test_the_line_under_the_select_is_the_help_of_the_chosen_rule(app, rule):
    text, css, disabled = _rule_status(app, rule)
    assert text == ce.RULE_LABELS[rule].help and css == form_section.STATUS_CLASS_OK and disabled is False


def test_a_blank_rule_counts_as_the_default(app):
    assert _rule_status(app, None)[0] == ce.RULE_LABELS["remove"].help


def test_the_select_is_disabled_with_the_reason_while_the_profile_places_no_towers(app):
    text, css, disabled = _rule_status(app, "keep", base={"min_span_m": 5})
    assert disabled is True and text == ce.rule_support({"min_span_m": 5}).reason
    assert css == form_section.STATUS_CLASS_OK and _rule_status(app, base={})[2] is True


def test_a_gpu_backend_says_that_never_placing_an_end_tower_is_refused_there(app):
    text, css, disabled = _rule_status(app, "keep", backend="raster_gpu", experimental=True)
    assert disabled is False and css == form_section.STATUS_CLASS_OK              # another rule works: no error
    assert text.startswith(ce.RULE_LABELS["keep"].help)
    assert "Refused on this backend: Never place one" in text and "raster_gpu" in text
    text, css, disabled = _rule_status(app, "enforce", backend="raster_gpu_v4", experimental=True)
    assert css == form_section.STATUS_CLASS_PROBLEM and disabled is False and "raster_gpu_v4" in text


def test_the_rule_line_of_an_experimental_backend_without_the_flag_has_no_refusal(app):
    text, css, _ = _rule_status(app, "enforce", backend="raster_gpu", experimental=False)
    assert "Refused" not in text and css == form_section.STATUS_CLASS_OK


def test_the_rule_status_callback_listens_to_the_select_backend_flag_and_the_profile(app):
    _key, entry = next((k, e) for k, e in app.callback_map.items() if f"{ids.OHLE_RULE_STATUS}.children" in k)
    assert [i["id"] for i in entry["inputs"]] == [RULE_ID, ids.OHL_BACKEND, ids.OHL_EXPERIMENTAL, ids.OHLF_BASE]


# ====================================================================== clear
def test_clear_puts_both_heights_back_to_blank(app):
    resp = invoke(app, CLEAR, 1, triggered=[f"{ids.OHLE_CLEAR_BTN}.n_clicks"])
    assert resp[SRC_ID]["value"] is None and resp[TGT_ID]["value"] is None


def test_clear_without_a_click_changes_nothing(app):
    with pytest.raises(PreventUpdate):
        invoke(app, CLEAR, None, triggered=[f"{ids.OHLE_CLEAR_BTN}.n_clicks"])


# ====================================================================== restore after Open and New
def _restore(app, current=(None, None), status="opened: p"):
    return invoke(app, RESTORE, status, *current, triggered=[f"{ids.PROJECT_STATUS}.children"])


def test_open_project_puts_the_saved_heights_back_and_opens_the_group(app, state):
    state.gui_settings = {"constrained": {"ends": {SRC: 25.0, TGT: 12.0}}}
    resp = _restore(app)
    assert resp[SRC_ID]["value"] == 25.0 and resp[TGT_ID]["value"] == 12.0 and resp[ids.OHLE_BLOCK]["open"] is True


def test_new_project_puts_the_heights_back_to_blank_and_closes_the_group(app, state):
    state.gui_settings = {}
    resp = _restore(app, (25.0, 12.0), status="new project")
    assert resp[SRC_ID]["value"] is None and resp[TGT_ID]["value"] is None and resp[ids.OHLE_BLOCK]["open"] is False


def test_a_saved_rule_other_than_the_default_opens_the_group_too(app, state):
    state.gui_settings = {"constrained": {"profile": {**BASE, "terminal_span_rule": "keep"}}}
    assert _restore(app, (25.0, None))[ids.OHLE_BLOCK]["open"] is True


def test_a_project_saved_before_the_group_existed_opens_with_blank_heights(app, state):
    state.gui_settings = {"constrained": {"backend": "cython"}}
    assert _restore(app, (30.0, 30.0))[SRC_ID]["value"] is None


def test_a_page_that_already_shows_the_project_is_left_alone(app, state):
    state.gui_settings = {"constrained": {"ends": {SRC: 25.0, TGT: 12.0}}}
    with pytest.raises(PreventUpdate):
        _restore(app, (25.0, 12.0))
    state.gui_settings = {}
    with pytest.raises(PreventUpdate):
        _restore(app, (None, None))


def test_saving_a_project_does_not_rewrite_the_inputs(app, state):
    state.gui_settings = {"constrained": {"ends": {SRC: 25.0, TGT: 12.0}}}
    with pytest.raises(PreventUpdate):
        _restore(app, (None, None), status="saved: C:/p/project.json")


def test_a_saved_value_that_is_no_number_comes_back_as_a_blank_input(app, state):
    state.gui_settings = {"constrained": {"ends": {SRC: "tall", TGT: 12.0}}}
    resp = _restore(app)
    assert resp[SRC_ID]["value"] is None and resp[TGT_ID]["value"] == 12.0


# ====================================================================== first paint and the project file
def test_a_blank_project_prefills_nothing_for_the_group(state):
    puts: list = []
    form_section.prefill_ends({}, lambda *args: puts.append(args))
    form_section.prefill_ends({"ends": {}}, lambda *args: puts.append(args))
    form_section.prefill_ends({"backend": "cython", "experimental": False}, lambda *args: puts.append(args))
    form_section.prefill(state, lambda *args: puts.append(args))
    assert puts == []


def test_a_saved_project_without_heights_or_rule_keeps_the_group_collapsed_and_blank(state):
    state.gui_settings = {"constrained": {"backend": "cython", "experimental": False, "profile": {**BASE}}}
    page = build_layout(state)
    assert _by_id(page, SRC_ID).value is None and _by_id(page, ids.OHLE_BLOCK).open is False
    assert _by_id(page, RULE_ID).value == "remove" and _by_id(page, RULE_ID).disabled is False


def test_a_reloaded_page_shows_the_saved_heights_status_and_an_open_group(state):
    state.gui_settings = {"constrained": {"ends": {SRC: 25.0, TGT: 12.0}, "backend": "cython", "profile": {**BASE}}}
    page = build_layout(state)
    assert _by_id(page, SRC_ID).value == 25.0 and _by_id(page, TGT_ID).value == 12.0
    assert _by_id(page, ids.OHLE_SETTINGS).data == {SRC: 25.0, TGT: 12.0}
    assert _by_id(page, ids.OHLE_BLOCK).open is True
    assert _by_id(page, ids.OHLE_STATUS).children.startswith(ce.summary({SRC: 25.0, TGT: 12.0}))
    assert "reported only" in _by_id(page, ids.OHLE_STATUS).children          # no terrain model saved
    assert _by_id(page, ids.OHLE_FIELDS).disabled is False
    assert _by_id(page, ids.OHL_SETTINGS).data == state.gui_settings["constrained"]


def test_a_reloaded_page_shows_the_saved_rule_enabled_with_its_help_and_an_open_group(state):
    state.gui_settings = {"constrained": {"profile": {**BASE, "terminal_span_rule": "keep"}}}
    page = build_layout(state)
    assert _by_id(page, RULE_ID).value == "keep" and _by_id(page, RULE_ID).disabled is False
    assert _by_id(page, ids.OHLE_RULE_STATUS).children == ce.RULE_LABELS["keep"].help
    assert _by_id(page, ids.OHLE_BLOCK).open is True


def test_a_reloaded_page_disables_saved_heights_that_the_saved_backend_cannot_take(state):
    state.gui_settings = {"constrained": {"ends": {SRC: 25.0}, "backend": "raster_gpu_v4", "experimental": True,
                                          "dem": "dem.tif", "profile": {**WITH_CLEARANCE}}}
    page = build_layout(state)
    fieldset = _by_id(page, ids.OHLE_FIELDS)
    assert fieldset.disabled is True and "raster_gpu_v4" in fieldset.title
    assert _by_id(page, ids.OHLE_STATUS).className == form_section.STATUS_CLASS_PROBLEM
    assert "Refused on this backend: Never place one" in _by_id(page, ids.OHLE_RULE_STATUS).children


def test_a_reloaded_page_disables_the_group_while_the_saved_profile_has_no_span_limits(state):
    state.gui_settings = {"constrained": {"profile": {"min_span_m": 5}}}
    page = build_layout(state)
    reason = ce.support(backend="cython", profile={"min_span_m": 5}).reason
    assert _by_id(page, ids.OHLE_FIELDS).disabled is True and _by_id(page, ids.OHLE_FIELDS).title == reason
    assert _by_id(page, RULE_ID).disabled is True
    assert _by_id(page, ids.OHLE_RULE_STATUS).children == ce.rule_support({"min_span_m": 5}).reason


def test_the_heights_survive_save_and_open_and_show_on_the_new_page(app, state, tmp_path):
    invoke(app, MIRROR, 25, 12)
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings["constrained"]["ends"] == {SRC: 25.0, TGT: 12.0}
        other.gui_settings = json.loads(json.dumps(other.gui_settings))
        other_page = build_layout(other)
        assert _by_id(other_page, SRC_ID).value == 25.0 and _by_id(other_page, TGT_ID).value == 12.0
    finally:
        other.shutdown()


def test_a_project_without_heights_is_saved_without_an_ends_key(app, state, tmp_path):
    invoke(app, PROFILE_MIRROR, "cython", False, None, None, yaml.safe_dump(BASE))
    invoke(app, MIRROR, None, None)
    path = project_io.save_project(state, tmp_path / "proj")
    manifest = json.loads(project_io.manifest_path(path).read_text(encoding="utf-8"))
    assert "ends" not in manifest["gui_settings"]["constrained"]


# ====================================================================== the run: a classic run is as before
def test_a_classic_run_calls_find_route_without_a_keyword_and_stores_nothing_new(app, state, ohl_layer, route_calls):
    resp = _run(app, ohl_layer.id)
    assert route_calls == [{}]
    assert _routes(state), _notices(resp)
    route = _routes(state)[0]
    assert "connection_heights" not in route.meta["params"]
    assert "connection_heights" not in route.meta["constrained_summary"]
    assert "connection heights" not in resp[ids.OHL_STATUS]["children"]
    assert "Connection heights" not in str(_notices(resp))


@pytest.mark.parametrize("held", [None, {}, {SRC: None, TGT: None}, {SRC: "", TGT: "  "}])
def test_no_height_set_passes_nothing_whatever_the_state_holds(app, state, ohl_layer, route_calls, held):
    if held is not None:
        state.gui_settings["constrained"] = {"ends": held}
    _run(app, ohl_layer.id)
    assert route_calls == [{}]


def test_a_classic_background_run_has_no_ends_in_its_snapshot_or_finalize(app, state, ohl_layer, heavy):
    _run(app, ohl_layer.id)
    job = state.jobs[constrained.JOB_KIND]
    assert "ends" not in job.params["run"] and "connection_heights" not in job.params["finalize"]
    job.thread.join(JOIN_S)


# ====================================================================== the run: heights reach the library
def test_heights_reach_the_library_in_a_synchronous_run_and_are_stored_and_named(
        app, state, ohl_layer, dem_file, route_calls):
    _heights(state, source=20, target=12)
    resp = _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file)
    assert state.jobs == {} and ids.OHL_POLL not in resp
    assert route_calls == [{SRC: 20.0, TGT: 12.0}]
    route = _routes(state)[0]
    assert route.meta["params"]["connection_heights"] == {SRC: 20.0, TGT: 12.0}
    summary = route.meta["constrained_summary"]
    assert summary["connection_heights"] == {SRC: 20.0, TGT: 12.0} and summary[SRC] == 20.0 and summary[TGT] == 12.0
    assert route.meta["params"]["constrained"] is True and "profile" in route.meta["params"]
    status = resp[ids.OHL_STATUS]["children"]
    assert status.startswith("OHL: ") and "; connection heights: source 20 m, target 12 m" in status
    success = _notice(resp, "Overhead line routed")
    assert "Connection heights: source 20 m, target 12 m" in success["meaning"]
    assert _notice(resp, "Line ends: notes on this route")["meaning"].count("exact span bins") >= 1


def test_heights_reach_the_library_in_a_background_job(app, state, ohl_layer, dem_file, route_calls, heavy):
    _heights(state, source=20, target=12)
    start = _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file)
    job = state.jobs[constrained.JOB_KIND]
    assert job.params["run"]["ends"] == {SRC: 20.0, TGT: 12.0}
    assert job.params["finalize"]["connection_heights"] == {SRC: 20.0, TGT: 12.0}
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive()
    done = poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    assert route_calls == [{SRC: 20.0, TGT: 12.0}]
    route = _routes(state)[0]
    assert route.meta["params"]["connection_heights"] == {SRC: 20.0, TGT: 12.0}
    assert "; connection heights: source 20 m, target 12 m" in done[ids.OHL_STATUS]["children"]


def test_the_job_keeps_the_heights_of_its_start_when_the_page_changes_them_afterwards(
        app, state, ohl_layer, dem_file, heavy):
    _heights(state, source=20, target=12)
    start = _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file)
    job = state.jobs[constrained.JOB_KIND]
    state.gui_settings["constrained"]["ends"][TGT] = 99.0                    # the user edits while it runs
    job.thread.join(JOIN_S)
    poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    assert _routes(state)[0].meta["params"]["connection_heights"] == {SRC: 20.0, TGT: 12.0}


def test_the_stored_heights_do_not_change_when_the_state_is_changed_later(app, state, ohl_layer, dem_file):
    _heights(state, source=20)
    _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file)
    state.gui_settings["constrained"]["ends"][SRC] = 5.0
    assert _routes(state)[0].meta["params"]["connection_heights"] == {SRC: 20.0}


def test_a_waypoint_run_gives_the_source_height_to_the_first_segment_and_the_target_height_to_the_last(
        app, state, ohl_layer, dem_file, route_calls):
    _heights(state, source=20, target=12)
    _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file, waypoints=[(30.0, 30.0)])
    assert route_calls == [{SRC: 20.0, TGT: None}, {SRC: None, TGT: 12.0}]
    assert len(_routes(state)) == 1


def test_a_height_without_a_terrain_model_is_reported_and_the_route_is_the_one_without_it(
        app, state, ohl_layer, route_calls):
    plain = _run(app, ohl_layer.id, profile=WITH_CLEARANCE)
    _heights(state, source=20)
    resp = _run(app, ohl_layer.id, profile=WITH_CLEARANCE)
    assert route_calls == [{}, {SRC: 20.0, TGT: None}]
    first, second = _routes(state)
    assert first.gdf.geometry.iloc[0].equals(second.gdf.geometry.iloc[0])
    assert second.meta["params"]["connection_heights"] == {SRC: 20.0}
    assert "connection heights: source 20 m" in resp[ids.OHL_STATUS]["children"]
    assert "connection heights" not in plain[ids.OHL_STATUS]["children"]
    assert "A connection height does not change the route" in _titles(resp)    # the library's warning, plainly


# ====================================================================== the run: problems start nothing
@pytest.mark.parametrize(("key", "value", "control", "fragment"), [
    (SRC, 0, SRC_ID, "too low"), (SRC, 301, SRC_ID, "too high"), (TGT, -5, TGT_ID, "too low"),
    (TGT, "abc", TGT_ID, "not a finite number")])
def test_a_height_that_is_not_allowed_is_an_error_notice_that_focuses_the_field_and_starts_nothing(
        app, state, ohl_layer, route_calls, key, value, control, fragment):
    state.gui_settings["constrained"] = {"ends": {key: value}}
    resp = _run(app, ohl_layer.id)
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and notice["focus_control"] == control
    assert notice["focus_id"] == ids.TAB_ROUTES and fragment in notice["meaning"]
    assert ce.FIELD_SPECS[key].label in notice["title"] and "No overhead-line route was computed" in notice["impact"]
    _nothing_started(state, route_calls)
    assert ids.OHL_POLL not in resp and ids.OHL_STOP_BTN not in resp


def test_a_height_that_is_not_allowed_starts_no_background_job_either(app, state, ohl_layer, route_calls, heavy):
    _heights(state, target=500)
    resp = _run(app, ohl_layer.id)
    assert _notices(resp)[-1]["focus_control"] == TGT_ID
    _nothing_started(state, route_calls)


def test_the_enforce_rule_on_a_gpu_backend_is_refused_and_focuses_the_backend(app, state, ohl_layer, route_calls):
    resp = _run(app, ohl_layer.id, profile=ENFORCE, backend="raster_gpu", experimental=True)
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and notice["focus_control"] == ids.OHL_BACKEND
    assert notice["title"] == "Line ends: this backend cannot run them"
    assert "raster_gpu" in notice["meaning"] and "'enforce'" in notice["meaning"]
    _nothing_started(state, route_calls)


def test_the_enforce_rule_on_a_gpu_backend_is_refused_in_the_background_path_too(
        app, state, ohl_layer, route_calls, heavy):
    resp = _run(app, ohl_layer.id, profile=ENFORCE, backend="raster_gpu_v4", experimental=True)
    assert _notices(resp)[-1]["focus_control"] == ids.OHL_BACKEND
    _nothing_started(state, route_calls)


def test_a_height_the_search_uses_on_a_gpu_backend_is_refused_and_focuses_the_backend(
        app, state, ohl_layer, dem_file, route_calls):
    _heights(state, source=20)
    resp = _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file, backend="raster_gpu", experimental=True)
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.OHL_BACKEND and "source height 20 m" in notice["meaning"]
    _nothing_started(state, route_calls)


def test_a_backend_that_the_run_resets_to_cython_takes_the_enforce_rule(app, state, ohl_layer, route_calls):
    """Without the experimental flag the run uses cython (the gate), so the rule is fine."""
    resp = _run(app, ohl_layer.id, profile=ENFORCE, backend="raster_gpu", experimental=False)
    assert "Backend reset to cython" in _titles(resp) and len(_routes(state)) == 1
    assert route_calls == [{}]


def _small_limit(monkeypatch, states):
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_STATES", states)


def test_an_exact_search_that_would_not_fit_is_a_notice_with_the_numbers_that_focuses_the_rule_and_starts_nothing(
        app, state, ohl_layer, route_calls, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    resp = _run(app, ohl_layer.id, profile=ENFORCE)
    notice = _notices(resp)[-1]
    assert notice["severity"] == "error" and notice["title"] == "Line ends: the exact search is too large"
    assert notice["focus_control"] == RULE_ID and notice["focus_id"] == ids.TAB_ROUTES
    est = ce.estimate_exact_bins(60, 60, max_span_m=20, cell_size_m=1.0, neighborhood="r1")
    assert f"{est.states:,}" in notice["meaning"] and "200,000" in notice["meaning"]
    assert "60 x 60 cells" in notice["meaning"] and "coarser raster" in notice["fix"]
    _nothing_started(state, route_calls)


def test_the_size_problem_stops_a_background_run_too(app, state, ohl_layer, route_calls, heavy, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    resp = _run(app, ohl_layer.id, profile=ENFORCE)
    assert _notices(resp)[-1]["title"] == "Line ends: the exact search is too large"
    _nothing_started(state, route_calls)


def test_the_size_problem_of_a_height_focuses_that_height(app, state, ohl_layer, dem_file, route_calls, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    _heights(state, target=15)
    resp = _run(app, ohl_layer.id, profile=WITH_CLEARANCE, dem=dem_file)
    notice = _notices(resp)[-1]
    assert notice["title"] == "Line ends: the exact search is too large" and notice["focus_control"] == TGT_ID
    _nothing_started(state, route_calls)


def test_the_limit_only_concerns_the_exact_search_a_default_run_goes_on(app, state, ohl_layer, monkeypatch):
    _small_limit(monkeypatch, 10)
    for rule in ("remove", "remove_if_valid", "keep"):
        resp = _run(app, ohl_layer.id, profile={**BASE, "terminal_span_rule": rule})
        assert "Line ends: the exact search is too large" not in _titles(resp)
    assert len(_routes(state)) == 3


def test_the_ends_focus_controls_follow_the_cause():
    controls = form_section.ends_controls({SRC: 20}, WITH_CLEARANCE, "dem.tif")
    assert controls["bins"] == SRC_ID and controls["rule"] == RULE_ID and controls["backend"] == ids.OHL_BACKEND
    assert controls[SRC] == SRC_ID and controls[TGT] == TGT_ID
    assert form_section.ends_controls({TGT: 20}, WITH_CLEARANCE, "dem.tif")["bins"] == TGT_ID
    assert form_section.ends_controls({SRC: 20}, {**WITH_CLEARANCE, "terminal_span_rule": "enforce"},
                                      "dem.tif")["bins"] == RULE_ID
    assert form_section.ends_controls({}, ENFORCE, None)["bins"] == RULE_ID
    assert set(controls) == set(ce.PROBLEM_KEYS)


def test_the_run_callback_gained_no_state_and_no_input(app):
    spec = next(s for s in app._callback_list  # noqa: SLF001
                if f"{ids.OHL_STATUS}.children" in str(s["output"]) and ids.RUN_ROUTING_BTN in str(s["inputs"]))
    assert [i["id"] for i in spec["inputs"]] == [ids.RUN_ROUTING_BTN]
    assert len(spec["state"]) == 12


# ====================================================================== the result: what is said about the line ends
def test_a_classic_run_that_lost_an_end_tower_says_so_in_status_notice_and_notes(app, state, corridor_raster):
    layer = _layer(state, corridor_raster)
    resp = _run(app, layer.id, profile=CORRIDOR_PROFILE, points=CORRIDOR_POINTS)
    summary = _routes(state)[0].meta["constrained_summary"]
    assert summary["removed_towers"]["count"] == 1 and "terminal_span_rule" not in summary
    status = resp[ids.OHL_STATUS]["children"]
    assert status.endswith("; 1 end tower removed") and "end-span rule" not in status
    success = _notice(resp, "Overhead line routed")
    assert "Removed end tower: 1 (cost 150, not part of the route's cost)" in success["meaning"]
    assert "closer to a line end than the shortest span" in success["meaning"]
    notes = _notice(resp, "Line ends: notes on this route")
    assert "removed the tower at (21.5, 1.5)" in notes["meaning"] and notes["severity"] == "info"
    assert notes["focus_control"] == ids.OHLE_BLOCK and notes["focus_id"] == ids.TAB_ROUTES


def test_the_span_rule_warning_of_that_run_has_its_own_plain_box(app, state, corridor_raster):
    layer = _layer(state, corridor_raster)
    resp = _run(app, layer.id, profile=CORRIDOR_PROFILE, points=CORRIDOR_POINTS)
    box = _notice(resp, "The route breaks a span or clearance rule of its profile")
    assert box["severity"] == "warning" and "terminal_span_rule='remove'" in box["details"]
    assert box["focus_id"] == ids.TAB_ROUTES and "Line ends" in box["fix"]


def test_the_keep_rule_is_named_and_flags_the_short_end_span_without_a_removed_tower(app, state, corridor_raster):
    layer = _layer(state, corridor_raster)
    resp = _run(app, layer.id, profile={**CORRIDOR_PROFILE, "terminal_span_rule": "keep"}, points=CORRIDOR_POINTS)
    status = resp[ids.OHL_STATUS]["children"]
    assert "; end-span rule: Keep it" in status and "removed" not in status
    success = _notice(resp, "Overhead line routed")
    assert f"End-span rule: {ce.rule_label('keep')}." in success["meaning"]
    assert "Removed end tower" not in success["meaning"]
    assert "kept by terminal_span_rule='keep'" in _notice(resp, "Line ends: notes on this route")["meaning"]
    assert "The route breaks a span or clearance rule of its profile" not in _titles(resp)


def test_the_enforce_rule_is_named_with_the_exact_search_in_status_and_notes(app, state, corridor_raster):
    layer = _layer(state, corridor_raster)
    resp = _run(app, layer.id, profile={**CORRIDOR_PROFILE, "terminal_span_rule": "enforce"}, points=CORRIDOR_POINTS)
    route = _routes(state)[0]
    assert route.meta["params"]["profile"]["terminal_span_rule"] == "enforce"
    summary = route.meta["constrained_summary"]
    assert summary["terminal_span_rule"] == "enforce" and summary["span_bin_size_m"] < 1.0
    assert "; end-span rule: Never place one" in resp[ids.OHL_STATUS]["children"]
    assert "exact span bins" in _notice(resp, "Line ends: notes on this route")["meaning"]
    assert summary["spans_min_max_avg"][0] >= 3.0 - 1e-6                      # every span, the end spans too


def test_the_background_run_names_the_line_ends_like_the_synchronous_one(app, state, corridor_raster, heavy):
    layer = _layer(state, corridor_raster)
    start = _run(app, layer.id, profile={**CORRIDOR_PROFILE, "terminal_span_rule": "keep"}, points=CORRIDOR_POINTS)
    state.jobs[constrained.JOB_KIND].thread.join(JOIN_S)
    done = poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    assert "; end-span rule: Keep it" in done[ids.OHL_STATUS]["children"]
    assert "Line ends: notes on this route" in [n["title"] for n in done[ids.NOTICES]["data"]]


def test_a_route_without_line_end_facts_has_no_readout():
    summary = {"n_towers": 3, "spans_min_max_avg": (7.0, 9.0, 8.0)}
    assert run_callbacks.end_status_text(summary) == "" and run_callbacks.end_facts(summary) == []
    assert run_callbacks.end_notes_notice(summary) is None
    assert run_callbacks.end_notes_notice({"span_notes": [], "search_notes": []}) is None


def test_the_readout_names_every_fact_the_library_reports():
    summary = {"terminal_span_rule": "keep", SRC: 25.0, TGT: 12.5, "removed_towers": {"count": 2, "cost": 1234.0},
               "height_premiums": 9000.0, "span_notes": ["one note"], "search_notes": ["searched exactly"]}
    assert run_callbacks.end_status_text(summary) == (
        "; end-span rule: Keep it; connection heights: source 25 m, target 12.5 m; 2 end towers removed")
    facts = run_callbacks.end_facts(summary)
    assert facts[0] == f"End-span rule: {ce.rule_label('keep')}."
    assert facts[1] == "Connection heights: source 25 m, target 12.5 m (metres above the ground at the end point)."
    assert facts[2].startswith("Removed end towers: 2 (cost 1,234, not part of the route's cost). Why: ")
    assert facts[3].startswith("Height premiums: 9,000 (")
    notice = run_callbacks.end_notes_notice(summary)
    assert notice["meaning"] == "one note searched exactly" and notice["severity"] == "info"


@pytest.mark.parametrize(("rule", "reason"), [
    (None, "the rule removes such a tower afterwards"), ("remove", "the rule removes such a tower afterwards"),
    ("remove_if_valid", "the line stays valid without it")])
def test_a_removed_end_tower_is_explained_by_the_rule_in_force(rule, reason):
    summary = {"removed_towers": {"count": 1, "cost": 150.0}, **({"terminal_span_rule": rule} if rule else {})}
    (fact,) = [f for f in run_callbacks.end_facts(summary) if f.startswith("Removed end tower: 1")]
    assert reason in fact


def test_one_single_height_is_named_alone():
    assert run_callbacks.end_status_text({TGT: 12.0}) == "; connection heights: target 12 m"
    assert run_callbacks.end_facts({SRC: 20.0}) == [
        "Connection heights: source 20 m (metres above the ground at the end point)."]


# ====================================================================== no route under the rule or the heights
def test_no_route_under_the_enforce_rule_is_a_notice_that_names_the_rule_not_the_generic_box(
        app, state, wall_raster):
    layer = _layer(state, wall_raster)
    resp = _run(app, layer.id, profile=ENFORCE)
    notice = _notices(resp)[-1]
    assert notice["title"] == "No route satisfies the end-span rule" and notice["severity"] == "error"
    assert "No route could be found" not in _titles(resp) and "Something went wrong" not in _titles(resp)
    assert "terminal_span_rule='enforce'" in notice["details"] and "no route exists" in notice["details"]
    assert "Remove it" in notice["fix"] and notice["focus_id"] == ids.TAB_ROUTES
    assert _routes(state) == []


def test_no_route_under_the_enforce_rule_in_a_background_job_is_the_same_notice(app, state, wall_raster, heavy):
    layer = _layer(state, wall_raster)
    start = _run(app, layer.id, profile=ENFORCE)
    state.jobs[constrained.JOB_KIND].thread.join(JOIN_S)
    done = poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    assert done[ids.NOTICES]["data"][-1]["title"] == "No route satisfies the end-span rule"
    assert _routes(state) == []


def test_the_generic_box_of_a_search_without_the_line_end_settings_stays(app, state, wall_raster, monkeypatch):
    """A restricted search is the only one that names the rule; every other empty search keeps the old box."""
    layer = _layer(state, wall_raster)
    monkeypatch.setattr(constrained, "run_constrained_path", lambda *_args, **_kwargs: None)
    resp = _run(app, layer.id)
    assert _notices(resp)[-1]["title"] == "No route could be found" and _routes(state) == []

"""
Round 14, package P0 (scaffold): pins the contract the later packages build on.

P0 adds, without any logic: every new component id (``ids.py``), the stores,
poll timers and Stop buttons in the page, the empty section / callback modules
that P2 to P12 fill, ``ProjectState.gui_settings / jobs / loaded_cost_table /
corridor_graphs``, the ``gui_settings`` entry of the project file, and the two
first-paint steps ``_prefill_sections`` and ``_prefill_jobs``. A later package
that breaks one of these breaks its neighbours, so each is pinned here.
"""
from __future__ import annotations

import inspect
import json
import logging
import threading
import types

import pytest
from dash import Dash, dcc, html

from pyorps.gui import ids, layout
from pyorps.gui.layout import build_layout
from pyorps.gui.services import project_io
from pyorps.gui.state import ProjectState

# =========================================================================== ids
#: every constant P0 adds, with its value. ``OBJ_LAYERS_GRID`` differs from the
#: written plan on purpose: "obj-mlayers-grid" contains "layers-grid", the id
#: that existing tests use as a fragment to find callbacks.
PLANNED_IDS: dict[str, str] = {
    # objective & terrain
    "OBJ_CARD": "obj-card", "OBJ_PRESET_CLASSIC_BTN": "obj-preset-classic",
    "OBJ_PRESET_SHORTEST_BTN": "obj-preset-shortest", "OBJ_PRESET_GENTLEST_BTN": "obj-preset-gentlest",
    "OBJ_W_COST": "obj-w-cost", "OBJ_W_LENGTH": "obj-w-length", "OBJ_W_GRADIENT": "obj-w-gradient",
    "OBJ_W_GRADIENT_WRAP": "obj-w-gradient-wrap", "OBJ_LAYER_NAME": "obj-mlayer-name",
    "OBJ_LAYER_SOURCE": "obj-mlayer-source", "OBJ_LAYER_WEIGHT": "obj-mlayer-weight",
    "OBJ_LAYER_ADD_BTN": "obj-mlayer-add", "OBJ_LAYER_DEL_BTN": "obj-mlayer-del",
    "OBJ_LAYERS_GRID": "obj-mlayer-grid", "OBJ_DEM_LAYER": "obj-dem-layer", "OBJ_SLOPE_WRAP": "obj-slope-wrap",
    "OBJ_GRAD_MAX_PCT": "obj-grad-max-pct", "OBJ_GRAD_ADDITIVE": "obj-grad-additive",
    "OBJ_GRAD_ADDITIVE_SCALE": "obj-grad-additive-scale", "OBJ_GRAD_MULTIPLIER": "obj-grad-multiplier",
    "OBJ_GRAD_MULT_SCALE": "obj-grad-mult-scale", "OBJ_GRAD_SIG_CENTER": "obj-grad-sig-center",
    "OBJ_GRAD_SIG_STEEPNESS": "obj-grad-sig-steepness", "OBJ_GRAD_SIG_MAX": "obj-grad-sig-max",
    "OBJ_GRAD_BIN_PCT": "obj-grad-bin-pct", "OBJ_GRAD_SMAX_PCT": "obj-grad-smax-pct",
    "OBJ_PRECISION": "obj-precision", "OBJ_PRECISION_WRAP": "obj-precision-wrap", "OBJ_STATUS": "obj-status",
    "OBJ_SETTINGS": "obj-settings",
    # cost field
    "CF_CARD": "cf-card", "CF_ORIGINS": "cf-origins", "CF_COMBINE": "cf-combine", "CF_ALGORITHM": "cf-algorithm",
    "CF_THREADS": "cf-threads", "CF_RUN_BTN": "cf-run-btn", "CF_STOP_BTN": "cf-stop-btn", "CF_POLL": "cf-poll",
    "CF_STATUS": "cf-status", "CF_BEST_INFO": "cf-best-info", "CF_SETTINGS": "cf-settings",
    # corridor graph
    "COR_CARD": "cor-card", "COR_TERMINALS": "cor-terminals", "COR_K": "cor-k", "COR_MIN_SHARED": "cor-min-shared",
    "COR_MAX_PAIR_COST": "cor-max-pair-cost", "COR_VALIDATE": "cor-validate", "COR_RUN_BTN": "cor-run-btn",
    "COR_POLL": "cor-poll", "COR_STATUS": "cor-status", "COR_SAVE_PATH": "cor-save-path",
    "COR_SAVE_BTN": "cor-save-btn", "COR_SETTINGS": "cor-settings",
    # profile form
    "OHLF_ANGLE_GRID": "ohlf-angle-grid", "OHLF_ANGLE_ADD_BTN": "ohlf-angle-add",
    "OHLF_ANGLE_DEL_BTN": "ohlf-angle-del", "OHLF_TERRAIN_GRID": "ohlf-terrain-costs-grid",
    "OHLF_TERRAIN_ADD_BTN": "ohlf-terrain-add", "OHLF_TERRAIN_DEL_BTN": "ohlf-terrain-del",
    "OHLF_TYPES_GRID": "ohlf-types-grid", "OHLF_TYPES_ADD_BTN": "ohlf-types-add",
    "OHLF_TYPES_DEL_BTN": "ohlf-types-del", "OHLF_FROM_YAML_BTN": "ohlf-from-yaml", "OHLF_STATUS": "ohlf-status",
    "OHLF_CLEARANCE_INFO": "ohlf-clearance-info", "OHLF_BASE": "ohlf-base",
    "OHLF_YAML_DETAILS": "ohlf-yaml-details", "OHL_SETTINGS": "ohlf-settings",
    # background jobs
    "RASTERIZE_STOP_BTN": "rasterize-stop-btn", "RASTERIZE_POLL": "rasterize-poll",
    "OHL_STOP_BTN": "ohl-stop-btn", "OHL_POLL": "ohl-poll",
}

#: the 24 scalar fields of the profile form, in form order (gap 5 table; round 17 added the end-span rule)
OHLF_KEYS = (
    "name", "description", "soft_angle_limit_deg", "hard_angle_limit_deg", "angle_cost_function",
    "angle_cost_scale", "min_span_m", "max_span_m", "span_bin_size_m", "terrain_interpolation",
    "max_gradient_percent", "gradient_scale", "tower_height_m", "tower_heights_m",
    "tower_height_cost_per_increment", "terminal_tower_height_m", "terminal_tower_cost",
    "conductor_weight_per_m", "conductor_tension_n", "min_clearance_m", "tower_ground_area_m2",
    "tower_area_cost_mode", "recommended_buffer_m", "terminal_span_rule",
)

#: fragments of ids that existing tests and callbacks match by substring
EXISTING_FRAGMENTS = ("routing-poll", "routing-stop-btn", "run-routing-btn", "ohl-status", "rasterize-btn",
                      "rasterize-log", "layers-grid")

#: the new stores / timers / buttons that must be in the page exactly once
NEW_STORES = (ids.OBJ_SETTINGS, ids.CF_SETTINGS, ids.COR_SETTINGS, ids.OHL_SETTINGS, ids.OHLF_BASE)
NEW_POLLS = (ids.RASTERIZE_POLL, ids.OHL_POLL, ids.CF_POLL, ids.COR_POLL)


def _old_id_values() -> list[str]:
    """Every id value of ``ids.py`` that P0 did not add (pattern-type names excluded)."""
    return [v for k, v in vars(ids).items()
            if k.isupper() and isinstance(v, str) and k not in PLANNED_IDS and not k.startswith("TYPE_")]


def test_every_planned_id_exists_with_its_value():
    missing = [k for k in PLANNED_IDS if not hasattr(ids, k)]
    assert missing == []
    wrong = {k: getattr(ids, k) for k, v in PLANNED_IDS.items() if getattr(ids, k) != v}
    assert wrong == {}


def test_planned_values_are_unique_and_new():
    new_values = list(PLANNED_IDS.values()) + list(ids.OHLF_IDS.values())
    assert len(new_values) == len(set(new_values)), "a new id value is used twice"
    old = _old_id_values()
    assert not set(new_values) & set(old), "a new id value is an older id"
    all_values = [v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str)]
    assert len(all_values) == len(set(all_values)), "ids.py holds one value under two names"


def test_no_new_value_contains_an_older_id_or_one_of_the_fragments_tests_use():
    """The harness finds a callback by substring, so a new id must not contain an old one."""
    new_values = list(PLANNED_IDS.values()) + list(ids.OHLF_IDS.values())
    swallowed = {v: [o for o in _old_id_values() if o in v] for v in new_values}
    assert {v: o for v, o in swallowed.items() if o} == {}
    for fragment in EXISTING_FRAGMENTS:
        assert [v for v in new_values if fragment in v] == [], fragment


@pytest.mark.parametrize(("name", "value"), sorted(PLANNED_IDS.items()))
def test_each_planned_id_carries_its_package_prefix(name, value):
    if name.startswith("OBJ_"):
        assert value.startswith("obj-")
    elif name.startswith("CF_"):
        assert value.startswith("cf-")
    elif name.startswith("COR_"):
        assert value.startswith("cor-")
    elif name.startswith("OHLF_") or name == "OHL_SETTINGS":
        assert value.startswith("ohlf-")
    else:
        assert name in {"RASTERIZE_STOP_BTN", "RASTERIZE_POLL", "OHL_STOP_BTN", "OHL_POLL"}


def test_the_profile_form_keys_and_their_ids():
    assert ids.OHLF_KEYS == OHLF_KEYS
    assert len(ids.OHLF_KEYS) == 24 and ids.OHLF_KEYS[-1] == "terminal_span_rule"
    assert tuple(ids.OHLF_IDS) == ids.OHLF_KEYS
    assert ids.OHLF_IDS == {k: "ohlf-" + k.replace("_", "-") for k in OHLF_KEYS}
    assert ids.OHLF_IDS["soft_angle_limit_deg"] == "ohlf-soft-angle-limit-deg"
    assert ids.OHLF_IDS["terminal_span_rule"] == "ohlf-terminal-span-rule"
    assert len(set(ids.OHLF_IDS.values())) == 24


# ===================================================================== the page
def _children(component) -> list:
    kids = getattr(component, "children", None)
    if kids is None:
        return []
    if isinstance(kids, (list, tuple)):
        return [k for k in kids if hasattr(k, "_traverse")]
    return [kids] if hasattr(kids, "_traverse") else []


def _walk(component):
    yield component
    for child in _children(component):
        yield from _walk(child)


def _has_id(root, component_id: str) -> bool:
    return any(getattr(c, "id", None) == component_id for c in _walk(root))


def _find(root, component_id: str):
    found = [c for c in _walk(root) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id!r} is in the layout {len(found)} times"
    return found[0]


def _parent(root, component):
    parents = [c for c in _walk(root) if any(k is component for k in _children(c))]
    assert len(parents) == 1
    return parents[0]


def _index(items: list, item) -> int:
    return next(i for i, other in enumerate(items) if other is item)


def _ordered_ids(root) -> list[str]:
    return [c.id for c in _walk(root) if isinstance(getattr(c, "id", None), str)]


@pytest.fixture()
def page(state):
    return build_layout(state)


def test_no_component_id_is_used_twice(page):
    page_ids = _ordered_ids(page)
    assert len(page_ids) == len(set(page_ids))
    assert set(page_ids) == {c.id for c in page._traverse() if isinstance(getattr(c, "id", None), str)}  # noqa: SLF001


@pytest.mark.parametrize("store_id", NEW_STORES)
def test_the_new_stores_are_in_the_page_once_and_empty(page, store_id):
    store = _find(page, store_id)
    assert isinstance(store, dcc.Store)
    assert store.data == {}
    assert _parent(page, store) is page             # a direct child, like every other store


@pytest.mark.parametrize("poll_id", NEW_POLLS)
def test_the_new_poll_timers_are_off_and_run_at_800_ms(page, poll_id):
    poll = _find(page, poll_id)
    assert isinstance(poll, dcc.Interval)
    assert poll.disabled is True
    assert poll.interval == 800
    assert _parent(page, poll) is page


def test_the_routing_poll_is_unchanged(page):
    poll = _find(page, ids.ROUTING_POLL)
    assert poll.disabled is True and poll.interval == 600


@pytest.mark.parametrize("stop_id", [ids.RASTERIZE_STOP_BTN, ids.OHL_STOP_BTN])
def test_the_new_stop_buttons_are_hidden_danger_buttons(page, stop_id):
    stop = _find(page, stop_id)
    assert stop.style == {"display": "none"}
    assert stop.color == "danger"
    assert "Stop" in stop.children
    assert stop.title


def test_the_rasterize_stop_button_sits_beside_the_build_button(page):
    build = _find(page, ids.RASTERIZE_BTN)
    stop = _find(page, ids.RASTERIZE_STOP_BTN)
    row = _parent(page, build)
    assert _parent(page, stop) is row
    classes = row.className.split()
    assert "d-flex" in classes and "gap-2" in classes
    assert [getattr(c, "id", None) for c in _children(row)] == [ids.RASTERIZE_BTN, ids.RASTERIZE_STOP_BTN]
    assert _has_id(_parent(page, row), ids.RASTERIZE_LOG)       # in the Rasterization card


def test_the_constrained_stop_button_comes_right_before_the_status_line(page):
    options = _find(page, ids.OHL_COLLAPSE)
    stop = _find(page, ids.OHL_STOP_BTN)
    status = _find(page, ids.OHL_STATUS)
    assert _parent(page, stop) is options and _parent(page, status) is options
    siblings = _children(options)
    assert _index(siblings, status) == _index(siblings, stop) + 1


def test_the_yaml_text_is_still_there_once_inside_raw_yaml_advanced(page):
    textarea = _find(page, ids.OHL_PROFILE_TEXT)
    assert textarea.style["height"] == "160px" and textarea.style["fontFamily"] == "monospace"
    details = _parent(page, textarea)
    assert details.id == ids.OHLF_YAML_DETAILS
    assert isinstance(details, html.Details) and isinstance(details.open, bool)   # closed once the form exists
    summary = _children(details)[0]
    assert isinstance(summary, html.Summary) and summary.children == "Raw YAML (advanced)"
    label = next(c for c in _children(details) if getattr(c, "html_for", None))
    assert label.html_for == ids.OHL_PROFILE_TEXT
    options = _find(page, ids.OHL_COLLAPSE)
    assert _parent(page, details) is options
    siblings = _children(options)
    before = siblings[:_index(siblings, details)]           # after the "Profile file" input group (and the form)
    assert any(_has_id(child, ids.OHL_PROFILE_PATH) and _has_id(child, ids.OHL_PROFILE_SAVE_BTN)
               for child in before)


def _marker(name: str) -> html.Div:
    return html.Div(id=f"marker-{name}")


def test_section_cards_are_spliced_into_the_routes_tab_in_the_planned_places(monkeypatch, state):
    from pyorps.gui.sections import corridor, cost_field, objective

    monkeypatch.setattr(objective, "section", lambda: [_marker("objective")])
    monkeypatch.setattr(cost_field, "section", lambda: [_marker("cost-field")])
    monkeypatch.setattr(corridor, "section", lambda: [_marker("corridor")])
    page = build_layout(state)
    tab = _parent(page, _find(page, "marker-objective"))
    assert _has_id(tab, ids.RUN_ROUTING_BTN) and _has_id(tab, ids.OHL_COLLAPSE)    # the Routes tab
    cards = _children(tab)

    def at(component_id: str) -> int:
        return next(i for i, card in enumerate(cards) if _has_id(card, component_id))

    algorithm, constrained, computed = at(ids.GRAPH_API), at(ids.OHL_COLLAPSE), at(ids.EDIT_ROUTE_SELECT)
    assert at("marker-objective") == algorithm + 1
    assert constrained == algorithm + 2
    assert at("marker-cost-field") == constrained + 1
    assert at("marker-corridor") == constrained + 2
    assert computed == constrained + 3
    for marker in ("marker-objective", "marker-cost-field", "marker-corridor"):
        assert _parent(page, _find(page, marker)) is tab


def test_the_profile_form_is_spliced_between_the_profile_file_row_and_the_yaml_block(monkeypatch, state):
    from pyorps.gui.sections import profile_form

    monkeypatch.setattr(profile_form, "section", lambda: [_marker("form")])
    page = build_layout(state)
    form = _find(page, "marker-form")
    options = _find(page, ids.OHL_COLLAPSE)
    assert _parent(page, form) is options
    siblings = _children(options)
    position = _index(siblings, form)
    assert _has_id(siblings[position - 1], ids.OHL_PROFILE_PATH)
    assert siblings[position + 1].id == ids.OHLF_YAML_DETAILS


# ================================================================= stub modules
SECTION_MODULES = ("objective", "cost_field", "corridor", "profile_form")
CALLBACK_MODULES = ("objective", "cost_field", "corridor", "profile_form", "raster_job")


@pytest.mark.parametrize("name", SECTION_MODULES)
def test_a_section_module_keeps_the_scaffold_interface(name, state):
    """The scaffold's stubs returned [] and prefilled nothing; the packages that fill them keep the interface."""
    import importlib

    module = importlib.import_module(f"pyorps.gui.sections.{name}")
    assert isinstance(module.section(), list)       # spliced into the Routes tab with *section()
    assert list(inspect.signature(module.prefill).parameters) == ["state", "put"]
    assert module.prefill(state, lambda *args: None) is None


def test_the_profile_form_module_also_owns_the_raw_yaml_block():
    from pyorps.gui.sections import profile_form

    block = profile_form.yaml_block()
    assert len(block) == 1 and isinstance(block[0], html.Details) and block[0].id == ids.OHLF_YAML_DETAILS
    assert _has_id(block[0], ids.OHL_PROFILE_TEXT)


@pytest.mark.parametrize("name", CALLBACK_MODULES)
def test_a_callback_module_keeps_the_scaffold_interface(name, state):
    """The scaffold's stubs registered nothing; the packages that fill them keep the signature."""
    import importlib

    module = importlib.import_module(f"pyorps.gui.callbacks.{name}")
    assert list(inspect.signature(module.register).parameters) == ["app", "state"]
    assert module.register(Dash(__name__), state) is None


def test_register_all_calls_the_new_modules_after_constrained(monkeypatch, state):
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks import constrained

    calls: list[str] = []
    original = constrained.register

    def constrained_register(app, st):
        calls.append("constrained")
        return original(app, st)

    monkeypatch.setattr(constrained, "register", constrained_register)
    for name in CALLBACK_MODULES:
        module = __import__(f"pyorps.gui.callbacks.{name}", fromlist=["register"])
        monkeypatch.setattr(module, "register", lambda app, st, _n=name: calls.append(_n))
    build_app(state)
    assert calls[0] == "constrained"
    assert sorted(calls[1:]) == sorted(CALLBACK_MODULES)         # each once, all after constrained


# ===================================================================== the state
class _FakeJob:
    """What ``services.jobs.BackgroundJob`` offers the scaffold (P1 builds the real one)."""

    def __init__(self, status: str = "running", can_stop: bool = True, text: str = "working: 3 of 9"):
        self.status = status
        self.can_stop = can_stop
        self.cancel = threading.Event()
        self._text = text

    def progress_text(self) -> str:
        return self._text


def test_a_new_state_starts_with_the_new_fields_empty(state):
    assert state.gui_settings == {} and state.jobs == {}
    assert state.loaded_cost_table == {} and state.corridor_graphs == {}


def test_clear_resets_the_new_fields_and_cancels_every_job(state):
    jobs = {"rasterize": _FakeJob(), "corridor": _FakeJob(status="done")}
    state.gui_settings = {"objective": {"w_cost": 1.0}}
    state.jobs = dict(jobs)
    state.loaded_cost_table = {"rows": [1]}
    state.corridor_graphs = {"segments": object()}
    routing_job = _FakeJob()
    state.routing_job = routing_job
    state.clear()
    assert state.gui_settings == {} and state.jobs == {}
    assert state.loaded_cost_table == {} and state.corridor_graphs == {}
    assert all(job.cancel.is_set() for job in jobs.values())
    assert routing_job.cancel.is_set() and state.routing_job is None


def test_clear_copes_with_a_job_that_has_no_cancel_event(state):
    state.jobs = {"rasterize": types.SimpleNamespace(status="running")}
    state.clear()
    assert state.jobs == {}


# =============================================================== the project file
SETTINGS = {
    "objective": {"preset": "gentlest", "weights": {"cost": 1.0, "length": 0.0, "gradient": 2.5}},
    "cost_field": {"origins": "sources", "combine": "sum", "threads": 0},
    "corridor": {"k": 3, "min_shared_m": 50.0},
    "constrained": {"backend": "cython", "profile": {"name": "gui_profile"}},
}


def _save(state, tmp_path, name: str = "proj") -> str:
    return project_io.save_project(state, tmp_path / name)


def test_gui_settings_round_trip_through_save_and_open(state, tmp_path):
    state.gui_settings = json.loads(json.dumps(SETTINGS))
    path = _save(state, tmp_path)
    assert json.loads(open(path, encoding="utf-8").read())["gui_settings"] == SETTINGS
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings == SETTINGS
        assert other.dirty is False
    finally:
        other.shutdown()


def test_the_manifest_version_stays_1():
    assert project_io.MANIFEST_VERSION == 1


def test_a_project_saved_before_round_14_opens_with_empty_settings(state, tmp_path):
    path = _save(state, tmp_path)
    manifest = json.loads(open(path, encoding="utf-8").read())
    for key in ("gui_settings", "routing_settings", "raster_settings", "skipped_layers"):
        manifest.pop(key, None)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    state.gui_settings = json.loads(json.dumps(SETTINGS))        # opening replaces what the page had
    project_io.load_project(state, path)
    assert state.gui_settings == {}


def test_a_minimal_old_project_file_opens(state, tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "project.json").write_text(
        json.dumps({"version": 1, "project_crs": "EPSG:32632", "layers": [], "cost_table": {}}), encoding="utf-8")
    project_io.load_project(state, folder / "project.json")
    assert state.gui_settings == {} and state.routing_settings == {} and state.raster_settings == {}
    assert state.layers == {}


@pytest.mark.parametrize("raw", [None, [], "objective", 7, [{"objective": {}}]])
def test_a_gui_settings_entry_that_is_not_a_mapping_opens_as_empty(state, tmp_path, raw):
    path = _save(state, tmp_path)
    manifest = json.loads(open(path, encoding="utf-8").read())
    manifest["gui_settings"] = raw
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    project_io.load_project(state, path)
    assert state.gui_settings == {}


def test_only_mappings_survive_inside_gui_settings():
    kept = project_io.normalise_gui_settings({"objective": {"a": 1}, "oops": 5, "list": [1]})
    assert kept == {"objective": {"a": 1}}


def test_gui_settings_are_saved_as_a_json_safe_copy(state, tmp_path):
    state.gui_settings = {"corridor": {"path": tmp_path, "tags": {"a", "b"} - {"b"}}, "bad": {(1, 2): 3}}
    path = _save(state, tmp_path)
    saved = json.loads(open(path, encoding="utf-8").read())["gui_settings"]
    assert saved["corridor"]["path"] == str(tmp_path)          # not JSON: written as text (default=str)
    assert saved["corridor"]["tags"] == "{'a'}"
    assert "bad" not in saved                                  # tuple keys cannot be written: left out
    assert state.gui_settings["corridor"]["path"] == tmp_path  # the live settings are not touched


def test_the_saved_copy_does_not_share_memory_with_the_state():
    live = {"objective": {"weights": {"cost": 1.0}}}
    copy = project_io.normalise_gui_settings(live)
    copy["objective"]["weights"]["cost"] = 9.0
    assert live["objective"]["weights"]["cost"] == 1.0


# ===================================================== first paint from the state
def test_a_blank_project_prefills_none_of_the_new_components(state):
    values = layout._state_prefill(state)  # noqa: SLF001
    new = set(PLANNED_IDS.values()) | set(ids.OHLF_IDS.values())
    assert set(values) & new == set()


def test_prefill_sections_calls_the_prefill_of_every_section(monkeypatch, state):
    from pyorps.gui.sections import corridor, cost_field, objective, profile_form

    seen: list[str] = []

    def stub(name: str, component_id: str):
        def prefill(st, put):
            assert st is state
            seen.append(name)
            put(component_id, "data", {"from": name})
        return prefill

    monkeypatch.setattr(objective, "prefill", stub("objective", ids.OBJ_SETTINGS))
    monkeypatch.setattr(cost_field, "prefill", stub("cost_field", ids.CF_SETTINGS))
    monkeypatch.setattr(corridor, "prefill", stub("corridor", ids.COR_SETTINGS))
    monkeypatch.setattr(profile_form, "prefill", stub("profile_form", ids.OHL_SETTINGS))
    page = build_layout(state)
    assert sorted(seen) == ["corridor", "cost_field", "objective", "profile_form"]
    assert _find(page, ids.OBJ_SETTINGS).data == {"from": "objective"}
    assert _find(page, ids.CF_SETTINGS).data == {"from": "cost_field"}
    assert _find(page, ids.COR_SETTINGS).data == {"from": "corridor"}
    assert _find(page, ids.OHL_SETTINGS).data == {"from": "profile_form"}


def test_a_failing_section_prefill_is_logged_and_skips_only_itself(monkeypatch, state, caplog):
    from pyorps.gui.sections import corridor, objective

    def broken(st, put):
        put(ids.OBJ_SETTINGS, "data", {"half": "done"})
        raise RuntimeError("boom")

    monkeypatch.setattr(objective, "prefill", broken)
    monkeypatch.setattr(corridor, "prefill", lambda st, put: put(ids.COR_SETTINGS, "data", {"ok": 1}))
    with caplog.at_level(logging.WARNING, logger=layout.__name__):
        page = build_layout(state)
    assert _find(page, ids.OBJ_SETTINGS).data == {}                  # nothing half-restored
    assert _find(page, ids.COR_SETTINGS).data == {"ok": 1}
    assert any("objective_prefill" in record.getMessage() for record in caplog.records)


def test_the_job_controls_map_each_kind_to_its_poll_stop_and_status():
    assert layout.JOB_CONTROLS == {
        "rasterize": (ids.RASTERIZE_POLL, ids.RASTERIZE_STOP_BTN, ids.RASTERIZE_LOG),
        "constrained": (ids.OHL_POLL, ids.OHL_STOP_BTN, ids.OHL_STATUS),
        "cost_field": (ids.CF_POLL, ids.CF_STOP_BTN, ids.CF_STATUS),
        "corridor": (ids.COR_POLL, None, ids.COR_STATUS),
    }


def _put_recorder():
    puts: list[tuple] = []
    return puts, lambda component_id, prop, value: puts.append((component_id, prop, value))


@pytest.mark.parametrize(("kind", "poll", "stop", "status_id"), [
    ("rasterize", ids.RASTERIZE_POLL, ids.RASTERIZE_STOP_BTN, ids.RASTERIZE_LOG),
    ("constrained", ids.OHL_POLL, ids.OHL_STOP_BTN, ids.OHL_STATUS),
    ("cost_field", ids.CF_POLL, ids.CF_STOP_BTN, ids.CF_STATUS),
])
def test_a_running_job_enables_its_poll_shows_stop_and_its_progress(state, kind, poll, stop, status_id):
    state.jobs[kind] = _FakeJob()
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    assert puts == [(poll, "disabled", False), (stop, "style", {}), (status_id, "children", "working: 3 of 9")]


def test_a_running_corridor_job_has_no_stop_button(state):
    from pyorps.gui.services import corridor
    state.jobs["corridor"] = job = _FakeJob()
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    # the kind's own wording: the corridor job says that it cannot be stopped (the generic line would not)
    assert puts == [(ids.COR_POLL, "disabled", False), (ids.COR_STATUS, "children", corridor.status_text(job))]
    assert corridor.CANNOT_STOP_NOTE in puts[1][2]


def test_a_running_job_that_cannot_be_stopped_keeps_stop_hidden(state):
    state.jobs["rasterize"] = _FakeJob(can_stop=False)
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    assert puts == [(ids.RASTERIZE_POLL, "disabled", False), (ids.RASTERIZE_LOG, "children", "working: 3 of 9")]


@pytest.mark.parametrize("status", ["done", "cancelled", "failed"])
def test_a_finished_job_enables_the_poll_so_the_result_is_collected(state, status):
    state.jobs["rasterize"] = _FakeJob(status=status)
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    assert puts == [(ids.RASTERIZE_POLL, "disabled", False)]


def test_unknown_job_kinds_and_a_state_without_jobs_put_nothing(state):
    state.jobs["something-else"] = _FakeJob()
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    layout._prefill_jobs(types.SimpleNamespace(), put)  # noqa: SLF001
    assert puts == []


def test_a_job_that_outlived_the_page_turns_the_polls_on_in_the_built_layout(state):
    state.jobs["rasterize"] = _FakeJob(text="Rasterizing: stage 2 of 5")
    state.jobs["constrained"] = _FakeJob(can_stop=False, text="segment 1 of 4")
    page = build_layout(state)
    assert _find(page, ids.RASTERIZE_POLL).disabled is False
    assert _find(page, ids.RASTERIZE_STOP_BTN).style == {}
    assert _find(page, ids.RASTERIZE_LOG).children == "Rasterizing: stage 2 of 5"
    assert _find(page, ids.OHL_POLL).disabled is False
    assert _find(page, ids.OHL_STOP_BTN).style == {"display": "none"}
    assert _find(page, ids.OHL_STATUS).children == "segment 1 of 4"
    assert _find(page, ids.CF_POLL).disabled is True and _find(page, ids.COR_POLL).disabled is True

"""Round 14, P6: the constrained profile as a form in the "Constrained routing" card.

Headless callback tests (the harness of ``conftest.py`` drives the callbacks as Dash would) plus layout checks.
Components are addressed as "<id>.<prop>": several ids contain each other. A form edit must reach the YAML text
(the run reads it), keys the form does not show must survive, a bad value must give an inline problem and no
exception, and the form and the text must never feed each other in a loop.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import dash_bootstrap_components as dbc
import pytest
import yaml
from conftest import invoke
from dash import html
from dash.exceptions import PreventUpdate

from pyorps.core import optional_deps
from pyorps.core.exceptions import MissingDependencyError
from pyorps.gui import ids
from pyorps.gui.layout import FIT_COLUMNS_RESPONSIVE, build_layout
from pyorps.gui.sections import profile_form as form_section
from pyorps.gui.services import constrained, project_io
from pyorps.gui.services import profile_form as service
from pyorps.gui.state import ProjectState

FILL = (f"{ids.OHLF_BASE}.data", ids.OHLF_FROM_YAML_BTN)
EDIT = (f"{ids.OHLF_BASE}.data", ids.OHLF_IDS["name"])
RESTORE = (f"{ids.OHLF_BASE}.data", ids.PROJECT_STATUS)
MIRROR = (f"{ids.OHL_SETTINGS}.data",)
TOGGLE = (f"{form_section.ANGLE_SCALE_WRAP}.style",)
ANGLE_ADD = (f"{ids.OHLF_ANGLE_GRID}.rowData", ids.OHLF_ANGLE_ADD_BTN)
ANGLE_DEL = (f"{ids.OHLF_ANGLE_GRID}.rowData", ids.OHLF_ANGLE_DEL_BTN)

PRESET_NAMES = tuple(constrained.shipped_profiles())

PROFILE = {
    "name": "mini", "description": "small test profile",
    "soft_angle_limit_deg": 10, "hard_angle_limit_deg": 35,
    "angle_cost_function": "piecewise",
    "angle_cost_params": {"breakpoints": [0, 10, 35], "costs": [0, 100, 5000]},
    "min_span_m": 40, "max_span_m": 200, "span_bin_size_m": 10,
    "tower_cost_params": {"terrain_cost_map": {"0": 300, "100": 500},
                          "terrain_interpolation": "linear",
                          "angle_types": {"light": {"max_angle_deg": 15, "base_cost": 40000},
                                          "strain": {"max_angle_deg": 35, "base_cost": 90000}}},
    "tower_cost_function": "terrain_plus_angle",          # a key the router never reads, not on the form
    "my_note": "a key the form knows nothing about",
}


# ------------------------------------------------------------------ helpers
def _preset(name: str) -> str:
    return constrained.shipped_profiles()[name]


def _preset_profile(name: str) -> dict:
    return yaml.safe_load(Path(_preset(name)).read_text(encoding="utf-8"))


def _rows(resp, grid) -> list[dict]:
    """The rows of a grid response without their ``id``."""
    return [{k: v for k, v in row.items() if k != "id"} for row in resp[grid]["rowData"]]


def _form_args(profile=None, *, edits=None, rows=None, text="", base=None, status="", notices=None,
               live=None):
    """The positional arguments of ``form_to_yaml``: 24 field values, the 3 grids' ``rowData`` and
    ``cellValueChanged``, then the states text, base, status, notices and the 3 grids' ``virtualRowData``."""
    profile = PROFILE if profile is None else profile
    values, angle, terrain, types = service.form_values(profile)
    values = {**values, **(edits or {})}
    grids = rows or (angle, terrain, types)
    return (*(values[key] for key in service.FORM_KEYS), *grids, None, None, None, text,
            profile if base is None else base, status, notices or [], *(live or (None, None, None)))


def _edit(app, *args, **kwargs):
    return invoke(app, EDIT, *args, **kwargs)


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _components(page):
    return list(page._traverse())  # noqa: SLF001


def _by_id(page, component_id):
    found = [c for c in _components(page) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id}: {len(found)} components"
    return found[0]


def _text(component) -> str:
    """All plain text inside a component."""
    pieces = []
    for part in component._traverse():  # noqa: SLF001
        child = getattr(part, "children", None)
        if isinstance(child, str):
            pieces.append(child)
    return " ".join(pieces)


@pytest.fixture()
def page(state):
    return build_layout(state)


# ================================================================== the layout
def test_the_form_has_one_field_per_key_in_the_order_of_the_keys(page):
    order = [c.id for c in _components(page) if getattr(c, "id", None) in set(ids.OHLF_IDS.values())]
    assert order == [ids.OHLF_IDS[key] for key in ids.OHLF_KEYS]
    assert ids.OHLF_KEYS == service.FORM_KEYS
    assert len(order) == 24


@pytest.mark.parametrize("key", service.FORM_KEYS)
def test_each_field_comes_from_its_spec(page, key):
    spec = service.FIELD_SPECS[key]
    control = _by_id(page, ids.OHLF_IDS[key])
    label = next(c for c in _components(page)
                 if isinstance(c, dbc.Label) and getattr(c, "html_for", None) == ids.OHLF_IDS[key])
    expected = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
    assert label.children == expected
    wrapper = next(c for c in _components(page) if isinstance(c, html.Div)
                   and isinstance(c.children, list) and label in c.children and control in c.children)
    assert spec.help in wrapper.title and service.describe_range(spec) in wrapper.title
    if spec.kind == "select":
        assert isinstance(control, dbc.Select)
        assert [o["value"] for o in control.options] == list(spec.options)
        assert control.value == spec.default
    elif spec.kind == "text":
        assert control.type == "text" and control.maxLength == spec.max_length
    elif spec.kind == "list":
        assert control.type == "text"
    else:
        assert control.type == "number"
        assert control.min == spec.minimum and control.max == spec.maximum
        assert control.step == (1 if spec.whole else "any")
        assert control.debounce is True


def test_every_unit_is_in_a_label(page):
    for key in service.FORM_KEYS:
        spec = service.FIELD_SPECS[key]
        if spec.unit:
            label = next(c for c in _components(page)
                         if isinstance(c, dbc.Label) and getattr(c, "html_for", None) == ids.OHLF_IDS[key])
            assert f"({spec.unit})" in label.children


def test_a_fresh_form_shows_the_spec_defaults(page):
    defaults = service.form_values({})[0]
    for key in service.FORM_KEYS:
        control = _by_id(page, ids.OHLF_IDS[key])
        assert control.value == defaults[key]


def test_the_three_tables_are_built_from_the_table_specs(page):
    for key, spec in service.TABLE_SPECS.items():
        grid_id, add_id, delete_id = form_section.TABLE_CONTROLS[key]
        grid = _by_id(page, grid_id)
        assert [c["field"] for c in grid.columnDefs] == [col.key for col in spec.columns]
        for definition, column in zip(grid.columnDefs, spec.columns):
            assert definition["headerName"] == (f"{column.label} ({column.unit})" if column.unit
                                                else column.label)
            assert definition["editable"] is True
            assert definition["cellDataType"] == ("text" if column.kind == "text" else "number")
        # fitted by assets/grid-fit.js whenever its width changes (round 15, see FIT_COLUMNS_RESPONSIVE)
        assert grid.className == FIT_COLUMNS_RESPONSIVE and getattr(grid, "columnSize", None) is None
        assert grid.getRowId == "params.data.id"
        assert grid.rowData == []
        assert _by_id(page, add_id).children == "+ row"
        assert _by_id(page, delete_id).children == "− selected"


def test_no_grid_uses_row_grouping_or_a_function_cell_style(page):
    for grid_id in form_section.GRID_IDS:
        grid = _by_id(page, grid_id)
        text = json.dumps([grid.columnDefs, grid.defaultColDef])
        assert "rowGroup" not in text and "cellStyle" not in text


def test_the_yaml_text_is_collapsed_and_still_there_once(page):
    details = _by_id(page, ids.OHLF_YAML_DETAILS)
    assert details.open is False
    assert _by_id(page, ids.OHL_PROFILE_TEXT) is not None
    assert details.children[0].children == "Raw YAML (advanced)"


def test_the_form_sits_between_the_profile_file_row_and_the_yaml_block(page):
    options = _by_id(page, ids.OHL_COLLAPSE)
    top = options.children
    form = next(c for c in top if getattr(c, "className", "") == "ohlf-form my-1")
    position = top.index(form)
    assert top[position + 1].id == ids.OHLF_YAML_DETAILS
    assert any(getattr(part, "id", None) == ids.OHL_PROFILE_PATH for part in top[position - 1]._traverse())  # noqa: SLF001


def test_the_status_clearance_and_button_exist_once(page):
    for component_id in (ids.OHLF_STATUS, ids.OHLF_CLEARANCE_INFO, ids.OHLF_FROM_YAML_BTN, ids.OHLF_BASE,
                         ids.OHL_SETTINGS, form_section.ANGLE_SCALE_WRAP, form_section.ANGLE_TABLE_WRAP):
        _by_id(page, component_id)
    assert _by_id(page, ids.OHLF_CLEARANCE_INFO).children == service.clearance_status({})


def test_the_form_targets_cover_every_field_table_and_line():
    targets = form_section.FORM_TARGETS
    assert len(set(targets)) == len(targets) == 24 + 3 + 4
    props = form_section.form_props(PROFILE)
    assert len(props) == len(targets)


# ================================================================== fill the form
def test_a_shipped_profile_fills_the_form(app):
    name = PRESET_NAMES[0]
    profile = _preset_profile(name)
    resp = invoke(app, FILL, _preset(name), None, None, "", "", [],
                  triggered=[f"{ids.OHL_PROFILE_PRESET}.value"])
    values, angle, terrain, types = service.form_values(profile)
    for key in service.FORM_KEYS:
        assert resp[ids.OHLF_IDS[key]]["value"] == values[key], key
    assert _rows(resp, ids.OHLF_ANGLE_GRID) == angle
    assert _rows(resp, ids.OHLF_TERRAIN_GRID) == terrain
    assert _rows(resp, ids.OHLF_TYPES_GRID) == types
    assert all(row["id"] for row in resp[ids.OHLF_ANGLE_GRID]["rowData"])
    assert resp[ids.OHLF_BASE]["data"] == profile
    assert resp[ids.OHLF_CLEARANCE_INFO]["children"] == service.clearance_status(profile)
    assert ids.NOTICES not in resp or resp[ids.NOTICES]["data"] == []


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_shipped_profile_fills_without_a_problem_and_notes_what_the_router_ignores(app, name):
    profile = _preset_profile(name)
    resp = invoke(app, FILL, _preset(name), None, None, "", "", [],
                  triggered=[f"{ids.OHL_PROFILE_PRESET}.value"])
    status = resp[ids.OHLF_STATUS]["children"]
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_OK
    unused = service.unused_keys(profile)
    assert ("Not read by the router" in status) == bool(unused)
    for key in unused:
        assert key in status


def test_clearing_the_preset_dropdown_changes_nothing(app):
    with pytest.raises(PreventUpdate):
        invoke(app, FILL, None, None, None, "", "", [], triggered=[f"{ids.OHL_PROFILE_PRESET}.value"])


def test_loading_a_profile_file_fills_the_form(app, tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump(PROFILE), encoding="utf-8")
    resp = invoke(app, FILL, None, 1, None, str(path), "", [], triggered=[f"{ids.OHL_PROFILE_LOAD_BTN}.n_clicks"])
    assert resp[ids.OHLF_IDS["name"]]["value"] == "mini"
    assert resp[ids.OHLF_IDS["soft_angle_limit_deg"]]["value"] == 10
    assert resp[ids.OHLF_BASE]["data"] == PROFILE
    assert _rows(resp, ids.OHLF_TYPES_GRID)[0] == {"name": "light", "max_angle_deg": 15, "base_cost": 40000}


def test_loading_a_missing_profile_file_leaves_the_error_to_the_load_callback(app, tmp_path):
    with pytest.raises(PreventUpdate):
        invoke(app, FILL, None, 1, None, str(tmp_path / "nope.yaml"), "", [],
               triggered=[f"{ids.OHL_PROFILE_LOAD_BTN}.n_clicks"])


def test_yaml_to_form_reads_the_text(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    resp = invoke(app, FILL, None, None, 1, "", text, [], triggered=[f"{ids.OHLF_FROM_YAML_BTN}.n_clicks"])
    assert resp[ids.OHLF_IDS["hard_angle_limit_deg"]]["value"] == 35
    assert resp[ids.OHLF_IDS["name"]]["value"] == "mini"
    assert resp[ids.OHLF_BASE]["data"] == PROFILE            # the whole text, also keys the form does not show
    assert resp[ids.OHLF_STATUS]["children"].startswith("Not read by the router")
    assert "tower_cost_function" in resp[ids.OHLF_STATUS]["children"]
    assert "my_note" in resp[ids.OHLF_STATUS]["children"]     # ignored by the library, kept in the YAML


@pytest.mark.parametrize("text", ["a: [unclosed", "- just\n- a list\n", "", "42"])
def test_yaml_to_form_reports_bad_text_as_a_notice_and_changes_nothing(app, text):
    resp = invoke(app, FILL, None, None, 1, "", text, [], triggered=[f"{ids.OHLF_FROM_YAML_BTN}.n_clicks"])
    assert set(resp) == {ids.NOTICES}                       # no form value, table, base or status is written
    notice = _notices(resp)[-1]
    assert notice["title"] == "The profile text could not be read"
    assert notice["severity"] == "warning" and "Traceback" not in json.dumps(notice)
    assert notice["focus_control"] == ids.OHLF_YAML_DETAILS


def test_yaml_to_form_without_pyyaml_ends_in_a_clear_notice(app, monkeypatch):
    def no_yaml():
        raise MissingDependencyError("PyYAML is needed", extra="yaml", name="yaml")

    monkeypatch.setattr(optional_deps, "import_yaml", no_yaml)
    resp = invoke(app, FILL, None, None, 1, "", "name: x\n", [], triggered=[f"{ids.OHLF_FROM_YAML_BTN}.n_clicks"])
    assert set(resp) == {ids.NOTICES}
    notice = _notices(resp)[-1]
    assert notice["title"] == "PyYAML is not installed" and "pyorps[yaml]" in notice["fix"]


def test_yaml_to_form_with_a_profile_that_has_problems_shows_them_and_a_notice(app):
    text = yaml.safe_dump({**PROFILE, "hard_angle_limit_deg": 5})
    resp = invoke(app, FILL, None, None, 1, "", text, [], triggered=[f"{ids.OHLF_FROM_YAML_BTN}.n_clicks"])
    assert "Hard turn limit" in resp[ids.OHLF_STATUS]["children"]
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_PROBLEM
    assert _notices(resp)[-1]["focus_control"] == ids.OHLF_IDS["hard_angle_limit_deg"]
    assert resp[ids.OHLF_IDS["hard_angle_limit_deg"]]["value"] == 5       # the form shows the text as it is


def test_a_fill_without_a_known_trigger_changes_nothing(app):
    with pytest.raises(PreventUpdate):
        invoke(app, FILL, None, None, None, "", "", [])


# ================================================================== form -> YAML
def test_a_form_edit_updates_the_yaml_text_and_keeps_unknown_keys(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    resp = _edit(app, *_form_args(edits={"soft_angle_limit_deg": 12}, text=text))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["soft_angle_limit_deg"] == 12
    assert written["tower_cost_function"] == "terrain_plus_angle"      # not on the form, survives
    assert written["my_note"] == "a key the form knows nothing about"
    assert written["name"] == "mini"
    assert resp[ids.OHLF_BASE]["data"] == written
    assert resp[ids.OHLF_STATUS]["className"] in (form_section.STATUS_CLASS_OK,
                                                  form_section.STATUS_CLASS_PROBLEM)


def test_every_field_edit_reaches_the_text(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    edits = {"description": "changed", "min_span_m": 45, "max_span_m": 210, "span_bin_size_m": 5,
             "max_gradient_percent": 30, "tower_heights_m": "25, 34", "tower_height_m": 25,
             "conductor_weight_per_m": 10, "conductor_tension_n": 20000, "min_clearance_m": 7,
             "tower_ground_area_m2": 100, "tower_area_cost_mode": "exact", "recommended_buffer_m": 25,
             "terminal_tower_cost": 1000, "gradient_scale": 3, "terrain_interpolation": "nearest"}
    resp = _edit(app, *_form_args(edits=edits, text=text))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["description"] == "changed" and written["min_span_m"] == 45
    assert written["tower_heights_m"] == [25, 34] and written["tower_area_cost_mode"] == "exact"
    assert written["gradient_cost_params"] == {"scale": 3}
    assert written["tower_cost_params"]["terrain_interpolation"] == "nearest"
    assert resp[ids.OHLF_CLEARANCE_INFO]["children"] == service.clearance_status(written)
    assert "Clearance check: on" in resp[ids.OHLF_CLEARANCE_INFO]["children"]


def test_a_table_edit_reaches_the_text(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    values, angle, terrain, types = service.form_values(PROFILE)
    terrain = [*terrain, {"raster_value": 200, "eur_per_m2": 800}]
    types = [{**types[0], "base_cost": 41000}, *types[1:]]
    resp = _edit(app, *_form_args(rows=(angle, terrain, types), text=text))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["tower_cost_params"]["terrain_cost_map"]["200"] == 800
    assert written["tower_cost_params"]["angle_types"]["light"]["base_cost"] == 41000
    assert written["tower_cost_params"]["angle_types"]["strain"]["base_cost"] == 90000


def test_a_cell_edit_reads_the_rows_the_grid_holds_now(app):
    """After a cell edit the grid's ``virtualRowData`` is current while its ``rowData`` still lags."""
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    values, angle, terrain, types = service.form_values(PROFILE)
    edited = [dict(row) for row in angle]
    edited[1]["cost"] = 777
    args = _form_args(rows=(angle, terrain, types), text=text, live=(edited, None, None))
    resp = _edit(app, *args, triggered=[f"{ids.OHLF_ANGLE_GRID}.cellValueChanged"])
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["angle_cost_params"]["costs"] == [0, 777, 5000]
    # without the edit trigger the live rows are ignored (a server-side change of ``rowData`` wins)
    with pytest.raises(PreventUpdate):
        _edit(app, *args, triggered=[f"{ids.OHLF_ANGLE_GRID}.rowData"])


def test_the_form_rows_carry_ids_and_the_service_never_sees_them(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    values, angle, terrain, types = service.form_values(PROFILE)
    with_ids = [form_section.with_row_ids(rows) for rows in (angle, terrain, types)]
    resp = _edit(app, *_form_args(edits={"soft_angle_limit_deg": 12}, rows=tuple(with_ids), text=text))
    assert yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])["angle_cost_params"] == PROFILE["angle_cost_params"]


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_a_fill_never_writes_the_text_back(app, name):
    """What a fill puts into the form is exactly what the base store holds: the edit callback stays silent."""
    profile = _preset_profile(name)
    values, angle, terrain, types = service.form_values(profile)
    grids = tuple(form_section.with_row_ids(rows) for rows in (angle, terrain, types))
    with pytest.raises(PreventUpdate):
        _edit(app, *_form_args(profile, rows=grids, text=Path(_preset(name)).read_text(encoding="utf-8")))


def test_an_empty_page_with_the_default_form_writes_nothing(app):
    with pytest.raises(PreventUpdate):
        _edit(app, *_form_args({}, text="", base={}))


def test_a_first_edit_on_an_empty_page_builds_the_text_from_the_defaults(app):
    resp = _edit(app, *_form_args({}, edits={"name": "fresh", "soft_angle_limit_deg": 6}, text="", base={}))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["name"] == "fresh" and written["soft_angle_limit_deg"] == 6
    assert written["hard_angle_limit_deg"] == 40 and written["angle_cost_function"] == "piecewise"


def test_a_form_edit_merges_into_a_text_the_user_edited_by_hand(app):
    hand = yaml.safe_dump({**PROFILE, "my_note": "typed by hand", "extra": {"a": 1}}, sort_keys=False)
    resp = _edit(app, *_form_args(edits={"description": "form edit"}, text=hand))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["my_note"] == "typed by hand" and written["extra"] == {"a": 1}
    assert written["description"] == "form edit"


def test_a_form_edit_with_broken_text_falls_back_to_the_base_store(app):
    resp = _edit(app, *_form_args(edits={"description": "after"}, text="a: [broken"))
    written = yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])
    assert written["description"] == "after" and written["my_note"] == "a key the form knows nothing about"


# ================================================================== problems
def test_a_bad_value_is_an_inline_problem_and_a_notice_that_focuses_the_field(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    resp = _edit(app, *_form_args(edits={"hard_angle_limit_deg": 5}, text=text))
    assert "(hard_angle_limit_deg): 5 deg is below" in resp[ids.OHLF_STATUS]["children"]
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_PROBLEM
    notice = _notices(resp)[-1]
    assert notice["title"] == "Hard turn limit (forbidden above): value not usable"
    assert notice["focus_control"] == ids.OHLF_IDS["hard_angle_limit_deg"] and notice["focus_id"] == ids.TAB_ROUTES
    # the text still follows the form: a run would stop at its own check
    assert yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])["hard_angle_limit_deg"] == 5


@pytest.mark.parametrize(("edits", "key"), [
    ({"soft_angle_limit_deg": -1}, "soft_angle_limit_deg"),
    ({"soft_angle_limit_deg": 181}, "soft_angle_limit_deg"),
    ({"max_span_m": 6000}, "max_span_m"),
    ({"max_span_m": 20}, "min_span_m"),            # the shortest span is above it
    ({"span_bin_size_m": None}, "span_bin_size_m"),
    ({"tower_ground_area_m2": 0.5}, "tower_ground_area_m2"),
    ({"tower_heights_m": "25, abc"}, "tower_heights_m"),
    ({"name": ""}, "name"),
    ({"name": "x" * 81}, "name"),
    ({"angle_cost_function": "wavy"}, "angle_cost_function"),
    ({"max_gradient_percent": 2000}, "max_gradient_percent"),
    ({"conductor_weight_per_m": 0}, "conductor_weight_per_m")])
def test_each_limit_of_the_specs_is_checked_again_by_the_callback(app, edits, key):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    resp = _edit(app, *_form_args(edits=edits, text=text))              # no exception
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_PROBLEM
    assert _notices(resp)[-1]["focus_control"] == ids.OHLF_IDS[key]


def test_the_same_problem_does_not_make_a_second_notice(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    first = _edit(app, *_form_args(edits={"hard_angle_limit_deg": 5}, text=text))
    status = first[ids.OHLF_STATUS]["children"]
    second = _edit(app, *_form_args(edits={"hard_angle_limit_deg": 5, "description": "again"}, text=text,
                                    status=status, notices=_notices(first)))
    assert ids.NOTICES not in second or len(_notices(second)) == len(_notices(first))
    assert second[ids.OHLF_STATUS]["children"] == status


def test_a_half_filled_row_is_shown_inline_only(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    values, angle, terrain, types = service.form_values(PROFILE)
    half = {"name": "heavy", "max_angle_deg": None, "base_cost": None}
    resp = _edit(app, *_form_args(rows=(angle, terrain, [*types, half]), text=text))
    assert "no value for" in resp[ids.OHLF_STATUS]["children"]
    assert ids.NOTICES not in resp
    blank = {"name": None, "max_angle_deg": None, "base_cost": None}      # a row that was just added
    resp = _edit(app, *_form_args(rows=(angle, terrain, [*types, blank]), text=text))
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_OK
    assert ids.NOTICES not in resp


def test_partial_clearance_settings_are_an_info_line_not_an_error(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    resp = _edit(app, *_form_args(edits={"tower_height_m": 25, "min_clearance_m": 7}, text=text))
    info = resp[ids.OHLF_CLEARANCE_INFO]["children"]
    assert info.startswith("Clearance check: off, still missing:")
    assert "conductor weight" in info and "conductor tension" in info
    assert resp[ids.OHLF_STATUS]["className"] == form_section.STATUS_CLASS_OK
    assert ids.NOTICES not in resp


def test_the_clearance_line_follows_the_four_settings(app):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    full = {"tower_height_m": 25, "conductor_weight_per_m": 10, "conductor_tension_n": 20000,
            "min_clearance_m": 7}
    resp = _edit(app, *_form_args(edits=full, text=text))
    assert resp[ids.OHLF_CLEARANCE_INFO]["children"] == (
        "Clearance check: on, the profile is complete (the run needs a terrain model (DEM); a surface model (DSM) is "
        "optional and adds trees and buildings)")
    resp = _edit(app, *_form_args(edits={"name": "again"}, text=text))
    assert resp[ids.OHLF_CLEARANCE_INFO]["children"] == service.clearance_status(PROFILE)


def test_a_form_edit_without_pyyaml_keeps_the_text_and_says_why(app, monkeypatch):
    def no_yaml():
        raise MissingDependencyError("PyYAML is needed", extra="yaml", name="yaml")

    monkeypatch.setattr(optional_deps, "import_yaml", no_yaml)
    args = _form_args(edits={"hard_angle_limit_deg": 5}, text="name: mini\n")
    resp = _edit(app, *args)
    assert ids.OHL_PROFILE_TEXT not in resp                              # the text is left alone
    titles = [n["title"] for n in _notices(resp)]
    assert "PyYAML is not installed" in titles
    assert "Hard turn limit" in resp[ids.OHLF_STATUS]["children"]        # the checks still work
    # the notice about PyYAML is not repeated on every edit
    again = _edit(app, *_form_args(edits={"hard_angle_limit_deg": 5}, text="name: mini\n",
                                   notices=_notices(resp), status=resp[ids.OHLF_STATUS]["children"]))
    assert [n["title"] for n in _notices(again)].count("PyYAML is not installed") <= 1


# ================================================================== table rows
def test_plus_row_adds_an_empty_row_with_an_id_and_keeps_the_others(app):
    existing = form_section.with_row_ids([{"angle_deg": 0, "cost": 0}])
    resp = invoke(app, ANGLE_ADD, 1, existing, triggered=[f"{ids.OHLF_ANGLE_ADD_BTN}.n_clicks"])
    rows = resp[ids.OHLF_ANGLE_GRID]["rowData"]
    assert rows[0] == existing[0]
    assert len(rows) == 2 and rows[1]["id"] and rows[1]["id"] != rows[0]["id"]
    assert rows[1]["angle_deg"] is None and rows[1]["cost"] is None
    with pytest.raises(PreventUpdate):
        invoke(app, ANGLE_ADD, None, existing)


@pytest.mark.parametrize(("table", "grid", "add", "delete"), [
    (key, *controls) for key, controls in form_section.TABLE_CONTROLS.items()])
def test_every_table_adds_and_deletes_rows(app, table, grid, add, delete):
    columns = form_section.TABLE_COLUMN_KEYS[table]
    rows = form_section.with_row_ids([{c: i for c in columns} for i in range(3)])
    added = invoke(app, (f"{grid}.rowData", add), 1, rows)[grid]["rowData"]
    assert len(added) == 4 and set(added[3]) == {"id", *columns}
    kept = invoke(app, (f"{grid}.rowData", delete), 1, added, [added[1], added[3]])[grid]["rowData"]
    assert [r["id"] for r in kept] == [added[0]["id"], added[2]["id"]]


def test_minus_selected_without_a_selection_changes_nothing(app):
    rows = form_section.with_row_ids([{"angle_deg": 0, "cost": 0}])
    for selected in (None, []):
        with pytest.raises(PreventUpdate):
            invoke(app, ANGLE_DEL, 1, rows, selected)
    with pytest.raises(PreventUpdate):
        invoke(app, ANGLE_DEL, None, rows, rows)


def test_two_identical_blank_rows_are_deleted_one_at_a_time(app):
    first, second = form_section.with_row_ids([{"angle_deg": None, "cost": None}] * 2)
    resp = invoke(app, ANGLE_DEL, 1, [first, second], [second])
    assert resp[ids.OHLF_ANGLE_GRID]["rowData"] == [first]


# ================================================================== show / hide
@pytest.mark.parametrize(("function", "scale_hidden", "table_hidden"), [
    ("piecewise", True, False), ("linear", False, True), ("quadratic", False, True), (None, True, False)])
def test_the_turn_cost_factor_and_the_turn_table_swap(app, function, scale_hidden, table_hidden):
    resp = invoke(app, TOGGLE, function)
    assert (resp[form_section.ANGLE_SCALE_WRAP]["style"] == {"display": "none"}) is scale_hidden
    assert (resp[form_section.ANGLE_TABLE_WRAP]["style"] == {"display": "none"}) is table_hidden


# ================================================================== settings mirror and restore
def test_the_constrained_settings_are_mirrored_into_the_state(app, state):
    text = yaml.safe_dump(PROFILE, sort_keys=False)
    resp = invoke(app, MIRROR, "raster_gpu", True, "dem.tif", "dsm.tif", text)
    saved = state.gui_settings["constrained"]
    assert saved == {"backend": "raster_gpu", "experimental": True, "dem": "dem.tif", "dsm": "dsm.tif",
                     "profile": json.loads(json.dumps(PROFILE))}
    assert resp[ids.OHL_SETTINGS]["data"] == saved


def test_blank_fields_fall_back_to_the_defaults_in_the_mirror(app, state):
    invoke(app, MIRROR, None, None, "", None, "")
    assert state.gui_settings["constrained"] == {"backend": "cython", "experimental": False,
                                                 "dem": None, "dsm": None}


def test_a_profile_being_typed_keeps_the_last_good_one_but_blank_text_drops_it(app, state):
    good = yaml.safe_dump(PROFILE, sort_keys=False)
    invoke(app, MIRROR, "cython", False, None, None, good)
    invoke(app, MIRROR, "cython", False, None, None, "name: [half typed")
    assert state.gui_settings["constrained"]["profile"]["name"] == "mini"
    invoke(app, MIRROR, "cython", False, None, None, "")
    assert "profile" not in state.gui_settings["constrained"]


def test_the_mirror_keeps_the_profile_when_the_text_cannot_be_read_without_pyyaml(app, state, monkeypatch):
    invoke(app, MIRROR, "cython", False, None, None, yaml.safe_dump(PROFILE))

    def no_yaml():
        raise MissingDependencyError("PyYAML is needed", extra="yaml", name="yaml")

    monkeypatch.setattr(optional_deps, "import_yaml", no_yaml)
    invoke(app, MIRROR, "cython", False, None, None, "")
    assert state.gui_settings["constrained"]["profile"]["name"] == "mini"


def test_the_mirror_changes_the_dict_in_place(app, state):
    held = state.gui_settings
    invoke(app, MIRROR, "cython", False, None, None, "name: x\n")
    assert state.gui_settings is held and "constrained" in held


def _restore(app, status="opened: p", text="", backend="cython", experimental=False, dem=None, dsm=None):
    return invoke(app, RESTORE, status, text, backend, experimental, dem, dsm,
                  triggered=[f"{ids.PROJECT_STATUS}.children"])


def test_open_project_puts_the_saved_profile_and_settings_back(app, state):
    state.gui_settings = {"constrained": {"profile": json.loads(json.dumps(PROFILE)), "backend": "raster_gpu",
                                          "experimental": True, "dem": "d.tif", "dsm": "s.tif"}}
    resp = _restore(app, text="name: old project\n")
    assert resp[ids.OHLF_IDS["name"]]["value"] == "mini"
    assert resp[ids.OHLF_IDS["hard_angle_limit_deg"]]["value"] == 35
    assert _rows(resp, ids.OHLF_ANGLE_GRID) == service.form_values(PROFILE)[1]
    assert resp[ids.OHLF_BASE]["data"] == json.loads(json.dumps(PROFILE))
    assert yaml.safe_load(resp[ids.OHL_PROFILE_TEXT]["value"])["my_note"] == PROFILE["my_note"]
    assert resp[ids.OHL_BACKEND]["value"] == "raster_gpu" and resp[ids.OHL_EXPERIMENTAL]["value"] is True
    assert resp[ids.OHL_DEM]["value"] == "d.tif" and resp[ids.OHL_DSM]["value"] == "s.tif"
    assert resp[ids.OHLF_CLEARANCE_INFO]["children"] == service.clearance_status(PROFILE)


def test_new_project_puts_the_defaults_back(app, state):
    state.gui_settings = {}
    resp = _restore(app, status="new project", text=yaml.safe_dump(PROFILE), backend="raster_gpu",
                    experimental=True, dem="d.tif", dsm="s.tif")
    defaults = service.form_values({})[0]
    for key in service.FORM_KEYS:
        assert resp[ids.OHLF_IDS[key]]["value"] == defaults[key]
    assert resp[ids.OHL_PROFILE_TEXT]["value"] == ""
    assert resp[ids.OHL_BACKEND]["value"] == "cython" and resp[ids.OHL_EXPERIMENTAL]["value"] is False
    assert resp[ids.OHL_DEM]["value"] == "" and resp[ids.OHL_DSM]["value"] == ""
    assert resp[ids.OHLF_BASE]["data"] == {} and resp[ids.OHLF_STATUS]["children"] == ""
    for grid in form_section.GRID_IDS:
        assert resp[grid]["rowData"] == []


def test_a_project_without_saved_constrained_settings_resets_the_page(app, state):
    state.gui_settings = {"objective": {"x": 1}}                    # an older project file
    resp = _restore(app, text="name: stale\n")
    assert resp[ids.OHL_PROFILE_TEXT]["value"] == ""


def test_saving_a_project_does_not_rewrite_the_form_or_the_text(app, state):
    state.gui_settings = {"constrained": {"profile": PROFILE, "backend": "cython"}}
    with pytest.raises(PreventUpdate):
        _restore(app, status="saved: C:/p/project.json", text="# my comment\nname: other\n")


def test_a_page_that_already_shows_the_project_is_left_alone(app, state):
    state.gui_settings = {"constrained": {"profile": json.loads(json.dumps(PROFILE)), "backend": "cython"}}
    with pytest.raises(PreventUpdate):
        _restore(app, text="# my comment\n" + yaml.safe_dump(PROFILE))


def test_restore_without_pyyaml_still_fills_the_form(app, state, monkeypatch):
    state.gui_settings = {"constrained": {"profile": json.loads(json.dumps(PROFILE))}}

    def no_yaml():
        raise MissingDependencyError("PyYAML is needed", extra="yaml", name="yaml")

    monkeypatch.setattr(optional_deps, "import_yaml", no_yaml)
    resp = _restore(app, text="")
    assert resp[ids.OHLF_IDS["name"]]["value"] == "mini"
    assert ids.OHL_PROFILE_TEXT not in resp


def test_the_settings_survive_save_and_open(app, state, tmp_path):
    invoke(app, MIRROR, "raster_gpu", True, "dem.tif", None, yaml.safe_dump(PROFILE))
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings["constrained"] == state.gui_settings["constrained"]
        other.gui_settings = json.loads(json.dumps(other.gui_settings))
        other_page = build_layout(other)
        assert _by_id(other_page, ids.OHLF_IDS["name"]).value == "mini"
        assert yaml.safe_load(_by_id(other_page, ids.OHL_PROFILE_TEXT).value)["my_note"] == PROFILE["my_note"]
    finally:
        other.shutdown()


# ================================================================== first paint
def test_a_blank_project_prefills_nothing(state):
    puts: list = []
    form_section.prefill(state, lambda *args: puts.append(args))
    assert puts == []


def test_a_reloaded_page_shows_the_saved_form(state):
    saved = json.loads(json.dumps(PROFILE))
    state.gui_settings = {"constrained": {"profile": saved, "backend": "cython_parallel", "experimental": True,
                                          "dem": "d.tif", "dsm": "s.tif"}}
    page = build_layout(state)
    values = service.form_values(saved)[0]
    for key in service.FORM_KEYS:
        assert _by_id(page, ids.OHLF_IDS[key]).value == values[key], key
    assert [{k: v for k, v in r.items() if k != "id"} for r in _by_id(page, ids.OHLF_ANGLE_GRID).rowData] \
        == service.form_values(saved)[1]
    assert _by_id(page, ids.OHLF_BASE).data == saved
    assert yaml.safe_load(_by_id(page, ids.OHL_PROFILE_TEXT).value) == saved
    assert _by_id(page, ids.OHL_BACKEND).value == "cython_parallel"
    assert _by_id(page, ids.OHL_EXPERIMENTAL).value is True
    assert _by_id(page, ids.OHL_DEM).value == "d.tif" and _by_id(page, ids.OHL_DSM).value == "s.tif"
    assert _by_id(page, ids.OHL_SETTINGS).data == state.gui_settings["constrained"]
    assert "Not read by the router" in _by_id(page, ids.OHLF_STATUS).children


def test_a_reload_without_a_saved_profile_keeps_the_layout_defaults(state):
    state.gui_settings = {"constrained": {"backend": "raster_gpu_v4", "experimental": False}}
    page = build_layout(state)
    assert _by_id(page, ids.OHL_BACKEND).value == "raster_gpu_v4"
    assert _by_id(page, ids.OHLF_IDS["name"]).value == service.FIELD_SPECS["name"].default
    assert getattr(_by_id(page, ids.OHL_PROFILE_TEXT), "value", None) is None


def test_a_reload_without_pyyaml_still_shows_the_form(state, monkeypatch):
    def no_yaml():
        raise MissingDependencyError("PyYAML is needed", extra="yaml", name="yaml")

    monkeypatch.setattr(optional_deps, "import_yaml", no_yaml)
    state.gui_settings = {"constrained": {"profile": json.loads(json.dumps(PROFILE))}}
    page = build_layout(state)
    assert _by_id(page, ids.OHLF_IDS["name"]).value == "mini"
    assert getattr(_by_id(page, ids.OHL_PROFILE_TEXT), "value", None) is None


# ================================================================== no callback loop
def _prop(component_id, prop) -> tuple[str, str]:
    return str(component_id).split("@", maxsplit=1)[0], prop


def _graph(app):
    """Callback key -> (outputs, inputs) as ``(id, prop)`` pairs; States are not triggers."""
    graph = {}
    for key, entry in app.callback_map.items():
        outputs = entry["output"] if isinstance(entry["output"], (list, tuple)) else [entry["output"]]
        graph[key] = ({_prop(o.component_id, o.component_property) for o in outputs},
                      {_prop(i["id"], i["property"]) for i in entry.get("inputs", [])
                       if isinstance(i["id"], str)})
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


def test_no_form_callback_can_reach_itself_through_other_callbacks(app):
    graph = _graph(app)
    mine = [key for key, (outputs, inputs) in graph.items()
            if any(cid.startswith("ohlf-") for cid, _ in outputs | inputs)]
    assert len(mine) >= 8
    for key in mine:
        assert key not in _reachable(graph, key), key


def test_the_yaml_text_is_never_an_input_of_a_callback_that_writes_it(app):
    text = (ids.OHL_PROFILE_TEXT, "value")
    graph = _graph(app)
    for key, (outputs, inputs) in graph.items():
        assert not (text in outputs and text in inputs), key


def test_the_form_to_yaml_callback_does_not_feed_the_fill_callbacks(app):
    graph = _graph(app)
    edit_key = next(k for k, (o, i) in graph.items() if (ids.OHLF_IDS["name"], "value") in i)
    after = _reachable(graph, edit_key)
    for key in after:
        outputs, _ = graph[key]
        assert (ids.OHLF_IDS["name"], "value") not in outputs


# ================================================================== the callbacks are the only writers
def test_the_existing_preset_and_load_callbacks_are_not_replaced(app):
    """The old callbacks still write the text; the form callbacks only add to them."""
    text_writers = [k for k in app.callback_map if f"{ids.OHL_PROFILE_TEXT}.value" in k]
    assert len(text_writers) >= 3          # load_preset, load_profile_file, form_to_yaml (and the restore)


def test_copies_of_the_presets_survive_a_form_round_trip():
    """The idempotence the "nothing changed" check relies on."""
    for name in PRESET_NAMES:
        profile = _preset_profile(name)
        merged = service.profile_from_form(*service.form_values(profile), base=copy.deepcopy(profile))
        assert merged == profile, name

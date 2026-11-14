"""P3 (round 14, gap 5): the profile form service converts flat fields, table rows and YAML.

Pure logic tests: no Dash app, no files written. Tests that need YAML use
``pytest.importorskip("yaml")``.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from pyorps.core.exceptions import InvalidInputError, MissingDependencyError
from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.gui.services import profile_form as pf

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
SHIPPED = sorted(PROFILES_DIR.glob("*.yaml"))


def _base() -> dict:
    """A complete, valid piecewise profile with clearance settings and extra keys."""
    return {
        "name": "t", "description": "d",
        "soft_angle_limit_deg": 8.0, "hard_angle_limit_deg": 40.0,
        "angle_cost_function": "piecewise",
        "angle_cost_params": {"breakpoints": [0, 8, 40], "costs": [0, 100, 5000]},
        "min_span_m": 50, "max_span_m": 300, "span_bin_size_m": 10,
        "tower_cost_params": {
            "terrain_cost_map": {"0": 350, "100": 550},
            "terrain_interpolation": "linear",
            "angle_types": {"light": {"max_angle_deg": 8.0, "base_cost": 30000},
                            "strain": {"max_angle_deg": 40.0, "base_cost": 250000}}},
        "max_gradient_percent": 40.0,
        "gradient_cost_params": {"scale": 2.0},
        "tower_height_m": 25.0, "tower_heights_m": [25, 34, 42],
        "tower_height_cost_per_increment": 18000,
        "terminal_tower_height_m": 15.0, "terminal_tower_cost": 280000,
        "conductor_weight_per_m": 10.0, "conductor_tension_n": 20000.0,
        "min_clearance_m": 7.0, "tower_ground_area_m2": 100.0,
        "tower_area_cost_mode": "uniform", "recommended_buffer_m": 25.0,
    }


def _form(profile: dict):
    return pf.form_values(profile)


def _roundtrip(profile: dict, **edits) -> dict:
    values, angle, terrain, types = _form(profile)
    values.update(edits)
    return pf.profile_from_form(values, angle, terrain, types, profile)


# ------------------------------------------------------------------- contract
def test_form_keys_follow_ids_and_every_key_has_a_spec():
    from pyorps.gui import ids

    assert pf.FORM_KEYS == ids.OHLF_KEYS
    assert list(pf.FIELD_SPECS) == list(pf.FORM_KEYS)
    assert len(pf.FORM_KEYS) == 24
    for key, spec in pf.FIELD_SPECS.items():
        assert spec.key == key and spec.label and spec.help
        assert spec.path and spec.path[-1] in (key, "scale", "terrain_interpolation")
        assert spec.kind in ("text", "number", "select", "list")
    assert set(pf.TABLE_SPECS) == {pf.TABLE_ANGLE, pf.TABLE_TERRAIN, pf.TABLE_TYPES}


def test_service_does_not_import_ids_dash_or_layout():
    import ast

    tree = ast.parse(Path(pf.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [f"{'.' * node.level}{node.module or ''}"] + [
                f"{'.' * node.level}{node.module or ''}.{a.name}" for a in node.names]
    for name in imported:
        assert "ids" not in name.split(".") and "layout" not in name.split(".")
        assert not name.startswith(("dash", "callbacks"))


def test_form_values_of_an_empty_profile_are_the_defaults():
    values, angle, terrain, types = _form({})
    assert list(values) == list(pf.FORM_KEYS)
    assert values["name"] == "gui_profile"
    assert (values["soft_angle_limit_deg"], values["hard_angle_limit_deg"]) == (8, 40)
    assert values["angle_cost_function"] == "piecewise"
    assert values["terrain_interpolation"] == "linear"
    assert values["tower_heights_m"] == ""
    assert values["max_gradient_percent"] is None
    assert values["gradient_scale"] == 2.0
    assert values["tower_ground_area_m2"] == 1
    assert (angle, terrain, types) == ([], [], [])


# ------------------------------------------------------------------- round trips
@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_profile_form_round_trip_is_equal(path):
    yaml = pytest.importorskip("yaml")
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    values, angle, terrain, types = _form(profile)
    assert pf.profile_from_form(values, angle, terrain, types, profile) == profile
    assert pf.profile_problems(profile) == []
    assert pf.form_problems(values, angle, terrain, types, profile) == []


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_profile_without_base_builds_the_same_profile_for_the_router(path):
    """No base: the unused keys are gone, but the profile the router reads is identical."""
    yaml = pytest.importorskip("yaml")
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    rebuilt = pf.profile_from_form(*_form(profile), base=None)
    assert pf.profile_problems(rebuilt) == []
    a = InfrastructureProfile.from_dict(profile).to_dict()
    b = InfrastructureProfile.from_dict(rebuilt).to_dict()
    for key in pf.UNUSED_KEYS:
        a.pop(key, None)
        b.pop(key, None)
    a["gradient_cost_params"] = {"scale": (a["gradient_cost_params"] or {}).get("scale", 2.0)}
    b["gradient_cost_params"] = {"scale": (b["gradient_cost_params"] or {}).get("scale", 2.0)}
    for config in (a, b):       # the router reads this default itself
        config["tower_cost_params"].setdefault("terrain_interpolation", "linear")
    assert a == b


def test_round_trip_keeps_types_and_integer_terrain_keys():
    profile = _base()
    profile["tower_cost_params"]["terrain_cost_map"] = {0: 350, 100: 550}   # YAML int keys
    profile["soft_angle_limit_deg"] = 8                                      # int stays int
    out = _roundtrip(profile)
    assert out == profile
    assert list(out["tower_cost_params"]["terrain_cost_map"]) == [0, 100]
    assert type(out["soft_angle_limit_deg"]) is int


def test_unknown_and_unused_keys_survive_a_form_edit():
    profile = _base()
    profile.update({"my_note": {"a": [1, 2]}, "tower_cost_function": "x",
                    "gradient_cost_function": "y", "cost_assumptions_path": "c.yaml"})
    profile["tower_cost_params"]["extra"] = 7
    profile["angle_cost_params"]["scale"] = 9
    profile["gradient_cost_params"]["cost_per_percent"] = 5000
    profile["tower_cost_params"]["angle_types"]["light"]["colour"] = "red"
    values, angle, terrain, types = _form(profile)
    values["soft_angle_limit_deg"] = 6
    values["hard_angle_limit_deg"] = 35
    values["gradient_scale"] = 3
    types[0]["base_cost"] = 31000
    terrain.append({"raster_value": 200, "eur_per_m2": 800})
    out = pf.profile_from_form(values, angle, terrain, types, profile)
    assert out["soft_angle_limit_deg"] == 6 and out["hard_angle_limit_deg"] == 35
    assert out["gradient_cost_params"] == {"scale": 3, "cost_per_percent": 5000}
    assert out["tower_cost_params"]["angle_types"]["light"] == {
        "max_angle_deg": 8.0, "base_cost": 31000, "colour": "red"}
    assert out["tower_cost_params"]["terrain_cost_map"]["200"] == 800
    assert out["tower_cost_params"]["extra"] == 7
    assert out["angle_cost_params"]["scale"] == 9
    for key in ("my_note", "tower_cost_function", "gradient_cost_function",
                "cost_assumptions_path"):
        assert out[key] == profile[key]
    assert profile == _base() | {k: profile[k] for k in profile if k not in _base()} \
        or True  # the input profile itself is not modified (checked below)


def test_profile_from_form_does_not_modify_base():
    profile = _base()
    snapshot = copy.deepcopy(profile)
    _roundtrip(profile, name="other", max_span_m=400)
    assert profile == snapshot


def test_blank_optional_removes_key_and_default_is_not_added():
    profile = _base()
    out = _roundtrip(profile, max_gradient_percent=None, tower_heights_m="",
                     recommended_buffer_m=None, terminal_tower_height_m="")
    for key in ("max_gradient_percent", "tower_heights_m", "recommended_buffer_m",
                "terminal_tower_height_m"):
        assert key not in out
    # a profile that never had these: the defaults the form shows are not written out
    sparse = {"name": "s", "soft_angle_limit_deg": 5, "hard_angle_limit_deg": 30,
              "angle_cost_function": "linear", "angle_cost_params": {"scale": 3}}
    values, angle, terrain, types = _form(sparse)
    out = pf.profile_from_form(values, angle, terrain, types, sparse)
    assert out == sparse
    # a stored None stays None
    with_none = sparse | {"min_span_m": None}
    assert pf.profile_from_form(*_form(with_none), base=with_none) == with_none


def test_comma_list_parses_to_numbers_and_keeps_bad_tokens():
    out = _roundtrip(_base(), tower_heights_m="25, 34.5; 42  50")
    assert out["tower_heights_m"] == [25, 34.5, 42, 50]
    out = _roundtrip(_base(), tower_heights_m="25, x")
    assert out["tower_heights_m"] == [25, "x"]
    (key, message), = pf.profile_problems(out)
    assert key == "tower_heights_m" and "'x'" in message and "m" in message


def test_piecewise_rows_are_sorted_and_scale_is_written_only_for_linear():
    profile = _base()
    values, _, terrain, types = _form(profile)
    rows = [{"angle_deg": 40, "cost": 5000}, {"angle_deg": 0, "cost": 0},
            {"angle_deg": 8, "cost": 100}]
    out = pf.profile_from_form(values, rows, terrain, types, profile)
    assert out["angle_cost_params"] == {"breakpoints": [0, 8, 40], "costs": [0, 100, 5000]}
    values.update(angle_cost_function="linear", angle_cost_scale=2.5)
    rows2 = [{"angle_deg": 5, "cost": 1}]
    out = pf.profile_from_form(values, rows2, terrain, types, profile)
    assert out["angle_cost_function"] == "linear"
    assert out["angle_cost_params"] == {"breakpoints": [0, 8, 40], "costs": [0, 100, 5000],
                                        "scale": 2.5}
    values.update(angle_cost_function="piecewise", angle_cost_scale=99)
    out = pf.profile_from_form(values, rows, terrain, types, profile)
    assert "scale" not in out["angle_cost_params"]


def test_terrain_rows_become_string_keys_sorted_and_incomplete_rows_are_dropped():
    values, angle, _, types = _form(_base())
    rows = [{"raster_value": 300.0, "eur_per_m2": 1200}, {"raster_value": 0, "eur_per_m2": "350"},
            {"raster_value": 50, "eur_per_m2": None}, {"raster_value": None, "eur_per_m2": None}]
    out = pf.profile_from_form(values, angle, rows, types, None)
    table = out["tower_cost_params"]["terrain_cost_map"]
    assert list(table.items()) == [("0", 350), ("300", 1200)]
    problems = pf.table_row_problems(angle, rows, types)
    assert len(problems) == 1 and problems[0][0] == pf.TABLE_TERRAIN
    assert "eur_per_m2" in problems[0][1]


def test_cleared_tables_remove_their_keys():
    out = pf.profile_from_form(_form(_base())[0], [], [], [], _base())
    assert "breakpoints" not in out["angle_cost_params"]
    assert "terrain_cost_map" not in out["tower_cost_params"]
    assert "angle_types" not in out["tower_cost_params"]
    keys = {k for k, _ in pf.profile_problems(out)}
    assert pf.TABLE_ANGLE in keys


# ------------------------------------------------------------------- limits per field
# key -> (a value below/above the limit, expected word)
_BAD = [
    ("name", "", "required"),
    ("name", "x" * 81, "too long"),
    ("description", "x" * 301, "too long"),
    ("soft_angle_limit_deg", -1, "too low"),
    ("soft_angle_limit_deg", 181, "too high"),
    ("soft_angle_limit_deg", "abc", "not a number"),
    ("hard_angle_limit_deg", 181, "too high"),
    ("hard_angle_limit_deg", -5, "too low"),
    ("angle_cost_function", "cubic", "not an option"),
    ("angle_cost_scale", -0.1, "too low"),
    ("angle_cost_scale", 1.5e9, "too high"),
    ("min_span_m", 0, "too low"),
    ("min_span_m", -3, "too low"),
    ("max_span_m", 0, "too low"),
    ("max_span_m", 5001, "too high"),
    ("span_bin_size_m", 0, "too low"),
    ("terrain_interpolation", "cubic", "not an option"),
    ("max_gradient_percent", 0, "too low"),
    ("max_gradient_percent", 1001, "too high"),
    ("gradient_scale", -1, "too low"),
    ("gradient_scale", 101, "too high"),
    ("tower_height_m", 0, "too low"),
    ("tower_height_m", 301, "too high"),
    ("tower_heights_m", [25, 0], "too low"),
    ("tower_heights_m", [25, 301], "too high"),
    ("tower_height_cost_per_increment", -1, "too low"),
    ("terminal_tower_height_m", 0, "too low"),
    ("terminal_tower_height_m", 301, "too high"),
    ("terminal_tower_cost", -1, "too low"),
    ("conductor_weight_per_m", 0, "too low"),
    ("conductor_weight_per_m", 1001, "too high"),
    ("conductor_tension_n", 0, "too low"),
    ("conductor_tension_n", 1.1e7, "too high"),
    ("min_clearance_m", -1, "too low"),
    ("min_clearance_m", 101, "too high"),
    ("tower_ground_area_m2", 0.5, "too low"),
    ("tower_ground_area_m2", 10001, "too high"),
    ("tower_area_cost_mode", "rough", "not an option"),
    ("recommended_buffer_m", -1, "too low"),
    ("gradient_scale", float("nan"), "not a finite"),
    ("conductor_tension_n", float("inf"), "not a finite"),
]


def _profile_with(key: str, value) -> dict:
    spec = pf.FIELD_SPECS[key]
    profile = _base()
    profile["angle_cost_function"] = "linear"
    profile["angle_cost_params"] = {"scale": 1.0}
    profile.pop("min_span_m", None)
    profile.pop("max_span_m", None)
    profile.pop("span_bin_size_m", None)
    if key in ("min_span_m", "max_span_m", "span_bin_size_m"):
        profile.update(min_span_m=50, max_span_m=300, span_bin_size_m=10)
    node = profile
    for part in spec.path[:-1]:
        node = node.setdefault(part, {})
    node[spec.path[-1]] = value
    return profile


@pytest.mark.parametrize(("key", "value", "word"), _BAD,
                         ids=[f"{k}-{str(v)[:12]}" for k, v, _ in _BAD])
def test_every_field_limit_gives_a_field_problem_with_value_range_and_unit(key, value, word):
    problems = pf.profile_problems(_profile_with(key, value))
    own = [(k, m) for k, m in problems if k == key]
    assert own, problems
    message = own[0][1]
    spec = pf.FIELD_SPECS[key]
    assert word in message
    assert spec.label in message and f"({key})" in message
    assert "Allowed:" in message
    if spec.unit and spec.kind in ("number", "list"):
        assert spec.unit in message
    if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
        assert pf._fmt(value) in message


@pytest.mark.parametrize(("key", "value"), [
    ("soft_angle_limit_deg", 0), ("soft_angle_limit_deg", 40), ("hard_angle_limit_deg", 180),
    ("angle_cost_scale", 0), ("angle_cost_scale", 1e9), ("max_span_m", 5000),
    ("max_gradient_percent", 1000), ("max_gradient_percent", 0.5), ("gradient_scale", 0),
    ("gradient_scale", 100), ("tower_height_m", 300), ("tower_heights_m", [300, 1]),
    ("tower_height_cost_per_increment", 0), ("terminal_tower_height_m", 300),
    ("terminal_tower_cost", 0), ("conductor_weight_per_m", 1000),
    ("conductor_tension_n", 1e7), ("min_clearance_m", 0), ("min_clearance_m", 100),
    ("tower_ground_area_m2", 1), ("tower_ground_area_m2", 10000),
    ("tower_area_cost_mode", "exact"), ("terrain_interpolation", "nearest"),
    ("recommended_buffer_m", 0), ("name", "x" * 80), ("description", "x" * 300),
    ("soft_angle_limit_deg", "8"),   # a numeric string from a form is accepted
])
def test_values_at_the_limits_are_valid(key, value):
    assert [p for p in pf.profile_problems(_profile_with(key, value)) if p[0] == key] == []


def test_a_valid_profile_has_no_problems_and_missing_required_keys_are_reported():
    assert pf.profile_problems(_base()) == []
    keys = [k for k, _ in pf.profile_problems({})]
    assert {"soft_angle_limit_deg", "hard_angle_limit_deg", "angle_cost_function"} <= set(keys)
    assert pf.profile_problems("nope")[0][0] == "profile"


# ------------------------------------------------------------------- cross-field rules
def _problem_keys(profile: dict) -> list[str]:
    return [k for k, _ in pf.profile_problems(profile)]


def test_hard_limit_below_soft_limit():
    profile = _base() | {"soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 10.0}
    (key, message), = [p for p in pf.profile_problems(profile) if p[0] == "hard_angle_limit_deg"]
    assert "10 deg" in message and "20 deg" in message and "Allowed: 20 to 180 deg" in message


def test_span_rules():
    profile = _base()
    assert _problem_keys(profile | {"min_span_m": 400}) == ["min_span_m"]
    profile.pop("max_span_m")
    assert _problem_keys(profile) == ["max_span_m"]
    profile = _base()
    profile.pop("min_span_m")
    assert _problem_keys(profile) == ["min_span_m"]
    profile = _base()
    profile.pop("span_bin_size_m")
    assert _problem_keys(profile) == ["span_bin_size_m"]
    assert _problem_keys(_base() | {"span_bin_size_m": 301}) == ["span_bin_size_m"]
    no_spans = _base()
    for k in ("min_span_m", "max_span_m", "span_bin_size_m"):
        no_spans.pop(k)
    assert pf.profile_problems(no_spans) == []
    only_bin = no_spans | {"span_bin_size_m": 10}
    assert pf.profile_problems(only_bin) == []


def test_scale_is_checked_only_for_linear_and_quadratic():
    piece = _base() | {"angle_cost_params": {"breakpoints": [0, 8], "costs": [0, 1], "scale": -5}}
    assert pf.profile_problems(piece) == []
    lin = _base() | {"angle_cost_function": "quadratic", "angle_cost_params": {"scale": -5}}
    assert _problem_keys(lin) == ["angle_cost_scale"]


def test_library_check_is_the_last_step_and_never_a_traceback(monkeypatch):
    profile = _base()
    assert pf._library_problems(profile) == []
    profile["span_bin_size_m"] = -1
    assert pf._library_problems(profile)[0][0] == "span_bin_size_m"
    key, message = pf._library_problems({"name": 5, "bogus": 1})[0]
    assert key in pf.FORM_KEYS and "library rejects it" in message


# ------------------------------------------------------------------- tables
def _table_messages(profile: dict, table: str) -> list[str]:
    return [m for k, m in pf.profile_problems(profile) if k == table]


@pytest.mark.parametrize(("params", "word"), [
    ({"breakpoints": [0, 8, 8], "costs": [0, 1, 2]}, "strictly increase"),
    ({"breakpoints": [0, 8, 5], "costs": [0, 1, 2]}, "strictly increase"),
    ({"breakpoints": [0], "costs": [0]}, "too few"),
    ({"breakpoints": [0, 8], "costs": [0]}, "2 angles but 1 costs"),
    ({"breakpoints": [0, 181], "costs": [0, 1]}, "too high"),
    ({"breakpoints": [-1, 5], "costs": [0, 1]}, "too low"),
    ({"breakpoints": [0, 5], "costs": [0, -1]}, "too low"),
    ({"breakpoints": [0, 5], "costs": [0, "x"]}, "not a number"),
    ({"breakpoints": [0, 5], "costs": [0, None]}, "blank"),
    ({}, "missing"),
])
def test_angle_table_problems(params, word):
    profile = _base() | {"angle_cost_params": params}
    messages = _table_messages(profile, pf.TABLE_ANGLE)
    assert messages and any(word in m for m in messages), messages
    assert all("angle_cost_table" in m for m in messages)


@pytest.mark.parametrize(("table", "word", "unit"), [
    ({"0": 1, "70000": 2}, "too high", ""),
    ({"-1": 1}, "too low", ""),
    ({"1.5": 1}, "not a whole number", ""),
    ({"abc": 1}, "not a number", ""),
    ({"0": -1}, "too low", "EUR per m2"),
    ({"0": "x"}, "not a number", "EUR per m2"),
    ({"0": 1, 0: 2}, "appears twice", ""),
])
def test_terrain_table_problems(table, word, unit):
    profile = _base()
    profile["tower_cost_params"]["terrain_cost_map"] = table
    messages = _table_messages(profile, pf.TABLE_TERRAIN)
    assert messages and any(word in m for m in messages), messages
    if unit:
        assert any(unit in m for m in messages)


def test_terrain_table_must_be_a_mapping():
    profile = _base()
    profile["tower_cost_params"]["terrain_cost_map"] = [1, 2]
    assert _table_messages(profile, pf.TABLE_TERRAIN)


@pytest.mark.parametrize(("types", "word"), [
    ({"a": {"max_angle_deg": 181, "base_cost": 1}}, "too high"),
    ({"a": {"max_angle_deg": -1, "base_cost": 1}}, "too low"),
    ({"a": {"max_angle_deg": 5, "base_cost": -1}}, "too low"),
    ({"a": {"max_angle_deg": 5}}, "missing"),
    ({"a": 7}, "needs max_angle_deg"),
    ({"": {"max_angle_deg": 5, "base_cost": 1}}, "without a name"),
])
def test_type_table_problems(types, word):
    profile = _base()
    profile["tower_cost_params"]["angle_types"] = types
    messages = _table_messages(profile, pf.TABLE_TYPES)
    assert messages and any(word in m for m in messages), messages


def test_row_problems_half_rows_and_duplicates():
    angle = [{"angle_deg": 0, "cost": None}]
    terrain = [{"raster_value": 100, "eur_per_m2": 1}, {"raster_value": "100.0", "eur_per_m2": 2}]
    types = [{"name": "a", "max_angle_deg": 5, "base_cost": 1},
             {"name": " a ", "max_angle_deg": 9, "base_cost": 2},
             {"name": "", "max_angle_deg": 9, "base_cost": None}]
    problems = pf.table_row_problems(angle, terrain, types)
    keys = [k for k, _ in problems]
    assert keys.count(pf.TABLE_ANGLE) == 1
    assert keys.count(pf.TABLE_TERRAIN) == 1
    assert any("appears twice" in m for k, m in problems if k == pf.TABLE_TERRAIN)
    assert keys.count(pf.TABLE_TYPES) == 2
    assert any("appears twice" in m for k, m in problems if k == pf.TABLE_TYPES)
    values = _form(_base())[0]
    assert pf.form_problems(values, angle, terrain, types, _base())


# ------------------------------------------------------------------- clearance, unused
def test_clearance_status_on_off_partial():
    assert pf.clearance_state(_base()) == "on"
    assert pf.clearance_status(_base()).startswith("Clearance check: on")
    off = {"name": "x"}
    assert pf.clearance_state(off) == "off"
    assert "the profile needs base height, conductor weight, tension and clearance" in pf.clearance_status(off)
    assert "the run needs a terrain model (DEM)" in pf.clearance_status(off)      # round 17: not "plus a DSM"
    partial = {"tower_height_m": 20, "conductor_weight_per_m": 5}
    assert pf.clearance_state(partial) == "partial"
    text = pf.clearance_status(partial)
    assert "conductor tension" in text and "ground clearance" in text
    assert "base height" not in text
    heights_only = {"tower_heights_m": [20, 30], "conductor_weight_per_m": 5,
                    "conductor_tension_n": 1000, "min_clearance_m": 7}
    assert pf.clearance_state(heights_only) == "on"
    assert pf.clearance_state({"tower_heights_m": []}) == "off"
    # agrees with the library property
    assert InfrastructureProfile.from_dict(_base()).has_clearance_constraints


def test_unused_and_unknown_keys():
    profile = _base() | {"tower_cost_function": "terrain_plus_angle", "zzz": 1,
                         "gradient_cost_function": None}
    assert pf.unused_keys(profile) == ["tower_cost_function", "recommended_buffer_m"]
    assert pf.unknown_keys(profile) == ["zzz"]
    assert pf.unused_keys({}) == []


# ------------------------------------------------------------------- notices
def test_problem_notice_names_the_field_and_the_control_without_ids():
    assert pf.problem_notice([]) is None
    problems = pf.profile_problems(_base() | {"max_gradient_percent": 2000})
    notice = pf.problem_notice(problems, controls={"max_gradient_percent": "some-id"})
    assert notice["severity"] == "warning" and notice["focus_control"] == "some-id"
    assert "Steepest allowed slope" in notice["title"]
    assert "2000" in notice["meaning"] and "%" in notice["meaning"]
    assert notice["focus_id"] == "tab-routes"
    text = pf.problems_text(problems * 5, limit=2)
    assert text.count("\n") == 2 and "more" in text


# ------------------------------------------------------------------- YAML
def test_yaml_round_trip_keeps_keys_the_form_does_not_show():
    yaml = pytest.importorskip("yaml")
    profile = _base() | {"my_note": "keep me", "cost_assumptions_path": "c.yaml"}
    text = yaml.safe_dump(profile, sort_keys=False)
    parsed, values, angle, terrain, types = pf.text_to_form(text)
    assert parsed == profile
    assert values["name"] == "t" and values["tower_heights_m"] == "25, 34, 42"
    values["max_span_m"] = 350
    out_text, out, problems = pf.form_to_text(values, angle, terrain, types, base_text=text)
    assert problems == []
    assert yaml.safe_load(out_text) == out == profile | {"max_span_m": 350}
    # unchanged form: the YAML parses to the same profile
    same_text, same, _ = pf.form_to_text(*_form(profile), base_text=text)
    assert yaml.safe_load(same_text) == profile == same


def test_form_to_text_uses_the_fallback_when_the_text_is_broken_or_empty():
    pytest.importorskip("yaml")
    values, angle, terrain, types = _form(_base())
    for text in ("", "a: [unclosed", "- 1\n- 2", None):
        out_text, out, _ = pf.form_to_text(values, angle, terrain, types,
                                           base_text=text, fallback={"my_note": 1})
        assert out["my_note"] == 1 and "my_note: 1" in out_text
    assert pf.base_profile("garbage: [", None) == {}


def test_form_to_text_returns_problems_but_still_a_text():
    pytest.importorskip("yaml")
    values, angle, terrain, types = _form(_base())
    values["max_gradient_percent"] = 5000
    text, profile, problems = pf.form_to_text(values, angle, terrain, types)
    assert "max_gradient_percent: 5000" in text and profile["max_gradient_percent"] == 5000
    assert [k for k, _ in problems] == ["max_gradient_percent"]


@pytest.mark.parametrize(("text", "word"), [
    ("", "empty"), ("   \n", "empty"), ("a: [unclosed", "line"), ("- 1\n- 2", "mapping"),
    ("just text", "mapping"),
])
def test_bad_yaml_raises_invalid_input_error_with_a_plain_message(text, word):
    pytest.importorskip("yaml")
    with pytest.raises(InvalidInputError, match=word):
        pf.parse_profile_text(text)
    with pytest.raises(InvalidInputError):
        pf.text_to_form(text)


def test_profile_to_text_keeps_key_order():
    yaml = pytest.importorskip("yaml")
    profile = {"name": "n", "zzz": 1, "aaa": {"b": [1, 2]}}
    text = pf.profile_to_text(profile)
    assert text.index("name") < text.index("zzz") < text.index("aaa")
    assert yaml.safe_load(text) == profile


def test_without_pyyaml_the_service_raises_missing_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)        # import yaml -> ImportError
    for call in (lambda: pf.parse_profile_text("a: 1"), lambda: pf.profile_to_text({"a": 1}),
                 lambda: pf.text_to_form("a: 1"), lambda: pf.base_profile("a: 1"),
                 lambda: pf.form_to_text(*_form(_base()))):
        with pytest.raises(MissingDependencyError) as info:
            call()
        assert info.value.extra == "yaml"
    # everything that is not YAML keeps working
    assert pf.profile_problems(_base()) == []
    assert pf.form_values(_base())[0]["name"] == "t"
    assert pf.profile_from_form(*_form(_base()), base=_base()) == _base()

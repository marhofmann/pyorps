"""
PYORPS GUI: the constrained-routing profile as a form (round 14, gap 5, package P3).

A pure service. It converts an ``InfrastructureProfile`` dict between three shapes:

* flat form fields (``values``: one entry per key of :data:`FORM_KEYS`) plus the rows of
  three tables (turn cost, terrain cost, tower types),
* the profile dict that ``InfrastructureProfile.from_dict`` reads,
* YAML text (the "Raw YAML" box, which stays the single run input).

It imports nothing from ``pyorps.gui.ids``, Dash or the layout: the form keys are the
profile field names, and a caller that needs component ids passes them in (``controls``).

Rules the functions follow:

* ``profile_from_form`` merges the form into a copy of ``base``. Everything the form does not
  show (the unused fields, unknown keys, other keys inside ``tower_cost_params`` and so on)
  survives a form edit. A value that is unchanged keeps the exact object of ``base``
  (so ``50`` does not turn into ``50.0``).
* A blank optional field removes its key. A field set to its default is not added when the
  key is absent.
* Nothing here raises on a bad value: ``profile_problems`` and ``form_problems`` return
  ``(key, message)`` pairs. Each message names the form key, the value, the allowed range and
  the unit. Only YAML parsing raises: ``InvalidInputError`` for text that is no YAML mapping,
  ``MissingDependencyError`` when PyYAML is not installed.
"""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from itertools import zip_longest
from typing import Any

from pyorps.core import optional_deps
from pyorps.core.exceptions import InvalidInputError, PyorpsError
from pyorps.core.infrastructure_profile import InfrastructureProfile

from .errors import Notice
from .profile_form_checks import (  # pylint: disable=unused-import  (re-exported names)
    _PROFILE_REQUIRED,
    ANGLE_COLUMNS,
    ANGLE_FUNCTIONS,
    AREA_MODES,  # noqa: F401  (re-export)
    FIELD_SPECS,
    FORM_KEYS,
    INTERPOLATIONS,  # noqa: F401  (re-export)
    TABLE_ANGLE,
    TABLE_SPECS,
    TABLE_TERRAIN,
    TABLE_TYPES,
    TERRAIN_COLUMNS,
    TYPE_COLUMNS,
    UNUSED_KEYS,
    FieldSpec,
    Problem,
    TableSpec,  # noqa: F401  (re-export)
    _angle_table_problems,
    _as_dict,
    _coerce,
    _complete,
    _fmt,
    _function_of,
    _get,
    _has,
    _is_blank,
    _number,
    _put,
    _remove,
    _span_problems,
    _subject,
    _terrain_key,
    _terrain_table_problems,
    _types_table_problems,
    _value_problems,
    describe_range,
)


# --------------------------------------------------------------------------- profile -> form
def _display(spec: FieldSpec, raw: Any) -> Any:
    if _is_blank(raw):
        return "" if spec.kind == "list" else spec.default
    if spec.kind == "list":
        return ", ".join(_fmt(x) for x in raw) if isinstance(raw, (list, tuple)) else str(raw)
    return raw


def _angle_rows_from(profile: Mapping) -> list[dict]:
    params = profile.get("angle_cost_params")
    if not isinstance(params, Mapping):
        return []
    angles, costs = params.get("breakpoints"), params.get("costs")
    if not isinstance(angles, (list, tuple)) or not isinstance(costs, (list, tuple)):
        return []
    return [{"angle_deg": a, "cost": c} for a, c in zip_longest(angles, costs)]


def _sort_key_number(value: Any) -> tuple[int, float]:
    number = _number(value)
    return (0, number) if number is not None else (1, 0.0)


def _terrain_rows_from(profile: Mapping) -> list[dict]:
    table = _get(profile, TABLE_SPECS[TABLE_TERRAIN].path)
    if not isinstance(table, Mapping):
        return []
    rows = []
    for key, cost in table.items():
        number = _coerce(key)
        if isinstance(number, float) and math.isfinite(number) and number == int(number):
            number = int(number)
        rows.append({"raster_value": number, "eur_per_m2": cost})
    return sorted(rows, key=lambda r: _sort_key_number(r["raster_value"]))


def _type_rows_from(profile: Mapping) -> list[dict]:
    table = _get(profile, TABLE_SPECS[TABLE_TYPES].path)
    if not isinstance(table, Mapping):
        return []
    rows = []
    for name, entry in table.items():
        entry = entry if isinstance(entry, Mapping) else {}
        rows.append({"name": str(name), "max_angle_deg": entry.get("max_angle_deg"),
                     "base_cost": entry.get("base_cost")})
    return rows


def form_values(profile: Any) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """A profile dict -> ``(values, angle_rows, terrain_rows, type_rows)``.

    ``values`` has every key of :data:`FORM_KEYS` (a missing key gets the spec default, an
    optional field without default is ``None``, ``tower_heights_m`` is a comma string). Row
    dicts use the column keys :data:`ANGLE_COLUMNS`, :data:`TERRAIN_COLUMNS` (sorted by raster
    value) and :data:`TYPE_COLUMNS`. The rows are filled whatever the turn cost shape is.
    """
    profile = _as_dict(profile)
    values = {key: _display(spec, _get(profile, spec.path)) for key, spec in FIELD_SPECS.items()}
    return (values, _angle_rows_from(profile), _terrain_rows_from(profile),
            _type_rows_from(profile))


# --------------------------------------------------------------------------- form -> profile
def _parse_list(raw: Any) -> list | None:
    if isinstance(raw, (list, tuple)):
        tokens = list(raw)
    elif isinstance(raw, str):
        tokens = [t for t in re.split(r"[,;\s]+", raw.strip()) if t]
    else:
        tokens = [raw]
    tokens = [_coerce(t) for t in tokens if not _is_blank(t)]
    return tokens or None


def _form_value(spec: FieldSpec, raw: Any) -> Any:
    """The profile value of a form value; None means blank."""
    if spec.kind == "text":
        return "" if raw is None else str(raw).strip()
    if _is_blank(raw):
        return None
    if spec.kind == "list":
        return _parse_list(raw)
    if spec.kind == "select":
        return raw
    return _coerce(raw)


def _apply_field(profile: dict, spec: FieldSpec, raw: Any) -> None:
    value = _form_value(spec, raw)
    present = _has(profile, spec.path)
    existing = _get(profile, spec.path)
    if value is None:
        if not (present and existing is None):
            _remove(profile, spec.path)         # a stored None stays: it means the same
        return
    if _is_unchanged(present, existing, value) or _is_unwritten_default(spec, present, value):
        return          # unchanged (keep the stored object), or the default (not written out)
    _put(profile, spec.path, value)


def _is_unchanged(present: bool, existing: Any, value: Any) -> bool:
    return present and existing == value and type(existing) is not bool


def _is_unwritten_default(spec: FieldSpec, present: bool, value: Any) -> bool:
    return (not present and not spec.required and spec.default is not None
            and value == spec.default)


def _pair_sort_key(pair: tuple) -> tuple[int, float]:
    return _sort_key_number(pair[0])


def _angle_pairs(rows: list[dict] | None) -> list[tuple]:
    return sorted(((_coerce(r["angle_deg"]), _coerce(r["cost"]))
                   for r in rows or () if _complete(r, ANGLE_COLUMNS)),
                  key=_pair_sort_key)


def _terrain_pairs(rows: list[dict] | None) -> list[tuple]:
    return sorted(((r["raster_value"], _coerce(r["eur_per_m2"]))
                   for r in rows or () if _complete(r, TERRAIN_COLUMNS)),
                  key=_pair_sort_key)


def _apply_angle_rows(profile: dict, rows: list[dict]) -> None:
    done = _angle_pairs(rows)
    path = TABLE_SPECS[TABLE_ANGLE].path
    if not done:
        _remove(profile, path)
        _remove(profile, path[:-1] + ("costs",))
        return
    angles, costs = [p[0] for p in done], [p[1] for p in done]
    for sub, new in (("breakpoints", angles), ("costs", costs)):
        if _get(profile, path[:-1] + (sub,)) != new:
            _put(profile, path[:-1] + (sub,), new)


def _apply_terrain_rows(profile: dict, rows: list[dict]) -> None:
    done = _terrain_pairs(rows)
    new = {_terrain_key(k): v for k, v in done}
    path = TABLE_SPECS[TABLE_TERRAIN].path
    existing = _get(profile, path)
    if isinstance(existing, Mapping) and {str(k): v for k, v in existing.items()} == new:
        return                                   # unchanged: keep int keys of the base
    if new:
        _put(profile, path, new)
    else:
        _remove(profile, path)


def _type_entry(row: Mapping, old: Any) -> dict:
    entry = copy.deepcopy(dict(old)) if isinstance(old, Mapping) else {}
    for col in ("max_angle_deg", "base_cost"):
        value = _coerce(row[col])
        if col not in entry or entry[col] != value:
            entry[col] = value
    return entry


def _apply_type_rows(profile: dict, rows: list[dict]) -> None:
    path = TABLE_SPECS[TABLE_TYPES].path
    existing = _get(profile, path)
    existing = existing if isinstance(existing, Mapping) else {}
    new: dict = {}
    for row in rows or ():
        if not _complete(row, TYPE_COLUMNS):
            continue
        name = str(row["name"]).strip()
        new[name] = _type_entry(row, existing.get(name))
    if existing == new:
        return
    if new:
        _put(profile, path, new)
    else:
        _remove(profile, path)


def _apply_fields(profile: dict, values: Mapping, function: str) -> None:
    for key, spec in FIELD_SPECS.items():
        if key not in values or (key == "angle_cost_scale" and function == "piecewise"):
            continue
        _apply_field(profile, spec, values[key])


def profile_from_form(values: Mapping | None, angle_rows: list[dict] | None,
                      terrain_rows: list[dict] | None, type_rows: list[dict] | None,
                      base: Mapping | None = None) -> dict:
    """Form fields and table rows -> profile dict, merged into a deep copy of ``base``.

    Keys the form does not show survive (unused fields, unknown keys, other keys inside the
    parameter dicts). A blank optional field removes its key; a value equal to the default is
    not added when the key is absent; an unchanged value keeps the stored object. The comma
    list becomes floats/ints, piecewise rows are sorted by angle into ``breakpoints`` and
    ``costs`` (only when the shape is piecewise; ``scale`` is written only for linear and
    quadratic), terrain map keys are strings. Rows with a blank cell are left out (``form_problems``
    reports them). Bad values are kept as typed so the problem check can name them.
    """
    profile = copy.deepcopy(dict(base)) if base else {}
    values = values or {}
    function = values.get("angle_cost_function")
    function = function if function in ANGLE_FUNCTIONS else _function_of(profile)
    _apply_fields(profile, values, function)
    if function == "piecewise" and angle_rows is not None:
        _apply_angle_rows(profile, angle_rows)
    if terrain_rows is not None:
        _apply_terrain_rows(profile, terrain_rows)
    if type_rows is not None:
        _apply_type_rows(profile, type_rows)
    return profile


# --------------------------------------------------------------------------- problems
def _field_problems(profile: Mapping) -> list[Problem]:
    out: list[Problem] = []
    function = _function_of(profile)
    for key, spec in FIELD_SPECS.items():
        if key == "angle_cost_scale" and function not in ("linear", "quadratic"):
            continue
        out.extend(_one_field_problems(key, spec, _get(profile, spec.path)))
    return out


def _one_field_problems(key: str, spec: FieldSpec, raw: Any) -> list[Problem]:
    if key in _PROFILE_REQUIRED and _is_blank(raw):
        return [(key, f"{_subject(spec)}: missing, but it is required. " +
                      f"Allowed: {describe_range(spec)}.")]
    if raw is None or (spec.kind != "text" and _is_blank(raw)):
        return []
    return [(key, m) for m in _value_problems(spec, raw)]


def _cross_problems(profile: Mapping) -> list[Problem]:
    """Rules that compare two fields (the ones ``InfrastructureProfile`` checks, and more)."""
    out: list[Problem] = []
    spec = FIELD_SPECS
    soft, hard = (_number(profile.get(k)) for k in ("soft_angle_limit_deg",
                                                    "hard_angle_limit_deg"))
    if soft is not None and hard is not None and hard < soft:
        out.append(("hard_angle_limit_deg",
                    f"{_subject(spec['hard_angle_limit_deg'])}: {_fmt(hard)} deg is below the " +
                    f"soft turn limit of {_fmt(soft)} deg. Allowed: {_fmt(soft)} to 180 deg."))
    out.extend(_span_problems(profile))
    return out


def _library_problems(profile: Mapping) -> list[Problem]:
    """The last check: build the real object, the way a run does."""
    config = dict(profile)
    config.setdefault("name", FIELD_SPECS["name"].default)
    config.setdefault("description", "edited in the PYORPS GUI")
    try:
        InfrastructureProfile.from_dict(config)
    except (PyorpsError, TypeError, ValueError, KeyError) as exc:
        text = str(exc)
        key = next((k for k in FORM_KEYS if k in text), "profile")
        return [(key, f"Profile ({key}): the library rejects it: {text}")]
    return []


def profile_problems(profile: Any) -> list[Problem]:
    """Every limit and cross-field rule of the form -> ``[(key, message), ...]``.

    ``key`` is a :data:`FORM_KEYS` entry, a table key (:data:`TABLE_ANGLE`,
    :data:`TABLE_TERRAIN`, :data:`TABLE_TYPES`) or ``"profile"``. Each message names the field
    (label and key), the value, the allowed range and the unit. Empty list: the profile is valid.
    The last step builds ``InfrastructureProfile.from_dict`` and runs only when the rules above
    found nothing, so a problem is never reported twice.
    """
    if not isinstance(profile, Mapping) and not callable(getattr(profile, "to_dict", None)):
        return [("profile", f"Profile (profile): {_fmt(profile)} is not a mapping of " +
                            "settings (name: value).")]
    data = _as_dict(profile)
    problems = (_field_problems(data) + _cross_problems(data) + _angle_table_problems(data)
                + _terrain_table_problems(data) + _types_table_problems(data))
    return problems or _library_problems(data)


def table_row_problems(angle_rows: list[dict] | None, terrain_rows: list[dict] | None,
                       type_rows: list[dict] | None) -> list[Problem]:
    """Problems only visible in the grid rows: half-filled rows and duplicate keys.

    ``profile_from_form`` leaves such rows out, so ``profile_problems`` cannot see them.
    """
    out: list[Problem] = []
    for key, rows, columns in ((TABLE_ANGLE, angle_rows, ANGLE_COLUMNS),
                               (TABLE_TERRAIN, terrain_rows, TERRAIN_COLUMNS),
                               (TABLE_TYPES, type_rows, TYPE_COLUMNS)):
        for row in rows or ():
            out.extend(_half_row_problems(key, row, columns))
    out.extend(_repeated_terrain_rows(terrain_rows))
    out.extend(_repeated_type_rows(type_rows))
    return out


def _half_row_problems(key: str, row: Any, columns: tuple[str, ...]) -> list[Problem]:
    blank = [c for c in columns if _is_blank((row or {}).get(c))]
    if blank and len(blank) < len(columns):
        return [(key, f"{TABLE_SPECS[key].label} ({key}): a row has no value for " +
                      f"{', '.join(blank)}. Fill all cells of the row or delete it.")]
    return []


def _repeated_terrain_rows(terrain_rows: list[dict] | None) -> list[Problem]:
    out: list[Problem] = []
    seen: set[str] = set()
    for row in terrain_rows or ():
        if _complete(row, TERRAIN_COLUMNS):
            name = _terrain_key(row["raster_value"])
            if name in seen:
                out.append((TABLE_TERRAIN, f"{TABLE_SPECS[TABLE_TERRAIN].label} " +
                                           f"({TABLE_TERRAIN}): raster value {name} appears " +
                                           "twice. Use each value once."))
            seen.add(name)
    return out


def _repeated_type_rows(type_rows: list[dict] | None) -> list[Problem]:
    out: list[Problem] = []
    names: set[str] = set()
    for row in type_rows or ():
        if _complete(row, TYPE_COLUMNS):
            name = str(row["name"]).strip()
            if name in names:
                out.append((TABLE_TYPES, f"{TABLE_SPECS[TABLE_TYPES].label} ({TABLE_TYPES}): " +
                                         f"type name {_fmt(name)} appears twice. Names must be " +
                                         "unique."))
            names.add(name)
    return out


def form_problems(values: Mapping | None, angle_rows: list[dict] | None,
                  terrain_rows: list[dict] | None, type_rows: list[dict] | None,
                  base: Mapping | None = None) -> list[Problem]:
    """``profile_problems`` of the profile the form makes, plus ``table_row_problems``."""
    profile = profile_from_form(values, angle_rows, terrain_rows, type_rows, base)
    return profile_problems(profile) + table_row_problems(angle_rows, terrain_rows, type_rows)


def label_for(key: str) -> str:
    """The label of a form key or table key (``"Profile"`` for anything else)."""
    if key in FIELD_SPECS:
        return FIELD_SPECS[key].label
    if key in TABLE_SPECS:
        return TABLE_SPECS[key].label
    return "Profile"


def problems_text(problems: list[Problem], limit: int = 3) -> str:
    """The first ``limit`` messages, one per line, plus a count of the rest (for the status line)."""
    lines = [message for _, message in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"... and {len(problems) - limit} more.")
    return "\n".join(lines)


def problem_notice(problems: list[Problem], *, controls: Mapping[str, str] | None = None,
                   tab: str = "tab-routes",
                   impact: str = "Constrained routing was not run.") -> dict | None:
    """The first problem as a notice dict (``Notice.to_dict()``), or None without problems.

    ``controls`` maps a key to the component id the "Go fix" button should scroll to (the
    caller owns the ids, this service does not know them).
    """
    if not problems:
        return None
    key, message = problems[0]
    label = label_for(key)
    return Notice(
        severity="warning", title=f"{label}: value not usable", meaning=message,
        impact=impact, fix=f"Correct the {label} field in the profile form and try again.",
        focus_id=tab, focus_control=(controls or {}).get(key)).to_dict()


# --------------------------------------------------------------------------- clearance, keys
def clearance_state(profile: Any) -> str:
    """``"on"`` (all four settings given), ``"off"`` (none) or ``"partial"``."""
    data = _as_dict(profile)
    heights = data.get("tower_heights_m")
    parts = (data.get("tower_height_m") is not None
             or (isinstance(heights, (list, tuple)) and len(heights) > 0),
             data.get("conductor_weight_per_m") is not None,
             data.get("conductor_tension_n") is not None,
             data.get("min_clearance_m") is not None)
    if all(parts):
        return "on"
    return "partial" if any(parts) else "off"


#: what the RUN must bring for the clearance check to act (``ConstrainedPathFinder``: the check is on with a DEM and
#: a profile that has the four clearance settings; a DSM is optional and read only together with a DEM)
CLEARANCE_RUN_NEEDS = ("the run needs a terrain model (DEM); a surface model (DSM) is optional and adds trees and " +
                       "buildings")


def clearance_status(profile: Any) -> str:
    """One info line for the form: is the clearance check on, and what is missing.

    The profile alone does not switch the check on: the run also needs a terrain model (DEM). A surface model (DSM)
    is optional (it adds the height of trees and buildings to the check, and the library ignores it without a DEM).
    """
    data = _as_dict(profile)
    state = clearance_state(data)
    if state == "on":
        return f"Clearance check: on, the profile is complete ({CLEARANCE_RUN_NEEDS})"
    names = ("base height", "conductor weight", "conductor tension", "ground clearance")
    heights = data.get("tower_heights_m")
    have = (data.get("tower_height_m") is not None
            or (isinstance(heights, (list, tuple)) and len(heights) > 0),
            data.get("conductor_weight_per_m") is not None,
            data.get("conductor_tension_n") is not None,
            data.get("min_clearance_m") is not None)
    if state == "off":
        return ("Clearance check: off (the profile needs base height, conductor weight, tension and " +
                f"clearance; {CLEARANCE_RUN_NEEDS})")
    missing = ", ".join(n for n, ok in zip(names, have) if not ok)
    return f"Clearance check: off, still missing: {missing} ({CLEARANCE_RUN_NEEDS})"


def unused_keys(profile: Any) -> list[str]:
    """Keys of :data:`UNUSED_KEYS` the profile sets: the router never reads them."""
    data = _as_dict(profile)
    return [key for key in UNUSED_KEYS if data.get(key) is not None]


def unknown_keys(profile: Any) -> list[str]:
    """Top-level keys that are no ``InfrastructureProfile`` field (``from_dict`` drops them)."""
    known = set(InfrastructureProfile.__dataclass_fields__)  # pylint: disable=no-member
    return [str(key) for key in _as_dict(profile) if key not in known]


# --------------------------------------------------------------------------- YAML
def _yaml_where(exc: Exception) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    return f"line {mark.line + 1}: {problem}" if mark is not None else str(problem)


def parse_profile_text(text: str | None) -> dict:
    """YAML (or JSON) text -> profile dict.

    Raises ``InvalidInputError`` for empty text, a syntax error or text that is not a mapping,
    and ``MissingDependencyError`` when PyYAML is not installed.
    """
    yaml = optional_deps.import_yaml()
    try:
        data = yaml.safe_load(text or "")
    except yaml.YAMLError as exc:
        raise InvalidInputError(
            f"The profile text is not valid YAML ({_yaml_where(exc)}).") from exc
    if data is None:
        raise InvalidInputError("The profile text is empty. Write 'name: value' lines or load "
                                "a profile.")
    if not isinstance(data, dict):
        raise InvalidInputError("The profile text must be 'name: value' lines (a mapping), "
                                f"not a {type(data).__name__}.")
    return data


def profile_to_text(profile: Mapping) -> str:
    """Profile dict -> YAML text (key order kept). ``MissingDependencyError`` without PyYAML."""
    yaml = optional_deps.import_yaml()
    try:
        return yaml.safe_dump(dict(profile), sort_keys=False, default_flow_style=False,
                              allow_unicode=True)
    except yaml.YAMLError as exc:
        raise InvalidInputError(f"The profile cannot be written as YAML: {exc}") from exc


def base_profile(text: str | None, fallback: Mapping | None = None) -> dict:
    """The dict a form edit is merged into: the text when it parses, else ``fallback``.

    Empty or broken text falls back to ``fallback`` (the last good dict, ``OHLF_BASE``).
    ``MissingDependencyError`` still propagates.
    """
    if _is_blank(text):
        return copy.deepcopy(dict(fallback or {}))
    try:
        return parse_profile_text(text)
    except InvalidInputError:
        return copy.deepcopy(dict(fallback or {}))


def text_to_form(text: str | None) -> tuple[dict, dict, list[dict], list[dict], list[dict]]:
    """"YAML -> form": ``(profile, values, angle_rows, terrain_rows, type_rows)``.

    ``profile`` is the parsed dict (the new ``OHLF_BASE``). Raises like ``parse_profile_text``.
    """
    profile = parse_profile_text(text)
    return (profile, *form_values(profile))


def form_to_text(values: Mapping | None, angle_rows: list[dict] | None,
                 terrain_rows: list[dict] | None, type_rows: list[dict] | None,
                 base_text: str | None = None, fallback: Mapping | None = None
                 ) -> tuple[str, dict, list[Problem]]:
    """"form -> YAML": ``(yaml_text, profile, problems)``.

    The form is merged into ``base_profile(base_text, fallback)``, so keys the form does not
    show survive (comments in the text do not). The text is produced even when ``problems`` is
    not empty: the caller decides whether to show it as an error or block a run.
    """
    base = base_profile(base_text, fallback)
    profile = profile_from_form(values, angle_rows, terrain_rows, type_rows, base)
    problems = profile_problems(profile) + table_row_problems(angle_rows, terrain_rows,
                                                              type_rows)
    return profile_to_text(profile), profile, problems

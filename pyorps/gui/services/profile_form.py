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
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

from pyorps.core import optional_deps
from pyorps.core.exceptions import InvalidInputError, PyorpsError
from pyorps.core.infrastructure_profile import InfrastructureProfile

from .errors import Notice

# --------------------------------------------------------------------------- specs
ANGLE_FUNCTIONS = ("linear", "quadratic", "piecewise")
INTERPOLATIONS = ("linear", "nearest")
AREA_MODES = ("uniform", "exact")

#: keys of the three tables (also the ``key`` of a table problem)
TABLE_ANGLE = "angle_cost_table"
TABLE_TERRAIN = "terrain_cost_table"
TABLE_TYPES = "angle_types_table"

#: profile fields the router never reads (``tower_cost_function``, ``gradient_cost_function``,
#: ``cost_assumptions_path``) or that only inform (``recommended_buffer_m``)
UNUSED_KEYS = ("tower_cost_function", "gradient_cost_function",
               "cost_assumptions_path", "recommended_buffer_m")

#: profile keys that must be present (the others have a library default or are filled by the run)
_PROFILE_REQUIRED = ("soft_angle_limit_deg", "hard_angle_limit_deg", "angle_cost_function")


@dataclass(frozen=True)
class FieldSpec:
    """One form field (or one table column): label, unit, limits, default, path."""

    key: str
    label: str
    unit: str = ""
    kind: str = "number"        # "text" | "number" | "select" | "list" (comma list of numbers)
    minimum: float | None = None
    maximum: float | None = None
    allow_equal: bool = True    # False: the value must be strictly above ``minimum``
    whole: bool = False
    default: Any = None         # shown when the profile does not have the key
    options: tuple[str, ...] = ()
    help: str = ""
    path: tuple[str, ...] = ()  # where the value sits in the profile dict
    required: bool = False      # the form always writes the key
    max_length: int | None = None


def _spec(key: str, label: str, **kw: Any) -> FieldSpec:
    kw.setdefault("path", (key,))
    return FieldSpec(key=key, label=label, **kw)


FIELD_SPECS: dict[str, FieldSpec] = {s.key: s for s in (
    _spec("name", "Profile name", unit="", kind="text", default="gui_profile",
          required=True, max_length=80,
          help="A short name for this profile. It is saved in the YAML and in each route."),
    _spec("description", "Description", unit="", kind="text", default="", max_length=300,
          help="Free text for yourself."),
    _spec("soft_angle_limit_deg", "Soft turn limit", unit="deg", minimum=0, maximum=180,
          default=8, required=True,
          help="Turns up to this angle at a tower cost nothing extra (piecewise) or are cheap."),
    _spec("hard_angle_limit_deg", "Hard turn limit (forbidden above)", unit="deg", minimum=0,
          maximum=180, default=40, required=True,
          help="The route may never turn by more than this at a tower. At least the soft limit."),
    _spec("angle_cost_function", "Turn cost shape", kind="select", options=ANGLE_FUNCTIONS,
          default="piecewise", required=True,
          help="How the cost of a turn grows with the angle: a table (piecewise), "
               "factor times angle (linear) or factor times angle squared (quadratic)."),
    _spec("angle_cost_scale", "Turn cost factor", unit="cost per deg or deg2", minimum=0,
          maximum=1e9, default=1.0, path=("angle_cost_params", "scale"),
          help="Used for linear and quadratic turn cost only. The turn table is used for "
               "piecewise."),
    _spec("min_span_m", "Shortest span", unit="m", minimum=0, allow_equal=False,
          help="Shortest allowed distance between two towers. Set both spans or neither."),
    _spec("max_span_m", "Longest span", unit="m", minimum=0, allow_equal=False, maximum=5000,
          help="Longest allowed distance between two towers. Set both spans or neither."),
    _spec("span_bin_size_m", "Span step", unit="m", minimum=0, allow_equal=False,
          help="Resolution of the span search. Needed with spans; keep it at or below the "
               "raster cell size or the search gets slow."),
    _spec("terrain_interpolation", "Between table values", kind="select",
          options=INTERPOLATIONS, default="linear",
          path=("tower_cost_params", "terrain_interpolation"),
          help="How foundation cost is read between two raster values of the terrain table."),
    _spec("max_gradient_percent", "Steepest allowed slope", unit="%", minimum=0,
          allow_equal=False, maximum=1000,
          help="Cells steeper than this are blocked. Blank: the router uses 100 %. "
               "Needs a DEM."),
    _spec("gradient_scale", "Slope penalty factor", unit="", minimum=0, maximum=100,
          default=2.0, path=("gradient_cost_params", "scale"),
          help="How strongly steeper ground is penalised. The router reads only this value; "
               "blank means 2."),
    _spec("tower_height_m", "Base attachment height", unit="m", minimum=0, allow_equal=False,
          maximum=300, help="Lowest conductor attachment height. Part of the clearance check."),
    _spec("tower_heights_m", "Height options (comma list)", unit="m", kind="list", minimum=0,
          allow_equal=False, maximum=300,
          help="Several attachment heights the search may pick, e.g. 25, 34, 42."),
    _spec("tower_height_cost_per_increment", "Extra cost per 3 m step", unit="EUR", minimum=0,
          default=0, help="Added to a tower for each 3 m above the lowest height."),
    _spec("terminal_tower_height_m", "Height at the end points", unit="m", minimum=0,
          allow_equal=False, maximum=300, help="Attachment height of the towers at both ends."),
    _spec("terminal_tower_cost", "Cost of each end tower", unit="EUR", minimum=0, default=0,
          help="Price of one end (dead-end) tower."),
    _spec("conductor_weight_per_m", "Conductor weight", unit="N/m", minimum=0,
          allow_equal=False, maximum=1000,
          help="Weight of the conductor per metre. Part of the clearance check."),
    _spec("conductor_tension_n", "Conductor tension", unit="N", minimum=0, allow_equal=False,
          maximum=1e7, help="Horizontal conductor tension. Part of the clearance check."),
    _spec("min_clearance_m", "Ground clearance", unit="m", minimum=0, maximum=100,
          help="Minimum distance between conductor and ground or obstacles. "
               "Part of the clearance check."),
    _spec("tower_ground_area_m2", "Foundation area", unit="m2", minimum=1, maximum=10000,
          default=1, help="Footprint of one tower foundation."),
    _spec("tower_area_cost_mode", "Foundation pricing", kind="select", options=AREA_MODES,
          default="uniform",
          help="uniform: every cell under the foundation costs like the centre cell. "
               "exact: the real footprint is summed."),
    _spec("recommended_buffer_m", "Suggested corridor width (info only)", unit="m", minimum=0,
          help="Information only: the router does not read it."),
)}

#: the 23 form keys in form order (equal to ``ids.OHLF_KEYS``, a test checks it)
FORM_KEYS: tuple[str, ...] = tuple(FIELD_SPECS)

#: column keys of the three tables (the keys of each row dict)
ANGLE_COLUMNS = ("angle_deg", "cost")
TERRAIN_COLUMNS = ("raster_value", "eur_per_m2")
TYPE_COLUMNS = ("name", "max_angle_deg", "base_cost")


@dataclass(frozen=True)
class TableSpec:
    """One table of the form: label, columns (as ``FieldSpec``) and its place in the profile."""

    key: str
    label: str
    columns: tuple[FieldSpec, ...]
    path: tuple[str, ...]
    help: str = ""
    min_rows: int = 0


TABLE_SPECS: dict[str, TableSpec] = {t.key: t for t in (
    TableSpec(
        TABLE_ANGLE, "Turn cost table",
        (_spec("angle_deg", "Turn angle", unit="deg", minimum=0, maximum=180),
         _spec("cost", "Cost", unit="cost units", minimum=0)),
        ("angle_cost_params", "breakpoints"),
        help="Cost of a turn at each angle. Angles must increase. Used for piecewise only.",
        min_rows=2),
    TableSpec(
        TABLE_TERRAIN, "Terrain cost table",
        (_spec("raster_value", "Raster value", unit="", minimum=0, maximum=65534, whole=True),
         _spec("eur_per_m2", "Foundation cost", unit="EUR per m2", minimum=0)),
        ("tower_cost_params", "terrain_cost_map"),
        help="Foundation cost per square metre for each raster value."),
    TableSpec(
        TABLE_TYPES, "Tower types",
        (_spec("name", "Type name", kind="text"),
         _spec("max_angle_deg", "Largest turn", unit="deg", minimum=0, maximum=180),
         _spec("base_cost", "Base cost", unit="EUR", minimum=0)),
        ("tower_cost_params", "angle_types"),
        help="Tower type chosen by the turn angle at the tower: the first type whose "
             "largest turn is not exceeded."),
)}

Problem = tuple[str, str]


# --------------------------------------------------------------------------- small helpers
def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _coerce(value: Any) -> Any:
    """A number from a number or a numeric string; everything else comes back unchanged."""
    if isinstance(value, bool) or not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def _number(value: Any) -> float | None:
    """A finite float, or None for anything that is not a usable number."""
    value = _coerce(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value: Any) -> str:
    """A value as the user should read it in a message or a comma list."""
    if value is None:
        return "blank"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return repr(str(value))


def _unit(spec: FieldSpec) -> str:
    return f" {spec.unit}" if spec.unit else ""


def describe_range(spec: FieldSpec) -> str:
    """The allowed range of a field in words, e.g. ``>= 0 and <= 180 deg``."""
    if spec.kind == "select":
        return ", ".join(spec.options)
    if spec.kind == "text":
        low = 1 if spec.required else 0
        return f"{low} to {spec.max_length} characters"
    parts = []
    if spec.minimum is not None:
        parts.append(f"{'>=' if spec.allow_equal else '>'} {_fmt(spec.minimum)}")
    if spec.maximum is not None:
        parts.append(f"<= {_fmt(spec.maximum)}")
    text = " and ".join(parts) if parts else "any number"
    if spec.whole:
        text += ", whole numbers"
    if spec.kind == "list":
        text += " for each entry"
    return text + _unit(spec)


def _subject(spec: FieldSpec, table: TableSpec | None = None) -> str:
    if table is None:
        return f"{spec.label} ({spec.key})"
    return f"{table.label} ({table.key}), {spec.label}"


def _number_problem(spec: FieldSpec, raw: Any, subject: str) -> str | None:
    """Why a (non-blank) number is not allowed, or None."""
    allowed = f"Allowed: {describe_range(spec)}."
    value = _coerce(raw)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{subject}: {_fmt(raw)} is not a number. {allowed}"
    if not math.isfinite(value):
        return f"{subject}: {_fmt(value)} is not a finite number. {allowed}"
    shown = f"{_fmt(value)}{_unit(spec)}"
    if spec.whole and value != int(value):
        return f"{subject}: {shown} is not a whole number. {allowed}"
    if spec.minimum is not None and (
            value < spec.minimum or (value == spec.minimum and not spec.allow_equal)):
        return f"{subject}: {shown} is too low. {allowed}"
    if spec.maximum is not None and value > spec.maximum:
        return f"{subject}: {shown} is too high. {allowed}"
    return None


def _text_problem(spec: FieldSpec, raw: Any, subject: str) -> str | None:
    allowed = f"Allowed: {describe_range(spec)}."
    if not isinstance(raw, str):
        return f"{subject}: {_fmt(raw)} is not text. {allowed}"
    length = len(raw.strip())
    if spec.required and length == 0:
        return f"{subject}: blank, but a name is required. {allowed}"
    if spec.max_length is not None and length > spec.max_length:
        return f"{subject}: {length} characters is too long. {allowed}"
    return None


def _select_problem(spec: FieldSpec, raw: Any, subject: str) -> str | None:
    if isinstance(raw, str) and raw in spec.options:
        return None
    return f"{subject}: {_fmt(raw)} is not an option. Allowed: {describe_range(spec)}."


def _list_problems(spec: FieldSpec, raw: Any, subject: str) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return [f"{subject}: {_fmt(raw)} is not a list of numbers. "
                f"Allowed: {describe_range(spec)}."]
    out = []
    for item in raw:
        message = (f"{subject}: a blank entry. Allowed: {describe_range(spec)}."
                   if _is_blank(item) else _number_problem(spec, item, subject))
        if message:
            out.append(message)
    return out


def _value_problems(spec: FieldSpec, raw: Any, subject: str | None = None) -> list[str]:
    """Messages for one present (non-blank) value of any kind."""
    subject = subject or _subject(spec)
    if spec.kind == "number":
        found = [_number_problem(spec, raw, subject)]
    elif spec.kind == "text":
        found = [_text_problem(spec, raw, subject)]
    elif spec.kind == "select":
        found = [_select_problem(spec, raw, subject)]
    else:
        found = _list_problems(spec, raw, subject)
    return [m for m in found if m]


def _get(profile: Any, path: tuple[str, ...]) -> Any:
    node = profile
    for part in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _has(profile: Any, path: tuple[str, ...]) -> bool:
    node = profile
    for part in path[:-1]:
        if not isinstance(node, Mapping):
            return False
        node = node.get(part)
    return isinstance(node, Mapping) and path[-1] in node


def _put(profile: dict, path: tuple[str, ...], value: Any) -> None:
    node = profile
    for part in path[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[path[-1]] = value


def _remove(profile: dict, path: tuple[str, ...]) -> None:
    node = profile
    for part in path[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            return
    node.pop(path[-1], None)
    if len(path) == 2 and path[0] == "gradient_cost_params" and not node:
        profile.pop(path[0], None)       # the library default is None, not {}


def _as_dict(profile: Any) -> dict:
    if isinstance(profile, Mapping):
        return dict(profile)
    to_dict = getattr(profile, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {}


def _function_of(profile: Mapping) -> str:
    value = profile.get("angle_cost_function")
    return value if isinstance(value, str) else FIELD_SPECS["angle_cost_function"].default


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
    if present and existing == value and type(existing) is not bool:
        return                                   # unchanged: keep the stored object
    if (not present and not spec.required and spec.default is not None
            and value == spec.default):
        return                                   # the default is not written out
    _put(profile, spec.path, value)


def _complete(row: Any, columns: tuple[str, ...]) -> bool:
    return isinstance(row, Mapping) and all(not _is_blank(row.get(c)) for c in columns)


def _apply_angle_rows(profile: dict, rows: list[dict]) -> None:
    done = sorted(((_coerce(r["angle_deg"]), _coerce(r["cost"]))
                   for r in rows or () if _complete(r, ANGLE_COLUMNS)),
                  key=lambda pair: _sort_key_number(pair[0]))
    path = TABLE_SPECS[TABLE_ANGLE].path
    if not done:
        _remove(profile, path)
        _remove(profile, path[:-1] + ("costs",))
        return
    angles, costs = [p[0] for p in done], [p[1] for p in done]
    for sub, new in (("breakpoints", angles), ("costs", costs)):
        if _get(profile, path[:-1] + (sub,)) != new:
            _put(profile, path[:-1] + (sub,), new)


def _terrain_key(value: Any) -> str:
    number = _coerce(value)
    if isinstance(number, float) and math.isfinite(number) and number == int(number):
        number = int(number)
    return str(number).strip()


def _apply_terrain_rows(profile: dict, rows: list[dict]) -> None:
    done = sorted(((r["raster_value"], _coerce(r["eur_per_m2"]))
                   for r in rows or () if _complete(r, TERRAIN_COLUMNS)),
                  key=lambda pair: _sort_key_number(pair[0]))
    new = {_terrain_key(k): v for k, v in done}
    path = TABLE_SPECS[TABLE_TERRAIN].path
    existing = _get(profile, path)
    if isinstance(existing, Mapping) and {str(k): v for k, v in existing.items()} == new:
        return                                   # unchanged: keep int keys of the base
    if new:
        _put(profile, path, new)
    else:
        _remove(profile, path)


def _apply_type_rows(profile: dict, rows: list[dict]) -> None:
    path = TABLE_SPECS[TABLE_TYPES].path
    existing = _get(profile, path)
    existing = existing if isinstance(existing, Mapping) else {}
    new: dict = {}
    for row in rows or ():
        if not _complete(row, TYPE_COLUMNS):
            continue
        name = str(row["name"]).strip()
        old = existing.get(name)
        entry = copy.deepcopy(dict(old)) if isinstance(old, Mapping) else {}
        for col in ("max_angle_deg", "base_cost"):
            value = _coerce(row[col])
            if col not in entry or entry[col] != value:
                entry[col] = value
        new[name] = entry
    if existing == new:
        return
    if new:
        _put(profile, path, new)
    else:
        _remove(profile, path)


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
    for key, spec in FIELD_SPECS.items():
        if key not in values or (key == "angle_cost_scale" and function == "piecewise"):
            continue
        _apply_field(profile, spec, values[key])
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
        raw = _get(profile, spec.path)
        if key in _PROFILE_REQUIRED and _is_blank(raw):
            out.append((key, f"{_subject(spec)}: missing, but it is required. "
                             f"Allowed: {describe_range(spec)}."))
        elif raw is None or (spec.kind != "text" and _is_blank(raw)):
            continue
        else:
            out.extend((key, m) for m in _value_problems(spec, raw))
    return out


def _cross_problems(profile: Mapping) -> list[Problem]:
    """Rules that compare two fields (the ones ``InfrastructureProfile`` checks, and more)."""
    out: list[Problem] = []
    spec = FIELD_SPECS
    soft, hard = (_number(profile.get(k)) for k in ("soft_angle_limit_deg",
                                                    "hard_angle_limit_deg"))
    if soft is not None and hard is not None and hard < soft:
        out.append(("hard_angle_limit_deg",
                    f"{_subject(spec['hard_angle_limit_deg'])}: {_fmt(hard)} deg is below the "
                    f"soft turn limit of {_fmt(soft)} deg. Allowed: {_fmt(soft)} to 180 deg."))
    out.extend(_span_problems(profile))
    return out


def _span_problems(profile: Mapping) -> list[Problem]:
    out: list[Problem] = []
    spec = FIELD_SPECS
    raw = {k: profile.get(k) for k in ("min_span_m", "max_span_m", "span_bin_size_m")}
    lo, hi, step = (_number(raw[k]) for k in raw)
    has_lo, has_hi = not _is_blank(raw["min_span_m"]), not _is_blank(raw["max_span_m"])
    if has_lo != has_hi:
        missing, other = ("max_span_m", "min_span_m") if has_lo else ("min_span_m", "max_span_m")
        out.append((missing, f"{_subject(spec[missing])}: blank, but {other} is set. "
                             "Set both spans or neither."))
    if lo is not None and hi is not None and lo > hi:
        out.append(("min_span_m",
                    f"{_subject(spec['min_span_m'])}: {_fmt(lo)} m is above the longest span of "
                    f"{_fmt(hi)} m. Allowed: > 0 and <= {_fmt(hi)} m."))
    if has_lo and has_hi:
        if _is_blank(raw["span_bin_size_m"]):
            out.append(("span_bin_size_m",
                        f"{_subject(spec['span_bin_size_m'])}: blank, but it is required when "
                        f"spans are set. Allowed: {describe_range(spec['span_bin_size_m'])}."))
        elif step is not None and hi is not None and step > hi:
            out.append(("span_bin_size_m",
                        f"{_subject(spec['span_bin_size_m'])}: {_fmt(step)} m is above the "
                        f"longest span of {_fmt(hi)} m. Allowed: > 0 and <= {_fmt(hi)} m."))
    return out


def _angle_table_problems(profile: Mapping) -> list[Problem]:
    if _function_of(profile) != "piecewise":
        return []
    table = TABLE_SPECS[TABLE_ANGLE]
    ang_col, cost_col = table.columns
    params = profile.get("angle_cost_params")
    params = params if isinstance(params, Mapping) else {}
    angles, costs = params.get("breakpoints"), params.get("costs")
    head = f"{table.label} ({table.key})"
    if not isinstance(angles, (list, tuple)) or not isinstance(costs, (list, tuple)):
        return [(table.key, f"{head}: missing. The piecewise shape needs a list of angles and "
                            f"a list of costs, at least {table.min_rows} rows.")]
    if len(angles) != len(costs):
        return [(table.key, f"{head}: {len(angles)} angles but {len(costs)} costs. "
                            "Every angle needs one cost.")]
    out = [(table.key, m) for a, c in zip(angles, costs)
           for spec, raw in ((ang_col, a), (cost_col, c))
           for m in (_value_problems(spec, raw, _subject(spec, table))
                     if not _is_blank(raw) else
                     [f"{_subject(spec, table)}: a blank value. "
                      f"Allowed: {describe_range(spec)}."])]
    if len(angles) < table.min_rows:
        out.append((table.key, f"{head}: {len(angles)} rows is too few. Allowed: at least "
                               f"{table.min_rows} rows."))
    numbers = [n for n in (_number(a) for a in angles) if n is not None]
    for before, after in zip(numbers, numbers[1:]):
        if after <= before:
            out.append((table.key, f"{head}: angle {_fmt(after)} deg does not come after "
                                   f"{_fmt(before)} deg. Angles must strictly increase."))
            break
    return out


def _terrain_table_problems(profile: Mapping) -> list[Problem]:
    table = TABLE_SPECS[TABLE_TERRAIN]
    raw = _get(profile, table.path)
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return [(table.key, f"{table.label} ({table.key}): {_fmt(raw)} is not a table of raster "
                            "value to cost.")]
    val_col, cost_col = table.columns
    out: list[Problem] = []
    seen: set[int] = set()
    for key, cost in raw.items():
        out.extend((table.key, m) for m in _value_problems(val_col, key, _subject(val_col, table)))
        out.extend((table.key, m) for m in _value_problems(cost_col, cost, _subject(cost_col, table))
                   if not _is_blank(cost))
        if _is_blank(cost):
            out.append((table.key, f"{_subject(cost_col, table)}: a blank value for raster "
                                   f"value {_fmt(key)}. Allowed: {describe_range(cost_col)}."))
        number = _number(key)
        if number is not None and number == int(number):
            if int(number) in seen:
                out.append((table.key, f"{table.label} ({table.key}): raster value "
                                       f"{int(number)} appears twice. Use each value once."))
            seen.add(int(number))
    return out


def _types_table_problems(profile: Mapping) -> list[Problem]:
    table = TABLE_SPECS[TABLE_TYPES]
    raw = _get(profile, table.path)
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return [(table.key, f"{table.label} ({table.key}): {_fmt(raw)} is not a table of "
                            "type name to settings.")]
    out: list[Problem] = []
    for name, entry in raw.items():
        if not str(name).strip():
            out.append((table.key, f"{table.label} ({table.key}): a type without a name. "
                                   "Allowed: 1 to 80 characters."))
        if not isinstance(entry, Mapping):
            out.append((table.key, f"{table.label} ({table.key}), type {_fmt(name)}: "
                                   "needs max_angle_deg and base_cost."))
            continue
        for spec in table.columns[1:]:
            subject = f"{_subject(spec, table)} of type {_fmt(name)}"
            value = entry.get(spec.key)
            if _is_blank(value):
                out.append((table.key, f"{subject}: missing. Allowed: {describe_range(spec)}."))
            else:
                out.extend((table.key, m) for m in _value_problems(spec, value, subject))
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
        return [("profile", f"Profile (profile): {_fmt(profile)} is not a mapping of "
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
        table = TABLE_SPECS[key]
        for row in rows or ():
            blank = [c for c in columns if _is_blank((row or {}).get(c))]
            if blank and len(blank) < len(columns):
                out.append((key, f"{table.label} ({key}): a row has no value for "
                                 f"{', '.join(blank)}. Fill all cells of the row or delete it."))
    seen: set[str] = set()
    for row in terrain_rows or ():
        if _complete(row, TERRAIN_COLUMNS):
            name = _terrain_key(row["raster_value"])
            if name in seen:
                out.append((TABLE_TERRAIN, f"{TABLE_SPECS[TABLE_TERRAIN].label} "
                                           f"({TABLE_TERRAIN}): raster value {name} appears "
                                           "twice. Use each value once."))
            seen.add(name)
    names: set[str] = set()
    for row in type_rows or ():
        if _complete(row, TYPE_COLUMNS):
            name = str(row["name"]).strip()
            if name in names:
                out.append((TABLE_TYPES, f"{TABLE_SPECS[TABLE_TYPES].label} ({TABLE_TYPES}): "
                                         f"type name {_fmt(name)} appears twice. Names must be "
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


def clearance_status(profile: Any) -> str:
    """One info line for the form: is the clearance check on, and what is missing."""
    data = _as_dict(profile)
    state = clearance_state(data)
    if state == "on":
        return "Clearance check: on (the run also needs a DSM)"
    names = ("base height", "conductor weight", "conductor tension", "ground clearance")
    heights = data.get("tower_heights_m")
    have = (data.get("tower_height_m") is not None
            or (isinstance(heights, (list, tuple)) and len(heights) > 0),
            data.get("conductor_weight_per_m") is not None,
            data.get("conductor_tension_n") is not None,
            data.get("min_clearance_m") is not None)
    if state == "off":
        return ("Clearance check: off (needs base height, conductor weight, tension and "
                "clearance, plus a DSM)")
    missing = ", ".join(n for n, ok in zip(names, have) if not ok)
    return f"Clearance check: off, still missing: {missing} (and a DSM)"


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

"""
PYORPS GUI: the specs and the problem finder behind the profile form (split from ``profile_form``).

The lower half of the profile service: the form specs (``FIELD_SPECS``, ``TABLE_SPECS``), the small
value helpers and the table checks that return ``(key, message)`` pairs (turn cost, terrain cost, tower
types). It imports nothing from ``profile_form``: that module re-exports every
public name here, so callers keep importing from ``pyorps.gui.services.profile_form``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pyorps.core.infrastructure_profile import TERMINAL_SPAN_RULES

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
          help="How the cost of a turn grows with the angle: a table (piecewise), " +
               "factor times angle (linear) or factor times angle squared (quadratic)."),
    _spec("angle_cost_scale", "Turn cost factor", unit="cost per deg or deg2", minimum=0,
          maximum=1e9, default=1.0, path=("angle_cost_params", "scale"),
          help="Used for linear and quadratic turn cost only. The turn table is used for " +
               "piecewise."),
    _spec("min_span_m", "Shortest span", unit="m", minimum=0, allow_equal=False,
          help="Shortest allowed distance between two towers. Set both spans or neither."),
    _spec("max_span_m", "Longest span", unit="m", minimum=0, allow_equal=False, maximum=5000,
          help="Longest allowed distance between two towers. Set both spans or neither."),
    _spec("span_bin_size_m", "Span step", unit="m", minimum=0, allow_equal=False,
          help="Resolution of the span search. Needed with spans; keep it at or below the " +
               "raster cell size or the search gets slow."),
    _spec("terrain_interpolation", "Between table values", kind="select",
          options=INTERPOLATIONS, default="linear",
          path=("tower_cost_params", "terrain_interpolation"),
          help="How foundation cost is read between two raster values of the terrain table."),
    _spec("max_gradient_percent", "Steepest allowed slope", unit="%", minimum=0,
          allow_equal=False, maximum=1000,
          help="Cells steeper than this are blocked. Blank: the router uses 100 %. " +
               "Needs a DEM."),
    _spec("gradient_scale", "Slope penalty factor", unit="", minimum=0, maximum=100,
          default=2.0, path=("gradient_cost_params", "scale"),
          help="How strongly steeper ground is penalised. The router reads only this value; " +
               "blank means 2."),
    _spec("tower_height_m", "Base attachment height", unit="m", minimum=0, allow_equal=False,
          maximum=300, help="Lowest conductor attachment height. Part of the clearance check."),
    _spec("tower_heights_m", "Height options (comma list)", unit="m", kind="list", minimum=0,
          allow_equal=False, maximum=300,
          help="Several attachment heights the search may pick, e.g. 25, 34, 42."),
    _spec("tower_height_cost_per_increment", "Extra cost per 3 m step", unit="EUR", minimum=0,
          default=0, help="Added to a tower for each 3 m above the lowest height."),
    _spec("terminal_tower_height_m", "Height at the end points", unit="m", minimum=0,
          allow_equal=False, maximum=300,
          help="Reported only, the search does not use it: the attachment height of the towers at both ends in " +
               "the route report. The search uses the connection heights of the Line ends block."),
    _spec("terminal_tower_cost", "Cost of each end tower", unit="EUR", minimum=0, default=0,
          help="Price of one end (dead-end) tower."),
    _spec("conductor_weight_per_m", "Conductor weight", unit="N/m", minimum=0,
          allow_equal=False, maximum=1000,
          help="Weight of the conductor per metre. Part of the clearance check."),
    _spec("conductor_tension_n", "Conductor tension", unit="N", minimum=0, allow_equal=False,
          maximum=1e7, help="Horizontal conductor tension. Part of the clearance check."),
    _spec("min_clearance_m", "Ground clearance", unit="m", minimum=0, maximum=100,
          help="Minimum distance between conductor and ground or obstacles. " +
               "Part of the clearance check."),
    _spec("tower_ground_area_m2", "Foundation area", unit="m2", minimum=1, maximum=10000,
          default=1, help="Footprint of one tower foundation."),
    _spec("tower_area_cost_mode", "Foundation pricing", kind="select", options=AREA_MODES,
          default="uniform",
          help="uniform: every cell under the foundation costs like the centre cell. " +
               "exact: the real footprint is summed."),
    _spec("recommended_buffer_m", "Suggested corridor width (info only)", unit="m", minimum=0,
          help="Information only: the router does not read it."),
    # round 17 (P5): the options are the library's four values; the page words them from
    # ``constrained_ends.RULE_LABELS`` (that module imports this one, so the labels cannot be read here)
    _spec("terminal_span_rule", "End-span rule", kind="select", options=TERMINAL_SPAN_RULES, default="remove",
          help="What happens to an end tower, the first or the last tower of the line next to its connection point, " +
               "when the search placed it closer to the line end than the shortest span: it can be removed " +
               "afterwards (the end span then gets longer), removed only if the line stays valid without it, or kept " +
               "(the end span stays shorter than the shortest span). 'Never place one' makes the search itself " +
               "keep every end span at least as long as the shortest span; it needs the exact search (more memory " +
               "and time) and a CPU backend. Needs both span limits."),
)}

#: the 24 form keys in form order (equal to ``ids.OHLF_KEYS``, a test checks it)
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
        help="Tower type chosen by the turn angle at the tower: the first type whose " +
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


def _bounds_text(spec: FieldSpec) -> str:
    parts = []
    if spec.minimum is not None:
        parts.append(f"{'>=' if spec.allow_equal else '>'} {_fmt(spec.minimum)}")
    if spec.maximum is not None:
        parts.append(f"<= {_fmt(spec.maximum)}")
    return " and ".join(parts) if parts else "any number"


def describe_range(spec: FieldSpec) -> str:
    """The allowed range of a field in words, e.g. ``>= 0 and <= 180 deg``."""
    if spec.kind == "select":
        return ", ".join(spec.options)
    if spec.kind == "text":
        low = 1 if spec.required else 0
        return f"{low} to {spec.max_length} characters"
    text = _bounds_text(spec)
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
    return _range_problem(spec, value, f"{subject}: {shown}", allowed)


def _range_problem(spec: FieldSpec, value: float, head: str, allowed: str) -> str | None:
    if spec.minimum is not None and (
            value < spec.minimum or (value == spec.minimum and not spec.allow_equal)):
        return f"{head} is too low. {allowed}"
    if spec.maximum is not None and value > spec.maximum:
        return f"{head} is too high. {allowed}"
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
        return [f"{subject}: {_fmt(raw)} is not a list of numbers. " +
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


def _complete(row: Any, columns: tuple[str, ...]) -> bool:
    return isinstance(row, Mapping) and all(not _is_blank(row.get(c)) for c in columns)


def _terrain_key(value: Any) -> str:
    number = _coerce(value)
    if isinstance(number, float) and math.isfinite(number) and number == int(number):
        number = int(number)
    return str(number).strip()


def _one_span_missing(has_lo: bool, has_hi: bool) -> list[Problem]:
    if has_lo == has_hi:
        return []
    spec = FIELD_SPECS
    missing, other = ("max_span_m", "min_span_m") if has_lo else ("min_span_m", "max_span_m")
    return [(missing, f"{_subject(spec[missing])}: blank, but {other} is set. " +
                      "Set both spans or neither.")]


def _span_order_problem(lo: float | None, hi: float | None) -> list[Problem]:
    if lo is None or hi is None or lo <= hi:
        return []
    return [("min_span_m",
             f"{_subject(FIELD_SPECS['min_span_m'])}: {_fmt(lo)} m is above the longest span of " +
             f"{_fmt(hi)} m. Allowed: > 0 and <= {_fmt(hi)} m.")]


def _bin_size_problems(bin_raw: Any, step: float | None, hi: float | None) -> list[Problem]:
    spec = FIELD_SPECS
    if _is_blank(bin_raw):
        return [("span_bin_size_m",
                 f"{_subject(spec['span_bin_size_m'])}: blank, but it is required when " +
                 f"spans are set. Allowed: {describe_range(spec['span_bin_size_m'])}.")]
    if step is not None and hi is not None and step > hi:
        return [("span_bin_size_m",
                 f"{_subject(spec['span_bin_size_m'])}: {_fmt(step)} m is above the " +
                 f"longest span of {_fmt(hi)} m. Allowed: > 0 and <= {_fmt(hi)} m.")]
    return []


def _span_problems(profile: Mapping) -> list[Problem]:
    raw = {k: profile.get(k) for k in ("min_span_m", "max_span_m", "span_bin_size_m")}
    lo, hi, step = (_number(raw[k]) for k in raw)
    has_lo, has_hi = not _is_blank(raw["min_span_m"]), not _is_blank(raw["max_span_m"])
    out = _one_span_missing(has_lo, has_hi)
    out.extend(_span_order_problem(lo, hi))
    if has_lo and has_hi:
        out.extend(_bin_size_problems(raw["span_bin_size_m"], step, hi))
    return out


def _angle_table_problems(profile: Mapping) -> list[Problem]:
    if _function_of(profile) != "piecewise":
        return []
    table = TABLE_SPECS[TABLE_ANGLE]
    params = profile.get("angle_cost_params")
    params = params if isinstance(params, Mapping) else {}
    angles, costs = params.get("breakpoints"), params.get("costs")
    head = f"{table.label} ({table.key})"
    if not isinstance(angles, (list, tuple)) or not isinstance(costs, (list, tuple)):
        return [(table.key, f"{head}: missing. The piecewise shape needs a list of angles and " +
                            f"a list of costs, at least {table.min_rows} rows.")]
    if len(angles) != len(costs):
        return [(table.key, f"{head}: {len(angles)} angles but {len(costs)} costs. " +
                            "Every angle needs one cost.")]
    out = _angle_cell_problems(table, angles, costs)
    if len(angles) < table.min_rows:
        out.append((table.key, f"{head}: {len(angles)} rows is too few. Allowed: at least " +
                               f"{table.min_rows} rows."))
    out.extend(_angle_order_problems(table, head, angles))
    return out


def _angle_cell_problems(table: TableSpec, angles: Any, costs: Any) -> list[Problem]:
    ang_col, cost_col = table.columns
    return [(table.key, m) for a, c in zip(angles, costs)
            for spec, raw in ((ang_col, a), (cost_col, c))
            for m in _cell_problems(table, spec, raw)]


def _cell_problems(table: TableSpec, spec: FieldSpec, raw: Any) -> list[str]:
    if _is_blank(raw):
        return [f"{_subject(spec, table)}: a blank value. Allowed: {describe_range(spec)}."]
    return _value_problems(spec, raw, _subject(spec, table))


def _angle_order_problems(table: TableSpec, head: str, angles: Any) -> list[Problem]:
    numbers = [n for n in (_number(a) for a in angles) if n is not None]
    for before, after in zip(numbers, numbers[1:]):
        if after <= before:
            return [(table.key, f"{head}: angle {_fmt(after)} deg does not come after " +
                                f"{_fmt(before)} deg. Angles must strictly increase.")]
    return []


def _terrain_table_problems(profile: Mapping) -> list[Problem]:
    table = TABLE_SPECS[TABLE_TERRAIN]
    raw = _get(profile, table.path)
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return [(table.key, f"{table.label} ({table.key}): {_fmt(raw)} is not a table of raster " +
                            "value to cost.")]
    out: list[Problem] = []
    seen: set[int] = set()
    for key, cost in raw.items():
        out.extend(_terrain_entry_problems(table, key, cost))
        out.extend(_raster_repeat_problems(table, key, seen))
    return out


def _terrain_entry_problems(table: TableSpec, key: Any, cost: Any) -> list[Problem]:
    val_col, cost_col = table.columns
    out: list[Problem] = [(table.key, m) for m in
                          _value_problems(val_col, key, _subject(val_col, table))]
    if _is_blank(cost):
        out.append((table.key, f"{_subject(cost_col, table)}: a blank value for raster " +
                               f"value {_fmt(key)}. Allowed: {describe_range(cost_col)}."))
    else:
        out.extend((table.key, m) for m in
                   _value_problems(cost_col, cost, _subject(cost_col, table)))
    return out


def _raster_repeat_problems(table: TableSpec, key: Any, seen: set[int]) -> list[Problem]:
    number = _number(key)
    if number is None or number != int(number):
        return []
    found: list[Problem] = []
    if int(number) in seen:
        found.append((table.key, f"{table.label} ({table.key}): raster value " +
                                 f"{int(number)} appears twice. Use each value once."))
    seen.add(int(number))
    return found


def _types_table_problems(profile: Mapping) -> list[Problem]:
    table = TABLE_SPECS[TABLE_TYPES]
    raw = _get(profile, table.path)
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return [(table.key, f"{table.label} ({table.key}): {_fmt(raw)} is not a table of " +
                            "type name to settings.")]
    out: list[Problem] = []
    for name, entry in raw.items():
        out.extend(_type_problems(table, name, entry))
    return out


def _type_problems(table: TableSpec, name: Any, entry: Any) -> list[Problem]:
    out: list[Problem] = []
    if not str(name).strip():
        out.append((table.key, f"{table.label} ({table.key}): a type without a name. " +
                               "Allowed: 1 to 80 characters."))
    if not isinstance(entry, Mapping):
        out.append((table.key, f"{table.label} ({table.key}), type {_fmt(name)}: " +
                               "needs max_angle_deg and base_cost."))
        return out
    for spec in table.columns[1:]:
        subject = f"{_subject(spec, table)} of type {_fmt(name)}"
        value = entry.get(spec.key)
        if _is_blank(value):
            out.append((table.key, f"{subject}: missing. Allowed: {describe_range(spec)}."))
        else:
            out.extend((table.key, m) for m in _value_problems(spec, value, subject))
    return out

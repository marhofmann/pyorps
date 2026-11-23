"""
PYORPS GUI: the checks behind the corridor card (round 14, gap 4, package P9).

The settings table of the corridor card (:data:`FIELD_SPECS`), the validation of the settings values, the points of
the route draft and their checks against the cost raster, and the backend check. Every function returns
``(key, message)`` pairs (:data:`Problem`) instead of raising; only input that is not a point at all raises
``InvalidInputError``. :mod:`pyorps.gui.services.corridor` re-exports every public name, and callers import them from
there.

It imports nothing from ``pyorps.gui.ids``, Dash or the layout, and nothing from the corridor module itself.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyorps.core.exceptions import InvalidInputError
from pyorps.core.types import IMPASSABLE_CELL_COST

from . import routing

Problem = tuple[str, str]
Point = tuple[float, float]

TERMINALS_ALL = "all"
TERMINALS_ENDS = "sources_targets"
TERMINAL_CHOICES = (TERMINALS_ALL, TERMINALS_ENDS)

#: how many items one message lists before it says "and N more"
_LISTED = 5


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class SettingSpec:
    """One corridor setting: label, unit, limits, default and help (the card builds its controls from these)."""

    key: str
    label: str
    unit: str = ""
    kind: str = "number"            # "number" | "select" | "bool"
    minimum: float | None = None
    maximum: float | None = None
    allow_equal: bool = True        # False: the value must be strictly above ``minimum``
    whole: bool = False
    default: Any = None
    options: tuple[str, ...] = ()
    required: bool = True           # False: a blank value is allowed (it means ``blank_means``)
    blank_means: str = ""
    help: str = ""


FIELD_SPECS: dict[str, SettingSpec] = {s.key: s for s in (
    SettingSpec(
        "terminals", "Points to connect", kind="select", options=TERMINAL_CHOICES, default=TERMINALS_ALL,
        help="All control points: sources, waypoints and targets. Sources and targets only: waypoints are left out."),
    SettingSpec(
        "k_per_pair", "Crossings kept per point pair", minimum=1, maximum=5, whole=True, default=1,
        help="How many of the cheapest crossings between two neighbouring points are kept. The extra crossings are "
             "usually near-duplicates of the first, so 1 is almost always right."),
    SettingSpec(
        "min_shared_length_m", "Ignore shared runs shorter than", unit="m", minimum=0, maximum=10000, default=0.0,
        help="Two routes that run side by side for less than this length are not treated as sharing a trench. "
             "0 keeps the construction exact."),
    SettingSpec(
        "max_pair_cost", "Skip pairs costing more than", unit="cost units x m", minimum=0, allow_equal=False,
        default=None, required=False, blank_means="every pair is kept",
        help="Connections between two points that cost more than this are not built. The cost is the cell value "
             "times the length in metres (the unit of a section's construction cost: EUR when the raster holds "
             "EUR/m). Blank keeps every pair."),
    SettingSpec(
        "validate_pair_costs", "Also compute exact pair costs", kind="bool", default=False,
        help="Runs one extra least-cost search per point and reports how far the network's pair prices are above "
             "the exact ones. Slower."),
)}
SETTING_KEYS = tuple(FIELD_SPECS)
CORRIDOR_DEFAULTS: dict[str, Any] = {key: spec.default for key, spec in FIELD_SPECS.items()}

#: labels and fix texts of the problem keys that are not settings
_PROBLEM_TEXT: dict[str, tuple[str, str, str]] = {
    # key: (label, title, fix)
    "points": ("Points to connect", "The points cannot be connected",
               "Correct the points on the map (Routes tab) and try again."),
    "raster": ("Cost raster", "The cost raster cannot be used",
               "Pick a readable cost raster under 'Route on raster' and try again."),
    "graph_api": ("Backend", "The corridor graph needs the cython backend",
                  "Choose cython under Backend in 'Algorithm & search', then try again."),
    "neighborhood_k": ("Neighbourhood order k", "The neighbourhood order is not usable",
                       "Correct the k field in 'Algorithm & search' and try again."),
    "settings": ("Corridor settings", "The corridor settings are not usable",
                 "Reset the corridor card and try again."),
    "graph": ("Corridor graph", "There is no corridor graph to save",
              "Build the corridor graph first (a reopened project does not keep it), then save."),
    "save_path": ("Save path", "The file cannot be written",
                  "Correct the save path (a .gpkg file name) and try again."),
}


def label_for(key: str) -> str:
    """The label of a setting or problem key (the key itself when it is unknown)."""
    if key in FIELD_SPECS:
        return FIELD_SPECS[key].label
    return _PROBLEM_TEXT.get(key, (key,))[0]


# --------------------------------------------------------------------------- small helpers
def _fmt_real(value: Any) -> str:
    number = float(value)
    if math.isfinite(number) and number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


def _fmt(value: Any) -> str:
    """A value as the user should read it in a message."""
    if value is None:
        return "blank"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return _fmt_real(value)
    return repr(str(value))


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value: Any) -> float | None:
    """A finite float from a number or a numeric string; None for anything else (a bool is not a number)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _plain(number: float | None) -> float | int | None:
    """A whole float as an int (``2.0`` -> ``2``), anything else unchanged."""
    if number is not None and number == int(number) and abs(number) < 1e15:
        return int(number)
    return number


def _bounds_text(spec: SettingSpec) -> str:
    parts = []
    if spec.minimum is not None:
        parts.append(f"{'>=' if spec.allow_equal else '>'} {_fmt(spec.minimum)}")
    if spec.maximum is not None:
        parts.append(f"<= {_fmt(spec.maximum)}")
    return " and ".join(parts) if parts else "any number"


def _range_suffix(spec: SettingSpec) -> str:
    text = ""
    if spec.whole:
        text += ", whole numbers"
    if spec.unit:
        text += f" {spec.unit}"
    if not spec.required and spec.blank_means:
        text += f" (blank = {spec.blank_means})"
    return text


def describe_range(spec: SettingSpec) -> str:
    """The allowed range of a setting in words, e.g. ``>= 1 and <= 5, whole numbers``."""
    if spec.kind == "select":
        return ", ".join(spec.options)
    if spec.kind == "bool":
        return "on or off"
    return _bounds_text(spec) + _range_suffix(spec)


def _subject(spec: SettingSpec) -> str:
    return f"{spec.label} ({spec.key})"


def _select_problem(spec: SettingSpec, raw: Any, subject: str, allowed: str) -> str | None:
    if isinstance(raw, str) and raw in spec.options:
        return None
    return f"{subject}: {_fmt(raw)} is not an option. {allowed}"


def _bool_problem(raw: Any, subject: str, allowed: str) -> str | None:
    if raw is None or isinstance(raw, bool):
        return None                     # a switch has no blank: unset means off
    return f"{subject}: {_fmt(raw)} is not on or off. {allowed}"


def _limit_problem(spec: SettingSpec, value: float, lead: str, allowed: str) -> str | None:
    if spec.minimum is not None and (value < spec.minimum or (value == spec.minimum and not spec.allow_equal)):
        return f"{lead} is too low. {allowed}"
    if spec.maximum is not None and value > spec.maximum:
        return f"{lead} is too high. {allowed}"
    return None


def _number_setting_problem(spec: SettingSpec, raw: Any, subject: str, allowed: str) -> str | None:
    if _is_blank(raw):
        return None if not spec.required else f"{subject}: blank, but a value is required. {allowed}"
    value = _number(raw)
    if value is None:
        return f"{subject}: {_fmt(raw)} is not a usable number. {allowed}"
    shown = f"{_fmt(value)}{' ' + spec.unit if spec.unit else ''}"
    if spec.whole and value != int(value):
        return f"{subject}: {shown} is not a whole number. {allowed}"
    return _limit_problem(spec, value, f"{subject}: {shown}", allowed)


def _setting_problem(spec: SettingSpec, raw: Any) -> str | None:
    """Why one present value of a setting is not allowed, or None."""
    subject = _subject(spec)
    allowed = f"Allowed: {describe_range(spec)}."
    if spec.kind == "select":
        return _select_problem(spec, raw, subject, allowed)
    if spec.kind == "bool":
        return _bool_problem(raw, subject, allowed)
    return _number_setting_problem(spec, raw, subject, allowed)


# --------------------------------------------------------------------------- settings
def normalise_corridor_settings(raw: Any) -> dict[str, Any]:
    """The corridor settings of a project file (or of the form), as the controls take them.

    A project without them (an older file) gives ``{}``: the card keeps its own defaults. Otherwise every key of
    :data:`SETTING_KEYS` is present: a missing or unusable value falls back to its default. Values that are usable
    numbers but outside the allowed range are kept, not clamped, so that :func:`corridor_problems` can name them.
    The result holds plain JSON types only.
    """
    if not isinstance(raw, Mapping) or not raw:
        return {}
    out = dict(CORRIDOR_DEFAULTS)
    terminals = raw.get("terminals")
    out["terminals"] = terminals if isinstance(terminals, str) and terminals in TERMINAL_CHOICES else TERMINALS_ALL
    k_per_pair = _number(raw.get("k_per_pair"))
    out["k_per_pair"] = _plain(k_per_pair) if k_per_pair is not None else CORRIDOR_DEFAULTS["k_per_pair"]
    shared = _number(raw.get("min_shared_length_m"))
    out["min_shared_length_m"] = shared if shared is not None else CORRIDOR_DEFAULTS["min_shared_length_m"]
    out["max_pair_cost"] = _number(raw.get("max_pair_cost"))             # blank or unusable: no limit
    switch = raw.get("validate_pair_costs")
    out["validate_pair_costs"] = switch if isinstance(switch, bool) else CORRIDOR_DEFAULTS["validate_pair_costs"]
    return out


def corridor_problems(raw: Any) -> list[Problem]:
    """``(key, message)`` for every value of the settings that is not allowed; ``[]`` when all are fine.

    Give it the RAW values (the form's, or a stored dict), not the output of :func:`normalise_corridor_settings`:
    a blank required field is a problem here, while the normaliser would have replaced it by its default. A key
    that is absent counts as its default; unknown keys are ignored. ``None`` means "no settings", all defaults.
    """
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return [("settings", f"The corridor settings are {_fmt(raw)}, not a mapping of setting names to values.")]
    problems: list[Problem] = []
    for key, spec in FIELD_SPECS.items():
        if key in raw:
            message = _setting_problem(spec, raw[key])
            if message:
                problems.append((key, message))
    return problems


def _settings_with_defaults(raw: Any) -> dict[str, Any]:
    return {**CORRIDOR_DEFAULTS, **normalise_corridor_settings(raw)}


def max_pair_distance_cells(max_pair_cost: float | None, cell_size_m: float) -> float | None:
    """``max_pair_distance`` of the library (search CELL units) for a cost limit in cost units x m.

    One cell unit is the cost of crossing one cell of value 1, so ``cell units x cell size = cost units x m`` and
    the limit divided by the cell size is what ``build_corridor_graph(max_pair_distance=...)`` wants. ``None``
    (keep every pair) stays ``None``.
    """
    if max_pair_cost is None:
        return None
    if not (isinstance(cell_size_m, numbers.Real) and math.isfinite(cell_size_m) and cell_size_m > 0):
        raise InvalidInputError(f"The cell size must be a positive number of metres, not {_fmt(cell_size_m)}.")
    return float(max_pair_cost) / float(cell_size_m)


# --------------------------------------------------------------------------- the points
def _point(item: Any, where: str) -> Point:
    """One draft point (``{"x":, "y":}`` or ``[x, y]``) as an ``(x, y)`` float pair."""
    try:
        if isinstance(item, Mapping):
            return float(item["x"]), float(item["y"])
        x, y = item
        return float(x), float(y)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError(f"{where} is not a point with x and y coordinates: {item!r}.") from exc


def _as_points(terminals: Any) -> list[Point]:
    if isinstance(terminals, (str, bytes)) or not isinstance(terminals, Sequence):
        raise InvalidInputError(f"The points must be a list of (x, y) pairs, not {type(terminals).__name__}.")
    return [_point(item, f"Point {i}") for i, item in enumerate(terminals, start=1)]


def terminals_from_draft(draft: Any, which: str = TERMINALS_ALL) -> list[Point]:
    """The points to connect, from the route draft (``ROUTE_DRAFT``): ``(x, y)`` in the raster's CRS.

    ``which`` is ``"all"`` (sources, then waypoints, then targets) or ``"sources_targets"`` (waypoints left out).
    The order is the order of the point numbers in every message. Nothing is removed here: two points on the same
    spot are reported by :func:`terminal_problems` so the user can fix the draft. Raises ``InvalidInputError`` for
    an unknown ``which`` or an entry that is not a point.
    """
    if which not in TERMINAL_CHOICES:
        raise InvalidInputError(
            f"Points to connect: {which!r} is not an option. Allowed: {', '.join(TERMINAL_CHOICES)}.")
    if draft is None:
        return []
    if not isinstance(draft, Mapping):
        raise InvalidInputError(f"The route draft must be a mapping with sources, waypoints and targets, not "
                                f"{type(draft).__name__}.")
    groups = ("sources", "waypoints", "targets") if which == TERMINALS_ALL else ("sources", "targets")
    points: list[Point] = []
    for group in groups:
        for index, item in enumerate(draft.get(group) or [], start=1):
            points.append(_point(item, f"The {group[:-1]} {index} of the draft"))
    return points


def _where(index: int, point: Point) -> str:
    return f"point {index + 1} ({point[0]:,.1f}, {point[1]:,.1f})"


def _listed(items: list[str]) -> str:
    shown = "; ".join(items[:_LISTED])
    return shown + (f"; and {len(items) - _LISTED} more" if len(items) > _LISTED else "")


def _is_blocked(value: Any, floating: bool) -> bool:
    import numpy as np

    return (not np.isfinite(value)) if floating else value >= IMPASSABLE_CELL_COST


def _sample_cells(src: Any, indexed: list[tuple[int, Point]], rows: Any, cols: Any, inside_ids: list[int],
                  ignore_max_cost: bool) -> tuple[list[str], dict[tuple[int, int], list[int]]]:
    """``(forbidden, cells)``: the forbidden-cell messages and the point numbers - 1 found on each raster cell."""
    import numpy as np

    forbidden: list[str] = []
    cells: dict[tuple[int, int], list[int]] = {}
    if not inside_ids:
        return forbidden, cells
    values = [v[0] for v in src.sample([indexed[n][1] for n in inside_ids], indexes=1)]
    floating = np.issubdtype(src.dtypes[0], np.floating)
    for n, value in zip(inside_ids, values):
        i, point = indexed[n]
        cells.setdefault((int(rows[n]), int(cols[n])), []).append(i)
        blocked = _is_blocked(value, floating)
        if ignore_max_cost and blocked:
            forbidden.append(f"{_where(i, point)}, cell value {value:g}")
    return forbidden, cells


def _duplicate_messages(cells: dict[tuple[int, int], list[int]]) -> list[str]:
    return [f"points {', '.join(str(i + 1) for i in ids)} are on the same raster cell (row {row}, column {col})"
            for (row, col), ids in cells.items() if len(ids) > 1]


def _inspect_raster(src: Any, indexed: list[tuple[int, Point]],
                    ignore_max_cost: bool) -> tuple[list[str], list[str], list[str]]:
    """``(outside, forbidden, duplicates)`` message parts for ``indexed`` = ``[(point number - 1, point), ...]``."""
    import numpy as np
    from rasterio.transform import rowcol

    rows, cols = rowcol(src.transform, [p[0] for _, p in indexed], [p[1] for _, p in indexed])
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    outside = [_where(i, p) for n, (i, p) in enumerate(indexed) if not inside[n]]
    inside_ids = [n for n in range(len(indexed)) if inside[n]]
    forbidden, cells = _sample_cells(src, indexed, rows, cols, inside_ids, ignore_max_cost)
    return outside, forbidden, _duplicate_messages(cells)


def _point_count_problems(points: list[Point]) -> tuple[list[Problem], list[tuple[int, Point]]]:
    """``(problems, finite)``: too few points, non-finite coordinates, and the finite points with their index."""
    problems: list[Problem] = []
    if len(points) < 2:
        problems.append(("points", f"A corridor graph connects at least 2 points, but the draft gives {len(points)}. " +
                                   "Add more points on the map (sources, targets and, with 'All control points', " +
                                   "waypoints)."))
    finite = [(i, p) for i, p in enumerate(points) if math.isfinite(p[0]) and math.isfinite(p[1])]
    if len(finite) < len(points):
        bad = [_where(i, p) for i, p in enumerate(points) if (i, p) not in finite]
        problems.append(("points", f"{len(bad)} point(s) have a coordinate that is not a finite number: " +
                                   f"{_listed(bad)}."))
    return problems, finite


def _raster_point_problems(outside: list[str], forbidden: list[str], duplicates: list[str], bounds: Any,
                           crs: Any) -> list[Problem]:
    problems: list[Problem] = []
    if outside:
        problems.append(("points", f"{len(outside)} point(s) lie outside the cost raster: {_listed(outside)}. The " +
                                   f"raster covers x {bounds.left:,.1f} to {bounds.right:,.1f} and y " +
                                   f"{bounds.bottom:,.1f} to {bounds.top:,.1f} ({crs})."))
    if forbidden:
        problems.append(("points", f"{len(forbidden)} point(s) lie on forbidden cells (no-go, value " +
                                   f"{IMPASSABLE_CELL_COST}): {_listed(forbidden)}. Move them onto a passable cell."))
    if duplicates:
        problems.append(("points", f"Some points are duplicates: {_listed(duplicates)}. Remove all but one of each " +
                                   "group."))
    return problems


def terminal_problems(raster_path: Any, terminals: Sequence[Sequence[float]], *,
                      ignore_max_cost: bool = True) -> list[Problem]:
    """``("points" | "raster", message)`` for points that cannot be connected on this raster; ``[]`` when fine.

    Refused: fewer than two points, a coordinate that is not a finite number, a point outside the raster, a point
    on a forbidden cell (only while ``ignore_max_cost`` is on: then forbidden cells are hard barriers and the
    library would silently move the point), two points on the same raster cell (a corridor graph connects distinct
    points), an unreadable raster. Points are numbered from 1 in the order given. The points must be in the CRS of
    the raster.
    """
    points = _as_points(terminals)
    problems, finite = _point_count_problems(points)
    if not raster_path:
        problems.append(("raster", "No cost raster is selected, so the points cannot be checked against it."))
        return problems
    if not finite:
        return problems
    try:
        import rasterio

        with rasterio.open(str(raster_path)) as src:
            outside, forbidden, duplicates = _inspect_raster(src, finite, bool(ignore_max_cost))
            bounds, crs = src.bounds, src.crs
    except (OSError, ValueError) as exc:          # rasterio's errors are OSError / ValueError subclasses
        problems.append(("raster", f"The cost raster cannot be read ({raster_path}): {exc}"))
        return problems
    problems.extend(_raster_point_problems(outside, forbidden, duplicates, bounds, crs))
    return problems


def backend_problem(graph_api: Any) -> Problem | None:
    """``("graph_api", message)`` unless the Backend field is cython (the only backend that builds the graph)."""
    name = str(graph_api or routing.DEFAULT_BACKEND).strip().lower()
    if name == "cython":
        return None
    return ("graph_api", f"The corridor graph runs on the cython backend only; the Backend field is set to '{name}'. "
                         f"Choose cython under Backend in 'Algorithm & search'.")

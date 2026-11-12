"""
PYORPS GUI: the shared-trench corridor graph (round 14, gap 4, package P9).

A pure service behind the corridor card (package P12). It builds the library's corridor graph
(``PathFinder.build_corridor_graph``: the terminals, the cells where least-cost routes merge or diverge, and the
trench sections between them, each charged once) on the project's cost raster.

It imports nothing from ``pyorps.gui.ids``, Dash or the layout. A caller that needs component ids passes them in
(``controls=`` of :func:`problem_notice`).

The flow, in the order the card uses it:

1. **Before the run** (Dash thread): ``terminals_from_draft`` takes the points of the route draft,
   ``prepare_run`` validates the settings, the backend and the points against the raster and returns a
   :class:`RunPlan` (the ``params`` snapshot for the job, or the problems). ``problem_notice`` turns the problems
   into a notice.
2. **The run** (worker thread): ``jobs.start_job(state, JOB_KIND, run_job, label=JOB_LABEL, params=plan.params,
   can_stop=False)``. ``run_job`` calls :func:`build_corridor`, which reads only its arguments (never
   ``ProjectState``), reports its stage through ``progress(done, total, stage)`` and returns a
   :class:`CorridorResult` with everything the poll side needs. **The run cannot be stopped:** the library builds
   the graph in one uninterruptible call. The stage texts and :func:`status_text` say so.
3. **After the run** (Dash thread, poll callback): :func:`finalize_corridor` stores the graph in
   ``state.corridor_graphs`` and registers the map layers; :func:`save_corridor` writes the graph as a GeoPackage.

Settings (``state.gui_settings["corridor"]``, the shape :func:`normalise_corridor_settings` returns)::

    {"terminals": "all",              # "all" (sources, waypoints, targets) | "sources_targets"
     "k_per_pair": 1,                 # whole number, 1 to 5: crossings kept per point pair
     "min_shared_length_m": 0.0,      # metres, 0 to 10000: shared runs shorter than this do not count (0 = exact)
     "max_pair_cost": None,           # cost units x m, > 0, or None: skip pairs costing more (None = keep all)
     "validate_pair_costs": False}    # also compute the exact pair costs (one extra search per point)

``max_pair_cost`` is in the unit of a section's ``construction_cost`` (raster cell value times metres, EUR when
the raster holds EUR/m). The library wants it in search CELL units; :func:`max_pair_distance_cells` converts with
the cell size of the raster (``cell units x cell size = cost units x m``).

Rules the functions follow:

* A bad setting value or an unusable point does not raise: ``corridor_problems`` and ``terminal_problems`` return
  ``(key, message)`` pairs. Each message names the setting, the value, the allowed range and the unit.
  ``build_corridor`` checks the same things again and raises ``InvalidInputError`` (it is the last line of defence,
  not the way the card reports mistakes). Only input that is not a point at all (a draft entry without x and y, a
  string instead of a list of points) raises ``InvalidInputError`` straight away.
* No A* and no goal-directed search: the library builds the graph from one multi-source Dijkstra sweep.
* Plain cost raster only: no DEM, no objective, no constrained profile. Cython backend only (the library refuses
  every other backend).
"""
from __future__ import annotations

import math
import numbers
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyorps.core.exceptions import InvalidInputError, NoPathFoundError
from pyorps.core.types import IMPASSABLE_CELL_COST

from . import errors, geo, jobs, routing

__all__ = [
    "CANNOT_STOP_NOTE", "CORRIDOR_DEFAULTS", "FIELD_SPECS", "JOB_KIND", "JOB_LABEL", "NODES_LAYER",
    "SEGMENTS_LAYER", "SETTING_KEYS", "TERMINALS_ALL", "TERMINALS_ENDS", "TERMINAL_CHOICES",
    "CorridorResult", "Finalized", "RunPlan", "SettingSpec", "backend_problem", "build_corridor",
    "corridor_problems", "describe_range", "estimate_run", "finalize_corridor", "label_for",
    "max_pair_distance_cells", "normalise_corridor_settings", "prepare_run", "problem_notice", "problems_text",
    "run_job", "save_corridor", "save_problems", "status_text", "summary_text", "terminal_problems",
    "terminals_from_draft",
]

Problem = tuple[str, str]
Point = tuple[float, float]

#: the job kind in ``state.jobs`` and its label (the status line reads "Corridor graph... 2/4 ...")
JOB_KIND = "corridor"
JOB_LABEL = "Corridor graph"
#: what the status text says about Stop: this job has none
CANNOT_STOP_NOTE = "cannot be stopped; you can keep working"

TERMINALS_ALL = "all"
TERMINALS_ENDS = "sources_targets"
TERMINAL_CHOICES = (TERMINALS_ALL, TERMINALS_ENDS)

#: the two layers of the saved GeoPackage (the library's own names)
SEGMENTS_LAYER = "corridor_segments"
NODES_LAYER = "corridor_nodes"

#: the map style of the layers a finished run adds (one style per layer: the "shared" overlay carries the
#: per-section distinction a single Leaflet style cannot)
SEGMENTS_STYLE = {"color": "#1f78b4", "weight": 3, "opacity": 0.9}
SHARED_STYLE = {"color": "#d62728", "weight": 6, "opacity": 0.85}
NODES_STYLE = {"color": "#111111", "weight": 2, "fillOpacity": 0.8}

#: a window above either of these is "heavy": the plan carries a warning (the run is still allowed)
HEAVY_WINDOW_CELLS = 30_000_000
HEAVY_MEMORY_MB = 2000.0
#: rough bytes per cell of the multi-source sweep (cost, distance, predecessor, region label, heap), order of magnitude
_BYTES_PER_CELL = 24

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
def _fmt(value: Any) -> str:
    """A value as the user should read it in a message."""
    if value is None:
        return "blank"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number == int(number) and abs(number) < 1e15:
            return str(int(number))
        return repr(number)
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


def describe_range(spec: SettingSpec) -> str:
    """The allowed range of a setting in words, e.g. ``>= 1 and <= 5, whole numbers``."""
    if spec.kind == "select":
        return ", ".join(spec.options)
    if spec.kind == "bool":
        return "on or off"
    parts = []
    if spec.minimum is not None:
        parts.append(f"{'>=' if spec.allow_equal else '>'} {_fmt(spec.minimum)}")
    if spec.maximum is not None:
        parts.append(f"<= {_fmt(spec.maximum)}")
    text = " and ".join(parts) if parts else "any number"
    if spec.whole:
        text += ", whole numbers"
    if spec.unit:
        text += f" {spec.unit}"
    if not spec.required and spec.blank_means:
        text += f" (blank = {spec.blank_means})"
    return text


def _subject(spec: SettingSpec) -> str:
    return f"{spec.label} ({spec.key})"


def _setting_problem(spec: SettingSpec, raw: Any) -> str | None:
    """Why one present value of a setting is not allowed, or None."""
    subject = _subject(spec)
    allowed = f"Allowed: {describe_range(spec)}."
    if spec.kind == "select":
        if isinstance(raw, str) and raw in spec.options:
            return None
        return f"{subject}: {_fmt(raw)} is not an option. {allowed}"
    if spec.kind == "bool":
        if raw is None or isinstance(raw, bool):
            return None                     # a switch has no blank: unset means off
        return f"{subject}: {_fmt(raw)} is not on or off. {allowed}"
    if _is_blank(raw):
        return None if not spec.required else f"{subject}: blank, but a value is required. {allowed}"
    value = _number(raw)
    if value is None:
        return f"{subject}: {_fmt(raw)} is not a usable number. {allowed}"
    shown = f"{_fmt(value)}{' ' + spec.unit if spec.unit else ''}"
    if spec.whole and value != int(value):
        return f"{subject}: {shown} is not a whole number. {allowed}"
    if spec.minimum is not None and (value < spec.minimum or (value == spec.minimum and not spec.allow_equal)):
        return f"{subject}: {shown} is too low. {allowed}"
    if spec.maximum is not None and value > spec.maximum:
        return f"{subject}: {shown} is too high. {allowed}"
    return None


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


def _inspect_raster(src: Any, indexed: list[tuple[int, Point]],
                    ignore_max_cost: bool) -> tuple[list[str], list[str], list[str]]:
    """``(outside, forbidden, duplicates)`` message parts for ``indexed`` = ``[(point number - 1, point), ...]``."""
    import numpy as np
    from rasterio.transform import rowcol

    rows, cols = rowcol(src.transform, [p[0] for _, p in indexed], [p[1] for _, p in indexed])
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    outside = [_where(i, p) for n, (i, p) in enumerate(indexed) if not inside[n]]

    forbidden: list[str] = []
    cells: dict[tuple[int, int], list[int]] = {}
    inside_ids = [n for n in range(len(indexed)) if inside[n]]
    if inside_ids:
        values = [v[0] for v in src.sample([indexed[n][1] for n in inside_ids], indexes=1)]
        floating = np.issubdtype(src.dtypes[0], np.floating)
        for n, value in zip(inside_ids, values):
            i, point = indexed[n]
            cells.setdefault((int(rows[n]), int(cols[n])), []).append(i)
            blocked = (not np.isfinite(value)) if floating else value >= IMPASSABLE_CELL_COST
            if ignore_max_cost and blocked:
                forbidden.append(f"{_where(i, point)}, cell value {value:g}")
    duplicates = [f"points {', '.join(str(i + 1) for i in ids)} are on the same raster cell (row {row}, column {col})"
                  for (row, col), ids in cells.items() if len(ids) > 1]
    return outside, forbidden, duplicates


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
    problems: list[Problem] = []
    if len(points) < 2:
        problems.append(("points", f"A corridor graph connects at least 2 points, but the draft gives {len(points)}. "
                                   f"Add more points on the map (sources, targets and, with 'All control points', "
                                   f"waypoints)."))
    finite = [(i, p) for i, p in enumerate(points) if math.isfinite(p[0]) and math.isfinite(p[1])]
    if len(finite) < len(points):
        bad = [_where(i, p) for i, p in enumerate(points) if (i, p) not in finite]
        problems.append(("points", f"{len(bad)} point(s) have a coordinate that is not a finite number: "
                                   f"{_listed(bad)}."))
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
    if outside:
        problems.append(("points", f"{len(outside)} point(s) lie outside the cost raster: {_listed(outside)}. The "
                                   f"raster covers x {bounds.left:,.1f} to {bounds.right:,.1f} and y "
                                   f"{bounds.bottom:,.1f} to {bounds.top:,.1f} ({crs})."))
    if forbidden:
        problems.append(("points", f"{len(forbidden)} point(s) lie on forbidden cells (no-go, value "
                                   f"{IMPASSABLE_CELL_COST}): {_listed(forbidden)}. Move them onto a passable cell."))
    if duplicates:
        problems.append(("points", f"Some points are duplicates: {_listed(duplicates)}. Remove all but one of each "
                                   f"group."))
    return problems


def backend_problem(graph_api: Any) -> Problem | None:
    """``("graph_api", message)`` unless the Backend field is cython (the only backend that builds the graph)."""
    name = str(graph_api or routing.DEFAULT_BACKEND).strip().lower()
    if name == "cython":
        return None
    return ("graph_api", f"The corridor graph runs on the cython backend only; the Backend field is set to '{name}'. "
                         f"Choose cython under Backend in 'Algorithm & search'.")


# --------------------------------------------------------------------------- notices of problems
def problems_text(problems: list[Problem], limit: int = 3) -> str:
    """The first ``limit`` messages, one per line, plus a count of the rest (for the status line)."""
    lines = [message for _, message in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"... and {len(problems) - limit} more.")
    return "\n".join(lines)


def problem_notice(problems: list[Problem], *, controls: Mapping[str, str] | None = None, tab: str = "tab-routes",
                   impact: str = "No corridor graph was built.") -> dict | None:
    """The first problem as a notice dict (``Notice.to_dict()``), or None without problems.

    ``controls`` maps a problem key (a setting name, ``"points"``, ``"raster"``, ``"graph_api"``,
    ``"save_path"``) to the component id the "Go fix" button scrolls to; the caller owns the ids.
    """
    if not problems:
        return None
    key, message = problems[0]
    label, title, fix = _PROBLEM_TEXT.get(key, (label_for(key), "", ""))
    if key in FIELD_SPECS:
        title = f"{label}: value not usable"
        fix = f"Correct the '{label}' field in the corridor card and try again."
    return errors.Notice(
        severity="warning", title=title or f"{label}: not usable", meaning=message, impact=impact,
        fix=fix or "Correct the value and try again.", focus_id=tab,
        focus_control=(controls or {}).get(key)).to_dict()


# --------------------------------------------------------------------------- the run plan
@dataclass
class RunPlan:
    """What :func:`prepare_run` decided: the job parameters, or the reasons it refuses."""

    #: the snapshot for ``jobs.start_job(params=...)``; None when there are problems
    params: dict[str, Any] | None
    #: ``(key, message)`` pairs; empty when the run may start
    problems: list[Problem] = field(default_factory=list)
    #: advisory notices (a heavy window); they do not stop the run
    notices: list[dict] = field(default_factory=list)
    #: ``window_cells``, ``est_memory_mb``, ``buffer_m``, ``cell_size_m``, ``heavy``
    estimate: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.params is not None and not self.problems


def _search_buffer(raster_path: Any, points: list[Point], search_buffer_m: float | None, full_window: bool) -> float:
    """The search buffer in metres: the whole raster, the user's value, or the automatic one (never blank)."""
    if full_window:
        return routing.whole_raster_buffer(str(raster_path))
    if search_buffer_m is not None and search_buffer_m > 0:
        return float(search_buffer_m)
    return routing.default_search_buffer(points, points)


def estimate_run(raster_path: Any, terminals: Sequence[Sequence[float]], *, search_buffer_m: float | None = None,
                 full_window: bool = False) -> dict[str, Any]:
    """A rough prognosis of the search window: its cells, the memory the sweep needs, whether that is heavy.

    The window is the bounding box of the points plus the buffer, clipped to the raster (the whole raster for
    ``full_window``). The numbers are orders of magnitude, not promises.
    """
    import rasterio

    points = _as_points(terminals)
    buffer_m = _search_buffer(raster_path, points, search_buffer_m, full_window)
    with rasterio.open(str(raster_path)) as src:
        cell = abs(src.transform.a) or 1.0
        left, bottom, right, top = src.bounds
        total = float(src.width) * float(src.height)
    if full_window or not points:
        cells = total
    else:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        width = max(0.0, min(max(xs) + buffer_m, right) - max(min(xs) - buffer_m, left))
        height = max(0.0, min(max(ys) + buffer_m, top) - max(min(ys) - buffer_m, bottom))
        cells = min(width / cell * (height / cell), total)
    memory_mb = cells * _BYTES_PER_CELL / 1e6
    return {"window_cells": int(cells), "est_memory_mb": float(memory_mb), "buffer_m": float(buffer_m),
            "cell_size_m": float(cell), "heavy": bool(cells > HEAVY_WINDOW_CELLS or memory_mb > HEAVY_MEMORY_MB)}


def _routing_options(routing_settings: Any) -> tuple[dict[str, Any], list[Problem]]:
    """The routing settings the corridor run reads (neighbourhood, buffer, window, forbidden cells), or problems."""
    raw = routing_settings if isinstance(routing_settings, Mapping) else {}
    problems: list[Problem] = []
    problem = backend_problem(raw.get("graph_api"))
    if problem:
        problems.append(problem)
    k_value = raw.get("neighborhood_k")
    try:
        neighborhood = routing.neighborhood_string(k_value)
        if not routing.NEIGHBORHOOD_MIN <= int(neighborhood[1:]) <= routing.NEIGHBORHOOD_MAX:
            raise ValueError(k_value)
    except (TypeError, ValueError):
        neighborhood = routing.neighborhood_string(None)
        problems.append(("neighborhood_k", f"Neighbourhood order k: {_fmt(k_value)} is not usable. Allowed: whole "
                                           f"numbers from {routing.NEIGHBORHOOD_MIN} to {routing.NEIGHBORHOOD_MAX}."))
    buffer_m = _number(raw.get("search_buffer_m"))
    ignore = raw.get("ignore_max_cost")
    return ({"neighborhood": neighborhood,
             "search_buffer_m": buffer_m if buffer_m is not None and buffer_m > 0 else None,
             "full_window": bool(raw.get("full_window")),
             "ignore_max_cost": ignore if isinstance(ignore, bool) else True}, problems)


def prepare_run(raster_path: Any, draft: Any, settings: Any, routing_settings: Any = None) -> RunPlan:
    """Check everything a run needs and build the job's parameter snapshot.

    ``raster_path`` is the cost raster file, ``draft`` the route draft (sources / waypoints / targets with ``x`` and
    ``y`` in the raster's CRS), ``settings`` the corridor settings (raw form values or the stored dict) and
    ``routing_settings`` the routing card's values (``graph_api``, ``neighborhood_k``, ``search_buffer_m``,
    ``full_window``, ``ignore_max_cost``; ``state.routing_settings`` has them; anything missing means the routing
    defaults). Nothing raises: a refused run returns the problems (in the order backend, settings, points).
    The snapshot (``plan.params``) holds only plain values and is what :func:`run_job` reads.
    """
    options, problems = _routing_options(routing_settings)
    problems = [*problems, *corridor_problems(settings)]
    cfg = _settings_with_defaults(settings)
    try:
        points = terminals_from_draft(draft, cfg["terminals"])
    except InvalidInputError as exc:
        points = None
        problems.append(("points", str(exc)))
    if points is not None:
        problems.extend(terminal_problems(raster_path, points, ignore_max_cost=options["ignore_max_cost"]))
    if problems:
        return RunPlan(params=None, problems=problems)
    estimate = estimate_run(raster_path, points, search_buffer_m=options["search_buffer_m"],
                            full_window=options["full_window"])
    notices = []
    if estimate["heavy"]:
        notices.append(errors.Notice(
            severity="warning", title="This corridor graph covers a large area",
            meaning=f"The search window holds about {estimate['window_cells']:,} cells and the sweep needs roughly "
                    f"{estimate['est_memory_mb']:,.0f} MB (a rough estimate).",
            impact="The run may take a long time and cannot be stopped once it has started.",
            fix="Set a smaller search buffer under 'Algorithm & search' if that is more than you need.",
            focus_id="tab-routes").to_dict())
    params = {"raster_path": str(raster_path), "terminals": [list(p) for p in points], "settings": cfg, **options}
    return RunPlan(params=params, problems=[], notices=notices, estimate=estimate)


# --------------------------------------------------------------------------- the run
_STAGES = ("checking the points and the raster", "reading the search window", "building the shared trenches",
           "preparing the map layers")


@dataclass(eq=False)
class CorridorResult:
    """What a finished run returns (the worker's result; :func:`finalize_corridor` shows it).

    ``graph`` is None when no pair of points could be connected (every candidate cost more than the limit, or the
    points are cut off from each other): then the frames and GeoJSON are None, the counts in ``summary`` are 0 and
    ``notices`` says why. The GeoJSON is a WGS84 FeatureCollection (what the map layers carry); the frames stay in
    the raster's CRS (``crs``).
    """

    graph: Any
    summary: dict[str, Any]
    segments_gdf: Any = None
    nodes_gdf: Any = None
    shared_gdf: Any = None
    segments_geojson: dict | None = None
    nodes_geojson: dict | None = None
    shared_geojson: dict | None = None
    crs: Any = None
    notices: list[dict] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


def _report(progress: Callable[..., None] | None, done: int) -> None:
    if progress is not None:
        progress(done, len(_STAGES), f"{_STAGES[done]} ({CANNOT_STOP_NOTE})")


def _validation(graph: Any) -> dict[str, Any] | None:
    """What the pair-cost validation found, or None when it was not run."""
    gap = graph.provenance.get("pair_cost_gap")
    if gap is None:
        return None
    per_pair = gap.get("per_pair") or {}
    worst = None
    if per_pair:
        a, b = max(per_pair, key=per_pair.get).split("-")
        worst = [int(a) + 1, int(b) + 1]
    out = {"n_pairs": len(per_pair), "max_excess": float(gap.get("max_excess", 0.0)),
           "mean_excess": float(gap.get("mean_excess", 0.0)), "min_excess": float(gap.get("min_excess", 0.0)),
           "worst_pair_points": worst}
    out["text"] = (
        f"Exact pair costs were computed for {out['n_pairs']} pair(s): the network prices a pair at most "
        f"{out['max_excess'] * 100:.1f} % above the exact least-cost route (mean {out['mean_excess'] * 100:.1f} %)."
        if per_pair else "Exact pair costs could not be computed for any pair.")
    return out


def summary_text(summary: Mapping[str, Any]) -> str:
    """The summary as plain sentences (the status line and the success notice)."""
    if not summary.get("n_corridors"):
        return "No corridors were built."
    sections = summary["n_corridors"]
    text = (f"{summary['n_terminals']} points joined by {sections} trench section{'' if sections == 1 else 's'}; "
            f"{summary['n_shared_sections']} of them shared by several connections. "
            f"Together the sections are {summary['total_length_m']:,.0f} m long, {summary['shared_length_m']:,.0f} m "
            f"of that shared.")
    if summary["saved_length_m"] > 0:
        text += (f" Digging every connection on its own would need {summary['per_route_length_m']:,.0f} m, "
                 f"so sharing saves {summary['saved_length_m']:,.0f} m.")
    validation = summary.get("validation")
    return f"{text} {validation['text']}" if validation else text


def _notice(severity: str, title: str, meaning: str, impact: str, fix: str) -> dict:
    return errors.Notice(severity=severity, title=title, meaning=meaning, impact=impact, fix=fix,
                         focus_id="tab-routes").to_dict()


def _graph_notices(graph: Any, n_points: int, validation: dict | None) -> list[dict]:
    """What the user should see about the run, derived from the graph (warnings of a worker thread are not captured)."""
    notices: list[dict] = []
    connected = {t for key in graph.pair_routes for t in key[:2]}
    left_out = [i + 1 for i in range(n_points) if i not in connected]
    if left_out:
        notices.append(_notice(
            "warning", "Some points are not connected",
            f"Point(s) {', '.join(map(str, left_out))} appear in no connected pair: they are enclosed by forbidden "
            f"cells, outside the search window, or every route to them runs through another point's region.",
            "They are missing from the corridor graph.",
            "Check the cost raster around them, or widen the search buffer under 'Algorithm & search'."))
    for note in graph.provenance.get("notes") or []:
        if not str(note).startswith("terminals "):          # the isolated-points note is covered above
            notices.append(_notice("info", "A note from the corridor construction", str(note),
                                   "The graph is built, but this detail may matter.", "No action needed."))
    if validation and validation["min_excess"] < -1e-9:
        notices.append(_notice(
            "warning", "The exact pair costs disagree with the network",
            f"A pair was priced {abs(validation['min_excess']) * 100:.2f} % below its exact least-cost route, which "
            f"cannot happen for a real route.", "Treat the pair prices of this graph with care.",
            "Report it together with the raster and the points."))
    return notices


def _crs_notices(crs: Any) -> list[dict]:
    try:
        from pyproj import CRS

        projected = CRS.from_user_input(crs).is_projected
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # an unknown CRS is not a reason to fail
        return []
    if projected:
        return []
    return [_notice("info", "The raster is not in a metric CRS",
                    f"The raster's CRS ({crs}) is not projected, so lengths are in its own units, not metres.",
                    "Lengths and costs per metre in the summary are not metric.",
                    "Use a raster in a projected (metric) CRS for meaningful lengths.")]


def _summary(graph: Any, n_points: int, cfg: dict[str, Any], validation: dict | None,
             crs: Any) -> dict[str, Any]:
    total = float(graph.total_length)
    shared = float(sum(s.length for s in graph.shared_segments))
    report = graph.overlap_report()
    summary = {
        "n_points": n_points, "n_terminals": len(graph.terminal_nodes),
        "n_pairs": len({key[:2] for key in graph.pair_routes}), "n_routes": len(graph.pair_routes),
        "n_corridors": len(graph.segments), "n_shared_sections": len(graph.shared_segments),
        "n_junctions": len(graph.junctions),
        "total_length_m": total, "shared_length_m": shared, "shared_fraction": shared / total if total else 0.0,
        "per_route_length_m": float(report.route_length_m), "saved_length_m": float(report.route_length_m - total),
        "total_construction_cost": float(graph.total_construction_cost),
        "shared_construction_cost": float(report.shared_cost),
        "cell_size_m": float(graph.provenance.get("cell_size_m", graph.cell_size)), "crs": str(crs),
        "validation": validation, "settings": dict(cfg),
    }
    summary["text"] = summary_text(summary)
    return summary


def _empty_result(n_points: int, cfg: dict[str, Any], cell_size: float) -> CorridorResult:
    """The outcome when no pair of points is connected: no graph, and a notice that says why."""
    limit = cfg["max_pair_cost"]
    if limit is not None:
        meaning = (f"Every connection between two points costs more than the limit of {limit:,g} cost units x m "
                   f"(about {limit / cell_size:,.0f} cell units at {cell_size:g} m per cell).")
        fix = "Raise 'Skip pairs costing more than' or leave it blank."
    else:
        meaning = ("No two points share a region boundary inside the search window: they are cut off from each other "
                   "by forbidden cells or lie outside the window.")
        fix = "Widen the search buffer under 'Algorithm & search' or check the cost raster for barriers."
    summary = {"n_points": n_points, "n_terminals": 0, "n_pairs": 0, "n_routes": 0, "n_corridors": 0,
               "n_shared_sections": 0, "n_junctions": 0, "total_length_m": 0.0, "shared_length_m": 0.0,
               "shared_fraction": 0.0, "per_route_length_m": 0.0, "saved_length_m": 0.0,
               "total_construction_cost": 0.0, "shared_construction_cost": 0.0, "cell_size_m": cell_size,
               "crs": "", "validation": None, "settings": dict(cfg)}
    summary["text"] = summary_text(summary)
    return CorridorResult(
        graph=None, summary=summary, settings=dict(cfg),
        notices=[_notice("warning", "No corridors were built", meaning, "The project and the map are unchanged.",
                         fix)])


def _clean_crs(crs: Any) -> Any:
    """The CRS as a pyproj object that prints as ``EPSG:xxxx`` (the library hands over rasterio's, which prints WKT)."""
    try:
        from pyproj import CRS

        parsed = CRS.from_user_input(crs)
        epsg = parsed.to_epsg()
        return CRS.from_epsg(epsg) if epsg else parsed
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # keep what the library gave us
        return crs


def _cell_size(raster_path: Any) -> float:
    import rasterio

    with rasterio.open(str(raster_path)) as src:
        return abs(src.transform.a)


def build_corridor(raster_path: Any, terminals: Sequence[Sequence[float]], settings: Any = None, *,
                   neighborhood: str = "r2", search_buffer_m: float | None = None, full_window: bool = False,
                   ignore_max_cost: bool = True,
                   progress: Callable[..., None] | None = None) -> CorridorResult:
    """Build the shared-trench corridor graph of ``terminals`` on the cost raster ``raster_path``.

    **This cannot be stopped.** The library builds the graph in one uninterruptible call (a multi-source
    Dijkstra sweep, then the overlay of the recovered routes); it has no cancel hook. ``progress(done, total,
    stage)`` is called at four stage boundaries (``total`` is 4); each ``stage`` text says that the run cannot be
    stopped. It is safe for a worker thread: it reads only its arguments, never ``ProjectState``.

    ``terminals`` are ``(x, y)`` in the raster's CRS; ``settings`` are the corridor settings (any subset of
    :data:`SETTING_KEYS`, missing ones default). ``neighborhood`` ("rK"), ``search_buffer_m`` (None = automatic),
    ``full_window`` and ``ignore_max_cost`` are the routing card's values. Plain cost raster, cython backend, no
    DEM and no objective.

    Raises ``InvalidInputError`` when the settings or the points would have been refused by
    :func:`corridor_problems` / :func:`terminal_problems` (the message lists them). A run in which no pair can be
    connected is not an error: the result has ``graph=None`` and a notice.
    """
    from pyorps import PathFinder

    _report(progress, 0)
    points = _as_points(terminals)
    problems = [*corridor_problems(settings), *terminal_problems(raster_path, points, ignore_max_cost=ignore_max_cost)]
    if problems:
        raise InvalidInputError("The corridor graph cannot be built:\n" + "\n".join(m for _, m in problems))
    cfg = _settings_with_defaults(settings)
    cell_size = _cell_size(raster_path)
    limit = max_pair_distance_cells(cfg["max_pair_cost"], cell_size)
    buffer_m = _search_buffer(raster_path, points, search_buffer_m, full_window)

    _report(progress, 1)
    finder = PathFinder(
        str(raster_path), source_coords=points, target_coords=points, graph_api="cython",
        neighborhood_str=neighborhood, search_space_buffer_m=buffer_m, ignore_max_cost=bool(ignore_max_cost))

    _report(progress, 2)
    try:
        graph = finder.build_corridor_graph(
            points, k_per_pair=int(cfg["k_per_pair"]), min_shared_length_m=float(cfg["min_shared_length_m"]),
            max_pair_distance=limit, validate_pair_costs=bool(cfg["validate_pair_costs"]))
    except NoPathFoundError:
        return _empty_result(len(points), cfg, cell_size)

    _report(progress, 3)
    validation = _validation(graph)
    crs = _clean_crs(graph.crs)
    summary = _summary(graph, len(points), cfg, validation, crs)
    segments = graph.to_geodataframe().set_crs(crs, allow_override=True)
    nodes = graph.nodes_to_geodataframe().set_crs(crs, allow_override=True)
    shared = segments[segments["shared"] == 1]
    return CorridorResult(
        graph=graph, summary=summary, segments_gdf=segments, nodes_gdf=nodes,
        shared_gdf=shared if len(shared) else None,
        segments_geojson=geo.gdf_to_wgs84_geojson(segments), nodes_geojson=geo.gdf_to_wgs84_geojson(nodes),
        shared_geojson=geo.gdf_to_wgs84_geojson(shared) if len(shared) else None,
        crs=segments.crs, settings=cfg,
        notices=[*_graph_notices(graph, len(points), validation), *_crs_notices(graph.crs)])


#: the ``params`` keys :func:`run_job` hands to :func:`build_corridor` (others are for the poll side)
_BUILD_KEYS = ("raster_path", "terminals", "settings", "neighborhood", "search_buffer_m", "full_window",
               "ignore_max_cost")


def run_job(job: Any) -> CorridorResult:
    """``work(job)`` of the corridor job.

    Start it as ``jobs.start_job(state, JOB_KIND, run_job, label=JOB_LABEL, params=plan.params, can_stop=False)``.
    Reads ``job.params`` (a :class:`RunPlan` snapshot; keys it does not know are ignored), reports progress through
    ``job.report`` and returns the :class:`CorridorResult`. The run cannot be stopped.
    """
    kwargs = {key: job.params[key] for key in _BUILD_KEYS if key in job.params}
    return build_corridor(progress=job.report, **kwargs)


def status_text(job: Any) -> str:
    """The status line of a corridor job: ``job.progress_text()`` plus, while it runs, that it cannot be stopped."""
    text = job.progress_text()
    if getattr(job, "status", jobs.RUNNING) == jobs.RUNNING and CANNOT_STOP_NOTE not in text:
        text += f" ({CANNOT_STOP_NOTE})"
    return text


# --------------------------------------------------------------------------- after the run (Dash thread)
@dataclass
class Finalized:
    """What :func:`finalize_corridor` did to the project state."""

    segments_layer_id: str | None
    nodes_layer_id: str | None
    shared_layer_id: str | None
    layer_ids: list[str]
    summary: dict[str, Any]
    #: the run's own notices plus the success notice (when a graph was built)
    notices: list[dict]
    #: the one-paragraph summary for the card's status component
    status_text: str


def _run_number(state: Any) -> int:
    runs = [int((layer.meta or {}).get("corridor_run") or 0) for layer in state.layers.values()
            if (layer.meta or {}).get("corridor_role") == "segments"]
    return max(runs, default=0) + 1


def finalize_corridor(state: Any, result: CorridorResult, params: Mapping[str, Any] | None = None) -> Finalized:
    """Store the graph and register the map layers; call it from the poll callback (main thread), once per run.

    Adds a vector layer "Corridor segments" (every trench section, with the graph's attributes; later runs are
    named "Corridor segments (2)" and so on), a layer "Corridor shared runs" with only the sections used by more
    than one connection (drawn thicker and red, on top: one Leaflet style cannot colour by attribute), and a layer
    "Corridor nodes" (the points and derived junctions). The graph is kept in
    ``state.corridor_graphs[segments_layer_id]``; entries of layers that no longer exist are dropped. When the run
    found no corridor (``result.graph is None``) the state is left alone. ``params`` is the job's snapshot; its
    ``raster_layer_id``, when the caller put one there, is recorded on the layers.
    """
    from . import data_io

    if result.graph is None:
        return Finalized(None, None, None, [], result.summary, list(result.notices), result.summary["text"])
    meta_extra = {"raster_layer_id": params["raster_layer_id"]} if params and params.get("raster_layer_id") else {}
    with state.locked():
        for layer_id in [lid for lid in state.corridor_graphs if lid not in state.layers]:
            del state.corridor_graphs[layer_id]
        run = _run_number(state)
        suffix = "" if run == 1 else f" ({run})"
        segments = state.add_layer(
            f"Corridor segments{suffix}", "vector", gdf=result.segments_gdf, crs=result.crs,
            geojson=result.segments_geojson, style=dict(SEGMENTS_STYLE),
            meta={"summary": data_io.dataset_summary(result.segments_gdf), "corridor_role": "segments",
                  "corridor_run": run, "corridor_settings": dict(result.settings), **meta_extra})
        shared = None
        if result.shared_gdf is not None:
            shared = state.add_layer(
                f"Corridor shared runs{suffix}", "vector", gdf=result.shared_gdf, crs=result.crs,
                geojson=result.shared_geojson, style=dict(SHARED_STYLE),
                meta={"summary": data_io.dataset_summary(result.shared_gdf), "corridor_role": "shared",
                      "corridor_run": run, "corridor_for": segments.id})
        nodes = state.add_layer(
            f"Corridor nodes{suffix}", "vector", gdf=result.nodes_gdf, crs=result.crs,
            geojson=result.nodes_geojson, style=dict(NODES_STYLE),
            meta={"summary": data_io.dataset_summary(result.nodes_gdf), "corridor_role": "nodes",
                  "corridor_run": run, "corridor_for": segments.id})
        state.corridor_graphs[segments.id] = result.graph
    ids_added = [segments.id] + ([shared.id] if shared is not None else []) + [nodes.id]
    text = result.summary["text"]
    done = errors.success(
        f"Corridor graph built: {result.summary['n_corridors']} trench sections", meaning=text,
        impact="The sections and points were added as map layers; the graph is kept so you can save it as a "
               "GeoPackage (a reopened project keeps the layers but not the graph).")
    return Finalized(segments.id, nodes.id, shared.id if shared is not None else None, ids_added, result.summary,
                     [*result.notices, done], text)


# --------------------------------------------------------------------------- save
def save_problems(graph: Any, path: Any, *, overwrite: bool = False) -> list[Problem]:
    """``("graph" | "save_path", message)`` for a save that would be refused; ``[]`` when it can go ahead.

    A file that already exists is refused unless ``overwrite`` is true (the card asks the user first).
    """
    problems: list[Problem] = []
    if graph is None:
        problems.append(("graph", "There is no corridor graph to save. Build one first: a reopened project keeps "
                                  "the map layers but not the graph."))
    if _is_blank(path):
        problems.append(("save_path", "No file name was given. Enter a path ending in .gpkg."))
        return problems
    target = Path(str(path).strip())
    if target.suffix.lower() != ".gpkg":
        problems.append(("save_path", f"Save path: '{target.name}' is not a GeoPackage. Allowed: a file name ending "
                                      f"in .gpkg."))
    elif target.is_dir():
        problems.append(("save_path", f"Save path: '{target}' is a folder. Enter a file name ending in .gpkg."))
    elif target.exists() and not overwrite:
        problems.append(("save_path", f"'{target.name}' already exists in {target.parent}. Nothing was written; "
                                      f"choose another file name or confirm replacing it."))
    return problems


def save_corridor(graph: Any, path: Any, *, overwrite: bool = False) -> str:
    """Write the corridor graph as a GeoPackage with the layers ``corridor_segments`` and ``corridor_nodes``.

    The library's own writer (``CorridorGraph.save``) does the writing. An existing file is never replaced
    silently: it is refused (``InvalidInputError``) unless ``overwrite`` is true, and then it is replaced in one
    step (the new file is written beside it first, so a failed write leaves the old file alone). The folder is
    created when it is missing. Returns the path written. :func:`save_problems` gives the same refusals as
    ``(key, message)`` pairs, for a card that wants to show a notice instead of an exception.
    """
    problems = save_problems(graph, path, overwrite=overwrite)
    if problems:
        raise InvalidInputError("\n".join(message for _, message in problems))
    target = Path(str(path).strip())
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f"{target.stem}.{uuid.uuid4().hex[:8]}.tmp.gpkg")
    try:
        graph.save(str(scratch))
        os.replace(scratch, target)
    except PermissionError as exc:
        raise InvalidInputError(f"'{target.name}' cannot be replaced: it is probably open in another program. "
                                f"Close it or choose another file name.") from exc
    finally:
        if scratch.exists():
            scratch.unlink()
    return str(target)

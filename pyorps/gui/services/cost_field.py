"""
Cost-field service of the GUI (round 14, package P8).

A *cost field* is the library's ``CostField``: the accumulated least cost from ONE fixed point to every cell of
the search window. This module turns several fixed points into one map: the cells' cost summed over the points
("where would one station connecting all of them be cheapest?") or the lowest of the points' costs ("which point
is nearest"). It is a pure service: no Dash, no component ids, no ``ProjectState`` access in the part that runs in
a worker thread. The card (``sections/cost_field.py``) and its callbacks build on the public names below.

Settings dict (``state.gui_settings["cost_field"]``, mirrored from the card's controls; all four keys)::

    {"origins":   "sources" | "all",                      # which fixed points: the sources, or every control point
     "combine":   "sum" | "min",                          # sum over the points, or the lowest of them
     "algorithm": "auto" | "dijkstra" | "delta-stepping", # kernel (see below)
     "threads":   0 .. max_threads()}                     # worker threads (0 = automatic, whole numbers)

The routing controls the card reuses (backend, neighbourhood, search buffer, whole-raster switch, forbidden cells
are barriers) are NOT part of this dict: they stay in ``state.routing_settings`` and arrive as arguments.

Kernels. ``auto`` picks the fastest full-field kernel (delta-stepping on the CPU, the GPU kernel on
``raster_gpu``); ``dijkstra`` is the serial solver; ``delta-stepping`` names the parallel CPU kernel explicitly.
``raster_gpu`` and ``raster_fim`` choose their kernel from the backend, so the setting has no effect there. With a
terrain model (DEM) only the Dijkstra solver applies the slope terms, so an explicit ``delta-stepping`` and both
GPU backends are refused as a problem (``auto`` resolves to Dijkstra on cython).

Where a run can and cannot stop. :func:`compute_cost_field` solves one fixed point after another, each with ONE
library call that has no cancel hook, so Stop is honoured only BETWEEN the steps: before the raster window is
opened, before each fixed point, and before the file is written. A point in flight (the long part, seconds to
minutes on a large window) always finishes; a stopped run writes no file and returns nothing usable (it raises
:class:`~pyorps.gui.services.jobs.JobCancelled`). Writing the GeoTIFF cannot be stopped either.

Memory. One solve holds a search tree, the field array and the running result at once; the module estimates
``cells x FIELD_BYTES_PER_CELL`` BEFORE anything is allocated and refuses above :data:`MEMORY_LIMIT_MB` (GPU
backends also above :data:`GPU_MEMORY_LIMIT_MB`), with the numbers the user needs to act on.

Warnings. The worker thread does not capture Python warnings (``warnings.catch_warnings`` is not thread-safe
against the guard of the Dash threads). Everything the user must see is therefore either refused up front (a fixed
point on a forbidden cell would be silently moved by the library) or derived from facts the run observed, and
returned in ``result["notices"]``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pyorps.core.exceptions import InvalidInputError
from pyorps.raster.handler import RasterHandler

from . import errors, geo, routing

# The run half lives in ``cost_field_run``; its public names stay importable from here (same objects).
from .cost_field_run import (  # noqa: F401
    _TAB,
    FIELD_BACKENDS,
    NODATA,
    OBJECTIVE_UNIT,
    OBJECTIVE_UNIT_NOTE,
    UNIT,
    UNIT_NOTE,
    CostFieldRefused,
    Problem,
    _as_geometry,
    _check_buffer,
    _check_dtype,
    _check_neighborhood,
    _checkpoint,
    _has_objective,
    _is_blank,
    _raster_point_problems,
    _run_fields,
    _step,
    best_text,
    finalize_cost_field,
    is_cost_field_layer,
    raster_info,
    study_area_geometry,
    summary_text,
    unit_for,
)

# ------------------------------------------------------------------ names of the settings
GPU_FIELD_BACKENDS = ("raster_gpu", "raster_fim")

ORIGIN_CHOICES = ("sources", "all")
COMBINE_CHOICES = ("sum", "min")
ALGORITHM_CHOICES = ("auto", "dijkstra", "delta-stepping")

COST_FIELD_DEFAULTS: dict[str, Any] = {
    "origins": "sources", "combine": "sum", "algorithm": "auto", "threads": 0}
COST_FIELD_KEYS = tuple(COST_FIELD_DEFAULTS)

#: the options of the card's controls (label, value), so the card does not retype them
ORIGIN_OPTIONS = [{"label": "Sources", "value": "sources"},
                  {"label": "All control points", "value": "all"}]
COMBINE_OPTIONS = [{"label": "Sum over the points (siting)", "value": "sum"},
                   {"label": "Lowest of the points (catchment)", "value": "min"}]
ALGORITHM_OPTIONS = [{"label": "auto (fastest full field)", "value": "auto"},
                     {"label": "dijkstra", "value": "dijkstra"}]

#: what a problem's key stands for (the labels of the card's controls)
FIELD_LABELS = {
    "origins": "Fixed points", "combine": "Combine", "algorithm": "Kernel", "threads": "Threads",
    "graph_api": "Backend", "neighborhood": "Neighbourhood k", "search_buffer_m": "Search buffer",
    "raster": "Cost raster", "memory": "Memory"}


# ------------------------------------------------------------------ limits (monkeypatch these names in tests)
#: bytes per window cell at the peak of a run (the second fixed point being settled: search tree, field array and
#: its copy, running result, the window raster and the kernel's own copies). The plan assumed 21; the peak
#: resident memory measured on a 6.25 M-cell window was 33 B/cell with Dijkstra and 31.5 B/cell with
#: delta-stepping (2 fixed points, r2, 4 threads), so 36 is used (a margin of about 10 %).
FIELD_BYTES_PER_CELL = 36
#: the run is refused above this (decimal MB), warned about above ``MEMORY_WARN_MB`` [A limit of the plan]
MEMORY_LIMIT_MB = 4000.0
MEMORY_WARN_MB = 1000.0
#: device memory of the GPU backends: the library names ~18 B/cell for a live session; the machine rule keeps a
#: chunk under 2 GB (a heavy job once caused a thermal power-off) [A cap]
GPU_BYTES_PER_CELL = 18
GPU_MEMORY_LIMIT_MB = 2000.0
#: one full search per fixed point: more points than this are refused up front [A cap]
MAX_ORIGINS = 100


# ------------------------------------------------------------------ small helpers
def label_for(key: str) -> str:
    """The label of a settings key as the card shows it (the key itself when unknown)."""
    return FIELD_LABELS.get(key, key)


def max_threads() -> int:
    """Largest thread count the card accepts: one core less than the machine has.

    The delta-stepping kernel synchronises every bucket on a spin barrier; asking for ALL cores makes it
    hundreds of times slower (library documentation), so the core count itself is never offered.
    """
    return max(1, (os.cpu_count() or 2) - 1)


def _fmt_mb(mb: float) -> str:
    """Decimal MB as a person reads it: whole megabytes from 10 up, three significant digits below."""
    return f"{mb:,.0f}" if mb >= 10 else f"{mb:.3g}"


def _whole_number(value: Any) -> tuple[int | None, str | None]:
    """``(int, None)`` for a whole number (an integral float is fine), else ``(None, why)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None, "is not a number"
    number = float(value)
    if not math.isfinite(number):
        return None, "is not a finite number"
    if number != int(number):
        return None, "is not a whole number"
    return int(number), None


# ------------------------------------------------------------------ settings
def normalise_cost_field_settings(raw: Any) -> dict[str, Any]:
    """The cost-field settings of a project file, as the controls take them.

    A project without them (an older file) gives ``{}``: the card keeps its own defaults. Otherwise every key is
    present: a missing or unusable value falls back to its default (this is for RESTORING a saved dict, so it
    never refuses; :func:`validate_cost_field_settings` is the strict check of a run).
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    out = dict(COST_FIELD_DEFAULTS)
    for key, choices in (("origins", ORIGIN_CHOICES), ("combine", COMBINE_CHOICES),
                         ("algorithm", ALGORITHM_CHOICES)):
        if raw.get(key) in choices:
            out[key] = raw[key]
    threads, why = _whole_number(raw.get("threads"))
    if why is None and 0 <= threads <= max_threads():
        out["threads"] = threads
    return out


def _settings_values(raw: Any) -> dict[str, Any]:
    """The four settings of the card: the defaults, overridden by every known key that is not blank."""
    values = dict(COST_FIELD_DEFAULTS)
    if isinstance(raw, dict):
        values.update({k: v for k, v in raw.items() if k in COST_FIELD_KEYS and not _is_blank(v)})
    return values


def _choice_sets() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The settings that take one of a few options, with their allowed values."""
    return (("origins", ORIGIN_CHOICES), ("combine", COMBINE_CHOICES), ("algorithm", ALGORITHM_CHOICES))


def _choice_problems(values: dict[str, Any]) -> list[Problem]:
    problems: list[Problem] = []
    for key, choices in _choice_sets():
        if values[key] not in choices:
            problems.append(Problem(key, f"{label_for(key)} ({key}): {values[key]!r} is not an option. "
                                         f"Allowed: {', '.join(choices)}."))
    return problems


def _threads_problems(value: Any) -> list[Problem]:
    threads, why = _whole_number(value)
    limit = max_threads()
    allowed = f"Allowed: whole numbers from 0 to {limit} threads (0 = automatic)."
    if why is not None:
        return [Problem("threads", f"Threads (threads): {value!r} {why}. {allowed}")]
    if not 0 <= threads <= limit:
        return [Problem("threads", f"Threads (threads): {threads} threads is outside the range. {allowed}")]
    return []


def validate_cost_field_settings(raw: Any) -> list[Problem]:
    """Problems of the four settings as typed into the card, named by setting, value, allowed range and unit.

    A key that is missing or blank counts as its default. Returns ``[]`` when everything is usable.
    """
    values = _settings_values(raw)
    return _choice_problems(values) + _threads_problems(values["threads"])


def problem_notice(problems: list[Problem], *, controls: dict[str, str] | None = None, tab: str = _TAB,
                   impact: str = "No cost field was computed.") -> dict | None:
    """The first problem as a notice dict (``Notice.to_dict()``), or None without problems.

    ``controls`` maps a problem key to the component id the "Go fix" button should scroll to (the caller owns the
    ids, this service does not know them). Further problems are listed in the notice's details.
    """
    if not problems:
        return None
    key, message = problems[0]
    label = label_for(key)
    rest = "\n".join(m for _k, m in problems[1:])
    if key == "memory":
        return errors.Notice(
            severity="error", title="The cost field would not fit into memory", meaning=message, impact=impact,
            fix="Follow the numbers in the message: a smaller search buffer, a coarser raster or fewer points.",
            details=rest, focus_id=tab, focus_control=(controls or {}).get(key)).to_dict()
    return errors.Notice(
        severity="warning", title=f"{label}: value not usable", meaning=message, impact=impact,
        fix=f"Correct the {label} setting and try again.", details=rest, focus_id=tab,
        focus_control=(controls or {}).get(key)).to_dict()


# ------------------------------------------------------------------ the fixed points
def origins_from_draft(draft: Any, which: str = "sources", *, crs: Any = None) -> list[tuple[float, float]]:
    """The fixed points of the route draft as ``[(x, y), ...]``, without exact duplicates, in draft order.

    ``draft`` is the page's route draft (``{"sources": [...], "targets": [...], "waypoints": [...]}``, each point
    ``{"x", "y", "lat", "lng", "name"}``). ``which`` is ``"sources"`` or ``"all"`` (sources, then targets, then
    waypoints). With ``crs`` (the raster's CRS) a point that carries ``lat``/``lng`` is converted from those
    through ``services.geo`` (so a raster picked after the click still gets its own coordinates); otherwise, and for
    a point without them, the stored ``x``/``y`` are used. Raises ``ValueError`` for another ``which``.
    """
    if which not in ORIGIN_CHOICES:
        raise ValueError(f"which must be one of {', '.join(ORIGIN_CHOICES)}; got {which!r}")
    draft = draft if isinstance(draft, dict) else {}
    kinds = ("sources",) if which == "sources" else ("sources", "targets", "waypoints")
    return _unique_points(draft, kinds, crs)


def _unique_points(draft: dict, kinds: tuple[str, ...], crs: Any) -> list[tuple[float, float]]:
    """The points of the draft's lists ``kinds`` in order, each exact duplicate once."""
    points: list[tuple[float, float]] = []
    for kind in kinds:
        for point in draft.get(kind) or []:
            xy = _point_xy(point, crs)
            if xy is not None and xy not in points:
                points.append(xy)
    return points


def _point_xy(point: Any, crs: Any) -> tuple[float, float] | None:
    if not isinstance(point, dict):
        return None
    try:
        if crs is not None and point.get("lat") is not None and point.get("lng") is not None:
            return geo.point_wgs84_to_crs(float(point["lat"]), float(point["lng"]), crs)
        if point.get("x") is not None and point.get("y") is not None:
            return float(point["x"]), float(point["y"])
    except (TypeError, ValueError):
        return None
    return None


def new_out_path(work_dir: Any) -> Path:
    """A fresh file name for the result GeoTIFF inside ``work_dir`` (``state.work_dir``); nothing is created."""
    import uuid

    return Path(work_dir) / f"cost_field_{uuid.uuid4().hex}.tif"


# ------------------------------------------------------------------ backend and raster checks
def check_backend(graph_api: Any, algorithm: str = "auto", *, has_dem: bool = False) -> list[Problem]:
    """Problems of the backend (and, with a DEM, of the kernel) for a field."""
    name = str(graph_api or "").strip().lower()
    if name not in routing.BACKENDS:
        return [Problem("graph_api", f"Backend (graph_api): {graph_api!r} is not a known backend. " +
                                     f"Allowed for a cost field: {', '.join(FIELD_BACKENDS)}.")]
    if name not in FIELD_BACKENDS:
        return [Problem("graph_api", (
            f"Backend (graph_api): {name!r} cannot compute a cost field. The graph libraries use one-way half step " +
            "sets, so a field rooted at a fixed point would not be symmetric. " +
            f"Allowed: {', '.join(FIELD_BACKENDS)}."))]
    if not routing.backend_available(name):
        return [Problem("graph_api", f"Backend (graph_api): {name!r} is not installed. {routing.BACKENDS[name].hint}")]
    return _dem_problems(name, algorithm, has_dem)


def _dem_problems(name: str, algorithm: str, has_dem: bool) -> list[Problem]:
    """Problems of a backend and kernel that cannot apply the slope terms of a terrain model."""
    if not has_dem:
        return []
    if name != "cython":
        return [Problem("graph_api", (
            f"Backend (graph_api): {name!r} cannot compute a cost field with a terrain model (DEM): its kernel does " +
            "not apply the slope terms, so it would price flat terrain. Allowed with a DEM: cython."))]
    if algorithm == "delta-stepping":
        return [Problem("algorithm", (
            "Kernel (algorithm): 'delta-stepping' cannot compute a cost field with a terrain model (DEM): only " +
            "the Dijkstra solver applies the slope terms. Allowed with a DEM: auto, dijkstra."))]
    return []


def _point_list_problems(points: list[tuple[float, float]]) -> list[Problem]:
    """The problems of the list itself: none given, too many, a point that is not finite."""
    if not points:
        return [Problem("origins", ("Fixed points (origins): none given. Allowed: 1 to " +
                                    f"{MAX_ORIGINS} points in the raster's coordinate system (m). Place a source " +
                                    "on the map, or choose 'All control points'."))]
    if len(points) > MAX_ORIGINS:
        return [Problem("origins", (f"Fixed points (origins): {len(points)} points. Allowed: 1 to {MAX_ORIGINS} " +
                                    "points, because every point needs one full search of the window."))]
    bad = [(i, p) for i, p in enumerate(points) if not all(math.isfinite(c) for c in p)]
    if bad:
        return [Problem("origins", f"Fixed points (origins): point {bad[0][0] + 1} is not finite ({bad[0][1]}). " +
                                   "Allowed: finite (x, y) numbers in the raster's coordinate system (m).")]
    return []


def check_origins(raster_path: Any, origins: Any, *, ignore_max_cost: bool = True,
                  check_forbidden: bool = True) -> list[Problem]:
    """Problems of the fixed points: none given, too many, not finite, outside the raster, on a forbidden cell.

    A fixed point on a forbidden cell (value 65535 with "forbidden cells are barriers" on) is REFUSED here: the
    library would silently move it to the nearest routable cell and only warn, and a siting map must start where
    the user put the point. The one cell value is read per point; no raster is loaded.
    """
    points = [(float(x), float(y)) for x, y in (origins or [])]
    problems = _point_list_problems(points)
    if problems:
        return problems
    return _raster_point_problems(raster_path, points, ignore_max_cost=ignore_max_cost,
                                  check_forbidden=check_forbidden)


# ------------------------------------------------------------------ memory
def resolve_buffer_m(raster_path: Any, origins: Any, search_buffer_m: Any = None, *,
                     full_window: bool = False) -> float:
    """The explicit search buffer of a run, in metres: the whole raster, the user's value, or the automatic one.

    Blank or 0 means automatic, exactly as for routes: the larger of 1000 m and 1.5 times the largest distance
    between the fixed points.
    """
    if full_window:
        return routing.whole_raster_buffer(raster_path)
    if not _is_blank(search_buffer_m) and float(search_buffer_m) > 0:
        return float(search_buffer_m)
    points = [(float(x), float(y)) for x, y in origins]
    return routing.default_search_buffer(points, points)


def _hull_bounds(origins: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in origins]
    ys = [p[1] for p in origins]
    return min(xs), min(ys), max(xs), max(ys)


def _window_shape(info: dict, hull: tuple[float, float, float, float], buffer_m: float) -> tuple[int, int]:
    """``(rows, cols)`` of the search window: the buffered hull of the points, clipped to the raster.

    The same arithmetic as the library's ``RasterHandler.window_from_bounds`` (the static method is used).
    """
    minx, miny, maxx, maxy = hull
    window = RasterHandler.window_from_bounds(
        (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m), info["transform"],
        (info["height"], info["width"]))
    return max(0, int(window.height)), max(0, int(window.width))


def _max_buffer_m(info: dict, hull: tuple[float, float, float, float], max_cells: int) -> float | None:
    """The largest buffer whose window has at most ``max_cells`` cells; None when even buffer 0 is too big."""
    def cells(buffer_m: float) -> int:
        rows, cols = _window_shape(info, hull, buffer_m)
        return rows * cols

    if cells(0.0) > max_cells:
        return None
    low = 0.0
    high = math.hypot(info["bounds"][2] - info["bounds"][0], info["bounds"][3] - info["bounds"][1])
    if cells(high) <= max_cells:
        return high
    for _ in range(60):
        mid = (low + high) / 2.0
        if cells(mid) <= max_cells:
            low = mid
        else:
            high = mid
    return low


def estimate(raster_path: Any, origins: Any, *, buffer_m: float | None = None, full_window: bool = False,
             graph_api: str = "cython") -> dict:
    """Size and memory of the run, from the raster header alone (nothing is loaded or allocated).

    ``buffer_m`` is the explicit search buffer; None or 0 means automatic, ``full_window`` the whole raster
    (see :func:`resolve_buffer_m`). Returns a dict: ``rows``, ``cols``, ``cells`` (the search window),
    ``raster_cells``, ``resolution_m``, ``buffer_m``, ``bytes_per_cell`` and ``mb`` (decimal MB), ``limit_mb``,
    ``warn_mb``, ``max_cells`` (what fits under the limit), ``verdict`` ("ok", "warn" or "refuse"), and for the GPU
    backends ``device_mb``, ``device_limit_mb`` and ``device_max_cells``. The limits are read from the module
    constants at call time.
    """
    points = [(float(x), float(y)) for x, y in origins]
    info = raster_info(raster_path)
    buffer_m = resolve_buffer_m(raster_path, points, buffer_m, full_window=full_window)
    rows, cols = _window_shape(info, _hull_bounds(points), buffer_m)
    cells = rows * cols
    mb = cells * FIELD_BYTES_PER_CELL / 1e6
    out = {"rows": rows, "cols": cols, "cells": cells, "raster_cells": info["height"] * info["width"],
           "resolution_m": info["resolution_m"], "buffer_m": float(buffer_m), "dtype": info["dtype"],
           "bytes_per_cell": FIELD_BYTES_PER_CELL, "mb": mb, "limit_mb": MEMORY_LIMIT_MB, "warn_mb": MEMORY_WARN_MB,
           "max_cells": int(MEMORY_LIMIT_MB * 1e6 // FIELD_BYTES_PER_CELL), "n_origins": len(points),
           "graph_api": graph_api}
    verdict = "refuse" if mb > MEMORY_LIMIT_MB else "warn" if mb > MEMORY_WARN_MB else "ok"
    if graph_api in GPU_FIELD_BACKENDS:
        out["device_mb"] = cells * GPU_BYTES_PER_CELL / 1e6
        out["device_limit_mb"] = GPU_MEMORY_LIMIT_MB
        out["device_max_cells"] = int(GPU_MEMORY_LIMIT_MB * 1e6 // GPU_BYTES_PER_CELL)
        if out["device_mb"] > GPU_MEMORY_LIMIT_MB:
            verdict = "refuse"
    out["verdict"] = verdict
    return out


def _memory_cause(est: dict) -> tuple[str, int]:
    """``(what, max_cells)``: the sentence naming the limit that was hit, and the cell count that limit allows."""
    gpu_over = "device_mb" in est and est["device_mb"] > est["device_limit_mb"]
    if gpu_over and est["mb"] <= est["limit_mb"]:
        what = (f"on the graphics card it needs about {_fmt_mb(est['device_mb'])} MB ({GPU_BYTES_PER_CELL} bytes per " +
                f"cell); the limit for one run is {_fmt_mb(est['device_limit_mb'])} MB (about " +
                f"{est['device_max_cells']:,} cells)")
        return what, est["device_max_cells"]
    what = (f"it needs about {_fmt_mb(est['mb'])} MB ({est['bytes_per_cell']} bytes per cell for the search, the " +
            f"field and the running result); the limit is {_fmt_mb(est['limit_mb'])} MB (about " +
            f"{est['max_cells']:,} cells)")
    return what, est["max_cells"]


def _memory_advice(est: dict, info: dict, hull: Any, max_cells: int, full_window: bool) -> str:
    """What to change so the window fits: a smaller search buffer when that helps, else a coarser raster."""
    limit_buffer = _max_buffer_m(info, hull, max_cells)
    if limit_buffer is not None and 1.0 <= limit_buffer < est["buffer_m"]:
        step = "Switch off the whole-raster option and set" if full_window else "Set"
        return (f"{step} the search buffer to at most {math.floor(limit_buffer):,} m (the largest window that " +
                "fits).")
    factor = math.sqrt(est["cells"] / max_cells)
    return ("Even the fixed points alone need that much: use a coarser cost raster (at least " +
            f"{est['resolution_m'] * factor:,.1f} m per cell) or fewer, closer fixed points.")


def memory_problems(est: dict, raster_path: Any, origins: Any, *, full_window: bool = False) -> list[Problem]:
    """The refusal of an estimate over the limit, with numbers the user can act on; ``[]`` for ok and warn."""
    if est["verdict"] != "refuse":
        return []
    info = raster_info(raster_path)
    hull = _hull_bounds([(float(x), float(y)) for x, y in origins])
    what, max_cells = _memory_cause(est)
    head = (f"Memory (cost field): the search window would be {est['rows']:,} x {est['cols']:,} = " +
            f"{est['cells']:,} cells at {est['resolution_m']:g} m per cell, and {what}. ")
    return [Problem("memory", head + _memory_advice(est, info, hull, max_cells, full_window))]


# ------------------------------------------------------------------ the plan of a run (main thread, cheap)
@dataclass
class RunPlan:
    """What :func:`plan_run` decided: refused (``problems``) or ready (``kwargs``).

    ``notices`` are soft warnings to show when the job starts (memory above the warning level, the kernel the
    terrain model forces). ``kwargs`` are the arguments of :func:`compute_cost_field` except ``cancel`` and
    ``progress`` (only set when ``ok``).
    """

    problems: list[Problem] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    estimate: dict | None = None
    notices: list[dict] = field(default_factory=list)
    kwargs: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def _has_dem(finder_options: dict | None) -> bool:
    return bool(finder_options) and finder_options.get("dem") is not None


def _objective_scale(finder: Any, finder_options: dict | None) -> float | None:
    """The quantisation scale of the objective's search raster (``weights ~ objective * scale``), None without one.

    With an objective the search runs on a 16-bit surface that holds the weighted objective times a scale (or on
    the untouched cost raster, scale 1 / cost weight, when only the cost is weighted); the library says to divide
    the kernel distances by it to get the objective itself. 1.0 when the library reports no scale.
    """
    if not _has_objective(finder_options):
        return None
    combined = getattr(finder, "_combine_result", None)  # pylint: disable=protected-access  # the one public record
    scale = float(getattr(combined, "scale", 1.0) or 1.0)
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def _collect_problems(raster_path: Any, origins: list[tuple[float, float]], *, combine: str, algorithm: str,
                      num_threads: Any, neighborhood: Any, ignore_max_cost: bool, search_buffer_m: Any,
                      graph_api: Any, full_window: bool,
                      finder_options: dict | None) -> tuple[list[Problem], dict | None]:
    # lizard forgives: one keyword per routing control, called by keyword from plan_run and compute_cost_field only
    """Every problem found before anything is allocated, and the memory estimate when one could be made."""
    problems = validate_cost_field_settings(
        {"combine": combine, "algorithm": algorithm, "threads": num_threads})
    problems += check_backend(graph_api, algorithm if algorithm in ALGORITHM_CHOICES else "auto",
                              has_dem=_has_dem(finder_options))
    problems += _check_buffer(search_buffer_m) + _check_neighborhood(neighborhood)
    skip_cells = bool(finder_options and finder_options.get("objective") is not None)
    origin_problems = check_origins(raster_path, origins, ignore_max_cost=ignore_max_cost,
                                    check_forbidden=not skip_cells)
    problems += origin_problems
    if origin_problems or any(k == "search_buffer_m" for k, _m in problems):
        return problems, None
    info = raster_info(raster_path)
    problems += _check_dtype(info, finder_options)
    est = estimate(raster_path, origins, buffer_m=search_buffer_m, full_window=full_window, graph_api=str(graph_api))
    problems += memory_problems(est, raster_path, origins, full_window=full_window)
    return problems, est


def _soft_notices(est: dict | None, graph_api: str, algorithm: str, finder_options: dict | None) -> list[dict]:
    notices = []
    if est is not None and est["verdict"] == "warn":
        notices.append(errors.Notice(
            severity="warning", title="This cost field needs a lot of memory",
            meaning=(f"The search window has {est['cells']:,} cells and needs about {_fmt_mb(est['mb'])} MB " +
                     f"({est['bytes_per_cell']} bytes per cell); the warning level is {_fmt_mb(est['warn_mb'])} MB, " +
                     f"the limit {_fmt_mb(est['limit_mb'])} MB."),
            impact="Other programs may get slow while it runs; every fixed point is one full search.",
            fix="A smaller search buffer or a coarser raster makes it cheaper.", focus_id=_TAB).to_dict())
    if _has_dem(finder_options) and algorithm == "auto":
        notices.append(errors.Notice(
            severity="info", title="With a terrain model the cost field uses the Dijkstra solver",
            meaning="Only the Dijkstra solver applies the DEM's slope terms; the parallel kernels would price flat "
                    "terrain.",
            impact="The field is computed one core at a time and takes longer.", fix="", focus_id=_TAB).to_dict())
    if graph_api == "raster_fim":
        notices.append(errors.Notice(
            severity="info", title="raster_fim computes a continuous field",
            meaning="The eikonal backend returns the continuous cost it minimised, not the cost of a cell-to-cell "
                    "route.",
            impact="Its values differ from (usually lie below) the route costs of the other backends; "
                   "do not compare them.",
            fix="", focus_id=_TAB).to_dict())
    return notices


def plan_run(raster_path: Any, raw_settings: Any, origins: Any, *, graph_api: Any = "cython",
             neighborhood: Any = "r2", ignore_max_cost: bool = True, search_buffer_m: Any = None,
             full_window: bool = False, out_path: Any = None, study_area: Any = None,
             finder_options: dict | None = None) -> RunPlan:
    # lizard forgives: one keyword per routing control of the card; every caller and test passes them by keyword
    """Check everything a run needs, from the page's values, before any job starts (main thread, cheap).

    ``raw_settings`` is the dict of the card's controls (:func:`validate_cost_field_settings`); ``origins`` the
    fixed points in the raster's CRS (:func:`origins_from_draft`); ``graph_api``, ``neighborhood`` (``"r2"`` or
    k), ``ignore_max_cost``, ``search_buffer_m`` (blank/0 = automatic) and ``full_window`` are the routing
    controls; ``finder_options`` is the optional extra ``PathFinder`` keywords (``dem``, ``objective``, ...) of the
    objective card. Returns a :class:`RunPlan`: with ``problems`` when the run is refused (show
    :func:`problem_notice`), else ``kwargs`` for :func:`compute_cost_field` (for example through
    :func:`job_work`) and the soft ``notices``.
    """
    settings = _settings_values(raw_settings)
    points = [(float(x), float(y)) for x, y in (origins or [])]
    problems, est = _collect_problems(
        raster_path, points, combine=settings["combine"], algorithm=settings["algorithm"],
        num_threads=settings["threads"], neighborhood=neighborhood, ignore_max_cost=bool(ignore_max_cost),
        search_buffer_m=search_buffer_m, graph_api=graph_api, full_window=bool(full_window),
        finder_options=finder_options)
    # an invalid "origins" choice is a settings problem even though the points are already resolved
    problems = [*validate_cost_field_settings({"origins": settings["origins"]}), *problems]
    plan = RunPlan(problems=_unique(problems), settings=settings, estimate=est)
    if plan.problems:
        return plan
    api = str(graph_api).strip().lower()
    plan.notices = _soft_notices(est, api, settings["algorithm"], finder_options)
    area = _as_geometry(study_area, raster_info(raster_path)["crs"] or None)
    plan.kwargs = {
        "raster_path": str(raster_path), "origins": points, "combine": settings["combine"],
        "algorithm": settings["algorithm"], "num_threads": int(settings["threads"]),
        "neighborhood": routing.neighborhood_string(neighborhood), "ignore_max_cost": bool(ignore_max_cost),
        "search_buffer_m": est["buffer_m"], "graph_api": api, "out_path": str(out_path) if out_path else None,
        "study_area": area, "finder_options": dict(finder_options) if finder_options else None}
    return plan


def _unique(problems: list[Problem]) -> list[Problem]:
    seen: set[Problem] = set()
    out = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            out.append(Problem(*problem))
    return out


# ------------------------------------------------------------------ the computation (worker thread)


def _finder_for(raster_path: str, origins: list, *, graph_api: str, neighborhood: str, buffer_m: float,
                ignore_max_cost: bool, finder_options: dict | None):
    from pyorps import PathFinder

    finder = PathFinder(
        raster_path, source_coords=list(origins), target_coords=list(origins), graph_api=graph_api,
        neighborhood_str=neighborhood, search_space_buffer_m=buffer_m, ignore_max_cost=ignore_max_cost,
        **(finder_options or {}))
    finder.create_raster_handler()
    return finder


def compute_cost_field(raster_path: Any, origins: Any, *, combine: str = "sum", algorithm: str = "auto",
                       num_threads: int = 0, neighborhood: Any = "r2", ignore_max_cost: bool = True,
                       search_buffer_m: float | None = None, graph_api: str = "cython", out_path: Any = None,
                       study_area: Any = None, cancel: Any = None, progress: Any = None,
                       full_window: bool = False, finder_options: dict | None = None) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Compute the accumulated cost field of the fixed points and write it as a GeoTIFF; returns a result dict.

    Runs in a worker thread (it never touches ``ProjectState``): ``progress(done, total, stage)`` is called at each
    step (``total`` = number of points + 1; ``BackgroundJob.report`` fits) and ``cancel`` is a
    ``threading.Event`` (or a callable returning True) checked BEFORE the raster window is opened, before each
    fixed point and before the file is written, never inside one solve (the library kernels have no cancel hook, see
    the module docstring). A stopped run raises :class:`~pyorps.gui.services.jobs.JobCancelled` and writes no file.

    One ``PathFinder`` whose window covers all points is built; then one ``finder.cost_field(point)`` per point is
    settled, folded into one float32 array (``combine``: ``"sum"`` or ``"min"``) and closed, so the peak is one
    tree, one field and the running result. Unreachable cells and every forbidden cell of the window are nodata
    (-1) in the file. ``num_threads=0`` passes nothing (the library picks three quarters of the cores).
    ``search_buffer_m`` None or 0 is automatic, ``full_window`` the whole raster. ``finder_options`` are optional
    extra ``PathFinder`` keywords (``dem``, ``objective``, ``gradient_options``). Every check of
    :func:`plan_run` is repeated first and raises :class:`CostFieldRefused` (before any allocation).

    Result keys: ``path``, ``min``, ``max`` (over the valid cells), ``unit``, ``unit_note``, ``nodata``, ``origins``
    (list of ``[x, y]``), ``n_origins``, ``combine``, ``algorithm`` (resolved), ``algorithm_requested``,
    ``graph_api``, ``neighborhood``, ``rows``, ``cols``, ``cells``, ``valid_cells``, ``nodata_cells``,
    ``resolution_m``, ``bounds`` (left, bottom, right, top of the window), ``crs``, ``best`` (``{"x", "y", "row",
    "col", "cost"}`` or None), ``best_note``, ``estimate`` and ``notices`` (notice dicts for the poll side).
    ``forbidden_cells`` counts the window cells that are forbidden (65535) or outside the buffered search area
    (the library masks those as forbidden); ``unreachable_cells`` the other cells without a finite value;
    ``nodata_cells`` is their sum. ``objective`` is True when ``finder_options`` carried an objective: then the
    values are weighted objective units (the kernel's search-raster distances divided by the quantisation scale,
    ``objective_scale``; ``unit`` / ``unit_note`` say so), else False and ``objective_scale`` None.
    ``study_area`` is a shapely geometry in the raster CRS or a WGS84 GeoJSON dict.
    """
    points = [(float(x), float(y)) for x, y in (origins or [])]
    problems, est = _collect_problems(
        raster_path, points, combine=combine, algorithm=algorithm, num_threads=num_threads,
        neighborhood=neighborhood, ignore_max_cost=bool(ignore_max_cost), search_buffer_m=search_buffer_m,
        graph_api=graph_api, full_window=bool(full_window), finder_options=finder_options)
    if problems:
        raise CostFieldRefused(_unique(problems))
    if not out_path:
        raise InvalidInputError("compute_cost_field needs out_path, the GeoTIFF to write")
    api = str(graph_api).strip().lower()
    k_text = routing.neighborhood_string(neighborhood)
    _checkpoint(cancel, "The cost field")
    _step(progress, 0, len(points) + 1, "opening the raster window")
    finder = _finder_for(str(raster_path), points, graph_api=api, neighborhood=k_text, buffer_m=est["buffer_m"],
                         ignore_max_cost=bool(ignore_max_cost), finder_options=finder_options)
    try:
        return _run_fields(finder, points, est, api, combine=combine, algorithm=algorithm,
                           num_threads=int(num_threads), neighborhood=k_text, out_path=Path(out_path),
                           study_area=study_area, cancel=cancel, progress=progress,
                           has_dem=_has_dem(finder_options),
                           objective_scale=_objective_scale(finder, finder_options))
    finally:
        finder.release_device_resources()


# ------------------------------------------------------------------ job glue and finishing
def job_work(job: Any) -> dict:
    """The ``work(job)`` of ``jobs.start_job``: runs :func:`compute_cost_field` with ``job.params["compute"]``.

    Start it as ``jobs.start_job(state, "cost_field", cost_field.job_work, label="Cost field",
    params={"compute": plan.kwargs, ...})``. ``job.cancel`` and ``job.report`` are wired in; the result dict
    becomes ``job.result``.
    """
    return compute_cost_field(**job.params["compute"], cancel=job.cancel, progress=job.report)


def job_error_notice(error: BaseException | None) -> dict | None:
    """The notice of a failed job's exception: the field-level notice of a refusal, else the generic translation."""
    if error is None:
        return None
    if isinstance(error, CostFieldRefused):
        return problem_notice(error.problems)
    return errors.translate_exception(error).to_dict()



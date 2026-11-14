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
from typing import Any, NamedTuple

import numpy as np

from pyorps.core.exceptions import FeatureNotImplementedError, InvalidInputError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.raster.handler import RasterHandler

from . import errors, geo, jobs, routing, tiles

# ------------------------------------------------------------------ names of the settings
#: backends that can root a field: the graph libraries use half step sets, so a field rooted at a fixed point
#: would not be symmetric (library ``CostField`` refuses them)
FIELD_BACKENDS = ("cython", "raster_gpu", "raster_fim")
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

#: unit of every cost-field value: cell value x metres (EUR when the raster holds EUR per metre)
UNIT = "cost units"
UNIT_NOTE = "cell value x metres (EUR on a raster that holds EUR per metre)"

#: unit of a field computed with the objective of the Objective & terrain card: the search runs on the weighted
#: combination of the cost raster, the length term, the slope term and the metric layers (the library's
#: "objective units x m, not EUR"), so these values are not comparable with the plain cost units above
OBJECTIVE_UNIT = "objective units"
OBJECTIVE_UNIT_NOTE = ("weighted objective value x metres: the weighted sum of the cost raster, the length term, "
                       "the slope term and any extra layers, not EUR")

#: nodata of the result raster: costs are never negative, so no reachable cell can take it
NODATA = -1.0

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

_TAB = "tab-routes"


class Problem(NamedTuple):
    """One field-level problem: the key of the setting (see :data:`FIELD_LABELS`) and the full message.

    A two-tuple on purpose, the same shape as the profile form's problems.
    """

    key: str
    message: str


class CostFieldRefused(InvalidInputError):
    """Raised by :func:`compute_cost_field` when the run must not start (or a library check refused it).

    ``problems`` carries the same ``(key, message)`` list :func:`plan_run` would have returned; the message is
    their text. :func:`job_error_notice` turns it into a notice that names the field.
    """

    def __init__(self, problems: list[Problem]):
        self.problems = [Problem(*p) for p in problems]
        super().__init__(" ".join(m for _k, m in self.problems) or "The cost field cannot be computed.")


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


def _apply_transform(transform: Any, col: float, row: float) -> tuple[float, float]:
    """Map (col, row) to (x, y) with the six coefficients of an affine transform.

    Plain arithmetic instead of multiplying the transform with the point: affine 3 deprecates the ``*``
    operator for that (a warning per call, an error under ``-W error``), and its replacement ``@`` does
    not exist in affine 2.
    """
    return (transform.a * col + transform.b * row + transform.c,
            transform.d * col + transform.e * row + transform.f)


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else repr(value)


def _fmt_mb(mb: float) -> str:
    """Decimal MB as a person reads it: whole megabytes from 10 up, three significant digits below."""
    return f"{mb:,.0f}" if mb >= 10 else f"{mb:.3g}"


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


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


def validate_cost_field_settings(raw: Any) -> list[Problem]:
    """Problems of the four settings as typed into the card, named by setting, value, allowed range and unit.

    A key that is missing or blank counts as its default. Returns ``[]`` when everything is usable.
    """
    values = dict(COST_FIELD_DEFAULTS)
    if isinstance(raw, dict):
        values.update({k: v for k, v in raw.items() if k in COST_FIELD_KEYS and not _is_blank(v)})
    problems: list[Problem] = []
    for key, choices in (("origins", ORIGIN_CHOICES), ("combine", COMBINE_CHOICES),
                         ("algorithm", ALGORITHM_CHOICES)):
        if values[key] not in choices:
            problems.append(Problem(key, f"{label_for(key)} ({key}): {values[key]!r} is not an option. "
                                         f"Allowed: {', '.join(choices)}."))
    threads, why = _whole_number(values["threads"])
    limit = max_threads()
    allowed = f"Allowed: whole numbers from 0 to {limit} threads (0 = automatic)."
    if why is not None:
        problems.append(Problem("threads", f"Threads (threads): {values['threads']!r} {why}. {allowed}"))
    elif not 0 <= threads <= limit:
        problems.append(Problem("threads", f"Threads (threads): {threads} threads is outside the range. {allowed}"))
    return problems


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


def study_area_geometry(study_area_geojson: Any, crs: Any):
    """The drawn study area (``state.study_area``, a WGS84 GeoJSON geometry) as a shapely geometry in ``crs``.

    Returns None without a study area. :func:`plan_run` and :func:`compute_cost_field` accept the GeoJSON dict
    itself (and convert it with the raster's CRS); a snapshot taken in the callback is all the worker needs, it
    never sees ``ProjectState``.
    """
    if not study_area_geojson:
        return None
    geom = geo.geojson_feature_geometry(study_area_geojson)
    return geo.geometry_to_crs(geom, geo.WGS84, str(crs)) if crs is not None else geom


def _as_geometry(study_area: Any, crs: Any):
    """``study_area`` as a shapely geometry in the raster CRS: a GeoJSON dict is converted, a geometry kept."""
    if isinstance(study_area, dict):
        return study_area_geometry(study_area, crs)
    return study_area or None


def new_out_path(work_dir: Any) -> Path:
    """A fresh file name for the result GeoTIFF inside ``work_dir`` (``state.work_dir``); nothing is created."""
    import uuid

    return Path(work_dir) / f"cost_field_{uuid.uuid4().hex}.tif"


# ------------------------------------------------------------------ backend and raster checks
def check_backend(graph_api: Any, algorithm: str = "auto", *, has_dem: bool = False) -> list[Problem]:
    """Problems of the backend (and, with a DEM, of the kernel) for a field."""
    name = str(graph_api or "").strip().lower()
    if name not in routing.BACKENDS:
        return [Problem("graph_api", f"Backend (graph_api): {graph_api!r} is not a known backend. "
                                     f"Allowed for a cost field: {', '.join(FIELD_BACKENDS)}.")]
    if name not in FIELD_BACKENDS:
        return [Problem("graph_api", (
            f"Backend (graph_api): {name!r} cannot compute a cost field. The graph libraries use one-way half step "
            f"sets, so a field rooted at a fixed point would not be symmetric. "
            f"Allowed: {', '.join(FIELD_BACKENDS)}."))]
    if not routing.backend_available(name):
        return [Problem("graph_api", f"Backend (graph_api): {name!r} is not installed. {routing.BACKENDS[name].hint}")]
    if has_dem and name != "cython":
        return [Problem("graph_api", (
            f"Backend (graph_api): {name!r} cannot compute a cost field with a terrain model (DEM): its kernel does "
            f"not apply the slope terms, so it would price flat terrain. Allowed with a DEM: cython."))]
    if has_dem and algorithm == "delta-stepping":
        return [Problem("algorithm", (
            "Kernel (algorithm): 'delta-stepping' cannot compute a cost field with a terrain model (DEM): only "
            "the Dijkstra solver applies the slope terms. Allowed with a DEM: auto, dijkstra."))]
    return []


def raster_info(raster_path: Any) -> dict:
    """Header facts of the cost raster, read without loading any cell.

    Raises ``FileNotFoundError`` (plain message) when the file is gone.
    """
    import rasterio

    if not raster_path or not os.path.exists(str(raster_path)):
        raise FileNotFoundError(f"the raster file '{raster_path}' no longer exists")
    with rasterio.open(raster_path) as src:
        return {"bounds": (float(src.bounds.left), float(src.bounds.bottom),
                           float(src.bounds.right), float(src.bounds.top)),
                "transform": src.transform, "height": int(src.height), "width": int(src.width),
                "dtype": str(src.dtypes[0]), "crs": str(src.crs) if src.crs else "",
                "resolution_m": float(abs(src.transform.a)) or 1.0}


def _origin_cells(raster_path: Any, info: dict, origins: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """``(index, row, col)`` of the origins inside the raster, with the cell the library snaps them to."""
    from rasterio.transform import rowcol

    cells = []
    for i, (x, y) in enumerate(origins):
        left, bottom, right, top = info["bounds"]
        if left <= x <= right and bottom <= y <= top:
            row, col = rowcol(info["transform"], x, y)
            if 0 <= row < info["height"] and 0 <= col < info["width"]:
                cells.append((i, int(row), int(col)))
    return cells


def check_origins(raster_path: Any, origins: Any, *, ignore_max_cost: bool = True,
                  check_forbidden: bool = True) -> list[Problem]:
    """Problems of the fixed points: none given, too many, not finite, outside the raster, on a forbidden cell.

    A fixed point on a forbidden cell (value 65535 with "forbidden cells are barriers" on) is REFUSED here: the
    library would silently move it to the nearest routable cell and only warn, and a siting map must start where
    the user put the point. The one cell value is read per point; no raster is loaded.
    """
    points = [(float(x), float(y)) for x, y in (origins or [])]
    if not points:
        return [Problem("origins", ("Fixed points (origins): none given. Allowed: 1 to "
                                    f"{MAX_ORIGINS} points in the raster's coordinate system (m). Place a source "
                                    "on the map, or choose 'All control points'."))]
    if len(points) > MAX_ORIGINS:
        return [Problem("origins", (f"Fixed points (origins): {len(points)} points. Allowed: 1 to {MAX_ORIGINS} "
                                    "points, because every point needs one full search of the window."))]
    bad = [(i, p) for i, p in enumerate(points) if not all(math.isfinite(c) for c in p)]
    if bad:
        return [Problem("origins", f"Fixed points (origins): point {bad[0][0] + 1} is not finite ({bad[0][1]}). "
                                   "Allowed: finite (x, y) numbers in the raster's coordinate system (m).")]
    try:
        info = raster_info(raster_path)
    except FileNotFoundError as exc:
        return [Problem("raster", f"Cost raster (raster): {exc}. Rebuild the cost raster or load the .tif again.")]
    except Exception as exc:  # pylint: disable=broad-exception-caught  # an unreadable file is a problem, not a crash
        return [Problem("raster", f"Cost raster (raster): '{raster_path}' cannot be read ({exc}).")]
    inside = {i for i, _r, _c in _origin_cells(raster_path, info, points)}
    outside = [i for i in range(len(points)) if i not in inside]
    if outside:
        left, bottom, right, top = info["bounds"]
        listed = "; ".join(f"point {i + 1} at ({points[i][0]:,.1f}, {points[i][1]:,.1f})" for i in outside[:5])
        return [Problem("origins", (
            f"Fixed points (origins): {len(outside)} of {len(points)} points are outside the raster: {listed}. "
            f"Allowed: x from {left:,.1f} to {right:,.1f} m and y from {bottom:,.1f} to {top:,.1f} m. "
            "Move the points inside the raster."))]
    if ignore_max_cost and check_forbidden and info["dtype"] == "uint16":
        return _forbidden_problems(raster_path, info, points)
    return []


def _forbidden_problems(raster_path: Any, info: dict, points: list[tuple[float, float]]) -> list[Problem]:
    import rasterio
    from rasterio.windows import Window

    on_forbidden = []
    with rasterio.open(raster_path) as src:
        for i, row, col in _origin_cells(raster_path, info, points):
            value = int(src.read(1, window=Window(col, row, 1, 1))[0, 0])
            if value == IMPASSABLE_CELL_COST:
                on_forbidden.append(i)
    if not on_forbidden:
        return []
    listed = "; ".join(f"point {i + 1} at ({points[i][0]:,.1f}, {points[i][1]:,.1f})" for i in on_forbidden[:5])
    return [Problem("origins", (
        f"Fixed points (origins): {len(on_forbidden)} of {len(points)} points stand on a forbidden cell "
        f"(value {IMPASSABLE_CELL_COST}): {listed}. With 'Forbidden cells are barriers' on, a field cannot start "
        "there. Move the point to a passable cell, or switch the barrier option off."))]


def _check_buffer(search_buffer_m: Any) -> list[Problem]:
    if _is_blank(search_buffer_m):
        return []
    allowed = "Allowed: 0 (automatic) or a positive number of metres (m)."
    try:
        value = float(search_buffer_m)
    except (TypeError, ValueError):
        return [Problem("search_buffer_m", f"Search buffer (search_buffer_m): {search_buffer_m!r} is not a number. "
                                           f"{allowed}")]
    if not math.isfinite(value) or value < 0:
        return [Problem("search_buffer_m", f"Search buffer (search_buffer_m): {_fmt(value)} m is not usable. "
                                           f"{allowed}")]
    return []


def _check_neighborhood(neighborhood: Any) -> list[Problem]:
    try:
        k = int(routing.neighborhood_string(neighborhood).lstrip("r"))
    except (TypeError, ValueError):
        k = -1
    if routing.NEIGHBORHOOD_MIN <= k <= routing.NEIGHBORHOOD_MAX:
        return []
    return [Problem("neighborhood", (
        f"Neighbourhood k (neighborhood): {neighborhood!r} is not usable. Allowed: whole numbers from "
        f"{routing.NEIGHBORHOOD_MIN} to {routing.NEIGHBORHOOD_MAX}."))]


def _check_dtype(info: dict, finder_options: dict | None) -> list[Problem]:
    if finder_options and finder_options.get("objective") is not None:
        return []                                    # the objective pipeline builds its own search raster
    if info["dtype"] != "uint16":
        return [Problem("raster", (
            f"Cost raster (raster): the cell type is {info['dtype']}. Allowed: uint16 (cell values 0 to "
            f"{IMPASSABLE_CELL_COST - 1}, {IMPASSABLE_CELL_COST} = forbidden). Rebuild the raster in the Raster tab."))]
    return []


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


def memory_problems(est: dict, raster_path: Any, origins: Any, *, full_window: bool = False) -> list[Problem]:
    """The refusal of an estimate over the limit, with numbers the user can act on; ``[]`` for ok and warn."""
    if est["verdict"] != "refuse":
        return []
    info = raster_info(raster_path)
    hull = _hull_bounds([(float(x), float(y)) for x, y in origins])
    gpu_over = "device_mb" in est and est["device_mb"] > est["device_limit_mb"]
    if gpu_over and est["mb"] <= est["limit_mb"]:
        what = (f"on the graphics card it needs about {_fmt_mb(est['device_mb'])} MB ({GPU_BYTES_PER_CELL} bytes per "
                f"cell); the limit for one run is {_fmt_mb(est['device_limit_mb'])} MB (about "
                f"{est['device_max_cells']:,} cells)")
        max_cells = est["device_max_cells"]
    else:
        what = (f"it needs about {_fmt_mb(est['mb'])} MB ({est['bytes_per_cell']} bytes per cell for the search, the "
                f"field and the running result); the limit is {_fmt_mb(est['limit_mb'])} MB (about "
                f"{est['max_cells']:,} cells)")
        max_cells = est["max_cells"]
    head = (f"Memory (cost field): the search window would be {est['rows']:,} x {est['cols']:,} = "
            f"{est['cells']:,} cells at {est['resolution_m']:g} m per cell, and {what}. ")
    limit_buffer = _max_buffer_m(info, hull, max_cells)
    if limit_buffer is not None and 1.0 <= limit_buffer < est["buffer_m"]:
        step = "Switch off the whole-raster option and set" if full_window else "Set"
        advice = (f"{step} the search buffer to at most {math.floor(limit_buffer):,} m (the largest window that "
                  "fits).")
    else:
        factor = math.sqrt(est["cells"] / max_cells)
        advice = (f"Even the fixed points alone need that much: use a coarser cost raster (at least "
                  f"{est['resolution_m'] * factor:,.1f} m per cell) or fewer, closer fixed points.")
    return [Problem("memory", head + advice)]


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


def _has_objective(finder_options: dict | None) -> bool:
    return bool(finder_options) and finder_options.get("objective") is not None


def unit_for(finder_options: dict | None) -> tuple[str, str]:
    """``(unit, unit_note)`` of the values a run with these ``finder_options`` produces.

    Without an objective the plain cost units; with one the search runs on the weighted objective surface, so
    the values are objective units (:data:`OBJECTIVE_UNIT`), not the cost units of the raster.
    """
    return (OBJECTIVE_UNIT, OBJECTIVE_UNIT_NOTE) if _has_objective(finder_options) else (UNIT, UNIT_NOTE)


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
            meaning=(f"The search window has {est['cells']:,} cells and needs about {_fmt_mb(est['mb'])} MB "
                     f"({est['bytes_per_cell']} bytes per cell); the warning level is {_fmt_mb(est['warn_mb'])} MB, "
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
    """Check everything a run needs, from the page's values, before any job starts (main thread, cheap).

    ``raw_settings`` is the dict of the card's controls (:func:`validate_cost_field_settings`); ``origins`` the
    fixed points in the raster's CRS (:func:`origins_from_draft`); ``graph_api``, ``neighborhood`` (``"r2"`` or
    k), ``ignore_max_cost``, ``search_buffer_m`` (blank/0 = automatic) and ``full_window`` are the routing
    controls; ``finder_options`` is the optional extra ``PathFinder`` keywords (``dem``, ``objective``, ...) of the
    objective card. Returns a :class:`RunPlan`: with ``problems`` when the run is refused (show
    :func:`problem_notice`), else ``kwargs`` for :func:`compute_cost_field` (for example through
    :func:`job_work`) and the soft ``notices``.
    """
    settings = dict(COST_FIELD_DEFAULTS)
    if isinstance(raw_settings, dict):
        settings.update({k: v for k, v in raw_settings.items() if k in COST_FIELD_KEYS and not _is_blank(v)})
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
def _stopped(cancel: Any) -> bool:
    if cancel is None:
        return False
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel())


def _library_refusal(exc: BaseException, point_index: int | None) -> CostFieldRefused | None:
    """The plain-language refusal for a library error the checks above should have caught, else None."""
    text = str(exc)
    where = f"point {point_index + 1}" if point_index is not None else "a fixed point"
    low = text.lower()
    if isinstance(exc, FeatureNotImplementedError) and "slope terms" in low:
        return CostFieldRefused([Problem("algorithm", (
            "Kernel (algorithm): the chosen kernel cannot compute a cost field with a terrain model (DEM) because "
            "it does not apply the slope terms. Allowed with a DEM: cython with auto or dijkstra."))])
    if isinstance(exc, InvalidInputError) and "outside the search window" in low:
        return CostFieldRefused([Problem("origins", (
            f"Fixed points (origins): {where} lies outside the search window. Allowed: inside the raster; widen "
            "the search buffer (m) or move the point."))])
    if isinstance(exc, InvalidInputError) and "impassable cell" in low:
        return CostFieldRefused([Problem("origins", (
            f"Fixed points (origins): {where} snapped to a forbidden cell (value {IMPASSABLE_CELL_COST}). Move it "
            "to a passable cell."))])
    if isinstance(exc, InvalidInputError) and "symmetric edge weights" in low:
        return CostFieldRefused([Problem("graph_api", (
            "Backend (graph_api): the backend's edge weights are not symmetric, so a field rooted at a fixed point "
            f"would be wrong. Allowed: {', '.join(FIELD_BACKENDS)}. ({text})"))])
    return None


def _checkpoint(cancel: Any, label: str) -> None:
    if _stopped(cancel):
        raise jobs.JobCancelled(f"{label} was stopped")


def _step(progress: Any, done: int, total: int, stage: str) -> None:
    if progress is not None:
        progress(done, total, stage)


def _combine_into(acc: np.ndarray | None, arr: np.ndarray, combine: str) -> np.ndarray:
    """Fold one field into the running result in place (the first field becomes the result)."""
    if acc is None:
        return arr
    if combine == "sum":
        np.add(acc, arr, out=acc)            # inf stays inf: a cell one point cannot reach is unreachable
    else:
        np.minimum(acc, arr, out=acc)
    return acc


def _best_cell(values: np.ndarray, valid: np.ndarray, transform: Any, study_area: Any, combine: str,
               n_origins: int) -> tuple[dict | None, str, bool]:
    """``(best, note, study_area_hit)``: the cheapest cell (first in row-major order among ties) and a note.

    A single fixed point or "lowest of the points" has no meaningful cheapest cell (every fixed point itself costs
    0), so ``best`` is then None and the note says why.
    """
    if combine == "min" or n_origins < 2:
        why = "lowest of the points" if combine == "min" else "a single fixed point"
        return None, (f"No cheapest cell is named for {why}: the fixed point itself costs 0. "
                      "Use 'Sum over the points' with two or more fixed points."), True
    candidates = valid
    hit = True
    if study_area is not None:
        from rasterio.features import geometry_mask

        inside = geometry_mask([study_area], out_shape=values.shape, transform=transform, invert=True)
        candidates = valid & inside
        hit = bool(candidates.any())
        del inside
    if not candidates.any():
        where = " inside the study area" if study_area is not None else ""
        return None, f"No cell{where} is reachable from every fixed point.", hit
    masked = np.where(candidates, values, np.float32(np.inf))
    flat = int(np.argmin(masked))
    row, col = divmod(flat, values.shape[1])
    x, y = _apply_transform(transform, col + 0.5, row + 0.5)
    return ({"x": float(x), "y": float(y), "row": int(row), "col": int(col), "cost": float(masked[row, col])},
            "", hit)


def _write_field(values: np.ndarray, crs: Any, transform: Any, out_path: Path) -> None:
    """Write the field as a tiled, overviewed GeoTIFF (nodata -1) through a temporary name."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    try:
        tiles._write_tiled_geotiff(  # pylint: disable=protected-access  # same-package collaborator
            values, crs, transform, NODATA, part)
        os.replace(part, out_path)
    finally:
        if part.exists():
            part.unlink()


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


def _run_fields(finder: Any, points: list, est: dict, api: str, *, combine: str, algorithm: str, num_threads: int,
                neighborhood: str, out_path: Path, study_area: Any, cancel: Any, progress: Any,
                has_dem: bool, objective_scale: float | None = None) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    total = len(points) + 1
    unit, unit_note = unit_for({"objective": True} if objective_scale is not None else None)
    to_objective = np.float32(1.0 / objective_scale) if objective_scale not in (None, 1.0) else None
    kwargs = {"num_threads": num_threads} if (num_threads > 0 and api == "cython") else {}
    handler = finder.raster_handler
    crs = handler.raster_dataset.crs
    window = handler.data[0]
    forbidden = (window == IMPASSABLE_CELL_COST) if np.issubdtype(window.dtype, np.integer) else ~np.isfinite(window)
    acc = None
    used = None
    for i, point in enumerate(points):
        _checkpoint(cancel, "The cost field")
        _step(progress, i, total, f"point {i + 1} of {len(points)}: settling the whole field "
                                  "(Stop acts between points)")
        try:
            with finder.cost_field(point, algorithm=algorithm, **kwargs) as fld:
                used = fld.algorithm
                arr = fld.field_array()
        except (InvalidInputError, FeatureNotImplementedError) as exc:
            refusal = _library_refusal(exc, i)
            if refusal is None:
                raise
            raise refusal from exc
        if to_objective is not None:                # search-raster units -> the objective's own units
            arr = np.multiply(arr, to_objective, out=arr) if arr.flags.writeable else arr * to_objective
        acc = _combine_into(acc, arr, combine)
        del arr
    _checkpoint(cancel, "The cost field")
    _step(progress, len(points), total, "writing the GeoTIFF")
    valid = np.isfinite(acc) & ~forbidden
    n_forbidden = int(forbidden.sum())
    del forbidden
    n_valid = int(valid.sum())
    low = float(acc.min(where=valid, initial=np.inf)) if n_valid else 0.0
    high = float(acc.max(where=valid, initial=-np.inf)) if n_valid else 0.0
    study_area = _as_geometry(study_area, crs)
    best, best_note, area_hit = _best_cell(acc, valid, handler.window_transform, study_area, combine, len(points))
    acc[~valid] = np.float32(NODATA)
    transform = handler.window_transform
    rows, cols = acc.shape
    _write_field(acc, crs, transform, out_path)
    left, top = _apply_transform(transform, 0, 0)
    right, bottom = _apply_transform(transform, cols, rows)
    result = {
        "path": str(out_path), "min": low, "max": high, "unit": unit, "unit_note": unit_note, "nodata": NODATA,
        "origins": [[float(x), float(y)] for x, y in points], "n_origins": len(points), "combine": combine,
        "algorithm": used, "algorithm_requested": algorithm, "graph_api": api, "neighborhood": neighborhood,
        "rows": int(rows), "cols": int(cols), "cells": int(rows * cols), "valid_cells": n_valid,
        "nodata_cells": int(rows * cols - n_valid), "forbidden_cells": n_forbidden,
        "unreachable_cells": int(rows * cols - n_valid - n_forbidden), "resolution_m": float(abs(transform.a)),
        "bounds": [float(left), float(bottom), float(right), float(top)], "crs": str(crs) if crs else "",
        "best": best, "best_note": best_note, "estimate": est, "threads_requested": num_threads,
        "objective": objective_scale is not None, "objective_scale": objective_scale}
    result["notices"] = _result_notices(result, study_area_hit=area_hit, study_area=study_area, has_dem=has_dem)
    return result


def _result_notices(result: dict, *, study_area_hit: bool, study_area: Any, has_dem: bool) -> list[dict]:
    """Notices the poll side shows, derived from what the run observed (no warning capture in the worker)."""
    notices = []
    cells, valid = result["cells"], result["valid_cells"]
    if valid == 0:
        notices.append(errors.Notice(
            severity="warning", title="The cost field has no reachable cell",
            meaning="No cell of the search window can be reached from the fixed points.",
            impact="The result layer is empty.",
            fix="Widen the search buffer, move the fixed points, or check the cost raster for barriers.",
            focus_id=_TAB).to_dict())
    elif result["unreachable_cells"] > 0:
        share = 100.0 * result["unreachable_cells"] / cells
        reach = "any fixed point" if result["combine"] == "min" else "every fixed point"
        notices.append(errors.Notice(
            severity="info", title="Some cells cannot be reached",
            meaning=(f"{result['unreachable_cells']:,} of {cells:,} window cells ({share:.1f} %) cannot be reached "
                     f"from {reach} (cut off by forbidden cells); they are transparent in the layer, like the "
                     f"{result['forbidden_cells']:,} forbidden or out-of-area cells."),
            impact="The colour scale covers the reachable cells only.", fix="", focus_id=_TAB).to_dict())
    if study_area is not None and not study_area_hit and valid:
        notices.append(errors.Notice(
            severity="info", title="No cheapest cell inside the study area",
            meaning="No reachable cell lies inside the drawn study area, so no cheapest cell is named.",
            impact="The layer itself is complete.", fix="Draw the study area over reachable cells.",
            focus_id=_TAB).to_dict())
    used = str(result["algorithm"] or "")
    if result["threads_requested"] > 0 and (result["graph_api"] != "cython" or not used.startswith("delta")):
        reason = ("only the parallel CPU kernel (delta-stepping on cython) has worker threads"
                  + (", and the terrain model forces Dijkstra" if has_dem else ""))
        notices.append(errors.Notice(
            severity="info", title="The thread count was not used",
            meaning=f"Threads were set to {result['threads_requested']}, but {reason}.",
            impact="The run used the kernel's own parallelism, if any.", fix="", focus_id=_TAB).to_dict())
    return notices


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


def best_text(result: dict) -> str:
    """The readout under the run button, e.g. ``Cheapest cell: (500,010.5, 5,000,020.5), total 12,345 cost units``."""
    best = (result or {}).get("best")
    if not best:
        return (result or {}).get("best_note") or "No cheapest cell."
    unit = result.get("unit") or UNIT
    return (f"Cheapest cell: ({best['x']:,.1f}, {best['y']:,.1f}), total {best['cost']:,.0f} {unit}")


def summary_text(result: dict) -> str:
    """One status line for the card after a finished run."""
    used = result.get("algorithm") or result.get("algorithm_requested")
    how = "sum over" if result["combine"] == "sum" else "lowest of"
    unit = result.get("unit") or UNIT
    text = (f"Cost field done: {how} {result['n_origins']} fixed point(s), kernel {used}, {result['valid_cells']:,} "
            f"of {result['cells']:,} cells reachable, {result['min']:,.0f} to {result['max']:,.0f} {unit}.")
    if result.get("objective"):
        text += (" Computed with the objective of the Objective & terrain card, so the values are weighted objective "
                 "units, not cost units.")
        if result.get("objective_summary"):
            text += f" {result['objective_summary']}"
    return text


def is_cost_field_layer(layer: Any) -> bool:
    """True for a raster layer that holds a cost-field result (``meta["cost_field"]`` is set).

    Such a layer is a raster of accumulated cost, not a cost raster: nothing may be routed on it, so the
    dropdowns that expect a cost raster leave it out. Dropdowns that accept a raster of any meaning keep it.
    """
    return bool((getattr(layer, "meta", None) or {}).get("cost_field"))


def _objective_meta(result: dict) -> dict:
    """What ``meta["cost_field"]`` records about the objective (nothing for a plain run).

    ``objective`` holds the settings spec (layer ids, no paths) and the one-line summary the caller put into the
    result as ``objective_spec`` / ``objective_summary``, and the scale; ``unit_note`` says what the values mean.
    """
    if not result.get("objective"):
        return {}
    return {"objective": {"spec": result.get("objective_spec"), "summary": result.get("objective_summary"),
                          "scale": result.get("objective_scale")},
            "unit_note": result.get("unit_note")}


def finalize_cost_field(state: Any, result: dict, notices: list | None = None, *, name: str | None = None,
                        colormap: str = "viridis", raster_layer_id: str | None = None):
    """Register a finished result as a raster layer; returns ``(layer_or_None, notices)``.

    Call it from the poll callback, in the main thread (it changes ``state``). The layer is served through
    ``tiles.build_tile_layer`` and added with ``state.add_layer(..., tile=...)``, so its tile client belongs to the
    ``TileManager`` and is released with the layer. The colour scale runs over the valid cells' ``min`` to ``max``;
    nodata cells are transparent. ``layer.meta["cost_field"]`` holds ``origins``, ``combine``, ``algorithm``,
    ``best_xy``, ``best_cost``, ``unit`` and ``raster_layer_id``. ``notices`` (a list of dicts) gets the result's
    own notices and a success box; a failure to serve the file becomes an error notice and the layer is None.
    """
    notices = list(notices or [])
    default_name = f"Cost field ({'sum' if result['combine'] == 'sum' else 'min'} of {result['n_origins']} point(s))"
    display = name or default_name
    tile_layer, notices = errors.guard(
        tiles.build_tile_layer, result["path"], name=display, colormap=colormap or "viridis",
        work_dir=state.work_dir, vmin=result["min"], vmax=result["max"], notices=notices)
    if tile_layer is None:
        return None, notices
    crs = None
    try:
        import rasterio

        with rasterio.open(tile_layer.source_path) as src:
            crs = src.crs
    except Exception:  # pylint: disable=broad-exception-caught  # the CRS probe is best effort, as for any raster layer
        crs = None
    best = result.get("best")
    layer = state.add_layer(
        display, "raster", tile=tile_layer, crs=crs,
        meta={"source_path": tile_layer.source_path, "vmin": tile_layer.vmin, "vmax": tile_layer.vmax,
              "cost_field": {
                  "origins": result["origins"], "combine": result["combine"], "algorithm": result["algorithm"],
                  "best_xy": [best["x"], best["y"]] if best else None,
                  "best_cost": best["cost"] if best else None, "unit": result["unit"],
                  "raster_layer_id": raster_layer_id,
                  **_objective_meta(result)}})
    notices.extend(result.get("notices") or [])
    notices.append(errors.success(
        f"Cost field layer '{display}' added",
        meaning=f"{summary_text(result)} {best_text(result)}",
        impact=f"Values are {result['unit_note']}; unreachable and forbidden cells are transparent."))
    return layer, errors.cap_notices(notices)


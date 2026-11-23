"""
Run half of the cost-field service: the worker-thread computation and the finishing of a result.

``cost_field`` is the module the GUI imports; it re-exports every public name of this one. This module holds the
parts that never read the limits the tests patch (``MEMORY_LIMIT_MB``, ``MAX_ORIGINS``, ...) and never call
``plan_run`` or ``compute_cost_field``, so it imports nothing from ``cost_field``:

* the shared names (``Problem``, ``CostFieldRefused``, the units, ``NODATA``, ``FIELD_BACKENDS``, the tab id),
* the checks that read the raster header or one setting and no patched limit (``raster_info``, the points' cells,
  buffer, neighbourhood and cell type),
* the settling of one field per fixed point, the cheapest cell, the GeoTIFF writer and the notices the run derives
  from what it observed (``_run_fields``),
* the texts of a finished run and its registration as a raster layer (``finalize_cost_field``).

It is a pure service like ``cost_field``: no Dash, no component ids.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from pyorps.core.exceptions import FeatureNotImplementedError, InvalidInputError
from pyorps.core.types import IMPASSABLE_CELL_COST

from . import errors, geo, jobs, routing, tiles

# ------------------------------------------------------------------ shared names
#: backends that can root a field: the graph libraries use half step sets, so a field rooted at a fixed point
#: would not be symmetric (library ``CostField`` refuses them)
FIELD_BACKENDS = ("cython", "raster_gpu", "raster_fim")


#: unit of every cost-field value: cell value x metres (EUR when the raster holds EUR per metre)
UNIT = "cost units"


UNIT_NOTE = "cell value x metres (EUR on a raster that holds EUR per metre)"


#: unit of a field computed with the objective of the Objective & terrain card: the search runs on the weighted
#: combination of the cost raster, the length term, the slope term and the metric layers (the library's
#: "objective units x m, not EUR"), so these values are not comparable with the plain cost units above
OBJECTIVE_UNIT = "objective units"


OBJECTIVE_UNIT_NOTE = ("weighted objective value x metres: the weighted sum of the cost raster, the length term, " +
                       "the slope term and any extra layers, not EUR")


#: nodata of the result raster: costs are never negative, so no reachable cell can take it
NODATA = -1.0


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


def _apply_transform(transform: Any, col: float, row: float) -> tuple[float, float]:
    """Map (col, row) to (x, y) with the six coefficients of an affine transform.

    Plain arithmetic instead of multiplying the transform with the point: affine 3 deprecates the ``*``
    operator for that (a warning per call, an error under ``-W error``), and its replacement ``@`` does
    not exist in affine 2.
    """
    return (transform.a * col + transform.b * row + transform.c,
            transform.d * col + transform.e * row + transform.f)


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


def _has_objective(finder_options: dict | None) -> bool:
    return bool(finder_options) and finder_options.get("objective") is not None


def unit_for(finder_options: dict | None) -> tuple[str, str]:
    """``(unit, unit_note)`` of the values a run with these ``finder_options`` produces.

    Without an objective the plain cost units; with one the search runs on the weighted objective surface, so
    the values are objective units (:data:`OBJECTIVE_UNIT`), not the cost units of the raster.
    """
    return (OBJECTIVE_UNIT, OBJECTIVE_UNIT_NOTE) if _has_objective(finder_options) else (UNIT, UNIT_NOTE)


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
    # (exception type, fragment of the lower-cased message, key of the setting, message), first match wins
    rules = (
        (FeatureNotImplementedError, "slope terms", "algorithm", (
            "Kernel (algorithm): the chosen kernel cannot compute a cost field with a terrain model (DEM) because " +
            "it does not apply the slope terms. Allowed with a DEM: cython with auto or dijkstra.")),
        (InvalidInputError, "outside the search window", "origins", (
            f"Fixed points (origins): {where} lies outside the search window. Allowed: inside the raster; widen " +
            "the search buffer (m) or move the point.")),
        (InvalidInputError, "impassable cell", "origins", (
            f"Fixed points (origins): {where} snapped to a forbidden cell (value {IMPASSABLE_CELL_COST}). Move it " +
            "to a passable cell.")),
        (InvalidInputError, "symmetric edge weights", "graph_api", (
            "Backend (graph_api): the backend's edge weights are not symmetric, so a field rooted at a fixed point " +
            f"would be wrong. Allowed: {', '.join(FIELD_BACKENDS)}. ({text})")))
    for kind, fragment, key, message in rules:
        if isinstance(exc, kind) and fragment in low:
            return CostFieldRefused([Problem(key, message)])
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
        return None, (f"No cheapest cell is named for {why}: the fixed point itself costs 0. " +
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


def _reach_notices(result: dict) -> list[dict]:
    """The notice for a field without reachable cells, or for cells that cannot be reached."""
    cells, valid = result["cells"], result["valid_cells"]
    if valid == 0:
        return [errors.Notice(
            severity="warning", title="The cost field has no reachable cell",
            meaning="No cell of the search window can be reached from the fixed points.",
            impact="The result layer is empty.",
            fix="Widen the search buffer, move the fixed points, or check the cost raster for barriers.",
            focus_id=_TAB).to_dict()]
    if result["unreachable_cells"] > 0:
        share = 100.0 * result["unreachable_cells"] / cells
        reach = "any fixed point" if result["combine"] == "min" else "every fixed point"
        return [errors.Notice(
            severity="info", title="Some cells cannot be reached",
            meaning=(f"{result['unreachable_cells']:,} of {cells:,} window cells ({share:.1f} %) cannot be reached " +
                     f"from {reach} (cut off by forbidden cells); they are transparent in the layer, like the " +
                     f"{result['forbidden_cells']:,} forbidden or out-of-area cells."),
            impact="The colour scale covers the reachable cells only.", fix="", focus_id=_TAB).to_dict()]
    return []


def _study_area_notices(result: dict, study_area_hit: bool, study_area: Any) -> list[dict]:
    """The notice for a study area that holds no reachable cell."""
    if study_area is not None and not study_area_hit and result["valid_cells"]:
        return [errors.Notice(
            severity="info", title="No cheapest cell inside the study area",
            meaning="No reachable cell lies inside the drawn study area, so no cheapest cell is named.",
            impact="The layer itself is complete.", fix="Draw the study area over reachable cells.",
            focus_id=_TAB).to_dict()]
    return []


def _thread_notices(result: dict, has_dem: bool) -> list[dict]:
    """The notice for a thread count the chosen kernel did not use."""
    used = str(result["algorithm"] or "")
    if result["threads_requested"] > 0 and (result["graph_api"] != "cython" or not used.startswith("delta")):
        reason = ("only the parallel CPU kernel (delta-stepping on cython) has worker threads"
                  + (", and the terrain model forces Dijkstra" if has_dem else ""))
        return [errors.Notice(
            severity="info", title="The thread count was not used",
            meaning=f"Threads were set to {result['threads_requested']}, but {reason}.",
            impact="The run used the kernel's own parallelism, if any.", fix="", focus_id=_TAB).to_dict()]
    return []


def _result_notices(result: dict, *, study_area_hit: bool, study_area: Any, has_dem: bool) -> list[dict]:
    """Notices the poll side shows, derived from what the run observed (no warning capture in the worker)."""
    return [*_reach_notices(result), *_study_area_notices(result, study_area_hit, study_area),
            *_thread_notices(result, has_dem)]


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
    text = (f"Cost field done: {how} {result['n_origins']} fixed point(s), kernel {used}, {result['valid_cells']:,} " +
            f"of {result['cells']:,} cells reachable, {result['min']:,.0f} to {result['max']:,.0f} {unit}.")
    if result.get("objective"):
        text += (" Computed with the objective of the Objective & terrain card, so the values are weighted objective " +
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


def _tile_crs(tile_layer: Any) -> Any:
    """The CRS of the served file, None when it cannot be probed (best effort, as for any raster layer)."""
    try:
        import rasterio

        with rasterio.open(tile_layer.source_path) as src:
            return src.crs
    except Exception:  # pylint: disable=broad-exception-caught  # the CRS probe is best effort, as for any raster layer
        return None


def _cost_field_meta(result: dict, raster_layer_id: str | None) -> dict:
    """``meta["cost_field"]`` of the result layer."""
    best = result.get("best")
    return {"origins": result["origins"], "combine": result["combine"], "algorithm": result["algorithm"],
            "best_xy": [best["x"], best["y"]] if best else None,
            "best_cost": best["cost"] if best else None, "unit": result["unit"],
            "raster_layer_id": raster_layer_id,
            **_objective_meta(result)}


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
    layer = state.add_layer(
        display, "raster", tile=tile_layer, crs=_tile_crs(tile_layer),
        meta={"source_path": tile_layer.source_path, "vmin": tile_layer.vmin, "vmax": tile_layer.vmax,
              "cost_field": _cost_field_meta(result, raster_layer_id)})
    notices.extend(result.get("notices") or [])
    notices.append(errors.success(
        f"Cost field layer '{display}' added",
        meaning=f"{summary_text(result)} {best_text(result)}",
        impact=f"Values are {result['unit_note']}; unreachable and forbidden cells are transparent."))
    return layer, errors.cap_notices(notices)


# ------------------------------------------------------------------ raster header and point checks
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


def _listed_points(points: list[tuple[float, float]], indices: list[int]) -> str:
    """The first five of the points ``indices`` as ``point 3 at (x, y)``, joined by semicolons."""
    return "; ".join(f"point {i + 1} at ({points[i][0]:,.1f}, {points[i][1]:,.1f})" for i in indices[:5])


def _outside_problems(raster_path: Any, info: dict, points: list[tuple[float, float]]) -> list[Problem]:
    """The problem of the points that lie outside the raster, ``[]`` when all are inside."""
    inside = {i for i, _r, _c in _origin_cells(raster_path, info, points)}
    outside = [i for i in range(len(points)) if i not in inside]
    if not outside:
        return []
    left, bottom, right, top = info["bounds"]
    listed = _listed_points(points, outside)
    return [Problem("origins", (
        f"Fixed points (origins): {len(outside)} of {len(points)} points are outside the raster: {listed}. " +
        f"Allowed: x from {left:,.1f} to {right:,.1f} m and y from {bottom:,.1f} to {top:,.1f} m. " +
        "Move the points inside the raster."))]


def _raster_point_problems(raster_path: Any, points: list[tuple[float, float]], *, ignore_max_cost: bool,
                           check_forbidden: bool) -> list[Problem]:
    """The problems that need the raster header: unreadable file, points outside, points on a forbidden cell."""
    try:
        info = raster_info(raster_path)
    except FileNotFoundError as exc:
        return [Problem("raster", f"Cost raster (raster): {exc}. Rebuild the cost raster or load the .tif again.")]
    except Exception as exc:  # pylint: disable=broad-exception-caught  # an unreadable file is a problem, not a crash
        return [Problem("raster", f"Cost raster (raster): '{raster_path}' cannot be read ({exc}).")]
    outside = _outside_problems(raster_path, info, points)
    if outside:
        return outside
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
    listed = _listed_points(points, on_forbidden)
    return [Problem("origins", (
        f"Fixed points (origins): {len(on_forbidden)} of {len(points)} points stand on a forbidden cell " +
        f"(value {IMPASSABLE_CELL_COST}): {listed}. With 'Forbidden cells are barriers' on, a field cannot start " +
        "there. Move the point to a passable cell, or switch the barrier option off."))]


def _check_dtype(info: dict, finder_options: dict | None) -> list[Problem]:
    if finder_options and finder_options.get("objective") is not None:
        return []                                    # the objective pipeline builds its own search raster
    if info["dtype"] != "uint16":
        return [Problem("raster", (
            f"Cost raster (raster): the cell type is {info['dtype']}. Allowed: uint16 (cell values 0 to " +
            f"{IMPASSABLE_CELL_COST - 1}, {IMPASSABLE_CELL_COST} = forbidden). Rebuild the raster in the Raster tab."))]
    return []


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else repr(value)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _check_buffer(search_buffer_m: Any) -> list[Problem]:
    if _is_blank(search_buffer_m):
        return []
    allowed = "Allowed: 0 (automatic) or a positive number of metres (m)."
    try:
        value = float(search_buffer_m)
    except (TypeError, ValueError):
        return [Problem("search_buffer_m", f"Search buffer (search_buffer_m): {search_buffer_m!r} is not a number. " +
                                           f"{allowed}")]
    if not math.isfinite(value) or value < 0:
        return [Problem("search_buffer_m", f"Search buffer (search_buffer_m): {_fmt(value)} m is not usable. " +
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
        f"Neighbourhood k (neighborhood): {neighborhood!r} is not usable. Allowed: whole numbers from " +
        f"{routing.NEIGHBORHOOD_MIN} to {routing.NEIGHBORHOOD_MAX}."))]

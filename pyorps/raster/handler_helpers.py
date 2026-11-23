"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland

Module-level helpers of :mod:`pyorps.raster.handler`: CRS names, input checks,
error messages and coordinate predicates. :mod:`pyorps.raster.handler`
re-imports every name, so each one stays importable from there as the same
object. This module never imports the handler back, and it never calls the
rasterio functions that tests patch in the handler module (``rowcol``,
``transform_xy``, ``rasterize``, ``rio_open``): the code that calls those stays
in :mod:`pyorps.raster.handler`.
"""
from typing import Any

import numpy as np
from pyproj import CRS
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from pyorps.core.exceptions import InvalidInputError, PyorpsError
from pyorps.core.types import CoordinateList, CoordinateTuple


def _crs_name(crs: Any) -> str:
    """Short printable name of a CRS (``EPSG:32632`` where possible)."""
    if crs is None:
        return "None (unknown)"
    try:
        parsed = CRS.from_user_input(crs)
    except Exception:  # noqa: BLE001  # any unparsable CRS: print it raw
        return str(crs)
    epsg = parsed.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else parsed.to_string()


def _require_integer_dtype(dtype: Any, why: str, remedy: str) -> None:
    """Raise PyorpsError when ``dtype`` has no integer maximum."""
    if np.issubdtype(np.dtype(dtype), np.integer):
        return
    raise PyorpsError(
        f"Unsupported raster dtype {np.dtype(dtype).name!r}: {why}, which "
        f"exists only for integer dtypes. Either {remedy}, or use a uint16 "
        f"cost raster (the dtype PathFinder routes on; cast with "
        f".astype('uint16') after clipping to 0..65534, 65535 = forbidden).")


def _is_projected(crs: Any) -> bool:
    """True when ``crs`` parses as a projected CRS (metres, not degrees)."""
    if crs is None:
        return False
    try:
        return bool(CRS.from_user_input(crs).is_projected)
    except Exception:  # noqa: BLE001  # unparsable: make no claim
        return False


def _is_single_coordinate(coords: Any) -> bool:
    """True for one ``(x, y)`` pair: a tuple, or a flat list of two items."""
    return isinstance(coords, tuple) or (
            isinstance(coords, list) and
            len(coords) == 2 and
            not isinstance(coords[0], (list, tuple))
    )


def _empty_search_space_error(buffer_m: float, hull_type: str) -> InvalidInputError:
    """The error for a ``search_space_buffer_m`` that leaves no search space.

    ``hull_type`` is the geometry type of the source/target hull that was
    buffered. A negative buffer shrinks the hull, a zero buffer removes a
    hull without area; the message names the cause and the remedy.
    """
    if buffer_m < 0:
        cause = (f"a negative buffer shrinks the source/target hull "
                 f"(a {hull_type}) by "
                 f"{abs(buffer_m)} m on every side, "
                 f"and nothing of it is left")
    else:
        cause = (f"the source/target hull is a "
                 f"{hull_type} with no area, so buffering "
                 f"it by zero removes it")
    return InvalidInputError(
        f"search_space_buffer_m={buffer_m} leaves an "
        f"empty search space: {cause}. Pass a positive buffer, or "
        f"pyorps.full_window_buffer_m(dataset) to search the whole "
        f"raster.")


def _max_distance_pair(
        coords1: CoordinateTuple | CoordinateList,
        coords2: CoordinateTuple | CoordinateList
):
    """Body of :meth:`pyorps.raster.handler.RasterHandler.max_distance_pair`."""

    # Normalize inputs to lists of tuples
    def normalize_coords(coords):
        is_coord_list = (len(coords) == 0 or not isinstance(coords[0], tuple))
        if isinstance(coords, tuple) and is_coord_list:
            return [coords]
        return coords

    coords1_list = normalize_coords(coords1)
    coords2_list = normalize_coords(coords2)

    if not coords1_list or not coords2_list:
        return None  # Handle empty inputs

    max_distance = -1
    max_pair = None

    for point1 in coords1_list:
        for point2 in coords2_list:
            # Calculate Euclidean distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

            if distance > max_distance:
                max_distance = distance
                max_pair = (point1, point2)

    return max_pair, max_distance


def _check_finite_points(xs, ys) -> None:
    """Raise PyorpsError naming the first non-finite (x, y) point."""
    try:
        x_arr = np.atleast_1d(np.asarray(xs, dtype=float))
        y_arr = np.atleast_1d(np.asarray(ys, dtype=float))
    except (TypeError, ValueError):
        return  # not numeric: leave the error to rowcol, as before
    if x_arr.shape != y_arr.shape:
        return
    bad = ~(np.isfinite(x_arr) & np.isfinite(y_arr))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise PyorpsError(
            f"Coordinate ({float(x_arr[i])!r}, {float(y_arr[i])!r}) is not finite "
            f"({int(bad.sum())} of {bad.size} point(s) are NaN or "
            f"infinite). Allowed: finite (x, y) numbers in the raster "
            f"CRS.")


def _raise_if_outside_window(
        indices: np.ndarray,
        xs,
        ys,
        window_shape: tuple[int, int],
        window: Window,
        dataset: Any
) -> None:
    """Raise PyorpsError if any (row, col) lies outside the window data.

    ``window_shape`` is the (rows, columns) shape of the window data,
    ``window`` the search window and ``dataset`` the raster dataset it was
    cut from. Runs once per call on the given points (never per cell).
    """
    n_rows, n_cols = window_shape
    rows = indices[:, 0]
    cols = indices[:, 1]
    outside = (rows < 0) | (rows >= n_rows) | (cols < 0) | (cols >= n_cols)
    if not outside.any():
        return
    i = int(np.flatnonzero(outside)[0])
    x = float(np.atleast_1d(np.asarray(xs, dtype=float))[i])
    y = float(np.atleast_1d(np.asarray(ys, dtype=float))[i])
    transform = dataset.transform
    left, bottom, right, top = window_bounds(window, transform)
    height, width = dataset.shape[-2:]
    full_w, full_s, full_e, full_n = array_bounds(height, width, transform)
    crs = dataset.crs
    message = (
        f"Point ({x!r}, {y!r}) lies outside the search window: it maps to "
        f"window cell (row {int(rows[i])}, col {int(cols[i])}), but the "
        f"window has {n_rows} x {n_cols} cells "
        f"({int(outside.sum())} of {outside.size} point(s) are outside). "
        f"Search window bounds (left, bottom, right, top) = "
        f"({left:.3f}, {bottom:.3f}, {right:.3f}, {top:.3f}); full raster "
        f"bounds = ({full_w:.3f}, {full_s:.3f}, {full_e:.3f}, {full_n:.3f}); "
        f"raster CRS = {_crs_name(crs)}. Allowed: points inside the "
        f"search window, given in the raster CRS.")
    if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0 and _is_projected(crs):
        message += (
            f" The point looks like longitude/latitude degrees while the "
            f"raster CRS is projected: this is probably a CRS mismatch. "
            f"Reproject the point to {_crs_name(crs)} first, e.g. with "
            f"pyproj.Transformer.from_crs('EPSG:4326', "
            f"'{_crs_name(crs)}', always_xy=True).")
    raise PyorpsError(message)

"""LineString simplification utilities for post-routing path simplification.

Three QGIS-equivalent methods are exposed via :func:`simplify_linestring`:

* ``"douglas_peucker"``      – Shapely's distance-based simplifier,
                                 ``preserve_topology=False`` (Ramer-Douglas-Peucker).
* ``"douglas_peucker_topology"`` – Shapely's distance-based simplifier with
                                 ``preserve_topology=True`` (GEOS topology-preserving).
* ``"visvalingam"``          – Area-based Visvalingam-Whyatt (Rust-backed
                                 ``simplification`` package, optional dep).
* ``"grid"``                 – Snap every vertex to a regular grid of spacing
                                 ``tolerance`` and collapse consecutive
                                 duplicates.

All methods preserve endpoints. Tolerance units:

* Douglas-Peucker variants – CRS units (metres for projected rasters).
* Visvalingam              – CRS units squared (the minimum effective area).
* Grid                     – CRS units (the grid cell size).
"""
from __future__ import annotations

from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString


_METHODS: Final[tuple[str, ...]] = (
    "douglas_peucker",
    "douglas_peucker_topology",
    "visvalingam",
    "grid",
)


def simplify_linestring(
    line: LineString,
    method: str,
    tolerance: float,
) -> LineString:
    """Simplify *line* with the requested method.

    Parameters
    ----------
    line:
        A 2-D :class:`shapely.geometry.LineString` with at least two vertices.
    method:
        One of ``"douglas_peucker"``, ``"douglas_peucker_topology"``,
        ``"visvalingam"``, ``"grid"``.
    tolerance:
        Non-negative float. See the module docstring for unit conventions per
        method.

    Returns
    -------
    LineString
        A new LineString. Endpoints are always preserved. May equal *line*
        for short inputs.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown simplification method {method!r}; "
            f"expected one of {_METHODS}"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    if len(line.coords) <= 2:
        return LineString(line.coords)

    if method == "douglas_peucker":
        return shapely.simplify(line, tolerance, preserve_topology=False)
    if method == "douglas_peucker_topology":
        return shapely.simplify(line, tolerance, preserve_topology=True)
    if method == "visvalingam":
        return _simplify_visvalingam(line, tolerance)
    if method == "grid":
        return _simplify_grid(line, tolerance)
    raise AssertionError("unreachable")  # pragma: no cover


def _simplify_visvalingam(line: LineString, area_tolerance: float) -> LineString:
    try:
        from simplification.cutil import simplify_coords_vw
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'simplification' package is required for the 'visvalingam' "
            "method. Install it with `pip install pyorps[simplify]`."
        ) from exc

    coords = np.asarray(line.coords, dtype=np.float64)
    simplified = simplify_coords_vw(coords, area_tolerance)
    if len(simplified) < 2:
        # Pathological: every triangle was below tolerance. Keep endpoints.
        simplified = np.array([coords[0], coords[-1]], dtype=np.float64)
    return LineString(simplified)


def _simplify_grid(line: LineString, grid_size: float) -> LineString:
    if grid_size == 0:
        return LineString(line.coords)
    snapped = shapely.set_precision(line, grid_size, mode="valid_output")
    # `set_precision` can collapse a line to a point if all vertices snap to
    # the same cell. Fall back to the two raw endpoints in that case.
    if snapped.is_empty or snapped.geom_type != "LineString":
        coords = list(line.coords)
        return LineString([coords[0], coords[-1]])
    # Drop consecutive duplicates that survive Shapely's snap.
    raw = list(snapped.coords)
    deduped = [raw[0]]
    for c in raw[1:]:
        if c != deduped[-1]:
            deduped.append(c)
    if len(deduped) < 2:
        deduped = [raw[0], raw[-1]]
    return LineString(deduped)

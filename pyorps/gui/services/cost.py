"""
PYORPS GUI: cost evaluation for interactively edited routes.

When a user drags a route vertex on the map we need to answer "what does this
edit cost?" using the *same* metric pyorps reports for a routed path. We do that
by discretizing the edited polyline into an 8-connected sequence of raster cells
(the exact shape a real least-cost path has) and feeding it to pyorps' own
``calculate_path_metrics_numba``. The numbers are therefore directly comparable
to ``Path.total_length`` / ``Path.total_cost``.

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString

from pyorps.raster.handler import RasterHandler
from pyorps.utils.traversal import calculate_path_metrics_numba


@dataclass
class RouteCost:
    """Cost breakdown of an edited route, in pyorps' native metric units."""

    total_length: float          # path length in CRS units (metres)
    total_cost: float            # distance-weighted terrain cost (cost x length)
    total_cell_cost: float       # raw sum of raster values along the path
    geodesic_length_m: float     # true geometric length in CRS units (metres)
    length_by_category: dict[float, float] = field(default_factory=dict)
    n_forbidden_cells: int = 0   # cells at the raster's max/forbidden value
    n_out_of_window: int = 0     # vertices/steps clipped to the raster window
    crosses_forbidden: bool = False


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Return the 8-connected cells from (r0, c0) to (r1, c1) inclusive.

    Classic integer Bresenham; the resulting cell chain moves one step at a time
    (orthogonal or diagonal), matching how pyorps' routed paths step through the
    grid so the metric function measures them identically.
    """
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def line_to_cell_indices(line: LineString,
                         handler: RasterHandler) -> tuple[np.ndarray, int]:
    """Discretize a projected LineString into flat window-local cell indices.

    Returns the flat indices (row * width + col) into ``handler.data[0]`` and the
    number of vertices that fell outside the raster window (clipped).
    """
    coords = [(float(x), float(y)) for x, y in line.coords]
    # coords_to_indices returns window-local (row, col) pairs.
    rc = handler.coords_to_indices(coords)
    rc = np.atleast_2d(np.asarray(rc))

    height, width = handler.data.shape[-2], handler.data.shape[-1]

    n_out = 0
    vertices: list[tuple[int, int]] = []
    for row, col in rc:
        row_i, col_i = int(row), int(col)
        clipped_r = min(max(row_i, 0), height - 1)
        clipped_c = min(max(col_i, 0), width - 1)
        if clipped_r != row_i or clipped_c != col_i:
            n_out += 1
        vertices.append((clipped_r, clipped_c))

    # Walk each segment as an 8-connected cell chain, dropping duplicate cells.
    cells: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(vertices[:-1], vertices[1:]):
        seg = _bresenham(r0, c0, r1, c1)
        if cells and seg and cells[-1] == seg[0]:
            seg = seg[1:]
        cells.extend(seg)
    if not cells and vertices:
        cells = [vertices[0]]

    flat = np.array([r * width + c for r, c in cells], dtype=np.uint32)
    return flat, n_out


def evaluate_route_cost(line: LineString, handler: RasterHandler) -> RouteCost:
    """Compute the cost of an edited route against the loaded cost raster.

    ``line`` must be in the raster's CRS. The metric mirrors
    ``PathFinder.calculate_path_metrics`` exactly (same numba kernel), so the
    returned totals are comparable to the original routed path's totals.
    """
    raster_data = handler.data[0]
    flat, n_out = line_to_cell_indices(line, handler)

    if flat.size < 2:
        return RouteCost(
            total_length=0.0, total_cost=0.0, total_cell_cost=0.0,
            geodesic_length_m=float(line.length), n_out_of_window=n_out,
        )

    total_length, categories, lengths = calculate_path_metrics_numba(
        raster_data, flat)

    # The kernel counts cell steps (1.0 orthogonal, sqrt(2) diagonal) and
    # never sees the transform. Scale to CRS units exactly as
    # PathFinder.calculate_path_metrics does, so total_length stays
    # comparable to Path.total_length and to geodesic_length_m below.
    # pyorps assumes square cells - distances use |transform.a|.
    cell_size = float(abs(handler.window_transform.a))
    total_length = float(total_length) * cell_size
    lengths = np.asarray(lengths, dtype=float) * cell_size

    length_by_category = {float(c): float(_len)
                          for c, _len in zip(categories, lengths) if _len > 0}
    total_cost = float(sum(c * _len for c, _len in zip(categories, lengths)))

    rows_idx = flat // raster_data.shape[1]
    cols_idx = flat % raster_data.shape[1]
    total_cell_cost = float(raster_data[rows_idx, cols_idx].sum())

    forbidden = np.iinfo(raster_data.dtype).max
    n_forbidden = int(np.count_nonzero(raster_data[rows_idx, cols_idx] == forbidden))

    return RouteCost(
        total_length=float(total_length),
        total_cost=total_cost,
        total_cell_cost=total_cell_cost,
        geodesic_length_m=float(line.length),
        length_by_category=length_by_category,
        n_forbidden_cells=n_forbidden,
        n_out_of_window=n_out,
        crosses_forbidden=n_forbidden > 0,
    )

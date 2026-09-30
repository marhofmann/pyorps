"""Line-integral re-pricing of a route (plan rev. 5, D9; Stage-5 fidelity).

The search prices a step with the implemented PYORPS rule: the sum of the
cells it touches (both ends and the intermediates) times
``hypot(dr, dc) / (2 + n_int)`` -- a length-weighted MEAN of those cells
(CIRED 2027 draft, eq. 1). A cable laid along the same polyline actually
pays ``integral of c(x) ds``: each cell's value times the length of the
line inside that cell. The two agree on straight orthogonal runs over a
uniform raster and differ wherever a diagonal or knight step clips cells
unevenly. This module computes the integral exactly (a grid traversal
with the exact per-cell chord lengths) so the fidelity gap of the
headline design can be stated, not guessed.

Units follow ``Path.total_cost``: cell value x metres, with square cells
of ``cell_m`` metres.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["fidelity_delta", "line_integral_cost"]

EXCLUDED = 65535


def _segment(values, r0: float, c0: float, r1: float, c1: float,
             excluded: int) -> float:
    """``integral c ds`` from ``(r0, c0)`` to ``(r1, c1)`` in cell units."""
    dr, dc = r1 - r0, c1 - c0
    length = math.hypot(dr, dc)
    if length == 0.0:
        return 0.0
    ts = [0.0, 1.0]
    lo, hi = sorted((r0, r1))
    for k in range(math.floor(lo) + 1, math.ceil(hi)):
        ts.append((k - r0) / dr)
    lo, hi = sorted((c0, c1))
    for k in range(math.floor(lo) + 1, math.ceil(hi)):
        ts.append((k - c0) / dc)
    ts = sorted(set(t for t in ts if 0.0 <= t <= 1.0))
    total = 0.0
    rows, cols = values.shape
    for ta, tb in zip(ts, ts[1:]):
        if tb <= ta:
            continue
        tm = 0.5 * (ta + tb)
        r = int(math.floor(r0 + dr * tm))
        c = int(math.floor(c0 + dc * tm))
        if not (0 <= r < rows and 0 <= c < cols):
            return math.inf
        v = int(values[r, c])
        if v == excluded:
            return math.inf
        total += v * (tb - ta) * length
    return total


def line_integral_cost(values, cells, *, cell_m: float = 1.0,
                       excluded: int = EXCLUDED) -> float:
    """Exact ``integral c ds`` along the polyline through the cell centres.

    Parameters:
        values: The cost raster (cost per metre per cell).
        cells: Flat cell indices of the route, in order.
        cell_m: Cell size in metres (square cells).
        excluded: The exclusion value; a positive length inside such a
            cell makes the integral ``inf`` (touching a corner does not).

    Returns:
        The integral in cell value x metres; ``0.0`` for one cell.
    """
    v = np.asarray(values)
    cols = v.shape[1]
    idx = np.asarray(cells, dtype=np.int64).ravel()
    if idx.size < 2:
        return 0.0
    rr, cc = np.divmod(idx, cols)
    total = 0.0
    for k in range(idx.size - 1):
        total += _segment(v, rr[k] + 0.5, cc[k] + 0.5, rr[k + 1] + 0.5,
                          cc[k + 1] + 0.5, excluded)
        if total == math.inf:
            return math.inf
    return total * float(cell_m)


def fidelity_delta(values, steps, cells, *, cell_m: float = 1.0
                   ) -> tuple[float, float, float]:
    """``(discrete, integral, relative gap)`` of one route.

    ``discrete`` is the kernel's own price (``price_route_cython``) times
    ``cell_m``; ``integral`` is :func:`line_integral_cost`; the gap is
    ``(integral - discrete) / discrete``.
    """
    from pyorps.utils._dijkstra import price_route_cython

    raster = np.ascontiguousarray(np.asarray(values, dtype=np.uint16))
    st = np.ascontiguousarray(np.asarray(steps, dtype=np.int8))
    discrete = float(price_route_cython(
        raster, st, np.asarray(cells, dtype=np.int64))) * float(cell_m)
    integral = line_integral_cost(raster, cells, cell_m=cell_m)
    gap = (integral - discrete) / discrete if discrete > 0 else 0.0
    return discrete, integral, gap

"""Directional raster primitives: ray scans and ray sliding windows.

Pure array in, array out. No routing concepts live here -- this is the
layer :mod:`pyorps.graph.tower_field` is built on, and the layer where an
off-by-one is invisible and fatal, so every operator has a brute-force
reference next to it in ``tests/test_utils/test_directional.py``.

A *ray* is the sequence of cells ``x, x + d, x + 2d, ...`` for a primitive
integer direction ``d = (p, q)`` with ``gcd(|p|, |q|) == 1``. Because ``d``
is primitive the rays of one direction partition the grid, and a cell's
position along its own ray is the integer

    ``t(r, c) = alpha * r + beta * c``    with    ``alpha * p + beta * q = 1``

(Bezout; solvable exactly because ``gcd(p, q) == 1``), which satisfies
``t(x + d) = t(x) + 1``. Every operator below is a segmented scan keyed on
that index, so all of them are vectorised over the whole grid and none of
them loops over cells.

Costs, for a grid of ``N`` cells and a window of ``w`` offsets:

===========================  ==========================================
:func:`ray_prefix_sum`       ``O(N log n)`` -- Hillis-Steele doubling
:func:`ray_window_min`/max   ``O(N log w)`` -- van Herk/Gil-Werman with
                             the two half-block scans done by doubling
                             instead of by a ``w``-step loop
:func:`ray_run_length`       ``O(N log cap)``
===========================  ==========================================

The point of the window operators is that the cost does **not** grow with
the window width the way an explicit minimum over every admissible span
length does; ``log w`` is the price of vectorising the two block scans.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "primitive_directions",
    "direction_angles",
    "bezout",
    "ray_index",
    "shift",
    "ray_prefix_sum",
    "ray_window_min",
    "ray_window_max",
    "ray_run_length",
    "circular_window_min",
    "ray_window_min_bruteforce",
    "ray_window_max_bruteforce",
    "ray_prefix_sum_bruteforce",
]


# --------------------------------------------------------------- directions

def primitive_directions(dmax: int) -> np.ndarray:
    """Primitive integer directions with ``|p|, |q| <= dmax``.

    One entry per distinct angle -- ``(2, 2)`` is dropped because it is
    the same ray as ``(1, 1)`` -- sorted by ``atan2(q, p)`` so the
    direction axis is an angle axis, which is what the circular window in
    :func:`circular_window_min` needs.

    Parameters:
        dmax: Largest component magnitude. ``1`` gives the 8 r1 steps,
            ``2`` gives 16, ``3`` gives 32, ``4`` gives 48, ``5`` gives 80.

    Returns:
        ``(K, 2)`` int64 array of ``(dr, dc)``.
    """
    if dmax < 1:
        raise ValueError(f"dmax must be >= 1, got {dmax}")
    out = [(p, q)
           for p in range(-dmax, dmax + 1)
           for q in range(-dmax, dmax + 1)
           if (p or q) and math.gcd(abs(p), abs(q)) == 1]
    arr = np.array(out, dtype=np.int64)
    order = np.argsort(np.arctan2(arr[:, 1], arr[:, 0]), kind="stable")
    return np.ascontiguousarray(arr[order])


def direction_angles(dirs, scale=None) -> np.ndarray:
    """Angle of each direction in radians, ``atan2(dc, dr)`` in ``(-pi, pi]``.

    Parameters:
        dirs: ``(K, 2)`` integer directions as ``(dr, dc)``.
        scale: Optional ``(row_m, col_m)`` physical size of one step.
            A raster whose two pixel sizes differ scales the two axes
            differently, so the angle between two INTEGER steps is not
            the angle between the lines on the ground they describe;
            pass the scale and the angles come out physical. Omit it --
            the default -- on a square grid, where scaling both axes by
            the same number would change nothing but the round-off.
    """
    arr = np.asarray(dirs, dtype=np.float64).reshape(-1, 2)
    if scale is not None:
        arr = arr * np.asarray(scale, dtype=np.float64).reshape(1, 2)
    return np.arctan2(arr[:, 1], arr[:, 0])


def bezout(p: int, q: int) -> tuple[int, int]:
    """``(alpha, beta)`` with ``alpha * p + beta * q == 1``.

    Raises:
        ValueError: ``(p, q)`` is not primitive, so no such pair exists.
    """
    p, q = int(p), int(q)
    if math.gcd(abs(p), abs(q)) != 1:
        raise ValueError(f"direction ({p}, {q}) is not primitive")
    old_r, r = abs(p), abs(q)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quot = old_r // r
        old_r, r = r, old_r - quot * r
        old_s, s = s, old_s - quot * s
        old_t, t = t, old_t - quot * t
    a = old_s * (1 if p >= 0 else -1)
    b = old_t * (1 if q >= 0 else -1)
    if a * p + b * q != 1:
        raise AssertionError(f"bezout failed for ({p}, {q})")
    return int(a), int(b)


def ray_index(shape, p: int, q: int) -> np.ndarray:
    """Position along its own ray for every cell, as an int64 grid.

    ``t[x + (p, q)] == t[x] + 1`` exactly. The absolute value is
    arbitrary (it differs between rays); only differences along one ray
    are meaningful, which is all the segmented scans use it for.
    """
    alpha, beta = bezout(p, q)
    rows, cols = int(shape[0]), int(shape[1])
    r = np.arange(rows, dtype=np.int64).reshape(-1, 1)
    c = np.arange(cols, dtype=np.int64).reshape(1, -1)
    return alpha * r + beta * c


def shift(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """``out[x] = a[x - (dr, dc)]``, with ``fill`` where that is off-grid.

    Pulls from the cell ``(dr, dc)`` steps BACK, i.e. the cell one step
    earlier along the ray of direction ``(dr, dc)``.
    """
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    r0, r1 = max(0, dr), min(h, h + dr)
    c0, c1 = max(0, dc), min(w, w + dc)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


# ------------------------------------------------------------------- scans

def ray_prefix_sum(a, p: int, q: int) -> np.ndarray:
    """Inclusive cumulative sum along every ray of direction ``(p, q)``.

    ``P[x] = a[x] + a[x - d] + a[x - 2d] + ...`` over the cells that are
    still on the grid. A span cost between two cells of one ray is then
    the difference ``P[x] - P[y]``, evaluated in O(1) however long the
    span -- which is the whole reason the tower field is cheap.

    Built once per direction; the terrain does not change between sweeps.
    """
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    k = 1
    reach = max(a.shape[0], a.shape[1])
    while k < reach:
        out = out + shift(out, k * p, k * q, 0.0)
        k *= 2
    return out


def _segmented_block_scans(a, p, q, w, combine, neutral, track_arg):
    """van Herk's two half-block scans, by doubling instead of by a loop.

    Returns ``(g, h, ga, ha)`` where, with blocks of ``w`` consecutive
    positions along each ray,

    * ``g[x]`` reduces ``[block_start(x) .. x]`` (backward looking)
    * ``h[x]`` reduces ``[x .. block_end(x)]`` (forward looking)

    and ``ga`` / ``ha`` carry the ray index of the winning cell when
    ``track_arg``. Blocks are cut on ``t mod w``, which is consistent
    along a ray because ``t`` increments by one per step.
    """
    t = ray_index(a.shape, p, q)
    off = np.mod(t, w)

    g = a.copy()
    h = a.copy()
    ga = t.copy() if track_arg else None
    ha = t.copy() if track_arg else None

    k = 1
    while k < w:
        src = shift(g, k * p, k * q, neutral)
        take = (off >= k) & combine(src, g)
        g = np.where(take, src, g)
        if track_arg:
            ga = np.where(take, shift(ga, k * p, k * q, 0), ga)

        src = shift(h, -k * p, -k * q, neutral)
        take = ((off + k) <= (w - 1)) & combine(src, h)
        h = np.where(take, src, h)
        if track_arg:
            ha = np.where(take, shift(ha, -k * p, -k * q, 0), ha)
        k *= 2
    return g, h, ga, ha


def _ray_window(a, p, q, m_lo, m_hi, *, kind, return_arg):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {a.shape}")
    m_lo, m_hi = int(m_lo), int(m_hi)
    if m_lo < 0 or m_hi < m_lo:
        raise ValueError(f"need 0 <= m_lo <= m_hi, got {m_lo}, {m_hi}")
    p, q = int(p), int(q)
    if math.gcd(abs(p), abs(q)) != 1:
        raise ValueError(f"direction ({p}, {q}) is not primitive")

    w = m_hi - m_lo + 1
    if kind == "min":
        neutral = np.inf
        combine = np.less
    else:
        neutral = -np.inf
        combine = np.greater

    # van Herk reads the forward block scan at ``x - (w-1)*d``, which for
    # a cell near the grid edge lies OUTSIDE the grid even though cells
    # between it and x are inside. Reading a neutral there silently drops
    # those cells from the window (measured: a 17-cell row lost its last
    # two window members). Pad by the furthest offset any read can reach,
    # scan on the padded grid, crop back.
    rows, cols = a.shape
    pad = m_hi * max(abs(p), abs(q))
    ap = (np.pad(a, pad, mode="constant", constant_values=neutral)
          if pad else a)

    if w == 1:
        u = ap
        ua = ray_index(ap.shape, p, q) if return_arg else None
    else:
        g, h, ga, ha = _segmented_block_scans(
            ap, p, q, w, combine, neutral, return_arg)
        hs = shift(h, (w - 1) * p, (w - 1) * q, neutral)
        take = combine(hs, g)
        u = np.where(take, hs, g)
        ua = (np.where(take, shift(ha, (w - 1) * p, (w - 1) * q, 0), ga)
              if return_arg else None)

    out = shift(u, m_lo * p, m_lo * q, neutral)
    crop = (slice(pad, pad + rows), slice(pad, pad + cols))
    if not return_arg:
        return np.ascontiguousarray(out[crop])
    arg_t = shift(ua, m_lo * p, m_lo * q, 0)
    m = ray_index(ap.shape, p, q) - arg_t
    m = np.where(np.isfinite(out), m, -1).astype(np.int64)
    return np.ascontiguousarray(out[crop]), np.ascontiguousarray(m[crop])


def ray_window_min(a, p, q, m_lo, m_hi, *, return_arg: bool = False):
    """``min`` of ``a[x - m*d]`` over ``m`` in ``[m_lo, m_hi]``.

    Off-grid offsets contribute ``+inf``, so a cell with no admissible
    predecessor reads ``inf`` rather than silently borrowing a
    neighbour's value.

    Parameters:
        a: 2-D float array.
        p, q: A primitive integer direction.
        m_lo, m_hi: Inclusive offset range, in lattice steps.
        return_arg: Also return the winning ``m`` per cell (``-1`` where
            the minimum is ``inf``). Which ``m`` wins a tie is an
            implementation detail of the scan order; the VALUE never is.

    Returns:
        The windowed minimum, or ``(minimum, m)`` when ``return_arg``.
    """
    return _ray_window(a, p, q, m_lo, m_hi, kind="min", return_arg=return_arg)


def ray_window_max(a, p, q, m_lo, m_hi, *, return_arg: bool = False):
    """``max`` counterpart of :func:`ray_window_min`; off-grid is ``-inf``."""
    return _ray_window(a, p, q, m_lo, m_hi, kind="max", return_arg=return_arg)


def ray_run_length(blocked, p: int, q: int, cap: int) -> np.ndarray:
    """Consecutive unblocked cells ending at each cell, looking back.

    ``R[x]`` counts ``x`` itself, so ``R[x] == 0`` means ``x`` is blocked
    and ``R[x] == 3`` means ``x``, ``x - d`` and ``x - 2d`` are clear
    while ``x - 3d`` is blocked or off-grid. The longest span that can
    land on ``x`` without touching a blocked cell is therefore
    ``R[x] - 1`` steps.

    Parameters:
        blocked: Boolean grid, ``True`` where a span may not pass.
        p, q: A primitive integer direction.
        cap: Stop counting here. Runs longer than ``cap`` report ``cap``,
            which is all a caller with a bounded span needs and is what
            bounds the doubling loop.
    """
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    r = (~np.asarray(blocked, dtype=bool)).astype(np.int64)
    k = 1
    while k < cap:
        ext = shift(r, k * p, k * q, 0)
        r = np.where(r == k, r + ext, r)
        k *= 2
    return np.minimum(r, cap)


def circular_window_min(a, half_width: int) -> np.ndarray:
    """Sliding minimum over axis 0, wrapping around.

    ``out[i] = min(a[i - h], ..., a[i + h])`` with indices taken modulo
    ``len(a)``. Used on the DIRECTION axis, where "wrapping" is just the
    fact that angles are a circle.

    Only valid for a UNIFORM angular spacing, because the window is a
    constant number of indices.
    :class:`~pyorps.graph.tower_field.TowerFieldSolver` uses per-direction
    index sets instead, which stay exact for the non-uniform spacing of a
    primitive direction set.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    h = int(half_width)
    if h < 0:
        raise ValueError("half_width must be >= 0")
    if h == 0:
        return a.copy()
    if 2 * h + 1 >= n:
        return np.broadcast_to(a.min(axis=0), a.shape).copy()
    out = a.copy()
    for k in range(1, h + 1):
        out = np.minimum(out, np.roll(a, k, axis=0))
        out = np.minimum(out, np.roll(a, -k, axis=0))
    return out


# --------------------------------------------------------- brute references

def ray_prefix_sum_bruteforce(a, p, q):
    """Reference for :func:`ray_prefix_sum`; loops, for tests only."""
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(a)
    rows, cols = a.shape
    for r in range(rows):
        for c in range(cols):
            s = 0.0
            rr, cc = r, c
            while 0 <= rr < rows and 0 <= cc < cols:
                s += a[rr, cc]
                rr, cc = rr - p, cc - q
            out[r, c] = s
    return out


def _ray_window_bruteforce(a, p, q, m_lo, m_hi, reduce_, neutral):
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.shape, neutral, dtype=np.float64)
    rows, cols = a.shape
    for r in range(rows):
        for c in range(cols):
            best = neutral
            for m in range(m_lo, m_hi + 1):
                rr, cc = r - m * p, c - m * q
                if 0 <= rr < rows and 0 <= cc < cols:
                    best = reduce_(best, a[rr, cc])
            out[r, c] = best
    return out


def ray_window_min_bruteforce(a, p, q, m_lo, m_hi):
    """Reference for :func:`ray_window_min`; loops, for tests only."""
    return _ray_window_bruteforce(a, p, q, m_lo, m_hi, min, np.inf)


def ray_window_max_bruteforce(a, p, q, m_lo, m_hi):
    """Reference for :func:`ray_window_max`; loops, for tests only."""
    return _ray_window_bruteforce(a, p, q, m_lo, m_hi, max, -np.inf)

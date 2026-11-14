"""Conductor clearance of one straight overhead-line span, in plain Python.

``span_clearance`` repeats the test of the constrained search, ``_check_span_clearance_vh`` in
``pyorps/utils/_constrained_context.pyx``, cell for cell and in the same single-precision arithmetic, so that a
route can be checked independently of the kernels:

- A span runs from tower A to tower B along ONE lattice step ``(dr, dc)`` repeated ``n_steps`` times (the search
  forces a tower at every change of direction, so every span of a route is straight).
- The conductor hangs on a parabola: ``z(x) = z_A + (z_B - z_A) * x / L - w * x * (L - x) / (2 * T)`` with the
  attachment heights ``z_A = DEM(A) + height_a`` and ``z_B = DEM(B) + height_b``, the span length
  ``L = n_steps * step_distance``, the conductor weight ``w`` (N/m) and tension ``T`` (N).
- The conductor must stay at least ``min_clearance_m`` above the ground plus the obstacle height (DSM minus DEM)
  at every sampled cell. The samples are the cells the steps pass over, as :func:`tower_field.intermediate_offsets`
  gives them (``x`` is placed evenly between the step's two end points), and the cell each step ends on.

What the search does, and this check repeats on purpose:

- A span of one step or less is not checked (it is clear).
- The first lattice cell after tower A (``x`` = one step) is not sampled: the walk tests the cell a step ends on
  only from the second step on. The cell of tower B itself (``x = L``) is sampled.
- A span of tall towers is clear without a walk when ``min(height_a, height_b) - max_sag - min_clearance_m > 50 m``
  (``max_sag = w * L^2 / (8 * T)``): ground that rises more than that under the span is not seen.
- A span that leaves the raster is not clear.

The single-precision steps follow the C code the Cython compiler emits (a ``float`` numerator over a ``double``
denominator in both sag formulas); a conductor within about 1e-6 m of the limit can therefore still differ from the
kernel's answer by that much.

``strict=True`` removes those blind spots, for a report that must not repeat them: every cell of the span is sampled,
including the first one after tower A, a span of one step is checked too, and the tall-tower shortcut is not taken.
The default (``strict=False``) mirrors the kernel as described above, so that a check that has to agree with the
search keeps doing so. A span that passes the strict check passes the default check; the reverse does not hold, which
is exactly what the strict check is for.

The kernels sample a span at a connection point (the first span from a source connection height, the last span to a
target connection height) in full, as ``strict=True`` does: every cell after the start, the first one included, the
end cell, and a span of one step too. The first span from a source connection height also samples the start cell
itself, where the conductor leaves the structure: ``start_cell=True``. ``strict=True`` (with ``start_cell=True``
for the first span) therefore agrees with the search for the two end spans of a run with connection heights; for
the spans between two line towers the search keeps the sampling of the default mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyorps.core.exceptions import InvalidInputError
from pyorps.graph.tower_field import intermediate_offsets

__all__ = ["SpanClearance", "path_span_clearance", "span_clearance", "straight_step"]

_F32 = np.float32


@dataclass(frozen=True)
class SpanClearance:
    """Result of :func:`span_clearance`.

    Attributes:
        ok: The kernel's answer: True when the span is clear.
        checked: False when the kernel's own shortcuts call the span clear without sampling it (one step or less,
            or towers so tall that the walk is skipped); then the clearance values are None.
        worst_clearance_m: Smallest conductor height above ground plus obstacle over all sampled cells (the
            limit is ``min_clearance_m``), None when nothing was sampled.
        worst_cell: ``(row, col)`` of that sample.
        worst_offset_m: Distance of that sample from tower A along the span (``x`` of the formula).
        reason: Why ``ok`` is False when the span is not clear for another reason than the clearance (leaves the
            raster); empty otherwise.
    """

    ok: bool
    checked: bool
    worst_clearance_m: float | None = None
    worst_cell: tuple[int, int] | None = None
    worst_offset_m: float | None = None
    reason: str = ""


def _clearance_at(x, row, col, *, attach_a, attach_b, span_len, cond_w, cond_t, dem, obstacle):
    """Conductor height above ground plus obstacle at distance ``x`` from tower A (float32, like the kernel)."""
    chord_z = attach_a + (attach_b - attach_a) * x / span_len
    # a float numerator over a double denominator, stored as float (see the module docstring)
    sag_x = _F32(float(cond_w * x * (span_len - x)) / (2.0 * float(cond_t)))
    cond_z = chord_z - sag_x
    obs_z = obstacle[row, col] if obstacle is not None else _F32(0.0)
    return cond_z - dem[row, col] - obs_z


def span_clearance(dem, obstacle_heights, tower_a, step, n_steps, step_distance_m, height_a_m, height_b_m, *,
                   conductor_weight_per_m, conductor_tension_n, min_clearance_m, strict=False,
                   start_cell=False) -> SpanClearance:
    """Check the conductor clearance of one straight span the way the constrained search does (or strictly).

    Parameters:
        dem: (rows, cols) ground elevation in metres (the array the kernel gets: the DEM resampled to the cost
            raster).
        obstacle_heights: Same shape, obstacle height above the ground (DSM minus DEM), or None.
        tower_a: ``(row, col)`` of the first tower (or end point). Tower B is ``n_steps`` steps further.
        step: ``(dr, dc)`` lattice step of the span (one of the neighbourhood's steps).
        n_steps: Number of steps from A to B.
        step_distance_m: Length of one step in metres (``sqrt(dr^2 + dc^2) * cell_size``).
        height_a_m: Conductor attachment height above the ground at A.
        height_b_m: Same at B.
        conductor_weight_per_m: Conductor weight per metre (N/m).
        conductor_tension_n: Horizontal conductor tension (N).
        min_clearance_m: Smallest allowed conductor height above ground plus obstacle.
        strict: False (default) = the kernel's test, blind spots included (see the module docstring). True = sample
            every cell after tower A (the first one too), check a span of one step, and do not skip the walk for tall
            towers.
        start_cell: Also sample tower A's own cell (x = 0), where the conductor leaves the structure: its clearance is
            ``height_a_m`` minus the obstacle there. The search does this only for the first span from a source
            connection height. Implies ``strict=True``.

    Returns:
        A :class:`SpanClearance`; with ``strict=False`` ``.ok`` equals the kernel's answer for the same span.

    Raises:
        InvalidInputError: ``step`` is (0, 0), a raster is not two-dimensional or the two do not have one shape.
    """
    dr, dc = int(step[0]), int(step[1])
    if dr == 0 and dc == 0:
        raise InvalidInputError("span_clearance: step must not be (0, 0)")
    strict = bool(strict or start_cell)
    n_steps = int(n_steps)
    if n_steps < 1 or (n_steps == 1 and not strict):
        return SpanClearance(ok=True, checked=False)

    dem = np.asarray(dem, dtype=np.float32)
    if dem.ndim != 2:
        raise InvalidInputError(f"span_clearance: dem must be two-dimensional, got shape {dem.shape}")
    obstacle = None if obstacle_heights is None else np.asarray(obstacle_heights, dtype=np.float32)
    if obstacle is not None and obstacle.shape != dem.shape:
        raise InvalidInputError(
            f"span_clearance: obstacle_heights {obstacle.shape} and dem {dem.shape} must have one shape")
    rows, cols = dem.shape

    ra, ca = int(tower_a[0]), int(tower_a[1])
    rb, cb = ra + n_steps * dr, ca + n_steps * dc
    for name, (row, col) in (("A", (ra, ca)), ("B", (rb, cb))):
        if not (0 <= row < rows and 0 <= col < cols):
            return SpanClearance(ok=False, checked=True, reason=f"tower {name} lies outside the raster")

    step_distance = _F32(step_distance_m)
    cond_w, cond_t, min_clearance = _F32(conductor_weight_per_m), _F32(conductor_tension_n), _F32(min_clearance_m)
    height_a, height_b = _F32(height_a_m), _F32(height_b_m)
    span_len = _F32(n_steps) * step_distance
    attach_a = dem[ra, ca] + height_a
    attach_b = dem[rb, cb] + height_b

    max_sag = _F32(float(cond_w * span_len * span_len) / (8.0 * float(cond_t)))
    min_h = height_a if height_a < height_b else height_b
    if not strict and min_h - max_sag - min_clearance > 50.0:
        return SpanClearance(ok=True, checked=False)

    inter = intermediate_offsets(dr, dc)
    n_inter = len(inter)
    common = dict(attach_a=attach_a, attach_b=attach_b, span_len=span_len, cond_w=cond_w, cond_t=cond_t,
                  dem=dem, obstacle=obstacle)
    samples = []   # (clearance, row, col, x)
    if start_cell:
        samples.append((_clearance_at(_F32(0.0), ra, ca, **common), ra, ca, _F32(0.0)))
    walk_row, walk_col = ra, ca
    for step_no in range(n_steps):
        for i, (idr, idc) in enumerate(inter):
            row, col = walk_row + idr, walk_col + idc
            if not (0 <= row < rows and 0 <= col < cols):
                return SpanClearance(ok=False, checked=True, reason="the span leaves the raster")
            x = _F32(step_no) * step_distance + step_distance * (_F32(i + 1) / _F32(n_inter + 1))
            samples.append((_clearance_at(x, row, col, **common), row, col, x))
        walk_row += dr
        walk_col += dc
        if step_no > 0 or strict:
            x = _F32(step_no + 1) * step_distance
            samples.append((_clearance_at(x, walk_row, walk_col, **common), walk_row, walk_col, x))

    # the kernel's test is "clr < min_clearance fails", so a NaN sample (no elevation) does not fail here either
    worst, row, col, x = min(samples, key=lambda s: math.inf if math.isnan(float(s[0])) else float(s[0]))
    return SpanClearance(ok=not any(s[0] < min_clearance for s in samples), checked=True,
                         worst_clearance_m=float(worst), worst_cell=(int(row), int(col)), worst_offset_m=float(x))


def straight_step(rows, cols, pos_a: int, pos_b: int) -> tuple[int, int] | None:
    """The one lattice step ``(dr, dc)`` that a route repeats from position ``pos_a`` to ``pos_b``, or None.

    ``rows`` and ``cols`` are the row and column of every cell of the route. None when the two positions are the
    same or when the stretch between them changes direction (a span with a bend is not a span of the kernel's
    model).
    """
    if pos_b <= pos_a:
        return None
    d_rows = np.diff(np.asarray(rows[pos_a:pos_b + 1], dtype=np.int64))
    d_cols = np.diff(np.asarray(cols[pos_a:pos_b + 1], dtype=np.int64))
    if not (np.all(d_rows == d_rows[0]) and np.all(d_cols == d_cols[0])):
        return None
    return int(d_rows[0]), int(d_cols[0])


def path_span_clearance(dem, obstacle_heights, rows, cols, pos_a: int, pos_b: int, cell_size, height_a_m, height_b_m, *,
                        conductor_weight_per_m, conductor_tension_n, min_clearance_m,
                        strict=False, start_cell=False) -> SpanClearance | None:
    """:func:`span_clearance` for the span of a route between its positions ``pos_a`` and ``pos_b``.

    ``rows``/``cols`` are the route's cells. The step and its length follow from the route
    (``sqrt(dr^2 + dc^2) * cell_size``, as the search computes them). Returns None when the stretch is not
    straight (see :func:`straight_step`). ``strict`` and ``start_cell`` are passed on (see :func:`span_clearance`).
    """
    step = straight_step(rows, cols, pos_a, pos_b)
    if step is None:
        return None
    step_distance = math.sqrt(step[0] ** 2 + step[1] ** 2) * float(cell_size)
    return span_clearance(
        dem, obstacle_heights, (int(rows[pos_a]), int(cols[pos_a])), step, pos_b - pos_a, step_distance,
        height_a_m, height_b_m, conductor_weight_per_m=conductor_weight_per_m,
        conductor_tension_n=conductor_tension_n, min_clearance_m=min_clearance_m, strict=strict,
        start_cell=start_cell)

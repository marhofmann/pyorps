"""Raster primitives of the multi-source drain (plan rev. 5, sections 3.2 and 3.5).

:func:`drain` is ``min_k labels[k] + d_w(seeds[k], v)`` on a cost raster: the
Cython multi-source solver when the built extension has it, otherwise a
pure-Python heap Dijkstra with the kernel's exact arithmetic.
:func:`step_table` holds the step set the way the kernels hold it, and
:func:`boundary_cells` finds the cells of a window from which some step
leaves it. They live in the core (``pyorps.graph``) because the mixed-leg
solver and the collector engines use them; the window certificates of
:mod:`pyorps.certify.windows` build on them.

All values here are in CELL units, like the Dijkstra kernels; multiply by
the cell size for EUR. ``EXCLUDED`` is the uint16 cell value (65535) that is
neither entered nor crossed.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = [
    "EXCLUDED",
    "boundary_cells",
    "drain",
    "step_table",
]

EXCLUDED = 65535


def _factor32(dr: int, dc: int, n: int) -> float:
    d = np.sqrt(np.float32(dr * dr + dc * dc), dtype=np.float32)
    return float(d / np.float32(2.0 + n))


def step_table(steps) -> list[tuple[int, int, list[tuple[int, int]], float,
                                    float]]:
    """``(dr, dc, intermediates, float32 cost factor, length)`` per step,
    exactly as the Cython kernels hold them (plan D8)."""
    from pyorps.graph.tower_field import intermediate_offsets
    out = []
    for dr, dc in np.asarray(steps)[:, :2]:
        dr, dc = int(dr), int(dc)
        inter = intermediate_offsets(dr, dc)
        out.append((dr, dc, inter, _factor32(dr, dc, len(inter)),
                    math.hypot(dr, dc)))
    return out


def boundary_cells(inside: np.ndarray, steps) -> np.ndarray:
    """``dW``: cells of ``W`` with a step whose end or an intermediate cell
    lies outside ``W`` (or off the raster)."""
    inside = np.asarray(inside, dtype=bool)
    rows, cols = inside.shape
    out = np.zeros_like(inside)
    padded = np.zeros((rows + 8, cols + 8), dtype=bool)
    padded[4:4 + rows, 4:4 + cols] = inside
    for dr, dc, inter, _f, _l in step_table(steps):
        for a, b in [(dr, dc)] + list(inter):
            if abs(a) > 4 or abs(b) > 4:
                raise ValueError("steps longer than 4 cells are not supported")
            shifted = padded[4 + a:4 + a + rows, 4 + b:4 + b + cols]
            out |= inside & ~shifted
    return out


def drain(values: np.ndarray, steps, seeds, labels, *, length_rate=0.0,
          weight_mult=1.0, excluded: np.ndarray | None = None,
          no_transit=None, engine: str = "auto", return_prev: bool = False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """``min_k labels[k] + d_w(seeds[k], v)`` for every cell, in cell units.

    ``excluded`` cells (and ``values == 65535``) are neither entered nor
    crossed. Uses ``MultiSourceSolver.solve_stream`` when the built
    extension has it (``engine="kernel"``), otherwise a pure-Python heap
    Dijkstra with the kernel's exact arithmetic (``engine="python"``).

    With ``return_prev`` it returns ``(dist, prev)``: ``prev`` holds, per
    flat cell, the cell the last step came from, ``-1`` where the seed's
    own label won (a seed wins ties) and where nothing arrived -- the
    trace code of the drain (plan section 3.2, item 7).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    vals = np.asarray(values)
    ras = np.asarray(vals, dtype=np.uint16).copy()
    if excluded is not None:
        ras[np.asarray(excluded, dtype=bool)] = EXCLUDED
    seeds = np.asarray(seeds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    order = np.argsort(labels, kind="stable")
    seeds, labels = seeds[order], labels[order]
    use_kernel = engine == "kernel"
    if engine == "auto":
        try:
            from pyorps.utils import _dijkstra as dj
            use_kernel = hasattr(dj.MultiSourceSolver, "solve_stream")
        except ImportError:
            use_kernel = False
    if use_kernel:
        from pyorps.utils import _dijkstra as dj
        st = np.asarray(steps, dtype=np.int8)
        solver = dj.make_multi_source_solver(ras, st, lean=True)
        solver.solve_stream(seeds, labels, length_rate=float(length_rate),
                            weight_mult=float(weight_mult),
                            no_transit=no_transit, keep_prev=return_prev)
        dist = solver.dist_array().reshape(ras.shape)
        if return_prev:
            prev = solver.prev_array().astype(np.int64)
            prev[~np.isfinite(dist.ravel())] = -1
            return dist, prev
        return dist
    return _python_drain(ras, steps, seeds, labels, float(length_rate),
                         float(weight_mult), no_transit, return_prev)


def _python_drain(ras, steps, seeds, labels, length_rate, weight_mult,
                  no_transit, return_prev=False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    rows, cols = ras.shape
    blocked = ras == EXCLUDED
    nt = np.zeros_like(blocked)
    if no_transit is not None:
        nt.ravel()[np.asarray(no_transit, dtype=np.int64)] = True
    table = step_table(steps)
    dist = np.full(rows * cols, math.inf)
    prev = np.full(rows * cols, -1, dtype=np.int64)
    done = np.zeros(rows * cols, dtype=bool)
    heap = []
    for c, lab in zip(seeds.tolist(), labels.tolist()):
        if blocked.ravel()[c]:
            continue
        if lab < dist[c]:
            dist[c] = lab
            heapq.heappush(heap, (lab, c))
    while heap:
        d, u = heapq.heappop(heap)
        if done[u] or d > dist[u]:
            continue
        done[u] = True
        ur, uc = divmod(u, cols)
        for dr, dc, inter, fac, ln in table:
            vr, vc = ur + dr, uc + dc
            if not (0 <= vr < rows and 0 <= vc < cols):
                continue
            if blocked[vr, vc] or nt[vr, vc]:
                continue
            ok = True
            s = float(ras[ur, uc]) + float(ras[vr, vc])
            for a, b in inter:
                ir, ic = ur + a, uc + b
                if (not (0 <= ir < rows and 0 <= ic < cols)
                        or blocked[ir, ic] or nt[ir, ic]):
                    ok = False
                    break
                s += float(ras[ir, ic])
            if not ok:
                continue
            v = vr * cols + vc
            nd = d + weight_mult * (s * fac) + length_rate * ln
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    dist[~done] = math.inf
    if return_prev:
        prev[~done] = -1
        return dist.reshape(rows, cols), prev
    return dist.reshape(rows, cols)

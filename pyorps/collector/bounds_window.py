"""Window decomposition of the rho-hat DW bound for roots far outside the turbine window (paper math.md M7.6d).

For a root ``g`` outside a window ``W`` that contains every turbine, split any collector tree ``T`` into the edges
inside ``W`` and the edges outside. The two parts are bounded separately and add up:

* **inside:** ``T`` cut at the window boundary is a forest in ``W`` whose components each reach a boundary cell (the
  tree has to connect them to ``g``). Its cost is at least the relaxed subset DW of :mod:`pyorps.collector.bounds` with
  a *virtual root* that every boundary cell reaches at zero cost, no root bays and one panel per turbine:
  ``Win = sum_turbine panels + min over partitions {Y} of sum_Y min_{x in boundary} arr_Y(x)``.
  It is one number, computed once per window (:func:`rho_hat_exit_value`).
* **outside:** the edges outside ``W`` connect ``g`` to the boundary, so their trench weight is at least the trench
  distance from the boundary to ``g`` (one zero-seeded drain), and every turbine's cable runs at least ``dist(g, W)``
  outside, so the cables cost at least ``alpha * n * dist(g, W)`` with ``alpha = min_S rho_hat(S) / |S|``
  (:func:`cable_rate_floor`).

``LB(g) = Win + trench_dist(boundary, g) + alpha * n * dist(g, W) + bay`` is admissible for every ``g`` outside ``W``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyorps.collector.bounds import rho_hat
from pyorps.collector.model import INF, CollectorModel, all_submasks, bits
from pyorps.core.exceptions import InvalidInputError

__all__ = ["cable_rate_floor", "exit_cells", "rho_hat_exit_value"]


def cable_rate_floor(model: CollectorModel) -> float:
    """``alpha = min over non-empty S of rho_hat(S) / |S|`` (EUR per metre and turbine).

    Every turbine's cable has that rate at least.
    """
    rho, _ = rho_hat(model)
    vals = [rho[s] / bin(s).count("1") for s in range(1, model.full + 1) if math.isfinite(rho[s])]
    return float(min(vals)) if vals else INF


def exit_cells(values: np.ndarray, steps) -> np.ndarray:
    """Flat indices of the allowed cells of the window that have a step leaving it (the window boundary)."""
    from pyorps.graph.drain import EXCLUDED, boundary_cells

    v = np.asarray(values)
    inside = np.ones(v.shape, dtype=bool)
    b = boundary_cells(inside, steps) & (v != EXCLUDED)
    return np.flatnonzero(b.ravel())


@dataclass(frozen=True, eq=False)
class _ExitCtx:
    """The read-only inputs shared by every subset step of :func:`rho_hat_exit_value`."""

    v: np.ndarray
    steps: object
    rows: int
    cols: int
    N: int
    turbines: list
    t_idx: np.ndarray
    is_turb: np.ndarray
    w_arr: object
    mult: float
    no_transit: np.ndarray
    ex: np.ndarray
    ex_turb: np.ndarray
    rho: object
    drain: Callable
    drain_engine: str


def _one_step(F, r, c: _ExitCtx):
    """Arrival field of one exact kernel step from the field ``F`` at length rate ``r``."""
    rows, cols = c.rows, c.cols
    Fg = F.reshape(rows, cols)
    out = np.full((rows, cols), INF)
    for dr, dc, w, ln in c.w_arr:
        cand = Fg + c.mult * w + r * ln
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        dst = out[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        np.minimum(dst, cand[r0:r1, c0:c1], out=dst)
    return out.ravel()


def _split_exit_cells(v, steps, exits, is_turb, excluded):
    """The allowed exit cells, split into the non-turbine ones and the turbine ones."""
    ex = np.asarray(exit_cells(v, steps) if exits is None else exits, dtype=np.int64)
    ex = ex[v.ravel()[ex] != excluded]
    ex_turb = ex[is_turb[ex]]                      # a turbine on the boundary can be the top of its component
    ex = ex[~is_turb[ex]]
    if ex.size == 0 and ex_turb.size == 0:
        raise InvalidInputError("the window has no exit cell")
    return ex, ex_turb


def _split_cost(X: int, i: int, arr_t: dict, G: dict) -> float:
    """Cheapest way to serve the turbine set ``X`` of the component below turbine ``i`` (low bit of ``X`` first)."""
    low = X & -X
    best = INF
    for sub in all_submasks(X ^ low):
        Y = sub | low
        a = arr_t[Y][i] if Y in arr_t else INF
        if a < INF and G.get(X ^ Y, INF) < INF:
            best = min(best, a + G[X ^ Y])
    return best


def _turbine_seed_cost(rest: int, i: int, arr_t: dict) -> float:
    """Seed value of turbine ``i`` for a subset whose other turbines are ``rest``."""
    G = {0: 0.0}
    for X in sorted(all_submasks(rest), key=lambda x: bin(x).count("1")):
        if X == 0:
            continue
        G[X] = _split_cost(X, i, arr_t, G)
    return G[rest]


def _subset_seed(S: int, c: _ExitCtx, F: dict, arr_t: dict) -> np.ndarray:
    """Seed field of the subset ``S``: turbine seeds, then the best two-way split of ``S`` at every cell."""
    seed = np.full(c.N, INF)
    for i in bits(S):
        t = c.turbines[i]
        rest = S & ~(1 << i)
        seed[t] = min(seed[t], _turbine_seed_cost(rest, i, arr_t))
    low = S & -S
    for sub in all_submasks(S ^ low):
        S1 = sub | low
        if S1 == S:
            continue
        np.minimum(seed, np.where(c.is_turb, INF, F[S1] + F[S ^ S1]), out=seed)
    return seed


def _subset_fields(S: int, seed: np.ndarray, c: _ExitCtx):
    """The drained field ``dist`` of the subset ``S`` and its arrival field ``a`` (one exact kernel step)."""
    r = c.rho[S]
    cells = np.flatnonzero(np.isfinite(seed))
    if r < INF and cells.size:
        dist = c.drain(c.v, c.steps, cells, seed[cells], length_rate=r, weight_mult=c.mult,
                       no_transit=c.no_transit, engine=c.drain_engine).ravel()
        return dist, _one_step(dist, r, c)
    return seed, np.full(c.N, INF)


def _min_at(field: np.ndarray, idx: np.ndarray) -> float:
    """Minimum of ``field`` over ``idx`` (``INF`` when ``idx`` is empty)."""
    return float(field[idx].min()) if idx.size else INF


def _combine_exit_values(h: dict, full: int) -> float:
    """Best partition of the full turbine set into components, summing their exit values ``h``."""
    H = {0: 0.0}
    for X in range(1, full + 1):
        low = X & -X
        best = INF
        for sub in all_submasks(X ^ low):
            Y = sub | low
            if h[Y] < INF and H[X ^ Y] < INF:
                best = min(best, h[Y] + H[X ^ Y])
        H[X] = best
    return H[full]


def rho_hat_exit_value(values, steps, cell_m: float, turbines, model: CollectorModel, *, trench_mult: float = 1.0,
                       drain_engine: str = "auto", exits=None) -> float:
    """``Win``: the relaxed subset DW of the window with a zero-cost virtual root on the boundary cells (EUR).

    Same recursion as :func:`pyorps.collector.bounds.rho_hat_lower_bound_raster` (one seeded drain per turbine subset,
    turbines no-transit, arrivals by one exact kernel step); the arrival fields are kept only at the exit cells; a
    turbine on the boundary can itself be the top of its component (its own label counts).
    """
    from pyorps.collector.raster import _step_weights
    from pyorps.graph.drain import EXCLUDED, drain

    v = np.asarray(values)
    rows, cols = v.shape
    turbines = [int(t) for t in turbines]
    tmask = np.zeros(v.shape, dtype=bool)
    tmask.ravel()[turbines] = True
    is_turb = tmask.ravel()
    w_arr = _step_weights(v, steps, tmask, into_turbines=True)
    cell = float(cell_m)
    mult = float(trench_mult)
    no_transit = np.flatnonzero(is_turb)
    ex, ex_turb = _split_exit_cells(v, steps, exits, is_turb, EXCLUDED)
    rho, _nbm = rho_hat(model)
    t_idx = np.asarray(turbines, dtype=np.int64)
    c = _ExitCtx(v, steps, rows, cols, v.size, turbines, t_idx, is_turb, w_arr, mult, no_transit, ex, ex_turb, rho,
                 drain, drain_engine)

    full = model.full
    F: dict[int, np.ndarray] = {}
    arr_t: dict[int, np.ndarray] = {}              # arrival value at each turbine cell (all the turbine seeds need)
    h: dict[int, float] = {}
    order = sorted(range(1, full + 1), key=lambda s: (bin(s).count("1"), s))
    for S in order:
        dist, a = _subset_fields(S, _subset_seed(S, c, F, arr_t), c)
        F[S] = dist
        arr_t[S] = a[t_idx].copy()
        h[S] = min(_min_at(a, ex), _min_at(dist, ex_turb))
    return (_combine_exit_values(h, full) + model.turbine_panel_eur * model.n / cell) * cell

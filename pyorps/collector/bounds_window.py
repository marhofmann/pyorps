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

import numpy as np

from pyorps.collector.bounds import rho_hat
from pyorps.collector.model import INF, CollectorModel, all_submasks, bits

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
    from pyorps.certify.windows import EXCLUDED, boundary_cells

    v = np.asarray(values)
    inside = np.ones(v.shape, dtype=bool)
    b = boundary_cells(inside, steps) & (v != EXCLUDED)
    return np.flatnonzero(b.ravel())


def rho_hat_exit_value(values, steps, cell_m: float, turbines, model: CollectorModel, *, trench_mult: float = 1.0,
                       drain_engine: str = "auto", exits=None) -> float:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """``Win``: the relaxed subset DW of the window with a zero-cost virtual root on the boundary cells (EUR).

    Same recursion as :func:`pyorps.collector.bounds.rho_hat_lower_bound_raster` (one seeded drain per turbine subset,
    turbines no-transit, arrivals by one exact kernel step); the arrival fields are kept only at the exit cells; a
    turbine on the boundary can itself be the top of its component (its own label counts).
    """
    from pyorps.certify.windows import EXCLUDED, drain
    from pyorps.collector.raster import _step_weights

    v = np.asarray(values)
    rows, cols = v.shape
    N = v.size
    turbines = [int(t) for t in turbines]
    tmask = np.zeros(v.shape, dtype=bool)
    tmask.ravel()[turbines] = True
    is_turb = tmask.ravel()
    w_arr = _step_weights(v, steps, tmask, into_turbines=True)
    cell = float(cell_m)
    mult = float(trench_mult)
    no_transit = np.flatnonzero(is_turb)
    ex = np.asarray(exit_cells(v, steps) if exits is None else exits, dtype=np.int64)
    ex = ex[v.ravel()[ex] != EXCLUDED]
    ex_turb = ex[is_turb[ex]]                      # a turbine on the boundary can be the top of its component
    ex = ex[~is_turb[ex]]
    if ex.size == 0 and ex_turb.size == 0:
        raise ValueError("the window has no exit cell")
    rho, _nbm = rho_hat(model)

    def one_step(F, r):
        Fg = F.reshape(rows, cols)
        out = np.full((rows, cols), INF)
        for dr, dc, w, ln in w_arr:
            cand = Fg + mult * w + r * ln
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            dst = out[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            np.minimum(dst, cand[r0:r1, c0:c1], out=dst)
        return out.ravel()

    full = model.full
    F: dict[int, np.ndarray] = {}
    arr_t: dict[int, np.ndarray] = {}              # arrival value at each turbine cell (all the turbine seeds need)
    t_idx = np.asarray(turbines, dtype=np.int64)
    h: dict[int, float] = {}
    order = sorted(range(1, full + 1), key=lambda s: (bin(s).count("1"), s))
    for S in order:
        seed = np.full(N, INF)
        for i in bits(S):
            t = turbines[i]
            rest = S & ~(1 << i)
            G = {0: 0.0}
            for X in sorted(all_submasks(rest), key=lambda x: bin(x).count("1")):
                if X == 0:
                    continue
                low = X & -X
                best = INF
                for sub in all_submasks(X ^ low):
                    Y = sub | low
                    a = arr_t[Y][i] if Y in arr_t else INF
                    if a < INF and G.get(X ^ Y, INF) < INF:
                        best = min(best, a + G[X ^ Y])
                G[X] = best
            seed[t] = min(seed[t], G[rest])
        low = S & -S
        for sub in all_submasks(S ^ low):
            S1 = sub | low
            if S1 == S:
                continue
            np.minimum(seed, np.where(is_turb, INF, F[S1] + F[S ^ S1]), out=seed)
        r = rho[S]
        cells = np.flatnonzero(np.isfinite(seed))
        if r < INF and cells.size:
            dist = drain(v, steps, cells, seed[cells], length_rate=r, weight_mult=mult, no_transit=no_transit,
                         engine=drain_engine).ravel()
            a = one_step(dist, r)
        else:
            dist = seed
            a = np.full(N, INF)
        F[S] = dist
        arr_t[S] = a[t_idx].copy()
        h[S] = min(float(a[ex].min()) if ex.size else INF, float(dist[ex_turb].min()) if ex_turb.size else INF)
    H = {0: 0.0}
    for X in range(1, full + 1):
        low = X & -X
        best = INF
        for sub in all_submasks(X ^ low):
            Y = sub | low
            if h[Y] < INF and H[X ^ Y] < INF:
                best = min(best, h[Y] + H[X ^ Y])
        H[X] = best
    return (H[full] + model.turbine_panel_eur * model.n / cell) * cell

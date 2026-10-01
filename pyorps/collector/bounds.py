"""Admissible lower bound on the collector field: the rho-hat DW (plan D4a).

Stage 2 of the plan prunes UW positions with a cheap bound that never
exceeds the exact collector cost. This is the relaxed subset
Dreyfus--Wagner of the appendix (Part IV, "Relaxed subset DW", design A;
verifier fix V-03 for the bay term), extended to the 2026-09-24 rules:

* every trench step carrying the turbine set ``S`` below it costs at least
  ``c(e) + rho_hat(S) l(e)``, where ``rho_hat(S)`` is the cheapest per-metre
  rate of ANY label with that set: the cheapest partition of ``S`` into
  systems, each with the cheapest feasible type for its ``p`` (down-systems
  only add cost, since rates are non-decreasing in the cable count);
* junctions and stations are free, turbines may take any number of
  children (no panel or switchgear limit), and each turbine pays one panel
  (its out-cable has at least one conductor);
* at the root, a feeder edge carrying ``Y`` pays at least
  ``bay_eur * nb_min(Y)`` bays, ``nb_min(Y)`` the fewest conductors of any
  bay-feasible label with set ``Y`` -- the per-part bay floor, which stays
  admissible because the model has no junction on the UW cell (V-03, V-08).

Each relaxation only enlarges the feasible set or lowers a price, and the
subset DW is exact for the relaxed problem (Dreyfus--Wagner, Erickson,
Monma and Veinott), so ``lower_bound(g) <= MV(g)`` at every root, in both
the single-root and the field model (collectors may cross ``g``).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from pyorps.collector.model import INF, CollectorModel, all_submasks, bits
from pyorps.collector.reference import CollectorGraph

__all__ = ["completion_bounds_raster", "mu_star", "rho_hat",
           "rho_hat_lower_bound", "rho_hat_lower_bound_raster"]


def _block_cost(model: CollectorModel, mask: int, p: int, m: int) -> float:
    """Cheapest type for one system of ``p`` cables carrying ``mask`` in a
    trench of ``m`` cables; ``inf`` if none is feasible."""
    best = INF
    for ti in range(len(model.types)):
        if model.feasible(mask, (ti, p), m):
            best = min(best, model.rho(mask, (ti, p)))
    return best


def rho_hat(model: CollectorModel) -> tuple[list[float], list[int]]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """``rho_hat(S)`` and ``nb_min(S)`` for every turbine bitmask ``S``.

    ``rho_hat(S)``: min over partitions of ``S`` into systems and over their
    ``p`` of ``sum rho + sigma (m - 1)``, every system feasible at the
    total cable count ``m``. ``nb_min(S)``: the fewest conductors of such a
    partition whose systems all pass the bay rating (``inf`` if none).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    full = model.full
    rho = [INF] * (full + 1)
    nbm = [math.inf] * (full + 1)
    m_cap = model.m_cap()
    for S in range(1, full + 1):
        atoms = [1 << i for i in bits(S)]
        best, best_nb = INF, math.inf
        for part in _partitions(atoms):
            blocks = [sum(b) for b in part]
            k = len(blocks)
            if model.m_max is not None and k > model.m_max:
                continue
            # choose p per block; m = sum p
            for ps in _p_choices(k, model.p_max, m_cap):
                m = sum(ps)
                total = model.sigma_eur_per_m * (m - 1)
                ok = True
                for mask, p in zip(blocks, ps):
                    c = _block_cost(model, mask, p, m)
                    if c == INF:
                        ok = False
                        break
                    total += c
                if not ok:
                    continue
                best = min(best, total)
                if all(model.bay_ok(mask, (0, p))
                       for mask, p in zip(blocks, ps)):
                    best_nb = min(best_nb, m)
        rho[S] = best
        nbm[S] = best_nb
    rho[0] = 0.0
    return rho, nbm


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


def _p_choices(k: int, p_max: int, m_cap: int):
    """Every tuple of k parallel counts in 1..p_max with sum <= m_cap."""
    def rec(i, acc, s):
        if i == k:
            yield tuple(acc)
            return
        for p in range(1, p_max + 1):
            if s + p > m_cap:
                break
            yield from rec(i + 1, acc + [p], s + p)
    yield from rec(0, [], 0)


def rho_hat_lower_bound(graph: CollectorGraph, turbines, model: CollectorModel
                        ) -> np.ndarray:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Admissible lower bound on ``MV(g)`` for every node ``g``.

    ``inf`` at turbine nodes and where even the relaxation is infeasible.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    turbines = [int(t) for t in turbines]
    N = graph.n_nodes
    is_turb = np.zeros(N, dtype=bool)
    is_turb[turbines] = True
    rho, nbm = rho_hat(model)
    full = model.full
    F: dict[int, np.ndarray] = {}
    arr: dict[int, np.ndarray] = {}
    order = sorted(range(1, full + 1), key=lambda s: (bin(s).count("1"), s))
    for S in order:
        seed = np.full(N, INF)
        # turbine seeds: t with any grouping of arriving children
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
                    a = arr[Y][t] if Y in arr else INF
                    if a < INF and G.get(X ^ Y, INF) < INF:
                        best = min(best, a + G[X ^ Y])
                G[X] = best
            seed[t] = min(seed[t], G[rest])
        # merges at non-turbine nodes, junctions free
        low = S & -S
        for sub in all_submasks(S ^ low):
            S1 = sub | low
            if S1 == S:
                continue
            np.minimum(seed, np.where(is_turb, INF, F[S1] + F[S ^ S1]),
                       out=seed)
        r = rho[S]
        dist = seed.copy()
        if r < INF:
            heap = [(dist[v], v) for v in np.flatnonzero(np.isfinite(dist))]
            heapq.heapify(heap)
            while heap:
                d, v = heapq.heappop(heap)
                if d > dist[v]:
                    continue
                for (w, ei) in graph.adj[v]:
                    if is_turb[w]:
                        continue
                    _, _, c, ln = graph.edges[ei]
                    nd = d + c + r * ln
                    if nd < dist[w]:
                        dist[w] = nd
                        heapq.heappush(heap, (nd, int(w)))
        F[S] = dist
        a = np.full(N, INF)
        if r < INF:
            for y in range(N):
                for (x, ei) in graph.adj[y]:
                    if dist[x] < INF:
                        _, _, c, ln = graph.edges[ei]
                        a[y] = min(a[y], dist[x] + c + r * ln)
        arr[S] = a
    H = {0: np.zeros(N)}
    for X in range(1, full + 1):
        low = X & -X
        h = np.full(N, INF)
        for sub in all_submasks(X ^ low):
            Y = sub | low
            if nbm[Y] == math.inf:
                continue
            np.minimum(h, arr[Y] + model.bay_eur * nbm[Y] + H[X ^ Y], out=h)
        H[X] = h
    lb = H[full] + model.turbine_panel_eur * model.n
    lb[is_turb] = INF
    return lb


def rho_hat_lower_bound_raster(values, steps, cell_m: float, turbines,
                               model: CollectorModel, *,
                               trench_mult: float = 1.0,
                               drain_engine: str = "auto") -> np.ndarray:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """:func:`rho_hat_lower_bound` on a raster window (plan D10, Stage 2).

    The same relaxed subset DW, laid out like
    :class:`~pyorps.collector.raster.RasterCollector`: one seeded drain per
    turbine subset (``2**n - 1`` drains, 127 for the case study) with
    ``weight_mult = 1 + omega`` and ``length_rate = rho_hat(S)``, the
    turbines no-transit, arrivals by one exact kernel step. Admissible
    against the raster engine at every cell; equal to the graph version on
    :func:`~pyorps.collector.raster.graph_from_raster` of the same window.

    Returns:
        EUR per root cell, ``inf`` at turbines, excluded cells and where
        even the relaxation is infeasible.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps.certify.windows import EXCLUDED, drain
    from pyorps.collector.raster import _step_weights

    v = np.asarray(values)
    rows, cols = v.shape
    N = v.size
    turbines = [int(t) for t in turbines]
    tmask = np.zeros(v.shape, dtype=bool)
    tmask.ravel()[turbines] = True
    is_turb = tmask.ravel()
    excl = (v == EXCLUDED).ravel()
    w_arr = _step_weights(v, steps, tmask, into_turbines=True)
    cell = float(cell_m)
    mult = float(trench_mult)
    no_transit = np.flatnonzero(is_turb)
    rho, nbm = rho_hat(model)

    def one_step(F, r):
        """``min over steps u -> y`` of ``F[u] + mult * w + r * len``."""
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
    arr: dict[int, np.ndarray] = {}
    order = sorted(range(1, full + 1), key=lambda s: (bin(s).count("1"), s))
    for S in order:
        seed = np.full(N, INF)
        for i in bits(S):
            t = turbines[i]
            rest = S & ~(1 << i)
            G = {0: 0.0}
            for X in sorted(all_submasks(rest),
                            key=lambda x: bin(x).count("1")):
                if X == 0:
                    continue
                low = X & -X
                best = INF
                for sub in all_submasks(X ^ low):
                    Y = sub | low
                    a = arr[Y][t] if Y in arr else INF
                    if a < INF and G.get(X ^ Y, INF) < INF:
                        best = min(best, a + G[X ^ Y])
                G[X] = best
            seed[t] = min(seed[t], G[rest])
        low = S & -S
        for sub in all_submasks(S ^ low):
            S1 = sub | low
            if S1 == S:
                continue
            np.minimum(seed, np.where(is_turb, INF, F[S1] + F[S ^ S1]),
                       out=seed)
        r = rho[S]
        cells = np.flatnonzero(np.isfinite(seed))
        if r < INF and cells.size:
            dist = drain(v, steps, cells, seed[cells], length_rate=r,
                         weight_mult=mult, no_transit=no_transit,
                         engine=drain_engine).ravel()
            arr[S] = one_step(dist, r)
        else:
            dist = seed
            arr[S] = np.full(N, INF)
        F[S] = dist
    H = {0: np.zeros(N)}
    for X in range(1, full + 1):
        low = X & -X
        h = np.full(N, INF)
        for sub in all_submasks(X ^ low):
            Y = sub | low
            if nbm[Y] == math.inf:
                continue
            np.minimum(h, arr[Y] + model.bay_eur * nbm[Y] / cell + H[X ^ Y],
                       out=h)
        H[X] = h
    lb = (H[full] + model.turbine_panel_eur * model.n / cell) * cell
    lb[is_turb | excl] = INF
    return lb.reshape(v.shape)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite


def mu_star(model: CollectorModel) -> list[float]:
    """``mu*(k)``: the cheapest per-metre rate of any label carrying ``k``
    turbines, ``min over |X| = k`` of ``rho_hat(X)`` (index 0 unused)."""
    rho, _nbm = rho_hat(model)
    mu = [INF] * (model.n + 1)
    for S in range(1, model.full + 1):
        k = bin(S).count("1")
        mu[k] = min(mu[k], rho[S])
    mu[0] = 0.0
    return mu


def completion_bounds_raster(values, steps, cell_m: float, turbines,
                             model: CollectorModel, *,
                             root_cost=None, trench_mult: float = 1.0,
                             drain_engine: str = "auto") -> dict:
    """Consistent admissible completion bounds ``Out_k`` (plan section 3.2,
    item 6; appendix Part IV, design B's aggregated form).

    ``Out_k(v)`` bounds from below the cost of completing ANY partial
    design that carries ``k`` turbines at cell ``v`` into a full design --
    the other turbines, the way to a root and the root's own cost
    ``root_cost`` -- so a label may be dropped where
    ``F(v) + Out_k(v) > budget``. The recursion, 2n drains:

        In_1(v)  = min_t d_mu*(1)(t, v)
        In_j     = drain_mu*(j)( min_{a+b=j} In_a + In_b )
        Out_n    = drain_mu*(n)( root_cost )
        Out_k    = drain_mu*(k)( min_j In_j + Out_{k+j} )

    with step weight ``(1 + omega) w + mu*(k) len``. Merges are allowed at
    every cell, turbines included, and the drains may cross turbines: that
    is the relaxation that contains the transposed turbine rule (verifier
    V-04 -- a chain t2 -> t1 -> g must not be cut at t1). Junctions,
    stations, panels and bays are free.

    Parameters:
        root_cost: EUR per cell for placing the UW there (the per-position
            budget term B(g)); ``None`` for 0 everywhere. ``inf`` marks a
            cell that cannot be the root.

    Returns:
        ``{k: array}`` in CELL units (EUR / ``cell_m``), ``k = 1..n``.
    """
    from pyorps.certify.windows import EXCLUDED, drain

    v = np.asarray(values)
    N = v.size
    cell = float(cell_m)
    turbines = [int(t) for t in turbines]
    n = model.n
    mu = mu_star(model)
    mult = float(trench_mult)

    def grow(seed, k):
        cells = np.flatnonzero(np.isfinite(seed))
        if mu[k] == INF or not cells.size:
            return seed
        return drain(v, steps, cells, seed[cells], length_rate=mu[k],
                     weight_mult=mult, engine=drain_engine).ravel()

    In: dict[int, np.ndarray] = {}
    seed = np.full(N, INF)
    seed[turbines] = 0.0
    In[1] = grow(seed, 1)
    for j in range(2, n + 1):
        seed = np.full(N, INF)
        for a in range(1, j // 2 + 1):
            np.minimum(seed, In[a] + In[j - a], out=seed)
        In[j] = grow(seed, j)
    if root_cost is None:
        rc = np.zeros(N)
    else:
        rc = np.asarray(root_cost, dtype=np.float64).ravel() / cell
    rc = np.where((v.ravel() == EXCLUDED), INF, rc)
    rc[turbines] = INF
    Out: dict[int, np.ndarray] = {n: grow(rc.copy(), n)}
    for k in range(n - 1, 0, -1):
        seed = np.full(N, INF)
        for j in range(1, n - k + 1):
            np.minimum(seed, In[j] + Out[k + j], out=seed)
        Out[k] = grow(seed, k)
    return Out

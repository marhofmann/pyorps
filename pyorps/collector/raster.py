"""The trench-sharing collector on a raster window (plan rev. 5, D4c prototype).

The same cut-signature Dreyfus--Wagner as :mod:`pyorps.collector.reference`
-- the same token algebra, turbine rule, merges, stations and root -- but
laid out the way the plan runs it at raster scale:

* **one drain per growable label**, a seeded multi-source Dijkstra
  (``MultiSourceSolver.solve_stream`` with ``weight_mult = 1 + omega`` for
  the trench and ``length_rate = R(L)`` for the systems), no heap over
  labels;
* labels of one turbine set processed in the plan's **potential order**:
  every junction strictly increases ``(|union D|, -|U|, |D|)`` (set form;
  ``(sum D, -|U|, |D|)`` in count form), so when a label is drained, every
  transition into it has already been computed;
* station states (``st = 1``) never claim a cell in a drain: they expand
  by ONE step, computed here with the kernel's exact step weights, and
  those neighbours seed the drain. That keeps the building-once rule exact
  without a kernel change.

All arithmetic runs in CELL units like the kernels (EUR divided by the
cell size) and is scaled back at the end. The step weights are the
kernel's own (float32 factor, plan D8), so on the graph of the same raster
:func:`graph_from_raster` the two engines agree to rounding.

This is a prototype for measuring label counts, support and time on real
windows (plan D4 timing prototype, Phase F); memory is labels x cells x 8 B,
about 28 B with ``keep_trace``.

**Traceback** (engine B, ``keep_trace=True``). Plan section 3.2, item 7:
trace codes store the argmin step or operation, and argmins are never
recovered by label equality. Every value keeps the code of what set it:

* the drain's predecessor forest (``-1`` where the seed won);
* per seed cell, the operation that seeded it -- a turbine record, a
  merge pair, or one step out of a station state (with its direction);
* per station cell, the merge or junction that set ``F1``;
* per arrival, the direction of the step.

The root's partition is recomputed at the one root cell with its argmins
tracked. :meth:`RasterCollector.design` follows the codes into a
:class:`~pyorps.collector.design.Design` whose trench nodes are raster
cells, and :class:`~pyorps.collector.raster_pricer.RasterStepPricer`
re-prices it independently (plan D7).

**Pruning** (``budget=``; plan section 3.2, item 6). A certificate that
never steers anything: a label value is dropped where
``F(v) + Out_|S|(v) > budget + eps``, with ``Out`` the consistent
admissible completion bounds of
:func:`~pyorps.collector.bounds.completion_bounds_raster` (2n drains,
the transposed turbine rule included). Because ``Out`` is consistent, a
cell dropped after the drain has only dropped cells beyond it, so the
filter loses nothing a surviving design needs: ``MV(g)`` stays exact
wherever ``MV(g) + root_cost(g) <= budget + eps``; elsewhere it is only
certified to exceed the budget (it may read higher, or ``inf``).
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from pyorps.certify.windows import EXCLUDED, drain, step_table
from pyorps.collector.model import (
    INF,
    CollectorModel,
    all_submasks,
    bits,
    popcount,
    set_partitions,
)
from pyorps.collector.reference import (
    CollectorGraph,
    _CountAlgebra,
    _SetAlgebra,
)

__all__ = ["RasterCollector", "graph_from_raster"]


def _step_weights(values, steps, turbines_mask, *, into_turbines: bool):
    """Per direction: (dr, dc, weight array over ORIGIN cells, length).

    Weight = the kernel's step weight in cell units, ``inf`` where the step
    leaves the grid, touches an excluded cell, or crosses a turbine cell.
    ``into_turbines`` allows a turbine as the step's END (arrivals); the
    drain itself never enters one.
    """
    v = np.asarray(values)
    rows, cols = v.shape
    vf = v.astype(np.float64)
    bad = v == EXCLUDED
    out = []
    for dr, dc, inter, fac, ln in step_table(steps):
        w = np.full((rows, cols), INF)
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        # every cell the step touches must lie on the grid
        for a, b in inter:
            r0, r1 = max(r0, -a), min(r1, rows - max(0, a))
            c0, c1 = max(c0, -b), min(c1, cols - max(0, b))
        if r0 >= r1 or c0 >= c1:
            out.append((dr, dc, w, ln))
            continue
        org = (slice(r0, r1), slice(c0, c1))
        tgt = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        s = vf[org] + vf[tgt]
        ok = ~bad[org] & ~bad[tgt]
        if not into_turbines:
            ok &= ~turbines_mask[tgt]
        for a, b in inter:
            cell = (slice(r0 + a, r1 + a), slice(c0 + b, c1 + b))
            s = s + vf[cell]
            ok &= ~bad[cell] & ~turbines_mask[cell]
        w[org] = np.where(ok, s * fac, INF)
        out.append((dr, dc, w, ln))
    return out


def graph_from_raster(values, steps, cell_m: float, turbines, *,
                      trench_mult: float = 1.0) -> CollectorGraph:
    """The collector graph of a raster window, for the reference engine.

    One undirected edge per usable step pair, with trench cost
    ``trench_mult * cell_m * w`` (EUR) and length ``cell_m * |step|``;
    steps touching an excluded cell or crossing a turbine cell are dropped,
    exactly as the raster engine blocks them.
    """
    v = np.asarray(values)
    rows, cols = v.shape
    tmask = np.zeros((rows, cols), dtype=bool)
    tmask.ravel()[np.asarray(turbines, dtype=np.int64)] = True
    edges = []
    seen = set()
    for dr, dc, w, ln in _step_weights(v, steps, tmask, into_turbines=True):
        for u in np.flatnonzero(np.isfinite(w.ravel())):
            r, c = divmod(int(u), cols)
            t = (r + dr) * cols + (c + dc)
            key = (min(int(u), t), max(int(u), t))
            if key in seen:
                continue
            seen.add(key)
            edges.append((int(u), t, float(trench_mult * cell_m * w[r, c]),
                          float(cell_m * ln)))
    return CollectorGraph(rows * cols, edges)


class _Seeds:
    """The pending seeds of one label and, when tracing, their codes.

    ``v0``/``v1``: the values offered for ``F0`` (drain seeds) and ``F1``
    (station states). ``c0``/``c1``: per cell, the index into
    ``ops0``/``ops1`` of the operation that set the value; the first offer
    wins a tie.
    """
    __slots__ = ("v0", "c0", "ops0", "v1", "c1", "ops1", "_trace")

    def __init__(self, n: int, trace: bool):
        self._trace = trace
        self.v0 = np.full(n, INF)
        self.c0 = np.full(n, -1, dtype=np.int32) if trace else None
        self.ops0: list = []
        self.v1 = None
        self.c1 = None
        self.ops1: list = []

    def offer0_at(self, cell: int, val: float, op) -> None:
        if val < self.v0[cell]:
            self.v0[cell] = val
            if self._trace:
                self.c0[cell] = len(self.ops0)
                self.ops0.append(op)

    def offer0(self, vals: np.ndarray, op) -> None:
        if not self._trace:
            np.minimum(self.v0, vals, out=self.v0)
            return
        better = vals < self.v0
        if better.any():
            self.c0[better] = len(self.ops0)
            self.ops0.append(op)
            self.v0[better] = vals[better]

    def offer1(self, vals: np.ndarray, op) -> None:
        if self.v1 is None:
            self.v1 = np.full(vals.shape, INF)
            if self._trace:
                self.c1 = np.full(vals.shape, -1, dtype=np.int32)
        if not self._trace:
            np.minimum(self.v1, vals, out=self.v1)
            return
        better = vals < self.v1
        if better.any():
            self.c1[better] = len(self.ops1)
            self.ops1.append(op)
            self.v1[better] = vals[better]


@dataclass
class _Label:
    F0: np.ndarray
    F1: np.ndarray | None = None
    growable: bool = False
    rate: float = INF
    # trace codes (keep_trace only)
    prev: np.ndarray | None = None      # drain predecessor, -1 at seeds
    c0: np.ndarray | None = None
    ops0: list | None = None
    c1: np.ndarray | None = None
    ops1: list | None = None


@dataclass
class RasterCollector:
    """``MV(g)`` for every cell of a raster window (see the module docstring).

    Parameters:
        values: uint16 trench cost per metre of trench (EUR/m) per cell;
            65535 excluded.
        steps: The neighbourhood step table (``get_neighborhood_steps``).
        cell_m: Cell size in metres.
        turbines: Flat cell indices of the turbines (bit i = turbine i).
        model: The ``D_share`` model of one MV level.
        engine: ``"A"`` (count tokens, type free per section) or ``"B"``.
        trench_mult: ``1 + omega`` on the trench.
        drain_engine: passed to :func:`pyorps.certify.windows.drain`.
        keep_trace: Keep the trace codes :meth:`design` follows (engine B
            only).
        budget: Prune against this total (EUR), e.g. an incumbent
            ``Z_UB``; ``None`` for no pruning.
        root_cost: EUR per cell of placing the UW there, counted against
            the budget (``None``: 0).
        prune_eps: Added to the budget (the safe sign: it can only keep
            more).
    """
    values: np.ndarray
    steps: np.ndarray
    cell_m: float
    turbines: list
    model: CollectorModel
    engine: str = "A"
    trench_mult: float = 1.0
    drain_engine: str = "auto"
    keep_trace: bool = False
    budget: float | None = None
    root_cost: np.ndarray | None = None
    prune_eps: float = 0.0
    stats: dict = field(default_factory=dict)

    def run(self) -> np.ndarray:
        if self.keep_trace and self.engine != "B":
            raise ValueError("traceback needs engine B (set tokens)")
        v = np.asarray(self.values)
        self._shape = v.shape
        N = v.size
        self._N = N
        tmask = np.zeros(v.shape, dtype=bool)
        tmask.ravel()[np.asarray(self.turbines, dtype=np.int64)] = True
        self._tmask = tmask.ravel()
        self._excl = (v == EXCLUDED).ravel()
        self._steiner = ~self._tmask & ~self._excl
        self._w_drain = _step_weights(v, self.steps, tmask,
                                      into_turbines=False)
        self._w_arr = _step_weights(v, self.steps, tmask, into_turbines=True)
        self._dirs = [(dr, dc) for dr, dc, _w, _ln in self._w_arr]
        model = self.model
        per_cable = self.engine == "B"
        self.alg = (_SetAlgebra if self.engine == "B" else _CountAlgebra)(
            model, per_cable)
        self._cell = float(self.cell_m)
        self.labels: dict = {}
        self.by_S: dict = {}
        self.Aarr: dict = {}
        self.Aarg: dict = {}                # label -> arrival direction
        self.stats.update(drains=0, labels=0, merge_pairs=0,
                          cells_kept=0, cells_pruned=0)
        self._out = None
        if self.budget is not None:
            from pyorps.collector.bounds import completion_bounds_raster
            if self.prune_eps < 0:
                raise ValueError("prune_eps must be >= 0 (the safe sign)")
            self._out = completion_bounds_raster(
                v, self.steps, self.cell_m, self.turbines, model,
                root_cost=self.root_cost, trench_mult=self.trench_mult,
                drain_engine=self.drain_engine)
            self._limit = (float(self.budget) + float(self.prune_eps))                 / self._cell
            self.stats["drains_bounds"] = 2 * model.n
        order = sorted(range(1, model.full + 1),
                       key=lambda s: (popcount(s), s))
        for S in order:
            self._level(S)
        self._mv = self._root()
        return self._mv * self._cell

    # ---------------------------------------------------------- helpers
    def _eur(self, x):
        return x / self._cell

    def _prune(self, F, k: int) -> None:
        """Drop, in place, the cells where ``F + Out_k > budget + eps``."""
        if self._out is None or F is None:
            return
        cut = np.isfinite(F) & (F + self._out[k] > self._limit)
        n_cut = int(cut.sum())
        if n_cut:
            F[cut] = INF
            self.stats["cells_pruned"] += n_cut

    def _potential(self, L):
        S, U, D = L
        if self.alg.mode == "set":
            u = 0
            for t in D:
                u |= t[0]
            return (popcount(u), -len(U), len(D))
        return (sum(t[0] for t in D), -len(U), len(D))

    def _one_step(self, F, rate, weights, *, arg: bool = False):
        """``min over steps of F[origin] + mult * w + rate * len`` at the
        step's end, for every cell; with ``arg`` also the winning
        direction (``-1`` where nothing arrives)."""
        rows, cols = self._shape
        Fg = F.reshape(rows, cols)
        out = np.full((rows, cols), INF)
        k_arg = np.full((rows, cols), -1, dtype=np.int16) if arg else None
        for k, (dr, dc, w, ln) in enumerate(weights):
            cand = Fg + self.trench_mult * w + rate * ln
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            src = cand[r0:r1, c0:c1]
            dst = out[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if arg:
                better = src < dst
                dst[better] = src[better]
                k_arg[r0 + dr:r1 + dr, c0 + dc:c1 + dc][better] = k
            else:
                np.minimum(dst, src, out=dst)
        if arg:
            return out.ravel(), k_arg.ravel()
        return out.ravel()

    # ------------------------------------------------------------ level
    def _level(self, S):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        alg, model, N = self.alg, self.model, self._N
        trace = self.keep_trace
        pend: dict = {}

        def seeds_of(L) -> _Seeds:
            e = pend.get(L)
            if e is None:
                e = _Seeds(N, trace)
                pend[L] = e
            return e

        # turbine seeds
        for i in bits(S):
            t = int(self.turbines[i])
            for L, cost, rec in self._turbine_seeds(S, i, t):
                seeds_of(L).offer0_at(t, cost, ("turb",) + rec)
        # merges at Steiner cells
        low = S & -S
        st = self._steiner
        for sub in all_submasks(S ^ low):
            S1 = sub | low
            if S1 == S:
                continue
            S2 = S ^ S1
            for b1, b2 in alg.merge_pairs(self.by_S.get(S1, ()),
                                          self.by_S.get(S2, ()), S1, S2):
                merged = alg.merge(b1, b2)
                if not merged:
                    continue
                l1, l2 = self.labels[b1], self.labels[b2]
                f00 = np.where(st, l1.F0 + l2.F0, INF)
                f10 = (np.where(st, l1.F1 + l2.F0, INF)
                       if l1.F1 is not None else None)
                f01 = (np.where(st, l1.F0 + l2.F1, INF)
                       if l2.F1 is not None else None)
                for L in merged:
                    self.stats["merge_pairs"] += 1
                    e = seeds_of(L)
                    e.offer0(f00, ("merge", b1, 0, b2, 0))
                    if f10 is not None:
                        e.offer1(f10, ("merge", b1, 1, b2, 0))
                    if f01 is not None:
                        e.offer1(f01, ("merge", b1, 0, b2, 1))
        # potential-ordered processing; junctions add labels on the fly
        heap = [(self._potential(L), L) for L in pend]
        heapq.heapify(heap)
        done = set()
        finished = []
        no_transit = np.flatnonzero(self._tmask)
        while heap:
            _p, L = heapq.heappop(heap)
            if L in done:
                continue
            done.add(L)
            e = pend.pop(L)
            k_S = popcount(S)
            self._prune(e.v0, k_S)
            self._prune(e.v1, k_S)
            F1 = e.v1
            growable = alg.growable(L)
            rate = alg.rate(L[1], L[2]) if growable else INF
            prev = None
            if growable:
                seeds = e.v0.copy()
                if F1 is not None:
                    step_in = self._one_step(F1, rate, self._w_drain,
                                             arg=trace)
                    if trace:
                        vals, karg = step_in
                        better = vals < seeds
                        if better.any():
                            ks = np.unique(karg[better])
                            lut = np.full(len(self._dirs), -1,
                                          dtype=np.int32)
                            lut[ks] = len(e.ops0) + np.arange(ks.size)
                            e.ops0.extend(("f1step", int(k)) for k in ks)
                            e.c0[better] = lut[karg[better]]
                            seeds[better] = vals[better]
                    else:
                        np.minimum(seeds, step_in, out=seeds)
                cells = np.flatnonzero(np.isfinite(seeds))
                res = drain(self.values, self.steps, cells, seeds[cells],
                            length_rate=rate, weight_mult=self.trench_mult,
                            no_transit=no_transit, engine=self.drain_engine,
                            return_prev=trace)
                if trace:
                    F0, prev = res[0].ravel(), res[1].ravel()
                else:
                    F0 = res.ravel()
                self._prune(F0, k_S)
                self.stats["drains"] += 1
            else:
                F0 = e.v0
            if not (np.isfinite(F0).any()
                    or (F1 is not None and np.isfinite(F1).any())):
                continue
            self.stats["cells_kept"] += int(np.isfinite(F0).sum())
            self.labels[L] = _Label(
                F0=F0, F1=F1, growable=growable, rate=rate, prev=prev,
                c0=e.c0, ops0=e.ops0 if trace else None,
                c1=e.c1, ops1=e.ops1 if trace else None)
            finished.append(L)
            # junction transitions into later labels, at Steiner cells
            if model.allow_stations:
                pan_eur = model.station_panel_eur
                bld = model.station_building_eur
                for L2, pan in alg.junctions(L):
                    cand0 = np.where(st, F0 + self._eur(bld + pan_eur * pan),
                                     INF)
                    cand1 = (np.where(st, F1 + self._eur(pan_eur * pan), INF)
                             if F1 is not None else None)
                    if not (np.isfinite(cand0).any()
                            or (cand1 is not None
                                and np.isfinite(cand1).any())):
                        continue
                    if L2 in done:
                        raise RuntimeError("junction into a finished label: "
                                           "the potential order is broken")
                    e2 = pend.get(L2)
                    if e2 is None:
                        e2 = _Seeds(N, trace)
                        pend[L2] = e2
                        heapq.heappush(heap, (self._potential(L2), L2))
                    e2.offer1(cand0, ("junc", L, 0, pan))
                    if cand1 is not None:
                        e2.offer1(cand1, ("junc", L, 1, pan))
        self.by_S[S] = finished
        self.stats["labels"] += len(finished)
        for L in finished:
            lab = self.labels[L]
            if lab.growable and not L[2]:
                Fmin = lab.F0 if lab.F1 is None else np.minimum(lab.F0, lab.F1)
                if trace:
                    self.Aarr[L], self.Aarg[L] = self._one_step(
                        Fmin, lab.rate, self._w_arr, arg=True)
                else:
                    self.Aarr[L] = self._one_step(Fmin, lab.rate, self._w_arr)
                self._prune(self.Aarr[L], popcount(S))
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    def _turbine_seeds(self, S, i, t):
        """Mirror of the reference turbine rule, arrivals from ``Aarr``.

        Returns ``(label, cost, (i, kids, ins, downs, out token))``.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        alg, model = self.alg, self.model
        Y = S & ~(1 << i)
        children: dict = {}
        if Y == 0:
            children[()] = (0.0, ())
        else:
            for part in set_partitions([1 << j for j in bits(Y)]):
                groups = [sum(b) for b in part]
                opts = []
                for Yg in groups:
                    lst = [(L, self.Aarr[L][t]) for L in self.by_S.get(Yg, ())
                           if not L[2] and L in self.Aarr
                           and self.Aarr[L][t] < INF]
                    if not lst:
                        break
                    opts.append(lst)
                else:
                    for combo in itertools.product(*opts):
                        ins = tuple(sorted(tok for (L, _a) in combo
                                           for tok in L[1]))
                        if alg.cables(ins) > model.ring_panels - 1:
                            continue
                        cost = sum(a for (_L, a) in combo)
                        if cost < children.get(ins, (INF,))[0]:
                            children[ins] = (cost, tuple(L for (L, _a)
                                                         in combo))
        out = []
        avail = model.full & ~S
        downs_all = alg.down_collections(avail)
        own = alg.key_of_turbine(i)
        for ins, (cost_in, kids) in children.items():
            in_c = alg.cables(ins)
            in_key = alg.child_key_union(tok[0] for tok in ins)
            for downs in downs_all:
                cond_in = in_c + alg.cables(downs)
                if cond_in > model.ring_panels - 1:
                    continue
                if alg.mode == "count":
                    out_key = own + in_key + sum(tok[0] for tok in downs)
                else:
                    out_key = own | in_key | alg.union(downs)
                for o in alg.new_tokens(out_key):
                    cond = cond_in + o[1]
                    if not alg.turbine_ok(out_key, cond):
                        continue
                    L = (S, (o,), downs)
                    if not alg.growable(L):
                        continue
                    out.append((L, cost_in
                                + self._eur(model.turbine_panel_eur * cond),
                                (i, kids, ins, downs, o)))
        return out

    def _root_values(self):
        """``{S: [(label, value array)]}``: the arriving feeders of each
        turbine set with their bays."""
        alg, model = self.alg, self.model
        best: dict = {}
        for L, arrv in self.Aarr.items():
            S, U, _D = L
            if not all(alg.bay_ok(tok) for tok in U):
                continue
            best.setdefault(S, []).append(
                (L, arrv + self._eur(model.bay_eur * alg.cables(U))))
        return best

    def _root(self):
        model, N = self.model, self._N
        best = {S: np.minimum.reduce([v for _L, v in lst])
                for S, lst in self._root_values().items()}
        H = {0: np.zeros(N)}
        for X in range(1, model.full + 1):
            low = X & -X
            h = np.full(N, INF)
            for sub in all_submasks(X ^ low):
                Y = sub | low
                if Y in best:
                    np.minimum(h, best[Y] + H[X ^ Y], out=h)
            H[X] = h
        mv = H[model.full].copy()
        mv[self._tmask | self._excl] = INF
        return mv.reshape(self._shape)

    # ------------------------------------------------------------ trace
    def _origin(self, v: int, k: int) -> int:
        """The cell a step of direction ``k`` into ``v`` started from."""
        cols = self._shape[1]
        dr, dc = self._dirs[k]
        r, c = divmod(int(v), cols)
        return (r - dr) * cols + (c - dc)

    def _arrival(self, L, v: int) -> tuple[int, int]:
        """``(origin cell, state)`` of the arrival ``Aarr[L][v]``."""
        k = int(self.Aarg[L][v])
        if k < 0:
            raise RuntimeError(f"no arrival code for {L} at cell {v}")
        x = self._origin(v, k)
        lab = self.labels[L]
        f1 = INF if lab.F1 is None else lab.F1[x]
        return x, (0 if lab.F0[x] <= f1 else 1)

    def design(self, g: int):
        """The optimal design rooted at cell ``g`` (engine B, ``keep_trace``).

        Trench nodes are raster cells; each trench edge is one kernel step
        (edge key ``None``), priced by
        :class:`~pyorps.collector.raster_pricer.RasterStepPricer`.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from pyorps.collector.design import Design, Junction, System

        if not self.keep_trace:
            raise RuntimeError("run with keep_trace=True to trace designs")
        g = int(g)
        if not np.isfinite(self._mv.ravel()[g]):
            raise ValueError(f"no feasible collector at cell {g}")
        tnodes: list[int] = []
        tedges: list = []
        systems: list = []
        junctions: list = []
        turbine_tnode: dict = {}
        sys_of: dict = {}
        limit = 4 * self._N + 16

        def system_id(tok):
            sid = sys_of.get(tok)
            if sid is None:
                sid = len(systems)
                systems.append(System(mask=tok[0], option=(tok[2], tok[1])))
                sys_of[tok] = sid
            return sid

        def new_node(cell):
            tnodes.append(int(cell))
            return len(tnodes) - 1

        def rooted(L, st, v, top=None):
            # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
            first = top = new_node(v) if top is None else top
            for _ in range(limit):              # grow chains iterate
                lab = self.labels[L]
                if st == 0 and lab.prev is not None and lab.prev[v] >= 0:
                    u = int(lab.prev[v])
                    child = new_node(u)
                    tedges.append((child, top, None))
                    top, v = child, u
                    continue
                if st == 0:
                    code = int(lab.c0[v])
                    op = lab.ops0[code] if code >= 0 else None
                    if op is not None and op[0] == "f1step":
                        u = self._origin(v, op[1])
                        child = new_node(u)
                        tedges.append((child, top, None))
                        top, v, st = child, u, 1
                        continue
                else:
                    code = int(lab.c1[v]) if lab.c1 is not None else -1
                    op = lab.ops1[code] if code >= 0 else None
                break
            else:
                raise RuntimeError("trace codes form a cycle")
            if op is None:
                raise RuntimeError(f"no trace code for {L} (st={st}) at {v}")
            kind = op[0]
            if kind == "turb":
                _, i, kids, ins, downs, o = op
                turbine_tnode[i] = top
                systems[system_id(o)].source = ("turbine", i)
                for tok in ins + downs:
                    systems[system_id(tok)].sink = ("turbine", i)
                for kid in kids:
                    x, st_k = self._arrival(kid, v)
                    tedges.append((rooted(kid, st_k, x), top, None))
            elif kind == "merge":
                _, b1, s1, b2, s2 = op
                rooted(b1, s1, v, top)
                rooted(b2, s2, v, top)
            else:                               # "junc"
                _, Lf, stf, _pan = op
                rooted(Lf, stf, v, top)
                _S, Uf, Df = Lf
                _S, Ut, Dt = L
                jid = len(junctions)
                if set(Ut) != set(Uf):          # Ja: output goes up
                    out_tok = next(iter(set(Ut) - set(Uf)))
                    ins = list(set(Uf) - set(Ut)) + list(set(Dt) - set(Df))
                else:                           # Jb: output goes down
                    out_tok = next(iter(set(Df) - set(Dt)))
                    ins = list(set(Dt) - set(Df))
                junctions.append(Junction(tnode=top, inputs=[], output=-1))
                oid = system_id(out_tok)
                systems[oid].source = ("junction", jid)
                junctions[jid].output = oid
                for tok in ins:
                    sid = system_id(tok)
                    systems[sid].sink = ("junction", jid)
                    junctions[jid].inputs.append(sid)
            return first

        # the root's partition at g, recomputed with its argmins tracked
        best: dict = {}
        for S, lst in self._root_values().items():
            for L, vals in lst:
                if vals[g] < best.get(S, (INF,))[0]:
                    best[S] = (float(vals[g]), L)
        H: dict = {0: (0.0, 0)}
        for X in range(1, self.model.full + 1):
            low = X & -X
            h = (INF, 0)
            for sub in all_submasks(X ^ low):
                Y = sub | low
                if Y in best:
                    val = best[Y][0] + H[X ^ Y][0]
                    if val < h[0]:
                        h = (val, Y)
            H[X] = h
        root_node = new_node(g)
        X = self.model.full
        while X:
            Y = H[X][1]
            if Y == 0:
                raise RuntimeError(f"no root partition for {X:b} at {g}")
            L = best[Y][1]
            x, st_x = self._arrival(L, g)
            tedges.append((rooted(L, st_x, x), root_node, None))
            for tok in L[1]:
                systems[system_id(tok)].sink = ("root", 0)
            X ^= Y
        return Design(tnodes=tnodes, tedges=tedges, root_tnode=root_node,
                      turbine_tnode=turbine_tnode, systems=systems,
                      junctions=junctions,
                      cost=float(self._mv.ravel()[g] * self._cell))
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

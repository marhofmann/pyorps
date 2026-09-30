"""Batched voltage checks on the fully meshed slot network (C5).

Many designs are checked at once -- the argmins of many substation sites,
the repair steps of one site, a voltage map over a whole window. Every
design of one MV level lives on the same small **slot network**:

* slot ``0`` the UW busbar, slots ``1..n`` the turbines, slots
  ``n+1..n+J`` the switching-station busbar sections (junctions);
* every unordered slot pair ``(u, v)``, ``u < v``, is a potential branch:
  ``E = S (S - 1) / 2`` edges for ``S = 1 + n + J`` slots (91 for seven
  turbines and six junction slots).

A :class:`DesignBatch` is that fully meshed network plus, per design (row)
and edge (column), an ``active`` flag and the branch values ``r_ohm``,
``x_ohm``, ``b_s`` (series resistance and reactance, total shunt
susceptance of the pi-model). A design switches on at most ``2n - 1``
edges: one out-branch per used non-root slot.

Backends (:func:`batch_voltages`), all exact AC load flows of the same
model, all feeding the one limit rule
:func:`~pyorps.collector.voltage.evaluate_limits`:

* ``"numba"`` (default, the fastest by far) -- one compiled kernel per
  design: breadth-first orientation, the backward--forward sweep of
  :func:`~pyorps.collector.voltage.load_flow` at every corner and the LV
  voltage at every tap, one design per parallel iteration;
* ``"numpy"`` -- the same sweep vectorised over designs and corners (slots
  relabelled in depth order per design, so every sweep is ``S - 1`` vector
  steps);
* ``"inflated_nr"`` -- Newton--Raphson on the inflated network: every
  design stamps its branch values into one dense admittance matrix of the
  fully meshed slot network (one pattern for all designs, the
  ``powerklujax`` idea);
* ``"pgm_islands"`` / ``"pgm_inflated"`` -- power-grid-model, see
  :mod:`pyorps.collector.voltage_pgm`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pyorps.collector.voltage import (
    ElectricalTree,
    LimitResult,
    VoltageModel,
    decide_taps,
    electrical_tree,
    evaluate_limits,
)

__all__ = [
    "BACKENDS",
    "BatchLoadFlow",
    "BatchVoltageCheck",
    "DesignBatch",
    "batch_load_flow",
    "batch_voltages",
    "check_batch",
    "design_batch",
    "inflated_nr_load_flow",
    "numba_load_flow",
    "slot_edges",
]


def slot_edges(n_slots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(edge_u, edge_v, index)`` of the fully meshed slot network;
    ``index[u, v] = index[v, u]`` is the edge of the pair (-1 on the
    diagonal)."""
    u, v = np.triu_indices(n_slots, 1)
    index = np.full((n_slots, n_slots), -1, dtype=np.int64)
    e = np.arange(len(u), dtype=np.int64)
    index[u, v] = e
    index[v, u] = e
    return u.astype(np.int64), v.astype(np.int64), index


@dataclass
class DesignBatch:
    """``B`` designs on one fully meshed slot network (module docstring).

    Attributes:
        n: Turbines.
        n_slots: ``S = 1 + n + J``.
        u_kv: Nominal voltage of the level.
        active: ``(B, E)`` bool, the branches each design switches on.
        r_ohm, x_ohm, b_s: ``(B, E)`` branch values (0 where inactive).
        keys: One identifier per design (e.g. the root cell).
    """
    n: int
    n_slots: int
    u_kv: float
    active: np.ndarray
    r_ohm: np.ndarray
    x_ohm: np.ndarray
    b_s: np.ndarray
    keys: list = field(default_factory=list)
    edge_u: np.ndarray = field(init=False, repr=False)
    edge_v: np.ndarray = field(init=False, repr=False)
    edge_index: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.edge_u, self.edge_v, self.edge_index = slot_edges(self.n_slots)
        E = len(self.edge_u)
        self.active = np.asarray(self.active, dtype=bool)
        if self.active.ndim != 2 or self.active.shape[1] != E:
            raise ValueError(f"active must be (B, {E}) for {self.n_slots} "
                             f"slots")
        for name in ("r_ohm", "x_ohm", "b_s"):
            a = np.asarray(getattr(self, name), dtype=np.float64)
            if a.shape != self.active.shape:
                raise ValueError(f"{name} must have the shape of active")
            if not (np.isfinite(a).all() and (a >= 0).all()):
                raise ValueError(f"{name} must be finite and >= 0")
            setattr(self, name, a)
        if self.n_slots < 1 + self.n:
            raise ValueError("need at least 1 + n slots")
        if not self.keys:
            self.keys = list(range(len(self.active)))
        if len(self.keys) != len(self.active):
            raise ValueError("one key per design")

    def __len__(self) -> int:
        return len(self.active)

    @property
    def n_edges(self) -> int:
        return len(self.edge_u)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """The per-design arrays as a plain dictionary."""
        return {"active": self.active, "r_ohm": self.r_ohm,
                "x_ohm": self.x_ohm, "b_s": self.b_s}

    @classmethod
    def from_arrays(cls, arrays, *, n: int, n_slots: int, u_kv: float,
                    keys=None) -> DesignBatch:
        return cls(n=n, n_slots=n_slots, u_kv=u_kv,
                   active=arrays["active"], r_ohm=arrays["r_ohm"],
                   x_ohm=arrays["x_ohm"], b_s=arrays["b_s"],
                   keys=list(keys) if keys is not None else [])

    def subset(self, rows) -> DesignBatch:
        rows = np.asarray(rows)
        return DesignBatch(n=self.n, n_slots=self.n_slots, u_kv=self.u_kv,
                           active=self.active[rows], r_ohm=self.r_ohm[rows],
                           x_ohm=self.x_ohm[rows], b_s=self.b_s[rows],
                           keys=[self.keys[i] for i in rows.tolist()])

    @classmethod
    def from_trees(cls, trees: Sequence[ElectricalTree], *, n: int,
                   n_slots: int | None = None, keys=None) -> DesignBatch:
        """Pack electrical trees (:func:`electrical_tree`) into a batch.

        Node labels map to slots: ``("root", 0) -> 0``,
        ``("turbine", t) -> 1 + t``, ``("junction", j) -> 1 + n + j``.
        """
        if not trees:
            raise ValueError("no designs")
        need = max(1 + n + sum(1 for lab in t.labels if lab[0] == "junction")
                   for t in trees)
        S = need if n_slots is None else int(n_slots)
        if S < need:
            raise ValueError(f"{need} slots needed, {S} given")
        _u, _v, index = slot_edges(S)
        B, E = len(trees), S * (S - 1) // 2
        active = np.zeros((B, E), dtype=bool)
        r = np.zeros((B, E))
        x = np.zeros((B, E))
        b = np.zeros((B, E))
        base = {"root": 0, "turbine": 1, "junction": 1 + n}
        for i, tree in enumerate(trees):
            if tree.shunts:
                raise ValueError(f"design {i}: node shunts are not part of "
                                 f"the slot network (plant trees are "
                                 f"checked one by one)")
            if tree.u_kv != trees[0].u_kv:
                raise ValueError("all designs of a batch share one level")
            slot = [base[k] + j for k, j in tree.labels]
            for br in tree.branches:
                e = index[slot[br.child], slot[br.parent]]
                if active[i, e]:
                    raise ValueError(f"design {i}: two branches on one "
                                     f"slot pair")
                active[i, e] = True
                r[i, e], x[i, e], b[i, e] = br.r_ohm, br.x_ohm, br.b_s
        return cls(n=n, n_slots=S, u_kv=trees[0].u_kv, active=active,
                   r_ohm=r, x_ohm=x, b_s=b,
                   keys=list(keys) if keys is not None else [])


def design_batch(designs, graph, turbines, model, vmodel: VoltageModel, *,
                 root_transit: bool = True, n_slots: int | None = None,
                 keys=None) -> DesignBatch:
    """A :class:`DesignBatch` from traced designs.

    ``graph`` is one graph or pricer for all designs, or a sequence with one
    per design. Each design is validated and routed by
    :func:`~pyorps.collector.design.layout` (through
    :func:`~pyorps.collector.voltage.electrical_tree`), exactly as the
    single-design check does.
    """
    designs = list(designs)
    graphs = (list(graph) if isinstance(graph, (list, tuple))
              else [graph] * len(designs))
    if len(graphs) != len(designs):
        raise ValueError("one graph per design")
    trees = [electrical_tree(d, g, turbines, model, vmodel,
                             root_transit=root_transit)
             for d, g in zip(designs, graphs)]
    return DesignBatch.from_trees(trees, n=model.n, n_slots=n_slots,
                                  keys=keys)


# ------------------------------------------------------- orientation


@dataclass
class _Oriented:
    """Per design: the slots relabelled in depth order (``perm[b, k]`` is
    the slot at position ``k``), the parent POSITION of every position, the
    series impedance of its out-branch and its node shunt (p.u.), which
    slots the design uses, the parent SLOT of every slot, and ``rep``: the
    slot each slot is merged into through zero-impedance branches (a system
    that never leaves its trench node joins two busbar sections of one
    station; solvers that cannot take an ideal short circuit use one node
    for both)."""
    perm: np.ndarray
    pos_of: np.ndarray
    parent_pos: np.ndarray
    z_pu: np.ndarray
    y_pu: np.ndarray
    used: np.ndarray
    parent: np.ndarray
    rep: np.ndarray


def _orient(batch: DesignBatch, s_base_mva: float) -> _Oriented:
    B, S, n = len(batch), batch.n_slots, batch.n
    bi, ei = np.nonzero(batch.active)
    u, v = batch.edge_u[ei], batch.edge_v[ei]
    depth = np.full((B, S), -1, dtype=np.int64)
    parent = np.zeros((B, S), dtype=np.int64)
    pedge = np.full((B, S), -1, dtype=np.int64)       # flat (bi, ei) row
    depth[:, 0] = 0
    open_ = np.ones(len(bi), dtype=bool)
    for d in range(1, S):
        du, dv = depth[bi, u], depth[bi, v]
        down_v = open_ & (du == d - 1) & (dv < 0)
        down_u = open_ & (dv == d - 1) & (du < 0)
        m = down_v | down_u
        if not m.any():
            break
        child = np.where(down_v, v, u)[m]
        par = np.where(down_v, u, v)[m]
        rows = bi[m]
        flat = rows * S + child
        if len(np.unique(flat)) != len(flat):
            bad = int(rows[np.argmax(np.bincount(flat) > 1)])
            raise ValueError(f"design {bad}: the branches form a cycle")
        depth[rows, child] = d
        parent[rows, child] = par
        pedge[rows, child] = np.flatnonzero(m)
        open_ &= ~m
    if open_.any():
        bad = int(bi[np.flatnonzero(open_)[0]])
        raise ValueError(f"design {bad}: a branch is on a cycle or not "
                         f"connected to the UW")
    if (depth[:, 1:1 + n] < 0).any():
        bad = int(np.flatnonzero((depth[:, 1:1 + n] < 0).any(axis=1))[0])
        raise ValueError(f"design {bad}: a turbine is not connected")
    used = depth >= 0
    key = np.where(used, depth, S)
    perm = np.argsort(key, axis=1, kind="stable")
    ar = np.arange(B)[:, None]
    pos_of = np.empty_like(perm)
    pos_of[ar, perm] = np.arange(S)[None, :]
    parent_pos = np.take_along_axis(pos_of, np.take_along_axis(parent, perm,
                                                               1), 1)
    parent_pos[:, 0] = 0
    zb = batch.u_kv ** 2 / s_base_mva
    z_slot = np.zeros((B, S), dtype=np.complex128)
    has = pedge >= 0
    pe = pedge[has]
    z_slot[has] = (batch.r_ohm[bi[pe], ei[pe]]
                   + 1j * batch.x_ohm[bi[pe], ei[pe]]) / zb
    half = 0.5j * batch.b_s[bi, ei] * zb
    y_slot = np.zeros(B * S, dtype=np.complex128)
    for ends in (u, v):
        f = bi * S + ends
        y_slot += (np.bincount(f, weights=half.imag, minlength=B * S) * 1j)
    y_slot = y_slot.reshape(B, S)
    rep = np.broadcast_to(np.arange(S), (B, S)).copy()
    zero = has & (z_slot == 0)
    if zero.any():
        rows = np.arange(B)
        for k in range(1, S):                  # depth order: parents first
            slot = perm[:, k]
            m = zero[rows, slot]
            if m.any():
                rep[rows[m], slot[m]] = rep[rows[m], parent[rows[m],
                                                           slot[m]]]
    return _Oriented(perm=perm, pos_of=pos_of, parent_pos=parent_pos,
                     z_pu=np.take_along_axis(z_slot, perm, 1),
                     y_pu=np.take_along_axis(y_slot, perm, 1), used=used,
                     parent=parent, rep=rep)


# --------------------------------------------------------- load flow


@dataclass
class BatchLoadFlow:
    """Voltages of ``B`` designs at ``C`` corners.

    ``v_pu[c, b, s]``: complex voltage of slot ``s`` (the UW busbar voltage
    on slots a design does not use); ``converged[c, b]``; ``used[b, s]``;
    ``iterations``: sweeps of the slowest row.
    """
    v_pu: np.ndarray
    converged: np.ndarray
    used: np.ndarray
    iterations: int


def _slot_injections(batch: DesignBatch, vmodel: VoltageModel) -> np.ndarray:
    """``(C, S)`` complex injections (MVA) of the corners by slot."""
    C, S, n = len(vmodel.corners), batch.n_slots, batch.n
    s = np.zeros((C, S), dtype=np.complex128)
    for c, cor in enumerate(vmodel.corners):
        s[c, 1:1 + n] = np.asarray(cor.p_mw) + 1j * np.asarray(cor.q_mvar)
    return s


def batch_load_flow(batch: DesignBatch, vmodel: VoltageModel, *,
                    tol_pu: float | None = None,
                    max_iter: int | None = None) -> BatchLoadFlow:
    """The backward--forward sweep of
    :func:`~pyorps.collector.voltage.load_flow`, vectorised over designs and
    corners. Rows that do not converge are flagged, never returned as
    numbers (their voltages are NaN)."""
    _check_level(batch, vmodel)
    tol = vmodel.tol_pu if tol_pu is None else tol_pu
    its = vmodel.max_iter if max_iter is None else max_iter
    o = _orient(batch, vmodel.s_base_mva)
    B, S = len(batch), batch.n_slots
    C = len(vmodel.corners)
    s_pos = (_slot_injections(batch, vmodel) / vmodel.s_base_mva)[:, o.perm]
    v0 = np.array([c.bus_pu for c in vmodel.corners], dtype=np.complex128)
    V = np.broadcast_to(v0[:, None, None], (C, B, S)).copy()
    J = np.zeros((C, B, S), dtype=np.complex128)
    ar = np.arange(B)
    pp = o.parent_pos
    z, y = o.z_pu[None], o.y_pu[None]
    live = np.ones((C, B), dtype=bool)
    it = 0
    with np.errstate(all="ignore"):
        for it in range(1, its + 1):
            acc = np.zeros((C, B, S), dtype=np.complex128)
            cur = np.conj(s_pos / V) - y * V
            for k in range(S - 1, 0, -1):
                jk = cur[:, :, k] + acc[:, :, k]
                J[:, :, k] = jk
                acc[:, ar, pp[:, k]] += jk
            Vn = np.empty_like(V)
            Vn[:, :, 0] = v0[:, None]
            zj = z * J
            for k in range(1, S):
                Vn[:, :, k] = Vn[:, ar, pp[:, k]] + zj[:, :, k]
            delta = np.abs(Vn - V).max(axis=2)
            V = Vn
            bad = ~np.isfinite(delta)
            if bad.any():
                live &= ~bad
                V[bad] = v0[np.nonzero(bad)[0]][:, None]
            if (delta[live] <= tol).all():
                break
        converged = live & (delta <= tol)
    v_slot = np.take_along_axis(V, o.pos_of[None].repeat(C, 0), 2)
    v_slot[~converged] = np.nan
    return BatchLoadFlow(v_pu=v_slot, converged=converged, used=o.used,
                         iterations=it)


def _check_level(batch: DesignBatch, vmodel: VoltageModel) -> None:
    if vmodel.n != batch.n or abs(vmodel.u_kv - batch.u_kv) > 1e-12:
        raise ValueError("the batch and the voltage model differ in n or "
                         "voltage level")


_KERNEL_ERRORS = {1: "the branches form a cycle",
                  2: "a turbine is not connected",
                  3: "a branch is not connected to the UW"}


def _numba_solve(batch: DesignBatch, vmodel: VoltageModel, *,
                 tol_pu: float | None = None, max_iter: int | None = None,
                 threads: int = 4):
    """Run the fused kernel; return the load flow, the per-tap band margins
    ``(B, n, K)`` and ``max |V|`` per corner and design ``(C, B)``."""
    import numba as nb

    from pyorps.collector._voltage_numba import solve_batch

    _check_level(batch, vmodel)
    tol = vmodel.tol_pu if tol_pu is None else tol_pu
    its = vmodel.max_iter if max_iter is None else max_iter
    B, S, n = len(batch), batch.n_slots, batch.n
    C = len(vmodel.corners)
    bi, ei = np.nonzero(batch.active)
    ptr = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(np.bincount(bi, minlength=B), out=ptr[1:])
    eu = batch.edge_u[ei]
    ev = batch.edge_v[ei]
    zb = batch.u_kv ** 2 / vmodel.s_base_mva
    zser = (batch.r_ohm[bi, ei] + 1j * batch.x_ohm[bi, ei]) / zb
    ysh = 1j * batch.b_s[bi, ei] * zb
    s_slot = _slot_injections(batch, vmodel) / vmodel.s_base_mva
    v0 = np.array([c.bus_pu for c in vmodel.corners], dtype=np.complex128)
    taps = np.asarray(vmodel.taps_kv, dtype=np.float64)
    tr = vmodel.transformer
    zt = (np.zeros(len(taps), dtype=np.complex128) if tr is None else
          np.array([tr.z_ohm_mv(t) for t in taps]) / vmodel.z_base_ohm)
    ratio = vmodel.u_kv / taps
    lo, hi = vmodel.lv_band
    out_v = np.empty((C, B, S), dtype=np.complex128)
    conv = np.empty((C, B), dtype=np.bool_)
    it = np.empty((C, B), dtype=np.int64)
    vmax = np.empty((C, B), dtype=np.float64)
    margin = np.empty((B, n, len(taps)), dtype=np.float64)
    err = np.empty(B, dtype=np.int64)
    prev = nb.get_num_threads()
    nb.set_num_threads(max(1, min(int(threads), nb.config.NUMBA_NUM_THREADS)))
    try:
        solve_batch(ptr, eu, ev, zser, ysh, S, n, s_slot, v0, float(tol),
                    int(its), zt, ratio, float(lo), float(hi), out_v, conv,
                    it, vmax, margin, err)
    finally:
        nb.set_num_threads(prev)
    if err.any():
        b = int(np.flatnonzero(err)[0])
        raise ValueError(f"design {b}: {_KERNEL_ERRORS[int(err[b])]}")
    used = np.zeros((B, S), dtype=bool)
    used[:, 0] = True
    used[bi, eu] = True
    used[bi, ev] = True
    out_v[~conv] = np.nan
    lf = BatchLoadFlow(v_pu=out_v, converged=conv, used=used,
                       iterations=int(it.max()) if it.size else 0)
    return lf, margin, vmax


def numba_load_flow(batch: DesignBatch, vmodel: VoltageModel, *,
                    tol_pu: float | None = None,
                    max_iter: int | None = None,
                    threads: int = 4) -> BatchLoadFlow:
    """The sweep of :func:`batch_load_flow`, compiled with the orientation
    and the LV evaluation in one kernel (numba), one design per parallel
    iteration on ``threads`` threads."""
    return _numba_solve(batch, vmodel, tol_pu=tol_pu, max_iter=max_iter,
                        threads=threads)[0]


def inflated_nr_load_flow(batch: DesignBatch, vmodel: VoltageModel, *,
                          tol_pu: float | None = None, max_iter: int = 30,
                          chunk: int = 1024) -> BatchLoadFlow:
    """Newton--Raphson on the inflated slot network (the ``powerklujax``
    idea): every design stamps its branch VALUES into one dense ``S x S``
    admittance matrix of the fully meshed network, so all designs share one
    pattern and are solved as one batch of dense systems.

    Polar Newton--Raphson, the busbar as the slack, every other used slot
    PQ; unused and merged slots (zero-impedance branches, ``_Oriented.rep``)
    are held by identity rows. Converged when the largest voltage step is
    ``<= tol_pu``.
    """
    _check_level(batch, vmodel)
    tol = vmodel.tol_pu if tol_pu is None else tol_pu
    o = _orient(batch, vmodel.s_base_mva)
    B, S, C = len(batch), batch.n_slots, len(vmodel.corners)
    zb = batch.u_kv ** 2 / vmodel.s_base_mva
    s_spec = _slot_injections(batch, vmodel) / vmodel.s_base_mva   # (C, S)
    v0 = np.array([c.bus_pu for c in vmodel.corners], dtype=np.complex128)
    out_v = np.empty((C, B, S), dtype=np.complex128)
    conv = np.zeros((C, B), dtype=bool)
    it_max = 0
    for lo in range(0, B, chunk):
        hi = min(B, lo + chunk)
        K = hi - lo
        rep = o.rep[lo:hi]
        bi, ei = np.nonzero(batch.active[lo:hi])
        u = rep[bi, batch.edge_u[ei]]
        w = rep[bi, batch.edge_v[ei]]
        z = (batch.r_ohm[lo:hi][bi, ei]
             + 1j * batch.x_ohm[lo:hi][bi, ei]) / zb
        half = 0.5j * batch.b_s[lo:hi][bi, ei] * zb
        keep = z != 0
        ys = np.zeros_like(z)
        ys[keep] = 1.0 / z[keep]
        Y = np.zeros((K, S, S), dtype=np.complex128)
        np.add.at(Y, (bi, u, u), ys + half)
        np.add.at(Y, (bi, w, w), ys + half)
        np.add.at(Y, (bi, u, w), -ys)
        np.add.at(Y, (bi, w, u), -ys)
        pq = o.used[lo:hi] & (rep == np.arange(S)[None, :])
        pq[:, 0] = False
        # a merged slot's injection belongs to its representative
        flat = (np.arange(K)[:, None] * S + rep).ravel()
        s_node = np.empty((C, K, S), dtype=np.complex128)
        for c in range(C):
            w_c = np.broadcast_to(s_spec[c], (K, S)).ravel()
            s_node[c] = (np.bincount(flat, w_c.real, K * S)
                         + 1j * np.bincount(flat, w_c.imag, K * S)
                         ).reshape(K, S)
        fixed = ~pq                                           # (K, S)
        V = np.broadcast_to(v0[:, None, None], (C, K, S)).copy()
        live = np.ones((C, K), dtype=bool)
        eye = np.eye(2 * S)
        fix2 = np.concatenate([fixed, fixed], axis=1)          # (K, 2S)
        Yc = np.conj(Y)[None]
        with np.errstate(all="ignore"):
            for it in range(1, max_iter + 1):
                cur = np.einsum("kij,ckj->cki", Y, V)
                F = s_node - V * np.conj(cur)
                vn = V / np.abs(V)
                dsa = (1j * V[..., :, None]
                       * (np.conj(cur)[..., :, None] * np.eye(S)
                          - Yc * np.conj(V)[..., None, :]))
                dsm = (V[..., :, None] * Yc * np.conj(vn)[..., None, :]
                       + np.eye(S) * (np.conj(cur) * vn)[..., :, None])
                Jm = np.concatenate(
                    [np.concatenate([dsa.real, dsm.real], axis=-1),
                     np.concatenate([dsa.imag, dsm.imag], axis=-1)],
                    axis=-2)                                 # (C, K, 2S, 2S)
                rhs = np.concatenate([F.real, F.imag], axis=-1)
                Jm = np.where(fix2[None, :, :, None] | fix2[None, :, None, :],
                              eye, Jm)
                rhs = np.where(fix2[None], 0.0, rhs)
                # one singular matrix would fail the whole stack: rows that
                # already diverged get an identity system
                dead = ~(live & np.isfinite(Jm).all(axis=(2, 3))
                         & np.isfinite(rhs).all(axis=2))
                Jm = np.where(dead[..., None, None], eye, Jm)
                rhs = np.where(dead[..., None], 0.0, rhs)
                try:
                    dx = np.linalg.solve(Jm, rhs[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    sing = np.linalg.cond(Jm) > 1e15
                    live &= ~sing
                    Jm = np.where(sing[..., None, None], eye, Jm)
                    dx = np.linalg.solve(Jm, rhs[..., None])[..., 0]
                live &= ~dead
                va = np.angle(V) + dx[..., :S]
                vm = np.abs(V) + dx[..., S:]
                Vn = vm * np.exp(1j * va)
                Vn = np.where(fixed[None], V, Vn)
                step = np.abs(Vn - V).max(axis=2)
                V = Vn
                bad = ~np.isfinite(step)
                live &= ~bad
                it_max = max(it_max, it)
                if (step[live] <= tol).all():
                    break
            ok = live & (step <= tol)
        # merged and unused slots take their representative's voltage
        V = np.take_along_axis(V, np.broadcast_to(rep[None], V.shape), 2)
        V = np.where(o.used[lo:hi][None], V, v0[:, None, None])
        out_v[:, lo:hi] = V
        conv[:, lo:hi] = ok
    out_v[~conv] = np.nan
    return BatchLoadFlow(v_pu=out_v, converged=conv, used=o.used,
                         iterations=it_max)


BACKENDS = ("numba", "numpy", "inflated_nr", "pgm_islands", "pgm_inflated")


def batch_voltages(batch: DesignBatch, vmodel: VoltageModel, *,
                   backend: str = "numba", **kw) -> BatchLoadFlow:
    """Voltages of every design at every corner, by one backend
    (:data:`BACKENDS`)."""
    if backend == "numba":
        return numba_load_flow(batch, vmodel, **kw)
    if backend == "numpy":
        return batch_load_flow(batch, vmodel, **kw)
    if backend == "inflated_nr":
        return inflated_nr_load_flow(batch, vmodel, **kw)
    if backend in ("pgm_islands", "pgm_inflated"):
        from pyorps.collector import voltage_pgm

        fn = (voltage_pgm.islands_load_flow if backend == "pgm_islands"
              else voltage_pgm.inflated_load_flow)
        return fn(batch, vmodel, **kw)
    raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")


# -------------------------------------------------------------- check


@dataclass
class BatchVoltageCheck:
    """The limit for ``B`` designs (see
    :func:`~pyorps.collector.voltage.check_voltage` for the meaning).

    ``passed[b]`` is False where a corner did not converge
    (``converged[b]`` False)."""
    keys: list
    passed: np.ndarray
    converged: np.ndarray
    tap_kv: np.ndarray
    margin_pu: np.ndarray
    um_margin_kv: np.ndarray
    worst_turbine: np.ndarray
    max_v_pu: np.ndarray
    limits: LimitResult
    backend: str
    taps_kv: tuple[float, ...]
    tap_policy: str

    def record(self, i: int) -> dict:
        """JSON-able summary of design ``i`` for the manifest."""
        return {
            "key": repr(self.keys[i]),
            "passed": bool(self.passed[i]),
            "converged": bool(self.converged[i]),
            "tap_kv": [float(t) for t in self.tap_kv[i]],
            "margin_pu": [float(m) for m in self.margin_pu[i]],
            "um_margin_kv": float(self.um_margin_kv[i]),
            "worst_turbine": int(self.worst_turbine[i]),
            "max_v_pu": [float(v) for v in self.max_v_pu[:, i]],
            "backend": self.backend,
            "tap_policy": self.tap_policy,
        }


def check_batch(batch: DesignBatch, vmodel: VoltageModel, *,
                backend: str = "numba", tap_policy: str = "per_turbine",
                **kw) -> BatchVoltageCheck:
    """Check every design of ``batch`` against the voltage limit.

    The ``"numba"`` backend evaluates the LV voltages inside its kernel;
    every other backend hands its voltages to
    :func:`~pyorps.collector.voltage.evaluate_limits`. Both end in
    :func:`~pyorps.collector.voltage.decide_taps`.
    """
    n = batch.n
    if backend == "numba":
        lf, tap_margin, vmax = _numba_solve(batch, vmodel, **kw)
        conv = lf.converged.all(axis=0)
        um = vmodel.u_m_kv - vmax.max(axis=0) * vmodel.u_kv
        lim = decide_taps(tap_margin, um, tap_policy=tap_policy)
    else:
        lf = batch_voltages(batch, vmodel, backend=backend, **kw)
        conv = lf.converged.all(axis=0)
        v = np.where(np.isfinite(lf.v_pu), lf.v_pu, 1.0)
        mags = np.where(lf.used[None], np.abs(v), 0.0)
        vmax = mags.max(axis=2)
        s = _slot_injections(batch, vmodel)[:, 1:1 + n]
        lim = evaluate_limits(v[:, :, 1:1 + n], s, vmax, vmodel,
                              tap_policy=tap_policy)
    taps = np.asarray(vmodel.taps_kv, dtype=np.float64)
    passed = lim.passed & conv
    margin = np.where(conv[:, None], lim.margin, -math.inf)
    return BatchVoltageCheck(
        keys=list(batch.keys), passed=passed, converged=conv,
        tap_kv=taps[lim.choice], margin_pu=margin,
        um_margin_kv=np.where(conv, lim.um_margin_kv, -math.inf),
        worst_turbine=margin.argmin(axis=1), max_v_pu=vmax, limits=lim,
        backend=backend, taps_kv=tuple(float(t) for t in taps),
        tap_policy=tap_policy)

"""power-grid-model backends for the batched voltage check (C5).

Both backends solve the same model as
:func:`~pyorps.collector.voltage_batch.batch_load_flow` (pi-model branches,
PQ turbines, the UW busbar as the slack) and return a
:class:`~pyorps.collector.voltage_batch.BatchLoadFlow`, so they feed the one
limit rule unchanged.

power-grid-model updates a branch's STATUS per scenario, never its
impedance: in the component documentation (1.12 and the 1.13 fork) every
electrical parameter of ``line`` and ``generic_branch`` is marked "not
updatable"; ``BranchUpdate`` carries only ``from_status`` / ``to_status``.
A design's branch values depend on its routed lengths, so they cannot be
written into a fixed model per scenario. Two ways around it:

* :func:`islands_load_flow` -- one model holds a whole chunk of designs,
  each as its own island with its own source; the batch scenarios are only
  the operating corners (turbine P/Q and the busbar voltage), so the
  topology never changes between scenarios.
* :func:`inflated_load_flow` -- the pgf style (``powergridforge``'s
  ``grid_inflator``): one fully meshed slot network in which every branch
  of every design of the chunk is pre-added, switched off; each scenario
  (design x corner) switches on its design's branches by a sparse status
  update. The model grows with the chunk, and every scenario changes the
  topology.

Exactness details (both measured against the exact sweep):

* the source is made stiff (``sk_va``, default 1e18 VA): power-grid-model
  puts ``U^2 / sk`` behind every source, and its default ``sk`` = 1e10 VA
  moves the voltages by ~4e-4 p.u. on a 7-turbine collector;
* zero-impedance branches (a system that never leaves its trench node joins
  two busbar sections of one station) are MERGED into one node: a ``link``
  is a finite admittance and was 5.5e-7 p.u. off.
"""

from __future__ import annotations

import math
import threading as _threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyorps.collector.voltage import VoltageModel
from pyorps.collector.voltage_batch import (
    BatchLoadFlow,
    DesignBatch,
    _check_level,
    _orient,
    batch_load_flow,
)

__all__ = ["inflated_load_flow", "islands_load_flow"]

_LOCK = _threading.Lock()


def _pgm():
    try:
        import power_grid_model as pgm
    except ImportError as exc:                          # pragma: no cover
        raise ImportError("the power-grid-model backends need "
                          "`pip install power-grid-model`") from exc
    return pgm


def _branches(batch: DesignBatch, rows: slice, rep: np.ndarray):
    """Active non-zero branches of ``rows``: chunk-local design, end slots
    after merging, values."""
    bi, ei = np.nonzero(batch.active[rows])
    r = batch.r_ohm[rows][bi, ei]
    x = batch.x_ohm[rows][bi, ei]
    b = batch.b_s[rows][bi, ei]
    zero = (r == 0) & (x == 0)
    if (b[zero] != 0).any():
        raise ValueError("a zero-impedance branch with shunt susceptance")
    keep = ~zero
    bi, ei = bi[keep], ei[keep]
    u = rep[bi, batch.edge_u[ei]]
    w = rep[bi, batch.edge_v[ei]]
    return bi, u, w, r[keep], x[keep], b[keep]


def _corner_arrays(vmodel: VoltageModel):
    p = np.array([c.p_mw for c in vmodel.corners], dtype=np.float64) * 1e6
    q = np.array([c.q_mvar for c in vmodel.corners], dtype=np.float64) * 1e6
    u = np.array([c.bus_pu for c in vmodel.corners], dtype=np.float64)
    return p, q, u


def _lines(pgm, ids0, u, w, r, x, b, status, omega):
    DT, CT = pgm.DatasetType, pgm.ComponentType
    line = pgm.initialize_array(DT.input, CT.line, len(u))
    line["id"] = ids0 + np.arange(len(u))
    line["from_node"] = u
    line["to_node"] = w
    line["from_status"] = status
    line["to_status"] = status
    line["r1"] = r
    line["x1"] = x
    line["c1"] = b / omega
    line["tan1"] = 0.0
    line["i_n"] = 1e6
    return line


def _source(pgm, ids0, nodes, sk_va):
    DT, CT = pgm.DatasetType, pgm.ComponentType
    src = pgm.initialize_array(DT.input, CT.source, len(nodes))
    src["id"] = ids0 + np.arange(len(nodes))
    src["node"] = nodes
    src["status"] = 1
    src["u_ref"] = 1.0
    src["sk"] = sk_va
    return src


def _gens(pgm, ids0, nodes):
    DT, CT = pgm.DatasetType, pgm.ComponentType
    gen = pgm.initialize_array(DT.input, CT.sym_gen, len(nodes))
    gen["id"] = ids0 + np.arange(len(nodes))
    gen["node"] = nodes
    gen["status"] = 1
    gen["type"] = 0                                   # constant power
    gen["p_specified"] = 0.0
    gen["q_specified"] = 0.0
    return gen


def _add(stats, key, dt):
    if stats is not None:
        with _LOCK:
            stats[key] = stats.get(key, 0.0) + dt


def _run(pgm, input_data, update, vmodel, *, tol, max_iter, method,
         threading, stats=None):
    CT = pgm.ComponentType
    t0 = time.perf_counter()
    model = pgm.PowerGridModel(input_data, system_frequency=vmodel.f_hz)
    t1 = time.perf_counter()
    _add(stats, "model_s", t1 - t0)
    out = model.calculate_power_flow(
        update_data=update, symmetric=True, error_tolerance=tol,
        max_iterations=max_iter, calculation_method=method,
        threading=threading, continue_on_batch_error=True,
        output_component_types={CT.node: ["u_pu", "u_angle"]})
    _add(stats, "calc_s", time.perf_counter() - t1)
    err = model.batch_error
    failed = (np.asarray(err.failed_scenarios, dtype=np.int64)
              if err is not None else np.zeros(0, dtype=np.int64))
    node = out[CT.node]
    return node["u_pu"] * np.exp(1j * node["u_angle"]), failed


def _fallback(batch, vmodel, idx, v_out, conv_out, tol):
    """Re-solve designs PGM could not take with the exact sweep, so a PGM
    failure is never reported as a design failure by itself."""
    if len(idx) == 0:
        return
    lf = batch_load_flow(batch.subset(idx), vmodel, tol_pu=tol)
    v_out[:, idx] = lf.v_pu
    conv_out[:, idx] = lf.converged


def _finish(vmodel, o, v, conv):
    u = np.array([c.bus_pu for c in vmodel.corners])
    v = np.where(o.used[None], v, u[:, None, None])
    v[~conv] = np.nan
    return BatchLoadFlow(v_pu=v, converged=conv, used=o.used, iterations=-1)


def _map(chunks, fn, threads):
    if threads > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(fn, chunks))
    else:
        for c in chunks:
            fn(c)


def islands_load_flow(batch: DesignBatch, vmodel: VoltageModel, *,
                      chunk: int = 2048, threads: int = 1,
                      pgm_threading: int = -1,
                      method: str = "newton_raphson",
                      tol_pu: float | None = None, max_iter: int = 30,
                      sk_va: float = 1e18,
                      stats: dict | None = None) -> BatchLoadFlow:
    """Chunks of designs as islands of one model; corners as scenarios.

    ``threads`` chunks run at once (power-grid-model releases the GIL);
    ``pgm_threading`` is power-grid-model's own scenario threading.
    ``stats`` (optional) collects seconds spent building the input and
    update arrays (``build_s``), constructing models (``model_s``) and
    calculating (``calc_s``), summed over chunks.
    """
    pgm = _pgm()
    _check_level(batch, vmodel)
    o = _orient(batch, vmodel.s_base_mva)
    CT, DT = pgm.ComponentType, pgm.DatasetType
    B, S, n = len(batch), batch.n_slots, batch.n
    C = len(vmodel.corners)
    tol = vmodel.tol_pu if tol_pu is None else tol_pu
    p_c, q_c, u_c = _corner_arrays(vmodel)
    omega = 2.0 * math.pi * vmodel.f_hz
    v_out = np.empty((C, B, S), dtype=np.complex128)
    conv_out = np.ones((C, B), dtype=bool)

    def one(rows: slice):
        t_build = time.perf_counter()
        K = rows.stop - rows.start
        rep = o.rep[rows]
        bi, u, w, r, x, b = _branches(batch, rows, rep)
        node = pgm.initialize_array(DT.input, CT.node, K * S)
        node["id"] = np.arange(K * S)
        node["u_rated"] = batch.u_kv * 1e3
        line = _lines(pgm, K * S, bi * S + u, bi * S + w, r, x, b, 1, omega)
        nid = K * S + len(line)
        src = _source(pgm, nid, np.arange(K) * S, sk_va)
        nid += K
        gnode = (np.arange(K)[:, None] * S + rep[:, 1:1 + n]).ravel()
        gen = _gens(pgm, nid, gnode)
        g_upd = pgm.initialize_array(DT.update, CT.sym_gen, (C, K * n))
        g_upd["id"] = gen["id"][None, :]
        g_upd["p_specified"] = np.tile(p_c, (1, K))
        g_upd["q_specified"] = np.tile(q_c, (1, K))
        s_upd = pgm.initialize_array(DT.update, CT.source, (C, K))
        s_upd["id"] = src["id"][None, :]
        s_upd["u_ref"] = u_c[:, None]
        _add(stats, "build_s", time.perf_counter() - t_build)
        v, failed = _run(
            pgm, {CT.node: node, CT.line: line, CT.source: src,
                  CT.sym_gen: gen},
            {CT.sym_gen: g_upd, CT.source: s_upd}, vmodel, tol=tol,
            max_iter=max_iter, method=method, threading=pgm_threading,
            stats=stats)
        v = v.reshape(C, K, S)
        v_out[:, rows] = np.take_along_axis(
            v, np.broadcast_to(rep[None], v.shape), 2)
        if failed.size:
            # one failing island fails the corner of the whole chunk
            _fallback(batch, vmodel, np.arange(rows.start, rows.stop),
                      v_out, conv_out, tol)

    _map([slice(i, min(i + chunk, B)) for i in range(0, B, chunk)], one,
         threads)
    return _finish(vmodel, o, v_out, conv_out)


def inflated_load_flow(batch: DesignBatch, vmodel: VoltageModel, *,
                       chunk: int = 32, threads: int = 1,
                       pgm_threading: int = -1,
                       method: str = "newton_raphson",
                       tol_pu: float | None = None, max_iter: int = 30,
                       sk_va: float = 1e18,
                       stats: dict | None = None) -> BatchLoadFlow:
    """The fully meshed slot network with every branch of a chunk
    pre-added and switched per scenario (design x corner).

    Turbine generators sit on their fixed slots, so a design whose turbine
    slot is merged away (a zero-impedance branch at a turbine, which a
    ``D_share`` design never has) is solved by the exact sweep instead.
    """
    pgm = _pgm()
    _check_level(batch, vmodel)
    o = _orient(batch, vmodel.s_base_mva)
    CT, DT = pgm.ComponentType, pgm.DatasetType
    B, S, n = len(batch), batch.n_slots, batch.n
    C = len(vmodel.corners)
    tol = vmodel.tol_pu if tol_pu is None else tol_pu
    p_c, q_c, u_c = _corner_arrays(vmodel)
    omega = 2.0 * math.pi * vmodel.f_hz
    v_out = np.empty((C, B, S), dtype=np.complex128)
    conv_out = np.ones((C, B), dtype=bool)
    moved = (o.rep[:, 1:1 + n] != np.arange(1, 1 + n)[None, :]).any(axis=1)

    def one(rows: slice):
        t_build = time.perf_counter()
        K = rows.stop - rows.start
        rep = o.rep[rows]
        bi, u, w, r, x, b = _branches(batch, rows, rep)
        node = pgm.initialize_array(DT.input, CT.node, S)
        node["id"] = np.arange(S)
        node["u_rated"] = batch.u_kv * 1e3
        line = _lines(pgm, S, u, w, r, x, b, 0, omega)
        nid = S + len(line)
        src = _source(pgm, nid, np.zeros(1, dtype=np.int64), sk_va)
        gen = _gens(pgm, nid + 1, 1 + np.arange(n))
        # scenario (k, c) = k * C + c switches on design k's lines
        cnt = np.bincount(bi, minlength=K)                 # bi is sorted
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        sd = np.repeat(np.arange(K), C)
        c_s = cnt[sd]
        indptr = np.concatenate([[0], np.cumsum(c_s)]).astype(np.int64)
        tot = int(indptr[-1])
        pick = (np.repeat(start[sd], c_s)
                + np.arange(tot) - np.repeat(indptr[:-1], c_s))
        l_upd = pgm.initialize_array(DT.update, CT.line, tot)
        l_upd["id"] = line["id"][pick]
        l_upd["from_status"] = 1
        l_upd["to_status"] = 1
        cs = np.tile(np.arange(C), K)
        g_upd = pgm.initialize_array(DT.update, CT.sym_gen, (K * C, n))
        g_upd["id"] = gen["id"][None, :]
        g_upd["p_specified"] = p_c[cs]
        g_upd["q_specified"] = q_c[cs]
        s_upd = pgm.initialize_array(DT.update, CT.source, (K * C, 1))
        s_upd["id"] = src["id"][0]
        s_upd["u_ref"] = u_c[cs][:, None]
        _add(stats, "build_s", time.perf_counter() - t_build)
        v, failed = _run(
            pgm, {CT.node: node, CT.line: line, CT.source: src,
                  CT.sym_gen: gen},
            {CT.sym_gen: g_upd, CT.source: s_upd,
             CT.line: {"indptr": indptr, "data": l_upd}},
            vmodel, tol=tol, max_iter=max_iter, method=method,
            threading=pgm_threading, stats=stats)
        v = v.reshape(K, C, S).transpose(1, 0, 2)
        v_out[:, rows] = np.take_along_axis(
            v, np.broadcast_to(rep[None], v.shape), 2)
        bad = moved[rows].copy()
        if failed.size:
            bad[np.unique(failed // C)] = True
        _fallback(batch, vmodel, np.arange(rows.start, rows.stop)[bad],
                  v_out, conv_out, tol)

    _map([slice(i, min(i + chunk, B)) for i in range(0, B, chunk)], one,
         threads)
    return _finish(vmodel, o, v_out, conv_out)

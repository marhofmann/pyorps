"""Shared helpers for the runtime study (step R, plan 2026-10-05-runtime-and-pruning-study.md).

Nothing here starts a run. The scripts that import it (``r1_measure.py``, ``r2_parallel.py``) are
announced to the owner before they are started, need an idle machine for any timing, and write only
small JSON files (no binary, no raster).

Rules built in:
* BLAS and OpenMP threads are pinned to one (set before numpy is imported), or every timing is wrong.
* ``require_idle`` measures the system-wide CPU load; timings are refused while the machine is busy
  (owner rule: no wall-clock numbers while the machine is busy). ``--allow-busy`` keeps counts, drops times.
* A resident-memory guard (``--max-gb``) and a per-job time limit stop a job cleanly instead of
  filling the machine (the owner keeps headroom for his own work).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (after the thread pins)
import psutil  # noqa: E402

CASE = Path(__file__).resolve().parents[2]
REPO = CASE.parents[1]
RASTER = CASE / "data" / "rasters" / "cost_priced_1m.tif"
INPUTS = CASE / "config" / "collector_inputs_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
LOSS_TABLE = CASE / "data" / "loss_categories.npz"
RESULTS = CASE / "results" / "runtime_study"
EXCLUDED = 65535
# the farm window of the site raster (build_site_raster.py): column/row of its top-left cell on the 1 m grid
# (origin x 431373, y top 5593379; farm window x from 441701, y top 5587711)
FARM_COL, FARM_ROW = 441701 - 431373, 5593379 - 5587711
IDLE_LIMIT_PCT = 15.0


class JobAborted(RuntimeError):
    """A job stopped by the memory guard or its time limit (partial counts are kept)."""


def require_idle(seconds: float = 10.0, limit_pct: float = IDLE_LIMIT_PCT) -> tuple[bool, float]:
    """System-wide mean CPU load over ``seconds``; idle if below ``limit_pct``."""
    load = psutil.cpu_percent(interval=seconds)
    return load < limit_pct, float(load)


def sha256_of(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def load_window(cells: int, *, col: int | None = None, row: int | None = None) -> np.ndarray:
    """A square window of about ``cells`` cells of the priced 1 m cost raster (uint16, EUR/m;
    65535 forbidden), read with rasterio (the file is 7.5 MB). Default position: the farm window's corner."""
    import rasterio
    from rasterio.windows import Window

    side = max(8, int(round(cells ** 0.5)))
    c0 = FARM_COL if col is None else col
    r0 = FARM_ROW if row is None else row
    with rasterio.open(RASTER) as src:
        if c0 + side > src.width or r0 + side > src.height:
            raise ValueError("window leaves the raster")
        arr = src.read(1, window=Window(c0, r0, side, side))
    return np.ascontiguousarray(arr, dtype=np.uint16)


def pick_turbines(vals: np.ndarray, n: int, seed: int = 1, min_gap: int = 3) -> list[int]:
    """``n`` free cells (flat indices) at least ``min_gap`` cells apart, seeded. The turbines are placed
    at random inside the window: the cost texture is real, the layout is a stand-in (the real
    turbines are kilometres apart and need the pruned windows of step R4)."""
    rng = np.random.default_rng(seed)
    rows, cols = vals.shape
    free = np.flatnonzero(vals.ravel() != EXCLUDED)
    if free.size < n:
        raise ValueError("not enough free cells")
    chosen: list[int] = []
    for idx in rng.permutation(free):
        r, c = divmod(int(idx), cols)
        if all(max(abs(r - divmod(o, cols)[0]), abs(c - divmod(o, cols)[1])) >= min_gap for o in chosen):
            chosen.append(int(idx))
            if len(chosen) == n:
                return chosen
    raise ValueError("could not place the turbines; use a larger window")


def build_base(u_kv: float = 33.0):
    """The real built model at n = 7 for one MV level (tagged stand-ins allowed: ``certified=False``)."""
    from pyorps.costmodel import CostCatalogue, LossCategories
    from pyorps.costmodel.collector_inputs import CollectorInputs, build_collector_model

    return build_collector_model(
        CollectorInputs.load(INPUTS), CostCatalogue.load(CATALOGUE), LossCategories.load(LOSS_TABLE),
        u_kv=u_kv, rated_mw=[6.8] * 7)


def model_for(base, n: int, *, m_max: int | None = None, p_max: int | None = None):
    """The base model's cables, prices, derating and rules for ``n`` identical turbines (n = 7 reuses
    the base unchanged apart from m_max and p_max). For n < 7 the loss weight is ``|X|^2 W_1`` with
    ``W_1`` the base model's full-set weight over 49, a stand-in for the joint table, which exists
    only for 7 turbines. Per-operation costs do not depend on the weights; state counts may differ
    slightly from the real model and are measured, not assumed."""
    import dataclasses

    from pyorps.collector.model import CollectorModel

    b = base.model
    kw = {f.name: getattr(b, f.name) for f in dataclasses.fields(b)
          if f.init and f.name not in ("n", "current_a", "loss_weight")}
    if m_max is not None:
        kw["m_max"] = m_max
    if p_max is not None:
        kw["p_max"] = p_max
    if n == b.n:
        return CollectorModel(n=n, current_a=b.current_a, loss_weight=b.loss_weight, **kw)
    w1 = b.loss_weight[(1 << b.n) - 1] / float(b.n ** 2)
    return CollectorModel.identical_turbines(n, current_one_a=b.current_a[1], loss_one=w1, **kw)


@dataclass
class Meter:
    """What the patched drain and junction functions record during one job."""
    t0: float = field(default_factory=time.perf_counter)
    limit_s: float = 1800.0
    max_gb: float = 4.0
    time_on: bool = True
    drain_s: float = 0.0
    drain_calls: int = 0
    drain_cells_seeded: int = 0
    junction_s: float = 0.0
    junction_arcs: int = 0
    peak_rss_gb: float = 0.0
    _proc: psutil.Process = field(default_factory=lambda: psutil.Process(os.getpid()))

    def check(self) -> None:
        rss = self._proc.memory_info().rss / 1e9
        self.peak_rss_gb = max(self.peak_rss_gb, rss)
        if rss > self.max_gb:
            raise JobAborted(f"memory guard: {rss:.2f} GB > {self.max_gb} GB")
        if time.perf_counter() - self.t0 > self.limit_s:
            raise JobAborted(f"time limit {self.limit_s} s")


@contextmanager
def instrumented(meter: Meter):
    """Patch ``raster.drain`` and the two algebras' ``junctions`` to count and time them."""
    import pyorps.collector.raster as raster
    import pyorps.collector.reference as reference

    orig_drain = raster.drain
    orig_j = {c: c.junctions for c in (reference._SetAlgebra, reference._CountAlgebra)}  # pylint: disable=protected-access

    def drain(values, steps, cells, labels, **kw):
        meter.check()
        t = time.perf_counter()
        out = orig_drain(values, steps, cells, labels, **kw)
        meter.drain_s += time.perf_counter() - t
        meter.drain_calls += 1
        meter.drain_cells_seeded += int(np.size(cells))
        return out

    def wrap(cls, fn):
        def junctions(self, L):
            t = time.perf_counter()
            res = list(fn(self, L))
            meter.junction_s += time.perf_counter() - t
            meter.junction_arcs += len(res)
            return res
        return junctions

    raster.drain = drain
    for c, fn in orig_j.items():
        c.junctions = wrap(c, fn)
    try:
        yield meter
    finally:
        raster.drain = orig_drain
        for c, fn in orig_j.items():
            c.junctions = fn


def run_collector(vals, turbines, model, *, engine: str, k: int, meter: Meter) -> dict:
    """One RasterCollector run, instrumented; returns counts always, times only when ``meter.time_on``."""
    from pyorps.collector.raster import RasterCollector
    from pyorps.utils.neighborhood import get_neighborhood_steps

    steps = np.asarray(get_neighborhood_steps(k, directed=True), dtype=np.int8)
    rc = RasterCollector(vals, steps, 1.0, turbines, model, engine=engine, drain_engine="auto")
    status, err = "ok", None
    t = time.perf_counter()
    try:
        with instrumented(meter):
            rc.run()
    except JobAborted as exc:
        status, err = "aborted", str(exc)
    wall = time.perf_counter() - t
    if status == "ok":
        meter.check()
    out = {
        "status": status, "error": err, "cells": int(vals.size), "engine": engine, "n": model.n,
        "m_max": model.m_max, "p_max": model.p_max, "neighbourhood_k": k,
        "stats": {k_: (int(v) if isinstance(v, (int, np.integer)) else v) for k_, v in rc.stats.items()},
        "junction_arcs": meter.junction_arcs, "drain_calls": meter.drain_calls,
        "drain_cells_seeded": meter.drain_cells_seeded, "peak_rss_gb": round(meter.peak_rss_gb, 3),
    }
    if meter.time_on:
        out.update(wall_s=wall, drain_s=meter.drain_s, junction_s=meter.junction_s,
                   node_ops_s=wall - meter.drain_s)
    return out


def save(name: str, payload: dict) -> Path:
    """Write one small JSON result (a few KB)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


class LoadSampler:
    """Samples, every ``every`` seconds, the system CPU load and this process tree's share of it, so a
    result can say how busy the machine was with OTHER work during the job (the owner may return mid-run).
    ``others_mean_pct`` above ``CONTAMINATED_PCT`` marks a timing as contaminated."""
    CONTAMINATED_PCT = 10.0

    def __init__(self, every: float = 5.0):
        import threading

        self.every, self._stop = every, threading.Event()
        self._root = psutil.Process(os.getpid())
        self._known: dict = {}
        self.samples: list[tuple[float, float]] = []     # (system %, own %), both of all logical CPUs
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _own(self) -> float:
        """This process tree's CPU share (percent of all logical CPUs) since the last call. The Process
        objects are kept: psutil's first cpu_percent call on a new object always returns 0."""
        procs = {self._root.pid: self._root}
        try:
            for c in self._root.children(recursive=True):
                procs[c.pid] = self._known.setdefault(c.pid, c)
        except psutil.Error:
            pass
        self._known = {pid: pr for pid, pr in procs.items() if pid != self._root.pid}
        tot = 0.0
        for p in procs.values():
            try:
                tot += p.cpu_percent(None)
            except psutil.Error:
                pass
        return tot / psutil.cpu_count()

    def _loop(self) -> None:
        psutil.cpu_percent(None)
        self._own()
        while not self._stop.wait(self.every):
            self.samples.append((psutil.cpu_percent(None), self._own()))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.every + 1)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0, "others_mean_pct": None, "contaminated": None}
        others = [max(0.0, s - o) for s, o in self.samples]
        mean = float(np.mean(others))
        return {"samples": len(others), "others_mean_pct": round(mean, 1),
                "others_max_pct": round(float(max(others)), 1), "own_mean_pct": round(float(np.mean([o for _, o in self.samples])), 1),
                "contaminated": mean > self.CONTAMINATED_PCT}


def header(allow_busy: bool, idle_seconds: float) -> dict:
    """Provenance for every result file: machine state at the start, inputs, versions."""
    import platform

    idle, load = require_idle(idle_seconds)
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu_load_pct_before": round(load, 1),
        "machine_idle": bool(idle), "timings_kept": bool(idle),
        "raster": RASTER.name, "raster_sha256": sha256_of(RASTER), "python": platform.python_version(),
        "numpy": np.__version__, "cpus_logical": psutil.cpu_count(), "free_ram_gb": round(psutil.virtual_memory().available / 1e9, 1),
        "env_threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }

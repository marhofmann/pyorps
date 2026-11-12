"""R2: how far the long runs parallelise (plan 2026-10-05-runtime-and-pruning-study.md, section 5).

Three independent measurements, each at 1, 2, 4 and 8 workers (never more than ``--max-workers``, default 8,
the owner's limit for the paper's runs):

* ``node``    cells inside a node operation: the merge pass ``min(seed, where(st, a + b, inf))`` cut into cell
              chunks on a thread pool (numpy releases the GIL on large arrays); window 1e5, 1e6, 1e7 cells;
              result compared BITWISE with the one-thread result. Shows the memory-bandwidth limit.
* ``drains``  16 independent seeded drains of a 1e6-cell crop of the real cost raster on a thread pool; result
              compared bitwise with the serial run. Shows whether the kernel drain runs concurrently.
* ``procs``   P processes, each running the same small collector job (n = 4, 1e4 cells, engine B); throughput
              against P. Shows whether independent runs (levels, options, scenarios) scale.

For each: wall seconds (best of ``--repeats``), speed-up, and the Amdahl serial fraction ``f = (1/S - 1/P) / (1 - 1/P)``.
Same safety rules as ``r1_measure.py``: timings only on an idle machine (else refused, or ``--allow-busy`` checks the
bitwise equality only), BLAS pinned to one thread, memory guard, small JSON output, nothing else written.

Usage::

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/runtime_study/r2_parallel.py --plan
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/runtime_study/r2_parallel.py [--tests node,drains,procs]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

INF = float("inf")


def _amdahl(speedups: dict[int, float]) -> dict[int, float | None]:
    return {p: (None if p == 1 else float((1 / s - 1 / p) / (1 - 1 / p))) for p, s in speedups.items()}


def _best(fn, repeats: int) -> float:
    best = INF
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# --------------------------------------------------------------------- node operations
def _merge_chunk(a, b, st, seed, out, lo, hi):
    np.minimum(seed[lo:hi], np.where(st[lo:hi], a[lo:hi] + b[lo:hi], INF), out=out[lo:hi])


def test_node(workers, repeats, timed) -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for N in (100_000, 1_000_000, 10_000_000):
        a, b = rng.random(N) * 1e3, rng.random(N) * 1e3
        st = rng.random(N) > 0.1
        seed = np.full(N, INF)
        reps = max(1, 20_000_000 // N)               # about the same work per timing at every size
        ref = np.empty(N)
        _merge_chunk(a, b, st, seed, ref, 0, N)
        times, equal = {}, {}
        for T in workers:
            out = np.empty(N)
            bounds = np.linspace(0, N, T + 1).astype(int)

            def once():
                for _ in range(reps):
                    if T == 1:
                        _merge_chunk(a, b, st, seed, out, 0, N)
                    else:
                        list(pool.map(lambda i: _merge_chunk(a, b, st, seed, out, bounds[i], bounds[i + 1]), range(T)))

            with ThreadPoolExecutor(T) as pool:
                once()
                equal[T] = bool(np.array_equal(out, ref))
                if timed:
                    times[T] = _best(once, repeats) / reps
        row = {"test": "node", "cells": N, "bitwise_equal": equal}
        if timed:
            sp = {T: times[1] / times[T] for T in workers}
            row.update(seconds_per_pass=times, speedup=sp, amdahl_serial_fraction=_amdahl(sp),
                       ns_per_cell_one_thread=times[1] / N * 1e9)
        rows.append(row)
        print(row, flush=True)
    return rows


# --------------------------------------------------------------------- drains
def test_drains(workers, repeats, timed) -> list[dict]:
    from pyorps.graph.drain import drain
    from pyorps.utils.neighborhood import get_neighborhood_steps

    vals = C.load_window(1_000_000)
    steps = np.asarray(get_neighborhood_steps(3, directed=True), dtype=np.int8)
    rng = np.random.default_rng(2)
    free = np.flatnonzero(vals.ravel() != C.EXCLUDED)
    K = 16
    jobs = []
    for _ in range(K):
        cells = rng.choice(free, size=50, replace=False)
        jobs.append((cells, rng.random(50) * 100.0))
    drain(vals, steps, *jobs[0])                       # warm-up (loads and caches the kernel)

    def one(j):
        return drain(vals, steps, jobs[j][0], jobs[j][1], length_rate=0.0, weight_mult=1.0)

    ref = [one(j) for j in range(K)]
    times, equal = {}, {}
    for P in workers:
        res: list = []

        def once():
            with ThreadPoolExecutor(P) as pool:
                res[:] = list(pool.map(one, range(K)))

        once()
        equal[P] = bool(all(np.array_equal(x, y) for x, y in zip(res, ref)))
        if timed:
            times[P] = _best(once, repeats)
    row = {"test": "drains", "cells": int(vals.size), "drains": K, "bitwise_equal": equal}
    if timed:
        sp = {P: times[1] / times[P] for P in workers}
        row.update(seconds=times, speedup=sp, amdahl_serial_fraction=_amdahl(sp))
    print(row, flush=True)
    return [row]


# --------------------------------------------------------------------- processes
def _proc_job(seed: int) -> tuple[float, int, str]:
    vals = C.load_window(10_000)
    turb = C.pick_turbines(vals, 4, seed=seed)
    base = C.build_base(33.0)
    model = C.model_for(base, 4, m_max=4, p_max=2)
    meter = C.Meter(limit_s=900, max_gb=1.5, time_on=True)
    t = time.perf_counter()
    r = C.run_collector(vals, turb, model, engine="B", k=3, meter=meter)
    import hashlib
    return time.perf_counter() - t, r["stats"]["merge_pairs"], hashlib.sha256(repr(sorted(r["stats"].items())).encode()).hexdigest()[:12]


def test_procs(workers, repeats, timed) -> list[dict]:
    if not timed:
        return [{"test": "procs", "skipped": "timings need an idle machine"}]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    times, serial = {}, None
    for P in workers:
        best = INF
        for _ in range(max(1, repeats - 1)):
            t = time.perf_counter()
            with ProcessPoolExecutor(P, mp_context=ctx) as pool:
                out = list(pool.map(_proc_job, [1] * P))      # the same job P times: identical counts expected
            best = min(best, time.perf_counter() - t)
        times[P] = best
        if P == 1:
            serial = out[0][2]
        assert all(o[2] == serial for o in out), "counts differ between processes"
    thr = {P: P * times[1] / times[P] for P in workers}      # jobs per unit time relative to one process
    row = {"test": "procs", "job": "n=4, 1e4 cells, engine B, k=3", "wall_s": times,
           "throughput_vs_1_process": thr, "efficiency": {P: thr[P] / P for P in workers},
           "amdahl_serial_fraction": _amdahl(thr)}
    print(row, flush=True)
    return [row]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default="node,drains,procs")
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--allow-busy", action="store_true", help="check bitwise equality only, keep no seconds")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    args = ap.parse_args()
    workers = [w for w in map(int, args.workers.split(",")) if w <= args.max_workers]
    tests = args.tests.split(",")
    if args.plan:
        print("tests", tests, "workers", workers, "repeats", args.repeats,
              "\nwrites one JSON of a few KB; the process test starts up to", max(workers), "processes of ~0.1 GB each")
        return
    head = C.header(args.allow_busy, args.idle_seconds)
    if not head["machine_idle"] and not args.allow_busy:
        print(f"machine busy (CPU load {head['cpu_load_pct_before']} %): timings refused.")
        sys.exit(3)
    timed = head["timings_kept"]
    out = {"header": head, "workers": workers, "results": []}
    with C.LoadSampler() as ls:
        for t in tests:
            out["results"] += {"node": test_node, "drains": test_drains, "procs": test_procs}[t](workers, args.repeats, timed)
    out["load"] = ls.summary()
    print("saved", C.save(f"r2_parallel_{time.strftime('%Y%m%d_%H%M')}.json", out))


if __name__ == "__main__":
    main()

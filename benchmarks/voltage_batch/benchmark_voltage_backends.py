"""Benchmark the batched voltage-check backends (plan C5).

Every backend solves the same exact AC load flow of B random 7-turbine
collector designs at the four corners (H1, L1, H0, N1) on the fully meshed
slot network (1 UW + 7 turbines + up to 6 junction slots = 14 slots, 91
potential branches), then the one limit rule
(:func:`pyorps.collector.voltage.evaluate_limits`) runs on the voltages.

Backends: the scalar sweep per design (``check_tree`` in a loop), the
compiled sweep (numba, 1/4/8 threads), the numpy sweep, Newton--Raphson on
the inflated dense network, and power-grid-model 1.12 as islands and as the
inflated pgf-style model (status switching).

Needs power-grid-model for the PGM rows; the pyorps venv does not have it,
so run it with powergridforge's venv and this repository on the path::

    PYTHONPATH=. ../powergridforge/.venv/Scripts/python.exe \\
        benchmarks/voltage_batch/benchmark_voltage_backends.py

Writes ``benchmarks/voltage_batch/results/backends_<date>.json`` and prints a
Markdown table. At most 8 of the machine's threads are used.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

from pyorps.collector.voltage import (
    Branch,
    ElectricalTree,
    check_tree,
    voltage_model_from_yaml,
)
from pyorps.collector.voltage_batch import (
    DesignBatch,
    batch_voltages,
    check_batch,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
YAML = ROOT / "case_studies/runkel_free_siting/config/voltage_2026.yaml"
OMEGA = 2 * math.pi * 50


def random_trees(rng, B, n, J, vm, zero_p=0.01):
    trees = []
    for _ in range(B):
        j = rng.randint(0, J)
        labels = ([("root", 0)] + [("turbine", t) for t in range(n)]
                  + [("junction", k) for k in range(j)])
        order = list(range(1, len(labels)))
        rng.shuffle(order)
        placed, br = [0], []
        for node in order:
            par = rng.choice(placed)
            placed.append(node)
            if node > n and rng.random() < zero_p:     # junction-junction
                br.append(Branch(node, par, 0.0, 0.0, 0.0))
                continue
            L = rng.uniform(200, 3000)
            c = rng.choice(vm.cables)
            p = rng.choice([1, 2])
            br.append(Branch(node, par, c.r_ohm_per_m * L / p,
                             c.x_ohm_per_m * L / p,
                             OMEGA * c.c_f_per_m * L * p))
        trees.append(ElectricalTree(vm.u_kv, labels, br,
                                    {t: 1 + t for t in range(n)}))
    return trees


def best_of(fn, reps):
    fn()                                           # warm-up (numba, caches)
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


def variants(B, have_pgm):
    v = [("numba", "1 thread", {"threads": 1}),
         ("numba", "4 threads", {"threads": 4}),
         ("numba", "8 threads", {"threads": 8}),
         ("numpy", "", {}),
         ("inflated_nr", "chunk 1024", {"chunk": 1024})]
    if have_pgm:
        v += [("pgm_islands", "1 model per 2048 designs", {"chunk": 2048}),
              ("pgm_islands", "4 threads", {"chunk": max(1, -(-B // 4)),
                                            "threads": 4}),
              ("pgm_inflated", "chunk 32", {"chunk": 32}),
              ("pgm_inflated", "chunk 32, PGM 4 threads",
               {"chunk": 32, "pgm_threading": 4}),
              ("pgm_inflated", "chunk 8", {"chunk": 8}),
              ("pgm_inflated", "chunk 128", {"chunk": 128})]
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="1,10,100,1000,10000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scalar-max", type=int, default=1000,
                    help="largest batch the scalar loop is timed on")
    ap.add_argument("--slow-max", type=int, default=10000,
                    help="largest batch for inflated_nr and pgm_inflated")
    args = ap.parse_args(argv)
    try:
        import power_grid_model  # noqa: F401
        from importlib.metadata import version
        pgm_version = version("power-grid-model")
        have_pgm = True
    except ImportError:
        pgm_version, have_pgm = None, False
    import numba
    vm = voltage_model_from_yaml(YAML, u_kv=20,
                                 cable_mm2=[150, 240, 400, 630],
                                 n_turbines=7)
    sizes = [int(s) for s in args.sizes.split(",")]
    rng = random.Random(2026)
    all_trees = random_trees(rng, max(sizes), 7, 6, vm)
    rows, meta = [], {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "machine": platform.processor(), "cpus": os.cpu_count(),
        "python": platform.python_version(), "numpy": np.__version__,
        "numba": numba.__version__, "power_grid_model": pgm_version,
        "reps": args.reps, "turbines": 7, "slots": 14, "edges": 91,
        "corners": [c.name for c in vm.corners],
    }
    for B in sizes:
        trees = all_trees[:B]
        t_pack, batch = best_of(
            lambda: DesignBatch.from_trees(trees, n=7, n_slots=14),
            args.reps)
        ref = batch_voltages(batch, vm, backend="numba", threads=8)
        base = {"B": B, "pack_us_per_design": 1e6 * t_pack / B}
        if B <= args.scalar_max:
            t, _ = best_of(lambda: [check_tree(t, vm) for t in trees],
                           max(1, args.reps if B <= 100 else 1))
            rows.append({**base, "backend": "scalar", "variant":
                         "check_tree per design", "seconds": t,
                         "us_per_design": 1e6 * t / B, "max_dv": 0.0})
        for be, label, kw in variants(B, have_pgm):
            if be in ("inflated_nr", "pgm_inflated") and B > args.slow_max:
                continue
            stats = {} if be.startswith("pgm") else None
            extra = {"stats": stats} if stats is not None else {}
            t, lf = best_of(lambda: batch_voltages(batch, vm, backend=be,
                                                   **kw, **extra),
                            args.reps)
            t_chk, _ = best_of(lambda: check_batch(batch, vm, backend=be,
                                                   **kw), args.reps)
            d = float(np.nanmax(np.abs(np.where(lf.used[None],
                                                lf.v_pu - ref.v_pu, 0))))
            row = {**base, "backend": be, "variant": label, "seconds": t,
                   "us_per_design": 1e6 * t / B,
                   "check_seconds": t_chk, "max_dv": d,
                   "converged": bool(lf.converged.all())}
            if stats:
                runs = args.reps + 1
                row.update({k: v / runs for k, v in stats.items()})
            rows.append(row)
            print(f"B={B:6d} {be:13s} {label:26s} {1e6 * t / B:10.2f} "
                  f"us/design  |dV| {d:.1e}", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    f = out / f"backends_{_dt.date.today().isoformat()}.json"
    f.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1),
                 encoding="utf-8")
    print(f"\nwrote {f}\n")
    print_table(rows, sizes)


def print_table(rows, sizes):
    keys = []
    for r in rows:
        k = (r["backend"], r["variant"])
        if k not in keys:
            keys.append(k)
    head = "| backend | variant | " + " | ".join(f"B={b}" for b in sizes)
    print(head + " | max \\|dV\\| |")
    print("|---|---|" + "---:|" * len(sizes) + "---:|")
    for k in keys:
        cells, dv = [], 0.0
        for b in sizes:
            r = next((r for r in rows if r["B"] == b
                      and (r["backend"], r["variant"]) == k), None)
            cells.append("-" if r is None else f"{r['us_per_design']:.1f}")
            if r is not None:
                dv = max(dv, r["max_dv"])
        print(f"| {k[0]} | {k[1]} | " + " | ".join(cells)
              + f" | {dv:.0e} |")
    print("\n(microseconds per design, 4 corners each; best of the runs)")
    print("\nFull check (load flow + taps + limits), microseconds per design:")
    print("| backend | variant | " + " | ".join(f"B={b}" for b in sizes)
          + " |")
    print("|---|---|" + "---:|" * len(sizes))
    for k in keys:
        cells = []
        for b in sizes:
            r = next((r for r in rows if r["B"] == b
                      and (r["backend"], r["variant"]) == k), None)
            t = None if r is None else r.get("check_seconds", r["seconds"])
            cells.append("-" if t is None else f"{1e6 * t / b:.1f}")
        print(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

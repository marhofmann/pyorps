"""Benchmark of the batched voltage-check backends (C5), rewritten 2026-10-20.

The first version (results ``backends_2026-09-25.json``, log deviation 21) was never committed and is lost; this one
rebuilds it from the test helpers. It measures the full check (load flow at every corner, taps, band) in microseconds per
7-turbine design, best of ``--repeats``, for the pyorps backends and the two power-grid-model backends.

Run it only on an idle machine (no other heavy job), from the repository root::

    PYTHONPATH=<folder with power-grid-model 1.13.199> NUMBA_NUM_THREADS=4 \
        .venv/Scripts/python.exe benchmarks/voltage_batch/benchmark_voltage_backends.py --threads 4

The script refuses to report timings when the machine is busy (CPU load above ``--max-load`` before the run) unless
``--force`` is given, and writes ``results/backends_<date>.json`` with the pgm version.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _cpu_load() -> float:
    import subprocess

    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
                         capture_output=True, text=True).stdout.strip()
    return float(out or "nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 1000, 10000])
    ap.add_argument("--backends", nargs="+", default=["scalar", "numba", "numpy", "inflated_nr", "pgm_islands", "pgm_inflated"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-load", type=float, default=15.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    load = _cpu_load()
    if not args.force and not load <= args.max_load:
        print(f"machine busy (CPU load {load} %), no timings taken; use --force to override")
        return 2

    from pyorps.collector.voltage import check_tree, voltage_model_from_yaml
    from pyorps.collector.voltage_batch import DesignBatch, check_batch
    from tests.test_collector.test_voltage import YAML
    from tests.test_collector.test_voltage_batch import _random_trees

    try:
        import power_grid_model as pgm
        pgm_version = pgm.__version__
    except ImportError:
        pgm_version = None
    vm = voltage_model_from_yaml(YAML, u_kv=20, cable_mm2=[150, 240, 400, 630], n_turbines=7)
    rng = random.Random(9)
    bmax = max(args.sizes)
    trees = _random_trees(rng, bmax, 7, 4, vm, zero_p=0.05)
    batch_all = DesignBatch.from_trees(trees, n=7)

    kw = {"pgm_islands": dict(threads=args.threads), "pgm_inflated": dict(threads=args.threads)}
    results: dict = {}
    for name in args.backends:
        if name.startswith("pgm") and pgm_version is None:
            continue
        results[name] = {}
        if name != "scalar":                                    # compile / import once, outside the timing
            check_batch(batch_all.subset(np.arange(2)), vm, backend=name, **kw.get(name, {}))
        for B in args.sizes:
            if name == "scalar" and B > 1000:
                continue
            best = float("inf")
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                if name == "scalar":
                    for tr in trees[:B]:
                        check_tree(tr, vm)
                else:
                    check_batch(batch_all.subset(np.arange(B)), vm, backend=name, **kw.get(name, {}))
                best = min(best, time.perf_counter() - t0)
            results[name][str(B)] = best / B * 1e6
            print(f"{name:13s} B={B:6d}  {best / B * 1e6:9.1f} us per design", flush=True)

    out = args.out or HERE / "results" / f"backends_{dt.date.today().isoformat()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"unit": f"microseconds per 7-turbine design, best of {args.repeats}", "pgm": pgm_version,
                               "threads": args.threads, "cpu_load_before_percent": load, "results": results}, indent=1),
                   encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

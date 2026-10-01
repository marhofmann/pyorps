"""
Benchmark: fused delta-stepping kernel (bucket fusion + adaptive window +
VGC cap) vs the unmodified persistent delta-stepping baseline and Dijkstra.

Acceptance protocol: the fused kernel is only integrated if it is proven
faster than delta_stepping_2d_persistent (the current production kernel
behind algorithm="delta-stepping") on the benchmark matrix.

Every timed run is also checked for cost-optimality against Dijkstra.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_fused_kernel.py
    ... --sizes 500 1000 2000 --reps 5 --threads 8 [--big]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

from pyorps.utils._delta_stepping import delta_stepping_2d_persistent
from pyorps.utils._delta_stepping_fused import delta_stepping_2d_fused
from pyorps.utils._dijkstra import dijkstra_2d_cython
from pyorps.utils._raster_context import path_cost

STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)

DELTA = 100.0
MARGIN = 1.1  # mirrors the CythonAPI production default


def make_raster(pattern, n, rng):
    if pattern == "random":
        return rng.integers(1, 200, (n, n), dtype=np.uint16)
    if pattern == "heavy_tail":
        r = np.clip(rng.lognormal(2.0, 1.5, (n, n)), 1, 5000)
        return r.astype(np.uint16)
    if pattern == "obstacles":
        r = rng.integers(1, 60, (n, n), dtype=np.uint16)
        r[rng.random((n, n)) < 0.2] = 65535
        r[0, 0] = 1
        r[-1, -1] = 1
        return r
    if pattern == "corridors":
        # Expensive terrain crossed by cheap winding channels -- the
        # narrow-frontier / many-buckets regime of real cost surfaces,
        # which is the regime the published levers target.
        r = np.full((n, n), 500, dtype=np.uint16)
        for start_frac in (0.0, 0.25, 0.5, 0.75):
            row = int(n * start_frac)
            col = 0
            while col < n:
                r[max(0, row - 1):min(n, row + 2), col] = 2
                row += int(rng.integers(-2, 3))
                row = min(max(row, 0), n - 1)
                col += 1
        # guaranteed diagonal channel connecting the corners
        for i in range(n):
            r[max(0, i - 1):min(n, i + 2), i] = 2
        return r
    raise ValueError(pattern)


def run_config(fn, kwargs, raster, src, tgt, reps, ref_cost):
    times = []
    cost = None
    for i in range(reps + 1):  # first run = warmup
        t0 = time.perf_counter()
        path = fn(raster, STEPS_8, src, tgt, **kwargs)
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt)
        if len(path) == 0:
            return None, None, "NO PATH"
        cost = path_cost(np.asarray(path, dtype=np.uint64), raster,
                         raster.shape[1])
    rel = abs(cost - ref_cost) / ref_cost if ref_cost else 0.0
    status = "ok" if rel < 1e-4 else f"COST MISMATCH ({rel:.2e})"
    return statistics.median(times), min(times), status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = min(8, logical cores - 4)")
    ap.add_argument("--big", action="store_true", help="add 3000^2")
    ap.add_argument("--patterns", nargs="+",
                    default=["random", "heavy_tail"])
    args = ap.parse_args()

    threads = args.threads
    if threads <= 0:
        threads = max(2, min(8, (psutil.cpu_count(logical=True) or 8) - 4))
    sizes = list(args.sizes) + ([3000] if args.big else [])

    fused_common = dict(delta=DELTA, num_threads=threads, margin=MARGIN)
    configs = [
        ("dijkstra", "dij", None, None),
        ("delta-persistent (baseline)", "base", delta_stepping_2d_persistent,
         dict(delta=DELTA, num_threads=threads, margin=MARGIN)),
        ("fused: no levers", "f-none", delta_stepping_2d_fused,
         dict(fused_common, fusion_cap=0, window_init=1,
              adaptive_window=False)),
        ("fused: fusion only", "f-fus", delta_stepping_2d_fused,
         dict(fused_common, fusion_cap=1024, window_init=1,
              adaptive_window=False)),
        ("fused: window only", "f-win", delta_stepping_2d_fused,
         dict(fused_common, fusion_cap=0, window_init=8, window_max=256,
              adaptive_window=True)),
        ("fused: fusion+window", "f-full", delta_stepping_2d_fused,
         dict(fused_common, fusion_cap=1024, window_init=8, window_max=256,
              adaptive_window=True)),
    ]

    rng = np.random.default_rng(20260805)
    results = []
    print(f"threads={threads}, delta={DELTA}, margin={MARGIN}, "
          f"reps={args.reps} (median), corner-to-corner pairs\n")

    for pattern in args.patterns:
        for n in sizes:
            raster = make_raster(pattern, n, rng)
            src, tgt = np.uint64(0), np.uint64(n * n - 1)

            # Dijkstra: reference cost + timing
            t0 = time.perf_counter()
            p_ref = dijkstra_2d_cython(raster, STEPS_8, np.uint32(src),
                                       np.uint32(tgt))
            dij_warm = time.perf_counter() - t0
            if len(p_ref) == 0:
                print(f"[{pattern} {n}x{n}] unreachable — skipped")
                continue
            ref_cost = path_cost(np.asarray(p_ref, dtype=np.uint64), raster, n)
            dij_times = []
            for _ in range(max(2, args.reps - 2)):
                t0 = time.perf_counter()
                dijkstra_2d_cython(raster, STEPS_8, np.uint32(src),
                                   np.uint32(tgt))
                dij_times.append(time.perf_counter() - t0)
            dij_med = statistics.median(dij_times)

            print(f"[{pattern} {n}x{n}]  dijkstra: {dij_med*1e3:8.1f} ms  "
                  f"(cost {ref_cost:.0f})")
            base_med = None
            for name, key, fn, kwargs in configs:
                if key == "dij":
                    results.append(dict(pattern=pattern, n=n, algo=key,
                                        median_s=dij_med, status="ref"))
                    continue
                med, best, status = run_config(fn, kwargs, raster, src, tgt,
                                               args.reps, ref_cost)
                if med is None:
                    print(f"    {name:<28}  FAILED: {status}")
                    results.append(dict(pattern=pattern, n=n, algo=key,
                                        median_s=None, status=status))
                    continue
                if key == "base":
                    base_med = med
                spd_base = (base_med / med) if (base_med and key != "base") \
                    else 1.0
                spd_dij = dij_med / med
                flag = "" if status == "ok" else f"  <<< {status}"
                print(f"    {name:<28} {med*1e3:8.1f} ms  "
                      f"vs-base x{spd_base:5.2f}  vs-dijkstra x{spd_dij:5.2f}"
                      f"{flag}")
                results.append(dict(pattern=pattern, n=n, algo=key,
                                    median_s=med, best_s=best, status=status,
                                    speedup_vs_base=spd_base,
                                    speedup_vs_dijkstra=spd_dij))
            print()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"fused_kernel_{stamp}.json").write_text(
        json.dumps(dict(threads=threads, delta=DELTA, margin=MARGIN,
                        reps=args.reps, results=results), indent=1))

    # Verdict
    print("=" * 64)
    wins = losses = 0
    for r in results:
        if r["algo"] == "f-full" and r.get("median_s"):
            if r["speedup_vs_base"] > 1.05:
                wins += 1
            elif r["speedup_vs_base"] < 0.95:
                losses += 1
    print(f"VERDICT (fused fusion+window vs baseline, >5% margin): "
          f"{wins} wins, {losses} losses over scenarios")


if __name__ == "__main__":
    main()

"""
Benchmark: GPU V4 persistent kernel with the window (super-bucket) and
tail-chase fusion levers vs the classic configuration (window=1, fuse=0).

Rationale: on high-diameter rasters the GPU's bottleneck is under-occupancy
(per-bucket frontiers of ~10^2-10^3 cells cannot feed ~10^4 CUDA threads),
plus ~3 grid barriers per light iteration. The window multiplies per-phase
parallelism by W; tail-chase fusion shortens in-window chains. Both are
distance-exact (verified: bit-identical dist arrays vs window=1).

Full-SSSP runs (no targets) -- the pure kernel-throughput measure.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_gpu_window.py
    ... [--sizes 1000 2000 3000] [--reps 3]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pyorps.utils.sssp_gpu import sssp_raster_gpu_v4

STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)


def make_raster(pattern, n, rng):
    if pattern == "random":
        return rng.integers(1, 200, (n, n), dtype=np.uint16)
    if pattern == "heavy_tail":
        return np.clip(rng.lognormal(2.0, 1.5, (n, n)), 1, 5000).astype(
            np.uint16)
    raise ValueError(pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 3000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--patterns", nargs="+", default=["random", "heavy_tail"])
    args = ap.parse_args()

    configs = [(1, 0)] + [(w, f) for w in (2, 4, 8, 16, 32) for f in (0,)] \
        + [(1, 4), (4, 4), (8, 4), (16, 4)]

    rng = np.random.default_rng(20260805)
    results = []
    print(f"GPU V4 window/fusion sweep, full SSSP, reps={args.reps} (median)\n")

    for pattern in args.patterns:
        for n in args.sizes:
            raster = make_raster(pattern, n, rng)
            ref = None
            base_med = None
            print(f"[{pattern} {n}x{n}]")
            for w, f in configs:
                times = []
                dist = None
                for i in range(args.reps + 1):
                    t0 = time.perf_counter()
                    dist = sssp_raster_gpu_v4(raster, STEPS_8, 0,
                                              delta="auto", window=w,
                                              fuse_depth=f)
                    dt = time.perf_counter() - t0
                    if i > 0:
                        times.append(dt)
                med = statistics.median(times)
                if ref is None:
                    ref = dist
                    base_med = med
                    status = "ref"
                else:
                    finite = np.isfinite(ref) & (ref < 1e29)
                    same = (np.array_equal(
                        finite, np.isfinite(dist) & (dist < 1e29)) and
                        (not finite.any() or float(np.max(np.abs(
                            dist[finite] - ref[finite]))) == 0.0))
                    status = "exact" if same else "MISMATCH"
                spd = base_med / med if base_med else 1.0
                print(f"    w={w:>2} fuse={f}: {med*1e3:8.1f} ms  "
                      f"x{spd:5.2f} vs classic  [{status}]")
                results.append(dict(pattern=pattern, n=n, window=w, fuse=f,
                                    median_s=med, speedup=spd, status=status))
            print()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"gpu_window_{stamp}.json").write_text(
        json.dumps(dict(reps=args.reps, results=results), indent=1))

    best = {}
    for r in results:
        key = (r["pattern"], r["n"])
        if key not in best or r["median_s"] < best[key]["median_s"]:
            best[key] = r
    print("=" * 60)
    for key, r in best.items():
        print(f"BEST {key[0]} {key[1]}x{key[1]}: w={r['window']} "
              f"fuse={r['fuse']} -> x{r['speedup']:.2f} vs classic "
              f"[{r['status']}]")


if __name__ == "__main__":
    main()

"""
V4 window x launch-geometry joint sweep (open-levers plan section 2 precursor).

Question this answers: how far does *synchronous* occupancy tuning go?
The window lever proved occupancy is the V4 bottleneck; before building the
async V5 kernel, map window (4..32) jointly with threads_per_block and
blocks-per-SM. If a tuned synchronous configuration already saturates the
SMs at 3000^2, V5's headroom estimate shrinks accordingly.

Baseline = production defaults (window=4, tpb=256, 2 blocks/SM).
All configs are checked bit-exact against the baseline dist array.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_gpu_launch_sweep.py
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

BASELINE = dict(window=4, threads_per_block=256, blocks_per_sm=2)


def make_raster(pattern, n, rng):
    if pattern == "random":
        return rng.integers(1, 200, (n, n), dtype=np.uint16)
    if pattern == "heavy_tail":
        return np.clip(rng.lognormal(2.0, 1.5, (n, n)), 1, 5000).astype(
            np.uint16)
    raise ValueError(pattern)


def run_config(raster, reps, **cfg):
    times = []
    dist = None
    for i in range(reps + 1):
        t0 = time.perf_counter()
        dist = sssp_raster_gpu_v4(raster, STEPS_8, 0, delta="auto", **cfg)
        dt = time.perf_counter() - t0
        if i > 0:  # first run warms the kernel cache
            times.append(dt)
    return statistics.median(times), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 3000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--patterns", nargs="+", default=["random", "heavy_tail"])
    args = ap.parse_args()

    configs = [
        dict(window=w, threads_per_block=tpb, blocks_per_sm=bpsm)
        for w in (4, 8, 16, 32)
        for tpb in (128, 256, 512)
        for bpsm in (1, 2)
    ]

    rng = np.random.default_rng(20260806)
    results = []
    print(f"V4 window x launch-geometry sweep, full SSSP, "
          f"reps={args.reps} (median)\n")

    for pattern in args.patterns:
        for n in args.sizes:
            raster = make_raster(pattern, n, rng)
            base_med, ref = run_config(raster, args.reps, **BASELINE)
            print(f"[{pattern} {n}x{n}]  baseline w=4 tpb=256 bpsm=2: "
                  f"{base_med*1e3:8.1f} ms")
            finite = np.isfinite(ref) & (ref < 1e29)
            for cfg in configs:
                if cfg == BASELINE:
                    continue
                try:
                    med, dist = run_config(raster, args.reps, **cfg)
                except Exception as exc:  # cooperative launch too large etc.
                    print(f"    w={cfg['window']:>2} tpb={cfg['threads_per_block']:>3} "
                          f"bpsm={cfg['blocks_per_sm']}: LAUNCH FAIL "
                          f"({type(exc).__name__})")
                    results.append(dict(pattern=pattern, n=n, **cfg,
                                        status="launch_fail"))
                    continue
                same = (np.array_equal(
                    finite, np.isfinite(dist) & (dist < 1e29)) and
                    (not finite.any() or float(np.max(np.abs(
                        dist[finite] - ref[finite]))) == 0.0))
                status = "exact" if same else "MISMATCH"
                spd = base_med / med
                print(f"    w={cfg['window']:>2} tpb={cfg['threads_per_block']:>3} "
                      f"bpsm={cfg['blocks_per_sm']}: {med*1e3:8.1f} ms  "
                      f"x{spd:5.2f} vs baseline  [{status}]")
                results.append(dict(pattern=pattern, n=n, **cfg,
                                    median_s=med, speedup=spd, status=status))
            print()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"gpu_launch_sweep_{stamp}.json").write_text(
        json.dumps(dict(reps=args.reps, baseline=BASELINE, results=results),
                   indent=1))

    best = {}
    for r in results:
        if r.get("status") != "exact":
            continue
        key = (r["pattern"], r["n"])
        if key not in best or r["median_s"] < best[key]["median_s"]:
            best[key] = r
    print("=" * 64)
    for key, r in best.items():
        print(f"BEST {key[0]} {key[1]}x{key[1]}: w={r['window']} "
              f"tpb={r['threads_per_block']} bpsm={r['blocks_per_sm']} "
              f"-> x{r['speedup']:.2f} vs production baseline")


if __name__ == "__main__":
    main()

"""
Benchmark: V5 asynchronous bucket-queue kernel vs the V4 persistent kernel
(production defaults, window=4) — the open-levers plan section 2 acceptance
run. Full SSSP, bit-exactness asserted on every measurement.

Acceptance gate: >1.15x median speedup over V4-with-window-4 across the
matrix at 1000^2-3000^2, bit-exact dist arrays.

The sweep is parameterised over the neighbourhood (plan item 0.2). Every
GPU number recorded before 2026-08-07 is an R1 number because this file
hard-coded the 8-step set, while the production accuracy default is R2
and users run R3 — R2 doubles and R3 quadruples the per-pop relax work.
R1 remains the default so the recorded result files stay comparable, and
the neighbourhood is an explicit column in the emitted JSON.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_gpu_v5.py
    ... [--sizes 1000 2000 3000] [--reps 3] [--chunks 64 128 256]
    ... [--neighborhoods r1 r2 r3] [--no-sampling]
"""

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_env import (                                          # noqa: E402
    environment_snapshot, print_environment, print_gate, gpu_gate,
    timed_reps, write_results)

from pyorps.utils.neighborhood import get_neighborhood_steps      # noqa: E402
from pyorps.utils.sssp_gpu import (                               # noqa: E402
    sssp_raster_gpu_v4, sssp_raster_gpu_v5)

#: R1 as this file has always spelled it — identical *set* to
#: get_neighborhood_steps("r1"), kept literal so r1 rows stay comparable
#: with the result files recorded before the neighbourhood sweep existed.
STEPS_8 = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)

#: every timed region of this run, with its cold run and clock/temperature
#: envelope (plan item 0.12); the rows below quote the steady-state median.
TIMINGS = []


def resolve_steps(name):
    return (STEPS_8 if str(name).lower() in ("r1", "1")
            else get_neighborhood_steps(name, directed=True))


def make_raster(pattern, n, rng):
    if pattern == "random":
        return rng.integers(1, 200, (n, n), dtype=np.uint16)
    if pattern == "heavy_tail":
        return np.clip(rng.lognormal(2.0, 1.5, (n, n)), 1, 5000).astype(
            np.uint16)
    raise ValueError(pattern)


def bench(fn, raster, steps, reps, label, sample, **kw):
    """Median of `reps` timed runs after one discarded warm-up — the
    historical semantics; the discarded run and the thermal envelope are
    kept in TIMINGS instead of being thrown away."""
    stats, d = timed_reps(
        lambda: fn(raster, steps, 0, delta="auto", **kw),
        reps=max(1, int(reps)) + 1, sample=sample, label=label)
    TIMINGS.append(stats)
    return stats["steady_median_s"], d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 3000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--patterns", nargs="+", default=["random", "heavy_tail"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--neighborhoods", nargs="+", default=["r1"],
                    help="r1 (default, comparable with older results), "
                         "r2 (production accuracy default), r3")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--no-sampling", action="store_true",
                    help="skip the nvidia-smi clock/temperature sampling")
    args = ap.parse_args()

    env = environment_snapshot()
    print_environment(env)
    gate = gpu_gate(allow_shared=args.allow_shared_gpu)
    print_gate(gate)
    if not gate["ok"]:
        raise SystemExit("GPU gate closed — nothing to measure")

    rng = np.random.default_rng(20260806)
    results = []
    default_speedups = {}
    sample = not args.no_sampling
    print(f"\nV5 async vs V4 (window=4 production default), "
          f"reps={args.reps} (median)\n")
    for nb in args.neighborhoods:
        steps = resolve_steps(nb)
        n_steps = int(steps.shape[0])
        default_speedups[nb] = []
        for pattern in args.patterns:
            for n in args.sizes:
                raster = make_raster(pattern, n, rng)
                tag = f"{pattern} {n}x{n} {nb}"
                t4, ref = bench(sssp_raster_gpu_v4, raster, steps, args.reps,
                                f"v4 {tag}", sample)
                f = np.isfinite(ref) & (ref < 1e29)
                print(f"[{tag} ({n_steps} steps)]  V4: {t4*1e3:8.1f} ms")
                for chunk in args.chunks:
                    t5, d5 = bench(sssp_raster_gpu_v5, raster, steps,
                                   args.reps, f"v5 chunk={chunk} {tag}",
                                   sample, chunk=chunk)
                    exact = (np.array_equal(
                        f, np.isfinite(d5) & (d5 < 1e29)) and
                        (not f.any() or float(np.max(np.abs(
                            d5[f] - ref[f]))) == 0.0))
                    spd = t4 / t5
                    status = "exact" if exact else "MISMATCH"
                    print(f"    V5 chunk={chunk:>4}: {t5*1e3:8.1f} ms  "
                          f"x{spd:5.2f}  [{status}]")
                    results.append(dict(pattern=pattern, n=n, chunk=chunk,
                                        neighborhood=nb, n_steps=n_steps,
                                        v4_s=t4, v5_s=t5, speedup=spd,
                                        status=status))
                    if chunk == 256 and exact:
                        default_speedups[nb].append(spd)
                print()

    write_results("gpu_v5", dict(
        reps=args.reps, neighborhoods=list(args.neighborhoods),
        environment=env, gpu_gate=gate, results=results, timings=TIMINGS))

    print("=" * 60)
    for nb, speedups in default_speedups.items():
        if not speedups:
            continue
        med = statistics.median(speedups)
        if nb == "r1":
            note = ("(acceptance gate: >1.15) -> "
                    + ("PASS" if med > 1.15 else "FAIL"))
        else:
            note = "(the acceptance gate is defined on r1 only)"
        print(f"{nb}: MEDIAN speedup at default chunk=256: x{med:.2f}  "
              f"{note}")


if __name__ == "__main__":
    main()

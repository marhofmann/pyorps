"""Item 1.11 acceptance: what does shrinking the V5 ring arena actually cost?

Phase 1b cut the arena from ``3n/32`` entries per ring to ``arena_factor*n/32``
(12 -> 4 B/cell at factor 1.0, V5 total 22 -> 14 B/cell) and shipped it with no
wall-clock number behind it. The review was right to flag that: real frontiers
are O(perimeter), but the hot path pushes on every strict improvement, so pushes
are NOT bounded by one per pixel. A too-small arena spills more, and a full ring
set drops - which costs an O(n) rewind rescan.

This is an A/B across factors on identical rasters, so it stays valid under CPU
contention: every arm pays the same host-side overhead. Absolute numbers do not
transfer to a quiet machine; the ratios do.

    python benchmarks/benchmark_arena_factor.py --sizes 1000 2000
    python benchmarks/benchmark_arena_factor.py --dry-run     # no GPU

Verdict rule: keep 1.0 only if it costs < 5% against 3.0 (the old sizing) on
every pattern. Otherwise ship 2.0, which still frees 4 B/cell.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bench_env as env  # noqa: E402

STEPS_8 = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],
                    [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)

#: 1.0 is what Phase 1b shipped; 3.0 reproduces the pre-1.11 sizing exactly.
FACTORS = (1.0, 2.0, 3.0)
TOLERANCE_PCT = 5.0


def make_raster(pattern, n, seed=20260810):
    rng = np.random.default_rng(seed)
    if pattern == "random":
        return rng.integers(1, 900, (n, n)).astype(np.uint16)
    if pattern == "heavy_tail":
        base = rng.integers(1, 60, (n, n)).astype(np.uint16)
        spikes = rng.random((n, n)) < 0.02
        base[spikes] = rng.integers(3000, 9000, int(spikes.sum()))
        return base
    if pattern == "plateau":
        # The case that actually stresses the arena: a wide 0-cost region puts
        # a huge number of vertices in one delta-bucket.
        base = rng.integers(1, 900, (n, n)).astype(np.uint16)
        side = n // 2
        base[:side, :side] = 0
        return base
    raise ValueError(pattern)


def run(raster, factor, reps):
    from pyorps.utils.sssp_gpu import GpuSsspSession

    times, field = [], None
    for _ in range(reps + 1):                       # +1 warm-up, discarded
        t0 = time.perf_counter()
        with GpuSsspSession(raster, STEPS_8, arena_factor=factor) as s:
            out = s.solve(0)
        times.append(time.perf_counter() - t0)
        field = np.asarray(out[0] if isinstance(out, tuple) else out)
    return sorted(times[1:])[len(times[1:]) // 2], field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--patterns", nargs="+",
                    default=["random", "heavy_tail", "plateau"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the plumbing without touching the GPU")
    args = ap.parse_args()

    snap = env.environment_snapshot(include_gpu=not args.dry_run)
    env.print_environment(snap)
    if args.dry_run:
        print("dry run: no GPU work, no results file")
        return 0

    gate = env.gpu_gate()
    env.print_gate(gate)
    if not gate["ok"] and not args.allow_shared_gpu:
        print("refusing to run: pass --allow-shared-gpu to override")
        return 2

    from pyorps.utils.sssp_gpu import _v5_arena_cap, _V5_N_BUCKETS

    rows, verdict_fail = [], []
    print(f"\n{'pattern':<12}{'n':>6}{'factor':>8}{'B/cell':>8}"
          f"{'median s':>10}{'vs 3.0':>9}  field")
    for pattern in args.patterns:
        for n in args.sizes:
            raster = make_raster(pattern, n)
            base_t, base_field = None, None
            for factor in sorted(FACTORS, reverse=True):   # 3.0 first = baseline
                med, field = run(raster, factor, args.reps)
                if base_t is None:
                    base_t, base_field = med, field
                ratio = med / base_t
                same = bool(np.array_equal(field, base_field))
                arena_b = 4 * _V5_N_BUCKETS * _v5_arena_cap(n * n, factor)
                per_cell = arena_b / (n * n)
                rows.append({"pattern": pattern, "n": n, "factor": factor,
                             "median_s": med, "ratio_vs_3": ratio,
                             "arena_bytes_per_cell": per_cell,
                             "field_matches_baseline": same})
                if not same:
                    verdict_fail.append(f"{pattern} {n}^2 factor {factor}: "
                                        f"field differs from the 3.0 baseline")
                if factor == 1.0 and (ratio - 1.0) * 100 > TOLERANCE_PCT:
                    verdict_fail.append(
                        f"{pattern} {n}^2: factor 1.0 costs "
                        f"{(ratio - 1) * 100:.1f}% vs 3.0")
                print(f"{pattern:<12}{n:>6}{factor:>8.1f}{per_cell:>8.1f}"
                      f"{med:>10.3f}{ratio:>8.2f}x  "
                      f"{'same' if same else 'DIFFERS'}")
            env.free_gpu_pool()

    print()
    if verdict_fail:
        print("VERDICT: do NOT keep arena_factor=1.0 as the default")
        for line in verdict_fail:
            print(f"  - {line}")
    else:
        print(f"VERDICT: arena_factor=1.0 is safe - within {TOLERANCE_PCT}% of "
              f"the old 3.0 sizing on every case, field identical throughout")

    env.write_results("arena_factor", {
        "environment": snap,
        "cpu_contention_end": env.cpu_contention(),
        "tolerance_pct": TOLERANCE_PCT,
        "rows": rows,
        "verdict_failures": verdict_fail,
    })
    return 1 if verdict_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

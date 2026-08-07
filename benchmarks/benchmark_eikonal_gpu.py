"""
Performance benchmark: eikonal block-FIM vs Cython Dijkstra (CPU
champion) and the V5 async GPU kernel (production discrete default) —
eikonal plan section 7.

Corner-to-corner single pair per raster. Each solver does its production
thing: Cython Dijkstra single-target, V5 with targeted early exit, FIM a
full-field solve (reported with and without path tracing — the field is
reusable for every further target). Costs are NOT compared here (FIM is
supposed to differ — see benchmark_eikonal_accuracy.py); this file only
measures wall-clock, iterations and throughput.

Heavy-tail is FIM's expected worst case — measuring it honestly matters
more than winning it.

The discrete solvers are swept over the neighbourhood (plan item 0.2):
this file hard-coded the 8-step set, so every FIM-vs-V5 and FIM-vs-cython
ratio recorded before 2026-08-07 is an R1 ratio while the production
accuracy default is R2. R1 stays the default so the recorded result files
remain comparable. **raster_fim has no neighbourhood** — it discretizes a
continuous PDE rather than a lattice graph — so its solve is measured
once per (pattern, size) and reused across the discrete rows, which carry
``fim_neighborhood: null`` and the neighbourhood it was measured under.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_eikonal_gpu.py
    ... [--sizes 500 1000 2000 3000] [--reps 3] [--tuning]
    ... [--neighborhoods r1 r2 r3] [--no-sampling]

GPU etiquette: modest reps, strictly sequential runs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_env import (                                          # noqa: E402
    environment_snapshot, gpu_gate, print_environment, print_gate,
    timed_reps, write_results)

from pyorps.utils._dijkstra import dijkstra_2d_cython             # noqa: E402
from pyorps.utils.eikonal_gpu import (                            # noqa: E402
    eikonal_raster_gpu, trace_paths_gpu, polyline_to_cells,
    FINITE_LIMIT)
from pyorps.utils.neighborhood import get_neighborhood_steps      # noqa: E402
from pyorps.utils.sssp_gpu import sssp_raster_gpu                 # noqa: E402

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

#: nvidia-smi sampling around timed regions; cleared by --no-sampling.
SAMPLE = True


def resolve_steps(name):
    return (STEPS_8 if str(name).lower() in ("r1", "1")
            else get_neighborhood_steps(name, directed=True))


def blur(field, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma, mode="reflect")
    except ImportError:
        n = field.shape[0]
        k = np.exp(-0.5 * (np.fft.fftfreq(n) * n / max(sigma, 1e-9)) ** 2)
        return np.real(np.fft.ifft2(np.fft.fft2(field) * np.outer(k, k)))


def make_raster(pattern, n, rng):
    if pattern == "uniform":
        return np.full((n, n), 10, dtype=np.uint16)
    if pattern == "random":
        return rng.integers(1, 200, (n, n), dtype=np.uint16)
    if pattern == "heavy_tail":
        return np.clip(rng.lognormal(2.0, 1.5, (n, n)), 1,
                       5000).astype(np.uint16)
    if pattern == "smooth":
        base = blur(rng.normal(0, 1, (n, n)), 16)
        base = (base - base.min()) / (base.max() - base.min() + 1e-12)
        return (1 + base * 199).astype(np.uint16)
    raise ValueError(pattern)


def median_time(fn, reps, label=""):
    """Median of `reps` timed runs after one discarded warm-up — the
    historical semantics; the discarded run and the thermal envelope are
    kept in TIMINGS instead of being thrown away (plan item 0.12)."""
    stats, result = timed_reps(fn, reps=max(1, int(reps)) + 1, sample=SAMPLE,
                               label=label)
    TIMINGS.append(stats)
    return stats["steady_median_s"], result


def run_fim(raster, src, tgt, n, reps, tag):
    """Solve + trace once per (pattern, size). The eikonal backend has no
    neighbourhood, so re-measuring it per neighbourhood would fabricate a
    dependency it does not have."""
    it_holder = {}

    def fim_solve():
        t, its, d_t = eikonal_raster_gpu(raster, src,
                                         return_iterations=True,
                                         return_device=True)
        it_holder["iters"] = its
        it_holder["d_t"] = d_t
        return t

    t_fim, t_field = median_time(fim_solve, reps, f"fim solve {tag}")

    def fim_trace():
        polys = trace_paths_gpu(t_field, src, [tgt],
                                t_device=it_holder["d_t"])
        if polys[0] is None:
            return []
        return polyline_to_cells(polys[0][::-1], n, n,
                                 forbidden_mask=t_field >= FINITE_LIMIT)

    t_trace, _ = median_time(fim_trace, max(1, reps - 1), f"fim trace {tag}")
    return dict(fim_solve_s=t_fim, fim_trace_s=t_trace,
                fim_total_s=t_fim + t_trace,
                fim_outer_passes=it_holder["iters"],
                fim_mcells_per_s=n * n / t_fim / 1e6)


def bench_size(pattern, n, rng, reps, neighborhoods, results):
    raster = make_raster(pattern, n, rng)
    src, tgt = 0, n * n - 1

    fim = None
    fim_measured_at = None
    for nb in neighborhoods:
        steps = resolve_steps(nb)
        tag = f"{pattern} {n}x{n} {nb}"

        t_cy, path = median_time(
            lambda: dijkstra_2d_cython(raster, steps, np.uint32(src),
                                       np.uint32(tgt)),
            max(1, reps - 1), f"cython {tag}")
        if len(path) == 0:
            print(f"[{tag}] unreachable — skipped")
            continue

        t_v5, _ = median_time(
            lambda: sssp_raster_gpu(raster, steps, src,
                                    target_indices=np.array(
                                        [tgt], dtype=np.int32)),
            reps, f"v5 {tag}")

        if fim is None:
            fim = run_fim(raster, src, tgt, n, reps, tag)
            fim_measured_at = nb

        row = dict(pattern=pattern, n=n,
                   neighborhood=nb, n_steps=int(steps.shape[0]),
                   cython_s=t_cy, v5_s=t_v5,
                   # raster_fim has no neighbourhood; the column stays
                   # null and the run it was measured in is named.
                   fim_neighborhood=None,
                   fim_measured_at_neighborhood=fim_measured_at,
                   speedup_vs_cython=t_cy / fim["fim_total_s"],
                   speedup_vs_v5=t_v5 / fim["fim_total_s"],
                   **fim)
        results.append(row)
        print(f"[{tag} ({row['n_steps']} steps)]  cython {t_cy*1e3:8.1f} ms "
              f"| V5 {t_v5*1e3:7.1f} ms | FIM {fim['fim_solve_s']*1e3:7.1f}"
              f"+{fim['fim_trace_s']*1e3:5.1f} ms "
              f"({fim['fim_outer_passes']:4d} passes, "
              f"{row['fim_mcells_per_s']:6.1f} Mc/s) | "
              f"x{row['speedup_vs_cython']:.2f} vs cython, "
              f"x{row['speedup_vs_v5']:.2f} vs V5")


def breakdown(sizes, reps):
    """Per-stage cost of a single-pair FIM route (performance-plan
    phase 0): slowness prep, transfers, kernel passes, trace,
    rasterize. Also runs the 4096^2 real-world window when its cached
    input exists (see benchmark_eikonal_realworld.py)."""
    import cupy as cp

    from pyorps.utils.eikonal_gpu import _slowness_field

    cases = []
    rng = np.random.default_rng(3)
    for n in sizes:
        raster = make_raster("random", n, rng)
        cases.append((f"random {n}x{n}", raster, 0, n * n - 1))
    real = Path(__file__).parent / "realworld_data" / "window_cost.npz"
    if real.exists():
        z = np.load(real)
        cost = z["cost"]
        nr, nc = cost.shape

        def snap(r, c):
            ok = np.argwhere(cost[r - 60:r + 61, c - 60:c + 61] != 65535)
            rr, cc = ok[np.argmin(((ok - 60) ** 2).sum(1))]
            return (r - 60 + int(rr)) * nc + (c - 60 + int(cc))

        cases.append((f"real {nr}x{nc}", cost,
                      snap(600, 1900), snap(3080, 3560)))

    rows = []
    print("\n=== stage breakdown (single pair, median of "
          f"{reps}) ===")
    for name, raster, src, tgt in cases:
        eikonal_raster_gpu(raster[:256, :256], 0)      # warm compile

        t_prep, slow = median_time(
            lambda: _slowness_field(raster, True), reps, f"prep {name}")
        t_up, d = median_time(lambda: cp.asarray(slow), reps, f"h2d {name}")
        t_down, _ = median_time(lambda: d.get(), reps, f"d2h {name}")

        holder = {}

        def solve():
            t, its, d_t = eikonal_raster_gpu(
                raster, src, target_index=tgt,
                return_iterations=True, return_device=True)
            holder.update(t=t, d_t=d_t, its=its)
            return t

        t_solve, _ = median_time(solve, reps, f"solve {name}")

        t_trace, poly = median_time(
            lambda: trace_paths_gpu(holder["t"], src, [tgt],
                                    t_device=holder["d_t"])[0], reps,
            f"trace {name}")
        t_cells, _ = median_time(
            lambda: polyline_to_cells(
                poly[::-1], *raster.shape,
                forbidden_mask=holder["t"] >= FINITE_LIMIT),
            reps, f"cells {name}")

        kernel = t_solve - t_prep - t_up - t_down
        row = dict(case=name, prep_ms=t_prep * 1e3, h2d_ms=t_up * 1e3,
                   d2h_ms=t_down * 1e3, solve_ms=t_solve * 1e3,
                   kernel_est_ms=kernel * 1e3, passes=holder["its"],
                   trace_ms=t_trace * 1e3, cells_ms=t_cells * 1e3,
                   total_ms=(t_solve + t_trace + t_cells) * 1e3)
        rows.append(row)
        print(f"[{name:>16}] prep {row['prep_ms']:5.1f} | H2D "
              f"{row['h2d_ms']:5.1f} | D2H {row['d2h_ms']:5.1f} | "
              f"kernel~{row['kernel_est_ms']:6.1f} ({row['passes']:3d}p)"
              f" | trace {row['trace_ms']:5.1f} | cells "
              f"{row['cells_ms']:4.1f} | total {row['total_ms']:6.1f} ms")
    return rows


def tuning_sweep(reps):
    """tile x n_inner sweep at 1000^2 random (once; defaults get fixed
    from this table)."""
    rng = np.random.default_rng(1)
    raster = make_raster("random", 1000, rng)
    print("\n=== tuning sweep (1000^2 random) ===")
    rows = []
    for tile in (8, 16, 32):
        for n_inner in (tile // 2, tile, 2 * tile):
            t, _ = median_time(
                lambda: eikonal_raster_gpu(raster, 0, tile=tile,
                                           n_inner=n_inner), reps,
                f"tuning tile={tile} n_inner={n_inner}")
            rows.append(dict(tile=tile, n_inner=n_inner, median_s=t))
            print(f"  tile={tile:>2} n_inner={n_inner:>2}: "
                  f"{t*1e3:7.1f} ms")
    best = min(rows, key=lambda r: r["median_s"])
    print(f"  best: tile={best['tile']} n_inner={best['n_inner']}")
    return rows


def main():
    global SAMPLE

    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 3000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--patterns", nargs="+",
                    default=["uniform", "random", "heavy_tail", "smooth"])
    ap.add_argument("--neighborhoods", nargs="+", default=["r1"],
                    help="r1 (default, comparable with older results), "
                         "r2 (production accuracy default), r3; raster_fim "
                         "has no neighbourhood and is measured once")
    ap.add_argument("--tuning", action="store_true",
                    help="also run the tile/n_inner sweep")
    ap.add_argument("--breakdown", action="store_true",
                    help="per-stage single-pair breakdown only")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--no-sampling", action="store_true",
                    help="skip the nvidia-smi clock/temperature sampling")
    args = ap.parse_args()
    SAMPLE = not args.no_sampling

    env = environment_snapshot()
    print_environment(env)
    gate = gpu_gate(allow_shared=args.allow_shared_gpu)
    print_gate(gate)
    if not gate["ok"]:
        raise SystemExit("GPU gate closed — nothing to measure")

    if args.breakdown:
        rows = breakdown([s for s in args.sizes if s >= 1000],
                         args.reps)
        write_results("eikonal_breakdown", dict(
            reps=args.reps, environment=env, gpu_gate=gate,
            results=rows, timings=TIMINGS))
        return

    rng = np.random.default_rng(20260806)
    results = []
    print(f"eikonal FIM vs cython dijkstra vs GPU V5, reps={args.reps} "
          f"(median), corner-to-corner\n")
    for pattern in args.patterns:
        for n in args.sizes:
            bench_size(pattern, n, rng, args.reps, args.neighborhoods,
                       results)
        print()

    tuning = tuning_sweep(args.reps) if args.tuning else None

    path = write_results("eikonal_perf", dict(
        reps=args.reps, neighborhoods=list(args.neighborhoods),
        environment=env, gpu_gate=gate, results=results, tuning=tuning,
        timings=TIMINGS))
    print(f"results -> {path}")


if __name__ == "__main__":
    main()

"""Tier A (anisotropic 3D-length) GPU eikonal benchmark.

Measures what the implementation plan lists as UNMEASURED:

  P1  anisotropic solve vs the isotropic solve at 3000^2      (gate: <= 3x)
  P2  end-to-end vs Cython Dijkstra R2 with the SAME DEM       (gate: >= 4x)
  P3  vs GPU V5 (raster_gpu) with the SAME DEM + LUTs          (report only)
  P4  q-build kernel                                           (gate: <= 5 ms)
  P5  grade-limit loop iteration counts                        (report)
  P6  multi-target amortisation break-even                     (report)
  plus inner-iteration counts (updates/cell) and the flat fast-path hit rate.

THE COMPARATOR MUST BE LIKE-FOR-LIKE. V5 and the Cython kernels both
support DEM + gradient LUTs, so they are run WITH the DEM here — comparing
Tier A against the isotropic V5 numbers would flatter it.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_eikonal_anisotropic.py
    ... --sizes 1000,2000,3000 --reps 3 --skip-cython
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from pyorps.core.objective import Objective, GradientOptions
from pyorps.utils.eikonal_gpu import (
    _device_metric,
    eikonal_raster_gpu,
    metric_from_dem,
    polyline_to_cells,
    trace_paths_gpu,
)

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    GPU = True
except Exception:                                    # pragma: no cover
    GPU = False

CELL = 10.0
STEPS_R2 = None          # filled in main()


# ---------------------------------------------------------------------------
# rasters and terrain
# ---------------------------------------------------------------------------

def make_raster(kind: str, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return np.full((n, n), 10, dtype=np.uint16)
    if kind == "random":
        return rng.integers(1, 100, (n, n)).astype(np.uint16)
    if kind == "heavy_tail":
        v = rng.pareto(1.5, (n, n)) * 10 + 1
        return np.clip(v, 1, 60000).astype(np.uint16)
    if kind == "smooth":
        k = max(1, n // 32)
        coarse = rng.integers(1, 100, (k + 2, k + 2)).astype(np.float64)
        rr = np.linspace(0, k, n)
        idx0 = np.clip(rr.astype(int), 0, k)
        out = coarse[np.ix_(idx0, idx0)]
        return np.clip(out, 1, 60000).astype(np.uint16)
    raise ValueError(kind)


def make_dem(n: int, peak_grade: float = 0.35, seed: int = 1) -> np.ndarray:
    """Smooth, realistic-ish terrain scaled to a chosen peak grade."""
    rng = np.random.default_rng(seed)
    z = np.zeros((n, n), dtype=np.float64)
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for _ in range(8):
        r0, c0 = rng.integers(0, n, 2)
        sig = rng.uniform(n / 16.0, n / 5.0)
        z += rng.normal() * 200.0 * np.exp(
            -(((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sig ** 2)))
    gz_r, gz_c = np.gradient(z, CELL)     # unclamped, to scale the terrain
    peak = float(np.hypot(gz_r, gz_c).max())
    if peak > 0:
        z *= peak_grade / peak
    return z.astype(np.float32)


def sync():
    if GPU:
        cp.cuda.Stream.null.synchronize()


def timeit(fn, reps=3):
    fn()                                # warm-up (kernel compile)
    sync()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------

def bench_metric_build(sizes, reps):
    print("\n=== P4: q-build kernel (gate <= 5 ms at 3000^2) ===")
    print(f"{'size':>8} {'cells':>12} {'ms':>9} {'GB/s':>8}")
    for n in sizes:
        dem = make_dem(n)
        ms = timeit(lambda: _device_metric(dem, CELL), reps)
        gb = (n * n * (4 + 8)) / 1e9
        print(f"{n:>8} {n*n:>12,} {ms:>9.2f} {gb/(ms/1e3):>8.1f}")


def bench_solve(sizes, kinds, reps):
    print("\n=== P1: anisotropic vs isotropic solve (gate <= 3x) ===")
    print(f"{'size':>6} {'class':>11} {'iso ms':>9} {'aniso ms':>9} "
          f"{'ratio':>7} {'iso upd/cell':>13} {'ani upd/cell':>13} "
          f"{'flat %':>7}")
    rows = []
    for n in sizes:
        dem = make_dem(n)
        q_r, q_c, _ = metric_from_dem(dem, CELL)
        flat_frac = float((np.hypot(q_r, q_c) < 1e-3).mean()) * 100.0
        for kind in kinds:
            raster = make_raster(kind, n)
            src = [0]
            iso_ms = timeit(lambda: eikonal_raster_gpu(
                raster, src, download=False), reps)
            ani_ms = timeit(lambda: eikonal_raster_gpu(
                raster, src, dem=dem, cell_size=CELL, download=False), reps)
            _t, st_i = eikonal_raster_gpu(raster, src, return_stats=True,
                                          download=False)
            _t, st_a = eikonal_raster_gpu(raster, src, dem=dem,
                                          cell_size=CELL,
                                          return_stats=True, download=False)
            print(f"{n:>6} {kind:>11} {iso_ms:>9.1f} {ani_ms:>9.1f} "
                  f"{ani_ms/iso_ms:>7.2f} {st_i['updates_per_cell']:>13.1f} "
                  f"{st_a['updates_per_cell']:>13.1f} {flat_frac:>7.2f}")
            rows.append((n, kind, iso_ms, ani_ms))
    return rows


def bench_vs_backends(sizes, kinds, reps, skip_cython):
    print("\n=== P2/P3: end-to-end, single pair, ALL BACKENDS WITH THE "
          "SAME DEM ===")
    from pyorps.graph.api.raster_gpu_api import RasterGPUAPI
    try:
        from pyorps.graph.api.cython_api import CythonAPI
    except Exception:
        CythonAPI = None
        skip_cython = True

    obj = Objective({"cost": 1.0}, GradientOptions())
    print(f"{'size':>6} {'class':>11} {'FIM ms':>9} {'V5+dem ms':>11} "
          f"{'Cy R2+dem ms':>13} {'vs V5':>7} {'vs Cy':>7} "
          f"{'FIM cost':>12} {'V5 cost':>12} {'ratio':>7}")
    for n in sizes:
        dem = make_dem(n)
        luts = obj.build_gradient_luts(STEPS_R2, CELL)
        for kind in kinds:
            raster = make_raster(kind, n)
            src = int(0)
            tgt = int((n - 1) * n + (n - 1))

            def run_fim():
                _t, d_t, (_tr, d_tr), q = eikonal_raster_gpu(
                    raster, [src], dem=dem, cell_size=CELL,
                    target_index=tgt, download=False,
                    return_device=True, return_trace_field=True,
                    return_metric=True)
                poly = trace_paths_gpu(None, [src], [tgt], t_device=d_tr,
                                       q_device=q, shape=(n, n))[0]
                cost = float(d_t[tgt])
                if poly is not None:
                    polyline_to_cells(poly, n, n)
                return cost

            fim_ms = timeit(run_fim, reps)
            fim_cost = run_fim()

            v5 = RasterGPUAPI(raster, STEPS_R2, dem_data=dem,
                              gradient_luts=luts)
            v5_ms = timeit(lambda: v5.shortest_path(src, tgt), reps)
            v5.shortest_path(src, tgt)
            v5_cost = float(getattr(v5, "last_costs", [np.nan])[0]) \
                if hasattr(v5, "last_costs") else float("nan")

            cy_ms = float("nan")
            if not skip_cython:
                cy = CythonAPI(raster, STEPS_R2, dem_data=dem,
                               gradient_luts=luts)
                cy_ms = timeit(lambda: cy.shortest_path(src, tgt), 1)

            print(f"{n:>6} {kind:>11} {fim_ms:>9.1f} {v5_ms:>11.1f} "
                  f"{cy_ms:>13.1f} {v5_ms/fim_ms:>7.2f} "
                  f"{cy_ms/fim_ms:>7.2f} {fim_cost:>12.1f} "
                  f"{v5_cost:>12.1f} "
                  f"{(fim_cost/v5_cost if v5_cost == v5_cost and v5_cost else float('nan')):>7.4f}")


def bench_multitarget(n, reps):
    print("\n=== P6: multi-target amortisation (one FIM field vs k V5 "
          "solves) ===")
    from pyorps.graph.api.raster_gpu_api import RasterGPUAPI
    obj = Objective({"cost": 1.0}, GradientOptions())
    luts = obj.build_gradient_luts(STEPS_R2, CELL)
    dem = make_dem(n)
    raster = make_raster("random", n)
    rng = np.random.default_rng(2)
    src = 0
    v5 = RasterGPUAPI(raster, STEPS_R2, dem_data=dem, gradient_luts=luts)
    print(f"{'k':>5} {'FIM ms':>9} {'V5 ms':>9} {'speedup':>9}")
    for k in (1, 2, 4, 8, 16):
        tgts = [int(x) for x in rng.integers(n * n // 2, n * n, k)]
        fim_ms = timeit(lambda: _fim_multi(raster, dem, src, tgts, n), reps)
        v5_ms = timeit(lambda: [v5.shortest_path(src, t) for t in tgts],
                       max(1, reps // 2))
        print(f"{k:>5} {fim_ms:>9.1f} {v5_ms:>9.1f} {v5_ms/fim_ms:>9.2f}")


def _fim_multi(raster, dem, src, tgts, n):
    t, d_t, (t_tr, d_tr), q = eikonal_raster_gpu(
        raster, [src], dem=dem, cell_size=CELL, return_device=True,
        return_trace_field=True, return_metric=True)
    polys = trace_paths_gpu(t_tr, [src], tgts, t_device=d_tr, q_device=q)
    for p in polys:
        if p is not None:
            polyline_to_cells(p, n, n)


def bench_grade_limit(n, reps):
    print("\n=== P5: grade-limit loop (lazy) ===")
    from pyorps.graph.api.raster_fim_api import RasterFIMAPI
    dem = make_dem(n, peak_grade=0.9)
    raster = make_raster("random", n)
    rng = np.random.default_rng(3)
    print(f"{'limit %':>8} {'mode':>7} {'pairs':>6} {'ok':>4} {'med it':>7} "
          f"{'p95 it':>7} {'ms/pair':>9} {'cost':>12}")
    pairs = [(int(rng.integers(0, n * n)), int(rng.integers(0, n * n)))
             for _ in range(8)]
    for limit in (15.0, 30.0):
        for mode in ("lazy", "eager"):
            obj = Objective({"cost": 1.0},
                            GradientOptions(max_gradient_pct=limit))
            api = RasterFIMAPI(raster, STEPS_R2, dem_data=dem,
                               cell_size=CELL,
                               gradient_luts=obj.build_gradient_luts(
                                   STEPS_R2, CELL),
                               grade_limit_mode=mode)
            iters, costs, ok = [], [], 0
            t0 = time.perf_counter()
            for s, t in pairs:
                try:
                    path = api.shortest_path(s, t)
                    assert api.grade_violations(path).size == 0
                    ok += 1
                    costs.append(api.last_field_costs[-1])
                except Exception:
                    pass
                iters.append(api.last_mask_iterations[-1]
                             if api.last_mask_iterations else 0)
            dt = (time.perf_counter() - t0) * 1e3 / len(pairs)
            print(f"{limit:>8.1f} {mode:>7} {len(pairs):>6} {ok:>4} "
                  f"{np.median(iters):>7.1f} "
                  f"{np.percentile(iters, 95):>7.1f} {dt:>9.1f} "
                  f"{(np.mean(costs) if costs else float('nan')):>12.1f}")


def bench_tuning(n, reps):
    print("\n=== tuning sweep (anisotropic optimum may differ from the "
          "isotropic B=16, n_inner=2B) ===")
    dem = make_dem(n)
    raster = make_raster("random", n)
    print(f"{'B':>4} {'n_inner':>8} {'ms':>9}")
    for b in (8, 16, 32):
        for ni in (b, 2 * b):
            try:
                ms = timeit(lambda: eikonal_raster_gpu(
                    raster, [0], dem=dem, cell_size=CELL, tile=b,
                    n_inner=ni, download=False), reps)
                print(f"{b:>4} {ni:>8} {ms:>9.1f}")
            except Exception as exc:
                print(f"{b:>4} {ni:>8}    failed: {type(exc).__name__}")


def main():
    global STEPS_R2
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,2000,3000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-cython", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if not GPU:
        print("NO CUDA GPU AVAILABLE — every number below would be "
              "invented. Aborting.")
        return
    sizes = [int(s) for s in args.sizes.split(",")]
    kinds = ["uniform", "random", "heavy_tail", "smooth"]

    from pyorps.utils.neighborhood import get_neighborhood_steps
    STEPS_R2 = np.asarray(get_neighborhood_steps(2))

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    print(f"GPU: {props['name'].decode()}  "
          f"{total_b/2**20:.0f} MiB total, {free_b/2**20:.0f} MiB free  "
          f"SMs={props['multiProcessorCount']}")
    print(f"steps R2: {len(STEPS_R2)} directions, cell_size {CELL} m")

    only = set(args.only.split(",")) if args.only else None

    def want(name):
        return only is None or name in only

    if want("metric"):
        bench_metric_build(sizes, args.reps)
    if want("solve"):
        bench_solve(sizes, kinds, args.reps)
    if want("backends"):
        bench_vs_backends(sizes, kinds, args.reps, args.skip_cython)
    if want("multitarget"):
        bench_multitarget(max(sizes), args.reps)
    if want("grade"):
        bench_grade_limit(min(600, min(sizes)), args.reps)
    if want("tuning"):
        bench_tuning(max(sizes), args.reps)


if __name__ == "__main__":
    main()

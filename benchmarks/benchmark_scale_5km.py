"""Scale benchmark above 4096^2 — performance-plan item 0.1.

No measurement above 4096^2 exists anywhere in this repository, so every
5000^2 / 8000^2 / 10000^2 figure in
``docs/superpowers/plans/2026-08-07-realworld-5km-performance-plan.md``
is an extrapolation. This harness produces the measurements.

Rasters
-------
The real 4096^2 ALKIS planning window cached by
``benchmark_eikonal_realworld.py`` (``realworld_data/window_cost.npz``,
1 m, 12.92 % forbidden, cost 125-750) is **mosaicked by mirroring** to the
target size — never stretched, never randomised:

    index map:  0, 1, ... n-1, n-1, ... 1, 0, 0, 1, ...   (period 2n)

Every mosaic cell is therefore an exact copy of a real cell, so the class
histogram, the cost contrast and the ~13 % forbidden fraction survive
(measured forbidden fraction: 4096^2 12.92 %, 5000^2 12.43 %, 8000^2
12.17 %, 10000^2 14.01 %), and reflecting instead of wrapping keeps the
seam continuous so no artificial cost cliff is introduced. The map is a
pure function of the sizes — the mosaic is bit-reproducible.

Scenario B (``--slope``) applies the isotropic DGM1 slope multiplier of
``benchmark_eikonal_realworld.py`` (m = min(1+3s^2, 3), s > 45 % =
forbidden) to the 4096^2 base and mosaics the *result*, which keeps the
peak memory of the gradient computation at base size. Gradients across
the three mosaic seams therefore differ from a mosaic-then-slope raster;
everywhere else the two are identical.

Pairs
-----
``fixed5km`` keeps the route length constant (5001 m) across sizes so
area scaling is isolated from route-length scaling; ``diag`` spans 5 % to
95 % of the window so it grows with it; ``west_east`` is a straight
horizontal span. Endpoints are snapped to the nearest passable cell.

What it settles
---------------
* the TDR threshold on this machine (registry TdrDelay is printed at
  start-up; killed runs are logged as data, never as a crash),
* the V5 22 B/cell device-memory model (pool high-water per run),
* the FIM pass floor at scale,
* whether the cython serial kernel is usable at all at 100 M cells.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_scale_5km.py
    ... [--max-size 10000] [--sizes 4096 5000 8000 10000]
    ... [--neighborhoods r1 r2] [--backends cython raster_gpu raster_fim]
    ... [--reps 2] [--slope] [--allow-shared-gpu] [--dry-run]

Sizes run smallest first and are capped by ``--max-size`` (default 5000):
a 10000^2 GPU solve may be killed by the watchdog and take the CUDA
context with it, so the large sizes are opt-in.

GPU etiquette: the machine is shared. The run refuses to touch the GPU
when nvidia-smi reports another tenant or an elevated idle temperature
and continues with CPU-only rows; ``--allow-shared-gpu`` is an explicit
operator override.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_env import (                                          # noqa: E402
    GpuFaultLog, cuda_device_info, environment_snapshot, free_gpu_pool,
    gpu_gate, gpu_pool_bytes, peak_rss_bytes, print_environment,
    print_gate, timed_reps, windows_tdr_events, write_results)

HERE = Path(__file__).resolve().parent
DATA = HERE / "realworld_data"
BASE_COST = DATA / "window_cost.npz"
BASE_DEM = DATA / "window_dem.npy"

SENTINEL = np.uint16(65535)

DEFAULT_SIZES = (4096, 5000, 8000, 10000)
BACKENDS = ("cython", "raster_gpu", "raster_fim")
PAIRS = ("fixed5km", "diag", "west_east")

#: Device bytes per cell used only to skip a run that cannot fit.
#: raster_gpu (V5): raster 2 + dist 4 + pred 4 + ring arena 12 = 22.
#: raster_fim: slowness 4 + T 4 + frozen/aux 8 = 16.
VRAM_PER_CELL = {"raster_gpu": 22, "raster_fim": 16}

#: 4096^2 base, R1 as previously hard-coded in the GPU harnesses.
R1_LITERAL = np.array([
    [0, 1], [0, -1], [1, 0], [-1, 0],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
], dtype=np.int8)


# ----------------------------------------------------------------------
# Raster construction
# ----------------------------------------------------------------------

def reflect_map(n_out, n_src):
    """Mirror index map with period ``2 * n_src`` (numpy 'symmetric')."""
    i = np.arange(n_out)
    period = 2 * n_src
    j = i % period
    return np.where(j < n_src, j, period - 1 - j).astype(np.intp)


def mosaic(base, n_rows, n_cols=None):
    """Mirror-tile ``base`` up to (n_rows, n_cols). A crop when smaller."""
    n_cols = n_rows if n_cols is None else n_cols
    ri = reflect_map(n_rows, base.shape[0])
    ci = reflect_map(n_cols, base.shape[1])
    return np.ascontiguousarray(base[np.ix_(ri, ci)])


def load_base(dry_run):
    if dry_run:
        rng = np.random.default_rng(20260807)
        base = rng.integers(125, 751, (192, 192)).astype(np.uint16)
        base[rng.random(base.shape) < 0.13] = SENTINEL
        return base, None
    if not BASE_COST.exists():
        raise SystemExit(
            f"missing {BASE_COST} — run benchmark_eikonal_realworld.py "
            f"once to cache the real planning window")
    base = np.load(BASE_COST)["cost"]
    dem = np.load(BASE_DEM) if BASE_DEM.exists() else None
    return base, dem


def slope_fraction(dem, cell=1.0):
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    gr, gc = np.gradient(d.astype(np.float64), cell)
    return np.hypot(gr, gc)


def apply_slope(cost, dem):
    """Isotropic slope multiplier, identical to the real-world harness."""
    s = slope_fraction(dem)
    mult = np.minimum(1.0 + 3.0 * s * s, 3.0)
    out = np.clip(np.rint(cost.astype(np.float64) * mult), 1,
                  65534).astype(np.uint16)
    out[s > 0.45] = SENTINEL
    out[cost == SENTINEL] = SENTINEL
    return out


def snap_passable(cost, r, c, max_radius=300):
    """Nearest passable cell (same convention as the real-world harness)."""
    rows, cols = cost.shape
    r = int(min(max(r, 0), rows - 1))
    c = int(min(max(c, 0), cols - 1))
    for rad in range(0, max_radius, 10):
        rs = slice(max(r - rad - 10, 0), min(r + rad + 11, rows))
        cs = slice(max(c - rad - 10, 0), min(c + rad + 11, cols))
        ok = np.argwhere(cost[rs, cs] != SENTINEL)
        if ok.size:
            d2 = ((ok[:, 0] + rs.start - r) ** 2
                  + (ok[:, 1] + cs.start - c) ** 2)
            rr, cc = ok[int(d2.argmin())]
            return int(rr + rs.start), int(cc + cs.start)
    raise ValueError(f"no passable cell near ({r},{c})")


def build_pair(cost, name):
    rows, cols = cost.shape
    if name == "diag":
        raw = (0.05 * rows, 0.05 * cols, 0.95 * rows, 0.95 * cols)
    elif name == "west_east":
        raw = (rows // 2, 0.05 * cols, rows // 2, 0.95 * cols)
    elif name == "fixed5km":
        # 2 * half * sqrt(2) = 5001 m at 1 m resolution; clamped on
        # windows too small to hold it (dry run).
        half = min(1768, (min(rows, cols) - 1) // 2 - 1)
        cr, cc = rows // 2, cols // 2
        raw = (cr - half, cc - half, cr + half, cc + half)
    else:
        raise ValueError(name)
    sr, sc = snap_passable(cost, raw[0], raw[1])
    tr, tc = snap_passable(cost, raw[2], raw[3])
    return sr, sc, tr, tc


# ----------------------------------------------------------------------
# Route accounting
# ----------------------------------------------------------------------

def route_stats(cells, cost):
    """Project cost convention (``path_cost``: sum of the path's cell
    values) plus geometric length. Comparable across backends only within
    one neighbourhood — an R2 path skips the intermediate cells an R1
    path would have to enter."""
    from pyorps.utils.path_core import path_cost_uint32

    if cells is None or len(cells) == 0:
        return {"route_cells": 0, "route_cost_cellsum": None,
                "route_len_m": None}
    idx = np.ascontiguousarray(np.asarray(cells, dtype=np.uint32))
    cols = int(cost.shape[1])
    rr = (idx.astype(np.int64) // cols)
    cc = (idx.astype(np.int64) % cols)
    return {
        "route_cells": int(idx.size),
        "route_cost_cellsum": float(path_cost_uint32(idx, cost, cols)),
        "route_len_m": float(np.hypot(np.diff(rr), np.diff(cc)).sum()),
    }


def walk_pred(pred, dist, src, tgt):
    """Target -> source predecessor walk (mirrors RasterGPUAPI)."""
    if not np.isfinite(dist[tgt]):
        return None
    path = [int(tgt)]
    cur = int(tgt)
    guard = dist.size
    while cur != src and guard > 0:
        nxt = int(pred[cur])
        if nxt < 0:
            return None
        path.append(nxt)
        cur = nxt
        guard -= 1
    if cur != src:
        return None
    path.reverse()
    return path


def _base_row(size, scenario, pair, backend, neighborhood):
    return {
        "size": size, "cells": size * size, "scenario": scenario,
        "pair": pair, "backend": backend,
        "neighborhood": neighborhood,
        "neighborhood_applicable": neighborhood is not None,
        "status": "ok",
    }


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------

def run_cython(cost, steps, src, tgt, reps):
    from pyorps.utils._dijkstra import dijkstra_2d_cython

    stats, path = timed_reps(
        lambda: dijkstra_2d_cython(cost, steps, np.uint32(src),
                                   np.uint32(tgt)),
        reps=reps, label="cython solve")
    row = {"phases": {"solve_s": stats["report_s"]},
           "timing": stats,
           "solver_dist": None}      # dijkstra_2d_cython returns the path only
    if path is None or len(path) == 0:
        row["status"] = "unreachable"
        return row
    row.update(route_stats(path, cost))
    return row


def run_raster_gpu(cost, steps, src, tgt, reps, faults, sample_interval):
    from pyorps.utils.sssp_gpu import sssp_raster_gpu_v5

    targets = np.array([int(tgt)], dtype=np.int32)

    def solve():
        return sssp_raster_gpu_v5(cost, steps, int(src),
                                  target_indices=targets,
                                  return_predecessor=True)

    ok, packed, incident = faults.guard(
        f"raster_gpu {cost.shape[0]}^2 x{steps.shape[0]}",
        lambda: timed_reps(solve, reps=reps, sample=True,
                           sample_interval=sample_interval,
                           label="v5 solve"))
    if not ok:
        return {"status": "gpu_fault", "incident": incident}
    stats, (dist, pred) = packed
    pool = gpu_pool_bytes()

    t0 = time.perf_counter()
    path = walk_pred(pred, dist, int(src), int(tgt))
    extract_s = time.perf_counter() - t0

    row = {"phases": {"solve_s": stats["report_s"],
                      "pred_walk_s": extract_s},
           "timing": stats,
           "gpu_pool": pool,
           "solver_dist": (float(dist[tgt])
                           if np.isfinite(dist[tgt]) else None)}
    if path is None:
        row["status"] = "unreachable"
    else:
        row.update(route_stats(path, cost))
    del dist, pred
    free_gpu_pool()
    return row


def run_raster_fim(cost, src, tgt, reps, faults, sample_interval):
    from pyorps.utils.eikonal_gpu import (
        FINITE_LIMIT, eikonal_raster_gpu, polyline_to_cells, trace_paths_gpu)

    rows, cols = cost.shape
    holder = {}

    def solve():
        t_field, n_outer, (t_tr, d_tr) = eikonal_raster_gpu(
            cost, int(src), target_index=int(tgt),
            return_iterations=True, return_trace_field=True)
        holder.update(t_field=t_field, n_outer=n_outer, t_tr=t_tr, d_tr=d_tr)
        return t_field

    ok, packed, incident = faults.guard(
        f"raster_fim {rows}^2",
        lambda: timed_reps(solve, reps=reps, sample=True,
                           sample_interval=sample_interval,
                           label="fim solve"))
    if not ok:
        return {"status": "gpu_fault", "incident": incident}
    solve_stats, t_field = packed
    pool = gpu_pool_bytes()

    row = {"phases": {"solve_s": solve_stats["report_s"]},
           "timing": solve_stats,
           "gpu_pool": pool,
           "fim_outer_passes": int(holder["n_outer"]),
           "solver_dist": float(t_field.ravel()[tgt])}

    ok, packed, incident = faults.guard(
        f"raster_fim trace {rows}^2",
        lambda: timed_reps(
            lambda: trace_paths_gpu(holder["t_tr"], int(src), [int(tgt)],
                                    t_device=holder["d_tr"])[0],
            reps=1, label="fim trace"))
    if not ok:
        row["status"] = "gpu_fault"
        row["incident"] = incident
        return row
    trace_stats, poly = packed
    row["phases"]["trace_s"] = trace_stats["report_s"]

    if poly is None or row["solver_dist"] >= FINITE_LIMIT:
        row["status"] = "unreachable"
        free_gpu_pool()
        return row

    forbidden = t_field >= FINITE_LIMIT
    cells_stats, cells = timed_reps(
        lambda: polyline_to_cells(poly[::-1], rows, cols,
                                  forbidden_mask=forbidden),
        reps=1, label="fim rasterize")
    row["phases"]["cells_s"] = cells_stats["report_s"]
    row.update(route_stats(cells, cost))
    holder.clear()
    free_gpu_pool()
    return row


def transfer_probe(cost, faults):
    """H2D / D2H wall-clock for one full-window raster and one full float32
    field — the transfer terms phase-1 items 1.2/1.3/5.6 attack."""
    def probe():
        import cupy as cp
        t0 = time.perf_counter()
        d_raster = cp.asarray(cost)
        cp.cuda.Stream.null.synchronize()
        h2d = time.perf_counter() - t0

        d_field = cp.empty(cost.size, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        host = d_field.get()
        d2h = time.perf_counter() - t0

        out = {"raster_h2d_s": h2d, "field_d2h_s": d2h,
               "raster_bytes": int(cost.nbytes),
               "field_bytes": int(host.nbytes)}
        del d_raster, d_field, host
        free_gpu_pool()
        return out

    ok, value, incident = faults.guard("transfer probe", probe)
    return value if ok else {"error": incident}


def warm_compile(steps, faults, backends):
    """Compile the CUDA kernels on a 128^2 raster so the first sized run
    measures the solve, not NVRTC."""
    out = {}
    tiny = np.full((128, 128), 10, dtype=np.uint16)
    if "raster_gpu" in backends:
        from pyorps.utils.sssp_gpu import sssp_raster_gpu_v5
        ok, packed, _ = faults.guard(
            "warm v5", lambda: timed_reps(
                lambda: sssp_raster_gpu_v5(tiny, steps, 0,
                                           return_predecessor=True),
                reps=1, label="warm v5"))
        out["v5_s"] = packed[0]["report_s"] if ok else None
    if "raster_fim" in backends:
        from pyorps.utils.eikonal_gpu import eikonal_raster_gpu
        ok, packed, _ = faults.guard(
            "warm fim", lambda: timed_reps(
                lambda: eikonal_raster_gpu(tiny, 0), reps=1,
                label="warm fim"))
        out["fim_s"] = packed[0]["report_s"] if ok else None
    free_gpu_pool()
    return out


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------

def resolve_steps(name):
    """R1 keeps the literal step array the GPU harnesses have always used
    (identical set to ``get_neighborhood_steps('r1')``), so r1 rows stay
    comparable with the recorded result files."""
    from pyorps.utils.neighborhood import get_neighborhood_steps

    return (R1_LITERAL if str(name).lower() in ("r1", "1")
            else get_neighborhood_steps(name, directed=True))


def fits_in_vram(backend, cells, free_bytes):
    if free_bytes is None:
        return True, None
    need = VRAM_PER_CELL[backend] * cells
    return need < 0.85 * free_bytes, need


def run_scenario(name, cost, args, gpu_ok, faults, free_bytes, rows_out):
    size, cols = cost.shape
    forbidden = float((cost == SENTINEL).mean())
    print(f"\n=== {name} | {size}x{cols} = {cost.size / 1e6:.1f} M cells "
          f"| {forbidden:.2%} forbidden ===")

    for pair_name in args.pairs:
        sr, sc, tr, tc = build_pair(cost, pair_name)
        src = sr * cols + sc
        tgt = tr * cols + tc
        euclid = float(np.hypot(tr - sr, tc - sc))
        meta = {"source": [sr, sc], "target": [tr, tc], "euclid_m": euclid}
        print(f"-- pair {pair_name}: ({sr},{sc}) -> ({tr},{tc}), "
              f"{euclid:.0f} m euclid")

        for nb in args.neighborhoods:
            steps = resolve_steps(nb)

            if "cython" in args.backends:
                row = _base_row(size, name, pair_name, "cython", nb)
                row.update(meta)
                row["n_steps"] = int(steps.shape[0])
                if size > args.cython_max_size:
                    row["status"] = "skipped_size_limit"
                else:
                    row.update(run_cython(cost, steps, src, tgt, args.reps))
                row["host_peak_rss_bytes"] = peak_rss_bytes()
                rows_out.append(row)
                _print_row(row)

            if "raster_gpu" in args.backends:
                row = _base_row(size, name, pair_name, "raster_gpu", nb)
                row.update(meta)
                row["n_steps"] = int(steps.shape[0])
                fits, need = fits_in_vram("raster_gpu", size * size,
                                          free_bytes)
                row["vram_estimate_bytes"] = need
                if not gpu_ok:
                    row["status"] = "skipped_gpu_gate"
                elif faults.context_dead:
                    row["status"] = "skipped_context_dead"
                elif not fits:
                    row["status"] = "skipped_vram"
                else:
                    row.update(run_raster_gpu(cost, steps, src, tgt,
                                              args.reps, faults,
                                              args.sample_interval))
                row["host_peak_rss_bytes"] = peak_rss_bytes()
                rows_out.append(row)
                _print_row(row)

        if "raster_fim" in args.backends:
            # The eikonal backend has no neighbourhood: it discretizes a
            # continuous PDE, not a lattice graph. Emitting one row per
            # neighbourhood would fabricate a dependency it does not have.
            row = _base_row(size, name, pair_name, "raster_fim", None)
            row.update(meta)
            fits, need = fits_in_vram("raster_fim", size * size, free_bytes)
            row["vram_estimate_bytes"] = need
            if not gpu_ok:
                row["status"] = "skipped_gpu_gate"
            elif faults.context_dead:
                row["status"] = "skipped_context_dead"
            elif not fits:
                row["status"] = "skipped_vram"
            else:
                row.update(run_raster_fim(cost, src, tgt, args.reps, faults,
                                          args.sample_interval))
            row["host_peak_rss_bytes"] = peak_rss_bytes()
            rows_out.append(row)
            _print_row(row)


def _print_row(row):
    tag = f"{row['backend']:>10}"
    if row["neighborhood"]:
        tag += f" {row['neighborhood']:>2}"
    else:
        tag += " --"
    if row["status"] != "ok":
        print(f"   {tag}: {row['status']}")
        return
    phases = row["phases"]
    total = sum(v for v in phases.values() if v is not None)
    parts = " ".join(f"{k.replace('_s', '')} {v:.3f}"
                     for k, v in phases.items() if v is not None)
    thermal = row.get("timing", {}).get("thermal", {})
    temp = (thermal.get("temp_c") or {}).get("max")
    sm = (thermal.get("sm_mhz") or {}).get("min")
    env = ""
    if temp is not None:
        env = f" | {temp:.0f} C, SM min {sm:.0f} MHz" if sm else \
            f" | {temp:.0f} C"
    print(f"   {tag}: total {total:7.3f} s ({parts}) | cost "
          f"{row.get('route_cost_cellsum')} | "
          f"{row.get('route_len_m', 0) or 0:.0f} m{env}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=list(DEFAULT_SIZES))
    ap.add_argument("--max-size", type=int, default=5000,
                    help="cap on --sizes; the large sizes are opt-in "
                         "because a 10000^2 GPU solve may take the CUDA "
                         "context down with it")
    ap.add_argument("--reps", type=int, default=1,
                    help="timed runs per measurement; the first is always "
                         "reported separately from the steady state")
    ap.add_argument("--neighborhoods", nargs="+", default=["r1"])
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS),
                    choices=list(BACKENDS))
    ap.add_argument("--pairs", nargs="+", default=["fixed5km", "diag"],
                    choices=list(PAIRS))
    ap.add_argument("--cython-max-size", type=int, default=8000,
                    help="skip the serial CPU kernel above this size")
    ap.add_argument("--slope", action="store_true",
                    help="also run scenario B (isotropic DGM1 slope)")
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="override the shared-machine refusal")
    ap.add_argument("--max-idle-temp", type=float, default=60.0)
    ap.add_argument("--sample-interval", type=float, default=0.5,
                    help="nvidia-smi clock/temperature sampling period")
    ap.add_argument("--no-nvidia-smi", action="store_true",
                    help="skip every nvidia-smi/registry query (offline "
                         "smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny synthetic raster, CPU backend only — "
                         "exercises the plumbing and the result file")
    args = ap.parse_args()

    if args.dry_run:
        args.sizes = [192]
        args.max_size = 192
        args.backends = ["cython"]
        args.reps = min(args.reps, 2)
        args.slope = False

    env = environment_snapshot(include_gpu=not args.no_nvidia_smi)
    print_environment(env)

    wants_gpu = any(b in args.backends for b in ("raster_gpu", "raster_fim"))
    gate = None
    gpu_ok = False
    free_bytes = None
    if wants_gpu and not args.no_nvidia_smi:
        gate = gpu_gate(max_idle_temp_c=args.max_idle_temp,
                        allow_shared=args.allow_shared_gpu)
        print_gate(gate)
        gpu_ok = gate["ok"]
        free_mib = gate["snapshot"].get("memory_free")
        if free_mib is not None:
            free_bytes = free_mib * 1024 * 1024
    elif wants_gpu:
        print("GPU gate: skipped (--no-nvidia-smi) — GPU rows disabled")

    faults = GpuFaultLog()
    base, base_dem = load_base(args.dry_run)

    device = None
    warm = None
    if gpu_ok:
        device = cuda_device_info()
        print(f"CUDA device: {device.get('multiprocessor_count')} SMs, "
              f"cc {device.get('compute_capability')}, cupy "
              f"{device.get('cupy_version')}")
        warm = warm_compile(resolve_steps(args.neighborhoods[0]), faults,
                            args.backends)
        print(f"kernel warm-compile: {warm}")

    sizes = sorted({int(s) for s in args.sizes if s <= args.max_size})
    if not sizes:
        raise SystemExit(f"no size <= --max-size {args.max_size}")

    rows = []
    probes = {}
    started = time.time()
    for size in sizes:
        cost = mosaic(base, size)
        if gpu_ok and "raster_gpu" in args.backends:
            probes[str(size)] = transfer_probe(cost, faults)
        run_scenario("A: combined raster", cost, args, gpu_ok, faults,
                     free_bytes, rows)

        if args.slope:
            if base_dem is None:
                print("  scenario B SKIPPED: no cached window_dem.npy")
            else:
                cost_b = mosaic(apply_slope(base, base_dem), size)
                run_scenario("B: combined x slope", cost_b, args, gpu_ok,
                             faults, free_bytes, rows)
                del cost_b
        del cost

    payload = {
        "benchmark": "scale_5km",
        "dry_run": bool(args.dry_run),
        "args": vars(args),
        "environment": env,
        "gpu_gate": gate,
        "cuda_device": device,
        "warm_compile_s": warm,
        "tiling": {
            "base_shape": list(base.shape),
            "scheme": "mirror (symmetric), period 2*n_src, reproducible",
            "base_forbidden_fraction": float((base == SENTINEL).mean()),
        },
        "reps": args.reps,
        "transfer_probe": probes,
        "results": rows,
        "tdr": {
            "settings": env.get("tdr"),
            "incidents": faults.incidents,
            "context_dead": faults.context_dead,
            "event_log": (windows_tdr_events()
                          if (faults.incidents and not args.no_nvidia_smi)
                          else {"available": False,
                                "error": "not queried", "events": []}),
        },
        "wall_clock_s": time.time() - started,
        "host_peak_rss_bytes": peak_rss_bytes(),
    }
    path = write_results("scale_5km", payload)
    print(f"\n{len(rows)} rows, {len(faults.incidents)} GPU incident(s)")
    print(f"results -> {path}")


if __name__ == "__main__":
    main()

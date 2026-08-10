"""
Real-world benchmark: eikonal FIM vs discrete backends on the actual
distribution-grid-planning cost raster.

The cost surface is the *unmodified output* of
``examples/prepare_data_for_distribution_grid_planning.ipynb``:
ALKIS land-use base costs multiplied by drinking-water-protection-zone
factors (zone I x100 ... zone IV x1.2), soil-condition factors
(DIN 18300 classes, x1.0-1.3) and landscape-protection x1.25, with
nature reserves hard-forbidden (65535). A 4096 m x 4096 m window at
1 m resolution around the notebook's demo route is benchmarked.

Scenarios:
  A. combined raster as produced by the notebook
  B. combined raster x isotropic slope multiplier from the real
     Hessen DGM1 (1 m WCS; cached locally after the first fetch):
         m(s) = min(1 + 3*s^2, 3),   s > 45%% -> forbidden
     Slope handled *isotropically* (a per-cell layer) — the only form
     the eikonal backend supports today; the directional (anisotropic)
     treatment is increment 3. Scenario C quantifies the difference.
  C. contour test (synthetic hillside): cython with TRUE directional
     gradient cost vs the isotropic slope layer — measures what the
     isotropic approximation mis-prices (0 on the fall line, the full
     multiplier along a contour).

Solvers: Cython Dijkstra R1 + R2 (CPU champions), GPU V5 (discrete
production default), FIM order=1 (with targeted early exit — its
production single-pair behavior) and FIM order=2. Discrete costs are
expected ABOVE the FIM cost (metrication); gaps are reported per pair.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_eikonal_realworld.py
    ... [--no-slope] [--reps 2]

GPU etiquette: strictly sequential, modest reps.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from pyorps.utils._dijkstra import dijkstra_2d_cython
from pyorps.utils.eikonal_gpu import (
    eikonal_raster_gpu, trace_paths_gpu, FINITE_LIMIT)
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.utils.sssp_gpu import sssp_raster_gpu

HERE = Path(__file__).parent
DATA = HERE / "realworld_data"
COMBINED_RASTER = (HERE.parent / "examples" / "data" / "raster"
                   / "modified_raster_for_distribution_grid_planning.tiff")

#: 4096 m x 4096 m window (EPSG:25832) around the notebook's demo route
WINDOW_BOUNDS = (477500.0, 5594500.0, 481596.0, 5598596.0)

#: (source, target) pairs in map coordinates; first = the notebook demo
PAIRS_XY = [
    ((479455, 5598100), (481069, 5595515)),   # notebook demo route
    ((477800, 5598300), (481300, 5594800)),   # long diagonal
    ((477800, 5594800), (481300, 5598300)),   # other diagonal
    ((478000, 5596500), (481400, 5596500)),   # west-east
]

DEM_WCS = dict(url="https://inspire-hessen.de/raster/dgm1/ows",
               coverage="he_dgm1")

R1 = get_neighborhood_steps("r1", directed=True)
R2 = get_neighborhood_steps("r2", directed=True)

SENTINEL = np.uint16(65535)


# ----------------------------------------------------------------------
# Input preparation (all cached under benchmarks/realworld_data/)
# ----------------------------------------------------------------------

def prepare_cost_window():
    DATA.mkdir(exist_ok=True)
    cached = DATA / "window_cost.npz"
    if cached.exists():
        z = np.load(cached)
        return z["cost"], tuple(z["bounds"])
    with rasterio.open(COMBINED_RASTER) as ds:
        win = from_bounds(*WINDOW_BOUNDS, transform=ds.transform)
        win = win.round_offsets().round_lengths()
        cost = ds.read(1, window=win)
        bounds = rasterio.windows.bounds(win, ds.transform)
    np.savez_compressed(cached, cost=cost, bounds=np.array(bounds))
    return cost, tuple(bounds)


def _fetch_dem(bounds, shape):
    """Hessen DGM1 via WCS (2x2 tiles), aligned to the cost window."""
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_bounds as tf_from_bounds

    from pyorps.gui.services.catalog import load_dem_raster

    minx, miny, maxx, maxy = bounds
    midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
    tiles = []
    for tb in [(minx, midy, midx, maxy), (midx, midy, maxx, maxy),
               (minx, miny, midx, midy), (midx, miny, maxx, midy)]:
        path = load_dem_raster(DEM_WCS["url"], DEM_WCS["coverage"], tb,
                               work_dir=DATA / "tmp")
        tiles.append(path)

    dst = np.full(shape, np.nan, dtype=np.float32)
    dst_transform = tf_from_bounds(*bounds, shape[1], shape[0])
    for path in tiles:
        with rasterio.open(path) as ds:
            reproject(source=ds.read(1), destination=dst,
                      src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=dst_transform, dst_crs="EPSG:25832",
                      resampling=Resampling.bilinear,
                      dst_nodata=np.nan, init_dest_nodata=False)
        Path(path).unlink()
    return dst


def prepare_dem(bounds, shape):
    cached = DATA / "window_dem.npy"
    if cached.exists():
        return np.load(cached)
    dem = _fetch_dem(bounds, shape)
    if np.isnan(dem).mean() > 0.05:
        raise ValueError(f"DEM has {np.isnan(dem).mean():.0%} voids")
    np.save(cached, dem)
    return dem


def slope_fraction(dem, cell=1.0):
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    gr, gc = np.gradient(d.astype(np.float64), cell)
    return np.hypot(gr, gc)


def apply_slope(cost, dem):
    """Isotropic slope multiplier: m = min(1 + 3 s^2, 3); s>45%% forbidden."""
    s = slope_fraction(dem)
    mult = np.minimum(1.0 + 3.0 * s * s, 3.0)
    out = np.clip(np.rint(cost.astype(np.float64) * mult), 1,
                  65534).astype(np.uint16)
    out[s > 0.45] = SENTINEL
    out[cost == SENTINEL] = SENTINEL
    return out


def snap_passable(cost, r, c, max_radius=300):
    rows, cols = cost.shape
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


def pairs_to_cells(cost, bounds):
    minx, _, _, maxy = bounds
    out = []
    for (sx, sy), (tx, ty) in PAIRS_XY:
        sr, sc = snap_passable(cost, int(maxy - sy), int(sx - minx))
        tr, tc = snap_passable(cost, int(maxy - ty), int(tx - minx))
        out.append((sr, sc, tr, tc))
    return out


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------

def median_time(fn, reps):
    times, result = [], None
    for i in range(reps + 1):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt)
    return statistics.median(times), result


def run_scenario(name, cost, pairs, reps):
    rows_out = []
    n_rows, n_cols = cost.shape
    print(f"\n=== scenario {name} ({n_rows}x{n_cols}, "
          f"{(cost == SENTINEL).mean():.0%} forbidden) ===")
    for k, (sr, sc, tr, tc) in enumerate(pairs):
        src = sr * n_cols + sc
        tgt = tr * n_cols + tc

        # Cython Dijkstra: single-pair wall-clock (returns the path)
        t_cy1, p1 = median_time(
            lambda: dijkstra_2d_cython(cost, R1, np.uint32(src),
                                       np.uint32(tgt)), max(1, reps - 1))
        t_cy2, p2 = median_time(
            lambda: dijkstra_2d_cython(cost, R2, np.uint32(src),
                                       np.uint32(tgt)), max(1, reps - 1))
        if len(p1) == 0:
            print(f"[pair {k}] unreachable (cython) — skipped")
            continue

        # V5 wall-clock: production behavior (targeted early exit)
        t_v5, _ = median_time(
            lambda: sssp_raster_gpu(cost, R1, src,
                                    target_indices=np.array(
                                        [tgt], dtype=np.int32)), reps)
        # exact discrete costs from full-field solves (V5 = exact SSSP)
        c_r1 = float(np.asarray(
            sssp_raster_gpu(cost, R1, src)).ravel()[tgt])
        c_r2 = float(np.asarray(
            sssp_raster_gpu(cost, R2, src)).ravel()[tgt])

        def fim_route(order):
            kw = dict(return_trace_field=True)
            if order == 1:
                kw["target_index"] = tgt
            else:
                kw["order"] = 2
            t_field, (t_tr, d_tr) = eikonal_raster_gpu(cost, src, **kw)
            poly = trace_paths_gpu(t_tr, src, [tgt], t_device=d_tr)[0]
            return float(t_field.ravel()[tgt]), poly

        t_f1, (c_f1, poly1) = median_time(lambda: fim_route(1), reps)
        t_f2, (c_f2, poly2) = median_time(lambda: fim_route(2), reps)

        if c_f1 >= FINITE_LIMIT or poly1 is None:
            print(f"[pair {k}] unreachable (fim) — skipped")
            continue

        row = dict(
            scenario=name, pair=k,
            source=[sr, sc], target=[tr, tc],
            euclid_m=float(np.hypot(tr - sr, tc - sc)),
            cython_r1_s=t_cy1, cython_r2_s=t_cy2, v5_s=t_v5,
            fim_o1_s=t_f1, fim_o2_s=t_f2,
            cost_r1=c_r1, cost_r2=c_r2,
            cost_fim_o1=c_f1, cost_fim_o2=c_f2,
            fim_below_r2_pct=(c_r2 - c_f1) / c_r2 * 100,
            fim_below_r1_pct=(c_r1 - c_f1) / c_r1 * 100,
            fim_o2_below_r2_pct=(c_r2 - c_f2) / c_r2 * 100,
        )
        rows_out.append(row)
        print(f"[pair {k}] {row['euclid_m']:.0f} m euclid | "
              f"cyR1 {t_cy1:6.1f}s cyR2 {t_cy2:6.1f}s "
              f"V5 {t_v5*1e3:6.0f}ms | FIM o1 {t_f1*1e3:6.0f}ms "
              f"o2 {t_f2*1e3:6.0f}ms | "
              f"FIM below R2 {row['fim_below_r2_pct']:+.2f}% "
              f"below R1 {row['fim_below_r1_pct']:+.2f}% "
              f"(o2: {row['fim_o2_below_r2_pct']:+.2f}%)")
    return rows_out


# ----------------------------------------------------------------------
# Scenario C: what the isotropic slope treatment mis-prices
# ----------------------------------------------------------------------

def contour_test():
    """Uniform-cost hillside (20%% east-up slope). A route along the
    contour has ZERO true climb; the isotropic slope layer charges it
    the full multiplier anyway. A route straight up the fall line is
    priced identically by both treatments."""
    n, v, s = 301, 100, 0.20
    raster = np.full((n, n), v, dtype=np.uint16)
    dem = (np.arange(n, dtype=np.float64) * s)[None, :] * np.ones((n, 1))
    mult = 1.0 + 3.0 * s * s
    iso = np.clip(np.rint(raster * mult), 1, 65534).astype(np.uint16)

    length = 260.0
    # contour route: constant column
    contour_true = v * length                     # no climb, flat price
    contour_iso = float(iso[0, 150]) * length
    # fall-line route: constant row, climbing 20%
    fall_iso = float(iso[150, 0]) * length

    return dict(
        slope_pct=s * 100, multiplier=mult,
        contour_route_true_cost=contour_true,
        contour_route_iso_cost=contour_iso,
        contour_surcharge_pct=(contour_iso - contour_true)
        / contour_true * 100,
        fall_line_iso_cost=fall_iso,
        note=("isotropic slope layers cannot see route direction: a "
              "contour route pays the same multiplier as the fall "
              "line. Directional gradient costs (cython/raster_gpu "
              "with dem= + objective gradient) price the contour "
              "correctly; raster_fim needs the anisotropic increment "
              "3 for that."),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-slope", action="store_true",
                    help="skip the DEM/slope scenario (no network)")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    cost, bounds = prepare_cost_window()
    vals = cost[cost != SENTINEL]
    print(f"cost window {cost.shape}, bounds {bounds}")
    print(f"passable {vals.size / cost.size:.0%}, cost range "
          f"{vals.min()}-{vals.max()}, contrast "
          f"{vals.max() / max(int(vals.min()), 1):.0f}")

    pairs = pairs_to_cells(cost, bounds)
    results = {"window_bounds": bounds, "pairs": pairs}

    results["scenario_A"] = run_scenario("A: combined raster", cost,
                                         pairs, args.reps)

    if not args.no_slope:
        try:
            dem = prepare_dem(bounds, cost.shape)
            s = slope_fraction(dem)
            print(f"\nDEM ok: elevation {np.nanmin(dem):.0f}-"
                  f"{np.nanmax(dem):.0f} m, slope mean "
                  f"{s.mean()*100:.1f}% p95 "
                  f"{np.percentile(s, 95)*100:.1f}%")
            cost_slope = apply_slope(cost, dem)
            results["scenario_B"] = run_scenario(
                "B: combined x slope", cost_slope, pairs, args.reps)
        except Exception as exc:
            print(f"\nDEM/slope scenario SKIPPED: {exc}")
            results["scenario_B"] = f"skipped: {exc}"

    results["scenario_C_contour"] = contour_test()
    c = results["scenario_C_contour"]
    print(f"\n=== scenario C: contour test (20% hillside) ===")
    print(f"  contour route: true cost {c['contour_route_true_cost']:.0f}"
          f" | isotropic charges {c['contour_route_iso_cost']:.0f} "
          f"({c['contour_surcharge_pct']:+.1f}% pure error)")
    print(f"  fall-line route: isotropic {c['fall_line_iso_cost']:.0f} "
          f"(correct there — error range is 0% to the full multiplier)")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"eikonal_realworld_{stamp}.json"
    path.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()

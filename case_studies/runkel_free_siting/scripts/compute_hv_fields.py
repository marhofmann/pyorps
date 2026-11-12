"""The 110 kV cable-leg field on the GPU (plan 2026-10-06-paper-run-pipeline.md, stage S2).

For each of the three grid connection points (``layers/pccs.gpkg``, ids kept) one search over
``hv_cost_priced_1m.tif`` gives the cheapest 110 kV cable to every cell; the output is the cheapest of the three and
which connection point wins. Float32 on the GPU: a BOUND with a rounding margin, used only to remove sites; the
winner's leg is re-solved in float64 on the CPU (owner decision D-a). ``--check-cpu`` runs the exact CPU kernel on
the same crop. Cable field only: overhead and mixed legs are a separate bound.

Writes two small GeoTIFFs (the cheapest field and the winning connection point, every 5th cell, deflate) and a
sidecar JSON. A temperature guard ends the process at ``--max-temp``.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_hv_fields.py --box 441701 5585000 443701 5587000 --check-cpu --out-dir <scratch> --no-register
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_hv_fields.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402
from compute_access_field import TempGuard  # noqa: E402

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
OUT = CASE / "data" / "rasters"
SENTINEL_U32 = 4294967295
SEARCH_RADIUS_M = 200
TIE_M = 1.5
COOL_S = int(os.environ.get("PYORPS_COOL_S", 90))     # 0 on a desktop card with its own cooling
NODATA = 3.4e38


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--check-cpu", action="store_true")
    ap.add_argument("--max-temp", type=float, default=85.0)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    from pyorps.siting.data_manifest import DataManifest, LayerEntry
    from pyorps.siting.terminals import load_terminals
    from pyorps.utils.neighborhood import get_neighborhood_steps
    from pyorps.utils.sssp_gpu import GpuSsspSession

    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["hv_cost_priced_1m", "terminals"])
    with rasterio.open(m.file_of("hv_cost_priced_1m")) as s:
        gx0, gy1, gw, gh, crs = int(s.bounds.left), int(s.bounds.top), s.width, s.height, s.crs
    x0, y0, x1, y1 = args.box if args.box else (gx0, gy1 - gh, gx0 + gw, gy1)
    width, height = x1 - x0, y1 - y0
    with rasterio.open(m.file_of("hv_cost_priced_1m")) as s:
        cost = s.read(1, window=Window(x0 - gx0, gy1 - y1, width, height))
    steps = np.asarray(get_neighborhood_steps(args.k, directed=True), dtype=np.int8)

    # Owner rule (2026-10-06, D-h): no site and no route on forbidden cells. A connection point inside a forbidden area
    # (a fenced yard) is therefore reached at the allowed cells nearest to it: every allowed cell within TIE_M of the
    # nearest distance is a terminal with label 0. The gap to the point is recorded, not priced (connection-point term).
    c_min = float(cost[cost != 65535].min())
    seeds_of: dict[int, tuple] = {}
    for t_ in load_terminals(m.file_of("terminals")):
        col, row = int(t_.xy[0] - x0), int(y1 - t_.xy[1])
        if not (0 <= col < width and 0 <= row < height):
            continue
        rad = SEARCH_RADIUS_M
        r0_, r1_, c0_, c1_ = max(0, row - rad), min(height, row + rad + 1), max(0, col - rad), min(width, col + rad + 1)
        yy, xx = np.nonzero(cost[r0_:r1_, c0_:c1_] != 65535)
        if yy.size == 0:
            continue
        dist = np.hypot(yy + r0_ - row, xx + c0_ - col)
        keep = dist <= dist.min() + TIE_M
        idx = (yy[keep] + r0_) * width + (xx[keep] + c0_)
        seeds_of[int(t_.id)] = (idx.astype(np.int64), np.zeros(int(keep.sum()), dtype=np.float32), float(dist.min()))
    fake = False
    if not seeds_of:                                      # a validation crop without a connection point
        rng = np.random.default_rng(0)
        for i, c in enumerate(rng.choice(np.flatnonzero(cost.ravel() != 65535), size=3, replace=False)):
            seeds_of[i] = (np.array([int(c)], dtype=np.int64), np.zeros(1, dtype=np.float32), 0.0)
        fake = True
    pcc = {k: v[0] for k, v in seeds_of.items()}
    print(f"box {x0},{y0}..{x1},{y1}: {width} x {height} = {width * height} cells; connection points "
          f"{sorted(pcc)}{' (STAND-IN SEEDS: no connection point in this crop)' if fake else ''}; gaps to the "
          f"nearest allowed cell (m): { {k: round(v[2], 1) for k, v in seeds_of.items()} }", flush=True)

    best = np.full(cost.shape, np.inf, dtype=np.float32)
    who = np.full(cost.shape, 255, dtype=np.uint8)
    per_pcc = {}
    with TempGuard(args.max_temp) as guard:
        sess = GpuSsspSession(cost, steps, arena_factor=1.0)
        print(f"GPU session holds {sess.device_bytes / 1e9:.2f} GB", flush=True)
        for pid, idx in sorted(pcc.items()):
            t1 = time.time()
            f = sess.solve_seeded(idx, seeds_of[pid][1]).reshape(cost.shape)
            better = f < best
            best[better], who[better] = f[better], pid
            per_pcc[pid] = {"reached_share": float(np.isfinite(f).mean()),
                            "gpu_solve_s_not_a_performance_figure": round(time.time() - t1, 1)}
            del f, better
            time.sleep(COOL_S if not args.box else 0)
        sess.close(free_pool=True)
    reach = np.isfinite(best)
    d_max = float(best[reach].max())
    k_max = int(np.ceil(d_max / c_min)) + 2
    # Per-step relative error eta = 3 u, u = 2^-24: the float32 cost factor (<= 0.44 u), the product with the integer
    # cell sum (<= u) and the addition (<= u) give 2.44 u <= 3 u. A path of k steps then has d_f in d (1 +- k eta), so
    # d_f (1 - k_max eta) is a lower bound (math.md, M-9.1). The first version used eta = u and was too tight.
    slack = k_max * 3.0 * 2.0 ** -24
    stats = {"cells": int(cost.size), "connection_points": sorted(pcc), "stand_in_seeds": fake, "terminal_rule": "nearest allowed cells within TIE_M, label 0 (no route on forbidden cells)", "tie_m": TIE_M, "min_step_cost": c_min, "gap_to_nearest_allowed_cell_m": {k: v[2] for k, v in seeds_of.items()},
             "reached_share_of_all_cells": float(reach.mean()), "dist_max_eur": d_max,
             "k_max_rigorous": k_max, "rounding_slack_relative": slack, "per_pcc": per_pcc,
             "gpu_temp_max_c": guard.max_seen}
    print(stats, flush=True)

    if args.check_cpu:
        from pyorps.graph.drain import drain

        order = sorted(seeds_of)
        seeds = np.concatenate([seeds_of[k][0] for k in order])
        labels = np.concatenate([seeds_of[k][1] for k in order]).astype(np.float64)
        exact = drain(cost, steps, seeds, labels)
        fin = np.isfinite(exact)
        rel = np.abs(best[fin] - exact[fin]) / np.maximum(exact[fin], 1e-9)
        stats["cpu_check"] = {"same_reachable_set": bool(np.array_equal(fin, reach)),
                              "max_rel_deviation": float(rel.max()),
                              "lower_bound_violations": int((best[fin] * (1 - slack) > exact[fin] * (1 + 1e-12)).sum())}
        print("CPU check:", stats["cpu_check"], flush=True)

    site = m.file_of("site_field_envelope_cents")
    with rasterio.open(site) as s:
        sx0, sy1, sw, sh = int(s.bounds.left), int(s.bounds.top), s.width, s.height
        ix0, ix1, iy0, iy1 = max(x0, sx0), min(x1, sx0 + sw), max(y0, sy1 - sh), min(y1, sy1)
        if ix0 < ix1 and iy0 < iy1:
            sf = s.read(1, window=Window(ix0 - sx0, sy1 - iy1, ix1 - ix0, iy1 - iy0))
            d = best[(y1 - iy1):(y1 - iy0), (ix0 - x0):(ix1 - x0)][sf != SENTINEL_U32]
            ok = np.isfinite(d)
            if d.size:
                stats["site_anchors"] = {"feasible": int(d.size), "reached_share": float(ok.mean()),
                                         "hv_eur_min": float(d[ok].min()), "hv_eur_median": float(np.median(d[ok])),
                                         "hv_eur_p95": float(np.percentile(d[ok], 95)),
                                         "note": "stand-in prices and assumption-tagged inputs: not quotable"}
                print("anchors:", stats["site_anchors"], flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = from_origin(x0, y1, 5.0, 5.0)

    def write(path: Path, arr, dtype, nodata, predictor):
        profile = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype=dtype, crs=crs,
                       transform=tr, nodata=nodata, tiled=True, blockxsize=256, blockysize=256, compress="deflate",
                       predictor=predictor, zlevel=6)
        part = path.with_name(path.name + ".partial")
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(arr, 1)
        part.replace(path)

    f_path, w_path = out_dir / "hv_field_min_5m_lattice_eur.tif", out_dir / "hv_field_argmin_5m_lattice.tif"
    lat = np.where(np.isfinite(best[::5, ::5]), best[::5, ::5], np.float32(NODATA)).astype(np.float32)
    write(f_path, lat, "float32", NODATA, 3)
    write(w_path, who[::5, ::5], "uint8", 255, 2)
    note = ("float32 GPU field of the 110 kV cable leg to the cheapest connection point (a BOUND: multiply by 1 - "
            "rounding_slack_relative for a lower bound); argmin = connection point id; cable only; every 5th cell")
    (out_dir / "hv_field_sidecar.json").write_text(json.dumps(
        {"stats": stats, "grid_epsg25832": [x0, y0, x1, y1], "note": note,
         "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()}}, indent=1, default=str), encoding="utf-8")
    print(f"wrote {f_path.name} ({f_path.stat().st_size / 1e6:.1f} MB), {w_path.name} ({w_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    if args.no_register or fake:
        print("not registered")
        return 0

    today, script_hash = dt.date.today().isoformat(), dl.file_sha256(__file__)
    for lid, path, what in (("hv_field_min_5m_lattice", f_path, "110 kV cable-leg field (GPU, float32 bound)"),
                            ("hv_field_argmin_5m_lattice", w_path, "winning connection point id of the 110 kV field")):
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(id=lid, kind="other", format="raster", path=None, status="pending",
                                       source=f"built by compute_hv_fields.py: {what}",
                                       licence="derived work; follows the licences of its inputs")
        stamp = dl.now_stamps()
        e = m.register(lid, path, captured=today, crs=str(crs))
        m.annotate(lid, captured_at=stamp["time_local"], notes=note,
                   retrieved_with=f"compute_hv_fields.py (sha256 {script_hash[:16]}), GpuSsspSession.solve_seeded; "
                                  f"python {dl.tool_versions()['python']}",
                   used_for=[f"done {today}: 110 kV field on a 5 m lattice", "planned: stage S2 lower bound; not yet used in a result"])
        dl.append_event(layer_id=lid, service="none (built locally)", what=f"built raster: {what}",
                        file=str(path.relative_to(CASE)), sha256=e.sha256, bytes=path.stat().st_size, **stamp,
                        reconstructed=False, request={"stats": stats, "grid_epsg25832": [x0, y0, x1, y1]}, crs=str(crs),
                        script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
        print(f"registered {lid}, sha256 {e.sha256[:12]}")
    m.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

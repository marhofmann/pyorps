"""Window decomposition bound of the collector for roots outside the turbine window (paper math.md M7.6d).

Two pieces, both written next to the other bound fields:

1. ``Win(u)`` for each MV level: the relaxed subset DW (127 drains) on a window around the seven turbines with a zero-cost
   virtual root on the boundary cells (``pyorps.collector.bounds_window.rho_hat_exit_value``, tested against engine B). CPU,
   float64, exact kernel. The values, the cable-rate floor ``alpha(u)`` and the root bay of each level go to the sidecar.
2. ``mv_window_dtr_5m_lattice_eur.tif``: the trench distance from the window boundary to every cell (one GPU drain over the
   trench weight ``floor((1 + omega) c)``, float32 bound; the driver applies ``d (1 - k_max eta)``).

The driver (``run_bound_and_prune.py``) combines them for a root ``g`` outside the window:
``LB(g) = min_u [Win(u) + alpha(u) n dist(g, W) + bay(u)] + dtr(g)``, and keeps the older bound inside the window.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_window_bound.py --levels 33 30 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
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
NODATA = 3.4e38
FORBIDDEN = 65535


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--levels", type=float, nargs="+", default=[33.0, 30.0, 20.0])
    ap.add_argument("--margin", type=int, default=100, help="window margin around the turbines in m")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-temp", type=float, default=85.0)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    from pyorps.collector.bounds_window import cable_rate_floor, exit_cells, rho_hat_exit_value
    from pyorps.costmodel import CostCatalogue, LossCategories
    from pyorps.costmodel.collector_inputs import CollectorInputs, build_collector_model
    from pyorps.siting.data_manifest import DataManifest, LayerEntry
    from pyorps.utils.neighborhood import get_neighborhood_steps

    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["cost_priced_1m", "turbines"])
    side = json.loads((OUT / "hv_cost_sidecar.json").read_text(encoding="utf-8"))
    omega = float(re.search(r"omega ([0-9.]+)", side["note"]).group(1))
    steps = np.asarray(get_neighborhood_steps(args.k, directed=True), dtype=np.int8)
    turb = m.read_vector("turbines", verify=False)
    with rasterio.open(m.file_of("cost_priced_1m")) as s:
        gx0, gy1, gw, gh, crs = int(s.bounds.left), int(s.bounds.top), s.width, s.height, s.crs
        trow = [int(np.floor(gy1 - g.y)) for g in turb.geometry]
        tcol = [int(np.floor(g.x - gx0)) for g in turb.geometry]
        r0, r1 = max(0, min(trow) - args.margin), min(gh, max(trow) + args.margin + 1)
        c0, c1 = max(0, min(tcol) - args.margin), min(gw, max(tcol) + args.margin + 1)
        win = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    h, w = win.shape
    t_win = [(r - r0) * w + (c - c0) for r, c in zip(trow, tcol)]
    ex = exit_cells(win, steps)
    print(f"window {w} x {h} = {w * h / 1e6:.2f} M cells; {len(ex)} exit cells; omega {omega:.4f}", flush=True)

    sidecar_path = OUT / "mv_window_sidecar.json"
    stats: dict = {"window_full_grid_cells": {"col0": c0, "col1": c1, "row0": r0, "row1": r1}, "omega": omega, "levels": {}}
    if sidecar_path.exists():
        stats["levels"] = json.loads(sidecar_path.read_text(encoding="utf-8")).get("stats", {}).get("levels", {})

    if not args.skip_cpu:
        for u in args.levels:
            model = build_collector_model(CollectorInputs.load(CASE / "config" / "collector_inputs_2026.yaml"),
                                          CostCatalogue.load(CASE / "config" / "cost_catalogue_2026.yaml"),
                                          LossCategories.load(CASE / "data" / "loss_categories.npz"), u_kv=u, rated_mw=[6.8] * 7).model
            t0 = time.time()
            value = rho_hat_exit_value(win, steps, 1.0, t_win, model, trench_mult=1.0 + omega)
            stats["levels"][str(int(u))] = {"win_eur": value, "alpha_eur_per_m_turbine": cable_rate_floor(model),
                                            "bay_eur": float(model.bay_eur), "n_turbines": model.n,
                                            "cpu_s_not_a_performance_figure": round(time.time() - t0, 1)}
            print(f"  {u:.0f} kV: Win = {value:,.0f} EUR", flush=True)
            sidecar_path.write_text(json.dumps({"stats": stats}, indent=1), encoding="utf-8")

    written = []
    if not args.skip_gpu:
        from pyorps.utils.sssp_gpu import GpuSsspSession

        with rasterio.open(m.file_of("cost_priced_1m")) as s:
            c = s.read(1)
        free = c != FORBIDDEN
        tw = np.full(c.shape, FORBIDDEN, dtype=np.uint16)
        tw[free] = np.minimum(np.floor((1.0 + omega) * c[free].astype(np.float64)), FORBIDDEN - 1).astype(np.uint16)
        c_min = float(tw[free].min())
        del c
        er, ec = np.divmod(ex, w)
        seeds = ((er + r0) * gw + (ec + c0)).astype(np.int64)
        seeds = seeds[tw.ravel()[seeds] != FORBIDDEN]
        with TempGuard(args.max_temp) as guard:
            sess = GpuSsspSession(tw, steps, arena_factor=1.0)
            f = sess.solve_seeded(seeds, np.zeros(seeds.size, dtype=np.float32))
            sess.close(free_pool=True)
        fin = np.isfinite(f)
        d_max = float(f[fin].max())
        k_max = int(np.ceil(d_max / c_min)) + 2
        lat = np.where(fin.reshape(tw.shape)[::5, ::5], f.reshape(tw.shape)[::5, ::5], np.float32(NODATA)).astype(np.float32)
        stats["dtr"] = {"seed_cells": int(seeds.size), "d_max_eur": d_max, "min_step_cost": c_min, "k_max_rigorous": k_max,
                        "rounding_slack_relative": k_max * 3 * 2.0 ** -24, "reached_share": float(fin.mean()),
                        "gpu_temp_max_c": guard.max_seen,
                        "note": "float32 GPU bound of the trench distance from the window boundary; multiply by 1 - rounding_slack_relative"}
        print(stats["dtr"], flush=True)
        out = OUT / "mv_window_dtr_5m_lattice_eur.tif"
        profile = dict(driver="GTiff", width=lat.shape[1], height=lat.shape[0], count=1, dtype="float32", crs=crs,
                       transform=from_origin(gx0, gy1, 5.0, 5.0), nodata=NODATA, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3, zlevel=6)
        part = out.with_name(out.name + ".partial")
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(lat, 1)
        part.replace(out)
        written.append(("mv_window_dtr_5m_lattice", out, "trench distance from the turbine-window boundary (GPU, float32 bound)"))
        print(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)

    sidecar_path.write_text(json.dumps({"stats": stats, "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()}}, indent=1), encoding="utf-8")
    if args.no_register or not written:
        return 0
    today, script_hash = dt.date.today().isoformat(), dl.file_sha256(__file__)
    for lid, path, what in written:
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(id=lid, kind="other", format="raster", path=None, status="pending",
                                       source=f"built by compute_window_bound.py: {what}",
                                       licence="derived work; follows the licences of its inputs")
        stamp = dl.now_stamps()
        e = m.register(lid, path, captured=today, crs=str(crs))
        m.annotate(lid, captured_at=stamp["time_local"], notes=what,
                   retrieved_with=f"compute_window_bound.py (sha256 {script_hash[:16]}); python {dl.tool_versions()['python']}",
                   used_for=[f"done {today}: {what}", "planned: window decomposition bound of the driver; not yet used in a result"])
        dl.append_event(layer_id=lid, service="none (built locally, GPU)", what=what, file=str(path.relative_to(CASE)), sha256=e.sha256,
                        bytes=path.stat().st_size, **stamp, reconstructed=False, request={"stats": stats}, crs=str(crs),
                        script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
        print(f"registered {lid}, sha256 {e.sha256[:12]}", flush=True)
    m.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

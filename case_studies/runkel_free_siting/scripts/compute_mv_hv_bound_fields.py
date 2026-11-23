"""Collector and 110 kV tap bound fields on the GPU (paper math.md M7.6a and M6.3; plan 2026-09-23 sections 3.2, 3.3).

Two lower-bound fields for the bound-and-prune driver (``run_bound_and_prune.py``), every 5th cell, both float32 GPU bounds
that the driver turns into float64 lower bounds with ``d (1 - k_max eta)`` (math.md M9.1):

1. ``mv_lb_5m_lattice_eur.tif``: a lower bound of the trench weight of the collector at every root cell. Seven drains, one per
   turbine, over the trench-only weight ``floor((1 + omega) c)`` of the shared priced raster (system terms, junctions, bays and
   turbine no-transit are dropped: all are non-negative). The cell value is
   ``max(max_i d_i, MST / (2 (1 - 1/8)))`` over the metric closure of the seven turbines and the root
   (``pyorps.siting.lattice_bounds.steiner_lower_bound``). The trench set of any collector is connected and contains every
   terminal, so this bound is admissible. It is weaker than the plan's rho-hat-DW (127 drains), which is a later refinement:
   a looser bound keeps more positions and never changes the answer.
2. ``hv_field_taps_5m_lattice_eur.tif``: the 110 kV cable field of ``compute_hv_fields.py`` with the verified 110 kV towers
   added as seeds next to the three connection points, all with label 0 (tap works and bays are non-negative and dropped).
   Cable-only: the bound is for the all-cable scenario.

A terminal on a forbidden cell is reached at the allowed cells nearest to it (within 1.5 m of the nearest distance), label 0
(owner decision D-h). Temperature guard and 90 s pauses as in the other GPU scripts.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_mv_hv_bound_fields.py
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
SEARCH_RADIUS_M = 200
TIE_M = 1.5
COOL_S = int(os.environ.get("PYORPS_COOL_S", 90))     # 0 on a desktop card with its own cooling
NODATA = 3.4e38
FORBIDDEN = 65535


def nearest_allowed_cells(cost, row, col):
    """Cell indices (flat) of the allowed cells nearest to (row, col), within TIE_M of the nearest distance; gap in m."""
    import numpy as np

    h, w = cost.shape
    r0, r1, c0, c1 = max(0, row - SEARCH_RADIUS_M), min(h, row + SEARCH_RADIUS_M + 1), max(0, col - SEARCH_RADIUS_M), min(w, col + SEARCH_RADIUS_M + 1)
    if not (0 <= row < h and 0 <= col < w):
        return None
    yy, xx = np.nonzero(cost[r0:r1, c0:c1] != FORBIDDEN)
    if yy.size == 0:
        return None
    dist = np.hypot(yy + r0 - row, xx + c0 - col)
    keep = dist <= dist.min() + TIE_M
    return ((yy[keep] + r0) * w + (xx[keep] + c0)).astype(np.int64), float(dist.min())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--max-temp", type=float, default=85.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--skip-mv", action="store_true")
    ap.add_argument("--skip-hv", action="store_true")
    ap.add_argument("--register-only", action="store_true", help="register the existing output files, compute nothing")
    ap.add_argument("--ledger", default=str(CASE / "config" / "branch_constants_2026.yaml"),
                    help="branch constants: connection points are seeded with their bay cost, towers with the tap works")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from pyorps.siting.data_manifest import DataManifest, LayerEntry
    from pyorps.siting.lattice_bounds import float32_lower_bound, steiner_lower_bound
    from pyorps.siting.terminals import load_terminals
    from pyorps.utils.neighborhood import get_neighborhood_steps
    from pyorps.utils.sssp_gpu import GpuSsspSession

    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["cost_priced_1m", "hv_cost_priced_1m", "turbines", "terminals", "towers_110_verified"])
    with rasterio.open(m.file_of("cost_priced_1m")) as s:
        x0, y1, crs = int(s.bounds.left), int(s.bounds.top), s.crs
    steps = np.asarray(get_neighborhood_steps(args.k, directed=True), dtype=np.int8)
    tr = from_origin(x0, y1, 5.0, 5.0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell_of(x, y):
        return int(np.floor(y1 - y)), int(np.floor(x - x0))

    def write(path: Path, arr):
        profile = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32", crs=crs, transform=tr,
                       nodata=NODATA, tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3, zlevel=6)
        part = path.with_name(path.name + ".partial")
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(arr, 1)
        part.replace(path)

    stats: dict = {}
    written = []
    if args.register_only:
        args.skip_mv = args.skip_hv = True
        written = [("hv_field_taps_5m_lattice", out_dir / "hv_field_taps_5m_lattice_eur.tif", "110 kV cable field, connection points (with bay cost) and verified towers as seeds (GPU, float32 bound)")]
        stats = json.loads((out_dir / "mv_hv_bound_sidecar.json").read_text(encoding="utf-8")).get("stats", {})

    # ---------------------------------------------------------------- collector bound
    if not args.skip_mv:
        with rasterio.open(m.file_of("cost_priced_1m")) as s:
            c = s.read(1)
        omega = None
        sc = json.loads((OUT / "hv_cost_sidecar.json").read_text(encoding="utf-8"))

        def find(d, key):
            if isinstance(d, dict):
                if key in d and isinstance(d[key], (int, float)):
                    return d[key]
                for v in d.values():
                    r = find(v, key)
                    if r is not None:
                        return r
            return None

        omega = find(sc, "omega")
        if omega is None:
            import re
            mt = re.search(r"omega ([0-9.]+)", sc.get("note", ""))
            omega = float(mt.group(1)) if mt else None
        if omega is None:
            raise SystemExit("omega not found in hv_cost_sidecar.json")
        free = c != FORBIDDEN
        tw = np.full(c.shape, FORBIDDEN, dtype=np.uint16)
        tw[free] = np.minimum(np.floor((1.0 + omega) * c[free].astype(np.float64)), FORBIDDEN - 1).astype(np.uint16)
        del c
        c_min = float(tw[free].min())
        turbines = m.read_vector("turbines", verify=False)
        seeds = []
        gaps = []
        for g in turbines.geometry:
            row, col = cell_of(g.x, g.y)
            r = nearest_allowed_cells(tw, row, col)
            if r is None:
                raise SystemExit(f"turbine at {g.x:.1f},{g.y:.1f} has no allowed cell within {SEARCH_RADIUS_M} m")
            seeds.append(r[0])
            gaps.append(r[1])
        n_t = len(seeds)
        print(f"omega {omega}; trench weight min {c_min}; {n_t} turbines; gap to nearest allowed cell (m): {[round(x, 1) for x in gaps]}", flush=True)
        lat_shape = tw[::5, ::5].shape
        d_lat = np.empty((n_t,) + lat_shape, dtype=np.float32)
        tt_f32 = np.zeros((n_t, n_t), dtype=np.float64)
        d_max = 0.0
        with TempGuard(args.max_temp) as guard:
            sess = GpuSsspSession(tw, steps, arena_factor=1.0)
            print(f"GPU session holds {sess.device_bytes / 1e9:.2f} GB", flush=True)
            for i, idx in enumerate(seeds):
                f = sess.solve_seeded(idx, np.zeros(idx.size, dtype=np.float32))
                fin = np.isfinite(f)
                d_max = max(d_max, float(f[fin].max()))
                for j, jdx in enumerate(seeds):
                    tt_f32[i, j] = float(f[jdx].min())
                d_lat[i] = f.reshape(tw.shape)[::5, ::5]
                del f, fin
                print(f"  turbine {i} drained", flush=True)
                time.sleep(COOL_S)
            sess.close(free_pool=True)
        k_max = int(np.ceil(d_max / c_min)) + 2
        dist = np.stack([float32_lower_bound(d_lat[i], k_max) for i in range(n_t)])
        tt = float32_lower_bound(tt_f32, k_max)
        tt = np.minimum(tt, tt.T)
        np.fill_diagonal(tt, 0.0)
        lb = steiner_lower_bound(dist, tt)
        stats["mv"] = {"omega": omega, "turbines": n_t, "turbine_gap_to_allowed_m": gaps, "trench_weight_min": c_min,
                       "d_max_eur": d_max, "k_max": k_max, "eta": 3 * 2.0 ** -24, "slack_relative": k_max * 3 * 2.0 ** -24,
                       "terminal_distances_eur": tt.tolist(), "gpu_temp_max_c": guard.max_seen,
                       "lb_finite_share": float(np.isfinite(lb).mean()),
                       "lb_min": float(lb[np.isfinite(lb)].min()), "lb_median": float(np.median(lb[np.isfinite(lb)])),
                       "lb_far_share_of_steiner": float((lb > dist.max(axis=0) * (1 + 1e-9)).mean()),
                       "note": "stored value is already the float64 lower bound rounded toward -inf to float32; no further slack"}
        print(stats["mv"], flush=True)
        stored = np.where(np.isfinite(lb), (lb * (1.0 - 2.0 ** -23)).astype(np.float32), np.float32(NODATA)).astype(np.float32)
        p = out_dir / "mv_lb_5m_lattice_eur.tif"
        write(p, stored)
        written.append(("mv_lb_5m_lattice", p, "lower bound of the collector trench weight (7 turbine drains, Steiner/MST bound, slack already applied)"))
        del d_lat, dist, lb, stored, tw

    # ---------------------------------------------------------------- 110 kV field with taps
    if not args.skip_hv:
        with rasterio.open(m.file_of("hv_cost_priced_1m")) as s:
            hc = s.read(1)
        from pyorps.costmodel import CostCatalogue
        from pyorps.costmodel.branch_constants import BranchConstants

        ledger = BranchConstants.load(args.ledger, CostCatalogue.load(CASE / "config" / "cost_catalogue_2026.yaml"))
        seeds_all, labels_all = [], []
        n_pcc = n_tw = n_tw_moved = 0
        for t_ in load_terminals(m.file_of("terminals")):
            row, col = cell_of(t_.xy[0], t_.xy[1])
            r = nearest_allowed_cells(hc, row, col)
            if r is not None:
                seeds_all.append(r[0]); labels_all.append(np.full(r[0].size, ledger.pcc_bay_eur)); n_pcc += 1
        towers = m.read_vector("towers_110_verified", verify=False)
        for g in towers.geometry:
            row, col = cell_of(g.x, g.y)
            r = nearest_allowed_cells(hc, row, col)
            if r is not None:
                seeds_all.append(r[0]); labels_all.append(np.full(r[0].size, ledger.tap_works_eur)); n_tw += 1
                n_tw_moved += int(r[1] > 0.0)
        idx = np.concatenate(seeds_all)
        lab = np.concatenate(labels_all)
        o_ = np.lexsort((lab, idx))                                 # per cell the smallest label first
        idx, lab = idx[o_], lab[o_]
        keep_ = np.concatenate(([True], idx[1:] != idx[:-1]))
        idx, lab = idx[keep_], lab[keep_]
        c_min_hv = float(hc[hc != FORBIDDEN].min())
        print(f"110 kV taps: {n_pcc} connection points, {n_tw} verified towers ({n_tw_moved} on forbidden cells, moved to the nearest allowed cell), {idx.size} seed cells", flush=True)
        with TempGuard(args.max_temp) as guard:
            sess = GpuSsspSession(hc, steps, arena_factor=1.0)
            f = sess.solve_seeded(idx, lab.astype(np.float32))
            sess.close(free_pool=True)
        fin = np.isfinite(f)
        d_max_hv = float(f[fin].max())
        k_max_hv = int(np.ceil(d_max_hv / c_min_hv)) + 2
        lat = np.where(fin.reshape(hc.shape)[::5, ::5], f.reshape(hc.shape)[::5, ::5], np.float32(NODATA)).astype(np.float32)
        stats["hv_taps"] = {"connection_points": n_pcc, "towers": n_tw, "towers_moved_to_allowed": n_tw_moved, "seed_cells": int(idx.size),
                            "d_max_eur": d_max_hv, "min_step_cost": c_min_hv, "k_max_rigorous": k_max_hv,
                            "rounding_slack_relative": k_max_hv * 3 * 2.0 ** -24, "reached_share": float(fin.mean()),
                            "gpu_temp_max_c": guard.max_seen,
                            "seed_labels_eur": {"connection_point_bay": ledger.pcc_bay_eur, "tower_tap_works": ledger.tap_works_eur},
                            "note": "float32 GPU bound; multiply by 1 - rounding_slack_relative; cable only; connection points carry the bay cost, towers the tap works of the branch-constants ledger"}
        print(stats["hv_taps"], flush=True)
        p = out_dir / "hv_field_taps_5m_lattice_eur.tif"
        write(p, lat)
        written.append(("hv_field_taps_5m_lattice", p, "110 kV cable field with verified towers and connection points as seeds (GPU, float32 bound)"))
        del hc, f, fin, lat

    if not args.register_only:
      sc_path = out_dir / "mv_hv_bound_sidecar.json"
      if sc_path.exists():                                      # keep the entries of the part that was not recomputed
          merged = json.loads(sc_path.read_text(encoding="utf-8")).get("stats", {})
          merged.update(stats)
          stats = merged
      sc_path.write_text(json.dumps(
        {"stats": stats, "grid_epsg25832_origin": [x0, y1], "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()}},
        indent=1, default=str), encoding="utf-8")
    for _, p, _ in written:
        print(f"wrote {p.name} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
    if args.no_register:
        return 0
    today, script_hash = dt.date.today().isoformat(), dl.file_sha256(__file__)
    for lid, path, what in written:
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(id=lid, kind="other", format="raster", path=None, status="pending",
                                       source=f"built by compute_mv_hv_bound_fields.py: {what}",
                                       licence="derived work; follows the licences of its inputs")
        stamp = dl.now_stamps()
        e = m.register(lid, path, captured=today, crs=str(crs))
        m.annotate(lid, captured_at=stamp["time_local"], notes=what,
                   retrieved_with=f"compute_mv_hv_bound_fields.py (sha256 {script_hash[:16]}); python {dl.tool_versions()['python']}",
                   used_for=[f"done {today}: {what}", "planned: lower bound of the bound-and-prune driver; not yet used in a result"])
        dl.append_event(layer_id=lid, service="none (built locally, GPU)", what=what, file=str(path.relative_to(CASE)),
                        sha256=e.sha256, bytes=path.stat().st_size, **stamp, reconstructed=False, request={"stats": stats}, crs=str(crs),
                        script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
        print(f"registered {lid}, sha256 {e.sha256[:12]}", flush=True)
    m.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

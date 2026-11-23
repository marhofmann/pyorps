"""The construction-access field on the GPU (plan 2026-10-06-paper-run-pipeline.md, stage S3).

One many-seed search from every entry cell (existing public roads, label 0) over ``access_cost_1m.tif`` with the
kernel's per-step grade table (grade limit and cut-and-fill bands of ``config/access_constraints_2026.yaml``, DEM at
1 m) gives, for every cell, the cheapest heavy-transport road to it. Float32 on the GPU: the field is a BOUND, used only
to remove sites, with a stated rounding margin; the winner's road is re-solved in float64 on the CPU (owner decision
D-a, 2026-10-06). ``--check-cpu`` runs the exact CPU kernel on the same crop and reports the largest deviation.

Writes one small GeoTIFF (the field sampled on a 5 m lattice, float32, deflate) and a sidecar JSON; nothing else.
A temperature guard ends the process if the GPU reaches ``--max-temp``.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_access_field.py --box 441701 5585000 443701 5587000 --check-cpu --out-dir <scratch> --no-register
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_access_field.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
CONSTRAINTS = CASE / "config" / "access_constraints_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
OUT = CASE / "data" / "rasters"
SENTINEL_U32 = 4294967295
FIELD_ID = "access_field_5m_lattice"


class TempGuard:
    """Polls ``nvidia-smi``; ends the process if the GPU temperature reaches the limit."""

    def __init__(self, limit_c: float, every: float = 2.0):
        self.limit, self.every, self.max_seen = limit_c, every, 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> float | None:
        try:
            first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()      # nvidia-smi ignores the mask: ask for the physical card
            sel = ["-i", first] if first.isdigit() else []
            out = subprocess.run(["nvidia-smi", *sel, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            return float(out[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _loop(self):
        while not self._stop.wait(self.every):
            t = self._read()
            if t is None:
                continue
            self.max_seen = max(self.max_seen, t)
            if t >= self.limit:
                print(f"GPU TEMPERATURE GUARD: {t:.0f} C >= {self.limit:.0f} C; ending the process", flush=True)
                os._exit(3)

    def __enter__(self):
        self.max_seen = self._read() or 0.0
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=self.every + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--check-cpu", action="store_true", help="compare with the exact CPU kernel on the same crop (small crops only)")
    ap.add_argument("--arena", type=float, default=1.0, help="V5 arena factor, at least 1.0 (1.0 = 4 B/cell)")
    ap.add_argument("--max-temp", type=float, default=85.0)
    ap.add_argument("--k", type=int, default=3, help="neighbourhood k")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    from pyorps.core.objective import Objective
    from pyorps.costmodel.access_road import AccessConstraints
    from pyorps.costmodel.catalogue import CostCatalogue
    from pyorps.siting.data_manifest import DataManifest, LayerEntry
    from pyorps.utils.neighborhood import get_neighborhood_steps
    from pyorps.utils.sssp_gpu import GpuSsspSession

    t0 = time.time()
    acc = AccessConstraints.load(CONSTRAINTS, CostCatalogue.load(CATALOGUE))
    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["access_raster", "access_entry_1m", "dem"])
    with rasterio.open(m.file_of("access_raster")) as s:
        gx0, gy1, crs = int(s.bounds.left), int(s.bounds.top), s.crs
        gw, gh = s.width, s.height
    x0, y0, x1, y1 = args.box if args.box else (gx0, gy1 - gh, gx0 + gw, gy1)
    width, height = x1 - x0, y1 - y0
    win = Window(x0 - gx0, gy1 - y1, width, height)
    with rasterio.open(m.file_of("access_raster")) as s:
        cost = s.read(1, window=win)
    with rasterio.open(m.file_of("access_entry_1m")) as s:
        entry = s.read(1, window=win)
    with rasterio.open(m.file_of("dem")) as s:
        dem = s.read(1, window=win, out_dtype="float32")
    if not np.isfinite(dem).all():
        raise SystemExit("the DEM has a missing cell inside the box")
    steps = np.asarray(get_neighborhood_steps(args.k, directed=True), dtype=np.int8)
    luts = Objective({"cost": 1.0}, acc.gradient_options()).build_gradient_luts(steps, 1.0)
    seeds = np.flatnonzero(entry.ravel())
    if seeds.size == 0:
        raise SystemExit("no entry cell in the box")
    print(f"box {x0},{y0}..{x1},{y1}: {width} x {height} = {width * height} cells, {seeds.size} entry cells, "
          f"{len(steps)} steps, loaded in {time.time() - t0:.0f} s", flush=True)

    with TempGuard(args.max_temp) as guard:
        sess = GpuSsspSession(cost, steps, dem=dem, gradient_luts=luts, arena_factor=args.arena)
        print(f"GPU session holds {sess.device_bytes / 1e9:.2f} GB", flush=True)
        t1 = time.time()
        field = sess.solve_seeded(seeds).reshape(cost.shape)
        solve_s = time.time() - t1
        sess.close(free_pool=True)
    reach = np.isfinite(field)
    passable = cost != 65535
    d_max = float(field[reach].max())
    nonentry = cost[passable & (entry == 0)]
    c_min = float(nonentry.min())
    k_max = int(np.ceil(d_max / (c_min / 2.0))) + 2
    # Per-step relative error eta = 3 u, u = 2^-24: the float32 cost factor (<= 0.44 u), the product with the integer
    # cell sum (<= u) and the addition (<= u) give 2.44 u <= 3 u. A path of k steps then has d_f in d (1 +- k eta), so
    # d_f (1 - k_max eta) is a lower bound (math.md, M-9.1). The first version used eta = u and was too tight.
    slack = k_max * 3.0 * 2.0 ** -24
    stats = {"cells": int(cost.size), "entry_cells": int(seeds.size), "passable_share": float(passable.mean()),
             "reached_share_of_passable": float((reach & passable).sum() / passable.sum()),
             "unreached_passable_cells": int((passable & ~reach).sum()),
             "dist_max_eur": d_max, "min_nonentry_cost": c_min, "k_max_rigorous": k_max,
             "rounding_slack_relative": slack, "gpu_solve_s_not_a_performance_figure": round(solve_s, 1),
             "gpu_temp_max_c": guard.max_seen}
    print(stats, flush=True)

    if args.check_cpu:
        from pyorps.utils import _dijkstra as dj

        solver = dj.make_multi_source_solver(np.ascontiguousarray(cost, dtype=np.uint16), steps, dem=dem,
                                             gradient_luts=luts, lean=True)
        solver.solve_stream(seeds.astype(np.int64), np.zeros(seeds.size))
        exact = solver.dist_array().reshape(cost.shape)
        fin = np.isfinite(exact)
        same_reach = bool(np.array_equal(fin, reach))
        rel = np.abs(field[fin] - exact[fin]) / np.maximum(exact[fin], 1e-9)
        lower = field[fin] * (1.0 - slack)
        stats.update(cpu_check={"same_reachable_set": same_reach, "max_rel_deviation": float(rel.max()),
                                "lower_bound_violations": int((lower > exact[fin] * (1 + 1e-12)).sum())})
        print("CPU check:", stats["cpu_check"], flush=True)

    # feasible pad anchors of the 1 m site field that the road reaches
    site = m.file_of("site_field_envelope_cents")
    with rasterio.open(site) as s:
        sx0, sy1, sw, sh = int(s.bounds.left), int(s.bounds.top), s.width, s.height
        ix0, ix1, iy0, iy1 = max(x0, sx0), min(x1, sx0 + sw), max(y0, sy1 - sh), min(y1, sy1)
        if ix0 < ix1 and iy0 < iy1:
            sf = s.read(1, window=Window(ix0 - sx0, sy1 - iy1, ix1 - ix0, iy1 - iy0))
            sub = field[(y1 - iy1):(y1 - iy0), (ix0 - x0):(ix1 - x0)]
            feas = sf != SENTINEL_U32
            if feas.any():
                d = sub[feas]
                ok = np.isfinite(d)
                stats["site_anchors"] = {"feasible": int(feas.sum()), "reached": int(ok.sum()),
                                         "reached_share": float(ok.mean()),
                                         "access_eur_min": float(d[ok].min()) if ok.any() else None,
                                         "access_eur_median": float(np.median(d[ok])) if ok.any() else None,
                                         "access_eur_p95": float(np.percentile(d[ok], 95)) if ok.any() else None,
                                         "note": "stand-in prices and assumption-tagged constraints: not quotable"}
                print("anchors:", stats["site_anchors"], flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat = np.where(np.isfinite(field[::5, ::5]), field[::5, ::5], np.float32(3.4e38)).astype(np.float32)
    path = out_dir / "access_field_5m_lattice_eur.tif"
    profile = dict(driver="GTiff", width=lat.shape[1], height=lat.shape[0], count=1, dtype="float32", crs=crs,
                   transform=from_origin(x0, y1, 5.0, 5.0), nodata=3.4e38, tiled=True, blockxsize=256,
                   blockysize=256, compress="deflate", predictor=3, zlevel=6)
    part = path.with_name(path.name + ".partial")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(lat, 1)
    part.replace(path)
    note = ("float32 GPU field (a BOUND: multiply by 1 - rounding_slack_relative for a lower bound), sampled every 5th "
            "cell of the 1 m grid from its origin; the 5 m pad convention is still open")
    (out_dir / "access_field_sidecar.json").write_text(json.dumps(
        {"constraints": acc.record(), "stats": stats, "grid_epsg25832": [x0, y0, x1, y1], "note": note,
         "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()}}, indent=1, default=str), encoding="utf-8")
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.no_register:
        print("not registered (--no-register)")
        return 0
    today, script_hash, stamp = dt.date.today().isoformat(), dl.file_sha256(__file__), dl.now_stamps()
    if FIELD_ID not in m.layers:
        m.layers[FIELD_ID] = LayerEntry(id=FIELD_ID, kind="access_raster", format="raster", path=None, status="pending",
                                        source="built by compute_access_field.py (GPU many-seed field)",
                                        licence="derived work; follows the licences of its inputs")
    e = m.register(FIELD_ID, path, captured=today, crs=str(crs))
    m.annotate(FIELD_ID, captured_at=stamp["time_local"], notes=note,
               retrieved_with=f"compute_access_field.py (sha256 {script_hash[:16]}), GpuSsspSession.solve_seeded; "
                              f"python {dl.tool_versions()['python']}",
               used_for=[f"done {today}: access field on a 5 m lattice", "planned: stage S3 lower bound; not yet used in a result"])
    dl.append_event(layer_id=FIELD_ID, service="none (built locally)", what="built raster: access field (GPU, float32 bound)",
                    file=str(path.relative_to(CASE)), sha256=e.sha256, bytes=path.stat().st_size, **stamp,
                    reconstructed=False, request={"stats": stats, "grid_epsg25832": [x0, y0, x1, y1]}, crs=str(crs),
                    script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
    m.save()
    print(f"registered {FIELD_ID}, sha256 {e.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

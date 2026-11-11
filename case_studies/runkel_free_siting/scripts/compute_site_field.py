"""Exact site-cost field of the switchyard pad over a site raster (plan Phase D3).

For every anchor cell the cheapest rotation of the 60 x 80 m pad (36 rotations of 5 degrees over 180) is found by the
integer-exact FFT of ``pyorps.siting.site_field`` (base-256 limb layers, every convolved value asserted within 0.05 of an integer).
A placement is infeasible when the pad leaves the grid or touches a forbidden (65535) pixel. Prices are the raster's EUR/m2 times
100 (integer cents). Before anything is registered, a random sample of anchors is re-priced by direct pixel sums (``site_values``,
independent of the FFT) and must agree to the cent at the winning rotation, and no other rotation may be cheaper at a sampled anchor.

Outputs, on the site raster's grid: ``site_field_<window>_cents.tif`` (uint32 cents, 4294967295 = infeasible) and
``site_field_<window>_rotation.tif`` (uint8 index into the rotations, 255 = none), registered with their input hash.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compute_site_field.py --window farm
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
OUT = CASE / "data" / "rasters"
FORBIDDEN = 65535
SENTINEL = 4294967295
PAD_LENGTH_M, PAD_WIDTH_M = 80.0, 60.0         # 60 x 80 m (x by y), the TopoMILP template; to be sourced (plan tau_UW)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", choices=("farm", "envelope"), default="farm")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check", type=int, default=400, help="anchors re-priced directly")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio

    from pyorps.siting.data_manifest import DataManifest, LayerEntry
    from pyorps.siting.site_field import INFEASIBLE, site_field, site_values

    t0 = time.time()
    m = DataManifest.load(MANIFEST)
    src_id = f"site_cost_{args.window}"
    in_hash = m.verify([src_id])[src_id]
    with rasterio.open(m.file_of(src_id)) as src:
        eur = src.read(1)
        transform, crs = src.transform, src.crs
    forb = eur == FORBIDDEN
    cents = np.where(forb, 0, eur.astype(np.int64) * 100)
    rows, cols = eur.shape
    print(f"{src_id}: {rows} x {cols} = {eur.size} cells, {forb.mean():.1%} forbidden, pad {PAD_WIDTH_M:g} x {PAD_LENGTH_M:g} m, "
          f"36 rotations, {args.workers} FFT workers", flush=True)
    field = site_field(cents, forb, length_m=PAD_LENGTH_M, width_m=PAD_WIDTH_M, cell_m=1.0,
                       block=args.block, workers=args.workers)
    feas = field.feasible
    print(f"field computed ({time.time() - t0:.0f} s): {int(feas.sum())} feasible anchors ({feas.mean():.2%})", flush=True)

    # ---------------- independent re-pricing at sampled anchors
    rng = np.random.default_rng(20261005)
    fr, fc = np.nonzero(feas)
    n_ok = min(args.check, fr.size)
    pick = rng.choice(fr.size, size=n_ok, replace=False) if n_ok else np.array([], int)
    inf_r, inf_c = np.nonzero(~feas)
    n_inf = min(args.check // 4, inf_r.size)
    pick_i = rng.choice(inf_r.size, size=n_inf, replace=False) if n_inf else np.array([], int)
    bad = 0
    rot_conflicts = 0
    thetas = field.thetas
    for idx in pick:
        r, c = int(fr[idx]), int(fc[idx])
        k = int(field.rotation[r, c])
        direct, _ = site_values(cents, forb, [(r, c)], float(thetas[k]), length_m=PAD_LENGTH_M,
                                width_m=PAD_WIDTH_M, cell_m=1.0)
        if int(direct[0]) != int(field.cents[r, c]):
            bad += 1
        for j, th in enumerate(thetas):                      # no other rotation may be cheaper
            other, _ = site_values(cents, forb, [(r, c)], float(th), length_m=PAD_LENGTH_M,
                                   width_m=PAD_WIDTH_M, cell_m=1.0)
            if int(other[0]) != INFEASIBLE and int(other[0]) < int(field.cents[r, c]):
                rot_conflicts += 1
    wrong_inf = 0
    for idx in pick_i:
        r, c = int(inf_r[idx]), int(inf_c[idx])
        for th in thetas:
            v, _ = site_values(cents, forb, [(r, c)], float(th), length_m=PAD_LENGTH_M,
                               width_m=PAD_WIDTH_M, cell_m=1.0)
            if int(v[0]) != INFEASIBLE:
                wrong_inf += 1
                break
    print(f"independent check: {n_ok} feasible anchors re-priced, {bad} cost mismatches, {rot_conflicts} cheaper rotations; "
          f"{n_inf} infeasible anchors tested, {wrong_inf} wrongly infeasible", flush=True)
    if bad or rot_conflicts or wrong_inf:
        raise SystemExit("the FFT field disagrees with the direct pixel sums; nothing registered")

    # ---------------- summary
    eur_field = field.cents[feas] / 100.0
    best = np.argsort(field.cents[feas])[:5]
    rr, cc = np.nonzero(feas)
    summary = dict(anchors=int(eur.size), feasible=int(feas.sum()), feasible_share=round(float(feas.mean()), 4),
                   cheapest_eur=float(eur_field.min()), median_eur=float(np.median(eur_field)),
                   p95_eur=float(np.percentile(eur_field, 95)), max_eur=float(eur_field.max()))
    cheapest = []
    for b in best:
        r, c = int(rr[b]), int(cc[b])
        x, y = transform * (c + 0.5, r + 0.5)
        cheapest.append(dict(easting=round(x, 1), northing=round(y, 1), eur=field.cents[r, c] / 100.0,
                             rotation_deg=float(thetas[field.rotation[r, c]])))
    print("summary:", summary, flush=True)
    print("five cheapest anchors:", cheapest, flush=True)

    # ---------------- write
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(path, arr, dtype):
        prof = dict(driver="GTiff", width=cols, height=rows, count=1, dtype=dtype, crs=crs, transform=transform,
                    tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2, zlevel=6,
                    BIGTIFF="IF_SAFER")
        part = path.with_name(path.name + ".partial")
        with rasterio.open(part, "w", **prof) as dst:
            dst.write(arr, 1)
        part.replace(path)

    cents_u32 = np.where(feas, field.cents, SENTINEL).astype(np.uint32)
    rot_u8 = np.where(feas, field.rotation, 255).astype(np.uint8)
    p_cents = out_dir / f"site_field_{args.window}_cents.tif"
    p_rot = out_dir / f"site_field_{args.window}_rotation.tif"
    write(p_cents, cents_u32, "uint32")
    write(p_rot, rot_u8, "uint8")
    if args.no_register:
        return 0
    today = dt.date.today().isoformat()
    script_hash = dl.file_sha256(__file__)
    for lid, path, what in ((f"site_field_{args.window}_cents", p_cents, "pad cost in cents at the best rotation, 4294967295 = infeasible"),
                             (f"site_field_{args.window}_rotation", p_rot, "index of the best rotation (5 degree steps), 255 = none")):
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(id=lid, kind="site_raster", format="raster", path=None, status="pending",
                                       source=f"built by compute_site_field.py from {src_id}: {what}",
                                       licence="derived work; follows the licences of its inputs")
        stamp = dl.now_stamps()
        e = m.register(lid, path, captured=today, crs=str(crs))
        m.annotate(lid, captured_at=stamp["time_local"],
                   notes=f"exact FFT site field of a {PAD_WIDTH_M:g} x {PAD_LENGTH_M:g} m pad, 36 rotations; checked against direct pixel sums "
                         f"at {n_ok} feasible and {n_inf} infeasible anchors (0 disagreements); {summary['feasible_share']:.2%} feasible",
                   retrieved_with=f"compute_site_field.py (sha256 {script_hash[:16]}), pyorps.siting.site_field; python {dl.tool_versions()['python']}",
                   used_for=[f"done {today}: built and checked", "planned: the site term of the total cost field; not yet used in a result"])
        dl.append_event(layer_id=lid, service="none (built locally)", what=f"built: {what}", file=str(path.relative_to(CASE)),
                        sha256=e.sha256, bytes=path.stat().st_size, **stamp, reconstructed=False,
                        request={"input": src_id, "input_sha256": in_hash, "pad_m": [PAD_WIDTH_M, PAD_LENGTH_M], "rotations": 36,
                                 "summary": summary, "cheapest": cheapest, "check_anchors": [n_ok, n_inf]},
                        crs=str(crs), script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
        print(f"registered {lid}: {path.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]}")
    m.save()
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

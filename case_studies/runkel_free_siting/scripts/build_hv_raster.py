"""Build the 1 m weight raster of the 110 kV cable leg from the SHARED trench raster (plan 2026-09-23 section 3.3).

Digging cost is the same for the collector voltages and 110 kV (owner, 2026-10-06), so the leg is routed on the same
priced trench raster, ``cost_priced_1m.tif``. The weight of a metre in a cell is ``(1 + omega) * c(cell) + rho``
(``pyorps.costmodel.hv_leg``): omega the capitalised O&M multiple, rho the cell-independent part (3 cables of the
cheapest section plus capitalised losses from the whole-farm power-category table). Forbidden cells stay forbidden,
no cell is below its trench rate (checked block by block). Output: ``hv_cost_priced_1m.tif`` (uint16 EUR/m) and a
sidecar JSON; nothing else.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_hv_raster.py --box 441701 5585000 443701 5587000 --out-dir <scratch> --no-register
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_hv_raster.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
CONSTRAINTS = CASE / "config" / "hv_leg_constraints_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
INPUTS = CASE / "config" / "collector_inputs_2026.yaml"
LOSS_TABLE = CASE / "data" / "loss_categories.npz"
OUT = CASE / "data" / "rasters"
BLOCK_ROWS = 1024
LID = "hv_cost_priced_1m"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    from rasterio.windows import Window

    from pyorps.costmodel import CostCatalogue, LossCategories
    from pyorps.costmodel.collector_inputs import CollectorInputs
    from pyorps.costmodel.hv_leg import FORBIDDEN, HvLegConstraints
    from pyorps.siting.data_manifest import DataManifest, LayerEntry

    t0 = time.time()
    hv = HvLegConstraints.load(CONSTRAINTS, CostCatalogue.load(CATALOGUE), CollectorInputs.load(INPUTS),
                               LossCategories.load(LOSS_TABLE))
    m = DataManifest.load(MANIFEST)
    hashes = m.verify([hv.trench_raster])
    with rasterio.open(m.file_of(hv.trench_raster)) as src:
        gx0, gy1, gw, gh, crs = int(src.bounds.left), int(src.bounds.top), src.width, src.height, src.crs
    x0, y0, x1, y1 = args.box if args.box else (gx0, gy1 - gh, gx0 + gw, gy1)
    width, height = x1 - x0, y1 - y0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hv_cost_priced_1m.tif"
    part = path.with_name(path.name + ".partial")
    print(f"grid {x0},{y0}..{x1},{y1}: {width} x {height} cells; omega {hv.omega:.4f}, section {hv.section_mm2} mm2, "
          f"rho {hv.rho_eur_m:.1f} EUR/m, {len(hv.assumptions())} assumption(s)", flush=True)
    lo = hi = None
    n_free = 0
    profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="uint16", crs=crs,
                   transform=rasterio.transform.from_origin(x0, y1, 1.0, 1.0), tiled=True, blockxsize=512,
                   blockysize=512, compress="deflate", predictor=2, zlevel=6, BIGTIFF="IF_SAFER")
    with rasterio.open(m.file_of(hv.trench_raster)) as src, rasterio.open(part, "w", **profile) as dst:
        for r0 in range(0, height, BLOCK_ROWS):
            h = min(BLOCK_ROWS, height - r0)
            win = Window(x0 - gx0, gy1 - y1 + r0, width, h)
            t = src.read(1, window=win)
            w = hv.cost_block(t)
            free = t != FORBIDDEN
            if not np.array_equal(w == FORBIDDEN, ~free) or (w[free] <= t[free]).any():
                raise SystemExit(f"invariant violated in rows {r0}..{r0 + h}: forbidden cells or a weight not above the trench rate")
            if free.any():
                lo = int(w[free].min()) if lo is None else min(lo, int(w[free].min()))
                hi = int(w[free].max()) if hi is None else max(hi, int(w[free].max()))
                n_free += int(free.sum())
            dst.write(w, 1, window=Window(0, r0, width, h))
    part.replace(path)
    note = (f"built from {hv.trench_raster}: (1 + omega) c + rho, omega {hv.omega:.4f}, rho {hv.rho_eur_m:.1f} EUR/m "
            f"({hv.section_mm2} mm2, 3 cores + capitalised losses from W(all)); forbidden cells equal the trench "
            f"raster's; no weight below its trench rate; free-cell weight {lo}..{hi} EUR/m over {n_free} cells; "
            f"{hv.saturated_cells} cells capped at 65534 (already capped in the trench raster); priced at zero: {', '.join(hv.unpriced)}")
    print(note, f"({time.time() - t0:.0f} s, not a performance figure)", flush=True)
    (out_dir / "hv_cost_sidecar.json").write_text(json.dumps(
        {"constraints": hv.record(), "grid_epsg25832": [x0, y0, x1, y1], "note": note,
         "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()},
         "limits": ["one circuit, no trench sharing", "cable field only: overhead and mixed legs are a separate bound",
                    "joints, cross-bonding, easement, fibre and dielectric loss priced at zero (no catalogue keys)",
                    "cells whose weight exceeds 65534 EUR/m are capped there (protected-area pricing in the trench raster is already capped)", "the HDD multiplier of the consent model acts on the trench part only", "the section's ampacity is assumed"]},
        indent=1, default=str), encoding="utf-8")
    if args.no_register:
        print("not registered (--no-register)")
        return 0

    today, script_hash, stamp = dt.date.today().isoformat(), dl.file_sha256(__file__), dl.now_stamps()
    if LID not in m.layers:
        m.layers[LID] = LayerEntry(id=LID, kind="other", format="raster", path=None, status="pending",
                                   source="built by build_hv_raster.py from cost_priced_1m",
                                   licence="derived work; follows the licences of its inputs (ALKIS, HLNUG, HMLU)")
    e = m.register(LID, path, captured=today, crs=str(crs))
    m.annotate(LID, captured_at=stamp["time_local"], notes=note,
               retrieved_with=f"build_hv_raster.py (sha256 {script_hash[:16]}), {CONSTRAINTS.name} (sha256 {hv.sha256[:16]}); "
                              f"python {dl.tool_versions()['python']}",
               used_for=[f"done {today}: built on the DEM grid {width} x {height}, 1 m (replaces a withdrawn first version "
                         "priced from catalogue civil costs)", "planned: 110 kV leg field, stage S2; not yet used in a result"])
    dl.append_event(layer_id=LID, service="none (built locally)", what="built raster: 110 kV weight on the shared trench raster",
                    file=str(path.relative_to(CASE)), sha256=e.sha256, bytes=path.stat().st_size, **stamp, reconstructed=False,
                    request={"grid_epsg25832": [x0, y0, x1, y1], "constraints_sha256": hv.sha256, "check": note},
                    crs=str(crs), script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
    m.save()
    print(f"registered {LID}: {path.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

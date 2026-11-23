"""Build the priced cost map: closed land-use cost x protected-area multipliers (owner, 2026-10-05).

Protected areas are PRICED, not forbidden. This takes ``cost_base_1m_closed`` (forbidden land-use
classes stay forbidden, closed burn) and ``protection_flags_1m`` and multiplies the cost of every cell
inside a protected area by the class multiplier of ``consent_classes_2026.yaml`` (HDD price; water zones
by zone). The result is ``cost_priced_1m.tif`` (uint16, 65535 forbidden, priced cells capped at 65534).

Invariants checked over the whole raster before anything is registered:
  * a cell is forbidden in the result exactly when it is forbidden in the closed cost map;
  * no cell is cheaper than in the closed cost map;
  * a cell outside every protected area and water zone is unchanged.

The one-off permit costs and procedure times are not in the raster; they belong to the consent lattice
(``ConsentTable.one_off`` and ``delay_months``).

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_priced_cost.py
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
CONSENT = CASE / "config" / "consent_classes_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
OUT = CASE / "data" / "rasters"
BLOCK_ROWS = 512


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio

    from pyorps.costmodel import CostCatalogue
    from pyorps.costmodel.consent import FORBIDDEN, ConsentTable
    from pyorps.siting.data_manifest import DataManifest, LayerEntry

    t0 = time.time()
    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["cost_base_1m_closed", "protection_flags_1m"])
    table = ConsentTable.load(CONSENT, CostCatalogue.load(CATALOGUE))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cost_priced_1m.tif"
    part = out.with_name(out.name + ".partial")
    changed = forb_same = 0
    n_cells = 0
    with rasterio.open(m.file_of("cost_base_1m_closed")) as base, \
            rasterio.open(m.file_of("protection_flags_1m")) as flg:
        if (base.width, base.height, base.transform) != (flg.width, flg.height, flg.transform):
            raise SystemExit("the cost map and the flag raster are not on the same grid")
        profile = base.profile.copy()
        profile.update(driver="GTiff", dtype="uint16", count=1, tiled=True, blockxsize=512,
                       blockysize=512, compress="deflate", predictor=2, zlevel=6, BIGTIFF="IF_SAFER")
        profile.pop("nodata", None)
        with rasterio.open(part, "w", **profile) as dst:
            for r0 in range(0, base.height, BLOCK_ROWS):
                h = min(BLOCK_ROWS, base.height - r0)
                win = rasterio.windows.Window(0, r0, base.width, h)
                b, f = base.read(1, window=win), flg.read(1, window=win)
                p = table.apply(b, f)
                # invariants, block by block
                if not np.array_equal(p == FORBIDDEN, b == FORBIDDEN):
                    raise SystemExit(f"forbidden cells differ in rows {r0}..{r0 + h}")
                if (p < b).any():
                    raise SystemExit(f"a priced cell is cheaper than its base in rows {r0}..{r0 + h}")
                free_flag = f == 0
                if (p[free_flag] != b[free_flag]).any():
                    raise SystemExit(f"a cell outside every protected area changed in rows {r0}..{r0 + h}")
                changed += int((p != b).sum())
                forb_same += int((b == FORBIDDEN).sum())
                n_cells += b.size
                dst.write(p, 1, window=win)
    part.replace(out)
    print(f"priced cost map written ({time.time() - t0:.0f} s): {n_cells} cells, {changed} priced "
          f"({changed / n_cells:.1%}), {forb_same} forbidden unchanged", flush=True)
    if args.no_register:
        return 0

    today = dt.date.today().isoformat()
    script_hash = dl.file_sha256(__file__)
    lid = "cost_priced_1m"
    note = (f"every protected-area cell priced by the class multiplier (consent_classes_2026.yaml, sha256 "
            f"{table.sha256[:16]}; {len(table.assumptions())} stand-in values); forbidden cells unchanged; "
            f"{changed} of {n_cells} cells changed; one-off permit costs and procedure times are NOT in this "
            "raster (consent lattice)")
    if lid not in m.layers:
        m.layers[lid] = LayerEntry(
            id=lid, kind="other", format="raster", path=None, status="pending",
            source="built by build_priced_cost.py from cost_base_1m_closed and protection_flags_1m",
            licence="derived work; follows the licences of its inputs (ALKIS, HLNUG, HMLU)")
    stamp = dl.now_stamps()
    e = m.register(lid, out, captured=today, crs=str(rasterio.open(out).crs))
    m.annotate(lid, captured_at=stamp["time_local"], notes=note,
               retrieved_with=f"build_priced_cost.py (sha256 {script_hash[:16]}), pyorps.costmodel.consent, "
                              f"numpy/rasterio in {BLOCK_ROWS}-row blocks; python {dl.tool_versions()['python']}",
               used_for=[f"done {today}: built; invariants checked over the whole raster",
                         "planned: the routing cost surface for the cost fields; not yet used in a result"])
    dl.append_event(layer_id=lid, service="none (built locally)", what="built raster: priced cost map",
                    file=str(out.relative_to(CASE)), sha256=e.sha256, bytes=out.stat().st_size, **stamp,
                    reconstructed=False,
                    request={"inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()},
                             "consent_table_sha256": table.sha256, "cells": n_cells,
                             "cells_priced": changed, "assumptions": table.assumptions()},
                    crs=e.crs, script=Path(__file__).name, script_sha256=script_hash,
                    licence_text=e.licence)
    m.save()
    print(f"registered {lid}: {out.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

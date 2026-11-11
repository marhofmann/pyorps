"""Build the 1 m construction-access rasters on the DEM grid (plan 2026-10-06-paper-run-pipeline.md, step 1).

Outputs (uint16 / uint8, deflate, same grid as ``cost_priced_1m.tif``):

* ``access_cost_1m.tif``   EUR per metre of heavy-transport road centre line; 65535 forbidden. Land-use class
                           price (catalogue prices, ``config/access_constraints_2026.yaml``); forbidden by
                           class, by ground slope at least ``cross_slope_max_percent``, or where no free disc
                           of the road width fits. Existing public roads cost ``public_road_eur_per_m``.
                           The grade ALONG the route (limit and cut-and-fill bands) is not in the raster:
                           it is the kernel's per-step table, ``AccessConstraints.gradient_options()``.
* ``access_entry_1m.tif``  1 on existing public-road cells: the entry cells (seeds) of the access field.

Closed burn as in ``build_cost_raster.py``: every cell touched by a forbidden class polygon is forbidden, except
cells an existing public road covers (a bridge over a stream stays passable). The check is independent of the
rasterizer and refuses the build on a miss. Limits written into the sidecar: protected areas are not priced for
the road in this version (the cable raster prices them); the curve radius is not modelled.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_access_raster.py --box 431373 5582217 433373 5584217 --out-dir <scratch> --no-register
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_access_raster.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_recipe as rec  # noqa: E402
import download_log as dl  # noqa: E402

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
CONSTRAINTS = CASE / "config" / "access_constraints_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
OUT = CASE / "data" / "rasters"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                    help="build only this box (whole metres, EPSG:25832); default the DEM grid")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true", help="do not touch the manifest or the log")
    ap.add_argument("--block-rows", type=int, default=1024)
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio import features
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    from shapely.geometry import box

    from pyorps import CostAssumptions, GeoRasterizer, initialize_geo_dataset
    from pyorps.costmodel.access_road import FORBIDDEN, AccessConstraints, block_cost, class_cost
    from pyorps.costmodel.catalogue import CostCatalogue
    from pyorps.siting.data_manifest import DataManifest, LayerEntry

    t0 = time.time()
    acc = AccessConstraints.load(CONSTRAINTS, CostCatalogue.load(CATALOGUE))
    table = acc.class_table(rec.LAND_USE_COSTS)
    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["alkis_land_use", "dem"])
    with rasterio.open(m.file_of("dem")) as dem:
        gx0, gy0, gx1, gy1 = (int(round(v)) for v in dem.bounds)
        crs = dem.crs
    x0, y0, x1, y1 = args.box if args.box else (gx0, gy0, gx1, gy1)
    if not (gx0 <= x0 < x1 <= gx1 and gy0 <= y0 < y1 <= gy1):
        raise SystemExit("the box must lie inside the DEM grid")
    width, height = x1 - x0, y1 - y0
    transform = from_origin(x0, y1, 1.0, 1.0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"grid {x0},{y0}..{x1},{y1}: {width} x {height} = {width * height} cells; "
          f"{len(acc.assumptions())} assumption(s) in the constraints", flush=True)

    # ---- class costs, closed burn
    ds = initialize_geo_dataset(str(m.file_of("alkis_land_use")))
    ds.load_data()
    ras = GeoRasterizer(ds, CostAssumptions({("nutzart", "bez"): table}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ras.rasterize(bounding_box=box(x0, y0, x1, y1), preprocessing_function=rec.street_classes,
                      resolution_in_m=1.0, all_touched=True)
    for w in caught:
        print("pyorps warning:", str(w.message)[:600].replace("\n", " "), flush=True)
    cls = np.asarray(ras.raster).astype(np.uint16)
    if cls.shape != (height, width) or any(
            abs(a - b) > 1e-6 for a, b in zip(tuple(ras.transform)[:6], tuple(transform)[:6])):
        raise SystemExit(f"the rasterizer's grid {cls.shape} is not the box grid ({height}, {width}); nothing written")

    frame = ds.data.copy()
    forb = pd.Series([class_cost(table, n, b) == FORBIDDEN for n, b in zip(frame["nutzart"], frame["bez"])],
                     index=frame.index)
    shapes = [(g, 1) for g in frame.loc[forb, "geometry"] if g is not None and not g.is_empty]
    touched = features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=0,
                                 dtype="uint8", all_touched=True).astype(bool)
    public = cls == acc.public_eur_m
    missed = int((touched & (cls != FORBIDDEN) & ~public).sum())
    bridged = int((touched & public).sum())
    print(f"closure check: {int(touched.sum())} cells touched by {len(shapes)} forbidden polygons; "
          f"not forbidden and not public road: {missed}; kept as public road (bridges, crossings): {bridged}",
          flush=True)
    if missed:
        raise SystemExit(f"{missed} cells touched by a forbidden polygon are not forbidden; nothing written")
    del touched, public

    # ---- slope and clearance, in row blocks with a halo
    halo = acc.half_window + acc.radius_cells()
    cost = np.empty((height, width), dtype=np.uint16)
    entry = np.empty((height, width), dtype=np.uint8)
    col_off, row_off = x0 - gx0, gy1 - y1
    with rasterio.open(m.file_of("dem")) as dem:
        for r0 in range(0, height, args.block_rows):
            r1 = min(height, r0 + args.block_rows)
            a, b = max(0, r0 - halo), min(height, r1 + halo)
            z = dem.read(1, window=Window(col_off, row_off + a, width, b - a)).astype(np.float32)
            if not np.isfinite(z).all():
                raise SystemExit("the DEM has a missing cell inside the box; nothing written")
            c_, e_ = block_cost(cls[a:b], z, acc, top=r0 - a, bottom=b - r1)
            cost[r0:r1], entry[r0:r1] = c_, e_
    share = float((cost == FORBIDDEN).mean())
    print(f"access cost built ({time.time() - t0:.0f} s, not a performance figure): forbidden {share:.2%}, "
          f"entry cells {int(entry.sum())}", flush=True)

    def write_raster(path: Path, arr, dtype):
        profile = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype=dtype, crs=crs,
                       transform=transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                       predictor=2, zlevel=6, BIGTIFF="IF_SAFER")
        part = path.with_name(path.name + ".partial")
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(arr, 1)
        part.replace(path)

    cost_path, entry_path = out_dir / "access_cost_1m.tif", out_dir / "access_entry_1m.tif"
    write_raster(cost_path, cost, "uint16")
    write_raster(entry_path, entry, "uint8")
    free = cost[cost != FORBIDDEN]
    note = (f"closure check passed ({bridged} cells kept as public road); forbidden {share:.2%}; "
            f"entry cells {int(entry.sum())}; free-cell cost min {int(free.min()) if free.size else None}, "
            f"max {int(free.max()) if free.size else None}")
    sidecar = {"constraints": acc.record(), "grid_epsg25832": [x0, y0, x1, y1], "cells": width * height,
               "forbidden_share": share, "entry_cells": int(entry.sum()), "closure_note": note,
               "limits": ["protected areas are not priced for the road (the cable raster prices them)",
                          "curve radius is not modelled", "the slope rule is a cell rule, not along the route"],
               "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()}}
    (out_dir / "access_cost_sidecar.json").write_text(json.dumps(sidecar, indent=1, default=str), encoding="utf-8")

    if args.no_register:
        print("not registered (--no-register)")
        return 0

    today = dt.date.today().isoformat()
    script_hash = dl.file_sha256(__file__)
    inputs = {k: v[:16] for k, v in hashes.items()}
    built = [("access_raster", cost_path, "uint16", "EUR per metre of heavy-transport access road, 65535 forbidden"),
             ("access_entry_1m", entry_path, "uint8", "1 on existing public-road cells (entry cells of the access field)")]
    for lid, path, dtype, what in built:
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(
                id=lid, kind="access_raster", format="raster", path=None, status="pending",
                source=f"built by build_access_raster.py from {', '.join(sorted(inputs))}: {what}",
                licence="derived work; follows the licences of its inputs (ALKIS, DEM)")
        stamp = dl.now_stamps()
        e = m.register(lid, path, captured=today, crs=str(crs))
        m.annotate(lid, captured_at=stamp["time_local"], notes=note,
                   retrieved_with=f"build_access_raster.py (sha256 {script_hash[:16]}), constraints "
                                  f"{CONSTRAINTS.name} (sha256 {acc.sha256[:16]}); python {dl.tool_versions()['python']}",
                   used_for=[f"done {today}: built on the DEM grid {width} x {height}, 1 m",
                             "planned: access-road term of the substation siting; not yet used in a result"])
        dl.append_event(layer_id=lid, service="none (built locally)", what=f"built raster: {what}",
                        file=str(path.relative_to(CASE)), sha256=e.sha256, bytes=path.stat().st_size, **stamp,
                        reconstructed=False,
                        request={"inputs_sha256_prefix": inputs, "grid_epsg25832": [x0, y0, x1, y1],
                                 "cells": width * height, "constraints_sha256": acc.sha256, "dtype": dtype,
                                 "check": note},
                        crs=str(crs), script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
        print(f"registered {lid}: {path.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]}")
    m.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch the 1 m terrain model (Hessen DGM1, WCS 2.0.1) of the study mask into ONE tiled GeoTIFF.

Plan Phase B, B4. The mask is 23.5 x 11.2 km (about 263 M cells at 1 m), too big for one WCS
request, so it is fetched in square tiles. Each tile is held in memory and written once, as a
window of one deflate-compressed, internally tiled float32 GeoTIFF (``data/dem/dgm1.tif``), one
tile at a time with a pause in between. This keeps the write pattern gentle for the university
ransomware guard: one file, sequential, no raw intermediate copies.

The file is written as ``dgm1.tif.partial`` and renamed when every tile is in; the manifest
entry ``dem`` is registered after that (source, licence, capture date, SHA-256).

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/fetch_dem.py --test
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/fetch_dem.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402  # sibling module, written beside the scripts

CASE = HERE.parent
REPO = CASE.parents[1]
MANIFEST = CASE / "data" / "data_manifest.yaml"
OUT_DIR = CASE / "data" / "dem"
MASK = REPO / "case_studies" / "cired2026" / "data" / "input" / "raster_mask.shp"

URL = "https://inspire-hessen.de/raster/dgm1/ows"
COVERAGE = "he_dgm1"
SOURCE = ("state survey (HVBG) DGM1 1 m terrain model, WCS 2.0.1 "
          f"{URL} coverage {COVERAGE}, fetched in tiles with pyorps' catalogue fetcher")
LICENCE = "dl-de/zero-2-0 (Datenlizenz Deutschland Zero 2.0), as stated in the pyorps service preset"


def tile_query(minx, miny, maxx, maxy) -> str:
    return (f"{URL}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
            f"&COVERAGEID={COVERAGE}&FORMAT=image/tiff"
            f"&SUBSET=E({minx},{maxx})&SUBSET=N({miny},{maxy})")


def fetch_tile(minx, miny, maxx, maxy, *, retries=3):
    from pyorps.gui.services.catalog import _fetch

    last = None
    for k in range(retries):
        try:
            raw = _fetch(tile_query(minx, miny, maxx, maxy), 180.0)
            if raw[:2] in (b"II", b"MM"):
                return raw
            last = raw[:300]
        except Exception as exc:                       # network trouble is retried
            last = str(exc)
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"tile ({minx},{miny},{maxx},{maxy}) failed: {last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tile-m", type=int, default=2000)
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between tiles")
    ap.add_argument("--test", action="store_true", help="fetch one tile and report, write nothing")
    args = ap.parse_args(argv)

    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    mask = gpd.read_file(MASK).to_crs(25832)
    bx0, by0, bx1, by1 = mask.total_bounds
    x0, y0 = math.floor(bx0), math.floor(by0)           # snap outward to whole metres
    x1, y1 = math.ceil(bx1), math.ceil(by1)
    width, height = x1 - x0, y1 - y0
    nx, ny = math.ceil(width / args.tile_m), math.ceil(height / args.tile_m)
    print(f"mask {x0},{y0} .. {x1},{y1}: {width} x {height} cells, {nx} x {ny} = {nx * ny} tiles "
          f"of {args.tile_m} m; about {width * height * 4 / 1e9:.2f} GB raw float32")

    if args.test:
        raw = fetch_tile(x0, y0, x0 + args.tile_m, y0 + args.tile_m)
        with MemoryFile(raw) as mf, mf.open() as src:
            a = src.read(1)
            print("tile:", src.width, "x", src.height, src.dtypes[0], "nodata", src.nodata, "crs", src.crs,
                  "transform", tuple(round(v, 6) for v in src.transform)[:6], "bytes", len(raw))
            fin = a[np.isfinite(a) & (a != src.nodata if src.nodata is not None else True)]
            print("elevation range:", float(fin.min()) if fin.size else None,
                  float(fin.max()) if fin.size else None, "finite share", round(fin.size / a.size, 4))
        return 0

    started = dl.now_stamps()
    script_hash = dl.file_sha256(__file__)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / "dgm1.tif"
    part = OUT_DIR / "dgm1.tif.partial"
    if final.exists():
        print(f"{final} exists; refusing to overwrite")
        return 1
    profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="float32",
                   crs="EPSG:25832", transform=from_origin(x0, y1, 1.0, 1.0), nodata=-9999.0,
                   tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                   predictor=3, zlevel=6, BIGTIFF="IF_SAFER")
    done = 0
    with rasterio.open(part, "w", **profile) as dst:
        for j in range(ny):
            for i in range(nx):
                tx0, ty0 = x0 + i * args.tile_m, y0 + j * args.tile_m
                tx1, ty1 = min(tx0 + args.tile_m, x1), min(ty0 + args.tile_m, y1)
                raw = fetch_tile(tx0, ty0, tx1, ty1)
                with MemoryFile(raw) as mf, mf.open() as src:
                    a = src.read(1).astype("float32")
                    if src.nodata is not None:
                        a[a == src.nodata] = -9999.0
                    a[~np.isfinite(a)] = -9999.0
                    h, w = a.shape
                    col = round(src.transform.c - x0)
                    row = round(y1 - src.transform.f)
                    if abs(src.transform.c - x0 - col) > 1e-6 or abs(y1 - src.transform.f - row) > 1e-6:
                        raise RuntimeError(f"tile {i},{j} is not aligned to the 1 m grid: {src.transform}")
                    col = max(col, 0)
                    row = max(row, 0)
                    h = min(h, height - row)
                    w = min(w, width - col)
                    dst.write(a[:h, :w], 1, window=Window(col, row, w, h))
                done += 1
                print(f"tile {done}/{nx * ny} ({i},{j}) ok", flush=True)
                time.sleep(args.pause)
    part.rename(final)
    finished = dl.now_stamps()

    from pyorps.siting.data_manifest import DataManifest

    m = DataManifest.load(MANIFEST)
    e = m.register("dem", final, captured=dt.date.today().isoformat(), source=SOURCE,
                   licence=LICENCE, crs="EPSG:25832")
    m.annotate("dem", captured_at=finished["time_local"],
               retrieved_with=(f"{Path(__file__).name} (sha256 {script_hash[:16]}): {nx * ny} WCS "
                               f"tiles of {args.tile_m} m with the pyorps catalogue fetcher, written "
                               f"into one deflate-compressed tiled GeoTIFF; python "
                               f"{dl.tool_versions()['python']}"))
    dl.append_event(
        layer_id="dem", service=URL, what=f"WCS 2.0.1 coverage {COVERAGE}, 1 m, {nx * ny} tiles",
        file=str(final.relative_to(CASE)), sha256=e.sha256, bytes=final.stat().st_size,
        started_local=started["time_local"], started_utc=started["time_utc"],
        time_local=finished["time_local"], time_utc=finished["time_utc"], reconstructed=False,
        request={"protocol": "WCS 2.0.1 GetCoverage per tile", "tile_m": args.tile_m,
                 "tiles": nx * ny, "extent_epsg25832": [x0, y0, x1, y1],
                 "mask_file": str(MASK.relative_to(REPO)), "mask_shp_sha256": dl.file_sha256(MASK),
                 "cells": width * height, "pause_s": args.pause},
        crs="EPSG:25832", licence_text=LICENCE, script=Path(__file__).name,
        script_sha256=script_hash)
    m.save()
    print(f"saved {final.name}: {final.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

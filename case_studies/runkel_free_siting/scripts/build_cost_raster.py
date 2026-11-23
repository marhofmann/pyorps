"""Build the 1 m rasters of the study area from the downloaded layers (plan Phase B1, B5).

All on ONE grid (the DEM's: whole-metre aligned). ``--build`` picks what to build:

* ``cost``  ``cost_base_1m_closed.tif``: uint16, EUR per metre of trench by land-use class (the CIRED
            2026 stand-in recipe in ``cost_recipe.py``; 65535 = forbidden), main roads buffered.
            **Closed burn (owner, 2026-10-05: routes must never traverse forbidden features):** every
            cell that a forbidden land-use polygon touches, even by a corner or a sliver, is
            forbidden (``all_touched``). Before the file is registered, an independent check burns the
            forbidden polygons itself and requires every touched cell to be 65535; a violation
            refuses the build. The price is that forbidden areas grow by up to one cell, which can
            close real sub-cell openings; that is the intended, conservative direction.
* ``flags`` ``protection_flags_1m.tif`` uint16: bits 0-5 nature reserve, landscape protection,
            Natura 2000 habitat, bird area, protected biotope, protected biotope complex; bits 8-11
            water-protection zone code (1 = zone I, most restrictive). Protected areas are PRICED
            (owner, 2026-10-05), so they are flags for the pricing step, not exclusions.
* ``soil``  ``soil_unit_ids_1m.tif`` uint8 id of the soil unit (0 = none) and a JSON id table.

Every input is verified against its manifest hash first; outputs are registered in the manifest with
their input hashes and one event per file goes into the download log.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_cost_raster.py --box 431373 5582217 433373 5584217 --out-dir <scratch> --no-register
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_cost_raster.py            # cost only
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_cost_raster.py --build cost flags soil
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
OUT = CASE / "data" / "rasters"

PROTECTION = tuple(rec.FLAG_BITS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", nargs="+", choices=("cost", "flags", "soil"), default=["cost"])
    ap.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                    help="build only this box (whole metres, EPSG:25832); default the DEM grid")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-register", action="store_true", help="do not touch the manifest or the log")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from pyorps import CostAssumptions, GeoRasterizer, initialize_geo_dataset
    from pyorps.siting.data_manifest import DataManifest, LayerEntry

    t0 = time.time()
    m = DataManifest.load(MANIFEST)
    needed = ["alkis_land_use", "dem", "soil_units", *PROTECTION, "water_zones"]
    hashes = m.verify(needed)                          # refuses a changed or missing input
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
    print(f"grid {x0},{y0}..{x1},{y1}: {width} x {height} = {width * height} cells", flush=True)
    built: list[tuple[str, Path, str, str]] = []
    extra_notes: dict[str, str] = {}

    def write_raster(path: Path, arr, dtype):
        profile = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype=dtype, crs=crs, transform=transform, tiled=True, blockxsize=512,
                       blockysize=512, compress="deflate", predictor=2, zlevel=6,
                       BIGTIFF="IF_SAFER")
        part = path.with_name(path.name + ".partial")
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(arr, 1)
        part.replace(path)

    # ---------------- cost by land use, closed burn
    if "cost" in args.build:
        cost_path = out_dir / "cost_base_1m_closed.tif"
        ds = initialize_geo_dataset(str(m.file_of("alkis_land_use")))
        ds.load_data()
        ras = GeoRasterizer(ds, CostAssumptions({("nutzart", "bez"): rec.LAND_USE_COSTS}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ras.rasterize(bounding_box=box(x0, y0, x1, y1), preprocessing_function=rec.street_classes,
                          resolution_in_m=1.0, all_touched=True)
        for w in caught:
            print("pyorps warning:", str(w.message)[:600].replace("\n", " "), flush=True)
        cost = np.asarray(ras.raster)
        grid_ok = cost.shape == (height, width) and all(
            abs(a - b) < 1e-6 for a, b in zip(tuple(ras.transform)[:6], tuple(transform)[:6]))
        if not grid_ok:
            raise SystemExit(f"the rasterizer's grid {cost.shape} {tuple(ras.transform)[:6]} is not the "
                             f"DEM grid ({height}, {width}) {tuple(transform)[:6]}; nothing registered")

        # independent closure check: burn the forbidden polygons ourselves, all cells touched
        frame = ds.data.copy()
        forb = rec.forbidden_mask(frame)
        shapes = [(g, 1) for g in frame.loc[forb, "geometry"] if g is not None and not g.is_empty]
        touched = features.rasterize(shapes, out_shape=(height, width), transform=transform,
                                     fill=0, dtype="uint8", all_touched=True).astype(bool)
        missed = int((touched & (cost != rec.FORBIDDEN)).sum())
        extra = int(((cost == rec.FORBIDDEN) & ~touched).sum())
        print(f"closure check: {int(touched.sum())} cells are touched by {len(shapes)} forbidden "
              f"polygons; cells touched but not forbidden: {missed}; forbidden cells no forbidden "
              f"polygon touches (classes the recipe does not list, outside the box edge): {extra}",
              flush=True)
        if missed:
            raise SystemExit(f"{missed} cells touched by a forbidden polygon are not forbidden; "
                             "nothing registered")
        write_raster(cost_path, cost, "uint16")
        share = float((cost == rec.FORBIDDEN).mean())
        print(f"closed cost map written ({time.time() - t0:.0f} s): forbidden {share:.2%}", flush=True)
        extra_notes["cost_base_1m_closed"] = (
            f"closure check passed: all {int(touched.sum())} cells touched by the {len(shapes)} forbidden "
            f"polygons are 65535; {share:.2%} of the cells are forbidden; {extra} further forbidden cells "
            "come from land-use classes the recipe does not list")
        built.append(("cost_base_1m_closed", cost_path, "uint16",
                      "EUR per metre of trench by land-use class (CIRED 2026 stand-in recipe), "
                      "65535 forbidden, every cell touched by a forbidden polygon forbidden"))
        del cost, touched

    # ---------------- protection flags (priced later, not burned into the cost)
    if "flags" in args.build:
        flags = np.zeros((height, width), dtype=np.uint16)
        stats = {}
        for lid, bit in rec.FLAG_BITS.items():
            g = m.read_vector(lid, verify=False)
            shapes = [(geom, 1) for geom in g.geometry if geom is not None and not geom.is_empty]
            mask = features.rasterize(shapes, out_shape=(height, width), transform=transform,
                                      fill=0, dtype="uint8") if shapes else np.zeros((height, width), "uint8")
            flags |= (mask.astype(np.uint16) << bit)
            stats[lid] = int(mask.sum())
            del mask
        wz = m.read_vector("water_zones", verify=False)
        wz = wz.assign(code=wz["ZONE"].map(rec.WATER_ZONE_CODES))
        if wz["code"].isna().any():
            raise SystemExit(f"unknown water zone text: {sorted(wz.loc[wz['code'].isna(), 'ZONE'].unique())}")
        shapes = [(geom, int(c)) for geom, c in sorted(zip(wz.geometry, wz["code"]), key=lambda t: -t[1])]
        zone = features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=0,
                                  dtype="uint8")
        flags |= (zone.astype(np.uint16) << rec.WATER_ZONE_SHIFT)
        stats["water_zone_cells"] = int((zone > 0).sum())
        del zone
        flags_path = out_dir / "protection_flags_1m.tif"
        write_raster(flags_path, flags, "uint16")
        print(f"protection flags written ({time.time() - t0:.0f} s): {stats}", flush=True)
        built.append(("protection_flags_1m", flags_path, "uint16",
                      "protection flags (bits 0-5) and water zone code (bits 8-11); priced later, not exclusions"))
        del flags

    # ---------------- soil unit ids
    if "soil" in args.build:
        soil = m.read_vector("soil_units", verify=False)
        names = sorted(soil["BODENEINHEIT"].fillna("").unique())
        ids = {n: i + 1 for i, n in enumerate(names)}
        shapes = [(geom, ids[n]) for geom, n in zip(soil.geometry, soil["BODENEINHEIT"].fillna(""))]
        sid = features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=0,
                                 dtype="uint8")
        soil_path = out_dir / "soil_unit_ids_1m.tif"
        write_raster(soil_path, sid, "uint8")
        (out_dir / "soil_unit_ids_1m.json").write_text(json.dumps(ids, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        print(f"soil ids written ({time.time() - t0:.0f} s): {len(ids)} units", flush=True)
        built.append(("soil_unit_ids_1m", soil_path, "uint8", "soil unit id (table in soil_unit_ids_1m.json)"))
        del sid

    if args.no_register:
        print("not registered (--no-register)")
        return 0

    # ---------------- register what was built, with its input hashes
    today = dt.date.today().isoformat()
    script_hash = dl.file_sha256(__file__)
    recipe_hash = dl.file_sha256(HERE / "cost_recipe.py")
    inputs = {k: v[:16] for k, v in hashes.items()}
    for lid, path, dtype, what in built:
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(
                id=lid, kind="other", format="raster", path=None, status="pending",
                source=f"built by build_cost_raster.py from {', '.join(sorted(inputs))}: {what}",
                licence="derived work; follows the licences of its inputs (ALKIS, HLNUG, HMLU)")
        stamp = dl.now_stamps()
        e = m.register(lid, path, captured=today, crs=str(crs))
        m.annotate(lid, captured_at=stamp["time_local"],
                   notes=extra_notes.get(lid),
                   retrieved_with=f"build_cost_raster.py (sha256 {script_hash[:16]}) with cost_recipe.py "
                                  f"(sha256 {recipe_hash[:16]}), pyorps GeoRasterizer / rasterio; "
                                  f"python {dl.tool_versions()['python']}",
                   used_for=[f"done {today}: built on the DEM grid {width} x {height}, 1 m",
                             "planned: routing cost surface, protection pricing, trench multipliers by soil; "
                             "not yet used in a result"])
        dl.append_event(layer_id=lid, service="none (built locally)", what=f"built raster: {what}",
                        file=str(path.relative_to(CASE)), sha256=e.sha256,
                        bytes=path.stat().st_size, **stamp, reconstructed=False,
                        request={"inputs_sha256_prefix": inputs, "grid_epsg25832": [x0, y0, x1, y1],
                                 "cells": width * height, "recipe": "cost_recipe.py",
                                 "recipe_sha256": recipe_hash, "dtype": dtype,
                                 "check": extra_notes.get(lid)},
                        crs=str(crs), script=Path(__file__).name, script_sha256=script_hash,
                        licence_text=e.licence)
        print(f"registered {lid}: {path.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]}")
    m.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

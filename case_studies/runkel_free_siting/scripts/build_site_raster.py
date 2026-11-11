"""Build the substation SITE raster (EUR per m2 of switchyard, 65535 = cannot build) on whole-metre grids.

Plan Phase B1. The constraint model (statutes, setbacks, premiums) is the TopoMILP site-raster script's (folder examples/cired2027: only a folder name, no paper is planned for that conference) and is IMPORTED
from ``TopoMILP/examples/cired2027/build_site_raster.py`` unchanged (its sha256 is logged); only the
way the layers are obtained and burned differs:

* layers we already hold (ALKIS land use, protected areas, water zones, soil, biotopes) are read from the
  manifest, hash-verified; the others (buildings, flood areas, further biotopes, natural monuments,
  compensation areas) are fetched once with the same pyorps WFS / ArcGIS REST helpers, saved as GeoPackages
  in ``data/layers/site_constraints/`` and logged with time, counts and hash;
* the burn is CLOSED (owner, 2026-10-05, routes and sites must not use forbidden features): every cell
  a forbidden polygon or an exclusion touches is forbidden (``all_touched``), and an independent check
  refuses the result unless every touched cell is 65535;
* a sidecar JSON records the constraint model, the ledger, the input hashes and the warnings.

Treatments for protected areas follow the owner's answer of 2026-10-05: nature reserves, Natura 2000,
bird areas, water zones I and II, protected biotopes and biotope complexes are excluded for the switchyard;
landscape protection and water zone III are premiums.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_site_raster.py --window farm --check-candidates
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
REPO = CASE.parents[1]
LEGACY_DIR = REPO.parent / "TopoMILP" / "examples" / "cired2027"
LEGACY = LEGACY_DIR / "build_site_raster.py"
MANIFEST = CASE / "data" / "data_manifest.yaml"
OUT = CASE / "data" / "rasters"
SC_DIR = CASE / "data" / "layers" / "site_constraints"

#: legacy (service URL, layer) -> manifest id of a layer we already hold
LOCAL_BY_LAYER = {
    "Naturschutzgebiete": "nature_reserves", "Landschaftsschutzgebiete": "landscape_protection",
    "Fauna-Flora-Habitate": "natura2000_habitats", "Vogelschutzgebiete": "bird_protection_areas",
    "TWS_HQS_TK25": "water_zones", "Bodeneinheiten_Bodenuebersicht_500000": "soil_units",
    "GESCH_BIOTOPE": "biotopes_protected", "GESCH_KOMPLEXE": "biotope_complexes_protected",
}


def _bare(name: str) -> str:
    return str(name).split(":")[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", choices=("farm", "envelope", "study"), default="farm")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--check-candidates", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import box

    sys.path.insert(0, str(LEGACY_DIR))
    import build_site_raster as legacy  # noqa: E402  the constraint model, unchanged

    from pyorps import CostAssumptions, GeoRasterizer, initialize_geo_dataset
    from pyorps.siting.data_manifest import DataManifest, LayerEntry

    t0 = time.time()
    FORB = legacy.FORBIDDEN
    m = DataManifest.load(MANIFEST)
    local_ids = ["alkis_land_use", "dem", *LOCAL_BY_LAYER.values(), "turbines", "terminals"]
    hashes = m.verify(sorted(set(local_ids)))
    constraints = legacy.CONSTRAINTS
    script_hash = dl.file_sha256(__file__)
    legacy_hash = dl.file_sha256(LEGACY)

    # ---------------- the window (whole metres)
    tur = gpd.read_file(m.file_of("turbines")).to_crs(25832)
    if args.window == "study":
        with rasterio.open(m.file_of("dem")) as dem:
            x0, y0, x1, y1 = (int(round(v)) for v in dem.bounds)
    else:
        pad = 1000.0 if args.window == "farm" else 2000.0
        pts = list(tur.geometry)
        if args.window == "envelope":
            pts += list(gpd.read_file(m.file_of("terminals")).to_crs(25832).geometry)
        bx0 = min(p.x for p in pts); by0 = min(p.y for p in pts)
        bx1 = max(p.x for p in pts); by1 = max(p.y for p in pts)
        x0, y0 = int(bx0 - pad), int(by0 - pad)
        x1, y1 = int(bx1 + pad) + 1, int(by1 + pad) + 1
        with rasterio.open(m.file_of("dem")) as dem:          # keep every raster on the DEM grid's extent
            gx0, gy0, gx1, gy1 = (int(round(v)) for v in dem.bounds)
        if (x0, y0, x1, y1) != (max(x0, gx0), max(y0, gy0), min(x1, gx1), min(y1, gy1)):
            print(f"NOTE: the {args.window} window leaves the DEM grid and is clipped to it", flush=True)
        x0, y0, x1, y1 = max(x0, gx0), max(y0, gy0), min(x1, gx1), min(y1, gy1)
    width, height = x1 - x0, y1 - y0
    transform = from_origin(x0, y1, 1.0, 1.0)
    bbox = box(x0, y0, x1, y1)
    pad_m = max(c.buffer_m for c in constraints)
    fetch_bounds = (x0 - pad_m, y0 - pad_m, x1 + pad_m, y1 + pad_m)
    print(f"window {args.window}: {x0},{y0}..{x1},{y1} = {width} x {height} = {width * height} cells; "
          f"fetch padded by {pad_m:.0f} m", flush=True)
    with rasterio.open(m.file_of("dem")) as dem:
        mx0, my0, mx1, my1 = dem.bounds
    if not (fetch_bounds[0] >= mx0 and fetch_bounds[1] >= my0 and fetch_bounds[2] <= mx1
            and fetch_bounds[3] <= my1):
        print("NOTE: the padded window leaves the study mask; layers we hold are clipped to the mask, so "
              "setbacks drawn from features outside it are missing at the edge (permissive there)", flush=True)

    # ---------------- layers: local ones from the manifest, new ones fetched once and logged
    SC_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    new_layers = {}

    def local_gdf(lid):
        return legacy._clean(m.read_vector(lid, verify=False))

    def get_layer(constraint):
        src = constraint.source
        if src in cache:
            return cache[src]
        kind = src[0]
        if kind == "alkis":
            gdf = local_gdf("alkis_land_use")
        elif kind == "wfs" and _bare(src[2]) in LOCAL_BY_LAYER:
            gdf = local_gdf(LOCAL_BY_LAYER[_bare(src[2])])
        elif kind == "file" and "wea" in str(src[1]):
            gdf = local_gdf("turbines")                      # the legacy file input/wea.geojson
        elif kind == "file":
            raise legacy.LayerUnavailable(f"{constraint.key}: legacy file source {src[1]!r} is not available here")
        else:
            name = {"buildings": "buildings"}.get(kind) or _bare(str(src[2]))
            lid = f"sc_{name}_{args.window}".lower().replace("-", "_")
            f = SC_DIR / f"{lid}.gpkg"
            if lid in m.layers and m.layer(lid).status == "present" and f.is_file():
                m.verify([lid])
                gdf = legacy._clean(gpd.read_file(f))
            else:
                with legacy._WfsWatch() as watch:
                    if kind == "buildings":
                        url, layer = legacy.BUILDINGS_WFS["url"], legacy.BUILDINGS_WFS["layer"]
                        gdf = legacy._fetch_wfs(url, layer, fetch_bounds)
                    elif kind == "wfs":
                        url, layer = src[1], src[2]
                        gdf = legacy._fetch_wfs(url, layer, fetch_bounds)
                    elif kind == "rest":
                        url, layer = src[1], src[2]
                        gdf = legacy._fetch_rest(url, layer, fetch_bounds)
                    else:
                        raise legacy.LayerUnavailable(f"unknown source kind {kind!r}")
                    watch.raise_if_incomplete(constraint.key)
                new_layers[lid] = (gdf, f, str(url), str(layer), kind)
        cache[src] = gdf
        return gdf

    # ---------------- base burn (closed)
    base_costs = dict(legacy.SITE_COST_EUR_PER_M2["nutzart"])
    alkis = local_gdf("alkis_land_use")
    ds = initialize_geo_dataset(str(m.file_of("alkis_land_use")))
    ds.load_data()
    ras = GeoRasterizer(ds, CostAssumptions({"nutzart": base_costs}))
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ras.rasterize(resolution_in_m=1.0, fill_value=FORB, bounding_box=bbox, all_touched=True)
    raster = ras.raster
    if raster.shape != (height, width) or any(abs(a - b) > 1e-6 for a, b in
                                              zip(tuple(ras.transform)[:6], tuple(transform)[:6])):
        raise SystemExit(f"rasterizer grid {raster.shape} {tuple(ras.transform)[:6]} is not the window grid")
    warn_texts = [str(w.message)[:300] for w in caught]
    total = raster.size
    ledger = []
    count0 = int((raster != FORB).sum())
    print(f"base land use burned (closed): buildable {count0} cells ({count0 / total:.2%})", flush=True)

    # ---------------- constraints: premiums first, then closed exclusions
    order = {"premium": 0, "exclude": 1, "ignore": 2}
    burned = {}
    forced = []                                   # (key, geometries) of exclusions, for the closure check
    count = count0
    for c in sorted(constraints, key=lambda c: order[c.treatment]):
        if c.treatment == "ignore":
            ledger.append(dict(key=c.key, treat="ignore", n=0, before=count, after=count))
            continue
        try:
            gdf = legacy._apply_selects(get_layer(c), c.selects, c.key)
        except legacy.LayerUnavailable as exc:
            if c.treatment == "exclude" and c.required:
                print(f"FATAL: {c.key} is a required exclusion layer and could not be fetched: {exc}",
                      file=sys.stderr)
                return 2
            print(f"WARNING: {c.key}: {exc}", file=sys.stderr)
            ledger.append(dict(key=c.key, treat=c.treatment, n=None, before=count, after=count))
            continue
        if c.buffer_m > 0 and len(gdf):
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(c.buffer_m)
            gdf = gdf[~gdf.geometry.is_empty]
        burned[c.key] = gdf
        before = count
        if len(gdf) == 0:
            pass
        elif c.treatment == "exclude":
            shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
            touched = features.rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0,
                                         dtype="uint8", all_touched=True).astype(bool)
            raster[touched] = FORB
            forced.append((c.key, shapes))
        elif c.premium_by_value is not None:
            col = c.premium_column
            for value, factor in c.premium_by_value.items():
                part = gdf[gdf[col] == value]
                if len(part):
                    ras.modify_raster_with_geodataframe(part, value=factor, multiply=True, ignore_value=FORB)
        else:
            ras.modify_raster_with_geodataframe(gdf, value=c.premium, multiply=True, ignore_value=FORB)
        count = int((raster != FORB).sum())
        mean_after = float(raster[raster != FORB].mean()) if count else float("nan")
        ledger.append(dict(key=c.key, treat=c.treatment, n=len(gdf), before=before, after=count,
                           mean_after=round(mean_after, 1)))
        print(f"  {c.key:<26} {c.treatment:<8} {len(gdf):>6} feat  buildable {before} -> {count} "
              f"({(before - count) / total:.2%} lost)", flush=True)

    # ---------------- independent closure check: base forbidden classes and every exclusion
    from pyorps.siting import data_manifest as _dm  # noqa: F401  (keeps the import graph explicit)
    sys.path.insert(0, str(HERE))

    base_forb = [(g, 1) for g, n in zip(alkis.geometry, alkis["nutzart"])
                 if base_costs.get(n if isinstance(n, str) else "", FORB) == FORB]
    forced.append(("base_forbidden_classes", base_forb))
    missed = {}
    for key, shapes in forced:
        if not shapes:
            continue
        touched = features.rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0,
                                     dtype="uint8", all_touched=True).astype(bool)
        bad = int((touched & (raster != FORB)).sum())
        if bad:
            missed[key] = bad
    if missed:
        raise SystemExit(f"closure check failed, cells touched by a forbidden polygon but not forbidden: {missed}")
    print("closure check passed: every cell touched by a forbidden class or exclusion is 65535", flush=True)

    near = int(((raster != FORB) & (raster > 60000)).sum())
    if near:
        print(f"WARNING: {near} cells priced above 60000 EUR/m2", file=sys.stderr)

    # ---------------- write
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lid = f"site_cost_{args.window}"
    path = out_dir / f"{lid}_eur_m2.tif"
    part = path.with_name(path.name + ".partial")
    profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="uint16", crs="EPSG:25832",
                   transform=transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                   predictor=2, zlevel=6, BIGTIFF="IF_SAFER")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(raster.astype("uint16"), 1)
    part.replace(path)
    buildable = raster[raster != FORB]
    vals, cnts = np.unique(buildable, return_counts=True)
    summary = dict(cells=total, buildable=int(buildable.size), buildable_share=round(buildable.size / total, 4),
                   distinct_prices=int(len(vals)), price_min=int(buildable.min()), price_max=int(buildable.max()))
    print(f"written {path.name}: {summary}", flush=True)

    # ---------------- candidate screen (legacy function, output captured)
    screen_text = None
    if args.check_candidates:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            legacy._check_candidates(ras, burned, constraints, alkis)
        screen_text = buf.getvalue()
        print(screen_text, flush=True)

    # ---------------- sidecar and registration
    model = [dict(key=c.key, treatment=c.treatment, buffer_m=c.buffer_m, buffer_basis=c.buffer_basis,
                  premium=c.premium, why=c.why) for c in constraints]
    side = dict(window=args.window, grid_epsg25832=[x0, y0, x1, y1], summary=summary, ledger=ledger,
                constraint_model=model, base_costs_eur_m2=base_costs,
                inputs_sha256={k: v for k, v in hashes.items()},
                new_layers={k: v[1].name for k, v in new_layers.items()},
                legacy_script_sha256=legacy_hash, script_sha256=script_hash,
                closure_check="passed", pyorps_warnings=warn_texts, candidate_screen=screen_text,
                built=dl.now_stamps())
    (out_dir / f"{lid}_sidecar.json").write_text(json.dumps(side, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.no_register:
        print("not registered (--no-register)")
        return 0
    today = dt.date.today().isoformat()
    for nid, (gdf, f, url, layer, kind) in new_layers.items():
        gdf.to_file(f, driver="GPKG")
        stamp = dl.now_stamps()
        if nid not in m.layers:
            m.layers[nid] = LayerEntry(id=nid, kind="protected_areas" if "biotop" in nid or "naturdenk" in nid
                                       else "other", format="vector", path=None, status="pending",
                                       source=f"site constraint '{layer}' from {url} ({kind})",
                                       licence="to be confirmed at download")
        e = m.register(nid, f, captured=today, crs=str(gdf.crs))
        m.annotate(nid, captured_at=stamp["time_local"],
                   retrieved_with=f"build_site_raster.py (sha256 {script_hash[:16]}) with the legacy helpers "
                                  f"(sha256 {legacy_hash[:16]}); window {args.window}, fetched over {fetch_bounds}",
                   used_for=[f"done {today}: fetched and saved for the site raster",
                             "used: site raster constraint " + nid])
        dl.append_event(layer_id=nid, service=url, what=f"site constraint {layer} ({kind})", file=str(f.relative_to(CASE)),
                        sha256=e.sha256, bytes=f.stat().st_size, **stamp, reconstructed=False,
                        request={"fetch_bounds_epsg25832": list(fetch_bounds), "window": args.window,
                                 "via": "legacy _fetch_wfs/_fetch_rest (pyorps WFS loader)"},
                        features=len(gdf), crs=str(gdf.crs), script=Path(__file__).name,
                        script_sha256=script_hash, licence_text=e.licence)
    if lid not in m.layers:
        m.layers[lid] = LayerEntry(
            id=lid, kind="site_raster", format="raster", path=None, status="pending",
            source=f"built by build_site_raster.py, window {args.window}, EUR per m2 of switchyard, 65535 = cannot build",
            licence="derived work; follows the licences of its inputs")
    stamp = dl.now_stamps()
    e = m.register(lid, path, captured=today, crs="EPSG:25832")
    m.annotate(lid, captured_at=stamp["time_local"],
               notes=f"closed burn; closure check passed; {summary['buildable_share']:.2%} buildable; constraint model "
                     f"of the TopoMILP site-raster script (sha256 {legacy_hash[:16]}); sidecar {lid}_sidecar.json",
               retrieved_with=f"build_site_raster.py (sha256 {script_hash[:16]}), python {dl.tool_versions()['python']}",
               used_for=[f"done {today}: built", "planned: the site field and the footprint screen; not yet used in a result"])
    dl.append_event(layer_id=lid, service="none (built locally)", what=f"built site raster, window {args.window}",
                    file=str(path.relative_to(CASE)), sha256=e.sha256, bytes=path.stat().st_size, **stamp,
                    reconstructed=False,
                    request={"grid_epsg25832": [x0, y0, x1, y1], "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()},
                             "legacy_script_sha256": legacy_hash, "summary": summary,
                             "closure_check": "passed"},
                    crs="EPSG:25832", script=Path(__file__).name, script_sha256=script_hash,
                    licence_text=e.licence)
    m.save()
    print(f"registered {lid}: {path.stat().st_size / 1e6:.1f} MB, sha256 {e.sha256[:12]} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch the public WFS layers of the free-siting case study (plan Phase B, G4).

Uses the pyorps WFS API (`initialize_geo_dataset`) with the same service URLs as the
CIRED 2026 build scripts, clips to the study mask, saves each layer as a GeoPackage under
``data/layers/`` and registers it in ``data/data_manifest.yaml`` (source, licence text from the
service's capabilities, capture date, SHA-256).

Safety: a layer with more than ``--max-features`` features is REFUSED unless ``--allow-large``
is given (the owner approves big downloads first; large binary writes trip the university
ransomware guard). ``--dry-run`` only counts features (``resultType=hits``).

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/fetch_layers.py --dry-run
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/fetch_layers.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402  # sibling module, written beside the scripts

CASE = HERE.parent
REPO = CASE.parents[1]
MANIFEST = CASE / "data" / "data_manifest.yaml"
LAYER_DIR = CASE / "data" / "layers"
MASK = REPO / "case_studies" / "cired2026" / "data" / "input" / "raster_mask.shp"

ALKIS = "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs"
BEWIRT = ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
          "bewirtschaftungsgebiete/MapServer/WFSServer")
BODEN = ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
         "boden/MapServer/WFSServer")
SCHUTZ = ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
          "schutzgebiete/MapServer/WFSServer")
BIOTOPE = ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
           "lebensraeume_biotope/MapServer/WFSServer")

#: manifest id -> (service URL, layer name)
LAYERS = {
    "water_zones": (BEWIRT, "TWS_HQS_TK25"),
    "soil_units": (BODEN, "Bodeneinheiten_Bodenuebersicht_500000"),
    "nature_reserves": (SCHUTZ, "Naturschutzgebiete"),
    "landscape_protection": (SCHUTZ, "Landschaftsschutzgebiete"),
    "natura2000_habitats": (SCHUTZ, "Fauna-Flora-Habitate"),
    "bird_protection_areas": (SCHUTZ, "Vogelschutzgebiete"),
    "biotopes_protected": (BIOTOPE, "GESCH_BIOTOPE"),
    "biotope_complexes_protected": (BIOTOPE, "GESCH_KOMPLEXE"),
    "alkis_land_use": (ALKIS, "ave:Nutzung"),
}
SMALL = ("water_zones", "soil_units", "nature_reserves", "landscape_protection",
         "natura2000_habitats", "bird_protection_areas", "biotopes_protected",
         "biotope_complexes_protected")
UA = {"User-Agent": "pyorps-fetch/0.1 (research; public data)"}


def licence_text(url: str) -> str:
    """The service's own statement (``Fees`` and ``AccessConstraints``), decoded as UTF-8."""
    r = requests.get(url, params={"service": "WFS", "request": "GetCapabilities",
                                  "version": "2.0.0"}, timeout=90, headers=UA)
    t = r.content.decode("utf-8", "replace")
    parts = []
    for tag in ("Fees", "AccessConstraints"):
        m = re.search(rf"<(?:ows:)?{tag}>(.*?)</(?:ows:)?{tag}>", t, re.S)
        if m and m.group(1).strip().lower() not in ("none", "nolimitations"):
            parts.append(re.sub(r"\s+", " ", m.group(1).strip()))
    return " | ".join(parts) or "no statement in the service capabilities"


def count_hits(url: str, layer: str, bbox) -> int | None:
    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
              "typeNames": layer, "resultType": "hits",
              "srsName": "urn:ogc:def:crs:EPSG::25832",
              "bbox": ",".join(str(v) for v in bbox) + ",urn:ogc:def:crs:EPSG::25832"}
    r = requests.get(url, params=params, timeout=90, headers=UA)
    m = re.search(r'numberMatched="(\d+)"', r.text)
    return int(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layers", nargs="*", default=list(SMALL),
                    help=f"manifest ids (default the small ones: {', '.join(SMALL)})")
    ap.add_argument("--dry-run", action="store_true", help="count features only")
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--allow-large", action="store_true")
    args = ap.parse_args(argv)

    import geopandas as gpd

    from pyorps import initialize_geo_dataset
    from pyorps.siting.data_manifest import DataManifest

    mask = gpd.read_file(MASK)
    bbox = tuple(float(v) for v in mask.to_crs(25832).total_bounds)
    manifest = DataManifest.load(MANIFEST)
    today = dt.date.today().isoformat()
    mask_hash = dl.file_sha256(MASK)
    script_hash = dl.file_sha256(__file__)
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    for lid in args.layers:
        url, layer = LAYERS[lid]
        n = count_hits(url, layer, bbox)
        print(f"{lid}: {n} feature(s) in the study bbox")
        if args.dry_run:
            continue
        if n is not None and n > args.max_features and not args.allow_large:
            print(f"  refused: more than {args.max_features} features; "
                  "ask the owner, then use --allow-large")
            continue
        ds = initialize_geo_dataset({"url": url, "layer": layer}, mask=mask)
        ds.load_data()
        gdf = ds.data
        out = LAYER_DIR / f"{lid}.gpkg"
        gdf.to_file(out, driver="GPKG")
        licence = licence_text(url)
        stamp = dl.now_stamps()
        entry = manifest.register(
            lid, out, captured=today,
            source=f"{manifest.layer(lid).source} (fetched with pyorps WFS API, "
                   f"{len(gdf)} features)",
            licence=licence, crs=str(gdf.crs) if gdf.crs else None)
        retrieved = (f"{Path(__file__).name} (sha256 {script_hash[:16]}) with the pyorps WFS API "
                     f"initialize_geo_dataset(request, mask=raster_mask.shp); "
                     f"python {dl.tool_versions()['python']}")
        manifest.annotate(lid, captured_at=stamp["time_local"], retrieved_with=retrieved)
        currency = None
        if "aktualit" in gdf.columns:
            cur = gdf["aktualit"].astype(str)
            currency = {"feature_update_dates": [cur.min(), cur.max()]}
        dl.append_event(
            layer_id=lid, service=url, what=f"WFS layer {layer}", file=str(out.relative_to(CASE)),
            sha256=entry.sha256, bytes=out.stat().st_size, **stamp, reconstructed=False,
            request={"protocol": "WFS 2.0.0 GetFeature through the pyorps WFS loader",
                     "clip": "study mask polygon", "mask_file": str(MASK.relative_to(REPO)),
                     "mask_shp_sha256": mask_hash, "bbox_epsg25832": list(bbox)},
            service_hits_in_bbox=n, features_saved=len(gdf),
            crs=str(gdf.crs) if gdf.crs else None, licence_text=licence,
            script=Path(__file__).name, script_sha256=script_hash, data_currency=currency)
        print(f"  saved {out.name}: {len(gdf)} features, {out.stat().st_size / 1e6:.2f} MB, "
              f"sha256 {entry.sha256[:12]}")
        manifest.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

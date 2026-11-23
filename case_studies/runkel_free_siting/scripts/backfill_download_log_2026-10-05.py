"""One-off: write the download events of 2026-10-05 into the download log AFTER the fact.

The first seven files of the case study (five small WFS layers, ALKIS land use, the 1 m DEM)
were fetched before the fetch scripts wrote a log. This script reconstructs one event per file
from what is certain: the file's modification time (local, CEST = UTC+2), the file's hash, the
manifest entry, the feature counts the scripts printed, and the hash of each script as it was
when it ran. Every event is flagged ``"reconstructed": true`` with the reason. Running it twice
does nothing the second time.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/backfill_download_log_2026-10-05.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
REPO = CASE.parents[1]
MASK = REPO / "case_studies" / "cired2026" / "data" / "input" / "raster_mask.shp"

# the scripts as they were when they ran (full SHA-256)
FETCH_LAYERS_SHA = "b1d9944a20b45bbd4680420e49e72fa942e990705bd2d37b5985d2482ad673d0"
FETCH_DEM_SHA = "6433cc6a3dd735d3f863be2dccfe0795b6fd3ae9e865cb9ac6bfbdac0dd1abbd"
WHY = ("written after the fact from the file's modification time, the hash and the counts the "
       "script printed; the fetch scripts did not write a log yet")

SERVICES = {
    "water_zones": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
                    "bewirtschaftungsgebiete/MapServer/WFSServer", "TWS_HQS_TK25", 99, 98),
    "soil_units": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
                   "boden/MapServer/WFSServer", "Bodeneinheiten_Bodenuebersicht_500000", 31, 31),
    "nature_reserves": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
                        "schutzgebiete/MapServer/WFSServer", "Naturschutzgebiete", 13, 13),
    "landscape_protection": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
                             "schutzgebiete/MapServer/WFSServer", "Landschaftsschutzgebiete", 1, 1),
    "natura2000_habitats": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
                            "schutzgebiete/MapServer/WFSServer", "Fauna-Flora-Habitate", 9, 9),
    "alkis_land_use": ("https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
                       "ave:Nutzung", 32199, 31938),
}

USED_FOR_DONE = ("done 2026-10-05: fetched, clipped to the study mask, saved, hashed, registered; "
                 "geometry validity and CRS checked; no result uses it yet")
USED_FOR = {
    "alkis_land_use": [
        USED_FOR_DONE,
        "planned: base trench-cost raster and site-cost classes (class costs as in the CIRED 2026 "
        "recipe create_raster_for_windfarm_layout.py); not yet used",
        "planned: road and path classes for the access term and road crossings; not yet used"],
    "water_zones": [
        USED_FOR_DONE,
        "planned: trench-cost multiplier or hard exclusion by zone; hard or priced is the owner's "
        "open legal decision (plan Q12); not yet used"],
    "soil_units": [
        USED_FOR_DONE,
        "planned: trench-cost multiplier by soil unit, as in CIRED 2026; not yet used"],
    "nature_reserves": [
        USED_FOR_DONE,
        "planned: consent class (hard or priced) in the cost raster and in relax-and-verify; "
        "owner's decision Q12 open; not yet used"],
    "landscape_protection": [
        USED_FOR_DONE,
        "planned: consent class in the cost raster and in relax-and-verify; owner's decision Q12 "
        "open; not yet used"],
    "natura2000_habitats": [
        USED_FOR_DONE,
        "planned: consent class in the cost raster and in relax-and-verify; owner's decision Q12 "
        "open; not yet used"],
    "dem": [
        "done 2026-10-05: fetched as 72 WCS tiles into one GeoTIFF, hashed, registered; checked: "
        "23 502 x 11 162 cells, no missing cell, elevation 105.8 to 398.3 m; no result uses it yet",
        "planned: slope limit for access roads (g_max) and terrain for the tower fields; not yet used"],
}


def main() -> int:
    import geopandas as gpd

    from pyorps.siting.data_manifest import DataManifest

    done = {e["layer_id"] for e in dl.read_events() if e.get("reconstructed")}
    m = DataManifest.load(CASE / "data" / "data_manifest.yaml")
    bbox = [float(v) for v in gpd.read_file(MASK).to_crs(25832).total_bounds]
    mask_hash = dl.file_sha256(MASK)
    py = dl.tool_versions()["python"]
    for lid, (url, layer, hits, saved) in SERVICES.items():
        if lid in done:
            continue
        e = m.layer(lid)
        f = m.file_of(lid)
        stamp = dl.stamps_from(f.stat().st_mtime)
        currency = {"feature_update_dates": ["2009-09-17Z", "2026-10-01Z"]} if lid == "alkis_land_use" else None
        dl.append_event(
            layer_id=lid, service=url, what=f"WFS layer {layer}", file=str(f.relative_to(CASE)),
            sha256=e.sha256, bytes=f.stat().st_size, **stamp, reconstructed=True,
            reconstructed_because=WHY, time_meaning="file modification time = end of the download",
            request={"protocol": "WFS 2.0.0 GetFeature through the pyorps WFS loader",
                     "clip": "study mask polygon", "mask_file": str(MASK.relative_to(REPO)),
                     "mask_shp_sha256": mask_hash, "bbox_epsg25832": bbox},
            service_hits_in_bbox=hits, features_saved=saved, crs=e.crs, licence_text=e.licence,
            script="fetch_layers.py", script_sha256=FETCH_LAYERS_SHA, data_currency=currency)
        m.annotate(lid, captured_at=stamp["time_local"], used_for=USED_FOR[lid],
                   retrieved_with=f"fetch_layers.py (sha256 {FETCH_LAYERS_SHA[:16]}) with the pyorps "
                                  f"WFS API initialize_geo_dataset(request, mask=raster_mask.shp); "
                                  f"python {py}")
    if "dem" not in done:
        e = m.layer("dem")
        f = m.file_of("dem")
        end = dl.stamps_from(f.stat().st_mtime)
        dl.append_event(
            layer_id="dem", service="https://inspire-hessen.de/raster/dgm1/ows",
            what="WCS 2.0.1 coverage he_dgm1, 1 m, 72 tiles", file=str(f.relative_to(CASE)),
            sha256=e.sha256, bytes=f.stat().st_size,
            started_local="2026-10-05T14:47:22+02:00", started_utc="2026-10-05T12:47:22+00:00",
            time_local=end["time_local"], time_utc=end["time_utc"], reconstructed=True,
            reconstructed_because=WHY + "; the start time is the creation time of the script's "
                                        "log file, the end time the file's modification time",
            request={"protocol": "WCS 2.0.1 GetCoverage per tile", "tile_m": 2000, "tiles": 72,
                     "extent_epsg25832": [431373, 5582217, 454875, 5593379],
                     "mask_file": str(MASK.relative_to(REPO)), "mask_shp_sha256": mask_hash,
                     "cells": 262329324, "pause_s": 3.0},
            crs="EPSG:25832", licence_text=e.licence, script="fetch_dem.py",
            script_sha256=FETCH_DEM_SHA,
            checks={"cells": "23502 x 11162 = 262 329 324", "missing_cells": 0,
                    "elevation_m": [105.8, 398.3], "grid": "1 m, aligned to whole metres"})
        m.annotate("dem", captured_at=end["time_local"], used_for=USED_FOR["dem"],
                   retrieved_with=f"fetch_dem.py (sha256 {FETCH_DEM_SHA[:16]}): 72 WCS tiles of "
                                  f"2000 m with the pyorps catalogue fetcher, written into one "
                                  f"deflate-compressed tiled GeoTIFF; python {py}")
    m.save()
    print(f"{len(dl.read_events())} events in the log; manifest saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())

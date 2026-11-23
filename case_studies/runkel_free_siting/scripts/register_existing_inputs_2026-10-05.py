"""One-off: freeze the CIRED 2026 input files that already existed as hashed copies in this case study.

The files in ``TopoMILP/data/cired2026/input`` were recovered from a pyorps git stash on 2026-09-04;
their original capture dates are not recorded. This script copies the small ones unchanged into
``data/layers/`` (so the case study owns a frozen, hashed copy), checks that each copy has the
same SHA-256 as its source, registers them in the manifest and writes one download-log event each
(``what`` says "copy of an existing file", not a download). The connection points are a multi-file
shapefile; they are converted to one GeoPackage with the same rows, order and ids, and the source
shapefile's hash is recorded. Running it twice does nothing the second time.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/register_existing_inputs_2026-10-05.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
REPO = CASE.parents[1]
SRC = REPO.parent / "TopoMILP" / "data" / "cired2026" / "input"
DST = CASE / "data" / "layers"
CAVEAT = ("capture date unknown: the file was recovered from a pyorps git stash on 2026-09-04 "
          "(TopoMILP/data/cired2026/README.md); `captured` is that recovery date, not the date the "
          "data was first obtained")

# manifest id -> (file, kind, how it is used)
COPIES = {
    "lines": ("power_lines_110.geojson", "lines",
              ["planned: 110 kV network for the HV connection and the tap points; not yet used"]),
    "towers_110": ("towers_110.geojson", "lines",
                   ["planned: tower nodes (feasible tap cells) for overhead tap points; not yet used"]),
    "turbines": ("wea.geojson", "other",
                 ["planned: the seven turbine positions as collector nodes; not yet used"]),
}


def main() -> int:
    import geopandas as gpd

    from pyorps.siting.data_manifest import DataManifest, LayerEntry

    m = DataManifest.load(CASE / "data" / "data_manifest.yaml")
    DST.mkdir(parents=True, exist_ok=True)
    script_hash = dl.file_sha256(__file__)
    py = dl.tool_versions()["python"]
    logged = {e["layer_id"] for e in dl.read_events() if e.get("what", "").startswith("copy")}

    def add(lid, kind, source, notes):
        if lid not in m.layers:
            m.layers[lid] = LayerEntry(id=lid, kind=kind, format="vector", path=None,
                                       status="pending", source=source,
                                       licence="OpenStreetMap contributors, ODbL", notes=notes)

    for lid, (name, kind, uses) in COPIES.items():
        src, dst = SRC / name, DST / name
        if lid in logged:
            continue
        add(lid, kind, f"CIRED 2026 input {name} (OpenStreetMap / state data as in the CIRED 2026 study)", CAVEAT)
        if not dst.exists():
            shutil.copy2(src, dst)
        if dl.file_sha256(src) != dl.file_sha256(dst):
            raise RuntimeError(f"copy of {name} differs from its source")
        g = gpd.read_file(dst)
        stamp = dl.now_stamps()
        e = m.register(lid, dst, captured="2026-09-04", crs=str(g.crs) if g.crs else None)
        m.annotate(lid, notes=CAVEAT, used_for=["done 2026-10-05: copied unchanged and hashed; "
                   f"{len(g)} features, CRS {g.crs}"] + uses,
                   retrieved_with=f"register_existing_inputs_2026-10-05.py (sha256 {script_hash[:16]}): "
                                  f"shutil.copy2 from TopoMILP/data/cired2026/input/{name}; python {py}")
        dl.append_event(layer_id=lid, service="none (copy of an existing file)",
                        what=f"copy of an existing file, {name}", file=str(dst.relative_to(CASE)),
                        sha256=e.sha256, bytes=dst.stat().st_size, **stamp, reconstructed=False,
                        request={"source_file": f"TopoMILP/data/cired2026/input/{name}",
                                 "source_sha256_equals_copy": True, "caveat": CAVEAT},
                        features=len(g), crs=str(g.crs), script=Path(__file__).name,
                        script_sha256=script_hash, licence_text=e.licence)

    if "terminals" not in logged:
        shp = SRC / "PCCs.shp"
        g = gpd.read_file(shp)
        out = DST / "pccs.gpkg"
        g.to_file(out, driver="GPKG")
        back = gpd.read_file(out)
        if list(back["id"]) != list(g["id"]) or len(back) != len(g):
            raise RuntimeError("the GeoPackage does not keep the shapefile's ids and row order")
        stamp = dl.now_stamps()
        src_hashes = {s: dl.file_sha256(shp.with_suffix(s)) for s in (".shp", ".dbf", ".shx", ".prj")}
        e = m.register("terminals", out, captured="2026-09-04", crs=str(g.crs))
        m.annotate("terminals", notes=CAVEAT + ". Rows keep the shapefile order (ids 1, 2, 0); key by "
                   "the `id` attribute, never by row.",
                   used_for=[f"done 2026-10-05: converted from PCCs.shp to one GeoPackage, ids and row "
                             f"order checked equal; {len(g)} connection points",
                             "planned: the grid connection points (PCC) of the case study; not yet used"],
                   retrieved_with=f"register_existing_inputs_2026-10-05.py (sha256 {script_hash[:16]}): "
                                  f"geopandas read of PCCs.shp, written to GeoPackage; python {py}")
        dl.append_event(layer_id="terminals", service="none (conversion of an existing file)",
                        what="copy of an existing file, PCCs.shp converted to GeoPackage",
                        file=str(out.relative_to(CASE)), sha256=e.sha256, bytes=out.stat().st_size,
                        **stamp, reconstructed=False,
                        request={"source_file": "TopoMILP/data/cired2026/input/PCCs.shp",
                                 "source_sha256": src_hashes, "caveat": CAVEAT},
                        features=len(g), crs=str(g.crs), script=Path(__file__).name,
                        script_sha256=script_hash, licence_text=e.licence)
    m.save()
    print("registered:", sorted(k for k, v in m.layers.items() if v.status == "present"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

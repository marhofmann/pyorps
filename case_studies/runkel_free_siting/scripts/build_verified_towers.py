"""Verified 110 kV towers: OSM support points on lines whose voltage field contains 110000 (owner rule, 2026-10-06).

Reads the frozen ``lines`` and ``towers_110`` layers (hash-verified), keeps the towers and poles that lie on such a
line (``pyorps.siting.hv_towers``), writes ``layers/towers_110_verified.gpkg`` (one small file) and registers it.
Railway traction lines are kept by the owner's rule and flagged in the ``traction`` column; the counts with and without
them are printed and recorded so the choice can be revisited.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/build_verified_towers.py [--exclude-traction] [--no-register]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download_log as dl  # noqa: E402

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
LID = "towers_110_verified"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tol-m", type=float, default=1.0)
    ap.add_argument("--keep-traction", action="store_true", help="keep railway traction lines (decision D-e drops them)")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    from pyorps.siting.data_manifest import DataManifest, LayerEntry
    from pyorps.siting.hv_towers import verified_110kv_towers

    m = DataManifest.load(MANIFEST)
    hashes = m.verify(["lines", "towers_110"])
    lines, towers = m.read_vector("lines", verify=False), m.read_vector("towers_110", verify=False)
    out = verified_110kv_towers(lines, towers, tol_m=args.tol_m, exclude_traction=not args.keep_traction)
    n_all = len(verified_110kv_towers(lines, towers, tol_m=args.tol_m, exclude_traction=False))
    n_pub = len(verified_110kv_towers(lines, towers, tol_m=args.tol_m, exclude_traction=True))
    note = (f"{len(out)} towers and poles on lines whose voltage contains 110000 (tolerance {args.tol_m} m; "
            f"railway traction lines dropped={not args.keep_traction}, decision D-e); with railway traction lines {n_all}, without {n_pub}; of "
            f"{len(towers)} OSM support and equipment points, only 5 carry a voltage tag themselves")
    print(note, flush=True)
    path = CASE / "data" / "layers" / "towers_110_verified.gpkg"
    out.to_file(path, driver="GPKG")
    if args.no_register:
        return 0
    today, script_hash, stamp = dt.date.today().isoformat(), dl.file_sha256(__file__), dl.now_stamps()
    if LID not in m.layers:
        m.layers[LID] = LayerEntry(id=LID, kind="lines", format="vector", path=None, status="pending",
                                   source="derived from lines and towers_110 by build_verified_towers.py (owner rule 2026-10-06)",
                                   licence="derived work; follows the licence of its inputs (OpenStreetMap, ODbL)")
    e = m.register(LID, path, captured=today, crs="EPSG:25832")
    m.annotate(LID, captured_at=stamp["time_local"], notes=note,
               retrieved_with=f"build_verified_towers.py (sha256 {script_hash[:16]}), pyorps.siting.hv_towers; python {dl.tool_versions()['python']}",
               used_for=[f"done {today}: tower pool for tap points and overhead lattice", "planned: HV tap points and overhead bound; not yet used in a result"])
    dl.append_event(layer_id=LID, service="none (built locally)", what="derived layer: verified 110 kV towers",
                    file=str(path.relative_to(CASE)), sha256=e.sha256, bytes=path.stat().st_size, **stamp, reconstructed=False,
                    request={"inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()}, "check": note}, crs="EPSG:25832",
                    script=Path(__file__).name, script_sha256=script_hash, licence_text=e.licence)
    m.save()
    print(f"registered {LID}: {path.stat().st_size / 1e6:.2f} MB, sha256 {e.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

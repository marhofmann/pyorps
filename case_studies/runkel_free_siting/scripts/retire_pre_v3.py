#!/usr/bin/env python
"""Rename the pre-plan caches aside as ``*_pre_v3``, read-only (plan rev. 5, A3).

Plan A3 and section 15 retire every field, screen and result that was built
before the provenance record existed, on inputs the review found defective
(a base-only site raster, row-keyed PCC fields, the envelope crop, retired
grids, a mis-joined free-siting result). Nothing is deleted: each path is
renamed with a ``_pre_v3`` suffix (before the extension for files) and its
files are marked read-only, so no script picks them up by name and none can
overwrite them. A manifest of every rename is written first; ``--undo``
reverses it exactly.

The Sophos CryptoGuard constraint on this machine forbids bursts of file
operations, so renames are paced (``--pace`` seconds apart, default 2 s)
and read-only flags are set file by file with a short pause.

Usage (dry run by default):
    python retire_pre_v3.py                      # list what would move
    python retire_pre_v3.py --apply              # rename and protect
    python retire_pre_v3.py --undo MANIFEST      # put everything back
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path

TOPO = Path(r"C:\Users\mhnn82\Documents\2_python_projects\TopoMILP")
PYORPS = Path(__file__).resolve().parents[3]

#: (path, why) -- plan section 15 "A3 retire list", plus the field caches
#: the A1-A3 reader found (row-keyed, envelope crop, no provenance).
RETIRE = [
    (TOPO / "results/cired2027/free_siting.json",
     "9 953 235 EUR free-siting result, internally mis-joined (row-keyed PCC)"),
    (TOPO / "results/cired2027/free_siting_variants.json",
     "the 9 284 693 EUR 66 kV headline, invalid (plan section 0)"),
    (TOPO / "results/cired2027/corridor_free_sites",
     "row-keyed substations_to_pcc routes of the retired free siting"),
    (TOPO / "results/cired2027/corridor_free_sites_variants",
     "HV-variants bundle: ST3-ST5 are poles, PCC bay unpriced (plan 15)"),
    (TOPO / "data/cired2026/input/free_sites.geojson",
     "old 15-site shortlist from the base-only site raster"),
    (TOPO / "results/cired2027/free_site_costs.json", "priced on retired inputs"),
    (TOPO / "data/cired2026/site_screen.json", "screen on the base-only site raster"),
    (TOPO / "results/cired2027/site_screen.json", "screen on the base-only site raster"),
    (TOPO / "results/cired2027/site_evaluation.json", "evaluated on retired inputs"),
    (TOPO / "results/cired2027/corridor_sweep_refreshed.json",
     "different routing bundle (cell 0.99998766), unexplained 1 014 EUR drift"),
    (TOPO / "data/cired2026/raster/site_cost_full.tiff",
     "base-only burn: no constraint layers, no turbine buffer (V1-14)"),
    (TOPO / "data/cired2026/raster/mod1_raster_wp_fixed.tiff", "another grid"),
    (TOPO / "data/cired2026/raster/mod1_raster_wp_legacy.tiff", "another grid"),
    (TOPO / "data/cired2026/raster/mod2_raster_wp_fixed.tiff",
     "another grid; fails #1277"),
    (TOPO / "data/cired2026/raster/mod2_raster_wp_legacy.tiff",
     "another grid; lacks the multipliers"),
    (TOPO / "data/cired2026/raster/fields", "field cache without provenance"),
    (TOPO / "data/cired2026/raster/fields_len5m",
     "5 m pooled length fields (not bounds, plan 3.5)"),
    (TOPO / "data/cired2026/raster/fields_min5m",
     "5 m min-pooled fields: NOT lower bounds (M3-04)"),
    (TOPO / "data/cired2026/raster/hv_fields",
     "row-keyed HV fields on the envelope crop, delta-stepping"),
    (PYORPS / "case_studies/cired2026/results",
     "HV-variants bundle (pyorps side)"),
]
# NOT retired automatically: pyorps case_studies/cired2026/results_spanover
# (the span-over HV-variants bundle). No .gitignore rule covers it, so it may
# be tracked, and a rename would show up as git changes -- retire it by hand
# (git mv) if wanted.


def retired_name(p: Path) -> Path:
    if p.is_dir() or not p.suffix:
        return p.with_name(p.name + "_pre_v3")
    return p.with_name(p.stem + "_pre_v3" + p.suffix)


def make_read_only(p: Path, pause: float) -> int:
    files = [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
    for f in files:
        mode = f.stat().st_mode
        os.chmod(f, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        time.sleep(pause)
    return len(files)


def make_writable(p: Path, pause: float) -> None:
    files = [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
    for f in files:
        os.chmod(f, f.stat().st_mode | stat.S_IWUSR)
        time.sleep(pause)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="do it (default: dry run)")
    ap.add_argument("--undo", type=Path, help="reverse a manifest")
    ap.add_argument("--pace", type=float, default=2.0, help="seconds between renames")
    ap.add_argument("--manifest-dir", type=Path,
                    default=PYORPS / "case_studies/runkel_free_siting/results")
    args = ap.parse_args(argv)

    if args.undo:
        rows = json.loads(args.undo.read_text(encoding="utf-8"))["renamed"]
        for row in reversed(rows):
            new, old = Path(row["new"]), Path(row["old"])
            if not new.exists() or old.exists():
                print(f"skip {new} (missing, or {old} exists)")
                continue
            make_writable(new, 0.05)
            new.rename(old)
            print(f"restored {old}")
            time.sleep(args.pace)
        return 0

    plan = []
    for p, why in RETIRE:
        if not p.exists():
            print(f"absent   {p}")
            continue
        dest = retired_name(p)
        if dest.exists():
            print(f"CLASH    {dest} already exists; leaving {p}")
            continue
        plan.append((p, dest, why))
        print(f"{'retire' if args.apply else 'would'}   {p} -> {dest.name}  ({why})")
    if not args.apply:
        print(f"\n{len(plan)} path(s) would be retired; rerun with --apply")
        return 0

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    manifest = args.manifest_dir / f"retire_pre_v3_{stamp}.json"
    rows = [{"old": str(p), "new": str(d), "why": w} for p, d, w in plan]
    manifest.write_text(json.dumps({"plan": "rev. 5 A3", "renamed": rows},
                                   indent=1), encoding="utf-8")
    print(f"manifest {manifest}")
    for p, dest, _why in plan:
        p.rename(dest)
        n = make_read_only(dest, 0.05)
        print(f"retired  {dest} ({n} file(s) read-only)")
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    sys.exit(main())

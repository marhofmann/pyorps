"""Prepare everything the history benchmark needs, once.

Idempotent: every step is skipped when its output already exists, so re-running
is cheap and safe.

Steps
  1. create the data root (outside the repository)
  2. seed / build the case-study inputs (bbox, source, targets, ALKIS vector)
  3. build the ONE reference cost raster every version routes on
  4. export each version with ``git archive`` and build its Cython extensions
  5. export the shared search window for the scikit-image prototype

On git: this module only ever READS the repository -- ``git archive``,
``git rev-parse``, ``git tag``. It creates no worktree, changes no branch,
stages nothing and commits nothing.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=C.REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------
# 1 + 2: directories and inputs
# --------------------------------------------------------------------------
INPUT_FILES = {
    "bbox.geojson": C.BBOX,
    "source.geojson": C.SOURCE,
    "targets.geojson": C.TARGETS,
    "alkis_mvo.gpkg": C.ALKIS,
    "ref_raster.tiff": C.REF_RASTER,
}


def ensure_dirs() -> None:
    for d in (C.DATA_ROOT, C.INPUTS, C.CHECKOUTS, C.RUNS):
        d.mkdir(parents=True, exist_ok=True)
    log(f"data root: {C.DATA_ROOT}")


def seed_inputs(seed_from: Path | None) -> None:
    """Copy cached inputs from a previous run / scratch directory."""
    if seed_from is None:
        return
    seed_from = Path(seed_from)
    if not seed_from.is_dir():
        log(f"seed dir does not exist, skipping: {seed_from}")
        return
    for name, dest in INPUT_FILES.items():
        src = seed_from / name
        if dest.exists():
            continue
        if not src.exists():
            continue
        log(f"seeding {name} ({src.stat().st_size / 2**20:.0f} MiB) ...")
        t0 = time.perf_counter()
        shutil.copy2(src, dest)
        log(f"  copied in {time.perf_counter() - t0:.1f} s")


def build_case_geometry() -> None:
    """Derive bbox / source / targets from pandapower's mv_oberrhein."""
    if C.BBOX.exists() and C.SOURCE.exists() and C.TARGETS.exists():
        return
    log("building case geometry from pandapower.networks.mv_oberrhein ...")
    try:
        import pandapower as pp
    except ImportError:
        raise SystemExit(
            "pandapower is required to derive the MV-Oberrhein geometry, and it "
            "is not installed.\n"
            "Either install it (pip install pandapower) or seed the cached "
            "files with --seed-from <dir containing bbox/source/targets.geojson>."
        )
    import geopandas as gpd
    from shapely.geometry import Polygon

    net = pp.networks.mv_oberrhein(scenario="generation",
                                   separation_by_sub=True,
                                   include_substations=True)[1]
    gs = net.bus.geo.geojson.as_geoseries.to_crs("epsg:4326").to_crs(C.CRS)

    if not C.BBOX.exists():
        minx, miny, maxx, maxy = gs.total_bounds
        minx, miny, maxx, maxy = minx - 1000, miny - 1000, maxx + 1000, maxy + 1000
        poly = Polygon([(maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)])
        gpd.GeoDataFrame(index=[0], geometry=[poly], crs=C.CRS).to_file(C.BBOX)
        log(f"  bbox -> {C.BBOX}")

    src_path = (C.REPO_ROOT / "case_studies" / "mv_oberrhein" / "data" /
                "shapes" / "sources.geojson")
    if not C.SOURCE.exists():
        if not src_path.exists():
            raise SystemExit(f"PV plant source geometry not found: {src_path}")
        gpd.read_file(src_path).to_crs(C.CRS).to_file(C.SOURCE)
        log(f"  source -> {C.SOURCE}")

    if not C.TARGETS.exists():
        source = gpd.read_file(C.SOURCE).geometry.iloc[0]
        buses = net.bus.loc[net.bus.index.isin(net.trafo.hv_bus)
                            & (net.bus.vn_kv == 20)].index.to_list()
        bgs = (net.bus.loc[buses].geo.geojson.as_geoseries
               .to_crs("epsg:4326").to_crs(C.CRS))
        closest = bgs.loc[bgs.geometry.distance(source).sort_values()
                          .iloc[0:20].index]
        sel = closest.loc[C.CASE_BUS_ORDER]
        gpd.GeoDataFrame({"bus": sel.index}, geometry=sel.values,
                         crs=C.CRS).to_file(C.TARGETS)
        log(f"  targets -> {C.TARGETS}")


def fetch_alkis() -> None:
    if C.ALKIS.exists():
        return
    log("fetching ALKIS land-use data from the LGL-BW WFS (one time) ...")
    import geopandas as gpd
    sys.path.insert(0, str(C.REPO_ROOT))
    import pyorps
    from pyorps.io import vector_loader

    # The loader forces pyogrio's Arrow fast path, which raises
    # NotImplementedError on the nonlinear (curve) geometries this WFS returns.
    _orig = gpd.read_file

    def _no_arrow(path, *a, **kw):
        kw.pop("use_arrow", None)
        return _orig(path, *a, **kw)

    vector_loader.gpd.read_file = _no_arrow

    bbox = gpd.read_file(C.BBOX)
    t0 = time.perf_counter()
    ds = pyorps.initialize_geo_dataset(C.WFS_REQUEST, bbox=bbox)
    ds.load_data()
    log(f"  {len(ds.data)} features in {time.perf_counter() - t0:.1f} s")
    ds.data.to_file(C.ALKIS, layer="nutzung", driver="GPKG")
    log(f"  -> {C.ALKIS}")


def build_reference_raster() -> None:
    if C.REF_RASTER.exists():
        return
    log("building the reference cost raster (1 m, ~580 Mcells) ...")
    import geopandas as gpd
    sys.path.insert(0, str(C.REPO_ROOT))
    import pyorps

    bbox = gpd.read_file(C.BBOX)
    t0 = time.perf_counter()
    ds = pyorps.initialize_geo_dataset(str(C.ALKIS), bbox=bbox)
    ds.load_data()
    rasterizer = pyorps.GeoRasterizer(ds, C.COST_ASSUMPTIONS, bbox)
    rasterizer.rasterize(save_path=str(C.REF_RASTER))
    log(f"  -> {C.REF_RASTER} in {time.perf_counter() - t0:.1f} s "
        f"({C.REF_RASTER.stat().st_size / 2**30:.2f} GiB)")


# --------------------------------------------------------------------------
# 4: version checkouts
# --------------------------------------------------------------------------
def export_version(version: dict, python: str, rebuild: bool = False) -> dict:
    """Materialise one version and build its Cython extensions."""
    label = version["label"]
    if version["kind"] == "worktree":
        return dict(label=label, checkout=str(C.REPO_ROOT), commit=git(
            "rev-parse", "--short", "HEAD"), built="working tree (as-is)")
    if version["kind"] == "prototype":
        return dict(label=label, checkout=None, commit=None, built="n/a")

    dest = C.CHECKOUTS / label
    commit = git("rev-parse", "--short", version["ref"] + "^{commit}")
    stamp = dest / ".bench_stamp.json"

    if stamp.exists() and not rebuild:
        info = json.loads(stamp.read_text(encoding="utf-8"))
        if info.get("commit") == commit:
            log(f"{label}: cached ({commit})")
            return info

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    log(f"{label}: exporting {version['ref']} ({commit}) with git archive ...")
    raw = subprocess.run(["git", "archive", "--format=tar", version["ref"]],
                         cwd=C.REPO_ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(dest)

    info = dict(label=label, checkout=str(dest), commit=commit, ref=version["ref"])

    if (dest / "setup.py").exists():
        log(f"{label}: building Cython extensions ...")
        t0 = time.perf_counter()
        proc = subprocess.run([python, "setup.py", "build_ext", "--inplace"],
                              cwd=dest, capture_output=True, text=True)
        (dest / "_build_ext.log").write_text(
            proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
        pyds = sorted(p.name for p in (dest / "pyorps").rglob("*.pyd"))
        info["build_rc"] = proc.returncode
        info["build_s"] = round(time.perf_counter() - t0, 1)
        info["extensions"] = pyds
        info["built"] = f"rc={proc.returncode}, {len(pyds)} extension(s)"
        if proc.returncode != 0:
            log(f"  !! build FAILED (rc={proc.returncode}); see "
                f"{dest / '_build_ext.log'}")
            log("     compiled backends will be unavailable for this version; "
                "jobs that need them will be recorded as 'error', not skipped")
        else:
            log(f"  built {len(pyds)} extension(s) in {info['build_s']} s")
    else:
        info["built"] = "no setup.py (pure Python version)"

    stamp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


# --------------------------------------------------------------------------
# 5: shared search window
# --------------------------------------------------------------------------
def export_window(force: bool = False) -> None:
    """Export the exact cost window pyorps searches, for the prototype.

    Taken from a real ``PathFinder`` construction rather than reimplemented, so
    the scikit-image baseline provably solves the same problem on the same
    cells with the same buffer.
    """
    if C.WINDOW_COST.exists() and C.WINDOW_META.exists() and not force:
        log("search window: cached")
        return
    log("exporting the shared search window from a HEAD PathFinder ...")
    import numpy as np
    import geopandas as gpd
    sys.path.insert(0, str(C.REPO_ROOT))
    import pyorps

    source = gpd.read_file(C.SOURCE)
    targets = gpd.read_file(C.TARGETS)
    pf = pyorps.PathFinder(str(C.REF_RASTER), source_coords=source,
                           target_coords=targets, neighborhood_str="r2",
                           ignore_max_cost=C.IGNORE_MAX_COST)
    rh = pf.raster_handler
    data = np.asarray(rh.data)
    if data.ndim == 3:
        data = data[0]
    tr = rh.window_transform

    np.save(C.WINDOW_COST, data.astype(np.uint16))
    meta = dict(
        transform=[tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
        shape=list(data.shape),
        crs=C.CRS,
        search_space_buffer_m=getattr(pf, "search_space_buffer_m", None),
        ignore_max_cost=C.IGNORE_MAX_COST,
        source_xy=[float(source.geometry.iloc[0].x),
                   float(source.geometry.iloc[0].y)],
        target_xy={str(int(b)): [float(g.x), float(g.y)]
                   for b, g in zip(targets["bus"], targets.geometry)},
    )
    C.WINDOW_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"  window {data.shape} -> {C.WINDOW_COST} "
        f"({C.WINDOW_COST.stat().st_size / 2**20:.0f} MiB), "
        f"buffer={meta['search_space_buffer_m']} m")


def prepare_all(versions: list[dict], python: str, seed_from: Path | None,
                rebuild: bool = False) -> dict:
    ensure_dirs()
    seed_inputs(seed_from)
    build_case_geometry()
    fetch_alkis()
    build_reference_raster()
    export_window(force=rebuild)
    checkouts = {}
    for v in versions:
        checkouts[v["label"]] = export_version(v, python, rebuild=rebuild)
    return checkouts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="directory holding cached inputs to copy in")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    info = prepare_all(C.VERSIONS + C.EXTRA_VERSIONS, a.python, a.seed_from,
                       a.rebuild)
    print(json.dumps(info, indent=1))

"""
Export the analytic benchmark cases as GeoTIFF + source features for
ArcGIS Pro Distance Accumulation (eikonal plan section 6.4).

Produces, per case, under this directory:
    cases/<name>_cost.tif      float32 cost raster, NoData = barriers
    cases/<name>_source.gpkg   source point feature (layer "source")
    fim/<name>_fim.tif         our block-FIM accumulation field
                               (skipped without a CUDA GPU)
    cases/cases.json           case metadata (source cell, probes)

The analytic closed forms are the referee — see PROCEDURE.md for the
manual ArcGIS steps and compare_results.py for the evaluation.

Usage:
    .venv/Scripts/python.exe benchmarks/arcgis_comparison/generate_cases.py
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cases import ALL_CASES, CRS, CELL  # noqa: E402

HERE = Path(__file__).parent
NODATA = -9999.0


def write_geotiff(path, data, nodata=None, crs=CRS, origin=None):
    """origin = (x_left, y_top) map coordinates; default synthetic
    frame (0, rows). Cells are always exactly 1 m square."""
    rows, cols = data.shape
    x0, y0 = origin if origin is not None else (0.0, float(rows))
    profile = dict(
        driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs=crs,
        transform=from_origin(x0, y0, CELL, CELL),
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def cell_to_xy(r, c, rows, origin=None):
    """Cell (row, col) center -> map coordinates."""
    x0, y0 = origin if origin is not None else (0.0, float(rows))
    return x0 + c + 0.5, y0 - r - 0.5


def write_source_point(path, r, c, rows, crs=CRS, origin=None):
    import geopandas as gpd
    from shapely.geometry import Point
    x, y = cell_to_xy(r, c, rows, origin)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(x, y)], crs=crs)
    gdf.to_file(path, layer="source", driver="GPKG")


def main():
    (HERE / "cases").mkdir(exist_ok=True)
    (HERE / "fim").mkdir(exist_ok=True)
    (HERE / "results_arcgis").mkdir(exist_ok=True)

    try:
        from pyorps.utils.eikonal_gpu import eikonal_raster_gpu
        from pyorps.utils.traversal_gpu import GPU_AVAILABLE
        gpu = GPU_AVAILABLE
    except ImportError:
        gpu = False
    if not gpu:
        print("NOTE: no CUDA GPU — cost rasters and sources are "
              "exported, the FIM side is skipped.")

    meta = {}
    for factory in ALL_CASES:
        try:
            case = factory()
        except FileNotFoundError as exc:
            print(f"SKIPPED {factory.__name__}: missing input {exc}")
            continue
        name = case["name"]
        raster = case["raster"]
        rows, cols = raster.shape
        sr, sc = case["source"]
        crs = case.get("crs", CRS)
        origin = case.get("origin")

        cost = raster.copy()
        cost[~np.isfinite(cost)] = NODATA
        write_geotiff(HERE / "cases" / f"{name}_cost.tif", cost,
                      nodata=NODATA, crs=crs, origin=origin)
        write_source_point(HERE / "cases" / f"{name}_source.gpkg",
                           sr, sc, rows, crs=crs, origin=origin)

        entry = dict(rows=rows, cols=cols, source_cell=[int(sr), int(sc)],
                     source_xy=list(cell_to_xy(sr, sc, rows, origin)),
                     crs=crs)
        if "reference" in case:
            entry["reference"] = case["reference"]
        if "probes" in case:
            entry["probes"] = case["probes"]
        meta[name] = entry

        if gpu:
            solve_input = raster.astype(np.float32).copy()
            solve_input[~np.isfinite(solve_input)] = np.inf
            t = eikonal_raster_gpu(solve_input, sr * cols + sc)
            t_out = t.astype(np.float32).copy()
            t_out[t_out >= 1e29] = NODATA
            write_geotiff(HERE / "fim" / f"{name}_fim.tif", t_out,
                          nodata=NODATA, crs=crs, origin=origin)

        print(f"{name}: {rows}x{cols} exported"
              + (" (+ FIM field)" if gpu else ""))

    (HERE / "cases" / "cases.json").write_text(json.dumps(meta, indent=1))
    print(f"\nmetadata -> {HERE / 'cases' / 'cases.json'}")
    print("Next: run the ArcGIS side per PROCEDURE.md, drop results "
          "into results_arcgis/, then run compare_results.py")


if __name__ == "__main__":
    main()

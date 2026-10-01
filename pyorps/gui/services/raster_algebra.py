"""
PYORPS GUI: raster algebra — combine several cost rasters into one (Feature 6).

Cost rasters can be concatenated / overlaid / merged / multiplied / added.
Inputs rarely share a grid (different resolution, extent or CRS), so every
input is reprojected + resampled onto the FIRST selected raster's grid (the
"reference") with nearest-neighbour resampling — which preserves the discrete
cost / forbidden sentinels exactly (a bilinear average of 405 and 65535 would
be meaningless). The forbidden sentinel (the dtype max, 65535 for uint16) is
treated as "no data" per operation so it never pollutes an arithmetic result:

- ``add`` / ``multiply``  : combine only where BOTH inputs are passable; if any
  input is forbidden at a cell, the result is forbidden.
- ``min`` / ``max``       : ignore forbidden inputs; forbidden only if ALL are.
- ``overlay``             : first passable input in the stack wins (top layer
  over the ones below); forbidden only where every input is forbidden.
- ``merge`` / mosaic      : same rule as overlay (fill gaps top-down), the
  natural "mosaic several partial rasters" operation.

The result is written as a tiled + overviewed GeoTIFF (via
:func:`pyorps.gui.services.tiles._write_tiled_geotiff`) so it serves fast.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .tiles import _write_tiled_geotiff

OPERATIONS = ("add", "multiply", "min", "max", "overlay", "merge")


def _forbidden_value(dtype: np.dtype) -> float:
    """The sentinel that marks impassable cells for a dtype (max for ints)."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float("inf")


def _read(path: str | Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {"data": src.read(1), "crs": src.crs, "transform": src.transform,
                "width": src.width, "height": src.height, "dtype": src.dtypes[0],
                "nodata": src.nodata}


def _align_to_reference(source: dict, ref: dict) -> np.ndarray:
    """Resample ``source`` onto the reference grid (nearest — keeps sentinels)."""
    forbidden = _forbidden_value(np.dtype(ref["dtype"]))
    out = np.full((ref["height"], ref["width"]), forbidden,
                  dtype=source["data"].dtype)
    reproject(
        source=source["data"], destination=out,
        src_transform=source["transform"], src_crs=source["crs"],
        dst_transform=ref["transform"], dst_crs=ref["crs"],
        src_nodata=(source["nodata"]
                    if source["nodata"] is not None
                    else _forbidden_value(np.dtype(source["dtype"]))),
        dst_nodata=forbidden,
        resampling=Resampling.nearest)
    return out


def combine_rasters(paths: list[str | Path], operation: str, *,
                    work_dir: str | Path = ".",
                    save_path: str | Path | None = None) -> tuple[str, list[str]]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Combine cost rasters with ``operation``; return ``(out_path, log)``.

    All inputs are aligned onto ``paths[0]``'s grid (the reference). The output
    keeps the reference dtype and CRS. See the module docstring for how each
    operation treats forbidden cells.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if operation not in OPERATIONS:
        raise ValueError(
            f"Unknown raster operation '{operation}'. Use one of: "
            f"{', '.join(OPERATIONS)}.")
    paths = [str(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("Combining rasters needs at least two raster layers "
                         "— select two or more.")

    log: list[str] = []
    ref = _read(paths[0])
    ref_dtype = np.dtype(ref["dtype"])
    forbidden = _forbidden_value(ref_dtype)
    log.append(f"reference grid: {ref['width']}x{ref['height']} "
               f"{ref['crs']} ({ref_dtype})")

    # aligned stack of float arrays with NaN marking forbidden/no-data
    stack: list[np.ndarray] = []
    for i, path in enumerate(paths):
        raw = ref["data"] if i == 0 else _align_to_reference(_read(path), ref)
        arr = raw.astype("float64")
        arr[raw == _forbidden_value(np.dtype(raw.dtype))] = np.nan
        if i != 0:
            log.append(f"aligned {Path(path).name} onto reference")
        stack.append(arr)
    cube = np.stack(stack, axis=0)                 # (n, H, W), NaN = forbidden
    all_forbidden = np.all(np.isnan(cube), axis=0)

    if operation in ("add", "multiply"):
        any_forbidden = np.any(np.isnan(cube), axis=0)
        filled = np.nan_to_num(cube, nan=(0.0 if operation == "add" else 1.0))
        result = (filled.sum(axis=0) if operation == "add"
                  else filled.prod(axis=0))
        result[any_forbidden] = np.nan          # forbidden if ANY input is
    elif operation == "min":
        result = np.nanmin(cube, axis=0)
    elif operation == "max":
        result = np.nanmax(cube, axis=0)
    else:                                        # overlay / merge — top wins
        result = np.full(cube.shape[1:], np.nan)
        for layer in cube:                       # first (top) passable value
            take = np.isnan(result) & ~np.isnan(layer)
            result[take] = layer[take]

    # forbidden cells (all inputs forbidden, or NaN produced) -> sentinel
    result[all_forbidden] = np.nan
    if np.issubdtype(ref_dtype, np.integer):
        info = np.iinfo(ref_dtype)
        clipped = np.clip(np.nan_to_num(result, nan=forbidden),
                          info.min, info.max)
        out = clipped.astype(ref_dtype)
    else:
        out = result.astype(ref_dtype)
    out[np.isnan(result)] = forbidden
    log.append(f"{operation}: combined {len(paths)} rasters")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(save_path) if save_path
                else work_dir / f"combined_{operation}_{uuid.uuid4().hex[:8]}.tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tiled_geotiff(out, ref["crs"], ref["transform"], forbidden, out_path)
    log.append(f"saved {out_path}")
    return str(out_path), log

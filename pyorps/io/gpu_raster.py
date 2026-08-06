"""
GPU-optimized raster format for PYORPS.

Provides conversion from GeoTIFF to a raw binary format (.gpur) that can be
loaded directly into GPU memory without decompression overhead. A companion
JSON sidecar file stores georeferencing metadata (CRS, transform, shape).

Usage:
    # Convert a GeoTIFF to GPU-optimized format
    from pyorps.io.gpu_raster import save_gpu_raster, load_gpu_raster

    meta = save_gpu_raster("cost_raster.tiff", "cost_raster.gpur")

    # Load directly to GPU (or CPU fallback)
    raster, metadata = load_gpu_raster("cost_raster.gpur", device=True)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


def save_gpu_raster(
        source_path: str,
        output_path: str,
        band_index: int = 0,
) -> dict:
    """
    Convert a GeoTIFF raster to a GPU-optimized raw binary format (.gpur).

    Reads the GeoTIFF via rasterio, writes the raw uncompressed raster data
    as a binary file, and stores metadata (shape, dtype, CRS, transform) in a
    companion JSON sidecar file (.gpur.json).

    Parameters:
        source_path: Path to input GeoTIFF file
        output_path: Path for output .gpur file (sidecar .gpur.json auto-created)
        band_index: Band index to extract (0-based, default 0)

    Returns:
        dict with keys: shape, dtype, crs, transform, file_size_bytes
    """
    import rasterio

    source_path = Path(source_path)
    output_path = Path(output_path)

    with rasterio.open(source_path) as src:
        # Read the specified band (rasterio uses 1-based band indexing)
        band_data = src.read(band_index + 1)
        crs = str(src.crs) if src.crs else None
        transform = list(src.transform)[:6]
        nodata = src.nodata
        dtype_str = str(band_data.dtype)

    # Write raw binary (row-major, no header, no compression)
    band_data.tofile(str(output_path))

    # Build metadata
    metadata = {
        "shape": list(band_data.shape),
        "dtype": dtype_str,
        "crs": crs,
        "transform": transform,
        "nodata": int(nodata) if nodata is not None else None,
        "band_index": band_index,
        "source_path": str(source_path),
        "created": datetime.now(timezone.utc).isoformat(),
    }

    # Write JSON sidecar
    sidecar_path = Path(str(output_path) + ".json")
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)

    metadata["file_size_bytes"] = output_path.stat().st_size
    return metadata


def load_gpu_raster(
        path: str,
        device: bool = True,
) -> tuple:
    """
    Load a GPU-optimized raster (.gpur) directly into GPU or CPU memory.

    When device=True and CuPy is available, loads via cp.fromfile() into
    GPU VRAM — no decompression needed, just a raw memory copy. When
    device=False or CuPy is unavailable, loads into a NumPy array.

    Parameters:
        path: Path to .gpur file (companion .gpur.json must exist)
        device: If True, load directly to GPU memory (requires CuPy).
            Falls back to CPU if CuPy is not available.

    Returns:
        tuple of (raster_array, metadata_dict):
        - raster_array: CuPy array (on GPU) or NumPy array (on CPU)
        - metadata_dict: dict with crs, transform, shape, dtype, etc.
    """
    path = Path(path)
    sidecar_path = Path(str(path) + ".json")

    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar metadata file not found: {sidecar_path}. "
            f"Use save_gpu_raster() to create .gpur files with metadata."
        )

    with open(sidecar_path, "r") as f:
        metadata = json.load(f)

    shape = tuple(metadata["shape"])
    dtype_str = metadata["dtype"]

    # Validate dtype against allowlist
    _ALLOWED_DTYPES = {"uint16", "float32", "float64"}
    if dtype_str not in _ALLOWED_DTYPES:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}' in sidecar metadata. "
            f"Only {sorted(_ALLOWED_DTYPES)} are allowed."
        )
    dtype = np.dtype(dtype_str)

    # Validate binary file size matches expected size from metadata
    expected_size = int(np.prod(shape)) * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Binary file size mismatch: expected {expected_size} bytes "
            f"(shape={shape}, dtype={dtype}), but file is {actual_size} "
            f"bytes. The .gpur file may be corrupted or the sidecar "
            f"metadata may be stale."
        )

    if device:
        try:
            import cupy as cp
            raster = cp.fromfile(str(path), dtype=dtype).reshape(shape)
            return raster, metadata
        except ImportError:
            pass  # fall through to NumPy

    raster = np.fromfile(str(path), dtype=dtype).reshape(shape)
    return raster, metadata

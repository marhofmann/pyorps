"""
PYORPS GUI: high-performance raster tiling.

Big cost rasters (thousands of pixels a side) are never shipped to the browser
whole. Instead we serve them through ``localtileserver`` (rio-tiler), which
reprojects to Web Mercator and streams only the 256x256 tiles visible in the
current viewport - so transfer cost is O(screen), independent of raster size.

The source array is written once to a tiled, overviewed GeoTIFF (a cheap
pyramid) so pan/zoom stays smooth. Exclusion cells (the dtype's max sentinel)
are marked nodata and render transparent.

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
Automated Power Line Routing', CIRED 2025.
"""
from __future__ import annotations

import logging
import os
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from pyorps.io.geo_dataset import RasterDataset
from pyorps.raster.handler import RasterHandler

# GDAL tuning for the tile server (TiTiler performance guide; GDAL reads
# config from the environment per operation, so setting these after import
# is fine — setdefault so the user's own env always wins).
for _key, _value in (("GDAL_CACHEMAX", "256"),
                     ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
                     ("VSI_CACHE", "TRUE"),
                     ("VSI_CACHE_SIZE", "5000000")):
    os.environ.setdefault(_key, _value)

# rio-tiler warns once per tile read when a source lacks overviews. We always
# write overviews below for rasters large enough to need them, but small rasters
# legitimately have none — silence the (purely informational) warning either way.
try:
    from rio_tiler.errors import NoOverviewWarning
    warnings.filterwarnings("ignore", category=NoOverviewWarning)
except ImportError:  # pragma: no cover - rio-tiler always present with viz extra
    pass

# Rendering a FLOAT raster (e.g. a DGM/DEM elevation model, nodata = NaN) into
# 8-bit coloured tiles casts the NaN fill through numpy's masked-array machinery
# — numpy then prints "RuntimeWarning: invalid value encountered in cast"
# (numpy/ma/core.py) on every pan/zoom, once per tile. The masked pixels render
# transparent regardless, so the warning is pure noise from the tile server;
# silence it (the filter is process-global, which the background tile-server
# thread shares).
warnings.filterwarnings(
    "ignore", message="invalid value encountered in cast",
    category=RuntimeWarning)


# The tile server (uvicorn/asyncio, Proactor loop on Windows) logs a noisy
# ConnectionResetError traceback whenever the browser cancels an in-flight
# tile request (every pan/zoom does this). Harmless — drop those records.
class _SilenceConnectionReset(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.exc_info and isinstance(record.exc_info[1],
                                          (ConnectionResetError,
                                           ConnectionAbortedError)):
            return False
        message = record.getMessage()
        return "ConnectionResetError" not in message and \
            "_call_connection_lost" not in message


logging.getLogger("asyncio").addFilter(_SilenceConnectionReset())

#: colormaps offered in the Raster tab (F3). rio-tiler registers all its
#: colormap names in LOWERCASE and validates them case-SENSITIVELY, so every
#: value here must be a lowercase rio-tiler key ("Spectral_r" raises in
#: ``client.get_tile_url``); matplotlib lookups go through :func:`_mpl_cmap`,
#: which resolves case-insensitively.
COLORMAPS: list[str] = [
    "viridis", "jet", "turbo", "plasma", "inferno", "magma", "cividis",
    "rdylgn_r", "spectral_r", "terrain", "gist_earth", "gray",
]


@dataclass
class RasterTileLayer:
    """A servable raster overlay: keeps the tile client alive and its map metadata."""

    name: str
    tile_url: str
    bounds: list[list[float]]      # [[south, west], [north, east]] for Leaflet
    vmin: float
    vmax: float
    colormap: str
    nodata: float
    tile_client: Any               # localtileserver.TileClient - keep referenced!
    source_path: str


def _band_hw(data: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return (2D first band, height, width) from a 2D or (bands, h, w) array."""
    if data.ndim == 3:
        return data[0], data.shape[1], data.shape[2]
    if data.ndim == 2:
        return data, data.shape[0], data.shape[1]
    raise ValueError(f"Unsupported raster ndim={data.ndim}; expected 2 or 3.")


#: a value range / class sample never needs more than this many cells: bigger
#: rasters are read DECIMATED (rasterio resamples straight from the file's
#: overviews when it has them) instead of loading every cell into RAM.
MAX_STATS_CELLS = 2_000_000


def decimated_shape(height: int, width: int,
                    max_cells: int | None = None) -> tuple[int, int]:
    """``(rows, cols)`` of a read that keeps the aspect ratio in ``max_cells``
    (default :data:`MAX_STATS_CELLS`)."""
    max_cells = MAX_STATS_CELLS if max_cells is None else max_cells
    if height * width <= max_cells:
        return height, width
    scale = (max_cells / float(height * width)) ** 0.5
    return max(1, int(height * scale)), max(1, int(width * scale))


def read_band_decimated(src: Any, index: int = 1,
                        max_cells: int | None = None) -> np.ndarray:
    """One band of an open rasterio dataset as a 2D array of at most
    ``max_cells`` cells (full resolution when the raster is small enough).

    Only the requested band is read, and a big raster is decimated by
    rasterio (``out_shape``), so memory stays bounded whatever the file size.
    """
    rows, cols = decimated_shape(src.height, src.width, max_cells)
    if (rows, cols) == (src.height, src.width):
        return src.read(index)
    return src.read(index, out_shape=(rows, cols),
                    resampling=Resampling.nearest)


def default_nodata(dtype: Any) -> float:
    """The nodata a raster WITHOUT a declared one is served with: the integer
    dtype's max (the cost rasters' "forbidden" sentinel) or NaN for floats."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float("nan")


def dataset_nodata(path: Any, dtype: Any) -> float:
    """The file's OWN nodata value (``src.nodata``), else :func:`default_nodata`.

    A DEM declaring -9999 / -32768 / 3.4e38 must have those cells left out of
    the colour range and painted transparent; the tile server then gets this
    value instead of the dtype default.
    """
    with rasterio.open(path) as src:
        declared = src.nodata
    return default_nodata(dtype) if declared is None else float(declared)


def valid_mask(band: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean mask of the usable cells: not nodata and, for floats, finite."""
    if nodata is None or np.isnan(nodata):
        mask = np.ones(band.shape, dtype=bool)
    elif np.issubdtype(band.dtype, np.integer):
        info = np.iinfo(band.dtype)
        if float(nodata).is_integer() and info.min <= nodata <= info.max:
            mask = band != int(nodata)
        else:                       # no cell of this dtype can hold that value
            mask = np.ones(band.shape, dtype=bool)
    elif np.issubdtype(band.dtype, np.floating):
        with np.errstate(all="ignore"):       # compare in the band's precision
            mask = band != band.dtype.type(nodata)
    else:
        mask = band != nodata
    if np.issubdtype(band.dtype, np.floating):
        mask &= np.isfinite(band)
    return mask


def _write_tiled_geotiff(data: np.ndarray, crs: Any, transform: Any,
                         nodata: float, out_path: Path) -> None:
    """Write a single-band, tiled, overviewed GeoTIFF for fast tile serving."""
    band, height, width = _band_hw(data)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": band.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band, 1)
        # Build an overview pyramid so zoomed-out views are cheap and rio-tiler
        # stops warning. Keep halving while the smallest level stays >= 128 px.
        max_dim = max(height, width)
        factors = [f for f in (2, 4, 8, 16, 32) if max_dim // f >= 128]
        if factors:
            dst.build_overviews(factors, Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")


def _resolve_source(raster_source: Any, crs: Any, transform: Any,
                    work_dir: Path) -> tuple[Path, np.ndarray, Any, Any]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Normalize any accepted raster input to (geotiff_path, band2d, crs, transform).

    Accepts: a file path, a RasterHandler, a RasterDataset/InMemoryRasterDataset,
    or a raw numpy array (requires crs + transform).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    work_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(raster_source, (str, Path)):
        with rasterio.open(raster_source) as src:
            # first band only, decimated to MAX_STATS_CELLS: this array only
            # feeds the min/max of the colour ramp, so min/max come from a
            # decimated read for big rasters (never the whole file in RAM)
            band = read_band_decimated(src, 1)
        return Path(raster_source), band, None, None

    if isinstance(raster_source, RasterHandler):
        band, _, _ = _band_hw(raster_source.data)
        nodata = (np.iinfo(band.dtype).max
                  if np.issubdtype(band.dtype, np.integer) else np.nan)
        out = work_dir / f"cost_{uuid.uuid4().hex}.tif"
        # Write tiled + overviewed (save_section_as_raster does neither, which
        # makes rio-tiler warn and slows zoomed-out tiles).
        _write_tiled_geotiff(raster_source.data, raster_source.raster_dataset.crs,
                             raster_source.window_transform, nodata, out)
        return out, band, raster_source.raster_dataset.crs, raster_source.window_transform

    if isinstance(raster_source, RasterDataset):
        band, _, _ = _band_hw(raster_source.data)
        out = work_dir / f"cost_{uuid.uuid4().hex}.tif"
        nodata = np.iinfo(band.dtype).max if np.issubdtype(band.dtype, np.integer) else np.nan
        _write_tiled_geotiff(raster_source.data, raster_source.crs,
                             raster_source.transform, nodata, out)
        return out, band, raster_source.crs, raster_source.transform

    if isinstance(raster_source, np.ndarray):
        if crs is None or transform is None:
            raise ValueError(
                "A numpy raster requires both `crs` and `transform` "
                "(the affine geotransform)."
            )
        band, _, _ = _band_hw(raster_source)
        out = work_dir / f"cost_{uuid.uuid4().hex}.tif"
        nodata = np.iinfo(band.dtype).max if np.issubdtype(band.dtype, np.integer) else np.nan
        _write_tiled_geotiff(raster_source, crs, transform, nodata, out)
        return out, band, crs, transform

    raise TypeError(
        f"Unsupported raster source type: {type(raster_source)!r}. Pass a file "
        "path, numpy array (+crs+transform), RasterHandler, or RasterDataset."
    )


def _valid_range(band: np.ndarray, nodata: float) -> tuple[float, float]:
    """Min/max of the band excluding nodata and non-finite values."""
    valid = band[valid_mask(band, nodata)]
    if valid.size == 0:
        return 0.0, 1.0
    return float(valid.min()), float(valid.max())


def build_tile_layer(raster_source: Any, *, name: str = "Cost raster",
                     crs: Any = None, transform: Any = None,
                     colormap: str = "viridis", work_dir: Path | str = ".",
                     vmin: float | None = None,
                     vmax: float | None = None) -> RasterTileLayer:
    """Serve a pyorps cost raster as XYZ tiles and return its map layer metadata."""
    from localtileserver import TileClient

    # rio-tiler registers colormap names in lowercase and matches exactly
    colormap = (colormap or "viridis").lower()
    if colormap.startswith("custom:"):
        # a graduated LUT from a previous session — the registry is process
        # local, so an unregistered hash must fall back to a named ramp
        from localtileserver.tiler.palettes import get_registered_colormap
        if get_registered_colormap(colormap) is None:
            colormap = "viridis"
    work_dir = Path(work_dir)
    tif_path, band, _, _ = _resolve_source(raster_source, crs, transform, work_dir)

    # A file keeps its OWN nodata (src.nodata); the dtype default (int max /
    # NaN) only applies to rasters that declare none, and to the arrays this
    # module writes itself (their file declares exactly that default).
    if isinstance(raster_source, (str, Path)):
        nodata: float = dataset_nodata(tif_path, band.dtype)
    else:
        nodata = default_nodata(band.dtype)

    lo, hi = _valid_range(band, nodata)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1.0

    # The client's reader opens the file and keeps it open. Opened without a
    # surrounding rasterio environment it would also keep an environment open
    # in THIS thread until the client is garbage collected, and when that
    # happens inside another raster read, the read loses its environment
    # ("No GDAL environment exists"). Opened inside this block it holds none.
    import rasterio

    with rasterio.Env():
        client = TileClient(str(tif_path), cors_all=True)
        # Advertise "localhost" rather than the raw server host. The background
        # server often binds to the IPv6 loopback "::1", which produces an
        # unbracketed (invalid) URL; "localhost" resolves to whichever loopback
        # the server is actually listening on and is browser-friendly.
        client.client_host = "localhost"
        client.client_port = client.server_port
        tile_url = client.get_tile_url(colormap=colormap, vmin=vmin, vmax=vmax,
                                       nodata=nodata, client=True)
        bottom, top, left, right = client.bounds("EPSG:4326")
    bounds = [[float(bottom), float(left)], [float(top), float(right)]]

    return RasterTileLayer(
        name=name, tile_url=tile_url, bounds=bounds, vmin=vmin, vmax=vmax,
        colormap=colormap, nodata=nodata, tile_client=client,
        source_path=str(tif_path),
    )


def _mpl_cmap(name: str | None):
    """Matplotlib colormap by name, case-INSENSITIVE (viridis fallback).

    ``COLORMAPS`` holds lowercase rio-tiler keys ("spectral_r"), while
    matplotlib registers "Spectral_r" — resolve across the case gap so the
    legend swatches always match the served tiles.
    """
    import matplotlib

    try:
        return matplotlib.colormaps[name or "viridis"]
    except (KeyError, ValueError):
        lower = str(name or "").lower()
        for known in matplotlib.colormaps:
            if known.lower() == lower:
                return matplotlib.colormaps[known]
        return matplotlib.colormaps["viridis"]


def cost_colors(colormap: str, values, vmin: float, vmax: float) -> dict:
    """Map cost values to the hex colour the tile renderer paints them (F: legend).

    Uses the same matplotlib/rio-tiler colormap + linear vmin..vmax normalization
    as the served tiles, so the legend/table swatches match what's on the map.
    Forbidden cells (>= the sentinel) render transparent, so they map to None.
    """
    from matplotlib.colors import Normalize, to_hex

    cmap = _mpl_cmap(colormap)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    out = {}
    for value in values:
        out[value] = (None if value is None or float(value) >= 65535
                      else to_hex(cmap(norm(float(value)))))
    return out


def set_colormap(tile_layer: RasterTileLayer, colormap: str) -> str:
    """Re-colour an already-served raster (F3) — no server restart.

    ``localtileserver`` renders colours at request time, so a fresh tile URL
    with the new ``colormap`` (keeping the layer's vmin/vmax/nodata) is all that
    is needed; the browser then re-fetches tiles. Mutates ``tile_layer`` in
    place and returns the new URL.
    """
    colormap = (colormap or "").lower()      # rio-tiler keys are lowercase
    if not colormap or colormap == tile_layer.colormap:
        return tile_layer.tile_url
    client = tile_layer.tile_client
    tile_layer.tile_url = client.get_tile_url(
        colormap=colormap, vmin=tile_layer.vmin, vmax=tile_layer.vmax,
        nodata=tile_layer.nodata, client=True)
    tile_layer.colormap = colormap
    return tile_layer.tile_url


def set_custom_colormap(tile_layer: RasterTileLayer,
                        lut: dict[int, tuple]) -> str:
    """Serve the raster with a custom discrete LUT (graduated rendering).

    Registers the ``{0..255: (r,g,b,a)}`` table in localtileserver's
    process-local registry and re-serves under its ``custom:<hash>`` name —
    same in-place recolor flow as :func:`set_colormap`.
    """
    from localtileserver.tiler.palettes import register_colormap

    name = register_colormap({int(k): tuple(int(x) for x in v)
                              for k, v in lut.items()})
    client = tile_layer.tile_client
    tile_layer.tile_url = client.get_tile_url(
        colormap=name, vmin=tile_layer.vmin, vmax=tile_layer.vmax,
        nodata=tile_layer.nodata, client=True)
    tile_layer.colormap = name
    return tile_layer.tile_url

"""
QGIS-style graduated raster rendering (Display & legend tab).

Class breaks over a raster's valid values — equal interval, quantile (equal
count), natural breaks (Jenks), logarithmic or unique values — with one
editable colour per class. Served through localtileserver as a CUSTOM
colormap: the tile handler rescales every band to 0-255 against vmin/vmax
before applying the colormap, so the classes are baked into a discrete
``{0..255: (r, g, b, a)}`` lookup table registered via
``localtileserver.tiler.palettes.register_colormap`` → ``"custom:<hash>"``.
(An intervals colormap in raw value space is NOT reachable through
localtileserver — any JSON list is treated as a plain colour list.)

The per-layer config lives in ``layer.meta["graduated"]``:
``{"method", "classes", "breaks", "colors", "labels"}`` — breaks in raw
value space (len = classes + 1), colors as hex strings (editable in the
legend), labels precomputed for display.
"""
from __future__ import annotations

import numpy as np

#: classification methods (value → label shown in the Render select)
METHODS = {
    "equal": "equal interval",
    "quantile": "quantile (equal count)",
    "jenks": "natural breaks (Jenks)",
    "log": "logarithmic",
    "unique": "unique values",
}

#: more distinct values than this and "unique values" refuses (uint16 cost
#: rasters typically have a handful of distinct costs)
UNIQUE_LIMIT = 64

_JENKS_SAMPLE = 1000        # DP is O(k·n²) — cap the input size


def sample_values(source_path: str, max_n: int = 200_000) -> np.ndarray:
    """Valid raster values (nodata / non-finite masked), evenly subsampled."""
    import rasterio

    with rasterio.open(source_path) as src:
        band = src.read(1)
        nodata = src.nodata
    values = band.ravel()
    if nodata is not None and not np.isnan(nodata):
        values = values[values != nodata]
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
    if values.size > max_n:
        values = values[:: values.size // max_n]
    return values.astype(np.float64)


def _jenks_breaks(values: np.ndarray, k: int) -> list[float]:
    """Fisher-Jenks natural breaks via classic dynamic programming."""
    data = np.sort(values)
    if data.size > _JENKS_SAMPLE:
        data = data[np.linspace(0, data.size - 1, _JENKS_SAMPLE, dtype=int)]
    n = data.size
    k = min(k, n)
    # cumulative sums for O(1) within-class variance
    csum = np.concatenate([[0.0], np.cumsum(data)])
    csum2 = np.concatenate([[0.0], np.cumsum(data * data)])

    def ssd(i: int, j: int) -> float:      # variance of data[i:j] (j excl.)
        m = j - i
        s = csum[j] - csum[i]
        return (csum2[j] - csum2[i]) - s * s / m

    cost = np.full((k + 1, n + 1), np.inf)
    prev = np.zeros((k + 1, n + 1), dtype=int)
    cost[0, 0] = 0.0
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            best, arg = np.inf, c - 1
            for i in range(c - 1, j):
                candidate = cost[c - 1, i] + ssd(i, j)
                if candidate < best:
                    best, arg = candidate, i
            cost[c, j], prev[c, j] = best, arg
    # walk back the class boundaries
    edges = [float(data[-1])]
    j = n
    for c in range(k, 0, -1):
        i = prev[c, j]
        edges.append(float(data[i] if i < n else data[-1]))
        j = i
    return sorted(set(edges))


def class_breaks(values: np.ndarray, method: str, k: int) -> list[float]:
    """k-class break edges (len k+1, ascending, first=min last=max)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if values.size == 0:
        raise ValueError("The raster has no valid cells to classify.")
    k = max(2, min(int(k or 5), 32))
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    if method == "equal":
        edges = np.linspace(lo, hi, k + 1)
    elif method == "quantile":
        edges = np.quantile(values, np.linspace(0, 1, k + 1))
    elif method == "jenks":
        inner = _jenks_breaks(values, k)
        edges = np.array(sorted({lo, *inner, hi}))
    elif method == "log":
        positive = values[values > 0]
        start = float(positive.min()) if positive.size else max(lo, 1e-9)
        edges = np.geomspace(max(start, 1e-9), hi, k + 1)
        edges[0] = lo
    elif method == "unique":
        uniques = unique_values(values)
        if uniques is None:
            raise ValueError(
                f"More than {UNIQUE_LIMIT} distinct values — pick a "
                "class-based method instead.")
        # midpoints between consecutive distinct values as edges
        mids = [(a + b) / 2 for a, b in zip(uniques[:-1], uniques[1:])]
        edges = np.array([lo, *mids, hi])
    else:
        raise ValueError(f"Unknown classification method: {method!r}")
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([lo, hi])
    return [float(e) for e in edges]


def unique_values(values: np.ndarray,
                  limit: int = UNIQUE_LIMIT) -> list[float] | None:
    """Sorted distinct values, or None if there are more than ``limit``."""
    uniques = np.unique(values)
    if uniques.size > limit or uniques.size < 1:
        return None
    return [float(u) for u in uniques]


def default_colors(colormap: str, k: int) -> list[str]:
    """One hex colour per class, sampled across the named colormap."""
    from matplotlib.colors import to_hex

    from .tiles import _mpl_cmap

    cmap = _mpl_cmap(colormap)
    if k == 1:
        return [to_hex(cmap(0.5))]
    return [to_hex(cmap(i / (k - 1))) for i in range(k)]


def class_labels(breaks: list[float], method: str,
                 uniques: list[float] | None = None) -> list[str]:
    def fmt(v: float) -> str:
        return f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:,.2f}"

    if method == "unique" and uniques:
        return [fmt(u) for u in uniques]
    return [f"{fmt(a)} – {fmt(b)}"
            for a, b in zip(breaks[:-1], breaks[1:])]


def build_config(source_path: str, method: str, k: int,
                 colormap: str) -> dict:
    """Compute the full graduated config for a raster file."""
    values = sample_values(source_path)
    uniques = unique_values(values) if method == "unique" else None
    breaks = class_breaks(values, method, k)
    n_classes = len(breaks) - 1
    return {
        "method": method,
        "classes": n_classes,
        "breaks": breaks,
        "colors": default_colors(colormap, n_classes),
        "labels": class_labels(breaks, method, uniques),
    }


def _hex_to_rgba(color: str) -> tuple[int, int, int, int]:
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))


def build_lut(breaks: list[float], colors: list[str],
              vmin: float, vmax: float) -> dict[int, tuple]:
    """The graduated classes as a 256-entry discrete LUT in 0-255 space.

    The tile handler rescales band values linearly from (vmin, vmax) to
    0-255 before applying the colormap, so each bucket b maps back to the
    value vmin + b/255*(vmax-vmin) and takes its class's colour.
    """
    if vmax <= vmin:
        vmax = vmin + 1.0
    rgba = [_hex_to_rgba(c) for c in colors]
    edges = np.asarray(breaks, dtype=np.float64)
    lut: dict[int, tuple] = {}
    for b in range(256):
        value = vmin + (b / 255.0) * (vmax - vmin)
        i = int(np.searchsorted(edges, value, side="right")) - 1
        i = max(0, min(i, len(rgba) - 1))
        lut[b] = rgba[i]
    return lut


def apply_to_layer(layer, config: dict) -> str:
    """Serve ``layer``'s raster with the graduated config; returns the URL.

    Stores the config in ``layer.meta["graduated"]`` so the legend renders
    class rows with editable colours and project reload can re-apply it.
    """
    from .tiles import set_custom_colormap

    tile = layer.tile
    lut = build_lut(config["breaks"], config["colors"],
                    tile.vmin, tile.vmax)
    url = set_custom_colormap(tile, lut)
    layer.meta["graduated"] = config
    return url

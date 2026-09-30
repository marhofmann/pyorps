"""Is the 46 EUR error floor spread over the field, or a handful of tiles?

If it is a few tiles straddling a big cost discontinuity, promoting just those
to exact float32 collapses the worst case for almost no extra size -- and the
worst case is the only statistic a bound can use.
"""
from __future__ import annotations

import numpy as np
import zlib

FIELD = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
         r"cired2026/raster/fields_min5m/field_PCC0.npz")
TILE = 256


def main():
    f = np.asarray(np.load(FIELD)["field"], dtype=np.float64)
    h, w = f.shape
    th, tw = -(-h // TILE), -(-w // TILE)
    pad = np.full((th * TILE, tw * TILE), np.nan)
    pad[:h, :w] = f
    pad[~np.isfinite(pad)] = np.nan
    b = pad.reshape(th, TILE, tw, TILE).transpose(0, 2, 1, 3)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(b, axis=(2, 3)), np.nanmax(b, axis=(2, 3))
    span = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    scale = span / 65535.0
    live = scale[span > 0]
    print(f"{th * tw} tiles, {live.size} with a non-zero span\n")
    print("per-tile understatement (EUR), distribution:")
    for q in (50, 90, 99, 99.9, 100):
        print(f"   p{q:<6} {np.percentile(live, q):9.3f}")
    print()
    for thr in (1.0, 2.0, 5.0, 10.0):
        n = int((live > thr).sum())
        print(f"   tiles above {thr:5.1f} EUR: {n:5d}  "
              f"({100 * n / live.size:5.2f} % of live tiles)")

    # cost of promoting the worst tiles to exact float32
    base = None
    print(f"\n{'promote worst':>15}{'extra MB':>11}{'total MB':>11}"
          f"{'worst under':>14}")
    print("-" * 52)
    order = np.argsort(scale.ravel())[::-1]
    q = np.clip(np.nan_to_num(
        np.floor((b - lo[:, :, None, None]) / np.where(
            scale > 0, scale, 1.0)[:, :, None, None]), nan=0.0),
        0, 65535).astype(np.uint16)
    qi = q.transpose(0, 2, 1, 3).reshape(th * TILE, tw * TILE)[:h, :w]
    d = np.ascontiguousarray(qi).astype(np.int32)
    res = np.empty_like(d)
    res[:, 0] = d[:, 0]
    res[:, 1:] = d[:, 1:] - d[:, :-1]
    r = (res & 0xFFFF).astype(np.uint16)
    v = r.view(np.uint8).reshape(-1, 2)
    base = len(zlib.compress(v.T.copy().tobytes(), 6)) + scale.nbytes * 2
    for k in (0, 1, 8, 32, 128, 512):
        keep = order[:k]
        extra = k * TILE * TILE * 4          # float32, uncompressed worst case
        flat = scale.ravel().copy()
        flat[keep] = 0.0
        rest = flat[flat > 0]
        worst = rest.max() if rest.size else 0.0
        print(f"{k:>15}{extra / 1e6:11.2f}{(base + extra) / 1e6:11.1f}"
              f"{worst:12.3f} EUR")


if __name__ == "__main__":
    main()

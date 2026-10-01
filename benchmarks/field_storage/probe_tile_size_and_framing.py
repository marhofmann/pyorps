"""Two questions the plan asserted without measuring.

1. TILE SIZE. The error a tile admits is its own span / 65535, so smaller
   tiles quantise finer. The plan fixed TILE=256 arbitrarily and then had to
   argue that 46 EUR was comfortably inside the MILP's 527 EUR proof
   tolerance. If a smaller tile buys an order of magnitude for a few percent
   of size, that argument stops being tight.

2. RANDOM ACCESS. The plan promises "decode lazily per tile", but the encoding
   it measured deltas across the FULL row and deflates the whole array as one
   stream -- so nothing can be decoded without decoding everything. This
   measures what independent per-tile-row framing actually costs.
"""
from __future__ import annotations

import time
import zlib

import numpy as np

FIELD = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
         r"cired2026/raster/fields_min5m/field_PCC0.npz")


def quantise(f, tile, bits=16):
    h, w = f.shape
    th, tw = -(-h // tile), -(-w // tile)
    pad = np.full((th * tile, tw * tile), np.nan, dtype=np.float64)
    pad[:h, :w] = f
    pad[~np.isfinite(pad)] = np.nan
    b = pad.reshape(th, tile, tw, tile).transpose(0, 2, 1, 3)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(b, axis=(2, 3)), np.nanmax(b, axis=(2, 3))
    span = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    scale = np.where(span > 0, span / (2 ** bits - 1), 1.0)
    q = np.clip(np.nan_to_num(
        np.floor((b - lo[:, :, None, None]) / scale[:, :, None, None]),
        nan=0.0), 0, 2 ** bits - 1).astype(np.uint16)
    qi = q.transpose(0, 2, 1, 3).reshape(th * tile, tw * tile)[:h, :w]
    worst = float(scale[span > 0].max()) if (span > 0).any() else 0.0
    side = (lo.astype(np.float64).nbytes + scale.astype(np.float64).nbytes)
    return np.ascontiguousarray(qi), worst, side


def pack(qi, delta_axis_full=True, tile=None):
    """delta + byte shuffle + deflate over the whole plane."""
    d = qi.astype(np.int32)
    res = np.empty_like(d)
    res[:, 0] = d[:, 0]
    res[:, 1:] = d[:, 1:] - d[:, :-1]
    r = (res & 0xFFFF).astype(np.uint16)
    v = r.view(np.uint8).reshape(-1, 2)
    return len(zlib.compress(v.T.copy().tobytes(), 6))


def pack_framed(qi, tile):
    """Each tile-ROW is its own delta + shuffle + deflate frame, so a reader
    can seek to one stripe and decode only that."""
    h = qi.shape[0]
    total = 0
    for r0 in range(0, h, tile):
        blk = qi[r0:r0 + tile]
        d = blk.astype(np.int32)
        res = np.empty_like(d)
        res[:, 0] = d[:, 0]
        res[:, 1:] = d[:, 1:] - d[:, :-1]
        r = (res & 0xFFFF).astype(np.uint16)
        v = r.view(np.uint8).reshape(-1, 2)
        total += len(zlib.compress(v.T.copy().tobytes(), 6)) + 8
    return total


def main():
    z = np.load(FIELD)
    f = np.asarray(z["field"], dtype=np.float64)
    n = f.size
    print(f"{FIELD.split('/')[-1]}  {f.shape}  {n / 1e6:.1f} M cells, "
          f"raw float32 {n * 4 / 1e6:.1f} MB\n")

    print(f"{'tile':>6}{'one stream MB':>16}{'framed MB':>12}{'framing cost':>14}"
          f"{'side MB':>10}{'worst under':>14}")
    print("-" * 74)
    for tile in (32, 64, 128, 256, 512):
        qi, worst, side = quantise(f, tile)
        a = pack(qi)
        b = pack_framed(qi, tile)
        print(f"{tile:>6}{(a + side) / 1e6:16.1f}{(b + side) / 1e6:12.1f}"
              f"{100 * (b - a) / a:13.1f} %{side / 1e6:10.2f}"
              f"{worst:12.2f} EUR")
        del qi

    print("\nAccumulation: an objective that SUMS k field lookups inherits")
    print("k x the per-lookup understatement.")
    for tile, per in ((256, None), (64, None), (32, None)):
        qi, worst, _ = quantise(f, tile)
        print(f"  tile {tile:>3}: {worst:8.2f} EUR/lookup  ->  "
              f"2 lookups {2 * worst:9.2f}   10 lookups {10 * worst:9.2f}"
              f"   (MILP proof tolerance 527 EUR)")
        del qi


if __name__ == "__main__":
    main()

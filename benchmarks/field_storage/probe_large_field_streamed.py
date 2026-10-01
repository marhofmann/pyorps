"""Does the 16x saving hold on the field that actually dominates the cache?

The 41.8 M-cell probe was run on a route-cost field. The HV fields are 240 M
cells on a different cost surface and are 3.4 GB of the 5.6 GB total, so the
headline number is theirs, not the small one's. Streamed in row stripes so the
peak stays near 1 GB instead of the ~6 GB a whole-array float64 pass would take.
"""
from __future__ import annotations

import json
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

FIELD = Path(r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
             r"cired2026/raster/hv_fields/hv_field_PCC0.npz")
TILE = 256


def stripe_encode(dist2d, bits=16):
    """Quantise one stripe of whole tile-rows; return payload + max under."""
    h, w = dist2d.shape
    tw = -(-w // TILE)
    pad = np.full((h, tw * TILE), np.nan, dtype=np.float64)
    pad[:, :w] = dist2d
    pad[~np.isfinite(pad)] = np.nan
    th = h // TILE
    b = pad.reshape(th, TILE, tw, TILE).transpose(0, 2, 1, 3)
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(b, axis=(2, 3))
        hi = np.nanmax(b, axis=(2, 3))
    span = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    scale = np.where(span > 0, span / (2 ** bits - 1), 1.0)
    q = np.floor((b - lo[:, :, None, None]) / scale[:, :, None, None])
    q = np.clip(np.nan_to_num(q, nan=0.0), 0, 2 ** bits - 1)
    under = float(np.nanmax(span / (2 ** bits - 1))) if np.isfinite(
        span).any() else 0.0
    qi = q.astype(np.uint16).transpose(0, 2, 1, 3).reshape(h, tw * TILE)[:, :w]
    d = qi.astype(np.int32)
    res = np.empty_like(d)
    res[:, 0] = d[:, 0]
    res[:, 1:] = d[:, 1:] - d[:, :-1]
    r16 = (res & 0xFFFF).astype(np.uint16)
    v = r16.view(np.uint8).reshape(-1, 2)
    side = lo.astype(np.float64).tobytes() + scale.astype(np.float64).tobytes()
    return v.T.copy().tobytes() + side, under


def main():
    z = zipfile.ZipFile(FIELD)
    meta = json.loads(str(np.load(FIELD)["meta"]))
    rows, cols = meta["rows"], meta["cols"]
    print(f"{FIELD.name}: {rows} x {cols} = {rows * cols:,} cells, "
          f"neighborhood {meta['neighborhood']}, has_paths {meta['has_paths']}")
    on_disk = sum(i.compress_size for i in z.infolist())
    print(f"on disk now {on_disk / 1e6:,.1f} MB "
          f"({on_disk / (rows * cols):.2f} B/cell)\n")

    dist = np.load(FIELD, mmap_mode="r")["dist"]
    STRIPE = TILE * 8
    comp = 0
    worst = 0.0
    t0 = time.perf_counter()
    for r0 in range(0, rows, STRIPE):
        r1 = min(r0 + STRIPE, rows)
        n = ((r1 - r0) // TILE) * TILE or (r1 - r0)
        chunk = np.asarray(dist[r0 * cols:(r0 + n) * cols],
                           dtype=np.float64).reshape(n, cols)
        payload, under = stripe_encode(chunk)
        comp += len(zlib.compress(payload, 6))
        worst = max(worst, under)
        del chunk, payload
    enc_s = time.perf_counter() - t0

    raw = rows * cols * 4
    mask = None
    print(f"{'encoding':<46}{'MB':>10}{'B/cell':>9}{'of today':>10}")
    print("-" * 76)
    print(f"{'today: float32 dist + uint8 pred, stored raw':<46}"
          f"{on_disk / 1e6:10.1f}{on_disk / (rows * cols):9.2f}{100.0:9.1f} %")
    print(f"{'dist alone, float32 raw':<46}{raw / 1e6:10.1f}"
          f"{4.0:9.2f}{100 * raw / on_disk:9.1f} %")
    print(f"{'quantised + delta + shuffle + deflate, no pred':<46}"
          f"{comp / 1e6:10.1f}{comp / (rows * cols):9.2f}"
          f"{100 * comp / on_disk:9.1f} %")
    print(f"\nworst one-sided understatement {worst:.2f} EUR "
          f"(encode {enc_s:.0f} s for the whole field)")


if __name__ == "__main__":
    main()

"""Pick the operating point: bit depth vs error vs size, on every field kind.

The error that matters is one-sided -- the codec rounds DOWN, so a decoded
value is always <= the true one and any screening bound built on it stays a
bound. What changes with bit depth is how loose that bound gets.
"""
from __future__ import annotations

import glob
import os
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
            r"cired2026/raster")
TILE = 256


def encode(f2d, bits):
    h, w = f2d.shape
    th, tw = -(-h // TILE), -(-w // TILE)
    pad = np.full((th * TILE, tw * TILE), np.nan, dtype=np.float64)
    pad[:h, :w] = f2d
    pad[~np.isfinite(pad)] = np.nan
    b = pad.reshape(th, TILE, tw, TILE).transpose(0, 2, 1, 3)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(b, axis=(2, 3)), np.nanmax(b, axis=(2, 3))
    span = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    scale = np.where(span > 0, span / (2 ** bits - 1), 1.0)
    q = np.clip(np.nan_to_num(
        np.floor((b - lo[:, :, None, None]) / scale[:, :, None, None]),
        nan=0.0), 0, 2 ** bits - 1)
    dt = np.uint8 if bits <= 8 else np.uint16
    qi = q.astype(dt).transpose(0, 2, 1, 3).reshape(th * TILE, tw * TILE)[:h, :w]
    d = qi.astype(np.int32)
    res = np.empty_like(d)
    res[:, 0] = d[:, 0]
    res[:, 1:] = d[:, 1:] - d[:, :-1]
    rr = (res & (2 ** (8 * dt().itemsize) - 1)).astype(dt)
    v = rr.view(np.uint8).reshape(-1, dt().itemsize)
    side = lo.astype(np.float64).tobytes() + scale.astype(np.float64).tobytes()
    payload = v.T.copy().tobytes() + side
    return len(zlib.compress(payload, 6)), float(np.nanmax(scale[span > 0])) \
        if (span > 0).any() else 0.0


def load(path):
    z = np.load(path)
    key = "field" if "field" in z.files else "dist"
    a = z[key]
    if a.ndim == 1:
        import json
        m = json.loads(str(z["meta"]))
        a = a.reshape(m["rows"], m["cols"])
    return np.asarray(a, dtype=np.float64)


def main():
    kinds = [("fields", "route cost, 10 m stride"),
             ("fields_min5m", "route cost, 5 m min-pooled"),
             ("fields_len5m", "route length, 5 m min-pooled"),
             ("hv_fields", "HV cost, full 1 m")]
    grand_now = grand_new = 0.0
    for d, what in kinds:
        fs = sorted(glob.glob(str(ROOT / d / "*.npz")))
        if not fs:
            continue
        n_files = len(fs)
        dir_now = sum(os.path.getsize(f) for f in fs)
        f2d = load(fs[0])
        cells = f2d.size
        one_now = os.path.getsize(fs[0])
        print(f"\n{d}  --  {what}")
        print(f"  {n_files} files, {cells / 1e6:.1f} M cells each, "
              f"{dir_now / 1e6:,.0f} MB on disk "
              f"({one_now / cells:.2f} B/cell)")
        print(f"    {'bits':>5}{'MB/field':>11}{'B/cell':>9}{'vs now':>9}"
              f"{'worst understatement':>24}")
        best16 = None
        for bits in (8, 12, 16):
            size, step = encode(f2d, bits)
            if bits == 16:
                best16 = size
            print(f"    {bits:>5}{size / 1e6:11.1f}{size / cells:9.3f}"
                  f"{one_now / size:8.1f}x{step:>20.2f} EUR"
                  if "len" not in d else
                  f"    {bits:>5}{size / 1e6:11.1f}{size / cells:9.3f}"
                  f"{one_now / size:8.1f}x{step:>22.3f} m")
        grand_now += dir_now
        grand_new += best16 * n_files
        del f2d

    print(f"\n{'':-<66}")
    print(f"whole preprocessing cache, 16-bit:  "
          f"{grand_now / 1e6:,.0f} MB  ->  {grand_new / 1e6:,.0f} MB"
          f"   ({grand_now / grand_new:.1f}x, "
          f"{(grand_now - grand_new) / 1e9:.1f} GB freed)")


if __name__ == "__main__":
    main()

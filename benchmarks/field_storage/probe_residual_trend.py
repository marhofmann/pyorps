"""Can the error floor be removed by quantising a RESIDUAL instead of the field?

A cost field is dominated by one trend: it grows roughly linearly with distance
from its origin. That trend is what fills each tile's dynamic range, and the
range is what sets the 16-bit step. Subtract an analytic trend the decoder can
recompute for free and only the deviation has to be stored.

Soundness is preserved for any slope c: decoded = c*euclid + floor_quantise(
dist - c*euclid) <= c*euclid + (dist - c*euclid) = dist. The stored value is
still never above the true one, so a bound built on it is still a bound.
"""
from __future__ import annotations

import numpy as np
import zlib

FIELD = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
         r"cired2026/raster/fields_min5m/field_PCC0.npz")
TILE = 256


def encode(vals2d, tile=TILE, bits=16):
    h, w = vals2d.shape
    th, tw = -(-h // tile), -(-w // tile)
    pad = np.full((th * tile, tw * tile), np.nan)
    pad[:h, :w] = vals2d
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
    d = np.ascontiguousarray(qi).astype(np.int32)
    res = np.empty_like(d)
    res[:, 0] = d[:, 0]
    res[:, 1:] = d[:, 1:] - d[:, :-1]
    r = (res & 0xFFFF).astype(np.uint16)
    v = r.view(np.uint8).reshape(-1, 2)
    size = len(zlib.compress(v.T.copy().tobytes(), 6)) + 2 * scale.nbytes
    worst = float(scale[span > 0].max()) if (span > 0).any() else 0.0
    return size, worst


def main():
    z = np.load(FIELD)
    f = np.asarray(z["field"], dtype=np.float64)
    h, w = f.shape
    fin = np.isfinite(f)
    o = int(np.nanargmin(np.where(fin, f, np.inf)))
    orow, ocol = divmod(o, w)
    print(f"field {f.shape}, origin cell ({orow}, {ocol}), "
          f"min {f[fin].min():.1f}, max {f[fin].max():,.0f}")

    rr, cc = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing="ij")
    euclid = np.hypot(rr - orow, cc - ocol)
    del rr, cc

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(fin & (euclid > 0), f / np.maximum(euclid, 1e-9),
                         np.nan)
    rmin = np.nanmin(ratio)
    rmed = np.nanmedian(ratio)
    print(f"dist/euclid over the field: min {rmin:.1f}, median {rmed:.1f}, "
          f"max {np.nanmax(ratio):,.0f}  (EUR per cell of straight line)")
    del ratio

    base_size, base_worst = encode(np.where(fin, f, np.nan))
    print(f"\n{'trend removed':<34}{'MB':>9}{'vs plain':>10}"
          f"{'worst understatement':>23}")
    print("-" * 78)
    print(f"{'none (the plan as written)':<34}{base_size / 1e6:9.1f}"
          f"{100.0:9.0f} %{base_worst:18.2f} EUR")

    for name, c in (("c = min(dist/euclid)", rmin),
                    ("c = median(dist/euclid)", rmed),
                    ("c = 0.5 x median", 0.5 * rmed),
                    ("c = 2 x median", 2.0 * rmed)):
        resid = np.where(fin, f - c * euclid, np.nan)
        size, worst = encode(resid)
        print(f"{name:<34}{size / 1e6:9.1f}{100 * size / base_size:9.0f} %"
              f"{worst:18.2f} EUR")
        del resid

    print("\n(decoder recomputes c*euclid analytically; only c, the origin")
    print(" cell and the quantised residual are stored)")


if __name__ == "__main__":
    main()

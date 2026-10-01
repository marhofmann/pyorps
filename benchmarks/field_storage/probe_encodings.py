"""How small can a settled cost field be made, and what does it cost to read?

Measured on a real cached field. Every candidate encoding is checked for the
property the screening stage actually needs: a stored value must never be
ABOVE the true one, because the bound it feeds must never overestimate. So the
quantised variants round DOWN and the probe reports the one-sided error.

stdlib only (no zstd/blosc in this venv); zstd would reach similar ratios at
roughly 5-10x the speed, which is worth stating but not worth installing here.
"""
from __future__ import annotations

import lzma
import time
import zlib
from pathlib import Path

import numpy as np

FIELD = Path(r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
             r"cired2026/raster/fields_min5m/field_PCC0.npz")
TILE = 256


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def shuffle_bytes(a: np.ndarray) -> bytes:
    """Byte-transpose: all byte-0s, then all byte-1s, ... The classic float
    trick -- exponent bytes of a smooth field are nearly constant, so they
    collapse, while the noisy mantissa bytes stay together."""
    v = a.view(np.uint8).reshape(-1, a.dtype.itemsize)
    return v.T.copy().tobytes()


def tile_quantise(f: np.ndarray, bits: int = 16):
    """Per-tile affine quantisation, rounded DOWN.

    A cost field is smooth, so the dynamic range inside a 256x256 tile is a
    tiny fraction of the field's global range -- which is what buys the
    precision back after dropping to 16 bits. Rounding down makes the decoded
    value a valid lower bound on the true cost, so a screening bound built on
    it stays a bound.
    """
    h, w = f.shape
    th, tw = -(-h // TILE), -(-w // TILE)
    pad = np.full((th * TILE, tw * TILE), np.nan, dtype=np.float64)
    pad[:h, :w] = f
    # unreachable cells arrive as +inf, and nanmin/nanmax do NOT filter those --
    # a single inf blows its tile's span up to inf and silently destroys the
    # scale for every other cell in that tile.
    pad[~np.isfinite(pad)] = np.nan
    blocks = pad.reshape(th, TILE, tw, TILE).transpose(0, 2, 1, 3)
    lo = np.nanmin(blocks, axis=(2, 3))
    hi = np.nanmax(blocks, axis=(2, 3))
    span = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    scale = np.where(span > 0, span / (2 ** bits - 1), 1.0)
    q = np.floor((blocks - lo[:, :, None, None]) / scale[:, :, None, None])
    q = np.clip(np.nan_to_num(q, nan=0.0), 0, 2 ** bits - 1)
    dtype = np.uint16 if bits == 16 else np.uint32
    packed = q.astype(dtype).transpose(0, 2, 1, 3).reshape(th * TILE, tw * TILE)
    deq = (q * scale[:, :, None, None] + lo[:, :, None, None])
    deq = deq.transpose(0, 2, 1, 3).reshape(th * TILE, tw * TILE)[:h, :w]
    side = lo.astype(np.float64).tobytes() + scale.astype(np.float64).tobytes()
    return packed[:h, :w].copy(), deq, side


def main():
    z = np.load(FIELD)
    f = z["field"].astype(np.float32)
    h, w = f.shape
    n = f.size
    fin = np.isfinite(f)
    print(f"field {h} x {w} = {n:,} cells,  {100 * fin.mean():.1f} % reachable")
    print(f"range {np.nanmin(f[fin]):,.0f} .. {np.nanmax(f[fin]):,.0f}\n")

    raw = n * 4
    results = []

    def record(name, payload, deq=None, dec_s=None, note=""):
        size = len(payload) if isinstance(payload, bytes) else payload
        err = ""
        if deq is not None:
            d = (f[fin].astype(np.float64) - deq[fin])
            err = (f"max under {d.max():8.2f} EUR, "
                   f"over {max(-d.min(), 0):6.2f}, "
                   f"rel {abs(d).max() / np.nanmax(f[fin]):.2e}")
        results.append((name, size, size / raw, dec_s, err, note))

    # 0 -- the current on-disk format, for reference
    record("float32 raw (what save(compress=False) writes)", raw)

    # 1 -- deflate straight on the float32 buffer: what npz does today
    fb = f.tobytes()
    c1, t1 = timed(zlib.compress, fb, 6)
    d1 = timed(zlib.decompress, c1)[1]
    record("float32 + deflate-6  (today's npz)", c1, dec_s=d1)

    # 2 -- byte shuffle first
    sh, _ = timed(shuffle_bytes, f)
    c2, _ = timed(zlib.compress, sh, 6)
    d2 = timed(zlib.decompress, c2)[1]
    record("float32 + byte-shuffle + deflate-6", c2, dec_s=d2)

    # 3 -- per-tile 16-bit quantisation, rounded down
    (q16, deq, side), tq = timed(tile_quantise, f.astype(np.float64), 16)
    body = q16.tobytes()
    c3, _ = timed(zlib.compress, body + side, 6)
    d3 = timed(zlib.decompress, c3)[1]
    record("uint16 per-tile affine + deflate-6", c3, deq=deq, dec_s=d3,
           note=f"quantise {tq:.1f} s")

    # 4 -- same, but shuffle the two bytes apart first
    sh16 = shuffle_bytes(q16)
    c4, _ = timed(zlib.compress, sh16 + side, 6)
    d4 = timed(zlib.decompress, c4)[1]
    record("uint16 per-tile + byte-shuffle + deflate-6", c4, deq=deq, dec_s=d4)

    # 5 -- predictive: horizontal delta of the quantised integers.  A field is
    #      smooth, so the residual is small and highly skewed.
    dq = q16.astype(np.int32)
    res = np.empty_like(dq)
    res[:, 0] = dq[:, 0]
    res[:, 1:] = dq[:, 1:] - dq[:, :-1]
    res16 = (res & 0xFFFF).astype(np.uint16)
    c5, _ = timed(zlib.compress, shuffle_bytes(res16) + side, 6)
    d5 = timed(zlib.decompress, c5)[1]
    record("uint16 per-tile + horiz. delta + shuffle + deflate", c5, deq=deq,
           dec_s=d5)

    # 6 -- what a stronger entropy coder reaches on the best representation
    c6, t6 = timed(lzma.compress, shuffle_bytes(res16) + side,
                   format=lzma.FORMAT_RAW,
                   filters=[{"id": lzma.FILTER_LZMA2, "preset": 1}])
    record("   ... same, LZMA2 preset 1 (ceiling probe)", c6, deq=deq,
           note=f"encode {t6:.0f} s")

    print(f"{'encoding':<52}{'MB':>9}{'of raw':>9}{'decode':>9}  error")
    print("-" * 110)
    for name, size, frac, dec, err, note in results:
        d = f"{dec:6.2f} s" if dec else "     -- "
        print(f"{name:<52}{size / 1e6:9.1f}{frac * 100:8.1f} %{d:>9}  "
              f"{err}{('  ' + note) if note else ''}")

    print(f"\nreachability mask, packed 1 bit/cell + deflate: "
          f"{len(zlib.compress(np.packbits(fin).tobytes(), 6)) / 1e6:.2f} MB")
    print(f"predecessor plane (uint8) at this size would add "
          f"{n / 1e6:.0f} MB raw")


if __name__ == "__main__":
    main()

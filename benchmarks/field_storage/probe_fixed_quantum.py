"""The prior-art review's central claim, tested directly.

My codec maps each tile's RANGE onto a fixed 16-bit budget, so the error is a
by-product of the tile's dynamic range -- which is why shrinking tiles,
promoting outliers and removing the radial trend all failed: they all attack
the range.

SZ-class compressors do the opposite: fix the QUANTUM (bin width 2*eb) and let
the bit count fall out of entropy coding. The error bound is then a knob, not a
measurement, and is independent of dynamic range entirely.

That reordering needs no new dependency to test: quantise with a global fixed
quantum, then delta + shuffle + deflate exactly as before. If the review is
right, error becomes a free parameter and size grows only slowly as eb shrinks.
"""
from __future__ import annotations

import zlib

import numpy as np

FIELD = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
         r"cired2026/raster/fields_min5m/field_PCC0.npz")


def pack_fixed_quantum(f, eb, order=1):
    """floor(x / q) with q = 2*eb, then delta, shuffle, deflate.

    Rounding down keeps the decoded value <= the true one, so this is still a
    one-sided lower bound; the reader recovers an upper bound by adding q.
    `order` picks the predictor: 1 = horizontal delta, 2 = second difference
    along rows (a cheap stand-in for a 2-D Lorenzo predictor).
    """
    q = 2.0 * eb
    fin = np.isfinite(f)
    qi = np.zeros(f.shape, dtype=np.int64)
    qi[fin] = np.floor(f[fin] / q).astype(np.int64)
    d = qi
    for _ in range(order):
        nd = np.empty_like(d)
        nd[:, 0] = d[:, 0]
        nd[:, 1:] = d[:, 1:] - d[:, :-1]
        d = nd
    # zigzag so small negatives stay small unsigned, then 4 bytes, shuffled
    zz = ((d << 1) ^ (d >> 63)).astype(np.uint64).astype(np.uint32)
    v = zz.view(np.uint8).reshape(-1, 4)
    body = zlib.compress(v.T.copy().tobytes(), 6)
    mask = zlib.compress(np.packbits(fin).tobytes(), 6)
    return len(body) + len(mask)


def main():
    f = np.asarray(np.load(FIELD)["field"], dtype=np.float64)
    n = f.size
    fin = np.isfinite(f)
    print(f"{f.shape} = {n / 1e6:.1f} M cells, range 0 .. "
          f"{f[fin].max():,.0f} EUR")
    print(f"today's npz (deflated float32): 121.2 MB")
    print(f"my tiled 16-bit codec:           27.1 MB "
          f"at a MEASURED 45.89 EUR worst case\n")

    print(f"{'eb (EUR)':>10}{'guaranteed':>13}{'delta MB':>11}{'B/cell':>9}"
          f"{'2nd-diff MB':>14}{'vs my codec':>13}")
    print("-" * 70)
    for eb in (50.0, 25.0, 10.0, 5.0, 1.0, 0.5):
        a = pack_fixed_quantum(f, eb, order=1)
        b = pack_fixed_quantum(f, eb, order=2)
        best = min(a, b)
        print(f"{eb:10.1f}{2 * eb:12.1f} {a / 1e6:10.1f}{a / n:9.3f}"
              f"{b / 1e6:14.1f}{best / 27.1e6:12.2f}x")

    print("\n'guaranteed' is the worst-case understatement by construction,")
    print("independent of the field's dynamic range -- that is the whole point.")


if __name__ == "__main__":
    main()

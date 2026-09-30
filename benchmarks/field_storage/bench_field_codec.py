"""The shipped fixed-quantum codec, measured against the right baseline.

The eight ``probe_*`` scripts next to this one are the exploration that
produced ``pyorps.io.field_codec``, including three levers that do not
work and one probe that was retracted for an accounting bug. This is the
implementation, measured, with the corrections baked in:

* the baseline is **deflated float32**, not the uncompressed save. Just
  compressing is 1.65x free, and quoting against the raw file is what
  turned a 6.2x codec into a "9.1x" headline;
* the error bound is checked, not assumed. Every row reports the worst
  observed understatement next to the guarantee;
* the reachability mask is part of the format, so the unreachable-cell
  column is a real check and not a footnote.

Reads the CIRED 2026 field cache when it is present and falls back to a
synthetic field of the same size and range when it is not.

    python benchmarks/field_storage/bench_field_codec.py
"""
from __future__ import annotations

import time
import zlib

import numpy as np

from pyorps.io.field_codec import decode_field, encode_field

FIELD = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
         r"cired2026/raster/fields_min5m/field_PCC0.npz")


def load():
    try:
        f = np.asarray(np.load(FIELD)["field"], dtype=np.float64)
        return f, "CIRED 2026 field_PCC0"
    except Exception:                                    # noqa: BLE001
        pass
    rows, cols = 2000, 2500
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:rows, 0:cols]
    f = (np.hypot(yy - rows / 2, xx - cols / 2) * 900.0
         + 6e4 * np.sin(xx / 61.0) + 4e4 * np.cos(yy / 47.0)
         + rng.random((rows, cols)) * 8e3)
    f -= f.min()
    f[400:700, 900:1400] = np.inf                        # an exclusion
    return f, "synthetic"


def main():
    f, kind = load()
    finite = np.isfinite(f)
    n = f.size
    print(f"field: {kind}, {f.shape} = {n / 1e6:.1f} M cells, "
          f"0 .. {f[finite].max():,.0f} EUR, "
          f"{100 * (~finite).mean():.1f} % unreachable\n")

    raw = np.where(finite, f, np.inf).astype(np.float32)
    t0 = time.perf_counter()
    deflated = zlib.compress(raw.tobytes(), 6)
    t_base = time.perf_counter() - t0
    print(f"  float32, uncompressed : {raw.nbytes / 1e6:8.1f} MB")
    print(f"  float32, deflated     : {len(deflated) / 1e6:8.1f} MB "
          f"({raw.nbytes / len(deflated):.2f}x free, {t_base:.1f} s)")
    print("  ^ THE BASELINE. Quoting against the uncompressed file is what "
          "turned 6.2x into '9.1x'.\n")

    print(f"{'guarantee':>11}{'MB':>9}{'B/cell':>9}{'vs deflate':>12}"
          f"{'worst under':>13}{'holds':>7}{'enc s':>8}{'dec s':>8}")
    print("-" * 78)
    for eb in (100.0, 10.0, 1.0, 0.1):
        t0 = time.perf_counter()
        payload, meta = encode_field(f, error_bound=eb)
        t_enc = time.perf_counter() - t0
        size = sum(v.nbytes for v in payload.values())
        t0 = time.perf_counter()
        d = decode_field(payload, meta)
        t_dec = time.perf_counter() - t0
        under = f[finite] - d.lower[finite]
        holds = bool((under >= 0).all() and under.max() <= eb * (1 + 1e-9)
                     and (d.upper[finite] >= f[finite]).all()
                     and np.array_equal(d.reachable, finite))
        print(f"{eb:11.2f}{size / 1e6:9.1f}{size / n:9.3f}"
              f"{size / len(deflated):11.2f}x{under.max():13.4f}"
              f"{str(holds):>7}{t_enc:8.1f}{t_dec:8.1f}")

    print("\nThe error is a PARAMETER, not a measurement: it does not depend")
    print("on the field's dynamic range. That is what fixing the QUANTUM")
    print("buys, and it is why shrinking tiles (-5 % error for +44 % size),")
    print("promoting outlier tiles and removing a radial trend (+19 % size,")
    print("no gain) all failed -- all three attacked the RANGE.")

    print("\n== decode is slower than deflated float32, and that is fine ==")
    t0 = time.perf_counter()
    np.frombuffer(zlib.decompress(deflated), dtype=np.float32)
    t_read = time.perf_counter() - t0
    payload, meta = encode_field(f, error_bound=1.0)
    t0 = time.perf_counter()
    decode_field(payload, meta)
    t_dec = time.perf_counter() - t0
    print(f"  deflated float32 read : {t_read:6.2f} s")
    print(f"  codec decode          : {t_dec:6.2f} s "
          f"({t_dec / max(t_read, 1e-9):.1f}x)")
    print("  Timing only zlib.decompress -- and omitting unshuffle, the")
    print("  cumulative sum and the dequantise -- is what produced the")
    print("  earlier '2.7x faster' claim. The codec buys SIZE, not speed.")

    print("\n== tile size does not move the error, only the size ==")
    for tile in (64, 128, 256, 512):
        payload, meta = encode_field(f, error_bound=1.0, tile=tile)
        size = sum(v.nbytes for v in payload.values())
        d = decode_field(payload, meta)
        print(f"  tile {tile:>4}: {size / 1e6:7.1f} MB, worst under "
              f"{(f[finite] - d.lower[finite]).max():.4f} EUR")
    print("  Identical error at every tile size -- under a fixed quantum")
    print("  there is no outlier tile to promote, which is why that lever")
    print("  (real against a RANGE-based codec) is not implemented here.")


if __name__ == "__main__":
    main()

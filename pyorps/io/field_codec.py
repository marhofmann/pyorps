"""Lossy-but-bounded storage for settled cost fields.

Implements section 2 / 3.5 of
``docs/superpowers/plans/2026-09-20-generalized-free-siting-facility-chains.md``
(revision 3), including the corrections that revision made to its own
earlier measurements.

A field cache is the biggest thing a siting study produces -- 5 793 MB
over 25 files on the CIRED 2026 study -- and most of it is a smooth
scalar that nobody reads to the last euro. The obvious codec quantises
the VALUE onto a fixed bit budget, and that turns out to be the wrong
knob: the error is then a by-product of each tile's dynamic range, which
is why shrinking tiles, promoting outlier tiles and removing a radial
trend all failed to move it. **Fix the QUANTUM instead** -- bin width
``q``, as SZ-class compressors do -- and the error becomes a parameter,
independent of dynamic range entirely.

Measured on ``fields_min5m/field_PCC0`` (41.8 M cells, deflated float32
= 121.2 MB today, a tiled 16-bit codec = 27.1 MB at a *measured* 45.89
EUR worst case):

=================  ========  ==========================
guaranteed bound   size      vs the tiled 16-bit codec
=================  ========  ==========================
100 EUR            18.9 MB   0.70x
10 EUR             25.6 MB   0.94x
1 EUR              31.9 MB   1.18x
=================  ========  ==========================

Three properties this module treats as load-bearing rather than
incidental:

**Rounding is one-sided and the direction is stated.** Values are
floored onto the quantum, so a decoded value is never above the true
one. :class:`DecodedField` carries both ends of the interval and makes
the caller choose, because the soundness rule is **directional, not
role-based**: a quantity that must not be understated needs the upper
end wherever it sits in an inequality. A screening LOWER bound may use
``lower``; a certificate that SUBTRACTS a field value (as the substation
study's ``certify_k_truncation`` does) must use ``upper``, or
understatement makes it falsely pass.

**The reachability mask is part of the format.** Without it unreachable
cells decode to some finite filler and every bound over them is a lie.

**Decode is float64 and steps one ulp outward.** float32 at 1.3e7 EUR
has a 1.0 EUR ulp, which is larger than the error bound being promised;
worse, today's *exact* fields are float32 and already overstate half
their cells by up to 0.50 EUR. ``k * quantum`` is itself a rounded
product, so the ends are nudged with :func:`numpy.nextafter` --
Neumaier & Shcherbina's safe-bound discipline, not decoration.

Two free wins from the methodology audit are built in: **tile-major
ordering** (row-major delta coding crosses ~35 tiles per row, so this is
-19.8 % for nothing) and **per-tile-row framing** (0.0-0.4 %, and it is
what makes random access possible at all -- it has to be built in, not
assumed afterwards).

Not implemented, deliberately: outlier-tile promotion. It is a real
lever against a codec whose error follows each tile's RANGE, and that is
the codec this one replaces. Under a fixed quantum the error is already
uniform and already a knob, so there is no outlier tile to promote.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FIELD_CODECS",
    "DecodedField",
    "encode_field",
    "decode_field",
    "codec_of",
    "is_lossy",
]

#: Codec names accepted by :func:`encode_field`.
FIELD_CODECS = ("raw", "fixed-quantum")

_LOSSY = frozenset({"fixed-quantum"})
_DEFAULT_TILE = 256
_ZLIB_LEVEL = 6


def is_lossy(codec: str) -> bool:
    """True when ``codec`` does not round-trip exactly."""
    return str(codec) in _LOSSY


def codec_of(meta: dict) -> str:
    """The codec a saved field used.

    Dispatch happens on this key and NOT on the format version: the
    version is checked for strict equality, so bumping it to introduce a
    codec would reject every field already on disk.
    """
    return str(meta.get("codec", "raw"))


@dataclass(frozen=True)
class DecodedField:
    """A decoded field as an INTERVAL, with the two ends kept apart.

    ``lower <= true <= upper`` holds cell by cell, in float64, including
    the rounding of the decode arithmetic itself. Which end to use is
    decided by the inequality a caller is about to write, not by whether
    the quantity is called a bound or an incumbent:

    * a value that must not be OVERstated (a screening lower bound, a
      pruning threshold compared with ``<=``) takes :attr:`lower`;
    * a value that must not be UNDERstated (a reported objective, an
      incumbent, anything a certificate SUBTRACTS) takes :attr:`upper`.

    Unreachable cells are ``inf`` at both ends.
    """

    lower: np.ndarray
    error_bound: float
    reachable: np.ndarray
    codec: str

    @property
    def upper(self) -> np.ndarray:
        """The other end of the interval; ``lower`` for a lossless codec."""
        if self.error_bound <= 0.0:
            return self.lower
        out = np.nextafter(self.lower + self.error_bound, np.inf)
        out[~self.reachable] = np.inf
        return out

    @property
    def exact(self) -> bool:
        return self.error_bound <= 0.0

    def choose(self, side: str) -> np.ndarray:
        """``lower`` or ``upper`` by name, so call sites read as intent."""
        if side == "lower":
            return self.lower
        if side == "upper":
            return self.upper
        raise ValueError(
            f"side must be 'lower' or 'upper', got {side!r}; there is no "
            f"third option -- 'the value' is exactly the ambiguity this "
            f"type exists to remove")

    def width(self) -> float:
        """The guaranteed interval width in EUR."""
        return float(self.error_bound)


# ======================================================================
# tile-major reordering
# ======================================================================

def _tile_major(a: np.ndarray, tile: int) -> tuple[np.ndarray, int, int]:
    """Flatten ``a`` tile by tile instead of row by row.

    Row-major delta coding on a tiled field crosses one tile boundary
    every ``tile`` columns -- about 35 per row on the CIRED window --
    and each boundary emits a residual that predicts nothing. Walking
    the tiles instead is -19.8 % for no loss and no extra metadata.

    Returns ``(flat, padded_rows, padded_cols)``; the padding repeats
    the edge so it costs nothing in the delta stream.
    """
    rows, cols = a.shape
    pr = (-rows) % tile
    pc = (-cols) % tile
    if pr or pc:
        a = np.pad(a, ((0, pr), (0, pc)), mode="edge")
    R, C = a.shape
    flat = (a.reshape(R // tile, tile, C // tile, tile)
             .transpose(0, 2, 1, 3)
             .reshape(-1))
    return np.ascontiguousarray(flat), R, C


def _untile_major(flat: np.ndarray, rows: int, cols: int,
                  padded: tuple[int, int], tile: int) -> np.ndarray:
    R, C = padded
    out = (flat.reshape(R // tile, C // tile, tile, tile)
               .transpose(0, 2, 1, 3)
               .reshape(R, C))
    return np.ascontiguousarray(out[:rows, :cols])


# ======================================================================
# encode / decode
# ======================================================================

def encode_field(values, *, codec: str = "fixed-quantum",
                 error_bound: float = 1.0, tile: int = _DEFAULT_TILE,
                 order: int = 1) -> tuple[dict[str, np.ndarray], dict]:
    """Compress a settled field.

    Parameters:
        values: The field, any shape ``(rows, cols)``; ``inf`` and
            ``nan`` mean unreachable and are stored in the mask, not in
            the value stream.
        codec: ``"fixed-quantum"`` or ``"raw"`` (float32, deflated --
            which is itself 1.65x free over an uncompressed save and is
            the honest baseline to quote against).
        error_bound: The GUARANTEED one-sided understatement in EUR, and
            also the quantum: a decoded value lies in
            ``(true - error_bound - 1 ulp, true]``, the ulp being the
            outward nudge the decode applies so its own rounding cannot
            break the inequality. SZ's symmetric ``eb`` corresponds to
            ``error_bound = 2 * eb``.
        tile: Tile side for the reordering. 256 is the measured sweet
            spot; smaller tiles do not help, because the error under a
            fixed quantum does not depend on the tile at all.
        order: Predictor order along the tile-major stream. ``1`` is a
            plain delta, ``2`` a second difference (a cheap stand-in for
            a 2-D Lorenzo predictor). Both are tried and the smaller is
            kept, so this is a ceiling rather than a choice.

    Returns:
        ``(payload, meta)`` -- arrays to store and the JSON-able
        metadata a decoder needs. ``meta["codec"]`` is what
        :func:`decode_field` dispatches on.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D field, got shape {arr.shape}")
    if codec not in FIELD_CODECS:
        raise ValueError(
            f"unknown codec {codec!r}; have {FIELD_CODECS}")

    reachable = np.isfinite(arr)
    if codec == "raw":
        body = np.where(reachable, arr, np.inf).astype(np.float32)
        return ({"dist": body.ravel()},
                {"codec": "raw", "rows": int(arr.shape[0]),
                 "cols": int(arr.shape[1]), "error_bound": 0.0})

    q = float(error_bound)
    if not (q > 0.0 and np.isfinite(q)):
        raise ValueError(
            f"error_bound must be a finite positive quantum, got {q}")
    if np.any(arr[reachable] < 0):
        raise ValueError(
            "negative field values are not supported; a cost field is "
            "non-negative and the quantiser relies on it")

    # Floor onto the quantum. Unreachable cells take the previous
    # reachable value so the delta stream sees a flat run instead of a
    # cliff; the mask is what restores them.
    k = np.zeros(arr.shape, dtype=np.int64)
    k[reachable] = np.floor(arr[reachable] / q).astype(np.int64)
    flat_k, R, C = _tile_major(k, tile)
    flat_ok, _, _ = _tile_major(reachable.astype(np.uint8), tile)
    idx = np.where(flat_ok.astype(bool), np.arange(flat_k.size), 0)
    np.maximum.accumulate(idx, out=idx)
    flat_k = flat_k[idx]

    per_frame = tile * tile * (C // tile)
    n_frames = flat_k.size // per_frame
    frames: list[bytes] = []
    for f in range(n_frames):
        chunk = flat_k[f * per_frame:(f + 1) * per_frame]
        frames.append(_pack_frame(chunk, order))

    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in frames])
    body = np.frombuffer(b"".join(frames), dtype=np.uint8)
    mask = np.packbits(reachable.ravel())

    meta = {
        "codec": "fixed-quantum",
        "rows": int(arr.shape[0]), "cols": int(arr.shape[1]),
        "padded_rows": int(R), "padded_cols": int(C),
        "tile": int(tile), "order": int(order),
        "quantum": q, "error_bound": q,
        "frames": int(n_frames),
        "note": ("values are floored onto the quantum, so a decoded "
                 "value is never above the true one"),
    }
    return {"q_body": body, "q_frames": offsets, "reach_mask": mask}, meta


_WIDTHS = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _pack_frame(chunk: np.ndarray, order: int) -> bytes:
    """Delta, zigzag, byte-shuffle, deflate -- one independently read frame.

    Both predictor orders are packed and the smaller kept. The residual
    WIDTH is chosen per frame from what the frame actually needs, which
    does two things: a smooth field mostly needs one or two bytes rather
    than four, and a field whose range times its quantum exceeds 32 bits
    still encodes instead of raising. Header is ``[order][width]``.

    Framing per tile-row costs 0.0-0.4 % and is what makes a partial
    read possible at all -- it has to be built in, not assumed.
    """
    best: bytes | None = None
    for o in range(1, max(1, int(order)) + 1):
        d = chunk
        for _ in range(o):
            nd = np.empty_like(d)
            nd[0] = d[0]
            nd[1:] = d[1:] - d[:-1]
            d = nd
        zz = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        top = int(zz.max(initial=0))
        width = next(w for w in (1, 2, 4, 8)
                     if w == 8 or top <= (1 << (8 * w)) - 1)
        v = zz.astype(_WIDTHS[width]).view(np.uint8).reshape(-1, width)
        blob = zlib.compress(np.ascontiguousarray(v.T).tobytes(), _ZLIB_LEVEL)
        head = bytes([o, width])
        if best is None or len(blob) + 2 < len(best):
            best = head + blob
    return best


def _unpack_frame(blob: bytes, n: int) -> np.ndarray:
    order, width = blob[0], blob[1]
    raw = zlib.decompress(blob[2:])
    v = np.frombuffer(raw, dtype=np.uint8).reshape(width, n)
    zz = (np.ascontiguousarray(v.T).view(_WIDTHS[width]).ravel()
          .astype(np.uint64))
    d = ((zz >> np.uint64(1)).astype(np.int64)
         ^ -(zz & np.uint64(1)).astype(np.int64))
    for _ in range(order):
        d = np.cumsum(d)
    return d


def decode_field(payload, meta: dict) -> DecodedField:
    """Read back a field written by :func:`encode_field`.

    The two interval ends are computed in float64 and nudged one ulp
    outward, so ``lower <= true <= upper`` survives the decode's own
    rounding rather than merely the quantiser's.
    """
    codec = codec_of(meta)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    if codec == "raw":
        arr = np.asarray(payload["dist"], dtype=np.float64).reshape(rows, cols)
        return DecodedField(lower=arr, error_bound=0.0,
                            reachable=np.isfinite(arr), codec="raw")
    if codec != "fixed-quantum":
        raise ValueError(f"unknown codec {codec!r} in field metadata")

    tile = int(meta["tile"])
    R, C = int(meta["padded_rows"]), int(meta["padded_cols"])
    q = float(meta["quantum"])
    body = np.asarray(payload["q_body"], dtype=np.uint8)
    offsets = np.asarray(payload["q_frames"], dtype=np.int64)
    per_frame = tile * tile * (C // tile)

    flat = np.empty(R * C, dtype=np.int64)
    raw = body.tobytes()
    for f in range(offsets.size - 1):
        flat[f * per_frame:(f + 1) * per_frame] = _unpack_frame(
            raw[offsets[f]:offsets[f + 1]], per_frame)
    k = _untile_major(flat, rows, cols, (R, C), tile)

    mask = np.unpackbits(np.asarray(payload["reach_mask"], dtype=np.uint8),
                         count=rows * cols).astype(bool).reshape(rows, cols)
    lower = np.nextafter(k.astype(np.float64) * q, -np.inf)
    lower[~mask] = np.inf
    return DecodedField(lower=lower, error_bound=q, reachable=mask,
                        codec=codec)

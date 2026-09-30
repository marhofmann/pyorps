"""Exact site-cost field over every anchor cell and rotation (plan rev. 5, D3).

The substation pad costs the sum of the site prices of the pixels its
rotated footprint covers -- the ``"pixels"`` rule of
:func:`pyorps.siting.footprint.verify_exact`: a pixel belongs to the pad
when its CENTRE lies in the rectangle (``rotated_kernel(..., supersample=1)
> 0``), and a placement is infeasible when the footprint leaves the grid or
touches a forbidden pixel. This module computes that quantity EXACTLY for
every anchor cell and every rotation at once, by FFT convolution.

Exactness (plan section 3.7)
----------------------------
Prices are integer cents per pixel. A float64 FFT convolution is exact
after rounding only when the convolved layers are small integers: its
round-off grows with the layer's total magnitude, so convolving a raster of
cents directly can drift by whole cents. The plan therefore convolves
integer INDICATOR layers only. Two ways of doing that are offered, both
asserting ``|x - round(x)| <= 0.05`` on every convolved value:

``method="limbs"`` (default)
    The price raster is split into base-256 digit layers,
    ``cents = sum_i d_i 256^i`` with ``0 <= d_i <= 255``; each digit layer
    is convolved and the rounded digit sums are recombined in int64. By
    linearity this is exactly ``sum over pixels of cents``, at
    ``ceil(log_256(max cents))`` FFT layers (4 for any price below
    42.9 MEUR per pixel) instead of one per price class.
``method="classes"``
    One indicator layer per price class, as plan D3 is written; the value
    is ``sum_c price_c * count_c`` in int64. Needed when the class counts
    themselves are wanted everywhere, and a cross-check of the default.

Either way the forbidden layer is convolved as its own indicator.

Anchors and rotations
---------------------
The anchor ``(r, c)`` is the kernel centre, exactly as ``verify_exact``
places it, and so is the border rule: a placement is infeasible when the
kernel's bounding BOX leaves the grid, which excludes a margin of the
kernel radius (the pad's half-diagonal plus one pixel) along every edge.
Rotations default to 36 steps of 5 degrees over 180 (plan section 2.2;
``build_site_raster.py:887-888``). Ties between rotations go
to the LOWEST rotation index, so the argmin is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from pyorps.siting.footprint import rotated_kernel

__all__ = [
    "SiteField",
    "footprint_mask",
    "site_field",
    "site_values",
]

#: Largest acceptable distance of a convolved count from an integer.
_ROUND_TOL = 0.05
#: int64 sentinel for "infeasible" in a cents field.
INFEASIBLE = np.iinfo(np.int64).max


def footprint_mask(theta_deg: float, length_m: float, width_m: float,
                   cell_m: float) -> np.ndarray:
    """The pad's pixel set at one rotation, as a boolean kernel.

    ``rotated_kernel(..., supersample=1) > 0``: the pixels whose centre the
    rectangle covers. The kernel is centred on its middle element.
    """
    return rotated_kernel(theta_deg, length_m, width_m, cell_m,
                          supersample=1) > 0


@dataclass
class SiteField:
    """Cheapest rotation per anchor cell, in integer cents.

    Attributes:
        cents: ``(rows, cols)`` int64 pad cost at the best rotation;
            :data:`INFEASIBLE` where no rotation fits.
        rotation: ``(rows, cols)`` int16 index into ``thetas`` of the best
            rotation; ``-1`` where infeasible.
        thetas: The rotations evaluated, degrees.
        per_rotation: ``(n_rot, rows, cols)`` int64, only with
            ``keep_rotations=True``.
    """
    cents: np.ndarray
    rotation: np.ndarray
    thetas: np.ndarray
    length_m: float
    width_m: float
    cell_m: float
    per_rotation: np.ndarray | None = None

    @property
    def feasible(self) -> np.ndarray:
        return self.cents != INFEASIBLE

    def eur(self) -> np.ndarray:
        """The field in EUR, float64, ``inf`` where infeasible."""
        out = self.cents.astype(np.float64) / 100.0
        out[~self.feasible] = np.inf
        return out


def _limbs(values: np.ndarray) -> list[np.ndarray]:
    """Base-256 digit layers of a non-negative int64 array."""
    v = np.asarray(values, dtype=np.int64)
    if v.size and v.min() < 0:
        raise ValueError("prices must be non-negative integer cents")
    top = int(v.max()) if v.size else 0
    n = max(1, math.ceil(math.log(top + 1, 256)) if top > 0 else 1)
    while 256 ** n <= top:
        n += 1
    return [((v >> (8 * i)) & 0xFF).astype(np.float64) for i in range(n)]


def _fft_shape(shape, kshape):
    from scipy import fft as sfft
    return tuple(sfft.next_fast_len(int(a + b - 1), real=True)
                 for a, b in zip(shape, kshape))


def _rounded(x: np.ndarray, what: str) -> np.ndarray:
    r = np.rint(x)
    err = float(np.max(np.abs(x - r))) if x.size else 0.0
    if err > _ROUND_TOL:
        raise ArithmeticError(
            f"{what}: FFT sum {err:.3f} away from an integer; the layer is "
            f"too large for exact float64 convolution")
    return r.astype(np.int64)


def site_field(price_cents: np.ndarray, forbidden: np.ndarray, *,
               length_m: float, width_m: float, cell_m: float,
               thetas: Sequence[float] | None = None, method: str = "limbs",
               classes: np.ndarray | None = None,
               class_cents: Sequence[int] | None = None,
               block: int = 2048, keep_rotations: bool = False,
               workers: int = 2) -> SiteField:
    """Exact pad cost for every anchor cell, cheapest over ``thetas``.

    Parameters:
        price_cents: ``(rows, cols)`` integer cents per pixel (>= 0). With
            ``method="classes"`` it may be ``None`` and is then built from
            ``classes`` and ``class_cents``.
        forbidden: ``(rows, cols)`` bool, pixels no pad may touch.
        length_m, width_m: Pad sides; ``length_m`` runs along the rows at
            rotation 0, as in :func:`rotated_kernel`.
        cell_m: Pixel size.
        thetas: Rotations in degrees; default 0, 5, ..., 175.
        method: ``"limbs"`` or ``"classes"`` (see the module docstring).
        classes, class_cents: For ``method="classes"``: the class id per
            pixel (0 .. C-1) and the price of each class in cents.
        block: Anchor rows and columns per FFT block.
        keep_rotations: Also return the value of every rotation.
        workers: Threads for scipy.fft.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from scipy import fft as sfft

    forb = np.asarray(forbidden, dtype=bool)
    rows, cols = forb.shape
    if method == "classes":
        if classes is None or class_cents is None:
            raise ValueError("method='classes' needs classes and class_cents")
        cls = np.asarray(classes, dtype=np.int64)
        cc = np.asarray(class_cents, dtype=np.int64)
        if cls.shape != forb.shape:
            raise ValueError("classes and forbidden differ in shape")
        if cls.min() < 0 or cls.max() >= cc.size:
            raise ValueError("a class id has no price")
        if np.any(cc < 0):
            raise ValueError("class prices must be >= 0 cents")
        price = cc[cls]
    elif method == "limbs":
        price = np.asarray(price_cents, dtype=np.int64)
        if price.shape != forb.shape:
            raise ValueError("price_cents and forbidden differ in shape")
    else:
        raise ValueError("method must be 'limbs' or 'classes'")
    thetas = (np.arange(0.0, 180.0, 5.0) if thetas is None
              else np.asarray(thetas, dtype=np.float64))
    masks = [footprint_mask(t, length_m, width_m, cell_m) for t in thetas]
    ksz = max(m.shape[0] for m in masks)
    # pad every kernel to the same odd size, centred
    kern = []
    for m in masks:
        pad = (ksz - m.shape[0]) // 2
        k = np.zeros((ksz, ksz), dtype=np.float64)
        k[pad:pad + m.shape[0], pad:pad + m.shape[1]] = m
        kern.append(k[::-1, ::-1])       # correlation = convolution by the flip
    rad = ksz // 2
    n_rot = len(thetas)

    best = np.full((rows, cols), INFEASIBLE, dtype=np.int64)
    arg = np.full((rows, cols), -1, dtype=np.int16)
    per = (np.full((n_rot, rows, cols), INFEASIBLE, dtype=np.int64)
           if keep_rotations else None)

    for r0, c0, r1, c1 in _blocks(rows, cols, block):
        # input window: anchors +- rad; anchors whose pad leaves the grid
        # stay infeasible
        wr0, wc0 = r0 - rad, c0 - rad
        wr1, wc1 = r1 + rad, c1 + rad
        ir0, ic0 = max(wr0, 0), max(wc0, 0)
        ir1, ic1 = min(wr1, rows), min(wc1, cols)
        shape = (wr1 - wr0, wc1 - wc0)
        fshape = _fft_shape(shape, (ksz, ksz))

        def window(a, fill):
            w = np.full(shape, fill, dtype=np.float64)
            w[ir0 - wr0:ir1 - wr0, ic0 - wc0:ic1 - wc0] = a[ir0:ir1, ic0:ic1]
            return w

        # outside the grid counts as forbidden, so a pad that leaves the
        # grid is infeasible exactly as in verify_exact
        layers = [window(forb.astype(np.float64), 1.0)]
        if method == "limbs":
            layers += [window(d, 0.0) for d in _limbs(price)]
            weights = [256 ** i for i in range(len(layers) - 1)]
        else:
            ids = np.unique(cls)
            layers += [window((cls == c).astype(np.float64), 0.0) for c in ids]
            weights = [int(cc[c]) for c in ids]
        spectra = [sfft.rfft2(a, s=fshape, workers=workers) for a in layers]
        orow = slice(ksz - 1, ksz - 1 + (r1 - r0))
        ocol = slice(ksz - 1, ksz - 1 + (c1 - c0))
        for ri, k in enumerate(kern):
            kf = sfft.rfft2(k, s=fshape, workers=workers)
            vals = []
            for li, sp in enumerate(spectra):
                conv = sfft.irfft2(sp * kf, s=fshape, workers=workers)
                vals.append(_rounded(conv[orow, ocol],
                                     "forbidden count" if li == 0
                                     else f"layer {li}"))
            ok = vals[0] == 0
            total = np.zeros_like(vals[0])
            for w, v in zip(weights, vals[1:]):
                total += np.int64(w) * v
            total[~ok] = INFEASIBLE
            if per is not None:
                per[ri, r0:r1, c0:c1] = total
            cur = best[r0:r1, c0:c1]
            better = total < cur
            cur[better] = total[better]
            arg[r0:r1, c0:c1][better] = ri
    # verify_exact's border rule: a placement whose kernel BOX (the pad's
    # bounding square, radius rad) leaves the grid is infeasible even when
    # the pad's own pixels would fit. Only anchors within rad of the edge.
    edge = np.zeros((rows, cols), dtype=bool)
    edge[:rad, :] = edge[rows - rad:, :] = True
    edge[:, :rad] = edge[:, cols - rad:] = True
    best[edge] = INFEASIBLE
    arg[edge] = -1
    if per is not None:
        per[:, edge] = INFEASIBLE
    return SiteField(cents=best, rotation=arg, thetas=thetas,
                     length_m=float(length_m), width_m=float(width_m),
                     cell_m=float(cell_m), per_rotation=per)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite


def _blocks(rows: int, cols: int, block: int
            ) -> Iterator[tuple[int, int, int, int]]:
    for r0 in range(0, rows, block):
        for c0 in range(0, cols, block):
            yield r0, c0, min(r0 + block, rows), min(c0 + block, cols)


def site_values(price_cents: np.ndarray, forbidden: np.ndarray,
                rows_cols, theta_deg: float, *, length_m: float,
                width_m: float, cell_m: float,
                classes: np.ndarray | None = None,
                n_classes: int | None = None):
    """Direct pixel sums at a few anchors: cents, and class counts.

    The independent check of :func:`site_field` and the source of the
    per-class counts at an argmin (plan D3, Stage-2 output). Returns
    ``(cents, counts)``: ``cents`` int64 with :data:`INFEASIBLE` where the
    pad leaves the grid or touches a forbidden pixel; ``counts`` an
    ``(n, n_classes)`` int64 array when ``classes`` is given, else ``None``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    price = np.asarray(price_cents, dtype=np.int64)
    forb = np.asarray(forbidden, dtype=bool)
    mask = footprint_mask(theta_deg, length_m, width_m, cell_m)
    kh, kw = mask.shape
    pairs = np.asarray(rows_cols, dtype=np.int64).reshape(-1, 2)
    rows, cols = price.shape
    cents = np.empty(len(pairs), dtype=np.int64)
    counts = None
    if classes is not None:
        cls = np.asarray(classes, dtype=np.int64)
        k = int(n_classes if n_classes is not None else cls.max() + 1)
        counts = np.zeros((len(pairs), k), dtype=np.int64)
    for i, (r, c) in enumerate(pairs):
        r0, c0 = int(r) - kh // 2, int(c) - kw // 2
        r1, c1 = r0 + kh, c0 + kw
        if r0 < 0 or c0 < 0 or r1 > rows or c1 > cols:
            cents[i] = INFEASIBLE
            continue
        if forb[r0:r1, c0:c1][mask].any():
            cents[i] = INFEASIBLE
            continue
        cents[i] = int(price[r0:r1, c0:c1][mask].sum())
        if counts is not None:
            counts[i] = np.bincount(cls[r0:r1, c0:c1][mask],
                                    minlength=counts.shape[1])
    return cents, counts

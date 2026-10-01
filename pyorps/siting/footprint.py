"""Footprint screening: what a facility costs to build at every cell.

Lifted out of ``case_studies/substation_planning2`` (plan A Phase 7,
plan B section 4a). The study had a correct implementation of a general
idea locked inside a driver script, and a second copy of it in TopoMILP;
this is the library version, with the two defects that review found
pinned by name rather than inherited.

The screen answers, for every cell and every rotation, "what would a
``w x h`` rectangle centred here cost to build, and is it allowed?".
Convolution does it for all cells at once: one FFT of the cost layer and
one of the blocked layer, times one FFT per rotation. The running
minimum over rotations keeps the cheapest FEASIBLE orientation per cell.

Two things the driver got right and are kept:

* the kernel is **supersampled and area-renormalised**, because the
  integer rasterisation of a rotated rectangle swings ~4 % in area
  across rotations (827-861 px for a 20x40 m rectangle at 1 m) and that
  swing would otherwise read as a cost difference between orientations;
* the screen never decides anything on its own. :func:`verify_exact`
  re-scores the shortlist with the TRUE integer pixel set, and that is
  what picks a winner.

One thing it got wrong and is fixed here: the study's own field sampler
used ``np.round`` where every library path uses rasterio's ``rowcol``
(floor). With candidates at pixel centres that shifts half of them by
one cell. :func:`~pyorps.siting.candidates.sample_field` pins the floor
rule and ``tests/test_siting/test_candidates.py`` holds it there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Footprint",
    "ScreenResult",
    "rotated_kernel",
    "screen_footprints",
    "verify_exact",
]


@dataclass(frozen=True)
class Footprint:
    """A rectangular facility footprint and the rotations to try.

    Parameters:
        length_m, width_m: Rectangle sides in metres.
        rotation_step_deg: Angular resolution of the sweep.
        rotation_max_deg: Exclusive upper end. A rectangle repeats every
            180 deg, so 180 is the full sweep and 90 is enough only for
            a square.
    """

    length_m: float
    width_m: float
    rotation_step_deg: float = 5.0
    rotation_max_deg: float = 180.0

    def __post_init__(self):
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("footprint sides must be positive")
        if self.rotation_step_deg <= 0:
            raise ValueError("rotation_step_deg must be positive")

    @property
    def thetas(self) -> np.ndarray:
        """Rotations to evaluate, in degrees."""
        return np.arange(0.0, self.rotation_max_deg, self.rotation_step_deg)

    @property
    def half_diagonal_m(self) -> float:
        """Furthest a footprint corner can be from its centre."""
        return math.hypot(self.length_m, self.width_m) / 2.0

    @property
    def area_m2(self) -> float:
        return float(self.length_m * self.width_m)


def rotated_kernel(theta_deg: float, h_m: float, w_m: float, cell_m: float,
                   supersample: int = 4) -> np.ndarray:
    """Fractional-coverage rotated rectangle kernel, area-renormalised.

    Mirrors ``ConstrainedPathFinder._precompute_area_offsets`` (rotate
    the offset grid, keep ``|local|`` within the half-sides) but
    supersampled and renormalised to ``h * w / cell^2``. The integer
    version's rasterised area swings about 4 % across rotations -- 827
    to 861 px for a 20 x 40 m rectangle at 1 m -- and without the
    renormalisation that swing is indistinguishable from the terrain
    preferring one orientation over another.

    Parameters:
        theta_deg: Rotation, degrees.
        h_m, w_m: Rectangle sides in metres.
        cell_m: Raster resolution.
        supersample: Sub-pixel samples per axis. ``1`` gives the exact
            integer pixel set, which is what :func:`verify_exact` wants.
    """
    half_h_px = (h_m / 2.0) / cell_m
    half_w_px = (w_m / 2.0) / cell_m
    r_max = int(math.ceil(math.hypot(half_h_px, half_w_px))) + 1
    size = 2 * r_max + 1

    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)

    ss = int(supersample)
    offsets = (np.arange(ss) + 0.5) / ss - 0.5
    centre = r_max
    rr, cc = np.meshgrid(np.arange(size) - centre, np.arange(size) - centre,
                         indexing="ij")
    hits = np.zeros((size, size), dtype=np.float64)
    for dr in offsets:
        for dc in offsets:
            local_r = (rr + dr) * cos_t - (cc + dc) * sin_t
            local_c = (rr + dr) * sin_t + (cc + dc) * cos_t
            hits += ((np.abs(local_r) <= half_h_px)
                     & (np.abs(local_c) <= half_w_px))
    kernel = hits / (ss * ss)

    true_area_px = (h_m * w_m) / (cell_m * cell_m)
    measured = kernel.sum()
    if measured > 0:
        kernel *= true_area_px / measured
    return kernel.astype(np.float32)


@dataclass
class ScreenResult:
    """Per-cell build cost and the orientation that achieved it."""

    build_cost_eur: np.ndarray    #: float64, ``nan`` where infeasible
    best_theta_deg: np.ndarray    #: float32
    feasible: np.ndarray          #: bool
    transform: Any
    resolution_m: float
    footprint: Footprint

    @property
    def shape(self) -> tuple[int, int]:
        return self.build_cost_eur.shape

    def to_geotiff(self, path, crs=None, nodata: float = -1.0):
        """Write :attr:`build_cost_eur` as a single-band GeoTIFF."""
        import rasterio
        out = np.where(self.feasible, self.build_cost_eur,
                       nodata).astype(np.float32)
        with rasterio.open(
                path, "w", driver="GTiff", height=out.shape[0],
                width=out.shape[1], count=1, dtype="float32", crs=crs,
                transform=self.transform, nodata=nodata) as dst:
            dst.write(out, 1)
            dst.set_band_description(1, "footprint build cost, EUR")
        return path


def screen_footprints(cost, blocked, *, footprint: Footprint,
                      transform=None, resolution_m: float = 1.0,
                      workers: int | None = None) -> ScreenResult:
    """Cheapest feasible orientation of ``footprint`` at every cell.

    One rFFT of each layer, one per rotation, and a running minimum. The
    cost is ``O(rotations * N log N)`` regardless of how big the
    footprint is, which is the whole reason to do it this way rather
    than sliding a window.

    Parameters:
        cost: Per-cell build cost, EUR per cell (already multiplied by
            cell area if that is how the cost model is expressed).
        blocked: Per-cell 0/1 -- anything non-zero under the footprint
            makes that placement infeasible.
        footprint: Rectangle and rotation sweep.
        transform: Affine transform, carried into the result.
        resolution_m: Cell size in metres.
        workers: Threads for the FFTs; ``None`` lets scipy decide.

    Returns:
        A :class:`ScreenResult`. The feasibility test is
        ``convolved_blocked < 0.5``: the true value is 0 for a feasible
        placement, so anything below a half is FFT round-off, and
        anything at or above it is a real overlap.
    """
    from scipy.fft import irfft2, rfft2

    cost = np.asarray(cost, dtype=np.float32)
    blocked = np.asarray(blocked, dtype=np.float32)
    if cost.shape != blocked.shape:
        raise ValueError(
            f"cost {cost.shape} and blocked {blocked.shape} must match")
    rows, cols = cost.shape
    pad = int(2 * math.ceil(math.hypot(
        footprint.length_m / 2 / resolution_m,
        footprint.width_m / 2 / resolution_m)) + 1) + 2
    shape = (rows + pad, cols + pad)

    fc = rfft2(cost, s=shape, workers=workers)
    fb = rfft2(blocked, s=shape, workers=workers)

    best = np.full((rows, cols), np.inf, dtype=np.float64)
    best_theta = np.zeros((rows, cols), dtype=np.float32)
    for theta in footprint.thetas:
        kernel = rotated_kernel(theta, footprint.length_m,
                                footprint.width_m, resolution_m)
        kr, kc = kernel.shape
        fk = rfft2(kernel, s=shape, workers=workers)
        c = irfft2(fc * fk, s=shape, workers=workers)
        b = irfft2(fb * fk, s=shape, workers=workers)
        off_r, off_c = kr // 2, kc // 2
        c_valid = c[off_r:off_r + rows, off_c:off_c + cols]
        b_valid = b[off_r:off_r + rows, off_c:off_c + cols]
        improves = (b_valid < 0.5) & (c_valid < best)
        best = np.where(improves, c_valid, best)
        best_theta = np.where(improves, np.float32(theta), best_theta)

    # The FFT pads with zeros, so a footprint hanging off the edge of
    # the grid reads as both cheap and unblocked -- the two most
    # attractive properties a placement can have. Those cells are not
    # screened, they are unknown, so they are infeasible here rather
    # than quietly competing for the minimum. verify_exact independently
    # rejects them, and this keeps the two consistent.
    margin = int(math.ceil(math.hypot(
        footprint.length_m / 2 / resolution_m,
        footprint.width_m / 2 / resolution_m))) + 1
    if margin:
        best[:margin, :] = np.inf
        best[-margin:, :] = np.inf
        best[:, :margin] = np.inf
        best[:, -margin:] = np.inf

    feasible = np.isfinite(best)
    return ScreenResult(
        build_cost_eur=np.where(feasible, best, np.nan),
        best_theta_deg=best_theta, feasible=feasible, transform=transform,
        resolution_m=float(resolution_m), footprint=footprint)


def verify_exact(cost, blocked, screen: ScreenResult, rows_cols, *,
                 mode: str = "pixels", supersample: int = 8) -> np.ndarray:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Re-score placements without the FFT, at each cell's winning rotation.

    The screen is a SCREEN: its kernel is supersampled but still
    approximate and the FFT adds round-off on top. This decides, for the
    shortlist.

    The two modes are two different quantities, and they do not agree --
    on a uniform layer a 20 x 20 m footprint at 5 m prices at 75 in
    ``"pixels"`` and 48 in ``"coverage"``, because the pixel set has 25
    cells where the rectangle covers 16. Neither is a rounding error of
    the other:

    ``"pixels"`` (the default, and what the substation study did)
        Sum the cells whose CENTRE the rectangle covers. This is what
        you actually have to buy, so it is the right basis for a price,
        and it is the arbiter the study intended. Its area swings with
        rotation, which is precisely why it is not what the screen
        ranks on.
    ``"coverage"``
        Area-weighted by the fractional kernel -- the same quantity the
        screen ranks on, computed exactly instead of through an FFT.
        Use this to check the screen, not to price a purchase.

    Returns:
        Cost per ``(row, col)``; ``inf`` where the footprint overlaps a
        blocked cell or leaves the grid.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if mode not in ("pixels", "coverage"):
        raise ValueError(
            f"mode must be 'pixels' or 'coverage', got {mode!r}")
    cost = np.asarray(cost, dtype=np.float64)
    blocked = np.asarray(blocked, dtype=np.float64)
    fp = screen.footprint
    pairs = np.asarray(rows_cols, dtype=np.int64).reshape(-1, 2)
    out = np.empty(len(pairs), dtype=np.float64)
    rows_max, cols_max = cost.shape
    ss = 1 if mode == "pixels" else int(supersample)
    for i, (r, c) in enumerate(pairs):
        theta = float(screen.best_theta_deg[r, c])
        kernel = rotated_kernel(theta, fp.length_m, fp.width_m,
                                screen.resolution_m, supersample=ss)
        kh, kw = kernel.shape
        r0, c0 = int(r) - kh // 2, int(c) - kw // 2
        r1, c1 = r0 + kh, c0 + kw
        if r0 < 0 or c0 < 0 or r1 > rows_max or c1 > cols_max:
            out[i] = np.inf
            continue
        touched = kernel > 0
        if (blocked[r0:r1, c0:c1][touched] > 0.5).any():
            out[i] = np.inf
        elif mode == "pixels":
            out[i] = float(cost[r0:r1, c0:c1][touched].sum())
        else:
            out[i] = float((cost[r0:r1, c0:c1] * kernel).sum())
    return out

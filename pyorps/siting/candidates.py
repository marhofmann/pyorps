"""Candidate positions on a lattice, and how to price them from a field.

Three things live here, all of them lifted out of
``case_studies/substation_planning2`` with their latent defects named:

:func:`candidate_lattice`
    Feasible centres on a stride grid, eroded by half the footprint
    diagonal so a candidate centre can never place the rotated rectangle
    partly outside the region the screen verified.
:func:`sample_field`
    Field lookup at map coordinates. **The cell-selection rule is
    pinned to FLOOR here**, which is what rasterio's ``rowcol`` and
    every library reader use. The study's own ``_sample_field`` used
    ``np.round``; with candidates sitting at pixel centres that moves
    half of them one cell, and the study's winning margin was 28.14 EUR.
    That is why a migration to this code cannot be bit-for-bit, and why
    ``tests/test_siting/test_candidates.py`` pins the rule rather than
    the outcome.
:class:`CandidateSet`
    The table the MILP eventually reads, with its own ids.

Hakimi's vertex optimality (Oper. Res. 12(3), 1964) is why a raster
lattice is a legitimate candidate set at all, and BSSS/BTST geometric
branch and bound (Drezner & Suzuki, Oper. Res. 52(1), 2004) is what
closes the gap the STRIDE opens -- see
:func:`~pyorps.siting.bounds.lipschitz_stride_gap`, because a stride
without that bound certifies the lattice and not the space.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any

import numpy as np

__all__ = ["CandidateSet", "candidate_lattice", "sample_field", "cell_xy"]


def cell_xy(transform, rows, cols):
    """Cell CENTRES as ``(x, y)`` for a rasterio-style affine transform."""
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    x = transform.c + (c + 0.5) * transform.a + (r + 0.5) * transform.b
    y = transform.f + (c + 0.5) * transform.d + (r + 0.5) * transform.e
    return x, y


def sample_field(field: np.ndarray, transform, xs, ys, *,
                 rule: str = "floor", clip: bool = False) -> np.ndarray:
    """Field value at map coordinates, by an EXPLICIT cell-selection rule.

    Parameters:
        field: 2-D array in the CRS of ``transform``.
        transform: Affine transform of ``field``.
        xs, ys: Coordinates.
        rule: ``"floor"`` -- what ``rasterio.transform.rowcol`` does and
            what every reader in PYORPS uses -- or ``"round"``, which is
            what the substation study's driver did. They differ by one
            cell for exactly the points that sit on a pixel centre,
            which is where a lattice of candidates puts all of them, so
            the two rules are NOT interchangeable and neither is a
            default worth leaving implicit.
        clip: Clamp out-of-range coordinates to the edge instead of
            returning ``inf``. The study clipped; a silent clamp turns
            "outside the window" into "expensive but reachable", so it
            is off here.

    Returns:
        float64 values, ``inf`` outside the field unless ``clip``.
    """
    if rule not in ("floor", "round"):
        raise ValueError(f"rule must be 'floor' or 'round', got {rule!r}")
    inv = ~transform
    cols, rows = inv @ (np.asarray(xs, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64))
    fn = np.floor if rule == "floor" else np.round
    rows = fn(rows).astype(np.int64)
    cols = fn(cols).astype(np.int64)
    if clip:
        rows = np.clip(rows, 0, field.shape[0] - 1)
        cols = np.clip(cols, 0, field.shape[1] - 1)
        return field[rows, cols].astype(np.float64)
    out = np.full(np.shape(rows), np.inf, dtype=np.float64)
    ok = ((rows >= 0) & (rows < field.shape[0])
          & (cols >= 0) & (cols < field.shape[1]))
    if np.any(ok):
        out[ok] = field[rows[ok], cols[ok]]
    return out


@dataclass
class CandidateSet:
    """Feasible facility positions, with ids the export keeps stable."""

    rows: np.ndarray
    cols: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    transform: Any = None
    crs: Any = None
    stride_m: float = 0.0
    build_cost_eur: np.ndarray | None = None
    best_theta_deg: np.ndarray | None = None
    area_id: np.ndarray | None = None
    meta: dict = _field(default_factory=dict)

    def __len__(self) -> int:
        return int(np.size(self.rows))

    @property
    def ids(self) -> np.ndarray:
        """Stable integer ids, ``0 .. n-1`` in construction order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def points(self) -> np.ndarray:
        """``(n, 2)`` array of ``(x, y)``."""
        return np.column_stack([self.xs, self.ys])

    def subset(self, keep) -> CandidateSet:
        """A new set holding only ``keep`` (a mask or an index array)."""
        k = np.asarray(keep)
        def take(a):
            return None if a is None else np.asarray(a)[k]
        return CandidateSet(
            rows=self.rows[k], cols=self.cols[k], xs=self.xs[k],
            ys=self.ys[k], transform=self.transform, crs=self.crs,
            stride_m=self.stride_m, build_cost_eur=take(self.build_cost_eur),
            best_theta_deg=take(self.best_theta_deg),
            area_id=take(self.area_id), meta=dict(self.meta))

    def __repr__(self) -> str:
        return (f"CandidateSet({len(self)} positions, "
                f"stride {self.stride_m:g} m)")


def candidate_lattice(screen, *, stride_m: float, areas=None,
                      erode: bool = True, crs=None) -> CandidateSet:
    """Feasible centres on a stride grid, eroded for the footprint.

    Parameters:
        screen: A :class:`~pyorps.siting.footprint.ScreenResult`.
        stride_m: Lattice spacing. Anything above one cell leaves the
            skipped positions UNCERTIFIED unless the caller also carries
            :func:`~pyorps.siting.bounds.lipschitz_stride_gap`.
        areas: Optional GeoDataFrame with ``id`` and ``geometry``; each
            candidate gets the id of the polygon containing it, or -1.
        erode: Shrink the feasible region by half the footprint diagonal
            first, so a candidate centre can never place the rotated
            rectangle partly outside the verified region.
        crs: CRS to record on the result.

    Returns:
        A :class:`CandidateSet` carrying the screen's build cost and
        winning rotation at each kept position.
    """
    from scipy.ndimage import binary_erosion

    feasible = screen.feasible & np.isfinite(screen.build_cost_eur)
    if erode:
        px = max(1, round(screen.footprint.half_diagonal_m
                          / screen.resolution_m))
        feasible = binary_erosion(
            feasible, structure=np.ones((2 * px + 1, 2 * px + 1), bool))

    stride_px = max(1, round(stride_m / screen.resolution_m))
    rr, cc = np.meshgrid(np.arange(0, feasible.shape[0], stride_px),
                         np.arange(0, feasible.shape[1], stride_px),
                         indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    keep = feasible[rr, cc]
    rr, cc = rr[keep], cc[keep]

    if screen.transform is not None:
        xs, ys = cell_xy(screen.transform, rr, cc)
    else:
        xs, ys = cc.astype(np.float64), rr.astype(np.float64)

    area_id = None
    if areas is not None and len(rr):
        import geopandas as gpd
        area_id = np.full(rr.shape, -1, dtype=np.int64)
        pts = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=areas.crs)
        for aid, geom in zip(areas["id"], areas.geometry):
            area_id[pts.within(geom).to_numpy()] = int(aid)

    return CandidateSet(
        rows=rr, cols=cc, xs=xs, ys=ys, transform=screen.transform,
        crs=crs, stride_m=float(stride_m),
        build_cost_eur=screen.build_cost_eur[rr, cc],
        best_theta_deg=screen.best_theta_deg[rr, cc], area_id=area_id,
        meta={"eroded": bool(erode),
              "erosion_m": (screen.footprint.half_diagonal_m if erode
                            else 0.0),
              "resolution_m": screen.resolution_m,
              "cell_rule": "floor"})

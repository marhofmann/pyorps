"""Reproduce the seal trade-off table quoted in ``pyorps.raster.thinness``.

    ./.venv/Scripts/python.exe tests/test_raster/sweep_forbidden_repair_tradeoff.py

NOT a test module (the leading ``sweep_`` keeps pytest from collecting it): it
takes about a minute and it measures a trade-off, it does not pin a contract.
It exists because the figures it produces were once quoted as bare percentages
whose parameterization had not been recorded, and a number nobody can reproduce
is not a measurement. Every quoted figure in the package comes from here.

THE PARAMETERIZATION, in full:

* extent 64 x 64 m at 1 m cells, bounding box anchored at EPSG:25832
  magnitudes (350000, 5600000) so the transform coefficients are ~1e6;
* background: one 'free' polygon covering the whole extent;
* barrier: a forbidden wall of thickness T spanning the extent vertically, its
  left edge at x = 350030 + offset, carrying ONE gate of width G centred on the
  extent;
* T in {0.4, 2.0} m - thin (option C widens it) and fat (option C skips it);
* offset in {0.000, 0.125, ..., 0.875} m - 8 sub-pixel alignments;
* G: 8 widths, and FOUR different sets of them, because that is the axis the
  result actually moves along;
* denominator: only the configurations in which the PLAIN rule leaves the left
  edge connected to the right edge through free space, since a configuration
  the plain rule already seals cannot be sealed by a repair. 4-connectivity,
  because a pyorps router admits a diagonal step only when both flanking cells
  are passable; the script prints the 8-connected counts too and they have been
  identical in every run so far.

WHAT THE OUTPUT SHOWS, and the only part that transfers: the plain rule and the
default seal 0 of N in every row; on the FAT wall C seals 0 of N in every row
while D seals a large fraction; on the THIN wall both seal a substantial
fraction and neither is reliably better than the other.
"""
import warnings

import geopandas as gpd
import numpy as np
from scipy import ndimage
from shapely.geometry import box

from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer

RES = 1.0
N = 64
X0, Y0 = 350000.0, 5600000.0
BBOX = box(X0, Y0, X0 + N * RES, Y0 + N * RES)
CRS = "EPSG:25832"
MID = Y0 + N / 2.0
COSTS = {'category': {'free': 10, 'barrier': IMPASSABLE_CELL_COST}}
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

BACKGROUND = (box(X0, Y0, X0 + N, Y0 + N), 'free')
OFFSETS = [i / 8.0 for i in range(8)]

#: The axis the numbers actually move along. Every set is 8 widths.
GATE_WIDTH_SETS = {
    "0.4 .. 1.8 step 0.20": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
    "0.6 .. 2.7 step 0.30": [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7],
    "0.5 .. 4.0 step 0.50": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "0.25 .. 2.0 step 0.25": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
}


def burn(features, **kwargs):
    """Rasterize ``[(geometry, category), ...]`` through the real API."""
    gdf = gpd.GeoDataFrame({'category': [c for _, c in features]},
                           geometry=[g for g, _ in features], crs=CRS)
    rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf, crs=CRS), COSTS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rasterizer.rasterize(resolution_in_m=RES, bounding_box=BBOX, **kwargs)
    return rasterizer.raster


def left_right_connected(grid, four=True):
    labels, _ = ndimage.label(grid != IMPASSABLE_CELL_COST,
                              structure=FOUR if four else None)
    left = set(labels[:, 0].tolist()) - {0}
    right = set(labels[:, -1].tolist()) - {0}
    return bool(left & right)


def wall_with_gate(thickness, gate, offset):
    x0 = X0 + 30.0 + offset
    half = gate / 2.0
    return box(x0, Y0, x0 + thickness, MID - half).union(
        box(x0, MID + half, x0 + thickness, Y0 + N))


def sweep(thickness, gates, four=True):
    """``(plain-open configurations, sealed by C, sealed by D)``."""
    n_open = c_seal = d_seal = 0
    for gate in gates:
        for offset in OFFSETS:
            features = [BACKGROUND,
                        (wall_with_gate(thickness, gate, offset), 'barrier')]
            if not left_right_connected(burn(features), four):
                continue
            n_open += 1
            if not left_right_connected(
                    burn(features, widen_thin_forbidden=True), four):
                c_seal += 1
            if not left_right_connected(
                    burn(features, all_touched=True), four):
                d_seal += 1
    return n_open, c_seal, d_seal


def main():
    print(f"{'gate widths':22s} {'wall':>6s} {'conn':>6s} {'open':>5s} "
          f"{'C':>14s} {'D':>14s}")
    for name, gates in GATE_WIDTH_SETS.items():
        for thickness in (0.4, 2.0):
            for four in (True, False):
                n, c, d = sweep(thickness, gates, four)
                pct = 100.0 / max(n, 1)
                print(f"{name:22s} {thickness:5.1f}m {'4' if four else '8':>5s}- "
                      f"{n:5d} {c:5d}/{n:<3d}({c * pct:5.1f}%) "
                      f"{d:5d}/{n:<3d}({d * pct:5.1f}%)", flush=True)


if __name__ == "__main__":
    main()

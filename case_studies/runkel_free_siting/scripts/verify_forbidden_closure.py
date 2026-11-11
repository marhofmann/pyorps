"""Check that least-cost routes never traverse a forbidden land-use feature (owner's requirement, 2026-10-05).

For random windows of a cost raster it runs the kernel's own seeded Dijkstra with the widest step set
(r3: knight and (1,3), (2,3) steps), traces routes to random targets, and tests every route twice:

1. **cells**: no step of the route has a positive-length chord inside a forbidden (65535) cell, the exact
   chord computed on the polyline through the cell centres;
2. **vectors**: the route polyline has zero-length intersection with the union of the forbidden land-use
   polygons themselves (so the guarantee holds against the vector data, not only against the raster).

A closed cost map must give 0 violations. The same check on the first, plain burn is reported for
comparison. Used by ``tests/test_siting/test_forbidden_closure.py`` on a small window.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/verify_forbidden_closure.py --windows 6
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_recipe as rec  # noqa: E402

CASE = HERE.parent
FORBIDDEN = rec.FORBIDDEN


def chord_cells(r0: int, c0: int, r1: int, c1: int):
    """Cells (row, col) that the segment between two cell centres crosses with positive length."""
    a0, b0, a1, b1 = r0 + 0.5, c0 + 0.5, r1 + 0.5, c1 + 0.5
    dr, dc = a1 - a0, b1 - b0
    ts = {0.0, 1.0}
    for k in range(math.floor(min(a0, a1)) + 1, math.ceil(max(a0, a1))):
        if dr:
            ts.add((k - a0) / dr)
    for k in range(math.floor(min(b0, b1)) + 1, math.ceil(max(b0, b1))):
        if dc:
            ts.add((k - b0) / dc)
    ts = sorted(t for t in ts if 0.0 <= t <= 1.0)
    out = []
    for ta, tb in zip(ts, ts[1:]):
        if tb - ta > 1e-12:
            tm = 0.5 * (ta + tb)
            out.append((int(math.floor(a0 + dr * tm)), int(math.floor(b0 + dc * tm))))
    return out


def trace(prev: np.ndarray, flat: int, ncols: int):
    """Cell sequence (row, col) from a target back to its seed through the predecessor array."""
    seq = [flat]
    while prev[seq[-1]] >= 0:
        seq.append(int(prev[seq[-1]]))
        if len(seq) > prev.size:
            raise RuntimeError("predecessor loop")
    return [divmod(v, ncols) for v in reversed(seq)]


def check_window(values: np.ndarray, transform, forbidden_union, *, n_seeds: int, n_targets: int,
                 rng: np.random.Generator) -> dict:
    """Route violations in one window. ``forbidden_union`` is a shapely geometry in map coordinates
    (or ``None`` to skip the vector test)."""
    from shapely import prepared
    from shapely.geometry import LineString

    from pyorps.certify.windows import drain
    from pyorps.utils.neighborhood import get_neighborhood_steps

    steps = np.asarray(get_neighborhood_steps(3, directed=True), dtype=np.int8)
    rows, cols = values.shape
    free = np.flatnonzero(values.ravel() != FORBIDDEN)
    prep = prepared.prep(forbidden_union) if forbidden_union is not None else None
    res = dict(routes=0, steps=0, cell_violations=0, vector_violations=0)
    for _ in range(n_seeds):
        seed = int(rng.choice(free))
        dist, prev = drain(values, steps, [seed], [0.0], return_prev=True, engine="auto")
        dist, prev = dist.ravel(), prev.ravel()
        reach = np.flatnonzero(np.isfinite(dist))
        for tgt in rng.choice(reach, size=min(n_targets, reach.size), replace=False):
            cells = trace(prev, int(tgt), cols)
            res["routes"] += 1
            res["steps"] += len(cells) - 1
            bad = False
            for (ra, ca), (rb, cb) in zip(cells, cells[1:]):
                for r, c in chord_cells(ra, ca, rb, cb):
                    if values[r, c] == FORBIDDEN:
                        bad = True
            res["cell_violations"] += int(bad)
            if prep is not None and len(cells) > 1:
                xy = [transform * (c + 0.5, r + 0.5) for r, c in cells]
                line = LineString(xy)
                if prep.intersects(line) and line.intersection(forbidden_union).length > 1e-9:
                    res["vector_violations"] += 1
    return res


def forbidden_union_in(frame, bounds):
    """Union of the forbidden land-use polygons that meet ``bounds`` (xmin, ymin, xmax, ymax)."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    forb = frame.loc[rec.forbidden_mask(frame)]
    sel = forb[forb.intersects(box(*bounds))]
    return unary_union(list(sel.geometry)) if len(sel) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--size", type=int, default=1200, help="window side in cells")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args(argv)

    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window

    from pyorps.siting.data_manifest import DataManifest

    m = DataManifest.load(CASE / "data" / "data_manifest.yaml")
    frame = gpd.read_file(m.file_of("alkis_land_use"))
    totals = {}
    paths = {"closed (cost_base_1m_closed)": m.file_of("cost_base_1m_closed")}
    if m.layer("cost_base_1m").status == "present":
        paths["first plain burn (cost_base_1m)"] = m.file_of("cost_base_1m")
    for label, path in paths.items():
        totals[label] = dict(routes=0, steps=0, cell_violations=0, vector_violations=0)
    with rasterio.open(next(iter(paths.values()))) as ref:
        H, W = ref.height, ref.width
    rng_w = np.random.default_rng(args.seed)
    for k in range(args.windows):
        r0 = int(rng_w.integers(0, H - args.size)); c0 = int(rng_w.integers(0, W - args.size))
        for label, path in paths.items():
            with rasterio.open(path) as src:
                win = Window(c0, r0, args.size, args.size)
                values = src.read(1, window=win)
                tr = src.window_transform(win)
            xs, ys = tr * (0, args.size), tr * (args.size, 0)
            union = forbidden_union_in(frame, (xs[0], xs[1], ys[0], ys[1]))
            r = check_window(values, tr, union, n_seeds=args.seeds, n_targets=args.targets,
                             rng=np.random.default_rng(args.seed + k))
            for key, v in r.items():
                totals[label][key] += v
    for label, t in totals.items():
        print(f"{label}: {t['routes']} routes, {t['steps']} steps; routes through a forbidden cell: "
              f"{t['cell_violations']}; routes through a forbidden polygon: {t['vector_violations']}")
    closed = totals["closed (cost_base_1m_closed)"]
    return 0 if closed["cell_violations"] == 0 and closed["vector_violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

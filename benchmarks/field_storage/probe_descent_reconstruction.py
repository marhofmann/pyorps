"""Can the predecessor plane be thrown away and the route re-derived on read?

A saved field stores two planes: the cost `dist` (4 B/cell) and a predecessor
`pred_step` (1 B/cell) that exists only so a reopened field can return routes.
But PYORPS weights every edge as

    w(u,v) = (raster[u] + sum(intermediates) + raster[v])
             * sqrt(dr^2+dc^2) / (2 + n_inter) * cell_size

which is SYMMETRIC in u and v. So for the true predecessor u of v,
dist[u] + w(u,v) == dist[v] exactly, and u can be recovered by testing the
neighbours -- no stored plane needed. This probe checks that claim on real
data, and then checks how far it survives when `dist` is lossily quantised,
because that is the combination that would actually be shipped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent))
from probe_encodings import tile_quantise            # noqa: E402

import pyorps                                                # noqa: E402
from pyorps.graph.search_session import CostField            # noqa: E402

SRC = Path(r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
           r"cired2026/raster/mod1_raster_wp_fixed.tiff")
HERE = Path(__file__).parent
CROP = HERE / "descent_crop.tiff"
FIELD = HERE / "descent_field.npz"
N = 1600
NBH = "r2"


def make_crop():
    if CROP.exists():
        return
    with rasterio.open(SRC) as r:
        w = Window(9000, 4000, N, N)
        arr = r.read(1, window=w)
        prof = r.profile | {"height": N, "width": N,
                            "transform": r.window_transform(w),
                            "compress": "deflate"}
    with rasterio.open(CROP, "w", **prof) as d:
        d.write(arr, 1)
    print(f"crop {N}x{N}  values {arr.min()}..{arr.max()}  "
          f"impassable {100 * (arr >= 65535).mean():.1f} %")


def build_field():
    with rasterio.open(CROP) as r:
        tr, (h, w) = r.transform, (r.height, r.width)
    ox, oy = tr * (w / 2, h / 2)
    finder = pyorps.PathFinder(dataset_source=str(CROP),
                               source_coords=(ox, oy), target_coords=[(ox, oy)],
                               search_space_buffer_m=4 * N,
                               neighborhood_str=NBH, graph_api="cython",
                               ignore_max_cost=True)
    t0 = time.perf_counter()
    with finder.cost_field((ox, oy), algorithm="dijkstra") as f:
        f.settle_all()
        print(f"settled in {time.perf_counter() - t0:.2f} s")
        f.save(FIELD, with_paths=True, compress=False)
    return (ox, oy)


def neighbour_table(steps, raster, cell_size):
    """Per-direction (flat offset, intermediate offsets, factor)."""
    from pyorps.utils.metric_edges import _intermediate_offsets
    cols = raster.shape[1]
    out = []
    for dr, dc in steps:
        inter = np.asarray(_intermediate_offsets(int(dr), int(dc)),
                           dtype=np.int64).reshape(-1, 2)
        off_i = inter[:, 0] * cols + inter[:, 1] if inter.size else \
            np.empty(0, dtype=np.int64)
        fac = np.hypot(dr, dc) / (2.0 + len(off_i)) * cell_size
        out.append((int(dr) * cols + int(dc), off_i, fac))
    return out


def step_w(u, cur, off, inter, fac, flat):
    tot = float(flat[u]) + float(flat[cur])
    for io in inter:
        tot += float(flat[u + io])
    return tot * fac


def descend(v, dist, flat, table, origin_idx):
    """Walk v -> origin using `dist` alone.

    The rule is the textbook one and needs no tolerance: Dijkstra optimality
    says dist[v] == min over neighbours u of (dist[u] + w(u,v)), so the argmin
    IS a valid predecessor. Requiring a strict decrease in the stored value on
    top of that is what keeps a quantised field from walking a 2-cycle between
    two cells whose codes collide.
    """
    walk = [v]
    cur = v
    for _ in range(4 * int(np.sqrt(dist.size)) * 8):
        if cur == origin_idx:
            walk.reverse()
            return np.array(walk, dtype=np.int64)
        best, best_val = -1, np.inf
        for off, inter, fac in table:
            u = cur - off
            if u < 0 or u >= dist.size or not np.isfinite(dist[u]):
                continue
            if dist[u] >= dist[cur]:
                continue
            val = dist[u] + step_w(u, cur, off, inter, fac, flat)
            if val < best_val:
                best, best_val = u, val
        if best < 0:
            return np.empty(0, dtype=np.int64)
        cur = best
        walk.append(cur)
    return np.empty(0, dtype=np.int64)


def walk_cost(walk, flat, table, cols):
    """True accumulated cost of a cell walk, under PYORPS' own edge rule."""
    by_off = {off: (inter, fac) for off, inter, fac in table}
    tot = 0.0
    for a, b in zip(walk[:-1], walk[1:]):
        off = int(b) - int(a)
        if off not in by_off:
            return np.inf
        inter, fac = by_off[off]
        tot += step_w(int(a), int(b), off, inter, fac, flat)
    return tot


def main():
    make_crop()
    origin = build_field()
    saved = CostField.open(FIELD)
    rows, cols = saved.shape
    dist = np.asarray(saved.field_array(np.float64)).ravel()
    with rasterio.open(CROP) as r:
        flat = r.read(1).astype(np.float64).ravel()
        cell = abs(r.transform[0])
    steps = np.asarray(saved._steps)[:, :2]
    table = neighbour_table(steps, np.empty((rows, cols)), cell)
    oidx = saved._origin_idx
    print(f"{len(table)} directions, cell {cell:.3f} m, origin {oidx}")

    rng = np.random.default_rng(7)
    ok = np.flatnonzero(np.isfinite(dist))
    targets = rng.choice(ok, 300, replace=False)

    # quantised copy, same encoder as the compression probe
    q, deq, _ = tile_quantise(np.where(np.isfinite(dist), dist, np.nan)
                              .reshape(rows, cols), 16)
    dq = deq.ravel()
    dq[~np.isfinite(dist)] = np.inf
    print(f"quantisation: max understatement {np.nanmax(dist[ok] - dq[ok]):.4f} EUR")

    res = {"exact": {"n": 0, "same": 0, "dcost": []},
           "quant": {"n": 0, "same": 0, "dcost": []}}
    npred = 0
    ref_costs = []
    t_pred = t_ex = t_q = 0.0
    for v in targets:
        t0 = time.perf_counter()
        r_, c_ = divmod(int(v), cols)
        x, y = xy(saved.transform, r_, c_)
        ref = saved.path_cells((x, y))
        t_pred += time.perf_counter() - t0
        if ref.size == 0:
            continue
        npred += 1
        c_ref = walk_cost(ref, flat, table, cols)
        ref_costs.append(c_ref)
        for key, d in (("exact", dist), ("quant", dq)):
            t0 = time.perf_counter()
            got = descend(int(v), d, flat, table, oidx)
            dt = time.perf_counter() - t0
            if key == "exact":
                t_ex += dt
            else:
                t_q += dt
            r = res[key]
            if got.size and got[0] == oidx and got[-1] == v:
                r["n"] += 1
                if got.size == ref.size and np.array_equal(got, ref):
                    r["same"] += 1
                r["dcost"].append(walk_cost(got, flat, table, cols) - c_ref)

    print(f"\n{npred} routes recovered from the stored predecessor plane")
    for key in ("exact", "quant"):
        r = res[key]
        dc = np.array(r["dcost"]) if r["dcost"] else np.array([np.nan])
        rc = np.array(ref_costs)[:dc.size]
        rel = np.abs(dc) / np.maximum(np.abs(rc), 1.0)
        n_ok, n_same = r["n"], r["same"]
        print(f"  descent on {key:<5} dist: reached the origin {n_ok}/{npred}"
              f", cell-for-cell identical {n_same}/{npred}")
        print(f"      route cost vs the stored route: mean {dc.mean():+.4f}"
              f" EUR, worst {dc.max():+.4f}, best {dc.min():+.4f}, "
              f"worst relative {rel.max():.2e}")
    print(f"\ntime per route: pred walk {1e3 * t_pred / npred:.2f} ms, "
          f"descent exact {1e3 * t_ex / npred:.2f} ms, "
          f"descent quantised {1e3 * t_q / npred:.2f} ms")
    saved.close()


if __name__ == "__main__":
    main()

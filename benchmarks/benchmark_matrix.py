"""
Matrix / batch benchmark: a 19 x 19 meshed source-target route matrix.

The dominant real workload is a MATRIX, not a single pair. It has had zero
benchmark representation, so the batch levers of the 5 km performance plan
(per-pair corridor cropping, GPU device-session persistence, same-source
dedup, pair-level pipelining) have nothing to be accepted against.

Access patterns mirrored from the real drivers (do not "improve" them —
they are the measurement subject):

  star     ``case_studies/substation_planning/find_paths_for_meshed_network``
           -> ``substation_planning.create_meshed_network`` /
           ``find_and_save_paths``: ONE PathFinder per source, built with
           that source plus all of its targets, ``search_space_buffer_m=3000``,
           ``neighborhood_str="r2"``, ``algorithm="dijkstra"``, then
           ``create_path_geodataframe()`` -> ``to_file()`` per source.
  shared   ``examples/batch_route_planning_pandapower.route_all_pairs``: ONE
           PathFinder over all source+target coordinates (hull window, cached
           graph API), one ``find_route(source=s, target=[...])`` per source,
           a single export at the end.
  pairwise ``examples/preprocess_mv_connections`` / GA inner loops: one
           ``find_route`` per (source, target) pair — the pattern the
           same-source-dedup lever (plan 1.5) attacks. Measured on a sample
           (``--pairwise-limit``) and extrapolated, because 361 discrete CPU
           solves take hours.

What is decomposed per pair (the plan's claim is that non-solver work
dominates at matrix scale):

  setup_s      PathFinder construction (raster window + buffer mask) and
               graph-API construction, charged where the driver pays it
  solver_s     ``PathFinder.runtimes["shortest_path"]`` — the kernel call.
               Path reconstruction happens INSIDE this call for every
               backend (cython returns paths; raster_gpu walks ``pred`` on
               the host). ``gpu_split`` separates the two for raster_gpu.
  geometry_s   find_route minus solver: node->coord conversion, LineString,
               Path objects
  metrics_s    ``PathFinder.calculate_path_metrics`` per path — contains the
               ``np.sort(np.unique(raster))`` full-window scan
               (``_traversal_numba.py:624``), charged once PER PATH
  export_s     ``create_path_geodataframe()`` + ``to_file()``

Probes that name the levers directly:

  unique_scan          ``np.sort(np.unique(window_raster))`` — the fixed
                       per-path tax of the reporting stage (plan 1.1)
  cpu_context_build    ``RasterContext(raster, steps, max_value)`` — rebuilt
                       per public cython call (plan 1.7)
  gpu_per_query_setup  ``_common_gpu_setup``: host uint16 cast, H2D raster
                       upload, auto-delta full-raster scan, dist/pred alloc
                       — paid per query (plan 1.2, ``sssp_gpu.py:2052-2057``)
  gpu_d2h              full-field ``dist``+``pred`` download per query
                       (plan 1.3)
  gpu_split            one extra solve on raster_gpu with ``_run_sssp`` and
                       ``_extract_path`` timed apart (plan 1.4)

Also reported: peak host RSS, CuPy pool high-water when GPU backends run,
and the extent-clipped geometric corridor ratio (hull-buffer area vs each
pair's own line-buffer area) that sizes the per-pair cropping lever
(plan 5.2) without implementing it. Note that the ratio is ~1 at the case
study's 3000 m buffer on a 5 km window — that lever needs a state-wide
extent or a tighter buffer to pay.

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_matrix.py --small
    .venv/Scripts/python.exe benchmarks/benchmark_matrix.py
    ... [--size 5000] [--sources 19] [--targets 19] [--buffer 3000]
    ... [--backends cython,raster_gpu,raster_fim] [--modes star,shared]
    ... [--pairwise-limit 20] [--no-export]

The full run is expensive (cython solves 25 M cells per source; expect
tens of minutes). GPU etiquette: backends run strictly sequentially and the
CuPy pool is freed between them.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from math import hypot
from pathlib import Path

import numpy as np
import psutil
from rasterio.transform import array_bounds, from_origin
from shapely.geometry import box, LineString, MultiPoint

from pyorps import PathFinder
from pyorps.core.exceptions import NoPathFoundError
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.utils.traversal import _CYTHON_TRAVERSAL

HERE = Path(__file__).parent
DATA = HERE / "realworld_data"
RESULTS = HERE / "results"

SENTINEL = np.uint16(65535)
CRS = "EPSG:25832"

#: Fallback origin when the cached real window is unavailable (--small).
FALLBACK_ORIGIN = (477500.0, 5598596.0)

BACKEND_ALGORITHM = {
    "cython": "dijkstra",
    "raster_gpu": "delta-stepping",
    "raster_fim": "fim",
}
GPU_BACKENDS = ("raster_gpu", "raster_fim")


# ----------------------------------------------------------------------
# Raster preparation
# ----------------------------------------------------------------------

def load_real_window():
    """The cached real 4096^2 planning window (cost, (minx, maxy))."""
    cached = DATA / "window_cost.npz"
    if not cached.exists():
        raise FileNotFoundError(
            f"{cached} is missing — run benchmark_eikonal_realworld.py "
            f"once to cache the real cost window, or use --small.")
    z = np.load(cached)
    bounds = tuple(float(v) for v in z["bounds"])
    return z["cost"], (bounds[0], bounds[3])


def mosaic(tile, size):
    """Mirror-tile ``tile`` up to ``size`` x ``size``.

    Mirroring keeps the seams continuous, so the cost statistics and the
    local structure the solvers see stay those of the real raster.
    """
    n = tile.shape[0]
    reps = -(-size // n)
    big = np.empty((reps * n, reps * n), dtype=tile.dtype)
    for i in range(reps):
        rows = tile[::-1, :] if i % 2 else tile
        for j in range(reps):
            block = rows[:, ::-1] if j % 2 else rows
            big[i * n:(i + 1) * n, j * n:(j + 1) * n] = block
    return np.ascontiguousarray(big[:size, :size])


def synthetic_window(size, seed=7):
    """Small connected obstacle raster for --small (no cached data needed)."""
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 60, size=(size, size), dtype=np.uint16)
    # Heavy-cost blobs plus vertical walls with a gap, so the paths bend.
    for _ in range(6):
        r, c = rng.integers(0, size - 40, 2)
        cost[r:r + 40, c:c + 40] = rng.integers(200, 900)
    for k in range(1, 4):
        c = k * size // 4
        cost[:, c:c + 3] = SENTINEL
        # Gap kept central: the buffer mask clips the raster corners, and a
        # masked-away gap would make the smoke test spuriously unreachable.
        gap = 3 * size // 8
        cost[gap:gap + size // 4, c:c + 3] = 40
    return cost


# ----------------------------------------------------------------------
# Point placement
# ----------------------------------------------------------------------

def snap_passable(cost, r, c, max_radius=300):
    """Nearest passable cell to (r, c) — substations land on buildings."""
    rows, cols = cost.shape
    for rad in range(0, max_radius, 10):
        rs = slice(max(r - rad - 10, 0), min(r + rad + 11, rows))
        cs = slice(max(c - rad - 10, 0), min(c + rad + 11, cols))
        ok = np.argwhere(cost[rs, cs] != SENTINEL)
        if ok.size:
            d2 = ((ok[:, 0] + rs.start - r) ** 2
                  + (ok[:, 1] + cs.start - c) ** 2)
            rr, cc = ok[int(d2.argmin())]
            return int(rr + rs.start), int(cc + cs.start)
    raise ValueError(f"no passable cell near ({r},{c})")


def place_points(cost, n, seed, offset=0.0):
    """``n`` scattered points on a jittered grid, snapped to passable cells.

    Scattered — not two opposing clusters — because that is what a meshed
    substation network looks like: every pair is solved on the hull of all
    of them, so the searched area is decoupled from the pair's own extent
    (``corridor_stats`` quantifies that).
    """
    rows, cols = cost.shape
    rng = np.random.default_rng(seed)
    g = int(np.ceil(np.sqrt(n)))
    margin = 0.06
    span = 1.0 - 2 * margin
    cells = []
    for k in range(g * g):
        gi, gj = divmod(k, g)
        fr = margin + span * (gi + 0.5 + offset) / g
        fc = margin + span * (gj + 0.5 + offset) / g
        fr += float(rng.uniform(-0.35, 0.35)) * span / g
        fc += float(rng.uniform(-0.35, 0.35)) * span / g
        r = int(min(max(fr, 0.02), 0.98) * rows)
        c = int(min(max(fc, 0.02), 0.98) * cols)
        cells.append(snap_passable(cost, r, c))
    order = rng.permutation(len(cells))[:n]
    return [cells[i] for i in sorted(order)]


def cells_to_coords(cells, transform):
    """Cell centers as (x, y) map coordinates.

    Explicit affine arithmetic rather than ``rasterio.transform.xy`` so the
    scalar/array return shape cannot surprise us; ``rowcol`` inverts pixel
    centers back to the same cell exactly.
    """
    return [(transform.c + (c + 0.5) * transform.a,
             transform.f + (r + 0.5) * transform.e) for r, c in cells]


# ----------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------

class Clock:
    """Wall-clock accumulator with a context-manager face."""

    def __init__(self):
        self.t0 = None
        self.last = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.last = time.perf_counter() - self.t0
        return False


def median_of(fn, reps=3):
    """Median wall-clock of ``reps`` calls; the result of the last call."""
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def peak_rss_bytes():
    """Peak working set (Windows) or current RSS elsewhere."""
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def rss_bytes():
    return int(psutil.Process().memory_info().rss)


def _cupy():
    """The cupy module if importable, else None (never raises)."""
    try:
        import cupy as cp
        return cp
    except Exception:
        return None


def gpu_pool_stats():
    """CuPy default-pool occupancy; ``total_bytes`` is the high-water."""
    cp = _cupy()
    if cp is None:
        return None
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def free_gpu_pool():
    cp = _cupy()
    if cp is None:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def gpu_environment():
    """Device name, clocks and temperature (plan 0.12 variance protocol)."""
    query = ("name,clocks.sm,clocks.max.sm,temperature.gpu,"
             "utilization.gpu,memory.used,memory.total")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True)
        row = out.stdout.strip().splitlines()[0]
        return dict(zip(query.split(","),
                        [f.strip() for f in row.split(",")]))
    except Exception as exc:
        return {"error": str(exc)}


# ----------------------------------------------------------------------
# Stage probes — these name the levers the matrix is meant to accept
# ----------------------------------------------------------------------

def probe_unique_scan(raster):
    """Per-path fixed cost of the reporting stage (plan 1.1)."""
    t, _ = median_of(lambda: np.sort(np.unique(raster)))
    return t


def probe_cpu_context(raster, steps):
    """RasterContext build, rebuilt by every public cython call (plan 1.7)."""
    try:
        from pyorps.utils._raster_context import RasterContext
    except ImportError as exc:
        return {"error": str(exc)}
    t, _ = median_of(
        lambda: RasterContext(raster, steps, np.uint16(65535)))
    return {"context_build_s": t}


def probe_gpu_per_query(raster, steps, source_idx):
    """Per-query GPU tax: host cast, H2D, auto-delta, alloc, D2H.

    Mirrors ``sssp_gpu._common_gpu_setup`` (``sssp_gpu.py:2035-2062``) and
    the full-field download (``:1644-1650``) — every item is paid again on
    every query today, which is what plan items 1.2/1.3 remove.
    """
    cp = _cupy()
    if cp is None:
        return {"error": "cupy unavailable"}
    from pyorps.utils.sssp_gpu import _common_gpu_setup, _compute_auto_delta
    from pyorps.utils.traversal_gpu import prepare_step_lookup_tables

    _steps, _lut, _n_inter, cost_factors = prepare_step_lookup_tables(steps)
    n_pixels = raster.size

    def _sync(fn):
        def wrapped():
            out = fn()
            cp.cuda.runtime.deviceSynchronize()
            return out
        return wrapped

    t_cast, _ = median_of(lambda: raster.astype(np.uint16))
    t_h2d, _ = median_of(_sync(lambda: cp.asarray(raster.astype(np.uint16))))
    t_delta, delta = median_of(
        lambda: _compute_auto_delta(raster, cost_factors, True))
    t_setup, ctx = median_of(_sync(
        lambda: _common_gpu_setup(raster, steps, int(source_idx), "auto",
                                  True, True)))
    del ctx
    d_dist = cp.zeros(n_pixels, dtype=cp.float32)
    d_pred = cp.zeros(n_pixels, dtype=cp.int32)
    t_d2h, _ = median_of(
        _sync(lambda: (cp.asnumpy(d_dist), cp.asnumpy(d_pred))))
    del d_dist, d_pred
    free_gpu_pool()
    return {
        "host_uint16_cast_s": t_cast,
        "h2d_raster_s": t_h2d,
        "auto_delta_s": t_delta,
        "common_setup_s": t_setup,
        "d2h_dist_pred_s": t_d2h,
        "auto_delta_value": float(delta),
        "raster_mib": raster.nbytes / 2 ** 20,
        "dist_pred_mib": n_pixels * 8 / 2 ** 20,
    }


def probe_gpu_solve_split(path_finder, source_coord, target_coords):
    """Split one raster_gpu query into solve and predecessor walk (plan 1.4).

    Uses the API the PathFinder already built, so no second upload of the
    step tables and no second graph construction is charged here.
    """
    api = path_finder.graph_api
    if not hasattr(api, "_run_sssp"):
        return {"error": "backend has no _run_sssp"}
    src = int(path_finder.get_node_indices_from_coords(source_coord))
    tgts = np.asarray(path_finder.get_node_indices_from_coords(
        list(target_coords)), dtype=np.int32)
    with Clock() as c:
        dist, pred = api._run_sssp(src, tgts)
    solve_s = c.last
    with Clock() as c:
        for t in tgts:
            try:
                api._extract_path(pred, dist, src, int(t))
            except NoPathFoundError:
                pass
    return {
        "run_sssp_s": solve_s,
        "extract_paths_s": c.last,
        "n_targets": int(tgts.size),
        "extract_share_pct": 100 * c.last / max(solve_s + c.last, 1e-12),
    }


# ----------------------------------------------------------------------
# Corridor geometry (plan 5.2 sizing, no solving involved)
# ----------------------------------------------------------------------

def corridor_stats(source_coords, target_coords, buffer_m, cell_size,
                   extent, window_cells):
    """Hull-buffer area vs each pair's own line-buffer area, extent-clipped.

    Every pair in a matrix is solved on the buffered convex hull of ALL
    endpoints. The ratio below is the factor by which per-pair cropping
    (plan 5.2) would shrink the searched area — geometric, so it holds
    regardless of backend. Both geometries are clipped to the raster
    extent, so the ratio is >= 1 (a pair's segment lies inside the hull,
    hence its buffer lies inside the hull buffer).

    The ratio collapses toward 1 when the buffer is large relative to the
    extent: at buffer 3000 m on a 5 km window every pair already covers
    the whole raster. The lever needs the real state-wide extent (or a
    tighter buffer) to pay — that is the honest reading of a ~1 here.
    """
    hull = MultiPoint(list(source_coords) + list(target_coords)).convex_hull
    hull_area = hull.buffer(buffer_m, quad_segs=32).intersection(
        extent).area / cell_size ** 2
    ratios, pair_cells = [], []
    for s in source_coords:
        for t in target_coords:
            if s == t:
                continue
            area = LineString([s, t]).buffer(
                buffer_m, quad_segs=32).intersection(
                    extent).area / cell_size ** 2
            pair_cells.append(area)
            ratios.append(hull_area / max(area, 1.0))
    return {
        "buffer_m": float(buffer_m),
        "window_cells": int(window_cells),
        "hull_buffer_cells_clipped": float(hull_area),
        "pair_corridor_cells_median": float(statistics.median(pair_cells)),
        "pair_corridor_cells_min": float(min(pair_cells)),
        "searched_over_pair_corridor_median": float(statistics.median(ratios)),
        "searched_over_pair_corridor_max": float(max(ratios)),
    }


# ----------------------------------------------------------------------
# Matrix drivers
# ----------------------------------------------------------------------

def _empty_totals():
    return {
        "raster_copy_s": 0.0,
        "setup_s": 0.0,
        "graph_api_s": 0.0,
        "solver_s": 0.0,
        "geometry_s": 0.0,
        "metrics_s": 0.0,
        "export_s": 0.0,
        "n_solver_calls": 0,
        "n_paths": 0,
        "n_failed_pairs": 0,
    }


def _finalize(totals, wall_s):
    solver = totals["solver_s"]
    totals["total_s"] = wall_s
    totals["non_solver_s"] = wall_s - solver
    totals["non_solver_pct"] = 100 * (wall_s - solver) / max(wall_s, 1e-12)
    totals["solver_s_per_call"] = solver / max(totals["n_solver_calls"], 1)
    totals["metrics_s_per_path"] = (
        totals["metrics_s"] / max(totals["n_paths"], 1))
    return totals


def _measure_paths(path_finder, collection, target_of_index, source_id,
                   totals, per_pair):
    """Charge path metrics per returned path and record the decomposition.

    Paths are matched to targets by their final node index — a one-to-many
    call silently drops unreachable targets, so positional matching is
    wrong (see examples/preprocess_mv_connections.py:253).
    """
    for path in collection:
        with Clock() as c:
            path_finder.calculate_path_metrics(path.path_indices, path)
        totals["metrics_s"] += c.last
        totals["n_paths"] += 1
        end = int(path.path_indices[-1])
        per_pair.append({
            "source_id": source_id,
            "target_id": target_of_index.get(end, -1),
            "n_cells": len(path.path_indices),
            "length_m": path.total_length,
            "total_cost": path.total_cost,
            "euclid_m": path.euclidean_distance,
            "metrics_s": c.last,
        })


def _as_collection(result):
    """find_route returns a Path for 1:1 and a PathCollection otherwise."""
    return [result] if hasattr(result, "path_indices") else list(result)


def run_star(raster, transform, sources, targets, backend, buffer_m,
             neighborhood, export_dir):
    """One PathFinder per source — the substation case-study driver."""
    algorithm = BACKEND_ALGORITHM[backend]
    totals, per_pair, per_source = _empty_totals(), [], []
    t_wall = time.perf_counter()
    for i, src in enumerate(sources):
        rec = {"source_id": i}
        # The buffer mask writes the sentinel THROUGH a view into the
        # parent array (plan 2.1), so every finder needs pristine input.
        # The real driver pays a full GeoTIFF re-read here instead.
        with Clock() as c:
            work = raster.copy()
        totals["raster_copy_s"] += c.last
        rec["raster_copy_s"] = c.last

        with Clock() as c:
            path_finder = PathFinder(
                dataset_source=work,
                source_coords=src,
                target_coords=list(targets),
                search_space_buffer_m=buffer_m,
                neighborhood_str=neighborhood,
                graph_api=backend,
                crs=CRS,
                transform=transform,
            )
        totals["setup_s"] += c.last
        rec["setup_s"] = c.last

        # Materialize the graph API outside find_route: otherwise its cost
        # lands inside runtimes["shortest_path"] (plan 0.4).
        with Clock() as c:
            _ = path_finder.graph_api
        totals["graph_api_s"] += c.last
        rec["graph_api_s"] = c.last

        idx = np.asarray(
            path_finder.get_node_indices_from_coords(list(targets)))
        target_of_index = {int(v): j for j, v in enumerate(idx)}

        try:
            with Clock() as c:
                result = path_finder.find_route(
                    source=src, target=list(targets), algorithm=algorithm,
                    calculate_metrics=False)
            solver = path_finder.runtimes["shortest_path"]
            rec["solver_s"] = solver
            rec["geometry_s"] = c.last - solver
            totals["solver_s"] += solver
            totals["geometry_s"] += c.last - solver
            totals["n_solver_calls"] += 1
            collection = _as_collection(result)
        except NoPathFoundError as exc:
            rec["error"] = str(exc)
            totals["n_failed_pairs"] += len(targets)
            per_source.append(rec)
            continue

        n_before = totals["n_paths"]
        _measure_paths(path_finder, collection, target_of_index, i,
                       totals, per_pair)
        rec["n_paths"] = totals["n_paths"] - n_before
        totals["n_failed_pairs"] += len(targets) - rec["n_paths"]

        if export_dir is not None:
            with Clock() as c:
                gdf = path_finder.create_path_geodataframe()
                gdf.to_file(export_dir / f"star_{backend}_{i:02d}.geojson")
            totals["export_s"] += c.last
            rec["export_s"] = c.last
        rec["window_cells"] = int(path_finder.raster_handler.data[0].size)
        rec["rss_bytes"] = rss_bytes()
        per_source.append(rec)
        del path_finder, work
    wall_s = time.perf_counter() - t_wall
    return _finalize(totals, wall_s), per_source, per_pair


def run_shared(raster, transform, sources, targets, backend, buffer_m,
               neighborhood, export_dir):
    """One PathFinder for the whole matrix — the batch-example driver."""
    algorithm = BACKEND_ALGORITHM[backend]
    totals, per_pair, per_source = _empty_totals(), [], []
    t_wall = time.perf_counter()
    with Clock() as c:
        work = raster.copy()
    totals["raster_copy_s"] = c.last

    with Clock() as c:
        path_finder = PathFinder(
            dataset_source=work,
            source_coords=list(sources),
            target_coords=list(targets),
            search_space_buffer_m=buffer_m,
            neighborhood_str=neighborhood,
            graph_api=backend,
            crs=CRS,
            transform=transform,
        )
    totals["setup_s"] = c.last

    with Clock() as c:
        _ = path_finder.graph_api
    totals["graph_api_s"] = c.last

    idx = np.asarray(path_finder.get_node_indices_from_coords(list(targets)))
    target_of_index = {int(v): j for j, v in enumerate(idx)}

    for i, src in enumerate(sources):
        rec = {"source_id": i}
        try:
            with Clock() as c:
                result = path_finder.find_route(
                    source=src, target=list(targets), algorithm=algorithm,
                    calculate_metrics=False)
            solver = path_finder.runtimes["shortest_path"]
            rec["solver_s"] = solver
            rec["geometry_s"] = c.last - solver
            totals["solver_s"] += solver
            totals["geometry_s"] += c.last - solver
            totals["n_solver_calls"] += 1
            collection = _as_collection(result)
        except NoPathFoundError as exc:
            rec["error"] = str(exc)
            totals["n_failed_pairs"] += len(targets)
            per_source.append(rec)
            continue

        n_before = totals["n_paths"]
        _measure_paths(path_finder, collection, target_of_index, i,
                       totals, per_pair)
        rec["n_paths"] = totals["n_paths"] - n_before
        totals["n_failed_pairs"] += len(targets) - rec["n_paths"]
        rec["rss_bytes"] = rss_bytes()
        per_source.append(rec)

    if export_dir is not None:
        with Clock() as c:
            gdf = path_finder.create_path_geodataframe()
            gdf.to_file(export_dir / f"shared_{backend}.geojson")
        totals["export_s"] = c.last

    wall_s = time.perf_counter() - t_wall   # the probe below is extra work
    totals["window_cells"] = int(path_finder.raster_handler.data[0].size)
    extra = {}
    if backend == "raster_gpu":
        extra["gpu_solve_split"] = probe_gpu_solve_split(
            path_finder, sources[0], targets)
    del path_finder, work
    return _finalize(totals, wall_s), per_source, per_pair, extra


def run_pairwise(raster, transform, sources, targets, backend, buffer_m,
                 neighborhood, limit):
    """One find_route per pair — what same-source dedup (plan 1.5) removes.

    Measured on a stride sample of ``limit`` pairs (spread over all sources,
    not the first source's row) and extrapolated to the full matrix; the
    extrapolation is flagged in the record.
    """
    algorithm = BACKEND_ALGORITHM[backend]
    totals, per_pair = _empty_totals(), []
    all_pairs = [(i, j) for i in range(len(sources))
                 for j in range(len(targets))]
    stride = max(1, len(all_pairs) // max(limit, 1))
    pairs = all_pairs[::stride][:limit]
    t_wall = time.perf_counter()
    with Clock() as c:
        work = raster.copy()
    totals["raster_copy_s"] = c.last

    with Clock() as c:
        path_finder = PathFinder(
            dataset_source=work,
            source_coords=list(sources),
            target_coords=list(targets),
            search_space_buffer_m=buffer_m,
            neighborhood_str=neighborhood,
            graph_api=backend,
            crs=CRS,
            transform=transform,
        )
    totals["setup_s"] = c.last
    with Clock() as c:
        _ = path_finder.graph_api
    totals["graph_api_s"] = c.last

    for i, j in pairs:
        if sources[i] == targets[j]:
            continue
        try:
            with Clock() as c:
                result = path_finder.find_route(
                    source=sources[i], target=targets[j],
                    algorithm=algorithm, calculate_metrics=False)
            solver = path_finder.runtimes["shortest_path"]
            totals["solver_s"] += solver
            totals["geometry_s"] += c.last - solver
            totals["n_solver_calls"] += 1
        except NoPathFoundError:
            totals["n_failed_pairs"] += 1
            continue
        path = _as_collection(result)[0]
        with Clock() as c:
            path_finder.calculate_path_metrics(path.path_indices, path)
        totals["metrics_s"] += c.last
        totals["n_paths"] += 1
        per_pair.append({
            "source_id": i, "target_id": j,
            "n_cells": len(path.path_indices),
            "length_m": path.total_length,
            "solver_s": solver,
            "metrics_s": c.last,
        })

    totals = _finalize(totals, time.perf_counter() - t_wall)
    # Extrapolate the per-pair part only; setup is paid once by the driver.
    fixed = (totals["setup_s"] + totals["graph_api_s"]
             + totals["raster_copy_s"])
    n_measured = max(len(per_pair), 1)
    totals["n_pairs_measured"] = len(per_pair)
    totals["per_pair_s"] = (totals["total_s"] - fixed) / n_measured
    totals["extrapolated_full_matrix_s"] = (
        fixed + totals["per_pair_s"] * len(all_pairs))
    totals["extrapolated"] = True
    del path_finder, work
    return totals, per_pair


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def build_inputs(args):
    """Cost raster, transform and the source/target point sets."""
    if args.small:
        cost = synthetic_window(args.size)
        origin = FALLBACK_ORIGIN
        provenance = f"synthetic {args.size}^2 (seed 7)"
    else:
        tile, origin = load_real_window()
        cost = mosaic(tile, args.size)
        provenance = (f"mirror-mosaic of realworld_data/window_cost.npz "
                      f"({tile.shape[0]}^2) to {args.size}^2")
    transform = from_origin(origin[0], origin[1], 1.0, 1.0)
    src_cells = place_points(cost, args.sources, seed=101)
    tgt_cells = place_points(cost, args.targets, seed=202, offset=0.5)
    return cost, transform, provenance, src_cells, tgt_cells


def describe_pairs(src_cells, tgt_cells):
    d = [hypot(sr - tr, sc - tc)
         for sr, sc in src_cells for tr, tc in tgt_cells
         if (sr, sc) != (tr, tc)]
    return {
        "n_pairs": len(d),
        "distance_m_min": float(min(d)),
        "distance_m_median": float(statistics.median(d)),
        "distance_m_max": float(max(d)),
        "pairs_over_5km_pct": 100 * sum(x > 5000 for x in d) / len(d),
    }


def print_totals(tag, totals):
    print(f"  {tag:<26} total {totals['total_s']:8.2f}s | "
          f"solver {totals['solver_s']:8.2f}s "
          f"({100 - totals['non_solver_pct']:4.1f}%) | "
          f"setup {totals['setup_s'] + totals['graph_api_s']:6.2f}s | "
          f"geom {totals['geometry_s']:6.2f}s | "
          f"metrics {totals['metrics_s']:7.2f}s "
          f"({totals['metrics_s_per_path'] * 1e3:.0f} ms/path) | "
          f"export {totals['export_s']:6.2f}s | "
          f"{totals['n_paths']} paths")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true",
                    help="3x3 on a small synthetic raster, CPU only "
                         "(plumbing smoke test, no GPU, no cached data)")
    ap.add_argument("--size", type=int, default=5000,
                    help="square window edge in cells (1 m resolution)")
    ap.add_argument("--sources", type=int, default=19)
    ap.add_argument("--targets", type=int, default=19)
    ap.add_argument("--buffer", type=float, default=3000.0,
                    help="search_space_buffer_m (case study uses 3000)")
    ap.add_argument("--neighborhood", default="r2")
    ap.add_argument("--backends", default="cython,raster_gpu,raster_fim")
    ap.add_argument("--modes", default=None,
                    help="star,shared[,pairwise] (default: star,shared; "
                         "--small adds pairwise to smoke-test it)")
    ap.add_argument("--pairwise-limit", type=int, default=20,
                    help="pairs measured in the 'pairwise' mode")
    ap.add_argument("--no-export", action="store_true",
                    help="skip create_path_geodataframe + to_file")
    args = ap.parse_args()

    if args.small:
        args.size = 600 if args.size == 5000 else min(args.size, 1200)
        args.sources = min(args.sources, 3)
        args.targets = min(args.targets, 3)
        args.buffer = min(args.buffer, 200.0)
        args.backends = "cython"
        args.pairwise_limit = min(args.pairwise_limit, 9)
    if args.modes is None:
        args.modes = "star,shared,pairwise" if args.small else "star,shared"

    backends = [b for b in args.backends.split(",") if b]
    modes = [m for m in args.modes.split(",") if m]
    unknown = set(backends) - set(BACKEND_ALGORITHM)
    if unknown:
        raise SystemExit(f"unknown backend(s): {sorted(unknown)}")

    cost, transform, provenance, src_cells, tgt_cells = build_inputs(args)
    sources = cells_to_coords(src_cells, transform)
    targets = cells_to_coords(tgt_cells, transform)
    steps = get_neighborhood_steps(args.neighborhood, directed=True)

    passable = cost[cost != SENTINEL]
    print(f"raster {cost.shape} ({cost.size / 1e6:.1f} M cells), "
          f"{(cost == SENTINEL).mean():.0%} forbidden, cost "
          f"{passable.min()}-{passable.max()}")
    print(f"source: {provenance}")
    pair_info = describe_pairs(src_cells, tgt_cells)
    print(f"{args.sources}x{args.targets} matrix = {pair_info['n_pairs']} "
          f"pairs, euclid {pair_info['distance_m_min']:.0f}-"
          f"{pair_info['distance_m_max']:.0f} m "
          f"(median {pair_info['distance_m_median']:.0f} m, "
          f"{pair_info['pairs_over_5km_pct']:.0f}% over 5 km)")

    export_dir = None
    if not args.no_export:
        export_dir = DATA / "tmp" / "matrix_export"
        export_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "small": args.small,
            "size": args.size,
            "n_sources": args.sources,
            "n_targets": args.targets,
            "buffer_m": args.buffer,
            "neighborhood": args.neighborhood,
            "backends": backends,
            "modes": modes,
            "export": export_dir is not None,
            "cython_traversal": _CYTHON_TRAVERSAL,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
        },
        "raster": {
            "provenance": provenance,
            "shape": list(cost.shape),
            "cells": int(cost.size),
            "forbidden_pct": float((cost == SENTINEL).mean() * 100),
            "cost_min": int(passable.min()),
            "cost_max": int(passable.max()),
        },
        "points": {
            "sources": [[int(r), int(c)] for r, c in src_cells],
            "targets": [[int(r), int(c)] for r, c in tgt_cells],
            **pair_info,
        },
        "corridor": corridor_stats(
            sources, targets, args.buffer, 1.0,
            box(*array_bounds(cost.shape[0], cost.shape[1], transform)),
            cost.size),
        "probes": {},
        "runs": [],
        "notes": [
            "solver_s includes path reconstruction for every backend "
            "(cython returns paths from the kernel; raster_gpu walks pred "
            "on the host inside shortest_path). gpu_solve_split separates "
            "them for raster_gpu.",
            "metrics_s is charged per path and contains the full-window "
            "np.sort(np.unique(raster)) scan (plan 1.1).",
            "raster_copy_s is a harness artifact: the buffer mask mutates "
            "the input array through a view (plan 2.1), so each finder "
            "gets a pristine copy. The real drivers re-read the GeoTIFF, "
            "which costs more.",
            "pairwise totals are extrapolated from a measured sample.",
            "per_pair carries length_m/total_cost per backend for the same "
            "(source_id, target_id), so backends can be cross-compared "
            "offline. For raster_fim the authoritative metric is T[target] "
            "(api.last_field_costs), not the re-quantized cell path "
            "reported here.",
            "a meshed matrix is a MIXED-length workload by construction "
            "(the case study meshes each substation to its 5 nearest "
            "neighbours): on a 5000^2 window the pair distances run "
            "~0.1-5.5 km with a ~2.3 km median. The >5 km single-pair "
            "regime is the scale benchmark's subject; raise --size for a "
            "longer-pair matrix.",
        ],
    }

    cor = results["corridor"]
    print(f"corridor: searched {cor['hull_buffer_cells_clipped'] / 1e6:.1f} M "
          f"cells vs pair corridor "
          f"{cor['pair_corridor_cells_median'] / 1e6:.1f} M (median) -> "
          f"{cor['searched_over_pair_corridor_median']:.1f}x "
          f"(max {cor['searched_over_pair_corridor_max']:.1f}x) "
          f"at buffer {cor['buffer_m']:.0f} m")

    print("\n=== probes ===")
    scan_s = probe_unique_scan(cost)
    results["probes"]["unique_scan_s"] = scan_s
    print(f"  np.sort(np.unique(raster)) : {scan_s * 1e3:8.1f} ms  "
          f"(x {pair_info['n_pairs']} paths = "
          f"{scan_s * pair_info['n_pairs']:.1f} s)")
    ctx_probe = probe_cpu_context(cost, steps)
    results["probes"]["cpu_context"] = ctx_probe
    if "context_build_s" in ctx_probe:
        print(f"  RasterContext build        : "
              f"{ctx_probe['context_build_s'] * 1e3:8.1f} ms "
              f"(per public cython call)")

    if any(b in GPU_BACKENDS for b in backends):
        results["meta"]["gpu_before"] = gpu_environment()
        src_idx = src_cells[0][0] * cost.shape[1] + src_cells[0][1]
        gpu_probe = probe_gpu_per_query(cost, steps, src_idx)
        results["probes"]["gpu_per_query"] = gpu_probe
        if "error" not in gpu_probe:
            print(f"  GPU per-query setup        : "
                  f"{gpu_probe['common_setup_s'] * 1e3:8.1f} ms "
                  f"(cast {gpu_probe['host_uint16_cast_s'] * 1e3:.0f} + "
                  f"h2d {gpu_probe['h2d_raster_s'] * 1e3:.0f} + "
                  f"auto-delta {gpu_probe['auto_delta_s'] * 1e3:.0f})")
            print(f"  GPU full-field D2H         : "
                  f"{gpu_probe['d2h_dist_pred_s'] * 1e3:8.1f} ms "
                  f"({gpu_probe['dist_pred_mib']:.0f} MiB)")

    for backend in backends:
        for mode in modes:
            print(f"\n=== {backend} / {mode} "
                  f"({BACKEND_ALGORITHM[backend]}) ===")
            rss0 = rss_bytes()
            extra = {}
            try:
                if mode == "star":
                    totals, per_source, per_pair = run_star(
                        cost, transform, sources, targets, backend,
                        args.buffer, args.neighborhood, export_dir)
                elif mode == "shared":
                    totals, per_source, per_pair, extra = run_shared(
                        cost, transform, sources, targets, backend,
                        args.buffer, args.neighborhood, export_dir)
                elif mode == "pairwise":
                    totals, per_pair = run_pairwise(
                        cost, transform, sources, targets, backend,
                        args.buffer, args.neighborhood, args.pairwise_limit)
                    per_source = []
                else:
                    raise SystemExit(f"unknown mode: {mode}")
            except Exception as exc:            # keep the matrix going
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                results["runs"].append({
                    "backend": backend, "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}"})
                free_gpu_pool()
                continue

            run = {
                "backend": backend,
                "mode": mode,
                "algorithm": BACKEND_ALGORITHM[backend],
                "totals": totals,
                "per_source": per_source,
                "per_pair": per_pair,
                "memory": {
                    "rss_before_bytes": rss0,
                    "rss_after_bytes": rss_bytes(),
                    "peak_rss_bytes": peak_rss_bytes(),
                    "gpu": gpu_pool_stats() if backend in GPU_BACKENDS
                    else None,
                },
                **extra,
            }
            results["runs"].append(run)
            print_totals(f"{backend}/{mode}", totals)
            if totals.get("extrapolated"):
                print(f"  {totals['n_pairs_measured']} pairs measured, "
                      f"{totals['per_pair_s']:.2f} s/pair -> full matrix "
                      f"{totals['extrapolated_full_matrix_s']:.0f} s "
                      f"[EXTRAPOLATED]")
            if "gpu_solve_split" in extra and "error" not in extra[
                    "gpu_solve_split"]:
                split = extra["gpu_solve_split"]
                print(f"  solve {split['run_sssp_s'] * 1e3:.0f} ms | "
                      f"pred walk {split['extract_paths_s'] * 1e3:.0f} ms "
                      f"({split['extract_share_pct']:.0f}% of the query)")
            free_gpu_pool()

        if backend in GPU_BACKENDS:
            results["meta"]["gpu_after"] = gpu_environment()

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"matrix_{stamp}.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()

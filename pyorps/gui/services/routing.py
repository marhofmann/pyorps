"""
PYORPS GUI: route building and control-point routing.

Interactive routes follow a *control-point* model: every route is defined by an
ordered list ``[source, w1, ..., wn, target]`` and its geometry is always the
least-cost path threaded through those points in order (pyorps has no native
waypoint routing — C13 — so the GUI chains ``find_route`` per consecutive pair
and stitches the legs).

Hard rules encoded here:

- C14: the backend/algorithm matrix — invalid combinations are resolved to a
  valid backend, and CPU is the default (GPU strictly opt-in).
- F1: ``search_space_buffer_m`` is never None/0 — :func:`default_search_buffer`
  computes an explicit buffer when the caller leaves it blank.
- F5: pyorps' ``simplify=`` is a no-op, so :func:`simplify_line` implements
  Douglas-Peucker GUI-side; costs are always taken from the un-simplified line.
- F8: the A* heuristic is disabled by default (user preference).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import LineString

from . import cost

# ---------------------------------------------------------------------- matrix
# C14: UI algorithm -> the graph APIs that implement it, in preference order.
CPU_BACKENDS: dict[str, list[str]] = {
    "dijkstra": ["cython"],
    "delta-stepping": ["cython"],
    "delta-stepping-circular": ["cython"],
    "bidirectional_dijkstra": ["networkit", "networkx"],
    "astar": ["networkit", "networkx"],
    "bellman_ford": ["igraph", "rustworkx"],
}
GPU_BACKENDS: dict[str, list[str]] = {
    "dijkstra": ["raster_gpu"],
    "delta-stepping": ["raster_gpu"],
    "delta-stepping-circular": ["raster_gpu"],
}
ALGORITHMS = tuple(CPU_BACKENDS)

#: above this cell count the GUI recommends the GPU backend (banner only —
#: never auto-selected; the user's standing rule is CPU by default).
GPU_RECOMMEND_CELLS = 4_000_000


def resolve_backend(algorithm: str, hardware: str = "cpu") -> tuple[str, str]:
    """Map (algorithm, hardware) to a valid pyorps ``(graph_api, algorithm)``.

    Unknown algorithms fall back to delta-stepping; a GPU request for an
    algorithm without a GPU implementation falls back to its CPU backend.
    """
    if algorithm not in CPU_BACKENDS:
        algorithm = "delta-stepping"
    if hardware == "gpu" and algorithm in GPU_BACKENDS:
        return GPU_BACKENDS[algorithm][0], algorithm
    return CPU_BACKENDS[algorithm][0], algorithm


def valid_algorithms(hardware: str = "cpu") -> list[str]:
    """Algorithms available for the given hardware (drives the UI dropdown)."""
    return list(GPU_BACKENDS if hardware == "gpu" else CPU_BACKENDS)


def recommend_gpu(n_cells: int) -> bool:
    """Size-based recommendation banner (pyorps has no auto-selection)."""
    return n_cells > GPU_RECOMMEND_CELLS


class RoutingCancelled(Exception):
    """Raised inside the routing loops when the user pressed Stop."""


# ------------------------------------------------------------- run estimation
#: outgoing edges per cell for each neighborhood ring
NEIGHBORHOOD_EDGES = {"r0": 4, "r1": 8, "r2": 16, "r3": 32}

#: very rough end-to-end throughput (graph build + search) per backend, in
#: edges/second on a typical desktop CPU/GPU — order-of-magnitude only, used
#: for the pre-run runtime prognosis.
EDGES_PER_SECOND = {
    "cython": 6e6, "networkit": 4e6, "networkx": 1.5e5,
    "igraph": 2e6, "rustworkx": 2e6,
    "raster_gpu": 40e6, "raster_gpu_v3": 40e6, "raster_gpu_v4": 40e6,
}

#: bytes per graph edge / per cell for the memory prognosis (index + weight
#: arrays + distance/predecessor arrays)
_BYTES_PER_EDGE = 12
_BYTES_PER_CELL = 24

#: confirmation thresholds — beyond any of these the run needs an explicit
#: "accept waiting" from the user (task: never process long/many routes
#: without an up-front warning)
CONFIRM_SECONDS = 10.0
CONFIRM_MEMORY_MB = 2000.0
CONFIRM_PAIRS = 15
CONFIRM_TOTAL_CELLS = 30_000_000


def estimate_routing(raster_path: str, *,
                     sources: list[tuple[float, float]],
                     targets: list[tuple[float, float]],
                     waypoints: list[tuple[float, float]] | None = None,
                     algorithm: str = "delta-stepping", hardware: str = "cpu",
                     neighborhood: str = "r2", pairwise: bool = False,
                     search_buffer_m: float | None = None) -> dict:
    """Prognosis for a routing run: cells, runtime, memory, storage.

    Per pair, the search window is the bounding box of its control points
    buffered by the (auto) search buffer, clipped to the raster. Numbers are
    rough, honest estimates for the confirm dialog — not promises.
    """
    import rasterio

    waypoints = list(waypoints or [])
    with rasterio.open(raster_path) as src:
        res = abs(src.transform.a) or 1.0
        raster_w, raster_h = src.width, src.height
        left, bottom, right, top = src.bounds
        import numpy as np
        dtype_size = np.dtype(src.dtypes[0]).itemsize

    buffer_m = search_buffer_m or default_search_buffer(sources, targets)
    pairs = make_pairs(sources, targets, pairwise)

    total_cells = 0
    max_cells = 0
    for s, t in pairs:
        xs = [s[0], t[0], *(w[0] for w in waypoints)]
        ys = [s[1], t[1], *(w[1] for w in waypoints)]
        minx = max(min(xs) - buffer_m, left)
        maxx = min(max(xs) + buffer_m, right)
        miny = max(min(ys) - buffer_m, bottom)
        maxy = min(max(ys) + buffer_m, top)
        w_cells = max(0.0, (maxx - minx)) / res
        h_cells = max(0.0, (maxy - miny)) / res
        cells = min(w_cells * h_cells, float(raster_w) * raster_h)
        total_cells += cells
        max_cells = max(max_cells, cells)

    graph_api, algo = resolve_backend(algorithm, hardware)
    edges_per_cell = NEIGHBORHOOD_EDGES.get(neighborhood, 16)
    n_segments_per_pair = len(waypoints) + 1
    total_edges = total_cells * edges_per_cell
    # graph build dominates; each extra waypoint segment re-searches the graph
    work_edges = total_edges * (1 + 0.3 * (n_segments_per_pair - 1))
    seconds = work_edges / EDGES_PER_SECOND.get(graph_api, 4e6)
    memory_mb = (max_cells * edges_per_cell * _BYTES_PER_EDGE
                 + max_cells * _BYTES_PER_CELL) / 1e6
    storage_mb = max_cells * dtype_size / 1e6      # temp window raster

    return {
        "n_pairs": len(pairs),
        "n_segments": len(pairs) * n_segments_per_pair,
        "buffer_m": float(buffer_m),
        "resolution_m": float(res),
        "neighborhood": neighborhood,
        "graph_api": graph_api,
        "algorithm": algo,
        "max_window_cells": int(max_cells),
        "total_cells": int(total_cells),
        "est_seconds": float(seconds),
        "est_memory_mb": float(memory_mb),
        "est_storage_mb": float(storage_mb),
    }


def needs_confirmation(est: dict) -> bool:
    """True when the prognosis crosses any accept-waiting threshold."""
    return (est["est_seconds"] > CONFIRM_SECONDS
            or est["est_memory_mb"] > CONFIRM_MEMORY_MB
            or est["n_pairs"] > CONFIRM_PAIRS
            or est["total_cells"] > CONFIRM_TOTAL_CELLS)


# -------------------------------------------------------------------- F1 guard
def default_search_buffer(sources: list[tuple[float, float]],
                          targets: list[tuple[float, float]]) -> float:
    """Explicit search-space buffer: ``max(1000, 1.5 x max euclid distance)``.

    Never returns None/0 — leaving ``search_space_buffer_m`` blank makes pyorps
    route the ENTIRE raster (F1), the single biggest performance footgun.
    """
    max_dist = 0.0
    for sx, sy in sources:
        for tx, ty in targets:
            max_dist = max(max_dist, math.hypot(tx - sx, ty - sy))
    return max(1000.0, 1.5 * max_dist)


# ------------------------------------------------------------------- F5 helper
def simplify_line(line: LineString, tolerance_m: float) -> LineString:
    """GUI-side Douglas-Peucker (pyorps' ``simplify=`` kwarg is a no-op, F5).

    Only for display/export — cost metrics must come from the un-simplified
    routed line, since a shortcut can cross forbidden cells the router avoided.
    """
    return line.simplify(tolerance_m, preserve_topology=False)


def route_display_line(layer) -> LineString:
    """A route layer's display geometry: the full line, simplified when the
    layer carries a ``simplify_tol`` (post-hoc, changeable any time — F5:
    the stored/metric geometry always stays the full routed line)."""
    line = layer.gdf.geometry.iloc[0]
    tol = (layer.meta or {}).get("simplify_tol")
    return simplify_line(line, float(tol)) if tol else line


def refresh_route_display(layer) -> LineString:
    """Regenerate the cached WGS84 geojson from the full line + tolerance."""
    from . import geo

    display = route_display_line(layer)
    metrics = (layer.meta or {}).get("metrics") or {}
    feature = geo.linestring_to_wgs84_feature(
        display, layer.crs, properties={"name": layer.name, **metrics})
    layer.geojson = {"type": "FeatureCollection", "features": [feature]}
    return display


# ------------------------------------------------------------ chained routing
def route_through_points(finder, ordered_points: list[tuple[float, float]],
                         algorithm: str = "dijkstra", cancel=None,
                         **algo_kwargs):
    """Least-cost path threaded through ``ordered_points`` (routing CRS).

    ``ordered_points`` is ``[source, *waypoints, target]`` with at least two
    entries; consecutive pairs are routed and stitched (C13). Returns
    ``(stitched_line, RouteCost)`` with the cost from pyorps' own metric.
    ``cancel`` (threading.Event) aborts between legs with RoutingCancelled.
    """
    if len(ordered_points) < 2:
        raise ValueError("Need at least a source and a target control point.")

    coords: list[tuple[float, float]] = []
    for start, end in zip(ordered_points[:-1], ordered_points[1:]):
        if cancel is not None and cancel.is_set():
            raise RoutingCancelled()
        path = finder.find_route(source=tuple(start), target=tuple(end),
                                 algorithm=algorithm, calculate_metrics=False,
                                 **algo_kwargs)
        leg = [tuple(c) for c in path.path_coords]
        if coords and leg and coords[-1] == leg[0]:
            leg = leg[1:]
        coords.extend(leg)

    line = LineString(coords)
    route_cost = cost.evaluate_route_cost(line, finder.raster_handler)
    return line, route_cost


def make_pairs(sources: list, targets: list, pairwise: bool) -> list[tuple]:
    """Source/target pairs. ``pairwise`` zips equal-length lists; else cross."""
    if pairwise:
        from pyorps.core.exceptions import PairwiseError
        if len(sources) != len(targets):
            raise PairwiseError()
        return list(zip(sources, targets))
    return [(s, t) for s in sources for t in targets]


@dataclass
class BuiltRoute:
    """One routed source->target result (routing CRS)."""

    line: LineString
    cost: cost.RouteCost
    source: tuple[float, float]
    target: tuple[float, float]
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def control_points(self) -> list[tuple[float, float]]:
        return [self.source, *self.waypoints, self.target]


def run_routing(raster_source: Any, *, sources: list[tuple[float, float]],
                targets: list[tuple[float, float]],
                waypoints: list[tuple[float, float]] | None = None,
                algorithm: str = "delta-stepping", hardware: str = "cpu",
                neighborhood: str = "r2", pairwise: bool = False,
                search_buffer_m: float | None = None,
                ignore_max_cost: bool = True,
                delta: float = 100, num_threads: int = 0,
                use_astar: bool = False, margin: float | None = None,
                cancel=None, progress=None):
    """Route every source/target pair through the shared, ordered waypoints.

    All coordinates are in the raster's CRS. Returns ``(finder, [BuiltRoute])``
    — the PathFinder is returned so callers can keep it for later editing.
    Pairs that raise ``NoPathFoundError`` are skipped and reported in the
    failed list: the actual return is ``(finder, results, failed_pairs)``.

    ``cancel`` (threading.Event) stops the run cooperatively: it is checked
    before every pair AND between waypoint legs, so pressing Stop aborts
    after the segment currently in flight; routes already completed are kept.
    ``progress(done, total)`` is called before each pair.
    """
    from pyorps import PathFinder
    from pyorps.core.exceptions import NoPathFoundError

    if not sources or not targets:
        raise ValueError("Source and target coordinates must not be None — "
                         "place at least one source and one target.")

    waypoints = list(waypoints or [])
    graph_api, algo = resolve_backend(algorithm, hardware)

    if not search_buffer_m or search_buffer_m <= 0:      # F1: never blank
        search_buffer_m = default_search_buffer(sources, targets)

    # One PathFinder seeded with every point so the search window covers them
    # all (and the waypoints); the chained routing reuses its graph.
    seed_sources = [*sources, *waypoints]
    finder = PathFinder(
        raster_source, source_coords=seed_sources, target_coords=targets,
        graph_api=graph_api, neighborhood_str=neighborhood,
        search_space_buffer_m=search_buffer_m,
        ignore_max_cost=ignore_max_cost)

    # margin is only forwarded when the caller sets one: passing a value here
    # overrides the API's exact termination floor, and the old default of 1.1
    # made every interactive run relax a ~21% larger annulus for nothing.
    algo_kwargs = {"delta": delta, "num_threads": num_threads}
    if margin is not None:
        algo_kwargs["margin"] = margin
    params = {"algorithm": algo, "graph_api": graph_api,
              "neighborhood": neighborhood,
              "search_space_buffer_m": search_buffer_m,
              "ignore_max_cost": ignore_max_cost, "hardware": hardware,
              "delta": delta, "num_threads": num_threads,
              "use_astar": use_astar}

    results: list[BuiltRoute] = []
    failed: list[tuple[tuple, tuple, str]] = []
    pairs = make_pairs(sources, targets, pairwise)
    for i, (src, tgt) in enumerate(pairs):
        if cancel is not None and cancel.is_set():
            break
        if progress is not None:
            progress(i, len(pairs))
        pts = [tuple(src), *map(tuple, waypoints), tuple(tgt)]
        try:
            line, rc = route_through_points(finder, pts, algorithm=algo,
                                            cancel=cancel, **algo_kwargs)
        except RoutingCancelled:
            break                       # keep the routes finished so far
        except NoPathFoundError as exc:
            failed.append((tuple(src), tuple(tgt), str(exc)))
            continue
        results.append(BuiltRoute(line=line, cost=rc, source=tuple(src),
                                  target=tuple(tgt), waypoints=list(map(tuple, waypoints)),
                                  params=dict(params)))
    return finder, results, failed

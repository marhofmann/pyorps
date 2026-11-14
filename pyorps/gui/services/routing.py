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
  computes an explicit buffer when the caller leaves it blank. (Neither None
  nor 0 means "whole raster"; see that function.)
- F5: pyorps' ``simplify=`` is a no-op, so :func:`simplify_line` implements
  Douglas-Peucker GUI-side; costs are always taken from the un-simplified line.
- F8: the A* heuristic is never switched on by the GUI (user preference); the
  library backends' own ``astar`` algorithm stays selectable by name.
- Backends: the user picks the ``graph_api`` (:data:`BACKENDS`); the algorithm
  list is what that backend really accepts. Only the Cython, GPU and FIM
  backends can use a :class:`~pyorps.graph.search_session.SearchSession`
  (:func:`uses_session`); every other backend runs ``PathFinder.find_route``
  directly, because a session would silently run the Cython Dijkstra instead.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from dataclasses import dataclass, field
from functools import lru_cache
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


# -------------------------------------------------------------------- backends
@dataclass(frozen=True)
class Backend:
    """One ``graph_api`` the Backend dropdown offers."""

    name: str                       # the graph_api value
    label: str                      # short text of the dropdown entry
    algorithms: tuple[str, ...]     # what the backend really accepts
    default_algorithm: str
    modules: tuple[str, ...]        # all of these must be importable
    hint: str                       # what to do when they are not
    gpu: bool = False
    about: str = ""                 # one line for the help text


_GRAPH_HINT = 'pip install "pyorps[graph]"'
_GPU_HINT = 'pip install "pyorps[gpu]"'
_LIBRARY_ALGORITHMS = ("dijkstra", "bidirectional_dijkstra", "astar")

#: The algorithm lists follow the ``ALGORITHMS`` tuple of each library API class
#: (cython_api.CythonAPI, networkx_api.NetworkxAPI, ...). The GPU backend only
#: accepts delta-stepping and dijkstra (raster_gpu_api.shortest_path) and the FIM
#: backend fim, eikonal and dijkstra (raster_fim_api._ACCEPTED_ALGORITHMS); a
#: test compares this table with the library. ``cugraph`` is left out on
#: purpose: it raises NotImplementedError.
BACKENDS: dict[str, Backend] = {b.name: b for b in (
    Backend("cython", "cython", ("dijkstra", "delta-stepping",
                                 "delta-stepping-circular"), "delta-stepping",
            ("pyorps.utils._delta_stepping", "pyorps.utils._dijkstra"),
            "the compiled search kernels are missing: reinstall pyorps or run "
            "python setup.py build_ext --inplace",
            about="Compiled kernels on the raster itself (default, fastest "
                  "on the CPU)."),
    Backend("raster_gpu", "raster_gpu (GPU)", ("delta-stepping", "dijkstra"),
            "delta-stepping", ("cupy",), _GPU_HINT, gpu=True,
            about="Delta-stepping on the NVIDIA GPU (needs CuPy and a CUDA "
                  "GPU)."),
    Backend("raster_fim", "raster_fim (GPU)", ("fim", "eikonal", "dijkstra"),
            "fim", ("cupy",), _GPU_HINT, gpu=True,
            about="Continuous eikonal solver on the NVIDIA GPU (needs CuPy "
                  "and a CUDA GPU)."),
    Backend("networkx", "networkx", _LIBRARY_ALGORITHMS, "dijkstra",
            ("networkx",), _GRAPH_HINT,
            about="Graph library; slow but simple. Builds an explicit graph."),
    Backend("networkit", "networkit", _LIBRARY_ALGORITHMS, "dijkstra",
            ("networkit",), _GRAPH_HINT,
            about="Graph library; builds an explicit graph."),
    Backend("igraph", "igraph", ("dijkstra", "bellman_ford", "astar"),
            "dijkstra", ("igraph",), _GRAPH_HINT,
            about="Graph library; builds an explicit graph."),
    Backend("rustworkx", "rustworkx", ("dijkstra", "astar", "bellman_ford"),
            "dijkstra", ("rustworkx",), _GRAPH_HINT,
            about="Graph library; builds an explicit graph."),
)}
DEFAULT_BACKEND = "cython"

#: backends that can run through a retained SearchSession; every other one
#: runs PathFinder.find_route directly (see :func:`uses_session`)
SESSION_BACKENDS = ("cython", "raster_gpu", "raster_fim")


def backend_available(name: str) -> bool:
    """True when every package of backend ``name`` can be found."""
    backend = BACKENDS.get(name)
    if backend is None:
        return False
    for module in backend.modules:
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def backend_options() -> list[dict]:
    """Options of the Backend dropdown: unavailable ones disabled, with the
    install hint in the label (and as the option's tooltip)."""
    options = []
    for backend in BACKENDS.values():
        if backend_available(backend.name):
            options.append({"label": backend.label, "value": backend.name,
                            "title": backend.about})
        else:
            options.append({
                "label": f"{backend.label} (not installed: {backend.hint})",
                "value": backend.name, "disabled": True,
                "title": backend.hint})
    return options


def backend_algorithms(graph_api: str) -> list[str]:
    """The algorithms ``graph_api`` accepts (empty for an unknown backend)."""
    backend = BACKENDS.get(graph_api)
    return list(backend.algorithms) if backend is not None else []


def default_algorithm(graph_api: str) -> str:
    """The algorithm a backend starts with (cython falls back to itself)."""
    return BACKENDS.get(graph_api, BACKENDS[DEFAULT_BACKEND]).default_algorithm


def uses_session(graph_api: str, algorithm: str, margin: float | None = None
                 ) -> bool:
    """True when a route runs through a retained ``SearchSession``.

    A session picks its kernel from the BACKEND KIND only (cython, GPU, FIM):
    on any graph library it would run the Cython Dijkstra, whatever the
    algorithm says (audit 4b.1, confirmed by a run), so those backends run
    ``find_route`` directly. A termination ``margin`` also needs the direct
    path: the session settles the whole field with a fixed margin of 1e30 and
    ignores the keyword.
    """
    if graph_api not in SESSION_BACKENDS:
        return False
    return not (margin is not None and uses_delta_kernel(graph_api, algorithm))


def uses_delta_kernel(graph_api: str, algorithm: str) -> bool:
    """True when the search is a delta-stepping kernel: ``delta`` and
    ``margin`` then change what runs (both GPU algorithm names run it)."""
    return graph_api == "raster_gpu" or (
        graph_api == "cython" and str(algorithm).startswith("delta"))


def uses_threads(graph_api: str, algorithm: str) -> bool:
    """True when ``num_threads`` is read: the CPU delta-stepping kernel only."""
    return graph_api == "cython" and str(algorithm).startswith("delta")


def direct_route_kwargs(graph_api: str, algorithm: str, *, delta: float = 100,
                        num_threads: int = 0,
                        margin: float | None = None) -> dict[str, Any]:
    """Keywords for ``find_route`` of a direct (session-less) route.

    ``find_route`` rejects a keyword its backend's ``shortest_path`` does not
    read (``FIND_ROUTE_OPTIONS``), so only what the backend takes is passed:
    ``delta``, ``num_threads`` and ``margin`` on the cython delta kernel,
    nothing on the graph libraries (the GPU's ``delta``/``margin`` are
    constructor settings: :func:`graph_api_options`).
    """
    if graph_api == "cython" and uses_delta_kernel(graph_api, algorithm):
        kwargs: dict[str, Any] = {"delta": delta, "num_threads": num_threads}
        if margin is not None:
            kwargs["margin"] = margin
        return kwargs
    return {}


def graph_api_options(graph_api: str, algorithm: str, *, delta: float = 100,
                      margin: float | None = None) -> dict[str, Any] | None:
    """``PathFinder(graph_api_options=...)`` for a direct GPU route.

    Only the settings the user gave: ``margin`` is the GPU kernel's
    termination margin and is only passed when it was set.
    """
    if graph_api == "raster_gpu" and margin is not None:
        return {"delta": delta, "margin": margin}
    return None


def resolve_route_backend(algorithm: str, hardware: str = "cpu",
                          graph_api: str | None = None) -> tuple[str, str]:
    """``(graph_api, algorithm)`` of a run.

    An explicit ``graph_api`` (the Backend dropdown) must accept the
    algorithm: a mismatch raises ``ValueError`` instead of quietly running
    something else. Without one, the older (algorithm, hardware) matrix of
    :func:`resolve_backend` applies.
    """
    if graph_api is None:
        return resolve_backend(algorithm, hardware)
    name = str(graph_api).strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {graph_api!r}: choose one of "
                         f"{', '.join(BACKENDS)}.")
    if algorithm not in BACKENDS[name].algorithms:
        raise ValueError(
            f"The {name} backend does not offer the algorithm {algorithm!r}: "
            f"choose one of {', '.join(BACKENDS[name].algorithms)}.")
    return name, algorithm


def hardware_of(graph_api: str) -> str:
    """``"gpu"`` for the GPU backends, else ``"cpu"`` (stored with a route)."""
    backend = BACKENDS.get(graph_api)
    return "gpu" if backend is not None and backend.gpu else "cpu"


#: tooltips of the three delta-stepping settings; the *_OFF_HELP text says why a
#: field is switched off for the chosen backend/algorithm
DELTA_HELP = ("Bucket width of delta-stepping, in cost units. Larger buckets "
              "mean fewer rounds but more repeated work. Default 100.")
DELTA_OFF_HELP = ("Not used: delta is the bucket width of delta-stepping, "
                  "and this backend/algorithm does not run it.")
THREADS_HELP = ("Worker threads of the CPU delta-stepping kernel. 0 = all "
                "cores.")
THREADS_OFF_HELP = ("Not used: only the CPU delta-stepping kernel (cython) "
                    "has worker threads.")
MARGIN_HELP = ("Delta-stepping termination margin. 1 stops as soon as the "
               "target is provably settled (exact); larger values search on "
               "and buy no exactness. Empty = library default. Setting one "
               "runs the route as a single search (no retained field).")
MARGIN_OFF_HELP = ("Not used: the margin belongs to the delta-stepping "
                   "kernels, and this backend/algorithm does not run one.")


# --------------------------------------------------------------- neighbourhood
#: range of the neighbourhood order K the library accepts
NEIGHBORHOOD_MIN, NEIGHBORHOOD_MAX = 0, 127
DEFAULT_NEIGHBORHOOD_K = 2


def neighborhood_string(value: Any) -> str:
    """The ``"rK"`` string for the k field's value (``"r2"`` for a blank one).

    Also takes an ``"r1"`` / ``"1"`` string, as saved routes and older
    callers carry it. Validation of the range is the caller's job
    (:func:`~pyorps.gui.callbacks.routes.numeric_field_problem`).
    """
    if value is None or value == "":
        return f"r{DEFAULT_NEIGHBORHOOD_K}"
    if isinstance(value, str):
        text = value.strip().lstrip("rR")
        return f"r{int(float(text))}"
    return f"r{int(value)}"


@lru_cache(maxsize=64)
def edges_per_cell(neighborhood: str) -> int:
    """Outgoing edges per cell of an ``"rK"`` neighbourhood (4, 8, 16, 32, ...)."""
    from pyorps.utils.neighborhood import get_neighborhood_steps

    try:
        return int(len(get_neighborhood_steps(neighborhood)))
    except Exception:  # pylint: disable=broad-exception-caught  # an estimate must not fail
        return 16


# ------------------------------------------------------------ project settings
#: the routing settings a project keeps, in the order of the controls:
#: key -> value of a blank page (the layout's own defaults)
ROUTING_SETTING_DEFAULTS: dict[str, Any] = {
    "graph_api": DEFAULT_BACKEND,
    "algorithm": BACKENDS[DEFAULT_BACKEND].default_algorithm,
    "neighborhood_k": DEFAULT_NEIGHBORHOOD_K,
    "search_buffer_m": None,
    "full_window": False,
    "ignore_max_cost": True,
    "pairwise": False,
    "simplify": False,
    "simplify_tol": 1.0,
    "delta": 100,
    "num_threads": 0,
    "margin": None,
}
ROUTING_SETTING_KEYS = tuple(ROUTING_SETTING_DEFAULTS)


def _plain_number(value: Any, default: Any) -> Any:
    """``value`` when it is a plain number (or None where None is the default)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return value
    return default


def normalise_routing_settings(raw: Any) -> dict[str, Any]:
    """The routing settings of a project file, as the controls take them.

    A project without them (an older file) gives ``{}``: the page keeps its
    own defaults. Otherwise every key is present: a missing or unusable value
    falls back to the default, an unknown backend to cython and an algorithm
    the backend does not run to that backend's default.
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    out = dict(ROUTING_SETTING_DEFAULTS)
    graph_api = raw.get("graph_api")
    out["graph_api"] = graph_api if graph_api in BACKENDS else DEFAULT_BACKEND
    algorithm = raw.get("algorithm")
    out["algorithm"] = (algorithm if algorithm in backend_algorithms(out["graph_api"])
                        else default_algorithm(out["graph_api"]))
    for key in ("neighborhood_k", "search_buffer_m", "simplify_tol", "delta", "num_threads", "margin"):
        out[key] = _plain_number(raw.get(key), ROUTING_SETTING_DEFAULTS[key])
    for key in ("full_window", "ignore_max_cost", "pairwise", "simplify"):
        value = raw.get(key)
        out[key] = value if isinstance(value, bool) else ROUTING_SETTING_DEFAULTS[key]
    return out


class RoutingCancelled(Exception):
    """Raised inside the routing loops when the user pressed Stop."""


# ------------------------------------------------------------- run estimation
#: very rough end-to-end throughput (graph build + search) per backend, in
#: edges/second on a typical desktop CPU/GPU — order-of-magnitude only, used
#: for the pre-run runtime prognosis.
EDGES_PER_SECOND = {
    "cython": 6e6, "networkit": 4e6, "networkx": 1.5e5,
    "igraph": 2e6, "rustworkx": 2e6,
    "raster_gpu": 40e6, "raster_gpu_v4": 40e6,
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
                     search_buffer_m: float | None = None,
                     graph_api: str | None = None,
                     stack_bands: int = 0) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Prognosis for a routing run: cells, runtime, memory, storage.

    Per pair, the search window is the bounding box of its control points
    buffered by the (auto) search buffer, clipped to the raster. Numbers are
    rough, honest estimates for the confirm dialog — not promises.

    ``stack_bands`` is the number of full-raster float32 bands an objective
    run holds besides the cost raster (``objective.stack_bands``): the metric
    stack covers the whole raster, not the window, so each band adds 4 bytes
    per raster cell to the memory prognosis. The runtime is NOT scaled for the
    slower slope kernels (no measurement exists for it).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import rasterio

    waypoints = list(waypoints or [])
    with rasterio.open(raster_path) as src:
        res = abs(src.transform.a) or 1.0
        raster_w, raster_h = src.width, src.height
        left, bottom, right, top = src.bounds
        import numpy as np
        dtype_size = np.dtype(src.dtypes[0]).itemsize

    buffer_m = search_buffer_m
    if buffer_m is None or buffer_m <= 0:          # blank or 0 = automatic
        buffer_m = default_search_buffer(sources, targets)
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

    graph_api, algo = resolve_route_backend(algorithm, hardware, graph_api)
    n_edges = edges_per_cell(neighborhood)
    n_segments_per_pair = len(waypoints) + 1
    total_edges = total_cells * n_edges
    # graph build dominates; each extra waypoint segment re-searches the graph
    work_edges = total_edges * (1 + 0.3 * (n_segments_per_pair - 1))
    seconds = work_edges / EDGES_PER_SECOND.get(graph_api, 4e6)
    memory_mb = (max_cells * n_edges * _BYTES_PER_EDGE
                 + max_cells * _BYTES_PER_CELL) / 1e6
    if stack_bands > 0:
        memory_mb += float(raster_w) * raster_h * 4 * stack_bands / 1e6
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

    Never returns None/0 (F1). Neither of those means "the whole raster", as
    this note used to claim: ``None`` runs ``estimate_buffer_width``, a
    heuristic sized for ONE source/target pair, and ``0`` is a zero-width
    buffer that raises on a single pair. Both are wrong for a GUI that routes
    whatever the user drew, so the buffer is always computed explicitly here.
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


def has_undo(state, layer_id: str | None) -> bool:
    """True when route ``layer_id`` holds a one-step undo snapshot."""
    return bool(layer_id) and layer_id in state.route_undo


def undo_last_edit(state, layer_id: str | None):
    """Put the route that the last edit of ``layer_id`` replaced back.

    The snapshot (``state.route_undo[layer_id]``) takes the place of the edited
    route: its paint-order position, group, line style and visibility. It keeps
    its own line, control points, params, metrics and name. The edited route is
    removed, the restored one becomes the active route, and no snapshot is left
    (one step only, no redo). Returns the restored layer; raises
    :class:`~pyorps.gui.services.errors.NothingToUndoError` when there is no
    snapshot for ``layer_id``.
    """
    from .errors import NothingToUndoError

    with state.locked():
        edited = state.get(layer_id) if layer_id else None
        snapshot = state.route_undo.get(layer_id) if layer_id else None
        if edited is None or edited.kind != "route" or snapshot is None:
            raise NothingToUndoError(
                "This route has not been edited since it was computed, "
                "opened or last undone.")
        refresh_route_display(snapshot)       # new geojson revision: no stale tile
        snapshot.meta["group"] = edited.meta.get("group")
        snapshot.style = dict(edited.style or {})
        snapshot.visible = edited.visible
        snapshot.z = edited.z
        state.layers[snapshot.id] = snapshot
        state.remove_layer(edited.id)         # also drops its snapshot entry
        state.active_route_id = snapshot.id
        state.dirty = True
    return snapshot


# ------------------------------------------------------------ chained routing
def plain_cost_handler(finder, raster_source: Any = None):
    """The PLAIN cost raster of the window ``finder`` searches, as a ``RasterHandler``.

    With an objective, ``finder.raster_handler`` holds the combined surface (quantised objective units, or
    float32), so pricing a route on it gives objective units, not the euros per metre of the cost raster (and
    float32 cannot be priced at all). A plain finder over the same points, buffer and neighbourhood has the
    same window; no graph is built for it (only the window is read). ``raster_source`` defaults to the file
    the finder was built from. Also usable as ``route_through_points(cost_handler=plain_cost_handler)``.
    """
    from pyorps import PathFinder

    source = raster_source if raster_source is not None else finder.dataset.file_source
    plain = PathFinder(
        source, source_coords=finder.source_coords, target_coords=finder.target_coords,
        graph_api="cython", neighborhood_str=finder.neighborhood_str,
        search_space_buffer_m=finder.search_space_buffer_m,
        ignore_max_cost=finder.ignore_max_cost)
    return plain.raster_handler


#: the metrics key of a route's length along the terrain model (a route without a DEM has none); the map
#: length stays ``total_length_m``
TERRAIN_LENGTH_KEY = "terrain_length_m"


def window_dem(finder):
    """The elevation grid ``finder`` searched over (float32, aligned to its search window), or None.

    None for a finder without a terrain model. This is the array the search itself read (the library aligns the
    DEM to the window once, with bilinear resampling, and keeps it: ``PathFinder._gradient_dem``), so a length
    taken from it agrees with the library's own ``Path.total_length_3d``. A finder that has not built its graph
    yet gets the library's own alignment (cached in the finder, so a later search reads the same grid).
    """
    handler = getattr(finder, "raster_handler", None)
    if handler is None or getattr(finder, "dem_dataset", None) is None \
            or getattr(finder, "dem_raster_handler", None) is None:
        return None
    dem = getattr(finder, "_gradient_dem", None)
    if dem is None:
        data = handler.data[0] if handler.data.ndim == 3 else handler.data
        dem = finder._align_dem_to_window(data)[0]  # pylint: disable=protected-access  # same-package collaborator
    return dem


def terrain_length_m(line: LineString, finder) -> float | None:
    """Length of ``line`` along the terrain model ``finder`` searched over, in metres; None without a DEM.

    Every step of the route (from one vertex to the next: the vertices of a routed line are the nodes of the
    least-cost path) is measured as ``hypot(map length, elevation difference)``, the elevation read from the
    window-aligned DEM (:func:`window_dem`) at the two cells. That is the formula of the library's 3-D length
    (``pyorps.utils.metric_eval``: ``Path.total_length_3d``, which is ``Path.total_length`` over a DEM), so the
    number is the length the search itself measured, not a second reading of the DEM. The map length
    (``RouteCost.total_length``, metres on the plain raster) stays the 2-D length. Also None when a vertex lies
    outside the search window (the length cannot be told).
    """
    import numpy as np

    dem = window_dem(finder)
    if dem is None:
        return None
    handler = finder.raster_handler
    coords = [(float(x), float(y)) for x, y in line.coords]
    if len(coords) < 2:
        return 0.0
    cells = np.atleast_2d(np.asarray(handler.coords_to_indices(coords)))
    rows, cols = cells[:, 0].astype(np.int64), cells[:, 1].astype(np.int64)
    height, width = dem.shape[-2:]
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= height or cols.max() >= width:
        return None
    cell_size = float(abs(handler.window_transform.a))
    map_steps = np.hypot(np.diff(rows).astype(np.float64), np.diff(cols).astype(np.float64)) * cell_size
    rise = np.abs(np.diff(np.asarray(dem)[rows, cols].astype(np.float64)))
    return float(np.hypot(map_steps, rise).sum())


def terrain_length_sentence(lengths) -> str:
    """One sentence for the success notice: the length along the terrain of the routes that have one.

    ``lengths`` is a list of ``(map_length_m, terrain_length_m)`` pairs, one per route; a pair whose terrain
    length is None (a route without a DEM) is left out, and nothing at all gives ``""``. One route reads
    ``"Terrain length 1,010 m (map length 1,000 m)."``, several ``"Terrain length of the 3 routes 980 to 1,010 m
    (the map lengths are 2-D)."``.
    """
    known = [(plain, terrain) for plain, terrain in lengths if terrain is not None]
    if not known:
        return ""
    if len(known) == 1:
        plain, terrain = known[0]
        map_part = "" if plain is None else f" (map length {float(plain):,.0f} m)"
        return f"Terrain length {float(terrain):,.0f} m{map_part}."
    shortest = min(float(terrain) for _plain, terrain in known)
    longest = max(float(terrain) for _plain, terrain in known)
    return (f"Terrain length of the {len(known)} routes {shortest:,.0f} to {longest:,.0f} m "
            "(the map lengths are 2-D).")


def route_through_points(finder, ordered_points: list[tuple[float, float]],
                         algorithm: str = "dijkstra", cancel=None,
                         session=None, cost_handler=None, **algo_kwargs):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Least-cost path threaded through ``ordered_points`` (routing CRS).

    ``ordered_points`` is ``[source, *waypoints, target]`` with at least two
    entries; consecutive pairs are routed and stitched (C13). Returns
    ``(stitched_line, RouteCost)`` with the cost from pyorps' own metric. ``RouteCost.total_length`` is the map
    length (2-D); a route searched over a DEM also carries ``RouteCost.terrain_length_m``, the length along the
    terrain (:func:`terrain_length_m`), and every other route has None there.
    ``cancel`` (threading.Event) aborts between legs with RoutingCancelled.
    When ``session`` is a :class:`~pyorps.graph.search_session.SearchSession`,
    the first call uses ``session.route`` and later edits should call
    ``session.update`` instead of this helper.

    ``cost_handler`` says what the route is priced on. None (the default):
    the search raster of ``finder``. With an objective that is wrong, so the
    caller passes the plain cost raster: a ``RasterHandler``, or a callable
    that takes the finder that really searched (a session rebuilds its own
    finder when a point leaves the window) and returns one, such as
    :func:`plain_cost_handler`.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if len(ordered_points) < 2:
        raise ValueError("Need at least a source and a target control point.")

    def priced_on(searched_finder):
        if cost_handler is None:
            return searched_finder.raster_handler
        return cost_handler(searched_finder) if callable(cost_handler) else cost_handler

    if session is not None:
        if cancel is not None and cancel.is_set():
            raise RoutingCancelled()
        path = (session.update(ordered_points) if session.has_route
                else session.route(ordered_points))
        line = path.path_geometry
        route_cost = cost.evaluate_route_cost(
            line, priced_on(session.finder if cost_handler is not None else finder))
        # a session that rebuilt its window searched on its own finder, so the terrain is read from that one
        route_cost.terrain_length_m = terrain_length_m(line, session.finder)
        return line, route_cost

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
    route_cost = cost.evaluate_route_cost(line, priced_on(finder))
    route_cost.terrain_length_m = terrain_length_m(line, finder)
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
    session: Any = None

    @property
    def control_points(self) -> list[tuple[float, float]]:
        return [self.source, *self.waypoints, self.target]


def whole_raster_buffer(raster_source: Any) -> float:
    """Search-space buffer that makes the window the WHOLE raster.

    The library's own helper (``search_session.full_window_buffer_m``): the
    raster's diagonal, which no buffer needs to exceed.
    """
    from pyorps.graph.search_session import full_window_buffer_m

    return float(full_window_buffer_m(raster_source))


def points_in_finder_window(finder, points: list[tuple[float, float]]) -> bool:
    """True when every point lies inside the search window ``finder`` holds.

    A direct (session-less) route cannot leave the window of its finder; a
    ``SearchSession`` rebuilds the window itself, a direct route needs a new
    finder instead.
    """
    handler = getattr(finder, "raster_handler", None)
    if handler is None:
        return False
    rows, cols = handler.data.shape[-2:]
    for row, col in handler.coords_to_indices([tuple(p) for p in points]):
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return False
    return True


def run_routing(raster_source: Any, *, sources: list[tuple[float, float]],
                targets: list[tuple[float, float]],
                waypoints: list[tuple[float, float]] | None = None,
                algorithm: str = "delta-stepping", hardware: str = "cpu",
                neighborhood: str = "r2", pairwise: bool = False,
                search_buffer_m: float | None = None,
                ignore_max_cost: bool = True,
                delta: float = 100, num_threads: int = 0,
                margin: float | None = None,
                graph_api: str | None = None, full_window: bool = False,
                cancel=None, progress=None,
                objective_kwargs: dict[str, Any] | None = None,
                objective_spec: dict[str, Any] | None = None,
                objective_key: str | None = None,
                force_direct: bool = False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Route every source/target pair through the shared, ordered waypoints.

    All coordinates are in the raster's CRS. Returns ``(finder, [BuiltRoute])``
    — the PathFinder is returned so callers can keep it for later editing.
    Pairs that raise ``NoPathFoundError`` are skipped and reported in the
    failed list: the actual return is ``(finder, results, failed_pairs)``.

    ``graph_api`` is the backend the user picked (``algorithm`` must be one it
    accepts); without it the older (``algorithm``, ``hardware``) matrix picks
    one. Cython, GPU and FIM routes run through a retained ``SearchSession``;
    every other backend, and any route with a termination ``margin`` on a
    delta-stepping kernel, runs ``PathFinder.find_route`` directly so exactly
    the chosen backend, algorithm and margin run (:func:`uses_session`).
    ``full_window`` makes the whole raster the search window.

    ``cancel`` (threading.Event) stops the run cooperatively: it is checked
    before every pair AND between waypoint legs, so pressing Stop aborts
    after the segment currently in flight; routes already completed are kept.
    ``progress(done, total)`` is called before each pair.

    ``objective_kwargs`` are the extra ``PathFinder`` arguments of an
    objective run (``objective``, ``weight_precision`` and, when set, ``dem``,
    ``gradient_options``, ``metric_layers``: ``objective.run_kwargs``).
    Without them nothing below changes. With them: ``force_direct`` runs
    ``find_route`` instead of a session, every route is priced on the PLAIN
    cost raster of the same window (euros per metre, whatever the objective
    says), and ``objective_spec`` / ``objective_key`` are stored in each
    route's ``params`` (the finder carries ``gui_objective_key``) so that an
    edit can rebuild the same objective.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps import PathFinder
    from pyorps.core.exceptions import NoPathFoundError

    if not sources or not targets:
        raise ValueError("Source and target coordinates must not be None — "
                         "place at least one source and one target.")

    waypoints = list(waypoints or [])
    graph_api, algo = resolve_route_backend(algorithm, hardware, graph_api)

    if full_window:
        search_buffer_m = whole_raster_buffer(raster_source)
    elif not search_buffer_m or search_buffer_m <= 0:      # F1: never blank
        search_buffer_m = default_search_buffer(sources, targets)

    use_session = uses_session(graph_api, algo, margin) and not force_direct
    finder_options = (None if use_session else
                      graph_api_options(graph_api, algo, delta=delta,
                                        margin=margin))
    # One PathFinder seeded with every point so the search window covers them
    # all (and the waypoints); the chained routing reuses its graph.
    seed_sources = [*sources, *waypoints]
    finder_kwargs = {"graph_api_options": finder_options} if finder_options         else {}
    finder = PathFinder(
        raster_source, source_coords=seed_sources, target_coords=targets,
        graph_api=graph_api, neighborhood_str=neighborhood,
        search_space_buffer_m=search_buffer_m,
        ignore_max_cost=ignore_max_cost, **finder_kwargs,
        **(objective_kwargs or {}))
    price_kwargs: dict[str, Any] = {}
    if objective_kwargs:
        # an objective finder searches a combined surface: price on the plain cost raster of its window
        finder.gui_objective_key = objective_key
        price_kwargs["cost_handler"] = plain_cost_handler(finder, raster_source)

    # margin is only forwarded when the caller sets one: passing a value here
    # overrides the API's exact termination floor. A session ignores it, so a
    # route that has one runs find_route directly (see uses_session).
    algo_kwargs = {"delta": delta, "num_threads": num_threads}
    direct_kwargs = direct_route_kwargs(graph_api, algo, delta=delta,
                                        num_threads=num_threads,
                                        margin=margin)
    params = {"algorithm": algo, "graph_api": graph_api,
              "neighborhood": neighborhood,
              "search_space_buffer_m": search_buffer_m,
              "full_window": bool(full_window),
              "ignore_max_cost": ignore_max_cost,
              "hardware": hardware_of(graph_api),
              "delta": delta, "num_threads": num_threads, "margin": margin}
    if objective_kwargs:
        params["objective_spec"] = copy.deepcopy(objective_spec)
        params["objective_key"] = objective_key

    results: list[BuiltRoute] = []
    failed: list[tuple[tuple, tuple, str]] = []
    pairs = make_pairs(sources, targets, pairwise)
    for i, (src, tgt) in enumerate(pairs):
        if cancel is not None and cancel.is_set():
            break
        if progress is not None:
            progress(i, len(pairs))
        pts = [tuple(src), *map(tuple, waypoints), tuple(tgt)]
        session = None
        try:
            if use_session:
                session = finder.search_session(algorithm=algo, **algo_kwargs)
                line, rc = route_through_points(
                    finder, pts, algorithm=algo, cancel=cancel,
                    session=session, **price_kwargs, **algo_kwargs)
            else:
                line, rc = route_through_points(
                    finder, pts, algorithm=algo, cancel=cancel,
                    **price_kwargs, **direct_kwargs)
        except RoutingCancelled:
            if session is not None:
                session.close()
            break                       # keep the routes finished so far
        except NoPathFoundError as exc:
            if session is not None:
                session.close()
            failed.append((tuple(src), tuple(tgt), str(exc)))
            continue
        results.append(BuiltRoute(line=line, cost=rc, source=tuple(src),
                                  target=tuple(tgt), waypoints=list(map(tuple, waypoints)),
                                  params=dict(params), session=session))
    return finder, results, failed

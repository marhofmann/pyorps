"""
PYORPS GUI: the backend / algorithm matrix of the routing service (C14).

Split out of :mod:`pyorps.gui.services.routing`, which re-exports every name.
The user picks the ``graph_api`` (:data:`BACKENDS`); the algorithm list is what
that backend really accepts. Only the Cython, GPU and FIM backends can use a
:class:`~pyorps.graph.search_session.SearchSession` (:func:`uses_session`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
            "the compiled search kernels are missing: reinstall pyorps or run " +
            "python setup.py build_ext --inplace",
            about="Compiled kernels on the raster itself (default, fastest " +
                  "on the CPU)."),
    Backend("raster_gpu", "raster_gpu (GPU)", ("delta-stepping", "dijkstra"),
            "delta-stepping", ("cupy",), _GPU_HINT, gpu=True,
            about="Delta-stepping on the NVIDIA GPU (needs CuPy and a CUDA " +
                  "GPU)."),
    Backend("raster_fim", "raster_fim (GPU)", ("fim", "eikonal", "dijkstra"),
            "fim", ("cupy",), _GPU_HINT, gpu=True,
            about="Continuous eikonal solver on the NVIDIA GPU (needs CuPy " +
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
DELTA_HELP = ("Bucket width of delta-stepping, in cost units. Larger buckets " +
              "mean fewer rounds but more repeated work. Default 100.")
DELTA_OFF_HELP = ("Not used: delta is the bucket width of delta-stepping, " +
                  "and this backend/algorithm does not run it.")
THREADS_HELP = ("Worker threads of the CPU delta-stepping kernel. 0 = all " +
                "cores.")
THREADS_OFF_HELP = ("Not used: only the CPU delta-stepping kernel (cython) " +
                    "has worker threads.")
MARGIN_HELP = ("Delta-stepping termination margin. 1 stops as soon as the " +
               "target is provably settled (exact); larger values search on " +
               "and buy no exactness. Empty = library default. Setting one " +
               "runs the route as a single search (no retained field).")
MARGIN_OFF_HELP = ("Not used: the margin belongs to the delta-stepping " +
                   "kernels, and this backend/algorithm does not run one.")


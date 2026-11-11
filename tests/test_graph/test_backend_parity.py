"""Every exact path-finding backend must return a path of the same cost.

"Same result" means *equal cost*, not equal cells. Two paths of equal cost are equally good answers, and the
backends legitimately pick different ones: a uniform raster has many equal-cost staircases, and each library has its
own heap and tie rule. A test on cell sequences would fail on correct code.

The cost is also not ``Path.total_cost`` (a recomputation that differs between equal-objective paths) and not
the number a backend reports. Every returned path is re-priced in float64 by the independent edge model of
``tests/exactness_referee.py`` and compared with the re-priced path of one reference: the Cython Dijkstra, which is
itself checked against a plain-Python Dijkstra on the same edge model
(``test_reference_matches_plain_python_dijkstra``).
Re-pricing in float64 keeps float32 accumulation on the GPU and in delta-stepping from hiding a real difference.

Tolerances, both strict, both relative to the reference cost, and never loosened to the 1 % of older tests (the
bug class this file guards, a lost relaxation in multi-threaded delta-stepping, costs 0.08 to 2.8 %):

* float64 group (the Cython Dijkstra, networkit, networkx, igraph, rustworkx): ``1e-7``. They minimise the float64
  step factor while the referee prices the float32 factor that the kernels hold; the two differ by at most 3e-8 per
  step, so the re-priced excess stays below 6e-8 (``test_step_cost_parity.py``).
* float32 group (Cython delta-stepping at 1 and 4 threads, ``raster_gpu``): ``optimality_tolerance`` of the referee,
  ``(H_ref + 8) * 2**-53 + (H_cand + 8) * 2**-24``, the forward error of summing H non-negative float32 terms.

Cases: ``cython`` (dijkstra, delta-stepping and delta-stepping-circular, the delta variants at 1 and 4 threads),
``networkit`` and ``networkx`` (dijkstra, bidirectional_dijkstra, astar), ``igraph`` and ``rustworkx`` (dijkstra,
bellman_ford, astar) and ``raster_gpu``, on the neighbourhoods r0 to r3 (4, 8, 16 and 32 steps). Fixtures are at
most 64 x 64 cells, so the A* heuristic of networkit and rustworkx, which takes its minimum cost only inside a window
of 200 cells around source and target, is admissible; ``test_astar_window_*`` documents what happens outside it.
Besides one source-target pair, the file covers multi-target and pairwise queries, unreachable targets (both a target
cut off by a wall and a target that has no edge at all), and repeated 4-thread delta-stepping runs.

Left out on purpose: ``raster_fim`` solves a continuous eikonal equation and ignores the neighbourhood, so its
cost differs from the graph optimum by design (the referee judges it against an envelope, never for equality);
``cugraph`` is not implemented and raises.

Skips: an optional backend is skipped when its package is not installed (macOS CI has no networkit), the GPU cases
when CuPy or a CUDA device is missing. The module imports and collects with none of them. The Cython extensions
are required: a missing build fails loudly at import.
"""

from __future__ import annotations

import contextlib
import functools
import heapq
import importlib.util
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pytest
from exactness_referee import (
    APPROXIMATE_BACKENDS,
    EdgeModel,
    optimality_tolerance,
    reprice_path,
    self_consistency_tolerance,
)

from pyorps.core.exceptions import NoPathFoundError
from pyorps.graph.api.cython_api import CythonAPI
from pyorps.graph.path_finder import PathFinder, get_graph_api_class
from pyorps.utils.neighborhood import get_neighborhood_steps

# igraph warns whenever a target cannot be reached; that is exactly what the unreachable-target tests ask for.
pytestmark = pytest.mark.filterwarnings("ignore:Couldn't reach:RuntimeWarning")

IMPASSABLE = 65535
NEIGHBOURHOODS = [0, 1, 2, 3]            # k of get_neighborhood_steps: 4, 8, 16 and 32 neighbours
FLOAT64, FLOAT32 = "float64", "float32"
#: Relative excess allowed for the float64 group. Derivation in the module docstring.
FLOAT64_TOLERANCE = 1e-7
#: Backends whose neighbourhood is the full (directed) step set; the others get the half set (see PathFinder).
FULL_STEP_BACKENDS = frozenset({"cython", "raster_gpu"})


# ------------------------------------------------------------------------------------------------ availability

def _installed(package: str) -> bool:
    """Whether an optional package can be imported (False when it is absent or hidden)."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _gpu_usable() -> bool:
    """CuPy is importable and sees a CUDA device (the guard of tests/test_graph/test_phase1b_gpu_dispatch.py)."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # any failure to reach a device means "no GPU"
        return False


GPU = _gpu_usable()


# ------------------------------------------------------------------------------------------------ cases

@dataclass(frozen=True, eq=False)
class Case:
    """One (graph_api, algorithm) combination under test."""

    id: str
    graph_api: str
    algorithm: str
    group: str                       # FLOAT64 or FLOAT32: which tolerance applies
    kwargs: dict = field(default_factory=dict)
    package: str | None = None       # optional dependency that must be importable
    gpu: bool = False
    pairwise: bool = True            # False: the API ignores a caller's thread count there (cython_api.py)


CASES = [
    Case("cy-dijkstra", "cython", "dijkstra", FLOAT64),
    Case("cy-delta-1", "cython", "delta-stepping", FLOAT32, {"num_threads": 1}, pairwise=False),
    Case("cy-delta-4", "cython", "delta-stepping", FLOAT32, {"num_threads": 4}, pairwise=False),
    Case("cy-delta-circular-1", "cython", "delta-stepping-circular", FLOAT32, {"num_threads": 1}, pairwise=False),
    Case("cy-delta-circular-4", "cython", "delta-stepping-circular", FLOAT32, {"num_threads": 4}, pairwise=False),
    Case("nk-dijkstra", "networkit", "dijkstra", FLOAT64, package="networkit"),
    Case("nk-bidirectional", "networkit", "bidirectional_dijkstra", FLOAT64, package="networkit"),
    Case("nk-astar", "networkit", "astar", FLOAT64, package="networkit"),
    Case("nx-dijkstra", "networkx", "dijkstra", FLOAT64, package="networkx"),
    Case("nx-bidirectional", "networkx", "bidirectional_dijkstra", FLOAT64, package="networkx"),
    Case("nx-astar", "networkx", "astar", FLOAT64, package="networkx"),
    Case("ig-dijkstra", "igraph", "dijkstra", FLOAT64, package="igraph"),
    Case("ig-bellman-ford", "igraph", "bellman_ford", FLOAT64, package="igraph"),
    Case("ig-astar", "igraph", "astar", FLOAT64, package="igraph"),
    Case("rx-dijkstra", "rustworkx", "dijkstra", FLOAT64, package="rustworkx"),
    Case("rx-bellman-ford", "rustworkx", "bellman_ford", FLOAT64, package="rustworkx"),
    Case("rx-astar", "rustworkx", "astar", FLOAT64, package="rustworkx"),
    Case("gpu-delta-stepping", "raster_gpu", "delta-stepping", FLOAT32, gpu=True),
    Case("gpu-dijkstra-alias", "raster_gpu", "dijkstra", FLOAT32, gpu=True),
]


def _param(case: Case):
    """A pytest parameter that skips itself when the backend's package or the GPU is missing."""
    marks = []
    if case.package is not None:
        marks.append(pytest.mark.skipif(not _installed(case.package), reason=f"{case.package} is not installed"))
    if case.gpu:
        marks.append(pytest.mark.skipif(not GPU, reason="no usable CUDA GPU (CuPy missing or no device)"))
    return pytest.param(case, id=case.id, marks=marks)


ALL_CASES = [_param(c) for c in CASES]
PAIRWISE_CASES = [_param(c) for c in CASES if c.pairwise]


# ------------------------------------------------------------------------------------------------ known defects
# The no-path contract of a graph API is NoPathFoundError (or ``[]`` inside a multi-target answer). A backend that
# raises a library exception instead gets an entry ``(graph_api, algorithm): (exception, reason)`` here; each entry
# is xfail(strict=True): when it is fixed the test passes, strict xfail turns that into a failure, and the entry
# must be deleted. Both tables are empty: rustworkx (IndexError on the path mapping of an unreachable target) and
# networkx (NodeNotFound for a target without edges) were fixed when this module first ran.

#: Target exists in the graph but no path leads to it ("wall_closed_14", "enclosed_room_24").
KNOWN_UNREACHABLE_DEFECTS: dict = {}
#: Target has no edge at all ("isolated_target_24"), so the graph library has no such node.
KNOWN_ISOLATED_DEFECTS: dict = {}
UNREACHABLE_FIXTURES = frozenset({"wall_closed_14", "enclosed_room_24"})


def _expect_known_defect(request, table: dict, case: Case) -> None:
    """Mark the running test as an expected failure when ``table`` lists this backend."""
    entry = table.get((case.graph_api, case.algorithm))
    if entry is None:
        return
    raises, reason = entry
    if isinstance(raises, tuple):
        module, name = raises
        raises = getattr(importlib.import_module(module), name)
    request.applymarker(pytest.mark.xfail(strict=True, raises=raises, reason=reason))


# ------------------------------------------------------------------------------------------------ fixtures
# All deterministic, uint16, cell cost >= 1 except barriers (65535). Source = first cell, target = last cell.
# No 0-cost cells: they are a separate delta-stepping hazard and would zero the A* heuristic.

def _ties_uniform_12() -> np.ndarray:
    """Uniform cost: C(22, 11) equal-cost staircases on r0, so every backend meets many exact ties."""
    return np.full((12, 12), 10, dtype=np.uint16)


def _wall_gap_14() -> np.ndarray:
    """A full-height barrier column with a one-cell opening: every route, and every corner cut, must use the gap."""
    raster = np.full((14, 14), 10, dtype=np.uint16)
    raster[:, 7] = IMPASSABLE
    raster[6, 7] = 10
    return raster


def _wall_closed_14() -> np.ndarray:
    """The same wall without an opening: the target is unreachable, every backend must say so."""
    raster = np.full((14, 14), 10, dtype=np.uint16)
    raster[:, 7] = IMPASSABLE
    return raster


def _diagonal_wall_16() -> np.ndarray:
    """A one-cell-thick anti-diagonal barrier with one gap: a diagonal step would cut the corner between two walls."""
    raster = np.full((16, 16), 10, dtype=np.uint16)
    index = np.arange(16)
    raster[index, 15 - index] = IMPASSABLE
    raster[8, 7] = 10
    return raster


def _random_24() -> np.ndarray:
    return np.random.default_rng(7).integers(1, 200, (24, 24)).astype(np.uint16)


def _texture_ties_24() -> np.ndarray:
    """Costs 1 to 4 (many exactly tied alternatives) with 15 % barriers; both corners are kept passable."""
    rng = np.random.default_rng(7)
    raster = rng.integers(1, 5, (24, 24)).astype(np.uint16)
    raster[rng.random((24, 24)) < 0.15] = IMPASSABLE
    raster[:2, :2] = np.minimum(raster[:2, :2], 4)
    raster[-2:, -2:] = np.minimum(raster[-2:, -2:], 4)
    return raster


def _random_64() -> np.ndarray:
    """The largest fixture (4096 cells): enough buckets and frontier for several OpenMP threads to share work."""
    return np.random.default_rng(12345).integers(1, 200, (64, 64)).astype(np.uint16)


def _enclosed_room_24() -> np.ndarray:
    """A random raster with a closed two-cell room (18, 18)-(18, 19) inside a barrier block wider than any step."""
    raster = np.random.default_rng(11).integers(1, 200, (24, 24)).astype(np.uint16)
    raster[14:23, 14:23] = IMPASSABLE
    raster[18, 18:20] = 50
    return raster


def _isolated_target_24() -> np.ndarray:
    """The same block with ONE free cell: (18, 18) has no edge at all, so a graph library has no such node."""
    raster = _enclosed_room_24()
    raster[18, 19] = IMPASSABLE
    return raster


FIXTURES: dict[str, Callable[[], np.ndarray]] = {
    "ties_uniform_12": _ties_uniform_12,
    "wall_gap_14": _wall_gap_14,
    "wall_closed_14": _wall_closed_14,
    "diagonal_wall_16": _diagonal_wall_16,
    "random_24": _random_24,
    "texture_ties_24": _texture_ties_24,
    "random_64": _random_64,
    "enclosed_room_24": _enclosed_room_24,
    "isolated_target_24": _isolated_target_24,
}
#: Fixtures for the single-pair matrix (the others have their own tests).
PAIR_FIXTURES = ["ties_uniform_12", "wall_gap_14", "wall_closed_14", "diagonal_wall_16", "random_24",
                 "texture_ties_24"]

#: Target cells (row, col) for the multi-target test (source: cell 0); the last target of "enclosed_room_24" has
#: no path.
MULTI_TARGETS = {
    "random_24": [(23, 23), (0, 23), (23, 0), (12, 20)],
    "enclosed_room_24": [(23, 23), (0, 23), (18, 18)],
}
#: (source, target) cells (row, col) for the pairwise test.
PAIRS = {
    "random_24": [((0, 0), (23, 23)), ((23, 0), (0, 23)), ((5, 3), (20, 17)), ((12, 12), (3, 20))],
    "wall_gap_14": [((0, 0), (13, 13)), ((13, 0), (0, 13)), ((3, 2), (10, 12))],
}


def _flat(cell: tuple[int, int], raster: np.ndarray) -> int:
    return int(cell[0]) * int(raster.shape[1]) + int(cell[1])


def _first_and_last(raster: np.ndarray) -> tuple[int, int]:
    return 0, int(raster.size) - 1


# ------------------------------------------------------------------------------------------------ running a backend

def _steps(graph_api: str, k: int) -> np.ndarray:
    return get_neighborhood_steps(k, directed=graph_api in FULL_STEP_BACKENDS)


@contextlib.contextmanager
def _api(graph_api: str, raster: np.ndarray, k: int) -> Iterator:
    """The graph API as PathFinder builds it (same class, same step set); closes GPU memory afterwards."""
    api = get_graph_api_class(graph_api)(raster, _steps(graph_api, k), ignore_max=True)
    try:
        yield api
    finally:
        close = getattr(api, "close", None)
        if close is not None:
            close()


def _route(api, source, target, algorithm: str, **kwargs):
    """The path(s) of one query as plain int lists; ``[]`` stands for "no path" (backends differ in how they say it)."""
    try:
        result = api.shortest_path(source, target, algorithm=algorithm, **kwargs)
    except NoPathFoundError:
        return []
    if np.ndim(source) == 0 and np.ndim(target) == 0:
        return [int(cell) for cell in result]
    return [[int(cell) for cell in path] for path in result]


# ------------------------------------------------------------------------------------------------ reference

@dataclass(frozen=True)
class Reference:
    """The Cython Dijkstra answer for one (fixture, k, source, target), re-priced in float64."""

    path: tuple[int, ...]
    cost: float
    hops: int

    @property
    def reachable(self) -> bool:
        return len(self.path) > 0


@functools.cache
def _reference(fixture: str, k: int, source: int, target: int) -> Reference:
    raster = FIXTURES[fixture]()
    with _api("cython", raster, k) as api:
        path = _route(api, source, target, "dijkstra")
    if not path:
        return Reference((), math.inf, 0)
    priced = reprice_path(path, raster, EdgeModel(_steps("cython", k)))
    assert priced.ok, f"the reference path is illegal on {fixture} r{k}: {priced.violations}"
    assert path[0] == source and path[-1] == target
    return Reference(tuple(path), priced.cost, priced.hops)


def _assert_same_cost(case: Case, fixture: str, k: int, raster: np.ndarray, source: int, target: int, path) -> None:
    """The candidate's path is legal and costs the same as the reference (or both report "no path")."""
    where = f"{case.id} on {fixture} r{k} ({source} -> {target})"
    reference = _reference(fixture, k, source, target)
    if not reference.reachable:
        assert path == [], f"{where}: the reference finds no path but the backend returned {path}"
        return
    assert path, f"{where}: the backend found no path, the reference did (cost {reference.cost!r})"
    assert path[0] == source and path[-1] == target, f"{where}: path runs {path[0]} -> {path[-1]}"
    priced = reprice_path(path, raster, EdgeModel(_steps("cython", k)))
    assert priced.ok, f"{where}: illegal path {priced.violations}"
    excess = (priced.cost - reference.cost) / reference.cost
    if case.group == FLOAT64:
        tolerance = FLOAT64_TOLERANCE
    else:
        tolerance = optimality_tolerance(reference.hops, priced.hops)
    # Two-sided: a cheaper path than the reference means the reference is not optimal.
    assert abs(excess) <= tolerance, (
        f"{where}: re-priced cost {priced.cost!r} vs reference {reference.cost!r}, "
        f"excess {excess:+.3e} (tolerance {tolerance:.1e})")


# ------------------------------------------------------------------------------------------------ the parity matrix

@pytest.mark.parametrize("k", NEIGHBOURHOODS, ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", PAIR_FIXTURES)
@pytest.mark.parametrize("case", ALL_CASES)
def test_same_cost_as_reference(request, case, fixture, k):
    """One source-target pair: the backend's path costs what the reference path costs."""
    if fixture in UNREACHABLE_FIXTURES:
        _expect_known_defect(request, KNOWN_UNREACHABLE_DEFECTS, case)
    raster = FIXTURES[fixture]()
    source, target = _first_and_last(raster)
    with _api(case.graph_api, raster, k) as api:
        path = _route(api, source, target, case.algorithm, **case.kwargs)
    _assert_same_cost(case, fixture, k, raster, source, target, path)


@pytest.mark.parametrize("k", [1, 2, 3], ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", sorted(MULTI_TARGETS))
@pytest.mark.parametrize("case", ALL_CASES)
def test_single_source_multi_target_same_cost(request, case, fixture, k):
    """One source, several targets in one call: each path costs what its own reference costs."""
    if fixture in UNREACHABLE_FIXTURES:
        _expect_known_defect(request, KNOWN_UNREACHABLE_DEFECTS, case)
    raster = FIXTURES[fixture]()
    source = 0
    targets = [_flat(cell, raster) for cell in MULTI_TARGETS[fixture]]
    with _api(case.graph_api, raster, k) as api:
        paths = _route(api, source, targets, case.algorithm, **case.kwargs)
    assert len(paths) == len(targets)
    for target, path in zip(targets, paths):
        _assert_same_cost(case, fixture, k, raster, source, target, path)


@pytest.mark.parametrize("k", [1, 2, 3], ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", sorted(PAIRS))
@pytest.mark.parametrize("case", PAIRWISE_CASES)
def test_pairwise_same_cost(case, fixture, k):
    """Several (source, target) pairs in one call with ``pairwise=True``: each pair costs what its reference costs.

    The Cython delta-stepping cases are not run here: its pairwise branch ignores the caller's thread count
    (``CythonAPI._pairwise_paths``), so the 4-thread cap of the test machine could not be honoured.
    """
    raster = FIXTURES[fixture]()
    pairs = [(_flat(s, raster), _flat(t, raster)) for s, t in PAIRS[fixture]]
    with _api(case.graph_api, raster, k) as api:
        paths = _route(api, [s for s, _ in pairs], [t for _, t in pairs], case.algorithm, pairwise=True,
                       **case.kwargs)
    assert len(paths) == len(pairs)
    for (source, target), path in zip(pairs, paths):
        _assert_same_cost(case, fixture, k, raster, source, target, path)


@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"r{k}")
@pytest.mark.parametrize("case", ALL_CASES)
def test_isolated_target_reports_no_path(request, case, k):
    """A target whose neighbours are all barriers (no edge, so no node in a graph library) has no path."""
    _expect_known_defect(request, KNOWN_ISOLATED_DEFECTS, case)
    raster = FIXTURES["isolated_target_24"]()
    target = 18 * raster.shape[1] + 18
    assert not _reference("isolated_target_24", k, 0, target).reachable
    with _api(case.graph_api, raster, k) as api:
        path = _route(api, 0, target, case.algorithm, **case.kwargs)
    _assert_same_cost(case, "isolated_target_24", k, raster, 0, target, path)


# ------------------------------------------------------------------------------------------ is the reference right?

def _plain_python_dijkstra(raster: np.ndarray, steps: np.ndarray, source: int, target: int) -> float:
    """Least cost on the referee's edge model, by a heap search that shares no code with any backend."""
    model = EdgeModel(steps)
    rows, cols = raster.shape
    table = []
    for dr, dc in steps:
        _, intermediates, factor = model.edge(int(dr), int(dc))
        table.append((int(dr), int(dc), intermediates, factor))
    best = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        dist, node = heapq.heappop(heap)
        if node == target:
            return dist
        if dist > best.get(node, math.inf):
            continue
        row, col = divmod(node, cols)
        for dr, dc, intermediates, factor in table:
            cells = [(row + dr, col + dc)] + [(row + a, col + b) for a, b in intermediates]
            if any(not (0 <= r < rows and 0 <= c < cols) or raster[r, c] == IMPASSABLE for r, c in cells):
                continue
            weight = (float(raster[row, col]) + sum(float(raster[r, c]) for r, c in cells)) * factor
            neighbour = cells[0][0] * cols + cells[0][1]
            if dist + weight < best.get(neighbour, math.inf):
                best[neighbour] = dist + weight
                heapq.heappush(heap, (dist + weight, neighbour))
    return math.inf


@pytest.mark.parametrize("k", NEIGHBOURHOODS, ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", PAIR_FIXTURES)
def test_reference_matches_plain_python_dijkstra(fixture, k):
    """The reference all other backends are measured against is itself optimal (it is not a circular check)."""
    raster = FIXTURES[fixture]()
    source, target = _first_and_last(raster)
    reference = _reference(fixture, k, source, target)
    optimum = _plain_python_dijkstra(raster, _steps("cython", k), source, target)
    if math.isinf(optimum):
        assert not reference.reachable
    else:
        assert reference.reachable
        assert reference.cost == pytest.approx(optimum, rel=FLOAT64_TOLERANCE)


# ------------------------------------------------------------------------------------------------ determinism, alias

@pytest.mark.parametrize("delta", [10, 100], ids=lambda d: f"delta{d}")
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"r{k}")
@pytest.mark.parametrize("pair", [((0, 0), (63, 63)), ((32, 32), (63, 0))], ids=["corner", "centre"])
def test_four_thread_delta_stepping_is_exact_on_every_repetition(pair, k, delta):
    """Threads race: a lost relaxation shows up as a cost excess on SOME run, so repeat each query five times.

    This is the bug class of the 2026-08-05 fix (0.08 to 2.8 % excess at 2 or more threads). The bucket width is
    varied because it changes how many buckets, and so how many bucket hand-overs between threads, a query makes.
    """
    raster = FIXTURES["random_64"]()
    source, target = (_flat(cell, raster) for cell in pair)
    case = Case(f"cy-delta-4-d{delta}", "cython", "delta-stepping", FLOAT32, {"num_threads": 4, "delta": delta})
    with _api("cython", raster, k) as api:
        for _ in range(5):
            path = _route(api, source, target, case.algorithm, **case.kwargs)
            _assert_same_cost(case, "random_64", k, raster, source, target, path)


@pytest.mark.parametrize("k", NEIGHBOURHOODS, ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", ["ties_uniform_12", "texture_ties_24"])
@pytest.mark.parametrize("algorithm,kwargs", [
    pytest.param("dijkstra", {}, id="dijkstra"),
    pytest.param("delta-stepping", {"num_threads": 1}, id="delta-stepping-1-thread"),
])
def test_single_thread_cython_is_deterministic(algorithm, kwargs, fixture, k):
    """A repeated query on ties returns the identical cell list (cost parity alone would not notice a flip)."""
    raster = FIXTURES[fixture]()
    source, target = _first_and_last(raster)
    runs = []
    for _ in range(2):
        with _api("cython", raster, k) as api:
            runs.append(_route(api, source, target, algorithm, **kwargs))
    assert runs[0] == runs[1]


@pytest.mark.parametrize("k", NEIGHBOURHOODS, ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", ["ties_uniform_12", "random_24"])
def test_delta_stepping_circular_is_an_alias_at_one_thread(fixture, k):
    """``delta-stepping-circular`` has no code path of its own: same cells as ``delta-stepping`` (cython_api.py)."""
    raster = FIXTURES[fixture]()
    source, target = _first_and_last(raster)
    with _api("cython", raster, k) as api:
        plain = _route(api, source, target, "delta-stepping", num_threads=1)
        circular = _route(api, source, target, "delta-stepping-circular", num_threads=1)
    assert plain == circular


@pytest.mark.skipif(not GPU, reason="no usable CUDA GPU (CuPy missing or no device)")
@pytest.mark.parametrize("k", [1, 2, 3], ids=lambda k: f"r{k}")
@pytest.mark.parametrize("fixture", ["random_24", "texture_ties_24", "wall_gap_14"])
def test_gpu_reported_cost_matches_its_own_path(fixture, k):
    """``RasterGPUAPI.last_path_costs`` is the float32 field value at the target: equal to the re-priced path."""
    raster = FIXTURES[fixture]()
    source, target = _first_and_last(raster)
    with _api("raster_gpu", raster, k) as api:
        path = _route(api, source, target, "delta-stepping")
        reported = api.last_path_costs[0]
    priced = reprice_path(path, raster, EdgeModel(_steps("raster_gpu", k)))
    assert priced.ok
    assert abs(reported - priced.cost) <= self_consistency_tolerance(priced.hops) * priced.cost


# ------------------------------------------------------------------------------------------------ the A* window

#: The two backends whose A* heuristic uses a local minimum (the others use the global one).
ASTAR_WINDOW_BACKENDS = [
    pytest.param(name, marks=pytest.mark.skipif(not _installed(name), reason=f"{name} is not installed"))
    for name in ("networkit", "rustworkx")
]


def _highway_raster() -> np.ndarray:
    """Expensive terrain (50) with a cheap highway row (cost 1) far away from the line source -> target."""
    raster = np.full((21, 61), 50, dtype=np.uint16)
    raster[2, :] = 1
    return raster


def _astar_excess(graph_api: str, **kwargs) -> float:
    raster = _highway_raster()
    source, target = 18 * 61, 18 * 61 + 60
    with _api("cython", raster, 1) as api:
        reference = _route(api, source, target, "dijkstra")
    model = EdgeModel(_steps("cython", 1))
    best = reprice_path(reference, raster, model).cost
    with _api(graph_api, raster, 1) as api:
        path = _route(api, source, target, "astar", **kwargs)
    return (reprice_path(path, raster, model).cost - best) / best


@pytest.mark.parametrize("graph_api", ASTAR_WINDOW_BACKENDS)
def test_astar_finds_the_cheap_corridor_far_from_the_route(graph_api):
    """A* is exact when the cheapest cells lie far away from the straight line between source and target."""
    assert abs(_astar_excess(graph_api)) <= FLOAT64_TOLERANCE


@pytest.mark.parametrize("graph_api", ASTAR_WINDOW_BACKENDS)
def test_astar_ignores_the_former_window_setting_and_stays_exact(graph_api):
    """Regression: ``buffer_radius`` once limited the heuristic's minimum cost to a window around the route.

    A cheaper corridor outside that window made the heuristic overestimate, and networkit and rustworkx returned
    a path 86.2 % more expensive than the optimum on this raster (21 x 61 cells, a cost-1 row 16 cells away,
    ``buffer_radius=1``). The heuristic now always uses the minimum of the whole raster; the setting is ignored
    with a warning.
    """
    with pytest.warns(UserWarning, match="buffer_radius is ignored"):
        excess = _astar_excess(graph_api, buffer_radius=1)
    assert abs(excess) <= FLOAT64_TOLERANCE


# ------------------------------------------------------------------------------------------------ excluded backends

def test_every_selectable_backend_is_in_the_parity_set_or_documented_as_excluded():
    """A new backend name must get a parity case in this file, or be excluded here with a reason."""
    selectable = set(PathFinder.GRADIENT_BACKENDS) | set(PathFinder.FLOAT_BACKENDS)
    covered = {case.graph_api for case in CASES}
    assert selectable - {"raster_fim"} <= covered, sorted(selectable - {"raster_fim"} - covered)


def test_raster_fim_is_excluded_from_the_equality_set_on_purpose():
    """``raster_fim`` solves the continuous eikonal equation and ignores the neighbourhood: a different model.

    Its route cost deviates from the graph optimum by design (measured -1.1 % to +3.7 % on real rasters), its path
    is not a legal edge-model path, and the referee judges it against that envelope, never for equality. Asserting
    equality would be a false failure; asserting nothing would let it drift unnoticed, so the exclusion is pinned.
    """
    assert "raster_fim" in APPROXIMATE_BACKENDS
    assert "raster_fim" not in {case.graph_api for case in CASES}


def test_cugraph_is_not_implemented_and_says_so():
    """``cugraph`` is accepted as a name but raises; it measured 5 to 25 times slower than the Cython kernel."""
    with pytest.raises(NotImplementedError, match="cugraph"):
        get_graph_api_class("cugraph")


def test_cython_api_is_the_default_backend_class():
    """The reference of this file is the class that ``PathFinder(graph_api='cython')`` instantiates."""
    assert get_graph_api_class("cython") is CythonAPI

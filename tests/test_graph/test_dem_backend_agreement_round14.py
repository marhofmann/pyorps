"""Terrain gives the same answer on every routing backend, and never a silently flat route.

One synthetic raster (41 x 41 cells of 1 m, uniform cost) carries a DEM that is a steep cone (slope 200 %, 20 m high)
on the straight line between source and target, so the flat route crosses the summit and every terrain-aware route
has to leave that line. Three requests are run on every installed backend, through ``PathFinder.find_route`` and
through a retained ``SearchSession``:

* ``bare``  - a DEM and no objective. The reference meaning: the edge length is the 3-D length in metres and the
  edge cost is multiplied by ``exp(2 s^2)`` (s = rise over run in metres), i.e. the objective ``{"cost": 1}`` with
  the ``"exponential"`` slope multiplier. This is NOT the old graph-library behaviour at cell sizes other than 1 m
  (that builder measured the run in cells); at 1 m, the cell size of this scenario, the two agree;
* ``cost1`` - the objective ``{"cost": 1}`` and the DEM: the 3-D length stretch only;
* ``grad``  - the objective ``{"cost": 1, "gradient": 30}`` and the DEM: stretch plus a linear slope exposure term.

What is checked, per backend, entry path and request:

* the route is legal and its cost on the reference edge model equals the optimum of an independent plain-Python
  Dijkstra on that model (float64 group ``1e-7``; ``raster_gpu`` accumulates in float32, so the referee's
  ``optimality_tolerance``). ``raster_fim`` solves a continuous eikonal problem whose route on this steep cone costs
  tens of percent more than the discrete optimum, so PathFinder does not route a DEM with it: it announces the
  fallback and runs the cython Dijkstra, which this test then holds to the same equality;
* the route is not the flat route (it stays below the flat route's summit height);
* the reported length is the 3-D length of the returned cells;
* delta-stepping, the retained GPU field and the eikonal backend cannot evaluate slope terms: they run as Dijkstra and
  say so, once per finder or session, with ``AlgorithmFallbackWarning``; the route's ``algorithm`` and
  ``graph_api`` name what really ran.

Skips follow ``test_backend_parity.py``: an optional package that is not installed, and the GPU cases without CuPy or
a CUDA device. The rasters are tiny, so the shared GPU stays free for other work.
"""

from __future__ import annotations

import functools
import heapq
import importlib.util
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest
from exactness_referee import EdgeModel, optimality_tolerance, reprice_path
from rasterio.transform import from_origin

from pyorps import AlgorithmFallbackWarning, GradientOptions, Objective, PathCollection, PathFinder, PyorpsWarning
from pyorps.core.exceptions import FeatureNotImplementedError, InvalidInputError
from pyorps.utils.neighborhood import get_neighborhood_steps

N = 41
CRS = "EPSG:25832"
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)
SOURCE = (2.5, N - 0.5 - 20)             # row 20, column 2
TARGET = (N - 2.5, N - 0.5 - 20)         # row 20, column 38
SOURCE_CELL = 20 * N + 2
TARGET_CELL = 20 * N + 38
IMPASSABLE = 65535
HILL_RADIUS_CELLS = 10.0
HILL_SLOPE = 2.0                         # 200 %: rise per cell on the flank
GRADIENT_WEIGHT = 30.0
FLOAT64_TOLERANCE = 1e-7


# ------------------------------------------------------------------------------------------------ availability

def _installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _gpu_usable() -> bool:
    """CuPy is importable and sees a CUDA device (the guard of test_backend_parity.py)."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # any failure to reach a device means "no GPU"
        return False


GPU = _gpu_usable()


@dataclass(frozen=True)
class Backend:
    """One (graph_api, algorithm) combination."""

    id: str
    graph_api: str
    algorithm: str
    package: str | None = None
    gpu: bool = False


BACKENDS = [
    Backend("cy-dijkstra", "cython", "dijkstra"),
    Backend("cy-delta", "cython", "delta-stepping"),
    Backend("cy-delta-circular", "cython", "delta-stepping-circular"),
    Backend("nx", "networkx", "dijkstra", package="networkx"),
    Backend("nk", "networkit", "dijkstra", package="networkit"),
    Backend("ig", "igraph", "dijkstra", package="igraph"),
    Backend("rx", "rustworkx", "dijkstra", package="rustworkx"),
    Backend("gpu", "raster_gpu", "delta-stepping", gpu=True),
    Backend("fim", "raster_fim", "dijkstra", gpu=True),
]


def _param(backend: Backend):
    marks = []
    if backend.package is not None:
        marks.append(pytest.mark.skipif(not _installed(backend.package), reason=f"{backend.package} is not installed"))
    if backend.gpu:
        marks.append(pytest.mark.skipif(not GPU, reason="no usable CUDA GPU (CuPy missing or no device)"))
    return pytest.param(backend, id=backend.id, marks=marks)


ALL_BACKENDS = [_param(b) for b in BACKENDS]


# ------------------------------------------------------------------------------------------------ the scenario

def hill_dem() -> np.ndarray:
    """A cone centred on (row 20, column 20): 20 m high, 200 % slope, radius 10 cells."""
    rows, cols = np.mgrid[0:N, 0:N]
    distance = np.hypot(rows - 20, cols - 20)
    return np.maximum(0.0, HILL_SLOPE * (HILL_RADIUS_CELLS - distance)).astype(np.float32)


def uniform_raster() -> np.ndarray:
    return np.full((N, N), 100, dtype=np.uint16)


def make_finder(graph_api: str, dem, objective=None, **kwargs) -> PathFinder:
    return PathFinder(
        dataset_source=uniform_raster(), crs=CRS, transform=TRANSFORM,
        source_coords=kwargs.pop("source_coords", SOURCE), target_coords=kwargs.pop("target_coords", TARGET),
        search_space_buffer_m=kwargs.pop("search_space_buffer_m", 200),
        graph_api=graph_api, dem=dem, objective=objective, **kwargs)


#: case -> (what the caller passes as ``objective``, the objective the request MEANS).
def _meant(case: str) -> Objective:
    if case == "bare":
        return Objective({"cost": 1.0}, GradientOptions(multiplier="exponential"))
    if case == "cost1":
        return Objective({"cost": 1.0})
    return Objective({"cost": 1.0, "gradient": GRADIENT_WEIGHT})


PASSED = {"bare": None, "cost1": {"cost": 1.0}, "grad": {"cost": 1.0, "gradient": GRADIENT_WEIGHT}}
CASES = list(PASSED)


# ------------------------------------------------------------------------------------------------ the reference

def _reference_model(case: str) -> EdgeModel:
    steps = get_neighborhood_steps("r2", directed=True)
    luts = _meant(case).build_gradient_luts(steps, cell_size=1.0, quant_scale=1.0)
    return EdgeModel(steps, dem=hill_dem(), gradient_luts=luts)


def _plain_dijkstra(model: EdgeModel, raster: np.ndarray, source: int, target: int) -> tuple[float, int]:
    """Least cost on the reference edge model by a heap search that shares no code with any backend."""
    rows, cols = raster.shape
    dem, grad = model.dem, model.gradient
    table = []
    for dr, dc in model.steps.tolist():
        d_index, intermediates, factor = model.edge(dr, dc)
        table.append((dr, dc, d_index, intermediates, factor))
    best = {source: 0.0}
    hops = {source: 0}
    heap = [(0.0, source)]
    while heap:
        dist, node = heapq.heappop(heap)
        if node == target:
            return dist, hops[node]
        if dist > best.get(node, math.inf):
            continue
        row, col = divmod(node, cols)
        for dr, dc, d_index, intermediates, factor in table:
            cells = [(row + dr, col + dc)] + [(row + a, col + b) for a, b in intermediates]
            if any(not (0 <= r < rows and 0 <= c < cols) or raster[r, c] == IMPASSABLE for r, c in cells):
                continue
            weight = (float(raster[row, col]) + sum(float(raster[r, c]) for r, c in cells)) * factor
            height_diff = abs(float(dem[cells[0]]) - float(dem[row, col]))
            slope_bin = min(int(height_diff * float(grad["bin_factor"][d_index])), grad["n_bins"] - 1)
            mult = float(grad["mult"][slope_bin])
            if not math.isfinite(mult):
                continue
            weight = weight * mult + float(grad["add"][slope_bin]) * float(grad["step_len"][d_index])
            neighbour = cells[0][0] * cols + cells[0][1]
            if dist + weight < best.get(neighbour, math.inf):
                best[neighbour] = dist + weight
                hops[neighbour] = hops[node] + 1
                heapq.heappush(heap, (dist + weight, neighbour))
    return math.inf, 0


@functools.cache
def _optimum(case: str) -> tuple[float, int]:
    return _plain_dijkstra(_reference_model(case), uniform_raster(), SOURCE_CELL, TARGET_CELL)


def _length_3d(cells) -> float:
    """3-D length of a cell sequence, summed per step as sqrt(run^2 + rise^2)."""
    dem = hill_dem()
    cells = [int(c) for c in cells]
    total = 0.0
    for u, v in zip(cells[:-1], cells[1:]):
        run2 = float((v // N - u // N) ** 2 + (v % N - u % N) ** 2)
        rise = abs(float(dem[v // N, v % N]) - float(dem[u // N, u % N]))
        total += math.sqrt(run2 + rise * rise)
    return total


def _peak(cells) -> float:
    dem = hill_dem()
    return float(max(dem[int(c) // N, int(c) % N] for c in cells))


@functools.cache
def _flat_route() -> tuple[int, ...]:
    """The route of the same request without a DEM: the straight line over the summit."""
    path = make_finder("cython", None).find_route()
    return tuple(int(c) for c in path.path_indices)


# ------------------------------------------------------------------------------------------------ running a request

@dataclass
class Run:
    cells: list[int]
    length: float | None
    algorithm: str
    graph_api: str
    fallbacks: list[warnings.WarningMessage] = field(default_factory=list)


def _fallbacks(caught) -> list:
    return [w for w in caught if issubclass(w.category, AlgorithmFallbackWarning)]


def run_request(backend: Backend, case: str, entry: str) -> Run:
    """Route SOURCE -> TARGET twice (a second route must not warn again) and collect what a caller can see."""
    finder = None
    session = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finder = make_finder(backend.graph_api, hill_dem(), PASSED[case])
            if entry == "direct":
                path = finder.find_route(algorithm=backend.algorithm)
                path = finder.find_route(algorithm=backend.algorithm)
                reporter = finder
            else:
                session = finder.search_session(algorithm=backend.algorithm)
                path = session.route([SOURCE, TARGET])
                path = session.route([SOURCE, TARGET])
                reporter = session.finder
                reporter.calculate_path_metrics(path.path_indices, path)   # a session route carries no metrics
        return Run([int(c) for c in path.path_indices], path.total_length, path.algorithm, reporter.graph_api_name,
                   _fallbacks(caught))
    finally:
        if session is not None:
            session.close()
        if finder is not None:
            finder.release_device_resources()


def _expected_fallbacks(backend: Backend, case: str, entry: str) -> int:
    """How many AlgorithmFallbackWarnings one finder or session must emit for two routes."""
    if backend.graph_api == "cython" and backend.algorithm.startswith("delta"):
        return 1                                    # delta-stepping cannot evaluate slope terms
    if backend.graph_api == "raster_fim":
        return 1                                    # the eikonal solver does not route a DEM
    if backend.graph_api == "raster_gpu" and entry == "session":
        return 1                                    # the retained GPU field has no slope terms
    return 0


def _expected_algorithm(backend: Backend, case: str, entry: str) -> str:
    if _expected_fallbacks(backend, case, entry):
        return "dijkstra"
    return backend.algorithm


# ------------------------------------------------------------------------------------------------ the agreement matrix

@pytest.mark.parametrize("entry", ["direct", "session"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_every_backend_returns_the_terrain_aware_optimum(backend, case, entry):
    run = run_request(backend, case, entry)
    where = f"{backend.id}/{case}/{entry}"

    # 1. legal route, same cost as the independent optimum on the reference edge model
    optimum, optimum_hops = _optimum(case)
    priced = reprice_path(run.cells, uniform_raster(), _reference_model(case))
    assert priced.ok, f"{where}: illegal route {priced.violations}"
    assert run.cells[0] == SOURCE_CELL and run.cells[-1] == TARGET_CELL, f"{where}: wrong end points"
    excess = (priced.cost - optimum) / optimum
    tolerance = (optimality_tolerance(optimum_hops, priced.hops) if backend.graph_api == "raster_gpu"
                 else FLOAT64_TOLERANCE)
    assert abs(excess) <= tolerance, (
        f"{where}: cost {priced.cost!r} against the optimum {optimum!r}, excess {excess:+.3e} "
        f"(tolerance {tolerance:.1e})")

    # 2. not the flat route
    assert tuple(run.cells) != _flat_route(), f"{where}: the route is the flat route"
    assert _peak(run.cells) < _peak(_flat_route()), f"{where}: the route still crosses the summit"

    # 3. the reported length is the 3-D length of the returned cells
    assert run.length is not None, f"{where}: no length reported"
    assert run.length == pytest.approx(_length_3d(run.cells), rel=1e-6), f"{where}: not the 3-D length"

    # 4. fallbacks are announced once and the route names the algorithm that ran
    expected = _expected_fallbacks(backend, case, entry)
    assert len(run.fallbacks) == expected, (
        f"{where}: {len(run.fallbacks)} AlgorithmFallbackWarning(s), expected {expected}: "
        f"{[str(w.message) for w in run.fallbacks]}")
    for warning in run.fallbacks:
        assert isinstance(warning.message, PyorpsWarning)
    assert run.algorithm == _expected_algorithm(backend, case, entry), f"{where}: algorithm {run.algorithm!r}"
    if backend.graph_api == "raster_fim":
        assert run.graph_api == "cython", f"{where}: the route says it ran on {run.graph_api!r}"


# ---------------------------------------------------------------------------------------------- bare DEM

def _cost_of(cells, case: str) -> float:
    return reprice_path(cells, uniform_raster(), _reference_model(case), strict=False).cost


def test_a_bare_dem_means_the_exponential_objective():
    """The documented meaning: a DEM without an objective == {"cost": 1} + the "exponential" multiplier."""
    bare = make_finder("cython", hill_dem()).find_route()
    explicit = make_finder("cython", hill_dem(), Objective({"cost": 1.0}, GradientOptions(multiplier="exponential")))
    explicit_path = explicit.find_route()
    assert _cost_of(bare.path_indices, "bare") == pytest.approx(_cost_of(explicit_path.path_indices, "bare"),
                                                                rel=FLOAT64_TOLERANCE)
    assert bare.total_length == pytest.approx(explicit_path.total_length, rel=1e-6)
    # ... and it is not the plain 3-D-length route
    stretch_only = make_finder("cython", hill_dem(), {"cost": 1.0}).find_route()
    assert _peak(bare.path_indices) < _peak(stretch_only.path_indices)


@pytest.mark.parametrize("function", ["exponential", "power", "energy", "sigmoid"])
def test_dem_kwargs_gradient_function_still_selects_the_slope_model(function):
    """``dem_kwargs={"gradient_function": ...}`` was read by the graph-library backends; it still is, on all."""
    bare = make_finder("cython", hill_dem(), dem_kwargs={"gradient_function": function}).find_route()
    meant = make_finder("cython", hill_dem(), Objective({"cost": 1.0}, GradientOptions(multiplier=function)))
    meant_path = meant.find_route()
    assert tuple(bare.path_indices) == tuple(meant_path.path_indices)


def test_a_dem_that_is_not_a_raster_is_refused():
    import geopandas as gpd
    from shapely.geometry import Point
    points = gpd.GeoDataFrame({"z": [1.0]}, geometry=[Point(1.0, 1.0)], crs=CRS)
    with pytest.raises(InvalidInputError, match="raster"):
        make_finder("cython", points)


# ------------------------------------------------------------------------------- delta-stepping variants

def _delta_finder(case: str = "bare") -> PathFinder:
    return make_finder("cython", hill_dem(), PASSED[case])


MULTI_TARGETS = [(N - 2.5, N - 0.5 - row) for row in (15, 20, 25)]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", ["single", "multi_target", "pairwise", "multi_multi"])
def test_delta_stepping_with_slope_terms_equals_dijkstra(case, variant):
    """Every way of calling find_route: asking for delta-stepping gives the Dijkstra answer, with one warning."""
    sources = [SOURCE, (2.5, N - 0.5 - 22), (2.5, N - 0.5 - 18)]

    def call(finder, algorithm):
        if variant == "single":
            return finder.find_route(algorithm=algorithm)
        if variant == "multi_target":
            return finder.find_route(source=SOURCE, target=MULTI_TARGETS, algorithm=algorithm)
        if variant == "pairwise":
            return finder.find_route(source=sources, target=MULTI_TARGETS, algorithm=algorithm, pairwise=True)
        return finder.find_route(source=sources[:2], target=MULTI_TARGETS[:2], algorithm=algorithm)

    def paths_of(result):
        return list(result) if isinstance(result, PathCollection) else [result]

    def cells(result):
        return [[int(c) for c in p.path_indices] for p in paths_of(result)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder = _delta_finder(case)
        asked = call(finder, "delta-stepping")
        again = call(finder, "delta-stepping")
    reference = call(_delta_finder(case), "dijkstra")
    assert len(_fallbacks(caught)) == 1, [str(w.message) for w in _fallbacks(caught)]
    assert cells(asked) == cells(reference) == cells(again)
    for route in cells(asked):
        assert _peak(route) < _peak(_flat_route())
    for path in paths_of(asked):
        assert path.algorithm == "dijkstra"


def test_delta_stepping_without_terrain_is_unchanged_and_silent():
    """No DEM, no slope terms: delta-stepping still runs as delta-stepping and nothing is announced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder = make_finder("cython", None, {"cost": 1.0})
        path = finder.find_route(algorithm="delta-stepping")
    assert not _fallbacks(caught)
    assert path.algorithm == "delta-stepping"


def test_a_dem_with_a_plain_dijkstra_request_is_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_finder("cython", hill_dem()).find_route(algorithm="dijkstra")
    assert not _fallbacks(caught)


# ------------------------------------------------------------------------------------------------ route edits

def test_session_edits_keep_the_terrain_when_the_window_is_rebuilt():
    """A waypoint outside the search window rebuilds the window; the new finder must still see the DEM."""
    top_left = (2.5, N - 0.5 - 5)
    top_right = (N - 2.5, N - 0.5 - 5)
    below_hill = (N - 2.5, N - 0.5 - 20)
    finder = make_finder("cython", hill_dem(), None, source_coords=top_left, target_coords=top_right,
                         search_space_buffer_m=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = finder.search_session(algorithm="delta-stepping")
        session.route([top_left, top_right])
        updated = session.update([top_left, below_hill])      # row 20 is outside the 3 m window around row 5
    try:
        assert session.finder is not finder, "the window was not rebuilt, so this test checks nothing"
        assert session.finder.dem_dataset is not None
        assert len(_fallbacks(caught)) == 1
        fresh = make_finder("cython", hill_dem(), None, source_coords=top_left, target_coords=below_hill,
                            search_space_buffer_m=3).find_route(algorithm="dijkstra")
        cost_session = _cost_of(updated.path_indices, "bare")
        cost_fresh = _cost_of(fresh.path_indices, "bare")
        assert cost_session == pytest.approx(cost_fresh, rel=FLOAT64_TOLERANCE)
        flat = make_finder("cython", None, None, source_coords=top_left, target_coords=below_hill,
                           search_space_buffer_m=3).find_route()
        assert _peak(updated.path_indices) < _peak(flat.path_indices)
    finally:
        session.close()


def test_session_update_with_a_waypoint_reuses_the_dijkstra_trees():
    """update() after route() on a delta-stepping session with a DEM: terrain-aware and still one warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder = make_finder("cython", hill_dem())
        session = finder.search_session(algorithm="delta-stepping")
        session.route([SOURCE, TARGET])
        waypoint = (20.5, N - 0.5 - 5)
        edited = session.update([SOURCE, waypoint, TARGET])
    try:
        assert len(_fallbacks(caught)) == 1
        assert edited.algorithm == "dijkstra"
        cells = [int(c) for c in edited.path_indices]
        assert _peak(cells) < _peak(_flat_route())
    finally:
        session.close()


# ------------------------------------------------------------------------------------------------ weight_precision

def _float_finder(graph_api="networkx", objective="default", dem="none", buffer=200, precision="float32"):
    return PathFinder(
        dataset_source=uniform_raster(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
        search_space_buffer_m=buffer, graph_api=graph_api,
        objective={"cost": 1.0} if objective == "default" else objective,
        dem=hill_dem() if dem == "dem" else None, weight_precision=precision)


@pytest.mark.parametrize("make, fragment", [
    (lambda: _float_finder(objective=None), "objective"),                       # no objective
    (lambda: _float_finder(objective=None, dem="dem"), "objective"),            # a bare DEM is no objective
    (lambda: _float_finder(buffer=None), "search_space_buffer_m"),              # no explicit buffer
    (lambda: _float_finder(precision="float64"), "weight_precision"),           # not a precision at all
    (lambda: _float_finder(precision="Float32"), "weight_precision"),           # a typo is not read as uint16
    (lambda: _float_finder(precision=None), "weight_precision"),
])
def test_weight_precision_refusals_are_invalid_input_errors(make, fragment):
    with pytest.raises(InvalidInputError, match=fragment):
        make()


def test_float32_on_a_backend_without_float32_kernels_is_a_feature_not_implemented_error():
    """The cython kernels are uint16-only: the documented class for "this backend cannot do that" is raised."""
    with pytest.raises(FeatureNotImplementedError, match="cython") as raised:
        _float_finder(graph_api="cython")
    assert isinstance(raised.value, NotImplementedError)
    assert "uint16" in str(raised.value)               # the message still names the way out


@pytest.mark.parametrize("graph_api", [
    pytest.param("networkx", marks=pytest.mark.skipif(not _installed("networkx"), reason="networkx is not installed")),
    pytest.param("raster_gpu", marks=pytest.mark.skipif(not GPU, reason="no usable CUDA GPU")),
])
def test_float32_is_honoured_not_ignored_with_a_dem(graph_api):
    """float32 with an objective and a DEM: the search raster really is float32 and the terrain still counts."""
    finder = _float_finder(graph_api=graph_api, dem="dem")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = finder.find_route()
        assert finder.raster_handler.data.dtype == np.float32
        assert _peak(path.path_indices) < _peak(_flat_route())
        priced = _cost_of(path.path_indices, "cost1")
        assert priced == pytest.approx(_optimum("cost1")[0], rel=1e-5)
    finally:
        finder.release_device_resources()


def test_float32_with_a_dem_on_the_eikonal_backend_is_refused():
    """raster_fim does not route a DEM and falls back to the uint16-only cython kernels: float32 cannot be honoured."""
    with pytest.raises(InvalidInputError, match="weight_precision"):
        _float_finder(graph_api="raster_fim", dem="dem")


@pytest.mark.parametrize("graph_api, dem", [
    pytest.param("networkx", "none",
                 marks=pytest.mark.skipif(not _installed("networkx"), reason="networkx is not installed")),
    pytest.param("raster_gpu", "dem", marks=pytest.mark.skipif(not GPU, reason="no usable CUDA GPU")),
])
def test_a_retained_session_refuses_a_float32_finder_it_cannot_search(graph_api, dem):
    """The Dijkstra and delta trees read uint16 rasters: float32 used to die inside the kernel with a buffer error."""
    finder = _float_finder(graph_api=graph_api, dem=dem)
    try:
        with pytest.raises(InvalidInputError, match="float32"):
            finder.search_session(algorithm="dijkstra")
    finally:
        finder.release_device_resources()


# ------------------------------------------------------------------------------------------------ DEM voids

VOID = (slice(18, 23), slice(18, 23))        # the summit of the cone


@pytest.mark.parametrize("precision", ["uint16", "float32"])
@pytest.mark.parametrize("graph_api", ["cython", "networkx"])
def test_cells_without_elevation_are_forbidden_on_both_precisions(graph_api, precision):
    """A void in the DEM forbids its cells: 65535 on a uint16 raster, +inf on the float32 weight raster."""
    if graph_api == "networkx" and not _installed("networkx"):
        pytest.skip("networkx is not installed")
    if precision == "float32" and graph_api == "cython":
        pytest.skip("the cython kernels are uint16-only")
    dem = hill_dem()
    dem[VOID] = np.nan
    finder = PathFinder(
        dataset_source=uniform_raster(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
        search_space_buffer_m=200, graph_api=graph_api, dem=dem, objective={"cost": 1.0},
        weight_precision=precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = finder.find_route()
    rows, cols = np.divmod(np.asarray(path.path_indices), N)
    in_void = (rows >= 18) & (rows < 23) & (cols >= 18) & (cols < 23)
    assert not in_void.any(), "the route crosses cells that have no elevation"


def test_the_new_warning_is_a_pyorps_warning_exported_everywhere():
    import pyorps
    import pyorps.core
    assert pyorps.AlgorithmFallbackWarning is pyorps.core.AlgorithmFallbackWarning
    assert issubclass(AlgorithmFallbackWarning, PyorpsWarning)
    assert issubclass(AlgorithmFallbackWarning, UserWarning)
    assert "AlgorithmFallbackWarning" in pyorps.__all__ and "AlgorithmFallbackWarning" in pyorps.core.__all__

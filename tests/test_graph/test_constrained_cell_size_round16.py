"""Constrained routing prices the terrain per metre for every cell size (round 16).

Defect: the kernels got the raster's cell size only together with a DEM. Without a DEM they priced the terrain per CELL
(``cost x cost factor``), while the tower and turn costs are per tower and the report (``_terrain_eur``) prices the
terrain per METRE. On a raster that is not 1 m the towers and turns were over-weighted by the cell size, the route
differed from the one found with a flat DEM, and the reported total differed from what the search minimised.

Decision (owner): the search prices the terrain per metre everywhere. The finder now passes ``cell_size`` to the
kernels with or without a DEM. At 1 m (the kernels' default) nothing changes; the values pinned below were recorded
before the change (scratch script, 2026-11-12).

A flat DEM is the oracle: its gradient factors are exactly 1 and its slope terms 0, so with ``cell_size`` passed the
run without a DEM must take the same route and report the same numbers. The GPU solvers are not covered here: their
kernels never multiply the terrain by the cell size (with or without a DEM) and need a kernel change (open point).
"""
import warnings

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps.utils.constrained_path_algorithms as constrained_algorithms
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

CRS = "EPSG:25832"
N = 30
CELL_SIZES = (0.5, 2.0, 5.0)
BACKENDS = ("cython", "cython_parallel")


def profile(s=1.0, constant_towers=False):
    """The test overhead line with its spans in metres scaled by the cell size ``s`` (same shape in cells)."""
    terrain_cost_map = {0: 300, 65535: 300} if constant_towers else {0: 100, 100: 500, 500: 2000}
    return dict(
        name="test_ohl", description="test overhead line",
        soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
        angle_cost_function="linear", angle_cost_params={"scale": 10.0},
        min_span_m=3.0 * s, max_span_m=8.0 * s, span_bin_size_m=1.0 * s,
        tower_cost_params={
            "terrain_cost_map": terrain_cost_map,
            "angle_types": {
                "suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                "angle": {"max_angle_deg": 90.0, "base_cost": 200.0},
            },
        },
    )


def zigzag(expensive=400):
    """Expensive everywhere except a cheap (10) corridor that goes up and down."""
    r = np.full((N, N), expensive, dtype=np.uint16)
    pts = [(15, 3), (15, 8), (9, 12), (21, 17), (15, 22), (15, 26)]
    for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:]):
        n = max(abs(r1 - r0), abs(c1 - c0)) * 2 + 1
        for t in np.linspace(0, 1, n):
            rr = int(round(r0 + (r1 - r0) * t))
            cc = int(round(c0 + (c1 - c0) * t))
            r[rr, cc] = 10
            r[max(rr - 1, 0), cc] = 10
    return r


def ramp():
    """Cost rises to the south, with an expensive block on the direct line (the reviewer's raster)."""
    rows = np.arange(N, dtype=np.float64)[:, None]
    r = (60 + 12 * rows + np.zeros((1, N))).astype(np.uint16)
    r[10:20, 12:16] = 900
    return r


def noise(seed):
    rng = np.random.default_rng(seed)
    r = rng.integers(20, 400, (N, N)).astype(np.uint16)
    r[rng.random((N, N)) < 0.04] = 65535
    r[14, 3] = 100  # the two end points (cells of row 14) are never blocked
    r[14, 26] = 100
    return r


RASTERS = {"ramp": ramp, "zigzag": zigzag, **{f"noise{k}": (lambda k=k: noise(k)) for k in range(4)}}


def route(raster, s=1.0, backend="cython", dem=False, objective=None, prof=None, tr=None, source=None, target=None):
    """Find the route on ``raster`` with cell size ``s``; the end points sit 23 cells apart on the middle row."""
    kwargs = {}
    if dem:
        kwargs["dem"] = np.zeros(raster.shape, np.float32)
    if objective is not None:
        kwargs["objective"] = objective
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=raster.copy(), crs=CRS,
            transform=from_origin(0.0, N * s, s, s) if tr is None else tr,
            source_coords=(3.5 * s, 15.5 * s) if source is None else source,
            target_coords=(26.5 * s, 15.5 * s) if target is None else target,
            search_space_buffer_m=100 * s, profile=prof or profile(s), graph_api=backend, **kwargs)
        return finder, finder.find_route()


def signature(path):
    """Everything a route reports, as plain Python objects (compared with ``==``)."""
    return dict(
        path=[int(i) for i in path.path_indices],
        towers=[(t.cell_index, t.tower_type, t.terrain_cost, t.angle_cost, t.total_cost) for t in path.towers],
        terrain=path.total_terrain_cost, towers_total=path.total_tower_cost,
        angle=path.total_angle_penalty_cost, total=path.total_cost, spans=list(path.spans),
        turns=list(path.turn_angles), breakdown=dict(path.cost_breakdown), length=path.total_length)


@pytest.fixture
def kernel_spy(monkeypatch):
    """Record the kwargs, the optimal distance and the towers of every constrained kernel call."""
    calls = []
    original = constrained_algorithms.constrained_dijkstra_2d
    original_parallel = constrained_algorithms.constrained_delta_stepping_2d

    def spy(**kwargs):
        path, towers, dist = original(return_dist=1, **kwargs)
        calls.append({"kernel": "dijkstra", "kwargs": kwargs, "dist": dist, "towers": [int(t) for t in towers]})
        return path, towers

    def spy_parallel(**kwargs):
        calls.append({"kernel": "parallel", "kwargs": kwargs})
        return original_parallel(**kwargs)

    monkeypatch.setattr(constrained_algorithms, "constrained_dijkstra_2d", spy)
    monkeypatch.setattr(constrained_algorithms, "constrained_delta_stepping_2d", spy_parallel)
    return calls


# ------------------------------------------------------------------ (a) 1 m: nothing changes
#: Recorded before the change at 1 m cells. The plain no-objective cases of ramp and zigzag are pinned in
#: test_constrained_objective_units_round16.py; these are the objective runs {"cost": 1, "length": 10}.
RECORDED_1M = {
    "ramp": dict(
        path=[423, 394, 365, 336, 307, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
              322, 353, 384, 415, 446],
        towers=[(423, "terminal", 980.0, 0.0), (278, "angle", 755.0, 200.0), (285, "suspension", 755.0, 50.0),
                (291, "angle", 755.0, 200.0), (446, "terminal", 980.0, 0.0)],
        terrain=5255.591784211245, towers_total=4675.0, angle=900.0, total=10830.591784211245,
        length=27.142135623730955, kernel_dist=638819.082898736, scale=72.01538461538462),
    "zigzag": dict(
        path=[423, 424, 425, 426, 427, 428, 429, 461, 493, 525, 526, 527, 528, 529, 530, 502, 474, 446],
        towers=[(423, "terminal", 140.0, 0.0), (429, "angle", 140.0, 200.0), (525, "angle", 140.0, 200.0),
                (530, "angle", 140.0, 200.0), (446, "terminal", 140.0, 0.0)],
        terrain=3884.5492635684027, towers_total=1300.0, angle=796.9515353123397, total=5981.500798880742,
        length=24.41640786499874, kernel_dist=911322.33492589, scale=159.8390243902439),
}


@pytest.mark.parametrize("name", sorted(RECORDED_1M))
def test_objective_run_at_one_metre_is_bit_identical_to_the_recorded_values(name, kernel_spy):
    rec = RECORDED_1M[name]
    finder, path = route(RASTERS[name](), objective={"cost": 1.0, "length": 10.0})
    assert [int(i) for i in path.path_indices] == rec["path"]
    assert [(t.cell_index, t.tower_type, t.terrain_cost, t.angle_cost) for t in path.towers] == rec["towers"]
    assert path.total_terrain_cost == rec["terrain"]
    assert path.total_tower_cost == rec["towers_total"]
    assert path.total_angle_penalty_cost == rec["angle"]
    assert path.total_cost == rec["total"]
    assert path.total_length == rec["length"]
    assert finder._search_scale == rec["scale"]
    (call,) = kernel_spy
    assert call["dist"] == rec["kernel_dist"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["ramp", "zigzag", "noise0", "noise1"])
def test_at_one_metre_the_run_without_a_dem_is_the_run_with_a_flat_dem(name, backend):
    """The two paths were already identical at 1 m (recorded: 20 of 20 runs); they must stay so."""
    _, plain = route(RASTERS[name](), backend=backend)
    _, flat = route(RASTERS[name](), backend=backend, dem=True)
    assert signature(plain) == signature(flat)


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_metre_kernel_call_carries_a_cell_size_of_exactly_one(backend, kernel_spy):
    route(ramp(), backend=backend)
    (call,) = kernel_spy
    assert call["kwargs"]["cell_size"] == 1.0


# ------------------------------------------------------------------ the kernels get the cell size without a DEM
@pytest.mark.parametrize("dem", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("s", (0.5, 1.0, 2.0, 5.0))
def test_kernel_gets_the_cell_size_with_and_without_a_dem(s, backend, dem, kernel_spy):
    route(ramp(), s=s, backend=backend, dem=dem)
    (call,) = kernel_spy
    assert call["kwargs"]["cell_size"] == s
    assert ("dem_data" in call["kwargs"]) is dem


# ------------------------------------------------------------------ (b) equivalence oracle: no DEM == flat zero DEM
@pytest.mark.parametrize("name", sorted(RASTERS))
@pytest.mark.parametrize("s", CELL_SIZES)
def test_no_dem_and_a_flat_dem_give_the_same_route_and_numbers(s, name):
    _, plain = route(RASTERS[name](), s=s)
    _, flat = route(RASTERS[name](), s=s, dem=True)
    assert signature(plain) == signature(flat)


@pytest.mark.parametrize("name", ["ramp", "zigzag", "noise0", "noise2"])
@pytest.mark.parametrize("s", CELL_SIZES)
def test_flat_dem_equivalence_holds_for_the_parallel_backend_up_to_ties(s, name):
    """The parallel kernel may break a tie differently, so the routes are compared by what they cost."""
    _, plain = route(RASTERS[name](), s=s, backend="cython_parallel")
    _, flat = route(RASTERS[name](), s=s, backend="cython_parallel", dem=True)
    assert plain.total_cost == pytest.approx(flat.total_cost, rel=1e-9)
    assert plain.total_terrain_cost + plain.total_tower_cost + plain.total_angle_penalty_cost == pytest.approx(
        plain.total_cost, rel=1e-12)


@pytest.mark.parametrize("name", ["ramp", "zigzag", "noise0", "noise1", "noise3"])
@pytest.mark.parametrize("s", CELL_SIZES)
def test_flat_dem_equivalence_holds_with_an_objective(s, name):
    objective = {"cost": 1.0, "length": 10.0}
    _, plain = route(RASTERS[name](), s=s, objective=objective)
    _, flat = route(RASTERS[name](), s=s, objective=objective, dem=True)
    assert signature(plain) == signature(flat)


def test_reviewers_ramp_at_two_metres_prices_the_terrain_per_metre(kernel_spy):
    """Before: kernel optimum 8599.1 against a report of 13583.3 (a flat DEM gave exactly 13583.3)."""
    _, plain = route(ramp(), s=2.0)
    _, flat = route(ramp(), s=2.0, dem=True)
    assert signature(plain) == signature(flat)
    terminals = sum(t.total_cost for t in plain.towers if t.tower_type == "terminal")
    assert plain.total_cost - terminals == pytest.approx(13583.3, abs=0.05)
    plain_call, _ = kernel_spy
    assert plain.total_cost - terminals == pytest.approx(plain_call["dist"], rel=1e-6)
    assert plain_call["dist"] != pytest.approx(8599.1, abs=1.0)


@pytest.mark.parametrize("xsize, ysize", [(2.0, 1.0), (1.0, 2.0)])
def test_non_square_cells_give_the_same_route_with_and_without_a_dem(xsize, ysize):
    """The finder measures every step with the x size of the cell (``transform.a``), with or without a DEM.

    That is wrong for the y direction of a raster whose cells are not square (open point, reported separately); this
    test pins only that the two runs take the same route.
    """
    tr = from_origin(0.0, N * ysize, xsize, ysize)
    kw = dict(tr=tr, source=(3.5 * xsize, 15.5 * ysize), target=(26.5 * xsize, 15.5 * ysize), prof=profile(2.0))
    _, plain = route(ramp(), **kw)
    _, flat = route(ramp(), dem=True, **kw)
    assert signature(plain) == signature(flat)


# ------------------------------------------------------------------ (c) scaling oracles
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("s", (0.5, 1.0, 2.0, 4.0))
def test_uniform_terrain_is_priced_per_metre_at_every_cell_size(s, backend):
    """A straight 24 m route over terrain of 100 per metre, hand computed.

    Spans of 4 to 12.5 m (a span stays below the maximal span): 24 m need two spans of exactly 12 m, so three towers,
    the same at every cell size: two terminals (foundation 500, no type cost) and one suspension tower (foundation
    500 + type 50). Terrain 100 x 24 = 2400, towers 2 x 500 + 550 = 1550, no turns, total 3950. End points 24 m
    apart and spans in metres: the same physical problem at 0.5, 1, 2 and 4 m cells, so the same numbers.
    """
    cells = int(round(24.0 / s))  # up to 48 cells: a raster twice as wide as the others
    _, path = route(np.full((N, 2 * N), 100, np.uint16), s=s, backend=backend,
                    prof=dict(profile(1.0), min_span_m=4.0, max_span_m=12.5),
                    target=((3 + cells + 0.5) * s, 15.5 * s))
    assert path.total_length == pytest.approx(24.0, rel=1e-12)
    assert path.n_towers == 3
    assert path.spans == pytest.approx([12.0, 12.0], rel=1e-12)
    assert path.total_terrain_cost == pytest.approx(100.0 * 24.0, rel=1e-12)
    assert path.total_tower_cost == pytest.approx(2 * 500.0 + 550.0, rel=1e-12)
    assert path.total_angle_penalty_cost == 0.0
    assert path.total_cost == pytest.approx(2400.0 + 1550.0, rel=1e-12)


@pytest.mark.parametrize("name", ["ramp", "zigzag", "noise0", "noise1"])
@pytest.mark.parametrize("s", (2.0, 4.0, 5.0))
def test_a_raster_at_cell_size_s_is_the_raster_times_s_at_one_metre(s, name, kernel_spy):
    """Terrain per metre makes ``(R, cell size s)`` the same search as ``(s x R, cell size 1)``.

    The tower foundation costs are constant here (they are keyed by the raster VALUE, which the factor s changes),
    and the spans of the one-metre run are the spans of the other in cells. Both searches then add up exactly the
    same numbers, so route, towers, optimal distance and every report agree bit for bit (lengths differ by s).
    """
    raster = RASTERS[name]()
    scaled = raster.astype(np.int64) * int(s)
    scaled[raster == 65535] = 65535
    _, fine = route(raster, s=s, prof=profile(s, constant_towers=True))
    _, unit = route(scaled.astype(np.uint16), s=1.0, prof=profile(1.0, constant_towers=True))
    fine_call, unit_call = kernel_spy
    assert fine_call["dist"] == unit_call["dist"]
    a, b = signature(fine), signature(unit)
    assert a.pop("length") == pytest.approx(s * b.pop("length"), rel=1e-12)
    assert a.pop("spans") == pytest.approx([s * x for x in b.pop("spans")], rel=1e-12)
    assert a == b


# ------------------------------------------------------------------ (d) the kernel optimum is what the report adds up
@pytest.mark.parametrize("objective", [
    None, {"cost": 1.0}, {"cost": 1.0, "length": 10.0}, {"cost": 1.0, "smoothness": 2.0, "tower": 5.0},
])
@pytest.mark.parametrize("name, s", [
    ("ramp", 1.0), ("ramp", 0.5), ("ramp", 2.0), ("ramp", 5.0), ("zigzag", 1.0), ("zigzag", 0.5),
    ("noise1", 1.0), ("noise1", 2.0), ("noise2", 0.5), ("noise2", 5.0)])
def test_total_cost_minus_terminals_is_the_kernel_optimum_at_every_cell_size(s, name, objective, kernel_spy):
    """Reported total minus the terminal towers = kernel optimum / search scale.

    Only for routes where the report keeps every tower the search placed: the terminal post-processing
    (``_enforce_terminal_tower_spans``, a separate decision) drops a search tower close to a terminal on some routes
    (e.g. the zigzag raster at 2 m and 5 m cells, ``noise0`` at every cell size), and the identity then misses it.
    """
    finder, path = route(RASTERS[name](), s=s, objective=objective)
    (call,) = kernel_spy
    kept = [t.cell_index for t in path.towers if t.tower_type != "terminal"]
    assert kept == call["towers"]
    terminals = sum(t.total_cost for t in path.towers if t.tower_type == "terminal")
    assert path.total_cost - terminals == pytest.approx(call["dist"] / finder._search_scale, rel=1e-6)

"""Constrained routing adds and reports every cost in one unit (round 16).

Three defects of ``ConstrainedPathFinder`` under an objective, found through the GUI's objective weights:

1. An objective that leaves the cost-only fast path (a ``length`` weight, a second layer) searched on the quantised
   combined raster (objective x scale) while the turn and tower tables stayed in raw profile units, and reported the
   quantised terrain sum: ``{"cost": 1, "length": 1e-6}`` gave another route than ``{"cost": 1}`` and a terrain cost
   of 1.5 million on a uniform raster. Now the tables are multiplied by the same scale and the report divides by it.
2. The reported tower cost mixed the weighted foundation table with the unweighted tower-type cost (``tower: 0``
   still showed 100). Now every tower number is the profile cost times the ``tower`` weight.
3. ``total_angle_penalty_cost`` was always 0.0 although the search charges turn penalties. Now it is computed from
   the route with the table the search used.

Without an objective, routes and every number except the turn penalty (and so ``total_cost``) are unchanged; the
values below were recorded before the change.
"""
import warnings

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps.utils.constrained_path_algorithms as constrained_algorithms
from pyorps.core.exceptions import NumericalPrecisionError
from pyorps.core.warning_types import PyorpsWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

CRS = "EPSG:25832"
N = 30
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)
SOURCE = (3.5, 15.5)
TARGET = (26.5, 15.5)
TURN_SCALE = 10.0  # linear turn penalty: 10 per degree

PROFILE = dict(
    name="test_ohl", description="test overhead line",
    soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
    angle_cost_function="linear", angle_cost_params={"scale": TURN_SCALE},
    min_span_m=3.0, max_span_m=8.0, span_bin_size_m=1.0,
    tower_cost_params={
        "terrain_cost_map": {0: 100, 100: 500, 500: 2000},
        "angle_types": {
            "suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
            "angle": {"max_angle_deg": 90.0, "base_cost": 200.0},
        },
    },
)


def uniform():
    return np.full((N, N), 100, dtype=np.uint16)


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
    """Cost rises to the south, with an expensive block on the direct line."""
    rows = np.arange(N, dtype=np.float64)[:, None]
    r = (60 + 12 * rows + np.zeros((1, N))).astype(np.uint16)
    r[10:20, 12:16] = 900
    return r


def make(objective=None, raster=None, profile=None, target=TARGET):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=uniform() if raster is None else raster,
            crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=target,
            search_space_buffer_m=100, profile=dict(profile or PROFILE),
            objective=objective, graph_api="cython")


def route(objective=None, raster=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make(objective, raster, **kwargs).find_route()


@pytest.fixture
def kernel_spy(monkeypatch):
    """Record the optimal distance and the towers of every constrained Dijkstra call."""
    original = constrained_algorithms.constrained_dijkstra_2d
    calls = []

    def spy(**kwargs):
        path, towers, dist = original(return_dist=1, **kwargs)
        calls.append({"dist": dist, "towers": [int(t) for t in towers], "raster": kwargs["raster"]})
        return path, towers

    monkeypatch.setattr(constrained_algorithms, "constrained_dijkstra_2d", spy)
    return calls


def terminal_cost(path):
    return sum(t.total_cost for t in path.towers if t.tower_type == "terminal")


# ------------------------------------------------------------------ results without an objective are unchanged
#: Recorded before the change (scratch script, 2026-11-12). The uniform raster is left out on purpose: its report
#: drops a search tower near the target and shows a 9 m span above max_span_m (an open defect of the terminal
#: post-processing, not touched here), and a test must not pin it.
RECORDED = {
    "zigzag": dict(
        raster=zigzag, target=TARGET,
        path=[423, 424, 425, 426, 427, 428, 429, 461, 493, 525, 526, 527, 528, 529, 530, 502, 474, 446],
        towers=[(423, "terminal", 140.0, 0.0), (429, "angle", 140.0, 200.0), (525, "angle", 140.0, 200.0),
                (530, "angle", 140.0, 200.0), (446, "terminal", 140.0, 0.0)],
        terrain=3640.3636123247406, towers_total=1300.0, length=24.41640786499874,
        spans=[6.0, 6.708203932499369, 5.0, 6.708203932499369],
        turns=[0.0, 26.56505117707799, 26.56505117707799, 26.56505117707799, 0.0]),
    "ramp": dict(
        raster=ramp, target=TARGET,
        path=[423, 394, 365, 336, 307, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
              322, 353, 384, 415, 446],
        towers=[(423, "terminal", 980.0, 0.0), (278, "angle", 755.0, 200.0), (285, "suspension", 755.0, 50.0),
                (291, "angle", 755.0, 200.0), (446, "terminal", 980.0, 0.0)],
        terrain=4984.142853498728, towers_total=4675.0, length=27.142135623730955,
        spans=[7.0710678118654755, 7.0, 6.0, 7.0710678118654755], turns=[0.0, 45.0, 0.0, 45.0, 0.0]),
    "ramp_diagonal": dict(
        raster=ramp, target=(24.5, 4.5),
        path=[423, 454, 485, 516, 548, 580, 612, 613, 614, 615, 616, 617, 618, 619, 650, 681, 712, 743, 774],
        towers=[(423, "terminal", 980.0, 0.0), (516, "angle", 1115.0, 200.0), (612, "angle", 1250.0, 200.0),
                (619, "angle", 1250.0, 200.0), (774, "terminal", 1475.0, 0.0)],
        terrain=7368.855495911773, towers_total=6670.0, length=25.02191243148413,
        spans=[4.242640687119285, 6.708203932499369, 7.0, 7.0710678118654755],
        turns=[0.0, 18.43494882292201, 26.56505117707799, 45.0, 0.0]),
}


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_results_without_objective_match_the_recorded_values(name):
    rec = RECORDED[name]
    path = route(None, rec["raster"](), target=rec["target"])
    assert [int(i) for i in path.path_indices] == rec["path"]
    assert [(t.cell_index, t.tower_type) for t in path.towers] == [t[:2] for t in rec["towers"]]
    for tower, (_, _, terrain, type_cost) in zip(path.towers, rec["towers"]):
        assert tower.terrain_cost == pytest.approx(terrain, rel=1e-12)
        assert tower.angle_cost == pytest.approx(type_cost, rel=1e-12)
        assert tower.total_cost == pytest.approx(terrain + type_cost, rel=1e-12)
    assert path.total_terrain_cost == pytest.approx(rec["terrain"], rel=1e-12)
    assert path.total_tower_cost == pytest.approx(rec["towers_total"], rel=1e-12)
    assert path.total_length == pytest.approx(rec["length"], rel=1e-12)
    assert path.spans == pytest.approx(rec["spans"], rel=1e-12)
    assert path.turn_angles == pytest.approx(rec["turns"], rel=1e-12)
    # defect 3: the turn penalty was reported as 0.0; it is 10 per degree of every turn of the route
    expected_turns = TURN_SCALE * sum(rec["turns"])
    assert expected_turns > 0
    assert path.total_angle_penalty_cost == pytest.approx(expected_turns, rel=1e-9)
    assert path.cost_breakdown == pytest.approx(
        {"terrain": rec["terrain"], "towers": rec["towers_total"], "angle_penalties": expected_turns}, rel=1e-9)
    assert path.total_cost == pytest.approx(rec["terrain"] + rec["towers_total"] + expected_turns, rel=1e-12)
    assert path.metrics is None and path.feasibility is None


@pytest.mark.parametrize("objective", [{"cost": 1.0}, {"cost": 1.0, "smoothness": 1.0, "tower": 1.0}])
def test_cost_one_objective_reports_exactly_what_no_objective_reports(objective):
    legacy = route(None, zigzag())
    weighted = route(objective, zigzag())
    assert list(weighted.path_indices) == list(legacy.path_indices)
    for a, b in zip(legacy.towers, weighted.towers):
        assert (a.cell_index, a.terrain_cost, a.angle_cost, a.total_cost) == (
            b.cell_index, b.terrain_cost, b.angle_cost, b.total_cost)
    for field in ("total_terrain_cost", "total_tower_cost", "total_angle_penalty_cost", "total_cost"):
        assert getattr(weighted, field) == getattr(legacy, field), field
    assert {k: weighted.cost_breakdown[k] for k in legacy.cost_breakdown} == legacy.cost_breakdown
    # with an objective the profile's own costs are listed too; at weight 1 they equal the weighted ones
    assert weighted.cost_breakdown["profile_towers"] == legacy.total_tower_cost
    assert weighted.cost_breakdown["profile_angle_penalties"] == legacy.total_angle_penalty_cost


# ------------------------------------------------------------------ defect 1: one unit for the search
def test_tiny_length_weight_gives_the_cost_only_route():
    cost_only = route({"cost": 1.0}, zigzag())
    tiny = route({"cost": 1.0, "length": 1e-6}, zigzag())
    assert list(tiny.path_indices) == list(cost_only.path_indices)
    assert [t.cell_index for t in tiny.towers] == [t.cell_index for t in cost_only.towers]
    assert tiny.total_tower_cost == cost_only.total_tower_cost
    assert tiny.total_angle_penalty_cost == pytest.approx(cost_only.total_angle_penalty_cost, rel=1e-12)
    # the terrain is priced on the quantised raster: equal up to the quantisation step
    assert tiny.total_terrain_cost == pytest.approx(cost_only.total_terrain_cost, rel=1e-4)


@pytest.mark.parametrize("length_weight, expected_terrain", [(1e-6, 2300.0 + 23e-6), (50.0, 2300.0 + 50 * 23.0)])
def test_terrain_is_reported_in_the_users_units(length_weight, expected_terrain):
    """Uniform cost 100, a 23 m straight route: terrain = 100 x 23 + w_length x 23 (it was 1 507 282)."""
    path = route({"cost": 1.0, "length": length_weight}, uniform())
    assert path.total_length == pytest.approx(23.0)
    assert path.total_terrain_cost == pytest.approx(expected_terrain, rel=1e-6)
    assert path.feasibility == pytest.approx(expected_terrain, rel=1e-6)
    assert path.total_tower_cost == route({"cost": 1.0}, uniform()).total_tower_cost


def test_length_weight_equals_the_length_folded_into_the_raster(kernel_spy):
    """``{"cost": 1, "length": 50}`` on R is ``{"cost": 1}`` on R + 50 (exact for cost plus a constant per metre).

    The foundation cost is the same on every cell here (it would otherwise read the shifted land use), and the
    largest objective value is 302 = 65534 / 217, so the quantisation is exact: the search inputs of the weighted
    run are exactly 217 times those of the folded run.
    """
    profile = dict(PROFILE, tower_cost_params=dict(PROFILE["tower_cost_params"],
                                                   terrain_cost_map={0: 100, 1000: 100}))
    raster = zigzag(expensive=252)
    weighted = route({"cost": 1.0, "length": 50.0}, raster, profile=profile)
    folded = route(None, (raster + 50).astype(np.uint16), profile=profile)
    weighted_call, folded_call = kernel_spy
    assert np.array_equal(weighted_call["raster"], folded_call["raster"].astype(np.int64) * 217)
    assert list(weighted.path_indices) == list(folded.path_indices)
    assert [t.cell_index for t in weighted.towers] == [t.cell_index for t in folded.towers]
    assert weighted_call["dist"] / 217.0 == pytest.approx(folded_call["dist"], rel=1e-6)
    for field in ("total_terrain_cost", "total_tower_cost", "total_angle_penalty_cost", "total_cost"):
        assert getattr(weighted, field) == pytest.approx(getattr(folded, field), rel=1e-9), field


def test_cost_weight_is_weighed_against_the_tower_and_turn_costs():
    """Only the ratios of the weights matter: ``cost 2`` is ``cost 1`` with half-price towers and turns."""
    doubled = route({"cost": 2.0}, zigzag())
    halved = route({"cost": 1.0, "tower": 0.5, "smoothness": 0.5}, zigzag())
    assert list(doubled.path_indices) == list(halved.path_indices)
    assert [t.cell_index for t in doubled.towers] == [t.cell_index for t in halved.towers]
    for field in ("total_terrain_cost", "total_tower_cost", "total_angle_penalty_cost", "total_cost"):
        assert getattr(doubled, field) == pytest.approx(2.0 * getattr(halved, field), rel=1e-12), field
    # cheaper towers and turns buy a route over the cheap corridor with more turning (profile units)
    assert (halved.cost_breakdown["profile_angle_penalties"]
            > route({"cost": 1.0}, zigzag()).total_angle_penalty_cost)


def test_too_small_cell_weights_are_refused_instead_of_overflowing():
    finder = make({"cost": 1e-300})
    with pytest.raises(NumericalPrecisionError, match="float32 range"):
        finder.find_route()


# ------------------------------------------------------------------ defect 2: every tower number is weighted
def test_tower_weight_zero_reports_no_tower_cost():
    path = route({"cost": 1.0, "tower": 0.0}, uniform())
    assert path.total_tower_cost == 0.0
    assert all(t.terrain_cost == 0.0 and t.angle_cost == 0.0 and t.total_cost == 0.0 for t in path.towers)
    assert path.cost_breakdown["profile_towers"] == route(None, uniform()).total_tower_cost


def test_tower_weight_five_reports_five_times_the_profile_cost():
    base = route({"cost": 1.0}, uniform())
    heavy = route({"cost": 1.0, "tower": 5.0}, uniform())
    assert list(heavy.path_indices) == list(base.path_indices)
    assert heavy.total_tower_cost == 5.0 * base.total_tower_cost == 10500.0
    for a, b in zip(base.towers, heavy.towers):
        assert (b.terrain_cost, b.angle_cost, b.total_cost) == (
            5.0 * a.terrain_cost, 5.0 * a.angle_cost, 5.0 * a.total_cost)
    assert heavy.tower_type_costs == {k: 5.0 * v for k, v in base.tower_type_costs.items()}
    assert heavy.cost_breakdown["profile_towers"] == base.total_tower_cost


def test_terminal_tower_cost_follows_the_tower_weight():
    profile = dict(PROFILE, terminal_tower_cost=300.0)
    base = route({"cost": 1.0}, uniform(), profile=profile)
    heavy = route({"cost": 1.0, "tower": 3.0}, uniform(), profile=profile)
    terminals = [(a, b) for a, b in zip(base.towers, heavy.towers) if a.tower_type == "terminal"]
    assert len(terminals) == 2
    assert all(a.angle_cost == 300.0 and b.angle_cost == 900.0 for a, b in terminals)


# ------------------------------------------------------------------ defect 3: the turn penalty is reported
def test_smoothness_weight_scales_the_reported_turn_penalty():
    base = route({"cost": 1.0}, zigzag())
    smooth = route({"cost": 1.0, "smoothness": 2.0}, zigzag())
    assert list(smooth.path_indices) == list(base.path_indices)  # the corridor wins either way
    assert base.total_angle_penalty_cost == pytest.approx(TURN_SCALE * sum(base.turn_angles), rel=1e-9)
    assert smooth.total_angle_penalty_cost == pytest.approx(2.0 * base.total_angle_penalty_cost, rel=1e-12)
    assert smooth.cost_breakdown["profile_angle_penalties"] == pytest.approx(base.total_angle_penalty_cost)
    assert smooth.cost_breakdown["angle_penalties"] == smooth.total_angle_penalty_cost


def test_straight_route_has_no_turn_penalty():
    path = route(None, uniform())
    assert path.total_angle_penalty_cost == 0.0
    assert path.total_cost == path.total_terrain_cost + path.total_tower_cost


@pytest.mark.parametrize("objective", [
    None, {"cost": 1.0}, {"cost": 1.0, "length": 1e-6}, {"cost": 1.0, "length": 50.0},
    {"cost": 1.0, "smoothness": 2.0}, {"cost": 1.0, "tower": 5.0}, {"cost": 1.0, "smoothness": 0.5},
    {"smoothness": 2.0},
])
def test_total_cost_is_what_the_search_minimised(objective, kernel_spy):
    """total = terrain + towers + turns, and minus the terminal towers (which the search does not charge, they
    are the same for every route) it is the kernel's optimal distance divided by the search scale.

    Cases where the report drops a search tower next to a terminal (``tower: 0`` and ``smoothness: 0`` here) are left
    out: that post-processing is an open defect outside this change, and the identity then misses the dropped tower.
    """
    finder = make(objective, zigzag())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = finder.find_route()
    (call,) = kernel_spy
    kept = [t.cell_index for t in path.towers if t.tower_type != "terminal"]
    assert kept == call["towers"]  # the report kept every tower the search placed
    assert path.total_cost == pytest.approx(
        path.total_terrain_cost + path.total_tower_cost + path.total_angle_penalty_cost, rel=1e-12)
    assert path.total_cost - terminal_cost(path) == pytest.approx(call["dist"] / finder._search_scale, rel=1e-6)


# ------------------------------------------------------------------ an objective without a cost weight
def test_objective_without_cell_weight_prices_the_ground_at_zero(kernel_spy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder = ConstrainedPathFinder(
            dataset_source=zigzag(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
            search_space_buffer_m=100, profile=dict(PROFILE), objective={"smoothness": 2.0}, graph_api="cython")
    ours = [w for w in caught if "a missing weight is 0" in str(w.message)]
    assert len(ours) == 1 and issubclass(ours[0].category, PyorpsWarning)
    stack = [w for w in caught if "evaluates to 0 on every traversable cell" in str(w.message)]
    assert stack and all(issubclass(w.category, PyorpsWarning) for w in stack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = finder.find_route()
    (call,) = kernel_spy
    searched = call["raster"]
    assert set(np.unique(searched[searched != 65535]).tolist()) == {0}
    assert path.total_terrain_cost == 0.0
    # with free ground and paid turns the straight line wins
    assert path.total_angle_penalty_cost == 0.0
    assert path.max_turn_angle_deg == 0.0
    assert path.total_cost - terminal_cost(path) == pytest.approx(call["dist"], rel=1e-6)

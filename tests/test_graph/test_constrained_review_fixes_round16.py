"""Four review findings on the constrained objective change (round 16, review fixes).

1. The weighted turn and tower tables and the weights follow the CURRENT objective: ``set_objective()`` (also through
   ``find_route_ensemble``) used to leave the tables of the construction objective in place, and the refusals checked at
   construction were not checked by ``set_objective``.
2. A turn sharper than every tower angle type is reported with the type and cost the search charges (the last type),
   not as a free suspension tower.
3. A route with a step that is not a neighbourhood step is reported with a NaN turn penalty and ONE warning, it does not
   raise in the reporting stage.
4. (documentation only, see ``docs/source/api/constrained_path_finder.md``)
"""
import math
import warnings

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps.utils.constrained_path_algorithms as constrained_algorithms
from pyorps.core.exceptions import FeatureNotImplementedError, InvalidInputError
from pyorps.core.warning_types import PyorpsWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

CRS = "EPSG:25832"
N = 30
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)
SOURCE = (3.5, 15.5)
TARGET = (26.5, 15.5)

PROFILE = dict(
    name="test_ohl", description="test overhead line",
    soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
    angle_cost_function="linear", angle_cost_params={"scale": 10.0},
    min_span_m=3.0, max_span_m=8.0, span_bin_size_m=1.0,
    tower_cost_params={
        "terrain_cost_map": {0: 100, 100: 500, 500: 2000},
        "angle_types": {
            "suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
            "angle": {"max_angle_deg": 90.0, "base_cost": 200.0},
        },
    },
)
#: Angle types only up to 30 degrees, but the hard limit allows 90: a 45 degree turn is sharper than every type.
SHARP_PROFILE = dict(PROFILE, tower_cost_params={
    "terrain_cost_map": {0: 100, 100: 500, 500: 2000},
    "angle_types": {
        "suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
        "angle": {"max_angle_deg": 30.0, "base_cost": 200.0},
    },
})


def uniform():
    return np.full((N, N), 100, dtype=np.uint16)


def zigzag(expensive=400):
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
    rows = np.arange(N, dtype=np.float64)[:, None]
    r = (60 + 12 * rows + np.zeros((1, N))).astype(np.uint16)
    r[10:20, 12:16] = 900
    return r


def make(objective=None, raster=None, profile=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=zigzag() if raster is None else raster,
            crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
            search_space_buffer_m=100, profile=dict(profile or PROFILE),
            objective=objective, graph_api=kwargs.pop("graph_api", "cython"), **kwargs)


def run(finder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return finder.find_route()


def switch(finder, objective):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder.set_objective(objective)


def fingerprint(path):
    """Everything the report says about a route that does not depend on the quantisation of the terrain term."""
    return {
        "path": [int(i) for i in path.path_indices],
        "towers": [(t.cell_index, t.tower_type, t.terrain_cost, t.angle_cost, t.total_cost) for t in path.towers],
        "tower": path.total_tower_cost, "turns": path.total_angle_penalty_cost,
        "breakdown": {k: v for k, v in path.cost_breakdown.items() if k in ("towers", "angle_penalties")},
        "type_costs": dict(path.tower_type_costs),
    }


def same_report(a, b):
    """Equal routes and tower and turn numbers; the terrain term to the quantisation of the uint16 search raster
    (a finder that was switched with set_objective() quantises on the narrowed stack, a fresh one on the full one)."""
    assert fingerprint(a) == fingerprint(b)
    assert a.total_terrain_cost == pytest.approx(b.total_terrain_cost, rel=1e-3)
    assert a.total_cost == pytest.approx(b.total_cost, rel=1e-3)


@pytest.fixture
def kernel_spy(monkeypatch):
    original = constrained_algorithms.constrained_dijkstra_2d
    calls = []

    def spy(**kwargs):
        path, towers, dist = original(return_dist=1, **kwargs)
        calls.append({"dist": dist, "towers": [int(t) for t in towers]})
        return path, towers

    monkeypatch.setattr(constrained_algorithms, "constrained_dijkstra_2d", spy)
    return calls


# ------------------------------------------------------------------ 1. set_objective: tables follow the objective
OTHER = {"cost": 1.0, "tower": 2.0, "smoothness": 3.0}
HEAVY = {"cost": 1.0, "tower": 5.0}


def test_none_to_weights_to_other_weights_and_back():
    """no weights -> tower 5 -> tower 2 + smoothness 3 -> no weights: each run equals a fresh finder's run."""
    finder = make({"cost": 1.0})
    assert finder._w_tower == 1.0 and finder._w_smooth == 1.0
    legacy = run(finder)
    same_report(legacy, run(make(None)))

    switch(finder, HEAVY)  # none -> weights
    assert finder._w_tower == 5.0 and finder._w_smooth == 1.0
    heavy = run(finder)
    same_report(heavy, run(make(HEAVY)))
    # the reported tower cost is 5 times the profile's own cost of the SAME route (it was the unweighted cost)
    assert heavy.total_tower_cost == 5.0 * heavy.cost_breakdown["profile_towers"] > 0.0

    switch(finder, OTHER)  # weights -> other weights
    assert finder._w_tower == 2.0 and finder._w_smooth == 3.0
    other = run(finder)
    same_report(other, run(make(OTHER)))
    assert other.total_tower_cost == 2.0 * other.cost_breakdown["profile_towers"]
    assert other.total_angle_penalty_cost == pytest.approx(3.0 * other.cost_breakdown["profile_angle_penalties"])

    switch(finder, {"cost": 1.0})  # weights -> no weights
    assert finder._w_tower == 1.0 and finder._w_smooth == 1.0
    assert np.array_equal(finder._tower_terrain_costs, finder._profile_tower_terrain_costs)
    assert np.array_equal(finder._tower_angle_costs, finder._profile_tower_angle_costs)
    assert np.array_equal(finder._angle_cost_lut, finder._profile_angle_cost_lut)
    same_report(run(finder), legacy)


def test_weights_to_none_by_assignment_and_through_the_ensemble():
    """The tables are derived in find_route, so an objective changed without set_objective() cannot leave them stale."""
    finder = make(OTHER)
    run(finder)
    assert finder._w_tower == 2.0
    finder.objective = None  # what a caller may do; the ensemble's restore sets the objective directly too
    again = run(finder)
    assert finder._w_tower == 1.0 and finder._w_smooth == 1.0
    assert np.array_equal(finder._tower_terrain_costs, finder._profile_tower_terrain_costs)
    assert np.array_equal(finder._angle_cost_lut, finder._profile_angle_cost_lut)
    assert fingerprint(again) == fingerprint(run(make(None)))

    # find_route_ensemble: set_objective for every member, then a deferred restore of the caller's objective
    finder = make({"cost": 1.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble = finder.find_route_ensemble({"heavy": HEAVY, "plain": {"cost": 1.0}, "other": OTHER})
    by_name = {name: path for name, path in ensemble}
    same_report(by_name["heavy"], run(make(HEAVY)))
    same_report(by_name["plain"], run(make({"cost": 1.0})))
    same_report(by_name["other"], run(make(OTHER)))
    assert by_name["heavy"].total_tower_cost == 5.0 * by_name["heavy"].cost_breakdown["profile_towers"]
    same_report(run(finder), by_name["plain"])  # the restored objective, with its own tables


def test_set_objective_refuses_a_gradient_weight_like_the_constructor_and_keeps_the_old_objective():
    dem = np.zeros((N, N), dtype=np.float32)
    objective = {"cost": 1.0, "gradient": 1.0}
    for kwargs in ({}, {"dem": dem}):  # without a DEM the stack refuses first, with one the finder does
        with pytest.raises((InvalidInputError, FeatureNotImplementedError)) as built:
            make(objective, **kwargs)
        finder = make({"cost": 1.0}, **kwargs)
        before = run(finder)
        with pytest.raises(type(built.value)) as swapped:
            switch(finder, objective)
        assert str(swapped.value) == str(built.value)
        assert finder.objective.weights == {"cost": 1.0}
        assert finder._w_tower == 1.0
        after = run(finder)
        assert fingerprint(after) == fingerprint(before)
        assert after.total_terrain_cost == before.total_terrain_cost  # nothing was recombined


def test_set_objective_refuses_other_backends_and_clearance_with_the_constructor_errors(monkeypatch):
    with pytest.raises(FeatureNotImplementedError, match="'cython' backend") as built:
        make({"cost": 1.0}, graph_api="cython_parallel")
    finder = make({"cost": 1.0})
    monkeypatch.setattr(finder, "_constrained_backend", "cython_parallel")
    with pytest.raises(FeatureNotImplementedError) as swapped:
        switch(finder, {"cost": 2.0})
    assert str(swapped.value) == str(built.value)
    assert finder.objective.weights == {"cost": 1.0}

    monkeypatch.setattr(finder, "_constrained_backend", "cython")
    monkeypatch.setattr(type(finder._profile), "has_clearance_constraints", property(lambda self: True))
    with pytest.raises(FeatureNotImplementedError, match="clearance constraints") as swapped:
        switch(finder, {"cost": 2.0})
    assert finder.objective.weights == {"cost": 1.0}


def test_set_objective_warns_about_an_unpriced_ground():
    finder = make({"cost": 1.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.set_objective({"smoothness": 2.0})
    ours = [w for w in caught if "a missing weight is 0" in str(w.message)]
    assert len(ours) == 1 and issubclass(ours[0].category, PyorpsWarning)
    assert "that varies" not in str(ours[0].message)
    assert ours[0].filename == __file__  # attributed to the caller


def test_constant_positive_layer_is_priced_not_unpriced():
    """The condition is 'the objective is 0 on every traversable cell': a constant positive layer is priced."""
    finder = make({"cost": 1.0}, raster=uniform())
    assert finder._ground_unpriced() is False
    finder = make({"smoothness": 2.0}, raster=uniform())
    assert finder._ground_unpriced() is True


# ------------------------------------------------------------------ 2. a turn sharper than every type
@pytest.mark.parametrize("objective", [None, {"cost": 1.0}, {"cost": 1.0, "tower": 3.0},
                                       {"cost": 1.0, "length": 50.0, "tower": 0.5}])
def test_turn_beyond_every_angle_type_is_reported_like_the_search_charges_it(objective, kernel_spy):
    finder = make(objective, raster=ramp(), profile=SHARP_PROFILE)
    path = run(finder)
    (call,) = kernel_spy
    inner = [t for t in path.towers if t.tower_type != "terminal"]
    assert [t.cell_index for t in inner] == call["towers"]  # every search tower is in the report
    sharp = [t for t in inner if t.turn_angle_deg > 30.0]
    assert sharp, "the route must have a turn sharper than every type"
    w_tower = 1.0 if objective is None else objective.get("tower", 1.0)
    for tower in sharp:
        assert tower.tower_type == "angle"  # it was 'suspension' at cost 0
        assert tower.angle_cost == pytest.approx(200.0 * w_tower)
    terminal = sum(t.total_cost for t in path.towers if t.tower_type == "terminal")
    assert path.total_cost - terminal == pytest.approx(call["dist"] / finder._search_scale, rel=1e-6)


# ------------------------------------------------------------------ 3. an unmatched step is NaN, not an error
def test_turn_penalty_of_unmatched_step_is_nan_with_one_warning():
    finder = make(None)
    lut = finder._angle_cost_lut
    straight = np.arange(3, 9)  # steps (0, 1) only
    east = [tuple(s) for s in finder.steps.tolist()].index((0, 1))
    assert finder._turn_penalty_cost(straight, N, lut) == float(lut[east, east])
    path = np.array([3, 4, 7, 8])  # the step (0, 3) is not a neighbourhood step
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = finder._turn_penalty_cost(path, N, lut)
        both = finder._turn_penalty_costs(path, N, (lut, lut))
    assert math.isnan(value)
    assert len(both) == 2 and all(math.isnan(v) for v in both)
    ours = [w for w in caught if issubclass(w.category, PyorpsWarning)]
    assert len(ours) == 2  # one per call, one per route
    message = str(ours[0].message)
    assert "turn penalties of this route could not be recomputed" in message
    assert "(0, 3)" in message and "NaN" in message
    assert ours[0].filename == __file__  # attributed to the caller, not to the library


@pytest.mark.parametrize("objective", [None, {"cost": 1.0, "smoothness": 2.0}])
def test_route_with_unmatched_step_is_reported_with_nan_totals(objective, monkeypatch):
    finder = make(objective, raster=uniform())
    # a path with a jump of three columns, which no neighbourhood step of 'r2' covers
    cells = np.array([15 * N + c for c in (3, 4, 5, 6, 9, 10, 11, 12)], dtype=np.int64)
    monkeypatch.setattr(finder, "_find_route_coupled",
                        lambda raster, sr, sc, tr, tc, backend: (cells, np.array([cells[3]], dtype=np.int64)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = finder.find_route()
    recompute = [w for w in caught if "could not be recomputed" in str(w.message)]
    assert len(recompute) == 1 and issubclass(recompute[0].category, PyorpsWarning)
    assert math.isnan(path.total_angle_penalty_cost)
    assert math.isnan(path.total_cost)
    assert math.isnan(path.cost_breakdown["angle_penalties"])
    if objective is not None:
        assert math.isnan(path.cost_breakdown["profile_angle_penalties"])
    assert not math.isnan(path.total_terrain_cost) and not math.isnan(path.total_tower_cost)
    assert path.n_towers >= 2
    # the reporting code downstream copes with NaN
    assert "cost=nan" in str(path)
    assert "nan" in path.analyze().lower()
    assert math.isnan(path.to_geodataframe_dict()["total_angle_penalty_cost"])

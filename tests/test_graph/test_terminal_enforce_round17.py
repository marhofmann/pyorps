"""``terminal_span_rule="enforce"`` through the finder (round 17, package P3).

The kernels learned ``target_min_span`` in package P2; here the finder hands it to them (the value is
``min_span_m``) and takes exact span bins for the run. Tested end to end:

- the uniform 1 x 24 corridor of the kernel agent: with coarse bins the search finds NO route with a last span of at
  least 3 m, with exact bins it finds the optimum of cost 2750 (and the finder uses exact bins itself);
- a brute-force reference (the exhaustive programme of ``test_constrained_kernel_ends_round17.py``) on random
  one-row corridors, with and without a clearance check, on the Dijkstra and the delta-stepping backend;
- the cost identity on the uniform raster (``"enforce"`` gives every span in ``[min_span_m, max_span_m)`` at the cost
  of ``"keep"``, which ties), random rasters, the L-shaped corridor whose last leg is too short, the default empty
  result, the bins, their memory bound and the GPU refusal of the new settings.
"""

import math
import warnings

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps.utils.constrained_path_algorithms as algorithms
from pyorps.core.exceptions import AlgorithmNotImplementedError, InvalidInputError, NoPathFoundError
from pyorps.core.warning_types import SpanRuleWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
from test_graph.test_constrained_kernel_ends_round17 import FORBIDDEN, Corridor, random_corridor

CRS = "EPSG:25832"
N = 30
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)
SOURCE = (3.5, 15.5)
TARGET = (26.5, 15.5)
FLAT_TOWER_MAP = {0: 100, 500: 100}

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


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # the delta-stepping kernels read OMP_NUM_THREADS per call: one thread keeps ties deterministic
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def uniform():
    return np.full((N, N), 100, dtype=np.uint16)


def noise(seed):
    rng = np.random.default_rng(seed)
    r = rng.integers(20, 400, (N, N)).astype(np.uint16)
    r[rng.random((N, N)) < 0.04] = 65535
    r[14, 3] = 100
    r[14, 26] = 100
    return r


def make(raster, rule="enforce", source=SOURCE, target=TARGET, backend="cython", neighborhood="r2", **extra):
    profile = {**PROFILE, "terminal_span_rule": rule, **extra}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=raster.copy(), crs=CRS, transform=TRANSFORM, source_coords=source, target_coords=target,
            search_space_buffer_m=100, profile=profile, graph_api=backend, neighborhood_str=neighborhood)


def corridor_finder(case, rule="keep", heights=None, backend="cython", tower_map=None, cost_per_increment=0.0):
    """A finder whose search is the one-row corridor ``case`` of the kernel tests (same raster, DEM, obstacles and
    cost tables), so that ``case.oracle`` is its reference. ``heights`` None = no clearance check. The default rule
    is "keep": the reported line is exactly the searched one (no tower removed), so it can be checked span by span."""
    rows = case.raster.shape[0]
    tower_map = tower_map or {0: 40, 200: 300}
    profile = dict(
        name="corridor", description="one open row", soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
        angle_cost_function="linear", angle_cost_params={"scale": 10.0},
        min_span_m=case.min_span, max_span_m=case.max_span, span_bin_size_m=1.0, terminal_span_rule=rule,
        tower_cost_params={"terrain_cost_map": tower_map,
                           "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                                           "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}})
    extra = {}
    if heights is not None:
        # the kernel tests switch the gradient penalty off; the profile does the same
        profile.update(conductor_weight_per_m=case.w, conductor_tension_n=case.tension, min_clearance_m=case.mc,
                       max_gradient_percent=1.0e6, gradient_cost_function="linear",
                       gradient_cost_params={"scale": 0.0})
        if len(heights) == 1:
            profile["tower_height_m"] = float(heights[0])
        else:
            profile.update(tower_heights_m=[float(h) for h in heights],
                           tower_height_cost_per_increment=cost_per_increment)
        extra["dem"] = case.dem.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=case.raster.copy(), crs=CRS, transform=from_origin(0.0, float(rows), 1.0, 1.0),
            source_coords=(0.5, rows - case.row - 0.5), target_coords=(case.n - 0.5, rows - case.row - 0.5),
            search_space_buffer_m=100, profile=profile, graph_api=backend, neighborhood_str="r1", **extra)
    if heights is not None:
        finder._obstacle_data = case.obs.copy()      # DSM minus DEM, as the finder would have loaded it
    return finder


def run(finder, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return finder.find_route(**kwargs)


def terminal_cost(path):
    """Foundation and ``terminal_tower_cost`` of the two terminal towers (the search charges neither)."""
    return sum(t.terrain_cost + t.angle_cost for t in path.towers if t.tower_type == "terminal")


def search_cost(path):
    """The reported total in the kernel's terms: minus the terminals, plus the towers the end-span rule removed."""
    return path.total_cost - terminal_cost(path) + path.cost_breakdown.get("removed_towers", 0.0)


@pytest.fixture
def kernel_spy(monkeypatch):
    """Record the keyword arguments and the optimal distance of every constrained Dijkstra call."""
    original = algorithms.constrained_dijkstra_2d
    calls = []

    def spy(**kwargs):
        path, towers, dist = original(return_dist=1, **kwargs)
        calls.append({"kwargs": kwargs, "dist": dist, "towers": [int(t) for t in towers]})
        return path, towers

    monkeypatch.setattr(algorithms, "constrained_dijkstra_2d", spy)
    return calls


# ------------------------------------------------------------------------------- the planner's 1 x 24 corridor
def flat_corridor():
    return Corridor([100] * 24, min_span=3, max_span=8, tower_map=((0, 100.0), (500, 100.0)))


def test_enforce_finds_the_route_of_cost_2750_on_the_1x24_corridor_through_the_finder():
    case = flat_corridor()
    path = run(corridor_finder(case, "enforce", tower_map=FLAT_TOWER_MAP))
    assert search_cost(path) == pytest.approx(2750.0)
    assert case.oracle({"target_min_span": 3.0}, False) == pytest.approx(2750.0)
    assert len(path.towers) - 2 == 3 and path.removed_towers == [] and path.span_notes == []
    assert all(3.0 - 1e-9 <= s < 8.0 for s in path.spans), path.spans
    assert path.terminal_span_rule == "enforce"
    # exact bins: under the shortest step (1 m), says so in the result
    assert 0.9 < path.span_bin_size_m < 1.0
    (note,) = path.search_notes
    assert "terminal_span_rule='enforce'" in note and "exact span bins" in note and "min_span_m 3 m" in note


def test_the_default_rule_keeps_todays_bins_and_todays_short_last_span_on_the_1x24_corridor():
    case = flat_corridor()
    finder = corridor_finder(case, "remove", tower_map=FLAT_TOWER_MAP)
    path = run(finder)
    assert path.span_bin_size_m == 3.0 == finder._effective_span_bin_size       # min_span_m, as before
    assert path.search_notes == []
    # today's result: the search tower 2 m from the target is dropped, the merged last span is 9 m
    assert path.spans == pytest.approx([7.0, 7.0, 9.0])
    assert len(path.removed_towers) == 1


def test_coarse_bins_are_refused_with_the_rule_exact_bins_find_the_route(monkeypatch):
    """The reason for exact bins (kernel agent, uniform 1 x 24 corridor): the bins of ``min_span_m`` merge the span
    states the end rule needs, and the search used to find NO route with them. Since the kernel review the kernels
    refuse such bins when a line-end option is on, and the finder reports the refusal as an ``InvalidInputError``
    (never a raw ``ValueError``); the finder's own exact bins find the route."""
    case = flat_corridor()
    exact = corridor_finder(case, "enforce", tower_map=FLAT_TOWER_MAP)
    assert len(run(exact).path_indices) > 0
    coarse = corridor_finder(case, "enforce", tower_map=FLAT_TOWER_MAP)
    monkeypatch.setattr(coarse, "_exact_span_bins", lambda raster, dem: (coarse._effective_n_span_bins,
                                                                         coarse._effective_span_bin_size))
    with pytest.raises(InvalidInputError, match="too coarse") as caught:
        coarse.find_route()
    assert type(caught.value) is InvalidInputError and "refused the span bins" in str(caught.value)


# ------------------------------------------------------------------------------------ the brute-force reference
def plain_corridor(seed, n=28, min_span=3, max_span=8):
    rng = np.random.default_rng(2000 + seed)
    return Corridor(rng.integers(1, 200, size=n), min_span=min_span, max_span=max_span)


@pytest.mark.parametrize("backend", ["cython", "cython_parallel"])
@pytest.mark.parametrize("seed", range(6))
def test_enforce_is_optimal_on_random_corridors_without_clearance(seed, backend):
    case = plain_corridor(seed)
    path = run(corridor_finder(case, "enforce", backend=backend))
    ref = case.oracle({"target_min_span": 3.0}, False)
    assert math.isfinite(ref)
    assert search_cost(path) == pytest.approx(ref, rel=1e-6)          # the oracle's tables are float32
    assert all(3.0 - 1e-9 <= s < 8.0 for s in path.spans), path.spans
    assert path.removed_towers == []


@pytest.mark.parametrize("seed", range(4))
def test_enforce_is_optimal_on_random_corridors_with_a_clearance_check(seed):
    case = random_corridor(seed, heights=(20.0,), premiums=(0.0,))
    path = run(corridor_finder(case, "enforce", heights=(20.0,)))
    ref = case.oracle({"target_min_span": 3.0}, True)
    assert math.isfinite(ref)
    assert search_cost(path) == pytest.approx(ref, rel=1e-6)          # the height kernels hold float32 distances
    assert all(3.0 - 1e-9 <= s < 8.0 for s in path.spans), path.spans
    assert path.removed_towers == []


@pytest.mark.parametrize("seed", range(4))
def test_enforce_costs_at_least_the_unrestricted_optimum_and_coarse_bins_can_cost_more_than_enforce(seed):
    """The rule restricts the search, so its exact optimum is at least the unrestricted exact optimum. The default
    run's coarse bins (``min_span_m``) lose the optimum on some corridors (plan A6.3): there the "keep" route costs
    more than the exact "enforce" route, which is why the default bins are not trusted for the new rule."""
    case = plain_corridor(seed)
    keep = run(corridor_finder(case, "keep"))
    enforce = run(corridor_finder(case, "enforce"))
    free = case.oracle({}, False)
    assert search_cost(enforce) >= free * (1 - 1e-6)
    assert search_cost(keep) >= free * (1 - 1e-6)           # also with coarse bins: never below the exact optimum


# ----------------------------------------------------------------------------- the uniform raster, the identity
def test_enforce_on_the_uniform_raster_has_every_span_in_range_at_the_cost_of_keep(kernel_spy):
    keep = run(make(uniform(), "keep"))
    assert keep.spans == pytest.approx([7.0, 7.0, 7.0, 2.0])                 # the short end span the rule forbids
    kernel_spy.clear()
    finder = make(uniform(), "enforce")
    path = run(finder)
    (call,) = kernel_spy
    assert all(3.0 - 1e-9 <= s < 8.0 for s in path.spans), path.spans
    assert path.removed_towers == [] and path.span_notes == []
    assert path.total_cost == pytest.approx(keep.total_cost)                    # a tie on the uniform raster
    assert search_cost(path) == pytest.approx(call["dist"] / finder._search_scale, rel=1e-6)
    assert call["kwargs"]["target_min_span"] == 3.0
    assert call["kwargs"]["span_bin_size"] < 1.0


@pytest.mark.parametrize("objective", [{"cost": 1.0, "tower": 5.0}, {"cost": 1.0, "smoothness": 0.5}])
def test_enforce_under_an_objective_keeps_the_identity_and_the_spans(objective, kernel_spy):
    """The objective's per-cell tower costs and search scale go through the same kernel call as ``target_min_span``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=uniform(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
            search_space_buffer_m=100, profile={**PROFILE, "terminal_span_rule": "enforce"}, graph_api="cython",
            objective=objective)
        path = finder.find_route()
    (call,) = kernel_spy
    assert call["kwargs"]["target_min_span"] == 3.0
    assert search_cost(path) == pytest.approx(call["dist"] / finder._search_scale, rel=1e-6)
    assert all(3.0 - 1e-9 <= s < 8.0 for s in path.spans), path.spans


def test_enforce_is_the_same_on_the_delta_stepping_backend():
    dijkstra = run(make(uniform(), "enforce"))
    parallel = run(make(uniform(), "enforce", backend="cython_parallel"))
    assert parallel.total_cost == pytest.approx(dijkstra.total_cost)
    assert all(3.0 - 1e-9 <= s < 8.0 for s in parallel.spans)


@pytest.mark.parametrize("seed", range(6))
def test_span_rules_and_the_identity_hold_on_random_rasters(seed, kernel_spy):
    finder = make(noise(seed), "enforce")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            path = finder.find_route()
        except NoPathFoundError:
            pytest.skip("no route satisfies the rule on this raster")
    assert not [w for w in caught if issubclass(w.category, SpanRuleWarning)]
    (call,) = kernel_spy
    assert search_cost(path) == pytest.approx(call["dist"] / finder._search_scale, rel=1e-6)
    spans = path.spans
    assert all(3.0 - 1e-6 <= s < 8.0 + 1e-6 for s in spans), spans            # every span, the end spans included
    path_arr = np.asarray(path.path_indices)
    steps = np.diff(path_arr)
    at_tower = {t.cell_index for t in path.towers}
    bends = [int(path_arr[k]) for k in range(1, len(steps)) if steps[k] != steps[k - 1]]
    assert all(cell in at_tower for cell in bends)                            # a tower at every bend
    assert path.removed_towers == []


# --------------------------------------------------------------------------------------- no feasible line end
def l_shaped_raster():
    r = np.full((8, 10), FORBIDDEN, np.uint16)
    r[1, 1:8] = 50
    r[1:4, 7] = 50
    return r


def l_shaped_finder(rule):
    profile = {**PROFILE, "terminal_span_rule": rule, "min_span_m": 3.0, "max_span_m": 8.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=l_shaped_raster(), crs=CRS, transform=from_origin(0.0, 8.0, 1.0, 1.0),
            source_coords=(1.5, 6.5), target_coords=(7.5, 4.5), search_space_buffer_m=100, profile=profile,
            graph_api="cython", neighborhood_str="r1")


def test_enforce_without_a_feasible_line_end_raises_with_the_rule_and_the_fix_in_the_message():
    """The last leg (2 m) is shorter than min_span_m (3 m) and the bend needs a tower: no valid line end."""
    with pytest.raises(NoPathFoundError) as caught:
        l_shaped_finder("enforce").find_route()
    text = str(caught.value)
    assert "terminal_span_rule='enforce'" in text and "min_span_m 3" in text
    assert "'keep'" in text and "'remove_if_valid'" in text and "no route exists" in text


@pytest.mark.parametrize("rule", ["remove", "keep", "remove_if_valid"])
def test_the_other_rules_find_a_route_on_the_same_raster(rule):
    path = run(l_shaped_finder(rule))
    assert len(path.path_indices) > 0 and path.towers


def wall_raster():
    r = uniform()
    r[:, 15] = FORBIDDEN
    return r


def test_a_default_run_with_an_empty_search_result_returns_the_empty_result_as_before():
    path = run(make(wall_raster(), "remove"))
    assert len(path.path_indices) == 0 and path.towers == [] and path.path_geometry is None


@pytest.mark.parametrize("rule", ["keep", "remove_if_valid"])
def test_the_other_rules_also_return_the_empty_result_on_a_walled_raster(rule):
    assert len(run(make(wall_raster(), rule)).path_indices) == 0


def test_an_empty_enforce_run_raises_and_names_the_rule():
    with pytest.raises(NoPathFoundError, match="terminal_span_rule='enforce'"):
        run(make(wall_raster(), "enforce"))


# --------------------------------------------------------------------------------- default calls stay as they were
def test_a_default_run_passes_no_new_argument_to_the_kernel_and_keeps_its_bins(kernel_spy):
    finder = make(uniform(), "remove")
    path = run(finder)
    (call,) = kernel_spy
    assert "target_min_span" not in call["kwargs"]
    assert call["kwargs"]["span_bin_size"] == 3.0 and call["kwargs"]["n_span_bins"] == 3
    assert path.span_bin_size_m == 3.0 and path.search_notes == []
    assert path.source_connection_height_m is None and path.target_connection_height_m is None


def test_enforce_without_span_limits_has_no_effect_and_no_warning():
    no_spans = dict(name="road", description="road", soft_angle_limit_deg=15.0, hard_angle_limit_deg=90.0,
                    angle_cost_function="linear", angle_cost_params={"scale": 50})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=uniform(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
            search_space_buffer_m=100, profile={**no_spans, "terminal_span_rule": "enforce"}, graph_api="cython")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = finder.find_route()
    assert len(path.path_indices) > 0 and path.search_notes == []


# -------------------------------------------------------------------------------------------------- exact bins
@pytest.mark.parametrize("neighborhood,cell", [("r1", 1.0), ("r2", 1.0), ("r2", 0.7), ("r3", 2.5)])
def test_exact_bins_tell_every_span_of_a_direction_apart(neighborhood, cell):
    """Within a cell and direction the spans are multiples k * d of that direction's step d >= the shortest step; the
    bin must be at most the shortest step and put the span of k steps (running float32 sum, as the kernel adds it)
    into bin k for every k that fits below max_span_m."""
    transform = from_origin(0.0, N * cell, cell, cell)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=uniform(), crs=CRS, transform=transform, source_coords=(3.5 * cell, 15.5 * cell),
            target_coords=(26.5 * cell, 15.5 * cell), search_space_buffer_m=100 * cell,
            profile={**PROFILE, "min_span_m": 3.0 * cell, "max_span_m": 8.0 * cell, "terminal_span_rule": "enforce"},
            graph_api="cython", neighborhood_str=neighborhood)
    n_bins, bin_size = finder._exact_span_bins(uniform(), None)
    shortest = float(np.min(finder._step_distances))
    assert bin_size <= shortest and n_bins * bin_size >= 8.0 * cell
    bin_f32 = np.float32(bin_size)
    for d in np.unique(finder._step_distances.astype(np.float32)):
        span, bins = 0.0, []
        while span + float(d) < 8.0 * cell:
            span += float(d)
            bins.append(int(span / float(bin_f32)))
        assert len(set(bins)) == len(bins), (d, bins)           # no two spans share a bin
        assert max(bins, default=0) < n_bins


def test_the_state_space_of_exact_bins_has_a_bound_and_the_error_gives_the_numbers(monkeypatch):
    finder = make(uniform(), "enforce")
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_STATES", 1000)
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert "terminal_span_rule='enforce'" in text and f"{N * N * len(finder.steps) * 9:,}" in text
    assert "EXACT_BINS_MAX_STATES" in text and "coarser raster" in text and "search_space_buffer_m" in text
    # a run that keeps the coarse bins is not subject to the bound
    assert len(run(make(uniform(), "keep")).path_indices) > 0


def test_the_bin_index_of_the_kernels_has_a_limit_too(monkeypatch):
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_COUNT", 5)
    with pytest.raises(InvalidInputError, match=r"9 span bins .* at most 5"):
        make(uniform(), "enforce").find_route()


# ------------------------------------------------------------------------------------ the GPU backends refuse it
@pytest.mark.parametrize("backend", ["raster_gpu", "raster_gpu_v4"])
def test_enforce_on_a_gpu_backend_is_refused_before_any_gpu_code(backend, monkeypatch):
    finder = make(uniform(), "enforce")
    monkeypatch.setattr(finder, "_constrained_backend", backend)
    with pytest.raises(AlgorithmNotImplementedError, match=r"target_min_span=3 \(terminal_span_rule='enforce'"):
        finder.find_route()

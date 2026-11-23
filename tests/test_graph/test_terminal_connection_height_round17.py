"""Connection-point heights at the line ends, height premiums and the honest warnings (round 17, package P3).

``ConstrainedPathFinder(..., source_connection_height_m=, target_connection_height_m=)`` (and the same two keywords of
``find_route``) give the height at which the conductor leaves the structure at an end. With a DEM and a profile that has
clearance data the search uses them for the first and the last span (kernel arguments ``source_attach_height`` and
``target_attach_height``, package P2) and takes exact span bins; the height is also the reported height of that end's
terminal tower. Tested here:

- validation, the per-call override and the warning of a height that cannot act (no DEM);
- optimality against the exhaustive reference of the kernel tests, for a source height, a target height and both;
- the independent Python clearance check (``pyorps.graph.span_clearance``) of EVERY span of a reported line, with the
  reported heights, for fixed and variable tower heights; the closed form of the conductor over flat ground;
- an end span the search does not check (no target height) warns, with the height it is checked and the line is clear;
- the height premiums the search charges (``Tower.height_premium``, ``cost_breakdown["height_premiums"]``), the cost
  identity with them, and ``remove_if_valid`` with several tower heights;
- the warnings for non-square cells and for a constrained GPU solver on a cell size other than 1.
"""

import math
import warnings

import numpy as np
import pytest
from rasterio import Affine
from rasterio.transform import from_origin

import pyorps.utils.constrained_path_algorithms as algorithms
from pyorps.core.exceptions import InvalidInputError, InvalidTypeError, NoPathFoundError
from pyorps.core.warning_types import PyorpsWarning, SpanRuleWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
from pyorps.graph.span_clearance import path_span_clearance, span_clearance
from test_graph.test_constrained_kernel_ends_round17 import Corridor, random_corridor
from test_graph.test_terminal_enforce_round17 import (
    CRS,
    FLAT_TOWER_MAP,
    PROFILE,
    TRANSFORM,
    corridor_finder,
    run,
    search_cost,
    uniform,
)


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def failing_spans(case, path):
    """Indices of the spans of the reported line that fail the independent Python check (heights as reported)."""
    path_arr = np.asarray(path.path_indices, dtype=np.int64)
    rows, cols = path_arr // case.n, path_arr % case.n
    where = [int(np.flatnonzero(path_arr == t.cell_index)[0]) for t in path.towers]
    assert where == sorted(where)
    failing = []
    pairs = zip(zip(where[:-1], path.towers[:-1]), zip(where[1:], path.towers[1:]))
    for k, ((pos_a, tower_a), (pos_b, tower_b)) in enumerate(pairs):
        result = path_span_clearance(
            case.dem, case.obs, rows, cols, pos_a, pos_b, 1.0, tower_a.height_m, tower_b.height_m,
            conductor_weight_per_m=case.w, conductor_tension_n=case.tension, min_clearance_m=case.mc)
        if result is not None and not result.ok:
            failing.append(k)
    return failing


def all_spans_clear(case, path):
    """True when every span of the reported line passes the independent check."""
    return not failing_spans(case, path)


def assert_line_clear(case, path, label=""):
    """Every span clears, except possibly the last one when no target height was given: the search never checks that
    span, and then the result must say so (``span_notes``)."""
    failing = failing_spans(case, path)
    last = len(path.towers) - 2
    if path.target_connection_height_m is None and failing == [last]:
        assert any("last span" in note and "clearance" in note for note in path.span_notes), label
    else:
        assert failing == [], (label, failing)


# ------------------------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("bad,error", [(True, InvalidTypeError), ("12", InvalidTypeError), ([12.0], InvalidTypeError),
                                       (0, InvalidInputError), (-3.0, InvalidInputError),
                                       (float("nan"), InvalidInputError), (float("inf"), InvalidInputError),
                                       (1.0e40, InvalidInputError)])
@pytest.mark.parametrize("name", ["source_connection_height_m", "target_connection_height_m"])
def test_a_wrong_connection_height_is_refused_at_construction_and_per_call(name, bad, error):
    case = random_corridor(0, heights=(20.0,), premiums=(0.0,))
    rows = case.raster.shape[0]
    with pytest.raises(error, match=name):
        ConstrainedPathFinder(
            dataset_source=case.raster.copy(), crs=CRS, transform=from_origin(0.0, float(rows), 1.0, 1.0),
            source_coords=(0.5, 1.5), target_coords=(27.5, 1.5), search_space_buffer_m=100, profile=dict(PROFILE),
            graph_api="cython", **{name: bad})
    finder = corridor_finder(case, heights=(20.0,))
    with pytest.raises(error, match=name):
        finder.find_route(**{name: bad})


def test_numbers_of_any_kind_are_accepted_and_the_attributes_are_settable():
    case = random_corridor(0, heights=(20.0,), premiums=(0.0,))
    finder = corridor_finder(case, heights=(20.0,))
    assert finder.source_connection_height_m is None and finder.target_connection_height_m is None
    finder.source_connection_height_m = np.float32(12.5)
    finder.target_connection_height_m = 9
    assert finder.source_connection_height_m == 12.5 and finder.target_connection_height_m == 9.0
    assert isinstance(finder.target_connection_height_m, float)
    with pytest.raises(InvalidInputError):
        finder.source_connection_height_m = -1.0
    finder.source_connection_height_m = None
    assert finder.source_connection_height_m is None


def test_find_route_override_none_means_not_set_and_leaving_it_out_means_the_finders_setting():
    # a corridor on which the default run (coarse bins) finds a route at all: on some the merged states lose it
    seed = next(s for s in range(10) if len(run(corridor_finder(
        random_corridor(s, heights=(20.0,), premiums=(0.0,)), heights=(20.0,))).path_indices) > 0)
    case = random_corridor(seed, heights=(20.0,), premiums=(0.0,))
    finder = corridor_finder(case, heights=(20.0,))
    finder.source_connection_height_m = 9.5
    kept = run(finder)
    assert kept.source_connection_height_m == 9.5 and kept.target_connection_height_m is None
    assert kept.towers[0].height_m == 9.5
    off = run(finder, source_connection_height_m=None)
    assert off.source_connection_height_m is None and off.towers[0].height_m == 20.0
    over = run(finder, source_connection_height_m=30.0, target_connection_height_m=9.5)
    assert (over.source_connection_height_m, over.target_connection_height_m) == (30.0, 9.5)
    assert (over.towers[0].height_m, over.towers[-1].height_m) == (30.0, 9.5)
    # the override is for the call: the finder's own setting is back afterwards
    assert finder.source_connection_height_m == 9.5 and finder.target_connection_height_m is None
    assert run(finder).source_connection_height_m == 9.5


def test_find_route_names_the_connection_heights_among_its_accepted_keywords():
    finder = corridor_finder(random_corridor(0, heights=(20.0,), premiums=(0.0,)), heights=(20.0,))
    with pytest.raises(TypeError, match=r"'source_connection_height_m'.*'target_connection_height_m'"):
        finder.find_route(algorithm="dijkstra")


# ---------------------------------------------------------------------- a height that cannot act is said so
def test_a_height_without_a_dem_changes_nothing_warns_once_and_is_still_reported():
    finder = ConstrainedPathFinder(
        dataset_source=uniform(), crs=CRS, transform=TRANSFORM, source_coords=(3.5, 15.5),
        target_coords=(26.5, 15.5), search_space_buffer_m=100, profile=dict(PROFILE), graph_api="cython",
        target_connection_height_m=15.0)
    plain = run(ConstrainedPathFinder(
        dataset_source=uniform(), crs=CRS, transform=TRANSFORM, source_coords=(3.5, 15.5),
        target_coords=(26.5, 15.5), search_space_buffer_m=100, profile=dict(PROFILE), graph_api="cython"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = finder.find_route()
        finder.find_route()                                   # the same condition: no second warning
    unused = [w for w in caught if issubclass(w.category, PyorpsWarning) and "has no effect" in str(w.message)]
    assert len(unused) == 1
    text = str(unused[0].message)
    assert "target_connection_height_m=15 m" in text and "there is no DEM" in text
    assert unused[0].filename == __file__                      # the caller's line, not a pyorps module
    assert path.towers[-1].height_m == 15.0 and path.target_connection_height_m == 15.0
    assert np.array_equal(path.path_indices, plain.path_indices)       # the route is the default route
    assert path.span_bin_size_m == 3.0 and path.search_notes == []      # no exact bins: the search is unchanged
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_route(source_connection_height_m=8.0)      # a new setting, a new warning
    assert len([w for w in caught if "has no effect" in str(w.message)]) == 1


def test_the_no_effect_warning_names_a_profile_without_clearance_data():
    case = random_corridor(0, heights=(20.0,), premiums=(0.0,))
    finder = corridor_finder(case)                            # a DEM-less finder: heights=None
    with pytest.warns(PyorpsWarning, match=r"has no effect.*there is no DEM and the profile lacks tower height"):
        finder.find_route(source_connection_height_m=12.0)


# ----------------------------------------------------------------------------------------------- optimality
OPTION_SETS = {
    "src": ({"source_attach_height": 9.5}, {"source_connection_height_m": 9.5}),
    "tgt": ({"target_attach_height": 7.0}, {"target_connection_height_m": 7.0}),
    "both": ({"source_attach_height": 30.0, "target_attach_height": 9.5},
             {"source_connection_height_m": 30.0, "target_connection_height_m": 9.5}),
    "all": ({"target_min_span": 3.0, "source_attach_height": 16.5, "target_attach_height": 30.0},
            {"source_connection_height_m": 16.5, "target_connection_height_m": 30.0}),
}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("opt", sorted(OPTION_SETS))
def test_the_finder_is_optimal_with_connection_heights_against_the_exhaustive_reference(seed, opt):
    case = random_corridor(seed, heights=(20.0,), premiums=(0.0,))
    kernel_opts, finder_kwargs = OPTION_SETS[opt]
    rule = "enforce" if "target_min_span" in kernel_opts else "keep"
    finder = corridor_finder(case, rule, heights=(20.0,))
    ref = case.oracle(kernel_opts, True)
    if not math.isfinite(ref):
        with pytest.raises(NoPathFoundError, match="connection_height_m"):
            finder.find_route(**finder_kwargs)
        return
    path = run(finder, **finder_kwargs)
    assert search_cost(path) == pytest.approx(ref, rel=1e-6), (seed, opt)
    assert_line_clear(case, path, (seed, opt))
    assert path.removed_towers == []
    for key, value in finder_kwargs.items():
        assert getattr(path, key) == value
    assert path.towers[0].height_m == finder_kwargs.get("source_connection_height_m", 20.0)
    assert path.towers[-1].height_m == finder_kwargs.get("target_connection_height_m", 20.0)
    assert path.span_bin_size_m < 1.0                           # exact bins, as the result says
    assert any("exact span bins" in note and "connection height" in note for note in path.search_notes)


def test_a_connection_height_given_at_construction_is_the_same_as_given_to_find_route():
    case = random_corridor(1, heights=(20.0,), premiums=(0.0,))
    by_call = run(corridor_finder(case, heights=(20.0,)), source_connection_height_m=9.5,
                  target_connection_height_m=9.5)
    finder = corridor_finder(case, heights=(20.0,))
    finder.source_connection_height_m = finder.target_connection_height_m = 9.5
    by_attribute = run(finder)
    assert np.array_equal(by_call.path_indices, by_attribute.path_indices)
    assert by_call.total_cost == by_attribute.total_cost


def test_a_target_height_no_tower_can_clear_raises_and_names_the_height():
    """An obstacle of 8 m at the target cell needs 13 m of conductor there (minimum clearance 5 m); a 10 m end
    structure cannot give it, whatever the last tower. (A last span of one step is never checked, by the search's own
    rule, so with the default end-span rule a tower next to the target gets round it; "enforce" keeps the end span at
    least 3 steps long and the height is refused, with both settings named.)"""
    case = Corridor([100] * 30, min_span=3, max_span=8, obstacles={29: 8.0}, heights=(12.0,), premiums=(0.0,),
                    tower_map=((0, 100.0), (500, 100.0)))
    with pytest.raises(NoPathFoundError) as caught:
        run(corridor_finder(case, "enforce", heights=(12.0,), tower_map=FLAT_TOWER_MAP),
            target_connection_height_m=10.0)
    text = str(caught.value)
    assert "target_connection_height_m 10 m" in text and "terminal_span_rule='enforce'" in text
    with pytest.raises(NoPathFoundError, match="target_connection_height_m 10 m"):
        # the same height with a 5 m obstacle two steps before the target: no tower of 12 m can clear it either
        case2 = Corridor([100] * 30, min_span=3, max_span=8, obstacles={27: 8.0, 28: 8.0, 29: 8.0}, heights=(12.0,),
                         premiums=(0.0,), tower_map=((0, 100.0), (500, 100.0)))
        run(corridor_finder(case2, "enforce", heights=(12.0,), tower_map=FLAT_TOWER_MAP),
            target_connection_height_m=10.0)


def test_a_last_span_the_search_does_not_check_warns_and_a_target_height_closes_it():
    """The same 8 m obstacle at the target cell: without a target height the search never looks at the last span, the
    reported end tower has the line height (12 m) and the span fails the independent check: one SpanRuleWarning. With a
    20 m connection height the search checks the span, the line is clear and nothing warns."""
    case = Corridor([100] * 30, min_span=3, max_span=8, obstacles={29: 8.0}, heights=(12.0,), premiums=(0.0,),
                    tower_map=((0, 100.0), (500, 100.0)))
    finder = corridor_finder(case, "enforce", heights=(12.0,), tower_map=FLAT_TOWER_MAP)    # end span >= 3 steps
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unchecked = finder.find_route()
    warned = [w for w in caught if issubclass(w.category, SpanRuleWarning)]
    assert len(warned) == 1
    text = str(warned[0].message)
    assert "the last span from" in text and "fails the clearance check" in text and "heights 12 m and 12 m" in text
    assert "The search does not check the last span" in text
    assert not all_spans_clear(case, unchecked)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpanRuleWarning)
        checked = finder.find_route(target_connection_height_m=20.0)
    assert checked.towers[-1].height_m == 20.0 and all_spans_clear(case, checked)
    assert checked.span_notes == []


@pytest.mark.parametrize("seed", range(4))
def test_a_source_height_above_the_tower_never_costs_more_and_a_low_one_never_less(seed):
    """Raising the attachment at the source can only help the first span (the chord rises); lowering it below the
    tower height can only hurt. Both against the exact optimum with the tower as the source structure. (One tower
    height: with several, the finder's base-height first pass is not exact, plan A6.4, and the comparison would not
    be about the heights.)"""
    case = random_corridor(seed, heights=(20.0,), premiums=(0.0,))
    free = case.oracle({"source_attach_height": 20.0}, True)
    high = run(corridor_finder(case, heights=(20.0,)), source_connection_height_m=40.0)
    assert search_cost(high) <= free * (1 + 1e-6)
    assert high.towers[0].height_m == 40.0
    try:
        low = run(corridor_finder(case, heights=(20.0,)), source_connection_height_m=6.0)
    except NoPathFoundError:
        return                      # no route at all from a 6 m source: infinitely dearer than the reference
    assert search_cost(low) >= free * (1 - 1e-6)
    assert low.towers[0].height_m == 6.0


# ------------------------------------------------------------------------------------------ closed form, flat
def flat_clearance_min(h_a, h_b, length, w, tension):
    """Smallest conductor height over flat ground at the sampled whole metres x = 2 .. L of a straight span of
    ``length`` one-metre steps (the search does not sample the first cell after tower A):
    ``f(x) = h_a + (h_b - h_a) x / L - w x (L - x) / (2 T)``."""
    return min(h_a + (h_b - h_a) * x / length - w * x * (length - x) / (2.0 * tension) for x in range(2, length + 1))


@pytest.mark.parametrize("h_a,h_b,length", [(10.0, 10.0, 6), (10.0, 30.0, 8), (30.0, 7.0, 9), (20.0, 20.0, 11),
                                            (12.0, 5.0, 4)])
def test_the_python_clearance_check_equals_the_closed_form_over_flat_ground(h_a, h_b, length):
    w, tension, mc = 8.0, 100.0, 5.0
    ground = np.zeros((3, 30), np.float32)
    result = span_clearance(ground, None, (1, 0), (0, 1), length, 1.0, h_a, h_b, conductor_weight_per_m=w,
                            conductor_tension_n=tension, min_clearance_m=mc)
    closed = flat_clearance_min(h_a, h_b, length, w, tension)
    assert result.checked and result.worst_clearance_m == pytest.approx(closed, abs=1e-4)
    assert result.ok == (closed >= mc)


def test_a_flat_corridor_with_a_high_gantry_keeps_the_base_height_and_a_low_one_needs_more():
    """Flat ground, base tower 10 m, tall tower 20 m, sag 0.4 x (L - x), minimum clearance 6 m. Over the closed form:
    a 10 m tower reaches a 30 m gantry across 9 m and a 7 m gantry only across 4 m or less, a 20 m tower a 7 m gantry
    across 6 m. The search must therefore pick end towers whose reported pair of heights clears by the formula."""
    case = Corridor([100] * 24, min_span=3, max_span=12, heights=(20.0, 10.0), premiums=(300.0, 0.0), w=80.0,
                    tension=100.0, min_clearance=6.0, tower_map=((0, 100.0), (500, 100.0)))
    margins = {}
    for gantry in (30.0, 7.0):
        finder = corridor_finder(case, "enforce", heights=(20.0, 10.0), tower_map=FLAT_TOWER_MAP,
                                 cost_per_increment=90.0)
        path = run(finder, target_connection_height_m=gantry)
        last_a, last_b = path.towers[-2], path.towers[-1]
        length = int(round(path.spans[-1]))
        closed = flat_clearance_min(last_a.height_m, gantry, length, 80.0, 100.0)
        margins[gantry] = closed - 6.0
        assert closed >= 6.0, (gantry, length, last_a.height_m)        # the reported last span clears by the formula
        assert last_b.height_m == gantry and all_spans_clear(case, path)
        # never below the exact optimum (the base-height first pass of a variable-height run need not reach it)
        reference = case.oracle({"target_min_span": 3.0, "target_attach_height": gantry}, True)
        assert search_cost(path) >= reference * (1 - 1e-6)
    assert all(math.isfinite(m) for m in margins.values())


# ---------------------------------------------------------------------- premiums, own heights, the identity
def probe_corridor():
    """The planner's probe: 5 x 61, tall class 30 m (premium 200), base 10 m, an obstacle of 20 m on columns 29..31."""
    return Corridor([100] * 61, rows=5, row=2, min_span=5, max_span=12, heights=(30.0, 10.0), premiums=(200.0, 0.0),
                    obstacles={c: 20.0 for c in (29, 30, 31)}, min_clearance=5.0,
                    tower_map=((0, 100.0), (500, 100.0)), bin_size=5.0, n_bins=3, w=0.001, tension=1.0e6)


def test_the_probe_corridor_reports_each_towers_own_height_its_premium_and_the_identity():
    case = probe_corridor()
    finder = corridor_finder(case, heights=(30.0, 10.0), tower_map=FLAT_TOWER_MAP, cost_per_increment=30.0)
    path = run(finder)
    cols = [t.cell_index % case.n for t in path.towers]
    assert cols == [0, 11, 22, 31, 42, 53, 60]
    assert [t.height_m for t in path.towers[1:-1]] == [10.0, 10.0, 30.0, 10.0, 10.0]   # the tower in the obstacle
    assert [t.height_premium for t in path.towers] == [0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0]
    assert path.cost_breakdown["height_premiums"] == 200.0
    tall = path.towers[3]
    assert tall.total_cost == tall.terrain_cost + tall.angle_cost + 200.0
    assert path.total_tower_cost == pytest.approx(sum(t.total_cost for t in path.towers))
    assert path.cost_breakdown["towers"] == path.total_tower_cost
    assert search_cost(path) == pytest.approx(6950.0)              # the kernel's optimum, premium included
    assert path.total_cost == pytest.approx(path.total_terrain_cost + path.total_tower_cost
                                            + path.total_angle_penalty_cost)
    assert sum(path.tower_type_costs.values()) == pytest.approx(path.total_tower_cost)
    assert all_spans_clear(case, path)
    assert path.towers[0].height_premium == 0.0 and path.search_notes == []


def test_the_premium_of_a_gantry_run_is_charged_for_the_towers_but_not_for_the_source():
    """The kernel charges the source's own class when no source height is set; with a source connection height it
    charges nothing for the source. The report follows."""
    case = probe_corridor()
    finder = corridor_finder(case, heights=(30.0, 10.0), tower_map=FLAT_TOWER_MAP, cost_per_increment=30.0)
    path = run(finder, source_connection_height_m=12.0)
    assert path.towers[0].height_m == 12.0 and path.towers[0].height_premium == 0.0
    assert all_spans_clear(case, path)
    assert path.cost_breakdown["height_premiums"] == sum(t.height_premium for t in path.towers)
    assert search_cost(path) == pytest.approx(case.oracle({"source_attach_height": 12.0}, True), rel=1e-6)


@pytest.fixture
def height_spy(monkeypatch):
    """Record the optimal distance of every height-kernel call (the finder may call it twice: base height, all)."""
    original = algorithms.constrained_delta_stepping_height_2d
    calls = []

    def spy(**kwargs):
        path, towers, heights, dist = original(return_dist=1, **kwargs)
        calls.append({"kwargs": kwargs, "dist": dist, "found": len(path) > 0})
        return path, towers, heights

    monkeypatch.setattr(algorithms, "constrained_delta_stepping_height_2d", spy)
    return calls


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("opts", [{}, {"source_connection_height_m": 18.0},
                                  {"target_connection_height_m": 15.0},
                                  {"source_connection_height_m": 30.0, "target_connection_height_m": 30.0}])
def test_variable_heights_every_span_clears_with_the_reported_heights_and_the_identity_holds(seed, opts, height_spy):
    case = random_corridor(seed, heights=(24.0, 12.0), premiums=(150.0, 0.0))
    finder = corridor_finder(case, heights=(24.0, 12.0), cost_per_increment=37.5)
    try:
        path = run(finder, **opts)
    except NoPathFoundError:
        pytest.skip("no route for this height")
    final = [c for c in height_spy if c["found"]][-1]
    assert search_cost(path) == pytest.approx(final["dist"], rel=1e-6)
    assert_line_clear(case, path, (seed, opts))
    heights = {float(h) for h in (24.0, 12.0)}
    inner = path.towers[1:-1]
    assert all(t.height_m in heights for t in inner)
    assert [t.height_premium for t in inner] == [150.0 if t.height_m == 24.0 else 0.0 for t in inner]
    assert path.cost_breakdown["height_premiums"] == pytest.approx(sum(t.height_premium for t in path.towers))
    for key, value in opts.items():
        assert getattr(path, key) == value


def test_a_fixed_height_run_has_no_premium_and_no_premium_entry():
    case = random_corridor(0, heights=(20.0,), premiums=(0.0,))
    path = run(corridor_finder(case, heights=(20.0,)), source_connection_height_m=9.5)
    assert all(t.height_premium == 0.0 for t in path.towers)
    assert "height_premiums" not in path.cost_breakdown
    assert search_cost(path) == pytest.approx(case.oracle({"source_attach_height": 9.5}, True), rel=1e-6)


def test_the_source_height_class_and_premium_the_search_charged_are_reported():
    """No source height: the kernel seeds the source with every class and charges the premium of the one the first
    span needs. A 12 m source cannot clear a long first span over a 20 m obstacle; the 30 m class can: the report gives
    the source terminal that class (height and premium), so the identity holds."""
    case = Corridor([100] * 40, rows=5, row=2, min_span=5, max_span=12, heights=(30.0, 10.0), premiums=(200.0, 0.0),
                    obstacles={5: 20.0, 6: 20.0}, min_clearance=5.0, tower_map=((0, 100.0), (500, 100.0)),
                    bin_size=1.0, w=0.001, tension=1.0e6)
    finder = corridor_finder(case, heights=(30.0, 10.0), tower_map=FLAT_TOWER_MAP, cost_per_increment=30.0)
    path = run(finder)
    source = path.towers[0]
    ref = case.oracle({}, True)
    if source.height_premium > 0.0:
        assert source.height_m == 30.0 and source.height_premium == 200.0
        assert any("height class (premium 200)" in note for note in path.search_notes)
    assert search_cost(path) >= ref * (1 - 1e-6)
    assert_line_clear(case, path)
    assert source.total_cost == pytest.approx(source.terrain_cost + source.angle_cost + source.height_premium)


# ------------------------------------------------- remove_if_valid and the end-span check, several tower heights
ROWS, COLS, ROW = 5, 40, 2
SYN_TRANSFORM = from_origin(0.0, float(ROWS), 1.0, 1.0)
CLEARANCE = dict(tower_height_m=10.0, conductor_weight_per_m=1.0, conductor_tension_n=1000.0, min_clearance_m=5.0)
SEVERAL = {**CLEARANCE, "tower_heights_m": [10.0, 14.0], "tower_height_cost_per_increment": 0.0}


def synthetic(rule, cells, towers, heights, obstacle=None, **extra):
    """Report hand-made kernel output with the height of every search tower (``heights`` in the order of ``towers``)."""
    xy = lambda row, col: (col + 0.5, ROWS - row - 0.5)  # noqa: E731
    profile = {**PROFILE, "terminal_span_rule": rule, **extra}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=np.full((ROWS, COLS), 100, np.uint16), crs=CRS, transform=SYN_TRANSFORM,
            source_coords=xy(*cells[0]), target_coords=xy(*cells[-1]), search_space_buffer_m=100, profile=profile,
            graph_api="cython", dem=np.zeros((ROWS, COLS), np.float32))
    finder._obstacle_data = obstacle
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = finder._build_constrained_path(
            np.array([r * COLS + c for r, c in cells]), np.array([r * COLS + c for r, c in towers]),
            finder.raster_handler.data[0], xy(*cells[0]), xy(*cells[-1]), 0.0,
            tower_heights_arr=np.array(heights, np.float32))
    return path, [w for w in caught if issubclass(w.category, SpanRuleWarning)]


def line(first, last):
    return [(ROW, c) for c in range(first, last + 1)]


def test_remove_if_valid_judges_the_merged_span_with_each_towers_own_height():
    """Tower 8 stands 2 m from the end. Removing it merges tower 4 and the end into one span 4 -> 10; an obstacle of
    6 m on column 6 needs 11 m of conductor there. With a 14 m tower at column 4 the chord is 12.7 m there and the
    span clears (the tower goes); with a 10 m tower it is 10 m and the span fails (the tower stays, with a note)."""
    cells, towers = line(0, 10), [(ROW, 4), (ROW, 8)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 6] = 6.0
    tall, warned = synthetic("remove_if_valid", cells, towers, [14.0, 10.0], obstacle, **SEVERAL)
    assert [t.cell_index % COLS for t in tall.towers] == [0, 4, 10] and not warned
    assert [t.cell_index % COLS for t in tall.removed_towers] == [8]
    short, warned = synthetic("remove_if_valid", cells, towers, [10.0, 10.0], obstacle, **SEVERAL)
    assert [t.cell_index % COLS for t in short.towers] == [0, 4, 8, 10]
    # round 17 fix 3: the report checks EVERY span strictly. This hand-made line (not a search result) has an interior
    # span 4 -> 8 that the same obstacle breaks (10 m tower, 6 m obstacle on column 6), and the report now names it.
    assert len(warned) == 1 and "from (4.5, 2.5) to (8.5, 2.5)" in str(warned[0].message)
    assert "fails the clearance check" in short.towers[2].note and "cannot be checked" not in short.towers[2].note
    removed, warned = synthetic("remove", cells, towers, [10.0, 10.0], obstacle, **SEVERAL)
    assert [t.cell_index % COLS for t in removed.towers] == [0, 4, 10] and len(warned) == 1
    clear, warned = synthetic("remove", cells, towers, [14.0, 10.0], obstacle, **SEVERAL)
    assert [t.cell_index % COLS for t in clear.towers] == [0, 4, 10] and not warned


def test_the_end_span_check_uses_the_reported_heights_of_a_variable_height_line():
    """An 8 m obstacle at the target cell fails the last span from a 10 m tower and clears from a 20 m connection
    height at the target; the check names the heights it used."""
    cells, towers = line(0, 23), [(ROW, 7), (ROW, 14), (ROW, 21)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 23] = 8.0
    several = {**SEVERAL, "tower_heights_m": [10.0, 14.0]}
    path, warned = synthetic("keep", cells, towers, [14.0, 14.0, 10.0], obstacle, **several)
    assert len(warned) == 1 and "heights 10 m and 10 m" in str(warned[0].message)
    path, warned = synthetic("keep", cells, towers, [14.0, 14.0, 14.0], obstacle, **several)
    assert len(warned) == 1 and "heights 14 m and 10 m" in str(warned[0].message)
    assert [t.height_m for t in path.towers] == [10.0, 14.0, 14.0, 14.0, 10.0]


# ------------------------------------------------------------------------------------------ honest warnings
def square_finder(transform, **kwargs):
    return ConstrainedPathFinder(
        dataset_source=uniform(), crs=CRS, transform=transform, source_coords=(3.5, 15.5), target_coords=(26.5, 15.5),
        search_space_buffer_m=100, profile=dict(PROFILE), graph_api="cython", **kwargs)


def test_non_square_cells_warn_once_with_both_sizes_and_the_size_that_is_used():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder = square_finder(from_origin(0.0, 45.0, 1.0, 1.5))
        finder.find_route()
        finder.find_route()
    warned = [w for w in caught if "not square" in str(w.message)]
    assert len(warned) == 1 and issubclass(warned[0].category, PyorpsWarning)
    text = str(warned[0].message)
    assert "x size 1" in text and "y size 1.5" in text and "x size (1)" in text and "length" in text
    assert warned[0].filename == __file__


def test_square_cells_and_a_tiny_difference_do_not_warn():
    for transform in (from_origin(0.0, 30.0, 1.0, 1.0), Affine(1.0, 0.0, 0.0, 0.0, -1.0000005, 30.0)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SpanRuleWarning)
            warnings.filterwarnings("error", message=".*not square")
            square_finder(transform).find_route()


def test_a_gpu_solver_on_a_cell_size_other_than_one_warns_once_that_it_prices_per_cell():
    finder = square_finder(from_origin(0.0, 150.0, 5.0, 5.0))
    assert finder._cell_size == 5.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder._warn_gpu_cell_size("raster_gpu")
        finder._warn_gpu_cell_size("raster_gpu")
    assert len(caught) == 1 and issubclass(caught[0].category, PyorpsWarning)
    text = str(caught[0].message)
    assert "cell size is 5" in text and "'raster_gpu'" in text and "per cell" in text and "per metre" in text
    assert caught[0].filename == __file__
    unit = square_finder(from_origin(0.0, 30.0, 1.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unit._warn_gpu_cell_size("raster_gpu_v4")


@pytest.mark.parametrize("backend,module_name,function", [
    ("raster_gpu", "pyorps.utils.constrained_sssp_gpu_v2", "constrained_sssp_raster_gpu_v2"),
    ("raster_gpu_v4", "pyorps.utils.constrained_sssp_gpu_v4", "constrained_sssp_raster_gpu_v4")])
def test_a_stubbed_gpu_backend_warns_about_the_cell_size_after_it_ran(backend, module_name, function, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    seen = []

    def stub(**kwargs):
        seen.append(kwargs)
        return np.array([], np.uint32), np.array([], np.uint32)

    monkeypatch.setattr(module, function, stub)
    finder = ConstrainedPathFinder(
        dataset_source=uniform(), crs=CRS, transform=from_origin(0.0, 150.0, 5.0, 5.0),
        source_coords=(17.5, 77.5), target_coords=(132.5, 77.5), search_space_buffer_m=500,
        profile={**PROFILE, "min_span_m": 15.0, "max_span_m": 45.0}, graph_api=backend)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_route()
    assert len(seen) == 1
    assert len([w for w in caught if "price" in str(w.message) and "per cell" in str(w.message)]) == 1

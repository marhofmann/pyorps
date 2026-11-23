"""What becomes of a line tower next to an end point: the profile's ``terminal_span_rule`` (round 17).

The search keeps the first span at least ``min_span_m`` long but accepts the target at any span, so the last search
tower can stand closer than ``min_span_m`` to the target. Until round 16 the report silently removed such a tower
(``"remove"``, still the default). The rule is now a profile field with four values:

- ``"remove"``: as before; a ``SpanRuleWarning`` names the cases where the merged span reaches ``max_span_m``, the
  removed tower stood at a bend or the merged span fails the clearance check;
- ``"remove_if_valid"``: removed only when every span stays valid, otherwise kept and flagged;
- ``"keep"``: kept and flagged;
- ``"enforce"``: the search itself keeps every end span at least ``min_span_m`` long (P3; tested in
  ``test_terminal_enforce_round17.py``).

The rule function is tested on synthetic kernel output (deterministic, ``_build_constrained_path`` with hand-made
paths and towers), the four values end to end on the uniform and zigzag rasters, and the cost identity (reported total
minus terminal towers, plus ``cost_breakdown["removed_towers"]``, equals the kernel's optimum) for every value.
"""

import warnings

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps
import pyorps.core.exceptions as pex
import pyorps.utils.constrained_path_algorithms as constrained_algorithms
from pyorps.core.exceptions import InvalidInputError
from pyorps.core.infrastructure_profile import TERMINAL_SPAN_RULES, InfrastructureProfile
from pyorps.core.warning_types import PyorpsWarning, SpanRuleWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

CRS = "EPSG:25832"
N = 30
TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)
SOURCE = (3.5, 15.5)
TARGET = (26.5, 15.5)
RULES = ("remove", "remove_if_valid", "keep")

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
CLEARANCE = dict(tower_height_m=10.0, conductor_weight_per_m=1.0, conductor_tension_n=1000.0, min_clearance_m=5.0)


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


def noise(seed):
    rng = np.random.default_rng(seed)
    r = rng.integers(20, 400, (N, N)).astype(np.uint16)
    r[rng.random((N, N)) < 0.04] = 65535
    r[14, 3] = 100
    r[14, 26] = 100
    return r


def profile(rule="remove", **extra):
    return {**PROFILE, "terminal_span_rule": rule, **extra}


def make(raster, rule="remove", source=SOURCE, target=TARGET, transform=TRANSFORM, dem=None, **extra):
    kwargs = {} if dem is None else {"dem": dem}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=raster.copy(), crs=CRS, transform=transform, source_coords=source, target_coords=target,
            search_space_buffer_m=100, profile=profile(rule, **extra), graph_api="cython", **kwargs)


def route(raster, rule="remove", **kwargs):
    """The route and the SpanRuleWarnings it raised."""
    finder = make(raster, rule, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = finder.find_route()
    return path, [w for w in caught if issubclass(w.category, SpanRuleWarning)]


@pytest.fixture
def kernel_spy(monkeypatch):
    """Record the optimal distance and the towers of every constrained Dijkstra call."""
    original = constrained_algorithms.constrained_dijkstra_2d
    calls = []

    def spy(**kwargs):
        path, towers, dist = original(return_dist=1, **kwargs)
        calls.append({"dist": dist, "towers": [int(t) for t in towers]})
        return path, towers

    monkeypatch.setattr(constrained_algorithms, "constrained_dijkstra_2d", spy)
    return calls


def terminal_cost(path):
    return sum(t.total_cost for t in path.towers if t.tower_type == "terminal")


def removed_cost(path):
    return path.cost_breakdown.get("removed_towers", 0.0)


# ------------------------------------------------------------------------------------------------------- the profile
def test_the_rule_is_a_profile_field_that_defaults_to_the_old_behaviour():
    assert TERMINAL_SPAN_RULES == ("remove", "remove_if_valid", "keep", "enforce")
    assert InfrastructureProfile.from_dict(PROFILE).terminal_span_rule == "remove"      # a profile without the key
    assert InfrastructureProfile.from_dict(PROFILE).to_dict()["terminal_span_rule"] == "remove"


@pytest.mark.parametrize("rule", TERMINAL_SPAN_RULES)
def test_every_value_is_accepted_and_survives_a_json_round_trip(rule, tmp_path):
    prof = InfrastructureProfile.from_dict(profile(rule))
    assert prof.terminal_span_rule == rule
    path = tmp_path / "profile.json"
    prof.save(str(path))
    assert InfrastructureProfile.load(str(path)).terminal_span_rule == rule
    assert InfrastructureProfile.from_dict(prof.to_dict()) == prof


@pytest.mark.parametrize("rule", TERMINAL_SPAN_RULES)
def test_every_value_survives_a_yaml_round_trip(rule, tmp_path):
    pytest.importorskip("yaml")
    prof = InfrastructureProfile.from_dict(profile(rule))
    path = tmp_path / "profile.yaml"
    prof.save(str(path))
    assert InfrastructureProfile.load(str(path)).terminal_span_rule == rule


@pytest.mark.parametrize("bad", ["bogus", "", "Remove", None, 3, ["keep"]])
def test_an_unknown_value_is_refused_with_the_four_values_named(bad):
    with pytest.raises(InvalidInputError) as info:
        InfrastructureProfile.from_dict(profile(bad))
    for name in TERMINAL_SPAN_RULES:
        assert name in str(info.value)
    assert "terminal_span_rule" in str(info.value)


def test_the_rule_is_checked_even_without_span_limits():
    no_spans = dict(name="road", description="road", soft_angle_limit_deg=15.0, hard_angle_limit_deg=90.0,
                    angle_cost_function="linear", angle_cost_params={"scale": 50})
    with pytest.raises(InvalidInputError):
        InfrastructureProfile.from_dict({**no_spans, "terminal_span_rule": "bogus"})
    # a valid value has no effect without span limits, not even "enforce": no error, no warning
    for rule in TERMINAL_SPAN_RULES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = ConstrainedPathFinder(
                dataset_source=uniform(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
                search_space_buffer_m=100, profile={**no_spans, "terminal_span_rule": rule}, graph_api="cython")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SpanRuleWarning)
            path = finder.find_route()
        assert path.span_notes == [] and path.removed_towers == [] and path.terminal_span_rule == rule


# ---------------------------------------------------------------------------------------------------------- the warning
def test_span_rule_warning_is_a_pyorps_warning_exported_everywhere():
    assert issubclass(SpanRuleWarning, PyorpsWarning) and issubclass(SpanRuleWarning, UserWarning)
    assert pyorps.SpanRuleWarning is SpanRuleWarning is pex.SpanRuleWarning is pyorps.core.SpanRuleWarning
    assert "SpanRuleWarning" in pyorps.__all__ and "SpanRuleWarning" in pex.__all__


# ------------------------------------------------------------------- synthetic kernel output: the rule, deterministic
ROWS, COLS = 5, 40
SYN_TRANSFORM = from_origin(0.0, float(ROWS), 1.0, 1.0)
ROW = 2


def cell(row, col):
    return row * COLS + col


def xy(row, col):
    return (col + 0.5, ROWS - row - 0.5)


def synthetic(rule, cells, towers, *, dem=False, obstacle=None, **extra):
    """Report hand-made kernel output: ``cells`` are (row, col) of the path, ``towers`` those of the search towers."""
    finder = make(np.full((ROWS, COLS), 100, np.uint16), rule, source=xy(*cells[0]), target=xy(*cells[-1]),
                  transform=SYN_TRANSFORM, dem=np.zeros((ROWS, COLS), np.float32) if dem else None, **extra)
    if obstacle is not None:
        finder._obstacle_data = obstacle
    raster = finder.raster_handler.data[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = finder._build_constrained_path(
            np.array([cell(*c) for c in cells]), np.array([cell(*t) for t in towers]), raster,
            xy(*cells[0]), xy(*cells[-1]), 0.0)
    return path, [w for w in caught if issubclass(w.category, SpanRuleWarning)]


def east(first, last, row=ROW):
    return [(row, c) for c in range(first, last + 1)]


def tower_cols(path):
    return [t.cell_index % COLS for t in path.towers]


def test_a_straight_end_tower_is_removed_by_remove_and_remove_if_valid_and_kept_by_keep():
    cells, towers = east(0, 10), [(ROW, 4), (ROW, 8)]        # the last tower is 2 m from the end, spans 4, 4, 2
    removed, warned = synthetic("remove", cells, towers)
    assert tower_cols(removed) == [0, 4, 10] and not warned and removed.span_notes == []
    assert [t.cell_index % COLS for t in removed.removed_towers] == [8]
    assert removed.spans == pytest.approx([4.0, 6.0])        # merged span below max_span_m: valid
    valid, warned = synthetic("remove_if_valid", cells, towers)
    assert tower_cols(valid) == [0, 4, 10] and not warned and valid.span_notes == []
    assert [t.cell_index % COLS for t in valid.removed_towers] == [8]
    kept, warned = synthetic("keep", cells, towers)
    assert tower_cols(kept) == [0, 4, 8, 10] and not warned
    assert kept.spans == pytest.approx([4.0, 4.0, 2.0]) and kept.removed_towers == []
    assert "removed_towers" not in kept.cost_breakdown


def test_the_source_end_is_treated_like_the_target_end():
    cells, towers = east(0, 10), [(ROW, 2), (ROW, 6)]        # the first tower is 2 m from the start
    for rule in ("remove", "remove_if_valid"):
        path, warned = synthetic(rule, cells, towers)
        assert tower_cols(path) == [0, 6, 10] and not warned, rule
        assert [t.cell_index % COLS for t in path.removed_towers] == [2]
    kept, _ = synthetic("keep", cells, towers)
    assert tower_cols(kept) == [0, 2, 6, 10]
    assert "2.00 m is shorter than min_span_m 3 m" in kept.towers[1].note and kept.towers[2].note is None


def test_a_tower_close_to_both_ends_is_removed_once():
    path, warned = synthetic("remove", east(0, 4), [(ROW, 2)])
    assert tower_cols(path) == [0, 4] and not warned and len(path.removed_towers) == 1
    assert "both end points" in path.removed_towers[0].note
    kept, _ = synthetic("keep", east(0, 4), [(ROW, 2)])
    assert tower_cols(kept) == [0, 2, 4] and kept.span_notes and len(kept.span_notes) == 1


def test_a_far_enough_tower_is_left_alone_by_every_rule():
    cells, towers = east(0, 12), [(ROW, 4), (ROW, 8)]       # spans 4, 4, 4: nothing is closer than 3 m
    for rule in RULES:
        path, warned = synthetic(rule, cells, towers)
        assert tower_cols(path) == [0, 4, 8, 12] and not warned
        assert path.removed_towers == [] and path.span_notes == [] and all(t.note is None for t in path.towers)


def test_a_merged_span_above_the_longest_span_is_a_warning_under_remove_and_a_kept_tower_otherwise():
    cells, towers = east(0, 23), [(ROW, 7), (ROW, 14), (ROW, 21)]       # spans 7, 7, 7, 2; merged 9 m > 8 m
    path, warned = synthetic("remove", cells, towers)
    assert tower_cols(path) == [0, 7, 14, 23] and path.spans == pytest.approx([7.0, 7.0, 9.0])
    (warning,) = warned                                                 # one warning for the route
    text = str(warning.message)
    assert "terminal_span_rule='remove'" in text and "9.00 m" in text and "max_span_m 8" in text
    assert path.span_notes and "9.00 m" in path.span_notes[0]
    valid, warned = synthetic("remove_if_valid", cells, towers)
    assert tower_cols(valid) == [0, 7, 14, 21, 23] and not warned and valid.removed_towers == []
    assert "kept by terminal_span_rule='remove_if_valid'" in valid.towers[3].note and "9.00 m" in valid.towers[3].note
    assert valid.max_span_actual_m < 8.0


def test_a_merged_span_equal_to_the_longest_span_counts_as_reaching_it():
    cells, towers = east(0, 8), [(ROW, 6)]                  # the tower is 2 m from the end; merged span 8 m = max
    path, warned = synthetic("remove", cells, towers)
    assert len(warned) == 1 and path.spans == pytest.approx([8.0])
    valid, warned = synthetic("remove_if_valid", cells, towers)
    assert tower_cols(valid) == [0, 6, 8] and not warned
    below, warned = synthetic("remove", east(0, 7), [(ROW, 5)])   # 7 m < 8 m: fine
    assert not warned and below.spans == pytest.approx([7.0])


def test_a_tower_at_a_bend_is_a_warning_under_remove_and_kept_otherwise():
    cells = east(0, 8) + [(1, 8), (0, 8)]                   # east, then north: the tower at (2, 8) is the bend
    towers = [(ROW, 4), (ROW, 8)]                           # and stands 2 m from the end (0, 8)
    path, warned = synthetic("remove", cells, towers)
    assert tower_cols(path) == [0, 4, 8] and len(path.towers) == 3
    (warning,) = warned
    assert "a bend without a tower" in str(warning.message)
    assert path.towers[-1].cell_index == cell(0, 8)
    valid, warned = synthetic("remove_if_valid", cells, towers)
    assert [t.cell_index for t in valid.towers] == [cell(2, 0), cell(2, 4), cell(2, 8), cell(0, 8)]
    assert not warned and "a bend without a tower" in valid.towers[2].note
    assert valid.towers[2].turn_angle_deg == pytest.approx(90.0)
    kept, warned = synthetic("keep", cells, towers)
    assert [t.cell_index for t in kept.towers] == [t.cell_index for t in valid.towers] and not warned


def test_the_removed_tower_carries_the_cost_the_search_charged_and_no_line_id():
    cells, towers = east(0, 10), [(ROW, 4), (ROW, 8)]
    kept, _ = synthetic("keep", cells, towers)
    removed, _ = synthetic("remove", cells, towers)
    (gone,) = removed.removed_towers
    twin = next(t for t in kept.towers if t.cell_index == gone.cell_index)
    assert gone.tower_id == -1 and gone.tower_type == twin.tower_type
    assert (gone.terrain_cost, gone.angle_cost, gone.total_cost) == (twin.terrain_cost, twin.angle_cost,
                                                                     twin.total_cost)
    assert gone.span_to_previous_m == pytest.approx(4.0) and gone.span_to_next_m == pytest.approx(2.0)
    assert removed.cost_breakdown["removed_towers"] == gone.total_cost
    assert removed.n_towers == len(removed.towers) == 3
    assert removed.total_tower_cost == pytest.approx(kept.total_tower_cost - gone.total_cost)
    assert removed.total_cost == pytest.approx(kept.total_cost - gone.total_cost)   # the identity's second term
    assert [t.tower_id for t in removed.towers] == [0, 1, 2]
    assert removed.terminal_span_rule == "remove" and kept.terminal_span_rule == "keep"


def test_the_clearance_of_the_merged_span_is_judged_with_the_python_check():
    """An obstacle on the first cell after tower A is invisible to the SEARCH's check (a quirk of the search), on a
    later cell it is not: tower 8 can go only when the merged span 4 -> 10 clears. The report's strict check does see
    the first cell (round 17 fix 3), so the kept end span 8 -> 10 is named in a warning that says what the search
    does not test."""
    cells, towers = east(0, 10), [(ROW, 4), (ROW, 8)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 9] = 8.0            # not sampled on the span 8 -> 10, sampled on the merged span 4 -> 10
    path, warned = synthetic("remove", cells, towers, dem=True, obstacle=obstacle, **CLEARANCE)
    assert tower_cols(path) == [0, 4, 10] and len(warned) == 1
    assert "fails the clearance check" in str(warned[0].message) and "from (4.5, 2.5) to (10.5, 2.5)" in \
        str(warned[0].message)
    valid, warned = synthetic("remove_if_valid", cells, towers, dem=True, obstacle=obstacle, **CLEARANCE)
    assert tower_cols(valid) == [0, 4, 8, 10]
    assert len(warned) == 1 and "from (8.5, 2.5) to (10.5, 2.5)" in str(warned[0].message)   # the blind spot
    assert "does not test the first cell after a tower" in str(warned[0].message)
    assert "fails the clearance check" in valid.towers[2].note
    kept, warned = synthetic("keep", cells, towers, dem=True, obstacle=obstacle, **CLEARANCE)
    assert tower_cols(kept) == [0, 4, 8, 10] and len(warned) == 1
    clear, warned = synthetic("remove_if_valid", cells, towers, dem=True, **CLEARANCE)   # no obstacle: removable
    assert tower_cols(clear) == [0, 4, 10] and not warned


def test_several_tower_heights_are_judged_like_one_height_since_the_kernels_report_each_towers_own_height():
    """Until round 17 the kernels reported each tower's height shifted by one tower, so with several tower heights
    the Python check was switched off (remove_if_valid kept the tower with 'cannot be checked', remove stayed silent,
    no end span was judged). The heights are the towers' own now: the same cases give the same answers as with one
    height. (The synthetic kernel output carries no height map: every tower gets the lowest class, 10 m.)"""
    cells, towers = east(0, 10), [(ROW, 4), (ROW, 8)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 9] = 8.0
    several = {**CLEARANCE, "tower_heights_m": [10.0, 14.0]}
    valid, warned = synthetic("remove_if_valid", cells, towers, dem=True, obstacle=obstacle, **several)
    assert tower_cols(valid) == [0, 4, 8, 10]
    assert len(warned) == 1 and "from (8.5, 2.5) to (10.5, 2.5)" in str(warned[0].message)   # strict report check
    assert "fails the clearance check" in valid.towers[2].note
    assert "cannot be checked" not in valid.towers[2].note
    path, warned = synthetic("remove", cells, towers, dem=True, obstacle=obstacle, **several)
    assert tower_cols(path) == [0, 4, 10] and len(warned) == 1
    assert "fails the clearance check" in str(warned[0].message)
    cells, towers = east(0, 23), [(ROW, 7), (ROW, 14), (ROW, 21)]
    obstacle[ROW, 23] = 8.0           # at the target cell, under the last span: a warning, with one height or several
    for rule in ("keep", "remove_if_valid"):
        _, warned = synthetic(rule, cells, towers, dem=True, obstacle=obstacle, **several)
        assert len(warned) == 1
        _, warned = synthetic(rule, cells, towers, dem=True, obstacle=obstacle, **CLEARANCE)
        assert len(warned) == 1


def test_an_end_span_that_fails_the_clearance_is_a_warning_under_every_rule():
    cells, towers = east(0, 23), [(ROW, 7), (ROW, 14), (ROW, 21)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 22] = 8.0           # the first cell after tower 21: not sampled by the check
    obstacle[ROW, 23] = 8.0           # the target cell: sampled
    for rule in RULES:
        path, warned = synthetic(rule, cells, towers, dem=True, obstacle=obstacle, **CLEARANCE)
        # "remove" removes tower 21; its merged span 14 -> 23 fails too and is the one reported: one warning each
        assert len(warned) == 1, rule
        text = str(warned[0].message)
        assert "fails the clearance check" in text and "The search does not check the last span" in text
        assert any("fails the clearance check" in line for line in path.span_notes)
    clear, warned = synthetic("keep", cells, towers, dem=True, **CLEARANCE)
    assert not warned and len(clear.span_notes) == 1          # only the note on the kept two-metre span


def test_the_terminal_height_the_report_states_is_what_the_end_spans_are_checked_with():
    cells, towers = east(0, 14), [(ROW, 7)]
    plain, warned = synthetic("keep", cells, towers, dem=True, **CLEARANCE)
    assert not warned and [t.height_m for t in plain.towers] == [10.0, 10.0, 10.0]
    low, warned = synthetic("keep", cells, towers, dem=True, terminal_tower_height_m=2.0, **CLEARANCE)
    assert [t.height_m for t in low.towers] == [2.0, 10.0, 2.0]
    (warning,) = warned                                       # both end spans, one warning
    text = str(warning.message)
    assert "the first span from" in text and "the last span from" in text
    assert "heights 2 m and 10 m" in text and "heights 10 m and 2 m" in text
    assert len(low.span_notes) == 2


def test_a_straight_end_span_of_one_step_is_not_checked_by_the_search_but_is_named_by_the_report():
    path, warned = synthetic("keep", east(0, 8), [(ROW, 7)], dem=True,
                             obstacle=np.full((ROWS, COLS), 50.0, np.float32), **CLEARANCE)
    # span 7 -> 8 is one step: clear by the search's own rule. The report's strict check (round 17 fix 3) names it
    # too, and says that the search does not test a span of one step; the span 0 -> 7 fails (obstacle 50 m) as before
    assert len(warned) == 1 and "the first span from" in str(warned[0].message)
    text = str(warned[0].message)
    assert "the last span from (7.5, 2.5) to (8.5, 2.5)" in text and "does not test a span of one step" in text


# --------------------------------------------------------------------------------------------- "enforce" (round 17, P3)
def test_enforce_is_no_longer_refused_the_search_keeps_every_end_span():
    """P1 refused 'enforce' ('needs the search update'); P3 wires ``target_min_span`` into the search. The full
    behaviour is tested in ``test_terminal_enforce_round17.py``; here: the run works and no tower is removed."""
    path, warned = route(uniform(), "enforce")
    assert not warned and path.removed_towers == [] and path.terminal_span_rule == "enforce"
    assert min(path.spans) >= 3.0 * (1 - 1e-6) and max(path.spans) < 8.0


def test_enforce_flags_an_end_span_that_a_search_without_the_rule_left_short():
    """Output that did not come from an ``enforce`` search (hand-made here) is kept and warned about, not
    repaired: nothing is removed under 'enforce'."""
    path, warned = synthetic("enforce", east(0, 10), [(ROW, 4), (ROW, 8)])
    assert tower_cols(path) == [0, 4, 8, 10] and path.removed_towers == []
    (warning,) = warned
    assert "although the rule asks the search" in str(warning.message)
    exact, warned = synthetic("enforce", east(0, 11), [(ROW, 4), (ROW, 8)])     # last span exactly 3 m: fine
    assert tower_cols(exact) == [0, 4, 8, 11] and not warned and exact.span_notes == []


# --------------------------------------------------------------------------------- the four values, end to end
def test_remove_reproduces_the_old_result_and_warns_once_on_the_uniform_raster():
    path, warned = route(uniform(), "remove")
    # recorded with the code of round 16: the tower at column 21 is dropped, the last span is 9 m (above max_span_m)
    assert [t.cell_index for t in path.towers] == [423, 430, 437, 446]
    assert path.spans == pytest.approx([7.0, 7.0, 9.0])
    assert [t.cell_index for t in path.removed_towers] == [444]
    assert len(warned) == 1 and issubclass(warned[0].category, PyorpsWarning)
    assert "9.00 m" in str(warned[0].message) and "terminal_span_rule='remove'" in str(warned[0].message)
    assert len(path.span_notes) == 1 and "9.00 m" in path.span_notes[0]
    assert path.cost_breakdown["removed_towers"] == path.removed_towers[0].total_cost > 0.0
    assert path.total_cost == pytest.approx(path.total_terrain_cost + path.total_tower_cost +
                                            path.total_angle_penalty_cost)


def test_keep_and_remove_if_valid_keep_the_two_metre_span_without_a_warning():
    for rule in ("keep", "remove_if_valid"):
        path, warned = route(uniform(), rule)
        assert [t.cell_index for t in path.towers] == [423, 430, 437, 444, 446], rule
        assert path.spans == pytest.approx([7.0, 7.0, 7.0, 2.0]) and not warned and path.removed_towers == []
        assert path.max_span_actual_m < 8.0 and "removed_towers" not in path.cost_breakdown
        assert path.towers[3].note and "2.00 m is shorter than min_span_m 3 m" in path.towers[3].note
        assert path.span_notes == [path.towers[3].note]
        assert all(t.note is None for t in path.towers[:3] + path.towers[4:])
        assert path.terminal_span_rule == rule
    valid, _ = route(uniform(), "remove_if_valid")
    assert "9.00 m" in valid.towers[3].note       # why it stays: removing it would leave a 9 m span


def test_a_route_without_a_close_tower_is_the_same_under_every_rule():
    results = {rule: route(zigzag(), rule) for rule in RULES}
    for rule in RULES:
        path, warned = results[rule]
        assert not warned and path.removed_towers == [] and path.span_notes == []
        assert all(t.note is None for t in path.towers)
        reference = results["remove"][0]
        assert path.towers and [t.cell_index for t in path.towers] == [t.cell_index for t in reference.towers]
        assert path.total_cost == reference.total_cost
        assert path.cost_breakdown == reference.cost_breakdown


def test_export_columns_appear_only_when_they_have_something_to_say():
    plain, _ = route(zigzag(), "remove")
    assert "note" not in plain.towers_to_geodataframe(crs=CRS).columns
    assert "n_removed_towers" not in plain.to_geodataframe_dict()
    kept, _ = route(uniform(), "keep")
    frame = kept.towers_to_geodataframe(crs=CRS)
    assert "note" in frame.columns and frame["note"].notna().sum() == 1
    assert "n_removed_towers" not in kept.to_geodataframe_dict()
    removed, _ = route(uniform(), "remove")
    assert "note" not in removed.towers_to_geodataframe(crs=CRS).columns      # the removed tower is not a line tower
    assert removed.to_geodataframe_dict()["n_removed_towers"] == 1


# --------------------------------------------------------------------------------------- the cost identity per value
def test_total_cost_identity_holds_for_every_rule_on_the_uniform_raster(kernel_spy):
    for rule in RULES:
        kernel_spy.clear()
        finder = make(uniform(), rule)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = finder.find_route()
        (call,) = kernel_spy
        assert path.total_cost - terminal_cost(path) + removed_cost(path) == pytest.approx(
            call["dist"] / finder._search_scale, rel=1e-6), rule
        kept = [t.cell_index for t in path.towers if t.tower_type != "terminal"]
        removed = [t.cell_index for t in path.removed_towers]
        assert sorted(kept + removed) == sorted(call["towers"]), rule        # nothing is lost, only moved


@pytest.mark.parametrize("objective", [None, {"cost": 1.0, "tower": 5.0}, {"cost": 1.0, "smoothness": 0.5}])
def test_total_cost_identity_holds_for_every_rule_under_an_objective(objective, kernel_spy):
    for rule in RULES:
        kernel_spy.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = ConstrainedPathFinder(
                dataset_source=uniform(), crs=CRS, transform=TRANSFORM, source_coords=SOURCE, target_coords=TARGET,
                search_space_buffer_m=100, profile=profile(rule), graph_api="cython", objective=objective)
            path = finder.find_route()
        (call,) = kernel_spy
        assert path.total_cost - terminal_cost(path) + removed_cost(path) == pytest.approx(
            call["dist"] / finder._search_scale, rel=1e-6), (rule, objective)


def test_total_cost_identity_and_span_rules_on_random_rasters(kernel_spy):
    removed_by = {rule: 0 for rule in RULES}
    flagged = 0
    for seed in range(24):
        for rule in RULES:
            kernel_spy.clear()
            finder = make(noise(seed), rule, source=(3.5, 15.5), target=(26.5, 15.5))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                path = finder.find_route()
            warned = [w for w in caught if issubclass(w.category, SpanRuleWarning)]
            (call,) = kernel_spy
            if len(path.path_indices) < 2:
                continue
            assert path.total_cost - terminal_cost(path) + removed_cost(path) == pytest.approx(
                call["dist"] / finder._search_scale, rel=1e-6), (seed, rule)
            removed_by[rule] += len(path.removed_towers)
            flagged += sum(t.note is not None for t in path.towers)
            spans = path.spans
            path_arr = np.asarray(path.path_indices)
            steps = np.diff(path_arr)
            at_tower = {t.cell_index for t in path.towers}
            bends_without_tower = [int(path_arr[k]) for k in range(1, len(steps))
                                   if steps[k] != steps[k - 1] and int(path_arr[k]) not in at_tower]
            if rule in ("keep", "remove_if_valid"):
                # every span is in order: nothing to warn about, no span above the longest span, a tower at every bend
                assert not warned and not bends_without_tower, (seed, rule)
                assert max(spans) < 8.0 * (1 + 1e-6), (seed, rule)
                assert min(spans[1:-1], default=3.0) >= 3.0 * (1 - 1e-6), (seed, rule)     # interior spans
                assert spans[0] >= 3.0 * (1 - 1e-6), (seed, rule)
            if rule == "keep":
                assert path.removed_towers == [], seed
            if rule == "remove":
                # the warning is there exactly when a merged span reaches the longest span or a bend lost its tower
                assert bool(warned) == (max(spans) >= 8.0 * (1 - 1e-9) or bool(bends_without_tower)), (seed, spans)
    assert removed_by["keep"] == 0
    assert removed_by["remove"] >= 3 and flagged >= 3          # the loop meets the cases it is about

"""
Tests for the exactness referee (plan item 0.5).

The referee is what every accepted performance lever is judged by, so it
must itself be trustworthy before it judges anything. Every arithmetic
assertion here is a hand-computed value, and nothing in this file runs a
kernel, a backend or the GPU: Layer 2 is exercised with fabricated
``BackendRun`` objects so the judgement logic is testable on its own.

Covers:
- the intermediate-cell set and the cost factor per step class
- re-pricing a straight step, a diagonal (2 intermediates) and an R2 knight
  move (2 asymmetric intermediates)
- rejection of impassable path cells, impassable intermediates, illegal
  steps and revisits, under both exclusion sentinels (65535 and the
  ignore_max=False 0-sentinel)
- the gradient/LUT term: multiplier, additive, bin clamping, hard limit
- the tolerance policy and the verdict classification, including the
  recorded 0.04 % Dijkstra-vs-delta anomaly and the FIM envelope
- Layer 2 judgement: suboptimal, cheaper-than-reference, missing path,
  self-consistency mismatch, approximate backend
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.exactness_referee import (
    EXACT,
    FIM_ENVELOPE,
    INVALID_PATH,
    NO_EXCLUSION_VALUE,
    NO_PATH,
    OUTSIDE_ENVELOPE,
    REFERENCE_SUSPECT,
    SUBOPTIMAL,
    U_FLOAT32,
    U_FLOAT64,
    WITHIN_ENVELOPE,
    WITHIN_FLOAT32,
    BackendRun,
    EdgeModel,
    PairVerdict,
    RefereeReport,
    accumulation_bound,
    classify_excess,
    cost_factor,
    intermediate_steps,
    judge_run,
    optimality_tolerance,
    random_pairs,
    reprice_path,
    self_consistency_tolerance,
    synthetic_raster,
)


# ==================== FIXTURES ====================

@pytest.fixture
def steps_r2():
    """R2-style step set; index 0 is (0, 1), index 3 is the knight move."""
    return np.array([
        [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [0, -1], [-1, 0], [-1, -1],
    ], dtype=np.int8)


@pytest.fixture
def uniform_raster():
    """3x3 raster, every cell cost 10."""
    return np.full((3, 3), 10, dtype=np.uint16)


@pytest.fixture
def labelled_raster():
    """3x3 raster with a distinct value per cell of the knight move."""
    r = np.full((3, 3), 100, dtype=np.uint16)
    r[0, 0] = 1
    r[0, 1] = 2
    r[1, 1] = 4
    r[1, 2] = 8
    return r


def gradient_luts(n_dirs, bin_factor=1.0):
    """4-bin LUT pair: bin 1 doubles, bin 2 is forbidden, bin 3 quadruples."""
    return dict(
        mult=np.array([1.0, 2.0, np.inf, 4.0], dtype=np.float32),
        add=np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32),
        bin_factor=np.full(n_dirs, bin_factor, dtype=np.float32),
        step_len_cells=np.ones(n_dirs, dtype=np.float32),
        n_bins=4,
    )


# ==================== THE EDGE MODEL ====================

def test_intermediate_steps_cardinal_has_none():
    assert intermediate_steps(0, 1) == ()
    assert intermediate_steps(-1, 0) == ()


def test_intermediate_steps_diagonal_decomposes_orthogonally():
    assert intermediate_steps(1, 1) == ((1, 0), (0, 1))
    assert intermediate_steps(-1, 1) == ((-1, 0), (0, 1))


def test_intermediate_steps_knight_move():
    # k = 2, p = 1 -> (0.5, 1.0): floor (0, 1) and ceil (1, 1) differ.
    assert intermediate_steps(1, 2) == ((0, 1), (1, 1))
    assert intermediate_steps(2, 1) == ((1, 0), (1, 1))


def test_intermediate_steps_long_diagonal_is_collinear():
    # (2, 2): p = 1 -> (1.0, 1.0), floor == ceil, one intermediate only.
    assert intermediate_steps(2, 2) == ((1, 1),)


def test_cost_factor_matches_hand_values():
    assert cost_factor(0, 1, 0) == 0.5
    assert cost_factor(1, 1, 2) == float(
        np.float32(math.sqrt(2.0)) / np.float32(4.0))
    assert cost_factor(1, 2, 2) == pytest.approx(math.sqrt(5.0) / 4.0,
                                                 rel=1e-7)
    assert cost_factor(1, 2, 2, precision="float64") == math.sqrt(5.0) / 4.0


def test_cost_factor_rejects_unknown_precision():
    with pytest.raises(ValueError):
        cost_factor(0, 1, 0, precision="float16")


def test_edge_model_reports_step_legality(steps_r2):
    model = EdgeModel(steps_r2)
    assert model.is_legal_step(1, 2)
    assert not model.is_legal_step(0, 3)
    index, inter, factor = model.edge(0, 3)
    assert index == -1          # priced anyway, but flagged as illegal
    assert factor > 0.0


# ==================== LAYER 1: RE-PRICING ARITHMETIC ====================

def test_straight_step(uniform_raster, steps_r2):
    """(10 + 10) * (1 / 2) = 10 exactly."""
    priced = reprice_path([0, 1], uniform_raster, EdgeModel(steps_r2))
    assert priced.cost == 10.0
    assert priced.hops == 1
    assert priced.n_cells == 2
    assert priced.ok


def test_diagonal_step_with_two_intermediates(uniform_raster, steps_r2):
    """(10 + 10 + 10 + 10) * sqrt(2) / 4 over cells (0,0),(1,0),(0,1),(1,1)."""
    priced = reprice_path([0, 4], uniform_raster, EdgeModel(steps_r2))
    expected = 40.0 * float(np.float32(math.sqrt(2.0)) / np.float32(4.0))
    assert priced.cost == expected
    assert priced.cost == pytest.approx(40.0 * math.sqrt(2.0) / 4.0, rel=1e-7)
    assert priced.ok


def test_knight_move_prices_exactly_its_two_intermediates(labelled_raster,
                                                          steps_r2):
    """(1 + 2 + 4 + 8) * sqrt(5) / 4: u, both intermediates, v."""
    priced = reprice_path([0, 5], labelled_raster, EdgeModel(steps_r2))
    assert priced.cost == pytest.approx(15.0 * math.sqrt(5.0) / 4.0, rel=1e-7)
    assert priced.ok
    # A wrong intermediate set would have picked up a 100-cost cell.
    assert priced.cost < 20.0


def test_single_cell_path_costs_nothing(uniform_raster, steps_r2):
    priced = reprice_path([4], uniform_raster, EdgeModel(steps_r2))
    assert priced.cost == 0.0
    assert priced.hops == 0
    assert priced.ok


def test_multi_hop_path_sums_its_edges(uniform_raster, steps_r2):
    priced = reprice_path([0, 1, 2], uniform_raster, EdgeModel(steps_r2))
    assert priced.cost == 20.0
    assert priced.hops == 2
    assert priced.length_cells == 2.0


def test_row_col_input_is_equivalent_to_flat_indices(uniform_raster, steps_r2):
    model = EdgeModel(steps_r2)
    flat = reprice_path([0, 1, 2], uniform_raster, model)
    pairs = reprice_path([[0, 0], [0, 1], [0, 2]], uniform_raster, model)
    assert flat.cost == pairs.cost


def test_empty_path_is_a_violation(uniform_raster, steps_r2):
    priced = reprice_path([], uniform_raster, EdgeModel(steps_r2))
    assert not priced.ok
    assert priced.violations[0]["kind"] == "empty_path"


# ==================== LAYER 1: EXCLUSION SEMANTICS ====================

def test_impassable_path_cell_is_rejected(uniform_raster, steps_r2):
    raster = uniform_raster.copy()
    raster[0, 1] = 65535
    priced = reprice_path([0, 1], raster, EdgeModel(steps_r2))
    assert not priced.ok
    assert any(v["kind"] == "impassable_cell" for v in priced.violations)


def test_impassable_intermediate_is_rejected(uniform_raster, steps_r2):
    """The diagonal (0,0)->(1,1) must not squeeze past a blocked (1,0)."""
    raster = uniform_raster.copy()
    raster[1, 0] = 65535
    priced = reprice_path([0, 4], raster, EdgeModel(steps_r2))
    assert not priced.ok
    assert any(v["kind"] == "impassable_intermediate"
               for v in priced.violations)
    assert priced.cost == math.inf


def test_in_domain_zero_sentinel_excludes_zero_cost_cells(uniform_raster,
                                                          steps_r2):
    """max_value=0, passed explicitly, is a generic EdgeModel property."""
    raster = uniform_raster.copy()
    raster[0, 1] = 0
    model = EdgeModel(steps_r2, max_value=0)
    priced = reprice_path([0, 1], raster, model)
    assert not priced.ok
    assert any(v["kind"] == "impassable_cell" for v in priced.violations)


def test_no_exclusion_sentinel_leaves_zero_cost_cells_traversable(
        uniform_raster, steps_r2):
    """ignore_max=False -> NO_EXCLUSION_VALUE, not 0 (plan item 0.7)."""
    raster = uniform_raster.copy()
    raster[0, 1] = 0
    model = EdgeModel(steps_r2, max_value=NO_EXCLUSION_VALUE)
    priced = reprice_path([0, 1], raster, model)
    assert priced.ok
    assert priced.cost == 5.0


def test_zero_cost_cells_are_free_under_the_default_sentinel(uniform_raster,
                                                             steps_r2):
    raster = uniform_raster.copy()
    raster[0, 1] = 0
    priced = reprice_path([0, 1], raster, EdgeModel(steps_r2))
    assert priced.cost == 5.0
    assert priced.ok


def test_out_of_bounds_cell_is_rejected(uniform_raster, steps_r2):
    priced = reprice_path([0, 99], uniform_raster, EdgeModel(steps_r2))
    assert not priced.ok
    assert any(v["kind"] == "out_of_bounds" for v in priced.violations)


# ==================== LAYER 1: STEP LEGALITY ====================

def test_illegal_step_is_flagged(uniform_raster):
    cardinal = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
    priced = reprice_path([0, 4], uniform_raster, EdgeModel(cardinal))
    assert not priced.ok
    assert any(v["kind"] == "illegal_step" for v in priced.violations)


def test_illegal_step_is_only_a_note_when_not_strict(uniform_raster):
    """Approximate backends return supercover chains, not neighbourhood
    steps -- their legality is reported, not asserted."""
    cardinal = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
    priced = reprice_path([0, 4], uniform_raster, EdgeModel(cardinal),
                          strict=False)
    assert priced.ok
    assert priced.notes


def test_revisited_cell_is_flagged(uniform_raster, steps_r2):
    priced = reprice_path([0, 1, 0], uniform_raster, EdgeModel(steps_r2))
    assert not priced.ok
    assert any(v["kind"] == "revisited_cell" for v in priced.violations)


# ==================== LAYER 1: GRADIENT / LUT TERM ====================

def test_gradient_multiplier_and_additive(uniform_raster, steps_r2):
    """terrain 10 -> bin 1 -> 10 * 2.0 + 3.0 * 1.0 = 23."""
    dem = np.zeros((3, 3), dtype=np.float32)
    dem[0, 1] = 1.5
    model = EdgeModel(steps_r2, dem=dem,
                      gradient_luts=gradient_luts(len(steps_r2)))
    priced = reprice_path([0, 1], uniform_raster, model)
    assert priced.cost == 23.0
    assert priced.ok


def test_gradient_bin_is_clamped_to_the_last_bin(uniform_raster, steps_r2):
    """|dh| = 100 -> bin 100 -> clamped to 3 -> 10 * 4.0 + 0.0."""
    dem = np.zeros((3, 3), dtype=np.float32)
    dem[0, 1] = 100.0
    model = EdgeModel(steps_r2, dem=dem,
                      gradient_luts=gradient_luts(len(steps_r2)))
    priced = reprice_path([0, 1], uniform_raster, model)
    assert priced.cost == 40.0


def test_gradient_hard_limit_forbids_the_edge(uniform_raster, steps_r2):
    """mult[bin] == inf marks the edge forbidden."""
    dem = np.zeros((3, 3), dtype=np.float32)
    dem[0, 1] = 2.5
    model = EdgeModel(steps_r2, dem=dem,
                      gradient_luts=gradient_luts(len(steps_r2)))
    priced = reprice_path([0, 1], uniform_raster, model)
    assert priced.cost == math.inf
    assert not priced.ok
    assert any(v["kind"] == "forbidden_gradient" for v in priced.violations)


def test_gradient_uses_the_absolute_height_difference(uniform_raster,
                                                      steps_r2):
    """Downhill and uphill price identically (the kernel takes fabs)."""
    up = np.zeros((3, 3), dtype=np.float32)
    up[0, 1] = 1.5
    down = np.zeros((3, 3), dtype=np.float32)
    down[0, 0] = 1.5
    luts = gradient_luts(len(steps_r2))
    cost_up = reprice_path([0, 1], uniform_raster,
                           EdgeModel(steps_r2, dem=up, gradient_luts=luts)).cost
    cost_down = reprice_path(
        [0, 1], uniform_raster,
        EdgeModel(steps_r2, dem=down, gradient_luts=luts)).cost
    assert cost_up == cost_down == 23.0


def test_gradient_luts_accept_an_attribute_object(uniform_raster, steps_r2):
    """GradientLUTs is a dataclass; the mapping form is a test convenience."""
    dem = np.zeros((3, 3), dtype=np.float32)
    dem[0, 1] = 1.5
    luts = SimpleNamespace(**gradient_luts(len(steps_r2)))
    model = EdgeModel(steps_r2, dem=dem, gradient_luts=luts)
    assert reprice_path([0, 1], uniform_raster, model).cost == 23.0


def test_gradient_luts_require_a_dem(steps_r2):
    with pytest.raises(ValueError):
        EdgeModel(steps_r2, gradient_luts=gradient_luts(len(steps_r2)))


def test_gradient_luts_validate_bin_count(steps_r2):
    dem = np.zeros((3, 3), dtype=np.float32)
    luts = gradient_luts(len(steps_r2))
    luts["n_bins"] = 3
    with pytest.raises(ValueError):
        EdgeModel(steps_r2, dem=dem, gradient_luts=luts)


# ==================== TOLERANCE POLICY ====================

def test_accumulation_bound_is_hops_times_unit_roundoff():
    assert accumulation_bound(0, U_FLOAT32) == 8 * U_FLOAT32
    assert accumulation_bound(10_000, U_FLOAT32) == pytest.approx(
        5.965e-4, rel=1e-3)
    assert accumulation_bound(10_000, U_FLOAT64) == pytest.approx(
        1.11e-12, rel=1e-2)


def test_self_consistency_tolerance_carries_the_safety_factor():
    assert (self_consistency_tolerance(1000, U_FLOAT32)
            == 4.0 * accumulation_bound(1000, U_FLOAT32))


def test_optimality_tolerance_adds_both_searches():
    tol = optimality_tolerance(10_000, 10_000, U_FLOAT64, U_FLOAT32)
    assert tol == pytest.approx(accumulation_bound(10_000, U_FLOAT32),
                                rel=1e-6)
    assert optimality_tolerance(10_000, 10_000, U_FLOAT64,
                                U_FLOAT64) < 1e-11


def test_classification_separates_rounding_from_suboptimality():
    tol = optimality_tolerance(10_000, 10_000, U_FLOAT64, U_FLOAT32)
    tol_exact = optimality_tolerance(10_000, 10_000, U_FLOAT64, U_FLOAT64)
    assert classify_excess(0.0, tol, tol_exact) == EXACT
    assert classify_excess(1e-5, tol, tol_exact) == WITHIN_FLOAT32
    assert classify_excess(1e-2, tol, tol_exact) == SUBOPTIMAL


def test_recorded_004_percent_anomaly_is_flagged():
    """FUSED_KERNEL_FINDINGS.md:87-89 -- delta-stepping found a path 0.04 %
    cheaper than dijkstra_2d_cython on a random 500^2 raster (~700 hops).
    Both signs must be errors, and the cheaper one must indict Dijkstra."""
    tol = optimality_tolerance(700, 700, U_FLOAT64, U_FLOAT32)
    tol_exact = optimality_tolerance(700, 700, U_FLOAT64, U_FLOAT64)
    assert tol < 4e-4
    assert classify_excess(-4e-4, tol, tol_exact) == REFERENCE_SUSPECT
    assert classify_excess(+4e-4, tol, tol_exact) == SUBOPTIMAL


def test_parallel_delta_suboptimality_is_flagged_at_target_scale():
    """The measured 0.58-2.8 % at >= 2 threads is orders above the float32
    accumulation bound even at 10 k hops."""
    tol = optimality_tolerance(10_000, 10_000, U_FLOAT64, U_FLOAT32)
    assert classify_excess(0.0058, tol, 0.0) == SUBOPTIMAL


def test_fim_is_judged_against_an_envelope_not_equality():
    tol = optimality_tolerance(10_000, 10_000, U_FLOAT64, U_FLOAT32)
    low, high = FIM_ENVELOPE
    assert (low, high) == (-0.011, 0.037)
    assert classify_excess(0.023, tol, 0.0, approximate=True) == \
        WITHIN_ENVELOPE
    assert classify_excess(0.05, tol, 0.0, approximate=True) == \
        OUTSIDE_ENVELOPE
    assert classify_excess(-0.02, tol, 0.0, approximate=True) == \
        OUTSIDE_ENVELOPE
    # The same excess is a hard failure for an exact backend.
    assert classify_excess(0.023, tol, 0.0) == SUBOPTIMAL


# ==================== LAYER 2: JUDGEMENT ====================

@pytest.fixture
def corridor():
    """3x11 raster: row 1 costs 10, row 0 costs 1, row 2 costs 100."""
    raster = np.full((3, 11), 10, dtype=np.uint16)
    raster[0, :] = 1
    raster[2, :] = 100
    return raster


@pytest.fixture
def cardinal_model():
    return EdgeModel(np.array([[0, 1], [0, -1], [1, 0], [-1, 0]],
                              dtype=np.int8))


def _row_path(row, cols=11):
    return [row * cols + c for c in range(cols)]


def test_identical_path_is_exact(corridor, cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("delta-stepping", 11, 21, _row_path(1), None, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == EXACT
    assert verdict.excess_rel == 0.0
    assert verdict.self_consistency == "SKIPPED_NO_REPORTED_COST"
    assert not verdict.failed


def test_more_expensive_path_is_suboptimal(corridor, cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("delta-stepping", 22, 32, _row_path(2), None, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == SUBOPTIMAL
    assert verdict.excess_rel == pytest.approx(9.0)
    assert verdict.failed


def test_cheaper_path_indicts_the_reference(corridor, cardinal_model):
    """A backend beating the exact reference is an error against the
    reference -- the missing gate behind the 0.04 % anomaly."""
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("delta-stepping", 0, 10, _row_path(0), None, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == REFERENCE_SUSPECT
    assert verdict.excess_rel < 0
    assert verdict.failed


def test_missing_path_against_a_reachable_reference_fails(corridor,
                                                          cardinal_model):
    """The wall-with-gap class: reference routes, candidate reports none."""
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("raster_gpu", 11, 21, None, None, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == NO_PATH
    assert verdict.failed


def test_path_where_the_reference_found_none_indicts_the_reference(
        corridor, cardinal_model):
    run = BackendRun("raster_gpu", 11, 21, _row_path(1), None, 0.01)
    verdict = judge_run(run, None, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == REFERENCE_SUSPECT
    assert verdict.failed


def test_both_unreachable_agree(corridor, cardinal_model):
    run = BackendRun("raster_gpu", 11, 21, None, None, 0.01)
    verdict = judge_run(run, None, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == EXACT
    assert not verdict.failed


def test_reported_cost_must_match_the_repriced_cost(corridor,
                                                    cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("raster_gpu", 11, 21, _row_path(1),
                     reference.cost * 2.0, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.self_consistency == "MISMATCH"
    assert verdict.verdict == INVALID_PATH
    assert verdict.failed


def test_reported_cost_within_float32_accumulation_is_accepted(
        corridor, cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    drift = 1.0 + 0.5 * self_consistency_tolerance(reference.hops, U_FLOAT32)
    run = BackendRun("raster_gpu", 11, 21, _row_path(1),
                     reference.cost * drift, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.self_consistency == "OK"
    assert verdict.verdict == EXACT


def test_backend_error_becomes_an_invalid_path_verdict(corridor,
                                                       cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("raster_gpu", 11, 21, None, None, 0.01,
                     error="RuntimeError: out of memory")
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == INVALID_PATH
    assert verdict.failed


def test_invalid_candidate_path_overrides_the_cost_comparison(
        corridor, cardinal_model):
    """A cheap path through an impassable cell must never read as EXACT."""
    raster = corridor.copy()
    raster[0, 5] = 65535
    reference = reprice_path(_row_path(1), raster, cardinal_model)
    run = BackendRun("delta-stepping", 0, 10, _row_path(0), None, 0.01)
    verdict = judge_run(run, reference, "dijkstra", raster, cardinal_model)
    assert verdict.verdict == INVALID_PATH
    assert verdict.failed


def test_approximate_backend_skips_self_consistency_and_uses_the_envelope(
        corridor, cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("raster_fim", 11, 21, _row_path(1), 42.0, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.self_consistency == "SKIPPED_APPROXIMATE"
    assert verdict.verdict == WITHIN_ENVELOPE
    assert not verdict.failed


def test_approximate_backend_outside_the_envelope_fails(corridor,
                                                        cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    run = BackendRun("raster_fim", 22, 32, _row_path(2), 42.0, 0.01)
    verdict = judge_run(run, reference, "dijkstra", corridor, cardinal_model)
    assert verdict.verdict == OUTSIDE_ENVELOPE
    assert verdict.failed


# ==================== REPORTING ====================

def test_report_summarises_and_serialises(corridor, cardinal_model):
    reference = reprice_path(_row_path(1), corridor, cardinal_model)
    runs = [
        BackendRun("delta-stepping", 11, 21, _row_path(1), None, 0.01),
        BackendRun("delta-stepping", 22, 32, _row_path(2), None, 0.02),
    ]
    report = RefereeReport(reference="dijkstra", n_pairs=2,
                           raster_shape=(3, 11), neighborhood="r1")
    report.verdicts = [judge_run(r, reference, "dijkstra", corridor,
                                 cardinal_model) for r in runs]

    summary = report.summary()["delta-stepping"]
    assert summary["n"] == 2
    assert summary["n_failed"] == 1
    assert summary["max_excess"] == pytest.approx(9.0)
    assert not report.ok
    assert len(report.failures) == 1

    payload = report.to_dict()
    assert payload["tolerance_policy"]["u_float32"] == U_FLOAT32
    assert json.loads(json.dumps(payload))["ok"] is False
    assert "SUBOPTIMAL" in report.format_table()


def test_synthetic_rasters_have_the_advertised_structure():
    rng = np.random.default_rng(0)
    assert synthetic_raster("random", 32, rng).min() >= 1
    walls = synthetic_raster("walls", 40, rng)
    assert (walls == 65535).any()
    zeros = synthetic_raster("zero_cost", 40, rng)
    assert (zeros == 0).any()
    with pytest.raises(ValueError):
        synthetic_raster("nope", 8, rng)


def test_random_pairs_are_passable_and_distinct():
    rng = np.random.default_rng(1)
    raster = synthetic_raster("walls", 40, rng)
    pairs = random_pairs(raster, 5, rng)
    flat = raster.ravel()
    assert len(pairs) == 5
    for source, target in pairs:
        assert source != target
        assert flat[source] != 65535 and flat[target] != 65535


def test_pair_verdict_failed_flag_follows_the_verdict():
    verdict = PairVerdict("x", 0, 1, WITHIN_FLOAT32, 1e-9, 1e-4, 1.0, 1.0,
                          10, 10, "OK", None, 0.0)
    assert not verdict.failed
    verdict.verdict = SUBOPTIMAL
    assert verdict.failed

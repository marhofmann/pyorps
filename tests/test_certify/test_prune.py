"""Bound-and-prune against brute force (math.md M4): the result never depends on how tight the bound is."""

import numpy as np
import pytest

from pyorps.certify.prune import bound_and_prune, branch_lower_bound, prune_mask


def _instance(seed, n=400, branches=3, slack_scale=50.0, eps=0.5):
    rng = np.random.default_rng(seed)
    exact, lb = {}, {}
    for b in range(branches):
        e = rng.uniform(100.0, 1000.0, n)
        e[rng.random(n) < 0.1] = np.inf                       # infeasible positions
        slack = rng.uniform(0.0, slack_scale, n)              # LB <= exact: admissible, looser or tighter
        exact[b], lb[b] = e, np.maximum(e - slack, 0.0)
    return exact, lb, eps


def _brute(exact):
    best = min((float(v.min()), b, int(v.argmin())) for b, v in exact.items())
    return best


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("slack", [0.0, 5.0, 200.0, 5000.0])
def test_result_equals_brute_force_for_any_admissible_bound(seed, slack):
    exact, lb, eps = _instance(seed, slack_scale=slack)
    z_ub = float(max(v[np.isfinite(v)].max() for v in exact.values()))      # a poor incumbent
    res = bound_and_prune(lb, lambda b, i: exact[b][i], z_ub=z_ub, eps_adm=eps)
    z_best, b_best, i_best = _brute(exact)
    assert res.z_star == pytest.approx(z_best)
    assert res.argmin == (b_best, i_best)
    assert res.optimal and res.strict
    assert (b_best, i_best) in res.ties
    assert sum(res.kept.values()) <= sum(res.total.values())


def test_tighter_bound_prunes_more_but_gives_the_same_answer():
    exact, loose, eps = _instance(1, slack_scale=500.0)
    _, tight, _ = _instance(1, slack_scale=1.0)
    z_ub = 1e9
    a = bound_and_prune(loose, lambda b, i: exact[b][i], z_ub=z_ub, eps_adm=eps)
    t = bound_and_prune(tight, lambda b, i: exact[b][i], z_ub=z_ub, eps_adm=eps)
    assert a.z_star == t.z_star and a.argmin == t.argmin
    assert sum(t.kept.values()) < sum(a.kept.values())


def test_ties_are_listed_and_strictness_is_reported():
    exact = {0: np.array([10.0, 10.0, 12.0, 30.0])}
    lb = {0: np.array([10.0, 10.0, 11.0, 12.0])}
    res = bound_and_prune(lb, lambda b, i: exact[b][i], z_ub=10.0, eps_adm=0.0)
    assert res.z_star == 10.0 and res.ties == ((0, 0), (0, 1))
    assert res.strict                           # dropped positions have LB 11 and 12, both > Z* = 10
    assert res.optimal
    tied = bound_and_prune({0: np.array([10.0, 10.0, 12.0])}, lambda b, i: np.array([10.0, 10.0, 12.0])[i],
                           z_ub=10.0, eps_adm=0.0)
    assert tied.strict                          # LB 12 > 10
    eq = bound_and_prune({0: np.array([10.0, 10.0])}, lambda b, i: np.array([10.0, 10.0])[i], z_ub=10.0, eps_adm=0.0)
    assert eq.ties == ((0, 0), (0, 1)) and eq.optimal


def test_a_position_whose_bound_equals_the_incumbent_is_kept_so_ties_are_not_lost():
    keep = prune_mask(np.array([9.0, 10.0, 10.5, 11.0]), z_ub=10.0, eps_adm=0.5)
    assert keep.tolist() == [True, True, True, False]


def test_inadmissible_bound_stops_the_run():
    exact = {0: np.array([100.0, 200.0])}
    lb = {0: np.array([150.0, 150.0])}         # 150 > exact 100 at position 0
    with pytest.raises(AssertionError, match="not admissible"):
        bound_and_prune(lb, lambda b, i: exact[b][i], z_ub=1e9, eps_adm=1.0)


def test_f0_loop_lowers_the_incumbent_before_pruning():
    exact, lb, eps = _instance(3, slack_scale=20.0)
    res = bound_and_prune(lb, lambda b, i: exact[b][i], z_ub=1e9, eps_adm=eps)
    assert res.z_ub_history[0] == 1e9 and res.z_ub_history[-1] < 1e9
    assert list(res.z_ub_history) == sorted(res.z_ub_history, reverse=True)


def test_branch_lower_bound_sums_terms_and_refuses_nan_and_negative():
    out = branch_lower_bound(5.0, [np.array([1.0, np.inf]), np.array([2.0, 3.0])])
    assert out[0] == 8.0 and np.isinf(out[1])
    with pytest.raises(ValueError):
        branch_lower_bound(0.0, [np.array([np.nan])])
    with pytest.raises(ValueError):
        branch_lower_bound(0.0, [np.array([-1.0])])

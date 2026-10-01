"""Design B's CS-DW against design B's brute force, its pruning and the MILP.

Recorded 2026-09-23 (appendix Part IV, Design BUNDLE_B): CS-DW = brute force
at every root cell of 365 instances (max deviation 1.14e-13); budget pruning
exact on 55 instances; the strict physical model as an independent HiGHS
MILP equal to CS-DW on 55/55 instances without derating. The brute force
enumerates electrical designs x abstract trench topologies and embeds each
topology with a tree DP, so it shares no recursion with the CS-DW.
"""
import numpy as np
import pytest

from .oracles.b_brute import brute
from .oracles.b_csdw import aggregated_bounds, csdw, king_graph, mu_star, relaxed_dw
from .oracles.b_model import default_params

TOL = 1e-9


def _case(rng, R, C, n, lo=10.0, hi=20.0):
    cell = rng.uniform(lo, hi, size=(R, C))
    G = king_graph(R, C, cell)
    A = [int(x) for x in rng.choice(R * C, size=n, replace=False)]
    return G, A


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_csdw_equals_brute_force_with_joints(seed):
    """n = 3 on 4x4, joints allowed (q <= 2 = n - 1), every root cell."""
    rng = np.random.default_rng(seed)
    for it in range(2):
        G, A = _case(rng, 4, 4, 3)
        kw = dict(n=3, m_max=[None, 2, 3][it % 3], J=30.0, k_max=3,
                  sigma=0.4, bay=30.0, panel=0.3)
        mv, _ = csdw(G, A, default_params(**kw), r1=True)
        bf, _ = brute(G, A, default_params(**kw), q_max=2)
        fin = np.isfinite(bf)
        np.testing.assert_array_equal(np.isfinite(mv), fin)
        assert np.max(np.abs(mv[fin] - bf[fin])) <= TOL * np.max(bf[fin])


@pytest.mark.parametrize("seed", [11, 12])
def test_up_only_is_never_better(seed):
    rng = np.random.default_rng(seed)
    G, A = _case(rng, 4, 4, 3)
    kw = dict(n=3, m_max=None, J=30.0, k_max=3, sigma=0.4, bay=30.0, panel=0.3)
    mv, _ = csdw(G, A, default_params(**kw), r1=True)
    up, _ = csdw(G, A, default_params(**kw), with_down=False, r1=True)
    fin = np.isfinite(mv)
    assert np.all(up[fin] >= mv[fin] - TOL)


@pytest.mark.parametrize("seed", [5, 6])
def test_budget_pruning_is_exact_within_the_budget(seed):
    """prune_check.py: prune states with L + Out_k > B_max, B_max = the
    up-only upper bound; every root whose optimum is within B_max keeps its
    exact value."""
    rng = np.random.default_rng(seed)
    for it in range(3):
        G, A = _case(rng, 5, 5, 3, lo=15.0, hi=25.0)
        kw = dict(n=3, m_max=[None, 2, 3][it % 3], J=30.0, k_max=3,
                  sigma=0.4, bay=30.0, panel=0.3)
        up, _ = csdw(G, A, default_params(**kw), with_down=False, r1=True)
        ub = float(np.min(up))
        ms = mu_star(default_params(**kw))
        Out, _ = aggregated_bounds(G, A, default_params(**kw), ms)
        mv, _ = csdw(G, A, default_params(**kw), r1=True, outk=Out,
                     bmax=ub * (1 + 1e-12))
        bf, _ = brute(G, A, default_params(**kw), q_max=1)
        assert abs(float(np.min(mv)) - float(np.min(bf))) <= TOL * ub
        ok = np.isfinite(bf) & (bf <= ub)
        assert np.all(np.abs(mv[ok] - bf[ok]) <= TOL * ub)


@pytest.mark.parametrize("seed", [7, 8])
def test_corrected_mu_star_bound_is_admissible(seed):
    rng = np.random.default_rng(seed)
    G, A = _case(rng, 4, 4, 3)
    kw = dict(n=3, m_max=3, J=30.0, k_max=3, sigma=0.4, bay=30.0, panel=0.3)
    mv, _ = csdw(G, A, default_params(**kw), r1=True)
    lb, _ = relaxed_dw(G, A, default_params(**kw))
    fin = np.isfinite(mv)
    assert np.all(lb[fin] <= mv[fin] + TOL)


def test_strict_milp_equals_csdw_without_derating():
    """strict_milp.py on a 3x3 grid: without derating and with sigma below
    every trench rate, the strict physical model and D_share agree."""
    pytest.importorskip("scipy.optimize")
    from .oracles.b_strict_milp import strict_value
    rng = np.random.default_rng(0)
    for _ in range(2):
        cell = rng.uniform(10.0, 20.0, size=(3, 3))
        G = king_graph(3, 3, cell)
        cells = rng.choice(9, size=4, replace=False)
        A = [int(x) for x in cells[:3]]
        g = int(cells[3])
        kw = dict(n=3, m_max=None, J=1e9, k_max=3, sigma=0.4, bay=30.0,
                  panel=0.3, derate=(1.0,))
        mv, _ = csdw(G, A, default_params(**kw), r1=True)
        sv, status = strict_value(G, A, g, default_params(**kw), derate=False)
        assert status == 0
        assert abs(mv[g] - sv) <= 1e-7 * sv


@pytest.mark.slow
@pytest.mark.parametrize("seed", [21, 22, 23])
def test_csdw_equals_brute_force_four_turbines_no_joints(seed):
    rng = np.random.default_rng(seed)
    for it in range(3):
        G, A = _case(rng, 5, 5, 4)
        kw = dict(n=4, m_max=[None, 2, 3][it % 3], J=1e9, k_max=3,
                  sigma=0.4, bay=30.0, panel=0.3)
        mv, _ = csdw(G, A, default_params(**kw), r1=True)
        bf, _ = brute(G, A, default_params(**kw), q_max=0)
        fin = np.isfinite(bf)
        np.testing.assert_array_equal(np.isfinite(mv), fin)
        assert np.max(np.abs(mv[fin] - bf[fin])) <= TOL * np.max(bf[fin])


@pytest.mark.slow
@pytest.mark.parametrize("derate", [False, True])
def test_strict_milp_many(derate):
    """Without derating: equal. With derating: D_share may be LOWER (the
    parallel-trench escape, scope sentence ii), never higher."""
    from .oracles.b_strict_milp import strict_value
    rng = np.random.default_rng(3 if derate else 2)
    for _ in range(6):
        cell = rng.uniform(10.0, 20.0, size=(3, 4))
        G = king_graph(3, 4, cell)
        cells = rng.choice(12, size=4, replace=False)
        A = [int(x) for x in cells[:3]]
        g = int(cells[3])
        dr = (1.0, 0.85, 0.75, 0.7, 0.65) if derate else (1.0,)
        kw = dict(n=3, m_max=None, J=1e9, k_max=3, sigma=0.4, bay=30.0,
                  panel=0.3, derate=dr)
        mv, _ = csdw(G, A, default_params(**kw), r1=True)
        sv, status = strict_value(G, A, g, default_params(**kw), derate=derate)
        if status != 0:
            continue
        if derate:
            assert mv[g] <= sv + 1e-7 * sv
        else:
            assert abs(mv[g] - sv) <= 1e-7 * sv

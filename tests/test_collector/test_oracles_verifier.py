"""The adversarial verifier's checks (appendix Part IV, "Adversarial verification").

* xcheck: design A's DP and design B's DP in one common model -- 0
  differences over 108 682 roots were recorded;
* the physical-model oracle (one trench per used step, cycles allowed):
  DP <= best forest design always (completeness), and with no derating and
  sigma <= min c/l also DP >= strict optimum (soundness);
* the hand-built cases that shaped the model: the non-additive kiosk
  (119.1 with the Ja operation, 120.2 without), split-and-rejoin (the tree
  rule's price when condition (C) fails), and the inadmissible bay term
  of the mu*-DW (41.0 against an optimum of 23.0).
"""
import random

import numpy as np
import pytest

from .oracles import a_dp, a_model, b_csdw, b_model
from .oracles.a_brute import m1_cost
from .oracles.a_trace import reprice, traceback
from .oracles.adapters import (
    TYPESETS,
    BParamsA,
    delta_max,
    make_graph,
    rand_case,
    rand_case_strict,
    run_A_field,
    run_B_field,
)
from .oracles.v_strict import Model

INF = float("inf")
TOL = 1e-9


# ------------------------------------------------------------- xcheck


@pytest.mark.parametrize("seed,conv", [(1, "A"), (2, "B")])
def test_design_a_and_design_b_agree_everywhere(seed, conv):
    rng = random.Random(seed)
    roots = 0
    for _ in range(40):
        case = rand_case(rng, 3)
        mvA, _ = run_A_field(case["N"], case["edges"], case["turb"],
                             case["prm"], conv)
        mvB, _, _ = run_B_field(case["N"], case["edges"], case["turb"],
                                case["prm"], conv)
        finA, finB = np.isfinite(mvA), np.isfinite(mvB)
        np.testing.assert_array_equal(finA, finB)
        if finA.any():
            assert np.max(np.abs(mvA[finA] - mvB[finA])) <= 1e-7
        roots += int(finA.sum())
    assert roots > 100


@pytest.mark.slow
def test_design_a_and_design_b_agree_four_turbines():
    rng = random.Random(3)
    for _ in range(15):
        case = rand_case(rng, 4)
        mvA, _ = run_A_field(case["N"], case["edges"], case["turb"],
                             case["prm"], "A")
        mvB, _, _ = run_B_field(case["N"], case["edges"], case["turb"],
                                case["prm"], "A")
        fin = np.isfinite(mvA)
        np.testing.assert_array_equal(fin, np.isfinite(mvB))
        if fin.any():
            assert np.max(np.abs(mvA[fin] - mvB[fin])) <= 1e-7


# --------------------------------------------------- physical-model oracle


@pytest.mark.parametrize("seed,mode", [(11, "nc0"), (12, "gen")])
def test_dp_against_the_physical_oracle(seed, mode):
    rng = random.Random(seed)
    roots = 0
    for _ in range(8):
        case = rand_case_strict(rng, 3, mode == "nc0")
        prm = case["prm"]
        mv, _, _ = run_B_field(case["N"], case["edges"], case["turb"], prm, "A")
        M = Model(case["N"], case["edges"], case["turb"], prm["types"],
                  prm["derate"], prm["sigma"], prm["J"], prm["bay"],
                  prm["panel"], prm["kmax"], prm["mmax"], root_transit=True,
                  joint_at_root=True, qmax=2)
        free = [g for g in range(case["N"]) if g not in case["turb"]]
        for g in rng.sample(free, min(2, len(free))):
            strict, forest = M.solve(g)
            dp = mv[g]
            roots += 1
            # completeness: every forest design is a D_share design
            assert dp <= forest + TOL
            if mode == "nc0" and prm["sigma"] <= case["minrate"] + 1e-12:
                # soundness: merging co-located trenches never costs more
                assert dp >= strict - TOL
    assert roots >= 10


def test_condition_c_makes_the_tree_rule_lossless():
    """cond_c.py: sigma + max derating premium <= every trench rate ==>
    D_share equals the physical optimum."""
    rng = random.Random(21)
    checked = 0
    tries = 0
    while checked < 6 and tries < 200:
        tries += 1
        types = rng.choice(TYPESETS)
        f2 = rng.choice([1.0, 0.9, 0.8]); f3 = f2 * rng.choice([1.0, 0.9])
        derate = [1.0, f2, f3, f3, f3, f3, f3, f3]
        sigma = rng.choice([0.0, 0.05, 0.2])
        dm = delta_max(types, derate, 3, sigma, 5)
        if dm == INF:
            continue
        R, C = rng.choice([(3, 3), (2, 4), (2, 3)])
        base = dm * rng.choice([1.0, 1.05, 1.5])
        cell = [[base * rng.uniform(1.0, rng.choice([1.0, 2.0, 5.0]))
                 for _ in range(C)] for _ in range(R)]
        N, edges = make_graph(R, C, cell, rng.random() < 0.25)
        if min(c / l for (_, _, c, l) in edges) < dm - 1e-12:
            continue
        turb = rng.sample(range(N), 3)
        prm = dict(types=types, derate=derate, sigma=sigma,
                   J=rng.choice([0.0, 0.5 * base, 2 * base, 1e6]),
                   bay=rng.choice([0.0, base, 4 * base]), panel=0.0,
                   kmax=rng.choice([1, 2, 3]), mmax=None)
        mv, _, _ = run_B_field(N, edges, turb, prm, "A")
        M = Model(N, edges, turb, types, derate, sigma, prm["J"], prm["bay"],
                  0.0, prm["kmax"], None, root_transit=True,
                  joint_at_root=True, qmax=2)
        g = rng.choice([v for v in range(N) if v not in turb])
        strict, _forest = M.solve(g)
        assert abs(mv[g] - strict) <= TOL * max(1.0, strict) or (
            mv[g] == INF and strict == INF)
        checked += 1
    assert checked >= 3


# ------------------------------------------------------ hand-built cases


def test_kiosk_needs_the_ja_operation():
    """Non-additive junction prices (J_T = 5 at degree 3, J_K = 6 at 4):
    the optimum is a kiosk at a whose output goes UP while one input comes
    DOWN from b. Design A's DP, its re-price and the physical oracle all give
    119.1; the best structure without Ja (kiosk at b fed by three separate
    systems) costs 120.2."""
    # cells: 0=g 1=b 2=a 3=t1 4=t2 5=t3 6=t4 7=spare
    E = [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (2, 3, 1.0, 1.0), (2, 4, 1.0, 1.0),
         (2, 5, 1.0, 1.0), (6, 1, 1.0, 1.0), (7, 0, 1.0, 1.0), (7, 6, 3.0, 1.0)]
    turb = [3, 4, 5, 6]
    JT, JK = 5.0, 6.0
    types = [(10.0, 1.0, 0.0)]
    I = a_model.Inst(n=8, edges=E, turb=turb, root=0, JT=JT, JK=JK, dmax=5,
                     mmax=99, types=types, derate=[1.0] * 9, sigma=0.1,
                     bay=100.0, panel=0.0, kmax=1)
    dp = a_dp.ShareDP(I)
    v = dp.solve()
    Tn, Te, En, rn = traceback(dp)
    rp, _ = reprice(I, Tn, Te, En, rn)
    assert v == pytest.approx(119.1, abs=1e-9)
    assert rp == pytest.approx(119.1, abs=1e-9)
    jf = lambda d_in: JT if d_in == 2 else (JK if d_in <= 4 else INF)
    M = Model(8, E, turb, types, [1.0] * 8, 0.1, 0.0, 100.0, 0.0, 1, None,
              root_transit=False, joint_at_root=False, qmax=1, jfun=jf)
    strict, forest = M.solve(0)
    assert strict == pytest.approx(119.1, abs=1e-9)
    assert forest == pytest.approx(119.1, abs=1e-9)
    # kiosk at b fed by three separate systems, no Ja: 6 trench steps,
    # the priced walks, bay 100 and J_K = 6.
    arcs = [(1, [3, 2, 1]), (2, [4, 2, 1]), (4, [5, 2, 1]), (8, [6, 1]),
            (15, [1, 0])]
    assert m1_cost(I, arcs, 100.0 + JK) == pytest.approx(120.2, abs=1e-9)


@pytest.mark.parametrize("label,prm,cL,cE,share,strict,forest", [
    ("sigma premium", dict(sigma=1.0), 0.2, 5.0, 28.6, 26.2, 28.6),
    ("derating only", dict(sigma=0.1, types=[(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)],
                           derate=[1.0, 0.8] + [0.8] * 6), 0.6, 5.0,
     33.7, 30.8, 35.3),
    ("condition (C)", dict(sigma=0.1, types=[(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)],
                           derate=[1.0, 0.8] + [0.8] * 6), 2.5, 5.0,
     41.0, 41.0, 41.0),
])
def test_split_and_rejoin(label, prm, cL, cE, share, strict, forest):
    """Two turbines, two cheap lanes between expensive shared ends. When
    sigma or the derating premium exceeds the lane trench rate, splitting
    and rejoining (a cycle) beats every tree design; under condition (C)
    all three agree. This is why scope sentence (i) and the Stage-5 strict
    re-price exist."""
    E = [(0, 2, 0.5, 1.0), (1, 2, 0.5, 1.0), (2, 3, cE, 1.0),
         (3, 4, cL, 1.0), (4, 5, cL, 1.0), (5, 8, cL, 1.0),
         (3, 6, cL, 1.0), (6, 7, cL, 1.0), (7, 8, cL, 1.0),
         (8, 9, cE, 1.0)]
    base = dict(types=[(10.0, 1.0, 0.0)], derate=[1.0] * 8, J=1e6, bay=0.0,
                panel=0.0, kmax=1, mmax=None)
    p = dict(base, **prm)
    mvA, _ = run_A_field(10, E, [0, 1], p, "A")
    mvB, _, _ = run_B_field(10, E, [0, 1], p, "A")
    M = Model(10, E, [0, 1], p["types"], p["derate"], p["sigma"], p["J"],
              p["bay"], p["panel"], p["kmax"], p["mmax"], root_transit=True,
              joint_at_root=True, qmax=1)
    st, fo = M.solve(9)
    assert mvA[9] == pytest.approx(share, abs=1e-9), label
    assert mvB[9] == pytest.approx(share, abs=1e-9), label
    assert st == pytest.approx(strict, abs=1e-9), label
    assert fo == pytest.approx(forest, abs=1e-9), label


def test_naive_mu_star_bay_term_is_inadmissible_and_the_fix_is_not():
    """lb_bay.py: k_cap(1) = 1 < k_bay = 3 and a joint on the UW cell may
    merge feeders into one bay, so the optimum uses fewer bays than
    ceil(n / min(k_cap(1), k_bay)) assumes."""
    cell = [[1.0] * 3 for _ in range(3)]
    N, edges = make_graph(3, 3, cell, False)
    turb, g = [0, 2, 6], 4
    types = [(1.0, 1.0, 0.0)]
    kw = dict(I1=1.0, derate=(1.0,) * 8, lossc=1.0, sigma=0.0, omega=0.0,
              J=1.0, bay=10.0, panel=0.0, k_max=1, m_max=None, m_cap_T=9,
              max_in=2, k_bay=3)
    G = b_csdw.Graph(N, edges)
    mv, _ = b_csdw.csdw(G, turb, b_model.Params(3, types, **kw))
    naive, _ = b_csdw.relaxed_dw(G, turb, b_model.Params(3, types, **kw),
                                 bay_term="naive")
    fixed, _ = b_csdw.relaxed_dw(G, turb, b_model.Params(3, types, **kw))
    M = Model(N, edges, turb, types, [1.0] * 8, 0.0, 1.0, 10.0, 0.0, 1, None,
              root_transit=True, joint_at_root=True, qmax=2)
    strict, forest = M.solve(g)
    assert mv[g] == pytest.approx(23.0, abs=1e-9)
    assert strict == pytest.approx(23.0, abs=1e-9)
    assert forest == pytest.approx(23.0, abs=1e-9)
    assert naive[g] == pytest.approx(41.0, abs=1e-9)
    assert fixed[g] <= mv[g] + TOL


# ------------------------------------------------------ lower bounds


def test_lower_bounds_are_admissible():
    """lb_check.py: design B's mu*-DW (corrected bay term) <= CS-DW at every
    root; design A's rho_hat-DW <= A's DP in A's native model."""
    rng = random.Random(31)
    for _ in range(10):
        R, C = rng.choice([(3, 3), (2, 4), (3, 4)])
        cell = [[round(rng.uniform(0.3, 3.0), 3) for _ in range(C)]
                for _ in range(R)]
        N, edges = make_graph(R, C, cell, rng.random() < 0.5)
        turb = rng.sample(range(N), 3)
        types = rng.choice(TYPESETS)
        f2 = rng.choice([1.0, 0.8, 0.6])
        derate = [1.0, f2, f2 * 0.9] + [f2 * 0.9] * 5
        prm = dict(sigma=rng.choice([0.0, 0.1, 0.5]),
                   J=rng.choice([0.0, 0.5, 2.0]),
                   bay=rng.choice([2.0, 6.0, 20.0]),
                   panel=rng.choice([0.0, 0.1]), kmax=rng.choice([2, 3]),
                   mmax=rng.choice([None, 2, 3]))

        def params():
            return BParamsA(3, types, I1=1.0, derate=tuple(derate), lossc=1.0,
                            sigma=prm["sigma"], omega=0.0, J=prm["J"],
                            bay=prm["bay"], panel=prm["panel"],
                            k_max=prm["kmax"], m_max=prm["mmax"], m_cap_T=9,
                            max_in=2, k_bay=3)
        G = b_csdw.Graph(N, edges)
        mv, _ = b_csdw.csdw(G, turb, params(), with_down=True)
        lb, _ = b_csdw.relaxed_dw(G, turb, params())
        fin = np.isfinite(mv)
        assert np.all(lb[fin] <= mv[fin] + TOL)
        g = rng.choice([v for v in range(N) if v not in turb])
        I = a_model.Inst(n=N, edges=edges, turb=turb, root=g,
                         sigma=prm["sigma"], JT=prm["J"], JK=2 * prm["J"],
                         dmax=4, bay=prm["bay"], panel=prm["panel"],
                         kmax=prm["kmax"], mmax=prm["mmax"] or 99,
                         types=types, derate=[1.0] + derate)
        v = a_dp.ShareDP(I).solve()
        if v < INF:
            assert a_dp.relaxed_lb(I) <= v + TOL


# ------------------------------------------------ generality of design A


@pytest.mark.parametrize("seed", [41, 42])
def test_heterogeneous_turbines(seed):
    """hetero.py: design A with different turbine powers against the
    physical oracle: A' <= best forest; no derating and sigma <= min c/l
    gives A' >= strict."""
    rng = random.Random(seed)
    for _ in range(4):
        R, C = rng.choice([(3, 3), (2, 4), (2, 3)])
        cell = [[round(rng.uniform(0.3, 3.0), 3) for _ in range(C)]
                for _ in range(R)]
        N, edges = make_graph(R, C, cell, rng.random() < 0.25)
        turb = rng.sample(range(N), 3)
        P = [rng.choice([0.5, 0.8, 1.0, 1.3, 2.0]) for _ in range(3)]
        nc0 = rng.random() < 0.5
        minrate = min(c / l for (_, _, c, l) in edges)
        if nc0:
            derate = [1.0] * 8; mmax = None
            sig = rng.choice([0.0, 0.5, 1.0]) * minrate
        else:
            f2 = rng.choice([0.9, 0.7, 0.5])
            derate = [1.0, f2, f2 * 0.9] + [f2 * 0.9] * 5
            mmax = rng.choice([None, 2]); sig = rng.choice([0.0, 0.5, 2.0]) * minrate
        types = rng.choice(TYPESETS)
        J = rng.choice([0.0, 1.0, 1e6]); bay = rng.choice([0.5, 4.0])
        kmax = rng.choice([2, 3])
        g0 = [v for v in range(N) if v not in turb][0]
        I = a_model.Inst(n=N, edges=edges, turb=turb, root=g0, sigma=sig,
                         JT=J, JK=2 * J, dmax=4, bay=bay, panel=0.0,
                         kmax=kmax, mmax=mmax or 99, types=types,
                         derate=[1.0] + derate, P=P)
        dp = a_dp.ShareDP(I, root_transit=True)
        dp.solve()
        mv = np.full(N, INF)
        for L, arr in dp.F.items():
            if L[0] == I.full and not L[2]:
                np.minimum(mv, np.array(arr) + bay * len(L[1]), out=mv)
        M = Model(N, edges, turb, types, derate, sig, J, bay, 0.0, kmax,
                  mmax, P=P, root_transit=True, joint_at_root=True, qmax=2)
        for g in rng.sample([v for v in range(N) if v not in turb], 2):
            s, f = M.solve(g)
            assert mv[g] <= f + TOL
            if nc0 and sig <= minrate + 1e-12:
                assert mv[g] >= s - TOL


@pytest.mark.parametrize("seed", [51, 52])
def test_design_a_native_model_with_non_additive_joints(seed):
    """native_a.py: root no-transit, no joint on the UW cell, J_T/J_K
    non-additive, heterogeneous powers, against the physical oracle."""
    rng = random.Random(seed)
    for _ in range(5):
        R, C = rng.choice([(3, 3), (2, 4), (2, 3)])
        cell = [[round(rng.uniform(0.3, 3.0), 3) for _ in range(C)]
                for _ in range(R)]
        N, edges = make_graph(R, C, cell, rng.random() < 0.25)
        turb = rng.sample(range(N), 3)
        g = rng.choice([v for v in range(N) if v not in turb])
        P = [rng.choice([0.5, 1.0, 1.3, 2.0]) for _ in range(3)]
        nc0 = rng.random() < 0.5
        minrate = min(c / l for (_, _, c, l) in edges)
        if nc0:
            derate = [1.0] * 8; mmax = None
            sig = rng.choice([0.0, 0.5, 1.0]) * minrate
        else:
            f2 = rng.choice([0.9, 0.7, 0.5])
            derate = [1.0, f2, f2 * 0.9] + [f2 * 0.9] * 5
            mmax = rng.choice([None, 2]); sig = rng.choice([0.0, 0.5, 2.0]) * minrate
        types = rng.choice(TYPESETS)
        JT = rng.choice([0.0, 1.0, 3.0, 1e6]); JK = JT + rng.choice([0.0, 0.5, 4.0])
        bay = rng.choice([0.5, 4.0]); kmax = rng.choice([2, 3])
        I = a_model.Inst(n=N, edges=edges, turb=turb, root=g, sigma=sig,
                         JT=JT, JK=JK, dmax=4, bay=bay, panel=0.0, kmax=kmax,
                         mmax=mmax or 99, types=types, derate=[1.0] + derate,
                         P=P)
        d = a_dp.ShareDP(I).solve()
        jf = lambda k, JT=JT, JK=JK: JT if k == 2 else (JK if k == 3 else INF)
        M = Model(N, edges, turb, types, derate, sig, 0.0, bay, 0.0, kmax,
                  mmax, P=P, root_transit=False, joint_at_root=False, qmax=2,
                  jfun=jf)
        s, f = M.solve(g)
        assert d <= f + TOL
        if nc0 and sig <= minrate + 1e-12:
            assert d >= s - TOL

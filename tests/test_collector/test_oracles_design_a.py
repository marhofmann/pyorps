"""Design A's cut-state DW against design A's exhaustive D_share enumerator.

Recorded 2026-09-23 (appendix Part IV, Design BUNDLE_A): DP = enumerator on
1 524 of 1 530 instances at K = 2; the other 6 were typed designs needing
three parallel trenches on one cell, which K = 2 cannot represent, with the
DP BELOW the enumerator. Every DP witness re-priced exactly, and the relaxed
lower bound never exceeded the DP. These tests re-run that evidence on
fixed seeds.
"""
import math
import random

import pytest

from .oracles import a_dp, a_gen
from .oracles.a_brute import BruteM2
from .oracles.a_model import INF
from .oracles.a_trace import reprice, traceback

TOL = 1e-9


def _instances(seed, count, kinds=("grid", "rand", "spur")):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        kind = kinds[i % len(kinds)]
        if kind == "spur":
            out.append(a_gen.spur_inst(rng, k=3))
        else:
            out.append(a_gen.rand_inst(rng, kind=kind, k=3))
    return out


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_dp_equals_the_enumerator(seed):
    checked = 0
    for inst in _instances(seed, 6):
        dp = a_dp.ShareDP(inst)
        v = dp.solve()
        brute = BruteM2(inst, K=2, max_trees=300_000)
        b = brute.solve()
        if brute.truncated:
            continue
        if v == INF or b == INF:
            assert v == b == INF
            continue
        assert abs(v - b) <= TOL * max(1.0, abs(b)), (seed, v, b)
        checked += 1
    assert checked >= 3


@pytest.mark.slow
@pytest.mark.parametrize("seed", [21, 22])
def test_typed_dp_never_exceeds_the_enumerator(seed):
    """The typed variant may use three parallel trenches on a cell, which
    K = 2 cannot build, so only DP <= enumerator is a theorem here."""
    for inst in _instances(seed, 4, kinds=("grid", "spur")):
        v = a_dp.ShareDP(inst, typed=True).solve()
        brute = BruteM2(inst, K=2, typed=True, max_trees=200_000)
        b = brute.solve()
        if brute.truncated or b == INF:
            continue
        assert v <= b + TOL * max(1.0, abs(b))


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_witness_reprices_exactly_and_lb_is_admissible(seed):
    feasible = 0
    for inst in _instances(seed, 8):
        dp = a_dp.ShareDP(inst)
        v = dp.solve()
        if v == INF:
            continue
        feasible += 1
        Tn, Te, En, rn = traceback(dp)
        cost, _info = reprice(inst, Tn, Te, En, rn)
        assert math.isclose(cost, v, rel_tol=1e-12, abs_tol=1e-12)
        assert a_dp.relaxed_lb(inst) <= v + TOL
    assert feasible >= 4


def test_up_only_dp_is_never_better_than_the_exact_dp():
    """The partition-state (up-only) class is a restriction of D_share."""
    rng = random.Random(41)
    worse = 0
    for _ in range(12):
        inst = a_gen.spur_inst(rng, k=3)
        exact = a_dp.ShareDP(inst).solve()
        up = a_dp.ShareDP(inst, allow_down=False).solve()
        assert up >= exact - TOL
        worse += up > exact + TOL
    # Spur geometries are where the loop-in spur matters (appendix: 115 of 247).
    assert worse >= 1


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(101, 106))
def test_dp_equals_the_enumerator_many(seed):
    for inst in _instances(seed, 20):
        v = a_dp.ShareDP(inst).solve()
        brute = BruteM2(inst, K=2, max_trees=1_000_000)
        b = brute.solve()
        if brute.truncated:
            continue
        if v == INF or b == INF:
            assert v == b == INF
            continue
        assert abs(v - b) <= TOL * max(1.0, abs(b))

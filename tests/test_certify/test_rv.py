"""Relax-and-verify, the consent lattice (plan section 6, oracle 14) and the
epsilon manifest."""
import json
import math
import random

import pytest

from pyorps.certify.rv import (
    EpsilonManifest,
    consent_lattice,
    relax_and_verify,
)


def _candidates(seed, n=60):
    rng = random.Random(seed)
    exact = {k: rng.uniform(100, 200) for k in range(n)}
    # a few exact ties with the minimum
    m = min(exact, key=exact.get)
    for k in rng.sample(sorted(exact), 3):
        exact[k] = exact[m]
    relaxed = {k: v - rng.uniform(0, 15) for k, v in exact.items()}
    return exact, relaxed


@pytest.mark.parametrize("seed", range(5))
def test_rv_finds_the_optimum_and_every_tie(seed):
    exact, relaxed = _candidates(seed)
    calls = []

    def verify(k):
        calls.append(k)
        return exact[k], exact[k]

    res = relax_and_verify(relaxed.items(), verify, eps=0.0)
    best = min(exact.values())
    assert res.optimal and res.z_ub == best and res.z_lb >= best
    assert set(res.ties) == {k for k, v in exact.items() if v == best}
    # it stopped early: everything not verified has F_rel > z_ub
    skipped = set(exact) - set(calls)
    assert skipped and all(relaxed[k] > best for k in skipped)


def test_rv_escalates_open_brackets_and_refuses_to_report_one():
    exact, relaxed = _candidates(7, n=20)

    def verify(k):
        return exact[k] - 5.0, exact[k] + 5.0

    def escalate(k, bracket):
        lo, up = bracket
        mid_lo = lo + (exact[k] - lo) / 2
        mid_up = up - (up - exact[k]) / 2
        return (exact[k], exact[k]) if up - lo < 1.0 else (mid_lo, mid_up)

    res = relax_and_verify(relaxed.items(), verify, eps=0.0,
                           escalate=escalate, max_escalations=16)
    assert res.optimal and res.z_ub == min(exact.values())
    with pytest.raises(RuntimeError, match="never a result"):
        relax_and_verify(relaxed.items(), verify, eps=0.0)


def test_rv_catches_a_relaxation_that_is_not_a_lower_bound():
    with pytest.raises(ValueError, match="not a lower bound"):
        relax_and_verify([("a", 10.0)], lambda k: (5.0, 5.0))


def test_rv_ties_within_eps_are_enumerated():
    cands = [("a", 9.0), ("b", 10.05)]
    vals = {"a": 10.0, "b": 10.05}
    res = relax_and_verify(cands, lambda k: (vals[k], vals[k]), eps=0.1)
    assert set(res.ties) == {"a", "b"}
    res = relax_and_verify(cands, lambda k: (vals[k], vals[k]), eps=0.0)
    assert res.ties == ["a"] and res.next_relaxed == 10.05


@pytest.mark.parametrize("seed", range(8))
def test_consent_lattice_equals_brute_force_over_designs(seed):
    """Oracle 14: designs touch permitting classes; the lattice optimum
    equals the best design including its one-off costs and delay."""
    rng = random.Random(seed)
    K = ["FFH", "LSG", "forest", "water"][:rng.choice([2, 3, 4])]
    designs = []
    for _ in range(40):
        touched = frozenset(k for k in K if rng.random() < 0.35)
        designs.append((rng.uniform(100, 140) - 6 * len(touched), touched))
    C = {k: rng.choice([0.0, 2.0, 5.0, 12.0]) for k in K}
    delay_k = {k: rng.choice([0.0, 1.0, 4.0]) for k in K}

    def V(S):               # longest parallel chain: monotone in S
        return max((delay_k[k] for k in S), default=0.0)

    runs = []

    def masked(S):
        runs.append(S)
        return min((c for c, T in designs if T <= S), default=math.inf)

    res = consent_lattice(K, C, masked, delay=V)
    brute = min(c + sum(C[k] for k in T) + V(T) for c, T in designs)
    assert res.value == pytest.approx(brute)
    assert len(runs) <= 2 ** len(K)
    assert len(runs) + len(res.skipped) == 2 ** len(K)
    for S, lb in res.skipped.items():       # every skipped bound is sound
        exact = masked(S) + sum(C[k] for k in S) + V(S)
        assert lb <= exact + 1e-9
        assert lb > res.value


def test_consent_lattice_rejects_bad_inputs():
    with pytest.raises(ValueError, match=">= 0"):
        consent_lattice(["a"], {"a": -1.0}, lambda S: 1.0)
    with pytest.raises(ValueError, match="cannot make the optimum cheaper"):
        consent_lattice(["a"], {"a": 0.0},
                        lambda S: 10.0 if S else 5.0)


def test_epsilon_manifest():
    m = EpsilonManifest()
    m.add("float32 cost_factor", 2.6e-8 * 1e7, source="plan 3.7, V4-12")
    m.add("access weight scaling", 0.5, source="COMPUTE-11")
    m.add("storage on argmin chain", 40.0, kind="E", source="M3-13")
    assert m.eps_adm == pytest.approx(0.26 + 0.5)
    assert m.eps_E == 40.0
    json.dumps(m.as_record())
    with pytest.raises(ValueError, match=">= 0"):
        m.add("bad", -1.0)
    with pytest.raises(ValueError, match="already"):
        m.add("access weight scaling", 0.1)

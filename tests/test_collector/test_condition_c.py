"""Condition (C): sigma + the worst group-derating premium against the cheapest
trench rate (plan rev. 5, section 2.1 (i)); the check printed with every result."""
import itertools
import math
import random

import pytest

from pyorps.collector.condition_c import condition_c
from pyorps.collector.model import CableType, CollectorModel, bits, set_partitions


def _model(n=2, *, types, derating=(1.0,), sigma=0.0, current=100.0,
           m_max=4):
    return CollectorModel.identical_turbines(
        n, current_one_a=current, loss_one=0.0, types=tuple(types),
        derating=tuple(derating), sigma_eur_per_m=sigma, m_max=m_max)


def _ct(name, amp, cost):
    return CableType(name, amp, cost, 0.0)


def test_sigma_alone_is_the_premium_without_derating():
    m = _model(3, types=[_ct("a", 1000.0, 1.0)], sigma=0.1)
    c = condition_c(m, 5.0)
    assert c.delta == pytest.approx(0.1)
    assert c.holds and c.margin == pytest.approx(4.9)


def test_derating_forces_a_thicker_cable_and_adds_its_premium():
    # one turbine: 100 A. Alone, the thin cable (150 A, 1 EUR/m) is enough.
    # Two systems in one trench derate to 0.5: the thin one carries 75 A (too
    # little), the thick one (300 A, 3 EUR/m) carries 150 A.
    m = _model(2, types=[_ct("thin", 150.0, 1.0), _ct("thick", 300.0, 3.0)],
               derating=(1.0, 0.5), sigma=0.1)
    c = condition_c(m, 10.0)
    # r({t1,t2} in one trench) = 0.1 + 3 + 3; alone 1 each => delta = 0.1 + 4
    assert c.delta == pytest.approx(4.1)
    assert c.holds and c.margin == pytest.approx(5.9)
    assert not condition_c(m, 4.0).holds


def test_a_merge_that_makes_a_system_infeasible_is_inf():
    m = _model(2, types=[_ct("thin", 150.0, 1.0), _ct("thick", 120.0, 3.0)],
               derating=(1.0, 0.5), sigma=0.0)
    c = condition_c(m, 1e9)
    assert math.isinf(c.delta) and not c.holds


def test_report_text_states_the_margin_and_the_consequence():
    m = _model(2, types=[_ct("a", 1000.0, 1.0)], sigma=0.1)
    ok = str(condition_c(m, 5.0))
    assert "holds" in ok and "margin 4.9" in ok
    bad = str(condition_c(m, 0.05))
    assert "FAILS" in bad and "strict one-trench-per-cell re-price" in bad
    rec = condition_c(m, 5.0).as_record()
    assert rec["holds"] is True and rec["groups"] >= 1


def test_negative_rate_is_refused():
    with pytest.raises(ValueError):
        condition_c(_model(2, types=[_ct("a", 1000.0, 1.0)]), -1.0)


def test_parallel_cables_let_a_big_system_share_a_derated_trench():
    # 100 A per turbine, one cable type of 300 A at 1 EUR/m, derating 0.6 from
    # two conductors on. The system {t1, t2} (200 A) is fine alone, but beside
    # {t3} it needs two parallel cables (0.6 * 2 * 300 = 360 A), so merging
    # costs one more cable: delta = 1 EUR/m. With single cables only, the
    # merge is impossible (inf), which is what the check must not confuse with
    # a model that has parallel cables.
    kw = dict(current_one_a=100.0, loss_one=0.0, types=(_ct("a", 300.0, 1.0),),
              derating=(1.0, 0.6, 0.6, 0.6), m_max=4)
    single = CollectorModel.identical_turbines(3, p_max=1, **kw)
    assert math.isinf(condition_c(single, 1e9).delta)
    par = CollectorModel.identical_turbines(3, p_max=2, **kw)
    c = condition_c(par, 1e9)
    assert c.delta == pytest.approx(1.0) and c.holds


def _brute_delta(model):
    """An independent evaluation: every cable assignment, ``model.rate``."""
    opts = [(ti, p) for ti in range(len(model.types))
            for p in range(1, model.p_max + 1)]

    def best(masks):
        out = math.inf
        for assign in itertools.product(opts, repeat=len(masks)):
            out = min(out, model.rate(list(zip(masks, assign))))
        return out

    worst = 0.0
    for union in range(1, 1 << model.n):
        for blocks in set_partitions(list(bits(union))):
            masks = [sum(1 << i for i in b) for b in blocks]
            k = len(masks)
            if k < 2 or k > min(model.m_cap(), model.n):
                continue
            for r in range(1, k):
                for ia in itertools.combinations(range(k), r):
                    a = [masks[i] for i in ia]
                    b = [masks[i] for i in range(k) if i not in ia]
                    ra, rb, rab = best(a), best(b), best(masks)
                    if math.isinf(ra) or math.isinf(rb):
                        continue
                    worst = math.inf if math.isinf(rab) else max(
                        worst, rab - ra - rb)
    return worst


@pytest.mark.parametrize("seed", range(6))
def test_matches_a_brute_force_over_all_cable_assignments(seed):
    """Options (type, p): the optimum over every assignment equals the DP."""
    rng = random.Random(seed)
    n = 3
    types = [_ct(f"t{i}", rng.choice([120.0, 200.0, 350.0, 600.0]),
                 rng.choice([1.0, 2.0, 4.0, 7.0])) for i in range(3)]
    f2 = rng.choice([1.0, 0.9, 0.7])
    f3 = f2 * rng.choice([1.0, 0.8])
    m = CollectorModel.identical_turbines(
        n, current_one_a=100.0, loss_one=0.0, types=tuple(types),
        derating=(1.0, f2, f3, f3), sigma_eur_per_m=rng.choice([0.0, 0.2, 1.0]),
        m_max=rng.choice([2, 3, 4]), p_max=rng.choice([1, 2]))
    got = condition_c(m, 100.0).delta
    want = _brute_delta(m)
    if math.isinf(want):
        assert math.isinf(got)
    else:
        assert got == pytest.approx(want, abs=1e-9)

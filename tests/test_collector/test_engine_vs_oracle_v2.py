"""Engine B against the independent oracle for the 2026-09-24 rules.

``oracles/share_brute_v2.py`` enumerates every ``D_share`` design on tiny
graphs -- trench trees in a K-copy expansion, electrical templates with
junction slots, every cable option per system -- and prices them with the
model's own ``rate``/``turbine_ok``/``bay_ok``. It was written from the
model specification alone, without the engine, and knows nothing of the
candidate-size rule. Here it checks what the ported 2026-09-23 oracles
cannot: stations with a building paid once per trench node, parallel
cables (p panels and bays, p toward m), heterogeneous turbines, bay
current limits, with and without root transit.

The enumerator places at most K trench nodes per cell, so the engine may
find a design it cannot build; such a root is re-checked at K + 1.
Measured 2026-09-24: 80 of 80 roots equal at K = 2 (seed 11, 40 instances).
"""
import math
import random

import pytest

from pyorps.collector import (
    CableType,
    CollectorGraph,
    CollectorModel,
    reprice,
    solve_collector,
)

from .oracles.share_brute_v2 import brute_share

INF = math.inf


def _grid(R, C, rng, diag):
    idx = lambda r, c: r * C + c  # noqa: E731
    E = []
    for r in range(R):
        for c in range(C):
            steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if diag else [])
            for dr, dc in steps:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < R and 0 <= c2 < C:
                    ln = math.hypot(dr, dc)
                    E.append((idx(r, c), idx(r2, c2),
                              round(rng.uniform(0.3, 3.0) * ln, 3), ln))
    return R * C, E


def _model(rng, n):
    types = tuple(CableType(f"t{i}", rng.choice([1.0, 1.5, 2.0, 3.0]),
                            rng.choice([0.3, 0.6, 1.0, 1.5]),
                            rng.choice([0.0, 0.05, 0.2]))
                  for i in range(rng.choice([1, 2])))
    P = [rng.choice([0.6, 1.0, 1.3]) for _ in range(n)]
    cur = [sum(P[i] for i in range(n) if m >> i & 1) for m in range(1 << n)]
    f2 = rng.choice([1.0, 0.85, 0.6])
    f3 = f2 * rng.choice([1.0, 0.9])
    return CollectorModel(
        n=n, types=types, current_a=tuple(cur),
        loss_weight=tuple(c * c for c in cur), loss_coef=1.0,
        derating=(1.0, f2, f3, f3, f3, f3),
        sigma_eur_per_m=rng.choice([0.0, 0.1, 0.4]),
        m_max=rng.choice([2, 3, 4]), p_max=rng.choice([1, 2]),
        switchgear_a=rng.choice([INF, 2.0, 2.7]), ring_panels=3,
        turbine_panel_eur=rng.choice([0.0, 0.2]),
        station_building_eur=rng.choice([0.0, 1.5, 5.0]),
        station_panel_eur=rng.choice([0.0, 0.1, 0.5]),
        allow_stations=rng.random() < 0.9,
        bay_eur=rng.choice([0.0, 1.0]), bay_a=rng.choice([INF, INF, 1.4]))


def _check(seed, count):
    rng = random.Random(seed)
    roots = 0
    for _ in range(count):
        R, C = rng.choice([(2, 3), (3, 3), (2, 4)])
        N, E = _grid(R, C, rng, rng.random() < 0.4)
        n = rng.choice([2, 3])
        cells = rng.sample(range(N), n + 1)
        turb, g = cells[:n], cells[n]
        model = _model(rng, n)
        G = CollectorGraph(N, E)
        for rt in (False, True):
            res = (solve_collector(G, turb, model, engine="B") if rt else
                   solve_collector(G, turb, model, engine="B", root=g))
            ve = res.mv[g]
            vo, _, truncated = brute_share(N, E, turb, g, model, K=2,
                                           root_transit=rt, max_trees=200_000)
            if truncated:
                continue
            roots += 1
            if ve == INF or vo == INF:
                assert ve == vo == INF
                continue
            assert ve <= vo + 1e-9 * max(1.0, vo), (ve, vo)
            if ve < vo - 1e-9 * max(1.0, vo):
                vo3, _, tr3 = brute_share(N, E, turb, g, model, K=3,
                                          root_transit=rt,
                                          max_trees=400_000)
                assert not tr3
                vo = vo3
            assert ve == pytest.approx(vo, rel=1e-9, abs=1e-12)
            cost, _ = reprice(res.design(g), G, turb, model, root_transit=rt)
            assert cost == pytest.approx(ve, rel=1e-12, abs=1e-12)
    return roots


def test_engine_b_equals_the_independent_oracle():
    assert _check(11, 6) >= 10


@pytest.mark.slow
@pytest.mark.parametrize("seed", [12, 13, 14])
def test_engine_b_equals_the_independent_oracle_many(seed):
    assert _check(seed, 40) >= 60

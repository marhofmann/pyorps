"""Seeded tower fields (plan rev. 5, D5; CODE-11).

Oracle: 3 seeds on a 30 x 30 lattice against a brute force that enumerates
every (direction, span) and takes ``min(T + node, seed)`` at each chain
origin -- node costs never overwritten. Tier 2 is checked by the
superposition identity: with uncharged terminals a seeded field equals
``min_k S_k + field_k`` of the single-source fields. Every seeded walk
starts at a seed and re-prices to its field value.
"""
import math

import numpy as np
import pytest

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.tower_field import (
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
    angle_tables_from_profile,
    cost_factor,
    intermediate_offsets,
)
from pyorps.utils.directional import primitive_directions

SHAPE = (30, 30)
SIGMA = 10.0


def _brute_seeded(values, tower, dirs, sigma, model, seed, blocked):
    h, w = values.shape

    def step_cost(p, q, r, c):
        acc = values[r - p, c - q] + values[r, c]
        for orow, ocol in intermediate_offsets(p, q):
            acc += values[r - p + orow, c - q + ocol]
        return acc * cost_factor(p, q) * sigma

    T = np.full(values.shape, np.inf)
    for _ in range(400):
        new = T.copy()
        for r in range(h):
            for c in range(w):
                best = new[r, c]
                for p, q in dirs:
                    ds = math.hypot(p, q) * sigma
                    for m in range(1, int(model.max_span_m / ds) + 2):
                        length = m * ds
                        if (length > model.max_span_m + 1e-9
                                or (not model.max_span_inclusive
                                    and length >= model.max_span_m - 1e-9)):
                            break
                        yr, yc = r - m * p, c - m * q
                        if not (0 <= yr < h and 0 <= yc < w):
                            break
                        if length < model.min_span_m - 1e-9:
                            continue
                        chain = min(T[yr, yc] + tower[yr, yc], seed[yr, yc])
                        if not np.isfinite(chain):
                            continue
                        if any(blocked[r - j * p, c - j * q]
                               for j in range(m + 1)):
                            continue
                        if any(blocked[r - (j + 1) * p + orow,
                                       c - (j + 1) * q + ocol]
                               for j in range(m)
                               for orow, ocol in intermediate_offsets(p, q)):
                            continue
                        total = chain + sum(
                            step_cost(p, q, r - j * p, c - j * q)
                            for j in range(m))
                        best = min(best, total)
                new[r, c] = best
        if np.array_equal(np.nan_to_num(new, posinf=-1.0),
                          np.nan_to_num(T, posinf=-1.0)):
            return T
        T = new
    raise AssertionError("the reference did not converge")


def _setup(seed_rng=3):
    rng = np.random.default_rng(seed_rng)
    values = rng.integers(1, 40, size=SHAPE).astype(np.float64)
    tower = rng.integers(100, 400, size=SHAPE).astype(np.float64)
    blocked = rng.random(SHAPE) < 0.06
    seeds = [(4, 3), (25, 7), (15, 26)]
    for s in seeds:
        blocked[s] = False
    tower[blocked] = np.inf
    labels = [0.0, 850.0, 1300.0]
    seed = np.full(SHAPE, np.inf)
    for (r, c), v in zip(seeds, labels):
        seed[r, c] = v
    return values, tower, blocked, seeds, labels, seed


def test_three_seeds_match_the_brute_force():
    values, tower, blocked, _seeds, _labels, seed = _setup()
    dirs = primitive_directions(1)
    lattice = TowerLattice(cell_size_m=SIGMA, factor=1, directions=dirs)
    model = TowerFieldModel(min_span_m=10.0, max_span_m=45.0,
                            charge_terminal_towers=False)
    field = TowerFieldSolver(values=values, tower_cost=tower,
                             lattice=lattice, model=model,
                             blocked=blocked).solve(seed_chain=seed)
    T = _brute_seeded(values, tower, dirs.tolist(), SIGMA, model, seed,
                      blocked)
    want = np.where(np.isfinite(seed) & (seed <= T), seed, T)
    want = np.where(np.isfinite(tower) | np.isfinite(seed), want, np.inf)
    assert np.array_equal(np.isfinite(field.arrival), np.isfinite(want))
    fin = np.isfinite(want)
    assert np.allclose(field.arrival[fin], want[fin], rtol=0, atol=1e-8)
    assert field.source is None and field.meta["n_seeds"] == 3


@pytest.mark.parametrize("tier", [1, 2])
def test_seeded_field_is_the_minimum_of_shifted_single_source_fields(tier):
    values, tower, blocked, seeds, labels, seed = _setup(5)
    profile = InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")
    lattice = TowerLattice(cell_size_m=SIGMA, factor=1,
                           directions=primitive_directions(2))
    model = TowerFieldModel(min_span_m=10.0, max_span_m=60.0,
                            charge_terminal_towers=False, angle_tier=tier)
    kw = dict(values=values, tower_cost=tower, lattice=lattice, model=model,
              blocked=blocked)
    if tier == 2:
        kw["angles"] = angle_tables_from_profile(profile, lattice)
    solver = TowerFieldSolver(**kw)
    seeded = solver.solve(seed_chain=seed).arrival
    shifted = np.full(SHAPE, np.inf)
    for s, v in zip(seeds, labels):
        np.minimum(shifted, v + solver.solve(s).arrival, out=shifted)
    np.testing.assert_allclose(seeded, shifted, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("tier", [1, 2])
def test_seeded_walks_start_at_a_seed_and_reprice(tier):
    from pyorps.graph.tower_field_oracle import score_tower_chain

    values, tower, blocked, seeds, labels, seed = _setup(9)
    profile = InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")
    lattice = TowerLattice(cell_size_m=SIGMA, factor=1,
                           directions=primitive_directions(2))
    model = TowerFieldModel(min_span_m=10.0, max_span_m=60.0,
                            charge_terminal_towers=False, angle_tier=tier)
    kw = dict(values=values, tower_cost=tower, lattice=lattice, model=model,
              blocked=blocked)
    angles = None
    if tier == 2:
        angles = angle_tables_from_profile(profile, lattice)
        kw["angles"] = angles
    field = TowerFieldSolver(**kw).solve(seed_chain=seed)
    label_of = dict(zip(seeds, labels))
    rng = np.random.default_rng(1)
    reach = np.argwhere(np.isfinite(field.arrival) & ~field.seeded)
    checked = 0
    for r, c in reach[rng.choice(len(reach), size=12, replace=False)]:
        seq = field.tower_sequence(int(r), int(c))
        start = (seq[0].row, seq[0].col)
        assert start in label_of, start
        assert (seq[-1].row, seq[-1].col) == (r, c)
        score = score_tower_chain([(t.row, t.col) for t in seq],
                                  values=values, tower_cost=tower,
                                  lattice=lattice, model=model,
                                  angles=angles)
        assert score.violations == (), score.violations
        assert label_of[start] + score.total == pytest.approx(
            float(field.arrival[r, c]), rel=1e-9)
        checked += 1
    assert checked == 12
    for s in seeds:                      # a seed reads its own label
        assert field.tower_sequence(*s)[0].is_terminal
        assert len(field.tower_sequence(*s)) == 1


def test_bad_seeds_are_refused():
    values, tower, blocked, seeds, _labels, seed = _setup()
    lattice = TowerLattice(cell_size_m=SIGMA, factor=1,
                           directions=primitive_directions(1))
    model = TowerFieldModel(min_span_m=10.0, max_span_m=45.0)
    solver = TowerFieldSolver(values=values, tower_cost=tower,
                              lattice=lattice, model=model, blocked=blocked)
    with pytest.raises(ValueError, match="shape"):
        solver.solve(seed_chain=np.zeros((3, 3)))
    bad = seed.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        solver.solve(seed_chain=bad)
    on_blocked = np.full(SHAPE, np.inf)
    on_blocked[tuple(np.argwhere(blocked)[0])] = 1.0
    with pytest.raises(ValueError, match="blocked"):
        solver.solve(seed_chain=on_blocked)
    with pytest.raises(ValueError, match="nothing starts"):
        solver.solve(seed_chain=np.full(SHAPE, np.inf))


# ---------------------------------------------------------------------------
# matched tier 1 and the raster wrapper's D5 options


def _pair(**tier1_overrides):
    from pyorps.graph.tower_field import assert_matched_tier1  # noqa: F401
    values, tower, blocked, seeds, labels, seed = _setup(11)
    profile = InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")
    lattice = TowerLattice(cell_size_m=SIGMA, factor=1,
                           directions=primitive_directions(2))
    base = dict(min_span_m=10.0, max_span_m=60.0,
                charge_terminal_towers=False)
    m2 = TowerFieldModel(angle_tier=2, **base)
    m1 = TowerFieldModel(angle_tier=1, **{**base, **tier1_overrides})
    angles = angle_tables_from_profile(profile, lattice)
    f2 = TowerFieldSolver(values=values, tower_cost=tower, lattice=lattice,
                          model=m2, blocked=blocked,
                          angles=angles).solve(seed_chain=seed)
    f1 = TowerFieldSolver(values=values, tower_cost=tower, lattice=lattice,
                          model=m1, blocked=blocked).solve(seed_chain=seed)
    return f1, f2, (values, tower, blocked, lattice, m1, seed)


def test_matched_tier1_is_accepted_and_is_a_lower_bound():
    from pyorps.graph.tower_field import assert_matched_tier1

    f1, f2, _ = _pair()
    assert_matched_tier1(f1, f2)
    fin = np.isfinite(f2.arrival)
    assert np.all(f1.arrival[fin] <= f2.arrival[fin] + 1e-9)


def test_unmatched_tier1_is_refused():
    from pyorps.graph.tower_field import assert_matched_tier1

    f1, f2, _ = _pair(min_span_m=20.0)
    with pytest.raises(AssertionError, match="min span"):
        assert_matched_tier1(f1, f2)
    _f1, f2, (values, tower, blocked, lattice, m1, seed) = _pair()
    more = blocked.copy()
    more[3, 20] = True
    tower2 = tower.copy()
    tower2[3, 20] = np.inf
    f1 = TowerFieldSolver(values=values + 1.0, tower_cost=tower2,
                          lattice=lattice, model=m1,
                          blocked=more).solve(seed_chain=seed)
    with pytest.raises(AssertionError) as info:
        assert_matched_tier1(f1, f2)
    text = str(info.value)
    assert "terrain values" in text and "placement" in text \
        and "crossing" in text
    with pytest.raises(AssertionError, match="tiers"):
        assert_matched_tier1(f2, f2)


def test_wrapper_crossing_values_mask_and_terminal_ok():
    from pyorps.graph.tower_field import tower_field_from_raster

    profile = InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")
    rng = np.random.default_rng(4)
    raster = rng.integers(1, 200, size=(40, 40)).astype(np.uint16)
    raster[18:22, 10:30] = 65535                 # a protected strip
    model = TowerFieldModel.matching_kernel(profile)
    kw = dict(cell_size_m=10.0, profile=profile, model=model,
              source_cell=(3, 10), factor=2,          # lattice 20 x 20
              directions=primitive_directions(2), record_pred=False,
              spans_cross_exclusions=True, blocked_fraction=0.5)
    zero = tower_field_from_raster(raster, **kw)
    priced = tower_field_from_raster(
        raster, crossing_values=np.full(raster.shape, 500.0), **kw)
    fin = np.isfinite(zero.arrival) & np.isfinite(priced.arrival)
    # pricing the crossing can only raise the field, and it does raise it
    assert np.all(priced.arrival[fin] >= zero.arrival[fin] - 1e-9)
    assert np.any(priced.arrival[fin] > zero.arrival[fin] + 1e-6)
    # a C7 mask across the whole width (raster row 30 = lattice row 15)
    # cuts the far side off
    c7 = np.zeros(raster.shape, dtype=bool)
    c7[30, :] = True
    cut = tower_field_from_raster(raster, crossing_mask=c7, **kw)
    assert np.isfinite(zero.arrival[16:, :]).any()
    assert not np.isfinite(cut.arrival[16:, :]).any()
    # terminal_ok is its own mask: refuse line ends on the left quarter
    ok = zero.masks["placement"].copy()
    ok[:, :5] = False
    ends = tower_field_from_raster(raster, terminal_ok=ok, **kw)
    assert not np.isfinite(ends.arrival[:, :5]).any()
    np.testing.assert_array_equal(ends.arrival[:, 5:], zero.arrival[:, 5:])

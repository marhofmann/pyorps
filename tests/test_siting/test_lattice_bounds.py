"""The lattice bound pieces against exact oracles (math.md M7.6a, M9.1, M4)."""


import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pyorps.siting.lattice_bounds import float32_lower_bound, offset_min_filter, steiner_lower_bound


def _grid_graph(n, rng):
    def idx(r, c):
        return r * n + c

    rows, cols, w = [], [], []
    for r in range(n):
        for c in range(n):
            for dr, dc in ((0, 1), (1, 0)):
                if r + dr < n and c + dc < n:
                    ww = rng.uniform(1.0, 10.0)
                    rows += [idx(r, c), idx(r + dr, c + dc)]
                    cols += [idx(r + dr, c + dc), idx(r, c)]
                    w += [ww, ww]
    return csr_matrix((w, (rows, cols)), shape=(n * n, n * n))


def _steiner_exact(dist_all, terminals):
    """Dreyfus-Wagner on the metric closure of a small graph; returns the Steiner tree weight."""
    nv, k = dist_all.shape[0], len(terminals)
    dp = np.full((1 << k, nv), np.inf)
    for i, t in enumerate(terminals):
        dp[1 << i] = dist_all[t]
    for mask in range(1, 1 << k):
        if mask & (mask - 1) == 0:
            continue
        sub = (mask - 1) & mask
        while sub:
            dp[mask] = np.minimum(dp[mask], dp[sub] + dp[mask ^ sub])
            sub = (sub - 1) & mask
        dp[mask] = np.min(dp[mask][None, :] + dist_all, axis=1)      # relax over the metric closure
    return float(dp[(1 << k) - 1][terminals[0]])


@pytest.mark.parametrize("seed", range(6))
def test_steiner_bound_never_exceeds_the_exact_steiner_tree(seed):
    rng = np.random.default_rng(seed)
    n = 6
    g = _grid_graph(n, rng)
    d_all = dijkstra(g, directed=False)
    turbines = list(rng.choice(n * n, size=3, replace=False))
    dist = np.stack([d_all[t] for t in turbines])                       # (3, 36): distance turbine -> every cell
    tt = d_all[np.ix_(turbines, turbines)]
    lb = steiner_lower_bound(dist, tt)
    for root in range(n * n):
        terms = [root] + turbines
        if len(set(terms)) < len(terms):
            continue
        exact = _steiner_exact(d_all, terms)
        assert lb[root] <= exact + 1e-9, (seed, root, lb[root], exact)
    # it is not trivial: somewhere it is strictly above the farthest-turbine distance
    assert (lb > dist.max(axis=0) + 1e-9).any()


def test_steiner_bound_infinite_where_a_turbine_is_unreachable():
    dist = np.array([[1.0, np.inf], [2.0, 3.0]])
    tt = np.array([[0.0, 1.5], [1.5, 0.0]])
    out = steiner_lower_bound(dist, tt)
    assert np.isfinite(out[0]) and np.isinf(out[1])


def test_steiner_bound_rejects_a_bad_terminal_matrix():
    with pytest.raises(ValueError):
        steiner_lower_bound(np.ones((2, 3)), np.array([[0.0, 1.0], [2.0, 0.0]]))


def test_float32_lower_bound_is_below_the_float64_distance_on_a_grid():
    rng = np.random.default_rng(0)
    n = 40
    w = rng.uniform(100, 5000, (n, n)).astype(np.float32)
    # float32 prefix sums along a row: the computed value of a k-step path is within k * eta of the float64 sum
    s32 = np.cumsum(w, axis=1, dtype=np.float32)
    s64 = np.cumsum(w.astype(np.float64), axis=1)
    lb = float32_lower_bound(s32, k_max=n)
    assert (lb <= s64).all()
    assert np.isinf(float32_lower_bound(np.array([3.4e38], dtype=np.float32), 10)[0])
    with pytest.raises(ValueError):
        float32_lower_bound(s32, k_max=int(1e9))


def test_offset_min_filter_is_the_minimum_over_the_square():
    f = np.arange(49, dtype=float).reshape(7, 7)
    out = offset_min_filter(f, 1)
    assert out[3, 3] == f[2:5, 2:5].min() and out[0, 0] == 0.0
    assert np.array_equal(offset_min_filter(f, 0), f)
    g = np.full((5, 5), np.inf)
    assert np.isinf(offset_min_filter(g, 2)).all()

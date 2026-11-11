"""The line-integral re-pricer (plan D9) against hand values and sampling."""
import math

import numpy as np
import pytest

from pyorps.certify.line_integral import fidelity_delta, line_integral_cost
from pyorps.utils.neighborhood import get_neighborhood_steps


def _sampled(values, cells, n=200_000):
    """Midpoint-rule integral along the polyline, for comparison."""
    cols = values.shape[1]
    rr, cc = np.divmod(np.asarray(cells), cols)
    total = 0.0
    for k in range(len(cells) - 1):
        p0 = np.array([rr[k] + 0.5, cc[k] + 0.5])
        p1 = np.array([rr[k + 1] + 0.5, cc[k + 1] + 0.5])
        t = (np.arange(n) + 0.5) / n
        pts = p0 + np.outer(t, p1 - p0)
        v = values[np.floor(pts[:, 0]).astype(int),
                   np.floor(pts[:, 1]).astype(int)]
        total += v.mean() * float(np.hypot(*(p1 - p0)))
    return total


def test_orthogonal_run_on_a_uniform_raster():
    v = np.full((3, 6), 7, dtype=np.uint16)
    cells = [6, 7, 8, 9, 10]                       # row 1, four steps east
    assert line_integral_cost(v, cells, cell_m=2.0) == pytest.approx(
        7 * 4 * 2.0)


def test_one_diagonal_step_uses_only_its_end_cells():
    v = np.array([[1, 9], [9, 3]], dtype=np.uint16)
    got = line_integral_cost(v, [0, 3])
    assert got == pytest.approx((1 + 3) / 2 * math.sqrt(2))
    steps = np.asarray(get_neighborhood_steps(1, directed=True),
                       dtype=np.int8)
    discrete, integral, gap = fidelity_delta(v, steps, [0, 3])
    assert discrete == pytest.approx((1 + 9 + 9 + 3) / 4 * math.sqrt(2),
                                     rel=1e-6)
    assert integral == pytest.approx(got)
    assert gap < 0


@pytest.mark.parametrize("seed", range(5))
def test_equals_fine_sampling_on_knight_routes(seed):
    rng = np.random.default_rng(seed)
    v = rng.integers(1, 50, size=(20, 20)).astype(np.uint16)
    moves = [(1, 2), (2, 1), (1, 1), (0, 1), (1, 0), (2, 3)]
    r, c = 1, 1
    cells = [r * 20 + c]
    for _ in range(6):
        dr, dc = moves[int(rng.integers(len(moves)))]
        r, c = r + dr, c + dc
        cells.append(r * 20 + c)
    exact = line_integral_cost(v, cells)
    assert exact == pytest.approx(_sampled(v, cells), rel=2e-4)


def test_crossing_an_excluded_cell_is_infinite_but_a_corner_is_not():
    v = np.full((3, 3), 5, dtype=np.uint16)
    v[0, 1] = 65535
    assert math.isinf(line_integral_cost(v, [0, 2]))       # straight through
    assert math.isinf(line_integral_cost(v, [3, 4, 1]))    # ends in it
    v2 = np.full((2, 2), 5, dtype=np.uint16)
    v2[0, 1] = 65535
    assert line_integral_cost(v2, [0, 3]) == pytest.approx(5 * math.sqrt(2))


# ------------------------------------------------------------ V4-02 (limit)


def _exact_cell_lengths(step, shape=(5, 7)):
    """Exact chord length (in cell units) of the step ``(dr, dc)`` through
    every cell, from the line integral over a one-cell-at-a-time raster."""
    import numpy as np

    from pyorps.certify.line_integral import line_integral_cost

    dr, dc = step
    cols = shape[1]
    start, end = 1 * cols + 1, (1 + dr) * cols + (1 + dc)
    out = {}
    for r in range(shape[0]):
        for c in range(shape[1]):
            v = np.zeros(shape, dtype=np.uint16)
            v[r, c] = 1
            w = line_integral_cost(v, [start, end])
            if w > 1e-12:
                out[(r - 1, c - 1)] = w
    return out


def test_r3_corner_steps_are_priced_inexactly_by_the_compiled_rule():
    """Review finding V4-02, kept as a documented LIMIT, not a repair.

    The compiled step rule charges both end cells and the cells of
    ``intermediate_offsets(p, q)`` each ``hypot(p, q) / (2 + n_int)``. That is
    exact for the cardinal, knight (1, 2) steps, but for the r3 steps (1, 3)
    and (2, 3) the weights are wrong and (1, 3) charges two cells the line only
    touches at a corner. The Stage-5 line-integral re-price
    (``pyorps.certify.line_integral``) measures the gap per route; exact
    per-cell weights for every step class are NOT built. The overhead lattice
    of the siting study uses steps up to (1, 3)/(2, 3), so its pricing carries
    this error and must be reported with the fidelity gap.
    """
    import math

    from pyorps.graph.tower_field import intermediate_offsets

    def compiled(step):
        p, q = step
        cells = [(0, 0), (p, q)] + list(intermediate_offsets(p, q))
        w = math.hypot(p, q) / len(cells)
        return {c: w for c in cells}

    def max_error(step):
        exact, comp = _exact_cell_lengths(step), compiled(step)
        keys = set(exact) | set(comp)
        return max(abs(exact.get(k, 0.0) - comp.get(k, 0.0)) for k in keys)

    # exact for the cardinal, diagonal-free knight step
    assert max_error((1, 0)) == pytest.approx(0.0, abs=1e-12)
    assert max_error((1, 2)) == pytest.approx(0.0, abs=1e-12)
    # the r1 diagonal is also inexact (2 cells at 1/2 L against 4 at sqrt2/4)
    assert max_error((1, 1)) > 0.3
    # (1, 3): exact is 1/6, 1/3, 1/3, 1/6 of L on four cells; the rule spreads
    # L/6 over six, two of them touched only at a corner
    ex = _exact_cell_lengths((1, 3))
    L = math.sqrt(10.0)
    assert sorted(round(v / L, 9) for v in ex.values()) == pytest.approx(
        sorted([1 / 6, 1 / 3, 1 / 3, 1 / 6]), abs=1e-9)
    assert set(compiled((1, 3))) - set(ex) == {(1, 1), (0, 2)}
    assert max_error((1, 3)) == pytest.approx(math.sqrt(10.0) / 6, rel=1e-9)
    # (2, 3): no corner cells, but the weights are uniform instead of
    # length-proportional (max error 0.3005 in cell units)
    assert set(compiled((2, 3))) == set(_exact_cell_lengths((2, 3)))
    assert max_error((2, 3)) == pytest.approx(0.3005, abs=5e-4)

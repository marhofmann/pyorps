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

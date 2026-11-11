"""Capped window regrowth and brackets (plan 3.5, D2; review finding CODE-10).

A failed border check used to mean "grow the window and re-run" with no cap and
no exit. Growth is now capped (cells and retries), and where the certificate
does not hold the caller reports a bracket instead of a value.
"""
import numpy as np
import pytest

from pyorps.certify import (
    certify_path_field,
    drain,
    grow_window,
)
from pyorps.utils.neighborhood import get_neighborhood_steps

TOL = 1e-9


def _steps(k):
    return np.asarray(get_neighborhood_steps(k, directed=True), dtype=np.int8)


def _block(shape, r0, r1, c0, c1):
    w = np.zeros(shape, dtype=bool)
    w[r0:r1, c0:c1] = True
    return w


def test_growth_adds_a_square_margin_clipped_to_the_raster():
    w = _block((20, 30), 5, 10, 8, 14)
    g = grow_window(w, margin_cells=2, cap_cells=10_000)
    assert g.grown and not g.capped and g.regrowths == 1
    np.testing.assert_array_equal(g.window, _block((20, 30), 3, 12, 6, 16))
    assert g.cells == int(g.window.sum()) == 9 * 10
    edge = _block((20, 30), 0, 4, 0, 5)
    g2 = grow_window(edge, margin_cells=3, cap_cells=10_000)
    np.testing.assert_array_equal(g2.window, _block((20, 30), 0, 7, 0, 8))


def test_a_cell_cap_refuses_growth_and_keeps_the_window():
    w = _block((20, 30), 5, 10, 8, 14)
    g = grow_window(w, margin_cells=2, cap_cells=80)         # would be 90
    assert g.capped and not g.grown
    np.testing.assert_array_equal(g.window, w)
    assert "cap is 80" in g.reason and g.regrowths == 0
    ok = grow_window(w, margin_cells=2, cap_cells=90)         # exactly at the cap
    assert ok.grown and ok.cells == 90


def test_at_most_one_regrowth_by_default():
    w = _block((40, 40), 15, 20, 15, 20)
    g1 = grow_window(w, margin_cells=2, cap_cells=10_000)
    assert g1.grown and g1.regrowths == 1
    g2 = grow_window(g1.window, margin_cells=2, cap_cells=10_000,
                     regrowths=g1.regrowths)
    assert g2.capped and not g2.grown and "at most 1" in g2.reason
    np.testing.assert_array_equal(g2.window, g1.window)
    g3 = grow_window(g1.window, margin_cells=2, cap_cells=10_000,
                     regrowths=1, max_regrowths=2)
    assert g3.grown and g3.regrowths == 2


def test_a_window_that_already_covers_the_raster_cannot_grow():
    g = grow_window(np.ones((6, 6), dtype=bool), margin_cells=1,
                    cap_cells=10_000)
    assert g.capped and "already covers" in g.reason


def test_bad_margin_is_refused():
    with pytest.raises(ValueError):
        grow_window(np.ones((4, 4), dtype=bool), margin_cells=0, cap_cells=9)


def _walled():
    """Source left of a wall whose only gap lies outside the window."""
    vals = np.full((9, 15), 200, dtype=np.uint16)
    vals[:, 7] = 65535
    vals[0, 7] = 200
    inside = np.zeros_like(vals, dtype=bool)
    inside[2:, :] = True
    src = np.array([4 * 15 + 2])
    return vals, inside, src


def test_a_bracket_contains_the_true_value_and_is_a_point_where_exact():
    vals, inside, src = _walled()
    steps = _steps(1)
    full = drain(vals, steps, src, [0.0], engine="python")
    cert = certify_path_field(vals, steps, inside, src, [0.0], engine="python")
    # exact cells: a point equal to the unwindowed value
    ex = np.argwhere(cert.exact)
    assert len(ex)
    for r, c in ex[:25]:
        lo, hi = cert.bracket(int(r), int(c))
        assert lo == hi == pytest.approx(float(full[r, c]), rel=1e-12)
    # uncertified cells inside the window: the true value lies in the bracket
    unc = np.argwhere(cert.uncertified & np.isfinite(full))
    assert len(unc)
    for r, c in unc:
        lo, hi = cert.bracket(int(r), int(c))
        assert lo <= full[r, c] * (1 + TOL)
        assert full[r, c] <= hi * (1 + TOL) or not np.isfinite(hi)
        assert hi == float(cert.field[r, c])


def test_uncertified_is_the_window_minus_the_exact_cells():
    vals, inside, src = _walled()
    cert = certify_path_field(vals, _steps(1), inside, src, [0.0],
                              engine="python")
    np.testing.assert_array_equal(cert.uncertified, inside & ~cert.exact)
    with pytest.raises(ValueError, match="outside the window"):
        cert.bracket(0, 0)

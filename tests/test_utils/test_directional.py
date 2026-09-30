"""Directional primitives against brute force.

Phase 1 of ``docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md``
calls this "the layer where an off-by-one is invisible and fatal", and
it was right: the first implementation passed every hand-checked case
and still dropped the last two members of a window at the grid edge,
because van Herk reads its forward block scan at ``x - (w-1)*d`` and
that position can be off the grid while cells between it and ``x`` are
not. :func:`test_window_min_matches_bruteforce` is what caught it, so
it sweeps every primitive direction rather than a chosen few.
"""

import numpy as np
import pytest

from pyorps.utils.directional import (
    bezout,
    circular_window_min,
    direction_angles,
    primitive_directions,
    ray_index,
    ray_prefix_sum,
    ray_prefix_sum_bruteforce,
    ray_run_length,
    ray_window_max,
    ray_window_max_bruteforce,
    ray_window_min,
    ray_window_min_bruteforce,
    shift,
)

SHAPES = [(9, 11), (13, 7), (1, 17), (16, 16), (5, 5)]
WINDOWS = [(0, 0), (1, 1), (1, 3), (2, 7), (0, 5), (3, 12), (1, 30)]


def _grid(shape, seed=0):
    return np.random.default_rng(seed).random(shape) * 100.0


class TestDirectionSets:
    def test_primitive_counts(self):
        assert len(primitive_directions(1)) == 8
        assert len(primitive_directions(2)) == 16
        assert len(primitive_directions(3)) == 32
        assert len(primitive_directions(4)) == 48

    def test_one_entry_per_distinct_angle(self):
        for dmax in (1, 2, 3, 4, 5):
            ang = direction_angles(primitive_directions(dmax))
            assert len(np.unique(np.round(ang, 12))) == len(ang)

    def test_sorted_by_angle(self):
        ang = direction_angles(primitive_directions(4))
        assert np.all(np.diff(ang) > 0)

    def test_closed_under_negation(self):
        dirs = {tuple(d) for d in primitive_directions(3).tolist()}
        assert all((-p, -q) in dirs for p, q in dirs)

    @pytest.mark.parametrize("dmax", [1, 2, 3, 4])
    def test_bezout_and_ray_index(self, dmax):
        for p, q in primitive_directions(dmax).tolist():
            a, b = bezout(p, q)
            assert a * p + b * q == 1
            t = ray_index((15, 15), p, q)
            # One step along the ray advances the index by exactly one.
            # Compare well inside the grid: the shift fills off-grid
            # cells, and dmax=4 reaches four cells out.
            core = (slice(5, 10), slice(5, 10))
            assert np.array_equal(shift(t, -p, -q, 0)[core],
                                  (t + 1)[core])

    def test_non_primitive_rejected(self):
        with pytest.raises(ValueError, match="not primitive"):
            bezout(2, 4)
        with pytest.raises(ValueError, match="not primitive"):
            ray_window_min(_grid((5, 5)), 2, 2, 1, 2)


class TestRayPrefixSum:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_matches_bruteforce(self, shape):
        a = _grid(shape)
        for p, q in primitive_directions(3).tolist():
            assert np.allclose(ray_prefix_sum(a, p, q),
                               ray_prefix_sum_bruteforce(a, p, q),
                               atol=1e-9)

    def test_difference_is_the_span_sum(self):
        """The property the tower field actually uses."""
        a = _grid((12, 12), seed=3)
        p, q = 1, 2
        P = ray_prefix_sum(a, p, q)
        r, c, m = 9, 9, 3
        span = P[r, c] - P[r - m * p, c - m * q]
        explicit = sum(a[r - j * p, c - j * q] for j in range(m))
        assert span == pytest.approx(explicit, rel=1e-12)


class TestRayWindows:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_window_min_matches_bruteforce(self, shape):
        a = _grid(shape, seed=1)
        for p, q in primitive_directions(3).tolist():
            for lo, hi in WINDOWS:
                got = ray_window_min(a, p, q, lo, hi)
                want = ray_window_min_bruteforce(a, p, q, lo, hi)
                assert np.array_equal(got, want), (shape, p, q, lo, hi)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_window_max_matches_bruteforce(self, shape):
        a = _grid(shape, seed=2)
        for p, q in primitive_directions(3).tolist():
            for lo, hi in WINDOWS:
                got = ray_window_max(a, p, q, lo, hi)
                want = ray_window_max_bruteforce(a, p, q, lo, hi)
                assert np.array_equal(got, want), (shape, p, q, lo, hi)

    def test_offgrid_is_infinite_not_borrowed(self):
        a = np.ones((4, 4))
        out = ray_window_min(a, 1, 0, 1, 2)
        assert np.isinf(out[0]).all()          # nothing above row 0
        assert np.isfinite(out[2:]).all()

    @pytest.mark.parametrize("shape", [(11, 13), (6, 6)])
    def test_argmin_points_at_the_minimum(self, shape):
        a = _grid(shape, seed=5)
        for p, q in primitive_directions(2).tolist():
            for lo, hi in [(1, 1), (1, 4), (2, 9)]:
                val, m = ray_window_min(a, p, q, lo, hi, return_arg=True)
                ok = np.isfinite(val)
                rr, cc = np.nonzero(ok)
                assert np.all(m[ok] >= lo) and np.all(m[ok] <= hi)
                assert np.allclose(a[rr - m[ok] * p, cc - m[ok] * q],
                                   val[ok])
                assert np.all(m[~ok] == -1)

    def test_rejects_backwards_window(self):
        with pytest.raises(ValueError, match="m_lo <= m_hi"):
            ray_window_min(_grid((5, 5)), 1, 0, 3, 1)


class TestRunLength:
    @pytest.mark.parametrize("shape", [(9, 11), (12, 12)])
    def test_matches_bruteforce(self, shape):
        blocked = np.random.default_rng(3).random(shape) < 0.25
        rows, cols = shape
        for p, q in primitive_directions(2).tolist():
            for cap in (1, 3, 8, 17):
                got = ray_run_length(blocked, p, q, cap)
                want = np.zeros(shape, dtype=np.int64)
                for r in range(rows):
                    for c in range(cols):
                        n, rr, cc = 0, r, c
                        while (0 <= rr < rows and 0 <= cc < cols
                               and not blocked[rr, cc] and n < cap):
                            n += 1
                            rr, cc = rr - p, cc - q
                        want[r, c] = n
                assert np.array_equal(got, want), (p, q, cap)

    def test_blocked_cell_reads_zero(self):
        blocked = np.zeros((5, 5), bool)
        blocked[2, 2] = True
        assert ray_run_length(blocked, 0, 1, 4)[2, 2] == 0
        assert ray_run_length(blocked, 0, 1, 4)[2, 3] == 1


class TestCircularWindow:
    @pytest.mark.parametrize("half", [0, 1, 2, 3])
    def test_matches_rolls(self, half):
        a = np.random.default_rng(7).random((12, 5, 5))
        want = np.stack([
            np.min(np.stack([a[(i + k) % 12] for k in range(-half, half + 1)]),
                   axis=0) for i in range(12)])
        assert np.allclose(circular_window_min(a, half), want)

    def test_full_window_is_the_global_min(self):
        a = np.random.default_rng(8).random((6, 4, 4))
        assert np.allclose(circular_window_min(a, 5),
                           np.broadcast_to(a.min(axis=0), a.shape))

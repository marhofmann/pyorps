"""Tests for pyorps.utils._constrained_context — state packing/unpacking."""

import pytest


class TestPackState:
    """Test pack_state / unpack_state roundtrip."""

    def test_roundtrip_basic(self):
        from pyorps.utils._constrained_context import pack_state, unpack_state

        cell, direction, span_bin = 42, 3, 7
        n_dirs, n_span_bins = 16, 20

        state = pack_state(cell, direction, span_bin, n_dirs, n_span_bins)
        c, d, s = unpack_state(state, n_dirs, n_span_bins)

        assert c == cell
        assert d == direction
        assert s == span_bin

    def test_roundtrip_cell_zero(self):
        from pyorps.utils._constrained_context import pack_state, unpack_state

        state = pack_state(0, 0, 0, 8, 10)
        c, d, s = unpack_state(state, 8, 10)

        assert c == 0
        assert d == 0
        assert s == 0

    def test_roundtrip_max_span_bin(self):
        from pyorps.utils._constrained_context import pack_state, unpack_state

        n_dirs, n_span_bins = 16, 100
        max_span = n_span_bins - 1

        state = pack_state(1000, 15, max_span, n_dirs, n_span_bins)
        c, d, s = unpack_state(state, n_dirs, n_span_bins)

        assert c == 1000
        assert d == 15
        assert s == max_span

    def test_roundtrip_large_cell(self):
        from pyorps.utils._constrained_context import pack_state, unpack_state

        # Large cell index (e.g., 4000x4000 raster = 16M cells)
        cell = 16_000_000
        n_dirs, n_span_bins = 16, 50

        state = pack_state(cell, 7, 25, n_dirs, n_span_bins)
        c, d, s = unpack_state(state, n_dirs, n_span_bins)

        assert c == cell
        assert d == 7
        assert s == 25

    def test_different_states_produce_different_indices(self):
        from pyorps.utils._constrained_context import pack_state

        n_dirs, n_span_bins = 16, 20
        s1 = pack_state(10, 3, 5, n_dirs, n_span_bins)
        s2 = pack_state(10, 3, 6, n_dirs, n_span_bins)
        s3 = pack_state(10, 4, 5, n_dirs, n_span_bins)
        s4 = pack_state(11, 3, 5, n_dirs, n_span_bins)

        assert len({s1, s2, s3, s4}) == 4

    def test_state_index_is_contiguous(self):
        """State indices should be contiguous starting from 0."""
        from pyorps.utils._constrained_context import pack_state

        n_dirs, n_span_bins = 4, 3
        indices = set()
        for cell in range(5):
            for d in range(n_dirs):
                for s in range(n_span_bins):
                    indices.add(pack_state(cell, d, s, n_dirs, n_span_bins))

        assert indices == set(range(5 * n_dirs * n_span_bins))


class TestPackStateH:
    """Test pack_state_h / unpack_state_h roundtrip (with height class)."""

    def test_roundtrip_basic(self):
        from pyorps.utils._constrained_context import pack_state_h, unpack_state_h

        cell, direction, span_bin, height = 42, 3, 7, 2
        n_dirs, n_span_bins, n_heights = 16, 20, 5

        state = pack_state_h(cell, direction, span_bin, height,
                             n_dirs, n_span_bins, n_heights)
        c, d, s, h = unpack_state_h(state, n_dirs, n_span_bins, n_heights)

        assert c == cell
        assert d == direction
        assert s == span_bin
        assert h == height

    def test_roundtrip_all_zeros(self):
        from pyorps.utils._constrained_context import pack_state_h, unpack_state_h

        state = pack_state_h(0, 0, 0, 0, 8, 10, 3)
        c, d, s, h = unpack_state_h(state, 8, 10, 3)

        assert (c, d, s, h) == (0, 0, 0, 0)

    def test_roundtrip_max_values(self):
        from pyorps.utils._constrained_context import pack_state_h, unpack_state_h

        n_dirs, n_span_bins, n_heights = 16, 100, 8
        cell = 500_000
        direction = n_dirs - 1
        span_bin = n_span_bins - 1
        height = n_heights - 1

        state = pack_state_h(cell, direction, span_bin, height,
                             n_dirs, n_span_bins, n_heights)
        c, d, s, h = unpack_state_h(state, n_dirs, n_span_bins, n_heights)

        assert c == cell
        assert d == direction
        assert s == span_bin
        assert h == height

    def test_height_classes_differ(self):
        """Different height classes must produce different state indices."""
        from pyorps.utils._constrained_context import pack_state_h

        n_dirs, n_span_bins, n_heights = 8, 10, 4
        states = [
            pack_state_h(100, 3, 5, h, n_dirs, n_span_bins, n_heights)
            for h in range(n_heights)
        ]
        assert len(set(states)) == n_heights

    def test_state_index_contiguous(self):
        """State indices should be contiguous starting from 0."""
        from pyorps.utils._constrained_context import pack_state_h

        n_dirs, n_span_bins, n_heights = 2, 3, 2
        indices = set()
        for cell in range(3):
            for d in range(n_dirs):
                for s in range(n_span_bins):
                    for h in range(n_heights):
                        indices.add(pack_state_h(cell, d, s, h,
                                                 n_dirs, n_span_bins, n_heights))

        assert indices == set(range(3 * n_dirs * n_span_bins * n_heights))

    def test_roundtrip_sweep(self):
        """Exhaustive roundtrip over a small state space."""
        from pyorps.utils._constrained_context import pack_state_h, unpack_state_h

        n_dirs, n_span_bins, n_heights = 4, 5, 3
        for cell in range(10):
            for d in range(n_dirs):
                for s in range(n_span_bins):
                    for h in range(n_heights):
                        state = pack_state_h(cell, d, s, h,
                                             n_dirs, n_span_bins, n_heights)
                        c2, d2, s2, h2 = unpack_state_h(state, n_dirs,
                                                         n_span_bins, n_heights)
                        assert (c2, d2, s2, h2) == (cell, d, s, h), (
                            f"Mismatch for ({cell},{d},{s},{h}): "
                            f"got ({c2},{d2},{s2},{h2})"
                        )

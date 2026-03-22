"""
Tests for the extracted _heap Cython module.

Covers:
- PyBinaryHeap32: push/pop ordering, bulk elements, empty check
- PyBinaryHeap64: push/pop with uint64 values > 2^32
- py_ravel_index / py_unravel_index roundtrip
- py_round_up_power_of_two edge cases
- py_get_circular_index wrapping behavior
"""

import pytest

from pyorps.utils._heap import (
    PyBinaryHeap32,
    PyBinaryHeap64,
    py_ravel_index,
    py_unravel_index,
    py_round_up_power_of_two,
    py_get_circular_index,
)


# ==================== PyBinaryHeap32 ====================

class TestPyBinaryHeap32:
    """Tests for the 32-bit binary min-heap."""

    def test_empty_on_creation(self):
        heap = PyBinaryHeap32()
        assert heap.empty()

    def test_not_empty_after_push(self):
        heap = PyBinaryHeap32()
        heap.push(0, 1.0)
        assert not heap.empty()

    def test_single_push_pop(self):
        heap = PyBinaryHeap32()
        heap.push(42, 3.14)
        idx, pri = heap.top()
        assert idx == 42
        assert pri == pytest.approx(3.14)
        heap.pop()
        assert heap.empty()

    def test_min_heap_ordering(self):
        """Elements should come out in ascending priority order."""
        heap = PyBinaryHeap32()
        heap.push(1, 5.0)
        heap.push(2, 1.0)
        heap.push(3, 3.0)
        heap.push(4, 2.0)
        heap.push(5, 4.0)

        expected_order = [(2, 1.0), (4, 2.0), (3, 3.0), (5, 4.0), (1, 5.0)]
        for exp_idx, exp_pri in expected_order:
            idx, pri = heap.top()
            assert idx == exp_idx
            assert pri == pytest.approx(exp_pri)
            heap.pop()

        assert heap.empty()

    def test_duplicate_priorities(self):
        """Both nodes with the same priority should be retrievable."""
        heap = PyBinaryHeap32()
        heap.push(10, 2.0)
        heap.push(20, 2.0)

        first_idx, first_pri = heap.top()
        assert first_pri == pytest.approx(2.0)
        heap.pop()

        second_idx, second_pri = heap.top()
        assert second_pri == pytest.approx(2.0)
        heap.pop()

        assert {first_idx, second_idx} == {10, 20}
        assert heap.empty()

    def test_1000_elements(self):
        """Verify correct ordering with 1000 elements."""
        heap = PyBinaryHeap32()
        import random
        random.seed(12345)

        priorities = list(range(1000))
        random.shuffle(priorities)

        for i, p in enumerate(priorities):
            heap.push(i, float(p))

        prev_priority = -1.0
        count = 0
        while not heap.empty():
            _, pri = heap.top()
            assert pri >= prev_priority, f"Heap violated at step {count}: {pri} < {prev_priority}"
            prev_priority = pri
            heap.pop()
            count += 1

        assert count == 1000

    def test_size_tracking(self):
        heap = PyBinaryHeap32()
        assert heap.size() == 0
        heap.push(1, 1.0)
        heap.push(2, 2.0)
        assert heap.size() == 2
        heap.pop()
        assert heap.size() == 1


# ==================== PyBinaryHeap64 ====================

class TestPyBinaryHeap64:
    """Tests for the 64-bit binary min-heap."""

    def test_empty_on_creation(self):
        heap = PyBinaryHeap64()
        assert heap.empty()

    def test_single_push_pop(self):
        heap = PyBinaryHeap64()
        heap.push(99, 7.5)
        idx, pri = heap.top()
        assert idx == 99
        assert pri == pytest.approx(7.5)
        heap.pop()
        assert heap.empty()

    def test_large_indices_beyond_uint32(self):
        """Verify correct handling of indices exceeding 2^32."""
        heap = PyBinaryHeap64()
        large_idx1 = 2**32 + 1       # 4294967297
        large_idx2 = 2**32 + 1000    # 4294968296
        large_idx3 = 2**48           # 281474976710656

        heap.push(large_idx3, 1.0)
        heap.push(large_idx1, 3.0)
        heap.push(large_idx2, 2.0)

        idx, pri = heap.top()
        assert idx == large_idx3
        assert pri == pytest.approx(1.0)
        heap.pop()

        idx, pri = heap.top()
        assert idx == large_idx2
        assert pri == pytest.approx(2.0)
        heap.pop()

        idx, pri = heap.top()
        assert idx == large_idx1
        assert pri == pytest.approx(3.0)
        heap.pop()

        assert heap.empty()

    def test_min_heap_ordering(self):
        heap = PyBinaryHeap64()
        heap.push(10, 50.0)
        heap.push(20, 10.0)
        heap.push(30, 30.0)

        idx, _ = heap.top()
        assert idx == 20
        heap.pop()

        idx, _ = heap.top()
        assert idx == 30
        heap.pop()

        idx, _ = heap.top()
        assert idx == 10
        heap.pop()

        assert heap.empty()

    def test_1000_elements_64(self):
        """Verify correct ordering with 1000 uint64 elements."""
        heap = PyBinaryHeap64()
        import random
        random.seed(54321)

        priorities = [random.random() * 1000.0 for _ in range(1000)]
        for i, p in enumerate(priorities):
            heap.push(2**33 + i, p)

        prev = -1.0
        count = 0
        while not heap.empty():
            _, pri = heap.top()
            assert pri >= prev
            prev = pri
            heap.pop()
            count += 1

        assert count == 1000


# ==================== Index conversion ====================

class TestRavelUnravel:
    """Tests for ravel_index / unravel_index roundtrip."""

    def test_origin(self):
        assert py_ravel_index(0, 0, 100) == 0

    def test_first_row(self):
        assert py_ravel_index(0, 5, 10) == 5

    def test_second_row(self):
        assert py_ravel_index(1, 0, 10) == 10

    def test_general_case(self):
        assert py_ravel_index(3, 7, 20) == 3 * 20 + 7

    def test_roundtrip(self):
        """ravel then unravel should return the original coordinates."""
        cols = 150
        for row in range(0, 50, 7):
            for col in range(0, cols, 13):
                idx = py_ravel_index(row, col, cols)
                r, c = py_unravel_index(idx, cols)
                assert r == row, f"row mismatch for ({row},{col})"
                assert c == col, f"col mismatch for ({row},{col})"

    def test_large_grid(self):
        """Test with a grid near uint32 boundary (65535 x 65535)."""
        cols = 65535
        row, col = 10000, 30000
        idx = py_ravel_index(row, col, cols)
        r, c = py_unravel_index(idx, cols)
        assert r == row
        assert c == col


# ==================== round_up_power_of_two ====================

class TestRoundUpPowerOfTwo:
    """Tests for round_up_power_of_two edge cases."""

    def test_zero(self):
        assert py_round_up_power_of_two(0) == 1

    def test_one(self):
        assert py_round_up_power_of_two(1) == 1

    def test_two(self):
        assert py_round_up_power_of_two(2) == 2

    def test_three(self):
        assert py_round_up_power_of_two(3) == 4

    def test_exact_powers(self):
        for exp in range(1, 20):
            val = 2 ** exp
            assert py_round_up_power_of_two(val) == val

    def test_just_above_power(self):
        for exp in range(1, 20):
            val = 2 ** exp + 1
            assert py_round_up_power_of_two(val) == 2 ** (exp + 1)

    def test_just_below_power(self):
        for exp in range(2, 20):
            val = 2 ** exp - 1
            assert py_round_up_power_of_two(val) == 2 ** exp

    def test_large_value(self):
        val = 1_000_000
        result = py_round_up_power_of_two(val)
        assert result == 1_048_576  # 2^20
        assert result >= val
        # Verify it's actually a power of 2
        assert (result & (result - 1)) == 0


# ==================== get_circular_index ====================

class TestGetCircularIndex:
    """Tests for circular buffer index wrapping."""

    def test_within_buffer(self):
        """Index within buffer size should be unchanged."""
        assert py_get_circular_index(3, 8) == 3

    def test_exact_buffer_size(self):
        """Index equal to buffer size should wrap to 0."""
        assert py_get_circular_index(8, 8) == 0

    def test_wrapping(self):
        """Index beyond buffer should wrap correctly."""
        assert py_get_circular_index(10, 8) == 2  # 10 % 8 = 2

    def test_zero(self):
        assert py_get_circular_index(0, 16) == 0

    def test_sequential_wrapping(self):
        """Verify a full cycle of wrapping for buffer_size=4."""
        buffer_size = 4
        expected = [0, 1, 2, 3, 0, 1, 2, 3]
        for i in range(8):
            assert py_get_circular_index(i, buffer_size) == expected[i], (
                f"Failed at index {i}"
            )

    def test_large_index(self):
        """Large logical index should still wrap correctly."""
        buffer_size = 1024  # 2^10
        logical = 1024 * 500 + 77
        assert py_get_circular_index(logical, buffer_size) == 77

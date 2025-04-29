# test_utils/test_neighborhood.py
import unittest
import numpy as np
import re

from pyorps.utils.neighborhood import (
    get_neighborhood_steps, _generate_full_steps, _get_undirected_subset
)


class TestNeighborhood(unittest.TestCase):
    """Test cases for the neighborhood utility functions."""

    def test_get_neighborhood_steps_k0(self):
        """Test get_neighborhood_steps with k=0."""
        # k=0 should return the 4 cardinal directions
        expected = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], dtype=np.int8)
        result = get_neighborhood_steps(0)

        # Sort both arrays for consistent comparison
        expected = sorted(expected.tolist())
        result = sorted(result.tolist())

        self.assertEqual(result, expected)

    def test_get_neighborhood_steps_k1(self):
        """Test get_neighborhood_steps with k=1."""
        # k=1 should return cardinal + diagonal directions
        expected = np.array([
            (1, 0), (0, 1), (-1, 0), (0, -1),  # cardinal
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # diagonal
        ], dtype=np.int8)
        result = get_neighborhood_steps(1)

        # Sort both arrays for consistent comparison
        expected = sorted(expected.tolist())
        result = sorted(result.tolist())

        self.assertEqual(result, expected)

    def test_get_neighborhood_steps_k2(self):
        """Test get_neighborhood_steps with k=2."""
        # k=2 should include knight's moves and (2,0), (0,2) etc.
        result = get_neighborhood_steps(2)

        # Check specific expected moves are in the result
        expected_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # cardinal
            (1, 1), (-1, 1), (1, -1), (-1, -1),  # diagonal
            (2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2),  # knight's moves
            (2, 0), (0, 2), (-2, 0), (0, -2)  # double cardinal
        ]

        # Convert result to a list of tuples for easier checking
        result_tuples = [tuple(step) for step in result]

        # Check that all expected moves are in the result
        for move in expected_moves:
            self.assertIn(move, result_tuples)

        # Check total number of steps (k=2 has 20 steps)
        self.assertEqual(len(result), 20)

    def test_get_neighborhood_steps_string_input(self):
        """Test get_neighborhood_steps with string input."""
        # Should extract k=1 from strings like "k=1" or "radius-1"
        result1 = get_neighborhood_steps("k=1")
        result2 = get_neighborhood_steps("radius-1")
        result3 = get_neighborhood_steps("neighborhood1")

        # Convert to lists of tuples for comparison
        result1 = sorted([tuple(step) for step in result1])
        result2 = sorted([tuple(step) for step in result2])
        result3 = sorted([tuple(step) for step in result3])

        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)

    def test_get_neighborhood_steps_invalid_input(self):
        """Test get_neighborhood_steps with invalid input."""
        # Negative k should raise ValueError
        with self.assertRaises(ValueError):
            get_neighborhood_steps(-1)

        # Too large k should raise ValueError
        with self.assertRaises(ValueError):
            get_neighborhood_steps(128)

        # String with no number should raise ValueError
        with self.assertRaises(ValueError):
            get_neighborhood_steps("no-number-here")

    def test_get_neighborhood_steps_undirected(self):
        """Test get_neighborhood_steps with directed=False."""
        # Undirected k=0 should include only positive directions
        result0 = get_neighborhood_steps(0, directed=False)
        expected0 = np.array([(1, 0), (0, 1)], dtype=np.int8)

        self.assertEqual(sorted([tuple(step) for step in result0]),
                         sorted([tuple(step) for step in expected0]))

        # Undirected k=1 should include specific diagonal directions
        result1 = get_neighborhood_steps(1, directed=False)
        expected1 = np.array([
            (1, 0), (0, 1),  # cardinal
            (1, 1), (-1, 1)  # diagonal
        ], dtype=np.int8)

        self.assertEqual(sorted([tuple(step) for step in result1]),
                         sorted([tuple(step) for step in expected1]))


if __name__ == '__main__':
    unittest.main()

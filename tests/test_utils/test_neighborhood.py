# test_utils/test_neighborhood.py
import unittest
import numpy as np
import math

from pyorps.utils.neighborhood import (
    get_neighborhood_steps, normalize_angle,
    get_move_directions, find_adjacent_directions, elongation_error,
    max_deviation, calculate_errors, find_max_errors
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
        # k=2 should include knight's moves and other moves
        result = get_neighborhood_steps(2)

        # Convert result to a list of tuples for easier checking
        result_tuples = [tuple(step) for step in result]

        # Check specific expected moves are in the result - knight's moves
        knight_moves = [
            (2, 1), (1, 2), (-2, 1), (-1, 2),
            (2, -1), (1, -2), (-2, -1), (-1, -2)
        ]
        for move in knight_moves:
            self.assertIn(move, result_tuples)

        # Based on the current implementation, verify that k=1 moves are included
        k1_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1)
        ]
        for move in k1_moves:
            self.assertIn(move, result_tuples)

        # Check total number of steps matches implementation
        self.assertEqual(len(result), 16)  # 8 from k=1 + 8 knight's moves

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

    def test_normalize_angle(self):
        """Test normalize_angle function."""
        self.assertAlmostEqual(normalize_angle(0), 0)
        self.assertAlmostEqual(normalize_angle(math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(2 * math.pi), 0)
        self.assertAlmostEqual(normalize_angle(4 * math.pi), 0)
        self.assertAlmostEqual(normalize_angle(-math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(-2 * math.pi), 0)
        self.assertAlmostEqual(normalize_angle(2.5 * math.pi), 0.5 * math.pi)

    def test_get_move_directions(self):
        """Test get_move_directions function."""
        # Test with cardinal directions
        moves = np.array([
            [1, 0],  # 0 radians (right)
            [0, 1],  # pi/2 radians (up)
            [-1, 0],  # pi radians (left)
            [0, -1]  # 3pi/2 radians (down)
        ])

        expected = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        result = get_move_directions(moves)

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

        # Test with diagonal directions
        moves = np.array([
            [1, 1],  # pi/4 radians
            [-1, 1],  # 3pi/4 radians
            [-1, -1],  # 5pi/4 radians
            [1, -1]  # 7pi/4 radians
        ])

        expected = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
        result = get_move_directions(moves)

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_find_adjacent_directions(self):
        """Test find_adjacent_directions function."""
        # Test with cardinal directions
        directions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]

        # Test phi between directions
        phi = math.pi / 4  # Between 0 and pi/2
        theta_j, theta_j_plus_1 = find_adjacent_directions(phi, directions)
        self.assertAlmostEqual(theta_j, 0)
        self.assertAlmostEqual(theta_j_plus_1, math.pi / 2)

        # Test phi at exactly a direction
        phi = math.pi / 2  # Exactly at pi/2
        theta_j, theta_j_plus_1 = find_adjacent_directions(phi, directions)
        self.assertAlmostEqual(theta_j, 0)
        self.assertAlmostEqual(theta_j_plus_1, math.pi)

        # Test phi wrap around
        phi = 7 * math.pi / 4  # Between 3pi/2 and 2pi (which wraps to 0)
        theta_j, theta_j_plus_1 = find_adjacent_directions(phi, directions)
        self.assertAlmostEqual(theta_j, 3 * math.pi / 2)
        self.assertAlmostEqual(theta_j_plus_1, 2 * math.pi)  # Implementation returns 2π, not 0

    def test_elongation_error(self):
        """Test elongation_error function."""
        # Test with 90 degree angle between directions
        theta_j = 0
        theta_j_plus_1 = math.pi / 2
        phi = math.pi / 4  # Exactly in the middle

        # For phi exactly in the middle, elongation should be sqrt(2)
        expected = math.sqrt(2)
        result = elongation_error(theta_j, theta_j_plus_1, phi)
        self.assertAlmostEqual(result, expected, places=5)

        # Test with exactly parallel directions
        theta_j = 0
        theta_j_plus_1 = 0  # Same angle
        phi = 0
        with self.assertRaises(ValueError):
            elongation_error(theta_j, theta_j_plus_1, phi)

    def test_max_deviation(self):
        """Test max_deviation function."""
        # Test with 90 degree angle between directions
        theta_j = 0
        theta_j_plus_1 = math.pi / 2
        phi = math.pi / 4  # Exactly in the middle

        # For phi exactly in the middle, maximum deviation should be 0.5
        expected = 0.5
        result = max_deviation(theta_j, theta_j_plus_1, phi)
        self.assertAlmostEqual(result, expected, places=5)

        # Test with exactly parallel directions
        theta_j = 0
        theta_j_plus_1 = 0  # Same angle
        phi = 0
        with self.assertRaises(ValueError):
            max_deviation(theta_j, theta_j_plus_1, phi)

    def test_calculate_errors(self):
        """Test calculate_errors function."""
        # Test with cardinal directions
        directions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        phi = math.pi / 4  # Between 0 and pi/2

        result = calculate_errors(directions, phi)

        self.assertIn('elongation_error', result)
        self.assertIn('max_deviation', result)
        self.assertIn('phi_degrees', result)
        self.assertIn('theta_j_degrees', result)
        self.assertIn('theta_j_plus_1_degrees', result)

        # Check values
        self.assertAlmostEqual(result['elongation_error'], math.sqrt(2), places=5)
        self.assertAlmostEqual(result['max_deviation'], 0.5, places=5)
        self.assertAlmostEqual(result['phi_degrees'], 45.0)
        self.assertAlmostEqual(result['theta_j_degrees'], 0.0)
        self.assertAlmostEqual(result['theta_j_plus_1_degrees'], 90.0)

    def test_find_max_errors(self):
        """Test find_max_errors function."""
        # Test with cardinal directions
        directions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]

        result = find_max_errors(directions)

        self.assertIn('max_elongation', result)
        self.assertIn('max_elongation_phi_degrees', result)
        self.assertIn('max_deviation', result)
        self.assertIn('max_deviation_phi_degrees', result)

        # For evenly spaced directions (like cardinal), max elongation should be sqrt(2)
        # and max deviation should be 0.5 at all midpoints
        self.assertAlmostEqual(result['max_elongation'], math.sqrt(2), places=5)
        self.assertAlmostEqual(result['max_deviation'], 0.5, places=5)


if __name__ == '__main__':
    unittest.main()

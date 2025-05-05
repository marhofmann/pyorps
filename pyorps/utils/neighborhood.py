import math
import re

import numpy as np


def get_neighborhood_steps(k: int, directed: bool = True) -> np.ndarray:
    """
    Generate the steps for a k-neighborhood.

    Parameters:
        k (int): The neighborhood parameter (k >= 0)
        directed (bool): If True, includes all possible step directions;
                         if False, includes a minimal set of steps that ensures
                         bidirectional connectivity in the graph

    Returns:
        numpy.ndarray: A numpy array with dtype int8 containing all steps
    """
    if isinstance(k, str):
        numbers = re.findall(r'^\D*(\d+)', k)
        if not numbers:
            raise ValueError("k must be an integer or neighbourhood string!")
        else:
            _k = int(numbers[0])
    else:
        _k = k

    if _k < 0:
        raise ValueError("k must be non-negative")
    if _k > 127:
        raise ValueError("k is too large for int8 dtype (max value is 127)")

    # Generate all steps with full directionality
    steps = _generate_full_steps(_k, {}, directed)
    return np.array(list(steps), dtype=np.int8)


def _generate_full_steps(k, memo, directed):
    """
    Generate the complete set of steps for neighborhood k. Recursive function as described in
    """
    if k in memo:
        return memo[k]

    k = int(k)

    if k == 0:
        # R_0: cardinal directions
        steps = {(1, 0), (0, 1)}
        steps |= {(-1, 0), (0, -1)} if directed else set()

    elif k == 1:
        # R_1: R_0 plus diagonal directions
        steps = _generate_full_steps(0, memo, directed)
        steps |= {(1, 1), (-1, 1)}
        steps |= {(1, -1), (-1, -1)} if directed else set()
    else:
        # For k > 1: R_k = R_{k-1} ∪ N_k
        prev_steps = _generate_full_steps(k-1, memo, directed)
        new_steps = set()

        # Check boundary points
        for i in range(-k, k+1):
            for x, y in [(i, k), (i, -k), (k, i), (-k, i)]:
                # Skip (0,0) and points already in prev_steps
                if (x, y) in prev_steps or x == 0 or y == 0:
                    continue

                # Check if point is a multiple of a previous step
                gcd = math.gcd(abs(x) if x != 0 else 1, abs(y) if y != 0 else 1)
                if gcd == 1 or ((x // gcd, y // gcd) not in prev_steps and (x // -gcd, y // -gcd) not in prev_steps):
                    if directed or ((x, y) not in new_steps and (-x, -y) not in new_steps):
                        new_steps.add((x, y))

        steps = prev_steps | new_steps

    # Cache result
    memo[k] = steps
    return steps


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [0, 2π).
    """
    return angle % (2 * math.pi)


def get_move_directions(moves: np.ndarray[int]) -> list[float]:
    """
    Get all possible move directions in radians for a given move set.

    Args:
        moves: List of move vectors

    Returns:
        A sorted list of angles in radians [0, 2π)
    """
    directions = [normalize_angle(math.atan2(move[1], move[0])) for move in moves]

    # Remove duplicates and sort
    return sorted(list(set(directions)))


def find_adjacent_directions(phi: float, directions: list[float]) -> tuple[float, float]:
    """
    Find the adjacent directions θ_j and θ_{j+1} such that θ_j < φ < θ_{j+1}.

    Args:
        phi: The path direction in radians
        directions: Sorted list of all possible move directions in radians

    Returns:
        A tuple (θ_j, θ_{j+1})
    """
    # Normalize phi to [0, 2π)
    phi = normalize_angle(phi)

    # Handle the case where phi exactly matches a direction
    if phi in directions:
        idx = directions.index(phi)
        # Use adjacent directions
        return directions[idx - 1], directions[(idx + 1) % len(directions)]

    # Find the adjacent directions
    for i in range(len(directions)):
        # Handle the wrap-around case
        next_i = (i + 1) % len(directions)

        curr_dir = directions[i]
        next_dir = directions[next_i]

        # Handle the wrap-around for angles
        if next_i == 0:  # If next_dir is at the beginning
            next_dir += 2 * math.pi

        if curr_dir <= phi < next_dir:
            return curr_dir, next_dir

    # This should not happen if directions list is properly sorted and normalized
    raise ValueError(f"Could not find adjacent directions for phi={phi}")


def elongation_error(theta_j: float, theta_j_plus_1: float, phi: float) -> float:
    """
    Calculate the elongation error for a lattice path.

    The elongation error is given by:
    e(φ) = (sin(θ_{j+1} - φ) + sin(φ - θ_j)) / sin(θ_{j+1} - θ_j)

    References:
        Goodchild, M. F.: ‘An evaluation of lattice solutions to the problem of corridor location’,
        Environment and Planning A: Economy and Space, 1977, 9, (7), pp 727-738
    """
    numerator = math.sin(theta_j_plus_1 - phi) + math.sin(phi - theta_j)
    denominator = math.sin(theta_j_plus_1 - theta_j)

    if abs(denominator) < 1e-10:
        raise ValueError("Denominator is zero, directions may be identical")

    return numerator / denominator


def max_deviation(theta_j: float, theta_j_plus_1: float, phi: float) -> float:
    """
    Calculate the maximum deviation for a lattice path.

    The maximum deviation is given by:
    δ(φ) = (sin(θ_{j+1} - φ) * sin(φ - θ_j)) / sin(θ_{j+1} - θ_j)

    References:
        Goodchild, M. F.: ‘An evaluation of lattice solutions to the problem of corridor location’,
        Environment and Planning A: Economy and Space, 1977, 9, (7), pp 727-738
    """
    numerator = math.sin(theta_j_plus_1 - phi) * math.sin(phi - theta_j)
    denominator = math.sin(theta_j_plus_1 - theta_j)

    if abs(denominator) < 1e-10:
        raise ValueError("Denominator is zero, directions may be identical")

    return numerator / denominator


def calculate_errors(directions: list[float], phi: float) -> dict[str, float]:
    """
    Calculate elongation error and maximum deviation for a given set of directions and path angle.

    Args:
        directions: Sorted list of all possible move directions in radians
        phi: The path direction in radians

    Returns:
        A dictionary with the calculated errors
    """
    theta_j, theta_j_plus_1 = find_adjacent_directions(phi, directions)

    e = elongation_error(theta_j, theta_j_plus_1, phi)
    d = max_deviation(theta_j, theta_j_plus_1, phi)

    # Convert to degrees for better readability
    phi_deg = phi * 180 / math.pi
    theta_j_deg = theta_j * 180 / math.pi
    theta_j_plus_1_deg = theta_j_plus_1 * 180 / math.pi

    return {
        'elongation_error': e,
        'max_deviation': d,
        'phi_degrees': phi_deg,
        'theta_j_degrees': theta_j_deg,
        'theta_j_plus_1_degrees': theta_j_plus_1_deg
    }


def find_max_errors(directions: list[float]) -> dict[str, float]:
    """
    Find the maximum elongation error and maximum deviation for a given set of directions.

    Args:
        directions: Sorted list of all possible move directions in radians

    Returns:
        A dictionary with the maximum calculated errors
    """
    max_e = 0
    max_d = 0
    max_e_phi = 0
    max_d_phi = 0
    max_e_theta_j = 0
    max_e_theta_j_plus_1 = 0
    max_d_theta_j = 0
    max_d_theta_j_plus_1 = 0

    # Check at the midpoint between each adjacent pair of directions
    for i in range(len(directions)):
        theta_j = directions[i]
        theta_j_plus_1 = directions[(i + 1) % len(directions)]

        # Ensure theta_j < theta_j_plus_1
        if theta_j_plus_1 <= theta_j:
            theta_j_plus_1 += 2 * math.pi

        # Midpoint angle (where both errors are maximized)
        phi = (theta_j + theta_j_plus_1) / 2

        e = elongation_error(theta_j, theta_j_plus_1, phi)
        d = max_deviation(theta_j, theta_j_plus_1, phi)

        if e > max_e:
            max_e = e
            max_e_phi = phi
            max_e_theta_j = theta_j
            max_e_theta_j_plus_1 = theta_j_plus_1

        if d > max_d:
            max_d = d
            max_d_phi = phi
            max_d_theta_j = theta_j
            max_d_theta_j_plus_1 = theta_j_plus_1

    # Convert angles to degrees for better readability
    return {
        'max_elongation': max_e,
        'max_elongation_phi_degrees': max_e_phi * 180 / math.pi,
        'max_elongation_theta_j_degrees': max_e_theta_j * 180 / math.pi,
        'max_elongation_theta_j_plus_1_degrees': max_e_theta_j_plus_1 * 180 / math.pi,
        'max_deviation': max_d,
        'max_deviation_phi_degrees': max_d_phi * 180 / math.pi,
        'max_deviation_theta_j_degrees': max_d_theta_j * 180 / math.pi,
        'max_deviation_theta_j_plus_1_degrees': max_d_theta_j_plus_1 * 180 / math.pi
    }


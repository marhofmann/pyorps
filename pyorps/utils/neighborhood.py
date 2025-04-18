"""
This file defines different kinds of search spaces. The search space determines which cells can be directly reached
from a raster cell by an array of steps. It is also referred to as "neighborhood".
"""
from typing import Any
import re

# Third party
from numpy import array, int8, ndarray, dtype, sum, abs, uint32

import numpy as np
import math

from pyorps.core import CostAssumptionsError


def get_neighborhood_steps(k, directed=True):
    """
    Generate the steps for a k-neighborhood.

    Parameters:
        k (int): The neighborhood parameter (k >= 0)
        directed (bool): If True, includes steps in all directions;
                         if False, only includes steps with non-negative coordinates

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

    # Generate steps directly with direction control
    steps = _steps_recursive(k, directed, {})

    return np.array(list(steps), dtype=np.int8)


def _steps_recursive(k, directed, memo):
    """
    Recursive helper to compute the step set R_k with direction control.
    Only generates steps according to the directed parameter.
    """
    # Create a unique key for memoization
    key = (k, directed)
    if key in memo:
        return memo[key]

    if k == 0:
        # R_0: cardinal directions (filtered if not directed)
        if directed:
            steps = {(1, 0), (0, 1), (-1, 0), (0, -1)}
        else:
            steps = {(1, 0), (0, 1)}
    elif k == 1:
        # R_1: R_0 plus diagonal directions (filtered if not directed)
        prev_steps = _steps_recursive(0, directed, memo)
        if directed:
            diagonals = {(1, 1), (-1, 1), (1, -1), (-1, -1)}
        else:
            diagonals = {(1, 1)}
        steps = prev_steps | diagonals
    else:
        # For k > 1: R_k = R_{k-1} ∪ N_k
        prev_steps = _steps_recursive(k - 1, directed, memo)
        new_steps = set()

        # Define the range of coordinates to check based on directed parameter
        if directed:
            i_range = range(-k, k + 1)
            k_values = [k, -k]
        else:
            i_range = range(0, k + 1)
            k_values = [k]

        # Check boundary points
        for i in i_range:
            for kval in k_values:
                # Create points where one coordinate is exactly ±k
                points = []
                if directed or i >= 0:
                    points.append((i, kval))
                if directed or kval >= 0:
                    points.append((kval, i))

                for x, y in points:
                    # Skip duplicates, (0,0), and points in prev_steps
                    if (x, y) in prev_steps or (x == 0 and y == 0):
                        continue

                    # Check if point is a multiple of a previous step
                    gcd = math.gcd(abs(x) if x != 0 else 1, abs(y) if y != 0 else 1)
                    if gcd == 1 or (x // gcd, y // gcd) not in prev_steps:
                        new_steps.add((x, y))

        steps = prev_steps | new_steps

    # Cache result
    memo[key] = steps
    return steps

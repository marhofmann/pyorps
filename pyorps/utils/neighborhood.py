import math
import re

import numpy as np


def get_neighborhood_steps(k, directed=True):
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

    # Generate all steps first with full directionality
    all_steps = _generate_full_steps(_k, {})

    # If directed is False, filter to a minimal set that maintains bidirectional connectivity
    if not directed:
        steps = _get_undirected_subset(all_steps)
    else:
        steps = all_steps

    return np.array(list(steps), dtype=np.int8)


def _generate_full_steps(k, memo):
    """Generate the complete set of steps for neighborhood k."""
    if k in memo:
        return memo[k]

    k = int(k)

    if k == 0:
        # R_0: cardinal directions
        steps = {(1, 0), (0, 1), (-1, 0), (0, -1)}
    elif k == 1:
        # R_1: R_0 plus diagonal directions
        steps = _generate_full_steps(0, memo) | {(1, 1), (-1, 1), (1, -1), (-1, -1)}
    else:
        # For k > 1: R_k = R_{k-1} ∪ N_k
        prev_steps = _generate_full_steps(k-1, memo)
        new_steps = set()

        # Check boundary points
        for i in range(-k, k+1):
            for x, y in [(i, k), (i, -k), (k, i), (-k, i)]:
                # Skip (0,0) and points already in prev_steps
                if (x, y) in prev_steps or (x == 0 and y == 0):
                    continue

                # Check if point is a multiple of a previous step
                gcd = math.gcd(abs(x) if x != 0 else 1, abs(y) if y != 0 else 1)
                if gcd == 1 or (x // gcd, y // gcd) not in prev_steps:
                    new_steps.add((x, y))

        steps = prev_steps | new_steps

    # Cache result
    memo[k] = steps
    return steps


def _get_undirected_subset(all_steps):
    """
    Return a subset of steps that ensures bidirectional connectivity.
    For k=0: Include only positive cardinal directions
    For k=1: Include specific diagonal directions as specified
    For k>=2: Include specific steps as per the specified examples
    """
    # Start with the basic positive cardinal directions
    subset = {(1, 0), (0, 1)}

    # Add diagonal directions
    if (1, 1) in all_steps:
        subset.add((1, 1))
        subset.add((-1, 1))

    # Add higher-order steps based on the examples
    for x, y in all_steps:
        # Skip already added steps
        if (x, y) in subset:
            continue

        # Skip basic cardinal and diagonal directions
        if abs(x) <= 1 and abs(y) <= 1:
            continue

        # For k>=2, add specific steps according to the pattern
        if (x > 0 and y > 0) or (x < 0 < y) or (x > 0 > y):
            # Check that this isn't a multiple of an existing step
            gcd = math.gcd(abs(x), abs(y))
            reduced = (x // gcd, y // gcd)
            if reduced not in subset:
                subset.add((x, y))

    return subset

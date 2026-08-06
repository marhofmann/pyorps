"""GPU step lookup table preparation.

Computes intermediate cells and cost factors for all step directions,
formatted for GPU kernel consumption. Pure Python/NumPy (no Numba/Cython).
"""

import math
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def _intermediate_steps_cpu(dr: int, dc: int) -> np.ndarray:
    """Bresenham-like intermediate cells for a (dr, dc) step.

    Returns (N, 2) int8 array of (row_offset, col_offset) pairs.
    Matches ``_traversal_numba.intermediate_steps_numba``.
    """
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    if abs_dr + abs_dc <= 1:
        return np.zeros((0, 2), dtype=np.int8)

    k = max(abs_dr, abs_dc)
    if k == 1:
        return np.array([[dr, 0], [0, dc]], dtype=np.int8)

    result = np.zeros((2 * (k - 1), 2), dtype=np.int8)
    for i in range(k - 1):
        dr_k = (i + 1) * dr / k
        dc_k = (i + 1) * dc / k
        idx = i * 2
        result[idx, 0] = np.int8(math.floor(dr_k))
        result[idx, 1] = np.int8(math.floor(dc_k))
        result[idx + 1, 0] = np.int8(math.ceil(dr_k))
        result[idx + 1, 1] = np.int8(math.ceil(dc_k))
    return result


def prepare_step_lookup_tables(steps: np.ndarray):
    """Pre-compute intermediate cells and cost factors for all step directions.

    Parameters
    ----------
    steps : (n_dirs, 2) array
        Step directions as (dr, dc) pairs.

    Returns
    -------
    steps_i8 : (n_dirs, 2) int8
    intermediates_lut : (n_dirs, max_inter, 2) int16
    n_intermediates : (n_dirs,) int32
    cost_factors : (n_dirs,) float32
    """
    n_dirs = len(steps)

    # Compute intermediates for each direction
    inter_lists = []
    max_inter = 0
    for i in range(n_dirs):
        dr, dc = int(steps[i, 0]), int(steps[i, 1])
        inter = _intermediate_steps_cpu(dr, dc)
        inter_lists.append(inter)
        max_inter = max(max_inter, len(inter))

    if max_inter == 0:
        max_inter = 1  # avoid zero-sized dimension

    # Pack into fixed-size arrays
    intermediates_lut = np.zeros((n_dirs, max_inter, 2), dtype=np.int16)
    n_intermediates = np.zeros(n_dirs, dtype=np.int32)
    cost_factors = np.zeros(n_dirs, dtype=np.float32)
    steps_i8 = steps.astype(np.int8)

    for i in range(n_dirs):
        inter = inter_lists[i]
        n_inter = len(inter)
        n_intermediates[i] = n_inter
        if n_inter > 0:
            intermediates_lut[i, :n_inter, :] = inter.astype(np.int16)

        dr, dc = float(steps[i, 0]), float(steps[i, 1])
        dist = math.sqrt(dr * dr + dc * dc)
        cost_factors[i] = dist / (2.0 + n_inter)

    return steps_i8, intermediates_lut, n_intermediates, cost_factors

"""Lower-bound pieces on a position lattice (paper math.md M4, M7.6, M9.1; plan 2026-09-23 sections 3.1, 3.7).

Three small, data-independent pieces of the bound-and-prune driver:

:func:`float32_lower_bound`
    Turns a float32 GPU distance field into a rigorous float64 lower bound, ``d_f (1 - k_max eta)`` (M9.1).
:func:`steiner_lower_bound`
    A cheap admissible bound of the trench of the collector at every root cell from turbine-to-cell distances: the
    larger of the farthest-turbine distance and ``MST / (2 (1 - 1/k))`` of the metric closure of the ``k`` terminals
    (turbines and root). The trench set of any collector is connected and contains all terminals, so its weight is at
    least the Steiner tree weight, and the MST of the metric closure is at most ``2 (1 - 1/k)`` times that (M7.6a).
:func:`offset_min_filter`
    The offset-free relaxation: a pad terminal lies within the pad's half-diagonal of its anchor, so the bound of a
    position is the minimum of the field over a square of that half-width (M4, eq. 4.2 with unknown offsets).
"""

from __future__ import annotations

import numpy as np

__all__ = ["ETA_FLOAT32", "float32_lower_bound", "offset_min_filter", "steiner_lower_bound"]

ETA_FLOAT32 = 3.0 * 2.0 ** -24          # per-step relative error of a float32 drain (math.md M9.1)
NODATA = 1e29                            # fields store 3.4e38 for unreached cells


def float32_lower_bound(field: np.ndarray, k_max: int, *, eta: float = ETA_FLOAT32) -> np.ndarray:
    """Float64 lower bound of the exact field from a float32 drain: ``d (1 - k_max eta)``.

    Unreached cells become inf.
    """
    if k_max * eta >= 1.0:
        raise ValueError("k_max * eta must be below 1")
    d = np.asarray(field, dtype=np.float64)
    return np.where(d >= NODATA, np.inf, d * (1.0 - k_max * eta))


def steiner_lower_bound(dist: np.ndarray, terminal_dist: np.ndarray) -> np.ndarray:
    """Admissible bound of the trench weight connecting all turbines to a root cell, for every cell at once.

    Args:
        dist: ``(T, ...)`` lower bounds of the distance from turbine ``i`` to every cell (``inf`` if unreachable).
        terminal_dist: ``(T, T)`` lower bounds of the distances between the turbines (symmetric, zero diagonal).

    Returns:
        ``max( max_i dist_i , MST({root} + turbines) / (2 (1 - 1/(T + 1))) )`` per cell, in the units of ``dist``.
    """
    dist = np.asarray(dist, dtype=np.float64)
    tt = np.asarray(terminal_dist, dtype=np.float64)
    t = dist.shape[0]
    if tt.shape != (t, t) or not np.allclose(tt, tt.T, equal_nan=False) or np.any(np.diag(tt) != 0):
        raise ValueError("terminal_dist must be a symmetric (T, T) matrix with zero diagonal")
    shape = dist.shape[1:]
    flat = dist.reshape(t, -1)
    far = flat.max(axis=0)
    # Prim from the root over the metric closure: node 0 is the root, nodes 1..T the turbines.
    key = flat.copy()                                   # key[i]: cheapest edge from the tree to turbine i
    in_tree = np.zeros((t, flat.shape[1]), dtype=bool)
    mst = np.zeros(flat.shape[1])
    for _ in range(t):
        masked = np.where(in_tree, np.inf, key)
        j = masked.argmin(axis=0)
        cols = np.arange(flat.shape[1])
        step = masked[j, cols]
        mst += step
        in_tree[j, cols] = True
        key = np.minimum(key, tt[j].T)                  # tt[j] has shape (N, T); transpose to (T, N)
    k = t + 1
    bound = np.maximum(far, mst / (2.0 * (1.0 - 1.0 / k)))
    return bound.reshape(shape)


def offset_min_filter(field: np.ndarray, radius_cells: int) -> np.ndarray:
    """Minimum of ``field`` over the square of half-width ``radius_cells`` around every cell (inf stays inf)."""
    from scipy.ndimage import minimum_filter

    if radius_cells < 0:
        raise ValueError("radius_cells must be >= 0")
    if radius_cells == 0:
        return np.asarray(field).copy()
    return minimum_filter(np.asarray(field), size=2 * int(radius_cells) + 1, mode="nearest")

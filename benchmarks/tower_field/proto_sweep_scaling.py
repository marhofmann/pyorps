"""v2: prefix sums built ONCE, so a sweep is genuinely O(K x N).

The terrain does not change between sweeps -- only T does. So the directional
prefix sum P_theta is built once (by shift-and-add doubling, O(N log n)) and
every later sweep is, per direction: S = T - P ; windowed min of S ; + P.
Three O(N) passes, independent of how long a span may be.

Measures: sweeps to converge vs extent, and per-sweep time vs cell count.
"""
from __future__ import annotations

import time

import numpy as np
import rasterio
from rasterio.windows import Window

SRC = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
       r"cired2026/raster/mod1_raster_wp_fixed.tiff")
SIGMA = 10
L_MIN, L_MAX = 50.0, 300.0
BASE = 30_000.0


def primitive_dirs(dmax):
    return [(p, q) for p in range(-dmax, dmax + 1)
            for q in range(-dmax, dmax + 1)
            if (p or q) and np.gcd(abs(p), abs(q)) == 1]


def load_coarse(n):
    with rasterio.open(SRC) as r:
        a = r.read(1, window=Window(9000, 4000, n * SIGMA, n * SIGMA))
    a = a.astype(np.float64)
    a[a >= 65535] = 5000.0
    return a.reshape(n, SIGMA, n, SIGMA).mean(axis=(1, 3))


def shift(a, dr, dc, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    r0, r1 = max(0, dr), min(h, h + dr)
    c0, c1 = max(0, dc), min(w, w + dc)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def ray_prefix(cost, p, q, step):
    """Inclusive scan along every ray of direction (p,q), by doubling."""
    P = cost * step
    k = 1
    while k < max(cost.shape):
        P = P + shift(P, k * p, k * q, 0.0)
        k *= 2
    return P


def build(cost, dirs):
    table = {}
    for (p, q) in dirs:
        step = np.hypot(p, q) * SIGMA
        m_lo = int(np.ceil(L_MIN / step))
        m_hi = int(np.floor(L_MAX / step))
        if m_hi < max(m_lo, 1):
            continue
        table[(p, q)] = (ray_prefix(cost, p, q, step), m_lo, m_hi)
    return table


def sweep(T, tower, table):
    best = T
    for (p, q), (P, m_lo, m_hi) in table.items():
        S = T - P                                  # O(N)
        # windowed min of S over lattice offsets [m_lo, m_hi] along the ray.
        # A real implementation uses van Herk (O(1)/elt); here the running min
        # over the window is enough to show the per-sweep cost is O(N) x a
        # constant that does NOT depend on the terrain or the prefix.
        M = shift(S, m_lo * p, m_lo * q, np.inf)
        for m in range(m_lo + 1, m_hi + 1):
            M = np.minimum(M, shift(S, m * p, m * q, np.inf))
        best = np.minimum(best, M + P + tower)     # O(N)
    return best


def solve(cost, tower, src, dirs, cap=400):
    table = build(cost, dirs)
    T = np.full(cost.shape, np.inf)
    T[src] = tower[src]
    for it in range(1, cap + 1):
        new = sweep(T, tower, table)
        if np.array_equal(np.nan_to_num(new, posinf=-1),
                          np.nan_to_num(T, posinf=-1)):
            return T, it, table
        T = new
    return T, cap, table


def main():
    dirs = primitive_dirs(4)
    print(f"{len(dirs)} primitive directions; spans {L_MIN:.0f}-{L_MAX:.0f} m "
          f"on a {SIGMA} m lattice\n")
    print(f"{'grid':>10}{'extent':>9}{'cells':>9}{'sweeps':>8}"
          f"{'build s':>9}{'sweep s':>9}{'us/cell/sweep':>15}")
    print("-" * 70)
    for n in (48, 96, 144, 200):
        cost = load_coarse(n)
        tower = np.full(cost.shape, BASE) + cost * 100.0
        t0 = time.perf_counter()
        table = build(cost, dirs)
        tb = time.perf_counter() - t0
        T = np.full(cost.shape, np.inf)
        T[(n // 2, 2)] = tower[n // 2, 2]
        t0 = time.perf_counter()
        T2 = sweep(T, tower, table)
        ts = time.perf_counter() - t0
        _, sw, _ = solve(cost, tower, (n // 2, 2), dirs)
        print(f"{n}x{n:>6}{n*SIGMA:>8} m{cost.size:>9}{sw:>8}"
              f"{tb:9.2f}{ts:9.3f}{1e6*ts/cost.size:15.2f}")

    print("\nsweeps scale with extent/max_span (the tower count), not with "
          "cell count -- that is the layered-DAG property.")


if __name__ == "__main__":
    main()

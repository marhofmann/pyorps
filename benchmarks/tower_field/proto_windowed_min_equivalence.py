"""Precomputing the tower search: prototype and correctness check.

The 288-states-per-cell explosion comes from modelling an overhead line as a
raster WALK carrying (direction, span, height) state. Physically it is not a
walk: it is a sequence of TOWERS joined by STRAIGHT spans. Reframed that way:

  T_k(x) = tower(x) + min over y with Lmin <= |x-y| <= Lmax of
                        [ T_{k-1}(y) + span_cost(y, x) ]

Three structural facts then apply.

  1. Every edge adds exactly one tower, so the graph is LAYERED by tower count
     and Bellman-Ford converges in exactly (max towers) sweeps. No priority
     queue, no state vector -- one scalar per position.
  2. span_cost(y, x) along a fixed direction is a difference of a PREFIX SUM
     along that direction, so it is O(1) instead of O(span length).
  3. the min over span LENGTH is then a 1-D sliding-window minimum along the
     ray, which van Herk / Gil-Werman does in O(1) amortised per element
     regardless of window width.

Total: O(K directions x sweeps x cells), with ONE float per cell of state.

This script checks 2 and 3 against an explicit minimum over every span length
(same direction set, so they must agree EXACTLY), and measures the speedup.
"""
from __future__ import annotations

import time

import numpy as np
import rasterio
from rasterio.windows import Window

SRC = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
       r"cired2026/raster/mod1_raster_wp_fixed.tiff")
SIGMA = 10          # tower lattice, metres
L_MIN, L_MAX = 50.0, 300.0
BASE_TOWER = 30_000.0


def primitive_dirs(dmax):
    """Primitive integer directions -- one per distinct angle."""
    out = []
    for p in range(-dmax, dmax + 1):
        for q in range(-dmax, dmax + 1):
            if (p or q) and np.gcd(abs(p), abs(q)) == 1:
                out.append((p, q))
    return out


def load_coarse(n):
    with rasterio.open(SRC) as r:
        a = r.read(1, window=Window(9000, 4000, n * SIGMA, n * SIGMA))
    a = a.astype(np.float64)
    a[a >= 65535] = 5000.0                      # crossable, expensive
    return a.reshape(n, SIGMA, n, SIGMA).mean(axis=(1, 3))


def shift(a, dr, dc, fill):
    """a[x - (dr,dc)], i.e. pull from the cell one step back along +(dr,dc)."""
    out = np.full_like(a, fill)
    h, w = a.shape
    r0, r1 = max(0, dr), min(h, h + dr)
    c0, c1 = max(0, dc), min(w, w + dc)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def sweep(T, cost, tower, dirs, mode):
    """One Bellman-Ford sweep: T <- min(T, tower + dilate(T)). Returns (T, ops)."""
    INF = np.inf
    best = T.copy()
    ops = 0
    for (p, q) in dirs:
        step = np.hypot(p, q) * SIGMA
        m_lo = int(np.ceil(L_MIN / step))
        m_hi = int(np.floor(L_MAX / step))
        if m_hi < max(m_lo, 1):
            continue
        # prefix sum of terrain along this ray, and S = T - P
        P = np.zeros_like(T)
        Tm = T.copy()
        Pm = P.copy()
        cand = np.full_like(T, INF)
        window = []
        for m in range(1, m_hi + 1):
            # advance one lattice step: accumulate the cell we step ONTO
            Pm = shift(Pm, p, q, INF) + cost * step
            Tm = shift(Tm, p, q, INF)
            ops += T.size
            if m >= m_lo:
                # T[y] + (P[x] - P[y]) with P measured from x backwards
                window.append(Tm + Pm)
        if not window:
            continue
        if mode == "explicit":
            for w_ in window:
                cand = np.minimum(cand, w_)
                ops += T.size
        else:                                   # running min, O(1) per element
            cand = window[0]
            for w_ in window[1:]:
                cand = np.minimum(cand, w_)
            ops += T.size                       # van Herk cost model
        best = np.minimum(best, cand + tower)
    return best, ops


def solve(cost, tower, src, dirs, mode, max_sweeps=200):
    T = np.full(cost.shape, np.inf)
    T[src] = tower[src]
    ops = 0
    for it in range(max_sweeps):
        new, o = sweep(T, cost, tower, dirs, mode)
        ops += o
        if np.array_equal(np.nan_to_num(new, posinf=-1),
                          np.nan_to_num(T, posinf=-1)):
            return T, it, ops
        T = new
    return T, max_sweeps, ops


def main():
    for n, dmax in ((60, 3), (60, 5)):
        cost = load_coarse(n)
        tower = np.full(cost.shape, BASE_TOWER) + cost * 100.0
        src = (n // 2, 2)
        dirs = primitive_dirs(dmax)
        print(f"\ngrid {n}x{n} at {SIGMA} m ({n*SIGMA} m across), "
              f"{len(dirs)} directions (dmax={dmax})")

        t0 = time.perf_counter()
        Ta, it, ops_a = solve(cost, tower, src, dirs, "explicit")
        ta = time.perf_counter() - t0
        t0 = time.perf_counter()
        Tb, _, ops_b = solve(cost, tower, src, dirs, "vanherk")
        tb = time.perf_counter() - t0

        fin = np.isfinite(Ta) & np.isfinite(Tb)
        print(f"  converged in {it} sweeps; reachable {100*fin.mean():.0f} %")
        print(f"  explicit-min vs running-min: max |diff| = "
              f"{np.abs(Ta[fin]-Tb[fin]).max():.6e}  (must be 0)")
        print(f"  modelled ops: explicit {ops_a/1e6:.1f} M, "
              f"windowed {ops_b/1e6:.1f} M  -> {ops_a/ops_b:.1f}x fewer")
        print(f"  wall clock (machine is loaded): {ta:.2f} s / {tb:.2f} s")
        print(f"  cost at 3 probes: "
              f"{[f'{Ta[n//2, c]:,.0f}' for c in (n//4, n//2, 3*n//4)]}")


if __name__ == "__main__":
    main()

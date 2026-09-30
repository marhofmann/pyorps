"""Slope indicatrix / stencil study for the Tier A anisotropic eikonal.

Reproduces in-repo, with no GPU, the numerical results the Tier A plan
rests on:

  G1  The Tier A term is EXACTLY Riemannian. The unconditional 3D-length
      stretch sqrt(1 + (s/100)^2) equals sqrt(d^T M d) / |d| for
      M = c^2 (I + q q^T); Sherman-Morrison inverts it in closed form.
  G2  Tier B is NOT. A configured multiplier curve makes the indicatrix
      non-convex, so a continuum solver silently solves the convexified
      problem F** instead — the switchback discount is printed here.
  G3  The 4-point (axis-quadrant) stencil is metric-OBTUSE for every
      non-axis-aligned slope and carries a fixed directional bias; the
      8-simplex stencil is exact on linear fields up to
      |q| = sqrt(2(1+sqrt2)) = 2.19737, i.e. kappa = 1 + sqrt(2).
  G4  Every 8-simplex update raises T by at least c (needed for the
      targeted early exit to stay exact).

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_slope_indicatrix.py
"""
from __future__ import annotations

import numpy as np

from pyorps.core.objective import Objective, GradientOptions
from pyorps.utils.eikonal_gpu import ANISO_OFFSETS, Q_ACUTE_LIMIT

OFF = np.asarray(ANISO_OFFSETS, dtype=float)
LEN2 = (OFF ** 2).sum(axis=1)


# ---------------------------------------------------------------------------
# the local solver, in numpy (host twin of aniso_update8)
# ---------------------------------------------------------------------------

def simplex_pairs(n_simplex: int):
    if n_simplex == 8:
        return [(k, (k + 1) % 8) for k in range(8)]
    return [(0, 2), (2, 4), (4, 6), (6, 0)]      # the four axis quadrants


def local_update(tn, c, q, pairs):
    qe = OFF @ q
    best = np.inf
    for k in range(8):
        if tn[k] < 1e29:
            best = min(best, tn[k] + c * np.sqrt(LEN2[k] + qe[k] ** 2))
    for i, j in pairs:
        t1, t2 = tn[i], tn[j]
        if t1 >= 1e29 or t2 >= 1e29:
            continue
        m11 = LEN2[i] + qe[i] ** 2
        m22 = LEN2[j] + qe[j] ** 2
        m12 = OFF[i] @ OFF[j] + qe[i] * qe[j]
        ss = m11 - 2 * m12 + m22
        det = m11 * m22 - m12 * m12
        d = t1 - t2
        disc = c * c * ss - d * d
        if disc < 0:
            continue
        t = ((m22 - m12) * t1 + (m11 - m12) * t2
             + np.sqrt(det * disc)) / ss
        a, b = t - t1, t - t2
        if a < 0 or b < 0:
            continue
        if m22 * a - m12 * b < 0 or m11 * b - m12 * a < 0:
            continue
        best = min(best, t)
    return best


def stencil_error(grade, n_simplex, azis=37, dirs=181):
    """Max relative over-estimate on EXACT linear fields."""
    worst = 0.0
    pairs = simplex_pairs(n_simplex)
    for phi in np.linspace(0, np.pi / 2, azis):
        q = grade * np.array([np.cos(phi), np.sin(phi)])
        d_inv = np.linalg.inv(np.eye(2) + np.outer(q, q))
        for th in np.linspace(0, 2 * np.pi, dirs):
            u = np.array([np.cos(th), np.sin(th)])
            p = u / np.sqrt(u @ d_inv @ u)      # p^T M^-1 p = 1
            tn = OFF @ p                        # exact T, T(cell) = 0
            got = local_update(tn, 1.0, q, pairs)
            worst = max(worst, abs(got) / max(np.abs(tn).max(), 1e-12))
    return worst


# ---------------------------------------------------------------------------
# G1: the Tier A term is exactly Riemannian
# ---------------------------------------------------------------------------

def g1_riemannian():
    print("=" * 74)
    print("G1  Tier A is exactly Riemannian:  M = c^2 (I + q q^T)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    res_sm = res_len = res_stretch = 0.0
    for _ in range(20000):
        q = rng.normal(size=2) * 1.0
        d = rng.normal(size=2)
        c = float(np.exp(rng.normal()))
        m = c * c * (np.eye(2) + np.outer(q, q))
        d_inv = (1.0 / (c * c)) * (np.eye(2)
                                   - np.outer(q, q) / (1.0 + q @ q))
        res_sm = max(res_sm, float(np.abs(m @ d_inv - np.eye(2)).max()))
        # 3D length of a displacement d over ground of gradient q
        lhs = c * np.hypot(np.linalg.norm(d), q @ d)
        res_len = max(res_len, abs(lhs - np.sqrt(d @ m @ d)))
        # ... and that IS pyorps' unconditional stretch
        u = d / np.linalg.norm(d)
        s_pct = 100.0 * abs(q @ u)
        stretch = np.sqrt(1.0 + (s_pct / 100.0) ** 2)
        res_stretch = max(res_stretch, abs(
            np.sqrt(u @ m @ u) / c - stretch))
    print(f"  Sherman-Morrison  max |M M^-1 - I|        = {res_sm:.3e}")
    print(f"  3D length         max |c*hypot - sqrt|    = {res_len:.3e}")
    print(f"  == pyorps stretch max |sqrt(u'Mu)/c - g|  = {res_stretch:.3e}")
    print("  -> the DEFAULT GradientOptions term is a metric. Exactly.")


# ---------------------------------------------------------------------------
# G2: Tier B is not (non-convex indicatrix -> switchback pricing)
# ---------------------------------------------------------------------------

def g2_tier_b():
    print()
    print("=" * 74)
    print("G2  Tier B is NOT a metric: the convexified problem F** is")
    print("    what a continuum solver would silently minimise.")
    print("=" * 74)
    print(f"  {'multiplier':>14} {'peak grade':>11} {'F':>10} "
          f"{'F**':>10} {'discount':>10}")
    steps = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.int8)
    for model in ("power", "squared", "exponential", "sigmoid", "energy"):
        obj = Objective({"cost": 1.0}, GradientOptions(multiplier=model))
        luts = obj.build_gradient_luts(steps, 10.0)
        s = (np.arange(luts.n_bins) + 0.5) * luts.bin_width_pct
        mult = np.asarray(luts.mult, dtype=np.float64)
        for peak in (25.0, 50.0, 100.0):
            k = int(np.searchsorted(s, peak))
            if k >= len(s):
                continue
            # cost per unit of HORIZONTAL progress at grade s, versus the
            # best convex combination of a steeper and a flatter leg that
            # achieves the same net rise (a switchback).
            direct = mult[k]
            i = np.arange(k + 1, len(s))
            j = np.arange(0, k)
            if not len(i) or not len(j):
                continue
            si, sj = s[i][:, None], s[j][None, :]
            # fraction of horizontal run on the steep leg
            lam = (peak - sj) / (si - sj)
            mix = lam * mult[i][:, None] + (1 - lam) * mult[j][None, :]
            mix = np.where((lam >= 0) & (lam <= 1), mix, np.inf)
            best = float(np.nanmin(mix))
            disc = (direct / best - 1.0) * 100.0
            print(f"  {model:>14} {peak:10.0f}% {direct:10.4f} "
                  f"{best:10.4f} {disc:+9.2f}%")
    print("  -> any positive discount means the PDE would price a")
    print("     switchback cheaper than the objective as posed. Refused.")


# ---------------------------------------------------------------------------
# G3: the stencil
# ---------------------------------------------------------------------------

def g3_stencil():
    print()
    print("=" * 74)
    print("G3  Stencil error on exact linear fields (721 dirs x 37 azis)")
    print("=" * 74)
    print(f"  {'|q|':>8} {'grade':>8} {'kappa':>8} {'8-simplex':>12} "
          f"{'4-simplex':>12}")
    for g in (0.10, 0.20, 0.50, 1.00, np.sqrt(2), 2.00, Q_ACUTE_LIMIT,
              2.50, 3.00, 4.00):
        e8 = stencil_error(g, 8)
        e4 = stencil_error(g, 4)
        print(f"  {g:8.4f} {g*100:7.0f}% {np.sqrt(1+g*g):8.4f} "
              f"{e8*100:11.4f}% {e4*100:11.4f}%")

    print()
    print("  Acuteness of the (axis, adjacent-diagonal) pairs reduces to")
    print("  |q_r q_c| <= 1 + min(q_r^2, q_c^2).  Sweep on a 0.01 grid:")
    first = None
    for k in np.arange(0.0, 4.0, 0.01):
        phis = np.linspace(0, np.pi / 2, 721)
        qr, qc = k * np.cos(phis), k * np.sin(phis)
        if np.any(np.abs(qr * qc) > 1 + np.minimum(qr ** 2, qc ** 2)
                  + 1e-12):
            first = k
            break
    print(f"    first failure at |q| = {first:.2f} "
          f"(kappa = {np.sqrt(1+first**2):.4f})")
    print(f"    derived threshold    {Q_ACUTE_LIMIT:.5f} "
          f"(kappa = {1+np.sqrt(2):.5f})")
    print(f"    pyorps default s_max_pct = 200 % -> |q| = 2.0, "
          f"kappa = {np.sqrt(5):.4f}  INSIDE")

    print()
    print("  Why 4 is not enough: a simplex (e1, e2) is metric-acute iff")
    print("  e1' M e2 >= 0, and for the axis quadrants that is")
    print("  +- c^2 q_r q_c -> two of the four are obtuse whenever the")
    print("  slope is not axis-aligned.")
    q = np.array([0.5, 0.5])
    m = np.eye(2) + np.outer(q, q)
    for i, j in simplex_pairs(4):
        print(f"    quadrant ({OFF[i]}, {OFF[j]}): e1'Me2 = "
              f"{OFF[i] @ m @ OFF[j]:+.4f}")


# ---------------------------------------------------------------------------
# G4: minimum metric increment per update
# ---------------------------------------------------------------------------

def g4_increment():
    print()
    print("=" * 74)
    print("G4  Minimum M-length of the 8 simplex chords (c = 1).")
    print("    The targeted early exit needs a fixed positive increment;")
    print("    the 4-point diamond only gives c/sqrt(2) = 0.7071.")
    print("=" * 74)
    ts = np.linspace(0, 1, 501)
    for g in (0.0, 0.5, 1.0, 2.0, Q_ACUTE_LIMIT):
        worst = np.inf
        for phi in np.linspace(0, np.pi / 2, 91):
            q = g * np.array([np.cos(phi), np.sin(phi)])
            m = np.eye(2) + np.outer(q, q)
            for k in range(8):
                e1, e2 = OFF[k], OFF[(k + 1) % 8]
                v = (1 - ts)[:, None] * e1 + ts[:, None] * e2
                worst = min(worst, float(
                    np.sqrt(np.einsum("ij,jk,ik->i", v, m, v)).min()))
        print(f"  |q| = {g:7.4f}:  min ||v||_M = {worst:.4f}")
    print("  -> always >= c, so information travels further per update")
    print("     than the isotropic 4-point stencil, not less.")


if __name__ == "__main__":
    g1_riemannian()
    g2_tier_b()
    g3_stencil()
    g4_increment()

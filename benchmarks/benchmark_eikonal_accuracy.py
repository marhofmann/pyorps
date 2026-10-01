"""
Accuracy benchmark: eikonal block-FIM vs discrete graph LCP (R1 / R2)
with analytic ground truth as the neutral referee where it exists
(eikonal plan section 6.2 — the metrication table — and 6.3, the
honest-caveat measurements).

Cases:
  1. uniform direction sweep — elongation error per solver vs c*r
  2. Snell two-half-plane refraction — error vs the closed form
  3. smooth random fields (Gaussian-blurred noise, several correlation
     lengths, >= 20 seeds) — FIM vs R2 vs R1 relative gaps
  4. terrain-like composite (smooth base + patches + corridors +
     exclusions; a controllable synthetic sibling of case 6) — same gaps
  5. caveats: corner under-estimation near thin barriers, shock
     smearing width, iteration/wall-clock blowup vs cost contrast
  6. real cost raster (examples/data/raster/small_raster.tiff — 10
     land-use cost classes, 51% forbidden) — same gaps over several
     long source/target pairs

Usage:
    .venv/Scripts/python.exe benchmarks/benchmark_eikonal_accuracy.py
    ... [--quick]

Results: JSON into benchmarks/results/, table text to stdout; findings
are curated into benchmarks/EIKONAL_FINDINGS.md.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pyorps.utils.eikonal_gpu import eikonal_raster_gpu, FINITE_LIMIT
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.utils.sssp_gpu import sssp_raster_gpu

R1 = get_neighborhood_steps("r1", directed=True)
R2 = get_neighborhood_steps("r2", directed=True)


def blur(field, sigma):
    """Gaussian blur; scipy if present, separable FFT fallback."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma, mode="reflect")
    except ImportError:
        n = field.shape[0]
        k = np.exp(-0.5 * (np.fft.fftfreq(n) * n / max(sigma, 1e-9)) ** 2)
        return np.real(np.fft.ifft2(np.fft.fft2(field)
                                    * np.outer(k, k)))


def solve_discrete(raster, steps, source):
    dist = sssp_raster_gpu(raster, steps, int(source))
    return dist.reshape(raster.shape)


def solve_fim(raster, source, **kw):
    if np.isscalar(source) or getattr(source, "ndim", 0) == 0:
        source = int(source)
    return eikonal_raster_gpu(raster, source, **kw)


# ----------------------------------------------------------------------
# Case 1: uniform direction sweep
# ----------------------------------------------------------------------

def case_uniform_directions(n=401, v=10, radius=180):
    sr = sc = n // 2
    raster = np.full((n, n), v, dtype=np.uint16)
    src = sr * n + sc

    fields = {
        "fim": solve_fim(raster, src).astype(np.float64),
        "dijkstra_r1": solve_discrete(raster, R1, src).astype(np.float64),
        "dijkstra_r2": solve_discrete(raster, R2, src).astype(np.float64),
    }
    angles = np.deg2rad(np.arange(0.0, 45.01, 1.5))
    rows = []
    for name, field in fields.items():
        errs = []
        for a in angles:
            tr = sr + int(round(radius * np.sin(a)))
            tc = sc + int(round(radius * np.cos(a)))
            exact = v * float(np.hypot(tr - sr, tc - sc))
            errs.append((field[tr, tc] - exact) / exact)
        errs = np.array(errs)
        rows.append(dict(
            solver=name,
            max_elongation_pct=float(errs.max() * 100),
            mean_elongation_pct=float(errs.mean() * 100),
            worst_angle_deg=float(np.rad2deg(angles[int(errs.argmax())])),
        ))
    return rows


# ----------------------------------------------------------------------
# Case 2: Snell refraction
# ----------------------------------------------------------------------

def snell_reference(c1, c2, src, tgt, iface):
    x = np.linspace(min(src[1], tgt[1]) - 100.0,
                    max(src[1], tgt[1]) + 100.0, 400001)
    leg1 = np.hypot(iface - src[0], x - src[1])
    leg2 = np.hypot(tgt[0] - iface, tgt[1] - x)
    return float(np.min(c1 * leg1 + c2 * leg2))


def case_snell(n=400, c1=5, c2=15):
    raster = np.full((n, n), c1, dtype=np.uint16)
    raster[n // 2:, :] = c2
    src = (int(0.15 * n), int(0.2 * n))
    tgt = (int(0.85 * n), int(0.8 * n))
    s_idx = src[0] * n + src[1]
    expected = snell_reference(c1, c2, src, tgt, n // 2 - 0.5)

    rows = []
    for name, field in [
        ("fim", solve_fim(raster, s_idx)),
        ("dijkstra_r1", solve_discrete(raster, R1, s_idx)),
        ("dijkstra_r2", solve_discrete(raster, R2, s_idx)),
    ]:
        got = float(field[tgt])
        rows.append(dict(solver=name, cost=got, analytic=expected,
                         error_pct=(got - expected) / expected * 100))
    return rows


# ----------------------------------------------------------------------
# Cases 3 + 4: smooth random and terrain-like fields (no analytic truth:
# pairwise gaps, expected ordering FIM <= R2 <= R1)
# ----------------------------------------------------------------------

def _compare_three(raster, s_idx, t_idx):
    f = solve_fim(raster, s_idx)
    d1 = solve_discrete(raster, R1, s_idx)
    d2 = solve_discrete(raster, R2, s_idx)
    tf = float(f.ravel()[t_idx])
    t1 = float(d1.ravel()[t_idx])
    t2 = float(d2.ravel()[t_idx])
    if not all(v < FINITE_LIMIT and np.isfinite(v) for v in (tf, t1, t2)):
        return None
    return dict(fim=tf, r1=t1, r2=t2,
                fim_vs_r2_pct=(t2 - tf) / t2 * 100,
                fim_vs_r1_pct=(t1 - tf) / t1 * 100,
                r2_vs_r1_pct=(t1 - t2) / t1 * 100)


def make_smooth(n, sigma, rng):
    base = blur(rng.normal(0, 1, (n, n)), sigma)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    return (1 + base * 199).astype(np.uint16)


def case_smooth_random(n=300, seeds=20, sigmas=(4, 16, 64)):
    results = {}
    for sigma in sigmas:
        gaps = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 * sigma + seed)
            raster = make_smooth(n, sigma, rng)
            cmp_ = _compare_three(raster, 0, n * n - 1)
            if cmp_ is not None:
                gaps.append(cmp_)
        key = f"sigma_{sigma}"
        results[key] = dict(
            n_seeds=len(gaps),
            fim_vs_r2_mean_pct=statistics.mean(
                g["fim_vs_r2_pct"] for g in gaps),
            fim_vs_r2_max_pct=max(g["fim_vs_r2_pct"] for g in gaps),
            fim_vs_r2_min_pct=min(g["fim_vs_r2_pct"] for g in gaps),
            fim_vs_r1_mean_pct=statistics.mean(
                g["fim_vs_r1_pct"] for g in gaps),
            r2_vs_r1_mean_pct=statistics.mean(
                g["r2_vs_r1_pct"] for g in gaps),
            ordering_violations=sum(
                1 for g in gaps
                if g["fim_vs_r2_pct"] < -0.1 or g["r2_vs_r1_pct"] < -0.1),
        )
    return results


def make_terrain(n, rng):
    """Terrain-like composite: smooth base, expensive patches, cheap
    corridors, exclusion blobs — a seedable synthetic sibling of the
    real-raster case (case 6)."""
    raster = make_smooth(n, 16, rng).astype(np.float64)
    for _ in range(12):     # expensive land-use patches
        r0, c0 = rng.integers(0, n - 40, 2)
        h, w = rng.integers(15, 40, 2)
        raster[r0:r0 + h, c0:c0 + w] *= rng.uniform(2, 5)
    for _ in range(3):      # cheap corridors
        c0 = rng.integers(0, n - 8)
        raster[:, c0:c0 + 6] *= 0.2
    out = np.clip(raster, 1, 5000).astype(np.uint16)
    for _ in range(6):      # exclusion blobs
        r0, c0 = rng.integers(20, n - 40, 2)
        h, w = rng.integers(8, 25, 2)
        out[r0:r0 + h, c0:c0 + w] = np.iinfo(np.uint16).max
    out[0, 0] = 100
    out[-1, -1] = 100
    return out


def case_terrain(n=300, seeds=5):
    gaps = []
    for seed in range(seeds):
        rng = np.random.default_rng(777 + seed)
        raster = make_terrain(n, rng)
        cmp_ = _compare_three(raster, 0, n * n - 1)
        if cmp_ is not None:
            gaps.append(cmp_)
    return dict(
        n_seeds=len(gaps),
        fim_vs_r2_mean_pct=statistics.mean(
            g["fim_vs_r2_pct"] for g in gaps),
        fim_vs_r2_max_pct=max(g["fim_vs_r2_pct"] for g in gaps),
        fim_vs_r2_min_pct=min(g["fim_vs_r2_pct"] for g in gaps),
        fim_vs_r1_mean_pct=statistics.mean(
            g["fim_vs_r1_pct"] for g in gaps),
        r2_vs_r1_mean_pct=statistics.mean(
            g["r2_vs_r1_pct"] for g in gaps),
    )


# ----------------------------------------------------------------------
# Case 6: real cost raster (plan 6.2 — "a real cost raster from the
# test data"). No analytic truth: pairwise gaps, expected ordering
# FIM <= R2 <= R1, on real land-use geometry (51% forbidden cells).
# ----------------------------------------------------------------------

REAL_RASTER = (Path(__file__).parent.parent
               / "examples" / "data" / "raster" / "small_raster.tiff")

# fractional (row, col) anchors, snapped to the nearest passable cell
REAL_SOURCES = ((0.20, 0.20), (0.50, 0.50), (0.80, 0.30))
REAL_TARGETS = ((0.25, 0.75), (0.75, 0.70), (0.60, 0.15), (0.85, 0.85))


def _nearest_passable(raster, frac_rc, max_radius=400):
    """Snap a fractional (row, col) anchor to the nearest passable cell."""
    rows, cols = raster.shape
    r0 = int(frac_rc[0] * rows)
    c0 = int(frac_rc[1] * cols)
    sentinel = np.iinfo(np.uint16).max
    for rad in range(0, max_radius, 10):
        rs = slice(max(r0 - rad - 10, 0), min(r0 + rad + 11, rows))
        cs = slice(max(c0 - rad - 10, 0), min(c0 + rad + 11, cols))
        window = raster[rs, cs]
        ok = np.argwhere(window != sentinel)
        if ok.size:
            d2 = ((ok[:, 0] + rs.start - r0) ** 2
                  + (ok[:, 1] + cs.start - c0) ** 2)
            r, c = ok[int(d2.argmin())]
            return int(r + rs.start), int(c + cs.start)
    raise ValueError(f"no passable cell within {max_radius} of {frac_rc}")


def case_real_raster(path=REAL_RASTER):
    import rasterio
    with rasterio.open(path) as ds:
        raster = ds.read(1).astype(np.uint16)
    rows, cols = raster.shape

    sources = [_nearest_passable(raster, a) for a in REAL_SOURCES]
    targets = [_nearest_passable(raster, a) for a in REAL_TARGETS]

    pairs, skipped = [], 0
    for sr, sc in sources:
        s_idx = sr * cols + sc
        f = solve_fim(raster, s_idx)
        d1 = solve_discrete(raster, R1, s_idx)
        d2 = solve_discrete(raster, R2, s_idx)
        for tr, tc in targets:
            tf, t1, t2 = (float(f[tr, tc]), float(d1[tr, tc]),
                          float(d2[tr, tc]))
            if not all(v < FINITE_LIMIT and np.isfinite(v)
                       for v in (tf, t1, t2)):
                skipped += 1     # target unreachable from this source
                continue
            pairs.append(dict(
                source=[sr, sc], target=[tr, tc],
                fim=tf, r1=t1, r2=t2,
                fim_vs_r2_pct=(t2 - tf) / t2 * 100,
                fim_vs_r1_pct=(t1 - tf) / t1 * 100,
                r2_vs_r1_pct=(t1 - t2) / t1 * 100))
    if not pairs:
        raise RuntimeError("no reachable source/target pair on the "
                           "real raster — anchors need adjusting")
    return dict(
        raster=str(path.name), shape=[rows, cols],
        n_pairs=len(pairs), n_skipped_unreachable=skipped,
        fim_vs_r2_mean_pct=statistics.mean(
            p["fim_vs_r2_pct"] for p in pairs),
        fim_vs_r2_max_pct=max(p["fim_vs_r2_pct"] for p in pairs),
        fim_vs_r2_min_pct=min(p["fim_vs_r2_pct"] for p in pairs),
        fim_vs_r1_mean_pct=statistics.mean(
            p["fim_vs_r1_pct"] for p in pairs),
        r2_vs_r1_mean_pct=statistics.mean(
            p["r2_vs_r1_pct"] for p in pairs),
        pairs=pairs,
    )


# ----------------------------------------------------------------------
# Case 5a: corner error near thin barriers
# ----------------------------------------------------------------------

def case_thin_barrier(n=200, v=10):
    """Wall of thickness w with its tip mid-grid: T at the mirror point
    vs the analytic geodesic around the wall tip."""
    rows = []
    for w in (1, 2, 3):
        raster = np.full((n, n), v, dtype=np.uint16)
        tip_row = 150
        raster[0:tip_row, 100:100 + w] = np.iinfo(np.uint16).max
        sr, sc = 75, 50
        tr, tc = 75, 150
        t = solve_fim(raster, sr * n + sc)
        # analytic: source -> wall tip corners -> target. The wall
        # occupies [-0.5, tip_row - 0.5] x [99.5, 99.5 + w]; the path
        # bends around (tip_row - 0.5, 99.5) and (tip_row - 0.5,
        # 99.5 + w).
        a = (tip_row - 0.5, 99.5)
        b = (tip_row - 0.5, 99.5 + w)
        exact = v * (np.hypot(a[0] - sr, a[1] - sc)
                     + (b[1] - a[1])
                     + np.hypot(tr - b[0], tc - b[1]))
        got = float(t[tr, tc])
        rows.append(dict(thickness=w, cost=got, analytic=float(exact),
                         signed_error_pct=(got - exact) / exact * 100))
        # tunneling check: a full wall (touching the border) must block
        raster2 = np.full((n, n), v, dtype=np.uint16)
        raster2[:, 100:100 + w] = np.iinfo(np.uint16).max
        t2 = solve_fim(raster2, sr * n + sc)
        rows[-1]["tunneling"] = bool(t2[tr, tc] < FINITE_LIMIT)
    return rows


# ----------------------------------------------------------------------
# Case 5b: shock smearing
# ----------------------------------------------------------------------

def case_shock_smearing(n=201, v=10, sep=120):
    """Two sources: T cross-section across the equidistant shock.

    Analytically T has a kink (gradient jumps from +v to -v). Measure
    the width (in cells) over which the discrete gradient transitions
    between 90% of its asymptotic values.
    """
    sr = n // 2
    s1 = (sr, (n - sep) // 2)
    s2 = (sr, (n + sep) // 2)
    raster = np.full((n, n), v, dtype=np.uint16)
    t = solve_fim(raster, np.array([s1[0] * n + s1[1],
                                    s2[0] * n + s2[1]]))
    # cross-section along the row through both sources
    probe_row = sr + 60    # off the source row: shock still at mid-col
    profile = t[probe_row, :].astype(np.float64)
    grad = np.gradient(profile)
    mid = n // 2
    window = grad[mid - 15: mid + 16]
    g_left = np.median(grad[s1[1] + 5: mid - 10])
    g_right = np.median(grad[mid + 10: s2[1] - 5])
    inside = np.where(np.abs(window) < 0.9 * min(abs(g_left),
                                                 abs(g_right)))[0]
    width = int(inside.size)
    return dict(kink_transition_width_cells=width,
                grad_left=float(g_left), grad_right=float(g_right),
                profile_around_shock=[float(x) for x in
                                      profile[mid - 8: mid + 9]])


# ----------------------------------------------------------------------
# Case 5c: iteration blowup vs cost contrast
# ----------------------------------------------------------------------

def case_contrast(n=500, contrasts=(10, 100, 1000, 10000), reps=3):
    rows = []
    rng = np.random.default_rng(5)
    for contrast in contrasts:
        raster = np.exp(rng.uniform(0, np.log(contrast), (n, n)))
        raster = np.clip(raster, 1, 65000).astype(np.uint16)
        times, iters = [], 0
        for _ in range(reps):
            t0 = time.perf_counter()
            _, iters = eikonal_raster_gpu(raster, 0,
                                          return_iterations=True)
            times.append(time.perf_counter() - t0)
        rows.append(dict(contrast=contrast,
                         outer_passes=int(iters),
                         wall_ms=statistics.median(times) * 1e3))
    return rows


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced seeds/sizes for a smoke run")
    ap.add_argument("--only", default=None,
                    help="run a single case (uniform, snell, smooth, "
                         "terrain, barrier, shock, contrast, real)")
    args = ap.parse_args()

    seeds = 5 if args.quick else 20
    results = {}

    def wanted(name):
        return args.only is None or args.only == name

    if wanted("uniform"):
        print("=== 1. uniform direction sweep (elongation vs c*r) ===")
        results["uniform_directions"] = case_uniform_directions()
        for r in results["uniform_directions"]:
            print(f"  {r['solver']:<12} max "
                  f"+{r['max_elongation_pct']:.3f}% "
                  f"(at {r['worst_angle_deg']:.1f} deg)  "
                  f"mean +{r['mean_elongation_pct']:.3f}%")

    if wanted("snell"):
        print("=== 2. Snell refraction ===")
        results["snell"] = case_snell()
        for r in results["snell"]:
            print(f"  {r['solver']:<12} {r['cost']:.1f} vs analytic "
                  f"{r['analytic']:.1f}  ({r['error_pct']:+.3f}%)")

    if wanted("smooth"):
        print("=== 3. smooth random fields ===")
        results["smooth_random"] = case_smooth_random(seeds=seeds)
        for key, r in results["smooth_random"].items():
            print(f"  {key}: FIM below R2 by "
                  f"{r['fim_vs_r2_mean_pct']:.2f}% mean "
                  f"[{r['fim_vs_r2_min_pct']:.2f}, "
                  f"{r['fim_vs_r2_max_pct']:.2f}]%, below R1 by "
                  f"{r['fim_vs_r1_mean_pct']:.2f}% "
                  f"({r['n_seeds']} seeds, "
                  f"{r['ordering_violations']} ordering violations)")

    if wanted("terrain"):
        print("=== 4. terrain-like composite ===")
        results["terrain"] = case_terrain(seeds=max(3, seeds // 4))
        r = results["terrain"]
        print(f"  FIM below R2 by {r['fim_vs_r2_mean_pct']:.2f}% mean "
              f"[{r['fim_vs_r2_min_pct']:.2f}, "
              f"{r['fim_vs_r2_max_pct']:.2f}]"
              f"%, below R1 by {r['fim_vs_r1_mean_pct']:.2f}% "
              f"({r['n_seeds']} seeds)")

    if wanted("barrier"):
        print("=== 5a. thin-barrier corner error ===")
        results["thin_barrier"] = case_thin_barrier()
        for r in results["thin_barrier"]:
            print(f"  w={r['thickness']}: {r['signed_error_pct']:+.3f}% "
                  f"vs analytic geodesic, tunneling={r['tunneling']}")

    if wanted("shock"):
        print("=== 5b. shock smearing ===")
        results["shock"] = case_shock_smearing()
        width = results["shock"]["kink_transition_width_cells"]
        print(f"  kink transition width ~{width} cells")

    if wanted("contrast"):
        print("=== 5c. contrast blowup ===")
        results["contrast"] = case_contrast()
        for r in results["contrast"]:
            print(f"  contrast {r['contrast']:>6}: "
                  f"{r['outer_passes']:4d} "
                  f"passes, {r['wall_ms']:7.1f} ms")

    if wanted("real"):
        print("=== 6. real cost raster ===")
        if REAL_RASTER.exists():
            results["real_raster"] = case_real_raster()
            r = results["real_raster"]
            print(f"  {r['raster']} {r['shape'][0]}x{r['shape'][1]}: "
                  f"FIM below R2 by {r['fim_vs_r2_mean_pct']:.2f}% mean "
                  f"[{r['fim_vs_r2_min_pct']:.2f}, "
                  f"{r['fim_vs_r2_max_pct']:.2f}]%, below R1 by "
                  f"{r['fim_vs_r1_mean_pct']:.2f}% "
                  f"({r['n_pairs']} pairs, "
                  f"{r['n_skipped_unreachable']} unreachable skipped)")
        else:
            print(f"  SKIPPED: {REAL_RASTER} not found")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"eikonal_accuracy_{stamp}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()

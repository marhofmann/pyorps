"""Shared case definitions for the ArcGIS Distance Accumulation
comparison harness (eikonal plan section 6.4).

Each case provides: the cost raster (float32; np.nan marks barriers →
exported as NoData so ArcGIS treats them as such), the source cell, and
an analytic reference where a closed form exists (the referee is the
closed form, not any of the solvers). The plan-6.2 cases without a
closed form (smooth_random, real_raster) carry ``reference="fim"``
instead: there the comparison is a cross-check ArcGIS vs our FIM field
— agreement within discretization error is the expected outcome, and a
systematic gap flags an algorithmic difference, not an error of either.

Unit convention: 1 m cell size, cost = cost per metre of traversal, so
accumulation fields are directly comparable between ArcGIS, FIM and the
analytic truth (see plan section 1).
"""

from pathlib import Path

import numpy as np

CRS = "EPSG:32632"
CELL = 1.0     # metres

REAL_RASTER = (Path(__file__).parents[2]
               / "examples" / "data" / "raster" / "small_raster.tiff")


def uniform_case(n=401, v=10.0):
    """Point source in a uniform field: T = v * r."""
    raster = np.full((n, n), v, dtype=np.float32)
    src = (n // 2, n // 2)

    def analytic(rr, cc):
        return v * np.hypot(rr - src[0], cc - src[1])

    return dict(name="uniform", raster=raster, source=src,
                analytic=analytic)


def radial_case(n=401, a=10.0, b=0.5):
    """Radially increasing slowness c = a + b*r: T = a r + b r^2 / 2
    (the straight radial ray is optimal for b > 0)."""
    src = (n // 2, n // 2)
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.hypot(rr - src[0], cc - src[1])
    raster = (a + b * r).astype(np.float32)

    def analytic(rr_, cc_):
        r_ = np.hypot(rr_ - src[0], cc_ - src[1])
        return a * r_ + b * r_ ** 2 / 2.0

    return dict(name="radial", raster=raster, source=src,
                analytic=analytic)


def snell_case(n=400, c1=5.0, c2=15.0):
    """Two homogeneous half-planes; interface between cell rows
    n/2 - 1 and n/2 (continuous y = n/2 - 0.5). Analytic T for points
    in the lower (slow) half via 1D minimization over the crossing."""
    raster = np.full((n, n), c1, dtype=np.float32)
    raster[n // 2:, :] = c2
    src = (int(0.15 * n), int(0.2 * n))
    iface = n // 2 - 0.5

    def analytic(rr, cc):
        """Vectorized reference; valid for scalar or 1D arrays of probe
        points (chunked minimization over the crossing coordinate)."""
        rr = np.atleast_1d(np.asarray(rr, dtype=np.float64))
        cc = np.atleast_1d(np.asarray(cc, dtype=np.float64))
        x = np.linspace(-100.0, n + 100.0, 8001)
        out = np.empty(rr.shape, dtype=np.float64)
        upper = rr <= iface
        # same-side points: straight line unless refraction through the
        # fast medium never helps (c1 < c2 -> it doesn't for upper pts)
        out[upper] = c1 * np.hypot(rr[upper] - src[0], cc[upper] - src[1])
        low = ~upper
        if low.any():
            leg1 = c1 * np.hypot(iface - src[0], x - src[1])   # (nx,)
            leg2 = c2 * np.hypot(rr[low][:, None] - iface,
                                 cc[low][:, None] - x[None, :])
            out[low] = (leg1[None, :] + leg2).min(axis=1)
        return out if out.size > 1 else float(out[0])

    return dict(name="snell", raster=raster, source=src,
                analytic=analytic)


def barrier_case(n=200, v=10.0, tip_row=150, wall_col=100):
    """Wall of 1-cell thickness from the top edge down to tip_row; the
    optimal continuous path bends around the wall tip. Analytic values
    only at probe points with a hand-derived corner geodesic."""
    raster = np.full((n, n), v, dtype=np.float32)
    raster[0:tip_row, wall_col] = np.nan     # NoData -> ArcGIS barrier
    src = (75, 50)
    a = (tip_row - 0.5, wall_col - 0.5)
    b = (tip_row - 0.5, wall_col + 0.5)

    probes = {}
    for name, (pr, pc) in {"mirror": (75, 150),
                           "behind_high": (40, 160),
                           "behind_low": (120, 140)}.items():
        direct_visible = pr >= tip_row     # below the tip: straight line
        if direct_visible:
            cost = v * np.hypot(pr - src[0], pc - src[1])
        else:
            cost = v * (np.hypot(a[0] - src[0], a[1] - src[1])
                        + (b[1] - a[1])
                        + np.hypot(pr - b[0], pc - b[1]))
        probes[name] = dict(row=pr, col=pc, analytic=float(cost))

    return dict(name="barrier", raster=raster, source=src,
                analytic=None, probes=probes)


def _fft_blur(field, sigma):
    """Deterministic separable Gaussian blur (exact transfer function
    exp(-2 pi^2 sigma^2 f^2); no scipy dependency)."""
    n_r, n_c = field.shape
    kr = np.exp(-2.0 * (np.pi * np.fft.fftfreq(n_r) * sigma) ** 2)
    kc = np.exp(-2.0 * (np.pi * np.fft.fftfreq(n_c) * sigma) ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.outer(kr, kc)))


def smooth_random_case(n=400, sigma=16.0, seed=20260806,
                       lo=1.0, hi=200.0):
    """Smooth random field (plan 6.2, sigma=16 correlation length).

    No closed form exists — ``reference="fim"``: the ArcGIS field is
    cross-compared against ours. Fully deterministic (fixed seed, FFT
    blur), so the exported GeoTIFF is reproducible bit-for-bit.
    """
    rng = np.random.default_rng(seed)
    base = _fft_blur(rng.normal(0.0, 1.0, (n, n)), sigma)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    raster = (lo + base * (hi - lo)).astype(np.float32)
    return dict(name="smooth_random", raster=raster,
                source=(n // 8, n // 8), analytic=None, reference="fim")


def real_raster_case():
    """Real land-use cost raster from the repository test data
    (plan 6.2): 10 cost classes 92-438, 51% forbidden (65535 -> NaN).

    ``reference="fim"`` (no closed form). The exported grid is
    idealized to exact 1 m square cells — the raw raster's cells are
    0.99999 x 1.00017 m, so re-declaring the transform (data unchanged)
    makes ArcGIS's map-unit accumulation exactly comparable to the
    cell-unit T field; the 1.7e-4 relative distortion is far below
    every effect measured here.
    """
    import rasterio
    if not REAL_RASTER.exists():
        raise FileNotFoundError(REAL_RASTER)
    with rasterio.open(REAL_RASTER) as ds:
        data = ds.read(1)
        origin = (float(ds.transform.c), float(ds.transform.f))
    raster = data.astype(np.float32)
    raster[data == np.iinfo(np.uint16).max] = np.nan

    # deterministic source: nearest passable cell to the raster center
    rows, cols = raster.shape
    r0, c0 = rows // 2, cols // 2
    for rad in range(10, 500, 10):
        window = raster[r0 - rad:r0 + rad + 1, c0 - rad:c0 + rad + 1]
        ok = np.argwhere(np.isfinite(window))
        if ok.size:
            d2 = ((ok[:, 0] - rad) ** 2 + (ok[:, 1] - rad) ** 2)
            r, c = ok[int(d2.argmin())]
            src = (int(r + r0 - rad), int(c + c0 - rad))
            break
    else:
        raise ValueError("no passable cell near the raster center")

    return dict(name="real_raster", raster=raster, source=src,
                analytic=None, reference="fim", crs="EPSG:25832",
                origin=origin)


ALL_CASES = [uniform_case, radial_case, snell_case, barrier_case,
             smooth_random_case, real_raster_case]

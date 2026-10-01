"""Differential fuzz: raster_fim grade-limit loop vs the Cython kernel.

300 trials, 40x40, r1 steps on BOTH sides, identical GradientLUTs,
limits {8, 12, 20, 30} %, four terrain generators. Reports the
false-negative rate: the loop reports infeasible / fails where the
discrete kernel does find a route.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from pyorps.core.exceptions import NoPathFoundError
from pyorps.core.objective import Objective, GradientOptions
from pyorps.graph.api.cython_api import CythonAPI
from pyorps.graph.api.raster_fim_api import RasterFIMAPI
from pyorps.utils.neighborhood import get_neighborhood_steps

N = 40
CELL = 10.0
STEPS = get_neighborhood_steps("r1", directed=True)
LIMITS = (8.0, 12.0, 20.0, 30.0)


def terrain(kind, rng, n=N):
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    if kind == "hills":
        z = np.zeros((n, n))
        for _ in range(rng.integers(2, 6)):
            r0, c0 = rng.integers(0, n, 2)
            amp = rng.normal() * 120.0
            sig = rng.uniform(3.0, 10.0)
            z += amp * np.exp(-(((rr - r0) ** 2 + (cc - c0) ** 2)
                                / (2 * sig ** 2)))
    elif kind == "noise":
        z = rng.normal(scale=rng.uniform(1.0, 6.0), size=(n, n))
    elif kind == "fault":
        a = np.deg2rad(rng.uniform(0, 360))
        g = rng.uniform(0.02, 0.25)
        z = g * CELL * (np.cos(a) * rr + np.sin(a) * cc)
        # a fault line: a step discontinuity across a random half-plane
        b = np.deg2rad(rng.uniform(0, 360))
        side = (np.cos(b) * (rr - n / 2) + np.sin(b) * (cc - n / 2)) > 0
        z = z + side * rng.uniform(5.0, 40.0)
    elif kind == "walk":
        z = (rng.normal(size=(n, n)).cumsum(0).cumsum(1)
             * rng.uniform(0.05, 0.5))
    else:
        raise ValueError(kind)
    return z.astype(np.float32)


def run(n_trials=300, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    kinds = ("hills", "noise", "fault", "walk")
    raster = np.ones((N, N), dtype=np.uint16)

    cy_found = 0
    fn = 0
    fim_found = 0
    fn_by_kind = {}
    fn_by_limit = {}
    tot_by_kind = {}
    tot_by_limit = {}
    reasons = {}
    for i in range(n_trials):
        kind = kinds[i % len(kinds)]
        limit = LIMITS[(i // len(kinds)) % len(LIMITS)]
        dem = terrain(kind, rng)
        s = int(rng.integers(0, N * N))
        t = int(rng.integers(0, N * N))
        while t == s:
            t = int(rng.integers(0, N * N))
        obj = Objective({"cost": 1.0},
                        GradientOptions(max_gradient_pct=limit))
        luts = obj.build_gradient_luts(STEPS, CELL)

        cy = CythonAPI(raster, STEPS, dem_data=dem, gradient_luts=luts)
        try:
            cy_route = list(cy.shortest_path(s, t))
        except Exception:
            cy_route = []
        if not cy_route:
            continue
        cy_found += 1
        tot_by_kind[kind] = tot_by_kind.get(kind, 0) + 1
        tot_by_limit[limit] = tot_by_limit.get(limit, 0) + 1

        api = RasterFIMAPI(raster, STEPS, dem_data=dem, cell_size=CELL,
                           gradient_luts=luts)
        try:
            route = api.shortest_path(s, t)
            ok = bool(len(route)) and api.grade_violations(route).size == 0
            why = "" if ok else "empty/invalid"
        except NoPathFoundError as e:
            ok, why = False, "NoPathFound"
            if "DIFFERENT components" in str(e):
                why = "certificate-infeasible"
            elif "ISOLATED" in str(e):
                why = "certificate-isolated"
            elif "reachability was lost" in str(e):
                why = "reachability-lost"
            elif "only cells violating" in str(e):
                why = "terminals-steep"
            elif "stopped growing" in str(e):
                why = "mask-stalled"
        except RuntimeError:
            ok, why = False, "cap"
        except Exception as e:                       # pragma: no cover
            ok, why = False, f"other:{type(e).__name__}"
        if ok:
            fim_found += 1
        else:
            fn += 1
            fn_by_kind[kind] = fn_by_kind.get(kind, 0) + 1
            fn_by_limit[limit] = fn_by_limit.get(limit, 0) + 1
            reasons[why] = reasons.get(why, 0) + 1
            if verbose:
                print(f"  FN trial {i} kind={kind} limit={limit} "
                      f"s={s} t={t} why={why}")

    print(f"trials={n_trials}  cython found a route in {cy_found}")
    print(f"fim found+verified: {fim_found}")
    print(f"FALSE NEGATIVES: {fn}/{cy_found} = "
          f"{100.0 * fn / max(cy_found, 1):.1f} %")
    print("by kind:", {k: f"{fn_by_kind.get(k, 0)}/{tot_by_kind.get(k, 0)}"
                       for k in kinds})
    print("by limit:", {k: f"{fn_by_limit.get(k, 0)}/{tot_by_limit.get(k, 0)}"
                        for k in LIMITS})
    print("reasons:", reasons)
    return fn, cy_found


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(n, seed, verbose="-v" in sys.argv)

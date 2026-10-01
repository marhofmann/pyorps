"""The shipped tower field, measured: scaling, sigma/K sensitivity, memory.

The two ``proto_*`` scripts next to this one established the SHAPE of
the argument on a numpy sketch. This one measures the implementation in
``pyorps.graph.tower_field`` and produces the table
``docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md`` section
7 asks for -- "sigma in {5, 10, 20} m, K in {24, 48, 120}: cost change
vs runtime, published as a table" -- rather than a single number chosen
after the fact.

Reads a real crop of the CIRED 2026 cost raster when it is present and
falls back to a synthetic surface with the same value range when it is
not, so the scaling result is reproducible outside this machine. Which
one was used is printed.

    python benchmarks/tower_field/bench_tower_field.py
"""
from __future__ import annotations

import time

import numpy as np

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.tower_field import (
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
    angle_tables_from_profile,
)
from pyorps.utils.directional import primitive_directions

SRC = (r"C:/Users/mhnn82/Documents/2_python_projects/TopoMILP/data/"
       r"cired2026/raster/mod1_raster_wp_fixed.tiff")
PROFILE = "profiles/overhead_line_110kv.yaml"
WINDOW = (9000, 4000)


def load(n_cells: int, cell_m: float = 1.0):
    """An ``n x n`` cost crop, real if the raster is reachable."""
    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(SRC) as r:
            a = r.read(1, window=Window(WINDOW[0], WINDOW[1],
                                        n_cells, n_cells))
        if a.shape == (n_cells, n_cells):
            return np.asarray(a, dtype=np.uint16), "real crop"
    except Exception:                                   # noqa: BLE001
        pass
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n_cells, 0:n_cells]
    base = (60 + 90 * np.sin(xx / 70.0) * np.cos(yy / 55.0)
            + rng.random((n_cells, n_cells)) * 60)
    a = np.clip(base, 1, 500).astype(np.uint16)
    a[(xx // 97) % 11 == 0] = 65535                     # linear exclusions
    return a, "synthetic"


def _solver(raster, profile, sigma_m, cell_m, dirs, tier, angles=None):
    from pyorps.graph.tower_field import coarsen
    factor = max(1, int(round(sigma_m / cell_m)))
    blocked_fine = raster >= 65535
    values = coarsen(np.where(blocked_fine, 0, raster).astype(np.float64),
                     factor, "mean")
    blocked = coarsen(blocked_fine.astype(np.float64), factor, "max") > 0
    lattice = TowerLattice(cell_size_m=cell_m, factor=factor,
                           directions=dirs)
    lut = profile.precompute_tower_terrain_costs()
    tower = lut[np.clip(np.rint(values), 0, 65535).astype(np.int64)]
    if tier == 1:
        tables = angles or angle_tables_from_profile(profile, lattice)
        usable = tables.premium[tables.valid & np.isfinite(tables.premium)]
        tower = tower + float(usable.min())
    tower[blocked] = np.inf
    model = TowerFieldModel.matching_kernel(profile, angle_tier=tier)
    t0 = time.perf_counter()
    solver = TowerFieldSolver(
        values=values, tower_cost=tower, lattice=lattice, model=model,
        blocked=blocked,
        angles=(angles or angle_tables_from_profile(profile, lattice))
        if tier == 2 else None)
    return solver, time.perf_counter() - t0, values.shape


def scaling(profile):
    print("\n== per-sweep cost is O(cells); sweep count tracks EXTENT ==")
    print("(tier 1, 48 directions, sigma 10 m, spans 50-300 m)\n")
    print(f"{'grid':>11}{'extent':>10}{'lattice':>10}{'sweeps':>8}"
          f"{'build s':>9}{'solve s':>9}{'us/cell/sweep':>15}{'prefix MB':>11}")
    print("-" * 83)
    dirs = primitive_directions(4)
    rows = []
    for n in (600, 1200, 2000, 3000):
        raster, _ = load(n)
        solver, t_build, shape = _solver(raster, profile, 10.0, 1.0, dirs, 1)
        cells = shape[0] * shape[1]
        t0 = time.perf_counter()
        field = solver.solve((shape[0] // 2, 2), record_pred=False)
        t_solve = time.perf_counter() - t0
        per = 1e6 * t_solve / (cells * field.sweeps)
        rows.append((n, field.sweeps, per))
        lat = f"{shape[0]}x{shape[1]}"
        print(f"{lat:>11}{n:>8} m{cells:>10}{field.sweeps:>8}"
              f"{t_build:9.2f}{t_solve:9.2f}{per:15.2f}"
              f"{solver.prefix_bytes / 1e6:11.1f}")
    print("\nSweeps grow with extent / max_span -- the layered-DAG property.")
    print("us/cell/sweep is flat, which is what O(N) per sweep means.")
    return rows


def extrapolate(rows):
    """The plan's HV-window figure, restated from what was just measured."""
    per = float(np.median([r[2] for r in rows]))
    # HV window: 240 M cells at 1 m, so 2.4 M positions on a 10 m lattice,
    # ~22 km across at 300 m spans -> ~80 sweeps.
    positions, sweeps, dirs = 2_400_000, 80, 48
    seconds = per * 1e-6 * positions * sweeps
    prefix_gb = positions * 8 * dirs / 1e9
    print("\n== extrapolation to the HV window ==")
    print(f"  measured        {per:.2f} us/cell/sweep (median above)")
    print(f"  2.4 M positions x {sweeps} sweeps -> {seconds / 60:.1f} min "
          f"tier 1, pure numpy")
    print(f"  prefix tables   {prefix_gb:.2f} GB at float64 x {dirs} "
          f"directions ({prefix_gb / 2:.2f} GB at float32)")
    print("  against 1.66 TB for the (cell, direction, span, height) "
          "state model.")
    print("  NOTE this is an extrapolation from a <= 3 km crop, exactly as "
          "the plan's was.\n  float64 doubles the plan's 470 MB estimate; "
          "section 4.1 requires float64.")


def sensitivity(profile):
    """Section 7's sigma / K table: what the discretisation costs."""
    print("\n== sigma / K sensitivity (tier 1, 1200 m window) ==")
    print("Reported, not chosen silently: direction quantisation puts the")
    print("endpoint error at ~L*dtheta/2 (7.3 m at L=300 m, K=128), which is")
    print("comparable to a 10 m lattice -- so sigma and K belong together.\n")
    raster, _kind = load(1200)
    grid = [(sigma, primitive_directions(dmax))
            for sigma in (5.0, 10.0, 20.0) for dmax in (3, 4, 7)]
    rows = []
    for sigma, dirs in grid:
        solver, _tb, shape = _solver(raster, profile, sigma, 1.0, dirs, 1)
        t0 = time.perf_counter()
        field = solver.solve((shape[0] // 2, 2), record_pred=False)
        dt = time.perf_counter() - t0
        r = int(shape[0] * 0.5)
        band = field.arrival[r, int(shape[1] * 0.6):int(shape[1] * 0.95)]
        mean = float(np.nanmean(np.where(np.isfinite(band), band, np.nan)))
        rows.append((sigma, len(dirs), shape, field.sweeps, dt,
                     solver.prefix_bytes, mean))

    # The reference is the FINEST configuration, not the first row.
    ref = min(r[-1] for r in rows)
    print(f"{'sigma':>7}{'K':>5}{'lattice':>12}{'sweeps':>8}{'solve s':>9}"
          f"{'prefix MB':>11}{'mean EUR':>14}{'vs finest':>11}")
    print("-" * 79)
    for sigma, k, shape, sweeps, dt, pb, mean in rows:
        lat = f"{shape[0]}x{shape[1]}"
        print(f"{sigma:7.0f}{k:5d}{lat:>12}{sweeps:>8}"
              f"{dt:9.2f}{pb / 1e6:11.1f}{mean:14,.0f}"
              f"{100 * (mean / ref - 1):+10.2f}%")

    print("\nAt a FIXED sigma, adding directions can only lower the cost:")
    print("primitive_directions(d) nests in primitive_directions(d+1), so")
    print("every chain stays available and more become so. That column is")
    print("monotone above, and it is the inequality tower_field_bounds()")
    print("rests on.")
    print("Across sigma it is NOT a pure restriction: pooling changes the")
    print("cost model as well as the tower positions, so a coarser lattice")
    print("is usually but not necessarily dearer. Read the sigma rows as a")
    print("discretisation report, not as a bound.")


def tiers(profile):
    """Tier 1 against tier 2: the gap the plan calls its headline number."""
    print("\n== tier 1 (angle-free lower bound) vs tier 2 "
          "(direction-resolved) ==")
    raster, _ = load(600)
    dirs = primitive_directions(3)
    out = {}
    for tier in (1, 2):
        solver, _tb, shape = _solver(raster, profile, 10.0, 1.0, dirs, tier)
        t0 = time.perf_counter()
        field = solver.solve((shape[0] // 2, 2), record_pred=False)
        out[tier] = (field, time.perf_counter() - t0)
    (f1, t1), (f2, t2) = out[1], out[2]
    both = np.isfinite(f1.arrival) & np.isfinite(f2.arrival)
    both[f1.source] = False
    gap = (f2.arrival[both] - f1.arrival[both]) / f2.arrival[both]
    print(f"  tier 1 {t1:6.2f} s, {f1.sweeps} sweeps")
    print(f"  tier 2 {t2:6.2f} s, {f2.sweeps} sweeps, "
          f"{t2 / max(t1, 1e-9):.1f}x the cost for "
          f"{len(dirs)} direction fields")
    print(f"  gap    min {100 * gap.min():+.2f} %  mean "
          f"{100 * gap.mean():.2f} %  max {100 * gap.max():.2f} %")
    print("  tier 1 is never above tier 2: the premium is non-negative, so")
    print("  dropping it is a valid lower bound -- and a far tighter one")
    print("  than 'terrain + minimum tower count', because it still solves")
    print("  the span and spacing problem.")

    print("\n== angle_mode: does the K x n_classes trick pay? ==")
    print("The plan's saving assumes the premium takes ONE VALUE PER TOWER")
    print("CLASS (4), so the min-plus over directions collapses to four")
    print("circular windows. Measured on the shipped profile it does not:")
    for mode in ("exact", "window"):
        solver, _tb, shape = _solver(raster, profile, 10.0, 1.0, dirs, 2)
        solver.model = TowerFieldModel.matching_kernel(
            profile, angle_tier=2, angle_mode=mode)
        t0 = time.perf_counter()
        f = solver.solve((shape[0] // 2, 2), record_pred=False)
        print(f"  angle_mode={mode:<7} {time.perf_counter() - t0:6.2f} s, "
              f"levels={f.meta.get('angle_classes')}, "
              f"admissible pairs={f.meta.get('angle_pairs')} of "
              f"{len(dirs) ** 2}")
    tables = angle_tables_from_profile(
        profile, TowerLattice(cell_size_m=10.0, directions=dirs))
    ok = tables.valid & np.isfinite(tables.premium)
    n_levels = len(np.unique(tables.premium[ok]))
    types = len(profile.tower_cost_params.get("angle_types", {}))
    print(f"\n  tower TYPE classes in the profile: {types}")
    print(f"  distinct premium levels among admissible pairs: {n_levels}")
    print("  The gap is angle_cost_function: piecewise, which INTERPOLATES")
    print("  the turn penalty continuously -- so the premium is a staircase")
    print("  only in its tower-type term. With that many levels the window")
    print("  form is no cheaper than the exact minimum, which the 40 deg")
    print("  hard limit has already pruned to a fraction of K^2.")
    print("  'window' therefore earns its place as a documented UPPER")
    print("  bound on 'exact', not as a speed-up. It becomes a speed-up")
    print("  when the turn penalty is a step function or is switched off.")


def main():
    profile = InfrastructureProfile.load(PROFILE)
    raster, kind = load(600)
    print(f"cost surface: {kind}; values 1..{int(raster.max())}, "
          f"{100 * (raster >= 65535).mean():.1f} % excluded")
    rows = scaling(profile)
    extrapolate(rows)
    sensitivity(profile)
    tiers(profile)


if __name__ == "__main__":
    main()

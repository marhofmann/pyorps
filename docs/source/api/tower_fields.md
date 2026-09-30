---
title: "Tower Fields"
summary: "Overhead-line siting with one scalar field per tower position."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.graph.tower_field"
api:
  - pyorps.TowerField
  - pyorps.TowerFieldModel
  - pyorps.TowerFieldSolver
  - pyorps.TowerLattice
  - pyorps.AngleTables
  - pyorps.ClearanceModel
  - pyorps.solve_tower_field
  - pyorps.tower_field_from_raster
  - pyorps.tower_field_bounds
  - pyorps.check_bounds
  - pyorps.angle_tables_from_profile
  - pyorps.clearance_from_profile
---
# 🗼 Tower Fields: overhead siting without a state explosion

`ConstrainedPathFinder.find_route` answers **one** question: what is the best
overhead line from A to B, with towers placed and turn angles respected. A
siting study asks it for hundreds of thousands of candidate positions.

The obvious way to precompute that — settle a *field* of the constrained
search — does not work, and for a while the measurement said so: the extended
state `(cell, incoming direction, span bin, tower height)` is 288 states per
cell at 24 bytes each, which is **1.66 TB** dense on a 240 M-cell HV window.

That is a correct measurement of the wrong model. **An overhead line is not a
raster walk.** The conductor does not follow cells; it goes *straight* from
tower to tower. The state vector only exists to carry information along a path
that has no physical counterpart. Drop the walk and the state goes with it:

```
T(x) = tower(x) + min over y with L_min <= |x - y| <= L_max
                    of [ T(y) + span_cost(y, x) ]
```

One scalar per position.

```python
from pyorps import tower_field_from_raster
from pyorps.core.infrastructure_profile import InfrastructureProfile

profile = InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")
field = tower_field_from_raster(
    raster, cell_size_m=1.0, profile=profile,
    source_xy=(x0, y0), transform=transform, factor=10)   # 10 m tower lattice

costs = field.costs_to(candidate_sites)          # (n,), inf where impossible
towers = field.tower_sequence(row, col)          # the line, tower by tower
line = field.route_geometry(row, col)            # ... as a LineString
```

---

(tower-fields-why-it-is-cheap)=
## Why it is cheap

Three structural facts, and the implementation uses all three.

**1. Layered by tower count.** Every edge adds exactly one tower, so the graph
is a DAG in the tower-count dimension and Bellman–Ford converges in exactly
(max towers on an optimal line) sweeps. No priority queue, no divergence,
perfectly data-parallel. `field.sweeps` reports the count, and it grows with
the **extent** of the window, not with how finely you sample it — measured
600 m→4, 1200 m→7, 2000 m→9, 3000 m→14.

**2. Span cost is a prefix difference.** Along a fixed direction the terrain
integral from `y` to `x` is `P(x) − P(y)` for a cumulative sum along that
direction's rays. Terrain does not change between sweeps, so the tables are
built once (`solver.prefix_bytes`).

**3. The minimum over span length is a sliding-window minimum**, which van
Herk / Gil-Werman evaluates without the cost growing with the window width.

The primitives live in `pyorps.utils.directional` and are array-in, array-out
with no routing concepts: `ray_prefix_sum`, `ray_window_min`,
`ray_window_max`, `ray_run_length`.

### One primitive, three uses

| need | operator |
|---|---|
| terrain cost under a span | prefix **sum**, differenced |
| ground clearance under a span | windowed **max** on the DEM |
| forbidden crossing | run length of a 0/1 mask along the ray |

Which is why **height is not a state dimension**: the required tower top is a
function of the highest ground under the span, and a directional max filter
answers that in O(1).

---

(tower-fields-say-what-you-are-computing)=
## Say what you are computing

Three quantities in this codebase all answer to "the cost of that overhead
line", and they are not the same number:

- what the **kernel minimises** — terrain plus interior towers, with *neither*
  terminal tower charged, because the search stops the moment the target cell
  settles;
- what `ConstrainedPath` **reports** — the same line plus two terminal towers,
  ≥ 560 kEUR more on the shipped 110 kV profile, whose own footer says so;
- what `_terrain_eur` prices — category length × category value, with
  `IMPASSABLE_CELL_COST` dropped from the sum.

`TowerFieldModel` is that choice written down, and `.describe()` prints it:

```python
from pyorps import TowerFieldModel

TowerFieldModel.matching_kernel(profile)   # what find_route's kernel optimises
TowerFieldModel.as_reported(profile)       # what ConstrainedPath reports
```

The two differ by exactly `2 × terminal_tower_cost`, and a field carries its
model in `field.meta["model"]`.

### Verified against the router

`pyorps.graph.tower_field_oracle` runs `constrained_dijkstra_2d` itself and
compares its objective to the field at the same target. In the matched regime
— lattice factor 1, the kernel's own step set, `span_integral="pyorps"` — the
two agree to about **1e-8 relative**, which is the kernel's float32 step
factors and not a modelling difference; against an explicit float64 reference
the same solver agrees to 1e-12.

`score_tower_chain` is the definition made executable: hand it tower positions
and it prices them, naming every rule they break instead of raising.

---

(tower-fields-angle-premiums-in-tiers)=
## Angle premiums, in tiers

`angle_tier=1` drops the turn premium entirely. Because the premium is
non-negative, that is a **valid lower bound** — and a far tighter one than
"terrain plus minimum tower count", because it still solves the span and
spacing problem.

`angle_tier=2` carries the direction of the last span: K fields instead of
one, with the hard angle limit pruning most direction pairs (232 of 1024 at
K = 32 on the 110 kV profile). Measured gap between the tiers: mean 14.8 %.

Tier 3 of the design is `ConstrainedPathFinder.find_route` on the shortlist,
used as an oracle. It is not a setting of this solver.

---

(tower-fields-bounds-and-which-way-each-one-errs)=
## Bounds, and which way each one errs

```python
from pyorps import tower_field_bounds

lower, upper = tower_field_bounds(raster, cell_size_m=1.0, profile=profile,
                                  source_xy=(x0, y0), transform=transform)
```

| | how | direction |
|---|---|---|
| **lower** | tier 1, **min**-pooled terrain and tower ground cost, clearance and crossings ignored | never above the optimum |
| **upper** | tier 2, **max**-pooled, crossings enforced, clearance charged at both ends, towers on the σ lattice | attained by a buildable design |

Note the direction of the lattice restriction: confining towers to a σ grid
*shrinks* the feasible set, so it **raises** cost. It belongs on the upper
side and must never be used to claim a lower bound.

`check_bounds(lower, upper)` verifies `LB ≤ UB` on every cell and refuses a
cell that is reachable only in the upper bound. It is the cheapest strong
invariant available, and `tower_field_bounds` runs it by default.

**Precision.** Fields accumulate and export in float64. float32 at ~1.3e7 EUR
overstates half of all cells by up to 0.50 EUR, and the substation study's
winning margin was 28 EUR.

---

(tower-fields-exporting-for-a-solver)=
## Exporting for a solver

PYORPS does routing and siting. **Selecting among candidates is out of scope**
— a third-party MILP does that, under its own separation and topology
constraints, and it reads arrays.

```python
from pyorps.siting import (Footprint, screen_footprints, candidate_lattice,
                           export_tower_fields)

screen = screen_footprints(cost, blocked, footprint=Footprint(40.0, 20.0),
                           transform=transform, resolution_m=1.0)
cands = candidate_lattice(screen, stride_m=5.0)
export = export_tower_fields(cands, {"PCC0": (lower, upper)})
export.write_npz("candidates.npz")
```

Both ends of every cost are always present. A single number would force the
reader to guess whether it may prune with it.

Two things `pyorps.siting` will not let you leave implicit:

- `sample_field(rule=…)` — FLOOR (rasterio's `rowcol`) or ROUND. Candidates
  sit at pixel centres, where the two disagree by a whole cell.
- `lipschitz_stride_gap(stride_m, max_value)` — a stride lattice certifies the
  lattice, not the space. The gap is never zero.

---

(tower-fields-when-this-does-not-apply)=
## When this does not apply

- If spans are **not** straight in plan view for the voltage class — if the
  conductor's ground track must itself be routed — the reframing fails and the
  state-space model was right after all.
- If the angle premium depends on more than the deflection angle (span lengths
  either side, tension), the direction step becomes a full K×K product per
  cell.
- If tower ground cost depends on the **sequence** of towers (shared access
  roads, say) rather than the position, it stops being a node cost and the
  layered-DAG argument weakens.

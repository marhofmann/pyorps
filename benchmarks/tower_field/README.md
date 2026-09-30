# Tower-field prototypes and benchmarks

Evidence for `docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md`,
which is now **implemented** in `pyorps/graph/tower_field.py` and
`pyorps/utils/directional.py`.

An overhead line is a sequence of towers joined by *straight* spans, not a
raster walk. Dropping the walk drops the `(direction, span, height)` state
with it, and what is left is one scalar per position:

    T(x) = tower(x) + min over y, L_min <= |x-y| <= L_max, of
                        [ T(y) + span_cost(y, x) ]

| script | shows |
|---|---|
| `proto_windowed_min_equivalence.py` | the minimum over admissible span lengths can be replaced by a sliding-window minimum along each ray. Agrees with an explicit minimum to **0.000e+00**. |
| `proto_sweep_scaling.py` | with the directional prefix sums built once, a sweep is O(cells) and the number of sweeps scales with **extent**, not cell count — 480 m→4, 960 m→6, 1440 m→8, 2000 m→10. |
| `bench_tower_field.py` | the **shipped** solver: scaling, the σ/K sensitivity table section 7 asks for, tier 1 vs tier 2, and the HV-window extrapolation restated from what was just measured. |

The two `proto_*` scripts are the numpy sketch that established the shape of
the argument. `bench_tower_field.py` measures the library.

## What the library run confirmed, and what it corrected

Measured on a real crop of `mod1_raster_wp_fixed.tiff`, 110 kV profile,
spans 50–300 m. Wall-clock is on a loaded machine; the *shape* is the result.

**Confirmed.**

- Sweeps track extent, not cell count: 600 m→4, 1200 m→7, 2000 m→9, 3000 m→14.
  That is the layered-DAG property, and it is why there is no priority queue.
- Per-sweep cost is flat in cells (4.1–4.6 µs/cell/sweep at K = 48, tier 1,
  pure numpy), i.e. O(N) per sweep.
- Tier 1 never exceeds tier 2 (measured gap: mean 14.8 %, max 93 %), so
  dropping the angle premium is a valid lower bound.
- At a fixed σ, adding directions can only lower the cost —
  `primitive_directions(d)` nests in `primitive_directions(d+1)`.

**Corrected against the plan.**

- **The HV-window extrapolation is ~14 min, not ~5 min**, and the prefix
  tables are **0.92 GB, not 470 MB**, at 48 directions. Both follow from
  float64, which section 4.1 requires (float32 at 1.3e7 EUR overstates half
  of all cells by up to 0.50 EUR, against a 28 EUR winning margin). At
  float32 the tables would be 0.46 GB. Still an extrapolation from a ≤3 km
  crop, exactly as the plan's was.
- **`angle_mode="window"` is not a speed-up on the shipped profile.** The
  plan's `K × n_classes` saving assumes the premium takes one value per
  tower class (4). Measured, it takes **20** distinct values among the
  admissible pairs, because `angle_cost_function: piecewise` interpolates
  the turn penalty *continuously* — only the tower-type term is a staircase.
  With 20 levels the circular-window form costs more (0.84 s vs 0.43 s at
  K = 32) than the exact minimum, which the 40° hard limit has already
  pruned to 232 of 1024 pairs. It is kept as a stated **upper bound** on
  `"exact"`, and it becomes a speed-up where the turn penalty is a step
  function or switched off.
- **Comparing σ values is not a pure feasible-set restriction.** Pooling
  changes the cost model as well as the tower positions, so a coarser
  lattice is usually but not necessarily dearer. The σ rows are a
  discretisation report, not a bound.

## Phase 0, which used to be the caveat, is done

These prototypes were **not** validated against `ConstrainedPathFinder`.
The library is: `tests/test_graph/test_tower_field_oracle.py` runs
`constrained_dijkstra_2d` itself and compares its objective to the field at
the same target, in the matched regime (lattice factor 1, the kernel's own
step set, `span_integral="pyorps"`, terminals uncharged, no minimum on the
last span). The two agree to **~1e-8 relative**, which is the kernel's
float32 step factors (`_get_cost_factor_cython_f32`) and not a modelling
difference — against an explicit float64 reference the same solver agrees
to 1e-12.

Getting there found one real defect, which is what Phase 0 is for: the
field allowed a diagonal span to slip between two excluded cells, because
it tested only the lattice cells ON the ray. PYORPS' step model samples the
supercover, and the kernel refuses a step whose intermediates are excluded.
The field undercut the kernel by 25–380 EUR wherever a route passed a wall
until `_step_blocked` was added.

## Still open

- The full-scale run on the 2.4 M-position HV lattice. Everything above is
  ≤3 km.
- A GPU port. A sweep is a pure stencil with no queue and no divergence,
  which is the friendliest possible shape for the existing `raster_gpu`
  infrastructure.
- Clearance is charged from the ground maximum over the WIDEST admissible
  window and the sag of the LONGEST admissible span. That is conservative
  by construction; how conservative has not been quantified against the
  oracle (plan risk 5).

# Eikonal / block-FIM GPU backend — findings (increment 1, isotropic)

- **Date:** 2026-08-06
- **Plan:** `docs/superpowers/plans/2026-08-06-eikonal-fim-gpu-backend.md`
- **Hardware:** RTX PRO 500 Blackwell (14 SMs, 6 GB), CUDA 13.0
- **Data:** `benchmarks/results/eikonal_accuracy_20260806_211327.json`,
  `benchmarks/results/eikonal_perf_20260806_224507.json`
- **Scripts:** `benchmark_eikonal_accuracy.py`, `benchmark_eikonal_gpu.py`,
  `arcgis_comparison/`

## 1. Summary

The `raster_fim` backend solves the continuous eikonal equation
`|∇T| = c(x)` on the cost raster (first-order Godunov, Jeong–Whitaker
block-FIM, CuPy RawKernels) and traces routes by steepest descent. It
eliminates the graph backends' metrication (elongation) bias by
construction:

- **Worst-direction elongation on uniform cost: FIM +0.76% vs
  R1 +8.24% and R2 (pyorps default) +2.75%** — both discrete numbers
  land exactly on their theoretical bounds (8.2% / 2.79%), confirming
  the referee setup.
- Costs are **first-order convergent** (measured order 0.98–1.03) against
  analytic truth; the error is discretization error that shrinks with
  resolution, not a fixed directional bias.
- End-to-end (solve + trace) the backend is **4.6–23× faster than the
  Cython CPU champion** (increment 2: device tracer + vectorized
  rasterizer + targeted early exit), the margin growing with raster
  size. At 3000² it **beats the discrete GPU V5 kernel end-to-end on
  uniform/smooth rasters** (1.28×/1.06×) and near-matches it on random
  (0.82×) — the plan's V5-parity bonus, met. A FIM field still costs
  one solve for *all* targets of a source.
- **`order=2`** (increment 2): second-order refinement with observed
  convergence order 1.86–2.08 — worst-direction elongation drops to
  **+0.032%** (first order +0.76%, R2 +2.75%, R1 +8.24%) at ~2× the
  solve cost. See §10.3 for the scheme and its safeguards.
- All 74 unit tests green (`tests/test_utils/test_eikonal_gpu.py`,
  `tests/test_graph/test_raster_fim_api.py`, plus the float32-mode
  test in `tests/test_graph/test_float_precision.py`), including
  PathFinder end-to-end via `graph_api="raster_fim"`.
- The backend is a full **FLOAT_BACKENDS** member: with
  `weight_precision="float32"` the lossless combined feasibility
  surface flows through unchanged (the solver consumes float32
  natively; +inf/NaN = forbidden), and the **MetricStack/objective
  pipeline works unchanged** (plan §5.3) — objective weights steer the
  route and per-criterion honest metrics report normally (tested).

## 2. Cost calibration and semantics

Contract (plan §1, pinned by the Phase-0 gate at ≤ 1e-5 relative):
`c(x)` = raster value of the containing cell, `h = 1` cell; on a uniform
raster of value `v` an axis-aligned line of `L` cells gives `T = v·L` —
directly comparable to the discrete backends' `dist`, no rescaling.

- The API's authoritative routing metric is `T[target]`
  (`RasterFIMAPI.last_field_costs`). PathFinder's edge-based recompute
  over the rasterized cell path re-quantizes the continuous polyline and
  reports slightly higher values — the known dual-metric reporting issue
  applies here unchanged.
- Continuous polylines (float row/col, source→target) are kept on
  `RasterFIMAPI.last_polylines`.
- Exclusions mirror the discrete GPU line: uint16 sentinel 65535 with
  `ignore_max=True`; float rasters use the ≥1e30 / non-finite forbidden
  convention. Unreached cells hold `T = 1e30` (finite check `< 1e29`).

## 3. Metrication table (headline result)

Analytic truth as referee where it exists; measured 2026-08-06.

| Case | Dijkstra R1 (8-n) | Dijkstra R2 (16-n, default) | FIM (this work) |
|---|---|---|---|
| uniform, worst direction (401², sweep 0–45°) | **+8.24%** (at 22.5°) | **+2.75%** (at 13.5°) | **+0.76%** (at 45°) |
| uniform, mean over directions | +5.30% | +1.39% | +0.46% |
| Snell refraction (400², vs closed form) | +5.47% | +1.67% | **+0.38%** |
| smooth random σ=16 (300², 20 seeds, mean) | FIM below R1 by 2.07% | FIM below R2 by −0.19%* | — |
| smooth random σ=64 (20 seeds, mean) | FIM below R1 by 1.94% | FIM below R2 by 0.11% | — |
| terrain-like composite (5 seeds, mean) | FIM below R1 by 2.39% | FIM below R2 by 0.09% | — |
| **real cost raster** (small_raster.tiff, 3278×5364, 10 land-use classes, 51% forbidden; 12 long pairs) | FIM below R1 by 3.49% | FIM below R2 by 0.46% [−0.32, +1.14]* | — |

\* negative = FIM *above* R2 — see caveat §5.1.

Reading: on smooth/terrain rasters FIM and R2 price routes within a few
tenths of a percent of each other while R1 overpays ~2%; on adversarial
directions (straight corridors not aligned with any neighborhood step)
the discrete backends pay their full elongation bound and FIM does not.

The real-raster row (plan §6.2's "real cost raster from the test data"):
`examples/data/raster/small_raster.tiff`, 3 sources × 4 targets snapped
to the nearest passable cell. FIM prices 9 of 12 routes below R2 (up to
+1.14%) and sits marginally above R2 on 3 of 12 (worst −0.32%) — real
land-use surfaces have hard class boundaries, which reintroduce a
fraction of the §5.1 first-order error; FIM stays 2.3–4.5% below R1 on
every pair. Run: `benchmark_eikonal_accuracy.py --only real`.

## 4. Analytic validation and convergence

Grid convergence (L∞ relative error, exact-init disk fixed in
**physical** units across refinements):

| Case | n | error | observed order |
|---|---|---|---|
| uniform cone | 101 → 201 → 401 | 4.54% → 2.23% → 1.13% | 1.03, 0.98 |
| Snell target | 100 → 200 → 400 | 1.24% → 0.63% → 0.32% | 0.98, 0.98 |

float32 does **not** floor the order at these sizes (plan §3.1 question
answered — no double-precision science flag needed for increment 1).

Additional validated cases (tests): radial linear slowness
`c = a + b·r` vs `T = aR + bR²/2` (≤2%), multi-source = pointwise min
(§5.3), exclusion wall/gap geometry, unreachable enclosure = 1e30,
refracted path bends at the Snell interface.

## 5. Honest caveats

### 5.1 Cell-scale cost noise (FIM's worst accuracy case)

On Gaussian-blurred noise with **short correlation length (σ=4 cells)**
FIM prices corner-to-corner routes **~2.3% above R2** (mean over 20
seeds; range 1.4–3.2%; still 0.4% below R1). The first-order scheme
smears cell-scale cost structure, while the discrete backends evaluate
exact per-cell edge sums. At σ=16+ the effect vanishes (±0.2%).
**Practical rule: if the cost surface varies at the single-cell scale,
prefer the discrete backends; FIM's accuracy advantage lives where cost
is piecewise-smooth at the working resolution.**

### 5.2 Point-source singularity (why `disk_radius` exists)

With the default 3-cell exact-init disk, the source singularity pollutes
the field along diagonals: L∞ ≈ 4.7% near the source (r≈7 cells),
decaying ~1/√r; axis directions are exact. This is the known O(√h)
first-order behavior — a fixed-cell disk shrinks physically under
refinement. `eikonal_raster_gpu(..., disk_radius=...)` scales the disk;
fixed-physical disks restore clean O(h) (the §4 numbers). The disk only
seeds where cost is locally constant (seeding a too-low T could never be
corrected — updates are monotone non-increasing).

### 5.3 Multi-source vs min-of-singles

`min(T_s1, T_s2)` of the discrete single-source fields is a
*supersolution*: the multi-source solve is never above it, but at shock
cells the Godunov quadratic mixes the wavefronts and lands up to ~0.5%
*below* (closer to the continuum truth). Tests assert the one-sided
bound + tight agreement away from shocks — never pointwise equality.

### 5.4 Thin barriers

**No tunneling**: barriers are hard-excluded from the stencil (`c=1e30`
cells never update), so a full wall blocks at any thickness — measured
and tested. The corner cost around a wall tip is *over*-estimated by
~+2.0% (w = 1, 2, 3 cells, vs the analytic corner geodesic) — the
under-estimation the survey warns about for smoothed formulations does
not occur in this hard-exclusion formulation. Grazing artifacts in path
*rasterization* are handled by the `forbidden_mask` in
`polyline_to_cells` (a legal continuous path may round into a wall
corner cell).

### 5.5 Shock smearing

Two-source kink: the gradient transitions within **~1 cell** of the
shock line (profile in the accuracy JSON). No visible smearing at
first order with the eps used.

### 5.6 Convergence vs cost contrast (expected FIM weakness — small here)

| max/min contrast | outer passes (500²) | wall |
|---|---|---|
| 10 | 72 | 3.9 ms |
| 100 | 80 | 6.9 ms |
| 1000 | 88 | 6.8 ms |
| 10000 | 88 | 6.7 ms |

The feared value-iteration blowup does not materialize at these
contrasts — tile-granular re-activation localizes the extra work. The
cap (`64·max(tiles_r, tiles_c)`) raises loudly with the measured
contrast in the message; the fallback recommendation for pathological
rasters is the discrete `raster_gpu` backend.

### 5.7 Tracer cost (known bottleneck → increment 2)

Path tracing is host-side Python/numpy; after the scalar-math rewrite it
still dominates end-to-end time (e.g. 3000²: solve 63–115 ms, trace
645–725 ms for a corner-to-corner diagonal, ~8500 Heun steps). The plan's
"negligible next to the solve" assumption is wrong at 2000²+ —
a device-side (or numba) tracer is the first increment-2 item. Note the
field amortizes: multi-target routing pays one solve and reuses the
prepared gradient fields across traces.

**RESOLVED in increment 2** (§10.1): the device tracer + vectorized
rasterizer cut the 3000² trace to 88–102 ms; the host tracer remains
as the reference implementation and the fallback for plateau/cap
states.

Tracer robustness (all tested): monotone-T rule (a gradient step may
never raise interpolated T — kills the interpolation-attractor cycles
found on cell-scale-noise fields), plateau BFS escape (zero-cost plazas),
shock-oscillation discrete hop, wall avoidance, hard step cap that
raises.

## 6. Performance (median of 3, corner-to-corner)

FIM = full-field solve + trace; Cython Dijkstra single-target; V5 with
targeted early exit (each solver's production behavior).

| Raster | n | Cython | V5 | FIM solve | FIM trace | vs Cython | vs V5 |
|---|---|---|---|---|---|---|---|
| uniform | 1000² | 323 ms | 12 ms | 9 ms | 101 ms | 2.9× | 0.11× |
| uniform | 3000² | 3296 ms | 191 ms | 63 ms | 645 ms | 4.7× | 0.27× |
| random | 1000² | 405 ms | 17 ms | 13 ms | 112 ms | 3.2× | 0.13× |
| random | 3000² | 4092 ms | 163 ms | 102 ms | 666 ms | 5.3× | 0.21× |
| heavy_tail | 1000² | 481 ms | 15 ms | 15 ms | 103 ms | 4.1× | 0.13× |
| heavy_tail | 3000² | 4717 ms | 123 ms | 115 ms | 652 ms | 6.1× | 0.16× |
| smooth | 1000² | 418 ms | 16 ms | 13 ms | 115 ms | 3.3× | 0.13× |
| smooth | 3000² | 4402 ms | 190 ms | 90 ms | 725 ms | 5.4× | 0.23× |

(500² rows in the JSON; the uniform-500² V5 sample is an outlier —
36.8 ms vs 4–5 ms in every neighboring measurement.)

- Solve throughput: 60–142 Mcells/s, *rising* with raster size.
- Heavy-tail — nominally FIM's worst case — shows the best
  Cython-relative scaling (6.1× at 3000²): passes grow only mildly
  (392 vs 376 uniform).
- **V5 wins single-pair wall-clock** (plan: parity was a bonus, not a
  gate). FIM's accuracy is the product; its full field additionally
  amortizes over multi-target requests where the discrete backends pay
  per pair.

### Tuning (1000² random, fixed as defaults)

tile=16 with `n_inner = 2·tile = 32` wins (14.1 ms vs 22.5 ms at
`n_inner=16`); tile=8/32 are 3–40% behind at their best `n_inner`.
Larger `n_inner` reduces re-activation churn — outer passes drop ~30%.
Defaults in `eikonal_raster_gpu`: `tile=16`, `n_inner=2·tile`,
`sweep_blocks=16·SMs` (grid-stride over the device-built active list).

## 7. ArcGIS Distance Accumulation comparison

Harness complete under `benchmarks/arcgis_comparison/` (generator ran;
GeoTIFFs + source GeoPackages + our FIM fields + `PROCEDURE.md` +
`compare_results.py`). ArcGIS runs are manual (license); the pyorps side
of the table:

| case | solver | L∞ rel | mean rel | bias |
|---|---|---|---|---|
| uniform | fim | 4.65% | 0.64% | +0.64% |
| radial | fim | 12.56%* | 0.98% | +0.98% |
| snell | fim | 4.59% | 0.56% | +0.56% |
| barrier | fim | probes: mirror +1.98%, high +1.67%, low +2.31% | | |

| smooth_random | fim | *reference* (no closed form) | | |
| real_raster | fim | *reference* (no closed form) | | |

\* L∞ values sit in the near-source ring (r≈5 cells) — §5.2; the radial
case additionally gets no exact-init disk (cost not locally constant).
The ArcGIS column fills in whenever a licensed session is available
(drop rasters into `results_arcgis/`, re-run `compare_results.py`).

The two §6.2 cases without a closed form (`smooth_random` σ=16 400²,
deterministic seed; `real_raster` = small_raster.tiff at its true
EPSG:25832 location on an idealized exact-1 m grid) are cross-checks:
`compare_results.py` reports the ArcGIS-vs-FIM relative difference and
a coverage-mismatch count (cells reached by exactly one solver would
expose differing barrier semantics). Agreement within discretization
error is the expected outcome; a systematic gap flags an algorithmic
difference (ArcGIS's richer 12-approximation local stencil vs our
4-stencil), not an error of either side.

## 7b. Relation to the original FIM formulation (source verification)

Verified against Fu, Jeong, Pan, Kirby & Whitaker, *A Fast Iterative
Method for Solving the Eikonal Equation on Triangulated Surfaces*, SIAM
J. Sci. Comput. 33(5), 2011 (PMC3360588) — the paper that states the FIM
convergence invariants explicitly and introduces the patch-based GPU
scheme (patchFIM) this tile solver follows.

**The three FIM invariants (paper §2.3) and our mapping:**

| Invariant | patchFIM (paper) | block-FIM (ours) |
|---|---|---|
| (a) any vertex whose value *may* be inconsistent with its neighbors must be on the active list | neighbors of a *converged* patch added unconditionally | tiles whose halo changed (neighbor `boundary_changed`) activate — a conservative superset, since a 4-stencil cell can only become inconsistent through a changed halo |
| (b) removed only when consistent with its neighbors | per-vertex `Cv` flags + tree reduction → patch flag | post-loop probe: "can any cell still improve > eps against the current halo?"; concurrent halo changes re-activate via (a) in the same pass |
| (c) terminate only when the active list is empty | `UpdateActiveList` rebuilds L from flags | host loop ends at device-built list count 0 |

**Shared design decisions** (same as the paper): patch/tile-granular
active list; halo duplicated and held *fixed* during internal iterations
(the paper's acknowledged "boundary conditions lag" — corrected by
re-activation, both there and here); fixed internal iteration count
tuned empirically (their optimum n≈7 for ~64-vertex patches ≈ 0.9×patch
side, and "the optimal choice of n depends … on the input speed
function" — our sweep landed on 2×tile side for 256-cell tiles on this
hardware); single precision throughout; iterations grow with
speed-function complexity (their Speed 1→Speed 2; our contrast curve
§5.6).

**Deliberate deviations (both correctness-preserving):**

1. *Neighbor activation on boundary change instead of on patch
   convergence.* The paper's `CheckNeighbor` adds all neighbors of a
   patch only once it converges; we activate as soon as boundary values
   actually changed (> eps), so the wavefront crosses tile borders one
   pass sooner and converged-but-unchanged tiles never wake their
   neighbors. Still a superset of invariant (a).
2. *Monotone chaotic relaxation in one shared buffer + a single
   convergence probe* instead of double-buffered Jacobi + per-vertex
   flags + tree reduction. Racy reads of monotone non-increasing values
   are always valid upper bounds, so the fixed point is unchanged; and
   because a chaotic sweep's "no change in the last iteration" would
   *not* imply a fixed point (unlike deterministic Jacobi), the probe
   re-evaluates the update operator once — the correct convergence test
   for this variant. Bonus: half the shared memory, no reduction tree.

**Update-redundancy metric** (the paper's Table 3.4 reports 105–291
local-solver calls per vertex for patchFIM vs ~18 for FMM — the honest
cost of value iteration; now measured here via
`eikonal_raster_gpu(..., return_stats=True)`; nominal upper bound —
counts all `n_inner` iterations of every tile sweep):

| Scenario (1000²) | outer passes | tile sweeps | updates/cell |
|---|---|---|---|
| uniform | 64 | 7 934 | 65 |
| smooth | 72 | 16 100 | 132 |
| random | 72 | 35 012 | 287 |
| heavy_tail | 88 | 45 351 | 372 |
| spiral corridor (200²) | 40 | 450 | 92 |

Same order as the paper's numbers — the redundancy is what the massive
parallelism pays for.

**Spiral stress test** (wound characteristics — the geometric extreme of
the paper's complexity caveat, not covered by their experiments): a
rectangular spiral corridor on 200² (path cost > 5× the grid diagonal)
converges in 40 passes (default cap: 832), matches the naive oracle to
< 1e-3, and the tracer follows the corridor to the source. Added as a
permanent regression test.

## 8. Engineering notes (for the next kernel author)

- **No atomics/queues/barriers in the cell-update path**; the only
  atomic is tile-granular list building in the activation kernel. The
  entire V5 race-bug class is structurally absent. No cooperative
  launch → no Blackwell grid-sync / ≤512-threads-per-SM hazards.
- **Sweep kernel = grid-stride over the device-built active list.**
  The active count lives on-device; sizing the grid to all tiles
  scheduled thousands of empty blocks per pass (~50% of pass time on a
  14-SM part). Fixed `sweep_blocks` + per-block tile loop fixed it.
- **Shared-flag inner loop race (the one real bug of the campaign):**
  breaking a `__syncthreads()`-bounded relaxation loop on a shared flag
  that thread 0 clears next iteration lets a lagging thread observe the
  cleared flag, break alone, and (with per-block tile looping) corrupt
  the shared tile for the still-running threads → undershoot below the
  fixed point. Fixed by removing the per-iteration protocol entirely:
  monotone chaotic relaxation needs none; convergence is decided by a
  single post-loop probe ("can any cell still improve > eps?").
- **Host loop = CUDA-graph windows.** Python launch overhead (~5 CuPy
  calls/pass × up to hundreds of passes) is captured once (8 passes) and
  replayed; one D2H count read per window. Plain per-pass loop remains
  as fallback (no capture support / tiny iteration caps). Termination
  inside a window is safe: empty-list sweeps are no-ops.
- **`boundary_changed` memset per pass is load-bearing** — stale flags
  on tiles that never get re-swept would re-activate their neighbors
  forever (stamp-based schemes alias at window phase boundaries; do not
  "optimize" the memset away without solving that).
- uint16/float32 rasters both feed one kernel via a host-side float32
  slowness field (sentinel → 1e30) — deliberate deviation from
  sssp_gpu's textual kernel-variant transform; one kernel, one sentinel
  path, 4 B/px as budgeted.

## 9. Future increments (plan §10)

Items 1–3 were DELIVERED in increment 2 (§10 below); remaining:
1. Anisotropy (slope/DEM via HFM-style stencils) — `raster_fim`
   currently *raises* on DEM/gradient_luts rather than silently
   ignoring them.
2. Trace-kernel register caching of the bilinear corner block
   (positions move ≤ 0.5 cells/step — most global reads repeat); the
   single-path device trace is latency-chain-bound at ~107 ms for
   ~9000 points at 3000².

## 10. Increment 2 (implemented 2026-08-07)

Three deliverables: device tracer, targeted early exit, second-order
refinement. All 108 eikonal/API tests green.

### 10.1 Device-side tracer (`trace_paths_gpu`)

One CUDA thread per target runs the exact host-tracer semantics in
double precision (Heun descent on on-the-fly masked gradients,
NaN-aware bilinear sampling, monotone-interpolated-T rule, discrete
descent hops with per-reason stall bookkeeping). States the kernel
does not implement — plateau BFS, step cap — flag the target and fall
back transparently to the host tracer, so behavior (including the
loud RuntimeError on broken fields) is identical. On parity fields the
device and host tracers produced *identical* polylines and cell paths.
`polyline_to_cells` sampling was vectorized in the same pass
(88 ms → 2.8 ms at 3000²).

End-to-end effect at 3000² (trace was 645–725 ms in increment 1):
solve + trace now 155–223 ms total, and the updated §6 table becomes:

| Raster | n | Cython | V5 | FIM solve | FIM trace | vs Cython | vs V5 |
|---|---|---|---|---|---|---|---|
| uniform | 1000² | 337 ms | 13 ms | 10 ms | 31 ms | 8.3× | 0.31× |
| uniform | 3000² | 3358 ms | 205 ms | 66 ms | 94 ms | 21.0× | **1.28×** |
| random | 1000² | 416 ms | 17 ms | 12 ms | 34 ms | 9.0× | 0.36× |
| random | 3000² | 4380 ms | 169 ms | 106 ms | 102 ms | 21.2× | 0.82× |
| heavy_tail | 3000² | 4824 ms | 124 ms | 121 ms | 88 ms | 23.0× | 0.59× |
| smooth | 3000² | 4128 ms | 199 ms | 90 ms | 98 ms | 21.9× | **1.06×** |

**The plan §11 bonus (V5 wall-clock parity at large sizes) is met**:
FIM beats V5 end-to-end at 3000² on uniform and smooth, near-parity on
random. The remaining trace time is the inherently serial single-path
walk; multiple targets trace concurrently (§10.4 of the module docs).

### 10.2 Targeted early exit (`target_index=`)

Stops the solve once the target's tile is converged AND min T over all
active tiles (cells + 1-cell halo) ≥ T[target]. Monotone updates make
this exact up to the solver's ordinary eps guarantee: any future write
is ≥ the min over the writing tile ∪ halo, so nothing can improve the
target or any cell at T ≤ T[target] (strict-equality tests are wrong
even between two full solves — the chaotic sweep leaves
nondeterministic sub-eps slack; measured 0.0098 max vs eps ≈ 0.02).
Measured gains (random): 1.55–1.64× for quarter-grid targets,
1.18–1.28× mid-grid, 0.96–1.03× corner-to-corner (a cheap T[target]
scalar gate skips the check kernel until the wave arrives). Wired into
`RasterFIMAPI` for single-pair solves only (the field is not reusable
for other targets).

### 10.3 Second-order refinement (`order=2`)

Two-stage: the converged first-order field seeds a second-order
one-sided upwind re-solve ((3T − 4T1 + T2)/2h per axis where
applicable). The engineering that made it work — each step measured,
not guessed:

1. **The mixed-order operator is not monotone** (the second-order β
   *decreases* with T2), so it cannot ride the stage-1 chaotic
   machinery: an undershoot computed from an unconverged upper bound
   would lock in under monotone-min. Stage 2 is deterministic damped
   Jacobi, globally double-buffered, active-tile list as in stage 1.
2. **A live T2 ≤ T1 switch never settles**: candidates jump O(c/3) at
   the switch boundary, and float32 jitter flips it forever. The
   upwind structure (direction + order per axis) is **frozen once**
   from the converged first-order field (`fim_freeze2` codes).
3. **ENO-style smoothness test** at freeze time
   (|T0 − 2T1 + T2| ≤ 0.5·c): keeps cell-scale-noise regions first
   order — for accuracy *and* stability (frozen second-order chains
   through rough data amplify perturbations faster than damping
   contracts; with a 2c threshold the random 500² capped at 2048
   passes, with 0.5c it converges in 240).
4. **Causality safeguard**: a candidate below a used second-order
   axis's T1 falls back to plain first order on the frozen directions
   — without it, noise fields produced −inf runaways (β < 0 chains
   pulling each other down; 1573 cells at −inf before the fix).
5. **float32 termination floor** (eps_rel ≥ 3e-4 in stage 2): near
   the fixed point the iteration wanders in a ~6e-5-relative ULP-noise
   ball that never reaches 1e-6 (the field is correct — measured
   against float64); the floor stays 5–50× below the discretization
   error being corrected.
6. **Stall degrade**: if the active count stops improving over 6
   windows, the stuck tiles' cells demote to first order; after
   repeated stalls they freeze outright — termination is guaranteed
   by strict shrinkage of the updatable set.

**Results** (analytic referees):

| Metric | order=1 | order=2 |
|---|---|---|
| uniform worst-direction elongation (401², r=180 sweep) | +0.764% | **+0.032%** |
| uniform cone mean error (401², r ≥ 10) | 0.633% | **0.070%** |
| Snell target error (400²) | +0.375% | **+0.114%** |
| observed convergence order (101→201→401, physically scaled disk) | 0.97–0.98 | **1.86–2.08** |

For context: R2 (pyorps default discrete) worst-direction is +2.75%,
R1 +8.24%. Cost: roughly 2× the solve time (e.g. random 500²:
29 ms vs 6 ms — still far below the discrete backends). Caveats: on
cell-scale-noise fields most cells stay first order (by design) and
the refinement shifts the field down by a bounded ~1.3% mean — §5.1's
recommendation (discrete backends for cell-scale noise) stands;
`target_index` early exit is disabled with order=2 (the refinement
needs the full field). Exposed as `RasterFIMAPI(..., order=2)`.

**Trace-field contract** (found on the real-world surface, §11):
refined fields carry the *costs* but are NOT descent-connected — the
mixed-order fixed point can hold genuine local minima near cost
shocks (measured: an isolated cell 49 units below all 8 neighbors on
the planning raster). Tracing always consumes the preserved
first-order field (`return_trace_field=True`; the API does this
automatically). The first-order Godunov fixed point is
descent-connected by construction.

## 11. Real-world benchmark (2026-08-07)

`benchmarks/benchmark_eikonal_realworld.py` — the cost surface is the
*unmodified output* of
`examples/prepare_data_for_distribution_grid_planning.ipynb`: ALKIS
land-use base costs × drinking-water-protection-zone multipliers
(zone I ×100 … IV ×1.2) × soil-condition factors (DIN 18300) ×
landscape-protection ×1.25, nature reserves hard-forbidden (65535).
Window: 4096 m × 4096 m at 1 m around the notebook's demo route
(13% forbidden, passable costs 125–750). Scenario B multiplies in an
isotropic slope layer from the **real Hessen DGM1** (1 m WCS,
elevation 166–287 m, mean slope 7.9%, p95 16.3%; cached under
`benchmarks/realworld_data/`): m(s) = min(1+3s², 3), s > 45% forbidden.
4 source–target pairs (3.0–5.0 km), snapped to passable cells.

### 11.1 Wall-clock (per pair, solve + trace)

| Scenario | Cython R1 | Cython R2 | V5 (GPU) | FIM o1 | FIM o2 |
|---|---|---|---|---|---|
| A combined | 5.5–18.6 s | 14.2–29.3 s | 0.34–0.47 s | **0.24–0.50 s** | 9.4–11.9 s |
| B + slope | 6.2–31.3 s | 14.5–32.0 s | 0.43–0.50 s | **0.26–0.77 s** | 12.0–14.4 s |

FIM order=1 (with targeted early exit) is **30–60× faster than the
production Cython R2 route** and at or slightly ahead of the discrete
GPU V5 — while producing a full T field that amortizes over further
targets from the same source.

### 11.2 Route costs (gap vs the discrete baselines)

FIM o1 vs R2 per pair: scenario A **−2.9%, −0.4%, −0.3%, +0.6%**;
scenario B **−3.7%, −1.4%, −1.0%, −0.4%** (negative = FIM *above*
R2). vs R1: FIM below by 0.5–3.9% everywhere.

Honest reading: on piecewise-constant planning surfaces (hard class
boundaries everywhere, plus cell-scale roughness from the slope
multiplier in B) the first-order scheme loses most of its accuracy
edge over R2 — consistent with §5.1 — and can price a few percent
*above* it. **On this raster class FIM's advantages are speed, the
reusable field, and freedom from R1-style directional bias — not
beating R2 on cost.** R2 remains an excellent accuracy baseline; FIM
o1 matches it within a few percent at 1/40th the CPU cost.

**order=2 is NOT recommended here**: costs undershoot up to ~10%
below R2 and solves take ~10 s — second-order extrapolation
mis-behaves across the dense shock structure of piecewise-constant
surfaces. Its validated domain is smooth/analytic cost fields (§10.3).

### 11.3 Slope handling and the direction-blindness of isotropic layers

Scenario B treats slope isotropically (a per-cell multiplier) — the
only form the eikonal backend supports until the anisotropic
increment 3. The contour test quantifies what that misses: on a
uniform 20% hillside the isotropic layer charges a contour-following
route (zero true climb) the **full multiplier, +12.0% pure error**,
while a fall-line route is priced correctly — the error spans 0% to
the whole multiplier depending on route direction. The discrete
backends with `dem=` + gradient objective price direction correctly
today; that comparison point stands until increment 3.

### 11.4 Engineering fallout (fixed during this benchmark)

- Plateau-BFS band widened to 4× tol: near-shock slack pockets on
  large-T fields (T ~ 10⁶) could exceed one tol (nondeterministic,
  chaotic-sweep slack) and strand the tracer.
- The order=2 trace-field contract above — the refined field held a
  genuine 49-unit-deep local minimum on this surface.

## 12. Performance increment 2.5 (implemented 2026-08-07)

Plan: `2026-08-07-eikonal-performance-increment.md`. Baseline (§0
there): 4096² real-world single pair ~223 ms = ~85 kernel + ~61
pipeline + 74 trace. **Result: 95.9 ms end-to-end (2.3×) — goal G1
(≤130 ms) met; V5 beaten on all four synthetic patterns at 3000²
(G2); order=2 bounded everywhere (G3). All 172 tests green.**

### 12.1 What was done, with measured deltas

1. **Pipeline (P1)**: slowness conversion moved to the device (raw
   uint16 upload = half the bytes; kills the 28 ms host pass);
   source validation + disk-init windows read the raster directly;
   eps from a device reduction; `download=False` solver mode + a lean
   API single-pair path (T[target] as a device scalar, device-only
   tracing, uint8 forbidden mask instead of the 20 ms float D2H).
   Real-world pair 212 → ~146 ms. Pinned staging was dropped: after
   the uint16 upload there is too little transfer left to matter.
2. **Sweep settle checks (P2)**: every 8th inner iteration, a
   two-barrier settle check lets finished tiles exit (the break is
   read between two barriers with no intervening writes — uniform;
   NOT the increment-1 racy-flag pattern). updates/cell 65→40
   (uniform), 287→190 (random), 372→236 (heavy_tail); 3000² random
   solve 105 → 62 ms; outer passes unchanged. The stats counter now
   counts actual iterations (`stats[1]`). Re-tuning confirmed
   `n_inner = 2·tile` stays optimal (48 wins at 1000² but loses 58%
   at 3000²).
3. **Tracer (P3) — hypothesis falsified, pivot recorded**: the
   planned 4×4 register-block cache landed with exact parity but
   ZERO speedup — the walk was never memory-bound (path locality
   keeps L2 warm); the serial FP64 dependency chain was (consumer
   silicon: 1/64 FP64 rate). Converting the walk math to float32
   (T is float32 anyway; polyline output stays float64; host
   reference tracer stays float64) gave 99 → **28 ms** at 3000².
   The block cache is kept — with fast ALU, memory is next in line.
4. **order=2 guard + budget (P4)**: the refinement runs only inside
   its validated domain — cost-jump edge density ≤ 2% AND forbidden
   fraction ≤ 2% (eligibility fraction was tried first and measured
   useless: the piecewise-constant real raster is 99.9% "eligible"
   yet misbehaves — its shocks come from obstacle diffraction, not
   cost jumps). Outside: warning + first-order field. Pass budget
   6× stage-1 with warn-and-fallback instead of a raise; the stall
   degrade only fires in the second half of the budget (early
   demotion froze legitimately-converging tiles and cost the
   convergence order at 401²). Real-raster order=2: 10 s → 157 ms.
5. **Coarse-to-fine seeding (P5) — REJECTED BY MEASUREMENT**: the
   4×-max-pooled + dilated supersolution seed cut outer passes only
   ~20% (its pool-scale slack means the correction wave still
   crosses the whole grid) while costing ~110 ms of coarse solve +
   full-tile settling: a 10× net loss at every size, plus
   above-slack deviations on barrier rasters. Removed; a note in the
   module marks the grave.

### 12.2 Final numbers

3000² single pair (solve + trace), median of 3:

| Pattern | Cython R1 | V5 | FIM | vs V5 (was §10.1) |
|---|---|---|---|---|
| uniform | 3758 ms | 223 ms | **58 ms** | **3.87×** (1.28×) |
| random | 4568 ms | 171 ms | **99 ms** | **1.72×** (0.82×) |
| heavy_tail | 5120 ms | 124 ms | **105 ms** | **1.18×** (0.59×) |
| smooth | 4291 ms | 173 ms | **95 ms** | **1.82×** (1.06×) |

Solve throughput up to 305 Mcells/s (was ~135). Real-world 4096²
planning raster, lean single-pair flow: **95.9 ms** end-to-end
(increment 1: ~500 ms; increment 2: ~223 ms). Remaining split at
4096²: ~solve 60 ms (272 passes) + trace 19 ms + rasterize 8 ms +
residual transfers/prep.

### 12.3 Where the next factor would come from

- The outer-pass count itself (272–392) is now the solve's floor;
  the P5 negative result says naive coarse seeding cannot buy it —
  a tighter bound (factor 2, exact prolongation) or frontier-band
  scheduling would be the next research item.
- The trace walk (19–28 ms) is a serial float32 chain now; a
  warp-parallel step evaluation is the remaining idea (plan §8).
- Host-side residuals (~30 ms at 4096²: slowness-prep numpy view +
  polyline rasterize + Python) — diminishing returns.

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


---

## 13. Tier A - slope-aware (3D-length) anisotropic solver (2026-08-11)

Implements `docs/superpowers/plans/2026-08-11-tier-a-3d-fim-implementation.md`.
Machine: RTX PRO 500 Blackwell Laptop, 6113 MiB, 14 SMs, CUDA 13.0.
Every GPU number below is MEASURED on this machine; nothing here is
carried over from the plan's predictions.

### 13.0 What was built

With a DEM the solve becomes Riemannian with metric
`M(x) = c(x)^2 (I + grad_z grad_z^T)`, which is exactly - and only - the
unconditional 3D-length stretch `sqrt(1 + (s/100)^2)` that is pyorps'
default `GradientOptions`. Configured multiplier curves and the additive
exposure term are refused (they are not of the form `sqrt(d^T M d)`; a
continuum solver would silently solve the convexified problem). The hard
grade limit is enforced OUTSIDE the solver by a solve/check/mask/re-solve
loop. Local solver: 8 angularly-ordered offsets, 8 one-sided edge
candidates + 8 simplex candidates with a mandatory causality test.
Isotropic behaviour is untouched - a separate kernel, not a special case.

Scope of that last sentence, since "untouched" invites over-reading. What
is measured is: with `dem=None` the anisotropic code is not reachable
(different kernel, same tuning defaults), and with an EXACTLY CONSTANT
DEM at the SAME tiling the field is BIT-identical to the no-DEM solve.
The bit-identity is a statement about the operator, and it only survives
where the block schedule is reproducible: on a uniform cost raster it
held at every size tried (up to 256^2), but on random / barrier cost
rasters it held 24/24 at <= 160 px per side, 23/24 at 176 and 13/24 at
192. The solve is chaotic relaxation with an atomic active list, so above
that the run-to-run field is not bit-reproducible either (identical
through 192^2, differing at 208^2 and above, max |dT| 2.4e-3 on fields of
O(10^3)) - a scheduling artefact bounded by the convergence epsilon, not
a correctness bug. The two assertions in
`tests/test_utils/test_eikonal_gpu_anisotropic.py` carry the regime as
`BITWISE_DETERMINISTIC_MAX_DIM` and refuse to run outside it.

### 13.1 The stencil result (why 8 simplices, not 4)

A simplex `(e1, e2)` is metric-acute iff `e1^T M e2 >= 0`. For the four
AXIS QUADRANTS that quantity is `+-c^2 q_r q_c`, so **two of the four are
obtuse for every non-axis-aligned slope** - the 4-point quadrant stencil
the parent plan proposed is biased at every slope azimuth. Maximum
relative over-estimate on EXACT LINEAR FIELDS (721 gradient directions x
37 slope azimuths, `c = 1`; reproduced in-repo by
`tests/test_utils/test_eikonal_gpu_anisotropic.py::TestSimplexExactnessOnLinearFields`):

| grade | kappa | 8-simplex | 4-simplex |
|---|---|---|---|
| 10 % | 1.0050 | **0.0000 %** | 0.0012 % |
| 20 % | 1.0198 | **0.0000 %** | 0.0192 % |
| 50 % | 1.1180 | **0.0000 %** | 0.6231 % |
| 100 % | 1.4142 | **0.0000 %** | 6.0660 % |
| 141 % | 1.7321 | **0.0000 %** | 15.4701 % |
| 200 % | 2.2361 | **0.0000 %** | 28.7290 % |
| 220 % | 2.4142 | **0.0000 %** | 34.9591 % |
| 250 % | 2.6926 | 0.5960 % | 41.7828 % |
| 300 % | 3.1623 | 3.6650 % | 57.6221 % |
| 400 % | 4.1231 | 14.6690 % | 87.7806 % |

(The 4-simplex column is larger than the parent plan's figures because it
is normalised against the neighbour-value scale rather than a modelled
path cost; the ordering and the conclusion are identical.)

Acuteness for the (axis, adjacent-diagonal) pairs reduces to
`|q_r q_c| <= 1 + min(q_r^2, q_c^2)`, which holds for all
`|q| <= sqrt(2(1+sqrt2)) = 2.19737` (`kappa <= 1+sqrt2 = 2.41421`).
Numerical sweep on a 0.01 grid: **first failure at `|q| = 2.20`,
kappa = 2.4166** - the derived threshold, confirmed. pyorps' default
`s_max_pct = 200 %` is inside the provably exact regime, so the guard
never fires by default; above it the solver RAISES rather than
over-pricing silently.

Supporting identities, verified: Sherman-Morrison `max |M M^-1 - I| =
7.5e-16`; the 3D-length identity `hypot(|d|, q.d) = sqrt(d^T (I+qq^T) d)`
to 1.8e-15. `det(Gram) = 1 + |q|^2` exactly and `S = 1 + q_axis^2`, so
neither can vanish and neither is a difference of large numbers - that is
what keeps the float32 update well conditioned at planning-raster cost
magnitudes.

### 13.2 Accuracy: calibration anchors (constant-slope plane, 301^2)

The referee is the analytic 3D length, never the discrete cost.

| grade | case | exact | FIM | R2 | FIM gain over R2 |
|---|---|---|---|---|---|
| 10 % | A3 axis-aligned | 300.749 | **+0.000 %** | +0.004 % | +0.003 % |
| 10 % | A4 exact R2 direction (1,2) | 336.916 | +0.125 % | -0.010 % | -0.135 % |
| 10 % | A5 R2 worst direction | 308.937 | +0.121 % | **+2.732 %** | **+2.608 %** |
| 25 % | A5 R2 worst direction | 314.668 | +0.118 % | +2.658 % | +2.537 % |
| 50 % | A5 R2 worst direction | 334.337 | +0.109 % | +2.461 % | +2.349 % |
| 100 % | A5 R2 worst direction | 401.664 | +0.081 % | +1.853 % | +1.770 % |

- **A3 is exact to 0.000 %** - the calibration gate (<= 1e-5 rel) passes.
  A plane sloping along the traverse is priced at exactly
  `sqrt(1 + (s/100)^2)` per cell of run: Tier A and nothing else.
- **The harness validates against the isotropic record**: R2's worst
  direction at 10 % grade measures **+2.732 %**, reproducing section 3's
  isotropic +2.75 % metrication figure.
- **The plan's predicted gain (2.77 % -> 4.93 %, RISING with grade) is
  wrong in its trend.** Measured: **2.61 % -> 1.77 %, FALLING with
  grade.** The reason is clean: as the slope steepens more of the cost is
  vertical rise, and the discrete kernel gets the rise exactly right
  (the height difference between two cells is not a metrication
  estimate). Only the horizontal component carries the elongation bias,
  so its relative weight - and the gap - shrinks. Honest headline: **a
  ~1.8-2.6 % correction on a planar slope, largest on gentle terrain**;
  not a step change.

**Neighborhood refinement** (50 % grade, R2 worst direction, exact =
360.064) - the operational meaning of "agrees with the discrete kernels":

| | cost | vs exact | vs FIM |
|---|---|---|---|
| FIM | 360.480 | +0.116 % | - |
| R1 | 385.250 | +6.995 % | +6.871 % |
| R2 | 363.736 | +1.020 % | +0.903 % |
| R3 | 360.705 | +0.178 % | +0.062 % |
| R4 | 360.705 | +0.178 % | +0.062 % |

`discrete_R` approaches the FIM value **monotonically from above**. That
is the contract of the plan's section 8.1, and it holds.

**Contour test** (pure hillside, slope along rows only, 301^2):

| grade | traverse | exact | FIM+DEM | R2+DEM | 2D-only (no DEM) |
|---|---|---|---|---|---|
| 20 % | contour (across the slope) | 300.000 | +0.000 % | +0.000 % | +0.000 % |
| 20 % | fall line (down the slope) | 305.941 | +0.000 % | +0.024 % | **-1.942 %** |
| 50 % | contour | 300.000 | +0.000 % | +0.000 % | +0.000 % |
| 50 % | fall line | 335.410 | -0.000 % | +0.050 % | **-10.557 %** |

Tier A is exact on both traverses at both grades. A DEM-less solve is
exact on the contour and under-prices the fall line by up to 10.6 % -
the direction dependence a per-cell slope layer structurally cannot see.

**Realistic terrain, random cost raster** (401^2, 12 random pairs, peak
grade 50 %): `T_FIM <= R2` on **12/12** pairs, mean gain **+30.6 %**, min
+19.5 %. This is far larger than the planar figure and it is NOT a slope
effect: on a high-contrast random cost surface the continuum threads
between expensive cells at any angle while R2 is confined to 16
directions. Quote the planar ~2 % as the slope-accuracy number and this
as a raster-roughness number; they measure different things.

### 13.3 Performance

**P4 - q-build (gate <= 5 ms at 3000^2): MET.**

| size | cells | total ms | of which H2D | kernel ms | kernel GB/s |
|---|---|---|---|---|---|
| 1000^2 | 1.0 M | 0.64 | 0.33 | 0.05 | 241 |
| 2000^2 | 4.0 M | 1.75 | 1.73 | 0.30 | 173 |
| 3000^2 | 9.0 M | **3.72** | 3.56 | 0.69 | 169 |

The kernel is bandwidth-bound and essentially free; the step is dominated
by the DEM upload. *Trap found and fixed:* computing the float32
reference elevation with a full-array `isfinite` + boolean gather cost
**25 ms at 9 M cells - 36x the kernel** and made the step look
bandwidth-bound when it was not. It is taken from a strided subsample now
(it only has to be a representative offset).

**P1 - anisotropic vs isotropic solve (gate <= 3x): MISSED, reported.**

| size | class | iso ms | aniso ms | ratio | iso upd/cell | aniso upd/cell |
|---|---|---|---|---|---|---|
| 3000^2 | uniform | 19.8 | 88.1 | 4.44 | 39.7 | 24.4 |
| 3000^2 | random | 57.2 | 484.5 | **8.48** | 212.8 | 194.4 |
| 3000^2 | heavy_tail | 66.1 | 440.0 | 6.66 | 244.6 | 187.6 |
| 3000^2 | smooth | 68.2 | 344.8 | 5.06 | 237.1 | 148.7 |
| 2000^2 | random | 26.9 | 187.2 | 6.96 | 220.5 | 176.7 |
| 1000^2 | random | 9.3 | 50.7 | 5.47 | 281.1 | 182.2 |

Ratio **1.8-8.5x** against a gate of 3x - missed on every class except
uniform at 2000^2. Two things are worth separating:

- **Iteration count went DOWN, not up.** `updates_per_cell` is 194 vs 213
  (random, 3000^2) and 24 vs 40 (uniform). The parent plan predicted
  +25-35 % from Fu/Kirby/Whitaker; the implementation plan's section 4.4
  predicted the opposite sign, because every 8-simplex update raises `T`
  by at least `c * 1` (the chords are the edges of the square [-1,1]^2;
  minimum M-distance measured 1.0000) against the 4-point diamond's
  `c/sqrt(2)`. **Section 4.4's prediction is the one that held.**
- So the whole cost is per-update arithmetic: 16 candidates, 8 runtime
  square roots, ~64 registers against the isotropic 30. Register count is
  what binds - see the tuning result.

**Tuning: the anisotropic optimum is NOT the isotropic one.** 3000^2,
random: `B=16, n_inner=32` (isotropic default) = 1009 ms; `B=12,
n_inner=16` = 527 ms; `B=8, n_inner=12` = 550 ms. B=12 wins on every
class tested. Two effects: at 64 registers a 256-thread tile (B=16) drops
to 4 blocks/SM where B=12 gets 7; and the larger per-update reach means
`n_inner = 2B` runs iterations that no longer buy anything. New defaults
WITH a DEM: **`tile = 12`, `n_inner = 4B/3`**. The isotropic defaults
(16, 2B) are untouched.

Optimisations applied and their measured worth: hoisting the T-independent
metric algebra out of the inner loop (removes 8 of the 16 sqrt) - small,
the compiler was already doing much of it; exploiting the **period-4
symmetry** of the metric (offset k+4 is the negation of offset k, so the
diagonal, edge, off-diagonal and S tables all repeat) to halve its
register footprint; the tile/n_inner retune - **2x, the only large win**.

**P2 - end-to-end vs Cython Dijkstra R2 WITH THE SAME DEM (gate >= 4x):
MET with margin.** Single pair, corner to corner, targeted early exit.

| size | class | FIM ms | V5+DEM ms | Cython R2+DEM ms | vs V5 | vs Cython |
|---|---|---|---|---|---|---|
| 1000^2 | uniform | 26.7 | 26.9 | 919.8 | 1.01 | 34.4 |
| 1000^2 | random | 66.9 | 37.7 | 1059.8 | 0.56 | 15.8 |
| 1000^2 | heavy_tail | 64.5 | 41.4 | 1269.0 | 0.64 | 19.7 |
| 1000^2 | smooth | 50.0 | 46.2 | 1103.4 | 0.92 | 22.1 |
| 2000^2 | uniform | 100.6 | 88.3 | 4507.5 | 0.88 | 44.8 |
| 2000^2 | random | 230.2 | 119.0 | 4603.9 | 0.52 | 20.0 |
| 2000^2 | heavy_tail | 234.1 | 131.5 | 7201.7 | 0.56 | 30.8 |
| 2000^2 | smooth | 175.4 | 145.0 | 5507.4 | 0.83 | 31.4 |
| 3000^2 | uniform | 135.0 | 219.9 | 13601.9 | **1.63** | **100.7** |
| 3000^2 | random | 488.7 | 264.1 | 11536.1 | **0.54** | 23.6 |
| 3000^2 | heavy_tail | 483.5 | 258.2 | 13222.4 | 0.53 | 27.4 |
| 3000^2 | smooth | 362.5 | 268.0 | 10923.5 | 0.74 | 30.1 |

**P3 - vs V5 + DEM: LOST on high-contrast rasters. That is the finding,
and it is a change of sign from the isotropic case.** Isotropically the
FIM backend beat V5 at 3000^2 (1.28x uniform, 1.06x smooth, 0.82x
random). With the DEM it is **0.52-0.74x on random / heavy_tail / smooth**
and wins only on `uniform` (1.63x at 3000^2). V5's cost rises modestly
when a DEM is added (its per-edge LUT lookup is cheap) while the FIM
solve pays 5-8x. Recommendation: **with a DEM, `raster_gpu` (V5) is the
faster single-pair backend on rough cost surfaces**; `raster_fim` is
chosen for accuracy (no metrication bias) and for multi-target work.

**P6 - multi-target amortisation** (2000^2, one FIM field vs k V5
solves):

| k targets | FIM ms | V5 ms | speedup |
|---|---|---|---|
| 1 | 193.8 | 121.7 | 0.63 |
| 2 | 193.1 | 124.0 | 0.64 |
| 4 | 208.4 | 322.5 | 1.55 |
| 8 | 228.0 | 748.5 | 3.28 |
| 16 | 242.0 | 1191.8 | **4.93** |

**Break-even at k ~ 3.** One field serves every target of a source at
essentially constant cost (194 -> 242 ms from k=1 to k=16, all of the
growth in tracing); the discrete backends pay per pair. The structural
advantage, quantified rather than asserted.

**Memory** (measured, CuPy pool high-water): **19.0 B/cell** with the DEM
against 10.4 B/cell isotropic - 163 MB at 3000^2, 453 MB at 5000^2. The
plan predicted 17 B/cell; the excess is pool granularity. The DEM is
uploaded in 1024-row slabs so it is never resident in full alongside the
metric planes.

### 13.4 The grade limit: solve, check, mask, re-solve

`max_gradient_pct` is enforced outside the solver. Each returned cell path
is verified with the DISCRETE kernel's own binned arithmetic (slope bin =
int(height difference * 100 / (step length in cells * cell size * bin
width)), clamped to the last bin, forbidden iff that bin's multiplier is
infinite) before it is returned - deliberately binned rather than
compared against the raw limit, so a route accepted here is the route a
discrete re-check accepts.

Measured (2000^2, terrain peak grade 90 %, 8 random pairs):

| limit | mode | feasible | median iters | p95 iters | ms/pair | mean cost |
|---|---|---|---|---|---|---|
| 15 % | lazy | 1/8 | 5.5 | 6.6 | 124.6 | 4674.6 |
| 15 % | eager | 1/8 | 0 | 0 | 10.1 | 4793.3 |
| 30 % | lazy | 5/8 | 2.0 | 6.6 | 85.5 | 5649.2 |
| 30 % | eager | 5/8 | 0 | 0 | 16.8 | 5676.1 |

- Typical iteration counts (median 2-5.5) are near the plan's guess
  (0-3), slightly above it.
- Lazy buys a **2.5 % cheaper route** than eager for ~8x the wall clock.
- **Worst case measured: 22 iterations** on a synthetic Gaussian hill at
  a 20 % limit, and the plan's minimal endpoint-only mask rule **blew the
  32-iteration cap entirely** at a 12 % limit. The shipped rule therefore
  also masks the steep cells ON THE ROUTE JUST TRIED - every masked cell
  still individually exceeds the limit, and only cells on routes actually
  attempted are ever masked, so it remains strictly lazy. With it the
  12 % case converges in 24 iterations to a route costing 93.30 against
  eager's 95.67.
- The cap **raises**; it never escalates to `eager` automatically,
  because that is a strictly smaller feasible set and substituting it
  silently is the failure class this increment exists to prevent.

**What this is honestly worth — measured, not asserted.** The returned
route is never invalid, and it is optimal for the MASKED problem. But the
masked problem is a long way from the rule, and "slightly conservative"
(the wording this section used to carry) is wrong by a factor of roughly
ten.

Differential fuzz against the Cython kernel, 4 seeds x 300 trials, 40x40
rasters, r1 steps on BOTH sides, identical `GradientLUTs`, limits {8, 12,
20, 30} %, terrain = smooth hills / white noise / plane+fault / random
walk (`benchmarks/fuzz_grade_limit_differential.py`, seeds 0-3):

| | routes found |
|---|---|
| Cython kernel (the rule, exactly) | 662 / 1200 |
| raster_fim grade-limit loop, verified | 342 / 662 |
| **false negatives** | **320 / 662 = 48.3 %** |

Per seed: 50.9 %, 40.6 %, 51.7 %, 49.7 % - the seed-to-seed spread is
real (11 points), so the pooled number is the one to quote. Seed 0 by
terrain: hills 10/29, white noise 33/38, plane+fault 5/28, random walk
34/66; by limit: 8 % 15/23, 12 % 24/37, 20 % 24/48, 30 % 19/53. White
noise is the worst family in every seed, smooth hills the best - the
rougher the terrain, the more the legal route depends on per-step
direction control that a cell mask cannot express.

**Why it misses.** The limit forbids STEPS - it is direction-dependent -
and the solver can only be told to avoid CELLS. Forbidding a cell also
forbids the contour-following traverse across it that the rule allows. On
the failing cases 89-100 % of the cells of the Cython route are "steep"
(they have at least one illegal chord), so *any* cell mask that makes
progress deletes the legal route. This was checked, not assumed: three
mask rules were implemented and fuzzed - endpoints-only, escalating
(endpoints first, route-steep cells after k iterations), and
connectivity-preserving - and on the seed-0 sample they span 50.9 %
to 55.6 %. The gap is
masking cells against a rule on steps, not the choice of rule.

**What changed as a result** (the recall was not fixable; the diagnosis
was):

- the loop never masks a cell whose removal would disconnect source from
  target in the legal-chord graph. Before, 82 of 87 failures were
  reported as "reachability was lost" - a verdict the loop had inflicted
  on itself. After, that is 3, and the rest say the mask cannot grow
  without cutting the last legal corridor;
- a patience cut-off (12 consecutive iterations without reducing the
  number of illegal steps) ends a hopeless run instead of grinding to the
  32-iteration cap: same recall as no cut-off at all (seed 0: 82/161
  either way),
  20.6 s instead of 38.6 s over the fuzz, 7 cap-outs instead of 50;
- every failure message, the module docstring and `_MASK_CAVEAT` now
  carry the 48.3 % figure and name the discrete backends as the
  authority.

**So**: a route this backend RETURNS is verified against the discrete
rule and is safe to use. A FAILURE is not a verdict on the problem -
about half of them are routes `cython` / `raster_gpu` find. The loop is a
screen; the discrete kernel is the authority on `max_gradient_pct`.

The infeasibility certificates are exact, and are now taken over the
CALLER's own step set rather than a hardcoded 8-neighbourhood. That was a
real defect: PathFinder's default neighborhood is `r2`, and a single 2 m
rise at 10 m cells is illegal for every 8-chord at a 10 % limit (axis
20 %, diagonal 14.1 %) but legal for every knight move (8.9 %) - so the
certificate declared problems INFEASIBLE that the default configuration
routes across.

The one genuine advantage over the discrete backends is that the
continuum has no stencil, hence no neighbourhood-radius reachability
limit `r >= sqrt((s_max/limit)^2 - 1)` - **but that advantage is still
derived, not observed**; the discrete reachability failure it depends on
has not been confirmed here, and the 48.3 % false-negative rate above is
the opposite result on the same axis.

### 13.5 Engineering notes (for the next kernel author)

1. **The corner halo was a real bug waiting to happen.** The isotropic
   sweep loads edge halo only and says so ("the 4-point stencil needs no
   corners"), leaving the four corner slots of the shared tile **never
   written**. The 8-simplex stencil reads them, and stale values from the
   previous tile of the same block are sometimes small - silently
   under-priced routes. `fim_sweep_aniso` loads them (threads 0-3); there
   is a regression test whose only viable route runs diagonally across
   tile corners.
2. **The admissibility test is not an optimisation.** An inadmissible
   simplex root is the minimum over the extended LINE through the two
   offsets rather than the segment, and can be strictly below the truth.
   Dropping it under-prices silently.
3. **The metric needs no halo.** Only the updated cell's own M enters its
   own update, so it is 2 registers per thread and never crosses a tile
   boundary.
4. **The tracer change is the silent one.** Steepest descent must become
   the Riemannian tangent; the field stays correct and every polyline
   goes wrong. Measured on the corrugated-ramp analytic case (a cylinder,
   isometric to the plane by unrolling, so geodesics are straight in the
   unrolled frame and curved in the raster frame): the correct tracer
   deviates **0.71 cells** from the analytic geodesic, the old isotropic
   tracer **14.33 cells**. The suite keeps that as a mandatory CONTROL -
   a tracer test the old code also passes proves nothing.

   It bit once already. `RasterFIMAPI._paths_from_field` re-uploaded an
   older CACHED field for tracing and passed `q_device=(None, None)` with
   it, which is exactly how `trace_paths_gpu` decides between `-grad T`
   and `-M^-1 grad T` - so a pairwise multi-source call whose source
   repeated traced under the wrong tangent and returned the 14.33-cell
   polyline silently. The fix is structural rather than local: the metric
   depends only on the DEM, never on the field, so it is cached once
   (`_trace_metric`) and every trace under a DEM uses it; there is no
   code path left that can hand the tracer a DEM-solved field without its
   metric. Regression: exercise the CACHED path and compare against the
   analytic geodesic, so a silent fallback shows up as a distance error
   rather than as a well-formed wrong path.
5. **`cell_size` is the highest-probability silent bug.** The elevation
   gradient must be rise per METRE of run, not per cell. `RasterFIMAPI`
   recovers the cell size from `GradientLUTs.inv_horiz_m` and refuses a
   mismatch at construction, for free.
6. **Acceptance is decided from the LUT ARRAYS, not the option names** -
   the only check a `Callable` multiplier cannot bypass. A callable that
   happens to BE the identity is accepted, correctly: Tier A is defined
   by the metric produced, not by how the user spelled it.

### 13.6 Open / not measured

- ArcGIS vertical-factor comparison - harness exists, cases still
  "pending".
- Whether the derived discrete reachability limit
  `r >= sqrt((s_max/limit)^2 - 1)` is real (section 13.4 depends on it).
- Whether a direction-aware relaxation could close the 48.3 % grade-limit
  false-negative gap. A soft anisotropic penalty (inflating `q` on steep
  cells instead of masking them) is expressible as a metric and would
  keep contour traverses available, but it changes the objective the
  field reports and is bounded by the acuteness limit `|q| <= 2.19737`;
  not attempted. Cell masking has been fuzzed to exhaustion - three rules,
  all within 5 points of each other.
- A real 1 m DGM window with Tier A end to end (the section 13.2
  realistic case is synthetic terrain).
- Whether the P1 gap can be closed: remaining levers are a
  `__launch_bounds__` register cap, a half-precision metric, and skipping
  simplex candidates whose two neighbours are both already above the
  running best (not branch-free, so it needs measuring, not reasoning).

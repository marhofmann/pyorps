# Real-World Route-Planning Performance Plan (routes > 5 km)

**Status:** DRAFT for review — no code changed yet.
**Date:** 2026-08-07
**Scope:** CPU Dijkstra, CPU delta-stepping, GPU SSSP (V5/V4/V3), constrained CPU + GPU planners,
eikonal/FIM backend, rasterization + vector data path, and end-to-end orchestration.
**Evidence base:** 9 read-only code investigations + measured history in
`benchmarks/FUSED_KERNEL_FINDINGS.md`, `benchmarks/EIKONAL_FINDINGS.md`,
`benchmarks/results/*.json`, and the open-lever/eikonal/BMSSP plan docs.
Every number below is tagged **[M]** measured, **[X]** extrapolated from a measured anchor, or
**[E]** estimated from code structure. **No measurement above 4096² exists in this repo** — every
5000²/10000² figure is [X].

---

## 1. The diagnosis in one sentence

> At > 5 km, nine of the ten pipeline stages cost **O(window area)** while the deliverable is
> **O(route length)** — and the three stages that legitimately scale with area (solve, upload,
> state allocation) run on a window that is 2–13× larger than the corridor anyone actually wants
> searched.

A 5 km route on a 6000² window is 36 M cells producing a ~7 000-cell path. Today:

| Stage | Cost today | What it should be | Evidence |
|---|---|---|---|
| WFS fetch | O(data extent) | O(corridor) | `vector_loader.py:763-793`; PathFinder never derives a corridor bbox (`path_finder.py:273-275, 311`) |
| Rasterize metrics | O(K+2 geometry passes × extent) | O(1 pass × corridor) | `rasterizer.py:408-438` |
| `MetricStack.combine` | O(extent) **per objective variant** | O(corridor) | `metric_stack.py:435-497`; `window()` exists but is unused (`:348-369`) |
| Raster / DEM file read | O(whole file) | O(window) | `geo_dataset.py:177-186`, `path_finder.py:844` |
| Solve (unconstrained) | O(window) — legitimate | O(window) | — |
| Solve (constrained) | O(window × dirs × bins) → OOM | O(corridor × dirs × bins) | `_constrained_dijkstra.pyx:142-165` (verified `dense_limit = 500000000`) |
| GPU upload / download | O(window) **per pair** | O(1) upload + O(path) download | `sssp_gpu.py:2052-2057, 1644-1650` |
| Predecessor repair | O(window × dirs) **per query** | O(path × dirs) | `sssp_gpu.py:2260-2280`; API always requests pred (`raster_gpu_api.py:101`) |
| Path metrics | O(window log window) **per path** | O(path) | `_traversal_numba.py:624` — verified `np.sort(np.unique(raster))` |
| Metric evaluation | O(K × window) **per path** | O(K × path) | `metric_eval.py:191-244` |

Two supporting findings shape the priorities:

**Finding 2 — three fail-classes, not slow-classes.** At target scale these do not run slowly, they
do not run:
1. **Constrained routing above ~15.6 M cells** (≈ 3961² at R2 × 2 span bins) falls off the
   `dense_limit = 500 M` states cliff into a **sequential `unordered_map` Dijkstra**: ~150–250 B per
   touched label across four hash containers ⇒ ~69–115 GB at 6000² [E] ⇒ OOM or 25–60 min.
   The parallel variant is worse — it `calloc`s the full 27.6 GB unconditionally
   (`_constrained_delta.pyx:173`). Independently, per-(cell,dir) caches are allocated over the
   **full window** in every mode: 9 B × cells × dirs = **5.2 GB at 6000²/R2, 10.4 GB at R3**
   (`_constrained_dijkstra.pyx:264-265`).
2. **GPU solves at ≥ 8000²** run 1.3–3.5 s in a *single* cooperative kernel launch [X] against a
   Windows WDDM watchdog. No TDR handling exists anywhere in the repo. (The actual TdrDelay on this
   machine is unknown — project memory records a ~60 s observation, the WDDM default is 2 s. **Measure
   before building.**)
3. **Library backends** (networkit/igraph/rustworkx) materialise ~8N-edge lists preallocated at the
   theoretical maximum: 3.2 GB at 25 M cells, 12.8 GB at 100 M (`_traversal_numba.py:442-555`), and
   float32 lossless mode *advertises* them (`path_finder.py:356-359`) while excluding the cython
   kernels — so the documented float32 path steers users straight into OOM at target scale.

Plus four latent silent-suboptimality hazards that are **only reachable at > 5 km**:
per-thread relaxation buffers that silently drop already-committed improvements
(`_delta_stepping.pyx:297,408`; `_constrained_delta.pyx:216-227` and 13 further sites);
`ignore_max=False → max_value=0` making every **0-cost** cell impassable (`cython_api.py:28`);
and `/fp:fast` + `-ffast-math` applied to kernels that rely on `inf` sentinels
(`setup.py:52,65,94` — verified; `-ffast-math` implies `-ffinite-math-only` on GCC/Clang).

**Finding 3 — the dominant workload is a matrix, not a single pair.** `case_studies/substation_planning/`
runs meshed-network **source × target route matrices** over Hessen-scale rasters feeding a GA
(committed result files carry 19–6000+ route entries at 50 000/500 000 evaluation counts);
`case_study_mv_oberrhein.py:284-307` sweeps R0–R3 per connection as a product and already
**hand-shards R3 for memory**; `rhein_main_link.py:224` uses a **50 km buffer**. Yet every
multi-pair driver is a serial Python loop that re-uploads the identical raster per pair, and
`raster_gpu` does not deduplicate repeated sources (`raster_gpu_api.py:198-214`) although
`raster_fim` already does (`raster_fim_api.py:264-268`). **Single-pair kernel micro-gains are not
where the wall-clock is.**

---

## 2. Measured baseline and target end-state

### 2.1 What is measured today

Real 4096² planning raster (1 m ALKIS, 13 % forbidden, 4 pairs of 3.0–5.0 km),
`EIKONAL_FINDINGS.md §11.1`, all **[M]**:

| Backend | Scenario A (per pair) | Scenario B (+ isotropic DGM1 slope) |
|---|---|---|
| Cython R1 | 5.54 / 18.60 / 7.32 / 9.46 s | 6.16 – 31.33 s |
| Cython R2 (accuracy default) | 14.21 / 29.32 / 17.76 / 18.91 s | 14.46 – 32.04 s |
| GPU V5 | 0.344 / 0.466 / 0.464 / 0.305 s | 0.433 – 0.499 s |
| FIM order 1 | 0.288 / 0.296 / 0.503 / 0.244 s | 0.261 – 0.766 s |
| FIM order 2 | 9.4 – 11.9 s (guarded off on planning rasters) | — |

Constrained CPU, real R4 raster, 2.55 M cells, spc = 432: **148 s [M]** (project memory). No
constrained benchmark above 2.55 M cells exists anywhere.

**Caveat that undermines the whole ranking:** `benchmark_gpu_v5.py:24` and `benchmark_eikonal_gpu.py`
hard-code `STEPS_8` — **every GPU and FIM number above is R1**, while the production accuracy default
is R2 and users demonstrably run R3. R2 doubles and R3 quadruples per-pop relax work.

### 2.2 Target end-state (6000² = 36 M cells, R2)

| Scenario | Today | After this plan | Factor |
|---|---|---|---|
| **S1** single pair, GPU V5, incl. reporting | 1.6–3.9 s [X] | 0.7–1.3 s | ~2.5× |
| **S1** single pair, CPU cython (fallback / no-GPU) | 26–72 s [X] | 9–25 s | ~2.5–3× |
| **S1** single pair, CPU cython **with DEM/objective** | 26–72 s (serial only) | 5–10 s (parallel unlocked) | ~5× |
| **S2** 20-pair matrix, GPU | 33–79 s [X] | 4–10 s | ~7× |
| **S2** 19 × 19 meshed matrix (361 pairs, shared sources) | 361 solves | 19 solves | ~19× |
| **S3** constrained tower/span run | **OOM or 25–60+ min** | 1–3 min | fail → works |
| **S3** constrained + clearance | **hours** | 10–25 min | ~10× |
| **S4** cold start (WFS → raster → solve → report) | 1.5–6.5 min, 85–97 % data path | 25–60 s | ~4× |
| GUI cost-table edit → new raster | full re-burn (10s of s) | LUT gather (~50–100 ms) | ~100–500× |

None of these require a new solver algorithm. They come from making stages proportional to the
corridor and the path, from unlocking already-written code that is gated off, and from removing
per-pair repetition.

---

## 3. What is settled — do not re-propose

These were **measured** and lost. Re-proposing them wastes a cycle:

| Rejected | Measured outcome | Source |
|---|---|---|
| CPU bucket fusion / decreasing-Δ window / VGC | lost on all 10 scenarios; fusion-only 0.91–1.04×, fusion+window 0.26–0.79× | `FUSED_KERNEL_FINDINGS.md §1` |
| GPU tail-chase fusion (`fuse_depth`) | 0.59–1.16×, warp divergence | ibid. |
| V4 window > 4 as default | no all-win config in the launch sweep | ibid. |
| FIM coarse-to-fine **seeding** (P5) | **10× net loss** at every size | `EIKONAL_FINDINGS.md §12.1` |
| FIM order 2 on planning rasters | up to 10 % undershoot, ~10 s solves | ibid. |
| FIM 4×4 register block trace cache | exact parity, **zero** speedup | ibid. |
| faithful BMSSP (v1) | 3–4× slower than Dijkstra everywhere (external, arXiv 2511.03007) | survey |
| cuGraph SSSP | 5–25× slower than Cython | memory |
| ML-guided routing | no measured wall-clock win exists | survey |

**Killed during this analysis** (see §8 for reasons): ρ-stepping frontier cap, CCH/CRP pilot,
BMSSP-as-performance-item, bidirectional *delta-stepping* (duplicate), (dist,span) Pareto labels,
the "provable 2-bin span merge" (its dominance proof is **wrong**), fixing the four V4-ADDS defects,
uint16-native FIM slowness read, and multi-source GPU batching.

---

## 4. The corridor certificate (keystone — read before Phase 3)

Three dossier items independently proposed "corridor pruning". They are one infrastructure with
three consumers, and the exactness argument differs per consumer. Getting it right matters because
the project's brand is exact routing.

### 4.1 Lemma

Let `G` have non-negative weights, and let `L_s(v) ≤ d_G(s,v)`, `L_t(v) ≤ d_G(v,t)` be **lower**
bounds and `UB ≥ d_G(s,t)` an **upper** bound (the cost of any feasible s–t path). Then every vertex
`v` on some optimal path satisfies `L_s(v) + L_t(v) ≤ UB`.

*Proof:* for `v` on an optimal path, `d(s,v) + d(v,t) = d(s,t) ≤ UB`, and
`L_s(v) + L_t(v) ≤ d(s,v) + d(v,t)`. ∎

So excluding every cell with `L_s(v) + L_t(v) > UB` leaves at least one optimal path intact, and an
exact search on the reduced raster returns the true optimum. A loose `UB` widens the corridor —
degradation is graceful, never wrong.

### 4.2 Tier A — constrained from unconstrained (rigorous, no epsilon) ✅ ship this

For the constrained planner, the **unconstrained terrain field at the same resolution with the full
direction set** is a valid lower bound, because per edge:

```
w_con = (raster[u] + Σinter + raster[v]) · cost_factor · grad_penalty + angle_cost [+ tower costs]
w_unc = (raster[u] + Σinter + raster[v]) · cost_factor
```

with `grad_penalty ≥ 1`, `angle_cost ≥ 0`, `tower ≥ 0`; and the constrained planner's out-degree is
**angle-pruned to a subset** of the full direction set, so the unconstrained search has strictly more
moves available. Span/height feasibility only *adds* constraints. Therefore
`d_unc(s,v) ≤ d_con(s,v)` for every cell. Two unconstrained solves (≈ 1–2 s on V5 at 25 M cells [X])
in front of a 25–60 min constrained run is free.

**Required hardening:**
- One-time audit that the unconstrained edge model is byte-for-byte the terrain term of the
  constrained one (same `cost_factor`, same intermediate set, same exclude mask).
- If the field comes from **V5 (float32)**, apply a relative safety factor: `L := d_V5 × (1 − 1e-3)`
  and `UB := UB × (1 + 1e-3)`. Worst-case float32 accumulation over 10 k hops is ~2.5e-4 relative
  [E] — 1e-3 restores rigour and widens the corridor negligibly.
- `UB` from a genuinely **feasible constrained** solution. Two-stage self-certifying recipe:
  run once on a heuristically-narrow corridor → feasible route → that cost is a valid `UB` →
  build the certified corridor from it → re-run. If the answer is unchanged, it is *certified
  optimal*; if the certified corridor is a subset of the heuristic one, one run sufficed.

**Gain:** at 6000²/R2, a 700–1500 m corridor is ~7 M of 36 M cells ⇒ 7 M × 16 × 2 = **224 M states,
below the 500 M dense cliff** — the run flips from sequential hash-map mode to dense bucket mode.
That single flip is 15–50× [E] before any other constrained work lands.

### 4.3 Tier B — coarse min-pooled fields for unconstrained searches (rigorous with derived slack) 🔬

Min-pool the cost raster by `s` (coarse cell cost = **min** over its `s²` children), solve on the
coarse grid, then `L(v) = d_coarse(v)/(1 + β_k) − slack`, where `β_k` is the neighbourhood's
metrication upper bound (β_R2 = 2.79 %, β_R1 = 8.24 %) and `slack` covers block-boundary projection
(order `2·s·c_max_local`). Cost is ~`1/s²` of a fine solve. **Do not implement before the slack term
is derived on paper and stress-tested** — this is the one corridor tier with a real chance of being
subtly wrong.

### 4.4 ⚠️ Correction: R1 fields are **not** a valid lower bound for R2/R3

One vetted proposal offered "corridor from two discrete R1 solves, ε ≥ β_R1 = 8.24 %, provably
contains the R2/R3 optimum". **This is false.** R1's direction set is a *subset* of R2's, so
`d_R2 ≤ d_R1` pointwise — R1 distances are **upper** bounds, not lower ones. Worse, on obstacle
rasters the gap is unbounded: a knight-move gap in a wall is traversable at R2 while `d_R1 = ∞`
(this is precisely the recorded wall-with-gap geometry). The metrication bounds β are worst-direction
elongation results for **uniform** cost fields and do not transfer to heterogeneous rasters with
exclusions. Coarser *neighbourhoods* can never give lower bounds; only coarser *grids* (Tier B) or
cheaper *cost models* (Tier A) can.

### 4.5 Tier C — FIM corridors (fast, **approximate**, must be labelled) ⚠️

FIM solves a different (continuous) problem and its route costs sit in a measured envelope of
**−1.1 % to +3.7 % versus R2** on real planning rasters, with no certified worst-case bound on
piecewise-constant surfaces. A FIM-derived corridor is therefore a **heuristic**. It is excellent for
exploration, ensembles and generating a `UB`, and must never be used to claim an exact route.
Validation of any FIM-corridor work must include the σ = 4 cell-noise class (FIM's measured worst
case, ~2.3 % above R2).

### 4.6 The *other* corridor — geometric, and already the contract

Distinct from the cost-based corridor: `search_space_buffer_m` already restricts every search to a
buffered line/hull (`handler.py:133-160`). Applying that **earlier** (before WFS fetch, before
burning, before file read, per pair instead of per hull) is *not* a new approximation — it is the
documented contract, applied where it pays. That is Phase 2 and needs no sign-off.

**But it has an uncertified hole worth closing** (§5.5, item 5.3): the auto-buffer is a terrain
sampling heuristic clamped to [200, 4000] m and **nothing checks whether the returned optimum was
clipped by it**. A post-solve O(path) test — did the path touch the outer ring of the buffer? — plus
geometric buffer doubling on a hit, converts the product's central approximation into a certified one
for a few lines of code.

---

## 5. The plan

Effort key: **S** ≤ 1 day, **M** 2–4 days, **L** 1–2 weeks, **XL** > 2 weeks.
Exactness key: **E** exact/bit-identical, **EC** exact-with-certificate, **A** approximate (must be
labelled), **C** correctness fix.

### Phase 0 — Gates: measure the target scale, close the silent-failure classes (≈ 4 days)

Nothing later can be *accepted* without these; the correctness items should land first regardless.

| # | Item | Where | Kind | Effort |
|---|---|---|---|---|
| 0.1 | **Scale benchmark above 4096²** — mosaic the real raster to 5000²/8000²/10000², run V5 / FIM / Cython-R1 once each with wall-clock, `cp.get_default_memory_pool().used_bytes()` high-water, peak RSS, and **TDR incident logging**. Settles: the TDR threshold (2 s vs 60 s conflict), the V5 22 B/cell model, the rendezvous-rescan share, the FIM pass floor. | `benchmarks/` | gate | S |
| 0.2 | **R2/R3 rows in the GPU + FIM benchmarks** — both harnesses hard-code `STEPS_8`; the entire backend ranking rests on R1 numbers while users run R2/R3. | `benchmark_gpu_v5.py:24`, `benchmark_eikonal_gpu.py` | gate | S |
| 0.3 | **Matrix benchmark** — 19 × 19 meshed source/target matrix on a ≥ 5000² window mirroring `find_paths_for_meshed_network`. The dominant workload currently has *zero* benchmark representation, so its biggest levers have nothing to be accepted against. | `benchmarks/` | gate | S |
| 0.4 | **Cold-start pipeline benchmark** + fix runtime accounting (graph build is timed inside `shortest_path`; cython `total` omits it entirely). The "data path is 85–97 %" claim is currently unmeasured. | `path_finder.py:1334-1344, 1403-1412` | gate | M |
| 0.5 | **Exactness referee** — re-walk every returned path in float64 over the raster (O(path), trivial) and cross-compare backends under a written tolerance policy for 5–10 km paths. Makes "exact" testable at 25–100 M cells instead of asserted from 2000² runs. Also re-tests the outstanding 0.04 % Dijkstra-vs-delta anomaly. | new harness | gate | S |
| 0.6 | **Silent relaxation drops** — replace `if count < capacity` (no `else`) with the fused kernel's guarded chunk/rollover protocol. Reachable at > 5 km: one bucket can exceed the 131 072-entry per-thread buffers. | `_delta_stepping.pyx:297,408`; `_constrained_delta.pyx:216-227` + 13 sites | C | M |
| 0.7 | **`ignore_max=False` 0-sentinel** — `max_value=0` currently makes every 0-cost cell impassable. CPU twin of the GPU bug fixed in `40c0f3a`; most plausible root of the recorded wall-with-gap report. Add an `ignore_max × {0-cost cells, 65535 walls}` config-matrix regression test. | `cython_api.py:28`, `_raster_context.pyx:346-349` | C | S |
| 0.8 | **fast-math audit** — `/fp:fast` and `-ffast-math` are applied to every extension including kernels that compare against `inf` sentinels; `-ffast-math` implies `-ffinite-math-only`, making those comparisons formally UB on all non-Windows builds. Switch to `/fp:precise` + `-fno-finite-math-only`, measure the (likely small) speed delta, add a both-ways CI parity job. | `setup.py:52,65,94` | C | S |
| 0.9 | **Build provenance** — embed a source hash in each compiled module and assert it in a test; delete the orphaned `_traversal.pyd` that shadows `_traversal_numba.py`. A stale `.pyd` has already silently invalidated a measurement once, and this plan is entirely measurement-gated. | `setup.py`, new test | C | S |
| 0.10 | **Library-backend size guard + dead-code fix** — refuse networkit/igraph/rustworkx above ~5 M cells with an actionable message; `graph_library_api.py:152` imports `construct_edges_gpu`, which does not exist, and the `ImportError` is swallowed as "cupy not installed". | `path_finder.py:94-139`, `graph_library_api.py:143-161` | C | S |
| 0.11 | **Dual-metric reporting bug** — 3D/gradient routing reports 2D terrain cost. | `path_finder.py:1490` (plan `2026-06-09-dual-metric-cost-model.md`) | C | S |
| 0.12 | **Thermal/variance protocol** — record `nvidia-smi` clocks + temperature during runs; report first-run vs steady state. Long persistent kernels at 10000² will throttle, and medians-of-3 cannot distinguish a real 1.2× lever from throttle noise. | harness docs | gate | S |

### Phase 1 — O(window) → O(path): the free sweep (≈ 1 week)

Highest gain-to-risk in the plan. All exact, all touching every scenario and every backend.

| # | Item | Where | Gain | Kind | Effort |
|---|---|---|---|---|---|
| 1.1 | **O(path) reporting** — replace `np.sort(np.unique(raster))` per path with a 65 536-bin count cached per raster (or derive categories from path cells only); in `evaluate_path_metrics`, gather layer values at the ~42 k path+intermediate cells instead of `np.stack`-copying K full-window layers plus a full-window weighted surface. | `_traversal_numba.py:624`, `metric_eval.py:191-244` | **0.4–1.3 s per path → sub-ms**; 12–38 s per 20-path job [X] | E | M |
| 1.2 | **Persistent GPU session** — upload raster/DEM/step tables once, compute auto-delta once, keep `dist`/`pred`/arena allocated across queries; `astype(copy=False)`. Free device buffers when the API is dropped (6 GB headroom rule). | `sssp_gpu.py:2035-2062`, `raster_gpu_api.py:83-105` | 70–200 ms/query at 25 M, 360–1100 ms at 100 M [X] | E | M |
| 1.3 | **On-device path extraction** — a one-thread-per-target `pred` walk writing a compact chain; download only chains + `dist[targets]` instead of the full `dist`+`pred` (763 MiB at 10000²) plus a host inf-rewrite pass. Keep the full-field mode for cost-surface consumers. | `sssp_gpu.py:1644-1650`, `raster_gpu_api.py:107-144` | 55–140 ms/query at 25 M, 200–550 ms at 100 M [X] | E | M |
| 1.4 | **Path-local predecessor repair** — `v5_repair_pred` re-evaluates all incoming edges of *all* cells on **every** query (the API always requests pred); validate and repair only the ~10 k links on requested paths. | `sssp_gpu.py:2260-2280, 1293-1361` | 10–25 % of every GPU solve [E] | E | M |
| 1.5 | **Same-source dedup + target batching for `raster_gpu`** — mirror what `raster_fim` already does. For a 19 × 19 meshed matrix: **361 solves → 19**. | `raster_gpu_api.py:198-227` | ~19× on the matrix workload's solver time | E | S |
| 1.6 | **Relax the reversed-single-solve guard** — disabled whenever `dem_data is not None`, but a bare DEM never reaches the kernel and the gradient term `fabsf(dem[v]−dem[u])` is direction-symmetric. k-source→1-target drops from k solves to 1. Encode the symmetry assumption in a test (directional slope, if it ever lands, must re-tighten this). | `raster_gpu_api.py:170` | (k−1)/k on star topologies | E | S |
| 1.7 | **Cython context/solver caching + O(1) target bitmap** — every public wrapper rebuilds `RasterContext` (O(N) exclude-mask scan + a `psutil` call `DijkstraSolver` never uses) and reallocates 13 B/cell per call; and the multi-target check linearly scans the target list at **every settled node**. | `_dijkstra.pyx:723-848, 400-403`; `_raster_context.pyx:430-439`; `cython_api.py:111-130` | 100–300 ms/call; ~2× for T ≥ 100 targets | E | S |
| 1.8 | **`margin` default 1.1 → 1.00001** — margin > 1 buys no exactness (once `bucket_start ≥ d(target)` nothing unsettled can improve it); 1.1 relaxes a ~21 % larger annulus on every single-pair delta-stepping run. Fix the misleading docstrings and the pairwise-entry inconsistency in the same commit. | `cython_api.py:128`, `_delta_stepping.pyx:463-468` | ~15–21 % of every delta-stepping run | E | S |
| 1.9 | **Reusable per-raster workspace (CPU delta)** — cache exclude mask, P1.3 max-cost stats and `SystemLimits`; ~450 MB of re-initialisation traffic per call at 25 M cells, paid per pair in ensembles/matrices. Key on array identity/version (the GUI edits rasters). | `_delta_stepping.pyx:1631,1666,1689-1701` | 30–60 ms/call × K | E | S–M |
| 1.10 | **Lazy objective restore** in `find_route_ensemble` — the `finally` block triggers a full combine + handler rebuild even if the user never routes again. | `path_finder.py:980-982` | one full combine per ensemble | E | S |
| 1.11 | **Right-size the V5 ring arena** — `cap = max(4096, 3n/32)` per ring = 12 of V5's 22 B/cell, while real frontiers are ~O(perimeter). Overflow self-heal is already lossless, so frontier-proportional sizing is safe and moves the 6 GB ceiling from ~15 300² to well beyond. *(Ordering: only because the lossless rewind already exists — never do this on a lossy-overflow kernel.)* | `sssp_gpu.py:2220-2224` | ~0.9 GB freed at 100 M cells | E | S |

### Phase 2 — Corridor-first data path (≈ 1 week)

The single largest **end-to-end** lever for cold starts, where the data path is 85–97 % of wall-clock.

| # | Item | Where | Gain | Kind | Effort |
|---|---|---|---|---|---|
| 2.1 | **Copy-based masking** *(prerequisite for everything cached)* — `apply_geometry_mask` writes the sentinel **through a view into the parent array**, permanently mutating the loaded raster. This silently blocks every caching opportunity below. | `handler.py:188-197, 404-423` | enabler | C/E | S |
| 2.2 | **Derive the corridor bbox/mask before loading and burning** — PathFinder knows the coordinates and buffer at `__init__` (`:273-275`) but passes only *user-supplied* bbox/mask to the dataset (`:311`) and calls `rasterize()` with no `bounding_box` (`:763`). Pass the buffered-line bbox as the WFS/vector bbox and its polygon as the mask. Fall back to `max_buffer = 4000 m` when `search_space_buffer_m=None` (the estimator needs raster values — chicken-and-egg). | `path_finder.py:273-321, 760-779` | fetch + burn ∝ corridor: **2.1× at 8×8 km extent, 13× at 20×20 km**; minutes of network time | E | M |
| 2.3 | **Windowed raster/DEM file reads** — `src.read()` loads whole GeoTIFFs; a state-wide 40000² raster is a 3.2 GB read for 61 MB of need (today: unusable). | `geo_dataset.py:177-186`, `path_finder.py:844` | hard-fail → works; 1–5 s typical | E | S |
| 2.4 | **Single index-burn + LUT gather for `rasterize_metrics`** — K+2 separate full-extent rasterize passes re-convert every shapely geometry to GDAL shapes each time. Burn **one** uint32 sorted-row-index band, derive cost/metric/category bands by gathers. Painting order and the winner-per-cell invariant are preserved exactly (same sorted sequence, same replace semantics). | `rasterizer.py:408-438` | **20–60 s → 5–12 s** at F ≈ 300 k, K = 4 [E] | E | M |
| 2.5 | **Class-id band + LUT re-cost for the legacy path** — recombine-without-reburn exists only in the metric pipeline; the legacy uint16 path fully re-burns on every cost-table change. **Mandatory invalidation rule:** legacy sort is by *cost value*, so the cache must invalidate whenever the cost **order** of classes changes, or results change silently. | `rasterizer.py:211-333` | GUI re-cost **100–500×** | E | S–M |
| 2.6 | **Window-first combine** — scalarize only the search window per objective variant; `MetricStack.window()` already exists and returns views. Documented semantic: the uint16 quantization scale derives from the windowed max (usually *finer* resolution), so results are exact w.r.t. the searched window but not bit-identical to today. | `path_finder.py:438-459`, `metric_stack.py:348-369, 435-497` | 5–30× per-variant combine cost + transients | E | S |
| 2.7 | **One sorted `rasterize(out=)` for overlays** + drop the redundant bbox burn + drop the unconditional `add_layer` copy. | `rasterizer.py:698-728, 305-311`; `metric_stack.py:235` | overlays 5–10× | E | S |
| 2.8 | **WFS loader** — continuous scheduling (chunks currently wait for the next batch round), WFS 2.0 `COUNT`/`STARTINDEX` paging with `resultType=hits` instead of the magic-number limit heuristic (**a server with a non-magic limit silently truncates data — a correctness bug**), `gml_id` dedup instead of geometry-WKB hashing, bytes-parse instead of temp files. Keep subdivision as fallback (16 state endpoints vary) and `max_workers = 4` (headroom rule). | `vector_loader.py:763-793, 407, 811, 989-999, 1024-1027` | 1.5–2.5× on WFS wall-clock + a correctness fix | E/C | M |

### Phase 3 — Make constrained routing work at > 5 km (≈ 2 weeks)

This is the fail-class. Sequencing matters: **the corridor comes first and re-baselines everything
after it.**

| # | Item | Where | Gain | Kind | Effort |
|---|---|---|---|---|---|
| 3.1 | **Corridor certificate, Tier A** (§4.2) — two unconstrained solves + a feasible `UB` + `exclude_mask` + `initial_best_dist`. Both kernel parameters already exist and **the finder never passes either** (verified: zero references in `constrained_path_finder.py`). Includes the one-time edge-model audit and the float32 safety factor. | `constrained_path_finder.py:518-642`; kernels `_constrained_dijkstra.pyx:54-55, 333, 370` | **flips 800 M states → 224 M, below the dense cliff: 15–50×** [E] | EC 🔓 | M |
| 3.2 | **Corridor-bbox allocation** — allocate the state array *and* the per-(cell,dir) `icache`/`grad_penalty` caches over the corridor bounding box only. Today they are full-window in every mode including "sparse": **5.2 GB at 6000²/R2, 10.4 GB at R3** (verified `calloc` at `_constrained_dijkstra.pyx:264-265`). | `_constrained_dijkstra.pyx:263-296` + all variants | 3–10× memory; makes R3 possible | E | M |
| 3.3 | **Hoist clearance + tower-terrain out of the neighbour loop** — `_check_span_clearance` walks the whole span (~300 evaluations for a 300 m span) and is called **inside** the neighbour loop although its arguments do not depend on the neighbour; the "quick skip" needs `height − sag − clearance > 50 m`, which never fires for 25–40 m towers. Same for the uniform-mode tower slope multiplier (depends only on the cell). | `_constrained_delta.pyx:820-824, 841-845, 1295-1305, 2339-2347`; `_constrained_context.pyx:556-728, 149-210` | **5–13× off the dominant clearance cost centre**, bit-identical | E | S |
| 3.4 | **Precompute tower slope multiplier + base cost rasters** — uniform mode becomes 2 loads instead of 4 DEM reads + `expf` per tower candidate; exact-area mode caches footprint sums per (cell, bisector). Allocate over the corridor bbox. | `_constrained_context.pyx:149-210, 75-146` | 10–25 % uniform; 2–5× on tower-heavy exact-area profiles | E | S–M |
| 3.5 | **Paged/tiled dense state store** — replaces the sequential `unordered_map` mode (~150–250 B/label, 100–300 ns/op) with tile-paged dense arrays (O(1) shift+mask, 10–30 ns/op), allocated on first touch. **Default to a 14 B/state record with `double` dist**; the 10 B float32 variant is a precision trade the lossless brand must opt into explicitly. *Re-baseline after 3.1* — if the corridor already drops states below the cliff, this only matters for R3/10000². | `_constrained_dijkstra.pyx:597-953` + variants | 5–15× on whatever the corridor cannot rescue | E | L |
| 3.6 | **Parallel constrained by default** — after 0.6 (overflow safety), add the missing `tower_cost_raster` parameter to `constrained_delta_stepping_2d`, give it the same dense/paged fallback as the sequential kernel (it currently `calloc`s unconditionally), expose `num_threads` (it defaults to the whole machine — headroom rule), then flip the default. | `_constrained_delta.pyx:173, 212`; `constrained_path_finder.py:31, 619-625` | 3–5× on 8 threads (gate: no constrained thread-scaling data exists — measure first) | E | M |
| 3.7 | **Bimodal delta** — one scalar delta cannot classify terrain edges (~250–1700) and tower placements (~2 000–18 M) together; both observed GPU failure regimes came from this. Derive delta from the terrain population only and treat tower edges as unconditionally heavy. Applies to CPU and to any surviving GPU variant. | `_constrained_*.pyx`, `constrained_sssp_gpu_v4.py:140-186` | dispatches both pathological regimes | E | S |
| 3.8 | **Retire dead constrained-GPU modes** — V1 (never fits: 9.6–32 GB), V2-sparse-hash (crashes on real data), **V2-managed (would page 19–38 GB through a 6 GB card and can freeze the machine — this violates the standing system-resources rule)**, V2-block (8 GB + unresolved 0-paths bug). Keep `constrained_persistent.cu`'s warp-cooperative clearance/area device functions as a library. | `constrained_sssp_gpu*.py` | −2500 lines of dead-end surface; enforces the headroom rule | hygiene | S |

### Phase 4 — CPU kernel throughput and feature parity (≈ 1 week)

| # | Item | Where | Gain | Kind | Effort |
|---|---|---|---|---|---|
| 4.1 | **Gradient-LUT / DEM support in delta-stepping** — the parallel kernel raises `NotImplementedError` on `gradient_luts`, and cython is excluded from `FLOAT_BACKENDS`, so **every multi-objective/DEM run — the exact workload this plan targets — falls back to serial Dijkstra**. Mechanical port of an existing per-edge term; must extend the P1.3 circular-buffer span validation by the max LUT multiplier. | `cython_api.py:76-80`, `_delta_stepping.pyx:362-419`, `path_finder.py:356-359` | **serial 26–72 s → parallel 5–10 s** per pair [X] | E | M |
| 4.2 | **Cached intermediates in `DijkstraSolver`** — the hot loop calls the *allocating* `check_path`, building a fresh C++ vector per relaxation for 12/16 directions at R2 and 28/32 at R3, while `check_path_cached` + `ctx.cached_steps` already exist and are used by the delta kernels. Add a double-accumulating variant to stay bit-identical. | `_dijkstra.pyx:285-288, 430-433`; `_raster_context.pyx:160, 237` | **1.5–2.5× at R2/R3** [E] | E | S |
| 4.3 | **Lower-bound prefilter before path validation** — `check_path` runs *before* the `new_dist < dist[neighbor]` test, so ~70–85 % of relaxations pay full validation to be discarded. `(c_u + c_v)·factor` is an admissible lower bound (intermediates are uint16 ≥ 0). With gradients, use the precomputed LUT minimum, or skip the prefilter under `use_grad` for v1. | `_dijkstra.pyx:283-322` | +1.2–1.4× on top of 4.2 (stacked ≈ 2–3× total) | E | S |
| 4.4 | **Epoch/touched-list reset** — three O(N) array fills per query (and per source in multi-source runs) regardless of how small the settled region was. | `_dijkstra.pyx:155-160` | 150–300 ms/query at 25 M; up to 5–10× on many-source batches | E | M |
| 4.5 | **Settle-loop micro-cleanups** (bundled with 4.2 only) — reuse the register-resident `current_dist`, per-direction linear offsets instead of `ravel_index`, hoist `directions.size()`, avoid the div/mod unravel per pop. | `_dijkstra.pyx:318` etc. | 5–10 % combined | E | S |

### Phase 5 — Scale-up enablers and batch throughput (≈ 1 week)

| # | Item | Where | Gain | Kind | Effort |
|---|---|---|---|---|---|
| 5.1 | **Chunked resumable V5 launches (TDR)** — all algorithm state already lives in device memory; add a "max rendezvous per launch" exit and a host relaunch loop, with correct barrier re-initialisation between launches. Also lets the host yield the GPU between chunks (headroom rule) and poll for cancellation. **Gated on 0.1's measured TDR threshold** — if it is really ~60 s, demote to 20000²-insurance. | `sssp_gpu.py:2247-2258, 1131-1201` | enabler: ≥ 8000² solves survive at all | E | M |
| 5.2 | **Per-pair corridor cropping for multi-pair runs** — every pair is solved on the convex hull of *all* endpoints buffered; for scattered substation pairs that is 10–40× the pair's own corridor. Applying the same documented `search_space_buffer_m` contract per pair changes nothing philosophically. Node-index remapping is the real work; keep hull behaviour as fallback. | `handler.py:133-160`, `raster_gpu_api.py:198-214`, `cython_api.py:154-176` | **10–40× per pair on scattered matrices** | E | L |
| 5.3 | **Buffer-clipping certificate** (§4.6) — post-solve O(path) check whether the optimum touched the buffer's outer ring; warn and/or auto-rerun with a doubled buffer until untouched. Geometric doubling terminates fast and costs < 2× the final window. Turns the product's central approximation into a certified one. | `handler.py:242-337`, `path_finder.py` | closes an exactness hole under the brand | EC | S |
| 5.4 | **Reachability guard** — an 8-connected CCL over the exclude mask (≈ 100–200 ms once at 25 M cells), cached per raster, checked before dispatch. Today a disconnected pair makes *every* backend exhaust the entire reachable set before reporting failure — the system's worst-case latency — and in a matrix one dead target poisons every source's solve. 8-connectivity is provably sufficient for all R_k here, because R2/R3 moves validate every intermediate cell against the mask. | new, `_raster_context.pyx` | worst-case latency guard | E | S |
| 5.5 | **Pair-level pipelining** — overlap host post-processing of pair *k* (reconstruction, metrics, export ≈ 0.5 s) with the GPU solve of pair *k+1* (CuPy calls release the GIL). Free ~1.5–2× on matrix throughput, multiplicative with every other item. Cap workers per the headroom rule. | multi-pair dispatch | ~1.5–2× on matrices | E | M |
| 5.6 | **Pinned host memory + async streams for V5 transfers** — `sssp_gpu.py` contains no pinned/stream/graph machinery at all (the FIM backend already uses non-blocking streams + CUDA-graph capture). Whatever transfer volume survives 1.2/1.3 runs at pageable bandwidth (~half of pinned) and serialises with host work. | `sssp_gpu.py:2035-2062, 1644-1650` | ~30–50 % of residual transfer time | E | S |
| 5.7 | **Lean device-only flow for FIM multi-target/multi-source** — the `download=False` treatment exists only for single pairs; other paths pay a full-field D2H (~105 ms at 10000²) plus host mask allocation **per solve**. | `raster_fim_api.py:155-198, 236-274` | 30–50 % of multi-target wall-clock at 10000² | E(in an A backend) | S |
| 5.8 | **FIM multi-target amortisation as a workflow tier** — one solve serves all targets of a source; a 50-target job at 4096² is ~1.6 s versus ~20 s on V5 and ~15 min on Cython R2 [X]. Already built; needs dispatch plumbing and honest labelling (approximate tier; binding runs stay discrete). | `raster_fim_api.py:236-274` | ~12× over V5 on many-target jobs | A | S |
| 5.9 | **FIM ensemble warm start (pure rescale only)** — `c' = α·c ⇒ T' = α·T` exactly: **zero solves** for pure re-weighting. The general supersolution seed (`T_init = T_old · max(c'/c)`) is provable but sits one step from the P5 grave — gate it on measurement, keep cold start as default. | `eikonal_gpu.py`, ensemble path | −(K−1) solves for pure rescale | E/A | S |

### Phase 6 — Gated pilots and research (do not start before Phase 3 lands)

| # | Item | Verdict | Effort |
|---|---|---|---|
| 6.1 | **FIM increment 3 — anisotropic (directional slope)** via a semi-Lagrangian one-ring update for the Finsler metric. **The single biggest capability gap**: today `raster_fim` *raises* on DEM/gradient_luts, so the fastest backend is excluded from all slope-aware work, and the isotropic workaround mis-prices a 20 % hillside contour route by **+12.0 %**. Solve cost estimated 60 → 200–600 ms at 4096² (still ≥ V5-class), but the +0.76 % isotropic accuracy headline does **not** transfer — a full re-validation campaign is mandatory. Remains **approximate**; never sell as exact. | KEEP as its own gated program | XL |
| 6.2 | **Corridor Tier B** (coarse min-pooled lower-bound fields) — derive the slack term on paper first, then validate corridor-restricted optima against full-raster optima on the real benchmark pairs. Would give CPU-only users the corridor win. | research, gated | M |
| ~~6.3~~ | ~~Exact tighter FIM early-exit bound~~ | **WITHDRAWN 2026-08-07** (§7.1: no heuristics anywhere) | — |
| ~~6.4~~ | ~~Bidirectional Dijkstra pilot~~ | **WITHDRAWN 2026-08-07** (§7.1) | — |
| ~~6.5~~ | ~~Goal-directed bucket priority on V5~~ | **WITHDRAWN 2026-08-07** (§7.1) | — |
| 6.6 | **Constrained GPU V6** — port the proven V5 async protocol (rolling 32-bucket rings, self-claiming chunks, **lossless overflow rewind**) onto direct-indexed 6 B/slot block storage. The four V4 hang causes are diagnosed and concrete (frozen `head_logical=0` at `constrained_adds.cu:170-175`; the non-atomic `reset_bucket` racing enqueues; the WCC-prefix assumption; the 256-item/pass manager serialisation). **But this is attempt #5, and Phase 3's corridor may shrink the problem to where dense CPU already suffices.** Pilot only if corridor + CPU still misses the latency target. | DOWNGRADE, sequenced last | L |
| 6.7 | **Resolution-sensitivity study** — solve the same corridor at 1/2/5/10 m and report route-cost and geometry deltas. If 5 m costs < 1 % route error, the entire 50 km mega-corridor class (`rhein_main_link.py:224`, `search_space_buffer_m=50_000`) collapses into the already-analysed 10000² regime and the answer is a **documented resolution policy** instead of out-of-core machinery. **Do this before contemplating banded out-of-core solving.** | KEEP as a cheap decision gate | S |
| 6.8 | **uint16 quantization decision study** — measure the *route-level* cost of uint16 quantization (uint16-optimal route re-priced in float32 vs the float32-optimal route). Decides whether float32 lossless mode is ever needed — it currently forfeits the parallel CPU backend entirely and doubles GPU raster bytes. | KEEP as a decision gate | S |
| 6.9 | **uint8 class + LUT cost raster** — halves the dominant scattered-read stream and would make float32 semantics cost 1 B/cell instead of 4. Bounded upside; measure. | DOWNGRADE, pilot | S–M |
| 6.10 | Single-scan relaxation (fold heavy phase); parallel sharded merge **or** Wasp-style async CPU SSSP (**profile the actual thread-0 serial share first — build at most one**); static delta auto-tuning; Dial/calendar queue for sequential Dijkstra; V5 dirty-tile rendezvous; the 512-resident-thread root cause (timeboxed spike); parallel tiled burning; cell-side node placement. *(LPA*/D* Lite withdrawn 2026-08-07 — §7.1.)* | DOWNGRADE — see §8 for the gate on each | various |

*All items formerly marked 🔓 were resolved on 2026-08-07: the corridor certificate is approved, and
every goal-directed item is withdrawn. See §7.1.*

---

## 6. Dependency and sequencing notes

- **2.1 (copy-based masking) blocks 2.5, 1.9 and the correctness of every cache** — the current
  write-through-view mutation is why the metric pipeline had to keep pristine full-extent float
  copies as a workaround.
- **0.6 (overflow safety) blocks 3.6 and 1.11-style right-sizing** — never shrink a buffer whose
  overflow silently drops committed work.
- **3.1 (corridor) re-baselines 3.5 and 6.6** — measure after the corridor lands before sizing either.
- **4.1 before 4.2/4.3 in priority order**: unlocking the parallel kernel for DEM/objective runs is a
  ~5× step for the target workload; the serial-kernel work is a ~2.5× step on a fallback tier.
- **0.1/0.2 gate the entire GPU workstream** — the backend ranking currently rests on R1 numbers at
  ≤ 4096².
- Gains are **not multiplicative when they attack the same cycles**: 4.2 + 4.3 stack to ≈ 2–3×, not
  3.5×; the corridor partially obsoletes paging and constrained-GPU.

---

## 7. Decisions

### 7.1 Settled 2026-08-07

1. **Corridor certificate: APPROVED.** Tier A (§4.2) is cleared for Phase 3. It never enters the
   constrained search loop — it is a preprocessing mask plus an upper bound, with the exactness proof
   in §4.2 and no heuristic function anywhere. Both kernel hooks (`exclude_mask`,
   `initial_best_dist`) already exist and are documented for exactly this use.
2. **No heuristics anywhere — the rule extends to ALL backends, not just the constrained planner.**
   Consequently the following are **withdrawn from the plan** and must not be re-proposed:
   - 6.3 exact tighter FIM early-exit bound (`min_active + c_min · dist(tile, target)`) — exact
     arithmetic, but structurally a distance-to-goal bound;
   - 6.4 bidirectional Dijkstra pilot;
   - 6.5 goal-directed bucket priority on V5;
   - LPA*/D* Lite incremental replanning (was inside 6.10).

   *Unaffected* (these are not heuristics): the **existing** FIM targeted early exit, which is a
   monotonicity stopping rule of the same class as Dijkstra's settle-the-target criterion; the
   delta-stepping `margin` (item 1.8 tightens it *toward* exactness); and the approved corridor
   certificate, which is bound-based preprocessing.

### 7.2 Still open

3. **float32 lossless mode** — 6.8 decides whether to invest in a float-capable Cython raster kernel
   or to formally park float32 mode as GPU-only. Cheap to measure; expensive to guess wrong.
4. **Constrained span-payload exactness** — the (dist, span) Pareto-label fix is estimated
   1.5–3× *slower* and repairs a defect seen on 1/300 adversarial 0-cost seeds (real rasters with
   min cost ≥ 1 matched the exact reference 300/300). I have it as an exactness-brand backlog item,
   not a performance item. Confirm that ranking.
5. **Resolution policy for the 50 km class** (6.7) — if a coarser resolution is acceptable there,
   several XL items never need to exist.

---

## 8. Considered and **not** recommended (with reasons)

Recorded so these are not re-proposed:

- **"Corridor from two R1 field solves is provably exact for R2/R3"** — **false**, see §4.4. R1 ⊂ R2
  ⇒ `d_R2 ≤ d_R1`, so R1 distances are upper bounds; and on obstacle rasters `d_R1` can be ∞ where
  `d_R2` is finite.
- **"Merge all span bins ≥ min_span into one (2-bin quantization)"** — the dominance argument
  requires *both* coordinates smaller. Crossing pairs (smaller dist, larger span) vs (larger dist,
  smaller span) retain crossing utility above `min_span` via the `max_span` cap, so merging drops a
  label that today survives. Strictly worse exactness for 1.5–2.5× on wide-span profiles only.
- **Multi-source GPU batching** (pack B independent SSSPs into one launch) — appealing given the
  under-occupied GPU at small frontiers, but V5's gain over V4 already collapses to 1.02× at random
  3000², i.e. **one source already saturates the 14 SMs at target scale**. Batching would only help
  below the sizes this plan targets.
- **Fixing the four V4-ADDS defects** — after all four, the 256-items-per-manager-pass serialisation
  remains, capping throughput an order of magnitude below the V5-protocol design it competes with.
  The diagnoses were the value; keep the write-up, skip the fixes.
- **BMSSP as a performance item** — externally measured 3–4× slower; the project's own addendum
  predicts sequential v2 lands 1–3× slower. Route to the science track with its pre-registered kill
  criterion.
- **CCH/CRP pilot** — the docs' own expected outcome is a documented do-not-build (break-even
  ~1700 queries, region-scale cost edits force recustomization, and FIM already serves repeated
  queries at ~0.1 s/field).
- **ρ-stepping frontier cap** — same family as the three levers the 10-scenario campaign already
  buried, with a predicted null result.
- **uint16-native FIM slowness read** — zero expected wall-clock gain; reintroduces dual-variant
  complexity for a VRAM ceiling move (21 900² → 24 800²) beyond any realistic scale.
- **Rotating the raster so the corridor is axis-aligned** (would cut ~2.3× off every window-proportional
  stage) — requires resampling a class raster with hard boundaries, which changes the answer. The
  exact alternative, compacting to in-corridor cells with an index map, destroys the 2D neighbour
  arithmetic every kernel relies on. Keep compaction for the **constrained** planner only (Phase 3),
  where memory is O(cells × dirs × bins) and the reindex pays for itself.

---

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Corridor certificate has a hole (bad `UB`, edge-model mismatch) | One-time edge-model audit; float32 safety factor; two-stage self-certifying run (§4.2); degenerate case is a full-raster corridor, i.e. graceful |
| Paged state store corrupts silently under `boundscheck=False` (span-alias precedent) | 300-seed adversarial sweep + new large-scale spot checks + the 0.5 exactness referee on every accepted lever |
| V5 chunked relaunch breaks barrier state | 30-rep stress regression (the pattern that caught the two async bugs) |
| Class-LUT re-cost changes overlap winners | Mandatory invalidation whenever the cost *order* of classes changes |
| Window-first combine changes the quantization scale | Documented; scale is already per-combine and recorded in `objective_spec` |
| Benchmarks measure a stale `.pyd` | 0.9 build provenance |
| GPU work saturates the shared machine | Single resident device session, freed on API drop; `nvidia-smi` pre-check before every GPU run; chunked launches yield between chunks |
| Plan is too large to land | Phases 0–2 alone deliver ~2.5× single-pair, ~7× matrix and ~4× cold-start with zero algorithmic risk |

---

## 10. Known measurement gaps (all become Phase 0 deliverables)

- No measurement above **4096²** anywhere in the repo.
- No **R2/R3** GPU or FIM measurement — the production accuracy default is unmeasured on GPU.
- No **matrix/batch** benchmark, although that is the dominant workload.
- No **cold-start / data-path** timing, although it is claimed to be 85–97 % of end-to-end.
- No **constrained** benchmark above 2.55 M cells; no constrained **thread-scaling** data at all.
- The **TDR threshold** on this machine is unknown (2 s WDDM default vs a ~60 s recorded observation).
- The RTX PRO 500's **SM count** is not recorded anywhere (14 assumed throughout).

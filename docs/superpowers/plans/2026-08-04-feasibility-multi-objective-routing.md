# Feasibility-Based Multi-Objective Routing: User-Weighted Objective Functions

**Date:** 2026-08-04 (v2: directional metrics moved **into the search kernels**)
**Revised:** 2026-08-05 (v3: geometry/response separation — 3D length is
unconditional geometry, only the slope→feasibility *response* is configurable;
incorporates a first-hand re-audit of the 3D code that found four verified defects
the new design fixes structurally, §2.3.1)
**Status:** Phases 1–9 IMPLEMENTED (2026-08-05/06, uncommitted).
Phase 9 scope: `weight_precision="float32"` (lossless combine, inf-forbidden)
on the library backends + GPU V4 (templated float kernel); the cython kernels
stay uint16-pinned by design. Remaining: delta-stepping gradient support (4b);
constrained: objective gradient responses / LUT-parity in constrained GPU
V1–V4 + cython_parallel + clearance variants, profile `routing:`/`accounting:`
schema split. Docs: `core_features/cost_semantics.md`; example:
`examples/feasibility_multi_objective_routing.ipynb`.
**Scope:** PathFinder (2D + 3D), ConstrainedPathFinder, GeoRasterizer, CostAssumptions,
Path reporting, all graph backends (Cython Dijkstra/delta-stepping, networkit/-x/
igraph/rustworkx, raster GPU V4, constrained CPU/GPU kernels).
**Relationship to prior plans:** absorbs and generalizes
`2026-06-09-dual-metric-cost-model.md`; composes with
`2026-03-12-multi-infrastructure-routing-design.md`. As a byproduct, resolves the
CPU/GPU gradient-model parity gap that the dual-metric plan declared out of scope.

---

## 0. The non-negotiable requirement (v2 clarification)

**Every criterion influences the route *during* the search.** Nothing about the
route choice is ever post-processed. The post-hoc `MetricEvaluator` (§6.8) exists
solely to *report* the metrics of the already-optimal path in honest units — it never
adjusts, re-ranks, or corrects a route.

There are two structurally different ways a criterion enters the search, and
conflating them was the flaw of v1:

- **Cell metrics** (cost, landscape index, permitting difficulty, planning-duration
  proxy, …): properties of a location. These enter through the combined weight
  raster — which, by the linearity of the edge formula (§2.2), is *mathematically
  identical* to weighting them inside every edge relaxation. In-search, exact, zero
  kernel changes.
- **Edge metrics** (the along-route gradient): properties of a *step* `(u → v)`.
  The DEM stores only heights; the gradient of a step is
  `|DEM[v] − DEM[u]| / horizontal_dist(direction)` — it depends on both endpoint
  cells, the step length, and the direction. **No per-cell surface can represent
  it**: on a uniformly tilted plane every cell has the same slope magnitude
  `|∇DEM|`, so a cell-based slope layer cannot distinguish climbing the fall line
  from following the contour — the exact distinction a "lowest-gradient route" must
  make. Edge metrics therefore must be computed inside the relaxation loop, from
  the DEM, per direction. That is the main addition of v2 (§3.2, §7).

**Geometry vs. response (v3).** When a DEM is present, slope affects the objective
through two channels that must never be conflated:

1. **Geometry — unconditional.** The true step length is
   `L_3d = L_2d · sqrt(1 + (s/100)²)`. Every per-meter quantity (cost, landscape
   exposure, the `length` metric itself) accrues over **3D meters, always** — this
   is physics (trench/conductor length), not a preference, and it is **not a
   configuration knob**.
2. **Response — configurable.** *How* slope additionally contributes to the
   feasibility beyond geometry is the user's choice: a multiplicative penalty
   (exponential, power, energy, sigmoid, squared — the five existing models — or a
   custom callable) and/or an additive exposure term weighted by
   `objective["gradient"]` with its own shape (linear, quadratic, exponential,
   custom). Defaults: multiplier ≡ 1, additive weight 0 — pure geometry.

Both channels ride the same kernel mechanism (§3.2) at zero marginal cost, because
stretch and response are folded into one lookup table host-side.

Two slope quantities are genuinely different and **both are kept**, under
unambiguous names:

| Name | Kind | Meaning | Use |
|---|---|---|---|
| `gradient` | **edge** metric (in-kernel, directional) | along-route grade of a step | "lowest-slope route", grade-limited infrastructure, switchback-finding |
| `terrain_slope` | cell metric (optional layer) | steepness of the terrain at a cell, `|∇DEM|`, direction-independent | constructability: a side-hill trench is hard to build even when traversed dead-level along the contour |

---

## 1. Problem statement

pyorps minimizes exactly one quantity: the single uint16 cost raster accumulated
along the path. Users who want routes to reflect anything that is not money —
landscape value, permitting risk, planning duration, along-route gradient — must
hand-bake it into that one number, losing:

1. **Expressiveness** — "shortest path", "lowest-gradient path", or "70 % cost /
   30 % landscape" cannot be requested. The fast backends (Cython, raster-GPU)
   cannot see slope *at all* today (§2.3).
2. **Transparency** — once penalties are mixed in, reported `total_cost` is neither
   money nor any interpretable quantity (`path_finder.py:929` reports
   `Σ category × 2D-length` while the 3D search minimizes
   `terrain × 3D-length × gradient_penalty`).
3. **Comparability** — route variants computed under different hand-tuned rasters
   cannot be compared criterion by criterion.

**Goal:** a first-class **feasibility objective**

```
F(path) = Σ_k  w_k · M_k(path)        k ∈ {cost, length, gradient, landscape, permit, duration, …}
```

minimized *by the search* on every backend, with user-supplied weights, a default
`{cost: 1.0}` that reproduces today's behavior bit-for-bit, and results that report
**all** metrics regardless of which objective was minimized.

---

## 2. Evidence: where the objective lives today

### 2.1 One canonical edge-weight formula, implemented seven times

Every unconstrained backend computes

```
w(u→v) = ( raster[u] + Σ raster[intermediates] + raster[v] ) × cost_factor(dr,dc)
cost_factor(dr,dc) = sqrt(dr² + dc²) / (2 + n_intermediates)
```

i.e. *mean cell value along the step × Euclidean step length (cell units)*:

| Implementation | Location | Weight dtype |
|---|---|---|
| numba/Cython edge construction (feeds networkit, networkx, igraph, rustworkx) | `_traversal_numba.py:211-214`, `:354-357`, `:430`; driver `:476-555` | float64 edge list |
| numba 3D edge construction | `_traversal_numba.py:1035-1065` (`× edge_length_3d`, `× gradient_penalty`, **clamped at 65535** at `:1064-1065`) | float64 edge list |
| Cython Dijkstra (on the fly) | `_dijkstra.pyx:240-243` | float64 dist |
| Cython delta-stepping (on the fly) | `_delta_stepping.pyx:281`, `:392` | float32 dist (packed with uint32 pred) |
| Raster GPU V4 (+ v1–v3 fallbacks) | `sssp_gpu.py:780`, `:821` (and 5 siblings) | float32 dist |
| Constrained CPU kernels | `_constrained_dijkstra.pyx:388-393` (`terrain × grad_mult + angle_cost`; tower branch adds `_tower_terrain + tower_angle_cost`) | float64 dist |
| Constrained GPU kernels | `relax_constrained_v3.cu:185-198`, `adds_tower.cuh:106-126`, `adds_wtb.cuh:182-184` | float32 dist |

### 2.2 Load-bearing observation: the formula is **linear in the cell values**

For per-cell surfaces `A`, `B` and scalars `α`, `β`:
`edge_weight(α·A + β·B) = α·edge_weight(A) + β·edge_weight(B)`.
A weighted sum of **cell-metric** rasters combined once in NumPy therefore yields
*exactly* the same optimal path as weighting those metrics inside every kernel.
Cell metrics require **zero kernel changes**. (This argument does not apply to edge
metrics — hence §3.2.)

### 2.3 Where slope machinery exists today — and where it is absent

- **Edge-list path (library backends): full directional slope.**
  `construct_edges_3d` computes `height_diff = |dem[v] − dem[u]|`, 3D edge length,
  and a multiplicative `gradient_penalty` with **five** selectable models
  (`exponential`, `power`, `energy`, `sigmoid`, `squared` —
  `_traversal_numba.py:837-954`, dispatch `:1056-1061`).
- **Constrained CPU kernel: directional slope**, precomputed into a
  per-`(cell, direction)` multiplier cache (`_precompute_gradient_cache`, consumed
  at `_constrained_dijkstra.pyx:391-392`), hard `max_gradient_pct` rejection.
- **Constrained GPU kernels: directional slope**, computed in-kernel from the DEM
  (`relax_constrained_v3.cu:188-195`) with a **hard-coded** exponential form.
- **Unconstrained Cython Dijkstra / delta-stepping / raster GPU V4: no DEM, no
  slope, nothing.** `RasterGPUAPI` accepts `dem_data` but never forwards it
  (`raster_gpu_api.py:84-93`); its only effect is disabling the symmetric-search
  optimization (`raster_gpu_api.py:159`). `grep dem sssp_gpu.py` → zero hits.

So the fast backends cannot express slope preferences at all today, and the slow
path expresses them only multiplicatively. v2 fixes both with one uniform mechanism.

#### 2.3.1 Verified defects in the existing 3D path (2026-08-05 re-audit)

Re-reading the 3D code first-hand surfaced four concrete defects. They both justify
the redesign and define its correctness bar:

1. **Unit mixing — resolution-dependent slope.** `find_valid_nodes_3d` computes
   `horizontal_dist = sqrt(dr² + dc²)` in **cell units** (`_traversal_numba.py:1008`)
   but `height_diff` in **meters** (`:1044`); `edge_length_3d`, `cost_factor_3d`,
   and the slope fed to `calculate_gradient_penalty` (`:1045-1058`) mix the two.
   The 3D math is only correct when `cell_size == 1 m`; at 10 m resolution the
   slope is overestimated 10× and gradient penalties fire absurdly early. The
   constrained kernels do it right (`step_distances` in meters,
   `relax_constrained_v3.cu:191`). The new design's per-direction
   `inv_horiz_m[d] = 1/(sqrt(dr²+dc²)·cell_size)` is correct at any resolution.
2. **Saturation clamp.** `final_cost > 65535 → 65535` (`:1064-1065`): expensive
   steep edges become indistinguishable to the optimizer. The new float
   composition (quantized terrain × float LUTs) has no clamp.
3. **DEM alignment unchecked.** `PathFinder.create_graph` passes
   `dem_raster_handler.data[0]` straight into `construct_edges_3d`
   (`path_finder.py:527-549` → `graph_library_api.py:96-102`) with **no shape
   check or resampling**; numba then indexes the DEM with cost-raster indices,
   boundscheck off — out-of-bounds garbage or a crash whenever DEM resolution
   differs from the cost raster. (`ConstrainedPathFinder._prepare_dem_data`
   resamples; the unconstrained path does not.) The MetricStack's aligned DEM band
   (§6.2) fixes this structurally for every consumer.
4. **`use_gpu` edge construction is dead code.** `graph_library_api.py:124` imports
   `construct_edges_gpu` from `traversal_gpu.py` — **the function does not exist
   anywhere in the codebase**; the ImportError is caught and mislabeled as "cupy
   not installed", silently falling back to CPU. Schedule removal or restoration
   independently of this plan.

Consequence for compatibility (§11): the legacy 3D path is kept bit-identical
(bugs included) for users who don't opt into the objective system; the new path is
the corrected one, and a warning on the legacy path flags `cell_size ≠ 1 m`.

### 2.4 Hard dtype boundaries

- Cython + GPU raster kernels are pinned to **uint16** with the **65535** sentinel
  (buffer-level errors verified; GPU host wrapper hard-casts
  `raster.astype(np.uint16)` at `sssp_gpu.py:1234`, `:1432` — silent truncation for
  float input).
- The four graph-library backends already consume **float64** edge weights — only
  the shared edge constructor blocks float input.
- Reporting is also uint16-pinned: `calculate_path_metrics_numba`
  (`_traversal_numba.py:587-700`) builds a dense category LUT of size `max−min+1`.

### 2.5 Couplings the design must respect

1. **65535 is sentinel and magnitude ceiling** — 3D edge weights clamp at 65535
   (`_traversal_numba.py:1064-1065`); the delta-stepping circular-buffer guard
   bounds bucket span by `max_cell × max_step_dist / delta`
   (`_delta_stepping.pyx:1679-1687`) — any multiplicative slope term must enter
   this bound (§7.4).
2. **Constrained tower-terrain LUT is keyed by raster value**
   (`tower_terrain_lut[raster_val]`, 65536-entry — `adds_tower.cuh:66`). Meaningless
   once the raster holds combined feasibility values. §8 resolves this.
3. **`_traversal.pyx` source is missing** — only the generated `.cpp` and a stale
   `.pyd` exist; `_traversal` is absent from `setup.py`; `traversal.py:13-44`
   prefers the compiled module. **Editing `_traversal_numba.py` is dead code** until
   the orphan `.pyd` is removed. All new edge/metric code goes into new modules.
4. GPU `ignore_max=False` latent bug: `max_cost=65536` casts to `unsigned short` 0,
   making value-0 cells impassable (`sssp_gpu.py:1423-1424`). The quantizer never
   emits 0-valued cells (§5.2) and the bug gets fixed in passing.
5. Symmetric-search optimization on the GPU assumes symmetric edge weights; the
   gradient uses `|Δh|` (symmetric), so it remains valid. Signed uphill-only
   penalties would break it — explicitly out of scope.

---

## 3. Design principle

> **The search minimizes one non-negative scalar per edge. That scalar is composed
> from (a) a combined cell-metric raster — exact by linearity — and (b) per-edge
> gradient terms evaluated inside the relaxation loop from the DEM through a pair
> of slope-response lookup tables. Post-processing only ever *reports*.**

```
                ┌─────────────────────────────────────────────────┐
  vector data ─►│ MetricStack: K aligned float32 CELL layers      │
  rasters ─────►│ cost | landscape | permit | duration |          │
  DEM ─────────►│ terrain_slope … (+ implicit length, + category) │
                └──────────────┬──────────────────────────────────┘
                               │ combine(Objective) — one fused NumPy pass
                               ▼
        float32 F[cell] = Σ w_k·m_k[cell]  ──quantize──►  uint16 W (forbidden→65535)
                               │
        DEM (float32, aligned) ┤        Objective gradient terms
                               ▼               ▼
                ┌─────────────────────────────────────────────────┐
                │ KERNELS (all backends):                         │
                │  s = |DEM[v]−DEM[u]| · inv_horiz[d]             │
                │  b = bin(s)          (Γ LUTs, ~1–3 KB, L1/smem) │
                │  w(u→v) = W_sum · cf[d] · Γ_mult[b]             │
                │           + Γ_add[b] · dist[d]                  │
                │  Γ_mult[b] = ∞  ⇒ edge forbidden (max gradient) │
                │  (no DEM ⇒ Γ_mult≡1, Γ_add≡0 ⇒ today's formula) │
                └──────────────┬──────────────────────────────────┘
                               │ path indices
                               ▼
                MetricEvaluator (reporting only): one walk over ALL layers + DEM
                               ▼
                Path.metrics = {cost: EUR, length: m, gradient: …, landscape: …}
                Path.feasibility = achieved objective (recomputed, quantization-free)
```

### 3.1 Cell metrics — exact scalarization (unchanged from v1)

Combined before the search; in-search by linearity; §4–§5.

### 3.2 Edge metrics — the gradient, in-kernel

Per relaxation of `u → v` along direction `d`:

```
s        = |DEM[v] − DEM[u]| / horiz_m(d)          # chord slope of the step, %
b        = min( floor(s · bins_per_pct), n_bins−1 )
w(u→v)   = W_sum(step cells) · cf(d) · Γ_mult[b]  +  Γ_add[b] · dist(d)
```

- `horiz_m(d) = sqrt(dr²+dc²) · cell_size` — precomputed per direction **in
  meters** (fixing defect §2.3.1-1; the constrained kernels already carry exactly
  this as `step_distances`).
- `Γ_mult[b] = stretch(s) × response_mult(s)` where `stretch(s) =
  sqrt(1 + (s/100)²)` is the **unconditional 3D-length geometry** (present
  whenever a DEM is given — not configurable, per §0) and `response_mult` is the
  user-chosen multiplicative penalty (default ≡ 1; five legacy models or a custom
  callable). Both are **baked into one LUT host-side**, so the kernel performs no
  sqrt/exp at all. Entries beyond `max_gradient_pct` are `+inf` ⇒ the same LUT
  read implements the hard grade limit.
- `Γ_add[b] = stretch(s) × Σ_j w_j · g_j(s)` (additive responses, per **3D**
  meter, pre-scaled by the quantization scale, §5.4) — the term that makes
  *"minimize gradient exposure"* a first-class objective:
  `M_gradient(path) = Σ s·L_3d` (÷ length = mean grade). Response shapes `g_j`:
  linear, quadratic, exponential, custom — the user's "how it adds" dial.
- Because `stretch` multiplies the combined raster term, every cell metric —
  including the implicit `length` layer — automatically accrues over 3D meters:
  *slope is respected in path length in every case*, with zero extra kernel
  arithmetic.
- The two tables are interleaved as a `(float2)` pair per bin — one cache-line /
  shared-memory access fetches both.
- Chord semantics: for R2+ steps the slope uses the endpoints (`Δh` over the full
  step) — identical to the existing 3D and constrained implementations. Local
  steepness *within* a long step is invisible (ridge-hiding); users with strict
  grade limits at large R are pointed to smaller neighborhoods or `hard_max` on
  `terrain_slope`. Documented, tested, not silently改.
- Response functions are **arbitrary**: any of the five existing CPU models, or a
  user callable, discretized host-side into the LUTs (default bin width 0.25 %,
  ~400–800 entries, 2 LUTs × 4 B ⇒ ≤ 6.4 KB — L1-resident on CPU, shared memory on
  GPU). Discretization error is bounded and testable; it also ends the CPU-vs-GPU
  penalty-model divergence, because both sides consume the *same tables*.

Hot-loop cost when active: one extra float32 global read (`DEM[v]`; `DEM[u]` is
hoisted per settled state), one multiply, one float→int cast, one 8-byte cached LUT
read, two FMAs. When inactive (`dem == NULL`): the existing code path,
bit-identical, zero overhead.

**Why this is the right mechanism for a high-performance library** (alternatives
re-examined and rejected):

- **vs. transcendentals in the loop:** today's 3D numba path computes a `sqrt` and
  an `exp`/`pow`/`atan` **per edge** (`_traversal_numba.py:1045-1058`). The LUT
  path replaces all of it with one multiply + cast + cached load — the new 3D hot
  loop is *cheaper per edge than the existing one*, while supporting arbitrary
  response shapes and exact CPU/GPU parity (both consume the same tables).
- **vs. per-(cell,direction) precomputed weight caches** (the constrained kernel's
  pattern): unconstrained Dijkstra/delta-stepping relaxes each edge O(1) times, so
  a cache is pure overhead — 0.6 GB (R2) to 2.3 GB (R4) at 9 Mpx of wasted
  allocation and memory bandwidth. The constrained kernels *keep* their cache
  because their `n_dirs × span_bins` states revisit the same cell edges many times.
- **vs. materialized 3D edge lists everywhere:** that is the library-backend path
  (kept); for the raster kernels it would forfeit their defining advantage — no
  edge storage at all.
- **Preserves every existing optimization untouched:** uint16 raster pipeline,
  IEEE-754 `atomicMin` trick, packed float32-dist‖uint32-pred CAS, `-dlcm=cg`,
  bucket queues, symmetric search (`|Δh|` is symmetric).
- DEM ingestion sanitizes nodata (`NaN`/nodata → cell forbidden, never reaching
  the kernels — a `(int)(NaN·x)` cast is undefined on GPU).

### 3.3 Explicitly not built (and why)

- **Pareto label-setting search** (multi-label Dijkstra/Martins): label sets grow
  combinatorially; incompatible with the delta-stepping/GPU designs. Weighted-sum
  scalarization + cheap weight sweeps (§9) trace the convex front, which covers the
  practical need.
- **Resource-constrained shortest path** ("cheapest with ≤ 5 km through forest"):
  NP-hard; the constrained state-space machinery could host a budgeted variant
  later; out of scope. Hard per-cell/per-edge limits (masks, `Γ_mult=∞`) cover the
  engineering cases.
- **A\*** in any form (standing decision).
- **Signed/asymmetric gradient** (uphill ≠ downhill) — breaks the GPU symmetric
  search; revisit only with a concrete use case.

---

## 4. User-facing data model

### 4.1 Metric layers and the semantic contract

Cell layer: `m_k[cell]` = intensity of criterion *k* **per meter of path** through
that cell. Edge metric `gradient`: `s(u→v)` in percent, exposure `Σ s·L`.

| Layer | Kind | Unit | `M_k(path)` |
|---|---|---|---|
| `cost` | cell | EUR/m | construction cost (EUR) |
| `length` | implicit | 1 | route length (m; **always 3D when a DEM is present**) |
| `gradient` | **edge** | % | grade exposure (%·m); `/length` = mean grade |
| `terrain_slope` | cell | % | cross-slope constructability exposure |
| `landscape` | cell | index 0–1 | protected-landscape exposure (index·m) |
| `permit` | cell | index 0–1 | permitting exposure |
| `duration` | cell | months/km ÷1000 | planning-duration proxy (months) |

Forbidden in any layer (`NaN`/`inf`/65535) forbids the cell in the combined surface
regardless of weights. `Γ_mult[b]=∞` forbids an *edge* without forbidding its cells.

### 4.2 Multi-valued cost assumptions (backward compatible)

```python
cost_assumptions = {
    "Forest":      {"cost": 365, "landscape": 0.9, "permit": 0.6, "duration": 0.5},
    "Agriculture": {"cost": 107, "landscape": 0.2, "permit": 0.2, "duration": 0.1},
    "Residential": 65535,          # forbidden in ALL metrics
    "Grassland":   130,            # legacy scalar ⇒ {"cost": 130} — today's behavior
}
```

The dual-metric plan's `{cost, factor, weight}` forms stay valid (re-expressed as a
steering layer). CSV/Excel: extra numeric columns named after metrics; JSON: dict
leaves. Resolution rule in one place in `CostAssumptions`.

### 4.3 Layer sources and gradient configuration

```python
PathFinder(
    ...,
    dem="dgm1.tif",                                  # enables `gradient` + derived layers
    metric_layers={
        "terrain_slope": {"derive": "slope_from_dem", "hard_max": 45.0},
        "landscape":     {"source": "schutzgebiete.gpkg",
                          "values": {"NSG": 1.0, "LSG": 0.6, "": 0.0}},
        "noise":         "noise_index.tif",          # prebuilt, resampled to window
    },
    objective={"cost": 1.0, "gradient": 40.0, "landscape": 800.0},
    gradient_options=dict(            # all optional
        # NOTE: the 3D length stretch is NOT an option — with a DEM it is always
        # applied (geometry, §0). Only the RESPONSE below is configurable.
        additive="quadratic",         # g(s) shape for Γ_add, weighted by objective["gradient"]
        additive_params={},           #   ("linear" | "quadratic" | "exponential" | callable)
        multiplier=None,              # optional multiplicative penalty → Γ_mult
        multiplier_params=None,       #   (the 5 legacy models or a callable; None ⇒ ×1)
        max_gradient_pct=None,        # hard limit ⇒ Γ_mult = ∞ beyond
        bin_width_pct=0.25,
    ),
)
```

- `length` is implicit (constant `c` contributes exactly `c × path_length`; the
  unconditional stretch in `Γ_mult` makes that **3D** length whenever a DEM is
  present) — never materialized.
- `gradient` is reserved: no raster behind it; requires `dem`.
- Prebuilt rasters are windowed/resampled to the cost window (existing DEM
  resampling pattern, `constrained_path_finder.py:296-318`).

### 4.4 The `Objective`

```python
Objective.cheapest()                     # {"cost": 1.0} — the default
Objective.shortest()                     # {"length": 1.0}
Objective.gentlest(length_eps=0.01)      # {"gradient": 1.0, "length": 0.01}
Objective.from_priorities(cost=5, landscape=3, gradient=2, stack=stack)
   # auto-scales by layer p95 so priorities express relative influence; RESOLVED
   # absolute weights are stored on the result (auto-scaling is window-dependent
   # and never silently re-derived).
```

Weights ≥ 0, ≥ 1 positive, unknown names error listing available layers.
`set_objective()` swaps objectives between runs. Zero-cost degeneracy guard: pure
`{"gradient": 1}` on flat terrain has zero-weight regions ⇒ warning + presets always
include a small `length` term.

### 4.5 Overlays / modifiers target metrics

`datasets_to_modify` gains `applies_to: [metric, ...]` (scalar default `["cost"]` =
today; `"all"` supported); forbidden always applies to all.

### 4.6 Hard constraints as masks

Layer-level `hard_max`/`hard_min` (cells) and `max_gradient_pct` (edges, via
`Γ_mult=∞`). Exact, cheap, and expressible on every backend.

---

## 5. Feasibility surface construction

### 5.1 Combine

One fused float32 pass per non-zero cell weight (+ implicit length offset on
traversable cells, + forbidden-mask union). ≈ 10 ms/layer @ 9 M cells.

### 5.2 Quantize — pure scaling only

Multiplying all cell values by α > 0 preserves the argmin exactly; adding an offset
does not (it injects a hidden length term):

```
scale = 65534.0 / F.max(traversable)
W     = clip(round(F * scale), 1, 65534).astype(uint16)     # forbidden → 65535
```

Lower clip at 1 keeps zero cells out of the kernels (GPU 0-sentinel bug, §2.5.4;
delta sanity); distortion ≤ 0.5/65534 of max. Diagnostics (mandatory): effective
resolution report; warnings when distinct float values collapse or the median cell
quantizes below 8 levels. The float `F` is kept for evaluation (and Phase 8).

### 5.3 Cost of an objective change

| Backend | `set_objective()` + `find_route()` |
|---|---|
| cython / raster_gpu | combine + quantize (ms) + LUT rebuild (µs) + search — **no graph build** |
| library backends | combine + quantize + full edge rebuild + graph build + search |

Sweeps → cython/raster_gpu, run **sequentially** (standing resource policy).

### 5.4 Unit bookkeeping (kernel invariant)

All in-kernel terms live in *quantized-feasibility × cell-length* units:
- `W_sum × cf(d)` — as today; `× Γ_mult` adds the dimensionless stretch×response.
- `Γ_add` is pre-multiplied host-side by the same `scale` (and by `cell_size`,
  folding meters→cells), so additive gradient terms are commensurable with the
  quantized terrain term **without consuming uint16 resolution** (they stay float32
  in-kernel).
- Slope itself is computed in true meters via `inv_horiz_m[d]` (per-direction,
  `1/(sqrt(dr²+dc²)·cell_size)`) — resolution-independent, unlike the legacy 3D
  path (§2.3.1-1).
The evaluator converts back to user units from the float layers + DEM; the kernels
stay unit-agnostic.

---

## 6. Architecture changes per module

### 6.1 New: `pyorps/core/objective.py`
`Objective` (weights, validation, presets, `from_priorities`, gradient options,
serialization). Builds the Γ LUT pair from response specs (`build_gradient_luts
(scale, cell_size) -> (mult_lut, add_lut, inv_horiz_m, bin_inv, n_bins)`).

### 6.2 New: `pyorps/core/metric_stack.py`
`MetricStack`: named float32 cell layers + forbidden mask + category band (uint16)
+ DEM band + `cell_size`. `combine(objective) -> (F, W, scale)`; joint windowing;
multi-band GeoTIFF save/load (band names in tags); `from_single_raster` zero-copy
legacy alias.

### 6.3 `CostAssumptions`
Dict-valued leaves → per-metric mappings; `apply_to_geodataframe` writes one column
per metric; validation (finite, ≥0, forbidden all-metric). Legacy scalars pinned
byte-identical.

### 6.4 `GeoRasterizer`
One geometry sort/buffer pass, K value bindings + category band → `MetricStack`.
Metric bands are float32, unrounded. Modifiers per `applies_to`.

### 6.5 `RasterHandler`
Stack-wide windowing (one window, all bands + DEM). Multi-band load restores a
stack; single-band degenerates to the alias.

### 6.6 `PathFinder`
New keyword-only params `metric_layers`, `objective`, `gradient_options`. Holds the
stack; lazy `W` + LUTs; `set_objective()` invalidates `_graph_api` for library
backends only. Passes `dem` + LUTs to capable backends (§7). Per-call
`find_route(objective=...)` override for sweeps.

### 6.7 New: `pyorps/utils/metric_edges.py` (edge construction v2)
Numba module (fresh — never `_traversal_numba.py`, which is shadowed by the stale
`_traversal.pyd`): `construct_edges_weighted(W, dem, mult_lut, add_lut, …)` producing
float64 edge lists for the library backends with the §3.2 formula (no 65535 clamp —
that was a uint16-ism). The legacy `construct_edges[_3d]` path remains untouched and
is used whenever no custom objective is set.

### 6.8 Reporting: `MetricEvaluator` (new `pyorps/utils/metric_eval.py`)
One numba walk of the path over all K layers + DEM (same step decomposition and the
same Γ LUTs as the search ⇒ no formula drift):
totals per metric, gradient exposure + max grade, 2D/3D lengths, category-band
breakdowns (fixes the dense-LUT `length_by_category`), and
`feasibility = Σ w_k·M_k` recomputed quantization-free.
**Length semantics (behavior change, flagged in release notes):** with a DEM,
`total_length` and all per-meter metrics use **3D meters** (`total_length_2d`
retained alongside) — today's 2D-only reporting during 3D routing is one of the
inconsistencies this plan exists to remove.
`Path` gains `metrics`, `feasibility`, `objective_spec` (weights + resolved scales +
quantization scale + LUT hashes), `total_length_3d`, `max_gradient_pct`.
Deprecation of `total_cost`/`total_cell_cost` per the dual-metric plan §4.6.
Cross-check tests against `distances[target]/scale` where backends expose it.

---

## 7. Per-backend implementation of the gradient terms

The mechanism is identical everywhere: nullable `dem` + `Γ_mult`/`Γ_add` LUTs +
per-direction `inv_horiz_m`. No slope objective ⇒ null pointers ⇒ existing code
path, bit-identical.

| Backend | What changes | Pattern source | Est. size |
|---|---|---|---|
| **Library backends** | consume `metric_edges.construct_edges_weighted` (§6.7) — host-side only | existing `construct_edges_3d` | new numba module |
| **Cython `_dijkstra.pyx`** | optional params (`dem`, LUTs, `inv_horiz_m`, `bin_inv`, `n_bins`); per-neighbor slope + LUT lookup **on the fly** (each edge relaxed O(1) times — a per-(cell,dir) cache would cost GBs for zero reuse, unlike the constrained case) | `relax_constrained_v3.cu:188-195` logic, Cython-ized | ~60 lines |
| **Cython `_delta_stepping.pyx`** | same, in both relax variants; **bucket-span guard** (`:1679-1687`) extended: `max_span = (max_cell·max_step·max_finite(Γ_mult) + max(Γ_add)·max_step) / delta`; auto-delta scaled by the mean multiplier estimated from the `|∇DEM|` histogram (one vectorized numpy pass host-side) | same | ~80 lines |
| **GPU raster V4 (`sssp_gpu.py`)** | kernel: `const float* dem` (nullable, `np.intp(0)` convention), interleaved float2 LUT in shared memory (+≤6.4 KB — budget fine, current smem ≤1.5 KB), slope + LUT in light & heavy relax; host: sanitize + upload DEM (+4 B/px ⇒ 36 MB @ 9 Mpx — trivial vs the ~2.4 GB queues), `_compute_auto_delta` × mean multiplier from the same histogram | `relax_constrained_v3.cu` (already written for constrained) | ~120 lines kernel + host |
| GPU v1–v3 fallback kernels | reject gradient objectives with a clear error ("requires V4/cython") — not worth 5× duplication | — | guard only |
| **Constrained CPU** | gradient cache (`_precompute_gradient_cache`) extended to two channels (mult, add) fed **from the same LUTs**; cache retained (states revisit edges n_dirs×span_bins times, unlike unconstrained) | existing cache | ~40 lines |
| **Constrained GPU (V2/V3)** | replace hard-coded `expf(gradient_scale·s/100)` (6 sites) with `Γ_mult[b]` + add `Γ_add[b]·dist` — **CPU/GPU penalty parity achieved as a byproduct**, closing the gap the dual-metric plan deferred | LUT upload alongside existing angle LUTs | ~60 lines |
| Constrained GPU V4 (ADDS) | currently ignores DEM entirely (`constrained_sssp_gpu_v4.py:931-938` accepted, never passed); wire LUT terms when that backend is next touched — phase-gated with its validation | — | deferred |

Correctness notes:
- Weights stay non-negative (LUTs validated ≥ 0; `Γ_mult ≥ ε > 0` below the hard
  limit) ⇒ Dijkstra/delta-stepping optimality untouched.
- `|Δh|` symmetry keeps the GPU symmetric-search optimization valid (§2.5.5).
- LUT discretization: route stability under bin-width halving is a test, not a hope.

---

## 8. Constrained routing integration

1. **Terrain term** = combined `W` (drop-in).
2. **Term weights host-side:** `angle_cost_lut × w_smoothness`,
   `tower_*_costs × w_tower` before upload (defaults 1.0 ⇒ bit-identical); profile
   `routing:`/`accounting:` split from the dual-metric plan Phase 5 slots in here.
3. **Tower terrain lookup fix:** replace the value-keyed 65536-entry LUT with a
   **per-cell tower-cost raster** `tower_cost[cell] = LUT[category[cell]]` built
   from the category band (float32; 36 MB @ 3000²; same load count in-kernel —
   index by cell instead of value; `adds_tower.cuh:66`, `relax_constrained_v3.cu:
   322-364`, `_tower_terrain` + call sites). Legacy inputs reproduce the LUT
   bit-identically through the same code path.
4. **Gradient terms** via §7 (LUT parity fix included).
5. `ConstrainedPath` reporting joins the evaluator model + tower/conductor EUR
   accounting from the dual-metric plan Phase 5.

---

## 9. Route ensembles and trade-off exploration

```python
variants = finder.find_route_ensemble({
    "cheapest":   Objective.cheapest(),
    "shortest":   Objective.shortest(),
    "gentlest":   Objective.gentlest(),
    "balanced":   {"cost": 1.0, "gradient": 40.0, "landscape": 800.0},
})
variants.to_dataframe()                                   # rows=variants, cols=ALL metrics
pareto = variants.pareto_front(["cost", "gradient", "landscape"])
```

Sequential execution; every path carries all metrics, so the comparison table is
free. `compare_optimal=("cost",)` reports per-metric deltas vs the single-criterion
optimum ("your policy costs +X EUR, saves Y %·m grade exposure"). Honest caveat:
weight sweeps trace the **convex** Pareto front only (§3.3). Test property: raising
`w_k` weakly decreases `M_k` of the returned route. The weights dict is GUI-ready
(sliders); GUI wiring out of scope.

---

## 10. Performance budget

| Cost | Where | Magnitude |
|---|---|---|
| Stack memory | K float32 bands, windowed | 4·K B/cell (6 layers ≈ 216 MB @ 3000²) |
| Combine + quantize + LUT build | numpy, per objective change | ≈ 10 ms/layer + 20 ms + µs |
| Search, no gradient objective | unchanged kernels | **identical to today** (null-pointer fast path) |
| Search, with gradient | +1 float read + ~5 ALU + 2 LUT hits per edge | few % on CPU; noise on GPU vs atomics/queues |
| GPU VRAM | DEM 4 B/px + 6.4 KB smem | 36 MB @ 9 Mpx vs ~2.4 GB queues |
| Multi-band rasterization | K bindings, one geometry pass | ≈ K× one `rasterize` (one-time) |
| Evaluation | one numba walk × K layers | sub-ms/path |

---

## 11. Backward compatibility

- Default objective + scalar assumptions ⇒ combined surface **is** the legacy raster
  (zero-copy alias, scale=1, LUTs null) ⇒ byte-identical rasters, identical
  `path_indices`, identical timings; kernel fast paths bit-identical when gradient
  terms are inactive.
- `total_cost`/`total_cell_cost`/`length_by_category` keep formulas through the
  deprecation window. All new parameters keyword-only, `None` defaults. Persisted
  single-band GeoTIFFs load as before.
- **3D routing:** the legacy `construct_edges_3d` path stays bit-identical —
  including its verified defects (§2.3.1: cell-unit slope, 65535 clamp, unchecked
  DEM alignment) — for users who pass `dem=` without an objective. The corrected
  math lives in the new path (any `objective`/`gradient_options` usage). The
  legacy path gains warnings when `cell_size ≠ 1 m` (wrong slopes) or when DEM
  shape ≠ raster shape (undefined behavior today), and is deprecated over the
  same window as `total_cost`. Fixing-by-default was rejected: silently changing
  existing 3D routes is worse than a loud migration.

---

## 12. Implementation phases

| Phase | Content | Touches | Risk |
|---|---|---|---|
| **1** | `Objective` (incl. gradient-response LUT builder) + multi-valued `CostAssumptions` + validation + pinning tests | new `core/objective.py`, `core/cost_assumptions.py` | low |
| **2** | `MetricStack` + multi-band `GeoRasterizer` + category band + joint windowing + multi-band GeoTIFF I/O + legacy aliasing + DEM band alignment | new `core/metric_stack.py`, `raster/rasterizer.py`, `raster/handler.py` | low–medium (alignment invariants) |
| **3** | combine + quantize + diagnostics + `PathFinder` wiring (`objective`, `metric_layers`, `gradient_options`, `set_objective`); GPU `ignore_max=False` fix | `graph/path_finder.py`, `metric_stack.py`, `sssp_gpu.py` | low |
| **4** | **Gradient in the search — CPU:** `metric_edges.py` (library backends) + `_dijkstra.pyx` + `_delta_stepping.pyx` (incl. bucket-span/auto-delta bounds); derived layers (`terrain_slope`, `hard_max`, prebuilt ingestion) | new `utils/metric_edges.py`, `_dijkstra.pyx`, `_delta_stepping.pyx`, `metric_stack.py` | **medium — the heart of v2**; contour/parity tests gate it |
| **5** | **Gradient in the search — GPU:** raster V4 kernel + host (DEM upload, smem LUTs, auto-delta); v1–v3 guards | `sssp_gpu.py` | medium (kernel edit; port of proven constrained code) |
| **6** | `MetricEvaluator` + `Path.metrics/feasibility/objective_spec` + category breakdowns + `analyze()`/export + deprecations + docs `cost_semantics.md` | new `utils/metric_eval.py`, `core/path.py`, `graph/path_finder.py`, docs | medium (formula parity — shared LUTs/geometry mitigate) |
| **7** | `find_route_ensemble`, `pareto_front`, `compare_optimal` | `graph/path_finder.py`, `core/path.py` | low |
| **8** | Constrained integration: LUT weight-scaling, per-cell tower-cost raster, two-channel gradient cache (CPU), LUT-based gradient in constrained GPU V2/V3 (parity fix), profile split, `ConstrainedPath` metrics | `_constrained_*.pyx`, `kernels/*.cu(h)`, `constrained_sssp_gpu*.py`, `core/infrastructure_profile.py` | medium; gate on existing constrained suites |
| **9** *(optional)* | Float32 raster kernels (fused-type Cython, `const float*` GPU) if Phase 3 diagnostics show real resolution problems | `_dijkstra.pyx`, `_delta_stepping.pyx`, `sssp_gpu.py` | medium |

Phases 1–3 ship cell-metric objectives on every backend with zero kernel edits;
Phase 4 delivers "lowest-gradient route" on the default (Cython) backend; Phase 5
on the GPU. Each phase independently shippable.

## 13. Testing strategy

- **Pinning:** legacy configs ⇒ raster + `path_indices` identical to `main`;
  default objective takes the alias path; kernels with null gradient pointers
  bit-identical.
- **Scalarization exactness:** random small rasters/weights — combined+quantized
  cython path ≡ float64 per-edge weighted networkx path (quantization-bounded
  tolerance).
- **Contour test (the v2 discriminator):** uniform terrain, uniformly tilted-plane
  DEM, source/target on one contour. High `w_gradient` ⇒ route follows the contour;
  `w=0` ⇒ straight line. A per-cell slope layer **cannot** pass this test (all cells
  identical) — proving the edge formulation is doing the work.
- **Switchback test:** valley climb produces zigzags under high `w_gradient`;
  `max_gradient_pct` produces them as the *only* feasible routes.
- **Cross-backend parity:** cython Dijkstra ≡ delta-stepping ≡ `metric_edges`+
  networkit ≡ GPU V4 on identical LUTs (same routes; distances within float32).
- **LUT discretization:** halving `bin_width_pct` leaves test routes unchanged.
- **Legacy gradient models:** each of the 5 models LUT-ized reproduces
  `construct_edges_3d` weights within tolerance **at cell_size = 1 m** (the only
  resolution where the legacy path is correct); 65535-clamp removal documented in
  the comparison.
- **Resolution invariance (new-path property the legacy path fails):** the same
  terrain sampled at 1 m, 5 m, 10 m yields the same route and the same reported
  mean grade — pins the meter-based `inv_horiz_m` fix of §2.3.1-1.
- **DEM alignment:** DEM at a different resolution than the cost raster is
  resampled into the stack; a deliberate mismatch fed to the legacy path triggers
  the new warning.
- **3D length always:** with a DEM and default responses (multiplier ≡ 1, no
  additive weight), routes differ from 2D exactly by the stretch — verified
  against brute-force float 3D weights; `total_length` equals the 3D geometric
  length of the polyline over the DEM.
- **Bucket-span/auto-delta bounds:** steep synthetic DEM + large multiplier does not
  trip the circular-buffer guard falsely and never overflows silently.
- **Objective semantics:** `shortest()` ≡ uniform-raster path; cheap-but-protected
  corridor avoided when weighted, entered when not, with `metrics["cost"]` lower
  when forced through.
- **Monotonicity:** raising `w_k` weakly decreases `M_k` along a weight ladder.
- **Evaluator parity:** `feasibility` vs `distances[target]/scale` (where exposed);
  hand-computed 5×5 totals; category breakdown separates classes sharing a value.
- **Constrained (Phase 8):** per-cell tower raster ≡ LUT bit-identical on legacy;
  LUT-based gradient ≡ `expf` within discretization tolerance; weight 1.0 ⇒
  bit-identical.

## 14. Risks & open questions

1. **Kernel surface area** — v2 touches the four hot kernels (Dijkstra,
   delta-stepping, GPU V4, constrained). Mitigations: the formula is a strict
   superset behind null-pointer guards; the GPU logic is a port of already-working
   constrained code; parity tests across backends are the gate.
2. **uint16 resolution under extreme weight ratios** — diagnostics (Phase 3) +
   float32 escape hatch (Phase 9). Additive gradient terms bypass quantization
   entirely (float32 LUT).
3. **Chord-slope semantics at large R** — documented; `terrain_slope` `hard_max`
   and smaller neighborhoods as remedies; a per-intermediate max-slope check inside
   the step is a possible future strictness option (cheap: intermediates are
   already visited).
4. **`_traversal.pyd` shadowing** — all new code in new modules; schedule the
   orphan-`.pyd` cleanup / `_traversal.pyx` restoration independently (likewise
   the dead `use_gpu` edge-construction import, §2.3.1-4).
4b. **DEM nodata** — voids sanitized at ingestion (forbid or fill), never reaching
   kernels; `(int)(NaN·x)` is undefined on GPU.
4c. **Length-reporting behavior change** — `total_length` becomes 3D with a DEM
   (§6.8); loud release note, `total_length_2d` retained.
5. **Formula drift** search ↔ evaluator — same Γ LUTs and step geometry shared by
   construction; cross-check tests.
6. **Window-dependent auto-scales** (`from_priorities`) — resolved weights
   persisted; never re-derived silently.
7. **Lumpy criteria** (per-crossing permits/parcels): per-meter proxies for now;
   event-counting evaluator feature later (post-hoc counting only — routing impact
   would need cell-metric encodings such as boundary-buffer layers).
8. **Open:** dedupe identical paths across ensemble variants (cheap
   `path_indices` hash)? Decide in Phase 7.
9. **Open:** persist objectives + resolved scales + LUT hashes in project files /
   path GeoDataFrames for full reproducibility — coordinate with the GUI project
   format when sliders arrive.

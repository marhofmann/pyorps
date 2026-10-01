# Fused delta-stepping kernel — implementation, tests, benchmark & verdict

- **Date:** 2026-08-05
- **Task:** implement the three published multicore SSSP levers — bucket
  fusion (GraphIt, CGO 2020), decreasing effective Δ (PD3, HPEC 2024) as an
  adaptive super-bucket window, VGC local-work cap (PASGAL, SPAA 2024) — on
  top of the persistent delta-stepping architecture; accept only if proven
  faster than the unmodified production kernel.
- **Verdict: NOT ACCEPTED as a speedup.** No lever configuration beat
  `delta_stepping_2d_persistent` on any of 10 scenario points (4 raster
  families × sizes 500²–2000², 8 threads, median of 5, corner-to-corner).
  Best result: fusion-only at 0.91–1.04× (statistical tie on random
  rasters); fusion+window 0.26–0.79×.
- **However: the benchmark exposed a real correctness bug in the production
  baseline** (§3) — the fused kernel is currently the only *exact* parallel
  delta-stepping in the codebase.

## 1. What was built (all kept, none wired into CythonAPI)

- `pyorps/utils/_delta_stepping_fused.pyx` — `delta_stepping_2d_fused`:
  persistent-pool kernel with three independent knobs
  (`fusion_cap`, `window_init`/`window_max`/`adaptive_window`);
  overflow-safe by construction (guarded chunk grabbing + work rollover —
  no silent drops); pending-marker coalescing with a lossless
  producer/consumer atomic protocol (`atomic_exchange_u8` added to
  `atomic_cas.h`).
- `tests/test_graph/test_delta_stepping_fused.py` — 38 tests: contract
  parity, cost-optimality vs Dijkstra across 5 raster patterns × 5 lever
  combos, 8/16-neighborhoods, thread invariance (1/2/4), tiny-Δ stress,
  repeated-run stability. All green; existing suites unaffected (69 pass).
- `benchmarks/benchmark_fused_kernel.py` + `benchmarks/results/*.json`.

Two implementation bugs were found and fixed during development (both
would have silently produced wrong paths): buffer-guard granularity vs
worst-case chunk fan-out, and a per-window settled buffer that could stall
whole-search-in-one-window cases (now drained per round).

## 2. Benchmark summary (8 threads, Δ=100, margin=1.1, median of 5)

Speedup vs baseline (>1 = faster than production kernel):

| Scenario | no levers | fusion only | window only | fusion+window |
|---|---|---|---|---|
| random 500² / 1000² | 0.75 / 0.73 | 0.97 / 1.00 | 0.86 / 0.86 | 0.47 / 0.79 |
| heavy_tail 500² / 1000² | 0.56 / 0.58 | 0.35 / 0.40 | 0.43 / 0.55 | 0.26 / 0.27 |
| corridors 500²/1000²/2000² | 0.86/0.84/0.84 | 0.93/0.86/0.85 | 0.62/0.80/0.81 | 0.52/0.72/0.78 |
| obstacles 500²/1000²/2000² | 0.74/0.77/0.78 | 0.95/0.92/0.91 | 0.80/0.78/0.77 | 0.39/0.61/0.72 |

**Why the published levers don't transfer to raster grids:**

1. The baseline already banks the win the levers target. Bucket fusion's
   ~10× round reduction (GraphIt) pays off against per-round parallel-for
   launch overhead; our persistent pool with sense-reversing barriers
   already reduced a sync round to ~1–2 µs. On rasters, per-round frontiers
   are hundreds-to-thousands of cells even on corridor rasters, so barriers
   are a few percent of runtime — there is little left to save.
2. What fusion adds instead is work amplification: processing vertices
   immediately (LIFO, before their distances converge) causes repeated
   improve→requeue cycles that round-synchronized execution coalesces.
   On dense cheap regions (heavy_tail) this costs 2–3× the total work —
   pending-marker dedup does not remove it (it dedups queue entries, not
   re-improvements).
3. The window lever (decreasing effective Δ) widens rounds that are already
   wide; its parallelism gain is nil on saturated frontiers while the
   misordering cost grows. PD3's reported wins are on scale-free graphs
   (LiveJournal/Twitter); no road/grid data appears in that paper — the
   transfer simply fails.

## 3. Side-finding: the production baseline returns suboptimal paths

`delta_stepping_2d_persistent` (behind `CythonAPI` `algorithm=
"delta-stepping"`) is **nondeterministically suboptimal at ≥2 threads**:

- Observed 0.08 %–2.8 % above optimal across benchmark scenarios; on the
  diagnostic raster: exact at 1 thread, 0.58 %–1.27 % worse at 8 threads,
  *independent of the margin parameter* (1.00001 and 1.1 both affected).
- Suspected mechanism: the `last_bucket` dedup drops a re-insertion when a
  vertex is improved again into its current bucket after it was already
  popped in the same light iteration — the improved distance is then never
  re-relaxed. Single-threaded execution mostly masks the ordering; parallel
  execution exposes it.
- The fused kernel handles exactly this case (in-window re-queue via the
  pending protocol) and is deterministic-exact vs Dijkstra in every test
  and benchmark run — at ~0–25 % speed cost depending on pattern (the
  fusion-only configuration is the closest exact drop-in: 0.91–1.04×).
- Also observed (separate, small): classic/1-thread delta-stepping found a
  path 0.04 % *cheaper* than `dijkstra_2d_cython` on a random 500² raster —
  consistent with the previously noted Dijkstra wall-gap quirk; the Cython
  Dijkstra's optimality itself deserves an audit.

**Open decision (user):** keep production as-is (fast, ≤~3 % suboptimal at
8 threads), or offer the fused kernel (fusion-only config) as an exact
parallel delta-stepping at roughly equal-to-10 % slower — or fix the
baseline's dedup directly (re-add on same-bucket improvement; expected to
cost some of its speed advantage, since part of that advantage *is* the
skipped work).

---

# UPDATE 2026-08-05 (later the same day): fix applied + GPU levers

## 4. Production delta-stepping bug FIXED and verified

The `last_bucket` dedup fix (reset the stamp for every vertex popped from a
bucket — popped vertices are no longer queued, so a same-bucket improvement
must be able to re-queue them) was applied at all 8 pop sites in
`_delta_stepping.pyx` (both plain and persistent variants, single- and
multi-thread paths).

- **Verified exact**: worst excess over Dijkstra = 0.00e+00 across 36 runs
  (random / heavy_tail / obstacles × 500²/1000² × 1/8 threads × 3 reps);
  75 tests green.
- **Speed cost of correctness**: ~0–10 % on random rasters, up to ~50 % on
  heavy-tail (the skipped work *was* part of the old speed). Still 3–8×
  faster than Dijkstra everywhere.
- Re-verdict of the fused kernel vs the FIXED baseline: still no
  consistent win (fusion-only reaches 1.08× at 500² random but loses at
  larger sizes). **The fixed production kernel remains the CPU champion.**

## 5. GPU levers: window ACCEPTED (1.15–1.80×), fusion rejected

The same levers were ported to the V4 persistent cooperative GPU kernel
(`sssp_gpu.py`), where the bottleneck analysis is *opposite* to the CPU:
per-bucket frontiers (~10²–10³ cells) cannot occupy ~10⁴ CUDA threads, and
each light iteration costs ~3 grid barriers.

- **Super-bucket window** (`window` parameter, span = W·Δ, W clamped to
  [1, 32]): processes W buckets per phase → W× more per-phase parallelism,
  ~W× fewer barrier phases. **Benchmarked bit-exact against window=1 in
  every run**, and faster on every tested scenario:

  | Scenario | best W | speedup | W=4 speedup |
  |---|---|---|---|
  | random 1000² | 16 | **2.28×** | 1.80× |
  | random 2000² | 8 | 1.77× | 1.49× |
  | random 3000² | 8 | 1.61× | 1.37× |
  | heavy_tail 1000² | 2 | 1.18× | 1.15× |
  | heavy_tail 2000² | 4 | 1.26× | 1.26× |
  | heavy_tail 3000² | 2 | 1.19× | 1.17× |

  **Accepted: `window=4` is now the default** in `sssp_raster_gpu_v4` (the
  production `sssp_raster_gpu` entry inherits it) — the all-win setting
  (≥1.15× everywhere tested); smooth cost surfaces can pass `window=8` for
  more. 87 tests green with the new default.
- **Tail-chase fusion** (`fuse_depth`): implemented, correctness-preserving
  (chased vertices are still queued), but 0.59–1.16× — loses almost
  everywhere on GPU (warp divergence + duplicated work). Default 0; kept
  for experimentation.
- Two latent V4 hazards found during validation: (a) the
  `max_light_iterations` cap-break silently drops leftover frontier — the
  launcher now scales the cap by the window; (b) very large windows (≥64)
  can overflow the pending buffer (silent `p < buf_size` drops) — hence the
  [1, 32] clamp. Both pre-existing patterns, now documented.

**Why GPU won where CPU lost:** the window multiplies per-phase work. On a
saturated 8-thread CPU that only adds misordering; on an under-occupied GPU
it converts idle cycles into useful work. Same lever, opposite verdicts —
the bottleneck decides.

## 6. Constrained-kernel audit (same-day follow-up)

Audited `_constrained_delta.pyx` (4 variants) and `_constrained_dijkstra.pyx`
for the lost-relaxation bug class. Findings:

1. **Different design, same hazard class.** Both are Dial-style bucket
   kernels: `Δ = max(1.0, 2·min_raster·min_cf·(1−1e-9))` (just below the
   min terrain edge cost) with **visited-on-first-touch** — every batch
   member is marked settled before relaxation, and improvements to visited
   states are discarded at both relax and merge time. Exact **iff** no edge
   is cheaper than Δ. The `max(1.0, ·)` floor breaks that invariant for
   rasters containing **0-cost cells** (free corridors) → silent
   improvement-dropping, the constrained analog of the fixed CPU bug.
2. **Gradient-penalty hole closed by construction**: penalties are
   `exp(slope²·scale) ≥ 1`; the 0.0 "too steep" sentinel always co-occurs
   with `icache_status = 1`, which blocks the edge before the multiplier is
   consumed. Verified in `_precompute_gradient_cache`.
3. **A reproducible LIVELOCK, worse than suboptimality**: the inner loop's
   `batch.swap(buckets[phys])` never clears `batch`, so the previously
   processed batch bounces back into the bucket; the two vectors oscillate
   forever as soon as ANY push lands in the current bucket. Unreachable
   while the Dial invariant holds (all pushes go to later buckets) — a
   6×6 raster with 0-cost cells hangs `constrained_dijkstra_2d`
   indefinitely (reproduced; the delta variant escapes via its
   `n_active == 0` break, at the price of leaving work behind).
4. **Fixes applied** (dense-mode variants in both files):
   - Livelock: `buckets[phys].clear()` immediately after every swap
     (6 sites) — restores true double-buffering.
   - **Re-open protocol**: improvements to visited states are now buffered,
     applied, and the state returns to the frontier (`visited = 0`) — the
     kernels become label-correcting, making exactness independent of the
     Δ invariant. Zero-cost in the common case (the branch never fires when
     Δ ≤ min edge). The all-span-bins-visited early skip now also compares
     distances (skip only if no bin can possibly improve).
   - The hash-map/compact modes (lazy variant, sparse fallbacks) use
     separate structures; their swap sites received the livelock fix, and a
     dedicated guard audit is still open.
5. Validation status at time of writing: rebuild + 0-cost repro + 48-test
   constrained suite pending (tooling interruption); results to be appended.

---

# UPDATE 2026-08-06: §0 validation complete + lazy/compact audit closed

## 7. Validation results (open-levers plan §0)

- Rebuild + 0-cost livelock repro (subprocess with timeout) + 48-test
  constrained CPU suite: **all green**.
- New suite `tests/test_graph/test_constrained_zero_cost.py` (26 tests +
  1 documented xfail): termination on 0-cost corridors and all-zero
  rasters; randomized exactness of every bucket variant against the
  heap-mode reference (`force_sparse=1` — plain lazy-deletion Dijkstra,
  exact by construction); regression guard for min-raster ≥ 1.
- 300-seed adversarial sweep (16², ~3/8 cells 0-cost, all five bucket
  variants vs heap): **299/300 exact**. The single mismatch is the span-
  payload limitation below, not an improvement-drop.

## 8. Lazy/compact-hash guard audit (was open): fixed

`_height_sparse` (compact-dense) dropped improvements to FLAG_VISITED
states at all 3 relax guards and in the merge; the lazy hash-map variant
dropped them in its merge. Both now follow the re-open protocol
(dist-aware relax guards; merge applies the improvement, clears
visited, re-pushes). All five bucket variants share the protocol now.

## 9. New findings from the audit session

1. **Span-bin aliasing (memory safety, all variants)**: the kernels
   assume `n_span_bins * span_bin_size >= max_span`. Violating configs
   (e.g. the old test modules' 5 x 20 m vs max_span 200) push span bins
   past `n_span_bins`, aliasing the packed state index into neighboring
   direction/cell states — silent out-of-bounds writes in dense modes
   (`boundscheck=False`). All five public entries now raise ValueError
   on such configs (`_validate_span_bins`); production
   `ConstrainedPathFinder` always derived consistent values and is
   unaffected. The old CPU module tests were corrected (bins 5 -> 10).
   In a 300-seed sweep the aliasing accounted for 95/300 false
   mismatches before the validation existed.
2. **`float('inf')` in Cython returns 1e300, not IEEE inf** — the
   no-path `return_dist` paths now return C `INFINITY`.
3. **Known model approximation (span payload)**: state
   `(cell, dir, span_bin)` stores ONE exact-span float; two same-bin
   labels with different accumulated spans can have crossing utility
   (max_span headroom vs the min_span tower gate). Neither the heap
   nor the bucket kernels are exact under the full
   `(cell, dir, exact span)` model — they break such ties differently.
   Observed once in 300 adversarial seeds (seed 279): the heap's
   ordering finds a degenerate out-and-back route over 0-cost cells
   (span = *traveled* distance, so walking out and back buys the
   min_span tower gate almost for free) that all five bucket kernels
   miss identically. Documented as a strict xfail. A true fix is a
   multi-label search with (dist, span) Pareto dominance — out of
   scope; also note the *model* itself treats span as traveled length,
   which is physically questionable for conductor spans.
4. New testing hooks on the CPU kernels: `force_sparse` (dijkstra +
   height entry) and `return_dist` (all five entries) — both default
   off, return shapes unchanged unless requested.

## 10. V4 window x launch-geometry joint sweep (plan §2 precursor)

`benchmarks/benchmark_gpu_launch_sweep.py`, RTX PRO 500 Blackwell,
full SSSP, median of 3, baseline = production defaults
(window=4, tpb=256, 2 blocks/SM). Results (exact configs only):

| Scenario | best config | speedup vs prod default |
|---|---|---|
| random 1000² | w=16 tpb=512 bpsm=1 | **1.33×** |
| random 2000² | w=16 tpb=256 bpsm=2 | 1.15× |
| random 3000² | w=8  tpb=256 bpsm=2 | 1.10× |
| heavy_tail 1000² | baseline | 1.00× (nothing beats it) |
| heavy_tail 2000² | w=4 tpb=512 bpsm=1 | 1.07× |
| heavy_tail 3000² | baseline | 1.00× |

- **No all-win config: production defaults stay.** Guidance: smooth or
  uniform-random cost surfaces gain 10–33 % from `window=8..16` (with
  tpb ≥ 256); heavy-tail surfaces lose from any window > 4 (work
  amplification dominates — misordering, not occupancy, binds there).
- **Two latent V4 bugs found and fixed by the sweep**:
  1. Queue counters (`CTL_NEAR/FAR/COUNT_A/COUNT_B/PENDING`) are
     atomicAdd-reserved past `buf_size` and were read back *unclamped*
     as loop bounds → out-of-bounds queue reads (reproducible
     `cudaErrorIllegalAddress` at heavy_tail 2000², w=8 tpb=512
     bpsm=2, killing the CUDA context). All count reads are now
     clamped, all pushes go through a `QPUSH` macro that sets a new
     `CTL_OVERFLOW` flag, and an end-of-phase **self-heal** rewinds
     the bucket cursor and rebuilds the frontier from `dist[]`
     (lossless — dropped vertices keep their atomicMin-updated dist).
  2. **512 resident threads/SM is a correctness boundary**: 256×2 and
     512×1 are exact everywhere; 512×2 (=1024/SM) returns wrong dist
     arrays on heavy_tail, and the historical "Bug #3" (256×3) fits
     the same pattern. Root cause still unknown (memory ordering
     scales with resident warps?); the launcher now clamps
     `tpb*blocks_per_sm <= 512` (and tpb to [32, 512]).
- **V5 headroom reading**: synchronous tuning leaves ≤1.33×/1.15×/1.10×
  on random and ~nothing on heavy-tail, where *coarser priority*
  (larger windows) actively hurts. An async V5 therefore should keep
  bucket granularity FINE (delta-sized, not super-bucketed) and win by
  removing phase barriers, not by widening the priority window.

## 11. V5 asynchronous bucket-queue kernel — ACCEPTED (plan §2)

`sssp_raster_gpu_v5` (`sssp_gpu.py`, kernel `sssp_async_v5`), the
ADDS-style async lever, built exactly on the §10 reading: fine
delta-granular priority, barriers removed from the hot path.

**Design** (simplified from ADDS PPoPP'21 for raster workloads):
- 32 delta-granular buckets over a rolling span `[base, base+32)·δ`;
  per-bucket MPMC rings (write/read reservation counters + EMPTY_SLOT
  sentinels, CAS-claimed block-level chunks of 256).
- Hot path: blocks claim → relax (atomicMin + re-queue) → push, atomics
  only. **In-span** improvements queue; **out-of-span** improvements
  only update `dist[]` — no far list, no pending buffer: a
  span-boundary rescan rebuilds the frontier from `dist[]`, which is
  lossless because a vertex whose dist lies past the new span base was
  never relaxed-from at its current dist.
- Quiescence: a single work counter (queued + in-flight; incremented
  BEFORE the slot reservation — the reversed order deadlocks: a fast
  consumer can process-and-decrement an uncounted item, letting another
  block read 0 mid-flight). work==0 ⟹ nothing queued and nobody
  mid-chunk ⟹ every block funnels into the sense-reversing rendezvous
  barrier (V4's Blackwell-safe protocol) which advances the span,
  checks the target bound, or terminates.
- Ring overflow (guarded, rare): drop the push, flag, rendezvous
  rescans from the span base instead of its end. Memory: 3·n int32
  arena + 2×32 counters (vs V4's 4×8·n queues — ~10× less).

**Acceptance run** (`benchmarks/benchmark_gpu_v5.py`, reps=3, full SSSP,
vs V4 production defaults, all bit-exact):

| Scenario | V4 | V5 (chunk=256) | speedup |
|---|---|---|---|
| random 1000² | 38.8 ms | 17.6 ms | **2.21×** |
| random 2000² | 88.7 ms | 54.6 ms | 1.62× |
| random 3000² | 161.2 ms | 158.5 ms | 1.02× |
| heavy_tail 1000² | 34.6 ms | 15.7 ms | **2.20×** |
| heavy_tail 2000² | 93.2 ms | 48.6 ms | 1.92× |
| heavy_tail 3000² | 181.9 ms | 117.4 ms | 1.55× |

**Median 1.77× > 1.15× gate → PASS.** chunk 256 beat 64/128/512/1024
everywhere; default set to 256. Notably V5 wins BIG exactly where V4
tuning gave nothing (heavy-tail): the async fine-priority design avoids
the window's work amplification while still filling the SMs. The weak
spot is random 3000² (parity; frontier there already saturates V4).
Tests: `tests/test_utils/test_sssp_gpu_v5.py` (17 tests — exactness
across families/connectivities/chunks, float32 raster, gradient mode,
targeted early exit, predecessor chains). Same >512 threads/SM clamp as
V4 (shared barrier protocol).

**Status: wired as the production `sssp_raster_gpu` default 2026-08-06**
(dispatch order V5 → V4 → V3, gated by a cached compile check).

**Two async-specific bugs found by the dispatcher's NetworKit test suite
during wiring** (both were invisible in single runs — repetition was the
test; a 30-rep targeted stress regression now guards them):
1. **Post-barrier scratch-reuse race**: the rendezvous target check
   zeroed `C5_MIN_DIST` right after the barrier that had *published gm
   through that same word*. Slower blocks read gm == 0, computed a
   different span base / termination branch, and the blocks' barrier
   sequences diverged (~11 % corrupted targeted runs at 50×50 — full
   runs were always exact because nothing writes that word post-
   barrier). Fix: dedicated `C5_TGT_MAX` scratch word. Rule for any
   barrier-published value: no thread may overwrite the publishing
   word until another barrier confirms every block has consumed it.
2. **pred/dist store race** (latent in V4 too, but its phase barriers
   keep the window tiny): `pred[v] = u` is a plain store after an
   atomicMin win; with concurrent winners the older winner's store can
   land last, leaving pred pointing at the worse predecessor (~12 %
   path-cost error observed via pred-walk reconstruction). Fix: V5
   keeps the hot-path store (race-free for once-improved vertices,
   which covers 0-cost plateaus) and runs a `v5_repair_pred` post-pass
   when predecessors are requested: each pred entry is validated
   against the final dist (recomputed edge weight, small tolerance for
   cross-compilation FMA differences) and inconsistent entries are
   replaced by the best strict-decrease incoming edge. Only runs with
   `return_predecessor=True`; full-SSSP benchmark numbers unchanged.

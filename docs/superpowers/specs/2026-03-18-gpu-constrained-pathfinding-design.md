# GPU Constrained Pathfinding with Warp-Cooperative Clearance

**Date:** 2026-03-18
**Status:** Draft
**Author:** Claude + User

## Summary

A CUDA persistent cooperative kernel that performs the entire constrained overhead-line routing algorithm in a single GPU launch. Jointly optimizes route, tower placement, tower angle, tower height, and tower type on a 3D surface with catenary clearance checking. Uses warp-cooperative parallelism for tower placement operations (clearance, area cost, slope cost) to eliminate warp divergence waste.

## Context

The existing Cython constrained delta-stepping (`_constrained_delta.pyx`) is the performance bottleneck for large rasters. The project already has a proven GPU architecture:

- **V4 persistent cooperative kernel** (`sssp_gpu.py`): single-launch delta-stepping for unconstrained SSSP, 4-20x faster than Cython
- **GPU constrained SSSP v1** (`constrained_sssp_gpu.py`): Python-loop delta-stepping with angle limits + span bins, but no clearance/heights/area cost

This design extends the v4 persistent kernel architecture to the full constrained problem.

## Goals

- **Feature parity** with the Cython constrained pathfinder: angle limits, span constraints, variable tower heights, catenary clearance, tower ground area cost (exact rotated mode), slope-dependent foundation cost, forbidden footprint rejection
- **Single kernel launch**: no Python-loop overhead, no kernel re-launches
- **5-20x speedup** over Cython for 1000x1000+ rasters
- **Fits 16 GB VRAM** for R3 neighborhoods on 2000x2000 rasters

## Non-Goals

- Sparse/lazy state allocation on GPU (future work if larger rasters needed)
- Multi-GPU support
- Automatic neighborhood downgrading

## Invariants

- **All costs are non-negative.** The atomic relaxation uses `atomicMin` on float32 via IEEE 754 bit-cast (`__float_as_int`), which is only monotonically ordered for non-negative values. The host validates this before kernel launch: all terrain costs, angle costs, height premiums, and gradient penalties must be >= 0.

---

## Architecture

### Kernel State Machine

The persistent kernel runs a complete delta-stepping algorithm in a single launch:

```
PERSISTENT KERNEL (single launch, runs to completion)
|
+-- Load shared memory (step LUTs, angle LUTs, catenary params)
|
+-- OUTER LOOP (per bucket)
     |
     +-- Fill frontier from pending/classify queues
     |
     +-- LIGHT PHASE (up to 100 inner iterations)
     |    |
     |    +-- For each frontier item (strided across threads):
     |         +-- Unpack state -> (cell, dir, span_bin, height_class)
     |         +-- For each valid neighbor:
     |         |    +-- Check intermediate path validity
     |         |    +-- Compute edge cost (terrain + angle + gradient)
     |         |    |
     |         |    +-- BRANCH A: Continue span (no tower)
     |         |    |    +-- Atomic relaxation -> queue_b
     |         |    |
     |         |    +-- BRANCH B: Place tower (span >= min_span)
     |         |         +-- WARP-COOPERATIVE TOWER PROTOCOL
     |         |         +-- For ALL feasible height classes: clearance + area + slope
     |         |         +-- Owner thread writes relaxation for each feasible height
     |         |
     |         +-- Queue swap (queue_a <-> queue_b)
     |
     +-- HEAVY PHASE (single pass over settled nodes with edge weight > delta)
     |    Same relaxation logic as light phase, but only processes edges
     |    whose terrain cost exceeds delta. Tower placement branches are
     |    included — tower costs are typically "heavy" edges.
     |
     +-- Bucket advance + classify pending
     |
     +-- Early termination check (every 10 buckets)
```

### Source Initialization

Before the kernel launches, the host seeds the frontier with source states:

```python
# All directions, span_bin=0, all height classes at the source cell
for d in range(n_dirs):
    for h in range(n_heights):
        state = pack_state(source_cell, d, 0, h, spc, n_span_bins, n_heights)
        d_dist[state] = height_premiums[h]  # taller source towers cost more
        d_span_dist[state] = 0.0
        queue_a.append(state)
```

The initial distance is `height_premiums[h]`, not 0 — matching the Cython behavior where starting with a taller tower at the source incurs a cost premium.

### Warp-Cooperative Tower Protocol

The key innovation. When threads in a warp want to place towers, the entire warp cooperates on each placement instead of each thread working independently.

**Problem solved:** In naive inline clearance (Approach A), if 5 of 32 warp threads place towers, the other 27 idle during the 30-iteration clearance loop = 84% wasted compute. In a two-phase approach (Approach B), a grid barrier + candidate buffer adds latency and complexity.

**Solution:** Round-robin through tower requests within the warp. For each request, all 32 threads cooperate on clearance checking, area cost summation, and slope computation.

```
1. unsigned active = __activemask()
   tower_mask = __ballot_sync(active, want_tower)
2. While tower_mask != 0:
   a. owner = __ffs(tower_mask) - 1  (pick next tower thread, always >0 since mask!=0)
   b. __shfl_sync(active, ...): broadcast tower params to all 32 threads
   c. For EACH height class (tallest first):
      - PARALLEL CLEARANCE: each thread checks ceil(span_cells/32) cells
        * Compute catenary sag at assigned position
        * __ballot_sync(active, ...): all cells must pass
        * If clearance FAILS: skip remaining shorter heights (early rejection)
      - PARALLEL AREA COST: each thread sums ceil(n_offsets/32) pixels
        * Check forbidden (65535) via __ballot_sync
        * Warp-reduce sum via __shfl_down_sync
      - PARALLEL SLOPE: each thread computes slope for assigned pixels
        * Warp-reduce average via __shfl_down_sync
      - Owner thread: compute final cost, write atomic relaxation + TowerRecord
   d. tower_mask &= tower_mask - 1  (clear processed bit)
```

**Height exploration:** ALL feasible heights are explored (each generating a separate relaxation with its cost), matching Cython behavior. The early-exit optimization is on FAILURE: if the tallest height fails clearance, all shorter heights are skipped (they would also fail). But if the tallest succeeds, shorter heights are still checked — a shorter, cheaper tower may be part of the globally optimal solution.

**`__activemask()` usage:** The protocol uses `__activemask()` instead of `0xFFFFFFFF` because threads may have exited early from the neighbor loop (no valid neighbors, already visited, etc.), making the warp divergent. Using a full mask would be incorrect.

**Performance properties:**
- Zero warp divergence waste: all 32 threads do useful work
- Zero grid barriers: warp-level __shfl_sync + __ballot_sync only
- Zero extra buffers: no candidate buffer, no separate phase
- 30-cell clearance in ceil(30/32) = 1 warp cycle instead of 30 sequential
- 100-pixel area cost in ceil(100/32) = 4 warp cycles instead of 100 sequential

### Custom Grid Barrier

Inherited from v4. Replaces broken `grid.sync()` on Blackwell/sm_120:

```cuda
__device__ void grid_barrier(volatile int* control, int n_blocks) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int my_sense = control[CTL_BARRIER_SENSE];
        int arrived = atomicAdd(&control[CTL_BARRIER_CNT], 1) + 1;
        if (arrived == n_blocks) {
            control[CTL_BARRIER_CNT] = 0;
            __threadfence();
            control[CTL_BARRIER_SENSE] = 1 - my_sense;
        } else {
            while (control[CTL_BARRIER_SENSE] == my_sense) {}
        }
    }
    __syncthreads();
}
```

Required compiler flag: `-Xptxas -dlcm=cg` (bypass L1 cache for global loads).

---

## State Encoding

4-dimensional state packed into int64:

```
state = cell * spc + dir * (n_span_bins * n_heights) + span_bin * n_heights + height_class

Where:
  spc = n_dirs * n_span_bins * n_heights  (states per cell)
  cell = row * cols + col                 (0..rows*cols-1)
  dir = direction index                   (0..n_dirs-1)
  span_bin = floor(span_m / bin_size)     (0..n_span_bins-1)
  height_class = tower height index       (0..n_heights-1)

Note: n_dirs is implicitly spc / (n_span_bins * n_heights). The kernel
receives spc, n_span_bins, n_heights as separate parameters; n_dirs
is derived or passed explicitly.
```

Unpacking (CUDA device inline):

```cuda
__device__ __forceinline__ void unpack_state(
    long long state, int spc, int n_dirs, int n_span_bins, int n_heights, int cols,
    int* cell, int* row, int* col, int* dir, int* span_bin, int* height_class
) {
    *cell = (int)(state / spc);
    int rem = (int)(state % spc);
    *dir = rem / (n_span_bins * n_heights);
    rem = rem % (n_span_bins * n_heights);
    *span_bin = rem / n_heights;
    *height_class = rem % n_heights;
    *row = *cell / cols;
    *col = *cell % cols;
}
```

---

## Memory Layout

### GPU Global Memory

| Array | Type | Elements | Purpose |
|-------|------|----------|---------|
| d_raster | uint16 | rows*cols | Cost grid |
| d_dem | float32 | rows*cols | Ground elevation |
| d_obstacle | float32 | rows*cols | Obstacle heights (DSM-DGM), NULL if none |
| d_exclude_mask | uint8 | rows*cols | Traversable mask |
| d_tower_terrain | float32 | 65536 | Raster value -> EUR lookup |
| d_tower_heights | float32 | n_heights | Sorted descending |
| d_height_premiums | float32 | n_heights | Cost premium per height |
| d_area_offsets | int32 | total_pairs*2 | Flat (dr,dc) for all direction pairs, row-major: pair_idx = d_in * n_dirs + d_out |
| d_area_starts | int32 | n_dirs^2 | Start index per (d_in, d_out) into d_area_offsets |
| d_area_counts | int32 | n_dirs^2 | Offset count per pair |
| **d_dist** | **float32** | **total_states** | **Distance array (dense)** |
| **d_span_dist** | **float16** | **total_states** | **Exact span distance (meters)** |
| d_queue_a | int64 | buf_size | Current frontier |
| d_queue_b | int64 | buf_size | Output / swap |
| d_pending | int64 | buf_size | Cross-bucket pending |
| d_settled | int64 | buf_size | Settled for heavy phase |
| d_tower_records | TowerRecord | max_records | Tower placement log (atomic-append) |
| d_tower_count | int32 | 1 | Atomic append counter |
| d_control | int32 | 16 | Volatile control buffer |

**No per-state predecessor array (d_pred) is stored.** This is a deliberate departure from the v1 GPU code and Cython code, which store int64 predecessors for every state (18.4 GB for the reference budget). Instead, path reconstruction uses the sparse TowerRecord buffer and direction-walking between towers (see Path Reconstruction section).

### TowerRecord Structure

```cuda
struct __align__(8) TowerRecord {
    long long state;        // placed tower state (packed)
    long long pred_state;   // predecessor tower state
    __half    span_dist;    // exact span (meters)
    __half    tower_height; // selected height (meters)
};  // 24 bytes (20 + 4 padding from __align__(8))
```

Appended atomically by the owner thread after successful tower relaxation:

```cuda
// Only record when this relaxation actually improved the distance
int old_bits = atomicMin((int*)&dist[new_state], __float_as_int(new_dist));
float old_dist = __int_as_float(old_bits);
if (new_dist < old_dist) {
    int idx = atomicAdd(d_tower_count, 1);
    if (idx < max_tower_records) {
        d_tower_records[idx] = {new_state, cur_state,
                                __float2half(span_m), __float2half(height_m)};
    } else {
        control[CTL_TOWER_OVERFLOW] = 1;
    }
}
```

### d_span_dist float16 Precision

float16 has ~3.3 decimal digits of precision. At 400m span, resolution is ~0.4m. The catenary sag error from this is on the order of centimeters — acceptable for practical clearance checking. However, this means the GPU may make marginally different clearance decisions than the Cython reference (which uses float32). Phase 1 validation should use a 0.5m tolerance on span distances and 1% tolerance on clearance-derived costs.

### Memory Budget (2000x2000, R3, 6 span bins, 3 heights)

| Component | Size |
|-----------|------|
| d_dist (4M * 32 * 6 * 3 * 4B) | 9.2 GB |
| d_span_dist (same * 2B) | 4.6 GB |
| Input data + LUTs | 40 MB |
| Queues (4 * 16M * 8B) | 512 MB |
| Tower records (1M * 24B) | 24 MB |
| **Total** | **~14.4 GB** |

Fits 16 GB VRAM with 1.6 GB headroom.

### Shared Memory (per block)

| Data | Formula | Size (R3, 32 dirs) |
|------|---------|-------------------|
| s_steps | n_dirs * 2 | 64 B |
| s_cost_factors | n_dirs * 4 | 128 B |
| s_step_dist | n_dirs * 4 | 128 B |
| s_angle_cost | n_dirs^2 * 4 | 4096 B |
| s_angle_valid | n_dirs^2 * 1 | 1024 B |
| s_tower_angle | n_dirs^2 * 4 | 4096 B |
| s_inter_lut | n_dirs * max_inter * 2 | 960 B |
| s_n_inter | n_dirs * 4 | 128 B |
| s_tower_heights + premiums | n_heights * 4 * 2 | n_heights * 8 B |
| s_catenary_params | 4 * 4 | 16 B |
| **Total** | | **~10.6 KB** |

Well under the 96 KB per-block limit.

### Launch Configuration

```python
n_sms = device.multiProcessorCount   # 14 for RTX PRO 500
blocks = n_sms * 2                   # 28 blocks (proven stable limit)
threads_per_block = 256               # 8 warps per block
shared_mem = compute_shared_size(n_dirs, max_inter_cols, n_heights)
```

---

## Catenary Clearance Model

Parabolic sag approximation (same as Cython):

```
sag(x) = w * x * (L - x) / (2 * T)

conductor_z(x) = chord_z(x) - sag(x)
chord_z(x) = attach_a + (attach_b - attach_a) * x / L
attach_a = DEM[tower_A] + height_A
attach_b = DEM[tower_B] + height_B

clearance_ok = conductor_z(x) - DEM[x] - obstacle[x] >= min_clearance
```

Where:
- `w` = conductor weight per meter (N/m)
- `T` = conductor tension (N)
- `L` = span length (meters)
- `x` = distance along span from tower A

In the warp-cooperative protocol, each of 32 threads checks a different position along the span. `__ballot_sync` collects pass/fail across all positions in one cycle.

---

## Tower Cost Model

Total tower placement cost at cell C with direction pair (d_in, d_out):

```
tower_cost = area_terrain_cost * slope_multiplier + angle_type_cost + height_premium

Where:
  area_terrain_cost = sum of tower_terrain_lut[raster[pixel]] for all pixels
                      in the rotated square footprint centered on C
                      (INFINITY if any pixel == 65535)

  slope_multiplier  = exp(gradient_scale * avg_slope_pct / 100)
  avg_slope_pct     = mean of per-pixel max-neighbor slope over the footprint

  angle_type_cost   = tower_angle_lut[d_in, d_out]
  height_premium    = height_premiums[height_class]
```

All three components (area cost, slope, forbidden check) computed cooperatively by the warp using the same rotated square pixel offsets.

---

## Control Buffer

```
Index  Name               Purpose
0      CTL_COUNT_A        Frontier size in queue_a
1      CTL_COUNT_B        Output count in queue_b
2      CTL_SETTLED        Settled count for heavy phase
3      CTL_PENDING        Persistent pending count
4      CTL_NEAR           Near-bucket classify count
5      CTL_FAR            Far-bucket classify count
6      CTL_BUCKET         Current bucket index
7      CTL_DONE           Termination flag
8      CTL_EARLY_CTR      Early termination check counter
9      CTL_MIN_DIST       Min target distance (bit-cast float)
10     CTL_BARRIER_CNT    Custom barrier arrival counter
11     CTL_BARRIER_SENSE  Barrier sense bit
12     CTL_TOWER_OVERFLOW Tower record buffer overflow flag
13     CTL_QUEUE_OVERFLOW Queue buffer overflow flag
```

---

## Path Reconstruction (CPU, Python)

**No per-state predecessor array is needed.** Between consecutive towers, the direction is constant (constrained Dijkstra only places towers at direction changes). The path between towers can be deterministically reconstructed by walking forward in the known direction for the known span distance.

After kernel completes:

1. **Find best target state**: scan `d_dist[target_cell * spc .. (target_cell+1) * spc]` for minimum — at most `spc` entries (e.g., 576), trivial.

2. **Copy tower records to host**: `d_tower_records[:d_tower_count].get()`

3. **Build state-to-record map**: dict mapping `state -> TowerRecord` (keep only the record whose `pred_state` is consistent with the final `d_dist` values). Multiple records may exist for the same state if it was relaxed multiple times; pick the one where `d_dist[state]` matches.

4. **Walk backward** from best target through tower chain via `pred_state` links until reaching a source state.

5. **Reconstruct cell-by-cell path** between consecutive towers:
   ```python
   for i in range(len(towers) - 1):
       tower_a = towers[i]
       tower_b = towers[i + 1]
       cell_a, dir_b, _, _ = unpack_state(tower_b.state)
       cell_b_row, cell_b_col = cell_a // cols, cell_a % cols
       # Walk backward from tower_b in direction dir_b
       # Each step: row -= steps[dir_b][0], col -= steps[dir_b][1]
       # Until reaching tower_a's cell
       # (span_dist / step_distance gives exact step count)
   ```

6. **Return** `(path_indices, tower_indices, tower_heights)` — same interface as Cython

---

## Edge Cases

| Case | Handling |
|------|----------|
| No feasible path | Kernel terminates on empty frontier+pending. Host checks dist[target] == INF |
| Tower record buffer overflow | CTL_TOWER_OVERFLOW flag set. Host re-runs with 2x buffer or raises error |
| Queue buffer overflow | CTL_QUEUE_OVERFLOW flag set. Host warns about potential suboptimality |
| All heights fail clearance | Tower placement rejected. Must find alternative route |
| No DEM provided | dem/obstacle pointers NULL. Clearance skipped, slope_multiplier = 1.0 |
| Area offsets NULL (uniform mode) | Warp-cooperative protocol falls back to single-pixel lookup |
| Source/target on forbidden cell | Checked on host before launch. Raises ValueError |

---

## Python Wrapper

New file: `pyorps/utils/constrained_sssp_gpu_v2.py`

```python
def constrained_sssp_raster_gpu_v2(
    raster, source_row, source_col, target_row, target_col,
    steps, angle_cost_lut, angle_valid_lut, step_distances,
    tower_terrain_costs, tower_angle_costs,
    n_span_bins, span_bin_size, min_span, max_span,
    # 3D / clearance (optional)
    dem_data=None, obstacle_heights=None, cell_size=1.0,
    tower_heights=None, height_premiums=None,
    conductor_weight_per_m=0.0, conductor_tension=1.0,
    min_clearance=0.0, max_gradient_pct=100.0, gradient_scale=2.0,
    # Area cost (optional)
    area_offsets=None, area_offset_starts=None, area_offset_counts=None,
    exclude_mask=None,
):
    """GPU persistent-kernel constrained pathfinding with full 3D support."""
```

Integrated into `ConstrainedPathFinder._find_route_coupled()` via the existing `backend == "raster_gpu"` code path.

---

## Validation Strategy

**Phase 1 — Correctness (small rasters, match vs Cython):**
- 50x50 uniform raster, no DEM: path + towers must match Cython exactly
- 50x50 with DEM + obstacles: clearance decisions must match (0.5m span tolerance for float16)
- 50x50 with variable heights: height selection must match
- 50x50 with area cost (exact mode): tower terrain costs must match
- 100x100 with forbidden cells in footprint: towers must avoid them
- Edge cases: no feasible path, single-cell path, source == target

**Phase 2 — Optimality (medium rasters, cost comparison):**
- 500x500: GPU total_cost within 0.1% of Cython total_cost (1% for clearance-heavy routes due to float16)
- Tower count and positions within 1 cell difference
- Heights and tower types must match

**Phase 3 — Performance (large rasters, benchmark):**
- 1000x1000 R3: measure speedup vs Cython
- 2000x2000 R3: target 5-20x faster than Cython
- 3000x3000 R3: verify fits in VRAM, completes without OOM

---

## Known Gotchas (from prior GPU work)

- **Blackwell sm_120**: `grid.sync()` does not provide memory ordering. Use custom barrier with `__threadfence()` + sense-reversing protocol.
- **L1 cache**: Not invalidated by barriers on Blackwell. Compile with `-Xptxas -dlcm=cg`.
- **Block count**: Limit to 2 blocks/SM (28 on 14-SM GPU). Higher counts cause incorrect results.
- **`__ballot_sync` mask**: Use `__activemask()` instead of `0xFFFFFFFF` in divergent code paths (e.g., inside the neighbor loop where some threads may have exited early).
- **CuPy kernel caching**: `cp.RawKernel()` compiles lazily on first `.kernel` access. Force compilation in availability check.
- **`volatile int*`**: Required for control buffers read by spinning threads.
- **Non-negative costs**: `atomicMin` via bit-cast only works for non-negative float32. Validate on host before launch.

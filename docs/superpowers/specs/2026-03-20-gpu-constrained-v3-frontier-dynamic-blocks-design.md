# GPU Constrained Pathfinding V3: Frontier-Based Delta-Stepping with Dynamic Block Allocation

**Date:** 2026-03-20
**Status:** Approved
**Supersedes:** V2 persistent cooperative kernel (retained as fallback)

## Problem

The V2 GPU constrained pathfinding uses a single-launch persistent cooperative kernel. On a 6 GB GPU (RTX PRO 500 Blackwell), dense storage exceeds VRAM for real-world rasters with R3+ neighborhoods (13 GB for R4 on 1433x1834). Alternative storage backends in V2 either crash (sparse hash overflow), are slower than Cython (managed memory: 191s vs 148s), or have a correctness bug (block-sparse: finds 0 paths).

The persistent kernel also suffers from 2% GPU utilization because constrained routing produces small, sparse frontiers — 98% of threads spin-wait at grid barriers.

## Solution

Replace the persistent cooperative kernel with **frontier-based delta-stepping** (per-bucket kernel launches from Python) and **dynamic block-sparse storage** (blocks allocated on demand when a cell is first visited).

### Design Principles

- All heavy computation in CUDA — Python loop is only kernel dispatch (~10us overhead per iteration)
- Memory scales with visited cells, not total cells
- BLOCK_SIZE adapts to available VRAM — larger GPUs get larger blocks (less/no eviction)
- Reuse existing domain-specific CUDA: clearance, area cost, tower placement, angle penalties
- V2 code untouched — V3 is independent, wired as a new backend

## Architecture

### Kernel Architecture: Multi-Launch Frontier-Based Delta-Stepping

```
Python driver loop:
    while frontier_size > 0:
        1. Launch relax_kernel(frontier, frontier_size, phase=LIGHT)
           - Phase A: per-thread non-tower relaxation (same-dir span continuation)
           - Phase B: warp-cooperative tower placement (area cost, clearance)
           - Writes new states to output queue (within bucket) or pending (outside)
           - Each dequeued frontier item is appended to settled queue (once)
        2. Read back output_count (single int32 D2H copy)
        3. Swap queues, repeat until output_count == 0 (light phase done)
        4. Check early termination: read target state distances, stop if settled
        5. Launch relax_kernel(settled, settled_count, phase=HEAVY)
        6. Advance bucket, launch classify_kernel on pending
        7. If no pending, launch scan_kernel for next unsettled minimum
        8. If scan finds nothing → done
```

Grid size for each launch = `ceil(frontier_size / threads_per_block)` — threads match actual work.

### Storage: Dynamic Block-Sparse with On-Demand Allocation

```
cell_to_block[n_cells]                  — int32, -1 = unvisited (~10 MB for 2.6M cells)
block_to_cell[max_blocks]               — int32, reverse map for scan/extract (~1 MB for 260K blocks)
block_pool[max_blocks * BLOCK_SIZE]     — BlockEntry, 8 bytes each
span_pool[max_blocks * BLOCK_SIZE]      — float16, 2 bytes each
n_allocated_blocks                      — atomic int32 counter
```

**Block allocation (lock-free, idempotent):**

```cuda
__device__ int get_block(int cell, int* cell_to_block, int* block_to_cell,
                         int* n_allocated, int max_blocks) {
    int idx = cell_to_block[cell];
    if (idx >= 0) return idx;                              // fast path: already allocated

    int new_idx = atomicAdd(n_allocated, 1);
    if (new_idx >= max_blocks) return -1;                  // pool exhausted

    int old = atomicCAS(&cell_to_block[cell], -1, new_idx);
    if (old == -1) {
        block_to_cell[new_idx] = cell;                     // write reverse map
        return new_idx;                                    // we won the race
    }
    // Another thread won — our new_idx is wasted (block stays at init state:
    // all entries BLOCK_EMPTY/1e30, safe for scan/extract to skip).
    // block_to_cell[new_idx] remains -1 (init value), so extract_bucket
    // skips it when reconstructing states.
    return old;                                            // use winner's block
}
```

Race-wasted blocks are bounded by n_cells (at most one per cell). The `init_pool` kernel initializes the **entire** pool (all max_blocks * BLOCK_SIZE entries) to BLOCK_EMPTY/1e30, so wasted blocks contain no valid entries. `block_to_cell` is initialized to -1; only the race winner writes the actual cell index.

**Caller responsibility for pool exhaustion:** When `get_block()` returns -1, the calling thread **must skip the relaxation** (drop the edge) and increment `CTL_OVERFLOW` to signal pool exhaustion. The Python driver checks `CTL_OVERFLOW` after each bucket and warns the user if pool capacity was exceeded.

**BlockEntry struct (unchanged from V2):**

```cuda
struct __align__(8) BlockEntry {
    unsigned short local_key;   // dir * (n_span_bins * n_heights) + span_bin * n_heights + height
    unsigned short _pad;
    float dist;                 // best distance (init 1e30)
};
// Span stored separately: span_pool[block_index * BLOCK_SIZE + slot] (float16)
```

Within each block, open-addressing hash with multiplicative hash on local_key (same as V2). BLOCK_SIZE is a power of 2, injected at compile time.

**Dynamic block operations (`block_find_dyn`, `block_upsert_dyn`, `block_relax_dyn`):**
Same as V2's `block_find`/`block_upsert`/`block_relax` but with base offset = `get_block(cell) * BLOCK_SIZE` instead of `cell * BLOCK_SIZE`. The internal hash probing, eviction, and span-write logic is identical.

**BLOCK_SIZE selection:**

```python
target_bs = next_power_of_2(spc)                # ideal: no eviction
available_vram = total_vram - 1 GB              # reserve for input + queues + tower records
max_blocks = estimated_visited_cells * 1.5      # headroom for races
while target_bs * max_blocks * 10 > available_vram:
    target_bs //= 2                             # halve until it fits
block_size = max(32, target_bs)
```

The 10% visitation estimate should be validated empirically. `max_blocks` is configurable via a `max_visited_fraction` parameter (default 0.15). The Python driver checks `n_allocated` after completion and warns if it reached >90% of `max_blocks`.

When BLOCK_SIZE >= spc, no eviction occurs. When BLOCK_SIZE < spc, the block_relax eviction policy (evict highest-dist entry) applies.

### Memory Budget

| Config | spc | 6 GB GPU (BS / pool) | 16 GB GPU | 24 GB GPU |
|--------|-----|----------------------|-----------|-----------|
| R2, 3h | 288 | BS=512, 1.33 GB | BS=512, 1.33 GB | BS=512, 1.33 GB |
| R3, 3h | 576 | BS=512, 1.33 GB | BS=1024, 2.66 GB | BS=1024, 2.66 GB |
| R4, 3h | 1728 | BS=256, 666 MB | BS=1024, 2.66 GB | BS=2048, 5.32 GB |
| R4, 5h | 2880 | BS=128, 333 MB | BS=1024, 2.66 GB | BS=4096, 10.6 GB |

Pool sizes assume 10-15% cell visitation (260-390K blocks for 2.6M cell raster). Actual usage is lower (pool is maximum, not typical). The `cell_to_block` map adds ~10 MB, `block_to_cell` adds ~1 MB (both constant).

## CUDA Kernels

### Kernel 1: `relax_constrained` (main workhorse)

**Signature:**
```
relax_constrained(frontier, frontier_count, phase, buf_size,
                  raster, rows, cols, max_cost,
                  steps, cost_factors, inter_lut, n_inter, n_steps, max_inter_cols,
                  angle_cost_lut, angle_valid_lut, step_distances,
                  tower_terrain_lut, tower_angle_lut,
                  height_premiums, tower_heights, n_heights,
                  n_span_bins, span_bin_size, min_span_bin, spc, total_states,
                  cell_to_block, block_to_cell, block_pool, span_pool,
                  n_allocated, max_blocks,
                  delta, output_queue, control,
                  pending_queue, settled_queue,
                  tower_records, max_tower_records,
                  dem, obstacle, cell_size,
                  cond_weight, cond_tension, min_clearance,
                  max_gradient_pct, gradient_scale,
                  area_offsets, area_starts, area_counts)
```

**`phase` parameter:** 0 = light (process edges with cost <= delta), 1 = heavy (process edges with cost > delta).

**Processing pattern (per warp, 32 threads):**

1. Each lane loads one frontier item (my_idx = batch_start + lane)
2. Unpack state → (cell, dir, span_bin, height_class)
3. `get_block(cell)` to resolve block index; if -1, skip (pool exhausted, increment CTL_OVERFLOW)
4. Read current dist from block
5. Append this frontier item to the settled queue (for heavy phase processing later). Each dequeued frontier item enters settled exactly once — when first loaded from the frontier in the light phase kernel.
6. **Phase A (per-thread):** Each thread independently processes all same-direction neighbors. For each valid neighbor: compute terrain cost + gradient + angle penalty. If same direction and cost within phase range: compute new span_bin, `get_block(nb_cell)` (skip if -1), `block_relax_dyn()`. On improvement: append to output or pending queue. **All queue appends check atomicAdd result against buf_size; if full, increment CTL_OVERFLOW and skip the write.**
7. **Phase B (warp-cooperative tower placement):** Lanes with span >= min_span ballot. Round-robin: for each owner lane, all 32 lanes cooperate on area cost summation, forbidden pixel check, slope reduction, catenary clearance. Lane 0 writes relaxation + TowerRecord on improvement. Same buf_size guard on queue writes.

This is the same Phase A + Phase B logic as the existing V2 kernel, with:
- `get_block()` + `block_to_cell` replacing `block_offset(cell)`
- `block_relax_dyn()` using dynamic base offset
- No persistent loop, no grid barriers
- Explicit pool-exhaustion and queue-overflow guards

**Shared memory:** Same layout as V2 (steps, n_inter, cost_factors, step_distances, angle_cost, angle_valid, tower_angle, height_premiums, tower_heights, inter_lut).

### Kernel 2: `classify_bucket`

```
classify_bucket(pending, pending_count, bucket, delta, buf_size,
                cell_to_block, block_pool, spc, n_span_bins, n_heights,
                near_queue, far_queue, control)
```

Per-thread: read one pending state, look up distance via `get_block()` + `block_find_dyn()`. Classification:
- dist within [bucket\*delta, (bucket+1)\*delta): write to near_queue (check buf_size).
- dist >= (bucket+1)\*delta: write to far_queue (check buf_size).
- dist < bucket\*delta: **discard** — state already settled in a previous bucket.
- dist >= 1e30: **discard** — state was superseded by a better relaxation.

### Kernel 3: `scan_min_dist`

```
scan_min_dist(block_pool, block_to_cell, n_allocated_blocks,
              bucket_lower_bound, control)
```

Scans only allocated blocks (n_allocated_blocks * BLOCK_SIZE entries) to find the minimum distance >= bucket_lower_bound. Skips blocks where `block_to_cell[block_idx] == -1` (wasted race slots). Each thread processes a strided subset, does local min, then atomicMin to global. Much faster than V2's full-state-space scan because unvisited cells have no blocks.

### Kernel 4: `extract_bucket`

```
extract_bucket(block_pool, block_to_cell, n_allocated_blocks,
               bucket, delta, spc, n_span_bins, n_heights,
               output_queue, control, buf_size)
```

After scan_min_dist finds the next bucket, this kernel extracts all states within that bucket range from the allocated blocks into the frontier queue.

**State reconstruction:** For each block entry with matching dist:
```cuda
int block_idx = i / BLOCK_SIZE;
int cell = block_to_cell[block_idx];        // reverse map lookup
if (cell < 0) continue;                      // wasted race slot
unsigned short lk = block_pool[i].local_key;
if (lk == BLOCK_EMPTY) continue;
int dir = lk / (n_span_bins * n_heights);
int rem = lk % (n_span_bins * n_heights);
int sb = rem / n_heights;
int hc = rem % n_heights;
long long state = (long long)cell * spc + dir * sh + sb * n_heights + hc;
```

Queue writes check buf_size as with all other kernels.

### Kernel 5: `init_pool` + `init_source`

```
init_pool(block_pool, max_total_entries)        // set local_key=0xFFFF, dist=1e30
                                                // covers ALL max_blocks * BLOCK_SIZE entries
init_source(block_pool, span_pool, cell_to_block, block_to_cell,
            n_allocated, source_states, init_dists, n_source,
            spc, n_span_bins, n_heights, max_blocks)
```

`init_pool` initializes the **entire** pre-allocated pool so that race-wasted blocks and unoccupied slots are safe for scan/extract to encounter.

`init_source` allocates blocks for source cells via `get_block()` and inserts source states.

## Python Driver

```python
def constrained_sssp_raster_gpu_v3(raster, source, target, ...,
                                    max_visited_fraction=0.15):
    # --- Setup ---
    # Compute BLOCK_SIZE from spc and VRAM
    # Compute max_blocks from max_visited_fraction and VRAM
    # Allocate: cell_to_block (int32, n_cells, memset -1 via 0xFF)
    #           block_to_cell (int32, max_blocks, memset -1 via 0xFF)
    #           block_pool (BlockEntry, max_blocks * BLOCK_SIZE)
    #           span_pool (float16, max_blocks * BLOCK_SIZE)
    #           n_allocated (int32, scalar, init 0)
    #           queue_a, queue_b, settled, pending (int64, buf_size each)
    #           control (int32, CTL_SIZE entries, init 0)
    #           tower_records (TowerRecord, max_tower_records)
    # Launch init_pool (covers full pool)
    # Launch init_source (allocates source blocks, inserts source states)
    # Seed frontier queue with source states

    bucket = 0
    while True:
        # --- Reset counters for this bucket ---
        d_control[CTL_OUTPUT] = 0
        d_control[CTL_SETTLED] = 0
        d_control[CTL_PENDING] = 0

        # --- Light phase ---
        while frontier_count > 0:
            grid = ceildiv(frontier_count, TPB)
            relax_kernel((grid,), (TPB,), (frontier, frontier_count, LIGHT, ...),
                         shared_mem=smem_bytes)
            cp.cuda.Stream.null.synchronize()
            frontier_count = int(d_control[CTL_OUTPUT].get())
            d_control[CTL_OUTPUT] = 0
            frontier, output = output, frontier  # swap

        # --- Early termination check ---
        # Read target cell's block, find best dist across all (dir, sb, hc)
        # If best_target_dist < best_known * margin: done
        target_block_idx = int(d_cell_to_block[target_cell].get())
        if target_block_idx >= 0:
            # Small kernel or CPU check of target block entries
            # If target settled: break

        # --- Heavy phase ---
        settled_count = int(d_control[CTL_SETTLED].get())
        if settled_count > 0:
            d_control[CTL_OUTPUT] = 0  # heavy output goes here too
            grid = ceildiv(settled_count, TPB)
            relax_kernel((grid,), (TPB,), (settled, settled_count, HEAVY, ...),
                         shared_mem=smem_bytes)
            cp.cuda.Stream.null.synchronize()

            # Check if heavy phase produced within-bucket states.
            # If so, feed them back into the light phase (don't advance bucket).
            heavy_output = int(d_control[CTL_OUTPUT].get())
            if heavy_output > 0:
                frontier_count = heavy_output
                d_control[CTL_OUTPUT] = 0
                d_control[CTL_SETTLED] = 0
                frontier, output = output, frontier
                continue  # back to light phase, same bucket

        # --- Advance bucket ---
        bucket += 1
        pending_count = int(d_control[CTL_PENDING].get())

        if pending_count > 0:
            d_control[CTL_NEAR] = 0
            d_control[CTL_FAR] = 0
            grid = ceildiv(pending_count, TPB)
            classify_kernel((grid,), (TPB,), (pending, pending_count, bucket, delta, ...))
            cp.cuda.Stream.null.synchronize()
            frontier_count = int(d_control[CTL_NEAR].get())
            far_count = int(d_control[CTL_FAR].get())
            if frontier_count > 0:
                # Copy far back to pending, reset counters
                # (pending and far share a buffer via double-buffering)
                continue
            # Empty bucket — advance until we find states or pending exhausted
            if far_count > 0:
                pending_count = far_count
                continue

        # --- Full scan fallback ---
        n_alloc = int(d_n_allocated.get())
        if n_alloc == 0:
            break
        d_control[CTL_MIN_DIST] = int_as_float_bits(1e30)
        scan_kernel((...), (...), (block_pool, block_to_cell, n_alloc,
                                    bucket * delta, d_control))
        cp.cuda.Stream.null.synchronize()
        min_dist = read_float_from_control(d_control, CTL_MIN_DIST)
        if min_dist >= 1e29:
            break  # no path

        bucket = int(min_dist / delta)
        d_control[CTL_NEAR] = 0
        extract_kernel((...), (...), (block_pool, block_to_cell, n_alloc,
                                      bucket, delta, spc, ...))
        cp.cuda.Stream.null.synchronize()
        frontier_count = int(d_control[CTL_NEAR].get())
        if frontier_count == 0:
            break

    # --- Check pool usage ---
    overflow = int(d_control[CTL_OVERFLOW].get())
    n_used = int(d_n_allocated.get())
    if overflow > 0:
        warnings.warn(f"Block pool exhausted: {overflow} edges dropped. "
                      f"Increase max_visited_fraction (used {n_used}/{max_blocks}).")
    if n_used > max_blocks * 0.9:
        warnings.warn(f"Block pool >90% full ({n_used}/{max_blocks}). "
                      f"Consider increasing max_visited_fraction.")

    # --- Path reconstruction (CPU) ---
    # Download tower records, block_pool, span_pool, cell_to_block
    # Find best target state across all (dir, sb, hc) in target's block
    # Walk backward via _reconstruct_from_tower_records with _DynamicBlockDistProxy
```

## Control Buffer (V3-specific)

Defined in `relax_constrained_v3.cu` (NOT in `common.cuh`, to avoid breaking V2):

```
CTL_OUTPUT   = 0   # relax kernel output count (frontier for next light iteration)
CTL_SETTLED  = 1   # settled states count (for heavy phase)
CTL_PENDING  = 2   # pending queue count (states outside current bucket)
CTL_NEAR     = 3   # classify near count / extract count
CTL_FAR      = 4   # classify far count
CTL_TOWER    = 5   # tower record count (append-only across all kernels)
CTL_MIN_DIST = 6   # scan_min result (stored as float bits via atomicMin on int)
CTL_OVERFLOW = 7   # pool exhaustion + queue overflow counter (diagnostic)
CTL_SIZE     = 8
```

8 entries vs 15 in V2. No barrier counters needed. `CTL_TOWER` is never reset (cumulative). All others are reset by the Python driver at appropriate points (see driver pseudocode).

## Path Reconstruction

CPU-side path reconstruction from TowerRecord chain is reused from V2. The distance lookup proxy changes:

```python
class _DynamicBlockDistProxy:
    """Block-pool proxy for CPU-side distance lookups during path reconstruction."""
    _BLOCK_EMPTY = 0xFFFF

    def __init__(self, block_entries_cpu, span_pool_cpu,
                 cell_to_block_cpu, block_size, spc, n_span_bins, n_heights):
        self._blocks = block_entries_cpu
        self._cell_to_block = cell_to_block_cpu
        self._block_size = block_size
        self._mask = block_size - 1
        self._spc = spc
        self._n_span_bins = n_span_bins
        self._n_heights = n_heights
        self._sh = n_span_bins * n_heights

    def __getitem__(self, state):
        state = int(state)
        cell = state // self._spc
        block_idx = self._cell_to_block[cell]
        if block_idx < 0:
            return 1e30  # cell never visited
        base = block_idx * self._block_size
        # Compute local_key
        rem = state % self._spc
        direction = rem // self._sh
        rem2 = rem % self._sh
        span_bin = rem2 // self._n_heights
        hc = rem2 % self._n_heights
        local_key = direction * self._sh + span_bin * self._n_heights + hc
        # Hash probe within block (same as V2 multiplicative hash)
        h = (local_key * 2654435761) & self._mask
        for probe in range(self._block_size):
            slot = (h + probe) & self._mask
            entry = self._blocks[base + slot]
            k = int(entry['local_key'])
            if k == local_key:
                return float(entry['dist'])
            if k == self._BLOCK_EMPTY:
                return 1e30
        return 1e30
```

## Files

### New files
| File | Purpose | Est. lines |
|------|---------|-----------|
| `pyorps/utils/kernels/dynamic_blocks.cuh` | Block allocator (`get_block`) + dynamic block ops (`block_find_dyn`, `block_upsert_dyn`, `block_relax_dyn`) | ~150 |
| `pyorps/utils/kernels/relax_constrained_v3.cu` | Main relax kernel (light/heavy) + V3 control constants | ~550 |
| `pyorps/utils/kernels/classify_bucket.cu` | Bucket classification kernel | ~60 |
| `pyorps/utils/kernels/scan_min.cu` | Min-dist scan + bucket extraction | ~100 |
| `pyorps/utils/kernels/init_dynamic.cu` | Pool init + source init | ~70 |
| `pyorps/utils/constrained_sssp_gpu_v3.py` | Python wrapper + driver loop + DynamicBlockDistProxy | ~700 |
| `tests/test_graph/test_constrained_gpu_v3.py` | Test suite | ~500 |

### Reused unchanged
| File | What's reused |
|------|--------------|
| `common.cuh` | TowerRecord struct only (V3 defines own control indices separately) |
| `clearance.cuh` | `warp_cooperative_clearance()` + sequential fallback |
| `state_access.cuh` | `make_local_key()`, `shfl_sync_i64()` (V3 does NOT use `read_dist`/`relax_dist` — uses dynamic block equivalents instead) |

### Not modified
All V2 files remain untouched. V3 is a parallel implementation.

## Integration

`ConstrainedPathFinder` gets `backend="raster_gpu_v3"` (or just update `"raster_gpu"` to point to V3 when available, with V2 as fallback if V3 import fails).

## Testing Strategy

1. **Unit tests:** State encoding roundtrip, block allocation correctness (including race-wasted blocks), memory budget computation
2. **Small raster tests (50x50, 100x100):** Compare V3 path cost against Cython reference (tolerance 1e-3)
3. **Medium raster tests (200x200, 500x500):** Verify path found, cost within margin of Cython
4. **DEM + clearance tests:** Verify tower heights and clearance match Cython
5. **Area cost tests:** Verify forbidden pixel rejection and rotated footprint
6. **Edge cases:** No-path (surrounded by forbidden), single-cell path, source == target
7. **Pool exhaustion:** Verify graceful handling when max_blocks reached (warning emitted, path still found if possible)
8. **Eviction tests:** Verify correctness with BLOCK_SIZE < spc (forced eviction)
9. **Early termination:** Verify that target detection stops search early
10. **Queue overflow:** Verify CTL_OVERFLOW incremented and no crash when queues fill up

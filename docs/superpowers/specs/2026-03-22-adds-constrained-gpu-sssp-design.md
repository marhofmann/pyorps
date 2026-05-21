# ADDS-Inspired Constrained GPU Delta-Stepping (V4)

Design spec for a new high-performance constrained SSSP algorithm for GPU, targeting overhead power line routing with tower placement, span constraints, angle limits, and catenary clearance.

## Motivation

Current constrained GPU implementations have clear gaps:
- **V3 (frontier-based)**: Correct and memory-efficient but slow — Python host loop with per-bucket kernel launches causes CPU-GPU round-trip overhead
- **V2 (persistent kernel)**: Fast architecture but VRAM-limited — 6 GB GPU can't fit R4 dense (13 GB needed), block-sparse BLOCK_SIZE=64 too small for spc=864
- **Cython**: Correct and production-ready but single-threaded — R4 on 2000x2000 takes 600+ seconds

The new algorithm combines insights from the ADDS paper (Wang et al., PPoPP 2021) with proven components from V2/V3/V4 to achieve 5-10x speedup over Cython with bounded memory.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Full ADDS (MTB delegation) | Async scheduling, no barrier stalls, 2.9x over Near-Far in paper |
| Memory: distances | Block-sparse | O(1) lookup, bounded VRAM, handles spc=864 |
| Memory: work queue | 32 circular FIFO buckets | Priority ordering without full PQ overhead |
| Tower placement | Adaptive: per-thread (uniform) / warp-cooperative (exact) | Uniform is common case; exact added later |
| Delta strategy | Full dynamic (ADDS-style) | Runtime utilization monitoring, ~1.5x benefit |
| Clearance | Eager (during search) | Required for path optimality |
| File organization | Separate .cu/.cuh + Python driver | Syntax highlighting, selective V2 header reuse |
| Scope priority | Design for large (5000+), optimize on common (1000-3000) | Future-proof architecture |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Python Driver                          │
│  constrained_sssp_gpu_v4.py                             │
│  - Profile → LUT precomputation                         │
│  - Memory budget → block-sparse sizing                  │
│  - Kernel compilation (CuPy RawKernel)                  │
│  - Single kernel launch (cooperative groups)             │
│  - Post: download tower records, reconstruct path        │
└──────────────────────┬──────────────────────────────────┘
                       │ single launch
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Persistent CUDA Kernel                      │
│                                                          │
│  ┌──────────┐    ┌──────────┐         ┌──────────┐      │
│  │  MTB      │───▶│  Bucket  │────────▶│  WTB 0   │      │
│  │ (Block 0) │    │  Queue   │    AF   │  WTB 1   │      │
│  │           │◀───│ 32 FIFO  │◀────────│  WTB 2   │      │
│  │ - read    │    │ buckets  │  write  │  ...     │      │
│  │ - assign  │    │          │         │  WTB N   │      │
│  │ - delta   │    └──────────┘         └──────────┘      │
│  └──────────┘                                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Block-Sparse Distance Storage                    │    │
│  │  cell_to_block[n_cells] → pool[max_blocks×BS]    │    │
│  │  atomicMin on dist, atomicExch on span            │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Tower Record Buffer (global append)              │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Roles:**
- **Python driver**: One-time setup, single kernel launch, post-processing path reconstruction
- **MTB (Block 0)**: Reads bucket queue, assigns work chunks to idle WTBs, monitors utilization, adjusts delta dynamically
- **WTBs (Blocks 1..N)**: Process assigned work items — relax edges, place towers, write new states to bucket queue
- **Block-sparse storage**: Distance/span lookup with atomicMin, independent of work queue
- **Bucket queue**: 32 circular FIFO buckets with dynamically allocated pool blocks

## State Space

States encode `(cell, direction, span_bin, height_class)` packed into `int64`:

```
state = cell * (n_dirs * n_span_bins * n_heights)
      + direction * (n_span_bins * n_heights)
      + span_bin * n_heights
      + height_class
```

- **Span bins**: coarsened to `min_span_m` bin size (typically 2-4 bins, not 40)
- **Exact span**: tracked as `float16` separately (avoids quantization drift). Float16 precision: ~0.25m for typical spans (200-400m), ~2m for long spans (1000m+). Acceptable for clearance checking and tower placement decisions.
- **States per cell (spc)**: `n_dirs * n_span_bins * n_heights` (e.g., 16 dirs * 3 bins * 3 heights = 144 for moderate profile, up to 864 for R4)

## Source Initialization

Before the persistent kernel's main loop begins, source states must be seeded. This happens in a **separate init kernel** launched before the persistent kernel (like V2/V3's `init_source` pattern):

1. **Python driver** computes source states: for each valid starting direction `dir_out`, create `state = pack_state(source_cell, dir_out, 0, 0)` with `dist=0.0`, `span=0.0`
2. **Python driver** sets `best_target_dist = 0x7F800000` (+inf) via `cp.full(1, 0x7F800000, dtype=cp.int32)` — no kernel needed
3. **Single init kernel**: Seeds both block-sparse pool (allocate block for `source_cell`, insert source states with `dist=0.0`) and bucket queue (write source work items to bucket 0, set `resv_ptr` and `WCC[0]`). Trivially small launch (n_source_dirs threads).

The persistent kernel then launches with bucket 0 already populated. The MTB's first scan finds bucket 0 non-empty and begins assignment immediately.

## ADDS Bucket Queue

### Structure

32 circular FIFO buckets, each backed by dynamically allocated 64K-word pool blocks.

**Work item (16 bytes):**
```c
struct WorkItem {
    int64_t state;      // packed (cell, dir, span_bin, height_class)
    float   dist;       // distance at time of insertion
    __half  span_dist;  // exact span in meters
    uint16_t _pad;      // alignment
};
```

**Per-bucket metadata:**
```c
struct BucketMeta {
    int32_t resv_ptr;   // atomic: next write position (WTBs increment)
    int32_t read_ptr;   // MTB only: next read position
    int32_t* WCC;       // Write Completed Counter per segment (N=32)
    int32_t* CWC;       // Completed Work Counter per pool block
    int32_t n_blocks;   // allocated pool blocks for this bucket
    int32_t* block_ptrs; // pointers to pool blocks
};
```

### SRMW Protocol

**WTB writes:**
1. `slot = atomicAdd(&bucket.resv_ptr, 1)` — claim slot
2. `bucket.data[slot] = work_item` — write data
3. `__threadfence()` — flush to global memory
4. `atomicAdd(&bucket.WCC[slot/32], 1)` — signal "written"

**MTB reads:**
1. Check `WCC[segment] == 32` — full segment ready?
2. Bulk read 32 items from `read_ptr`
3. Advance `read_ptr += 32`
4. Compare `resv_ptr` after `__threadfence()` for partial segments
5. Assign items to idle WTBs via Assignment Flags

**SRMW liveness guarantee:** A WTB only enqueues during active processing (AF.status=2). All enqueues complete and WCC is incremented before the WTB writes AF.status=0 (with `__threadfence()` between). Therefore, there can never be a reserved-but-unwritten slot while the owning WTB is idle. The MTB will never deadlock waiting for WCC.

### Assignment Flags

Per-WTB global memory structure:
```c
struct AssignmentFlag {
    int32_t status;     // 0=idle, 1=assigned, 2=working
    int32_t bucket_id;  // source bucket
    int32_t offset;     // start index in bucket data
    int32_t count;      // number of items to process
};
```

### Bucket Pool Allocator

Buckets are backed by a global pool of **64K-item blocks** (each 64K * 16 bytes = 1 MB).

**Pool sizing:**
```python
bucket_pool_budget = min(400_MB, free_vram * 0.08)  # ~400 MB default
n_pool_blocks = bucket_pool_budget / (65536 * 16)   # ~25 blocks
```

**Slot-to-address mapping:**
```cuda
WorkItem* get_slot(BucketMeta* bucket, int slot) {
    int pool_block = slot / POOL_BLOCK_SIZE;
    int offset = slot % POOL_BLOCK_SIZE;
    return &pool[bucket->block_ptrs[pool_block] * POOL_BLOCK_SIZE + offset];
}
```

**Growth protocol:** When a WTB's `atomicAdd(&resv_ptr, 1)` returns a slot that crosses a pool block boundary, the WTB attempts `atomicCAS` on `bucket->block_ptrs[pool_block]` to claim a free pool block from the global free list (atomic counter `next_free_pool_block`). If the pool is exhausted, the WTB drops the work item and increments `d_control[CTL_POOL_OVERFLOW]`.

**Deallocation:** Pool blocks are NOT returned to a free list. The bucket pool uses **one-way monotonic allocation** — `next_free_pool_block` only increments. When a bucket is fully consumed, the MTB simply resets its metadata (resv_ptr, read_ptr, WCC, CWC, generation++) but does NOT free the pool blocks back. This is acceptable because: (a) the pool budget (~25 blocks = ~400 MB) is sized for the entire search lifetime, (b) buckets are consumed roughly in order (head → tail), so the working set of active pool blocks stays bounded, (c) adding a lock-free free list adds complexity with negligible benefit given the small pool size. If pool blocks are exhausted, `CTL_POOL_OVERFLOW` is incremented and the Python driver warns.

**CWC semantics:** `CWC[pool_block_index]` counts the number of work items completed by WTBs from that pool block. A WTB increments `CWC[block_id]` by its assignment `count` when it finishes processing. The MTB considers a pool block fully consumed when `CWC[pool_block_index] >= items_in_block` (where `items_in_block` is derived from the bucket's `read_ptr` advancement). This tells the MTB that no WTB is still reading from this block.

### Circular Bucket Mapping

Logical bucket ID from distance: `logical = (int)(dist / current_delta)`.

Physical mapping: `physical = logical % N_BUCKETS` (where N_BUCKETS=32).

Active range: at any time, logical buckets `[head_logical, head_logical + N_BUCKETS)` are valid. Items with `logical >= head_logical + N_BUCKETS` are clamped to the tail bucket (`physical = (head_logical + N_BUCKETS - 1) % N_BUCKETS`).

**Generation counter:** Each physical bucket has a `generation` field (incremented on each reuse). WTBs store `(physical_bucket, generation)` when enqueuing. If a WTB enqueues to a recycled bucket (generation mismatch), the item is redirected to the tail bucket. This prevents stale enqueues from corrupting recycled buckets.

When the MTB advances head, the freed physical bucket's generation is incremented and its `resv_ptr`/`read_ptr`/WCC/CWC are reset.

### Bucket Lifecycle

- Head pointer tracks highest-priority non-empty bucket
- When head bucket empty + all CWCs confirm no in-flight writes: deallocate pool blocks, advance head
- Tail bucket catches overflow (distances beyond `32 * delta`)
- On head advance: increment generation, reset metadata, re-split tail if needed

## Block-Sparse Distance Storage

### Two-Level Structure

**Level 1 (Dense):** `cell_to_block[n_cells] → int32`
- Maps each raster cell to its pool block index
- Initialized to -1 (unallocated)
- Allocated on first visit via `atomicCAS(-1, new_block_idx)`
- Memory: `n_cells * 4 bytes` (36 MB for 3000x3000)

**Level 2 (Per-cell hash):** `pool[block_idx * BLOCK_SIZE + slot]`
- Each block has `BLOCK_SIZE` slots (power of 2, auto-sized to fit VRAM)
- `local_key = dir * (n_span_bins * n_heights) + span_bin * n_heights + height_class`
- Hash: `(local_key * 2654435761) & (BLOCK_SIZE - 1)`
- Linear probing within block boundary

**Entry (8 bytes):**
```c
struct BlockEntry {
    uint16_t local_key;  // packed sub-state
    uint16_t _pad;
    float    dist;       // best known distance (atomicMin target)
};
```

**Separate span array:** `span_pool[block_idx * BLOCK_SIZE + slot] → float16`

### BLOCK_SIZE Auto-Sizing

```python
spc = n_dirs * n_span_bins * n_heights
target_bs = next_power_of_2(spc * 1.5)     # 50% headroom

available = vram - 1.5_GB                    # reserve for inputs/queues/towers
max_bs = available / (n_cells * 10)          # 10 bytes per slot

BLOCK_SIZE = clamp(min(target_bs, max_bs), 32, 1024)
```

If `BLOCK_SIZE < spc`: eviction mode (replace worst-distance entry). If `BLOCK_SIZE >= spc`: no eviction (all states fit).

### Memory Budget Example (3000x3000 R4, 6 GB GPU)

| Component | Formula | Size |
|-----------|---------|------|
| Raster (uint16) | 9M * 2 | 18 MB |
| DEM (float32) | 9M * 4 | 36 MB |
| Obstacle heights (float32) | 9M * 4 | 36 MB |
| LUTs in global memory | ~100 KB | ~0.1 MB |
| `cell_to_block` (int32) | 9M * 4 | 36 MB |
| Block-sparse pool (BS=32) | 9M * 0.4 * 32 * 10 | **1.15 GB** |
| Bucket queue pool | 25 blocks * 64K * 16 | 25 MB |
| Tower records (2M cap) | 2M * 24 | 48 MB |
| Control + AF + metadata | ~10 KB | ~0.01 MB |
| **Total** | | **~1.37 GB** |
| **Available (6 GB - 500 MB OS)** | | **~5.5 GB** |
| **Headroom** | | **~4.1 GB** |

With `max_visited_fraction=0.4` (40% of cells visited), pool allocates blocks for 3.6M cells. At BS=32 and 10 bytes/slot, that's 1.15 GB — well within budget. The 40% fraction is conservative; typical routing visits 10-30% of cells.

**Eviction impact for BS=32, spc=864:** Most cells have <32 actively competitive states near the optimal corridor. Cells far from the optimal path are visited once with high distance and never revisited. Eviction primarily affects cells at corridor boundaries where multiple directions compete — these states are eventually re-discovered via neighboring cells. Empirical testing with V3 (BS=64, spc=864) showed paths within 0.1% of Cython optimal.

### Relaxation

```cuda
int block_idx = cell_to_block[cell];
if (block_idx == -1)
    block_idx = allocate_block(cell);  // atomicCAS on cell_to_block
if (block_idx < 0) {
    atomicAdd(&d_control[CTL_BLOCK_OVERFLOW], 1);
    return;  // pool exhausted, skip
}

uint16_t lk = pack_local_key(dir, span_bin, hc);
int slot = find_or_insert(block_idx, lk);  // hash + linear probe

int old = atomicMin((int*)&pool[block_idx * BS + slot].dist,
                    __float_as_int(new_dist));

if (__float_as_int(new_dist) < old) {
    pool_span[block_idx * BS + slot] = __float2half(new_span);
    enqueue_to_bucket(state, new_dist, new_span);  // SRMW write

    // Update best_target_dist if this is the target cell
    if (cell == target_cell)
        atomicMin((int*)&best_target_dist, __float_as_int(new_dist));
}
```

**Known limitation — span race:** Between `atomicMin` on dist and the span write, another thread could win a better distance and read a stale span. This is inherited from V2/V3 and is benign: stale-span work items fail the stale check when processed. For path reconstruction, the final span at each state may not correspond to the optimal path — tower records carry their own span values and are authoritative for reconstruction. The block-sparse span is only used for span constraint checking during relaxation, where a slightly stale value is acceptable (it may cause a redundant tower or skip a marginal one, but `atomicMin` on distance ensures the optimal distance is always found).

## MTB Manager Logic

### MTB Threading Model

All 256 threads in Block 0 cooperate on MTB duties:
- **Warp 0 (threads 0-31)**: Scans WCC arrays across buckets in parallel (each thread checks one bucket's WCC segment), manages AF assignment, performs delta adjustment
- **Warps 1-7 (threads 32-255)**: Cooperate on bulk-reading ready segments from buckets (32 threads read 32-item segments in parallel), scanning idle WTB list

This ensures the MTB block is not single-threaded bottlenecked. The MTB can read and assign at the rate of ~32 items per warp cycle.

### Main Loop

```
while (!done) {
    // 1. Find highest-priority non-empty bucket
    //    Warp 0: each thread checks one bucket's WCC readiness
    head = cooperative_scan_buckets();
    if (!found) {
        if (second_empty_sweep) done = true;
        continue;
    }

    // 2. Count idle WTBs first — only read as many items as we can assign
    n_idle = cooperative_scan_AFs();  // warps 1-7 scan AF statuses
    if (n_idle == 0) continue;        // no idle WTBs, skip reading

    // 3. Bulk-read ready segments from head bucket (up to n_idle chunks)
    items = cooperative_read_segments(head, n_idle * chunk_size);

    // 4. Assign to idle WTBs
    for each idle WTB (up to items available):
        set AF = {status=1, bucket, offset, count};
        __threadfence();  // ensure AF visible before status write

    // 5. Multi-bucket assignment (if idle WTBs remain)
    if (idle_remain && head_nearly_empty)
        assign from buckets [head+1, head+2, head+3];

    // 6. Dynamic delta adjustment (periodic)
    if (assignments >= SETTLE_PERIOD)
        adjust_delta(count_working_wtbs() / total_wtbs);

    // 7. Cleanup: deallocate fully consumed buckets

    // 8. Early termination: head_floor > best_target_dist * margin
    //    best_target_dist is a global atomic updated by WTBs
    //    whenever they relax a state at the target cell
}
```

**Key detail (step 2)**: The MTB only reads from buckets when idle WTBs exist. This prevents the "read but can't assign" item loss scenario — `read_ptr` only advances when items will be assigned.

### Dynamic Delta

Delta changes only at **safe transition points**: when all buckets below the new head are empty and no WTBs have in-flight work referencing the old delta. This avoids bucket aliasing (items enqueued under old delta ending up in wrong buckets under new delta).

```
adjust_delta(utilization):
    // Only adjust at safe points (all buckets below head are empty)
    if (!at_safe_transition_point()) return;

    // Clipping guard: if tail has >=65% of items, delta too small
    if (tail_fraction >= 0.65) clip_floor = current_delta;

    if (avg_util < 0.5)  current_delta *= 2.0;   // severely underutilized
    elif (avg_util < 0.7) current_delta *= 1.25;  // slightly low
    elif (avg_util > 0.95) current_delta *= 0.8;  // too much redundant work

    current_delta = max(current_delta, clip_floor);

    // After delta change: recompute bucket boundaries for future inserts
    // Existing bucket contents remain valid (distances unchanged)
    // head_logical is updated to match new delta scale
```

**Initial delta:**
```python
avg_terrain = mean(raster[raster < 65535]) * mean(cost_factors)
initial_delta = 2.0 * avg_terrain * n_dirs
```

### Target Distance Tracking

A global atomic `best_target_dist` (int32, IEEE 754 bits) is maintained:
- Initialized to `0x7F800000` (+inf) by the init kernel
- WTBs update via `atomicMin((int*)&best_target_dist, __float_as_int(dist))` whenever they successfully relax any state at the target cell
- MTB reads this value for early termination checks

### Termination

1. **Target margin**: `head_logical * current_delta > __int_as_float(best_target_dist) * margin` (default `margin=1.0001`, configurable via Python driver — matches the 0.01% correctness tolerance)
2. **Double empty sweep**: MTB scans all 32 buckets twice with zero items
3. **Safeguard**: configurable max iterations (default 100,000)

### Pool Exhaustion Handling

**Block-sparse pool**: When `allocate_block()` fails (pool full), the WTB skips relaxation for that cell and increments `d_control[CTL_BLOCK_OVERFLOW]`. After kernel completion, the Python driver checks this counter and warns: "Block-sparse pool exhausted — N states dropped. Results may be suboptimal. Increase max_visited_fraction or reduce raster size."

**Bucket pool**: When bucket pool blocks are exhausted, the WTB drops the enqueue and increments `d_control[CTL_POOL_OVERFLOW]`. Same post-kernel warning pattern.

Both counters are checked by the Python driver. If either > 0, the returned path is flagged with `path.overflow_warning = True`.

## Worker Thread Block (WTB) Logic

### Main Loop

```
while (!done) {
    // 1. Poll AF until assigned or done
    while (AF.status != 1 && !done) { spin; }
    AF.status = 2;  // "working"

    // 2. Process items (strided across threads)
    for (i = threadIdx; i < count; i += blockDim) {
        item = read_item(assignment);

        // Stale check
        if (item.dist > read_dist(item.state)) continue;

        // 3. Relax non-tower edges (same direction, no tower)
        for each valid same-direction neighbor:
            compute edge cost (terrain + angle + gradient)
            new_span = span + step_distance
            if (new_span > max_span) skip
            relax via block-sparse atomicMin
            if improved: enqueue to bucket

        // 4. Tower placement (if span >= min_span)
        if (span >= min_span):
            for each valid outgoing direction:
                compute tower cost (terrain + angle + height + gradient)
                check clearance (catenary sag)
                relax with span reset to 0
                if improved: enqueue + record tower
    }

    // 5. Signal completion
    __syncthreads();  // ensure all threads finished enqueuing
    if (threadIdx.x == 0) {
        atomicAdd(&bucket.CWC[smem.block_id], smem.count);
        __threadfence();  // ensure CWC visible before advertising idle

        // Report stale ratio for MTB diagnostics
        if (stale_count > smem.count * 0.9)
            atomicAdd(&d_control[CTL_STALE_ASSIGNMENTS], 1);

        AF.status = 0;  // "idle" — safe because all enqueues + CWC are flushed
    }
    __syncthreads();
}
```

### Tower Placement (Uniform Mode, Per-Thread)

```cuda
place_towers(cell, dir_in, span, dist, hc):
    for each dir_out where angle_valid[dir_in][dir_out]:
        tower_terrain = tower_terrain_lut[raster[cell]];
        if (tower_terrain >= FORBIDDEN) continue;
        tower_angle = tower_angle_lut[dir_in * n_dirs + dir_out];
        tower_premium = height_premiums[hc];

        slope = compute_slope(cell, dir_in);
        foundation = exp(gradient_scale * slope / 100.0);

        tower_cost = (tower_terrain + tower_angle + tower_premium)
                     * foundation;

        // Clearance (if DEM)
        if (HAS_CLEARANCE) {
            if (!check_clearance(cell, dir_in, span, hc)) {
                // Try taller heights
                for (h = hc-1; h >= 0; h--)
                    if (check_clearance(..., h)) { hc_out = h; break; }
                if (no_height_works) continue;
            }
        }

        new_dist = dist + tower_cost;
        new_state = pack_state(cell, dir_out, 0, hc_out);
        if (relax_dist(new_state, new_dist, 0.0))
            enqueue + record_tower;
```

### Tower Placement (Exact Mode, Warp-Cooperative — Future)

Added later as compile-time switch `TOWER_AREA_MODE=1`. Uses `__shfl_sync` to broadcast tower params, 32 lanes cooperate on rotated square footprint pixel sum, `__ballot_sync` for forbidden pixel rejection.

## File Organization

```
pyorps/utils/kernels/
├── common.cuh                    # REUSE from V2
├── grid_barrier.cuh              # REUSE (for init kernels only)
├── clearance.cuh                 # REUSE (sequential catenary check)
├── dynamic_blocks.cuh            # REUSE from V3 (cell_to_block, dynamic alloc)
├── state_access.cuh              # REUSE (read_dist, relax_dist)
│
├── adds_common.cuh          (NEW) # WorkItem, BucketMeta, constants
├── adds_bucket_queue.cuh    (NEW) # FIFO: enqueue, read_segment, grow
├── adds_mtb.cuh             (NEW) # MTB loop: scan, assign, delta, term
├── adds_wtb.cuh             (NEW) # WTB loop: poll, relax, tower
├── adds_tower.cuh           (NEW) # Tower: uniform + exact mode switch
│
└── constrained_adds.cu      (NEW) # Main kernel + shared mem setup

pyorps/utils/
└── constrained_sssp_gpu_v4.py (NEW) # Python driver
```

**Reused V2/V3 headers**:
- `common.cuh` — control indices; TowerRecord is extended for V4 (adds `tower_cost` field in place of V2's padding bytes; defined as `TowerRecordV4` in `adds_common.cuh` to avoid breaking V2)
- `clearance.cuh` — sequential per-thread catenary sag check
- `dynamic_blocks.cuh` — dynamic block allocation (`get_block`, `block_upsert_dyn`, `block_relax_dyn`) with `cell_to_block` indirection. This is the V3 header, not `block_sparse.cuh` (which uses dense pre-allocated blocks)
- `state_access.cuh` — `read_dist`/`relax_dist` abstraction (extended to support dynamic blocks)
- `grid_barrier.cuh` — Blackwell-safe barrier (for init kernels only — main kernel uses async MTB, not barriers)

**Compile-time constants** injected by Python driver:
```c
#define N_DIRS, N_SPAN_BINS, N_HEIGHTS, BLOCK_SIZE
#define N_BUCKETS 32
#define SEGMENT_SIZE 32
#define POOL_BLOCK_SIZE 65536
#define MAX_BLOCKS {grid_size}
#define THREADS_PER_BLK 256
#define HAS_DEM {0|1}
#define HAS_CLEARANCE {0|1}
#define TOWER_AREA_MODE {0=uniform|1=exact}
#define SPC {states_per_cell}
```

### Shared Memory Layout (per block, ~4-8 KB)

```c
__shared__ struct {
    int8_t  steps[MAX_DIRS * 2];
    float   cost_factors[MAX_DIRS];
    int16_t intermediates[MAX_DIRS * MAX_INTER * 2];
    int32_t n_intermediates[MAX_DIRS];
    uint8_t angle_valid[MAX_DIRS * MAX_DIRS];
    float   angle_cost[MAX_DIRS * MAX_DIRS];
    float   tower_angle_cost[MAX_DIRS * MAX_DIRS];
    float   height_premiums[MAX_HEIGHTS];
    union {
        struct { int32_t idle_list[MAX_BLOCKS]; ... } mtb;
        struct { int32_t bucket_id, offset, count; }  wtb;
    };
};
```

### Kernel Launch

```python
n_sms = device.attributes['MultiProcessorCount']
max_blocks = n_sms * 2                     # 28 on RTX PRO 500
threads_per_block = 256

kernel = cp.RawKernel(source, "constrained_adds_main",
                      enable_cooperative_groups=True,
                      options=("--std=c++17", "-Xptxas", "-dlcm=cg"))
kernel((max_blocks,), (threads_per_block,), args=(...))
```

## Path Reconstruction

**Tower Record Buffer** (global atomic append during search):
```c
// Defined in adds_common.cuh as TowerRecordV4 (not reusing V2's TowerRecord
// which lacks tower_cost and has different padding). V2's common.cuh is only
// reused for control buffer index constants, not the TowerRecord struct.
struct TowerRecordV4 {    // 24 bytes, 8-byte aligned
    int64_t state;         // state AFTER tower (new dir, span=0)
    int64_t pred_state;    // state BEFORE tower (incoming dir, span=X)
    __half  span_dist;     // span length at tower placement
    __half  tower_height;  // selected tower height (meters)
    float   tower_cost;    // total tower cost (terrain + angle + height)
};
```

Tower count tracked via `atomicAdd(&d_control[CTL_TOWER_COUNT], 1)`. If count exceeds `max_tower_records`, the record is silently dropped (path reconstruction still works via block-sparse distance walking, just slower).

**CPU-side reconstruction** (after kernel completes):
1. Find best target state by scanning block-sparse pool at target cell
2. Build `state → TowerRecord` lookup from downloaded records
3. Walk backward from target to source via tower chain
4. Direction-walk between towers for intermediate cells
5. Return `ConstrainedPath` with geometry, towers, costs

## Integration

New backend `"raster_gpu_v4"` in `ConstrainedPathFinder`:
```python
finder = ConstrainedPathFinder(
    dataset_source=raster,
    source_coords=src, target_coords=tgt,
    profile=profile,
    graph_api="raster_gpu_v4"
)
path = finder.find_route()
```

## Testing Strategy

**Level 1 — Unit tests:** Bucket queue round-trip, block-sparse insert/lookup/eviction, SRMW concurrent writes, state pack/unpack, tower cost vs Cython, clearance vs Cython.

**Level 2 — Algorithm tests (small synthetic):** Uniform 50x50, gradient 100x100, wall-with-gap, straight line, max/min span enforcement, angle constraints, variable height clearance.

**Level 3 — Cython comparison (medium):** 500x500 and 1000x1000 random rasters — path cost within 0.01% of Cython Dijkstra, tower heights match, tower count within +/-1.

**Level 4 — Performance benchmarks (manual):**

| Size | Profile | Cython | GPU V4 Target | Speedup |
|------|---------|--------|---------------|---------|
| 500x500 | R3 | ~30s | 3-5s | 6-10x |
| 1000x1000 | R4 | ~150s | 15-30s | 5-10x |
| 2000x2000 | R4 | ~600s+ | 60-120s | 5-10x |
| 3000x3000 | R4 | hours | 3-5 min | 10x+ |
| 5000x5000 | R5 | impractical | 10-20 min | -- |

**Correctness invariants (debug mode):**
1. No path crosses impassable cells
2. All towers satisfy `min_span <= span <= max_span`
3. All turn angles satisfy `hard_angle_limit`
4. Clearance passes at every tower (if DEM provided)
5. Path cost = sum of edge costs + tower costs
6. Total cost <= Cython Dijkstra cost * 1.001

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MTB becomes bottleneck | WTBs starve | All 256 MTB threads cooperate; multi-bucket assignment (3-4 buckets); only read when idle WTBs exist |
| Block-sparse eviction loses optimal state | Suboptimal path | Track eviction count via `CTL_BLOCK_OVERFLOW`; warn if >1% of inserts evict |
| Bucket pool memory exhaustion | Dropped work items | Cap pool size; dealloc consumed buckets aggressively; `CTL_POOL_OVERFLOW` counter |
| Dynamic delta aliasing | Wrong bucket assignments | Delta changes only at safe transition points (all lower buckets empty) |
| Dynamic delta oscillation | Unstable performance | Settling period between adjustments; utilization history smoothing (8-sample window) |
| Stale item thundering herd | Wasted WTB cycles | WTBs report >90% stale ratio via `CTL_STALE_ASSIGNMENTS`; MTB can skip to next bucket or increase delta |
| Span race in relaxation | Stale span value stored | Benign: stale items filtered at processing time; tower records are authoritative for reconstruction |
| Warp divergence in tower placement | Low throughput | Uniform mode (per-thread) minimizes divergence; exact mode deferred |
| Blackwell memory ordering | Silent corruption | `-dlcm=cg` flag + `__threadfence()` at all sync points (proven in V4) |
| Block-sparse pool exhausted | States dropped | `CTL_BLOCK_OVERFLOW` counter; Python driver warns user; `path.overflow_warning` flag |

## References

1. Wang, Fussell, Lin. "A Fast Work-Efficient SSSP Algorithm for GPUs." PPoPP 2021.
2. Davidson, Baxter, Garland, Owens. "Work-Efficient Parallel GPU Methods for SSSP." IPDPS 2014.
3. Berney, Iacono, Karsin, Sitchinava. "A Parallel Priority Queue with Fast Updates for GPU." arXiv 2023.
4. Lesnikov, Chernoskutov. "Performance analysis of Delta-stepping on CPU and GPU." CEUR-WS 2016.
5. Safari, Ebnenasir. "Locality-Based Relaxation for GPU Shortest Paths." TTCS 2017.

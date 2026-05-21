# ADDS Constrained GPU Delta-Stepping V4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-performance ADDS-inspired constrained SSSP algorithm on GPU that achieves 5-10x speedup over Cython for overhead power line routing with tower placement.

**Architecture:** Full ADDS delegation pattern — one Manager Thread Block (MTB) reads from 32 circular FIFO buckets and assigns work to Worker Thread Blocks (WTBs) via Assignment Flags. Block-sparse distance storage provides O(1) lookups with bounded VRAM. Single persistent kernel launch (no CPU-GPU round trips).

**Tech Stack:** CuPy (RawKernel with cooperative groups), CUDA C++17, Python 3.11+

**Spec:** `docs/superpowers/specs/2026-03-22-adds-constrained-gpu-sssp-design.md`

**Key reference files:**
- V3 GPU driver: `pyorps/utils/constrained_sssp_gpu_v3.py`
- V2 CUDA kernels: `pyorps/utils/kernels/` (common.cuh, dynamic_blocks.cuh, clearance.cuh, state_access.cuh, grid_barrier.cuh)
- V4 unconstrained kernel: `pyorps/utils/sssp_gpu.py:635-883`
- Constrained path finder: `pyorps/graph/constrained_path_finder.py`
- Infrastructure profile: `pyorps/core/infrastructure_profile.py`
- Existing tests: `tests/test_graph/test_constrained_gpu_v3.py`

---

### Task 1: CUDA Foundation Headers (adds_common.cuh)

**Files:**
- Create: `pyorps/utils/kernels/adds_common.cuh`
- Test: `tests/test_graph/test_constrained_gpu_v4.py`

Define all shared structs, constants, and device utility functions used across the ADDS kernel.

- [ ] **Step 1: Create adds_common.cuh with core structs**

```c
// Contents:
// 1. #include <cuda_fp16.h>
// 2. #include "dynamic_blocks.cuh"  ← reuse BlockEntry (8 bytes) from V3; do NOT redefine it
// 3. V4-specific control buffer indices (CTL_V4_DONE, CTL_V4_TOWER_COUNT,
//    CTL_V4_BLOCK_OVERFLOW, CTL_V4_POOL_OVERFLOW, CTL_V4_STALE_ASSIGNMENTS,
//    CTL_V4_BEST_TARGET_DIST, etc.)
// 4. WorkItem struct (16 bytes: int64 state, float dist, __half span_dist, uint16 pad)
// 5. TowerRecordV4 struct (24 bytes: int64 state, int64 pred_state, __half span_dist,
//    __half tower_height, float tower_cost)
// 6. AssignmentFlag struct (16 bytes: int32 status, bucket_id, offset, count)
//    NOTE: AF pointers MUST be declared volatile for WTB polling loops
// 7. pack_state() / unpack_state() / pack_local_key() device functions
// 8. FORBIDDEN constant: #define FORBIDDEN 65535 (uint16 max = impassable cell)
```

**Important:** Do NOT redefine `BlockEntry` — it is already defined in `dynamic_blocks.cuh` (lines 39-43). `adds_common.cuh` includes `dynamic_blocks.cuh` transitively, so `BlockEntry`, `get_block`, `block_relax_dyn`, `block_read_dist_dyn` are all available.

Reference V3's state encoding in `constrained_sssp_gpu_v3.py` lines 50-80 for `pack_state`/`unpack_state`. Reference `common.cuh` lines 6-22 for control buffer index naming convention.

- [ ] **Step 2: Create test file with compilation check**

```python
# tests/test_graph/test_constrained_gpu_v4.py
# Test that the header compiles and structs have correct sizes.
# Write a tiny CuPy RawKernel that instantiates each struct and writes sizeof() to output.
# Verify: sizeof(WorkItem)==16, sizeof(TowerRecordV4)==24, sizeof(AssignmentFlag)==16
# Verify: pack/unpack state round-trip for known (cell, dir, span_bin, hc) tuples
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_struct_sizes or test_state_packing"`

- [ ] **Step 3: Verify tests pass**

- [ ] **Step 4: Commit**

---

### Task 2: Bucket Queue Header (adds_bucket_queue.cuh)

**Files:**
- Create: `pyorps/utils/kernels/adds_bucket_queue.cuh`
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Implement the FIFO bucket queue with SRMW protocol, pool allocator, and circular bucket mapping.

- [ ] **Step 1: Create adds_bucket_queue.cuh**

```c
// Contents:
// 1. Constants: N_BUCKETS=32, SEGMENT_SIZE=32, POOL_BLOCK_SIZE=65536
// 2. Bucket metadata array layout in global memory:
//    - bucket_resv_ptr[N_BUCKETS]    (int32, atomic write position — WTBs atomicAdd)
//    - bucket_read_ptr[N_BUCKETS]    (int32, MTB read position)
//    - bucket_generation[N_BUCKETS]  (int32, reuse counter — stale enqueue detection)
//    - bucket_wcc[N_BUCKETS * max_segments]  (int32, Write Completed Counters per segment)
//    - bucket_cwc[N_BUCKETS * max_pool_blocks_per_bucket]  (int32, Completed Work Counters
//      per pool block — WTBs increment by assignment count when done processing;
//      MTB checks CWC >= items_read_from_block to confirm no in-flight readers)
//    - bucket_pool_base[N_BUCKETS]   (int32, first pool block index for this bucket)
//    - bucket_pool_count[N_BUCKETS]  (int32, number of pool blocks allocated)
// 3. WorkItem pool: flat array of WorkItem, indexed by pool_block * POOL_BLOCK_SIZE + offset
// 4. Device functions:
//    - enqueue_to_bucket(WorkItem* pool, int* bucket_meta, int* next_free_block,
//                        int* control, WorkItem item, float delta, int head_logical,
//                        int max_pool_blocks)
//      → computes logical bucket, maps to physical, checks generation (redirect to tail on mismatch),
//        atomicAdd resv_ptr to claim slot, if slot crosses pool block boundary:
//        new_block = atomicAdd(next_free_block, 1); if new_block >= max_pool_blocks:
//        atomicAdd(control[CTL_POOL_OVERFLOW], 1) and return (drop item).
//        Write item to pool[pool_block * POOL_BLOCK_SIZE + offset],
//        __threadfence(), atomicAdd(WCC[slot/SEGMENT_SIZE], 1)
//    - read_segment(WorkItem* pool, int* bucket_meta, int physical_bucket,
//                   WorkItem* out, int segment_idx)
//      → check WCC[segment]==SEGMENT_SIZE, bulk-read, return count
//    - reset_bucket(int* bucket_meta, int physical_bucket)
//      → zero resv_ptr/read_ptr/WCC, increment generation
```

Refer to spec sections "ADDS Bucket Queue" and "Bucket Pool Allocator" for the exact protocol. Reference V3's queue management pattern in `constrained_sssp_gpu_v3.py` lines 400-500 for the control buffer approach.

- [ ] **Step 2: Write test: single-thread enqueue + read round-trip**

```python
# Test kernel: one block, 1 thread enqueues 100 items to bucket 0,
# then reads them back via read_segment.
# Verify: all 100 items recovered in order, correct dist/state values.
```

- [ ] **Step 3: Write test: multi-thread SRMW concurrent enqueue**

```python
# Test kernel: 8 blocks × 256 threads, each thread enqueues 1 item to bucket 0.
# Then single thread reads all segments.
# Verify: exactly 2048 items read, no duplicates, no missing items.
# Verify: WCC[segment] == SEGMENT_SIZE for all 64 complete segments.
# This tests atomicAdd on resv_ptr + WCC correctness under contention.
```

- [ ] **Step 4: Write test: partial segment read**

```python
# Enqueue 50 items (1 full segment of 32 + 18 partial).
# read_segment for segment 0: should return 32 items (WCC==32).
# read_segment for segment 1: WCC==18 < 32, should return 0 from segment check.
# Use resv_ptr comparison to detect partial: resv_ptr=50, read_ptr=32 → 18 remaining.
# MTB should handle this by reading items individually from read_ptr to resv_ptr.
```

- [ ] **Step 5: Write test: circular bucket mapping + generation mismatch**

```python
# Test: enqueue items with distances spanning 3 bucket cycles (logical 0-95).
# Verify items end up in correct physical buckets (logical % 32).
# Reset bucket 0 (increment generation).
# Attempt to enqueue with old generation → verify item redirected to tail bucket.
# Enqueue with new generation → verify item accepted in bucket 0.
```

- [ ] **Step 6: Run all tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_bucket"`

- [ ] **Step 7: Commit**

---

### Task 3: Block-Sparse Distance Storage Integration

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Verify that the existing `dynamic_blocks.cuh` works correctly for V4's use case (allocate, upsert, relax, read). No new CUDA files — just write tests against the reused V3 header.

- [ ] **Step 1: Write test: allocate + relax + read round-trip**

```python
# Test kernel using dynamic_blocks.cuh functions directly:
# 1. get_block(cell=42) → allocates block
# 2. block_relax_dyn(pool, span_pool, block_idx, local_key=5, dist=100.0, span=50.0) → returns 1
# 3. block_read_dist_dyn(pool, block_idx, local_key=5) → returns 100.0
# 4. block_relax_dyn(..., local_key=5, dist=80.0, span=40.0) → returns 1 (improved)
# 5. block_read_dist_dyn(..., local_key=5) → returns 80.0
# 6. block_relax_dyn(..., local_key=5, dist=90.0, span=45.0) → returns 0 (not improved)
```

- [ ] **Step 2: Write test: concurrent relaxation (atomicMin correctness)**

```python
# 256 threads all relax the same (cell, local_key) with different distances.
# Thread i writes dist = 1000 - i. After kernel: read_dist should return 1000 - 255 = 745.
```

- [ ] **Step 3: Write test: eviction under full block**

```python
# BLOCK_SIZE=32, insert 64 different local_keys into the same cell.
# Assign dist = local_key * 10 (so key 0 has dist=0, key 63 has dist=630).
# dynamic_blocks.cuh eviction policy: replaces the WORST (highest dist) entry.
# After all inserts: read all 32 slots.
# Verify: the 32 surviving entries have the 32 lowest distances (0, 10, ..., 310).
# Verify: entries with dist > 310 were evicted.
```

- [ ] **Step 4: Run all tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_block_sparse"`

- [ ] **Step 5: Commit**

---

### Task 4: Source Initialization Kernel

**Files:**
- Create: `pyorps/utils/kernels/adds_init.cu`
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Write the init kernel that seeds block-sparse pool + bucket 0 with source states.

- [ ] **Step 1: Create adds_init.cu**

```c
// Kernel: adds_init_source(pool, span_pool, cell_to_block, block_to_cell, n_allocated,
//                          bucket_pool, bucket_meta, control,
//                          source_cell, n_dirs, n_span_bins, n_heights,
//                          BLOCK_SIZE, max_sparse_blocks)
// Each thread handles one starting direction (threadIdx.x < n_dirs):
//   1. state = pack_state(source_cell, threadIdx.x, 0, 0)
//   2. get_block(source_cell, ...) → block_idx (only first thread allocates, rest get same block)
//   3. block_relax_dyn(pool, ..., local_key, 0.0f, 0.0f) → seed with dist=0
//   4. Write WorkItem{state, 0.0f, __float2half(0.0f), 0} to bucket 0 at position threadIdx.x
//   5. Thread 0: set bucket_resv_ptr[0] = n_dirs, bucket_wcc[0] = n_dirs
```

Include `adds_common.cuh`, `dynamic_blocks.cuh`, `adds_bucket_queue.cuh`.

- [ ] **Step 2: Write test: init + verify block-sparse and bucket 0**

```python
# Launch init kernel for a 100×100 raster, source at (50, 50), 8 directions.
# Verify: cell_to_block[50*100+50] != -1 (block allocated)
# Verify: 8 entries in block-sparse pool with dist=0.0
# Verify: bucket_resv_ptr[0] == 8, bucket_wcc[0] == 8
# Verify: first 8 WorkItems in bucket 0 have correct states and dist=0
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

---

### Task 5: WTB Worker Logic (adds_wtb.cuh)

**Files:**
- Create: `pyorps/utils/kernels/adds_wtb.cuh`
- Create: `pyorps/utils/kernels/adds_tower.cuh`
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Implement the WTB main loop: poll AF, process work items, relax edges, place towers, signal completion.

- [ ] **Step 1: Create adds_tower.cuh with uniform-mode tower placement**

```c
// Device function: place_towers_uniform(...)
// For each valid dir_out: compute tower cost, check clearance (if HAS_CLEARANCE),
// relax new state with span reset, record tower.
// Reference: spec "Tower Placement (Uniform Mode, Per-Thread)" and
// V2's constrained_persistent.cu tower placement section.
// Reuse clearance.cuh::check_span_clearance() for catenary check.
```

- [ ] **Step 2: Create adds_wtb.cuh with WTB main loop**

```c
// Device function: wtb_main_loop(block_id, ...)
// 1. Load shared memory cooperatively (strided across 256 threads):
//    - steps[i] loaded by thread i % n_dirs
//    - cost_factors, step_distances: same pattern
//    - angle_valid[i*n_dirs+j], angle_cost, tower_angle_cost: thread (i*n_dirs+j) % 256
//    - intermediates: larger, needs multi-pass strided load
//    - height_premiums, tower_heights: single warp loads (small arrays)
//    - MTB block uses union { mtb { idle_list } }, WTBs use union { wtb { bucket_id, offset, count } }
//    - __syncthreads() after all loads complete
// 2. Poll AF until assigned or done (AF pointer MUST be volatile int*)
// 3. Process items strided across threads:
//    a. Stale check via block_read_dist_dyn
//    b. For each valid neighbor: compute edge cost, check intermediates, relax
//    c. If span >= min_span: call place_towers_uniform
// 4. __syncthreads, CWC increment, __threadfence, stale ratio report, AF.status=0
//
// Reference: spec "Worker Thread Block (WTB) Logic"
// Reference: V4 unconstrained kernel in sssp_gpu.py:700-850 for edge relaxation pattern
// Reference: V3's relax kernel for constrained edge cost computation
```

- [ ] **Step 3: Write test: single WTB processes hand-crafted work items**

```python
# Setup: 10×10 uniform raster (cost=100), 8 directions, simple span profile.
# Pre-seed block-sparse with source states at (5,5).
# Pre-fill bucket 0 with 8 source work items.
# Pre-set one AF with {status=1, bucket=0, offset=0, count=8}.
# Launch kernel with 1 block running wtb_main_loop.
# Verify: neighbors of (5,5) have been relaxed in block-sparse pool.
# Verify: new work items enqueued to bucket queue.
# Verify: AF.status == 0 (idle) after processing.
```

- [ ] **Step 4: Write test: tower placement at direction change**

```python
# Setup: 20×20 uniform raster, straight-line path from (0,10) to (19,10).
# Profile: min_span=3 cells, max_span=10 cells.
# Pre-seed with a state at cell (5,10) facing east, span=4 (>= min_span).
# Launch WTB with this work item.
# Verify: tower records written for direction-change neighbors.
# Verify: new states have span_bin=0 (reset after tower).
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_wtb"`

- [ ] **Step 6: Commit**

---

### Task 6: MTB Manager Logic (adds_mtb.cuh)

**Files:**
- Create: `pyorps/utils/kernels/adds_mtb.cuh`
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Implement the MTB main loop: cooperative bucket scanning, AF assignment, dynamic delta, termination.

- [ ] **Step 1: Create adds_mtb.cuh**

```c
// Device function: mtb_main_loop(...)
// Block 0 (all 256 threads cooperate):
// 1. Warp 0: cooperative_scan_buckets() — each thread checks one bucket's WCC
// 2. Warps 1-7: cooperative_scan_AFs() — scan AF statuses, count idle WTBs
// 3. If idle WTBs exist: cooperative_read_segments() — bulk read from head bucket
// 4. Assign items to idle WTBs via AF (status=1, bucket, offset, count)
// 5. Multi-bucket assignment if idle WTBs remain
// 6. adjust_delta() — periodic, only at safe transition points
// 7. Cleanup fully consumed buckets (reset generation)
// 8. Early termination check (best_target_dist * margin)
// 9. Double-empty-sweep termination
//
// Reference: spec "MTB Manager Logic" section
// Reference: ADDS paper (Wang 2021) Section 5 for delegation pattern
```

- [ ] **Step 2: Write test: MTB assigns pre-filled bucket to WTBs**

```python
# Setup: pre-fill bucket 0 with 64 work items (2 segments of 32).
# Launch with 3 blocks: block 0 = MTB, blocks 1-2 = WTBs (but WTBs just poll AF and exit).
# Verify: MTB reads 64 items, sets AF for both WTBs.
# Verify: AF[0].count + AF[1].count == 64.
```

- [ ] **Step 3: Write test: dynamic delta adjustment**

```python
# PRECONDITION: delta adjustment requires "safe transition point" — all buckets
# below current head must be empty and all WTBs must be idle (no in-flight work).
# Test 1 (low util): Set initial_delta=100. Pre-fill only bucket 5 with items.
#   All WTBs idle (AF.status=0), buckets 0-4 empty → safe transition point.
#   After MTB runs: verify delta increased to 200 (avg_util < 0.5 → *= 2.0).
# Test 2 (high util): Set all WTBs to AF.status=2 (busy), fill utilization_history
#   with 0.98 values. At safe point: verify delta decreased (× 0.8).
# Test 3 (clip floor): Set tail bucket with >65% of items → clip_floor = current_delta.
#   Verify: delta never goes below clip_floor even with high utilization.
```

- [ ] **Step 4: Write test: termination conditions**

```python
# Test 1 (target margin): set best_target_dist to 500.0, head_logical=6, delta=100.
#   head_logical * delta = 600 > 500 * 1.0001 → should terminate.
# Test 2 (double empty sweep): all buckets empty, two consecutive scans → done.
# Test 3 (max iterations): set max_assignments low (10), verify kernel exits.
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_mtb"`

- [ ] **Step 6: Commit**

---

### Task 7: Main Kernel Assembly (constrained_adds.cu)

**Files:**
- Create: `pyorps/utils/kernels/constrained_adds.cu`
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Assemble the complete persistent kernel by combining all headers. The kernel entry point dispatches to MTB or WTB based on `blockIdx.x`.

- [ ] **Step 1: Create constrained_adds.cu**

```c
// #include all headers (adds_common, adds_bucket_queue, dynamic_blocks,
//                       clearance, adds_tower, adds_wtb, adds_mtb)
//
// __global__ void constrained_adds_main(
//     // Raster inputs
//     const unsigned short* raster, int rows, int cols,
//     // DEM/obstacle (optional, guarded by HAS_DEM/HAS_CLEARANCE)
//     const float* dem, const float* obstacle,
//     // Block-sparse distance storage
//     BlockEntry* pool, __half* span_pool,
//     int* cell_to_block, int* block_to_cell, int* n_allocated,
//     int max_sparse_blocks,
//     // Bucket queue
//     WorkItem* bucket_pool, int* bucket_meta, int* bucket_wcc,
//     int* next_free_pool_block, int max_bucket_pool_blocks,
//     // Assignment flags
//     AssignmentFlag* assignment_flags, int n_wtbs,
//     // Tower records
//     TowerRecordV4* tower_records, int max_tower_records,
//     // Control buffer
//     int* control,
//     // Best target distance (atomic)
//     int* best_target_dist, int target_cell,
//     // Profile data (angles, towers, heights, steps, intermediates)
//     const signed char* steps_gpu, const float* cost_factors_gpu,
//     const float* step_distances_gpu,  // (n_dirs,) physical distance per step in meters
//     const short* intermediates_gpu, const int* n_intermediates_gpu,
//     const unsigned char* angle_valid_gpu, const float* angle_cost_gpu,
//     const float* tower_terrain_costs_gpu, const float* tower_angle_costs_gpu,
//     const float* height_premiums_gpu,
//     const float* tower_heights_gpu,   // (n_heights,) actual tower heights in meters
//     // Parameters
//     int n_dirs, int n_span_bins, int n_heights,
//     float min_span, float max_span, float span_bin_size,
//     float cell_size, float initial_delta, float margin,
//     float cond_weight, float cond_tension, float min_clearance,
//     float gradient_scale, float max_gradient_pct,
//     int max_assignments
// ) {
//     // Load shared memory (all blocks)
//     load_shared_memory(...);
//     __syncthreads();
//
//     if (blockIdx.x == 0) {
//         mtb_main_loop(...);
//     } else {
//         wtb_main_loop(blockIdx.x, ...);
//     }
// }
```

- [ ] **Step 2: Write end-to-end test on tiny raster (10×10 uniform)**

```python
# Full kernel launch: init source, launch persistent kernel, read results.
# 10×10 uniform raster (cost=100), source=(0,0), target=(9,9).
# Simple profile: 8 dirs, 2 span bins, 1 height, min_span=2, max_span=5.
# No DEM/clearance.
# Verify: path found (best_target_dist < inf).
# Verify: tower records exist.
# Don't verify optimality yet — just that it terminates and finds a path.
```

- [ ] **Step 3: Write test on 50×50 uniform raster**

```python
# Same as above but larger. Verify path cost is reasonable
# (within 2x of straight-line lower bound).
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_kernel"`

- [ ] **Step 5: Commit**

---

### Task 8: Python Driver (constrained_sssp_gpu_v4.py)

**Files:**
- Create: `pyorps/utils/constrained_sssp_gpu_v4.py`
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Full Python driver: memory budget computation, kernel source loading with `#include` resolution, GPU array allocation, kernel launch, result download, path reconstruction.

- [ ] **Step 1: Write Python driver skeleton**

```python
# Model after constrained_sssp_gpu_v3.py (lines 611-800) for the function signature.
# Key sections:
# 1. _load_v4_kernel_source() — read .cu/.cuh files, resolve #include, inject #defines
#    Test: verify no remaining #include directives in resolved output
# 2. _compute_block_size(spc, n_cells, free_vram) — BLOCK_SIZE auto-sizing
#    Test: returns power of 2, clamp(32, 1024), respects VRAM budget
# 3. _compute_initial_delta(raster, cost_factors, n_dirs) — bimodal heuristic
#    Test: returns 2.0 * mean_terrain * n_dirs for uniform raster
# 4. constrained_sssp_raster_gpu_v4(...) — main entry:
#    a. Compute LUTs via prepare_step_lookup_tables() (from traversal_gpu.py:79-130)
#       and profile methods (precompute_angle_lut, precompute_tower_terrain_costs,
#       precompute_tower_angle_costs, precompute_height_premium, compute_step_distances)
#       NOTE: LUTs are class attributes in ConstrainedPathFinder.__init__, but the
#       V4 function receives them as pre-computed numpy arrays (same as V3 signature)
#    b. Auto-size BLOCK_SIZE and max_sparse_blocks
#    c. Allocate all GPU arrays (pool, span_pool, cell_to_block, bucket queue, etc.)
#    d. Launch init kernel (adds_init.cu)
#    e. Launch persistent kernel (constrained_adds.cu) with cooperative groups
#    f. Check overflow counters (CTL_BLOCK_OVERFLOW, CTL_POOL_OVERFLOW) → warn if > 0
#    g. Download tower records + block-sparse pool
#    h. Reconstruct path via _reconstruct_path_v4()
# 5. _reconstruct_path_v4(tower_records, dist_cpu, span_cpu, cell_to_block_cpu,
#                          source_cell, target_cell, n_dirs, n_span_bins, n_heights,
#                          BLOCK_SIZE, steps) — CPU-side path reconstruction:
#    a. Find best target state: scan all local_keys in target cell's block,
#       find the one with minimum dist. Download pool via cp.asnumpy() first.
#    b. Build tower_map: dict[state_after → TowerRecordV4] from downloaded records
#    c. Walk backward from best target state:
#       - If state in tower_map: record tower, jump to pred_state
#       - Else: direction-walk backward one cell (reverse the step direction)
#    d. Reverse path, extract tower_cells and tower_heights
#    e. Return (path_cells, tower_cells, tower_heights)
#    Reference V3's inline reconstruction at constrained_sssp_gpu_v3.py lines 395-503
#    for the tower-chain walking pattern
```

Match the V3 function signature exactly (lines 611-644 of `constrained_sssp_gpu_v3.py`) so the path finder can dispatch to it with the same arguments. Accept `area_offsets`/`area_offset_starts`/`area_offset_counts` parameters but ignore them (exact tower area mode is deferred; log a warning if passed). Return type: `tuple[np.ndarray, np.ndarray, np.ndarray]` (path_cells, tower_cells, tower_heights).

- [ ] **Step 2: Write test: driver with simple 50×50 raster**

```python
# Call constrained_sssp_raster_gpu_v4() directly with a uniform 50×50 raster.
# Verify: returns (path_cells, tower_cells, tower_heights) with non-empty arrays.
# Verify: path starts near source, ends near target.
# Verify: tower_cells are subset of path_cells.
```

- [ ] **Step 3: Write path reconstruction test**

```python
# Use a known 20×20 raster where optimal path is a straight line.
# Verify: reconstructed path is a straight line.
# Verify: towers are spaced between min_span and max_span.
# Verify: path cost equals sum of edge costs + tower costs.
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_driver"`

- [ ] **Step 5: Commit**

---

### Task 9: Cython Comparison Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Validate GPU V4 results against Cython constrained Dijkstra as ground truth.

- [ ] **Step 1: Write comparison test on 100×100 random raster (no eviction)**

```python
# Generate random raster (costs 10-1000, some 65535 impassable).
# Use a small profile where BLOCK_SIZE >= spc (no eviction): 8 dirs, 2 span bins, 1 height = spc=16.
# Run Cython constrained Dijkstra to get reference path + cost.
# Run GPU V4 to get test path + cost.
# Assert: GPU cost <= Cython cost * 1.001 (within 0.1% — tight, no eviction).
# Assert: GPU tower count within ±1 of Cython.
# Assert: all GPU towers satisfy min_span <= span <= max_span.
# Assert: all GPU turn angles satisfy hard_angle_limit.
# Assert: path cost == sum of edge costs + tower costs (cost decomposition check).
```

Reference `tests/test_graph/test_constrained_gpu_v3.py` for how V3 comparison tests are structured (look for `test_cython_comparison` or similar patterns).

- [ ] **Step 2: Write comparison test on 200×200 with DEM + clearance**

```python
# Generate random raster + synthetic DEM (gradient terrain).
# Profile with clearance enabled (tower_height=30m, min_clearance=8m).
# Run both Cython and GPU V4.
# Assert: costs match within 0.5%.
# Assert: all towers pass clearance check (invariant 4: recheck catenary on CPU).
# Assert: tower heights match (if variable height mode).
```

- [ ] **Step 3: Write comparison test with eviction (high spc)**

```python
# 100×100 random raster with R4-like profile (16 dirs, 3 span bins, 3 heights = spc=144).
# Force BLOCK_SIZE=32 to trigger eviction.
# Assert: GPU cost <= Cython cost * 1.01 (1% tolerance for eviction).
# Assert: CTL_BLOCK_OVERFLOW counter reported (some eviction expected).
# Assert: path is still valid (all constraints satisfied).
```

- [ ] **Step 3: Write wall-with-gap test (correctness edge case)**

```python
# 50×50 raster with impassable wall at column 25, gap at row 25.
# Source at (25, 10), target at (25, 40).
# Path must route through the gap.
# Assert: no path cell is on the wall (raster==65535).
# Assert: path passes through or near the gap at (25, 25).
```

- [ ] **Step 5: Write span enforcement tests**

```python
# Test min_span: path with short segments → towers not placed before min_span.
# Test max_span: long straight section → towers placed before max_span exceeded.
# For each tower in result: assert min_span <= span <= max_span.
```

- [ ] **Step 6: Write pool exhaustion warning test**

```python
# Use a tiny max_visited_fraction (0.001) to force block-sparse pool exhaustion.
# Run GPU V4 on a 100×100 raster.
# Verify: CTL_BLOCK_OVERFLOW > 0 in control buffer.
# Verify: Python driver logs a warning about pool exhaustion.
```

- [ ] **Step 7: Run all comparison tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_cython" --timeout=120`

- [ ] **Step 8: Commit**

---

### Task 10: ConstrainedPathFinder Integration

**Files:**
- Modify: `pyorps/graph/constrained_path_finder.py:27-28` (add "raster_gpu_v4" to SUPPORTED_BACKENDS)
- Modify: `pyorps/graph/constrained_path_finder.py:394-456` (add V4 dispatch branch)
- Modify: `tests/test_graph/test_constrained_gpu_v4.py`

Wire the V4 driver into the user-facing `ConstrainedPathFinder` API.

- [ ] **Step 1: Add "raster_gpu_v4" to SUPPORTED_BACKENDS**

In `constrained_path_finder.py` line 27-28, add `"raster_gpu_v4"` to the tuple.

- [ ] **Step 2: Add V4 dispatch branch in _find_route_coupled()**

Model after the V3 dispatch branch at lines 394-456. Add a new `elif self._constrained_backend == "raster_gpu_v4":` block that:
1. Imports `constrained_sssp_raster_gpu_v4` from `pyorps.utils.constrained_sssp_gpu_v4`
2. Calls it with the same argument preparation as V3
3. Processes the return value the same way

- [ ] **Step 3: Write integration test via ConstrainedPathFinder API**

```python
# Create a ConstrainedPathFinder with graph_api="raster_gpu_v4".
# Use a 100×100 raster, simple OHL profile.
# Call finder.find_route().
# Verify: returns a ConstrainedPath with valid geometry, towers, cost.
```

- [ ] **Step 4: Run integration test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v4.py -v -k "test_path_finder"`

- [ ] **Step 5: Commit**

---

### Task 11: Performance Benchmarking & Tuning

**Files:**
- Create: `examples/benchmark_constrained_gpu_v4.py`

Not a test — a manual benchmark script to measure speedup vs Cython.

- [ ] **Step 1: Write benchmark script**

```python
# Generate random rasters at 500², 1000², 2000² with R3 and R4 profiles.
# For each: run Cython, run GPU V4, print timing comparison.
# Print: speedup ratio, memory usage, overflow counters.
# Target: 5-10x speedup over Cython at 1000² R4.
```

Reference `benchmark.py` in project root for existing benchmark patterns.

- [ ] **Step 2: Run benchmarks, document results**

- [ ] **Step 3: Tune parameters if needed**

Based on benchmark results, adjust:
- Initial delta heuristic
- BLOCK_SIZE sizing formula
- max_visited_fraction default
- Bucket pool budget
- Chunk size for MTB assignments

- [ ] **Step 4: Commit any tuning changes**

---

## Implementation Notes

### CUDA Development Workflow

Since CuPy RawKernel compiles CUDA at runtime, the development cycle is:
1. Edit `.cuh`/`.cu` file
2. Run Python test (triggers recompilation)
3. Check output — CuPy shows NVCC errors as Python exceptions

No separate `nvcc` build step needed. The `#include` resolution is done in the Python driver's `_load_v4_kernel_source()` function (same pattern as V2/V3).

### Blackwell GPU Gotchas (from project memory)

- **`grid.sync()` broken on sm_120**: Use custom barrier in `grid_barrier.cuh` for init kernels. The persistent ADDS kernel doesn't use global barriers (async MTB instead).
- **L1 cache bypass**: Compile with `-Xptxas -dlcm=cg` — essential for persistent kernels.
- **`__ballot_sync(0xFFFFFFFF)` in divergent code**: Use `__activemask()` or `atomicAdd` instead.
- **Block count limit**: Max 2 blocks/SM (28 total on 14-SM GPU). Already accounted for in launch config.
- **`volatile int*` for control buffers**: Required for cooperative kernel polling loops. Specifically:
  - `AssignmentFlag* af` must be cast to `volatile int*` in WTB AF polling loop
  - `control[]` array reads in MTB (done flag, best_target_dist) must use volatile
  - Bucket `resv_ptr` reads by MTB must use volatile (WTBs write concurrently)
  - `bucket_generation[]` reads by WTBs must use volatile (MTB writes on reset)
- **`FORBIDDEN = 65535`**: Define as compile-time constant in `adds_common.cuh`. Used to skip impassable cells in both WTB edge relaxation and tower terrain cost checks. Same as V3's `max_cost` parameter.

### Key Reference Patterns

- **#include resolution**: See `constrained_sssp_gpu_v3.py` `_load_v3_kernel_source()` or `constrained_sssp_gpu_v2.py` `_load_kernel_source()` for the regex-based include resolution.
- **Kernel compilation**: See `sssp_gpu.py` lines 893-907 for the cooperative groups + `-dlcm=cg` compilation pattern.
- **State encoding**: V3 `constrained_sssp_gpu_v3.py` lines 50-80 for pack/unpack.
- **Path reconstruction**: V3 `constrained_sssp_gpu_v3.py` `_reconstruct_path_v3()` for tower-record-based reconstruction.
- **Profile LUT preparation**: `constrained_path_finder.py` `_find_route_coupled()` lines 285-320 for how profile LUTs are computed and passed to the GPU function.

### Testing Against Cython

To get a Cython reference result for comparison:
```python
from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d
# or
from pyorps.utils._constrained_delta import constrained_delta_stepping_2d
```
See `tests/test_graph/test_constrained_cython.py` for example usage patterns.

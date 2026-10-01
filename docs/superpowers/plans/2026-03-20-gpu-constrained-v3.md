# GPU Constrained V3: Frontier-Based Delta-Stepping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V2 persistent cooperative GPU kernel with frontier-based delta-stepping and dynamic block-sparse storage, enabling constrained OHL routing on 6 GB GPUs while scaling to larger hardware.

**Architecture:** Multi-launch CUDA kernels dispatched from a Python loop. Each kernel processes the current frontier and exits. Storage uses on-demand block allocation: a `cell_to_block` map gives O(1) cell→block lookup, with blocks allocated from a pre-initialized pool via lock-free atomicCAS. Within each block, open-addressing hash stores (local_key, dist) pairs.

**Tech Stack:** CuPy (RawKernel), CUDA C++17, NumPy, Python 3.11+

**Spec:** `docs/superpowers/specs/2026-03-20-gpu-constrained-v3-frontier-dynamic-blocks-design.md`

**Constraints:**
- NEVER use git commands — user manages git themselves
- Leave CPU/GPU headroom — never exhaust system resources
- No A* heuristic in Dijkstra — user explicitly does not want this
- Test command: `.venv/Scripts/python.exe -m pytest <path> -v`
- Build Cython: `python setup.py build_ext --inplace` (only if .pyx files changed)

---

## File Structure

### New CUDA headers (`pyorps/utils/kernels/`)
| File | Responsibility |
|------|---------------|
| `dynamic_blocks.cuh` | V3 control constants, `make_local_key()` (copied from `state_access.cuh`), `shfl_sync_i64()` (copied), `get_block()` allocator, `block_find_dyn`/`block_upsert_dyn`/`block_relax_dyn` |

**IMPORTANT:** `dynamic_blocks.cuh` must NOT include `state_access.cuh` or `block_sparse.cuh`. Those V2 headers define `BlockEntry`, `BLOCK_SIZE`, `BLOCK_EMPTY`, and `local_hash()` — the same symbols V3 defines independently. Including both causes redefinition errors. Instead, `dynamic_blocks.cuh` includes only `common.cuh` (for `TowerRecord`) and self-contains copies of `make_local_key()` and `shfl_sync_i64()`.

V3 control constants (`V3_CTL_OUTPUT`, etc.) are defined in `dynamic_blocks.cuh` — the single source of truth for all V3 `.cu` files.

### New CUDA kernels (`pyorps/utils/kernels/`)
| File | Responsibility |
|------|---------------|
| `relax_constrained_v3.cu` | Main relax kernel (light+heavy) |
| `classify_bucket.cu` | Bucket classification from pending queue |
| `scan_min.cu` | Min-dist scan + bucket extraction kernels |
| `init_dynamic.cu` | Pool init + source state init |

All V3 `.cu` files include `dynamic_blocks.cuh` (for storage ops + control constants) and `clearance.cuh` (for warp-cooperative clearance). They do NOT include `state_access.cuh` or `block_sparse.cuh`.

### New Python files
| File | Responsibility |
|------|---------------|
| `pyorps/utils/constrained_sssp_gpu_v3.py` | Python wrapper: memory allocation, kernel compilation/caching, driver loop, path reconstruction |
| `tests/test_graph/test_constrained_gpu_v3.py` | Full test suite |

### Existing files reused (read-only, `#include` from V3 headers)
| File | What's reused |
|------|---------------|
| `pyorps/utils/kernels/common.cuh` | `TowerRecord` struct only (included by `dynamic_blocks.cuh`) |
| `pyorps/utils/kernels/clearance.cuh` | `warp_cooperative_clearance()` unchanged (included by `relax_constrained_v3.cu`) |

**NOT included by V3:** `state_access.cuh`, `block_sparse.cuh`, `hash_table.cuh`, `grid_barrier.cuh`

### Existing file modified
| File | Change |
|------|--------|
| `pyorps/graph/constrained_path_finder.py:327-390` | Add `"raster_gpu_v3"` backend option + update `SUPPORTED_BACKENDS` |

---

## Task 1: Dynamic Block Allocator Header

**Files:**
- Create: `pyorps/utils/kernels/dynamic_blocks.cuh`

This is the foundation — all other CUDA files depend on it.

- [ ] **Step 1: Create `dynamic_blocks.cuh` with BlockEntry struct and get_block()**

```cuda
// File: pyorps/utils/kernels/dynamic_blocks.cuh
#pragma once
#include "common.cuh"
// NOTE: Do NOT include state_access.cuh or block_sparse.cuh here.
// V3 is self-contained to avoid symbol conflicts with V2 headers.

// BLOCK_SIZE injected at compile time by Python wrapper via #define
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#define BLOCK_MASK (BLOCK_SIZE - 1)
#define BLOCK_EMPTY 0xFFFF

// ---- V3 control buffer indices (single source of truth) ----
#define V3_CTL_OUTPUT   0
#define V3_CTL_SETTLED  1
#define V3_CTL_PENDING  2
#define V3_CTL_NEAR     3
#define V3_CTL_FAR      4
#define V3_CTL_TOWER    5
#define V3_CTL_MIN_DIST 6
#define V3_CTL_OVERFLOW 7

// ---- Helpers copied from state_access.cuh (avoid transitive V2 includes) ----
__device__ __forceinline__ long long shfl_sync_i64(unsigned int mask, long long val, int src) {
    int lo = (int)(val & 0xFFFFFFFF);
    int hi = (int)((val >> 32) & 0xFFFFFFFF);
    lo = __shfl_sync(mask, lo, src);
    hi = __shfl_sync(mask, hi, src);
    return ((long long)hi << 32) | (unsigned int)lo;
}

__device__ __forceinline__ unsigned short make_local_key(
    int dir, int span_bin, int height_class, int n_span_bins, int n_heights
) {
    return (unsigned short)(dir * n_span_bins * n_heights
                           + span_bin * n_heights + height_class);
}

struct __align__(8) BlockEntry {
    unsigned short local_key;
    unsigned short _pad;
    float dist;
};

// ---- Block allocator: lock-free, idempotent ----
__device__ __forceinline__ int get_block(
    int cell, int* cell_to_block, int* block_to_cell,
    int* n_allocated, int max_blocks
) {
    int idx = cell_to_block[cell];
    if (idx >= 0) return idx;

    int new_idx = atomicAdd(n_allocated, 1);
    if (new_idx >= max_blocks) return -1;  // pool exhausted

    int old = atomicCAS(&cell_to_block[cell], -1, new_idx);
    if (old == -1) {
        block_to_cell[new_idx] = cell;
        return new_idx;
    }
    // Race lost — new_idx wasted (pool entry stays at init: BLOCK_EMPTY/1e30)
    return old;
}

// ---- Dynamic base offset ----
__device__ __forceinline__ int block_offset_dyn(int block_index) {
    return block_index * BLOCK_SIZE;
}

// ---- Hash a local key into [0, BLOCK_SIZE) ----
__device__ __forceinline__ int local_hash(unsigned short local_key) {
    unsigned int h = (unsigned int)local_key * 2654435761u;
    return (int)(h & BLOCK_MASK);
}

// ---- Find entry in block ----
__device__ __forceinline__ BlockEntry* block_find_dyn(
    BlockEntry* pool, int block_index, unsigned short local_key
) {
    int base = block_offset_dyn(block_index);
    int h = local_hash(local_key);
    for (int probe = 0; probe < BLOCK_SIZE; probe++) {
        int slot = (h + probe) & BLOCK_MASK;
        BlockEntry* e = &pool[base + slot];
        unsigned short k = e->local_key;
        if (k == local_key) return e;
        if (k == BLOCK_EMPTY) return NULL;
    }
    return NULL;
}

// ---- Upsert: insert-or-find with atomic CAS ----
__device__ __forceinline__ BlockEntry* block_upsert_dyn(
    BlockEntry* pool, int block_index, unsigned short local_key
) {
    int base = block_offset_dyn(block_index);
    int h = local_hash(local_key);
    for (int probe = 0; probe < BLOCK_SIZE; probe++) {
        int slot = (h + probe) & BLOCK_MASK;
        BlockEntry* e = &pool[base + slot];
        unsigned short k = e->local_key;
        if (k == local_key) return e;
        if (k == BLOCK_EMPTY) {
            unsigned int expected = (unsigned int)BLOCK_EMPTY | ((unsigned int)0xFFFF << 16);
            unsigned int desired = (unsigned int)local_key | ((unsigned int)0 << 16);
            unsigned int old = atomicCAS((unsigned int*)e, expected, desired);
            unsigned short old_key = (unsigned short)(old & 0xFFFF);
            if (old_key == BLOCK_EMPTY || old_key == local_key) return e;
        }
    }
    return NULL;  // block full
}

// ---- Read distance from dynamic block ----
__device__ __forceinline__ float block_read_dist_dyn(
    BlockEntry* pool, int block_index, unsigned short local_key
) {
    if (block_index < 0) return 1e30f;
    BlockEntry* e = block_find_dyn(pool, block_index, local_key);
    return (e != NULL) ? e->dist : 1e30f;
}

// ---- Read span from separate span array ----
__device__ __forceinline__ float block_read_span_dyn(
    __half* span_pool, BlockEntry* pool, int block_index, unsigned short local_key
) {
    if (block_index < 0) return 0.0f;
    int base = block_offset_dyn(block_index);
    int h = local_hash(local_key);
    for (int probe = 0; probe < BLOCK_SIZE; probe++) {
        int slot = (h + probe) & BLOCK_MASK;
        if (pool[base + slot].local_key == local_key)
            return __half2float(span_pool[base + slot]);
        if (pool[base + slot].local_key == BLOCK_EMPTY) return 0.0f;
    }
    return 0.0f;
}

// ---- Relax with eviction: atomicMin on dist, write span on improvement ----
__device__ __forceinline__ int block_relax_dyn(
    BlockEntry* pool, __half* span_pool,
    int block_index, unsigned short local_key,
    float new_dist, float new_span_m
) {
    if (block_index < 0) return 0;  // pool exhausted
    BlockEntry* e = block_upsert_dyn(pool, block_index, local_key);
    if (e != NULL) {
        int ndi = __float_as_int(new_dist);
        int odi = atomicMin((int*)&e->dist, ndi);
        if (ndi < odi) {
            int base = block_offset_dyn(block_index);
            int slot = (int)(e - &pool[base]);
            span_pool[base + slot] = __float2half(new_span_m);
            return 1;
        }
        return 0;
    }
    // Block full — eviction: find worst (highest dist) entry
    int base = block_offset_dyn(block_index);
    int worst_slot = 0;
    float worst_dist = pool[base].dist;
    for (int s = 1; s < BLOCK_SIZE; s++) {
        float d = pool[base + s].dist;
        if (d > worst_dist) { worst_dist = d; worst_slot = s; }
    }
    if (new_dist >= worst_dist) return 0;
    BlockEntry* victim = &pool[base + worst_slot];
    unsigned int old_val = *(unsigned int*)victim;
    unsigned int new_val = (unsigned int)local_key;
    unsigned int cas = atomicCAS((unsigned int*)victim, old_val, new_val);
    if (cas == old_val) {
        victim->dist = new_dist;
        span_pool[base + worst_slot] = __float2half(new_span_m);
        return 1;
    }
    return 0;
}
```

- [ ] **Step 2: Verify file is syntactically valid**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; p = Path('pyorps/utils/kernels/dynamic_blocks.cuh'); assert p.exists(); print(f'OK: {len(p.read_text())} chars')"`

---

## Task 2: Init Kernels

**Files:**
- Create: `pyorps/utils/kernels/init_dynamic.cu`

- [ ] **Step 1: Create `init_dynamic.cu` with pool init and source init kernels**

```cuda
// File: pyorps/utils/kernels/init_dynamic.cu
#include "dynamic_blocks.cuh"
// make_local_key and get_block provided by dynamic_blocks.cuh

extern "C" __global__
void init_pool_v3(BlockEntry* pool, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        pool[i].local_key = BLOCK_EMPTY;
        pool[i]._pad = 0;
        pool[i].dist = 1e30f;
    }
}

extern "C" __global__
void init_source_v3(
    BlockEntry* pool, __half* span_pool,
    int* cell_to_block, int* block_to_cell,
    int* n_allocated,
    long long* source_states, float* init_dists,
    int n_source, int spc, int n_span_bins, int n_heights,
    int max_blocks
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_source) {
        long long state = source_states[i];
        int cell = (int)(state / spc);
        int rem = (int)(state % spc);
        int sh_val = n_span_bins * n_heights;
        int dir = rem / sh_val;
        int rem2 = rem % sh_val;
        int sb = rem2 / n_heights;
        int hc = rem2 % n_heights;
        unsigned short lk = make_local_key(dir, sb, hc, n_span_bins, n_heights);

        int block_idx = get_block(cell, cell_to_block, block_to_cell,
                                  n_allocated, max_blocks);
        if (block_idx < 0) return;  // pool exhausted at init (shouldn't happen)

        BlockEntry* e = block_upsert_dyn(pool, block_idx, lk);
        if (e != NULL) {
            e->dist = init_dists[i];
            int base = block_offset_dyn(block_idx);
            int slot = (int)(e - &pool[base]);
            span_pool[base + slot] = __float2half(0.0f);
        }
    }
}
```

- [ ] **Step 2: Verify file exists**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; p = Path('pyorps/utils/kernels/init_dynamic.cu'); assert p.exists(); print(f'OK: {len(p.read_text())} chars')"`

---

## Task 3: Classify and Scan Kernels

**Files:**
- Create: `pyorps/utils/kernels/classify_bucket.cu`
- Create: `pyorps/utils/kernels/scan_min.cu`

- [ ] **Step 1: Create `classify_bucket.cu`**

```cuda
// File: pyorps/utils/kernels/classify_bucket.cu
#include "dynamic_blocks.cuh"
// V3 control constants defined in dynamic_blocks.cuh (single source of truth)
// Do NOT include state_access.cuh — V3 is self-contained

extern "C" __global__
void classify_bucket(
    long long* pending, int pending_count,
    int bucket, float delta, int buf_size,
    int* cell_to_block, BlockEntry* pool,
    long long spc, int n_span_bins, int n_heights,
    long long* near_queue, long long* far_queue,
    volatile int* control
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pending_count) return;

    long long state = pending[i];
    long long sh = (long long)n_span_bins * n_heights;

    // Unpack state
    long long cell_ll = state / spc;
    int cell = (int)cell_ll;
    long long rem = state - cell_ll * spc;
    int dir = (int)(rem / sh);
    long long rem2 = rem - (long long)dir * sh;
    int sb = (int)(rem2 / n_heights);
    int hc = (int)(rem2 % n_heights);
    unsigned short lk = make_local_key(dir, sb, hc, n_span_bins, n_heights);

    int block_idx = cell_to_block[cell];
    float d = block_read_dist_dyn(pool, block_idx, lk);

    float blo = bucket * delta;
    float bhi = (bucket + 1) * delta;

    if (d < blo || d >= 1e30f) return;  // already settled or invalid

    if (d < bhi) {
        int p = atomicAdd((int*)&control[V3_CTL_NEAR], 1);
        if (p < buf_size) near_queue[p] = state;
        else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
    } else {
        int p = atomicAdd((int*)&control[V3_CTL_FAR], 1);
        if (p < buf_size) far_queue[p] = state;
        else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
    }
}
```

- [ ] **Step 2: Create `scan_min.cu`**

```cuda
// File: pyorps/utils/kernels/scan_min.cu
#include "dynamic_blocks.cuh"
// V3 control constants defined in dynamic_blocks.cuh (single source of truth)

extern "C" __global__
void scan_min_dist(
    BlockEntry* pool, int* block_to_cell,
    int n_allocated_blocks, float bucket_lower_bound,
    volatile int* control
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    long long scan_size = (long long)n_allocated_blocks * BLOCK_SIZE;

    float local_min = 1e30f;
    for (long long i = gtid; i < scan_size; i += stride) {
        int block_idx = (int)(i / BLOCK_SIZE);
        if (block_to_cell[block_idx] < 0) continue;  // wasted race slot
        if (pool[i].local_key == BLOCK_EMPTY) continue;
        float d = pool[i].dist;
        if (d >= bucket_lower_bound && d < local_min) local_min = d;
    }
    if (local_min < 1e30f)
        atomicMin((int*)&control[V3_CTL_MIN_DIST], __float_as_int(local_min));
}

extern "C" __global__
void extract_bucket(
    BlockEntry* pool, int* block_to_cell,
    int n_allocated_blocks,
    int bucket, float delta,
    long long spc, int n_span_bins, int n_heights,
    long long* output_queue, volatile int* control, int buf_size
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    long long scan_size = (long long)n_allocated_blocks * BLOCK_SIZE;
    long long sh = (long long)n_span_bins * n_heights;

    float blo = bucket * delta;
    float bhi = (bucket + 1) * delta;

    for (long long i = gtid; i < scan_size; i += stride) {
        int block_idx = (int)(i / BLOCK_SIZE);
        int cell = block_to_cell[block_idx];
        if (cell < 0) continue;  // wasted race slot
        unsigned short lk = pool[i].local_key;
        if (lk == BLOCK_EMPTY) continue;
        float d = pool[i].dist;
        if (d >= blo && d < bhi) {
            // Reconstruct full state
            int dir = lk / (n_span_bins * n_heights);
            int rem = lk % (n_span_bins * n_heights);
            int sb = rem / n_heights;
            int hc = rem % n_heights;
            long long state = (long long)cell * spc
                + (long long)dir * sh
                + (long long)sb * n_heights + hc;
            int p = atomicAdd((int*)&control[V3_CTL_NEAR], 1);
            if (p < buf_size) output_queue[p] = state;
            else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
        }
    }
}
```

- [ ] **Step 3: Verify both files exist**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; assert Path('pyorps/utils/kernels/classify_bucket.cu').exists(); assert Path('pyorps/utils/kernels/scan_min.cu').exists(); print('OK')"`

---

## Task 4: Main Relax Kernel

**Files:**
- Create: `pyorps/utils/kernels/relax_constrained_v3.cu`

This is the biggest kernel (~550 lines). It adapts the Phase A + Phase B logic from V2's `constrained_persistent.cu` (lines 306-963) but removes the persistent loop and grid barriers, and replaces storage calls with dynamic block operations.

- [ ] **Step 1: Create the relax kernel file**

The kernel signature and structure:

```cuda
// File: pyorps/utils/kernels/relax_constrained_v3.cu
#include "dynamic_blocks.cuh"   // V3 storage + control constants + make_local_key + shfl_sync_i64
#include "clearance.cuh"        // warp_cooperative_clearance (reused from V2)
// Do NOT include state_access.cuh or block_sparse.cuh — V3 is self-contained

extern "C" __global__
void relax_constrained_v3(
    // Frontier
    long long* frontier, int frontier_count,
    int phase,        // 0 = light (cost <= delta), 1 = heavy (cost > delta)
    int buf_size,
    // Raster data
    const unsigned short* __restrict__ raster,
    int rows, int cols, int max_cost,
    // Step LUTs
    const signed char*  __restrict__ steps,
    const float*        __restrict__ cost_factors,
    const signed char*  __restrict__ inter_lut,
    const int*          __restrict__ n_inter,
    int n_steps, int max_inter_cols,
    // Constrained LUTs
    const float*        __restrict__ angle_cost_lut,
    const unsigned char* __restrict__ angle_valid_lut,
    const float*        __restrict__ step_distances,
    const float*        __restrict__ tower_terrain_lut,
    const float*        __restrict__ tower_angle_lut,
    // Height parameters
    const float*        __restrict__ height_premiums,
    const float*        __restrict__ tower_heights,
    int n_heights,
    // Span parameters
    int n_span_bins, float span_bin_size, int min_span_bin,
    // State space
    long long spc, long long total_states,
    // Dynamic block storage
    int* cell_to_block, int* block_to_cell,
    BlockEntry* pool, __half* span_pool,
    int* n_allocated, int max_blocks,
    // Delta-stepping
    float delta,
    // Output queues + control
    long long* output_queue,
    long long* pending_queue,
    long long* settled_queue,
    volatile int* control,
    // Tower records
    TowerRecord* tower_records, int max_tower_records,
    // DEM + clearance
    const float* __restrict__ dem,
    const float* __restrict__ obstacle,
    float cell_size,
    float cond_weight, float cond_tension, float min_clearance,
    float max_gradient_pct, float gradient_scale,
    // Area cost offsets (NULL if uniform)
    const int* __restrict__ area_offsets,
    const int* __restrict__ area_starts,
    const int* __restrict__ area_counts
) {
    // ... shared memory setup (identical to V2 lines 87-125) ...
    // ... warp-based processing (adapted from V2 lines 306-963) ...
    // Key differences from V2:
    //   - get_block() replaces block_offset(cell)
    //   - block_read_dist_dyn/block_relax_dyn replace read_dist/relax_dist
    //   - All queue writes guarded by buf_size check
    //   - get_block returning -1 → skip + increment CTL_OVERFLOW
    //   - Frontier items appended to settled_queue for later heavy phase
    //   - No persistent loop, no grid_barrier calls
}
```

The inner Phase A (per-thread non-tower, lines 370-441 of V2) and Phase B (warp-cooperative tower, lines 446-638 of V2) are ported with these substitutions:
- `read_dist(dist, state_table, hash_mask, block_entries, state, cell, lk, storage_mode)` → `block_read_dist_dyn(pool, block_idx, lk)` where `block_idx = get_block(cell, ...)`
- `read_span(span_dist, ...)` → `block_read_span_dyn(span_pool, pool, block_idx, lk)`
- `relax_dist(dist, span_dist, ...)` → `block_relax_dyn(pool, span_pool, nb_block_idx, nb_lk, nd, new_span_m)`
- Every `atomicAdd` on queue counters checks result < buf_size before writing
- Frontier items → `settled_queue` via `atomicAdd(CTL_SETTLED)`

**IMPORTANT:** The full kernel code should be written by the implementing agent by adapting V2's `constrained_persistent.cu` lines 87-963. The key reference files are:
- `pyorps/utils/kernels/constrained_persistent.cu` (V2 kernel — the source of Phase A/B logic)
- `pyorps/utils/kernels/dynamic_blocks.cuh` (new — created in Task 1; provides all V3 storage ops, control constants, `make_local_key`, `shfl_sync_i64`)
- `pyorps/utils/kernels/clearance.cuh` (reuse — `warp_cooperative_clearance`)
- `pyorps/utils/kernels/common.cuh` (reuse — `TowerRecord`, included transitively via `dynamic_blocks.cuh`)
- Do NOT include `state_access.cuh` or `block_sparse.cuh` — V3 is self-contained

- [ ] **Step 2: Verify file exists and includes are present**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; src = Path('pyorps/utils/kernels/relax_constrained_v3.cu').read_text(); assert 'dynamic_blocks.cuh' in src; assert 'relax_constrained_v3' in src; print(f'OK: {len(src)} chars')"`

---

## Task 5: Python Wrapper — Kernel Loading and Memory Helpers

**Files:**
- Create: `pyorps/utils/constrained_sssp_gpu_v3.py`

This task creates the Python wrapper with: state encoding (reused from V2), memory budget computation, kernel compilation/caching, shared memory computation, and the `_DynamicBlockDistProxy`.

- [ ] **Step 1: Create the wrapper with helpers (no driver loop yet)**

Reuse from `pyorps/utils/constrained_sssp_gpu_v2.py`:
- `pack_state()` / `unpack_state()` (lines 66-83) — copy as-is
- `_compute_constrained_delta()` (lines 323-352) — copy as-is
- `_reconstruct_from_tower_records()` (lines 418+) — copy as-is
- `_compute_v2_smem()` (lines 298-316) — adapt (same calculation, rename to `_compute_v3_smem`)

New code:
- `_DynamicBlockDistProxy` class (from spec)
- `compute_memory_budget_v3()` — computes pool bytes from max_blocks × BLOCK_SIZE × 10
- `_load_v3_kernel_source()` — loads `.cu` files with `#include` resolution (adapted from V2's `_load_kernel_source`)
- `_get_v3_kernel()` — compiles/caches CuPy RawKernels
- `_ensure_cuda_path()` — copy from V2
- Control buffer constants: `_V3_CTL_OUTPUT = 0` through `_V3_CTL_SIZE = 8`
- `_compute_block_size()` — selects optimal BLOCK_SIZE from spc and VRAM

Key reference: `pyorps/utils/constrained_sssp_gpu_v2.py` lines 1-352 (helpers) and lines 374-412 (`_BlockDistProxy` — adapt to `_DynamicBlockDistProxy`).

Note: `_DynamicBlockDistProxy` only needs `block_entries_cpu` and `cell_to_block_cpu` — NOT `span_pool_cpu`. The `_reconstruct_from_tower_records` function only uses `dist_cpu[state]` for distance lookups, never spans.

- [ ] **Step 2: Write a basic smoke test for helpers**

Create `tests/test_graph/test_constrained_gpu_v3.py` with:
```python
import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")


class TestStateEncodingV3:
    def test_pack_unpack_roundtrip(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import pack_state, unpack_state
        cell, d, sb, hc = 12345, 7, 3, 2
        spc = 32 * 6 * 3
        state = pack_state(cell, d, sb, hc, spc, 6, 3)
        c2, d2, sb2, hc2 = unpack_state(state, spc, 6, 3)
        assert (c2, d2, sb2, hc2) == (cell, d, sb, hc)


class TestBlockSizeSelection:
    def test_block_size_power_of_two(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _compute_block_size
        bs = _compute_block_size(spc=288, vram_free_bytes=4 * 1024**3,
                                 n_cells=2_600_000, max_visited_fraction=0.15)
        assert bs & (bs - 1) == 0  # power of 2
        assert bs >= 32

    def test_block_size_capped_by_vram(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _compute_block_size
        # Very small VRAM: should get minimum block size
        bs = _compute_block_size(spc=1728, vram_free_bytes=500 * 1024**2,
                                 n_cells=2_600_000, max_visited_fraction=0.15)
        assert bs >= 32
        assert bs <= 1728
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py -v`

---

## Task 6: Python Wrapper — Kernel Compilation Tests

**Files:**
- Modify: `pyorps/utils/constrained_sssp_gpu_v3.py`
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write test that compiles each V3 kernel**

```python
class TestKernelCompilation:
    """Test that all V3 CUDA kernels compile without error."""

    def test_init_pool_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("init_pool_v3")
        assert k is not None

    def test_init_source_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("init_source_v3")
        assert k is not None

    def test_classify_bucket_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("classify_bucket")
        assert k is not None

    def test_scan_min_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("scan_min_dist")
        assert k is not None

    def test_extract_bucket_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("extract_bucket")
        assert k is not None

    def test_relax_kernel_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("relax_constrained_v3")
        assert k is not None
```

- [ ] **Step 2: Run tests — fix any compilation errors**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestKernelCompilation -v`

All 6 must PASS. If any fail, fix the CUDA source and re-run.

---

## Task 7: Init Kernel Integration Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write test for pool initialization**

```python
class TestInitKernels:
    def test_init_pool_sets_empty(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        import cupy as cp
        block_entry_dtype = cp.dtype([
            ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)])
        n_entries = 256  # 4 blocks * 64 BLOCK_SIZE
        d_pool = cp.empty(n_entries, dtype=block_entry_dtype)
        kernel = _get_v3_kernel("init_pool_v3")
        kernel((1,), (256,), (d_pool, np.int32(n_entries)))
        cp.cuda.Stream.null.synchronize()
        host = d_pool.get()
        for entry in host:
            assert int(entry['local_key']) == 0xFFFF
            assert float(entry['dist']) > 1e29

    def test_init_source_allocates_blocks(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            _get_v3_kernel, pack_state)
        import cupy as cp
        n_cells = 100
        spc = 8 * 6 * 1  # 8 dirs, 6 span_bins, 1 height
        max_blocks = 50
        block_size = 64  # must match compiled BLOCK_SIZE
        pool_size = max_blocks * block_size

        block_entry_dtype = cp.dtype([
            ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)])
        d_pool = cp.empty(pool_size, dtype=block_entry_dtype)
        d_span = cp.zeros(pool_size, dtype=cp.float16)
        d_c2b = cp.full(n_cells, -1, dtype=cp.int32)
        d_b2c = cp.full(max_blocks, -1, dtype=cp.int32)
        d_n_alloc = cp.zeros(1, dtype=cp.int32)

        # Init pool
        init_pool = _get_v3_kernel("init_pool_v3")
        init_pool(((pool_size + 255)//256,), (256,),
                  (d_pool, np.int32(pool_size)))

        # Create source states: cell 5, all 8 directions, span_bin=0, hc=0
        source_states = []
        source_dists = []
        for d in range(8):
            st = pack_state(5, d, 0, 0, spc, 6, 1)
            source_states.append(st)
            source_dists.append(0.0)
        d_src = cp.asarray(np.array(source_states, dtype=np.int64))
        d_sdist = cp.asarray(np.array(source_dists, dtype=np.float32))

        # Init source
        init_src = _get_v3_kernel("init_source_v3")
        n_source = len(source_states)
        init_src(((n_source + 255)//256,), (min(256, n_source),),
                 (d_pool, d_span, d_c2b, d_b2c, d_n_alloc,
                  d_src, d_sdist, np.int32(n_source),
                  np.int32(spc), np.int32(6), np.int32(1),
                  np.int32(max_blocks)))
        cp.cuda.Stream.null.synchronize()

        # Check: cell 5 should have a block allocated
        c2b_host = d_c2b.get()
        assert c2b_host[5] >= 0, "Cell 5 should have a block"
        assert int(d_n_alloc.get()) >= 1, "At least 1 block allocated"

        # Check: block_to_cell reverse map
        b2c_host = d_b2c.get()
        block_idx = c2b_host[5]
        assert b2c_host[block_idx] == 5, "Reverse map should point back to cell 5"
```

- [ ] **Step 2: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestInitKernels -v`

---

## Task 8: Python Driver Loop

**Files:**
- Modify: `pyorps/utils/constrained_sssp_gpu_v3.py`

- [ ] **Step 1: Implement `constrained_sssp_raster_gpu_v3()` main function**

This is the main entry point. Structure:

1. **Parameter validation** (reuse from V2 lines 776-817)
2. **BLOCK_SIZE computation** via `_compute_block_size()`
3. **GPU allocation**: cell_to_block, block_to_cell, block_pool, span_pool, n_allocated, queues, control, tower_records, input data (raster, DEM, LUTs)
4. **Pool and source initialization** via `init_pool_v3` and `init_source_v3` kernels
5. **Driver loop** (from spec pseudocode, with C2 fix for heavy-phase output):
   - Outer loop per bucket:
     - Light phase: `while frontier_count > 0: launch relax(LIGHT), sync, swap`
     - Early termination: check target block
     - Heavy phase: launch relax(HEAVY) on settled
     - **Check heavy output:** if heavy produced within-bucket states (CTL_OUTPUT > 0), set frontier = output, loop back to light phase (do NOT advance bucket yet)
     - Only advance bucket when both light and heavy phases are quiescent
   - Classify pending into next bucket
   - Full scan fallback via scan_min_dist + extract_bucket
   - Safety: `max_iterations = n_cells * 10` with warning + break if exceeded
6. **Post-run checks**: overflow warning, pool usage warning
7. **Path reconstruction**: download tower records + blocks, find best target state, call `_reconstruct_from_tower_records()` with `_DynamicBlockDistProxy`

Key reference: `pyorps/utils/constrained_sssp_gpu_v2.py` lines 673-1308 (the V2 main function). The setup/allocation/reconstruction is very similar; the kernel launch section is completely different.

**Signature must match V2** (same parameters) plus `max_visited_fraction=0.15`:
```python
def constrained_sssp_raster_gpu_v3(
    raster, source_row, source_col, target_row, target_col,
    steps, angle_cost_lut, angle_valid_lut, step_distances,
    tower_terrain_costs, tower_angle_costs,
    n_span_bins, span_bin_size, min_span, max_span,
    height_premiums=None, n_heights=1, exclude_mask=None,
    dem=None, obstacle_heights=None, cell_size=1.0,
    conductor_weight_per_m=0.0, conductor_tension=1.0,
    min_clearance=0.0, max_gradient_pct=100.0, gradient_scale=2.0,
    tower_heights=None, area_offsets=None, area_offset_starts=None,
    area_offset_counts=None, threads_per_block=256,
    margin=1.00001, max_tower_records=2_000_000,
    max_visited_fraction=0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

- [ ] **Step 2: Verify the module imports without error**

Run: `.venv/Scripts/python.exe -c "from pyorps.utils.constrained_sssp_gpu_v3 import constrained_sssp_raster_gpu_v3; print('OK')"`

---

## Task 9: End-to-End Path Finding Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write small uniform raster test (50x50)**

```python
class TestPathFinding:
    """End-to-end constrained path finding on GPU V3."""

    @staticmethod
    def _make_test_params(rows=50, cols=50, raster_val=10,
                          cell_size=10.0, min_span=50.0, max_span=300.0,
                          span_bin_size=50.0, n_dirs=8):
        """Create a uniform raster + minimal constrained profile for testing.

        Returns (raster, kwargs_dict) suitable for passing to
        constrained_sssp_raster_gpu_v3().
        """
        from pyorps.core.infrastructure_profile import InfrastructureProfile

        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        n_span_bins = int(max_span / span_bin_size) + 1
        # Use r1 (8 dirs) for speed. Build steps + LUTs via the profile.
        from pyorps.utils.neighborhood import get_neighborhood
        steps_arr = get_neighborhood(f"r{n_dirs // 8}" if n_dirs >= 8 else "r1")

        profile = InfrastructureProfile(
            voltage_kv=110, min_span_m=min_span, max_span_m=max_span,
            n_span_bins=n_span_bins, span_bin_size_m=span_bin_size,
            tower_terrain_cost_per_unit=1.0,
            soft_angle_limit_deg=15.0, hard_angle_limit_deg=45.0,
        )
        n_steps = len(steps_arr)
        step_dists = np.array([
            np.sqrt(float(steps_arr[i, 0])**2 + float(steps_arr[i, 1])**2) * cell_size
            for i in range(n_steps)], dtype=np.float32)

        # Build angle LUTs
        angle_cost = np.zeros((n_steps, n_steps), dtype=np.float32)
        angle_valid = np.ones((n_steps, n_steps), dtype=np.uint8)
        tower_terrain = np.full(65536, 100.0, dtype=np.float32)
        tower_terrain[65535] = 0.0
        tower_angle = np.zeros((n_steps, n_steps), dtype=np.float32)

        return raster, dict(
            steps=steps_arr,
            angle_cost_lut=angle_cost,
            angle_valid_lut=angle_valid,
            step_distances=step_dists,
            tower_terrain_costs=tower_terrain,
            tower_angle_costs=tower_angle,
            n_span_bins=n_span_bins,
            span_bin_size=span_bin_size,
            min_span=min_span,
            max_span=max_span,
            cell_size=cell_size,
        )

    def test_basic_path_found(self):
        """V3 should find a path on a uniform 50x50 raster."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = self._make_test_params()
        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 49, 49, **kwargs)
        assert len(path) > 0, "Should find a path"
        assert path[0] == 0, "Path should start at source cell (0,0) = 0"
        assert path[-1] == 49 * 50 + 49, "Path should end at target cell"

    def test_no_path_forbidden(self):
        """V3 should return empty when target is surrounded by forbidden."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = self._make_test_params()
        # Wall around target
        raster[48, :] = 65535
        raster[:, 48] = 65535
        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 49, 49, **kwargs)
        assert len(path) == 0, "Should find no path"
```

- [ ] **Step 2: Run the path finding tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestPathFinding -v`

- [ ] **Step 3: Debug and fix until both tests pass**

This is the critical integration step. Common issues:
- Kernel launch grid size off by one
- Control buffer counter not reset at the right time
- Queue swap logic (frontier ↔ output)
- Shared memory size mismatch
- Kernel parameter count/type mismatch

---

## Task 10: Cython Reference Comparison Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write test comparing V3 against Cython reference**

```python
class TestCythonComparison:
    """Compare V3 GPU results against Cython reference implementation."""

    def test_cost_matches_cython_50x50(self):
        """V3 path cost should match Cython within tolerance."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d

        raster = np.full((50, 50), 10, dtype=np.uint16)
        # ... build shared params (steps, LUTs, etc.) ...
        # Run both GPU V3 and Cython
        # Compare: path cost within 1e-3 relative tolerance
        # Compare: same number of towers (or within 1)
        pass  # Implementing agent fills in using _make_test_params pattern

    def test_cost_matches_cython_100x100(self):
        """Larger raster comparison."""
        pass  # Same pattern at 100x100
```

**IMPORTANT constraints for Cython comparison:**
- `constrained_dijkstra_2d` does NOT support `height_premiums`, `n_heights`, `tower_heights`, `conductor_weight_per_m`, `conductor_tension`, `min_clearance`, or `obstacle_heights`.
- Cython comparison tests MUST use `n_heights=1`, no DEM, no clearance, no obstacle heights, and no area cost offsets when calling V3.
- Set `height_premiums=None` and `tower_heights=None` when calling V3 for comparison.
- Check `pyorps/utils/_constrained_dijkstra.pyx` line 40 for the exact Cython function signature.
- Look at how `test_constrained_gpu_v2.py` constructs test cases (class `TestBasicKernel._make_params()` at line 69) for the parameter construction pattern.

- [ ] **Step 2: Run comparison tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestCythonComparison -v`

---

## Task 11: DEM, Clearance, and Area Cost Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write DEM + clearance test**

```python
class TestDEMAndClearance:
    def test_dem_gradient_penalty(self):
        """Path should avoid steep slopes when DEM is provided."""
        # Create raster with a ridge (high DEM values in center row)
        # V3 with DEM should route around the ridge
        pass

    def test_clearance_rejects_short_towers(self):
        """With tall obstacles, only tall tower heights should produce paths."""
        # Create obstacle array with 20m obstacles along path
        # Shortest tower height (e.g., 25m) should fail clearance
        # Taller height should succeed
        pass

    def test_forbidden_area_cost(self):
        """Tower with forbidden pixel in footprint should be rejected."""
        # Place forbidden pixel (65535) adjacent to a cell
        # Use area_offsets that include that pixel
        # Tower should not be placed there
        pass
```

- [ ] **Step 2: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestDEMAndClearance -v`

---

## Task 12: Pool Exhaustion and Eviction Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write pool exhaustion test**

```python
class TestEdgeCases:
    def test_pool_exhaustion_warns(self):
        """When max_visited_fraction is very small, pool exhaustion should warn."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = TestPathFinding._make_test_params(rows=100, cols=100)
        with pytest.warns(UserWarning, match="pool"):
            path, _, _ = constrained_sssp_raster_gpu_v3(
                raster, 0, 0, 99, 99, max_visited_fraction=0.001, **kwargs)

    def test_eviction_still_finds_path(self):
        """With BLOCK_SIZE < spc (forced eviction), path should still be found."""
        # This tests correctness of the eviction policy
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = TestPathFinding._make_test_params(rows=50, cols=50)
        # Force small BLOCK_SIZE by using very small max_visited_fraction
        # combined with many height classes
        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 49, 49, **kwargs)
        assert len(path) > 0
```

- [ ] **Step 2: Run edge case tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestEdgeCases -v`

---

## Task 13: Medium Rasters, Early Termination, and Queue Overflow Tests

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v3.py`

- [ ] **Step 1: Write medium raster tests (200x200)**

```python
class TestMediumRasters:
    def test_200x200_finds_path(self):
        """V3 should find a path on 200x200 raster (realistic frontier sizes)."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = TestPathFinding._make_test_params(rows=200, cols=200)
        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 199, 199, **kwargs)
        assert len(path) > 0
        assert len(towers) > 0

    def test_source_equals_target(self):
        """Degenerate case: source == target should return trivial path."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = TestPathFinding._make_test_params()
        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster, 25, 25, 25, 25, **kwargs)
        # Either empty path or single-cell path is acceptable
        assert len(path) <= 1

    def test_early_termination_faster(self):
        """With a close target, V3 should terminate before exploring everything."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        import time
        raster, kwargs = TestPathFinding._make_test_params(rows=200, cols=200)
        # Close target
        t0 = time.perf_counter()
        path_close, _, _ = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 20, 20, **kwargs)
        t_close = time.perf_counter() - t0
        # Far target
        t0 = time.perf_counter()
        path_far, _, _ = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 199, 199, **kwargs)
        t_far = time.perf_counter() - t0
        assert len(path_close) > 0
        assert len(path_far) > 0
        # Close target should be meaningfully faster (at least 2x)
        # Use generous threshold to avoid flaky CI
        assert t_close < t_far * 0.8 or t_close < 0.5  # fast enough either way

    def test_queue_overflow_no_crash(self):
        """Very small buf_size should not crash, just warn."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)
        raster, kwargs = TestPathFinding._make_test_params(rows=100, cols=100)
        # Force tiny buffer (implementing agent: add buf_size parameter or
        # test indirectly by checking CTL_OVERFLOW after run)
        # The key assertion: no crash, path may be suboptimal
        path, _, _ = constrained_sssp_raster_gpu_v3(
            raster, 0, 0, 99, 99, **kwargs)
        # Should complete without exception
```

- [ ] **Step 2: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py::TestMediumRasters -v`

---

## Task 14: Integration with ConstrainedPathFinder

(Renumbered from original Task 13)

**Files:**
- Modify: `pyorps/graph/constrained_path_finder.py:27` (SUPPORTED_BACKENDS)
- Modify: `pyorps/graph/constrained_path_finder.py:327-390` (GPU dispatch)

- [ ] **Step 1: Add `"raster_gpu_v3"` to SUPPORTED_BACKENDS**

In `constrained_path_finder.py` line 27:
```python
SUPPORTED_BACKENDS = ("cython", "cython_parallel", "raster_gpu", "raster_gpu_v3")
```

- [ ] **Step 2: Add V3 dispatch in `_find_constrained_path`**

After the existing `if backend == "raster_gpu":` block (around line 327), add:
```python
if backend == "raster_gpu_v3":
    try:
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3,
        )
        # ... same parameter preparation as V2 block ...
        gpu_kwargs['max_visited_fraction'] = 0.15
        return constrained_sssp_raster_gpu_v3(**gpu_kwargs)
    except (ImportError, RuntimeError) as e:
        warnings.warn(f"GPU v3 unavailable ({e}), falling back to Cython")
        backend = "cython"
```

- [ ] **Step 3: Verify import works**

Run: `.venv/Scripts/python.exe -c "from pyorps.graph.constrained_path_finder import ConstrainedPathFinder; print('Backends:', ConstrainedPathFinder.SUPPORTED_BACKENDS)"`

Expected output should include `raster_gpu_v3`.

---

## Task 15: Full Test Suite Pass

- [ ] **Step 1: Run the entire V3 test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v3.py -v`

All tests should pass.

- [ ] **Step 2: Run existing V2 tests to verify no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v2.py -v`

V2 tests should still pass (V2 code was not modified).

- [ ] **Step 3: Run existing constrained path finder tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_path_finder.py -v`

Should still pass.

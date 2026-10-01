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
    if (new_idx >= max_blocks) return -1;
    int old = atomicCAS(&cell_to_block[cell], -1, new_idx);
    if (old == -1) {
        block_to_cell[new_idx] = cell;
        return new_idx;
    }
    return old;
}

__device__ __forceinline__ int block_offset_dyn(int block_index) {
    return block_index * BLOCK_SIZE;
}

__device__ __forceinline__ int local_hash(unsigned short local_key) {
    unsigned int h = (unsigned int)local_key * 2654435761u;
    return (int)(h & BLOCK_MASK);
}

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
    return NULL;
}

__device__ __forceinline__ float block_read_dist_dyn(
    BlockEntry* pool, int block_index, unsigned short local_key
) {
    if (block_index < 0) return 1e30f;
    BlockEntry* e = block_find_dyn(pool, block_index, local_key);
    return (e != NULL) ? e->dist : 1e30f;
}

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

__device__ __forceinline__ int block_relax_dyn(
    BlockEntry* pool, __half* span_pool,
    int block_index, unsigned short local_key,
    float new_dist, float new_span_m
) {
    if (block_index < 0) return 0;
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

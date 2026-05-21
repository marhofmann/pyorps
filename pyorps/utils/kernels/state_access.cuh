#pragma once
#include "hash_table.cuh"
#include "block_sparse.cuh"

// Storage mode constants (passed as kernel parameter)
#define STORAGE_DENSE   0
#define STORAGE_SPARSE  1   // global hash table (legacy)
#define STORAGE_BLOCK   2   // block-sparse (BSR-inspired)

// ---- Device helper: shuffle a long long (int64) across warp lanes ----
__device__ __forceinline__ long long shfl_sync_i64(unsigned int mask, long long val, int src) {
    int lo = (int)(val & 0xFFFFFFFF);
    int hi = (int)((val >> 32) & 0xFFFFFFFF);
    lo = __shfl_sync(mask, lo, src);
    hi = __shfl_sync(mask, hi, src);
    return ((long long)hi << 32) | (unsigned int)lo;
}

// ---- Compute local key from state components ----
__device__ __forceinline__ unsigned short make_local_key(
    int dir, int span_bin, int height_class, int n_span_bins, int n_heights
) {
    return (unsigned short)(dir * n_span_bins * n_heights
                           + span_bin * n_heights + height_class);
}

// ---- Distance read: works for all three storage modes ----
__device__ __forceinline__ float read_dist(
    float* dist, StateEntry* state_table, int hash_mask,
    BlockEntry* blocks,
    long long state, int cell, unsigned short local_key,
    int storage_mode
) {
    if (storage_mode == STORAGE_BLOCK) {
        return block_read_dist(blocks, cell, local_key);
    }
    if (storage_mode == STORAGE_SPARSE) {
        StateEntry* e = hash_find(state_table, hash_mask, state);
        return (e != NULL) ? e->dist : 1e30f;
    }
    return dist[state];
}

// ---- Span read: works for all three storage modes ----
__device__ __forceinline__ float read_span(
    __half* span_dist, StateEntry* state_table, int hash_mask,
    BlockEntry* blocks, __half* block_span,
    long long state, int cell, unsigned short local_key,
    int storage_mode
) {
    if (storage_mode == STORAGE_BLOCK) {
        return block_read_span(block_span, cell, local_key, blocks);
    }
    if (storage_mode == STORAGE_SPARSE) {
        StateEntry* e = hash_find(state_table, hash_mask, state);
        return (e != NULL) ? __half2float(e->span_dist) : 0.0f;
    }
    return __half2float(span_dist[state]);
}

// ---- Relax: atomicMin on dist, write span. Returns 1 if improved. ----
__device__ __forceinline__ int relax_dist(
    float* dist, __half* span_dist,
    StateEntry* state_table, int hash_mask,
    BlockEntry* blocks, __half* block_span,
    long long new_state, int new_cell, unsigned short new_local_key,
    float new_dist, float new_span_m,
    int storage_mode
) {
    if (storage_mode == STORAGE_BLOCK) {
        return block_relax(blocks, block_span, new_cell, new_local_key,
                           new_dist, new_span_m);
    }
    int ndi = __float_as_int(new_dist);
    if (storage_mode == STORAGE_SPARSE) {
        StateEntry* ne = hash_upsert(state_table, hash_mask, new_state);
        if (ne == NULL) return 0;
        int odi = atomicMin((int*)&ne->dist, ndi);
        if (ndi < odi) {
            ne->span_dist = __float2half(new_span_m);
            return 1;
        }
        return 0;
    }
    int odi = atomicMin((int*)&dist[new_state], ndi);
    if (ndi < odi) {
        span_dist[new_state] = __float2half(new_span_m);
        return 1;
    }
    return 0;
}

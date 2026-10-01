#pragma once
#include "common.cuh"

// ============================================================================
// Block-Sparse State Storage
// ============================================================================
//
// Two-level dense-sparse structure inspired by cuSPARSE's BSR format.
//
// Level 1: Dense per-cell — O(1) access via cell index * BLOCK_SIZE
// Level 2: Fixed-size open-addressing hash within each cell's block
//
// Each cell has BLOCK_SIZE slots. The local key is:
//   local_key = dir * (n_span_bins * n_heights) + span_bin * n_heights + height
//
// Memory: n_cells * BLOCK_SIZE * sizeof(BlockEntry) bytes
// With BLOCK_SIZE=32 and 8-byte entries: 2.6M cells * 32 * 8 = 665 MB
//
// Advantages over global hash table:
// - Level 1 is O(1) (just multiply, no probing)
// - Level 2 probing stays within 32 slots (2 cache lines)
// - No global overflow risk
// - All states for one cell are contiguous (cache-friendly)
// ============================================================================

// BLOCK_SIZE is set at compile time via Python wrapper based on spc and VRAM.
// Default 64; Python injects the actual value via #define before the source.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#define BLOCK_MASK (BLOCK_SIZE - 1)
#define BLOCK_EMPTY 0xFFFF   // sentinel for empty local key (uint16)

struct __align__(8) BlockEntry {
    unsigned short local_key;   // dir * sh + span_bin * n_heights + height (max ~864)
    unsigned short _pad;        // alignment padding
    float dist;                 // best distance (init to 1e30 via 0x7F memset)
};  // 8 bytes per entry

// Note: span_dist is stored in a SEPARATE flat array indexed the same way:
//   span_array[cell * BLOCK_SIZE + local_slot]
// This keeps BlockEntry at exactly 8 bytes (fits in one atomic transaction).

// ---- Compute block offset for a cell ----
__device__ __forceinline__ int block_offset(int cell) {
    return cell * BLOCK_SIZE;
}

// ---- Hash a local key into [0, BLOCK_SIZE) ----
__device__ __forceinline__ int local_hash(unsigned short local_key) {
    // Simple multiplicative hash for small keys
    unsigned int h = (unsigned int)local_key * 2654435761u;
    return (int)(h & BLOCK_MASK);
}

// ---- Find entry in cell's block ----
// Returns pointer to entry, or NULL if not found.
__device__ __forceinline__ BlockEntry* block_find(
    BlockEntry* blocks, int cell, unsigned short local_key
) {
    int base = block_offset(cell);
    int h = local_hash(local_key);
    for (int probe = 0; probe < BLOCK_SIZE; probe++) {
        int slot = (h + probe) & BLOCK_MASK;
        BlockEntry* e = &blocks[base + slot];
        unsigned short k = e->local_key;
        if (k == local_key) return e;
        if (k == BLOCK_EMPTY) return NULL;
    }
    return NULL;  // block full (all 32 slots occupied by other keys)
}

// ---- Insert-or-find in cell's block ----
// Atomically claims a slot for local_key. Returns pointer, or NULL if block full.
__device__ __forceinline__ BlockEntry* block_upsert(
    BlockEntry* blocks, int cell, unsigned short local_key
) {
    int base = block_offset(cell);
    int h = local_hash(local_key);
    for (int probe = 0; probe < BLOCK_SIZE; probe++) {
        int slot = (h + probe) & BLOCK_MASK;
        BlockEntry* e = &blocks[base + slot];
        unsigned short k = e->local_key;
        if (k == local_key) return e;   // already exists
        if (k == BLOCK_EMPTY) {
            // Try to claim via atomic CAS on the local_key (16-bit)
            // Use 32-bit CAS on the first 4 bytes (local_key + _pad)
            unsigned int expected = (unsigned int)BLOCK_EMPTY | ((unsigned int)0xFFFF << 16);
            unsigned int desired = (unsigned int)local_key | ((unsigned int)0 << 16);
            unsigned int old = atomicCAS((unsigned int*)e, expected, desired);
            unsigned short old_key = (unsigned short)(old & 0xFFFF);
            if (old_key == BLOCK_EMPTY || old_key == local_key) {
                return e;  // claimed or was already claimed by same key
            }
            // Slot taken by different key, continue probing
        }
    }
    return NULL;  // block full
}

// ---- Read distance from block-sparse storage ----
__device__ __forceinline__ float block_read_dist(
    BlockEntry* blocks, int cell, unsigned short local_key
) {
    BlockEntry* e = block_find(blocks, cell, local_key);
    return (e != NULL) ? e->dist : 1e30f;
}

// ---- Read span from separate span array ----
__device__ __forceinline__ float block_read_span(
    __half* block_span, int cell, unsigned short local_key,
    BlockEntry* blocks
) {
    // Find the slot index (not the entry pointer) to index into span array
    int base = block_offset(cell);
    int h = local_hash(local_key);
    for (int probe = 0; probe < BLOCK_SIZE; probe++) {
        int slot = (h + probe) & BLOCK_MASK;
        if (blocks[base + slot].local_key == local_key) {
            return __half2float(block_span[base + slot]);
        }
        if (blocks[base + slot].local_key == BLOCK_EMPTY) return 0.0f;
    }
    return 0.0f;
}

// ---- Relax: atomicMin on dist in block, write span on improvement ----
// If block is full, evicts the worst (highest dist) entry when new_dist is better.
__device__ __forceinline__ int block_relax(
    BlockEntry* blocks, __half* block_span,
    int cell, unsigned short local_key,
    float new_dist, float new_span_m
) {
    BlockEntry* e = block_upsert(blocks, cell, local_key);
    if (e != NULL) {
        // Normal path: slot found or created
        int ndi = __float_as_int(new_dist);
        int odi = atomicMin((int*)&e->dist, ndi);
        if (ndi < odi) {
            int base = block_offset(cell);
            int slot = (int)(e - &blocks[base]);
            block_span[base + slot] = __float2half(new_span_m);
            return 1;
        }
        return 0;
    }

    // Block full — try eviction: find the slot with the worst (highest) dist
    int base = block_offset(cell);
    int worst_slot = 0;
    float worst_dist = blocks[base].dist;
    for (int s = 1; s < BLOCK_SIZE; s++) {
        float d = blocks[base + s].dist;
        if (d > worst_dist) {
            worst_dist = d;
            worst_slot = s;
        }
    }

    // Only evict if our new dist is strictly better than the worst
    if (new_dist >= worst_dist) return 0;

    // Atomic CAS to claim the worst slot: replace its local_key with ours
    BlockEntry* victim = &blocks[base + worst_slot];
    unsigned int old_val = *(unsigned int*)victim;  // {old_key, old_pad}
    unsigned int new_val = (unsigned int)local_key;
    unsigned int cas = atomicCAS((unsigned int*)victim, old_val, new_val);
    if (cas == old_val) {
        // Claimed — write our distance (non-atomic ok, we own the slot now)
        victim->dist = new_dist;
        block_span[base + worst_slot] = __float2half(new_span_m);
        return 1;
    }
    // Another thread evicted first — give up (rare)
    return 0;
}

#pragma once
#include "common.cuh"

// ---- Sparse hash table for distance/span storage ----
// Replaces dense dist[] and span_dist[] arrays when use_sparse == 1.
// 16 bytes per entry, power-of-2 open addressing with linear probing.
// Hash table is memset to 0x7F. Key = 0x7F7F7F7F7F7F7F7F means empty.
// dist = 0x7F7F7F7F (~3.3e38 float, large positive int for atomicMin).
#define HASH_EMPTY 0x7F7F7F7F7F7F7F7FLL

struct __align__(16) StateEntry {
    long long key;      // state index (HASH_EMPTY = unused slot)
    float dist;         // best distance (init to 1e30 on first insert)
    __half span_dist;   // exact span in meters
    __half _pad;        // alignment padding
};

// Murmur3-inspired hash
__device__ __forceinline__ unsigned int hash_state(long long key, int mask) {
    unsigned long long h = (unsigned long long)key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (unsigned int)(h & mask);
}

// Find: returns pointer or NULL (NULL = INFINITY / not visited)
__device__ __forceinline__ StateEntry* hash_find(
    StateEntry* table, int mask, long long key
) {
    unsigned int h = hash_state(key, mask);
    for (int probe = 0; probe < 128; probe++) {
        unsigned int idx = (h + probe) & mask;
        long long k = table[idx].key;
        if (k == key) return &table[idx];
        if (k == HASH_EMPTY) return NULL;
    }
    return NULL;
}

// Insert-or-find: atomically claim slot, init dist to 1e30
__device__ __forceinline__ StateEntry* hash_upsert(
    StateEntry* table, int mask, long long key
) {
    unsigned int h = hash_state(key, mask);
    unsigned long long ukey = (unsigned long long)key;
    unsigned long long empty = (unsigned long long)HASH_EMPTY;
    for (int probe = 0; probe < 128; probe++) {
        unsigned int idx = (h + probe) & mask;
        long long k = table[idx].key;
        if (k == key) return &table[idx];
        if (k == HASH_EMPTY) {
            unsigned long long old = atomicCAS(
                (unsigned long long*)&table[idx].key, empty, ukey);
            if (old == empty) {
                // Slot claimed. dist already set by memset(0x7F):
                // dist bits = 0x7F7F7F7F = large positive float/int.
                // atomicMin will correctly replace it.
                return &table[idx];
            }
            if (old == ukey) return &table[idx];
            // Slot taken by different key, continue probing
        }
    }
    return NULL;  // table full
}

#pragma once
#include <cuda_fp16.h>

// Include dynamic_blocks.cuh for BlockEntry reuse (do NOT redefine BlockEntry)
// BlockEntry is defined in dynamic_blocks.cuh lines 39-43 as:
//   struct __align__(8) BlockEntry { unsigned short local_key; unsigned short _pad; float dist; };
// Also provides: get_block(), block_relax_dyn(), block_read_dist_dyn()
#include "dynamic_blocks.cuh"

// ============ V4 Control Buffer Indices ============
// These are SEPARATE from V2's indices in common.cuh to avoid conflicts
#define CTL_V4_DONE              0
#define CTL_V4_TOWER_COUNT       1
#define CTL_V4_BLOCK_OVERFLOW    2
#define CTL_V4_POOL_OVERFLOW     3
#define CTL_V4_STALE_ASSIGNMENTS 4
#define CTL_V4_HEAD_LOGICAL      5
#define CTL_V4_EMPTY_SWEEPS      6
#define CTL_V4_ASSIGNMENTS_TOTAL 7
#define CTL_V4_SIZE              16  // total control buffer size

// Forbidden cell value (impassable)
#define FORBIDDEN 65535

// ============ WorkItem (16 bytes) ============
// Stored in bucket queue FIFO pools
struct __align__(16) WorkItem {
    long long state;        // packed (cell, dir, span_bin, height_class)
    float     dist;         // distance at time of insertion
    __half    span_dist;    // exact span in meters
    unsigned short _pad;    // alignment padding
};

// ============ TowerRecordV4 (24 bytes) ============
// Appended atomically during search for path reconstruction
struct __align__(8) TowerRecordV4 {
    long long state;        // state AFTER tower (new dir, span=0)
    long long pred_state;   // state BEFORE tower (incoming dir, span=X)
    __half    span_dist;    // span length at tower placement
    __half    tower_height; // selected tower height (meters)
    float     tower_cost;   // total tower cost (terrain + angle + height)
};

// ============ AssignmentFlag (16 bytes) ============
// Per-WTB, in global memory. MTB writes, WTB polls (volatile reads required)
struct __align__(16) AssignmentFlag {
    int status;      // 0=idle, 1=assigned, 2=working
    int bucket_id;   // which physical bucket the work came from
    int offset;      // start index in bucket pool
    int count;       // number of items to process
};

// ============ State Packing/Unpacking ============

__device__ __forceinline__ long long pack_state(
    int cell, int dir, int span_bin, int hc,
    int n_dirs, int n_span_bins, int n_heights
) {
    return (long long)cell * (n_dirs * n_span_bins * n_heights)
         + (long long)dir * (n_span_bins * n_heights)
         + (long long)span_bin * n_heights
         + (long long)hc;
}

__device__ __forceinline__ void unpack_state(
    long long state, int n_dirs, int n_span_bins, int n_heights,
    int* cell, int* dir, int* span_bin, int* hc
) {
    int spc = n_dirs * n_span_bins * n_heights;
    *cell = (int)(state / spc);
    long long rem = state % spc;
    int sbh = n_span_bins * n_heights;
    *dir = (int)(rem / sbh);
    rem = rem % sbh;
    *span_bin = (int)(rem / n_heights);
    *hc = (int)(rem % n_heights);
}

__device__ __forceinline__ unsigned short pack_local_key(
    int dir, int span_bin, int hc, int n_span_bins, int n_heights
) {
    return (unsigned short)(dir * (n_span_bins * n_heights) + span_bin * n_heights + hc);
}

#pragma once
#include "adds_common.cuh"

// ============ Bucket Queue Constants ============
// N_BUCKETS: 32 circular FIFO buckets for delta-stepping
// SEGMENT_SIZE: WCC granularity -- MTB reads in segments of this size
// ITEMS_PER_BUCKET: derived from total pool size / N_BUCKETS
// MAX_SEGMENTS_PER_BUCKET: derived from ITEMS_PER_BUCKET / SEGMENT_SIZE
//
// These must be defined at compile time (injected by Python driver):
//   N_BUCKETS, SEGMENT_SIZE, ITEMS_PER_BUCKET, MAX_SEGMENTS_PER_BUCKET

#ifndef N_BUCKETS
#define N_BUCKETS 32
#endif

#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 32
#endif

// ITEMS_PER_BUCKET and MAX_SEGMENTS_PER_BUCKET are injected by Python.
// Provide safe defaults for standalone compilation / testing.
#ifndef ITEMS_PER_BUCKET
#define ITEMS_PER_BUCKET 65536
#endif

#ifndef MAX_SEGMENTS_PER_BUCKET
#define MAX_SEGMENTS_PER_BUCKET (ITEMS_PER_BUCKET / SEGMENT_SIZE)
#endif

// ============ Global Memory Layout (allocated by Python driver) ============
//
// bucket_resv_ptr    [N_BUCKETS]                    -- atomic write position per bucket (WTBs atomicAdd)
// bucket_read_ptr    [N_BUCKETS]                    -- MTB read position per bucket
// bucket_generation  [N_BUCKETS]                    -- reuse counter for stale enqueue detection
// bucket_wcc         [N_BUCKETS * MAX_SEGMENTS_PER_BUCKET] -- write completed counters (per segment)
// bucket_pool        [N_BUCKETS * ITEMS_PER_BUCKET] -- flat work item storage
//
// Pool addressing: bucket b, slot s -> bucket_pool[b * ITEMS_PER_BUCKET + s]

// ============ enqueue_to_bucket ============
// Called by WTBs (multiple concurrent writers) to add a work item.
//
// Returns 0 on success, -1 on overflow.
__device__ __forceinline__ int enqueue_to_bucket(
    WorkItem* __restrict__ bucket_pool,
    int*      __restrict__ bucket_resv_ptr,
    int*      __restrict__ bucket_generation,
    int*      __restrict__ bucket_wcc,
    volatile int* __restrict__ control,
    WorkItem  item,
    float     current_delta,
    int       head_logical
) {
    // 1. Compute logical bucket index from item distance
    int logical = (int)(item.dist / current_delta);

    // Stale item: already in a processed bucket range
    if (logical < head_logical) return 0;

    // Clamp to tail bucket if beyond the window
    if (logical >= head_logical + N_BUCKETS) {
        logical = head_logical + N_BUCKETS - 1;
    }

    // 2. Physical bucket = logical modulo N_BUCKETS (circular)
    int physical = logical % N_BUCKETS;

    // 3. Reserve a slot atomically
    int slot = atomicAdd(&bucket_resv_ptr[physical], 1);

    // 4. Check overflow
    if (slot >= ITEMS_PER_BUCKET) {
        // Undo reservation (best-effort, won't harm correctness
        // since MTB checks WCC, not resv_ptr, for completion)
        atomicAdd(&bucket_resv_ptr[physical], -1);
        atomicAdd((int*)&control[CTL_V4_POOL_OVERFLOW], 1);
        return -1;
    }

    // 5. Write item to pool
    int pool_idx = physical * ITEMS_PER_BUCKET + slot;
    bucket_pool[pool_idx] = item;

    // 6. Ensure the write is visible to MTB before signaling completion
    __threadfence();

    // 7. Increment WCC for this segment
    int seg_idx = slot / SEGMENT_SIZE;
    atomicAdd(&bucket_wcc[physical * MAX_SEGMENTS_PER_BUCKET + seg_idx], 1);

    return 0;
}


// ============ read_bucket_segment ============
// Called by MTB (single reader) to read a completed segment.
//
// Returns SEGMENT_SIZE if the segment is complete and items were copied,
// or 0 if the segment is not yet complete.
__device__ __forceinline__ int read_bucket_segment(
    const WorkItem* __restrict__ bucket_pool,
    const int*      __restrict__ bucket_wcc,
    int physical_bucket,
    int segment_idx,
    WorkItem* out_items
) {
    // Check if all SEGMENT_SIZE items in this segment have been written
    int wcc_val = bucket_wcc[physical_bucket * MAX_SEGMENTS_PER_BUCKET + segment_idx];
    if (wcc_val < SEGMENT_SIZE) return 0;

    // Copy items from pool to output buffer
    int base = physical_bucket * ITEMS_PER_BUCKET + segment_idx * SEGMENT_SIZE;
    for (int i = 0; i < SEGMENT_SIZE; i++) {
        out_items[i] = bucket_pool[base + i];
    }
    return SEGMENT_SIZE;
}


// ============ read_bucket_partial ============
// Called by MTB for the last (incomplete) segment of a drained bucket.
// Reads items from read_ptr_val up to the current reservation pointer,
// but only within the current segment boundary.
//
// Returns the number of items read (0 to SEGMENT_SIZE-1).
__device__ __forceinline__ int read_bucket_partial(
    const WorkItem* __restrict__  bucket_pool,
    const volatile int* __restrict__ bucket_resv_ptr,
    const int*      __restrict__  bucket_wcc,
    int physical_bucket,
    int read_ptr_val,
    WorkItem* out_items
) {
    // Read the current reservation pointer (volatile for visibility)
    int resv = bucket_resv_ptr[physical_bucket];
    if (resv <= read_ptr_val) return 0;

    // We read up to the end of the current segment or resv, whichever is smaller
    int seg_start = (read_ptr_val / SEGMENT_SIZE) * SEGMENT_SIZE;
    int seg_end = seg_start + SEGMENT_SIZE;
    int end = resv < seg_end ? resv : seg_end;
    int count = end - read_ptr_val;
    if (count <= 0) return 0;

    // For partial reads, we need to ensure items are actually written.
    // Check individual WCC: if the segment's WCC >= (end - seg_start), all
    // items up to 'end' have been committed. Otherwise, we check against
    // what has been committed so far.
    int seg_idx = read_ptr_val / SEGMENT_SIZE;
    int wcc_val = bucket_wcc[physical_bucket * MAX_SEGMENTS_PER_BUCKET + seg_idx];
    int committed_end = seg_start + wcc_val;
    if (committed_end < end) {
        end = committed_end;
        count = end - read_ptr_val;
        if (count <= 0) return 0;
    }

    int base = physical_bucket * ITEMS_PER_BUCKET + read_ptr_val;
    for (int i = 0; i < count; i++) {
        out_items[i] = bucket_pool[base + i];
    }
    return count;
}


// ============ reset_bucket ============
// Called by MTB when advancing head past this bucket.
// Resets all counters so the physical bucket can be reused for a new logical bucket.
__device__ __forceinline__ void reset_bucket(
    int* __restrict__ bucket_resv_ptr,
    int* __restrict__ bucket_read_ptr,
    int* __restrict__ bucket_generation,
    int* __restrict__ bucket_wcc,
    int physical_bucket
) {
    bucket_resv_ptr[physical_bucket] = 0;
    bucket_read_ptr[physical_bucket] = 0;
    bucket_generation[physical_bucket]++;

    // Zero WCC for all segments in this bucket
    for (int s = 0; s < MAX_SEGMENTS_PER_BUCKET; s++) {
        bucket_wcc[physical_bucket * MAX_SEGMENTS_PER_BUCKET + s] = 0;
    }
}


// ============ get_bucket_count ============
// Returns the number of items currently enqueued in a bucket
// (reservation pointer minus read pointer).
__device__ __forceinline__ int get_bucket_count(
    const volatile int* __restrict__ bucket_resv_ptr,
    const int* __restrict__ bucket_read_ptr,
    int physical_bucket
) {
    int resv = bucket_resv_ptr[physical_bucket];
    int read = bucket_read_ptr[physical_bucket];
    return resv - read;
}

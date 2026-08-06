#pragma once
#include "adds_bucket_queue.cuh"

// ============ MTB (Manager Thread Block) Main Loop ============
//
// Called by block 0 of the persistent ADDS kernel. All 256 threads
// cooperate on scanning assignment flags and reading bucket items.
//
// Key responsibilities:
//   1. Find non-empty bucket (scan resv_ptr > read_ptr from head forward)
//   2. Count idle WTBs (parallel AF status scan)
//   3. Read completed segments from head bucket
//   4. Assign work chunks to idle WTBs
//   5. Advance head when bucket fully consumed
//   6. Early termination (target margin, double empty sweep, max assignments)
//
// Design decisions (V1 simplifications):
//   - No dynamic delta adjustment (initial_delta is fixed). Delta changes
//     require "safe transition points" (all WTBs idle, no lower-bucket work)
//     which adds substantial complexity. Can be added in V2.
//   - Thread 0 does most coordination; threads 1..N-1 assist with AF scanning.
//   - Only reads from current head bucket (no multi-bucket scatter).
//   - Items are assigned by pointing WTBs directly to bucket pool offsets
//     that the MTB has read past (read_ptr already advanced, so the memory
//     region is stable and won't be overwritten until bucket reset).

// Maximum WTBs we support (= max gridDim.x - 1).
// Sized generously; actual n_wtbs is passed at runtime.
#ifndef MAX_WTBS
#define MAX_WTBS 256
#endif

// Maximum items the MTB can stage in shared memory per read pass.
// Must be a multiple of SEGMENT_SIZE for clean segment reads.
// 256 items * 16 bytes = 4 KB shared memory, well within limits.
#ifndef MTB_STAGING_SIZE
#define MTB_STAGING_SIZE 256
#endif

__device__ void mtb_main_loop(
    // Bucket queue metadata
    volatile int* bucket_resv_ptr,  // volatile: WTBs write concurrently
    int* bucket_read_ptr,
    int* bucket_generation,
    volatile int* bucket_wcc,       // volatile: WTBs increment
    WorkItem* bucket_pool,
    // Assignment flags (one per WTB)
    volatile AssignmentFlag* assignment_flags,
    int n_wtbs,  // number of WTBs (gridDim.x - 1)
    // Control buffer
    volatile int* control,
    // Best target distance (stores __float_as_int of best dist)
    volatile int* best_target_dist,
    // Parameters
    float initial_delta,
    float margin,
    int max_assignments
) {
    // ================================================================
    // Shared memory for MTB coordination
    // ================================================================
    __shared__ int s_head_logical;       // current logical head bucket
    __shared__ int s_empty_sweeps;       // consecutive empty scans
    __shared__ int s_done;               // termination flag
    __shared__ int s_total_assignments;  // total items assigned so far
    __shared__ int s_n_idle;             // count of idle WTBs this iteration
    __shared__ int s_idle_wtbs[MAX_WTBS]; // indices of idle WTBs
    __shared__ int s_n_items;            // items read this pass
    __shared__ int s_read_base;          // pool offset where staged items start

    // Initialize (thread 0 only, then sync)
    if (threadIdx.x == 0) {
        s_head_logical = 0;
        s_empty_sweeps = 0;
        s_done = 0;
        s_total_assignments = 0;
        s_n_idle = 0;
        s_n_items = 0;
        s_read_base = 0;
    }
    __syncthreads();

    // ================================================================
    // Main loop
    // ================================================================
    while (!s_done) {

        // ============================================================
        // STEP 1: Find non-empty bucket starting from head
        // ============================================================
        // Thread 0 scans forward from head_logical. If the head bucket
        // is empty, advance past empty buckets (resetting them) until a
        // non-empty one is found or we've scanned all N_BUCKETS.
        if (threadIdx.x == 0) {
            int head_phys = s_head_logical % N_BUCKETS;
            int count = bucket_resv_ptr[head_phys]
                      - bucket_read_ptr[head_phys];

            if (count <= 0) {
                // Head is empty — scan forward
                bool found = false;
                for (int offset = 1; offset < N_BUCKETS; offset++) {
                    int phys = (s_head_logical + offset) % N_BUCKETS;
                    int bc = bucket_resv_ptr[phys]
                           - bucket_read_ptr[phys];
                    if (bc > 0) {
                        // Reset all skipped buckets so they can be reused
                        for (int skip = 0; skip < offset; skip++) {
                            int skip_phys =
                                (s_head_logical + skip) % N_BUCKETS;
                            reset_bucket((int*)bucket_resv_ptr,
                                         bucket_read_ptr,
                                         bucket_generation,
                                         (int*)bucket_wcc,
                                         skip_phys);
                        }
                        s_head_logical += offset;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    s_empty_sweeps++;
                    if (s_empty_sweeps >= 2) {
                        s_done = 1;
                    }
                }
            } else {
                // Head has items — reset sweep counter
                s_empty_sweeps = 0;
            }

            // Publish head_logical to control buffer for WTB enqueue
            control[CTL_V4_HEAD_LOGICAL] = s_head_logical;
        }
        __syncthreads();
        if (s_done) break;

        // ============================================================
        // STEP 2: Count idle WTBs (parallel AF scan)
        // ============================================================
        // Threads 0..n_wtbs-1 each check one AF's status.
        if (threadIdx.x == 0) s_n_idle = 0;
        __syncthreads();

        if ((int)threadIdx.x < n_wtbs) {
            int af_status =
                ((volatile int*)&assignment_flags[threadIdx.x].status)[0];
            if (af_status == 0) {
                int slot = atomicAdd(&s_n_idle, 1);
                s_idle_wtbs[slot] = threadIdx.x;
            }
        }
        __syncthreads();

        if (s_n_idle == 0) {
            // All WTBs busy — retry (implicit busy-wait)
            continue;
        }

        // ============================================================
        // STEP 3: Read items from head bucket
        // ============================================================
        // Thread 0 reads completed segments (and possibly a partial
        // trailing segment) from the head bucket. Rather than copying
        // items to shared memory, we simply advance the read pointer
        // and record the range [old_read_ptr, new_read_ptr). The WTBs
        // can then read directly from the bucket pool at those offsets
        // (the memory is stable because we've already advanced read_ptr
        // past it — it won't be overwritten until the bucket is reset).
        if (threadIdx.x == 0) {
            s_n_items = 0;
            int head_phys = s_head_logical % N_BUCKETS;
            int rp = bucket_read_ptr[head_phys];
            int wp = bucket_resv_ptr[head_phys];

            // Remember where we started reading
            s_read_base = rp;

            // Maximum items to read: enough for all idle WTBs,
            // but capped at staging buffer size
            int want = s_n_idle * SEGMENT_SIZE;
            if (want > MTB_STAGING_SIZE) want = MTB_STAGING_SIZE;

            // Read complete segments first
            while (s_n_items + SEGMENT_SIZE <= want && rp + SEGMENT_SIZE <= wp) {
                int seg_idx = rp / SEGMENT_SIZE;
                int wcc_val = bucket_wcc[
                    head_phys * MAX_SEGMENTS_PER_BUCKET + seg_idx];
                if (wcc_val >= SEGMENT_SIZE) {
                    // Full segment committed — consume it
                    s_n_items += SEGMENT_SIZE;
                    rp += SEGMENT_SIZE;
                } else {
                    // Segment not fully committed yet — stop
                    break;
                }
            }

            // Try partial read for the trailing segment
            if (s_n_items < want && rp < wp) {
                int seg_idx = rp / SEGMENT_SIZE;
                int seg_start = seg_idx * SEGMENT_SIZE;
                int wcc_val = bucket_wcc[
                    head_phys * MAX_SEGMENTS_PER_BUCKET + seg_idx];
                int committed_end = seg_start + wcc_val;
                // Only read what's committed past our current read pointer
                int avail = committed_end - rp;
                if (avail > 0) {
                    int take = avail;
                    if (s_n_items + take > want) take = want - s_n_items;
                    s_n_items += take;
                    rp += take;
                }
            }

            // Advance the read pointer (makes the range stable for WTBs)
            bucket_read_ptr[head_phys] = rp;
        }
        __syncthreads();

        if (s_n_items == 0) {
            // No committed items yet — but bucket isn't empty (WTBs still
            // writing). This does NOT count as an empty sweep.
            continue;
        }

        // ============================================================
        // STEP 4: Assign work chunks to idle WTBs
        // ============================================================
        // Divide s_n_items evenly across s_n_idle WTBs. Each WTB gets
        // a contiguous range [offset, offset + count) in the bucket pool.
        if (threadIdx.x == 0) {
            int head_phys = s_head_logical % N_BUCKETS;
            int items_per_wtb = (s_n_items + s_n_idle - 1) / s_n_idle;
            int assigned = 0;

            for (int w = 0; w < s_n_idle && assigned < s_n_items; w++) {
                int wtb_idx = s_idle_wtbs[w];
                int chunk = items_per_wtb;
                if (assigned + chunk > s_n_items) {
                    chunk = s_n_items - assigned;
                }

                // Write AF fields: point WTB to the bucket pool range
                // it should process. The offset is relative to the
                // bucket's pool start (physical_bucket * ITEMS_PER_BUCKET).
                assignment_flags[wtb_idx].bucket_id = head_phys;
                assignment_flags[wtb_idx].offset = s_read_base + assigned;
                assignment_flags[wtb_idx].count = chunk;
                __threadfence();  // ensure fields visible before status
                assignment_flags[wtb_idx].status = 1;  // signal WTB

                assigned += chunk;
            }
            s_total_assignments += assigned;
        }
        __syncthreads();

        // ============================================================
        // STEP 5: Early termination checks
        // ============================================================
        if (threadIdx.x == 0) {
            // 5a. Target margin termination
            // best_target_dist stores __float_as_int(best_dist_float),
            // which preserves order for positive floats.
            int best_bits = *best_target_dist;
            float best_dist = __int_as_float(best_bits);
            if (best_dist < 1e30f) {
                // The head bucket covers distances
                // [head_logical * delta, (head_logical + 1) * delta).
                // If the floor of the current head bucket exceeds the
                // margin-scaled best distance, no future bucket can
                // improve the result.
                float head_floor =
                    (float)s_head_logical * initial_delta;
                if (head_floor > best_dist * margin) {
                    s_done = 1;
                }
            }

            // 5b. Max assignments safeguard
            if (s_total_assignments >= max_assignments) {
                s_done = 1;
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 6: Advance head if bucket is fully consumed
        // ============================================================
        if (threadIdx.x == 0) {
            int head_phys = s_head_logical % N_BUCKETS;
            int rp = bucket_read_ptr[head_phys];
            int wp = bucket_resv_ptr[head_phys];
            if (rp >= wp && wp > 0) {
                // Bucket fully consumed — reset and advance
                reset_bucket((int*)bucket_resv_ptr,
                             bucket_read_ptr,
                             bucket_generation,
                             (int*)bucket_wcc,
                             head_phys);
                s_head_logical++;
                control[CTL_V4_HEAD_LOGICAL] = s_head_logical;
            }
        }
        __syncthreads();

        // (No dynamic delta in V1 — initial_delta stays fixed)
    }

    // ================================================================
    // Termination: signal all WTBs to exit
    // ================================================================
    if (threadIdx.x == 0) {
        // Write diagnostic counters to control buffer
        control[CTL_V4_EMPTY_SWEEPS] = s_empty_sweeps;
        control[CTL_V4_ASSIGNMENTS_TOTAL] = s_total_assignments;

        // Set the global done flag (WTBs poll this)
        control[CTL_V4_DONE] = 1;
        __threadfence();
    }
}

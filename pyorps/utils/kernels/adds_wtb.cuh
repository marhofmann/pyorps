#pragma once
#include "adds_tower.cuh"

// ============ WTB (Worker Thread Block) Main Loop ============
//
// Called by blocks 1..N in the persistent ADDS kernel. Each WTB:
//   1. Polls its AssignmentFlag until status == 1 (assigned) or done flag set
//   2. Reads the assignment (bucket_id, offset, count) from AF
//   3. Processes items in parallel (strided across blockDim.x threads)
//   4. For each item: stale check, relax non-tower edges, place towers
//   5. Signals completion: __syncthreads, __threadfence, AF.status = 0
//
// Shared memory layout (loaded by caller before entering this function):
//   s_steps[n_dirs * 2]               -- (dr, dc) per direction
//   s_cost_factors[n_dirs]            -- distance normalization per direction
//   s_step_distances[n_dirs]          -- physical distance in meters per direction
//   s_angle_valid[n_dirs * n_dirs]    -- bool: is this turn valid?
//   s_angle_cost[n_dirs * n_dirs]     -- soft turn penalty
//   s_tower_angle_cost[n_dirs * n_dirs] -- tower angle cost per direction pair
//   s_height_premiums[n_heights]      -- cost premium per height class
//   s_intermediates[n_dirs * max_inter_cols * 2] -- intermediate cell offsets
//   s_n_intermediates[n_dirs]         -- count of intermediates per direction
//
// NOTE: Tower terrain costs (65536 entries = 256 KB) are too large for shared
// memory (max 48-100 KB). They remain in global memory (g_tower_terrain).

#ifndef MAX_INTER
#define MAX_INTER 4
#endif

__device__ void wtb_main_loop(
    int block_id,  // blockIdx.x (1..N)
    // Assignment flag for this WTB (volatile for polling)
    volatile AssignmentFlag* my_af,
    volatile int* done_flag,  // &control[CTL_V4_DONE]
    // Raster
    const unsigned short* __restrict__ raster,
    int rows, int cols,
    // Block-sparse storage
    BlockEntry* pool, __half* span_pool,
    int* cell_to_block, int* block_to_cell, int* n_allocated,
    int max_sparse_blocks,
    // Bucket queue
    WorkItem* bucket_pool, int* bucket_resv_ptr, int* bucket_generation,
    int* bucket_wcc, volatile int* control,
    float current_delta, int head_logical,
    int max_pool_blocks,
    // Tower records
    TowerRecordV4* tower_records, int max_tower_records,
    // Best target distance
    int* best_target_dist, int target_cell,
    // Profile LUTs (in shared memory)
    const signed char* s_steps,
    const float* s_cost_factors,
    const float* s_step_distances,
    const unsigned char* s_angle_valid,
    const float* s_angle_cost,
    const float* g_tower_terrain,    // GLOBAL (65536 entries, too large for smem)
    const float* s_tower_angle_cost,
    const float* s_height_premiums,
    const short* s_intermediates,
    const int* s_n_intermediates,
    // Parameters
    int n_dirs, int n_span_bins, int n_heights,
    float min_span, float max_span, float span_bin_size,
    int max_inter_cols
) {
    // Shared memory for AF communication within the block
    __shared__ int sh_done;
    __shared__ int sh_bucket_id;
    __shared__ int sh_offset;
    __shared__ int sh_count;

    if (threadIdx.x == 0) sh_done = 0;
    __syncthreads();

    // ================================================================
    // Main WTB loop: poll AF -> process -> signal idle
    // ================================================================
    while (true) {
        // ---- Phase 1: Poll assignment flag ----
        if (threadIdx.x == 0) {
            // Volatile reads to avoid compiler caching the status
            while (((volatile int*)&my_af->status)[0] != 1) {
                if (((volatile int*)done_flag)[0] != 0) {
                    sh_done = 1;
                    break;
                }
            }
            if (!sh_done) {
                // Load assignment into shared memory
                sh_bucket_id = ((volatile int*)&my_af->bucket_id)[0];
                sh_offset    = ((volatile int*)&my_af->offset)[0];
                sh_count     = ((volatile int*)&my_af->count)[0];
                // Mark AF as working
                ((volatile int*)&my_af->status)[0] = 2;
            }
        }
        __syncthreads();

        if (sh_done) break;

        int assignment_bucket = sh_bucket_id;
        int assignment_offset = sh_offset;
        int assignment_count  = sh_count;

        // ---- Phase 2: Process items (strided across threads) ----
        int stale_count = 0;

        for (int i = threadIdx.x; i < assignment_count; i += blockDim.x) {
            // Read WorkItem from bucket pool
            int pool_idx = assignment_bucket * ITEMS_PER_BUCKET
                         + assignment_offset + i;
            WorkItem item = bucket_pool[pool_idx];

            // Unpack state
            int cell, dir_in, span_bin, hc;
            unpack_state(item.state, n_dirs, n_span_bins, n_heights,
                        &cell, &dir_in, &span_bin, &hc);

            // Stale check: compare item.dist against best known distance
            int bi = cell_to_block[cell];
            if (bi >= 0) {
                unsigned short lk = pack_local_key(dir_in, span_bin, hc,
                                                   n_span_bins, n_heights);
                float best = block_read_dist_dyn(pool, bi, lk);
                // Allow tiny tolerance for float comparison
                if (item.dist > best * 1.0001f) {
                    stale_count++;
                    continue;
                }
            }

            // Decode cell position
            int r = cell / cols;
            int c = cell % cols;
            unsigned short raster_val = raster[cell];
            if (raster_val == FORBIDDEN) continue;

            float span_m = __half2float(item.span_dist);

            // ---- Phase 2a: Relax non-tower edges (same direction only) ----
            // Direction changes require tower placement, handled in Phase 2b.
            for (int d = 0; d < n_dirs; d++) {
                // Only relax same-direction edges without tower
                if (d != dir_in) continue;

                // Angle valid check (should always pass for d == dir_in, but
                // keep for safety in case angle_valid encodes other constraints)
                if (!s_angle_valid[dir_in * n_dirs + d]) continue;

                int dr = (int)s_steps[d * 2];
                int dc = (int)s_steps[d * 2 + 1];
                int nr = r + dr;
                int nc = c + dc;
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

                int ncell = nr * cols + nc;
                unsigned short nval = raster[ncell];
                if (nval == FORBIDDEN) continue;

                // Check intermediates
                int n_inter = s_n_intermediates[d];
                bool inter_ok = true;
                float ic = 0.0f;
                for (int k = 0; k < n_inter; k++) {
                    int idr = (int)s_intermediates[
                        (d * max_inter_cols + k) * 2];
                    int idc = (int)s_intermediates[
                        (d * max_inter_cols + k) * 2 + 1];
                    int ir = r + idr, icc = c + idc;
                    if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) {
                        inter_ok = false; break;
                    }
                    unsigned short iv = raster[ir * cols + icc];
                    if (iv == FORBIDDEN) { inter_ok = false; break; }
                    ic += (float)iv;
                }
                if (!inter_ok) continue;

                // Edge cost: average terrain * cost_factor + angle penalty
                float edge_cost =
                    ((float)raster_val + (float)nval + ic) * s_cost_factors[d]
                    + s_angle_cost[dir_in * n_dirs + d];

                // Same direction: continue span without tower
                float new_span = span_m + s_step_distances[d];
                if (new_span > max_span) continue;

                int new_span_bin = (int)(new_span / span_bin_size);
                if (new_span_bin >= n_span_bins) {
                    new_span_bin = n_span_bins - 1;
                }

                float new_dist = item.dist + edge_cost;

                long long new_state = pack_state(ncell, d, new_span_bin, hc,
                                                 n_dirs, n_span_bins, n_heights);

                // Get/allocate block for neighbor cell
                int nbi = cell_to_block[ncell];
                if (nbi == -1) {
                    nbi = get_block(ncell, cell_to_block, block_to_cell,
                                   n_allocated, max_sparse_blocks);
                }
                if (nbi < 0) {
                    atomicAdd((int*)&control[CTL_V4_BLOCK_OVERFLOW], 1);
                    continue;
                }

                unsigned short nlk = pack_local_key(d, new_span_bin, hc,
                                                    n_span_bins, n_heights);
                int improved = block_relax_dyn(pool, span_pool, nbi, nlk,
                                               new_dist, new_span);

                if (improved) {
                    WorkItem wi;
                    wi.state = new_state;
                    wi.dist = new_dist;
                    wi.span_dist = __float2half(new_span);
                    wi._pad = 0;
                    enqueue_to_bucket(bucket_pool, bucket_resv_ptr,
                                     bucket_generation, bucket_wcc,
                                     control, wi,
                                     current_delta, head_logical);

                    // Update best target distance
                    if (ncell == target_cell) {
                        atomicMin(best_target_dist,
                                  __float_as_int(new_dist));
                    }
                }
            }

            // ---- Phase 2b: Tower placement (direction changes) ----
            // Only place towers if span >= min_span
            if (span_m >= min_span) {
                place_towers_uniform(
                    cell, dir_in, span_bin, hc,
                    item.dist, span_m, raster_val, r, c,
                    raster, rows, cols,
                    pool, span_pool,
                    cell_to_block, block_to_cell, n_allocated,
                    max_sparse_blocks,
                    bucket_pool, bucket_resv_ptr, bucket_generation,
                    bucket_wcc, control,
                    current_delta, head_logical,
                    tower_records, max_tower_records,
                    best_target_dist, target_cell,
                    s_steps, s_cost_factors, s_step_distances,
                    s_angle_valid, s_angle_cost,
                    g_tower_terrain, s_tower_angle_cost, s_height_premiums,
                    s_intermediates, s_n_intermediates,
                    n_dirs, n_span_bins, n_heights,
                    min_span, max_span, span_bin_size,
                    max_inter_cols
                );
            }
        }

        // ---- Phase 3: Signal completion ----
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();  // ensure all writes are globally visible

            // Track stale assignment ratio for diagnostics
            if (stale_count > assignment_count * 9 / 10) {
                atomicAdd((int*)&control[CTL_V4_STALE_ASSIGNMENTS], 1);
            }

            // Total assignments processed
            atomicAdd((int*)&control[CTL_V4_ASSIGNMENTS_TOTAL], 1);

            // Mark AF as idle (ready for next assignment)
            ((volatile int*)&my_af->status)[0] = 0;
        }
        __syncthreads();
    }
}

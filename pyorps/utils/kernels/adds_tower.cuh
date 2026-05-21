#pragma once
#include "adds_bucket_queue.cuh"

// ============ Tower Placement (Uniform Mode, Per-Thread) ============
//
// Place towers at the current cell for all valid outgoing directions.
// Called when span >= min_span.
//
// For each valid dir_out:
//   1. Check angle_valid[dir_in * n_dirs + dir_out]
//   2. Compute terrain edge cost to neighbor cell in dir_out direction
//   3. Tower cost = tower_terrain[raster_val] + tower_angle[dir_in * n_dirs + dir_out]
//                 + height_premiums[hc]
//   4. If tower_terrain >= FORBIDDEN cost: skip (impassable cell for tower)
//   5. new_dist = dist + edge_cost + tower_cost
//   6. new_state = pack_state(ncell, dir_out, reset_span_bin, hc)  // span resets
//   7. Relax via block_relax_dyn
//   8. If improved: enqueue new WorkItem to bucket queue
//   9. Record TowerRecordV4 (atomic append)
//
// V1: No DEM gradient multiplier, no clearance check.
//     These can be added later with #if HAS_CLEARANCE / HAS_DEM guards.

#ifndef MAX_INTER
#define MAX_INTER 4
#endif

__device__ __forceinline__ void place_towers_uniform(
    // Current state info
    int cell, int dir_in, int span_bin, int hc,
    float dist_val, float span_m,
    unsigned short raster_val,
    int row, int col,
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
    // Tower records
    TowerRecordV4* tower_records, int max_tower_records,
    // Best target distance
    int* best_target_dist, int target_cell,
    // Profile LUTs
    const signed char* s_steps,
    const float* s_cost_factors,
    const float* s_step_distances,
    const unsigned char* s_angle_valid,
    const float* s_angle_cost,
    const float* g_tower_terrain,    // GLOBAL memory (65536 entries)
    const float* s_tower_angle_cost,
    const float* s_height_premiums,
    const short* s_intermediates,
    const int* s_n_intermediates,
    // Parameters
    int n_dirs, int n_span_bins, int n_heights,
    float min_span, float max_span, float span_bin_size,
    int max_inter_cols
) {
    // Tower terrain cost at current cell
    float terrain_base = g_tower_terrain[raster_val];
    // If this cell is impassable for tower placement, skip entirely
    if (terrain_base >= 1e30f) return;

    long long pred_state = pack_state(cell, dir_in, span_bin, hc,
                                      n_dirs, n_span_bins, n_heights);

    for (int d_out = 0; d_out < n_dirs; d_out++) {
        // 1. Check if this turn is valid
        if (!s_angle_valid[dir_in * n_dirs + d_out]) continue;

        // 2. Compute neighbor cell in dir_out direction
        int dr = (int)s_steps[d_out * 2];
        int dc = (int)s_steps[d_out * 2 + 1];
        int nr = row + dr;
        int nc = col + dc;
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

        int ncell = nr * cols + nc;
        unsigned short nval = raster[ncell];
        if (nval == FORBIDDEN) continue;

        // 3. Check intermediates for the outgoing direction
        int n_inter = s_n_intermediates[d_out];
        bool inter_ok = true;
        float ic = 0.0f;
        for (int k = 0; k < n_inter; k++) {
            int idr = (int)s_intermediates[(d_out * max_inter_cols + k) * 2];
            int idc = (int)s_intermediates[(d_out * max_inter_cols + k) * 2 + 1];
            int ir = row + idr, icc = col + idc;
            if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) {
                inter_ok = false; break;
            }
            unsigned short iv = raster[ir * cols + icc];
            if (iv == FORBIDDEN) { inter_ok = false; break; }
            ic += (float)iv;
        }
        if (!inter_ok) continue;

        // 4. Edge terrain cost (raster average * cost_factor)
        float edge_terrain = ((float)raster_val + (float)nval + ic)
                           * s_cost_factors[d_out];

        // 5. Edge angle cost (soft penalty for direction change)
        float angle_penalty = s_angle_cost[dir_in * n_dirs + d_out];

        float edge_cost = edge_terrain + angle_penalty;

        // 6. Physical step distance in meters for span reset
        float step_dist_m = s_step_distances[d_out];

        // 7. Reset span bin (after tower, span starts at step distance)
        int reset_span_bin = (int)(step_dist_m / span_bin_size);
        if (reset_span_bin >= n_span_bins) continue;

        // 8. Tower cost: terrain base + angle + height premium
        float tower_angle = s_tower_angle_cost[dir_in * n_dirs + d_out];
        float tower_cost = terrain_base + tower_angle + s_height_premiums[hc];

        float total_cost = edge_cost + tower_cost;
        float new_dist = dist_val + total_cost;

        // 9. Pack new state: direction changes to d_out, span resets
        long long new_state = pack_state(ncell, d_out, reset_span_bin, hc,
                                         n_dirs, n_span_bins, n_heights);

        // 10. Get/allocate block for neighbor cell
        int nbi = cell_to_block[ncell];
        if (nbi == -1) {
            nbi = get_block(ncell, cell_to_block, block_to_cell,
                           n_allocated, max_sparse_blocks);
        }
        if (nbi < 0) {
            atomicAdd((int*)&control[CTL_V4_BLOCK_OVERFLOW], 1);
            continue;
        }

        // 11. Relax distance
        unsigned short nlk = pack_local_key(d_out, reset_span_bin, hc,
                                            n_span_bins, n_heights);
        int improved = block_relax_dyn(pool, span_pool, nbi, nlk,
                                       new_dist, step_dist_m);

        if (improved) {
            // 12. Enqueue to bucket queue
            WorkItem wi;
            wi.state = new_state;
            wi.dist = new_dist;
            wi.span_dist = __float2half(step_dist_m);
            wi._pad = 0;
            enqueue_to_bucket(bucket_pool, bucket_resv_ptr, bucket_generation,
                             bucket_wcc, control, wi,
                             current_delta, head_logical);

            // 13. Record tower placement for path reconstruction
            int tr_idx = atomicAdd((int*)&control[CTL_V4_TOWER_COUNT], 1);
            if (tr_idx < max_tower_records) {
                tower_records[tr_idx].state = new_state;
                tower_records[tr_idx].pred_state = pred_state;
                tower_records[tr_idx].span_dist = __float2half(span_m);
                tower_records[tr_idx].tower_height = __float2half(0.0f);
                tower_records[tr_idx].tower_cost = tower_cost;
            }

            // 14. Update best target distance
            if (ncell == target_cell) {
                atomicMin(best_target_dist, __float_as_int(new_dist));
            }
        }
    }
}

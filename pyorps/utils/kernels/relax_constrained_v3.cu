#include "dynamic_blocks.cuh"
#include "clearance.cuh"
// NOTE: Do NOT include state_access.cuh, block_sparse.cuh, hash_table.cuh,
// or grid_barrier.cuh. V3 is self-contained via dynamic_blocks.cuh.

extern "C" __global__
void relax_constrained_v3(
    long long* frontier, int frontier_count, int phase, int buf_size, int bucket,
    const unsigned short* __restrict__ raster, int rows, int cols, int max_cost,
    const signed char* __restrict__ steps, const float* __restrict__ cost_factors,
    const signed char* __restrict__ inter_lut, const int* __restrict__ n_inter,
    int n_steps, int max_inter_cols,
    const float* __restrict__ angle_cost_lut, const unsigned char* __restrict__ angle_valid_lut,
    const float* __restrict__ step_distances,
    const float* __restrict__ tower_terrain_lut, const float* __restrict__ tower_angle_lut,
    const float* __restrict__ height_premiums, const float* __restrict__ tower_heights, int n_heights,
    int n_span_bins, float span_bin_size, int min_span_bin,
    long long spc, long long total_states,
    int* cell_to_block, int* block_to_cell,
    BlockEntry* pool, __half* span_pool,
    int* n_allocated, int max_blocks,
    float delta,
    long long* output_queue, long long* pending_queue, long long* settled_queue,
    volatile int* control,
    TowerRecord* tower_records, int max_tower_records,
    const float* __restrict__ dem, const float* __restrict__ obstacle,
    float cell_size, float cond_weight, float cond_tension, float min_clearance,
    float max_gradient_pct, float gradient_scale,
    const int* __restrict__ area_offsets, const int* __restrict__ area_starts,
    const int* __restrict__ area_counts
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // ---- Shared memory layout ----
    // Steps, n_inter, cost_factors, step_distances,
    // angle_cost (n_steps*n_steps), angle_valid (n_steps*n_steps),
    // tower_angle (n_steps*n_steps),
    // height_premiums (n_heights), tower_heights_sm (n_heights), inter_lut
    extern __shared__ char smem[];

    int steps_bytes = n_steps * 2;
    int steps_padded = (steps_bytes + 3) & ~3;
    signed char* s_steps = (signed char*)smem;
    int*   s_n_inter      = (int*)(smem + steps_padded);
    float* s_cost_factors  = (float*)(s_n_inter + n_steps);
    float* s_step_dist     = s_cost_factors + n_steps;
    float* s_angle_cost    = s_step_dist + n_steps;
    int n2 = n_steps * n_steps;
    unsigned char* s_angle_valid = (unsigned char*)(s_angle_cost + n2);
    int av_padded = (n2 + 3) & ~3;
    float* s_tower_angle   = (float*)((unsigned char*)s_angle_valid + av_padded);
    float* s_height_premiums = s_tower_angle + n2;
    float* s_tower_heights = s_height_premiums + n_heights;
    signed char* s_inter_lut = (signed char*)(s_tower_heights + n_heights);

    // Load shared memory
    for (int i = threadIdx.x; i < steps_bytes; i += blockDim.x)
        s_steps[i] = steps[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x)
        s_n_inter[i] = n_inter[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x)
        s_cost_factors[i] = cost_factors[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x)
        s_step_dist[i] = step_distances[i];
    for (int i = threadIdx.x; i < n2; i += blockDim.x)
        s_angle_cost[i] = angle_cost_lut[i];
    for (int i = threadIdx.x; i < n2; i += blockDim.x)
        s_angle_valid[i] = angle_valid_lut[i];
    for (int i = threadIdx.x; i < n2; i += blockDim.x)
        s_tower_angle[i] = tower_angle_lut[i];
    for (int i = threadIdx.x; i < n_heights; i += blockDim.x)
        s_height_premiums[i] = height_premiums[i];
    for (int i = threadIdx.x; i < n_heights; i += blockDim.x)
        s_tower_heights[i] = tower_heights[i];
    int lut_size = n_steps * max_inter_cols * 2;
    for (int i = threadIdx.x; i < lut_size; i += blockDim.x)
        s_inter_lut[i] = inter_lut[i];
    __syncthreads();

    // Precompute constants
    long long sh = (long long)n_span_bins * n_heights;
    float blo = bucket * delta;
    float bhi = (bucket + 1) * delta;

    // Warp setup
    int lane = threadIdx.x & 31;
    int warp_id_global = gtid >> 5;
    int n_warps = stride >> 5;

    // Process frontier in warp-sized batches: each warp takes 32 items,
    // one per lane. Non-tower work is per-thread (full utilization).
    // After all lanes finish their neighbor loops, lanes cooperate
    // on tower placement via round-robin ballot protocol.
    for (int batch_start = warp_id_global * 32; batch_start < frontier_count;
         batch_start += n_warps * 32)
    {
        int my_idx = batch_start + lane;

        // ---- Per-thread variables for this lane's frontier item ----
        long long my_state = -1;
        float my_dist = 1e30f;
        int my_cell = -1, my_dir = -1, my_span_bin = -1, my_hc = -1;
        int my_row = -1, my_col_val = -1;
        unsigned short my_sv = 65535u;
        float my_span_m = 0.0f;
        int my_want_tower = 0;  // does this lane want tower placement?
        unsigned short my_local_key = BLOCK_EMPTY;

        if (my_idx < frontier_count) {
            my_state = frontier[my_idx];
            if (my_state < 0 || my_state >= total_states) {
                my_dist = 1e30f;  // corrupt queue entry
            } else {
                // Unpack state components
                long long cur_cell_ll = my_state / spc;
                my_cell = (int)cur_cell_ll;
                long long rem = my_state - cur_cell_ll * spc;
                my_dir = (int)(rem / sh);
                long long rem2 = rem - (long long)my_dir * sh;
                my_span_bin = (int)(rem2 / n_heights);
                my_hc = (int)(rem2 % n_heights);
                my_local_key = make_local_key(my_dir, my_span_bin, my_hc,
                                              n_span_bins, n_heights);

                // Read current distance from dynamic block-sparse storage
                int my_block_idx = cell_to_block[my_cell];
                my_dist = block_read_dist_dyn(pool, my_block_idx, my_local_key);

                if (my_dist < 1e30f) {
                    my_row = my_cell / cols;
                    my_col_val = my_cell - my_row * cols;
                    if (my_cell < 0 || my_cell >= rows * cols) {
                        my_dist = 1e30f;
                    } else {
                        my_sv = raster[my_cell];
                        if (my_sv == (unsigned short)max_cost) {
                            my_dist = 1e30f;
                        } else {
                            my_span_m = block_read_span_dyn(
                                span_pool, pool, my_block_idx, my_local_key);
                        }
                    }
                }
            }  // end state validation

            // Append to settled queue (light phase only — in heavy phase,
            // the frontier IS the settled queue, so appending would corrupt it)
            if (phase == 0 && my_dist < 1e30f) {
                int sp = atomicAdd((int*)&control[V3_CTL_SETTLED], 1);
                if (sp < buf_size) settled_queue[sp] = my_state;
            }
        }

        // ---- Phase A: Per-thread non-tower relaxations ----
        // Each thread independently processes ALL neighbors of its
        // own frontier item. Only same-direction (non-tower) edges.
        if (my_dist < 1e30f) {
            for (int d_out = 0; d_out < n_steps; d_out++) {
                if (s_angle_valid[my_dir * n_steps + d_out] == 0) continue;

                int dr = (int)s_steps[d_out * 2];
                int dc = (int)s_steps[d_out * 2 + 1];
                int nr = my_row + dr;
                int nc = my_col_val + dc;
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

                int nb_cell = nr * cols + nc;
                unsigned short dv = raster[nb_cell];
                if (dv == (unsigned short)max_cost) continue;

                float ic = 0.0f;
                bool ok = true;
                int ni = s_n_inter[d_out];
                for (int k = 0; k < ni; k++) {
                    int ir = my_row + (int)s_inter_lut[(d_out * max_inter_cols + k) * 2];
                    int icc = my_col_val + (int)s_inter_lut[(d_out * max_inter_cols + k) * 2 + 1];
                    if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) { ok = false; break; }
                    unsigned short iv = raster[ir * cols + icc];
                    if (iv == (unsigned short)max_cost) { ok = false; break; }
                    ic += (float)iv;
                }
                if (!ok) continue;

                float terrain_cost = ((float)my_sv + (float)dv + ic) * s_cost_factors[d_out];
                float step_dist_m = s_step_dist[d_out];

                if (dem != NULL) {
                    float elev_src = dem[my_cell];
                    float elev_dst = dem[nb_cell];
                    float slope = fabsf(elev_dst - elev_src) / step_dist_m * 100.0f;
                    if (slope > max_gradient_pct) continue;
                    float grad_mult = expf(gradient_scale * slope / 100.0f);
                    terrain_cost *= grad_mult;
                }

                float angle_penalty = s_angle_cost[my_dir * n_steps + d_out];
                float edge_cost = terrain_cost + angle_penalty;
                float new_span_m = my_span_m + step_dist_m;

                // Non-tower: continue span in same direction
                if (d_out == my_dir) {
                    int new_span_bin = (int)(new_span_m / span_bin_size);
                    if (new_span_bin >= n_span_bins) continue;

                    // Phase filter: light keeps edge_cost <= delta,
                    //               heavy keeps edge_cost > delta
                    if (phase == 0 && edge_cost > delta) continue;
                    if (phase == 1 && edge_cost <= delta) continue;

                    float nd = my_dist + edge_cost;
                    unsigned short nb_lk = make_local_key(
                        d_out, new_span_bin, my_hc, n_span_bins, n_heights);

                    // Allocate block for neighbor cell if needed
                    int nb_block_idx = get_block(
                        nb_cell, cell_to_block, block_to_cell,
                        n_allocated, max_blocks);
                    if (nb_block_idx < 0) {
                        atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
                        continue;
                    }

                    if (block_relax_dyn(pool, span_pool, nb_block_idx,
                                        nb_lk, nd, new_span_m)) {
                        // Determine which queue: within-bucket -> output,
                        // Route relaxed state to correct queue based on distance and bucket
                        {
                            long long new_state = (long long)nb_cell * spc
                                + (long long)d_out * sh
                                + (long long)new_span_bin * n_heights + my_hc;
                            if (nd >= blo && nd < bhi) {
                                // Within current bucket -> output (for next light iteration)
                                int p = atomicAdd((int*)&control[V3_CTL_OUTPUT], 1);
                                if (p < buf_size) output_queue[p] = new_state;
                                else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
                            } else {
                                // Outside bucket -> pending (for future buckets)
                                int p = atomicAdd((int*)&control[V3_CTL_PENDING], 1);
                                if (p < buf_size) pending_queue[p] = new_state;
                                else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
                            }
                        }
                    }
                }
            }
            // Mark tower eligibility for Phase B
            if (my_span_bin >= min_span_bin) my_want_tower = 1;
        }

        // ---- Phase B: Warp-cooperative tower placement ----
        // Round-robin through lanes that have tower-eligible items.
        // For each owner lane, ALL 32 lanes cooperate on area cost
        // summation and clearance checking.
        unsigned int active = __activemask();
        unsigned int tower_mask = __ballot_sync(active, my_want_tower);

        while (tower_mask != 0) {
            int owner = __ffs(tower_mask) - 1;
            tower_mask &= tower_mask - 1;

            // Broadcast owner's state to all lanes
            long long o_state = shfl_sync_i64(active, my_state, owner);
            float o_dist = __shfl_sync(active, my_dist, owner);
            int o_cell = __shfl_sync(active, my_cell, owner);
            int o_dir = __shfl_sync(active, my_dir, owner);
            int o_row = __shfl_sync(active, my_row, owner);
            int o_col_val = __shfl_sync(active, my_col_val, owner);
            int o_sv_int = __shfl_sync(active, (int)my_sv, owner);
            unsigned short o_sv = (unsigned short)o_sv_int;
            float o_span_m = __shfl_sync(active, my_span_m, owner);
            int o_hc = __shfl_sync(active, my_hc, owner);

            // Process each direction for tower placement
            for (int d_out = 0; d_out < n_steps; d_out++) {
                if (s_angle_valid[o_dir * n_steps + d_out] == 0) continue;

                int dr = (int)s_steps[d_out * 2];
                int dc = (int)s_steps[d_out * 2 + 1];
                int nr = o_row + dr;
                int nc = o_col_val + dc;
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

                int nb_cell = nr * cols + nc;
                unsigned short dv = raster[nb_cell];
                if (dv == (unsigned short)max_cost) continue;

                float ic = 0.0f;
                bool ok = true;
                int ni = s_n_inter[d_out];
                for (int k = 0; k < ni; k++) {
                    int ir = o_row + (int)s_inter_lut[(d_out * max_inter_cols + k) * 2];
                    int icc = o_col_val + (int)s_inter_lut[(d_out * max_inter_cols + k) * 2 + 1];
                    if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) { ok = false; break; }
                    unsigned short iv = raster[ir * cols + icc];
                    if (iv == (unsigned short)max_cost) { ok = false; break; }
                    ic += (float)iv;
                }
                if (!ok) continue;

                float terrain_cost = ((float)o_sv + (float)dv + ic) * s_cost_factors[d_out];
                float step_dist_m = s_step_dist[d_out];

                if (dem != NULL) {
                    float elev_src = dem[o_cell];
                    float elev_dst = dem[nb_cell];
                    float slope = fabsf(elev_dst - elev_src) / step_dist_m * 100.0f;
                    if (slope > max_gradient_pct) continue;
                    float grad_mult = expf(gradient_scale * slope / 100.0f);
                    terrain_cost *= grad_mult;
                }

                float angle_penalty = s_angle_cost[o_dir * n_steps + d_out];
                float edge_cost = terrain_cost + angle_penalty;

                // Warp-cooperative area cost computation
                float my_area_sum = 0.0f;
                int my_forbidden = 0;
                float my_slope_sum = 0.0f;
                int my_slope_count = 0;

                if (area_offsets != NULL) {
                    int pair_idx = o_dir * n_steps + d_out;
                    int ao_start = area_starts[pair_idx];
                    int ao_count = area_counts[pair_idx];

                    for (int p = lane; p < ao_count; p += 32) {
                        int a_dr = area_offsets[(ao_start + p) * 2];
                        int a_dc = area_offsets[(ao_start + p) * 2 + 1];
                        int pr = o_row + a_dr, pc = o_col_val + a_dc;
                        if (pr >= 0 && pr < rows && pc >= 0 && pc < cols) {
                            int pcell = pr * cols + pc;
                            unsigned short aval = raster[pcell];
                            if (aval == 65535u) my_forbidden = 1;
                            else my_area_sum += tower_terrain_lut[aval];

                            if (dem != NULL) {
                                float elev = dem[pcell];
                                float max_diff = 0.0f;
                                if (pr > 0) { float dd = fabsf(dem[pcell - cols] - elev); if (dd > max_diff) max_diff = dd; }
                                if (pr < rows-1) { float dd = fabsf(dem[pcell + cols] - elev); if (dd > max_diff) max_diff = dd; }
                                if (pc > 0) { float dd = fabsf(dem[pcell - 1] - elev); if (dd > max_diff) max_diff = dd; }
                                if (pc < cols-1) { float dd = fabsf(dem[pcell + 1] - elev); if (dd > max_diff) max_diff = dd; }
                                my_slope_sum += max_diff / cell_size * 100.0f;
                                my_slope_count += 1;
                            }
                        }
                    }
                } else {
                    // Uniform mode: single-pixel cost + slope
                    if (lane == 0) {
                        my_area_sum = tower_terrain_lut[o_sv];
                        if (dem != NULL) {
                            int idx = o_cell;
                            float elev = dem[idx];
                            float max_diff = 0.0f;
                            if (o_row > 0) { float dd = fabsf(dem[idx - cols] - elev); if (dd > max_diff) max_diff = dd; }
                            if (o_row < rows-1) { float dd = fabsf(dem[idx + cols] - elev); if (dd > max_diff) max_diff = dd; }
                            if (o_col_val > 0) { float dd = fabsf(dem[idx - 1] - elev); if (dd > max_diff) max_diff = dd; }
                            if (o_col_val < cols-1) { float dd = fabsf(dem[idx + 1] - elev); if (dd > max_diff) max_diff = dd; }
                            my_slope_sum = max_diff / cell_size * 100.0f;
                            my_slope_count = 1;
                        }
                    }
                }

                // Warp-reduce forbidden check
                unsigned int forbid_ballot = __ballot_sync(active, !my_forbidden);
                if (forbid_ballot != active) continue;

                // Warp-reduce area cost sum
                for (int off = 16; off > 0; off >>= 1)
                    my_area_sum += __shfl_down_sync(active, my_area_sum, off);
                float total_area_cost = __shfl_sync(active, my_area_sum, 0);

                // Warp-reduce slope
                for (int off = 16; off > 0; off >>= 1) {
                    my_slope_sum += __shfl_down_sync(active, my_slope_sum, off);
                    my_slope_count += __shfl_down_sync(active, my_slope_count, off);
                }
                float avg_slope = 0.0f;
                if (lane == 0 && my_slope_count > 0) avg_slope = my_slope_sum / (float)my_slope_count;
                avg_slope = __shfl_sync(active, avg_slope, 0);
                float slope_mult = (avg_slope > 0.0f) ? expf(gradient_scale * avg_slope / 100.0f) : 1.0f;

                float tower_terrain = total_area_cost * slope_mult;
                float tower_angle = s_tower_angle[o_dir * n_steps + d_out];

                // Height classes sorted descending (tallest first)
                for (int hc = 0; hc < n_heights; hc++) {
                    float tower_cost = tower_terrain + tower_angle
                                     + s_height_premiums[hc];
                    int reset_span_bin = (int)(step_dist_m / span_bin_size);
                    if (reset_span_bin >= n_span_bins) continue;

                    // Warp-cooperative clearance check
                    if (dem != NULL) {
                        float th = s_tower_heights[hc];
                        int clr_ok = warp_cooperative_clearance(
                            active, lane,
                            o_cell, o_dir, o_span_m, th,
                            dem, obstacle, rows, cols,
                            s_steps, s_step_dist,
                            cond_weight, cond_tension, min_clearance);
                        if (!clr_ok) break;
                    }

                    float total_edge = edge_cost + tower_cost;

                    // Phase filter: light keeps total_edge <= delta,
                    //               heavy keeps total_edge > delta
                    if (phase == 0 && total_edge > delta) continue;
                    if (phase == 1 && total_edge <= delta) continue;

                    // Only lane 0 writes relaxation + tower record
                    if (lane == 0) {
                        long long new_state = (long long)nb_cell * spc
                            + (long long)d_out * sh
                            + (long long)reset_span_bin * n_heights
                            + hc;
                        float nd = o_dist + total_edge;
                        unsigned short nb_lk = make_local_key(
                            d_out, reset_span_bin, hc, n_span_bins, n_heights);

                        // Allocate block for neighbor cell if needed
                        int nb_block_idx = get_block(
                            nb_cell, cell_to_block, block_to_cell,
                            n_allocated, max_blocks);
                        if (nb_block_idx < 0) {
                            atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
                        } else if (block_relax_dyn(pool, span_pool, nb_block_idx,
                                                   nb_lk, nd, step_dist_m)) {
                            // Record tower placement
                            int tr_idx = atomicAdd((int*)&control[V3_CTL_TOWER], 1);
                            if (tr_idx < max_tower_records) {
                                tower_records[tr_idx].state = new_state;
                                tower_records[tr_idx].pred_state = o_state;
                                tower_records[tr_idx].span_dist = __float2half(o_span_m);
                                tower_records[tr_idx].tower_height = __float2half(
                                    s_tower_heights[hc]);
                            }

                            // Route to correct queue based on distance and bucket
                            if (nd >= blo && nd < bhi) {
                                // Within current bucket -> output
                                int p = atomicAdd((int*)&control[V3_CTL_OUTPUT], 1);
                                if (p < buf_size) output_queue[p] = new_state;
                                else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
                            } else {
                                // Outside bucket -> pending
                                int p = atomicAdd((int*)&control[V3_CTL_PENDING], 1);
                                if (p < buf_size) pending_queue[p] = new_state;
                                else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
                            }
                        }
                    }
                }
            }
        }
    }
}

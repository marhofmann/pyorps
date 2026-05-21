#include "common.cuh"
#include "hash_table.cuh"
#include "block_sparse.cuh"
#include "grid_barrier.cuh"
#include "clearance.cuh"
#include "state_access.cuh"

extern "C" __global__
void constrained_persistent(
    // Raster data
    const unsigned short* __restrict__ raster,
    const int rows, const int cols, const int max_cost,
    // Step lookup tables
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    // Constrained LUTs
    const float*          __restrict__ angle_cost_lut,
    const unsigned char*  __restrict__ angle_valid_lut,
    const float*          __restrict__ step_distances,
    const float*          __restrict__ tower_terrain_lut,
    const float*          __restrict__ tower_angle_lut,
    // Height parameters
    const float*          __restrict__ height_premiums,
    const float*          __restrict__ tower_heights,
    const int n_heights,
    // Span parameters
    const int n_span_bins,
    const float span_bin_size,
    const int min_span_bin,
    // State space
    const long long spc,  // states_per_cell = n_steps * n_span_bins * n_heights
    const long long total_states,
    // Distance + span arrays
    float*                __restrict__ dist,
    __half*               __restrict__ span_dist,
    // Delta-stepping parameters
    const float delta,
    const int max_light_iters,
    // Target info for early termination
    const int*            __restrict__ targets,
    const int n_targets, const float margin,
    // Control + queues
    volatile int*         control,
    long long*            queue_a,
    long long*            queue_b,
    long long*            settled,
    long long*            pending,
    const int buf_size,
    // Tower records
    TowerRecord*          tower_records,
    const int max_tower_records,
    // DEM + clearance parameters
    const float*          __restrict__ dem,
    const float*          __restrict__ obstacle,
    const float cell_size,
    const float cond_weight,
    const float cond_tension,
    const float min_clearance,
    const float max_gradient_pct,
    const float gradient_scale,
    // Area cost: rotated square footprint offsets (NULL if uniform mode)
    const int*            __restrict__ area_offsets,
    const int*            __restrict__ area_starts,
    const int*            __restrict__ area_counts,
    // Sparse hash table parameters (NULL/0 if dense mode)
    StateEntry*           state_table,
    const int             hash_mask,
    const int             hash_capacity,
    // Block-sparse storage (NULL/0 if not used)
    BlockEntry*           block_entries,
    __half*               block_span,
    const int             storage_mode    // 0=dense, 1=sparse_hash, 2=block_sparse
) {
    int n_blocks = gridDim.x;
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
    int swap = 0;

    // ================================================================
    // Main persistent loop
    // ================================================================
    while (true) {
        // ---- Find next frontier ----
        while (control[CTL_COUNT_A] == 0) {
            int pc = control[CTL_PENDING];
            if (pc > 0) {
                // Classify pending into near/far for current bucket
                if (gtid == 0) { control[CTL_NEAR] = 0; control[CTL_FAR] = 0; }
                grid_barrier(control, n_blocks);
                int bkt = control[CTL_BUCKET];
                float bl = bkt * delta, bh = (bkt + 1) * delta;
                long long* qa = swap ? queue_b : queue_a;
                for (int i = gtid; i < pc; i += stride) {
                    long long s = pending[i];
                    // Unpack state for read_dist
                    long long s_cell_ll = s / spc;
                    int s_cell_i = (int)s_cell_ll;
                    long long s_rem = s - s_cell_ll * spc;
                    int s_dir = (int)(s_rem / sh);
                    long long s_rem2 = s_rem - (long long)s_dir * sh;
                    int s_sb = (int)(s_rem2 / n_heights);
                    int s_hc = (int)(s_rem2 % n_heights);
                    unsigned short s_lk = make_local_key(s_dir, s_sb, s_hc, n_span_bins, n_heights);
                    float d = read_dist(dist, state_table, hash_mask, block_entries, s, s_cell_i, s_lk, storage_mode);
                    if (d < bl || d >= 1e30f) continue;
                    if (d < bh) {
                        int p = atomicAdd((int*)&control[CTL_NEAR], 1);
                        if (p < buf_size) qa[p] = s;
                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                    } else {
                        int p = atomicAdd((int*)&control[CTL_FAR], 1);
                        if (p < buf_size) settled[p] = s;
                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                    }
                }
                grid_barrier(control, n_blocks);
                int fc = control[CTL_FAR];
                int fc2 = fc < buf_size ? fc : buf_size;
                for (int i = gtid; i < fc2; i += stride)
                    pending[i] = settled[i];
                if (gtid == 0) {
                    int nr = control[CTL_NEAR];
                    control[CTL_COUNT_A] = nr < buf_size ? nr : buf_size;
                    control[CTL_PENDING] = fc < buf_size ? fc : buf_size;
                }
                grid_barrier(control, n_blocks);
                if (control[CTL_COUNT_A] > 0) break;
                if (control[CTL_PENDING] > 0) {
                    if (gtid == 0) control[CTL_BUCKET] += 1;
                    grid_barrier(control, n_blocks);
                    continue;
                }
            }
            // Full scan fallback: find minimum unsettled distance
            // Limit full scans to prevent hangs on no-path scenarios
            if (gtid == 0) {
                control[CTL_FULL_SCANS] += 1;
                control[CTL_MIN_DIST] = __float_as_int(1e30f);
            }
            grid_barrier(control, n_blocks);
            if (control[CTL_FULL_SCANS] > MAX_FULL_SCANS) {
                if (gtid == 0) control[CTL_DONE] = 1;
                grid_barrier(control, n_blocks);
                break;
            }
            float blw = control[CTL_BUCKET] * delta;
            float local_min = 1e30f;
            if (storage_mode == STORAGE_BLOCK) {
                long long scan_size = (long long)rows * cols * BLOCK_SIZE;
                for (long long i = gtid; i < scan_size; i += stride) {
                    if (block_entries[i].local_key != BLOCK_EMPTY) {
                        float d = block_entries[i].dist;
                        if (d >= blw && d < local_min) local_min = d;
                    }
                }
            } else if (storage_mode == STORAGE_SPARSE) {
                for (int i = gtid; i < hash_capacity; i += stride) {
                    if (state_table[i].key != HASH_EMPTY) {
                        float d = state_table[i].dist;
                        if (d >= blw && d < local_min) local_min = d;
                    }
                }
            } else {
                for (long long i = gtid; i < total_states; i += stride) {
                    float d = dist[i];
                    if (d >= blw && d < local_min) local_min = d;
                }
            }
            if (local_min < 1e30f)
                atomicMin((int*)&control[CTL_MIN_DIST], __float_as_int(local_min));
            grid_barrier(control, n_blocks);
            float gm = __int_as_float(control[CTL_MIN_DIST]);
            if (gm >= 1e29f) {
                if (gtid == 0) control[CTL_DONE] = 1;
                grid_barrier(control, n_blocks);
                break;
            }
            int nb = (int)(gm / delta);
            float fl = nb * delta, fh = (nb + 1) * delta;
            if (gtid == 0) {
                control[CTL_BUCKET] = nb;
                control[CTL_NEAR] = 0;
            }
            grid_barrier(control, n_blocks);
            long long* qa = swap ? queue_b : queue_a;
            if (storage_mode == STORAGE_BLOCK) {
                long long scan_size = (long long)rows * cols * BLOCK_SIZE;
                for (long long i = gtid; i < scan_size; i += stride) {
                    if (block_entries[i].local_key != BLOCK_EMPTY) {
                        float d = block_entries[i].dist;
                        if (d >= fl && d < fh) {
                            // Reconstruct full state from block index
                            int cell = (int)(i / BLOCK_SIZE);
                            unsigned short lk = block_entries[i].local_key;
                            int dir = lk / (n_span_bins * n_heights);
                            int rem = lk % (n_span_bins * n_heights);
                            int sb = rem / n_heights;
                            int hc = rem % n_heights;
                            long long state = (long long)cell * spc + dir * sh + sb * n_heights + hc;
                            int p = atomicAdd((int*)&control[CTL_NEAR], 1);
                            if (p < buf_size) qa[p] = state;
                            else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                        }
                    }
                }
            } else if (storage_mode == STORAGE_SPARSE) {
                for (int i = gtid; i < hash_capacity; i += stride) {
                    if (state_table[i].key != HASH_EMPTY) {
                        long long i_state = state_table[i].key;
                        float d = state_table[i].dist;
                        if (d >= fl && d < fh) {
                            int p = atomicAdd((int*)&control[CTL_NEAR], 1);
                            if (p < buf_size) qa[p] = i_state;
                            else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                        }
                    }
                }
            } else {
                for (long long i = gtid; i < total_states; i += stride) {
                    float d = dist[i];
                    if (d >= fl && d < fh) {
                        int p = atomicAdd((int*)&control[CTL_NEAR], 1);
                        if (p < buf_size) qa[p] = i;
                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                    }
                }
            }
            grid_barrier(control, n_blocks);
            if (gtid == 0) {
                int nr = control[CTL_NEAR];
                control[CTL_COUNT_A] = nr < buf_size ? nr : buf_size;
            }
            grid_barrier(control, n_blocks);
            break;
        }
        if (control[CTL_DONE]) break;
        if (control[CTL_COUNT_A] == 0) break;

        int bkt = control[CTL_BUCKET];
        float blo = bkt * delta, bhi = (bkt + 1) * delta;

        // Copy frontier to settled
        {
            int ca = control[CTL_COUNT_A];
            if (ca > buf_size) ca = buf_size;
            long long* qa = swap ? queue_b : queue_a;
            for (int i = gtid; i < ca && i < buf_size; i += stride)
                settled[i] = qa[i];
            if (gtid == 0)
                control[CTL_SETTLED] = ca < buf_size ? ca : buf_size;
        }
        grid_barrier(control, n_blocks);

        // ---- Light phase (hybrid per-thread + warp-cooperative) ----
        // Per-thread iteration for non-tower edges (like v4 unconstrained).
        // Tower placement uses warp-cooperative protocol in batches of 32.
        // This eliminates the 75% lane waste of the old per-warp approach.
        int lane = threadIdx.x & 31;
        int warp_id_global = gtid >> 5;
        int n_warps = stride >> 5;

        for (int li = 0; li < max_light_iters; li++) {
            if (gtid == 0) control[CTL_COUNT_B] = 0;
            grid_barrier(control, n_blocks);

            long long* qa = swap ? queue_b : queue_a;
            long long* qb = swap ? queue_a : queue_b;
            int ca = control[CTL_COUNT_A];
            if (ca > buf_size) ca = buf_size;

            // Process in warp-sized batches: each warp takes 32 items,
            // one per lane. Non-tower work is per-thread (full utilization).
            // After all lanes finish their neighbor loops, lanes cooperate
            // on tower placement via round-robin ballot protocol.
            for (int batch_start = warp_id_global * 32; batch_start < ca;
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

                if (my_idx < ca) {
                    my_state = qa[my_idx];
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
                    my_local_key = make_local_key(my_dir, my_span_bin, my_hc, n_span_bins, n_heights);
                    my_dist = read_dist(dist, state_table, hash_mask, block_entries, my_state, my_cell, my_local_key, storage_mode);
                    if (my_dist < 1e30f) {
                        my_row = my_cell / cols;
                        my_col_val = my_cell - my_row * cols;
                        if (my_cell < 0 || my_cell >= rows * cols) my_dist = 1e30f;
                        else {
                        my_sv = raster[my_cell];
                        if (my_sv == (unsigned short)max_cost) my_dist = 1e30f;
                        else my_span_m = read_span(span_dist, state_table, hash_mask, block_entries, block_span, my_state, my_cell, my_local_key, storage_mode);
                        }
                    }
                    }  // end state validation
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
                            if (new_span_bin < n_span_bins && edge_cost <= delta) {
                                long long new_state = (long long)nb_cell * spc
                                    + (long long)d_out * sh
                                    + (long long)new_span_bin * n_heights
                                    + my_hc;
                                float nd = my_dist + edge_cost;
                                unsigned short nb_lk = make_local_key(d_out, new_span_bin, my_hc, n_span_bins, n_heights);
                                if (relax_dist(dist, span_dist, state_table, hash_mask,
                                               block_entries, block_span,
                                               new_state, nb_cell, nb_lk, nd, new_span_m, storage_mode)) {
                                    if (nd >= blo && nd < bhi) {
                                        int p = atomicAdd((int*)&control[CTL_COUNT_B], 1);
                                        if (p < buf_size) qb[p] = new_state;
                                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                                    } else {
                                        int p = atomicAdd((int*)&control[CTL_PENDING], 1);
                                        if (p < buf_size) pending[p] = new_state;
                                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
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
                            if (total_edge > delta) continue;  // light phase only

                            // Only lane 0 writes relaxation + tower record
                            if (lane == 0) {
                                long long new_state = (long long)nb_cell * spc
                                    + (long long)d_out * sh
                                    + (long long)reset_span_bin * n_heights
                                    + hc;
                                float nd = o_dist + total_edge;
                                unsigned short nb_lk = make_local_key(d_out, reset_span_bin, hc, n_span_bins, n_heights);
                                if (relax_dist(dist, span_dist, state_table, hash_mask,
                                               block_entries, block_span,
                                               new_state, nb_cell, nb_lk, nd, step_dist_m, storage_mode)) {
                                    int tr_idx = atomicAdd((int*)&control[CTL_TOWER_COUNT], 1);
                                    if (tr_idx < max_tower_records) {
                                        tower_records[tr_idx].state = new_state;
                                        tower_records[tr_idx].pred_state = o_state;
                                        tower_records[tr_idx].span_dist = __float2half(o_span_m);
                                        tower_records[tr_idx].tower_height = __float2half(
                                            s_tower_heights[hc]);
                                    }
                                    if (nd >= blo && nd < bhi) {
                                        int p = atomicAdd((int*)&control[CTL_COUNT_B], 1);
                                        if (p < buf_size) qb[p] = new_state;
                                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                                    } else {
                                        int p = atomicAdd((int*)&control[CTL_PENDING], 1);
                                        if (p < buf_size) pending[p] = new_state;
                                        else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            grid_barrier(control, n_blocks);
            int nc = control[CTL_COUNT_B];
            if (nc > buf_size) nc = buf_size;
            if (nc == 0) break;
            // Append new frontier to settled
            long long* qb2 = swap ? queue_a : queue_b;
            int os = control[CTL_SETTLED];
            for (int i = gtid; i < nc; i += stride) {
                int d = os + i;
                if (d < buf_size) settled[d] = qb2[i];
            }
            if (gtid == 0) {
                int ns = os + nc;
                control[CTL_SETTLED] = ns < buf_size ? ns : buf_size;
                control[CTL_COUNT_A] = nc < buf_size ? nc : buf_size;
            }
            swap ^= 1;
            grid_barrier(control, n_blocks);
        }

        // ---- Heavy phase (hybrid per-thread + warp-cooperative) ----
        if (gtid == 0) control[CTL_COUNT_B] = 0;
        grid_barrier(control, n_blocks);
        {
            int sc = control[CTL_SETTLED];
            if (sc > buf_size) sc = buf_size;
            long long* qb = swap ? queue_a : queue_b;

            // Process in warp-sized batches (same pattern as light phase)
            for (int batch_start = warp_id_global * 32; batch_start < sc;
                 batch_start += n_warps * 32)
            {
                int my_idx = batch_start + lane;

                // ---- Per-thread variables ----
                long long my_state = -1;
                float my_dist = 1e30f;
                int my_cell = -1, my_dir = -1, my_span_bin = -1, my_hc = -1;
                int my_row = -1, my_col_val = -1;
                unsigned short my_sv = 65535u;
                float my_span_m = 0.0f;
                int my_want_tower = 0;
                unsigned short my_local_key = BLOCK_EMPTY;

                if (my_idx < sc) {
                    my_state = settled[my_idx];
                    if (my_state < 0 || my_state >= total_states) {
                        my_dist = 1e30f;
                    } else {
                    // Unpack state components
                    long long cur_cell_ll = my_state / spc;
                    my_cell = (int)cur_cell_ll;
                    long long rem = my_state - cur_cell_ll * spc;
                    my_dir = (int)(rem / sh);
                    long long rem2 = rem - (long long)my_dir * sh;
                    my_span_bin = (int)(rem2 / n_heights);
                    my_hc = (int)(rem2 % n_heights);
                    my_local_key = make_local_key(my_dir, my_span_bin, my_hc, n_span_bins, n_heights);
                    my_dist = read_dist(dist, state_table, hash_mask, block_entries, my_state, my_cell, my_local_key, storage_mode);
                    if (my_dist < 1e30f) {
                        my_row = my_cell / cols;
                        my_col_val = my_cell - my_row * cols;
                        if (my_cell < 0 || my_cell >= rows * cols) my_dist = 1e30f;
                        else {
                        my_sv = raster[my_cell];
                        if (my_sv == (unsigned short)max_cost) my_dist = 1e30f;
                        else my_span_m = read_span(span_dist, state_table, hash_mask, block_entries, block_span, my_state, my_cell, my_local_key, storage_mode);
                        }
                    }
                    }
                }

                // ---- Phase A: Per-thread non-tower relaxations (heavy) ----
                if (my_dist < 1e30f) {
                    for (int d_out = 0; d_out < n_steps; d_out++) {
                        if (s_angle_valid[my_dir * n_steps + d_out] == 0) continue;

                        int dr = (int)s_steps[d_out * 2];
                        int dc = (int)s_steps[d_out * 2 + 1];
                        int nr = my_row + dr;
                        int nc_coord = my_col_val + dc;
                        if (nr < 0 || nr >= rows || nc_coord < 0 || nc_coord >= cols) continue;

                        int nb_cell = nr * cols + nc_coord;
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

                        // Non-tower: continue span in same direction (heavy: edge_cost > delta)
                        if (d_out == my_dir) {
                            int new_span_bin = (int)(new_span_m / span_bin_size);
                            if (new_span_bin < n_span_bins && edge_cost > delta) {
                                long long new_state = (long long)nb_cell * spc
                                    + (long long)d_out * sh
                                    + (long long)new_span_bin * n_heights
                                    + my_hc;
                                float nd = my_dist + edge_cost;
                                unsigned short nb_lk = make_local_key(d_out, new_span_bin, my_hc, n_span_bins, n_heights);
                                if (relax_dist(dist, span_dist, state_table, hash_mask,
                                               block_entries, block_span,
                                               new_state, nb_cell, nb_lk, nd, new_span_m, storage_mode)) {
                                    int p = atomicAdd((int*)&control[CTL_COUNT_B], 1);
                                    if (p < buf_size) qb[p] = new_state;
                                    else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                                }
                            }
                        }
                    }
                    // Mark tower eligibility for Phase B
                    if (my_span_bin >= min_span_bin) my_want_tower = 1;
                }

                // ---- Phase B: Warp-cooperative tower placement (heavy) ----
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

                    for (int d_out = 0; d_out < n_steps; d_out++) {
                        if (s_angle_valid[o_dir * n_steps + d_out] == 0) continue;

                        int dr = (int)s_steps[d_out * 2];
                        int dc = (int)s_steps[d_out * 2 + 1];
                        int nr = o_row + dr;
                        int nc_coord = o_col_val + dc;
                        if (nr < 0 || nr >= rows || nc_coord < 0 || nc_coord >= cols) continue;

                        int nb_cell = nr * cols + nc_coord;
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
                        float my_area_sum_h = 0.0f;
                        int my_forbidden_h = 0;
                        float my_slope_sum_h = 0.0f;
                        int my_slope_count_h = 0;

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
                                    if (aval == 65535u) my_forbidden_h = 1;
                                    else my_area_sum_h += tower_terrain_lut[aval];

                                    if (dem != NULL) {
                                        float elev = dem[pcell];
                                        float max_diff = 0.0f;
                                        if (pr > 0) { float dd = fabsf(dem[pcell - cols] - elev); if (dd > max_diff) max_diff = dd; }
                                        if (pr < rows-1) { float dd = fabsf(dem[pcell + cols] - elev); if (dd > max_diff) max_diff = dd; }
                                        if (pc > 0) { float dd = fabsf(dem[pcell - 1] - elev); if (dd > max_diff) max_diff = dd; }
                                        if (pc < cols-1) { float dd = fabsf(dem[pcell + 1] - elev); if (dd > max_diff) max_diff = dd; }
                                        my_slope_sum_h += max_diff / cell_size * 100.0f;
                                        my_slope_count_h += 1;
                                    }
                                }
                            }
                        } else {
                            if (lane == 0) {
                                my_area_sum_h = tower_terrain_lut[o_sv];
                                if (dem != NULL) {
                                    int idx = o_cell;
                                    float elev = dem[idx];
                                    float max_diff = 0.0f;
                                    if (o_row > 0) { float dd = fabsf(dem[idx - cols] - elev); if (dd > max_diff) max_diff = dd; }
                                    if (o_row < rows-1) { float dd = fabsf(dem[idx + cols] - elev); if (dd > max_diff) max_diff = dd; }
                                    if (o_col_val > 0) { float dd = fabsf(dem[idx - 1] - elev); if (dd > max_diff) max_diff = dd; }
                                    if (o_col_val < cols-1) { float dd = fabsf(dem[idx + 1] - elev); if (dd > max_diff) max_diff = dd; }
                                    my_slope_sum_h = max_diff / cell_size * 100.0f;
                                    my_slope_count_h = 1;
                                }
                            }
                        }

                        // Warp-reduce forbidden check
                        unsigned int forbid_ballot_h = __ballot_sync(active, !my_forbidden_h);
                        if (forbid_ballot_h != active) continue;

                        // Warp-reduce area cost sum
                        for (int off = 16; off > 0; off >>= 1)
                            my_area_sum_h += __shfl_down_sync(active, my_area_sum_h, off);
                        float total_area_cost_h = __shfl_sync(active, my_area_sum_h, 0);

                        // Warp-reduce slope
                        for (int off = 16; off > 0; off >>= 1) {
                            my_slope_sum_h += __shfl_down_sync(active, my_slope_sum_h, off);
                            my_slope_count_h += __shfl_down_sync(active, my_slope_count_h, off);
                        }
                        float avg_slope_h = 0.0f;
                        if (lane == 0 && my_slope_count_h > 0) avg_slope_h = my_slope_sum_h / (float)my_slope_count_h;
                        avg_slope_h = __shfl_sync(active, avg_slope_h, 0);
                        float slope_mult_h = (avg_slope_h > 0.0f) ? expf(gradient_scale * avg_slope_h / 100.0f) : 1.0f;

                        float tower_terrain = total_area_cost_h * slope_mult_h;
                        float tower_angle = s_tower_angle[o_dir * n_steps + d_out];

                        // Height classes sorted descending; early exit on clearance fail
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
                            if (total_edge <= delta) continue;  // heavy phase only

                            // Only lane 0 writes relaxation + tower record
                            if (lane == 0) {
                                long long new_state = (long long)nb_cell * spc
                                    + (long long)d_out * sh
                                    + (long long)reset_span_bin * n_heights
                                    + hc;
                                float nd = o_dist + total_edge;
                                unsigned short nb_lk = make_local_key(d_out, reset_span_bin, hc, n_span_bins, n_heights);
                                if (relax_dist(dist, span_dist, state_table, hash_mask,
                                               block_entries, block_span,
                                               new_state, nb_cell, nb_lk, nd, step_dist_m, storage_mode)) {
                                    int tr_idx = atomicAdd((int*)&control[CTL_TOWER_COUNT], 1);
                                    if (tr_idx < max_tower_records) {
                                        tower_records[tr_idx].state = new_state;
                                        tower_records[tr_idx].pred_state = o_state;
                                        tower_records[tr_idx].span_dist = __float2half(o_span_m);
                                        tower_records[tr_idx].tower_height = __float2half(
                                            s_tower_heights[hc]);
                                    }
                                    int p = atomicAdd((int*)&control[CTL_COUNT_B], 1);
                                    if (p < buf_size) qb[p] = new_state;
                                    else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                                }
                            }
                        }
                    }
                }
            }
        }
        grid_barrier(control, n_blocks);

        // ---- Advance bucket ----
        if (gtid == 0) control[CTL_BUCKET] += 1;
        grid_barrier(control, n_blocks);

        int nxt = control[CTL_BUCKET];
        float nl = nxt * delta, nh = (nxt + 1) * delta;
        int hc_count = control[CTL_COUNT_B];
        int pcc = control[CTL_PENDING];
        int cmb = hc_count + pcc;

        if (cmb > 0) {
            long long* qb = swap ? queue_a : queue_b;
            for (int i = gtid; i < hc_count && i < buf_size; i += stride)
                settled[i] = qb[i];
            for (int i = gtid; i < pcc; i += stride) {
                int d = hc_count + i;
                if (d < buf_size) settled[d] = pending[i];
            }
            if (gtid == 0) { control[CTL_NEAR] = 0; control[CTL_FAR] = 0; }
            grid_barrier(control, n_blocks);
            int cl = cmb < buf_size ? cmb : buf_size;
            long long* qao = swap ? queue_b : queue_a;
            for (int i = gtid; i < cl; i += stride) {
                long long s = settled[i];
                // Unpack state for read_dist
                long long s_cell_ll = s / spc;
                int s_cell_i = (int)s_cell_ll;
                long long s_rem = s - s_cell_ll * spc;
                int s_dir = (int)(s_rem / sh);
                long long s_rem2 = s_rem - (long long)s_dir * sh;
                int s_sb = (int)(s_rem2 / n_heights);
                int s_hc = (int)(s_rem2 % n_heights);
                unsigned short s_lk = make_local_key(s_dir, s_sb, s_hc, n_span_bins, n_heights);
                float d = read_dist(dist, state_table, hash_mask, block_entries, s, s_cell_i, s_lk, storage_mode);
                if (d < nl || d >= 1e30f) continue;
                if (d < nh) {
                    int p = atomicAdd((int*)&control[CTL_NEAR], 1);
                    if (p < buf_size) qao[p] = s;
                    else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                } else {
                    int p = atomicAdd((int*)&control[CTL_FAR], 1);
                    if (p < buf_size) pending[p] = s;
                    else atomicAdd((int*)&control[CTL_QUEUE_OVERFLOW], 1);
                }
            }
            grid_barrier(control, n_blocks);
            if (gtid == 0) {
                control[CTL_COUNT_A] = control[CTL_NEAR];
                control[CTL_PENDING] = control[CTL_FAR];
            }
        } else {
            if (gtid == 0) {
                control[CTL_COUNT_A] = 0;
                control[CTL_PENDING] = 0;
            }
        }
        grid_barrier(control, n_blocks);

        // ---- Early termination check ----
        if (n_targets > 0) {
            if (gtid == 0) control[CTL_EARLY_CTR] += 1;
            grid_barrier(control, n_blocks);
            if (control[CTL_EARLY_CTR] % 10 == 0) {
                // Find worst (max) distance among best target states
                if (gtid == 0) control[CTL_MIN_DIST] = 0;
                grid_barrier(control, n_blocks);
                if (storage_mode == STORAGE_BLOCK) {
                    // Block-sparse: scan each target cell's block
                    for (int i = gtid; i < n_targets; i += stride) {
                        int t_cell = targets[i];
                        int base = block_offset(t_cell);
                        float best = 1e30f;
                        for (int slot = 0; slot < BLOCK_SIZE; slot++) {
                            if (block_entries[base + slot].local_key != BLOCK_EMPTY) {
                                float d = block_entries[base + slot].dist;
                                if (d < best) best = d;
                            }
                        }
                        int di = __float_as_int(best);
                        atomicMax((int*)&control[CTL_MIN_DIST], di);
                    }
                } else if (storage_mode == STORAGE_SPARSE) {
                    // Sparse: scan hash table for target states
                    for (int i = gtid; i < n_targets; i += stride) {
                        long long t_start = (long long)targets[i] * spc;
                        float best = 1e30f;
                        for (long long j = 0; j < spc; j++) {
                            StateEntry* e = hash_find(state_table, hash_mask, t_start + j);
                            if (e != NULL && e->dist < best) best = e->dist;
                        }
                        int di = __float_as_int(best);
                        atomicMax((int*)&control[CTL_MIN_DIST], di);
                    }
                } else {
                    for (int i = gtid; i < n_targets; i += stride) {
                        long long t_start = (long long)targets[i] * spc;
                        float best = 1e30f;
                        for (long long j = 0; j < spc; j++) {
                            float d = dist[t_start + j];
                            if (d < best) best = d;
                        }
                        int di = __float_as_int(best);
                        atomicMax((int*)&control[CTL_MIN_DIST], di);
                    }
                }
                grid_barrier(control, n_blocks);
                float mt = __int_as_float(control[CTL_MIN_DIST]);
                if (mt < 1e29f) {
                    float cutoff = mt * margin;
                    if (control[CTL_BUCKET] * delta > cutoff) {
                        if (gtid == 0) control[CTL_DONE] = 1;
                        grid_barrier(control, n_blocks);
                    }
                }
            }
        }

        if (control[CTL_DONE]) break;
    }
}

#pragma once
#include "common.cuh"

// ---- Device helper: check catenary clearance along a span (sequential) ----
// Walk backward from tower B (cur_cell) along direction for n_steps cells
// to find tower A. Then walk forward checking sag clearance at each cell.
// Returns 1 if clearance OK, 0 if violated.
// Kept as reference/fallback; the hot path uses warp-cooperative clearance.
__device__ int check_span_clearance(
    int cur_cell, int direction, float span_m, float tower_height,
    const float* __restrict__ dem, const float* __restrict__ obstacle,
    int rows, int cols,
    const signed char* s_steps, const float* s_step_dist,
    float cell_size, float cond_weight, float cond_tension, float min_clearance
) {
    if (dem == NULL) return 1;  // no DEM -- skip clearance check
    float step_d = s_step_dist[direction];
    int n_walk = (int)(span_m / step_d + 0.5f);
    if (n_walk <= 1) return 1;

    int dr = (int)s_steps[direction * 2];
    int dc = (int)s_steps[direction * 2 + 1];

    // Find tower A position by walking backward
    int cur_row = cur_cell / cols;
    int cur_col = cur_cell - cur_row * cols;
    int ta_row = cur_row - n_walk * dr;
    int ta_col = cur_col - n_walk * dc;
    if (ta_row < 0 || ta_row >= rows || ta_col < 0 || ta_col >= cols) return 0;

    int ta_cell = ta_row * cols + ta_col;
    float span_len = (float)n_walk * step_d;
    float dem_a = dem[ta_cell];
    float dem_b = dem[cur_cell];
    float attach_a = dem_a + tower_height;
    float attach_b = dem_b + tower_height;

    // Quick check: if max sag is small relative to tower height, skip walk
    float max_sag = (cond_weight * span_len * span_len) / (8.0f * cond_tension);
    if (tower_height - max_sag - min_clearance > 50.0f) return 1;

    // Walk forward from A to B checking clearance at each cell
    int walk_row = ta_row;
    int walk_col = ta_col;
    for (int step = 1; step < n_walk; step++) {
        walk_row += dr;
        walk_col += dc;
        if (walk_row < 0 || walk_row >= rows || walk_col < 0 || walk_col >= cols)
            return 0;
        float x = (float)step * step_d;
        float chord_z = attach_a + (attach_b - attach_a) * x / span_len;
        float sag_x = (cond_weight * x * (span_len - x)) / (2.0f * cond_tension);
        float cond_z = chord_z - sag_x;
        int w_cell = walk_row * cols + walk_col;
        float ground_z = dem[w_cell];
        float obs_z = (obstacle != NULL) ? obstacle[w_cell] : 0.0f;
        float clr = cond_z - ground_z - obs_z;
        if (clr < min_clearance) return 0;
    }
    return 1;
}

// ---- Warp-cooperative clearance check ----
// All 32 lanes in the warp collaborate to check catenary clearance along a
// span. Each lane checks a subset of the span cells in parallel, then a
// warp ballot reduces to determine if ALL cells pass.
// Parameters are broadcast from the owning lane via __shfl_sync before call.
// Returns 1 if clearance OK for ALL cells, 0 if any cell violates.
__device__ int warp_cooperative_clearance(
    unsigned int active_mask, int lane,
    int t_cell, int t_dir, float span_m, float tower_height,
    const float* __restrict__ dem, const float* __restrict__ obstacle,
    int rows, int cols,
    const signed char* s_steps, const float* s_step_dist,
    float cond_weight, float cond_tension, float min_clearance
) {
    if (dem == NULL) return 1;
    float step_d = s_step_dist[t_dir];
    int n_walk = (int)(span_m / step_d + 0.5f);
    if (n_walk <= 1) return 1;

    int dr = (int)s_steps[t_dir * 2];
    int dc = (int)s_steps[t_dir * 2 + 1];

    // Find tower A position by walking backward from tower B (t_cell)
    int tb_row = t_cell / cols;
    int tb_col = t_cell - tb_row * cols;
    int ta_row = tb_row - n_walk * dr;
    int ta_col = tb_col - n_walk * dc;
    // Bounds check on tower A position
    if (ta_row < 0 || ta_row >= rows || ta_col < 0 || ta_col >= cols) return 0;

    int ta_cell = ta_row * cols + ta_col;
    float span_len = (float)n_walk * step_d;
    float dem_a = dem[ta_cell];
    float dem_b = dem[t_cell];
    float attach_a = dem_a + tower_height;
    float attach_b = dem_b + tower_height;

    // Quick check: if max sag is small relative to tower height, skip walk
    float max_sag = (cond_weight * span_len * span_len) / (8.0f * cond_tension);
    if (tower_height - max_sag - min_clearance > 50.0f) return 1;

    // Parallel walk: each lane checks a subset of span cells
    // n_walk-1 interior cells to check (skip endpoints = tower positions)
    int n_check = n_walk - 1;
    int my_clearance_ok = 1;
    for (int c = lane; c < n_check; c += 32) {
        int step = c + 1;  // step 1..n_walk-1
        int w_row = ta_row + step * dr;
        int w_col = ta_col + step * dc;
        if (w_row < 0 || w_row >= rows || w_col < 0 || w_col >= cols) {
            my_clearance_ok = 0;
            break;
        }
        float x = (float)step * step_d;
        float chord_z = attach_a + (attach_b - attach_a) * x / span_len;
        float sag_x = (cond_weight * x * (span_len - x)) / (2.0f * cond_tension);
        float cond_z = chord_z - sag_x;
        int w_cell = w_row * cols + w_col;
        float ground_z = dem[w_cell];
        float obs_z = (obstacle != NULL) ? obstacle[w_cell] : 0.0f;
        float clr = cond_z - ground_z - obs_z;
        if (clr < min_clearance) {
            my_clearance_ok = 0;
            break;
        }
    }

    // Warp-reduce: ALL lanes must pass clearance
    unsigned int clear_ballot = __ballot_sync(active_mask, my_clearance_ok);
    return (clear_ballot == active_mask) ? 1 : 0;
}

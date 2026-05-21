// constrained_adds.cu — Main persistent ADDS kernel entry point
//
// Combines all V4 ADDS components into a single kernel launch:
//   Block 0  -> MTB (Manager Thread Block)
//   Block 1+ -> WTBs (Worker Thread Blocks)
//
// MTB reads items from the bucket queue, assigns work to idle WTBs.
// WTBs relax edges and place towers, enqueuing new items back.
// Communication: MTB writes AssignmentFlags, WTBs poll them.
// Termination: MTB sets control[CTL_V4_DONE] when finished.
//
// This kernel does NOT require cooperative groups; it uses
// volatile flag polling for inter-block coordination.
//
// Compile-time constants (must be injected by Python driver):
//   BLOCK_SIZE, N_BUCKETS, SEGMENT_SIZE, ITEMS_PER_BUCKET,
//   MAX_SEGMENTS_PER_BUCKET, MAX_INTER, MAX_WTBS, MTB_STAGING_SIZE

#include "adds_common.cuh"
#include "adds_bucket_queue.cuh"
#include "adds_tower.cuh"
#include "adds_wtb.cuh"
#include "adds_mtb.cuh"

// ============ Main Persistent Kernel Entry Point ============

extern "C" __global__ void constrained_adds_main(
    // ---- Raster ----
    const unsigned short* __restrict__ raster,
    int rows, int cols,
    // ---- Block-sparse distance storage ----
    BlockEntry* pool,
    __half* span_pool,
    int* cell_to_block,
    int* block_to_cell,
    int* n_allocated,
    int max_sparse_blocks,
    // ---- Bucket queue ----
    WorkItem* bucket_pool,
    int* bucket_resv_ptr,
    int* bucket_read_ptr,
    int* bucket_generation,
    int* bucket_wcc,
    int max_pool_blocks,  // for overflow detection
    // ---- Assignment flags (n_wtbs entries) ----
    AssignmentFlag* assignment_flags,
    int n_wtbs,
    // ---- Tower records ----
    TowerRecordV4* tower_records,
    int max_tower_records,
    // ---- Control buffer (CTL_V4_SIZE entries) ----
    int* control,
    // ---- Best target distance (int-encoded float via __float_as_int) ----
    int* best_target_dist,
    int target_cell,
    // ---- Profile LUTs (global memory -> loaded to shared by each block) ----
    const signed char* g_steps,            // (n_dirs * 2)
    const float* g_cost_factors,           // (n_dirs)
    const float* g_step_distances,         // (n_dirs)
    const unsigned char* g_angle_valid,    // (n_dirs * n_dirs)
    const float* g_angle_cost,             // (n_dirs * n_dirs)
    const float* g_tower_terrain,          // (65536) stays in global (too large for smem)
    const float* g_tower_angle_cost,       // (n_dirs * n_dirs)
    const float* g_height_premiums,        // (n_heights)
    const short* g_intermediates,          // (n_dirs * max_inter_cols * 2)
    const int* g_n_intermediates,          // (n_dirs)
    // ---- Parameters ----
    int n_dirs,
    int n_span_bins,
    int n_heights,
    float min_span,
    float max_span,
    float span_bin_size,
    float initial_delta,
    float margin,
    int max_assignments,
    int max_inter_cols
) {
    // ================================================================
    // Shared memory layout for profile LUTs
    // ================================================================
    // All blocks load LUTs into shared memory for fast access.
    // MTB only uses shared memory for its own coordination variables
    // (declared inside mtb_main_loop), but loading LUTs here is harmless
    // and keeps the dispatch simple.
    //
    // Layout with proper alignment:
    //   s_steps:            n_dirs * 2 bytes (int8)     -> pad to 4-byte boundary
    //   s_cost_factors:     n_dirs * 4 bytes (float32)
    //   s_step_distances:   n_dirs * 4 bytes (float32)
    //   s_angle_valid:      n_dirs^2 bytes (uint8)      -> pad to 4-byte boundary
    //   s_angle_cost:       n_dirs^2 * 4 bytes (float32)
    //   s_tower_angle_cost: n_dirs^2 * 4 bytes (float32)
    //   s_height_premiums:  n_heights * 4 bytes (float32)
    //   s_intermediates:    n_dirs * max_inter_cols * 2 * 2 bytes (int16) -> pad to 4
    //   s_n_intermediates:  n_dirs * 4 bytes (int32)

    extern __shared__ char smem_raw[];

    int n2 = n_dirs * n_dirs;
    int steps_bytes = n_dirs * 2;
    int steps_padded = (steps_bytes + 3) & ~3;

    signed char*    s_steps          = (signed char*)smem_raw;
    float*          s_cost_factors   = (float*)(smem_raw + steps_padded);
    float*          s_step_distances = s_cost_factors + n_dirs;

    int av_offset = steps_padded + n_dirs * 4 + n_dirs * 4;
    unsigned char*  s_angle_valid    = (unsigned char*)(smem_raw + av_offset);
    int av_padded = (n2 + 3) & ~3;

    float*          s_angle_cost       = (float*)(smem_raw + av_offset + av_padded);
    float*          s_tower_angle_cost = s_angle_cost + n2;
    float*          s_height_premiums  = s_tower_angle_cost + n2;

    int inter_total = n_dirs * max_inter_cols * 2;  // number of int16 values
    int inter_bytes = inter_total * 2;              // bytes (int16 = 2 bytes)
    int inter_padded = (inter_bytes + 3) & ~3;
    short*          s_intermediates   = (short*)(s_height_premiums + n_heights);
    int*            s_n_intermediates = (int*)((char*)s_intermediates + inter_padded);

    // ---- Cooperative loading: all threads in each block participate ----
    for (int i = threadIdx.x; i < steps_bytes; i += blockDim.x)
        s_steps[i] = g_steps[i];
    for (int i = threadIdx.x; i < n_dirs; i += blockDim.x) {
        s_cost_factors[i] = g_cost_factors[i];
        s_step_distances[i] = g_step_distances[i];
    }
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        s_angle_valid[i] = g_angle_valid[i];
        s_angle_cost[i] = g_angle_cost[i];
        s_tower_angle_cost[i] = g_tower_angle_cost[i];
    }
    for (int i = threadIdx.x; i < n_heights; i += blockDim.x)
        s_height_premiums[i] = g_height_premiums[i];
    for (int i = threadIdx.x; i < inter_total; i += blockDim.x)
        s_intermediates[i] = g_intermediates[i];
    for (int i = threadIdx.x; i < n_dirs; i += blockDim.x)
        s_n_intermediates[i] = g_n_intermediates[i];

    __syncthreads();

    // ================================================================
    // Dispatch: block 0 -> MTB, blocks 1+ -> WTB
    // ================================================================
    if (blockIdx.x == 0) {
        // ---- Manager Thread Block ----
        mtb_main_loop(
            (volatile int*)bucket_resv_ptr,
            bucket_read_ptr,
            bucket_generation,
            (volatile int*)bucket_wcc,
            bucket_pool,
            (volatile AssignmentFlag*)assignment_flags,
            n_wtbs,
            (volatile int*)control,
            (volatile int*)best_target_dist,
            initial_delta,
            margin,
            max_assignments
        );
    } else {
        // ---- Worker Thread Block ----
        int wtb_idx = blockIdx.x - 1;  // AF index for this WTB
        if (wtb_idx >= n_wtbs) return;  // guard: excess blocks do nothing

        volatile AssignmentFlag* my_af = &((volatile AssignmentFlag*)assignment_flags)[wtb_idx];
        volatile int* done_flag = &((volatile int*)control)[CTL_V4_DONE];

        // WTBs read the initial head_logical=0. As MTB advances, WTBs
        // continue using 0 for enqueue bucket computation. This is safe:
        // enqueue_to_bucket silently drops items below head_logical, and
        // items above the window are clamped to the tail bucket.
        // For V2, WTBs could read control[CTL_V4_HEAD_LOGICAL] dynamically.
        int head_logical = 0;

        wtb_main_loop(
            blockIdx.x,
            my_af,
            done_flag,
            raster, rows, cols,
            pool, span_pool,
            cell_to_block, block_to_cell, n_allocated,
            max_sparse_blocks,
            bucket_pool, bucket_resv_ptr, bucket_generation,
            bucket_wcc, (volatile int*)control,
            initial_delta, head_logical,
            max_pool_blocks,
            tower_records, max_tower_records,
            best_target_dist, target_cell,
            s_steps, s_cost_factors, s_step_distances,
            s_angle_valid, s_angle_cost,
            g_tower_terrain,  // stays in global memory (256 KB)
            s_tower_angle_cost, s_height_premiums,
            s_intermediates, s_n_intermediates,
            n_dirs, n_span_bins, n_heights,
            min_span, max_span, span_bin_size,
            max_inter_cols
        );
    }
}

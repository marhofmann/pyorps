#pragma once
#include "adds_bucket_queue.cuh"

// ============ Initialize block-sparse pool entries to empty state ============
// Launch with enough threads to cover max_sparse_blocks * BLOCK_SIZE.
//
// BlockEntry empty state:
//   local_key = 0xFFFF  (BLOCK_EMPTY sentinel)
//   _pad      = 0xFFFF  (so first 4 bytes = 0xFFFFFFFF for atomicCAS in block_upsert_dyn)
//   dist      = +inf    (IEEE 754: 0x7F800000, ensures atomicMin works correctly)
//
// span_pool entries are zeroed (no span accumulated for empty slots).

extern "C" __global__ void adds_init_pool(
    BlockEntry* pool, __half* span_pool,
    int total_entries  // = max_sparse_blocks * BLOCK_SIZE
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_entries) return;
    pool[idx].local_key = 0xFFFF;
    pool[idx]._pad = 0xFFFF;
    pool[idx].dist = __int_as_float(0x7F800000);  // +inf
    span_pool[idx] = __float2half(0.0f);
}

// ============ Seed source states into block-sparse pool + bucket 0 ============
// Launch with 1 block, n_dirs threads.
//
// For each direction dir in [0, n_dirs):
// 1. Allocate a block for source_cell (idempotent — all threads get the same block)
// 2. Insert state (source_cell, dir, span_bin=0, height_class=0) with dist=0
// 3. Write a WorkItem to bucket 0 of the bucket queue
// 4. Thread 0 sets bucket metadata (resv_ptr, WCC)

extern "C" __global__ void adds_init_source(
    // Block-sparse storage
    BlockEntry* pool, __half* span_pool,
    int* cell_to_block, int* block_to_cell, int* n_allocated,
    int max_sparse_blocks,
    // Bucket queue
    WorkItem* bucket_pool, int* bucket_resv_ptr, int* bucket_wcc,
    // Source info
    int source_cell, int n_dirs, int n_span_bins, int n_heights
) {
    int dir = threadIdx.x;
    if (dir >= n_dirs) return;

    // 1. Thread 0 allocates the block for source_cell; others wait, then
    //    read the result from cell_to_block[]. This avoids the n_allocated
    //    over-count that occurs when N threads all call get_block().
    if (dir == 0) {
        get_block(source_cell, cell_to_block, block_to_cell,
                  n_allocated, max_sparse_blocks);
    }
    __syncthreads();
    int block_idx = cell_to_block[source_cell];
    if (block_idx < 0) return;  // pool exhausted (shouldn't happen for source)

    // 2. Insert into block-sparse pool with dist=0, span=0
    unsigned short lk = pack_local_key(dir, 0, 0, n_span_bins, n_heights);
    block_relax_dyn(pool, span_pool, block_idx, lk, 0.0f, 0.0f);

    // 3. Create WorkItem for bucket 0
    long long state = pack_state(source_cell, dir, 0, 0,
                                 n_dirs, n_span_bins, n_heights);
    WorkItem item;
    item.state = state;
    item.dist = 0.0f;
    item.span_dist = __float2half(0.0f);
    item._pad = 0;

    // 4. Write directly to bucket 0 (safe: no concurrent writers during init)
    //    Bucket 0 starts at pool offset 0 (physical bucket 0)
    bucket_pool[dir] = item;  // slot = dir

    // 5. Thread 0 sets bucket metadata after all items are written
    __syncthreads();
    if (dir == 0) {
        __threadfence();  // ensure all item writes are globally visible
        bucket_resv_ptr[0] = n_dirs;
        // WCC for segment 0 of bucket 0: count of items written
        bucket_wcc[0] = n_dirs;
    }
}

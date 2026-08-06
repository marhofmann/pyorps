#include "dynamic_blocks.cuh"

extern "C" __global__
void init_pool_v3(BlockEntry* pool, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        pool[i].local_key = BLOCK_EMPTY;
        pool[i]._pad = 0xFFFF;
        pool[i].dist = 1e30f;
    }
}

extern "C" __global__
void init_source_v3(
    BlockEntry* pool, __half* span_pool,
    int* cell_to_block, int* block_to_cell,
    int* n_allocated,
    long long* source_states, float* init_dists,
    int n_source, int spc, int n_span_bins, int n_heights,
    int max_blocks
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_source) {
        long long state = source_states[i];
        int cell = (int)(state / spc);
        int rem = (int)(state % spc);
        int sh_val = n_span_bins * n_heights;
        int dir = rem / sh_val;
        int rem2 = rem % sh_val;
        int sb = rem2 / n_heights;
        int hc = rem2 % n_heights;
        unsigned short lk = make_local_key(dir, sb, hc, n_span_bins, n_heights);

        int block_idx = get_block(cell, cell_to_block, block_to_cell,
                                  n_allocated, max_blocks);
        if (block_idx < 0) return;

        BlockEntry* e = block_upsert_dyn(pool, block_idx, lk);
        if (e != NULL) {
            e->dist = init_dists[i];
            int base = block_offset_dyn(block_idx);
            int slot = (int)(e - &pool[base]);
            span_pool[base + slot] = __float2half(0.0f);
        }
    }
}

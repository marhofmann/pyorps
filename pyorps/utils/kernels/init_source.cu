#include "hash_table.cuh"
#include "block_sparse.cuh"
#include "state_access.cuh"

extern "C" __global__
void init_source_states_sparse(
    StateEntry* table, int mask,
    long long* source_states, float* init_dists, int n_source
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n_source) {
        StateEntry* e = hash_upsert(table, mask, source_states[i]);
        if (e) {
            e->dist = init_dists[i];
            e->span_dist = __float2half(0.0f);
        }
    }
}

extern "C" __global__
void init_block_entries(BlockEntry* entries, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        entries[i].local_key = BLOCK_EMPTY;
        entries[i]._pad = 0;
        entries[i].dist = 1e30f;
    }
}

extern "C" __global__
void init_block_source(
    BlockEntry* blocks, __half* block_span_arr,
    long long* source_states, float* init_dists,
    int n_source, int spc, int n_span_bins, int n_heights
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
        unsigned short local_key = make_local_key(dir, sb, hc, n_span_bins, n_heights);

        BlockEntry* e = block_upsert(blocks, cell, local_key);
        if (e != NULL) {
            e->dist = init_dists[i];
            int base = block_offset(cell);
            int slot = (int)(e - &blocks[base]);
            block_span_arr[base + slot] = __float2half(0.0f);
        }
    }
}

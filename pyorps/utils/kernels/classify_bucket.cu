#include "dynamic_blocks.cuh"

extern "C" __global__
void classify_bucket(
    long long* pending, int pending_count,
    int bucket, float delta, int buf_size,
    int* cell_to_block, BlockEntry* pool,
    long long spc, int n_span_bins, int n_heights,
    long long* near_queue, long long* far_queue,
    volatile int* control
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pending_count) return;

    long long state = pending[i];
    long long sh = (long long)n_span_bins * n_heights;

    long long cell_ll = state / spc;
    int cell = (int)cell_ll;
    long long rem = state - cell_ll * spc;
    int dir = (int)(rem / sh);
    long long rem2 = rem - (long long)dir * sh;
    int sb = (int)(rem2 / n_heights);
    int hc = (int)(rem2 % n_heights);
    unsigned short lk = make_local_key(dir, sb, hc, n_span_bins, n_heights);

    int block_idx = cell_to_block[cell];
    float d = block_read_dist_dyn(pool, block_idx, lk);

    float blo = bucket * delta;
    float bhi = (bucket + 1) * delta;

    if (d < blo || d >= 1e30f) return;

    if (d < bhi) {
        int p = atomicAdd((int*)&control[V3_CTL_NEAR], 1);
        if (p < buf_size) near_queue[p] = state;
        else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
    } else {
        int p = atomicAdd((int*)&control[V3_CTL_FAR], 1);
        if (p < buf_size) far_queue[p] = state;
        else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
        // Track minimum far distance for bucket-skip optimization
        atomicMin((int*)&control[V3_CTL_MIN_DIST], __float_as_int(d));
    }
}

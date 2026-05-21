#include "dynamic_blocks.cuh"

extern "C" __global__
void scan_min_dist(
    BlockEntry* pool, int* block_to_cell,
    int n_allocated_blocks, float bucket_lower_bound,
    volatile int* control
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    long long scan_size = (long long)n_allocated_blocks * BLOCK_SIZE;

    float local_min = 1e30f;
    for (long long i = gtid; i < scan_size; i += stride) {
        int block_idx = (int)(i / BLOCK_SIZE);
        if (block_to_cell[block_idx] < 0) continue;
        if (pool[i].local_key == BLOCK_EMPTY) continue;
        float d = pool[i].dist;
        if (d >= bucket_lower_bound && d < local_min) local_min = d;
    }
    if (local_min < 1e30f)
        atomicMin((int*)&control[V3_CTL_MIN_DIST], __float_as_int(local_min));
}

extern "C" __global__
void extract_bucket(
    BlockEntry* pool, int* block_to_cell,
    int n_allocated_blocks,
    int bucket, float delta,
    long long spc, int n_span_bins, int n_heights,
    long long* output_queue, volatile int* control, int buf_size
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    long long scan_size = (long long)n_allocated_blocks * BLOCK_SIZE;
    long long sh = (long long)n_span_bins * n_heights;

    float blo = bucket * delta;
    float bhi = (bucket + 1) * delta;

    for (long long i = gtid; i < scan_size; i += stride) {
        int block_idx = (int)(i / BLOCK_SIZE);
        int cell = block_to_cell[block_idx];
        if (cell < 0) continue;
        unsigned short lk = pool[i].local_key;
        if (lk == BLOCK_EMPTY) continue;
        float d = pool[i].dist;
        if (d >= blo && d < bhi) {
            int dir = lk / (n_span_bins * n_heights);
            int rem = lk % (n_span_bins * n_heights);
            int sb = rem / n_heights;
            int hc = rem % n_heights;
            long long state = (long long)cell * spc
                + (long long)dir * sh
                + (long long)sb * n_heights + hc;
            int p = atomicAdd((int*)&control[V3_CTL_NEAR], 1);
            if (p < buf_size) output_queue[p] = state;
            else atomicAdd((int*)&control[V3_CTL_OVERFLOW], 1);
        }
    }
}

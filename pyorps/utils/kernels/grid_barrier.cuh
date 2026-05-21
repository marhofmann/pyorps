#pragma once
#include "common.cuh"

// Custom sense-reversing grid barrier (Blackwell-safe, no grid.sync()).
// grid.sync() on sm_120 (CUDA 13.0) does not guarantee memory ordering.
__device__ void grid_barrier(volatile int* control, int n_blocks) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int my_sense = control[CTL_BARRIER_SENSE];
        int arrived = atomicAdd((int*)&control[CTL_BARRIER_CNT], 1) + 1;
        if (arrived == n_blocks) {
            control[CTL_BARRIER_CNT] = 0;
            __threadfence();
            control[CTL_BARRIER_SENSE] = 1 - my_sense;
        } else {
            while (control[CTL_BARRIER_SENSE] == my_sense) {}
        }
    }
    __syncthreads();
}

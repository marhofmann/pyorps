"""
Raster-direct GPU delta-stepping SSSP for PYORPS.

This module implements a GPU-accelerated single-source shortest path algorithm
that operates directly on the raster grid — no explicit edge list, no graph
object, no cuGraph dependency. Neighbors are computed on-the-fly from the
raster during frontier expansion.

Algorithm:
    Delta-stepping [1] adapted for CUDA. Nodes are binned into "buckets" by
    distance range [i*delta, (i+1)*delta). Each bucket is processed in parallel:
    - Light phase: relax edges with weight <= delta (repeats until stable)
    - Heavy phase: relax edges with weight > delta (once per bucket)

    v2 uses atomic-append frontier queues with warp-aggregated atomics,
    eliminating O(n_pixels) scans per iteration. All work is O(frontier_size).

    Distance updates use atomicMin via IEEE 754 bit reinterpretation for
    lock-free, race-free relaxation of non-negative float distances.

Requirements:
    - NVIDIA GPU with CUDA support
    - cupy-cuda12x >= 13.0.0

References:
    [1] Meyer, U., Sanders, P.: delta-stepping: a parallelizable shortest path
        algorithm. J. Algorithms 49, 1 (2003), 114-152.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from pyorps.utils.traversal_gpu import (
    GPU_AVAILABLE, prepare_step_lookup_tables
)

if GPU_AVAILABLE:
    import cupy as cp


# ============================================================================
# Shared memory loader (same layout as traversal_gpu.py write kernels)
# ============================================================================

_SMEM_LOAD_RELAX = r"""
    // Shared memory layout:
    // [steps: n_steps*2 bytes] [pad to 4B] [n_inter: n_steps*4 bytes]
    // [cost_factors: n_steps*4 bytes] [inter_lut: n_steps*max_inter_cols*2 bytes]
    extern __shared__ char smem[];

    int steps_bytes = n_steps * 2;
    int steps_padded = (steps_bytes + 3) & ~3;
    signed char* s_steps = (signed char*)smem;
    int* s_n_inter = (int*)(smem + steps_padded);
    float* s_cost_factors = (float*)(s_n_inter + n_steps);
    signed char* s_inter_lut = (signed char*)(s_cost_factors + n_steps);

    int local_tid = threadIdx.x;
    int local_size = blockDim.x;

    for (int i = local_tid; i < steps_bytes; i += local_size)
        s_steps[i] = steps[i];
    for (int i = local_tid; i < n_steps; i += local_size)
        s_n_inter[i] = n_inter[i];
    for (int i = local_tid; i < n_steps; i += local_size)
        s_cost_factors[i] = cost_factors[i];
    int lut_size = n_steps * max_inter_cols * 2;
    for (int i = local_tid; i < lut_size; i += local_size)
        s_inter_lut[i] = inter_lut[i];

    __syncthreads();
"""

# ============================================================================
# CUDA C kernel sources
# ============================================================================

_RELAX_LIGHT_KERNEL = r"""
extern "C" __global__
void relax_light(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps,
    const int max_inter_cols,
    const int rows,
    const int cols,
    const int max_cost,
    float*                __restrict__ dist,
    int*                  __restrict__ pred,
    const float           delta,
    const int*            __restrict__ frontier,
    const int             frontier_size,
    int*                  __restrict__ updated_flags
) {
""" + _SMEM_LOAD_RELAX + r"""
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    int ur = u / cols;
    int uc = u - ur * cols;

    // Read current distance (relaxed ordering is fine — worst case we do
    // extra work with a stale value, next iteration corrects it)
    float u_dist = dist[u];
    if (u_dist >= 1e30f) return;

    unsigned short src_val = raster[u];
    if (src_val == max_cost) return;

    for (int s = 0; s < n_steps; s++) {
        int dr = (int)s_steps[s * 2];
        int dc = (int)s_steps[s * 2 + 1];
        int vr = ur + dr;
        int vc = uc + dc;

        if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;

        int v = vr * cols + vc;
        unsigned short dst_val = raster[v];
        if (dst_val == max_cost) continue;

        // Check intermediates
        float inter_cost = 0.0f;
        bool valid = true;
        int ni = s_n_inter[s];
        for (int k = 0; k < ni; k++) {
            int ir = ur + (int)s_inter_lut[(s * max_inter_cols + k) * 2];
            int ic = uc + (int)s_inter_lut[(s * max_inter_cols + k) * 2 + 1];
            if (ir < 0 || ir >= rows || ic < 0 || ic >= cols) {
                valid = false; break;
            }
            unsigned short ival = raster[ir * cols + ic];
            if (ival == max_cost) {
                valid = false; break;
            }
            inter_cost += (float)ival;
        }
        if (!valid) continue;

        float edge_weight = ((float)src_val + (float)dst_val + inter_cost)
                            * s_cost_factors[s];

        // Light phase: only edges with weight <= delta
        if (edge_weight > delta) continue;

        float new_dist = u_dist + edge_weight;

        // Atomic relaxation via IEEE 754 bit trick.
        // For non-negative floats: a < b  iff  __float_as_int(a) < __float_as_int(b)
        int new_dist_int = __float_as_int(new_dist);
        int old_dist_int = atomicMin((int*)&dist[v], new_dist_int);

        if (new_dist_int < old_dist_int) {
            // We improved dist[v] — record predecessor and mark updated
            if (pred != NULL) pred[v] = u;
            updated_flags[v] = 1;
        }
    }
}
"""

_RELAX_HEAVY_KERNEL = r"""
extern "C" __global__
void relax_heavy(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps,
    const int max_inter_cols,
    const int rows,
    const int cols,
    const int max_cost,
    float*                __restrict__ dist,
    int*                  __restrict__ pred,
    const float           delta,
    const int*            __restrict__ settled,
    const int             settled_size,
    int*                  __restrict__ updated_flags
) {
""" + _SMEM_LOAD_RELAX + r"""
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= settled_size) return;

    int u = settled[tid];
    int ur = u / cols;
    int uc = u - ur * cols;

    float u_dist = dist[u];
    if (u_dist >= 1e30f) return;

    unsigned short src_val = raster[u];
    if (src_val == max_cost) return;

    for (int s = 0; s < n_steps; s++) {
        int dr = (int)s_steps[s * 2];
        int dc = (int)s_steps[s * 2 + 1];
        int vr = ur + dr;
        int vc = uc + dc;

        if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;

        int v = vr * cols + vc;
        unsigned short dst_val = raster[v];
        if (dst_val == max_cost) continue;

        // Check intermediates
        float inter_cost = 0.0f;
        bool valid = true;
        int ni = s_n_inter[s];
        for (int k = 0; k < ni; k++) {
            int ir = ur + (int)s_inter_lut[(s * max_inter_cols + k) * 2];
            int ic = uc + (int)s_inter_lut[(s * max_inter_cols + k) * 2 + 1];
            if (ir < 0 || ir >= rows || ic < 0 || ic >= cols) {
                valid = false; break;
            }
            unsigned short ival = raster[ir * cols + ic];
            if (ival == max_cost) {
                valid = false; break;
            }
            inter_cost += (float)ival;
        }
        if (!valid) continue;

        float edge_weight = ((float)src_val + (float)dst_val + inter_cost)
                            * s_cost_factors[s];

        // Heavy phase: only edges with weight > delta
        if (edge_weight <= delta) continue;

        float new_dist = u_dist + edge_weight;

        int new_dist_int = __float_as_int(new_dist);
        int old_dist_int = atomicMin((int*)&dist[v], new_dist_int);

        if (new_dist_int < old_dist_int) {
            if (pred != NULL) pred[v] = u;
            updated_flags[v] = 1;
        }
    }
}
"""

# ============================================================================
# V2 CUDA kernels — atomic-append frontier queues (Tier 1 optimization)
#
# Instead of writing to a dense updated_flags[n_pixels] array and scanning
# with flatnonzero(), v2 kernels use warp-aggregated atomicAdd to append
# updated node IDs directly to a compact output queue. This eliminates
# O(n_pixels) work per iteration, replacing it with O(frontier_size).
# ============================================================================

_RELAX_LIGHT_V2_KERNEL = r"""
extern "C" __global__
void relax_light_v2(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps,
    const int max_inter_cols,
    const int rows,
    const int cols,
    const int max_cost,
    float*                __restrict__ dist,
    int*                  __restrict__ pred,
    const float           delta,
    const int*            __restrict__ frontier,
    const int             frontier_size,
    int*                  __restrict__ out_queue,
    int*                  __restrict__ out_count,
    const float           bucket_low,
    const float           bucket_high
) {
""" + _SMEM_LOAD_RELAX + r"""
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    int ur = u / cols;
    int uc = u - ur * cols;

    float u_dist = dist[u];
    if (u_dist >= 1e30f) return;

    unsigned short src_val = raster[u];
    if (src_val == max_cost) return;

    for (int s = 0; s < n_steps; s++) {
        int dr = (int)s_steps[s * 2];
        int dc = (int)s_steps[s * 2 + 1];
        int vr = ur + dr;
        int vc = uc + dc;

        if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;

        int v = vr * cols + vc;
        unsigned short dst_val = raster[v];
        if (dst_val == max_cost) continue;

        // Check intermediates
        float inter_cost = 0.0f;
        bool valid = true;
        int ni = s_n_inter[s];
        for (int k = 0; k < ni; k++) {
            int ir = ur + (int)s_inter_lut[(s * max_inter_cols + k) * 2];
            int ic = uc + (int)s_inter_lut[(s * max_inter_cols + k) * 2 + 1];
            if (ir < 0 || ir >= rows || ic < 0 || ic >= cols) {
                valid = false; break;
            }
            unsigned short ival = raster[ir * cols + ic];
            if (ival == max_cost) {
                valid = false; break;
            }
            inter_cost += (float)ival;
        }
        if (!valid) continue;

        float edge_weight = ((float)src_val + (float)dst_val + inter_cost)
                            * s_cost_factors[s];

        // Light phase: only edges with weight <= delta
        if (edge_weight > delta) continue;

        float new_dist = u_dist + edge_weight;

        // Atomic relaxation via IEEE 754 bit trick
        int new_dist_int = __float_as_int(new_dist);
        int old_dist_int = atomicMin((int*)&dist[v], new_dist_int);

        if (new_dist_int < old_dist_int) {
            if (pred != NULL) pred[v] = u;

            // Only append nodes whose new distance falls in current bucket
            if (new_dist >= bucket_low && new_dist < bucket_high) {
                int pos = atomicAdd(out_count, 1);
                out_queue[pos] = v;
            }
        }
    }
}
"""

_RELAX_HEAVY_V2_KERNEL = r"""
extern "C" __global__
void relax_heavy_v2(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps,
    const int max_inter_cols,
    const int rows,
    const int cols,
    const int max_cost,
    float*                __restrict__ dist,
    int*                  __restrict__ pred,
    const float           delta,
    const int*            __restrict__ settled,
    const int             settled_size,
    int*                  __restrict__ out_queue,
    int*                  __restrict__ out_count
) {
""" + _SMEM_LOAD_RELAX + r"""
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= settled_size) return;

    int u = settled[tid];
    int ur = u / cols;
    int uc = u - ur * cols;

    float u_dist = dist[u];
    if (u_dist >= 1e30f) return;

    unsigned short src_val = raster[u];
    if (src_val == max_cost) return;

    for (int s = 0; s < n_steps; s++) {
        int dr = (int)s_steps[s * 2];
        int dc = (int)s_steps[s * 2 + 1];
        int vr = ur + dr;
        int vc = uc + dc;

        if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;

        int v = vr * cols + vc;
        unsigned short dst_val = raster[v];
        if (dst_val == max_cost) continue;

        // Check intermediates
        float inter_cost = 0.0f;
        bool valid = true;
        int ni = s_n_inter[s];
        for (int k = 0; k < ni; k++) {
            int ir = ur + (int)s_inter_lut[(s * max_inter_cols + k) * 2];
            int ic = uc + (int)s_inter_lut[(s * max_inter_cols + k) * 2 + 1];
            if (ir < 0 || ir >= rows || ic < 0 || ic >= cols) {
                valid = false; break;
            }
            unsigned short ival = raster[ir * cols + ic];
            if (ival == max_cost) {
                valid = false; break;
            }
            inter_cost += (float)ival;
        }
        if (!valid) continue;

        float edge_weight = ((float)src_val + (float)dst_val + inter_cost)
                            * s_cost_factors[s];

        // Heavy phase: only edges with weight > delta
        if (edge_weight <= delta) continue;

        float new_dist = u_dist + edge_weight;

        int new_dist_int = __float_as_int(new_dist);
        int old_dist_int = atomicMin((int*)&dist[v], new_dist_int);

        if (new_dist_int < old_dist_int) {
            if (pred != NULL) pred[v] = u;
            // Append ALL updated nodes (no bucket filter)
            int pos = atomicAdd(out_count, 1);
            out_queue[pos] = v;
        }
    }
}
"""

_CLASSIFY_BUCKET_KERNEL = r"""
extern "C" __global__
void classify_bucket(
    const float* __restrict__ dist,
    const int*   __restrict__ input_queue,
    const int    input_count,
    const float  bucket_low,
    const float  bucket_high,
    int*         __restrict__ near_queue,
    int*         __restrict__ near_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= input_count) return;

    int v = input_queue[tid];
    float d = dist[v];
    if (d >= bucket_low && d < bucket_high) {
        int pos = atomicAdd(near_count, 1);
        near_queue[pos] = v;
    }
}
"""

# ============================================================================
# V3 CUDA kernels — eliminate per-bucket flatnonzero scan
#
# V3 builds on v2 by capturing ALL improved nodes (not just in-bucket ones).
# The light kernel writes to TWO output queues:
#   - in-bucket queue: nodes for the next light iteration (same as v2)
#   - deferred queue: nodes improved but outside current bucket (NEW)
#
# After heavy phase, deferred + heavy output are classified for the next
# bucket, replacing the O(n_pixels) flatnonzero scan with O(pending_size).
# ============================================================================

_RELAX_LIGHT_V3_KERNEL = r"""
extern "C" __global__
void relax_light_v3(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps,
    const int max_inter_cols,
    const int rows,
    const int cols,
    const int max_cost,
    float*                __restrict__ dist,
    int*                  __restrict__ pred,
    const float           delta,
    const int*            __restrict__ frontier,
    const int             frontier_size,
    int*                  __restrict__ out_inbucket,
    int*                  __restrict__ out_inbucket_count,
    int*                  __restrict__ out_deferred,
    int*                  __restrict__ out_deferred_count,
    const float           bucket_low,
    const float           bucket_high
) {
""" + _SMEM_LOAD_RELAX + r"""
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    int ur = u / cols;
    int uc = u - ur * cols;

    float u_dist = dist[u];
    if (u_dist >= 1e30f) return;

    unsigned short src_val = raster[u];
    if (src_val == max_cost) return;

    for (int s = 0; s < n_steps; s++) {
        int dr = (int)s_steps[s * 2];
        int dc = (int)s_steps[s * 2 + 1];
        int vr = ur + dr;
        int vc = uc + dc;

        if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;

        int v = vr * cols + vc;
        unsigned short dst_val = raster[v];
        if (dst_val == max_cost) continue;

        // Check intermediates
        float inter_cost = 0.0f;
        bool valid = true;
        int ni = s_n_inter[s];
        for (int k = 0; k < ni; k++) {
            int ir = ur + (int)s_inter_lut[(s * max_inter_cols + k) * 2];
            int ic = uc + (int)s_inter_lut[(s * max_inter_cols + k) * 2 + 1];
            if (ir < 0 || ir >= rows || ic < 0 || ic >= cols) {
                valid = false; break;
            }
            unsigned short ival = raster[ir * cols + ic];
            if (ival == max_cost) {
                valid = false; break;
            }
            inter_cost += (float)ival;
        }
        if (!valid) continue;

        float edge_weight = ((float)src_val + (float)dst_val + inter_cost)
                            * s_cost_factors[s];

        // Light phase: only edges with weight <= delta
        if (edge_weight > delta) continue;

        float new_dist = u_dist + edge_weight;

        // Atomic relaxation via IEEE 754 bit trick
        int new_dist_int = __float_as_int(new_dist);
        int old_dist_int = atomicMin((int*)&dist[v], new_dist_int);

        if (new_dist_int < old_dist_int) {
            if (pred != NULL) pred[v] = u;

            if (new_dist >= bucket_low && new_dist < bucket_high) {
                // In current bucket -> next light iteration
                int pos = atomicAdd(out_inbucket_count, 1);
                out_inbucket[pos] = v;
            } else {
                // Outside current bucket -> deferred for classify
                int pos = atomicAdd(out_deferred_count, 1);
                out_deferred[pos] = v;
            }
        }
    }
}
"""

_CLASSIFY_BUCKET_V2_KERNEL = r"""
extern "C" __global__
void classify_bucket_v2(
    const float* __restrict__ dist,
    const int*   __restrict__ input_queue,
    const int    input_count,
    const float  bucket_low,
    const float  bucket_high,
    int*         __restrict__ near_queue,
    int*         __restrict__ near_count,
    int*         __restrict__ far_queue,
    int*         __restrict__ far_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= input_count) return;

    int v = input_queue[tid];
    float d = dist[v];

    // Skip nodes already settled at better distance (below current bucket)
    // or unreachable
    if (d < bucket_low || d >= 1e30f) return;

    if (d < bucket_high) {
        int pos = atomicAdd(near_count, 1);
        near_queue[pos] = v;
    } else {
        int pos = atomicAdd(far_count, 1);
        far_queue[pos] = v;
    }
}
"""


# ============================================================================
# V4 CUDA kernel — Persistent cooperative kernel (Tier 3)
#
# Moves the ENTIRE delta-stepping loop onto the GPU as a single cooperative
# kernel launch. Uses grid.sync() (cooperative groups) instead of returning
# to Python between phases. This eliminates:
#   - ~30,000 GPU-to-CPU round-trips per SSSP call
#   - Python loop overhead (41% of v3 time)
#   - Queue management overhead (36% of v3 time)
#
# Requires: CUDA cooperative launch, CuPy enable_cooperative_groups=True
# Grid size limited to n_sms * (max_threads_per_sm / tpb) blocks.
# Each thread processes multiple frontier items via strided loops.
# ============================================================================

# Control buffer indices (single int32 array for all counters/flags)
_CTL_COUNT_A = 0       # frontier size in queue_a
_CTL_COUNT_B = 1       # output count in queue_b
_CTL_SETTLED = 2       # accumulated settled count
_CTL_PENDING = 3       # persistent pending count
_CTL_NEAR = 4          # classify near output count
_CTL_FAR = 5           # classify far output count
_CTL_BUCKET = 6        # current bucket index
_CTL_DONE = 7          # termination flag
_CTL_EARLY_CTR = 8     # early termination check counter
_CTL_MIN_DIST_INT = 9  # for fallback: min dist as int (IEEE 754)
_CTL_BARRIER_CNT = 10  # custom barrier: arrival counter
_CTL_BARRIER_SENSE = 11  # custom barrier: sense-reversing flag
_CTL_OVERFLOW = 12     # queue reservation exceeded buf_size -> self-heal
_CTL_TRUNCATED = 13    # light loop hit max_light_iters with work still queued
_CTL_SIZE = 14

_PERSISTENT_SSSP_KERNEL = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define CTL_COUNT_A 0
#define CTL_COUNT_B 1
#define CTL_SETTLED 2
#define CTL_PENDING 3
#define CTL_NEAR    4
#define CTL_FAR     5
#define CTL_BUCKET  6
#define CTL_DONE    7
#define CTL_EARLY_CTR 8
#define CTL_MIN_DIST 9
#define CTL_BARRIER_CNT 10
#define CTL_BARRIER_SENSE 11
#define CTL_OVERFLOW 12
#define CTL_TRUNCATED 13

// Queue-push helper: reservations past buf_size set the overflow flag
// instead of silently dropping the vertex. The dropped vertex keeps its
// (already atomicMin-updated) dist, and the end-of-phase self-heal path
// rebuilds the frontier from dist[] -- lossless, just slower on the rare
// overflow. Counts read back from control MUST be clamped to buf_size
// (reservations can exceed it) or the queue reads run out of bounds.
#define QPUSH(ctr, arr, v) do { \
    int _p = atomicAdd((int*)&control[(ctr)], 1); \
    if (_p < buf_size) (arr)[_p] = (v); \
    else control[CTL_OVERFLOW] = 1; \
} while (0)

// Custom atomic barrier replacing grid.sync().
// grid.sync() on Blackwell/sm_120 (CUDA 13.0) does not provide proper
// memory ordering between blocks — global writes made before grid.sync()
// by one block are not guaranteed visible to other blocks after grid.sync().
// This sense-reversing barrier with explicit __threadfence() fixes it.
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

extern "C" __global__
void delta_stepping_persistent(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const int rows, const int cols, const int max_cost,
    float* dist, int* pred,
    const float delta, const int window, const int fuse_depth,
    const int n_pixels, const int max_light_iters,
    const int* targets, const int n_targets, const float margin,
    volatile int* control, int* queue_a, int* queue_b,
    int* settled, int* pending, const int buf_size,
    // Per-edge gradient terms (feasibility plan section 3.2).
    // grad_n_bins == 0 disables everything (null-pointer fast path).
    const float* __restrict__ dem,
    const float* __restrict__ grad_lut,        // (n_bins*2) [mult, add]
    const float* __restrict__ grad_bin_factor, // (n_steps)
    const float* __restrict__ grad_step_len,   // (n_steps)
    const int grad_n_bins
) {
    int n_blocks = gridDim.x;
    extern __shared__ char smem[];
    int sb = n_steps * 2, sp = (sb + 3) & ~3;
    signed char* s_steps = (signed char*)smem;
    int* s_n_inter = (int*)(smem + sp);
    float* s_cost_factors = (float*)(s_n_inter + n_steps);
    signed char* s_inter_lut = (signed char*)(s_cost_factors + n_steps);
    for (int i = threadIdx.x; i < sb; i += blockDim.x) s_steps[i] = steps[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x) s_n_inter[i] = n_inter[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x) s_cost_factors[i] = cost_factors[i];
    int ls = n_steps * max_inter_cols * 2;
    for (int i = threadIdx.x; i < ls; i += blockDim.x) s_inter_lut[i] = inter_lut[i];
    // Gradient tables after the intermediate LUT (4-byte aligned).
    int lsp = (ls + 3) & ~3;
    float* s_grad_lut = (float*)(s_inter_lut + lsp);
    float* s_grad_bf  = s_grad_lut + 2 * grad_n_bins;
    float* s_grad_sl  = s_grad_bf + n_steps;
    if (grad_n_bins > 0) {
        for (int i = threadIdx.x; i < 2 * grad_n_bins; i += blockDim.x)
            s_grad_lut[i] = grad_lut[i];
        for (int i = threadIdx.x; i < n_steps; i += blockDim.x) {
            s_grad_bf[i] = grad_bin_factor[i];
            s_grad_sl[i] = grad_step_len[i];
        }
    }
    __syncthreads();
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int swap = 0;
    // Super-bucket window: 'window' consecutive base buckets are processed
    // as one phase (effective delta = span). window=1 == classic behavior.
    float span = delta * (float)window;

    while (true) {
        while (control[CTL_COUNT_A] == 0) {
            int pc = control[CTL_PENDING]; if (pc > buf_size) pc = buf_size;
            if (pc > 0) {
                if (gtid == 0) { control[CTL_NEAR] = 0; control[CTL_FAR] = 0; }
                grid_barrier(control, n_blocks);
                int bkt = control[CTL_BUCKET];
                float bl = bkt * delta, bh = (bkt + window) * delta;
                int* qa = swap ? queue_b : queue_a;
                for (int i = gtid; i < pc; i += stride) {
                    int v = pending[i]; float d = dist[v];
                    if (d < bl || d >= 1e30f) continue;
                    if (d < bh) QPUSH(CTL_NEAR, qa, v);
                    else QPUSH(CTL_FAR, settled, v);
                }
                grid_barrier(control, n_blocks);
                int fc = control[CTL_FAR]; int fc2 = fc < buf_size ? fc : buf_size;
                for (int i = gtid; i < fc2; i += stride) pending[i] = settled[i];
                if (gtid == 0) {
                    int nr = control[CTL_NEAR];
                    control[CTL_COUNT_A] = nr < buf_size ? nr : buf_size;
                    control[CTL_PENDING] = fc2;
                }
                grid_barrier(control, n_blocks);
                if (control[CTL_COUNT_A] > 0) break;
                if (control[CTL_PENDING] > 0) { if (gtid == 0) control[CTL_BUCKET] += window; grid_barrier(control, n_blocks); continue; }
            }
            if (gtid == 0) control[CTL_MIN_DIST] = __float_as_int(1e30f);
            grid_barrier(control, n_blocks);
            float blw = control[CTL_BUCKET] * delta;
            float lm = 1e30f;
            for (int i = gtid; i < n_pixels; i += stride) { float d = dist[i]; if (d >= blw && d < lm) lm = d; }
            if (lm < 1e30f) atomicMin((int*)&control[CTL_MIN_DIST], __float_as_int(lm));
            grid_barrier(control, n_blocks);
            float gm = __int_as_float(control[CTL_MIN_DIST]);
            if (gm >= 1e29f) { if (gtid == 0) control[CTL_DONE] = 1; grid_barrier(control, n_blocks); break; }
            int nb = (int)(gm / delta); float fl = nb * delta, fh = (nb + window) * delta;
            if (gtid == 0) { control[CTL_BUCKET] = nb; control[CTL_NEAR] = 0; }
            grid_barrier(control, n_blocks);
            int* qa = swap ? queue_b : queue_a;
            for (int i = gtid; i < n_pixels; i += stride) { float d = dist[i]; if (d >= fl && d < fh) QPUSH(CTL_NEAR, qa, i); }
            grid_barrier(control, n_blocks);
            if (gtid == 0) { int nr = control[CTL_NEAR]; control[CTL_COUNT_A] = nr < buf_size ? nr : buf_size; }
            grid_barrier(control, n_blocks); break;
        }
        if (control[CTL_DONE]) break;
        if (control[CTL_COUNT_A] == 0) break;

        int bkt = control[CTL_BUCKET];
        float blo = bkt * delta, bhi = (bkt + window) * delta;
        { int ca = control[CTL_COUNT_A]; int* qa = swap ? queue_b : queue_a;
          for (int i = gtid; i < ca && i < buf_size; i += stride) settled[i] = qa[i];
          if (gtid == 0) control[CTL_SETTLED] = ca < buf_size ? ca : buf_size; }
        grid_barrier(control, n_blocks);

        int drained = 0;
        for (int li = 0; li < max_light_iters; li++) {
            if (gtid == 0) control[CTL_COUNT_B] = 0;
            grid_barrier(control, n_blocks);
            int* qa = swap ? queue_b : queue_a;
            int* qb = swap ? queue_a : queue_b;
            int ca = control[CTL_COUNT_A];
            for (int idx = gtid; idx < ca; idx += stride) {
                int u = qa[idx];
                for (int hop = 0; ; hop++) {
                int chase = -1;
                int ur = u / cols, uc = u - ur * cols;
                float ud = dist[u]; if (ud >= 1e30f) break;
                unsigned short sv = raster[u]; if (sv == max_cost) break;
                float udem = (grad_n_bins > 0) ? dem[u] : 0.0f;
                for (int s = 0; s < n_steps; s++) {
                    int vr = ur + (int)s_steps[s*2], vc = uc + (int)s_steps[s*2+1];
                    if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;
                    int v = vr * cols + vc;
                    unsigned short dv = raster[v]; if (dv == max_cost) continue;
                    float ic = 0.0f; bool ok = true; int ni = s_n_inter[s];
                    for (int k = 0; k < ni; k++) {
                        int ir = ur + (int)s_inter_lut[(s*max_inter_cols+k)*2];
                        int icc = uc + (int)s_inter_lut[(s*max_inter_cols+k)*2+1];
                        if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) { ok = false; break; }
                        unsigned short iv = raster[ir*cols+icc]; if (iv == max_cost) { ok = false; break; }
                        ic += (float)iv;
                    }
                    if (!ok) continue;
                    float ew = ((float)sv + (float)dv + ic) * s_cost_factors[s];
                    if (grad_n_bins > 0) {
                        float dh = fabsf(dem[v] - udem);
                        int gb = (int)(dh * s_grad_bf[s]);
                        if (gb >= grad_n_bins) gb = grad_n_bins - 1;
                        float gmul = s_grad_lut[2*gb];
                        if (isinf(gmul)) continue;  // hard grade limit
                        ew = ew * gmul + s_grad_lut[2*gb+1] * s_grad_sl[s];
                    }
                    if (ew > span) continue;
                    float nd = ud + ew;
                    int ndi = __float_as_int(nd), odi = atomicMin((int*)&dist[v], ndi);
                    if (ndi < odi) {
                        if (pred != NULL) pred[v] = u;
                        if (nd >= blo && nd < bhi) {
                            QPUSH(CTL_COUNT_B, qb, v);
                            chase = v;
                        }
                        else QPUSH(CTL_PENDING, pending, v);
                    }
                }
                // Tail-chase fusion: follow one in-window improvement
                // immediately (bounded depth). The chased vertex was also
                // queued to qb, so correctness is a strict superset of the
                // classic behavior -- the chase only propagates improvements
                // deeper within this light iteration.
                if (chase < 0 || hop >= fuse_depth) break;
                u = chase;
                }
            }
            grid_barrier(control, n_blocks);
            int nc = control[CTL_COUNT_B]; if (nc > buf_size) nc = buf_size;
            if (nc == 0) { drained = 1; break; }
            int* qb2 = swap ? queue_a : queue_b;
            int os = control[CTL_SETTLED];
            for (int i = gtid; i < nc; i += stride) { int d = os + i; if (d < buf_size) settled[d] = qb2[i]; }
            if (gtid == 0) { int ns = os + nc; control[CTL_SETTLED] = ns < buf_size ? ns : buf_size; control[CTL_COUNT_A] = nc; }
            swap ^= 1; grid_barrier(control, n_blocks);
        }
        // Exhausting max_light_iters with a non-empty frontier means work was
        // abandoned: the remaining vertices keep whatever dist they had and
        // are never relaxed again, so the field comes back partial. On a
        // 0-cost region every edge weighs 0, so the whole raster stays in one
        // bucket and this fires after exactly max_light_iters rings - which is
        // how a 700x700 zero-cost raster returned 401^2 settled cells and inf
        // everywhere else. Record it; the launcher refuses to return the field.
        if (!drained && gtid == 0) control[CTL_TRUNCATED] = 1;

        if (gtid == 0) control[CTL_COUNT_B] = 0;
        grid_barrier(control, n_blocks);
        { int sc = control[CTL_SETTLED]; int* qb = swap ? queue_a : queue_b;
          for (int idx = gtid; idx < sc; idx += stride) {
              int u = settled[idx]; int ur = u / cols, uc = u - ur * cols;
              float ud = dist[u]; if (ud >= 1e30f) continue;
              unsigned short sv = raster[u]; if (sv == max_cost) continue;
              float udem = (grad_n_bins > 0) ? dem[u] : 0.0f;
              for (int s = 0; s < n_steps; s++) {
                  int vr = ur + (int)s_steps[s*2], vc = uc + (int)s_steps[s*2+1];
                  if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;
                  int v = vr * cols + vc;
                  unsigned short dv = raster[v]; if (dv == max_cost) continue;
                  float ic = 0.0f; bool ok = true; int ni = s_n_inter[s];
                  for (int k = 0; k < ni; k++) {
                      int ir = ur + (int)s_inter_lut[(s*max_inter_cols+k)*2];
                      int icc = uc + (int)s_inter_lut[(s*max_inter_cols+k)*2+1];
                      if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) { ok = false; break; }
                      unsigned short iv = raster[ir*cols+icc]; if (iv == max_cost) { ok = false; break; }
                      ic += (float)iv;
                  }
                  if (!ok) continue;
                  float ew = ((float)sv + (float)dv + ic) * s_cost_factors[s];
                  if (grad_n_bins > 0) {
                      float dh = fabsf(dem[v] - udem);
                      int gb = (int)(dh * s_grad_bf[s]);
                      if (gb >= grad_n_bins) gb = grad_n_bins - 1;
                      float gmul = s_grad_lut[2*gb];
                      if (isinf(gmul)) continue;  // hard grade limit
                      ew = ew * gmul + s_grad_lut[2*gb+1] * s_grad_sl[s];
                  }
                  if (ew <= span) continue;
                  float nd = ud + ew; int ndi = __float_as_int(nd), odi = atomicMin((int*)&dist[v], ndi);
                  if (ndi < odi) {
                      if (pred != NULL) pred[v] = u;
                      QPUSH(CTL_COUNT_B, qb, v);
                  }
              }
          }
        }
        grid_barrier(control, n_blocks);

        if (gtid == 0) control[CTL_BUCKET] += window;
        grid_barrier(control, n_blocks);
        int nxt = control[CTL_BUCKET]; float nl = nxt * delta, nh = (nxt + window) * delta;
        int hc = control[CTL_COUNT_B]; if (hc > buf_size) hc = buf_size;
        int pcc = control[CTL_PENDING]; if (pcc > buf_size) pcc = buf_size;
        int cmb = hc + pcc;
        if (cmb > 0) {
            int* qb = swap ? queue_a : queue_b;
            for (int i = gtid; i < hc && i < buf_size; i += stride) settled[i] = qb[i];
            for (int i = gtid; i < pcc; i += stride) { int d = hc + i; if (d < buf_size) settled[d] = pending[i]; }
            if (gtid == 0) { control[CTL_NEAR] = 0; control[CTL_FAR] = 0; }
            grid_barrier(control, n_blocks);
            int cl = cmb < buf_size ? cmb : buf_size;
            int* qao = swap ? queue_b : queue_a;
            for (int i = gtid; i < cl; i += stride) {
                int v = settled[i]; float d = dist[v];
                if (d < nl || d >= 1e30f) continue;
                if (d < nh) QPUSH(CTL_NEAR, qao, v);
                else QPUSH(CTL_FAR, pending, v);
            }
            grid_barrier(control, n_blocks);
            if (gtid == 0) {
                int nr = control[CTL_NEAR], fr = control[CTL_FAR];
                control[CTL_COUNT_A] = nr < buf_size ? nr : buf_size;
                control[CTL_PENDING] = fr < buf_size ? fr : buf_size;
            }
        } else {
            if (gtid == 0) { control[CTL_COUNT_A] = 0; control[CTL_PENDING] = 0; }
        }
        grid_barrier(control, n_blocks);

        // Self-heal after any queue overflow this phase: dropped vertices
        // kept their atomicMin-updated dist but left every queue, and may
        // lie below the advanced bucket base. Restart the frontier search
        // from this phase's base -- the fallback scan rebuilds the frontier
        // from dist[] (lossless; re-relaxing settled vertices in the span
        // produces no improvements, only bounded extra work).
        if (control[CTL_OVERFLOW]) {
            if (gtid == 0) {
                control[CTL_OVERFLOW] = 0;
                control[CTL_COUNT_A] = 0;
                control[CTL_PENDING] = 0;
                control[CTL_BUCKET] = bkt;
            }
            grid_barrier(control, n_blocks);
        }

        if (n_targets > 0) {
            if (gtid == 0) control[CTL_EARLY_CTR] += 1;
            grid_barrier(control, n_blocks);
            if (control[CTL_EARLY_CTR] % 10 == 0) {
                if (gtid == 0) control[CTL_MIN_DIST] = 0;
                grid_barrier(control, n_blocks);
                for (int i = gtid; i < n_targets; i += stride) {
                    int di = __float_as_int(dist[targets[i]]);
                    atomicMax((int*)&control[CTL_MIN_DIST], di);
                }
                grid_barrier(control, n_blocks);
                float mt = __int_as_float(control[CTL_MIN_DIST]);
                if (mt < 1e29f) {
                    float co = mt * margin;
                    if (control[CTL_BUCKET] * delta > co) {
                        if (gtid == 0) control[CTL_DONE] = 1;
                        grid_barrier(control, n_blocks);
                    }
                }
            }
        }

        if (control[CTL_DONE]) break;
    }
}
"""


# V5 control-word indices
_C5_DONE = 0
_C5_WORK = 1          # queued + in-flight items (quiescence == 0)
_C5_BASE = 2          # bucket index of the rolling span start
_C5_BARRIER_CNT = 3   # rendezvous barrier: arrival counter
_C5_BARRIER_SENSE = 4  # rendezvous barrier: sense-reversing flag
_C5_MIN_DIST = 5      # rescan scratch: min dist as int bits
_C5_OVERFLOW = 6      # a bucket ring overflowed -> rewind rescan
_C5_TGT_MAX = 7       # target-bound scratch: max target dist as int bits
_C5_SIZE = 8

_ASYNC_SSSP_KERNEL = r"""
#define C5_DONE 0
#define C5_WORK 1
#define C5_BASE 2
#define C5_BARRIER_CNT 3
#define C5_BARRIER_SENSE 4
#define C5_MIN_DIST 5
#define C5_OVERFLOW 6
#define C5_TGT_MAX 7

#define NB 32
#define EMPTY_SLOT (-1)

// Sense-reversing rendezvous barrier (same protocol as V4: grid.sync()
// lacks usable memory ordering on Blackwell; explicit __threadfence()).
__device__ void v5_barrier(volatile int* control, int n_blocks) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int my_sense = control[C5_BARRIER_SENSE];
        int arrived = atomicAdd((int*)&control[C5_BARRIER_CNT], 1) + 1;
        if (arrived == n_blocks) {
            control[C5_BARRIER_CNT] = 0;
            __threadfence();
            control[C5_BARRIER_SENSE] = 1 - my_sense;
        } else {
            while (control[C5_BARRIER_SENSE] == my_sense) {}
        }
    }
    __syncthreads();
}

// Reserve a ring slot for `v`, preferring bucket `bidx`.
//
// A full bucket spills into the next ring instead of dropping the push.
// That is sound because bucket order in this kernel is *advisory*: a
// vertex pulled from a later ring is still relaxed from its current
// dist[] value, and the label-correcting argument (atomicMin + re-queue)
// never depended on the order. Spilling turns the arena's capacity
// bound from "per bucket" into "over the whole ring set", which is what
// lets the arena be sized at NB*cap >= n_pixels: the span-boundary
// refill pushes at most one entry per pixel, so with spill it can never
// drop a vertex.
//
// That matters for more than memory. A refill that drops re-raises
// C5_OVERFLOW, and the resulting rewind rescans from the *same* span
// base and re-pushes the *same* oversubscribed bucket -- if a single
// delta-bucket holds more than the whole ring set can take (a 0-cost
// plateau, or any large region whose edges are tiny next to delta),
// the rendezvous makes no progress and the kernel spins forever. Every
// path that reserves a slot must go through here.
//
// Returns 1 when placed (the item is now claimable), 0 when every ring
// is full -- the caller then drops the push and raises C5_OVERFLOW,
// which the span-boundary rescan repairs losslessly (the dropped
// vertex kept its improved dist[], and the rewind rescans from
// base*delta, at or below which no push can land).
__device__ inline int v5_ring_push(volatile int* wres, int* arena,
                                   const int cap, const int bidx,
                                   const int v) {
    for (int t = 0; t < NB; t++) {
        int bb = bidx + t;
        if (bb >= NB) bb -= NB;
        // Reservations beyond cap are harmless: readers clamp wres to
        // cap, so nobody ever waits on a slot we did not write.
        int q = atomicAdd((int*)&wres[bb], 1);
        if (q < cap) {
            __threadfence();
            arena[bb * cap + q] = v;
            return 1;
        }
    }
    return 0;
}

// Asynchronous bucket-queue SSSP (ADDS-style, simplified for rasters).
//
// Hot path has NO grid-wide barriers: blocks claim chunks from a shared
// ring of NB delta-granular buckets covering the rolling span
// [base, base+NB)*delta, relax them, and push improvements with atomics
// only. Improvements landing outside the span only update dist[] (no
// queue entry); a span-boundary rescan rebuilds the frontier from dist[]
// -- lossless, because a vertex whose dist lies at/after the new span
// base was never relaxed-from at its current dist. Quiescence
// (work == 0 => nothing queued AND nobody mid-chunk) funnels every block
// into a rendezvous that advances the span, refills the ring, checks the
// target bound, or terminates. Bucket granularity stays FINE (delta, not
// a super-bucket): the 2026-08-06 sweep showed coarse priority windows
// amplify work on heavy-tail rasters; the async design removes barrier
// cost without giving up ordering.
extern "C" __global__
void sssp_async_v5(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const int rows, const int cols, const int max_cost,
    float* dist, int* pred,
    const float delta, const int n_pixels, const int chunk,
    const int* targets, const int n_targets, const float margin,
    volatile int* control,
    volatile int* wres,      // per-bucket write reservations [NB]
    volatile int* rres,      // per-bucket read reservations  [NB]
    int* arena,              // NB * cap item slots, EMPTY_SLOT-initialized
    const int cap,
    const float* __restrict__ dem,
    const float* __restrict__ grad_lut,
    const float* __restrict__ grad_bin_factor,
    const float* __restrict__ grad_step_len,
    const int grad_n_bins
) {
    int n_blocks = gridDim.x;
    extern __shared__ char smem[];
    int sb = n_steps * 2, sp = (sb + 3) & ~3;
    signed char* s_steps = (signed char*)smem;
    int* s_n_inter = (int*)(smem + sp);
    float* s_cost_factors = (float*)(s_n_inter + n_steps);
    signed char* s_inter_lut = (signed char*)(s_cost_factors + n_steps);
    for (int i = threadIdx.x; i < sb; i += blockDim.x) s_steps[i] = steps[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x) s_n_inter[i] = n_inter[i];
    for (int i = threadIdx.x; i < n_steps; i += blockDim.x) s_cost_factors[i] = cost_factors[i];
    int ls = n_steps * max_inter_cols * 2;
    for (int i = threadIdx.x; i < ls; i += blockDim.x) s_inter_lut[i] = inter_lut[i];
    int lsp = (ls + 3) & ~3;
    float* s_grad_lut = (float*)(s_inter_lut + lsp);
    float* s_grad_bf  = s_grad_lut + 2 * grad_n_bins;
    float* s_grad_sl  = s_grad_bf + n_steps;
    if (grad_n_bins > 0) {
        for (int i = threadIdx.x; i < 2 * grad_n_bins; i += blockDim.x)
            s_grad_lut[i] = grad_lut[i];
        for (int i = threadIdx.x; i < n_steps; i += blockDim.x) {
            s_grad_bf[i] = grad_bin_factor[i];
            s_grad_sl[i] = grad_step_len[i];
        }
    }
    __syncthreads();
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    __shared__ int sh_bucket, sh_start, sh_take;

    while (true) {
        if (control[C5_DONE]) break;

        // ---- claim a chunk from the lowest claimable bucket (leader) ----
        if (threadIdx.x == 0) {
            sh_take = 0;
            int base = control[C5_BASE];
            for (int b = 0; b < NB; b++) {
                for (;;) {
                    int rr = rres[b];
                    int wr = wres[b]; if (wr > cap) wr = cap;
                    int avail = wr - rr;
                    if (avail <= 0) break;
                    int take = avail < chunk ? avail : chunk;
                    if (atomicCAS((int*)&rres[b], rr, rr + take) == rr) {
                        sh_bucket = b; sh_start = rr; sh_take = take;
                        break;
                    }
                }
                if (sh_take > 0) break;
            }
            (void)base;
        }
        __syncthreads();
        int take = sh_take;

        if (take == 0) {
            // ---- no claimable work: quiescence check ----
            if (control[C5_WORK] != 0) {
            #if __CUDA_ARCH__ >= 700
                __nanosleep(200);
            #endif
                continue;
            }
            // work == 0: nothing queued, nobody mid-chunk (a processing
            // block keeps its chunk counted until fully done) -- every
            // block observes 0 and funnels into the rendezvous.
            v5_barrier(control, n_blocks);

            // -- span advance / termination (all blocks, barriered) --
            int base = control[C5_BASE];
            float span_end = (base + NB) * delta;
            // Overflow rewind: dropped pushes kept their dist inside the
            // current span -- rescan from the span base, not its end.
            float from_val = control[C5_OVERFLOW]
                           ? base * delta : span_end;
            if (base < 0) from_val = 0.0f;   // initial epoch
            if (gtid == 0) control[C5_MIN_DIST] = __float_as_int(1e30f);
            v5_barrier(control, n_blocks);
            float lm = 1e30f;
            for (int i = gtid; i < n_pixels; i += stride) {
                float d = dist[i];
                if (d >= from_val && d < lm) lm = d;
            }
            if (lm < 1e30f)
                atomicMin((int*)&control[C5_MIN_DIST], __float_as_int(lm));
            v5_barrier(control, n_blocks);
            float gm = __int_as_float(control[C5_MIN_DIST]);
            if (gm >= 1e29f) {
                if (gtid == 0) control[C5_DONE] = 1;
                v5_barrier(control, n_blocks);
                break;
            }
            int new_base = (int)(gm / delta);
            // target early exit: everything unprocessed is >= gm.
            // Scratch word MUST differ from C5_MIN_DIST: gm was published
            // through C5_MIN_DIST by the barrier above, and slower blocks
            // may not have read it yet -- zeroing it here would hand them
            // gm == 0 and diverge the blocks' barrier sequences (the
            // 2026-08-06 nondeterministic targeted-mode corruption).
            if (n_targets > 0) {
                if (gtid == 0) control[C5_TGT_MAX] = 0;
                v5_barrier(control, n_blocks);
                for (int i = gtid; i < n_targets; i += stride)
                    atomicMax((int*)&control[C5_TGT_MAX],
                              __float_as_int(dist[targets[i]]));
                v5_barrier(control, n_blocks);
                float mt = __int_as_float(control[C5_TGT_MAX]);
                if (mt < 1e29f && gm > mt * margin) {
                    if (gtid == 0) control[C5_DONE] = 1;
                    v5_barrier(control, n_blocks);
                    break;
                }
            }
            // -- reset ring, refill from dist[] --
            for (int b = gtid; b < NB; b += stride) { wres[b] = 0; rres[b] = 0; }
            if (gtid == 0) {
                control[C5_BASE] = new_base;
                control[C5_WORK] = 0;
                control[C5_OVERFLOW] = 0;
            }
            v5_barrier(control, n_blocks);
            float nbv = new_base * delta, nev = (new_base + NB) * delta;
            for (int i = gtid; i < n_pixels; i += stride) {
                float d = dist[i];
                if (d < nbv || d >= nev) continue;
                int bidx = (int)(d / delta) - new_base;
                if (bidx < 0) bidx = 0; if (bidx >= NB) bidx = NB - 1;
                atomicAdd((int*)&control[C5_WORK], 1);
                if (!v5_ring_push(wres, arena, cap, bidx, i)) {
                    atomicSub((int*)&control[C5_WORK], 1);
                    control[C5_OVERFLOW] = 1;
                }
            }
            v5_barrier(control, n_blocks);
            continue;
        }

        // ---- process the claimed chunk ----
        int b = sh_bucket, start = sh_start;
        int base = control[C5_BASE];
        float span_hi = (base + NB) * delta;
        for (int i = threadIdx.x; i < take; i += blockDim.x) {
            volatile int* slot = (volatile int*)&arena[b * cap + start + i];
            int u;
            while ((u = *slot) == EMPTY_SLOT) {}   // writer reserved: finite
            *slot = EMPTY_SLOT;
            int ur = u / cols, uc = u - ur * cols;
            float ud = dist[u]; if (ud >= 1e30f) continue;
            unsigned short sv = raster[u]; if (sv == max_cost) continue;
            float udem = (grad_n_bins > 0) ? dem[u] : 0.0f;
            for (int s = 0; s < n_steps; s++) {
                int vr = ur + (int)s_steps[s*2], vc = uc + (int)s_steps[s*2+1];
                if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) continue;
                int v = vr * cols + vc;
                unsigned short dv = raster[v]; if (dv == max_cost) continue;
                float ic = 0.0f; bool ok = true; int ni = s_n_inter[s];
                for (int k = 0; k < ni; k++) {
                    int ir = ur + (int)s_inter_lut[(s*max_inter_cols+k)*2];
                    int icc = uc + (int)s_inter_lut[(s*max_inter_cols+k)*2+1];
                    if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) { ok = false; break; }
                    unsigned short iv = raster[ir*cols+icc]; if (iv == max_cost) { ok = false; break; }
                    ic += (float)iv;
                }
                if (!ok) continue;
                float ew = ((float)sv + (float)dv + ic) * s_cost_factors[s];
                if (grad_n_bins > 0) {
                    float dh = fabsf(dem[v] - udem);
                    int gb = (int)(dh * s_grad_bf[s]);
                    if (gb >= grad_n_bins) gb = grad_n_bins - 1;
                    float gmul = s_grad_lut[2*gb];
                    if (isinf(gmul)) continue;
                    ew = ew * gmul + s_grad_lut[2*gb+1] * s_grad_sl[s];
                }
                float nd = ud + ew;
                // The pred store races concurrent better atomicMin winners
                // (a slower older winner's store can land last, leaving
                // pred pointing at the worse predecessor). Vertices
                // improved exactly once (0-cost plateaus) are race-free;
                // multiply-improved ones are validated and fixed by the
                // v5_repair_pred post-pass.
                int ndi = __float_as_int(nd), odi = atomicMin((int*)&dist[v], ndi);
                if (ndi < odi) {
                    if (pred != NULL) pred[v] = u;
                    if (nd < span_hi) {
                        int bidx = (int)(nd / delta) - base;
                        if (bidx < 0) bidx = 0; if (bidx >= NB) bidx = NB - 1;
                        // Count BEFORE reserving: once wres is advanced the
                        // item is claimable, and a consumer may fully process
                        // and decrement it -- if it were not yet counted,
                        // C5_WORK could read 0 with work still in flight
                        // (rendezvous deadlock).
                        atomicAdd((int*)&control[C5_WORK], 1);
                        if (!v5_ring_push(wres, arena, cap, bidx, v)) {
                            atomicSub((int*)&control[C5_WORK], 1);
                            control[C5_OVERFLOW] = 1;
                        }
                    }
                    // out-of-span: dist[] already updated; the span-boundary
                    // rescan recovers it -- no queue entry, no far list.
                }
            }
        }
        // chunk complete (including all its pushes): release the count
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicSub((int*)&control[C5_WORK], take);
    }
}
"""

# Post-pass predecessor validation/repair for V5. The hot-path pred store
# races concurrent better atomicMin winners: on a multiply-improved vertex
# a slower older winner's plain store can land last, leaving pred pointing
# at the worse predecessor. This pass keeps pred entries consistent with
# the final dist (|dist[pred] + ew - dist[v]| <= tol and dist[pred] <=
# dist[v]) and replaces inconsistent ones with the best strict-decrease
# incoming edge. Vertices improved exactly once -- notably 0-cost plateau
# members, whose only winning write is race-free -- validate and keep
# their arrival-order tree link. Matching uses a small relative tolerance:
# this kernel compiles separately from the relax kernel, so FMA
# contraction may differ by an ulp.
# The per-vertex decision, factored out so the full-raster repair and the
# path-local walk (Phase 1b item 1.4) run *identical* source. pred[v]
# depends only on the final dist[] (fixed) and on v's own incoming edges
# plus its own pre-repair pred[v] -- never on any other vertex's pred --
# which is what makes repairing one chain equivalent to repairing the
# whole raster, and makes the repair idempotent (a link chosen here
# validates on re-entry and is kept by the `u == cur` branch).
_V5_PRED_COMMON = r"""
__device__ inline int v5_best_pred(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const int rows, const int cols, const int max_cost,
    const float* __restrict__ dist,
    const float* __restrict__ dem,
    const float* __restrict__ grad_lut,
    const float* __restrict__ grad_bin_factor,
    const float* __restrict__ grad_step_len,
    const int grad_n_bins,
    const int v, const int cur
) {
    float target_d = dist[v];
    unsigned short dv = raster[v];
    if (dv == max_cost) return -1;
    int vr = v / cols, vc = v - vr * cols;
    float vdem = (grad_n_bins > 0) ? dem[v] : 0.0f;
    float tol = 1e-5f * fmaxf(target_d, 1.0f);
    int best = -1; float best_diff = 1e30f;
    for (int s = 0; s < n_steps; s++) {
        int ur = vr - (int)steps[s*2], uc = vc - (int)steps[s*2+1];
        if (ur < 0 || ur >= rows || uc < 0 || uc >= cols) continue;
        int u = ur * cols + uc;
        unsigned short sv = raster[u]; if (sv == max_cost) continue;
        float ud = dist[u]; if (ud >= 1e30f) continue;
        if (ud > target_d) continue;
        float ic = 0.0f; bool ok = true; int ni = n_inter[s];
        for (int k = 0; k < ni; k++) {
            int ir = ur + (int)inter_lut[(s*max_inter_cols+k)*2];
            int icc = uc + (int)inter_lut[(s*max_inter_cols+k)*2+1];
            if (ir < 0 || ir >= rows || icc < 0 || icc >= cols) { ok = false; break; }
            unsigned short iv = raster[ir*cols+icc]; if (iv == max_cost) { ok = false; break; }
            ic += (float)iv;
        }
        if (!ok) continue;
        float ew = ((float)sv + (float)dv + ic) * cost_factors[s];
        if (grad_n_bins > 0) {
            float dh = fabsf(vdem - dem[u]);
            int gb = (int)(dh * grad_bin_factor[s]);
            if (gb >= grad_n_bins) gb = grad_n_bins - 1;
            float gmul = grad_lut[2*gb];
            if (isinf(gmul)) continue;
            ew = ew * gmul + grad_lut[2*gb+1] * grad_step_len[s];
        }
        float diff = fabsf((ud + ew) - target_d);
        if (diff > tol) continue;
        if (u == cur) return u;  // existing link validates
        // prefer strict-decrease replacements (plateau links only
        // survive via validation of the race-free hot-path write)
        if (ud < target_d && diff < best_diff) {
            best = u; best_diff = diff;
        }
    }
    return best;
}
"""

_V5_PRED_REPAIR_KERNEL = _V5_PRED_COMMON + r"""
extern "C" __global__
void v5_repair_pred(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const int rows, const int cols, const int max_cost,
    const float* __restrict__ dist, int* pred,
    const int n_pixels, const int source_idx,
    const float* __restrict__ dem,
    const float* __restrict__ grad_lut,
    const float* __restrict__ grad_bin_factor,
    const float* __restrict__ grad_step_len,
    const int grad_n_bins
) {
    int v0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int v = v0; v < n_pixels; v += stride) {
        if (v == source_idx) { pred[v] = -1; continue; }
        float target_d = dist[v];
        if (target_d >= 1e30f) { pred[v] = -1; continue; }
        int cur = pred[v];
        pred[v] = v5_best_pred(
            raster, steps, cost_factors, inter_lut, n_inter,
            n_steps, max_inter_cols, rows, cols, max_cost,
            dist, dem, grad_lut, grad_bin_factor, grad_step_len,
            grad_n_bins, v, cur);
    }
}
"""

# On-device path extraction (Phase 1b items 1.3 + 1.4): one thread per
# target walks pred[] back to the source, optionally repairing each link
# it touches with exactly the arithmetic v5_repair_pred uses. Run in three
# launches: repair only, then count hops, then write the chains at the
# host-computed offsets. Repairing and counting must NOT share a launch --
# chains that merge are walked concurrently, so one thread can count a
# chain through a link another thread is repairing, and the counts then
# disagree with the write pass. Only the first launch repairs; the two
# after it read a pred field nobody is writing.
#
# Chains are written target-first; the host reverses them. lengths[t] is
# the hop count + 1, or -1 for "no path" (target out of range /
# unreachable / broken pred link / cycle) -- the same failure set the
# host-side pred walk in RasterGPUAPI._extract_path reports.
_V5_WALK_PRED_KERNEL = _V5_PRED_COMMON + r"""
extern "C" __global__
void v5_walk_pred(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const int rows, const int cols, const int max_cost,
    const float* __restrict__ dist, int* pred,
    const int n_pixels, const int source_idx,
    const float* __restrict__ dem,
    const float* __restrict__ grad_lut,
    const float* __restrict__ grad_bin_factor,
    const float* __restrict__ grad_step_len,
    const int grad_n_bins,
    const int* __restrict__ targets, const int n_targets,
    const int do_repair,
    int* __restrict__ lengths,
    const int* __restrict__ offsets,
    const int* __restrict__ limits,
    int* __restrict__ chains
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_targets) return;
    int wr = (chains != NULL) ? 1 : 0;
    // A target that failed the counting pass carries offset -1; never
    // write for it (its slice does not exist in the chain buffer).
    if (wr && offsets[t] < 0) { lengths[t] = -1; return; }
    int tgt = targets[t];
    if (tgt < 0 || tgt >= n_pixels) { lengths[t] = -1; return; }
    if (!(dist[tgt] < 1e30f)) { lengths[t] = -1; return; }
    int base = wr ? offsets[t] : 0;
    // Writes are hard-bounded by the length the counting pass measured,
    // so even if the two passes ever disagreed no thread could write
    // outside its own slice; the host compares the two length arrays
    // and refuses the result.
    int limit = wr ? limits[t] : n_pixels;
    int cnt = 0;
    int v = tgt;
    for (;;) {
        if (wr) {
            if (cnt >= limit) { lengths[t] = -1; return; }
            chains[base + cnt] = v;
        }
        cnt++;
        if (v == source_idx) break;
        if (cnt > n_pixels) { lengths[t] = -1; return; }  // cycle guard
        int p = pred[v];
        if (do_repair) {
            p = v5_best_pred(
                raster, steps, cost_factors, inter_lut, n_inter,
                n_steps, max_inter_cols, rows, cols, max_cost,
                dist, dem, grad_lut, grad_bin_factor, grad_step_len,
                grad_n_bins, v, p);
            pred[v] = p;
        }
        if (p < 0) { lengths[t] = -1; return; }
        v = p;
    }
    lengths[t] = cnt;
}
"""


# ============================================================================
# Kernel cache
# ============================================================================

_sssp_kernel_cache = {}


def _get_sssp_kernel(name, source, cooperative=False, variant="u16"):
    """Get a compiled CuPy RawKernel, compiling on first use."""
    key = (name, cooperative, variant)
    if key not in _sssp_kernel_cache:
        kwargs = {}
        if cooperative:
            kwargs["enable_cooperative_groups"] = True
            # -dlcm=cg: bypass L1 cache for global loads (use L2 only).
            # Required for persistent cooperative kernels because
            # grid.sync() does not invalidate L1 on all architectures
            # (confirmed on Blackwell/sm_120, CUDA 13.0). Without this,
            # stale dist[] reads from L1 cause suboptimal SSSP results.
            kwargs["options"] = (
                "--std=c++17", "-Xptxas", "-dlcm=cg",
            )
        _sssp_kernel_cache[key] = cp.RawKernel(source, name, **kwargs)
    return _sssp_kernel_cache[key]


def _float_raster_source(source):
    """Derive the float32-raster kernel variant from the uint16 source.

    Lossless weight_precision="float32" mode (feasibility plan Phase 9):
    the raster becomes ``const float*`` and the forbidden test becomes
    ``value >= 1e30f`` (combine() encodes forbidden as +inf) instead of
    the uint16 sentinel comparison. The ``(float)`` casts become no-ops;
    everything else (gradient terms, queues, atomics) is unchanged.
    """
    return (source
            .replace("const unsigned short* __restrict__ raster",
                     "const float* __restrict__ raster")
            .replace("unsigned short sv", "float sv")
            .replace("unsigned short dv", "float dv")
            .replace("unsigned short iv", "float iv")
            .replace("== max_cost", ">= 1e30f"))


def _compute_smem_bytes(n_steps: int, max_inter_cols: int) -> int:
    """Compute shared memory bytes for relaxation kernels."""
    steps_bytes = n_steps * 2
    steps_padded = (steps_bytes + 3) & ~3
    n_inter_bytes = n_steps * 4
    cost_factors_bytes = n_steps * 4
    lut_bytes = n_steps * max_inter_cols * 2
    return steps_padded + n_inter_bytes + cost_factors_bytes + lut_bytes


# ============================================================================
# Auto-delta computation (Tier 2)
# ============================================================================

def _compute_auto_delta(
        raster: np.ndarray,
        cost_factors: np.ndarray,
        ignore_max: bool = True,
) -> float:
    """Compute optimal delta from raster statistics.

    Uses mean edge weight as the bucket width, which balances bucket count
    (too many = iteration overhead) against redundant work (too few =
    oversized frontiers).

    Parameters:
        raster: 2D uint16 cost raster
        cost_factors: float32 array of cost normalization factors per step
        ignore_max: If True, treat 65535 as impassable (exclude from stats)

    Returns:
        Positive float delta value
    """
    if np.issubdtype(raster.dtype, np.floating):
        valid = raster.ravel()
        valid = valid[np.isfinite(valid) & (valid < 1e30)]
    else:
        max_val = (np.iinfo(np.uint16).max if ignore_max
                   else np.iinfo(np.uint16).max + 1)
        valid = raster.ravel()
        valid = valid[valid < max_val]
    if len(valid) == 0:
        return 100.0

    mean_cost = float(valid.mean())
    mean_cf = float(cost_factors.mean())

    # Mean edge weight ~ (src + dst) * cost_factor ~ 2 * mean_cost * mean_cf
    delta = 2.0 * mean_cost * mean_cf
    return max(1.0, delta)


# ============================================================================
# V4 availability check
# ============================================================================

_v4_available = None


def _check_v4_available():
    """Check if the v4 persistent cooperative kernel can be compiled.

    Caches the result so the compilation check only happens once.
    Returns True if cooperative groups are supported and the kernel compiles.
    """
    global _v4_available
    if _v4_available is not None:
        return _v4_available
    try:
        _ensure_cuda_path()
        kernel = _get_sssp_kernel(
            "delta_stepping_persistent",
            _PERSISTENT_SSSP_KERNEL,
            cooperative=True,
        )
        # Force compilation (RawKernel compiles lazily at first call).
        # Accessing .kernel triggers compilation without launching.
        _ = kernel.kernel
        _v4_available = True
    except Exception:
        _v4_available = False
    return _v4_available


_v5_available = None


def _check_v5_available():
    """Check if the v5 async bucket-queue kernel can be compiled.

    Caches the result so the compilation check only happens once.
    Returns True if cooperative groups are supported and the kernel compiles.
    """
    global _v5_available
    if _v5_available is not None:
        return _v5_available
    try:
        _ensure_cuda_path()
        kernel = _get_sssp_kernel(
            "sssp_async_v5",
            _ASYNC_SSSP_KERNEL,
            cooperative=True,
        )
        _ = kernel.kernel
        _v5_available = True
    except Exception:
        _v5_available = False
    return _v5_available


# ============================================================================
# Common GPU setup helpers
# ============================================================================

def _make_empty_result(n_pixels, return_predecessor):
    """Return an unreachable-everywhere result (inf distances, -1 predecessors)."""
    dist_out = np.full(n_pixels, np.inf, dtype=np.float32)
    if return_predecessor:
        return dist_out, np.full(n_pixels, -1, dtype=np.int32)
    return dist_out


def _validate_source(raster, source_idx, n_pixels, max_cost, return_predecessor):
    """Validate source index. Returns empty result if invalid, else None."""
    if source_idx < 0 or source_idx >= n_pixels:
        return _make_empty_result(n_pixels, return_predecessor)

    cols = raster.shape[1]
    source_r, source_c = divmod(int(source_idx), cols)
    value = raster[source_r, source_c]
    if np.issubdtype(raster.dtype, np.floating):
        if not np.isfinite(value) or value >= 1e30:
            return _make_empty_result(n_pixels, return_predecessor)
    elif value == max_cost:
        return _make_empty_result(n_pixels, return_predecessor)
    return None


def _resolve_delta(delta, raster, cost_factors, ignore_max):
    """Resolve delta from 'auto' or validate a numeric value."""
    if isinstance(delta, str) and delta == "auto":
        return _compute_auto_delta(raster, cost_factors, ignore_max)
    delta = float(delta)
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    return delta


def _upload_step_data(steps):
    """Upload step lookup tables to GPU, return (gpu_arrays, n_steps, smem_bytes)."""
    steps_arr, intermediates_lut, n_intermediates, cost_factors = \
        prepare_step_lookup_tables(steps)
    max_inter_cols = intermediates_lut.shape[1]
    smem_bytes = _compute_smem_bytes(len(steps), max_inter_cols)

    d_steps = cp.asarray(steps_arr.astype(np.int8))
    d_inter_lut = cp.asarray(intermediates_lut.astype(np.int8))
    d_n_inter = cp.asarray(n_intermediates.astype(np.int32))
    d_cost_factors = cp.asarray(cost_factors.astype(np.float32))

    return (d_steps, d_cost_factors, d_inter_lut, d_n_inter,
            cost_factors, max_inter_cols, smem_bytes)


def _prepare_gradient_gpu(dem, gradient_luts, raster_shape, n_steps,
                          max_inter_cols):
    """Upload the gradient inputs for the V4 kernel.

    Returns (dem_arg, lut_arg, bf_arg, sl_arg, n_bins, smem_extra,
    mean_mult). With gradient_luts None everything degenerates to null
    pointers, n_bins = 0 and mean_mult = 1.0 — the kernel's fast path.
    """
    if gradient_luts is None:
        null = np.intp(0)
        return null, null, null, null, 0, 0, 1.0
    if dem is None:
        raise ValueError("gradient_luts require a dem aligned to the raster")
    dem = np.ascontiguousarray(dem, dtype=np.float32)
    if dem.shape != raster_shape:
        raise ValueError(
            f"DEM shape {dem.shape} does not match the raster "
            f"{raster_shape} — align the DEM first.")
    if not np.isfinite(dem).all():
        raise ValueError(
            "DEM contains non-finite values — sanitize before the GPU "
            "((int)(NaN * x) is undefined on device).")

    d_dem = cp.asarray(dem)
    d_lut = cp.asarray(np.ascontiguousarray(
        gradient_luts.packed, dtype=np.float32).ravel())
    d_bf = cp.asarray(np.ascontiguousarray(
        gradient_luts.bin_factor, dtype=np.float32))
    d_sl = cp.asarray(np.ascontiguousarray(
        gradient_luts.step_len_cells, dtype=np.float32))
    n_bins = int(gradient_luts.n_bins)

    # Extra shared memory: 4-byte pad after the intermediate LUT, the
    # packed (mult, add) table and the two per-direction arrays.
    lut_bytes = n_steps * max_inter_cols * 2
    pad = ((lut_bytes + 3) & ~3) - lut_bytes
    smem_extra = pad + 8 * n_bins + 8 * n_steps

    # Delta scaling: with the multiplicative table active, the mean edge
    # weight grows by the mean multiplier over the DEM's actual slope
    # distribution (estimated from unit-step height differences).
    mean_mult = 1.0
    finite_mult = gradient_luts.mult[np.isfinite(gradient_luts.mult)]
    if finite_mult.size and finite_mult.max() > 1.0 + 1e-6:
        axis_d = int(np.argmin(gradient_luts.step_len_cells))
        dh = np.abs(np.diff(dem, axis=0)).ravel()
        if dh.size:
            bins = np.minimum(
                (dh * float(gradient_luts.bin_factor[axis_d])).astype(np.int64),
                n_bins - 1)
            mults = gradient_luts.mult[bins]
            mults = mults[np.isfinite(mults)]
            if mults.size:
                mean_mult = float(mults.mean())

    return d_dem, d_lut, d_bf, d_sl, n_bins, smem_extra, mean_mult


def _upload_raster(raster):
    """Upload the cost raster, casting only when the dtype demands it.

    ``astype`` copies unconditionally; the planning raster is already
    uint16 in every production path, so the copy was pure host traffic
    (50 MB at 25 M cells, 200 MB at 100 M). ``ascontiguousarray`` with an
    explicit dtype is ``astype(copy=False)`` plus the contiguity CuPy
    needs, and casts identically (same unsafe integer semantics) when the
    input really is a different dtype.
    """
    if np.issubdtype(raster.dtype, np.floating):
        # Lossless float32 mode (forbidden = +inf) — no uint16 cast.
        return cp.asarray(np.ascontiguousarray(raster, dtype=np.float32))
    return cp.asarray(np.ascontiguousarray(raster, dtype=np.uint16))


def _init_dist_pred(n_pixels, source_idx, return_predecessor):
    """Initialize distance and predecessor arrays on GPU."""
    d_dist = cp.full(n_pixels, np.float32(1e30), dtype=cp.float32)
    d_dist[source_idx] = np.float32(0.0)

    if return_predecessor:
        d_pred = cp.full(n_pixels, -1, dtype=cp.int32)
    else:
        d_pred = cp.ndarray(0, dtype=cp.int32)

    pred_arg = d_pred if return_predecessor else np.intp(0)
    return d_dist, d_pred, pred_arg


def _transfer_results(d_dist, d_pred, return_predecessor):
    """Transfer distance (and optionally predecessor) from GPU to CPU."""
    dist_out = d_dist.get()
    dist_out[dist_out >= 1e29] = np.inf
    if return_predecessor:
        return dist_out, d_pred.get()
    return dist_out


# ============================================================================
# Main entry point
# ============================================================================

def _alloc_v3_queues(n_pixels):
    """Allocate atomic-append frontier queues for v3 main loop."""
    return (
        cp.empty(n_pixels, dtype=cp.int32),   # queue_a (light frontier)
        cp.empty(n_pixels, dtype=cp.int32),   # queue_b (light/heavy output)
        cp.zeros(1, dtype=cp.int32),          # count_b
        cp.empty(n_pixels, dtype=cp.int32),   # settled
        cp.zeros(1, dtype=cp.int32),          # settled_count
        cp.empty(n_pixels, dtype=cp.int32),   # pending
        cp.zeros(1, dtype=cp.int32),          # pending_count
        cp.zeros(1, dtype=cp.int32),          # near_count
        cp.zeros(1, dtype=cp.int32),          # far_count
    )


def _v3_find_frontier(d_dist, d_pending, d_pending_count, d_queue_a,
                       d_settled, d_near_count, d_far_count,
                       classify_kernel, current_bucket, delta, tpb):
    """Find frontier for current bucket from pending queue or full scan."""
    frontier_size = 0
    while frontier_size == 0:
        pending_count = int(d_pending_count[0])
        if pending_count > 0:
            bucket_low = np.float32(current_bucket * delta)
            bucket_high = np.float32((current_bucket + 1) * delta)
            d_near_count[0] = 0
            d_far_count[0] = 0
            blocks = (pending_count + tpb - 1) // tpb
            classify_kernel(
                (blocks,), (tpb,),
                (d_dist, d_pending, np.int32(pending_count),
                 bucket_low, bucket_high,
                 d_queue_a, d_near_count, d_settled, d_far_count))
            frontier_size = int(d_near_count[0])
            far_count = int(d_far_count[0])
            if far_count > 0:
                d_pending[:far_count] = d_settled[:far_count]
            d_pending_count[0] = far_count
            if frontier_size > 0:
                break
            if far_count > 0:
                current_bucket += 1
                continue

        bucket_low = np.float32(current_bucket * delta)
        d_candidates = cp.where(
            d_dist >= bucket_low, d_dist, np.float32(1e30))
        min_remaining = float(d_candidates.min())
        if min_remaining >= 1e29:
            break
        current_bucket = int(min_remaining / delta)
        low = np.float32(current_bucket * delta)
        high = np.float32((current_bucket + 1) * delta)
        fallback = cp.flatnonzero(
            (d_dist >= low) & (d_dist < high)).astype(cp.int32)
        frontier_size = int(fallback.size)
        if frontier_size > 0:
            d_queue_a[:frontier_size] = fallback
        break
    return frontier_size, current_bucket


def _v3_light_phase(frontier_size, d_queue_a, d_queue_b, d_count_b,
                     d_settled, d_settled_count, d_pending, d_pending_count,
                     light_kernel, raster_args, bucket_low, bucket_high,
                     smem_bytes, tpb, max_light_iterations):
    """Run v3 light phase: relax light edges until frontier is empty."""
    d_settled[:frontier_size] = d_queue_a[:frontier_size]
    d_settled_count[0] = frontier_size
    for _ in range(max_light_iterations):
        d_count_b[0] = 0
        blocks = (frontier_size + tpb - 1) // tpb
        light_kernel(
            (blocks,), (tpb,),
            (*raster_args,
             d_queue_a, np.int32(frontier_size),
             d_queue_b, d_count_b,
             d_pending, d_pending_count,
             bucket_low, bucket_high),
            shared_mem=smem_bytes)
        frontier_size = int(d_count_b[0])
        if frontier_size == 0:
            break
        old_settled = int(d_settled_count[0])
        d_settled[old_settled:old_settled + frontier_size] = \
            d_queue_b[:frontier_size]
        d_settled_count[0] = old_settled + frontier_size
        d_queue_a, d_queue_b = d_queue_b, d_queue_a
    return d_queue_a, d_queue_b


def _v3_heavy_and_advance(d_settled, d_settled_count, d_queue_a, d_queue_b,
                           d_count_b, d_pending, d_pending_count,
                           d_near_count, d_far_count, d_dist, d_targets,
                           heavy_kernel, classify_kernel, raster_args,
                           smem_bytes, current_bucket, delta, margin,
                           early_term_counter, tpb):
    """Run heavy phase, classify next frontier, check early termination."""
    settled_size = int(d_settled_count[0])
    if settled_size > 0:
        d_count_b[0] = 0
        blocks = (settled_size + tpb - 1) // tpb
        heavy_kernel(
            (blocks,), (tpb,),
            (*raster_args,
             d_settled, np.int32(settled_size),
             d_queue_b, d_count_b),
            shared_mem=smem_bytes)

    current_bucket += 1
    next_low = np.float32(current_bucket * delta)
    next_high = np.float32((current_bucket + 1) * delta)
    heavy_count = int(d_count_b[0])
    pending_count = int(d_pending_count[0])
    combined_count = heavy_count + pending_count

    frontier_size = 0
    if combined_count > 0:
        offset = 0
        if heavy_count > 0:
            d_settled[:heavy_count] = d_queue_b[:heavy_count]
            offset = heavy_count
        if pending_count > 0:
            d_settled[offset:offset + pending_count] = \
                d_pending[:pending_count]
        d_near_count[0] = 0
        d_far_count[0] = 0
        blocks = (combined_count + tpb - 1) // tpb
        classify_kernel(
            (blocks,), (tpb,),
            (d_dist, d_settled, np.int32(combined_count),
             next_low, next_high,
             d_queue_a, d_near_count, d_pending, d_far_count))
        frontier_size = int(d_near_count[0])
        d_pending_count[0] = int(d_far_count[0])
    else:
        d_pending_count[0] = 0

    early_term_counter += 1
    done = False
    if d_targets is not None and early_term_counter % 10 == 0:
        max_target_dist = float(d_dist[d_targets].max())
        if max_target_dist < 1e29:
            if current_bucket * delta > max_target_dist * margin:
                done = True

    return frontier_size, current_bucket, early_term_counter, done


def _setup_v3(raster, steps, source_idx, delta, ignore_max,
              target_indices, return_predecessor):
    """Initialize GPU arrays, kernels, and queues for v3 loop."""
    rows, cols = raster.shape
    n_pixels = rows * cols
    n_steps = len(steps)
    max_cost = int(np.iinfo(np.uint16).max) if ignore_max else (
        int(np.iinfo(np.uint16).max) + 1)

    early_exit = _validate_source(
        raster, source_idx, n_pixels, max_cost, return_predecessor)
    if early_exit is not None:
        return early_exit, None

    (d_steps, d_cost_factors, d_inter_lut, d_n_inter,
     cost_factors, max_inter_cols, smem_bytes) = _upload_step_data(steps)
    delta = _resolve_delta(delta, raster, cost_factors, ignore_max)
    d_raster = cp.asarray(raster.astype(np.uint16))
    d_dist, d_pred, pred_arg = _init_dist_pred(
        n_pixels, source_idx, return_predecessor)

    d_targets = None
    if target_indices is not None and len(target_indices) > 0:
        d_targets = cp.asarray(np.asarray(target_indices, dtype=np.int32))

    kernels = (
        _get_sssp_kernel("relax_light_v3", _RELAX_LIGHT_V3_KERNEL),
        _get_sssp_kernel("relax_heavy_v2", _RELAX_HEAVY_V2_KERNEL),
        _get_sssp_kernel("classify_bucket_v2", _CLASSIFY_BUCKET_V2_KERNEL))

    raster_args = (d_raster, d_steps, d_cost_factors, d_inter_lut, d_n_inter,
                   np.int32(n_steps), np.int32(max_inter_cols),
                   np.int32(rows), np.int32(cols), np.int32(max_cost),
                   d_dist, pred_arg, np.float32(delta))

    return None, (d_dist, d_pred, d_targets, kernels, raster_args,
                  smem_bytes, delta, n_pixels, source_idx)


def _sssp_raster_gpu_v3(raster, steps, source_idx, delta, ignore_max,
                         target_indices, margin, return_predecessor,
                         max_light_iterations, threads_per_block):
    """V3 delta-stepping with atomic-append frontier queues."""
    early_exit, ctx = _setup_v3(
        raster, steps, source_idx, delta, ignore_max,
        target_indices, return_predecessor)
    if early_exit is not None:
        return early_exit

    (d_dist, d_pred, d_targets, kernels, raster_args,
     smem_bytes, delta, n_pixels, source_idx) = ctx
    light_kernel, heavy_kernel, classify_kernel = kernels

    queues = _alloc_v3_queues(n_pixels)
    (d_queue_a, d_queue_b, d_count_b, d_settled, d_settled_count,
     d_pending, d_pending_count, d_near_count, d_far_count) = queues

    tpb = threads_per_block
    d_queue_a[0] = source_idx
    frontier_size, current_bucket, early_term_counter = 1, 0, 0

    while True:
        frontier_size, current_bucket = _v3_find_frontier(
            d_dist, d_pending, d_pending_count, d_queue_a, d_settled,
            d_near_count, d_far_count, classify_kernel,
            current_bucket, delta, tpb)
        if frontier_size == 0:
            break
        bucket_low = np.float32(current_bucket * delta)
        bucket_high = np.float32((current_bucket + 1) * delta)
        d_queue_a, d_queue_b = _v3_light_phase(
            frontier_size, d_queue_a, d_queue_b, d_count_b,
            d_settled, d_settled_count, d_pending, d_pending_count,
            light_kernel, raster_args, bucket_low, bucket_high,
            smem_bytes, tpb, max_light_iterations)
        frontier_size, current_bucket, early_term_counter, done = \
            _v3_heavy_and_advance(
                d_settled, d_settled_count, d_queue_a, d_queue_b,
                d_count_b, d_pending, d_pending_count,
                d_near_count, d_far_count, d_dist, d_targets,
                heavy_kernel, classify_kernel, raster_args,
                smem_bytes, current_bucket, delta, margin,
                early_term_counter, tpb)
        if done:
            break

    return _transfer_results(d_dist, d_pred, return_predecessor)


def sssp_raster_gpu(
        raster: np.ndarray,
        steps: np.ndarray,
        source_idx: int,
        delta: Union[float, str] = "auto",
        ignore_max: bool = True,
        target_indices: Optional[np.ndarray] = None,
        margin: float = 1.00001,
        return_predecessor: bool = False,
        max_light_iterations: int = 100,
        threads_per_block: int = 256,
        dem: Optional[np.ndarray] = None,
        gradient_luts=None,
        session: Optional["GpuSsspSession"] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """GPU SSSP; uses the v5 async bucket-queue kernel, falling back to
    the v4 persistent kernel, then v3.

    V5 became the production default 2026-08-06 after its acceptance run:
    median 1.77x over V4-with-window-4 across random/heavy-tail at
    1000^2-3000^2, bit-exact dist arrays in every test and measurement
    (see benchmarks/FUSED_KERNEL_FINDINGS.md section 11).

    session: an open :class:`GpuSsspSession` reused across queries
        (Phase 1b item 1.2). V5-only -- a session cannot be built at all
        where V5 is unavailable, so there is no fallback to reach.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU not available. Install cupy with CUDA support: "
            "pip install cupy-cuda12x"
        )
    if session is not None or _check_v5_available():
        return sssp_raster_gpu_v5(
            raster, steps, source_idx, delta=delta,
            ignore_max=ignore_max, target_indices=target_indices,
            margin=margin, return_predecessor=return_predecessor,
            threads_per_block=threads_per_block,
            dem=dem, gradient_luts=gradient_luts, session=session)
    if _check_v4_available():
        return sssp_raster_gpu_v4(
            raster, steps, source_idx, delta=delta,
            ignore_max=ignore_max, target_indices=target_indices,
            margin=margin, return_predecessor=return_predecessor,
            max_light_iterations=max_light_iterations,
            threads_per_block=threads_per_block,
            dem=dem, gradient_luts=gradient_luts)
    if gradient_luts is not None:
        raise RuntimeError(
            "Per-edge gradient terms require the V4 persistent cooperative "
            "kernel, which is unavailable on this system (cooperative "
            "groups could not be compiled). Use the cython backend "
            "instead.")
    if np.issubdtype(np.asarray(raster).dtype, np.floating):
        raise RuntimeError(
            "Float32 weight rasters require the V4 persistent cooperative "
            "kernel, which is unavailable on this system. Use a library "
            "backend instead.")
    return _sssp_raster_gpu_v3(
        raster, steps, source_idx, delta, ignore_max,
        target_indices, margin, return_predecessor,
        max_light_iterations, threads_per_block)


# ============================================================================
# V4 entry point — Persistent cooperative kernel (Tier 3)
# ============================================================================

def _ensure_cuda_path():
    """Ensure CuPy can find cudadevrt for cooperative groups linking.

    CuPy caches the CUDA path at first access, so setting CUDA_PATH after
    import is too late. Instead, we find cudadevrt.lib and directly set
    CuPy's internal cached path variables.
    """
    import os
    import cupy.cuda.compiler as _compiler

    # Already resolved
    if getattr(_compiler, '_cudadevrt', None) is not None:
        return

    def _find_cudadevrt():
        """Search for cudadevrt.lib in pip-installed nvidia packages."""
        # Try pip-installed nvidia-cuda-runtime (namespace package)
        try:
            import nvidia.cuda_runtime as ncr
            for cuda_rt_dir in getattr(ncr, '__path__', []):
                devrt = os.path.join(cuda_rt_dir, "lib", "x64",
                                     "cudadevrt.lib")
                if os.path.isfile(devrt):
                    return cuda_rt_dir, devrt
        except ImportError:
            pass

        # Fallback: search in site-packages
        import site
        for sp in site.getsitepackages():
            cuda_rt_dir = os.path.join(sp, "nvidia", "cuda_runtime")
            devrt = os.path.join(cuda_rt_dir, "lib", "x64",
                                 "cudadevrt.lib")
            if os.path.isfile(devrt):
                return cuda_rt_dir, devrt

        return None, None

    cuda_rt_dir, devrt_path = _find_cudadevrt()
    if devrt_path is None:
        return

    # Set env var for any future CuPy path detection
    if not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = cuda_rt_dir

    # Reset CuPy's cached CUDA path so it picks up our CUDA_PATH
    import cupy._environment as _env
    _env._cuda_path = ''

    # Directly set the cudadevrt path in the compiler cache
    _compiler._cudadevrt = devrt_path


def _alloc_v4_buffers(n_pixels, n_steps, source_idx):
    """Allocate oversized frontier buffers and control array for v4."""
    buf_size = max(n_pixels * max(n_steps, 8), n_pixels * 4)
    d_queue_a = cp.empty(buf_size, dtype=cp.int32)
    d_queue_b = cp.empty(buf_size, dtype=cp.int32)
    d_settled = cp.empty(buf_size, dtype=cp.int32)
    d_pending = cp.empty(buf_size, dtype=cp.int32)
    d_control = cp.zeros(_CTL_SIZE, dtype=cp.int32)
    d_control[_CTL_COUNT_A] = 1
    d_queue_a[0] = source_idx
    return d_queue_a, d_queue_b, d_settled, d_pending, d_control, buf_size


def _prepare_v4_targets(target_indices):
    """Prepare target arrays for v4 kernel."""
    if target_indices is not None and len(target_indices) > 0:
        d_targets = cp.asarray(np.asarray(target_indices, dtype=np.int32))
        n_targets = len(target_indices)
        return d_targets, n_targets, d_targets
    d_targets = cp.ndarray(0, dtype=cp.int32)
    return d_targets, 0, np.intp(0)


def _common_gpu_setup(raster, steps, source_idx, delta, ignore_max,
                      return_predecessor):
    """Shared setup: validate source, upload data, init dist/pred."""
    rows, cols = raster.shape
    n_pixels = rows * cols
    # 65536 disables the sentinel: the kernels compare in int domain
    # (never cast max_cost to unsigned short — 65536 would truncate to 0
    # and make value-0 cells impassable).
    max_cost = int(np.iinfo(np.uint16).max) if ignore_max else (
        int(np.iinfo(np.uint16).max) + 1)
    early_exit = _validate_source(
        raster, source_idx, n_pixels, max_cost, return_predecessor)
    if early_exit is not None:
        return early_exit, None
    (d_steps, d_cost_factors, d_inter_lut, d_n_inter,
     cost_factors, max_inter_cols, smem_bytes) = _upload_step_data(steps)
    delta = _resolve_delta(delta, raster, cost_factors, ignore_max)
    d_raster = _upload_raster(raster)
    d_dist, d_pred, pred_arg = _init_dist_pred(
        n_pixels, source_idx, return_predecessor)
    return None, (d_raster, d_steps, d_cost_factors, d_inter_lut, d_n_inter,
                  max_inter_cols, smem_bytes, delta, d_dist, d_pred, pred_arg,
                  rows, cols, n_pixels, len(steps), max_cost)


def sssp_raster_gpu_v4(
        raster: np.ndarray,
        steps: np.ndarray,
        source_idx: int,
        delta: Union[float, str] = "auto",
        ignore_max: bool = True,
        target_indices: Optional[np.ndarray] = None,
        margin: float = 1.00001,
        return_predecessor: bool = False,
        max_light_iterations: int = 100,
        threads_per_block: int = 256,
        window: int = 4,
        fuse_depth: int = 0,
        dem: Optional[np.ndarray] = None,
        gradient_luts=None,
        blocks_per_sm: int = 2,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """V4 persistent cooperative kernel SSSP. Same API as sssp_raster_gpu.

    dem / gradient_luts: optional per-edge gradient terms (feasibility
        plan section 3.2) — a float32 DEM aligned to the raster plus a
        pyorps.core.objective.GradientLUTs slope-response pair.

    window: number of consecutive base buckets processed per phase
        (super-bucket / effective-delta lever; 1 = classic behavior).
        Default 4: benchmarked 1.15-1.80x faster than window=1 on every
        tested pattern/size (random & heavy-tail, 1000^2-3000^2), always
        bit-exact. window=8 is faster still on smooth cost distributions.
        Clamped to [1, 32] -- the validated range; larger windows can
        overflow the pending buffer on dense cheap terrain.
    fuse_depth: bounded in-thread tail-chase of in-window improvements
        (bucket-fusion lever; 0 = off -- benchmarked slower on GPU due to
        warp divergence; kept for experimentation).
    blocks_per_sm: resident blocks per SM for the cooperative launch.
        Clamped to [1, 2] -- at >2 blocks/SM the kernel produces
        incorrect results on Blackwell (root cause unknown, see
        gpu_optimization notes); 2 is the validated production value.
    """
    window = min(max(1, int(window)), 32)
    fuse_depth = max(0, int(fuse_depth))
    if not GPU_AVAILABLE:
        raise RuntimeError("CUDA GPU not available: pip install cupy-cuda12x")

    early_exit, ctx = _common_gpu_setup(
        raster, steps, source_idx, delta, ignore_max, return_predecessor)
    if early_exit is not None:
        return early_exit
    (d_raster, d_steps, d_cost_factors, d_inter_lut, d_n_inter,
     max_inter_cols, smem_bytes, delta, d_dist, d_pred, pred_arg,
     rows, cols, n_pixels, n_steps, max_cost) = ctx

    (dem_arg, grad_lut_arg, grad_bf_arg, grad_sl_arg, grad_n_bins,
     smem_extra, grad_mean_mult) = _prepare_gradient_gpu(
        dem, gradient_luts, (rows, cols), n_steps, max_inter_cols)
    smem_bytes += smem_extra
    # Keep the light/heavy balance sane under the multiplicative table.
    delta *= grad_mean_mult

    _, n_targets, targets_arg = _prepare_v4_targets(target_indices)
    d_qa, d_qb, d_settled, d_pending, d_ctl, buf_size = \
        _alloc_v4_buffers(n_pixels, n_steps, source_idx)

    _ensure_cuda_path()
    if np.issubdtype(np.asarray(raster).dtype, np.floating):
        kernel = _get_sssp_kernel(
            "delta_stepping_persistent",
            _float_raster_source(_PERSISTENT_SSSP_KERNEL),
            cooperative=True, variant="f32")
    else:
        kernel = _get_sssp_kernel(
            "delta_stepping_persistent", _PERSISTENT_SSSP_KERNEL,
            cooperative=True)
    tpb = min(max(32, int(threads_per_block)), 512)
    blocks_per_sm = min(max(1, int(blocks_per_sm)), 2)
    # Correctness boundary (2026-08-06 launch sweep): >512 resident threads
    # per SM produces wrong dist arrays (256x2 ok, 512x1 ok, 512x2 bad,
    # 256x3 bad -- the historical "Bug #3"). Root cause unknown; clamp.
    while blocks_per_sm > 1 and tpb * blocks_per_sm > 512:
        blocks_per_sm -= 1
    max_blocks = (cp.cuda.Device().attributes["MultiProcessorCount"]
                  * blocks_per_sm)
    kernel(
        (max_blocks,), (tpb,),
        (d_raster, d_steps, d_cost_factors, d_inter_lut, d_n_inter,
         np.int32(n_steps), np.int32(max_inter_cols),
         np.int32(rows), np.int32(cols), np.int32(max_cost),
         d_dist, pred_arg, np.float32(delta),
         np.int32(max(1, window)), np.int32(max(0, fuse_depth)),
         np.int32(n_pixels),
         # In-window light chains grow with the window span; the iteration
         # cap must scale with it or the cap-break drops leftover frontier
         # (latent V4 hazard, exposed by window > 8 on cheap dense terrain).
         np.int32(max_light_iterations * max(1, window)),
         targets_arg, np.int32(n_targets), np.float32(margin),
         d_ctl, d_qa, d_qb, d_settled, d_pending,
         np.int32(buf_size),
         dem_arg, grad_lut_arg, grad_bf_arg, grad_sl_arg,
         np.int32(grad_n_bins)),
        shared_mem=smem_bytes)
    cp.cuda.Stream.null.synchronize()
    if int(d_ctl[_CTL_TRUNCATED]):
        # Returning here would hand back a partial distance field with no
        # indication anything was lost: unreached cells read as inf, which is
        # indistinguishable from genuinely unreachable. Measured on a 700x700
        # zero-cost raster, V4 returned 401^2 settled cells and inf for the
        # other 329,199 - a wrong answer delivered silently. Fail instead.
        raise RuntimeError(
            f"V4 abandoned part of the frontier: the light phase hit its "
            f"{max_light_iterations * max(1, window)}-iteration cap with work "
            f"still queued, so the returned distances would be partial. This "
            f"happens when a single delta-bucket cannot drain - typically a "
            f"large 0-cost region, where every edge weighs 0 and the whole "
            f"raster stays in one bucket. Use the V5 kernel (the default), or "
            f"raise max_light_iterations, or increase delta.")
    return _transfer_results(d_dist, d_pred, return_predecessor)


# ============================================================================
# V5 entry point — Asynchronous bucket queue (ADDS-style)
# ============================================================================

_V5_N_BUCKETS = 32

#: Ring-arena size as entries per pixel across the whole 32-ring set.
#:
#: Measured 2026-08-10 (benchmarks/benchmark_arena_factor.py, 2000^2 and
#: 3000^2, random / heavy-tail / 0-cost-plateau, median of 5, field identical
#: in all 18 cases) against the pre-Phase-1b sizing of 3.0:
#:   2.0 -> 0.97-1.01x   free within noise, arena 12 -> 8 B/cell
#:   1.0 -> 0.93-1.05x   up to 5.4% slower on heavy-tail, arena -> 4 B/cell
#: So 2.0 is the default: it halves the arena at no measurable cost. Pass
#: arena_factor=1.0 explicitly when VRAM matters more than a few percent
#: (V5 total 18 -> 14 B/cell, i.e. 1.4 GB instead of 1.8 GB at 100M cells).
_V5_DEFAULT_ARENA_FACTOR = 2.0


def _v5_arena_cap(n_pixels: int,
                  arena_factor: float = _V5_DEFAULT_ARENA_FACTOR) -> int:
    """Per-bucket ring capacity for the V5 arena (Phase 1b item 1.11).

    The arena used to be ``max(4096, 3*n_pixels//32)`` per ring, i.e.
    **12 B/cell** -- over half of V5's 22 B/cell device footprint -- while
    a real frontier is O(perimeter), not O(area).

    Sizing is expressed as a fraction of one entry per pixel over the
    whole 32-ring set, because that is the bound that actually matters:
    the span-boundary refill pushes **at most one entry per pixel**, so
    with ``arena_factor >= 1.0`` and the spilling ring push
    (``v5_ring_push``) the refill can never drop a vertex. That is not a
    memory nicety -- a refill drop re-raises C5_OVERFLOW, and the rewind
    then rescans the *same* span base and re-pushes the *same*
    oversubscribed bucket, so the rendezvous makes no progress. (A
    0-cost plateau larger than the old ``3n/32`` cap reaches that state
    on today's sizing too.)

    Hot-path pushes are still allowed to overflow -- an improvement
    landing on a full ring set drops its queue entry, keeps its
    atomicMin'd dist[], raises C5_OVERFLOW and is recovered by the
    rewind rescan. That costs one extra rescan and is bounded: a drop
    can only follow a strict dist improvement, of which there are
    finitely many. It stays rare because a span holds a frontier band,
    not the raster.

    Returns ``max(4096, ceil(arena_factor * n_pixels / 32))``: 4 B/cell
    at the default factor 1.0 (V5 total 22 -> 14 B/cell), and unchanged
    behaviour below ~131 k pixels where the 4096 floor already exceeds
    one entry per pixel.
    """
    arena_factor = float(arena_factor)
    if arena_factor < 1.0:
        # Below 1.0 the ring set can hold fewer entries than the
        # span-boundary refill pushes, so the refill can drop -- and a
        # refill drop is what makes the rewind re-push the same
        # oversubscribed bucket forever. The no-drop guarantee this whole
        # sizing rests on is only valid at >= 1.0.
        raise ValueError(
            f"arena_factor must be >= 1.0 (the span-boundary refill pushes "
            f"up to one entry per pixel and must never drop), got "
            f"{arena_factor}")
    total = int(np.ceil(arena_factor * float(int(n_pixels))))
    per_bucket = -(-total // _V5_N_BUCKETS)
    return max(4096, int(per_bucket))


class GpuSsspSession:
    """Device-resident state for repeated V5 solves on one raster.

    Phase 1b item 1.2. Every ``sssp_raster_gpu*`` call re-did the whole
    host setup: an unconditional ``astype`` copy of the raster, a PCIe
    upload, a full-raster host mask+mean for auto-delta, and a fresh
    dist/pred/arena allocation. For a k-pair job on one raster, k-1 of
    those are waste (~70-200 ms per query at 25 M cells, ~360-1100 ms at
    100 M). A session pays them once.

    There is deliberately **no module-level cache**: this is a 6 GB card
    the user shares, so the lifetime is the caller's. Hold one for the
    length of a routing job and :meth:`close` it (or use it as a context
    manager).

    Device footprint, steady state (uint16 raster, default
    ``arena_factor``):

    ==========  =========  ======================================
    buffer      B/cell     freed by
    ==========  =========  ======================================
    raster      2 (4 f32)  ``close()``
    dist        4          ``close()``
    pred        4          ``close()`` (allocated on first request)
    arena       4          ``close()``
    DEM         4          ``close()`` (only with gradient routing)
    ==========  =========  ======================================

    Per-call scratch (targets, chain buffers) is O(targets + path) and
    is released when the call returns.
    """

    def __init__(
            self,
            raster: np.ndarray,
            steps: np.ndarray,
            *,
            ignore_max: bool = True,
            delta: Union[float, str] = "auto",
            dem: Optional[np.ndarray] = None,
            gradient_luts=None,
            threads_per_block: int = 256,
            blocks_per_sm: int = 2,
            chunk: int = 256,
            arena_factor: float = _V5_DEFAULT_ARENA_FACTOR,
    ):
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "CUDA GPU not available: pip install cupy-cuda12x")
        if not _check_v5_available():
            raise RuntimeError(
                "GpuSsspSession requires the V5 async kernel, which could "
                "not be compiled on this system (cooperative groups). Call "
                "sssp_raster_gpu() without a session to fall back to V4/V3.")

        raster = np.asarray(raster)
        if raster.ndim != 2:
            raise ValueError(f"raster must be 2D, got shape {raster.shape}")
        rows, cols = int(raster.shape[0]), int(raster.shape[1])
        self.shape = (rows, cols)
        self.n_pixels = rows * cols
        self.ignore_max = bool(ignore_max)
        self._float_raster = bool(np.issubdtype(raster.dtype, np.floating))
        # 65536 disables the sentinel: the kernels compare in int domain
        # (never cast max_cost to unsigned short -- 65536 truncates to 0
        # and would make value-0 cells impassable).
        self._max_cost = (int(np.iinfo(np.uint16).max) if self.ignore_max
                          else int(np.iinfo(np.uint16).max) + 1)
        # Kept only for the source-cell validation that _validate_source
        # does on the host; no copy is made.
        self._raster_host = raster

        (d_steps, d_cost_factors, d_inter_lut, d_n_inter,
         cost_factors, max_inter_cols, smem_bytes) = _upload_step_data(steps)
        self._d_steps = d_steps
        self._d_cost_factors = d_cost_factors
        self._d_inter_lut = d_inter_lut
        self._d_n_inter = d_n_inter
        self.n_steps = int(len(steps))
        self._max_inter_cols = int(max_inter_cols)

        # Auto-delta: one full-raster host pass, once per session.
        base_delta = _resolve_delta(delta, raster, cost_factors, ignore_max)

        self._d_raster = _upload_raster(raster)

        (dem_arg, grad_lut_arg, grad_bf_arg, grad_sl_arg, grad_n_bins,
         smem_extra, grad_mean_mult) = _prepare_gradient_gpu(
            dem, gradient_luts, self.shape, self.n_steps, max_inter_cols)
        self._dem_arg = dem_arg
        self._grad_lut_arg = grad_lut_arg
        self._grad_bf_arg = grad_bf_arg
        self._grad_sl_arg = grad_sl_arg
        self._grad_n_bins = int(grad_n_bins)
        self.smem_bytes = int(smem_bytes + smem_extra)
        #: Resolved bucket width actually handed to the kernel (auto-delta
        #: scaled by the gradient table's mean multiplier).
        self.delta = float(base_delta * grad_mean_mult)

        self.chunk = max(1, int(chunk))
        tpb = min(max(32, int(threads_per_block)), 512)
        bps = min(max(1, int(blocks_per_sm)), 2)
        # >512 resident threads/SM produces wrong dist arrays on
        # Blackwell (historical "Bug #3") -- same clamp as V4/V5.
        while bps > 1 and tpb * bps > 512:
            bps -= 1
        self.threads_per_block = tpb
        self.blocks_per_sm = bps
        _ensure_cuda_path()
        self._max_blocks = int(
            cp.cuda.Device().attributes["MultiProcessorCount"] * bps)

        variant_src = (_float_raster_source if self._float_raster
                       else (lambda s: s))
        variant = "f32" if self._float_raster else "u16"
        self._kernel = _get_sssp_kernel(
            "sssp_async_v5", variant_src(_ASYNC_SSSP_KERNEL),
            cooperative=True, variant=variant)
        self._repair_kernel = _get_sssp_kernel(
            "v5_repair_pred", variant_src(_V5_PRED_REPAIR_KERNEL),
            variant=variant)
        self._walk_kernel = _get_sssp_kernel(
            "v5_walk_pred", variant_src(_V5_WALK_PRED_KERNEL),
            variant=variant)

        self.arena_factor = float(arena_factor)
        self.cap = _v5_arena_cap(self.n_pixels, self.arena_factor)
        self._d_dist = cp.empty(self.n_pixels, dtype=cp.float32)
        self._d_pred = None            # allocated on first pred request
        self._d_arena = cp.empty(_V5_N_BUCKETS * self.cap, dtype=cp.int32)
        self._d_wres = cp.empty(_V5_N_BUCKETS, dtype=cp.int32)
        self._d_rres = cp.empty(_V5_N_BUCKETS, dtype=cp.int32)
        self._d_ctl = cp.empty(_C5_SIZE, dtype=cp.int32)
        self._pred_source = None       # source the device pred belongs to
        self._closed = False

    # -- lifecycle ---------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def device_bytes(self) -> int:
        """Bytes of device memory this session holds resident."""
        total = 0
        for buf in (self._d_raster, self._d_dist, self._d_pred,
                    self._d_arena, self._d_wres, self._d_rres, self._d_ctl,
                    self._d_steps, self._d_cost_factors, self._d_inter_lut,
                    self._d_n_inter, self._dem_arg, self._grad_lut_arg,
                    self._grad_bf_arg, self._grad_sl_arg):
            # np.intp(0) stands in for a null device pointer and has a
            # nbytes of its own — only real device arrays count.
            if isinstance(buf, cp.ndarray):
                total += int(buf.nbytes)
        return total

    def close(self, free_pool: bool = False) -> None:
        """Release every device buffer this session holds. Idempotent.

        CuPy returns freed blocks to its memory pool, so the driver still
        counts them against this process. Pass ``free_pool=True`` to hand
        them back to the driver as well -- the polite thing to do on a
        shared 6 GB card once a routing job is finished.
        """
        self._d_raster = None
        self._d_dist = None
        self._d_pred = None
        self._d_arena = None
        self._d_wres = None
        self._d_rres = None
        self._d_ctl = None
        self._d_steps = None
        self._d_cost_factors = None
        self._d_inter_lut = None
        self._d_n_inter = None
        self._dem_arg = None
        self._grad_lut_arg = None
        self._grad_bf_arg = None
        self._grad_sl_arg = None
        self._raster_host = None
        self._pred_source = None
        self._closed = True
        if free_pool:
            cp.get_default_memory_pool().free_all_blocks()

    def __enter__(self) -> "GpuSsspSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def check_raster(self, raster: np.ndarray) -> None:
        """Raise unless ``raster`` matches the one this session uploaded.

        Only the shape and the float/integer kind are compared: comparing
        values would cost exactly the O(window) pass a session exists to
        avoid. Mutating the raster behind a live session is the caller's
        responsibility -- rebuild the session instead.
        """
        raster = np.asarray(raster)
        if raster.shape != self.shape:
            raise ValueError(
                f"session was built for a {self.shape} raster, got "
                f"{raster.shape}; build a new GpuSsspSession")
        if bool(np.issubdtype(raster.dtype, np.floating)) != self._float_raster:
            raise ValueError(
                "session raster dtype kind differs (float32 lossless mode "
                "vs uint16); build a new GpuSsspSession")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("GpuSsspSession is closed")

    # -- solving -----------------------------------------------------

    def _reset_state(self, need_pred: bool) -> None:
        """Restore exactly the state a fresh allocation would have."""
        self._d_dist.fill(np.float32(1e30))
        if need_pred and self._d_pred is None:
            self._d_pred = cp.empty(self.n_pixels, dtype=cp.int32)
        if self._d_pred is not None:
            self._d_pred.fill(-1)
        # The arena drains to EMPTY_SLOT at a normal termination (every
        # reserved slot < cap is claimed and cleared before quiescence),
        # but re-filling is ~4 B/cell of device writes -- cheap next to
        # the solve, and it removes a subtle cross-query dependency.
        self._d_arena.fill(-1)
        self._d_wres.fill(0)
        self._d_rres.fill(0)
        self._d_ctl.fill(0)
        # base = -NB makes the first rendezvous rescan start at 0, where
        # it finds the source -- no host-side frontier setup needed.
        self._d_ctl[_C5_BASE] = -_V5_N_BUCKETS

    def solve(
            self,
            source_idx: int,
            target_indices: Optional[np.ndarray] = None,
            *,
            margin: float = 1.00001,
            return_predecessor: bool = False,
            download: bool = True,
            full_repair: bool = True,
    ):
        """Run one V5 solve on the session's raster.

        Identical arithmetic to :func:`sssp_raster_gpu_v5` without a
        session -- same resolved delta, same launch geometry, same
        kernels; only the host setup and the allocations are reused.

        download=True (default) returns ``dist`` (or ``(dist, pred)``)
        exactly as the module-level entry points do. download=False
        returns None and leaves the field on the device for
        :meth:`extract_paths` -- that is the O(path) route (item 1.3).

        full_repair: run the full-raster ``v5_repair_pred`` post-pass
        over all n_pixels vertices. Required for a *downloaded* pred
        array; set False when only paths are wanted, and let
        :meth:`extract_paths` repair the ~10 k links on the chains
        instead (item 1.4).
        """
        self._check_open()
        source_idx = int(source_idx)
        need_pred = bool(return_predecessor) or not download
        self._reset_state(need_pred)
        self._pred_source = None

        early = _validate_source(
            self._raster_host, source_idx, self.n_pixels, self._max_cost,
            bool(return_predecessor))
        if early is not None:
            # Unreachable-everywhere: device state is already all-1e30 /
            # all -1, so extract_paths reports "no path" for every target.
            self._pred_source = source_idx if need_pred else None
            if not download:
                return None
            return early

        self._d_dist[source_idx] = np.float32(0.0)
        _, n_targets, targets_arg = _prepare_v4_targets(target_indices)
        pred_arg = self._d_pred if need_pred else np.intp(0)

        self._kernel(
            (self._max_blocks,), (self.threads_per_block,),
            (self._d_raster, self._d_steps, self._d_cost_factors,
             self._d_inter_lut, self._d_n_inter,
             np.int32(self.n_steps), np.int32(self._max_inter_cols),
             np.int32(self.shape[0]), np.int32(self.shape[1]),
             np.int32(self._max_cost),
             self._d_dist, pred_arg, np.float32(self.delta),
             np.int32(self.n_pixels), np.int32(self.chunk),
             targets_arg, np.int32(n_targets), np.float32(margin),
             self._d_ctl, self._d_wres, self._d_rres, self._d_arena,
             np.int32(self.cap),
             self._dem_arg, self._grad_lut_arg, self._grad_bf_arg,
             self._grad_sl_arg, np.int32(self._grad_n_bins)),
            shared_mem=self.smem_bytes)

        if need_pred:
            self._pred_source = source_idx
            if full_repair:
                self._run_full_repair(source_idx)
        cp.cuda.Stream.null.synchronize()
        if not download:
            return None
        return _transfer_results(
            self._d_dist, self._d_pred, bool(return_predecessor))

    def _run_full_repair(self, source_idx: int) -> None:
        """Validate/repair every pred entry against the final dist.

        The async hot path can leave a raced pred entry pointing at a
        stale predecessor; this is the O(window x dirs) post-pass that
        fixes all of them. Only full-field consumers need it.
        """
        tpb = self.threads_per_block
        blocks = min(4096, (self.n_pixels + tpb - 1) // tpb)
        self._repair_kernel(
            (blocks,), (tpb,),
            (self._d_raster, self._d_steps, self._d_cost_factors,
             self._d_inter_lut, self._d_n_inter,
             np.int32(self.n_steps), np.int32(self._max_inter_cols),
             np.int32(self.shape[0]), np.int32(self.shape[1]),
             np.int32(self._max_cost),
             self._d_dist, self._d_pred, np.int32(self.n_pixels),
             np.int32(source_idx),
             self._dem_arg, self._grad_lut_arg, self._grad_bf_arg,
             self._grad_sl_arg, np.int32(self._grad_n_bins)))

    # -- O(path) extraction ------------------------------------------

    def extract_paths(
            self,
            source_idx: int,
            target_indices,
            *,
            repair: bool = True,
    ) -> Tuple[list, np.ndarray]:
        """Walk pred on the device and download only the chains.

        Items 1.3 + 1.4. Operates on whatever the last :meth:`solve`
        left on the device -- it does not solve. Downloads
        ``sum(path lengths) + 2*n_targets`` int32 plus one float32 per
        target, instead of the full dist+pred pair (191 MiB at 5000^2,
        763 MiB at 10000^2).

        repair: repair each pred link the walk touches with exactly the
            arithmetic ``v5_repair_pred`` uses (shared device function
            ``v5_best_pred``), so the chain is the one the full repair
            would have produced. Pass False only when :meth:`solve` was
            called with ``full_repair=True``.

        Returns ``(paths, costs)``:
            paths: list, one entry per target in the given order --
                an ``np.ndarray(dtype=int32)`` of flat raster indices
                ordered **source -> target** (both inclusive), or
                ``None`` where no path exists (unreachable target,
                broken pred chain, target out of range). ``.tolist()``
                gives the plain list ``RasterGPUAPI._extract_path``
                returns today.
            costs: ``np.ndarray(dtype=float32)`` of ``dist`` at each
                target with the ``>= 1e29`` sentinel mapped to ``inf``,
                i.e. bit-identical to the full-download values at those
                indices.
        """
        self._check_open()
        source_idx = int(source_idx)
        if self._d_pred is None or self._pred_source != source_idx:
            raise RuntimeError(
                "no predecessor field on the device for source "
                f"{source_idx}; call solve(..., download=False) or "
                "solve(..., return_predecessor=True) first")

        targets = np.ascontiguousarray(
            np.asarray(target_indices, dtype=np.int32).ravel())
        n_targets = int(targets.size)
        if n_targets == 0:
            return [], np.empty(0, dtype=np.float32)

        d_targets = cp.asarray(targets)
        d_lengths = cp.empty(n_targets, dtype=cp.int32)
        tpb = 64
        blocks = (n_targets + tpb - 1) // tpb
        null = np.intp(0)

        # Three launches, not two. Repairing and counting in the SAME launch
        # is a race: chains that merge are walked concurrently, so thread B
        # can count a chain through pred[v] = x while thread A repairs that
        # link to y. The counts then disagree with the write pass and the
        # reproducibility check below fires on a perfectly legal query. Kernel
        # launch boundaries give the grid-wide ordering that fixes it, and a
        # launch over n_targets threads costs microseconds.
        if repair:
            # Converge pred to its fixed point. The repair is idempotent (a
            # link that already satisfies the tolerance takes the `u == cur`
            # early-accept), so the counting pass below re-reads it unchanged.
            self._walk_kernel(
                (blocks,), (tpb,),
                self._walk_args(source_idx, d_targets, n_targets, 1,
                                d_lengths, null, null, null))
        # Count hops against a pred field nobody is writing any more.
        self._walk_kernel(
            (blocks,), (tpb,),
            self._walk_args(source_idx, d_targets, n_targets, 0,
                            d_lengths, null, null, null))
        lengths = d_lengths.get()

        ok = lengths > 0
        offsets = np.full(n_targets, -1, dtype=np.int32)
        total = int(lengths[ok].sum()) if ok.any() else 0
        if total:
            offsets[ok] = (np.cumsum(lengths[ok]) - lengths[ok]).astype(
                np.int32)
        # Gather dist only at in-range targets: CuPy's fancy indexing
        # does not bounds-check, and an out-of-range target is a
        # "no path" answer, not a device read.
        in_range = (targets >= 0) & (targets < self.n_pixels)
        costs = np.full(n_targets, np.inf, dtype=np.float32)
        if in_range.any():
            gathered = self._d_dist[
                cp.asarray(np.ascontiguousarray(targets[in_range]))].get()
            gathered[gathered >= 1e29] = np.inf
            costs[in_range] = gathered

        paths = [None] * n_targets
        if total:
            d_offsets = cp.asarray(offsets)
            d_limits = cp.asarray(np.maximum(lengths, 0).astype(np.int32))
            d_chains = cp.empty(total, dtype=cp.int32)
            d_lengths2 = cp.empty(n_targets, dtype=cp.int32)
            # Pass 2: write the chains. repair=0 -- pass 1 already fixed
            # every link on them, so this walk sees the same vertices and
            # writes exactly lengths[t] entries into its own slice.
            self._walk_kernel(
                (blocks,), (tpb,),
                self._walk_args(source_idx, d_targets, n_targets, 0,
                                d_lengths2, d_offsets, d_limits, d_chains))
            lengths2 = d_lengths2.get()
            if not np.array_equal(lengths2, lengths):
                raise RuntimeError(
                    "on-device pred walk was not reproducible between the "
                    "counting and writing passes; the predecessor field "
                    "changed underneath it")
            chains = d_chains.get()
            for t in range(n_targets):
                if not ok[t]:
                    continue
                off = int(offsets[t])
                ln = int(lengths[t])
                # written target-first; hand back source -> target
                paths[t] = np.ascontiguousarray(
                    chains[off:off + ln][::-1])
        return paths, costs

    def _walk_args(self, source_idx, d_targets, n_targets, do_repair,
                   d_lengths, offsets_arg, limits_arg, chains_arg):
        return (
            self._d_raster, self._d_steps, self._d_cost_factors,
            self._d_inter_lut, self._d_n_inter,
            np.int32(self.n_steps), np.int32(self._max_inter_cols),
            np.int32(self.shape[0]), np.int32(self.shape[1]),
            np.int32(self._max_cost),
            self._d_dist, self._d_pred, np.int32(self.n_pixels),
            np.int32(source_idx),
            self._dem_arg, self._grad_lut_arg, self._grad_bf_arg,
            self._grad_sl_arg, np.int32(self._grad_n_bins),
            d_targets, np.int32(n_targets), np.int32(do_repair),
            d_lengths, offsets_arg, limits_arg, chains_arg,
        )

    def solve_paths(
            self,
            source_idx: int,
            target_indices,
            *,
            margin: float = 1.00001,
    ) -> Tuple[list, np.ndarray]:
        """Solve, then return only the target paths and their costs.

        The O(path) query: no full-raster download (item 1.3) and no
        full-raster predecessor repair (item 1.4). Same return shape as
        :meth:`extract_paths`.
        """
        self.solve(source_idx, target_indices=target_indices,
                   margin=margin, return_predecessor=True,
                   download=False, full_repair=False)
        return self.extract_paths(source_idx, target_indices, repair=True)


def sssp_raster_gpu_v5(
        raster: np.ndarray,
        steps: np.ndarray,
        source_idx: int,
        delta: Union[float, str] = "auto",
        ignore_max: bool = True,
        target_indices: Optional[np.ndarray] = None,
        margin: float = 1.00001,
        return_predecessor: bool = False,
        threads_per_block: int = 256,
        blocks_per_sm: int = 2,
        chunk: int = 256,
        dem: Optional[np.ndarray] = None,
        gradient_luts=None,
        session: Optional["GpuSsspSession"] = None,
        arena_factor: float = _V5_DEFAULT_ARENA_FACTOR,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """V5 asynchronous bucket-queue SSSP. Same API as sssp_raster_gpu.

    Barrier-free hot path (open-levers plan section 2): blocks pull work
    chunks from a shared ring of 32 delta-granular buckets over a rolling
    span and relax with atomics only; grid-wide synchronization happens
    only at span-boundary rendezvous (quiescence-triggered). Exactness by
    the same argument as delta-stepping: atomicMin + re-queue is
    label-correcting; bucket order is advisory.

    chunk: items a block claims per grab. Smaller values follow small
        frontiers at lower latency; larger values amortize claim
        overhead on wide frontiers. 256 benchmarked best across
        random/heavy-tail x 1000^2-3000^2 (64/128/512/1024 all slower).

    session: an open :class:`GpuSsspSession` holding the device-resident
        raster/DEM/step tables and the reusable dist/pred/arena buffers
        (Phase 1b item 1.2). When given, every other configuration
        argument (delta, ignore_max, dem, gradient_luts, chunk,
        threads_per_block, blocks_per_sm, arena_factor) comes from the
        session and the values passed here are ignored; only `raster`'s
        shape is checked against it. When omitted a session is built for
        this call and freed on return -- i.e. exactly the pre-1.2
        behaviour.
    arena_factor: ring-arena sizing (see :func:`_v5_arena_cap`).
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("CUDA GPU not available: pip install cupy-cuda12x")

    own_session = session is None
    if own_session:
        session = GpuSsspSession(
            raster, steps, ignore_max=ignore_max, delta=delta, dem=dem,
            gradient_luts=gradient_luts,
            threads_per_block=threads_per_block,
            blocks_per_sm=blocks_per_sm, chunk=chunk,
            arena_factor=arena_factor)
    else:
        session.check_raster(raster)
    try:
        return session.solve(
            source_idx, target_indices=target_indices, margin=margin,
            return_predecessor=return_predecessor)
    finally:
        if own_session:
            session.close()


def sssp_raster_gpu_paths(
        raster: np.ndarray,
        steps: np.ndarray,
        source_idx: int,
        target_indices: np.ndarray,
        delta: Union[float, str] = "auto",
        ignore_max: bool = True,
        margin: float = 1.00001,
        threads_per_block: int = 256,
        blocks_per_sm: int = 2,
        chunk: int = 256,
        dem: Optional[np.ndarray] = None,
        gradient_luts=None,
        session: Optional["GpuSsspSession"] = None,
        arena_factor: float = _V5_DEFAULT_ARENA_FACTOR,
) -> Tuple[list, np.ndarray]:
    """Solve and return only the target paths and their costs (item 1.3).

    The O(path) counterpart of :func:`sssp_raster_gpu`: nothing
    proportional to the raster crosses PCIe. Returns
    ``(paths, costs)`` -- see :meth:`GpuSsspSession.extract_paths`.

    Requires V5 (the on-device walk lives in the V5 kernel family). For
    the full cost surface use :func:`sssp_raster_gpu` instead.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("CUDA GPU not available: pip install cupy-cuda12x")
    own_session = session is None
    if own_session:
        session = GpuSsspSession(
            raster, steps, ignore_max=ignore_max, delta=delta, dem=dem,
            gradient_luts=gradient_luts,
            threads_per_block=threads_per_block,
            blocks_per_sm=blocks_per_sm, chunk=chunk,
            arena_factor=arena_factor)
    else:
        session.check_raster(raster)
    try:
        return session.solve_paths(
            source_idx, target_indices, margin=margin)
    finally:
        if own_session:
            session.close()

"""
GPU-accelerated constrained SSSP for infrastructure-aware routing.

Extends the raster-direct GPU delta-stepping approach to handle
extended state (cell, direction, span_bin) for coupled route + tower
placement optimization.

State packing: state = cell * n_dirs * n_span_bins + dir * n_span_bins + span_bin
All state indices are int64 to handle large state spaces.

Algorithm:
    Delta-stepping with frontier queues. Each iteration relaxes all edges
    from the frontier. Two relaxation options per edge:
    - Option 1: Continue without placing a tower (span accumulates)
    - Option 2: Place a tower at current cell (span resets, tower cost added)

    Path reconstruction and tower detection run on CPU after kernel completion.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Tuple

from pyorps.utils.traversal_gpu import (
    GPU_AVAILABLE, prepare_step_lookup_tables
)

if GPU_AVAILABLE:
    import cupy as cp


# ============================================================================
# CUDA kernel — constrained relaxation (all edges in single pass)
# ============================================================================

_CONSTRAINED_RELAX_KERNEL = r"""
extern "C" __global__
void constrained_relax(
    const unsigned short* __restrict__ raster,
    const int rows, const int cols, const int max_cost,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const float*          __restrict__ angle_cost_lut,
    const unsigned char*  __restrict__ angle_valid_lut,
    const float*          __restrict__ step_distances,
    const float*          __restrict__ tower_terrain_costs,
    const float*          __restrict__ tower_angle_costs,
    const int n_span_bins,
    const float span_bin_size,
    const int min_span_bin,
    float*                __restrict__ dist,
    long long*            __restrict__ pred,
    const long long*      __restrict__ frontier,
    const int frontier_size,
    long long*            __restrict__ queue_out,
    int*                  __restrict__ count_out,
    long long*            __restrict__ pending_out,
    int*                  __restrict__ pending_count_out,
    const float bucket_low,
    const float bucket_high,
    const int buf_size
) {
    // Load step data into shared memory
    extern __shared__ char smem[];
    int steps_bytes = n_steps * 2;
    int steps_padded = (steps_bytes + 3) & ~3;
    signed char* s_steps = (signed char*)smem;
    int* s_n_inter = (int*)(smem + steps_padded);
    float* s_cost_factors = (float*)(s_n_inter + n_steps);
    float* s_step_dist = s_cost_factors + n_steps;
    // angle_cost and angle_valid after step_distances
    float* s_angle_cost = s_step_dist + n_steps;
    unsigned char* s_angle_valid = (unsigned char*)(s_angle_cost
                                                    + n_steps * n_steps);
    float* s_tower_angle = (float*)(s_angle_valid + n_steps * n_steps);
    signed char* s_inter_lut = (signed char*)(s_tower_angle
                                              + n_steps * n_steps);

    int tid_local = threadIdx.x;
    int bsz = blockDim.x;
    for (int i = tid_local; i < steps_bytes; i += bsz) s_steps[i] = steps[i];
    for (int i = tid_local; i < n_steps; i += bsz) s_n_inter[i] = n_inter[i];
    for (int i = tid_local; i < n_steps; i += bsz)
        s_cost_factors[i] = cost_factors[i];
    for (int i = tid_local; i < n_steps; i += bsz)
        s_step_dist[i] = step_distances[i];
    int n2 = n_steps * n_steps;
    for (int i = tid_local; i < n2; i += bsz)
        s_angle_cost[i] = angle_cost_lut[i];
    for (int i = tid_local; i < n2; i += bsz)
        s_angle_valid[i] = angle_valid_lut[i];
    for (int i = tid_local; i < n2; i += bsz)
        s_tower_angle[i] = tower_angle_costs[i];
    int lut_size = n_steps * max_inter_cols * 2;
    for (int i = tid_local; i < lut_size; i += bsz)
        s_inter_lut[i] = inter_lut[i];
    __syncthreads();

    long long states_per_cell = (long long)n_steps * n_span_bins;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = gtid; idx < frontier_size; idx += stride) {
        long long cur_state = frontier[idx];
        float cur_dist = dist[cur_state];
        if (cur_dist >= 1e30f) continue;

        // Unpack state
        int cur_cell = (int)(cur_state / states_per_cell);
        int rem = (int)(cur_state % states_per_cell);
        int cur_dir = rem / n_span_bins;
        int cur_span_bin = rem % n_span_bins;

        int cur_row = cur_cell / cols;
        int cur_col = cur_cell - cur_row * cols;
        unsigned short sv = raster[cur_cell];
        if (sv == (unsigned short)max_cost) continue;

        for (int d_out = 0; d_out < n_steps; d_out++) {
            // Angle validity
            if (s_angle_valid[cur_dir * n_steps + d_out] == 0) continue;

            int dr = (int)s_steps[d_out * 2];
            int dc = (int)s_steps[d_out * 2 + 1];
            int nr = cur_row + dr;
            int nc = cur_col + dc;
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

            int nb_cell = nr * cols + nc;
            unsigned short dv = raster[nb_cell];
            if (dv == (unsigned short)max_cost) continue;

            // Intermediate cell validation
            float ic = 0.0f;
            bool ok = true;
            int ni = s_n_inter[d_out];
            for (int k = 0; k < ni; k++) {
                int ir = cur_row
                    + (int)s_inter_lut[(d_out * max_inter_cols + k) * 2];
                int icc = cur_col
                    + (int)s_inter_lut[(d_out * max_inter_cols + k) * 2 + 1];
                if (ir < 0 || ir >= rows || icc < 0 || icc >= cols)
                    { ok = false; break; }
                unsigned short iv = raster[ir * cols + icc];
                if (iv == (unsigned short)max_cost) { ok = false; break; }
                ic += (float)iv;
            }
            if (!ok) continue;

            // Edge cost: terrain + angle penalty
            float terrain_cost = ((float)sv + (float)dv + ic)
                                 * s_cost_factors[d_out];
            float angle_penalty = s_angle_cost[cur_dir * n_steps + d_out];
            float edge_cost = terrain_cost + angle_penalty;

            float new_span_m = (float)cur_span_bin * span_bin_size
                              + s_step_dist[d_out];
            int new_span_bin = (int)(new_span_m / span_bin_size);

            // Option 1: Continue without placing a tower
            if (new_span_bin < n_span_bins) {
                long long new_state = (long long)nb_cell * states_per_cell
                                    + (long long)d_out * n_span_bins
                                    + new_span_bin;
                float new_dist = cur_dist + edge_cost;
                int ndi = __float_as_int(new_dist);
                int odi = atomicMin((int*)&dist[new_state], ndi);
                if (ndi < odi) {
                    pred[new_state] = cur_state;
                    if (new_dist >= bucket_low && new_dist < bucket_high) {
                        int p = atomicAdd(count_out, 1);
                        if (p < buf_size) queue_out[p] = new_state;
                    } else {
                        int p = atomicAdd(pending_count_out, 1);
                        if (p < buf_size) pending_out[p] = new_state;
                    }
                }
            }

            // Option 2: Place tower at current cell, reset span
            if (n_span_bins > 1 && cur_span_bin >= min_span_bin) {
                float tower_cost = tower_terrain_costs[sv]
                    + s_tower_angle[cur_dir * n_steps + d_out];
                int reset_span = (int)(s_step_dist[d_out] / span_bin_size);
                if (reset_span < n_span_bins) {
                    long long new_state = (long long)nb_cell * states_per_cell
                                        + (long long)d_out * n_span_bins
                                        + reset_span;
                    float new_dist = cur_dist + edge_cost + tower_cost;
                    int ndi = __float_as_int(new_dist);
                    int odi = atomicMin((int*)&dist[new_state], ndi);
                    if (ndi < odi) {
                        pred[new_state] = cur_state;
                        if (new_dist >= bucket_low && new_dist < bucket_high) {
                            int p = atomicAdd(count_out, 1);
                            if (p < buf_size) queue_out[p] = new_state;
                        } else {
                            int p = atomicAdd(pending_count_out, 1);
                            if (p < buf_size) pending_out[p] = new_state;
                        }
                    }
                }
            }
        }
    }
}
"""

# Classify pending states into near/far buckets
_CLASSIFY_KERNEL = r"""
extern "C" __global__
void classify_constrained(
    const float*          __restrict__ dist,
    const long long*      __restrict__ candidates,
    const int             n_candidates,
    const float           bucket_low,
    const float           bucket_high,
    long long*            __restrict__ near_out,
    int*                  __restrict__ near_count,
    long long*            __restrict__ far_out,
    int*                  __restrict__ far_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n_candidates; i += stride) {
        long long state = candidates[i];
        float d = dist[state];
        if (d < bucket_low || d >= 1e30f) continue;
        if (d < bucket_high) {
            int p = atomicAdd(near_count, 1);
            near_out[p] = state;
        } else {
            int p = atomicAdd(far_count, 1);
            far_out[p] = state;
        }
    }
}
"""


# ============================================================================
# Kernel cache and shared memory
# ============================================================================

_constrained_kernel_cache = {}


def _get_constrained_kernel(name, source):
    """Get a compiled CuPy RawKernel, compiling on first use."""
    if name not in _constrained_kernel_cache:
        _constrained_kernel_cache[name] = cp.RawKernel(source, name)
    return _constrained_kernel_cache[name]


def _compute_constrained_smem(n_steps, max_inter_cols):
    """Compute shared memory bytes for constrained relaxation kernel."""
    steps_bytes = n_steps * 2
    steps_padded = (steps_bytes + 3) & ~3
    n_inter_bytes = n_steps * 4           # int32
    cost_factors_bytes = n_steps * 4      # float32
    step_dist_bytes = n_steps * 4         # float32
    angle_cost_bytes = n_steps * n_steps * 4   # float32
    angle_valid_bytes = n_steps * n_steps       # uint8
    tower_angle_bytes = n_steps * n_steps * 4   # float32
    lut_bytes = n_steps * max_inter_cols * 2    # int8
    # Align angle_valid to 4 bytes for the float array after it
    angle_valid_padded = (angle_valid_bytes + 3) & ~3
    return (steps_padded + n_inter_bytes + cost_factors_bytes
            + step_dist_bytes + angle_cost_bytes + angle_valid_padded
            + tower_angle_bytes + lut_bytes)


# ============================================================================
# VRAM check
# ============================================================================

def _check_vram(total_states, n_steps, buf_size):
    """Check if GPU has enough VRAM for the constrained SSSP arrays."""
    dist_bytes = total_states * 4           # float32
    pred_bytes = total_states * 8           # int64
    # 5 queues of buf_size int64 entries
    queue_bytes = buf_size * 8 * 5
    total_needed = dist_bytes + pred_bytes + queue_bytes

    free_mem, _ = cp.cuda.Device().mem_info
    # Leave 256MB headroom
    available = free_mem - 256 * 1024 * 1024
    if total_needed > available:
        raise MemoryError(
            f"Constrained GPU SSSP needs {total_needed / 1e9:.1f} GB VRAM "
            f"but only {available / 1e9:.1f} GB available. "
            f"State space: {total_states:,} states."
        )


# ============================================================================
# Delta computation
# ============================================================================

def _compute_constrained_delta(raster, cost_factors):
    """Compute delta from terrain costs only (tower costs are heavy)."""
    valid = raster.ravel()
    valid = valid[valid < 65535]
    if len(valid) == 0:
        return 100.0
    mean_cost = float(valid.mean())
    mean_cf = float(cost_factors.mean())
    return max(1.0, 2.0 * mean_cost * mean_cf)


# ============================================================================
# Path reconstruction (CPU-side)
# ============================================================================

def _reconstruct_path(pred_cpu, best_state, n_dirs, n_span_bins,
                      cols, min_span_bin):
    """Walk predecessor chain and detect tower placements."""
    # Collect state chain in reverse
    state_chain = []
    state = best_state
    while state >= 0:
        state_chain.append(state)
        state = int(pred_cpu[state])
    state_chain.reverse()

    states_per_cell = n_dirs * n_span_bins
    path_cells = []
    tower_cells = []

    for k, st in enumerate(state_chain):
        cell = int(st // states_per_cell)
        rem = int(st % states_per_cell)
        span_bin = rem % n_span_bins
        path_cells.append(cell)

        # Detect tower: span_bin drops between consecutive states
        if k > 0 and n_span_bins > 1:
            prev_st = state_chain[k - 1]
            prev_rem = int(prev_st % states_per_cell)
            prev_span = prev_rem % n_span_bins
            prev_cell = int(prev_st // states_per_cell)
            if prev_span >= min_span_bin and span_bin < prev_span:
                if prev_cell not in tower_cells:
                    tower_cells.append(prev_cell)

    return (np.array(path_cells, dtype=np.uint32),
            np.array(tower_cells, dtype=np.uint32))


# ============================================================================
# Delta-stepping bucket management helpers
# ============================================================================

def _refill_from_pending_v1(d_dist, d_pending, d_pending_count,
                            d_queue_a, d_near_count, d_settled,
                            d_far_count, current_bucket, delta,
                            tpb, classify_kernel, buf_size):
    """Try to find next bucket from pending states.

    Returns:
        tuple: (frontier_size, current_bucket, should_break, should_continue)
    """
    pc = int(d_pending_count[0])
    if pc > 0:
        current_bucket += 1
        bucket_low = np.float32(current_bucket * delta)
        bucket_high = np.float32((current_bucket + 1) * delta)
        d_near_count[0] = 0
        d_far_count[0] = 0
        blocks = (pc + tpb - 1) // tpb
        classify_kernel(
            (blocks,), (tpb,),
            (d_dist, d_pending, np.int32(pc),
             bucket_low, bucket_high,
             d_queue_a, d_near_count, d_settled, d_far_count))
        frontier_size = int(d_near_count[0])
        far_count = int(d_far_count[0])
        if far_count > 0:
            d_pending[:far_count] = d_settled[:far_count]
        d_pending_count[0] = far_count
        if frontier_size > 0:
            return (frontier_size, current_bucket, False, False)
        elif far_count > 0:
            return (0, current_bucket, False, True)
        else:
            return (0, current_bucket, True, False)
    else:
        return (0, current_bucket, False, False)


def _refill_full_scan_v1(d_dist, current_bucket, delta, buf_size, d_queue_a):
    """Full scan fallback: find minimum unsettled distance.

    Returns:
        tuple: (frontier_size, current_bucket, should_break)
    """
    bucket_low_f = np.float32(current_bucket * delta)
    d_cand = cp.where(
        d_dist >= bucket_low_f, d_dist, np.float32(1e30))
    min_rem = float(d_cand.min())
    if min_rem >= 1e29:
        return (0, current_bucket, True)
    current_bucket = int(min_rem / delta)
    low = np.float32(current_bucket * delta)
    high = np.float32((current_bucket + 1) * delta)
    fallback = cp.flatnonzero(
        (d_dist >= low) & (d_dist < high)).astype(cp.int64)
    frontier_size = int(fallback.size)
    if frontier_size > 0:
        n_copy = min(frontier_size, buf_size)
        d_queue_a[:n_copy] = fallback[:n_copy]
        frontier_size = n_copy
    else:
        return (0, current_bucket, True)
    return (frontier_size, current_bucket, False)


def _refill_empty_frontier_v1(d_dist, d_pending, d_pending_count,
                              d_queue_a, d_near_count, d_settled,
                              d_far_count, current_bucket, delta,
                              tpb, classify_kernel, buf_size):
    """Refill frontier when empty: try pending first, then full scan fallback.

    Returns:
        tuple: (frontier_size, current_bucket, should_break, should_continue)
    """
    # Try to find next bucket from pending
    frontier_size, current_bucket, should_break, should_continue = \
        _refill_from_pending_v1(
            d_dist, d_pending, d_pending_count,
            d_queue_a, d_near_count, d_settled,
            d_far_count, current_bucket, delta,
            tpb, classify_kernel, buf_size)
    if should_break or should_continue or frontier_size > 0:
        return (frontier_size, current_bucket, should_break, should_continue)
    # Full scan fallback: find minimum unsettled distance
    frontier_size, current_bucket, should_break = \
        _refill_full_scan_v1(
            d_dist, current_bucket, delta, buf_size, d_queue_a)
    return (frontier_size, current_bucket, should_break, False)


def _advance_bucket_v1(d_dist, d_pending, d_pending_count,
                       d_queue_a, d_near_count, d_settled,
                       d_far_count, current_bucket, delta,
                       tpb, classify_kernel, buf_size):
    """Advance to next bucket after light phase.

    Returns:
        tuple: (frontier_size, current_bucket, pending_count)
    """
    # Advance to next bucket
    current_bucket += 1
    # Classify pending into near/far for next bucket
    pc = int(d_pending_count[0])
    if pc > 0:
        bucket_low = np.float32(current_bucket * delta)
        bucket_high = np.float32((current_bucket + 1) * delta)
        d_near_count[0] = 0
        d_far_count[0] = 0
        blocks = (pc + tpb - 1) // tpb
        classify_kernel(
            (blocks,), (tpb,),
            (d_dist, d_pending, np.int32(min(pc, buf_size)),
             bucket_low, bucket_high,
             d_queue_a, d_near_count, d_settled, d_far_count))
        frontier_size = int(d_near_count[0])
        far_count = int(d_far_count[0])
        if far_count > 0:
            d_pending[:min(far_count, buf_size)] = \
                d_settled[:min(far_count, buf_size)]
        d_pending_count[0] = min(far_count, buf_size)
        return (frontier_size, current_bucket, min(far_count, buf_size))
    else:
        d_pending_count[0] = 0
        return (0, current_bucket, 0)


def _run_light_phase_v1(d_raster, rows, cols, max_cost, d_steps,
                        d_cost_factors, d_inter_lut, d_n_inter,
                        n_dirs, max_inter_cols, d_angle_cost,
                        d_angle_valid, d_step_dist, d_tower_terrain,
                        d_tower_angle, n_span_bins, span_bin_size,
                        min_span_bin, d_dist, d_pred, d_queue_a,
                        d_queue_b, d_count_out, d_pending,
                        d_pending_count, bucket_low, bucket_high,
                        buf_size, tpb, smem_bytes, relax_kernel,
                        frontier_size):
    """Light phase: iterate until no new states in current bucket.

    Returns:
        tuple: (frontier_size, d_queue_a, d_queue_b)
    """
    for _ in range(100):
        d_count_out[0] = 0
        blocks = (frontier_size + tpb - 1) // tpb
        relax_kernel(
            (blocks,), (tpb,),
            (d_raster,
             np.int32(rows), np.int32(cols), np.int32(max_cost),
             d_steps, d_cost_factors, d_inter_lut, d_n_inter,
             np.int32(n_dirs), np.int32(max_inter_cols),
             d_angle_cost, d_angle_valid, d_step_dist,
             d_tower_terrain, d_tower_angle,
             np.int32(n_span_bins), np.float32(span_bin_size),
             np.int32(min_span_bin),
             d_dist, d_pred,
             d_queue_a, np.int32(frontier_size),
             d_queue_b, d_count_out,
             d_pending, d_pending_count,
             bucket_low, bucket_high,
             np.int32(buf_size)),
            shared_mem=smem_bytes)

        frontier_size = int(d_count_out[0])
        if frontier_size == 0:
            break
        frontier_size = min(frontier_size, buf_size)
        d_queue_a, d_queue_b = d_queue_b, d_queue_a

    return (frontier_size, d_queue_a, d_queue_b)


def _extract_best_path_v1(d_dist, d_pred, target_cell, states_per_cell,
                          n_dirs, n_span_bins, cols, min_span_bin):
    """Find best target state and reconstruct path.

    Returns:
        tuple: (path_cell_indices as uint32[], tower_cell_indices as uint32[])
    """
    # Find best target state (minimum distance across all directions/spans)
    target_start = target_cell * states_per_cell
    target_end = target_start + states_per_cell
    target_dists = d_dist[target_start:target_end].get()
    best_offset = int(np.argmin(target_dists))
    best_dist = float(target_dists[best_offset])

    if best_dist >= 1e29:
        warnings.warn("GPU constrained SSSP found no path to target")
        return (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32))

    best_state = target_start + best_offset

    # Transfer pred to CPU for path reconstruction
    pred_cpu = d_pred.get()

    return _reconstruct_path(
        pred_cpu, best_state, n_dirs, n_span_bins, cols, min_span_bin)


def _init_source_states_v1(source_cell, states_per_cell, n_dirs,
                           n_span_bins, d_dist):
    """Initialize source states: all directions, span_bin=0.

    Returns:
        np.ndarray: source_states_arr (int64)
    """
    source_states = []
    for d in range(n_dirs):
        st = source_cell * states_per_cell + d * n_span_bins
        source_states.append(st)
    source_states_arr = np.array(source_states, dtype=np.int64)
    for st in source_states_arr:
        d_dist[int(st)] = np.float32(0.0)
    return source_states_arr


# ============================================================================
# Main entry point
# ============================================================================

def constrained_sssp_raster_gpu(
    raster: np.ndarray,
    source_row: int, source_col: int,
    target_row: int, target_col: int,
    steps: np.ndarray,
    angle_cost_lut: np.ndarray,
    angle_valid_lut: np.ndarray,
    step_distances: np.ndarray,
    tower_terrain_costs: np.ndarray,
    tower_angle_costs: np.ndarray,
    n_span_bins: int,
    span_bin_size: float,
    min_span: float,
    max_span: float,
    exclude_mask: np.ndarray = None,
    threads_per_block: int = 256,
    max_iterations: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find constrained shortest path on GPU with tower placement.

    Parameters:
        raster: uint16 cost raster (65535 = impassable)
        source_row, source_col: source cell coordinates
        target_row, target_col: target cell coordinates
        steps: int8 (n_dirs, 2) neighborhood steps
        angle_cost_lut: float32 (n_dirs, n_dirs) angle penalty costs
        angle_valid_lut: uint8 (n_dirs, n_dirs) angle validity
        step_distances: float32 (n_dirs,) physical distance per step
        tower_terrain_costs: float32 (65536,) tower cost by terrain value
        tower_angle_costs: float32 (n_dirs, n_dirs) tower type cost
        n_span_bins: number of span discretization bins
        span_bin_size: physical distance per span bin (meters)
        min_span: minimum allowed span between towers (meters)
        max_span: maximum allowed span between towers (meters)
        exclude_mask: optional uint8 mask (1=traversable, 0=blocked)
        threads_per_block: CUDA threads per block
        max_iterations: maximum outer loop iterations

    Returns:
        tuple: (path_cell_indices as uint32[], tower_cell_indices as uint32[])
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU not available. Install cupy: pip install cupy-cuda12x")

    rows, cols = raster.shape
    n_cells = rows * cols
    n_dirs = len(steps)
    total_states = n_cells * n_dirs * n_span_bins
    max_cost = int(np.iinfo(np.uint16).max)
    min_span_bin = int(min_span / span_bin_size)

    source_cell = source_row * cols + source_col
    target_cell = target_row * cols + target_col

    # Apply exclude mask to raster
    if exclude_mask is not None:
        raster = raster.copy()
        raster[exclude_mask == 0] = 65535

    # Upload step lookup tables
    steps_arr, intermediates_lut, n_intermediates, cost_factors = \
        prepare_step_lookup_tables(steps)
    max_inter_cols = intermediates_lut.shape[1]
    smem_bytes = _compute_constrained_smem(n_dirs, max_inter_cols)

    # Compute delta from terrain costs
    delta = _compute_constrained_delta(raster, cost_factors)

    # Buffer size for queues: cap at reasonable fraction of state space
    buf_size = min(total_states, max(n_cells * n_dirs * 2, 1 << 20))

    # VRAM check
    _check_vram(total_states, n_dirs, buf_size)

    # Upload arrays to GPU
    d_raster = cp.asarray(raster.astype(np.uint16))
    d_steps = cp.asarray(steps_arr.astype(np.int8))
    d_cost_factors = cp.asarray(cost_factors.astype(np.float32))
    d_inter_lut = cp.asarray(intermediates_lut.astype(np.int8))
    d_n_inter = cp.asarray(n_intermediates.astype(np.int32))
    d_angle_cost = cp.asarray(
        angle_cost_lut.reshape(-1).astype(np.float32))
    d_angle_valid = cp.asarray(
        angle_valid_lut.reshape(-1).astype(np.uint8))
    d_step_dist = cp.asarray(step_distances.astype(np.float32))
    d_tower_terrain = cp.asarray(tower_terrain_costs.astype(np.float32))
    d_tower_angle = cp.asarray(
        tower_angle_costs.reshape(-1).astype(np.float32))

    # Initialize dist (float32, 1e30 = inf) and pred (int64, -1)
    d_dist = cp.full(total_states, np.float32(1e30), dtype=cp.float32)
    d_pred = cp.full(total_states, np.int64(-1), dtype=cp.int64)

    # Initialize source: all directions, span_bin=0
    states_per_cell = n_dirs * n_span_bins
    source_states_arr = _init_source_states_v1(
        source_cell, states_per_cell, n_dirs, n_span_bins, d_dist)

    # Allocate queues (int64)
    d_queue_a = cp.empty(buf_size, dtype=cp.int64)
    d_queue_b = cp.empty(buf_size, dtype=cp.int64)
    d_settled = cp.empty(buf_size, dtype=cp.int64)
    d_pending = cp.empty(buf_size, dtype=cp.int64)
    d_count_out = cp.zeros(1, dtype=cp.int32)
    d_pending_count = cp.zeros(1, dtype=cp.int32)
    d_near_count = cp.zeros(1, dtype=cp.int32)
    d_far_count = cp.zeros(1, dtype=cp.int32)

    # Seed initial frontier
    n_source = len(source_states_arr)
    d_queue_a[:n_source] = cp.asarray(source_states_arr)
    frontier_size = n_source

    # Compile kernels
    relax_kernel = _get_constrained_kernel(
        "constrained_relax", _CONSTRAINED_RELAX_KERNEL)
    classify_kernel = _get_constrained_kernel(
        "classify_constrained", _CLASSIFY_KERNEL)

    tpb = threads_per_block
    current_bucket = 0

    # Delta-stepping main loop
    for _ in range(max_iterations):
        if frontier_size == 0:
            frontier_size, current_bucket, should_break, should_continue = \
                _refill_empty_frontier_v1(
                    d_dist, d_pending, d_pending_count,
                    d_queue_a, d_near_count, d_settled,
                    d_far_count, current_bucket, delta,
                    tpb, classify_kernel, buf_size)
            if should_break:
                break
            if should_continue:
                continue

        if frontier_size == 0:
            break

        bucket_low = np.float32(current_bucket * delta)
        bucket_high = np.float32((current_bucket + 1) * delta)

        # Light phase: iterate until no new states in current bucket
        frontier_size, d_queue_a, d_queue_b = _run_light_phase_v1(
            d_raster, rows, cols, max_cost, d_steps,
            d_cost_factors, d_inter_lut, d_n_inter,
            n_dirs, max_inter_cols, d_angle_cost,
            d_angle_valid, d_step_dist, d_tower_terrain,
            d_tower_angle, n_span_bins, span_bin_size,
            min_span_bin, d_dist, d_pred, d_queue_a,
            d_queue_b, d_count_out, d_pending,
            d_pending_count, bucket_low, bucket_high,
            buf_size, tpb, smem_bytes, relax_kernel,
            frontier_size)

        # Advance to next bucket
        frontier_size, current_bucket, _ = _advance_bucket_v1(
            d_dist, d_pending, d_pending_count,
            d_queue_a, d_near_count, d_settled,
            d_far_count, current_bucket, delta,
            tpb, classify_kernel, buf_size)

    cp.cuda.Stream.null.synchronize()

    return _extract_best_path_v1(
        d_dist, d_pred, target_cell, states_per_cell,
        n_dirs, n_span_bins, cols, min_span_bin)

"""
GPU eikonal solver (isotropic fast iterative method) for PYORPS.

Solves the isotropic eikonal equation ``|grad T(x)| = c(x)``, ``T(src) = 0``
on the cost raster: ``T(x)`` is the minimal accumulated cost
``min over paths of integral c ds`` over all *continuous* paths — the
continuous limit of what the discrete graph kernels compute, without the
neighborhood metrication (elongation) bias.

Unit convention (calibration contract, plan section 1):
    ``c(x)`` is the raster value of the cell containing ``x`` (piecewise
    constant), grid spacing ``h = 1`` cell unit. On a uniform raster of
    value ``v``, an axis-aligned source-to-target line of ``L`` cells gives
    ``T = v * L`` — directly comparable to ``dist[target]`` from the
    discrete GPU/Cython backends (same units, no rescaling).

Solvers:
    - :func:`eikonal_raster_gpu` — block-FIM (Jeong & Whitaker 2008):
      tile-granular active list, shared-memory tile sweeps, host loop.
      No atomics, no queues, no cooperative groups in the hot path.
    - :func:`eikonal_raster_gpu_naive` — single-kernel Jacobi iteration
      over the full grid. Slow; kept as the test oracle.

Both kernels consume a float32 *slowness* field prepared host-side (uint16
sentinel / float32 forbidden values become ``1e30``), so a single compiled
kernel serves both raster dtypes. This deviates from the sssp_gpu textual
uint16/float variant transform on purpose: one kernel source, sentinel
handling in exactly one place, and the plan's memory budget (4 B/px for
cost) already assumed a float copy.

Unreachable cells keep ``T = 1e30`` (same convention as the discrete GPU
line: finite check ``< 1e29``).

Path extraction (:func:`trace_paths`) is host-side vectorized numpy —
steepest-descent RK2 integration of ``dx/dt = -grad T / |grad T|`` with
plateau / shock-line fallbacks and a hard step cap. It returns continuous
polylines; :func:`polyline_to_cells` rasterizes them for the existing
Path/GeoDataFrame machinery.

References:
    [1] Jeong, W.-K., Whitaker, R. T.: A fast iterative method for
        eikonal equations. SIAM J. Sci. Comput. 30(5), 2008.
    [2] Fu, Z., Jeong, W.-K., Pan, Y., Kirby, R. M., Whitaker, R. T.:
        A fast iterative method for solving the eikonal equation on
        triangulated surfaces. SIAM J. Sci. Comput. 33(5), 2011
        (PMC3360588) — states the three FIM convergence invariants and
        the patch-based GPU scheme this tile solver follows. Deliberate
        deviations from that scheme (documented in
        benchmarks/EIKONAL_FINDINGS.md): neighbors activate on
        boundary-value change instead of on patch convergence (a
        conservative superset that propagates one pass sooner), and the
        per-patch double-buffered Jacobi + per-vertex flag reduction is
        replaced by monotone chaotic relaxation in a single shared
        buffer with one post-loop "still improvable?" probe (racy reads
        of monotone non-increasing values are always valid upper
        bounds; for a chaotic sweep, "no change in the last iteration"
        would not imply a fixed point, so the probe re-evaluates the
        operator — the correct convergence test for this variant).
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Union

from pyorps.utils.traversal_gpu import GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp


#: Values >= this threshold mean "unreachable / impassable".
UNREACHED = np.float32(1e30)
#: Finite check threshold (mirrors the discrete GPU convention).
FINITE_LIMIT = 1e29


# ============================================================================
# CUDA kernel sources
# ============================================================================

# Naive full-grid Jacobi sweep (reference implementation / test oracle).
# Double-buffered: reads t_in, writes t_out, monotone non-increasing.
_JACOBI_KERNEL = r"""
extern "C" __global__
void eikonal_jacobi(
    const float* __restrict__ slowness,
    const float* __restrict__ t_in,
    float*       __restrict__ t_out,
    const unsigned char* __restrict__ frozen,
    const int rows,
    const int cols,
    const float eps_abs,
    const float eps_rel,
    int* __restrict__ changed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n) return;

    float t_old = t_in[idx];
    float c = slowness[idx];
    if (c >= 1e30f || frozen[idx]) {
        t_out[idx] = t_old;
        return;
    }

    int r = idx / cols;
    int col = idx - r * cols;

    float tw = (col > 0)        ? t_in[idx - 1]    : 1e30f;
    float te = (col < cols - 1) ? t_in[idx + 1]    : 1e30f;
    float tn = (r > 0)          ? t_in[idx - cols] : 1e30f;
    float ts = (r < rows - 1)   ? t_in[idx + cols] : 1e30f;

    float a = fminf(tw, te);
    float b = fminf(tn, ts);
    if (a > b) { float tmp = a; a = b; b = tmp; }

    float t_new;
    if (b - a >= c) {
        t_new = a + c;
    } else {
        float diff = a - b;
        t_new = 0.5f * (a + b + sqrtf(fmaxf(2.0f * c * c - diff * diff,
                                            0.0f)));
    }
    t_new = fminf(t_old, t_new);
    t_out[idx] = t_new;
    if (t_old - t_new > eps_abs + eps_rel * t_new)
        changed[0] = 1;
}
"""

# Block-FIM tile sweep (@B@ x @B@ threads per block). A fixed modest grid
# strides over the device-built active list — the active count lives on
# the GPU, and covering all tiles with one block each would schedule
# thousands of empty blocks per pass. Per tile: load tile + 1-cell halo
# of T into shared memory, run up to n_inner chaotic relaxation
# iterations (races read old-or-new values — both are valid upper bounds;
# monotone min keeps the fixed point exact), write back improvements,
# record per-tile convergence and boundary-change flags. Interior cells
# are written only by the block owning the tile (the list holds unique
# tiles), halo cells are read-only — no atomics in the cell-update path.
# Block 0 also clears next_count for the activation kernel that follows
# in stream order.
_FIM_SWEEP_KERNEL = r"""
extern "C" __global__
void fim_sweep(
    const float* __restrict__ slowness,
    float*       __restrict__ T,
    const unsigned char* __restrict__ frozen,
    const int*   __restrict__ active_list,
    const int*   __restrict__ active_count,
    const int rows,
    const int cols,
    const int tiles_c,
    const int n_inner,
    const float eps_abs,
    const float eps_rel,
    unsigned char* __restrict__ active,
    unsigned char* __restrict__ boundary_changed,
    int* __restrict__ next_count,
    unsigned long long* __restrict__ stats
) {
    const int B = @B@;
    __shared__ float s_t[(B + 2) * (B + 2)];
    __shared__ int s_changed_iter;
    __shared__ int s_boundary;
    __shared__ int s_settle;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * B + tx;

    if (blockIdx.x == 0 && tid == 0) next_count[0] = 0;

    int count = active_count[0];
    bool on_edge = (tx == 0 || tx == B - 1 || ty == 0 || ty == B - 1);

    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        int tile_id = active_list[w];
        int tile_r = tile_id / tiles_c;
        int tile_c = tile_id - tile_r * tiles_c;
        int base_r = tile_r * B;
        int base_c = tile_c * B;

        int gr = base_r + ty;
        int gc = base_c + tx;
        bool in_grid = (gr < rows) && (gc < cols);
        int gidx = gr * cols + gc;

        float cval = 1e30f;
        bool frz = true;
        if (in_grid) {
            cval = slowness[gidx];
            frz = (frozen[gidx] != 0);
        }
        float t0 = in_grid ? T[gidx] : 1e30f;
        int sp = (ty + 1) * (B + 2) + (tx + 1);
        s_t[sp] = t0;

        // Halo (edges only — the 4-point stencil needs no corners)
        if (ty == 0) {
            int hr = base_r - 1;
            s_t[tx + 1] =
                (hr >= 0 && gc < cols) ? T[hr * cols + gc] : 1e30f;
        }
        if (ty == B - 1) {
            int hr = base_r + B;
            s_t[(B + 1) * (B + 2) + tx + 1] =
                (hr < rows && gc < cols) ? T[hr * cols + gc] : 1e30f;
        }
        if (tx == 0) {
            int hc = base_c - 1;
            s_t[(ty + 1) * (B + 2)] =
                (hc >= 0 && gr < rows) ? T[gr * cols + hc] : 1e30f;
        }
        if (tx == B - 1) {
            int hc = base_c + B;
            s_t[(ty + 1) * (B + 2) + B + 1] =
                (hc < cols && gr < rows) ? T[gr * cols + hc] : 1e30f;
        }

        if (tid == 0) { s_changed_iter = 0; s_boundary = 0; }
        __syncthreads();

        bool can_update = in_grid && (cval < 1e30f) && !frz;

        // Relaxation iterations need no per-iteration convergence
        // protocol: writes are monotone upper bounds, so racy reads are
        // always valid. One barrier per iteration keeps cross-warp
        // propagation moving; convergence is decided by a single probe
        // afterwards. Every 8th iteration a SETTLE check lets a tile
        // whose values stopped moving exit early (measured 65-372
        // updates/cell vs ~18 for an ordered method — most inner
        // iterations behind the front are no-ops). The break decision
        // is read by every thread BETWEEN two barriers with no
        // intervening writes — uniform, so no thread can outlive the
        // barrier (the increment-1 shared-flag race pattern is exactly
        // what this protocol avoids: the flag cleared for the NEXT
        // check is separated from this read by a full barrier). An
        // early exit is a pure performance decision: the post-loop
        // probe remains the only convergence authority.
        int iters_run = 0;
        for (int it = 0; it < n_inner; ++it) {
            iters_run = it + 1;
            bool settle_check = ((it & 7) == 7);
            if (settle_check) {
                if (tid == 0) s_settle = 0;
                __syncthreads();
            }
            if (can_update) {
                float t_old = s_t[sp];
                float a = fminf(s_t[sp - 1], s_t[sp + 1]);
                float b = fminf(s_t[sp - (B + 2)], s_t[sp + (B + 2)]);
                if (a > b) { float tmp = a; a = b; b = tmp; }
                float t_new;
                if (b - a >= cval) {
                    t_new = a + cval;
                } else {
                    float diff = a - b;
                    t_new = 0.5f * (a + b +
                        sqrtf(fmaxf(2.0f * cval * cval - diff * diff,
                                    0.0f)));
                }
                if (t_new < t_old) {
                    s_t[sp] = t_new;
                    if (settle_check) s_settle = 1;
                }
            }
            __syncthreads();
            if (settle_check && s_settle == 0) break;
        }

        // Probe: can any cell still improve beyond eps? Then the tile is
        // not converged and stays active. Boundary activation uses the
        // net change over the whole sweep with the same eps threshold
        // (sub-eps drift must not re-activate neighbors forever).
        float tf = in_grid ? s_t[sp] : 1e30f;
        if (can_update) {
            float a = fminf(s_t[sp - 1], s_t[sp + 1]);
            float b = fminf(s_t[sp - (B + 2)], s_t[sp + (B + 2)]);
            if (a > b) { float tmp = a; a = b; b = tmp; }
            float t_new;
            if (b - a >= cval) {
                t_new = a + cval;
            } else {
                float diff = a - b;
                t_new = 0.5f * (a + b +
                    sqrtf(fmaxf(2.0f * cval * cval - diff * diff,
                                0.0f)));
            }
            if (t_new < tf - (eps_abs + eps_rel * t_new))
                s_changed_iter = 1;
            if (on_edge && t0 - tf > eps_abs + eps_rel * tf)
                s_boundary = 1;
        }
        __syncthreads();

        if (in_grid && tf < t0) T[gidx] = tf;
        if (tid == 0) {
            active[tile_id] = (s_changed_iter != 0) ? 1 : 0;
            boundary_changed[tile_id] = (s_boundary != 0) ? 1 : 0;
            // Tile-granular diagnostics counters (not the cell hot
            // path): stats[0] = tile sweeps, stats[1] = inner
            // iterations actually run (settle checks cut them short)
            // -> the honest update-redundancy metric a la
            // Fu/Jeong/Whitaker 2011 Table 3.4.
            atomicAdd(&stats[0], 1ULL);
            atomicAdd(&stats[1], (unsigned long long)iters_run);
        }
        __syncthreads();   // s_t reused by the next tile of this block
    }
}
"""

# Tile activation + next-pass list building. A deactivated tile
# re-activates when a 4-neighbor tile's boundary values changed this pass
# (its halo is stale); over-activation is harmless — a converged tile with
# an unchanged halo deactivates again after one no-op sweep. The compact
# active list for the next pass is appended here with a tile-granular
# atomic (one thread per tile, thousands of tiles — not the per-cell hot
# path), which keeps the host loop free of per-pass synchronization.
_FIM_ACTIVATE_KERNEL = r"""
extern "C" __global__
void fim_activate(
    unsigned char* __restrict__ active,
    const unsigned char* __restrict__ boundary_changed,
    const int tiles_r,
    const int tiles_c,
    int* __restrict__ list_out,
    int* __restrict__ count_out
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tiles = tiles_r * tiles_c;
    if (t >= n_tiles) return;

    if (!active[t]) {
        int tr = t / tiles_c;
        int tc = t - tr * tiles_c;
        bool act = false;
        if (tr > 0 && boundary_changed[t - tiles_c]) act = true;
        if (!act && tr < tiles_r - 1 && boundary_changed[t + tiles_c])
            act = true;
        if (!act && tc > 0 && boundary_changed[t - 1]) act = true;
        if (!act && tc < tiles_c - 1 && boundary_changed[t + 1])
            act = true;
        if (act) active[t] = 1;
    }
    if (active[t]) {
        int pos = atomicAdd(count_out, 1);
        list_out[pos] = t;
    }
}
"""

# Second-order refinement sweep (increment 2, order=2 stage 2). The
# mixed-order Godunov operator is NOT monotone (the one-sided
# second-order term (3T - 4*T1 + T2)/2h *decreases* with T2), so it
# cannot ride the chaotic single-buffer machinery of stage 1 — an
# undershoot computed from an unconverged T2 upper bound would lock in
# under monotone-min. Stage 2 therefore runs *deterministic* Jacobi:
# globally double-buffered (reads T_in, writes T_out; the host
# pre-copies T_in into T_out so inactive tiles stay intact), with
# double-buffered shared tiles inside. Per axis the standard switch
# applies: use the second-order difference iff the second upwind cell
# is finite and T2 <= T1 (then beta/alpha = (4*T1 - T2)/3 >= T1 keeps
# causality); else first order. The two-axis solve is the same
# sorted-cascade as stage 1 with per-axis (alpha, beta):
#   sum_k (alpha_k * T - beta_k)^2 = c^2,  T = larger root.
_FIM_SWEEP2_KERNEL = r"""
extern "C" {

// Per-axis structure comes from the FROZEN code byte (computed once
// from the converged first-order field by fim_freeze2): a live
// T2<=T1 switch flaps on float32 jitter / cell-scale noise and the
// iteration never settles (measured). Bits per axis: ok, dir (1 = use
// the +offset side), second-order.
__device__ __forceinline__ void coded_axis(
        const float* __restrict__ s, int sp, int off,
        int ok_bit, int dir_bit, int sec_bit, int code,
        float* alpha, float* beta, bool* ok, float* t1, bool* sec)
{
    *sec = false;
    if (!(code & ok_bit)) { *ok = false; return; }
    int sgn = (code & dir_bit) ? off : -off;
    float T1 = s[sp + sgn];
    if (T1 >= 1e29f) { *ok = false; return; }
    *ok = true;
    *t1 = T1;
    if (code & sec_bit) {
        float T2 = s[sp + 2 * sgn];
        if (T2 < 1e29f) {
            *alpha = 1.5f;
            *beta = 2.0f * T1 - 0.5f * T2;    // (4*T1 - T2) / 2
            *sec = true;
            return;
        }
    }
    *alpha = 1.0f;
    *beta = T1;
}

// Plain first-order Godunov on the frozen-direction neighbor values —
// the causality fallback (bounded below by min(T1): no runaway).
__device__ __forceinline__ float godunov1(
        bool ok_r, float t1_r, bool ok_c, float t1_c, float cval)
{
    if (!ok_r) return t1_c + cval;
    if (!ok_c) return t1_r + cval;
    float a = fminf(t1_r, t1_c);
    float b = fmaxf(t1_r, t1_c);
    if (b - a >= cval) return a + cval;
    float diff = a - b;
    return 0.5f * (a + b
                   + sqrtf(fmaxf(2.0f * cval * cval - diff * diff,
                                 0.0f)));
}

__device__ float mixed_update(
        const float* __restrict__ s, int sp, int W, float cval,
        int code)
{
    float a_r, b_r, a_c, b_c, t1_r, t1_c;
    bool ok_r, ok_c, sec_r, sec_c;
    coded_axis(s, sp, W, 1, 2, 4, code,
               &a_r, &b_r, &ok_r, &t1_r, &sec_r);
    coded_axis(s, sp, 1, 8, 16, 32, code,
               &a_c, &b_c, &ok_c, &t1_c, &sec_c);
    if (!ok_r && !ok_c) return 1e30f;

    float cand;
    bool used_r = ok_r, used_c = ok_c;
    if (ok_r && !ok_c) {
        cand = (b_r + cval) / a_r;
    } else if (!ok_r && ok_c) {
        cand = (b_c + cval) / a_c;
    } else {
        float arr_r = b_r / a_r;
        float arr_c = b_c / a_c;
        float a1, b1, a2, b2, arr2;
        bool first_is_r = (arr_r <= arr_c);
        if (first_is_r) { a1 = a_r; b1 = b_r; a2 = a_c; b2 = b_c;
                          arr2 = arr_c; }
        else            { a1 = a_c; b1 = b_c; a2 = a_r; b2 = b_r;
                          arr2 = arr_r; }
        float t1ax = (b1 + cval) / a1;
        if (t1ax <= arr2) {
            cand = t1ax;
            used_r = first_is_r;
            used_c = !first_is_r;
        } else {
            float qa = a1 * a1 + a2 * a2;
            float qb = -2.0f * (a1 * b1 + a2 * b2);
            float qc = b1 * b1 + b2 * b2 - cval * cval;
            float disc = qb * qb - 4.0f * qa * qc;
            cand = (-qb + sqrtf(fmaxf(disc, 0.0f))) / (2.0f * qa);
        }
    }

    // Causality safeguard: a second-order axis whose candidate falls
    // below its own upwind value T1 violates upwind causality (the
    // frozen T2 <= T1 ordering flipped during refinement); the
    // extrapolation is then unbounded below and neighbor chains can
    // run away to -inf (measured on cell-scale noise). Fall back to
    // plain first order on the frozen directions.
    if ((used_r && sec_r && cand < t1_r)
            || (used_c && sec_c && cand < t1_c))
        return godunov1(ok_r, t1_r, ok_c, t1_c, cval);
    return cand;
}

// Freeze the per-axis upwind structure from the converged first-order
// field: direction (smaller neighbor), order (second iff the second
// upwind cell is finite, T2 <= T1, and the axis second difference is
// small — |T0 - 2*T1 + T2| <= 0.5*c — an ENO-style smoothness test).
// The tight threshold matters twice over: it keeps cell-scale-noise
// regions at first order for accuracy AND for stability — frozen
// second-order chains through rough data amplify perturbations (the
// one-sided difference has gain 2 on T1) faster than the damped
// iteration contracts, and the refinement never settles (measured:
// random 500^2 capped at 2048 passes with a 2*c threshold).
__global__ void fim_freeze2(
    const float* __restrict__ T,
    const float* __restrict__ slowness,
    const int rows,
    const int cols,
    unsigned char* __restrict__ codes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n) return;
    int r = idx / cols;
    int c = idx - r * cols;
    float cval = slowness[idx];
    float t0 = T[idx];
    int code = 0;
    if (cval < 1e30f && t0 < 1e29f) {
        // row axis (offset cols), then col axis (offset 1)
        for (int axis = 0; axis < 2; ++axis) {
            int lo_ok = (axis == 0) ? (r > 0) : (c > 0);
            int hi_ok = (axis == 0) ? (r < rows - 1) : (c < cols - 1);
            int lo2_ok = (axis == 0) ? (r > 1) : (c > 1);
            int hi2_ok = (axis == 0) ? (r < rows - 2) : (c < cols - 2);
            int off = (axis == 0) ? cols : 1;
            float t1a = lo_ok ? T[idx - off] : 1e30f;
            float t1b = hi_ok ? T[idx + off] : 1e30f;
            int use_b = (t1b < t1a) ? 1 : 0;
            float T1 = use_b ? t1b : t1a;
            float T2 = use_b ? (hi2_ok ? T[idx + 2 * off] : 1e30f)
                             : (lo2_ok ? T[idx - 2 * off] : 1e30f);
            int shift = axis * 3;
            if (T1 < 1e29f) {
                code |= (1 << shift);                    // ok
                if (use_b) code |= (2 << shift);         // direction
                if (T2 < 1e29f && T2 <= T1
                        && fabsf(t0 - 2.0f * T1 + T2) <= 0.5f * cval)
                    code |= (4 << shift);                // second order
            }
        }
    }
    codes[idx] = (unsigned char)code;
}

// Stall degrade: mask code bits of every cell in the still-active
// tiles. Called by the host when the active count stops improving — a
// handful of frozen second-order cells can oscillate above the float32
// floor indefinitely (local amplification beats the damping). Level 1
// (keep_mask ~(4|32)) demotes to first order; level 2 (keep_mask 0)
// freezes the stuck cells at their current values entirely — updates
// stop, the probe reads no change, termination is guaranteed.
__global__ void fim_demote2(
    unsigned char* __restrict__ codes,
    const int* __restrict__ active_list,
    const int* __restrict__ active_count,
    const int rows,
    const int cols,
    const int tiles_c,
    const int keep_mask
) {
    const int B = @B@;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int count = active_count[0];
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        int tile_id = active_list[w];
        int tile_r = tile_id / tiles_c;
        int tile_c = tile_id - tile_r * tiles_c;
        int gr = tile_r * B + ty;
        int gc = tile_c * B + tx;
        if (gr < rows && gc < cols)
            codes[gr * cols + gc] &= (unsigned char)keep_mask;
    }
}

__global__ void fim_sweep2(
    const float* __restrict__ slowness,
    const float* __restrict__ t_in,
    float*       __restrict__ t_out,
    const unsigned char* __restrict__ frozen,
    const unsigned char* __restrict__ codes,
    const int*   __restrict__ active_list,
    const int*   __restrict__ active_count,
    const int rows,
    const int cols,
    const int tiles_c,
    const int n_inner,
    const float eps_abs,
    const float eps_rel,
    unsigned char* __restrict__ active,
    unsigned char* __restrict__ boundary_changed,
    int* __restrict__ next_count,
    unsigned long long* __restrict__ stats
) {
    const int B = @B@;
    const int W = B + 4;
    __shared__ float s_a[W * W];
    __shared__ float s_b[W * W];
    __shared__ int s_changed;
    __shared__ int s_boundary;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * B + tx;

    if (blockIdx.x == 0 && tid == 0) next_count[0] = 0;

    int count = active_count[0];
    bool on_band = (tx < 2 || tx >= B - 2 || ty < 2 || ty >= B - 2);

    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        int tile_id = active_list[w];
        int tile_r = tile_id / tiles_c;
        int tile_c = tile_id - tile_r * tiles_c;
        int base_r = tile_r * B;
        int base_c = tile_c * B;

        int gr = base_r + ty;
        int gc = base_c + tx;
        bool in_grid = (gr < rows) && (gc < cols);
        int gidx = gr * cols + gc;

        float cval = 1e30f;
        bool frz = true;
        int code = 0;
        if (in_grid) {
            cval = slowness[gidx];
            frz = (frozen[gidx] != 0);
            code = (int)codes[gidx];
        }
        float t0 = in_grid ? t_in[gidx] : 1e30f;
        int sp = (ty + 2) * W + (tx + 2);
        s_a[sp] = t0;
        s_b[sp] = t0;

        // 2-deep halo bands (axis stencil needs no corners)
        if (ty < 2) {
            int hr = base_r - 2 + ty;
            float v = (hr >= 0 && gc < cols) ? t_in[hr * cols + gc]
                                             : 1e30f;
            s_a[ty * W + tx + 2] = v;
            s_b[ty * W + tx + 2] = v;
        }
        if (ty >= B - 2) {
            int hr = base_r + B + (ty - (B - 2));
            float v = (hr < rows && gc < cols) ? t_in[hr * cols + gc]
                                               : 1e30f;
            s_a[(ty + 4) * W + tx + 2] = v;
            s_b[(ty + 4) * W + tx + 2] = v;
        }
        if (tx < 2) {
            int hc = base_c - 2 + tx;
            float v = (hc >= 0 && gr < rows) ? t_in[gr * cols + hc]
                                             : 1e30f;
            s_a[(ty + 2) * W + tx] = v;
            s_b[(ty + 2) * W + tx] = v;
        }
        if (tx >= B - 2) {
            int hc = base_c + B + (tx - (B - 2));
            float v = (hc < cols && gr < rows) ? t_in[gr * cols + hc]
                                               : 1e30f;
            s_a[(ty + 2) * W + tx + 4] = v;
            s_b[(ty + 2) * W + tx + 4] = v;
        }

        if (tid == 0) { s_changed = 0; s_boundary = 0; }
        __syncthreads();

        bool can_update = in_grid && (cval < 1e30f) && !frz;

        // Deterministic double-buffered damped Jacobi (no monotone min:
        // the refinement may move values in either direction). The 0.5
        // damping keeps the fixed point but kills the 2-cycles that
        // plain simultaneous updates produce on the non-monotone
        // +-2-coupled operator.
        for (int it = 0; it < n_inner; ++it) {
            const float* in_ = (it & 1) ? s_b : s_a;
            float* out_ = (it & 1) ? s_a : s_b;
            if (can_update) {
                float cand = mixed_update(in_, sp, W, cval, code);
                out_[sp] = (cand < 1e29f)
                    ? 0.5f * (cand + in_[sp]) : in_[sp];
            }
            __syncthreads();
        }

        const float* fin = (n_inner & 1) ? s_b : s_a;
        float tf = in_grid ? fin[sp] : 1e30f;
        if (can_update) {
            float cand = mixed_update(fin, sp, W, cval, code);
            if (cand < 1e29f
                    && fabsf(cand - tf) > eps_abs + eps_rel * fabsf(cand))
                s_changed = 1;
            if (on_band && fabsf(tf - t0) > eps_abs + eps_rel * fabsf(tf))
                s_boundary = 1;
        }
        __syncthreads();

        if (in_grid && tf != t0) t_out[gidx] = tf;
        if (tid == 0) {
            active[tile_id] = (s_changed != 0) ? 1 : 0;
            boundary_changed[tile_id] = (s_boundary != 0) ? 1 : 0;
            atomicAdd(&stats[0], 1ULL);
            atomicAdd(&stats[1], (unsigned long long)n_inner);
        }
        __syncthreads();   // shared buffers reused by the next tile
    }
}

}  // extern "C"
"""

# Targeted-early-exit check (increment 2): min T over the cells + 1-cell
# halo of every ACTIVE tile, plus "is the target's tile active". Any
# future write anywhere is >= that min (monotone updates; a new value is
# always >= the used neighbor values, which live in the tile or its
# halo), so once the target's tile is inactive and the active min is
# >= T[target], no future pass can improve the target: exact early exit.
# atomicMin on the int reinterpretation is order-correct for the
# non-negative floats T holds.
_FIM_TARGET_CHECK_KERNEL = r"""
extern "C" __global__
void fim_target_check(
    const float* __restrict__ T,
    const int* __restrict__ active_list,
    const int* __restrict__ active_count,
    const int rows,
    const int cols,
    const int tiles_c,
    const int target_tile,
    int* __restrict__ result      // [0] min-as-int, [1] target active
) {
    const int B = @B@;
    __shared__ float s_min[B * B];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * B + tx;
    int count = active_count[0];

    float local_min = 3.0e38f;
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        int tile_id = active_list[w];
        if (tid == 0 && tile_id == target_tile) result[1] = 1;
        int tile_r = tile_id / tiles_c;
        int tile_c = tile_id - tile_r * tiles_c;
        int gr = tile_r * B + ty;
        int gc = tile_c * B + tx;
        if (gr < rows && gc < cols)
            local_min = fminf(local_min, T[gr * cols + gc]);
        if (ty == 0) {
            int hr = tile_r * B - 1;
            if (hr >= 0 && gc < cols)
                local_min = fminf(local_min, T[hr * cols + gc]);
        }
        if (ty == B - 1) {
            int hr = tile_r * B + B;
            if (hr < rows && gc < cols)
                local_min = fminf(local_min, T[hr * cols + gc]);
        }
        if (tx == 0) {
            int hc = tile_c * B - 1;
            if (hc >= 0 && gr < rows)
                local_min = fminf(local_min, T[gr * cols + hc]);
        }
        if (tx == B - 1) {
            int hc = tile_c * B + B;
            if (hc < cols && gr < rows)
                local_min = fminf(local_min, T[gr * cols + hc]);
        }
    }
    s_min[tid] = local_min;
    __syncthreads();
    for (int s = (B * B) / 2; s > 0; s >>= 1) {
        if (tid < s) s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMin(&result[0], __float_as_int(fmaxf(s_min[0], 0.0f)));
}
"""

# Device-side path tracer: one thread per target, a semantic port of
# the host tracer (trace_path) — Heun/RK2 descent on on-the-fly masked
# central-difference gradients, NaN-aware bilinear sampling, the
# monotone-interpolated-T rule, discrete descent hops, shock-line stall
# detection. Pathological states the host handles with unbounded work
# (plateau BFS) or a raise (step cap) set status = 2 and leave the
# target to the host tracer — the fallback keeps semantics identical
# while the common case runs on the GPU (the host tracer's per-step
# Python arithmetic made tracing the end-to-end bottleneck: 645 ms
# trace vs 63 ms solve at 3000^2). The walk math is float32 (phase 3):
# the serial FP64 dependency chain was the measured bottleneck on
# consumer silicon (1/64 FP64 rate — 99 -> 28 ms at 3000^2); T itself
# is float32, so float32 positions/gradients lose nothing structural.
# Individual float-marginal decisions may differ from the float64 host
# tracer; both satisfy the same invariants. Polyline output stays
# float64 (API contract).
_TRACE_KERNEL = r"""
extern "C" {

#define TRACE_INVALID 1e29f

__device__ __forceinline__ float t_masked(
        const float* __restrict__ T, int rows, int cols, int r, int c)
{
    if (r < 0 || r >= rows || c < 0 || c >= cols) return nanf("");
    float v = T[r * cols + c];
    return (v < TRACE_INVALID) ? (float)v : nanf("");
}

// Shared bilinear-corner setup (port of _sample_bilinear's clamping):
// writes the 4 corner cells and weights.
__device__ void bilinear_corners(
        int rows, int cols, float r, float c,
        int* r0, int* c0, int* r1, int* c1, float* w)
{
    if (r < 0.0f) r = 0.0f; else if (r > rows - 1.0f) r = rows - 1.0f;
    if (c < 0.0f) c = 0.0f; else if (c > cols - 1.0f) c = cols - 1.0f;
    int r0_ = (int)r;
    if (r0_ > rows - 2) r0_ = (rows - 2 > 0) ? rows - 2 : 0;
    int c0_ = (int)c;
    if (c0_ > cols - 2) c0_ = (cols - 2 > 0) ? cols - 2 : 0;
    int r1_ = (r0_ + 1 < rows - 1) ? r0_ + 1 : rows - 1;
    int c1_ = (c0_ + 1 < cols - 1) ? c0_ + 1 : cols - 1;
    float fr = r - r0_;
    float fc = c - c0_;
    w[0] = (1.0f - fr) * (1.0f - fc);
    w[1] = (1.0f - fr) * fc;
    w[2] = fr * (1.0f - fc);
    w[3] = fr * fc;
    *r0 = r0_; *c0 = c0_; *r1 = r1_; *c1 = c1_;
}

// -------- cached 4x4 block (performance plan phase 3) ---------------
// A Heun step moves <= 0.5f cells, so the 4x4 T neighborhood anchored
// at (bilinear r0 - 1, c0 - 1) covers every read of a sample: the 4
// bilinear corners AND their axis neighbors for the corner gradients.
// Caching it in (register/local) memory cuts the ~40 dependent global
// reads per step to the ~4-16 of an occasional block reload. Values
// are t_masked verbatim — identical inputs, identical arithmetic,
// identical results to the uncached helpers.

__device__ __forceinline__ void anchor_of(
        int rows, int cols, float r, float c, int* ar, int* ac)
{
    int r0, c0, r1, c1;
    float w[4];
    bilinear_corners(rows, cols, r, c, &r0, &c0, &r1, &c1, w);
    *ar = r0 - 1;
    *ac = c0 - 1;
}

__device__ void load_blk(
        const float* __restrict__ T, int rows, int cols,
        int ar, int ac, float* blk)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            blk[i * 4 + j] = t_masked(T, rows, cols, ar + i, ac + j);
}

// NaN-aware bilinear T sample on the cached block (port of
// _sample_bilinear on masked T).
__device__ float sample_T_blk(
        const float* blk, int ar, int ac, int rows, int cols,
        float r, float c)
{
    int r0, c0, r1, c1;
    float w[4];
    bilinear_corners(rows, cols, r, c, &r0, &c0, &r1, &c1, w);
    float v[4];
    v[0] = blk[(r0 - ar) * 4 + (c0 - ac)];
    v[1] = blk[(r0 - ar) * 4 + (c1 - ac)];
    v[2] = blk[(r1 - ar) * 4 + (c0 - ac)];
    v[3] = blk[(r1 - ar) * 4 + (c1 - ac)];
    float total = 0.0f, wsum = 0.0f, fin_sum = 0.0f;
    int n_fin = 0;
    for (int k = 0; k < 4; ++k) {
        if (v[k] == v[k]) {
            fin_sum += v[k];
            n_fin += 1;
            if (w[k] > 0.0f) { total += v[k] * w[k]; wsum += w[k]; }
        }
    }
    if (wsum > 0.0f) return total / wsum;
    if (n_fin)      return fin_sum / n_fin;
    return nanf("");
}

// corner_grad on block indices (i, j in 1..2; neighbors 0..3).
__device__ __forceinline__ void corner_grad_blk(
        const float* blk, int i, int j, float* g_r, float* g_c)
{
    float ctr = blk[i * 4 + j];
    if (ctr != ctr) { *g_r = nanf(""); *g_c = nanf(""); return; }
    float nn = blk[(i - 1) * 4 + j];
    float ss = blk[(i + 1) * 4 + j];
    float ww = blk[i * 4 + j - 1];
    float ee = blk[i * 4 + j + 1];

    if (nn == nn && ss == ss)      *g_r = (ss - nn) * 0.5f;
    else if (ss == ss)             *g_r = ss - ctr;
    else if (nn == nn)             *g_r = ctr - nn;
    else                           *g_r = 0.0f;

    if (ww == ww && ee == ee)      *g_c = (ee - ww) * 0.5f;
    else if (ee == ee)             *g_c = ee - ctr;
    else if (ww == ww)             *g_c = ctr - ww;
    else                           *g_c = 0.0f;
}

// Bilinear masked-gradient sample on the cached block. The host
// samples the gr and gc fields independently, but a corner's gradient
// pair is NaN exactly when its center cell is invalid — identical NaN
// pattern, identical weights — so one pass with shared weights is
// equivalent.
__device__ bool descent_direction_blk(
        const float* blk, int ar, int ac, int rows, int cols,
        float r, float c, float* dr, float* dc)
{
    int r0, c0, r1, c1;
    float w[4];
    bilinear_corners(rows, cols, r, c, &r0, &c0, &r1, &c1, w);
    int ci[4], cj[4];
    ci[0] = r0 - ar; cj[0] = c0 - ac;
    ci[1] = r0 - ar; cj[1] = c1 - ac;
    ci[2] = r1 - ar; cj[2] = c0 - ac;
    ci[3] = r1 - ar; cj[3] = c1 - ac;
    float tot_r = 0.0f, tot_c = 0.0f, wsum = 0.0f;
    float fin_r = 0.0f, fin_c = 0.0f;
    int n_fin = 0;
    for (int k = 0; k < 4; ++k) {
        float gr_k, gc_k;
        corner_grad_blk(blk, ci[k], cj[k], &gr_k, &gc_k);
        if (gr_k == gr_k) {
            fin_r += gr_k; fin_c += gc_k; n_fin += 1;
            if (w[k] > 0.0f) {
                tot_r += gr_k * w[k];
                tot_c += gc_k * w[k];
                wsum += w[k];
            }
        }
    }
    float g_r, g_c;
    if (wsum > 0.0f)      { g_r = tot_r / wsum; g_c = tot_c / wsum; }
    else if (n_fin)      { g_r = fin_r / n_fin; g_c = fin_c / n_fin; }
    else                 return false;
    if (g_r != g_r || g_c != g_c) return false;
    float norm = sqrtf(g_r * g_r + g_c * g_c);
    if (norm < 1e-12f) return false;
    *dr = -g_r / norm;
    *dc = -g_c / norm;
    return true;
}

// Strictly-lower 8-neighbor with minimal T (port of the greedy branch
// of _discrete_descent_step). Returns false on a plateau -> host BFS.
__device__ bool lower_neighbor(
        const float* __restrict__ T, int rows, int cols,
        float r, float c, int* out_r, int* out_c)
{
    int ri = (int)(r + 0.5f);
    if (r < 0.0f) ri = 0;
    if (ri > rows - 1) ri = rows - 1;
    int ci = (int)(c + 0.5f);
    if (c < 0.0f) ci = 0;
    if (ci > cols - 1) ci = cols - 1;
    float t_here = (float)T[ri * cols + ci];
    float tol = 1e-6f * fabsf(t_here) + 1e-9f;
    float best_t = t_here - tol;
    int best_r = -1, best_c = -1;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int nr = ri + dr, nc = ci + dc;
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
            float t_n = (float)T[nr * cols + nc];
            if (t_n < (float)TRACE_INVALID && t_n < best_t) {
                best_t = t_n;
                best_r = nr;
                best_c = nc;
            }
        }
    }
    if (best_r < 0) return false;
    *out_r = best_r;
    *out_c = best_c;
    return true;
}

// status: 0 = reached a source, 1 = target unreachable,
//         2 = needs the host tracer (plateau BFS or step cap).
__global__ void eikonal_trace(
    const float* __restrict__ T,
    const double* __restrict__ src,    // n_src * 2 (row, col)
    const int n_src,
    const long long* __restrict__ targets,
    const int n_targets,
    const int rows,
    const int cols,
    const double step,
    const int max_steps,
    const int max_pts,
    double* __restrict__ out,          // n_targets * max_pts * 2
    int* __restrict__ out_len,
    int* __restrict__ status
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_targets) return;

    double* poly = out + (long long)tid * max_pts * 2;
    const float fstep = (float)step;
    int n_pts = 0;

    long long tgt = targets[tid];
    int tr = (int)(tgt / cols);
    int tc = (int)(tgt - (long long)tr * cols);
    if (T[tr * cols + tc] >= TRACE_INVALID) {
        out_len[tid] = 0;
        status[tid] = 1;
        return;
    }

    float pr = (float)tr, pc = (float)tc;
    poly[n_pts * 2] = pr; poly[n_pts * 2 + 1] = pc; n_pts += 1;

    // Two cached 4x4 blocks: A follows the position, B the Heun
    // midpoint (they alternate between two anchors while a step
    // straddles a cell boundary — two slots stop the thrash).
    float blkA[16], blkB[16];
    int kAr = -100000, kAc = -100000, kBr = -100000, kBc = -100000;
    int ar_, ac_;

    anchor_of(rows, cols, pr, pc, &ar_, &ac_);
    load_blk(T, rows, cols, ar_, ac_, blkA);
    kAr = ar_; kAc = ac_;
    float t_cur = sample_T_blk(blkA, kAr, kAc, rows, cols, pr, pc);

    // nearest source: distance + position (host nearest_src)
    float best_dist = 3e38f;
    for (int s = 0; s < n_src; ++s) {
        float dr_ = src[s * 2] - pr, dc_ = src[s * 2 + 1] - pc;
        float d2 = dr_ * dr_ + dc_ * dc_;
        if (d2 < best_dist) best_dist = d2;
    }
    best_dist = sqrtf(best_dist);
    int stall = 0;

    for (int it = 0; it < max_steps; ++it) {
        // nearest source
        float d2_best = 3e38f;
        float ns_r = src[0], ns_c = src[1];
        for (int s = 0; s < n_src; ++s) {
            float dr_ = src[s * 2] - pr, dc_ = src[s * 2 + 1] - pc;
            float d2 = dr_ * dr_ + dc_ * dc_;
            if (d2 < d2_best) {
                d2_best = d2;
                ns_r = src[s * 2];
                ns_c = src[s * 2 + 1];
            }
        }
        float d = sqrtf(d2_best);
        if (d <= 1.0f) {
            if (n_pts < max_pts) {
                poly[n_pts * 2] = ns_r;
                poly[n_pts * 2 + 1] = ns_c;
                n_pts += 1;
            }
            out_len[tid] = n_pts;
            status[tid] = 0;
            return;
        }

        // hop_reason: 0 = none (gradient move), 1 = shock-line stall
        // (host: stall reset, no stall bookkeeping), 2 = degenerate
        // gradient (host: stall untouched), 3 = wall graze / uphill-in-T
        // (host: falls through to the normal progress bookkeeping).
        int hop_reason = 0;
        if (stall >= 20) {
            hop_reason = 1;                   // shock-line oscillation
        } else {
            float d1r, d1c;
            anchor_of(rows, cols, pr, pc, &ar_, &ac_);
            if (ar_ != kAr || ac_ != kAc) {
                load_blk(T, rows, cols, ar_, ac_, blkA);
                kAr = ar_; kAc = ac_;
            }
            if (!descent_direction_blk(blkA, kAr, kAc, rows, cols,
                                       pr, pc, &d1r, &d1c)) {
                hop_reason = 2;               // degenerate gradient
            } else {
                // Heun / RK2
                float mr = pr + fstep * d1r;
                float mc = pc + fstep * d1c;
                if (mr < 0.0f) mr = 0.0f;
                if (mr > rows - 1.0f) mr = rows - 1.0f;
                if (mc < 0.0f) mc = 0.0f;
                if (mc > cols - 1.0f) mc = cols - 1.0f;
                float d2r, d2c, sr_, sc_;
                anchor_of(rows, cols, mr, mc, &ar_, &ac_);
                const float* mb;
                int mar, mac;
                if (ar_ == kAr && ac_ == kAc) {
                    mb = blkA; mar = kAr; mac = kAc;
                } else {
                    if (ar_ != kBr || ac_ != kBc) {
                        load_blk(T, rows, cols, ar_, ac_, blkB);
                        kBr = ar_; kBc = ac_;
                    }
                    mb = blkB; mar = kBr; mac = kBc;
                }
                if (!descent_direction_blk(mb, mar, mac, rows, cols,
                                           mr, mc, &d2r, &d2c)) {
                    sr_ = d1r; sc_ = d1c;
                } else {
                    sr_ = d1r + d2r;
                    sc_ = d1c + d2c;
                    float norm = sqrtf(sr_ * sr_ + sc_ * sc_);
                    if (norm < 1e-6f) { sr_ = d1r; sc_ = d1c; }
                    else             { sr_ /= norm; sc_ /= norm; }
                }
                float cr_ = pr + fstep * sr_;
                float cc_ = pc + fstep * sc_;
                if (cr_ < 0.0f) cr_ = 0.0f;
                if (cr_ > rows - 1.0f) cr_ = rows - 1.0f;
                if (cc_ < 0.0f) cc_ = 0.0f;
                if (cc_ > cols - 1.0f) cc_ = cols - 1.0f;
                anchor_of(rows, cols, cr_, cc_, &ar_, &ac_);
                const float* cb;
                int car, cac;
                if (ar_ == kAr && ac_ == kAc) {
                    cb = blkA; car = kAr; cac = kAc;
                } else if (ar_ == kBr && ac_ == kBc) {
                    cb = blkB; car = kBr; cac = kBc;
                } else {
                    // next position lands here: refresh slot A
                    load_blk(T, rows, cols, ar_, ac_, blkA);
                    kAr = ar_; kAc = ac_;
                    cb = blkA; car = kAr; cac = kAc;
                }
                float t_cand = sample_T_blk(cb, car, cac, rows, cols,
                                             cr_, cc_);
                int rr = (int)(cr_ + 0.5f);
                int rc = (int)(cc_ + 0.5f);
                if (T[rr * cols + rc] >= TRACE_INVALID
                        || t_cand != t_cand || t_cand > t_cur) {
                    hop_reason = 3;           // wall graze / uphill in T
                } else {
                    pr = cr_; pc = cc_; t_cur = t_cand;
                    if (n_pts >= max_pts) {   // defensive; cannot happen
                        out_len[tid] = n_pts;
                        status[tid] = 2;
                        return;
                    }
                    poly[n_pts * 2] = pr;
                    poly[n_pts * 2 + 1] = pc;
                    n_pts += 1;
                }
            }
        }

        if (hop_reason != 0) {
            int hr, hc;
            if (!lower_neighbor(T, rows, cols, pr, pc, &hr, &hc)) {
                out_len[tid] = n_pts;         // plateau -> host BFS
                status[tid] = 2;
                return;
            }
            pr = (float)hr;
            pc = (float)hc;
            t_cur = (float)T[hr * cols + hc];
            if (n_pts >= max_pts) {
                out_len[tid] = n_pts;
                status[tid] = 2;
                return;
            }
            poly[n_pts * 2] = pr;
            poly[n_pts * 2 + 1] = pc;
            n_pts += 1;
        }

        // progress bookkeeping (host: best_dist / stall, per hop reason)
        float d2n = 3e38f;
        for (int s = 0; s < n_src; ++s) {
            float dr_ = src[s * 2] - pr, dc_ = src[s * 2 + 1] - pc;
            float dd = dr_ * dr_ + dc_ * dc_;
            if (dd < d2n) d2n = dd;
        }
        float dn = sqrtf(d2n);
        if (hop_reason == 1) {
            stall = 0;
            if (dn < best_dist) best_dist = dn;
        } else if (hop_reason == 2) {
            if (dn < best_dist) best_dist = dn;
        } else if (dn < best_dist - 0.25f * fstep) {
            best_dist = dn;
            stall = 0;
        } else {
            stall += 1;
        }
    }

    out_len[tid] = n_pts;                     // step cap -> host tracer
    status[tid] = 2;
}

}  // extern "C"
"""


# ============================================================================
# Kernel cache
# ============================================================================

_eik_kernel_cache = {}


def _get_eik_kernel(name: str, source: str):
    """Get a compiled CuPy RawKernel, compiling lazily on first use."""
    if name not in _eik_kernel_cache:
        _eik_kernel_cache[name] = cp.RawKernel(source, name)
    return _eik_kernel_cache[name]


def _get_sweep_kernel(tile: int):
    """Sweep kernel variant for tile size B (compile-time constant)."""
    name = f"fim_sweep@{tile}"
    if name not in _eik_kernel_cache:
        source = _FIM_SWEEP_KERNEL.replace("@B@", str(tile))
        _eik_kernel_cache[name] = cp.RawKernel(source, "fim_sweep")
    return _eik_kernel_cache[name]


def _get_target_check_kernel(tile: int):
    """Early-exit check kernel variant for tile size B."""
    name = f"fim_target_check@{tile}"
    if name not in _eik_kernel_cache:
        source = _FIM_TARGET_CHECK_KERNEL.replace("@B@", str(tile))
        _eik_kernel_cache[name] = cp.RawKernel(source, "fim_target_check")
    return _eik_kernel_cache[name]


def _get_sweep2_kernel(tile: int):
    """Second-order refinement kernel variant for tile size B."""
    name = f"fim_sweep2@{tile}"
    if name not in _eik_kernel_cache:
        source = _FIM_SWEEP2_KERNEL.replace("@B@", str(tile))
        _eik_kernel_cache[name] = cp.RawKernel(source, "fim_sweep2")
    return _eik_kernel_cache[name]


def _get_freeze2_kernel(tile: int):
    """Upwind-structure freeze kernel (same source as fim_sweep2)."""
    name = f"fim_freeze2@{tile}"
    if name not in _eik_kernel_cache:
        source = _FIM_SWEEP2_KERNEL.replace("@B@", str(tile))
        _eik_kernel_cache[name] = cp.RawKernel(source, "fim_freeze2")
    return _eik_kernel_cache[name]


def _get_demote2_kernel(tile: int):
    """Stall-degrade kernel (same source as fim_sweep2)."""
    name = f"fim_demote2@{tile}"
    if name not in _eik_kernel_cache:
        source = _FIM_SWEEP2_KERNEL.replace("@B@", str(tile))
        _eik_kernel_cache[name] = cp.RawKernel(source, "fim_demote2")
    return _eik_kernel_cache[name]


# ============================================================================
# Host-side setup
# ============================================================================

def _slowness_field(raster: np.ndarray, ignore_max: bool) -> np.ndarray:
    """Convert the cost raster to a float32 slowness field.

    Impassable cells become ``1e30``: the uint16 sentinel 65535 (only with
    ``ignore_max=True`` — mirroring ``_common_gpu_setup``), respectively
    the float ``>= 1e30`` / non-finite forbidden convention (always, as in
    ``_float_raster_source``).
    """
    if raster.ndim != 2:
        raise ValueError(f"raster must be 2D, got shape {raster.shape}")
    if np.issubdtype(raster.dtype, np.floating):
        c = np.ascontiguousarray(raster, dtype=np.float32).copy()
        bad = ~np.isfinite(c) | (c >= FINITE_LIMIT)
        c[bad] = UNREACHED
        if (c[~bad] < 0).any():
            raise ValueError("negative cost values are not supported")
        return c
    c = raster.astype(np.float32)
    if ignore_max:
        c[raster == np.iinfo(np.uint16).max] = UNREACHED
    return np.ascontiguousarray(c)


def _resolve_eps(eps_rel: float, eps_abs: Optional[float],
                 slowness: np.ndarray) -> Tuple[float, float]:
    """Default eps_abs to 1e-6 * mean passable cell value (plan 3.2)."""
    if eps_abs is None:
        valid = slowness[slowness < FINITE_LIMIT]
        c_mean = float(valid.mean()) if valid.size else 1.0
        eps_abs = 1e-6 * max(c_mean, 1.0)
    if eps_rel < 0 or eps_abs < 0:
        raise ValueError("eps_rel and eps_abs must be >= 0")
    return float(eps_rel), float(eps_abs)


def _device_slowness(raster: np.ndarray, ignore_max: bool):
    """Slowness field built ON the device (performance plan phase 1).

    Uploads the raw raster (half the bytes for uint16) and converts
    with device elementwise ops — replaces the full-grid host pass of
    :func:`_slowness_field`, which stays as the host reference and for
    the small windows disk init inspects.
    """
    if np.issubdtype(raster.dtype, np.floating):
        d_raw = cp.asarray(
            np.ascontiguousarray(raster, dtype=np.float32)).ravel()
        bad = ~cp.isfinite(d_raw) | (d_raw >= FINITE_LIMIT)
        if bool(((d_raw < 0) & ~bad).any()):
            raise ValueError("negative cost values are not supported")
        return cp.where(bad, cp.float32(UNREACHED), d_raw)
    d_raw = cp.asarray(np.ascontiguousarray(raster)).ravel()
    d_c = d_raw.astype(cp.float32)
    if ignore_max:
        d_c[d_raw == np.iinfo(np.uint16).max] = cp.float32(UNREACHED)
    return d_c


def _resolve_eps_device(eps_rel: float, eps_abs: Optional[float],
                        d_c) -> Tuple[float, float]:
    """Device-reduction twin of :func:`_resolve_eps`."""
    if eps_abs is None:
        mask = d_c < FINITE_LIMIT
        n = int(cp.count_nonzero(mask))
        c_mean = (float(cp.where(mask, d_c, cp.float32(0.0)).sum()) / n
                  if n else 1.0)
        eps_abs = 1e-6 * max(c_mean, 1.0)
    if eps_rel < 0 or eps_abs < 0:
        raise ValueError("eps_rel and eps_abs must be >= 0")
    return float(eps_rel), float(eps_abs)


def _validate_sources(source_indices, raster: np.ndarray,
                      ignore_max: bool) -> np.ndarray:
    """Normalize + validate source indices (raster-based — no full
    slowness pass, performance plan phase 1).

    Out-of-range indices raise (programming error, loud). Sources on
    impassable cells are dropped (data condition, mirroring the discrete
    backends' unreachable-everywhere behavior); the caller receives the
    remaining valid sources — possibly empty.
    """
    src = np.atleast_1d(np.asarray(source_indices)).astype(np.int64)
    n_pixels = raster.size
    if src.size == 0:
        raise ValueError("source_indices must not be empty")
    if (src < 0).any() or (src >= n_pixels).any():
        raise ValueError(
            f"source index out of range [0, {n_pixels}): "
            f"{src[(src < 0) | (src >= n_pixels)].tolist()}")
    vals = raster.ravel()[src]
    if np.issubdtype(raster.dtype, np.floating):
        passable = np.isfinite(vals) & (vals < FINITE_LIMIT)
    elif ignore_max:
        passable = vals != np.iinfo(np.uint16).max
    else:
        passable = np.ones(src.size, dtype=bool)
    return src[passable]


def _disk_init_values(
        sources: np.ndarray,
        raster: np.ndarray,
        ignore_max: bool = True,
        r0: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic disk initialization around point sources (plan 3.3).

    For cells within radius ``r0`` of a source, seed ``T = c_src * r``
    (exact for locally-constant cost) and freeze them, removing the O(h)
    point-source singularity error. The disk is only applied where the
    analysis holds: all in-bounds cells within the disk must share the
    source cell's value (seeding a too-low T on cheaper-elsewhere rasters
    could never be corrected — updates are monotone non-increasing).
    Sources closer than ``2 * r0`` to another source skip the disk.

    Consumes the raw raster and converts only the small windows it
    inspects (phase 1: no full-grid host slowness pass).

    Returns (flat_indices, values) of the seeded cells (sources included
    with value 0).
    """
    rows, cols = raster.shape
    src_r, src_c = np.divmod(sources, cols)

    # Pairwise proximity: sources within 2*r0 of another source get no disk
    dr = src_r[:, None] - src_r[None, :]
    dc = src_c[:, None] - src_c[None, :]
    d2 = dr * dr + dc * dc
    np.fill_diagonal(d2, np.iinfo(np.int64).max)
    too_close = (d2 <= (2.0 * r0) ** 2).any(axis=1)

    # Disk offsets within radius r0
    rr = int(np.floor(r0))
    off = np.arange(-rr, rr + 1)
    orr, occ = np.meshgrid(off, off, indexing="ij")
    dist = np.sqrt(orr.astype(np.float64) ** 2 + occ ** 2)
    in_disk = dist <= r0
    orr, occ, dist = orr[in_disk], occ[in_disk], dist[in_disk]

    idx_out = []
    val_out = []
    for k in range(len(sources)):
        if too_close[k]:
            continue
        r_lo = max(int(src_r[k]) - rr, 0)
        c_lo = max(int(src_c[k]) - rr, 0)
        win = _slowness_field(
            np.ascontiguousarray(
                raster[r_lo:int(src_r[k]) + rr + 1,
                       c_lo:int(src_c[k]) + rr + 1]), ignore_max)
        cr = src_r[k] + orr
        cc = src_c[k] + occ
        inb = (cr >= 0) & (cr < rows) & (cc >= 0) & (cc < cols)
        cr, cc, d = cr[inb], cc[inb], dist[inb]
        c_src = win[src_r[k] - r_lo, src_c[k] - c_lo]
        if not (win[cr - r_lo, cc - c_lo] == c_src).all():
            continue  # cost not locally constant — analytic cone invalid
        idx_out.append(cr * cols + cc)
        val_out.append((c_src * d).astype(np.float32))

    if idx_out:
        return (np.concatenate(idx_out),
                np.concatenate(val_out))
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)


def _setup_solve(raster, source_indices, ignore_max, eps_rel, eps_abs,
                 disk_init, disk_radius):
    """Shared host+device setup for both solvers.

    Phase-1 layout: the full-grid slowness conversion happens on the
    device (raw upload = half the bytes for uint16; no 2-pass host
    copy); host work touches only source cells and disk windows.

    Returns None if no valid source remains (caller returns the
    all-unreachable field), else the device context tuple.
    """
    if raster.ndim != 2:
        raise ValueError(f"raster must be 2D, got shape {raster.shape}")
    rows, cols = raster.shape
    sources = _validate_sources(source_indices, raster, ignore_max)
    if sources.size == 0:
        return None

    d_c = _device_slowness(raster, ignore_max)
    eps_rel, eps_abs = _resolve_eps_device(eps_rel, eps_abs, d_c)

    seed_idx = sources
    seed_val = np.zeros(sources.size, dtype=np.float32)
    if disk_init:
        d_idx, d_val = _disk_init_values(sources, raster, ignore_max,
                                         r0=disk_radius)
        if d_idx.size:
            seed_idx = np.concatenate([seed_idx, d_idx])
            seed_val = np.concatenate([seed_val, d_val])
            # A cell can appear in several disks (or be a source): keep min
            order = np.argsort(seed_val, kind="stable")
            seed_idx, seed_val = seed_idx[order], seed_val[order]
            uniq, first = np.unique(seed_idx, return_index=True)
            seed_idx, seed_val = uniq, seed_val[first]

    d_t = cp.full(rows * cols, UNREACHED, dtype=cp.float32)
    d_t[cp.asarray(seed_idx)] = cp.asarray(seed_val)
    d_frozen = cp.zeros(rows * cols, dtype=cp.uint8)
    d_frozen[cp.asarray(seed_idx)] = 1

    return (d_c, d_t, d_frozen, seed_idx, eps_rel, eps_abs, rows, cols)


# NOTE (performance plan phase 5, REJECTED BY MEASUREMENT
# 2026-08-07): coarse-to-fine supersolution seeding (4x max-pooled +
# dilated coarse solve prolongated as an upper-bound init) was
# implemented and benchmarked here. It cut outer passes only ~20%
# (192->152 at 1536^2) — the bound's pool-scale slack means the
# correction wave still propagates across the whole grid — while the
# coarse solve + full-tile settling added ~110 ms: a 10x NET LOSS at
# every size tested, plus above-slack field deviations on barrier
# rasters. Do not reintroduce without a fundamentally tighter bound
# (details: FINDINGS section 12).


# ============================================================================
# Solvers
# ============================================================================

def eikonal_raster_gpu_naive(
        raster: np.ndarray,
        source_indices,
        ignore_max: bool = True,
        eps_rel: float = 1e-6,
        eps_abs: Optional[float] = None,
        max_iterations: Optional[int] = None,
        disk_init: bool = True,
        disk_radius: float = 3.0,
        return_iterations: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Naive full-grid Jacobi eikonal solver (reference implementation).

    Every iteration updates every cell (double-buffered), so information
    travels one cell per iteration — O(path length) iterations of O(n)
    work. Kept as the block-FIM test oracle; do not use for production.

    Returns the T field, float32, shape = raster.shape, 1e30 = unreachable.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU not available. Install cupy with CUDA support: "
            "pip install cupy-cuda12x")

    ctx = _setup_solve(raster, source_indices, ignore_max, eps_rel,
                       eps_abs, disk_init, disk_radius)
    rows, cols = raster.shape
    if ctx is None:
        return _unreachable_result(rows, cols, return_iterations)
    d_c, d_ta, d_frozen, _, eps_rel, eps_abs, rows, cols = ctx

    if max_iterations is None:
        max_iterations = 8 * (rows + cols)

    d_tb = d_ta.copy()
    d_changed = cp.zeros(1, dtype=cp.int32)
    kernel = _get_eik_kernel("eikonal_jacobi", _JACOBI_KERNEL)
    n = rows * cols
    tpb = 256
    blocks = (n + tpb - 1) // tpb

    iterations = 0
    while True:
        if iterations >= max_iterations:
            raise RuntimeError(
                f"naive eikonal solver hit the iteration cap "
                f"({max_iterations}) without converging — raster cost "
                f"contrast too extreme or paths longer than the cap "
                f"assumes. Raise max_iterations or use the block-FIM "
                f"solver.")
        d_changed[0] = 0
        kernel((blocks,), (tpb,),
               (d_c, d_ta, d_tb, d_frozen,
                np.int32(rows), np.int32(cols),
                np.float32(eps_abs), np.float32(eps_rel), d_changed))
        iterations += 1
        d_ta, d_tb = d_tb, d_ta
        if int(d_changed[0]) == 0:
            break

    t_field = d_ta.get().reshape(rows, cols)
    if return_iterations:
        return t_field, iterations
    return t_field


def eikonal_raster_gpu(
        raster: np.ndarray,
        source_indices,
        ignore_max: bool = True,
        tile: int = 16,
        n_inner: Optional[int] = None,
        eps_rel: float = 1e-6,
        eps_abs: Optional[float] = None,
        max_outer_iterations: Optional[int] = None,
        disk_init: bool = True,
        disk_radius: float = 3.0,
        sweep_blocks: Optional[int] = None,
        return_iterations: bool = False,
        return_stats: bool = False,
        return_device: bool = False,
        target_index: Optional[int] = None,
        order: int = 1,
        return_trace_field: bool = False,
        download: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, int],
           Tuple[np.ndarray, dict]]:
    """Block-FIM eikonal solve on the GPU (plan section 3.2).

    Parameters:
        raster: 2D cost raster — uint16 (65535 = impassable with
            ignore_max=True) or float32 (>= 1e30 / non-finite = impassable).
        source_indices: Flat cell index or array of indices; T = 0 there.
            Multiple sources give the pointwise-min field.
        ignore_max: uint16 sentinel handling (float rasters always use the
            >= 1e30 forbidden convention).
        tile: Tile edge length B (block = B x B threads).
        n_inner: Relaxation sweeps per tile per pass (default: 2*B —
            information can cross the whole tile and settle; measured
            ~30%% fewer outer passes than n_inner=B at equal accuracy).
        eps_rel / eps_abs: Convergence threshold — a cell counts as changed
            while its improvement exceeds ``eps_abs + eps_rel * T``.
            eps_abs defaults to 1e-6 * mean passable cell value.
        max_outer_iterations: Hard cap on host-loop passes (default
            ``64 * max(tiles_r, tiles_c)``); RuntimeError on cap — a
            silently unconverged field is never returned.
        disk_init: Seed the analytic cone ``T = c_src * r`` within
            ``disk_radius`` cells of each source where the cost is locally
            constant, and freeze those cells (mitigates the point-source
            singularity error of the first-order scheme).
        disk_radius: Disk radius in cells (default 3). Note for
            convergence studies: the singularity pollutes the field along
            diagonals with an O(sqrt(h)) contribution unless the exact
            disk is fixed in *physical* units — i.e. scale disk_radius
            with the resolution when comparing refinements.
        return_iterations: Also return the number of outer passes.
        return_stats: Return ``(T, stats)`` instead, where stats holds
            outer_passes, tile_sweeps, cell_updates (= sweeps * tile^2 *
            n_inner) and updates_per_cell — the update-redundancy metric
            of value iteration vs ordered methods (cf. Fu et al. 2011,
            Table 3.4). Supersedes return_iterations.
        return_device: Additionally return the flat float32 device
            array of the T field (appended last) — feeds
            :func:`trace_paths_gpu` without a re-upload. None when the
            solve short-circuits to all-unreachable.
        target_index: Optional flat cell index enabling **targeted
            early exit** (increment 2): the solve stops as soon as the
            target's tile is converged AND the min T over all active
            tiles (cells + halo) is >= T[target] — monotone updates
            guarantee no later pass can improve the target beyond the
            convergence tolerance. T[target] and every cell at
            ``T <= T[target]`` (the region a descent trace visits)
            carry the solver's ordinary eps guarantee; cells strictly
            above that level set may hold unconverged upper bounds.
            Do not reuse the field for other targets. Ignored with
            order=2 (the refinement needs the full field).
        order: 1 (default) — first-order Godunov; 2 — second-order
            one-sided upwind refinement (increment 2): after the
            first-order solve converges, a deterministic double-buffered
            Jacobi stage re-solves with per-axis second-order
            differences where the second upwind cell exists, is finite
            and T2 <= T1 (first-order fallback elsewhere — near shocks,
            sources and barriers). Roughly doubles the solve time;
            error on smooth fields drops by an order of magnitude.
            NOTE: refined fields are for COSTS — they do not inherit
            the first-order field's descent-connectivity (on rough
            real-world rasters genuine local minima appear near cost
            shocks; measured). Trace on the first-order field via
            ``return_trace_field``.
        return_trace_field: Append ``(t_trace, d_trace)`` — the field
            the path tracer should consume. For order=1 this is the
            solved field itself (no copies); for order=2 it is the
            preserved first-order field, whose Godunov fixed point is
            descent-connected to the sources by construction.
        download: When False, skip the full-field D2H transfer — every
            host-array slot in the return (t_field, trace host field)
            is None; only device arrays are populated. For single-pair
            flows that read T[target] as a device scalar and trace on
            the device (the D2H was ~20 ms at 4096²). Callers download
            on demand via ``d_t.get()``.

    Returns:
        T field, float32, shape = raster.shape, 1e30 = unreachable
        (finite check: ``T < 1e29``). With return_iterations=True:
        ``(T, n_outer)``; with return_stats=True: ``(T, stats_dict)``;
        with return_device=True the device array is appended last.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU not available. Install cupy with CUDA support: "
            "pip install cupy-cuda12x")
    if tile < 4 or tile > 32:
        raise ValueError(f"tile must be in [4, 32], got {tile}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if order == 2:
        target_index = None      # refinement needs the full field
    if n_inner is None:
        n_inner = 2 * tile

    ctx = _setup_solve(raster, source_indices, ignore_max, eps_rel,
                       eps_abs, disk_init, disk_radius)
    rows, cols = raster.shape
    if ctx is None:
        if return_stats:
            t_field = _unreachable_result(rows, cols, False)
            result = (t_field, dict(outer_passes=0, tile_sweeps=0,
                                    cell_updates=0, updates_per_cell=0.0))
        else:
            result = _unreachable_result(rows, cols, return_iterations)
            if not isinstance(result, tuple):
                result = (result,)
        if return_device:
            result = (*result, None)
        if return_trace_field:
            result = (*result, (result[0], None))
        return result if len(result) > 1 else result[0]
    d_c, d_t, d_frozen, seed_idx, eps_rel, eps_abs, rows, cols = ctx

    tiles_r = (rows + tile - 1) // tile
    tiles_c = (cols + tile - 1) // tile
    n_tiles = tiles_r * tiles_c
    if max_outer_iterations is None:
        max_outer_iterations = 64 * max(tiles_r, tiles_c)

    # Tiles containing seeded cells start active
    seed_tiles = np.unique((seed_idx // cols) // tile * tiles_c
                           + (seed_idx % cols) // tile)
    d_active = cp.zeros(n_tiles, dtype=cp.uint8)
    d_active[cp.asarray(seed_tiles)] = 1
    d_boundary = cp.zeros(n_tiles, dtype=cp.uint8)

    # Ping-pong active lists, built on-device by the activate kernel: the
    # host only reads the count every `sync_every` passes (sweeps with an
    # empty list are near-free — every block exits on the count check).
    d_list_a = cp.zeros(n_tiles, dtype=cp.int32)
    d_list_b = cp.zeros(n_tiles, dtype=cp.int32)
    d_count_a = cp.zeros(1, dtype=cp.int32)
    d_count_b = cp.zeros(1, dtype=cp.int32)
    d_stats = cp.zeros(2, dtype=cp.uint64)
    d_list_a[:seed_tiles.size] = cp.asarray(seed_tiles.astype(np.int32))
    d_count_a[0] = np.int32(seed_tiles.size)

    sweep = _get_sweep_kernel(tile)
    activate = _get_eik_kernel("fim_activate", _FIM_ACTIVATE_KERNEL)
    act_tpb = 256
    act_blocks = (n_tiles + act_tpb - 1) // act_tpb
    sync_every = 8
    if sweep_blocks is None:
        n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
        sweep_blocks = max(16 * n_sm, 64)
    sweep_blocks = min(n_tiles, int(sweep_blocks))

    check_target = None
    if target_index is not None:
        target_index = int(target_index)
        if not (0 <= target_index < rows * cols):
            raise ValueError(
                f"target_index {target_index} outside the raster "
                f"({rows}x{cols})")
        target_tile = ((target_index // cols) // tile * tiles_c
                       + (target_index % cols) // tile)
        check_kernel = _get_target_check_kernel(tile)
        d_check = cp.zeros(2, dtype=cp.int32)
        _int_inf = int(np.array(3.0e38, np.float32).view(np.int32))

        def check_target():
            """True when no future pass can improve T[target]."""
            t_tgt = float(d_t[target_index])
            if t_tgt >= FINITE_LIMIT:
                return False              # wave has not arrived yet:
            # cheap scalar gate above keeps far-target solves at
            # near-zero overhead — the min kernel only runs once the
            # wave has reached the target.
            d_check[0] = _int_inf
            d_check[1] = 0
            check_kernel((sweep_blocks,), (tile, tile),
                         (d_t, d_list_a, d_count_a,
                          np.int32(rows), np.int32(cols),
                          np.int32(tiles_c), np.int32(target_tile),
                          d_check))
            res = d_check.get()
            if res[1]:
                return False              # target tile still active
            min_active = float(res[:1].view(np.float32)[0])
            return min_active >= t_tgt

    def issue_pass(list_in, count_in, list_out, count_out):
        d_boundary.fill(0)
        sweep((sweep_blocks,), (tile, tile),
              (d_c, d_t, d_frozen, list_in, count_in,
               np.int32(rows), np.int32(cols), np.int32(tiles_c),
               np.int32(n_inner),
               np.float32(eps_abs), np.float32(eps_rel),
               d_active, d_boundary, count_out, d_stats))
        activate((act_blocks,), (act_tpb,),
                 (d_active, d_boundary, np.int32(tiles_r),
                  np.int32(tiles_c), list_out, count_out))

    def raise_cap(n_outer, n_active):
        c_valid = d_c[d_c < FINITE_LIMIT]
        contrast = (float(c_valid.max()) / max(float(c_valid.min()),
                                               1e-12)
                    if c_valid.size else float("nan"))
        raise RuntimeError(
            f"block-FIM hit the outer-iteration cap "
            f"({max_outer_iterations}, ran {n_outer}) with {n_active} "
            f"tiles still active — raster cost contrast (max/min ~ "
            f"{contrast:.3g}) is likely too extreme for value "
            f"iteration. Raise max_outer_iterations or use the "
            f"discrete 'raster_gpu' backend for this raster.")

    # Python-side launch overhead dominates the host loop (~5 CuPy calls
    # per pass), so the whole sync window is captured once into a CUDA
    # graph and replayed — one launch per sync_every passes. Passes after
    # convergence inside a window are no-ops (empty active list). Plain
    # per-pass loop as fallback (no capture support, or tiny caps as used
    # by the cap tests).
    graph = None
    if max_outer_iterations >= sync_every:
        try:
            capture_stream = cp.cuda.Stream(non_blocking=True)
            with capture_stream:
                capture_stream.begin_capture()
                la, ca = d_list_a, d_count_a
                lb, cb = d_list_b, d_count_b
                for _ in range(sync_every):
                    issue_pass(la, ca, lb, cb)
                    la, lb = lb, la
                    ca, cb = cb, ca
                graph = capture_stream.end_capture()
        except Exception:
            graph = None   # fall back to the plain loop

    n_outer = 0
    if graph is not None:
        while True:
            graph.launch()             # sync_every passes, one launch
            n_outer += sync_every
            n_active = int(d_count_a[0])   # single tiny D2H per window
            if n_active == 0:
                break
            if check_target is not None and check_target():
                break                  # exact targeted early exit
            if n_outer >= max_outer_iterations:
                raise_cap(n_outer, n_active)
    else:
        while True:
            issue_pass(d_list_a, d_count_a, d_list_b, d_count_b)
            d_list_a, d_list_b = d_list_b, d_list_a
            d_count_a, d_count_b = d_count_b, d_count_a
            n_outer += 1
            if (n_outer % sync_every == 0
                    or n_outer >= max_outer_iterations):
                n_active = int(d_count_a[0])
                if n_active == 0:
                    break
                if check_target is not None and check_target():
                    break              # exact targeted early exit
                if n_outer >= max_outer_iterations:
                    raise_cap(n_outer, n_active)

    # ------------------------------------------------------------------
    # Stage 2 (order=2): second-order refinement from the converged
    # first-order field. Deterministic double-buffered Jacobi — see the
    # kernel comment for why the chaotic machinery cannot be reused.
    # ------------------------------------------------------------------
    d_t_o1 = None
    skip_refine = False
    if order == 2:
        # Roughness guard (phase 4). The refinement's validated domain
        # is smooth passable cost fields with at most isolated
        # barriers; outside it the mixed-order fixed point misbehaves
        # (measured: up to 10% undershoot + seconds of stall churn on
        # the real planning raster, FINDINGS 11.2). Two cheap raster
        # metrics cover the two measured failure classes: cell-scale
        # noise shows up as a high density of large relative cost
        # jumps between neighbors; obstacle fields shape the solution
        # with diffraction shocks that are invisible in cost jumps but
        # scale with the forbidden fraction.
        c2d = d_c.reshape(rows, cols)
        fin = c2d < FINITE_LIMIT
        n_passable = int(cp.count_nonzero(fin))
        jr = (fin[1:, :] & fin[:-1, :]
              & (cp.abs(c2d[1:, :] - c2d[:-1, :])
                 > 0.5 * cp.minimum(c2d[1:, :], c2d[:-1, :])))
        jc = (fin[:, 1:] & fin[:, :-1]
              & (cp.abs(c2d[:, 1:] - c2d[:, :-1])
                 > 0.5 * cp.minimum(c2d[:, 1:], c2d[:, :-1])))
        n_edges = int(jr.sum()) + int(jc.sum())
        edge_frac = n_edges / max(n_passable, 1)
        forb_frac = 1.0 - n_passable / float(rows * cols)
        if edge_frac > 0.02 or forb_frac > 0.02:
            import warnings
            warnings.warn(
                f"order=2: the surface is outside the refinement's "
                f"validated domain (cost-jump edge density "
                f"{edge_frac:.1%}, forbidden fraction {forb_frac:.1%};"
                f" thresholds 2%) — second-order accuracy is only "
                f"established on piecewise-smooth fields with isolated"
                f" barriers. Returning the first-order solution.",
                RuntimeWarning, stacklevel=2)
            skip_refine = True
    if order == 2 and not skip_refine:
        # Upwind structure frozen once from the converged first-order
        # field (see fim_freeze2) — a live switch never settles.
        d_codes = cp.zeros(rows * cols, dtype=cp.uint8)
        freeze = _get_freeze2_kernel(tile)
        frz_tpb = 256
        freeze(((rows * cols + frz_tpb - 1) // frz_tpb,), (frz_tpb,),
               (d_t, d_c, np.int32(rows), np.int32(cols), d_codes))
        d_t_o1 = d_t.copy()    # descent-connected field (tracing +
        #                        the budget-exhaust fallback below)
        sweep2 = _get_sweep2_kernel(tile)
        n_inner2 = tile
        # float32 termination floor: near the mixed-order fixed point
        # the iteration wanders in a ULP-noise ball of ~6e-5 relative
        # amplitude (the quadratic cascade amplifies rounding jitter;
        # measured at 401^2 — the field is correct, the raw residual
        # just never reaches eps_rel=1e-6). A 3e-4 relative floor
        # terminates on the noise ball and stays 5-50x below the
        # discretization error the refinement corrects.
        eps2_rel = max(eps_rel, 3e-4)
        # Pass budget (phase 4): smooth fields converge in about the
        # stage-1 pass count; a surface that needs many times that is
        # in stall-degrade territory and gets the first-order field
        # back (with a warning) instead of seconds of churn.
        budget2 = min(max_outer_iterations, max(6 * n_outer, 96))
        d_t2 = d_t.copy()
        d_active.fill(1)
        d_list_a[:n_tiles] = cp.arange(n_tiles, dtype=cp.int32)
        d_count_a[0] = np.int32(n_tiles)

        def issue_pass2(t_in, t_out, list_in, count_in,
                        list_out, count_out):
            d_boundary.fill(0)
            cp.copyto(t_out, t_in)     # inactive tiles stay intact
            sweep2((sweep_blocks,), (tile, tile),
                   (d_c, t_in, t_out, d_frozen, d_codes,
                    list_in, count_in,
                    np.int32(rows), np.int32(cols), np.int32(tiles_c),
                    np.int32(n_inner2),
                    np.float32(eps_abs), np.float32(eps2_rel),
                    d_active, d_boundary, count_out, d_stats))
            activate((act_blocks,), (act_tpb,),
                     (d_active, d_boundary, np.int32(tiles_r),
                      np.int32(tiles_c), list_out, count_out))

        def budget_exhausted(n_outer2, n_active):
            """Phase-4 bounded bail: the refinement did not settle
            within its pass budget — warn and fall back to the
            first-order field instead of churning (was a hard raise;
            the budget keeps order=2 <= ~2x the order-1 time)."""
            import warnings
            warnings.warn(
                f"order=2 refinement did not settle within its pass "
                f"budget ({budget2}, ran {n_outer2}; {n_active} tiles "
                f"still active) — the surface is too irregular for the "
                f"second-order fixed point; returning the first-order "
                f"solution.", RuntimeWarning, stacklevel=3)

        demote = _get_demote2_kernel(tile)
        best_active = n_tiles + 1
        stagnant = 0
        demotions = 0

        def maybe_demote(n_active, n_outer2):
            """Stall degrade, active only in the SECOND half of the
            pass budget: a plateauing active count early on is normal
            correction propagation on smooth fields (demoting there
            froze legitimately-converging tiles and cost the observed
            convergence order at 401^2). Past half budget a stall of 3
            windows demotes the stuck tiles to first order; the next
            one freezes them (level 2) — each level strictly shrinks
            the updatable set."""
            nonlocal best_active, stagnant, demotions
            if n_active < best_active:
                best_active = n_active
                stagnant = 0
                return
            if n_outer2 < budget2 // 2:
                return
            stagnant += 1
            if stagnant >= 3:
                keep = 0xDB if demotions < 1 else 0x00
                demote((min(sweep_blocks, n_tiles),), (tile, tile),
                       (d_codes, d_list_a, d_count_a,
                        np.int32(rows), np.int32(cols),
                        np.int32(tiles_c), np.int32(keep)))
                demotions += 1
                stagnant = 0

        graph2 = None
        if max_outer_iterations >= sync_every:
            try:
                capture_stream = cp.cuda.Stream(non_blocking=True)
                with capture_stream:
                    capture_stream.begin_capture()
                    ta, tb = d_t, d_t2
                    la, ca = d_list_a, d_count_a
                    lb, cb = d_list_b, d_count_b
                    for _ in range(sync_every):
                        issue_pass2(ta, tb, la, ca, lb, cb)
                        ta, tb = tb, ta
                        la, lb = lb, la
                        ca, cb = cb, ca
                    graph2 = capture_stream.end_capture()
            except Exception:
                graph2 = None

        n_outer2 = 0
        if graph2 is not None:
            while True:
                graph2.launch()        # even window: d_t holds latest
                n_outer2 += sync_every
                n_active = int(d_count_a[0])
                if n_active == 0:
                    break
                maybe_demote(n_active, n_outer2)
                if n_outer2 >= budget2:
                    budget_exhausted(n_outer2, n_active)
                    d_t = d_t_o1       # first-order fallback
                    break
        else:
            cur_in, cur_out = d_t, d_t2
            while True:
                issue_pass2(cur_in, cur_out, d_list_a, d_count_a,
                            d_list_b, d_count_b)
                cur_in, cur_out = cur_out, cur_in
                d_list_a, d_list_b = d_list_b, d_list_a
                d_count_a, d_count_b = d_count_b, d_count_a
                n_outer2 += 1
                if (n_outer2 % sync_every == 0
                        or n_outer2 >= budget2):
                    n_active = int(d_count_a[0])
                    if n_active == 0:
                        break
                    maybe_demote(n_active, n_outer2)
                    if n_outer2 >= budget2:
                        budget_exhausted(n_outer2, n_active)
                        cur_in = d_t_o1
                        break
            d_t = cur_in               # latest refined field
        n_outer += n_outer2

    t_field = d_t.get().reshape(rows, cols) if download else None
    if return_stats:
        tile_sweeps = int(d_stats[0])
        # stats[1] counts inner iterations actually run (settle checks
        # end sweeps early), so this is the honest update count.
        cell_updates = int(d_stats[1]) * tile * tile
        result = (t_field, dict(
            outer_passes=n_outer,
            tile_sweeps=tile_sweeps,
            cell_updates=cell_updates,
            updates_per_cell=cell_updates / float(rows * cols),
        ))
    elif return_iterations:
        result = (t_field, n_outer)
    else:
        result = (t_field,)
    if return_device:
        result = (*result, d_t)
    if return_trace_field:
        if d_t_o1 is not None:
            t_o1 = d_t_o1.get().reshape(rows, cols) if download else None
            result = (*result, (t_o1, d_t_o1))
        else:
            result = (*result, (t_field, d_t))
    return result if len(result) > 1 else result[0]


def _unreachable_result(rows, cols, return_iterations):
    """All-unreachable field (every source invalid or impassable)."""
    t_field = np.full((rows, cols), UNREACHED, dtype=np.float32)
    if return_iterations:
        return t_field, 0
    return t_field


# ============================================================================
# Path extraction (pure numpy — testable without a GPU)
# ============================================================================

def _masked_gradient(t_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centered gradient with non-finite neighbors masked.

    Central differences where both neighbors are finite, one-sided where
    only one is, 0 where neither is (that axis contributes no descent
    information). Cells that are themselves unreachable get NaN.
    """
    t = np.where(t_field < FINITE_LIMIT, t_field, np.nan).astype(np.float64)
    p = np.pad(t, 1, constant_values=np.nan)
    ctr = p[1:-1, 1:-1]
    nn = p[:-2, 1:-1]
    ss = p[2:, 1:-1]
    ww = p[1:-1, :-2]
    ee = p[1:-1, 2:]

    def axis_grad(lo, hi):
        both = np.isfinite(lo) & np.isfinite(hi)
        g = np.where(both, (hi - lo) * 0.5, 0.0)
        only_hi = np.isfinite(hi) & ~np.isfinite(lo)
        g = np.where(only_hi, hi - ctr, g)
        only_lo = np.isfinite(lo) & ~np.isfinite(hi)
        g = np.where(only_lo, ctr - lo, g)
        return g

    gr = axis_grad(nn, ss)
    gc = axis_grad(ww, ee)
    invalid = ~np.isfinite(ctr)
    gr[invalid] = np.nan
    gc[invalid] = np.nan
    return gr, gc


def _sample_bilinear(field: np.ndarray, r: float, c: float) -> float:
    """NaN-aware bilinear sample; NaN when no finite corner exists.

    Scalar math on purpose — this runs thousands of times per traced
    path, and numpy temporaries here dominated the tracer's runtime
    (measured ~10x slower at 3000^2).
    """
    rows, cols = field.shape
    if r < 0.0:
        r = 0.0
    elif r > rows - 1.0:
        r = rows - 1.0
    if c < 0.0:
        c = 0.0
    elif c > cols - 1.0:
        c = cols - 1.0
    r0 = int(r)
    if r0 > rows - 2:
        r0 = max(rows - 2, 0)
    c0 = int(c)
    if c0 > cols - 2:
        c0 = max(cols - 2, 0)
    r1 = min(r0 + 1, rows - 1)
    c1 = min(c0 + 1, cols - 1)
    fr = r - r0
    fc = c - c0
    v00 = field[r0, c0]
    v01 = field[r0, c1]
    v10 = field[r1, c0]
    v11 = field[r1, c1]
    w00 = (1.0 - fr) * (1.0 - fc)
    w01 = (1.0 - fr) * fc
    w10 = fr * (1.0 - fc)
    w11 = fr * fc
    total = 0.0
    wsum = 0.0
    fin_sum = 0.0
    n_fin = 0
    for v, w in ((v00, w00), (v01, w01), (v10, w10), (v11, w11)):
        if v == v:                     # not NaN
            fin_sum += v
            n_fin += 1
            if w > 0.0:
                total += v * w
                wsum += w
    if wsum > 0.0:
        return total / wsum
    if n_fin:
        # Off-lattice point whose weighted corners are all invalid; fall
        # back to the finite corner values with uniform weight.
        return fin_sum / n_fin
    return float("nan")


def _descent_direction(gr, gc, r, c) -> Optional[Tuple[float, float]]:
    """Unit descent direction -grad T / |grad T| at (r, c), or None."""
    g_r = _sample_bilinear(gr, r, c)
    g_c = _sample_bilinear(gc, r, c)
    if g_r != g_r or g_c != g_c:       # NaN
        return None
    norm = (g_r * g_r + g_c * g_c) ** 0.5
    if norm < 1e-12:
        return None
    return -g_r / norm, -g_c / norm


_PLATEAU_BFS_CAP = 100_000


def _discrete_descent_step(t_field, r, c) -> List[Tuple[float, float]]:
    """Discrete descent when the gradient is degenerate (plan section 4).

    First choice: the strictly lower 8-neighbor with minimal T. On a
    genuine plateau (equal-T region, e.g. from zero-cost cells) a greedy
    step is directionless, so instead BFS across the connected equal-T
    region to the nearest cell with a strictly lower neighbor and return
    the whole crossing as polyline points. No admissible descent anywhere
    raises — a broken field must be loud.
    """
    rows, cols = t_field.shape
    ri = min(max(int(round(r)), 0), rows - 1)
    ci = min(max(int(round(c)), 0), cols - 1)
    t_here = float(t_field[ri, ci])
    tol = 1e-6 * abs(t_here) + 1e-9

    def lower_neighbor(cr, cc):
        best, best_t = None, t_field[cr, cc] - tol
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                t_n = t_field[nr, nc]
                if t_n < FINITE_LIMIT and t_n < best_t:
                    best, best_t = (nr, nc), t_n
        return best

    step = lower_neighbor(ri, ci)
    if step is not None:
        return [(float(step[0]), float(step[1]))]

    # Plateau: BFS over the equal-T region to the nearest exit. The
    # membership band is a few eps wide: the chaotic solver leaves
    # nondeterministic sub-eps slack, so near-shock cells on large-T
    # fields can form pockets a little deeper than one tol (observed on
    # the real-world planning raster at T ~ 1e6: a 1-cell pocket whose
    # neighbors all sat just above +tol). Anything within the band is
    # within solver tolerance of equal — crossing it is legitimate.
    from collections import deque
    band = 4.0 * tol
    seen = {(ri, ci)}
    parent = {}
    queue = deque([(ri, ci)])
    while queue and len(seen) <= _PLATEAU_BFS_CAP:
        cur = queue.popleft()
        exit_cell = lower_neighbor(*cur)
        if exit_cell is not None:
            chain = [exit_cell, cur]
            while cur in parent:
                cur = parent[cur]
                chain.append(cur)
            chain.reverse()          # (ri, ci) ... plateau ... exit
            return [(float(a), float(b)) for a, b in chain[1:]]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = cur[0] + dr, cur[1] + dc
                if (nr, nc) in seen:
                    continue
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if abs(float(t_field[nr, nc]) - t_here) <= band:
                    seen.add((nr, nc))
                    parent[(nr, nc)] = cur
                    queue.append((nr, nc))

    raise RuntimeError(
        f"path tracer stuck at cell ({ri}, {ci}): no admissible descent "
        f"from the plateau at T = {t_here:.6g} (searched {len(seen)} "
        f"cells). The T field looks unconverged or disconnected — re-run "
        f"the solve or use a discrete backend.")


def _prep_trace_fields(t_field: np.ndarray):
    """Masked gradient pair + NaN-masked T for bilinear sampling."""
    gr, gc = _masked_gradient(t_field)
    tm = np.where(t_field < FINITE_LIMIT, t_field,
                  np.nan).astype(np.float64)
    return gr, gc, tm


def trace_path(
        t_field: np.ndarray,
        target_idx: int,
        source_indices,
        step_size: float = 0.5,
        max_steps: Optional[int] = None,
        _fields=None,
) -> Optional[np.ndarray]:
    """Steepest-descent path from ``target_idx`` down to the nearest source.

    Heun (RK2) integration of ``dx/dt = -grad T/|grad T|`` with step
    ``step_size`` cells; gradient by masked central differences +
    bilinear interpolation. The interpolated T value is required to be
    non-increasing along the trace — a gradient step that would raise T
    (cell-scale noise can create spurious interpolation attractors) is
    replaced by a discrete descent step, which structurally prevents
    cycles. Further fallbacks (each tested): degenerate gradient and
    shock-line oscillation take a discrete descent step; a hard cap
    ``max_steps = 20 * (rows + cols)`` raises.

    Returns the polyline as (K, 2) float64 array of (row, col) points
    running target -> source, or None when the target is unreachable.
    """
    rows, cols = t_field.shape
    if max_steps is None:
        max_steps = 20 * (rows + cols)

    src = np.atleast_1d(np.asarray(source_indices)).astype(np.int64)
    src_pos = [(float(s // cols), float(s % cols)) for s in src]

    tr, tc = divmod(int(target_idx), cols)
    if t_field[tr, tc] >= FINITE_LIMIT:
        return None

    gr, gc, tm = (_fields if _fields is not None
                  else _prep_trace_fields(t_field))

    pr, pc = float(tr), float(tc)
    points = [(pr, pc)]
    t_cur = _sample_bilinear(tm, pr, pc)

    def nearest_src(qr, qc):
        best_d2, best = float("inf"), src_pos[0]
        for sr_, sc_ in src_pos:
            d2 = (sr_ - qr) ** 2 + (sc_ - qc) ** 2
            if d2 < best_d2:
                best_d2, best = d2, (sr_, sc_)
        return best_d2 ** 0.5, best

    best_dist, _ = nearest_src(pr, pc)
    stall = 0

    def discrete_hop(qr, qc):
        hops = _discrete_descent_step(t_field, qr, qc)
        points.extend(hops)
        hr, hc = hops[-1]
        return hr, hc, float(t_field[int(hr), int(hc)])

    for _ in range(max_steps):
        d, nearest = nearest_src(pr, pc)
        if d <= 1.0:
            points.append(nearest)
            return np.array(points)

        if stall >= 20:
            # Oscillation across a shock line: no progress in 20 steps
            pr, pc, t_cur = discrete_hop(pr, pc)
            stall = 0
            best_dist = min(best_dist, nearest_src(pr, pc)[0])
            continue

        d1 = _descent_direction(gr, gc, pr, pc)
        if d1 is None:
            pr, pc, t_cur = discrete_hop(pr, pc)
            best_dist = min(best_dist, nearest_src(pr, pc)[0])
            continue

        # Heun / RK2
        mr = min(max(pr + step_size * d1[0], 0.0), rows - 1.0)
        mc = min(max(pc + step_size * d1[1], 0.0), cols - 1.0)
        d2 = _descent_direction(gr, gc, mr, mc)
        if d2 is None:
            sr_, sc_ = d1
        else:
            sr_ = d1[0] + d2[0]
            sc_ = d1[1] + d2[1]
            norm = (sr_ * sr_ + sc_ * sc_) ** 0.5
            if norm < 1e-6:
                sr_, sc_ = d1             # opposing directions: shock
            else:
                sr_ /= norm
                sc_ /= norm
        cr = min(max(pr + step_size * sr_, 0.0), rows - 1.0)
        cc = min(max(pc + step_size * sc_, 0.0), cols - 1.0)
        t_cand = _sample_bilinear(tm, cr, cc)
        if (t_field[int(cr + 0.5), int(cc + 0.5)] >= FINITE_LIMIT
                or t_cand != t_cand or t_cand > t_cur):
            # Impassable destination (corner grazing near an exclusion)
            # or uphill-in-T move: discrete descent instead — keeps the
            # trace monotone in T and out of walls.
            pr, pc, t_cur = discrete_hop(pr, pc)
        else:
            pr, pc, t_cur = cr, cc, t_cand
            points.append((pr, pc))

        d, _ = nearest_src(pr, pc)
        if d < best_dist - 0.25 * step_size:
            best_dist = d
            stall = 0
        else:
            stall += 1

    raise RuntimeError(
        f"path tracer exceeded max_steps = {max_steps} without reaching "
        f"a source — the T field looks broken (unconverged or "
        f"inconsistent). Re-run the solve or use a discrete backend.")


def trace_paths(
        t_field: np.ndarray,
        source_indices,
        target_indices,
        step_size: float = 0.5,
        max_steps: Optional[int] = None,
) -> List[Optional[np.ndarray]]:
    """Trace one descent polyline per target (None where unreachable).

    Gradient and sampling fields are computed once, shared across
    targets.
    """
    fields = _prep_trace_fields(t_field)
    targets = np.atleast_1d(np.asarray(target_indices)).astype(np.int64)
    return [trace_path(t_field, int(t), source_indices,
                       step_size=step_size, max_steps=max_steps,
                       _fields=fields)
            for t in targets]


#: Polyline-buffer budget for one device-tracer batch (bytes).
_TRACE_BATCH_BYTES = 256 << 20


def trace_paths_gpu(
        t_field: Optional[np.ndarray],
        source_indices,
        target_indices,
        step_size: float = 0.5,
        max_steps: Optional[int] = None,
        t_device=None,
        shape: Optional[Tuple[int, int]] = None,
) -> List[Optional[np.ndarray]]:
    """Device-side tracer — drop-in for :func:`trace_paths`.

    One CUDA thread per target runs the host-tracer semantics in
    float32 (Heun descent, monotone-interpolated-T rule, discrete
    descent hops, shock-line stall handling; float32 because the
    serial FP64 chain was the measured bottleneck and T is float32
    anyway). Targets that hit a state the kernel does not implement
    (plateau BFS, step cap) are re-traced transparently by the host
    tracer, so results — including the loud RuntimeError on genuinely
    broken fields — match :func:`trace_paths`. Individual traced
    polylines may differ from the float64 host tracer in
    float-marginal direction decisions; both satisfy the same
    invariants (monotone T, passable cells, source reached).

    Parameters:
        t_field: Solved T field (host array), or None in device-only
            mode (then ``t_device`` and ``shape`` are required and the
            host field is downloaded lazily only if a fallback trace
            needs it — phase-1 lazy-D2H flow).
        t_device: Optional flat float32 CuPy array of the same field
            (e.g. from ``eikonal_raster_gpu(..., return_device=True)``);
            saves the upload.
        shape: (rows, cols) — required when ``t_field`` is None.
    """
    if t_field is None and (t_device is None or shape is None):
        raise ValueError(
            "device-only tracing needs t_device and shape")
    if not GPU_AVAILABLE:
        return trace_paths(t_field, source_indices, target_indices,
                           step_size=step_size, max_steps=max_steps)
    rows, cols = t_field.shape if t_field is not None else shape
    if max_steps is None:
        max_steps = 20 * (rows + cols)
    max_pts = max_steps + 2

    targets = np.atleast_1d(np.asarray(target_indices)).astype(np.int64)
    src = np.atleast_1d(np.asarray(source_indices)).astype(np.int64)
    if targets.size == 0:
        return []
    src_pos = np.empty(src.size * 2, dtype=np.float64)
    src_pos[0::2] = (src // cols).astype(np.float64)
    src_pos[1::2] = (src % cols).astype(np.float64)

    d_t = (t_device if t_device is not None
           else cp.asarray(np.ascontiguousarray(t_field.ravel())))
    d_src = cp.asarray(src_pos)
    kernel = _get_eik_kernel("eikonal_trace", _TRACE_KERNEL)

    per_target = max_pts * 2 * 8
    batch = max(1, min(targets.size, _TRACE_BATCH_BYTES // per_target))

    results: List[Optional[np.ndarray]] = [None] * targets.size
    fallback: List[int] = []
    tpb = 64
    for lo in range(0, targets.size, batch):
        chunk = targets[lo:lo + batch]
        n = chunk.size
        d_targets = cp.asarray(chunk)
        d_out = cp.empty(n * max_pts * 2, dtype=cp.float64)
        d_len = cp.zeros(n, dtype=cp.int32)
        d_status = cp.zeros(n, dtype=cp.int32)
        kernel(((n + tpb - 1) // tpb,), (tpb,),
               (d_t, d_src, np.int32(src.size),
                d_targets, np.int32(n),
                np.int32(rows), np.int32(cols),
                np.float64(step_size), np.int32(max_steps),
                np.int32(max_pts), d_out, d_len, d_status))
        lens = d_len.get()
        stats = d_status.get()
        for i in range(n):
            if stats[i] == 0:
                k = int(lens[i])
                start = i * max_pts * 2
                poly = d_out[start:start + k * 2].get()
                results[lo + i] = poly.reshape(k, 2)
            elif stats[i] == 1:
                results[lo + i] = None
            else:
                fallback.append(lo + i)

    if fallback:
        if t_field is None:      # lazy: only fallbacks pay the D2H
            t_field = d_t.get().reshape(rows, cols)
        fields = _prep_trace_fields(t_field)
        for j in fallback:
            results[j] = trace_path(
                t_field, int(targets[j]), source_indices,
                step_size=step_size, max_steps=max_steps, _fields=fields)
    return results


def polyline_to_cells(polyline: np.ndarray, rows: int, cols: int,
                      forbidden_mask: Optional[np.ndarray] = None
                      ) -> List[int]:
    """Rasterize a (row, col) polyline into a deduplicated cell path.

    Dense sampling (4 samples per cell of travel) + rounding + consecutive
    dedup + immediate-backtrack removal. Consecutive cells are normally
    8-adjacent — the same contract the discrete backends' knight-move
    paths already relax for downstream consumers.

    ``forbidden_mask`` (bool, raster shape) drops sampled cells that are
    impassable: a continuous path may legally graze the corner region of
    an excluded cell (the polyline stays in passable territory but a
    sample rounds into the wall). Dropping such grazing samples can leave
    a diagonal jump around the corner.

    Returns flat cell indices in polyline order.
    """
    if len(polyline) == 0:
        return []
    pts = np.asarray(polyline, dtype=np.float64)

    # Dense sampling, vectorized (the per-sample Python loop was ~half
    # of the remaining trace cost at 3000^2 after the device tracer).
    if len(pts) > 1:
        seg = pts[1:] - pts[:-1]
        span = np.maximum(np.abs(seg[:, 0]), np.abs(seg[:, 1]))
        n_samp = np.maximum(1, np.ceil(span / 0.25).astype(np.int64))
        total = int(n_samp.sum())
        seg_idx = np.repeat(np.arange(len(seg)), n_samp)
        cum = np.concatenate([[0], np.cumsum(n_samp)])
        within = np.arange(total) - cum[seg_idx]
        ts = (within + 1.0) / n_samp[seg_idx]
        samples = np.vstack([pts[:1],
                             pts[seg_idx] + ts[:, None] * seg[seg_idx]])
    else:
        samples = pts

    r = np.clip(np.round(samples[:, 0]).astype(np.int64), 0, rows - 1)
    c = np.clip(np.round(samples[:, 1]).astype(np.int64), 0, cols - 1)
    idxs = r * cols + c
    if forbidden_mask is not None:
        idxs = idxs[~forbidden_mask.ravel()[idxs]]
    if idxs.size == 0:
        return []
    # Consecutive dedup (vectorized; identical to the skip rule — the
    # backtrack loop below sees the same decisions on the reduced list).
    keep = np.empty(idxs.size, dtype=bool)
    keep[0] = True
    np.not_equal(idxs[1:], idxs[:-1], out=keep[1:])
    idxs = idxs[keep]

    cells: List[int] = []
    for idx in idxs.tolist():
        if cells and idx == cells[-1]:
            continue
        if len(cells) >= 2 and idx == cells[-2]:
            cells.pop()          # immediate backtrack A,B,A -> A
            continue
        cells.append(idx)
    return cells

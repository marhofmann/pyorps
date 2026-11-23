// V5 asynchronous bucket-queue SSSP on a raster (one cooperative launch).
//
// Loaded by pyorps/utils/sssp_gpu.py as the _ASYNC_SSSP_KERNEL source and
// compiled with NVRTC (--std=c++17 -Xptxas -dlcm=cg, cooperative groups).
// The float32-raster variant is derived textually (_float_raster_source):
// keep the `unsigned short sv/dv/iv` declarations and `== max_cost` tests.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Control words. 0..7 are single words; 8.. are parity-indexed PAIRS
// (index + par, par = rendezvous count & 1). Every block flips par at
// every rendezvous, so a word of the *other* parity is never read between
// the two barriers in which gtid 0 rewrites it -- the barrier discipline
// ("no thread reads a word that gtid 0 rewrites between the same two
// barriers") holds without extra barriers.
#define C5_DONE 0
#define C5_WORK 1
#define C5_BARRIER_CNT 3
#define C5_BARRIER_SENSE 4
#define C5_TGT_MAX 7
#define C5_BASE 8     // span base bucket of this parity's span
#define C5_OVF  10    // a ring push dropped this span -> rewind from base
#define C5_FOVF 12    // far list overflowed (or forced) -> full dist[] scan
#define C5_FCNT 14    // far list length (reservations may exceed far_cap)
#define C5_FMIN 16    // min dist (int bits) of everything appended far
#define C5_SMIN 18    // full-scan min scratch (int bits)

#define NB 32
#define EMPTY_SLOT (-1)
// Ring entries: v >= 0 for an entry in the ring of its own bucket,
// -(v+2) <= -2 for an entry that spilled into a later ring (see
// v5_ring_push); EMPTY_SLOT = -1 stays distinct from both.
#define SPILL_TAG(v) (-((v) + 2))

// Sense-reversing rendezvous barrier (same protocol as V4: grid.sync()
// lacks usable memory ordering on Blackwell; explicit __threadfence()).
// Every thread of every block must call it -- block-divergent calls
// pair its __syncthreads with a different one and deadlock the grid.
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
// drop a vertex. A spilled entry is stored tagged (SPILL_TAG) so the
// consumer's stale-entry test (ring index vs the vertex's current
// bucket) is only applied to entries that sit in their own ring.
//
// That matters for more than memory. A refill that drops re-raises
// C5_OVF, and the resulting rewind rescans from the *same* span
// base and re-pushes the *same* oversubscribed bucket -- if a single
// delta-bucket holds more than the whole ring set can take (a 0-cost
// plateau, or any large region whose edges are tiny next to delta),
// the rendezvous makes no progress and the kernel spins forever. Every
// path that reserves a slot must go through here.
//
// Returns 1 when placed (the item is now claimable), 0 when every ring
// is full -- the caller then drops the push, raises C5_OVF and parks
// the vertex on the far list, which the span-boundary rewind repairs
// losslessly (the dropped vertex kept its improved dist[], and the
// rewind restarts from base*delta, at or below which no push can land).
__device__ inline int v5_ring_push(volatile int* wres, int* arena,
                                   const int cap, const int bidx,
                                   const int v) {
    for (int t = 0; t < NB; t++) {
        int bb = bidx + t;
        if (bb >= NB) bb -= NB;
        // Spill path only: a ring known full stays full until the next
        // rendezvous, so skip it with a plain load instead of a returned
        // atomic (the spill chain was ~25 % of stall samples at k=3 on
        // the baseline). The first ring is tried with the atomic directly
        // (one round trip, not two, in the common no-spill case).
        if (t > 0 && wres[bb] >= cap) continue;
        // Reservations beyond cap are harmless: readers clamp wres to
        // cap, so nobody ever waits on a slot we did not write.
        int q = atomicAdd((int*)&wres[bb], 1);
        if (q < cap) {
            __threadfence();
            arena[bb * cap + q] = (t > 0) ? SPILL_TAG(v) : v;
            return 1;
        }
    }
    return 0;
}

// Append `v` to the far list of parity `par`: one tail atomic per
// coalesced group of callers (warp-aggregated). Past far_cap the entry
// is dropped and C5_FOVF raised: the next rendezvous then rebuilds the
// frontier from a full dist[] scan instead -- lossless, just the old
// O(n_pixels) cost for that one span. The list is only read between
// rendezvous barriers, so no publish fence is needed here. Returns 0 when
// the entry was dropped (list full), so a caller looping over many far
// pixels can stop appending to a list that is already void.
__device__ __forceinline__ int v5_far_append(volatile int* control,
                                             const int par, int* flist,
                                             const int far_cap, const int v) {
    cg::coalesced_group g = cg::coalesced_threads();
    int pos = 0;
    if (g.thread_rank() == 0)
        pos = atomicAdd((int*)&control[C5_FCNT + par], (int)g.size());
    pos = g.shfl(pos, 0) + (int)g.thread_rank();
    if (pos < far_cap) { flist[pos] = v; return 1; }
    control[C5_FOVF + par] = 1;
    return 0;
}

// Count `g.size()` pushes as in-flight work with one atomic per coalesced
// group, issued before the group's ring reservations. Relaxed atomics on
// different words are not ordered, so a consumer may claim, process and
// decrement an item before its producer's increment lands: C5_WORK can
// briefly read 0 while work is still in flight. That is safe because the
// rendezvous starts only when EVERY block has arrived at B1: a block that
// read 0 early just waits there, and the producer block arrives only after
// its end-of-chunk fence (which orders its increments before its
// atomicSub) and its own quiescence read, which then sees the true count.
__device__ __forceinline__ void v5_count_work(volatile int* control) {
    cg::coalesced_group g = cg::coalesced_threads();
    if (g.thread_rank() == 0)
        atomicAdd((int*)&control[C5_WORK], (int)g.size());
}

// Handler for a strict improvement of dist[v] to nd (int bits ndi) by a
// relaxation inside the hot path. ONE copy, shared with the generated
// relax bodies: in-span -> ring entry; otherwise far list. Ring drop ->
// far list + rewind flag. far_ok (read once per chunk) is 0 once this
// span's far list has overflowed: the next rendezvous then rebuilds the
// frontier from a full dist[] scan anyway, so further appends are wasted
// atomics (at k=3 the far set is ~40 % of the raster).
__device__ __forceinline__ void v5_on_improve(
    volatile int* control, volatile int* wres, int* arena, const int cap,
    int* flist, const int far_cap, const int far_ok,
    const int par, const int base, const float delta, const float span_hi,
    const int v, const float nd, const int ndi)
{
    if (nd < span_hi) {
        int bidx = (int)(nd / delta) - base;
        if (bidx < 0) bidx = 0; if (bidx >= NB) bidx = NB - 1;
        v5_count_work(control);
        if (!v5_ring_push(wres, arena, cap, bidx, v)) {
            atomicSub((int*)&control[C5_WORK], 1);
            control[C5_OVF + par] = 1;
            if (far_ok) {
                v5_far_append(control, par, flist, far_cap, v);
                atomicMin((int*)&control[C5_FMIN + par], ndi);
            }
        }
    } else if (far_ok) {
        // out-of-span: dist[] is updated, the vertex waits on the far
        // list for the span-boundary refill (no ring entry).
        v5_far_append(control, par, flist, far_cap, v);
        atomicMin((int*)&control[C5_FMIN + par], ndi);
    }
}

// Refill push (rendezvous): a drop parks the vertex on the next far list.
__device__ __forceinline__ void v5_refill_push(
    volatile int* control, volatile int* wres, int* arena, const int cap,
    int* nxt, const int far_cap,
    const int npar, const int new_base, const float delta,
    const int v, const float d, float& fmin)
{
    int bidx = (int)(d / delta) - new_base;
    if (bidx < 0) bidx = 0; if (bidx >= NB) bidx = NB - 1;
    v5_count_work(control);
    if (!v5_ring_push(wres, arena, cap, bidx, v)) {
        atomicSub((int*)&control[C5_WORK], 1);
        control[C5_OVF + npar] = 1;
        v5_far_append(control, npar, nxt, far_cap, v);
        if (d < fmin) fmin = d;
    }
}

// Asynchronous bucket-queue SSSP (ADDS-style, simplified for rasters).
//
// Hot path has NO grid-wide barriers: blocks claim chunks from a shared
// ring of NB delta-granular buckets covering the rolling span
// [base, base+NB)*delta, relax them, and push improvements with atomics
// only. Improvements landing outside the span only update dist[] and
// park the vertex on the span's FAR LIST (no ring entry). Quiescence
// (work == 0 => nothing queued AND nobody mid-chunk) funnels every block
// into a rendezvous that advances the span, refills the ring from the
// far list (or, after a far-list overflow and on the first epoch, from
// a full dist[] scan), checks the target bound, or terminates. The far
// list is exact: a vertex whose dist lies at/after the span end was
// written far (or dropped by a full ring, or carried over from the last
// refill) and is on the list, so the refill is lossless; the running
// far minimum gives the next span base without any scan. Bucket
// granularity stays FINE (delta, not a super-bucket): the 2026-08-06
// sweep showed coarse priority windows amplify work on heavy-tail
// rasters; the async design removes barrier cost without giving up
// ordering.
//
// Stale entries: every improvement of a vertex pushes one ring entry, so
// a vertex improved k times in a span is queued k times. An entry whose
// vertex has since moved to a LOWER bucket is stale: the improvement
// that moved it pushed another entry there, and the entry pushed by the
// last improvement (which is in the bucket of the final value) is never
// stale, so skipping stale entries at claim time keeps "every vertex is
// expanded from its final value" and drops 11-17 % of the atomicMin
// work at k=1. The test compares the ring index with the bucket of
// dist[u] computed by the same expression the push used, so it is
// exact; spilled entries are tagged and exempt. The argument needs
// every in-span improvement to have produced an entry: a push DROPPED
// by a full ring set has none, so the skip is disabled for the rest of
// a span in which a drop occurred (C5_OVF read once per chunk; the
// rewind recovers everything either way -- a stale read only delays a
// vertex to that rewind). Without that guard a k=3 run spent 90 rewinds
// instead of 22.
//
// Block-uniform control flow: every decision that leads to a
// __syncthreads (claim result, quiescence) is taken by warp 0, published
// through shared memory and read by all warps after one barrier; a
// second barrier protects the shared words before warp 0 may rewrite
// them. Per-thread reads of control[] for such decisions deadlocked the
// original kernel (warps pairing different __syncthreads instances).
extern "C" __global__
void sssp_async_v5(
    const unsigned short* __restrict__ raster,
    const signed char*    __restrict__ steps,
    const float*          __restrict__ cost_factors,
    const signed char*    __restrict__ inter_lut,
    const int*            __restrict__ n_inter,
    const int n_steps, const int max_inter_cols,
    const int rows, const int cols, const int max_cost,
    float* dist,
    // Pred solves only (NULL otherwise): packed (dist bits << 32 | pred),
    // all ones before the solve; the session keeps the low words.
    unsigned long long* pred,
    const float delta, const int n_pixels, const int chunk,
    const int* targets, const int n_targets, const float margin,
    volatile int* control,
    volatile int* wres,      // per-bucket write reservations [NB]
    volatile int* rres,      // per-bucket read reservations  [NB]
    int* arena,              // NB * cap item slots, EMPTY_SLOT-initialized
    const int cap,
    int* far_a, int* far_b,  // far lists, one per parity [far_cap each]
    const int far_cap,
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

    __shared__ int sh_bucket, sh_start, sh_take, sh_work;
    // Parity of the current span's control words; every block executes
    // every rendezvous, so all blocks flip it together.
    int par = 0;

    while (true) {
        // ---- claim a chunk from the lowest claimable bucket ----
        // Warp 0 probes all NB buckets in ONE round trip (lane == bucket)
        // instead of walking them serially, and the lane of the lowest
        // non-empty bucket claims min(avail, chunk) with one CAS (a retry
        // reuses the value the failed CAS returned). The other warps wait
        // at the __syncthreads below for two round trips, not for up to
        // 2*NB serial ones. Lane 0 also publishes the quiescence snapshot
        // so the whole block branches the same way (block-uniform
        // __syncthreads).
        //
        // Claims are instant and single-bucket on purpose: taking from
        // several buckets at once, or waiting for a fuller chunk, widens
        // the effective priority window and was measured to double the
        // expansions per cell (random 1000^2: 2.47 vs 1.12 items/cell).
        if (threadIdx.x < 32) {
            int lane = threadIdx.x;
            int rr = rres[lane];
            int wr = wres[lane]; if (wr > cap) wr = cap;
            int work_now = (lane == 0) ? control[C5_WORK] : 0;
            work_now = __shfl_sync(0xffffffffu, work_now, 0);
            int avail = wr - rr; if (avail < 0) avail = 0;
            unsigned m = __ballot_sync(0xffffffffu, avail > 0);
            int b0 = m ? (__ffs(m) - 1) : 0;
            int got = 0, start = 0;
            if (m && lane == b0) {
                int want = avail < chunk ? avail : chunk;
                int old = atomicCAS((int*)&rres[lane], rr, rr + want);
                if (old == rr) { got = want; start = rr; }
                else {
                    // Another block took some: retry once with the returned
                    // value; wr is a stale lower bound, so never over-claim.
                    rr = old; int av2 = wr - rr; if (av2 > want) av2 = want;
                    if (av2 > 0 && atomicCAS((int*)&rres[lane], rr, rr + av2) == rr) { got = av2; start = rr; }
                }
            }
            int take_all = __shfl_sync(0xffffffffu, got, b0);
            int start_all = __shfl_sync(0xffffffffu, start, b0);
            if (lane == 0) {
                sh_bucket = b0; sh_start = start_all; sh_take = take_all;
                sh_work = (take_all > 0) ? 1 : work_now;
            }
        }
        __syncthreads();
        int take = sh_take, work = sh_work;

        if (take == 0) {
            if (work != 0) {
                // ---- no claimable work yet: poll ----
            #if __CUDA_ARCH__ >= 700
                __nanosleep(200);
            #endif
                __syncthreads();   // all warps have read sh_* before warp 0 rewrites them
                continue;
            }
            // work == 0: nothing queued, nobody mid-chunk (a processing
            // block keeps its chunk counted until fully done) -- every
            // block observes 0 and funnels into the rendezvous.
            v5_barrier(control, n_blocks);                                  // B1

            // -- span advance / termination (all blocks, barriered) --
            int base = control[C5_BASE + par];
            int ring_ovf = control[C5_OVF + par];
            int fullscan = (control[C5_FOVF + par] != 0) || (base < 0);
            int cur_cnt = control[C5_FCNT + par]; if (cur_cnt > far_cap) cur_cnt = far_cap;
            int* cur = par ? far_b : far_a;
            int* nxt = par ? far_a : far_b;
            int npar = par ^ 1;
            // Overflow rewind: dropped pushes kept their dist inside the
            // current span -- restart from the span base, not its end.
            float from_val = ring_ovf ? base * delta : (base + NB) * delta;
            if (base < 0) from_val = 0.0f;   // initial epoch
            float gm;
            if (fullscan) {
                float lm = 1e30f;
                for (int i = gtid; i < n_pixels; i += stride) {
                    float d = dist[i];
                    if (d >= from_val && d < lm) lm = d;
                }
                if (lm < 1e30f)
                    atomicMin((int*)&control[C5_SMIN + par], __float_as_int(lm));
                v5_barrier(control, n_blocks);                              // B1b (full-scan spans only)
                gm = __int_as_float(control[C5_SMIN + par]);
            } else {
                // Running minimum of everything appended far this span
                // (far writes, ring drops, last refill's carry-overs). It
                // is a lower bound of the true min at/after from_val: a
                // far-written vertex may since have been improved into the
                // span. Then the new span starts early (part of it empty);
                // an entirely empty span costs one extra rendezvous whose
                // carry-over minimum is exact. Never below from_val.
                gm = __int_as_float(control[C5_FMIN + par]);
            }
            if (gm >= 1e29f) {
                if (gtid == 0) control[C5_DONE] = 1;
                v5_barrier(control, n_blocks);
                break;
            }
            // TODO(review finding 4, pre-existing): (int)(gm / delta) overflows once gm / delta >= 2^31.
            int new_base = (int)(gm / delta);
            // target early exit: everything unprocessed is >= gm.
            // Scratch word MUST differ from the word gm was published
            // through: slower blocks may not have read it yet -- zeroing
            // it here would hand them gm == 0 and diverge the blocks'
            // barrier sequences (the 2026-08-06 nondeterministic
            // targeted-mode corruption).
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
            // -- reset ring and the next parity's words (last read before
            //    B1 of the previous rendezvous), refill from the far list --
            for (int b = gtid; b < NB; b += stride) { wres[b] = 0; rres[b] = 0; }
            if (gtid == 0) {
                control[C5_BASE + npar] = new_base;
                control[C5_OVF + npar] = 0;
                control[C5_FOVF + npar] = 0;
                control[C5_FCNT + npar] = 0;
                control[C5_FMIN + npar] = __float_as_int(1e30f);
                control[C5_SMIN + npar] = __float_as_int(1e30f);
                control[C5_WORK] = 0;
            }
            v5_barrier(control, n_blocks);                                  // B2
            float nbv = new_base * delta, nev = (new_base + NB) * delta;
            float fmin = 1e30f;
            // Set once this thread saw the next far list overflow: the next
            // rendezvous then full-scans and ignores that list, so further
            // appends would only cost atomics (fields are unchanged).
            int far_full = 0;
            if (fullscan) {
                for (int i = gtid; i < n_pixels; i += stride) {
                    float d = dist[i];
                    if (d >= nbv && d < nev) {
                        v5_refill_push(control, wres, arena, cap, nxt, far_cap,
                                       npar, new_base, delta, i, d, fmin);
                    } else if (d >= nev && d < 1e30f) {
                        if (!far_full) far_full = !v5_far_append(control, npar, nxt, far_cap, i);
                        if (d < fmin) fmin = d;
                    }
                }
            } else {
                for (int k = gtid; k < cur_cnt; k += stride) {
                    int v = cur[k];
                    float d = dist[v];
                    if (d >= nbv && d < nev) {
                        v5_refill_push(control, wres, arena, cap, nxt, far_cap,
                                       npar, new_base, delta, v, d, fmin);
                    } else if (d >= nev && d < 1e30f) {
                        // still far: carry over (duplicates on the list are
                        // harmless, they only re-relax a settled vertex)
                        if (!far_full) far_full = !v5_far_append(control, npar, nxt, far_cap, v);
                        if (d < fmin) fmin = d;
                    }
                    // d < nbv: improved into an earlier span, already relaxed
                }
            }
            if (fmin < 1e30f)
                atomicMin((int*)&control[C5_FMIN + npar], __float_as_int(fmin));
            v5_barrier(control, n_blocks);                                  // B3
            par = npar;
            continue;
        }

        // ---- process the claimed chunk ----
        int b = sh_bucket, start = sh_start;
        int base = control[C5_BASE + par];
        float span_hi = (base + NB) * delta;
        int* flist = par ? far_b : far_a;
        const int skip_ok = (control[C5_OVF + par] == 0);    // no ring drop yet this span
        const int far_ok = (control[C5_FOVF + par] == 0);    // far list not overflowed
        for (int i = threadIdx.x; i < take; i += blockDim.x) {
            volatile int* slot = (volatile int*)&arena[b * cap + start + i];
            int u;
            while ((u = *slot) == EMPTY_SLOT) {}   // writer reserved: finite
            *slot = EMPTY_SLOT;
            int spilled = (u < EMPTY_SLOT);
            if (spilled) u = SPILL_TAG(u);          // -(u+2) is its own inverse
            // Prologue shared by every relax body: the vertex's current
            // distance, then the stale-entry test (see the kernel comment).
            // A generated body starts after `ud` and tests ud itself.
            float ud = dist[u];
            if (skip_ok && !spilled) {
                int bu = (int)(ud / delta) - base;
                if (bu < 0) bu = 0;
                if (bu < b) continue;   // stale: a lower-bucket entry expands u
            }
            // ---- generic relax body (a generated body may replace it) ----
            if (ud >= 1e30f) continue;
            int ur = u / cols, uc = u - ur * cols;
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
                // Predecessor of a pred solve: the winner of v's FINAL
                // improvement. A plain store raced (an older winner's late
                // store could overwrite a newer one; on a 0-cost plateau such
                // a stale sideways link still validates in v5_best_pred and
                // could close a cycle -> solve_paths returned None). The
                // packed 64-bit min orders by the distance first: at most one
                // thread wins each value, so the min is (final dist, final
                // winner) whatever the arrival order. That winner held its
                // own final value when it relaxed v, so pred chains run
                // backwards in time and are acyclic.
                int ndi = __float_as_int(nd), odi = atomicMin((int*)&dist[v], ndi);
                if (ndi < odi) {
                    if (pred != NULL)
                        atomicMin(&pred[v], ((unsigned long long)(unsigned)ndi << 32) | (unsigned)u);
                    v5_on_improve(control, wres, arena, cap, flist, far_cap, far_ok,
                                  par, base, delta, span_hi, v, nd, ndi);
                }
            }
            // ---- end of generic relax body ----
        }
        // chunk complete (including all its pushes): release the count
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicSub((int*)&control[C5_WORK], take);
    }
}

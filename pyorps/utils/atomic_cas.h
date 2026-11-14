/*
 * atomic_cas.h - Lock-free distance+predecessor updates for delta-stepping.
 *
 * Packs float32 dist (upper 32 bits) + uint32 pred (lower 32 bits) into a
 * single uint64_t.  IEEE 754 positive floats preserve integer ordering, so
 * packed_a < packed_b  iff  dist_a < dist_b  (for non-negative distances).
 *
 * Platform support:
 *   GCC/Clang: __atomic_compare_exchange_n  (C11 atomics)
 *   MSVC:      _InterlockedCompareExchange64
 *
 * Note: MISRA 11.3 pointer casts are necessary for cross-platform atomics
 * (MSVC intrinsics require volatile long* / volatile long long* types).
 */

#ifndef ATOMIC_CAS_H
#define ATOMIC_CAS_H

#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_InterlockedCompareExchange64)
#endif

/* Compile-time check: float and uint32_t must be the same size for
   bit-packing to work correctly. */
typedef char static_assert_float_u32_same_size
    [sizeof(float) == sizeof(uint32_t) ? 1 : -1];

/* ------------------------------------------------------------------ */
/*  Pack / Unpack helpers                                              */
/* ------------------------------------------------------------------ */

static inline uint64_t pack_dist_pred(float dist, uint32_t pred) {
    union { float f; uint32_t u; } cvt;
    cvt.f = dist;
    return ((uint64_t)cvt.u << 32) | (uint64_t)pred;
}

static inline float unpack_dist(uint64_t packed) {
    union { float f; uint32_t u; } cvt;
    cvt.u = (uint32_t)(packed >> 32);
    return cvt.f;
}

static inline uint32_t unpack_pred(uint64_t packed) {
    return (uint32_t)(packed & 0xFFFFFFFFULL);
}

/* ------------------------------------------------------------------ */
/*  Atomic load (aligned 64-bit reads are naturally atomic on x86-64,  */
/*  but we use compiler intrinsics for portability)                     */
/* ------------------------------------------------------------------ */

static inline uint64_t atomic_load_u64(volatile void* raw_addr) {
    volatile uint64_t* addr = (volatile uint64_t*)raw_addr;
#ifdef _MSC_VER
    /* MSVC: volatile read is fine on x86-64 for atomic load. */
    return *addr;
#else
    return __atomic_load_n(addr, __ATOMIC_SEQ_CST);
#endif
}

/* ------------------------------------------------------------------ */
/*  Relaxation acceptance rule                                         */
/*                                                                     */
/*  Accept a candidate label (new_dist, new_pred) over the stored one  */
/*  when it is strictly SHORTER, or when it ties the distance with a   */
/*  smaller predecessor id AND the edge it came over really added      */
/*  length (new_dist > src_dist, src_dist = the label of new_pred the  */
/*  candidate was computed from).                                      */
/*                                                                     */
/*  Comparing the packed words as a whole (the old rule) also accepted */
/*  ties over ZERO-weight edges: two neighbouring cost-0 cells at the  */
/*  same distance then became each other's predecessor, and every path */
/*  walk through them never ended.                                     */
/*                                                                     */
/*  Why the rule cannot close a cycle: every stored link x -> p obeys  */
/*  dist(x) >= dist(p) (dist(x) = fl(dist_read(p) + w) >= dist_read(p) */
/*  for w >= 0, and dist(p) only falls afterwards). If setting         */
/*  pred(v) = u closed a cycle, u's chain would already lead to v, so  */
/*  src_dist >= dist(u) >= old dist(v). A strict update needs          */
/*  new_dist < old dist(v) <= src_dist <= new_dist; a tie update needs */
/*  old dist(v) = new_dist > src_dist >= old dist(v). Both impossible. */
/*  The tie-break keeps the choice among tied candidates over positive */
/*  edges independent of the order in which threads relax them; over   */
/*  zero-weight edges the first tied label stays. Tie updates never    */
/*  change a distance, so the distances are those of the strict rule.  */
/* ------------------------------------------------------------------ */

static inline int relax_accepts(uint64_t new_packed, uint64_t old_packed,
                                float src_dist) {
    /* Non-negative IEEE 754 floats order like their bit patterns, so the
       high words compare the distances. */
    uint32_t new_hi = (uint32_t)(new_packed >> 32);
    uint32_t old_hi = (uint32_t)(old_packed >> 32);
    if (new_hi != old_hi) {
        return new_hi < old_hi;
    }
    return ((uint32_t)new_packed < (uint32_t)old_packed)
        && (unpack_dist(new_packed) > src_dist);
}

/* ------------------------------------------------------------------ */
/*  CAS-based distance+predecessor update                              */
/*                                                                     */
/*  Returns 1 if the update was applied, 0 otherwise.                  */
/*  No ABA risk: distances only decrease monotonically.                */
/*  src_dist: the label of new_pred that new_dist was computed from    */
/*  (see relax_accepts).                                               */
/* ------------------------------------------------------------------ */

static inline int atomic_try_update_dist_pred(
        volatile void* raw_dist_pred,
        uint64_t v_idx,
        float new_dist,
        uint32_t new_pred,
        float src_dist) {

    volatile uint64_t* dist_pred = (volatile uint64_t*)raw_dist_pred;
    uint64_t new_packed = pack_dist_pred(new_dist, new_pred);
    uint64_t old_packed = dist_pred[v_idx];
    int updated = 0;

    /* Loop while the candidate is accepted over the value in memory. */
    while (relax_accepts(new_packed, old_packed, src_dist)) {
#ifdef _MSC_VER
        // cppcheck-suppress misra-c2012-11.3
        uint64_t prev = (uint64_t)_InterlockedCompareExchange64(
            (volatile long long*)&dist_pred[v_idx],
            (long long)new_packed,
            (long long)old_packed
        );
#else
        uint64_t prev = old_packed;
        __atomic_compare_exchange_n(
            &dist_pred[v_idx],
            &prev,            /* expected (updated on failure) */
            new_packed,       /* desired  */
            0,                /* weak=false */
            __ATOMIC_SEQ_CST,
            __ATOMIC_SEQ_CST
        );
        /* After CAS, prev holds the value that was in memory */
#endif

        if (prev == old_packed) {
            updated = 1;  /* CAS succeeded */
            break;
        }
        old_packed = prev;  /* Retry with updated value */
    }

    return updated;
}

/* ------------------------------------------------------------------ */
/*  Bounded predecessor walk                                           */
/*                                                                     */
/*  Number of vertices on the chain target -> ... -> source.           */
/*  Returns 0 when the chain ends in a cell without a predecessor      */
/*  (no path), and -1 when it has not reached the source after         */
/*  max_vertices vertices: a simple path visits each cell at most      */
/*  once, so that chain runs in a cycle. Pass max_vertices = the cell  */
/*  count of the raster.                                               */
/* ------------------------------------------------------------------ */

/*  The array comes in as void*, like every other array in this file: */
/*  Cython's uint64 (numpy's npy_uint64) is 'unsigned long' on macOS   */
/*  and Linux but uint64_t is 'unsigned long long' on macOS, and C++   */
/*  does not convert between pointers to the two.                      */
static inline int64_t pred_chain_length(const void* raw_dist_pred,
                                        uint64_t source, uint64_t target,
                                        uint64_t max_vertices) {
    const uint64_t* dist_pred = (const uint64_t*)raw_dist_pred;
    uint64_t current = target;
    uint64_t count = 1;
    uint32_t pred;
    while (current != source) {
        if (count >= max_vertices) {
            return -1;
        }
        pred = unpack_pred(dist_pred[current]);
        if (pred == 0xFFFFFFFFU) {
            return 0;
        }
        current = (uint64_t)pred;
        count += 1;
    }
    return (int64_t)count;
}

/* ------------------------------------------------------------------ */
/*  Atomic fetch-and-add for dynamic work distribution                 */
/*                                                                     */
/*  Returns the OLD value before addition.                             */
/*  Used by persistent thread pool to grab work chunks atomically.     */
/* ------------------------------------------------------------------ */

static inline int atomic_fetch_add_int(volatile int* addr, int val) {
#ifdef _MSC_VER
    // cppcheck-suppress misra-c2012-11.3
    return (int)_InterlockedExchangeAdd((volatile long*)addr, (long)val);
#else
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
#endif
}

/* ------------------------------------------------------------------ */
/*  Atomic byte exchange (test-and-set / clear for pending markers)    */
/*                                                                     */
/*  Returns the OLD value. Full-fence RMW on both platforms, which     */
/*  makes the producer (CAS-dist then TAS-pending) / consumer          */
/*  (clear-pending then load-dist) protocol lossless: if the producer  */
/*  observes pending==1, the consumer's subsequent dist load is        */
/*  guaranteed to observe the producer's earlier CAS.                  */
/* ------------------------------------------------------------------ */

static inline uint8_t atomic_exchange_u8(volatile uint8_t* addr, uint8_t val) {
#ifdef _MSC_VER
    // cppcheck-suppress misra-c2012-11.3
    return (uint8_t)_InterlockedExchange8((volatile char*)addr, (char)val);
#else
    return __atomic_exchange_n(addr, val, __ATOMIC_SEQ_CST);
#endif
}

/* ------------------------------------------------------------------ */
/*  Sense-reversing barrier for persistent thread synchronization      */
/*                                                                     */
/*  Classic sense-reversing barrier using atomics.                     */
/*  Each thread has a private local_sense variable.                    */
/*  The last thread to arrive flips the global sense, releasing all.   */
/*                                                                     */
/*  IMPORTANT: local_sense must be initialized to 0 by each thread.   */
/*  The barrier alternates sense between 0 and 1 across invocations,  */
/*  preventing the ABA problem.                                        */
/*                                                                     */
/*  Platform support:                                                   */
/*    GCC/Clang: __atomic_add_fetch / __atomic_store_n / __atomic_load_n */
/*    MSVC:      _InterlockedIncrement / _InterlockedExchange / volatile reads */
/* ------------------------------------------------------------------ */

static inline void thread_barrier_wait(
        volatile int* arrive_count,
        volatile int* sense,
        int num_threads,
        int* local_sense) {

    /* Flip my local sense for this barrier phase */
    *local_sense = 1 - *local_sense;

    /* Atomically increment arrival count */
#ifdef _MSC_VER
    // cppcheck-suppress misra-c2012-11.3
    long val = _InterlockedIncrement((volatile long*)arrive_count);
#else
    int val = __atomic_add_fetch(arrive_count, 1, __ATOMIC_SEQ_CST);
#endif

    if (val == num_threads) {
        /* Last thread: reset counter and flip global sense to release waiters */
#ifdef _MSC_VER
        // cppcheck-suppress misra-c2012-11.3
        _InterlockedExchange((volatile long*)arrive_count, 0);
        // cppcheck-suppress misra-c2012-11.3
        _InterlockedExchange((volatile long*)sense, (long)*local_sense);
#else
        __atomic_store_n(arrive_count, 0, __ATOMIC_SEQ_CST);
        __atomic_store_n(sense, *local_sense, __ATOMIC_SEQ_CST);
#endif
    } else {
        /* Not the last: spin until global sense matches my local sense */
#ifdef _MSC_VER
        while (*((volatile int*)sense) != *local_sense) {
            _mm_pause();  /* Reduce power and improve latency on x86 */
        }
#else
        while (__atomic_load_n(sense, __ATOMIC_SEQ_CST) != *local_sense) {
#if defined(__aarch64__) || defined(__arm64__)
            __asm__ __volatile__("yield" ::: "memory");
#else
            __asm__ __volatile__("pause" ::: "memory");
#endif
        }
#endif
    }
}

#endif /* ATOMIC_CAS_H */

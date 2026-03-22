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
/*  CAS-based distance+predecessor update                              */
/*                                                                     */
/*  Returns 1 if the update was applied, 0 otherwise.                  */
/*  No ABA risk: distances only decrease monotonically.                */
/* ------------------------------------------------------------------ */

static inline int atomic_try_update_dist_pred(
        volatile void* raw_dist_pred,
        uint64_t v_idx,
        float new_dist,
        uint32_t new_pred) {

    volatile uint64_t* dist_pred = (volatile uint64_t*)raw_dist_pred;
    uint64_t new_packed = pack_dist_pred(new_dist, new_pred);
    uint64_t old_packed = dist_pred[v_idx];
    int updated = 0;

    /* Loop while our new distance is strictly better (smaller packed value). */
    while (new_packed < old_packed) {
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
            __asm__ __volatile__("pause" ::: "memory");
        }
#endif
    }
}

#endif /* ATOMIC_CAS_H */

/**
 * @file    milk_compiler.h
 * @brief   GCC performance hint macros
 *
 * Centralizes compiler-specific attributes and
 * built-in wrappers used throughout milk for
 * runtime performance optimization.
 *
 * All macros are no-ops on non-GCC compilers.
 *
 * Included automatically via milkDebugTools.h.
 */

#ifndef MILK_COMPILER_H
#define MILK_COMPILER_H


/* ========================================
 *  Branch prediction hints
 * ======================================== */

/**
 * @brief Hint that condition is usually true
 *
 * Use on fast-path checks:
 *   if (LIKELY(frame_ready)) { process(); }
 */
#ifdef __GNUC__
#    define LIKELY(x) __builtin_expect(!!(x), 1)
#else
#    define LIKELY(x) (!!(x))
#endif

/**
 * @brief Hint that condition is usually false
 *
 * Use on error-handling branches:
 *   if (UNLIKELY(err != 0)) { handle(); }
 */
#ifdef __GNUC__
#    define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#    define UNLIKELY(x) (!!(x))
#endif


/* ========================================
 *  Function attributes
 * ======================================== */

/**
 * @brief Mark function as performance-critical
 *
 * GCC optimizes more aggressively and places
 * the function in .text.hot for better icache
 * locality. Use on fpsexec() compute functions
 * and inner-loop helpers.
 */
#ifdef __GNUC__
#    define MILK_HOT __attribute__((hot))
#else
#    define MILK_HOT
#endif

/**
 * @brief Mark function as rarely executed
 *
 * GCC moves the function to .text.unlikely,
 * keeping it away from hot code paths. Use on
 * error handlers, init/cleanup, signal handlers.
 */
#ifdef __GNUC__
#    define MILK_COLD __attribute__((cold))
#else
#    define MILK_COLD
#endif

/**
 * @brief Function has no side effects
 *
 * Function reads memory but does not modify
 * any observable state. GCC may eliminate
 * redundant calls. Use on query functions
 * like image_ID(), variable_ID().
 */
#ifdef __GNUC__
#    define MILK_PURE __attribute__((pure))
#else
#    define MILK_PURE
#endif

/**
 * @brief Function depends only on arguments
 *
 * Stricter than MILK_PURE: function does not
 * read global memory. GCC may hoist calls out
 * of loops. Use on math helpers and type-size
 * lookups.
 */
#ifdef __GNUC__
#    define MILK_CONST __attribute__((const))
#else
#    define MILK_CONST
#endif

/**
 * @brief Force-inline all callees
 *
 * GCC inlines every function called from the
 * decorated function. Use on small driver /
 * wrapper functions that dispatch to helpers.
 * Avoid on large functions (icache bloat).
 */
#ifdef __GNUC__
#    define MILK_FLATTEN __attribute__((flatten))
#else
#    define MILK_FLATTEN
#endif


/* ========================================
 *  Memory and loop hints
 * ======================================== */

/**
 * @brief Software prefetch
 *
 * @param addr  Address to prefetch
 * @param rw    0 = read, 1 = write
 * @param loc   Locality 0-3 (3 = keep in all
 *              cache levels)
 *
 * Use for sequential walks through large
 * arrays in telemetry/logging loops.
 */
#ifdef __GNUC__
#    define MILK_PREFETCH(addr, rw, loc) __builtin_prefetch((addr), (rw), (loc))
#else
#    define MILK_PREFETCH(addr, rw, loc) ((void) 0)
#endif

/**
 * @brief Assert no loop-carried dependencies
 *
 * Place immediately before a for-loop to let
 * GCC vectorize even when it cannot prove
 * independence. Only use when you are certain
 * the loop body has no cross-iteration deps.
 *
 * Example:
 *   MILK_IVDEP
 *   for (long i = 0; i < n; i++)
 *       dst[i] = src[i] * scale;
 */
#ifdef __GNUC__
#    define MILK_IVDEP _Pragma("GCC ivdep")
#else
#    define MILK_IVDEP
#endif

/**
 * @brief Align variable for SIMD
 *
 * @param n  Alignment in bytes (e.g. 32 for
 *           AVX, 64 for AVX-512)
 *
 * Use on stack-allocated arrays in tight loops:
 *   float buf[256] MILK_ALIGNED(32);
 */
#ifdef __GNUC__
#    define MILK_ALIGNED(n) __attribute__((aligned(n)))
#else
#    define MILK_ALIGNED(n)
#endif


/* ========================================
 *  Pointer qualification
 * ======================================== */

/**
 * @brief Portable restrict qualifier
 *
 * Maps to C99 restrict or GCC __restrict.
 * Use on non-aliased data pointer params:
 *   void compute(
 *       float * MILK_RESTRICT dst,
 *       const float * MILK_RESTRICT src,
 *       long n);
 */
#define MILK_RESTRICT __restrict


/* ========================================
 *  Alignment hints
 * ======================================== */

/**
 * @brief Assert pointer is aligned
 *
 * Use on array pointers to allow GCC/Clang
 * to generate aligned vector instructions (e.g. AVX512).
 */
#ifdef __GNUC__
#    define MILK_ASSUME_ALIGNED(ptr) __builtin_assume_aligned((ptr), 64)
#else
#    define MILK_ASSUME_ALIGNED(ptr) (ptr)
#endif


#endif /* MILK_COMPILER_H */

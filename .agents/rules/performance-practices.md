---
trigger: always_on
---

# Runtime Performance Practices

Use the macros from `milk_compiler.h` (included
automatically via `milkDebugTools.h`) to give GCC
optimization hints. All macros are no-ops on
non-GCC compilers.

## Pointer Qualification & Alignment

- Use `restrict` (or `MILK_RESTRICT`) on all
  non-aliased pixel/array data pointer parameters
  in `compute_function` signatures and compute-heavy helper functions. This allows GCC to
  vectorize loops over those pointers.
- Use `MILK_ASSUME_ALIGNED(ptr)` to inform GCC
  that a restricted pointer is aligned to a 64-byte
  boundary, which forces the compiler to use
  faster, strictly aligned vector instructions
  (e.g., AVX `vmovaps` vs `vmovups`).
- **Not** required on `char*` parameters used as
  string names or file paths — the benefit is
  negligible and it hurts readability.

## Branch Prediction

- Use `UNLIKELY()` on error-handling branches:
  `if (UNLIKELY(err != 0)) { ... }`
- Use `LIKELY()` on fast-path conditions:
  `if (LIKELY(frame_ready)) { ... }`
- Only annotate branches where the bias is
  strong and the code is on a hot path.

## Function Attributes

- Use `MILK_HOT` on `fpsexec()` / compute
  functions and inner-loop helpers.
- Use `MILK_COLD` on error handlers,
  initialization, cleanup, and signal handlers.
- Use `MILK_PURE` on side-effect-free query
  functions (e.g., `image_ID()`,
  `variable_ID()`).
- Use `MILK_CONST` on functions that depend only
  on their arguments (math helpers, type-size
  lookups).
- Use `MILK_FLATTEN` on small wrapper functions
  that dispatch to several helpers. Avoid on
  large functions (icache bloat).
- **Inline Small Helpers**: Refactor small, heavily-called inner-loop operations (like single-pixel evaluators) into `static inline` functions to completely eliminate call overhead.

## Float vs Double

- Use `0.5f` not `0.5` in float arithmetic to
  avoid implicit promotion to double.
- Keep inner-loop arithmetic in `float` when
  the data is `float` — do not mix precisions
  unnecessarily.
- Use `sqrtf()` not `sqrt()` for float data;
  `fabsf()` not `fabs()`; `floorf()` not `floor()`;
  `ceilf()` not `ceil()`. Each double variant
  promotes its argument and returns double.

## Math & Transcendentals

- Avoid computing expensive scalar transcendentals
  (e.g. `sinf()`, `cosf()`, `expf()`) inside tight
  SIMD or OpenMP inner loops if possible.
- **Complex Math**: Replace polar coordinate conversions (`sqrt()` + `atan2()`) with direct algebraic alternatives (e.g., conjugate multiplication for complex division) to eliminate transcendental overhead in tight loops.
- If a few distinct angles or phases are used repeatedly,
  pre-compute these values during initialization into a
  static or `MILK_ALIGNED` array (LUT) and replace the
  expensive function calls with fast array lookups in
  the hot path.
- **Specialize `pow()`**: when the exponent is known
  at setup time, dispatch to a fast path:
  - exponent 1.0 → linear (no call)
  - exponent 0.5 → `sqrtf(x)`
  - exponent 2.0 → `x * x`
  - other → `powf(x, exp)` (not `pow()`)
- Use `powf()` instead of `pow()` for float data
  — `pow()` promotes to double and returns double.
- **Integer power-of-2**: never call `pow(2, n)`
  for integer `n` — use `1 << n` instead.
  `pow()` is a floating-point transcendental; the
  bit-shift is exact and orders of magnitude faster.

## Memory Allocation

- **Never `malloc`/`free` inside a per-frame
  compute loop.** Move allocations into the
  initialization phase (e.g., `dmcomb_init()`,
  stored in a state struct) and reuse buffers
  across iterations.
- Use `calloc()` for zero-initialized state
  structs to avoid uninitialized-memory bugs.
- Prefer stack allocation (`MILK_ALIGNED(32)`)
  for small, fixed-size arrays used in tight
  loops.

## Typed Fast-Paths

- When a stream-processing loop handles the
  common case (`_DATATYPE_FLOAT` or
  `_DATATYPE_UINT16`), add a typed fast-path
  with `MILK_RESTRICT` + `MILK_ASSUME_ALIGNED`
  on the pointer aliases. Fall through to
  `memcpy` for other types.
- Hoist invariant FPS parameter dereferences
  (`*ptr`) into local variables before the
  inner loop to avoid repeated pointer chasing.

## Struct Copying & Data Movement

- Replace manual struct copies or small array copies
  in hot paths with `__builtin_memcpy(dest, src,
  size)`. This allows GCC to emit optimal inline
  move instructions instead of calling libc `memcpy`
  or generating sub-optimal loop code.
- In stream-copy fallback paths that handle any
  datatype, copy via `.raw` (not `.F` or other typed
  union members) and use `__builtin_memcpy`:
  ```c
  __builtin_memcpy(
      imgout.im->array.raw,
      imgin.im->array.raw,
      byte_copy_size);
  ```
  Accessing `.F` for non-float streams is a latent
  type-aliasing correctness bug in addition to being
  a missed optimization opportunity.

## Datatype Dispatch

- Use `else if` (not bare `if`) chains for
  datatype dispatch. Bare `if` checks every
  type even after a match — `else if` skips
  ~9 redundant comparisons. Applies to all
  `imfunctions.c`-style per-type loops.

## BLAS for Matrix Operations

- **Never** hand-write matrix-vector or
  matrix-matrix multiply loops. Use
  `cblas_sgemv()` / `cblas_sgemm()` from
  MKL or OpenBLAS (already linked via CMake).
- Provide a plain-C fallback with `restrict`
  + `#pragma omp simd` for non-BLAS builds.
- Remove `printf()` from CPU fallback
  functions — IO in hot paths is fatal to
  latency.

## Loop Vectorization

- Use `#pragma omp for simd` (not bare
  `#pragma omp for`) on element-wise pixel
  loops — combines threading + SIMD.
- Place `MILK_IVDEP` before non-OMP loops you
  **know** have no loop-carried dependencies.
- Add `restrict` to local pixel pointer aliases:
  `float * restrict ptr = img->array.F;`
- Use `MILK_ALIGNED(32)` on stack arrays used
  in tight loops to enable AVX-width
  vectorization.
- Use `MILK_PREFETCH(addr, rw, locality)` for
  sequential walks through large arrays.
- Run `cmake .. -DVEC_REPORT=ON` to see which
  loops GCC couldn't vectorize and why.
- **Match loop index type to bound type.**
  A type mismatch prevents GCC from vectorizing
  (up to 8× slowdown). See
  `naming-conventions.md` §3.8 for the full
  type-matching table and examples.

## CPU Core Pinning (Thread Affinity)

- **Do NOT** manually call `sched_setaffinity` inside
  custom modules.
- The `milk` framework native `PROCESSINFO`
  architecture handles this automatically via the
  `procinfo->CPUmask` attribute. Leverage this existing
  system when binding streams to isolated processing cores.

## IPC & Synchronization

- **Semaphores**: Avoid recursive lock/unlock/post operations on semaphores within high-frequency IPC loops. Batch post operations or iterate across reader arrays natively to minimize kernel context switches.

## What NOT to Do

- Do not use `__attribute__((always_inline))`
  on large functions — let GCC and LTO decide.
- Do not use `register` — it is ignored by
  modern GCC and deprecated in C17.
- Do not scatter `#pragma GCC optimize` across
  files — use CMake flags instead.
- Do not place `printf()`, `fprintf()`, or
  `fflush()` in compute hot paths. Guard
  diagnostic output with `if (VERBOSE > 0)`.
  Even rare events (timeouts, frame saves) should
  be guarded — the branch predictor will treat
  them as free when VERBOSE is 0.
- Do not write naive loops for matrix
  operations when BLAS is available — the
  performance difference is orders of
  magnitude.
- Do not use standalone `if` for datatype
  dispatch — always use `else if` chains.
- Do not place `printf()` or `fflush()` inside
  `#pragma omp parallel` regions. In a parallel
  region all threads execute the call,
  causing N simultaneous kernel writes.
  Guard with `if(VERBOSE > 0)` and/or
  `#pragma omp master`.

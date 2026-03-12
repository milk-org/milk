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
  in compute-heavy functions. This allows GCC to
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

## Float vs Double

- Use `0.5f` not `0.5` in float arithmetic to
  avoid implicit promotion to double.
- Keep inner-loop arithmetic in `float` when
  the data is `float` — do not mix precisions
  unnecessarily.

## Math & Transcendentals

- Avoid computing expensive scalar transcendentals 
  (e.g. `sinf()`, `cosf()`, `expf()`) inside tight
  SIMD or OpenMP inner loops if possible.
- If a few distinct angles or phases are used repeatedly,
  pre-compute these values during initialization into a
  static or `MILK_ALIGNED` array (LUT) and replace the
  expensive function calls with fast array lookups in
  the hot path.

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

## CPU Core Pinning (Thread Affinity)

- **Do NOT** manually call `sched_setaffinity` inside
  custom modules.
- The `milk` framework native `PROCESSINFO` 
  architecture handles this automatically via the
  `procinfo->CPUmask` attribute. Leverage this existing
  system when binding streams to isolated processing cores.

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

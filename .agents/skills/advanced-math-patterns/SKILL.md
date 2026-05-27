---
name: advanced-math-patterns
description: Reference for implementing high-performance mathematical and DSP operations in milk, including BLAS, FFT, and vectorization.
---

# Advanced Math Patterns in milk

When writing highly computational logic in milk (especially for Adaptive Optics), naive `for` loops can become a severe bottleneck. This skill provides standard patterns for high-performance math, leveraging `milk`'s internal compiler hinting macros (`milkDebugTools.h`) and standard scientific libraries.

## 1. Matrix Operations (BLAS)

**NEVER** write manual nested `for` loops for matrix-vector multiplication (MVM) or matrix-matrix multiplication (SGEMM/DGEMM). Milk integrates with standard BLAS (MKL or OpenBLAS).

### Matrix-Vector Multiplication (MVM)

To multiply a matrix $A$ by a vector $x$ to get $y$ ($y = \alpha Ax + \beta y$):

```c
#include <cblas.h>

// For float data:
// cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
cblas_sgemv(CblasRowMajor, CblasNoTrans,
            rows, cols,
            1.0f, matrix_ptr, cols,
            vector_in_ptr, 1,
            0.0f, vector_out_ptr, 1);
```

### Matrix-Matrix Multiplication (SGEMM)

To multiply matrix $A$ and matrix $B$ to get $C$ ($C = \alpha AB + \beta C$):

```c
#include <cblas.h>

// cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M_rows, N_cols, K_inner,
            1.0f, matrixA_ptr, K_inner,
            matrixB_ptr, N_cols,
            0.0f, matrixC_ptr, N_cols);
```

_Note: Make sure your module links against `${BLAS_LIBRARIES}` in its `CMakeLists.txt`._

## 2. Fast Fourier Transforms (FFT)

Do not implement custom FFT logic or call `fftw` directly if you can avoid it. Milk has a dedicated `milkfft` plugin that wraps FFT operations safely into the stream processing architecture.

- Include `fft/fft.h` (from `plugins/milk-extra-src/fft`).
- In your `CMakeLists.txt`, link against `milkfft` (or `milkfft_compute` for standalone executables).
- Use `dofft()` or `init_fftwplan()` provided by the `milkfft` API to perform distributed or optimized 2D/3D transforms on `IMGID` streams.

## 3. Loop Vectorization (SIMD)

When writing element-wise math across large images, you must strictly guide GCC to vectorize the loop. If you miss one of these steps, GCC will fallback to scalar math (an 8x slowdown).

1. **Restrict Pointers:** Always declare local aliases using `MILK_RESTRICT` (or standard `restrict`) so the compiler knows output does not overlap input.
2. **Alignment:** Use `MILK_ASSUME_ALIGNED(ptr)` if your arrays are 64-byte aligned (true for all milk streams).
3. **Index Type Matching:** The loop index (`ii`) MUST match the bounds variable type (`uint64_t`). A type mismatch (`int ii = 0; ii < uint64_size`) completely breaks GCC auto-vectorization.
4. **OpenMP + SIMD:** Use OpenMP pragmas to distribute the loop across cores AND vector lanes.

```c
float * restrict in_ptr = inimg->im->array.F;
float * restrict out_ptr = outimg->im->array.F;
MILK_ASSUME_ALIGNED(in_ptr);
MILK_ASSUME_ALIGNED(out_ptr);

uint64_t xysize = inimg->mdt->size[0] * inimg->mdt->size[1];

#pragma omp parallel for simd
for (uint64_t ii = 0; ii < xysize; ii++)
{
    out_ptr[ii] = in_ptr[ii] * 2.5f;
}
```

_Use `MILK_PREFETCH(addr, rw, locality)` inside loops that walk sequentially through massive multi-gigabyte arrays._

## 4. Avoiding Transcendentals in Hot Loops

Mathematical function calls like `sinf()`, `cosf()`, `pow()`, and `sqrtf()` inside a pixel loop will destroy performance.

- **`pow()` vs `powf()`:** Never use `pow()` or `sin()` on `float` data! It promotes the float to double, does expensive double-precision math, and truncates back to float. Always use `powf()`, `sinf()`, `sqrtf()`.
- **Integer Powers:** Never use `powf(x, 2)`. Use `x * x`. Never use `pow(2, n)` for bitshifts; use `1 << n`.
- **Precomputation (Look-Up Tables):** If calculating phase angles, pre-compute `sinf()` and `cosf()` arrays in `INSERT_STD_PROCINFO_COMPUTEFUNC_INIT` and do an array lookup in the main loop.
- **Complex Arithmetic:** Replace expensive polar coordinate conversions (`sqrt()` + `atan2()`) with direct algebraic alternatives (e.g., conjugate multiplication for division).

## 5. Datatype Fast-Paths and Memory Movement

Milk streams can contain many different datatypes. Do not use generic fallback paths for hot streams.

- **Typed Fast-Paths:** If a stream-processing loop handles a common case (`_DATATYPE_FLOAT` or `_DATATYPE_UINT16`), write an explicit `else if` branch for that type, using `restrict` and SIMD.
- **`else if` Dispatch:** Always use `else if` chains (not bare `if`) for datatype dispatch to skip redundant checks.
- **Data Copying:** In fallback paths that handle any datatype, do not copy pixel-by-pixel. Use `__builtin_memcpy` via the `.raw` union member to let GCC emit optimal inline move instructions:

  ```c
  __builtin_memcpy(outimg.im->array.raw, inimg.im->array.raw, byte_copy_size);
  ```

## 6. What NOT to Do

- **No `malloc` / `free`:** Never allocate memory inside a per-frame compute loop. Allocate once during setup and reuse the buffers.
- **No `printf` / `fflush`:** I/O blocks the thread. Guard all print statements with `if (UNLIKELY(VERBOSE > 0))`. Do not place prints inside `#pragma omp parallel` blocks!
- **No Manual `sched_setaffinity`:** Do not attempt to pin CPU threads yourself. The `milk` `PROCESSINFO` architecture handles this automatically via the `procinfo->CPUmask` parameter.

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

_Note: Add `// MILK_CMAKE_REQUEST_BLAS` (or `MILK_CMAKE_MANDATE_BLAS`) to the source file; see `docs/developer/dependency_system.md`._

## 2. Fast Fourier Transforms (FFT)

Do not implement custom FFT logic. In calculator expressions and `milk.calc` units, use `fft`, `ifft`
and `fftshift` ([Calculator](../../../docs/guide/calculator.md)): they run cached FFTW plans
(real-to-complex for real input, SIMD, on the calculator's thread team) in
`src/calc/linalg/mc_fft_stack.c`. Plugin code can use the `milkfft` plugin
(`plugins/milk-extra-src/fft`, `dofft.h`: `do2dfft`, `do2drfft`, ... on image names; link `milkfft`),
which plans every call and runs single-threaded.

## 3. Loop Vectorization (SIMD)

Follow [performance-practices.md](../../rules/performance-practices.md) (restrict, alignment, index
types, thread counts with `milk_omp_team_size`). Shared-memory stream data is only 8-byte aligned;
local images are 64-byte aligned.

## 4. Avoiding Transcendentals in Hot Loops

Mathematical function calls like `sinf()`, `cosf()`, `pow()`, and `sqrtf()` inside a pixel loop will destroy performance.

- **`pow()` vs `powf()`:** Never use `pow()` or `sin()` on `float` data! It promotes the float to double, does expensive double-precision math, and truncates back to float. Always use `powf()`, `sinf()`, `sqrtf()`.
- **Integer Powers:** Never use `powf(x, 2)`. Use `x * x`. Never use `pow(2, n)` for bitshifts; use `1 << n`.
- **Precomputation (Look-Up Tables):** If calculating phase angles, pre-compute `sinf()` and `cosf()` arrays before `MILK_PROCINFO_LOOPSTART(self)` and do an array lookup in the main loop.
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

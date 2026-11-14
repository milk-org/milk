---
name: optimize-compute-function
description: Systematic methodology for optimizing
  performance-critical compute functions
---

# Optimize Compute Functions

This skill provides a step-by-step methodology for
optimizing `<stem>_compute()` and other hot-path
compute functions using the project's performance
macros and tools. The rules each phase applies are
in [`performance-practices`](../../rules/performance-practices.md).

## When to Use

- After creating a new compute function
- When profiling reveals a performance bottleneck
- During periodic optimization audits
- User asks to "optimize" or "speed up" a function

## Phase 1 — Identify Hot Functions

Before optimizing, confirm which functions are
actually hot: loop rates with `milk-procinfo-list`
([performance.md §1](../../../docs/operations/performance.md#1-measuring-performance)).

Focus on functions that run at high frame rates
(>100 Hz) or process large data (>1 MB/frame).

## Phase 2 — Apply Function Attributes

Add performance macros from `milk_compiler.h`:

```c
/**
 * xyz_compute() - Core computation for XYZ
 * @self:  Runnable descriptor of the unit
 */
static MILK_HOT errno_t xyz_compute(const MILK_RUNNABLE *self)
{
    // ...
}
```

Which macro goes where: `performance-practices`, *Function Attributes*.

## Phase 3 — Pointer Qualification

Add `restrict` and alignment hints to pointer
parameters in compute functions:

```c
static void process_frame(
    float * MILK_RESTRICT in,
    float * MILK_RESTRICT out,
    long    nelem
)
{
    in  = MILK_ASSUME_ALIGNED(in);
    out = MILK_ASSUME_ALIGNED(out);

    for (long i = 0; i < nelem; i++)
    {
        out[i] = in[i] * gain;
    }
}
```

Checklist: `performance-practices`, *Pointer Qualification & Alignment*.

## Phase 4 — Fix Float/Double Promotions

Search for implicit promotions:

```c
// BAD — promotes to double
float result = x * 0.5;
float s = sqrt(x);

// GOOD — stays in float
float result = x * 0.5f;
float s = sqrtf(x);
```

Substitutions: `performance-practices`, *Float vs Double* and *Math & Transcendentals*.

## Phase 5 — Vectorization Check

Build with vectorization report:

```bash
cd _build
cmake .. -DVEC_REPORT=ON
cmake --build . -- -j$(nproc) 2>&1 \
  | grep "not vectorized" \
  | grep "target_file.c"
```

Common vectorization blockers and fixes:

| Reason                 | Fix                         |
| ---------------------- | --------------------------- |
| "possible aliasing"    | Add `restrict`              |
| "call in loop body"    | Inline the callee           |
| "unsupported data-ref" | Break complex struct access |
| "data dependency"      | Restructure loop            |

## Phase 6 — BLAS for Matrix Operations

Never hand-write matrix multiply loops:

```c
// BAD — naive loop
for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
        for (k = 0; k < K; k++)
            C[i*N+j] += A[i*K+k] * B[k*N+j];

// GOOD — BLAS call
cblas_sgemm(CblasRowMajor, CblasNoTrans,
    CblasNoTrans, M, N, K,
    1.0f, A, K, B, N, 0.0f, C, N);
```

## Phase 7 — Memory and I/O Audit

Apply `performance-practices`, *Memory Allocation* and *What NOT to Do*.

## Phase 8 — Verify with Assembly

Run the `/inspect-machine-code` workflow to
verify optimizations took effect in the generated
assembly.

## Quick Optimization Checklist

The `performance-practices` rule is the checklist.

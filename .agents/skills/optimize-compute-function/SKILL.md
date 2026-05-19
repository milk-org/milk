---
name: optimize-compute-function
description: Systematic methodology for optimizing
  performance-critical compute functions
---

# Optimize Compute Functions

This skill provides a step-by-step methodology for
optimizing `fpsexec()` and other hot-path compute
functions using the project's performance macros
and tools.

## When to Use

- After creating a new compute function
- When profiling reveals a performance bottleneck
- During periodic optimization audits
- User asks to "optimize" or "speed up" a function

## Phase 1 — Identify Hot Functions

Before optimizing, confirm which functions are
actually hot:

```bash
# Check loop rate via processinfo
echo "procCTRL" | milk-cli 2>/dev/null
```

Focus on functions that run at high frame rates
(>100 Hz) or process large data (>1 MB/frame).

## Phase 2 — Apply Function Attributes

Add performance macros from `milk_compiler.h`:

```c
/**
 * fpsexec - Core computation for XYZ
 */
MILK_HOT
static errno_t fpsexec(void)
{
    // ...
}
```

| Macro | Where to Use |
|-------|-------------|
| `MILK_HOT` | `fpsexec()`, inner-loop helpers |
| `MILK_COLD` | Error handlers, init, cleanup |
| `MILK_PURE` | Side-effect-free query functions |
| `MILK_CONST` | Functions depending only on args |
| `MILK_FLATTEN` | Small wrappers dispatching to helpers |

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

**Checklist:**
- [ ] All non-aliased pixel pointers have `restrict`
- [ ] Hot-path pointers have `MILK_ASSUME_ALIGNED`
- [ ] String/path `char*` parameters do NOT have
  `restrict` (not worth it)

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

**Common substitutions:**

| Double Version | Float Version |
|---------------|---------------|
| `sqrt(x)` | `sqrtf(x)` |
| `pow(x, y)` | `powf(x, y)` |
| `sin(x)` | `sinf(x)` |
| `cos(x)` | `cosf(x)` |
| `exp(x)` | `expf(x)` |
| `fabs(x)` | `fabsf(x)` |
| `floor(x)` | `floorf(x)` |
| `0.5` | `0.5f` |
| `1.0` | `1.0f` |

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

| Reason | Fix |
|--------|-----|
| "possible aliasing" | Add `restrict` |
| "call in loop body" | Inline the callee |
| "unsupported data-ref" | Break complex struct access |
| "data dependency" | Restructure loop |

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

- [ ] No `malloc`/`free` inside per-frame loops
- [ ] No `printf`/`fprintf` in hot paths
  (guard with `if (VERBOSE > 0)`)
- [ ] No `fflush` inside loops
- [ ] Pre-allocate buffers in init, reuse
- [ ] Use `calloc()` for zero-initialized state

## Phase 8 — Verify with Assembly

Run the `/inspect-machine-code` workflow to
verify optimizations took effect in the generated
assembly.

## Quick Optimization Checklist

- [ ] `MILK_HOT` on compute function
- [ ] `restrict` on all pixel pointer params
- [ ] `MILK_ASSUME_ALIGNED` on hot pointers
- [ ] Float literals use `f` suffix (`0.5f`)
- [ ] Float math uses `f` variants (`sqrtf`)
- [ ] No `pow(2, n)` — use `1 << n`
- [ ] BLAS for matrix operations
- [ ] No alloc/IO in hot loop
- [ ] `else if` for datatype dispatch (not `if`)
- [ ] `MILK_IVDEP` or `#pragma omp for simd`
  on element-wise loops

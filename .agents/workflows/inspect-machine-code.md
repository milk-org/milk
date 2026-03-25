---
description: Build with static LTO, inspect machine code for missed optimizations
---

# Inspect Machine Code for Optimization Opportunities

Build standalone `fpsexec` executables with static
linking and LTO, then inspect both **compiler
diagnostics** and **generated assembly** for missed
optimization opportunities. Produce an actionable
report with source-level fix suggestions.

> **When to use:** After performance-sensitive code
> changes, or periodically to audit hot-path compute
> functions for codegen quality.

---

## Prerequisites

- GCC with LTO support (standard on the project)
- `objdump` (from binutils)
- Existing milk source tree at `/home/oguyon/src/milk`

---

## Phase 1 — Static LTO Build with Vectorization Report

Build in a **dedicated directory** so the normal build
is undisturbed.

// turbo-all

1. Create and configure the LTO inspection build:

```bash
mkdir -p /home/oguyon/src/milk/_build_lto_inspect && \
cd /home/oguyon/src/milk/_build_lto_inspect && \
cmake .. \
  -DUSE_STATIC_LTO=ON \
  -DVEC_REPORT=ON \
  -DCMAKE_BUILD_TYPE=Release
```

2. Build and capture vectorization diagnostics:

```bash
cd /home/oguyon/src/milk/_build_lto_inspect && \
cmake --build . -- -j$(nproc) 2>&1 \
  | tee vec_report_full.log
```

3. If the build fails, fix errors and re-run step 2.

---

## Phase 2 — Analyze Compiler Diagnostics

Parse the GCC `-fopt-info-vec-missed` output for
vectorization failures. This phase is **more
actionable** because GCC explains *why* each loop
was not vectorized and points to source lines.

4. Extract unique vectorization failures:

```bash
cd /home/oguyon/src/milk/_build_lto_inspect && \
grep -E 'note:.*not vectorized|missed:' \
  vec_report_full.log \
  | sed 's|.*/src/|src/|' \
  | sort -t: -k1,1 -k2,2n -u \
  > vec_missed_summary.txt
```

5. Categorize the failures by reading
   `vec_missed_summary.txt`. Common categories:

| GCC reason | Typical fix |
|-----------|------------|
| "not vectorized: call in loop body" | Inline the callee or move it out of the loop |
| "not vectorized: unsupported data-ref" | Add `restrict` to pointer params |
| "not vectorized: possible aliasing" | Add `MILK_RESTRICT` and `MILK_ASSUME_ALIGNED` |
| "not vectorized: not enough iterations" | Usually fine — small loops are not worth vectorizing |
| "not vectorized: data dependency" | Restructure loop to break dependency chain |
| "missed: couldn't vectorize loop" | Check for mixed types, function pointers, or volatile |

6. For each failure in a **hot-path file** (compute
   functions, stream processors, FPS exec functions),
   determine the source fix using the project's
   `performance-practices.md` rules.

---

## Phase 3 — Disassemble and Inspect Assembly

This phase catches problems GCC **never warns about**:
precision bugs, missed inlining across modules,
expensive math calls, I/O in hot paths.

7. Select target executables. Focus on the ones most
   performance-critical:

```bash
cd /home/oguyon/src/milk/_build_lto_inspect && \
TARGETS=$(find . \
  \( -name 'milk-fpsexec-*' \
     -o -name 'cacao-fpsexec-*' \) \
  -executable -type f)
echo "Found $(echo "$TARGETS" | wc -l) executables"
```

8. Disassemble each target:

```bash
cd /home/oguyon/src/milk/_build_lto_inspect && \
mkdir -p asm_output && \
for BIN in $TARGETS; do
  BASENAME=$(basename "$BIN")
  objdump -d -M intel --no-show-raw-insn "$BIN" \
    > "asm_output/${BASENAME}.asm"
done
```

9. Search for **code-gen anti-patterns** across all
   disassembly files:

```bash
cd /home/oguyon/src/milk/_build_lto_inspect/asm_output

echo "=== Float-Double promotions ==="
grep -rn 'cvtss2sd\|cvtsd2ss' *.asm | head -30

echo "=== PLT calls (not inlined by LTO) ==="
grep -rn 'call.*@plt' *.asm \
  | grep -v 'pthread\|sem_\|clock_\|shm_' | head -30

echo "=== Scalar math (should be float variants) ==="
grep -rn 'call.*<sqrt>\|call.*<pow>\|call.*<exp>' \
  *.asm | head -30

echo "=== Unaligned SIMD loads ==="
grep -rn 'movups\|movupd' *.asm | head -30

echo "=== I/O in possible hot paths ==="
grep -rn 'call.*<printf>\|call.*<fprintf>' \
  *.asm | head -30

echo "=== Allocation in possible hot paths ==="
grep -rn 'call.*<malloc>\|call.*<free>' \
  *.asm | head -30

echo "=== Division instructions (expensive) ==="
grep -rn 'divss\|divsd\|divps\|divpd' \
  *.asm | head -30
```

10. **Filter findings to hot functions only.** Not every
    `printf` or `malloc` is a problem — only those in
    compute loops matter. Use the function name from
    the disassembly section headers (e.g.,
    `<fpsexec>:`, `<compute_function>:`) to determine
    which findings are in hot paths.

---

## Phase 4 — Produce Optimization Report

11. For each finding, produce a report entry with:

| Field | Content |
|-------|---------|
| **File** | Source file and line (from VEC_REPORT or manual mapping) |
| **Issue** | What the compiler/assembly reveals |
| **Category** | Vectorization / Precision / Inlining / Math / I-O / Allocation / Alignment |
| **Severity** | High (hot loop) / Medium (warm path) / Low (cold path) |
| **Suggested Fix** | Concrete code change referencing `performance-practices.md` |

12. Common fix suggestions by category:

| Category | Fix |
|----------|-----|
| Vectorization / aliasing | Add `restrict` + `MILK_ASSUME_ALIGNED` to pointer params |
| Vectorization / call in loop | Inline the callee as `static inline`, or use `MILK_FLATTEN` on wrapper |
| Float↔double promotion | Use `sqrtf()` not `sqrt()`, `0.5f` not `0.5`, `powf()` not `pow()` |
| PLT calls surviving LTO | Check if the callee's library is included in `_MILK_STANDALONE_STATIC_LIBS` |
| Unaligned SIMD | Add `MILK_ASSUME_ALIGNED(ptr)` after `restrict` cast |
| Expensive division | Consider reciprocal approximation `_mm_rcp_ps` or hoist invariant divisor |
| I/O in hot path | Guard with `if (VERBOSE > 0)` |
| Allocation in hot path | Move to init phase, reuse buffers |
| `pow()` in loop | Specialize: `x*x` for exp=2, `sqrtf` for exp=0.5, `powf` for other |
| Missing `MILK_HOT` | Add `MILK_HOT` to `fpsexec()` and compute functions |

13. Present the report to the user, grouped by
    executable and severity.

---

## Phase 5 — Cleanup (optional)

14. Remove the inspection build directory when done:

```bash
rm -rf /home/oguyon/src/milk/_build_lto_inspect
```

---

## Notes

- **Filtering PLT calls**: System calls (`pthread_*`,
  `sem_*`, `clock_gettime`, `shm_open`) will always
  appear as PLT calls — ignore these. Focus on milk
  library functions that LTO should have inlined.
- **`movups` vs `movaps`**: On modern CPUs (Haswell+),
  the performance difference is negligible for L1-hot
  data. Focus on `movups` in tight inner loops over
  large arrays where icache pressure matters.
- **Build time**: Static LTO builds take significantly
  longer. Expect 2-5× the normal build time.
- **Combine with PGO**: For the most informative
  inspection, use `-DUSE_PGO=USE -DUSE_STATIC_LTO=ON`
  to see what PGO+LTO together produce. This requires
  a prior PGO GENERATE pass.

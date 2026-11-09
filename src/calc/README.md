# libmilkcalc (`src/calc/`)

Expression and n-dimensional image calculator, C extension SDK (`milk_calc_ext.h`),
and standalone `milk-fpsexec-milk-calc` (`milk.calc`) compute unit for the `milk` framework.

## Purpose

`libmilkcalc` evaluates scalar and n-dimensional image expressions via a unified pipeline:

1. **Front End (`mc_lex.c`, `mc_parse.c`, `mc_resolve.c`, `mc_func.c`)**:
   Tokenizes expressions and multi-statement `.calc` programs, builds an AST (`mc_ast.h`), expands
   inline `def` functions and `hist()`/`prev()` ring buffers, and resolves identifiers, types,
   shapes, and slice bounds.
2. **Scalar & Client-Side Fast Paths (`mc_scalar_eval.c`, `mc_scalar_ops.c`)**:
   Evaluates scalar expressions with an exact `int64_t` integer lane and provides the client-side
   zero-RPC evaluator (`mcalc_local`) for numeric and `s.`/`f.`/`p.` expressions in `libmilkbash`.
3. **Planner, C99 Code Generator & `.so` Cache (`mc_plan.c`, `mc_emit.c`, `mc_cc.c`, `mc_exec.c`)**:
   Fuses elementwise image operations, slices/views, stencils (`shift`, `roll`), `lut`, `ema`, and
   reductions into restricted C99 OpenMP/SIMD loops compiled into cached shared objects
   (`${MILK_CALC_CACHE:-${TMPDIR:-/tmp}/milk-calc-cache-<uid>}/<key>.so`).
4. **Reductions, Spatial Pre-Stages & Linear Algebra (`mc_reduce.c`, `mc_prestage.c`, `linalg/`)**:
   Single-pass full-image, masked, and axis reductions, spatial filters (`medfilt*`, `gauss`,
   `conv2`, `boxmean`, `imin3`, `imax3`), in-place BLAS/LAPACK solvers (`@`, `.T`, `svd`, `eigh`,
   `einsum`), and FFTW transforms (`fft`, `ifft`, `fftshift`).
5. **User C Extensions & `milk.calc` Unit (`mc_ext_*.c`, `units/milk_unit_calc.c`)**:
   Compiles and loads Tier 1/Tier 2 C extensions (`milk_calc_ext.h`) and runs multi-statement
   calculator programs on every frame of a trigger stream in `milk-fpsexec-milk-calc`.

---

## Source Files

The public headers (`mcalc.h`, `mcalc_host.h`, `milk_calc_ext.h`) and the entry points
(`mc_eval.c`, `mc_func.c`) stay at the top; the rest is grouped by stage: `front/` (lexer,
parser, resolver), `plan/` (planner, views, history rings, `--explain`), `jit/` (C code
generation and compilation), `run/` (execution, reductions, scalar evaluation), `ext/` (user C
extensions), `linalg/` and `units/`.

| File | Description |
| :--- | :--- |
| `mcalc.h` | Public calculator API (`mcalc_eval_*`, `mcalc_local`, `mcalc_prog_*`, `mcalc_ext_*`) |
| `mcalc_host.h` | Host callback registration (`mcalc_set_host`) for upper-tier string variables |
| `milk_calc_ext.h` | Public C extension SDK header (Tier 1 block/stencil/image & Tier 2 multi-port ops) |
| `front/mc_ast.h` | Token, expression AST (`mc_expr_t`), and statement AST (`mc_stmt_t`) definitions |
| `mc_func.h` / `mc_func.c` | Built-in function catalog, metadata, and `--list` introspection |
| `plan/mc_plan.h` | Planner, view (`mc_view_t`), pass (`mc_pass_t`), and program (`mcalc_prog_t`) structs |
| `jit/mc_prelude.h` | Embedded C99 SIMD/math/spatial prelude emitted at the top of generated `.so` kernels |
| `front/mc_lex.c` | Lexical analyzer for calculator expressions and `.calc` programs |
| `front/mc_parse.c` | Pratt expression parser and statement/control-flow (`if`/`for`/`def`) parser |
| `front/mc_resolve.c` | Identifier binding, inline `def` expansion, and static shape/type inference |
| `run/mc_scalar_eval.c` | Scalar AST evaluator (`int64_t` + `double` + string lanes) |
| `run/mc_scalar_ops.c` | Zero-RPC client-side evaluator (`mcalc_local`) and temporal stats (`RunningStat`, `P2`) |
| `plan/mc_views.c` | N-dimensional slice, stride, broadcast, and `::Nb` binned view resolution |
| `run/mc_out.c` | Output stream creation/preparation, `_tmpcalc*` scratch pool, and deferred posting |
| `plan/mc_plan.c` | AST-to-pass planner, basic-block fusion, and local scratch elision |
| `jit/mc_emit.c` | C99 OpenMP/SIMD kernel code generator |
| `jit/mc_cc.c` | Run-time compiler driver (JIT kernels and `.c` extensions, `MILK_CC_FLAGS`), on-disk (`<key>.so`) and in-memory kernel cache |
| `run/mc_exec.c` | Elementwise pass and constructor step execution with torn-read retry protection |
| `run/mc_reduce.c` | Single-pass full-image, masked, pair, and tuple (`istats`, `icentroid`, `iarg*`) reductions |
| `run/mc_prestage.c` | Axis reductions, `cumsum`, `pixunmap`, `rprof`, `gauss`, `medfilt*`, and `ema` pre-stages |
| `mc_eval.c` | Top-level `mcalc_eval_bash`, `mcalc_eval_str`, `mcalc_eval_math_to_val`, and `def` table |
| `run/mc_prog.c` | Multi-statement program compiler/runner (`mcalc_prog_*`) |
| `plan/mc_history.c` | `hist(s, N)` and `prev(s[, k])` ring-buffer stream management and source rewriter |
| `plan/mc_explain.c` | `--explain`: how each statement of a program is scheduled (`mcalc_prog_explain`) |
| `ext/mc_ext_registry.c` | Global registry for loaded Tier 1 and Tier 2 calculator C extensions |
| `ext/mc_ext_loader.c` | `.c` / `.so` extension loader (compiles through `mc_cc.c`), `dlopen`, the call-argument binder (`mc_ext_bind_args`, shared with `mc_ext_exec.c`) and FPS parameter scanner |
| `ext/mc_ext_exec.c` | Runtime execution of Tier 1 (block, stencil, whole-image) and Tier 2 (`MILK_EXT_OP`) extensions |
| `ext/mc_ext_builtin.c` | Built-in image extensions (`bblur1pix`, `rotate2d`, `circsym`, `circasym`, COREMOD_arith kernels) and stream-resolution helpers |
| `linalg/mc_linalg.h` | Internal linear algebra, FFT, stacking, and spatial convolution interface |
| `linalg/mc_linalg_blas.c` | Tensor constructors, BLAS Level 1-3 (`@`, `matmul`, `gemm`, `gemv`, `syrk`, ...), and `einsum` |
| `linalg/mc_linalg_lapack.c` | LAPACK linear solvers (`trsv`, `trsm`, `gesv`, `posv`, `gels`, `solve`, `inv`, `pinv`, `svd`, `eigh`) |
| `linalg/mc_fft_stack.c` | FFTW (`fft`, `ifft`, `fftshift`), stacking (`hstack`, `vstack`, `dstack`, `cat`, `rot90`), `conv2`, `boxmean` |
| `units/milk_unit_calc.c` | Standalone `milk-fpsexec-milk-calc` (`milk.calc`) compute unit |

---

## Standalone Executables

| Executable | Source | Description |
| :--- | :--- | :--- |
| `milk-fpsexec-milk-calc` | `units/milk_unit_calc.c` | Real-time calculator compute unit (`milk.calc`) |

---

## Dependencies & Build Tier

- **Build Tier**: Calculator & JIT tier (`USE_CALC=ON`, which needs `USE_COREMODS`; not `USE_CLI`).
- **Internal Dependencies**: `milkprop`, `milkCOREMODarith`, `milkCOREMODmemory`, `milkcomputeunit`,
  `milkfps`, `milkprocessinfo`, `milkdata`, `ImageStreamIO`, `milkcommon`.
- **Optional External Dependencies**: OpenMP, OpenBLAS / MKL (`HAVE_BLAS`), LAPACKE
  (`HAVE_LAPACKE`), FFTW3 (`HAVE_FFTW3`).

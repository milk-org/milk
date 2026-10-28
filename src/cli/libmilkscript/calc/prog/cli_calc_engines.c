// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_engines.c
 * @brief Catalog and listing of pre-compiled calculator bytecode engines, tokens, and C extensions.
 *
 * Role & Goal:
 *   Provides a comprehensive, human- and machine-readable (`--json`) listing of all
 *   tokens, functions, operators, pre-stage spatial kernels, reductions, kept BLAS/LAPACK
 *   engines, and user-loaded C extensions (`-l` / `loadlib`) that `calc` compiles into
 *   its execution tape. Shared by `@calc --list` (`libmilkbash` / `libmilkscript`) and
 *   `milk-fpsexec-milk-calc --list`.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli_calc_engines.h"

typedef struct
{
    const char *category;
    const char *syntax;
    const char *token;
    const char *engine;
    const char *stage;
    const char *descr;
} calc_engine_entry_t;

static const calc_engine_entry_t g_calc_engines[] = {
    /* 1. Element-Wise Operators (fprog SIMD Bytecode Tape) */
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "+  -  *  /  %  ^", "TOK_OP_*",
      "FI_BIN", "fprog_simd", "Binary arithmetic (^ specializes x^2, sqrt, 1/x, pow)" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "-x", "TOK_OP_MINUS", "FI_NEG",
      "fprog_simd", "Unary arithmetic negation" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "<  <=  >  >=  ==  !=", "TOK_OP_*",
      "FI_BIN", "fprog_simd", "Relational comparisons (1.0 if true, 0.0 if false)" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "&&  ||", "TOK_OP_AND/OR", "FI_BIN",
      "fprog_simd", "Logical AND / OR (short-circuits on scalar operands)" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "!x", "TOK_OP_NOT", "FI_NOT",
      "fprog_simd", "Logical NOT (1.0 if x == 0, else 0.0)" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "&  |  ^^  <<  >>", "TOK_OP_B*",
      "FI_BIN", "fprog_simd", "64-bit integer bitwise AND, OR, XOR (^^), and shifts" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "~x", "TOK_OP_BNOT", "FI_BNOT",
      "fprog_simd", "64-bit integer bitwise complement (~(int64_t)x)" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "c ? a : b, where(c,a,b)",
      "TOK_FUNC_WHERE", "FI_SEL", "fprog_simd",
      "Per-pixel branchless conditional select (c != 0 ? a : b)" },
    { "1. Element-Wise Operators (fprog SIMD Bytecode Tape)", "=  +=  -=  *=  /=", "TOK_EQUAL",
      "FI_STORE_DST", "fprog_simd",
      "Fused destination store (full image, N-D slice, or a, b = ...)" },

    /* 2. Scalar & Element-Wise Math Functions (scalar_fns[] -> FI_FN1/2/3) */
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "sin(x)  cos(x)  tan(x)",
      "TOK_FUNC_D_D", "FI_FN1", "fprog_simd", "Trigonometric functions (vectorized via libmvec)" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "asin(x) acos(x) atan(x)",
      "TOK_FUNC_D_D", "FI_FN1", "fprog_simd", "Inverse trigonometric functions" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "sinh(x) cosh(x) tanh(x)",
      "TOK_FUNC_D_D", "FI_FN1", "fprog_simd", "Hyperbolic functions" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "exp(x)  exp2(x)", "TOK_FUNC_D_D",
      "FI_FN1", "fprog_simd", "Base-e and base-2 exponential functions" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "ln(x)   log(x)  log2(x)",
      "TOK_FUNC_D_D", "FI_FN1", "fprog_simd",
      "Natural (ln), base-10 (log), and base-2 (log2) logarithms" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "sqrt(x) cbrt(x)", "TOK_FUNC_D_D",
      "FI_FN1", "fprog_simd", "Hardware SIMD square root and cube root" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "erf(x)  erfc(x)", "TOK_FUNC_D_D",
      "FI_FN1", "fprog_simd", "Error function and complementary error function" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "abs(x)  fabs(x)", "TOK_FUNC_D_D",
      "FI_FN1 / FI_CPART", "fprog_simd",
      "Absolute value (or complex magnitude |z| on complex leaf)" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "ceil(x) floor(x) round(x)",
      "TOK_FUNC_D_D", "FI_FN1", "fprog_simd", "Nearest-integer rounding operations" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "posi(x) sign(x)", "TOK_FUNC_D_D",
      "FI_FN1", "fprog_simd", "Heaviside step (x > 0 ? 1 : 0) and signum (-1, 0, +1)" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "isnan(x) isfinite(x)",
      "TOK_FUNC_D_D", "FI_FN1", "fprog_simd",
      "IEEE-754 bit-level NaN and finite checks (-ffast-math safe)" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "atan2(y, x)", "TOK_FUNC_DD_D",
      "FI_FN2", "fprog_simd", "Four-quadrant inverse tangent (image/scalar or image/image)" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "hypot(x, y)  fmod(x, y)",
      "TOK_FUNC_DD_D", "FI_FN2", "fprog_simd",
      "Euclidean norm sqrt(x^2 + y^2) and floating-point modulo" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "min(a, b)    max(a, b)",
      "TOK_FUNC_DD_D", "FI_FN2", "fprog_simd", "Hardware SIMD per-pixel minimum and maximum" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "clamp(x,lo,hi) trunc(...)",
      "TOK_FUNC_DDD_D", "FI_FN3", "fprog_simd", "Per-pixel branchless clamp to [lo, hi]" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "f32(x)  f64(x)", "TOK_FUNC_D_D",
      "FI_FN1 (cast)", "fprog_simd",
      "Explicit cast to float32 / float64 (sets output image dtype)" },
    { "2. Scalar & Element-Wise Math (fprog SIMD Bytecode Tape)", "u8 i8 u16 i16 u32 i32 u64 i64",
      "TOK_FUNC_D_D", "FI_FN1 (cast)", "fprog_simd",
      "Explicit cast to 8/16/32/64-bit signed or unsigned integer" },

    /* 3. Coordinates, PRNG & Complex Operations */
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "ix(im[, cx, cy])",
      "TOK_FUNC_COORD", "FI_COORD", "fprog_simd",
      "Per-pixel X coordinate relative to cx (default xsize/2)" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "iy(im[, cx, cy])",
      "TOK_FUNC_COORD", "FI_COORD", "fprog_simd",
      "Per-pixel Y coordinate relative to cy (default ysize/2)" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "ir(im[, cx, cy])",
      "TOK_FUNC_COORD", "FI_COORD", "fprog_simd", "Per-pixel polar radius hypot(x - cx, y - cy)" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "itheta(im[, cx, cy])",
      "TOK_FUNC_COORD", "FI_COORD", "fprog_simd",
      "Per-pixel polar azimuth angle atan2(y - cy, x - cx)" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "rand(im[, s]) urand(...)",
      "TOK_FUNC_COORD", "FI_COORD", "fprog_simd",
      "Vectorized xorshift128+ uniform PRNG in [0, 1)" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "randn(im[, s]) grand(...)",
      "TOK_FUNC_COORD", "FI_COORD", "fprog_simd",
      "Vectorized Box-Muller standard normal PRNG N(0, 1)" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "real(z) imag(z) arg(z)",
      "TOK_FUNC_CPLX", "FI_CPART", "fprog_simd",
      "Fused extraction of real, imaginary, or phase from complex z" },
    { "3. Coordinates, PRNG & Complex (fprog SIMD Bytecode Tape)", "conj(z) cplx(r,i) expi(p)",
      "TOK_FUNC_CPLX", "CPLX_KERNEL", "eager_cplx",
      "Complex conjugate, real-to-complex pack, and unit phasor e^(i*p)" },

    /* 4. Stencils, Lookup Tables, Gather & Interpolation */
    { "4. Stencils, Gather & Interpolation (fprog Spatial Tape)", "shift(im, dx[, dy])",
      "TOK_FUNC_SHIFT", "FI_SHIFT", "fprog_simd",
      "2D integer pixel offset with edge-clamped boundary load" },
    { "4. Stencils, Gather & Interpolation (fprog Spatial Tape)", "medfilt3/5/7(im)",
      "TOK_FUNC_MEDFILT", "FI_MEDFILT", "fprog_simd",
      "Branchless sorting-network 2D median filter (3x3, 5x5, 7x7)" },
    { "4. Stencils, Gather & Interpolation (fprog Spatial Tape)", "lut(expr, tbl[, x0, dx])",
      "TOK_FUNC_LUT", "FI_LUT", "fprog_simd",
      "1D linear-interpolated lookup into table tbl at (x - x0)/dx" },
    { "4. Stencils, Gather & Interpolation (fprog Spatial Tape)", "pixremap(im, map[, fill])",
      "TOK_FUNC_PIXREMAP", "FI_GATHER", "fprog_simd",
      "1D index-map gather im[map[i]] with out-of-bounds fill value" },
    { "4. Stencils, Gather & Interpolation (fprog Spatial Tape)", "interp2(im, xmap, ymap)",
      "TOK_FUNC_INTERP2", "FI_INTERP2", "fprog_simd",
      "2D bilinear sub-pixel gather of im at per-pixel (xmap, ymap)" },

    /* 5. Spatial & Projection Pre-Stage Engines (AX_KIND_* -> FI_IMG) */
    { "5. Spatial & Projection Pre-Stages (AX_KIND_* -> FI_IMG)", "gauss(expr, sx[, sy])",
      "TOK_FUNC_GAUSS", "AX_KIND_GAUSS", "prestage",
      "Separable 1D/2D Gaussian blur pre-stage feeding the SIMD tape" },
    { "5. Spatial & Projection Pre-Stages (AX_KIND_* -> FI_IMG)", "cumsum(expr[, axis])",
      "TOK_FUNC_CUMSUM", "AX_KIND_CUMSUM", "prestage",
      "Cumulative sum along an axis (0 when omitted), the expression's shape" },
    { "5. Spatial & Projection Pre-Stages (AX_KIND_* -> FI_IMG)", "pixunmap(expr, map[, n])",
      "TOK_FUNC_PIXUNMAP", "AX_KIND_SEG", "prestage",
      "Cached CSR segment-mean projection onto n regions of map" },
    { "5. Spatial & Projection Pre-Stages (AX_KIND_* -> FI_IMG)", "rprof(expr[, cx, cy])",
      "TOK_FUNC_PIXUNMAP", "AX_KIND_RPROF", "prestage",
      "Annular radial profile mean centered at (cx, cy)" },
    { "5. Spatial & Projection Pre-Stages (AX_KIND_* -> FI_IMG)", "<red>(expr, axis)",
      "TOK_FUNC_IM_D/L", "AX_KIND_AXIS", "prestage",
      "Single-axis projection (axis 0..2) for itot/imean/istd/imin/imax..." },

    /* 6. Reductions & Multi-Output Unpacking (FI_RED, FI_CENT, fsins) */
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "itot(im[, mask])",
      "TOK_FUNC_IM_D", "FI_RED (SUM)", "fused_reduce",
      "Fused single-pass sum (optional mask or axis; CSE deduplicated)" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "imean(im[, mask])",
      "TOK_FUNC_IM_D", "FI_RED (MEAN)", "fused_reduce", "Fused single-pass arithmetic mean" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "istd(im[, m]) irms ivar",
      "TOK_FUNC_IM_D", "FI_RED (RMS/VAR)", "fused_reduce",
      "Fused single-pass standard deviation (istd/irms) and variance" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "imin(im[, m]) imax(im[, m])",
      "TOK_FUNC_IM_D", "FI_RED (MIN/MAX)", "fused_reduce",
      "Fused single-pass minimum and maximum value" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "norm(im[, m]) nrm2 asum",
      "TOK_FUNC_IM_D", "FI_RED (NRM2/L1)", "fused_reduce",
      "Euclidean L2 norm sqrt(sum(x^2)) and L1 absolute sum sum(|x|)" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "icount(im[, mask])",
      "TOK_FUNC_IM_L", "FI_RED (COUNT)", "fused_reduce",
      "Count of non-zero pixels (or non-zero pixels within mask)" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "iargmax iargmin iamax",
      "TOK_FUNC_IM_L", "FI_RED (ARG*)", "fused_reduce",
      "Flat index of max, min, or max(|x|) element" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "dot(im1, im2)",
      "TOK_FUNC_IMIM_D", "FI_RED (DOT)", "fused_reduce",
      "Fused two-stream inner product sum(im1[i] * im2[i])" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "istats(expr)", "TOK_FUNC_IM_D",
      "FI_RED (STATS)", "fused_reduce",
      "Single-pass multi-LHS unpacking: m, s[, min, max] = istats(im)" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "icentroid(expr[, th])",
      "TOK_FUNC_IM_D", "FI_CENT", "fused_reduce",
      "Flux-weighted centroid unpacking: cx, cy[, flux] = icentroid(im)" },
    { "6. Reductions & Multi-Output Unpacking (FI_RED / FI_CENT)", "imedian(im) perc(im,q) trace",
      "TOK_FUNC_IM*_D", "EAGER_RED", "eager_reduce",
      "Exact quickselect median, percentile q in [0,1], matrix trace" },

    /* 7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_* & Solvers) */
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "matmul(A, B) gemm gemv",
      "TOK_FUNC_IMIM_IM", "FKEPT_MATMUL", "fkept_blas",
      "Zero-allocation kept CBLAS GEMM / GEMV (1D, 2D, batched 3D)" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "outer(x, y) ger(a,x,y,A)",
      "TOK_FUNC_IMIM_IM", "FKEPT_GER", "fkept_blas",
      "Zero-allocation kept CBLAS rank-1 outer product update" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "syrk(a, A, b, C[, tr])",
      "TOK_FUNC_IMIM_IM", "FKEPT_SYRK", "fkept_blas",
      "Zero-allocation kept CBLAS symmetric rank-k update" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "transpose(A)", "TOK_FUNC_IMIM_IM",
      "FKEPT_TRANSPOSE", "fkept_blas",
      "Zero-allocation kept cache-blocked 2D/3D matrix transpose" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "reshape(A, n0[, n1, n2])",
      "TOK_FUNC_IMIM_IM", "FKEPT_RESHAPE", "fkept_blas",
      "Zero-allocation kept dimension reshape preserving nelement" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "symv axpy scal trsv trsm",
      "TOK_FUNC_IMIM_IM", "CBLAS_DIRECT", "eager_blas",
      "Symmetric MV, y += a*x, x *= a, and triangular solvers" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "gesv posv gels solve inv pinv",
      "TOK_FUNC_IMIM_IM", "LAPACK_SOLVE", "eager_lapack",
      "LU, Cholesky, QR, Tikhonov ridge (solve), inverse, SVD pinv" },
    { "7. Kept BLAS / Matrix & LAPACK Engines (FKEPT_*)", "eye diag vec zeros", "TOK_FUNC_IMIM_IM",
      "TENSOR_INIT", "eager_init",
      "Identity matrix, diagonal extract/build, flatten, zero tensor" },

    /* 8. Inline Functions, Directives & User C Extension SDK (-l / loadlib) */
    { "8. Inline Functions, Directives & User C Extensions", "def f(a, b) = expr", "TOK_FUNC_UDEF",
      "FI_STORE/LOAD_SLOT", "fprog_simd",
      "Inline user macro function expanded into tape with CSE slots" },
    { "8. Inline Functions, Directives & User C Extensions", "strlen toupper tolower ...",
      "TOK_FUNC_*_S", "SCALAR_STR", "scalar",
      "String ops (strlen, toupper, tolower, substr, replace, hex/oct/bin)" },
    { "8. Inline Functions, Directives & User C Extensions", "if/elif/else for local skip",
      "DIRECTIVE", "PROG_CTRL", "prog_ctrl",
      "Program branching, bounded loops, local images, frame skip" },
    { "8. Inline Functions, Directives & User C Extensions", "post(s) hist(s,N) prev(s,k)",
      "DIRECTIVE", "PROG_STREAM", "prog_ctrl",
      "Explicit semaphore post and N-frame temporal ring buffer" },
    { "8. Inline Functions, Directives & User C Extensions", "MILK_CALC_REG_BLK / BLK2",
      "TOK_FUNC_USER", "FI_USER_BLK", "fprog_simd",
      "Tier 1a user C 1- or 2-arg SIMD block kernel (-l / loadlib)" },
    { "8. Inline Functions, Directives & User C Extensions", "MILK_CALC_REG_STENCIL",
      "TOK_FUNC_USER", "FI_USER_STENCIL", "fprog_simd",
      "Tier 1b user C 2D neighborhood stencil kernel (-l / loadlib)" },
    { "8. Inline Functions, Directives & User C Extensions", "MILK_CALC_REG_IMG", "TOK_FUNC_USER",
      "AX_KIND_USER", "prestage", "Tier 1c user C whole-image pre-stage kernel (-l / loadlib)" },
    { "8. Inline Functions, Directives & User C Extensions", "MILK_CALC_REG_OP", "TOK_FUNC_USER",
      "FKEPT_USER_OP", "fkept_uop",
      "Tier 2 user C multi-input/output stateful operator + FPS params" },
    { NULL, NULL, NULL, NULL, NULL, NULL }
};

/**
 * str_contains_ci - Case-insensitive substring search
 * @haystack: String to search in
 * @needle:   Substring to look for
 *
 * Return: 1 if @needle is empty or found in @haystack, 0 otherwise.
 */
static int str_contains_ci(const char *haystack, const char *needle)
{
    if (needle == NULL || needle[0] == '\0')
    {
        return 1;
    }
    if (haystack == NULL)
    {
        return 0;
    }
    size_t nlen = strlen(needle);
    size_t hlen = strlen(haystack);
    if (nlen > hlen)
    {
        return 0;
    }
    for (size_t i = 0; i + nlen <= hlen; i++)
    {
        size_t j = 0;
        while (j < nlen &&
               tolower((unsigned char) haystack[i + j]) == tolower((unsigned char) needle[j]))
        {
            j++;
        }
        if (j == nlen)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * entry_matches - Check if a built-in engine entry matches filter @pat
 * @e:   Engine table entry
 * @pat: Filter substring
 *
 * Return: 1 if @pat matches any field of @e, 0 otherwise.
 */
static int entry_matches(const calc_engine_entry_t *e, const char *pat)
{
    if (pat == NULL || pat[0] == '\0')
    {
        return 1;
    }
    return str_contains_ci(e->category, pat) || str_contains_ci(e->syntax, pat) ||
           str_contains_ci(e->token, pat) || str_contains_ci(e->engine, pat) ||
           str_contains_ci(e->stage, pat) || str_contains_ci(e->descr, pat);
}

/**
 * ext_kind_opcode - Return the bytecode engine/opcode name for a user C extension kind
 * @kind: `milk_calc_ext_kind_t` enum value
 *
 * Return: Static opcode name string.
 */
static const char *ext_kind_opcode(milk_calc_ext_kind_t kind)
{
    switch (kind)
    {
    case MILK_EXT_BLK1:
        return "FI_USER_BLK (1)";
    case MILK_EXT_BLK2:
        return "FI_USER_BLK (2)";
    case MILK_EXT_STENCIL:
        return "FI_USER_STENCIL";
    case MILK_EXT_IMG:
        return "AX_KIND_USER";
    case MILK_EXT_OP:
        return "FKEPT_USER_OP";
    default:
        return "TOK_FUNC_USER";
    }
}

/**
 * format_ext_info - Build signature and description strings for a loaded user C extension
 * @e:       Loaded extension entry
 * @sig:     Output buffer for `<name>(...)` signature
 * @sig_sz:  Capacity of @sig
 * @desc:    Output buffer for tier/geometry/parameter description
 * @desc_sz: Capacity of @desc
 */
static void format_ext_info(const milk_calc_ext_entry_t *e,
                            char                        *sig,
                            size_t                       sig_sz,
                            char                        *desc,
                            size_t                       desc_sz)
{
    switch (e->kind)
    {
    case MILK_EXT_BLK1:
        snprintf(sig, sig_sz, "%s(x)", e->name);
        snprintf(desc, desc_sz, "Tier 1 unary SIMD block kernel");
        break;
    case MILK_EXT_BLK2:
        snprintf(sig, sig_sz, "%s(a, b)", e->name);
        snprintf(desc, desc_sz, "Tier 1 binary SIMD block kernel");
        break;
    case MILK_EXT_STENCIL:
        snprintf(sig, sig_sz, "%s(im)", e->name);
        snprintf(desc, desc_sz, "Tier 2 2D stencil kernel (halo r=%d, cost=%d)", e->radius,
                 e->cost);
        break;
    case MILK_EXT_IMG:
        snprintf(sig, sig_sz, "%s(im)", e->name);
        snprintf(desc, desc_sz, "Tier 3 whole-image pre-stage kernel (cost=%d)", e->cost);
        break;
    case MILK_EXT_OP:
    {
        const milk_calc_op_ports_t *ports = (e->get_ports != NULL) ? e->get_ports() : NULL;
        int                         nin   = (ports != NULL) ? ports->nin : 0;
        int                         nout  = (ports != NULL) ? ports->nout : 0;
        snprintf(sig, sig_sz, "%s(%d in -> %d out)", e->name, nin, nout);
        snprintf(desc, desc_sz, "Tier 4 multi-port operator (%d in, %d out)", nin, nout);
        break;
    }
    default:
        snprintf(sig, sig_sz, "%s(...)", e->name);
        snprintf(desc, desc_sz, "User C extension");
        break;
    }

    const milk_calc_param_set_t *ps = (e->get_params != NULL) ? e->get_params() : NULL;
    if (ps != NULL && ps->nparams > 0)
    {
        size_t cur = strlen(desc);
        snprintf(desc + cur, desc_sz - cur, " [params:");
        for (int p = 0; p < ps->nparams; p++)
        {
            cur = strlen(desc);
            if (cur + strlen(ps->specs[p].name) + 4 >= desc_sz)
            {
                break;
            }
            snprintf(desc + cur, desc_sz - cur, "%s%s", (p > 0) ? "," : " ", ps->specs[p].name);
        }
        cur = strlen(desc);
        if (cur + 2 < desc_sz)
        {
            snprintf(desc + cur, desc_sz - cur, "]");
        }
    }
}

/**
 * ext_matches - Check if a loaded user extension matches filter @pat
 * @e:   Loaded user C extension
 * @pat: Filter substring
 *
 * Return: 1 if @pat matches the extension, 0 otherwise.
 */
static int ext_matches(const milk_calc_ext_entry_t *e, const char *pat)
{
    if (pat == NULL || pat[0] == '\0')
    {
        return 1;
    }
    char sig[64];
    char desc[256];
    format_ext_info(e, sig, sizeof(sig), desc, sizeof(desc));
    return str_contains_ci(e->name, pat) || str_contains_ci(sig, pat) ||
           str_contains_ci(ext_kind_opcode(e->kind), pat) || str_contains_ci(desc, pat) ||
           str_contains_ci("user", pat) || str_contains_ci("TOK_FUNC_USER", pat);
}

/**
 * json_escape - Write a JSON-escaped string to stdout
 * @s: Input string
 */
static void json_escape(const char *s)
{
    putchar('"');
    for (const char *p = (s != NULL) ? s : ""; *p != '\0'; p++)
    {
        if (*p == '"' || *p == '\\')
        {
            putchar('\\');
            putchar(*p);
        }
        else if (*p == '\n')
        {
            fputs("\\n", stdout);
        }
        else
        {
            putchar(*p);
        }
    }
    putchar('"');
}

/**
 * print_engines_json - Emit the filtered engine and extension catalog as JSON
 */
static int print_engines_json(const char *pattern, cli_calc_ext_getter_fn ext_get, int ext_count)
{
    int matched = 0;
    printf("{\n  \"engines\": [\n");
    for (int i = 0; g_calc_engines[i].category != NULL; i++)
    {
        const calc_engine_entry_t *e = &g_calc_engines[i];
        if (!entry_matches(e, pattern))
        {
            continue;
        }
        if (matched > 0)
        {
            printf(",\n");
        }
        printf("    {\"category\": ");
        json_escape(e->category);
        printf(", \"syntax\": ");
        json_escape(e->syntax);
        printf(", \"token\": ");
        json_escape(e->token);
        printf(", \"engine\": ");
        json_escape(e->engine);
        printf(", \"stage\": ");
        json_escape(e->stage);
        printf(", \"description\": ");
        json_escape(e->descr);
        printf("}");
        matched++;
    }
    printf("\n  ],\n  \"user_extensions\": [\n");
    int umatched = 0;
    for (int i = 0; ext_get != NULL && i < ext_count; i++)
    {
        const milk_calc_ext_entry_t *e = ext_get(i);
        if (e == NULL || !ext_matches(e, pattern))
        {
            continue;
        }
        char sig[64];
        char desc[256];
        format_ext_info(e, sig, sizeof(sig), desc, sizeof(desc));
        if (umatched > 0)
        {
            printf(",\n");
        }
        printf("    {\"name\": ");
        json_escape(e->name);
        printf(", \"signature\": ");
        json_escape(sig);
        printf(", \"token\": \"TOK_FUNC_USER\", \"engine\": ");
        json_escape(ext_kind_opcode(e->kind));
        printf(", \"description\": ");
        json_escape(desc);
        printf("}");
        umatched++;
        matched++;
    }
    printf("\n  ]\n}\n");
    fflush(stdout);
    return (matched > 0 || pattern == NULL || pattern[0] == '\0') ? 0 : 1;
}

/**
 * cli_calc_print_engines - Print the catalog of pre-compiled calculator bytecode engines and tokens
 * @pattern:   Optional case-insensitive substring filter (NULL or "" for all)
 * @json_mode: Non-zero to emit JSON output instead of a formatted table
 * @ext_get:   Optional callback returning loaded user C extension entry `i`
 * @ext_count: Number of currently loaded user C extensions (`0` if none)
 *
 * Return: 0 on success, 1 if @pattern matched no entries.
 */
int cli_calc_print_engines(const char            *pattern,
                           int                    json_mode,
                           cli_calc_ext_getter_fn ext_get,
                           int                    ext_count)
{
    if (json_mode)
    {
        return print_engines_json(pattern, ext_get, ext_count);
    }

    int         tty     = isatty(STDOUT_FILENO) && (getenv("NO_COLOR") == NULL);
    const char *c_title = tty ? "\033[1;36m" : "";
    const char *c_hdr   = tty ? "\033[1;35m" : "";
    const char *c_tok   = tty ? "\033[32m" : "";
    const char *c_eng   = tty ? "\033[33m" : "";
    const char *c_dim   = tty ? "\033[0;90m" : "";
    const char *c_rst   = tty ? "\033[0m" : "";

    printf("%s============================================================================="
           "=====================%s\n",
           c_title, c_rst);
    printf("%s                     MILK.CALC PRE-COMPILED BYTECODE ENGINES & TOKENS%s\n", c_title,
           c_rst);
    printf("%s============================================================================="
           "=====================%s\n",
           c_title, c_rst);

    const char *last_cat = NULL;
    int         matched  = 0;

    for (int i = 0; g_calc_engines[i].category != NULL; i++)
    {
        const calc_engine_entry_t *e = &g_calc_engines[i];
        if (!entry_matches(e, pattern))
        {
            continue;
        }
        if (last_cat == NULL || strcmp(last_cat, e->category) != 0)
        {
            printf("\n%s%s%s\n", c_hdr, e->category, c_rst);
            printf("  %s%-29s %-18s %s%s\n", c_dim, "TOKEN / SYNTAX", "BYTECODE ENGINE",
                   "DESCRIPTION", c_rst);
            last_cat = e->category;
        }
        printf("  %s%-29s%s %s%-18s%s %s\n", c_tok, e->syntax, c_rst, c_eng, e->engine, c_rst,
               e->descr);
        matched++;
    }

    int umatched = 0;
    for (int i = 0; ext_get != NULL && i < ext_count; i++)
    {
        const milk_calc_ext_entry_t *e = ext_get(i);
        if (e == NULL || !ext_matches(e, pattern))
        {
            continue;
        }
        if (umatched == 0)
        {
            printf("\n%s9. Loaded User C Extensions (-l / loadlib)%s\n", c_hdr, c_rst);
            printf("  %s%-29s %-18s %s%s\n", c_dim, "TOKEN / SYNTAX", "BYTECODE ENGINE",
                   "DESCRIPTION", c_rst);
        }
        char sig[64];
        char desc[256];
        format_ext_info(e, sig, sizeof(sig), desc, sizeof(desc));
        printf("  %s%-29s%s %s%-18s%s %s\n", c_tok, sig, c_rst, c_eng, ext_kind_opcode(e->kind),
               c_rst, desc);
        umatched++;
        matched++;
    }

    if (matched == 0)
    {
        printf("\n  No calculator bytecode engines or tokens matched '%s'.\n\n",
               (pattern != NULL) ? pattern : "");
        fflush(stdout);
        return 1;
    }

    printf("\n%sTotal: %d engine group(s) shown", c_dim, matched);
    if (ext_count > 0)
    {
        printf(" (%d loaded user C extension(s))", umatched);
    }
    printf(". Filter: @calc --list <pattern> [--json] | C plugins: @calc -l <lib.c> --list%s\n\n",
           c_rst);
    fflush(stdout);
    return 0;
}

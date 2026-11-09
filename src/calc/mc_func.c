// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_func.c
 * @brief Built-in function registry, scalar math helpers, and `--list` catalog for libmilkcalc.
 *
 * This module defines the master descriptor table (`g_mc_funcs`) for every
 * built-in calculator function and directive across all execution tiers:
 * - Scalar and elementwise SIMD math (`sin`, `cos`, `pow`, `clamp`, explicit
 *   datatype casts `f32`..`i64`, `where`).
 * - Coordinate generators (`ix`, `iy`, `ir`, `itheta`), PRNGs (`rand`, `randn`),
 *   complex number operations (`real`, `imag`, `arg`, `conj`, `cplx`, `expi`),
 *   and `linspace`.
 * - Spatial shifts (`shift`, `roll`), 1-D lookup tables (`lut`), and gather
 *   interpolation (`pixremap`, `interp2`).
 * - Fused and eager reductions (`itot`, `imean`, `istd`, `imin`, `imax`,
 *   `norm`, `asum`, `icount`, `iargmax`, `iargmin`, `iamax`, `dot`, `istats`,
 *   `icentroid`, `imedian`, `perc`, `trace`).
 * - Neighborhood stencils and spatial pre-stages (`medfilt3/5/7`, `gauss`,
 *   `conv2`, `imin3`, `imax3`, `boxmean`, `cumsum`, `pixunmap`, `rprof`).
 * - CBLAS, LAPACK, FFTW, and tensor constructors (`matmul`, `gemm`, `svd`,
 *   `eigh`, `einsum`, `fft`, `ifft`, `fftshift`, `transpose`, `reshape`, etc.).
 * - String manipulation (`strlen`, `toupper`, `tolower`, `substr`, `replace`,
 *   `hex`, `oct`, `bin`) and temporal unit directives (`post`, `hist`, `prev`,
 *   `ema`).
 *
 * It also provides IEEE-754 bit-level scalar helpers (`mc_isnan`,
 * `mc_isfinite`) that remain valid under `-ffast-math`, and implements
 * `mcalc_print_catalog()` for `@calc --list` / `milk-fpsexec-milk-calc --list`.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStruct.h"
#include "front/mc_ast.h"
#include "mc_func.h"
#include "mcalc.h"
#include "milk_calc_ext.h"

/* ========================================================================= */
/* Scalar Math and Datatype Cast Helpers                                     */
/* ========================================================================= */

/**
 * mc_isnan - Bit-level IEEE-754 NaN test safe under `-ffast-math`
 * @a: Double-precision value to inspect
 *
 * Inspects the raw 64-bit IEEE-754 representation so the check is not folded
 * away when compiled with `-ffinite-math-only` / `-ffast-math`.
 *
 * Return: 1.0 if @a is NaN, 0.0 otherwise
 */
double mc_isnan(double a)
{
    uint64_t u;
    memcpy(&u, &a, sizeof(u));
    return ((u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL &&
            (u & 0x000fffffffffffffULL) != 0)
               ? 1.0
               : 0.0;
}

/**
 * mc_isfinite - Bit-level IEEE-754 finite test safe under `-ffast-math`
 * @a: Double-precision value to inspect
 *
 * Return: 1.0 if @a is finite (neither NaN nor +/-Inf), 0.0 otherwise
 */
double mc_isfinite(double a)
{
    uint64_t u;
    memcpy(&u, &a, sizeof(u));
    return ((u & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL) ? 1.0 : 0.0;
}

/**
 * mc_posi - Heaviside unit step function
 * @a: Input value
 *
 * Return: 1.0 if `@a > 0.0`, 0.0 otherwise
 */
double mc_posi(double a)
{
    return (a > 0.0) ? 1.0 : 0.0;
}

/**
 * mc_sign - Signum function
 * @a: Input value
 *
 * Return: 1.0 if `@a > 0.0`, -1.0 if `@a < 0.0`, or 0.0 if `@a == 0.0` (or NaN)
 */
double mc_sign(double a)
{
    return (double) ((a > 0.0) - (a < 0.0));
}

/**
 * mc_fmodd - Remainder of @x / @y with the quotient truncated toward zero, as fmod()
 * @x: Dividend
 * @y: Divisor
 *
 * The formula of the generated kernels' `mc_fmodd()` (mc_prelude.h), so `%` on numbers and on
 * images agree, and it vectorizes there: q = trunc(x / y) may be off by one, and the remainder is
 * then recomputed with the neighbouring quotient by an FMA, which is exact.
 *
 * Return: fmod(@x, @y) while |@x / @y| < 2^52, NaN beyond
 */
double mc_fmodd(double x, double y)
{
    double q = trunc(x / y), s = copysign(1.0, x / y), r = fma(-q, y, x);
    r = (r != 0.0 && signbit(r) != signbit(x)) ? fma(-(q - s), y, x) : r;
    r = (fabs(r) >= fabs(y)) ? fma(-(q + s), y, x) : r;
    r = (r == 0.0) ? copysign(0.0, x) : r;
    return (fabs(x / y) < 0x1p52 && fabs(r) < fabs(y)) ? r : NAN;
}

/**
 * mc_wrap_pi - Wrap an angle in radians into the principal interval `(-pi, pi]`
 * @a: Input angle in radians
 *
 * The formula of the generated kernels' `mc_wrap_pid()` (mc_prelude.h), as mc_fmodd() with the
 * quotient rounded to nearest: remainder(@a, 2 pi), with `-pi` mapped to `pi`.
 *
 * Return: Wrapped angle in `(-pi, pi]` while |@a| < 2^52 * 2 pi, NaN beyond
 */
double mc_wrap_pi(double a)
{
    const double tp = 2.0 * M_PI;
    double       q = rint(a / tp), r = fma(-q, tp, a);
    r = (r > M_PI) ? fma(-(q + 1.0), tp, a) : ((r < -M_PI) ? fma(-(q - 1.0), tp, a) : r);
    r = (r == -M_PI) ? M_PI : ((r == 0.0) ? copysign(0.0, a) : r);
    return (fabs(a) < 0x1p52 * tp) ? r : NAN;
}

/**
 * mc_truth - Truth value of a condition (`@test`, `@assert`, `@waitfor`, `if`)
 * @v: Value of the condition
 *
 * A condition is a number: non-zero is true; zero and NaN are false. Shared by the session's
 * calculator and the shell's evaluator (mc_scalar_ops.c), so both read a condition alike.
 *
 * Return: 1 if true, 0 if false, -1 if @v is a string (not a condition)
 */
int mc_truth(const mc_val_t *v)
{
    if (v->type == MC_VAL_STRING)
    {
        return -1;
    }
    return (v->type == MC_VAL_LONG) ? (v->lval != 0) : (v->dval != 0.0 && mc_isnan(v->dval) == 0.0);
}

/**
 * mc_sinc - Normalized cardinal sine `sin(pi * a) / (pi * a)`
 * @a: Input value
 *
 * Return: 1.0 when `@a == 0.0`, otherwise `sin(pi * a) / (pi * a)`
 */
double mc_sinc(double a)
{
    if (a == 0.0)
    {
        return 1.0;
    }
    double px = M_PI * a;
    return sin(px) / px;
}

/**
 * mc_clamp - Clamp a value to the closed interval `[@lo, @hi]`
 * @a:  Input value
 * @lo: Lower bound
 * @hi: Upper bound
 *
 * Return: @lo if `@a < @lo`, @hi if `@a > @hi`, otherwise @a
 */
double mc_clamp(double a, double lo, double hi)
{
    return (a > hi) ? hi : (a < lo) ? lo : a;
}

/* X-macro generating scalar C callbacks for explicit datatype cast functions */
#define MC_CAST_TYPES(X)                                                       \
    X(f32, _DATATYPE_FLOAT, (double) (float) a)                                \
    X(f64, _DATATYPE_DOUBLE, a)                                                \
    X(u8, _DATATYPE_UINT8, (double) (uint8_t) (int64_t) a)                     \
    X(i8, _DATATYPE_INT8, (double) (int8_t) (int64_t) a)                       \
    X(u16, _DATATYPE_UINT16, (double) (uint16_t) (int64_t) a)                  \
    X(i16, _DATATYPE_INT16, (double) (int16_t) (int64_t) a)                    \
    X(u32, _DATATYPE_UINT32, (double) (uint32_t) (int64_t) a)                  \
    X(i32, _DATATYPE_INT32, (double) (int32_t) (int64_t) a)                    \
    X(u64, _DATATYPE_UINT64, (double) (uint64_t) (a >= 0.0 ? a : (int64_t) a)) \
    X(i64, _DATATYPE_INT64, (double) (int64_t) a)

/** DEF_MC_CAST_FN() - Instantiate a static scalar cast callback `mc_cast_<NAME>()`. */
#define DEF_MC_CAST_FN(NAME, DT, EXPR)     \
    static double mc_cast_##NAME(double a) \
    {                                      \
        return (EXPR);                     \
    }
MC_CAST_TYPES(DEF_MC_CAST_FN)
#undef DEF_MC_CAST_FN

/* ========================================================================= */
/* Master Built-In Function Table and Lookup API                             */
/* ========================================================================= */

/* Master function table: one entry per built-in calculator function */
static const mc_func_t g_mc_funcs[] = {
    /* 1. Unary elementwise math functions */
    { "sin", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, sin, NULL, NULL, "sinf", "sin", "sin(x)",
      "Sine (radians)" },
    { "cos", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, cos, NULL, NULL, "cosf", "cos", "cos(x)",
      "Cosine (radians)" },
    { "tan", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, tan, NULL, NULL, "tanf", "tan", "tan(x)",
      "Tangent (radians)" },
    { "asin", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, asin, NULL, NULL, "asinf", "asin", "asin(x)",
      "Inverse sine" },
    { "acos", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, acos, NULL, NULL, "acosf", "acos", "acos(x)",
      "Inverse cosine" },
    { "atan", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, atan, NULL, NULL, "atanf", "atan", "atan(x)",
      "Inverse tangent" },
    { "sinh", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, sinh, NULL, NULL, "sinhf", "sinh", "sinh(x)",
      "Hyperbolic sine" },
    { "cosh", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, cosh, NULL, NULL, "coshf", "cosh", "cosh(x)",
      "Hyperbolic cosine" },
    { "tanh", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, tanh, NULL, NULL, "tanhf", "tanh", "tanh(x)",
      "Hyperbolic tangent" },
    { "exp", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, exp, NULL, NULL, "expf", "exp", "exp(x)",
      "Base-e exponential" },
    { "exp2", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, exp2, NULL, NULL, "exp2f", "exp2", "exp2(x)",
      "Base-2 exponential" },
    { "log", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, log, NULL, NULL, "logf", "log", "log(x)",
      "Natural logarithm" },
    { "log10", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, log10, NULL, NULL, "log10f", "log10", "log10(x)",
      "Base-10 logarithm" },
    { "log2", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, log2, NULL, NULL, "log2f", "log2", "log2(x)",
      "Base-2 logarithm" },
    { "sqrt", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, sqrt, NULL, NULL, "sqrtf", "sqrt", "sqrt(x)",
      "Square root" },
    { "cbrt", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, cbrt, NULL, NULL, "cbrtf", "cbrt", "cbrt(x)",
      "Cube root" },
    { "erf", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, erf, NULL, NULL, "erff", "erf", "erf(x)",
      "Error function" },
    { "erfc", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, erfc, NULL, NULL, "erfcf", "erfc", "erfc(x)",
      "Complementary error function" },
    { "abs", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, fabs, NULL, NULL, "fabsf", "fabs", "abs(x)",
      "Absolute value (or complex magnitude)" },
    { "ceil", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, ceil, NULL, NULL, "ceilf", "ceil", "ceil(x)",
      "Smallest integer >= x" },
    { "floor", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, floor, NULL, NULL, "floorf", "floor", "floor(x)",
      "Largest integer <= x" },
    { "round", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, round, NULL, NULL, "roundf", "round", "round(x)",
      "Nearest integer" },
    { "trunc", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, trunc, NULL, NULL, "truncf", "trunc", "trunc(x)",
      "Integer part, rounded toward zero" },
    { "posi", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, mc_posi, NULL, NULL, "mc_posif", "mc_posid",
      "posi(x)", "Heaviside step (x > 0 ? 1 : 0)" },
    { "sign", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, mc_sign, NULL, NULL, "mc_signf", "mc_signd",
      "sign(x)", "Signum (-1, 0, +1)" },
    { "isnan", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, mc_isnan, NULL, NULL, "mc_isnanf", "mc_isnand",
      "isnan(x)", "IEEE-754 NaN test" },
    { "isfinite", 1, 1, 1, 1, MC_FN_SCALAR, 0, 1, 1, mc_isfinite, NULL, NULL, "mc_isfinitef",
      "mc_isfinited", "isfinite(x)", "IEEE-754 finite test" },
    { "wrap_pi", 1, 1, 1, 1, MC_FN_SCALAR, 0, 0, 0, mc_wrap_pi, NULL, NULL, "mc_wrap_pif",
      "mc_wrap_pid", "wrap_pi(x)", "Wrap radians into (-pi, pi]" },
    { "sinc", 1, 1, 1, 1, MC_FN_SCALAR, 0, 0, 0, mc_sinc, NULL, NULL, "mc_sincf", "mc_sincd",
      "sinc(x)", "Normalized sinc sin(pi*x)/(pi*x)" },

    /* 2. Explicit datatype casts */
    { "f32", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_FLOAT, 1, 1, mc_cast_f32, NULL, NULL,
      "mc_cast_f32", "mc_cast_f32", "f32(x)", "Explicit cast to float32" },
    { "f64", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_DOUBLE, 1, 1, mc_cast_f64, NULL, NULL,
      "mc_cast_f64", "mc_cast_f64", "f64(x)", "Explicit cast to float64" },
    { "u8", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_UINT8, 1, 1, mc_cast_u8, NULL, NULL, "mc_cast_u8",
      "mc_cast_u8", "u8(x)", "Explicit cast to uint8" },
    { "i8", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_INT8, 1, 1, mc_cast_i8, NULL, NULL, "mc_cast_i8",
      "mc_cast_i8", "i8(x)", "Explicit cast to int8" },
    { "u16", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_UINT16, 1, 1, mc_cast_u16, NULL, NULL,
      "mc_cast_u16", "mc_cast_u16", "u16(x)", "Explicit cast to uint16" },
    { "i16", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_INT16, 1, 1, mc_cast_i16, NULL, NULL,
      "mc_cast_i16", "mc_cast_i16", "i16(x)", "Explicit cast to int16" },
    { "u32", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_UINT32, 1, 1, mc_cast_u32, NULL, NULL,
      "mc_cast_u32", "mc_cast_u32", "u32(x)", "Explicit cast to uint32" },
    { "i32", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_INT32, 1, 1, mc_cast_i32, NULL, NULL,
      "mc_cast_i32", "mc_cast_i32", "i32(x)", "Explicit cast to int32" },
    { "u64", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_UINT64, 1, 1, mc_cast_u64, NULL, NULL,
      "mc_cast_u64", "mc_cast_u64", "u64(x)", "Explicit cast to uint64" },
    { "i64", 1, 1, 1, 1, MC_FN_SCALAR, _DATATYPE_INT64, 1, 1, mc_cast_i64, NULL, NULL,
      "mc_cast_i64", "mc_cast_i64", "i64(x)", "Explicit cast to int64" },

    /* 3. Binary & ternary elementwise math functions */
    { "pow", 2, 2, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, pow, NULL, "powf", "pow", "pow(a, b)",
      "Power a^b (exact int64 when b >= 0 fits in int64)" },
    { "atan2", 2, 2, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, atan2, NULL, "atan2f", "atan2",
      "atan2(y, x)", "Four-quadrant inverse tangent" },
    { "hypot", 2, 2, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, hypot, NULL, "mc_hypotf", "mc_hypotd",
      "hypot(x, y)", "Euclidean norm sqrt(x^2 + y^2)" },
    { "fmod", 2, 2, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, mc_fmodd, NULL, "mc_fmodf", "mc_fmodd",
      "fmod(x, y)", "Floating-point remainder" },
    { "min", 2, 2, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, fmin, NULL, "mc_minf", "mc_mind", "min(a, b)",
      "Minimum of a and b" },
    { "max", 2, 2, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, fmax, NULL, "mc_maxf", "mc_maxd", "max(a, b)",
      "Maximum of a and b" },
    { "clamp", 3, 3, 1, 1, MC_FN_SCALAR, 0, 1, 1, NULL, NULL, mc_clamp, "mc_clampf", "mc_clampd",
      "clamp(x, lo, hi)", "Clamp x to [lo, hi]" },
    { "where", 3, 3, 1, 1, MC_FN_WHERE, 0, 0, 1, NULL, NULL, NULL, "where", "where",
      "where(c, a, b)", "Conditional select (c != 0 ? a : b)" },

    /* 4. Coordinates, PRNG, complex, linspace */
    { "ix", 1, 3, 1, 1, MC_FN_COORD, 0, 0, 1, NULL, NULL, NULL, "mc_ix", "mc_ix",
      "ix(im[, cx, cy])", "Per-pixel X coordinate relative to cx" },
    { "iy", 1, 3, 1, 1, MC_FN_COORD, 0, 0, 1, NULL, NULL, NULL, "mc_iy", "mc_iy",
      "iy(im[, cx, cy])", "Per-pixel Y coordinate relative to cy" },
    { "ir", 1, 3, 1, 1, MC_FN_COORD, 0, 0, 1, NULL, NULL, NULL, "mc_ir", "mc_ir",
      "ir(im[, cx, cy])", "Per-pixel polar radius hypot(x - cx, y - cy)" },
    { "itheta", 1, 3, 1, 1, MC_FN_COORD, 0, 0, 1, NULL, NULL, NULL, "mc_itheta", "mc_itheta",
      "itheta(im[, cx, cy])", "Per-pixel polar angle atan2(y - cy, x - cx)" },
    { "rand", 0, 2, 1, 1, MC_FN_PRNG, 0, 0, 1, NULL, NULL, NULL, "mc_rand", "mc_rand",
      "rand([im, s])", "Uniform random numbers in [0, 1)" },
    { "randn", 0, 2, 1, 1, MC_FN_PRNG, 0, 0, 1, NULL, NULL, NULL, "mc_randn", "mc_randn",
      "randn([im, s])", "Standard normal random numbers N(0, 1)" },
    { "real", 1, 1, 1, 1, MC_FN_CPLX, 0, 0, 1, NULL, NULL, NULL, "crealf", "creal", "real(z)",
      "Real part of complex image" },
    { "imag", 1, 1, 1, 1, MC_FN_CPLX, 0, 0, 1, NULL, NULL, NULL, "cimagf", "cimag", "imag(z)",
      "Imaginary part of complex image" },
    { "arg", 1, 1, 1, 1, MC_FN_CPLX, 0, 0, 1, NULL, NULL, NULL, "cargf", "carg", "arg(z)",
      "Phase angle of complex image" },
    { "conj", 1, 1, 1, 1, MC_FN_CPLX, 0, 0, 1, NULL, NULL, NULL, "conjf", "conj", "conj(z)",
      "Complex conjugate" },
    { "cplx", 2, 2, 1, 1, MC_FN_CPLX, 0, 0, 1, NULL, NULL, NULL, "cplx", "cplx", "cplx(re, im)",
      "Pack real and imaginary parts into complex image" },
    { "expi", 1, 1, 1, 1, MC_FN_CPLX, 0, 0, 1, NULL, NULL, NULL, "expi", "expi", "expi(phi)",
      "Unit phasor cos(phi) + i*sin(phi)" },
    { "linspace", 3, 3, 1, 1, MC_FN_LINSPACE, 0, 0, 0, NULL, NULL, NULL, "linspace", "linspace",
      "linspace(a, b, n)", "1D float32 image of n evenly spaced points from a to b" },

    /* 5. Shifts, LUT, gather, interpolation */
    { "shift", 2, 3, 1, 1, MC_FN_SHIFT, 0, 0, 1, NULL, NULL, NULL, "mc_shift", "mc_shift",
      "shift(im, dx[, dy])", "2D integer shift with edge-clamped boundary" },
    { "roll", 2, 4, 1, 1, MC_FN_SHIFT, 0, 0, 0, NULL, NULL, NULL, "mc_roll", "mc_roll",
      "roll(im, dx[, dy, dz])", "Periodic shift (modular wrap-around)" },
    { "lut", 2, 4, 1, 1, MC_FN_LUT, 0, 0, 1, NULL, NULL, NULL, "mc_lut", "mc_lut",
      "lut(x, tbl[, x0, dx])", "1D linear-interpolated lookup into table tbl" },
    { "pixremap", 2, 3, 1, 1, MC_FN_GATHER, 0, 0, 1, NULL, NULL, NULL, "mc_pixremap", "mc_pixremap",
      "pixremap(im, map[, fill])", "1D index-map gather im[map[i]]" },
    { "interp2", 3, 3, 1, 1, MC_FN_GATHER, 0, 0, 1, NULL, NULL, NULL, "mc_interp2", "mc_interp2",
      "interp2(im, xmap, ymap)", "2D bilinear sub-pixel gather" },

    /* 6. Reductions and multi-output unpacking */
    { "itot", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "itot(im[, mask])",
      "Sum of pixels (optional mask or axis)" },
    { "imean", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "imean(im[, mask])", "Arithmetic mean of pixels" },
    { "istd", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "istd(im[, mask])",
      "Population standard deviation" },
    { "irms", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "irms(im[, mask])",
      "Root mean square sqrt(mean(x^2))" },
    { "ivar", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "ivar(im[, mask])",
      "Population variance" },
    { "imin", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "imin(im[, mask])",
      "Minimum pixel value" },
    { "imax", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "imax(im[, mask])",
      "Maximum pixel value" },
    { "norm", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "norm(im[, mask])",
      "Euclidean L2 norm sqrt(sum(x^2))" },
    { "asum", 1, 2, 1, 1, MC_FN_REDUCE_D, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "asum(im[, mask])",
      "L1 absolute sum sum(|x|)" },
    { "icount", 1, 2, 1, 1, MC_FN_REDUCE_L, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "icount(im)",
      "Count of finite non-zero pixels" },
    { "iargmax", 1, 1, 1, 3, MC_FN_REDUCE_L, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "iargmax(im)",
      "Index (or 2D/3D coordinates) of maximum pixel" },
    { "iargmin", 1, 1, 1, 3, MC_FN_REDUCE_L, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "iargmin(im)",
      "Index (or 2D/3D coordinates) of minimum pixel" },
    { "iamax", 1, 1, 1, 1, MC_FN_REDUCE_L, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "iamax(im)",
      "Flat index of maximum |x| pixel" },
    { "dot", 2, 2, 1, 1, MC_FN_DOT, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "dot(im1, im2)",
      "Inner product sum(im1[i] * im2[i])" },
    { "istats", 1, 2, 2, 4, MC_FN_ISTATS, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "istats(im[, mask])", "Multi-LHS unpacking: m, s[, lo, hi] = istats(im)" },
    { "icentroid", 1, 2, 2, 3, MC_FN_ICENTROID, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "icentroid(im[, th])", "Flux-weighted centroid: cx, cy[, flux] = icentroid(im)" },
    { "imedian", 1, 2, 1, 1, MC_FN_EAGER_RED, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "imedian(im[, mask])", "Exact median of pixels" },
    { "perc", 2, 3, 1, 1, MC_FN_EAGER_RED, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "perc(im, q[, mask])", "Quantile q in [0, 1]" },
    { "trace", 1, 1, 1, 1, MC_FN_EAGER_RED, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "trace(im)",
      "Sum of diagonal elements of 2D matrix" },

    /* 7. Stencil, neighborhood, and projection library functions */
    { "medfilt3", 1, 1, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "medfilt3(im)", "3x3 median filter" },
    { "medfilt5", 1, 1, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "medfilt5(im)", "5x5 median filter" },
    { "medfilt7", 1, 1, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "medfilt7(im)", "7x7 median filter" },
    { "gauss", 2, 3, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "gauss(im, sx[, sy])", "Separable 1D/2D Gaussian blur" },
    { "conv2", 2, 2, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "conv2(im, K)",
      "2D convolution with small kernel K" },
    { "imin3", 1, 1, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "imin3(im)",
      "3x3 neighborhood minimum (erosion)" },
    { "imax3", 1, 1, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "imax3(im)",
      "3x3 neighborhood maximum (dilation)" },
    { "boxmean", 2, 2, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 0, NULL, NULL, NULL, NULL, NULL,
      "boxmean(im, r)", "Separable (2r+1)x(2r+1) boxcar mean" },
    { "cumsum", 1, 2, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "cumsum(im[, axis])", "Cumulative sum along axis (default 0)" },
    { "pixunmap", 2, 3, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "pixunmap(im, map[, n])", "Segment-mean projection onto n regions of map" },
    { "rprof", 1, 3, 1, 1, MC_FN_STENCIL_LIB, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "rprof(im[, cx, cy])", "Annular radial profile mean" },

    /* 8. BLAS, LAPACK, FFT, tensor construction & concatenation */
    { "matmul", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "matmul(A, B)",
      "Matrix / vector product (or batched 3D)" },
    { "gemm", 5, 7, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "gemm(a, A, B, b, C[, tA, tB])", "CBLAS GEMM a*op(A)*op(B) + b*C" },
    { "gemv", 5, 6, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "gemv(a, A, x, b, y[, tA])", "CBLAS GEMV a*op(A)*x + b*y" },
    { "ger", 4, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "ger(a, x, y, A)",
      "CBLAS rank-1 update A + a*x*y^T" },
    { "outer", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "outer(x, y)",
      "Outer product x * y^T" },
    { "syrk", 4, 5, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "syrk(a, A, b, C[, tr])", "CBLAS symmetric rank-k update" },
    { "symv", 5, 6, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "symv(a, A, x, b, y[, uplo])", "CBLAS symmetric matrix-vector multiply" },
    { "axpy", 3, 3, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "axpy(a, x, y)",
      "Vector update a*x + y" },
    { "scal", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "scal(a, x)",
      "Scale tensor a*x" },
    { "trsv", 2, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "trsv(A, b[, uplo, tr])", "Triangular vector solve" },
    { "trsm", 2, 5, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "trsm(A, B[, uplo, tr, side])", "Triangular matrix solve" },
    { "gesv", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "gesv(A, B)",
      "General linear system solver A*X = B" },
    { "posv", 2, 3, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "posv(A, B[, uplo])",
      "Symmetric positive-definite solver" },
    { "gels", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "gels(A, B)",
      "QR least-squares solver" },
    { "solve", 2, 3, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "solve(A, B[, reg])", "Linear / Tikhonov ridge solver" },
    { "inv", 1, 1, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "inv(A)",
      "Matrix inverse" },
    { "pinv", 1, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "pinv(A[, rcond])",
      "SVD Moore-Penrose pseudoinverse" },
    { "svd", 1, 1, 1, 3, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "svd(A)",
      "Singular value decomposition (s = svd(A) or U, s, Vt = svd(A))" },
    { "eigh", 1, 1, 1, 2, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "eigh(C)",
      "Symmetric eigendecomposition (w = eigh(C) or w, V = eigh(C))" },
    { "einsum", 2, 3, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL,
      "einsum(sub, A[, B])", "Einstein summation tensor contraction" },
    { "fft", 1, 1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "fft(im)",
      "Forward 1D/2D/3D Discrete Fourier Transform" },
    { "ifft", 1, 1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "ifft(im)",
      "Normalized inverse 1D/2D/3D Discrete Fourier Transform" },
    { "fftshift", 1, 1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "fftshift(im)",
      "Swap half-spaces along all axes to center DC at (nx/2, ny/2)" },
    { "eye", 1, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "eye(n[, m])",
      "Identity matrix" },
    { "diag", 1, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "diag(v[, k])",
      "Extract diagonal or build diagonal matrix" },
    { "transpose", 1, 1, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "transpose(A)",
      "2D matrix transpose (or batched 3D)" },
    { "reshape", 2, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "reshape(A, n0[, n1, n2])", "Reshape dimensions preserving element count" },
    { "vec", 1, -1, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "vec(A)",
      "Flatten image to 1D vector or pack scalars" },
    { "zeros", 1, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "zeros(n0[, n1, n2, dt])", "Zero-filled tensor" },
    { "hstack", 2, -1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL,
      "hstack(A, B, ...)", "Concatenate images along axis 0 (x)" },
    { "vstack", 2, -1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL,
      "vstack(A, B, ...)", "Concatenate images along axis 1 (y)" },
    { "dstack", 2, -1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL,
      "dstack(A, B, ...)", "Concatenate images along axis 2 (z)" },
    { "cat", 3, -1, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL,
      "cat(axis, A, B, ...)", "Concatenate images along explicit axis 0..2" },
    { "rot90", 1, 2, 1, 1, MC_FN_LINALG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "rot90(im[, k])",
      "Rotate 2D image by 90*k degrees" },

    /* 9. String functions */
    { "strlen", 1, 1, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "strlen(s)",
      "Byte length of string s" },
    { "toupper", 1, 1, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "toupper(s)",
      "Convert string s to uppercase" },
    { "tolower", 1, 1, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "tolower(s)",
      "Convert string s to lowercase" },
    { "substr", 3, 3, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "substr(s, pos, len)",
      "1-based substring of s" },
    { "replace", 3, 3, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "replace(s, old, new)", "Replace first occurrence of old with new in s" },
    { "hex", 1, 1, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "hex(n)",
      "Format integer n as 0x... hexadecimal string" },
    { "oct", 1, 1, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "oct(n)",
      "Format integer n as 0o... octal string" },
    { "bin", 1, 1, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "bin(n)",
      "Format integer n as 0b... binary string" },

    /* 10. Program / temporal directives */
    { "post", 1, 1, 1, 1, MC_FN_PROG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "post(s)",
      "Explicitly post output stream s inside a program" },
    { "hist", 2, 2, 1, 1, MC_FN_PROG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "hist(s, N)",
      "Declare N-frame temporal history ring for stream s" },
    { "prev", 1, 2, 1, 1, MC_FN_PROG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "prev(s[, k])",
      "Read frame k ticks ago (default 1) from hist(s, N) ring" },
    { "ema", 2, 2, 1, 1, MC_FN_PROG, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, "ema(expr, alpha)",
      "Stateful exponential moving average with decay alpha" }
};

#define MC_FUNC_COUNT ((int) (sizeof(g_mc_funcs) / sizeof(g_mc_funcs[0])))

/**
 * mc_func_find_len - Look up a built-in function descriptor by name slice
 * @name: Pointer to function identifier characters (need not be null-terminated)
 * @len:  Length of the identifier in bytes
 *
 * Return: Pointer to matching `mc_func_t` entry, or NULL if not found
 */
const mc_func_t *mc_func_find_len(const char *name, size_t len)
{
    if (name == NULL || len == 0)
    {
        return NULL;
    }
    for (int i = 0; i < MC_FUNC_COUNT; i++)
    {
        if (strncmp(g_mc_funcs[i].name, name, len) == 0 && g_mc_funcs[i].name[len] == '\0')
        {
            return &g_mc_funcs[i];
        }
    }
    return NULL;
}

/**
 * mc_func_find - Look up a built-in function descriptor by null-terminated name
 * @name: Function identifier string (without `(`)
 *
 * Return: Pointer to matching `mc_func_t` entry, or NULL if not found
 */
const mc_func_t *mc_func_find(const char *name)
{
    if (name == NULL)
    {
        return NULL;
    }
    return mc_func_find_len(name, strlen(name));
}

/**
 * mc_func_replacement - Name the function that replaced a removed alias
 * @name: Function identifier string (without `(`)
 *
 * Return: The replacement's name, or NULL if @name is not a removed alias
 */
const char *mc_func_replacement(const char *name)
{
    static const char *const removed[][2] = {
        { "ln", "log" },      { "fabs", "abs" },  { "urand", "rand" },
        { "grand", "randn" }, { "nrm2", "norm" },
    };
    for (size_t i = 0; name != NULL && i < sizeof(removed) / sizeof(removed[0]); i++)
    {
        if (strcmp(name, removed[i][0]) == 0)
        {
            return removed[i][1];
        }
    }
    return NULL;
}

/**
 * mc_const_removed - Hint for a name that was one of the removed `_PI`-style constants
 * @name: Bare identifier
 *
 * Return: The hint, or NULL if @name was not one of them
 */
const char *mc_const_removed(const char *name)
{
    static const char *const removed[] = { "_PI", "_e",  "_gamma", "_c", "_h",
                                           "_k",  "_pc", "_ly",    "_AU" };
    for (size_t i = 0; name != NULL && i < sizeof(removed) / sizeof(removed[0]); i++)
    {
        if (strcmp(name, removed[i]) == 0)
        {
            return "the _PI-style constants were removed: use pi, exp(1), or write the value";
        }
    }
    return NULL;
}

/**
 * mc_is_reserved_const - Check whether @name is a calculator constant (`pi`, `nan`, `inf`)
 * @name: Bare identifier
 *
 * The constants and the built-in function names are reserved: they cannot be assigned or defined,
 * so the shell's evaluator and the session's always agree on them.
 *
 * Return: 1 if @name is a constant, 0 otherwise
 */
int mc_is_reserved_const(const char *name)
{
    return name != NULL &&
           (strcmp(name, "pi") == 0 || strcmp(name, "nan") == 0 || strcmp(name, "inf") == 0);
}

/**
 * mc_func_table - Return the complete built-in function table and entry count
 * @out_count: Optional output pointer receiving `MC_FUNC_COUNT`
 *
 * Return: Pointer to the static `g_mc_funcs` array
 */
const mc_func_t *mc_func_table(int *out_count)
{
    if (out_count != NULL)
    {
        *out_count = MC_FUNC_COUNT;
    }
    return g_mc_funcs;
}

/* ========================================================================= */
/* Calculator Catalog Display (--list / --list-json)                         */
/* ========================================================================= */

/**
 * struct mc_catalog_entry_t - Grouped operator/function summary row for `--list`
 * @category: Section heading grouping related operators/functions
 * @syntax:   Human-readable syntax summary
 * @token:    Legacy/symbolic token classification label
 * @opcode:   Execution opcode (`FI_BIN`, ...)
 * @stage:    Execution pipeline stage (`fprog_simd`, `prestage`, `fused_reduce`, etc.)
 * @descr:    Concise description of behavior
 */
typedef struct
{
    const char *category;
    const char *syntax;
    const char *token;
    const char *opcode;
    const char *stage;
    const char *descr;
} mc_catalog_entry_t;

static const mc_catalog_entry_t g_mc_catalog[] = {
    { "1. Element-Wise Operators", "+  -  *  /  %  @  .T", "TOK_OP_*", "FI_BIN", "fprog_simd",
      "Binary arithmetic (+, -, *, /, %), matrix multiply (@), transpose (.T)" },
    { "1. Element-Wise Operators", "-x", "TOK_OP_MINUS", "FI_NEG", "fprog_simd",
      "Unary arithmetic negation" },
    { "1. Element-Wise Operators", "<  <=  >  >=  ==  !=", "TOK_OP_*", "FI_BIN", "fprog_simd",
      "Relational and chained comparisons (a <= x <= b)" },
    { "1. Element-Wise Operators", "&&  ||  !x", "TOK_OP_AND/OR", "FI_BIN", "fprog_simd",
      "Logical AND, OR (short-circuits on scalars), and NOT" },
    { "1. Element-Wise Operators", "&  |  ^^  <<  >>  ~x", "TOK_OP_B*", "FI_BIN", "fprog_simd",
      "64-bit integer bitwise AND, OR, XOR (^^), shifts, complement" },
    { "1. Element-Wise Operators", "c ? a : b, where(c,a,b)", "TOK_FUNC_WHERE", "FI_SEL",
      "fprog_simd", "Per-pixel conditional select (c != 0 ? a : b)" },
    { "1. Element-Wise Operators", "=  +=  -=  *=  /=  im[mask]=", "TOK_EQUAL", "FI_STORE_DST",
      "fprog_simd", "Destination store (full image, N-D slice, masked LHS, or tuple)" },
    { "2. Scalar & Element-Wise Math", "sin(x)  cos(x)  tan(x)", "TOK_FUNC_D_D", "FI_FN1",
      "fprog_simd", "Trigonometric functions (vectorized via libmvec)" },
    { "2. Scalar & Element-Wise Math", "asin(x) acos(x) atan(x)", "TOK_FUNC_D_D", "FI_FN1",
      "fprog_simd", "Inverse trigonometric functions" },
    { "2. Scalar & Element-Wise Math", "sinh(x) cosh(x) tanh(x)", "TOK_FUNC_D_D", "FI_FN1",
      "fprog_simd", "Hyperbolic functions" },
    { "2. Scalar & Element-Wise Math", "exp(x)  exp2(x)", "TOK_FUNC_D_D", "FI_FN1", "fprog_simd",
      "Base-e and base-2 exponential functions" },
    { "2. Scalar & Element-Wise Math", "log(x)  log10(x)  log2(x)", "TOK_FUNC_D_D", "FI_FN1",
      "fprog_simd", "Natural (log), base-10 (log10), and base-2 (log2) logarithms" },
    { "2. Scalar & Element-Wise Math", "sqrt(x) cbrt(x)", "TOK_FUNC_D_D", "FI_FN1", "fprog_simd",
      "Square root and cube root" },
    { "2. Scalar & Element-Wise Math", "erf(x)  erfc(x)", "TOK_FUNC_D_D", "FI_FN1", "fprog_simd",
      "Error function and complementary error function" },
    { "2. Scalar & Element-Wise Math", "abs(x)", "TOK_FUNC_D_D", "FI_FN1 / FI_CPART", "fprog_simd",
      "Absolute value (or complex magnitude |z|)" },
    { "2. Scalar & Element-Wise Math", "ceil(x) floor(x) round(x) trunc(x)", "TOK_FUNC_D_D",
      "FI_FN1", "fprog_simd", "Nearest-integer rounding operations" },
    { "2. Scalar & Element-Wise Math", "posi(x) sign(x) wrap_pi(x) sinc(x)", "TOK_FUNC_D_D",
      "FI_FN1", "fprog_simd", "Step, signum, phase wrap to (-pi,pi], and normalized sinc" },
    { "2. Scalar & Element-Wise Math", "isnan(x) isfinite(x)", "TOK_FUNC_D_D", "FI_FN1",
      "fprog_simd", "IEEE-754 NaN and finite checks" },
    { "2. Scalar & Element-Wise Math", "pow(x, y)    atan2(y, x)", "TOK_FUNC_DD_D", "FI_FN2",
      "fprog_simd", "Power and four-quadrant inverse tangent" },
    { "2. Scalar & Element-Wise Math", "hypot(x, y)  fmod(x, y)", "TOK_FUNC_DD_D", "FI_FN2",
      "fprog_simd", "Euclidean norm sqrt(x^2 + y^2) and floating-point modulo" },
    { "2. Scalar & Element-Wise Math", "min(a, b)    max(a, b)", "TOK_FUNC_DD_D", "FI_FN2",
      "fprog_simd", "Per-pixel minimum and maximum" },
    { "2. Scalar & Element-Wise Math", "clamp(x, lo, hi)", "TOK_FUNC_DDD_D", "FI_FN3", "fprog_simd",
      "Per-pixel branchless clamp to [lo, hi]" },
    { "2. Scalar & Element-Wise Math", "f32 f64 u8 i8 u16 i16 u32 i32 u64 i64", "TOK_FUNC_D_D",
      "FI_FN1 (cast)", "fprog_simd", "Explicit scalar and image datatype casts" },
    { "3. Coordinates, PRNG & Complex", "ix iy ir itheta(im[, cx, cy])", "TOK_FUNC_COORD",
      "FI_COORD", "fprog_simd", "Per-pixel Cartesian and polar coordinates" },
    { "3. Coordinates, PRNG & Complex", "rand(im[, s]) randn(im[, s])", "TOK_FUNC_COORD",
      "FI_COORD", "fprog_simd", "Uniform [0, 1) and standard normal N(0, 1) PRNG" },
    { "3. Coordinates, PRNG & Complex", "real(z) imag(z) arg(z) conj(z) cplx expi", "TOK_FUNC_CPLX",
      "FI_CPART", "fprog_simd", "Complex parts, conjugate, pack, and phasor" },
    { "3. Coordinates, PRNG & Complex", "[expr for x=a:b] linspace(a,b,n)", "COMPREHENSION",
      "FI_COORD", "fprog_simd", "Array comprehensions, vector literals, and linear grids" },
    { "4. Stencils, Gather & Interpolation", "shift(im, dx[, dy]) roll(im, dx...)",
      "TOK_FUNC_SHIFT", "FI_SHIFT", "fprog_simd",
      "Edge-clamped shift and periodic wrap-around roll" },
    { "4. Stencils, Gather & Interpolation", "medfilt3/5/7(im) imin3 imax3 boxmean",
      "TOK_FUNC_MEDFILT", "FI_MEDFILT", "fprog_simd",
      "2D median, erosion, dilation, and separable boxcar filters" },
    { "4. Stencils, Gather & Interpolation", "lut(expr, tbl[, x0, dx])", "TOK_FUNC_LUT", "FI_LUT",
      "fprog_simd", "1D linear-interpolated lookup into table tbl" },
    { "4. Stencils, Gather & Interpolation", "pixremap(im, map[, fill])", "TOK_FUNC_PIXREMAP",
      "FI_GATHER", "fprog_simd", "1D index-map gather im[map[i]]" },
    { "4. Stencils, Gather & Interpolation", "interp2(im, xmap, ymap)", "TOK_FUNC_INTERP2",
      "FI_INTERP2", "fprog_simd", "2D bilinear sub-pixel gather" },
    { "5. Spatial & Projection Pre-Stages", "gauss(expr, sx[, sy]) conv2(im, K)", "TOK_FUNC_GAUSS",
      "AX_KIND_GAUSS", "prestage", "Separable Gaussian blur and small-kernel 2D convolution" },
    { "5. Spatial & Projection Pre-Stages", "cumsum(expr[, axis])", "TOK_FUNC_CUMSUM",
      "AX_KIND_CUMSUM", "prestage", "Cumulative sum along an axis" },
    { "5. Spatial & Projection Pre-Stages", "pixunmap(expr, map[, n]) rprof", "TOK_FUNC_PIXUNMAP",
      "AX_KIND_SEG", "prestage", "Segment-mean projection and annular radial profile" },
    { "6. Reductions & Multi-Output Unpacking", "itot imean istd irms ivar imin imax",
      "TOK_FUNC_IM_D", "FI_RED (SUM)", "fused_reduce", "Fused reductions (optional mask or axis)" },
    { "6. Reductions & Multi-Output Unpacking", "norm asum icount iargmax iargmin iamax",
      "TOK_FUNC_IM_L", "FI_RED (ARG*)", "fused_reduce",
      "Norms, non-zero count, and argmax/argmin (with 2D/3D unpacking)" },
    { "6. Reductions & Multi-Output Unpacking", "dot(im1, im2)", "TOK_FUNC_IMIM_D", "FI_RED (DOT)",
      "fused_reduce", "Two-stream inner product" },
    { "6. Reductions & Multi-Output Unpacking", "istats(expr)", "TOK_FUNC_IM_D", "FI_RED (STATS)",
      "fused_reduce", "Multi-LHS unpacking: m, s[, min, max] = istats(im)" },
    { "6. Reductions & Multi-Output Unpacking", "icentroid(expr[, th])", "TOK_FUNC_IM_D", "FI_CENT",
      "fused_reduce", "Flux-weighted centroid unpacking: cx, cy[, flux]" },
    { "6. Reductions & Multi-Output Unpacking", "imedian(im) perc(im,q) trace", "TOK_FUNC_IM*_D",
      "EAGER_RED", "eager_reduce", "Exact median, percentile, and matrix trace" },
    { "7. BLAS / Matrix, LAPACK & FFT", "matmul gemm gemv outer ger syrk symv axpy scal",
      "TOK_FUNC_IMIM_IM", "FKEPT_MATMUL", "fkept_blas", "CBLAS matrix and vector operations" },
    { "7. BLAS / Matrix, LAPACK & FFT", "trsv trsm gesv posv gels solve inv pinv svd eigh",
      "TOK_FUNC_IMIM_IM", "LAPACK_SOLVE", "eager_lapack",
      "Linear solvers, inverse, pseudoinverse, SVD, and eigensolver" },
    { "7. BLAS / Matrix, LAPACK & FFT", "einsum fft ifft fftshift transpose reshape rot90",
      "TOK_FUNC_IMIM_IM", "FKEPT_TRANSPOSE", "fkept_blas",
      "Einstein contraction, FFTW transforms, transpose, reshape, rot90" },
    { "7. BLAS / Matrix, LAPACK & FFT", "eye diag vec zeros hstack vstack dstack cat",
      "TOK_FUNC_IMIM_IM", "TENSOR_INIT", "eager_init",
      "Tensor constructors and axis concatenation" },
    { "8. Inline Functions, Directives & User C Extensions",
      "def if/elif/else for local skip post hist prev ema", "DIRECTIVE", "PROG_CTRL", "prog_ctrl",
      "Inline functions, control flow, stream posting, temporal state" },
    { NULL, NULL, NULL, NULL, NULL, NULL }
};

/**
 * str_contains_ci - Case-insensitive substring search
 * @haystack: String to search within (may be NULL)
 * @needle:   Substring pattern to find (empty or NULL matches everything)
 *
 * Return: 1 if @needle appears in @haystack (case-insensitive), 0 otherwise
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
 * catalog_entry_matches - Test whether a catalog row matches filter @pat
 * @e:   Catalog row to inspect
 * @pat: Optional case-insensitive filter substring (NULL or `""` matches all)
 *
 * Return: 1 if any column of @e matches @pat, 0 otherwise
 */
static int catalog_entry_matches(const mc_catalog_entry_t *e, const char *pat)
{
    if (pat == NULL || pat[0] == '\0')
    {
        return 1;
    }
    return str_contains_ci(e->category, pat) || str_contains_ci(e->syntax, pat) ||
           str_contains_ci(e->token, pat) || str_contains_ci(e->opcode, pat) ||
           str_contains_ci(e->stage, pat) || str_contains_ci(e->descr, pat);
}

/**
 * ext_kind_opcode - Map a user C extension tier to its displayed opcode name
 * @kind: Extension kind enum (`MILK_EXT_BLK1` .. `MILK_EXT_OP`)
 *
 * Return: Static opcode label string for catalog display
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
 * format_ext_info - Format signature and description strings for a user C extension
 * @e:       Registered user extension entry
 * @sig:     Output buffer for call signature
 * @sig_sz:  Size of @sig buffer in bytes
 * @desc:    Output buffer for human-readable tier/parameter description
 * @desc_sz: Size of @desc buffer in bytes
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
 * ext_matches - Test whether a registered user C extension matches filter @pat
 * @e:   Registered user extension entry
 * @pat: Optional case-insensitive filter substring
 *
 * Return: 1 if @e matches @pat, 0 otherwise
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
 * json_escape - Write a JSON-escaped double-quoted string literal to `stdout`
 * @s: Input string (NULL treated as empty string)
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
 * mcalc_print_catalog - Print the calculator operator/function catalog to `stdout`
 * @pattern:   Optional case-insensitive substring filter (NULL or `""` for all)
 * @json_mode: Non-zero to emit machine-readable JSON (`--list-json`), 0 for table
 *
 * Lists built-in operators and functions grouped by category along with any
 * dynamically loaded user C extensions (`-l` / `loadlib`).
 *
 * Return: 0 if at least one entry matched (or @pattern is empty), 1 if no matches
 */
int mcalc_print_catalog(const char *pattern, int json_mode)
{
    int ext_count = milk_calc_ext_count();
    if (json_mode)
    {
        int matched = 0;
        printf("{\n  \"builtins\": [\n");
        for (int i = 0; g_mc_catalog[i].category != NULL; i++)
        {
            const mc_catalog_entry_t *e = &g_mc_catalog[i];
            if (!catalog_entry_matches(e, pattern))
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
            printf(", \"opcode\": ");
            json_escape(e->opcode);
            printf(", \"stage\": ");
            json_escape(e->stage);
            printf(", \"description\": ");
            json_escape(e->descr);
            printf("}");
            matched++;
        }
        printf("\n  ],\n  \"user_extensions\": [\n");
        int umatched = 0;
        for (int i = 0; i < ext_count; i++)
        {
            const milk_calc_ext_entry_t *e = milk_calc_ext_get(i);
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
            printf(", \"token\": \"TOK_FUNC_USER\", \"opcode\": ");
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

    const char *last_cat = NULL;
    int         matched  = 0;
    for (int i = 0; g_mc_catalog[i].category != NULL; i++)
    {
        const mc_catalog_entry_t *e = &g_mc_catalog[i];
        if (!catalog_entry_matches(e, pattern))
        {
            continue;
        }
        if (last_cat == NULL || strcmp(last_cat, e->category) != 0)
        {
            printf("\n%s\n", e->category);
            printf("  %-29s %-18s %s\n", "TOKEN / SYNTAX", "OPCODE", "DESCRIPTION");
            last_cat = e->category;
        }
        printf("  %-29s %-18s %s\n", e->syntax, e->opcode, e->descr);
        matched++;
    }

    int umatched = 0;
    for (int i = 0; i < ext_count; i++)
    {
        const milk_calc_ext_entry_t *e = milk_calc_ext_get(i);
        if (e == NULL || !ext_matches(e, pattern))
        {
            continue;
        }
        if (umatched == 0)
        {
            printf("\n9. Loaded User C Extensions (-l / loadlib)\n");
            printf("  %-29s %-18s %s\n", "TOKEN / SYNTAX", "OPCODE", "DESCRIPTION");
        }
        char sig[64];
        char desc[256];
        format_ext_info(e, sig, sizeof(sig), desc, sizeof(desc));
        printf("  %-29s %-18s %s\n", sig, ext_kind_opcode(e->kind), desc);
        umatched++;
        matched++;
    }

    if (matched == 0)
    {
        printf("\n  No calculator functions or extensions matched '%s'.\n\n",
               (pattern != NULL) ? pattern : "");
        fflush(stdout);
        return 1;
    }
    fflush(stdout);
    return 0;
}

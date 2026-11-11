// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_func.c
 * @brief Built-in function table, scalar math helpers, and the `--list` catalog (see mc_func.h).
 *
 * One static table, g_mc_funcs, holds an entry per built-in function and directive, grouped by
 * kind; a lookup scans it. mc_isnan() and mc_isfinite() test the IEEE-754 bits, so they stay
 * valid under `-ffast-math`; mc_fmodd() and mc_wrap_pi() use the FMA formulas of the kernels, so
 * `%` and wrap_pi agree on numbers and images, with NaN instead of a wrong value once the
 * quotient reaches 2^52. mcalc_print_catalog() (`@calc --list`, `milk-fpsexec-milk-calc --list`)
 * prints the operators, statements and constants (g_mc_syntax), then g_mc_funcs by category with
 * each function's signature and help, then the C extensions, as text or as JSON (`--json`).
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
 * The formula of the generated kernels' `mc_fmodd()` (jit/mc_prelude.h), so `%` on numbers and
 * on images agree, and it vectorizes there: q = trunc(x / y) may be off by one, and the remainder
 * is then recomputed with the neighbouring quotient by an FMA, which is exact.
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
 * The formula of the generated kernels' `mc_wrap_pid()` (jit/mc_prelude.h), as mc_fmodd() with
 * the quotient rounded to nearest: remainder(@a, 2 pi), with `-pi` mapped to `pi`.
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
 * calculator and the shell's evaluator (run/mc_scalar_ops.c), so both read a condition alike.
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
    { "gemm", 5, 5, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "gemm(a, A, B, b, C)", "CBLAS GEMM a*op(A)*op(B) + b*C" },
    { "gemv", 5, 5, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "gemv(a, A, x, b, y)", "CBLAS GEMV a*op(A)*x + b*y" },
    { "ger", 4, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "ger(a, x, y, A)",
      "CBLAS rank-1 update A + a*x*y^T" },
    { "outer", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "outer(x, y)",
      "Outer product x * y^T" },
    { "syrk", 4, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "syrk(a, A, b, C)",
      "CBLAS symmetric rank-k update" },
    { "symv", 5, 5, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "symv(a, A, x, b, y)", "CBLAS symmetric matrix-vector multiply" },
    { "axpy", 3, 3, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "axpy(a, x, y)",
      "Vector update a*x + y" },
    { "scal", 2, 2, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "scal(a, x)",
      "Scale tensor a*x" },
    { "trsv", 2, 4, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "trsv(A, b[, uplo, diag])", "Triangular vector solve" },
    { "trsm", 2, 5, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "trsm(A, B[, uplo, diag, a])", "Triangular matrix solve" },
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
    { "vec", 1, -1, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL, "vec(A | x0, ...)",
      "Flatten image to 1D vector or pack scalars" },
    { "zeros", 1, 3, 1, 1, MC_FN_LINALG, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "zeros(nx[, ny, nz])", "Zero-filled tensor" },
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
      "0-based substring of s" },
    { "replace", 3, 3, 1, 1, MC_FN_STR, 0, 0, 1, NULL, NULL, NULL, NULL, NULL,
      "replace(s, old, new)", "Replace all occurrences of old with new in s" },
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
 * mc_is_stream_prefix - Check whether @name is `S` or `L`, the letters of `S:name` and `L:name`
 * @name: Bare identifier
 *
 * They name images only (`U, S, Vt = svd(A)`): a variable of that name before `:` (`im[S:n]`)
 * would read as a stream prefix.
 *
 * Return: 1 if @name is `S` or `L`, 0 otherwise
 */
int mc_is_stream_prefix(const char *name)
{
    return name != NULL && (name[0] == 'S' || name[0] == 'L') && name[1] == '\0';
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
/* Calculator Catalog Display (--list, --list --json)                        */
/* ========================================================================= */

/* What `--list` shows that is not a function of g_mc_funcs: { syntax, description } */
static const char *const g_mc_syntax[][2] = {
    { "+  -  *  /  %  @  .T", "Arithmetic, matrix product (@), transpose (.T)" },
    { "-x", "Negation" },
    { "<  <=  >  >=  ==  !=", "Comparisons; < <= > >= chain (a <= x <= b)" },
    { "&&  ||  !x", "Logical AND, OR (short-circuit on scalars), NOT" },
    { "&  |  ^^  <<  >>  ~x", "64-bit integer AND, OR, XOR (^^), shifts, complement" },
    { "c ? a : b", "Conditional select, per pixel on images" },
    { "=  +=  -=  *=  /=",
      "Assignment to an image, a slice, a mask (im[mask] =) or several names" },
    { "def  if/elif/else  for", "Inline function, condition, loop" },
    { "local  skip", "In a unit: an image that is not shared, the end of the tick" },
    { "pi  nan  inf", "Constants (reserved names)" },
};
#define MC_SYNTAX_COUNT ((int) (sizeof(g_mc_syntax) / sizeof(g_mc_syntax[0])))

/* The categories of `--list`, in the order shown: the first holds g_mc_syntax */
static const char *const g_mc_categories[] = { "Operators, statements and constants",
                                               "Math",
                                               "Datatype casts",
                                               "Coordinates and random numbers",
                                               "Complex images",
                                               "Shift, lookup and remap",
                                               "Reductions",
                                               "Filters, stencils and projections",
                                               "Matrix, linear algebra and FFT",
                                               "Strings",
                                               "Unit programs" };
#define MC_CATEGORY_COUNT ((int) (sizeof(g_mc_categories) / sizeof(g_mc_categories[0])))

/** func_category() - the index in g_mc_categories of built-in function @f. */
static int func_category(const mc_func_t *f)
{
    switch (f->kind)
    {
    case MC_FN_SCALAR:
        return (f->cast_dt != 0) ? 2 : 1;
    case MC_FN_WHERE:
        return 1;
    case MC_FN_COORD:
    case MC_FN_PRNG:
        return 3;
    case MC_FN_CPLX:
        return 4;
    case MC_FN_SHIFT:
    case MC_FN_LUT:
    case MC_FN_GATHER:
        return 5;
    case MC_FN_REDUCE_D:
    case MC_FN_REDUCE_L:
    case MC_FN_DOT:
    case MC_FN_ISTATS:
    case MC_FN_ICENTROID:
    case MC_FN_EAGER_RED:
        return 6;
    case MC_FN_STENCIL_LIB:
        return 7;
    case MC_FN_LINSPACE:
    case MC_FN_LINALG:
        return 8;
    case MC_FN_STR:
        return 9;
    case MC_FN_PROG:
        return 10;
    }
    return 1;
}

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
 * format_ext_info - Format signature and description strings for a C extension
 * @e:       Registered extension entry
 * @sig:     Output buffer for call signature
 * @sig_sz:  Size of @sig buffer in bytes
 * @desc:    Output buffer for the tier (docs/guide/calc_c_extensions.md) and parameter names
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
        snprintf(desc, desc_sz, "Tier 1 block kernel, one input");
        break;
    case MILK_EXT_BLK2:
        snprintf(sig, sig_sz, "%s(a, b)", e->name);
        snprintf(desc, desc_sz, "Tier 1 block kernel, two inputs");
        break;
    case MILK_EXT_STENCIL:
        snprintf(sig, sig_sz, "%s(im)", e->name);
        snprintf(desc, desc_sz, "Tier 1 stencil kernel (halo r=%d, cost=%d)", e->radius, e->cost);
        break;
    case MILK_EXT_IMG:
        snprintf(sig, sig_sz, "%s(im)", e->name);
        snprintf(desc, desc_sz, "Tier 1 whole-image kernel (cost=%d)", e->cost);
        break;
    case MILK_EXT_OP:
    {
        const milk_calc_op_ports_t *ports = (e->get_ports != NULL) ? e->get_ports() : NULL;
        int                         nin   = (ports != NULL) ? ports->nin : 0;
        int                         nout  = (ports != NULL) ? ports->nout : 0;
        snprintf(sig, sig_sz, "%s(%d in -> %d out)", e->name, nin, nout);
        snprintf(desc, desc_sz, "Tier 2 operator (%d in, %d out)", nin, nout);
        break;
    }
    default:
        snprintf(sig, sig_sz, "%s(...)", e->name);
        snprintf(desc, desc_sz, "C extension");
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

#define MC_EXT_CATEGORY "C extensions (built in, -l, loadlib)"

/**
 * list_row - Print one `--list` row if it matches @pat
 * @cat:  Category of the row
 * @name: Function or extension name; NULL for an operator row
 * @sig:  Syntax or call signature
 * @desc: Description
 * @pat:  Case-insensitive filter on @cat, @sig and @desc; NULL or "" matches all
 * @json: Non-zero for a JSON object, 0 for a text line under the category's heading
 * @last: Category of the last row printed, updated; NULL before the first row
 *
 * Return: 1 if the row was printed, 0 if @pat left it out
 */
static int list_row(const char  *cat,
                    const char  *name,
                    const char  *sig,
                    const char  *desc,
                    const char  *pat,
                    int          json,
                    const char **last)
{
    if (!str_contains_ci(cat, pat) && !str_contains_ci(sig, pat) && !str_contains_ci(desc, pat))
    {
        return 0;
    }
    if (json)
    {
        printf("%s    {\"category\": ", (*last != NULL) ? ",\n" : "");
        json_escape(cat);
        if (name != NULL)
        {
            printf(", \"name\": ");
            json_escape(name);
        }
        printf(", \"signature\": ");
        json_escape(sig);
        printf(", \"description\": ");
        json_escape(desc);
        printf("}");
    }
    else
    {
        if (*last != cat)
        {
            printf("\n%s\n", cat);
        }
        printf("  %-30s %s\n", sig, desc);
    }
    *last = cat;
    return 1;
}

/**
 * mcalc_print_catalog - Print the calculator operator/function catalog to `stdout`
 * @pattern:   Optional case-insensitive substring filter (NULL or `""` for all)
 * @json_mode: Non-zero to emit machine-readable JSON (`--list --json`), 0 for table
 *
 * Lists the operators, statements and constants, the built-in functions of g_mc_funcs by
 * category, and the C extensions: the built-in ones and those loaded with `-l` / `loadlib`.
 *
 * Return: 0 if at least one entry matched (or @pattern is empty), 1 if no matches
 */
int mcalc_print_catalog(const char *pattern, int json_mode)
{
    const char *last    = NULL;
    int         matched = 0;

    if (json_mode)
    {
        printf("{\n  \"builtins\": [\n");
    }
    for (int i = 0; i < MC_SYNTAX_COUNT; i++)
    {
        matched += list_row(g_mc_categories[0], NULL, g_mc_syntax[i][0], g_mc_syntax[i][1], pattern,
                            json_mode, &last);
    }
    for (int c = 1; c < MC_CATEGORY_COUNT; c++)
    {
        for (int i = 0; i < MC_FUNC_COUNT; i++)
        {
            const mc_func_t *f = &g_mc_funcs[i];
            if (func_category(f) == c)
            {
                matched += list_row(g_mc_categories[c], f->name, f->signature, f->help, pattern,
                                    json_mode, &last);
            }
        }
    }
    if (json_mode)
    {
        printf("\n  ],\n  \"user_extensions\": [\n");
        last = NULL;
    }
    int ext_count = milk_calc_ext_count();
    for (int i = 0; i < ext_count; i++)
    {
        const milk_calc_ext_entry_t *e = milk_calc_ext_get(i);
        if (e == NULL)
        {
            continue;
        }
        char sig[64];
        char desc[256];
        format_ext_info(e, sig, sizeof(sig), desc, sizeof(desc));
        matched += list_row(MC_EXT_CATEGORY, e->name, sig, desc, pattern, json_mode, &last);
    }
    if (json_mode)
    {
        printf("\n  ]\n}\n");
    }
    else if (matched == 0)
    {
        printf("\n  No calculator function or extension matched '%s'.\n\n",
               (pattern != NULL) ? pattern : "");
    }
    fflush(stdout);
    return (matched > 0 || pattern == NULL || pattern[0] == '\0') ? 0 : 1;
}

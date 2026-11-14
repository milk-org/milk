// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mathfuncs.h
 * @brief   Inline scalar function wrappers for image arithmetic dispatch
 */

#ifndef COREMOD_MODULE_ARITH_MATHFUNCS_H
#define COREMOD_MODULE_ARITH_MATHFUNCS_H

#include <math.h>

#define DEF_UNARY_P(name, expr_d, expr_f)   \
    static inline double P##name(double a)  \
    {                                       \
        return (expr_d);                    \
    }                                       \
    static inline float P##name##f(float a) \
    {                                       \
        return (expr_f);                    \
    }

DEF_UNARY_P(acos, acos(a), acosf(a))
DEF_UNARY_P(asin, asin(a), asinf(a))
DEF_UNARY_P(atan, atan(a), atanf(a))
DEF_UNARY_P(ceil, ceil(a), ceilf(a))
DEF_UNARY_P(cos, cos(a), cosf(a))
DEF_UNARY_P(cosh, cosh(a), coshf(a))
DEF_UNARY_P(exp, exp(a), expf(a))
DEF_UNARY_P(fabs, fabs(a), fabsf(a))
DEF_UNARY_P(floor, floor(a), floorf(a))
DEF_UNARY_P(ln, log(a), logf(a))
DEF_UNARY_P(log, log10(a), log10f(a))
DEF_UNARY_P(sqrt, sqrt(a), sqrtf(a))
DEF_UNARY_P(sin, sin(a), sinf(a))
DEF_UNARY_P(sinh, sinh(a), sinhf(a))
DEF_UNARY_P(tan, tan(a), tanf(a))
DEF_UNARY_P(tanh, tanh(a), tanhf(a))
DEF_UNARY_P(positive, (a > 0.0) ? 1.0 : 0.0, (a > 0.0f) ? 1.0f : 0.0f)
#undef DEF_UNARY_P

#define DEF_BINARY_P(name, expr_d, expr_f)           \
    static inline double P##name(double a, double b) \
    {                                                \
        return (expr_d);                             \
    }                                                \
    static inline float P##name##f(float a, float b) \
    {                                                \
        return (expr_f);                             \
    }

DEF_BINARY_P(fmod, fmod(a, b), fmodf(a, b))
DEF_BINARY_P(pow,
             (b == 2.0)   ? a *a
             : (b == 0.5) ? sqrt(a)
             : (b == 1.0) ? a
             : (b == 0.0) ? 1.0
                          : pow(a, b),
             (b == 2.0f)   ? a *a
             : (b == 0.5f) ? sqrtf(a)
             : (b == 1.0f) ? a
             : (b == 0.0f) ? 1.0f
                           : powf(a, b))
DEF_BINARY_P(add, a + b, a + b)
DEF_BINARY_P(subm, b - a, b - a)
DEF_BINARY_P(sub, a - b, a - b)
DEF_BINARY_P(mult, a *b, a *b)
DEF_BINARY_P(div, a / b, a / b)
DEF_BINARY_P(div1, b / a, b / a)
DEF_BINARY_P(minv, (a < b) ? a : b, (a < b) ? a : b)
DEF_BINARY_P(maxv, (a > b) ? a : b, (a > b) ? a : b)
DEF_BINARY_P(testlt, (a < b) ? 1.0 : 0.0, (a < b) ? 1.0f : 0.0f)
DEF_BINARY_P(testmt, (a >= b) ? 1.0 : 0.0, (a >= b) ? 1.0f : 0.0f)
DEF_BINARY_P(teste, (a == b) ? 1.0 : 0.0, (a == b) ? 1.0f : 0.0f)
DEF_BINARY_P(testne, (a != b) ? 1.0 : 0.0, (a != b) ? 1.0f : 0.0f)
DEF_BINARY_P(testle, (a <= b) ? 1.0 : 0.0, (a <= b) ? 1.0f : 0.0f)
DEF_BINARY_P(testge, (a >= b) ? 1.0 : 0.0, (a >= b) ? 1.0f : 0.0f)
DEF_BINARY_P(testgt, (a > b) ? 1.0 : 0.0, (a > b) ? 1.0f : 0.0f)
DEF_BINARY_P(and, (a != 0.0 && b != 0.0) ? 1.0 : 0.0, (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f)
DEF_BINARY_P(or, (a != 0.0 || b != 0.0) ? 1.0 : 0.0, (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f)
#undef DEF_BINARY_P

/** Clamp @a to `[b, c]`. */
static inline double Ptrunc(double a, double b, double c)
{
    return (a < b) ? b : (a > c) ? c : a;
}

#endif

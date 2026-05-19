/**
 * @file    mathfuncs.c
 * @brief   Scalar function wrappers for image arithmetic
 *
 * Provides named function-pointer wrappers around
 * standard math library functions. These wrappers
 * exist so the image arithmetic dispatch engine
 * (imfunctions.c) can apply any math operation
 * uniformly across image pixels via a function
 * pointer of type double(*)(double) or
 * double(*)(double,double).
 *
 * Function families:
 *  - P<func>(double)       — unary transcendentals
 *    (Pacos, Psin, Pexp, Psqrt, …)
 *  - P<op>(double,double)  — binary arithmetic
 *    (Padd, Psub, Pmult, Pdiv, Ppow, Pfmod)
 *  - Ptest<cmp>(double,double) — relational ops
 *    returning 1.0/0.0 (Ptestlt, Ptestge, …)
 *  - Pand/Por(double,double)   — logical ops
 *  - Ptrunc(double,double,double) — clamp to range
 *  - CP<op>_CD_CD / CP<op>_CF_CF — complex double/
 *    float arithmetic (add, sub, mult, div)
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include "libmilkdata/milkdata.h"
#endif // complex types
#include <math.h>

/* ============================================================
 * Unary transcendental wrappers (double → double)
 *
 * Each P<func> wraps the corresponding C math.h
 * function. The naming convention is P<func> where
 * <func> matches the standard name (acos, sin, …)
 * except Pln = log() and Plog = log10().
 * ========================================================== */

double Pacos(double a)
{
    return ((double) acos(a));
}
double Pasin(double a)
{
    return ((double) asin(a));
}
double Patan(double a)
{
    return ((double) atan(a));
}
double Pceil(double a)
{
    return ((double) ceil(a));
}
double Pcos(double a)
{
    return ((double) cos(a));
}
double Pcosh(double a)
{
    return ((double) cosh(a));
}
double Pexp(double a)
{
    return ((double) exp(a));
}
double Pfabs(double a)
{
    return ((double) fabs(a));
}
double Pfloor(double a)
{
    return ((double) floor(a));
}
double Pln(double a)
{
    return ((double) log(a));
}
double Plog(double a)
{
    return ((double) log10(a));
}
double Psqrt(double a)
{
    return ((double) sqrt(a));
}
double Psin(double a)
{
    return ((double) sin(a));
}
double Psinh(double a)
{
    return ((double) sinh(a));
}
double Ptan(double a)
{
    return ((double) tan(a));
}
double Ptanh(double a)
{
    return ((double) tanh(a));
}

/**
 * @brief Positive threshold (Heaviside step)
 *
 * Returns 1.0 if a > 0, else 0.0.
 */
double Ppositive(double a)
{
    double value = 0.0;
    if(a > 0.0)
    {
        value = (double) 1.0;
    }
    return (value);
}

/* ============================================================
 * Binary arithmetic wrappers (double,double → double)
 *
 * Psubm(a,b) returns b-a (reverse subtract), used
 * when the scalar is subtracted FROM the image.
 * Pdiv1(a,b) returns b/a (reverse divide).
 * ========================================================== */

double Pfmod(
    double a,
    double b)
{
    return ((double) fmod(a, b));
}

double Ppow(
    double a,
    double b)
{
    return ((double) pow(a, b));
}

double Padd(
    double a,
    double b)
{
    return ((double) a + b);
}

double Psubm(
    double a,
    double b)
{
    return ((double) b - a);
}

double Psub(
    double a,
    double b)
{
    return ((double) a - b);
}

double Pmult(
    double a,
    double b)
{
    return ((double) a * b);
}

double Pdiv(
    double a,
    double b)
{
    return ((double) a / b);
}

double Pdiv1(
    double a,
    double b)
{
    return ((double) b / a);
}

/* ============================================================
 * Min / Max
 * ========================================================== */

/**
 * @brief Element-wise minimum of two values
 */
double Pminv(
    double a,
    double b)
{
    if(a < b)
    {
        return (a);
    }
    else
    {
        return (b);
    }
}

/**
 * @brief Element-wise maximum of two values
 */
double Pmaxv(
    double a,
    double b)
{
    if(a > b)
    {
        return (a);
    }
    else
    {
        return (b);
    }
}

/* ============================================================
 * Relational ops (double,double → 1.0 or 0.0)
 *
 * Used as function pointers by the image comparison
 * dispatch (arith_image_testlt, arith_image_teste,
 * etc.).
 * ========================================================== */

/** @brief a < b → 1.0 */
double Ptestlt(
    double a,
    double b)
{
    if(a < b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/** @brief a >= b → 1.0 ("mt" = more than) */
double Ptestmt(
    double a,
    double b)
{
    if(a < b)
    {
        return ((double) 0.0);
    }
    else
    {
        return ((double) 1.0);
    }
}

/** @brief a == b → 1.0 */
double Pteste(
    double a,
    double b)
{
    if(a == b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/** @brief a != b → 1.0 */
double Ptestne(
    double a,
    double b)
{
    if(a != b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/** @brief a <= b → 1.0 */
double Ptestle(
    double a,
    double b)
{
    if(a <= b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/** @brief a >= b → 1.0 */
double Ptestge(
    double a,
    double b)
{
    if(a >= b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/* ============================================================
 * Logical ops (double,double → 1.0 or 0.0)
 * ========================================================== */

/** @brief Logical AND: (a!=0 && b!=0) → 1.0 */
double Pand(
    double a,
    double b)
{
    if((a != 0.0) && (b != 0.0))
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/** @brief Logical OR: (a!=0 || b!=0) → 1.0 */
double Por(
    double a,
    double b)
{
    if((a != 0.0) || (b != 0.0))
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

/**
 * @brief Clamp value to [b, c] range
 *
 * @param a  Input value
 * @param b  Lower bound
 * @param c  Upper bound
 * @return Clamped value
 */
double Ptrunc(
    double a,
    double b,
    double c)
{
    double value;
    value = a;
    if(a < b)
    {
        value = b;
    };
    if(a > c)
    {
        value = c;
    };
    return (value);
}

/* ============================================================
 * Complex double arithmetic
 *
 * Uses the complex_double struct {.re, .im} for
 * portability (avoids C99 _Complex which is not
 * consistently available across targets).
 * ========================================================== */

/** @brief Complex double addition */
complex_double CPadd_CD_CD(
    complex_double a, complex_double b)
{
    complex_double v;
    v.re = a.re + b.re;
    v.im = a.im + b.im;
    return (v);
}

/** @brief Complex double subtraction */
complex_double CPsub_CD_CD(
    complex_double a, complex_double b)
{
    complex_double v;
    v.re = a.re - b.re;
    v.im = a.im - b.im;
    return (v);
}

/** @brief Complex double multiplication */
complex_double CPmult_CD_CD(
    complex_double a, complex_double b)
{
    complex_double v;
    v.re = a.re * b.re - a.im * b.im;
    v.im = a.re * b.im + a.im * b.re;
    return (v);
}

/**
 * @brief Complex double division via conjugate
 *
 * Computes a/b = (a·conj(b)) / |b|² to avoid
 * transcendentals.
 */
complex_double CPdiv_CD_CD(
    complex_double a, complex_double b)
{
    complex_double v;
    double         den;

    den = b.re * b.re + b.im * b.im;

    v.re = (a.re * b.re + a.im * b.im) / den;
    v.im = (a.im * b.re - a.re * b.im) / den;

    return (v);
}

/* ============================================================
 * Complex float arithmetic
 * ========================================================== */

/** @brief Complex float addition */
complex_float CPadd_CF_CF(
    complex_float a, complex_float b)
{
    complex_float v;
    v.re = a.re + b.re;
    v.im = a.im + b.im;
    return (v);
}

/** @brief Complex float subtraction */
complex_float CPsub_CF_CF(
    complex_float a, complex_float b)
{
    complex_float v;
    v.re = a.re - b.re;
    v.im = a.im - b.im;
    return (v);
}

/** @brief Complex float multiplication */
complex_float CPmult_CF_CF(
    complex_float a, complex_float b)
{
    complex_float v;
    v.re = a.re * b.re - a.im * b.im;
    v.im = a.re * b.im + a.im * b.re;
    return (v);
}

/**
 * @brief Complex float division via conjugate
 */
complex_float CPdiv_CF_CF(
    complex_float a, complex_float b)
{
    complex_float v;
    float         den;

    den = b.re * b.re + b.im * b.im;

    v.re = (a.re * b.re + a.im * b.im) / den;
    v.im = (a.im * b.re - a.re * b.im) / den;

    return (v);
}

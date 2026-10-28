// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_stats.h
 * @brief   Whole-image, axis-wise, and masked statistical reductions
 */

#ifndef COREMOD_ARITH_IMAGE_STATS_H
#define COREMOD_ARITH_IMAGE_STATS_H

#include <math.h>
#include <stdint.h>
#include <libfps/IMGID.h>
#include "libmilkcommon/milk_compiler.h"

/** Reduction operation codes shared across `COREMOD_arith` and `libmilkscript/calc`. */
typedef enum
{
    ARITH_RED_SUM,
    ARITH_RED_MEAN,
    ARITH_RED_MIN,
    ARITH_RED_MAX,
    ARITH_RED_STD,
    ARITH_RED_VAR,
    ARITH_RED_NORM,
    ARITH_RED_ASUM,
    ARITH_RED_COUNT,
    ARITH_RED_DOT,
    ARITH_RED_PERC,
    ARITH_RED_ALL
} arith_reduce_op;

/**
 * arith_reduce_op_promote - Combine two reduction ops on the same input into a single-pass op
 * @a: First reduction operation
 * @b: Second reduction operation
 *
 * Return: Combined `arith_reduce_op`, or `(arith_reduce_op) -1` if @a and @b cannot share a pass
 */
static inline int arith_reduce_op_promote(arith_reduce_op a, arith_reduce_op b)
{
    if (a == b)
    {
        return (int) a;
    }
    int a_basic =
        (a == ARITH_RED_SUM || a == ARITH_RED_MEAN || a == ARITH_RED_MIN || a == ARITH_RED_MAX ||
         a == ARITH_RED_STD || a == ARITH_RED_VAR || a == ARITH_RED_ALL);
    int b_basic =
        (b == ARITH_RED_SUM || b == ARITH_RED_MEAN || b == ARITH_RED_MIN || b == ARITH_RED_MAX ||
         b == ARITH_RED_STD || b == ARITH_RED_VAR || b == ARITH_RED_ALL);
    if (!a_basic || !b_basic)
    {
        return -1;
    }
    if ((a == ARITH_RED_SUM || a == ARITH_RED_MEAN) && (b == ARITH_RED_SUM || b == ARITH_RED_MEAN))
    {
        return ARITH_RED_MEAN;
    }
    if ((a == ARITH_RED_SUM || a == ARITH_RED_MEAN || a == ARITH_RED_STD || a == ARITH_RED_VAR) &&
        (b == ARITH_RED_SUM || b == ARITH_RED_MEAN || b == ARITH_RED_STD || b == ARITH_RED_VAR))
    {
        return ARITH_RED_STD;
    }
    return ARITH_RED_ALL;
}

/**
 * arith_isnan_f - Test whether an IEEE-754 `float` is NaN using bit inspection
 * @a: Value to inspect
 *
 * Remains accurate when compiled under `-Ofast` / `-ffinite-math-only`.
 *
 * Return: 1 if @a is NaN, 0 otherwise
 */
static inline int arith_isnan_f(float a)
{
    uint32_t u;
    __builtin_memcpy(&u, &a, sizeof(u));
    return (u & 0x7fffffffU) > 0x7f800000U;
}

/**
 * arith_isnan_d - Test whether an IEEE-754 `double` is NaN using bit inspection
 * @a: Value to inspect
 *
 * Remains accurate when compiled under `-Ofast` / `-ffinite-math-only`.
 *
 * Return: 1 if @a is NaN, 0 otherwise
 */
static inline int arith_isnan_d(double a)
{
    uint64_t u;
    __builtin_memcpy(&u, &a, sizeof(u));
    return (u & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
}

/**
 * arith_isfinite_f - Test whether an IEEE-754 `float` is finite using bit inspection
 * @a: Value to inspect
 *
 * Return: 1 if @a is finite, 0 if infinite or NaN
 */
static inline int arith_isfinite_f(float a)
{
    uint32_t u;
    __builtin_memcpy(&u, &a, sizeof(u));
    return (u & 0x7f800000U) != 0x7f800000U;
}

/**
 * arith_isfinite_d - Test whether an IEEE-754 `double` is finite using bit inspection
 * @a: Value to inspect
 *
 * Return: 1 if @a is finite, 0 if infinite or NaN
 */
static inline int arith_isfinite_d(double a)
{
    uint64_t u;
    __builtin_memcpy(&u, &a, sizeof(u));
    return (u & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

/** Block and thread-local accumulator for single-pass Chan statistical reductions. */
typedef struct
{
    uint64_t n;
    double   sum;
    double   mean;
    double   m2;
    double   min;
    double   max;
} arith_stat_acc_t;

/**
 * arith_stat_merge_inline - Merge block or thread accumulator @b into @a using Chan's recurrence
 * @a: Destination accumulator updated in place
 * @b: Source accumulator to merge
 *
 * Return: void
 */
static inline void arith_stat_merge_inline(arith_stat_acc_t *a, const arith_stat_acc_t *b)
{
    if (b->n == 0)
    {
        return;
    }
    if (a->n == 0)
    {
        *a = *b;
        return;
    }
    uint64_t n = a->n + b->n;
    double   d = b->mean - a->mean;
    a->sum += b->sum;
    a->mean += d * ((double) b->n / (double) n);
    a->m2 += b->m2 + d * d * ((double) a->n * (double) b->n / (double) n);
    if (b->min < a->min || arith_isnan_d(a->min))
    {
        a->min = b->min;
    }
    if (b->max > a->max || arith_isnan_d(a->max))
    {
        a->max = b->max;
    }
    a->n = n;
}

/**
 * arith_stat_finish_inline - Finalize accumulator @acc into the scalar result of reduction @op
 * @acc:     Combined statistical accumulator
 * @op:      Reduction operation (`ARITH_RED_SUM`..`ARITH_RED_DOT`)
 * @is_long: Non-zero to truncate the final result to `long` (`ARITH_RED_COUNT`)
 *
 * Return: Reduced scalar value as `double`
 */
static inline double arith_stat_finish_inline(const arith_stat_acc_t *acc,
                                              arith_reduce_op         op,
                                              int                     is_long)
{
    if (acc->n == 0)
    {
        return is_long ? 0.0
                       : ((op == ARITH_RED_SUM || op == ARITH_RED_ASUM || op == ARITH_RED_NORM ||
                           op == ARITH_RED_DOT)
                              ? 0.0
                              : NAN);
    }
    double r = 0.0;
    switch (op)
    {
    case ARITH_RED_SUM:
    case ARITH_RED_ASUM:
    case ARITH_RED_DOT:
        r = acc->sum;
        break;
    case ARITH_RED_MEAN:
        r = acc->sum / (double) acc->n;
        break;
    case ARITH_RED_MIN:
        r = acc->min;
        break;
    case ARITH_RED_MAX:
        r = acc->max;
        break;
    case ARITH_RED_VAR:
        r = acc->m2 / (double) acc->n;
        break;
    case ARITH_RED_STD:
        r = sqrt(acc->m2 / (double) acc->n);
        break;
    case ARITH_RED_NORM:
        r = sqrt(acc->sum);
        break;
    case ARITH_RED_COUNT:
    default:
        r = (double) acc->n;
        break;
    }
    return is_long ? (double) (long) r : r;
}

/* clang-format off */
#define ARITH_SIMD_ISTATS \
    _Pragma("omp simd reduction(|:has_nan) reduction(+:s) reduction(min:bmin) reduction(max:bmax)")
/* clang-format on */

#define ARITH_DEFINE_STAT_BLOCK_INLINE(FN_NAME, FT, ISNAN_FN)                                     \
    static inline void FN_NAME(arith_reduce_op op, const FT *restrict v, const FT *restrict m,    \
                               int len, arith_stat_acc_t *acc)                                    \
    {                                                                                             \
        if (op == ARITH_RED_DOT)                                                                  \
        {                                                                                         \
            FT s = (FT) 0;                                                                        \
            _Pragma("omp simd reduction(+:s)") for (int j = 0; j < len; j++)                      \
            {                                                                                     \
                s += v[j] * m[j];                                                                 \
            }                                                                                     \
            acc->sum += (double) s;                                                               \
            acc->n += (uint64_t) len;                                                             \
            return;                                                                               \
        }                                                                                         \
        if (m != NULL)                                                                            \
        {                                                                                         \
            uint64_t nb = 0;                                                                      \
            double   sb = 0.0;                                                                    \
            double   mn = (acc->n == 0) ? NAN : acc->min;                                         \
            double   mx = (acc->n == 0) ? NAN : acc->max;                                         \
            if (op == ARITH_RED_MEAN || op == ARITH_RED_SUM || op == ARITH_RED_STD ||             \
                op == ARITH_RED_VAR)                                                              \
            {                                                                                     \
                _Pragma("omp simd reduction(+:sb,nb)") for (int j = 0; j < len; j++)              \
                {                                                                                 \
                    int on = (m[j] != (FT) 0);                                                    \
                    sb += on ? (double) v[j] : 0.0;                                               \
                    nb += (uint64_t) on;                                                          \
                }                                                                                 \
                if (nb == 0)                                                                      \
                {                                                                                 \
                    return;                                                                       \
                }                                                                                 \
                double bm = sb / (double) nb;                                                     \
                double m2 = 0.0;                                                                  \
                if (op == ARITH_RED_STD || op == ARITH_RED_VAR)                                   \
                {                                                                                 \
                    _Pragma("omp simd reduction(+:m2)") for (int j = 0; j < len; j++)             \
                    {                                                                             \
                        double d = (m[j] != (FT) 0) ? ((double) v[j] - bm) : 0.0;                 \
                        m2 += d * d;                                                              \
                    }                                                                             \
                }                                                                                 \
                arith_stat_acc_t blk = { nb, sb, bm, m2, mn, mx };                                \
                arith_stat_merge_inline(acc, &blk);                                               \
                return;                                                                           \
            }                                                                                     \
            if (op == ARITH_RED_NORM || op == ARITH_RED_ASUM)                                     \
            {                                                                                     \
                _Pragma("omp simd reduction(+:sb,nb)") for (int j = 0; j < len; j++)              \
                {                                                                                 \
                    int    on = (m[j] != (FT) 0);                                                 \
                    double x  = on ? (double) v[j] : 0.0;                                         \
                    sb += (op == ARITH_RED_ASUM) ? fabs(x) : (x * x);                             \
                    nb += (uint64_t) on;                                                          \
                }                                                                                 \
                acc->sum += sb;                                                                   \
                acc->n += nb;                                                                     \
                return;                                                                           \
            }                                                                                     \
            for (int j = 0; j < len; j++)                                                         \
            {                                                                                     \
                if (m[j] != (FT) 0)                                                               \
                {                                                                                 \
                    double x = (double) v[j];                                                     \
                    sb += x;                                                                      \
                    mn = (x < mn || arith_isnan_d(mn)) ? x : mn;                                  \
                    mx = (x > mx || arith_isnan_d(mx)) ? x : mx;                                  \
                    nb++;                                                                         \
                }                                                                                 \
            }                                                                                     \
            if (nb == 0)                                                                          \
            {                                                                                     \
                return;                                                                           \
            }                                                                                     \
            double bm = sb / (double) nb;                                                         \
            double m2 = 0.0;                                                                      \
            if (op == ARITH_RED_ALL)                                                              \
            {                                                                                     \
                _Pragma("omp simd reduction(+:m2)") for (int j = 0; j < len; j++)                 \
                {                                                                                 \
                    double d = (m[j] != (FT) 0) ? ((double) v[j] - bm) : 0.0;                     \
                    m2 += d * d;                                                                  \
                }                                                                                 \
            }                                                                                     \
            arith_stat_acc_t blk = { nb, sb, bm, m2, mn, mx };                                    \
            arith_stat_merge_inline(acc, &blk);                                                   \
            return;                                                                               \
        }                                                                                         \
        switch (op)                                                                               \
        {                                                                                         \
        case ARITH_RED_ALL:                                                                       \
        {                                                                                         \
            if (len <= 0)                                                                         \
            {                                                                                     \
                break;                                                                            \
            }                                                                                     \
            int    has_nan = 0;                                                                   \
            double s       = 0.0;                                                                 \
            FT     bmin    = v[0];                                                                \
            FT     bmax    = v[0];                                                                \
            ARITH_SIMD_ISTATS for (int j = 0; j < len; j++)                                       \
            {                                                                                     \
                FT x = v[j];                                                                      \
                has_nan |= ISNAN_FN(x);                                                           \
                s += (double) x;                                                                  \
                bmin = (x < bmin) ? x : bmin;                                                     \
                bmax = (x > bmax) ? x : bmax;                                                     \
            }                                                                                     \
            double mn  = (double) bmin;                                                           \
            double mx  = (double) bmax;                                                           \
            int    set = !has_nan;                                                                \
            if (UNLIKELY(has_nan))                                                                \
            {                                                                                     \
                s   = 0.0;                                                                        \
                mn  = 0.0;                                                                        \
                mx  = 0.0;                                                                        \
                set = 0;                                                                          \
                for (int j = 0; j < len; j++)                                                     \
                {                                                                                 \
                    FT     xv = v[j];                                                             \
                    double x  = (double) xv;                                                      \
                    s += x;                                                                       \
                    if (!ISNAN_FN(xv))                                                            \
                    {                                                                             \
                        if (!set || x < mn)                                                       \
                        {                                                                         \
                            mn = x;                                                               \
                        }                                                                         \
                        if (!set || x > mx)                                                       \
                        {                                                                         \
                            mx = x;                                                               \
                        }                                                                         \
                        set = 1;                                                                  \
                    }                                                                             \
                }                                                                                 \
            }                                                                                     \
            double bm = s / (double) len;                                                         \
            double m2 = 0.0;                                                                      \
            _Pragma("omp simd reduction(+:m2)") for (int j = 0; j < len; j++)                     \
            {                                                                                     \
                double d = (double) v[j] - bm;                                                    \
                m2 += d * d;                                                                      \
            }                                                                                     \
            arith_stat_acc_t blk = { (uint64_t) len, s, bm, m2, set ? mn : NAN, set ? mx : NAN }; \
            arith_stat_merge_inline(acc, &blk);                                                   \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_SUM:                                                                       \
        case ARITH_RED_MEAN:                                                                      \
        {                                                                                         \
            double s = 0.0;                                                                       \
            _Pragma("omp simd reduction(+:s)") for (int j = 0; j < len; j++)                      \
            {                                                                                     \
                s += (double) v[j];                                                               \
            }                                                                                     \
            acc->sum += s;                                                                        \
            acc->n += (uint64_t) len;                                                             \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_ASUM:                                                                      \
        {                                                                                         \
            double s = 0.0;                                                                       \
            _Pragma("omp simd reduction(+:s)") for (int j = 0; j < len; j++)                      \
            {                                                                                     \
                s += fabs((double) v[j]);                                                         \
            }                                                                                     \
            acc->sum += s;                                                                        \
            acc->n += (uint64_t) len;                                                             \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_MIN:                                                                       \
        {                                                                                         \
            if (len <= 0)                                                                         \
            {                                                                                     \
                break;                                                                            \
            }                                                                                     \
            int has_nan = 0;                                                                      \
            FT  bmin    = v[0];                                                                   \
            _Pragma("omp simd reduction(|:has_nan) reduction(min:bmin)") for (int j = 0; j < len; \
                                                                              j++)                \
            {                                                                                     \
                FT x = v[j];                                                                      \
                has_nan |= ISNAN_FN(x);                                                           \
                bmin = (x < bmin) ? x : bmin;                                                     \
            }                                                                                     \
            double mn = (acc->n == 0) ? NAN : acc->min;                                           \
            if (LIKELY(!has_nan))                                                                 \
            {                                                                                     \
                double x = (double) bmin;                                                         \
                mn       = (x < mn || arith_isnan_d(mn)) ? x : mn;                                \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                for (int j = 0; j < len; j++)                                                     \
                {                                                                                 \
                    double x = (double) v[j];                                                     \
                    mn       = (x < mn || arith_isnan_d(mn)) ? x : mn;                            \
                }                                                                                 \
            }                                                                                     \
            acc->min = mn;                                                                        \
            acc->n += (uint64_t) len;                                                             \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_MAX:                                                                       \
        {                                                                                         \
            if (len <= 0)                                                                         \
            {                                                                                     \
                break;                                                                            \
            }                                                                                     \
            int has_nan = 0;                                                                      \
            FT  bmax    = v[0];                                                                   \
            _Pragma("omp simd reduction(|:has_nan) reduction(max:bmax)") for (int j = 0; j < len; \
                                                                              j++)                \
            {                                                                                     \
                FT x = v[j];                                                                      \
                has_nan |= ISNAN_FN(x);                                                           \
                bmax = (x > bmax) ? x : bmax;                                                     \
            }                                                                                     \
            double mx = (acc->n == 0) ? NAN : acc->max;                                           \
            if (LIKELY(!has_nan))                                                                 \
            {                                                                                     \
                double x = (double) bmax;                                                         \
                mx       = (x > mx || arith_isnan_d(mx)) ? x : mx;                                \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                for (int j = 0; j < len; j++)                                                     \
                {                                                                                 \
                    double x = (double) v[j];                                                     \
                    mx       = (x > mx || arith_isnan_d(mx)) ? x : mx;                            \
                }                                                                                 \
            }                                                                                     \
            acc->max = mx;                                                                        \
            acc->n += (uint64_t) len;                                                             \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_STD:                                                                       \
        case ARITH_RED_VAR:                                                                       \
        {                                                                                         \
            double s = 0.0;                                                                       \
            _Pragma("omp simd reduction(+:s)") for (int j = 0; j < len; j++)                      \
            {                                                                                     \
                s += (double) v[j];                                                               \
            }                                                                                     \
            double bm = s / (double) len;                                                         \
            double m2 = 0.0;                                                                      \
            _Pragma("omp simd reduction(+:m2)") for (int j = 0; j < len; j++)                     \
            {                                                                                     \
                double d = (double) v[j] - bm;                                                    \
                m2 += d * d;                                                                      \
            }                                                                                     \
            arith_stat_acc_t blk = { (uint64_t) len, s, bm, m2, acc->min, acc->max };             \
            arith_stat_merge_inline(acc, &blk);                                                   \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_NORM:                                                                      \
        {                                                                                         \
            double s2 = 0.0;                                                                      \
            _Pragma("omp simd reduction(+:s2)") for (int j = 0; j < len; j++)                     \
            {                                                                                     \
                double x = (double) v[j];                                                         \
                s2 += x * x;                                                                      \
            }                                                                                     \
            acc->sum += s2;                                                                       \
            acc->n += (uint64_t) len;                                                             \
            break;                                                                                \
        }                                                                                         \
        case ARITH_RED_COUNT:                                                                     \
        default:                                                                                  \
        {                                                                                         \
            uint64_t c = 0;                                                                       \
            _Pragma("omp simd reduction(+:c)") for (int j = 0; j < len; j++)                      \
            {                                                                                     \
                c += (v[j] != (FT) 0);                                                            \
            }                                                                                     \
            acc->n += c;                                                                          \
            break;                                                                                \
        }                                                                                         \
        }                                                                                         \
    }

ARITH_DEFINE_STAT_BLOCK_INLINE(arith_stat_block_f_inline, float, arith_isnan_f)
ARITH_DEFINE_STAT_BLOCK_INLINE(arith_stat_block_d_inline, double, arith_isnan_d)
#undef ARITH_DEFINE_STAT_BLOCK_INLINE
#undef ARITH_SIMD_ISTATS

/**
 * arith_read_im_pix - Read linear element @idx of real image @im as `double`
 * @im:  Image structure
 * @dt:  ImageStreamIO datatype code (`_DATATYPE_*`)
 * @idx: Linear pixel index
 *
 * Return: Pixel value converted to `double`, or `0.0` for unsupported types.
 */
static inline double arith_read_im_pix(const IMAGE *im, uint8_t dt, uint64_t idx)
{
    switch (dt)
    {
    case _DATATYPE_FLOAT:
        return (double) im->array.F[idx];
    case _DATATYPE_DOUBLE:
        return im->array.D[idx];
    case _DATATYPE_UINT8:
        return (double) im->array.UI8[idx];
    case _DATATYPE_INT8:
        return (double) im->array.SI8[idx];
    case _DATATYPE_UINT16:
        return (double) im->array.UI16[idx];
    case _DATATYPE_INT16:
        return (double) im->array.SI16[idx];
    case _DATATYPE_UINT32:
        return (double) im->array.UI32[idx];
    case _DATATYPE_INT32:
        return (double) im->array.SI32[idx];
    case _DATATYPE_UINT64:
        return (double) im->array.UI64[idx];
    case _DATATYPE_INT64:
        return (double) im->array.SI64[idx];
    default:
        return 0.0;
    }
}

double arith_image_mean(const char *ID_name);
double arith_image_mean_IMGID(IMGID *imgin);

double arith_image_min(const char *ID_name);
double arith_image_min_IMGID(IMGID *imgin);

double arith_image_max(const char *ID_name);
double arith_image_max_IMGID(IMGID *imgin);

double arith_image_percentile(const char *ID_name, double fraction);
double arith_image_percentile_IMGID(IMGID *imgin, double fraction);

double arith_image_median(const char *ID_name);
double arith_image_median_IMGID(IMGID *imgin);

double arith_image_dot(const char *ID1_name, const char *ID2_name);
double arith_image_dot_IMGID(IMGID *imgin1, IMGID *imgin2);

double arith_image_norm(const char *ID_name);
double arith_image_norm_IMGID(IMGID *imgin);

double arith_image_rms(const char *ID_name);
double arith_image_rms_IMGID(IMGID *imgin);

double arith_image_var(const char *ID_name);
double arith_image_var_IMGID(IMGID *imgin);

double arith_image_count(const char *ID_name);
double arith_image_count_IMGID(IMGID *imgin);

/* argmax/argmin/iamax: linear pixel index (x + y*xsize + ...), -1 if every pixel is NaN */
double arith_image_argmax(const char *ID_name);
double arith_image_argmax_IMGID(IMGID *imgin);

double arith_image_argmin(const char *ID_name);
double arith_image_argmin_IMGID(IMGID *imgin);

double arith_image_iamax(const char *ID_name);
double arith_image_iamax_IMGID(IMGID *imgin);

double arith_image_asum(const char *ID_name);
double arith_image_asum_IMGID(IMGID *imgin);

double arith_image_trace(const char *ID_name);
double arith_image_trace_IMGID(IMGID *imgin);

/** Reduce @n strided elements of @im starting at @base with @stride using @op. */
double arith_image_reduce_lane(const IMAGE    *im,
                               uint8_t         dt,
                               uint64_t        base,
                               uint64_t        n,
                               uint64_t        stride,
                               arith_reduce_op op);

/** Reduce @im along @axis into pre-allocated `FLOAT` or `DOUBLE` image @oim. */
errno_t arith_image_reduce_axis_im(const IMAGE *im, long axis, arith_reduce_op op, IMAGE *oim);

/** Reduce pixels of @im where mask @mk is non-zero using @op (@frac for `ARITH_RED_PERC`). */
double arith_image_reduce_masked_im(const IMAGE    *im,
                                    const IMAGE    *mk,
                                    arith_reduce_op op,
                                    double          frac);

#endif // COREMOD_ARITH_IMAGE_STATS_H

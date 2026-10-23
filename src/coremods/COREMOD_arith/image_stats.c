// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_stats.c
 * @brief   Whole-image, axis-wise, and masked statistical reductions
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "image_stats.h"
#include "image_total.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 131072
#endif

/** Accumulate one sample @v into @acc for reduction @op. */
static inline double reduce_step(double acc, double v, arith_reduce_op op)
{
    return (op == ARITH_RED_MIN)     ? fmin(acc, v)
           : (op == ARITH_RED_MAX)   ? fmax(acc, v)
           : (op == ARITH_RED_ASUM)  ? acc + fabs(v)
           : (op == ARITH_RED_NORM)  ? acc + v * v
           : (op == ARITH_RED_COUNT) ? acc + (v != 0.0)
                                     : acc + v;
}

/**
 * arith_image_reduce_lane - Reduce @n strided elements of @im starting at @base
 * @im:     Source image
 * @dt:     ImageStreamIO datatype code
 * @base:   Starting linear element offset
 * @n:      Number of elements along the lane
 * @stride: Step between consecutive elements
 * @op:     Reduction operation (`ARITH_RED_SUM`..`ARITH_RED_COUNT`)
 *
 * Return: Reduced scalar value as `double`
 */
double arith_image_reduce_lane(const IMAGE    *im,
                               uint8_t         dt,
                               uint64_t        base,
                               uint64_t        n,
                               uint64_t        stride,
                               arith_reduce_op op)
{
    if (n == 0)
    {
        return 0.0;
    }

    double acc = (op == ARITH_RED_MIN) ? INFINITY : (op == ARITH_RED_MAX) ? -INFINITY : 0.0;
#define LANE_PASS1(READ_K)                    \
    for (uint64_t k = 0; k < n; k++)          \
    {                                         \
        acc = reduce_step(acc, (READ_K), op); \
    }
    if (dt == _DATATYPE_FLOAT)
    {
        const float *restrict p = im->array.F + base;
        LANE_PASS1((double) p[k * stride]);
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        const double *restrict p = im->array.D + base;
        LANE_PASS1(p[k * stride]);
    }
    else
    {
        LANE_PASS1(arith_read_im_pix(im, dt, base + k * stride));
    }
#undef LANE_PASS1

    if (op == ARITH_RED_SUM || op == ARITH_RED_MIN || op == ARITH_RED_MAX || op == ARITH_RED_ASUM ||
        op == ARITH_RED_COUNT)
    {
        return acc;
    }
    if (op == ARITH_RED_NORM)
    {
        return sqrt(acc);
    }
    double mean = acc / (double) n;
    if (op == ARITH_RED_MEAN)
    {
        return mean;
    }
    double s2 = 0.0;
    for (uint64_t k = 0; k < n; k++)
    {
        double d = arith_read_im_pix(im, dt, base + k * stride) - mean;
        s2 += d * d;
    }
    return (op == ARITH_RED_VAR) ? s2 / (double) n : sqrt(s2 / (double) n);
}

/** Resolve @imgin and reduce all elements using @op, returning @err_ret if unresolved. */
static double reduce_all_imgid(IMGID *imgin, arith_reduce_op op, double err_ret)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return err_ret;
    }
    return arith_image_reduce_lane(imgin->im, imgin->md[0].datatype, 0, imgin->md[0].nelement, 1,
                                   op);
}

double MILK_HOT arith_image_total_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_SUM, RETURN_FAILURE);
}

double arith_image_total(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_total_IMGID(&imgin);
}

double MILK_HOT arith_image_sumsquare_IMGID(IMGID *imgin)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }
    return arith_image_dot_IMGID(imgin, imgin);
}

double arith_image_sumsquare(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_sumsquare_IMGID(&imgin);
}

double arith_image_mean_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_MEAN, RETURN_FAILURE);
}

double arith_image_mean(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_mean_IMGID(&imgin);
}

double MILK_HOT arith_image_min_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_MIN, RETURN_FAILURE);
}

double arith_image_min(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_min_IMGID(&imgin);
}

double MILK_HOT arith_image_max_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_MAX, RETURN_FAILURE);
}

double arith_image_max(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_max_IMGID(&imgin);
}

double MILK_HOT arith_image_rms_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_STD, 0.0);
}

double arith_image_rms(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_rms_IMGID(&imgin);
}

double arith_image_var_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_VAR, 0.0);
}

double arith_image_var(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_var_IMGID(&imgin);
}

double MILK_HOT arith_image_count_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_COUNT, 0.0);
}

double arith_image_count(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_count_IMGID(&imgin);
}

double MILK_HOT arith_image_asum_IMGID(IMGID *imgin)
{
    return reduce_all_imgid(imgin, ARITH_RED_ASUM, 0.0);
}

double arith_image_asum(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_asum_IMGID(&imgin);
}

double MILK_HOT arith_image_trace_IMGID(IMGID *imgin)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return 0.0;
    }
    uint64_t M        = imgin->md[0].size[0];
    uint64_t N        = (imgin->md[0].naxis >= 2) ? imgin->md[0].size[1] : 1;
    uint64_t diag_len = (M < N) ? M : N;
    return arith_image_reduce_lane(imgin->im, imgin->md[0].datatype, 0, diag_len, M + 1,
                                   ARITH_RED_SUM);
}

double arith_image_trace(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_trace_IMGID(&imgin);
}

/** Linear index of the first extremum (`@sign == 1` max, `-1` min, `0` max-|x|); -1 if none. */
static double argext(IMGID *imgin, double sign)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return -1.0;
    }

    uint64_t nelement = imgin->md[0].nelement;
    uint8_t  dt       = imgin->md[0].datatype;
    int64_t  best     = -1;
    double   bestv    = 0.0;

    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double raw = arith_read_im_pix(imgin->im, dt, ii);
        double v   = (sign == 0.0) ? fabs(raw) : sign * raw;
        if (v == v && (best < 0 || v > bestv))
        {
            bestv = v;
            best  = (int64_t) ii;
        }
    }
    return (double) best;
}

double arith_image_argmax_IMGID(IMGID *imgin)
{
    return argext(imgin, 1.0);
}

double arith_image_argmax(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return argext(&imgin, 1.0);
}

double arith_image_argmin_IMGID(IMGID *imgin)
{
    return argext(imgin, -1.0);
}

double arith_image_argmin(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return argext(&imgin, -1.0);
}

double MILK_HOT arith_image_iamax_IMGID(IMGID *imgin)
{
    return argext(imgin, 0.0);
}

double arith_image_iamax(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return argext(&imgin, 0.0);
}

double arith_image_percentile_IMGID(IMGID *imgin, double fraction)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint64_t nelement = imgin->md[0].nelement;
    uint8_t  dt       = imgin->md[0].datatype;
    if (nelement == 0)
    {
        return 0.0;
    }

    long idx = (long) (fraction * (double) nelement);
    if (idx < 0)
    {
        idx = 0;
    }
    else if ((uint64_t) idx >= nelement)
    {
        idx = (long) nelement - 1;
    }

    if (dt == _DATATYPE_FLOAT)
    {
        float *buf = (float *) malloc(sizeof(float) * nelement);
        if (buf == NULL)
        {
            return RETURN_FAILURE;
        }
        memcpy(buf, imgin->im->array.F, sizeof(float) * nelement);
        quick_sort_float(buf, nelement);
        double value = (double) buf[idx];
        free(buf);
        return value;
    }

    double *buf = (double *) malloc(sizeof(double) * nelement);
    if (buf == NULL)
    {
        return RETURN_FAILURE;
    }
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        buf[ii] = arith_read_im_pix(imgin->im, dt, ii);
    }
    quick_sort_double(buf, nelement);
    double value = buf[idx];
    free(buf);
    return value;
}

double arith_image_percentile(const char *ID_name, double fraction)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_percentile_IMGID(&imgin, fraction);
}

double arith_image_median_IMGID(IMGID *imgin)
{
    return arith_image_percentile_IMGID(imgin, 0.5);
}

double arith_image_median(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_median_IMGID(&imgin);
}

double MILK_HOT arith_image_dot_IMGID(IMGID *imgin1, IMGID *imgin2)
{
    resolveIMGID(imgin1, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(imgin2, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin1->ID == -1 || imgin2->ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint8_t  dt1      = imgin1->md[0].datatype;
    uint8_t  dt2      = imgin2->md[0].datatype;
    uint64_t nelement = imgin1->md[0].nelement;
    if (dt1 != dt2 || nelement != imgin2->md[0].nelement)
    {
        return 0.0;
    }

    double value = 0.0;
    if (dt1 == _DATATYPE_FLOAT)
    {
        const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.F);
        const float *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.F);
#ifdef _OPENMP
#    pragma omp parallel for simd reduction(+ : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            value += (double) p1[ii] * (double) p2[ii];
        }
        return value;
    }
    if (dt1 == _DATATYPE_DOUBLE)
    {
        const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.D);
        const double *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.D);
#ifdef _OPENMP
#    pragma omp parallel for simd reduction(+ : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            value += p1[ii] * p2[ii];
        }
        return value;
    }

    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        value += arith_read_im_pix(imgin1->im, dt1, ii) * arith_read_im_pix(imgin2->im, dt2, ii);
    }
    return value;
}

double arith_image_dot(const char *ID1_name, const char *ID2_name)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    return arith_image_dot_IMGID(&imgin1, &imgin2);
}

double MILK_HOT arith_image_norm_IMGID(IMGID *imgin)
{
    return sqrt(arith_image_dot_IMGID(imgin, imgin));
}

double arith_image_norm(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_norm_IMGID(&imgin);
}

errno_t arith_image_reduce_axis_im(const IMAGE *im, long axis, arith_reduce_op op, IMAGE *oim)
{
    int     nax = im->md[0].naxis;
    uint8_t dt  = im->md[0].datatype;
    if (axis < 0 || axis >= nax)
    {
        return RETURN_FAILURE;
    }

    uint64_t inner = 1;
    uint64_t outer = 1;
    uint64_t n     = im->md[0].size[axis];
    for (int a = 0; a < nax; a++)
    {
        if (a < axis)
        {
            inner *= im->md[0].size[a];
        }
        else if (a > axis)
        {
            outer *= im->md[0].size[a];
        }
    }

    int      out_d = (oim->md[0].datatype == _DATATYPE_DOUBLE);
    uint64_t total = outer * inner;

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (total * n > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t idx = 0; idx < total; idx++)
    {
        uint64_t o = idx / inner;
        uint64_t i = idx % inner;
        double   r = arith_image_reduce_lane(im, dt, o * n * inner + i, n, inner, op);
        if (out_d)
        {
            oim->array.D[idx] = r;
        }
        else
        {
            oim->array.F[idx] = (float) r;
        }
    }
    return RETURN_SUCCESS;
}

double arith_image_reduce_masked_im(const IMAGE    *im,
                                    const IMAGE    *mk,
                                    arith_reduce_op op,
                                    double          frac)
{
    uint64_t nel = im->md[0].nelement;
    uint8_t  idt = im->md[0].datatype;
    uint8_t  mdt = mk->md[0].datatype;
    uint64_t n   = 0;

    if (op == ARITH_RED_PERC)
    {
        double *v = (double *) malloc(sizeof(double) * nel);
        if (v == NULL)
        {
            return NAN;
        }
        for (uint64_t i = 0; i < nel; i++)
        {
            if (arith_read_im_pix(mk, mdt, i) != 0.0)
            {
                v[n++] = arith_read_im_pix(im, idt, i);
            }
        }
        double r = NAN;
        if (n > 0)
        {
            long k = (long) (frac * (double) n);
            quick_sort_double(v, n);
            r = v[(k < 0) ? 0 : ((uint64_t) k >= n) ? (long) n - 1 : k];
        }
        free(v);
        return r;
    }

    double acc = (op == ARITH_RED_MIN) ? INFINITY : (op == ARITH_RED_MAX) ? -INFINITY : 0.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        if (arith_read_im_pix(mk, mdt, i) != 0.0)
        {
            acc = reduce_step(acc, arith_read_im_pix(im, idt, i), op);
            n++;
        }
    }
    if (n == 0)
    {
        return (op == ARITH_RED_SUM || op == ARITH_RED_ASUM || op == ARITH_RED_NORM) ? 0.0 : NAN;
    }
    if (op == ARITH_RED_SUM || op == ARITH_RED_MIN || op == ARITH_RED_MAX || op == ARITH_RED_ASUM)
    {
        return acc;
    }
    if (op == ARITH_RED_NORM)
    {
        return sqrt(acc);
    }
    double mean = acc / (double) n;
    if (op == ARITH_RED_MEAN)
    {
        return mean;
    }
    double s2 = 0.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        if (arith_read_im_pix(mk, mdt, i) != 0.0)
        {
            double d = arith_read_im_pix(im, idt, i) - mean;
            s2 += d * d;
        }
    }
    return (op == ARITH_RED_VAR) ? s2 / (double) n : sqrt(s2 / (double) n);
}

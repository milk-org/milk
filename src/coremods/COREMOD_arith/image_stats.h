// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_stats.h
 * @brief   Whole-image, axis-wise, and masked statistical reductions
 */

#ifndef COREMOD_ARITH_IMAGE_STATS_H
#define COREMOD_ARITH_IMAGE_STATS_H

#include <stdint.h>
#include <libfps/IMGID.h>

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
    ARITH_RED_PERC
} arith_reduce_op;

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

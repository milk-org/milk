// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_stats.c
 * @brief   simple stats functions
 *
 *
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif
#include <math.h>
#include "image_stats.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_tools/COREMOD_tools.h"
#include "libmilkdata/milk_type_dispatch.h"

#include "image_total.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 131072
#    define MILK_PRAGMA_OMP_MIN \
        _Pragma("omp parallel for simd reduction(min:value) if (nelement > OMP_NELEMENT_LIMIT)")
#    define MILK_PRAGMA_OMP_MAX \
        _Pragma("omp parallel for simd reduction(max:value) if (nelement > OMP_NELEMENT_LIMIT)")
#    define MILK_PRAGMA_OMP_SUM2 \
        _Pragma("omp parallel for simd reduction(+:sum2) if (nelement > OMP_NELEMENT_LIMIT)")
#    define MILK_PRAGMA_OMP_COUNT \
        _Pragma("omp parallel for simd reduction(+:n) if (nelement > OMP_NELEMENT_LIMIT)")
#else
#    define MILK_PRAGMA_OMP_MIN
#    define MILK_PRAGMA_OMP_MAX
#    define MILK_PRAGMA_OMP_SUM2
#    define MILK_PRAGMA_OMP_COUNT
#endif

double arith_image_mean_IMGID(IMGID *imgin)
{
    double value;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);

    value = (double) (arith_image_total_IMGID(imgin) / imgin->md[0].nelement);
    if (imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }

    return (value);
}

double arith_image_mean(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_mean_IMGID(&imgin);
}

double MILK_HOT arith_image_min_IMGID(IMGID *imgin)
{
    uint64_t nelement;
    uint8_t  datatype;
    int      OK = 0;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;

#define MIN_BODY(MBR)                                            \
    {                                                            \
        __typeof__(imgin->im->array.MBR[0]) *MILK_RESTRICT ptr = \
            MILK_ASSUME_ALIGNED(imgin->im->array.MBR);           \
        __typeof__(ptr[0]) value = ptr[0];                       \
        MILK_PRAGMA_OMP_MIN                                      \
        for (uint64_t ii = 0; ii < nelement; ii++)               \
        {                                                        \
            if (ptr[ii] < value)                                 \
            {                                                    \
                value = ptr[ii];                                 \
            }                                                    \
        }                                                        \
        OK = 1;                                                  \
        return ((double) value);                                 \
    }

    MILK_FOR_EACH_DATATYPE(datatype, MIN_BODY, MIN_BODY(D))
#undef MIN_BODY

    if (OK == 0)
    {
        PRINT_ERROR("invalid data type");
        return 0.0;
    }

    return 0.0;
}

double arith_image_min(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_min_IMGID(&imgin);
}

double MILK_HOT arith_image_max_IMGID(IMGID *imgin)
{
    uint32_t datatype;
    uint64_t nelement;
    int      OK = 0;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;

#define MAX_BODY(MBR)                                            \
    {                                                            \
        __typeof__(imgin->im->array.MBR[0]) *MILK_RESTRICT ptr = \
            MILK_ASSUME_ALIGNED(imgin->im->array.MBR);           \
        __typeof__(ptr[0]) value = ptr[0];                       \
        MILK_PRAGMA_OMP_MAX                                      \
        for (uint64_t ii = 0; ii < nelement; ii++)               \
        {                                                        \
            if (ptr[ii] > value)                                 \
            {                                                    \
                value = ptr[ii];                                 \
            }                                                    \
        }                                                        \
        OK = 1;                                                  \
        return ((double) value);                                 \
    }

    MILK_FOR_EACH_DATATYPE(datatype, MAX_BODY, MAX_BODY(D))
#undef MAX_BODY

    if (OK == 0)
    {
        printf("Error : Invalid data format for arith_image_max\n");
    }

    return (0);
}

double arith_image_max(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_max_IMGID(&imgin);
}

double arith_image_percentile_IMGID(IMGID *imgin, double fraction)
{
    double          value  = 0;
    long           *arrayL = NULL;
    float          *arrayF = NULL;
    double         *arrayD = NULL;
    unsigned short *arrayU = NULL;
    uint64_t        nelement;
    uint8_t         datatype;
    int             atypeOK = 1;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;


    void *array_raw = malloc(ImageStreamIO_typesize(datatype) * nelement);
    if (array_raw == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(EXIT_FAILURE);
    }
    memcpy(array_raw, imgin->im->array.raw, ImageStreamIO_typesize(datatype) * nelement);


    switch (datatype)
    {
    case _DATATYPE_FLOAT:
        arrayF = array_raw;
        quick_sort_float(arrayF, nelement);
        value = (double) arrayF[(long) (fraction * nelement)];
        break;

    case _DATATYPE_DOUBLE:
        arrayD = array_raw;
        quick_sort_double(arrayD, nelement);
        value = arrayD[(long) (fraction * nelement)];
        break;

    case _DATATYPE_UINT8:
        arrayU = (unsigned short *) malloc(sizeof(unsigned short) * nelement);
        if (arrayU == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayU[ii] = imgin->im->array.UI8[ii];
        }
        quick_sort_ushort(arrayU, nelement);
        value = arrayU[(long) (fraction * nelement)];
        free(arrayU);
        break;

    case _DATATYPE_UINT16:
        arrayU = (unsigned short *) malloc(sizeof(unsigned short) * nelement);
        if (arrayU == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayU[ii] = imgin->im->array.UI16[ii];
        }
        quick_sort_ushort(arrayU, nelement);
        value = arrayU[(long) (fraction * nelement)];
        free(arrayU);
        break;

    case _DATATYPE_UINT32:
        arrayL = (long *) malloc(sizeof(long) * nelement);
        if (arrayL == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayL[ii] = imgin->im->array.UI32[ii];
        }
        quick_sort_long(arrayL, nelement);
        value = arrayL[(long) (fraction * nelement)];
        free(arrayL);
        break;

    case _DATATYPE_UINT64:
        arrayL = (long *) malloc(sizeof(long) * nelement);
        if (arrayL == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayL[ii] = imgin->im->array.UI64[ii];
        }
        quick_sort_long(arrayL, nelement);
        value = arrayL[(long) (fraction * nelement)];
        free(arrayL);
        break;

    case _DATATYPE_INT8:
        arrayL = (long *) malloc(sizeof(long) * nelement);
        if (arrayL == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayL[ii] = (long) imgin->im->array.SI8[ii];
        }
        quick_sort_long(arrayL, nelement);
        value = (double) arrayL[(long) (fraction * nelement)];
        free(arrayL);
        break;

    case _DATATYPE_INT16:
        arrayL = (long *) malloc(sizeof(long) * nelement);
        if (arrayL == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayL[ii] = (long) imgin->im->array.SI16[ii];
        }
        quick_sort_long(arrayL, nelement);
        value = (double) arrayL[(long) (fraction * nelement)];
        free(arrayL);
        break;

    case _DATATYPE_INT32:
        arrayL = (long *) malloc(sizeof(long) * nelement);
        if (arrayL == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayL[ii] = (long) imgin->im->array.SI32[ii];
        }
        quick_sort_long(arrayL, nelement);
        value = (double) arrayL[(long) (fraction * nelement)];
        free(arrayL);
        break;

    case _DATATYPE_INT64:
        arrayL = (long *) malloc(sizeof(long) * nelement);
        if (arrayL == NULL)
        {
            PRINT_ERROR("malloc() error");
            exit(EXIT_FAILURE);
        }
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            arrayL[ii] = (long) imgin->im->array.SI64[ii];
        }
        quick_sort_long(arrayL, nelement);
        value = (double) arrayL[(long) (fraction * nelement)];
        free(arrayL);
        break;

    default:
        PRINT_ERROR("Image type not supported");
        atypeOK = 0;
        break;
    }

    if (atypeOK == 0)
    {
        exit(EXIT_FAILURE);
    }

    free(array_raw);

    return (value);
}

/**
 * @brief Compute a percentile value from an image.
 *
 * Sorts pixel values and returns the value at
 * the specified fractional position.
 */
double arith_image_percentile(const char *ID_name, double fraction)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_percentile_IMGID(&imgin, fraction);
}

double arith_image_median_IMGID(IMGID *imgin)
{
    double value = 0.0;

    value = arith_image_percentile_IMGID(imgin, 0.5);

    return (value);
}

/**
 * @brief Compute the median pixel value of an image.
 */
double arith_image_median(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_median_IMGID(&imgin);
}

double MILK_HOT arith_image_dot_IMGID(IMGID *imgin1, IMGID *imgin2)
{
    uint64_t nelement;
    uint8_t  datatype1, datatype2;
    double   value = 0.0;
    int      OK    = 0;

    resolveIMGID(imgin1, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(imgin2, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin1->ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (imgin2->ID == -1)
    {
        return RETURN_FAILURE;
    }

    datatype1 = imgin1->md[0].datatype;
    datatype2 = imgin2->md[0].datatype;
    nelement  = imgin1->md[0].nelement;

    if (datatype1 != datatype2 || nelement != imgin2->md[0].nelement)
    {
        printf("Error: Incompatible sizes or types for arith_image_dot\n");
        return 0.0;
    }

    if (datatype1 == _DATATYPE_FLOAT)
    {
        float *MILK_RESTRICT ptr1 = MILK_ASSUME_ALIGNED(imgin1->im->array.F);
        float *MILK_RESTRICT ptr2 = MILK_ASSUME_ALIGNED(imgin2->im->array.F);
#ifdef _OPENMP
#    pragma omp parallel for simd reduction(+ : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            value += (double) ptr1[ii] * (double) ptr2[ii];
        }
        OK = 1;
    }
    else if (datatype1 == _DATATYPE_DOUBLE)
    {
        double *MILK_RESTRICT ptr1 = MILK_ASSUME_ALIGNED(imgin1->im->array.D);
        double *MILK_RESTRICT ptr2 = MILK_ASSUME_ALIGNED(imgin2->im->array.D);
#ifdef _OPENMP
#    pragma omp parallel for simd reduction(+ : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            value += (double) ptr1[ii] * (double) ptr2[ii];
        }
        OK = 1;
    }
    else
    {
        /* Fallback for other datatypes using floatcast (not optimal but handles rare types) */
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            float v1, v2;
            switch (datatype1)
            {
            case _DATATYPE_UINT8:
                v1 = (float) imgin1->im->array.UI8[ii];
                v2 = (float) imgin2->im->array.UI8[ii];
                break;
            case _DATATYPE_UINT16:
                v1 = (float) imgin1->im->array.UI16[ii];
                v2 = (float) imgin2->im->array.UI16[ii];
                break;
            case _DATATYPE_UINT32:
                v1 = (float) imgin1->im->array.UI32[ii];
                v2 = (float) imgin2->im->array.UI32[ii];
                break;
            case _DATATYPE_UINT64:
                v1 = (float) imgin1->im->array.UI64[ii];
                v2 = (float) imgin2->im->array.UI64[ii];
                break;
            case _DATATYPE_INT8:
                v1 = (float) imgin1->im->array.SI8[ii];
                v2 = (float) imgin2->im->array.SI8[ii];
                break;
            case _DATATYPE_INT16:
                v1 = (float) imgin1->im->array.SI16[ii];
                v2 = (float) imgin2->im->array.SI16[ii];
                break;
            case _DATATYPE_INT32:
                v1 = (float) imgin1->im->array.SI32[ii];
                v2 = (float) imgin2->im->array.SI32[ii];
                break;
            case _DATATYPE_INT64:
                v1 = (float) imgin1->im->array.SI64[ii];
                v2 = (float) imgin2->im->array.SI64[ii];
                break;
            default:
                v1 = 0;
                v2 = 0;
                break;
            }
            value += (double) v1 * (double) v2;
        }
        OK = 1;
    }

    if (OK == 0)
    {
        printf("Error: Invalid data format for arith_image_dot\n");
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

/**
 * @brief Compute the RMS deviation around the mean of an image.
 *
 * Returns sqrt(mean((I - mean(I))^2)) across all pixels in @p imgin.
 *
 * @param[in,out] imgin  Target image descriptor.
 * @return RMS standard deviation across pixels, or 0.0 on error.
 */
double MILK_HOT arith_image_rms_IMGID(IMGID *imgin)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return 0.0;
    }

    uint64_t nelement = imgin->md[0].nelement;
    if (nelement == 0)
    {
        return 0.0;
    }

    double  mean     = arith_image_mean_IMGID(imgin);
    double  sum2     = 0.0;
    uint8_t datatype = imgin->md[0].datatype;

#define RMS_BODY(MBR)                                            \
    {                                                            \
        __typeof__(imgin->im->array.MBR[0]) *MILK_RESTRICT ptr = \
            MILK_ASSUME_ALIGNED(imgin->im->array.MBR);           \
        MILK_PRAGMA_OMP_SUM2                                     \
        for (uint64_t ii = 0; ii < nelement; ii++)               \
        {                                                        \
            double d = (double) ptr[ii] - mean;                  \
            sum2 += d * d;                                       \
        }                                                        \
    }

    MILK_FOR_EACH_DATATYPE(datatype, RMS_BODY, RMS_BODY(D))
#undef RMS_BODY

    return sqrt(sum2 / (double) nelement);
}

/**
 * @brief Compute the RMS deviation around the mean of a named image.
 *
 * @param[in] ID_name  Image name (with optional slice/modifiers).
 * @return RMS standard deviation across pixels.
 */
double arith_image_rms(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_rms_IMGID(&imgin);
}

/** arith_image_var_IMGID() - population variance of the pixels (square of arith_image_rms_IMGID()). */
double arith_image_var_IMGID(IMGID *imgin)
{
    double s = arith_image_rms_IMGID(imgin);
    return s * s;
}

double arith_image_var(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_var_IMGID(&imgin);
}

/** arith_image_count_IMGID() - number of non-zero pixels (e.g. of a mask). */
double MILK_HOT arith_image_count_IMGID(IMGID *imgin)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return 0.0;
    }

    uint64_t nelement = imgin->md[0].nelement;
    uint64_t n        = 0;
    uint8_t  datatype = imgin->md[0].datatype;

#define COUNT_BODY(MBR)                                          \
    {                                                            \
        __typeof__(imgin->im->array.MBR[0]) *MILK_RESTRICT ptr = \
            MILK_ASSUME_ALIGNED(imgin->im->array.MBR);           \
        MILK_PRAGMA_OMP_COUNT                                    \
        for (uint64_t ii = 0; ii < nelement; ii++)               \
        {                                                        \
            n += (ptr[ii] != 0);                                 \
        }                                                        \
    }

    MILK_FOR_EACH_DATATYPE(datatype, COUNT_BODY, COUNT_BODY(D))
#undef COUNT_BODY

    return (double) n;
}

double arith_image_count(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_count_IMGID(&imgin);
}

/** argext() - linear index of the first largest (@sign 1) or smallest (@sign -1) pixel, NaNs skipped; -1 if none. */
static double argext(IMGID *imgin, double sign)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin->ID == -1)
    {
        return -1.0;
    }

    uint64_t nelement = imgin->md[0].nelement;
    uint8_t  datatype = imgin->md[0].datatype;
    int64_t  best     = -1;
    double   bestv    = 0.0;

#define ARGEXT_BODY(MBR)                                         \
    {                                                            \
        __typeof__(imgin->im->array.MBR[0]) *MILK_RESTRICT ptr = \
            MILK_ASSUME_ALIGNED(imgin->im->array.MBR);           \
        for (uint64_t ii = 0; ii < nelement; ii++)               \
        {                                                        \
            double v = sign * (double) ptr[ii];                  \
            if (v == v && (best < 0 || v > bestv))               \
            {                                                    \
                bestv = v;                                       \
                best  = (int64_t) ii;                            \
            }                                                    \
        }                                                        \
    }

    MILK_FOR_EACH_DATATYPE(datatype, ARGEXT_BODY, ARGEXT_BODY(D))
#undef ARGEXT_BODY

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

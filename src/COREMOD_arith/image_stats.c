// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_stats.c
 * @brief   simple stats functions
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_stats.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_tools/COREMOD_tools.h"

#include "image_total.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 1000000
#endif

double arith_image_mean_IMGID(IMGID *imgin)
{
    double value;

    resolveIMGID(imgin, ERRMODE_ABORT);

    value = (double) (arith_image_total_IMGID(imgin) / imgin->md[0].nelement);

    return (value);
}

double arith_image_mean(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_mean_IMGID(&imgin);
}

double arith_image_min_IMGID(IMGID *imgin)
{
    uint64_t nelement;
    uint8_t  datatype;
    int      OK = 0;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

    nelement = imgin->md[0].nelement;

    if (datatype == _DATATYPE_FLOAT)
    {
        float *ptr   = imgin->im->array.F;
        float  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_DOUBLE)
    {
        double *ptr   = imgin->im->array.D;
        double  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return (value);
    }

    if (datatype == _DATATYPE_UINT8)
    {
        uint8_t *ptr   = imgin->im->array.UI8;
        uint8_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_UINT16)
    {
        uint16_t *ptr   = imgin->im->array.UI16;
        uint16_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_UINT32)
    {
        uint32_t *ptr   = imgin->im->array.UI32;
        uint32_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_UINT64)
    {
        uint64_t *ptr   = imgin->im->array.UI64;
        uint64_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT8)
    {
        int8_t *ptr   = imgin->im->array.SI8;
        int8_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT16)
    {
        int16_t *ptr   = imgin->im->array.SI16;
        int16_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT32)
    {
        int32_t *ptr   = imgin->im->array.SI32;
        int32_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT64)
    {
        int64_t *ptr   = imgin->im->array.SI64;
        int64_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(min : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] < value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (OK == 0)
    {
        printf("Error : Invalid data format for arith_image_min\n");
    }

    return (0);
}

double arith_image_min(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_min_IMGID(&imgin);
}

double arith_image_max_IMGID(IMGID *imgin)
{
    long    nelement;
    uint8_t datatype;
    int     OK = 0;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

    nelement = imgin->md[0].nelement;

    if (datatype == _DATATYPE_FLOAT)
    {
        float *ptr   = imgin->im->array.F;
        float  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_DOUBLE)
    {
        double *ptr   = imgin->im->array.D;
        double  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return (value);
    }

    if (datatype == _DATATYPE_UINT8)
    {
        uint8_t *ptr   = imgin->im->array.UI8;
        uint8_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_UINT16)
    {
        uint16_t *ptr   = imgin->im->array.UI16;
        uint16_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_UINT32)
    {
        uint32_t *ptr   = imgin->im->array.UI32;
        uint32_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_UINT64)
    {
        uint64_t *ptr   = imgin->im->array.UI64;
        uint64_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT8)
    {
        int8_t *ptr   = imgin->im->array.SI8;
        int8_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT16)
    {
        int16_t *ptr   = imgin->im->array.SI16;
        int16_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT32)
    {
        int32_t *ptr   = imgin->im->array.SI32;
        int32_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (datatype == _DATATYPE_INT64)
    {
        int64_t *ptr   = imgin->im->array.SI64;
        int64_t  value = ptr[0];
#ifdef _OPENMP
#    pragma omp parallel for reduction(max : value) if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            if (ptr[ii] > value)
            {
                value = ptr[ii];
            }
        }
        OK = 1;
        return ((double) value);
    }

    if (OK == 0)
    {
        printf("Error : Invalid data format for arith_image_max\n");
    }

    return (0);
}

double arith_image_max(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_max_IMGID(&imgin);
}

double arith_image_percentile_IMGID(IMGID *imgin, double fraction)
{
    long            ii;
    double          value  = 0;
    long           *arrayL = NULL;
    float          *arrayF = NULL;
    double         *arrayD = NULL;
    unsigned short *arrayU = NULL;
    long            nelement;
    uint8_t         datatype;
    int             atypeOK = 1;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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
        for (ii = 0; ii < nelement; ii++)
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

double arith_image_percentile(const char *ID_name, double fraction)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_percentile_IMGID(&imgin, fraction);
}

double arith_image_median_IMGID(IMGID *imgin)
{
    double value = 0.0;

    value = arith_image_percentile_IMGID(imgin, 0.5);

    return (value);
}

double arith_image_median(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_median_IMGID(&imgin);
}

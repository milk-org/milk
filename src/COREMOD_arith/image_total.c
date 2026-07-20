// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_total.c
 * @brief   sum image pixels
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_total.h"

#include "COREMOD_memory/COREMOD_memory.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

double arith_image_total_IMGID(IMGID *imgin)
{
    long double lvalue; // uses long double internally
    uint64_t    nelement;
    uint8_t     datatype;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

#ifdef _OPENMP
#    pragma omp parallel reduction(+ : lvalue) if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

        if (datatype == _DATATYPE_FLOAT)
        {
            float *ptr = imgin->im->array.F;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_DOUBLE)
        {
            double *ptr = imgin->im->array.D;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_UINT8)
        {
            uint8_t *ptr = imgin->im->array.UI8;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_UINT16)
        {
            uint16_t *ptr = imgin->im->array.UI16;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_UINT32)
        {
            uint32_t *ptr = imgin->im->array.UI32;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_UINT64)
        {
            uint64_t *ptr = imgin->im->array.UI64;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_INT8)
        {
            int8_t *ptr = imgin->im->array.SI8;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_INT16)
        {
            int16_t *ptr = imgin->im->array.SI16;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_INT32)
        {
            int32_t *ptr = imgin->im->array.SI32;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else if (datatype == _DATATYPE_INT64)
        {
            int64_t *ptr = imgin->im->array.SI64;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (long double) ptr[ii];
            }
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }

#ifdef _OPENMP
    }
#endif

    double value;
    value = (double) lvalue;

    return (value);
}

double arith_image_total(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_total_IMGID(&imgin);
}

double arith_image_sumsquare_IMGID(IMGID *imgin)
{
    double   lvalue; // uses double internally
    uint64_t nelement;
    uint8_t  datatype;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

#ifdef _OPENMP
#    pragma omp parallel reduction(+ : lvalue) if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

        if (datatype == _DATATYPE_FLOAT)
        {
            float *ptr = imgin->im->array.F;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_DOUBLE)
        {
            double *ptr = imgin->im->array.D;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_UINT8)
        {
            uint8_t *ptr = imgin->im->array.UI8;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_UINT16)
        {
            uint16_t *ptr = imgin->im->array.UI16;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_UINT32)
        {
            uint32_t *ptr = imgin->im->array.UI32;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_UINT64)
        {
            uint64_t *ptr = imgin->im->array.UI64;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_INT8)
        {
            int8_t *ptr = imgin->im->array.SI8;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_INT16)
        {
            int16_t *ptr = imgin->im->array.SI16;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_INT32)
        {
            int32_t *ptr = imgin->im->array.SI32;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else if (datatype == _DATATYPE_INT64)
        {
            int64_t *ptr = imgin->im->array.SI64;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                lvalue += (double) (ptr[ii] * ptr[ii]);
            }
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }

#ifdef _OPENMP
    }
#endif

    double value;
    value = (double) lvalue;

    return (value);
}

double arith_image_sumsquare(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_sumsquare_IMGID(&imgin);
}

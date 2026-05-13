/**
 * @file    image_total.c
 * @brief   sum image pixels
 *
 *
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#endif
#include "image_total.h"

#include <math.h>

#include "COREMOD_memory/COREMOD_memory.h"

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 100000
#endif

double MILK_HOT arith_image_total_IMGID(IMGID *imgin)
{
    long double lvalue; // uses long double internally
    uint64_t    nelement;
    uint8_t     datatype;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1) {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

#ifdef _OPENMP
    #pragma omp parallel reduction(+:lvalue) if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if(datatype == _DATATYPE_FLOAT)
    {
        float * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.F);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        double * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.D);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT8)
    {
        uint8_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI8);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT16)
    {
        uint16_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI16);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT32)
    {
        uint32_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI32);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT64)
    {
        uint64_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI64);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_INT8)
    {
        int8_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI8);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_INT16)
    {
        int16_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI16);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_INT32)
    {
        int32_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI32);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else if(datatype == _DATATYPE_INT64)
    {
        int64_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI64);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) ptr[ii];
        }
    }
    else
    {
        PRINT_ERROR("invalid data type");
        return NAN;
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
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_total_IMGID(&imgin);
}

double MILK_HOT arith_image_sumsquare_IMGID(IMGID *imgin)
{
    double lvalue; // uses double internally
    uint64_t    nelement;
    uint8_t     datatype;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1) {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

#ifdef _OPENMP
    #pragma omp parallel reduction(+:lvalue) if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if(datatype == _DATATYPE_FLOAT)
    {
        float * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.F);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        double * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.D);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT8)
    {
        uint8_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI8);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT16)
    {
        uint16_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI16);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT32)
    {
        uint32_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI32);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT64)
    {
        uint64_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.UI64);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT8)
    {
        int8_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI8);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT16)
    {
        int16_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI16);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT32)
    {
        int32_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI32);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT64)
    {
        int64_t * MILK_RESTRICT ptr =
            MILK_ASSUME_ALIGNED(imgin->im->array.SI64);
#ifdef _OPENMP
        #pragma omp for simd
#endif
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (double)(ptr[ii] * ptr[ii]);
        }
    }
    else
    {
        PRINT_ERROR("invalid data type");
        return NAN;
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
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_sumsquare_IMGID(&imgin);
}

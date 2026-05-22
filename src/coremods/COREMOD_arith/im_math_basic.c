/**
 * @file    im_math_basic.c
 * @brief   Basic binary arithmetic functions (add, sub, mult, div, pow, etc.)
 */

#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "libmilkcommon/milk_compiler.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "imgid_arith_helpers.h"

#include "imfunctions.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

#define ARITH_OPTIMIZED_FUNCTION(name, op)                                                         \
    errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)      \
    {                                                                                              \
        DEBUG_TRACE_FSTART();                                                                      \
        if (imgin1->im == NULL || imgin2->im == NULL)                                              \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        imgid_ensure_output(imgin1, imgout);                                                       \
        uint64_t nelement = imgout->md->nelement;                                                  \
        if (imgin1->md->datatype == _DATATYPE_FLOAT && imgin2->md->datatype == _DATATYPE_FLOAT &&  \
            imgout->mdt->datatype == _DATATYPE_FLOAT)                                              \
        {                                                                                          \
            float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.F);                    \
            float *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.F);                    \
            float *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.F);                    \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++) po[i] =  \
                p1[i] op p2[i];                                                                    \
        }                                                                                          \
        else if (imgin1->md->datatype == _DATATYPE_DOUBLE &&                                       \
                 imgin2->md->datatype == _DATATYPE_DOUBLE &&                                       \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.D);                   \
            double *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.D);                   \
            double *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.D);                   \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++) po[i] =  \
                p1[i] op p2[i];                                                                    \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &P##name);                      \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_OPTIMIZED_FUNCTION(add, +)
ARITH_OPTIMIZED_FUNCTION(sub, -)
ARITH_OPTIMIZED_FUNCTION(mult, *)
ARITH_OPTIMIZED_FUNCTION(div, /)

#define ARITH_CST_OPTIMIZED_FUNCTION(name, op)                                                     \
    errno_t arith_image_cst##name##_optimized_IMGID(IMGID *imgin, double f1, IMGID *imgout)        \
    {                                                                                              \
        DEBUG_TRACE_FSTART();                                                                      \
        if (imgin->im == NULL)                                                                     \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        imgid_ensure_output(imgin, imgout);                                                        \
        uint64_t nelement = imgout->md->nelement;                                                  \
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)    \
        {                                                                                          \
            float *MILK_RESTRICT p1  = MILK_ASSUME_ALIGNED(imgin->im->array.F);                    \
            float *MILK_RESTRICT po  = MILK_ASSUME_ALIGNED(imgout->im->array.F);                   \
            float                cf1 = (float) f1;                                                 \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++) po[i] =  \
                p1[i] op cf1;                                                                      \
        }                                                                                          \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                        \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            double *p1 = imgin->im->array.D;                                                       \
            double *po = imgout->im->array.D;                                                      \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++) po[i] =  \
                p1[i] op f1;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_1f_1_IMGID(imgin, f1, imgout, &P##name);                          \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_CST_OPTIMIZED_FUNCTION(add, +)
ARITH_CST_OPTIMIZED_FUNCTION(sub, -)
ARITH_CST_OPTIMIZED_FUNCTION(mult, *)
ARITH_CST_OPTIMIZED_FUNCTION(div, /)

errno_t arith_image_cstpow_optimized_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    DEBUG_TRACE_FSTART();
    if (imgin->im == NULL)
    {
        return RETURN_FAILURE;
    }
    imgid_ensure_output(imgin, imgout);
    uint64_t nelement = imgout->md->nelement;
    if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)
    {
        float *MILK_RESTRICT p1  = MILK_ASSUME_ALIGNED(imgin->im->array.F);
        float *MILK_RESTRICT po  = MILK_ASSUME_ALIGNED(imgout->im->array.F);
        float                cf1 = (float) f1;
        if (f1 == 0.0)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = 1.0f;
            }
        }
        else if (f1 == 1.0)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = p1[i];
            }
        }
        else if (f1 == 0.5)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = sqrtf(p1[i]);
            }
        }
        else if (f1 == 2.0)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = p1[i] * p1[i];
            }
        }
        else
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = powf(p1[i], cf1);
            }
        }
    }
    else if (imgin->md->datatype == _DATATYPE_DOUBLE && imgout->mdt->datatype == _DATATYPE_DOUBLE)
    {
        double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);
        double *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.D);
        if (f1 == 0.0)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = 1.0;
            }
        }
        else if (f1 == 1.0)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = p1[i];
            }
        }
        else if (f1 == 0.5)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = sqrt(p1[i]);
            }
        }
        else if (f1 == 2.0)
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = p1[i] * p1[i];
            }
        }
        else
        {
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i = 0;
                                                                                     i < nelement;
                                                                                     i++)
            {
                po[i] = pow(p1[i], f1);
            }
        }
    }
    else
    {
        arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ppow);
    }
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#define ARITH_OPTIMIZED_FUNCTION_CALL(name, funcname, funcname_f)                                  \
    errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)      \
    {                                                                                              \
        DEBUG_TRACE_FSTART();                                                                      \
        if (imgin1->im == NULL || imgin2->im == NULL)                                              \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        imgid_ensure_output(imgin1, imgout);                                                       \
        uint64_t nelement = imgout->md->nelement;                                                  \
        if (imgin1->md->datatype == _DATATYPE_FLOAT && imgin2->md->datatype == _DATATYPE_FLOAT &&  \
            imgout->mdt->datatype == _DATATYPE_FLOAT)                                              \
        {                                                                                          \
            float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.F);                    \
            float *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.F);                    \
            float *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.F);                    \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++) po[i] =  \
                funcname_f(p1[i], p2[i]);                                                          \
        }                                                                                          \
        else if (imgin1->md->datatype == _DATATYPE_DOUBLE &&                                       \
                 imgin2->md->datatype == _DATATYPE_DOUBLE &&                                       \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.D);                   \
            double *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.D);                   \
            double *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.D);                   \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++) po[i] =  \
                funcname(p1[i], p2[i]);                                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &P##name);                      \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_OPTIMIZED_FUNCTION_CALL(pow, pow, powf)
ARITH_OPTIMIZED_FUNCTION_CALL(fmod, fmod, fmodf)
ARITH_OPTIMIZED_FUNCTION_CALL(minv, fmin, fminf)
ARITH_OPTIMIZED_FUNCTION_CALL(maxv, fmax, fmaxf)

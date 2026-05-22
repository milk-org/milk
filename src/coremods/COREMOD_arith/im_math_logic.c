/**
 * @file    im_math_logic.c
 * @brief   Logic and comparison math functions
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

/* ---------------------------------------------------------- */
/* Binary optimized: comparison/logic with expression          */
/* ---------------------------------------------------------- */
#define ARITH_OPTIMIZED_FUNCTION_EXPR(name, expr_f, expr_d)                                        \
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
            const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.F);              \
            const float *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.F);              \
            float *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.F);              \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = (expr_f);                                                                  \
            }                                                                                      \
        }                                                                                          \
        else if (imgin1->md->datatype == _DATATYPE_DOUBLE &&                                       \
                 imgin2->md->datatype == _DATATYPE_DOUBLE &&                                       \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.D);             \
            const double *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.D);             \
            double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);             \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = (expr_d);                                                                  \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &P##name);                      \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_OPTIMIZED_FUNCTION_EXPR(testlt, (p1[i] < p2[i]) ? 1.0f : 0.0f, (p1[i] < p2[i]) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(testmt, (p1[i] >= p2[i]) ? 1.0f : 0.0f, (p1[i] >= p2[i]) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(teste, (p1[i] == p2[i]) ? 1.0f : 0.0f, (p1[i] == p2[i]) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(testne, (p1[i] != p2[i]) ? 1.0f : 0.0f, (p1[i] != p2[i]) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(testle, (p1[i] <= p2[i]) ? 1.0f : 0.0f, (p1[i] <= p2[i]) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(testge, (p1[i] >= p2[i]) ? 1.0f : 0.0f, (p1[i] >= p2[i]) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(and,
                              ((p1[i] != 0.0f) && (p2[i] != 0.0f)) ? 1.0f : 0.0f,
                              ((p1[i] != 0.0) && (p2[i] != 0.0)) ? 1.0 : 0.0)
ARITH_OPTIMIZED_FUNCTION_EXPR(or,
                              ((p1[i] != 0.0f) || (p2[i] != 0.0f)) ? 1.0f : 0.0f,
                              ((p1[i] != 0.0) || (p2[i] != 0.0)) ? 1.0 : 0.0)

/* ---------------------------------------------------------- */
/* Const-scalar optimized: comparison/logic with expression    */
/* ---------------------------------------------------------- */
#define ARITH_CST_OPTIMIZED_FUNCTION_EXPR(name, expr_f, expr_d)                                    \
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
            const float *MILK_RESTRICT p1  = MILK_ASSUME_ALIGNED(imgin->im->array.F);              \
            float *MILK_RESTRICT       po  = MILK_ASSUME_ALIGNED(imgout->im->array.F);             \
            float                      cf1 = (float) f1;                                           \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = (expr_f);                                                                  \
            }                                                                                      \
        }                                                                                          \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                        \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);              \
            double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);             \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = (expr_d);                                                                  \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_1f_1_IMGID(imgin, f1, imgout, &P##name);                          \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_CST_OPTIMIZED_FUNCTION_EXPR(testlt, (p1[i] < cf1) ? 1.0f : 0.0f, (p1[i] < f1) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(testmt, (p1[i] >= cf1) ? 1.0f : 0.0f, (p1[i] >= f1) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(teste, (p1[i] == cf1) ? 1.0f : 0.0f, (p1[i] == f1) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(testne, (p1[i] != cf1) ? 1.0f : 0.0f, (p1[i] != f1) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(testle, (p1[i] <= cf1) ? 1.0f : 0.0f, (p1[i] <= f1) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(testge, (p1[i] >= cf1) ? 1.0f : 0.0f, (p1[i] >= f1) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(and,
                                  ((p1[i] != 0.0f) && (cf1 != 0.0f)) ? 1.0f : 0.0f,
                                  ((p1[i] != 0.0) && (f1 != 0.0)) ? 1.0 : 0.0)
ARITH_CST_OPTIMIZED_FUNCTION_EXPR(or,
                                  ((p1[i] != 0.0f) || (cf1 != 0.0f)) ? 1.0f : 0.0f,
                                  ((p1[i] != 0.0) || (f1 != 0.0)) ? 1.0 : 0.0)

/* ---------------------------------------------------------- */
/* Unary optimized: comparison with expression                 */
/* ---------------------------------------------------------- */
#define ARITH_UNARY_OPTIMIZED_FUNCTION_EXPR(name, expr_f, expr_d)                                  \
    errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin, IMGID *imgout)                      \
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
            const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.F);               \
            float *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.F);              \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = (expr_f);                                                                  \
            }                                                                                      \
        }                                                                                          \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                        \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);              \
            double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);             \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = (expr_d);                                                                  \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_1_1_IMGID(imgin, imgout, &P##name);                               \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_UNARY_OPTIMIZED_FUNCTION_EXPR(positive,
                                    (p1[i] > 0.0f) ? 1.0f : 0.0f,
                                    (p1[i] > 0.0) ? 1.0 : 0.0)

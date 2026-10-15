// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_mk_reim_from_complex.c
 * @brief   complex -> re, im
 *
 * Uses FPS V2 framework.
 */


#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]    = "imc";
static char outreimname[FUNCTION_PARAMETER_STRMAXLEN] = "imre";
static char outimimname[FUNCTION_PARAMETER_STRMAXLEN] = "imim";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                            \
    X(".imre_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input complex image") \
    X(".imim_name", outreimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output real image")    \
    X(".out_name", outimimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output imaginary image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t mk_reim_from_complex_IMGID(IMGID *imgin, IMGID *imgre, IMGID *imgim)
{
    DEBUG_TRACE_FSTART();

    uint8_t datatype;

    resolveIMGID(imgin, ERRMODE_ABORT, dcimg, dcnimg);
    datatype      = imgin->md[0].datatype;
    uint8_t naxis = imgin->md[0].naxis;
    for (int i = 0; i < naxis; i++)
    {
        imgre->mdt->size[i] = imgin->md[0].size[i];
        imgim->mdt->size[i] = imgin->md[0].size[i];
    }
    imgre->mdt->naxis = naxis;
    imgim->mdt->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

#define MK_REIM_LOOP(DTYPE_OUT, TYPE_IN, TYPE_OUT, UNION_IN, UNION_OUT)                   \
    {                                                                                     \
        imgre->mdt->datatype = DTYPE_OUT;                                                 \
        if (imgre->ID == -1)                                                              \
            createimagefromIMGID(imgre);                                                  \
        imgim->mdt->datatype = DTYPE_OUT;                                                 \
        if (imgim->ID == -1)                                                              \
            createimagefromIMGID(imgim);                                                  \
        imgre->md[0].write             = 1;                                               \
        imgim->md[0].write             = 1;                                               \
        TYPE_IN *MILK_RESTRICT  ptr_in = MILK_ASSUME_ALIGNED(imgin->im->array.UNION_IN);  \
        TYPE_OUT *MILK_RESTRICT ptr_re = MILK_ASSUME_ALIGNED(imgre->im->array.UNION_OUT); \
        TYPE_OUT *MILK_RESTRICT ptr_im = MILK_ASSUME_ALIGNED(imgim->im->array.UNION_OUT); \
        _Pragma("omp parallel if (nelement > OMP_NELEMENT_LIMIT)")                        \
        {                                                                                 \
            _Pragma("omp for simd") for (uint64_t ii = 0; ii < nelement; ii++)            \
            {                                                                             \
                ptr_re[ii] = ptr_in[ii].re;                                               \
                ptr_im[ii] = ptr_in[ii].im;                                               \
            }                                                                             \
        }                                                                                 \
        if (imgre->md[0].shared == 1)                                                     \
            COREMOD_MEMORY_image_set_sempost_byID(imgre->ID, -1);                         \
        if (imgim->md[0].shared == 1)                                                     \
            COREMOD_MEMORY_image_set_sempost_byID(imgim->ID, -1);                         \
        imgre->md[0].cnt0++;                                                              \
        imgim->md[0].cnt0++;                                                              \
        imgre->md[0].write = 0;                                                           \
        imgim->md[0].write = 0;                                                           \
    }

    if (datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        MK_REIM_LOOP(_DATATYPE_FLOAT, complex_float, float, CF, F)
    }
    else if (datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        MK_REIM_LOOP(_DATATYPE_DOUBLE, complex_double, double, CD, D)
    }

#undef MK_REIM_LOOP
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        return RETURN_FAILURE;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_complex(const char *in_name,
                             const char *re_name,
                             const char *im_name,
                             int         sharedmem)
{
    IMGID imgin       = imgid_make_from_name(in_name);
    IMGID imgre       = imgid_make_from_name(re_name);
    IMGID imgim       = imgid_make_from_name(im_name);
    imgre.mdt->shared = sharedmem;
    imgim.mdt->shared = sharedmem;

    errno_t ret = mk_reim_from_complex_IMGID(&imgin, &imgre, &imgim);
    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);
    return ret;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_mk_reim_from_complex, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_mk_reim_from_complex_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(inimname);
    IMGID imgre = imgid_make_from_name(outreimname);
    IMGID imgim = imgid_make_from_name(outimimname);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    mk_reim_from_complex_IMGID(&imgin, &imgre, &imgim);

    MILK_PROCINFO_END(self)
    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_image_mk_reim_from_complex = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "c2ri",
              .cmdkey           = "c2ri",
              .description      = "complex -> re, im",
              .description_long = "Decompose a complex image into its real and imaginary "
                                  "components. Input is a complex-valued stream; outputs are two "
                                  "real-valued streams." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_mk_reim_from_complex),
    .compute = image_mk_reim_from_complex_compute,
};

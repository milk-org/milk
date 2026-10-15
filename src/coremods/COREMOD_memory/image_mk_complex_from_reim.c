// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_mk_complex_from_reim.c
 * @brief   real, imaginary -> complex
 *
 * Uses FPS V2 framework.
 */


#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inreimname[FUNCTION_PARAMETER_STRMAXLEN] = "imre";
static char inimimname[FUNCTION_PARAMETER_STRMAXLEN] = "imim";
static char outimname[FUNCTION_PARAMETER_STRMAXLEN]  = "imc";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                          \
    X(".imre_name", inreimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "real image")      \
    X(".imim_name", inimimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "imaginary image") \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output complex image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t mk_complex_from_reim_IMGID(IMGID *imgre, IMGID *imgim, IMGID *imgout)
{
    DEBUG_TRACE_FSTART();

    uint8_t datatype_re;
    uint8_t datatype_im;
    uint8_t datatype_out;

    resolveIMGID(imgre, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(imgim, ERRMODE_ABORT, dcimg, dcnimg);

    datatype_re = imgre->md[0].datatype;
    datatype_im = imgim->md[0].datatype;

    imgout->mdt->naxis = imgre->md[0].naxis;
    for (int8_t i = 0; i < imgout->mdt->naxis; i++)
    {
        imgout->mdt->size[i] = imgre->md[0].size[i];
    }
    uint64_t nelement = imgre->md[0].nelement;

#define MK_COMPLEX_LOOP(DTYPE_OUT, TYPE_RE, TYPE_IM, TYPE_OUT, UNION_RE, UNION_IM, UNION_OUT) \
    {                                                                                         \
        datatype_out          = DTYPE_OUT;                                                    \
        imgout->mdt->datatype = datatype_out;                                                 \
        createimagefromIMGID(imgout);                                                         \
        TYPE_RE *MILK_RESTRICT  ptr_re  = MILK_ASSUME_ALIGNED(imgre->im->array.UNION_RE);     \
        TYPE_IM *MILK_RESTRICT  ptr_im  = MILK_ASSUME_ALIGNED(imgim->im->array.UNION_IM);     \
        TYPE_OUT *MILK_RESTRICT ptr_out = MILK_ASSUME_ALIGNED(imgout->im->array.UNION_OUT);   \
        _Pragma("omp parallel if (nelement > OMP_NELEMENT_LIMIT)")                            \
        {                                                                                     \
            _Pragma("omp for simd") for (uint64_t ii = 0; ii < nelement; ii++)                \
            {                                                                                 \
                ptr_out[ii].re = ptr_re[ii];                                                  \
                ptr_out[ii].im = ptr_im[ii];                                                  \
            }                                                                                 \
        }                                                                                     \
    }

    if ((datatype_re == _DATATYPE_FLOAT) && (datatype_im == _DATATYPE_FLOAT))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_FLOAT, float, float, complex_float, F, F, CF)
    }
    else if ((datatype_re == _DATATYPE_FLOAT) && (datatype_im == _DATATYPE_DOUBLE))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_DOUBLE, float, double, complex_double, F, D, CD)
    }
    else if ((datatype_re == _DATATYPE_DOUBLE) && (datatype_im == _DATATYPE_FLOAT))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_DOUBLE, double, float, complex_double, D, F, CD)
    }
    else if ((datatype_re == _DATATYPE_DOUBLE) && (datatype_im == _DATATYPE_DOUBLE))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_DOUBLE, double, double, complex_double, D, D, CD)
    }

#undef MK_COMPLEX_LOOP
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        abort();
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_complex_from_reim(const char *re_name,
                             const char *im_name,
                             const char *out_name,
                             int         sharedmem)
{
    IMGID imgre        = imgid_make_from_name(re_name);
    IMGID imgim        = imgid_make_from_name(im_name);
    IMGID imgout       = imgid_make_from_name(out_name);
    imgout.mdt->shared = sharedmem;

    errno_t ret = mk_complex_from_reim_IMGID(&imgre, &imgim, &imgout);
    imgid_free(&imgre);
    imgid_free(&imgim);
    imgid_free(&imgout);
    return ret;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_mk_complex_from_reim, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_mk_complex_from_reim_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imgre  = imgid_make_from_name(inreimname);
    IMGID imgim  = imgid_make_from_name(inimimname);
    IMGID imgout = imgid_make_from_name(outimname);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    mk_complex_from_reim_IMGID(&imgre, &imgim, &imgout);

    MILK_PROCINFO_END(self)
    imgid_free(&imgre);
    imgid_free(&imgim);
    imgid_free(&imgout);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_image_mk_complex_from_reim = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "ri2c",
                 .cmdkey           = "ri2c",
                 .description      = "real, imaginary -> complex",
                 .description_long = "Construct a complex image from separate real and imaginary "
                                     "component images. Combines two real-valued streams into one "
                                     "complex stream." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_mk_complex_from_reim),
    .compute = image_mk_complex_from_reim_compute,
};

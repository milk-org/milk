// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_to_vec.c
 * @brief Image to vec module
 */

#include "milk_runnable.h"
#include "linopt_imtools.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]      = "";
static char inpixiname[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char inpixmultname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char outvecname[FUNCTION_PARAMETER_STRMAXLEN]    = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                               \
    X(".inim", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".inpixi", inpixiname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "input pixel index image")                                                    \
    X(".inpixmult", inpixmultname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,      \
      "input pixel mult image")                                                     \
    X(".outvec", outvecname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output vector image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_to_vec, FPS_PARAMS);


/**
 * Remap image to vector using pixel
 * index and multiplier tables.
 */
errno_t linopt_imtools_image_to_vec(const char *__restrict ID_name,
                                    const char *__restrict IDpixindex_name,
                                    const char *__restrict IDpixmult_name,
                                    const char *__restrict IDvec_name,
                                    imageID *outID)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s %s %s %s", ID_name, IDpixindex_name, IDpixmult_name, IDvec_name);

    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long    naxisin  = imgin.md->naxis;
    uint8_t datatype = imgin.md->datatype;

    IMGID imgpixi = imgid_make_from_name(IDpixindex_name);
    resolveIMGID(&imgpixi, ERRMODE_WARN, dcimg, dcnimg);
    if (imgpixi.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgpixm = imgid_make_from_name(IDpixmult_name);
    resolveIMGID(&imgpixm, ERRMODE_WARN, dcimg, dcnimg);
    if (imgpixm.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long NBpix = imgpixi.md->nelement;

    IMGID imgvec;

    if (naxisin < 3)
    {
        imgvec             = imgid_make_from_name_2D(IDvec_name, NBpix, 1);
        imgvec.mdt->shared = 0;
        imgid_create_registered(&imgvec);

        const float *restrict inptr  = imgin.im->array.F;
        const float *restrict mptr   = imgpixm.im->array.F;
        const int64_t *restrict pixi = imgpixi.im->array.SI64;
        float *restrict vecptr       = imgvec.im->array.F;
        int nth                      = milk_omp_team_size((uint64_t) NBpix, 262144);

#pragma omp parallel for simd num_threads(nth) if (parallel : nth > 1)
        for (long k = 0; k < NBpix; k++)
        {
            vecptr[k] = mptr[k] * inptr[pixi[k]];
        }
    }
    else
    {
        long     sizexy = (long) imgin.md->size[0] * (long) imgin.md->size[1];
        uint32_t zsize  = imgin.md->size[2];

        if (datatype == _DATATYPE_FLOAT)
        {
            imgvec             = imgid_make_from_name_2D(IDvec_name, NBpix, zsize);
            imgvec.mdt->shared = 0;
            imgid_create_registered(&imgvec);

            const float *restrict inptr  = imgin.im->array.F;
            const float *restrict mptr   = imgpixm.im->array.F;
            const int64_t *restrict pixi = imgpixi.im->array.SI64;
            float *restrict vecptr       = imgvec.im->array.F;
            int nth = milk_omp_team_size((uint64_t) zsize * (uint64_t) NBpix, 262144);

#pragma omp parallel for num_threads(nth) if (nth > 1)
            for (uint32_t kk = 0; kk < zsize; kk++)
            {
                const float *restrict slice = inptr + (uint64_t) kk * (uint64_t) sizexy;
                float *restrict vout        = vecptr + (uint64_t) kk * (uint64_t) NBpix;
#pragma omp simd
                for (long k = 0; k < NBpix; k++)
                {
                    vout[k] = mptr[k] * slice[pixi[k]];
                }
            }
        }
        else if (datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            imgvec             = imgid_make_from_name_2D(IDvec_name, NBpix * 2, zsize);
            imgvec.mdt->shared = 0;
            imgid_create_registered(&imgvec);

            const complex_float *restrict inptr = imgin.im->array.CF;
            const float *restrict mptr          = imgpixm.im->array.F;
            const int64_t *restrict pixi        = imgpixi.im->array.SI64;
            float *restrict vecptr              = imgvec.im->array.F;
            int nth = milk_omp_team_size((uint64_t) zsize * (uint64_t) NBpix * 2, 262144);

#pragma omp parallel for num_threads(nth) if (nth > 1)
            for (uint32_t kk = 0; kk < zsize; kk++)
            {
                const complex_float *restrict slice = inptr + (uint64_t) kk * (uint64_t) sizexy;
                float *restrict vout                = vecptr + (uint64_t) kk * (uint64_t) NBpix * 2;
                for (long k = 0; k < NBpix; k++)
                {
                    long  idx       = pixi[k];
                    float mult      = mptr[k];
                    vout[2 * k]     = mult * slice[idx].re;
                    vout[2 * k + 1] = mult * slice[idx].im;
                }
            }
        }
    }

    if (outID != NULL)
    {
        *outID = imgvec.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t image_to_vec_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    linopt_imtools_image_to_vec(inimname, inpixiname, inpixmultname, outvecname, NULL);

    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_linopt_imtools_image_to_vec = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "im2vec",
                 .cmdkey           = "im2vec",
                 .description      = "remap image to vector",
                 .description_long = "Remap a 2D image to a 1D vector using a pixel table "
                                     "(mask). Extracts active pixels into a compact array." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_to_vec),
    .compute = image_to_vec_compute,
};
EXPORT_FPSEXEC(milk, "linopt-img2vec", runnable_linopt_imtools_image_to_vec);

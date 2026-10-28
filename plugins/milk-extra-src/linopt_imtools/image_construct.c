// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_construct.c
 * @brief Image construct module
 */


#include "milk_runnable.h"
#include "linopt_imtools.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "milk_blas_lapacke.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char modesimname[FUNCTION_PARAMETER_STRMAXLEN];
static char invecname[FUNCTION_PARAMETER_STRMAXLEN];
static char outimname[FUNCTION_PARAMETER_STRMAXLEN];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".modes", modesimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "modes image cube") \
    X(".invec", invecname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input vector")       \
    X(".outim", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_construct, FPS_PARAMS);


/**
 * Construct an image as linear sum of
 * modes weighted by coefficients.
 */
errno_t linopt_imtools_image_construct(const char *IDmodes_name,
                                       const char *IDcoeff_name,
                                       const char *ID_name,
                                       imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    IMGID imgmodes = imgid_make_from_name(IDmodes_name);
    resolveIMGID(&imgmodes, ERRMODE_ABORT, dcimg, dcnimg);

    uint8_t  datatype = imgmodes.md->datatype;
    uint32_t xsize    = imgmodes.md->size[0];
    uint32_t ysize    = imgmodes.md->size[1];
    uint32_t zsize    = imgmodes.md->size[2];
    uint64_t sizexy   = xsize;
    sizexy *= ysize;

    IMGID imgout         = imgid_make_from_name_2D(ID_name, xsize, ysize);
    imgout.mdt->shared   = 0;
    imgout.mdt->datatype = datatype;
    imgid_create_registered(&imgout);

    IMGID imgcoeff = imgid_make_from_name(IDcoeff_name);
    resolveIMGID(&imgcoeff, ERRMODE_ABORT, dcimg, dcnimg);

    if (datatype == _DATATYPE_FLOAT)
    {
        cblas_sgemv(CblasRowMajor, CblasTrans, (int) zsize, (int) sizexy, 1.0f,
                    imgmodes.im->array.F, (int) sizexy, imgcoeff.im->array.F, 1, 0.0f,
                    imgout.im->array.F, 1);
    }
    else
    {
        cblas_dgemv(CblasRowMajor, CblasTrans, (int) zsize, (int) sizexy, 1.0, imgmodes.im->array.D,
                    (int) sizexy, imgcoeff.im->array.D, 1, 0.0, imgout.im->array.D, 1);
    }

    if (outID != NULL)
    {
        *outID = imgout.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_construct_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    linopt_imtools_image_construct(modesimname, invecname, outimname, NULL);
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

const MILK_RUNNABLE runnable_linopt_imtools_image_construct = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "imlinconstruct",
                 .cmdkey      = "imlinconstruct",
                 .description = "construct image as linear sum of modes",
                 .description_long =
                     "Reconstruct an image as a linear combination of mode images. Applies "
                     "coefficient vector to a mode cube to produce the output." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_construct),
    .compute = image_construct_compute,
};
EXPORT_FPSEXEC(milk, "linopt-imgconstruct", runnable_linopt_imtools_image_construct);

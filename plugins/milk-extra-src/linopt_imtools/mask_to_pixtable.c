// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mask_to_pixtable.c
 * @brief Mask to pixtable module
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
static char outpixiimname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char outpixmimname[FUNCTION_PARAMETER_STRMAXLEN] = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                          \
    X(".inim", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")            \
    X(".outpixi", outpixiimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output index image") \
    X(".outpixm", outpixmimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output mask image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(mask_to_pixtable, FPS_PARAMS);


/**
 * Create pixel index and multiplier tables
 * from mask image for vectorized operations.
 */
errno_t linopt_imtools_mask_to_pixtable(const char *IDmask_name,
                                        const char *IDpixindex_name,
                                        const char *IDpixmult_name,
                                        long       *outNBpix)
{
    DEBUG_TRACE_FSTART();

    float eps = 1.0e-8;

    IMGID imgmask = imgid_make_from_name(IDmask_name);
    resolveIMGID(&imgmask, ERRMODE_ABORT, dcimg, dcnimg);

    long size = imgmask.md->nelement;

    long NBpix = 0;
    for (long ii = 0; ii < size; ii++)
    {
        if (imgmask.im->array.F[ii] > eps)
        {
            NBpix++;
        }
    }

    /* Create INT64 index table */
    IMGID imgpixi         = imgid_make_from_name(IDpixindex_name);
    imgpixi.mdt->naxis    = 2;
    imgpixi.mdt->size[0]  = NBpix;
    imgpixi.mdt->size[1]  = 1;
    imgpixi.mdt->datatype = _DATATYPE_INT64;
    imgpixi.mdt->shared   = 0;
    imgpixi.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgpixi);

    /* Create FLOAT multiplier table */
    IMGID imgpixm         = imgid_make_from_name(IDpixmult_name);
    imgpixm.mdt->naxis    = 2;
    imgpixm.mdt->size[0]  = NBpix;
    imgpixm.mdt->size[1]  = 1;
    imgpixm.mdt->datatype = _DATATYPE_FLOAT;
    imgpixm.mdt->shared   = 0;
    imgpixm.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgpixm);

    long k = 0;
    for (long ii = 0; ii < size; ii++)
    {
        if (imgmask.im->array.F[ii] > eps)
        {
            imgpixi.im->array.SI64[k] = ii;
            imgpixm.im->array.F[k]    = imgmask.im->array.F[ii];
            k++;
        }
    }

    if (outNBpix != NULL)
    {
        *outNBpix = NBpix;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t mask_to_pixtable_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    linopt_imtools_mask_to_pixtable(inimname, outpixiimname, outpixmimname, NULL);

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

const MILK_RUNNABLE runnable_linopt_imtools_mask_to_pixtable = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mask2pixtable",
                 .cmdkey      = "mask2pixtable",
                 .description = "make pixel tables from mask",
                 .description_long =
                     "Convert a 2D binary mask into a pixel index table. Lists coordinates of "
                     "active (non-zero) pixels for efficient vectorized access." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mask_to_pixtable),
    .compute = mask_to_pixtable_compute,
};
EXPORT_FPSEXEC(milk, "linopt-mask2pixtab", runnable_linopt_imtools_mask_to_pixtable);

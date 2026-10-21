// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file vec_to_2Dimage.c
 * @brief Vec to 2dimage module
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

static char  imvecname[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char  inpixiname[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char  inpixmultname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char  outimname[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static long *xsizein                                     = NULL;
static long *ysizein                                     = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                         \
    X(".inim", imvecname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input vector")         \
    X(".inpixi", inpixiname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "pixel index image") \
    X(".inpixmult", inpixmultname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,                \
      "input pixel mult image")                                                               \
    X(".outim", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output 2D image")         \
    X(".xsize", &xsizein, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "X size")                    \
    X(".ysize", &ysizein, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "Y size")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(vec_to_2Dimage, FPS_PARAMS);


//
//
//
errno_t linopt_imtools_vec_to_2DImage(const char *IDvec_name,
                                      const char *IDpixindex_name,
                                      const char *IDpixmult_name,
                                      const char *ID_name,
                                      long        xsize,
                                      long        ysize,
                                      imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    IMGID imgvec = imgid_make_from_name(IDvec_name);
    resolveIMGID(&imgvec, ERRMODE_WARN, dcimg, dcnimg);
    if (imgvec.ID == -1)
    {
        return RETURN_FAILURE;
    }

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

    IMGID imgout       = imgid_make_from_name_2D(ID_name, xsize, ysize);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    for (long k = 0; k < NBpix; k++)
    {
        imgout.im->array.F[imgpixi.im->array.SI64[k]] =
            imgvec.im->array.F[k] / imgpixm.im->array.F[k];
    }

    if (outID != NULL)
    {
        *outID = imgout.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t vec_to_2Dimage_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    linopt_imtools_vec_to_2DImage(imvecname, inpixiname, inpixmultname, outimname, *xsizein,
                                  *ysizein, NULL);

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

const MILK_RUNNABLE runnable_linopt_imtools_vec_to_2Dimage = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "vec2im",
                 .cmdkey           = "vec2im",
                 .description      = "remap vector to image",
                 .description_long = "Reconstruct a 2D image from a 1D vector using a pixel table. "
                                     "Inverse of image_to_vec." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(vec_to_2Dimage),
    .compute = vec_to_2Dimage_compute,
};
EXPORT_FPSEXEC(milk, "linopt-vec2Dimg", runnable_linopt_imtools_vec_to_2Dimage);

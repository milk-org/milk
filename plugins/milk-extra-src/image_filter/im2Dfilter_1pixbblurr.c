// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    im2Dfilter_1pixbblurr.c
 * @brief   Apply 1 pixel radius blurr to image
 *
 */

#include "milk_runnable.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *iminname  = NULL;
static char     *imoutname = NULL;
static float    *blurramp  = NULL;
static uint32_t *NBloop    = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                         \
    X(".iminname", &iminname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image name") \
    X(".imoutname", &imoutname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image name")  \
    X(".blurramp", &blurramp, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT,                        \
      "value of side pixs (total = 1 for 3 pix)")                                             \
    X(".axis", &NBloop, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT,                               \
      "number of times operation is performed")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(im2Dfilter_1pixbblurr, FPS_PARAMS);

static errno_t imfilter_im2D_1pixblurr(IMGID imgin, IMGID *imgout, float amp, long NBiter)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);

    if (imgout->ID == -1)
    {
        imgout->mdt->naxis = 2;
        if (imgin.ID == -1)
        {
            return RETURN_FAILURE;
        }
        imgout->mdt->size[0]  = imgin.md->size[0];
        imgout->mdt->size[1]  = imgin.md->size[1];
        imgout->mdt->shared   = imgin.md->shared;
        imgout->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgout);
    }

    if (arith_image_bblur1pix_im(imgin.im, imgout->im, (double) amp, (int) NBiter) != 0)
    {
        return RETURN_FAILURE;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t im2Dfilter_1pixbblurr_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    // input
    IMGID imgin = imgid_make_from_name(iminname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);

    // output
    IMGID imgout = imgid_make_from_name(imoutname);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }


    MILK_PROCINFO_INIT(self)

    MILK_PROCINFO_LOOPSTART(self)
    {
        imfilter_im2D_1pixblurr(imgin, &imgout, *blurramp, *NBloop);

        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&imgin);
    imgid_free(&imgout);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_filter_im2Dfilter_1pixbblurr = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "im2Dfilt1pblurr",
                 .cmdkey      = "im2Dfilt1pblurr",
                 .description = "1 pixel radual blurr, can be iterated",
                 .description_long =
                     "Apply a 1-pixel box blur to a 2D image. Can be iterated multiple times to "
                     "approximate a Gaussian blur via the central limit theorem." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(im2Dfilter_1pixbblurr),
    .compute = im2Dfilter_1pixbblurr_compute,
};
EXPORT_FPSEXEC(milk, "imgfilt-im2Dfilt1pxbb", runnable_image_filter_im2Dfilter_1pixbblurr);

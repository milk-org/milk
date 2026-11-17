// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    gaussfilter.c
 * @brief   Gaussian 2D image filtering
 *
 * Uses FPS V2 framework.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "gaussfilter.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    gaussfilt_inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char    gaussfilt_outimname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static float   gaussfilt_sigma                                   = 0.0;
static int32_t gaussfilt_filtersize                              = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".in_name", gaussfilt_inimname, FPTYPE_STREAMNAME, 1,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "input image")                           \
    X(".out_name", gaussfilt_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "output image")                                                                        \
    X(".sigma", &gaussfilt_sigma, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "gaussian sigma") \
    X(".filter_size", &gaussfilt_filtersize, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT,          \
      "filter box size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void gauss_filter_step(IMAGE  *imgin,
                              IMAGE  *imgout,
                              float   sigma,
                              int     filter_size,
                              void  **scratch_buf,
                              size_t *scratch_cap)
{
    if (imgin == NULL || imgout == NULL || imgin->array.raw == NULL || imgout->array.raw == NULL)
    {
        return;
    }

    uint32_t nx    = imgin->md[0].size[0];
    uint32_t ny    = imgin->md[0].size[1];
    uint32_t nz    = (imgin->md[0].naxis == 3) ? imgin->md[0].size[2] : 1;
    int      fsize = filter_size;
    if (fsize < 0)
    {
        fsize = 0;
    }
    if (fsize > (int) nx / 2 - 1)
    {
        fsize = (int) nx / 2 - 1;
    }
    if (fsize > (int) ny / 2 - 1)
    {
        fsize = (int) ny / 2 - 1;
    }

    if (sigma <= 0.0f || fsize <= 0)
    {
        if (imgin->array.raw != imgout->array.raw)
        {
            size_t byte_copy_size = (size_t) nx * (size_t) ny * (size_t) nz * sizeof(float);
            __builtin_memcpy(imgout->array.raw, imgin->array.raw, byte_copy_size);
        }
        return;
    }

    arith_image_gauss2d_im(imgin, imgout, (double) sigma, (double) sigma, fsize, scratch_buf,
                           scratch_cap);
}


/* =========================================
 * Public convenience function
 * ========================================= */

imageID gauss_filter(const char *ID_name, const char *out_name, float sigma, int filter_size)
{
    IMGID in = imgid_make_from_name(ID_name);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(out_name, in.md->size[0], in.md->size[1]);
    gauss_filter_step(in.im, out.im, sigma, filter_size, NULL, NULL);
    ImageStreamIO_UpdateIm(out.im);
    imageID out_id = out.ID;
    imgid_free(&in);
    imgid_free(&out);
    return out_id;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(gaussfilter, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t gaussfilter_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(gaussfilt_inimname);
    resolveIMGID(&in, ERRMODE_NULL, dcimg, dcnimg);

    if (in.im == NULL)
    {
        imgid_free(&in);
        return RETURN_FAILURE;
    }

    IMGID out = stream_connect_create_2Df32(gaussfilt_outimname, in.md->size[0], in.md->size[1]);

    if (out.im == NULL)
    {
        imgid_free(&in);
        imgid_free(&out);
        return RETURN_FAILURE;
    }

    void  *scratch_buf = NULL;
    size_t scratch_cap = 0;

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    out.md->write = 1;
    gauss_filter_step(in.im, out.im, gaussfilt_sigma, gaussfilt_filtersize, &scratch_buf,
                      &scratch_cap);
    processinfo_update_output_stream(processinfo, out.im, in.im);

    MILK_PROCINFO_END(self)

    free(scratch_buf);
    imgid_free(&in);
    imgid_free(&out);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_filter_gaussfilter = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "gaussfilt",
                 .cmdkey      = "gaussfilt",
                 .description = "gaussian 2D filtering",
                 .description_long =
                     "Apply a 2D Gaussian convolution filter to an image. The kernel width is "
                     "specified as a standard deviation (sigma) in pixels. Implemented via FFT "
                     "for efficiency." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(gaussfilter),
    .compute = gaussfilter_compute,
};
EXPORT_FPSEXEC(milk, "imgfilt-gaussfilt", runnable_image_filter_gaussfilter);

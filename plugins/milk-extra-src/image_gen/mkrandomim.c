// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "ImageStreamIO/ImageStruct.h"
#include "milk_runnable.h"
#include "image_gen.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "statistic/statistic.h" // ran1, gauss, gauss_trc


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     outim_name[FUNCTION_PARAMETER_STRMAXLEN] = "outim";
static uint32_t outim_xsize                              = 256;
static uint32_t outim_ysize                              = 256;
static uint32_t distrib_val                              = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".outim.name", outim_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".outim.xsize", &outim_xsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "x size")        \
    X(".outim.ysize", &outim_ysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "y size")        \
    X(".distrib", &distrib_val, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                      \
      "distribution (0:uniform 1:gauss 2:trunc)")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(mkrandomim, FPS_PARAMS);


/**
 * @brief Make random image
 *
 * @param[out] img
 *      Output image
 *
 * @param[in] pdf
 *      Probability distribution function
 *
 * @return imageID
 */
static imageID make_image_random(IMGID *img, int pdf)
{
    DEBUG_TRACE_FSTART();

    // 0: uniform
    // 1: gauss
    // 2: truncated gauss

    // Create image if needed
    //imcreateIMGID(img);

    if (pdf == 0 || pdf == 1)
    {
        arith_image_fill_coord(img->im, (pdf == 1) ? ARITH_COORD_RANDN : ARITH_COORD_RAND, 0.0,
                               0.0);
    }
    else if (pdf == 2)
    {
        for (uint64_t ii = 0; ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] = (float) gauss_trc();
        }
    }
    else if (pdf == 3) // test pattern
    {
        static uint64_t ii   = 0;
        img->im->array.F[ii] = 1.0 - img->im->array.F[ii];
        ii++;
        if (ii == img->md->nelement)
        {
            ii = 0;
        }
    }

    DEBUG_TRACE_FEXIT();
    return (img->ID);
}

static MILK_HOT errno_t mkrandomim_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("make IMGID for %s", outim_name);

    /*
     * Connect to existing stream or create it.
     * Must be before the loop to avoid leaking
     * img.mdt on every iteration.
     */
    IMGID img = stream_connect_create_2D(outim_name, outim_xsize, outim_ysize, _DATATYPE_FLOAT);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_image_random(&img, distrib_val);

    DEBUG_TRACEPOINT("update output ID %ld", img.ID);
    processinfo_update_output_stream(processinfo, img.im, NULL);
    MILK_PROCINFO_END(self)

    imgid_free(&img);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_mkrandomim = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mkrnd",
                 .cmdkey           = "mkrnd",
                 .description      = "make random image",
                 .description_long = "Generate a random image with Poisson or Gaussian noise for "
                                     "testing and simulation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mkrandomim),
    .compute = mkrandomim_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkrandom", runnable_image_gen_mkrandomim);

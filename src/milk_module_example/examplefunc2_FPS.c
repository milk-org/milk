// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    simplefunc_FPS.c
 * @brief   simple function example with FPS and processinfo support
 *
 * Example 2
 * Demonstrates using FPS to hold function arguments and parameters with the Function Parameter Structure (FPS).
 * See script milk-test-simplefuncFPS for example usage.
 */

#include "milk_runnable.h"
#include "examplefunc2_FPS.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Local variables pointers

static char   *inimname;
static double *scoeff;

#define IMSUM2_PARAMS(X)                                                                \
    X(".in_name", &inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".scaling", &scoeff, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "scaling coefficient")

MILK_PARAMS_DEFINE(imsum2, IMSUM2_PARAMS);


/**
 * @brief Sum pixel values
 *
 */
static errno_t example_compute_2Dimage_total(IMGID *imgptr, double scalingcoeff)
{
    DEBUG_TRACE_FSTART();

    // Ensure the input image is in memory.
    // No harm calling this here and in the upstream function,
    // as the overhead is very small if the image is already resolved
    resolveIMGID(imgptr, ERRMODE_WARN, dcimg, dcnimg);

    uint32_t xsize = imgptr->md->size[0];
    if (imgptr->ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint32_t ysize  = imgptr->md->size[1];
    uint64_t xysize = xsize * ysize;


    double total = 0.0;
    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        total += imgptr->im->array.F[ii];
    }
    total *= scalingcoeff;

    printf("image %s total = %lf (scaling coeff %lf)\n", imgptr->im->name, total, scalingcoeff);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Wrapper function, used by all CLI calls
 *
 * MILK_PROCINFO_* macros enable processinfo support
 */
static MILK_HOT errno_t imsum2_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    // Check that the input image is in memory,
    // and link it to img if it is
    IMGID img = imgid_make_from_name(inimname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        example_compute_2Dimage_total(&img, *scoeff);
        if (img.ID == -1)
        {
            return RETURN_FAILURE;
        }
    }
    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_modex_imsum2 = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "imsum2",
                 .cmdkey      = "imsum2",
                 .description = "compute total of image example2, FPS-compatible" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imsum2),
    .compute = imsum2_compute,
};
EXPORT_FPSEXEC(milk, "imsum2", runnable_modex_imsum2);

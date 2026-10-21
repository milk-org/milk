// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    putphnoise.c
 * @brief   putphnoise command: add photon (Poisson) noise to an image
 */

#include "milk_runnable.h"
#include "statistic.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char phn_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im0";
static char phn_out[FUNCTION_PARAMETER_STRMAXLEN] = "im1";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                \
    X(".in_name", phn_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", phn_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")


/* ================================================================
 * 4.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(putphnoise, FPS_PARAMS);


/* ================================================================
 * 5.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t putphnoise_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    put_poisson_noise(phn_in, phn_out);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 6.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_statistic_putphnoise = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "putphnoise",
                 .cmdkey           = "putphnoise",
                 .description      = "add photon noise to image",
                 .description_long = "Statistical tools for image processing: add Gaussian noise, "
                                     "compute histograms, and generate statistical test images." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(putphnoise),
    .compute = putphnoise_compute,
};

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    putgaussnoise.c
 * @brief   putgaussnoise command: add Gaussian noise to an image
 */

#include "milk_runnable.h"
#include "statistic.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   gn_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im0";
static char   gn_out[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static double gn_ampl                              = 0.2;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                               \
    X(".in_name", gn_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", gn_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".ampl", &gn_ampl, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "noise amplitude")


/* ================================================================
 * 4.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(putgaussnoise, FPS_PARAMS);


/* ================================================================
 * 5.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t putgaussnoise_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    put_gauss_noise(gn_in, gn_out, gn_ampl);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 6.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_statistic_putgaussnoise = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "putgaussnoise",
                 .cmdkey           = "putgaussnoise",
                 .description      = "add gaussian noise to image",
                 .description_long = "Statistical tools for image processing: add Gaussian noise, "
                                     "compute histograms, and generate statistical test images." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(putgaussnoise),
    .compute = putgaussnoise_compute,
};

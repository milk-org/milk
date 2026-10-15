// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_runnable.h"
#include "psf_internal.h"

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "psf"

// Module short description
#define MODULE_DESCRIPTION "Point Spread Function analysis"

double FWHM_MEASURED;

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

static char   psm_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imc";
static double psm_size                              = 20.0;
static char   psm_out[FUNCTION_PARAMETER_STRMAXLEN] = "outimc.txt";

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", psm_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube")   \
    X(".psfsize", &psm_size, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "estimated PSF size") \
    X(".out_name", psm_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output file")

MILK_PARAMS_DEFINE(psf, FPS_PARAMS);

static MILK_HOT errno_t psf_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    PSF_sequence_measure(psm_in, (float) psm_size, psm_out);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_psf_psf = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "psfseqmeas",
              .cmdkey           = "psfseqmeas",
              .description      = "measure PSF sequence",
              .description_long = "Measure PSF properties from a sequence of images. "
                                  "Computes centroid, FWHM, Strehl ratio, and encircled energy." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(psf),
    .compute = psf_compute,
};

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_psf_psf,
};

MILK_MODULE_RUNNABLES(psf, NULL, runnables);

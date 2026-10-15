// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    pfloadascii.c
 * @brief   Load ascii files to PF input
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static double  param_tstart                                 = 200.0;
static double  param_dt                                     = 0.001;
static int64_t param_nbpt                                   = 10000;
static int64_t param_nbfr                                   = 4;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "pfin";

#define FPS_PARAMS(X)                                                              \
    X(".tstart", &param_tstart, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "tstart") \
    X(".dt", &param_dt, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "dt")             \
    X(".nbpt", &param_nbpt, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb points")    \
    X(".nbfr", &param_nbfr, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb frames")    \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_LoadASCIIfiles(param_tstart, param_dt, (long) param_nbpt, (long) param_nbfr,
                                   param_out_name);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(pfloadascii, FPS_PARAMS);

static MILK_HOT errno_t pfloadascii_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_pfloadascii = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "pfloadascii",
                 .cmdkey           = "pfloadascii",
                 .description      = "load ascii files to PF input",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(pfloadascii),
    .compute = pfloadascii_compute,
};
EXPORT_FPSEXEC(milk, "linAR-pfloadascii", runnable_linARfilterPred_pfloadascii);

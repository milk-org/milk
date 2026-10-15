// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    applyARpfilt.c
 * @brief   Apply linear AR filter
 */


#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char   param_in_name[FUNCTION_PARAMETER_STRMAXLEN]   = "indata";
static char   param_filt_name[FUNCTION_PARAMETER_STRMAXLEN] = "Pfilt";
static double param_pflag                                   = 2.4;
static char   param_out_name[FUNCTION_PARAMETER_STRMAXLEN]  = "outPF";

#define FPS_PARAMS(X)                                                                         \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input data")    \
    X(".filt_name", param_filt_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "predictor") \
    X(".pflag", &param_pflag, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "PF lag")              \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_Apply_LinPredictor(param_filt_name, param_in_name, (float) param_pflag,
                                       param_out_name);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(applyARpfilt, FPS_PARAMS);

static MILK_HOT errno_t applyARpfilt_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_applyARpfilt = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "applyARpfilt",
                 .cmdkey           = "applyARpfilt",
                 .description      = "apply linear AR filter",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(applyARpfilt),
    .compute = applyARpfilt_compute,
};
EXPORT_FPSEXEC(milk, "linAR-applyARpfilt", runnable_linARfilterPred_applyARpfilt);

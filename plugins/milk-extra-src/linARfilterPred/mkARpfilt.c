// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mkARpfilt.c
 * @brief   Make linear AR filter
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char    param_in_name[FUNCTION_PARAMETER_STRMAXLEN]  = "indata";
static int64_t param_pforder                                = 5;
static double  param_pflag                                  = 2.4;
static double  param_svdeps                                 = 0.0001;
static double  param_reglambda                              = 0.0;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "outPF";
static int64_t param_loopmode                               = 0;
static double  param_loopgain                               = 0.1;
static int64_t param_testmode                               = 1;

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input data")  \
    X(".pforder", &param_pforder, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PF order")        \
    X(".pflag", &param_pflag, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "PF lag")            \
    X(".svdeps", &param_svdeps, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "SVD eps")         \
    X(".reglambda", &param_reglambda, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "reg param") \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output PF")     \
    X(".loopmode", &param_loopmode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "loop mode")     \
    X(".loopgain", &param_loopgain, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "loop gain")   \
    X(".testmode", &param_testmode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "test mode")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_Build_LinPredictor(
        param_in_name, (long) param_pforder, (float) param_pflag, param_svdeps, param_reglambda,
        param_out_name, 1, (int) param_loopmode, (float) param_loopgain, (int) param_testmode);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(mkARpfilt, FPS_PARAMS);

static MILK_HOT errno_t mkARpfilt_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_mkARpfilt = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mkARpfilt",
                 .cmdkey           = "mkARpfilt",
                 .description      = "make linear AR filter",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mkARpfilt),
    .compute = mkARpfilt_compute,
};
EXPORT_FPSEXEC(milk, "linAR-mkARpfilt", runnable_linARfilterPred_mkARpfilt);

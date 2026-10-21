// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mscangain.c
 * @brief   Scan gain
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char   param_in_name[FUNCTION_PARAMETER_STRMAXLEN] = "olwfsmeas";
static double param_multfact                              = 0.98;
static double param_framelag                              = 2.65;

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "mode vals")   \
    X(".multfact", &param_multfact, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "mult factor") \
    X(".framelag", &param_framelag, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "frame lag")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_ScanGain(param_in_name, (float) param_multfact, (float) param_framelag);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(mscangain, FPS_PARAMS);

static MILK_HOT errno_t mscangain_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_mscangain = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mscangain",
                 .cmdkey           = "mscangain",
                 .description      = "scan gain",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mscangain),
    .compute = mscangain_compute,
};
EXPORT_FPSEXEC(milk, "linAR-mscangain", runnable_linARfilterPred_mscangain);

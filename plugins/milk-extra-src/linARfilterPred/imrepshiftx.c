// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imrepshiftx.c
 * @brief   Repeat and shift image along X
 */


#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char    param_in_name[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static int64_t param_nbstep                                 = 5;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "imout";

#define FPS_PARAMS(X)                                                                 \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input") \
    X(".nbstep", &param_nbstep, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb steps")    \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

static MILK_HOT errno_t fpsexec()
{
    linARfilterPred_repeat_shift_X(param_in_name, (long) param_nbstep, param_out_name);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(imrepshiftx, FPS_PARAMS);

static MILK_HOT errno_t imrepshiftx_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_imrepshiftx = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imrepshiftx",
                 .cmdkey           = "imrepshiftx",
                 .description      = "repeat and shift image along X",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imrepshiftx),
    .compute = imrepshiftx_compute,
};
EXPORT_FPSEXEC(milk, "linAR-imrepshiftx", runnable_linARfilterPred_imrepshiftx);

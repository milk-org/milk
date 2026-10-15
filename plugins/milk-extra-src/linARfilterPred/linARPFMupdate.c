// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    linARPFMupdate.c
 * @brief   Update predictive filter matrix
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char   param_pf_name[FUNCTION_PARAMETER_STRMAXLEN]  = "outPF";
static char   param_pfm_name[FUNCTION_PARAMETER_STRMAXLEN] = "PFMat";
static double param_alpha                                  = 0.1;

#define FPS_PARAMS(X)                                                                        \
    X(".pf_name", param_pf_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "3D predictor") \
    X(".pfm_name", param_pfm_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "2D matrix")      \
    X(".alpha", &param_alpha, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "update coeff")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_PF_updatePFmatrix(param_pf_name, param_pfm_name, (float) param_alpha);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(linARPFMupdate, FPS_PARAMS);

static MILK_HOT errno_t linARPFMupdate_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_linARPFMupdate = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "linARPFMupdate",
                 .cmdkey           = "linARPFMupdate",
                 .description      = "update predictive filter matrix",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(linARPFMupdate),
    .compute = linARPFMupdate_compute,
};
EXPORT_FPSEXEC(milk, "linAR-PFMupdate", runnable_linARfilterPred_linARPFMupdate);

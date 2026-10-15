// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    linARapplyRT.c
 * @brief   RT apply predictive filter
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char    param_in_name[FUNCTION_PARAMETER_STRMAXLEN]  = "modevalOL";
static int64_t param_offset                                 = 0;
static int64_t param_semtrig                                = 2;
static char    param_pfm_name[FUNCTION_PARAMETER_STRMAXLEN] = "PFmat";
static int64_t param_pforder                                = 5;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "outPFmodeval";
static int64_t param_nbgpu                                  = 0;
static int64_t param_loop                                   = 0;
static int64_t param_nbiter                                 = 0;
static int64_t param_savemode                               = 0;
static double  param_tlag                                   = 1.8;
static int64_t param_pfindex                                = 0;

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "OL coeffs")   \
    X(".offset", &param_offset, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "index off")         \
    X(".semtrig", &param_semtrig, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "sem trig")        \
    X(".pfm_name", param_pfm_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "PF matrix") \
    X(".pforder", &param_pforder, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "filter order")    \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")        \
    X(".nbgpu", &param_nbgpu, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb GPUs")             \
    X(".loop", &param_loop, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "loop flag")             \
    X(".nbiter", &param_nbiter, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb iter")           \
    X(".savemode", &param_savemode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "save mode")     \
    X(".tlag", &param_tlag, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "time lag")            \
    X(".pfindex", &param_pfindex, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PF index")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_PF_RealTimeApply(
        param_in_name, (long) param_offset, (int) param_semtrig, param_pfm_name,
        (long) param_pforder, param_out_name, (int) param_nbgpu, (long) param_loop,
        (long) param_nbiter, (int) param_savemode, (float) param_tlag, (long) param_pfindex);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(linARapplyRT, FPS_PARAMS);

static MILK_HOT errno_t linARapplyRT_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_linARapplyRT = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "linARapplyRT",
                 .cmdkey           = "linARapplyRT",
                 .description      = "RT apply predictive filter",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(linARapplyRT),
    .compute = linARapplyRT_compute,
};
EXPORT_FPSEXEC(milk, "linAR-applyRT", runnable_linARfilterPred_linARapplyRT);

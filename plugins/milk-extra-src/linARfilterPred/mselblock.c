// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mselblock.c
 * @brief   Select modes belonging to block
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "linARfilterPred.h"

static char    param_in_name[FUNCTION_PARAMETER_STRMAXLEN]  = "modevals";
static char    param_bm_name[FUNCTION_PARAMETER_STRMAXLEN]  = "blockmap";
static int64_t param_blk                                    = 23;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "blk23modevals";

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input modes") \
    X(".bm_name", param_bm_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "block map")   \
    X(".blk", &param_blk, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "block number")            \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_SelectBlock(param_in_name, param_bm_name, (long) param_blk, param_out_name);
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(mselblock, FPS_PARAMS);

static MILK_HOT errno_t mselblock_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linARfilterPred_mselblock = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mselblock",
                 .cmdkey           = "mselblock",
                 .description      = "select modes belonging to block",
                 .description_long = "Linear autoregressive filter prediction engine. Manages "
                                     "filter training, update, and application for real-time "
                                     "prediction." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mselblock),
    .compute = mselblock_compute,
};
EXPORT_FPSEXEC(milk, "linAR-mselblock", runnable_linARfilterPred_mselblock);

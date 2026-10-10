// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    rt_legacy_total.c
 * @brief   Legacy counterpart of runnable_rt_total: CLI-only command, hand-written CLIfunction
 */

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "runnabletest/rt_kernels.h"
#include "rt_legacy.h"

static char   *inimname;
static double *scoeff;

static CLICMDARGDEF farg[] = { { CLIARG_IMG, ".in_name", "input image", "im1",
                                 (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), (void **) &inimname,
                                 NULL },
                               { CLIARG_FLOAT64, ".scaling", "scaling coefficient", "1.0",
                                 FPFLAG_DEFAULT_INPUT, (void **) &scoeff, NULL } };

static CLICMDDATA CLIcmddata = { "total", "runnabletest legacy: total of image (CLI only)",
                                 CLICMD_FIELDS_NOFPS };

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = imgid_make_from_name(inimname);

    errno_t retval = rt_stream_total(&img, *scoeff);

    imgid_free(&img);

    DEBUG_TRACE_FEXIT();
    return retval;
}

static errno_t CLIfunction(void)
{
    errno_t retval = CLI_checkarg_array(farg, CLIcmddata.nbarg);
    if (retval == RETURN_SUCCESS)
    {
        STD_FARG_LINKfunction return compute_function();
    }
    if (retval == RETURN_CLICHECKARGARRAY_HELP)
    {
        return RETURN_SUCCESS;
    }
    if (retval == RETURN_CLICHECKARGARRAY_FUNCPARAMSET)
    {
        return RETURN_SUCCESS;
    }
    return retval;
}

errno_t CLIADDCMD_runnabletestlegacy__total()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

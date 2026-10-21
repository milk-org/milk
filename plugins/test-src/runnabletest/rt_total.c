// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    rt_total.c
 * @brief   CLI-only MILK_RUNNABLE: no FPS, no processinfo, arguments parsed by the CLI core
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "runnabletest/rt_kernels.h"
#include "runnabletest.h"

// Linked to the CLI arguments by the CLI core before total_func() is called
static char   *total_in;
static double *total_scaling;

static CLICMDARGDEF total_farg[] = { { CLIARG_IMG, ".in_name", "input image", "im1",
                                       (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
                                       (void **) &total_in, NULL },
                                     { CLIARG_FLOAT64, ".scaling", "scaling coefficient", "1.0",
                                       FPFLAG_DEFAULT_INPUT, (void **) &total_scaling, NULL } };

MILK_RUNNABLE_RT_DEFINE(total);

static errno_t total_func(void)
{
    IMGID img = imgid_make_from_name(total_in);

    errno_t retval = rt_stream_total(&img, *total_scaling);

    imgid_free(&img);
    return retval;
}

const MILK_RUNNABLE runnable_rt_total = {
    .kind     = MILK_RUNNABLE_CLI,
    .app      = { .fps_name    = "rttotal",
                  .cmdkey      = "total",
                  .description = "runnabletest: total of image (CLI only)" },
    .srcfile  = __FILE__,
    .farg     = total_farg,
    .nbarg    = (int) (sizeof(total_farg) / sizeof(total_farg[0])),
    .cli_func = total_func,
    .rt       = &total_rt,
};

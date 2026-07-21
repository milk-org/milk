// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_basic.c
 * @brief   basic image functions
 *
 * Simple image routines
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "imgbasic"

// Module short description
#define MODULE_DESCRIPTION "standard image operations"

//#include <stdint.h>
//#include <string.h>
//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <errno.h>
//#include <unistd.h>
//#include <sched.h>

//#include <fitsio.h>  /* required by every program that uses CFITSIO  */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
//#include "COREMOD_tools/COREMOD_tools.h"
//#include "COREMOD_memory/COREMOD_memory.h"
//#include "COREMOD_iofits/COREMOD_iofits.h"
//#include "COREMOD_arith/COREMOD_arith.h"

/*
#include "fft/fft.h"
#include "image_filter/image_filter.h"
#include "image_gen/image_gen.h"
#include "info/info.h"
#include "kdtree/kdtree.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"
*/

//#include "image_basic/image_basic.h"

#    include "cubecollapse.h"
#    include "im3Dto2D.h"
#    include "image_add.h"
#    include "imcontract.h"
#    include "imexpand.h"
#    include "imgetcircasym.h"
#    include "imgetcircsym.h"
#    include "imresize.h"
#    include "imrotate.h"
#    include "imswapaxis2D.h"
#    include "indexmap.h"
#    include "loadfitsimgcube.h"
#    include "streamfeed.h"
#    include "streamrecord.h"

/*
#define SBUFFERSIZE 1000

#define SWAP(x,y)  temp=(x);x=(y);y=temp;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


char errmsg[SBUFFERSIZE];
*/

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
static errno_t init_module_CLI()
{
    CLIADDCMD_image_basic__imswapaxis2D();
    CLIADDCMD_image_basic__im3Dto2D();
    CLIADDCMD_image_basic__image_add();
    CLIADDCMD_image_basic__imexpand();
    CLIADDCMD_image_basic__imgetcircsym();
    CLIADDCMD_image_basic__imgetcircasym();
    CLIADDCMD_image_basic__imresize();
    CLIADDCMD_image_basic__imcontract();
    CLIADDCMD_image_basic__imrotate();
    CLIADDCMD_image_basic__loadfitsimgcube();
    CLIADDCMD_image_basic__streamfeed();
    CLIADDCMD_image_basic__streamrecord();
    CLIADDCMD_image_basic__cubecollapse();
    CLIADDCMD_image_basic__indexmap();

    // add atexit functions here

    return RETURN_SUCCESS;
}

MILK_MODULE(image_basic, init_module_CLI, NULL);
#endif /* MILK_NO_CLI */

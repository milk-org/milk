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

#include "image_basic.h"

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

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_image_basic_imswapaxis2D,    &runnable_image_basic_im3Dto2D,
    &runnable_image_basic_addim,           &runnable_image_basic_addim3D,
    &runnable_image_basic_imexpand,        &runnable_image_basic_imexpand3D,
    &runnable_image_basic_imgetcircsym,    &runnable_image_basic_imgetcircasym,
    &runnable_image_basic_imresize,        &runnable_image_basic_imcontract,
    &runnable_image_basic_imcontract3D,    &runnable_image_basic_imrotate,
    &runnable_image_basic_loadfitsimgcube, &runnable_image_basic_streamfeed,
    &runnable_image_basic_streamrecord,    &runnable_image_basic_cubecollapse,
    &runnable_image_basic_indexmap,
};

MILK_MODULE_RUNNABLES(image_basic, NULL, runnables);

/**
 * @file    COREMOD_arith.c
 * @brief   Arithmeric operations on images
 *
 * Addition, multiplication and much more
 *
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
#define MODULE_SHORTNAME_DEFAULT ""

// Module short description
#define MODULE_DESCRIPTION "Image arithmetic operations"

/* ================================================================== */
/* ================================================================== */
/*            DEPENDANCIES                                            */
/* ================================================================== */
/* ================================================================== */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fitsio.h>

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
#endif

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

//#include "COREMOD_arith/COREMOD_arith.h"

#include "image_crop.h"
#include "image_cropmask.h"
#include "image_dxdy.h"
#include "image_merge3D.h"
#include "image_stats.h"
#include "image_total.h"
#include "imfunctions.h"
#include "set_pixel.h"

#include "image_arith__im__im.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"
#include "image_arith__im_im__im.h"
#include "mathfuncs.h"

#include "execute_arith.h"

/* ================================================================== */
/* ================================================================== */
/*           MACROS, DEFINES                                          */
/* ================================================================== */
/* ================================================================== */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_arith)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

static errno_t init_module_CLI()
{

    image_crop_addCLIcmd();

    set_pixel_addCLIcmd();

    image_arith__im_f_f__im_addCLIcmd();

    image_merge3D_addCLIcmd();

    CLIADDCMD_COREMODE_arith__cropmask();

    // add atexit functions here

    return RETURN_SUCCESS;
}

errno_t init_COREMOD_arith()
{
    init_module_CLI();

    return RETURN_SUCCESS;
}

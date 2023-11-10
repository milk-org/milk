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
#include "image_norm.h"
#include "image_slicenormalize.h"
#include "image_merge3D.h"
#include "image_stats.h"

#include "image_set_1Dpixrange.h"
#include "image_set_2Dpix.h"
#include "image_set_col.h"
#include "image_set_row.h"
#include "image_setzero.h"

#include "image_pixremap.h"
#include "image_pixunmap.h"

#include "image_total.h"
#include "imfunctions.h"

#include "image_arith__im__im.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"
#include "image_arith__im_im__im.h"
#include "mathfuncs.h"

#include "execute_arith.h"



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


INIT_MODULE_LIB(COREMOD_arith)



static errno_t init_module_CLI()
{

    image_crop_addCLIcmd();

    image_arith__im_f_f__im_addCLIcmd();

    CLIADDCMD_COREMOD_arith__image_merge();

    CLIADDCMD_COREMOD_arith__image_normslice();
    CLIADDCMD_COREMOD_arith__image_slicenormalize();

    CLIADDCMD_COREMODE_arith__cropmask();

    CLIADDCMD_COREMOD_arith__imset_1Dpixrange();
    CLIADDCMD_COREMOD_arith__imset_2Dpix();
    CLIADDCMD_COREMOD_arith__imset_col();
    CLIADDCMD_COREMOD_arith__imset_row();
    CLIADDCMD_COREMOD_arith__imsetzero();

    CLIADDCMD_COREMODE_arith__pixremap();
    CLIADDCMD_COREMODE_arith__pixunmap();

    return RETURN_SUCCESS;
}

errno_t init_COREMOD_arith()
{
    init_module_CLI();

    return RETURN_SUCCESS;
}

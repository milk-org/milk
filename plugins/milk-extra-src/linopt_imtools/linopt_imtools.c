/**
 * @file    linopt_imtools.c
 * @brief   linear optimization tools
 *
 * CPU-based lineal algebra tools: decomposition, SVD etc...
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
#define MODULE_SHORTNAME_DEFAULT "lintools"

// Module short description
#define MODULE_DESCRIPTION "Image linear decomposition and optimization tools"

#include <ctype.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#include <time.h>

#include <fitsio.h>

#include "CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "linalgebra/linalgebra.h"
#include "info/info.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"

#include "timeutils.h"

#include "compute_SVDdecomp.h"
#include "compute_SVDpseudoInverse.h"
#include "image_construct.h"
#include "image_fitModes.h"
#include "image_to_vec.h"
#include "imcube_crossproduct.h"
#include "lin1Dfit.h"
#include "makeCPAmodes.h"
#include "makeCosRadModes.h"
#include "mask_to_pixtable.h"


/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(linopt_imtools)


static errno_t init_module_CLI()
{

    // CONVERSION

    CLIADDCMD_linopt_imtools__mask_to_pixtable();

    CLIADDCMD_linopt_imtools__image_to_vec();

    CLIADDCMD_linopt_imtools__vec_to_2DImage();

    // CREATE MODES

    CLIADDCMD_linopt_imtools__makeCosRadModes();

    CLIADDCMD_linopt_imtools__makeCPAmodes();

    // LINEAR DECOMPOSITION

    CLIADDCMD_linopt_imtools__imcube_crossproduct();

    CLIADDCMD_linopt_imtools__image_fitModes();

    CLIADDCMD_linopt_imtools__image_construct();



    CLIADDCMD_linopt_imtools__compute_SVDdecomp();

    CLIADDCMD_linopt_imtools__compute_SVDpseudoinverse();

    CLIADDCMD_linopt_imtools__lin1Dfits();

    // OPTIMIZATION

    CLIADDCMD_linopt_imtools__linRM_from_inout();

    return RETURN_SUCCESS;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 3. CREATE MODES                                                                                 */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/**
 * Create 1D polynomial basis functions.
 */
imageID linopt_imtools_make1Dpolynomials(
    const char *IDout_name,
    long        NBpts,
    long        MaxOrder,
    float       r0pix)
{
    DEBUG_TRACE_FSTART();

    IMGID imgout =
        imgid_make_from_name_3D(
            IDout_name,
            NBpts, 1, MaxOrder);
    imgout.mdt->shared = 0;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for(long kk = 0; kk < MaxOrder; kk++)
    {
        for(long ii = 0; ii < NBpts; ii++)
        {
            float r =
                1.0 * ii / r0pix;
            imgout.im->array.F[
                kk * NBpts + ii] =
                pow(r, 1.0 * kk);
        }
    }

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}


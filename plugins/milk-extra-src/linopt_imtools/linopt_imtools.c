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
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

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

// r0pix is r=1 in pixel unit

imageID linopt_imtools_make1Dpolynomials(const char *IDout_name,
        long        NBpts,
        long        MaxOrder,
        float       r0pix)
{
    DEBUG_TRACE_FSTART();

    imageID IDout;
    long    xsize, ysize, zsize;
    long    ii, kk;

    xsize = NBpts;
    ysize = 1;
    zsize = MaxOrder;

    FUNC_CHECK_RETURN(
        create_3Dimage_ID(IDout_name, xsize, ysize, zsize, &IDout));

    for(kk = 0; kk < zsize; kk++)
    {
        for(ii = 0; ii < xsize; ii++)
        {
            float r                                    = 1.0 * ii / r0pix;
            dcimg[IDout].array.F[kk * xsize + ii] = pow(r, 1.0 * kk);
        }
    }

    DEBUG_TRACE_FEXIT();
    return IDout;
}


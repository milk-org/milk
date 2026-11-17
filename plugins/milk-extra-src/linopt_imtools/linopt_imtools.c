// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

#include "milk_runnable.h"

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
#include "lin1Dfit.h"
#include "makeCPAmodes.h"
#include "makeCosRadModes.h"
#include "mask_to_pixtable.h"


/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Mandated units are weak: NULL when their file is dropped by an unmet MILK_CMAKE_MANDATE_*.
MILK_RUNNABLE_WEAK_DECL(runnable_linopt_imtools_image_fitModes);
MILK_RUNNABLE_WEAK_DECL(runnable_linopt_imtools_compute_SVDdecomp);
MILK_RUNNABLE_WEAK_DECL(runnable_linopt_imtools_compute_SVDpseudoInverse);
MILK_RUNNABLE_WEAK_DECL(runnable_linopt_imtools_linRM_from_inout);

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_linopt_imtools_mask_to_pixtable,  &runnable_linopt_imtools_image_to_vec,
    &runnable_linopt_imtools_vec_to_2Dimage,    &runnable_linopt_imtools_makeCosRadModes,
    &runnable_linopt_imtools_makeCPAmodes,      &runnable_linopt_imtools_imcube_crossproduct,
    &runnable_linopt_imtools_image_fitModes,    &runnable_linopt_imtools_image_construct,
    &runnable_linopt_imtools_compute_SVDdecomp, &runnable_linopt_imtools_compute_SVDpseudoInverse,
    &runnable_linopt_imtools_lin1Dfit,          &runnable_linopt_imtools_linRM_from_inout,
};

MILK_MODULE_RUNNABLES(linopt_imtools, NULL, runnables);


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
imageID linopt_imtools_make1Dpolynomials(const char *IDout_name,
                                         long        NBpts,
                                         long        MaxOrder,
                                         float       r0pix)
{
    DEBUG_TRACE_FSTART();

    IMGID imgout       = imgid_make_from_name_3D(IDout_name, NBpts, 1, MaxOrder);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    for (long kk = 0; kk < MaxOrder; kk++)
    {
        for (long ii = 0; ii < NBpts; ii++)
        {
            float r                             = 1.0 * ii / r0pix;
            imgout.im->array.F[kk * NBpts + ii] = pow(r, 1.0 * kk);
        }
    }

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_f_f__im.c
 * @brief   arith functions
 *
 * input : image, float, float
 * output: image
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__im_f_f__im.h"


// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static char   *inimname;
static double *valmin;
static double *valmax;
static char   *outimname;

static CLICMDARGDEF farg[] = {
    { CLIARG_IMG, ".in_name", "input image", "im1", CLIARG_VISIBLE_DEFAULT, (void **) &inimname,
      NULL },
    { CLIARG_FLOAT64, ".min", "min value", "0.0", CLIARG_VISIBLE_DEFAULT, (void **) &valmin, NULL },
    { CLIARG_FLOAT64, ".max", "max value", "1.0", CLIARG_VISIBLE_DEFAULT, (void **) &valmax, NULL },
    { CLIARG_STR, ".out_name", "output image", "out1", CLIARG_VISIBLE_DEFAULT, (void **) &outimname,
      NULL }
};

static CLICMDDATA CLIcmddata = { "imtrunc", "truncate pixel values", CLICMD_FIELDS_DEFAULTS };

static errno_t help_function()
{
    printf("Truncate pixel values between min and max.\n");
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    IMGID imgin  = mkIMGID_from_name(inimname);
    IMGID imgout = mkIMGID_from_name(outimname);

    arith_image_trunc_IMGID(&imgin, *valmin, *valmax, &imgout);

    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

    // ==========================================
    // Register CLI command(s)
    // ==========================================

    errno_t image_arith__im_f_f__im_addCLIcmd()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

int arith_image_trunc_IMGID(IMGID *imgin, double f1, double f2, IMGID *imgout)
{
    arith_image_function_1ff_1_IMGID(imgin, f1, f2, imgout, &Ptrunc);
    return (0);
}

int arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);

    arith_image_trunc_IMGID(&imgin, f1, f2, &imgout);

    return (0);
}

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2)
{
    arith_image_function_1ff_1_inplace(ID_name, f1, f2, &Ptrunc);
    return (0);
}
int arith_image_trunc_inplace_byID(long ID, double f1, double f2)
{
    arith_image_function_1ff_1_inplace_byID(ID, f1, f2, &Ptrunc);
    return (0);
}

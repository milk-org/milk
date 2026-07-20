// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    simplefunc_FPS.c
 * @brief   simple function example with FPS and processinfo support
 *
 * Example 2
 * Demonstrates using FPS to hold function arguments and parameters with the Function Parameter Structure (FPS).
 * See script milk-test-simplefuncFPS for example usage.
 */

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers

static char *inimname;

static double *scoeff;


// List of arguments to function
static CLICMDARGDEF farg[] =
{
    //    FARG_INPUTIM(inim),
    {
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".scaling",
        "scaling coefficient",
        "1.0",
        CLIARG_HIDDEN_DEFAULT, // hidden argument is not part of CLI call, FPFLAG ignored
        (void **) &scoeff,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "imsum2",
    "compute total of image example2, FPS-compatible",
    CLICMD_FIELDS_DEFAULTS
};



static errno_t help_function()
{
    printf("\n");

    return RETURN_SUCCESS;
}




/**
 * @brief Sum pixel values
 *
 */
static errno_t example_compute_2Dimage_total(
    IMGID *imgptr,
    double scalingcoeff)
{
    DEBUG_TRACE_FSTART();

    // Ensure the input image is in memory.
    // No harm calling this here and in the upstream function,
    // as the overhead is very small if the image is already resolved
    resolveIMGID(imgptr, ERRMODE_ABORT);

    uint32_t xsize  = imgptr->md->size[0];
    uint32_t ysize  = imgptr->md->size[1];
    uint64_t xysize = xsize * ysize;


    double total = 0.0;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        total += imgptr->im->array.F[ii];
    }
    total *= scalingcoeff;

    printf("image %s total = %lf (scaling coeff %lf)\n",
           imgptr->im->name,
           total,
           scalingcoeff);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




/**
 * @brief Wrapper function, used by all CLI calls
 *
 * INSERT_STD_PROCINFO statements enable processinfo support
 */
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // Check that the input image is in memory,
    // and link it to img if it is
    IMGID img = mkIMGID_from_name(inimname);
    resolveIMGID(&img, ERRMODE_ABORT);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    example_compute_2Dimage_total(&img, *scoeff);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



// Enables FPS support
INSERT_STD_FPSCLIfunctions


errno_t
CLIADDCMD_milk_module_example__simplefunc_FPS()
{
    INSERT_STD_CLIREGISTERFUNC

    // Optional custom settings for this function can be included
    // CLIcmddata.cmdsettings->procinfo_loopcntMax = 9;

    return RETURN_SUCCESS;
}

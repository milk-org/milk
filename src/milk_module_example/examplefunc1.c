// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    examplefunc1.c
 * @brief   simple function example
 *
 * Example 1
 * Demonstrates how functions are registered and their arguments processed.
 * Demonstrates how functions connect to images.
 * See script milk-test-simplefunc for example usage.
 */

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers

// make sure to match types with farg types

// Within this translation unit, these point to the variables values shared between the FPS and the processes


static char *inimname;

// Make sure the type here matches the type flag in CLICMDARGDEF
static double *scoeff; // matches CLIARG_FLOAT64
// Include the following line, and add reference to pointer in CLICMDARGDEF entry to allow for dynamic access to parameter
// static long    fpi_scoeff = -1;


// Possible types are:
// static float *      for CLIARG_FLOAT32
// static double *     for CLIARG_FLOAT64
// static chat *       for CLIARG_IMG, CLIARG_FILENAME, CLIARG_FITSFILENAME, CLIARG_FPSNAME, CLIARG_STREAM, CLIARG_STR, CLIARG_STR_NOT_IMG
// static uint32_t *   for CLIARG_UINT32
// static uint64_t *   for CLIARG_UINT64
// static int32_t *    for CLIARG_INT32
// static int64_t *    for CLIARG_INT64, CLIARG_ONOFF, CLIARG_LONG




// List of arguments to function
// { CLItype, tag, description, initial value, flag, fptype, fpflag }
//
// A function variable is named by a tag, which is a hierarchical
// series of words separated by dot "."
// For example: .input.xsize (note that first dot is optional)
//
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, // type of argument
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT, // This will be exposed as a function argument in the milk CLI, which has to be entered
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

// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "imsum1",                          // keyword to call function in from the milk CLI
    "compute total of image example1", // brief (1-line) description of what the function does
    CLICMD_FIELDS_NOFPS                // do NOT use Function Parameter Structure (FPS)
};



/** @brief Compute function code
 *
 * Can be made non-static and called from outside this translation unit(TU)
 * Minimizes use of variables local to this TU.
 *
 * Functions should return error code of type errno_t (= int).
 * On success, return value is RETURN_SUCCESS (=0).
 */
static errno_t example_compute_2Dimage_total(
    IMGID *imgptr,
    double scalingcoeff
)
{
    // The preferred way to have images and streams as function args is to pass a pointer to IMGID struct.
    // Here, the function needs to change the IMGID content (call to resolveIMGID).
    // Accessing images through IMGID is faster than through their names, as there is no need to resolve
    // the image (find out which ID corresponds to its name) each time the function is called.

    // Entering function, updating trace accordingly
    DEBUG_TRACE_FSTART();

    // Resolve image if not already resolved.
    // This is a low-overhead function if the image is already in memory and imgptr already pointing to it.
    // If not already connected, the function will use imgptr->name to try to connect to it.
    resolveIMGID(imgptr, ERRMODE_ABORT);
    // Abort if unable to resolve.
    // Upon success, these are available for use:
    // imgptr->name, imgptr->naxis, imgptr->ID, imgptr->size, imgptr->im

    // From now on, we access the image and its metadata through its IMGID
    uint32_t  xsize  = imgptr->md->size[0];
    uint32_t  ysize  = imgptr->md->size[1];
    uint64_t  xysize = xsize * ysize;

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

    // normal successful return from function :
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Wrapper function, used by all CLI calls
 *
 * Defines how local variables are fed to computation code.
 * Always local to this translation unit.
 *
 * @return errno_t
 */
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    // Note how we are accessing the image from its name.
    IMGID img = mkIMGID_from_name(inimname);
    // The function mkIMGID_from_name takes the name as argument and forms an IMGID from it.
    // At this point the connection to the image has not been established. This will be done
    // on the first call of resolveIMGID inside the compute function.

    example_compute_2Dimage_total(
        &img,
        *scoeff);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_CLIfunction



/** @brief Register CLI command
*
* Adds function to list of CLI commands.
* Called by main module initialization function init_module_CLI().
*/
errno_t
CLIADDCMD_milk_module_example__simplefunc()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

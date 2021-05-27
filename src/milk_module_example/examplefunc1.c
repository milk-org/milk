/**
 * @file    simplefunc.c
 * @brief   simple function example
 *
 * Example 1
 * Demonstrates how functions are registered and their arguments processed.
 * See script milk-test-simplefunc for example usage.
 */

#include "CommandLineInterface/CLIcore.h"


// Local variables pointers
// Within this translation unit, these point to the variables values
static char *inimname;
// float point variable should be double. single precision float not supported
static double *scoeff;


// List of arguments to function
// { CLItype, tag, description, initial value, flag, fptype, fpflag}
//
// A function variable is named by a tag, which is a hierarchical
// series of words separated by dot "."
// For example: .input.xsize (note that first dot is optional)
//
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".in_name", "input image", "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname
    },
    {   // hidden argument is not part of CLI call, FPFLAG ignored
        CLIARG_FLOAT, ".scaling", "scaling coefficient", "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &scoeff
    }
};


// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "simplefunc",             // keyword to call function in CLI
    "compute total of image", // description of what the function does
    CLICMD_FIELDS_NOFPS
};



// detailed help
static errno_t help_function()
{
    printf(
        "Detailed help for function\n"
    );

    return RETURN_SUCCESS;
}







/** @brief Compute function code
 *
 * Can be made non-static and called from outside this translation unit(TU)
 * Minimizes use of variables local to this TU.
 */
static errno_t example_compute_2Dimage_total(
    IMGID img,
    double scalingcoeff)
{
    resolveIMGID(&img, ERRMODE_ABORT);

    uint_fast32_t xsize = img.md->size[0];
    uint_fast32_t ysize = img.md->size[1];
    uint_fast64_t xysize = xsize * ysize;

    double total = 0.0;
    for(uint_fast64_t ii = 0; ii < xysize; ii++)
    {
        total += img.im->array.F[ii];
    }
    total *= scalingcoeff;

    printf("image total = %lf, scaling coeff %lf\n", total, scalingcoeff);

    return RETURN_SUCCESS;
}


// Wrapper function, used by all CLI calls
// Defines how local variables are fed to computation code
// Always local to this translation unit
static errno_t compute_function()
{
    example_compute_2Dimage_total(
        makeIMGID(inimname),
        *scoeff
    );
    return RETURN_SUCCESS;
}





INSERT_STD_CLIfunction


/** @brief Register CLI command
 *
 * Adds function to list of CLI commands.
 * Called by main module initialization function init_module_CLI().
 */
errno_t CLIADDCMD_milk_module_example__simplefunc()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

/**
 * @file    examplefunc1.c
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
        (void **) &inimname, NULL
    },
    {
        // hidden argument is not part of CLI call, FPFLAG ignored
        CLIARG_FLOAT, ".scaling", "scaling coefficient", "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &scoeff, NULL
    }
};


// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "imsum1",                  // keyword to call function in CLI
    "compute total of image example1", // description of what the function does
    CLICMD_FIELDS_NOFPS
};




/**
 * @brief Detailed help
 *
 * @return errno_t
 */
static errno_t help_function()
{
    printf(
        "Example function demonstrating basic CLI interface\n"
        "Adds pixel values of an image with a global scaling parameter\n"
        "Input is image, output is scalar\n"
        "This function does not support fps or procinfo\n"
    );

    return RETURN_SUCCESS;
}







/** @brief Compute function code
 *
 * Can be made non-static and called from outside this translation unit(TU)
 * Minimizes use of variables local to this TU.
 *
 * Functions should return error code of type errno_t (= int).
 * On success, return value is RETURN_SUCCESS (=0).
 */
static errno_t example_compute_2Dimage_total(
    IMGID  img,
    double scalingcoeff)
{
    // entering function, updating trace accordingly
    DEBUG_TRACE_FSTART();

    // Resolve image if not already resolved
    resolveIMGID(&img, ERRMODE_ABORT);


    // If function fails and error cannot be recovered from, use :

    // abort();


    // If error, return from function with error code and have
    // caller handle it :

    // FUNC_RETURN_FAILURE("error description");


    // If calling other milk function, use following macro
    // to test and handle possible error return :

    // FUNC_CHECK_RETURN(othermilkfunc(img));

    uint32_t xsize = img.md->size[0];
    uint32_t ysize = img.md->size[1];
    uint_fast64_t xysize = xsize * ysize;

    double total = 0.0;
    for(uint_fast64_t ii = 0; ii < xysize; ii++)
    {
        total += img.im->array.F[ii];
    }
    total *= scalingcoeff;

    printf("image %s total = %lf (scaling coeff %lf)\n",
           img.im->name,
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

    example_compute_2Dimage_total(
        makeIMGID(inimname),
        *scoeff
    );

    DEBUG_TRACE_FEXIT();
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

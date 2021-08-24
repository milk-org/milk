/**
 * @file    simplefunc_FPS.c
 * @brief   simple function example with FPS and processinfo support
 *
 * Example 2
 * Demonstrates using FPS to hold function arguments and parameters.
 * See script milk-test-simplefuncFPS for example usage.
 */

#include "CommandLineInterface/CLIcore.h"


// Local variables pointers
static char *inimname;
static double *scoeff;


// List of arguments to function
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".in_name", "input image", "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname
    },
    {
        // argument is not part of CLI call, FPFLAG ignored
        CLIARG_FLOAT, ".scaling", "scaling coefficient", "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &scoeff
    }
};


// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "simplefuncFPS",
    "compute total of image using FPS",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




// Computation code
static errno_t example_compute_2Dimage_total(
    IMGID img,
    double scalingcoeff
)
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

    printf("image %s total = %lf (scaling coeff %lf)\n", img.im->name, total,
           scalingcoeff);

    return RETURN_SUCCESS;
}


// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    example_compute_2Dimage_total(
        makeIMGID(inimname),
        *scoeff
    );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_milk_module_example__simplefunc_FPS()
{
    INSERT_STD_CLIREGISTERFUNC

    // Optional custom settings for this function
    // CLIcmddata.cmdsettings->procinfo_loopcntMax = 9;

    return RETURN_SUCCESS;
}

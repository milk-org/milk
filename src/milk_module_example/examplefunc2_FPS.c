/**
 * @file    simplefunc_FPS.c
 * @brief   simple function example with FPS and processinfo support
 *
 * Example 2
 * Demonstrates using FPS to hold function arguments and parameters with the Function Parameter Structure (FPS).
 * See script milk-test-simplefuncFPS for example usage.
 */

#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

static FPS_APP_INFO FPS_app_info = { .fps_name = "imsum2",
                                     .cmdkey   = "imsum2",
                                     .description =
                                         "compute total of image example2, FPS-compatible" };

// Local variables pointers

static char   *inimname;
static double *scoeff;

#define FPS_PARAMS(X)                                                                   \
    X(".in_name", &inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".scaling", &scoeff, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "scaling coefficient")


static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


/**
 * @brief Sum pixel values
 *
 */
static errno_t example_compute_2Dimage_total(IMGID *imgptr, double scalingcoeff)
{
    DEBUG_TRACE_FSTART();

    // Ensure the input image is in memory.
    // No harm calling this here and in the upstream function,
    // as the overhead is very small if the image is already resolved
    resolveIMGID(imgptr, ERRMODE_WARN, dcimg, dcnimg);

    uint32_t xsize = imgptr->md->size[0];
    if (imgptr->ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint32_t ysize  = imgptr->md->size[1];
    uint64_t xysize = xsize * ysize;


    double total = 0.0;
    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        total += imgptr->im->array.F[ii];
    }
    total *= scalingcoeff;

    printf("image %s total = %lf (scaling coeff %lf)\n", imgptr->im->name, total, scalingcoeff);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Wrapper function, used by all CLI calls
 *
 * INSERT_STD_PROCINFO statements enable processinfo support
 */
static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // Check that the input image is in memory,
    // and link it to img if it is
    IMGID img = imgid_make_from_name(inimname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    example_compute_2Dimage_total(&img, *scoeff);
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

// Register function in CLI
errno_t CLIADDCMD_milk_module_example__simplefunc_FPS()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    // Optional custom settings for this function can be included
    // CLIcmddata.cmdsettings->procinfo_loopcntMax = 9;

    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif

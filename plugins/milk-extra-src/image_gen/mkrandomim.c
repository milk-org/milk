#include "ImageStreamIO/ImageStruct.h"
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include "CLIcore.h"
#endif
#include "statistic/statistic.h" // ran1, gauss, gauss_trc



/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkrnd",
    .cmdkey      = "mkrnd",
    .description = "make random image",
    .description_long =
        "Generate a random image with Poisson or Gaussian noise for testing and simulation."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     outim_name[FUNCTION_PARAMETER_STRMAXLEN]
    = "outim";
static uint32_t outim_xsize  = 256;
static uint32_t outim_ysize  = 256;
static uint32_t distrib_val  = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".outim.name", outim_name, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".outim.xsize", &outim_xsize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x size") \
    X(".outim.ysize", &outim_ysize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y size") \
    X(".distrib", &distrib_val, \
      FPTYPE_UINT32, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "distribution (0:uniform 1:gauss 2:trunc)")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)



/**
 * @brief Make random image
 *
 * @param[out] img
 *      Output image
 *
 * @param[in] pdf
 *      Probability distribution function
 *
 * @return imageID
 */
static imageID make_image_random(
    IMGID *img,
    int pdf
)
{
    DEBUG_TRACE_FSTART();

    // 0: uniform
    // 1: gauss
    // 2: truncated gauss

    // Create image if needed
    //imcreateIMGID(img);

    // openMP is slow when calling gsl random
    // number generator : do not use openMP here
    if (pdf == 0)
    {
        for (uint64_t ii = 0;
             ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] = (float) ran1();
        }
    }
    if (pdf == 1)
    {
        for (uint64_t ii = 0;
             ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] = (float) gauss();
        }
    }
    if (pdf == 2)
    {
        for (uint64_t ii = 0;
             ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] =
                (float) gauss_trc();
        }
    }
    if (pdf == 3)  // test pattern
    {
        static uint64_t ii   = 0;
        img->im->array.F[ii] =
            1.0 - img->im->array.F[ii];
        ii++;
        if (ii == img->md->nelement)
        {
            ii = 0;
        }
    }

    DEBUG_TRACE_FEXIT();
    return (img->ID);
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("make IMGID for %s",
                     outim_name);

    /*
     * Connect to existing stream or create it.
     * Must be before the loop to avoid leaking
     * img.mdt on every iteration.
     */
    IMGID img = stream_connect_create_2D(
        outim_name, outim_xsize, outim_ysize,
        _DATATYPE_FLOAT);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_image_random(&img, distrib_val);

    DEBUG_TRACEPOINT("update output ID %ld",
                     img.ID);
    processinfo_update_output_stream(
        processinfo, img.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&img);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_gen__mkrandomim()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif


/**
 * @file    mkspdisk.c
 * @brief   Create a subpixel disk image (fpsexec standalone)
 *
 * Generate a 2D float image containing a sub-pixel
 * accurate disk at specified center + radius.
 */

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include "CLIcore.h"
#endif

#include "image_gen/image_gen.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkspdisk",
    .cmdkey      = "mkspdisk",
    .description = "make subpixel disk image",
    .description_long =
        "Generate a sub-pixel accurate disk pattern using anti-aliasing. More precise than mkdisk for small radii."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    outim_name[FUNCTION_PARAMETER_STRMAXLEN]
    = "imdisk";
static int64_t outim_xsize  = 512;
static int64_t outim_ysize  = 512;
static double  outim_xcenter = 256.0;
static double  outim_ycenter = 256.0;
static double  outim_radius  = 100.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".out_name", outim_name, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image name") \
    X(".xsize", &outim_xsize, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x size") \
    X(".ysize", &outim_ysize, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y size") \
    X(".xcenter", &outim_xcenter, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x center") \
    X(".ycenter", &outim_ycenter, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y center") \
    X(".radius", &outim_radius, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "radius")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */

static errno_t compute_function(void)
{
    /* Pre-create output as shared memory stream.
     * make_subpixdisk() calls create_2Dimage_ID()
     * which reuses this existing shared image. */
    IMGID img = stream_connect_create_2D(
        outim_name,
        (uint32_t) outim_xsize,
        (uint32_t) outim_ysize,
        _DATATYPE_FLOAT);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_subpixdisk(
        outim_name,
        (uint32_t) outim_xsize,
        (uint32_t) outim_ysize,
        outim_xcenter,
        outim_ycenter,
        outim_radius);

    processinfo_update_output_stream(
        processinfo, img.im, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

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
CLIADDCMD_image_gen__mkspdisk(void)
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

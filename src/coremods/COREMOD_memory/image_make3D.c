/**
 * @file image_make3D.c
 * @brief Image make3d module
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_make3D.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mk3Dim",
    .cmdkey      = "mk3Dim",
    .description = "make 3D image",
    .description_long =
        "Create a new 3D image cube in shared memory with specified dimensions and data type. Initializes all pixel values to zero."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     outimname[FUNCTION_PARAMETER_STRMAXLEN] = "im3D";
static uint32_t imxsize = 256;
static uint32_t imysize = 256;
static uint32_t imzsize = 256;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".out_name", outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".xsize", &imxsize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x size") \
    X(".ysize", &imysize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y size") \
    X(".zsize", &imzsize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "z size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID make_3Dimage_IMGID(IMGID *img)
{
    imcreateIMGID(img);
    return (img->ID);
}

imageID make_3Dimage(
    const char *name,
    uint32_t   xsize,
    uint32_t   ysize,
    uint32_t   zsize)
{
    IMGID img = imgid_make_from_name_3D(name, xsize, ysize, zsize);
    return make_3Dimage_IMGID(&img);
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = imgid_make_from_name_3D(outimname, imxsize, imysize, imzsize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START  make_3Dimage_IMGID(&img);

    processinfo_update_output_stream(processinfo, img.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END  DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__mk3Dim()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
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

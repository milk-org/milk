/**
 * @file    image_set_1Dpixrange.c
 * @brief   Set pixels in a 1D index range
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "setpix1D",
    .cmdkey      = "setpix1Drange",
    .description =
        "set image pixel value over range",
    .description_long =
        "Set a contiguous range of pixels in a 1D image or along the linear memory layout of a multi-dimensional image. Specify start index, end index, and the value to assign."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *setpix1d_inimname = NULL;
static float    *setpix1d_pixval   = NULL;
static uint32_t *setpix1d_minindex = NULL;
static uint32_t *setpix1d_maxindex = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imname", &setpix1d_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".pixval", &setpix1d_pixval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "pixel value") \
    X(".mini", &setpix1d_minindex, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "min index") \
    X(".maxi", &setpix1d_maxindex, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "max index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    if (!setpix1d_pixval
        || !setpix1d_minindex
        || !setpix1d_maxindex)
    {
        return RETURN_FAILURE;
    }
    float    val = *setpix1d_pixval;
    uint32_t mi  = *setpix1d_minindex;
    uint32_t ma  = *setpix1d_maxindex;

    if (ma > inimg->md[0].nelement) {
        ma = inimg->md[0].nelement;
    }
    if (mi >= ma) {
        return RETURN_FAILURE;
    }
#define SET1D_CASE_(DT, ACC, CT)                \
    case DT:                                    \
        for (uint32_t i = mi; i < ma; i++)      \
            inimg->array.ACC[i] = (CT) val;     \
        break;

    switch (inimg->md[0].datatype) {
        FOREACH_REAL_DATATYPE(SET1D_CASE_) default: PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
#undef SET1D_CASE_
    return RETURN_SUCCESS;
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
    IMGID in = imgid_make_from_name(setpix1d_inimname);
    resolveIMGID(&in,   ERRMODE_ABORT, dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START  fpsexec(in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END  return RETURN_SUCCESS;
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
CLIADDCMD_COREMOD_arith__imset_1Dpixrange()
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

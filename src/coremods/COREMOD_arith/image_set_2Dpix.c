/**
 * @file    image_set_2Dpix.c
 * @brief   Set a single pixel value in a 2D image
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
    .fps_name    = "setpix",
    .cmdkey      = "setpix",
    .description = "set image pixel value",
    .description_long =
        "Set the value of a single pixel at coordinates (x, y) in a 2D image stream. Useful for injecting test signals or modifying individual pixel values in shared memory."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *setpix_inimname = NULL;
static float    *setpix_pixval   = NULL;
static uint32_t *setpix_colindex = NULL;
static uint32_t *setpix_rowindex = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imname", &setpix_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".pixval", &setpix_pixval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "pixel value") \
    X(".col", &setpix_colindex, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "col index") \
    X(".row", &setpix_rowindex, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "row index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    if (!setpix_pixval || !setpix_colindex
        || !setpix_rowindex)
    {
        return RETURN_FAILURE;
    }
    float    val = *setpix_pixval;
    uint32_t col = *setpix_colindex;
    uint32_t row = *setpix_rowindex;
    uint32_t xsize = inimg->md[0].size[0];

    if (col >= xsize
        || row >= inimg->md[0].size[1])
    {
        return RETURN_FAILURE;
    }
#define SET2D_CASE_(DT, ACC, CT)                    \
    case DT:                                        \
        inimg->array.ACC[row * xsize + col] =       \
            (CT) val;                               \
        break;

    switch (inimg->md[0].datatype) {
        FOREACH_REAL_DATATYPE(SET2D_CASE_) default: PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
#undef SET2D_CASE_
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
    IMGID in = imgid_make_from_name(setpix_inimname);
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

errno_t CLIADDCMD_COREMOD_arith__imset_2Dpix()
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
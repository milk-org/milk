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


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "setpix1D",
    .cmdkey      = "setpix1Drange",
    .description =
        "set image pixel value over range"
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

static errno_t fpsexec(IMAGE *inimg)
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
    switch (inimg->md[0].datatype) {
    case _DATATYPE_UINT8:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.UI8[i] =
                (uint8_t) val;
        }
        break;
    case _DATATYPE_INT8:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.SI8[i] =
                (int8_t) val;
        }
        break;
    case _DATATYPE_UINT16:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.UI16[i] =
                (uint16_t) val;
        }
        break;
    case _DATATYPE_INT16:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.SI16[i] =
                (int16_t) val;
        }
        break;
    case _DATATYPE_UINT32:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.UI32[i] =
                (uint32_t) val;
        }
        break;
    case _DATATYPE_INT32:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.SI32[i] =
                (int32_t) val;
        }
        break;
    case _DATATYPE_UINT64:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.UI64[i] =
                (uint64_t) val;
        }
        break;
    case _DATATYPE_INT64:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.SI64[i] =
                (int64_t) val;
        }
        break;
    case _DATATYPE_FLOAT:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.F[i] = val;
        }
        break;
    case _DATATYPE_DOUBLE:
        for (uint32_t i = mi; i < ma; i++) {
            inimg->array.D[i] =
                (double) val;
        }
        break;
    default:
        PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(
            setpix1d_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    fpsexec(in.im);
    processinfo_update_output_stream(
        processinfo, in.im, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_arith__imset_1Dpixrange()
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

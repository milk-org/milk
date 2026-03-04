/**
 * @file    image_set_col.c
 * @brief   Set image column pixels to a value
 *
 * Sets all pixels in a specified column of an image
 * stream to a given value.
 * Uses FPS V2 framework.
 */

#include "CLIcore.h"
#include "fps.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "setcol",
    .cmdkey      = "setcol",
    .description = "set image column pixel values"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *setcol_inimname = NULL;
static float    *setcol_pixval   = NULL;
static uint32_t *setcol_colindex = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imname", &setcol_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".pixval", &setcol_pixval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "pixel value") \
    X(".col", &setcol_colindex, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "column index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static errno_t fpsexec(IMAGE *inimg)
{
    if (!setcol_pixval || !setcol_colindex) {
        return RETURN_FAILURE;
    }
    float    val = *setcol_pixval;
    uint32_t col = *setcol_colindex;
    uint32_t xsize = inimg->md[0].size[0];
    uint32_t ysize = inimg->md[0].size[1];

    if (col >= xsize) {
        return RETURN_FAILURE;
    }
    switch (inimg->md[0].datatype) {
    case _DATATYPE_FLOAT:
        for (uint32_t j = 0; j < ysize; j++) {
            inimg->array.F[
                j * xsize + col] = val;
        }
        break;
    case _DATATYPE_DOUBLE:
        for (uint32_t j = 0; j < ysize; j++) {
            inimg->array.D[
                j * xsize + col] =
                (double) val;
        }
        break;
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
        imgid_make_from_name(setcol_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);

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

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_COREMOD_arith__imset_col()
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

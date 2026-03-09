/**
 * @file    image_arith__im_f_f__im.c
 * @brief   arith functions
 *
 * input : image, float, float
 * output: image
 *
 * Uses FPS V2 framework.
 */

#include <math.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__im_f_f__im.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imtrunc",
    .cmdkey      = "imtrunc",
    .description =
        "truncate pixel values between min and max"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   *inimname;
static double *valmin;
static double *valmax;
static char   *outimname;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".min", &valmin, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "min value") \
    X(".max", &valmax, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "max value") \
    X(".out_name", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

int arith_image_trunc_IMGID(IMGID *imgin,
                            double f1,
                            double f2,
                            IMGID *imgout)
{
    arith_image_function_1ff_1_IMGID(imgin, f1, f2, imgout, &Ptrunc);
    return (0);
}

int arith_image_trunc(const char *ID_name,
                      double      f1,
                      double      f2,
                      const char *ID_out)
{
    IMGID imgin  = imgid_make_from_name(ID_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    arith_image_trunc_IMGID(&imgin, f1, f2, &imgout);

    return (0);
}

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2)
{
    arith_image_function_1ff_1_inplace(ID_name, f1, f2, &Ptrunc);
    return (0);
}

int arith_image_trunc_inplace_byID(long ID, double f1, double f2)
{
    arith_image_function_1ff_1_inplace_byID(ID, f1, f2, &Ptrunc);
    return (0);
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
    IMGID imgin  = imgid_make_from_name(inimname);
    IMGID imgout = imgid_make_from_name(outimname);

    arith_image_trunc_IMGID(&imgin, *valmin, *valmax, &imgout);

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

errno_t image_arith__im_f_f__im_addCLIcmd()
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

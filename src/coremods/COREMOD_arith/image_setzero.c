/**
 * @file    image_setzero.c
 * @brief   Set all image pixels to zero
 *
 * Sets every element of a stream to zero.
 * Uses FPS V2 framework following the POC pattern
 * from examplefunc_fps_cli_poc.c.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imzero",
    .cmdkey      = "imzero",
    .description = "set all image pixels to zero",
    .description_long =
        "Zero-fill all pixel values in the target image stream. Operates in-place on the existing shared memory buffer, setting every element to 0."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char imsetzero_imname[
    FUNCTION_PARAMETER_STRMAXLEN] = "stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 *
 * Syntax: X(keyword, ptr, type, is_primary, flag, descr)
 * ============================================================= */

#define IMSETZERO_PARAMS(X) \
    X(".imname", imsetzero_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Zero all pixels in the resolved image.
 *
 * Called from within the processinfo loop. The stream
 * is resolved before the loop starts.
 */
static errno_t imsetzero_computation(
    IMAGE *inimg
)
{
    memset(
        inimg->array.raw, 0,
        ImageStreamIO_typesize(
            inimg->md[0].datatype)
        * inimg->md[0].nelement);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 *
 * Must appear before compute_function() because the
 * INSERT_STD_PROCINFO macros reference CLIcmddata.
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    IMSETZERO_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    IMSETZERO_PARAMS(FPS_X_FARG)
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

/*
 * Copy key and description from FPS_app_info so
 * they are defined in one place only.
 * Also provide a valid cmdsettings object.
 */
FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 * ============================================================= */

/**
 * @brief Wraps imsetzero_computation() in the standard
 *        processinfo loop macros.
 *
 * Resolves the stream IMGID before the loop, then
 * calls the computation and updates the output stream
 * on each iteration.
 */
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    IMGID in =
        imgid_make_from_name(imsetzero_imname);
    resolveIMGID(
        &in,   ERRMODE_ABORT,
        dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    imsetzero_computation(in.im);
    processinfo_update_output_stream(
        processinfo, in.im, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
/**
 * @brief Unified milk CLI entry point.
 */
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info,
        farg,
        &CLIcmddata,
        my_bindings,
        nb_bindings,
        compute_function);
}

/**
 * @brief Module registration function.
 */
errno_t CLIADDCMD_COREMOD_arith__imsetzero()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 *
 * FPS_MAIN_STANDALONE_V2 generates main() using
 * the generic library lifecycle functions.
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    IMSETZERO_PARAMS,
    compute_function)
#endif

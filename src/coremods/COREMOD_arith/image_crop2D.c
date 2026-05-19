/**
 * @file    image_crop2D.c
 * @brief   Crop a 2D rectangular region from stream
 *
 * Uses FPS V2 framework.
 */

#include <stdlib.h>
#include <string.h>

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
    .fps_name    = "crop2D",
    .cmdkey      = "crop2D",
    .description = "crop 2D image",
    .description_long =
        "Extract a rectangular sub-region from a 2D image stream. Specify origin coordinates (x0, y0) and output dimensions (xsize, ysize). The cropped output is written to a shared memory stream."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     cropinsname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     outsname[FUNCTION_PARAMETER_STRMAXLEN]    = "out";
static uint32_t cropxstart  = 0;
static uint32_t cropxsize   = 100;
static uint32_t cropystart  = 0;
static uint32_t cropysize   = 100;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".insname", cropinsname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, \
      "Input stream name") \
    X(".outsname", outsname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "Output stream name") \
    X(".cropxstart", &cropxstart, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "crop x coord start") \
    X(".cropxsize", &cropxsize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "crop x coord size") \
    X(".cropystart", &cropystart, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "crop y coord start") \
    X(".cropysize", &cropysize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "crop y coord size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(
    IMAGE *input_image,
    IMAGE *output_image)
{
    uint32_t xs = cropxstart;
    uint32_t xw = cropxsize;
    uint32_t ys = cropystart;
    uint32_t yw = cropysize;
    uint32_t iw = input_image->md[0].size[0];
    uint32_t ih = input_image->md[0].size[1];
    size_t ts = ImageStreamIO_typesize(
        input_image->md[0].datatype);

    for (uint32_t j = 0; j < yw; j++) {
        uint32_t oj = j + ys;
        if (oj >= ih) {
            continue;
        }
        __builtin_memcpy(
            ((char *)
             output_image->array.raw)
            + j * xw * ts,
            ((char *)
             input_image->array.raw)
            + (oj * iw + xs) * ts,
            xw * ts);
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Validate crop parameters against
 *        input stream dimensions.
 */
static errno_t __attribute__((unused)) crop2D_validate()
{
    IMAGE im;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(
            cropinsname, &im) == 0)
    {
        uint32_t w = im.md[0].size[0];
        uint32_t h = im.md[0].size[1];
        if (cropxstart + cropxsize > w) {
            if (cropxstart >= w) {
                cropxstart = 0;
            }
            if (cropxstart + cropxsize > w) {
                cropxsize = w - cropxstart;
            }
        }
        if (cropystart + cropysize > h) {
            if (cropystart >= h) {
                cropystart = 0;
            }
            if (cropystart + cropysize > h) {
                cropysize = h - cropystart;
            }
        }
        ImageStreamIO_closeIm(&im);
    }
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
    IMGID iin =
        imgid_make_from_name(cropinsname);
    resolveIMGID(
        &iin,  ERRMODE_ABORT,
        dcimg, dcnimg);
    IMGID iout = stream_connect_create_2D(
        outsname, cropxsize, cropysize,
        iin.md->datatype);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    fpsexec(iin.im, iout.im);
    processinfo_update_output_stream(
        processinfo, iout.im, iin.im);

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

errno_t CLIADDCMD_COREMODE_arith__crop2D()
{
    CLIcmddata.FPS_customCONFcheck =
        crop2D_validate;
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
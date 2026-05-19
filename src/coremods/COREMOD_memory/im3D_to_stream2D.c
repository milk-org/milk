/**
 * @file    im3D_to_stream2D.c
 * @brief   convert 3D image to 2D stream
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


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "im3Dto2D",
    .cmdkey      = "im3D_to_stream2D",
    .description = "convert 3D image to 2D stream",
    .description_long =
        "Convert a static 3D image cube into a 2D shared memory stream by sequentially playing back each slice as a frame. Useful for testing stream consumers with pre-recorded data."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    inimname[FUNCTION_PARAMETER_STRMAXLEN] = "im3d";
static char    outname[FUNCTION_PARAMETER_STRMAXLEN] = "im2d";
static int64_t slice_index = 0;
static int32_t loop_mode   = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input 3D image") \
    X(".outname", outname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output stream name") \
    X(".slice_index", &slice_index, \
      FPTYPE_INT64, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "initial slice index") \
    X(".loop_mode", &loop_mode, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "loop through slices")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Extract 2D slice from 3D image.
 */
static errno_t extract_slice_to_2D(
    IMGID *inimg,
    IMGID *outimg,
    long  slice_idx)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(inimg, ERRMODE_ABORT, dcimg, dcnimg);

    if(inimg->md->naxis != 3)
    {
        FUNC_RETURN_FAILURE("Input image is not 3D");
    }

    uint32_t xsize = inimg->mdt->size[0];
    uint32_t ysize = inimg->mdt->size[1];
    uint32_t zsize = inimg->mdt->size[2];

    if(slice_idx < 0 || slice_idx >= zsize)
    {
        FUNC_RETURN_FAILURE("Slice index out of bounds");
    }

    // Image is created once before the loop;
    // do not allocate inside the hot path.

    outimg->md->write = 1;

    long framesize = xsize * ysize * ImageStreamIO_typesize(inimg->md->datatype);

    __builtin_memcpy(outimg->im->array.raw,
           inimg->im->array.raw + slice_idx * framesize, framesize);

    DEBUG_TRACE_FEXIT();
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
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg,  dcnimg);

    IMGID outimg;
    outimg = imgid_make_from_name_2D(outname, inimg.mdt->size[0], inimg.mdt->size[1]);
    outimg.mdt->shared = 1;
    outimg.mdt->datatype = inimg.md->datatype;

    // Allocate output image once before the loop.
    imcreateIMGID(&outimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        if(loop_mode == 0)
        {
            extract_slice_to_2D(&inimg, &outimg, slice_index);
        }
        else
        {
            slice_index = (slice_index + 1) % inimg.mdt->size[2];
            extract_slice_to_2D(&inimg, &outimg, slice_index);
        }

        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END  imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
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
CLIADDCMD_COREMOD_memory__im3D_to_stream2D()
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
/**
 * @file    cubecollapse.c
 * @brief   Collapse a cube along z axis
 *
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "cubecollapse.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "cubecollapse",
    .cmdkey           = "cubecollapse",
    .description      = "collapse a cube along z",
    .description_long = "Collapse a 3D image cube along the z-axis by computing the mean, median, "
                        "or sum of all slices. Produces a single 2D output image."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char cubecollapse_inimname[FUNCTION_PARAMETER_STRMAXLEN];
static char cubecollapse_outimname[FUNCTION_PARAMETER_STRMAXLEN];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".in_name", cubecollapse_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,   \
      "input cube image")                                                              \
    X(".out_name", cubecollapse_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, \
      "output 2D image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void cube_collapse_step(IMAGE *imgin, IMAGE *imgout)
{
    uint32_t xsize = imgin->md[0].size[0];
    uint32_t ysize = imgin->md[0].size[1];
    uint32_t ksize = imgin->md[0].size[2];
    for (uint32_t i = 0; i < xsize * ysize; i++)
    {
        float v = 0.0;
        for (uint32_t k = 0; k < ksize; k++)
        {
            v += imgin->array.F[k * xsize * ysize + i];
        }
        imgout->array.F[i] = v;
    }
}


/* =========================================
 * Public convenience function
 * ========================================= */

#ifndef FPS_STANDALONE
imageID cube_collapse(const char *ID_in_name, const char *ID_out_name)
{
    IMGID in = imgid_make_from_name(ID_in_name);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(ID_out_name, in.md->size[0], in.md->size[1]);
    cube_collapse_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im);
    return out.ID;
}
#endif


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t compute_function()
{
    IMGID in = imgid_make_from_name(cubecollapse_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(cubecollapse_outimname, in.md->size[0], in.md->size[1]);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    cube_collapse_step(in.im, out.im);
    processinfo_update_output_stream(processinfo, out.im, in.im);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t __attribute__((cold)) CLIADDCMD_image_basic__cubecollapse()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif

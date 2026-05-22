/**
 * @file    imrotate.c
 * @brief   Rotate 2D image
 *
 * Uses FPS V2 framework.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "imrotate.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "rotateim",
    .cmdkey           = "rotateim",
    .description      = "rotate 2D image",
    .description_long = "Rotate a 2D image by an arbitrary angle using bilinear interpolation. "
                        "Rotation center and angle are configurable."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char  imrotate_inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char  imrotate_outimname[FUNCTION_PARAMETER_STRMAXLEN] = "out";
static float imrotate_angle                                   = 0.0f;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".in_name", imrotate_inimname, FPTYPE_STREAMNAME, 1,                                         \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "input image")                                 \
    X(".out_name", imrotate_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".angle", &imrotate_angle, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "rotate angle")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void imrotate_step(IMAGE *imgin, IMAGE *imgout, float angle)
{
    uint32_t nx = imgin->md[0].size[0];
    uint32_t ny = imgin->md[0].size[1];
    float    c  = cosf(angle);
    float    s  = sinf(angle);
    for (uint32_t jj = 0; jj < ny; jj++)
    {
        for (uint32_t ii = 0; ii < nx; ii++)
        {
            long iis = (long) (nx / 2 + (ii - (int) nx / 2) * c + (jj - (int) ny / 2) * s);
            long jjs = (long) (ny / 2 - (ii - (int) nx / 2) * s + (jj - (int) ny / 2) * c);
            if ((iis >= 0) && (jjs >= 0) && (iis < (long) nx) && (jjs < (long) ny))
            {
                imgout->array.F[jj * nx + ii] = imgin->array.F[jjs * nx + iis];
            }
            else
            {
                imgout->array.F[jj * nx + ii] = 0.0;
            }
        }
    }
}


/* =========================================
 * Public convenience function
 * ========================================= */

#ifndef FPS_STANDALONE
imageID basic_rotate(const char *ID_name, const char *IDout_name, float angle)
{
    IMGID in = imgid_make_from_name(ID_name);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(IDout_name, in.md->size[0], in.md->size[1]);
    imrotate_step(in.im, out.im, angle);
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
    IMGID in = imgid_make_from_name(imrotate_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(imrotate_outimname, in.md->size[0], in.md->size[1]);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    imrotate_step(in.im, out.im, imrotate_angle);
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

errno_t CLIADDCMD_image_basic__imrotate()
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

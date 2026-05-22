/**
 * @file    imresize.c
 * @brief   Resize 2D image
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
#include "imresize.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "resizeim",
    .cmdkey           = "resizeim",
    .description      = "resize 2D image",
    .description_long = "Resize a 2D image to arbitrary dimensions using bilinear interpolation."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    imresize_inimname[FUNCTION_PARAMETER_STRMAXLEN];
static char    imresize_outimname[FUNCTION_PARAMETER_STRMAXLEN];
static int64_t imresize_xsize = 64;
static int64_t imresize_ysize = 64;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".in_name", imresize_inimname, FPTYPE_STREAMNAME, 1,                                         \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "input image")                                 \
    X(".out_name", imresize_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".xsize", &imresize_xsize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "new x size")              \
    X(".ysize", &imresize_ysize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "new y size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void imresize_step(IMAGE *imgin, IMAGE *imgout)
{
    uint32_t nx_in  = imgin->md[0].size[0];
    uint32_t ny_in  = imgin->md[0].size[1];
    uint32_t nx_out = imgout->md[0].size[0];
    uint32_t ny_out = imgout->md[0].size[1];

    if (imgin->md[0].datatype != _DATATYPE_FLOAT)
    {
        return;
    }

    for (uint32_t ii = 0; ii < nx_out; ii++)
    {
        for (uint32_t jj = 0; jj < ny_out; jj++)
        {
            float xf1 = (float) ii * nx_in / nx_out;
            float yf1 = (float) jj * ny_in / ny_out;
            long  ii1 = (long) xf1;
            long  jj1 = (long) yf1;
            float uf  = xf1 - (float) ii1;
            float tf  = yf1 - (float) jj1;

            if ((ii1 >= 0) && (ii1 + 1 < (long) nx_in) && (jj1 >= 0) && (jj1 + 1 < (long) ny_in))
            {
                float v00                         = imgin->array.F[jj1 * nx_in + ii1];
                float v01                         = imgin->array.F[(jj1 + 1) * nx_in + ii1];
                float v10                         = imgin->array.F[jj1 * nx_in + ii1 + 1];
                float v11                         = imgin->array.F[(jj1 + 1) * nx_in + ii1 + 1];
                imgout->array.F[jj * nx_out + ii] = v00 * (1 - uf) * (1 - tf) +
                                                    v10 * uf * (1 - tf) + v01 * (1 - uf) * tf +
                                                    v11 * uf * tf;
            }
            else
            {
                imgout->array.F[jj * nx_out + ii] = 0.0;
            }
        }
    }
}


/* =========================================
 * Public convenience function
 * ========================================= */

#ifndef FPS_STANDALONE
long basic_resizeim(const char *imname_in, const char *imname_out, long xsizeout, long ysizeout)
{
    IMGID in = imgid_make_from_name(imname_in);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(imname_out, xsizeout, ysizeout);
    imresize_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im);
    return 0;
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
    IMGID in = imgid_make_from_name(imresize_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(imresize_outimname, imresize_xsize, imresize_ysize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    imresize_step(in.im, out.im);
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

errno_t CLIADDCMD_image_basic__imresize()
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

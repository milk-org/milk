/**
 * @file    image_arith__im_f_f__im.c
 * @brief   arith functions
 *
 * input : image, float, float
 * output: image
 *
 * Uses FPS V2 framework.
 */


#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
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
    .description = "truncate pixel values between min and max",
    .description_long =
        "Truncate pixel values in an image stream by clamping them to a specified range [min, "
        "max]. Pixels below min are set to min, pixels above max are set to max. Useful for "
        "filtering outliers or enforcing dynamic range limits in real-time streams."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   inimname[FUNCTION_PARAMETER_STRMAXLEN];
static double valmin;
static double valmax;
static char   outimname[FUNCTION_PARAMETER_STRMAXLEN];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, \
      "input image")                                                                            \
    X(".min", &valmin, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "min value")                    \
    X(".max", &valmax, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "max value")                    \
    X(".out_name", outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_OUTPUT, "output image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

int arith_image_trunc_IMGID(IMGID *imgin, double f1, double f2, IMGID *imgout)
{
    arith_image_function_1ff_1_IMGID(imgin, f1, f2, imgout, &Ptrunc);
    return (0);
}

int arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out)
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


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    IMGID imgin = imgid_make_from_name(inimname);
    resolveIMGID(&imgin, ERRMODE_NULL, dcimg, dcnimg);

    if (imgin.im == NULL)
    {
        return RETURN_FAILURE;
    }

    IMGID imgout = imgid_make_from_name(outimname);
    imgid_copy(&imgin, &imgout);
    imcreateIMGID(&imgout);

    if (imgout.im == NULL)
    {
        imgid_free(&imgin);
        return RETURN_FAILURE;
    }

    uint64_t nelement = imgin.md->nelement;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        if (imgin.md->datatype == _DATATYPE_FLOAT && imgout.mdt->datatype == _DATATYPE_FLOAT)
        {
            float *MILK_RESTRICT pin   = MILK_ASSUME_ALIGNED(imgin.im->array.F);
            float *MILK_RESTRICT pout  = MILK_ASSUME_ALIGNED(imgout.im->array.F);
            float                f_min = (float) valmin;
            float                f_max = (float) valmax;

#pragma omp simd
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                float v = pin[ii];
                if (v < f_min)
                {
                    v = f_min;
                }
                else if (v > f_max)
                {
                    v = f_max;
                }
                pout[ii] = v;
            }
        }
        else if (imgin.md->datatype == _DATATYPE_DOUBLE && imgout.mdt->datatype == _DATATYPE_DOUBLE)
        {
            double *MILK_RESTRICT pin   = MILK_ASSUME_ALIGNED(imgin.im->array.D);
            double *MILK_RESTRICT pout  = MILK_ASSUME_ALIGNED(imgout.im->array.D);
            double                d_min = valmin;
            double                d_max = valmax;

#pragma omp simd
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                double v = pin[ii];
                if (v < d_min)
                {
                    v = d_min;
                }
                else if (v > d_max)
                {
                    v = d_max;
                }
                pout[ii] = v;
            }
        }
        else
        {
            arith_image_trunc_IMGID(&imgin, valmin, valmax, &imgout);
        }

        processinfo_update_output_stream(processinfo, imgout.im, imgin.im);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END imgid_free(&imgin);
    imgid_free(&imgout);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t image_arith__im_f_f__im_addCLIcmd()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif

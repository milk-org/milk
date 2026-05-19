/**
 * @file    image_vecmult.c
 * @brief   multiply image by vector
 *
 */
#include "ImageStreamIO/ImageStruct.h"

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imvecmult",
    .cmdkey      = "imvecmult",
    .description = "multiply image by vector",
    .description_long =
        "Multiply each slice of a 3D image cube by the corresponding element of a 1D vector. Applies element-wise scaling across the z-axis. Useful for applying modal coefficients to a mode cube."
};

static char iminname[
    FUNCTION_PARAMETER_STRMAXLEN];
static char vecname[
    FUNCTION_PARAMETER_STRMAXLEN];
static char imoutname[
    FUNCTION_PARAMETER_STRMAXLEN];
static uint32_t multaxis = 0;


#define FPS_PARAMS(X) \
    X(".iminname", iminname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image name") \
    X(".vecname", vecname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input vector name") \
    X(".imoutname", imoutname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".axis", &multaxis, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "multiplication axis")


static MILK_COLD errno_t __attribute__((unused)) customCONFsetup()
{
    if(dcfpsptr != NULL)
    {
        long fpi = functionparameter_GetParamIndex(dcfpsptr, ".iminname");
        if(fpi >= 0)
        {
            dcfpsptr->parray[fpi].fpflag |=
                FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
        }
    }

    return RETURN_SUCCESS;
}

static MILK_COLD errno_t __attribute__((unused)) customCONFcheck()
{
    return RETURN_SUCCESS;
}


FPS_V2_SECTION5(FPS_PARAMS)


errno_t image_vect_multiply(
    IMGID imgin,
    IMGID imgvec,
    IMGID *imgout,
    uint32_t multaxis
)
{
    DEBUG_TRACE_FSTART();

    // check if images already exist
    //
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&imgvec, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1) {
        return RETURN_FAILURE;
    }
    if (imgvec.ID == -1) {
        return RETURN_FAILURE;
    }

    resolveIMGID(imgout, ERRMODE_NULL, dcimg, dcnimg);

    // Create output
    //
    if( (*imgout).ID == -1 )
    {
        imcreatelikewiseIMGID(
            imgout,
            &imgin
        );
    }

    uint32_t size0 = imgin.md->size[0];
    if (size0 == 0)
    {
        size0 = 1;
    }

    uint32_t size1 = imgin.md->size[1];
    if (size1 == 0)
    {
        size1 = 1;
    }

    uint32_t size2 = imgin.md->size[2];
    if (size2 == 0)
    {
        size2 = 1;
    }


    float * MILK_RESTRICT ptr_in = MILK_ASSUME_ALIGNED(imgin.im->array.F);
    float * MILK_RESTRICT ptr_v  = MILK_ASSUME_ALIGNED(imgvec.im->array.F);
    float * MILK_RESTRICT ptr_out= MILK_ASSUME_ALIGNED(imgout->im->array.F);

    uint64_t xsize = size0;
    uint64_t ysize = size1;
    uint64_t zsize = size2;
    uint64_t xysize = xsize * ysize;

    if (multaxis == 0)
    {
        _Pragma("omp parallel for")
        for (uint64_t kk = 0; kk < zsize; kk++)
        {
            for (uint64_t jj = 0; jj < ysize; jj++)
            {
                uint64_t offset = kk * xysize + jj * xsize;
                _Pragma("omp simd")
                for (uint64_t ii = 0; ii < xsize; ii++)
                {
                    ptr_out[offset + ii] = ptr_in[offset + ii] * ptr_v[ii];
                }
            }
        }
    }
    else if (multaxis == 1)
    {
        _Pragma("omp parallel for")
        for (uint64_t kk = 0; kk < zsize; kk++)
        {
            for (uint64_t jj = 0; jj < ysize; jj++)
            {
                uint64_t offset = kk * xysize + jj * xsize;
                float v = ptr_v[jj];
                _Pragma("omp simd")
                for (uint64_t ii = 0; ii < xsize; ii++)
                {
                    ptr_out[offset + ii] = ptr_in[offset + ii] * v;
                }
            }
        }
    }
    else if (multaxis == 2)
    {
        _Pragma("omp parallel for")
        for (uint64_t kk = 0; kk < zsize; kk++)
        {
            float v = ptr_v[kk];
            for (uint64_t jj = 0; jj < ysize; jj++)
            {
                uint64_t offset = kk * xysize + jj * xsize;
                _Pragma("omp simd")
                for (uint64_t ii = 0; ii < xsize; ii++)
                {
                    ptr_out[offset + ii] = ptr_in[offset + ii] * v;
                }
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    // input

    IMGID imgimin = imgid_make_from_name(iminname);
    resolveIMGID(&imgimin, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imgvec = imgid_make_from_name(vecname);
    if (imgimin.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgvec, ERRMODE_WARN, dcimg, dcnimg);

    // output

    IMGID imgout  = imgid_make_from_name(imoutname);
    if (imgvec.ID == -1) {
        return RETURN_FAILURE;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        image_vect_multiply(imgimin, imgvec, &imgout, multaxis);
        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__image_vecmult()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    FPS_PARAMS,
    compute_function,
    customCONFcheck)
#endif

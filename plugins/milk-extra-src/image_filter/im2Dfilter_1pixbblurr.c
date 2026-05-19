#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    im2Dfilter_1pixbblurr.c
 * @brief   Apply 1 pixel radius blurr to image
 *
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include "CLIcore.h"
#endif


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "im2Dfilt1pblurr",
    .cmdkey      = "im2Dfilt1pblurr",
    .description = "1 pixel radual blurr, can be iterated",
    .description_long =
        "Apply a 1-pixel box blur to a 2D image. Can be iterated multiple times to approximate a Gaussian blur via the central limit theorem."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * iminname = NULL;
static char * imoutname = NULL;
static float * blurramp = NULL;
static uint32_t * NBloop = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".iminname", &iminname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image name") \
    X(".imoutname", &imoutname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image name") \
    X(".blurramp", &blurramp, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "value of side pixs (total = 1 for 3 pix)") \
    X(".axis", &NBloop, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "number of times operation is performed")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

static errno_t imfilter_im2D_1pixblurr(
    IMGID imgin,
    IMGID *imgout,
    float amp,
    long NBiter
)
{
    DEBUG_TRACE_FSTART();
    // custom stream process function code

    // resolve imgpos
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);


    // create eigenvalues array if needed
    if( imgout->ID == -1)
    {
        imgout->mdt->naxis   = 2;
    if (imgin.ID == -1) {
        return RETURN_FAILURE;
    }
        imgout->mdt->size[0] = imgin.md->size[0];
        imgout->mdt->size[1] = imgin.md->size[1];
        imgout->mdt->shared = imgin.md->shared;
        imgout->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgout);
    }


    uint32_t xsize = imgin.md->size[0];
    uint32_t ysize = imgin.md->size[1];


    float coeff1 = amp; // side pixels (x4)
    float coeff2 = amp*amp; // corner pixels (x4)
    float coeff0 = 1.0 - 4.0*(coeff1 + coeff2); // central pixel


    // temp arrays
    float *tmpfim0 = (float*) malloc(sizeof(float) * xsize * ysize);
    float *tmpfim1 = (float*) malloc(sizeof(float) * xsize * ysize);


    // copy input to tmpfim0
    //
    switch (imgin.md->datatype)
    {
    case _DATATYPE_FLOAT:
        memcpy(tmpfim0, imgin.im->array.F, sizeof(float)*xsize*ysize);
        break;

    case _DATATYPE_DOUBLE:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.D[ii];
        }
        break;

    case _DATATYPE_UINT8:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI8[ii];
        }
        break;

    case _DATATYPE_UINT16:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI16[ii];
        }
        break;

    case _DATATYPE_UINT32:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI32[ii];
        }
        break;

    case _DATATYPE_UINT64:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI64[ii];
        }
        break;

    case _DATATYPE_INT8:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI8[ii];
        }
        break;

    case _DATATYPE_INT16:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI16[ii];
        }
        break;

    case _DATATYPE_INT32:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI32[ii];
        }
        break;

    case _DATATYPE_INT64:
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI64[ii];
        }
        break;
    }


    for ( int iter=0; iter < NBiter; iter++)
    {
        memset(tmpfim1, 0, sizeof(float)*xsize*ysize);

        _Pragma("omp parallel for")
        for(uint32_t jj=1; jj<ysize-1; jj++)
        {
            _Pragma("omp simd")
            for(uint32_t ii=1; ii<xsize-1; ii++)
            {
                float val = 0.0f;
                // central
                val += coeff0 * tmpfim0[jj*xsize + ii];

                // sides
                val += coeff1 * tmpfim0[jj*xsize + ii + 1];
                val += coeff1 * tmpfim0[jj*xsize + ii - 1];
                val += coeff1 * tmpfim0[(jj+1)*xsize + ii];
                val += coeff1 * tmpfim0[(jj-1)*xsize + ii];

                // corners
                val += coeff2 * tmpfim0[(jj+1)*xsize + ii + 1];
                val += coeff2 * tmpfim0[(jj+1)*xsize + ii - 1];
                val += coeff2 * tmpfim0[(jj-1)*xsize + ii + 1];
                val += coeff2 * tmpfim0[(jj-1)*xsize + ii - 1];

                tmpfim1[jj*xsize + ii] = val;
            }
        }
        memcpy(tmpfim0, tmpfim1, sizeof(float)*xsize*ysize);
    }

    memcpy(imgout->im->array.F, tmpfim0, sizeof(float)*xsize*ysize);

    free(tmpfim0);
    free(tmpfim1);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // input
    IMGID imgin = imgid_make_from_name(iminname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);

    // output
    IMGID imgout  = imgid_make_from_name(imoutname);
    if (imgin.ID == -1) {
        return RETURN_FAILURE;
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        imfilter_im2D_1pixblurr(imgin, &imgout, *blurramp, *NBloop);

        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgout);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_filter__im2Dfilter_1pixblurr()
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


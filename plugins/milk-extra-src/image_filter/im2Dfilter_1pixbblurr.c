#include "ImageStreamIO/ImageStruct.h"
/**
 * @file    im2Dfilter_1pixbblurr.c
 * @brief   Apply 1 pixel radius blurr to image
 *
 */

#include "CLIcore.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "im2Dfilt1pblurr",
    .cmdkey      = "im2Dfilt1pblurr",
    .description = "1 pixel radual blurr, can be iterated"
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
    "", "", CLICMD_FIELDS_DEFAULTS
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
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);


    // create eigenvalues array if needed
    if( imgout->ID == -1)
    {
        imgout->mdt->naxis   = 2;
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
        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.F[ii];
        }
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

        for(uint32_t ii=0; ii<xsize*ysize; ii++)
        {
            tmpfim1[ii] = 0.0;
        }

        for(uint32_t ii=1; ii<xsize-1; ii++)
        {
            for(uint32_t jj=1; jj<ysize-1; jj++)
            {
                float pixval = tmpfim0[jj*xsize+ii];

                tmpfim1[ (jj)*xsize + ii ] += coeff0 * pixval;

                tmpfim1[ (jj)*xsize + ii+1 ] += coeff1 * pixval;
                tmpfim1[ (jj)*xsize + ii-1 ] += coeff1 * pixval;
                tmpfim1[ (jj+1)*xsize + ii ] += coeff1 * pixval;
                tmpfim1[ (jj-1)*xsize + ii ] += coeff1 * pixval;


                tmpfim1[ (jj+1)*xsize + ii+1 ] += coeff2 * pixval;
                tmpfim1[ (jj+1)*xsize + ii-1 ] += coeff2 * pixval;
                tmpfim1[ (jj-1)*xsize + ii+1 ] += coeff2 * pixval;
                tmpfim1[ (jj-1)*xsize + ii-1 ] += coeff2 * pixval;
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


static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // input
    IMGID imgin = imgid_make_from_name(iminname);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    // output
    IMGID imgout  = imgid_make_from_name(imoutname);


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
static errno_t CLIfunction(void)
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


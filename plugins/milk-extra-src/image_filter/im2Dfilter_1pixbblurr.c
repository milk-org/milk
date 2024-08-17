/**
 * @file    im2Dfilter_1pixbblurr.c
 * @brief   Apply 1 pixel radius blurr to image
 *
 */

#include "CommandLineInterface/CLIcore.h"



// Local variables pointers


static char *iminname;
static long fpi_iminname;

static char *imoutname;
static long fpi_imoutname;

static float *blurramp;
static long fpi_blurramp;

static uint32_t *NBloop;
static long fpi_NBloop;





static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".iminname",
        "input image name",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &iminname,
        &fpi_iminname
    },
    {
        CLIARG_STR,
        ".imoutname",
        "output image name",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imoutname,
        &fpi_imoutname
    },
    {
        CLIARG_FLOAT32,
        ".blurramp",
        "value of side pixs (total = 1 for 3 pix)",
        "1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &blurramp,
        &fpi_blurramp
    },
    {
        CLIARG_UINT32,
        ".axis",
        "number of times operation is performed",
        "1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &NBloop,
        &fpi_NBloop
    }
};




static errno_t customCONFsetup()
{

    return RETURN_SUCCESS;
}


static errno_t customCONFcheck()
{
    if(data.fpsptr != NULL)
    {
    }

    return RETURN_SUCCESS;
}


static CLICMDDATA CLIcmddata =
{
    "im2Dfilt1pblurr",
    "1 pixel radual blurr, can be iterated",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
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
    resolveIMGID(&imgin, ERRMODE_ABORT);


    // create eigenvalues array if needed
    if( imgout->ID == -1)
    {
        imgout->naxis   = 2;
        imgout->size[0] = imgin.size[0];
        imgout->size[1] = imgin.size[1];
        imgout->shared = imgin.shared;
        imgout->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgout);
    }


    uint32_t xsize = imgin.size[0];
    uint32_t ysize = imgin.size[1];


    float coeff1 = amp; // side pixels (x4)
    float coeff2 = amp*amp; // corner pixels (x4)
    float coeff0 = 1.0 - 4.0*(coeff1 + coeff2); // central pixel

    printf("amp = %f   NBiter = %d\n", amp, NBiter);



    // temp arrays
    float *tmpfim0 = (float*) malloc(sizeof(float) * xsize * ysize);
    float *tmpfim1 = (float*) malloc(sizeof(float) * xsize * ysize);


    // copy input to tmpfim0
    //
    switch (imgin.md->datatype)
    {
    case _DATATYPE_FLOAT:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.F[ii];
        }
        break;

    case _DATATYPE_DOUBLE:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.D[ii];
        }
        break;

    case _DATATYPE_UINT8:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI8[ii];
        }
        break;

    case _DATATYPE_UINT16:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI16[ii];
        }
        break;

    case _DATATYPE_UINT32:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI32[ii];
        }
        break;

    case _DATATYPE_UINT64:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.UI64[ii];
        }
        break;

    case _DATATYPE_INT8:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI8[ii];
        }
        break;

    case _DATATYPE_INT16:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI16[ii];
        }
        break;

    case _DATATYPE_INT32:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI32[ii];
        }
        break;

    case _DATATYPE_INT64:
        for(uint32_t ii=1; ii<xsize*ysize; ii++)
        {
            tmpfim0[ii] = imgin.im->array.SI64[ii];
        }
        break;
    }


    for ( int iter=0; iter < NBiter; iter++)
    {

        for(uint32_t ii=1; ii<xsize*ysize; ii++)
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
    IMGID imgin = mkIMGID_from_name(iminname);
    resolveIMGID(&imgin, ERRMODE_ABORT);

    // output
    IMGID imgout  = mkIMGID_from_name(imoutname);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        imfilter_im2D_1pixblurr(imgin, &imgout, *blurramp, *NBloop);

        processinfo_update_output_stream(processinfo, imgout.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_image_filter__im2Dfilter_1pixblurr()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

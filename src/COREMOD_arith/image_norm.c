#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"



// input image names
static char *inimname;

static char *outimname;


static uint32_t *sliceaxis;
static long      fpi_sliceaxis = -1;


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in0name",
        "input image 0",
        "im0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outname",
        "output image",
        "im0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_UINT32,
        ".axis",
        "norm axis",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &sliceaxis,
        &fpi_sliceaxis
    },
};




static CLICMDDATA CLIcmddata =
{
    "normslice",
    "image norm by slice",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf("Image norm for each slice of array\n");

    return RETURN_SUCCESS;
}




errno_t image_slicenorm(
    IMGID inimg,
    IMGID *outimg,
    uint8_t sliceaxis
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_ABORT);


    resolveIMGID(outimg, ERRMODE_NULL);
    if( outimg->ID == -1)
    {
        copyIMGID(&inimg, outimg);
    }

    for ( uint8_t axis = 0; axis < inimg.md->naxis; axis++ )
    {
        if ( axis != sliceaxis )
        {
            outimg->size[axis] = 1;
        }
    }
    outimg->datatype = _DATATYPE_FLOAT;

    createimagefromIMGID(outimg);


    uint32_t sizescan[3];
    sizescan[0] = inimg.md->size[0];
    sizescan[1] = inimg.md->size[1];
    sizescan[2] = inimg.md->size[2];
    if( inimg.md->naxis < 3 )
    {
        sizescan[2] = 1;
    }
    if( inimg.md->naxis < 2 )
    {
        sizescan[1] = 1;
    }



    double * __restrict normarray = (double*) malloc(sizeof(double) * sizescan[sliceaxis]);
    for( uint32_t ii=0; ii<inimg.md->size[sliceaxis]; ii++)
    {
        normarray[ii] = 0.0;
    }

    uint32_t pixcoord[3];

    for( uint32_t ii = 0; ii < sizescan[0]; ii++)
    {
        pixcoord[0] = ii;
        for( uint32_t jj = 0; jj < sizescan[1]; jj++)
        {
            pixcoord[1] = jj;
            for( uint32_t kk = 0; kk < sizescan[2]; kk++)
            {
                pixcoord[2] = kk;

                uint64_t pixi = kk * sizescan[1] * sizescan[0];
                pixi += jj * sizescan[0];
                pixi += ii;

                double val;
                switch ( inimg.datatype )
                {
                case _DATATYPE_UINT8 :
                    val = inimg.im->array.UI8[pixi] * inimg.im->array.UI8[pixi];
                    break;
                case _DATATYPE_INT8 :
                    val = inimg.im->array.SI8[pixi] * inimg.im->array.SI8[pixi];
                    break;
                case _DATATYPE_UINT16 :
                    val = inimg.im->array.UI16[pixi] * inimg.im->array.UI16[pixi];
                    break;
                case _DATATYPE_INT16 :
                    val = inimg.im->array.SI16[pixi] * inimg.im->array.SI16[pixi];
                    break;
                case _DATATYPE_UINT32 :
                    val = inimg.im->array.UI32[pixi] * inimg.im->array.UI32[pixi];
                    break;
                case _DATATYPE_INT32 :
                    val = inimg.im->array.SI32[pixi] * inimg.im->array.SI32[pixi];
                    break;
                case _DATATYPE_UINT64 :
                    val = inimg.im->array.UI64[pixi] * inimg.im->array.UI64[pixi];
                    break;
                case _DATATYPE_INT64 :
                    val = inimg.im->array.SI64[pixi] * inimg.im->array.SI64[pixi];
                    break;
                case _DATATYPE_FLOAT :
                    val = inimg.im->array.F[pixi] * inimg.im->array.F[pixi];
                    break;
                case _DATATYPE_DOUBLE :
                    val = inimg.im->array.D[pixi] * inimg.im->array.D[pixi];
                    break;
                }
                normarray[pixcoord[sliceaxis]] += val;
            }
        }
    }

    for( uint32_t ii=0; ii < sizescan[sliceaxis]; ii++ )
    {
        printf("morm %3u : %lf\n", ii, normarray[ii]);
        outimg->im->array.F[ii] = sqrt(normarray[ii]);
    }

    free(normarray);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);


    IMGID outimg = mkIMGID_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        image_slicenorm(
            inimg,
            &outimg,
            *sliceaxis
        );

        processinfo_update_output_stream(processinfo, outimg.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__image_normslice()
{
    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

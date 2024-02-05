#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"



// input image names
static char *inimname;
static char *maskimname;

static char *outimname;


static uint32_t *sliceaxis;
static long      fpi_sliceaxis = -1;

static char *auxin;

static uint64_t *modeRMS;



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
        CLIARG_IMG,
        ".maskim",
        "input image mask",
        "imm",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &maskimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outname",
        "output image",
        "im0n",
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
    {
        CLIARG_IMG,
        ".auxin",
        "auxillary input image, in-place update",
        "auxin",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &auxin,
        NULL
    },
    {
        CLIARG_ONOFF,
        ".RMS",
        "output RMS=1 over mask",
        "OFF",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &modeRMS,
        NULL
    }
};




static CLICMDDATA CLIcmddata =
{
    "normalizeslice",
    "image normalize over mask by slice",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf("Image norm for each slice of array\n");

    return RETURN_SUCCESS;
}




errno_t image_slicenormalize(
    IMGID inimg,
    IMGID maskimg,
    IMGID *outimg,
    uint8_t sliceaxis,
    IMGID imgaux,
    int modeRMS
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_ABORT);
    resolveIMGID(&maskimg, ERRMODE_ABORT);

    resolveIMGID(&imgaux, ERRMODE_NULL);


    resolveIMGID(outimg, ERRMODE_NULL);
    if(outimg->ID == -1)
    {
        copyIMGID(&inimg, outimg);
    }


    outimg->datatype = _DATATYPE_FLOAT;

    createimagefromIMGID(outimg);


    // input image
    //
    uint32_t sizescan[3];
    sizescan[0] = inimg.md->size[0];
    sizescan[1] = inimg.md->size[1];
    sizescan[2] = inimg.md->size[2];
    if(inimg.md->naxis < 3)
    {
        sizescan[2] = 1;
    }
    if(inimg.md->naxis < 2)
    {
        sizescan[1] = 1;
    }


    // aux input image
    //
    uint32_t auxsizescan[3];
    if(imgaux.ID != -1)
    {
        auxsizescan[0] = imgaux.md->size[0];
        auxsizescan[1] = imgaux.md->size[1];
        auxsizescan[2] = imgaux.md->size[2];
        if(imgaux.md->naxis < 3)
        {
            auxsizescan[2] = 1;
        }
        if(imgaux.md->naxis < 2)
        {
            auxsizescan[1] = 1;
        }
    }




    // mask image
    //
    uint32_t sizescanm[3];
    sizescanm[0] = sizescan[0];
    sizescanm[1] = sizescan[1];
    sizescanm[2] = sizescan[2];
    sizescanm[sliceaxis] = 1;

    uint32_t sizemmask[3];
    sizemmask[0] = 1;
    sizemmask[1] = 1;
    sizemmask[2] = 1;
    sizemmask[sliceaxis] = 0;




    double *__restrict normarray = (double *) malloc(sizeof(
                                       double) * sizescan[sliceaxis]);
    for(uint32_t ii = 0; ii < inimg.md->size[sliceaxis]; ii++)
    {
        normarray[ii] = 0.0;
    }

    double *__restrict avarray = (double *) malloc(sizeof(double) *
                                 sizescan[sliceaxis]);
    for(uint32_t ii = 0; ii < inimg.md->size[sliceaxis]; ii++)
    {
        avarray[ii] = 0.0;
    }

    double *__restrict maskcntarray = (double *) malloc(sizeof(
                                          double) * sizescan[sliceaxis]);
    for(uint32_t ii = 0; ii < inimg.md->size[sliceaxis]; ii++)
    {
        maskcntarray[ii] = 0.0;
    }

    // input image
    uint32_t pixcoord[3];


    for(uint32_t ii = 0; ii < sizescan[0]; ii++)
    {
        pixcoord[0] = ii;
        uint32_t iim = ii * sizemmask[0];

        for(uint32_t jj = 0; jj < sizescan[1]; jj++)
        {
            pixcoord[1] = jj;
            uint32_t jjm = jj * sizemmask[1];

            for(uint32_t kk = 0; kk < sizescan[2]; kk++)
            {
                pixcoord[2] = kk;
                uint32_t kkm = kk * sizemmask[2];

                uint64_t pixi = kk * sizescan[1] * sizescan[0];
                pixi += jj * sizescan[0];
                pixi += ii;


                uint64_t pixim = kkm * sizescanm[1] * sizescanm[0];
                pixim += jjm * sizescanm[0];
                pixim += iim;

                double valm; // masked value

                switch(inimg.datatype)
                {
                    case _DATATYPE_UINT8 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.UI8[pixi];
                        break;
                    case _DATATYPE_INT8 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.SI8[pixi];
                        break;
                    case _DATATYPE_UINT16 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.UI16[pixi];
                        break;
                    case _DATATYPE_INT16 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.SI16[pixi];
                        break;
                    case _DATATYPE_UINT32 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.UI32[pixi];
                        break;
                    case _DATATYPE_INT32 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.SI32[pixi];
                        break;
                    case _DATATYPE_UINT64 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.UI64[pixi];
                        break;
                    case _DATATYPE_INT64 :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.SI64[pixi];
                        break;
                    case _DATATYPE_FLOAT :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.F[pixi];
                        break;
                    case _DATATYPE_DOUBLE :
                        valm = maskimg.im->array.F[pixim] * inimg.im->array.D[pixi];
                        break;
                }
                normarray[pixcoord[sliceaxis]] += valm * valm;
                avarray[pixcoord[sliceaxis]] += valm;
                maskcntarray[pixcoord[sliceaxis]] += maskimg.im->array.F[pixim];
            }
        }
    }


    for(uint32_t ii = 0; ii < sizescan[sliceaxis]; ii++)
    {
        avarray[ii] /= maskcntarray[ii];

        normarray[ii] /= maskcntarray[ii];
        // REMOVED FROM DEF BEHAVIOR: no mean sub.
        // normarray[ii] -= avarray[ii]*avarray[ii];
        if(normarray[ii] > 0.0)
        {
            normarray[ii] = sqrt(normarray[ii]);
        }
        // printf("slice %3u : cnt=%lf  av=%lf  std=%lf\n", ii, maskcntarray[ii], avarray[ii], normarray[ii]);


        if(modeRMS == 0)
        {
            normarray[ii] *= sqrt(maskcntarray[ii]);
        }
    }

    // process input image
    //
    for(uint32_t ii = 0; ii < sizescan[0]; ii++)
    {
        pixcoord[0] = ii;
        for(uint32_t jj = 0; jj < sizescan[1]; jj++)
        {
            pixcoord[1] = jj;
            for(uint32_t kk = 0; kk < sizescan[2]; kk++)
            {
                pixcoord[2] = kk;

                uint64_t pixi = kk * sizescan[1] * sizescan[0];
                pixi += jj * sizescan[0];
                pixi += ii;


                switch(inimg.datatype)
                {
                    // REMOVED FROM DEF BEHAVIOR: no mean sub.
                    // case _DATATYPE_UINT8 :
                    //     outimg->im->array.F[pixi] = (1.0*inimg.im->array.UI8[pixi] - avarray[pixcoord[sliceaxis]]) / normarray[pixcoord[sliceaxis]];
                    //     break;
                    case _DATATYPE_UINT8 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.UI8[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_INT8 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.SI8[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_UINT16 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.UI16[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_INT16 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.SI16[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_UINT32 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.UI32[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_INT32 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.SI32[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_UINT64 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.UI64[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_INT64 :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.SI64[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_FLOAT :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.F[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                    case _DATATYPE_DOUBLE :
                        outimg->im->array.F[pixi] = (1.0 * inimg.im->array.D[pixi]) /
                                                    normarray[pixcoord[sliceaxis]];
                        break;
                }
            }
        }
    }




    if(imgaux.ID != -1)
    {
        // process aux input image
        // FLOAT only
        // scaling only, no offset
        //
        for(uint32_t ii = 0; ii < auxsizescan[0]; ii++)
        {
            pixcoord[0] = ii;
            for(uint32_t jj = 0; jj < auxsizescan[1]; jj++)
            {
                pixcoord[1] = jj;
                for(uint32_t kk = 0; kk < auxsizescan[2]; kk++)
                {
                    pixcoord[2] = kk;

                    uint64_t pixi = kk * auxsizescan[1] * auxsizescan[0];
                    pixi += jj * auxsizescan[0];
                    pixi += ii;

                    imgaux.im->array.F[pixi] = imgaux.im->array.F[pixi] /
                                               normarray[pixcoord[sliceaxis]];
                }
            }
        }
    }


    free(normarray);
    free(avarray);
    free(maskcntarray);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    IMGID maskimg = mkIMGID_from_name(maskimname);
    resolveIMGID(&maskimg, ERRMODE_ABORT);

    IMGID imgaux = mkIMGID_from_name(auxin);
    resolveIMGID(&imgaux, ERRMODE_WARN);

    IMGID outimg = mkIMGID_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        image_slicenormalize(
            inimg,
            maskimg,
            &outimg,
            *sliceaxis,
            imgaux,
            *modeRMS
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
CLIADDCMD_COREMOD_arith__image_slicenormalize()
{
    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

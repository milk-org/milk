/** @file stream_ave.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"



static char *inimname;


static char *outimave;
static char *outimrms;


static uint64_t *NBcoadd;

static uint64_t *cntindex;
static long      fpi_cntindex = -1;


static uint64_t *compave;
static long     fpi_compave = -1;

static uint64_t *comprms;
static long     fpi_comprms = -1;


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outave_name",
        "output average image",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimave,
        NULL
    },
    {
        CLIARG_STR,
        ".outrms_name",
        "output RMS image",
        "out1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &outimrms,
        NULL
    },
    {
        CLIARG_UINT64,
        ".NBcoadd",
        "number of coadded frames",
        "100",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &NBcoadd,
        NULL
    },
    {
        CLIARG_UINT64,
        ".cntindex",
        "counter index",
        "5",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cntindex,
        &fpi_cntindex
    },
    {
        CLIARG_ONOFF,
        ".comp.ave",
        "compute average",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &compave,
        &fpi_compave
    },
    {
        CLIARG_ONOFF,
        ".comp.rms",
        "compute rms",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &comprms,
        &fpi_comprms
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
    "streamave",
    "average stream of images",
    CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Average frames from stream\n");
    printf("output is by default float type\n");

    return RETURN_SUCCESS;
}







static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    uint32_t xsize  = inimg.size[0];
    uint32_t ysize  = inimg.size[1];
    uint64_t xysize = xsize * ysize;

    IMGID outimgave  = makeIMGID_2D(outimave, xsize, ysize);
    outimgave.shared = 1;

    if((*compave) == 1)
    {
        imcreateIMGID(&outimgave);
    }

    IMGID outimgrms  = makeIMGID_2D(outimrms, xsize, ysize);
    outimgrms.shared = 1;

    if((*comprms) == 1)
    {
        imcreateIMGID(&outimgrms);
    }



    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
        printf("PROCINFO IS ON\n");
    }

    DEBUG_TRACEPOINT("Allocating summation array");
    double * restrict imdataarray    = (double *) malloc(sizeof(double) * xysize);
    if(imdataarray == NULL) {
        PRINT_ERROR("malloc returns NULL pointer, size %ld", (long) (sizeof(double) * xysize));
        abort();
    }

    double * restrict imdataarrayPOW = (double *) malloc(sizeof(double) * xysize);
    if(imdataarrayPOW == NULL) {
        PRINT_ERROR("malloc returns NULL pointer, size %ld", (long) (sizeof(double) * xysize));
        abort();
    }

    *cntindex = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    //DEBUG_TRACEPOINT("cntindex = %lu / %lu", *cntindex, *NBcoadd);

    printf("Adding image %lu / %lu\n", *cntindex, *NBcoadd);

    if(*cntindex == 0)
    {
        // initialization
        //

        switch(inimg.datatype)
        {

        case _DATATYPE_FLOAT :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = inimg.im->array.F[pixi];
            }
            break;

        case _DATATYPE_DOUBLE :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = inimg.im->array.D[pixi];
            }
            break;

        case _DATATYPE_INT16 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = (double) inimg.im->array.SI16[pixi];
            }
            break;

        case _DATATYPE_UINT16 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = (double) inimg.im->array.UI16[pixi];
            }
            break;

        case _DATATYPE_INT32 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = (double) inimg.im->array.SI32[pixi];
            }
            break;

        case _DATATYPE_UINT32 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = (double) inimg.im->array.UI32[pixi];
            }
            break;

        case _DATATYPE_INT64 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = (double) inimg.im->array.SI64[pixi];
            }
            break;

        case _DATATYPE_UINT64 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] = (double) inimg.im->array.UI64[pixi];
            }
            break;


        }

        if(*comprms == 1)
        {
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarrayPOW[pixi] = imdataarray[pixi] * imdataarray[pixi];
            }
        }

    }
    else
    {

        switch(inimg.datatype)
        {
        case _DATATYPE_FLOAT :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += inimg.im->array.F[pixi];
            }
            break;

        case _DATATYPE_DOUBLE :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += inimg.im->array.D[pixi];
            }
            break;

        case _DATATYPE_INT16 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += (double) inimg.im->array.SI16[pixi];
            }
            break;

        case _DATATYPE_UINT16 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += (double) inimg.im->array.UI16[pixi];
            }
            break;

        case _DATATYPE_INT32 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += (double) inimg.im->array.SI32[pixi];
            }
            break;

        case _DATATYPE_UINT32 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += (double) inimg.im->array.UI32[pixi];
            }
            break;

        case _DATATYPE_INT64 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += (double) inimg.im->array.SI64[pixi];
            }
            break;

        case _DATATYPE_UINT64 :
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarray[pixi] += (double) inimg.im->array.UI64[pixi];
            }
            break;
        }


        if(*comprms == 1)
        {
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                imdataarrayPOW[pixi] += imdataarray[pixi] * imdataarray[pixi];
            }
        }
    }


    (*cntindex)++;
    if((*cntindex) >= (*NBcoadd))
    {

        if(*compave == 1)
        {
            DEBUG_TRACEPOINT("Writing output AVE image");

            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                outimgave.im->array.F[pixi] = imdataarray[pixi] / (*cntindex);
            }

            processinfo_update_output_stream(processinfo, outimgave.ID);
        }

        if(*comprms == 1)
        {
            DEBUG_TRACEPOINT("Writing output RMS image");
            for(uint_fast64_t pixi = 0; pixi < xysize; pixi++)
            {
                outimgrms.im->array.F[pixi] =
                    sqrt(imdataarrayPOW[pixi]) / (*cntindex);
            }

            processinfo_update_output_stream(processinfo, outimgrms.ID);
        }

        (*cntindex) = 0;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(imdataarray);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_streamaverage()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;


    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

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
static uint64_t *comprms;



static CLICMDARGDEF farg[] = {{
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
        NULL
    },
    {
        CLIARG_ONOFF,
        ".comp.rms",
        "compute rms",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &comprms,
        NULL
    }
};


static CLICMDDATA CLIcmddata =
{
    "streamave", "average stream of images", CLICMD_FIELDS_DEFAULTS
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
    }

    DEBUG_TRACEPOINT("Allocating summation array");
    double *imdataarray    = (double *) malloc(sizeof(double) * xysize);
    double *imdataarrayPOW = (double *) malloc(sizeof(double) * xysize);

    *cntindex = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    DEBUG_TRACEPOINT("cntindex = %lu / %lu", *cntindex, *NBcoadd);

    printf("Adding image %lu / %lu\n", *cntindex, *NBcoadd);

    if(*cntindex == 0)
    {
        // initialization
        //

        switch(inimg.datatype)
        {

            case _DATATYPE_FLOAT :
                for(uint64_t pixi = 0; pixi < xysize; pixi++)
                {
                    imdataarray[pixi] = inimg.im->array.F[pixi];
                }
                break;

            case _DATATYPE_UINT16 :
                for(uint64_t pixi = 0; pixi < xysize; pixi++)
                {
                    imdataarray[pixi] = (float) inimg.im->array.UI16[pixi];
                }
                break;
        }

        if(*comprms == 1)
        {
            for(uint64_t pixi = 0; pixi < xysize; pixi++)
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
                for(uint64_t pixi = 0; pixi < xysize; pixi++)
                {
                    imdataarray[pixi] += inimg.im->array.F[pixi];
                }
                break;
            case _DATATYPE_UINT16 :
                for(uint64_t pixi = 0; pixi < xysize; pixi++)
                {
                    imdataarray[pixi] += (float) inimg.im->array.UI16[pixi];
                }
                break;
        }


        if(*comprms == 1)
        {
            for(uint64_t pixi = 0; pixi < xysize; pixi++)
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
            for(uint64_t pixi = 0; pixi < xysize; pixi++)
            {
                outimgave.im->array.F[pixi] = imdataarray[pixi] / (*cntindex);
            }

            processinfo_update_output_stream(processinfo, outimgave.ID);
        }

        if(*comprms == 1)
        {
            DEBUG_TRACEPOINT("Writing output RMS image");
            for(uint64_t pixi = 0; pixi < xysize; pixi++)
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
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}




/*




imageID COREMOD_MEMORY_streamAve(const char *IDstream_name,
                                 int         NBave,
                                 int         mode,
                                 const char *IDout_name);


static errno_t COREMOD_MEMORY_streamAve__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_LONG) +
            CLI_checkarg(3, CLIARG_LONG) + CLI_checkarg(4, 5) ==
        0)
    {
        COREMOD_MEMORY_streamAve(data.cmdargtoken[1].val.string,
                                 data.cmdargtoken[2].val.numl,
                                 data.cmdargtoken[3].val.numl,
                                 data.cmdargtoken[4].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



errno_t stream_ave_addCLIcmd()
{
    RegisterCLIcommand(
        "streamave",
        __FILE__,
        COREMOD_MEMORY_streamAve__cli,
        "averages stream",
        "<instream> <NBave> <mode, 1 for single local instance, 0 for loop> "
        "<outstream>",
        "streamave instream 100 0 outstream",
        "long COREMODE_MEMORY_streamAve(const char *IDstream_name, int NBave, "
        "int mode, const char *IDout_name)");

    return RETURN_SUCCESS;
}




imageID COREMOD_MEMORY_streamAve(const char *IDstream_name,
                                 int         NBave,
                                 int         mode,
                                 const char *IDout_name)
{
    DEBUG_TRACE_FSTART();

    imageID     IDout;
    imageID     IDout0;
    imageID     IDin;
    uint8_t     datatype;
    uint32_t    xsize;
    uint32_t    ysize;
    uint32_t    xysize;
    uint32_t   *imsize;
    int_fast8_t OKloop;
    int         cntin = 0;
    long        dtus  = 20;
    long        ii;
    long        cnt0;
    long        cnt0old;

    IDin     = image_ID(IDstream_name);
    datatype = data.image[IDin].md[0].datatype;
    xsize    = data.image[IDin].md[0].size[0];
    ysize    = data.image[IDin].md[0].size[1];
    xysize   = xsize * ysize;

    FUNC_CHECK_RETURN(
        create_2Dimage_ID("_streamAve_tmp", xsize, ysize, &IDout0));

    if (mode == 1) // local image
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID(IDout_name, xsize, ysize, &IDout));
    }
    else // shared memory
    {
        IDout = image_ID(IDout_name);
        if (IDout == -1) // CREATE IT
        {
            imsize    = (uint32_t *) malloc(sizeof(uint32_t) * 2);
            imsize[0] = xsize;
            imsize[1] = ysize;
            create_image_ID(IDout_name,
                            2,
                            imsize,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDout);
            COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);
            free(imsize);
        }
    }

    cntin   = 0;
    cnt0old = data.image[IDin].md[0].cnt0;

    for (ii = 0; ii < xysize; ii++)
    {
        data.image[IDout].array.F[ii] = 0.0;
    }

    OKloop = 1;
    while (OKloop == 1)
    {
        // has new frame arrived ?
        cnt0 = data.image[IDin].md[0].cnt0;
        if (cnt0 != cnt0old)
        {
            switch (datatype)
            {
            case _DATATYPE_UINT8:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.UI8[ii];
                }
                break;

            case _DATATYPE_INT8:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.SI8[ii];
                }
                break;

            case _DATATYPE_UINT16:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.UI16[ii];
                }
                break;

            case _DATATYPE_INT16:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.SI16[ii];
                }
                break;

            case _DATATYPE_UINT32:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.UI32[ii];
                }
                break;

            case _DATATYPE_INT32:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.SI32[ii];
                }
                break;

            case _DATATYPE_UINT64:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.UI64[ii];
                }
                break;

            case _DATATYPE_INT64:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.SI64[ii];
                }
                break;

            case _DATATYPE_FLOAT:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.F[ii];
                }
                break;

            case _DATATYPE_DOUBLE:
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout0].array.F[ii] +=
                        data.image[IDin].array.D[ii];
                }
                break;
            }

            cntin++;
            if (cntin == NBave)
            {
                cntin                         = 0;
                data.image[IDout].md[0].write = 1;
                for (ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        data.image[IDout0].array.F[ii] / NBave;
                }
                data.image[IDout].md[0].cnt0++;
                data.image[IDout].md[0].write = 0;
                COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

                if (mode != 1)
                {
                    for (ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[ii] = 0.0;
                    }
                }
                else
                {
                    OKloop = 0;
                }
            }
            cnt0old = cnt0;
        }
        usleep(dtus);
    }

    delete_image_ID("_streamAve_tmp", DELETE_IMAGE_ERRMODE_WARNING);

    DEBUG_TRACE_FEXIT();
    return IDout;
}
*/

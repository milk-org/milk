/**
 * @file stream_paste.c
 * @brief Paste two equal size 2D streams into an output 2D stream
*/

#include "CommandLineInterface/CLIcore.h"
#include "create_image.h"
#include "image_ID.h"
#include "stream_sem.h"

// ==========================================
// Forward declarations
// ==========================================

imageID COREMOD_MEMORY_streamPaste(const char *IDstream0_name,
                                   const char *IDstream1_name,
                                   const char *IDstreamout_name,
                                   long        semtrig0,
                                   long        semtrig1,
                                   int         master);

// ==========================================
// Command line interface wrapper functions
// ==========================================

static errno_t COREMOD_MEMORY_streamPaste__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_IMG) +
            CLI_checkarg(3, 5) + CLI_checkarg(4, CLIARG_LONG) +
            CLI_checkarg(5, CLIARG_LONG) + CLI_checkarg(6, CLIARG_LONG) ==
            0)
    {
        COREMOD_MEMORY_streamPaste(data.cmdargtoken[1].val.string,
                                   data.cmdargtoken[2].val.string,
                                   data.cmdargtoken[3].val.string,
                                   data.cmdargtoken[4].val.numl,
                                   data.cmdargtoken[5].val.numl,
                                   data.cmdargtoken[6].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t stream_paste_addCLIcmd()
{

    RegisterCLIcommand(
        "streampaste",
        __FILE__,
        COREMOD_MEMORY_streamPaste__cli,
        "paste two 2D image streams of same size",
        "<in stream 0> <in stream 1> <out stream> <sem trigger0> <sem "
        "trigger1> <master>",
        "streampaste stream0 stream1 outstream 3 3 0",
        "long COREMOD_MEMORY_streamPaste(const char *IDstream0_name, const "
        "char *IDstream1_name, const "
        "char *IDstreamout_name, long semtrig0, long semtrig1, int master)");

    return RETURN_SUCCESS;
}

//
// compute difference between two 2D streams
// triggers alternatively on stream0 and stream1
//
imageID COREMOD_MEMORY_streamPaste(const char *IDstream0_name,
                                   const char *IDstream1_name,
                                   const char *IDstreamout_name,
                                   long        semtrig0,
                                   long        semtrig1,
                                   int         master)
{
    imageID            ID0;
    imageID            ID1;
    imageID            IDout;
    imageID            IDin;
    long               Xoffset;
    uint32_t           xsize;
    uint32_t           ysize;
    uint32_t          *arraysize;
    unsigned long long cnt;
    uint8_t            datatype;
    int                FrameIndex;

    ID0 = image_ID(IDstream0_name);
    ID1 = image_ID(IDstream1_name);

    xsize    = data.image[ID0].md[0].size[0];
    ysize    = data.image[ID0].md[0].size[1];
    datatype = data.image[ID0].md[0].datatype;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(arraysize == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    arraysize[0] = 2 * xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name);
    if(IDout == -1)
    {
        create_image_ID(IDstreamout_name,
                        2,
                        arraysize,
                        datatype,
                        1,
                        0,
                        0,
                        &IDout);
        COREMOD_MEMORY_image_set_createsem(IDstreamout_name,
                                           IMAGE_NB_SEMAPHORE);
    }
    free(arraysize);

    FrameIndex = 0;

    while(1)
    {
        if(FrameIndex == 0)
        {
            // has new frame 0 arrived ?
            if(data.image[ID0].md[0].sem == 0)
            {
                while(cnt ==
                        data.image[ID0].md[0].cnt0) // test if new frame exists
                {
                    usleep(5);
                }
                cnt = data.image[ID0].md[0].cnt0;
            }
            else
            {
                sem_wait(data.image[ID0].semptr[semtrig0]);
            }
            Xoffset = 0;
            IDin    = 0;
        }
        else
        {
            // has new frame 1 arrived ?
            if(data.image[ID1].md[0].sem == 0)
            {
                while(cnt ==
                        data.image[ID1].md[0].cnt0) // test if new frame exists
                {
                    usleep(5);
                }
                cnt = data.image[ID1].md[0].cnt0;
            }
            else
            {
                sem_wait(data.image[ID1].semptr[semtrig1]);
            }
            Xoffset = xsize;
            IDin    = 1;
        }

        data.image[IDout].md[0].write = 1;

        switch(datatype)
        {
            case _DATATYPE_UINT8:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.UI8[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI8[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT16:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout]
                        .array.UI16[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI16[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT32:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout]
                        .array.UI32[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI32[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT64:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout]
                        .array.UI64[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.UI64[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT8:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.SI8[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI8[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT16:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout]
                        .array.SI16[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI16[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT32:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout]
                        .array.SI32[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI32[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT64:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout]
                        .array.SI64[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.SI64[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_FLOAT:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.F[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.F[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_DOUBLE:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.D[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.D[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_COMPLEX_FLOAT:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.CF[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.CF[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_COMPLEX_DOUBLE:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        data.image[IDout].array.CD[jj * 2 * xsize + ii + Xoffset] =
                            data.image[IDin].array.CD[jj * xsize + ii];
                    }
                break;

            default:
                printf("Unknown data type\n");
                exit(0);
                break;
        }
        if(FrameIndex == master)
        {
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
            ;
            data.image[IDout].md[0].cnt0++;
        }
        data.image[IDout].md[0].cnt1  = FrameIndex;
        data.image[IDout].md[0].write = 0;

        if(FrameIndex == 0)
        {
            FrameIndex = 1;
        }
        else
        {
            FrameIndex = 0;
        }
    }

    return IDout;
}

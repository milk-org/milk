/**
 * @file    stream_diff.c
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "create_image.h"
#include "image_ID.h"
#include "stream_sem.h"

// ==========================================
// forward declarations
// ==========================================

imageID COREMOD_MEMORY_streamDiff(const char *IDstream0_name,
                                  const char *IDstream1_name,
                                  const char *IDstreammask_name,
                                  const char *IDstreamout_name,
                                  long        semtrig);

// ==========================================
// command line interface wrapper functions
// ==========================================

#ifndef MILK_NO_CLI
static errno_t COREMOD_MEMORY_streamDiff__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_IMG) +
            CLI_checkarg(3, 5) + CLI_checkarg(4, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(5, CLIARG_INT64) ==
            0)
    {
        COREMOD_MEMORY_streamDiff(data.cmdargtoken[1].val.string,
                                  data.cmdargtoken[2].val.string,
                                  data.cmdargtoken[3].val.string,
                                  data.cmdargtoken[4].val.string,
                                  data.cmdargtoken[5].val.numl);
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

errno_t stream_diff_addCLIcmd()
{
    RegisterCLIcommand("streamdiff",
                       __FILE__,
                       COREMOD_MEMORY_streamDiff__cli,
                       "compute difference between two image streams",
                       "<in stream 0> <in stream 1> <out stream> <optional "
                       "mask> <sem trigger index>",
                       "streamdiff stream0 stream1 null outstream 3",
                       "long COREMOD_MEMORY_streamDiff(const char "
                       "*IDstream0_name, const char *IDstream1_name, const "
                       "char *IDstreamout_name, long semtrig)");

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
/**
 * ## Purpose
 *
 * Compute difference between two 2D streams\n
 * Triggers on stream0\n
 *
 */
imageID COREMOD_MEMORY_streamDiff(const char *IDstream0_name,
                                  const char *IDstream1_name,
                                  const char *IDstreammask_name,
                                  const char *IDstreamout_name,
                                  long        semtrig)
{
    imageID            ID0;
    imageID            ID1;
    imageID            IDout;
    uint32_t           xsize;
    uint32_t           ysize;
    uint64_t           xysize;
    uint32_t          *arraysize;
    unsigned long long cnt;
    imageID            IDmask; // optional

    ID0    = image_ID(IDstream0_name, dcimg, dcnimg);
    ID1    = image_ID(IDstream1_name, dcimg, dcnimg);
    IDmask = image_ID(IDstreammask_name, dcimg, dcnimg);

    xsize  = dcimg[ID0].md[0].size[0];
    ysize  = dcimg[ID0].md[0].size[1];
    xysize = xsize * ysize;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(arraysize == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name, dcimg, dcnimg);
    if(IDout == -1)
    {
        create_image_ID(IDstreamout_name,
                        2,
                        arraysize,
                        _DATATYPE_FLOAT,
                        1,
                        0,
                        0,
                        &IDout);
    }

    free(arraysize);

    while(1)
    {
        // has new frame arrived ?
        if(dcimg[ID0].md[0].sem == 0)
        {
            while(cnt ==
                    dcimg[ID0].md[0].cnt0) // test if new frame exists
            {
                usleep(5);
            }
            cnt = dcimg[ID0].md[0].cnt0;
        }
        else
        {
            ImageStreamIO_semwait(dcimg+ID0, semtrig);
        }

        dcimg[IDout].md[0].write = 1;
        if(IDmask == -1)
        {
            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                dcimg[IDout].array.F[ii] =
                    dcimg[ID0].array.F[ii] - dcimg[ID1].array.F[ii];
            }
        }
        else
        {
            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                dcimg[IDout].array.F[ii] = (dcimg[ID0].array.F[ii] -
                                                 dcimg[ID1].array.F[ii]) *
                                                dcimg[IDmask].array.F[ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        ;
        dcimg[IDout].md[0].cnt0++;
        dcimg[IDout].md[0].write = 0;
    }

    return IDout;
}


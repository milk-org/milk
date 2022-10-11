/**
 * @file    stream_diff.c
 */

#include "CommandLineInterface/CLIcore.h"
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

static errno_t COREMOD_MEMORY_streamDiff__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_IMG) +
            CLI_checkarg(3, 5) + CLI_checkarg(4, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(5, CLIARG_LONG) ==
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

    ID0    = image_ID(IDstream0_name);
    ID1    = image_ID(IDstream1_name);
    IDmask = image_ID(IDstreammask_name);

    xsize  = data.image[ID0].md[0].size[0];
    ysize  = data.image[ID0].md[0].size[1];
    xysize = xsize * ysize;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(arraysize == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name);
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
        COREMOD_MEMORY_image_set_createsem(IDstreamout_name,
                                           IMAGE_NB_SEMAPHORE);
    }

    free(arraysize);

    while(1)
    {
        // has new frame arrived ?
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
            sem_wait(data.image[ID0].semptr[semtrig]);
        }

        data.image[IDout].md[0].write = 1;
        if(IDmask == -1)
        {
            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                data.image[IDout].array.F[ii] =
                    data.image[ID0].array.F[ii] - data.image[ID1].array.F[ii];
            }
        }
        else
        {
            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                data.image[IDout].array.F[ii] = (data.image[ID0].array.F[ii] -
                                                 data.image[ID1].array.F[ii]) *
                                                data.image[IDmask].array.F[ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        ;
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }

    return IDout;
}

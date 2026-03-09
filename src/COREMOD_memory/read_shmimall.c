/**
 * @file    read_shmimall.c
 * @brief   read all shared memory stream
 */

#include <fcntl.h>    // open
#include <sys/mman.h> // mmap
#include <sys/stat.h>
#include <unistd.h> // close

#include "CLIcore.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"

#ifndef MILK_NO_CLI
#include "streamCTRL_find_streams.h"
#endif

errno_t read_sharedmem_image_all(const char *name);

static errno_t read_sharedmem_image_all__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_STR) == 0)
    {

        read_sharedmem_image_all(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t read_shmimall_addCLIcmd()
{

    RegisterCLIcommand("readshmimall",
                       __FILE__,
                       read_sharedmem_image_all__cli,
                       "read all shared memory images",
                       "<string filter>",
                       "readshmimall aol_",
                       "read_sharedmem_image_all(const char *name)");

    return RETURN_SUCCESS;
}

errno_t read_sharedmem_image_all(
    const char *strfilter)
{
#ifdef MILK_NO_CLI
    (void) strfilter;
    return RETURN_SUCCESS;
#else
    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    streaminfo = (STREAMINFO *)
        malloc(sizeof(STREAMINFO) * NBstreamMAX);

    int NBstream =
        find_streams(streaminfo, 1, strfilter);

    for(int sindex = 0;
         sindex < NBstream; sindex++)
    {
        imageID ID = image_ID(
            streaminfo[sindex].sname,
            dcimg, dcnimg);
        if(ID == -1)
        {
            read_sharedmem_image(
                streaminfo[sindex].sname,
                dcimg, dcnimg);
        }
    }

    free(streaminfo);

    return RETURN_SUCCESS;
#endif
}

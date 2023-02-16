/** @file streamrecord.c
 */

#include <sched.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_BASIC_streamrecord_cli()
{
    if(0 + CLI_checkarg(1, 4) + CLI_checkarg(2, 2) + CLI_checkarg(3, 3) == 0)
    {
        IMAGE_BASIC_streamrecord(data.cmdargtoken[1].val.string,
                                 data.cmdargtoken[2].val.numl,
                                 data.cmdargtoken[3].val.string);
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

errno_t __attribute__((cold)) streamrecord_addCLIcmd()
{

    RegisterCLIcommand("imgstreamrec",
                       __FILE__,
                       IMAGE_BASIC_streamrecord_cli,
                       "record stream of images",
                       "<stream> <# frames> <output>",
                       "imgstreamrec imstream 100 imrec",
                       "long IMAGE_BASIC_streamrecord(const char *streamname, "
                       "long NBframes, const char *IDname)");

    return RETURN_SUCCESS;
}

// works only for floats
//
imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname)
{
    imageID       ID;
    imageID       IDstream;
    long          xsize, ysize, zsize, xysize;
    unsigned long cnt;
    long          waitdelayus = 50;
    long          kk;
    char         *ptr;

    IDstream = image_ID(streamname);
    xsize    = data.image[IDstream].md[0].size[0];
    ysize    = data.image[IDstream].md[0].size[1];
    zsize    = NBframes;
    xysize   = xsize * ysize;

    create_3Dimage_ID(IDname, xsize, ysize, zsize, &ID);
    cnt = data.image[IDstream].md[0].cnt0;

    kk = 0;

    ptr = (char *) data.image[ID].array.F;
    while(kk != NBframes)
    {
        while(cnt > data.image[IDstream].md[0].cnt0)
        {
            usleep(waitdelayus);
        }

        cnt++;

        printf("\r%ld / %ld  [%ld %ld]      ",
               kk,
               NBframes,
               cnt,
               data.image[ID].md[0].cnt0);
        fflush(stdout);

        memcpy(ptr, data.image[IDstream].array.F, sizeof(float) * xysize);
        ptr += sizeof(float) * xysize;
        kk++;
    }
    printf("\n\n");

    return ID;
}

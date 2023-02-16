/** @file streamfeed.c
 */

#include <sched.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================
long IMAGE_BASIC_streamfeed(const char *__restrict IDname,
                            const char *__restrict streamname,
                            float frequ);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_streamfeed_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 1) == 0)
    {
        IMAGE_BASIC_streamfeed(data.cmdargtoken[1].val.string,
                               data.cmdargtoken[2].val.string,
                               data.cmdargtoken[3].val.numf);
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

errno_t __attribute__((cold)) streamfeed_addCLIcmd()
{
    RegisterCLIcommand("imgstreamfeed",
                       __FILE__,
                       image_basic_streamfeed_cli,
                       "feed stream of images",
                       "<input image/cube> <stream> <fequ [Hz]>",
                       "imgstreamfeed im imstream 100",
                       "long IMAGE_BASIC_streamfeed(const char *IDname, const "
                       "char *streamname, float frequ)");

    return RETURN_SUCCESS;
}

// feed image to data stream
// only works on slice #1 out output
long IMAGE_BASIC_streamfeed(const char *__restrict IDname,
                            const char *__restrict streamname,
                            float frequ)
{
    imageID            ID;
    imageID            IDs;
    long               xsize, ysize, xysize, zsize;
    long               k;
    long               tdelay;
    int                RT_priority = 95; //any number from 0-99
    struct sched_param schedpar;
    int                semval;
    const char        *ptr0;
    const char        *ptr1;
    int                loopOK;
    long               ii;

    schedpar.sched_priority = RT_priority;
    if(seteuid(data.euid) != 0)  //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if(seteuid(data.ruid) != 0)    //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    ID     = image_ID(IDname);
    xsize  = data.image[ID].md[0].size[0];
    ysize  = data.image[ID].md[0].size[1];
    xysize = xsize * ysize;

    tdelay = (long)(1000000.0 / frequ);

    printf("frequ = %f Hz\n", frequ);
    printf("tdelay = %ld us\n", tdelay);

    IDs = image_ID(streamname);
    if((xsize != data.image[IDs].md[0].size[0]) ||
            (ysize != data.image[IDs].md[0].size[1]))
    {
        printf("ERROR: images have different x and y sizes");
        exit(0);
    }
    zsize = data.image[ID].md[0].size[2];

    ptr1 = (char *) data.image[IDs].array.F; // destination

    if(sigaction(SIGINT, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGTERM, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGBUS, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGSEGV, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGABRT, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGHUP, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    k      = 0;
    loopOK = 1;
    while(loopOK == 1)
    {
        ptr0 = (char *) data.image[ID].array.F;
        ptr0 += sizeof(float) * xysize * k;
        data.image[IDs].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, sizeof(float) * xysize);

        data.image[IDs].md[0].write = 0;
        data.image[IDs].md[0].cnt0++;
        COREMOD_MEMORY_image_set_sempost_byID(IDs, -1);

        usleep(tdelay);
        k++;
        if(k == zsize)
        {
            k = 0;
        }

        if((data.signal_INT == 1) || (data.signal_TERM == 1) ||
                (data.signal_ABRT == 1) || (data.signal_BUS == 1) ||
                (data.signal_SEGV == 1) || (data.signal_HUP == 1) ||
                (data.signal_PIPE == 1))
        {
            loopOK = 0;
        }
    }

    data.image[IDs].md[0].write = 1;
    for(ii = 0; ii < xysize; ii++)
    {
        data.image[IDs].array.F[ii] = 0.0;
    }
    if(data.image[IDs].md[0].sem > 0)
    {
        sem_getvalue(data.image[IDs].semptr[0], &semval);
        if(semval < SEMAPHORE_MAXVAL)
        {
            sem_post(data.image[IDs].semptr[0]);
        }
    }
    data.image[IDs].md[0].write = 0;
    data.image[IDs].md[0].cnt0++;

    return (0);
}

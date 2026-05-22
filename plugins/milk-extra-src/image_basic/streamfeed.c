/**
 * @file streamfeed.c
 * @brief Feed stream of images
 */

#include <sched.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
long IMAGE_BASIC_streamfeed(const char *__restrict IDname,
                            const char *__restrict streamname,
                            float frequ);

static char   p_in[FUNCTION_PARAMETER_STRMAXLEN]     = "im";
static char   p_stream[FUNCTION_PARAMETER_STRMAXLEN] = "imstream";
static double p_freq                                 = 100.0;

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imgstreamfeed",
    .cmdkey           = "imgstreamfeed",
    .description      = "feed stream of images",
    .description_long = "Feed a sequence of images from a 3D cube into a 2D shared memory stream, "
                        "playing back slices at a configurable frame rate."
};

#define FPS_PARAMS(X)                                                                   \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image/cube") \
    X(".stream", p_stream, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output stream") \
    X(".freq", &p_freq, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "frequency [Hz]")

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };
static const int       nb_bindings   = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg[]        = { FPS_PARAMS(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata    = { "", "", CLICMD_FIELDS_DEFAULTS };
FPS_CMDSETTINGS_INIT(streamfeed, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t compute_function()
{
    IMAGE_BASIC_streamfeed(p_in, p_stream, (float) p_freq);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_basic__streamfeed()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
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
    if (seteuid(dceuid) != 0) //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if (seteuid(dcruid) != 0)      //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    ID     = image_ID(IDname, dcimg, dcnimg);
    xsize  = dcimg[ID].md[0].size[0];
    ysize  = dcimg[ID].md[0].size[1];
    xysize = xsize * ysize;

    tdelay = (long) (1000000.0 / frequ);

    printf("frequ = %f Hz\n", frequ);
    printf("tdelay = %ld us\n", tdelay);

    IDs = image_ID(streamname, dcimg, dcnimg);
    if ((xsize != dcimg[IDs].md[0].size[0]) || (ysize != dcimg[IDs].md[0].size[1]))
    {
        printf("ERROR: images have different x and y sizes");
        exit(0);
    }
    zsize = dcimg[ID].md[0].size[2];

    ptr1 = (char *) dcimg[IDs].array.F; // destination

    if (sigaction(SIGINT, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGTERM, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGBUS, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGSEGV, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGABRT, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGHUP, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGPIPE, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    k      = 0;
    loopOK = 1;
    while (loopOK == 1)
    {
        ptr0 = (char *) dcimg[ID].array.F;
        ptr0 += sizeof(float) * xysize * k;
        dcimg[IDs].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, sizeof(float) * xysize);

        dcimg[IDs].md[0].write = 0;
        dcimg[IDs].md[0].cnt0++;
        COREMOD_MEMORY_image_set_sempost_byID(IDs, -1);

        usleep(tdelay);
        k++;
        if (k == zsize)
        {
            k = 0;
        }

        if (DCSIG_ANY_SET())
        {
            loopOK = 0;
        }
    }

    dcimg[IDs].md[0].write = 1;
    for (ii = 0; ii < xysize; ii++)
    {
        dcimg[IDs].array.F[ii] = 0.0f;
    }
    if (dcimg[IDs].md[0].sem > 0)
    {
        semval = ImageStreamIO_semvalue(dcimg + IDs, 0);
        if (semval < SEMAPHORE_MAXVAL)
        {
            ImageStreamIO_sempost(dcimg + IDs, 0);
        }
    }
    dcimg[IDs].md[0].write = 0;
    dcimg[IDs].md[0].cnt0++;

    return (0);
}

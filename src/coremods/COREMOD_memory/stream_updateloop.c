/** @file stream_updateloop.c
 *
 * Uses FPS V2 framework.
 */

#include <sched.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#include "create_image.h"
#include "image_ID.h"
#include "stream_sem.h"
#include "processinfo_setup.h"

#include "COREMOD_tools/COREMOD_tools.h"

/* forward decls */
errno_t COREMOD_MEMORY_image_streamburst(
    const char *IDin_name,
    const char *IDout_name,
    long       periodus);

/**
 * @brief Continuous stream update loop.
 *
 * Monitors an input stream and triggers output
 * updates on each new frame.
 */
imageID COREMOD_MEMORY_image_streamupdateloop(
    const char *IDinname,
    const char *IDoutname,
    long       usperiod,
    long       NBcubes,
    long       period,
    long       offsetus,
    const char *IDsync_name,
    int        semtrig,
    int        timingmode);

imageID
COREMOD_MEMORY_image_streamupdateloop_semtrig(
    const char *IDinname,
    const char *IDoutname,
    long       period,
    long       offsetus,
    const char *IDsync_name,
    int        semtrig,
    int        timingmode);


/* ================================================================
 *  COMMON PARAMS
 * ============================================================= */

static char p_inname[FUNCTION_PARAMETER_STRMAXLEN] = "imcube";
static char p_outname[FUNCTION_PARAMETER_STRMAXLEN] = "outstream";
static long long p_usperiod = 1000;
static long long p_NBcubes = 3;
static long long p_period = 3;
static long long p_offsetus = 154;
static char p_syncname[FUNCTION_PARAMETER_STRMAXLEN] = "ircam1";
static long long p_semtrig = 3;
static long long p_timingmode = 0;


/* ================================================================
 *  CMD 1: streamburst (3 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_burst =
{
    .fps_name    = "streamburst",
    .cmdkey      = "streamburst",
    .description =
    "send burst of frames to stream",
    .description_long =
    "Continuously update a shared memory stream by re-posting its semaphores at a configurable rate. Keeps downstream consumers active even when no new data arrives."
};

#define FPS_PARAMS_BURST(X) \
    X(".inname", p_inname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input cube") \
    X(".outname", p_outname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output stream") \
    X(".usperiod", &p_usperiod, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "period [us]")

static CLICMDDATA CLIcmddata_burst =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(burst, CLIcmddata_burst, FPS_app_info_burst)

static errno_t __attribute__((unused)) compute_burst()
{
    COREMOD_MEMORY_image_streamburst(p_inname, p_outname, p_usperiod);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: creaimstream (9 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "creaimstream",
    .cmdkey      = "creaimstream",
    .description =
    "create 2D stream from 3D cube",
    .description_long =
    "Continuously update a shared memory stream by re-posting its semaphores at a configurable rate. Keeps downstream consumers active even when no new data arrives."
};

#define FPS_PARAMS(X) \
    X(".inname", p_inname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input 3D cube") \
    X(".outname", p_outname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output 2D stream") \
    X(".usperiod", &p_usperiod, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "interval [us]") \
    X(".NBcubes", &p_NBcubes, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "number of cubes") \
    X(".period", &p_period, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sync period") \
    X(".offsetus", &p_offsetus, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "time offset [us]") \
    X(".syncname", p_syncname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sync stream name") \
    X(".semtrig", &p_semtrig, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sem trigger index") \
    X(".timingmode", &p_timingmode, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "timing mode")

static FPS_CLI_BINDING my_bindings[] =
{
    FPS_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] =
{
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata =
{
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(main, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_MEMORY_image_streamupdateloop(
        p_inname, p_outname,
        p_usperiod, p_NBcubes, p_period, p_offsetus, p_syncname, p_semtrig, p_timingmode);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: creaimstreamstrig (7 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_strig =
{
    .fps_name    = "creaimstreamstrig",
    .cmdkey      = "creaimstreamstrig",
    .description =
    "create 2D stream from 3D cube "
    "(sem-triggered)",
    .description_long =
    "Continuously update a shared memory stream by re-posting its semaphores at a configurable rate. Keeps downstream consumers active even when no new data arrives."
};

static CLICMDDATA CLIcmddata_strig =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(strig, CLIcmddata_strig, FPS_app_info_strig)

static errno_t __attribute__((unused)) compute_strig()
{
    COREMOD_MEMORY_image_streamupdateloop_semtrig(
        p_inname, p_outname, p_period, p_offsetus, p_syncname, p_semtrig, p_timingmode);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

/* separate bindings for burst (3 args) */
static FPS_CLI_BINDING bindings_burst[] =
{
    FPS_PARAMS_BURST(FPS_X_BINDING)
};
static const int nb_bindings_burst = sizeof(bindings_burst) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_burst[] =
{
    FPS_PARAMS_BURST(FPS_X_FARG)
};

static errno_t CLIfunction_burst(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_burst,
               farg_burst, &CLIcmddata_burst, bindings_burst, nb_bindings_burst, compute_burst);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

static errno_t CLIfunction_strig(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_strig,
               farg, &CLIcmddata_strig, my_bindings, nb_bindings, compute_strig);
}

errno_t
CLIADDCMD_COREMOD_memory__stream_updateloop()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(farg_burst, bindings_burst, nb_bindings_burst);

    {
        int cmdi = RegisterCLIcmd(CLIcmddata_burst, CLIfunction_burst);
        CLIcmddata_burst.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(CLIcmddata_strig, CLIfunction_strig);
        CLIcmddata_strig.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
/** @brief Send single burst of frames to stream
 *
 */

errno_t COREMOD_MEMORY_image_streamburst(
    const char *IDin_name,
    const char *IDout_name,
    long       periodus)
{
    imageID IDin;
    imageID IDout;

    int                RT_priority = 80; //any number from 0-99
    struct sched_param schedpar;

    char *ptr0s; // source start 3D array ptr
    char *ptr0;  // source
    char *ptr1;  // dest
    long  framesize;

    struct timespec tim;

    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0, SCHED_FIFO, &schedpar);

    {
        IMGID img = imgid_make_from_name(IDin_name);
        resolveIMGID(&img,  ERRMODE_ABORT, dcimg, dcnimg);
        IDin = img.ID;
    }
    long      naxis = dcimg[IDin].md[0].naxis;
    uint32_t *arraysize;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        PRINT_ERROR("input image %s should be 3D", IDin_name);
        free(arraysize);
        return RETURN_FAILURE;
    }
    arraysize[0]     = dcimg[IDin].md[0].size[0];
    arraysize[1]     = dcimg[IDin].md[0].size[1];
    arraysize[2]     = dcimg[IDin].md[0].size[2];
    uint8_t datatype = dcimg[IDin].md[0].datatype;
    int     NBslice  = arraysize[2];

    // check that IDout has same format
    {
        IMGID img = imgid_make_from_name(IDout_name);
        resolveIMGID(&img,  ERRMODE_ABORT, dcimg, dcnimg);
        IDout = img.ID;
    }
    if(dcimg[IDout].md[0].size[0] != dcimg[IDin].md[0].size[0])
    {
        PRINT_ERROR("in and out have different size");
        free(arraysize);
        return RETURN_FAILURE;
    }
    if(dcimg[IDout].md[0].size[1] != dcimg[IDin].md[0].size[1])
    {
        PRINT_ERROR("in and out have different size");
        free(arraysize);
        return RETURN_FAILURE;
    }
    if(dcimg[IDout].md[0].datatype != dcimg[IDin].md[0].datatype)
    {
        PRINT_ERROR("in and out have different datatype");
        free(arraysize);
        return RETURN_FAILURE;
    }

    ptr0s     = (char *) dcimg[IDin].array.raw;
    ptr1      = (char *) dcimg[IDout].array.raw;
    framesize = dcimg[IDin].md[0].size[0] *
                dcimg[IDin].md[0].size[1] * ImageStreamIO_typesize(datatype);

    tim.tv_sec  = 0;
    tim.tv_nsec = (long)(1000 * periodus);

    for(int slice = 0; slice < NBslice; slice++)
    {
        if(nanosleep(&tim, NULL) < 0)
        {
            printf("Nano sleep system call failed \n");
        }

        ptr0 = ptr0s + slice * framesize;

        ptr0                          = ptr0s + slice * framesize;
        dcimg[IDout].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, framesize);
        dcimg[IDout].md[0].cnt1 = slice;
        dcimg[IDout].md[0].cnt0++;
        dcimg[IDout].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
    }

    free(arraysize);

    return RETURN_SUCCESS;
}

/** @brief takes a 3Dimage(s) (circular buffer(s)) and writes slices to a 2D image with time interval specified in us
 *
 *
 * If NBcubes=1, then the circular buffer named IDinname is sent to IDoutname at a frequency of 1/usperiod MHz
 * If NBcubes>1, several circular buffers are used, named ("%S_%03ld", IDinname, cubeindex). Semaphore semtrig of image IDsync_name triggers switch between circular buffers, with a delay of offsetus. The number of consecutive sem posts required to advance to the next circular buffer is period
 *
 * @param IDinname      Name of DM circular buffer (appended by _000, _001 etc... if NBcubes>1)
 * @param IDoutname     Output channel stream
 * @param usperiod      Interval between consecutive frames [us]
 * @param NBcubes       Number of input circular buffers
 * @param period        If NBcubes>1: number of input triggers required to advance to next input buffer
 * @param offsetus      If NBcubes>1: time offset [us] between input trigger and input buffer switch
 * @param IDsync_name   If NBcubes>1: Stream used for synchronization
 * @param semtrig       If NBcubes>1: semaphore used for synchronization
 * @param timingmode    Not used
 *
 *
 */
imageID
COREMOD_MEMORY_image_streamupdateloop(
    const char                  *IDinname,
    const char                  *IDoutname,
    long                        usperiod,
    long                        NBcubes,
    long                        period,
    long                        offsetus,
    const char                  *IDsync_name,
    int                         semtrig,
    __attribute__((unused)) int timingmode)
{
    imageID           *IDin;

    char               imname[200];
    long               IDsync;
    long               cubeindex; // used outside loop
    unsigned long long cntsync;
    long               pcnt         = 0;
    long               offsetfr     = 0;
    long               offsetfrcnt  = 0;
    int                cntDelayMode = 0;

    imageID   IDout;
    long      kk;
    uint32_t *arraysize;
    long      naxis;
    uint8_t   datatype;
    char     *ptr0s; // source start 3D array ptr
    char     *ptr0;  // source
    char     *ptr1;  // dest
    long      framesize;
    //    int        semval;

    int                RT_priority = 80; //any number from 0-99
    struct sched_param schedpar;

    long            twait1;
    struct timespec t0;
    struct timespec t1;
    double          tdiffv;
    struct timespec tdiff;

    int SyncSlice = 0;

    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster

    PROCESSINFO *processinfo;
    if(dcprocinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        snprintf(pinfoname, 200, "streamloop-%s", IDoutname);

        char msgstring[200];
        snprintf(msgstring, 200, "%s->%s", IDinname, IDoutname);

        PROCESSINFO_AUX_SETUP(processinfo, pinfoname, "", msgstring);
    }

    if(NBcubes < 1)
    {
        PRINT_ERROR("invalid number of input cubes, needs to be >0");
        return RETURN_FAILURE;
    }

    int sync_semwaitindex;
    IDin      = (long *) malloc(sizeof(long) * NBcubes);
    SyncSlice = 0;
    if(NBcubes == 1)
    {
        {
            IMGID img = imgid_make_from_name(IDinname);
            resolveIMGID(&img,  ERRMODE_ABORT, dcimg, dcnimg);
            IDin[0] = img.ID;
        }

        // in single cube mode, optional sync stream drives updates to next slice within cube
        {
            IMGID img = imgid_make_from_name(IDsync_name);
            resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
            IDsync = img.ID;
        }
        if(IDsync != -1)
        {
            SyncSlice = 1;
            sync_semwaitindex = ImageStreamIO_getsemwaitindex(&dcimg[IDsync], semtrig);
        }
    }
    else
    {
        {
            IMGID img = imgid_make_from_name(IDsync_name);
            resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
            IDsync = img.ID;
        }
        sync_semwaitindex = ImageStreamIO_getsemwaitindex(&dcimg[IDsync], semtrig);

        for(cubeindex = 0;
                cubeindex < NBcubes;
                cubeindex++)
        {
            snprintf(imname, sizeof(imname), "%s_%03ld", IDinname, cubeindex);
            {
                IMGID img = imgid_make_from_name(imname);
                resolveIMGID(&img,  ERRMODE_ABORT, dcimg, dcnimg);
                IDin[cubeindex] = img.ID;
            }
        }
        offsetfr = (long)(0.5 + 1.0 * offsetus / usperiod);

        printf("FRAMES OFFSET = %ld\n", offsetfr);
    }

    printf("SyncSlice = %d\n", SyncSlice);

    printf("Creating / connecting to image stream ...\n");
    fflush(stdout);

    naxis     = dcimg[IDin[0]].md[0].naxis;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        PRINT_ERROR("input image %s should be 3D", IDinname);
        free(arraysize);
        return RETURN_FAILURE;
    }
    arraysize[0] = dcimg[IDin[0]].md[0].size[0];
    arraysize[1] = dcimg[IDin[0]].md[0].size[1];
    arraysize[2] = dcimg[IDin[0]].md[0].size[2];

    datatype = dcimg[IDin[0]].md[0].datatype;

    {
        IMGID img = imgid_make_from_name(IDoutname);
        resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
        IDout = img.ID;
    }
    if(IDout == -1)
    {
        IMGID imgout_tmp = imgid_make_from_name(IDoutname);
        imgout_tmp.mdt->naxis = 2;
        imgout_tmp.mdt->size[0] = arraysize[0];
        imgout_tmp.mdt->size[1] = arraysize[1];
        imgout_tmp.mdt->datatype = datatype;
        imgout_tmp.mdt->shared = 1;
        imgout_tmp.im = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }

    cubeindex = 0;
    pcnt      = 0;
    if(NBcubes > 1)
    {
        cntsync = dcimg[IDsync].md[0].cnt0;
    }

    twait1       = usperiod;
    kk           = 0;
    cntDelayMode = 0;

    if(dcprocinfo == 1)
    {
        processinfo->loopstat = 1; // loop running
    }
    int  loopOK       = 1;
    int  loopCTRLexit = 0; // toggles to 1 when loop is set to exit cleanly
    long loopcnt      = 0;

    while(loopOK == 1)
    {

        // processinfo control
        if(dcprocinfo == 1)
        {
            while(processinfo->CTRLval == 1)  // pause
            {
                usleep(50);
            }

            if(processinfo->CTRLval == 2)  // single iteration
            {
                processinfo->CTRLval = 1;
            }

            if(processinfo->CTRLval == 3)  // exit loop
            {
                loopCTRLexit = 1;
            }
        }

        if(NBcubes > 1)
        {
            if(cntsync != dcimg[IDsync].md[0].cnt0)
            {
                pcnt++;
                cntsync = dcimg[IDsync].md[0].cnt0;
            }
            if(pcnt == period)
            {
                pcnt         = 0;
                offsetfrcnt  = 0;
                cntDelayMode = 1;
            }

            if(cntDelayMode == 1)
            {
                if(offsetfrcnt < offsetfr)
                {
                    offsetfrcnt++;
                }
                else
                {
                    cntDelayMode = 0;
                    cubeindex++;
                    kk = 0;
                }
            }
            if(cubeindex == NBcubes)
            {
                cubeindex = 0;
            }
        }

        ptr0s     = (char *) dcimg[IDin[cubeindex]].array.raw;
        ptr1      = (char *) dcimg[IDout].array.raw;
        framesize = dcimg[IDin[cubeindex]].md[0].size[0] *
                    dcimg[IDin[cubeindex]].md[0].size[1] * ImageStreamIO_typesize(datatype);

        clock_gettime(CLOCK_MILK, &t0);

        ptr0                          = ptr0s + kk * framesize;
        dcimg[IDout].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, framesize);
        dcimg[IDout].md[0].cnt1 = kk;
        dcimg[IDout].md[0].cnt0++;
        dcimg[IDout].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

        kk++;
        if(kk == dcimg[IDin[0]].md[0].size[2])
        {
            kk = 0;
        }

        if(SyncSlice == 0)
        {
            usleep(twait1);

            clock_gettime(CLOCK_MILK, &t1);
            tdiff  = timespec_diff(t0, t1);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

            if(tdiffv < 1.0e-6 * usperiod)
            {
                twait1++;
            }
            else
            {
                twait1--;
            }

            if(twait1 < 0)
            {
                twait1 = 0;
            }
            if(twait1 > usperiod)
            {
                twait1 = usperiod;
            }
        }
        else
        {
            ImageStreamIO_semwait(dcimg + IDsync, sync_semwaitindex);
        }

        if(loopCTRLexit == 1)
        {
            loopOK = 0;
            if(dcprocinfo == 1)
            {
                struct timespec tstop;
                struct tm      *tstoptm;
                char            msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

                clock_gettime(CLOCK_MILK, &tstop);
                tstoptm = gmtime(&tstop.tv_sec);

                snprintf(msgstring,
                         sizeof(msgstring),
                         "CTRLexit at"
                         " %02d:%02d:%02d.%03d",
                         tstoptm->tm_hour,
                         tstoptm->tm_min, tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
                strncpy(processinfo->statusmsg, msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG - 1);

                processinfo->loopstat = 3; // clean exit
            }
        }

        loopcnt++;
        if(dcprocinfo == 1)
        {
            processinfo->loopcnt = loopcnt;
        }
    }

    free(IDin);

    return IDout;
}

// takes a 3Dimage (circular buffer) and writes slices to a 2D image synchronized with an image semaphore
imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(
    const char                  *IDinname,
    const char                  *IDoutname,
    long                        period,
    long                        offsetus,
    const char                  *IDsync_name,
    int                         semtrig,
    __attribute__((unused)) int timingmode)
{
    imageID IDin;
    imageID IDout;
    imageID IDsync;

    long kk;
    long kk1;

    uint32_t *arraysize;
    long      naxis;
    uint8_t   datatype;
    char     *ptr0s; // source start 3D array ptr
    char     *ptr0;  // source
    char     *ptr1;  // dest
    long      framesize;
    //    int        semval;

    int                RT_priority = 80; //any number from 0-99
    struct sched_param schedpar;

    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster

    printf("Creating / connecting to image stream ...\n");
    fflush(stdout);

    {
        IMGID img = imgid_make_from_name(IDinname);
        resolveIMGID(&img,  ERRMODE_ABORT, dcimg, dcnimg);
        IDin = img.ID;
    }
    naxis     = dcimg[IDin].md[0].naxis;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        PRINT_ERROR("input image %s should be 3D", IDinname);
        free(arraysize);
        return RETURN_FAILURE;
    }
    arraysize[0] = dcimg[IDin].md[0].size[0];
    arraysize[1] = dcimg[IDin].md[0].size[1];
    arraysize[2] = dcimg[IDin].md[0].size[2];

    datatype = dcimg[IDin].md[0].datatype;

    {
        IMGID img = imgid_make_from_name(IDoutname);
        resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
        IDout = img.ID;
    }
    if(IDout == -1)
    {
        IMGID imgout_tmp = imgid_make_from_name(IDoutname);
        imgout_tmp.mdt->naxis = 2;
        imgout_tmp.mdt->size[0] = arraysize[0];
        imgout_tmp.mdt->size[1] = arraysize[1];
        imgout_tmp.mdt->datatype = datatype;
        imgout_tmp.mdt->shared = 1;
        imgout_tmp.im = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }

    ptr0s     = (char *) dcimg[IDin].array.raw;
    ptr1      = (char *) dcimg[IDout].array.raw;
    framesize = dcimg[IDin].md[0].size[0] *
                dcimg[IDin].md[0].size[1] * ImageStreamIO_typesize(datatype);

    {
        IMGID img = imgid_make_from_name(IDsync_name);
        resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
        IDsync = img.ID;
    }

    kk  = 0;
    kk1 = 0;

    int sync_semwaitindex;
    sync_semwaitindex = ImageStreamIO_getsemwaitindex(&dcimg[IDin], semtrig);

    while(1)
    {
        ImageStreamIO_semwait(dcimg + IDsync, sync_semwaitindex);

        kk++;
        if(kk == period)  // UPDATE
        {
            kk = 0;
            kk1++;
            if(kk1 == dcimg[IDin].md[0].size[2])
            {
                kk1 = 0;
            }
            usleep(offsetus);
            ptr0                          = ptr0s + kk1 * framesize;
            dcimg[IDout].md[0].write = 1;
            memcpy((void *) ptr1, (void *) ptr0, framesize);
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
            dcimg[IDout].md[0].cnt0++;
            dcimg[IDout].md[0].write = 0;
        }
    }

    // release semaphore
    dcimg[IDsync].semReadPID[sync_semwaitindex] = 0;

    return IDout;
}

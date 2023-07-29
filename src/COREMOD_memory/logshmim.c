/**
 * @file    logshmim.c
 * @brief   Save telemetry stream data
 */

#define _GNU_SOURCE

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_memory/image_keyword_addD.h"
#include "COREMOD_memory/image_keyword_addS.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"
#include "stream_sem.h"

#include "shmimlog_types.h"

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

static long tret = 0; // thread return value










// stream to save
//
static char *streamname;

static int64_t *saveON;
static long     fpi_saveON = -1;

static int64_t *lastcubeON;
static long     fpi_lastcubeON = -1;


static int64_t *nextcube;
static long     fpi_nextcube = -1;


static uint32_t *cubesize;
static long      fpi_cubesize = -1;

// directory where FITS files are written
static char *savedirname;
static long  fpi_savedirname = -1;


// current frame insdex within cube
static uint64_t *frameindex;
static long     fpi_frameindex = -1;

// current frame count since started logging
static uint64_t *framecnt;
static long     fpi_framecnt = -1;


static uint64_t *maxframecnt;
static long     fpi_maxframecnt = -1;


static uint64_t *filecnt;
static long     fpi_filecnt = -1;

static uint64_t *maxfilecnt;
static long     fpi_maxfilecnt = -1;


static int64_t *compressON;
static long     fpi_compressON = -1;



// time taken to save to filesystem
static float *savetime;
static long     fpi_savetime = -1;




static char *outfname;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".sname",
        "stream image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &streamname,
        NULL
    },
    {
        CLIARG_ONOFF,
        ".saveON",
        "toggle save on/off",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &saveON,
        &fpi_saveON
    },
    {
        CLIARG_ONOFF,
        ".lastcubeON",
        "toggle last cube on/off",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &lastcubeON,
        &fpi_lastcubeON
    },
    {
        CLIARG_ONOFF,
        ".nextcube",
        "force jump to next cube",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &nextcube,
        &fpi_nextcube
    },
    {
        CLIARG_UINT32,
        ".cubesize",
        "cube size, nb frame per cube",
        "10000",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &cubesize,
        &fpi_cubesize
    },
    {
        CLIARG_STR,
        ".dirname",
        "log directory",
        "/mnt/datalog/",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &savedirname,
        &fpi_savedirname
    },
    {
        CLIARG_UINT64,
        ".frameindex",
        "frame index within cube (output)",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &frameindex,
        &fpi_frameindex
    },
    {
        CLIARG_UINT64,
        ".framecnt",
        "frame counter since stated logging (output)",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &framecnt,
        &fpi_framecnt
    },
    {
        CLIARG_UINT64,
        ".maxframecnt",
        "max frame count",
        "100000000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &maxframecnt,
        &fpi_maxframecnt
    },
    {
        CLIARG_UINT64,
        ".filecnt",
        "file counter (output)",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &filecnt,
        &fpi_filecnt
    },
    {
        CLIARG_UINT64,
        ".maxfilecnt",
        "max file counter (output)",
        "100000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &maxfilecnt,
        &fpi_maxfilecnt
    },
    {
        CLIARG_STR,
        ".outfname",
        "output file name",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &outfname,
        NULL
    },
    {
        CLIARG_ONOFF,
        ".compress",
        "toggle compression on/off",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &compressON,
        &fpi_compressON
    },
    {
        CLIARG_FLOAT32,
        ".savetime",
        "time taken to save",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &savetime,
        &fpi_savetime
    }
};



static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        // can toggle while running
        data.fpsptr->parray[fpi_saveON].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_lastcubeON].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_nextcube].fpflag |= FPFLAG_WRITERUN;

        data.fpsptr->parray[fpi_savedirname].fpflag |= FPFLAG_WRITERUN;

        data.fpsptr->parray[fpi_cubesize].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_maxfilecnt].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_maxframecnt].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_compressON].fpflag |= FPFLAG_WRITERUN;
    }

    return RETURN_SUCCESS;
}


static errno_t customCONFcheck()
{

    return RETURN_SUCCESS;
}



static CLICMDDATA CLIcmddata =
{
    "streamFITSlog",
    "log stream to FITS file(s)",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}


















/**
 * ## Purpose
 *
 * Save telemetry stream data
 * Writes FITS file and timing file
 *
 */
static void *save_telemetry_fits_function(
    void *ptr
)
{
    STREAMSAVE_THREAD_MESSAGE *tmsg;
    tmsg = (STREAMSAVE_THREAD_MESSAGE *) ptr;

    printf(">>>>>>>> [%5d] tmsg->iname     = \"%s\"\n", __LINE__, tmsg->iname);
    printf(">>>>>>>> [%5d] tmsg->fname     = \"%s\"\n", __LINE__, tmsg->fname);
    printf(">>>>>>>> [%5d] tmsg->cubesize  = %ld\n", __LINE__, tmsg->cubesize);

//    struct timespec tstart;
//    clock_gettime(CLOCK_MILK, &tstart);

    /*

        // Set save function to RT priority 0
        // This is meant to be lower priority than the data collection into buffers
        //
        int                RT_priority = 0;
        struct sched_param schedpar;

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

    */




    /*
        // Add custom keywords
        int            NBcustomKW = 9;
        IMAGE_KEYWORD *imkwarray =
            (IMAGE_KEYWORD *) malloc(sizeof(IMAGE_KEYWORD) * NBcustomKW);


        // UT time

        strcpy(imkwarray->name, "UT");
        imkwarray->type = 'S';


        strcpy(imkwarray->value.valstr,
               timedouble_to_UTC_timeofdaystring(
                   0.5 * tmsg->arraytime[0] +
                   0.5 * tmsg->arraytime[tmsg->cubesize - 1]));
        strcpy(imkwarray->comment, "HH:MM:SS.SS typical UTC at exposure");


        strcpy(imkwarray[1].name, "UT-STR");
        imkwarray[1].type = 'S';
        strcpy(imkwarray[1].value.valstr,
               timedouble_to_UTC_timeofdaystring(tmsg->arraytime[0]));
        strcpy(imkwarray[1].comment, "HH:MM:SS.SS UTC at exposure start");

        strcpy(imkwarray[2].name, "UT-END");
        imkwarray[2].type = 'S';
        strcpy(
            imkwarray[2].value.valstr,
            timedouble_to_UTC_timeofdaystring(tmsg->arraytime[tmsg->cubesize - 1]));
        strcpy(imkwarray[2].comment, "HH:MM:SS.SS UTC at exposure end");

        // Modified Julian Date (MJD)


        strcpy(imkwarray[3].name, "MJD");
        imkwarray[3].type = 'D';
        imkwarray[3].value.numf =
            (0.5 * tmsg->arraytime[0] + 0.5 * tmsg->arraytime[tmsg->cubesize - 1]) /
            86400.0 +
            40587.0;
        strcpy(imkwarray[3].comment, "Modified Julian Day at exposure");


        strcpy(imkwarray[4].name, "MJD-STR");
        imkwarray[4].type       = 'D';
        imkwarray[4].value.numf = tmsg->arraytime[0] / 86400.0 + 40587.0;
        strcpy(imkwarray[4].comment, "Modified Julian Day at exposure start");

        strcpy(imkwarray[5].name, "MJD-END");
        imkwarray[5].type = 'D';
        imkwarray[5].value.numf =
            (tmsg->arraytime[tmsg->cubesize - 1] / 86400.0) + 40587.0;
        strcpy(imkwarray[5].comment, "Modified Julian Day at exposure end");

        // Local time

        // get time zone
        //char tm_zone[] = "HST";
        //double tm_utcoff = -36000; // HST = UTC - 10; Positive east of UTC.


        // Causes a race condition with gettime in other thread, which result in occasional HST filenames...
        //time_t t = time(NULL);
        // OVERRIDE localtime to HST
        //putenv("TZ=Pacific/Honolulu");
        //struct tm lt = *localtime(&t);
        //printf("TIMEZONE TIMEZONE %s\n", lt.tm_zone);
        //putenv("TZ=");
        //printf("TIMEZONE TIMEZONE %s\n", lt.tm_zone);


        // printf("Offset to GMT is %lds.\n", lt.tm_gmtoff);
        // printf("The time zone is '%s'.\n", lt.tm_zone);


        sprintf(imkwarray[6].name, "%s", TZ_MILK_STR);
        imkwarray[6].type = 'S';
        strcpy(imkwarray[6].value.valstr,
               timedouble_to_UTC_timeofdaystring(
                   (0.5 * tmsg->arraytime[0] +
                    0.5 * tmsg->arraytime[tmsg->cubesize - 1]) +
                   TZ_MILK_UTC_OFF));
        sprintf(imkwarray[6].comment,
                "HH:MM:SS.SS typical %s at exposure",
                TZ_MILK_STR);

        sprintf(imkwarray[7].name, "%s-STR", TZ_MILK_STR);
        imkwarray[7].type = 'S';
        strcpy(
            imkwarray[7].value.valstr,
            timedouble_to_UTC_timeofdaystring(tmsg->arraytime[0] + TZ_MILK_UTC_OFF));
        sprintf(imkwarray[7].comment,
                "HH:MM:SS.SS typical %s at exposure start",
                TZ_MILK_STR);

        sprintf(imkwarray[8].name, "%s-END", TZ_MILK_STR);
        imkwarray[8].type = 'S';
        strcpy(imkwarray[8].value.valstr,
               timedouble_to_UTC_timeofdaystring(
                   tmsg->arraytime[tmsg->cubesize - 1] + TZ_MILK_UTC_OFF));
        sprintf(imkwarray[8].comment,
                "HH:MM:SS.SS typical %s at exposure end",
                TZ_MILK_STR);



        //printf("auxFITSheader = \"%s\"\n", tmsg->fname_auxFITSheader);
        printf(">>>>>>>> [%5d] tmsg->iname  = \"%s\"\n", __LINE__, tmsg->iname);


        saveFITS_opt_trunc(tmsg->iname,
                           tmsg->partial ? tmsg->cubesize : -1,
                           tmsg->fname,
                           0,
                           tmsg->fname_auxFITSheader,
                           imkwarray,
                           NBcustomKW,
                           tmsg->compress_string);


        free(imkwarray);


        if(tmsg->saveascii == 1)
        {
            FILE *fp;

            if((fp = fopen(tmsg->fnameascii, "w")) == NULL)
            {
                printf("ERROR: cannot create file \"%s\"\n", tmsg->fnameascii);
                exit(0);
            }

            fprintf(fp, "# Telemetry stream timing data \n");
            fprintf(fp,
                    "# File written by function %s in file %s\n",
                    __FUNCTION__,
                    __FILE__);
            fprintf(fp, "# \n");
            fprintf(fp, "# col1 : datacube frame index\n");
            fprintf(fp, "# col2 : Main index\n");
            fprintf(fp, "# col3 : Time since cube origin (logging)\n");
            fprintf(fp, "# col4 : Absolute time (logging)\n");
            fprintf(fp, "# col5 : Absolute time (acquisition)\n");
            fprintf(fp, "# col6 : stream cnt0 index\n");
            fprintf(fp, "# col7 : stream cnt1 index\n");
            fprintf(fp, "# \n");

            double t0; // time reference
            t0 = tmsg->arraytime[0];
            for(long k = 0; k < tmsg->cubesize; k++)
            {
                //fprintf(fp, "%6ld   %10lu  %10lu   %15.9lf\n", k, tmsg->arraycnt0[k], tmsg->arraycnt1[k], tmsg->arraytime[k]);

                // entries are:
                // - index within cube
                // - loop index (if applicable)
                // - time since cube start
                // - time (absolute)
                // - cnt0
                // - cnt1

                fprintf(
                    fp,
                    "%10ld  %10lu  %15.9lf   %20.9lf  %17.6lf   %10ld   %10ld\n",
                    k,
                    tmsg->arrayindex[k],
                    tmsg->arraytime[k] - t0,
                    tmsg->arraytime[k],
                    tmsg->arrayaqtime[k],
                    tmsg->arraycnt0[k],
                    tmsg->arraycnt1[k]);
            }
            fclose(fp);
        }


        tret = image_ID(tmsg->iname);

        struct timespec tend;
        clock_gettime(CLOCK_MILK, &tend);

        double timediff = 1.0 * (tend.tv_sec - tstart.tv_sec) +
                          1.0e-9 * (tend.tv_nsec - tstart.tv_nsec);
        tmsg->timespan = timediff;

    */
    pthread_exit(&tret);
}









/** @brief Logs a shared memory stream onto disk
 *
 * uses semlog semaphore
 *
 * uses data cube buffer to store frames
 * if an image name logdata exists (should ideally be in shared mem),
 * then this will be included in the timing txt file
 */

/*
errno_t __attribute__((hot))
COREMOD_MEMORY_sharedMem_2Dim_log(
    const char *IDname,
    uint32_t    zsize,
    const char *logdir,
    const char *IDlogdata_name
)
{
    // WAIT time. If no new frame during this time, save existing cube
    int WaitSec = 5;

    imageID            ID;
    uint32_t           xsize;
    uint32_t           ysize;
    imageID            IDb;
    imageID            IDb0;
    imageID            IDb1;
    long               index = 0;
    unsigned long long cnt   = 0;
    int                buffer;
    uint8_t            datatype;
    uint32_t          *imsizearray;
    char               fname[STRINGMAXLEN_FILENAME];
    char               iname[STRINGMAXLEN_IMGNAME];

    time_t          t;
    struct tm      *uttimeStart;
    struct timespec ts;
    struct timespec timenow;
    struct timespec timenowStart;
    int             ret;
    imageID         IDlogdata;

    char *ptr0_0; // source image data
    char *ptr1_0; // destination image data
    char *ptr0;   // source image data, after offset
    char *ptr1;   // destination image data, after offset

    long framesize; // in bytes

    char fnameascii[200];

    pthread_t                  thread_savefits;
    int                        tOK = 0;
    int                        iret_savefits;
    STREAMSAVE_THREAD_MESSAGE *tmsg = malloc(sizeof(STREAMSAVE_THREAD_MESSAGE));

    long NBfiles = -1; // run forever

    long long cntwait;
    long      waitdelayus = 50;    // max speed = 20 kHz
    long long cntwaitlim  = 10000; // 5 sec
    int       wOK;
    int       noframe;

    char logb0name[500];
    char logb1name[500];

    int is3Dcube = 0; // this is a rolling buffer

    LOGSHIM_CONF *logshimconf;

    // recording time for each frame
    double *array_time;
    double *array_time_cp;
    double *array_aqtime; // acquisition time
    double *array_aqtime_cp;

    // counters
    uint64_t *array_cnt0;
    uint64_t *array_cnt0_cp;
    uint64_t *array_cnt1;
    uint64_t *array_cnt1_cp;

    int                RT_priority = 80; //any number from 0-99
    struct sched_param schedpar;

    int use_semlog;
    int semval;

    int VERBOSE = 0;
    // 0: don't print
    // 1: print statements outside fast loop
    // 2: print everything

    // convert wait time into number of couunter steps (counter mode only)
    cntwaitlim = (long)(WaitSec * 1000000 / waitdelayus);

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

    IDlogdata = image_ID(IDlogdata_name);
    if(IDlogdata != -1)
    {
        if(data.image[IDlogdata].md->datatype != _DATATYPE_FLOAT)
        {
            IDlogdata = -1;
        }
    }
    printf("log data name = %s\n", IDlogdata_name);



    //logshimconf = COREMOD_MEMORY_logshim_create_SHMconf(IDname);

    logshimconf->on       = 1;
    logshimconf->cnt      = 0;
    logshimconf->filecnt  = 0;
    logshimconf->logexit  = 0;
    logshimconf->interval = 1;

    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    read_sharedmem_image(IDname);
    ID       = image_ID(IDname);
    datatype = data.image[ID].md->datatype;
    xsize    = data.image[ID].md->size[0];
    ysize    = data.image[ID].md->size[1];

    if(data.image[ID].md->naxis == 3)
    {
        is3Dcube = 1;
    }

    // create the 2 buffers

    imsizearray[0] = xsize;
    imsizearray[1] = ysize;
    imsizearray[2] = zsize;

    sprintf(logb0name, "%s_logbuff0", IDname);
    sprintf(logb1name, "%s_logbuff1", IDname);

    create_image_ID(logb0name,
                    3,
                    imsizearray,
                    datatype,
                    1,
                    data.image[ID].md->NBkw,
                    0,
                    &IDb0);
    create_image_ID(logb1name,
                    3,
                    imsizearray,
                    datatype,
                    1,
                    data.image[ID].md->NBkw,
                    0,
                    &IDb1);

    // copy keywords
    {
        memcpy(data.image[IDb0].kw,
               data.image[ID].kw,
               sizeof(IMAGE_KEYWORD) * data.image[ID].md->NBkw);
        memcpy(data.image[IDb1].kw,
               data.image[ID].kw,
               sizeof(IMAGE_KEYWORD) * data.image[ID].md->NBkw);
    }

    // find creation time keyword
    // _MAQTIME
    int aqtimekwi = -1;
    for(int kwi = 0; kwi < data.image[ID].md->NBkw; kwi++)
    {
        if(strcmp(data.image[ID].kw[kwi].name, "_MAQTIME") == 0)
        {
            aqtimekwi = kwi;
        }
    }

    COREMOD_MEMORY_image_set_semflush(logb0name, -1);
    COREMOD_MEMORY_image_set_semflush(logb1name, -1);

    array_time   = (double *) malloc(sizeof(double) * zsize);
    array_aqtime = (double *) malloc(sizeof(double) * zsize);
    array_cnt0   = (uint64_t *) malloc(sizeof(uint64_t) * zsize);
    array_cnt1   = (uint64_t *) malloc(sizeof(uint64_t) * zsize);

    array_time_cp   = (double *) malloc(sizeof(double) * zsize);
    array_aqtime_cp = (double *) malloc(sizeof(double) * zsize);
    array_cnt0_cp   = (uint64_t *) malloc(sizeof(uint64_t) * zsize);
    array_cnt1_cp   = (uint64_t *) malloc(sizeof(uint64_t) * zsize);

    IDb = IDb0;

    int typesize = ImageStreamIO_typesize(datatype);
    if(typesize == -1)
    {
        printf("ERROR: WRONG DATA TYPE\n");
        exit(0);
    }
    framesize = typesize * xsize * ysize;
    ptr0_0 = (char *) data.image[ID].array.raw;

    ptr1_0 = (char *) data.image[IDb].array.raw;

    cnt = data.image[ID].md->cnt0 - 1;

    buffer = 0;
    index  = 0;

    printf("logdata ID = %ld\n", IDlogdata);
    list_image_ID();

    // using semlog ?
    use_semlog = 0;
    if(data.image[ID].semlog != NULL)
    {
        use_semlog = 1;
        sem_getvalue(data.image[ID].semlog, &semval);

        // bring semaphore value to 1 to only save 1 frame
        while(semval > 1)
        {
            sem_wait(data.image[ID].semlog);
            sem_getvalue(data.image[ID].semlog, &semval);
        }
        if(semval == 0)
        {
            sem_post(data.image[ID].semlog);
        }
    }

    while((logshimconf->filecnt != NBfiles) && (logshimconf->logexit == 0))
    {
        int timeout; // 1 if timeout has occurred

        cntwait = 0;
        noframe = 0;
        wOK     = 1;

        if(VERBOSE > 1)
        {
            printf("%5d  Entering wait loop   index = %ld %d\n",
                   __LINE__,
                   index,
                   noframe);
        }

        timeout = 0;
        if(likely(use_semlog == 1))
        {
            if(VERBOSE > 1)
            {
                printf("%5d  Waiting for semaphore\n", __LINE__);
            }

            if(clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += WaitSec;

            ret = sem_timedwait(data.image[ID].semlog, &ts);
            if(ret == -1)
            {
                if(errno == ETIMEDOUT)
                {
                    printf(
                        "%5d  sem_timedwait() timed out (%d "
                        "sec) -[index %ld]\n",
                        __LINE__,
                        WaitSec,
                        index);
                    if(VERBOSE > 0)
                    {
                        printf(
                            "%5d  sem time elapsed -> Save "
                            "current cube [index %ld]\n",
                            __LINE__,
                            index);
                    }

                    strcpy(tmsg->iname, iname);
                    strcpy(tmsg->fname, fname);
                    tmsg->partial  = 1; // partial cube
                    tmsg->cubesize = index;

                    memcpy(array_time_cp, array_time, sizeof(double) * index);
                    memcpy(array_aqtime_cp,
                           array_aqtime,
                           sizeof(double) * index);
                    memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t) * index);
                    memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t) * index);

                    tmsg->arrayindex  = array_cnt0_cp;
                    tmsg->arraycnt0   = array_cnt0_cp;
                    tmsg->arraycnt1   = array_cnt1_cp;
                    tmsg->arraytime   = array_time_cp;
                    tmsg->arrayaqtime = array_aqtime_cp;

                    timeout = 1;
                }
                if(errno == EINTR)
                {
                    printf(
                        "%5d  sem_timedwait [index %ld]: The "
                        "call was interrupted by a signal "
                        "handler\n",
                        __LINE__,
                        index);
                }

                if(errno == EINVAL)
                {
                    printf(
                        "%5d  sem_timedwait [index %ld]: Not a "
                        "valid semaphore\n",
                        __LINE__,
                        index);
                    printf(
                        "               The value of "
                        "abs_timeout.tv_nsecs is less than 0, "
                        "or greater than or equal "
                        "to 1000 million\n");
                }

                if(errno == EAGAIN)
                {
                    printf(
                        "%5d  sem_timedwait [index %ld]: The "
                        "operation could not be performed "
                        "without blocking "
                        "(i.e., the semaphore currently has "
                        "the value zero)\n",
                        __LINE__,
                        index);
                }

                wOK = 0;
                if(index == 0)
                {
                    noframe = 1;
                }
                else
                {
                    noframe = 0;
                }
            }
        }
        else
        {
            if(VERBOSE > 1)
            {
                printf("%5d  Not using semaphore, watching counter\n",
                       __LINE__);
            }

            while(((cnt == data.image[ID].md->cnt0) ||
                    (logshimconf->on == 0)) &&
                    (wOK == 1))
            {
                if(VERBOSE > 1)
                {
                    printf("%5d  waiting time step\n", __LINE__);
                }

                usleep(waitdelayus);
                cntwait++;

                if(VERBOSE > 1)
                {
                    printf("%5d  cntwait = %lld\n", __LINE__, cntwait);
                    fflush(stdout);
                }

                if(cntwait > cntwaitlim)  // save current cube
                {
                    if(VERBOSE > 0)
                    {
                        printf(
                            "%5d  cnt time elapsed -> Save "
                            "current cube\n",
                            __LINE__);
                    }

                    strcpy(tmsg->iname, iname);
                    strcpy(tmsg->fname, fname);
                    tmsg->partial  = 1; // partial cube
                    tmsg->cubesize = index;

                    memcpy(array_time_cp, array_time, sizeof(double) * index);
                    memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t) * index);
                    memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t) * index);

                    tmsg->arrayindex  = array_cnt0_cp;
                    tmsg->arraycnt0   = array_cnt0_cp;
                    tmsg->arraycnt1   = array_cnt1_cp;
                    tmsg->arraytime   = array_time_cp;
                    tmsg->arrayaqtime = array_aqtime_cp;

                    wOK = 0;
                    if(index == 0)
                    {
                        noframe = 1;
                    }
                    else
                    {
                        noframe = 0;
                    }
                }
            }
        }

        if(index == 0)
        {
            if(VERBOSE > 0)
            {
                printf("%5d  Setting cube start time [index %ld]\n",
                       __LINE__,
                       index);
            }

            /// measure time
            t           = time(NULL);
            uttimeStart = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenowStart);

            //     sprintf(fname,"%s/%s_%02d:%02d:%02ld.%09ld.fits", logdir, IDname, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
            //            sprintf(fnameascii,"%s/%s_%02d:%02d:%02ld.%09ld.txt", logdir, IDname, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
        }

        if(VERBOSE > 1)
        {
            printf("%5d  logshimconf->on = %d\n",
                   __LINE__,
                   logshimconf->on);
        }

        if(likely(logshimconf->on == 1))
        {
            if(likely(wOK == 1))  // normal step: a frame has arrived
            {
                if(VERBOSE > 1)
                {
                    printf("%5d  Frame has arrived [index %ld]\n",
                           __LINE__,
                           index);
                }

                /// measure time
                //   t = time(NULL);
                //   uttime = gmtime(&t);

                clock_gettime(CLOCK_MILK, &timenow);

                if(is3Dcube == 1)
                {
                    ptr0 = ptr0_0 + framesize * data.image[ID].md->cnt1;
                }
                else
                {
                    ptr0 = ptr0_0;
                }

                ptr1 = ptr1_0 + framesize * index;

                if(VERBOSE > 1)
                {
                    printf("%5d  memcpy framesize = %ld\n",
                           __LINE__,
                           framesize);
                }

                memcpy((void *) ptr1, (void *) ptr0, framesize);

                if(VERBOSE > 1)
                {
                    printf("%5d  memcpy done\n", __LINE__);
                }

                array_cnt0[index] = data.image[ID].md->cnt0;
                array_cnt1[index] = data.image[ID].md->cnt1;
                //array_time[index] = uttime->tm_hour*3600.0 + uttime->tm_min*60.0 + timenow.tv_sec % 60 + 1.0e-9*timenow.tv_nsec;
                array_time[index] = timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;
                if(aqtimekwi != -1)
                {
                    array_aqtime[index] =
                        1.0e-6 * data.image[ID].kw[aqtimekwi].value.numl;
                }
                else
                {
                    array_aqtime[index] = 0.0;
                }

                index++;
            }
        }
        else
        {
            // save partial if possible
            //if(index>0)
            wOK = 0;
        }

        if(VERBOSE > 1)
        {
            printf("%5d  index = %ld  wOK = %d\n", __LINE__, index, wOK);
        }

        // SAVE CUBE TO DISK
        // cases:
        //   index>zsize-1  buffer full
        //   timeout==1 && index>0 : partial due to timeout
        //   logshimconf->on == 0 && index>0 : partial due to logshimoff
        if((index > zsize - 1) || ((timeout == 1) && (index > 0)) ||
                ((logshimconf->on == 0) && (index > 0)))
        {
            long NBframemissing;

            /// save image
            if(VERBOSE > 0)
            {
                printf(
                    "%5d  Save image   [index  %ld]  [timeout %d] "
                    "[zsize %ld]\n",
                    __LINE__,
                    index,
                    timeout,
                    (long) zsize);
            }

            sprintf(iname, "%s_logbuff%d", IDname, buffer);
            if(buffer == 0)
            {
                IDb = IDb0;
            }
            else
            {
                IDb = IDb1;
            }

            // update buffer content
            memcpy(data.image[IDb].kw,
                   data.image[ID].kw,
                   sizeof(IMAGE_KEYWORD) * data.image[ID].md->NBkw);

            if(VERBOSE > 0)
            {
                printf("%5d  Building file name: ascii\n", __LINE__);
                fflush(stdout);
            }

            sprintf(fnameascii,
                    "%s/%s_%02d:%02d:%02ld.%09ld.txt",
                    logdir,
                    IDname,
                    uttimeStart->tm_hour,
                    uttimeStart->tm_min,
                    timenowStart.tv_sec % 60,
                    timenowStart.tv_nsec);

            if(VERBOSE > 0)
            {
                printf("%5d  Building file name: fits\n", __LINE__);
                fflush(stdout);
            }
            sprintf(fname,
                    "%s/%s_%02d:%02d:%02ld.%09ld.fits",
                    logdir,
                    IDname,
                    uttimeStart->tm_hour,
                    uttimeStart->tm_min,
                    timenowStart.tv_sec % 60,
                    timenowStart.tv_nsec);

            strcpy(tmsg->iname, iname);
            strcpy(tmsg->fname, fname);
            strcpy(tmsg->fnameascii, fnameascii);
            tmsg->saveascii = 1;

            if(wOK == 1)  // full cube
            {
                tmsg->partial = 0; // full cube
                if(VERBOSE > 0)
                {
                    printf("%5d  SAVING FULL CUBE\n", __LINE__);
                    fflush(stdout);
                }
            }
            else // partial cube
            {
                tmsg->partial = 1; // partial cube
                if(VERBOSE > 0)
                {
                    printf("%5d  SAVING PARTIAL CUBE\n", __LINE__);
                    fflush(stdout);
                }
            }

            {
                long cnt0start = data.image[ID].md->cnt0;

                // Wait for save thread to complete to launch next one
                if(tOK == 1)
                {
                    if(pthread_tryjoin_np(thread_savefits, NULL) == EBUSY)
                    {
                        if(VERBOSE > 0)
                        {
                            printf(
                                "%5d  PREVIOUS SAVE THREAD "
                                "NOT TERMINATED -> "
                                "waiting\n",
                                __LINE__);
                        }
                        pthread_join(thread_savefits, NULL);
                        if(VERBOSE > 0)
                        {
                            printf(
                                "%5d  PREVIOUS SAVE THREAD "
                                "NOW COMPLETED -> "
                                "continuing\n",
                                __LINE__);
                        }
                    }
                    else
                    {
                        if(VERBOSE > 0)
                        {
                            printf(
                                "%5d  PREVIOUS SAVE THREAD "
                                "ALREADY COMPLETED -> OK\n",
                                __LINE__);
                        }
                    }
                }

                printf("\n ************** MISSED = %ld\n",
                       data.image[ID].md->cnt0 - cnt0start);
            }

            COREMOD_MEMORY_image_set_sempost_byID(IDb, -1);
            data.image[IDb].md->cnt0++;
            data.image[IDb].md->write = 0;

            tmsg->cubesize = index;
            strcpy(tmsg->iname, iname);

            memcpy(array_time_cp, array_time, sizeof(double) * index);
            memcpy(array_aqtime_cp, array_aqtime, sizeof(double) * index);

            memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t) * index);

            memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t) * index);

            NBframemissing =
                (array_cnt0[index - 1] - array_cnt0[0]) - (index - 1);

            printf(
                "=>=>=>=>= CUBE %8lld   Number of missed frames = %8ld "
                " / %ld  / %8ld ====\n",
                logshimconf->filecnt,
                NBframemissing,
                index,
                (long) zsize);

            if(VERBOSE > 0)
            {
                printf("%5d  Starting image save thread\n", __LINE__);
                fflush(stdout);
            }

            tmsg->arrayindex  = array_cnt0_cp;
            tmsg->arraycnt0   = array_cnt0_cp;
            tmsg->arraycnt1   = array_cnt1_cp;
            tmsg->arraytime   = array_time_cp;
            tmsg->arrayaqtime = array_aqtime_cp;
            WRITE_FILENAME(tmsg->fname_auxFITSheader,
                           "%s/%s.aux.fits",
                           data.shmdir,
                           IDname);

            strcpy(tmsg->compress_string, "[compress R 1,1,80000]");

            if ( (*compressON) == 0 )
            {
                strcpy(tmsg->compress_string, "[compress R 1,1,90000]");
            }
            else
            {
                strcpy(tmsg->compress_string, "[compress R 1,1,10000]");
            }


            iret_savefits = pthread_create(&thread_savefits,
                                           NULL,
                                           save_telemetry_fits_function,
                                           tmsg);

            logshimconf->cnt++;

            tOK = 1;
            if(iret_savefits)
            {
                fprintf(stderr,
                        "Error - pthread_create() return code: %d\n",
                        iret_savefits);
                exit(EXIT_FAILURE);
            }

            index = 0;
            buffer++;
            if(buffer == 2)
            {
                buffer = 0;
            }
            //            printf("[%ld -> %d]", cnt, buffer);
            //           fflush(stdout);
            if(buffer == 0)
            {
                IDb = IDb0;
            }
            else
            {
                IDb = IDb1;
            }

            ptr1_0 = (char *) data.image[IDb].array.raw;

            data.image[IDb].md->write = 1;
            logshimconf->filecnt++;
        }

        cnt = data.image[ID].md->cnt0;
    }

    free(imsizearray);
    free(tmsg);

    free(array_time);
    free(array_aqtime);
    free(array_cnt0);
    free(array_cnt1);

    free(array_time_cp);
    free(array_aqtime_cp);
    free(array_cnt0_cp);
    free(array_cnt1_cp);

    return RETURN_SUCCESS;
}

*/

















static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();



    int VERBOSE = 2;
    // 0: don't print
    // 1: print statements outside fast loop
    // 2: print everything

    STREAMSAVE_THREAD_MESSAGE *tmsg = (STREAMSAVE_THREAD_MESSAGE*) malloc(sizeof(STREAMSAVE_THREAD_MESSAGE));


    IMGID inimg = mkIMGID_from_name(streamname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    uint32_t xsize = inimg.md->size[0];
    uint32_t ysize = inimg.md->size[1];
    uint32_t zsize = (*cubesize);
    uint8_t datatype = inimg.md->datatype;


    int typesize = ImageStreamIO_typesize(datatype);
    if(typesize == -1)
    {
        printf("ERROR: WRONG DATA TYPE\n");
        exit(0);
    }

    int buffindex = 0;

    // Create 2 log buffers
    //
    /*IMGID imgbuff0;
    {
        char name[STRINGMAXLEN_STREAMNAME];
        WRITE_IMAGENAME(name, "%s_logbuff0", streamname);
        imgbuff0 =
            stream_connect_create_3D(name, xsize, ysize, zsize, datatype);
    }
    IMGID imgbuff1;
    {
        char name[STRINGMAXLEN_STREAMNAME];
        WRITE_IMAGENAME(name, "%s_logbuff1", streamname);
        imgbuff1 =
            stream_connect_create_3D(name, xsize, ysize, zsize, datatype);
    }*/






    // copy keywords
    /*  {
          printf("Cppying %d keywords\n", inimg.md->NBkw);
          if( inimg.md->NBkw > 0 )
          {
              memcpy(imgbuff0.im->kw,
                     inimg.im->kw,
                     sizeof(IMAGE_KEYWORD) * inimg.md->NBkw);
              memcpy(imgbuff1.im->kw,
                     inimg.im->kw,
                     sizeof(IMAGE_KEYWORD) * inimg.md->NBkw);
          }
      }
    */


    // find creation time keyword
    // _MAQTIME
    int aqtimekwi = -1;
    /*
    for(int kwi = 0; kwi < inimg.md->NBkw; kwi++)
    {
        if(strcmp(inimg.im->kw[kwi].name, "_MAQTIME") == 0)
        {
            aqtimekwi = kwi;
        }
    }
    if(VERBOSE > 0)
    {
        printf("[%5d] aqtimekwi = %d\n", __LINE__, aqtimekwi);
    }
    */


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    // custom initialization
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    int saveON_last = (*saveON);

    char FITSffilename[STRINGMAXLEN_FULLFILENAME];
    strcpy(FITSffilename,"null");

    char ASCIITIMEffilename[STRINGMAXLEN_FULLFILENAME];
    strcpy(ASCIITIMEffilename,"null");



    // array are zsize * 2 long to hold double buffer
    //
    //double * array_time   = (double *) malloc(sizeof(double) * (*cubesize) * 2);
    //double * array_aqtime = (double *) malloc(sizeof(double) * (*cubesize) * 2);
    //uint64_t * array_cnt0   = (uint64_t *) malloc(sizeof(uint64_t) * (*cubesize) * 2);
    //uint64_t * array_cnt1   = (uint64_t *) malloc(sizeof(uint64_t) * (*cubesize) * 2);



    int thread_initialized = 0;

    // inittialization
    *framecnt = 0;
    *frameindex = 0;
    *filecnt = 0;

    // set to 1 if we're on the last cube
    int lastcube = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)
        {
            printf("------------ TIMEOUT\n");
        }
        else
        {
            // new frame has arrived

            if( (saveON_last == 0) && ((*saveON) == 1) )
            {
                // We just turned on saving
                lastcube = 0;
                (*framecnt) = 0;
                (*filecnt) = 0;
            }



            if ((*framecnt) >= (*maxframecnt))
            {
                // we've logged the requested number of frames
                (*saveON) = 0;
                data.fpsptr->parray[fpi_saveON].fpflag &= ~FPFLAG_ONOFF;
            }


            if ((*filecnt) >= (*maxfilecnt)-1 )
            {
                // last cube
                lastcube = 1;
            }





            if ( (*saveON) == 1 )
            {
                /*if((*frameindex) == 0)
                {
                    // measure time at cube start
                    // construc filenames

                    time_t          t;
                    struct tm      *uttimeStart;
                    t           = time(NULL);
                    uttimeStart = gmtime(&t);
                    struct timespec timenowStart;
                    clock_gettime(CLOCK_MILK, &timenowStart);

                    WRITE_FULLFILENAME(FITSffilename,
                                       "%s/%s_%02d:%02d:%02ld.%09ld.fits",
                                       savedirname,
                                       streamname,
                                       uttimeStart->tm_hour,
                                       uttimeStart->tm_min,
                                       timenowStart.tv_sec % 60,
                                       timenowStart.tv_nsec);

                    if(VERBOSE > 0)
                    {
                        printf("[%5d] FITSffilename = %s\n", __LINE__, FITSffilename);
                    }


                    WRITE_FULLFILENAME(ASCIITIMEffilename,
                                       "%s/%s_%02d:%02d:%02ld.%09ld.txt",
                                       savedirname,
                                       streamname,
                                       uttimeStart->tm_hour,
                                       uttimeStart->tm_min,
                                       timenowStart.tv_sec % 60,
                                       timenowStart.tv_nsec);

                    if(VERBOSE > 0)
                    {
                        printf("[%5d] ASCIITIMEffilename = %s\n", __LINE__, ASCIITIMEffilename);
                    }
                }*/


                // timing buffer index
                /*{
                    long tindex = (*frameindex) + buffindex*(*cubesize);
                    {
                        array_cnt0[tindex] = inimg.md->cnt0;
                        array_cnt1[tindex] = inimg.md->cnt1;

                        // get current time
                        struct timespec timenow;
                        clock_gettime(CLOCK_MILK, &timenow);
                        array_time[tindex] = timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

                        if(aqtimekwi != -1)
                        {
                            array_aqtime[tindex] =
                                1.0e-6 * inimg.im->kw[aqtimekwi].value.numl;
                        }
                        else
                        {
                            array_aqtime[tindex] = 0.0;
                        }
                    }
                }*/


                // copy frame to buffer
                /*
                {
                    long framesize = typesize * xsize * ysize;

                    char *ptr0_0; // source image data
                    char *ptr0;   // source image data, after offset

                    ptr0_0 = (char *) inimg.im->array.raw;
                    if( inimg.md->naxis == 3)
                    {
                        // this is a rolling buffer
                        ptr0 = ptr0_0 + framesize * inimg.md->cnt1;
                    }
                    else
                    {
                        ptr0 = ptr0_0;
                    }


                    char *ptr1_0; // destination image data
                    char *ptr1;   // destination image data, after offset
                    if(buffindex == 0 )
                    {
                        ptr1_0 = (char *) imgbuff0.im->array.raw;
                    }
                    else
                    {
                        ptr1_0 = (char *) imgbuff1.im->array.raw;
                    }
                    ptr1 = ptr1_0 + framesize * (*frameindex);


                    memcpy((void *) ptr1, (void *) ptr0, framesize);
                }*/




                processinfo_WriteMessage_fmt(
                    processinfo,
                    "buff %d file %lu frameindex %lu",
                    buffindex,
                    (*filecnt),
                    (*frameindex));

                (*frameindex) ++;
                (*framecnt) ++;
            }
            else
            {
                processinfo_WriteMessage(processinfo, "save = OFF");
            }
        }




        // Should we save current cube ?

        int SaveCube = 0;

        if( (*frameindex) >= (*cubesize) )
        {
            // cube is full
            SaveCube = 1;
        }

        if( (saveON_last == 1) && ((*saveON) == 0) )
        {
            // We just turned off saving
            SaveCube = 1;
        }

        if( (*nextcube) == 1)
        {
            (*nextcube) = 0;
            data.fpsptr->parray[fpi_nextcube].fpflag &= ~FPFLAG_ONOFF;
            SaveCube = 1;
        }

        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)
        {
            SaveCube = 1;
        }







        if(SaveCube == 1)
        {
            if((*frameindex) > 0)
            {
                // Saving buffer to filesystem
                //

                printf("SAVING %5ld FRAMES of BUFFER %d to FILE %s\n", (*frameindex), buffindex, FITSffilename);
                fflush(stdout);


                // update buffer content
                /*
                if(buffindex == 0 )
                {
                    memcpy(imgbuff0.im->kw,
                           inimg.im->kw,
                           sizeof(IMAGE_KEYWORD) * inimg.md->NBkw);
                }
                else
                {
                    memcpy(imgbuff1.im->kw,
                           inimg.im->kw,
                           sizeof(IMAGE_KEYWORD) * inimg.md->NBkw);
                }*/



                {
                    static pthread_t                  thread_savefits;
                    static int                        iret_savefits;


                    // Fill up thread message
                    //
                    /* strcpy(tmsg->fname, FITSffilename);
                     strcpy(tmsg->fnameascii, ASCIITIMEffilename);
                     tmsg->saveascii = 1;
                     tmsg->cubesize = (*frameindex);

                     if((*frameindex) != (*cubesize))
                     {
                         tmsg->partial = 1;
                     }
                     else
                     {
                         tmsg->partial = 0;
                     }
                    */


                    // test
                    strcpy(tmsg->iname, "staticn_amehere"); //imgbuff0.md->name);
                    /*
                    if(buffindex == 0 )
                    {
                        strcpy(tmsg->iname, imgbuff0.md->name);
                        tmsg->arrayindex  = array_cnt0;
                        tmsg->arraycnt0   = array_cnt0;
                        tmsg->arraycnt1   = array_cnt1;
                        tmsg->arraytime   = array_time;
                        tmsg->arrayaqtime = array_aqtime;
                    }
                    else
                    {
                        strcpy(tmsg->iname, imgbuff1.md->name);
                        tmsg->arrayindex  = &array_cnt0[(*cubesize)];
                        tmsg->arraycnt0   = &array_cnt0[(*cubesize)];
                        tmsg->arraycnt1   = &array_cnt1[(*cubesize)];
                        tmsg->arraytime   = &array_time[(*cubesize)];
                        tmsg->arrayaqtime = &array_aqtime[(*cubesize)];
                    }

                    WRITE_FILENAME(tmsg->fname_auxFITSheader,
                                   "%s/%s.aux.fits",
                                   data.shmdir,
                                   streamname);
                    */

                    /*     if ( (*compressON) == 0 )
                         {
                             strcpy(tmsg->compress_string, "");
                         }
                         else
                         {
                             strcpy(tmsg->compress_string, "[compress R 1,1,10000]");
                         }
                    */


                    // Wait for save thread to complete to launch next one
                    if(thread_initialized == 1)
                    {
                        long cnt0start = inimg.md->cnt0;

                        if(pthread_tryjoin_np(thread_savefits, NULL) == EBUSY)
                        {
                            if(VERBOSE > 0)
                            {
                                printf(
                                    "%5d  PREVIOUS SAVE THREAD "
                                    "NOT TERMINATED -> "
                                    "waiting\n",
                                    __LINE__);
                            }
                            pthread_join(thread_savefits, NULL);
                            if(VERBOSE > 0)
                            {
                                printf(
                                    "%5d  PREVIOUS SAVE THREAD "
                                    "NOW COMPLETED -> "
                                    "continuing\n",
                                    __LINE__);
                            }
                        }
                        else
                        {
                            if(VERBOSE > 0)
                            {
                                printf(
                                    "%5d  PREVIOUS SAVE THREAD "
                                    "ALREADY COMPLETED -> OK\n",
                                    __LINE__);
                            }
                        }
                        //  (*savetime) = tmsg->timespan;
                        printf("\n ************** MISSED  %ld frames\n", inimg.md->cnt0 - cnt0start);
                    }



                    // start thread
                    //
                    printf(">>>>>>>> [%5d] tmsg->iname  = \"%s\"\n", __LINE__, tmsg->iname);
                    iret_savefits = pthread_create(&thread_savefits,
                                                   NULL,
                                                   save_telemetry_fits_function,
                                                   tmsg);

                    thread_initialized = 1;
                    /*  if(iret_savefits)
                      {
                          fprintf(stderr,
                                  "Error - pthread_create() return code: %d\n",
                                  iret_savefits);
                          exit(EXIT_FAILURE);
                      }
                    */

                }


                SaveCube = 0;
            }



            // report buffer is ready
            //
            /*            if(buffindex == 0 )
                        {
                            processinfo_update_output_stream(processinfo, imgbuff0.ID);
                        }
                        else
                        {
                            processinfo_update_output_stream(processinfo, imgbuff1.ID);
                        }
            */

            // increment counters
            //
            (*frameindex) = 0;
            (*filecnt) ++;

            buffindex ++;
            if(buffindex > 1)
            {
                buffindex = 0;
            }

            if((lastcube == 1) || ((*lastcubeON) == 1))
            {
                (*saveON) = 0;
                data.fpsptr->parray[fpi_saveON].fpflag &= ~FPFLAG_ONOFF;

                (*lastcubeON) = 0;
                data.fpsptr->parray[fpi_lastcubeON].fpflag &= ~FPFLAG_ONOFF;
            }
        }


        saveON_last = (*saveON);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    //free(array_time);
    //free(array_aqtime);
    //free(array_cnt0);
    //free(array_cnt1);

    free(tmsg);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_MEMORY__logshmim()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

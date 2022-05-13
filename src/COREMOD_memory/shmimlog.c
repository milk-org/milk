#define _GNU_SOURCE

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"
#include "shmimlog_types.h"

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Local variables pointers
static char *instreamname;
static char *logdir;
static long *logcubesize;

// List of arguments to function
static CLICMDARGDEF farg[] = {{CLIARG_IMG,
                               ".in_sname",
                               "input stream name",
                               "im1",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &instreamname,
                               NULL},
                              {CLIARG_LONG,
                               ".cubesize",
                               "cube size",
                               "10000",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &logcubesize,
                               NULL},
                              {CLIARG_STR,
                               ".logdir",
                               "log directory",
                               "/media/data",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &logdir,
                               NULL}};

// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata = {
    "shmimlog", "log shared memory stream", CLICMD_FIELDS_DEFAULTS};

// Forward declarations

static errno_t __attribute__((hot)) shmimlog2D(const char *IDname,
                                               uint32_t    zsize,
                                               const char *logdir,
                                               const char *IDlogdata_name);

// adding INSERT_STD_PROCINFO statements enable processinfo support
static errno_t compute_function()
{
    //printf("Running comp func %s %s %ld\n", instreamname, logdir, *logcubesize);

    shmimlog2D(instreamname, *logcubesize, logdir, "");

    return RETURN_SUCCESS;
}

INSERT_STD_CLIfunction

    // Register function in CLI
    errno_t
    CLIADDCMD_COREMOD_memory__shmimlog()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

/** @brief creates logshimconf shared memory and loads it
 *
 */
static LOGSHIM_CONF *shmimlog_create_SHMconf(const char *logshimname)
{
    int           SM_fd;
    size_t        sharedsize = 0; // shared memory size in bytes
    char          SM_fname[STRINGMAXLEN_FILENAME];
    int           result;
    LOGSHIM_CONF *map;

    sharedsize = sizeof(LOGSHIM_CONF);

    WRITE_FILENAME(SM_fname, "%s/%s.logshimconf.shm", data.shmdir, logshimname);

    umask(0);
    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0666);
    if (SM_fd == -1)
    {
        printf("File \"%s\"\n", SM_fname);
        fflush(stdout);
        perror("Error opening file for writing");
        exit(0);
    }

    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if (result == -1)
    {
        close(SM_fd);
        PRINT_ERROR("Error calling lseek() to 'stretch' the file");
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1)
    {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    map = (LOGSHIM_CONF *)
        mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (map == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }

    map[0].on       = 0;
    map[0].cnt      = 0;
    map[0].filecnt  = 0;
    map[0].interval = 1;
    map[0].logexit  = 0;
    strcpy(map[0].fname, SM_fname);

    return map;
}

/** @brief Logs a shared memory stream onto disk
 *
 * uses semlog semaphore
 *
 * uses data cube buffer to store frames
 * if an image name logdata exists (should ideally be in shared mem), then this will be included in the timing txt file
 */
static errno_t __attribute__((hot)) shmimlog2D(const char *IDname,
                                               uint32_t    zsize,
                                               const char *logdir,
                                               const char *IDlogdata_name)
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

    time_t t;
    //    struct tm      *uttime;
    struct tm      *uttimeStart;
    struct timespec ts;
    struct timespec timenow;
    struct timespec timenowStart;
    //    long            kw;
    int     ret;
    imageID IDlogdata;

    char *ptr0_0; // source image data
    char *ptr1_0; // destination image data
    char *ptr0;   // source image data, after offset
    char *ptr1;   // destination image data, after offset

    long framesize; // in bytes

    //    FILE *fp;
    char fnameascii[STRINGMAXLEN_FULLFILENAME];

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

    char logb0name[STRINGMAXLEN_STREAMNAME];
    char logb1name[STRINGMAXLEN_STREAMNAME];

    int is3Dcube = 0; // this is a rolling buffer
        //    int exitflag = 0; // toggles to 1 when loop must exit

    LOGSHIM_CONF *logshimconf;

    // recording time for each frame
    double *array_time;
    double *array_time_cp;

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
    cntwaitlim = (long) (WaitSec * 1000000 / waitdelayus);

    schedpar.sched_priority = RT_priority;
    if (seteuid(data.euid) != 0) //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if (seteuid(data.ruid) != 0)   //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    IDlogdata = image_ID(IDlogdata_name);
    if (IDlogdata != -1)
    {
        if (data.image[IDlogdata].md[0].datatype != _DATATYPE_FLOAT)
        {
            IDlogdata = -1;
        }
    }
    printf("log data name = %s\n", IDlogdata_name);

    logshimconf             = shmimlog_create_SHMconf(IDname);
    logshimconf[0].on       = 1;
    logshimconf[0].cnt      = 0;
    logshimconf[0].filecnt  = 0;
    logshimconf[0].logexit  = 0;
    logshimconf[0].interval = 1;

    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    read_sharedmem_image(IDname);
    ID       = image_ID(IDname);
    datatype = data.image[ID].md[0].datatype;
    xsize    = data.image[ID].md[0].size[0];
    ysize    = data.image[ID].md[0].size[1];

    if (data.image[ID].md[0].naxis == 3)
    {
        is3Dcube = 1;
    }

    /** create the 2 buffers */

    imsizearray[0] = xsize;
    imsizearray[1] = ysize;
    imsizearray[2] = zsize;

    WRITE_IMAGENAME(logb0name, "%s_logbuff0", IDname);
    WRITE_IMAGENAME(logb1name, "%s_logbuff1", IDname);

    create_image_ID(logb0name, 3, imsizearray, datatype, 1, 1, 0, &IDb0);
    create_image_ID(logb1name, 3, imsizearray, datatype, 1, 1, 0, &IDb1);
    COREMOD_MEMORY_image_set_semflush(logb0name, -1);
    COREMOD_MEMORY_image_set_semflush(logb1name, -1);

    array_time = (double *) malloc(sizeof(double) * zsize);
    array_cnt0 = (uint64_t *) malloc(sizeof(uint64_t) * zsize);
    array_cnt1 = (uint64_t *) malloc(sizeof(uint64_t) * zsize);

    array_time_cp = (double *) malloc(sizeof(double) * zsize);
    array_cnt0_cp = (uint64_t *) malloc(sizeof(uint64_t) * zsize);
    array_cnt1_cp = (uint64_t *) malloc(sizeof(uint64_t) * zsize);

    IDb = IDb0;

    switch (datatype)
    {
    case _DATATYPE_FLOAT:
        framesize = SIZEOF_DATATYPE_FLOAT * xsize * ysize;
        break;

    case _DATATYPE_INT8:
        framesize = SIZEOF_DATATYPE_INT8 * xsize * ysize;
        break;

    case _DATATYPE_UINT8:
        framesize = SIZEOF_DATATYPE_UINT8 * xsize * ysize;
        break;

    case _DATATYPE_INT16:
        framesize = SIZEOF_DATATYPE_INT16 * xsize * ysize;
        break;

    case _DATATYPE_UINT16:
        framesize = SIZEOF_DATATYPE_UINT16 * xsize * ysize;
        break;

    case _DATATYPE_INT32:
        framesize = SIZEOF_DATATYPE_INT32 * xsize * ysize;
        break;

    case _DATATYPE_UINT32:
        framesize = SIZEOF_DATATYPE_UINT32 * xsize * ysize;
        break;

    case _DATATYPE_INT64:
        framesize = SIZEOF_DATATYPE_INT64 * xsize * ysize;
        break;

    case _DATATYPE_UINT64:
        framesize = SIZEOF_DATATYPE_UINT64 * xsize * ysize;
        break;

    case _DATATYPE_DOUBLE:
        framesize = SIZEOF_DATATYPE_DOUBLE * xsize * ysize;
        break;

    default:
        printf("ERROR: WRONG DATA TYPE\n");
        exit(0);
        break;
    }

    ptr0_0 = (char *) data.image[ID].array.raw;
    ptr1_0 = (char *) data.image[IDb].array.raw;

    cnt = data.image[ID].md[0].cnt0 - 1;

    buffer = 0;
    index  = 0;

    printf("logdata ID = %ld\n", IDlogdata);
    list_image_ID();

    // exitflag = 0;

    // initialize circuler buffer vars
    logshimconf->CBsize  = data.image[ID].md->CBsize;
    logshimconf->CBindex = data.image[ID].md->CBindex;
    logshimconf->CBcycle = data.image[ID].md->CBcycle;

    // using semlog ?
    use_semlog = 0;
    if (data.image[ID].semlog != NULL)
    {
        use_semlog = 1;
        sem_getvalue(data.image[ID].semlog, &semval);

        // bring semaphore value to 1 to only save 1 frame
        while (semval > 1)
        {
            sem_wait(data.image[ID].semlog);
            sem_getvalue(data.image[ID].semlog, &semval);
        }
        if (semval == 0)
        {
            sem_post(data.image[ID].semlog);
        }
    }

    DEBUG_TRACEPOINT(" ");

    int SkipWait = 0; // wait for update
    while ((logshimconf[0].filecnt != NBfiles) && (logshimconf[0].logexit == 0))
    {
        int timeout; // 1 if timeout has occurred

        if (logshimconf[0].filecnt == 3)
        {
            //test
            logshimconf[0].on = 0;
        }

        cntwait = 0;
        noframe = 0;
        wOK     = 1;

        if (VERBOSE > 1)
        {
            printf("%5d  Entering wait loop   index = %ld %d\n",
                   __LINE__,
                   index,
                   noframe);
        }

        timeout = 0;

        // Keep CPU load light when not logging
        if (logshimconf[0].on == 0)
        {
            float           tdelay_us = 100.0; // 10 kHz
            struct timespec tdel;
            tdel.tv_sec  = 0;
            tdel.tv_nsec = (long) (1000.0 * tdelay_us); // * rand() / RAND_MAX);
            //printf("Waiting 0.%09ld sec\n", tdel.tv_nsec);
            nanosleep(&tdel, NULL);
        }

        if (SkipWait == 1)
        {
            if (VERBOSE > 0)
            {
                if (logshimconf[0].on == 1)
                {
                    printf(">>>>>>> SKIPPING WAIT >>>>>>>>\n");
                }
            }
        }
        else
        {
            if (likely(use_semlog == 1))
            {
                if (VERBOSE > 1)
                {
                    printf("%5d  Waiting for semaphore\n", __LINE__);
                }

                if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
                {
                    perror("clock_gettime");
                    exit(EXIT_FAILURE);
                }
                ts.tv_sec += WaitSec;

                ret = sem_timedwait(data.image[ID].semlog, &ts);
                if (ret == -1)
                {
                    if (errno == ETIMEDOUT)
                    {
                        printf(
                            "%5d  sem_timedwait() timed "
                            "out (%d sec) -[index %ld]\n",
                            __LINE__,
                            WaitSec,
                            index);
                        if (VERBOSE > 0)
                        {
                            printf(
                                "%5d  sem time elapsed "
                                "-> Save current cube "
                                "[index %ld]\n",
                                __LINE__,
                                index);
                        }

                        strcpy(tmsg->iname, iname);
                        strcpy(tmsg->fname, fname);
                        tmsg->partial  = 1; // partial cube
                        tmsg->cubesize = index;

                        memcpy(array_time_cp,
                               array_time,
                               sizeof(double) * index);
                        memcpy(array_cnt0_cp,
                               array_cnt0,
                               sizeof(uint64_t) * index);
                        memcpy(array_cnt1_cp,
                               array_cnt1,
                               sizeof(uint64_t) * index);

                        tmsg->arrayindex = array_cnt0_cp;
                        tmsg->arraycnt0  = array_cnt0_cp;
                        tmsg->arraycnt1  = array_cnt1_cp;
                        tmsg->arraytime  = array_time_cp;

                        timeout = 1;
                    }
                    if (errno == EINTR)
                    {
                        printf(
                            "%5d  sem_timedwait [index "
                            "%ld]: The call was "
                            "interrupted by a signal "
                            "handler\n",
                            __LINE__,
                            index);
                    }

                    if (errno == EINVAL)
                    {
                        printf(
                            "%5d  sem_timedwait [index "
                            "%ld]: Not a valid semaphore\n",
                            __LINE__,
                            index);
                        printf(
                            "               The value of "
                            "abs_timeout.tv_nsecs is less "
                            "than 0, or greater than or "
                            "equal to 1000 million\n");
                    }

                    if (errno == EAGAIN)
                    {
                        printf(
                            "%5d  sem_timedwait [index "
                            "%ld]: The operation could not "
                            "be performed without blocking "
                            "(i.e., the semaphore "
                            "currently has the value "
                            "zero)\n",
                            __LINE__,
                            index);
                    }

                    wOK = 0;
                    if (index == 0)
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
                if (VERBOSE > 1)
                {
                    printf(
                        "%5d  Not using semaphore, watching "
                        "counter\n",
                        __LINE__);
                }

                while (((cnt == data.image[ID].md[0].cnt0) ||
                        (logshimconf[0].on == 0)) &&
                       (wOK == 1))
                {
                    if (VERBOSE > 1)
                    {
                        printf("%5d  waiting time step\n", __LINE__);
                    }

                    usleep(waitdelayus);
                    cntwait++;

                    if (VERBOSE > 1)
                    {
                        printf("%5d  cntwait = %lld\n", __LINE__, cntwait);
                        fflush(stdout);
                    }

                    if (cntwait > cntwaitlim) // save current cube
                    {
                        if (VERBOSE > 0)
                        {
                            printf(
                                "%5d  cnt time elapsed "
                                "-> Save current "
                                "cube\n",
                                __LINE__);
                        }

                        strcpy(tmsg->iname, iname);
                        strcpy(tmsg->fname, fname);
                        tmsg->partial  = 1; // partial cube
                        tmsg->cubesize = index;

                        memcpy(array_time_cp,
                               array_time,
                               sizeof(double) * index);
                        memcpy(array_cnt0_cp,
                               array_cnt0,
                               sizeof(uint64_t) * index);
                        memcpy(array_cnt1_cp,
                               array_cnt1,
                               sizeof(uint64_t) * index);

                        tmsg->arrayindex = array_cnt0_cp;
                        tmsg->arraycnt0  = array_cnt0_cp;
                        tmsg->arraycnt1  = array_cnt1_cp;
                        tmsg->arraytime  = array_time_cp;

                        wOK = 0;
                        if (index == 0)
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
        }

        DEBUG_TRACEPOINT(" ");

        if (index == 0)
        {
            if (VERBOSE > 0)
            {
                printf("%5d  Setting cube start time [index %ld]\n",
                       __LINE__,
                       index);
            }

            /// measure time
            t           = time(NULL);
            uttimeStart = gmtime(&t);
            clock_gettime(CLOCK_REALTIME, &timenowStart);

            //     sprintf(fname,"%s/%s_%02d:%02d:%02ld.%09ld.fits", logdir, IDname, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
            //            sprintf(fnameascii,"%s/%s_%02d:%02d:%02ld.%09ld.txt", logdir, IDname, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
        }

        if (VERBOSE > 1)
        {
            printf("%5d  logshimconf[0].on = %d\n",
                   __LINE__,
                   logshimconf[0].on);
        }

        DEBUG_TRACEPOINT(" ");

        if (likely(logshimconf[0].on == 1))
        {
            if (likely(wOK == 1)) // normal step: a frame has arrived
            {
                if (VERBOSE > 1)
                {
                    printf("%5d  Frame has arrived [index %ld]\n",
                           __LINE__,
                           index);
                }

                DEBUG_TRACEPOINT(" ");

                long NBgrab    = 0;
                long grabSpan0 = 0;
                long grabSpan1 = 0;

                long index0start, index0end;
                long index1start, index1end;

                long grabStart0 = 0;
                long grabEnd0   = 0;
                long grabStart1 = 0;
                long grabEnd1   = 0;

                if (data.image[ID].md->CBsize > 0)
                {
                    if (VERBOSE > 0)
                    {
                        printf("\n");

                        printf(
                            "    LAST SAVED          [%8ld "
                            "%3ld]\n",
                            logshimconf->CBcycle,
                            (long) logshimconf->CBindex);
                    }
                    long currentCBcycle = data.image[ID].md->CBcycle;
                    long currentCBindex = data.image[ID].md->CBindex;

                    if (VERBOSE > 0)
                    {
                        printf(
                            "    LAST WRITTEN IN CB  [%8ld "
                            "%3ld]\n",
                            (long) currentCBcycle,
                            (long) currentCBindex);
                    }

                    long CBcyclediff =
                        (long) currentCBcycle - (long) logshimconf->CBcycle;

                    long CBindexdiff =
                        (long) currentCBindex - (long) logshimconf->CBindex;

                    // number of frames behind
                    long NBframesbehind =
                        (long) (CBcyclediff * (long) data.image[ID].md->CBsize +
                                CBindexdiff);

                    if (VERBOSE > 0)
                    {
                        printf(
                            "    DIFF                [%8ld "
                            "%3ld]    %8ld frames\n",
                            CBcyclediff,
                            CBindexdiff,
                            NBframesbehind);
                    }

                    long NBgrabmax = 20;
                    NBgrab         = NBframesbehind;
                    if (NBgrab > NBgrabmax)
                    {
                        NBgrab = NBgrabmax;
                    }
                    // avoid crossing big cube boundary
                    long grabmax_logbuff = zsize - index;

                    if (VERBOSE > 0)
                    {
                        printf(
                            "    -> GRABBING %4ld frames "
                            "(max %6ld) -> ",
                            NBgrab,
                            grabmax_logbuff);
                    }
                    if (NBgrab > grabmax_logbuff)
                    {
                        NBgrab = grabmax_logbuff;
                    }
                    if (VERBOSE > 0)
                    {
                        printf("%6ld\n", NBgrab);
                    }

                    // Grab range
                    long grabStart = currentCBindex - NBgrab + 1;
                    long grabEnd   = currentCBindex;

                    grabStart0 = 0;
                    grabEnd0   = 0;
                    grabSpan0  = 0;

                    grabStart1 = 0;
                    grabEnd1   = 0;
                    grabSpan1  = 0;

                    if (grabStart < 0)
                    {
                        grabStart0 = grabStart + data.image[ID].md->CBsize;
                        grabEnd0   = data.image[ID].md->CBsize - 1;
                        grabSpan0  = grabEnd0 - grabStart0 + 1;

                        grabStart1 = 0;
                        grabEnd1   = grabEnd;
                        grabSpan1  = grabEnd1 - grabStart1 + 1;
                    }
                    else
                    {
                        grabStart0 = grabStart;
                        grabEnd0   = grabEnd;
                        grabSpan0  = grabEnd0 - grabStart0 + 1;

                        grabStart1 = 0;
                        grabEnd1   = 0;
                        grabSpan1  = 0;
                    }

                    index0start = index;
                    index0end   = index0start + (grabSpan0 - 1);
                    index1start = index0end + 1;
                    index1end   = index1start + (grabSpan1 - 1);

                    if (VERBOSE > 0)
                    {
                        printf(
                            "    CB BUFFER RANGE     %3ld "
                            "-> %3ld\n",
                            grabStart,
                            grabEnd);

                        printf(
                            "     GRAB RANGE %3ld %3ld  "
                            "(%3ld) ->  %6ld %6ld\n",
                            grabStart0,
                            grabEnd0,
                            grabSpan0,
                            index0start,
                            index0end);
                        printf(
                            "     GRAB RANGE %3ld %3ld  "
                            "(%3ld) ->  %6ld %6ld\n",
                            grabStart1,
                            grabEnd1,
                            grabSpan1,
                            index1start,
                            index1end);
                    }

                    // update last saved
                    logshimconf->CBcycle = currentCBcycle;
                    logshimconf->CBindex = currentCBindex;

                    // total number of frames grabbed
                    NBgrab = grabSpan0 + grabSpan1;

                    if (NBframesbehind > 1)
                    {
                        SkipWait = 1; // don't wait, proceed to grab next frame
                    }
                    else
                    {
                        SkipWait = 0;
                    }
                }
                else
                {
                    grabSpan0 = 1;
                    grabSpan1 = 0;
                    NBgrab    = 1;
                }

                /// measure time
                //   t = time(NULL);
                //   uttime = gmtime(&t);

                clock_gettime(CLOCK_REALTIME, &timenow);

                DEBUG_TRACEPOINT(" ");

                // memcpy
                //
                if (data.image[ID].md->CBsize > 0) // circ buffer mode
                {

                    if (grabSpan0 > 0)
                    {
                        // source
                        ptr0 = (char *) data.image[ID].CBimdata;
                        ptr0 += data.image[ID].md->imdatamemsize * grabStart0;

                        // destination
                        ptr1 = ptr1_0 + framesize * index0start;

                        memcpy((void *) ptr1,
                               (void *) ptr0,
                               framesize * grabSpan0);
                    }

                    if (grabSpan1 > 0)
                    {
                        // source
                        ptr0 = (char *) data.image[ID].CBimdata;
                        ptr0 += data.image[ID].md->imdatamemsize * grabStart1;

                        // destination
                        ptr1 = ptr1_0 + framesize * index1start;

                        memcpy((void *) ptr1,
                               (void *) ptr0,
                               framesize * grabSpan1);
                    }
                    index += NBgrab;
                }
                else
                {
                    // no circular buffer

                    // source
                    if (is3Dcube == 1)
                    {
                        ptr0 = ptr0_0 + framesize * data.image[ID].md[0].cnt1;
                    }
                    else
                    {
                        ptr0 = ptr0_0;
                    }

                    // destination
                    ptr1 = ptr1_0 + framesize * index;

                    if (NBgrab > 0)
                    {
                        memcpy((void *) ptr1,
                               (void *) ptr0,
                               framesize * NBgrab);
                        array_cnt0[index] = data.image[ID].md[0].cnt0;
                        array_cnt1[index] = data.image[ID].md[0].cnt1;
                        array_time[index] =
                            timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;
                    }
                    index += NBgrab;
                }

                DEBUG_TRACEPOINT(" ");
            }
        }
        else
        {
            // save partial if possible
            //if(index>0)
            wOK = 0;
        }

        if (VERBOSE > 1)
        {
            printf("%5d  index = %ld  wOK = %d\n", __LINE__, index, wOK);
        }

        // SAVE CUBE TO DISK
        /// cases:
        /// index>zsize-1  buffer full
        /// timeout==1 && index>0  : partial
        if ((index > zsize - 1) || ((timeout == 1) && (index > 0)))
        {
            long NBframemissing;

            /// save image
            if (VERBOSE > 0)
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
            if (buffer == 0)
            {
                IDb = IDb0;
            }
            else
            {
                IDb = IDb1;
            }

            if (VERBOSE > 0)
            {
                printf("%5d  Building file name: ascii\n", __LINE__);
                fflush(stdout);
            }

            sprintf(fnameascii,
                    "%s/%s.%04d%02d%02dT%02d%02d%02ld.%09ldZ.txt",
                    logdir,
                    IDname,
                    1900 + uttimeStart->tm_year,
                    1 + uttimeStart->tm_mon,
                    uttimeStart->tm_mday,
                    uttimeStart->tm_hour,
                    uttimeStart->tm_min,
                    timenowStart.tv_sec % 60,
                    timenowStart.tv_nsec);

            if (VERBOSE > 0)
            {
                printf("%5d  Building file name: fits\n", __LINE__);
                fflush(stdout);
            }
            sprintf(fname,
                    "%s/%s.%04d%02d%02dT%02d%02d%02ld.%09ldZ.fits",
                    logdir,
                    IDname,
                    1900 + uttimeStart->tm_year,
                    1 + uttimeStart->tm_mon,
                    uttimeStart->tm_mday,
                    uttimeStart->tm_hour,
                    uttimeStart->tm_min,
                    timenowStart.tv_sec % 60,
                    timenowStart.tv_nsec);

            strcpy(tmsg->iname, iname);
            strcpy(tmsg->fname, fname);
            strcpy(tmsg->fnameascii, fnameascii);
            tmsg->saveascii = 1;

            if (wOK == 1) // full cube
            {
                tmsg->partial = 0; // full cube
                if (VERBOSE > 0)
                {
                    printf("%5d  SAVING FULL CUBE\n", __LINE__);
                    fflush(stdout);
                }
            }
            else // partial cube
            {
                tmsg->partial = 1; // partial cube
                if (VERBOSE > 0)
                {
                    printf("%5d  SAVING PARTIAL CUBE\n", __LINE__);
                    fflush(stdout);
                }
            }

            //  fclose(fp);

            // Wait for save thread to complete to launch next one
            if (tOK == 1)
            {
                if (pthread_tryjoin_np(thread_savefits, NULL) == EBUSY)
                {
                    VERBOSE = 1;
                    if (VERBOSE > 0)
                    {
                        printf(
                            "%5d  PREVIOUS SAVE THREAD NOT "
                            "TERMINATED -> waiting\n",
                            __LINE__);
                    }

                    struct timespec t0;
                    struct timespec t1;
                    struct timespec tdiff;
                    clock_gettime(CLOCK_REALTIME, &t0);
                    pthread_join(thread_savefits, NULL);
                    clock_gettime(CLOCK_REALTIME, &t1);
                    tdiff = timespec_diff(t0, t1);
                    printf("WAITED %ld.%09ld sec\n",
                           tdiff.tv_sec,
                           tdiff.tv_nsec);

                    if (VERBOSE > 0)
                    {
                        printf(
                            "%5d  PREVIOUS SAVE THREAD NOW "
                            "COMPLETED -> continuing\n",
                            __LINE__);
                    }
                }
                else if (VERBOSE > 0)
                {
                    printf(
                        "%5d  PREVIOUS SAVE THREAD ALREADY "
                        "COMPLETED -> OK\n",
                        __LINE__);
                }
                VERBOSE = 0;
            }

            COREMOD_MEMORY_image_set_sempost_byID(IDb, -1);
            data.image[IDb].md[0].cnt0++;
            data.image[IDb].md[0].write = 0;

            tmsg->cubesize = index;
            strcpy(tmsg->iname, iname);
            memcpy(array_time_cp, array_time, sizeof(double) * index);
            memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t) * index);
            memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t) * index);

            NBframemissing =
                (array_cnt0[index - 1] - array_cnt0[0]) - (index - 1);

            printf("=== %6ld %6ld  %6ld   %6ld ====\n",
                   array_cnt0[0],
                   array_cnt0[index - 1],
                   index,
                   array_cnt0[index - 1] - array_cnt0[0]);
            printf(
                "===== CUBE %8lld   Number of missed frames = %8ld  / "
                "%ld  / %8ld ====\n",
                logshimconf[0].filecnt,
                NBframemissing,
                index,
                (long) zsize);

            if (VERBOSE > 0)
            {
                printf("%5d  Starting image save thread\n", __LINE__);
                fflush(stdout);
            }

            tmsg->arrayindex = array_cnt0_cp;
            tmsg->arraycnt0  = array_cnt0_cp;
            tmsg->arraycnt1  = array_cnt1_cp;
            tmsg->arraytime  = array_time_cp;
            WRITE_FULLFILENAME(tmsg->fname_auxFITSheader,
                               "%s/%s.auxFITSheader.shm",
                               data.shmdir,
                               IDname);
            iret_savefits = pthread_create(&thread_savefits,
                                           NULL,
                                           save_fits_function,
                                           tmsg);

            logshimconf[0].cnt++;

            tOK = 1;
            if (iret_savefits)
            {
                fprintf(stderr,
                        "Error - pthread_create() return code: %d\n",
                        iret_savefits);
                exit(EXIT_FAILURE);
            }

            index = 0;
            buffer++;
            if (buffer == 2)
            {
                buffer = 0;
            }
            //            printf("[%ld -> %d]", cnt, buffer);
            //           fflush(stdout);
            if (buffer == 0)
            {
                IDb = IDb0;
            }
            else
            {
                IDb = IDb1;
            }

            switch (datatype)
            {
            case _DATATYPE_FLOAT:
                ptr1_0 = (char *) data.image[IDb].array.F;
                break;

            case _DATATYPE_INT8:
                ptr1_0 = (char *) data.image[IDb].array.SI8;
                break;

            case _DATATYPE_UINT8:
                ptr1_0 = (char *) data.image[IDb].array.UI8;
                break;

            case _DATATYPE_INT16:
                ptr1_0 = (char *) data.image[IDb].array.SI16;
                break;

            case _DATATYPE_UINT16:
                ptr1_0 = (char *) data.image[IDb].array.UI16;
                break;

            case _DATATYPE_INT32:
                ptr1_0 = (char *) data.image[IDb].array.SI32;
                break;

            case _DATATYPE_UINT32:
                ptr1_0 = (char *) data.image[IDb].array.UI32;
                break;

            case _DATATYPE_INT64:
                ptr1_0 = (char *) data.image[IDb].array.SI64;
                break;

            case _DATATYPE_UINT64:
                ptr1_0 = (char *) data.image[IDb].array.UI64;
                break;

            case _DATATYPE_DOUBLE:
                ptr1_0 = (char *) data.image[IDb].array.D;
                break;
            }

            data.image[IDb].md[0].write = 1;
            logshimconf[0].filecnt++;
        }
        cnt = data.image[ID].md[0].cnt0;
    }

    free(imsizearray);
    free(tmsg);

    free(array_time);
    free(array_cnt0);
    free(array_cnt1);

    free(array_time_cp);
    free(array_cnt0_cp);
    free(array_cnt1_cp);

    return RETURN_SUCCESS;
}

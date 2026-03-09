#define _GNU_SOURCE
#include "ImageStreamIO/ImageStruct.h"
#define _GNU_SOURCE
/**
 * @file    logshmim.c
 * @brief   Save telemetry stream data
 *
 * Uses FPS V2 framework.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "timeutils.h"

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

static long tret = 0;


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamFITSlog",
    .cmdkey      = "streamFITSlog",
    .description =
        "log stream to FITS file(s)"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *streamname    = NULL;
static int32_t  *saveON        = NULL;
static int32_t  *lastcubeON    = NULL;
static int32_t  *nextcube      = NULL;
static uint32_t *cubesize      = NULL;
static char     *savedirname   = NULL;
static uint64_t *frameindex    = NULL;
static uint64_t *framecnt      = NULL;
static uint64_t *maxframecnt   = NULL;
static uint64_t *filecnt       = NULL;
static uint64_t *maxfilecnt    = NULL;
static char     *outfname      = NULL;
static int32_t  *compressON    = NULL;
static float    *savetime      = NULL;
static uint32_t *writerRTprio  = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".sname", &streamname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "stream image") \
    X(".saveON", &saveON, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "toggle save on/off") \
    X(".lastcubeON", &lastcubeON, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "toggle last cube on/off") \
    X(".nextcube", &nextcube, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "force jump to next cube") \
    X(".cubesize", &cubesize, \
      FPTYPE_UINT32, 1, \
      (FPFLAG_DEFAULT_INPUT), \
      "cube size, nb frame per cube") \
    X(".dirname", &savedirname, \
      FPTYPE_STRING, 1, \
      (FPFLAG_DEFAULT_INPUT), \
      "log directory") \
    X(".frameindex", &frameindex, \
      FPTYPE_UINT64, 0, \
      FPFLAG_DEFAULT_OUTPUT, \
      "frame index within cube") \
    X(".framecnt", &framecnt, \
      FPTYPE_UINT64, 0, \
      FPFLAG_DEFAULT_OUTPUT, \
      "frame counter since started") \
    X(".maxframecnt", &maxframecnt, \
      FPTYPE_UINT64, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "max frame count") \
    X(".filecnt", &filecnt, \
      FPTYPE_UINT64, 0, \
      FPFLAG_DEFAULT_OUTPUT, \
      "file counter") \
    X(".maxfilecnt", &maxfilecnt, \
      FPTYPE_UINT64, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "max file counter") \
    X(".outfname", &outfname, \
      FPTYPE_STRING, 0, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output file name") \
    X(".compress", &compressON, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "toggle compression on/off") \
    X(".savetime", &savetime, \
      FPTYPE_FLOAT32, 0, \
      FPFLAG_DEFAULT_OUTPUT, \
      "time taken to save") \
    X(".writerRTprio", &writerRTprio, \
      FPTYPE_UINT32, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "writer real-time priority")


/* ================================================================
 * 4.  CUSTOM CONF SETUP / CHECK
 * ============================================================= */

static errno_t customCONFsetup()
{
    if(dcfpsptr != NULL)
    {
        long fpi;

        fpi = functionparameter_GetParamIndex(
            dcfpsptr, ".saveON");
        if(fpi >= 0)
            dcfpsptr->parray[fpi].fpflag
                |= FPFLAG_WRITERUN;

        fpi = functionparameter_GetParamIndex(
            dcfpsptr, ".lastcubeON");
        if(fpi >= 0)
            dcfpsptr->parray[fpi].fpflag
                |= FPFLAG_WRITERUN;

        fpi = functionparameter_GetParamIndex(
            dcfpsptr, ".nextcube");
        if(fpi >= 0)
            dcfpsptr->parray[fpi].fpflag
                |= FPFLAG_WRITERUN;

        fpi = functionparameter_GetParamIndex(
            dcfpsptr, ".maxfilecnt");
        if(fpi >= 0)
            dcfpsptr->parray[fpi].fpflag
                |= FPFLAG_WRITERUN;

        fpi = functionparameter_GetParamIndex(
            dcfpsptr, ".maxframecnt");
        if(fpi >= 0)
            dcfpsptr->parray[fpi].fpflag
                |= FPFLAG_WRITERUN;

        fpi = functionparameter_GetParamIndex(
            dcfpsptr, ".writerRTprio");
        if(fpi >= 0)
            dcfpsptr->parray[fpi].fpflag
                |= FPFLAG_WRITERUN;
    }

    return RETURN_SUCCESS;
}

static errno_t customCONFcheck()
{
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 *  FITS SAVE THREAD
 * ============================================================= */

/**
 * ## Purpose
 *
 * Save telemetry stream data
 * Writes FITS file and timing file
 */
static void *save_telemetry_fits_function(
    void *ptr)
{
    STREAMSAVE_THREAD_MESSAGE *tmsg;
    tmsg = (STREAMSAVE_THREAD_MESSAGE *) ptr;

    struct timespec tstart;
    clock_gettime(CLOCK_MILK, &tstart);

    int RT_priority = tmsg->writerRTprio;
    struct sched_param schedpar;

    schedpar.sched_priority = RT_priority;
    if(seteuid(dceuid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar);
    if(seteuid(dcruid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }

    int NBcustomKW = 9;
    IMAGE_KEYWORD *imkwarray =
        (IMAGE_KEYWORD *)
        malloc(sizeof(IMAGE_KEYWORD)
               * NBcustomKW);

    strcpy(imkwarray->name, "UT");
    imkwarray->type = 'S';

    strcpy(imkwarray->value.valstr,
           timedouble_to_UTC_timeofdaystring(
               0.5 * tmsg->arraytime[0]
               + 0.5 * tmsg->arraytime[
                   tmsg->cubesize - 1]));
    strcpy(imkwarray->comment,
           "HH:MM:SS.SS typical UTC"
           " at exposure");

    strcpy(imkwarray[1].name, "UT-STR");
    imkwarray[1].type = 'S';
    strcpy(imkwarray[1].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               tmsg->arraytime[0]));
    strcpy(imkwarray[1].comment,
           "HH:MM:SS.SS UTC at exposure start");

    strcpy(imkwarray[2].name, "UT-END");
    imkwarray[2].type = 'S';
    strcpy(imkwarray[2].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               tmsg->arraytime[
                   tmsg->cubesize - 1]));
    strcpy(imkwarray[2].comment,
           "HH:MM:SS.SS UTC at exposure end");

    strcpy(imkwarray[3].name, "MJD");
    imkwarray[3].type = 'D';
    imkwarray[3].value.numf =
        (0.5 * tmsg->arraytime[0]
         + 0.5 * tmsg->arraytime[
             tmsg->cubesize - 1])
        / 86400.0 + 40587.0;
    strcpy(imkwarray[3].comment,
           "Modified Julian Day at exposure");

    strcpy(imkwarray[4].name, "MJD-STR");
    imkwarray[4].type = 'D';
    imkwarray[4].value.numf =
        tmsg->arraytime[0]
        / 86400.0 + 40587.0;
    strcpy(imkwarray[4].comment,
           "Modified Julian Day at"
           " exposure start");

    strcpy(imkwarray[5].name, "MJD-END");
    imkwarray[5].type = 'D';
    imkwarray[5].value.numf =
        (tmsg->arraytime[tmsg->cubesize - 1]
         / 86400.0) + 40587.0;
    strcpy(imkwarray[5].comment,
           "Modified Julian Day at"
           " exposure end");

    sprintf(imkwarray[6].name, "%s",
            TZ_MILK_STR);
    imkwarray[6].type = 'S';
    strcpy(imkwarray[6].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               (0.5 * tmsg->arraytime[0]
                + 0.5 * tmsg->arraytime[
                    tmsg->cubesize - 1])
               + TZ_MILK_UTC_OFF));
    sprintf(imkwarray[6].comment,
            "HH:MM:SS.SS typical %s"
            " at exposure",
            TZ_MILK_STR);

    sprintf(imkwarray[7].name, "%s-STR",
            TZ_MILK_STR);
    imkwarray[7].type = 'S';
    strcpy(imkwarray[7].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               tmsg->arraytime[0]
               + TZ_MILK_UTC_OFF));
    sprintf(imkwarray[7].comment,
            "HH:MM:SS.SS typical %s"
            " at exposure start",
            TZ_MILK_STR);

    sprintf(imkwarray[8].name, "%s-END",
            TZ_MILK_STR);
    imkwarray[8].type = 'S';
    strcpy(imkwarray[8].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               tmsg->arraytime[
                   tmsg->cubesize - 1]
               + TZ_MILK_UTC_OFF));
    sprintf(imkwarray[8].comment,
            "HH:MM:SS.SS typical %s"
            " at exposure end",
            TZ_MILK_STR);

    printf(">>>>>>>> [%5d] tmsg->iname"
           "  = \"%s\"\n",
           __LINE__, tmsg->iname);

    saveFITS_opt_trunc(tmsg->iname,
                       tmsg->partial
                       ? tmsg->cubesize : -1,
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

        if((fp = fopen(tmsg->fnameascii, "w"))
            == NULL)
        {
            printf("ERROR: cannot create"
                   " file \"%s\"\n",
                   tmsg->fnameascii);
            exit(0);
        }

        fprintf(fp,
                "# Telemetry stream"
                " timing data \n");
        fprintf(fp,
                "# File written by"
                " function %s in file %s\n",
                __FUNCTION__, __FILE__);
        fprintf(fp, "# \n");
        fprintf(fp,
                "# col1 : datacube"
                " frame index\n");
        fprintf(fp,
                "# col2 : Main index\n");
        fprintf(fp,
                "# col3 : Time since cube"
                " origin (logging)\n");
        fprintf(fp,
                "# col4 : Absolute time"
                " (logging)\n");
        fprintf(fp,
                "# col5 : Absolute time"
                " (acquisition)\n");
        fprintf(fp,
                "# col6 : stream cnt0"
                " index\n");
        fprintf(fp,
                "# col7 : stream cnt1"
                " index\n");
        fprintf(fp, "# \n");

        double t0;
        t0 = tmsg->arraytime[0];
        for(long k = 0;
             k < tmsg->cubesize; k++)
        {
            fprintf(fp,
                    "%10ld  %10lu  %15.9lf"
                    "   %20.9lf  %17.6lf"
                    "   %10ld   %10ld\n",
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

    tret = image_ID(tmsg->iname,
                    dcimg,
                    dcnimg);

    struct timespec tend;
    clock_gettime(CLOCK_MILK, &tend);

    double timediff =
        1.0 * (tend.tv_sec - tstart.tv_sec)
        + 1.0e-9 * (tend.tv_nsec
                     - tstart.tv_nsec);
    tmsg->timespan = timediff;

    pthread_exit(&tret);
}


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    int VERBOSE = 2;

    STREAMSAVE_THREAD_MESSAGE *tmsg =
        (STREAMSAVE_THREAD_MESSAGE *)
        malloc(sizeof(
            STREAMSAVE_THREAD_MESSAGE));

    IMGID inimg =
        imgid_make_from_name(streamname);
    resolveIMGID(
        &inimg, ERRMODE_ABORT,
        dcimg, dcnimg);

    if(inimg.md->naxis == 3)
    {
        PRINT_ERROR(
            "streamFITSlog with 3D data"
            " is NOT supported");
    }

    uint32_t xsize = inimg.md->size[0];
    uint32_t ysize = inimg.md->size[1];
    if(inimg.md->naxis == 1)
    {
        ysize = 1;
    }
    uint32_t zsize = (*cubesize);

    uint8_t datatype = inimg.md->datatype;

    int typesize =
        ImageStreamIO_typesize(datatype);
    if(typesize == -1)
    {
        printf("ERROR: WRONG DATA TYPE\n");
        exit(0);
    }

    int buffindex = 0;

    IMGID imgbuff0;
    {
        char name[STRINGMAXLEN_STREAMNAME];
        WRITE_IMAGENAME(name,
                        "%s_logbuff0",
                        streamname);
        imgbuff0 =
            stream_connect_create_3D(
                name, xsize, ysize,
                zsize, datatype);
    }
    IMGID imgbuff1;
    {
        char name[STRINGMAXLEN_STREAMNAME];
        WRITE_IMAGENAME(name,
                        "%s_logbuff1",
                        streamname);
        imgbuff1 =
            stream_connect_create_3D(
                name, xsize, ysize,
                zsize, datatype);
    }

    list_image_ID();

    {
        printf("Cppying %d keywords\n",
               inimg.md->NBkw);
        if(inimg.md->NBkw > 0)
        {
            memcpy(imgbuff0.im->kw,
                   inimg.im->kw,
                   sizeof(IMAGE_KEYWORD)
                   * inimg.md->NBkw);
            memcpy(imgbuff1.im->kw,
                   inimg.im->kw,
                   sizeof(IMAGE_KEYWORD)
                   * inimg.md->NBkw);
        }
    }

    int aqtimekwi = -1;
    for(int kwi = 0;
         kwi < inimg.md->NBkw; kwi++)
    {
        if(strcmp(inimg.im->kw[kwi].name,
                 "_MAQTIME") == 0)
        {
            aqtimekwi = kwi;
        }
    }
    if(VERBOSE > 0)
    {
        printf("[%5d] aqtimekwi = %d\n",
               __LINE__, aqtimekwi);
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    if(CLIcmddata.cmdsettings->flags
        & CLICMDFLAG_PROCINFO)
    {
    }

    int saveON_last = (*saveON);

    char FITSffilename[
        STRINGMAXLEN_FULLFILENAME];
    strcpy(FITSffilename, "null");

    char ASCIITIMEffilename[
        STRINGMAXLEN_FULLFILENAME];
    strcpy(ASCIITIMEffilename, "null");

    double *array_time =
        (double *) malloc(sizeof(double)
                          * (*cubesize) * 2);
    double *array_aqtime =
        (double *) malloc(sizeof(double)
                          * (*cubesize) * 2);
    uint64_t *array_cnt0 =
        (uint64_t *) malloc(sizeof(uint64_t)
                            * (*cubesize) * 2);
    uint64_t *array_cnt1 =
        (uint64_t *) malloc(sizeof(uint64_t)
                            * (*cubesize) * 2);

    int thread_initialized = 0;

    *framecnt   = 0;
    *frameindex = 0;
    *filecnt    = 0;

    int lastcube = 0;

    uint64_t lastcnt0 = 0;
    int IsNewFrame = 0;

    /* Runtime fpi lookups for params that
     * need direct FPS manipulation */
    long fpi_saveON = -1;
    long fpi_lastcubeON = -1;
    long fpi_nextcube = -1;

    if(dcfpsptr != NULL)
    {
        fpi_saveON =
            functionparameter_GetParamIndex(
                dcfpsptr, ".saveON");
        fpi_lastcubeON =
            functionparameter_GetParamIndex(
                dcfpsptr, ".lastcubeON");
        fpi_nextcube =
            functionparameter_GetParamIndex(
                dcfpsptr, ".nextcube");
    }

    printf("Start loop\n");
    fflush(stdout);

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        if(processinfo->triggerstatus
            == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)
        {
            printf("------------ TIMEOUT\n");
            fflush(stdout);
        }
        else
        {
            if(lastcnt0 != inimg.md->cnt0)
            {
                IsNewFrame = 1;
                lastcnt0 = inimg.md->cnt0;
            }
            else
            {
                IsNewFrame = 0;
            }

            if(IsNewFrame == 1)
            {

                if((saveON_last == 0)
                    && ((*saveON) == 1))
                {
                    lastcube = 0;
                    (*framecnt) = 0;
                    (*filecnt) = 0;
                }

                if((*framecnt)
                    >= (*maxframecnt))
                {
                    (*saveON) = 0;
                    if(fpi_saveON >= 0)
                        dcfpsptr
                            ->parray[fpi_saveON]
                            .fpflag
                            &= ~FPFLAG_ONOFF;
                }

                if((*filecnt)
                    >= (*maxfilecnt) - 1)
                {
                    lastcube = 1;
                }

                if((*saveON) == 1)
                {
                    if((*frameindex) == 0)
                    {
                        printf(
                            "========================="
                            " CONSTRUCT FILE NAMES"
                            " =========================\n");
                        fflush(stdout);

                        time_t t;
                        struct tm *uttimeStart;
                        t = time(NULL);
                        uttimeStart = gmtime(&t);

                        char hrminstring[6];
                        sprintf(hrminstring,
                                "%02d:%02d",
                                uttimeStart->tm_hour,
                                uttimeStart->tm_min);
                        printf("hrmin: %s\n",
                               hrminstring);

                        struct timespec
                            timenowStart;
                        clock_gettime(
                            CLOCK_MILK,
                            &timenowStart);

                        WRITE_FULLFILENAME(
                            FITSffilename,
                            "%s/%s_%s:%02ld"
                            ".%09ld.fits",
                            savedirname,
                            streamname,
                            hrminstring,
                            timenowStart.tv_sec
                            % 60,
                            timenowStart.tv_nsec);

                        if(VERBOSE > 0)
                        {
                            printf(
                                "    [%5d]"
                                " FITSffilename"
                                "      = %s\n",
                                __LINE__,
                                FITSffilename);
                        }

                        WRITE_FULLFILENAME(
                            ASCIITIMEffilename,
                            "%s/%s_%s:%02ld"
                            ".%09ld.txt",
                            savedirname,
                            streamname,
                            hrminstring,
                            timenowStart.tv_sec
                            % 60,
                            timenowStart.tv_nsec);

                        if(VERBOSE > 0)
                        {
                            printf(
                                "    [%5d]"
                                " ASCIITIMEffilename"
                                " = %s\n",
                                __LINE__,
                                ASCIITIMEffilename);
                        }

                        printf(
                            "========================="
                            " CONSTRUCT FILE NAMES"
                            " =========================\n");
                        fflush(stdout);
                    }

                    {
                        long tindex =
                            (*frameindex)
                            + buffindex
                            * (*cubesize);
                        {
                            array_cnt0[tindex] =
                                inimg.md->cnt0;
                            array_cnt1[tindex] =
                                inimg.md->cnt1;

                            struct timespec
                                timenow;
                            clock_gettime(
                                CLOCK_MILK,
                                &timenow);
                            array_time[tindex] =
                                timenow.tv_sec
                                + 1.0e-9
                                * timenow.tv_nsec;

                            if(aqtimekwi != -1)
                            {
                                array_aqtime[
                                    tindex] =
                                    1.0e-6
                                    * inimg.im
                                    ->kw[aqtimekwi]
                                    .value.numl;
                            }
                            else
                            {
                                array_aqtime[
                                    tindex]
                                    = 0.0;
                            }
                        }
                    }

                    {
                        long framesize =
                            typesize * xsize
                            * ysize;

                        char *ptr0_0;
                        char *ptr0;

                        ptr0_0 = (char *)
                            inimg.im->array.raw;
                        if(inimg.md->naxis == 3)
                        {
                            ptr0 = ptr0_0
                                + framesize
                                * inimg.md->cnt1;
                        }
                        else
                        {
                            ptr0 = ptr0_0;
                        }

                        char *ptr1_0;
                        char *ptr1;
                        if(buffindex == 0)
                        {
                            ptr1_0 = (char *)
                                imgbuff0.im
                                ->array.raw;
                        }
                        else
                        {
                            ptr1_0 = (char *)
                                imgbuff1.im
                                ->array.raw;
                        }
                        ptr1 = ptr1_0
                            + framesize
                            * (*frameindex);

                        memcpy((void *) ptr1,
                               (void *) ptr0,
                               framesize);
                    }

                    processinfo_WriteMessage_fmt(
                        processinfo,
                        "buff %d file %lu"
                        " frameindex %lu",
                        buffindex,
                        (*filecnt),
                        (*frameindex));

                    (*frameindex) ++;
                    (*framecnt) ++;
                }
                else
                {
                    processinfo_WriteMessage(
                        processinfo,
                        "save = OFF");
                }
            }
        }

        int SaveCube = 0;

        if((*frameindex) >= (*cubesize))
        {
            SaveCube = 1;
        }

        if((saveON_last == 1)
            && ((*saveON) == 0))
        {
            SaveCube = 1;
        }

        if((*nextcube) == 1)
        {
            (*nextcube) = 0;
            if(fpi_nextcube >= 0)
                dcfpsptr
                    ->parray[fpi_nextcube]
                    .fpflag &= ~FPFLAG_ONOFF;
            SaveCube = 1;
        }

        if(processinfo->triggerstatus
            == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)
        {
            SaveCube = 1;
        }

        if(SaveCube == 1)
        {
            if((*frameindex) > 0)
            {
                printf(
                    "SAVING %5ld FRAMES"
                    " of BUFFER %d to"
                    " FILE %s\n",
                    (*frameindex), buffindex,
                    FITSffilename);
                fflush(stdout);

                if(buffindex == 0)
                {
                    memcpy(imgbuff0.im->kw,
                           inimg.im->kw,
                           sizeof(IMAGE_KEYWORD)
                           * inimg.md->NBkw);
                }
                else
                {
                    memcpy(imgbuff1.im->kw,
                           inimg.im->kw,
                           sizeof(IMAGE_KEYWORD)
                           * inimg.md->NBkw);
                }

                {
                    static pthread_t
                        thread_savefits;
                    static int
                        iret_savefits;

                    strcpy(tmsg->fname,
                           FITSffilename);
                    strcpy(tmsg->fnameascii,
                           ASCIITIMEffilename);
                    tmsg->saveascii = 1;
                    tmsg->cubesize =
                        (*frameindex);

                    if((*frameindex)
                        != (*cubesize))
                    {
                        tmsg->partial = 1;
                    }
                    else
                    {
                        tmsg->partial = 0;
                    }

                    if(buffindex == 0)
                    {
                        strcpy(tmsg->iname,
                               imgbuff0.md
                               ->name);
                        tmsg->arrayindex =
                            array_cnt0;
                        tmsg->arraycnt0 =
                            array_cnt0;
                        tmsg->arraycnt1 =
                            array_cnt1;
                        tmsg->arraytime =
                            array_time;
                        tmsg->arrayaqtime =
                            array_aqtime;
                    }
                    else
                    {
                        strcpy(tmsg->iname,
                               imgbuff1.md
                               ->name);
                        tmsg->arrayindex =
                            &array_cnt0[
                                (*cubesize)];
                        tmsg->arraycnt0 =
                            &array_cnt0[
                                (*cubesize)];
                        tmsg->arraycnt1 =
                            &array_cnt1[
                                (*cubesize)];
                        tmsg->arraytime =
                            &array_time[
                                (*cubesize)];
                        tmsg->arrayaqtime =
                            &array_aqtime[
                                (*cubesize)];
                    }

                    WRITE_FILENAME(
                        tmsg
                        ->fname_auxFITSheader,
                        "%s/%s.aux.fits",
                        dcshmdir,
                        streamname);

                    if((*compressON) == 0)
                    {
                        strcpy(
                            tmsg
                            ->compress_string,
                            "");
                    }
                    else
                    {
                        strcpy(
                            tmsg
                            ->compress_string,
                            "[compress R"
                            " 1,1,10000]");
                    }

                    if(thread_initialized == 1)
                    {
                        long cnt0start =
                            inimg.md->cnt0;

                        if(pthread_tryjoin_np(
                                thread_savefits,
                                NULL)
                            == EBUSY)
                        {
                            if(VERBOSE > 0)
                            {
                                printf(
                                    "%5d  PREVIOUS"
                                    " SAVE THREAD"
                                    " NOT TERMINATED"
                                    " -> waiting\n",
                                    __LINE__);
                            }
                            pthread_join(
                                thread_savefits,
                                NULL);
                            if(VERBOSE > 0)
                            {
                                printf(
                                    "%5d  PREVIOUS"
                                    " SAVE THREAD"
                                    " NOW COMPLETED"
                                    " -> continuing"
                                    "\n",
                                    __LINE__);
                            }
                        }
                        else
                        {
                            if(VERBOSE > 0)
                            {
                                printf(
                                    "%5d  PREVIOUS"
                                    " SAVE THREAD"
                                    " ALREADY"
                                    " COMPLETED"
                                    " -> OK\n",
                                    __LINE__);
                            }
                        }
                        (*savetime) =
                            tmsg->timespan;
                        printf(
                            "\n **************"
                            " MISSED  %ld"
                            " frames\n",
                            inimg.md->cnt0
                            - cnt0start);
                    }

                    tmsg->writerRTprio =
                        (*writerRTprio);
                    iret_savefits =
                        pthread_create(
                            &thread_savefits,
                            NULL,
                            save_telemetry_fits_function,
                            tmsg);

                    thread_initialized = 1;
                    if(iret_savefits)
                    {
                        fprintf(stderr,
                                "Error -"
                                " pthread_create()"
                                " return code:"
                                " %d\n",
                                iret_savefits);
                        exit(EXIT_FAILURE);
                    }
                }
                SaveCube = 0;

                (*frameindex) = 0;
                (*filecnt) ++;
            }

            if(buffindex == 0)
            {
                processinfo_update_output_stream(
                    processinfo,
                    imgbuff0.im, NULL);
            }
            else
            {
                processinfo_update_output_stream(
                    processinfo,
                    imgbuff1.im, NULL);
            }

            buffindex ++;
            if(buffindex > 1)
            {
                buffindex = 0;
            }

            if((lastcube == 1)
                || ((*lastcubeON) == 1))
            {
                (*saveON) = 0;
                if(fpi_saveON >= 0)
                    dcfpsptr
                        ->parray[fpi_saveON]
                        .fpflag
                        &= ~FPFLAG_ONOFF;

                (*lastcubeON) = 0;
                if(fpi_lastcubeON >= 0)
                    dcfpsptr
                        ->parray[fpi_lastcubeON]
                        .fpflag
                        &= ~FPFLAG_ONOFF;
            }
        }

        saveON_last = (*saveON);
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(array_time);
    free(array_aqtime);
    free(array_cnt0);
    free(array_cnt1);

    free(tmsg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_MEMORY__logshmim()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    CLIcmddata.FPS_customCONFsetup =
        customCONFsetup;
    CLIcmddata.FPS_customCONFcheck =
        customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    FPS_PARAMS,
    compute_function,
    customCONFcheck)
#endif

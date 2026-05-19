/**
 * @file    logshmim.c
 * @brief   Save telemetry stream data
 *
 * Uses FPS V2 framework.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ImageStreamIO/ImageStruct.h"

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
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#ifdef USE_CFITSIO
#include "COREMOD_iofits/COREMOD_iofits.h"
#endif


#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"
#include "stream_sem.h"

#include "shmimlog_types.h"

/* LIKELY/UNLIKELY macros from milk_compiler.h */


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamFITSlog",
    .cmdkey      = "streamFITSlog",
    .description =
        "log stream to FITS file(s)",
    .description_long =
        "Log shared memory image stream frames to disk as FITS files. Records every new frame triggered by semaphore posting, with configurable logging duration and file naming."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     streamname[
    FUNCTION_PARAMETER_STRMAXLEN] = "stream";
static int32_t  saveON        = 0;
static int32_t  lastcubeON    = 0;
static int32_t  nextcube      = 0;
static uint32_t cubesize      = 10000;
static char     savedirname[
    FUNCTION_PARAMETER_STRMAXLEN] = ".";
static uint64_t frameindex    = 0;
static uint64_t framecnt      = 0;
static uint64_t maxframecnt   = 0;
static uint64_t filecnt       = 0;
static uint64_t maxfilecnt    = 0;
static char     outfname[
    FUNCTION_PARAMETER_STRMAXLEN] = "";
static int32_t  compressON    = 0;
static float    savetime      = 0.0;
static uint32_t writerRTprio  = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".sname", streamname, \
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
    X(".dirname", savedirname, \
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
    X(".outfname", outfname, \
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

static MILK_COLD errno_t __attribute__((unused)) customCONFsetup()
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

static MILK_COLD errno_t __attribute__((unused)) customCONFcheck()
{
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* save_telemetry_fits_function() is in
 * logshmim_save.c */
extern void *save_telemetry_fits_function(
    void *ptr);




/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
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
        dcimg,  dcnimg);

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
    uint32_t zsize = cubesize;

    uint8_t datatype = inimg.md->datatype;

    int typesize =
        ImageStreamIO_typesize(datatype);
    if(typesize == -1)
    {
        PRINT_ERROR("wrong data type %d",
                    (int) datatype);
        return RETURN_FAILURE;
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

    int saveON_last = saveON;

    char FITSffilename[
        STRINGMAXLEN_FULLFILENAME];
    snprintf(FITSffilename,
             sizeof(FITSffilename),
             "null");

    char ASCIITIMEffilename[
        STRINGMAXLEN_FULLFILENAME];
    snprintf(ASCIITIMEffilename,
             sizeof(ASCIITIMEffilename),
             "null");

    double *array_time =
        (double *) malloc(sizeof(double)
                          * cubesize * 2);
    double *array_aqtime =
        (double *) malloc(sizeof(double)
                          * cubesize * 2);
    uint64_t *array_cnt0 =
        (uint64_t *) malloc(sizeof(uint64_t)
                            * cubesize * 2);
    uint64_t *array_cnt1 =
        (uint64_t *) malloc(sizeof(uint64_t)
                            * cubesize * 2);

    int thread_initialized = 0;

    framecnt   = 0;
    frameindex = 0;
    filecnt    = 0;

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

    if(VERBOSE > 0)
    {
        printf("Start loop\n");
        fflush(stdout);
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        if(processinfo->triggerstatus
            == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)
        {
            if(VERBOSE > 0)
            {
                printf("------------ TIMEOUT\n");
                fflush(stdout);
            }
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
                    && (saveON == 1))
                {
                    lastcube = 0;
                    framecnt = 0;
                    filecnt = 0;
                }

                if(framecnt
                    >= maxframecnt)
                {
                    saveON = 0;
                    if(fpi_saveON >= 0)
                        dcfpsptr
                            ->parray[fpi_saveON]
                            .fpflag
                            &= ~FPFLAG_ONOFF;
                }

                if(filecnt
                    >= maxfilecnt - 1)
                {
                    lastcube = 1;
                }

                if(saveON == 1)
                {
                    if(frameindex == 0)
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
                        snprintf(
                            hrminstring,
                            sizeof(hrminstring),
                            "%02d:%02d",
                            uttimeStart->tm_hour,
                            uttimeStart->tm_min);
                        if(VERBOSE > 0)
                        {
                            printf("hrmin: %s\n",
                                   hrminstring);
                        }

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

                        if(VERBOSE > 0)
                        {
                            printf(
                                "========================="
                                " CONSTRUCT FILE NAMES"
                                " =========================\n");
                            fflush(stdout);
                        }
                    }

                    {
                        long tindex =
                            frameindex
                            + buffindex
                            * cubesize;
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
                            * frameindex;

                        __builtin_memcpy(
                               (void *) ptr1,
                               (void *) ptr0,
                               framesize);
                    }

                    processinfo_WriteMessage_fmt(
                        processinfo,
                        "buff %d file %lu"
                        " frameindex %lu",
                        buffindex,
                        filecnt,
                        frameindex);

                    frameindex ++;
                    framecnt ++;
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

        if(frameindex >= cubesize)
        {
            SaveCube = 1;
        }

        if((saveON_last == 1)
            && (saveON == 0))
        {
            SaveCube = 1;
        }

        if(nextcube == 1)
        {
            nextcube = 0;
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
            if(frameindex > 0)
            {
                if(VERBOSE > 0)
                {
                    printf(
                        "SAVING %5ld FRAMES"
                        " of BUFFER %d to"
                        " FILE %s\n",
                        (long) frameindex,
                        buffindex,
                        FITSffilename);
                    fflush(stdout);
                }

                if(buffindex == 0)
                {
                    __builtin_memcpy(
                           imgbuff0.im->kw,
                           inimg.im->kw,
                           sizeof(IMAGE_KEYWORD)
                           * inimg.md->NBkw);
                }
                else
                {
                    __builtin_memcpy(
                           imgbuff1.im->kw,
                           inimg.im->kw,
                           sizeof(IMAGE_KEYWORD)
                           * inimg.md->NBkw);
                }

                {
                    static pthread_t
                        thread_savefits;
                    static int
                        iret_savefits;

                    snprintf(
                        tmsg->fname,
                        sizeof(tmsg->fname),
                        "%s",
                        FITSffilename);
                    snprintf(
                        tmsg->fnameascii,
                        sizeof(
                            tmsg->fnameascii),
                        "%s",
                        ASCIITIMEffilename);
                    tmsg->saveascii = 1;
                    tmsg->cubesize =
                        frameindex;

                    if(frameindex
                        != cubesize)
                    {
                        tmsg->partial = 1;
                    }
                    else
                    {
                        tmsg->partial = 0;
                    }

                    if(buffindex == 0)
                    {
                        snprintf(
                            tmsg->iname,
                            sizeof(
                                tmsg->iname),
                            "%s",
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
                        snprintf(
                            tmsg->iname,
                            sizeof(
                                tmsg->iname),
                            "%s",
                            imgbuff1.md
                            ->name);
                        tmsg->arrayindex =
                            &array_cnt0[
                                cubesize];
                        tmsg->arraycnt0 =
                            &array_cnt0[
                                cubesize];
                        tmsg->arraycnt1 =
                            &array_cnt1[
                                cubesize];
                        tmsg->arraytime =
                            &array_time[
                                cubesize];
                        tmsg->arrayaqtime =
                            &array_aqtime[
                                cubesize];
                    }

                    snprintf(
                        tmsg->fname_auxFITSheader,
                        sizeof(tmsg->fname_auxFITSheader),
                        "%s/%s.aux.fits",
                        dcshmdir,
                        streamname);

                    if(compressON == 0)
                    {
                        tmsg->compress_string[0] = '\0';
                    }
                    else
                    {
                        snprintf(
                            tmsg
                            ->compress_string,
                            sizeof(
                                tmsg
                                ->compress_string),
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
                            void *tret_ptr = NULL;
                            pthread_join(
                                thread_savefits,
                                &tret_ptr);
                            free(tret_ptr);
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
                        savetime =
                            tmsg->timespan;
                        if(VERBOSE > 0)
                        {
                            printf(
                                "\n **************"
                                " MISSED  %ld"
                                " frames\n",
                                inimg.md->cnt0
                                - cnt0start);
                        }
                    }

                    tmsg->writerRTprio =
                        writerRTprio;
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

                frameindex = 0;
                filecnt ++;
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
                || (lastcubeON == 1))
            {
                saveON = 0;
                if(fpi_saveON >= 0)
                    dcfpsptr
                        ->parray[fpi_saveON]
                        .fpflag
                        &= ~FPFLAG_ONOFF;

                lastcubeON = 0;
                if(fpi_lastcubeON >= 0)
                    dcfpsptr
                        ->parray[fpi_lastcubeON]
                        .fpflag
                        &= ~FPFLAG_ONOFF;
            }
        }

        saveON_last = saveON;
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

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_delay.c
 * @brief   delay input stream to output stream
 *
 * Uses FPS V2 framework.
 */


#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"
#include "timeutils.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamdelay",
    .cmdkey      = "streamdelay",
    .description = "delay input stream to output stream",
    .description_long =
        "Introduce a configurable frame delay on an image stream. Buffers incoming frames in a "
        "circular buffer and outputs frames from N steps earlier. Useful for simulating latency."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static char     outimname[FUNCTION_PARAMETER_STRMAXLEN] = "imout";
static float    delaysec                                = 0.001;
static uint8_t  naive_mode                              = 0;
static uint64_t timebuffsize                            = 1000;
static int32_t  avemode                                 = 0;
static uint64_t avedtns                                 = 0;
static uint64_t statusframelag                          = 0;
static uint64_t statuskkin                              = 0;
static uint64_t statuskkout                             = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                             \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")            \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")             \
    X(".delaysec", &delaysec, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "delay [s]")               \
    X(".naive_mode", &naive_mode, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                          \
      "Naive wait mode (ON: no buffer, busywait sleep)")                                          \
    X(".timebuffsize", &timebuffsize, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT, "time buffer size") \
    X(".option.timeavemode", &avemode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Enable time window averaging (>0)")                                                        \
    X(".option.timeavedtns", &avedtns, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Averaging time window width [ns]")                                                         \
    X(".status.framelag", &statusframelag, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,               \
      "current time lag frame index")                                                             \
    X(".status.kkin", &statuskkin, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,                       \
      "input cube slice index")                                                                   \
    X(".status.kkout", &statuskkout, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,                     \
      "output cube slice index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_COLD errno_t __attribute__((unused)) customCONFcheck()
{
    if (dcfpsptr != NULL)
    {
        long fpi_avemode = functionparameter_GetParamIndex(dcfpsptr, ".option.timeavemode");
        long fpi_dtns    = functionparameter_GetParamIndex(dcfpsptr, ".option.timeavedtns");

        if (fpi_avemode >= 0 && fpi_dtns >= 0)
        {
            if (dcfpsptr->parray[fpi_avemode].val.i32[0] == 0)
            {
                dcfpsptr->parray[fpi_dtns].fpflag &= ~FPFLAG_USED;
                dcfpsptr->parray[fpi_dtns].fpflag &= ~FPFLAG_VISIBLE;
            }
            else
            {
                dcfpsptr->parray[fpi_dtns].fpflag |= FPFLAG_USED;
                dcfpsptr->parray[fpi_dtns].fpflag |= FPFLAG_VISIBLE;
            }
        }
    }

    return RETURN_SUCCESS;
}

static errno_t streamdelay(IMGID            inimg,
                           IMGID            outimg,
                           IMGID            bufferimg,
                           struct timespec *tarray,
                           int             *warray,
                           int             *status)
{
    static uint64_t cnt0prev           = 0;
    static uint64_t bufferindex_input  = 0;
    static uint64_t bufferindex_output = 0;

    struct timespec t_now = inimg.md->writetime;
    struct timespec t_new;

    //milk_clock_gettime(&t_now);
    if (naive_mode)
    {
        do
        {
            milk_clock_gettime(&t_new);
            usleep(1);
        } while ((t_new.tv_sec - t_now.tv_sec) + (t_new.tv_nsec - t_now.tv_nsec) / 1e9 < delaysec);

        outimg.md->write = 1;
        memcpy(outimg.im->array.raw, inimg.im->array.raw, inimg.md->imdatamemsize);

        *status = 1;
        return RETURN_SUCCESS;
    }

    if (cnt0prev != inimg.md->cnt0)
    {
        cnt0prev = inimg.md->cnt0;

        tarray[bufferindex_input].tv_sec  = t_now.tv_sec;
        tarray[bufferindex_input].tv_nsec = t_now.tv_nsec;

        char *destptr;
        destptr = (char *) bufferimg.im->array.raw;
        destptr += inimg.md->imdatamemsize * bufferindex_input;
        __builtin_memcpy(destptr, inimg.im->array.raw, inimg.md->imdatamemsize);

        warray[bufferindex_input] = 0;

        bufferindex_input++;
        if (bufferindex_input == (timebuffsize))
        {
            bufferindex_input = 0;
        }
    }

    struct timespec tdiff  = timespec_diff(tarray[bufferindex_output], t_now);
    double          tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    int  updateflag              = 0;
    long bufferindex_output_last = 0;
    while ((warray[bufferindex_output] == 0) && (tdiffv > (delaysec)))
    {
        updateflag                 = 1;
        warray[bufferindex_output] = 1;

        bufferindex_output_last = bufferindex_output;
        bufferindex_output++;
        if (bufferindex_output == (timebuffsize))
        {
            bufferindex_output = 0;
        }

        tdiff  = timespec_diff(tarray[bufferindex_output], t_now);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
    }

    if (updateflag == 1)
    {
        printf("     WRITE %8ld %8ld  :  "
               "%ld bytes\n",
               bufferindex_input, bufferindex_output_last, (long) inimg.md->imdatamemsize);
        char *srcptr;
        srcptr = (char *) bufferimg.im->array.raw;
        srcptr += inimg.md->imdatamemsize * bufferindex_output_last;
        outimg.md->write = 1;
        memcpy(outimg.im->array.raw, srcptr, inimg.md->imdatamemsize);

        *status = 1;
    }
    else
    {
        *status = 0;
    }

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // TODO: just doesn't work for not 2D streams. Because the buffer and all.
    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID outimg    = stream_connect_create_2D(outimname, inimg.md->size[0], inimg.md->size[1],
                                               inimg.md->datatype);
    IMGID bufferimg = imgid_make_from_name_3D("streamdelaybuff", inimg.mdt->size[0],
                                              inimg.mdt->size[1], timebuffsize);
    bufferimg.mdt->datatype = inimg.mdt->datatype;
    bufferimg.mdt->shared   = 0;
    imcreateIMGID(&bufferimg);

    struct timespec *timeinarray;
    timeinarray = (struct timespec *) calloc(timebuffsize, sizeof(struct timespec));
    struct timespec tnow;
    milk_clock_gettime(&tnow);
    for (uint64_t i = 0; i < timebuffsize; i++)
    {
        timeinarray[i].tv_sec  = tnow.tv_sec;
        timeinarray[i].tv_nsec = tnow.tv_nsec;
    }

    int *warray;
    warray = (int *) calloc(timebuffsize, sizeof(int));
    for (uint64_t i = 0; i < timebuffsize; i++)
    {
        warray[i] = 1;
    }

    list_image_ID();
    int status = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    streamdelay(inimg, outimg, bufferimg, timeinarray, warray, &status);
    if (status != 0)
    {
        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(timeinarray);
    free(warray);
    imgid_free(&inimg);
    imgid_free(&outimg);
    imgid_free(&bufferimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_memory__streamdelay()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(FPS_app_info, FPS_PARAMS, compute_function, customCONFcheck)
#endif

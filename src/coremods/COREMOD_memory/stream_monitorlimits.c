/**
 * @file    stream_monitorlimits.c
 * @brief   Monitor stream values for safety limits
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "image_ID.h"
#include "processinfo.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streammlim",
    .cmdkey      = "streammlim",
    .description =
        "monitor stream values for safety",
    .description_long =
        "Monitor pixel values in a stream and flag frames where values exceed configurable min/max thresholds. Reports out-of-range statistics."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    inimname[FUNCTION_PARAMETER_STRMAXLEN] = "stream";
static int64_t dtus     = 100000;
static int32_t minON    = 0;
static float   minVal   = 0.0;
static int32_t maxON    = 0;
static float   maxVal   = 1000.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".dtus", &dtus, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "loop period [us]") \
    X(".minON", &minON, \
      FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "minimum limit toggle") \
    X(".minVal", &minVal, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "minimum limit value") \
    X(".maxON", &maxON, \
      FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "maximum limit toggle") \
    X(".maxVal", &maxVal, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "maximum limit value")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Actual monitoring logic
 */
static errno_t monitor_logic(IMGID *imgptr)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgptr, ERRMODE_ABORT, dcimg, dcnimg);

    uint32_t xsize  = imgptr->md->size[0];
    uint32_t ysize  = imgptr->md->size[1];
    uint64_t xysize = (uint64_t)xsize * ysize;

    float minv = imgptr->im->array.F[0];
    float maxv = imgptr->im->array.F[0];

    for(uint64_t ii = 1; ii < xysize; ii++)
    {
        if(imgptr->im->array.F[ii] < minv) {
            minv = imgptr->im->array.F[ii];
        }
        if(imgptr->im->array.F[ii] > maxv) {
            maxv = imgptr->im->array.F[ii];
        }
    }

    int limit_exceeded = 0;
    char msg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];
    msg[0] = '\0';

    if(minON && (minv < minVal))
    {
        limit_exceeded = 1;
        snprintf(
            msg, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "MIN LIMIT EXCEEDED: %f < %f", minv, minVal);
    }

    if(maxON && (maxv > maxVal))
    {
        limit_exceeded = 1;
        if(msg[0] != '\0')
        {
            strncat(msg, " | ", STRINGMAXLEN_PROCESSINFO_STATUSMSG - strlen(msg) - 1);
        }
        char tmpmsg[100];
        snprintf(tmpmsg, 100, "MAX LIMIT EXCEEDED: %f > %f", maxv, maxVal);
        strncat(msg, tmpmsg, STRINGMAXLEN_PROCESSINFO_STATUSMSG - strlen(msg) - 1);
    }

    if(limit_exceeded)
    {
        PRINT_WARNING("%s", msg);
    }

    DEBUG_TRACE_FEXIT();
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

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg,  dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    if(dcprocinfo == 1)
    {
        processinfo_waitoninputstream_init(
            processinfo, inimg.im, PROCESSINFO_TRIGGERMODE_DELAY, -1);
        processinfo->triggerdelay.tv_sec = 0;
        processinfo->triggerdelay.tv_nsec = dtus * 1000;
        while(processinfo->triggerdelay.tv_nsec
              >= 1000000000)
        {
            processinfo->triggerdelay.tv_nsec -= 1000000000;
            processinfo->triggerdelay.tv_sec += 1;
        }
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        monitor_logic(&inimg);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END  DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

errno_t stream_monitorlimits_addCLIcmd()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif


/* ================================================================
 * BACKWARD COMPAT WRAPPER
 * ============================================================= */

errno_t stream_monitorlimits(
    const char *instreamname)
{
    strncpy(inimname, instreamname, FUNCTION_PARAMETER_STRMAXLEN - 1);
    dtus   = 100000;
    minON  = 0;
    minVal = 0.0;
    maxON  = 0;
    maxVal = 1000.0;

    return compute_function();
}
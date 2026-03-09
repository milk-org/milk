/**
 * @file    stream_monitorlimits.c
 * @brief   Monitor stream values for safety limits
 *
 * Uses FPS V2 framework.
 */

#include "CLIcore.h"
#include "fps.h"

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
        "monitor stream values for safety"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *inimname = NULL;
static int64_t *dtus     = NULL;
static int32_t *minON    = NULL;
static float   *minVal   = NULL;
static int32_t *maxON    = NULL;
static float   *maxVal   = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", &inimname, \
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

    resolveIMGID(
        imgptr, ERRMODE_ABORT,
        dcimg, dcnimg);

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
    char msg[
        STRINGMAXLEN_PROCESSINFO_STATUSMSG];
    msg[0] = '\0';

    if(*minON && (minv < *minVal))
    {
        limit_exceeded = 1;
        snprintf(
            msg,
            STRINGMAXLEN_PROCESSINFO_STATUSMSG,
            "MIN LIMIT EXCEEDED: %f < %f",
            minv, *minVal);
    }

    if(*maxON && (maxv > *maxVal))
    {
        limit_exceeded = 1;
        if(msg[0] != '\0')
        {
            strncat(msg, " | ",
                    STRINGMAXLEN_PROCESSINFO_STATUSMSG
                    - strlen(msg) - 1);
        }
        char tmpmsg[100];
        snprintf(tmpmsg, 100,
                 "MAX LIMIT EXCEEDED: %f > %f",
                 maxv, *maxVal);
        strncat(msg, tmpmsg,
                STRINGMAXLEN_PROCESSINFO_STATUSMSG
                - strlen(msg) - 1);
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
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg =
        imgid_make_from_name(inimname);
    resolveIMGID(
        &inimg, ERRMODE_ABORT,
        dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    if(dcprocinfo == 1)
    {
        processinfo_waitoninputstream_init(
            processinfo, inimg.im,
            PROCESSINFO_TRIGGERMODE_DELAY,
            -1);
        processinfo->triggerdelay.tv_sec = 0;
        processinfo->triggerdelay.tv_nsec =
            (*dtus) * 1000;
        while(processinfo->triggerdelay.tv_nsec
              >= 1000000000)
        {
            processinfo->triggerdelay.tv_nsec
                -= 1000000000;
            processinfo->triggerdelay.tv_sec
                += 1;
        }
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        monitor_logic(&inimg);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t stream_monitorlimits_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
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
    inimname = strdup(instreamname);
    int64_t default_dtus  = 100000;
    int32_t default_minON = 0;
    float   default_minVal = 0.0;
    int32_t default_maxON = 0;
    float   default_maxVal = 1000.0;

    dtus   = &default_dtus;
    minON  = &default_minON;
    minVal = &default_minVal;
    maxON  = &default_maxON;
    maxVal = &default_maxVal;

    return compute_function();
}
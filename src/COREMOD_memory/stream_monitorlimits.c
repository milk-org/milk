/**
 * @file stream_monitorlimits.c
 * @brief Monitors stream to fit within limits.
 *
 * Example of a stream monitoring loop with FPS and processinfo support.
 */

#include "CLIcore.h"
#include "create_image.h"
#include "image_ID.h"
#include "processinfo.h"

// Local variables pointers
static char *inimname;
static int64_t *dtus;
static int64_t *minON;
static float *minVal;
static int64_t *maxON;
static float *maxVal;

static long fpi_dtus = -1;
static long fpi_minON = -1;
static long fpi_minVal = -1;
static long fpi_maxON = -1;
static long fpi_maxVal = -1;

// List of arguments to function
static CLICMDARGDEF farg[] = {
    {
        CLIARG_IMG,
        ".in_name",
        "input stream",
        "im1",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **)&inimname,
        NULL
    },
    {
        CLIARG_INT64,
        ".dtus",
        "loop period [us]",
        "100000",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **)&dtus,
        &fpi_dtus
    },
    {
        CLIARG_ONOFF,
        ".minON",
        "minimum limit toggle",
        "0",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **)&minON,
        &fpi_minON
    },
    {
        CLIARG_FLOAT32,
        ".minVal",
        "minimum limit value",
        "0.0",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **)&minVal,
        &fpi_minVal
    },
    {
        CLIARG_ONOFF,
        ".maxON",
        "maximum limit toggle",
        "0",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **)&maxON,
        &fpi_maxON
    },
    {
        CLIARG_FLOAT32,
        ".maxVal",
        "maximum limit value",
        "1000.0",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **)&maxVal,
        &fpi_maxVal
    }
};

static CLICMDDATA CLIcmddata = {
    "streammlim",
    "monitor stream values for safety",
    CLICMD_FIELDS_DEFAULTS
};

static errno_t help_function() {
    printf("Monitors an input stream and checks if its values stay within limits.\n");
    printf("If limits are exceeded, a message is written to processinfo.\n");
    return RETURN_SUCCESS;
}

/**
 * @brief Actual monitoring logic
 */
static errno_t monitor_logic(IMGID *imgptr) {
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgptr, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    uint32_t xsize = imgptr->md->size[0];
    uint32_t ysize = imgptr->md->size[1];
    uint64_t xysize = (uint64_t)xsize * ysize;

    float minv = imgptr->im->array.F[0];
    float maxv = imgptr->im->array.F[0];

    for (uint64_t ii = 1; ii < xysize; ii++) {
        if (imgptr->im->array.F[ii] < minv) {
            minv = imgptr->im->array.F[ii];
        }
        if (imgptr->im->array.F[ii] > maxv) {
            maxv = imgptr->im->array.F[ii];
        }
    }

    int limit_exceeded = 0;
    char msg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];
    msg[0] = '\0';

    if (*minON && (minv < *minVal)) {
        limit_exceeded = 1;
        snprintf(msg, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "MIN LIMIT EXCEEDED: %f < %f", minv, *minVal);
    }

    if (*maxON && (maxv > *maxVal)) {
        limit_exceeded = 1;
        if (msg[0] != '\0') {
            strncat(msg, " | ", STRINGMAXLEN_PROCESSINFO_STATUSMSG - strlen(msg) - 1);
        }
        char tmpmsg[100];
        snprintf(tmpmsg, 100, "MAX LIMIT EXCEEDED: %f > %f", maxv, *maxVal);
        strncat(msg, tmpmsg, STRINGMAXLEN_PROCESSINFO_STATUSMSG - strlen(msg) - 1);
    }

    if (limit_exceeded) {
        PRINT_WARNING("%s", msg);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function() {
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    if (data.processinfo == 1) {
        processinfo_waitoninputstream_init(processinfo, inimg.im, PROCESSINFO_TRIGGERMODE_DELAY, -1);
        processinfo->triggerdelay.tv_sec = 0;
        processinfo->triggerdelay.tv_nsec = (*dtus) * 1000;
        while (processinfo->triggerdelay.tv_nsec >= 1000000000) {
            processinfo->triggerdelay.tv_nsec -= 1000000000;
            processinfo->triggerdelay.tv_sec += 1;
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

// Generate FPS and CLI functions
INSERT_STD_FPSCLIfunctions

// Registration function
errno_t stream_monitorlimits_addCLIcmd() {
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

// For backward compatibility if needed
errno_t stream_monitorlimits(const char *instreamname) {
    inimname = strdup(instreamname);
    int64_t default_dtus = 100000;
    int64_t default_minON = 0;
    float default_minVal = 0.0;
    int64_t default_maxON = 0;
    float default_maxVal = 1000.0;

    dtus = &default_dtus;
    minON = &default_minON;
    minVal = &default_minVal;
    maxON = &default_maxON;
    maxVal = &default_maxVal;

    return compute_function();
}
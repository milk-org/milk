/**
 * @file    shmim_purge.c
 * @brief   purge shared memory stream
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "CLIcore.h"
#include "fps.h"
#include "streamCTRL_find_streams.h"

#include "image_ID.h"
#include "read_shmim.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "shmimpurge",
    .cmdkey      = "shmimpurge",
    .description = "purge orphan streams"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *stringfilter = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".strfilter", &stringfilter, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "string filter")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/** @brief purge orphan shared memory streams */
errno_t shmim_purge(const char *strfilter)
{
    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    DEBUG_TRACEPOINT("Searching for streams");
    streaminfo = (STREAMINFO *)
        malloc(sizeof(STREAMINFO) * NBstreamMAX);
    int NBstream =
        find_streams(streaminfo, 1, strfilter);
    printf("%d stream(s) found\n", NBstream);

    DEBUG_TRACEPOINT(
        "scanning %d streams for purging",
        NBstream);
    for(int sindex = 0;
         sindex < NBstream; sindex++)
    {
        printf(" STREAM %3d   %s\n",
               sindex,
               streaminfo[sindex].sname);
        imageID ID = image_ID(
            streaminfo[sindex].sname,
            data.image, data.NB_MAX_IMAGE);
        if(ID == -1)
        {
            ID = read_sharedmem_image(
                streaminfo[sindex].sname,
                data.image, data.NB_MAX_IMAGE);
        }
        DEBUG_TRACEPOINT(
            "stream %s loaded ID %ld",
            streaminfo[sindex].sname,
            (long) ID);

        pid_t opid;
        opid = data.image[ID].md[0].ownerPID;
        DEBUG_TRACEPOINT("owner PID : %ld",
                         (long) opid);
        printf("owner PID : %ld\n",
               (long) opid);

        if(opid != 0)
        {
            if(getpgid(opid) >= 0)
            {
                printf("Keeping stream %s\n",
                       streaminfo[sindex].sname);
            }
            else
            {
                printf("Purging stream %s\n",
                       streaminfo[sindex].sname);
                ImageStreamIO_destroyIm(
                    &data.image[ID]);
            }
        }
        else
        {
            printf("Purging stream %s\n",
                   streaminfo[sindex].sname);
            ImageStreamIO_destroyIm(
                &data.image[ID]);
        }
    }

    free(streaminfo);

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
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    shmim_purge(stringfilter);

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

errno_t
CLIADDCMD_COREMOD_memory__shmim_purge()
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

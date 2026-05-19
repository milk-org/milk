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
#include "COREMOD_memory/COREMOD_memory.h"

#ifndef MILK_NO_CLI
#include "streamCTRL_find_streams.h"
#endif

#include "image_ID.h"
#include "read_shmim.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "shmimpurge",
    .cmdkey      = "shmimpurge",
    .description = "purge orphan streams",
    .description_long =
        "Scan /dev/shm for stale (orphaned) image streams and remove them. A stream is considered stale if no process has it open or if its creator PID no longer exists."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char stringfilter[FUNCTION_PARAMETER_STRMAXLEN] = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".strfilter", stringfilter, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "string filter")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/** @brief purge orphan shared memory streams */
errno_t shmim_purge(const char *strfilter)
{
#ifdef MILK_NO_CLI
    /* STREAMINFO and find_streams require
     * CLI/TUI headers not available in
     * standalone builds.
     */
    (void) strfilter;
    printf("shmim_purge: not available in " "standalone mode\n");
    return RETURN_SUCCESS;
#else
    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    DEBUG_TRACEPOINT("Searching for streams");
    streaminfo = (STREAMINFO *) malloc(sizeof(STREAMINFO) * NBstreamMAX);
    int NBstream = find_streams(streaminfo, 1, strfilter);
    printf("%d stream(s) found\n", NBstream);

    DEBUG_TRACEPOINT("scanning %d streams for purging", NBstream);
    for(int sindex = 0;
         sindex < NBstream; sindex++)
    {
        printf(" STREAM %3d   %s\n", sindex, streaminfo[sindex].sname);
        IMGID img = imgid_make_from_name(streaminfo[sindex].sname);
        resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
        if(img.ID == -1)
        {
            imageID fid = read_sharedmem_image(streaminfo[sindex].sname, dcimg, dcnimg);
            if(fid == -1)
            {
                printf("Failed to load stream %s\n", streaminfo[sindex].sname);
                continue;
            }

            resolveIMGID(&img,  ERRMODE_NULL, dcimg, dcnimg);
            if(img.ID == -1 || img.im == NULL)
            {
                printf("Failed to resolve stream %s after load\n", streaminfo[sindex].sname);
                continue;
            }
        }
        DEBUG_TRACEPOINT("stream %s loaded ID %ld", streaminfo[sindex].sname, (long) img.ID);

        pid_t opid;
        opid = img.im->md[0].ownerPID;
        DEBUG_TRACEPOINT("owner PID : %ld", (long) opid);
        printf("owner PID : %ld\n", (long) opid);

        if(opid != 0)
        {
            if(getpgid(opid) >= 0)
            {
                printf("Keeping stream %s\n", streaminfo[sindex].sname);
            }
            else
            {
                printf("Purging stream %s\n", streaminfo[sindex].sname);
                ImageStreamIO_destroyIm(img.im);
            }
        }
        else
        {
            printf("Purging stream %s\n", streaminfo[sindex].sname);
            ImageStreamIO_destroyIm(img.im);
        }
    }

    free(streaminfo);

    return RETURN_SUCCESS;
#endif /* MILK_NO_CLI */
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START  shmim_purge(stringfilter);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
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

errno_t
CLIADDCMD_COREMOD_memory__shmim_purge()
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

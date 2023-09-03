/**
 * @file    shmim_purge.c
 * @brief   purge shared memory stream
 */

#include <fcntl.h>    // open
#include <sys/mman.h> // mmap
#include <sys/stat.h>
#include <unistd.h> // close

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/streamCTRL/streamCTRL_find_streams.h"

#include "image_ID.h"
#include "read_shmim.h"

// Local variables pointers
static char *stringfilter;

static CLICMDARGDEF farg[] = {{
        CLIARG_STR,
        ".strfilter",
        "string filter",
        "im",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &stringfilter,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "shmimpurge", "purge orphan streams", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

/** @brief purge orphan share memory streams
 *
 *
 */
errno_t shmim_purge(const char *strfilter)
{
    //printf("PURGING ORPHAN STREAMS (matching %s)\n", strfilter);

    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    DEBUG_TRACEPOINT("Searching for streams");
    streaminfo   = (STREAMINFO *) malloc(sizeof(STREAMINFO) * NBstreamMAX);
    int NBstream = find_streams(streaminfo, 1, strfilter);
    printf("%d stream(s) found\n", NBstream);

    DEBUG_TRACEPOINT("scanning %d streams for purging", NBstream);
    for(int sindex = 0; sindex < NBstream; sindex++)
    {
        printf(" STREAM %3d   %s\n", sindex, streaminfo[sindex].sname);
        imageID ID = image_ID(streaminfo[sindex].sname);
        if(ID == -1)
        {
            ID = read_sharedmem_image(streaminfo[sindex].sname);
        }
        DEBUG_TRACEPOINT("stream %s loaded ID %ld",
                         streaminfo[sindex].sname,
                         (long) ID);

        pid_t opid; // owner PID
        opid = data.image[ID].md[0].ownerPID;
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
                ImageStreamIO_destroyIm(&data.image[ID]);
            }
        }
        else
        {
            // owner unset: assumes no owner
            printf("Purging stream %s\n", streaminfo[sindex].sname);
            ImageStreamIO_destroyIm(&data.image[ID]);
        }
    }

    free(streaminfo);

    return RETURN_SUCCESS;
}

// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    shmim_purge(stringfilter);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_memory__shmim_purge()
{
    INSERT_STD_CLIREGISTERFUNC

    // Optional custom settings for this function
    // CLIcmddata.cmdsettings->procinfo_loopcntMax = 9;

    return RETURN_SUCCESS;
}

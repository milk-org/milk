#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_setup.h"
#include "processinfo_shm_create.h"
#include "processinfo_signals.h"
#include "processinfo_WriteMessage.h"
#include "processinfo_SIGexit.h"

// High level processinfo function

/**
 * @brief Initialize and register a process for MILK process management.
 *
 * This function performs the first-time setup of a process's shared memory
 * status structure.
 *
 * Logic flow:
 * 1.  Ensure the processinfo instance name is valid.
 * 2.  Call `processinfo_shm_create` to physically create and map the SHM segment.
 * 3.  Initialize metadata fields: loop status (INIT), source code location,
 *     description, and initial message.
 * 4.  Set default values for loop control: infinite loop, no timing measurement,
 *     and no real-time priority.
 */
PROCESSINFO *processinfo_setup(
    char         *
    pinfoname, // short name for the processinfo instance, avoid spaces, name should be human-readable
    const char *descriptionstring,
    const char *msgstring,
    const char *functionname,
    const char *filename,
    int         linenumber)
{
    DEBUG_TRACE_FSTART();

    //fflush(stdout);

    static PROCESSINFO *processinfo = NULL;
    static int processinfoActive = 0;

    DEBUG_TRACEPOINT(" ");

    DEBUG_TRACEPOINT(" ");
    if(processinfoActive == 0)
    {
        DEBUG_TRACEPOINT(" ");

        char pinfoname0[STRINGMAXLEN_PROCESSINFO_NAME];
        {
            int slen = snprintf(pinfoname0,
                                STRINGMAXLEN_PROCESSINFO_NAME,
                                "%s",
                                pinfoname);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort();
            }
            if(slen >= STRINGMAXLEN_PROCESSINFO_NAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort();
            }
        }

        DEBUG_TRACEPOINT(" ");

        processinfo = processinfo_shm_create(pinfoname0, 0);

        DEBUG_TRACEPOINT(" ");
    }

    DEBUG_TRACEPOINT(" ");

    processinfo->loopstat = 0; // loop initialization
    strcpy(processinfo->source_FUNCTION, functionname);
    strcpy(processinfo->source_FILE, filename);
    processinfo->source_LINE = linenumber;
    strcpy(processinfo->description, descriptionstring);
    processinfo_WriteMessage(processinfo, msgstring);
    processinfoActive = 1;

    processinfo->loopcntMax = -1; // infinite loop

    processinfo->MeasureTiming = 0;  // default: do not measure timing
    processinfo->RT_priority   = -1; // default: do not assign RT priority

    DEBUG_TRACEPOINT(" ");

    DEBUG_TRACE_FEXIT();

    return processinfo;
}

/**
 * @brief Log an error and perform a clean exit.
 *
 * This function is used to handle fatal loop errors. it sets the loop
 * status to ERROR, writes the error message to SHM, and then calls
 * `processinfo_cleanExit` to detach.
 */
errno_t processinfo_error(PROCESSINFO *processinfo, char *errmsgstring)
{
    processinfo->loopstat = 4; // ERROR
    processinfo_WriteMessage(processinfo, errmsgstring);
    processinfo_cleanExit(processinfo);
    return RETURN_SUCCESS;
}

/**
 * @brief Finalize process initialization and enter active loop state.
 *
 * This function should be called just before entering the main processing loop.
 * It resets the loop counter, sets status to ACTIVE, and attempts to set
 * the process's real-time scheduler priority if configured.
 */
errno_t processinfo_loopstart(PROCESSINFO *processinfo)
{
    processinfo->loopcnt  = 0;
    processinfo->loopstat = 1;

    if(processinfo->RT_priority > -1)
    {
        struct sched_param schedpar;
        schedpar.sched_priority = processinfo->RT_priority;

        if (sched_setscheduler(0, SCHED_FIFO, &schedpar) != 0) {
            // perror("sched_setscheduler");
        }
    }

    return RETURN_SUCCESS;
}
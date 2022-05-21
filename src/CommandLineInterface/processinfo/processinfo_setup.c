#include <sys/stat.h>

#include "CLIcore.h"
#include <processtools.h>


// High level processinfo function

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

    static PROCESSINFO *processinfo;
    // Only one instance of processinfo created by process
    // subsequent calls to this function will re-use the same processinfo structure

    DEBUG_TRACEPOINT(" ");

    DEBUG_TRACEPOINT(" ");
    if (data.processinfoActive == 0)
    {
        //        PROCESSINFO *processinfo;
        DEBUG_TRACEPOINT(" ");

        char pinfoname0[STRINGMAXLEN_PROCESSINFO_NAME];
        {
            int slen = snprintf(pinfoname0,
                                STRINGMAXLEN_PROCESSINFO_NAME,
                                "%s",
                                pinfoname);
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_PROCESSINFO_NAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        DEBUG_TRACEPOINT(" ");

        processinfo = processinfo_shm_create(pinfoname0, 0);

        DEBUG_TRACEPOINT(" ");

        processinfo_CatchSignals();
    }

    DEBUG_TRACEPOINT(" ");

    processinfo->loopstat = 0; // loop initialization
    strcpy(processinfo->source_FUNCTION, functionname);
    strcpy(processinfo->source_FILE, filename);
    processinfo->source_LINE = linenumber;
    strcpy(processinfo->description, descriptionstring);
    processinfo_WriteMessage(processinfo, msgstring);
    data.processinfoActive = 1;

    processinfo->loopcntMax = -1; // infinite loop

    processinfo->MeasureTiming = 0;  // default: do not measure timing
    processinfo->RT_priority   = -1; // default: do not assign RT priority

    DEBUG_TRACEPOINT(" ");

    DEBUG_TRACE_FEXIT();

    return processinfo;
}

// report error
// should be followed by return(EXIT_FAILURE) call
//
errno_t processinfo_error(PROCESSINFO *processinfo, char *errmsgstring)
{
    processinfo->loopstat = 4; // ERROR
    processinfo_WriteMessage(processinfo, errmsgstring);
    processinfo_cleanExit(processinfo);
    return RETURN_SUCCESS;
}

errno_t processinfo_loopstart(PROCESSINFO *processinfo)
{
    processinfo->loopcnt  = 0;
    processinfo->loopstat = 1;

    if (processinfo->RT_priority > -1)
    {
        struct sched_param schedpar;
        // ===========================
        // Set realtime priority
        // ===========================
        schedpar.sched_priority = processinfo->RT_priority;

        if (seteuid(data.euid) != 0) //This goes up to maximum privileges
        {
            PRINT_ERROR("seteuid error");
        }
        sched_setscheduler(
            0,
            SCHED_FIFO,
            &schedpar);              //other option is SCHED_RR, might be faster
        if (seteuid(data.ruid) != 0) //Go back to normal privileges
        {
            PRINT_ERROR("seteuid error");
        }
    }

    return RETURN_SUCCESS;
}

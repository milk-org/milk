/**
 * @file    fps_add_RTsetting_entries.c
 * @brief   Add parameters to FPS for real-time process settings
 */

#include "CommandLineInterface/CLIcore.h"

#include "fps_GetParamIndex.h"

/** @brief Add parameters to FPS for real-time process settings
 *
 * Adds standard set of parameters for integration with process info
 * and other milk conventions.
 *
 * This function is typically called within the FPCONF fps creation step.
 *
 */
errno_t fps_add_processinfo_entries(FUNCTION_PARAMETER_STRUCT *fps)
{
    DEBUG_TRACE_FSTART();

    uint64_t FPFLAG;
    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;
    FPFLAG &= ~FPFLAG_WRITERUN;

    // run time string
    function_parameter_add_entry(fps,
                                 ".conf.timestring",
                                 "runstart time string",
                                 FPTYPE_STRING,
                                 FPFLAG,
                                 "undef",
                                 NULL);

    // custom label
    function_parameter_add_entry(fps,
                                 ".conf.label",
                                 "custom label",
                                 FPTYPE_STRING,
                                 FPFLAG,
                                 "",
                                 NULL);

    // output directory where results are saved
    //
    //	char outdir[FPS_DIR_STRLENMAX];
    //	snprintf(outdir, FPS_DIR_STRLENMAX, "fps.%s", fps->md->name);
    function_parameter_add_entry(fps,
                                 ".conf.datadir",
                                 "data directory",
                                 FPTYPE_DIRNAME,
                                 FPFLAG,
                                 fps->md->datadir,
                                 NULL);

    // input directory, FPS configuration files ready by FPSsync operation
    //
    //	char confdir[FPS_DIR_STRLENMAX];
    //	snprintf(confdir, FPS_DIR_STRLENMAX, "fpsconfdir-%s", fps->md->name);
    function_parameter_add_entry(fps,
                                 ".conf.confdir",
                                 "conf directory",
                                 FPTYPE_DIRNAME,
                                 FPFLAG,
                                 fps->md->confdir,
                                 NULL);

    // Where results are archived
    //
    function_parameter_add_entry(fps,
                                 ".conf.archivedir",
                                 "archive directory",
                                 FPTYPE_DIRNAME,
                                 FPFLAG,
                                 NULL,
                                 NULL);

    // value = -1 indicates no RT priority
    long RTprio_default[4] = {fps->cmdset.RT_priority, -1, 49, 20};
    function_parameter_add_entry(fps,
                                 ".procinfo.RTprio",
                                 "RTprio",
                                 FPTYPE_INT64,
                                 FPFLAG,
                                 &RTprio_default,
                                 NULL);

    // cset
    function_parameter_add_entry(fps,
                                 ".procinfo.cset",
                                 "CPUs set",
                                 FPTYPE_STRING,
                                 FPFLAG,
                                 "system",
                                 NULL);

    // taskset
    function_parameter_add_entry(fps,
                                 ".procinfo.taskset",
                                 "CPUs mask",
                                 FPTYPE_STRING,
                                 FPFLAG,
                                 "1-127",
                                 NULL);

    // value = 0 indicates process will adjust to available nb cores
    long maxNBthread_default[4] = {1, 0, 50, 1};
    function_parameter_add_entry(fps,
                                 ".procinfo.NBthread",
                                 "max NB threads",
                                 FPTYPE_INT64,
                                 FPFLAG,
                                 &maxNBthread_default,
                                 NULL);

    // PROCESSINFO
    long fp_pinfoenabled = 0;
    function_parameter_add_entry(fps,
                                 ".procinfo.enabled",
                                 "procinfo mode",
                                 FPTYPE_ONOFF,
                                 FPFLAG,
                                 NULL,
                                 &fp_pinfoenabled);
    fps->parray[fp_pinfoenabled].fpflag |= FPFLAG_ONOFF;

    // no max limit
    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT;
    FPFLAG &= ~FPFLAG_WRITERUN;
    // value = -1 indicates infinite loop
    long loopcntMax_default[4] = {fps->cmdset.procinfo_loopcntMax,
                                  -1,
                                  5000000,
                                  1};
    function_parameter_add_entry(fps,
                                 ".procinfo.loopcntMax",
                                 "max loop cnt",
                                 FPTYPE_INT64,
                                 FPFLAG,
                                 &loopcntMax_default,
                                 NULL);

    long triggermode_default[4] = {fps->cmdset.triggermode, -1, 10, 0};
    function_parameter_add_entry(fps,
                                 ".procinfo.triggermode",
                                 "trigger mode",
                                 FPTYPE_INT64,
                                 FPFLAG,
                                 &triggermode_default,
                                 NULL);

    function_parameter_add_entry(fps,
                                 ".procinfo.triggersname",
                                 "trigger stream name",
                                 FPTYPE_STREAMNAME,
                                 FPFLAG,
                                 fps->cmdset.triggerstreamname,
                                 NULL);

    // -1 : auto (recommended)
    long semindexrequested_default[4] = {fps->cmdset.semindexrequested,
                                         -1,
                                         10,
                                         0};
    function_parameter_add_entry(fps,
                                 ".procinfo.semindexrequested",
                                 "trigger requested semaphore index",
                                 FPTYPE_INT64,
                                 FPFLAG,
                                 &semindexrequested_default,
                                 NULL);

    struct timespec triggerdelay_default[2] = {fps->cmdset.triggerdelay,
                                               {1, 0}};
    function_parameter_add_entry(fps,
                                 ".procinfo.triggerdelay",
                                 "trigger delay",
                                 FPTYPE_TIMESPEC,
                                 FPFLAG,
                                 &triggerdelay_default,
                                 NULL);

    struct timespec triggertimeout_default[2] = {fps->cmdset.triggertimeout,
                                                 {1, 0}};
    function_parameter_add_entry(fps,
                                 ".procinfo.triggertimeout",
                                 "trigger timeout",
                                 FPTYPE_TIMESPEC,
                                 FPFLAG,
                                 &triggertimeout_default,
                                 NULL);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t fps_to_processinfo(FUNCTION_PARAMETER_STRUCT *fps,
                           PROCESSINFO               *procinfo)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("Checking fps pointer");
    if (fps == NULL)
    {
        PRINT_ERROR("Null pointer - cannot proceed\n");
        abort();
    }

    DEBUG_TRACEPOINT("set RT_priority if applicable");
    {
        long pindex = functionparameter_GetParamIndex(fps, ".procinfo.RTprio");
        if (pindex > -1)
        {
            long RTprio =
                functionparameter_GetParamValue_INT64(fps, ".procinfo.RTprio");
            procinfo->RT_priority = RTprio;
        }
    }

    DEBUG_TRACEPOINT("set loopcntMax if applicable");
    {
        long pindex =
            functionparameter_GetParamIndex(fps, ".procinfo.loopcntMax");
        if (pindex > -1)
        {
            long loopcntMax =
                functionparameter_GetParamValue_INT64(fps,
                                                      ".procinfo.loopcntMax");
            procinfo->loopcntMax = loopcntMax;
        }
    }

    DEBUG_TRACEPOINT("set triggermode if applicable");
    {

        long pindex =
            functionparameter_GetParamIndex(fps, ".procinfo.triggermode");
        if (pindex > -1)
        {
            long triggermode =
                functionparameter_GetParamValue_INT64(fps,
                                                      ".procinfo.triggermode");
            procinfo->triggermode = triggermode;
            printf(">>>>>>>>>>>>> SET TRIGGERMODE = %ld\n", triggermode);
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

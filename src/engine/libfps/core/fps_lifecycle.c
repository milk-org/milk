// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_lifecycle.c
 * @brief   Generic FPS lifecycle functions
 *
 * Implements init/conf/run/stop functions that handle
 * both local (_prefix) and shared-memory FPS modes.
 * Extracted from POC sections 2.6-2.9.
 */


#ifndef FPS_STANDALONE
#    include "CLIcore.h"
#else
#    include "libmilkdata/milkdata.h"
#endif
#include "fps.h"
#include "fps_apply_process_settings.h"
#include "fps_checkparameter.h"


#include "fps_globals.h"


/**
 * @brief Auto-populate processinfo from trigger stream.
 *
 * Scans bindings for the first FPFLAG_TRIGGER_STREAM
 * parameter. If found and processinfo entries exist,
 * sets:
 *   .procinfo.triggersname  -> stream default value
 *   .procinfo.triggermode   -> SEMAPHORE (3)
 *   .procinfo.loopcntMax    -> -1 (infinite)
 *   .procinfo.enabled       -> ON
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter bindings array
 * @param nb_b      Number of bindings
 */
static void fps_autopopulate_trigger_stream(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    /* Find the first TRIGGER_STREAM binding */
    const char *trigger_name = NULL;
    for (int ii = 0; ii < nb_b; ii++)
    {
        if ((bindings[ii].fpflag & FPFLAG_TRIGGER_STREAM) &&
            (bindings[ii].type == FPTYPE_STREAMNAME))
        {
            /*
             * ptr holds the default stream name
             * buffer (char[]) for string types.
             */
            trigger_name = (const char *) bindings[ii].ptr;
            break;
        }
    }

    if (trigger_name == NULL)
    {
        return;
    }

    if (trigger_name[0] == '\0')
    {
        return;
    }

    /* Only set if .procinfo.triggersname exists */
    int pidx = functionparameter_GetParamIndex(fps, ".procinfo.triggersname");
    if (pidx < 0)
    {
        return;
    }

    functionparameter_SetParamValue_STRING(fps, ".procinfo.triggersname", trigger_name);

    functionparameter_SetParamValue_INT64(fps, ".procinfo.triggermode",
                                          PROCESSINFO_TRIGGERMODE_SEMAPHORE);

    functionparameter_SetParamValue_INT64(fps, ".procinfo.loopcntMax", -1);

    functionparameter_SetParamValue_ONOFF(fps, ".procinfo.enabled", 1);
}


/**
 * @brief Initialize an FPS and register its parameter bindings.
 *
 * For local FPS names (prefixed with '_'), creates an in-process
 * FPS.  For shared-memory FPS names, creates an FPS in /dev/shm
 * with processinfo entries when requested.  In both cases, the
 * bindings array defines all parameters in the FPS.
 *
 * @param fps_name   FPS name (prefix '_' for local mode)
 * @param app_info   Application info (cmdkey, description)
 * @param bindings   Array of parameter bindings (C var <-> FPS)
 * @param nb_b       Number of bindings
 * @param procinfo   If nonzero, add processinfo entries
 * @return 0 on success, -1 on allocation failure
 */
int fps_generic_init(const char      *fps_name,
                     FPS_APP_INFO    *app_info,
                     FPS_CLI_BINDING *bindings,
                     int              nb_b,
                     int              procinfo)
{
    if (fps_name[0] == '_')
    {
        /* Local mode: in-process memory only */
        FPS *lfps = fps_local_get_or_create(fps_name, FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (lfps == NULL)
        {
            return -1;
        }

        fps_init_from_bindings(lfps, app_info->cmdkey, app_info->description, bindings, nb_b);

        /* Count active params */
        {
            int  cnt   = 0;
            long nbmax = lfps->md->NBparamMAX;
            for (int pi = 0; pi < nbmax; pi++)
            {
                if (lfps->parray[pi].fpflag & FPFLAG_ACTIVE)
                {
                    cnt++;
                }
            }
            lfps->NBparamActive = cnt;
        }
        return 0;
    }

    /* Shared-memory mode */
    FPS fps;
    FPS_INIT_STD_PREAMBLE(fps, fps_name, "", app_info->description, app_info->description);

#ifndef FPS_STANDALONE
    if (procinfo || (data.cmd[data.cmdindex].cmdsettings.flags & CLICMDFLAG_PROCINFO))
#else
    if (procinfo)
#endif
    {
        fps.cmdset.flags |= CLICMDFLAG_PROCINFO;
        fps_add_processinfo_entries(&fps);
    }

    fps_init_from_bindings(&fps, app_info->cmdkey, app_info->description, bindings, nb_b);

    /* Auto-populate processinfo from trigger stream */
    fps_autopopulate_trigger_stream(&fps, bindings, nb_b);

    fps_disconnect(&fps);
    return 0;
}


/**
 * @brief Check if bindings have a trigger stream.
 *
 * @param bindings  Parameter bindings array
 * @param nb_b      Number of bindings
 * @return          1 if trigger stream found
 */
int fps_check_has_trigger_binding(FPS_CLI_BINDING *bindings, int nb_b)
{
    for (int ii = 0; ii < nb_b; ii++)
    {
        if ((bindings[ii].fpflag & FPFLAG_TRIGGER_STREAM) &&
            (bindings[ii].type == FPTYPE_STREAMNAME))
        {
            return 1;
        }
    }
    return 0;
}


/**
 * @brief Force semaphore-triggered loop mode for -loops flag.
 *
 * Finds the trigger stream from the bindings (local variable
 * first, then FPS shared memory, then .procinfo.triggersname
 * as fallback).  Configures loopcntMax=-1, enabled=ON, and
 * triggermode=SEMAPHORE.  Falls back to DELAY if no trigger
 * stream is found.
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter bindings array
 * @param nb_b      Number of bindings
 */
void fps_loop_override_trigger(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    /*
     * Find trigger stream from bindings.
     * First try the local C variable (ptr).
     * If empty (CLI sync hasn't run yet),
     * read from FPS shared memory instead.
     */
    const char *trigger_name                             = NULL;
    char        trigger_kw[FUNCTION_PARAMETER_STRMAXLEN] = "";
    char        current_ts[FUNCTION_PARAMETER_STRMAXLEN] = "";

    for (int ii = 0; ii < nb_b; ii++)
    {
        if ((bindings[ii].fpflag & FPFLAG_TRIGGER_STREAM) &&
            (bindings[ii].type == FPTYPE_STREAMNAME))
        {
            /* Try local variable first */
            const char *local = (const char *) bindings[ii].ptr;
            if (local != NULL && local[0] != '\0')
            {
                trigger_name = local;
            }
            else
            {
                /*
                 * Local var empty -- CLI sync
                 * hasn't populated it yet.
                 * Read from FPS shared memory.
                 */
                strncpy(trigger_kw, bindings[ii].fpskeyword, sizeof(trigger_kw) - 1);
            }
            break;
        }
    }

    /*
     * Read trigger name from FPS if local var
     * was empty but we found the binding keyword.
     */
    if (trigger_name == NULL && trigger_kw[0] != '\0')
    {
        long pidx = functionparameter_GetParamIndex(fps, trigger_kw);
        if (pidx >= 0)
        {
            strncpy(current_ts, functionparameter_GetParamPtr_STRING(fps, trigger_kw),
                    sizeof(current_ts) - 1);
            if (current_ts[0] != '\0')
            {
                trigger_name = current_ts;
            }
        }
    }

    /*
     * If still no trigger stream, try
     * .procinfo.triggersname as last resort.
     */
    if (trigger_name == NULL || trigger_name[0] == '\0')
    {
        long pidx = functionparameter_GetParamIndex(fps, ".procinfo.triggersname");
        if (pidx >= 0)
        {
            strncpy(current_ts, functionparameter_GetParamPtr_STRING(fps, ".procinfo.triggersname"),
                    sizeof(current_ts) - 1);
            if (current_ts[0] != '\0')
            {
                trigger_name = current_ts;
            }
        }
    }

    printf("\033[33m-loops\033[0m"
           " Stream semaphore trigger\n");

    /* Force loop count and enable */
    functionparameter_SetParamValue_INT64(fps, ".procinfo.loopcntMax", -1);
    printf("  .procinfo.loopcntMax  = -1"
           " (infinite)\n");

    functionparameter_SetParamValue_ONOFF(fps, ".procinfo.enabled", 1);
    printf("  .procinfo.enabled     = ON\n");

    if (trigger_name != NULL && trigger_name[0] != '\0')
    {
        functionparameter_SetParamValue_STRING(fps, ".procinfo.triggersname", trigger_name);
        printf("  .procinfo.triggersname"
               " = %s\n",
               trigger_name);

        functionparameter_SetParamValue_INT64(fps, ".procinfo.triggermode",
                                              PROCESSINFO_TRIGGERMODE_SEMAPHORE);
        printf("  .procinfo.triggermode "
               " = %d (SEMAPHORE)\n",
               PROCESSINFO_TRIGGERMODE_SEMAPHORE);
    }
    else
    {
        PRINT_WARNING("[-loops] No trigger stream found -- semaphore trigger not configured.");
        PRINT_WARNING("  Loop will use delay mode. To fix, flag a stream parameter with "
                      "FPFLAG_TRIGGER_STREAM.");

        functionparameter_SetParamValue_INT64(fps, ".procinfo.triggermode",
                                              PROCESSINFO_TRIGGERMODE_DELAY);
        printf("  .procinfo.triggermode "
               " = %d (DELAY)\n",
               PROCESSINFO_TRIGGERMODE_DELAY);
    }
}


/**
 * @brief Force delay-loop settings for -loopd mode.
 *
 * Sets triggermode=DELAY, loopcntMax=-1, enabled=ON,
 * and triggerdelay to the specified seconds value.
 *
 * @param fps       Connected FPS
 * @param delay_sec Delay between iterations (seconds)
 */
void fps_loop_override_delay(FPS *fps, double delay_sec)
{
    printf("\033[33m-loopd\033[0m"
           " Delay loop (%.6f sec)\n",
           delay_sec);

    functionparameter_SetParamValue_INT64(fps, ".procinfo.loopcntMax", -1);
    printf("  .procinfo.loopcntMax  = -1"
           " (infinite)\n");

    functionparameter_SetParamValue_ONOFF(fps, ".procinfo.enabled", 1);
    printf("  .procinfo.enabled     = ON\n");

    functionparameter_SetParamValue_INT64(fps, ".procinfo.triggermode",
                                          PROCESSINFO_TRIGGERMODE_DELAY);
    printf("  .procinfo.triggermode "
           " = %d (DELAY)\n",
           PROCESSINFO_TRIGGERMODE_DELAY);

    functionparameter_SetParamValue_TIMESPEC(fps, ".procinfo.triggerdelay", (float) delay_sec);
    printf("  .procinfo.triggerdelay"
           " = %.6f sec\n",
           delay_sec);
}


/**
 * @brief Start the FPS configuration monitoring loop.
 *
 * For local FPS names, prints a message and returns.
 * For shared-memory FPS names, enters FPS_CONF_STD_BODY
 * which monitors parameter changes.  Optionally calls
 * confcheck_fn at each iteration for custom validation.
 *
 * @param fps_name     FPS name
 * @param loop         If nonzero, run continuously
 * @param confcheck_fn Optional callback for custom checks
 * @return 0 on success, nonzero on error
 */
int fps_generic_conf_cb(const char *fps_name, int loop, errno_t (*confcheck_fn)(void))
{
    if (fps_name[0] == '_')
    {
        printf("Local FPS '%s' -- "
               "monitoring loop skipped.\n",
               fps_name);
        return 0;
    }
    FPS_CONF_STD_BODY(fps_name, loop, {}, {
        if (confcheck_fn != NULL)
        {
#ifndef FPS_STANDALONE
            dcfpsptr = &fps;
#else
            milk_data.fpsptr = &fps;
#endif
            confcheck_fn();
        }
    });
    return 0;
}


/**
 * @brief Start the FPS configuration loop without a custom
 *        check callback (convenience wrapper).
 *
 * @param fps_name  FPS name
 * @param loop      If nonzero, run continuously
 * @return 0 on success
 */
int fps_generic_conf(const char *fps_name, int loop)
{
    return fps_generic_conf_cb(fps_name, loop, NULL);
}


/**
 * @brief Execute the FPS compute function.
 *
 * Connects to the FPS, applies CLI arguments via
 * fps_process_cli_and_sync(), then invokes the
 * compute function.  For local FPS names, creates
 * the FPS in-process if it doesn't exist yet.
 * For shared-memory FPS, performs a two-phase
 * connect (SIMPLE for CLI sync, then RUN for
 * stream loading).
 *
 * @param fps_name    FPS name
 * @param app_info    Application identity
 * @param farg        CLI argument definitions
 * @param bindings    Parameter bindings
 * @param nb_b        Number of bindings
 * @param compute_fn  Function generated by
 *                    INSERT_STD_PROCINFO_COMPUTEFUNC
 * @return 0 on success, 1 if FPS not found
 */
static void print_parameter_validation_errors(const FPS *fps)
{
    for (long ii = 0; ii < fps->md->msgcnt; ii++)
    {
        if (fps->md->msgcode[ii] & FPS_MSG_FLAG_ERROR)
        {
            long pindex = fps->md->msgpindex[ii];
            if (pindex >= 0 && pindex < fps->md->NBparamMAX)
            {
                fprintf(stderr, "  \033[31mParameter '%s' error: %s\033[0m\n",
                        fps->parray[pindex].keywordfull, fps->md->message[ii]);
            }
            else
            {
                fprintf(stderr, "  \033[31mError: %s\033[0m\n", fps->md->message[ii]);
            }
        }
    }
}


/**
 * @brief Execute the FPS compute function.
 *
 * Connects to the FPS, applies CLI arguments via
 * fps_process_cli_and_sync(), then invokes the
 * compute function.  For local FPS names, creates
 * the FPS in-process if it doesn't exist yet.
 * For shared-memory FPS, performs a two-phase
 * connect (SIMPLE for CLI sync, then RUN for
 * stream loading).
 *
 * @param fps_name    FPS name
 * @param app_info    Application identity
 * @param farg        CLI argument definitions
 * @param bindings    Parameter bindings
 * @param nb_b        Number of bindings
 * @param compute_fn  Function generated by
 *                    INSERT_STD_PROCINFO_COMPUTEFUNC
 * @return 0 on success, 1 if FPS not found
 */
int fps_generic_run(const char      *fps_name,
                    FPS_APP_INFO    *app_info,
                    CLICMDARGDEF    *farg,
                    FPS_CLI_BINDING *bindings,
                    int              nb_b,
                    fps_compute_fn   compute_fn)
{
    FPS fps;

    if (fps_name[0] == '_')
    {
        FPS *lfps = fps_local_get_or_create(fps_name, FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (lfps == NULL)
        {
            return -1;
        }
        if (lfps->NBparam == 0)
        {
            fps_generic_init(fps_name, app_info, bindings, nb_b, 0);
        }
        fps = *lfps;
        fps_process_cli_and_sync(&fps, farg, bindings, nb_b);

        /* Validate parameters and check the CHECKOK flag */
        functionparameter_CheckParametersAll(&fps);
        if (!(fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK))
        {
            PRINT_ERROR("FPS '%s' parameter validation failed (%lu errors). Cannot run.", fps_name,
                        (unsigned long) fps.md->conferrcnt);
            print_parameter_validation_errors(&fps);
            return 1;
        }
    }
    else
    {
        /* Phase 1: connect SIMPLE to apply CLI
         * args before streams are loaded. */
        if (fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1)
        {
            PRINT_ERROR("FPS '%s' not found. Run 'fpsinit' first.", fps_name);
            return 1;
        }
        fps_process_cli_and_sync(&fps, farg, bindings, nb_b);

        /* Validate parameters and check the CHECKOK flag */
        functionparameter_CheckParametersAll(&fps);
        if (!(fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK))
        {
            PRINT_ERROR("FPS '%s' parameter validation failed (%lu errors). Cannot run.", fps_name,
                        (unsigned long) fps.md->conferrcnt);
            print_parameter_validation_errors(&fps);
            fps_disconnect(&fps);
            return 1;
        }

        fps_disconnect(&fps);

        /* Phase 2: reconnect as RUN -- streams
         * now load with CLI-updated values. */
        FPS_RUN_STD_PREAMBLE(fps_name, fps, {});
    }

    fflush(stdout);

    /* Apply process-level settings (taskset,
     * cset, OMP_NUM_THREADS) from FPS params.
     * Must happen before compute_fn() so that
     * OpenMP picks up thread count and CPU
     * affinity is in effect for the loop. */
    fps_apply_process_settings(&fps);

    /*
     * The compute function (generated with
     * INSERT_STD_PROCINFO_COMPUTEFUNC_START/END)
     * manages its own processinfo loop and
     * loopcntMax termination internally.
     *
     * Setting dcfpsptr here allows the macro
     * to pick up all FPS-derived settings
     * (triggermode, loopcntMax, MeasureTiming...)
     * at its own processinfo_setup time.
     *
     * Do NOT wrap compute_fn() in a second
     * FPS_RUN_PROCESSINFO_LOOP: that would
     * multiply iterations by loopcntMax^2.
     */
    /*
     * Set FPS_name (the global that dcfpsname
     * copies from) so processinfo_setup inside
     * compute_fn() gets a valid process name.
     */
    strncpy(FPS_name, fps_name, STRINGMAXLEN_FPS_NAME - 1);
    FPS_name[STRINGMAXLEN_FPS_NAME - 1] = '\0';

    dcfpsptr = &fps;

    compute_fn();

    dcfpsptr = NULL;
    if (fps_name[0] != '_')
    {
        fps_disconnect(&fps);
    }

    printf("ran as PID %ld\n", (long) getpid());

    return 0;
}


/**
 * @brief Generically stops a running FPS process by name.
 */
int fps_generic_runstop(const char *fps_name)
{
    FPS fps;

    printf("Stopping run process for '%s'\n", fps_name);

    if (fps_name[0] == '_')
    {
        printf("Local FPS '%s' -- stop signal "
               "ignored (lifetime limited to "
               "process).\n",
               fps_name);
        return 0;
    }

    if (fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        PRINT_ERROR("FPS '%s' not found.", fps_name);
        return 1;
    }

    /*
     * Do NOT call functionparameter_RUNstop() here.
     * That function dispatches a "runstop" command
     * to the tmux :ctrl window via send-keys, but
     * fps_generic_runstop() is itself invoked FROM
     * the :ctrl window -- calling RUNstop() creates
     * an infinite tmux command loop.
     *
     * Instead, perform the stop actions directly:
     * 1. Send C-c to the :run window (interrupt)
     * 2. Clear the CMDRUN status flag
     * 3. Signal the GUI to update
     */
    EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run C-c"
                                   " 2>/dev/null",
                                   fps.md->name);

    fps.md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

    fps_disconnect(&fps);
    functionparameter_FPS_processinfo_signal(fps_name, 3);
    return 0;
}


/**
 * @brief Generically stops the configuration phase of an FPS process.
 */
int fps_generic_confstop(const char *fps_name)
{
    FPS fps;

    printf("Stopping configuration process "
           "for '%s'\n",
           fps_name);

    if (fps_name[0] == '_')
    {
        printf("Local FPS '%s' -- stop signal "
               "ignored (lifetime limited to "
               "process).\n",
               fps_name);
        return 0;
    }

    if (fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        PRINT_ERROR("FPS '%s' not found.", fps_name);
        return 1;
    }
    functionparameter_CONFstop(&fps);
    fps_disconnect(&fps);
    return 0;
}

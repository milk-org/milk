/**
 * @file    fps_lifecycle.c
 * @brief   Generic FPS lifecycle functions
 *
 * Implements init/conf/run/stop functions that handle
 * both local (_prefix) and shared-memory FPS modes.
 * Extracted from POC sections 2.6-2.9.
 */

#include <stdio.h>


#ifndef FPS_STANDALONE
#include "CLIcore.h"
#else
#include "libmilkdata/milkdata.h"
#endif
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_init.h"
#include "fps_cli_sync.h"
#include "fps_connect.h"
#include "fps_disconnect.h"

#include "fps_lifecycle.h"
#include "fps_local_store.h"
#include "fps_processinfo_entries.h"

#include "fps_globals.h"

#include "fps_RUNstop.h"

#include "fps_CONFstop.h"
#include "fps_processinfo.h"


int fps_generic_init(
    const char      *fps_name,
    FPS_APP_INFO    *app_info,
    FPS_CLI_BINDING *bindings,
    int              nb_b,
    int              procinfo
)
{
    if (fps_name[0] == '_') {
        /* Local mode: in-process memory only */
        FUNCTION_PARAMETER_STRUCT *lfps =
            fps_local_get_or_create(
                fps_name,
                FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (lfps == NULL) {
            return -1;
        }

        fps_init_from_bindings(
            lfps,
            app_info->cmdkey,
            app_info->description,
            bindings,
            nb_b);

        /* Count active params */
        {
            int cnt = 0;
            long nbmax = lfps->md->NBparamMAX;
            for (int pi = 0; pi < nbmax; pi++) {
                if (lfps->parray[pi].fpflag
                    & FPFLAG_ACTIVE)
                {
                    cnt++;
                }
            }
            lfps->NBparamActive = cnt;
        }
        return 0;
    }

    /* Shared-memory mode */
    FUNCTION_PARAMETER_STRUCT fps;
    FPS_INIT_STD_PREAMBLE(
        fps, fps_name, "", app_info->description,
        app_info->description);

#ifndef FPS_STANDALONE
    if (procinfo ||
        (data.cmd[data.cmdindex].cmdsettings.flags
         & CLICMDFLAG_PROCINFO))
#else
    if (procinfo)
#endif
    {
        fps.cmdset.flags |= CLICMDFLAG_PROCINFO;
        fps_add_processinfo_entries(&fps);
    }

    fps_init_from_bindings(
        &fps,
        app_info->cmdkey,
        app_info->description,
        bindings,
        nb_b);

    function_parameter_struct_disconnect(&fps);
    return 0;
}


int fps_generic_conf_cb(
    const char *fps_name,
    int         loop,
    errno_t   (*confcheck_fn)(void)
)
{
    if (fps_name[0] == '_') {
        printf("Local FPS '%s' — "
               "monitoring loop skipped.\n",
               fps_name);
        return 0;
    }
    FPS_CONF_STD_BODY(
        fps_name, loop,
        {},
        {
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


int fps_generic_conf(
    const char *fps_name,
    int         loop
)
{
    return fps_generic_conf_cb(
        fps_name, loop, NULL);
}


int fps_generic_run(
    const char      *fps_name,
    FPS_APP_INFO    *app_info,
    CLICMDARGDEF    *farg,
    FPS_CLI_BINDING *bindings,
    int              nb_b,
    fps_compute_fn   compute_fn
)
{
    FUNCTION_PARAMETER_STRUCT fps;
    long loopcnt = 0;

    if (fps_name[0] == '_') {
        FUNCTION_PARAMETER_STRUCT *lfps =
            fps_local_get_or_create(
                fps_name,
                FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (lfps == NULL) {
            return -1;
        }
        if (lfps->NBparam == 0) {
            fps_generic_init(
                fps_name, app_info,
                bindings, nb_b, 0);
        }
        fps = *lfps;
        fps_process_cli_and_sync(
            &fps, farg, bindings, nb_b);
    }
    else {
        /* Phase 1: connect SIMPLE to apply CLI
         * args before streams are loaded. */
        if (function_parameter_struct_connect(
                fps_name, &fps,
                FPSCONNECT_SIMPLE) == -1)
        {
            fprintf(stderr,
                    "Error: FPS '%s' not found."
                    " Run 'fpsinit' first.\n",
                    fps_name);
            return 1;
        }
        fps_process_cli_and_sync(
            &fps, farg, bindings, nb_b);
        function_parameter_struct_disconnect(
            &fps);

        /* Phase 2: reconnect as RUN — streams
         * now load with CLI-updated values. */
        FPS_RUN_STD_PREAMBLE(
            fps_name, fps, {});
    }

    fflush(stdout);

    /*
     * The compute function (generated with
     * INSERT_STD_PROCINFO_COMPUTEFUNC_START/END)
     * manages its own processinfo loop and
     * loopcntMax termination internally.
     *
     * Setting dcfpsptr here allows the macro
     * to pick up all FPS-derived settings
     * (triggermode, loopcntMax, MeasureTiming…)
     * at its own processinfo_setup time.
     *
     * Do NOT wrap compute_fn() in a second
     * FPS_RUN_PROCESSINFO_LOOP: that would
     * multiply iterations by loopcntMax².
     */
    /*
     * Set FPS_name (the global that dcfpsname
     * copies from) so processinfo_setup inside
     * compute_fn() gets a valid process name.
     */
    strncpy(FPS_name, fps_name,
            STRINGMAXLEN_FPS_NAME - 1);
    FPS_name[STRINGMAXLEN_FPS_NAME - 1] = '\0';

    dcfpsptr = &fps;

    compute_fn();
    loopcnt = 1; /* reported by compute_fn's procinfo */

    dcfpsptr = NULL;
    if (fps_name[0] != '_') {
        function_parameter_struct_disconnect(
            &fps);
    }

    printf("ran as PID %ld for %ld step%s\n",
           (long) getpid(),
           loopcnt,
           (loopcnt == 1) ? "" : "s");

    return 0;
}


int fps_generic_runstop(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;

    printf("Stopping run process for '%s'\n",
           fps_name);

    if (fps_name[0] == '_') {
        printf("Local FPS '%s' — stop signal "
               "ignored (lifetime limited to "
               "process).\n",
               fps_name);
        return 0;
    }

    if (function_parameter_struct_connect(
            fps_name, &fps,
            FPSCONNECT_SIMPLE) == -1)
    {
        fprintf(stderr,
                "Error: FPS '%s' not found.\n",
                fps_name);
        return 1;
    }

    /*
     * Do NOT call functionparameter_RUNstop() here.
     * That function dispatches a "runstop" command
     * to the tmux :ctrl window via send-keys, but
     * fps_generic_runstop() is itself invoked FROM
     * the :ctrl window — calling RUNstop() creates
     * an infinite tmux command loop.
     *
     * Instead, perform the stop actions directly:
     * 1. Send C-c to the :run window (interrupt)
     * 2. Clear the CMDRUN status flag
     * 3. Signal the GUI to update
     */
    EXECUTE_SYSTEM_COMMAND(
        "tmux send-keys -t %s:run C-c"
        " 2>/dev/null",
        fps.md->name);

    fps.md->status &=
        ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps.md->signal |=
        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

    function_parameter_struct_disconnect(&fps);
    functionparameter_FPS_processinfo_signal(
        fps_name, 3);
    return 0;
}


int fps_generic_confstop(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;

    printf("Stopping configuration process "
           "for '%s'\n",
           fps_name);

    if (fps_name[0] == '_') {
        printf("Local FPS '%s' — stop signal "
               "ignored (lifetime limited to "
               "process).\n",
               fps_name);
        return 0;
    }

    if (function_parameter_struct_connect(
            fps_name, &fps,
            FPSCONNECT_SIMPLE) == -1)
    {
        fprintf(stderr,
                "Error: FPS '%s' not found.\n",
                fps_name);
        return 1;
    }
    functionparameter_CONFstop(&fps);
    function_parameter_struct_disconnect(&fps);
    return 0;
}

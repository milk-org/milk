// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_fps_function.c
 * @brief   Generic CLIfunction for FPS modules
 *
 * Provides a generic CLIfunction that handles the full milk CLI
 * lifecycle for any FPS-based module.
 */


#include "CLIcore.h"
#include "CLIcore_fps_function.h"
#include "CLIcore_fps_getFPSargs.h"
#include "CLIcore_fps_query.h"

char fps_last_used_name[200]   = "";
char fps_last_used_cmdkey[200] = "";


/**
 * @brief Sync CLI argdata into FPS and copy FPS values back into module variables.
 */
static void fps_process_clidata_and_sync(FPS             *fps,
                                         CLICMDARGDEF    *farg,
                                         FPS_CLI_BINDING *bindings,
                                         int              nb_b)
{
    CLIargs_to_FPSparams_setval(farg, nb_b, fps);
    fps_to_modulevars_resync_bindings(fps, bindings, nb_b);
}


/**
 * @brief Generic CLIfunction for FPS-based modules.
 *
 * Implements the full milk CLI lifecycle for any FPS module:
 * parses the FPS name (local vs shared), connects or creates
 * the FPS, syncs CLI arguments into FPS parameters, sets up
 * processinfo if requested, and calls the compute function.
 *
 * Handles special cases: fpsinit, "?" query, ignore codes.
 * Local FPS names (prefixed '_') operate in-process; shared
 * FPS names use shared memory.
 *
 * @param app_info    FPS application identity
 * @param farg        CLI argument definitions
 * @param cmdata      CLI command metadata
 * @param bindings    Parameter bindings (C var <-> FPS)
 * @param nb_b        Number of bindings
 * @param compute_fn  Compute function to invoke
 * @return RETURN_SUCCESS on completion
 */
errno_t fps_generic_CLIfunction(FPS_APP_INFO    *app_info,
                                CLICMDARGDEF    *farg,
                                CLICMDDATA      *cmdata,
                                FPS_CLI_BINDING *bindings,
                                int              nb_b,
                                fps_compute_fn   compute_fn)
{
    FPS fps;
    int procinfo = (data.cmd[data.cmdindex].cmdsettings.flags & CLICMDFLAG_PROCINFO) != 0;

    /*
     * Default FPS is local (underscore prefix).
     * User can override with :fpsname syntax.
     */
    char fpsname_with_session[200];
    snprintf(fpsname_with_session, sizeof(fpsname_with_session), "_%s", app_info->fps_name);

    function_parameter_getFPSargs_from_CLIfunc(fpsname_with_session);

    if (dcfpscode == FPSCMDCODE_IGNORE)
    {
        return RETURN_SUCCESS;
    }

    /* Handle "?" query */
    if (data.cmdNBarg >= 2 && strcmp(data.cmdargtoken[1].val.string, "?") == 0)
    {
        fps_print_query_info(app_info, bindings, nb_b);
        return RETURN_SUCCESS;
    }

    /* Initialization action */
    if (dcfpscode == FPSCMDCODE_FPSINIT || dcfpscode == FPSCMDCODE_FPSINITCREATE)
    {
        fps_generic_init(dcfpsname, app_info, bindings, nb_b, procinfo);
        return RETURN_SUCCESS;
    }

    if (dcfpscode == FPSCMDCODE_IGNORE)
    {
        return RETURN_SUCCESS;
    }

    /* Connect to existing FPS or use local */
    memset(&fps, 0, sizeof(FPS));
    fps.SMfd = -1;

    if (dcfpsname[0] == '_')
    {
        FPS *lfps = fps_local_get_or_create(dcfpsname, FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (lfps == NULL)
        {
            return RETURN_FAILURE;
        }
        if (lfps->NBparam == 0)
        {
            fps_generic_init(dcfpsname, app_info, bindings, nb_b, procinfo);
        }
        fps = *lfps;
    }
    else
    {
        if (fps_connect(dcfpsname, &fps, FPSCONNECT_SIMPLE) == -1)
        {
            fps_generic_init(dcfpsname, app_info, bindings, nb_b, procinfo);
            if (fps_connect(dcfpsname, &fps, FPSCONNECT_SIMPLE) == -1)
            {
                printf("Failed to connect to "
                       "FPS %s\n",
                       dcfpsname);
                return RETURN_SUCCESS;
            }
        }
    }

    /* Print FPS name and type in color (interactive TTY only) */
    int suppress_banner = (!isatty(STDOUT_FILENO) || dcquiet != 0);
    for (int a = 1; a < data.cmdNBarg && !suppress_banner; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "-q") == 0 || strcmp(tok, "--quiet") == 0 || strcmp(tok, "--json") == 0 ||
            strcmp(tok, "--porcelain") == 0)
        {
            suppress_banner = 1;
        }
    }
    if (dcfpsname[0] == '_')
    {
        if (!suppress_banner)
        {
            printf("\033[36mFPS \033[1m%s\033[22m"
                   " \033[33m[LOCAL]\033[0m\n",
                   dcfpsname);
        }
        fps_local_set_creator(dcfpsname, app_info->fps_name);
    }
    else
    {
        if (!suppress_banner)
        {
            printf("\033[36mFPS \033[1m%s\033[22m"
                   " \033[32m[SHARED]\033[0m\n",
                   dcfpsname);
        }
        fps_shared_record_usage(dcfpsname, app_info->fps_name);
    }

    /* Record last-used FPS for ? query */
    strncpy(fps_last_used_name, dcfpsname, sizeof(fps_last_used_name) - 1);
    fps_last_used_name[sizeof(fps_last_used_name) - 1] = '\0';
    strncpy(fps_last_used_cmdkey, app_info->fps_name, sizeof(fps_last_used_cmdkey) - 1);
    fps_last_used_cmdkey[sizeof(fps_last_used_cmdkey) - 1] = '\0';

    dcfpsptr       = &fps;
    dcfpsbindings  = bindings;
    dcfpsnbindings = nb_b;
    errno_t retval = CLI_checkarg_array(farg, cmdata->nbarg);

    if (retval == RETURN_SUCCESS || retval == RETURN_CLICHECKARGARRAY_FUNCPARAMSET)
    {
        fps_process_clidata_and_sync(&fps, farg, bindings, nb_b);

        if (dcfpsname[0] == '\0')
        {
            strncpy(dcfpsname, fpsname_with_session, STRINGMAXLEN_FPS_NAME - 1);
            dcfpsname[STRINGMAXLEN_FPS_NAME - 1] = '\0';
        }

        cmdata->cmdsettings = &data.cmd[data.cmdindex].cmdsettings;

        if (cmdata->cmdsettings->flags & CLICMDFLAG_PROCINFO)
        {
            memset(&fps.cmdset, 0, sizeof(fps.cmdset));
            fps.cmdset.procinfo_loopcntMax = cmdata->cmdsettings->procinfo_loopcntMax;
            fps.cmdset.triggermode         = cmdata->cmdsettings->triggermode;
            strncpy(fps.cmdset.triggerstreamname, cmdata->cmdsettings->triggerstreamname,
                    STRINGMAXLEN_IMAGE_NAME - 1);
            fps.cmdset.triggerdelay           = cmdata->cmdsettings->triggerdelay;
            fps.cmdset.triggertimeout         = cmdata->cmdsettings->triggertimeout;
            fps.cmdset.semindexrequested      = cmdata->cmdsettings->semindexrequested;
            fps.cmdset.RT_priority            = cmdata->cmdsettings->RT_priority;
            fps.cmdset.procinfo_MeasureTiming = cmdata->cmdsettings->procinfo_MeasureTiming;

            fps_add_processinfo_entries(&fps);
        }

        retval = compute_fn();
    }
    else if (retval == RETURN_CLICHECKARGARRAY_HELP)
    {
        retval = RETURN_SUCCESS;
    }

    dcfpsptr       = NULL;
    dcfpsbindings  = NULL;
    dcfpsnbindings = 0;
    if (dcfpsname[0] != '_')
    {
        fps_disconnect(&fps);
    }
    return retval;
}

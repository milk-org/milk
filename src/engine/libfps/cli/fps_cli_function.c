/**
 * @file    fps_cli_function.c
 * @brief   Generic CLIfunction and CLIADDCMD for FPS modules
 *
 * Extracted from POC sections 2.12. Provides a generic
 * CLIfunction that handles the full milk CLI lifecycle for
 * any FPS-based module.
 */


#include "CLIcore.h"


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
        fps_generic_init(dcfpsname, app_info, bindings, nb_b, 0);
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
            fps_generic_init(dcfpsname, app_info, bindings, nb_b, 0);
        }
        fps = *lfps;
    }
    else
    {
        if (fps_connect(dcfpsname, &fps, FPSCONNECT_SIMPLE) == -1)
        {
            fps_generic_init(dcfpsname, app_info, bindings, nb_b, 0);
            if (fps_connect(dcfpsname, &fps, FPSCONNECT_SIMPLE) == -1)
            {
                printf("Failed to connect to "
                       "FPS %s\n",
                       dcfpsname);
                return RETURN_SUCCESS;
            }
        }
    }

    /* Print FPS name and type in color */
    if (dcfpsname[0] == '_')
    {
        printf("\033[36mFPS \033[1m%s\033[22m"
               " \033[33m[LOCAL]\033[0m\n",
               dcfpsname);
        fps_local_set_creator(dcfpsname, app_info->fps_name);
    }
    else
    {
        printf("\033[36mFPS \033[1m%s\033[22m"
               " \033[32m[SHARED]\033[0m\n",
               dcfpsname);
        fps_shared_record_usage(dcfpsname, app_info->fps_name);
    }

    /* Record last-used FPS for ? query */
    strncpy(fps_last_used_name, dcfpsname, sizeof(fps_last_used_name) - 1);
    fps_last_used_name[sizeof(fps_last_used_name) - 1] = '\0';
    strncpy(fps_last_used_cmdkey, app_info->fps_name, sizeof(fps_last_used_cmdkey) - 1);
    fps_last_used_cmdkey[sizeof(fps_last_used_cmdkey) - 1] = '\0';

    dcfpsptr       = &fps;
    errno_t retval = CLI_checkarg_array(farg, cmdata->nbarg);

    if (retval == RETURN_SUCCESS || retval == RETURN_CLICHECKARGARRAY_FUNCPARAMSET)
    {
        fps_process_cli_and_sync(&fps, farg, bindings, nb_b);

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

        compute_fn();
        retval = RETURN_SUCCESS;
    }
    else if (retval == RETURN_CLICHECKARGARRAY_HELP)
    {
        retval = RETURN_SUCCESS;
    }

    dcfpsptr = NULL;
    if (dcfpsname[0] != '_')
    {
        fps_disconnect(&fps);
    }
    return retval;
}


/**
 * @brief Fill CLI argument example strings from binding
 *        default values.
 *
 * For each binding, formats its current C variable value
 * into the corresponding farg[ii].example string.  This
 * provides meaningful defaults in help output.
 *
 * @param farg      CLI argument definitions (examples filled)
 * @param bindings  Parameter bindings with current values
 * @param nb_b      Number of bindings
 */
void fps_fill_farg_examples(CLICMDARGDEF *farg, FPS_CLI_BINDING *bindings, int nb_b)
{
    for (int ii = 0; ii < nb_b; ii++)
    {
        switch (bindings[ii].type)
        {
        case FPTYPE_INT32:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%d",
                     *(int32_t *) bindings[ii].ptr);
            break;
        case FPTYPE_UINT32:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%u",
                     *(uint32_t *) bindings[ii].ptr);
            break;
        case FPTYPE_INT64:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%ld",
                     *(int64_t *) bindings[ii].ptr);
            break;
        case FPTYPE_UINT64:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%lu",
                     *(uint64_t *) bindings[ii].ptr);
            break;
        case FPTYPE_FLOAT32:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%f",
                     *(float *) bindings[ii].ptr);
            break;
        case FPTYPE_FLOAT64:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%lf",
                     *(double *) bindings[ii].ptr);
            break;
        case FPTYPE_ONOFF:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%ld",
                     *(uint64_t *) bindings[ii].ptr);
            break;
        case FPTYPE_PID:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%d",
                     *(pid_t *) bindings[ii].ptr);
            break;
        case FPTYPE_TIMESPEC:
            snprintf(farg[ii].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE, "%ld.%09ld",
                     ((struct timespec *) bindings[ii].ptr)->tv_sec,
                     ((struct timespec *) bindings[ii].ptr)->tv_nsec);
            break;
        case FPTYPE_STRING:
        case FPTYPE_STREAMNAME:
        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_FPSNAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_PROCESS:
        case FPTYPE_STRING_NOT_STREAM:
            strncpy(farg[ii].example, (char *) bindings[ii].ptr,
                    STRINGMAXLEN_FPSCLIARG_EXAMPLE - 1);
            break;
        }
    }
}


/**
 * @brief Constructor: register strong function
 *        pointers into the milkfps registry.
 *
 * Runs when libmilkfpsCLI.so is loaded.
 */
__attribute__((constructor))
/**
 * @brief Register an FPS CLI command at library load time.
 *
 * Called via __attribute__((constructor)) from
 * generated registration functions. Populates the
 * CLIcmddata fields and calls RegisterCLIcmd.
 */
static void
fps_cli_register(void)
{
    fps_generic_CLIfunction_ptr = fps_generic_CLIfunction;
    fps_fill_farg_examples_ptr  = fps_fill_farg_examples;
}

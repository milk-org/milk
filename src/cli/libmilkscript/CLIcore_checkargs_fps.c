/**
 * @file CLIcore_checkargs_fps.c
 *
 * @brief FPS parameter bridging from CLI arguments
 *
 * Handles the translation between CLI command-line
 * arguments and FPS (Function Parameter Struct) shared
 * memory parameters. This is the bridge that allows
 * CLI-entered values to flow into the real-time FPS
 * parameter system.
 *
 * Key functions:
 *
 * - **CLIargs_to_FPSparams_setval**: Copy CLI arg
 *   values into an existing FPS, updating the shared
 *   memory parameter array with the values parsed
 *   from the command line.
 *
 * - **CMDargs_to_FPSparams_create**: Create new FPS
 *   parameter entries from the command's CLICMDARGDEF
 *   array, initializing them with description, type,
 *   and default values.
 *
 * - **get_farg_ptr**: Retrieve a void pointer to the
 *   typed value inside the command's argdata union,
 *   used by the generic function-call dispatch.
 *
 * - **function_parameter_getFPSargs_from_CLIfunc**:
 *   Top-level orchestrator that connects CLI to FPS —
 *   creates/opens an FPS, maps arguments, and runs
 *   the function.
 *
 * ## Design approach
 *
 * The functions use type-dispatch switch blocks over
 * the CLIARG_* / FPTYPE_* enums to convert between
 * CLI token types and FPS native types. Each function
 * covers the full set of supported types (float, int,
 * string, stream, onoff, timespec, pid, etc.).
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_checkargs.h"
#include "fps_globals.h"

/** @brief Build FPS content from FPSCLIARG list
 *
 * All CLI arguments converted to FPS parameters
 *
 */
int CLIargs_to_FPSparams_setval(CLICMDARGDEF fpscliarg[], int nbarg, FPS *fps)
{
    DEBUG_TRACE_FSTART();

    int NBarg_processed = 0;
    int cmdi            = data.cmdindex;

    /*
     * Limit iteration to the number of CLI args
     * actually registered (and thus allocated in
     * argdata).  The caller may pass nb_bindings
     * as nbarg, which can exceed nbparam when the
     * CLIcmddata uses CLICMD_FIELDS_NOPARAM.
     */
    int nreg = data.cmd[cmdi].nbparam;
    if (data.cmd[cmdi].argdata == NULL)
    {
        nreg = 0;
    }
    int nlimit = (nbarg < nreg) ? nbarg : nreg;

    for (int arg = 0; arg < nlimit; arg++)
    {
        // if argument is part of FPS
        switch (fpscliarg[arg].type)
        {
        case CLIARG_FLOAT32:
            functionparameter_SetParamValue_FLOAT32(fps, fpscliarg[arg].fpstag,
                                                    data.cmd[cmdi].argdata[arg].val.f32);
            NBarg_processed++;
            break;

        case CLIARG_FLOAT64:
            functionparameter_SetParamValue_FLOAT64(fps, fpscliarg[arg].fpstag,
                                                    data.cmd[cmdi].argdata[arg].val.f64);
            NBarg_processed++;
            break;

        case CLIARG_ONOFF:
            functionparameter_SetParamValue_ONOFF(fps, fpscliarg[arg].fpstag,
                                                  (int) data.cmd[cmdi].argdata[arg].val.i64);
            NBarg_processed++;
            break;

        case CLIARG_INT32:
            functionparameter_SetParamValue_INT32(fps, fpscliarg[arg].fpstag,
                                                  data.cmd[cmdi].argdata[arg].val.i32);
            NBarg_processed++;
            break;

        case CLIARG_UINT32:
            functionparameter_SetParamValue_UINT32(fps, fpscliarg[arg].fpstag,
                                                   data.cmd[cmdi].argdata[arg].val.ui32);
            NBarg_processed++;
            break;

        case CLIARG_INT64:
            functionparameter_SetParamValue_INT64(fps, fpscliarg[arg].fpstag,
                                                  data.cmd[cmdi].argdata[arg].val.i64);
            NBarg_processed++;
            break;

        case CLIARG_UINT64:
            functionparameter_SetParamValue_UINT64(fps, fpscliarg[arg].fpstag,
                                                   data.cmd[cmdi].argdata[arg].val.ui64);
            NBarg_processed++;
            break;

        case FPTYPE_PID:
            functionparameter_SetParamValue_INT64(fps, fpscliarg[arg].fpstag,
                                                  data.cmd[cmdi].argdata[arg].val.i64);
            NBarg_processed++;
            break;

        case FPTYPE_TIMESPEC:
            functionparameter_SetParamValue_TIMESPEC(fps, fpscliarg[arg].fpstag,
                                                     data.cmd[cmdi].argdata[arg].val.f64);
            NBarg_processed++;
            break;

        case CLIARG_STR_NOT_IMG:
        case CLIARG_IMG:
        case CLIARG_STR:
        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_FPSNAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_PROCESS:
            functionparameter_SetParamValue_STRING(fps, fpscliarg[arg].fpstag,
                                                   data.cmd[cmdi].argdata[arg].val.s);
            NBarg_processed++;
            break;
        }
    }

    DEBUG_TRACE_FEXIT();
    return NBarg_processed;
}


/** @brief Build FPS from command args
 */
int CMDargs_to_FPSparams_create(FPS *fps)
{
    DEBUG_TRACE_FSTART();

    int  NBarg_processed = 0;
    long fpi             = 0;


    for (int argi = 0; argi < data.cmd[data.cmdindex].nbparam; argi++)
    {
        // if argument is part of FPS
        long tmpvall = 0;

        switch (data.cmd[data.cmdindex].argdata[argi].type)
        {
            // float point types

        case FPTYPE_FLOAT32:
        {
            float tmpf = data.cmd[data.cmdindex].argdata[argi].val.f32;
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr,
                                         FPTYPE_FLOAT32,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag, &tmpf, NULL);
            NBarg_processed++;
        }
        break;

        case FPTYPE_FLOAT64:
        {
            double tmplf = data.cmd[data.cmdindex].argdata[argi].val.f64;
            function_parameter_add_entry(
                fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_FLOAT64,
                data.cmd[data.cmdindex].argdata[argi].fpflag, &tmplf, NULL);
            NBarg_processed++;
        }
        break;

            // integer typtes

        case FPTYPE_ONOFF: // default to INT64
        {
            tmpvall = data.cmd[data.cmdindex].argdata[argi].val.ui64;
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_ONOFF,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag, &tmpvall,
                                         NULL);
            NBarg_processed++;
        }
        break;

        case FPTYPE_INT32:
        {
            int32_t tmpi32 = data.cmd[data.cmdindex].argdata[argi].val.i32;
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_INT32,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag, &tmpi32,
                                         NULL);
            NBarg_processed++;
        }
        break;

        case FPTYPE_UINT32:
        {
            uint32_t tmpui32 = data.cmd[data.cmdindex].argdata[argi].val.ui32;
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_UINT32,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag, &tmpui32,
                                         NULL);
            NBarg_processed++;
        }
        break;

        case FPTYPE_INT64:
        {
            int64_t tmpi64 = data.cmd[data.cmdindex].argdata[argi].val.i64;
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_INT64,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag, &tmpi64,
                                         NULL);
            NBarg_processed++;
        }
        break;

        case FPTYPE_UINT64:
        {
            uint64_t tmpui64 = data.cmd[data.cmdindex].argdata[argi].val.ui64;
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_UINT64,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag, &tmpui64,
                                         NULL);
            NBarg_processed++;
        }
        break;

        case FPTYPE_STRING_NOT_STREAM:
        case FPTYPE_STRING:
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr, FPTYPE_STRING,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag,
                                         data.cmd[data.cmdindex].argdata[argi].val.s, NULL);
            NBarg_processed++;
            break;

        case FPTYPE_STREAMNAME:
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr,
                                         FPTYPE_STREAMNAME,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag,
                                         data.cmd[data.cmdindex].argdata[argi].val.s, NULL);
            NBarg_processed++;
            break;

        case FPTYPE_FILENAME:
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr,
                                         FPTYPE_FILENAME,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag,
                                         data.cmd[data.cmdindex].argdata[argi].val.s, NULL);
            NBarg_processed++;
            break;

        case FPTYPE_FITSFILENAME:
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr,
                                         FPTYPE_FITSFILENAME,
                                         data.cmd[data.cmdindex].argdata[argi].fpflag,
                                         data.cmd[data.cmdindex].argdata[argi].val.s, NULL);
            NBarg_processed++;
            break;

        case FPTYPE_FPSNAME:
            //printf("ADDING FPS ENTRY %s at index %d\n", data.cmd[data.cmdindex].argdata[argi].fpstag, argi);
            function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                         data.cmd[data.cmdindex].argdata[argi].descr,
                                         FPTYPE_FPSNAME,
                                         FPFLAG_DEFAULT_INPUT | FPFLAG_FPS_RUN_REQUIRED,
                                         data.cmd[data.cmdindex].argdata[argi].val.s, &fpi);
            //printf("fpi = %ld\n", fpi);
            fps->parray[fpi].info.fps.FPSNBparamMAX = 0;
            NBarg_processed++;
            break;
        }
    }

    DEBUG_TRACE_FEXIT();
    return NBarg_processed;
}


/** @brief get FPS pointer to function argument/parameter
 */
void *get_farg_ptr(char *tag, long *fpsi)
{
    DEBUG_TRACE_FSTART();

    void *ptr = NULL;

    DEBUG_TRACEPOINT("looking for pointer %s", tag);
    DEBUG_TRACEPOINT("FPS_CMDCODE = %d", dcfpscode);

    if (dcfpscode != 0)
    {
        // look for pointer in FPS
        // We use INT64 type as default
        ptr = (void *) functionparameter_GetParamPtr_generic(dcfpsptr, tag, fpsi);
    }
    else
    {
        for (int argi = 0; argi < data.cmd[data.cmdindex].nbparam; argi++)
        {
            if (strcmp(data.cmd[data.cmdindex].argdata[argi].fpstag, tag) == 0)
            {
                ptr = (void *) (&data.cmd[data.cmdindex].argdata[argi].val);
                break;
            }
        }
    }
    DEBUG_TRACEPOINT("found pointer");

    DEBUG_TRACE_FEXIT();
    return ptr;
}

/** @brief get FPS arguments from command line function call
 *
 * This function is intended to be used when running from the CLI.
 * For standalone applications, arguments are parsed in the main() function.
 *
 * This function has been moved from libfps to CLIcore_checkargs.c
 * to have native access to CLIcore data structures.
 */
errno_t function_parameter_getFPSargs_from_CLIfunc(char *fpsname_default)
{
    DEBUG_TRACE_FSTART();

    int INIT_MODE __attribute__((unused)) = 0;
    dcfpscode                             = 0;
    char FPS_name[STRINGMAXLEN_FPS_NAME];
    int  FPS_name_set = 0;

    // Initialize FPS_name to default
    if (fpsname_default != NULL)
    {
        strncpy(FPS_name, fpsname_default, STRINGMAXLEN_FPS_NAME - 1);
        FPS_name[STRINGMAXLEN_FPS_NAME - 1] = '\0';
    }
    else
    {
        FPS_name[0] = '\0';
    }

    // Check if the command itself has colon-separated syntax: cmdkey:fpsname:action
    char *cmd_str     = data.cmdargtoken[0].val.string;
    char *first_colon = strchr(cmd_str, ':');
    if (first_colon != NULL)
    {
        char *second_colon = strchr(first_colon + 1, ':');

        // Extract FPS name if present between first and second colon, or after first colon
        if (first_colon[1] != '\0' && first_colon[1] != ':')
        {
            size_t fps_len = second_colon ? (size_t) (second_colon - (first_colon + 1))
                                          : strlen(first_colon + 1);
            if (fps_len > 0 && fps_len < STRINGMAXLEN_FPS_NAME)
            {
                strncpy(FPS_name, first_colon + 1, fps_len);
                FPS_name[fps_len] = '\0';
                FPS_name_set      = 1;
            }
        }


        // Extract action if present after second colon
        if (second_colon != NULL && second_colon[1] != '\0')
        {
            char *action = second_colon + 1;
            if (strcmp(action, "init") == 0)
            {
                dcfpscode = FPSCMDCODE_FPSINIT;
                data.cmd[data.cmdindex].cmdsettings.flags &= ~CLICMDFLAG_PROCINFO;
            }
            else if (strcmp(action, "initp") == 0)
            {
                dcfpscode = FPSCMDCODE_FPSINIT;
                data.cmd[data.cmdindex].cmdsettings.flags |= CLICMDFLAG_PROCINFO;
            }
            else if (strcmp(action, "?") == 0)
            {
                // Ignore the command so it doesn't run, but allow FPS args check
                dcfpscode = FPSCMDCODE_IGNORE;
                // Print the FPS parameters by connecting to it
                FPS tmp_fps;
                if (fps_connect(FPS_name, &tmp_fps, FPSCONNECT_SIMPLE) == -1)
                {
                    printf("FPS %s does not exist.\n", FPS_name);
                }
                else
                {
                    function_parameter_print_info(&tmp_fps, 0, 0);
                    fps_disconnect(&tmp_fps);
                }
            }
        }
    }

    strncpy(dcfpsname, FPS_name, STRINGMAXLEN_FPS_NAME - 1);
    dcfpsname[STRINGMAXLEN_FPS_NAME - 1] = '\0';

    // Read FPS interface from args
    if (dcfpscode == 0)
    {
        // set to 0 as default if we didn't extract an action above
        dcfpscode = 0;
    }


    int printinfo __attribute__((unused)) = 0;
    // if using FPS implementation, FPSCMDCODE will be set to != 0
    DEBUG_TRACEPOINT("pre-processing CLI arg");

    // by default, pre-process argument
    int argpreprocess = 1;

    if (data.cmdNBarg < 2)
    {
        return RETURN_SUCCESS;
    }

    switch (data.cmdargtoken[1].type)
    {
    case CLIARG_FLOAT32:
    case CLIARG_FLOAT64:
    case CLIARG_INT32:
    case CLIARG_UINT32:
    case CLIARG_INT64:
    case CLIARG_UINT64:
        argpreprocess = 0;
        break;
    }

    if (argpreprocess == 1)
    {
        // modify function attribute

        if (strcmp(data.cmdargtoken[1].val.string, "..procinfo") == 0)
        {
            if (data.cmdargtoken[2].val.numl == 0)
            {
                printf("Command %ld: updating PROCINFO mode OFF\n", data.cmdindex);
                data.cmd[data.cmdindex].cmdsettings.flags &= ~CLICMDFLAG_PROCINFO;
            }
            else
            {
                printf("Command %ld: updating PROCINFO mode ON\n", data.cmdindex);
                data.cmd[data.cmdindex].cmdsettings.flags |= CLICMDFLAG_PROCINFO;
            }
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if (strcmp(data.cmdargtoken[1].val.string, "..RTprio") == 0)
        {
            printf("Command %ld: updating RTprio to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.RT_priority = data.cmdargtoken[2].val.numl;
            dcfpscode                                       = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if (strcmp(data.cmdargtoken[1].val.string, "..loopcntMax") == 0)
        {
            printf("Command %ld: updating loopcntMax to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.procinfo_loopcntMax = data.cmdargtoken[2].val.numl;
            dcfpscode                                               = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if (strcmp(data.cmdargtoken[1].val.string, "..triggermode") == 0)
        {
            printf("Command %ld: updating triggermode to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.triggermode = data.cmdargtoken[2].val.numl;
            dcfpscode                                       = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if (strcmp(data.cmdargtoken[1].val.string, "..triggersname") == 0)
        {
            printf("Command %ld: updating triggerstreamname to value %s\n", data.cmdindex,
                   data.cmdargtoken[2].val.string);
            strncpy(data.cmd[data.cmdindex].cmdsettings.triggerstreamname,
                    data.cmdargtoken[2].val.string, STRINGMAXLEN_IMAGE_NAME - 1);
            data.cmd[data.cmdindex].cmdsettings.triggerstreamname[STRINGMAXLEN_IMAGE_NAME - 1] =
                '\0';
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if (strcmp(data.cmdargtoken[1].val.string, "..semindexrequested") == 0)
        {
            printf("Command %ld: updating semindexrequested to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.semindexrequested = data.cmdargtoken[2].val.numl;
            dcfpscode                                             = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }


        if (strcmp(data.cmdargtoken[1].val.string, "..triggerdelay") == 0)
        {
            double x = 0.0;
            switch (data.cmdargtoken[2].type)
            {
            case CMDARGTOKEN_TYPE_FLOAT:
                x = data.cmdargtoken[2].val.numf;
                break;

            case CMDARGTOKEN_TYPE_LONG:
                x = data.cmdargtoken[2].val.numl;
                break;

            default:
                printf("wrong argument type, should be float or int "
                       "-> setting to zero\n");
            }
            printf("Command %ld: updating triggerdelay to value %f\n", data.cmdindex, x);
            x += 0.5e-9;
            long x_sec  = (long) x;
            long x_nsec = (x - x_sec) * 1000000000L;

            data.cmd[data.cmdindex].cmdsettings.triggerdelay.tv_sec  = x_sec;
            data.cmd[data.cmdindex].cmdsettings.triggerdelay.tv_nsec = x_nsec;
            dcfpscode                                                = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if (strcmp(data.cmdargtoken[1].val.string, "..triggertimeout") == 0)
        {
            printf("Command %ld: updating triggertimeout to value %f\n", data.cmdindex,
                   data.cmdargtoken[2].val.numf);
            double x = data.cmdargtoken[2].val.numf;
            x += 0.5e-9;
            long x_sec  = (long) x;
            long x_nsec = (x - x_sec) * 1000000000L;

            data.cmd[data.cmdindex].cmdsettings.triggertimeout.tv_sec  = x_sec;
            data.cmd[data.cmdindex].cmdsettings.triggertimeout.tv_nsec = x_nsec;
            dcfpscode                                                  = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        // check if recognized FPSCMDCODE
        if (strcmp(data.cmdargtoken[1].val.string,
                   "_FPSINIT_") == 0) // Initialize FPS
        {
            dcfpscode = FPSCMDCODE_FPSINIT;
        }
        else if (strcmp(data.cmdargtoken[1].val.string,
                        "_CONFSTART_") == 0) // Start conf process
        {
            dcfpscode = FPSCMDCODE_CONFSTART;
        }
        else if (strcmp(data.cmdargtoken[1].val.string,
                        "_CONFSTOP_") == 0) // Stop conf process
        {
            dcfpscode = FPSCMDCODE_CONFSTOP;
        }
        else if (strcmp(data.cmdargtoken[1].val.string,
                        "_RUNSTART_") == 0) // Run process
        {
            dcfpscode = FPSCMDCODE_RUNSTART;
        }
        else if (strcmp(data.cmdargtoken[1].val.string,
                        "_RUNSTOP_") == 0) // Stop process
        {
            dcfpscode = FPSCMDCODE_RUNSTOP;
        }
        else if (strcmp(data.cmdargtoken[1].val.string,
                        "_TMUXSTART_") == 0) // Start tmux session
        {
            dcfpscode = FPSCMDCODE_TMUXSTART;
        }
        else if (strcmp(data.cmdargtoken[1].val.string,
                        "_TMUXSTOP_") == 0) // Stop tmux session
        {
            dcfpscode = FPSCMDCODE_TMUXSTOP;
        }
    }


    // if recognized FPSCMDCODE, use FPS implementation
    if ((dcfpscode != 0) && (dcfpscode != FPSCMDCODE_IGNORE))
    {
        // ===============================
        //     SET FPS INTERFACE NAME
        // ===============================

        // if main CLI process has been named with -n option, then use the process name to construct fpsname
        if (FPS_name_set == 0)
        {
            if (data.processnameflag == 1)
            {
                // Automatically set fps name to be process name up to first instance of character '.'
                strncpy(FPS_name, data.processname0, STRINGMAXLEN_FPS_NAME - 1);
                FPS_name[STRINGMAXLEN_FPS_NAME - 1] = '\0';
            }
            else // otherwise, construct name as follows
            {
                // Adopt default name for fpsname
                int slen = snprintf(FPS_name, STRINGMAXLEN_FPS_NAME, "%s", fpsname_default);
                if (slen < 1)
                {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if (slen >= STRINGMAXLEN_FPS_NAME)
                {
                    PRINT_ERROR("snprintf string truncation.\n"
                                "Full string  : %s\n"
                                "Truncated to : %s",
                                fpsname_default, FPS_name);
                    abort(); // can't handle this error any other way
                }
            }
        }
    }

    // Always keep dcfpsname updated
    strncpy(dcfpsname, FPS_name, STRINGMAXLEN_FPS_NAME - 1);
    dcfpsname[STRINGMAXLEN_FPS_NAME - 1] = '\0';

    // if recognized FPSCMDCODE, use FPS implementation
    if ((dcfpscode != 0) && (dcfpscode != FPSCMDCODE_IGNORE))
    {
        // By convention, if there are optional arguments,
        // they should be appended to the default fps name
        //
        int argindex = 2; // start at arg #2
        while (data.cmdargtoken[argindex].type != CMDARGTOKEN_TYPE_UNSOLVED &&
               strlen(data.cmdargtoken[argindex].val.string) > 0)
        {
            if (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_UNSOLVED ||
                (strcmp(data.cmdargtoken[1].val.string, "_FPSINIT_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_CONFSTART_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_CONFSTOP_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_RUNSTART_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_RUNSTOP_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_TMUXSTART_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_TMUXSTOP_") != 0))
            {
                break; // Don't append regular arguments to the FPS name
            }

            char fpsname1[STRINGMAXLEN_FPS_NAME];

            int slen = snprintf(fpsname1, STRINGMAXLEN_FPS_NAME, "%s-%s", FPS_name,
                                data.cmdargtoken[argindex].val.string);
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_FPS_NAME)
            {
                PRINT_ERROR("snprintf string truncation.\n"
                            "Full string  : %s-%s\n"
                            "Truncated to : %s",
                            FPS_name, data.cmdargtoken[argindex].val.string, fpsname1);
                abort(); // can't handle this error any other way
            }

            strncpy(FPS_name, fpsname1, STRINGMAXLEN_FPS_NAME - 1);
            argindex++;
        }
    }

    return RETURN_SUCCESS;
}

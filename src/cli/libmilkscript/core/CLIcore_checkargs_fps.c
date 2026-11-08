// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
 * @return number of arguments written, or -1 (reported on stderr) when a string does
 *         not fit its FPS parameter
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
     * the runnable has no parameters.
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
        {
            double          value = data.cmd[cmdi].argdata[arg].val.f64;
            struct timespec t     = { .tv_sec = value, .tv_nsec = 0 };
            t.tv_nsec             = 1e9 * (value - t.tv_sec);
            functionparameter_SetParamValue_TIMESPEC(fps, fpscliarg[arg].fpstag, t);
            NBarg_processed++;
            break;
        }

        case CLIARG_STR_NOT_IMG:
        case CLIARG_IMG:
        case CLIARG_STR:
        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_FPSNAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_PROCESS:
        {
            /* an FPS string holds FUNCTION_PARAMETER_STRMAXLEN - 1 characters: refuse, never cut
             * (as milk-fpsexec-* does, fps_cli_sync.c) */
            const char *s   = data.cmd[cmdi].argdata[arg].val.s;
            const char *tag = fpscliarg[arg].fpstag;
            if (strlen(s) >= FUNCTION_PARAMETER_STRMAXLEN)
            {
                fprintf(
                    stderr,
                    "%s: invalid value '%s' for parameter '%s': expected at most %d characters\n",
                    data.cmd[cmdi].key, s, (tag[0] == '.') ? tag + 1 : tag,
                    FUNCTION_PARAMETER_STRMAXLEN - 1);
                DEBUG_TRACE_FEXIT();
                return -1;
            }
            functionparameter_SetParamValue_STRING(fps, tag, s);
            NBarg_processed++;
            break;
        }
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
 */
errno_t function_parameter_getFPSargs_from_CLIfunc(char *fpsname_default)
{
    DEBUG_TRACE_FSTART();

    int INIT_MODE __attribute__((unused)) = 0;
    dcfpscode                             = 0;
    char FPS_name[STRINGMAXLEN_FPS_NAME];

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

    strncpy(dcfpsname, FPS_name, STRINGMAXLEN_FPS_NAME - 1);
    dcfpsname[STRINGMAXLEN_FPS_NAME - 1] = '\0';

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
        static const char *const gone[] = { "_CONFSTART_", "_CONFSTOP_",  "_RUNSTART_",
                                            "_RUNSTOP_",   "_TMUXSTART_", "_TMUXSTOP_" };
        const char              *arg1   = data.cmdargtoken[1].val.string;
        if (strcmp(arg1, "_FPSINIT_") == 0) // Initialize FPS
        {
            dcfpscode = FPSCMDCODE_FPSINIT;
        }
        for (size_t i = 0; i < sizeof(gone) / sizeof(gone[0]); i++)
        {
            if (strcmp(arg1, gone[i]) == 0) // milk-cli no longer hosts units (U2)
            {
                fprintf(stderr,
                        "milk: %s %s was removed: a unit runs from its fpsexec binary "
                        "(<prefix>-fpsexec-<id> -n NAME fpsinit, confstart, runstart, runstop; "
                        "milk-fpsexec-list lists them)\n",
                        data.cmdargtoken[0].val.string, arg1);
                dcfpscode = FPSCMDCODE_IGNORE;
                DEBUG_TRACE_FEXIT();
                return RETURN_FAILURE;
            }
        }
    }


    // if recognized FPSCMDCODE, use FPS implementation
    if ((dcfpscode != 0) && (dcfpscode != FPSCMDCODE_IGNORE))
    {
        // ===============================
        //     SET FPS INTERFACE NAME
        // ===============================

        // if main CLI process has been named with -n option, then use the process name to construct fpsname
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
                strcmp(data.cmdargtoken[1].val.string, "_FPSINIT_") != 0)
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

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_sync.c
 * @brief   Sync standalone command-line arguments to FPS and local variables
 *
 * Takes the argv of a standalone executable and syncs it into FPS
 * shared memory, then copies FPS values back into module-local C
 * variables via the binding array. The milk-cli counterpart lives in
 * libmilkscript.
 */


#include <stdlib.h>
#include <string.h>

#include "fps_types.h"
#include "fps_internal.h"
#include "fps_GetParamIndex.h"

#include "fps_cli_binding.h"
#include "fps_modulevars_resync.h"


/* Standalone argc/argv captured by main() */
static int    standalone_argc;
static char **standalone_argv;


/**
 * @brief Capture argc/argv from standalone main() for later
 *        CLI-to-FPS argument synchronization.
 *
 * Must be called once before fps_process_argv_and_sync().
 * Never called by milk-cli, where standalone_argv stays NULL.
 *
 * @param argc  Argument count from main()
 * @param argv  Argument vector from main()
 */
void fps_cli_set_standalone_args(int argc, char **argv)
{
    standalone_argc = argc;
    standalone_argv = argv;
}


/**
 * @brief Write a single CLI string value into an FPS
 *        parameter, interpreting it according to the type.
 */
static void set_fps_value_from_string(FPS *fps, long pindex, uint64_t type, const char *str)
{
    if (type == FPTYPE_FLOAT64)
    {
        fps->parray[pindex].val.f64[0] = atof(str);
    }
    else if (type == FPTYPE_FLOAT32)
    {
        fps->parray[pindex].val.f32[0] = (float) atof(str);
    }
    else if (type == FPTYPE_INT64)
    {
        fps->parray[pindex].val.i64[0] = atoll(str);
    }
    else if (type == FPTYPE_UINT64)
    {
        fps->parray[pindex].val.ui64[0] = (uint64_t) atoll(str);
    }
    else if (type == FPTYPE_INT32 || type == FPTYPE_ONOFF)
    {
        fps->parray[pindex].val.i32[0] = atoi(str);
    }
    else if (type == FPTYPE_UINT32)
    {
        fps->parray[pindex].val.ui32[0] = (uint32_t) atoi(str);
    }
    else if (type == FPTYPE_PID)
    {
        fps->parray[pindex].val.pid[0] = (pid_t) atoi(str);
    }
    else if (type == FPTYPE_TIMESPEC)
    {
        double val                            = atof(str);
        fps->parray[pindex].val.ts[0].tv_sec  = (long) val;
        fps->parray[pindex].val.ts[0].tv_nsec = (long) ((val - (long) val) * 1e9);
    }
    else if (FPTYPE_IS_STRING(type))
    {
        strncpy(fps->parray[pindex].val.string[0], str, FUNCTION_PARAMETER_STRMAXLEN - 1);
    }
}


/**
 * @brief Sync standalone argv into FPS and copy FPS values
 *        back into module-local C variables.
 *
 * Two-step process:
 *   1. argv -> FPS: parse positional arguments and write them
 *      into the FPS shared memory.
 *   2. FPS -> local: for each binding, copy the FPS
 *      parameter value into the local C variable.
 *
 * Locates the subcommand keyword (run, exec, set, etc.) and
 * maps subsequent positional arguments to primary bindings in
 * order.  A "." value is treated as "keep default".
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return RETURN_SUCCESS
 */
errno_t fps_process_argv_and_sync(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    /* ---- Step 1: argv -> FPS ---- */
    if (standalone_argv != NULL)
    {
        /*
         * Find the subcommand position, then map
         * positional args after it to primary bindings.
         */
        int cmd_pos = -1;
        for (int jj = 1; jj < standalone_argc; jj++)
        {
            /* Extract command part after optional
             * 'name:' prefix */
            const char *arg      = standalone_argv[jj];
            const char *cmd_part = arg;
            {
                const char *cp = strchr(arg, ':');
                if (cp != NULL)
                {
                    cmd_part = cp + 1;
                }
            }
            if (strcmp(cmd_part, "runstart") == 0 || strcmp(cmd_part, "run") == 0 ||
                strcmp(cmd_part, "exec") == 0 || strcmp(cmd_part, "set") == 0 ||
                strcmp(cmd_part, "confstart") == 0 || strcmp(cmd_part, "confstep") == 0 ||
                strcmp(cmd_part, "fpsinit") == 0)
            {
                cmd_pos = jj;
                break;
            }
        }

        /* If no explicit command, try implicit run
         * (first non-flag arg) */
        if (cmd_pos == -1)
        {
            for (int jj = 1; jj < standalone_argc; jj++)
            {
                if (standalone_argv[jj][0] != '-')
                {
                    cmd_pos = jj;
                    break;
                }
            }
        }

        if (cmd_pos != -1)
        {
            int cli_idx = 0;
            for (int aa = cmd_pos + 1; aa < standalone_argc; aa++)
            {
                /* Skip -flag arguments */
                if (standalone_argv[aa][0] == '-')
                {
                    continue;
                }

                /* Map to next primary binding */
                while (cli_idx < nb_b && !bindings[cli_idx].is_primary)
                {
                    cli_idx++;
                }
                if (cli_idx >= nb_b)
                {
                    break;
                }

                long pindex = functionparameter_GetParamIndex(fps, bindings[cli_idx].fpskeyword);
                if (pindex != -1 && strcmp(standalone_argv[aa], ".") != 0)
                {
                    set_fps_value_from_string(fps, pindex, bindings[cli_idx].type,
                                              standalone_argv[aa]);
                }
                cli_idx++;
            }
        }
    }

    /* ---- Step 2: FPS -> local C variables ---- */
    return fps_to_modulevars_resync_bindings(fps, bindings, nb_b);
}

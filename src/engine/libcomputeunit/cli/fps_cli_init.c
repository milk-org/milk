// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_init.c
 * @brief   Initialize FPS entries from a bindings array
 *
 * Populates an FPS structure with entries defined by
 * the module's FPS_CLI_BINDING array. Handles primary
 * CLI argument indexing.
 */


#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include "computeunit.h"


/**
 * @brief Initialize FPS parameters from CLI bindings.
 *
 * Creates each parameter in the FPS using the type,
 * description, and flags specified in the binding
 * array. Called during fpsinit. An ON/OFF parameter
 * takes its variable's value as default.
 */
errno_t fps_init_from_bindings(FPS             *fps,
                               const char      *cmdkey,
                               const char      *description,
                               FPS_CLI_BINDING *bindings,
                               int              nb_b)
{
    strncpy(fps->md->callprogname, cmdkey, FPS_CALLPROGNAME_STRMAXLEN - 1);
    strncpy(fps->md->description, description, FPS_DESCR_STRMAXLEN - 1);

    int current_cli_index = 0;

    for (int bind_idx = 0; bind_idx < nb_b; bind_idx++)
    {
        long     pindex    = -1;
        uint64_t fpflag    = bindings[bind_idx].fpflag;
        int      cli_index = -1;

        if (bindings[bind_idx].is_primary)
        {
            fpflag |= FPFLAG_PRIMARY_CLI_INPUT;
            cli_index = current_cli_index++;
        }

        function_parameter_add_entry(fps, bindings[bind_idx].fpskeyword, bindings[bind_idx].descr,
                                     bindings[bind_idx].type, fpflag, bindings[bind_idx].ptr,
                                     &pindex);
        functionparameter_SetParamCLIindex(fps, pindex, cli_index);

        /* add_entry starts an ON/OFF entry OFF (the flag is its value): apply the default, read as
         * an int32 like the resync does */
        if (bindings[bind_idx].type == FPTYPE_ONOFF && pindex >= 0 &&
            bindings[bind_idx].ptr != NULL)
        {
            int32_t on = *(int32_t *) bindings[bind_idx].ptr ? 1 : 0;
            if (on)
            {
                fps->parray[pindex].fpflag |= FPFLAG_ONOFF;
            }
            fps->parray[pindex].val.i64[0] = on;
        }
    }
    return RETURN_SUCCESS;
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

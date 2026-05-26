/**
 * @file    fps_cli_init.c
 * @brief   Initialize FPS entries from a bindings array
 *
 * Populates an FPS structure with entries defined by
 * the module's FPS_CLI_BINDING array. Handles primary
 * CLI argument indexing.
 */


#ifndef FPS_STANDALONE
#else
#endif
#include "fps.h"


/**
 * @brief Initialize FPS parameters from CLI bindings.
 *
 * Creates each parameter in the FPS using the type,
 * description, and flags specified in the binding
 * array. Called during fpsinit.
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
        long     pindex;
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
    }
    return RETURN_SUCCESS;
}

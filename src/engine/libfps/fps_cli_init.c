/**
 * @file    fps_cli_init.c
 * @brief   Initialize FPS entries from a bindings array
 *
 * Populates an FPS structure with entries defined by
 * the module's FPS_CLI_BINDING array. Handles primary
 * CLI argument indexing.
 */

#include <string.h>

#ifndef FPS_STANDALONE
#include "CLIcore.h"
#else
#include "libmilkdata/milkdata.h"
#endif
#include "fps.h"
#include "fps_add_entry.h"
#include "fps_SetParamCLIindex.h"
#include "fps_cli_binding.h"
#include "fps_cli_init.h"


errno_t fps_init_from_bindings(
    FPS *fps,
    const char                *cmdkey,
    const char                *description,
    FPS_CLI_BINDING           *bindings,
    int                        nb_b
)
{
    strncpy(fps->md->callprogname, cmdkey,
            FPS_CALLPROGNAME_STRMAXLEN - 1);
    strncpy(fps->md->description, description,
            FPS_DESCR_STRMAXLEN - 1);

    int current_cli_index = 0;

    for (int i = 0; i < nb_b; i++) {
        long pindex;
        uint64_t fpflag = bindings[i].fpflag;
        int cli_index = -1;

        if (bindings[i].is_primary) {
            fpflag |= FPFLAG_PRIMARY_CLI_INPUT;
            cli_index = current_cli_index++;
        }

        function_parameter_add_entry(
            fps,
            bindings[i].fpskeyword,
            bindings[i].descr,
            bindings[i].type,
            fpflag,
            bindings[i].ptr,
            &pindex
        );
        functionparameter_SetParamCLIindex(
            fps, pindex, cli_index);
    }
    return RETURN_SUCCESS;
}

/**
 * @file    fps_CONFstart.c
 * @brief   FPS conf process start
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"
#include "fps_tmux.h"

/** @brief FPS start CONF process
 *
 * Requires setup performed by milk-fpsinit, which performs the following setup
 * - creates the FPS shared memory
 * - create up tmux sessions
 * - create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop
 */

errno_t functionparameter_CONFstart(FPS *fps)
{
    functionparameter_FPS_tmux_ensure(fps);

    // Move to correct launch directory
    //
    EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" cd %s\" C-m",
                           fps->md->name,
                           fps->md->workdir);

    if (strstr(fps->md->execfullpath, "fpsexec") != NULL) {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" %s %s:confstart\" C-m",
                               fps->md->name,
                               fps->md->execfullpath,
                               fps->md->name);
    } else {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" fpsconfstart\" C-m",
                               fps->md->name);
    }

    fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;

    // notify GUI loop to update
    fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

    return RETURN_SUCCESS;
}

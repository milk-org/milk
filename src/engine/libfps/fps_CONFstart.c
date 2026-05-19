/**
 * @file    fps_CONFstart.c
 * @brief   FPS conf process start
 */

#include "fps.h"

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
                                   fps->md->name, fps->md->workdir);

    char *exec_basename = strrchr(fps->md->execfullpath, '/');
    exec_basename = (exec_basename != NULL) ? exec_basename + 1 : fps->md->execfullpath;

    if(strcmp(exec_basename, "milk") != 0 &&
            strcmp(exec_basename, "cacao") != 0 &&
            strlen(exec_basename) > 0 &&
            strcmp(exec_basename, "unknown") != 0)
    {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" %s %s:confstart\" C-m",
                                       fps->md->name, fps->md->execfullpath, fps->md->name);
    }
    else
    {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" fpsconfstart\" C-m",
                                       fps->md->name);
    }

    fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;

    // notify GUI loop to update
    fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

    return RETURN_SUCCESS;
}

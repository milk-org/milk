/**
 * @file    fps_RUNstop.c
 * @brief   FPS run process stop
 */

#include "fps.h"

/** @brief FPS stop RUN process
 *
 * Run pre-set function fpsrunstop in tmux ctrl window
 */
errno_t functionparameter_RUNstop(FPS *fps)
{
    // Move to correct launch directory
    //
    EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:ctrl \" cd %s\" C-m",
                           fps->md->name,
                           fps->md->workdir);

    char * exec_basename = strrchr(fps->md->execfullpath, '/');
    exec_basename = (exec_basename != NULL) ? exec_basename + 1 : fps->md->execfullpath;

    if (strcmp(exec_basename, "milk") != 0 && 
        strcmp(exec_basename, "cacao") != 0 && 
        strlen(exec_basename) > 0 && 
        strcmp(exec_basename, "unknown") != 0) 
    {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:ctrl \" %s %s:runstop\" C-m",
                               fps->md->name,
                               fps->md->execfullpath,
                               fps->md->name);
    } else {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:ctrl \" fpsrunstop\" C-m",
                               fps->md->name);
    }

    // Send C-c in case runstop command is not implemented
    EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run C-c &> /dev/null",
                           fps->md->name);

    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps->md->signal |=
        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update

    return RETURN_SUCCESS;
}

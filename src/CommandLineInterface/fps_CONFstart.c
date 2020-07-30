/**
 * @file    fps_CONFstart.c
 * @brief   FPS conf process start
 */



#include "CommandLineInterface/CLIcore.h"




/** @brief FPS start CONF process
 * 
 * Requires setup performed by milk-fpsinit, which performs the following setup
 * - creates the FPS shared memory
 * - create up tmux sessions
 * - create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop
 */ 

errno_t functionparameter_CONFstart(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    // Move to correct launch directory
    //
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \"cd %s\" C-m",
                           fps->md->name, fps->md->fpsdirectory);

    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:conf \"fpsconfstart\" C-m",
                           fps->md->name);

    fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;

    // notify GUI loop to update
    fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

    return RETURN_SUCCESS;
}


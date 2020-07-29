/**
 * @file    fps_RUNstart.c
 * @brief   FPS run process start
 */



#include "CommandLineInterface/CLIcore.h"



/** @brief FPS start RUN process
 * 
 * Requires setup performed by milk-fpsinit, which performs the following setup
 * - creates the FPS shared memory
 * - create up tmux sessions
 * - create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop
 */ 
errno_t functionparameter_RUNstart(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
)
{

    if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK)
    {
        // Move to correct launch directory
        EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"cd %s\" C-m",
                               fps[fpsindex].md->name, fps[fpsindex].md->fpsdirectory);

        EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"fpsrunstart\" C-m",
                               fps[fpsindex].md->name);

        fps[fpsindex].md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
        fps[fpsindex].md->signal |=
            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
    }
    return RETURN_SUCCESS;
}


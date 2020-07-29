/**
 * @file    fps_RUNstop.c
 * @brief   FPS run process stop
 */



#include "CommandLineInterface/CLIcore.h"



/** @brief FPS stop RUN process
 * 
 * Run pre-set function fpsrunstop in tmux ctrl window
 */ 
errno_t functionparameter_RUNstop(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
)
{	
    // Move to correct launch directory
    // 
    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \"cd %s\" C-m",
                           fps[fpsindex].md->name, fps[fpsindex].md->fpsdirectory);

	EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:ctrl \"fpsrunstop\" C-m",
                           fps[fpsindex].md->name);

	// Send C-c in case runstop command is not implemented
	EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run C-c &> /dev/null",
                fps[fpsindex].md->name);

    fps[fpsindex].md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps[fpsindex].md->signal |=
        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update

    return RETURN_SUCCESS;
}

/**
 * @file    fps_CONFstop.c
 * @brief   FPS conf process stop
 */



#include "CommandLineInterface/CLIcore.h"



/** @brief FPS stop CONF process
 * 
 */
errno_t functionparameter_CONFstop(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
)
{
	// send conf stop signal
	fps[fpsindex].md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;

    return RETURN_SUCCESS;
}

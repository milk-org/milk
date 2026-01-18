/**
 * @file    fps_execFPScmd.c
 * @brief   Execute FPS command
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"
#include "fps_tmux.h"

/** @brief Execute FPS command
 *
 * This dispatch function is called by CLI
 * with the proper code to perform FPS-related operation.
 *
 *
 *
 */
errno_t function_parameter_execFPScmd()
{
    long fpsID;

    if(FPS_CMDCODE == FPSCMDCODE_FPSINIT)  // Initialize FPS
    {
        FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(FPS_CMDCODE == FPSCMDCODE_CONFSTART)  // Start CONF process
    {
        FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(FPS_CMDCODE == FPSCMDCODE_CONFSTOP)  // Stop CONF process
    {
        FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(FPS_CMDCODE == FPSCMDCODE_RUNSTART)  // Start RUN process
    {
        FPS_RUNfunc(); // call run function
        return RETURN_SUCCESS;
    }

    if(FPS_CMDCODE == FPSCMDCODE_RUNSTOP)  // Stop RUN process
    {
        FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(FPS_CMDCODE == FPSCMDCODE_TMUXSTART)  // Start tmux session
    {
        // load if not already in memory
        fpsID = function_parameter_structure_load(FPS_name);
        if(fpsID != -1)
        {
            functionparameter_FPS_tmux_init(&fpsarray[fpsID]);
        }
        return RETURN_SUCCESS;
    }

    if(FPS_CMDCODE == FPSCMDCODE_TMUXSTOP)  // Stop tmux session
    {
        // load if not already in memory
        fpsID = function_parameter_structure_load(FPS_name);
        if(fpsID != -1)
        {
            functionparameter_FPS_tmux_kill(&fpsarray[fpsID]);
        }
        return RETURN_SUCCESS;
    }

    return RETURN_SUCCESS;
}

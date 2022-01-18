/**
 * @file    fps_execFPScmd.c
 * @brief   Execute FPS command
 */

#include "CommandLineInterface/CLIcore.h"
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

    if (data.FPS_CMDCODE == FPSCMDCODE_FPSINIT) // Initialize FPS
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_CONFSTART) // Start CONF process
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_CONFSTOP) // Stop CONF process
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_RUNSTART) // Start RUN process
    {
        data.FPS_RUNfunc(); // call run function
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_RUNSTOP) // Stop RUN process
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_TMUXSTART) // Start tmux session
    {
        // load if not already in memory
        fpsID = function_parameter_structure_load(data.FPS_name);
        if (fpsID != -1)
        {
            functionparameter_FPS_tmux_init(&data.fpsarray[fpsID]);
        }
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_TMUXSTOP) // Stop tmux session
    {
        // load if not already in memory
        fpsID = function_parameter_structure_load(data.FPS_name);
        if (fpsID != -1)
        {
            functionparameter_FPS_tmux_kill(&data.fpsarray[fpsID]);
        }
        return RETURN_SUCCESS;
    }

    return RETURN_SUCCESS;
}

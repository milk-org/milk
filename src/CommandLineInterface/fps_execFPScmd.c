/**
 * @file    fps_execFPScmd.c
 * @brief   Execute FPS command
 */

#include "CommandLineInterface/CLIcore.h"



errno_t function_parameter_execFPScmd()
{
#ifndef STANDALONE
    if(data.FPS_CMDCODE == FPSCMDCODE_FPSINIT)   // Initialize FPS
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_CONFSTART)    // Start CONF process
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_CONFSTOP)   // Stop CONF process
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_RUNSTART)   // Start RUN process
    {
        data.FPS_RUNfunc(); // call run function
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_RUNSTOP)   // Stop RUN process
    {
        data.FPS_CONFfunc(); // call conf function
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_TMUXSTART)   // Start tmux session
    {

        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_TMUXSTOP)   // Stop tmux session
    {
        
        return RETURN_SUCCESS;
    }

#endif

    return RETURN_SUCCESS;
}


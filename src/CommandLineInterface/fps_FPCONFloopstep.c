/**
 * @file    fps_FPCONFloopstep.c
 * @brief   FPS conf process loop step
 */

#include "CommandLineInterface/CLIcore.h"

uint16_t function_parameter_FPCONFloopstep(FUNCTION_PARAMETER_STRUCT *fps)
{
    static int loopINIT = 0;
    uint16_t updateFLAG = 0;

    static uint32_t prev_status;
    //static uint32_t statuschanged = 0;

    if (loopINIT == 0)
    {
        loopINIT = 1; // update on first loop iteration
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

        if (fps->CMDmode & FPSCMDCODE_CONFSTART) // parameter configuration loop
        {
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
            fps->md->confpid = getpid();
            fps->localstatus |= FPS_LOCALSTATUS_CONFLOOP;
        }
        else
        {
            fps->localstatus &= ~FPS_LOCALSTATUS_CONFLOOP;
        }
    }

    if (fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN)
    {
        // Test if CONF process is running
        if ((getpgid(fps->md->confpid) >= 0) && (fps->md->confpid > 0))
        {
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CONF; // running
        }
        else
        {
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CONF; // not running
        }

        // Test if RUN process is running
        if ((getpgid(fps->md->runpid) >= 0) && (fps->md->runpid > 0))
        {
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_RUN; // running
        }
        else
        {
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_RUN; // not running
        }

        if (prev_status != fps->md->status)
        {
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // request an update
        }

        if (fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE) // update is required
        {
            updateFLAG = 1;
            fps->md->signal &=
                ~FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // disable update (should be moved to conf process)
        }
        usleep(fps->md->confwaitus);
    }
    else
    {
        fps->localstatus &= ~FPS_LOCALSTATUS_CONFLOOP;
    }

    prev_status = fps->md->status;

    return updateFLAG;
}

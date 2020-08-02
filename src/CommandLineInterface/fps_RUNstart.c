/**
 * @file    fps_RUNstart.c
 * @brief   FPS run process start
 */



#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_tools/timeutils.h"

#include "fps_GetParamIndex.h"


/** @brief FPS start RUN process
 *
 * Requires setup performed by milk-fpsinit, which performs the following setup
 * - creates the FPS shared memory
 * - create up tmux sessions
 * - create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop
 */
errno_t functionparameter_RUNstart(
    FUNCTION_PARAMETER_STRUCT *fps
)
{

    if(fps->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK)
    {
		long pindex;
		
		
        // Move to correct launch directory
        EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"cd %s\" C-m",
                               fps->md->name, fps->md->fpsdirectory);


        // set OMP_NUM_THREADS if applicable
        pindex = functionparameter_GetParamIndex(fps, ".conf.procinfo.NBthread");
        if(pindex > -1) {
            long NBthread = functionparameter_GetParamValue_INT64(fps, ".conf.procinfo.NBthread");
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"export OMP_NUM_THREADS=%ld\" C-m", fps->md->name, NBthread);
        }

        // set timestring if applicable
        pindex = functionparameter_GetParamIndex(fps, ".conf.timestring");
        if(pindex > -1) {
            char timestring[100];
            mkUTtimestring_millisec_now(timestring);
            if(snprintf(fps->parray[pindex].val.string[0],
                        FUNCTION_PARAMETER_STRMAXLEN, "%s", timestring) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
        }


        // Send run command
        EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"fpsrunstart\" C-m",
                               fps->md->name);

        fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
        fps->md->signal |=
            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
    }
    return RETURN_SUCCESS;
}


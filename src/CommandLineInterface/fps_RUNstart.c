/**
 * @file    fps_RUNstart.c
 * @brief   FPS run process start
 */



#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

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
                               fps->md->name, fps->md->workdir);


        // set cset if applicable
        //
        pindex = functionparameter_GetParamIndex(fps, ".procinfo.cset");
        if(pindex > -1) {
            //sprintf(cmdprefix, "taskset --cpu-list %s ", fps->parray[pindex].val.string[0]);
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"export TCSETCMDPREFIX=\\\"csetpmove %s;\\\"\" C-m", fps->md->name, fps->parray[pindex].val.string[0]);
        }
        else
        {
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"export TCSETCMDPREFIX=\"\"\" C-m", fps->md->name);
        }

        // set taskset if applicable
        //
        pindex = functionparameter_GetParamIndex(fps, ".procinfo.taskset");
        if(pindex > -1) {
            //sprintf(cmdprefix, "taskset --cpu-list %s ", fps->parray[pindex].val.string[0]);
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"export TCSETCMDPREFIX=\\\"\\${TCSETCMDPREFIX} tsetpmove \\\\\\\"%s\\\\\\\";\\\"\" C-m", fps->md->name, fps->parray[pindex].val.string[0]);
        }
        else
        {
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"export TCSETCMDPREFIX=\"\"\" C-m", fps->md->name);
        }

        // set OMP_NUM_THREADS if applicable
        //
        pindex = functionparameter_GetParamIndex(fps, ".procinfo.NBthread");
        if(pindex > -1) {
            long NBthread = functionparameter_GetParamValue_INT64(fps, ".conf.NBthread");
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"export OMP_NUM_THREADS=%ld\" C-m", fps->md->name, NBthread);
        }




        // override output directory if applicable
        //
        pindex = functionparameter_GetParamIndex(fps, ".conf.datadir");
        if(pindex > -1) {
            if(snprintf(fps->md->datadir,
                        FUNCTION_PARAMETER_STRMAXLEN, "%s", fps->parray[pindex].val.string[0]) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
        }

        // create output directory if it does not already exit
        EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"mkdir %s\" C-m",
                               fps->md->name, fps->md->datadir);





        // Send run command
        //
        EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"fpsrunstart\" C-m",
                               fps->md->name);

        fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
        fps->md->signal |=
            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
    }
    return RETURN_SUCCESS;
}


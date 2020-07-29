/**
 * @file    fps_RUNexit.c
 * @brief   Exit FPS run process
 */


#include "CommandLineInterface/CLIcore.h"


uint16_t function_parameter_RUNexit(FUNCTION_PARAMETER_STRUCT *fps)
{
    //fps->md->confpid = 0;


    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    function_parameter_struct_disconnect(fps);

    return 0;
}


/**
 * @file    fps_FPCONFexit.c
 * @brief   Exit FPS conf process
 */


#include "CommandLineInterface/CLIcore.h"

#include "fps_disconnect.h"


uint16_t function_parameter_FPCONFexit(FUNCTION_PARAMETER_STRUCT *fps)
{
    //fps->md->confpid = 0;


    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    function_parameter_struct_disconnect(fps);

    return 0;
}


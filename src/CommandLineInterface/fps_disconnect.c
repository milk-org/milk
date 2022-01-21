/**
 * @file    fps_disconnect.c
 * @brief   Disconnect from FPS
 */

#include <sys/mman.h> // munmap

#include "CommandLineInterface/CLIcore.h"

int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *funcparamstruct)
{
    int NBparamMAX;

    NBparamMAX = funcparamstruct->md->NBparamMAX;
    //funcparamstruct->md->NBparam = 0;
    funcparamstruct->parray = NULL;

    munmap(funcparamstruct->md,
           sizeof(FUNCTION_PARAMETER_STRUCT_MD) +
               sizeof(FUNCTION_PARAMETER) * NBparamMAX);

    close(funcparamstruct->SMfd);

    funcparamstruct->SMfd = -1;

    return RETURN_SUCCESS;
}

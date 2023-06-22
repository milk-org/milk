/**
 * @file    fps_disconnect.c
 * @brief   Disconnect from FPS
 */

#include <sys/mman.h> // munmap
#include <sys/stat.h> // fstat

#include "CommandLineInterface/CLIcore.h"

int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *funcparamstruct)
{
    //int NBparamMAX;

    //NBparamMAX = funcparamstruct->md->NBparamMAX;
    //funcparamstruct->md->NBparam = 0;
    funcparamstruct->parray = NULL;

    // get file size
    //
    struct stat file_stat;
    fstat(funcparamstruct->SMfd, &file_stat);

    munmap(funcparamstruct->md, file_stat.st_size);
    // note: file size should be equal to :
    // sizeof(FUNCTION_PARAMETER_STRUCT_MD) +
    // sizeof(FUNCTION_PARAMETER) * NBparamMAX)

    close(funcparamstruct->SMfd);

    funcparamstruct->SMfd = -1;

    return RETURN_SUCCESS;
}

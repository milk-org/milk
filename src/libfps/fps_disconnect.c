/**
 * @file    fps_disconnect.c
 * @brief   Disconnect from FPS
 */

#include <sys/mman.h> // munmap
#include <sys/stat.h> // fstat

#include "fps.h"
#include "fps_internal.h"

int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *funcparamstruct)
{
    if (funcparamstruct == NULL) {
        return RETURN_SUCCESS;
    }

    funcparamstruct->parray = NULL;

    if (funcparamstruct->SMfd > -1) {
        if (funcparamstruct->md != NULL && funcparamstruct->md != MAP_FAILED) {
            struct stat file_stat;
            if (fstat(funcparamstruct->SMfd, &file_stat) == 0) {
                munmap(funcparamstruct->md, file_stat.st_size);
            }
            funcparamstruct->md = NULL;
        }
        close(funcparamstruct->SMfd);
        funcparamstruct->SMfd = -1;
    }

    return RETURN_SUCCESS;
}

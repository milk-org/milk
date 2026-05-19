/**
 * @file    fps_SetParamCLIindex.c
 * @brief   set parameter CLI index
 */

#include "fps.h"

/**
 * @brief Set the CLI index for a parameter in an FPS.
 *
 * @param fps Pointer to the Function Parameter Structure.
 * @param pindex Index of the parameter.
 * @param cli_index The CLI argument index (1-indexed usually).
 * @return errno_t RETURN_SUCCESS on success, error code otherwise.
 */
int functionparameter_SetParamCLIindex(
    FPS  *fps,
    long pindex,
    int  cli_index)
{
    FPS_PARAM *funcparamarray;

    funcparamarray = fps->parray;

    if(pindex < 0 || pindex >= fps->md->NBparamMAX)
    {
        return RETURN_FAILURE;
    }

    funcparamarray[pindex].cli_index = cli_index;

    return RETURN_SUCCESS;
}

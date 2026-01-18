/**
 * @file    fps_CONFstop.c
 * @brief   FPS conf process stop
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"

/** @brief FPS stop CONF process
 *
 */
errno_t functionparameter_CONFstop(FUNCTION_PARAMETER_STRUCT *fps)
{
    // send conf stop signal
    fps->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;

    return RETURN_SUCCESS;
}

/**
 * @file    fps_tmux.h
 * @brief  tmux session management

 */

#ifndef FPS_TMUX_H
#define FPS_TMUX_H

#include "function_parameters.h"

errno_t functionparameter_FPS_tmux_kill(
    FUNCTION_PARAMETER_STRUCT *fps
);


errno_t functionparameter_FPS_tmux_init(
    FUNCTION_PARAMETER_STRUCT *fps
);

#endif

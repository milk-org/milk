/**
 * @file    fps_tmux.h
 * @brief  tmux session management

 */

#ifndef FPS_TMUX_H
#define FPS_TMUX_H

#include "fps.h"

errno_t functionparameter_FPS_tmux_kill(FUNCTION_PARAMETER_STRUCT *fps);

errno_t functionparameter_FPS_tmux_attach(FUNCTION_PARAMETER_STRUCT *fps);

errno_t functionparameter_FPS_tmux_init(FUNCTION_PARAMETER_STRUCT *fps);

errno_t functionparameter_FPS_tmux_ensure(FUNCTION_PARAMETER_STRUCT *fps);

errno_t functionparameter_FPS_tmux_standalone_setup(const char *fps_name);

errno_t functionparameter_FPS_tmux_send(const char *fps_name, const char *window, const char *cmd_str);

errno_t functionparameter_FPS_tmux_send_dispatch(const char *fps_name, const char *command, const char *exec_path, const char *extra_args);

char* functionparameter_FPS_get_executable_path(char *buffer, size_t size);

#endif

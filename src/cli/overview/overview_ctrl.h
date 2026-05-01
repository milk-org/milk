/**
 * @file overview_ctrl.h
 * @brief Control-mode actions for milkCTRL
 */

#ifndef OVERVIEW_CTRL_H
#define OVERVIEW_CTRL_H

#include "overview_data.h"

/**
 * ov_ctrl_fps_run_toggle - start or stop the FPS run process.
 */
void ov_ctrl_fps_run_toggle(const OV_FPS *f);

/**
 * ov_ctrl_fps_conf_toggle - start or stop the FPS conf process.
 */
void ov_ctrl_fps_conf_toggle(const OV_FPS *f);

/**
 * ov_ctrl_stream_delete - destroy a shared memory stream.
 */
void ov_ctrl_stream_delete(const OV_STREAM *s);

/**
 * ov_ctrl_proc_kill - send SIGTERM to a process.
 */
void ov_ctrl_proc_kill(const OV_PROC *p);

#endif /* OVERVIEW_CTRL_H */

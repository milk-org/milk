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
 * @p: process model entry (read-only snapshot from OV_MODEL)
 */
void ov_ctrl_proc_kill(const OV_PROC *p);

/**
 * ov_ctrl_proc_sigkill - send SIGKILL to a process.
 */
void ov_ctrl_proc_sigkill(const OV_PROC *p);

/**
 * ov_ctrl_proc_pause_toggle - toggle SIGSTOP/SIGCONT for a process.
 */
void ov_ctrl_proc_pause_toggle(const OV_PROC *p);

/**
 * ov_ctrl_fps_signal_pid - helper to send signal to FPS PIDs
 */
void ov_ctrl_fps_signal_pid(const OV_FPS *f, int sig);

/**
 * ov_ctrl_fps_pause_toggle - toggle SIGSTOP/SIGCONT for FPS processes.
 */
void ov_ctrl_fps_pause_toggle(const OV_FPS *f);

/**
 * ov_ctrl_fps_remove - stop conf/run and remove the FPS SHM entry.
 */
void ov_ctrl_fps_remove(const OV_FPS *f);

#endif /* OVERVIEW_CTRL_H */

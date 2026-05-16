/**
 * @file overview_ctrl.h
 * @brief Control-mode actions for milk-CTRL
 */

#ifndef OVERVIEW_CTRL_H
#define OVERVIEW_CTRL_H

#include "overview_data.h"
#include "overview_layout.h"

/**
 * ov_ctrl_fps_run_toggle - start or stop the FPS run process.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_run_toggle(
    const OV_FPS *f,
    OV_CMDLOG    *log);

/**
 * ov_ctrl_fps_conf_toggle - start or stop the FPS conf process.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_conf_toggle(
    const OV_FPS *f,
    OV_CMDLOG    *log);

/**
 * ov_ctrl_stream_delete - destroy a shared memory stream.
 * @s:   stream model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_stream_delete(
    const OV_STREAM *s,
    OV_CMDLOG       *log);

/**
 * ov_ctrl_proc_kill - send SIGTERM to a process.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_kill(
    const OV_PROC *p,
    OV_CMDLOG     *log);

/**
 * ov_ctrl_proc_sigkill - send SIGKILL to a process.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_sigkill(
    const OV_PROC *p,
    OV_CMDLOG     *log);

/**
 * ov_ctrl_proc_remove - remove a single process from shm.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_remove(
    const OV_PROC *p,
    OV_CMDLOG     *log);

/**
 * ov_ctrl_proc_pause_toggle - toggle SIGSTOP/SIGCONT.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_pause_toggle(
    const OV_PROC *p,
    OV_CMDLOG     *log);

/**
 * ov_ctrl_proc_set_ctrlval - mutate process CTRLval.
 * @p:   process model entry
 * @val: new value (-1 to toggle between 0 and 1)
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_set_ctrlval(
    const OV_PROC *p,
    int            val,
    OV_CMDLOG     *log);

/**
 * ov_ctrl_proc_zero_counters - reset process loopcnt.
 * @p:   process model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_proc_zero_counters(
    const OV_PROC *p,
    OV_CMDLOG     *log);

/**
 * ov_ctrl_fps_signal_pid - send signal to FPS PIDs.
 * @f:   FPS model entry
 * @sig: signal number
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_signal_pid(
    const OV_FPS *f,
    int           sig,
    OV_CMDLOG    *log);

/**
 * ov_ctrl_fps_pause_toggle - toggle SIGSTOP/SIGCONT
 * for FPS processes.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_pause_toggle(
    const OV_FPS *f,
    OV_CMDLOG    *log);

/**
 * ov_ctrl_fps_remove - stop conf/run and remove the
 * FPS SHM entry.
 * @f:   FPS model entry
 * @log: command log (may be NULL)
 */
void ov_ctrl_fps_remove(
    const OV_FPS *f,
    OV_CMDLOG    *log);

/**
 * ov_ctrl_procs_cleanup - remove crashed/stopped processes
 * @log: command log (may be NULL)
 */
void ov_ctrl_procs_cleanup(
    OV_CMDLOG *log);

/**
 * ov_ctrl_inspect_item - spawn an interactive detailed view
 * @panel: the active panel type
 * @item:  pointer to the selected item (OV_STREAM, OV_PROC, or OV_FPS)
 */
void ov_ctrl_inspect_item(
    ov_focus_t     panel,
    const void    *item);

#endif /* OVERVIEW_CTRL_H */

/**
 * @file    fps_apply_process_settings.h
 * @brief   Apply FPS process settings to the current process
 *
 * Reads taskset, cset, and NBthread from an FPS and applies
 * them programmatically. Called by fps_generic_run() before
 * invoking the compute function.
 */

#ifndef FPS_APPLY_PROCESS_SETTINGS_H
#define FPS_APPLY_PROCESS_SETTINGS_H

#include "fps.h"

/**
 * @brief Apply process-level settings from FPS parameters.
 *
 * Reads .procinfo.NBthread, .procinfo.taskset, and
 * .procinfo.cset from the FPS and applies them to the
 * current process:
 *   - OMP_NUM_THREADS via setenv()
 *   - CPU affinity via sched_setaffinity()
 *   - cgroup migration via milk-makecsetandrt
 *
 * @param fps  Connected FPS to read settings from
 * @return RETURN_SUCCESS or RETURN_FAILURE
 */
errno_t fps_apply_process_settings(FPS *fps);

#endif /* FPS_APPLY_PROCESS_SETTINGS_H */

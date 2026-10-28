// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_unit.h
 * @brief   Unit lifecycle: start a detached worker and wait for its loop, stop it, environment snapshot
 *
 * Front-ends that must not block run the milk-fps-<op> tools through fps_unit_spawn() or the
 * fps_unit_job_* helpers, and parameter scripts through fps_unit_script().
 *
 * A unit is an fpsexec binary and its FPS. fps_unit_start() runs `<fpsexec> NAME:<role>start --fg`
 * detached (milk_detach.h) and returns once the loop is active. The worker side (fps_unit_role_*,
 * fps_unit_notify_*) runs in fps_generic_run() and fps_generic_conf_cb(), for foreground runs too,
 * and tells a service manager what it tells a starter (milk_notify.h). A role that systemd runs is
 * started and stopped through systemctl (U17). State and the unit directory: fps_unit_state.h.
 */

#ifndef FPS_UNIT_H
#define FPS_UNIT_H

#include <sys/types.h>
#include <time.h>

#include "fps_unit_state.h"
#include "milk_compiler.h"

/* fps_unit_start() result when the loop was not active in time (the worker keeps starting) */
#define FPS_UNIT_START_TIMEOUT 124

/* fps_unit_start() and fps_unit_stop() flags */
#define FPS_UNIT_QUIET 0x1 /* print nothing: the caller reports through its own channel (a TUI) */

/**
 * fps_unit_start - start @role of unit @name in the background, return once its loop is active.
 * @timeout_s: readiness timeout; <= 0 for MILK_UNIT_START_TIMEOUT or 10 s.
 * Return: 0 ready or already running, 1 failed (the worker's messages are printed),
 * FPS_UNIT_START_TIMEOUT not ready in time.
 */
int fps_unit_start(const char *name, FPS_UNIT_ROLE role, double timeout_s, int flags);

/** fps_unit_stop - stop @role of unit @name: request, then signals (U6). Return: 0 stopped or not running, 1. */
int fps_unit_stop(const char *name, FPS_UNIT_ROLE role, int flags);

/**
 * fps_unit_exec_service - for a service manager: check unit @name (U2, owner, not running), then exec
 * its fpsexec as `NAME:<role>start --fg` in its work directory. Return: only on failure, 1.
 */
int fps_unit_exec_service(const char *name, FPS_UNIT_ROLE role);

/** fps_unit_release - forget the systemd units that run unit @name: milk starts and stops it again. Return: 0 or 1. */
int fps_unit_release(const char *name);

/** fps_unit_systemd_service - the systemd service running this process ($INVOCATION_ID, its cgroup). Return: 1 or 0. */
int fps_unit_systemd_service(char *unit, size_t size, int *user);

/**
 * fps_unit_spawn - start `milk-fps-<op> NAME` (runstart, runstop, confstart, confstop, confstep, or rm:
 * stop, then remove the FPS) without waiting, for callers that must not block (U3).
 * @envp: its environment, NULL for environ; @out_fd, @err_fd: its stdout and stderr, -1 for /dev/null.
 * Return: its pid to reap, or -1 with errno set.
 */
pid_t fps_unit_spawn(const char *name, const char *op, char *const envp[], int out_fd, int err_fd);

/* A fps_unit_spawn() tool run by a front-end: fps_unit_job_start(), then fps_unit_job_poll() from its
 * event loop until the tool ends */
typedef struct
{
    pid_t  pid; /* 0: none running */
    int    fd;  /* its output, non-blocking; -1 once closed */
    double t0;  /* start, CLOCK_MONOTONIC s */
    char   name[STRINGMAXLEN_FPS_NAME];
    char   op[16];
    char   msg[256]; /* the first line it printed, terminal sequences removed */
    size_t len;
    int    state;
} FPS_UNIT_JOB;

/** fps_unit_job_start - fps_unit_spawn() @op on @name, its output on a pipe. Return: 0, or -1 (errno). */
int fps_unit_job_start(FPS_UNIT_JOB *job, const char *name, const char *op);

/** fps_unit_job_poll - collect the job's output; reap it once it ended. Return: its exit status, -1 while it runs. */
int fps_unit_job_poll(FPS_UNIT_JOB *job);

/** fps_unit_job_cancel - stop following @job: the tool finishes on its own. */
void fps_unit_job_cancel(FPS_UNIT_JOB *job);

/**
 * fps_unit_script - run @path (an EXECFILENAME parameter) detached in unit @name's work directory, with
 * the FPS data file as argument, output to exec.out (U11). Return: 0 once execve() succeeded, or 1.
 */
int fps_unit_script(const char *name, const char *path, int flags);

/**
 * fps_unit_procinfo_leftovers - processinfo files of dead processes of unit @name: with @keep, the last
 * run's goes to run.procinfo.1 (U7); the others are removed, and their registry slots freed.
 */
void fps_unit_procinfo_leftovers(const char *name, int keep);

/** fps_unit_env_capture - write the calling process's environment to the unit's env file. Return: 0 or -1. */
int fps_unit_env_capture(const char *name);

/** fps_unit_role_begin - a process starts running @role of @name: record it (U7). Return: 0, 1 if already running. */
int fps_unit_role_begin(const char *name, FPS_UNIT_ROLE role);

/** fps_unit_role_end - the process's @role ends with exit status @code: record it. */
void fps_unit_role_end(int code);

/** fps_unit_notify_connected - worker: FPS connected and parameters valid. */
void fps_unit_notify_connected(void);

/** fps_unit_notify_failed - worker: cannot start, for reason @err and message @fmt. */
void fps_unit_notify_failed(int err, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/** fps_unit_notify_ready - worker: the loop is active; the starter returns. No-op elsewhere. */
void fps_unit_notify_ready(void);

/* A unit process's tick, at most once a second (fps_unit_tick()): its output size, and a service
 * manager's STATUS= and WATCHDOG=1 */
extern time_t fps_unit_tick_deadline;
void          fps_unit_tick_run(long loopcnt);

/** fps_unit_tick - fps_unit_tick_run() once a second in a unit's process, with the loop count (-1: none); a compare elsewhere. */
static inline void fps_unit_tick(long loopcnt)
{
    if (UNLIKELY(fps_unit_tick_deadline != 0) && time(NULL) >= fps_unit_tick_deadline)
    {
        fps_unit_tick_run(loopcnt);
    }
}

#endif
